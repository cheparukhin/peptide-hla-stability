"""Boltz-2 GPU folding: the stage 4 pilot and the hardware benchmark.

    modal run modal_app/boltz_bench.py::pilot        # 5 complexes, MSA vs single-sequence
    modal run modal_app/boltz_bench.py::benchmark    # 24 complexes x 4 GPUs

Run `boltz_setup.py::setup` first: it caches the weights and the HLA MSAs on
Volumes that this app mounts read-only. Shared configuration lives in
`boltz_common.py`; see that module for why the CPU setup is a separate app.

CLAUDE.md gates this: no batch GPU job without a passing end-to-end pilot on
3-5 examples. `::pilot` is that gate, and its poses must be checked with
`scripts/boltz_pose_check.py` before `::benchmark` runs.

Three cost decisions encoded below, each one a trap documented in
docs/BOLTZ_PIPELINE.md:

* **`gpu="H100!"`, with the bang.** Plain `"H100"` lets Modal substitute an
  H200, silently making the H100 column an H200 measurement at H100 prices.
  Modal's GPU docs call this out for benchmarking specifically.
* **A batch runs in one container** so the ~6.2 GB of weights load once, and
  the first fold is reported as warm-up rather than averaged into the
  steady-state mean.
* **`scaledown_window` is explicit.** Modal bills container load time and keeps
  containers alive 60 s after the last input by default; at roughly a minute
  per fold that tail is a large share of the bill.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import modal

from boltz_common import (
    BENCH_GPUS,
    CACHE_DIR,
    MINUTES,
    MSA_DIR,
    OUT_DIR,
    REPO,
    WORKER_CPU,
    WORKER_MEM,
    boltz_image,
    build_yaml,
    msa_vol,
    out_vol,
    read_panel,
    weights_vol,
    write_results,
)

app = modal.App("pepstab-boltz")


class _GpuMemoryProbe:
    """Polls ``nvidia-smi`` on a thread; boltz runs as a subprocess so in-process
    torch counters would see nothing."""

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.peak_mib = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                for line in out.stdout.strip().splitlines():
                    self.peak_mib = max(self.peak_mib, float(line.strip()))
            except Exception:
                pass  # a probe failure must never fail the fold
            self._stop.wait(self.interval)

    def __enter__(self) -> _GpuMemoryProbe:
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def peak_gb(self) -> float:
        return round(self.peak_mib / 1024, 2)


@app.function(
    image=boltz_image,
    volumes={CACHE_DIR.parent: weights_vol, MSA_DIR: msa_vol, OUT_DIR: out_vol},
    gpu="L40S",  # overridden per call via .with_options(gpu=...)
    cpu=WORKER_CPU,
    memory=WORKER_MEM,
    timeout=120 * MINUTES,
    scaledown_window=10,  # the 60 s default would bill a GPU tail per container
)
def fold_batch(
    complexes: list[dict],
    gpu_label: str,
    single_sequence: bool = False,
    save_structures: bool = True,
) -> dict:
    """Fold a batch of complexes in one container, timing each separately.

    Returns per-complex timings plus a ``startup_s`` for the container. The
    first complex is the warm-up: it absorbs weight load and CUDA kernel
    compilation, and is flagged so it can be excluded from the steady-state
    mean instead of inflating a 2,000-complex projection.
    """
    container_t0 = time.monotonic()
    work = Path("/tmp/work")
    work.mkdir(parents=True, exist_ok=True)
    run_out = OUT_DIR / gpu_label.replace("!", "") / (
        "single_seq" if single_sequence else "msa"
    )
    run_out.mkdir(parents=True, exist_ok=True)

    results = []
    startup_s = None

    for i, cx in enumerate(complexes):
        msa_ref = "empty" if single_sequence else str(MSA_DIR / f"{cx['msa_id']}.a3m")
        if not single_sequence and not Path(msa_ref).exists():
            results.append(
                {
                    "complex_id": cx["complex_id"],
                    "ok": False,
                    "error": f"missing MSA {msa_ref} -- run setup first",
                    "is_warmup": i == 0,
                }
            )
            continue

        case = work / cx["complex_id"]
        case.mkdir(parents=True, exist_ok=True)
        yaml_path = case / "input.yaml"
        yaml_path.write_text(build_yaml(cx["peptide"], cx["hla_seq"], msa_ref))

        cmd = [
            "boltz", "predict", str(yaml_path),
            "--out_dir", str(case),
            "--cache", str(CACHE_DIR),
            "--accelerator", "gpu",
            "--devices", "1",
            # Plan: one pose, standard sampling, full PAE for stage 5 features.
            "--diffusion_samples", "1",
            "--recycling_steps", "3",
            "--sampling_steps", "200",
            "--output_format", "mmcif",
            "--write_full_pae",
            "--override",
        ]

        t0 = time.monotonic()
        with _GpuMemoryProbe() as probe:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        fold_s = time.monotonic() - t0

        if startup_s is None:
            # Everything before the first fold began: image already pulled by
            # Modal, but volume mount and process import land here.
            startup_s = round(t0 - container_t0, 2)

        rec = {
            "complex_id": cx["complex_id"],
            "allele": cx["allele"],
            "peptide": cx["peptide"],
            "gpu": gpu_label,
            "msa_mode": "single_seq" if single_sequence else "msa",
            "is_warmup": i == 0,
            "ok": proc.returncode == 0,
            "fold_s": round(fold_s, 2),
            "peak_mem_gb": probe.peak_gb,
        }
        if proc.returncode != 0:
            rec["error"] = (proc.stderr or proc.stdout)[-2000:]
        else:
            rec.update(_collect_outputs(case, run_out, cx, save_structures))
        results.append(rec)

    if save_structures:
        out_vol.commit()

    return {
        "gpu": gpu_label,
        "msa_mode": "single_seq" if single_sequence else "msa",
        "startup_s": startup_s,
        "container_s": round(time.monotonic() - container_t0, 2),
        "n": len(complexes),
        "results": results,
    }


def _collect_outputs(
    case: Path, run_out: Path, cx: dict, save: bool
) -> dict:
    """Harvest the structure, the confidence JSON, and the PAE matrix."""
    info: dict = {}
    cifs = sorted(case.rglob("*_model_0.cif")) or sorted(case.rglob("*.cif"))
    confs = sorted(case.rglob("confidence_*.json"))
    paes = sorted(case.rglob("pae_*.npz")) + sorted(case.rglob("pae_*.npy"))

    info["has_structure"] = bool(cifs)
    info["has_pae"] = bool(paes)

    if confs:
        try:
            conf = json.loads(confs[0].read_text())
            for key in ("confidence_score", "ptm", "iptm", "complex_plddt"):
                if key in conf:
                    info[key] = conf[key]
        except Exception as exc:
            info["confidence_parse_error"] = str(exc)

    if save:
        dest = run_out / cx["complex_id"]
        dest.mkdir(parents=True, exist_ok=True)
        for src in cifs[:1] + confs[:1] + paes[:1]:
            dest.joinpath(src.name).write_bytes(src.read_bytes())
    return info


# --------------------------------------------------------------------------
# Local entrypoints
# --------------------------------------------------------------------------



@app.local_entrypoint()
def pilot(gpu: str = "L40S", limit: int = 3, both_arms: bool = False):
    """Fold the pilot complexes end to end. The gate on any batch job.

    Defaults are chosen to spend as little as possible while still clearing the
    gate: ``limit=3`` is the floor CLAUDE.md allows ("3-5 examples"), the
    cheapest candidate GPU, and the MSA arm only. The single-sequence arm is
    informative but answers a question the GPU decision does not need, so it is
    opt-in via ``--both-arms``.

    This run is what makes the real benchmark cheap: its peak-memory figure
    decides whether the 80 GB cards can be dropped from the sweep.
    """
    complexes = read_panel("boltz_pilot.csv")[:limit]
    print(f"pilot: {len(complexes)} complexes on {gpu}, arms={'2' if both_arms else '1'}")
    fn = fold_batch.with_options(gpu=gpu)

    batches = [fn.remote(complexes, gpu_label=gpu, single_sequence=False)]
    if both_arms:
        batches.append(fn.remote(complexes, gpu_label=gpu, single_sequence=True))

    for b in batches:
        ok = sum(1 for r in b["results"] if r["ok"])
        print(f"\n{b['msa_mode']}: {ok}/{b['n']} ok, startup {b['startup_s']} s")
        for r in b["results"]:
            tag = " (warmup)" if r["is_warmup"] else ""
            if r["ok"]:
                print(
                    f"  {r['complex_id']:22} {r['fold_s']:7.1f} s  "
                    f"{r['peak_mem_gb']:5.1f} GB  iptm={r.get('iptm', 'n/a')}"
                    f"  pae={r.get('has_pae')}{tag}"
                )
            else:
                print(f"  {r['complex_id']:22} FAILED{tag}: {r.get('error', '')[:300]}")

    peaks = [r["peak_mem_gb"] for b in batches for r in b["results"] if r["ok"]]
    if peaks:
        print(f"\npeak GPU memory across pilot folds: {max(peaks)} GB")
        print(
            "  If that fits 40 GB with headroom, A100-80GB is strictly dominated "
            "by A100-40GB (same silicon, higher rate) and can leave the sweep."
        )

    write_results(REPO / "reports" / "boltz_pilot_results.csv", batches)
    print(
        "\nBefore running ::benchmark, check the poses:\n"
        "  python scripts/boltz_pose_check.py --structures <dir>"
    )


@app.local_entrypoint()
def benchmark(gpus: str = ",".join(BENCH_GPUS), limit: int = 8):
    """Fold the benchmark panel on each GPU with identical settings.

    ``limit`` trades statistical quality for credits. Every complex is 191
    residues, so runtime spread should be small and a reduced panel still
    ranks hardware reliably -- but it weakens the *failure-rate* estimate,
    which is the other thing the plan's 20-30 figure buys. Raise it for the
    real run.
    """
    complexes = read_panel("boltz_bench_panel.csv")[:limit]
    labels = [g.strip() for g in gpus.split(",") if g.strip()]
    print(f"benchmark: {len(complexes)} complexes x {len(labels)} GPUs: {labels}")

    handles = [
        (g, fold_batch.with_options(gpu=g).spawn(complexes, gpu_label=g))
        for g in labels
    ]
    batches = []
    for g, h in handles:
        try:
            batches.append(h.get())
            print(f"  {g}: done")
        except Exception as exc:
            print(f"  {g}: batch failed -- {exc}")

    write_results(REPO / "reports" / "boltz_bench_results.csv", batches)
    print("\nNow run: python scripts/gpu_decision.py")
