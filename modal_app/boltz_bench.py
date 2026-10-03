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


def _child_peak_rss_gb() -> float:
    """Peak RSS of the boltz subprocess, in GB.

    ``RUSAGE_CHILDREN`` reports the high-water mark across all reaped children,
    so this is monotonic across folds in one container -- which is what we want
    for sizing a memory request.
    """
    try:
        import resource

        kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        return round(kb / 1024 / 1024, 2)  # Linux reports kibibytes
    except Exception:
        return -1.0


def _cgroup_peak_mem_gb() -> float:
    """Whole-container peak memory from the cgroup, in GB.

    This is the figure Modal's memory request has to cover, since it includes
    the parent process and page cache, not just the boltz child.
    """
    for path in (
        "/sys/fs/cgroup/memory.peak",
        "/sys/fs/cgroup/memory/memory.max_usage_in_bytes",
    ):
        try:
            with open(path) as fh:
                return round(int(fh.read().strip()) / 1e9, 2)
        except Exception:
            continue
    return -1.0


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


class _CompletionProbe:
    """Records when each complex's mmCIF first appears, on a thread.

    One ``boltz predict`` call over a directory loads the weights once, which
    is what the plan requires -- but it also collapses the whole batch into a
    single wall time. Watching the output tree for each structure to land
    recovers per-complex timings from that single process, so the steady-state
    figure is a fold with the model already resident.
    """

    def __init__(self, root: Path, interval: float = 0.25) -> None:
        self.root = root
        self.interval = interval
        self.seen: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                for cif in self.root.rglob("*_model_0.cif"):
                    # boltz names outputs after the input stem, which is the
                    # complex_id because that is what the YAML is called.
                    key = cif.name.split("_model_0")[0]
                    self.seen.setdefault(key, time.monotonic())
            except Exception:
                pass  # a probe failure must never fail the batch
            self._stop.wait(self.interval)

    def __enter__(self) -> _CompletionProbe:
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        try:  # catch anything written between the last poll and exit
            for cif in self.root.rglob("*_model_0.cif"):
                self.seen.setdefault(cif.name.split("_model_0")[0], time.monotonic())
        except Exception:
            pass


@app.function(
    image=boltz_image,
    volumes={CACHE_DIR.parent: weights_vol, MSA_DIR: msa_vol, OUT_DIR: out_vol},
    gpu="L40S",  # overridden per call via .with_options(gpu=...)
    cpu=WORKER_CPU,
    memory=WORKER_MEM,
    timeout=120 * MINUTES,
    scaledown_window=10,  # the 60 s default would bill a GPU tail per container
    max_containers=1,
    retries=0,
)
def fold_batch(
    complexes: list[dict],
    gpu_label: str,
    single_sequence: bool = False,
    save_structures: bool = True,
) -> dict:
    """Fold a batch in ONE ``boltz predict`` process, timing each complex.

    This is the plan's "keep models loaded across complexes to avoid reload
    overhead". An earlier version of this function spawned a fresh
    ``boltz predict`` per complex, which put a full Python start, torch import,
    6.2 GB weight load and CUDA init *inside every timed fold* -- and reported
    ``startup_s = 0.0`` because, by its own definition, nothing happened before
    the first fold. Every steady-state number it produced was a per-process
    overhead measurement wearing a per-fold label. Weights now load once per
    batch, and ``load_plus_first_s`` carries that cost explicitly so the cost
    model can amortise it over the production batch size instead of charging it
    to all 2,000 complexes.
    """
    container_t0 = time.monotonic()
    work = Path("/tmp/work/inputs")
    work.mkdir(parents=True, exist_ok=True)
    case_out = Path("/tmp/work/out")
    case_out.mkdir(parents=True, exist_ok=True)
    run_out = OUT_DIR / gpu_label.replace("!", "") / (
        "single_seq" if single_sequence else "msa"
    )
    run_out.mkdir(parents=True, exist_ok=True)

    mode = "single_seq" if single_sequence else "msa"
    ordered: list[dict] = []
    missing: list[dict] = []

    for cx in complexes:
        msa_ref = "empty" if single_sequence else str(MSA_DIR / f"{cx['msa_id']}.a3m")
        if not single_sequence and not Path(msa_ref).exists():
            missing.append(
                {
                    "complex_id": cx["complex_id"],
                    "allele": cx["allele"],
                    "peptide": cx["peptide"],
                    "gpu": gpu_label,
                    "msa_mode": mode,
                    "ok": False,
                    "error": f"missing MSA {msa_ref} -- run setup first",
                }
            )
            continue
        # One YAML per complex in a single directory: boltz predict accepts the
        # directory and processes every entry in one process.
        (work / f"{cx['complex_id']}.yaml").write_text(
            build_yaml(cx["peptide"], cx["hla_seq"], msa_ref)
        )
        ordered.append(cx)

    cmd = [
        "boltz", "predict", str(work),
        "--out_dir", str(case_out),
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

    run_t0 = time.monotonic()
    with _GpuMemoryProbe() as probe, _CompletionProbe(case_out) as done:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    total_s = time.monotonic() - run_t0

    # Order complexes by when their structure landed, then difference the
    # timestamps. The first interval carries the weight load; the rest are
    # folds with the model resident.
    finished = sorted(done.seen.items(), key=lambda kv: kv[1])
    results: list[dict] = []
    prev = run_t0
    load_plus_first = None

    for rank, (cid, at) in enumerate(finished):
        cx = next((c for c in ordered if c["complex_id"] == cid), None)
        if cx is None:
            continue
        delta = at - prev
        prev = at
        if rank == 0:
            load_plus_first = round(delta, 2)
        rec = {
            "complex_id": cid,
            "allele": cx["allele"],
            "peptide": cx["peptide"],
            "gpu": gpu_label,
            "msa_mode": mode,
            "is_warmup": rank == 0,
            "ok": True,
            "fold_s": round(delta, 2),
            "peak_mem_gb": probe.peak_gb,
            "host_peak_rss_gb": _child_peak_rss_gb(),
            "container_peak_mem_gb": _cgroup_peak_mem_gb(),
        }
        rec.update(_collect_outputs(case_out, run_out, cx, save_structures))
        results.append(rec)

    # Anything that never produced a structure failed, whatever the exit code.
    for cx in ordered:
        if not any(r["complex_id"] == cx["complex_id"] for r in results):
            results.append(
                {
                    "complex_id": cx["complex_id"],
                    "allele": cx["allele"],
                    "peptide": cx["peptide"],
                    "gpu": gpu_label,
                    "msa_mode": mode,
                    "ok": False,
                    "error": (proc.stderr or proc.stdout or "no structure written")[-2000:],
                }
            )
    results.extend(missing)

    if save_structures:
        out_vol.commit()

    return {
        "gpu": gpu_label,
        "msa_mode": mode,
        # The real container startup: mount and setup before boltz was invoked.
        "startup_s": round(run_t0 - container_t0, 2),
        # Weight load + CUDA init + the first fold, paid once per container.
        "load_plus_first_s": load_plus_first,
        "batch_total_s": round(total_s, 2),
        "container_s": round(time.monotonic() - container_t0, 2),
        "n": len(complexes),
        # boltz's own stdout carries the per-stage breakdown and was previously
        # discarded on success, which made the overhead unrecoverable.
        "stdout_tail": (proc.stdout or "")[-3000:],
        "returncode": proc.returncode,
        "results": results,
    }


def _collect_outputs(
    case: Path, run_out: Path, cx: dict, save: bool
) -> dict:
    """Harvest the structure, the confidence JSON, and the PAE matrix.

    ``case`` is the batch-wide output tree, so every glob is filtered to this
    complex: a bare rglob would return all eight complexes' files and attach
    the first one's scores to every row.
    """
    info: dict = {}
    cid = cx["complex_id"]

    def mine(pattern: str) -> list[Path]:
        return sorted(p for p in case.rglob(pattern) if cid in p.as_posix())

    cifs = mine(f"{cid}_model_0.cif") or mine("*_model_0.cif") or mine("*.cif")
    confs = mine("confidence_*.json")
    paes = mine("pae_*.npz") + mine("pae_*.npy")

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
        print(
            f"\n{b['msa_mode']}: {ok}/{b['n']} ok  "
            f"container setup {b['startup_s']} s  "
            f"weight load + first fold {b.get('load_plus_first_s')} s  "
            f"batch total {b.get('batch_total_s')} s"
        )
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
