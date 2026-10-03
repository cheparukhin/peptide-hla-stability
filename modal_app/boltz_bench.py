"""Boltz-2 on Modal: precompute HLA MSAs on CPU, then benchmark folding across GPUs.

Stage 4 of HACKATHON_PLAN.md. Three entrypoints, run in order:

    modal run modal_app/boltz_bench.py::setup       # weights + 8 HLA MSAs, CPU only
    modal run modal_app/boltz_bench.py::pilot       # 5 complexes, MSA vs single-sequence
    modal run modal_app/boltz_bench.py::benchmark   # 24 complexes x 4 GPUs

`setup` and `pilot` must both pass before `benchmark`: CLAUDE.md makes the
end-to-end pilot a hard gate on any batch GPU job.

Four design points, each one a cost or correctness trap documented in
docs/BOLTZ_PIPELINE.md:

1. **MSAs are built in a CPU-only function.** `boltz predict --use_msa_server`
   generates alignments *synchronously inside the predict process, before any
   GPU work*, so the accelerator sits idle on network I/O and is billed for it.
   MSAs are per unique chain sequence, so the whole production panel needs six
   alignments -- there is no reason to pay GPU rates for them.
2. **`gpu="H100!"`, with the bang.** Plain `"H100"` lets Modal substitute an
   H200, which would silently make the H100 column an H200 measurement. Modal's
   own GPU docs call this out for benchmarking specifically.
3. **Weights live on a Volume and load once per container.** A batch of
   complexes runs in one container so the ~6.2 GB load is paid once, and the
   first fold is reported separately as warm-up rather than averaged in.
4. **CPU and memory are requested explicitly.** Modal's default is 0.125 cores
   and 128 MiB; the published Boltz example leaves it there. Billing charges the
   greater of requested and used, and HACKATHON_PLAN.md's rate table assumes
   4 cores / 32 GiB, so the request is pinned to match the budget.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import modal

# --- pinned versions -------------------------------------------------------
# Matches the version pinned by Modal's published Boltz example. A floating
# version would make the benchmark unreproducible.
BOLTZ_VERSION = "2.1.1"
HF_REPO = "boltz-community/boltz-2"
HF_REVISION = "6fdef46d763fee7fbb83ca5501ccceff43b85607"

# --- worker shape ----------------------------------------------------------
# Pinned to HACKATHON_PLAN.md's rate-table assumption so measured cost matches
# the budget arithmetic in scripts/gpu_decision.py.
WORKER_CPU = 4.0
WORKER_MEM = 32 * 1024  # MiB

# The four candidates from the plan. "H100!" blocks the H200 upgrade.
BENCH_GPUS = ["L40S", "A100-40GB", "A100-80GB", "H100!"]

MINUTES = 60

CACHE_DIR = Path("/weights/boltz")
MSA_DIR = Path("/msa")
OUT_DIR = Path("/structures")

weights_vol = modal.Volume.from_name("pepstab-boltz-weights", create_if_missing=True)
msa_vol = modal.Volume.from_name("pepstab-hla-msa", create_if_missing=True)
out_vol = modal.Volume.from_name("pepstab-structures", create_if_missing=True)

app = modal.App("pepstab-boltz")

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface-hub==0.36.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

boltz_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    f"boltz=={BOLTZ_VERSION}"
)

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# YAML construction
# --------------------------------------------------------------------------
def build_yaml(peptide: str, hla_seq: str, msa_ref: str) -> str:
    """A 2-chain Boltz-2 input: chain A the HLA domain, chain B the peptide.

    ``msa_ref`` is either a path to a ``.a3m`` for the HLA chain or the literal
    ``empty``. The peptide chain is always ``empty``: a 9-residue query has no
    meaningful alignment, and leaving the field off would make Boltz try to
    generate one.

    Boltz only accepts ``.a3m`` when a *single* protein chain carries a
    precomputed MSA -- two aligned chains would need the paired CSV format. That
    constraint is satisfied here precisely because the peptide is ``empty``.
    """
    return (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {hla_seq}\n"
        f"      msa: {msa_ref}\n"
        "  - protein:\n"
        "      id: B\n"
        f"      sequence: {peptide}\n"
        "      msa: empty\n"
    )


# --------------------------------------------------------------------------
# Setup: weights and MSAs, both CPU-only
# --------------------------------------------------------------------------
@app.function(
    image=download_image,
    volumes={CACHE_DIR.parent: weights_vol},
    timeout=30 * MINUTES,
)
def download_weights(force: bool = False) -> dict:
    """Pull the pinned Boltz-2 snapshot (~6.2 GB) onto the weights Volume."""
    from huggingface_hub import snapshot_download

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    snapshot_download(
        repo_id=HF_REPO,
        revision=HF_REVISION,
        local_dir=str(CACHE_DIR),
        force_download=force,
    )
    weights_vol.commit()
    total = sum(f.stat().st_size for f in CACHE_DIR.rglob("*") if f.is_file())
    return {
        "seconds": round(time.monotonic() - t0, 1),
        "bytes": total,
        "gb": round(total / 1e9, 3),
    }


@app.function(
    image=boltz_image,
    volumes={MSA_DIR: msa_vol},
    cpu=WORKER_CPU,
    memory=WORKER_MEM,
    timeout=60 * MINUTES,
)
def build_msas(targets: list[dict], force: bool = False) -> list[dict]:
    """Build one ``.a3m`` per unique HLA domain sequence, on CPU.

    ``targets`` is ``reports/boltz_msa_targets.csv`` as records: ``msa_id``,
    ``hla_seq``. All sequences go to the MSA server in a single batched request
    -- it is a free shared community service (ColabFold/MMseqs2), and eight
    sequences in one ticket is the polite and fast way to ask.
    """
    from boltz.data.msa.mmseqs2 import run_mmseqs2

    MSA_DIR.mkdir(parents=True, exist_ok=True)
    pending = [
        t for t in targets if force or not (MSA_DIR / f"{t['msa_id']}.a3m").exists()
    ]
    if not pending:
        return [{"msa_id": t["msa_id"], "status": "cached"} for t in targets]

    t0 = time.monotonic()
    result = run_mmseqs2(
        [t["hla_seq"] for t in pending],
        prefix=str(MSA_DIR / "_tmp"),
        use_env=True,
        use_filter=True,
        use_pairing=False,
    )
    # Signature is annotated as a tuple but the single-chain path returns just
    # the a3m list; accept either so a boltz point release cannot break setup.
    a3m_lines = result[0] if isinstance(result, tuple) else result
    elapsed = time.monotonic() - t0

    if len(a3m_lines) != len(pending):
        raise RuntimeError(
            f"MSA server returned {len(a3m_lines)} alignments for {len(pending)} queries"
        )

    out = []
    for target, lines in zip(pending, a3m_lines):
        path = MSA_DIR / f"{target['msa_id']}.a3m"
        path.write_text(lines)
        depth = sum(1 for ln in lines.splitlines() if ln.startswith(">"))
        out.append(
            {
                "msa_id": target["msa_id"],
                "status": "built",
                "depth": depth,
                "bytes": path.stat().st_size,
            }
        )
    msa_vol.commit()
    for rec in out:
        rec["total_seconds"] = round(elapsed, 1)
    return out


# --------------------------------------------------------------------------
# Folding
# --------------------------------------------------------------------------
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
def _read_panel(name: str) -> list[dict]:
    """Join a panel CSV against the MSA target table to attach hla_seq + msa_id."""
    targets = list(csv.DictReader((REPO / "reports" / "boltz_msa_targets.csv").open()))
    by_allele = {
        allele: t for t in targets for allele in t["alleles"].split("|")
    }
    rows = []
    for r in csv.DictReader((REPO / "reports" / name).open()):
        t = by_allele[r["allele"]]
        rows.append(
            {
                "complex_id": r["complex_id"],
                "allele": r["allele"],
                "peptide": r["peptide"],
                "hla_seq": t["hla_seq"],
                "msa_id": t["msa_id"],
            }
        )
    return rows


def _write_results(path: Path, batches: list[dict]) -> None:
    cols = [
        "gpu", "msa_mode", "complex_id", "allele", "peptide", "ok", "is_warmup",
        "fold_s", "billed_s", "startup_s", "peak_mem_gb", "has_structure",
        "has_pae", "confidence_score", "ptm", "iptm", "complex_plddt", "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for batch in batches:
            for rec in batch["results"]:
                rec = dict(rec)
                rec["startup_s"] = batch["startup_s"]
                # gpu_decision.py reads billed_s; steady-state cost is the fold
                # itself, with startup amortised separately by that script.
                rec["billed_s"] = rec.get("fold_s", "")
                w.writerow(rec)
    print(f"wrote {path}")


@app.local_entrypoint()
def setup(force: bool = False):
    """Download weights and build the HLA MSAs. CPU only -- no GPU is billed."""
    targets = list(csv.DictReader((REPO / "reports" / "boltz_msa_targets.csv").open()))
    print(f"weights: pulling {HF_REPO}@{HF_REVISION[:8]} ...")
    w = download_weights.remote(force=force)
    print(f"  {w['gb']} GB in {w['seconds']} s")

    print(f"MSAs: {len(targets)} unique HLA domain sequences ...")
    for rec in build_msas.remote(targets, force=force):
        depth = rec.get("depth")
        print(f"  {rec['msa_id']}: {rec['status']}" + (f", depth {depth}" if depth else ""))


@app.local_entrypoint()
def pilot(gpu: str = "L40S", both_arms: bool = True):
    """Fold the 5 pilot complexes end to end. The gate on any batch job.

    Runs the MSA arm and, by default, a single-sequence arm on the same five
    complexes, so the pilot also measures what the MSA actually buys here.
    """
    complexes = _read_panel("boltz_pilot.csv")
    print(f"pilot: {len(complexes)} complexes on {gpu}")
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

    _write_results(REPO / "reports" / "boltz_pilot_results.csv", batches)
    print(
        "\nBefore running ::benchmark, confirm the poses sit in the groove "
        "(peptide RMSD against the crystal structures in reports/boltz_pilot.csv)."
    )


@app.local_entrypoint()
def benchmark(gpus: str = ",".join(BENCH_GPUS)):
    """Fold the 24-complex panel on each GPU with identical settings."""
    complexes = _read_panel("boltz_bench_panel.csv")
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

    _write_results(REPO / "reports" / "boltz_bench_results.csv", batches)
    print("\nNow run: python scripts/gpu_decision.py")
