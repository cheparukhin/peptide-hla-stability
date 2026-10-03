"""ESMFold2 GPU folding: the pilot gate and the hardware benchmark.

    modal run modal_app/esmfold_bench.py::pilot       # 3 complexes, the gate
    modal run modal_app/esmfold_bench.py::benchmark   # panel x GPUs

Run `esmfold_setup.py --download` first to cache the 26.74 GB snapshot on a
Volume. CLAUDE.md gates this the same way the Boltz app is gated: no batch GPU
job without a passing end-to-end pilot on 3-5 examples, poses checked with
`scripts/boltz_pose_check.py` (which is model-agnostic -- it locates chains by
sequence, so it reads ESMFold2 mmCIF unchanged).

Results are written in the Boltz result schema so `scripts/gpu_decision.py
--results` applies the identical cost model to both engines. The comparison is
only honest if nothing but the folding engine differs, so the panels, the host
shape (4 cores / 16 GiB) and the output format are all shared.

**One setting is deliberately not matched.** Boltz-2 ran `--recycling_steps 3`;
ESMFold2's analogue, `num_loops`, defaults to 20. Forcing either model off its
own default would measure a crippled configuration, so each runs at its
recommended settings and the comparison is *model-at-default* against
*model-at-default*. `--num-loops` exposes the knob so the trade can be measured
rather than assumed. `num_sampling_steps=200` and `num_diffusion_samples=1` do
match Boltz-2 exactly.

Unlike boltz, ESMFold2 runs in-process rather than as a subprocess, so torch's
own allocator counters are available and more precise than polling nvidia-smi.
Both are recorded: nvidia-smi is what the Boltz-2 numbers were measured with,
and the two must stay comparable.
"""

from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import modal

from boltz_common import MSA_DIR, msa_vol, read_panel, write_results
from esmfold_common import (
    ESM_CACHE,
    MINUTES,
    OUT_DIR,
    WORKER_CPU,
    WORKER_MEM,
    esm_weights_vol,
    esmfold_image,
    out_vol,
)

REPO = Path(__file__).resolve().parent.parent

app = modal.App("pepstab-esmfold")

# The pilot measured 26.0 GB single-sequence and 26.9 GB with an MSA, so the
# 24 GB cards that won the Boltz-2 sweep (A10, L4) cannot hold ESMFold2 at all
# -- the bf16 ESMC backbone is not enough to fit a 7B model plus a diffusion
# trunk into 24 GB. A100-80GB is dominated by A100-40GB for the same reason it
# was in the Boltz sweep: 13 GB of headroom is already enough.
BENCH_GPUS = ["L40S", "A100-40GB", "H100!"]


def _child_peak_rss_gb() -> float:
    """Peak host RSS, in GB. ESMFold2 is in-process, so count self, not children."""
    try:
        import resource

        me = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        return round(max(me, kids) / 1024 / 1024, 2)  # Linux reports kibibytes
    except Exception:
        return -1.0


class _GpuMemoryProbe:
    """Polls nvidia-smi on a thread, to stay comparable with the Boltz-2 figures."""

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.peak_mib = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
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
    image=esmfold_image,
    volumes={ESM_CACHE: esm_weights_vol, MSA_DIR: msa_vol, OUT_DIR: out_vol},
    gpu="L40S",  # overridden per call via .with_options(gpu=...)
    cpu=WORKER_CPU,
    memory=WORKER_MEM,
    timeout=120 * MINUTES,
    scaledown_window=10,
    max_containers=1,
    retries=0,
)
def fold_batch(
    complexes: list[dict],
    gpu_label: str,
    num_loops: int = 20,
    num_sampling_steps: int = 200,
    use_msa: bool = False,
    save_structures: bool = True,
    report_api: bool = False,
) -> dict:
    """Fold a batch in one container, timing the model load and each fold apart.

    ``model_load_s`` is reported separately from the folds because a 26.74 GB
    snapshot is a far larger per-container cost than Boltz-2's 6.2 GB, and the
    cost model amortises it over the production batch size rather than charging
    it to every complex.
    """
    import torch
    from esm.models.esmfold2 import (
        MSA,
        ESMFold2InputBuilder,
        EsmFold2Model,
        ProteinInput,
        StructurePredictionInput,
    )

    container_t0 = time.monotonic()
    local = ESM_CACHE / "ESMFold2"
    # ccd_cache is the *directory* holding ccd.pkl, not the file: load_ccd()
    # mkdirs the path it is given, so a file path raises FileExistsError.
    ccd_dir = local if (local / "ccd.pkl").exists() else None

    load_t0 = time.monotonic()
    with _GpuMemoryProbe() as load_probe:
        model = EsmFold2Model.from_pretrained(str(local), device="cuda").eval()
        builder = ESMFold2InputBuilder(ccd_cache=ccd_dir)
    model_load_s = round(time.monotonic() - load_t0, 2)

    mode = "msa" if use_msa else "single_seq"
    out: dict = {
        "gpu": gpu_label,
        "msa_mode": mode,
        "model_load_s": model_load_s,
        "load_peak_gb": load_probe.peak_gb,
        "num_loops": num_loops,
        "num_sampling_steps": num_sampling_steps,
        "esm_version": _version(),
    }

    # One MSA per unique HLA sequence, cached across folds in this container.
    # The alignments are the same .a3m files the Boltz-2 run used, so the MSA
    # arms of the two engines see byte-identical evolutionary input.
    msa_cache: dict[str, object] = {}

    def hla_msa(msa_id: str):
        if msa_id not in msa_cache:
            path = MSA_DIR / f"{msa_id}.a3m"
            msa_cache[msa_id] = MSA.from_a3m(str(path)) if path.exists() else None
        return msa_cache[msa_id]

    if report_api:
        # The MSA path is the obvious follow-up if single-sequence quality is
        # short. Capture its signature here rather than paying for another run.
        out["msa_api"] = _msa_api()

    results = []
    startup_s = None

    for i, cx in enumerate(complexes):
        # Only the HLA chain carries an MSA. A 9-residue peptide has no
        # meaningful alignment and its identity is the variable under study --
        # the same reasoning behind `msa: empty` in the Boltz-2 YAML.
        spi = StructurePredictionInput(
            sequences=[
                ProteinInput(
                    id="A",
                    sequence=cx["hla_seq"],
                    msa=hla_msa(cx["msa_id"]) if use_msa else None,
                ),
                ProteinInput(id="B", sequence=cx["peptide"]),
            ]
        )

        torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        rec: dict = {
            "complex_id": cx["complex_id"],
            "allele": cx["allele"],
            "peptide": cx["peptide"],
            "gpu": gpu_label,
            "msa_mode": mode,
            "is_warmup": i == 0,
        }
        try:
            with _GpuMemoryProbe() as probe:
                res = builder.fold(
                    model,
                    spi,
                    num_loops=num_loops,
                    num_sampling_steps=num_sampling_steps,
                    num_diffusion_samples=1,
                    seed=0,
                )
            if isinstance(res, list):
                res = res[0]
            fold_s = time.monotonic() - t0
            rec.update(
                {
                    "ok": True,
                    "fold_s": round(fold_s, 2),
                    "peak_mem_gb": probe.peak_gb,
                    "torch_peak_alloc_gb": round(
                        torch.cuda.max_memory_allocated() / 1e9, 2
                    ),
                    "torch_peak_reserved_gb": round(
                        torch.cuda.max_memory_reserved() / 1e9, 2
                    ),
                    "host_peak_rss_gb": _child_peak_rss_gb(),
                }
            )
            rec.update(_collect_outputs(res, cx, gpu_label, mode, save_structures))
        except Exception as exc:
            rec.update(
                {
                    "ok": False,
                    "fold_s": round(time.monotonic() - t0, 2),
                    "error": f"{type(exc).__name__}: {exc}"[:2000],
                }
            )

        if startup_s is None:
            # Everything before the first fold: volume mount, imports, and the
            # 26.74 GB weight load. Modal bills all of it.
            startup_s = round(t0 - container_t0, 2)
        results.append(rec)

    if save_structures:
        out_vol.commit()

    out.update(
        {
            "startup_s": startup_s,
            "container_s": round(time.monotonic() - container_t0, 2),
            "n": len(complexes),
            "results": results,
        }
    )
    return out


def _version() -> str:
    try:
        import esm

        return getattr(esm, "__version__", "unknown")
    except Exception:
        return "unknown"


def _msa_api() -> str:
    import inspect

    try:
        from esm.utils.msa.msa import MSA

        sig = str(inspect.signature(MSA))
        ctors = [
            f"{n}{inspect.signature(getattr(MSA, n))}"
            for n in dir(MSA)
            if n.startswith("from_") and callable(getattr(MSA, n, None))
        ]
        return f"MSA{sig} | constructors: {'; '.join(ctors)}"
    except Exception as exc:
        return f"<{exc!r}>"


def _collect_outputs(res, cx: dict, gpu_label: str, mode: str, save: bool) -> dict:
    """Harvest confidence scores and write the mmCIF for the pose check."""
    info: dict = {}

    def scalar(v):
        try:
            return round(float(v), 4)
        except Exception:
            try:
                return round(float(v.mean()), 4)
            except Exception:
                return ""

    info["complex_plddt"] = scalar(getattr(res, "plddt", None))
    info["ptm"] = scalar(getattr(res, "ptm", None))
    info["iptm"] = scalar(getattr(res, "iptm", None))
    # Stage 5 wants the peptide-HLA interface specifically, which
    # pair_chains_iptm gives directly -- Boltz-2 only exposes a global ipTM.
    pair = getattr(res, "pair_chains_iptm", None)
    if pair is not None:
        try:
            info["pair_chains_iptm"] = str(pair)[:200]
        except Exception:
            pass

    pae = getattr(res, "pae", None)
    info["has_pae"] = pae is not None
    if pae is not None:
        try:
            info["pae_shape"] = str(tuple(pae.shape))
        except Exception:
            pass

    cif = None
    try:
        cif = res.complex.to_mmcif()
    except Exception as exc:
        info["mmcif_error"] = str(exc)[:300]
    info["has_structure"] = cif is not None

    if save and cif is not None:
        dest = (
            OUT_DIR / "esmfold2" / gpu_label.replace("!", "") / mode / cx["complex_id"]
        )
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{cx['complex_id']}.cif").write_text(cif)
        if pae is not None:
            try:
                import numpy as np

                np.save(dest / "pae.npy", np.asarray(pae.float().cpu()))
            except Exception:
                pass
    return info


# --------------------------------------------------------------------------
# Local entrypoints
# --------------------------------------------------------------------------


@app.local_entrypoint()
def pilot(
    gpu: str = "L40S", limit: int = 3, num_loops: int = 20, use_msa: bool = False
):
    """Fold the pilot complexes end to end. The gate on any batch job.

    Defaults spend as little as possible while still clearing the gate:
    ``limit=3`` is CLAUDE.md's floor, single-sequence (the fast path and the
    reason ESMFold2 is a candidate at all), and L40S rather than a 24 GB card
    so a VRAM surprise cannot waste the run. The measured peak decides whether
    A10 and L4 enter the sweep.

    ``--use-msa`` runs the arm that is actually matched against the Boltz-2
    numbers, which were measured with these same alignments.
    """
    complexes = read_panel("boltz_pilot.csv")[:limit]
    print(
        f"esmfold2 pilot: {len(complexes)} complexes on {gpu}, "
        f"{'msa' if use_msa else 'single-sequence'}, num_loops={num_loops}"
    )
    b = fold_batch.with_options(gpu=gpu).remote(
        complexes,
        gpu_label=gpu,
        num_loops=num_loops,
        use_msa=use_msa,
        report_api=True,
    )

    ok = sum(1 for r in b["results"] if r["ok"])
    print(f"\nesm {b['esm_version']}  model load {b['model_load_s']} s "
          f"(peak {b['load_peak_gb']} GB)  startup {b['startup_s']} s")
    print(f"{ok}/{b['n']} ok\n")
    for r in b["results"]:
        tag = " (warmup)" if r["is_warmup"] else ""
        if r["ok"]:
            print(
                f"  {r['complex_id']:22} {r['fold_s']:7.1f} s  "
                f"smi {r['peak_mem_gb']:5.1f} GB  "
                f"torch {r.get('torch_peak_reserved_gb', '?')} GB  "
                f"iptm={r.get('iptm', 'n/a')}  pae={r.get('has_pae')}"
                f" {r.get('pae_shape', '')}{tag}"
            )
        else:
            print(f"  {r['complex_id']:22} FAILED{tag}: {r.get('error', '')[:400]}")

    if b.get("msa_api"):
        print(f"\nMSA API (for the MSA arm, if needed):\n  {b['msa_api']}")

    peaks = [r["peak_mem_gb"] for r in b["results"] if r["ok"]]
    if peaks:
        peak = max(peaks)
        print(f"\npeak GPU memory: {peak} GB")
        print(
            f"  24 GB cards (A10, L4): {'VIABLE' if peak < 21 else 'TOO TIGHT'}"
            "  <- decides the sweep's candidate set"
        )

    suffix = "_msa" if use_msa else ""
    write_results(REPO / "reports" / f"esmfold_pilot_results{suffix}.csv", [b])
    print(
        "\nBefore ::benchmark, check the poses:\n"
        "  python scripts/boltz_pose_check.py --structures <dir>"
    )


@app.local_entrypoint()
def benchmark(gpus: str = ",".join(BENCH_GPUS), limit: int = 8, num_loops: int = 20):
    """Fold the benchmark panel on each GPU with identical settings.

    Same panel and same ``limit`` as the Boltz-2 sweep, so the two cost tables
    are directly comparable.
    """
    complexes = read_panel("boltz_bench_panel.csv")[:limit]
    labels = [g.strip() for g in gpus.split(",") if g.strip()]
    print(f"esmfold2 benchmark: {len(complexes)} complexes x {len(labels)} GPUs: {labels}")

    handles = [
        (
            g,
            fold_batch.with_options(gpu=g).spawn(
                complexes, gpu_label=g, num_loops=num_loops
            ),
        )
        for g in labels
    ]
    batches = []
    for g, h in handles:
        try:
            batches.append(h.get())
            print(f"  {g}: done")
        except Exception as exc:
            print(f"  {g}: batch failed -- {exc}")

    write_results(REPO / "reports" / "esmfold_bench_results.csv", batches)
    print(
        "\nNow run:\n"
        "  python scripts/gpu_decision.py --results reports/esmfold_bench_results.csv"
        " --gib 16"
    )
