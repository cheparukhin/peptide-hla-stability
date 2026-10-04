"""CPU-only ESMFold2 setup: introspect the API, then cache the weights.

    modal run modal_app/esmfold_setup.py::probe    # free-ish: no weights, no GPU
    modal run modal_app/esmfold_setup.py::setup    # pull weights to the Volume

``probe`` exists because the previous stage lost credits to a container that
crash-looped on a wrong import while `modal run` buffered the traceback out of
sight. ESMFold2's Python API is documented by one example; the MSA entry point,
the PAE attribute and the dtype argument are all unshown. Guessing them on a
GPU worker costs money per attempt, so they get resolved here instead, where
the worker is 0.25 cores and no weights are loaded.

Split from the GPU app for the same reason as the Boltz pair: Modal validates
every function in an app at creation, so a GPU declaration can block CPU work.
"""

from __future__ import annotations

import inspect
import json

import modal

from esmfold_common import (
    ESM_CACHE,
    HF_REPO,
    HF_REVISION,
    MINUTES,
    esm_weights_vol,
    probe_image,
)

app = modal.App("pepstab-esmfold-setup")


@app.function(
    image=probe_image,
    timeout=10 * MINUTES,
    cpu=0.25,
    memory=2048,
    max_containers=1,
    retries=0,
)
def probe() -> dict:
    """Report the real API surface and the weight manifest. Loads no weights.

    Answers, in order of how much they can cost later:

    1. Which ``esm`` package did PyPI actually give us, and does
       ``esm.models.esmfold2`` import? (A wrong package is a silent crash-loop.)
    2. Can ``from_pretrained`` take a dtype? bf16 halves 28 GB to 14 GB, which
       is the difference between needing a 48 GB card and fitting our A10.
    3. Does ``ProteinInput`` accept an MSA, and under what argument name?
    4. What is the PAE attribute called on the result? Stage 5 needs the matrix.
    """
    out: dict = {}

    try:
        import esm

        out["esm_version"] = getattr(esm, "__version__", "unknown")
        out["esm_path"] = getattr(esm, "__file__", "unknown")
    except Exception as exc:
        return {"fatal": f"import esm failed: {exc!r}"}

    try:
        from esm.models import esmfold2 as m
    except Exception as exc:
        out["fatal"] = f"import esm.models.esmfold2 failed: {exc!r}"
        out["esm_submodules"] = _safe(lambda: sorted(_list_submodules("esm.models")))
        return out

    out["esmfold2_exports"] = sorted(n for n in dir(m) if not n.startswith("_"))

    for name in ("EsmFold2Model", "ProteinInput", "StructurePredictionInput",
                 "ESMFold2InputBuilder", "Modification"):
        obj = getattr(m, name, None)
        if obj is None:
            out[f"sig::{name}"] = "ABSENT"
            continue
        out[f"sig::{name}"] = _safe(lambda o=obj: str(inspect.signature(o)))
        if name == "EsmFold2Model":
            fp = getattr(obj, "from_pretrained", None)
            if fp is not None:
                out["sig::EsmFold2Model.from_pretrained"] = _safe(
                    lambda: str(inspect.signature(fp))
                )
        if name == "ESMFold2InputBuilder":
            fold = getattr(obj, "fold", None)
            if fold is not None:
                out["sig::ESMFold2InputBuilder.fold"] = _safe(
                    lambda: str(inspect.signature(fold))
                )
                out["doc::fold"] = (inspect.getdoc(fold) or "")[:1500]

    # Which field carries an MSA, if any. Named explicitly because the one
    # published example is single-sequence and the Biohub benchmark says MSA
    # mode is where ESMFold2 is strongest.
    pi = getattr(m, "ProteinInput", None)
    if pi is not None:
        out["protein_input_fields"] = _safe(
            lambda: sorted(getattr(pi, "model_fields", None) or _dataclass_fields(pi))
        )

    # The result type's attributes -- specifically whether PAE is exposed, and
    # under which of pae / pAE / predicted_aligned_error. Stage 5's confidence
    # feature arm needs the matrix, so an absent PAE is disqualifying.
    for cand in ("MolecularComplexResult", "MolecularComplex",
                 "MolecularComplexMetadata", "ChainInfo"):
        obj = getattr(m, cand, None)
        if obj is not None:
            out[f"result_fields::{cand}"] = _safe(
                lambda o=obj: sorted(
                    getattr(o, "model_fields", None) or _dataclass_fields(o)
                )
            )

    # The canonical repo ids, so the pin is the library's own constant rather
    # than a string we typed.
    for const in ("ESMFOLD2_HF_REPO", "ESMFOLD2_EXPERIMENTAL_HF_REPO"):
        out[f"const::{const}"] = _safe(lambda c=const: str(getattr(m, c)))

    # Anything PAE-shaped anywhere in the output module.
    try:
        from esm.models.esmfold2 import output as om

        out["output_module_exports"] = sorted(
            n for n in dir(om) if not n.startswith("_")
        )
        for cand in sorted(n for n in dir(om) if not n.startswith("_")):
            obj = getattr(om, cand)
            if inspect.isclass(obj):
                fields = _safe(
                    lambda o=obj: sorted(
                        getattr(o, "model_fields", None) or _dataclass_fields(o)
                    )
                )
                if isinstance(fields, list) and any(
                    "pae" in f.lower() or "align" in f.lower() for f in fields
                ):
                    out[f"PAE_IN::{cand}"] = fields
    except Exception as exc:
        out["output_module_error"] = repr(exc)

    # mmCIF writer signature: the Boltz run wrote mmCIF, and matching the
    # output format keeps scripts/boltz_pose_check.py reusable unchanged.
    mc = getattr(m, "MolecularComplex", None)
    if mc is not None:
        for meth in ("to_mmcif", "to_pdb"):
            fn = getattr(mc, meth, None)
            if fn is not None:
                out[f"sig::MolecularComplex.{meth}"] = _safe(
                    lambda f=fn: str(inspect.signature(f))
                )

    # Weight manifest: file sizes reveal the real on-disk footprint and whether
    # a half-precision or -Fast variant is shipped alongside the F32 weights.
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(HF_REPO, files_metadata=True)
        files = [
            {"name": s.rfilename, "gb": round((s.size or 0) / 1e9, 2)}
            for s in (info.siblings or [])
        ]
        out["hf_files"] = sorted(
            [f for f in files if f["gb"] > 0.01], key=lambda f: -f["gb"]
        )[:25]
        out["hf_total_gb"] = round(sum(f["gb"] for f in files), 2)
    except Exception as exc:
        out["hf_manifest_error"] = repr(exc)

    return out


def _safe(fn) -> str:
    try:
        return fn()
    except Exception as exc:
        return f"<{exc!r}>"


def _dataclass_fields(obj) -> list[str]:
    import dataclasses

    if dataclasses.is_dataclass(obj):
        return [f.name for f in dataclasses.fields(obj)]
    return [n for n in dir(obj) if not n.startswith("_")]


def _list_submodules(pkg: str) -> list[str]:
    import importlib
    import pkgutil

    mod = importlib.import_module(pkg)
    return [m.name for m in pkgutil.iter_modules(mod.__path__)]


@app.function(
    image=probe_image,
    volumes={ESM_CACHE: esm_weights_vol},
    timeout=60 * MINUTES,
    cpu=2.0,  # the download is throughput-bound, unlike the MSA poller
    memory=8192,
    max_containers=1,
    retries=0,
)
def download_weights(allow_patterns: list[str] | None = None) -> dict:
    """Snapshot ESMFold2 to the Volume so GPU workers never pay for the pull.

    ``allow_patterns`` lets the probe's manifest drive a narrower download --
    if the repo ships both F32 and half-precision shards there is no reason to
    pay for storage on both.
    """
    import time

    from huggingface_hub import snapshot_download

    t0 = time.monotonic()
    path = snapshot_download(
        HF_REPO,
    HF_REVISION,
        local_dir=str(ESM_CACHE / "ESMFold2"),
        allow_patterns=allow_patterns,
    )
    esm_weights_vol.commit()

    total = sum(f.stat().st_size for f in (ESM_CACHE / "ESMFold2").rglob("*") if f.is_file())
    return {
        "path": path,
        "seconds": round(time.monotonic() - t0, 1),
        "gb": round(total / 1e9, 3),
    }


@app.local_entrypoint()
def main(download: bool = False, patterns: str = ""):
    """Probe first; download only when asked.

    Default is probe-only: it is the cheap step and it decides what the
    download should even fetch.
    """
    info = probe.remote()
    print(json.dumps(info, indent=2, default=str))

    if info.get("fatal"):
        print("\nPROBE FAILED -- resolve the import before spending on a GPU.")
        return

    if not download:
        print(
            "\nProbe only. Re-run with --download once the manifest above is "
            "understood:\n  modal run modal_app/esmfold_setup.py --download"
        )
        return

    pats = [p.strip() for p in patterns.split(",") if p.strip()] or None
    res = download_weights.remote(allow_patterns=pats)
    print(f"\nweights: {res['gb']} GB in {res['seconds']} s -> {res['path']}")
