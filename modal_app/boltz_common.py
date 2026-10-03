"""Shared configuration for the Boltz-2 Modal apps.

Split from the apps themselves because Modal validates *every* function in an
app at creation time: a workspace without a payment method cannot even declare
a `gpu=` function, which would otherwise make the CPU-only MSA step
unreachable. Keeping the GPU and CPU work in separate apps over shared Volumes
means the MSA cache can be built while GPU access is still pending.
"""

from __future__ import annotations

import csv
from pathlib import Path

import modal

# --- pinned versions -------------------------------------------------------
# Matches the version pinned by Modal's published Boltz example. A floating
# version would make the benchmark unreproducible.
BOLTZ_VERSION = "2.1.1"
HF_REPO = "boltz-community/boltz-2"
HF_REVISION = "6fdef46d763fee7fbb83ca5501ccceff43b85607"

# --- worker shape ----------------------------------------------------------
# Pinned to HACKATHON_PLAN.md's rate-table assumption (4 cores / 32 GiB) so
# measured cost matches the budget arithmetic in scripts/gpu_decision.py.
# Modal bills the greater of requested and used.
WORKER_CPU = 4.0
WORKER_MEM = 32 * 1024  # MiB

# The four candidates from the plan. "H100!" blocks Modal's automatic upgrade
# to an H200, which would otherwise make the H100 column an H200 measurement.
BENCH_GPUS = ["L40S", "A100-40GB", "A100-80GB", "H100!"]

MINUTES = 60

CACHE_DIR = Path("/weights/boltz")
MSA_DIR = Path("/msa")
OUT_DIR = Path("/structures")

weights_vol = modal.Volume.from_name("pepstab-boltz-weights", create_if_missing=True)
msa_vol = modal.Volume.from_name("pepstab-hla-msa", create_if_missing=True)
out_vol = modal.Volume.from_name("pepstab-structures", create_if_missing=True)

# Every image must carry THIS module: Modal uploads the entrypoint file but not
# its siblings, so without add_local_python_source the container dies on
# `from boltz_common import ...` -- and because `modal run` buffers stdout, the
# failure shows up only in `modal app logs` while containers crash-loop on the
# clock. Learned the expensive way.
_SHARED = ("boltz_common",)

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface-hub==0.36.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source(*_SHARED)
)

boltz_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(f"boltz=={BOLTZ_VERSION}")
    .add_local_python_source(*_SHARED)
)

REPO = Path(__file__).resolve().parent.parent


def build_yaml(peptide: str, hla_seq: str, msa_ref: str) -> str:
    """A 2-chain Boltz-2 input: chain A the HLA domain, chain B the peptide.

    ``msa_ref`` is either a path to a ``.a3m`` for the HLA chain or the literal
    ``empty``. The peptide chain is always ``empty``: a 9-residue query has no
    meaningful alignment, and omitting the field would make Boltz try to build
    one.

    Boltz only accepts ``.a3m`` when a *single* protein chain carries a
    precomputed MSA -- two aligned chains would need the paired CSV format.
    That constraint holds here precisely because the peptide is ``empty``.
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


def read_msa_targets() -> list[dict]:
    return list(csv.DictReader((REPO / "reports" / "boltz_msa_targets.csv").open()))


def read_panel(name: str) -> list[dict]:
    """Join a panel CSV against the MSA target table to attach hla_seq + msa_id."""
    by_allele = {
        allele: t for t in read_msa_targets() for allele in t["alleles"].split("|")
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


def write_results(path: Path, batches: list[dict]) -> None:
    cols = [
        "gpu", "msa_mode", "complex_id", "allele", "peptide", "ok", "is_warmup",
        "fold_s", "billed_s", "startup_s", "peak_mem_gb", "has_structure",
        "has_pae", "confidence_score", "ptm", "iptm", "complex_plddt", "error",
    ]
    path.parent.mkdir(exist_ok=True)
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
