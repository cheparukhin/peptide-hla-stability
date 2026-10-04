"""Shared configuration for the ESMFold2 Modal apps.

ESMFold2 (Biohub, MIT) is a diffusion structure predictor on ESMC embeddings.
It is a candidate replacement for Boltz-2 in stage 4 on one axis only: speed.
Its published anchor is 15.8 s for a 1,024-residue complex against our measured
46.6 s for a 191-residue one on A10, so a large win is plausible -- but two
things have to be established by measurement before that matters:

* **It has to fit a cheap card.** The HF card lists 7B params in F32 = ~28 GB of
  weights before activations. Our cost winner (A10) has 24 GB. If F32 is the
  only option, ESMFold2 is forced onto >=40 GB cards and has to beat A10 from a
  worse rate, which the arithmetic says it still can -- but only if it is fast.
* **It has to put the peptide in the groove.** Its DockQ wins are antibody-
  antigen and general protein-protein. A 9-mer pinned in an HLA groove is a
  different regime, and Boltz-2 is already at 0.13-0.42 A peptide CA RMSD.

Panel loading is reused from ``boltz_common`` so both models fold byte-identical
inputs; only the folding engine differs.
"""

from __future__ import annotations

from pathlib import Path

import modal

# --- pinned versions ------------------------------------------------------
# Set from the probe's reported version. A floating version would make the
# benchmark unreproducible, the same reason boltz is pinned to 2.1.1.
ESM_PACKAGE = "esm==3.4.1.post1"  # version the probe reported; pinned for reproducibility
HF_REVISION = "69869f737beffec5294845ede23db5fc0b4f509e"
HF_REPO = "biohub/ESMFold2"  # == esm.models.esmfold2.ESMFOLD2_HF_REPO

# --- worker shape ---------------------------------------------------------
# Matches the Boltz-2 benchmark's right-sized request so the two models are
# compared on identical host cost. Re-measure if ESMFold2's host RSS differs:
# a 7B model's weight load touches far more host RAM than Boltz-2's 6.2 GB.
WORKER_CPU = 4.0
WORKER_MEM = 16 * 1024  # MiB

MINUTES = 60

ESM_CACHE = Path("/esmweights")
OUT_DIR = Path("/structures")

esm_weights_vol = modal.Volume.from_name("pepstab-esmfold2-weights", create_if_missing=True)
out_vol = modal.Volume.from_name("pepstab-structures", create_if_missing=True)

# Both modules must ride along: Modal uploads the entrypoint file but not its
# siblings, and a missing sibling crash-loops containers on the clock while
# `modal run` buffers the traceback out of sight. See boltz_common.
_SHARED = ("esmfold_common", "boltz_common")

# CPU-only: introspects the API and pulls weights to the Volume. No torch CUDA
# needed, so this stays a small image.
probe_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface-hub==0.36.0", ESM_PACKAGE)
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source(*_SHARED)
)

esmfold_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(ESM_PACKAGE)
    .env({"HF_HUB_DISABLE_TELEMETRY": "1"})
    .add_local_python_source(*_SHARED)
)
