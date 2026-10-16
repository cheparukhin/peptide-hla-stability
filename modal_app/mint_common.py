"""Shared configuration for the Phase C gate apps (MINT joint peptide-HLA encoder).

MINT (Ullanat & Zitnik, *Nat Commun* 2025) is ESM-2 650M with a per-layer
cross-chain attention module, pretrained on 96M STRING protein-protein
interactions. It is the interaction-pretrained, chain-aware backbone that
`reports/archive/plans/further-research-2026-10-04.md` §6 names as the principal GPU direction.

Only the **base** checkpoint is audited clean for us. SPEARMINT's stage-2 and
final checkpoints were fine-tuned on the NetMHCstabpan corpus, which is our
corpus, and its stage-1 affinity checkpoint saw supervised affinity labels
inside ~90% of our held-out peptide clusters. See
`reports/further_research/phase_c_gate.md`.

Three reasons this module exists separately from the gate app, all of them
copied from the ESMFold2 pair that preceded it:

* **Modal validates every function in an app at creation**, so a GPU
  declaration in the same app can block the CPU probe. The probe is the step
  that costs nothing and decides whether the GPU step is worth starting.
* **Modal uploads the entrypoint file but not its siblings.** A missing sibling
  crash-loops containers on the clock while `modal run` buffers the traceback
  out of sight. Hence `add_local_python_source`.
* **The MINT checkpoint is a PyTorch Lightning checkpoint loaded by plain
  `torch.load(weights_only=False)`.** Whether that unpickles without
  `lightning` installed is unknown until measured, and it is the single most
  likely way to lose a GPU container to a crash-loop. `mint_gate.py::setup`
  resolves it on a CPU worker and writes a clean state dict for the GPU to use.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

# --- pinned sources -------------------------------------------------------
# Pinned by commit and revision for the same reason boltz is pinned to 2.1.1:
# a floating version makes the benchmark, and the leakage audit behind it,
# unreproducible.
MINT_GITHUB = "VarunUllanat/mint"
MINT_GITHUB_SHA = "06694b7606e2d00b76ec58daf5c7aecdaf7cd283"  # main @ 2026-06-15
MINT_HF_REPO = "varunullanat2012/mint"
MINT_HF_REVISION = "75b1602c34e9df1a89089a1245f000a51ee3c746"
MINT_CKPT_FILE = "mint.ckpt"  # 3.254 GB, MIT
# The ESM-2 650M architecture config lives in the GitHub repo, not the HF repo:
# the HF repo holds only `mint.ckpt` and `bernett_mlp.pth`.
MINT_CONFIG = "data/esm2_t33_650M_UR50D.json"

# The SPEARMINT ladder, recorded so the gate report's access table is
# reproducible. NOT loaded by the smoke: both are disqualified by the leakage
# audit. Kept here because an audit that cannot be re-run is not an audit.
SPEARMINT_REPOS = {
    "dkarthikeyan1/mint-stage1-affinity": "8bf8e51906cf63336706d6cfc4f85e95a8109c86",
    "dkarthikeyan1/mint-stage2-stability": "d3bad29b1e2e9f4313b2187dfa215a748565a58e",
    "dkarthikeyan1/spearmint": "93e8e3e640e49e5a373f0dc432b122df19e4b969",
}

# --- worker shape ---------------------------------------------------------
# L4 (24 GiB) rather than the A10G the folding stages used. MINT is 813M params
# = 3.25 GiB in F32, and the construct is 195 tokens against the folding
# stages' 383 residues with a 1,024-row MSA, so this is a far smaller job than
# Boltz-2 was. L4 at $0.80/h is the cheapest card that holds F32 weights plus a
# top-layer backward pass with headroom; T4 at $0.59/h is 16 GiB and has no
# bf16, which would force a precision choice onto a cost decision.
GPU = "L4"
WORKER_CPU = 4.0
WORKER_MEM = 16 * 1024  # MiB

MINUTES = 60

WEIGHTS = Path("/mintweights")
mint_weights_vol = modal.Volume.from_name("pepstab-mint-weights", create_if_missing=True)

_SHARED = ("mint_common",)

# `mint`'s extraction path -- mint/__init__.py -> model.esm2 -> modules ->
# axial_attention, multihead_attention, rotary_embedding -- imports only torch.
# `apex` and `torchvision` are behind try/except or lazy, and `lightning`,
# `wandb` and `Bio` are reached only from mint/utils/ and mint/helpers/predict,
# which __init__ does not import. So the image needs torch and nothing else
# from the published conda environment, whose python=3.7.12 / pytorch=1.12.1
# pins we deliberately do not reproduce: the repo's own
# `torch.load(..., weights_only=False)` call needs torch >= 1.13 anyway.
# Installed from the commit tarball rather than `git+https://`: `debian_slim`
# ships no git, and an `apt_install("git")` layer would be 40 MB and a minute
# of build time to fetch a 2 MB source tree. The sha still pins it exactly.
_MINT_SRC = f"https://github.com/{MINT_GITHUB}/archive/{MINT_GITHUB_SHA}.tar.gz"

probe_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch==2.5.1", "pandas==2.2.3", "scipy==1.14.1", "huggingface-hub==0.36.0")
    .uv_pip_install(_MINT_SRC)
    .env({"HF_HUB_DISABLE_TELEMETRY": "1", "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source(*_SHARED)
)

# Same image for the GPU worker: torch 2.5.1 ships its own CUDA wheels, so
# there is no second build and no risk of the two steps disagreeing about the
# torch version that loaded the checkpoint.
mint_image = probe_image

# --- construct contract ---------------------------------------------------
# Chain 1 is the 9-mer. Chain 2 is the 182-residue alpha1/alpha2 groove domain
# carried by `hla_seq` in the dataset -- the same input set as the project's
# existing `pep_domain` arm, so a Phase C result is comparable to an arm we
# already measured. The 3-chain ectodomain variant (275 + 99 aa, from
# `hla75_ectodomain_b2m.csv`) is timed alongside it in the smoke purely to put
# a sensitivity range on the budget forecast; it is not the Phase C contract.
CONTRACTS = ("pep_domain", "pep_ectodomain_b2m")

# --- rates ----------------------------------------------------------------
RATES_JSON = Path(__file__).resolve().parent.parent / "reports" / "ectodomain_rates.json"


def rates() -> dict[str, float]:
    """The workspace rate card, as captured for the stage 4c production run."""
    return {k: float(v) for k, v in json.loads(RATES_JSON.read_text()).items()}


def hourly_usd(gpu: str | None = GPU, cpu: float = WORKER_CPU, mem_mib: int = WORKER_MEM) -> float:
    """Container rate for one worker of the given shape.

    A GPU container pays CPU and memory on top of the card, which is why the
    stage 4c A10G shape quoted $1.4812/h and not $1.10.
    """
    r = rates()
    usd = cpu * r["cpu_hour_cost"] + (mem_mib / 1024) * r["mem_gib_hour_cost"]
    if gpu:
        usd += r[f"gpu_hour_cost_{gpu.lower()}"]
    return usd
