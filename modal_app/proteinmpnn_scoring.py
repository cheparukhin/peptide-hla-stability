"""Stage 5 stretch arm: ProteinMPNN inverse-folding scores, next to the folds.

**NOT LAUNCHED.** This file exists so the cost can be forecast and the plan
reviewed before anything runs. Nothing here has executed against either
workspace. See ``reports/stage5_inverse_folding.md`` for the pilot that
justifies it and for the forecast.

What it computes, per production fold: the conditional log-likelihood of the
**native** peptide sequence given the predicted backbone, with the HLA
ectodomain and beta2m held fixed as visible context. The definition, the
scoring-not-sampling guarantee and the per-position breakdown all live in
``pepstab/inverse_folding.py``; this module only arranges for that code to run
on the Volume instead of on a laptop.

**CPU only, deliberately.** ProteinMPNN is a 1.7 M-parameter message-passing
network and does not want a GPU. More to the point, the production fold is
using up to 10 concurrent A10Gs per workspace; a ``gpu=`` function declared in
this app would contend for those slots. No function here declares one.

**Discovery is shared, not reimplemented.** ``pepstab.structural_features``
owns ``discover_folds`` / ``is_excluded``, which skip the ``_smoke``,
``_shards`` and ``_failed`` trees that production writes next to the cohort
output. Ingesting one of those raises nothing and produces a row that looks
normal, so there is exactly one implementation of that rule and both this app
and the stage 4c.5 extractor use it.

**One profile at a time.** The workspaces hold disjoint, pair-by-pair
interleaved halves of ``data/structural_cohort.csv`` (14,083 each), and
``--profile`` must match ``MODAL_PROFILE`` -- the project invariant, asserted
below. Concatenation is a separate local step that asserts the row count,
because nothing about a half-sized table looks wrong on its own.

    # 5 folds end to end, before any full pass (project invariant)
    MODAL_PROFILE=a-cheparukhin modal run modal_app/proteinmpnn_scoring.py::smoke \
        --profile a-cheparukhin

    # forecast only; spawns nothing
    modal run modal_app/proteinmpnn_scoring.py::forecast

    # one workspace's half
    MODAL_PROFILE=a-cheparukhin modal run --detach \
        modal_app/proteinmpnn_scoring.py::score --profile a-cheparukhin
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
RUN_ID = "ectodomain-20261004"
PROFILES = ("a-cheparukhin", "colleague")
PAIRS_PER_PROFILE = 14_083
COHORT_PAIRS = 28_166

OUT_DIR = Path("/structures")
# Mirrors modal_app/ectodomain_production.py: OUT_ROOT / run_id / boltz2 / production
PRODUCTION_ROOT = OUT_DIR / "stage4c" / RUN_ID / "boltz2" / "production"

CHECKPOINT = "v_48_020.pt"
MPNN_DIR = REPO / "external" / "ProteinMPNN"

# Measured on the stage 4c pilot's 15 Boltz-2 arm-B folds (383 residues,
# n_orders=16, 2 torch threads): mean 14.70 s, median 14.18 s per fold.
# See reports/stage5_inverse_folding.md. This is the number the forecast uses.
STEADY_S_PER_FOLD = 14.70
N_ORDERS = 16
WORKER_CPU = 2.0
WORKER_MEMORY = 4096  # MiB; peak RSS measured at ~1.2 GiB with batch_rows=5
BATCH_ROWS = 5
# Container count buys WALL TIME ONLY. Core-hours are fixed by the job, so
# cost is unchanged by these two numbers (up to per-container startup, which
# the forecast below charges for explicitly). Raised from CHUNK=200 /
# MAX_CONTAINERS=20 purely so the QC sample finishes in minutes rather than
# ~50 min. The experiment is untouched: same folds, same seed, same orders.
#
# CHUNK is the binding constraint, not MAX_CONTAINERS. At CHUNK=200 a
# 2,000-fold sample is only 10 chunks, so it could never use more than 10
# containers however high the cap went.
CHUNK = 25                 # 2,000 folds -> 80 chunks
MAX_CONTAINERS = 100       # not independently verified against the workspace
                           # CPU quota; Modal queues rather than failing if the
                           # real cap is lower, and the run reports what it got.
CONTAINER_STARTUP_S = 30.0  # measured: ~30 s to pull the image and import

# Rates metered from this workspace, read from the repo rather than from a
# pricing page: reports/ectodomain_rates.json, the same source the stage 4c
# forecast used. The project has already been bitten by a published-rate
# estimate that was off by 4.6x, so published numbers are not trusted here.
# (For the record, on CPU and memory they agreed to within 0.3% this time.)
# Read LAZILY, never at module scope. Modal imports this module inside every
# container, where the repo does not exist -- a module-scope read of a repo
# file makes the container die on import before it executes a line of its own,
# which looks like "0 tasks" and nothing else. This is the same defect stage 4c
# hit with `download_weights`, and neither ::forecast nor --dry-run can catch
# it, because both run locally where the file is present.
def _rates() -> tuple[float, float]:
    d = json.loads((REPO / "reports" / "ectodomain_rates.json").read_text())
    return float(d["cpu_hour_cost"]), float(d["mem_gib_hour_cost"])

# ---------------------------------------------------------------------------
# PREDECLARED QC SAMPLE -- fixed before any production score was looked at.
# ---------------------------------------------------------------------------
# Purpose: produce a DISTRIBUTION of pep_ll_mean over the cohort and a triage
# list of folds worth inspecting. This is NOT a pose-failure rate and must
# never be reported as one: the reference points below come from exactly ONE
# badly-folded complex (B*07:02/IPRRNVATL, arms A and C) out of the five in the
# pilot, all of them training rows.
#
# Draw: simple random sample without replacement over the *sorted* fold paths
# of ONE COMPLETE half. Sorting makes the draw reproducible; requiring a
# complete half is what makes it unbiased, because shards are ordered by
# allele, so any sample taken mid-run covers only the alleles folded so far.
QC_SAMPLE_N = 2000
QC_SAMPLE_SEED = 20261004
QC_SAMPLE_ORDERS = 16  # matches the pilot, so the thresholds below transfer

# Reference points, NOT thresholds. From reports/stage5_inverse_folding.md,
# measured on the 45 Boltz-2 pilot folds. The 6 folds with peptide heavy RMSD
# > 2.0 A are all the SAME complex in arms A and C; they spanned pep_ll_mean
# -3.180..-2.984 against the 39 good folds' -2.513..-1.881 (a 0.471
# nats/residue gap). Quoted here in pep_ll_total = pep_ll_mean * 9, the units
# score_folder emits. Note pep_ll_total = 9 * pep_ll_mean = -9 * mpnn_score:
# one quantity, three units.
QC_BAD_MAX = -26.858       # = pep_ll_mean -2.984, best-scoring known-bad fold
QC_GAP_MIDPOINT = -24.740  # = pep_ll_mean -2.749, midpoint of the gap
# Neither may be used to drop a row from the cohort, and neither is calibrated:
# they describe one complex.


def qc_sample(folders: list[str], n: int = QC_SAMPLE_N,
              seed: int = QC_SAMPLE_SEED) -> list[str]:
    """The predeclared draw. Deterministic given the fold list and the seed."""
    import numpy as np

    pool = sorted(folders)
    if len(pool) <= n:
        return pool
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in sorted(idx)]

out_vol = modal.Volume.from_name("pepstab-structures", create_if_missing=False)

app = modal.App("pepstab-stage5-proteinmpnn")

# CPU-only torch. The repo layout is reproduced under /root so that
# pepstab/inverse_folding.py resolves scripts/boltz_pose_check.py and the
# ProteinMPNN checkpoint exactly as it does locally -- the pilot and the cloud
# run must not drift apart.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.5.3",
        "biotite==1.7.1",
        "torch==2.14.1",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .add_local_file(REPO / "pepstab" / "inverse_folding.py",
                    "/root/inverse_folding.py", copy=True)
    .add_local_file(REPO / "pepstab" / "structural_features.py",
                    "/root/structural_features.py", copy=True)
    .add_local_file(REPO / "scripts" / "boltz_pose_check.py",
                    "/root/scripts/boltz_pose_check.py", copy=True)
    .add_local_file(MPNN_DIR / "protein_mpnn_utils.py",
                    "/root/external/ProteinMPNN/protein_mpnn_utils.py", copy=True)
    .add_local_file(MPNN_DIR / "vanilla_model_weights" / CHECKPOINT,
                    f"/root/external/ProteinMPNN/vanilla_model_weights/{CHECKPOINT}",
                    copy=True)
    .env({
        "PROTEINMPNN_DIR": "/root/external/ProteinMPNN",
        "PEPSTAB_SCRIPTS_DIR": "/root/scripts",
        "OMP_NUM_THREADS": str(int(WORKER_CPU)),
        "MKL_NUM_THREADS": str(int(WORKER_CPU)),
    })
)


def check_profile(profile: str) -> str:
    """Refuse to run if the cohort half and the active Modal profile disagree.

    The halves are disjoint, so a mismatch scores one twice and the other
    never. Same assertion, same reason, as ectodomain_production.py.
    """
    active = os.environ.get("MODAL_PROFILE")
    assert profile in PROFILES, f"--profile must be one of {PROFILES}, got {profile!r}"
    assert active in (None, profile), (
        f"--profile {profile!r} but MODAL_PROFILE={active!r}. "
        f"Each profile holds its own half of the cohort; set both the same."
    )
    if active is None:
        print(
            f"warning: MODAL_PROFILE is unset, so this runs against your active "
            f"profile. Scoring the {profile!r} half -- confirm that is the same "
            f"workspace."
        )
    return profile


def forecast_usd(n_folds: int, n_orders: int = N_ORDERS) -> dict:
    """Cost and wall forecast from the pilot's measured per-fold time.

    Scoring cost is linear in ``n_orders``: the pilot measurement is at 16.
    """
    per_fold = STEADY_S_PER_FOLD * (n_orders / N_ORDERS)
    container_s = n_folds * per_fold
    n_chunks = max(1, -(-n_folds // CHUNK))
    container_s += n_chunks * CONTAINER_STARTUP_S
    cpu_hour_usd, gib_hour_usd = _rates()
    core_hours = container_s * WORKER_CPU / 3600
    gib_hours = container_s * (WORKER_MEMORY / 1024) / 3600
    usd = core_hours * cpu_hour_usd + gib_hours * gib_hour_usd
    wall_h = container_s / max(1, min(MAX_CONTAINERS, n_chunks)) / 3600
    return {
        "n_folds": n_folds,
        "n_orders": n_orders,
        "s_per_fold": round(per_fold, 2),
        "chunks": n_chunks,
        "core_hours": round(core_hours, 1),
        "wall_hours_at_max_containers": round(wall_h, 2),
        "usd": round(usd, 2),
        "rates_source": "reports/ectodomain_rates.json (metered)",
        "s_per_fold_source": "stage 4c pilot, measured",
    }


# --------------------------------------------------------------------------
# remote functions
# --------------------------------------------------------------------------
@app.function(
    image=image, volumes={OUT_DIR: out_vol},
    cpu=0.5, memory=2048, timeout=20 * 60, max_containers=1, retries=0,
)
def list_folds(root: str = str(PRODUCTION_ROOT)) -> list[str]:
    """Cohort fold directories, with the harness siblings skipped."""
    import sys

    sys.path.insert(0, "/root")
    import structural_features as sf

    out_vol.reload()
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(f"{base} does not exist on the Volume")
    folders = [str(p) for p in sf.discover_folds(base)]
    skipped = sum(
        1 for p in base.rglob("metadata.json") if sf.is_excluded(p.parent, base)
    )
    print(f"{len(folders)} cohort folds; {skipped} skipped in {sorted(sf.EXCLUDED_DIR_NAMES)}")
    return folders


@app.function(
    image=image, volumes={OUT_DIR: out_vol},
    cpu=WORKER_CPU, memory=WORKER_MEMORY, timeout=60 * 60,
    max_containers=MAX_CONTAINERS, retries=1,
)
def score_chunk(folders: list[str], n_orders: int = N_ORDERS) -> list[dict]:
    """ProteinMPNN rows for one chunk. A failed fold becomes a recorded row."""
    import sys

    sys.path.insert(0, "/root")
    import torch

    import inverse_folding as inv

    torch.set_num_threads(int(WORKER_CPU))
    started = time.time()
    rows = []
    for f in folders:
        t0 = time.time()
        try:
            row = inv.score_folder(
                Path(f), n_orders=n_orders, checkpoint=CHECKPOINT,
                batch_rows=BATCH_ROWS,
            )
        except Exception as exc:
            row = {"folder": f, "status": f"{type(exc).__name__}: {exc}"}
        row["score_s"] = round(time.time() - t0, 2)
        rows.append(row)
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"{len(rows)} folds, {ok} ok, {len(rows) - ok} failed, "
          f"{time.time() - started:.1f}s")
    return rows


# --------------------------------------------------------------------------
# local entrypoints
# --------------------------------------------------------------------------
def _write(rows: list[dict], profile: str, label: str) -> Path:
    out = REPO / "reports" / f"stage5_inverse_folding_{label}_{profile}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
    out.write_text(buf.getvalue())
    return out


@app.local_entrypoint()
def forecast(profile: str = PROFILES[0], n_orders: int = N_ORDERS):
    """Print the cost forecast, and prove the app can deploy to this profile.

    Spawns no container. Deploying does *create* the functions in the target
    workspace, which is the point: a workspace that cannot declare a function
    fails here, cheaply, rather than at the start of the real run. Takes
    ``--profile`` so the MODAL_PROFILE invariant is asserted on the way.
    """
    check_profile(profile)
    half = forecast_usd(PAIRS_PER_PROFILE, n_orders)
    both = forecast_usd(COHORT_PAIRS, n_orders)
    print(json.dumps({"profile": profile, "per_profile": half, "both_profiles": both,
                      "basis": f"pilot-measured {STEADY_S_PER_FOLD}s/fold at "
                               f"n_orders={N_ORDERS}, cpu={WORKER_CPU}",
                      "rates": {
                          "cpu_usd_per_core_hour": _rates()[0],
                          "mem_usd_per_gib_hour": _rates()[1],
                          "source": "reports/ectodomain_rates.json, metered "
                                    "from this workspace -- not a pricing page",
                      },
                      "measured": "seconds per fold, and therefore core-hours",
                      "derived": "USD = measured seconds x metered rates",
                      "qc_sample": forecast_usd(QC_SAMPLE_N, QC_SAMPLE_ORDERS),
                      }, indent=2))


@app.local_entrypoint()
def smoke(profile: str = PROFILES[0], root: str = str(PRODUCTION_ROOT), n: int = 5):
    """End-to-end on 5 real production folds. Required before any full pass."""
    check_profile(profile)
    folders = list_folds.remote(root)
    if not folders:
        raise SystemExit(f"no cohort folds under {root}; has production written any?")
    sel = folders[:n]
    print(f"smoke on {len(sel)} of {len(folders)} folds")
    rows = score_chunk.remote(sel)
    for r in rows:
        print(json.dumps({k: r.get(k) for k in
                          ("allele", "peptide", "arm", "seed", "status",
                           "pep_ll_total", "score_s")}))
    bad = [r for r in rows if r.get("status") != "ok"]
    out = _write(rows, profile, "smoke")
    print(f"wrote {out}; {len(bad)} failed")
    if bad:
        raise SystemExit(1)


@app.local_entrypoint()
def sample(
    profile: str = PROFILES[0],
    root: str = str(PRODUCTION_ROOT),
    n: int = QC_SAMPLE_N,
    seed: int = QC_SAMPLE_SEED,
    n_orders: int = QC_SAMPLE_ORDERS,
    allow_partial: bool = False,
):
    """The predeclared QC sample: a distribution and a triage list.

    Refuses a partial half unless ``--allow-partial`` is passed, because
    shards are written in allele order and a mid-run sample covers only the
    alleles folded so far. This is NOT a failure-rate measurement: the
    reference points describe one badly-folded complex in the pilot.
    """
    check_profile(profile)
    folders = list_folds.remote(root)
    if len(folders) != PAIRS_PER_PROFILE:
        msg = (f"{len(folders)} folds for profile {profile!r}, expected "
               f"{PAIRS_PER_PROFILE}: the half is not complete, and shards are "
               f"written in allele order, so this sample would be allele-biased.")
        if not allow_partial:
            raise SystemExit(msg)
        print(f"warning: {msg}")
    sel = qc_sample(folders, n=n, seed=seed)
    print(json.dumps({"profile": profile, "pool": len(folders), "sampled": len(sel),
                      "seed": seed, "n_orders": n_orders,
                      "forecast": forecast_usd(len(sel), n_orders)}, indent=2))
    chunks = [sel[i:i + CHUNK] for i in range(0, len(sel), CHUNK)]
    started = time.time()
    rows: list[dict] = []
    for got in score_chunk.starmap((c, n_orders) for c in chunks):
        rows.extend(got)
        print(f"# {len(rows)}/{len(sel)}  {time.time() - started:.0f}s", flush=True)
    out = _write(rows, profile, "qcsample")
    ok = sum(1 for r in rows if r.get("status") == "ok")
    secs = sum(r.get("score_s", 0.0) for r in rows)
    print(json.dumps({
        "wrote": str(out), "rows": len(rows), "ok": ok, "failed": len(rows) - ok,
        "wall_s": round(time.time() - started, 1),
        "measured_s_per_fold": round(secs / max(1, len(rows)), 2),
        "forecast_s_per_fold": STEADY_S_PER_FOLD * (n_orders / N_ORDERS),
    }, indent=2))


@app.local_entrypoint()
def score(
    profile: str = PROFILES[0],
    root: str = str(PRODUCTION_ROOT),
    n_orders: int = N_ORDERS,
    limit: int = 0,
    dry_run: bool = False,
):
    """Score one workspace's half. ``--dry-run`` prints the plan and stops.

    ``--dry-run`` is **purely local**: it forecasts against the frozen cohort
    size and never calls ``list_folds``, which would spawn a container (and
    would fail outright before production has written anything). Creating the
    app's functions is all that touching the workspace amounts to here, which
    is the same free check stage 4c used.
    """
    check_profile(profile)
    if dry_run:
        n = limit or PAIRS_PER_PROFILE
        print(json.dumps(forecast_usd(n, n_orders), indent=2))
        print(f"dry run: nothing spawned; forecast assumes the frozen "
              f"{n} folds for profile {profile!r}, not a Volume listing")
        return
    folders = list_folds.remote(root)
    if limit:
        folders = folders[:limit]
    print(json.dumps(forecast_usd(len(folders), n_orders), indent=2))
    if len(folders) != PAIRS_PER_PROFILE and not limit:
        print(f"warning: {len(folders)} folds, expected {PAIRS_PER_PROFILE} for "
              f"one profile. Production may be incomplete -- a partial half is "
              f"still unbiased (the cohort is interleaved pair-by-pair), but "
              f"say so in the write-up.")
    chunks = [folders[i:i + CHUNK] for i in range(0, len(folders), CHUNK)]
    started = time.time()
    rows: list[dict] = []
    for got in score_chunk.starmap((c, n_orders) for c in chunks):
        rows.extend(got)
        print(f"# {len(rows)}/{len(folders)}  {time.time() - started:.0f}s", flush=True)
    out = _write(rows, profile, "production")
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"wrote {out}: {len(rows)} rows, {ok} ok, {len(rows) - ok} failed, "
          f"{time.time() - started:.0f}s")
