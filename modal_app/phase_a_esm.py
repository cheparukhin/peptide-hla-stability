"""Phase A A5/ESM: the frozen ESM-2 arm, one container per fitting partition.

The science is the same loop ``scripts/phase_a_esm_oof.py`` pilots locally; this
file arranges for it to run on 30 containers instead of one laptop, and is the
single home of the worker-side code so the two cannot drift. The local pilot
imports :func:`fit_esm_ensemble` and :func:`assemble_from_payload` from **here**,
which means ``--quick`` exercises the exact functions the batch runs.

    # free: imports no modal, starts nothing
    .venv/bin/python modal_app/phase_a_esm.py --self-check

    # the 146 MB training-only embedding artifact, then onto the Volume
    .venv/bin/python scripts/phase_a_esm_oof.py --write-payload /tmp/esm_payload.npz
    MODAL_PROFILE=a-cheparukhin modal volume put pepstab-phasea-esm-payload \
        /tmp/esm_payload.npz phase_a_esm_payload.npz

    # 3 units end to end, measuring per-unit time (project invariant)
    MODAL_PROFILE=a-cheparukhin modal run modal_app/phase_a_esm.py::smoke \
        --profile a-cheparukhin

    # forecast from the measured figure, then the batch
    MODAL_PROFILE=a-cheparukhin modal run modal_app/phase_a_esm.py::forecast \
        --profile a-cheparukhin \
        --measured-from reports/further_research/phase_a_esm_smoke_a-cheparukhin.json
    MODAL_PROFILE=a-cheparukhin modal run --detach modal_app/phase_a_esm.py::run \
        --profile a-cheparukhin \
        --measured-from reports/further_research/phase_a_esm_smoke_a-cheparukhin.json

**Why this is a separate app from ``phase_a_experiments.py``.** That app's unit
is one *fit*; this arm's unit is one *fitting partition* — 2 feature assemblies
and 30 networks — because the PCA is refitted per partition and a per-network
unit would multiply feature assembly by 30 for nothing. That app also documents
"No Volumes, no Secrets" with tests asserting it by reading decorators, and this
arm needs a 146 MB read-only Volume. Those are not tweaks to a tested app; they
are a different app sharing a protocol, and coupling the A1-A4 screen's
correctness to this arm's requirements would benefit neither.

**Arity.** 5 inner folds x 2 ESM-2 layers (``mid``, ``final``) x 3 seeds = 30
members per outer fold, which is A0's arity term for term — the layer axis is
the two-way axis stage 3 itself declared as the stand-in for stage 2's
{one-hot, BLOSUM62} (``scripts/esm_arm.py::mode_ensemble --axis layer``).
Ensembling is worth roughly +0.07 to +0.09 median Spearman here, larger than
most effects Phase A looks for, so an arity mismatch between A5's two blend
components would land as the blend's main effect. Two nesting levels, 5 + 25
units, **900 networks** — the same count A0 runs.

**Two properties of the payload artifact, both stronger than an assertion.**

1. It holds the **training sequences only** — 3,942 of the cache's 5,633
   peptides, and 75 HLA domains pre-sliced to their 34 contact positions. A
   container therefore *cannot* read a frozen validation or test sequence. This
   arm's one unrecoverable mistake is converted from "avoided by discipline"
   into "impossible by construction".
2. It is **bit-identical** to the local cache path, not merely close.
   ``build_payload`` preserves the cache's row order so the standardisation sums
   in the same order, and ``randomized_svd``'s projection is drawn over the
   4,320 columns, which are unchanged — so a 3,942-row payload gives the *same*
   transform as the 5,633-row cache. Verified with ``assert_array_equal`` on
   both layers and ``atol=0`` on predictions and ``inner_oof``
   (``tests/test_phase_a_esm_oof.py``). That is what makes the cloud run and the
   pilot comparable rather than similar.

Digest of the artifact this run was priced against: ``6f5200968670ea7d``.

**The client is assumed to die.** The A1-A4 screen lost its gRPC connection
mid-``.map()`` twice — ``StreamTerminatedError``, then ``AttributeError:
'Connection' object has no attribute '_transport'`` — and because rows
accumulate on the *client*, one run captured 8 of 3,360 units. ``--detach``
does not help: it keeps containers alive, but nothing is collecting their
results. So every unit's **full return**, predictions included, is journalled
with ``write``/``flush``/``os.fsync`` the moment it arrives, and ``--resume``
reconstructs the whole run from the journal alone without refitting anything
already recorded. 30 units at ~1.4 MB of predictions each is little comfort if
the client dies at unit 29.

CPU only: no function here declares ``gpu=``, which
``tests/test_phase_a_esm_modal.py`` asserts by reading the decorator. Per
``modal_app/boltz_common.py:3-8`` Modal validates every function in an app at
creation time, so one ``gpu=`` would make this app undeployable in a workspace
without GPU access and take the CPU work with it.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Same two profiles as every other app here; ``colleague`` points at the
#: ``sofyaleyn`` workspace. This arm does not split work between them -- the
#: Volume and the unit set are one workspace's -- but the guard is still
#: required, because the forecast is priced against one workspace's metered
#: rates and one workspace's reserved credit.
PROFILES = ("a-cheparukhin", "colleague")

#: A unit is 2 feature assemblies and 30 networks, all single-model: this
#: project's MLP is numpy/BLAS. So the shape that matters is many containers
#: running one partition each, not one wide container. 2.0 cores is the
#: smallest reservation that lets BLAS use a second thread without the
#: container thrashing against its own cgroup limit.
#:
#: memory: the local pilot's widest step is the peptide block -- 3,942 x 4,320
#: float32 standardised in place (68 MB), plus ``randomized_svd``'s float64
#: working copy of the fitting partition's rows (~155 MB) -- against a 20 MB
#: feature matrix. 4 GiB is a guess with headroom and ``::smoke`` writes the
#: measured cgroup peak so the guess can be replaced rather than inherited.
WORKER_CPU = 2.0
WORKER_MEMORY = 4096  # MiB

#: Pinned into the image ``.env`` and recorded on every row. A worker cannot
#: call a thread-pinning helper late enough to matter -- the BLAS pool is sized
#: when numpy is imported, and importing the science layer imports numpy -- so
#: the image environment, already in place when the interpreter starts, is the
#: only point early enough. Thread count moves a single network by ~0.02 median
#: Spearman at a fixed seed (prespecification §5), so this is part of the run
#: record, not a performance knob.
WORKER_THREADS = int(WORKER_CPU)

#: Container count buys WALL TIME ONLY: the unit-seconds are fixed by the
#: experiment, so cost is unchanged by this up to the per-container startup the
#: forecast charges for explicitly.
MAX_CONTAINERS = 30
DEFAULT_CHUNK = 1

#: 45 s until ``::smoke`` measures it. Stage 5 measured ~30 s to pull a CPU
#: image and import it; this one adds pandas and scikit-learn but **not** torch
#: or catboost, neither of which this arm uses.
CONTAINER_STARTUP_S = 45.0

#: 1.25, the same margin stages 4c and 8 forecast with.
COST_MARGIN = 1.25

#: **A local per-unit time is not a Modal per-unit time, in either direction.**
#: Two opposite ratios have been measured on this project within an hour. The
#: A1-A4 screen first reported Modal ~4.2x *slower* per unit than this laptop
#: (48.5 s against 11.5 s); Phase B then measured Modal *faster* per network
#: (6.60 s against ~15 s locally), and A1 units arrived at 4.12 s against a
#: 10.8 s local figure. The reconciliation is that the local numbers were taken
#: while four workstreams shared 8 cores, so they were measuring contention
#: rather than hardware -- and contention is not a stable quantity to scale by.
#:
#: Which is exactly why no ratio is applied here. The local pilot's timings are
#: a check on *correctness*, never an input to a price: ``::forecast`` refuses
#: to run without a figure ``::smoke`` measured on Modal, and ``self_check``
#: labels its single placeholder as a placeholder. This is the substitution the
#: project's "no assumed per-unit time" rule exists to catch, and it has now
#: been attempted twice with opposite signs.
LOCAL_PILOT_SECONDS_PER_UNIT = 215.0

#: Metered rates, read from the repo rather than a pricing page: an unmeasured
#: rate assumption cost this project a 4.6x error once already.
RATES_PATH = REPO / "reports" / "ectodomain_rates.json"

OUT_DIR = REPO / "reports" / "further_research"

#: Where the Volume is mounted, and the artifact's name inside it.
PAYLOAD_MOUNT = "/payload"
PAYLOAD_NAME = "phase_a_esm_payload.npz"
PAYLOAD_VOLUME = "pepstab-phasea-esm-payload"

#: Bumped whenever :func:`build_payload`'s layout changes, so a container
#: cannot be handed an artifact whose shape its reader predates.
PAYLOAD_SCHEMA_VERSION = 1

#: The two-way member axis, standing in for A0's {one-hot, BLOSUM62}.
LAYERS = ("mid", "final")

#: Stage 3's selected representations. Per-position peptide embeddings and the
#: 34 HLA contact positions; ``reconcile_selection`` checks them against
#: ``reports/stage3_headline.json`` rather than trusting this copy.
PEP_REP = "pos"
HLA_REP = "contact"

#: ``sub_fold`` for a member of the second nesting level, which has no
#: sub-fold. A0 writes the same sentinel, so the two runs tables read alike.
NO_SUB_FOLD = -1

#: Prespecification §2's declared sensitivity panel, reported next to the
#: primary >= 20-row panel so neither bar can be chosen after the fact.
SENSITIVITY_MIN_ROWS = 50

#: Expected training rows, from docs/README.md.
N_TRAIN_ROWS = 19_716

#: **The giveaway for a pooled fit is a row count, not a metric.** Phase B
#: fitted one calibration map on pooled ``inner_oof_30`` rows across all five
#: outer folds and applied it to pooled outer-held predictions; since a row is
#: outer-held in exactly one fold and inner-OOF in the other four, that fit
#: used every row's own label to build the map that then predicted it. The
#: leaked and correct versions agreed on Brier to four decimals -- what exposed
#: it was a log line saying "fit on 19,716 rows" where a per-fold fit can only
#: see about four fifths of that. A leak that moves no metric is the kind that
#: ships, so the count is checked directly.
#:
#: It is checked as a **partition identity** -- fit rows plus predicted rows
#: equal the whole -- and not against an expected constant. A constant was
#: tried first, at 15,772, and failed four of the five level-2 units: the folds
#: are water-filled by cluster and come out 3,944 / 3,943 x 4, so their
#: complements are 15,772 and 15,773 x 4. The identity is both correct for
#: every fold and strictly stronger, since it also catches a dropped row.
N_HELD_ROWS_PER_OUTER_FOLD = (3944, 3943, 3943, 3943, 3943)

#: Bumped when the unit dict's keys change, and recorded on every row so a
#: journal cannot be resumed by a runner that means something else by them.
UNITS_SCHEMA_VERSION = 1

#: What a unit must return non-null for ``::smoke`` to pass. The coverage keys
#: are here and not only the correlation because prespecification §2 requires
#: coverage -- eligible alleles and the share of fold rows they cover -- next to
#: every score, and a score whose coverage is missing cannot be read against
#: that rule.
REQUIRED_METRIC_KEYS = (
    "median_per_allele_spearman",
    "n_alleles_eligible",
    "eligible_row_coverage",
)


def load_rates() -> dict[str, float]:
    """Metered CPU and memory rates. Read **lazily**, never at module scope.

    Modal imports this module inside every container, where the repo does not
    exist. A module-scope read of a repo file kills the container on import
    before it executes a line of its own, which surfaces as "0 tasks" and
    nothing else -- the defect documented at
    ``modal_app/proteinmpnn_scoring.py:97-105``, which neither a local forecast
    nor ``--dry-run`` can catch, because both run where the file is present.
    """
    raw = json.loads(RATES_PATH.read_text())
    return {
        "cpu_hour": float(raw["cpu_hour_cost"]),
        "mem_gib_hour": float(raw["mem_gib_hour_cost"]),
        "volume_gib_month": float(raw["volume_storage_gib_month_cost"]),
    }


def check_profile(profile: str) -> str:
    """Refuse to run if the requested profile and the active one disagree.

    The house assertion, same form as ``modal_app/foldx_scoring.py``. What it
    protects here is the budget and the Volume: the forecast is priced against
    one workspace's metered rates, and the payload Volume exists in one
    workspace, so a run that silently lands in the other both spends the wrong
    credit and finds no artifact.
    """
    active = os.environ.get("MODAL_PROFILE")
    assert profile in PROFILES, f"--profile must be one of {PROFILES}, got {profile!r}"
    assert active in (None, profile), (
        f"--profile {profile!r} but MODAL_PROFILE={active!r}. The forecast is "
        f"priced against one workspace's metered rates and reserved credit, and "
        f"the payload Volume lives in one workspace; set both the same."
    )
    if active is None:
        print(
            f"warning: MODAL_PROFILE is unset, so this runs against your active "
            f"profile. Billing the {profile!r} workspace -- confirm that is the "
            f"same one."
        )
    return profile


# --------------------------------------------------------------------------
# cost model
# --------------------------------------------------------------------------


def forecast_numbers(
    n_units: int,
    seconds_per_unit: float,
    cpu: float = WORKER_CPU,
    memory_gib: float = WORKER_MEMORY / 1024,
    containers: int = MAX_CONTAINERS,
    chunk: int = DEFAULT_CHUNK,
    startup_s: float = CONTAINER_STARTUP_S,
    margin: float = COST_MARGIN,
) -> dict:
    """Container-hours and dollars at the repo's metered rates.

    Same model as ``modal_app/phase_a_experiments.py`` and
    ``modal_app/foldx_scoring.py:145-192``, not re-derived: Modal bills the
    cores and memory a container **reserves**, for as long as it lives, not the
    cores it keeps busy. So the cost is ``containers x cpu x wall``, and a
    container that is not full is a real expense rather than something the
    arithmetic hides.

    This arm's units are not equal in size -- a level-2 unit fits on ~12,618
    rows and a level-3 unit on ~10,094 -- so ``seconds_per_unit`` is the slowest
    unit's time, which is what sets the wall of the container that finishes
    last and therefore the bill.
    """
    assert chunk >= 1, f"--chunk must be at least 1 unit, got {chunk}"
    assert containers >= 1, f"--containers must be at least 1, got {containers}"
    assert n_units >= 1, f"nothing to forecast: {n_units} units"

    rates = load_rates()
    n_chunks = math.ceil(n_units / chunk)
    chunk_fill = n_units / (n_chunks * chunk)
    active = min(containers, n_chunks)
    container_fill = active / containers
    chunks_per_container = math.ceil(n_chunks / active)
    busy_s = chunks_per_container * chunk * seconds_per_unit
    wall_s = busy_s + startup_s
    container_hours = active * wall_s / 3600
    core_hours = container_hours * cpu
    mem_gib_hours = container_hours * memory_gib
    usd = core_hours * rates["cpu_hour"] + mem_gib_hours * rates["mem_gib_hour"]
    # The Volume is a standing monthly charge, not part of the run, so it is
    # reported beside the run cost rather than folded into it.
    volume_usd_month = (146.4 / 1024) * rates["volume_gib_month"]
    return {
        "n_units": n_units,
        "seconds_per_unit": seconds_per_unit,
        "cpu_per_container": cpu,
        "memory_gib_per_container": memory_gib,
        "threads": WORKER_THREADS,
        "containers_requested": containers,
        "containers_active": active,
        "container_fill": container_fill,
        "chunk": chunk,
        "n_chunks": n_chunks,
        "chunk_fill": chunk_fill,
        "chunks_per_container": chunks_per_container,
        "startup_s": startup_s,
        "wall_h": wall_s / 3600,
        "container_hours": container_hours,
        "core_hours": core_hours,
        "mem_gib_hours": mem_gib_hours,
        "cpu_hour_rate_usd": rates["cpu_hour"],
        "mem_gib_hour_rate_usd": rates["mem_gib_hour"],
        "usd": usd,
        "usd_with_margin": usd * margin,
        "margin": margin,
        "volume_usd_per_month": volume_usd_month,
        "rates_source": str(RATES_PATH.relative_to(REPO)),
    }


def print_forecast(f: dict, title: str, measured: bool = False) -> None:
    print(f"\n{title}")
    print(f"  units               {f['n_units']:,} in {f['n_chunks']:,} chunks of "
          f"{f['chunk']} (fill {f['chunk_fill']:.0%})")
    print(f"  per unit            {f['seconds_per_unit']:.1f}s "
          f"{'(MEASURED by ::smoke)' if measured else '(PLACEHOLDER -- not measured)'}")
    print(f"  containers          {f['containers_active']} active of "
          f"{f['containers_requested']} requested x {f['cpu_per_container']:g} "
          f"cores / {f['memory_gib_per_container']:g} GiB, "
          f"{f['threads']} BLAS threads")
    print(f"  wall                {f['wall_h'] * 60:.1f} min "
          f"(incl. {f['startup_s']:.0f}s startup per container)")
    print(f"  billed              {f['core_hours']:.3f} core-h + "
          f"{f['mem_gib_hours']:.3f} GiB-h")
    print(f"  cost                ${f['usd']:.4f} -> "
          f"${f['usd_with_margin']:.4f} at {f['margin']}x margin "
          f"[{f['rates_source']}]")
    print(f"  volume              ${f['volume_usd_per_month']:.4f}/month standing, "
          f"deletable after the run")


def seconds_from_smoke(path: Path) -> float:
    """The per-unit time ``::smoke`` measured, or refuse to guess one."""
    summary = json.loads(path.read_text())
    seconds = summary.get("seconds_per_unit")
    if not seconds:
        raise SystemExit(f"{path} carries no seconds_per_unit; rerun ::smoke")
    if not summary.get("all_ok"):
        raise SystemExit(f"{path} records a failed smoke; do not launch the batch")
    return float(seconds)


# --------------------------------------------------------------------------
# payload: the embedding artifact, and features built from it
# --------------------------------------------------------------------------


def compute_payload_digest(blocks: dict) -> str:
    """Digest of the embedding bytes themselves, not of the file around them.

    One definition, used when the artifact is built and again when it is read,
    so the two cannot disagree about what they are hashing.
    """
    import hashlib

    import numpy as np

    digest = hashlib.sha256()
    for name in sorted(blocks):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(blocks[name]).tobytes())
    return digest.hexdigest()


def read_payload(path: Path) -> dict:
    """Load a payload written by ``scripts/phase_a_esm_oof.py --write-payload``.

    The digest is **recomputed from the loaded bytes** and checked against the
    one recorded at build time. A ``modal volume put`` of this 146 MB artifact
    failed partway through a network outage on this machine, and a half-uploaded
    file that merely *exists* is worse than an absent one: it would give a
    container a different PCA basis and so a quietly different arm. np.load's
    zip CRC catches most truncation, but not a block that happens to survive
    it, and the cost of being sure is ~0.3 s against a ~200 s unit.
    """
    import numpy as np

    with np.load(path) as handle:
        meta = json.loads(str(handle["meta"]))
        index = json.loads(str(handle["index"]))
        blocks = {k: handle[k] for k in handle.files
                  if k not in ("meta", "index")}
    if meta["schema_version"] != PAYLOAD_SCHEMA_VERSION:
        raise AssertionError(
            f"{path} is payload schema {meta['schema_version']}, this reader is "
            f"{PAYLOAD_SCHEMA_VERSION}")
    actual = compute_payload_digest(blocks)
    if actual != meta["digest"]:
        raise AssertionError(
            f"{path} digest is {actual[:16]} but the artifact records "
            f"{meta['digest'][:16]}: the file is corrupt or was uploaded "
            f"partially. Rebuild it and re-put it; do not fit on it.")
    return {**meta, "blocks": blocks, "index": index}


def _fit_block_from_payload(payload: dict, kind: str, layer: str,
                            fit_values, n_pca: int):
    """``scripts.esm_arm._fit_block`` against a payload instead of the cache.

    Line for line the same arithmetic, so the cloud run and the local pilot
    cannot drift: standardise on the *unique sequences of the fitting
    partition*, then PCA fitted on those same rows. A peptide measured on 36
    alleles gets one vote in the mean and variance, not 36.

    The only difference is where the matrix comes from, and that difference is
    provably inert: ``randomized_svd``'s projection is drawn over **columns**
    (4,320 either way), and ``build_payload`` preserves row order, so a payload
    holding 3,942 sequences and the cache holding 5,633 give the same transform
    for every training row -- asserted bit-for-bit, not assumed.
    """
    import numpy as np
    from sklearn.utils.extmath import randomized_svd

    lookup = payload["index"][kind]
    M = payload["blocks"][f"{kind}_{layer}"]
    M = M.reshape(len(M), -1)
    fit_rows = np.unique([lookup[str(v)] for v in np.unique(fit_values)])
    fit = M[fit_rows]

    mu = fit.mean(axis=0)
    sd = fit.std(axis=0)
    sd[sd < 1e-6] = 1.0
    M = (M - mu) / sd
    fit = M[fit_rows]

    explained = 1.0
    if n_pca and n_pca < M.shape[1]:
        centre = fit.mean(axis=0)
        k = min(n_pca, min(fit.shape) - 1)
        _, s, Vt = randomized_svd(fit - centre, n_components=k, random_state=0)
        total = float(((fit - centre) ** 2).sum())
        explained = float((s ** 2).sum() / total) if total else 1.0
        M = (M - centre) @ Vt.T
    return M.astype(np.float32), lookup, explained


def assemble_from_payload(fit_df, predict_dfs: list, payload: dict, layer: str,
                          config: dict):
    """``scripts.esm_arm.assemble`` against a payload instead of the cache.

    Same block order (peptide, then HLA), same transforms, same ``info`` keys.
    ``hla_rep`` is not passed: the contact slice is already applied inside the
    payload, and slicing it twice would silently take 34 positions out of 34.
    """
    import gc

    import numpy as np

    pieces_fit: list = []
    pieces_predict: list[list] = [[] for _ in predict_dfs]
    info: dict = {"blocks": []}

    for kind, column, n_pca in (("peptide", "peptide", config["pep_pca"]),
                                ("hla", "hla_seq", config["hla_pca"])):
        M, lookup, explained = _fit_block_from_payload(
            payload, kind, layer, fit_df[column].to_numpy(), n_pca)
        pieces_fit.append(M[[lookup[str(v)] for v in fit_df[column].to_numpy()]])
        for i, frame in enumerate(predict_dfs):
            pieces_predict[i].append(
                M[[lookup[str(v)] for v in frame[column].to_numpy()]])
        info["blocks"].append({"kind": kind, "n_columns": M.shape[1],
                               "pca_explained": round(explained, 6)})
        info[f"{kind}_explained"] = round(explained, 6)
        del M
        gc.collect()

    X_fit = np.concatenate(pieces_fit, axis=1)
    X_predict = [np.concatenate(p, axis=1) for p in pieces_predict]
    info["n_features"] = int(X_fit.shape[1])
    info["train_mb"] = round(X_fit.nbytes / 1e6, 1)
    return (np.ascontiguousarray(X_fit),
            [np.ascontiguousarray(x) for x in X_predict], info)


# --------------------------------------------------------------------------
# the member loop
# --------------------------------------------------------------------------


def fit_esm_ensemble(fit_df, predict_dfs: list, inner, layers: tuple,
                     seeds: tuple, config: dict, base_config=None,
                     payload: dict | None = None) -> dict:
    """One ``inner x layer x seed`` ensemble, member for member as A0's.

    This is ``pepstab.nested.fit_mlp_ensemble`` with its encoding axis replaced
    by the ESM layer axis. It is mirrored rather than called because that
    function's ``input_set`` routes to ``pepstab.features.build_features``,
    which only knows one-hot and BLOSUM; the loop order, the member record, the
    ``inner_oof`` accumulation and the held-out-count assertion are otherwise
    identical, so the two arms' arity cannot drift.

    With ``payload`` set, features come from the Volume artifact; without it
    they come from ``features/esm/`` through ``scripts.esm_arm.assemble``. The
    two paths are asserted bit-identical, which is what lets the local pilot
    stand as a check on the cloud run.
    """
    import gc

    import numpy as np

    from pepstab import nested

    base_config = nested.REFERENCE_MLP if base_config is None else base_config
    n_inner = int(inner.max()) + 1
    per_inner = len(layers) * len(seeds)
    y_fit = fit_df["y_log1p"].to_numpy()

    members: list[dict] = []
    stacked: list[list] = [[] for _ in predict_dfs]
    inner_sum = np.zeros(len(fit_df))
    inner_count = np.zeros(len(fit_df), dtype=int)
    n_features = 0

    from pepstab.mlp import MLPRegressor

    for layer in layers:
        if payload is None:
            from scripts.esm_arm import assemble

            X_fit, X_predict, info = assemble(
                fit_df, predict_dfs, config["checkpoint"], layer,
                config["pep_rep"], config["hla_rep"], None,
                pep_pca=config["pep_pca"])
        else:
            X_fit, X_predict, info = assemble_from_payload(
                fit_df, predict_dfs, payload, layer, config)
        n_features = info["n_features"]
        for k in range(n_inner):
            is_fit = inner != k
            for seed in seeds:
                cfg = replace(base_config, hidden=tuple(config["hidden"]),
                              l2=config["l2"], seed=seed)
                model = MLPRegressor(cfg).fit(X_fit[is_fit], y_fit[is_fit],
                                              X_fit[~is_fit], y_fit[~is_fit])
                for slot, X in zip(stacked, X_predict):
                    slot.append(model.predict(X))
                inner_sum[~is_fit] += model.predict(X_fit[~is_fit])
                inner_count[~is_fit] += 1
                members.append({"inner_fold": k, "block": layer, "seed": seed,
                                "n_fit_rows": int(is_fit.sum()),
                                "best_epoch": model.best_epoch_,
                                "dev_mse": float(model.dev_mse_),
                                "fit_seconds": round(model.fit_seconds_, 2)})
        del X_fit, X_predict
        gc.collect()

    if not np.all(inner_count == per_inner):
        raise AssertionError(
            f"each fit row should be held out by exactly {per_inner} members "
            f"({len(layers)} layers x {len(seeds)} seeds), got counts "
            f"{sorted(set(inner_count.tolist()))}")
    return {
        "predictions": [np.mean(slot, axis=0) for slot in stacked],
        "inner_oof": inner_sum / per_inner,
        "members": members,
        "n_members": len(members),
        "n_features": n_features,
    }


# --------------------------------------------------------------------------
# units
# --------------------------------------------------------------------------


def enumerate_units(n_inner: int = 0) -> list[dict]:
    """The 30 units: 5 level-2 partitions and 25 level-3 partitions.

    A unit is one *fitting partition*, carrying all 30 of its members, because
    the PCA is refitted per partition: a per-network unit would do the same
    feature assembly 30 times. Level 2 produces ``outer_held`` and
    ``inner_oof_6`` for one outer fold; level 3 produces ``inner_oof_30`` for
    one ``(outer, inner)`` cell, which is the third nesting level A5's blend
    weights need so that fitting on inner predictions and applying to
    ``outer_held`` does not cross an arity boundary.
    """
    from pepstab import nested

    n_inner = n_inner or nested.ASSESS_INNER_FOLDS
    units = [{"key": f"L2_o{f}", "level": 2, "outer_fold": f, "inner_fold": -1,
              "schema": UNITS_SCHEMA_VERSION}
             for f in range(nested.OUTER_FOLDS)]
    units += [{"key": f"L3_o{f}_i{k}", "level": 3, "outer_fold": f,
               "inner_fold": k, "schema": UNITS_SCHEMA_VERSION}
              for f in range(nested.OUTER_FOLDS) for k in range(n_inner)]
    return units


def smoke_units() -> list[dict]:
    """The predeclared 3-unit subset: one of each level, plus a second level 3.

    Named rather than "the first n the enumerator happened to emit", so the
    smoke covers both nesting levels and the per-unit time it measures is the
    *slowest* kind of unit, which is the one the forecast must extrapolate from.
    """
    wanted = ("L2_o0", "L3_o0_i0", "L3_o0_i1")
    by_key = {u["key"]: u for u in enumerate_units()}
    return [by_key[k] for k in wanted]


def load_rows() -> dict:
    """The training rows, read client-side and passed as a function argument.

    House pattern: ``data/*.csv`` is read on the client, never in a container.
    ``hla_seq`` is sent as a 75-entry allele map rather than a column, because
    182 characters x 19,716 rows is 3.6 MB of repeated strings for 75 distinct
    values.

    Training split only. The frozen validation and test rows are not in this
    structure, so a container cannot score one even if it tried.
    """
    from pepstab.data import load_with_splits

    df = load_with_splits()
    train = df[df.split == "train"].sort_values("pair_id").reset_index(drop=True)
    assert len(train) == N_TRAIN_ROWS, (
        f"training split has {len(train):,} rows, expected {N_TRAIN_ROWS:,}")
    allele_hla = (train[["allele", "hla_seq"]].drop_duplicates()
                  .set_index("allele").hla_seq.to_dict())
    assert train.groupby("allele").hla_seq.nunique().max() == 1, (
        "an allele maps to more than one hla_seq; the allele map would lose rows")
    return {
        "pair_id": train.pair_id.astype(int).tolist(),
        "peptide": train.peptide.astype(str).tolist(),
        "allele": train.allele.astype(str).tolist(),
        "cluster_id": train.cluster_id.astype(int).tolist(),
        "y_log1p": train.y_log1p.astype(float).tolist(),
        "allele_hla": allele_hla,
        "n_rows": len(train),
    }


def rows_to_frame(rows: dict):
    """Rebuild the training frame a container was handed."""
    import pandas as pd

    frame = pd.DataFrame({k: rows[k] for k in
                          ("pair_id", "peptide", "allele", "cluster_id",
                           "y_log1p")})
    frame["hla_seq"] = frame.allele.map(rows["allele_hla"])
    assert frame.hla_seq.notna().all(), "an allele is missing from the hla map"
    return frame


def run_unit(unit: dict, rows: dict, config: dict,
             payload_path: str = "") -> dict:
    """One fitting partition: 30 members, their predictions, their records.

    Pure with respect to the repo -- reads only the Volume artifact and its two
    arguments, writes nothing -- so a retry recomputes rather than
    double-counting.

    Returns the predictions themselves, not only a metric. The client journals
    the whole return the moment it arrives, because the client is assumed to
    die and a unit's 30 fits are already paid for by the time its numbers exist.
    """
    import resource

    import numpy as np

    from pepstab import nested

    started = time.time()
    train = rows_to_frame(rows)
    payload = read_payload(Path(payload_path or
                                f"{PAYLOAD_MOUNT}/{PAYLOAD_NAME}"))
    n_inner = config.get("n_inner") or nested.ASSESS_INNER_FOLDS
    n_sub = config.get("n_sub") or nested.ASSESS_INNER_FOLDS
    seeds = tuple(config["seeds"])

    # The fold structure is inherited from pepstab.nested, never recomputed
    # into agreement: the client asserts the outer vector equals
    # reports/further_research/phase_a_outer_folds.csv before launching.
    outer = nested.outer_folds(train)
    f = unit["outer_fold"]
    held_mask = outer == f
    fit_rows = train[~held_mask].reset_index(drop=True)
    inner = nested.inner_folds(fit_rows, n_inner)

    if unit["level"] == 2:
        held_rows = train[held_mask].reset_index(drop=True)
        result = fit_esm_ensemble(fit_rows, [held_rows], inner, LAYERS, seeds,
                                  config, payload=payload)
        predicted = held_rows
        y_pred = result["predictions"][0]
        extra = {"inner_pair_id": fit_rows.pair_id.astype(int).tolist(),
                 "inner_oof_6": [float(v) for v in result["inner_oof"]]}
        members = [{**m, "sub_fold": NO_SUB_FOLD} for m in result["members"]]
        role = "outer_held"
    elif unit["level"] == 3:
        k = unit["inner_fold"]
        sub_fit = fit_rows[inner != k].reset_index(drop=True)
        target = fit_rows[inner == k].reset_index(drop=True)
        sub = nested.inner_folds(sub_fit, n_sub)
        result = fit_esm_ensemble(sub_fit, [target], sub, LAYERS, seeds, config,
                                  payload=payload)
        predicted = target
        y_pred = result["predictions"][0]
        extra = {}
        members = [{**m, "sub_fold": m.pop("inner_fold"), "inner_fold": k}
                   for m in result["members"]]
        role = "inner_oof_30"
    else:
        raise AssertionError(f"unknown level {unit['level']!r}")

    if not np.isfinite(y_pred).all():
        raise AssertionError(f"{unit['key']}: non-finite prediction")

    # Nothing in this unit may have been fitted on a row it predicts, and the
    # fitting partition must be a *partition* rather than the pooled training
    # split. Both are checked by count and by set, because a pooled fit here
    # would leak every scored row's own label while leaving the metric looking
    # ordinary -- see N_LEVEL2_FIT_ROWS.
    fitted_on = fit_rows if unit["level"] == 2 else sub_fit
    n_fit = len(fitted_on)
    if n_fit == N_TRAIN_ROWS:
        raise AssertionError(
            f"{unit['key']}: fitted on all {N_TRAIN_ROWS:,} training rows, "
            f"which is the pooled split and not a fold's fitting partition")
    # The partition identity, per level: what was fitted on plus what is
    # predicted must be exactly the frame they were cut from. Not an expected
    # constant -- the folds are water-filled and unequal by a row.
    whole = N_TRAIN_ROWS if unit["level"] == 2 else len(fit_rows)
    if n_fit + len(predicted) != whole:
        raise AssertionError(
            f"{unit['key']}: {n_fit:,} fit rows + {len(predicted):,} predicted "
            f"rows = {n_fit + len(predicted):,}, but the frame they were cut "
            f"from has {whole:,}. The fit and predict sets must partition it.")
    if unit["level"] == 2 and len(predicted) not in N_HELD_ROWS_PER_OUTER_FOLD:
        raise AssertionError(
            f"{unit['key']}: outer fold holds {len(predicted):,} rows, not one "
            f"of the water-filled sizes {N_HELD_ROWS_PER_OUTER_FOLD}")
    overlap = set(fitted_on.pair_id.astype(int)) & set(
        predicted.pair_id.astype(int))
    if overlap:
        raise AssertionError(
            f"{unit['key']}: {len(overlap)} pair_id(s) are both fitted and "
            f"predicted, first {sorted(overlap)[:5]}")
    for member in members:
        if member["n_fit_rows"] >= n_fit:
            raise AssertionError(
                f"{unit['key']}: a member reports {member['n_fit_rows']:,} fit "
                f"rows against a {n_fit:,}-row partition; every member must "
                f"hold out its own inner fold")

    # Scored on the rows this unit's members held out, which is a real
    # out-of-fold panel for every unit at either level. Raw predictions: no
    # calibration map goes in front of a ranking metric, because isotonic
    # regression is a step function and the frozen contract gives ties average
    # ranks, so it destroys rank resolution while preserving rank order.
    y_true = predicted.y_log1p.to_numpy()
    scores = nested.score_fold(unit["key"], predicted.allele, y_true, y_pred)
    coverage = nested.fold_coverage(predicted.allele, y_true)
    sens = nested.score_fold(f"{unit['key']}_min50", predicted.allele, y_true,
                             y_pred, min_rows=SENSITIVITY_MIN_ROWS)
    sens_cov = nested.fold_coverage(predicted.allele, y_true,
                                    min_rows=SENSITIVITY_MIN_ROWS)
    return {
        "key": unit["key"], "level": unit["level"], "outer_fold": f,
        "inner_fold": unit["inner_fold"], "role": role, "ok": True,
        "n_members": result["n_members"], "n_features": result["n_features"],
        "n_predicted": len(predicted),
        "pair_id": predicted.pair_id.astype(int).tolist(),
        "y_pred": [float(v) for v in y_pred],
        **extra,
        "members": members,
        "median_per_allele_spearman": scores.median_spearman,
        "iqr_low": scores.iqr_spearman[0], "iqr_high": scores.iqr_spearman[1],
        "mae_log1p": scores.mae_log1p,
        "n_alleles_unranked": scores.n_alleles_unranked,
        "n_alleles_eligible": coverage["n_alleles_eligible"],
        "eligible_row_coverage": coverage["eligible_row_coverage"],
        "median_per_allele_spearman_min50": sens.median_spearman,
        "n_alleles_eligible_min50": sens_cov["n_alleles_eligible"],
        "eligible_row_coverage_min50": sens_cov["eligible_row_coverage"],
        "payload_digest": payload["digest"],
        "payload_sequences": payload["n_sequences"],
        "units_schema_version": UNITS_SCHEMA_VERSION,
        "unit_seconds": round(time.time() - started, 3),
        # GiB, not MB: on Linux ru_maxrss is in kilobytes, so /1024**2 is
        # already gibibytes. This field was named peak_rss_mb and ::smoke then
        # divided it by 1024 again, reporting a 0.00 GiB peak against a real
        # 0.673. A memory guard that cannot fire is worse than none, so the
        # field is named for its actual unit.
        "peak_rss_gib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
    }


# --------------------------------------------------------------------------
# free self-check
# --------------------------------------------------------------------------


def self_check() -> int:
    """Everything checkable without Modal and without touching the network.

    Deliberately placed **above** ``import modal`` so that ``--self-check``
    cannot import the SDK at all. ``modal run --dry-run`` still builds and
    validates the image, so "imports modal" and "free" are not the same claim,
    and this project has already described a ``--dry-run`` as free while it
    quietly started a container.
    """
    print(f"repo                {REPO}")
    print(f"rates               {load_rates()}  [{RATES_PATH.relative_to(REPO)}]")
    print(f"worker shape        {WORKER_CPU:g} cores / {WORKER_MEMORY} MiB, "
          f"{WORKER_THREADS} BLAS threads (PROVISIONAL until ::smoke)")
    print(f"volume              {PAYLOAD_VOLUME} -> {PAYLOAD_MOUNT}/{PAYLOAD_NAME}")
    for module in PEPSTAB_MODULES:
        path = REPO / "pepstab" / module
        print(f"  pepstab/{module:<14} exists={path.exists()}")
    missing = [m for m in PEPSTAB_MODULES if not (REPO / "pepstab" / m).exists()]
    if missing:
        print(f"\nmissing from the repo: {missing}")
        return 1

    sys.path.insert(0, str(REPO))
    units = enumerate_units()
    by_level = {lv: sum(1 for u in units if u["level"] == lv) for lv in (2, 3)}
    from pepstab import nested

    per_outer = nested.ASSESS_INNER_FOLDS * len(LAYERS) * 3
    print(f"\nunits               {len(units)}  {by_level} "
          f"(level 2 = outer folds, level 3 = outer x inner)")
    print(f"arity               {nested.ASSESS_INNER_FOLDS} inner x {len(LAYERS)} "
          f"layers {list(LAYERS)} x 3 seeds = {per_outer} members/outer fold "
          f"(A0: 5 x 2 encodings x 3 = 30)")
    print(f"networks            {len(units) * per_outer} "
          f"= {by_level[2]} x {per_outer} + {by_level[3]} x {per_outer}")
    print(f"smoke subset        {[u['key'] for u in smoke_units()]}")
    # One placeholder, labelled as one. No ratio is applied to it: see
    # LOCAL_PILOT_SECONDS_PER_UNIT for why two opposite Modal/laptop ratios
    # were measured today and why neither may price a run.
    print_forecast(
        forecast_numbers(len(units),
                         seconds_per_unit=LOCAL_PILOT_SECONDS_PER_UNIT),
        "placeholder: local pilot timing on a contended laptop, NOT measured "
        "on Modal and NOT a price", measured=False)
    print("\nThis spends nothing and imports no modal. Next: ::smoke.")
    return 0


#: Repo modules copied into the image, mirroring the repo tree so that
#: ``import pepstab.nested`` resolves the same way it does locally. ``augment``
#: is here because ``pepstab/__init__.py`` imports it, so omitting it makes
#: every ``import pepstab.*`` fail -- the same class of silent failure as a
#: module-scope repo read. ``esm.py`` is **not** here: the payload arrives
#: pre-sliced, so nothing in a container reads the embedding cache, and that
#: module's only heavy dependency (torch) is therefore never needed.
PEPSTAB_MODULES = (
    "__init__.py",
    "augment.py",
    "data.py",
    "evaluation.py",
    "features.py",
    "mlp.py",
    "nested.py",
    "splits.py",
)


if __name__ == "__main__" and "--self-check" in sys.argv:
    raise SystemExit(self_check())


import modal  # noqa: E402 -- below the self-check on purpose; see self_check()

app = modal.App("pepstab-phasea-esm")

#: The 146 MB training-only embedding artifact, mounted read-only. A Volume and
#: not an image layer because it is bulk binary data, which is this repo's
#: convention, and not a function argument because 146 MB travelling with every
#: call is not a payload. ``create_if_missing`` so ``--self-check`` and the test
#: suite never have to reach the network to name it.
payload_volume = modal.Volume.from_name(PAYLOAD_VOLUME, create_if_missing=True)

# Exact pins matching the local venv. catboost and torch are deliberately
# absent: this arm's learner is this project's numpy MLP, and nothing in a
# container imports the ESM extractor. scikit-learn is required -- the peptide
# PCA is ``sklearn.utils.extmath.randomized_svd``, and the version is pinned
# because a different randomized_svd would change the basis and therefore the
# features, which is exactly the local/cloud drift this app exists to avoid.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.5.3",
        "pandas==3.0.6",
        "scipy==1.18.1",
        "scikit-learn==1.9.1",
    )
    .env({
        # Pinned, not tuned, and set in the image rather than by a call: the
        # BLAS pool is sized when numpy is imported and importing the science
        # layer imports numpy, so the image environment -- already in place
        # when the interpreter starts -- is the only point early enough. The
        # same seed at a different thread count moves a single network by ~0.02
        # median Spearman, so every row carries threads_observed and ::smoke
        # refuses to pass on a mismatch.
        "OMP_NUM_THREADS": str(WORKER_THREADS),
        "MKL_NUM_THREADS": str(WORKER_THREADS),
        "OPENBLAS_NUM_THREADS": str(WORKER_THREADS),
        "NUMEXPR_NUM_THREADS": str(WORKER_THREADS),
        "VECLIB_MAXIMUM_THREADS": str(WORKER_THREADS),
    })
)
for _module in PEPSTAB_MODULES:
    image = image.add_local_file(
        REPO / "pepstab" / _module, f"/root/pepstab/{_module}", copy=True
    )


def _container_memory() -> dict:
    """Peak memory this container reached. Telemetry must never fail a run.

    ``WORKER_MEMORY`` is a guess with headroom and an OOM partway through a
    paid batch is the failure this exists to prevent. Same reader as
    ``modal_app/foldx_scoring.py``.
    """
    import resource

    out: dict = {}
    for path in ("/sys/fs/cgroup/memory.peak",
                 "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            out["cgroup_peak_gib"] = round(
                int(Path(path).read_text().strip()) / 1024**3, 3
            )
            break
        except Exception:  # noqa: BLE001 -- telemetry must never fail a run
            continue
    try:
        out["self_max_rss_gib"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3
        )
    except Exception:  # noqa: BLE001
        pass
    return out


def n_threads() -> int:
    """The thread count the container actually got, from the environment."""
    return int(os.environ.get("OMP_NUM_THREADS", "0"))


@app.function(
    image=image,
    cpu=WORKER_CPU,
    memory=WORKER_MEMORY,
    timeout=60 * 60,
    max_containers=MAX_CONTAINERS,
    volumes={PAYLOAD_MOUNT: payload_volume},
    # retries=1 because a unit is idempotent: run_unit is pure, writes nothing
    # and reads only the read-only Volume, so a retry recomputes rather than
    # double-counting. One retry covers a preempted container without masking a
    # unit that fails deterministically -- that one returns ok=False twice and
    # shows up in the table.
    retries=1,
)
def run_chunk(units: list[dict], rows: dict, config: dict) -> list[dict]:
    """Run one chunk of units serially and return one result row each.

    The decorator's ``cpu`` / ``memory`` / ``max_containers`` are **defaults**;
    the entrypoints override them with ``Function.with_options``, per
    ``modal_app/foldx_scoring.py:532-534``, so the shape is a run parameter
    rather than something frozen into the source.

    A unit that raises becomes a recorded ``ok=False`` row: one bad unit must
    not discard the others that shared its container and are already paid for.
    """
    sys.path.insert(0, "/root")

    observed = n_threads()
    started = time.time()
    results: list[dict] = []
    for unit in units:
        t0 = time.time()
        try:
            result = run_unit(unit, rows, config)
        except Exception as exc:  # noqa: BLE001 -- recorded, not raised
            import traceback

            result = {"ok": False,
                      "error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc()[-2000:]}
        row = dict(result)
        # Every row carries its fold under an unambiguous name, including a
        # failed one that never got far enough to report it. The A1-A3 screen's
        # output has `outer_fold` null on every row with the fold hiding in
        # `fold` and in the `key` string: a consumer grouping on the null field
        # collapses all five folds into one bucket and gets a single
        # plausible-looking number that is wrong. Nothing here is named `fold`.
        row.setdefault("key", unit.get("key"))
        row.setdefault("outer_fold", unit.get("outer_fold"))
        row.setdefault("inner_fold", unit.get("inner_fold"))
        row.setdefault("level", unit.get("level"))
        row.setdefault("ok", True)
        if row.get("outer_fold") is None:
            raise AssertionError(
                f"unit {unit.get('key')!r} produced a row with no outer_fold; "
                f"a consumer grouping on it would silently merge folds")
        row["threads"] = WORKER_THREADS
        row["threads_observed"] = observed
        row["thread_pin_ok"] = observed == WORKER_THREADS
        row["worker_cpu"] = WORKER_CPU
        row["worker_memory_mib"] = WORKER_MEMORY
        row["unit_wall_s"] = round(time.time() - t0, 3)
        results.append(row)

    wall = time.time() - started
    memory = _container_memory()
    ok = sum(1 for r in results if r.get("ok"))
    print(f"{len(results)} units, {ok} ok, {len(results) - ok} failed, "
          f"{wall:.1f}s wall; threads {observed}; memory {memory}")
    for row in results:
        row["chunk_wall_s"] = round(wall, 1)
        row.update(memory)
    return results


# --------------------------------------------------------------------------
# entrypoints
# --------------------------------------------------------------------------


def _shape(cpu: float, memory_gib: float, containers: int):
    """``run_chunk`` with this run's shape applied at call time."""
    return run_chunk.with_options(
        cpu=cpu, memory=int(memory_gib * 1024), max_containers=containers
    )


def _client_config() -> dict:
    """The stage 3 selected configuration, reconciled on the client.

    Imported from the arm's own script so there is one definition, and resolved
    here rather than in a container so that a drift between
    ``reports/stage3_headline.json`` and ``scripts/esm_arm.py`` stops the run
    before anything is billed.
    """
    sys.path.insert(0, str(REPO))
    from scripts.phase_a_esm_oof import reconcile_selection
    from scripts.baseline_sequence import SEEDS

    config = reconcile_selection()
    return {**config, "seeds": list(SEEDS)}


def _assert_outer_folds_match() -> int:
    """The outer assignment must equal the file A5 joins the two arms on.

    Loaded and compared, never recomputed into agreement. Checked on the client
    before any container starts: a disagreement here would make the sequence
    and ESM components of A5's blend unjoinable, and finding that out after the
    batch would mean paying twice.
    """
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(REPO))
    from pepstab import nested
    from pepstab.data import load_with_splits

    path = OUT_DIR / "phase_a_outer_folds.csv"
    if not path.exists():
        raise SystemExit(
            f"{path.relative_to(REPO)} does not exist. A0 writes it before "
            f"fitting; this arm's outer folds must be asserted equal to it, "
            f"not assumed.")
    df = load_with_splits()
    train = df[df.split == "train"].sort_values("pair_id").reset_index(drop=True)
    published = pd.read_csv(path)
    if not np.array_equal(published.pair_id.to_numpy(), train.pair_id.to_numpy()):
        raise SystemExit("phase_a_outer_folds.csv covers a different pair_id "
                         "set, or order, than the training split")
    mine = nested.outer_folds(train)
    if not np.array_equal(published.outer_fold.to_numpy(), mine):
        bad = int((published.outer_fold.to_numpy() != mine).sum())
        raise SystemExit(f"outer fold assignment disagrees with "
                         f"phase_a_outer_folds.csv on {bad:,} rows")
    print(f"outer folds == {path.relative_to(REPO)} for all {len(published):,} "
          f"rows (loaded and compared)")
    return len(published)


def _check_volume(expect_digest: str = "") -> dict:
    """The artifact must be on the Volume, and be the one we priced.

    A stale Volume is the failure mode that would otherwise be discovered as a
    number that disagrees with the pilot, which is much later and much worse.
    """
    entries = {e.path: e for e in payload_volume.listdir("/")}
    if PAYLOAD_NAME not in entries:
        raise SystemExit(
            f"{PAYLOAD_NAME} is not on volume {PAYLOAD_VOLUME}. Build it with\n"
            f"  .venv/bin/python scripts/phase_a_esm_oof.py --write-payload "
            f"/tmp/{PAYLOAD_NAME}\n"
            f"then\n  modal volume put {PAYLOAD_VOLUME} /tmp/{PAYLOAD_NAME} "
            f"{PAYLOAD_NAME}")
    entry = entries[PAYLOAD_NAME]
    size_mb = (entry.size or 0) / 1e6
    print(f"volume {PAYLOAD_VOLUME}: {PAYLOAD_NAME} {size_mb:.1f} MB")
    if expect_digest:
        print(f"  expecting payload digest {expect_digest[:16]} in every row")
    return {"name": PAYLOAD_NAME, "size_mb": round(size_mb, 1)}


@app.local_entrypoint()
def smoke(
    profile: str = PROFILES[0],
    cpu: float = WORKER_CPU,
    memory_gib: float = WORKER_MEMORY / 1024,
):
    """3 real units end to end, before any batch. Project invariant.

    Runs the **same** ``run_chunk`` the batch does, on the same image and the
    same Volume, so what it measures is the thing that will be launched. It
    writes the measured seconds-per-unit and the peak memory, which are the two
    numbers ``::forecast`` and ``WORKER_MEMORY`` are otherwise guessing at.
    """
    check_profile(profile)

    _assert_outer_folds_match()
    config = _client_config()
    _check_volume()
    picked = smoke_units()
    n = len(picked)
    assert 3 <= n <= 5, f"{n} units: the invariant asks for 3-5 examples"

    rates = load_rates()
    budget_s = 60 * 60
    worst = ((cpu * rates["cpu_hour"] + memory_gib * rates["mem_gib_hour"])
             * budget_s / 3600)
    print(f"smoke: {n} units on 1 container x {cpu:g} cores / {memory_gib:g} GiB, "
          f"<= {budget_s // 60:.0f} min => at most ${worst:.4f}")
    for unit in picked:
        print(f"  {unit['key']} (level {unit['level']})")

    rows = load_rows()
    started = time.time()
    results = _shape(cpu, memory_gib, 1).remote(picked, rows, config)
    wall = time.time() - started

    bad = [r for r in results if not r.get("ok")]
    for row in bad:
        print(f"  FAILED {row.get('key')}: {row.get('error')}")
        if row.get("traceback"):
            print(row["traceback"])
    nulls = [(r.get("key"), k) for r in results if r.get("ok")
             for k in REQUIRED_METRIC_KEYS if r.get(k) is None]
    for key, missing in nulls:
        print(f"  MISSING {key}: {missing} is null")
    unpinned = [r.get("key") for r in results if not r.get("thread_pin_ok")]
    for key in unpinned:
        print(f"  UNPINNED {key}: container did not run at {WORKER_THREADS} threads")
    # Every unit must have read the same artifact, and it must be the one the
    # arity and the price were derived from.
    digests = {r.get("payload_digest") for r in results if r.get("ok")}
    if len(digests) > 1:
        print(f"  MIXED PAYLOADS: {digests}")
    arity = {r.get("n_members") for r in results if r.get("ok")}

    per_unit = [float(r.get("unit_seconds") or r.get("unit_wall_s") or 0.0)
                for r in results]
    per_unit = [s for s in per_unit if s > 0]
    # Three readings, all already in GiB: the cgroup's (the whole container),
    # the chunk runner's RSS, and each unit's own. The reservation has to clear
    # the largest, so no conversion may sit between them -- one did, dividing an
    # already-GiB figure by 1024, and this guard printed 0.00 GiB against a real
    # 0.673. A memory check that cannot fire is worse than none.
    peak = max([r.get("cgroup_peak_gib") or 0.0 for r in results]
               + [r.get("self_max_rss_gib") or 0.0 for r in results]
               + [r.get("peak_rss_gib") or 0.0 for r in results]
               ) if results else 0.0
    summary = {
        "profile": profile,
        "n": len(results),
        "ok": len(results) - len(bad),
        "failed": len(bad),
        "keys": [r.get("key") for r in results],
        "levels": [r.get("level") for r in results],
        "units_schema_version": UNITS_SCHEMA_VERSION,
        "payload_digests": sorted(d for d in digests if d),
        "n_members_per_unit": sorted(a for a in arity if a),
        "n_features": sorted({r.get("n_features") for r in results if r.get("ok")}),
        # Max, not mean: the batch's wall is set by its slowest container and a
        # 3-unit sample has no business claiming a tighter statistic. The units
        # are also unequal by construction -- level 2 fits on more rows.
        "seconds_per_unit": round(max(per_unit), 3) if per_unit else None,
        "seconds_per_unit_median": (
            round(sorted(per_unit)[len(per_unit) // 2], 3) if per_unit else None),
        "chunk_wall_s": results[0].get("chunk_wall_s") if results else None,
        "client_wall_s": round(wall, 1),
        "peak_gib": peak,
        "worker_memory_mib": WORKER_MEMORY,
        "threads": WORKER_THREADS,
        "shape": {"cpu": cpu, "memory_gib": memory_gib, "containers": 1},
        "scores": {r.get("key"): r.get("median_per_allele_spearman")
                   for r in results if r.get("ok")},
        "all_ok": not bad and not nulls and not unpinned and len(digests) <= 1,
    }
    out = OUT_DIR / f"phase_a_esm_smoke_{profile}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # The predictions are journalled by ::run, not here: a smoke summary is for
    # reading, so the 3 x ~1.4 MB of floats stay out of it.
    light = [{k: v for k, v in r.items()
              if k not in ("pair_id", "y_pred", "inner_pair_id", "inner_oof_6",
                           "members")} for r in results]
    out.write_text(json.dumps({**summary, "rows": light}, indent=2,
                              default=str) + "\n")
    print("\n" + json.dumps(summary, indent=2, default=str))
    print(f"wrote {out.relative_to(REPO)}")

    if not summary["all_ok"]:
        raise SystemExit("smoke failed; do not launch the batch")
    if peak and peak > 0.85 * WORKER_MEMORY / 1024:
        print(f"WARNING: peak {peak:.2f} GiB against a "
              f"{WORKER_MEMORY / 1024:.1f} GiB reservation. Raise --memory-gib "
              f"before the batch.")
    print(f"smoke passed: {len(results)}/{len(results)} units ok, "
          f"{summary['seconds_per_unit']:.1f}s/unit measured, {peak:.2f} GiB peak.")
    print("Note the per-unit scores above are single units at full arity, not "
          "a five-fold result.")
    print(f"Next: ::forecast --measured-from {out.relative_to(REPO)}")


@app.local_entrypoint()
def forecast(
    profile: str = PROFILES[0],
    measured_from: str = "",
    seconds_per_unit: float = 0.0,
    cpu: float = WORKER_CPU,
    memory_gib: float = WORKER_MEMORY / 1024,
    containers: int = MAX_CONTAINERS,
    chunk: int = DEFAULT_CHUNK,
):
    """Price the batch from a measured per-unit time. Refuses to guess one."""
    check_profile(profile)
    if measured_from:
        seconds_per_unit = seconds_from_smoke(Path(measured_from))
    if not seconds_per_unit:
        raise SystemExit(
            "::forecast needs --measured-from <smoke json> or an explicit "
            "--seconds-per-unit. There is no default on purpose: a forecast "
            "built on a placeholder is how this project once reported a cost "
            "4.6x wrong.")
    units = enumerate_units()
    f = forecast_numbers(len(units), seconds_per_unit, cpu, memory_gib,
                         containers, chunk)
    print_forecast(f, f"A5/ESM batch: {len(units)} units, "
                      f"{len(units) * 30} networks", measured=bool(measured_from))
    out = OUT_DIR / f"phase_a_esm_forecast_{profile}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {**f, "profile": profile, "measured_from": measured_from,
         "networks": len(units) * 30}, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(REPO)}")
    print("Log this in reports/compute_ledger.csv (append only) before ::run.")


@app.local_entrypoint()
def run(
    profile: str = PROFILES[0],
    measured_from: str = "",
    seconds_per_unit: float = 0.0,
    cpu: float = WORKER_CPU,
    memory_gib: float = WORKER_MEMORY / 1024,
    containers: int = MAX_CONTAINERS,
    chunk: int = DEFAULT_CHUNK,
    limit: int = 0,
    dry_run: bool = False,
    resume: bool = False,
):
    """The batch: 30 units, one fitting partition each, journalled as they land.

    Every unit's **full return** -- predictions included -- is written to
    ``reports/further_research/phase_a_esm_runs_<profile>.partial.jsonl`` with
    ``write`` / ``flush`` / ``os.fsync`` before anything else happens. Rows
    accumulate on the *client*, and the A1-A4 screen lost its gRPC connection
    mid-``.map()`` twice, once capturing 8 of 3,360 units; ``--detach`` keeps
    containers alive but nothing is collecting their results. ``--resume``
    rebuilds the run from the journal alone and refits nothing already in it.

    An existing journal without ``--resume`` is a hard stop, not a silent
    recompute of paid-for units.
    """
    check_profile(profile)
    assert chunk >= 1, f"--chunk must be at least 1 unit, got {chunk}"

    _assert_outer_folds_match()
    config = _client_config()
    units = enumerate_units()
    if limit:
        units = units[:limit]
    assert units, "no units enumerated"

    if measured_from:
        seconds_per_unit = seconds_from_smoke(Path(measured_from))
    if seconds_per_unit:
        print_forecast(
            forecast_numbers(len(units), seconds_per_unit, cpu, memory_gib,
                             containers, chunk),
            f"A5/ESM batch: {len(units)} units, {len(units) * 30} networks",
            measured=bool(measured_from))
    else:
        print("no --seconds-per-unit and no --measured-from: running "
              "unforecast. ::smoke writes the measured time; there is no "
              "default on purpose.")

    if dry_run:
        print(f"\ndry run: {len(units)} units planned on the client. No remote "
              f"call was made.\n  Not free, though: `modal run` built and "
              f"validated this app's image before this entrypoint ran. For a "
              f"forecast that cannot start anything, use --self-check, which "
              f"does not import modal.")
        return

    _check_volume()
    partial = OUT_DIR / f"phase_a_esm_runs_{profile}.partial.jsonl"
    done: dict[str, dict] = {}
    if resume and partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("ok"):
                    done[row.get("key", "")] = row
        units = [u for u in units if u["key"] not in done]
        print(f"resuming: {len(done)} units already in "
              f"{partial.relative_to(REPO)}, {len(units)} left")
    elif partial.exists():
        n_have = sum(1 for ln in partial.read_text().splitlines() if ln.strip())
        raise SystemExit(
            f"{partial.relative_to(REPO)} already exists, holding {n_have} "
            f"units from an interrupted run. Pass --resume to continue it, or "
            f"delete the file to recompute from scratch -- silently rescoring "
            f"paid-for units is not a safe default.")

    rows = load_rows()
    chunks = [units[i:i + chunk] for i in range(0, len(units), chunk)]
    fn = _shape(cpu, memory_gib, containers)
    print(f"{len(units)} units in {len(chunks)} chunks of up to {chunk}, "
          f"{containers} containers x {cpu:g} cores x {WORKER_THREADS} threads, "
          f"CPU only")

    started = time.time()
    results: list[dict] = list(done.values())
    if chunks:
        partial.parent.mkdir(parents=True, exist_ok=True)
        with partial.open("a") as journal:
            for part in fn.map(chunks, kwargs={"rows": rows, "config": config}):
                for row in part:
                    journal.write(json.dumps(row, default=str) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                results.extend(part)
                print(f"  {len(results)}/{len(units) + len(done)} "
                      f"[{(time.time() - started) / 60:.1f} min]", flush=True)
    wall = time.time() - started

    ok = sum(1 for r in results if r.get("ok"))
    print(json.dumps({"profile": profile, "units": len(results), "ok": ok,
                      "failed": len(results) - ok,
                      "unpinned_threads": sum(
                          1 for r in results if not r.get("thread_pin_ok")),
                      "wall_min": round(wall / 60, 1)}, indent=2))
    _write_runs_csv(results, OUT_DIR / f"phase_a_esm_unit_runs_{profile}.csv")
    print(f"\njournal {partial.relative_to(REPO)} kept, holding every unit's "
          f"predictions. Assemble the contract files with:\n"
          f"  .venv/bin/python scripts/phase_a_esm_oof.py --from-journal "
          f"{partial.relative_to(REPO)}")


def _write_runs_csv(results: list[dict], path: Path) -> None:
    """One row per unit, union of keys, in first-seen order.

    ``restval=""`` because a failed unit carries ``error`` and no metrics while
    a level-2 unit carries ``inner_oof_6`` and a level-3 one does not, so the
    header is the union rather than the first row's keys;
    ``extrasaction="ignore"`` would silently drop a column only some units
    emit. The bulk prediction arrays are left to the journal.
    """
    bulk = ("pair_id", "y_pred", "inner_pair_id", "inner_oof_6", "members",
            "traceback")
    rows = [{k: v for k, v in r.items() if k not in bulk} for r in results]
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.relative_to(REPO)} ({len(rows)} units)")
