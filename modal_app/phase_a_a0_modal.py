"""Phase A A0 refitted on Linux containers, beside the macOS reference.

A0 is Phase A's reference arm and the **only** arm in the study fitted on
macOS. Every other arm -- the A1-A3 screen, the A5/ESM arm, Phase B's arms --
ran in Linux containers, and the forthcoming 30-member assessment will too. So
a Linux-fitted A0 is needed for those comparisons to be within-platform.

Phase B measured the consequence, all at threads=1 on outer fold 0, 30 members:

    nested.fit_mlp_ensemble (macOS) vs A0's shipped predictions
        max|diff| 4.44e-16   corr 1.0000000000   -> A0 is bit-identical to its harness
    phase_b loop (macOS) vs fit_mlp_ensemble (macOS)
        max|diff| 1.18e-06                        -> implementations equivalent
    macOS vs Modal Linux, same code
        max|diff| 1.82e-01   mean 3.21e-02        -> the whole discrepancy

Not thread count, not implementation drift: macOS/Accelerate arm64 against
Linux/OpenBLAS x86-64 changing float32 accumulation order, worth 0.0020 on
median-of-medians there.

**This does not replace the macOS A0.** It writes ``*_linux`` names beside it;
the existing files are untouched, and the platform is recorded in the
provenance as a first-class fact rather than a footnote.

**Why thread count is a run parameter here.** The shipped A0 is macOS at
threads=1 and this run is Linux at threads=2, so a straight difference between
them confounds platform with thread count -- and this project has measured that
thread count moves a *single* network by ~0.02 median Spearman (ensembles are
far steadier, which is the point of the 30-member arity). Running the five
level-2 units a second time at threads=1 costs about a sixth of the batch and
decomposes the two cleanly on the headline metric, because level-3 units feed
only ``inner_oof_30`` and never a fold score. ``--threads`` and ``--levels``
exist for exactly that, and ``phase_a_platform_artifact.md`` reports the 2x2.

    # free: imports no modal, starts nothing
    .venv/bin/python modal_app/phase_a_a0_modal.py --self-check

    # 3 units end to end, measuring per-unit time (project invariant)
    MODAL_PROFILE=colleague modal run modal_app/phase_a_a0_modal.py::smoke \
        --profile colleague

    MODAL_PROFILE=colleague modal run modal_app/phase_a_a0_modal.py::forecast \
        --profile colleague \
        --measured-from reports/further_research/phase_a_a0_smoke_colleague.json

    MODAL_PROFILE=colleague modal run --detach \
        modal_app/phase_a_a0_modal.py::run --profile colleague \
        --measured-from reports/further_research/phase_a_a0_smoke_colleague.json

    # the threads=1 decomposition arm: level-2 units only
    MODAL_PROFILE=colleague modal run --detach \
        modal_app/phase_a_a0_modal.py::run --profile colleague --threads 1 \
        --levels 2 --measured-from <smoke json>

    # assemble the contract files from the journal, refitting nothing
    .venv/bin/python modal_app/phase_a_a0_modal.py --assemble \
        reports/further_research/phase_a_a0_runs_colleague_t2.partial.jsonl

**No Volume and no payload.** Unlike the A5/ESM arm, features come from
``pepstab.features.build_features``, which is row-local, so the row records
travel as a function argument exactly as ``modal_app/phase_a_experiments.py``
sends them. There is nothing bulk to stage.

Same contract as the macOS A0: two nesting levels, three roles, 5 inner x 2
encodings x 3 seeds = 30 members per outer fold, 900 networks, 177,444 output
rows. The member recipe is inherited from ``pepstab.nested.REFERENCE_MLP`` and
``scripts.baseline_ensemble.SELECTED``, not retyped.

CPU only: no function here declares ``gpu=``, asserted by reading the decorator
in ``tests/test_phase_a_a0_modal.py``.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Same two profiles as every other app here; ``colleague`` is the
#: ``sofyaleyn`` workspace, which the user authorised for Phase A work.
PROFILES = ("a-cheparukhin", "colleague")

#: A unit is one fitting partition and 30 single-model numpy/BLAS fits. 2.0
#: cores is the smallest reservation that lets BLAS use a second thread without
#: the container thrashing against its own cgroup limit.
#:
#: memory: the widest object is a one-hot ``pep_pseudo`` feature matrix,
#: 15,773 x 860 float32 = 54 MB, against the A5/ESM arm's 146 MB payload and
#: its float64 PCA working copy, which peaked at 0.68 GiB. 2 GiB is a guess
#: with headroom and ``::smoke`` writes the measured cgroup peak so the guess
#: can be replaced rather than inherited.
WORKER_CPU = 2.0
WORKER_MEMORY = 2048  # MiB

#: Thread counts this app can pin, each with its own image. Pinned in the image
#: ``.env`` and not by a call: the BLAS pool is sized when numpy is imported
#: and importing the science layer imports numpy, so the image environment --
#: already in place when the interpreter starts -- is the only point early
#: enough. 2 matches the A1-A3 screen and the A5/ESM arm; 1 matches the shipped
#: macOS A0 and exists to decompose platform from thread count.
THREAD_CHOICES = (1, 2)
DEFAULT_THREADS = 2

#: Every variable the pin has to cover. Applied twice -- in the image and
#: again in the worker before numpy is first imported -- because a
#: ``--threads 1`` run dispatched correctly and still came back at 2.
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

MAX_CONTAINERS = 30
DEFAULT_CHUNK = 1

#: 45 s until ``::smoke`` measures it. This image is lighter than the A5/ESM
#: one -- no scikit-learn, no torch, no catboost.
CONTAINER_STARTUP_S = 45.0

#: 1.25, the same margin stages 4c and 8 forecast with.
COST_MARGIN = 1.25

RATES_PATH = REPO / "reports" / "ectodomain_rates.json"
OUT_DIR = REPO / "reports" / "further_research"

#: A0 is the reference reproduction, so the input set is not an option.
INPUT_SET = "pep_pseudo"

N_TRAIN_ROWS = 19_716

#: The outer folds are water-filled by peptide cluster and are therefore
#: **not** equal: 3,944 on fold 0 and 3,943 on folds 1-4. A single expected
#: fit-row count was tried on the A5/ESM arm and failed four of its five
#: level-2 units for exactly this reason, so the anti-pooling check below is a
#: partition identity rather than a literal count.
N_HELD_ROWS_PER_OUTER_FOLD = (3944, 3943, 3943, 3943, 3943)

#: Prespecification §2's declared sensitivity panel.
SENSITIVITY_MIN_ROWS = 50

#: ``sub_fold`` for a level-2 member, which has no sub-fold. Same sentinel the
#: macOS A0 writes, so the two runs tables read alike.
NO_SUB_FOLD = -1

UNITS_SCHEMA_VERSION = 1

REQUIRED_METRIC_KEYS = (
    "median_per_allele_spearman",
    "n_alleles_eligible",
    "eligible_row_coverage",
)


def load_rates() -> dict[str, float]:
    """Metered CPU and memory rates. Read **lazily**, never at module scope.

    Modal imports this module inside every container, where the repo does not
    exist. A module-scope read of a repo file kills the container on import
    before it executes a line of its own, surfacing as "0 tasks" and nothing
    else -- the defect at ``modal_app/proteinmpnn_scoring.py:97-105``, which
    neither a local forecast nor ``--dry-run`` catches because both run where
    the file is present.
    """
    raw = json.loads(RATES_PATH.read_text())
    return {"cpu_hour": float(raw["cpu_hour_cost"]),
            "mem_gib_hour": float(raw["mem_gib_hour_cost"])}


def check_profile(profile: str) -> str:
    """Refuse to run if the requested profile and the active one disagree.

    The house assertion, same form as ``modal_app/foldx_scoring.py``. What it
    protects is the budget: the forecast is priced against one workspace's
    metered rates and one workspace's reserved credit, and spending the wrong
    workspace's credit is not a recoverable mistake.
    """
    active = os.environ.get("MODAL_PROFILE")
    assert profile in PROFILES, f"--profile must be one of {PROFILES}, got {profile!r}"
    assert active in (None, profile), (
        f"--profile {profile!r} but MODAL_PROFILE={active!r}. The forecast is "
        f"priced against one workspace's metered rates and reserved credit; "
        f"set both the same.")
    if active is None:
        print(f"warning: MODAL_PROFILE is unset, so this runs against your "
              f"active profile. Billing the {profile!r} workspace -- confirm "
              f"that is the same one.")
    return profile


# --------------------------------------------------------------------------
# cost model
# --------------------------------------------------------------------------


def forecast_numbers(n_units: int, seconds_per_unit: float,
                     cpu: float = WORKER_CPU,
                     memory_gib: float = WORKER_MEMORY / 1024,
                     containers: int = MAX_CONTAINERS,
                     chunk: int = DEFAULT_CHUNK,
                     startup_s: float = CONTAINER_STARTUP_S,
                     margin: float = COST_MARGIN) -> dict:
    """Container-hours and dollars at the repo's metered rates.

    Copied from ``modal_app/phase_a_esm.py``, which copied it from
    ``modal_app/foldx_scoring.py:145-192``, because the billing model is the
    same and should not be re-derived: Modal bills the cores and memory a
    container **reserves**, for as long as it lives, not the cores it keeps
    busy. So the cost is ``containers x cpu x wall``, and a container that is
    not full is a real expense rather than something the arithmetic hides.

    ``seconds_per_unit`` is the slowest unit's time, which sets the wall of the
    container that finishes last and therefore the bill; level-2 units fit on
    ~12,618 rows against level-3's ~10,095, so the units are not equal.
    """
    assert chunk >= 1, f"--chunk must be at least 1 unit, got {chunk}"
    assert containers >= 1, f"--containers must be at least 1, got {containers}"
    assert n_units >= 1, f"nothing to forecast: {n_units} units"

    rates = load_rates()
    n_chunks = math.ceil(n_units / chunk)
    chunk_fill = n_units / (n_chunks * chunk)
    active = min(containers, n_chunks)
    chunks_per_container = math.ceil(n_chunks / active)
    busy_s = chunks_per_container * chunk * seconds_per_unit
    wall_s = busy_s + startup_s
    container_hours = active * wall_s / 3600
    core_hours = container_hours * cpu
    mem_gib_hours = container_hours * memory_gib
    usd = core_hours * rates["cpu_hour"] + mem_gib_hours * rates["mem_gib_hour"]
    return {
        "n_units": n_units, "seconds_per_unit": seconds_per_unit,
        "cpu_per_container": cpu, "memory_gib_per_container": memory_gib,
        "containers_requested": containers, "containers_active": active,
        "container_fill": active / containers, "chunk": chunk,
        "n_chunks": n_chunks, "chunk_fill": chunk_fill,
        "chunks_per_container": chunks_per_container, "startup_s": startup_s,
        "wall_h": wall_s / 3600, "container_hours": container_hours,
        "core_hours": core_hours, "mem_gib_hours": mem_gib_hours,
        "cpu_hour_rate_usd": rates["cpu_hour"],
        "mem_gib_hour_rate_usd": rates["mem_gib_hour"],
        "usd": usd, "usd_with_margin": usd * margin, "margin": margin,
        "rates_source": str(RATES_PATH.relative_to(REPO)),
    }


def print_forecast(f: dict, title: str, measured: bool = False) -> None:
    print(f"\n{title}")
    print(f"  units               {f['n_units']:,} in {f['n_chunks']:,} chunks "
          f"of {f['chunk']} (fill {f['chunk_fill']:.0%})")
    print(f"  per unit            {f['seconds_per_unit']:.1f}s "
          f"{'(MEASURED by ::smoke)' if measured else '(PLACEHOLDER)'}")
    print(f"  containers          {f['containers_active']} active of "
          f"{f['containers_requested']} x {f['cpu_per_container']:g} cores / "
          f"{f['memory_gib_per_container']:g} GiB")
    print(f"  wall                {f['wall_h'] * 60:.1f} min "
          f"(incl. {f['startup_s']:.0f}s startup per container)")
    print(f"  billed              {f['core_hours']:.3f} core-h + "
          f"{f['mem_gib_hours']:.3f} GiB-h")
    print(f"  cost                ${f['usd']:.4f} -> "
          f"${f['usd_with_margin']:.4f} at {f['margin']}x margin "
          f"[{f['rates_source']}]")


def seconds_from_smoke(path: Path) -> float:
    """The per-unit time ``::smoke`` measured, or refuse to guess one."""
    summary = json.loads(path.read_text())
    seconds = summary.get("seconds_per_unit")
    if not seconds:
        raise SystemExit(f"{path} carries no seconds_per_unit; rerun ::smoke")
    if not summary.get("all_ok"):
        raise SystemExit(f"{path} records a failed smoke; do not launch")
    return float(seconds)


# --------------------------------------------------------------------------
# units
# --------------------------------------------------------------------------


def enumerate_units(levels: tuple[int, ...] = (2, 3)) -> list[dict]:
    """The 30 units: 5 level-2 partitions and 25 level-3 cells.

    A unit is one *fitting partition* carrying all 30 of its members, not one
    fit, so a container builds each encoding's feature matrix once rather than
    thirty times.

    ``levels`` exists for the threads=1 decomposition run: level-3 units feed
    only ``inner_oof_30`` and never a fold score, so ``--levels 2`` reproduces
    every per-fold median for a sixth of the batch.
    """
    from pepstab import nested

    units: list[dict] = []
    if 2 in levels:
        units += [{"key": f"L2_o{f}", "level": 2, "outer_fold": f,
                   "inner_fold": -1, "schema": UNITS_SCHEMA_VERSION}
                  for f in range(nested.OUTER_FOLDS)]
    if 3 in levels:
        units += [{"key": f"L3_o{f}_i{k}", "level": 3, "outer_fold": f,
                   "inner_fold": k, "schema": UNITS_SCHEMA_VERSION}
                  for f in range(nested.OUTER_FOLDS)
                  for k in range(nested.ASSESS_INNER_FOLDS)]
    return units


def smoke_units() -> list[dict]:
    """The predeclared 3-unit subset: one of each level, plus a second level 3.

    Named rather than "the first n the enumerator emitted", so the smoke covers
    both nesting levels and the time it measures is the slowest kind of unit --
    the one the forecast extrapolates from.
    """
    by_key = {u["key"]: u for u in enumerate_units()}
    return [by_key[k] for k in ("L2_o0", "L3_o0_i0", "L3_o0_i1")]


def load_rows() -> dict:
    """The training rows, read client-side and passed as a function argument.

    House pattern: ``data/*.csv`` is read on the client, never in a container.

    The sequence columns travel as 75-entry allele maps rather than columns:
    one value per allele repeated across 19,716 rows is a lot of identical
    strings. **Which columns** is taken from ``features.INPUT_SETS`` rather
    than chosen here -- the first attempt sent ``hla_seq`` and ``pep_pseudo``
    needs ``hla_pseudoseq``, which failed every unit of the smoke with a
    ``KeyError`` two seconds in.

    Training split only. The frozen validation and test rows are not in this
    structure, so a container cannot score one even if it tried.
    """
    from pepstab.data import load_with_splits
    from pepstab.features import INPUT_SETS

    df = load_with_splits()
    train = df[df.split == "train"].sort_values("pair_id").reset_index(drop=True)
    assert len(train) == N_TRAIN_ROWS, (
        f"training split has {len(train):,} rows, expected {N_TRAIN_ROWS:,}")
    # Whatever the input set needs beyond the peptide, sent as an allele map.
    per_allele = [c for c in INPUT_SETS[INPUT_SET] if c != "peptide"]
    maps = {}
    for column in per_allele:
        assert train.groupby("allele")[column].nunique().max() == 1, (
            f"an allele maps to more than one {column}; the allele map would "
            f"lose rows")
        maps[column] = (train[["allele", column]].drop_duplicates()
                        .set_index("allele")[column].astype(str).to_dict())
    return {
        "pair_id": train.pair_id.astype(int).tolist(),
        "peptide": train.peptide.astype(str).tolist(),
        "allele": train.allele.astype(str).tolist(),
        "cluster_id": train.cluster_id.astype(int).tolist(),
        "y_log1p": train.y_log1p.astype(float).tolist(),
        "allele_maps": maps,
        "n_rows": len(train),
    }


def rows_to_frame(rows: dict):
    """Rebuild the training frame a container was handed.

    Asserts the rebuilt frame carries every column the input set's features
    need, so a missing one fails here with a name rather than inside
    ``build_features`` as a bare ``KeyError`` thirty fits later.
    """
    import pandas as pd

    from pepstab.features import INPUT_SETS

    frame = pd.DataFrame({k: rows[k] for k in
                          ("pair_id", "peptide", "allele", "cluster_id",
                           "y_log1p")})
    for column, mapping in rows["allele_maps"].items():
        frame[column] = frame.allele.map(mapping)
        assert frame[column].notna().all(), (
            f"an allele is missing from the {column} map")
    missing = [c for c in INPUT_SETS[INPUT_SET] if c not in frame.columns]
    assert not missing, (
        f"the row payload does not carry {missing}, which "
        f"build_features({INPUT_SET!r}) requires")
    return frame


def member_config() -> dict:
    """A0's member recipe, inherited from the science layer rather than typed.

    ``REFERENCE_MLP`` carries lr, batch size, the 300-epoch cap and the
    15-epoch patience; ``SELECTED`` carries the per-encoding ``(hidden, l2)``
    chosen at stage 2 and never re-tuned. Resolved on the client so a drift
    between them stops the run before anything is billed.
    """
    from pepstab import nested
    from pepstab.features import ENCODINGS
    from scripts.baseline_ensemble import SELECTED
    from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE, SEEDS

    if (nested.REFERENCE_MLP.max_epochs, nested.REFERENCE_MLP.patience) != (
            MAX_EPOCHS, PATIENCE):
        raise AssertionError(
            f"nested.REFERENCE_MLP has {nested.REFERENCE_MLP.max_epochs}/"
            f"{nested.REFERENCE_MLP.patience} but stage 2 selected "
            f"{MAX_EPOCHS}/{PATIENCE}")
    selected = {enc: SELECTED[(INPUT_SET, enc)] for enc in ENCODINGS}
    return {
        "input_set": INPUT_SET,
        "encodings": list(ENCODINGS),
        "seeds": list(SEEDS),
        # JSON-safe: tuples become lists crossing the Modal boundary anyway.
        "selected": {enc: [list(hidden), l2]
                     for enc, (hidden, l2) in selected.items()},
        "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "lr": nested.REFERENCE_MLP.lr,
        "batch_size": nested.REFERENCE_MLP.batch_size,
        "source": "pepstab.nested.REFERENCE_MLP + scripts.baseline_ensemble.SELECTED",
    }


def run_unit(unit: dict, rows: dict, config: dict) -> dict:
    """One fitting partition: 30 members, their predictions, their records.

    Pure with respect to the repo -- reads only its arguments, writes nothing --
    so a retry recomputes rather than double-counting.

    The member loop is ``pepstab.nested.fit_mlp_ensemble`` itself, called and
    not mirrored: unlike the A5/ESM arm, A0's features come from
    ``build_features``, which is exactly what that function's ``input_set``
    routes to. The fold budget is therefore inherited rather than retyped, and
    a transplanted ladder has already cost an arm 0.093 median Spearman on this
    project.

    Returns the predictions themselves, not only a metric: the client journals
    the whole return the moment it arrives, because the client is assumed to
    die and a unit's 30 fits are paid for by the time its numbers exist.
    """
    import resource

    import numpy as np

    from pepstab import nested

    started = time.time()
    train = rows_to_frame(rows)
    seeds = tuple(config["seeds"])
    encodings = tuple(config["encodings"])
    selected = {enc: (tuple(hidden), l2)
                for enc, (hidden, l2) in config["selected"].items()}
    n_inner = config.get("n_inner") or nested.ASSESS_INNER_FOLDS
    n_sub = config.get("n_sub") or nested.ASSESS_INNER_FOLDS

    # Inherited from pepstab.nested, never recomputed into agreement: the
    # client asserts the outer vector equals phase_a_outer_folds.csv first.
    outer = nested.outer_folds(train)
    f = unit["outer_fold"]
    held_mask = outer == f
    fit_rows = train[~held_mask].reset_index(drop=True)
    inner = nested.inner_folds(fit_rows, n_inner)

    if unit["level"] == 2:
        held_rows = train[held_mask].reset_index(drop=True)
        result = nested.fit_mlp_ensemble(fit_rows, [held_rows], inner,
                                         encodings, seeds, selected,
                                         input_set=config["input_set"])
        fitted_on, predicted = fit_rows, held_rows
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
        result = nested.fit_mlp_ensemble(sub_fit, [target], sub, encodings,
                                         seeds, selected,
                                         input_set=config["input_set"])
        fitted_on, predicted = sub_fit, target
        y_pred = result["predictions"][0]
        extra = {}
        members = [{**m, "sub_fold": m.pop("inner_fold"), "inner_fold": k}
                   for m in result["members"]]
        role = "inner_oof_30"
    else:
        raise AssertionError(f"unknown level {unit['level']!r}")

    if not np.isfinite(y_pred).all():
        raise AssertionError(f"{unit['key']}: non-finite prediction")

    # Nothing may have been fitted on a row it predicts, and the fitting
    # partition must be a *partition* rather than the pooled split. Phase B
    # fitted one calibration map on pooled inner rows across all five outer
    # folds and applied it to pooled outer-held predictions, which used every
    # scored row's own label -- and the leaked and clean versions agreed on
    # Brier to four decimals. The giveaway was a row count, not a metric.
    #
    # Checked as an identity rather than against an expected count: the folds
    # are water-filled and come out 3,944 / 3,943 x 4, so a single constant was
    # wrong for four of five level-2 units when the A5/ESM arm tried one.
    n_fit = len(fitted_on)
    if n_fit == N_TRAIN_ROWS:
        raise AssertionError(
            f"{unit['key']}: fitted on all {N_TRAIN_ROWS:,} training rows, "
            f"which is the pooled split and not a fold's fitting partition")
    whole = N_TRAIN_ROWS if unit["level"] == 2 else len(fit_rows)
    if n_fit + len(predicted) != whole:
        raise AssertionError(
            f"{unit['key']}: {n_fit:,} fit + {len(predicted):,} predicted = "
            f"{n_fit + len(predicted):,}, but the frame they were cut from has "
            f"{whole:,}. The fit and predict sets must partition it.")
    if unit["level"] == 2 and len(predicted) not in N_HELD_ROWS_PER_OUTER_FOLD:
        raise AssertionError(
            f"{unit['key']}: outer fold holds {len(predicted):,} rows, not one "
            f"of the water-filled sizes {N_HELD_ROWS_PER_OUTER_FOLD}")
    overlap = set(fitted_on.pair_id.astype(int)) & set(
        predicted.pair_id.astype(int))
    if overlap:
        raise AssertionError(
            f"{unit['key']}: {len(overlap)} pair_id(s) both fitted and "
            f"predicted, first {sorted(overlap)[:5]}")
    for member in members:
        if member["n_fit_rows"] >= n_fit:
            raise AssertionError(
                f"{unit['key']}: a member reports {member['n_fit_rows']:,} fit "
                f"rows against a {n_fit:,}-row partition; every member must "
                f"hold out its own inner fold")

    # Scored on the rows this unit's members held out. Raw predictions: no
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
        "n_members": result["n_members"], "n_predicted": len(predicted),
        "n_fit_rows_partition": n_fit,
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
        "units_schema_version": UNITS_SCHEMA_VERSION,
        "unit_seconds": round(time.time() - started, 3),
        # GiB, not MB: on Linux ru_maxrss is in kilobytes, so /1024**2 is
        # already gibibytes. Naming this *_mb and then dividing by 1024 again
        # made the A5/ESM arm's memory guard print 0.00 against a real 0.673,
        # a check that could never fire.
        "peak_rss_gib": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3),
    }


# --------------------------------------------------------------------------
# free paths: self-check and journal assembly
# --------------------------------------------------------------------------


def self_check() -> int:
    """Everything checkable without Modal and without touching the network.

    Deliberately **above** ``import modal`` so ``--self-check`` cannot import
    the SDK. ``modal run --dry-run`` still builds and validates the image, so
    "imports modal" and "free" are not the same claim, and this project has
    already described a ``--dry-run`` as free while it quietly started a
    container.
    """
    print(f"repo                {REPO}")
    print(f"rates               {load_rates()}  [{RATES_PATH.relative_to(REPO)}]")
    print(f"worker shape        {WORKER_CPU:g} cores / {WORKER_MEMORY} MiB "
          f"(PROVISIONAL until ::smoke)")
    print(f"thread choices      {THREAD_CHOICES}, default {DEFAULT_THREADS} "
          f"(macOS A0 shipped at 1)")
    missing = [m for m in PEPSTAB_MODULES if not (REPO / "pepstab" / m).exists()]
    print(f"pepstab modules     {len(PEPSTAB_MODULES)} copied, missing {missing}")
    if missing:
        return 1

    sys.path.insert(0, str(REPO))
    from pepstab import nested

    cfg = member_config()
    units = enumerate_units()
    per_outer = (nested.ASSESS_INNER_FOLDS * len(cfg["encodings"])
                 * len(cfg["seeds"]))
    by_level = {lv: sum(1 for u in units if u["level"] == lv) for lv in (2, 3)}
    print(f"units               {len(units)}  {by_level}")
    print(f"arity               {nested.ASSESS_INNER_FOLDS} inner x "
          f"{len(cfg['encodings'])} encodings {cfg['encodings']} x "
          f"{len(cfg['seeds'])} seeds = {per_outer} members/outer fold")
    print(f"networks            {len(units) * per_outer}")
    print(f"member              hidden/l2 {cfg['selected']}, lr {cfg['lr']}, "
          f"{cfg['max_epochs']} epochs, patience {cfg['patience']}")
    print(f"smoke subset        {[u['key'] for u in smoke_units()]}")
    print_forecast(forecast_numbers(len(units), seconds_per_unit=300.0),
                   "placeholder forecast (NOT measured on Modal, NOT a price)",
                   measured=False)
    print("\nThis spends nothing and imports no modal. Next: ::smoke.")
    return 0


def assemble(journal: Path, out_dir: Path = OUT_DIR, suffix: str = "_linux",
             threads: int = DEFAULT_THREADS) -> int:
    """Write the three ``*_linux`` contract files from a run's journal.

    Refits nothing. The journal is the run's record of truth: rows accumulate
    on the client and the A1-A4 screen lost its gRPC connection mid-``.map()``
    twice, once capturing 8 units of 3,360.

    The generic journal reader is ``scripts.phase_a_esm_oof.collect_from_journal``
    -- it parses, deduplicates a resumed journal, redoes every fit/predict
    disjointness check client-side and enforces the anti-pooling counts, none
    of which is ESM-specific. Only the naming and the provenance differ here.
    """
    import numpy as np
    import pandas as pd
    import platform as platform_mod
    import scipy
    from datetime import datetime, timezone

    sys.path.insert(0, str(REPO))
    from pepstab import nested
    from pepstab.data import RAW_CSV, SPLITS_CSV, load_with_splits
    from scripts.phase_a_esm_oof import (collect_from_journal,
                                         inner_oof_residual_scale, sha256)

    df = load_with_splits()
    train = df[df.split == "train"].sort_values("pair_id").reset_index(drop=True)
    cfg = member_config()
    full_arity = (nested.ASSESS_INNER_FOLDS * len(cfg["encodings"])
                  * len(cfg["seeds"]))

    folds_path = out_dir / "phase_a_outer_folds.csv"
    if not folds_path.exists():
        raise SystemExit(f"{folds_path} missing; A0 writes it before fitting")
    fold_table = pd.read_csv(folds_path)
    if not np.array_equal(fold_table.pair_id.to_numpy(),
                          train.pair_id.to_numpy()):
        raise SystemExit("phase_a_outer_folds.csv covers a different pair_id "
                         "set, or order, than the training split")
    mine = nested.outer_folds(train)
    if not np.array_equal(fold_table.outer_fold.to_numpy(), mine):
        raise SystemExit("outer fold assignment disagrees with "
                         "phase_a_outer_folds.csv")

    (held_frames, inner_frames, runs, fold_scores, n_checks, _n_features,
     _digest, timing) = collect_from_journal(
         journal, train, nested.ASSESS_INNER_FOLDS, nested.ASSESS_INNER_FOLDS,
         full_arity)

    oof = pd.concat(held_frames + inner_frames, ignore_index=True)
    oof = oof.sort_values(["pair_id", "role", "outer_fold"]).reset_index(drop=True)
    folds = pd.DataFrame(fold_scores)
    medians = folds.median_spearman.to_numpy()
    runs_frame = pd.DataFrame(runs)

    checks = [f"outer fold assignment == phase_a_outer_folds.csv for all "
              f"{len(fold_table):,} rows (loaded and compared)"]
    if not np.isfinite(oof.y_pred.to_numpy()).all():
        raise AssertionError("non-finite prediction")
    checks.append(f"np.isfinite on all {len(oof):,} predictions")
    if len(oof) != N_TRAIN_ROWS * 9:
        raise AssertionError(f"{len(oof):,} rows, expected {N_TRAIN_ROWS * 9:,}")
    checks.append(f"{len(oof):,} rows = 19,716 outer_held + 4 x 19,716 "
                  f"inner_oof_6 + 4 x 19,716 inner_oof_30")
    per_role = oof.pivot_table(index="pair_id", columns="role",
                               values="y_pred", aggfunc="size")
    for role, want in (("outer_held", 1), ("inner_oof_6", 4),
                       ("inner_oof_30", 4)):
        if not (per_role[role] == want).all():
            raise AssertionError(f"a pair_id lacks exactly {want} {role} rows")
    checks.append("every pair_id: exactly 1 outer_held, 4 inner_oof_6, "
                  "4 inner_oof_30")
    if set(per_role.index) != set(train.pair_id.to_numpy().tolist()):
        raise AssertionError("pair_id set is not the training split")
    checks.append(f"pair_id set == the training split ({len(per_role):,})")
    held_only = oof[oof.role == "outer_held"].set_index("pair_id")
    if not np.array_equal(
            held_only.outer_fold.reindex(fold_table.pair_id).to_numpy(),
            fold_table.outer_fold.to_numpy()):
        raise AssertionError("outer_held fold ids disagree with the fold table")
    checks.append("outer_held fold id == phase_a_outer_folds.csv for every row")
    straddle = oof[oof.role != "outer_held"].merge(fold_table, on="pair_id",
                                                   suffixes=("", "_own"))
    if (straddle.outer_fold == straddle.outer_fold_own).any():
        raise AssertionError("an inner_oof row sits in its own held-out fold")
    checks.append("no inner_oof row is attributed to its own outer held-out fold")
    if len(runs_frame) != full_arity * (1 + nested.ASSESS_INNER_FOLDS) \
            * nested.OUTER_FOLDS:
        raise AssertionError(f"{len(runs_frame)} members, expected 900")
    checks.append(f"members: {len(runs_frame)} = 150 level-2 + 750 level-3")
    checks.append(f"{n_checks} fit/predict disjointness checks")

    runs_frame = runs_frame[["outer_fold", "inner_fold", "sub_fold", "encoding",
                             "seed", "n_fit_rows", "best_epoch", "dev_mse",
                             "fit_seconds"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out_dir / f"phase_a_seq_oof_cluster{suffix}.csv", index=False)
    runs_frame.to_csv(out_dir / f"phase_a_a0_runs{suffix}.csv", index=False)

    provenance = {
        "arm": "A0 (Linux rerun)",
        "description": ("the Phase A sequence reference refitted in Modal Linux "
                        "containers, beside the macOS original, so that "
                        "Linux-fitted arms are compared within platform"),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "does_not_replace": "reports/further_research/phase_a_a0_provenance.json",
        # First-class facts, not footnotes: this run exists because of them.
        "platform": {
            "where": "Modal Linux containers (debian_slim, x86-64)",
            "blas": "OpenBLAS x86-64 (numpy manylinux wheel)",
            "threads": threads,
            "contrast": ("the shipped A0 ran on macOS/Accelerate arm64 at "
                         "threads=1; float32 accumulation order differs, which "
                         "Phase B measured at max|diff| 1.82e-01 and 0.0020 on "
                         "median-of-medians for one fold"),
            "client_platform": platform_mod.platform(),
        },
        "input_set": INPUT_SET,
        "member_config": cfg,
        "arity": {"members_per_outer_fold": full_arity,
                  "axes": (f"{nested.ASSESS_INNER_FOLDS} inner x "
                           f"{len(cfg['encodings'])} encodings x "
                           f"{len(cfg['seeds'])} seeds")},
        "networks_total": len(runs_frame),
        "min_rows_per_allele_fold": nested.MIN_ROWS_PER_ALLELE_FOLD,
        "sensitivity_min_rows_per_allele_fold": SENSITIVITY_MIN_ROWS,
        "package_versions": {
            "python": platform_mod.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "note": "client versions; the container pins are in the image",
        },
        "input_digests": {"data/splits.csv": sha256(SPLITS_CSV),
                          "data/rasmussen_et_al_dataset.csv": sha256(RAW_CSV)},
        "rows": {"train": len(train), "oof_csv": len(oof),
                 "runs_csv": len(runs_frame)},
        "fold_scores": folds.to_dict(orient="records"),
        "five_fold_median_of_medians": float(np.median(medians)),
        "five_fold_median_of_medians_min50": float(
            np.median(folds.median_spearman_min50.to_numpy())),
        "inner_oof_residual_scale": inner_oof_residual_scale(oof, train),
        "modal_container_timing": timing,
        "asserted": checks,
    }
    (out_dir / f"phase_a_a0_provenance{suffix}.json").write_text(
        json.dumps(provenance, indent=2, default=str) + "\n")

    print(f"\nper-outer-fold scores (Linux, threads={threads}):")
    print(folds.round(4).to_string(index=False))
    print(f"\nfive-fold median of medians: {float(np.median(medians)):.4f} "
          f"(spread {medians.max() - medians.min():.4f})")
    print("\nassertions passed before writing:")
    for check in checks:
        print(f"  - {check}")
    print(f"\nwrote {out_dir.relative_to(REPO)}/: "
          f"phase_a_seq_oof_cluster{suffix}.csv, "
          f"phase_a_a0_runs{suffix}.csv, phase_a_a0_provenance{suffix}.json")
    return 0


#: Repo modules copied into the image, mirroring the repo tree so that
#: ``import pepstab.nested`` resolves as it does locally. ``augment`` is here
#: because ``pepstab/__init__.py`` imports it, so omitting it breaks every
#: ``import pepstab.*``. No ``esm.py`` and no scikit-learn: A0's features are
#: one-hot and BLOSUM from ``build_features``.
PEPSTAB_MODULES = (
    "__init__.py", "augment.py", "data.py", "evaluation.py", "features.py",
    "mlp.py", "nested.py", "splits.py",
)


if __name__ == "__main__" and "--self-check" in sys.argv:
    raise SystemExit(self_check())

if __name__ == "__main__" and "--assemble" in sys.argv:
    _i = sys.argv.index("--assemble")
    _journal = Path(sys.argv[_i + 1])
    _threads = int(sys.argv[sys.argv.index("--threads") + 1]) \
        if "--threads" in sys.argv else DEFAULT_THREADS
    _suffix = sys.argv[sys.argv.index("--suffix") + 1] \
        if "--suffix" in sys.argv else "_linux"
    raise SystemExit(assemble(_journal, suffix=_suffix, threads=_threads))


import modal  # noqa: E402 -- below the free paths on purpose; see self_check()

app = modal.App("pepstab-phasea-a0-linux")

_base = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.3", "pandas==3.0.6", "scipy==1.18.1")
)


def _with_threads(image, threads: int):
    """Pin the BLAS thread count into the image environment.

    Belt; ``_run_chunk_body`` is braces. A ``--threads 1`` run of this app
    dispatched correctly to ``run_chunk_t1`` and still reported
    ``OMP_NUM_THREADS=2`` in all five containers, so this mechanism alone is
    not trusted here. Both are applied and the worker asserts they agree.
    """
    return image.env({v: str(threads) for v in THREAD_ENV_VARS})


def _add_modules(image):
    for module in PEPSTAB_MODULES:
        image = image.add_local_file(REPO / "pepstab" / module,
                                     f"/root/pepstab/{module}", copy=True)
    return image


#: One image per thread count. Two images rather than one parameterised
#: container, because the pin has to be baked before the interpreter starts.
IMAGES = {t: _add_modules(_with_threads(_base, t)) for t in THREAD_CHOICES}


def _container_memory() -> dict:
    """Peak memory this container reached. Telemetry must never fail a run."""
    import resource

    out: dict = {}
    for path in ("/sys/fs/cgroup/memory.peak",
                 "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            out["cgroup_peak_gib"] = round(
                int(Path(path).read_text().strip()) / 1024**3, 3)
            break
        except Exception:  # noqa: BLE001 -- telemetry must never fail a run
            continue
    try:
        out["self_max_rss_gib"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3)
    except Exception:  # noqa: BLE001
        pass
    return out


def _cpu_fingerprint() -> dict:
    """Which CPU and BLAS this container got. Telemetry must never fail a run.

    Recorded because two runs of the same five units at the same nominal
    settings came back **bit-identical on three folds and different on two**,
    by up to 1.9e-01 in prediction -- the magnitude of the cross-platform
    difference this app exists to measure. Containers land on heterogeneous
    hosts and OpenBLAS dispatches a kernel per microarchitecture, so the host
    is a run variable and has to be in the record to be ruled in or out.
    """
    out: dict = {}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                out["cpu_model"] = line.split(":", 1)[1].strip()
                break
        flags = [ln for ln in Path("/proc/cpuinfo").read_text().splitlines()
                 if ln.startswith("flags")]
        if flags:
            have = set(flags[0].split(":", 1)[1].split())
            out["simd"] = ",".join(sorted(
                f for f in ("avx", "avx2", "avx512f", "avx512dq") if f in have))
    except Exception:  # noqa: BLE001 -- telemetry must never fail a run
        pass
    try:
        import numpy as np

        cfg = getattr(np, "__config__", None)
        blas = cfg.get_info("blas_opt") if hasattr(cfg, "get_info") else {}
        out["numpy_blas"] = str(blas.get("libraries", ""))[:80]
        out["numpy_version"] = np.__version__
    except Exception:  # noqa: BLE001
        pass
    return out


def _run_chunk_body(units: list[dict], rows: dict, config: dict,
                    threads: int) -> list[dict]:
    """Shared body for the per-thread-count functions.

    **The thread pin is set here as well as in the image.** The image ``.env``
    is the intended mechanism and is what the A5/ESM arm relied on, but a
    ``--threads 1`` run of this app came back with ``OMP_NUM_THREADS=2`` in
    every container -- the dispatch was right (``run_chunk_t1`` ran) and the
    environment was not. Setting it here is reliable *only because* this module
    imports numpy nowhere at module scope, so the BLAS pool has not been sized
    yet when this line runs; that property is asserted below rather than
    assumed, because if numpy were already imported the assignment would be a
    no-op that looked like a fix.
    """
    sys.path.insert(0, "/root")
    env_before = os.environ.get("OMP_NUM_THREADS")
    numpy_preimported = "numpy" in sys.modules
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(threads)
    observed = int(os.environ.get("OMP_NUM_THREADS", "0"))
    if numpy_preimported and str(env_before) != str(threads):
        raise AssertionError(
            f"numpy was already imported with OMP_NUM_THREADS={env_before!r} "
            f"before this chunk could pin it to {threads}; the BLAS pool is "
            f"already sized and the pin would be cosmetic")
    fingerprint = _cpu_fingerprint()
    started = time.time()
    results: list[dict] = []
    for unit in units:
        t0 = time.time()
        try:
            result = run_unit(unit, rows, config)
        except Exception as exc:  # noqa: BLE001 -- recorded, not raised
            import traceback

            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc()[-2000:]}
        row = dict(result)
        # Every row names its fold, including a failure that never got far
        # enough to report one. The A1-A3 screen's outer_fold is null on every
        # row with the fold hiding in `fold` and in the key string; a consumer
        # grouping on the null field merges all five folds into one bucket and
        # gets a single plausible-looking number that is wrong.
        row.setdefault("key", unit.get("key"))
        row.setdefault("outer_fold", unit.get("outer_fold"))
        row.setdefault("inner_fold", unit.get("inner_fold"))
        row.setdefault("level", unit.get("level"))
        row.setdefault("ok", True)
        if row.get("outer_fold") is None:
            raise AssertionError(f"unit {unit.get('key')!r} produced a row "
                                 f"with no outer_fold")
        row["threads"] = threads
        row["threads_observed"] = observed
        row["thread_pin_ok"] = observed == threads
        row["threads_env_before_pin"] = env_before
        row["numpy_preimported"] = numpy_preimported
        row.update(fingerprint)
        row["worker_cpu"] = WORKER_CPU
        row["unit_wall_s"] = round(time.time() - t0, 3)
        results.append(row)

    wall = time.time() - started
    memory = _container_memory()
    ok = sum(1 for r in results if r.get("ok"))
    print(f"{len(results)} units, {ok} ok, {len(results) - ok} failed, "
          f"{wall:.1f}s wall; threads {observed} (env was {env_before}); "
          f"cpu {fingerprint.get('cpu_model', '?')}; memory {memory}")
    for row in results:
        row["chunk_wall_s"] = round(wall, 1)
        row.update(memory)
    return results


@app.function(image=IMAGES[1], cpu=WORKER_CPU, memory=WORKER_MEMORY,
              timeout=60 * 60, max_containers=MAX_CONTAINERS, retries=1)
def run_chunk_t1(units: list[dict], rows: dict, config: dict) -> list[dict]:
    """Units at BLAS threads=1, matching the shipped macOS A0's pinning."""
    return _run_chunk_body(units, rows, config, 1)


@app.function(image=IMAGES[2], cpu=WORKER_CPU, memory=WORKER_MEMORY,
              timeout=60 * 60, max_containers=MAX_CONTAINERS, retries=1)
def run_chunk_t2(units: list[dict], rows: dict, config: dict) -> list[dict]:
    """Units at BLAS threads=2, matching the screen and the A5/ESM arm."""
    return _run_chunk_body(units, rows, config, 2)


CHUNK_FUNCTIONS = {1: run_chunk_t1, 2: run_chunk_t2}


def _shape(threads: int, cpu: float, memory_gib: float, containers: int):
    """The thread count's own function, with this run's shape applied."""
    assert threads in CHUNK_FUNCTIONS, (
        f"--threads must be one of {sorted(CHUNK_FUNCTIONS)}, got {threads}")
    return CHUNK_FUNCTIONS[threads].with_options(
        cpu=cpu, memory=int(memory_gib * 1024), max_containers=containers)


def _assert_outer_folds_match() -> int:
    """The outer assignment must equal the file every Phase A consumer joins on.

    Loaded and compared, never recomputed into agreement, and checked on the
    client before any container starts: discovering a disagreement after the
    batch would mean paying twice.
    """
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(REPO))
    from pepstab import nested
    from pepstab.data import load_with_splits

    path = OUT_DIR / "phase_a_outer_folds.csv"
    if not path.exists():
        raise SystemExit(f"{path.relative_to(REPO)} does not exist")
    df = load_with_splits()
    train = df[df.split == "train"].sort_values("pair_id").reset_index(drop=True)
    published = pd.read_csv(path)
    if not np.array_equal(published.pair_id.to_numpy(),
                          train.pair_id.to_numpy()):
        raise SystemExit("phase_a_outer_folds.csv covers a different pair_id set")
    if not np.array_equal(published.outer_fold.to_numpy(),
                          nested.outer_folds(train)):
        raise SystemExit("outer fold assignment disagrees with the published file")
    print(f"outer folds == {path.relative_to(REPO)} for all {len(published):,} "
          f"rows (loaded and compared)")
    return len(published)


@app.local_entrypoint()
def smoke(profile: str = PROFILES[0], threads: int = DEFAULT_THREADS,
          cpu: float = WORKER_CPU, memory_gib: float = WORKER_MEMORY / 1024):
    """3 real units end to end, before any batch. Project invariant."""
    check_profile(profile)

    _assert_outer_folds_match()
    config = member_config()
    picked = smoke_units()
    assert 3 <= len(picked) <= 5

    rates = load_rates()
    worst = ((cpu * rates["cpu_hour"] + memory_gib * rates["mem_gib_hour"])
             * 3600 / 3600)
    print(f"smoke: {len(picked)} units on 1 container x {cpu:g} cores / "
          f"{memory_gib:g} GiB at threads={threads}, <= 60 min "
          f"=> at most ${worst:.4f}")

    rows = load_rows()
    started = time.time()
    results = _shape(threads, cpu, memory_gib, 1).remote(picked, rows, config)
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
        print(f"  UNPINNED {key}")

    per_unit = [float(r.get("unit_seconds") or r.get("unit_wall_s") or 0.0)
                for r in results]
    per_unit = [s for s in per_unit if s > 0]
    # All three readings are already GiB; no conversion may sit between them.
    peak = max([r.get("cgroup_peak_gib") or 0.0 for r in results]
               + [r.get("self_max_rss_gib") or 0.0 for r in results]
               + [r.get("peak_rss_gib") or 0.0 for r in results]
               ) if results else 0.0
    summary = {
        "profile": profile, "threads": threads, "n": len(results),
        "ok": len(results) - len(bad), "failed": len(bad),
        "keys": [r.get("key") for r in results],
        "n_members_per_unit": sorted({r.get("n_members") for r in results
                                      if r.get("ok")}),
        # Max, not mean: the batch's wall is set by its slowest container and a
        # 3-unit sample has no business claiming a tighter statistic.
        "seconds_per_unit": round(max(per_unit), 3) if per_unit else None,
        "seconds_per_unit_median": (
            round(sorted(per_unit)[len(per_unit) // 2], 3) if per_unit else None),
        "client_wall_s": round(wall, 1), "peak_gib": peak,
        "worker_memory_mib": WORKER_MEMORY,
        "scores": {r.get("key"): r.get("median_per_allele_spearman")
                   for r in results if r.get("ok")},
        "all_ok": not bad and not nulls and not unpinned,
    }
    out = OUT_DIR / f"phase_a_a0_smoke_{profile}_t{threads}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
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
        print(f"WARNING: peak {peak:.2f} GiB against "
              f"{WORKER_MEMORY / 1024:.1f} GiB. Raise --memory-gib.")
    print(f"smoke passed: {summary['seconds_per_unit']:.1f}s/unit, "
          f"{peak:.2f} GiB peak.")
    print(f"Next: ::forecast --measured-from {out.relative_to(REPO)}")


@app.local_entrypoint()
def forecast(profile: str = PROFILES[0], measured_from: str = "",
             seconds_per_unit: float = 0.0, levels: str = "2,3",
             cpu: float = WORKER_CPU,
             memory_gib: float = WORKER_MEMORY / 1024,
             containers: int = MAX_CONTAINERS, chunk: int = DEFAULT_CHUNK):
    """Price the batch from a measured per-unit time. Refuses to guess one."""
    check_profile(profile)
    if measured_from:
        seconds_per_unit = seconds_from_smoke(Path(measured_from))
    if not seconds_per_unit:
        raise SystemExit(
            "::forecast needs --measured-from or an explicit "
            "--seconds-per-unit. There is no default on purpose: a forecast "
            "on a placeholder is how this project once reported a cost 4.6x "
            "wrong.")
    wanted = tuple(int(x) for x in levels.split(","))
    units = enumerate_units(wanted)
    f = forecast_numbers(len(units), seconds_per_unit, cpu, memory_gib,
                         containers, chunk)
    print_forecast(f, f"A0 Linux batch: {len(units)} units, levels {wanted}",
                   measured=bool(measured_from))
    print("\nLog this in reports/compute_ledger.csv (append only) before ::run.")


@app.local_entrypoint()
def run(profile: str = PROFILES[0], threads: int = DEFAULT_THREADS,
        levels: str = "2,3", measured_from: str = "",
        seconds_per_unit: float = 0.0, cpu: float = WORKER_CPU,
        memory_gib: float = WORKER_MEMORY / 1024,
        containers: int = MAX_CONTAINERS, chunk: int = DEFAULT_CHUNK,
        dry_run: bool = False, resume: bool = False):
    """The batch, journalled as it lands.

    Every unit's **full return** -- predictions included -- is written with
    ``write`` / ``flush`` / ``os.fsync`` before anything else happens. Rows
    accumulate on the *client*, and the A1-A4 screen lost its gRPC connection
    mid-``.map()`` twice, once capturing 8 units of 3,360; ``--detach`` keeps
    containers alive but nothing collects their results. ``--resume`` rebuilds
    from the journal alone.

    The journal name carries the thread count, so the threads=1 decomposition
    arm cannot resume from or overwrite the threads=2 batch.
    """
    check_profile(profile)
    _assert_outer_folds_match()
    config = member_config()
    wanted = tuple(int(x) for x in levels.split(","))
    units = enumerate_units(wanted)
    assert units, f"no units for levels {wanted}"

    if measured_from:
        seconds_per_unit = seconds_from_smoke(Path(measured_from))
    if seconds_per_unit:
        print_forecast(forecast_numbers(len(units), seconds_per_unit, cpu,
                                        memory_gib, containers, chunk),
                       f"A0 Linux batch: {len(units)} units at threads={threads}",
                       measured=bool(measured_from))
    else:
        print("no measured per-unit time: running unforecast.")

    if dry_run:
        print(f"\ndry run: {len(units)} units planned. No remote call was "
              f"made.\n  Not free: `modal run` built and validated the image "
              f"first. For a forecast that cannot start anything, use "
              f"--self-check.")
        return

    partial = OUT_DIR / f"phase_a_a0_runs_{profile}_t{threads}.partial.jsonl"
    done: dict[str, dict] = {}
    if resume and partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("ok"):
                    done[row.get("key", "")] = row
        units = [u for u in units if u["key"] not in done]
        print(f"resuming: {len(done)} units already journalled, "
              f"{len(units)} left")
    elif partial.exists():
        raise SystemExit(
            f"{partial.relative_to(REPO)} already exists. Pass --resume to "
            f"continue it, or delete it to recompute from scratch -- silently "
            f"rescoring paid-for units is not a safe default.")

    rows = load_rows()
    chunks = [units[i:i + chunk] for i in range(0, len(units), chunk)]
    fn = _shape(threads, cpu, memory_gib, containers)
    print(f"{len(units)} units in {len(chunks)} chunks, {containers} "
          f"containers x {cpu:g} cores x {threads} BLAS threads, CPU only")

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
    print(json.dumps({"profile": profile, "threads": threads,
                      "units": len(results), "ok": ok,
                      "failed": len(results) - ok,
                      "unpinned_threads": sum(
                          1 for r in results if not r.get("thread_pin_ok")),
                      "wall_min": round(wall / 60, 1)}, indent=2))
    _write_runs_csv(results,
                    OUT_DIR / f"phase_a_a0_unit_runs_{profile}_t{threads}.csv")
    print(f"\njournal {partial.relative_to(REPO)} kept. Assemble with:\n"
          f"  .venv/bin/python modal_app/phase_a_a0_modal.py --assemble "
          f"{partial.relative_to(REPO)} --threads {threads}")


def _write_runs_csv(results: list[dict], path: Path) -> None:
    """One row per unit, union of keys, in first-seen order.

    ``restval=""`` because a failed unit carries ``error`` and no metrics while
    a level-2 unit carries ``inner_oof_6`` and a level-3 one does not;
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
