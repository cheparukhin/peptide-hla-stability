"""Phase A arm A0: the sequence reference, refitted under the nested protocol.

    .venv/bin/python scripts/phase_a_a0.py
    .venv/bin/python scripts/phase_a_a0.py --smoke          # 1 outer fold, writes nothing
    .venv/bin/python scripts/phase_a_a0.py --out-dir /tmp/a0

Not a new model. This refits ``seq_ensemble_pep_pseudo`` — peptide +
pseudosequence, one-hot and BLOSUM, the stage 2 per-encoding ``l2`` values — on
the training-only nested cluster folds of
``reports/further_research/phase_a_prespecification.md`` §1, and emits the §A0
out-of-fold prediction contract that Phase B, D and A5 consume.

Three prediction roles, from two nesting levels:

- ``outer_held`` — for outer fold *f*, the mean of 30 members (5 inner folds x 2
  encodings x 3 seeds) fitted on the other four outer folds, each early-stopping
  on its own inner fold and so fitted on ~12,618 rows. One per ``pair_id``.
- ``inner_oof_6`` — the mean of the 6 members (2 encodings x 3 seeds) that
  early-stopped on the inner fold holding that row. Free: the same members'
  own held-out-inner-fold predictions. Four per ``pair_id``.
- ``inner_oof_30`` — an arity-matched 30-member mean for each inner fold, which
  needs a third nesting level: the inner fitting partition is split into five
  sub-folds, giving 5 x 2 x 3 = 30 members fitted on ~10,094 rows. Four per
  ``pair_id``.

``inner_oof_30`` exists because A5 fits its blend weights, and Phase D its
residual model, on inner out-of-fold predictions and then applies them to
``outer_held``. Fitting on a 6-member error distribution and applying to a
30-member one mismatches ensemble arity, which is worth roughly +0.07 to +0.09
median Spearman here — larger than most effects Phase A is looking for. See
prespecification §A0 "Why ``inner_oof_30`` is emitted" and the 4 October 2026
entry in §7. ``inner_oof_6`` is still emitted so the ratio between the two
inner residual scales can be measured rather than argued about.

**What this does not fix:** ``inner_oof_30`` members see ~10,094 fit rows against
``outer_held``'s ~12,618, because each nesting level removes another fifth. A
large systematic *arity* mismatch has been traded for a small *training-size*
mismatch. That remainder is irreducible in nested cross-validation and is
reported as a limitation, not claimed away.

The published reference figures (validation median per-allele Spearman 0.6931,
test 0.7064) are **not** reproduced here. Those are frozen-split numbers and
Phase A scores neither split. What reconciles is the machinery — row counts,
fold membership, ensemble arity, seed protocol and metric code.

Features are rebuilt inside each fitting partition. ``build_features`` is
row-local so this costs nothing, but the discipline is the point: no transform
is fitted across a fold boundary.
"""

from __future__ import annotations

import os

# Pin BLAS to one thread *before* numpy is imported -- the thread pool is sized
# at import time, so setting these afterwards does nothing. Thread count changes
# float32 results here: the same seed at a different thread count moves a single
# network by roughly 0.02 median Spearman, so it is pinned and recorded in the
# provenance file. VECLIB_MAXIMUM_THREADS is the one that matters on macOS,
# where numpy links against Accelerate; the others cover Linux/CI.
BLAS_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for _var in BLAS_THREAD_VARS:
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab import nested  # noqa: E402
from pepstab.data import RAW_CSV, SPLITS_CSV, TARGET_RAW, load_with_splits  # noqa: E402
from pepstab.features import ENCODINGS, build_features  # noqa: E402
from scripts.baseline_ensemble import SELECTED  # noqa: E402
from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE, SEEDS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "reports" / "further_research"

#: A0 is the reference reproduction, so the input set is not an option.
INPUT_SET = "pep_pseudo"

#: Expected training rows, from docs/README.md. Asserted rather than trusted:
#: the §A0 row-count identities are absolute, so a silent change in the frozen
#: splits would otherwise ship inside the prediction file.
N_TRAIN_ROWS = 19_716

#: ``inner_fold`` for a row that is outer-held in that outer fold, in
#: ``phase_a_fold_assignments.csv``. Those rows occupy no inner fold because
#: they are not in the fitting partition at all.
OUTER_HELD_INNER_FOLD = -1

#: ``sub_fold`` for a member of the second nesting level, which has no sub-fold.
NO_SUB_FOLD = -1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_no_overlap(label: str, fit_pairs: np.ndarray,
                      predicted_pairs: np.ndarray) -> None:
    """No member may be fitted on a row it predicts."""
    overlap = set(fit_pairs.tolist()) & set(predicted_pairs.tolist())
    if overlap:
        raise AssertionError(
            f"{label}: {len(overlap)} pair_id(s) are both fitted and predicted, "
            f"first {sorted(overlap)[:5]}")


def fit_outer_fold(split: nested.OuterSplit, train: pd.DataFrame, n_sub: int,
                   seeds: tuple[int, ...], selected: dict
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], int]:
    """Both nesting levels for one outer fold.

    Returns the 30-member ``outer_held`` mean, the ``inner_oof_6`` mean, the
    ``inner_oof_30`` mean, the member records, and the number of
    fit/predict disjointness checks performed.
    """
    fit_rows = train[split.fit_mask].reset_index(drop=True)
    held_rows = train[split.held_mask].reset_index(drop=True)
    fit_pairs = fit_rows.pair_id.to_numpy()
    held_pairs = held_rows.pair_id.to_numpy()
    n_inner = int(split.inner.max()) + 1
    runs: list[dict] = []
    checks = 0

    # Level 2: the arity-matched reference ensemble. Each member stops on its
    # own inner fold, so this is the deployed reference's exact recipe.
    for k in range(n_inner):
        is_fit = split.inner != k
        assert_no_overlap(f"outer {split.fold} / inner {k}", fit_pairs[is_fit],
                          np.concatenate([held_pairs, fit_pairs[~is_fit]]))
        checks += 1
    level2 = nested.fit_mlp_ensemble(fit_rows, [held_rows], split.inner,
                                     ENCODINGS, seeds, selected,
                                     input_set=INPUT_SET)
    runs.extend({"outer_fold": split.fold, "sub_fold": NO_SUB_FOLD, **record}
                for record in level2["members"])

    # Level 3: a full 30-member mean per inner fold, so a consumer fitting on
    # inner residuals and applying to outer_held is not crossing an arity
    # boundary. One more nesting level, one more fifth of the rows removed.
    inner30 = np.full(len(fit_rows), np.nan)
    for k in range(n_inner):
        is_fit = split.inner != k
        sub_fit = fit_rows[is_fit].reset_index(drop=True)
        target = fit_rows[~is_fit].reset_index(drop=True)
        sub = nested.inner_folds(sub_fit, n_sub)
        sub_pairs = sub_fit.pair_id.to_numpy()
        for s in range(n_sub):
            assert_no_overlap(f"outer {split.fold} / inner {k} / sub {s}",
                              sub_pairs[sub != s],
                              np.concatenate([target.pair_id.to_numpy(),
                                              sub_pairs[sub == s]]))
            checks += 1
        level3 = nested.fit_mlp_ensemble(sub_fit, [target], sub, ENCODINGS,
                                         seeds, selected, input_set=INPUT_SET)
        inner30[~is_fit] = level3["predictions"][0]
        runs.extend({"outer_fold": split.fold, "inner_fold": k,
                     "sub_fold": record.pop("inner_fold"), **record}
                    for record in level3["members"])

    if not np.isfinite(inner30).all():
        raise AssertionError(f"outer fold {split.fold}: some fitting-partition "
                             "row received no inner_oof_30 prediction")
    return level2["predictions"][0], level2["inner_oof"], inner30, runs, checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help="where the five §A0 contract files are written")
    ap.add_argument("--smoke", action="store_true",
                    help="one outer fold at 2 inner folds, 2 sub-folds and 1 "
                         "seed; prints the fold score and writes nothing. Per "
                         "CLAUDE.md, the end-to-end check before the full run.")
    args = ap.parse_args()

    n_inner = 2 if args.smoke else nested.ASSESS_INNER_FOLDS
    n_sub = 2 if args.smoke else nested.ASSESS_INNER_FOLDS
    seeds = SEEDS[:1] if args.smoke else SEEDS

    # The reference member recipe lives in nested.REFERENCE_MLP so other arms
    # can inherit it; stage 2's own constants are the authority on the two the
    # grid set, so a drift between them is caught here rather than reported.
    if (nested.REFERENCE_MLP.max_epochs, nested.REFERENCE_MLP.patience) != (
            MAX_EPOCHS, PATIENCE):
        raise AssertionError(
            f"nested.REFERENCE_MLP has max_epochs/patience "
            f"{nested.REFERENCE_MLP.max_epochs}/{nested.REFERENCE_MLP.patience}, "
            f"but stage 2 selected {MAX_EPOCHS}/{PATIENCE}")
    selected = {enc: SELECTED[(INPUT_SET, enc)] for enc in ENCODINGS}

    df = load_with_splits()
    train = df[df.split == "train"].sort_values("pair_id").reset_index(drop=True)
    if len(train) != N_TRAIN_ROWS:
        raise AssertionError(f"training split has {len(train):,} rows, "
                             f"expected {N_TRAIN_ROWS:,}")

    per_outer_l2 = n_inner * len(ENCODINGS) * len(seeds)
    per_outer_l3 = n_inner * n_sub * len(ENCODINGS) * len(seeds)
    n_outer = 1 if args.smoke else nested.OUTER_FOLDS
    n_networks = n_outer * (per_outer_l2 + per_outer_l3)
    hidden_oh, l2_oh = selected["onehot"]
    _, l2_bl = selected["blosum"]

    print(f"A0: {INPUT_SET} reference under training-only nested cluster folds")
    print(f"train={len(train):,} rows, {train.cluster_id.nunique():,} peptide clusters")
    print(f"outer: {n_outer} cluster folds (= baseline_ensemble.cv_folds)")
    print(f"level 2 (outer_held, inner_oof_6): {n_inner} inner x "
          f"{len(ENCODINGS)} enc x {len(seeds)} seeds = {per_outer_l2} members/fold")
    print(f"level 3 (inner_oof_30): {n_inner} inner x {n_sub} sub x "
          f"{len(ENCODINGS)} enc x {len(seeds)} seeds = {per_outer_l3} members/fold")
    print(f"total networks: {n_networks}")
    print(f"member: hidden={hidden_oh} lr={nested.REFERENCE_MLP.lr:g} "
          f"batch={nested.REFERENCE_MLP.batch_size} max_epochs={MAX_EPOCHS} "
          f"patience={PATIENCE}, l2={l2_oh:g} (onehot) / {l2_bl:g} (blosum)")
    print(f"BLAS threads pinned to {os.environ['OMP_NUM_THREADS']}\n")

    outer_fold = nested.outer_folds(train)
    splits = list(nested.iter_outer(train, n_inner))
    if args.smoke:
        splits = splits[:1]

    # Phase B is blocked on the fold assignments and they need no models, so
    # they are written before the fitting loop rather than after it.
    assignment_frames = []
    for split in splits:
        inner_column = np.full(len(train), OUTER_HELD_INNER_FOLD, dtype=int)
        inner_column[split.fit_mask] = split.inner
        assignment_frames.append(pd.DataFrame({
            "pair_id": train.pair_id.to_numpy(), "outer_fold": split.fold,
            "inner_fold": inner_column}))
    assignments = pd.concat(assignment_frames, ignore_index=True).sort_values(
        ["pair_id", "outer_fold"]).reset_index(drop=True)
    fold_table = pd.DataFrame({"pair_id": train.pair_id.to_numpy(),
                               "outer_fold": outer_fold})

    if not args.smoke:
        expected_assignments = N_TRAIN_ROWS * nested.OUTER_FOLDS
        if len(assignments) != expected_assignments:
            raise AssertionError(f"fold assignments have {len(assignments):,} "
                                 f"rows, expected {expected_assignments:,}")
        held_count = assignments.groupby("pair_id").inner_fold.apply(
            lambda col: int((col == OUTER_HELD_INNER_FOLD).sum()))
        if not (held_count == 1).all():
            raise AssertionError("every pair_id must be outer-held in exactly "
                                 "one of the five outer folds")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        fold_table.to_csv(args.out_dir / "phase_a_outer_folds.csv", index=False)
        assignments.to_csv(args.out_dir / "phase_a_fold_assignments.csv", index=False)
        print(f"wrote phase_a_outer_folds.csv ({len(fold_table):,} rows) and "
              f"phase_a_fold_assignments.csv ({len(assignments):,} rows) "
              f"before fitting\n")

    held_frames: list[pd.DataFrame] = []
    inner_frames: list[pd.DataFrame] = []
    runs: list[dict] = []
    fold_scores: list[dict] = []
    n_checks = 0

    started = time.perf_counter()
    for split in splits:
        t0 = time.perf_counter()
        held_mean, inner6, inner30, fold_runs, checks = fit_outer_fold(
            split, train, n_sub, seeds, selected)
        runs.extend(fold_runs)
        n_checks += checks

        held_rows = train[split.held_mask]
        y_held = held_rows.y_log1p.to_numpy()
        scores = nested.score_fold(f"a0_outer{split.fold}", held_rows.allele,
                                   y_held, held_mean)
        coverage = nested.fold_coverage(held_rows.allele, y_held)
        fold_scores.append({"outer_fold": split.fold, **coverage,
                            "median_spearman": scores.median_spearman,
                            "iqr_low": scores.iqr_spearman[0],
                            "iqr_high": scores.iqr_spearman[1],
                            "mae_log1p": scores.mae_log1p,
                            "n_alleles_unranked": scores.n_alleles_unranked})
        print(f"  fold {split.fold}: {int(split.held_mask.sum()):,} held-out rows, "
              f"{len(fold_runs)} members, median SCC {scores.median_spearman:.4f} "
              f"[IQR {scores.iqr_spearman[0]:.3f}, {scores.iqr_spearman[1]:.3f}], "
              f"MAE {scores.mae_log1p:.3f}, "
              f"{coverage['n_alleles_eligible']} eligible alleles covering "
              f"{coverage['eligible_row_coverage']:.1%} of rows, "
              f"{(time.perf_counter() - t0) / 60:.1f} min")

        fit_pairs = train.loc[split.fit_mask, "pair_id"].to_numpy()
        held_frames.append(pd.DataFrame({
            "pair_id": held_rows.pair_id.to_numpy(), "outer_fold": split.fold,
            "role": "outer_held", "y_pred": held_mean}))
        for role, values in (("inner_oof_6", inner6), ("inner_oof_30", inner30)):
            inner_frames.append(pd.DataFrame({
                "pair_id": fit_pairs, "outer_fold": split.fold,
                "role": role, "y_pred": values}))
    wall = time.perf_counter() - started

    folds = pd.DataFrame(fold_scores)
    medians = folds.median_spearman.to_numpy()
    runs_frame = pd.DataFrame(runs)

    print("\nper-outer-fold scores (median per-allele Spearman on the "
          "cluster-held-out fold):")
    print(folds.round(4).to_string(index=False))
    print(f"\nfive-fold median of medians: {float(np.median(medians)):.4f}  "
          f"(min {medians.min():.4f}, max {medians.max():.4f}, "
          f"spread {medians.max() - medians.min():.4f})")
    print(f"mean of fold medians:        {float(medians.mean()):.4f}")
    print(f"eligible-row coverage:       "
          f"{folds.eligible_row_coverage.min():.1%}-"
          f"{folds.eligible_row_coverage.max():.1%} per fold, "
          f"{folds.n_alleles_eligible.min()}-{folds.n_alleles_eligible.max()} "
          f"alleles of 75")
    print("\nThese are cluster-held-out training folds, not the frozen "
          "validation split.\nThe reference's published 0.6931 validation figure "
          "is a different partition\nand is not the comparison "
          "(prespecification §A0).")

    by_level = runs_frame.assign(
        level=np.where(runs_frame.sub_fold == NO_SUB_FOLD, "level2", "level3"))
    print("\nper-member fit rows and stopping, by nesting level:")
    print(by_level.groupby("level").agg(
        members=("seed", "size"), n_fit_rows=("n_fit_rows", "median"),
        best_epoch=("best_epoch", "median"), dev_mse=("dev_mse", "median"),
        fit_seconds=("fit_seconds", "median")).round(4).to_string())
    at_cap = int((runs_frame.best_epoch >= MAX_EPOCHS).sum())
    print(f"members stopping on the {MAX_EPOCHS}-epoch cap: {at_cap}")
    print(f"\ntotal wall {wall / 60:.1f} min for {n_networks} networks "
          f"({wall / n_networks:.1f}s per network)")

    if args.smoke:
        print("\n--smoke: nothing written. Rerun without --smoke for the "
              "full contract.")
        return 0

    oof = pd.concat(held_frames + inner_frames, ignore_index=True)
    oof = oof.sort_values(["pair_id", "role", "outer_fold"]).reset_index(drop=True)

    # Assert before writing. A broken contract file is worse than no file: Phase
    # B, D and A5 join on these columns and would inherit the error silently.
    checks_passed = [
        f"phase_a_outer_folds.csv = {len(fold_table):,} rows",
        f"phase_a_fold_assignments.csv = {len(assignments):,} rows "
        f"= 5 x 19,716; every pair_id outer-held in exactly one outer fold",
    ]

    expected_oof = N_TRAIN_ROWS * 9
    if len(oof) != expected_oof:
        raise AssertionError(f"OOF table has {len(oof):,} rows, "
                             f"expected {expected_oof:,}")
    checks_passed.append(
        f"phase_a_seq_oof_cluster.csv = {len(oof):,} rows = 19,716 outer_held "
        f"+ 4 x 19,716 inner_oof_6 + 4 x 19,716 inner_oof_30")

    if not np.isfinite(oof.y_pred.to_numpy()).all():
        bad = int((~np.isfinite(oof.y_pred.to_numpy())).sum())
        raise AssertionError(f"{bad:,} non-finite prediction(s)")
    checks_passed.append(f"np.isfinite on all {len(oof):,} predictions")

    per_role = oof.pivot_table(index="pair_id", columns="role", values="y_pred",
                               aggfunc="size")
    for role, want in (("outer_held", 1), ("inner_oof_6", 4), ("inner_oof_30", 4)):
        if not (per_role[role] == want).all():
            raise AssertionError(f"some pair_id does not have exactly {want} "
                                 f"{role} row(s)")
    if len(per_role) != N_TRAIN_ROWS:
        raise AssertionError(f"OOF table covers {len(per_role):,} pair_ids, "
                             f"expected the whole training split")
    checks_passed.append("every pair_id: exactly 1 outer_held, 4 inner_oof_6, "
                         "4 inner_oof_30")
    checks_passed.append(f"out-of-fold coverage = the training split exactly "
                         f"({len(per_role):,} pair_ids)")

    held_only = oof[oof.role == "outer_held"].set_index("pair_id")
    if not np.array_equal(held_only.outer_fold.reindex(fold_table.pair_id).to_numpy(),
                          outer_fold):
        raise AssertionError("outer_held fold ids disagree with the fold table")
    checks_passed.append("outer_held fold id == phase_a_outer_folds.csv for every row")

    straddle = oof[oof.role != "outer_held"].merge(
        fold_table, on="pair_id", suffixes=("", "_own"))
    if (straddle.outer_fold == straddle.outer_fold_own).any():
        raise AssertionError("an inner_oof row is attributed to the outer fold "
                             "that held it out, which never fitted it")
    checks_passed.append("no inner_oof row is attributed to its own outer "
                         "held-out fold")

    # Cross-check the prediction roles against the fold assignment file, which
    # was written before any model existed: an inner_oof row's outer fold must
    # be one where it occupied a real inner fold.
    occupied = assignments[assignments.inner_fold != OUTER_HELD_INNER_FOLD]
    inner_keys = set(map(tuple, straddle[["pair_id", "outer_fold"]].to_numpy()))
    if inner_keys != set(map(tuple, occupied[["pair_id", "outer_fold"]].to_numpy())):
        raise AssertionError("inner_oof (pair_id, outer_fold) keys disagree with "
                             "phase_a_fold_assignments.csv")
    checks_passed.append("inner_oof keys == the fitting-partition rows of "
                         "phase_a_fold_assignments.csv")

    expected_l2 = nested.OUTER_FOLDS * per_outer_l2
    expected_l3 = nested.OUTER_FOLDS * per_outer_l3
    counts = (int((runs_frame.sub_fold == NO_SUB_FOLD).sum()),
              int((runs_frame.sub_fold != NO_SUB_FOLD).sum()))
    if counts != (expected_l2, expected_l3):
        raise AssertionError(f"member counts {counts}, expected "
                             f"({expected_l2}, {expected_l3})")
    checks_passed.append(f"members: {expected_l2} level-2 + {expected_l3} "
                         f"level-3 = {len(runs_frame)}")
    checks_passed.append(f"{n_checks} fit/predict disjointness checks, covering "
                         f"every one of the {len(runs_frame)} members")

    runs_frame = runs_frame[["outer_fold", "inner_fold", "sub_fold", "encoding",
                             "seed", "n_fit_rows", "best_epoch", "dev_mse",
                             "fit_seconds"]]
    oof.to_csv(args.out_dir / "phase_a_seq_oof_cluster.csv", index=False)
    runs_frame.to_csv(args.out_dir / "phase_a_a0_runs.csv", index=False)

    # The floor event is the exact integer-valued 0 the assay records, not a
    # 0.1 h threshold: 443 rows sit at exactly 0.1 h, so a threshold there
    # swings the floor class by 1.57 percentage points on strict-versus-
    # non-strict alone. Prespecification §7, 4 October 2026.
    n_floor = int((train[TARGET_RAW] == 0).sum())

    provenance = {
        "arm": "A0",
        "description": ("seq_ensemble_pep_pseudo refitted under the Phase A "
                        "training-only nested cluster protocol"),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_set": INPUT_SET,
        "outer_folds": nested.OUTER_FOLDS,
        "inner_folds": n_inner,
        "sub_folds": n_sub,
        "members_per_outer_fold": {"level2_outer_held_and_inner_oof_6": per_outer_l2,
                                   "level3_inner_oof_30": per_outer_l3},
        "networks_total": len(runs_frame),
        "seeds": list(seeds),
        "encoding_axis": list(ENCODINGS),
        "member_config": {
            "hidden": list(nested.REFERENCE_MLP.hidden),
            "lr": nested.REFERENCE_MLP.lr,
            "batch_size": nested.REFERENCE_MLP.batch_size,
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "l2_by_encoding": {"onehot": l2_oh, "blosum": l2_bl},
            "source": "scripts.baseline_ensemble.SELECTED, selected at stage 2",
        },
        "fit_rows_by_level": {
            "outer_held": int(by_level.loc[by_level.level == "level2",
                                           "n_fit_rows"].median()),
            "inner_oof_30": int(by_level.loc[by_level.level == "level3",
                                             "n_fit_rows"].median()),
            "note": ("the irreducible remainder of nested CV: each level "
                     "removes another fifth, so inner_oof_30 members are "
                     "fitted on fewer rows than outer_held members. A large "
                     "arity mismatch is traded for a small training-size "
                     "mismatch, not eliminated."),
        },
        "min_rows_per_allele_fold": nested.MIN_ROWS_PER_ALLELE_FOLD,
        "floor_event": {"definition": "thalf_hours == 0", "n_train_rows": n_floor,
                        "share": round(n_floor / len(train), 6)},
        "blas_threads": {var: os.environ[var] for var in BLAS_THREAD_VARS},
        "package_versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "input_digests": {
            "data/splits.csv": sha256(SPLITS_CSV),
            "data/rasmussen_et_al_dataset.csv": sha256(RAW_CSV),
        },
        "rows": {"train": len(train), "outer_folds_csv": len(fold_table),
                 "fold_assignments_csv": len(assignments), "oof_csv": len(oof)},
        "fold_scores": folds.to_dict(orient="records"),
        "five_fold_median_of_medians": float(np.median(medians)),
        "wall_seconds": round(wall, 1),
        # The full member map, so a consumer can verify the arity it was handed
        # rather than assuming it.
        "members": runs_frame.to_dict(orient="records"),
        "asserted": checks_passed,
    }
    (args.out_dir / "phase_a_a0_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")

    print("\nassertions passed before writing:")
    for check in checks_passed:
        print(f"  - {check}")
    rel = (args.out_dir.relative_to(REPO_ROOT)
           if args.out_dir.is_relative_to(REPO_ROOT) else args.out_dir)
    print(f"\nwrote {rel}/: phase_a_outer_folds.csv, "
          f"phase_a_fold_assignments.csv, phase_a_seq_oof_cluster.csv, "
          f"phase_a_a0_runs.csv, phase_a_a0_provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
