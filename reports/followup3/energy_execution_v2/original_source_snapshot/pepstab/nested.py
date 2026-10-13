"""Nested cluster cross-validation inside the training split, for Phase A.

Phase A assesses arms on the **training split only** — validation has been
consulted across stages 2-7 and test is spent — so it needs an outer assessment
partition and an inner selection partition cut from ``train`` itself. See
``reports/further_research/phase_a_prespecification.md`` §1.

**Why the outer level is cluster-grouped ``cv_folds`` and not
:func:`pepstab.stage6.nested_folds`.** Both exist, and they are built for
opposite properties. ``stage6.nested_folds`` partitions by *peptide* and is
deliberately constructed so that the two members of a point-mutant pair land in
**different** folds — that is exactly what its mutant-ranking diagnostic needs,
and it is the wrong property here, because it would let a model early-stop on,
or be assessed on, a peptide one or two substitutions from one it fitted. This
module's folds move whole ``cluster_id`` groups, which are single-linkage
components at Hamming <= 3, so every held-out peptide stays >= 4 substitutions
from every peptide its model saw — the same guarantee the frozen splits give
between train and test. ``scripts/esm_arm.py::mode_oof`` uses ``nested_folds``
for its outer level; Phase A departs from that recipe on this point on purpose.

The water-fill itself is not reimplemented: both levels go through
:func:`pepstab.splits.assign_clusters`, the primitive
``scripts.baseline_ensemble.cv_folds`` calls, and
:func:`outer_folds` is asserted equal to ``cv_folds(train, 5)`` in
``tests/test_phase_a_nested.py``. The metric contract is likewise reused rather
than duplicated: :func:`score_fold` and :func:`fold_eligible_alleles` delegate
to :mod:`pepstab.evaluation`, so the tie rule, the
:data:`~pepstab.evaluation.UNRANKED_CONTRIBUTION` rule and
:func:`~pepstab.evaluation.validate_finite` are the frozen stage 1 ones.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import evaluation
from .evaluation import Scores
from .features import build_features
from .mlp import MLPConfig, MLPRegressor
from .splits import assign_clusters

#: Outer assessment folds over training peptide clusters. Five, per plan §3.2,
#: which is what ``scripts.baseline_ensemble.cv_folds`` already cuts.
OUTER_FOLDS = 5

#: Inner folds during a screen, per plan §3.3's screening budget.
SCREEN_INNER_FOLDS = 3

#: Inner folds at outer assessment. Five gives ``5 * 2 encodings * 3 seeds =
#: 30`` members per outer fold, the deployed reference ensemble's arity.
#: Ensembling is worth roughly +0.074 median Spearman here, larger than most
#: effects Phase A looks for, so an arm assessed at 18 members against a
#: 30-member reference would be measuring ensemble size.
ASSESS_INNER_FOLDS = 5

#: The deployed reference member, held identical across every MLP arm that
#: claims parity with it (prespecification §A0). ``hidden`` and ``l2`` are
#: overridden per encoding from the stage 2 selection; the rest is fixed. The
#: epoch cap is 300 because at 120 the slowest stage 2 arm still had runs
#: stopping on the cap, so a lower score could not be told apart from
#: undertraining. ``seed`` is a placeholder, always replaced.
REFERENCE_MLP = MLPConfig(hidden=(256, 64), lr=1e-3, batch_size=256,
                          max_epochs=300, patience=15)

#: Rows an allele needs on a single fold to enter that fold's panel. This is
#: ``evaluation.MIN_ROWS_BY_SPLIT["val"]`` and it transfers here on size: an
#: outer fold holds ~3,943 rows and an inner fold ~5,258 against validation's
#: 2,817. The test split's 50-row bar does not transfer. Frozen in
#: prespecification §2 before any Phase A fit.
MIN_ROWS_PER_ALLELE_FOLD = 20


def _cluster_folds(frame: pd.DataFrame, n_folds: int) -> np.ndarray:
    """``n_folds`` equal-fraction folds over ``frame``'s own peptide clusters.

    The body of ``scripts.baseline_ensemble.cv_folds``, expressed once so the
    outer and inner levels cannot drift apart. ``assign_clusters`` does the
    water-filling and is deterministic with no seed.
    """
    names = tuple(f"f{k}" for k in range(n_folds))
    placement = assign_clusters(frame.groupby("cluster_id").size(),
                                fractions=(1.0 / n_folds,) * n_folds, names=names)
    return frame["cluster_id"].map(placement).map(
        {name: k for k, name in enumerate(names)}).to_numpy()


def outer_folds(train: pd.DataFrame) -> np.ndarray:
    """Five whole-cluster outer folds over the training split.

    Ints ``0..4`` aligned to ``train``'s row order. Equal to
    ``scripts.baseline_ensemble.cv_folds(train, 5)`` by construction; the
    equality is asserted in ``tests/test_phase_a_nested.py`` rather than
    assumed, because Phase B and D consume these fold ids.
    """
    return _cluster_folds(train, OUTER_FOLDS)


def inner_folds(fit_rows: pd.DataFrame, n_folds: int) -> np.ndarray:
    """``n_folds`` whole-cluster folds inside one outer fitting partition.

    Ints ``0..n_folds-1`` aligned to ``fit_rows``'s row order. Built from the
    fitting partition's **own** clusters, so the outer held-out fold's clusters
    never enter the inner water-fill.
    """
    return _cluster_folds(fit_rows, n_folds)


@dataclass(frozen=True)
class OuterSplit:
    """One outer fold: who fits, who is assessed, and the inner cut inside."""

    fold: int
    fit_mask: np.ndarray
    held_mask: np.ndarray
    inner: np.ndarray


def iter_outer(train: pd.DataFrame, n_inner: int) -> Iterator[OuterSplit]:
    """Yield the five outer splits, each carrying its own inner fold vector."""
    outer = outer_folds(train)
    for fold in range(OUTER_FOLDS):
        held_mask = outer == fold
        fit_mask = ~held_mask
        yield OuterSplit(fold=fold, fit_mask=fit_mask, held_mask=held_mask,
                         inner=inner_folds(train[fit_mask], n_inner))


def fit_mlp_ensemble(fit_df: pd.DataFrame, predict_dfs: Sequence[pd.DataFrame],
                     inner: np.ndarray, encodings: Sequence[str],
                     seeds: Sequence[int],
                     selected: Mapping[str, tuple[tuple[int, ...], float]],
                     input_set: str = "pep_pseudo",
                     base_config: MLPConfig = REFERENCE_MLP) -> dict:
    """Fit one ``inner x encoding x seed`` ensemble and average its members.

    Public because the fold/seed/encoding budget has to be *inherited*, not
    retyped: in this project a transplanted ladder cost an arm 0.093 median
    Spearman and nearly produced a false negative, so any arm claiming parity
    with A0 calls this rather than rebuilding the loop. Phase B's censored arms
    and the A0 script's own two nesting levels all go through here.

    Every member fits on ``fit_df`` minus its inner fold and early-stops on that
    fold. Features are built from ``fit_df`` and each predict frame separately;
    ``build_features`` is row-local, so nothing is fitted across the boundary.

    ``selected`` maps an encoding to its ``(hidden, l2)``, which is the only
    thing that varies across the encoding axis.

    Returns:
        ``predictions`` — the member mean on each frame in ``predict_dfs``,
        in order; ``inner_oof`` — for each ``fit_df`` row, the mean of the
        ``len(encodings) * len(seeds)`` members that early-stopped on its inner
        fold; ``members`` — one record per member; ``n_members``.
    """
    n_inner = int(inner.max()) + 1
    per_inner = len(encodings) * len(seeds)
    y_fit = fit_df["y_log1p"].to_numpy()

    members: list[dict] = []
    stacked = [[] for _ in predict_dfs]
    inner_sum = np.zeros(len(fit_df))
    inner_count = np.zeros(len(fit_df), dtype=int)

    for encoding in encodings:
        hidden, l2 = selected[encoding]
        X_fit = build_features(fit_df, input_set, encoding)
        X_predict = [build_features(frame, input_set, encoding)
                     for frame in predict_dfs]
        for k in range(n_inner):
            is_fit = inner != k
            for seed in seeds:
                config = replace(base_config, hidden=tuple(hidden), l2=l2, seed=seed)
                model = MLPRegressor(config).fit(X_fit[is_fit], y_fit[is_fit],
                                                 X_fit[~is_fit], y_fit[~is_fit])
                for slot, X in zip(stacked, X_predict):
                    slot.append(model.predict(X))
                inner_sum[~is_fit] += model.predict(X_fit[~is_fit])
                inner_count[~is_fit] += 1
                members.append({"inner_fold": k, "encoding": encoding, "seed": seed,
                                "n_fit_rows": int(is_fit.sum()),
                                "best_epoch": model.best_epoch_,
                                "dev_mse": float(model.dev_mse_),
                                "fit_seconds": round(model.fit_seconds_, 2)})
        del X_fit, X_predict

    if not np.all(inner_count == per_inner):
        raise AssertionError(
            f"each fit row should be held out by exactly {per_inner} members "
            f"({len(encodings)} encodings x {len(seeds)} seeds), got counts "
            f"{sorted(set(inner_count.tolist()))}")
    return {
        "predictions": [np.mean(slot, axis=0) for slot in stacked],
        "inner_oof": inner_sum / per_inner,
        "members": members,
        "n_members": len(members),
    }


def fold_eligible_alleles(allele, y_true,
                          min_rows: int = MIN_ROWS_PER_ALLELE_FOLD) -> list[str]:
    """Alleles scoreable on one fold, from labels and counts alone.

    Delegates to :func:`pepstab.evaluation.eligible_alleles` with the fold bar
    passed explicitly, so the per-split thresholds in
    ``evaluation.MIN_ROWS_BY_SPLIT`` cannot be picked up by accident.
    """
    return evaluation.eligible_alleles(allele, y_true, min_rows=min_rows)


def score_fold(name: str, allele, y_true_log1p, y_pred_log1p,
               min_rows: int = MIN_ROWS_PER_ALLELE_FOLD) -> Scores:
    """Score one fold through the frozen stage 1 contract.

    Nothing about the metric is re-specified here: the panel comes from
    :func:`fold_eligible_alleles` and the scoring from
    :func:`pepstab.evaluation.score`, which brings average-rank ties,
    :data:`~pepstab.evaluation.UNRANKED_CONTRIBUTION` for a constant prediction
    on an eligible allele, and ``validate_finite`` rejection of NaN/inf.
    """
    alleles = fold_eligible_alleles(allele, y_true_log1p, min_rows=min_rows)
    return evaluation.score(name, allele, y_true_log1p, y_pred_log1p, alleles)


def fold_coverage(allele, y_true,
                  min_rows: int = MIN_ROWS_PER_ALLELE_FOLD) -> dict:
    """How much of a fold its eligible panel covers.

    Reported next to every fold score, per prespecification §2: a median over a
    panel that covers half the fold is a different claim from one that covers
    almost all of it.
    """
    allele_arr = np.asarray(allele)
    alleles = fold_eligible_alleles(allele_arr, y_true, min_rows=min_rows)
    n_rows = len(allele_arr)
    covered = int(np.isin(allele_arr, alleles).sum()) if n_rows else 0
    return {
        "n_alleles_eligible": len(alleles),
        "n_rows": n_rows,
        "eligible_row_coverage": (covered / n_rows) if n_rows else float("nan"),
    }
