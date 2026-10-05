"""Guards on the Phase A nested cross-validation harness (`pepstab/nested.py`).

Three kinds of check, mirroring ``tests/test_stage6.py``:

- **Partition parity** -- the outer level must *be* the existing five training
  cluster folds, so it is asserted equal to
  ``scripts.baseline_ensemble.cv_folds(train, 5)`` rather than assumed equal.
  Phase B and D join on these fold ids.
- **Leakage** -- no peptide cluster may span two folds at either level, and no
  inner-fold peptide may sit within the clustering radius of a peptide in its
  own fitting partition. That is the property the whole module exists for: an
  outer fold assessed on a peptide one substitution from one it fitted would be
  measuring memorisation.
- **Contract parity** -- eligibility is computed from labels and counts alone,
  and ``score_fold`` must reproduce ``evaluation.score`` on the same rows, so
  nothing frozen at stage 1 is quietly re-specified at a fold level.

Everything runs on the **training split only**. No test touches ``val`` or
``test``: Phase A scores neither.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab import evaluation, nested  # noqa: E402
from pepstab.data import PEPTIDE_LENGTH, encode_sequences, load_with_splits  # noqa: E402
from pepstab.mlp import MLPConfig  # noqa: E402
from pepstab.splits import HAMMING_THRESHOLD  # noqa: E402
from scripts.baseline_ensemble import cv_folds  # noqa: E402


@pytest.fixture(scope="module")
def split_df():
    return load_with_splits()


@pytest.fixture(scope="module")
def train(split_df):
    return split_df[split_df.split == "train"].sort_values(
        "pair_id").reset_index(drop=True)


def min_cross_distance(peptides_a, peptides_b) -> int:
    """Smallest Hamming distance between two peptide sets."""
    codes_a = encode_sequences(sorted(set(peptides_a)), PEPTIDE_LENGTH)
    codes_b = encode_sequences(sorted(set(peptides_b)), PEPTIDE_LENGTH)
    best = PEPTIDE_LENGTH
    for start in range(0, len(codes_a), 256):
        block = codes_a[start:start + 256]
        dist = (block[:, None, :] != codes_b[None, :, :]).sum(axis=2)
        best = min(best, int(dist.min()))
    return best


# --- partition parity -----------------------------------------------------

def test_outer_folds_equal_cv_folds(train):
    """The outer level is the *existing* five training cluster folds.

    Plan §3.2 asks for those folds, not a new partition. ``nested`` re-expresses
    the body of ``cv_folds`` so the library does not import a script, which only
    makes this equality more worth asserting.
    """
    np.testing.assert_array_equal(nested.outer_folds(train),
                                  cv_folds(train, nested.OUTER_FOLDS))


def test_outer_folds_partition_the_training_split_exactly_once(train):
    fold = nested.outer_folds(train)
    assert fold.shape == (len(train),)
    assert sorted(np.unique(fold).tolist()) == list(range(nested.OUTER_FOLDS))
    # Each row in exactly one fold, and the five folds reach every row.
    assert int(np.bincount(fold, minlength=nested.OUTER_FOLDS).sum()) == len(train)
    held = np.zeros(len(train), dtype=int)
    for split in nested.iter_outer(train, nested.SCREEN_INNER_FOLDS):
        assert not (split.fit_mask & split.held_mask).any()
        assert (split.fit_mask | split.held_mask).all()
        held += split.held_mask.astype(int)
    np.testing.assert_array_equal(held, np.ones(len(train), dtype=int))


def test_fold_vectors_are_deterministic(train):
    """``assign_clusters`` is seedless water-filling; nothing here may add noise."""
    np.testing.assert_array_equal(nested.outer_folds(train),
                                  nested.outer_folds(train))
    first = list(nested.iter_outer(train, nested.ASSESS_INNER_FOLDS))
    second = list(nested.iter_outer(train, nested.ASSESS_INNER_FOLDS))
    for a, b in zip(first, second):
        assert a.fold == b.fold
        np.testing.assert_array_equal(a.held_mask, b.held_mask)
        np.testing.assert_array_equal(a.inner, b.inner)


def test_inner_folds_sizes_and_labels(train):
    fit_rows = train[nested.outer_folds(train) != 0]
    inner = nested.inner_folds(fit_rows, nested.ASSESS_INNER_FOLDS)
    assert inner.shape == (len(fit_rows),)
    assert sorted(np.unique(inner).tolist()) == list(range(nested.ASSESS_INNER_FOLDS))


# --- leakage --------------------------------------------------------------

def test_no_cluster_spans_two_outer_folds(train):
    fold = nested.outer_folds(train)
    per_cluster = pd.DataFrame({"cluster_id": train.cluster_id.to_numpy(),
                                "fold": fold}).groupby("cluster_id").fold.nunique()
    assert (per_cluster == 1).all(), (
        f"{int((per_cluster > 1).sum())} cluster(s) span two outer folds")


def test_no_cluster_spans_two_inner_folds(train):
    for split in nested.iter_outer(train, nested.ASSESS_INNER_FOLDS):
        fit_rows = train[split.fit_mask]
        per_cluster = pd.DataFrame(
            {"cluster_id": fit_rows.cluster_id.to_numpy(), "fold": split.inner}
        ).groupby("cluster_id").fold.nunique()
        assert (per_cluster == 1).all(), (
            f"outer fold {split.fold}: a cluster spans two inner folds")


def test_no_cluster_spans_two_sub_folds(train):
    """The third nesting level inherits the guarantee.

    ``inner_oof_30`` splits an inner fitting partition again, so the sub-folds
    are built by the same primitive on a smaller frame. A cluster surviving
    whole through two levels but splitting at the third would leak just the
    same.
    """
    split = next(iter(nested.iter_outer(train, nested.ASSESS_INNER_FOLDS)))
    sub_fit = train[split.fit_mask][split.inner != 0]
    sub = nested.inner_folds(sub_fit, nested.ASSESS_INNER_FOLDS)
    per_cluster = pd.DataFrame({"cluster_id": sub_fit.cluster_id.to_numpy(),
                                "fold": sub}).groupby("cluster_id").fold.nunique()
    assert (per_cluster == 1).all()


def test_no_outer_held_peptide_is_a_near_neighbour_of_a_fit_peptide(train):
    """The assessment guarantee: nothing is scored near something it fitted.

    This is why the outer level is cluster-grouped ``cv_folds`` and not
    ``stage6.nested_folds``, which folds by peptide and deliberately splits
    mutant families across folds.
    """
    fold = nested.outer_folds(train)
    for k in range(nested.OUTER_FOLDS):
        held = fold == k
        best = min_cross_distance(train.loc[held, "peptide"],
                                  train.loc[~held, "peptide"])
        assert best > HAMMING_THRESHOLD, (
            f"outer fold {k}: nearest held/fit peptide pair is {best} "
            f"substitutions apart")


def test_no_inner_fold_peptide_is_a_near_neighbour_of_a_fit_peptide(train):
    """Mirrors ``test_no_dev_peptide_is_a_near_neighbour_of_a_fit_peptide``.

    Each member early-stops on its inner fold, so a near-duplicate across that
    cut would pick the epoch where the model had memorised it -- and here the
    stopped model's own inner-fold predictions are also *shipped* as
    ``inner_oof``, which A5 fits blend weights on.
    """
    for split in nested.iter_outer(train, nested.ASSESS_INNER_FOLDS):
        fit_rows = train[split.fit_mask]
        for k in range(nested.ASSESS_INNER_FOLDS):
            is_fit = split.inner != k
            best = min_cross_distance(fit_rows.loc[~is_fit, "peptide"],
                                      fit_rows.loc[is_fit, "peptide"])
            assert best > HAMMING_THRESHOLD, (
                f"outer fold {split.fold}, inner fold {k}: nearest pair is "
                f"{best} substitutions apart")


# --- contract parity ------------------------------------------------------

def test_fold_eligible_alleles_ignores_predictions(train):
    """Eligibility comes from labels and counts alone, never from predictions.

    The panel has to be identical for every arm, or an arm could raise its
    median by changing which alleles it is scored on. Permuting a prediction
    vector cannot touch it -- the function does not take one.
    """
    held = train[nested.outer_folds(train) == 0]
    alleles = nested.fold_eligible_alleles(held.allele, held.y_log1p.to_numpy())
    assert alleles
    rng = np.random.default_rng(0)
    y_pred = rng.normal(size=len(held))
    for _ in range(3):
        rng.shuffle(y_pred)
        again = nested.fold_eligible_alleles(held.allele, held.y_log1p.to_numpy())
        assert again == alleles
    # And it is the frozen helper, at the fold bar rather than a split bar.
    assert alleles == evaluation.eligible_alleles(
        held.allele, held.y_log1p.to_numpy(),
        min_rows=nested.MIN_ROWS_PER_ALLELE_FOLD)


def test_min_rows_per_allele_fold_is_the_val_bar():
    """Prespecification §2 freezes the fold bar at validation's 20 rows.

    The test split's 50-row bar is calibrated to a 20% split and does not
    transfer to a fold of the training split.
    """
    assert nested.MIN_ROWS_PER_ALLELE_FOLD == evaluation.MIN_ROWS_BY_SPLIT["val"]


def test_score_fold_reproduces_evaluation_score(train):
    """``score_fold`` must be ``evaluation.score``, not a second implementation.

    The tie rule, ``UNRANKED_CONTRIBUTION`` and ``validate_finite`` are the
    frozen stage 1 contract; a fold-level copy could drift from it.
    """
    held = train[nested.outer_folds(train) == 0]
    y_true = held.y_log1p.to_numpy()
    rng = np.random.default_rng(20261004)
    y_pred = y_true + rng.normal(scale=0.5, size=len(held))

    mine = nested.score_fold("a0", held.allele, y_true, y_pred)
    alleles = evaluation.eligible_alleles(
        held.allele, y_true, min_rows=nested.MIN_ROWS_PER_ALLELE_FOLD)
    theirs = evaluation.score("a0", held.allele, y_true, y_pred, alleles)

    assert mine.median_spearman == theirs.median_spearman
    assert mine.iqr_spearman == theirs.iqr_spearman
    assert mine.mae_log1p == theirs.mae_log1p
    assert mine.n_alleles_eligible == theirs.n_alleles_eligible
    pd.testing.assert_series_equal(mine.panel, theirs.panel)


def test_constant_predictor_scores_zero_on_an_eligible_fold_allele(train):
    """A constant prediction scores 0 and keeps its allele in the panel.

    Not NaN, and not dropped: dropping would let an arm raise its fold median
    by going constant on the alleles it finds hard. EVALUATION.md's
    ``UNRANKED_CONTRIBUTION``, reached through ``score_fold``.
    """
    held = train[nested.outer_folds(train) == 0]
    y_true = held.y_log1p.to_numpy()
    scores = nested.score_fold("constant", held.allele, y_true,
                               np.full(len(held), 1.234))
    assert scores.n_alleles_unranked == scores.n_alleles_scored
    assert scores.median_spearman == evaluation.UNRANKED_CONTRIBUTION
    assert scores.panel.notna().all()
    assert (scores.panel == evaluation.UNRANKED_CONTRIBUTION).all()
    assert scores.per_allele.isna().all()  # the honest NaN is kept alongside


def test_score_fold_rejects_non_finite_predictions(train):
    held = train[nested.outer_folds(train) == 0]
    y_pred = np.zeros(len(held))
    y_pred[7] = np.nan
    with pytest.raises(ValueError, match="invalid value"):
        nested.score_fold("bad", held.allele, held.y_log1p.to_numpy(), y_pred)


def test_fold_coverage_counts_rows_not_alleles(train):
    held = train[nested.outer_folds(train) == 0]
    y_true = held.y_log1p.to_numpy()
    coverage = nested.fold_coverage(held.allele, y_true)
    assert coverage["n_rows"] == len(held)
    assert coverage["n_alleles_eligible"] == len(
        nested.fold_eligible_alleles(held.allele, y_true))
    assert 0.0 < coverage["eligible_row_coverage"] <= 1.0
    # The bar is a row count, so a fold of this size keeps nearly every row.
    assert coverage["eligible_row_coverage"] > 0.95


# --- ensemble budget ------------------------------------------------------

def test_fit_mlp_ensemble_arity_and_holdout(train):
    """The shared budget: every fit row held out by exactly 2 x len(seeds).

    Phase B inherits this routine rather than retyping the loop -- a
    transplanted ladder has already cost an arm 0.093 median Spearman here.
    Run small: two inner folds, one seed, one encoding.
    """
    split = next(iter(nested.iter_outer(train, 2)))
    fit_rows = train[split.fit_mask].reset_index(drop=True).head(1200)
    inner = nested.inner_folds(fit_rows, 2)
    held = train[split.held_mask].reset_index(drop=True).head(300)

    result = nested.fit_mlp_ensemble(
        fit_rows, [held], inner, ("onehot",), (0,),
        {"onehot": ((16,), 1e-3)},
        base_config=MLPConfig(hidden=(16,), max_epochs=3, patience=2))
    assert result["n_members"] == 2  # 2 inner folds x 1 encoding x 1 seed
    assert len(result["predictions"]) == 1
    assert result["predictions"][0].shape == (len(held),)
    assert result["inner_oof"].shape == (len(fit_rows),)
    assert np.isfinite(result["inner_oof"]).all()
    assert {m["inner_fold"] for m in result["members"]} == {0, 1}
    assert all("dev_mse" in m and "best_epoch" in m for m in result["members"])
