"""Guards on the stage 1 contract. Run with: .venv/bin/python -m pytest -q

These are the properties later stages depend on. If one fails, a downstream
result is not trustworthy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.data import (
    AMINO_ACIDS,
    PEPTIDE_LENGTH,
    encode_sequences,
    load_raw,
    load_with_splits,
)
from pepstab.evaluation import (
    MIN_ROWS_BY_SPLIT,
    eligible_alleles,
    paired_cluster_bootstrap,
    per_allele_spearman,
    precision_at_k,
    score,
    score_by_distance,
)
from pepstab.splits import (
    HAMMING_THRESHOLD,
    assign_clusters,
    min_cross_split_distance,
    single_linkage_clusters,
)


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def split_df():
    return load_with_splits()


# --- data integrity -------------------------------------------------------

def test_raw_shape_and_cleanliness(raw):
    assert len(raw) == 28166
    assert raw.isna().sum().sum() == 0
    assert not raw.duplicated(["allele", "peptide"]).any()
    assert raw.peptide.str.len().eq(PEPTIDE_LENGTH).all()


def test_only_standard_residues(raw):
    for col in ("peptide", "hla_seq", "hla_pseudoseq"):
        assert set("".join(raw[col].unique())) <= set(AMINO_ACIDS), col


def test_pair_id_is_row_position(raw):
    assert raw.pair_id.tolist() == list(range(len(raw)))


def test_log1p_target_is_finite_and_defined_at_zero(raw):
    assert np.isfinite(raw.y_log1p).all()
    assert (raw.loc[raw.thalf_hours == 0, "y_log1p"] == 0).all()


def test_encode_rejects_non_standard_residue():
    with pytest.raises(ValueError, match="unknown residue"):
        encode_sequences(["ACDEFGHIX"], PEPTIDE_LENGTH)


def test_encode_rejects_wrong_length():
    with pytest.raises(ValueError, match="length"):
        encode_sequences(["ACDEF"], PEPTIDE_LENGTH)


# --- splits ---------------------------------------------------------------

def test_splits_cover_every_pair_exactly_once(split_df):
    assert len(split_df) == 28166
    assert split_df.split.isin(["train", "val", "test"]).all()
    assert split_df.pair_id.is_unique


def test_split_proportions_are_70_10_20(split_df):
    shares = split_df.split.value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.70, abs=0.005)
    assert shares["val"] == pytest.approx(0.10, abs=0.005)
    assert shares["test"] == pytest.approx(0.20, abs=0.005)


def test_no_cluster_spans_two_splits(split_df):
    assert (split_df.groupby("cluster_id").split.nunique() == 1).all()


def test_no_peptide_appears_in_two_splits(split_df):
    assert (split_df.groupby("peptide").split.nunique() == 1).all()


def test_held_out_peptides_are_far_from_training_peptides(split_df):
    """The headline leakage guarantee."""
    for pair, dist in min_cross_split_distance(split_df).items():
        assert dist > HAMMING_THRESHOLD, pair


def test_single_linkage_groups_a_known_chain():
    """A -> B -> C each 3 apart chains into one cluster; a 4-apart peptide does not."""
    codes = encode_sequences(
        ["AAAAAAAAA", "CCCAAAAAA", "CCCCCCAAA", "DDDDAAAAA"], PEPTIDE_LENGTH)
    labels = single_linkage_clusters(codes, threshold=3)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]


def test_water_fill_tracks_target_proportions_for_equal_weights():
    """The bug this guards: ranking by absolute deficit gives equal thirds."""
    weights = pd.Series({i: 1 for i in range(1000)})
    assigned = assign_clusters(weights)
    shares = assigned.value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.70, abs=0.02)
    assert shares["val"] == pytest.approx(0.10, abs=0.02)
    assert shares["test"] == pytest.approx(0.20, abs=0.02)


def test_water_fill_spreads_heavy_clusters_too():
    """Heavy clusters must not all land in one split."""
    weights = pd.Series({i: 100 for i in range(60)})
    assigned = assign_clusters(weights)
    assert set(assigned.unique()) == {"train", "val", "test"}


def test_every_allele_keeps_proportional_test_coverage(split_df):
    counts = split_df.pivot_table(index="allele", columns="split", aggfunc="size",
                                  fill_value=0)
    share = counts["test"] / counts.sum(axis=1)
    big = counts[counts.sum(axis=1) >= 100].index
    assert share.loc[big].min() > 0.10, "a well-populated allele lost test coverage"
    assert share.median() == pytest.approx(0.20, abs=0.02)


# --- evaluation -----------------------------------------------------------

def test_eligible_alleles_ignores_predictions(split_df):
    test = split_df[split_df.split == "test"]
    first = eligible_alleles(test.allele, test.y_log1p, split="test")
    assert len(first) == 67
    # Shuffling labels changes nothing about which alleles are *eligible*, since
    # eligibility is a function of counts and label spread, not of any model.
    assert eligible_alleles(test.allele, test.y_log1p.sample(frac=1, random_state=0),
                            split="test") == first


def test_val_threshold_keeps_enough_alleles_to_select_on(split_df):
    val = split_df[split_df.split == "val"]
    assert len(eligible_alleles(val.allele, val.y_log1p, split="val")) >= 60
    assert MIN_ROWS_BY_SPLIT["val"] < MIN_ROWS_BY_SPLIT["test"]


def test_perfect_predictions_score_one():
    allele = pd.Series(["A"] * 30 + ["B"] * 30)
    y = np.concatenate([np.linspace(0, 3, 30), np.linspace(0, 2, 30)])
    rho = per_allele_spearman(allele, y, y, ["A", "B"])
    assert rho["A"] == pytest.approx(1.0)
    assert rho["B"] == pytest.approx(1.0)


def test_reversed_predictions_score_minus_one():
    allele = pd.Series(["A"] * 30)
    y = np.linspace(0, 3, 30)
    assert per_allele_spearman(allele, y, -y, ["A"])["A"] == pytest.approx(-1.0)


def test_constant_prediction_gives_undefined_not_zero():
    allele = pd.Series(["A"] * 30)
    y = np.linspace(0, 3, 30)
    rho = per_allele_spearman(allele, y, np.ones(30), ["A"])
    assert np.isnan(rho["A"])
    s = score("const", allele, y, np.ones(30), ["A"])
    assert np.isnan(s.median_spearman)
    assert np.isnan(s.pooled_spearman)


def test_precision_at_k_on_a_constant_predictor_equals_base_rate():
    """Ties must not be broken by row order."""
    allele = pd.Series(["A"] * 20)
    hours = np.array([5.0] * 6 + [0.5] * 14)  # base rate 0.3 above 2 h
    table = precision_at_k(allele, hours, np.ones(20), ["A"])
    assert table.loc["A", "precision_at_k"] == pytest.approx(0.3)
    assert table.loc["A", "base_rate"] == pytest.approx(0.3)
    # Reordering the rows must not change the score.
    order = np.array([13, 2, 7, 19, 0, 5, 11, 1, 18, 4, 9, 15, 3, 8, 16, 6, 12, 10, 17, 14])
    assert precision_at_k(allele, hours[order], np.ones(20), ["A"]).loc[
        "A", "precision_at_k"] == pytest.approx(0.3)


def test_precision_at_k_rewards_a_correct_ranking():
    allele = pd.Series(["A"] * 20)
    hours = np.array([5.0] * 6 + [0.5] * 14)
    table = precision_at_k(allele, hours, -hours, ["A"])  # rank the stable ones last
    assert table.loc["A", "precision_at_k"] == 0.0
    table = precision_at_k(allele, hours, hours, ["A"])
    assert table.loc["A", "precision_at_k"] == pytest.approx(table.loc["A", "ceiling"])


def test_paired_bootstrap_is_undefined_against_a_constant_baseline():
    allele = pd.Series(["A"] * 40)
    clusters = pd.Series(np.arange(40))
    y = np.linspace(0, 3, 40)
    r = paired_cluster_bootstrap(clusters, allele, y, np.ones(40), y, ["A"],
                                 n_boot=20)
    assert r["undefined"] is True
    assert not r["conclusive"]


def test_paired_bootstrap_detects_a_real_improvement():
    rng = np.random.default_rng(0)
    allele = pd.Series(np.repeat([f"A{i}" for i in range(12)], 40))
    clusters = pd.Series(np.tile(np.arange(40), 12))
    y = rng.normal(size=480)
    good = y + rng.normal(0, 0.3, 480)
    bad = y + rng.normal(0, 2.5, 480)
    r = paired_cluster_bootstrap(clusters, allele, y, bad, good,
                                 sorted(allele.unique()), n_boot=200)
    assert r["delta_median_spearman"] > 0
    assert r["conclusive"]
    assert r["ci95"][0] > 0


def test_paired_bootstrap_is_inconclusive_when_models_are_identical():
    rng = np.random.default_rng(1)
    allele = pd.Series(np.repeat([f"A{i}" for i in range(12)], 40))
    clusters = pd.Series(np.tile(np.arange(40), 12))
    y = rng.normal(size=480)
    pred = y + rng.normal(0, 1.0, 480)
    r = paired_cluster_bootstrap(clusters, allele, y, pred, pred,
                                 sorted(allele.unique()), n_boot=200)
    assert r["delta_median_spearman"] == pytest.approx(0.0)
    assert not r["conclusive"]
    assert not r["rules_out_min_worthwhile"] or r["ci95"][1] < 0.05


# --- distance to training -------------------------------------------------

def test_dist_to_train_is_frozen_and_consistent(split_df):
    from pepstab.splits import distance_to_train
    recomputed = distance_to_train(split_df)
    assert (split_df.peptide.map(recomputed) == split_df.dist_to_train).all()


def test_train_peptides_are_zero_distance_from_training(split_df):
    assert (split_df.loc[split_df.split == "train", "dist_to_train"] == 0).all()


def test_held_out_peptides_are_at_least_four_from_training(split_df):
    held = split_df[split_df.split != "train"]
    assert held.dist_to_train.min() == 4


def test_distance_strata_cover_every_held_out_row(split_df):
    from pepstab.evaluation import DISTANCE_STRATA
    held = split_df[split_df.split != "train"]
    covered = np.zeros(len(held), dtype=bool)
    d = held.dist_to_train.to_numpy()
    for lo, hi in DISTANCE_STRATA.values():
        covered |= (d >= lo) & (d <= hi)
    assert covered.all()


def test_distance_strata_share_one_allele_set_on_test(split_df):
    """Strata must differ by distance only, never by which alleles they hold."""
    test = split_df[split_df.split == "test"]
    rng = np.random.default_rng(0)
    pred = test.y_log1p.to_numpy() + rng.normal(0, 1.0, len(test))
    table = score_by_distance("m", test.dist_to_train, test.allele,
                              test.y_log1p, pred, split="test")
    assert len(table) == 2
    assert table.n_alleles.nunique() == 1
    assert table.n_alleles.iloc[0] >= 60
