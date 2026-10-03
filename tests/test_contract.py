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
    UNRANKED_CONTRIBUTION,
    describe_delta,
    eligible_alleles,
    panel_spearman,
    paired_cluster_bootstrap,
    per_allele_spearman,
    precision_at_k,
    rankable_alleles,
    score,
    score_by_distance,
)
from scripts.evaluate import read_predictions
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


def test_constant_prediction_is_undefined_per_allele_but_scores_zero():
    """Per-allele reporting stays honest; the panel aggregate credits chance.

    The per-allele table must say "undefined", because nothing was measured. The
    panel median must still produce a number, or the primary metric would be
    unavailable for exactly the models it should rank last.
    """
    allele = pd.Series(["A"] * 30)
    y = np.linspace(0, 3, 30)
    rho = per_allele_spearman(allele, y, np.ones(30), ["A"])
    assert np.isnan(rho["A"])
    s = score("const", allele, y, np.ones(30), ["A"])
    assert s.median_spearman == pytest.approx(UNRANKED_CONTRIBUTION)
    assert s.n_alleles_unranked == 1
    assert np.isnan(s.pooled_spearman)  # a pooled correlation really is undefined


def test_partially_constant_predictions_do_not_shrink_the_panel():
    """A model cannot raise its median by going constant on the hard alleles.

    Perfect on A, constant on B and C. Dropping the undefined alleles would
    report 1.0 on a panel of three; the panel is fixed, so the answer is the
    median of [1, 0, 0].
    """
    n = 25
    allele = pd.Series(["A"] * n + ["B"] * n + ["C"] * n)
    y = np.tile(np.linspace(0, 3, n), 3)
    pred = np.concatenate([np.linspace(0, 3, n), np.zeros(n), np.zeros(n)])
    s = score("partial", allele, y, pred, ["A", "B", "C"])
    assert s.median_spearman == pytest.approx(0.0)
    assert s.n_alleles_eligible == 3
    assert s.n_alleles_scored == 3
    assert s.n_alleles_unranked == 2


def test_partially_constant_model_cannot_beat_a_full_ranker():
    """The bootstrap must not read abstention on hard alleles as an improvement."""
    n = 25
    allele = pd.Series(np.repeat(["A", "B", "C"], n))
    clusters = pd.Series(np.tile(np.arange(n), 3))
    y = np.tile(np.linspace(0, 3, n), 3)
    full = y.copy()                                    # ranks all three
    partial = np.concatenate([np.linspace(0, 3, n), np.zeros(n), np.zeros(n)])
    r = paired_cluster_bootstrap(clusters, allele, y, full, partial,
                                 ["A", "B", "C"], n_boot=100)
    assert not r["undefined"]
    assert r["delta_median_spearman"] < 0
    assert r["ci95"][1] < 0  # conclusively worse, not a silent win


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


def test_paired_bootstrap_compares_against_a_constant_baseline():
    """A constant baseline is scorable at chance, so the comparison is defined.

    Previously this returned "undefined" and sent the reader to MAE, which made
    a per-allele-mean baseline -- a meaningful reference -- impossible to compare
    on the primary metric.
    """
    allele = pd.Series(["A"] * 40)
    clusters = pd.Series(np.arange(40))
    y = np.linspace(0, 3, 40)
    r = paired_cluster_bootstrap(clusters, allele, y, np.ones(40), y, ["A"],
                                 n_boot=50)
    assert r["undefined"] is False
    assert r["delta_median_spearman"] == pytest.approx(1.0)
    assert r["conclusive"]


def test_paired_bootstrap_is_undefined_when_no_allele_has_label_spread():
    """The only remaining undefined case, and it hits both models alike."""
    allele = pd.Series(["A"] * 40)
    clusters = pd.Series(np.arange(40))
    y = np.zeros(40)  # every label at the floor: nothing to rank
    r = paired_cluster_bootstrap(clusters, allele, y, np.ones(40),
                                 np.linspace(0, 1, 40), ["A"], n_boot=20)
    assert r["undefined"] is True
    assert not r["conclusive"]


def test_verdict_distinguishes_a_straddling_ci_from_one_below_the_bar():
    """A CI that excludes 0 but straddles 0.05 leaves the size unresolved."""
    straddles = describe_delta({"ci95": (0.0215, 0.0863),
                               "delta_median_spearman": 0.0568})
    assert "unresolved" in straddles
    below = describe_delta({"ci95": (0.01, 0.04), "delta_median_spearman": 0.03})
    assert "rules out" in below and "unresolved" not in below
    meets = describe_delta({"ci95": (0.06, 0.10), "delta_median_spearman": 0.08})
    assert "meets" in meets
    worse = describe_delta({"ci95": (-0.20, -0.05), "delta_median_spearman": -0.1})
    assert "worse" in worse
    crosses = describe_delta({"ci95": (-0.01, 0.09), "delta_median_spearman": 0.02})
    assert "inconclusive" in crosses


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


# --- prediction files -----------------------------------------------------

def _truth(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({"pair_id": list(range(n))})


def test_read_predictions_rejects_non_finite_values(tmp_path):
    """inf passes a NaN check, scores a perfect Spearman and makes MAE infinite."""
    path = tmp_path / "inf.csv"
    pd.DataFrame({"pair_id": [0, 1, 2],
                  "y_pred": [0.0, 1.0, np.inf]}).to_csv(path, index=False)
    with pytest.raises(SystemExit, match="NaN or inf"):
        read_predictions(path, _truth())


def test_read_predictions_rejects_non_numeric_values(tmp_path):
    path = tmp_path / "text.csv"
    pd.DataFrame({"pair_id": [0, 1, 2],
                  "y_pred": [0.0, "not-a-number", 2.0]}).to_csv(path, index=False)
    with pytest.raises(SystemExit, match="non-numeric"):
        read_predictions(path, _truth())


def test_read_predictions_accepts_finite_values(tmp_path):
    path = tmp_path / "ok.csv"
    pd.DataFrame({"pair_id": [0, 1, 2],
                  "y_pred": [0.0, 1.5, -0.25]}).to_csv(path, index=False)
    out = read_predictions(path, _truth())
    assert np.isfinite(out.to_numpy()).all()
    assert out.tolist() == [0.0, 1.5, -0.25]


# --- the unranked convention, narrowly ------------------------------------

def test_degenerate_allele_leaves_the_panel_instead_of_scoring_zero():
    """Three cases must stay distinct, and only one of them scores 0.

    A: rankable and ranked. B: no label spread on these rows -- degenerate, so it
    leaves the panel rather than being credited 0, because the rows could not
    test it. C: label spread but constant predictions -- the model failed to rank
    it, which is what UNRANKED_CONTRIBUTION is for.
    """
    n = 20
    allele = pd.Series(["A"] * n + ["B"] * n + ["C"] * n)
    y = np.concatenate([np.linspace(0, 3, n), np.zeros(n), np.linspace(0, 3, n)])
    pred = np.concatenate([np.linspace(0, 3, n), np.linspace(0, 1, n), np.zeros(n)])

    rankable = rankable_alleles(allele, y, ["A", "B", "C"])
    assert rankable == ["A", "C"]  # B dropped: nothing to rank

    panel = panel_spearman(per_allele_spearman(allele, y, pred, ["A", "B", "C"]),
                           rankable)
    assert "B" not in panel.index
    assert panel["A"] == pytest.approx(1.0)
    assert panel["C"] == pytest.approx(UNRANKED_CONTRIBUTION)


def test_reversed_ranking_scores_below_an_unranked_one():
    """The ordering that makes 0 the right choice for 'no information'."""
    n = 30
    allele = pd.Series(["A"] * n)
    y = np.linspace(0, 3, n)
    reversed_ = score("rev", allele, y, -y, ["A"]).median_spearman
    constant = score("const", allele, y, np.ones(n), ["A"]).median_spearman
    skilled = score("good", allele, y, y, ["A"]).median_spearman
    assert reversed_ < constant < skilled
    assert constant == pytest.approx(UNRANKED_CONTRIBUTION)


def test_per_allele_frame_flags_unranked_without_imputing_spearman():
    """The table must carry the NaN and the convention side by side."""
    n = 25
    allele = pd.Series(["A"] * n + ["B"] * n)
    y = np.tile(np.linspace(0, 3, n), 2)
    pred = np.concatenate([np.linspace(0, 3, n), np.zeros(n)])
    frame = score("partial", allele, y, pred, ["A", "B"]).per_allele_frame()
    assert np.isnan(frame.loc["B", "spearman"])        # never imputed
    assert bool(frame.loc["B", "unranked"]) is True
    assert frame.loc["B", "scored"] == pytest.approx(UNRANKED_CONTRIBUTION)
    assert bool(frame.loc["A", "unranked"]) is False
    assert frame.loc["A", "scored"] == pytest.approx(1.0)


def test_read_predictions_rejects_nan_as_invalid(tmp_path):
    """NaN is invalid input, not a scoreable constant."""
    path = tmp_path / "nan.csv"
    pd.DataFrame({"pair_id": [0, 1, 2],
                  "y_pred": [0.0, np.nan, 2.0]}).to_csv(path, index=False)
    with pytest.raises(SystemExit, match="NaN or inf"):
        read_predictions(path, _truth())


def test_bootstrap_reports_degenerate_draws():
    rng = np.random.default_rng(3)
    allele = pd.Series(np.repeat([f"A{i}" for i in range(6)], 40))
    clusters = pd.Series(np.tile(np.arange(40), 6))
    y = rng.normal(size=240)
    r = paired_cluster_bootstrap(clusters, allele, y, y + rng.normal(0, 2, 240),
                                 y + rng.normal(0, 0.3, 240),
                                 sorted(allele.unique()), n_boot=100)
    assert r["panel_size_full"] == 6
    assert r["n_resamples_with_dropped_alleles"] == 0  # continuous labels
    assert r["n_degenerate_resamples"] == 0


# --- invalid predictions reach no metric ----------------------------------

@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf],
                         ids=["nan", "+inf", "-inf"])
def test_score_rejects_non_finite_predictions(bad_value):
    """Direct calls must not rely on the CLI loader for validation.

    Unchecked, these corrupt the primary metric in opposite directions: +inf
    outranks every real prediction and pulls Spearman to 1.0, while NaN makes the
    allele look constant and is absorbed as an unranked 0.
    """
    n = 30
    allele = pd.Series(["A"] * n)
    y = np.linspace(0, 3, n)
    pred = y.copy()
    pred[-1] = bad_value
    with pytest.raises(ValueError, match="invalid value"):
        score("bad", allele, y, pred, ["A"])


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf],
                         ids=["nan", "+inf", "-inf"])
def test_bootstrap_rejects_non_finite_predictions_in_either_arm(bad_value):
    """Both arms, and before resampling starts."""
    n = 40
    allele = pd.Series(["A"] * n)
    clusters = pd.Series(np.arange(n))
    y = np.linspace(0, 3, n)
    good = y.copy()
    bad = y.copy()
    bad[0] = bad_value
    with pytest.raises(ValueError, match="comparison predictions"):
        paired_cluster_bootstrap(clusters, allele, y, good, bad, ["A"], n_boot=10)
    with pytest.raises(ValueError, match="baseline predictions"):
        paired_cluster_bootstrap(clusters, allele, y, bad, good, ["A"], n_boot=10)


def test_score_by_distance_rejects_non_finite_predictions(split_df):
    val = split_df[split_df.split == "val"]
    pred = val.y_log1p.to_numpy().copy()
    pred[0] = np.inf
    with pytest.raises(ValueError, match="invalid value"):
        score_by_distance("bad", val.dist_to_train, val.allele, val.y_log1p,
                          pred, split="val")


def test_non_finite_labels_are_rejected_too():
    """Same silent-failure class; the frozen data is finite, misuse is not."""
    n = 30
    allele = pd.Series(["A"] * n)
    y = np.linspace(0, 3, n)
    y[-1] = np.nan
    with pytest.raises(ValueError, match="labels"):
        score("bad", allele, y, np.linspace(0, 3, n), ["A"])


def test_validation_does_not_reject_a_finite_constant_prediction():
    """The agreed convention survives: constant is valid and scores 0."""
    n = 30
    allele = pd.Series(["A"] * n)
    y = np.linspace(0, 3, n)
    s = score("const", allele, y, np.ones(n), ["A"])
    assert s.median_spearman == pytest.approx(UNRANKED_CONTRIBUTION)
    assert np.isnan(s.per_allele["A"])  # mathematical value still NaN
