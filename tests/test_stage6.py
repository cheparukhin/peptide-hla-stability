"""Guards for the stage 6 analysis machinery (`pepstab/stage6.py`).

Three kinds of check:

- **Contract parity** -- stage 6 must not quietly re-specify anything frozen at
  stage 1, so the distance stratification is asserted equal to
  ``evaluation.score_by_distance`` and the precision table equal to
  ``evaluation.precision_at_k``.
- **Scoring properties** -- a constant predictor scores chance on every
  concordance and its base rate on precision@10; an allele-only predictor
  scores 0 on the per-allele-pair Spearman. These are the properties that make
  the metrics readable.
- **Leakage and join safety** -- splits are never recomputed, joins are on
  ``(pair_id)`` from the frozen file, the nested folds split mutant pairs apart
  rather than holding whole families out, and no near-neighbour group straddles
  a frozen split.

Everything runs on the validation and training splits. **No test touches the
test split**; test is scored once, at stage 6, by the orchestrator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab import evaluation, stage6  # noqa: E402
from pepstab.data import TARGET  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PREDS = REPO_ROOT / "preds"
BASELINE = PREDS / "seq_baseline.csv"
ENSEMBLE = PREDS / "seq_ensemble_pep_domain.csv"
ALLELE_MEAN = PREDS / "allele_mean.csv"
GLOBAL_MEAN = PREDS / "global_mean.csv"


@pytest.fixture(scope="module")
def val():
    return stage6.load_split("val")


@pytest.fixture(scope="module")
def train():
    return stage6.load_split("train")


@pytest.fixture(scope="module")
def val_alleles(val):
    return evaluation.eligible_alleles(val["allele"], val[TARGET].to_numpy(),
                                       split="val")


@pytest.fixture(scope="module")
def ens(val):
    return stage6.read_predictions(ENSEMBLE, val).to_numpy()


# ---------------------------------------------------------------- loading

def test_load_split_matches_frozen_file(val):
    """The split comes from data/splits.csv, joined on pair_id, not recomputed."""
    frozen = pd.read_csv(REPO_ROOT / "data" / "splits.csv")
    expected = frozen.loc[frozen["split"] == "val", "pair_id"].sort_values()
    assert val["pair_id"].tolist() == expected.tolist()
    assert val["pair_id"].is_monotonic_increasing


def test_superseded_split_file_is_not_read():
    """The BLOSUM62 peptide_splits.csv is superseded and must never be loaded."""
    source = (REPO_ROOT / "pepstab" / "stage6.py").read_text()
    source += (REPO_ROOT / "scripts" / "stage6_report.py").read_text()
    assert "peptide_splits" not in source
    assert "c67s_cleanup" not in source


def test_read_predictions_agrees_with_evaluate_cli(val):
    """stage6.read_predictions and scripts/evaluate.py must align identically."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_evaluate_cli", REPO_ROOT / "scripts" / "evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mine = stage6.read_predictions(BASELINE, val).to_numpy()
    theirs = mod.read_predictions(BASELINE, val).to_numpy()
    np.testing.assert_array_equal(mine, theirs)


def test_read_predictions_rejects_nonfinite(val, tmp_path):
    bad = pd.read_csv(BASELINE)
    bad.loc[0, "y_pred"] = np.inf
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError):
        stage6.read_predictions(path, val)


def test_read_predictions_rejects_incomplete_coverage(val, tmp_path):
    short = pd.read_csv(BASELINE).iloc[:-5]
    path = tmp_path / "short.csv"
    short.to_csv(path, index=False)
    with pytest.raises(ValueError, match="no prediction for"):
        stage6.read_predictions(path, val)


# ------------------------------------------- 1. distance stratification

def test_score_strata_reproduces_the_frozen_contract(val, ens):
    """At the frozen 20-row bar this must equal evaluation.score_by_distance."""
    mine = stage6.score_strata("m", val, ens,
                               min_rows=evaluation.MIN_ROWS_PER_ALLELE_IN_STRATUM)
    theirs = evaluation.score_by_distance("m", val["dist_to_train"], val["allele"],
                                          val[TARGET].to_numpy(), ens)
    pd.testing.assert_frame_equal(mine.reset_index(drop=True),
                                  theirs.reset_index(drop=True))


def test_distance_census_covers_every_row(val):
    census = stage6.distance_census(val)
    assert census["n_rows"].sum() == len(val)
    assert np.isclose(census["row_share"].sum(), 1.0)
    # The frozen split guarantees no held-out peptide is within 3 of training.
    assert census["dist_to_train"].min() >= 4


def test_strata_are_scored_on_one_shared_allele_panel(val, ens):
    panel = stage6.stratum_panel(val, min_rows=10)
    table = stage6.score_strata("m", val, ens, min_rows=10)
    assert table["n_alleles"].nunique() == 1
    assert int(table["n_alleles"].iloc[0]) == panel["n_common"]
    assert panel["n_common"] <= min(panel["n_eligible"].values())


def test_stratum_rows_partition_the_split(val):
    strata = evaluation.DISTANCE_STRATA
    masks = [((val["dist_to_train"] >= lo) & (val["dist_to_train"] <= hi)).to_numpy()
             for lo, hi in strata.values()]
    stacked = np.vstack(masks)
    assert stacked.sum(axis=0).max() == 1, "strata overlap"
    assert stacked.any(axis=0).all(), "a row falls in no stratum"


# ------------------------------------------------ 2. differential target

def test_differential_pairs_are_canonical_and_complete(val, val_alleles):
    pairs = stage6.differential_pairs(val, val_alleles)
    assert (pairs["allele_a"] < pairs["allele_b"]).all()
    assert not pairs.duplicated(["peptide", "allele_a", "allele_b"]).any()
    # a peptide on k alleles contributes exactly k(k-1)/2 comparisons
    sub = val[val["allele"].isin(val_alleles)]
    k = sub.groupby("peptide")["allele"].nunique()
    expected = int((k * (k - 1) // 2).sum())
    assert len(pairs) == expected
    assert pairs["peptide"].nunique() == int((k >= 2).sum())


def test_differential_delta_matches_the_labels(val, val_alleles):
    pairs = stage6.differential_pairs(val, val_alleles).head(200)
    y = val.set_index("pair_id")[TARGET]
    np.testing.assert_allclose(
        pairs["delta_true"].to_numpy(),
        (y.reindex(pairs["pair_id_b"]).to_numpy()
         - y.reindex(pairs["pair_id_a"]).to_numpy()))


def test_constant_predictor_scores_chance_on_the_differential(val, val_alleles):
    pairs = stage6.differential_pairs(val, val_alleles)
    flat = pd.Series(np.full(len(val), 1.234), index=val["pair_id"].to_numpy())
    r = stage6.score_differential("flat", pairs, flat)
    assert r["concordance"] == pytest.approx(0.5)
    assert r["concordance_peptide_weighted"] == pytest.approx(0.5)
    # every allele pair is unranked, which the predeclared rule scores 0
    assert r["median_allele_pair_spearman"] == pytest.approx(0.0)


def test_allele_only_predictor_separates_the_two_differential_views(val, val_alleles):
    """The per-allele-pair Spearman is what needs peptide-specific chemistry.

    A predictor that outputs each allele's mean beats chance on direction --
    it knows which groove is stickier on average -- but carries no information
    about *which peptides* differ most between two alleles, so its per-allele-
    pair Spearman is undefined and scores 0 under the predeclared convention.
    """
    pairs = stage6.differential_pairs(val, val_alleles)
    pred = stage6.read_predictions(ALLELE_MEAN, val)
    lookup = pd.Series(pred.to_numpy(), index=val["pair_id"].to_numpy())
    r = stage6.score_differential("allele_mean", pairs, lookup)
    assert r["concordance"] > 0.6
    assert r["median_allele_pair_spearman"] == pytest.approx(0.0)


def test_differential_is_invariant_to_a_peptide_only_shift(val, val_alleles):
    """Anything intrinsic to the peptide cancels -- that is the point."""
    pairs = stage6.differential_pairs(val, val_alleles)
    base = stage6.read_predictions(ENSEMBLE, val).to_numpy()
    rng = np.random.default_rng(0)
    bump = pd.Series(rng.normal(size=val["peptide"].nunique()),
                     index=sorted(val["peptide"].unique()))
    shifted = base + val["peptide"].map(bump).to_numpy()
    idx = val["pair_id"].to_numpy()
    a = stage6.score_differential("a", pairs, pd.Series(base, index=idx))
    b = stage6.score_differential("b", pairs, pd.Series(shifted, index=idx))
    for key in ("concordance", "spearman_delta_pooled",
                "median_allele_pair_spearman", "mae_delta_log1p"):
        assert a[key] == pytest.approx(b[key]), key


# --------------------------------------------------- 3. paired bootstrap

def test_paired_bootstrap_is_zero_against_itself(val, ens):
    def stat(idx, pred):
        return float(pred[idx].mean())

    r = stage6.paired_cluster_delta(val["cluster_id"].to_numpy(), stat, ens, ens,
                                    n_boot=50)
    assert r["delta"] == pytest.approx(0.0)
    assert r["ci95"] == (pytest.approx(0.0), pytest.approx(0.0))


def test_paired_bootstrap_is_deterministic(val, ens):
    base = stage6.read_predictions(BASELINE, val).to_numpy()

    def stat(idx, pred):
        return float(np.corrcoef(pred[idx], val[TARGET].to_numpy()[idx])[0, 1])

    kw = dict(n_boot=40, seed=evaluation.BOOTSTRAP_SEED)
    a = stage6.paired_cluster_delta(val["cluster_id"].to_numpy(), stat, base, ens, **kw)
    b = stage6.paired_cluster_delta(val["cluster_id"].to_numpy(), stat, base, ens, **kw)
    assert a["ci95"] == b["ci95"]


def test_paired_bootstrap_keeps_clusters_whole(val):
    """Every row of a drawn cluster must enter the resample together."""
    clusters = val["cluster_id"].to_numpy()
    seen: list[np.ndarray] = []

    def stat(idx, pred):
        seen.append(idx)
        return 0.0

    stage6.paired_cluster_delta(clusters, stat, np.zeros(len(val)),
                                np.zeros(len(val)), n_boot=3)
    sizes = pd.Series(clusters).value_counts()
    for idx in seen[1:]:  # seen[0] is the full-data call
        drawn = pd.Series(clusters[idx]).value_counts()
        # each drawn cluster appears a whole multiple of its true size
        assert ((drawn % sizes.reindex(drawn.index)) == 0).all()


def test_row_bootstrap_is_narrower_than_the_cluster_bootstrap(val, val_alleles, ens):
    """The contract's reason for resampling clusters, measured not asserted."""
    base = stage6.read_predictions(BASELINE, val).to_numpy()
    widths = stage6.resampling_unit_widths(val, base, ens, val_alleles, n_boot=120)
    w = widths.set_index("unit")["ci_width"]
    assert w["row"] < w["cluster"]
    assert w["peptide"] <= w["cluster"]


def test_bootstrap_rejects_nonfinite(val):
    bad = np.full(len(val), np.nan)
    with pytest.raises(ValueError):
        stage6.paired_cluster_delta(val["cluster_id"].to_numpy(),
                                    lambda idx, p: 0.0, bad, bad, n_boot=2)


# ------------------------------------------------------- 4. precision@10

def test_precision_report_wraps_the_contract_unchanged(val, val_alleles, ens):
    mine = stage6.precision_report("m", val, ens, val_alleles)
    theirs = evaluation.precision_at_k(val["allele"],
                                       np.expm1(val[TARGET].to_numpy()), ens,
                                       val_alleles)
    pd.testing.assert_series_equal(mine["precision_at_k"], theirs["precision_at_k"])
    pd.testing.assert_series_equal(mine["base_rate"], theirs["base_rate"])
    pd.testing.assert_series_equal(mine["ceiling"], theirs["ceiling"])


def test_constant_predictor_scores_its_base_rate(val, val_alleles):
    """The predeclared tie rule, restated as a property of the whole table."""
    flat = np.full(len(val), 0.7)
    table = stage6.precision_report("flat", val, flat, val_alleles)
    np.testing.assert_allclose(table["precision_at_k"].to_numpy(),
                               table["base_rate"].to_numpy(), atol=1e-12)
    assert stage6.precision_summary(table)["median_lift"] == pytest.approx(0.0)


def test_precision_never_exceeds_its_ceiling(val, val_alleles, ens):
    table = stage6.precision_report("m", val, ens, val_alleles)
    assert (table["precision_at_k"] <= table["ceiling"] + 1e-9).all()
    # ceiling_share is undefined, not 0 or 1, where an allele has no peptide
    # above 2 h at all: there is nothing for a top-10 to find, so dividing
    # would manufacture a score out of an impossible task.
    reachable = table["ceiling"] > 0
    assert (table.loc[reachable, "ceiling_share"] <= 1 + 1e-9).all()
    assert table.loc[~reachable, "ceiling_share"].isna().all()
    assert (table.loc[~reachable, "precision_at_k"] == 0).all()


# -------------------------------------- 5. nested near-neighbour evaluation

def test_near_neighbour_groups_never_straddle_a_split():
    check = stage6.verify_groups_within_split()
    assert check["n_groups_spanning_splits"] == 0


def test_near_neighbour_radius_must_stay_below_the_frozen_radius(train):
    with pytest.raises(ValueError, match="strictly below"):
        stage6.near_neighbour_groups(train, radius=3)


def test_mutant_pairs_are_same_allele_and_within_radius(train):
    pairs = stage6.mutant_pairs(train)
    assert len(pairs) > 0
    assert pairs["hamming"].between(1, stage6.MUTANT_RADIUS).all()
    assert (pairs["peptide_a"] < pairs["peptide_b"]).all()
    sample = pairs.sample(50, random_state=0)
    for a, b, h in sample[["peptide_a", "peptide_b", "hamming"]].itertuples(index=False):
        assert sum(x != y for x, y in zip(a, b)) == h
    # both rows of a pair carry the same allele and exist in the split
    lab = train.set_index("pair_id")[TARGET]
    np.testing.assert_allclose(
        pairs["delta_true"].to_numpy(),
        (lab.reindex(pairs["pair_id_b"]).to_numpy()
         - lab.reindex(pairs["pair_id_a"]).to_numpy()))


def test_nested_folds_hold_out_whole_peptides(train):
    folds = stage6.nested_folds(train)
    assert folds.nunique() == stage6.NESTED_FOLDS
    per_peptide = train.assign(f=folds).groupby("peptide")["f"].nunique()
    assert per_peptide.max() == 1, "a peptide is split across folds"
    sizes = folds.value_counts()
    assert sizes.max() - sizes.min() <= 0.02 * len(train)


def test_nested_folds_split_mutant_pairs_apart(train):
    """The design requirement: the model must see one mutant and predict the other."""
    folds = stage6.nested_folds(train)
    fold_map = pd.Series(folds.to_numpy(), index=train["pair_id"].to_numpy())
    pairs = stage6.mutant_pairs(train)
    fa = fold_map.reindex(pairs["pair_id_a"].to_numpy()).to_numpy()
    fb = fold_map.reindex(pairs["pair_id_b"].to_numpy()).to_numpy()
    assert (fa != fb).mean() > 0.5


def test_nested_folds_are_deterministic(train):
    a = stage6.nested_folds(train)
    b = stage6.nested_folds(train)
    pd.testing.assert_series_equal(a, b)


def test_constant_predictor_scores_chance_on_mutant_ranking(train):
    pairs = stage6.mutant_pairs(train)
    folds = stage6.nested_folds(train)
    fold_map = pd.Series(folds.to_numpy(), index=train["pair_id"].to_numpy())
    flat = pd.Series(np.zeros(len(train)), index=train["pair_id"].to_numpy())
    r = stage6.score_mutant_ranking("flat", pairs, flat, fold_map)
    assert r["concordance"] == pytest.approx(0.5)
    assert r["n_pairs_scored"] < r["n_pairs_available"]
    assert r["n_decidable"] + r["n_undecidable"] == r["n_pairs_scored"]


def test_perfect_predictor_scores_one_on_mutant_ranking(train):
    """The labels themselves must score 1.0 -- a check on the sign convention."""
    pairs = stage6.mutant_pairs(train)
    oracle = pd.Series(train[TARGET].to_numpy(), index=train["pair_id"].to_numpy())
    r = stage6.score_mutant_ranking("oracle", pairs, oracle)
    assert r["concordance"] == pytest.approx(1.0)


def test_ridge_reference_arm_runs_and_is_out_of_sample(train):
    folds = stage6.nested_folds(train)
    oof = stage6.ridge_oof_predictions(train, folds)
    assert len(oof) == len(train)
    assert np.isfinite(oof.to_numpy()).all()
    assert oof.index.tolist() == train["pair_id"].tolist()


# ------------------------------------------------------------- CLI smoke

def test_cli_emits_every_table(tmp_path):
    import subprocess

    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "stage6_report.py"),
           "--split", "val", "--n-boot", "20", "--stratum-min-rows", "10",
           "--out-dir", str(tmp_path), "--prefix", "smoke",
           f"a={BASELINE}", f"b={ENSEMBLE}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stderr
    for stem in ["summary", "per_allele", "distance_census", "distance_strata",
                 "differential", "differential_allele_pairs",
                 "precision_summary", "precision_per_allele", "paired_ci",
                 "resampling_units"]:
        assert (tmp_path / f"smoke_{stem}.csv").exists(), stem
    assert (tmp_path / "smoke_manifest.json").exists()


def test_cli_refuses_duplicate_arm_names(tmp_path):
    import subprocess

    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "stage6_report.py"),
           "--split", "val", "--skip-ci", "--out-dir", str(tmp_path),
           str(BASELINE), str(BASELINE)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert proc.returncode != 0
    assert "duplicate arm names" in proc.stderr
