"""Guards for the stage 7b leave-allele-out contract.

Two kinds of test here. The first kind protects the frozen artifacts: this is a
*second* contract, and it is only legitimate as long as it leaves
``EVALUATION.md`` and ``data/splits.csv` exactly as it found them and never
reads a test row. The second kind checks the mechanics -- fold disjointness,
ensemble parity, feature alignment by ``pair_id``.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab import allele_holdout as ah  # noqa: E402
from pepstab.data import REPO_ROOT, SPLITS_CSV, load_with_splits  # noqa: E402

EVALUATION_MD = REPO_ROOT / "EVALUATION.md"


@pytest.fixture(scope="module")
def cohort():
    return ah.load_cohort()


@pytest.fixture(scope="module")
def info(cohort):
    return ah.cohort_table(cohort)


# --- the frozen artifacts are untouched -------------------------------------

def test_frozen_splits_and_contract_are_readable_and_unmodified(cohort):
    """Loading the cohort must not disturb what it reads.

    Hashes are recomputed after the cohort has been built, so an accidental
    in-place write anywhere in the import path shows up here.
    """
    assert SPLITS_CSV.exists() and EVALUATION_MD.exists()
    for path in (SPLITS_CSV, EVALUATION_MD):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert len(digest) == 64
    # The frozen split still loads, and still covers every row.
    full = load_with_splits()
    assert len(full) == 28_166
    assert set(full["split"].unique()) == {"train", "val", "test"}


def test_cohort_excludes_every_test_row(cohort):
    """The hard rule, enforced structurally rather than by convention."""
    assert set(cohort["split"].unique()) == {"train", "val"}
    assert "test" not in set(cohort["split"])
    assert len(cohort) == 19_716 + 2_817


def test_cohort_splits_constant_names_the_two_allowed_splits():
    assert ah.COHORT_SPLITS == ("train", "val")


def test_cohort_rows_match_the_frozen_split_exactly(cohort):
    """Joined on pair_id through load_with_splits, never positionally."""
    full = load_with_splits()
    want = full[full["split"].isin(("train", "val"))]
    assert sorted(cohort["pair_id"]) == sorted(want["pair_id"])


# --- the panel --------------------------------------------------------------

def test_eligibility_selects_68_alleles(cohort):
    assert len(ah.eligible_holdout_alleles(cohort)) == 68


def test_eligibility_is_insensitive_to_the_bar_inside_the_gap(cohort):
    """In-scope counts jump 30 -> 177, so the 50-row bar is not a judgement
    call. If this ever fails, the bar has become a choice and must be argued."""
    base = ah.eligible_holdout_alleles(cohort, min_rows=50)
    for bar in (31, 40, 100, 150, 177):
        assert ah.eligible_holdout_alleles(cohort, min_rows=bar) == base


def test_eligibility_uses_labels_not_predictions(cohort):
    """The panel must be identical for every arm, so it may only depend on the
    cohort and the labels."""
    a = ah.eligible_holdout_alleles(cohort)
    shuffled = cohort.sample(frac=1.0, random_state=0)
    assert ah.eligible_holdout_alleles(shuffled) == a


# --- distances and strata ---------------------------------------------------

def test_distance_matrix_is_symmetric_with_zero_diagonal(cohort):
    d = ah.pseudoseq_distances(cohort)
    assert d.shape == (75, 75)
    assert np.array_equal(d.to_numpy(), d.to_numpy().T)
    assert np.all(np.diag(d.to_numpy()) == 0)


def test_the_known_pseudosequence_collision_shows_up_as_distance_zero(cohort):
    """audit_summary.md §1: these two share a 34-residue pseudosequence."""
    d = ah.pseudoseq_distances(cohort)
    assert d.loc["HLA-B*14:01(C67S)", "HLA-B*14:02(C67S)"] == 0


def test_strata_partition_every_reachable_distance():
    seen = [ah.stratum_of(d) for d in range(0, 35)]
    assert set(seen) == set(ah.DISTANCE_STRATA)
    with pytest.raises(ValueError):
        ah.stratum_of(-1)


def test_stratum_sizes_match_the_predeclared_table(info):
    """reports/stage7_allele_holdout.md §3, fixed before any fold ran."""
    counts = info["stratum"].value_counts().to_dict()
    assert counts["near (d<=1)"] == 25
    assert counts["intermediate (d=2-3)"] == 23
    assert counts["distant (d>=4)"] == 20
    assert sum(counts.values()) == 68


def test_degenerate_alleles_are_flagged_not_dropped(info):
    flagged = set(info.index[info["degenerate_pseudoseq"]])
    assert flagged == {"HLA-B*14:01(C67S)", "HLA-B*14:02(C67S)"}
    assert flagged <= set(info.index), "degenerate alleles must stay in the panel"


# --- the measured confound --------------------------------------------------

def test_holding_out_an_allele_does_not_hold_out_its_peptide_panel(info):
    """The plan's reason 1, measured rather than assumed.

    If this ever fails, section 4 of the report is wrong and must be rewritten
    before any stratum number is quoted.
    """
    assert info["peptide_seen_share"].median() == pytest.approx(1.0)
    assert info["peptide_seen_share"].min() > 0.5


def test_cohort_table_carries_the_panel_difficulty_columns(info):
    """Distance is confounded with panel difficulty here, so the columns that
    show it must travel with the distances."""
    for col in ("zero_share", "median_label_log1p", "peptide_seen_share",
                "dist_to_nearest_fit_allele", "stratum"):
        assert col in info.columns
    assert info["zero_share"].between(0, 1).all()


# --- arms -------------------------------------------------------------------

def test_arm_rejects_a_mismatched_ensemble_size():
    arm = ah.sequence_arm(seeds=(0,))  # 2 representations x 1 seed = 2
    with pytest.raises(ValueError, match="ensemble"):
        arm.validate()


def test_sequence_arm_at_three_seeds_matches_the_contract():
    arm = ah.sequence_arm(seeds=(0, 1, 2)).validate()
    assert arm.n_members() == ah.ENSEMBLE_SIZE == 6


def test_arm_rejects_a_representation_with_no_config():
    arm = ah.Arm(name="broken",
                 sources=(ah.FeatureSource("x", "sequence", "pep_pseudo", "onehot"),),
                 configs={}, seeds=(0,))
    with pytest.raises(ValueError, match="no config"):
        arm.validate(expected=1)


def test_npy_feature_source_aligns_by_pair_id_not_by_position(cohort, tmp_path):
    """A feature table in a different row order must still line up. Aligning
    positionally is how a feature matrix silently desynchronises from labels."""
    small = cohort.head(200)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(small))
    matrix = np.arange(len(small) * 3, dtype=np.float32).reshape(len(small), 3)

    mpath, ipath = tmp_path / "m.npy", tmp_path / "i.npy"
    np.save(mpath, matrix[order])
    np.save(ipath, small["pair_id"].to_numpy()[order])

    src = ah.FeatureSource("esm", "npy", matrix_path=str(mpath),
                           pair_id_path=str(ipath))
    assert np.array_equal(src.build(small), matrix)


def test_npy_feature_source_rejects_missing_rows(cohort, tmp_path):
    small = cohort.head(50)
    mpath, ipath = tmp_path / "m.npy", tmp_path / "i.npy"
    np.save(mpath, np.zeros((40, 3), dtype=np.float32))
    np.save(ipath, small["pair_id"].to_numpy()[:40])
    src = ah.FeatureSource("esm", "npy", matrix_path=str(mpath),
                           pair_id_path=str(ipath))
    with pytest.raises(ValueError, match="no features"):
        src.build(small)


def test_npy_feature_source_rejects_a_length_mismatch(tmp_path):
    mpath, ipath = tmp_path / "m.npy", tmp_path / "i.npy"
    np.save(mpath, np.zeros((10, 3), dtype=np.float32))
    np.save(ipath, np.arange(9))
    src = ah.FeatureSource("esm", "npy", matrix_path=str(mpath),
                           pair_id_path=str(ipath))
    with pytest.raises(ValueError, match="pair_id index"):
        src.build(pd.DataFrame({"pair_id": np.arange(9)}))


# --- folds ------------------------------------------------------------------

def test_inner_dev_fold_keeps_peptide_clusters_whole(cohort):
    fit = cohort[cohort["allele"] != "HLA-A*02:01"]
    is_dev = ah.inner_dev_mask(fit)
    clusters = fit["cluster_id"].to_numpy()
    assert not (set(clusters[is_dev]) & set(clusters[~is_dev]))
    assert 0.05 < is_dev.mean() < 0.15


def test_inner_dev_fold_is_deterministic(cohort):
    fit = cohort[cohort["allele"] != "HLA-A*02:01"]
    assert np.array_equal(ah.inner_dev_mask(fit), ah.inner_dev_mask(fit))


def test_run_fold_never_fits_on_the_held_out_allele(cohort, monkeypatch):
    """The defining property of the design. Checked by capturing what the model
    is actually handed, not by inspecting the call site."""
    held = "HLA-B*57:01"
    small = cohort[cohort["allele"].isin([held, "HLA-B*58:01", "HLA-A*02:01"])]
    small = small.reset_index(drop=True)
    features = {"onehot": np.zeros((len(small), 4), dtype=np.float32)}
    # Mark each row with its allele so a leak is visible in the features.
    features["onehot"][:, 0] = (small["allele"] == held).to_numpy()

    seen = []

    class Spy(ah.MLPRegressor):
        def fit(self, X, y, X_dev, y_dev, sample_weight=None):
            seen.append(X[:, 0].sum() + X_dev[:, 0].sum())
            self.weights_ = [np.zeros((X.shape[1], 1), dtype=np.float32)]
            self.biases_ = [np.zeros(1, dtype=np.float32)]
            self.y_offset_ = 0.0
            self.best_epoch_ = self.n_epochs_ = 1
            self.dev_mse_ = 0.0
            self.fit_seconds_ = 0.0
            return self

    monkeypatch.setattr(ah, "MLPRegressor", Spy)
    arm = ah.Arm(name="t",
                 sources=(ah.FeatureSource("onehot", "sequence", "pep_pseudo",
                                           "onehot"),),
                 configs={"onehot": ((4,), 1e-5)}, seeds=(0,))
    pred, records = ah.run_fold(small, features, arm, held)

    assert seen and all(s == 0 for s in seen), "held-out allele reached the fit"
    assert len(pred) == int((small["allele"] == held).sum())
    assert records[0]["n_eval_rows"] == len(pred)


def test_run_fold_rejects_an_absent_allele(cohort):
    arm = ah.sequence_arm(seeds=(0,))
    with pytest.raises(ValueError, match="no rows"):
        ah.run_fold(cohort.head(100), {}, arm, "HLA-ZZ:99")


def test_score_fold_rejects_a_length_mismatch(cohort):
    with pytest.raises(ValueError, match="predictions for"):
        ah.score_fold(cohort, "HLA-B*57:01", np.zeros(3))


def test_score_fold_rejects_non_finite_predictions(cohort):
    n = int((cohort["allele"] == "HLA-B*57:01").sum())
    bad = np.zeros(n)
    bad[0] = np.nan
    with pytest.raises(ValueError):
        ah.score_fold(cohort, "HLA-B*57:01", bad)


def test_score_fold_recovers_a_perfect_ranking(cohort):
    rows = cohort[cohort["allele"] == "HLA-B*57:01"]
    out = ah.score_fold(cohort, "HLA-B*57:01", rows["y_log1p"].to_numpy())
    assert out["spearman"] == pytest.approx(1.0)
    assert out["mae_log1p"] == pytest.approx(0.0)
    assert out["n_rows"] == len(rows)


def test_score_fold_gives_a_constant_predictor_its_base_rate(cohort):
    """Matches the frozen contract's tie handling: a constant predictor scores
    its base rate on precision@10 rather than whatever the first rows hold."""
    rows = cohort[cohort["allele"] == "HLA-B*57:01"]
    out = ah.score_fold(cohort, "HLA-B*57:01", np.ones(len(rows)))
    assert out["precision_at_10"] == pytest.approx(out["base_rate"])
    assert np.isnan(out["spearman"])


# --- aggregation ------------------------------------------------------------

def _fake_per_allele(info, values):
    return pd.DataFrame({
        "held_out_allele": list(info.index),
        "n_rows": info["n_rows"].to_numpy(),
        "spearman": values,
        "mae_log1p": np.full(len(info), 0.5),
        "precision_at_10": np.full(len(info), 0.5),
    })


def test_stratum_summary_reports_every_stratum_plus_the_controls(info):
    rng = np.random.default_rng(0)
    per = _fake_per_allele(info, rng.uniform(0, 1, len(info)))
    out = ah.stratum_summary(per, info, n_boot=200)
    names = set(out["stratum"])
    assert set(ah.DISTANCE_STRATA) <= names
    assert "all strata pooled" in names
    assert "near (d<=1), excl. shared-pseudoseq pair" in names


def test_stratum_summary_interval_brackets_the_point_estimate(info):
    rng = np.random.default_rng(1)
    per = _fake_per_allele(info, rng.uniform(0, 1, len(info)))
    out = ah.stratum_summary(per, info, n_boot=500)
    assert (out["ci95_low"] <= out["median_spearman"] + 1e-9).all()
    assert (out["median_spearman"] <= out["ci95_high"] + 1e-9).all()
    assert (out["n_alleles"] > 0).all()


def test_stratum_summary_is_deterministic(info):
    rng = np.random.default_rng(2)
    per = _fake_per_allele(info, rng.uniform(0, 1, len(info)))
    a = ah.stratum_summary(per, info, n_boot=200)
    b = ah.stratum_summary(per, info, n_boot=200)
    pd.testing.assert_frame_equal(a, b)


def test_paired_bootstrap_centres_on_a_known_shift(info):
    rng = np.random.default_rng(3)
    base = rng.uniform(0.2, 0.8, len(info))
    a = _fake_per_allele(info, base)
    b = _fake_per_allele(info, base + 0.1)
    out = ah.paired_allele_bootstrap(a, b, info, n_boot=1000)
    assert out["delta_median_spearman"] == pytest.approx(0.1, abs=1e-9)
    assert out["ci95"][0] <= 0.1 <= out["ci95"][1]
    assert out["n_alleles"] == 68


def test_paired_bootstrap_is_paired_not_two_independent_intervals(info):
    """Identical arms must give a zero-width interval. Two separate intervals
    would not, which is the whole reason the contract pairs them."""
    rng = np.random.default_rng(4)
    vals = rng.uniform(0, 1, len(info))
    a = _fake_per_allele(info, vals)
    out = ah.paired_allele_bootstrap(a, a.copy(), info, n_boot=500)
    assert out["delta_median_spearman"] == pytest.approx(0.0)
    assert out["ci95"] == pytest.approx((0.0, 0.0))


def test_paired_bootstrap_can_restrict_to_one_stratum(info):
    rng = np.random.default_rng(5)
    base = rng.uniform(0.2, 0.8, len(info))
    a = _fake_per_allele(info, base)
    b = _fake_per_allele(info, base + 0.1)
    out = ah.paired_allele_bootstrap(a, b, info, stratum="distant (d>=4)",
                                     n_boot=500)
    assert out["n_alleles"] == 20
    assert out["stratum"] == "distant (d>=4)"


# --- arms must supply features, not predictions -----------------------------

def test_prediction_files_are_rejected_as_an_arm():
    """A preds/*.csv came from a model fitted on every allele. Scoring it under
    a leave-allele-out contract would report that leak as pan-allele
    generalisation -- a number that would look like a win."""
    from scripts.stage7_allele_holdout import parse_npy

    for spec in ("esm=preds/esm_ensemble.csv:preds/ids.csv",
                 "esm=features/esm.npy:preds/esm_ensemble.csv"):
        with pytest.raises(SystemExit, match="prediction file"):
            parse_npy([spec])


def test_npy_arm_specs_are_accepted():
    from scripts.stage7_allele_holdout import parse_npy

    got = parse_npy(["esm=features/esm2.npy:features/esm2_pair_ids.npy"])
    assert len(got) == 1
    assert got[0].name == "esm" and got[0].kind == "npy"
    assert got[0].matrix_path.endswith("esm2.npy")
