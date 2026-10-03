"""Guards on stage 2b weak-binder augmentation. Run with: .venv/bin/python -m pytest -q

Augmentation is the one stage that writes *new training rows*, so it is the one
stage where a mistake adds data instead of losing it. Three things have to hold:

1. the assumed label is really an assumption and really zero, and never lands on
   a pair that already carries a measured half-life;
2. no augmented peptide sits within 3 substitutions of an inner-dev, validation
   or test peptide, **across all alleles** -- checked here by recomputing the
   distances from the frozen splits, not by trusting the manifest's own columns;
3. the two matched arms differ in the source of their negatives and in nothing
   else -- same alleles, same per-allele counts.

The weighted loss gets its own checks, because the measured-only arm is only a
valid baseline if passing uniform weights is bit-identical to passing none.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.augment import (
    ASSUMED_THALF_HOURS,
    ASSUMED_Y_LOG1P,
    CAP_FRACTION,
    MANIFEST_COLUMNS,
    MIN_HAMMING,
    WEAK_NM,
    annotate_candidates,
    attach_hla,
    holdout_peptides,
    matched_counts,
    min_hamming,
    per_allele_cap,
    sample_per_allele,
    verify_manifest,
)
from pepstab.data import PEPTIDE_LENGTH, load_with_splits
from pepstab import mlp as mlp_module
from pepstab.mlp import MLPConfig, MLPRegressor
from pepstab.splits import HAMMING_THRESHOLD
from scripts.baseline_sequence import inner_folds

REPO_ROOT = Path(__file__).resolve().parent.parent
AUG_DIR = REPO_ROOT / "data" / "augmentation"
AFFINITY_CSV = (REPO_ROOT / "data" / "data_augmentation_iedb"
                / "affinity_reference_75alleles.csv")

MATCHED_ARMS = ("measured_affinity", "predicted_affinity")
ALL_ARMS = MATCHED_ARMS + ("predicted_affinity_full",)


@pytest.fixture(scope="module")
def split_df():
    return load_with_splits()


@pytest.fixture(scope="module")
def fold(split_df):
    return inner_folds(split_df[split_df.split == "train"])


@pytest.fixture(scope="module")
def manifests():
    missing = [a for a in ALL_ARMS if not (AUG_DIR / f"{a}.csv").exists()]
    if missing:
        pytest.skip(f"manifests not built: {missing}. Run "
                    "scripts/augment_affinity.py")
    return {a: pd.read_csv(AUG_DIR / f"{a}.csv") for a in ALL_ARMS}


# --- min_hamming ----------------------------------------------------------

def test_min_hamming_counts_substitutions():
    assert min_hamming(["AAAAAAAAA"], ["AAAAAAAAA"]).tolist() == [0]
    assert min_hamming(["AAAAAAAAA"], ["CAAAAAAAA"]).tolist() == [1]
    assert min_hamming(["AAAAAAAAA"], ["CCCCCCCCC"]).tolist() == [9]


def test_min_hamming_takes_the_nearest_reference():
    d = min_hamming(["AAAAAAAAA"], ["CCCCCCCCC", "CCAAAAAAA", "CCCCAAAAA"])
    assert d.tolist() == [2]


def test_min_hamming_empty_reference_is_maximally_distant():
    """A minimum over nothing must not admit a candidate by accident."""
    d = min_hamming(["AAAAAAAAA", "CCCCCCCCC"], [])
    assert d.tolist() == [PEPTIDE_LENGTH, PEPTIDE_LENGTH]
    assert (d >= MIN_HAMMING).all()


def test_min_hamming_is_chunk_invariant():
    rng = np.random.default_rng(0)
    alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    cands = ["".join(rng.choice(alphabet, 9)) for _ in range(300)]
    refs = ["".join(rng.choice(alphabet, 9)) for _ in range(120)]
    np.testing.assert_array_equal(min_hamming(cands, refs, chunk=7),
                                  min_hamming(cands, refs, chunk=4096))


# --- the held-out peptide set --------------------------------------------

def test_holdout_covers_dev_val_and_test(split_df, fold):
    train = split_df[split_df.split == "train"]
    held = set(holdout_peptides(split_df, fold))
    dev = set(train.loc[(fold == "dev").to_numpy(), "peptide"])
    val = set(split_df.loc[split_df.split == "val", "peptide"])
    test = set(split_df.loc[split_df.split == "test", "peptide"])
    assert dev <= held and val <= held and test <= held
    assert held == dev | val | test
    # The fit peptides are what augmentation is allowed to sit beside.
    fit = set(train.loc[(fold == "fit").to_numpy(), "peptide"])
    assert not (fit & held), "a peptide cannot be both fit and held out"


def test_holdout_rejects_a_misaligned_fold(split_df, fold):
    with pytest.raises(ValueError, match="fold has"):
        holdout_peptides(split_df, fold.iloc[:-1])


# --- annotate_candidates --------------------------------------------------

def test_annotation_flags_follow_the_distances(split_df, fold):
    cand = pd.DataFrame({"allele": ["HLA-A*02:01"] * 4,
                         "peptide": ["AAAAAAAAA", "CCCCCCCCC",
                                     "WWWWWWWWW", "YYYYYYYYY"]})
    out = annotate_candidates(cand, split_df, fold)
    assert (out["ok_inner_folds"] == (out["min_dist_holdout"] >= MIN_HAMMING)).all()
    assert (out["ok_cv_folds"] == (out["ok_inner_folds"]
                                   & (out["min_dist_train"] >= MIN_HAMMING))).all()


def test_cv_folds_rule_is_strictly_stronger(split_df, fold, manifests):
    """Under cv_folds() every training peptide is some member's stopping peptide."""
    out = annotate_candidates(manifests["measured_affinity"][["allele", "peptide"]],
                              split_df, fold)
    assert out["ok_inner_folds"].all()
    assert out["ok_cv_folds"].sum() < out["ok_inner_folds"].sum(), (
        "if the two rules agreed, the ensemble caveat would be unnecessary")


def test_a_held_out_peptide_is_never_admissible(split_df, fold):
    """The exact leak the reference table's own padding flag lets through."""
    val_peptide = split_df.loc[split_df.split == "val", "peptide"].iloc[0]
    # Offer it under a *different* allele, which is how it slips past a
    # pair-wise "not in the stability dataset" check.
    cand = pd.DataFrame({"allele": ["HLA-B*15:01"], "peptide": [val_peptide]})
    out = annotate_candidates(cand, split_df, fold)
    assert out["min_dist_holdout"].iloc[0] == 0
    assert not out["ok_inner_folds"].iloc[0]


# --- caps and matched counts ---------------------------------------------

def test_cap_is_the_declared_fraction_of_fit_rows(split_df, fold):
    train = split_df[split_df.split == "train"]
    fit = train[(fold == "fit").to_numpy()]
    cap = per_allele_cap(fit, CAP_FRACTION)
    counts = fit.groupby("allele").size()
    assert set(cap.index) == set(counts.index)
    np.testing.assert_array_equal(cap.to_numpy(),
                                  (counts * CAP_FRACTION).astype(int).to_numpy())


def test_matched_counts_take_the_elementwise_minimum():
    cap = pd.Series({"a": 10, "b": 10, "c": 10})
    measured = pd.Series({"a": 4, "b": 50})          # c absent entirely
    predicted = pd.Series({"a": 7, "b": 3, "c": 99})
    got = matched_counts(cap, measured, predicted)
    assert got.to_dict() == {"a": 4, "b": 3, "c": 0}
    assert list(got.index) == list(cap.index), "the allele panel must not shrink"


def test_matched_counts_keep_unsupported_alleles_at_zero():
    """They stay in the frame so they stay in the evaluation panel."""
    cap = pd.Series({"a": 5, "b": 5})
    got = matched_counts(cap, pd.Series(dtype=int))
    assert got.to_dict() == {"a": 0, "b": 0}


# --- sampling -------------------------------------------------------------

def _pool(n_per_allele=20, alleles=("a", "b")):
    rows = [{"allele": al, "peptide": f"P{i:08d}"}
            for al in alleles for i in range(n_per_allele)]
    return pd.DataFrame(rows)


def test_sampling_is_deterministic_and_respects_counts():
    pool, counts = _pool(), pd.Series({"a": 5, "b": 3})
    first = sample_per_allele(pool, counts, seed=7)
    second = sample_per_allele(pool, counts, seed=7)
    pd.testing.assert_frame_equal(first.reset_index(drop=True),
                                  second.reset_index(drop=True))
    assert first.groupby("allele").size().to_dict() == {"a": 5, "b": 3}
    assert sample_per_allele(pool, counts, seed=8).peptide.tolist() != \
        first.peptide.tolist()


def test_sampling_refuses_to_overdraw():
    with pytest.raises(ValueError, match="pool holds"):
        sample_per_allele(_pool(n_per_allele=2), pd.Series({"a": 5}), seed=0)


def test_sampling_is_nested_in_draw_order():
    """What lets one draw serve both the matched and full-coverage arms."""
    pool = _pool(n_per_allele=30)
    full = sample_per_allele(pool, pd.Series({"a": 12, "b": 12}), seed=3)
    smaller = full[full.draw_rank < 5]
    assert smaller.groupby("allele").size().to_dict() == {"a": 5, "b": 5}
    # And a smaller draw at the same seed agrees on the first rows.
    assert (sample_per_allele(pool, pd.Series({"a": 12, "b": 12}), seed=3)
            .peptide.tolist() == full.peptide.tolist())


def test_sampling_skips_zero_count_alleles():
    out = sample_per_allele(_pool(), pd.Series({"a": 0, "b": 4}), seed=1)
    assert out.allele.unique().tolist() == ["b"]


# --- verify_manifest ------------------------------------------------------

def _clean_manifest(manifests):
    return manifests["measured_affinity"].head(50).copy()


def test_verify_accepts_a_committed_manifest(split_df, fold, manifests):
    for name, manifest in manifests.items():
        checks = verify_manifest(manifest, split_df, fold)
        assert checks["n_rows"] == len(manifest), name
        assert checks["min_dist_holdout"] >= MIN_HAMMING, name


def test_verify_rejects_a_nonzero_assumed_label(split_df, fold, manifests):
    bad = _clean_manifest(manifests)
    bad.loc[bad.index[0], "thalf_hours"] = 3.5
    with pytest.raises(ValueError, match="non-zero half-life"):
        verify_manifest(bad, split_df, fold)


def test_verify_rejects_a_nonzero_target(split_df, fold, manifests):
    bad = _clean_manifest(manifests)
    bad.loc[bad.index[0], "y_log1p"] = 0.7
    with pytest.raises(ValueError, match="non-zero y_log1p"):
        verify_manifest(bad, split_df, fold)


def test_verify_rejects_a_peptide_near_a_held_out_one(split_df, fold, manifests):
    bad = _clean_manifest(manifests)
    bad.loc[bad.index[0], "peptide"] = split_df.loc[
        split_df.split == "test", "peptide"].iloc[0]
    with pytest.raises(ValueError, match="distance rule"):
        verify_manifest(bad, split_df, fold)


def test_verify_rejects_overwriting_a_measured_pair(split_df, fold, manifests):
    """An assumed zero must never displace a measurement."""
    bad = _clean_manifest(manifests)
    train = split_df[split_df.split == "train"]
    fit = train[(fold == "fit").to_numpy()].iloc[0]
    bad.loc[bad.index[0], ["allele", "peptide"]] = [fit.allele, fit.peptide]
    with pytest.raises(ValueError, match="measured half-life"):
        verify_manifest(bad, split_df, fold)


def test_verify_rejects_a_duplicated_pair(split_df, fold, manifests):
    bad = _clean_manifest(manifests)
    doubled = pd.concat([bad, bad.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="repeats an"):
        verify_manifest(doubled, split_df, fold)


def test_verify_rejects_an_unknown_protocol(split_df, fold, manifests):
    with pytest.raises(ValueError, match="unknown protocol"):
        verify_manifest(_clean_manifest(manifests), split_df, fold,
                        protocol="outer_folds")


def test_verify_handles_an_empty_manifest(split_df, fold, manifests):
    empty = manifests["measured_affinity"].iloc[:0]
    assert verify_manifest(empty, split_df, fold)["n_rows"] == 0


# --- the committed manifests ---------------------------------------------

def test_manifests_carry_the_declared_schema(manifests):
    for name, manifest in manifests.items():
        assert list(manifest.columns) == list(MANIFEST_COLUMNS), name


def test_assumed_labels_are_zero_on_both_scales(manifests):
    for name, manifest in manifests.items():
        assert (manifest.thalf_hours == ASSUMED_THALF_HOURS).all(), name
        assert (manifest.y_log1p == ASSUMED_Y_LOG1P).all(), name
        assert manifest.label_provenance.str.contains("assumed").all(), name


def test_no_manifest_peptide_is_near_a_held_out_peptide(split_df, fold, manifests):
    """Recomputed from the splits, not read from the manifest's own columns."""
    holdout = holdout_peptides(split_df, fold)
    for name, manifest in manifests.items():
        peptides = sorted(manifest.peptide.unique())
        d = min_hamming(peptides, holdout)
        assert d.min() > HAMMING_THRESHOLD, (
            f"{name}: nearest held-out peptide at Hamming {d.min()}")


def test_manifest_rows_never_duplicate_a_measured_pair(split_df, manifests):
    measured = set(zip(split_df.allele, split_df.peptide))
    for name, manifest in manifests.items():
        clash = measured & set(zip(manifest.allele, manifest.peptide))
        assert not clash, f"{name}: {len(clash)} pair(s) already measured"


def test_manifest_alleles_exist_in_the_dataset(split_df, manifests):
    known = set(split_df.allele.unique())
    for name, manifest in manifests.items():
        assert set(manifest.allele) <= known, name
        # attach_hla is how the training script gets pseudosequences; it must
        # not silently drop a row.
        assert len(attach_hla(manifest, split_df)) == len(manifest), name


def test_attach_hla_rejects_an_unknown_allele(split_df, manifests):
    bad = manifests["measured_affinity"].head(3).copy()
    bad.loc[bad.index[0], "allele"] = "HLA-Z*99:99"
    with pytest.raises(ValueError, match="absent from the stability dataset"):
        attach_hla(bad, split_df)


def test_the_two_matched_arms_are_matched(manifests):
    """Same alleles and the same count on each -- the source is the only change."""
    a = manifests["measured_affinity"].groupby("allele").size()
    b = manifests["predicted_affinity"].groupby("allele").size()
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index(),
                                   check_names=False)


def test_the_matched_arms_draw_from_different_peptide_universes(manifests):
    """Otherwise 'does the source matter' would have no content."""
    a = set(manifests["measured_affinity"].peptide)
    b = set(manifests["predicted_affinity"].peptide)
    assert len(a & b) < 0.05 * min(len(a), len(b))


def test_the_full_arm_is_a_superset_with_wider_coverage(manifests):
    matched = set(zip(manifests["predicted_affinity"].allele,
                      manifests["predicted_affinity"].peptide))
    full = set(zip(manifests["predicted_affinity_full"].allele,
                   manifests["predicted_affinity_full"].peptide))
    assert matched <= full
    assert (manifests["predicted_affinity_full"].allele.nunique()
            > manifests["predicted_affinity"].allele.nunique())


def test_added_rows_stay_under_the_per_allele_cap(split_df, fold, manifests):
    train = split_df[split_df.split == "train"]
    cap = per_allele_cap(train[(fold == "fit").to_numpy()], CAP_FRACTION)
    for name, manifest in manifests.items():
        counts = manifest.groupby("allele").size()
        over = counts[counts > cap.reindex(counts.index)]
        assert over.empty, f"{name}: over cap on {over.to_dict()}"


def test_measured_arm_rows_are_genuinely_weak_binders(manifests):
    """Each row traces back to a measurement or lower bound at >= 20,000 nM."""
    ref = pd.read_csv(AFFINITY_CSV)
    ref = ref.set_index(["dataset_allele", "peptide"])
    manifest = manifests["measured_affinity"]
    idx = pd.MultiIndex.from_arrays([manifest.allele, manifest.peptide])
    assert idx.isin(ref.index).all(), "a row is not in the affinity reference"
    rows = ref.loc[idx]
    assert (rows.affinity_nM >= WEAK_NM).all()
    assert rows.inequality.isin(["=", ">"]).all(), (
        "'<' is an upper bound and cannot establish weak binding")
    assert not rows.engineered_construct_mismatch.any(), (
        "a C67S construct cannot take a wild-type affinity")


def test_predicted_arm_rows_are_natural_peptides_with_provenance(manifests):
    for name in ("predicted_affinity", "predicted_affinity_full"):
        manifest = manifests[name]
        assert (manifest.affinity_kind == "predicted").all(), name
        assert (manifest.affinity_nM > WEAK_NM).all(), name
        assert manifest.protein_source.str.len().gt(0).all(), name
        assert (manifest.protein_offset >= 0).all(), name


def test_predicted_arm_is_not_contradicted_by_measured_affinity(manifests):
    """A measured strong binder is not a weak binder, whatever the model says."""
    ref = pd.read_csv(AFFINITY_CSV)
    strong = set(zip(ref.loc[ref.affinity_nM < WEAK_NM, "dataset_allele"],
                     ref.loc[ref.affinity_nM < WEAK_NM, "peptide"]))
    for name in ("predicted_affinity", "predicted_affinity_full"):
        manifest = manifests[name]
        assert not strong & set(zip(manifest.allele, manifest.peptide)), name


def test_measured_arm_peptides_are_more_reused_than_predicted_ones(manifests):
    """The assay panel is peptide-poor; random natural sampling is not.

    This is the structural difference the source comparison is testing, so it
    should be visible in the manifests themselves.
    """
    a, b = manifests["measured_affinity"], manifests["predicted_affinity"]
    assert len(a) == len(b)
    assert a.peptide.nunique() < b.peptide.nunique()


# --- the weighted loss ----------------------------------------------------

def _toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] + np.maximum(X[:, 2], 0.0)).astype(np.float32)
    return X, y


def test_uniform_weights_are_bit_identical_to_no_weights():
    """The measured-only arm is only a valid baseline if this holds exactly."""
    X, y = _toy()
    cfg = MLPConfig(hidden=(16,), max_epochs=20, patience=20, seed=5)
    plain = MLPRegressor(cfg).fit(X[:300], y[:300], X[300:], y[300:])
    weighted = MLPRegressor(cfg).fit(X[:300], y[:300], X[300:], y[300:],
                                     sample_weight=np.ones(300, dtype=np.float32))
    np.testing.assert_array_equal(plain.predict(X), weighted.predict(X))
    assert plain.y_offset_ == weighted.y_offset_


def test_weights_move_the_target_offset():
    """A block of assumed zeros at weight 0.1 must pull a tenth as hard."""
    X, y = _toy()
    y = y.copy()
    y[:100] = 0.0
    cfg = MLPConfig(hidden=(8,), max_epochs=1, patience=1, seed=0)
    w = np.ones(300, dtype=np.float32)
    w[:100] = 0.1
    light = MLPRegressor(cfg).fit(X[:300], y[:300], X[300:], y[300:], sample_weight=w)
    heavy = MLPRegressor(cfg).fit(X[:300], y[:300], X[300:], y[300:])
    assert light.y_offset_ > heavy.y_offset_
    expected = float(np.dot(w, y[:300]) / w.sum())
    assert light.y_offset_ == pytest.approx(expected, rel=1e-6)


def test_zero_weight_rows_do_not_affect_the_fit():
    """A weight of 0 has to mean 'absent', or the weighting is not a weighting."""
    X, y = _toy()
    cfg = MLPConfig(hidden=(12,), max_epochs=15, patience=15, seed=2)
    junk = np.full((40, 6), 9.0, dtype=np.float32)
    junk_y = np.full(40, -50.0, dtype=np.float32)
    X_aug = np.concatenate([X[:300], junk])
    y_aug = np.concatenate([y[:300], junk_y])
    w = np.concatenate([np.ones(300), np.zeros(40)]).astype(np.float32)
    with_junk = MLPRegressor(cfg).fit(X_aug, y_aug, X[300:], y[300:], sample_weight=w)
    clean = MLPRegressor(cfg).fit(X[:300], y[:300], X[300:], y[300:])
    # Not bit-identical -- 40 extra rows move the minibatch boundaries -- so the
    # claim is that the junk carries no pull, not that nothing moved. Rows with
    # a label 50 standard deviations out would wreck the fit at any real weight.
    assert with_junk.y_offset_ == pytest.approx(float(y[:300].mean()), rel=1e-6)
    assert with_junk.dev_mse_ == pytest.approx(clean.dev_mse_, rel=0.25)


def test_weighted_gradients_match_finite_differences(monkeypatch):
    """Same check as the unweighted gradients, with a non-uniform weight vector."""
    monkeypatch.setattr(mlp_module, "DTYPE", np.float64)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    y = rng.normal(size=20)
    w = rng.uniform(0.05, 1.5, size=20)
    model = MLPRegressor(MLPConfig(hidden=(7, 4), l2=0.0, seed=1))
    model._init_params(5, np.random.default_rng(3))
    model.y_offset_ = 0.0

    def loss() -> float:
        pred, _ = model._forward(X)
        return float(np.sum(w * (pred - y) ** 2) / len(y))

    pred, acts = model._forward(X)
    gw, gb = model._backward(acts, pred - y, w)

    eps, worst = 1e-6, 0.0
    for k in range(len(model.weights_)):
        for arr, grad in ((model.weights_[k], gw[k]), (model.biases_[k], gb[k])):
            for ix in list(np.ndindex(arr.shape))[:6]:
                old = float(arr[ix])
                arr[ix] = old + eps
                high = loss()
                arr[ix] = old - eps
                low = loss()
                arr[ix] = old
                numeric = (high - low) / (2 * eps)
                worst = max(worst, abs(numeric - grad[ix])
                            / max(1e-6, abs(numeric) + abs(grad[ix])))
    assert worst < 1e-6, f"worst relative gradient error {worst:.2e}"


@pytest.mark.parametrize("bad,match", [
    (np.ones(5), "expected"),
    (np.full(300, -0.1), "non-negative"),
    (np.full(300, np.nan), "finite"),
    (np.zeros(300), "sums to 0"),
])
def test_bad_weights_are_rejected(bad, match):
    X, y = _toy()
    cfg = MLPConfig(hidden=(4,), max_epochs=1, patience=1)
    with pytest.raises(ValueError, match=match):
        MLPRegressor(cfg).fit(X[:300], y[:300], X[300:], y[300:], sample_weight=bad)


# --- the affinity transform ----------------------------------------------

def test_affinity_transform_roundtrips_and_brackets_the_threshold():
    from scripts.augment_affinity import (
        WEAK_UNIT,
        affinity_to_unit,
        unit_to_affinity,
    )
    nm = np.array([1.0, 100.0, WEAK_NM, 50_000.0])
    np.testing.assert_allclose(unit_to_affinity(affinity_to_unit(nm)), nm, rtol=1e-9)
    assert affinity_to_unit(np.array([50_000.0]))[0] == pytest.approx(0.0)
    # A weaker binder scores at or below the threshold; a stronger one above.
    assert affinity_to_unit(np.array([WEAK_NM * 2]))[0] < WEAK_UNIT
    assert affinity_to_unit(np.array([WEAK_NM / 2]))[0] > WEAK_UNIT
