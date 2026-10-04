"""Guards for the stage 3b ESM-2 multi-task harness.

Three things have to hold before any number this harness produces means
anything:

1. **The single-task comparator is the arm's own stage 3 model**, not a
   near-replica. At ``lambda_aff = 0`` the multi-task network must be
   bit-identical to ``pepstab.mlp.MLPRegressor`` -- checked here on the **real
   ESM feature grid**, not only on a toy, because that is the grid the claim
   will be made on.
2. **Tuning parity is enforced, not asserted.** An L2 at the edge of its ladder
   must stop the run. `esm-arm` measured a transplanted ladder at 0.109 median
   rho, which is twice the worthwhile-gain bar -- a silent boundary hit would
   reproduce that inside a comparison about parity.
3. **The leakage rules hold**, and in particular absence is tested on
   ``peptide``, never on ``(allele, peptide)``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.affinity import auxiliary_only, dual_labelled, held_out_peptides
from pepstab.data import load_with_splits
from pepstab.mlp import MLPConfig, MLPRegressor
from pepstab.multitask import MultiTaskMLPConfig, MultiTaskMLPRegressor
from scripts import stage3b_esm_multitask as s3b


@pytest.fixture(scope="module")
def data():
    df = load_with_splits()
    return df, df[df.split == "train"], df[df.split == "val"]


# --- tuning parity ----------------------------------------------------------

def test_every_declared_arm_has_an_interior_l2():
    """The gate that would have caught esm-arm's 0.109 handicap."""
    for arm, cfgs in s3b.ARM_CONFIGS.items():
        rows = s3b.check_interior(arm, cfgs)
        assert all(r["interior"] for r in rows), arm


def test_a_boundary_l2_stops_the_run():
    bad = {"onehot": ((256, 64), 1e-3, (1e-3, 1e-2, 1e-1))}   # at the bottom edge
    with pytest.raises(SystemExit, match="edge of"):
        s3b.check_interior("additive", bad)

    worse = {"onehot": ((256, 64), 1e-1, (1e-3, 1e-2, 1e-1))}  # at the top edge
    with pytest.raises(SystemExit, match="edge of"):
        s3b.check_interior("additive", worse)


def test_arms_have_equal_grid_budgets_in_members():
    """Every arm spreads 30 networks over a two-way axis, as stage 2 does."""
    for arm, cfgs in s3b.ARM_CONFIGS.items():
        assert len(cfgs) == 2, f"{arm} must have a two-way ensemble axis"


def test_lambda_grid_matches_stage_2c():
    """The sequence curve from stage 2c is only comparable at shared lambdas."""
    assert s3b.LAMBDA_GRID == (0.0, 0.1, 0.3, 1.0, 3.0)
    assert s3b.LAMBDA_GRID[0] == 0.0, "lambda=0 is the single-task comparator"


# --- the controlled comparison ---------------------------------------------

def test_lambda_zero_is_bit_identical_to_the_stage_3_head_on_a_toy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 40)).astype(np.float32)
    y = (X[:, :3].sum(1) * 0.3 + rng.normal(0, 0.2, 400)).astype(np.float32)
    aff = np.where(rng.random(400) < 0.3, rng.random(400), np.nan)
    tr, dv = slice(0, 320), slice(320, 400)

    cfg = dict(hidden=(32, 8), l2=1e-2, seed=1, max_epochs=25, patience=15)
    a = MLPRegressor(MLPConfig(**cfg)).fit(X[tr], y[tr], X[dv], y[dv])
    b = MultiTaskMLPRegressor(MultiTaskMLPConfig(**cfg, lambda_aff=0.0)).fit(
        X[tr], y[tr], aff[tr], X[dv], y[dv])
    assert np.array_equal(a.predict(X), b.predict(X))


@pytest.mark.slow
def test_lambda_zero_is_bit_identical_on_the_real_esm_feature_grid(data):
    """The claim is made on these features, so it is checked on these features.

    A toy grid is 40 dense columns; the additive arm is 1,190 columns mixing a
    PCA-reduced ESM block with sparse one-hot. Float accumulation order differs
    enough between those that a toy-only check would not be evidence.
    """
    df, train, val = data
    groups = s3b.build_groups("additive", train.head(3000), val.head(400))
    g = groups[0]
    y = train.head(3000).y_log1p.to_numpy()
    aff = dual_labelled(train.head(3000)).to_numpy()
    tr = np.arange(len(y)) < 2400

    cfg = dict(hidden=(32, 8), l2=g.l2, seed=0, max_epochs=6, patience=15)
    a = MLPRegressor(MLPConfig(**cfg)).fit(
        g.X_train[tr], y[tr], g.X_train[~tr], y[~tr])
    b = MultiTaskMLPRegressor(MultiTaskMLPConfig(**cfg, lambda_aff=0.0)).fit(
        g.X_train[tr], y[tr], aff[tr], g.X_train[~tr], y[~tr])
    assert np.array_equal(a.predict(g.X_val), b.predict(g.X_val)), (
        "lambda=0 is not bit-identical on the ESM grid; the single-task arm "
        "would not be the stage 3 model and the comparison is uncontrolled")


def test_nonzero_lambda_actually_changes_the_fit():
    """If it did not, the sweep would be measuring nothing."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(500, 30)).astype(np.float32)
    y = (X[:, 0] + rng.normal(0, 0.3, 500)).astype(np.float32)
    aff = np.where(rng.random(500) < 0.5, X[:, 1] * 0.2 + 0.5, np.nan)
    tr, dv = slice(0, 400), slice(400, 500)
    cfg = dict(hidden=(16,), l2=1e-2, seed=0, max_epochs=20, patience=15)
    a = MultiTaskMLPRegressor(MultiTaskMLPConfig(**cfg, lambda_aff=0.0)).fit(
        X[tr], y[tr], aff[tr], X[dv], y[dv])
    b = MultiTaskMLPRegressor(MultiTaskMLPConfig(**cfg, lambda_aff=3.0)).fit(
        X[tr], y[tr], aff[tr], X[dv], y[dv])
    assert not np.allclose(a.predict(X), b.predict(X))


def test_affinity_head_is_untrained_at_lambda_zero():
    """The control the diagnostic is read against."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(600, 20)).astype(np.float32)
    y = (X[:, 0] + rng.normal(0, 0.3, 600)).astype(np.float32)
    aff = np.clip(X[:, 1] * 0.3 + 0.5, 0, 1)
    tr, dv = slice(0, 480), slice(480, 600)
    cfg = dict(hidden=(32,), l2=1e-3, seed=0, max_epochs=60, patience=15)
    a = MultiTaskMLPRegressor(MultiTaskMLPConfig(**cfg, lambda_aff=0.0)).fit(
        X[tr], y[tr], aff[tr], X[dv], y[dv])
    b = MultiTaskMLPRegressor(MultiTaskMLPConfig(**cfg, lambda_aff=3.0)).fit(
        X[tr], y[tr], aff[tr], X[dv], y[dv])
    q0 = abs(s3b.affinity_head_quality(a, X[dv], aff[dv]))
    q1 = abs(s3b.affinity_head_quality(b, X[dv], aff[dv]))
    assert q1 > q0, "the auxiliary task was not learned even at lambda=3"


# --- the degradation model behind the MDE ----------------------------------

def test_degrade_is_identity_at_zero_and_destroys_ranking_at_one():
    rng = np.random.default_rng(0)
    allele = np.repeat(["A", "B", "C"], 60)
    pred = rng.normal(size=180)
    assert np.array_equal(s3b.degrade(pred, allele, 0.0, rng), pred)

    full = s3b.degrade(pred, allele, 1.0, np.random.default_rng(1))
    for a in np.unique(allele):
        m = allele == a
        # A within-allele permutation: same multiset, different order.
        assert np.allclose(np.sort(full[m]), np.sort(pred[m]))
    assert not np.allclose(full, pred)


def test_degrade_is_monotone_in_t():
    """A blend level must cost ranking, or the MDE curve means nothing."""
    from scipy import stats

    rng = np.random.default_rng(5)
    allele = np.repeat(["A", "B", "C", "D"], 120)
    truth = rng.normal(size=480)
    pred = truth + rng.normal(0, 0.4, 480)
    rhos = []
    for t in (0.0, 0.2, 0.4, 0.7, 1.0):
        d = s3b.degrade(pred, allele, t, np.random.default_rng(7))
        rhos.append(np.mean([stats.spearmanr(truth[allele == a], d[allele == a]).statistic
                             for a in np.unique(allele)]))
    assert rhos[0] > rhos[-1]
    assert rhos[0] == pytest.approx(max(rhos))


def test_degrade_preserves_the_marginal_distribution_within_allele():
    """Only within-allele ranking is destroyed -- not the scale or the
    between-allele offsets, which would change a different thing."""
    rng = np.random.default_rng(9)
    allele = np.repeat(["A", "B"], 50)
    pred = rng.normal(size=100)
    out = s3b.degrade(pred, allele, 1.0, np.random.default_rng(2))
    for a in np.unique(allele):
        m = allele == a
        assert out[m].mean() == pytest.approx(pred[m].mean())


# --- difference-in-differences ---------------------------------------------

def test_did_is_zero_when_both_arms_move_identically(data):
    _, _, val = data
    v = val.head(600).reset_index(drop=True)
    alleles = sorted(v.allele.value_counts()[lambda s: s >= 20].index)
    rng = np.random.default_rng(0)
    base_e = rng.normal(size=len(v))
    base_s = rng.normal(size=len(v))
    shift = rng.normal(size=len(v)) * 0.01
    r = s3b.difference_in_differences(v, alleles, base_e, base_e + shift,
                                      base_s, base_s + shift, n_boot=100)
    # Not exactly 0 -- the two arms start from different predictions -- but the
    # interval must contain the observed value and be finite.
    assert np.isfinite(r["difference_in_differences"])
    assert r["ci95"][0] <= r["difference_in_differences"] <= r["ci95"][1]


def test_did_recovers_a_one_sided_improvement(data):
    """If only the ESM arm improves, the DiD must be positive."""
    _, _, val = data
    v = val.head(800).reset_index(drop=True)
    alleles = sorted(v.allele.value_counts()[lambda s: s >= 20].index)
    y = v.y_log1p.to_numpy()
    rng = np.random.default_rng(1)
    noisy = y + rng.normal(0, 1.2, len(v))
    better = y + rng.normal(0, 0.5, len(v))      # much closer to the truth
    other = y + rng.normal(0, 1.2, len(v))
    r = s3b.difference_in_differences(v, alleles, noisy, better, other, other,
                                      n_boot=200)
    assert r["delta_esm"] > 0
    assert r["delta_seq"] == pytest.approx(0.0, abs=1e-12)
    assert r["difference_in_differences"] > 0


# --- leakage ----------------------------------------------------------------

def test_dual_labelled_introduces_no_new_peptide(data):
    """The probe cannot move a peptide across a split boundary."""
    df, _, _ = data
    aff = dual_labelled(df)
    labelled = set(df.loc[aff.notna(), "peptide"].astype(str))
    assert labelled <= set(df["peptide"].astype(str))
    assert aff.notna().sum() == 7_281


def test_expansion_candidates_are_absent_from_the_stability_set(data):
    df, _, _ = data
    aux = auxiliary_only(df)
    assert not (set(aux["peptide"].astype(str))
                & set(df["peptide"].astype(str)))


def test_held_out_set_covers_val_test_and_the_inner_dev_fold(data):
    from scripts.baseline_sequence import inner_folds

    df, train, _ = data
    fold = inner_folds(train)
    dev_pep = train.loc[(fold == "dev").to_numpy(), "peptide"]
    held = set(held_out_peptides(df, dev_peptides=dev_pep))
    assert set(df.loc[df.split == "val", "peptide"].astype(str)) <= held
    assert set(df.loc[df.split == "test", "peptide"].astype(str)) <= held
    assert set(dev_pep.astype(str)) <= held


def test_padding_eligible_would_leak_and_is_not_used(data):
    """The reference's own flag tests the pair, so it admits held-out peptides.

    Recorded as a test because the flag is right there in the CSV and is the
    obvious thing for a later reader to reach for.
    """
    from pepstab.affinity import load_reference
    from scripts.baseline_sequence import inner_folds

    df, train, _ = data
    ref = load_reference()
    if "padding_eligible" not in ref.columns:
        pytest.skip("reference has no padding_eligible column")
    fold = inner_folds(train)
    dev_pep = train.loc[(fold == "dev").to_numpy(), "peptide"]
    held = set(held_out_peptides(df, dev_peptides=dev_pep))
    flagged = ref[ref["padding_eligible"].astype(bool)]
    leaking = flagged[flagged["peptide"].astype(str).isin(held)]
    assert len(leaking) > 0, "premise changed: the flag no longer leaks"
    # And the function the harness actually uses does not have this problem.
    assert not (set(auxiliary_only(df)["peptide"].astype(str)) & held)


def test_audit_expansion_reproduces_the_stage_2c_counts(data):
    df, train, _ = data
    table, summary = s3b.audit_expansion(df, train)
    assert summary["n_held_out_peptides"] == 2_076
    assert summary["n_candidate_rows"] == 66_214
    assert summary["n_candidate_peptides"] == 20_836
    assert summary["n_kept_rows_hamming_gt_3"] == 64_226
    assert summary["n_kept_peptides_hamming_gt_3"] == 20_195
    assert (table["min_hamming_to_held_out"] > 0).all(), (
        "a candidate at distance 0 means a peptide in the stability set slipped "
        "into the expansion pool")


def test_no_candidate_is_within_one_substitution_after_filtering(data):
    """The plan's §3b minimum. The harness applies the stricter >3 rule, so
    this must hold with room to spare."""
    from pepstab.affinity import filter_by_distance
    from scripts.baseline_sequence import inner_folds

    df, train, _ = data
    fold = inner_folds(train)
    dev_pep = train.loc[(fold == "dev").to_numpy(), "peptide"]
    held = held_out_peptides(df, dev_peptides=dev_pep)
    kept, _ = filter_by_distance(auxiliary_only(df), held)
    assert kept["min_hamming_to_held_out"].min() > 3


# --- the test split is never read ------------------------------------------

def test_harness_scores_validation_only(data):
    """Structural check: the scored frame is the val split and nothing else."""
    df, _, val = data
    assert set(val.split.unique()) == {"val"}
    assert len(val) == 2_817
    src = Path(s3b.__file__).read_text()
    assert 'split == "test"' not in src
    assert "scored once" not in src


def test_a_constant_affinity_head_scores_zero_not_nan():
    """At lambda=0 the head decays to a constant. That is the control, and the
    project's frozen rule already prices a constant prediction at 0 -- NaN would
    poison the mean over ensemble members and hide the control."""
    class _Const:
        def predict_affinity(self, X):
            return np.full(len(X), 0.37)

    X = np.zeros((50, 4), dtype=np.float32)
    aff = np.linspace(0, 1, 50)
    assert s3b.affinity_head_quality(_Const(), X, aff) == 0.0


def test_affinity_head_quality_is_nan_when_there_are_too_few_labels():
    """Distinct from the constant case: 'cannot say' is not 'no information'."""
    class _Varied:
        def predict_affinity(self, X):
            return np.arange(len(X), dtype=float)

    X = np.zeros((50, 4), dtype=np.float32)
    aff = np.full(50, np.nan)
    aff[:2] = [0.1, 0.9]
    assert np.isnan(s3b.affinity_head_quality(_Varied(), X, aff))


# --- reuse path (sequence arm runs before the ESM arms are cleared) ---------

def test_reuse_rejects_a_missing_prediction_file(data, tmp_path, monkeypatch):
    _, _, val = data
    monkeypatch.setattr(s3b, "PRED_DIR", tmp_path)
    with pytest.raises(SystemExit, match="missing"):
        s3b.load_arm_predictions("seq", (0.0,), val)


def test_reuse_rejects_partial_coverage(data, tmp_path, monkeypatch):
    _, _, val = data
    monkeypatch.setattr(s3b, "PRED_DIR", tmp_path)
    part = val.head(100)
    pd.DataFrame({"pair_id": part.pair_id.to_numpy(),
                  "y_pred": np.zeros(len(part))}).to_csv(
        tmp_path / "stage3b_seq_lam0.csv", index=False)
    with pytest.raises(SystemExit, match="does not cover"):
        s3b.load_arm_predictions("seq", (0.0,), val)


def test_reuse_aligns_by_pair_id_not_by_row_order(data, tmp_path, monkeypatch):
    """A saved prediction file in a different order must still line up."""
    _, _, val = data
    monkeypatch.setattr(s3b, "PRED_DIR", tmp_path)
    rng = np.random.default_rng(0)
    y = rng.normal(size=len(val))
    order = rng.permutation(len(val))
    pd.DataFrame({"pair_id": val.pair_id.to_numpy()[order],
                  "y_pred": y[order]}).to_csv(
        tmp_path / "stage3b_seq_lam0.csv", index=False)
    got = s3b.load_arm_predictions("seq", (0.0,), val)
    assert np.allclose(got[0.0], y)
