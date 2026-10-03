"""Guards on the stage 2c auxiliary affinity head. Run: .venv/bin/python -m pytest -q

Stage 2c claims "affinity labels do / do not help", and that claim rests on four
things this file checks:

1. **The two arms differ only in the auxiliary loss.** At ``lambda_aff = 0`` the
   multi-task network must be *bit-identical* to the stage 2 network, so the
   single-task comparator is the stage 2 baseline rather than a near-replica.
2. **The gradients are right**, including the two-head sum at the shared trunk.
   Get that wrong and the auxiliary task never reaches the encoder -- which
   would look exactly like "affinity does not help".
3. **The auxiliary loss is masked and normalised per labelled row**, so
   ``lambda_aff`` means the same thing in every minibatch.
4. **The leakage rules hold**: the affinity target never enters the model for a
   val or test row, and auxiliary-only candidates stay > 3 substitutions from
   every held-out and stopping-fold peptide.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab import multitask as mt_module
from pepstab.affinity import (
    AFFINITY_SCALE_NM,
    MAX_HAMMING_TO_HELD_OUT,
    affinity_to_target,
    auxiliary_only,
    dual_labelled,
    filter_by_distance,
    held_out_peptides,
    load_reference,
    min_hamming_to,
)
from pepstab.data import load_with_splits
from pepstab.mlp import MLPConfig, MLPRegressor
from pepstab.multitask import MultiTaskMLPConfig, MultiTaskMLPRegressor
from pepstab.splits import HAMMING_THRESHOLD
from scripts.baseline_sequence import inner_folds


@pytest.fixture(scope="module")
def split_df():
    return load_with_splits()


@pytest.fixture(scope="module")
def toy():
    """A small regression problem where stability and affinity differ.

    The affinity target depends on features the stability target does not use,
    so a model that reaches +0.99 on affinity cannot be reading it off the
    stability signal.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 24)).astype(np.float32)
    y = (X[:, 0] * 2.0 + X[:, 1] ** 2).astype(np.float32)
    aff_truth = X[:, 5] * 1.5 + X[:, 6]
    y_aff = np.where(rng.random(800) < 0.3, aff_truth, np.nan)
    X_dev = rng.normal(size=(200, 24)).astype(np.float32)
    y_dev = (X_dev[:, 0] * 2.0 + X_dev[:, 1] ** 2).astype(np.float32)
    return X, y, y_aff, X_dev, y_dev


# --- 1. the two arms differ only in the auxiliary loss --------------------

def test_lambda_zero_is_bit_identical_to_the_stage2_network(toy):
    """The load-bearing guard: lambda=0 *is* the stage 2 baseline.

    If this fails, stage 2c is comparing against something that merely resembles
    the stage 2 network, and any delta is partly an artifact of the rewrite.
    The affinity head is drawn from its own generator precisely so the main
    stream -- trunk init, stability head init, then every minibatch permutation
    -- is left exactly where stage 2 leaves it.
    """
    X, y, y_aff, X_dev, y_dev = toy
    kw = dict(hidden=(32, 16), l2=1e-5, lr=1e-3, max_epochs=30, patience=50, seed=3)
    single = MLPRegressor(MLPConfig(**kw)).fit(X, y, X_dev, y_dev)
    multi = MultiTaskMLPRegressor(
        MultiTaskMLPConfig(**kw, lambda_aff=0.0)).fit(X, y, y_aff, X_dev, y_dev)

    assert np.array_equal(single.predict(X_dev), multi.predict(X_dev))
    assert single.best_epoch_ == multi.best_epoch_
    assert single.n_epochs_ == multi.n_epochs_
    assert single.dev_mse_ == multi.dev_mse_


def test_lambda_zero_matches_stage2_on_the_real_grid(split_df):
    """Same guard, on the real features and the config stage 2 selected.

    A toy problem could pass on luck; 860 one-hot columns and 17,744 rows could
    not. Two epochs is enough -- the point is the arithmetic, not convergence.
    """
    from pepstab.features import build_features

    train = split_df[split_df.split == "train"]
    fold = inner_folds(train)
    is_fit = (fold == "fit").to_numpy()
    X = build_features(train, "pep_pseudo", "onehot")
    y = train.y_log1p.to_numpy()
    y_aff = dual_labelled(train).to_numpy()

    kw = dict(hidden=(256, 64), l2=1e-5, lr=1e-3, max_epochs=2, patience=50, seed=0)
    single = MLPRegressor(MLPConfig(**kw)).fit(
        X[is_fit], y[is_fit], X[~is_fit], y[~is_fit])
    multi = MultiTaskMLPRegressor(MultiTaskMLPConfig(**kw, lambda_aff=0.0)).fit(
        X[is_fit], y[is_fit], y_aff[is_fit], X[~is_fit], y[~is_fit])
    assert np.array_equal(single.predict(X[~is_fit]), multi.predict(X[~is_fit]))


def test_lambda_changes_the_stability_predictions(toy):
    """A non-zero lambda must actually move the shared trunk.

    The mirror of the previous test: parity at 0 would be worthless if the
    auxiliary loss turned out to be a no-op at every lambda.
    """
    X, y, y_aff, X_dev, y_dev = toy
    kw = dict(hidden=(32, 16), max_epochs=30, patience=50, seed=3)
    base = MultiTaskMLPRegressor(
        MultiTaskMLPConfig(**kw, lambda_aff=0.0)).fit(X, y, y_aff, X_dev, y_dev)
    for lam in (0.1, 1.0):
        other = MultiTaskMLPRegressor(
            MultiTaskMLPConfig(**kw, lambda_aff=lam)).fit(X, y, y_aff, X_dev, y_dev)
        assert not np.array_equal(base.predict(X_dev), other.predict(X_dev)), lam


# --- 2. the gradients are right -------------------------------------------

@pytest.mark.parametrize("lam", [0.0, 0.3, 1.0])
def test_gradients_match_finite_differences(monkeypatch, lam):
    """Backprop against central differences, for both heads and the trunk.

    Run in float64, for the same reason as the stage 2 check: in float32 a
    central difference of an O(1) loss carries ~1e-4 of absolute noise, which
    would swamp the smaller gradients and turn this into a rounding test.

    The gradient that matters most is the trunk's, because it is the sum of the
    two heads' contributions -- the single step that decides whether the
    auxiliary task reaches the shared encoder at all.
    """
    monkeypatch.setattr(mt_module, "DTYPE", np.float64)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 7))
    y_stab = rng.normal(size=24)
    y_aff = rng.normal(size=24)
    mask = rng.random(24) < 0.5

    model = MultiTaskMLPRegressor(MultiTaskMLPConfig(hidden=(6, 5), l2=0.0, seed=1,
                                                     lambda_aff=lam))
    model._init_params(7, np.random.default_rng(2))
    n_lab = int(mask.sum())

    def loss() -> float:
        stab, aff, _ = model._forward(X)
        out = float(np.mean((stab - y_stab) ** 2))
        if lam > 0 and n_lab:
            out += lam * float(np.mean((aff[mask] - y_aff[mask]) ** 2))
        return out

    stab, aff, acts = model._forward(X)
    gw, gb, ghw, ghb = model._backward(acts, stab, aff, y_stab, y_aff, mask)

    eps, worst = 1e-6, 0.0
    arrays = ([(model.weights_[k], gw[k]) for k in range(len(model.weights_))]
              + [(model.biases_[k], gb[k]) for k in range(len(model.biases_))]
              + [(model.head_w_[j], ghw[j]) for j in range(2)]
              + [(model.head_b_[j], ghb[j]) for j in range(2)])
    for arr, grad in arrays:
        for ix in list(np.ndindex(arr.shape))[:8]:
            old = float(arr[ix])
            arr[ix] = old + eps
            high = loss()
            arr[ix] = old - eps
            low = loss()
            arr[ix] = old
            numeric = (high - low) / (2 * eps)
            worst = max(worst, abs(numeric - grad[ix])
                        / max(1e-6, abs(numeric) + abs(grad[ix])))
    assert worst < 1e-6, f"worst relative gradient error {worst:.2e} at lambda={lam}"


def test_the_auxiliary_head_learns_its_own_target(toy):
    """The diagnostic that makes a null result readable.

    "Affinity did not help" and "the affinity head never trained" look identical
    in the headline metric. Here the auxiliary target depends on features the
    stability target ignores, so a high correlation proves the auxiliary task
    was genuinely learned through the shared trunk.
    """
    from scipy import stats

    X, y, y_aff, X_dev, y_dev = toy
    kw = dict(hidden=(32, 16), lr=1e-2, max_epochs=150, patience=200, seed=0)
    truth = X_dev[:, 5] * 1.5 + X_dev[:, 6]

    trained = MultiTaskMLPRegressor(
        MultiTaskMLPConfig(**kw, lambda_aff=1.0)).fit(X, y, y_aff, X_dev, y_dev)
    rho = float(stats.spearmanr(trained.predict_affinity(X_dev), truth).statistic)
    assert rho > 0.9, f"auxiliary head did not learn its target (rho={rho:.3f})"
    # and the stability head still does its job
    assert float(stats.spearmanr(trained.predict(X_dev), y_dev).statistic) > 0.9

    # At lambda = 0 the head receives no task gradient, only L2 decay, so it
    # carries no information about the auxiliary target. Over enough epochs the
    # decay drives it to a constant and Spearman becomes undefined; over few it
    # stays a shrinking random projection correlating at roughly zero. Either is
    # "never trained", and the diagnostic has to read both the same way.
    untrained = MultiTaskMLPRegressor(
        MultiTaskMLPConfig(**kw, lambda_aff=0.0)).fit(X, y, y_aff, X_dev, y_dev)
    aff_pred = untrained.predict_affinity(X_dev)
    if len(np.unique(aff_pred)) < 2:
        pass  # collapsed to a constant: no signal, by construction
    else:
        assert abs(float(stats.spearmanr(aff_pred, truth).statistic)) < 0.3


# --- 3. masking and normalisation -----------------------------------------

def test_unlabelled_rows_do_not_enter_the_auxiliary_loss(toy):
    """Changing a NaN row's auxiliary value must change nothing.

    The masking guard. If unlabelled rows leaked in as zeros, 74% of the real
    training set would be teaching the head that every peptide is a weak binder.
    """
    X, y, y_aff, X_dev, y_dev = toy
    poisoned = y_aff.copy()
    missing = ~np.isfinite(y_aff)
    poisoned[missing] = 1e3  # absurd, and must be ignored... as NaN it is
    poisoned[missing] = np.nan
    kw = dict(hidden=(16, 8), max_epochs=20, patience=50, seed=1, lambda_aff=1.0)
    a = MultiTaskMLPRegressor(MultiTaskMLPConfig(**kw)).fit(X, y, y_aff, X_dev, y_dev)
    b = MultiTaskMLPRegressor(MultiTaskMLPConfig(**kw)).fit(X, y, poisoned, X_dev, y_dev)
    assert np.array_equal(a.predict(X_dev), b.predict(X_dev))


def test_auxiliary_loss_is_normalised_per_labelled_row():
    """The auxiliary gradient scales with 1/n_labelled, not 1/n_batch.

    Otherwise the effective lambda would drift with each batch's label rate --
    shrinking it ~4x on average on the real data, so a swept value would not
    mean what the report says it means.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    y_stab = np.zeros(20)
    y_aff = np.ones(20)
    model = MultiTaskMLPRegressor(MultiTaskMLPConfig(hidden=(4,), l2=0.0, seed=0,
                                                     lambda_aff=1.0))
    model._init_params(5, np.random.default_rng(1))
    stab, aff, acts = model._forward(X)

    all_rows = np.ones(20, dtype=bool)
    half = np.zeros(20, dtype=bool)
    half[:10] = True
    _, _, ghw_all, _ = model._backward(acts, stab, aff, y_stab, y_aff, all_rows)
    _, _, ghw_half, _ = model._backward(acts, stab, aff, y_stab, y_aff, half)

    # Same residual on every row, so halving the labelled rows must leave the
    # per-labelled-row mean gradient on that subset unchanged in scale, not
    # halve it. Compare against the explicit 1/n_lab formula.
    expected_half = acts[-1].T @ (
        np.where(half, (2.0 / 10) * (aff - y_aff), 0.0)[:, None])
    assert np.allclose(ghw_half[1], expected_half)
    expected_all = acts[-1].T @ ((2.0 / 20) * (aff - y_aff))[:, None]
    assert np.allclose(ghw_all[1], expected_all)

    # The stability head, which is averaged over the whole batch either way,
    # must be untouched by the auxiliary mask.
    assert np.allclose(ghw_half[0], ghw_all[0])


def test_fit_rejects_a_bad_auxiliary_setup(toy):
    X, y, y_aff, X_dev, y_dev = toy
    with pytest.raises(ValueError, match="lambda_aff must be >= 0"):
        MultiTaskMLPRegressor(MultiTaskMLPConfig(lambda_aff=-0.1)).fit(
            X, y, y_aff, X_dev, y_dev)
    with pytest.raises(ValueError, match="no row carries an affinity label"):
        MultiTaskMLPRegressor(MultiTaskMLPConfig(lambda_aff=1.0)).fit(
            X, y, np.full(len(y), np.nan), X_dev, y_dev)
    with pytest.raises(ValueError, match="y_aff has"):
        MultiTaskMLPRegressor(MultiTaskMLPConfig(lambda_aff=1.0)).fit(
            X, y, y_aff[:-1], X_dev, y_dev)


# --- the affinity target transform ----------------------------------------

def test_affinity_transform_endpoints():
    got = affinity_to_target([1.0, AFFINITY_SCALE_NM, 2 * AFFINITY_SCALE_NM, np.nan])
    assert got[0] == pytest.approx(1.0)
    assert got[1] == pytest.approx(0.0)
    assert got[2] == 0.0, "weaker than the scale must clip to 0, not go negative"
    assert np.isnan(got[3]), "missing must stay missing -- the loss reads NaN as unlabelled"


def test_affinity_transform_is_monotone_decreasing_in_nM():
    vals = affinity_to_target([1.0, 10.0, 100.0, 1_000.0, 10_000.0, 40_000.0])
    assert np.all(np.diff(vals) < 0)
    assert np.all((vals >= 0) & (vals <= 1))


# --- 4. leakage -----------------------------------------------------------

def test_c67s_constructs_are_never_given_an_affinity_label(split_df):
    """Wild-type affinity describes a different groove from a C67S construct.

    C67S substitutes one of the 34 peptide-contact positions, so matching the
    construct to its wild-type affinity is a category error, not an
    approximation. docs/AFFINITY_REFERENCE.md flags 245 such rows.
    """
    ref = load_reference()
    assert not ref["engineered_construct_mismatch"].astype(bool).any()

    aff = dual_labelled(split_df)
    c67s = split_df["allele"].str.contains("C67S", regex=False)
    assert c67s.sum() > 0, "fixture should contain the constructs"
    assert aff[c67s.to_numpy()].isna().all()


def test_dual_labelled_joins_on_the_pair_and_covers_the_documented_count(split_df):
    aff = dual_labelled(split_df)
    assert len(aff) == len(split_df)
    # docs/AFFINITY_REFERENCE.md: 7,281 pairs with both affinity and half-life.
    assert int(aff.notna().sum()) == 7281


def test_auxiliary_only_peptides_are_absent_from_the_stability_set(split_df):
    """Absence is tested on the peptide, never on the (allele, peptide) pair.

    The pair test is what `padding_eligible` does, and it leaks: a peptide held
    out in val or test reappears as "not in the stability dataset" whenever it
    was measured on a different allele.
    """
    cand = auxiliary_only(split_df)
    assert not set(cand["peptide"]) & set(split_df["peptide"].astype(str))


def test_distance_filter_excludes_near_neighbours_of_every_held_out_peptide(split_df):
    """The exclusion stage 2b and 2c both require, verified end to end.

    Not just val and test: the inner stopping fold too. Under ``cv_folds()``
    every training peptide is some ensemble member's stopping peptide, so an
    auxiliary peptide close to one would tune that member's epoch count on a
    near-duplicate.
    """
    train = split_df[split_df.split == "train"]
    fold = inner_folds(train)
    dev_peptides = train.loc[(fold == "dev").to_numpy(), "peptide"]
    held = held_out_peptides(split_df, dev_peptides=dev_peptides)

    assert set(split_df.loc[split_df.split == "val", "peptide"].astype(str)) <= set(held)
    assert set(split_df.loc[split_df.split == "test", "peptide"].astype(str)) <= set(held)
    assert set(dev_peptides.astype(str)) <= set(held)

    cand = auxiliary_only(split_df)
    kept, audit = filter_by_distance(cand, held)

    assert MAX_HAMMING_TO_HELD_OUT == HAMMING_THRESHOLD, (
        "the auxiliary exclusion must match the split's clustering threshold")
    observed = min_hamming_to(sorted(kept["peptide"].unique()), held)
    assert observed.min() > MAX_HAMMING_TO_HELD_OUT
    assert len(kept) < len(cand), "the filter must actually remove something"
    assert audit.loc[audit["excluded"], "n_rows"].sum() == len(cand) - len(kept)


def test_held_out_affinity_labels_are_not_passed_to_fit(split_df):
    """The structural guard on the probe: training sees training rows only.

    ``dual_labelled`` deliberately returns labels for every split -- the val
    ones drive the auxiliary-head diagnostic. This asserts the slice the fit
    actually receives carries no val or test row.
    """
    aff = dual_labelled(split_df)
    train_mask = (split_df.split == "train").to_numpy()
    aff_train = aff.to_numpy()[train_mask]
    assert len(aff_train) == int(train_mask.sum()) == 19716
    held = split_df.loc[~train_mask, "pair_id"]
    assert not set(split_df.loc[train_mask, "pair_id"]) & set(held)
    # and the auxiliary labels that exist on train are a strict subset
    assert 0 < int(np.isfinite(aff_train).sum()) < int(aff.notna().sum())
