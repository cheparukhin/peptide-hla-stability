"""Guards on the stage 2 baselines. Run with: .venv/bin/python -m pytest -q

Three things have to hold or the stage 2 numbers mean nothing: the encodings say
what they claim to say, the MLP's gradients are right, and the inner fit/dev cut
keeps peptide clusters whole -- otherwise the epoch count is tuned on leaked
near-duplicates and every validation score is optimistic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.data import AMINO_ACIDS, PEPTIDE_LENGTH, encode_sequences, load_with_splits
from pepstab.features import (
    BLOSUM62,
    BLOSUM_SCALE,
    ENCODINGS,
    INPUT_SETS,
    SEQ_LENGTHS,
    build_features,
    encode_matrix,
    feature_names,
    n_features,
    residue_rows,
)
from pepstab import mlp as mlp_module
from pepstab.mlp import MLPConfig, MLPRegressor
from pepstab.splits import HAMMING_THRESHOLD
from scripts.baseline_sequence import DEV_FRACTION, inner_folds


@pytest.fixture(scope="module")
def split_df():
    return load_with_splits()


# --- BLOSUM62 -------------------------------------------------------------
# Checked against ftp.ncbi.nlm.nih.gov/blast/matrices/BLOSUM62, reordered into
# AMINO_ACIDS order. A transcription slip here would silently distort every
# BLOSUM arm, so the asserts name values that are easy to look up.

def test_blosum62_shape_and_symmetry():
    m = np.asarray(BLOSUM62)
    assert m.shape == (20, 20)
    np.testing.assert_array_equal(m, m.T)


def test_blosum62_reference_values():
    idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    m = np.asarray(BLOSUM62)
    assert m[idx["W"], idx["W"]] == 11  # largest diagonal: tryptophan is rarest
    assert m[idx["C"], idx["C"]] == 9
    assert m[idx["A"], idx["A"]] == 4   # smallest diagonal
    assert m[idx["I"], idx["L"]] == 2   # conservative substitution
    assert m[idx["A"], idx["W"]] == -3  # disfavoured
    assert m[idx["D"], idx["E"]] == 2
    # Every diagonal entry beats every off-diagonal entry in its own row: an
    # amino acid always scores highest against itself.
    assert all(m[i, i] > max(m[i, j] for j in range(20) if j != i) for i in range(20))


# --- encodings ------------------------------------------------------------

def test_onehot_is_one_per_position():
    X = encode_matrix(["ACDEFGHIK", "WWWWWWWWW"], PEPTIDE_LENGTH, "onehot")
    assert X.shape == (2, PEPTIDE_LENGTH * 20)
    blocks = X.reshape(2, PEPTIDE_LENGTH, 20)
    np.testing.assert_array_equal(blocks.sum(axis=2), np.ones((2, PEPTIDE_LENGTH)))
    # Position is preserved: A at position 0 sets column 0 of block 0 only.
    assert blocks[0, 0, AMINO_ACIDS.index("A")] == 1
    assert blocks[0, 1, AMINO_ACIDS.index("C")] == 1


def test_blosum_rows_are_the_matrix_over_scale():
    X = encode_matrix(["WACDEFGHI"], PEPTIDE_LENGTH, "blosum")
    block = X.reshape(PEPTIDE_LENGTH, 20)
    expected = np.asarray(BLOSUM62[AMINO_ACIDS.index("W")], dtype=np.float32) / BLOSUM_SCALE
    np.testing.assert_allclose(block[0], expected)


def test_encodings_distinguish_every_amino_acid():
    # Two residues encoding identically would make them interchangeable to the
    # model. True for one-hot by construction; worth checking for BLOSUM.
    for encoding in ENCODINGS:
        rows = residue_rows(encoding)
        assert len(np.unique(rows, axis=0)) == 20, encoding


def test_unknown_encoding_rejected():
    with pytest.raises(ValueError, match="unknown encoding"):
        encode_matrix(["ACDEFGHIK"], PEPTIDE_LENGTH, "blosum90")


def test_encoding_rejects_bad_sequences():
    with pytest.raises(ValueError, match="unknown residue"):
        encode_matrix(["ACDEFGHIX"], PEPTIDE_LENGTH, "onehot")
    with pytest.raises(ValueError, match="length"):
        encode_matrix(["ACDEF"], PEPTIDE_LENGTH, "onehot")


# --- feature assembly -----------------------------------------------------

@pytest.mark.parametrize("input_set", sorted(INPUT_SETS))
@pytest.mark.parametrize("encoding", ENCODINGS)
def test_feature_width_is_declared_width(split_df, input_set, encoding):
    X = build_features(split_df.head(50), input_set, encoding)
    assert X.shape == (50, n_features(input_set))
    assert len(feature_names(input_set, encoding)) == n_features(input_set)
    assert np.isfinite(X).all()


def test_caching_matches_row_by_row_encoding(split_df):
    """The per-unique-sequence cache must not reorder or mis-map rows."""
    sub = split_df.head(200)
    fast = build_features(sub, "pep_pseudo", "blosum")
    slow = np.concatenate([
        encode_matrix(sub.peptide.to_numpy(dtype=str), SEQ_LENGTHS["peptide"], "blosum"),
        encode_matrix(sub.hla_pseudoseq.to_numpy(dtype=str),
                      SEQ_LENGTHS["hla_pseudoseq"], "blosum"),
    ], axis=1)
    np.testing.assert_allclose(fast, slow)


def test_features_are_row_local(split_df):
    """A subset's features must equal those rows of the whole frame's features.

    This is what lets a model fit on train rows and predict val rows: the
    columns cannot depend on which rows were passed in.
    """
    sub = split_df.head(300)
    whole = build_features(sub, "pep_pseudo", "onehot")
    picked = [7, 11, 299, 0]
    part = build_features(sub.iloc[picked], "pep_pseudo", "onehot")
    np.testing.assert_array_equal(part, whole[picked])


def test_unknown_input_set_rejected(split_df):
    with pytest.raises(ValueError, match="unknown input set"):
        build_features(split_df.head(5), "pep_plus_structure", "onehot")


# --- MLP ------------------------------------------------------------------

def test_gradients_match_finite_differences(monkeypatch):
    """Backprop against central differences.

    Run in float64. The question here is whether the backward formulas are
    right, and in the float32 the model actually trains in, a central difference
    of an O(1) loss carries ~1e-4 of absolute noise -- which swamps the smaller
    gradients and would make this check measure rounding, not correctness.
    """
    monkeypatch.setattr(mlp_module, "DTYPE", np.float64)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 7))
    y = rng.normal(size=24)
    model = MLPRegressor(MLPConfig(hidden=(6, 5), l2=0.0, seed=1))
    model._init_params(7, np.random.default_rng(2))
    model.y_offset_ = 0.0

    def loss() -> float:
        pred, _ = model._forward(X)
        return float(np.mean((pred - y) ** 2))

    pred, acts = model._forward(X)
    gw, gb = model._backward(acts, pred - y)

    eps, worst = 1e-6, 0.0
    for k in range(len(model.weights_)):
        for arr, grad in ((model.weights_[k], gw[k]), (model.biases_[k], gb[k])):
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
    assert worst < 1e-6, f"worst relative gradient error {worst:.2e}"


def test_mlp_fits_a_nonlinear_function():
    """A linear model cannot do this; the MLP should get most of the variance."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1500, 5)).astype(np.float32)
    y = X[:, 0] * X[:, 1] + 2.0 * np.maximum(X[:, 2], 0.0)
    model = MLPRegressor(MLPConfig(hidden=(64, 32), max_epochs=200, patience=20))
    model.fit(X[:1200], y[:1200], X[1200:], y[1200:])
    assert model.dev_mse_ < 0.2 * float(y[1200:].var())
    assert 1 <= model.best_epoch_ <= model.n_epochs_
    assert len(model.dev_curve_) == model.n_epochs_
    assert model.fit_seconds_ > 0


def test_mlp_keeps_the_best_dev_weights():
    """``dev_mse_`` must be the minimum of the curve, not the last epoch's."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 8)).astype(np.float32)
    y = (X[:, 0] ** 2 - X[:, 3]).astype(np.float32)
    model = MLPRegressor(MLPConfig(hidden=(32,), max_epochs=60, patience=10, seed=3))
    model.fit(X[:320], y[:320], X[320:], y[320:])
    assert model.dev_mse_ == pytest.approx(min(model.dev_curve_))
    assert model.dev_curve_[model.best_epoch_ - 1] == pytest.approx(model.dev_mse_)
    refit = float(np.mean((model.predict(X[320:]) - y[320:]) ** 2))
    assert refit == pytest.approx(model.dev_mse_, rel=1e-5)


def test_mlp_is_deterministic_per_seed():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 6)).astype(np.float32)
    y = X.sum(axis=1).astype(np.float32)
    args = (X[:240], y[:240], X[240:], y[240:])

    def preds(seed: int) -> np.ndarray:
        return MLPRegressor(MLPConfig(hidden=(16,), max_epochs=15, seed=seed)
                            ).fit(*args).predict(X[240:])

    np.testing.assert_array_equal(preds(0), preds(0))
    assert not np.array_equal(preds(0), preds(1))


def test_mlp_input_validation():
    X = np.zeros((10, 3), dtype=np.float32)
    y = np.zeros(10, dtype=np.float32)
    with pytest.raises(RuntimeError, match="before fit"):
        MLPRegressor().predict(X)
    with pytest.raises(ValueError, match="dev fold is empty"):
        MLPRegressor(MLPConfig(max_epochs=1)).fit(X, y, X[:0], y[:0])
    with pytest.raises(ValueError, match="rows"):
        MLPRegressor(MLPConfig(max_epochs=1)).fit(X, y[:5], X, y)


# --- the inner fit/dev cut ------------------------------------------------

def test_inner_fold_keeps_clusters_whole(split_df):
    train = split_df[split_df.split == "train"]
    fold = inner_folds(train)
    assert set(fold.unique()) == {"fit", "dev"}
    per_cluster = pd.DataFrame({"cluster_id": train.cluster_id.to_numpy(),
                                "fold": fold.to_numpy()})
    assert per_cluster.groupby("cluster_id").fold.nunique().max() == 1


def test_inner_fold_hits_its_target_fraction(split_df):
    train = split_df[split_df.split == "train"]
    fold = inner_folds(train)
    share = float((fold == "dev").mean())
    assert abs(share - DEV_FRACTION) < 0.01


def test_inner_fold_is_deterministic(split_df):
    train = split_df[split_df.split == "train"]
    first = inner_folds(train).to_numpy()
    second = inner_folds(train.copy()).to_numpy()
    np.testing.assert_array_equal(first, second)


def test_no_dev_peptide_is_a_near_neighbour_of_a_fit_peptide(split_df):
    """The leakage guard the hand-rolled MLP exists for.

    Stopping on a fold that shares near-duplicate peptides with the fit rows
    would pick the epoch where the model had memorised them. Because the inner
    cut moves whole Hamming <= 3 clusters, the minimum distance across it must
    exceed the clustering threshold -- the same guarantee the frozen splits give
    between train and test.
    """
    train = split_df[split_df.split == "train"]
    fold = inner_folds(train)
    fit_codes = encode_sequences(
        sorted(train.loc[(fold == "fit").to_numpy(), "peptide"].unique()), PEPTIDE_LENGTH)
    dev_codes = encode_sequences(
        sorted(train.loc[(fold == "dev").to_numpy(), "peptide"].unique()), PEPTIDE_LENGTH)
    assert len(fit_codes) and len(dev_codes)

    best = PEPTIDE_LENGTH
    for start in range(0, len(dev_codes), 256):
        block = dev_codes[start:start + 256]
        dist = (block[:, None, :] != fit_codes[None, :, :]).sum(axis=2)
        best = min(best, int(dist.min()))
    assert best > HAMMING_THRESHOLD, (
        f"nearest fit/dev peptide pair is {best} substitutions apart; the inner "
        f"fold must keep Hamming <= {HAMMING_THRESHOLD} clusters whole")
