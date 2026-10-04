"""Guards for the stage 3 ESM-2 arm.

Three things can quietly invalidate this stage, and each gets a test here:

1. **Contact-position indexing.** The 34 "contact" embeddings are only the
   contact residues if the index into ``hla_seq`` is right. A wrong index still
   produces a plausible-looking feature block and a plausible-looking score.
2. **Cache identity.** Embeddings are cached per unique sequence and fanned out
   to rows. If the fan-out or the content hash is wrong, rows silently get
   another peptide's embedding.
3. **Comparison parity.** The ESM ensemble has to use the same folds, seeds and
   member count as the stage 2 sequence ensemble, or the headline number
   measures ensembling rather than pretraining.

Tests that need the embedding cache skip when it is absent, so the suite still
runs on a clean checkout; build it with ``scripts/esm_features.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab import esm as pesm
from pepstab.data import load_raw, load_with_splits

CKPT = pesm.DEFAULT_CHECKPOINT


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def unique_hla(raw):
    return raw[["hla_seq", "hla_pseudoseq"]].drop_duplicates()


def _cache_ready(kind: str) -> bool:
    return (pesm.cache_dir(CKPT) / f"{kind}.index.json").exists()


needs_cache = pytest.mark.skipif(
    not (_cache_ready("peptide") and _cache_ready("hla")),
    reason="ESM cache not built; run scripts/esm_features.py")


# --- 1. contact-position indexing ------------------------------------------


def test_contact_positions_wellformed():
    pos = pesm.CONTACT_POSITIONS_1BASED
    assert len(pos) == 34
    assert len(set(pos)) == 34
    assert list(pos) == sorted(pos)
    assert min(pos) >= 1 and max(pos) <= 182


def test_contact_index_reproduces_pseudoseq(unique_hla):
    """The load-bearing check: these columns *are* the supplied pseudosequence."""
    report = pesm.verify_contact_index(unique_hla.hla_seq.astype(str).tolist(),
                                       unique_hla.hla_pseudoseq.astype(str).tolist())
    assert report["exact_match"], report
    assert report["n_mismatched_alleles"] == 0
    assert report["n_alleles"] == 75


def test_contact_index_character_for_character(unique_hla):
    """Spelled out without the helper, so a bug in the helper cannot hide one."""
    idx = [p - 1 for p in pesm.CONTACT_POSITIONS_1BASED]
    for seq, pseudo in zip(unique_hla.hla_seq.astype(str),
                           unique_hla.hla_pseudoseq.astype(str)):
        assert "".join(seq[i] for i in idx) == pseudo


def test_ambiguous_positions_are_reported(unique_hla):
    """Four positions are invariant tyrosines; the report must not hide that."""
    report = pesm.verify_contact_index(unique_hla.hla_seq.astype(str).tolist(),
                                       unique_hla.hla_pseudoseq.astype(str).tolist())
    assigned = {a["assigned"] for a in report["ambiguous"]}
    assert assigned == set(pesm.AMBIGUOUS_CONTACT_POSITIONS)
    for a in report["ambiguous"]:
        assert a["residue"] == "Y"
        assert a["assigned"] in a["indistinguishable_columns"]


def test_a_shifted_index_would_be_caught(unique_hla):
    """A wrong index must fail the check -- otherwise the check proves nothing."""
    original = pesm.CONTACT_POSITIONS_1BASED
    try:
        pesm.CONTACT_POSITIONS_1BASED = tuple(p + 1 for p in original)
        report = pesm.verify_contact_index(unique_hla.hla_seq.astype(str).tolist(),
                                           unique_hla.hla_pseudoseq.astype(str).tolist())
        assert not report["exact_match"]
    finally:
        pesm.CONTACT_POSITIONS_1BASED = original


# --- 2. cache identity ------------------------------------------------------


def test_content_hash_is_order_and_content_sensitive():
    a = ["AAAAAAAAA", "CCCCCCCCC"]
    assert pesm.content_hash(a) == pesm.content_hash(list(a))
    assert pesm.content_hash(a) != pesm.content_hash(a[::-1])
    assert pesm.content_hash(a) != pesm.content_hash(a + ["DDDDDDDDD"])


@needs_cache
def test_cache_holds_unique_sequences_not_rows(raw):
    lookup, reps = pesm.load_cache(CKPT, "peptide", "final")
    assert len(lookup) == raw.peptide.nunique() == 5633
    assert reps.shape == (5633, 9, pesm.CHECKPOINTS[CKPT]["dim"])
    assert len(lookup) < len(raw)  # the whole point of caching per sequence

    lookup_h, reps_h = pesm.load_cache(CKPT, "hla", "final")
    assert len(lookup_h) == raw.hla_seq.nunique() == 75
    assert reps_h.shape == (75, 182, pesm.CHECKPOINTS[CKPT]["dim"])


@needs_cache
def test_is_cached_rejects_a_different_sequence_set(raw):
    seqs = sorted(raw.peptide.astype(str).unique().tolist())
    assert pesm.is_cached(seqs, CKPT, "peptide")
    assert not pesm.is_cached(seqs[:-1], CKPT, "peptide")
    assert not pesm.is_cached(seqs[::-1], CKPT, "peptide")


@needs_cache
def test_row_fanout_matches_the_sequence(raw):
    """Two rows carrying the same peptide must get bit-identical features."""
    sub = raw.head(400)
    block = pesm.peptide_block(sub.peptide.to_numpy(), CKPT, "final", "pos")
    assert block.shape == (len(sub), 9 * pesm.CHECKPOINTS[CKPT]["dim"])
    by_seq: dict[str, int] = {}
    checked = 0
    for i, p in enumerate(sub.peptide.astype(str)):
        if p in by_seq:
            assert np.array_equal(block[i], block[by_seq[p]])
            checked += 1
        else:
            by_seq[p] = i
    # and a direct lookup agrees
    lookup, reps = pesm.load_cache(CKPT, "peptide", "final")
    j = lookup[str(sub.peptide.iloc[0])]
    assert np.array_equal(block[0], reps[j].reshape(-1))


@needs_cache
def test_mean_pooling_is_the_mean_of_the_positions(raw):
    sub = raw.head(50)
    pos = pesm.peptide_block(sub.peptide.to_numpy(), CKPT, "final", "pos")
    mean = pesm.peptide_block(sub.peptide.to_numpy(), CKPT, "final", "mean")
    d = pesm.CHECKPOINTS[CKPT]["dim"]
    assert mean.shape == (len(sub), d)
    np.testing.assert_allclose(pos.reshape(len(sub), 9, d).mean(axis=1), mean,
                               rtol=1e-5, atol=1e-5)


@needs_cache
def test_hla_contact_block_selects_the_verified_positions(raw):
    sub = raw.head(20)
    lookup, reps = pesm.load_cache(CKPT, "hla", "final")
    block = pesm.hla_block(sub.hla_seq.to_numpy(), CKPT, "final", "contact")
    d = pesm.CHECKPOINTS[CKPT]["dim"]
    assert block.shape == (len(sub), 34 * d)
    j = lookup[str(sub.hla_seq.iloc[0])]
    want = reps[j][np.asarray(pesm.CONTACT_POSITIONS_1BASED) - 1].reshape(-1)
    np.testing.assert_array_equal(block[0], want)


@needs_cache
def test_layers_are_distinct(raw):
    """mid and final must not be the same array, or the layer comparison is empty."""
    _, mid = pesm.load_cache(CKPT, "peptide", "mid")
    _, final = pesm.load_cache(CKPT, "peptide", "final")
    assert mid.shape == final.shape
    assert not np.allclose(mid, final)


@needs_cache
def test_unknown_sequence_raises(raw):
    with pytest.raises(KeyError):
        pesm.peptide_block(np.array(["WWWWWWWWW"]), CKPT, "final", "pos")


# --- 3. comparison parity ---------------------------------------------------


def test_esm_ensemble_uses_the_stage2_folds():
    """The ESM arm must import cv_folds, not reimplement it."""
    from scripts import esm_arm
    from scripts import baseline_ensemble

    assert esm_arm.cv_folds is baseline_ensemble.cv_folds
    assert esm_arm.N_FOLDS == baseline_ensemble.N_FOLDS == 5


def test_esm_ensemble_member_count_matches_stage2():
    from scripts import esm_arm
    from scripts.baseline_sequence import SEEDS
    from pepstab.features import ENCODINGS

    seq_members = esm_arm.N_FOLDS * len(ENCODINGS) * len(SEEDS)
    esm_members = esm_arm.N_FOLDS * len(esm_arm.LAYERS) * len(SEEDS)
    assert seq_members == esm_members == 30


def test_esm_arm_uses_the_shared_mlp_and_grid():
    from scripts import esm_arm
    from scripts import baseline_sequence
    from pepstab.mlp import MLPRegressor

    assert esm_arm.MLPRegressor is MLPRegressor
    assert esm_arm.HIDDEN_GRID is baseline_sequence.HIDDEN_GRID
    assert esm_arm.L2_GRID is baseline_sequence.L2_GRID
    assert esm_arm.SEEDS is baseline_sequence.SEEDS
    assert esm_arm.MAX_EPOCHS == baseline_sequence.MAX_EPOCHS
    assert esm_arm.PATIENCE == baseline_sequence.PATIENCE


@needs_cache
def test_transforms_are_fitted_on_train_only():
    """Changing the evaluation frame must not change the training features."""
    from scripts.esm_arm import assemble

    df = load_with_splits()
    train = df[df.split == "train"].head(3000)
    val = df[df.split == "val"].head(400)
    Xa, (Va,), _ = assemble(train, [val], CKPT, "final", "mean", "mean")
    Xb, (Vb,), _ = assemble(train, [val.head(100)], CKPT, "final", "mean", "mean")
    np.testing.assert_array_equal(Xa, Xb)
    np.testing.assert_array_equal(Va[:100], Vb)


@needs_cache
def test_hla_pca_is_lossless_by_rank():
    """75 distinct HLA domains => rank <= 74, so 74 components lose nothing."""
    from scripts.esm_arm import HLA_PCA, _unique_matrix

    _, M = _unique_matrix("hla", "contact", CKPT, "final")
    assert M.shape == (75, 34 * pesm.CHECKPOINTS[CKPT]["dim"])
    s = np.linalg.svd(M - M.mean(axis=0), compute_uv=False)
    assert HLA_PCA == 74
    assert int((s > 1e-4 * s[0]).sum()) <= 74
    assert float((s[:74] ** 2).sum() / (s ** 2).sum()) > 0.9999


def test_no_test_rows_are_ever_loaded():
    """The arm script must not read the test split."""
    src = (Path(__file__).resolve().parent.parent / "scripts" / "esm_arm.py").read_text()
    assert 'split == "test"' not in src
    assert '"test"' not in src


# --- 4. the ridge fast path must equal sklearn ------------------------------


def test_ridge_path_matches_sklearn():
    """The shared-Gram ridge is an optimisation, not a different estimator."""
    from sklearn.linear_model import Ridge
    from scripts.esm_arm import ridge_path

    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 120)).astype(np.float32)
    w = rng.normal(size=120)
    y = (X @ w + rng.normal(scale=0.5, size=600)).astype(np.float32)
    X_dev = rng.normal(size=(150, 120)).astype(np.float32)
    y_dev = (X_dev @ w).astype(np.float32)
    X_eval = rng.normal(size=(90, 120)).astype(np.float32)

    for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
        _, mse, pred, _ = ridge_path(X, y, X_dev, y_dev, X_eval, alphas=(alpha,))
        sk = Ridge(alpha=alpha).fit(X, y)
        np.testing.assert_allclose(pred, sk.predict(X_eval), rtol=1e-3, atol=1e-4)
        assert abs(mse - float(np.mean((sk.predict(X_dev) - y_dev) ** 2))) < 1e-4


def test_ridge_path_picks_the_argmin_alpha_on_dev():
    """alpha is selected on dev, and the reported MSE is that grid minimum."""
    from scripts.esm_arm import ridge_path

    grid = (0.1, 1.0, 10.0, 100.0, 1000.0)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 40)).astype(np.float32)
    w = rng.normal(size=40)
    y = (X @ w + rng.normal(scale=2.0, size=300)).astype(np.float32)
    X_dev = rng.normal(size=(200, 40)).astype(np.float32)
    y_dev = (X_dev @ w).astype(np.float32)

    alpha, mse, _, _ = ridge_path(X, y, X_dev, y_dev, X_dev, alphas=grid)
    singles = {a: ridge_path(X, y, X_dev, y_dev, X_dev, alphas=(a,))[1]
               for a in grid}
    assert alpha == min(singles, key=singles.get)
    assert mse == pytest.approx(min(singles.values()))
