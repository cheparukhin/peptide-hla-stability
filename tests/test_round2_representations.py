"""Meaningful invariants for frozen representation indexing and provenance."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pepstab.boltz_embeddings import json_bytes, sha256_bytes, write_cache_entry
from pepstab.esm import CONTACT_POSITIONS_1BASED
from pepstab.round2_representations import (
    CRYSTALS, FEATURE_DIMENSIONS, REPORT_DIR, SEED, build_contact_map,
    build_feature_bundle, fixed_projection, load_contact_map, representation_features,
    validate_contact_map,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def contact_map():
    return load_contact_map()


@pytest.fixture
def blocks():
    rng = np.random.default_rng(42)
    return {"chain_means_s": rng.normal(size=(3, 384)).astype(np.float32),
            "peptide_s": rng.normal(size=(9, 384)).astype(np.float32),
            "groove_s": rng.normal(size=(182, 384)).astype(np.float32),
            "peptide_to_groove_z": rng.normal(size=(9, 182, 128)).astype(np.float32),
            "groove_to_peptide_z": rng.normal(size=(9, 182, 128)).astype(np.float32)}


def test_crystal_map_reproduces_and_verifies_numbering(contact_map):
    regenerated = build_contact_map(ROOT / "structures/crystal_cache")
    assert regenerated == contact_map
    assert [s["pdb_id"] for s in contact_map["sources"]] == list(CRYSTALS)
    assert all(s["numbering_verified"] for s in contact_map["sources"])
    assert contact_map["fallback_applied"] == []
    assert all(sum(row) > 0 for row in contact_map["mask"])


@pytest.mark.parametrize("shape", [(384, 8), (128, 4)])
def test_fixed_projection_orthonormal_and_frozen(shape):
    p = fixed_projection(*shape)
    assert p.dtype == np.float32
    np.testing.assert_allclose(p.T @ p, np.eye(shape[1]), atol=1e-6)
    assert not p.flags.writeable
    gaussian = np.random.default_rng(SEED).standard_normal(shape)
    q, r = np.linalg.qr(gaussian, mode="reduced")
    q *= np.where(np.diag(r) < 0, -1., 1.)
    np.testing.assert_array_equal(p, q.astype(np.float32))


def test_dimensions_and_full_chain_means(blocks, contact_map):
    features = representation_features(blocks, contact_map)
    assert {k: len(v) for k, v in features.items()} == FEATURE_DIMENSIONS
    np.testing.assert_array_equal(features["R2"], blocks["chain_means_s"].ravel())
    changed = {k: v.copy() for k, v in blocks.items()}
    changed["chain_means_s"][1] += 1
    difference = representation_features(changed, contact_map)["R2"] - features["R2"]
    np.testing.assert_array_equal(difference[:384], 0)
    np.testing.assert_allclose(difference[384:768], 1, atol=3e-7)
    np.testing.assert_array_equal(difference[768:], 0)


def test_token_projection_retains_exact_positions(blocks, contact_map):
    feature = representation_features(blocks, contact_map)["R3"].reshape(191, 8)
    tokens = np.concatenate([blocks["peptide_s"], blocks["groove_s"]])
    np.testing.assert_allclose(feature, tokens @ fixed_projection(384, 8))


def test_pair_directions_positions_and_central_block(blocks, contact_map):
    # Every pair encodes peptide and groove coordinates; reverse direction is distinct.
    index = np.arange(9)[:, None, None] * 1000 + np.arange(182)[None, :, None]
    blocks["peptide_to_groove_z"][:] = index
    blocks["groove_to_peptide_z"][:] = index + 10000
    features = representation_features(blocks, contact_map)
    r4 = features["R4"].reshape(9, 2, 128)
    np.testing.assert_allclose(r4[:, 0, 0], np.arange(9) * 1000 + 90.5)
    np.testing.assert_allclose(r4[:, 1, 0] - r4[:, 0, 0], 10000)
    per_position = np.asarray([p * 1000 + np.mean(np.asarray(pocket) - 1)
                               for p, pocket in enumerate(contact_map["pockets_1based"])])
    r5 = features["R5"].reshape(4, 2, 128)
    np.testing.assert_allclose(r5[:, 0, 0], per_position[[0, 1, 8]].tolist() + [per_position[2:8].mean()], atol=.001)
    np.testing.assert_allclose(r5[:, 1, 0] - r5[:, 0, 0], 10000, atol=.002)
    position = features["R5_position9x34"].reshape(9, 34, 2, 4)
    expected = blocks["peptide_to_groove_z"][:, np.asarray(CONTACT_POSITIONS_1BASED) - 1] @ fixed_projection(128, 4)
    np.testing.assert_allclose(position[:, :, 0], expected)


def test_shuffle_changes_actual_membership_and_features(blocks, contact_map):
    real, shuffled = validate_contact_map(contact_map)
    np.testing.assert_array_equal(real.sum(axis=1), shuffled.sum(axis=1))
    assert np.any(real != shuffled)
    # HLA-coordinate-only pairs make this independent of output column permutations.
    blocks["peptide_to_groove_z"][:] = np.arange(182)[None, :, None] ** 2
    blocks["groove_to_peptide_z"][:] = np.arange(182)[None, :, None] ** 3
    features = representation_features(blocks, contact_map)
    assert not np.array_equal(features["R5"], features["R5_shuffled"])


def test_matched_capacity_controls_share_projection_and_discard_positions(blocks, contact_map):
    features = representation_features(blocks, contact_map)
    pooled = features["R3_pooled"].reshape(191, 8)
    for start, stop, name in ((0, 9, "peptide_s"), (9, 191, "groove_s")):
        expected = blocks[name].mean(axis=0) @ fixed_projection(384, 8)
        np.testing.assert_allclose(pooled[start:stop], np.tile(expected, (stop-start, 1)), atol=2e-7)
    pairs = features["R5_position9x34"].reshape(9, 34, 2, 4)
    pooled_pairs = features["R5_position9x34_pooled"].reshape(9, 34, 2, 4)
    mask, _ = validate_contact_map(contact_map)
    for p in range(9):
        expected = pairs[p, mask[p]].mean(axis=0)
        np.testing.assert_allclose(pooled_pairs[p], np.tile(expected, (34, 1, 1)))
    assert features["R3_pooled"].shape == features["R3"].shape
    assert features["R5_position9x34_pooled"].shape == features["R5_position9x34"].shape
    assert not np.array_equal(features["R3_pooled"], features["R3"])
    assert not np.array_equal(features["R5_position9x34_pooled"], features["R5_position9x34"])


def test_nonfinite_and_v1_blocks_rejected(blocks, contact_map):
    with pytest.raises(ValueError, match="chain_means_s"):
        representation_features({k: v for k, v in blocks.items() if k != "chain_means_s"}, contact_map)
    blocks["groove_to_peptide_z"][3, 42, 0] = np.nan
    with pytest.raises(ValueError, match="groove_to_peptide_z"):
        representation_features(blocks, contact_map)


def test_contact_map_tampering_rejected(contact_map):
    contact_map["mask"][0][0] = 1 - contact_map["mask"][0][0]
    with pytest.raises(ValueError, match="hash"):
        validate_contact_map(contact_map)


def test_bundle_preserves_identity_and_raw_cache(tmp_path, blocks, contact_map):
    cache = tmp_path / "cache"
    write_cache_entry(cache, blocks, {"case": {"pair_id": "pair-01"}, "input_hash": "test-input"})
    original_marker = (cache / "complete.json").read_bytes()
    entries = [{"pair_id": "pair-01", "input_hash": "test-input", "cache_path": str(cache)}]
    result = build_feature_bundle(entries, tmp_path / "features", contact_map)
    assert (cache / "complete.json").read_bytes() == original_marker
    with np.load(tmp_path / "features" / result["array_file"], allow_pickle=False) as bundle:
        assert bundle["pair_id"].tolist() == ["pair-01"]
        assert bundle["input_hash"].tolist() == ["test-input"]
        for name, dimension in FEATURE_DIMENSIONS.items():
            assert bundle[name].shape == (1, dimension)
            assert np.isfinite(bundle[name]).all()
    marker = json.loads((tmp_path / "features/complete.json").read_text())
    assert marker["metadata_sha256"] == sha256_bytes((tmp_path / "features" / marker["metadata_file"]).read_bytes())
    with pytest.raises(ValueError, match="pair_id"):
        build_feature_bundle([{**entries[0], "pair_id": "wrong"}], tmp_path / "bad", contact_map)
    with pytest.raises(ValueError, match="stale"):
        build_feature_bundle([{**entries[0], "input_hash": "wrong"}], tmp_path / "bad", contact_map)
    with pytest.raises(ValueError, match="duplicate"):
        build_feature_bundle(entries * 2, tmp_path / "bad", contact_map)
