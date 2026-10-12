import hashlib
import inspect
import io
import json
import textwrap

import numpy as np
import pytest
import torch

from pepstab.boltz_embeddings import (
    BLOCKS, compile_trunk_forward, crop_interface, make_token_map,
    validate_cache_entry, write_cache_entry,
)


class FakePinnedModel:
    """A boundary contract test; never claimed as trained-model equivalence."""
    def __init__(self):
        self.head_calls = 0

    def distogram_module(self, z):
        self.head_calls += 1
        return z

    def forward(self, feats, recycling_steps=0):
        with torch.no_grad():
            s = feats["s"]
            z = feats["z"]
            for i in range(recycling_steps + 1):
                s = s + i + 1
                z = z + 2 * i
            pdistogram = self.distogram_module(z)
            dict_out = {"pdistogram": pdistogram}
        return dict_out


def test_source_pinned_boundary_and_original_forward_intact():
    model = FakePinnedModel()
    original = FakePinnedModel.forward
    checksum = hashlib.sha256(textwrap.dedent(inspect.getsource(original)).encode()).hexdigest()
    extraction, audit = compile_trunk_forward(original, expected_sha256=checksum)
    out = extraction(model, {"s": torch.zeros(2), "z": torch.zeros(2)}, recycling_steps=3)
    assert torch.equal(out["s"], torch.full((2,), 10.))
    assert torch.equal(out["z"], torch.full((2,), 12.))
    assert model.head_calls == 0
    assert FakePinnedModel.forward is original
    model.forward({"s": torch.zeros(2), "z": torch.zeros(2)})
    assert model.head_calls == 1
    assert audit["forward_sha256"] == checksum
    with pytest.raises(ValueError, match="source drift"):
        compile_trunk_forward(original)


def features():
    lengths = [275, 99, 9]
    chains = [{"id": c, "sequence": "A" * n} for c, n in zip("ABC", lengths)]
    n = sum(lengths)
    features = {"token_pad_mask": np.array([[1] * n + [0, 0]]),
                "asym_id": np.array([[i for i, length in enumerate(lengths) for _ in range(length)] + [0, 0]]),
                "residue_index": np.array([[j for length in lengths for j in range(length)] + [0, 0]]),
                "mol_type": np.zeros((1, n + 2)), "res_type": np.ones((1, n + 2, 1))}
    return features, chains


def test_mapping_checks_sequence_padding_and_duplicates():
    feats, chains = features()
    token_map = make_token_map(feats, chains, {0: "A"})
    assert len(token_map) == 383
    assert token_map[-1]["chain_id"] == "C" and token_map[-1]["residue_number"] == 9
    with pytest.raises(ValueError, match="sequence mismatch"):
        make_token_map(feats, chains, {0: "G"})
    feats["residue_index"][0, 1] = 0
    with pytest.raises(ValueError, match="residue indices"):
        make_token_map(feats, chains, {0: "A"})


def test_both_pair_directions_and_roundtrip_corruption(tmp_path):
    feats, chains = features()
    mapping = make_token_map(feats, chains, {0: "A"})
    n = 385
    s = np.arange(n * 2, dtype="float32").reshape(1, n, 2)
    # Deliberately asymmetric: the reverse block must not be copied from forward.
    z = (np.arange(n)[:, None] * .1 + np.arange(n)[None, :] * .01)[None, :, :, None]
    blocks = crop_interface(s, z, mapping)
    assert blocks[BLOCKS[2]].shape == blocks[BLOCKS[3]].shape == (9, 182, 1)
    assert blocks[BLOCKS[2]][0, 0, 0] != blocks[BLOCKS[3]][0, 0, 0]
    assert blocks["peptide_s"][0, 0] == s[0, 374, 0]
    assert blocks["chain_means_s"].shape == (3, 2)
    assert np.array_equal(blocks["chain_means_s"][0], s[0, :275].mean(axis=0).astype("float16"))
    assert np.array_equal(blocks["chain_means_s"][1], s[0, 275:374].mean(axis=0).astype("float16"))
    manifest = write_cache_entry(tmp_path, blocks, {"input_hash": "test-identity", "token_map": mapping})
    info = validate_cache_entry(tmp_path, expected_input_hash="test-identity")
    assert info["blocks"]["groove_s"]["shape"] == [182, 2]
    with pytest.raises(ValueError, match="stale cache"):
        validate_cache_entry(tmp_path, expected_input_hash="different")
    path = tmp_path / manifest["array_file"]
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_cache_entry(tmp_path)


def test_cif_does_not_count_as_complete(tmp_path):
    (tmp_path / "existing.cif").write_text("structure")
    with pytest.raises(FileNotFoundError):
        validate_cache_entry(tmp_path)


def test_original_v1_audit_cache_remains_readable(tmp_path):
    """Adding chain means must not invalidate the completed technical evidence."""
    from pepstab.boltz_embeddings import json_bytes, sha256_bytes, LEGACY_SCHEMA_VERSION
    shapes = {"peptide_s": (9, 2), "groove_s": (182, 2),
              "peptide_to_groove_z": (9, 182, 1), "groove_to_peptide_z": (9, 182, 1)}
    blocks = {k: np.zeros(v, dtype="float16") for k, v in shapes.items()}
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **blocks)
    array = buffer.getvalue()
    info = {"schema_version": LEGACY_SCHEMA_VERSION, "input_hash": "legacy", "npz_sha256": sha256_bytes(array),
            "blocks": {k: {"shape": list(v.shape), "dtype": "float16"} for k, v in blocks.items()}}
    info_bytes = json_bytes(info)
    (tmp_path / "embeddings.npz").write_bytes(array)
    (tmp_path / "metadata.json").write_bytes(info_bytes)
    (tmp_path / "complete.json").write_bytes(json_bytes({"schema_version": LEGACY_SCHEMA_VERSION,
        "array_file": "embeddings.npz", "metadata_file": "metadata.json", "input_hash": "legacy",
        "metadata_sha256": sha256_bytes(info_bytes), "npz_sha256": sha256_bytes(array)}))
    assert validate_cache_entry(tmp_path)["schema_version"] == LEGACY_SCHEMA_VERSION


def test_panel_covers_engineered_borrowed_and_training_only():
    from scripts.round2_boltz_panel import build_panel
    panel = build_panel()
    assert panel["unique_pairs"] == 8 and len(panel["cases"]) == 10
    assert all(c["split"] == "train" for c in panel["cases"])
    assert any("C67S" in c["allele"] for c in panel["cases"])
    assert any("borrowed" in c["construct_note"] for c in panel["cases"])
    assert {c["msa_depth_cap"] for c in panel["cases"]} == {32, 128, 1024}
    assert all("thalf_hours" not in c and "y_log1p" not in c for c in panel["cases"])
