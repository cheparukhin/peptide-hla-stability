import json

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from pepstab.round2_embedding import FeatureHead
from pepstab.round2_inference import (
    CachedHead, FeatureBatch, candidate_config, digest, fit_count, fit_head,
    load_feature_batch, load_head,
)
from pepstab.round2_nested import Ancestry
from pepstab.round2_representations import (
    FEATURE_DIMENSIONS, build_feature_bundle, feature_protocol, load_contact_map,
)
from pepstab.boltz_embeddings import write_cache_entry
from scripts.round2_inference import main


def fixture_batch(rep, n=15):
    rng = np.random.default_rng(131)
    X = rng.normal(size=(n, FEATURE_DIMENSIONS[rep])).astype(np.float32)
    contract = {"representation": rep, "feature_protocol": feature_protocol(),
                "contact_map": load_contact_map(), "projection_applied": False}
    batch = FeatureBatch(rep, X, [str(i) for i in range(n)], [f"input-{i}" for i in range(n)], contract, {})
    frame = pd.DataFrame({"pair_id": np.arange(n), "cluster_id": np.arange(n), "split": "train",
                          "y_log1p": X[:, 0] * .5 + 1.})
    return batch, frame


@pytest.mark.parametrize("rep,candidate", [("R4", "R4_ridge_100"), ("R5", "R5_mlp_0.01")])
def test_portable_predictions_match_existing_frozen_head(rep, candidate, tmp_path):
    batch, frame = fixture_batch(rep)
    # One unlabelled synthetic held row verifies exported inference matches the
    # existing head, including exact scaler precision and 30-member ordering.
    fit, held = frame.iloc[:-1], frame.iloc[-1:].drop(columns=["y_log1p"])
    with threadpool_limits(limits=1):
        reference = FeatureHead(frame, {rep: batch.values}, Ancestry(), max_epochs=1)
        expected = reference.predict(fit, held, candidate_config(candidate), "synthetic", final=True)
        model = fit_head(batch, fit, candidate, max_fits=30, authorization={"synthetic": True}, max_epochs=1)
        np.testing.assert_array_equal(expected, model.predict(batch.select(held.pair_id)).prediction_y_log1p)
        model.save(tmp_path / "portable")
        restored = load_head(tmp_path / "portable")
        np.testing.assert_array_equal(model.predict(batch).prediction_y_log1p, restored.predict(batch).prediction_y_log1p)
    assert len(restored.metadata["members"]) == fit_count(candidate)
    if rep == "R5":
        for member in restored.metadata["members"]:
            assert member["gradient_pair_ids_sha256"] == member["scaler_fit_pair_ids_sha256"]
            assert member["gradient_pair_ids_sha256"] != member["stopping_pair_ids_sha256"]
    with pytest.raises(FileExistsError):
        model.save(tmp_path / "portable")


def test_budget_and_training_scope_fail_before_fit():
    batch, frame = fixture_batch("R5")
    with pytest.raises(ValueError, match="requires 30 fits"):
        fit_head(batch, frame, "R5_mlp_0.01", max_fits=29, authorization={"synthetic": True})
    with pytest.raises(ValueError, match="original train"):
        fit_head(batch, frame.assign(split="test"), "R5_mlp_0.01", max_fits=30, authorization={"synthetic": True})
    with pytest.raises(ValueError, match="five nonempty"):
        fit_head(batch, frame.assign(cluster_id=0), "R5_mlp_0.01", max_fits=30, authorization={"synthetic": True})
    with pytest.raises(ValueError, match="explicit frozen"):
        candidate_config("global_winner")


def test_identity_shape_finiteness_and_contract_errors():
    batch, frame = fixture_batch("R4")
    with pytest.raises(ValueError, match="missing"):
        batch.select(["absent"])
    with pytest.raises(ValueError, match="unique"):
        batch.select(["0", "0"])
    batch.values[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite numeric"):
        batch.validate()
    batch, frame = fixture_batch("R4")
    with threadpool_limits(limits=1):
        model = fit_head(batch, frame, "R4_ridge_100", max_fits=1, authorization={"synthetic": True})
    wrong, _ = fixture_batch("R5")
    with pytest.raises(ValueError, match="provenance/representation"):
        model.predict(wrong)
    batch.contract = {**batch.contract, "projection_applied": True}
    with pytest.raises(ValueError, match="provenance/representation"):
        model.predict(batch)


def test_package_corruption_and_parameter_shape_rejected(tmp_path):
    batch, frame = fixture_batch("R4")
    with threadpool_limits(limits=1):
        model = fit_head(batch, frame, "R4_ridge_100", max_fits=1, authorization={"synthetic": True})
    model.save(tmp_path / "portable")
    parameter_file = tmp_path / "portable/parameters.npz"
    parameter_file.write_bytes(parameter_file.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="content hash"):
        load_head(tmp_path / "portable")
    changed = dict(model.arrays)
    changed["m0_coef"] = np.zeros(2)
    with pytest.raises(ValueError, match="finite numeric shape"):
        CachedHead(model.metadata, changed).validate()


def test_existing_compact_bundle_input_and_cli(tmp_path):
    rng = np.random.default_rng(133)
    blocks = {"chain_means_s": rng.normal(size=(3, 384)).astype(np.float32),
              "peptide_s": rng.normal(size=(9, 384)).astype(np.float32),
              "groove_s": rng.normal(size=(182, 384)).astype(np.float32),
              "peptide_to_groove_z": rng.normal(size=(9, 182, 128)).astype(np.float32),
              "groove_to_peptide_z": rng.normal(size=(9, 182, 128)).astype(np.float32)}
    raw = tmp_path / "raw"
    write_cache_entry(raw, blocks, {"pair_id": "case-a", "input_hash": "input-a"})
    bundle = tmp_path / "bundle"
    build_feature_bundle([{"pair_id": "case-a", "input_hash": "input-a", "cache_path": str(raw)}], bundle, load_contact_map())
    batch = load_feature_batch(bundle, "R4")
    assert batch.pair_ids == ["case-a"] and batch.input_hashes == ["input-a"]
    training = pd.DataFrame({"pair_id": ["case-a"], "cluster_id": ["cluster"], "split": ["train"], "y_log1p": [1.]})
    with threadpool_limits(limits=1):
        model = fit_head(batch, training, "R4_ridge_100", max_fits=1, authorization={"synthetic": True})
    model.save(tmp_path / "model")
    assert main(["plan", "--candidate", "R5_mlp_0.01"]) == 0
    assert main(["predict", "--model", str(tmp_path / "model"), "--features", str(bundle), "--out", str(tmp_path / "predictions.csv")]) == 0
    assert pd.read_csv(tmp_path / "predictions.csv").pair_id.tolist() == ["case-a"]
    complete_file = bundle / "complete.json"
    complete = json.loads(complete_file.read_text())
    complete["metadata_file"] = "../escape.json"
    complete_file.write_text(json.dumps(complete))
    with pytest.raises(ValueError, match="basename"):
        load_feature_batch(bundle, "R4")


def test_cli_rejects_spoofed_original_train_before_loading_features(tmp_path, monkeypatch, capsys):
    import pepstab.data
    canonical = pd.DataFrame({"pair_id": [1], "cluster_id": [9], "split": ["train"], "y_log1p": [1.]})
    monkeypatch.setattr(pepstab.data, "load_with_splits", lambda: canonical)
    wrong = canonical.assign(pair_id=2)
    csv = tmp_path / "spoofed.csv"
    wrong.to_csv(csv, index=False)
    with pytest.raises(SystemExit) as exit_info:
        main(["fit", "--candidate", "R4_ridge_100", "--max-fits", "1",
              "--training-csv", str(csv), "--features", str(tmp_path / "absent"),
              "--authorization", str(tmp_path / "absent-auth"), "--out", str(tmp_path / "model")])
    assert exit_info.value.code == 2
    assert "canonical original train" in capsys.readouterr().err
    assert not (tmp_path / "model").exists()
