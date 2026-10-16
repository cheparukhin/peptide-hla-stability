"""Synthetic complete-study fixtures exercise complementarity reporting gates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from pepstab import nested
from pepstab.boltz_embeddings import sha256_file
from pepstab.round2_complementarity import assess_outer
from pepstab.round2_embedding import FeatureHead
from pepstab.round2_nested import Ancestry, config_digest
from pepstab.round2_sequence import candidates
from scripts.round2_complementarity import specification
from scripts.round2_complementarity_report import report
from scripts.round2_results import load_completed_study
from test_round2_results import add_embedding, completed, write_json


def _fix_original(sequence):
    protocol_path = sequence / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    candidate = candidates()["mlp_log"][0]
    protocol["specification"]["candidate_grids"]["mlp_log"] = [candidate]
    protocol["protocol_hash"] = config_digest(protocol["specification"])
    write_json(protocol_path, protocol)
    for fold in range(5):
        path = sequence / f"fold_{fold}"
        choices = json.loads((path / "selections.json").read_text())
        choices["mlp_log"]["candidate"] = candidate
        choices["mlp_log"]["scores"][0]["candidate"] = candidate["id"]
        write_json(path / "selections.json", choices)
        table = pd.read_csv(path / "predictions.csv")
        table.loc[table.arm.isin(["mlp_log", "sequence_selector"]), "selected_candidate"] = "mlp_log"
        table.to_csv(path / "predictions.csv", index=False)
        marker = json.loads((path / "completed.json").read_text())
        marker["protocol_hash"] = protocol["protocol_hash"]
        marker["prediction_sha256"] = sha256_file(path / "predictions.csv")
        write_json(path / "completed.json", marker)
        members = []
        for stop in range(5):
            for encoding in ("onehot", "blosum"):
                for seed in range(3):
                    cfg = {"hidden": [256, 64], "l2": 1e-5 if encoding == "onehot" else 1e-3,
                           "max_epochs": 300, "patience": 15, "lr": .001, "seed": seed}
                    members.append({"context": f"outer/{fold}/mlp_log/outer_refit", "stop_fold": stop,
                                    "encoding": encoding, "seed": seed, "final_recipe": True,
                                    "config": json.dumps(cfg)})
        pd.DataFrame(members).to_csv(path / "members.csv", index=False)


@pytest.fixture
def full_study(completed):
    train, sequence, output = completed
    _fix_original(sequence)
    embedding = add_embedding(train, sequence, output.parent / "embedding")
    seq = load_completed_study(sequence, "sequence", train)
    emb = load_completed_study(embedding, "embedding", train)
    directory = output.parent / "complementarity"
    directory.mkdir()
    spec = specification(train, sequence, embedding)
    spec["min_rows_inner"] = 2  # synthetic fixture; production frozen value is20.
    digest = config_digest(spec)
    write_json(directory / "protocol.json", {"protocol_hash": digest, "specification": spec})
    folds = nested.outer_folds(train)
    train[["pair_id", "cluster_id", "allele"]].assign(outer_fold=folds).to_csv(directory / "fold_manifest.csv", index=False)
    rng = np.random.default_rng(21)
    features = {"R2": rng.normal(size=(len(train), 4)).astype(np.float32),
                "R4": rng.normal(size=(len(train), 5)).astype(np.float32)}
    def base(fit, held, context):
        assert "y_log1p" not in held
        return fit.y_log1p.mean() + np.sin(held.pair_id.to_numpy())
    for fold in range(5):
        path = directory / f"fold_{fold}"
        path.mkdir()
        held, fit = train[folds == fold], train[folds != fold]
        ancestry = Ancestry()
        head = FeatureHead(train, features, ancestry)
        alternatives = {rep: emb["predictions"][f"{rep}_ridge"][folds == fold] for rep in ("R2", "R4")}
        with threadpool_limits(limits=1):
            values, choices, meta = assess_outer(fit, held, base, head, ancestry,
                seq["predictions"]["mlp_log"][folds == fold], alternatives, min_rows=2)
        values["sequence_selector"] = seq["predictions"]["sequence_selector"][folds == fold]
        pd.concat([held[["pair_id", "cluster_id", "allele"]].assign(outer_fold=fold, arm=arm, y_pred=value)
                   for arm, value in values.items()]).to_csv(path / "predictions.csv", index=False)
        write_json(path / "selections.json", choices)
        write_json(path / "residual_target_provenance.json", meta["residual_target_provenance"])
        oof = fit[["pair_id", "cluster_id", "allele"]].assign(meta_fold=nested.inner_folds(fit, 3),
                                                            original_clean_oof=meta["base_oof"])
        for rep in ("R2", "R4"):
            oof[f"{rep}_nested_selected_oof"] = meta["embedding_oof"][rep]
            oof[f"{rep}_nested_residual_oof"] = meta["residual_oof"][rep]
        oof.to_csv(path / "meta_inputs.csv", index=False)
        write_json(path / "ancestry.json", ancestry.as_dict())
        pd.DataFrame(head.members).to_csv(path / "members.csv", index=False)
        write_json(path / "input_identity.json", {"protocol_hash": digest,
            "reference_hashes": {**seq["sources"], **emb["sources"]}})
        artifacts = ("selections.json", "residual_target_provenance.json", "meta_inputs.csv",
                     "ancestry.json", "members.csv", "input_identity.json")
        write_json(path / "completed.json", {"schema": "round2_complementarity_completion.1", "protocol_hash": digest,
            "n_rows": len(held), "audit": ancestry.audit(train, held),
            "prediction_sha256": sha256_file(path / "predictions.csv"),
            "artifact_sha256": {name: sha256_file(path / name) for name in artifacts}})
    return train, directory, sequence, embedding, directory / "results"


def test_preflight_never_reads_partial_outcomes(full_study, monkeypatch):
    train, directory, sequence, embedding, output = full_study
    (directory / "fold_4/completed.json").unlink()
    def forbidden(*args, **kwargs):
        raise AssertionError("partial outcomes loaded")
    monkeypatch.setattr(pd, "read_csv", forbidden)
    with pytest.raises(ValueError, match="all five"):
        report(directory, sequence, embedding, output, train=train, n_boot=11)


def test_complete_report_preserves_all_contrasts_and_shared_draws(full_study):
    train, directory, sequence, embedding, output = full_study
    status = report(directory, sequence, embedding, output, train=train, n_boot=31)
    assert status["all_five_folds_complete"] and status["shared_draws_across_procedures"]
    contrasts = pd.read_csv(output / "paired_comparisons.csv")
    assert len(contrasts) == 16 * 2
    assert set(contrasts.panel_min_rows) == {20, 50}
    assert len(contrasts.query("comparison_kind == 'complementarity_vs_embedding_alone'")) == 8
    pooled = pd.read_csv(output / "pooled_scores.csv")
    assert len(pooled) == 8 * 2
    boot = pd.read_csv(output / "bootstrap_scores_min20.csv")
    assert len(boot) == 31
    np.testing.assert_array_equal(boot.mlp_log, boot.sequence_selector)
    assert (output / "REPORT.md").exists()


def test_complementarity_hash_tamper_rejected(full_study):
    train, directory, sequence, embedding, output = full_study
    path = directory / "fold_0/meta_inputs.csv"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="completion hash"):
        report(directory, sequence, embedding, output, train=train, n_boot=5)


def test_weight_rule_rejects_rehashed_selection_tamper(full_study):
    train, directory, sequence, embedding, output = full_study
    path = directory / "fold_0/selections.json"
    item = json.loads(path.read_text())
    item["R2"]["blend"]["scores"][0]["mean"] += .1
    write_json(path, item)
    marker_path = directory / "fold_0/completed.json"
    marker = json.loads(marker_path.read_text())
    marker["artifact_sha256"]["selections.json"] = sha256_file(path)
    write_json(marker_path, marker)
    with pytest.raises(ValueError, match="stored weight"):
        report(directory, sequence, embedding, output, train=train, n_boot=5)


def test_population_and_reference_snapshot_rejected(full_study):
    train, directory, sequence, embedding, output = full_study
    changed = train.copy()
    changed.loc[0, "y_log1p"] += 1
    with pytest.raises(ValueError, match="population mismatch"):
        report(directory, sequence, embedding, output, train=changed, n_boot=5)
    # Source member recipe metadata is part of the fixed-baseline audit.
    path = sequence / "fold_0/members.csv"
    table = pd.read_csv(path)
    table.iloc[:-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="30-member"):
        report(directory, sequence, embedding, output, train=train, n_boot=5)


def test_meta_held_scaler_ancestry_rejected_even_with_rehashed_artifacts(full_study):
    train, directory, sequence, embedding, output = full_study
    path = directory / "fold_0/ancestry.json"
    item = json.loads(path.read_text())
    record = next(record for record in item["records"] if record["role"] == "ridge_head"
                  and "/meta/0/" in record.get("context", ""))
    meta = pd.read_csv(directory / "fold_0/meta_inputs.csv")
    held_id = int(meta[meta.meta_fold == 0].pair_id.iloc[0])
    leak = sorted(item["rowsets"][record["scaler_fit_rows"]] + [held_id])
    ancestry = Ancestry()
    key = ancestry.rows(train[train.pair_id.isin(leak)])
    item["rowsets"][key] = leak
    record["scaler_fit_rows"] = key
    write_json(path, item)
    marker_path = directory / "fold_0/completed.json"
    marker = json.loads(marker_path.read_text())
    marker["artifact_sha256"]["ancestry.json"] = sha256_file(path)
    write_json(marker_path, marker)
    with pytest.raises(ValueError, match="meta-held.*scaling"):
        report(directory, sequence, embedding, output, train=train, n_boot=5)
