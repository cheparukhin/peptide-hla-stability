"""Synthetic completed-fold fixtures test reporting gates and matched arithmetic."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pepstab import evaluation as ev, nested
from pepstab.boltz_embeddings import sha256_file
from pepstab.round2_embedding import REPRESENTATIONS
from pepstab.round2_nested import Ancestry, config_digest
from pepstab.round2_sequence import FAMILY_ORDER
from scripts.round2_results import (
    declared_contrasts, load_completed_study, population_contract, report, validate_identity,
)


def write_json(path, item):
    Path(path).write_text(json.dumps(item))


@pytest.fixture
def completed(tmp_path):
    n = 300
    rng = np.random.default_rng(9)
    population = pd.DataFrame({"pair_id": np.arange(n), "cluster_id": np.arange(n)//2,
                               "allele": ["A", "B"] * (n//2), "split": "train",
                               "y_log1p": rng.integers(0, 8, n).astype(float)})
    population["thalf_hours"] = np.expm1(population.y_log1p)
    contract = population_contract(population)
    folds = nested.outer_folds(population)
    component_values = {arm: population.y_log1p.to_numpy() + rng.normal(size=n)*.5
                        for arm in FAMILY_ORDER}
    sequence = tmp_path / "sequence"
    components = list(FAMILY_ORDER)
    spec = {**contract, "family_order": components,
            "candidate_grids": {arm: [{"id": f"{arm}_candidate"}] for arm in components}}
    protocol_hash = config_digest(spec)
    sequence.mkdir()
    write_json(sequence / "protocol.json", {"protocol_hash": protocol_hash, "specification": spec})
    population[["pair_id", "cluster_id", "allele"]].assign(outer_fold=folds).to_csv(sequence / "fold_manifest.csv", index=False)
    for fold in range(5):
        directory = sequence / f"fold_{fold}"
        directory.mkdir()
        held, fit = population[folds == fold], population[folds != fold]
        tables, selections = [], {}
        ancestry = Ancestry()
        for arm in components:
            candidate = spec["candidate_grids"][arm][0]
            selections[arm] = {"candidate": candidate, "scores": [{"candidate": candidate["id"],
                "fold_scores": [.6, .6, .6], "mean": .6, "se": 0}], "rule": "frozen"}
            tables.append(held[["pair_id", "cluster_id", "allele"]].assign(
                outer_fold=fold, arm=arm, y_pred=component_values[arm][folds == fold], selected_candidate=candidate["id"]))
            ancestry.add(role="outer_selected_prediction", fit=fit, held=held, selection=fit, candidate=candidate["id"])
        selected = tables[0].copy()
        selected["arm"] = "sequence_selector"
        pd.concat(tables + [selected]).to_csv(directory / "predictions.csv", index=False)
        write_json(directory / "selections.json", selections)
        write_json(directory / "selector.json", {"family": "mlp_log"})
        write_json(directory / "ancestry.json", ancestry.as_dict())
        write_json(directory / "completed.json", {"protocol_hash": protocol_hash, "audit": ancestry.audit(population, held),
                   "prediction_sha256": sha256_file(directory / "predictions.csv")})
    return population, sequence, tmp_path / "results"


def add_embedding(population, sequence, directory):
    directory.mkdir()
    contract = population_contract(population)
    folds = nested.outer_folds(population)
    components = [f"{rep}_{learner}" for rep in REPRESENTATIONS for learner in ("ridge", "mlp")]
    comparisons = {"positional_per_learner": [["R3", "R3_pooled"], ["R5", "R5_shuffled"],
                                              ["R5_position9x34", "R5_position9x34_pooled"]],
                   "confounded_component_contrasts": [["R3", "R2"], ["R5_position9x34", "R5"]]}
    spec = {**{k: v for k, v in contract.items() if k != "fold_hash"}, "outer_fold_hash": contract["fold_hash"],
            "representations": list(REPRESENTATIONS), "component_order": components, "comparisons": comparisons,
            "selector_order": ["mlp_log"] + [f"{r}_{l}" for r in ("R2", "R3", "R4", "R5") for l in ("ridge", "mlp")],
            "candidate_grids": {arm: [{"id": f"{arm}_candidate"}] for arm in components}}
    digest = config_digest(spec)
    write_json(directory / "protocol.json", {"protocol_hash": digest, "specification": spec})
    population[["pair_id", "cluster_id", "allele"]].assign(outer_fold=folds).to_csv(directory / "fold_manifest.csv", index=False)
    seq_protocol = json.loads((sequence / "protocol.json").read_text())["protocol_hash"]
    for fold in range(5):
        fold_dir = directory / f"fold_{fold}"
        fold_dir.mkdir()
        held, fit = population[folds == fold], population[folds != fold]
        sequence_table = pd.read_csv(sequence / f"fold_{fold}/predictions.csv")
        references = sequence_table[sequence_table.arm.isin(["mlp_log", "sequence_selector"])]
        tables = [references]
        ancestry = Ancestry()
        for arm in components:
            arm_dir = fold_dir / arm
            arm_dir.mkdir()
            candidate = spec["candidate_grids"][arm][0]
            choice = {"candidate": candidate, "scores": [{"candidate": candidate["id"],
                "fold_scores": [.6, .6, .6], "mean": .6, "se": 0}], "rule": "frozen"}
            write_json(arm_dir / "selection.json", choice)
            tables.append(held[["pair_id", "cluster_id", "allele"]].assign(
                outer_fold=fold, arm=arm, y_pred=held.y_log1p.to_numpy(), selected_candidate=candidate["id"]))
            ancestry.add(role="outer_selected_prediction", fit=fit, held=held, selection=fit, candidate=candidate["id"])
        selected = references[references.arm == "mlp_log"].copy()
        selected["arm"] = "embedding_selector"
        pd.concat(tables + [selected]).to_csv(fold_dir / "predictions.csv", index=False)
        write_json(fold_dir / "selector.json", {"arm": "mlp_log"})
        write_json(fold_dir / "ancestry.json", ancestry.as_dict())
        sequence_complete = json.loads((sequence / f"fold_{fold}/completed.json").read_text())
        write_json(fold_dir / "completed.json", {"protocol_hash": digest, "audit": ancestry.audit(population, held),
                    "sequence_protocol_hash": seq_protocol,
                    "sequence_prediction_sha256": sequence_complete["prediction_sha256"],
                    "prediction_sha256": sha256_file(fold_dir / "predictions.csv")})
    return directory


def test_requires_all_five_markers_before_reading_predictions(completed, monkeypatch):
    train, sequence, _ = completed
    (sequence / "fold_4/completed.json").unlink()
    def forbid(*args, **kwargs):
        raise AssertionError("partial outcomes must not be loaded")
    monkeypatch.setattr(pd, "read_csv", forbid)
    with pytest.raises(ValueError, match="all five"):
        load_completed_study(sequence, "sequence", train)


def test_hash_protocol_and_identity_rejection(completed):
    train, sequence, _ = completed
    path = sequence / "fold_0/predictions.csv"
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="content hash"):
        load_completed_study(sequence, "sequence", train)
    path.write_bytes(original)
    table = pd.read_csv(path)
    table.loc[0, "cluster_id"] = -1
    table.to_csv(path, index=False)
    complete_path = sequence / "fold_0/completed.json"
    complete = json.loads(complete_path.read_text())
    complete["prediction_sha256"] = sha256_file(path)
    write_json(complete_path, complete)
    with pytest.raises(ValueError, match="cluster"):
        load_completed_study(sequence, "sequence", train)


def test_no_unmatched_or_changed_allele_rows(completed):
    train, sequence, _ = completed
    held = train[nested.outer_folds(train) == 0]
    table = pd.read_csv(sequence / "fold_0/predictions.csv").query("arm == 'mlp_log'")
    with pytest.raises(ValueError, match="coverage"):
        validate_identity(held, table.iloc[:-1], 0)
    with pytest.raises(ValueError, match="allele"):
        validate_identity(held, table.assign(allele="wrong"), 0)
    with pytest.raises(ValueError, match="outer-fold"):
        validate_identity(held, table.assign(outer_fold=4), 0)


def test_sequence_report_shares_draws_and_keeps_fold_stats_separate(completed):
    train, sequence, output = completed
    status = report(train, sequence, output, n_boot=41)
    assert status["scope"] == "sequence_only"
    assert status["bootstrap"]["20"]["n_boot"] == 41
    assert status["shared_draws_across_all_procedures"]
    pooled = pd.read_csv(output / "pooled_scores.csv")
    folds = pd.read_csv(output / "fold_scores.csv")
    assert set(pooled.scope) == {"pooled_outer_held"}
    assert set(folds.scope) == {"outer_fold_only"}
    assert len(folds) == 5 * len(pooled)
    panel = ev.eligible_alleles(train.allele, train.y_log1p, min_rows=20)
    predictions = load_completed_study(sequence, "sequence", train)["predictions"]
    expected = ev.score("original", train.allele, train.y_log1p, predictions["mlp_log"], alleles=panel).median_spearman
    actual = pooled.query("procedure == 'mlp_log' and panel_min_rows == 20").iloc[0]
    assert actual.median_per_allele_spearman == pytest.approx(expected)
    bootstrap = pd.read_csv(output / "bootstrap_scores_min20.csv")
    np.testing.assert_array_equal(bootstrap.mlp_log, bootstrap.sequence_selector)
    assert not status["diagnostics"]["paired_seed0_1"]["complete"]
    assert (output / "REPORT.md").exists()


def test_embedding_all_declared_components_and_matched_contrasts(completed):
    train, sequence, output = completed
    embedding = add_embedding(train, sequence, output.parent / "embedding")
    status = report(train, sequence, output, embedding_directory=embedding, n_boot=31)
    assert status["scope"] == "sequence_and_embedding"
    comparisons = pd.read_csv(output / "paired_comparisons.csv")
    primary = comparisons.query("comparison_kind == 'primary'")
    assert set(primary.baseline) == {"sequence_selector"}
    assert set(primary.candidate) == {"embedding_selector"}
    assert len(primary) == 2
    positional = comparisons.query("comparison_kind == 'isolated_position_control' and panel_min_rows == 20")
    assert len(positional) == 6
    assert "R3_pooled_ridge" in set(positional.baseline)
    confounded = comparisons.query("comparison_kind == 'confounded_component' and panel_min_rows == 20")
    assert len(confounded) == 4
    assert confounded.interpretation.str.contains("cannot isolate").all()
    matched = comparisons.query("comparison_kind == 'matched_head' and panel_min_rows == 20")
    assert len(matched) == 2 * (len(REPRESENTATIONS)-1)


def test_diagnostics_require_full_cohort_and_report_seeds(completed):
    train, sequence, output = completed
    for fold in range(5):
        path = sequence / f"fold_{fold}"
        table = pd.read_csv(path / "predictions.csv")
        table = table[table.arm.isin(FAMILY_ORDER)]
        first = table.assign(seed=0)
        second = table.assign(seed=1, y_pred=table.y_pred + .2)
        pd.concat([first, second]).to_csv(path / "seed_predictions.csv", index=False)
        marker = json.loads((path / "completed.json").read_text())
        marker["seed_sha256"] = sha256_file(path / "seed_predictions.csv")
        write_json(path / "completed.json", marker)
    status = report(train, sequence, output, n_boot=23)
    assert status["diagnostics"]["paired_seed0_1"]["complete"]
    differences = pd.read_csv(output / "execution_differences.csv")
    np.testing.assert_allclose(differences.mean_absolute_prediction_difference, .2)
    assert len(differences) == len(FAMILY_ORDER)
    # Missing one complete-fold optional file must never produce a partial comparison.
    (sequence / "fold_4/seed_predictions.csv").unlink()
    status = report(train, sequence, output, n_boot=23)
    assert not status["diagnostics"]["paired_seed0_1"]["complete"]
    comparisons = pd.read_csv(output / "paired_comparisons.csv")
    assert "seed_diagnostic" not in set(comparisons.comparison_kind)


def test_validation_rejects_original_validation_rows(completed):
    train, sequence, output = completed
    with pytest.raises(ValueError, match="original-training"):
        report(train.assign(split="val"), sequence, output, n_boot=3)


def test_ancestry_audit_rejects_held_target_ancestry(completed):
    train, sequence, _ = completed
    path = sequence / "fold_0/ancestry.json"
    item = json.loads(path.read_text())
    item["records"][0]["selection_rows"] = item["records"][0]["prediction_rows"]
    write_json(path, item)
    with pytest.raises(AssertionError, match="ancestry violation"):
        load_completed_study(sequence, "sequence", train)


def test_repeat_outputs_are_locked_full_cohort_diagnostics(completed):
    train, sequence, output = completed
    for fold in range(5):
        path = sequence / f"fold_{fold}"
        table = pd.read_csv(path / "predictions.csv")
        table = table[table.arm.isin(["mlp_log", "tree_log", "pairlogit"])]
        pd.concat([table.assign(repeat=1, y_pred=table.y_pred + .01),
                   table.assign(repeat=2, y_pred=table.y_pred + .02)]).to_csv(path / "repeat_predictions.csv", index=False)
        ancestry = json.loads((path / "ancestry.json").read_text())
        write_json(path / "ancestry_with_repeats.json", ancestry)
        pd.DataFrame({"repeat": [1, 2], "arm": ["mlp_log", "mlp_log"]}).to_csv(path / "repeat_members.csv", index=False)
        primary = json.loads((path / "completed.json").read_text())
        write_json(path / "repeats_completed.json", {"repeats": 2,
            "protocol_hash": primary["protocol_hash"], "primary_completion_sha256": sha256_file(path / "completed.json"),
            "repeat_prediction_sha256": sha256_file(path / "repeat_predictions.csv"),
            "ancestry_sha256": sha256_file(path / "ancestry_with_repeats.json"),
            "member_table_sha256": sha256_file(path / "repeat_members.csv")})
    status = report(train, sequence, output, n_boot=29)
    assert status["diagnostics"]["identical_seed_repeats"]["complete"]
    contrasts = pd.read_csv(output / "paired_comparisons.csv").query("comparison_kind == 'execution_repeat'")
    assert len(contrasts) == 9 * 2
    differences = pd.read_csv(output / "execution_differences.csv")
    assert len(differences) == 9
    assert differences.n_rows.eq(len(train)).all()
    # Completion alone does not excuse a missing row in a repeat prediction.
    path = sequence / "fold_4/repeat_predictions.csv"
    table = pd.read_csv(path)
    table.iloc[1:].to_csv(path, index=False)
    marker_path = sequence / "fold_4/repeats_completed.json"
    marker = json.loads(marker_path.read_text())
    marker["repeat_prediction_sha256"] = sha256_file(path)
    write_json(marker_path, marker)
    with pytest.raises(ValueError, match="coverage"):
        report(train, sequence, output, n_boot=3)
