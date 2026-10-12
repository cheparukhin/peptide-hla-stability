"""Real bounded-head acceptance checks; no biological candidate is fitted here."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from pepstab import nested
from pepstab.boltz_embeddings import write_cache_entry
from pepstab.round2_embedding import FeatureHead, head_candidates
from pepstab.round2_nested import Ancestry, selected_predict, unlabelled
from pepstab.round2_representations import (
    FEATURE_DIMENSIONS, build_feature_bundle, load_contact_map,
)
from scripts.round2_embedding_screen import (
    SELECTOR_ORDER, choose_embedding_selector, load_compact_features, run_component,
)


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(7)
    n = 150
    X = rng.normal(size=(n, 5)).astype(np.float32)
    population = pd.DataFrame({"pair_id": np.arange(n), "cluster_id": np.arange(n)//2,
                               "allele": ["A", "B", "C"] * (n//3), "split": "train",
                               "y_log1p": X[:, 0]*2 + .1*X[:, 1],
                               "thalf_hours": np.exp(X[:, 0])})
    return population, {"R0": X}


@pytest.mark.parametrize("learner", ["ridge", "mlp"])
def test_real_head_outer_label_invariance(synthetic, learner):
    population, features = synthetic
    outer = nested.outer_folds(population)
    fit, held = population[outer != 0], population[outer == 0]
    changed = held.assign(y_log1p=-10000., thalf_hours=1e8)
    choices, values = [], []
    with threadpool_limits(limits=1):
        for assessment in (held, changed):
            ancestry = Ancestry()
            head = FeatureHead(population, features, ancestry, max_epochs=1)
            pred, choice = selected_predict(fit, assessment, head_candidates("R0", learner), head,
                                           min_rows=2, ancestry=ancestry)
            choices.append(choice)
            values.append(pred)
            assert ancestry.audit(population, held)["outer_cluster_exclusion"]
    assert choices[0] == choices[1]
    np.testing.assert_array_equal(values[0], values[1])


@pytest.mark.parametrize("learner", ["ridge", "mlp"])
def test_scaler_fits_exact_gradient_role(synthetic, learner, monkeypatch):
    import pepstab.round2_embedding as module
    from sklearn.preprocessing import StandardScaler

    population, features = synthetic
    outer = nested.outer_folds(population)
    fit, held = population[outer != 0], population[outer == 0]
    observed = []
    class RecordingScaler(StandardScaler):
        def fit(self, X, y=None, sample_weight=None):
            observed.append(X.copy())
            return super().fit(X, y, sample_weight=sample_weight)
    monkeypatch.setattr(module, "StandardScaler", RecordingScaler)
    ancestry = Ancestry()
    head = FeatureHead(population, features, ancestry, max_epochs=1)
    with threadpool_limits(limits=1):
        head.predict(fit, unlabelled(held), head_candidates("R0", learner)[0], "test")
    assert len(observed) == (1 if learner == "ridge" else 2)
    for X, record in zip(observed, ancestry.records):
        ids = ancestry.rowsets[record["scaler_fit_rows"]]
        assert ids == ancestry.rowsets[record["fit_rows"]]
        np.testing.assert_array_equal(np.sort(X[:, 0]), np.sort(features["R0"][ids, 0]))
        assert not set(ids) & set(held.pair_id)
        if learner == "mlp":
            assert not set(ids) & set(ancestry.rowsets[record["stopping_rows"]])


def test_final_mlp_has_exactly30_clean_members(synthetic):
    population, features = synthetic
    outer = nested.outer_folds(population)
    fit, held = population[outer != 0], population[outer == 0]
    ancestry = Ancestry()
    head = FeatureHead(population, features, ancestry, max_epochs=1)
    with threadpool_limits(limits=1):
        pred = head.predict(fit, unlabelled(held), head_candidates("R0", "mlp")[0], "final", final=True)
    assert len(head.members) == 30
    assert {(m["stop_fold"], m["seed"]) for m in head.members} == {(f, s) for f in range(5) for s in range(6)}
    assert np.isfinite(pred).all()
    assert set(head.last_by_seed) == set(range(6))
    np.testing.assert_allclose(pred, np.mean(list(head.last_by_seed.values()), axis=0), rtol=1e-6)
    assert ancestry.audit(population, held)["predicted_cluster_exclusion"]


def test_seed_diagnostics_survive_cache(synthetic, tmp_path):
    population, features = synthetic
    outer = nested.outer_folds(population)
    fit, held = population[outer != 0], population[outer == 0]
    candidate = head_candidates("R0", "mlp")[0]
    first = FeatureHead(population, features, Ancestry(), cache_dir=tmp_path, max_epochs=1)
    with threadpool_limits(limits=1):
        result = first.predict(fit, unlabelled(held), candidate, "seeds", final=True)
    second = FeatureHead(population, features, Ancestry(), cache_dir=tmp_path, max_epochs=1)
    np.testing.assert_array_equal(result, second.predict(fit, unlabelled(held), candidate, "seeds", final=True))
    for seed in range(6):
        np.testing.assert_array_equal(first.last_by_seed[seed], second.last_by_seed[seed])


def test_head_cache_identity_includes_features(synthetic, tmp_path):
    population, features = synthetic
    outer = nested.outer_folds(population)
    fit, held = population[outer != 0], population[outer == 0]
    candidate = head_candidates("R0", "ridge")[-1]
    first = FeatureHead(population, features, Ancestry(), cache_dir=tmp_path)
    a = first.predict(fit, unlabelled(held), candidate, "same")
    cached_ancestry = Ancestry()
    second = FeatureHead(population, features, cached_ancestry, cache_dir=tmp_path)
    np.testing.assert_array_equal(a, second.predict(fit, unlabelled(held), candidate, "same"))
    assert cached_ancestry.audit(population, held)["predicted_cluster_exclusion"]
    changed = {"R0": features["R0"].copy()}
    changed["R0"][held.index, 0] += 30
    third = FeatureHead(population, changed, Ancestry(), cache_dir=tmp_path)
    b = third.predict(fit, unlabelled(held), candidate, "same")
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert not np.array_equal(a, b)


def test_run_component_records_selection_predictions_and_ancestry(synthetic, tmp_path):
    population, features = synthetic
    outer = nested.outer_folds(population)
    fit, held = population[outer != 0], population[outer == 0]
    with threadpool_limits(limits=1):
        choice, table = run_component(population, features, fit, held, 0, "R0", "ridge",
                                      tmp_path / "arm", tmp_path / "cache", max_epochs=1, min_rows=2)
    assert len(choice["scores"]) == 3
    assert table.pair_id.tolist() == held.pair_id.tolist()
    complete = json.loads((tmp_path / "arm/completed.json").read_text())
    assert complete["actual_model_fits"] == 10  # 3 candidates x3 inner folds +1 final.
    assert complete["audit"]["outer_cluster_exclusion"]


def test_baseline_first_selector_and_secondary_controls_excluded():
    def choice(name, score):
        return {"candidate": {"id": name}, "scores": [{"candidate": name, "mean": score, "se": .1}]}
    components = {arm: choice(arm, .7) for arm in SELECTOR_ORDER[1:]}
    components["R5_shuffled_mlp"] = choice("shuffled", .99)
    components["R3_pooled_mlp"] = choice("pooled", .99)
    selector = choose_embedding_selector(components, {"mlp_log": choice("original", .65)})
    assert selector["arm"] == "mlp_log"
    assert all(row["arm"] in SELECTOR_ORDER for row in selector["scores"])
    components["R2_ridge"] = choice("winner", .9)
    assert choose_embedding_selector(components, {"mlp_log": choice("original", .65)})["arm"] == "R2_ridge"


@pytest.fixture
def compact_bundle(tmp_path):
    rng = np.random.default_rng(17)
    blocks = {"chain_means_s": rng.normal(size=(3, 384)).astype(np.float32),
              "peptide_s": rng.normal(size=(9, 384)).astype(np.float32),
              "groove_s": rng.normal(size=(182, 384)).astype(np.float32),
              "peptide_to_groove_z": rng.normal(size=(9, 182, 128)).astype(np.float32),
              "groove_to_peptide_z": rng.normal(size=(9, 182, 128)).astype(np.float32)}
    entries = []
    for pair_id in (0, 1, 2):
        cache = tmp_path / f"raw{pair_id}"
        write_cache_entry(cache, blocks, {"pair_id": pair_id, "input_hash": f"input{pair_id}"})
        entries.append({"pair_id": pair_id, "input_hash": f"input{pair_id}", "cache_path": str(cache)})
    bundle = tmp_path / "compact"
    build_feature_bundle(entries, bundle, load_contact_map())
    manifest = tmp_path / "production.json"
    manifest.write_text(json.dumps({"feature_bundles": [str(bundle)], "status": "complete",
                                    "expected_rows": 3, "completed_rows": 3}))
    return manifest, bundle


def test_compact_bundle_identity_join_and_coverage(compact_bundle):
    manifest, _ = compact_bundle
    train = pd.DataFrame({"pair_id": [2, 0, 1]})
    features, provenance = load_compact_features(train, manifest)
    assert provenance["pair_ids"] == ["2", "0", "1"]
    assert provenance["input_hashes"] == ["input2", "input0", "input1"]
    assert {k: v.shape for k, v in features.items()} == {k: (3, d) for k, d in FEATURE_DIMENSIONS.items()}
    with pytest.raises(ValueError, match="cohort"):
        load_compact_features(train.iloc[:-1], manifest)
    with pytest.raises(ValueError, match="duplicate"):
        load_compact_features(pd.DataFrame({"pair_id": [0, 0, 1]}), manifest)


def test_compact_corruption_and_incomplete_manifest_rejected(compact_bundle):
    manifest, bundle = compact_bundle
    item = json.loads(manifest.read_text())
    item["status"] = "partial"
    manifest.write_text(json.dumps(item))
    with pytest.raises(ValueError, match="incomplete"):
        load_compact_features(pd.DataFrame({"pair_id": [0, 1, 2]}), manifest)
    item["status"] = "complete"
    manifest.write_text(json.dumps(item))
    complete = json.loads((bundle / "complete.json").read_text())
    metadata = bundle / complete["metadata_file"]
    metadata.write_text(metadata.read_text() + " ")
    with pytest.raises(ValueError, match="hash"):
        load_compact_features(pd.DataFrame({"pair_id": [0, 1, 2]}), manifest)
