"""Synthetic regression tests for nested complementarity target isolation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from pepstab import nested
from pepstab.round2_complementarity import (
    FixedOriginalBase, apply_correction, assess_outer, base_crossfit,
    fitting_graph, meta_predictions,
)
from pepstab.round2_embedding import FeatureHead
from pepstab.round2_nested import Ancestry, blend


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(44)
    n = 300
    X = rng.normal(size=(n, 6)).astype(np.float32)
    frame = pd.DataFrame({"pair_id": np.arange(n), "cluster_id": np.arange(n)//2,
                          "allele": ["A", "B", "C"]*(n//3), "split": "train",
                          "x": X[:, 0], "y_log1p": X[:, 0] + X[:, 1]*.4,
                          "thalf_hours": np.expm1(np.abs(X[:, 0]))})
    return frame, {"R2": X, "R4": X[:, ::-1].copy()}


def fake_base(fit, held, context):
    # Fitting mean makes leaking held labels observable, rather than an inert stub.
    assert "y_log1p" not in held and "thalf_hours" not in held
    assert not set(fit.cluster_id) & set(held.cluster_id)
    return held.x.to_numpy()*.8 + fit.y_log1p.mean()


def test_fixed_original_provider_always_requests_full30_recipe(synthetic):
    population, _ = synthetic
    outer = nested.outer_folds(population)
    fit, held = population[outer != 0], population[outer == 0]
    class RecordingEnsemble:
        def predict(self, fitting, assessment, candidate, context, *, final=False):
            assert final is True
            assert candidate == {"id": "mlp_log", "learner": "mlp", "target": "log"}
            return np.ones(len(assessment))
    from pepstab.round2_nested import unlabelled
    np.testing.assert_array_equal(FixedOriginalBase(RecordingEnsemble())(fit, unlabelled(held), "test"), 1)


def test_outer_target_perturbation_leaves_weights_and_predictions_identical(synthetic):
    population, features = synthetic
    outer = nested.outer_folds(population)
    fit, held = population[outer != 0], population[outer == 0]
    baseline = fake_base(fit, held.drop(columns=["y_log1p", "thalf_hours"]), "outer")
    alternatives = {"R2": held.x.to_numpy(), "R4": held.x.to_numpy()*.5}
    outputs, choices = [], []
    with threadpool_limits(limits=1):
        for assessment in (held, held.assign(y_log1p=-1e8, thalf_hours=1e20)):
            ancestry = Ancestry()
            head = FeatureHead(population, features, ancestry)
            result, selected, _ = assess_outer(fit, assessment, fake_base, head, ancestry,
                                               baseline, alternatives, min_rows=2)
            outputs.append(result)
            choices.append(selected)
            assert ancestry.audit(population, held)["outer_cluster_exclusion"]
    assert choices[0] == choices[1]
    for arm in outputs[0]:
        np.testing.assert_array_equal(outputs[0][arm], outputs[1][arm])


def test_meta_held_labels_excluded_from_base_selection_and_residual_targets(synthetic):
    population, features = synthetic
    outer = nested.outer_folds(population)
    fit, held = population[outer != 0], population[outer == 0]
    meta_fold = nested.inner_folds(fit, 3)
    changed = fit.copy()
    changed.loc[meta_fold == 0, "y_log1p"] = -10000
    results = []
    with threadpool_limits(limits=1):
        for fitting in (fit, changed):
            ancestry = Ancestry()
            head = FeatureHead(population, features, ancestry)
            result = meta_predictions(fitting, fake_base, head, ancestry, min_rows=2)
            results.append(result)
            assert ancestry.audit(population, held)["predicted_cluster_exclusion"]
    np.testing.assert_array_equal(results[0]["base_oof"][meta_fold == 0], results[1]["base_oof"][meta_fold == 0])
    for rep in ("R2", "R4"):
        np.testing.assert_array_equal(results[0]["embedding_oof"][rep][meta_fold == 0], results[1]["embedding_oof"][rep][meta_fold == 0])
        np.testing.assert_array_equal(results[0]["residual_oof"][rep][meta_fold == 0], results[1]["residual_oof"][rep][meta_fold == 0])
        assert results[0]["nested_embedding_choices"][rep][0] == results[1]["nested_embedding_choices"][rep][0]
    assert results[0]["residual_target_provenance"][0] == results[1]["residual_target_provenance"][0]


def test_clean_residual_targets_require_new_crossfits_inside_meta_fitting(synthetic):
    population, features = synthetic
    outer = nested.outer_folds(population)
    fit, held = population[outer != 0], population[outer == 0]
    calls = []
    def recording_base(fitting, assessment, context):
        calls.append((set(fitting.pair_id), set(assessment.pair_id)))
        return fake_base(fitting, assessment, context)
    ancestry = Ancestry()
    head = FeatureHead(population, features, ancestry)
    with threadpool_limits(limits=1):
        meta_predictions(fit, recording_base, head, ancestry, min_rows=2)
    assert len(calls) == 12  # three outer-fit OOF + nine fresh inner-fit OOF calls.
    meta = nested.inner_folds(fit, 3)
    for k in range(3):
        forbidden = set(fit[meta == k].pair_id)
        for train_ids, predicted_ids in calls[3+k*3:6+k*3]:
            assert not forbidden & (train_ids | predicted_ids)
    assert not any(set(held.pair_id) & (a | b) for a, b in calls)


def test_zero_endpoints_are_exact_baseline_including_pooled_score(synthetic):
    from pepstab.evaluation import eligible_alleles, score
    population, _ = synthetic
    base = population.x.to_numpy()
    correction = np.random.default_rng(2).normal(size=len(base))*1000
    for result in (apply_correction(base, correction, 0.), blend(base, correction, 0.)):
        np.testing.assert_array_equal(result, base)
        panel = eligible_alleles(population.allele, population.y_log1p, min_rows=20)
        assert score("zero", population.allele, population.y_log1p, result, alleles=panel).median_spearman == score(
            "base", population.allele, population.y_log1p, base, alleles=panel).median_spearman


def test_explicit_fitting_graph_and_shared_base_crossfits():
    graph = fitting_graph()
    assert graph["per_outer"]["fixed_original_base_members"] == 12*30
    assert graph["per_outer"]["blend_ridge_fits"] == 2*3*(3*3+1)
    assert graph["per_outer"]["residual_ridge_fits"] == 2*(3+1)
    assert graph["five_outer_actual_model_fits"] == 2140
