"""Acceptance gates for per-partition selection and clean ancestry."""
import numpy as np
import pandas as pd
import pytest

from pepstab import nested
from pepstab.round2_nested import (Ancestry, align_predictions, apply_affine, blend,
    clean_crossfit, positive_affine, select, selected_predict, validate_partition)


@pytest.fixture
def population():
    n = 150
    return pd.DataFrame({"pair_id": np.arange(n), "cluster_id": np.arange(n)//2,
                         "allele": ["A", "B", "C"]*(n//3), "split": "train",
                         "x": np.arange(n)%13, "y_log1p": np.arange(n)%13 + .1})


def fake_predict(fit, held, candidate, context):
    assert "y_log1p" not in held
    assert "thalf_hours" not in held
    validate_partition(fit, held)
    return candidate["sign"] * held.x.to_numpy() + fit.y_log1p.mean()


GRID = [{"id": "positive", "sign": 1}, {"id": "negative", "sign": -1}]


def test_outer_label_change_cannot_change_selection_transform_or_prediction(population):
    f = nested.outer_folds(population)
    fit, held = population[f != 0], population[f == 0]
    a, sa = selected_predict(fit, held, GRID, fake_predict, min_rows=2)
    changed = held.copy()
    changed.y_log1p = np.random.default_rng(8).normal(size=len(changed))*100
    b, sb = selected_predict(fit, changed, GRID, fake_predict, min_rows=2)
    assert sa == sb
    np.testing.assert_array_equal(a, b)


def test_selection_is_local_to_fitting_partition(population):
    first = select(population, GRID, fake_predict, min_rows=2)
    reversed_labels = population.assign(y_log1p=-population.y_log1p)
    second = select(reversed_labels, GRID, fake_predict, min_rows=2)
    assert first.candidate["id"] == "positive"
    assert second.candidate["id"] == "negative"


def test_clean_crossfit_excludes_predicted_labels_from_selection(population):
    outer = nested.outer_folds(population)
    fit, held = population[outer != 0], population[outer == 0]
    ancestry = Ancestry()
    first, choices = clean_crossfit(fit, GRID, fake_predict, min_rows=2, ancestry=ancestry)
    inner = nested.inner_folds(fit, 3)
    changed = fit.copy()
    changed.loc[inner == 0, "y_log1p"] = -1e6
    second, _ = clean_crossfit(changed, GRID, fake_predict, min_rows=2)
    np.testing.assert_array_equal(first[inner == 0], second[inner == 0])
    assert len(choices) == 3
    assert ancestry.audit(population, held)["predicted_cluster_exclusion"]


def test_ancestry_rejects_stopping_and_selection_leakage(population):
    ancestry = Ancestry()
    fit, held = population.iloc[:-2], population.iloc[-2:]
    with pytest.raises(ValueError, match="overlapping"):
        ancestry.add(role="bad", fit=fit, held=held, stopping=held)
    with pytest.raises(ValueError, match="overlapping"):
        ancestry.add(role="bad", fit=fit, held=held, selection=population)
    ancestry.add(role="okay", fit=fit, held=held)
    record = ancestry.records[0]
    record["selection_rows"] = ancestry.rows(held)
    with pytest.raises(AssertionError, match="ancestry violation"):
        ancestry.audit(population, held)


def test_identity_join_checks_coverage_clusters_and_order(population):
    pred = population[["pair_id", "cluster_id"]].assign(y_pred=np.arange(len(population)))
    shuffled = pred.sample(frac=1, random_state=3)
    np.testing.assert_array_equal(align_predictions(population, shuffled), np.arange(len(population)))
    for bad in (pred.iloc[:-1], pd.concat([pred, pred.iloc[:1]]), pred.assign(cluster_id=-1)):
        with pytest.raises(ValueError):
            align_predictions(population, bad)


def test_zero_weight_preserves_pooled_baseline_exactly(population):
    from pepstab.evaluation import eligible_alleles, score
    y = population.y_log1p.to_numpy()
    base = y * .3 + np.sin(y)
    result = blend(base, -base, 0.)
    np.testing.assert_array_equal(result, base)
    panel = eligible_alleles(population.allele, y, min_rows=20)
    assert score("a", population.allele, y, result, panel).median_spearman == score(
        "b", population.allele, y, base, panel).median_spearman


def test_positive_affine_preserves_order_and_rejects_nonpositive():
    raw = np.array([-2., -.3, 0., 1.8])
    mapping = positive_affine(raw, 2*raw+4)
    np.testing.assert_allclose(apply_affine(raw, mapping), 2*raw+4)
    bad = positive_affine(raw, -raw)
    assert bad["slope"] > 0
    assert np.all(np.diff(apply_affine(raw, bad)) > 0)
    with pytest.raises(ValueError):
        apply_affine(raw, {"slope": 0, "intercept": 0})


def test_original_training_only(population):
    with pytest.raises(ValueError, match="original training"):
        validate_partition(population.iloc[:-2], population.iloc[-2:].assign(split="test"))
