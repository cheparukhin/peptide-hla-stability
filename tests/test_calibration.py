"""Guards on the NetMHCstabpan calibration. Run with: pytest tests/ -q

The bug these exist for: a model trained on the paper's ``s = 2^(-t0/th)``
target already emits paper-scale scores, and transforming them *again* before
correlating inflates PCC silently -- it read 0.611 for the t0=1 ensemble where
the true paper-scale value is 0.590. Nothing crashes and no metric goes out of
range, so only an explicit test catches it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.evaluation import PANEL_STATISTICS, paired_cluster_bootstrap
from scripts.compare_to_paper import PRED_SCALES, paper_scale, per_allele


# --- the target transform -------------------------------------------------

def test_paper_scale_is_monotone_and_bounded():
    """``s = 2^(-t0/th)`` must rise with half-life and stay in [0, 1)."""
    hours = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0])
    s = paper_scale(np.log1p(hours), t0=1.0)
    assert s[0] == 0.0, "a zero half-life maps to 0, the limit as th -> 0"
    assert np.all(np.diff(s) > 0), "must be strictly increasing in half-life"
    assert np.all((s >= 0) & (s < 1))


def test_paper_scale_clamps_negative_predictions():
    """A predicted half-life below zero is not a value on this scale."""
    assert paper_scale(np.array([-3.0, -0.5]))[0] == 0.0
    assert np.all(paper_scale(np.array([-3.0, -0.5])) == 0.0)


def test_paper_scale_matches_the_published_formula():
    # th = t0 gives 2^-1 exactly, which pins the parameterisation.
    assert paper_scale(np.log1p(np.array([1.0])), t0=1.0)[0] == pytest.approx(0.5)
    assert paper_scale(np.log1p(np.array([2.0])), t0=2.0)[0] == pytest.approx(0.5)
    assert paper_scale(np.log1p(np.array([2.0])), t0=1.0)[0] == pytest.approx(2 ** -0.5)


# --- the prediction-scale contract ---------------------------------------

@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    hours = rng.gamma(2.0, 2.0, size=120)
    hours[rng.random(120) < 0.2] = 0.0          # the assay floor
    allele = np.repeat(["A", "B", "C"], 40)
    return allele, np.log1p(hours), sorted(set(allele))


def test_log1p_predictions_are_converted_once(toy):
    """Perfect log1p predictions must score PCC 1 on both scales."""
    allele, y, alleles = toy
    out = per_allele(allele, y, y.copy(), alleles, pred_scale="log1p")
    assert out["pcc_log1p_mean"] == pytest.approx(1.0)
    assert out["pcc_paper_mean"] == pytest.approx(1.0)


def test_paper_scale_predictions_are_not_converted_again(toy):
    """A model trained on the paper target emits paper-scale scores already.

    Feeding exact paper-scale labels back in must score PCC 1. Declaring them
    ``log1p`` instead transforms them a second time and the correlation drops --
    that gap is the bug.
    """
    allele, y, alleles = toy
    native = paper_scale(y)
    correct = per_allele(allele, y, native, alleles, pred_scale="paper")
    assert correct["pcc_paper_mean"] == pytest.approx(1.0)

    # Declaring them log1p transforms them a second time. The distortion is
    # small -- on real data it moved the t0=1 ensemble from 0.590 to 0.611 --
    # so the guard is that perfect predictions stop scoring perfectly, not that
    # the number collapses.
    double = per_allele(allele, y, native, alleles, pred_scale="log1p")
    assert double["pcc_paper_mean"] < 1.0
    assert double["pcc_paper_mean"] < correct["pcc_paper_mean"]


def test_scc_is_invariant_to_the_prediction_scale(toy):
    """Why SCC is the metric compared against the paper: rank-based."""
    allele, y, alleles = toy
    a = per_allele(allele, y, y.copy(), alleles, pred_scale="log1p")
    b = per_allele(allele, y, paper_scale(y), alleles, pred_scale="paper")
    assert a["scc_mean"] == pytest.approx(b["scc_mean"])
    assert a["scc_median"] == pytest.approx(b["scc_median"])


def test_pcc_log1p_is_undefined_for_paper_scale_predictions(toy):
    """Inverting ``2^(-t0/th)`` is undefined at s=0, where 20% of labels sit."""
    allele, y, alleles = toy
    out = per_allele(allele, y, paper_scale(y), alleles, pred_scale="paper")
    assert np.isnan(out["pcc_log1p_mean"])
    assert out["pred_scale"] == "paper"


def test_unknown_prediction_scale_rejected(toy):
    allele, y, alleles = toy
    with pytest.raises(ValueError, match="pred_scale"):
        per_allele(allele, y, y.copy(), alleles, pred_scale="hours")
    assert set(PRED_SCALES) == {"log1p", "paper"}


# --- the panel statistic --------------------------------------------------

def test_bootstrap_statistic_changes_the_estimate_not_the_contract():
    """``mean`` exists for the paper comparison; ``median`` stays the default."""
    rng = np.random.default_rng(1)
    n = 240
    allele = pd.Series(np.repeat(["A", "B", "C", "D"], n // 4))
    cluster = pd.Series(np.repeat(np.arange(n // 4), 4))
    y = rng.normal(size=n)
    a = y + rng.normal(scale=1.5, size=n)
    b = y + rng.normal(scale=0.4, size=n)

    default = paired_cluster_bootstrap(cluster, allele, y, a, b, n_boot=60)
    median = paired_cluster_bootstrap(cluster, allele, y, a, b, n_boot=60,
                                      statistic="median")
    mean = paired_cluster_bootstrap(cluster, allele, y, a, b, n_boot=60,
                                    statistic="mean")
    assert default["statistic"] == "median"
    assert default["delta_median_spearman"] == median["delta_median_spearman"]
    assert mean["statistic"] == "mean"
    # Both must be finite intervals; they answer different questions, so a
    # median interval must never be quoted as bounding a mean effect.
    for r in (median, mean):
        lo, hi = r["ci95"]
        assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi


def test_unknown_panel_statistic_rejected():
    allele = pd.Series(["A"] * 20)
    cluster = pd.Series(np.arange(20))
    y = np.linspace(0, 1, 20)
    with pytest.raises(ValueError, match="statistic must be one of"):
        paired_cluster_bootstrap(cluster, allele, y, y, y, n_boot=5,
                                 statistic="iqr")
    assert set(PANEL_STATISTICS) == {"median", "mean"}
