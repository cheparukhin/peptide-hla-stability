"""Synthetic planning scores/resampling must match the fixed evaluation contract."""
import numpy as np
import pandas as pd
import pytest
from scripts.round2_confirmation_design import (
    Dependence, Panel, bootstrap_deltas, calibrate, endpoint, generate,
    rank_correlation, weighted_ranks, wilson,
)
from pepstab.evaluation import eligible_alleles, paired_cluster_bootstrap, per_allele_spearman, panel_spearman


def test_endpoint_ties_constants_and_coverage_match_contract():
    panel = Panel('test', (25, 25, 10))
    rng = np.random.default_rng(8)
    y = np.maximum(rng.normal(size=(3, 25)), 0)
    pa, pb = rng.normal(size=(2, 3, 25))
    pa[0] = 0
    a = np.concatenate([np.repeat(str(i), n) for i, n in enumerate(panel.counts)])
    yf, af, bf = [np.concatenate([v[i, :n] for i, n in enumerate(panel.counts)]) for v in (y, pa, pb)]
    eligible = eligible_alleles(pd.Series(a), yf, min_rows=20)
    expected = (panel_spearman(per_allele_spearman(a, yf, bf, eligible), eligible).median()
                - panel_spearman(per_allele_spearman(a, yf, af, eligible), eligible).median())
    assert endpoint(y, pa, pb, panel) == pytest.approx(expected, abs=1e-14)
    assert rank_correlation(y[0], pa[0]) == 0
    assert np.isnan(rank_correlation(np.ones(5), np.arange(5)))
    with pytest.raises(ValueError, match='invalid'):
        endpoint(y, pa, np.full(pb.shape, np.nan), panel)


def test_weighted_ranks_equal_explicit_average_ranks():
    from scipy.stats import rankdata
    values = np.array([0., 0., 1., 2., 2., 4.])
    weights = np.array([[1, 2, 0, 3, 1, 0], [0, 2, 4, 0, 1, 2]])
    actual = weighted_ranks(values, weights)
    for row in range(len(weights)):
        explicit = np.repeat(values, weights[row])
        ranks = rankdata(explicit)
        expanded_ranks = np.repeat(actual[row], weights[row])
        np.testing.assert_allclose(expanded_ranks, ranks)


def test_exact_fast_bootstrap_matches_contract_on_crossed_sparse_rows():
    panel = Panel('test', (24, 40, 24))
    y, pa, pb = generate(np.random.default_rng(11), panel, Dependence('test', .3, .2), .7, .75)
    pa[0] = 0
    nboot, seed = 29, 98
    draws = np.random.default_rng(seed).choice(panel.n_clusters, (nboot, panel.n_clusters), replace=True)
    fast = bootstrap_deltas(y, pa, pb, panel, draws)
    allele = np.concatenate([np.repeat(str(i), n) for i, n in enumerate(panel.counts)])
    clusters = np.concatenate([np.arange(n) for n in panel.counts])
    yf, af, bf = [np.concatenate([v[i, :n] for i, n in enumerate(panel.counts)]) for v in (y, pa, pb)]
    expected = paired_cluster_bootstrap(pd.Series(clusters), pd.Series(allele), yf, af, bf,
                                        alleles=['0', '1', '2'], n_boot=nboot, seed=seed)
    np.testing.assert_allclose(np.nanquantile(fast, [.025, .975]), expected['ci95'], atol=1e-14)
    assert endpoint(y, pa, pb, panel) == pytest.approx(expected['delta_median_spearman'], abs=1e-14)


def test_generator_dependence_and_marginal_calibration():
    panel = Panel('dependence', (20000,) * 8)
    weak, strong = Dependence('weak', 0, 0), Dependence('strong', .6, .25)
    yw, _, _ = generate(np.random.default_rng(91), panel, weak, .7, .75)
    ys, _, _ = generate(np.random.default_rng(91), panel, strong, .7, .75)
    assert abs(np.corrcoef(yw[0], yw[1])[0, 1]) < .04
    assert np.corrcoef(ys[0], ys[1])[0, 1] > .7
    assert np.corrcoef(ys[0], ys[4])[0, 1] < np.corrcoef(ys[0], ys[1])[0, 1]
    assert (ys == 0).mean() == pytest.approx(.2, abs=.02)
    calibration = calibrate(n=40000)
    assert np.max(abs(calibration.independent_population_gain - calibration.target_gain)) < .012
    assert np.max(abs(calibration.independent_population_baseline - .66)) < .015
    assert np.max(abs(calibration.independent_floor_share - .20)) < .015


def test_bounds_and_validation():
    low, high = wilson(0, 300)
    assert low == pytest.approx(0, abs=1e-14) and 0 < high < .02
    low, high = wilson(300, 300)
    assert .98 < low < 1 and high == pytest.approx(1)
    with pytest.raises(ValueError):
        Dependence('invalid', .8, .5)


def test_identical_predictions_give_exact_zero_for_every_paired_draw():
    panel = Panel('paired', (25, 25, 40))
    y, pa, _ = generate(np.random.default_rng(7), panel, Dependence('shared', .4, .3), .7, .75)
    draws = np.random.default_rng(12).choice(panel.n_clusters, (31, panel.n_clusters), replace=True)
    assert endpoint(y, pa, pa, panel) == 0
    np.testing.assert_array_equal(bootstrap_deltas(y, pa, pa, panel, draws), np.zeros(31))
