"""Tests for ``pepstab.calibration``.

Phase B. The properties worth pinning here are the ones whose failure would
silently invert a conclusion rather than raise: the sign convention on a
lower-is-better delta, ECE's inability to detect a base-rate predictor (which is
the whole reason it is a diagnostic and not the objective), and the one-row gap
between the two floor-event definitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pepstab import calibration as cal
from pepstab.data import load_with_splits
from pepstab.evaluation import describe_delta


# --------------------------------------------------------------------------
# the floor event


def test_floor_event_rules_differ_on_exactly_one_row():
    """The declared event and the loss's censoring rule disagree on one row.

    Pinned because Phase B scores one and censors on the other. If a data or
    loader change ever makes this gap bigger, the sensitivity check in the report
    stops being a formality and the two rules have to be reconciled.
    """
    df = load_with_splits()
    recorded = cal.floor_indicator(df.thalf_hours, "recorded_zero")
    below = cal.floor_indicator(df.thalf_hours, "below_limit")
    assert recorded.sum() == 5679
    assert below.sum() == 5680
    differing = np.nonzero(recorded != below)[0]
    assert len(differing) == 1
    row = df.iloc[differing[0]]
    assert row.pair_id == 1189
    assert row.thalf_hours == 0.05
    assert row.split == "train"
    # Never the other way round: a recorded zero is always below the limit.
    assert not (recorded & ~below).any()


def test_recorded_zero_share_matches_the_frozen_contract():
    """20.2% in EVALUATION.md is the exact-zero share, to the digit."""
    df = load_with_splits()
    share = cal.floor_indicator(df.thalf_hours, "recorded_zero").mean()
    assert round(share * 100, 1) == 20.2
    assert share == pytest.approx(0.201626, abs=1e-6)


def test_floor_indicator_rejects_unknown_rule():
    with pytest.raises(ValueError, match="rule must be one of"):
        cal.floor_indicator([0.0, 1.0], "whatever")


# --------------------------------------------------------------------------
# proper scores


def test_brier_is_zero_for_a_perfect_forecast_and_one_for_a_reversed_one():
    y = np.array([True, False, True, False])
    assert cal.brier(y, np.array([1.0, 0.0, 1.0, 0.0])) == 0.0
    assert cal.brier(y, np.array([0.0, 1.0, 0.0, 1.0])) == 1.0


def test_brier_rewards_the_base_rate_over_a_constant_half():
    """A proper score prefers the honest marginal to a hedged 0.5."""
    y = np.array([True] * 20 + [False] * 80)
    assert cal.brier(y, np.full(100, 0.2)) < cal.brier(y, np.full(100, 0.5))


def test_brier_is_minimised_at_the_true_rate():
    """Strict propriety, checked numerically: no constant beats the base rate."""
    rng = np.random.default_rng(0)
    y = rng.random(5000) < 0.3
    rate = y.mean()
    best = cal.brier(y, np.full(len(y), rate))
    for p in (0.1, 0.2, 0.25, 0.35, 0.4, 0.6):
        assert cal.brier(y, np.full(len(y), p)) > best


def test_log_loss_is_finite_at_the_endpoints():
    """Clipping, so one confident miss gives a large number rather than inf."""
    y = np.array([True, False])
    assert np.isfinite(cal.log_loss(y, np.array([0.0, 1.0])))


def test_scores_reject_out_of_range_and_mismatched_input():
    y = np.array([True, False])
    with pytest.raises(ValueError, match="outside"):
        cal.brier(y, np.array([1.5, 0.0]))
    with pytest.raises(ValueError, match="length mismatch"):
        cal.brier(y, np.array([0.5]))
    with pytest.raises(ValueError, match="invalid value"):
        cal.brier(y, np.array([np.nan, 0.5]))


# --------------------------------------------------------------------------
# ECE: a diagnostic, not an objective


def test_ece_cannot_see_a_base_rate_predictor_but_brier_can():
    """The reason ECE is not the Phase B objective.

    A predictor that returns the base rate for every row has no discrimination
    whatever. Its ECE is ~0 -- it looks perfectly calibrated -- while Brier
    correctly ranks it behind a model that actually separates the classes.
    """
    rng = np.random.default_rng(1)
    y = rng.random(4000) < 0.25
    useless = np.full(len(y), y.mean())
    skilled = np.where(y, 0.8, 0.15)

    ece_useless, _ = cal.fixed_bin_ece(y, useless)
    ece_skilled, _ = cal.fixed_bin_ece(y, skilled)
    assert ece_useless < 0.01                      # looks flawless
    assert ece_skilled > ece_useless               # and beats the skilled model
    assert cal.brier(y, skilled) < cal.brier(y, useless)   # Brier is not fooled


def test_fixed_bins_are_identical_across_arms_and_equal_count_bins_are_not():
    """The stage 7a binning hazard, demonstrated.

    Two arms with different prediction distributions get the same fixed-width
    edges, so an ECE difference is a calibration difference. Equal-count bins
    move with each arm's own predictions.
    """
    rng = np.random.default_rng(2)
    y = rng.random(2000) < 0.2
    concentrated = np.clip(rng.normal(0.2, 0.03, len(y)), 0, 1)
    spread = np.clip(rng.normal(0.2, 0.25, len(y)), 0, 1)

    _, t_a = cal.fixed_bin_ece(y, concentrated)
    _, t_b = cal.fixed_bin_ece(y, spread)
    assert np.array_equal(t_a.edge_low.values, t_b.edge_low.values)
    # Occupancy differs sharply even though the edges do not -- which is exactly
    # why ``n`` is reported next to every observed rate.
    assert t_a.n.max() > t_b.n.max()


def test_fixed_bin_ece_reports_empty_bins_without_crediting_them():
    """An empty bin contributes no weight, and is not scored as a zero gap."""
    y = np.array([True] * 50 + [False] * 50)
    p = np.where(y, 0.95, 0.05)          # occupies only the two end bins
    ece, table = cal.fixed_bin_ece(y, p, n_bins=10)
    assert len(table) == 10
    assert (table.n == 0).sum() == 8
    assert table.n.sum() == len(y)
    assert table.loc[table.n == 0, "observed_floor_share"].isna().all()
    assert ece == pytest.approx(0.05, abs=1e-9)


def test_endpoints_land_in_the_first_and_last_bin():
    y = np.array([False, True])
    _, table = cal.fixed_bin_ece(y, np.array([0.0, 1.0]), n_bins=10)
    assert table.loc[0, "n"] == 1
    assert table.loc[9, "n"] == 1


def test_equal_count_ece_reproduces_the_stage7_value_on_stage7_predictions():
    """Continuity with the published number, so the two ECEs can be compared.

    Stage 7a reported 0.0560 for the MSE arm under equal-count deciles; that is
    the quantity this function computes, and it is *not* the Phase B fixed-bin
    one.
    """
    from scipy.special import ndtr

    df = load_with_splits()
    val = df[df.split == "val"].reset_index(drop=True)
    pred = pd.read_csv("preds/stage7_mse.csv")
    merged = val.merge(pred, on="pair_id", how="left", validate="one_to_one")
    assert merged.y_pred.notna().all()

    c = np.log1p(0.1)
    p_floor = ndtr((c - merged.y_pred.values) / 0.724073891256963)
    is_floor = cal.floor_indicator(merged.thalf_hours, "recorded_zero")
    assert cal.equal_count_ece(is_floor, p_floor) == pytest.approx(0.0560, abs=5e-5)
    assert cal.brier(is_floor, p_floor) == pytest.approx(0.1050, abs=5e-5)


# --------------------------------------------------------------------------
# reliability and discrimination


def test_wilson_interval_stays_inside_the_unit_interval_at_the_extremes():
    lo, hi = cal._wilson(0, 9)
    assert lo == 0.0 and 0.0 < hi < 1.0
    lo, hi = cal._wilson(9, 9)
    assert hi == 1.0 and 0.0 < lo < 1.0
    assert cal._wilson(0, 0) == pytest.approx((float("nan"), float("nan")), nan_ok=True)


def test_small_bins_get_wide_intervals():
    """The point of reporting occupancy: a nine-row bin says almost nothing."""
    small = cal._wilson(5, 9)
    large = cal._wilson(500, 900)
    assert (small[1] - small[0]) > 5 * (large[1] - large[0])


def test_floor_auroc_matches_mannwhitneyu_including_ties():
    from scipy import stats

    rng = np.random.default_rng(3)
    y = rng.random(400) < 0.3
    p = np.round(np.clip(rng.normal(0.3, 0.2, len(y)) + 0.3 * y, 0, 1), 2)  # ties
    u = stats.mannwhitneyu(p[y], p[~y], alternative="two-sided").statistic
    assert cal.floor_auroc(y, p) == pytest.approx(u / (y.sum() * (~y).sum()))


def test_floor_auroc_is_half_for_a_constant_score():
    y = np.array([True] * 10 + [False] * 10)
    assert cal.floor_auroc(y, np.full(20, 0.3)) == pytest.approx(0.5)


def test_floor_auroc_undefined_with_one_class():
    assert np.isnan(cal.floor_auroc(np.ones(5, dtype=bool), np.linspace(0, 1, 5)))


# --------------------------------------------------------------------------
# P(floor) construction


def test_mixture_is_less_extreme_in_the_tails_than_the_plug_in():
    """The stage 7a mismatch, as a property.

    Averaging member probabilities keeps between-member disagreement; plugging
    the mean into one Gaussian throws it away and is more confident at both ends.
    That is why the two must not be mixed across arms in one comparison.
    """
    c = np.log1p(0.1)
    mus = np.array([[-1.0, 0.5, 2.0], [1.0, 0.5, 2.0]])   # members disagree on row 0
    sigmas = np.array([0.7, 0.7])
    mix = cal.prob_floor_mixture(mus, sigmas, c)
    plug = cal.prob_floor_plugin(mus.mean(axis=0), 0.7, c)
    # Row 0: members straddle the threshold, so the mixture is nearer 0.5.
    assert abs(mix[0] - 0.5) < abs(plug[0] - 0.5) or mix[0] == pytest.approx(plug[0])
    # Rows where members agree, the two coincide.
    assert mix[1] == pytest.approx(plug[1])
    assert mix[2] == pytest.approx(plug[2])


def test_mixture_equals_plugin_when_members_are_identical():
    c = np.log1p(0.1)
    mu = np.array([0.1, 1.0, -0.5])
    mus = np.vstack([mu] * 4)
    assert cal.prob_floor_mixture(mus, np.full(4, 0.6), c) == pytest.approx(
        cal.prob_floor_plugin(mu, 0.6, c))


def test_prob_floor_rejects_bad_scales():
    with pytest.raises(ValueError, match="positive"):
        cal.prob_floor_plugin(np.zeros(3), 0.0, 0.1)
    with pytest.raises(ValueError, match="positive"):
        cal.prob_floor_mixture(np.zeros((2, 3)), np.array([0.5, -1.0]), 0.1)
    with pytest.raises(ValueError, match="members but"):
        cal.prob_floor_mixture(np.zeros((2, 3)), np.array([0.5]), 0.1)


# --------------------------------------------------------------------------
# post-hoc calibrators


def _toy_floor_problem(n=4000, seed=5):
    """A latent half-life censored at the declared limit, with a known floor rate."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(1.0, 1.0, n)
    c = np.log1p(0.1)
    is_floor = latent <= c
    y = np.where(is_floor, 0.0, latent)
    mu = latent + rng.normal(0, 0.4, n)        # a noisy regressor
    return mu, y, is_floor


def test_platt_beats_an_uncalibrated_indicator_on_brier():
    mu, _, is_floor = _toy_floor_problem()
    fit, test = slice(0, 2000), slice(2000, None)
    platt = cal.PlattFloorCalibrator().fit(mu[fit], is_floor[fit])
    p = platt.predict_proba(mu[test])
    base = np.full(p.shape, is_floor[fit].mean())
    assert cal.brier(is_floor[test], p) < cal.brier(is_floor[test], base)
    assert platt.coef_ < 0        # higher mu -> lower floor probability


def test_isotonic_is_monotone_decreasing_in_mu():
    mu, _, is_floor = _toy_floor_problem()
    iso = cal.IsotonicFloorCalibrator().fit(mu, is_floor)
    grid = np.linspace(mu.min(), mu.max(), 50)
    p = iso.predict_proba(grid)
    assert np.all(np.diff(p) <= 1e-12)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_gaussian_scale_calibrator_excludes_floor_rows_from_its_scale():
    """Squaring ``(0 - mu)`` at the floor would measure the coding, not the spread."""
    mu, y, is_floor = _toy_floor_problem()
    with_floor = cal.GaussianScaleFloorCalibrator().fit(mu, y)
    without = cal.GaussianScaleFloorCalibrator().fit(mu, y, is_floor)
    assert without.sigma_ < with_floor.sigma_
    assert without.n_fit_ == int((~is_floor).sum())


def test_calibrators_raise_before_fit_and_on_one_class():
    with pytest.raises(RuntimeError, match="before fit"):
        cal.PlattFloorCalibrator().predict_proba(np.zeros(3))
    with pytest.raises(RuntimeError, match="before fit"):
        cal.IsotonicFloorCalibrator().predict_proba(np.zeros(3))
    with pytest.raises(RuntimeError, match="before fit"):
        cal.GaussianScaleFloorCalibrator().predict_proba(np.zeros(3))
    with pytest.raises(ValueError, match="one class"):
        cal.PlattFloorCalibrator().fit(np.arange(5.0), np.ones(5, dtype=bool))


def test_platt_probabilities_are_stable_at_extreme_inputs():
    mu, _, is_floor = _toy_floor_problem()
    platt = cal.PlattFloorCalibrator().fit(mu, is_floor)
    p = platt.predict_proba(np.array([-1e6, 0.0, 1e6]))
    assert np.all(np.isfinite(p))
    assert p[0] == pytest.approx(1.0, abs=1e-9)
    assert p[2] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# paired intervals -- the sign convention is the thing that must not break


def test_bootstrap_delta_is_positive_when_the_arm_improves_a_lower_is_better_score():
    """The sign convention, pinned.

    A silent flip here would turn every calibration conclusion in the report
    upside down while raising nothing, so this is the single most important
    assertion in the file.
    """
    rng = np.random.default_rng(6)
    n = 1200
    cluster = rng.integers(0, 150, n)
    y = rng.random(n) < 0.25
    good = np.where(y, 0.8, 0.1)                 # better
    bad = np.where(y, 0.6, 0.4)                  # worse
    res = cal.paired_cluster_bootstrap_metric(
        cluster, cal.brier, reference=bad, arm=good, labels=y, n_boot=200)
    assert cal.brier(y, good) < cal.brier(y, bad)
    assert res["delta_improvement"] > 0
    assert res["ci95"][0] > 0
    assert describe_delta(res, min_delta=0.0).startswith("improvement")
    # And reversed: the same pair the other way round must be negative.
    rev = cal.paired_cluster_bootstrap_metric(
        cluster, cal.brier, reference=good, arm=bad, labels=y, n_boot=200)
    assert rev["delta_improvement"] == pytest.approx(-res["delta_improvement"])


def test_bootstrap_higher_is_better_flips_the_sign():
    rng = np.random.default_rng(7)
    n = 800
    cluster = rng.integers(0, 100, n)
    y = rng.random(n) < 0.3
    strong = np.where(y, 0.9, 0.1)
    weak = np.full(n, 0.3)
    res = cal.paired_cluster_bootstrap_metric(
        cluster, cal.floor_auroc, reference=weak, arm=strong, labels=y,
        lower_is_better=False, n_boot=200)
    assert res["delta_improvement"] > 0


def test_bootstrap_is_reproducible_and_resamples_clusters_not_rows():
    rng = np.random.default_rng(8)
    n = 600
    # Ten big clusters: a cluster bootstrap must be far wider than a row one.
    cluster = rng.integers(0, 10, n)
    y = rng.random(n) < 0.25
    a = np.clip(rng.random(n), 0.01, 0.99)
    b = np.clip(rng.random(n), 0.01, 0.99)
    kw = dict(reference=a, arm=b, labels=y, n_boot=300)
    first = cal.paired_cluster_bootstrap_metric(cluster, cal.brier, **kw)
    again = cal.paired_cluster_bootstrap_metric(cluster, cal.brier, **kw)
    assert first["ci95"] == again["ci95"]
    assert first["n_clusters"] == 10
    per_row = cal.paired_cluster_bootstrap_metric(np.arange(n), cal.brier, **kw)
    assert first["bootstrap_half_width"] > per_row["bootstrap_half_width"]


def test_bootstrap_zero_delta_for_identical_arms():
    rng = np.random.default_rng(9)
    n = 400
    cluster = rng.integers(0, 50, n)
    y = rng.random(n) < 0.2
    p = np.clip(rng.random(n), 0.01, 0.99)
    res = cal.paired_cluster_bootstrap_metric(
        cluster, cal.brier, reference=p, arm=p, labels=y, n_boot=100)
    assert res["delta_improvement"] == 0.0
    assert res["ci95"] == (0.0, 0.0)
    assert not res["conclusive"]


def test_bootstrap_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        cal.paired_cluster_bootstrap_metric(
            np.zeros(3), cal.brier, reference=np.zeros(3), arm=np.zeros(2),
            labels=np.zeros(3, dtype=bool))
    with pytest.raises(ValueError, match="cluster ids"):
        cal.paired_cluster_bootstrap_metric(
            np.zeros(2), cal.brier, reference=np.zeros(3), arm=np.zeros(3),
            labels=np.zeros(3, dtype=bool))


def test_null_half_width_is_positive_and_brackets_zero():
    """A no-difference comparison: the point estimate should be small, the width real."""
    rng = np.random.default_rng(10)
    n = 1500
    cluster = rng.integers(0, 200, n)
    y = rng.random(n) < 0.25
    p = np.clip(np.where(y, 0.6, 0.2) + rng.normal(0, 0.1, n), 0.01, 0.99)
    res = cal.null_half_width(cluster, cal.brier, predictions=p, labels=y,
                              n_boot=150, n_splits=3)
    assert res["max_half_width"] > 0
    assert res["max_half_width"] >= res["median_half_width"]
    assert len(res["per_split"]) == 3


# --------------------------------------------------------------------------
# censored NLL


def test_censored_nll_matches_the_published_stage7_values():
    """Both arms' validation NLL, against ``stage7_censored_headline.json``."""
    df = load_with_splits()
    val = df[df.split == "val"].reset_index(drop=True)
    c = np.log1p(0.1)
    for arm, sigma, expected in (("mse", 0.724073891256963, 1.0452),
                                 ("censored", 0.7546598231311946, 1.0764)):
        pred = pd.read_csv(f"preds/stage7_{arm}.csv")
        m = val.merge(pred, on="pair_id", how="left", validate="one_to_one")
        # The loss censors on the limit, not on the recorded zero.
        is_floor = cal.floor_indicator(m.thalf_hours, "below_limit")
        got = cal.censored_nll_mean(m.y_pred.values, np.log1p(m.thalf_hours.values),
                                    is_floor, sigma, c)
        assert got == pytest.approx(expected, abs=5e-5), arm


def test_nll_and_brier_are_minimised_at_different_scales():
    """The mechanism behind the stage 7a disagreement, as a property.

    The two metrics are optimised by *different* sigmas on the same predictions,
    so an arm can move one and worsen the other without either number being
    wrong. Censored NLL pays a Gaussian density term on every measured row, so it
    wants the sigma that best describes the residual spread; Brier scores only
    the floor event, so it wants whatever sigma maps ``mu`` onto the floor rate
    best, and those two are not the same quantity.

    This is the honest version of the claim. A tighter sigma does **not** simply
    buy a better Brier -- pushed far enough it saturates the probabilities to 0
    and 1, and confident mistakes then cost more than the sharpness gains. So
    stage 7a's dev-MSE arm having both a tighter sigma (0.579) and a better Brier
    cannot be attributed to the scale alone; it was also a different fit.
    """
    mu, y, is_floor = _toy_floor_problem()
    c = np.log1p(0.1)
    log1p_y = np.where(is_floor, 0.0, y)
    grid = np.linspace(0.05, 1.5, 60)
    nll = [cal.censored_nll_mean(mu, log1p_y, is_floor, s, c) for s in grid]
    bri = [cal.brier(is_floor, cal.prob_floor_plugin(mu, s, c)) for s in grid]

    best_nll, best_bri = grid[int(np.argmin(nll))], grid[int(np.argmin(bri))]
    assert best_nll != best_bri
    # Each metric is genuinely worse at the other's optimum -- i.e. the
    # disagreement is a real trade, not two near-flat curves.
    assert cal.censored_nll_mean(mu, log1p_y, is_floor, best_bri, c) > min(nll)
    assert cal.brier(is_floor, cal.prob_floor_plugin(mu, best_nll, c)) > min(bri)
    # And an over-tight scale hurts Brier rather than helping it, which is the
    # specific intuition that does not survive contact with the numbers.
    assert cal.brier(is_floor, cal.prob_floor_plugin(mu, 0.05, c)) > min(bri)


def test_censored_nll_rejects_bad_sigma():
    with pytest.raises(ValueError, match="positive"):
        cal.censored_nll_mean(np.zeros(3), np.zeros(3), np.zeros(3, bool), 0.0, 0.1)


# --------------------------------------------------------------------------
# rank preservation: which calibrators may touch a ranking metric, and which may not


def test_platt_is_exactly_rank_preserving_but_isotonic_is_not():
    """Why Phase B's ranking metric is scored on ``mu`` and never on a probability.

    Platt is a strictly monotone logistic map, so ``P(floor)`` carries exactly the
    reverse ordering of ``mu`` -- Spearman against ``-mu`` is 1.0 to machine
    precision and no distinct prediction is lost. Isotonic is only *weakly*
    monotone: it is a step function, so it collapses distinct predictions into
    equal values, and because the frozen contract gives ties average ranks those
    new ties move Spearman. Measured on the stage 7a MSE predictions: isotonic on
    the floor event turns 2,776 distinct values into 19 and its Spearman against
    ``-mu`` falls below 1.

    This is not a reason to avoid isotonic for a *calibration* objective, which is
    what it is for. It is the reason a calibrated probability must never be fed to
    the ranking metric.
    """
    from scipy import stats

    df = load_with_splits()
    val = df[df.split == "val"].reset_index(drop=True)
    m = val[["pair_id", "thalf_hours"]].merge(
        pd.read_csv("preds/stage7_mse.csv"), on="pair_id", validate="one_to_one")
    mu = m.y_pred.values
    is_floor = cal.floor_indicator(m.thalf_hours, "recorded_zero")
    n_distinct = len(np.unique(mu))

    platt = cal.PlattFloorCalibrator().fit(mu, is_floor)
    p_platt = platt.predict_proba(mu)
    assert platt.coef_ < 0                                  # higher mu, lower floor risk
    assert len(np.unique(p_platt)) == n_distinct             # no value collapsed
    assert stats.spearmanr(p_platt, -mu).statistic == pytest.approx(1.0, abs=1e-12)

    iso = cal.IsotonicFloorCalibrator().fit(mu, is_floor)
    p_iso = iso.predict_proba(mu)
    assert len(np.unique(p_iso)) < n_distinct / 10           # heavy collapse
    assert stats.spearmanr(p_iso, -mu).statistic < 1.0       # order resolution lost


def test_isotonic_on_the_target_moves_per_allele_spearman_in_both_directions():
    """The shift is a confound, not a bounded one-sided bias.

    Measured on the stage 7a MSE predictions over the 68-allele validation panel:
    isotonic regression of ``y_log1p`` on ``mu`` moves per-allele Spearman both up
    and down, by more than the project's 0.05 bar on some alleles. So an arm
    cannot be defended with "the calibration map would only have hurt me".
    """
    from scipy import stats
    from sklearn.isotonic import IsotonicRegression

    from pepstab.evaluation import eligible_alleles

    df = load_with_splits()
    val = df[df.split == "val"].reset_index(drop=True)
    m = val[["pair_id", "allele", "thalf_hours"]].merge(
        pd.read_csv("preds/stage7_mse.csv"), on="pair_id", validate="one_to_one")
    y = np.log1p(m.thalf_hours.values)
    mu = m.y_pred.values
    alleles = eligible_alleles(m.allele, y, split="val")
    allele_arr = np.asarray(m.allele.astype(str).values)

    mapped = IsotonicRegression(out_of_bounds="clip").fit(mu, y).predict(mu)
    assert len(np.unique(mapped)) < len(np.unique(mu)) / 10

    shifts = []
    for a in alleles:
        sel = allele_arr == a
        r0 = stats.spearmanr(y[sel], mu[sel]).statistic
        r1 = stats.spearmanr(y[sel], mapped[sel]).statistic
        if np.isfinite(r0) and np.isfinite(r1):
            shifts.append(r1 - r0)
    shifts = np.asarray(shifts)
    assert shifts.min() < 0 and shifts.max() > 0          # both directions
    assert np.abs(shifts).max() > 0.05                   # bigger than the bar


def test_a_strictly_monotone_affine_map_cannot_move_the_ranking_metric():
    """The checkable alternative when a magnitude rescale is wanted.

    A strictly positive slope cannot create a tie, so the ranking metric is
    unchanged exactly -- asserted, not assumed.
    """
    from pepstab.evaluation import eligible_alleles, score

    df = load_with_splits()
    val = df[df.split == "val"].reset_index(drop=True)
    m = val[["pair_id", "allele", "thalf_hours"]].merge(
        pd.read_csv("preds/stage7_mse.csv"), on="pair_id", validate="one_to_one")
    y = np.log1p(m.thalf_hours.values)
    mu = m.y_pred.values
    alleles = eligible_alleles(m.allele, y, split="val")

    slope, intercept = np.polyfit(mu, y, 1)
    assert slope > 0
    rescaled = intercept + slope * mu
    assert len(np.unique(rescaled)) == len(np.unique(mu))
    before = score("raw", m.allele, y, mu, alleles=alleles).median_spearman
    after = score("affine", m.allele, y, rescaled, alleles=alleles).median_spearman
    assert after == pytest.approx(before, abs=1e-8)


def test_average_ranks_matches_scipy_including_a_tie_block():
    """The tie rule is the frozen contract's, so the implementation swap must be exact.

    A large tie block is the common case here, not an edge case: 20.2% of labels
    sit at the assay floor.
    """
    from scipy import stats

    rng = np.random.default_rng(11)
    x = rng.random(5000)
    x[:1000] = 0.5                       # a floor-sized tie block
    assert np.allclose(cal._average_ranks(x),
                       stats.rankdata(x, method="average"))
    # all-tied and all-distinct both degenerate correctly
    assert np.allclose(cal._average_ranks(np.ones(7)), np.full(7, 4.0))
    assert np.allclose(cal._average_ranks(np.array([3.0, 1.0, 2.0])),
                       np.array([3.0, 1.0, 2.0]))
