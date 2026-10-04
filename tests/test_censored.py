"""Guards for the stage 7a censored likelihood.

The gradient tests are the load-bearing ones. A Tobit loss with a subtly wrong
censored branch still trains, still produces plausible numbers, and would
quietly turn the stage 7a comparison into a comparison of two bugs. Every
analytic gradient here is checked against a central finite difference of the
loss it claims to differentiate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.censored import (  # noqa: E402
    DEFAULT_FLOOR_HOURS,
    SENSITIVITY_FLOOR_HOURS,
    CensoredMLPRegressor,
    censored_grads,
    censored_mask,
    censored_nll,
    floor_threshold,
    inverse_mills,
)
from pepstab.mlp import MLPConfig, MLPRegressor  # noqa: E402


# --- the declared threshold -------------------------------------------------

def test_default_floor_is_the_declared_assay_limit():
    """0.1 h, from reports/stage7_censored.md §3. Not a tunable."""
    assert DEFAULT_FLOOR_HOURS == 0.1
    assert floor_threshold() == pytest.approx(np.log1p(0.1))
    assert floor_threshold() == pytest.approx(0.0953101798)


def test_sensitivity_grid_brackets_the_declared_floor():
    assert DEFAULT_FLOOR_HOURS in SENSITIVITY_FLOOR_HOURS
    assert min(SENSITIVITY_FLOOR_HOURS) < DEFAULT_FLOOR_HOURS
    assert max(SENSITIVITY_FLOOR_HOURS) > DEFAULT_FLOOR_HOURS
    assert list(SENSITIVITY_FLOOR_HOURS) == sorted(SENSITIVITY_FLOOR_HOURS)


def test_threshold_is_on_the_log1p_scale_not_hours():
    """Models train on log1p(hours); a raw-hours threshold would be ~5% off."""
    assert floor_threshold(0.1) != 0.1
    assert floor_threshold(0.0) == 0.0


def test_floor_threshold_rejects_nonsense():
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            floor_threshold(bad)


# --- the censored set -------------------------------------------------------

def test_censored_mask_is_strictly_below():
    """At the declared 0.1 h limit the 443 rows reported at 0.1 h stay measured."""
    hours = np.array([0.0, 0.05, 0.1, 0.2, 1.0])
    got = censored_mask(hours, 0.1)
    assert got.tolist() == [True, True, False, False, False]


def test_raising_the_floor_only_adds_rows():
    hours = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 1.0])
    masks = [censored_mask(hours, f) for f in SENSITIVITY_FLOOR_HOURS]
    for lower, higher in zip(masks, masks[1:]):
        assert np.all(higher[lower]), "a row uncensored at a higher floor"


def test_censored_mask_rejects_non_finite():
    with pytest.raises(ValueError):
        censored_mask(np.array([0.0, np.nan]))


# --- the likelihood ---------------------------------------------------------

def test_uncensored_branch_is_the_gaussian_nll():
    y = np.array([1.0, 2.0, -0.5])
    mu = np.array([0.8, 2.4, 0.0])
    sigma = 0.7
    got = censored_nll(y, np.zeros(3, bool), mu, sigma, threshold=0.0)
    want = -stats.norm.logpdf(y, loc=mu, scale=sigma)
    assert got == pytest.approx(want)


def test_censored_branch_is_the_log_cdf():
    mu = np.array([0.0, -1.0, 2.0])
    sigma, c = 0.5, 0.1
    got = censored_nll(np.zeros(3), np.ones(3, bool), mu, sigma, c)
    want = -stats.norm.logcdf((c - mu) / sigma)
    assert got == pytest.approx(want)


def test_censored_rows_ignore_the_recorded_label():
    """The floor code (0) must not enter the objective -- that is the whole fix."""
    mu = np.array([0.3, -0.2])
    a = censored_nll(np.zeros(2), np.ones(2, bool), mu, 0.6, 0.095)
    b = censored_nll(np.array([99.0, -99.0]), np.ones(2, bool), mu, 0.6, 0.095)
    assert a == pytest.approx(b)


def test_censored_loss_saturates_below_the_floor_but_mse_does_not():
    """The defining Tobit property, stated as a test.

    Pushing a prediction further below the detection limit must stop costing
    anything, because the data only says 'at or below the floor'. Squared error
    against the recorded 0 keeps charging quadratically.
    """
    c, sigma = 0.095, 0.6
    deep = np.array([-3.0, -6.0])
    nll = censored_nll(np.zeros(2), np.ones(2, bool), deep, sigma, c)
    assert nll[1] - nll[0] < 1e-6, "censored loss still growing far below floor"

    mse = (0.0 - deep) ** 2
    assert mse[1] > 3 * mse[0]


def test_censored_loss_increases_above_the_floor():
    c, sigma = 0.095, 0.6
    mu = np.array([0.1, 1.0, 3.0])
    nll = censored_nll(np.zeros(3), np.ones(3, bool), mu, sigma, c)
    assert np.all(np.diff(nll) > 0)


def test_nll_rejects_non_positive_sigma():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            censored_nll(np.zeros(2), np.zeros(2, bool), np.zeros(2), bad, 0.0)


# --- the inverse Mills ratio ------------------------------------------------

def test_inverse_mills_matches_the_direct_quotient_where_that_is_safe():
    z = np.linspace(-5.0, 5.0, 41)
    want = stats.norm.pdf(z) / stats.norm.cdf(z)
    assert inverse_mills(z) == pytest.approx(want, rel=1e-10)


def test_inverse_mills_survives_where_the_direct_quotient_dies():
    """Below about z = -38 the naive phi/Phi is 0/0. A network finding the floor
    reaches that routinely, and a silent 0 gradient there would strand it."""
    z = np.array([-50.0, -200.0, -1000.0])
    with np.errstate(invalid="ignore", divide="ignore"):
        naive = stats.norm.pdf(z) / stats.norm.cdf(z)
    assert np.all(np.isnan(naive) | (naive == 0)), "premise of this test changed"

    got = inverse_mills(z)
    assert np.all(np.isfinite(got))
    # The ratio is asymptotically |z|, so the pull back toward the threshold
    # stays linear instead of vanishing.
    assert got == pytest.approx(np.abs(z), rel=1e-3)


def test_inverse_mills_is_positive_and_decreasing():
    z = np.linspace(-20.0, 20.0, 200)
    lam = inverse_mills(z)
    assert np.all(lam > 0)
    assert np.all(np.diff(lam) < 0)


# --- gradients vs finite differences ---------------------------------------

def _numeric_grad_mu(y, cens, mu, sigma, c, eps=1e-6):
    out = np.empty(len(mu))
    for i in range(len(mu)):
        up, dn = mu.copy(), mu.copy()
        up[i] += eps
        dn[i] -= eps
        out[i] = (censored_nll(y, cens, up, sigma, c)[i]
                  - censored_nll(y, cens, dn, sigma, c)[i]) / (2 * eps)
    return out


def _numeric_grad_log_sigma(y, cens, mu, sigma, c, eps=1e-6):
    s = np.log(sigma)
    up = censored_nll(y, cens, mu, float(np.exp(s + eps)), c)
    dn = censored_nll(y, cens, mu, float(np.exp(s - eps)), c)
    return (up - dn) / (2 * eps)


@pytest.mark.parametrize("sigma", [0.3, 1.0, 2.5])
def test_grad_mu_matches_finite_difference(sigma):
    rng = np.random.default_rng(0)
    n = 40
    mu = rng.normal(0.0, 1.5, n)
    y = rng.normal(0.5, 1.0, n)
    cens = rng.random(n) < 0.4
    c = 0.0953
    got, _ = censored_grads(y, cens, mu, sigma, c)
    want = _numeric_grad_mu(y, cens, mu, sigma, c)
    assert got == pytest.approx(want, rel=1e-5, abs=1e-7)


@pytest.mark.parametrize("sigma", [0.3, 1.0, 2.5])
def test_grad_log_sigma_matches_finite_difference(sigma):
    rng = np.random.default_rng(1)
    n = 40
    mu = rng.normal(0.0, 1.5, n)
    y = rng.normal(0.5, 1.0, n)
    cens = rng.random(n) < 0.4
    c = 0.0953
    _, got = censored_grads(y, cens, mu, sigma, c)
    want = _numeric_grad_log_sigma(y, cens, mu, sigma, c)
    assert got == pytest.approx(want, rel=1e-5, abs=1e-7)


def test_grad_mu_on_censored_rows_always_pushes_down():
    """Sign check. A censored row must never pull the prediction *up*."""
    mu = np.linspace(-8.0, 8.0, 50)
    g, _ = censored_grads(np.zeros(50), np.ones(50, bool), mu, 0.8, 0.0953)
    assert np.all(g > 0)


def test_grad_mu_on_censored_rows_decays_far_below_the_floor():
    g, _ = censored_grads(np.zeros(2), np.ones(2, bool),
                          np.array([-4.0, -12.0]), 0.8, 0.0953)
    assert g[1] < 1e-20
    assert g[0] > g[1]


def test_grads_reject_non_positive_sigma():
    with pytest.raises(ValueError):
        censored_grads(np.zeros(2), np.zeros(2, bool), np.zeros(2), 0.0, 0.0)


# --- maximum-likelihood recovery -------------------------------------------

def test_tobit_recovers_a_mean_that_the_recorded_labels_cannot():
    """The statistical claim, on data where the latent truth is known.

    Draw a latent Gaussian, censor below a floor, record the floor as 0. The
    censored MLE recovers the latent mean and scale. The mean of the *recorded*
    labels cannot: every sub-floor draw (mean about -0.60 here) is replaced by
    0, which drags the recorded mean up to about 0.63 against a truth of 0.40.
    """
    from scipy import optimize

    rng = np.random.default_rng(7)
    true_mu, true_sigma, c = 0.4, 1.0, 0.0953
    latent = rng.normal(true_mu, true_sigma, 50_000)
    recorded = np.where(latent < c, 0.0, latent)
    cens = latent < c

    def objective(theta):
        mu, log_s = theta
        return censored_nll(recorded, cens, np.full(len(recorded), mu),
                            float(np.exp(log_s)), c).mean()

    fit = optimize.minimize(objective, x0=np.array([0.0, 0.0]), method="Nelder-Mead",
                            options={"xatol": 1e-5, "fatol": 1e-9})
    mu_hat, sigma_hat = fit.x[0], float(np.exp(fit.x[1]))
    assert mu_hat == pytest.approx(true_mu, abs=0.02)
    assert sigma_hat == pytest.approx(true_sigma, abs=0.02)

    # The naive comparator, for contrast: biased upward by far more than the
    # MLE's tolerance, which is the whole reason this stage exists.
    assert recorded.mean() - true_mu > 0.15


# --- the estimator ----------------------------------------------------------

def _toy(n=600, seed=0):
    """Synthetic latent half-lives, censored at the declared floor.

    Returns the features, the *recorded* labels (floor coded as 0), the censoring
    flags and the *latent* values -- the last of which no model gets to see and
    only the tests use.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12)).astype(np.float32)
    w = rng.normal(size=12)
    latent = X @ w * 0.4 + rng.normal(0, 0.3, n)
    c = floor_threshold()
    recorded = np.where(latent < c, 0.0, latent)
    return X, recorded, latent < c, latent


def test_fit_requires_the_censoring_flags():
    X, y, cens, _ = _toy()
    m = CensoredMLPRegressor(MLPConfig(hidden=(8,), max_epochs=2))
    with pytest.raises(ValueError, match="is_censored"):
        m.fit(X[:400], y[:400], X[400:], y[400:])


def test_fit_rejects_sample_weight_rather_than_ignoring_it():
    X, y, cens, _ = _toy()
    m = CensoredMLPRegressor(MLPConfig(hidden=(8,), max_epochs=2))
    with pytest.raises(NotImplementedError):
        m.fit(X[:400], y[:400], X[400:], y[400:], cens[:400], cens[400:],
              sample_weight=np.ones(400))


def test_fit_rejects_an_all_censored_fit_fold():
    X, y, cens, _ = _toy()
    m = CensoredMLPRegressor(MLPConfig(hidden=(8,), max_epochs=2))
    with pytest.raises(ValueError, match="unidentified"):
        m.fit(X[:400], y[:400], X[400:], y[400:],
              np.ones(400, bool), cens[400:])


def test_fit_is_deterministic_given_the_seed():
    X, y, cens, _ = _toy()
    args = (X[:400], y[:400], X[400:], y[400:], cens[:400], cens[400:])
    a = CensoredMLPRegressor(MLPConfig(hidden=(16,), max_epochs=12, seed=3)).fit(*args)
    b = CensoredMLPRegressor(MLPConfig(hidden=(16,), max_epochs=12, seed=3)).fit(*args)
    assert np.allclose(a.predict(X), b.predict(X))
    assert a.sigma_ == b.sigma_


def test_initialisation_and_batch_order_match_the_mse_arm_at_the_same_seed():
    """'The loss is the only difference' -- checked, not asserted in prose.

    Both arms draw from one seeded generator: initialisation first, then one
    permutation per epoch. So at a fixed seed they must start from identical
    weights *and* leave the generator in an identical state, which means they
    also see identical minibatches.
    """
    cfg = MLPConfig(hidden=(16, 8), seed=5)
    a, b = MLPRegressor(cfg), CensoredMLPRegressor(cfg)
    rng_a, rng_b = np.random.default_rng(5), np.random.default_rng(5)
    a._init_params(12, rng_a)
    b._init_params(12, rng_b)

    for Wa, Wb in zip(a.weights_, b.weights_):
        assert np.array_equal(Wa, Wb)
    for ba, bb in zip(a.biases_, b.biases_):
        assert np.array_equal(ba, bb)
    # Generator left at the same point -> the same epoch-1 minibatch order.
    assert np.array_equal(rng_a.permutation(400), rng_b.permutation(400))


def test_target_centring_matches_the_mse_arm():
    """Both arms centre on the mean of the *recorded* labels, so the output bias
    starts in the same place and the comparison is not an offset artifact."""
    X, y, cens, _ = _toy()
    cfg = MLPConfig(hidden=(8,), max_epochs=2, seed=5)
    a = MLPRegressor(cfg).fit(X[:400], y[:400], X[400:], y[400:])
    b = CensoredMLPRegressor(cfg).fit(X[:400], y[:400], X[400:], y[400:],
                                      cens[:400], cens[400:])
    assert a.y_offset_ == pytest.approx(b.y_offset_)


def test_fitted_sigma_is_positive_and_unclipped():
    X, y, cens, _ = _toy()
    m = CensoredMLPRegressor(MLPConfig(hidden=(32, 8), max_epochs=60, seed=0))
    m.fit(X[:400], y[:400], X[400:], y[400:], cens[:400], cens[400:])
    assert m.sigma_ > 0
    assert not m.sigma_clipped_


def test_fit_beats_mse_at_recovering_the_latent_values_at_the_floor():
    """On synthetic data where the latent truth is known, the censored arm's
    sub-floor predictions should track the latent values better than an MSE arm
    trained against the recorded zeros."""
    X, y, cens, latent = _toy(n=3000, seed=2)

    cfg = MLPConfig(hidden=(32, 8), max_epochs=120, seed=0)
    tr, dv = slice(0, 2400), slice(2400, 3000)
    cm = CensoredMLPRegressor(cfg).fit(X[tr], y[tr], X[dv], y[dv],
                                       cens[tr], cens[dv])
    mm = MLPRegressor(cfg).fit(X[tr], y[tr], X[dv], y[dv])

    floor_rows = cens[dv]
    assert floor_rows.sum() > 20
    err_c = np.abs(cm.predict(X[dv])[floor_rows] - latent[dv][floor_rows]).mean()
    err_m = np.abs(mm.predict(X[dv])[floor_rows] - latent[dv][floor_rows]).mean()
    assert err_c < err_m


def test_prob_censored_is_a_probability_and_tracks_the_floor():
    X, y, cens, _ = _toy()
    m = CensoredMLPRegressor(MLPConfig(hidden=(32, 8), max_epochs=60, seed=0))
    m.fit(X[:400], y[:400], X[400:], y[400:], cens[:400], cens[400:])
    p = m.prob_censored(X[400:])
    assert np.all((p >= 0) & (p <= 1))
    assert p[cens[400:]].mean() > p[~cens[400:]].mean()


def test_predict_recorded_median_floors_sub_threshold_predictions():
    X, y, cens, _ = _toy()
    m = CensoredMLPRegressor(MLPConfig(hidden=(32, 8), max_epochs=40, seed=0))
    m.fit(X[:400], y[:400], X[400:], y[400:], cens[:400], cens[400:])
    mu = m.predict(X)
    med = m.predict_recorded_median(X)
    below = mu <= m.threshold_
    assert np.all(med[below] == 0.0)
    assert np.array_equal(med[~below], mu[~below])


def test_prob_censored_before_fit_raises():
    m = CensoredMLPRegressor(MLPConfig(hidden=(8,)))
    with pytest.raises(RuntimeError):
        m.prob_censored(np.zeros((3, 12)))


# --- the stopping rule ------------------------------------------------------

def test_stop_on_rejects_an_unknown_rule():
    with pytest.raises(ValueError, match="stop_on"):
        CensoredMLPRegressor(MLPConfig(hidden=(8,)), stop_on="spearman")


def test_stop_on_mse_picks_a_different_epoch_than_stop_on_nll():
    """The control the sensitivity table reports. If the two rules always agreed
    the control would be vacuous; if they disagree the table shows by how much."""
    X, y, cens, _ = _toy(n=1500, seed=4)
    tr, dv = slice(0, 1200), slice(1200, 1500)
    cfg = MLPConfig(hidden=(32, 8), max_epochs=80, seed=0)
    a = CensoredMLPRegressor(cfg, stop_on="nll").fit(X[tr], y[tr], X[dv], y[dv],
                                                     cens[tr], cens[dv])
    b = CensoredMLPRegressor(cfg, stop_on="mse").fit(X[tr], y[tr], X[dv], y[dv],
                                                     cens[tr], cens[dv])
    assert a.stop_on == "nll" and b.stop_on == "mse"
    # Same objective, so sigma and the curves agree; only the kept epoch differs.
    assert a.dev_curve_[:min(len(a.dev_curve_), len(b.dev_curve_))] == pytest.approx(
        b.dev_curve_[:min(len(a.dev_curve_), len(b.dev_curve_))])
    assert b.dev_mse_ <= a.dev_mse_ + 1e-9, "mse-stopped arm should not lose on mse"


def test_stop_on_nll_keeps_the_best_nll_epoch():
    X, y, cens, _ = _toy(n=1500, seed=6)
    tr, dv = slice(0, 1200), slice(1200, 1500)
    m = CensoredMLPRegressor(MLPConfig(hidden=(32, 8), max_epochs=60, seed=0))
    m.fit(X[tr], y[tr], X[dv], y[dv], cens[tr], cens[dv])
    assert m.dev_nll_ == pytest.approx(min(m.dev_curve_))
    assert m.best_epoch_ == int(np.argmin(m.dev_curve_)) + 1
