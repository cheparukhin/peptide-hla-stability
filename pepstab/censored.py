"""Tobit-style left-censored Gaussian likelihood as a drop-in MLP objective.

Stage 7a. 20.2% of ``thalf_hours`` labels are exactly 0, which stage 1 showed is
the assay's detection floor rather than a measurement
(``reports/audit_summary.md`` §3). Training on ``log1p`` with a squared error
asks the network to predict 0 at those rows, which is the wrong statement: the
evidence is only that the *latent* half-life lies at or below the floor.

This module says the right thing instead. Writing ``mu`` for the network output,
``sigma`` for a learned homoscedastic scale and ``c`` for the detection limit on
the ``log1p`` scale, the per-row negative log-likelihood is

    uncensored:  0.5 * ((y - mu)/sigma)**2 + log(sigma) + 0.5*log(2*pi)
    censored:    -log Phi((c - mu)/sigma)

The censored branch is the whole point: once ``mu`` sits well below ``c`` the
term flattens out, so a floor row stops pulling the prediction toward 0 and only
says "not above the floor". Under squared error that same row keeps pulling with
full force forever.

**The threshold is not a tunable.** It is an assay constant, declared in
``reports/stage7_censored.md`` from the label distribution before any model was
fitted, and :data:`DEFAULT_FLOOR_HOURS` records it. Choosing it from a
validation score would make the whole comparison circular; the sensitivity sweep
in :data:`SENSITIVITY_FLOOR_HOURS` exists so that the dependence is visible
rather than tuned away.

:class:`CensoredMLPRegressor` subclasses :class:`pepstab.mlp.MLPRegressor` and
overrides **only** the objective. Initialisation, the forward and backward
passes, Adam, the L2 term, target centring and best-epoch restore are inherited
untouched, so a censored arm and an MSE arm at the same seed differ in the loss
and in nothing else -- which is exactly what the stage 7a comparison claims.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import log_ndtr, ndtr

from .mlp import DTYPE, MLPConfig, MLPRegressor

#: The declared detection limit, in hours. Fixed from the assay's reporting grid
#: and the label distribution, never from a validation result -- see
#: ``reports/stage7_censored.md`` §3. 0.1 h is the smallest half-life the dataset
#: reports at any volume (443 rows); exactly one row anywhere sits strictly
#: between 0 and 0.1 h.
DEFAULT_FLOOR_HOURS = 0.1

#: Predeclared sensitivity grid around the floor. Stage 1 locates the detection
#: limit in a range (0.1-0.2 h), so the headline is reported with a sweep. 0.3 is
#: an over-censoring control, not a candidate: it discards 1,297 genuine 0.2 h
#: measurements.
SENSITIVITY_FLOOR_HOURS = (0.05, 0.10, 0.15, 0.20, 0.30)

#: ``log(sigma)`` is clipped to this range. The lower end keeps the Gaussian from
#: collapsing onto a handful of rows early in training (which would send
#: ``1/sigma**2`` gradients to infinity); the upper end is far above any scale
#: ``log1p`` half-life could have. Neither bound is active in a healthy fit, and
#: :attr:`CensoredMLPRegressor.sigma_clipped_` reports if one ever was.
LOG_SIGMA_BOUNDS = (np.log(1e-3), np.log(1e3))

_HALF_LOG_2PI = 0.5 * float(np.log(2.0 * np.pi))


def floor_threshold(floor_hours: float = DEFAULT_FLOOR_HOURS) -> float:
    """The censoring threshold on the ``log1p`` scale.

    Models train on ``y_log1p = log1p(thalf_hours)``, so the detection limit has
    to be carried through the same transform.
    """
    if not np.isfinite(floor_hours) or floor_hours < 0:
        raise ValueError(f"floor_hours must be finite and >= 0, got {floor_hours!r}")
    return float(np.log1p(floor_hours))


def censored_mask(thalf_hours, floor_hours: float = DEFAULT_FLOOR_HOURS) -> np.ndarray:
    """Which rows are floor codes: recorded half-life **strictly below** the limit.

    Strictly below, so that at the declared 0.1 h limit the 443 rows actually
    reported at 0.1 h stay measurements. One consistent rule across the whole
    sensitivity sweep: raising the assumed detection limit moves rows into the
    censored set, because a value below the limit could not have been observed.
    """
    hours = np.asarray(thalf_hours, dtype=float)
    if not np.all(np.isfinite(hours)):
        raise ValueError("thalf_hours contains non-finite values")
    return hours < floor_hours


def inverse_mills(z: np.ndarray) -> np.ndarray:
    """``phi(z) / Phi(z)``, computed in log space so it survives small ``Phi``.

    This ratio is the entire censored-branch gradient. The naive quotient
    underflows to ``0/0`` once ``z`` drops below about -38 in float64, which is
    reached routinely while a network is still finding the floor. Taking the
    difference of log-densities instead is stable: for large negative ``z`` the
    result tends to ``|z|``, which is the correct linear pull back toward the
    threshold rather than a silent zero gradient.
    """
    z = np.asarray(z, dtype=float)
    log_pdf = -0.5 * z * z - _HALF_LOG_2PI
    return np.exp(log_pdf - log_ndtr(z))


def censored_nll(y: np.ndarray, is_censored: np.ndarray, mu: np.ndarray,
                 sigma: float, threshold: float) -> np.ndarray:
    """Per-row negative log-likelihood. ``y`` is ignored on censored rows.

    Censored rows contribute ``-log Phi((c - mu)/sigma)`` -- the probability that
    the latent value is at or below the threshold -- so whatever placeholder sits
    in ``y`` there (0, in this dataset) never enters the objective.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    is_censored = np.asarray(is_censored, dtype=bool)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma!r}")

    out = np.empty(len(mu), dtype=float)
    unc = ~is_censored
    resid = (y[unc] - mu[unc]) / sigma
    out[unc] = 0.5 * resid * resid + np.log(sigma) + _HALF_LOG_2PI
    z = (threshold - mu[is_censored]) / sigma
    out[is_censored] = -log_ndtr(z)
    return out


def censored_grads(y: np.ndarray, is_censored: np.ndarray, mu: np.ndarray,
                   sigma: float, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-row gradients of :func:`censored_nll` w.r.t. ``mu`` and ``log sigma``.

    Returned per row and un-normalised; the caller decides how to average.

        d/dmu     uncensored:  (mu - y) / sigma**2
                  censored:    lambda(z) / sigma,  z = (c - mu)/sigma

        d/dlog s  uncensored:  1 - ((y - mu)/sigma)**2
                  censored:    z * lambda(z)

    The censored ``d/dmu`` is always positive, so gradient descent pushes ``mu``
    *down* toward and past the threshold, and it decays to 0 once ``mu`` is well
    below it -- the Tobit property that squared error lacks.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    is_censored = np.asarray(is_censored, dtype=bool)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma!r}")

    g_mu = np.empty(len(mu), dtype=float)
    g_logs = np.empty(len(mu), dtype=float)

    unc = ~is_censored
    resid = (y[unc] - mu[unc]) / sigma
    g_mu[unc] = -resid / sigma
    g_logs[unc] = 1.0 - resid * resid

    z = (threshold - mu[is_censored]) / sigma
    lam = inverse_mills(z)
    g_mu[is_censored] = lam / sigma
    g_logs[is_censored] = z * lam
    return g_mu, g_logs


class CensoredMLPRegressor(MLPRegressor):
    """:class:`~pepstab.mlp.MLPRegressor` with the squared error swapped for a
    left-censored Gaussian likelihood.

    Everything structural is inherited. ``fit`` is re-implemented because the
    parent hard-codes the residual, but it reuses the parent's ``_init_params``,
    ``_forward``, ``_backward``, ``_snapshot`` and ``_Adam`` and walks the seeded
    RNG in the same order, so the two arms start from identical weights and see
    identical minibatches.

    The gradient is routed through the parent's ``_backward``, which computes
    ``delta = (2/n) * residual``. Passing ``residual = g_mu / 2`` therefore makes
    ``delta = g_mu / n`` -- the gradient of the batch-mean NLL. That reuse is
    deliberate: a re-derived backward pass would be a second difference between
    the arms.

    ``predict`` is inherited and returns ``mu`` on the ``log1p`` scale, the
    estimated **latent** half-life. That is not the recorded label: at the floor
    the recorded value is 0 while ``mu`` may legitimately sit below it. Use
    :meth:`predict_recorded_median` when a prediction of the *recorded* value is
    wanted.
    """

    #: Allowed values of ``stop_on``.
    STOPPING_RULES = ("nll", "mse")

    def __init__(self, config: MLPConfig | None = None,
                 floor_hours: float = DEFAULT_FLOOR_HOURS,
                 stop_on: str = "nll", **kwargs):
        super().__init__(config, **kwargs)
        if stop_on not in self.STOPPING_RULES:
            raise ValueError(f"stop_on must be one of {self.STOPPING_RULES}, "
                             f"got {stop_on!r}")
        self.floor_hours = float(floor_hours)
        self.stop_on = stop_on
        self.threshold_ = floor_threshold(floor_hours)
        self.sigma_: float = float("nan")
        self.sigma_curve_: list[float] = []
        self.sigma_clipped_: bool = False
        self.dev_nll_: float = float("nan")
        self.n_censored_fit_: int = 0

    # --- objective --------------------------------------------------------

    def _mean_nll(self, X: np.ndarray, y_centred: np.ndarray,
                  is_censored: np.ndarray, threshold_c: float, sigma: float) -> float:
        mu, _ = self._forward(X)
        return float(np.mean(censored_nll(y_centred, is_censored, mu, sigma,
                                          threshold_c)))

    # --- public API -------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_dev: np.ndarray, y_dev: np.ndarray,
            is_censored: np.ndarray | None = None,
            is_censored_dev: np.ndarray | None = None,
            sample_weight: np.ndarray | None = None) -> "CensoredMLPRegressor":
        """Fit by maximum censored likelihood, stopping on the dev fold's NLL.

        ``is_censored`` marks the floor rows among ``(X, y)``; ``is_censored_dev``
        does the same for the dev fold. Both are required -- defaulting them to
        "nothing is censored" would silently turn this into a Gaussian MLE with a
        learned scale, which is a different model that happens not to crash.

        Stopping on the dev **NLL** rather than the dev MSE is the consistent
        analogue of the parent stopping on dev MSE: each arm stops on its own
        training objective. Construct with ``stop_on="mse"`` to force the
        parent's rule instead; ``scripts/stage7_censored.py`` runs that variant
        too, so the choice is visible rather than assumed.

        ``sample_weight`` is rejected rather than ignored. The parent supports it
        for stage 2b's assumed-zero augmentation rows, and a weighted censored
        likelihood is a coherent thing to want, but stage 7a does not use one and
        silently dropping the argument would be a trap for a later caller.
        """
        if sample_weight is not None:
            raise NotImplementedError(
                "CensoredMLPRegressor does not support sample_weight; stage 7a "
                "fits measured rows only. Dropping it silently would make a "
                "weighted call quietly unweighted."
            )
        if is_censored is None or is_censored_dev is None:
            raise ValueError(
                "is_censored and is_censored_dev are required. Pass "
                "pepstab.censored.censored_mask(thalf_hours, floor_hours) for "
                "each fold; omitting them would fit an ordinary Gaussian MLE."
            )

        cfg = self.config
        X = np.asarray(X, dtype=DTYPE)
        y = np.asarray(y, dtype=np.float64)
        X_dev = np.asarray(X_dev, dtype=DTYPE)
        y_dev = np.asarray(y_dev, dtype=np.float64)
        cens = np.asarray(is_censored, dtype=bool)
        cens_dev = np.asarray(is_censored_dev, dtype=bool)

        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
        if cens.shape != (len(X),):
            raise ValueError(f"is_censored has shape {cens.shape}, expected ({len(X)},)")
        if cens_dev.shape != (len(X_dev),):
            raise ValueError(f"is_censored_dev has shape {cens_dev.shape}, "
                             f"expected ({len(X_dev)},)")
        if not len(X_dev):
            raise ValueError("the dev fold is empty; early stopping needs rows")
        if cens.all():
            raise ValueError("every fit row is censored; the scale is unidentified")

        self.n_censored_fit_ = int(cens.sum())

        # Same RNG, same order of draws as the parent: initialisation first, then
        # one permutation per epoch. This is what makes the two arms comparable
        # at a fixed seed.
        rng = np.random.default_rng(cfg.seed)
        # Centre on the recorded labels exactly as the parent does, so both arms
        # start from the same output bias. Not a fitted parameter.
        self.y_offset_ = float(y.mean())
        y_centred = y - self.y_offset_
        threshold_c = self.threshold_ - self.y_offset_
        self._init_params(X.shape[1], rng)

        from .mlp import _Adam  # local: keeps the parent's optimiser, not a copy

        opt_w = [_Adam.for_param(W) for W in self.weights_]
        opt_b = [_Adam.for_param(b) for b in self.biases_]

        # Start the scale at the spread of the recorded labels. Any positive
        # start works -- it is a fitted parameter -- but this one is already the
        # right order of magnitude, so no epochs are spent travelling.
        log_sigma = np.array([np.log(max(float(y_centred.std()), 1e-2))], dtype=np.float64)
        opt_s = _Adam.for_param(log_sigma)

        started = time.perf_counter()
        best_criterion, best_nll, best_epoch = np.inf, np.inf, 0
        best_state, best_sigma = self._snapshot(), float(np.exp(log_sigma[0]))
        self.dev_curve_, self.sigma_curve_ = [], []
        self.sigma_clipped_ = False
        step = 0
        for epoch in range(1, cfg.max_epochs + 1):
            order = rng.permutation(len(X))
            for start in range(0, len(order), cfg.batch_size):
                idx = order[start:start + cfg.batch_size]
                sigma = float(np.exp(log_sigma[0]))
                mu, acts = self._forward(X[idx])
                g_mu, g_logs = censored_grads(y_centred[idx], cens[idx], mu,
                                              sigma, threshold_c)
                # _backward multiplies by 2/n, so halving here leaves the
                # gradient of the batch-mean NLL. See the class docstring.
                gw, gb = self._backward(acts, (g_mu * 0.5).astype(DTYPE))
                step += 1
                for k in range(len(self.weights_)):
                    opt_w[k].step(self.weights_[k], gw[k] + cfg.l2 * self.weights_[k],
                                  cfg.lr, step)
                    opt_b[k].step(self.biases_[k], gb[k], cfg.lr, step)
                opt_s.step(log_sigma, np.array([g_logs.mean()]), cfg.lr, step)
                lo, hi = LOG_SIGMA_BOUNDS
                if log_sigma[0] < lo or log_sigma[0] > hi:
                    self.sigma_clipped_ = True
                    log_sigma[0] = min(max(log_sigma[0], lo), hi)

            sigma = float(np.exp(log_sigma[0]))
            dev_nll = self._mean_nll(X_dev, y_dev - self.y_offset_, cens_dev,
                                     threshold_c, sigma)
            self.dev_curve_.append(dev_nll)
            self.sigma_curve_.append(sigma)
            # "nll" is the consistent analogue of the parent's rule; "mse" is
            # the control that shows the choice is not carrying the result.
            criterion = (dev_nll if self.stop_on == "nll"
                         else float(np.mean((self.predict(X_dev) - y_dev) ** 2)))
            if criterion < best_criterion:
                best_criterion, best_nll, best_epoch = criterion, dev_nll, epoch
                best_state, best_sigma = self._snapshot(), sigma
            elif epoch - best_epoch >= cfg.patience:
                break

        self.weights_, self.biases_ = best_state
        self.sigma_ = best_sigma
        self.n_epochs_ = epoch
        self.best_epoch_ = best_epoch
        self.dev_nll_ = best_nll
        # Always reported so both arms share the field; it is only what this arm
        # *stopped on* when stop_on == "mse".
        self.dev_mse_ = float(np.mean((self.predict(X_dev) - y_dev) ** 2))
        self.fit_seconds_ = time.perf_counter() - started
        return self

    # --- derived predictions ---------------------------------------------

    def prob_censored(self, X: np.ndarray) -> np.ndarray:
        """``P(latent <= threshold)`` per row, under the fitted Gaussian.

        The quantity the MSE arm cannot produce at all: an explicit probability
        that a pair sits at the detection floor.
        """
        if not np.isfinite(self.sigma_):
            raise RuntimeError("prob_censored() before fit()")
        return ndtr((self.threshold_ - self.predict(X)) / self.sigma_)

    def predict_recorded_median(self, X: np.ndarray) -> np.ndarray:
        """Median of the *recorded* label under the fitted model.

        The recorded value is 0 when the latent value falls below the floor and
        the latent value otherwise, so its median is 0 exactly when the model
        gives the floor at least even odds -- that is, when ``mu <= threshold``.
        This is the MAE-optimal point prediction of the recorded label, and it is
        the honest way to read a sub-floor ``mu``: not "0.03 hours" but "below
        what the assay can see".
        """
        mu = self.predict(X)
        return np.where(mu <= self.threshold_, 0.0, mu)
