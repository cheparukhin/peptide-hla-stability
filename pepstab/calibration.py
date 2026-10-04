"""Probability calibration for the assay-floor event.

Phase B. 20.2% of ``thalf_hours`` labels are floor codes (recorded 0), and stage
7a showed that a censored likelihood changes *where a model puts its predictive
mass* far more than it changes ranking. Judging that properly needs machinery the
project did not have: a proper score for the floor event with a **paired
interval**, a binning rule that cannot be gamed, and a reference calibrator built
from the point regressor we already trust.

This module supplies those three things.

**Everything here is oriented so that a positive delta means "better".** Brier,
log loss, ECE and censored NLL are all lower-is-better, so
:func:`paired_cluster_bootstrap_metric` returns ``metric(reference) -
metric(arm)``. That is not cosmetic: it lets
``pepstab.evaluation.describe_delta``'s six predeclared verdicts be read off a
calibration interval with no sign juggling, which is exactly the place where a
sign slip would invert a conclusion.

**Two things that look interchangeable and are not.**

1. *Which proper score.* Brier is strictly proper, so it cannot be improved by
   moving confidence around without also being more right. ECE is **not** a
   proper score -- a model that predicts the base rate for every row has an ECE
   near zero and no discrimination at all. ECE is therefore a diagnostic here,
   never a primary objective, and :func:`fixed_bin_ece` reports bin occupancy so
   a reader can see when a bin's "observed rate" rests on nine rows.
2. *How an ensemble's floor probability is formed.* For members with their own
   scales, ``mean_k Phi((c - mu_k)/sigma_k)`` (:func:`prob_floor_mixture`) is the
   predictive probability of the mixture the ensemble actually is, while
   ``Phi((c - mean_k mu_k)/sigma_bar)`` (:func:`prob_floor_plugin`) is a Gaussian
   approximation that discards between-member spread and is therefore more
   extreme in the tails. Stage 7a used the mixture for its censored arm and the
   plug-in for its MSE arm, which is why its reliability table and its reported
   ECE disagree (0.0474 against 0.0423 on the same predictions). Pick one rule
   and apply it to every arm; :data:`P_FLOOR_RULES` names the choices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import log_ndtr, ndtr

from .censored import DEFAULT_FLOOR_HOURS, censored_mask, floor_threshold
from .evaluation import BOOTSTRAP_SEED, N_BOOTSTRAP, validate_finite

#: How an ensemble's ``P(floor)`` is formed from its members. Declared per run,
#: never mixed across arms within one comparison -- see the module docstring.
P_FLOOR_RULES = ("mixture", "plugin")

#: Number of bins for the fixed-width ECE. Fixed *width*, not equal count: the
#: bins are the same for every arm, so a difference in ECE cannot come from two
#: arms being binned differently.
ECE_BINS = 10

_HALF_LOG_2PI = 0.5 * float(np.log(2.0 * np.pi))


# --------------------------------------------------------------------------
# the event


#: How the floor event is defined for calibration. ``"recorded_zero"`` is the
#: Phase B declaration; ``"below_limit"`` is the loss's censoring rule, kept
#: available for the one-row sensitivity check -- see :func:`floor_indicator`.
FLOOR_EVENT_RULES = ("recorded_zero", "below_limit")


def floor_indicator(thalf_hours, rule: str = "recorded_zero",
                    floor_hours: float = DEFAULT_FLOOR_HOURS) -> np.ndarray:
    """Which rows are at the assay floor, for the purpose of a *probability*.

    Two rules, and they are not the same event:

    - ``"recorded_zero"`` -- ``thalf_hours == 0``, the recorded floor code.
      **The Phase B declaration.** 5,679 rows, 20.1626% of the dataset, which is
      exactly the "20.2% at the assay floor" EVALUATION.md names. A probability
      target should be an observable, and the recorded 0 is one: it needs no assay
      constant, so no part of the calibration metric depends on where the
      detection limit is believed to be.
    - ``"below_limit"`` -- ``thalf_hours < floor_hours``, i.e.
      :func:`pepstab.censored.censored_mask`, the rule the *loss* censors on.

    The two differ on **exactly one row** of 28,166 — ``pair_id`` 1189,
    HLA-A*24:02 / WYGMEIRPL, measured at 0.05 h, in the training split — and
    nowhere on validation or test. That single row is why both rules are kept
    here rather than one being quietly assumed: Phase B scores the event on
    ``recorded_zero`` while its censored arms censor on ``below_limit``, so the
    loss and the metric genuinely disagree about that row, and the honest
    response is a sensitivity check rather than silently redefining one to match
    the other. Redefining the *loss* instead would be worse: it would make these
    arms non-comparable to the stage 7a result Phase B exists to resolve.

    Note what is *not* a reason to prefer one rule: there is no floating-point
    hazard at the 0.1 h boundary. ``censored_mask`` compares hours, the stored
    0.1 is the exact double ``0.1``, and ``0.1 < 0.1`` is deterministically
    ``False``; the 443 rows recorded at exactly 0.1 h are measurements under
    either rule, as stage 7a §3 declares.
    """
    if rule == "below_limit":
        return censored_mask(thalf_hours, floor_hours)
    if rule != "recorded_zero":
        raise ValueError(f"rule must be one of {FLOOR_EVENT_RULES}, got {rule!r}")
    hours = np.asarray(thalf_hours, dtype=float)
    if not np.all(np.isfinite(hours)):
        raise ValueError("thalf_hours contains non-finite values")
    return hours == 0.0


# --------------------------------------------------------------------------
# scores for the floor event


def _check_scores(p: np.ndarray, y: np.ndarray,
                  what: str = "floor probabilities") -> tuple[np.ndarray, np.ndarray]:
    """Shared validation for a score and a boolean floor label. No range check.

    Split from :func:`_check_probs` because AUROC is **scale-free**: it is a
    property of the ordering, so it is legitimately computed from a raw ``-mu``
    as well as from a probability, and requiring ``[0, 1]`` there would reject the
    discrimination number stage 7a actually reported.
    """
    p = validate_finite(np.asarray(p, dtype=float).ravel(), what)
    y = np.asarray(y).ravel()
    if y.dtype != bool:
        if not np.all(np.isin(y, (0, 1))):
            raise ValueError("floor labels must be boolean or 0/1")
        y = y.astype(bool)
    if len(p) != len(y):
        raise ValueError(f"length mismatch: {len(p)} scores, {len(y)} labels")
    if not len(p):
        raise ValueError("no rows to score")
    return p, y


def _check_probs(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """As :func:`_check_scores`, and additionally require a probability.

    Brier, log loss and ECE are only meaningful on ``[0, 1]``; a stray ``mu``
    passed where a probability was expected would otherwise produce a number that
    looks like a score and is not one.
    """
    p, y = _check_scores(p, y)
    if p.min() < 0.0 or p.max() > 1.0:
        raise ValueError(f"probabilities outside [0, 1]: [{p.min()}, {p.max()}]")
    return p, y


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of ``P(floor)`` against the floor indicator. Lower is better.

    The primary Phase B objective. Strictly proper, so the only way to improve it
    is to be better calibrated *and* no worse at discrimination -- unlike ECE,
    which a base-rate-constant predictor can win outright.
    """
    p, y = _check_probs(p, y)
    return float(np.mean((p - y.astype(float)) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood of the floor indicator. Lower is better.

    Also strictly proper, reported alongside Brier because the two disagree about
    how much a confident mistake should cost: log loss is unbounded, so one row
    predicted at ``p = 1e-12`` when it is in fact at the floor dominates it. That
    sensitivity is the reason Brier, not log loss, is primary.
    """
    p, y = _check_probs(p, y)
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(np.where(y, np.log(p), np.log1p(-p))))


def fixed_bin_ece(y: np.ndarray, p: np.ndarray,
                  n_bins: int = ECE_BINS) -> tuple[float, pd.DataFrame]:
    """Expected calibration error over **fixed-width** bins of ``[0, 1]``.

    Returns ``(ece, table)``. The ECE is the occupancy-weighted mean gap between
    predicted and observed floor rate, over bins that are the same for every arm.

    Stage 7a used equal-*count* bins, which is a different quantity: the bin
    edges then depend on the arm's own predictions, so two arms are binned
    differently and part of any difference is the binning. Fixed width removes
    that. The price is empty and near-empty bins when an arm concentrates its
    predictions, which is why ``n`` is returned and never hidden --
    an "observed rate" over nine rows is not evidence of miscalibration.

    Empty bins contribute nothing (weight zero), not a zero gap.
    """
    p, y = _check_probs(p, y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # ``right=True`` with a clamp keeps p == 0 in bin 0 and p == 1 in the last
    # bin, instead of sending the endpoints outside the range.
    idx = np.clip(np.searchsorted(edges, p, side="left") - 1, 0, n_bins - 1)
    rows = []
    total = 0.0
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        if n:
            mean_p = float(p[sel].mean())
            obs = float(y[sel].mean())
            total += n * abs(mean_p - obs)
        else:
            mean_p = float("nan")
            obs = float("nan")
        lo, hi = _wilson(y[sel].sum(), n)
        rows.append({"bin": b + 1, "edge_low": edges[b], "edge_high": edges[b + 1],
                     "n": n, "share": n / len(p),
                     "mean_predicted_p_floor": mean_p, "observed_floor_share": obs,
                     "observed_ci_low": lo, "observed_ci_high": hi})
    return float(total / len(p)), pd.DataFrame(rows)


def equal_count_ece(y: np.ndarray, p: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """Stage 7a's ECE: equal-count bins of the prediction. Reported for continuity only.

    Kept so Phase B's numbers can be placed next to
    ``reports/stage7_censored.md`` without the reader having to trust that two
    differently-defined quantities are comparable. Not the Phase B diagnostic --
    see :func:`fixed_bin_ece` for why.
    """
    p, y = _check_probs(p, y)
    order = np.argsort(p, kind="stable")
    gap = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            gap += len(chunk) * abs(p[chunk].mean() - y[chunk].mean())
    return float(gap / len(p))


def _wilson(successes: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial rate. ``(nan, nan)`` for an empty bin.

    Wilson rather than normal-approximation because reliability bins routinely
    sit near 0 or 1, where a Wald interval runs outside ``[0, 1]`` and would
    suggest a bin is miscalibrated when it is merely small.
    """
    if n <= 0:
        return float("nan"), float("nan")
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = ECE_BINS,
                      binning: str = "fixed", arm: str | None = None) -> pd.DataFrame:
    """Predicted against observed floor rate per bin, with Wilson intervals.

    ``binning="fixed"`` gives the Phase B definition (same edges for every arm);
    ``"equal_count"`` reproduces stage 7a's deciles.
    """
    if binning == "fixed":
        _, table = fixed_bin_ece(y, p, n_bins)
    elif binning == "equal_count":
        p, y = _check_probs(p, y)
        order = np.argsort(p, kind="stable")
        rows = []
        for i, chunk in enumerate(np.array_split(order, n_bins)):
            lo, hi = _wilson(y[chunk].sum(), len(chunk))
            rows.append({"bin": i + 1, "edge_low": float("nan"),
                         "edge_high": float("nan"), "n": len(chunk),
                         "share": len(chunk) / len(p),
                         "mean_predicted_p_floor": float(p[chunk].mean()),
                         "observed_floor_share": float(y[chunk].mean()),
                         "observed_ci_low": lo, "observed_ci_high": hi})
        table = pd.DataFrame(rows)
    else:
        raise ValueError(f"binning must be 'fixed' or 'equal_count', got {binning!r}")
    table.insert(0, "binning", binning)
    if arm is not None:
        table.insert(0, "arm", arm)
    return table


def floor_auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Probability that a floor row is scored above a measured row. Ties get half credit.

    *Discrimination*, which is the part of floor behaviour that is identifiable:
    ranking rows *within* the tied floor block is not, because those labels carry
    no order. Stage 7a found this flat (0.886 against 0.885), so a calibration
    gain with unchanged AUROC means the model moved mass, not order.

    Rank-based rather than via ``mannwhitneyu`` so it is cheap enough to live
    inside a 2,000-resample bootstrap. ``p`` may be any monotone score, not only a
    probability -- ``-mu`` is a valid input and is what stage 7a reported.
    """
    p, y = _check_scores(p, y, "floor scores")
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    ranks = _average_ranks(p)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged -- the tie rule the frozen contract uses.

    ``scipy.stats.rankdata`` rather than a hand-rolled tie scan: it is the same
    numbers (asserted in ``test_average_ranks_matches_scipy_including_a_tie_block``)
    and about 6x faster on a 19,716-row fold, which matters because this runs
    inside a 2,000-resample bootstrap -- the hand-rolled version cost roughly 8
    minutes per batch of AUROC bootstraps against 1.3 for this one. The floor
    block makes large ties routine, so the tie path is the common case, not an
    edge case.
    """
    return np.asarray(stats.rankdata(x, method="average"), dtype=float)


def censored_nll_mean(mu: np.ndarray, y_log1p: np.ndarray, is_floor: np.ndarray,
                      sigma: float, threshold: float) -> float:
    """Mean left-censored Gaussian NLL on held-out rows. Lower is better.

    Secondary, and reported *because* it disagreed with Brier and ECE at stage 7a
    rather than in spite of it. The disagreement is structural, not a bug: this
    number sums a Gaussian **density** term over measured rows with a
    **probability** term over floor rows, so an arm that sharpens its floor
    probabilities by shrinking ``sigma`` is charged for the sharper density on
    every measured row. Brier scores only the floor event and is not subject to
    that trade.
    """
    mu = validate_finite(np.asarray(mu, dtype=float).ravel(), "mu")
    y_log1p = np.asarray(y_log1p, dtype=float).ravel()
    is_floor = np.asarray(is_floor, dtype=bool).ravel()
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma!r}")
    z = (threshold - mu) / sigma
    out = np.where(is_floor,
                   -log_ndtr(z),
                   0.5 * ((y_log1p - mu) / sigma) ** 2 + np.log(sigma) + _HALF_LOG_2PI)
    return float(np.mean(out))


# --------------------------------------------------------------------------
# forming P(floor) from an ensemble


def prob_floor_mixture(mus: np.ndarray, sigmas: np.ndarray,
                       threshold: float) -> np.ndarray:
    """``mean_k Phi((c - mu_k) / sigma_k)`` over ensemble members.

    The predictive floor probability of the mixture the ensemble actually is.
    Averaging probabilities, not means, so between-member disagreement shows up
    as a probability nearer 0.5 rather than being thrown away.

    ``mus`` is ``(n_members, n_rows)``; ``sigmas`` is one scale per member.
    """
    mus = np.atleast_2d(np.asarray(mus, dtype=float))
    sigmas = np.asarray(sigmas, dtype=float).ravel()
    if len(sigmas) != len(mus):
        raise ValueError(f"{len(mus)} members but {len(sigmas)} scales")
    if np.any(sigmas <= 0):
        raise ValueError("every member scale must be positive")
    return np.asarray(ndtr((threshold - mus) / sigmas[:, None]).mean(axis=0), dtype=float)


def prob_floor_plugin(mu: np.ndarray, sigma: float, threshold: float) -> np.ndarray:
    """``Phi((c - mu) / sigma)`` from a single mean prediction and one scale.

    A Gaussian approximation to :func:`prob_floor_mixture` that ignores
    between-member spread, so it is **more extreme in both tails**. The only rule
    available to an arm with no per-member scale, which is why a post-hoc
    calibrated MSE arm uses it -- and why an arm that has per-member scales must
    not, if its comparator cannot.
    """
    mu = validate_finite(np.asarray(mu, dtype=float).ravel(), "mu")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma!r}")
    return np.asarray(ndtr((threshold - mu) / sigma), dtype=float)


# --------------------------------------------------------------------------
# post-hoc calibrators -- fitted on inner out-of-fold rows, never on an outer label


@dataclass
class GaussianScaleFloorCalibrator:
    """``P(floor) = Phi((c - mu)/sigma)`` with ``sigma`` an inner-dev residual RMSE.

    The cheapest honest way to give a plain MSE regressor a floor probability,
    and the construction stage 7a used for its comparator. ``sigma`` comes from
    held-out inner-fold residuals, so it is training-only -- but note what it
    assumes: one global scale, Gaussian, and residual spread read off *measured*
    rows standing in for spread at the floor.
    """

    floor_hours: float = DEFAULT_FLOOR_HOURS
    sigma_: float = float("nan")
    n_fit_: int = 0

    def fit(self, mu: np.ndarray, y_log1p: np.ndarray,
            is_floor: np.ndarray | None = None) -> "GaussianScaleFloorCalibrator":
        """``sigma`` from the RMSE of held-out residuals.

        Floor rows are excluded when ``is_floor`` is given: their recorded label
        is a detection limit, so squaring ``(0 - mu)`` there would measure the
        floor coding, not the model's spread.
        """
        mu = np.asarray(mu, dtype=float).ravel()
        y_log1p = np.asarray(y_log1p, dtype=float).ravel()
        keep = np.ones(len(mu), dtype=bool) if is_floor is None else ~np.asarray(is_floor, bool)
        if keep.sum() < 2:
            raise ValueError("need at least two measured rows to estimate a scale")
        self.sigma_ = float(np.sqrt(np.mean((y_log1p[keep] - mu[keep]) ** 2)))
        self.n_fit_ = int(keep.sum())
        return self

    def predict_proba(self, mu: np.ndarray) -> np.ndarray:
        if not np.isfinite(self.sigma_):
            raise RuntimeError("predict_proba() before fit()")
        return prob_floor_plugin(mu, self.sigma_, floor_threshold(self.floor_hours))


@dataclass
class PlattFloorCalibrator:
    """Logistic regression of the floor indicator on ``mu``. The declared baseline.

    One slope and one intercept, fitted on inner out-of-fold predictions inside
    the training partition. This is the simpler solution a distributional model
    has to beat: it needs no censored likelihood, no second head and no change to
    the deployed regressor, and it is free to learn that the mapping from ``mu``
    to floor probability is steeper or shallower than a Gaussian tail implies --
    which :class:`GaussianScaleFloorCalibrator` cannot.

    Two parameters is the point. Isotonic regression
    (:class:`IsotonicFloorCalibrator`) fits a monotone step function instead and
    can chase inner-fold noise, so it is the secondary of the pair.
    """

    sigma_unused: float = field(default=float("nan"), repr=False)
    coef_: float = float("nan")
    intercept_: float = float("nan")
    n_fit_: int = 0

    def fit(self, mu: np.ndarray, is_floor: np.ndarray) -> "PlattFloorCalibrator":
        from sklearn.linear_model import LogisticRegression

        mu = np.asarray(mu, dtype=float).reshape(-1, 1)
        y = np.asarray(is_floor, dtype=bool).ravel()
        if len(np.unique(y)) < 2:
            raise ValueError("fit rows contain only one class; cannot calibrate")
        # Effectively unpenalised: two parameters on thousands of rows, and a
        # penalty would shrink the slope toward a flat base-rate predictor, which
        # is exactly the failure mode a non-proper score would reward. ``C=inf``
        # rather than ``penalty=None``, which sklearn 1.8 deprecated.
        lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
        lr.fit(mu, y)
        self.coef_ = float(lr.coef_[0, 0])
        self.intercept_ = float(lr.intercept_[0])
        self.n_fit_ = len(y)
        return self

    def predict_proba(self, mu: np.ndarray) -> np.ndarray:
        if not np.isfinite(self.coef_):
            raise RuntimeError("predict_proba() before fit()")
        z = self.intercept_ + self.coef_ * np.asarray(mu, dtype=float).ravel()
        # Stable logistic, branch by sign. ``np.where`` would not do: it evaluates
        # *both* arms before selecting, so the negative branch's ``exp(-z)`` still
        # overflows on large positive z and warns even though the result is
        # discarded.
        out = np.empty_like(z)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out


@dataclass
class IsotonicFloorCalibrator:
    """Monotone (isotonic) map from ``-mu`` to ``P(floor)``. Secondary baseline.

    Monotone in ``-mu`` because a *lower* latent half-life means a *higher*
    chance of sitting at the floor; fitting on ``mu`` directly would impose the
    wrong direction. More flexible than Platt and correspondingly easier to
    overfit on an inner fold, so it is reported as the second point of the
    post-hoc pair rather than as the baseline.
    """

    clip_: tuple[float, float] = (0.0, 1.0)
    n_fit_: int = 0
    _iso: object = field(default=None, repr=False)

    def fit(self, mu: np.ndarray, is_floor: np.ndarray) -> "IsotonicFloorCalibrator":
        from sklearn.isotonic import IsotonicRegression

        y = np.asarray(is_floor, dtype=bool).ravel().astype(float)
        if len(np.unique(y)) < 2:
            raise ValueError("fit rows contain only one class; cannot calibrate")
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True,
                                 out_of_bounds="clip")
        iso.fit(-np.asarray(mu, dtype=float).ravel(), y)
        self._iso = iso
        self.n_fit_ = len(y)
        return self

    def predict_proba(self, mu: np.ndarray) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError("predict_proba() before fit()")
        return np.clip(self._iso.predict(-np.asarray(mu, dtype=float).ravel()), 0.0, 1.0)


# --------------------------------------------------------------------------
# paired intervals


def paired_cluster_bootstrap_metric(cluster_id, metric, *, reference: np.ndarray,
                                    arm: np.ndarray, labels: np.ndarray,
                                    lower_is_better: bool = True,
                                    n_boot: int = N_BOOTSTRAP,
                                    seed: int = BOOTSTRAP_SEED) -> dict:
    """Paired peptide-cluster bootstrap interval for any row-set metric.

    Same resampling contract as
    :func:`pepstab.evaluation.paired_cluster_bootstrap`: whole peptide clusters
    with replacement, both arms scored on the *same* resample, 2,000 resamples,
    seed 20261003. Clusters are the independence unit because one peptide appears
    on up to 36 alleles, so a row bootstrap would understate the spread.

    ``metric(labels, predictions) -> float``. The metric is recomputed on each
    resample rather than averaged from per-row contributions, because ECE and
    AUROC are not row-decomposable -- an ECE bootstrapped from fixed per-row
    terms would not be an ECE.

    **The returned delta is always an improvement**, i.e. ``reference - arm``
    when ``lower_is_better``. That makes the interval directly readable by
    :func:`pepstab.evaluation.describe_delta`.
    """
    cluster_id = np.asarray(cluster_id).ravel()
    reference = np.asarray(reference, dtype=float).ravel()
    arm = np.asarray(arm, dtype=float).ravel()
    labels = np.asarray(labels).ravel()
    n = len(labels)
    if not (len(reference) == len(arm) == n):
        raise ValueError(f"length mismatch: {n} labels, {len(reference)} reference, "
                         f"{len(arm)} arm")
    if len(cluster_id) != n:
        raise ValueError(f"{len(cluster_id)} cluster ids for {n} rows")

    sign = 1.0 if lower_is_better else -1.0

    def improvement(idx: np.ndarray) -> float:
        return sign * (metric(labels[idx], reference[idx]) - metric(labels[idx], arm[idx]))

    full = np.arange(n)
    observed = improvement(full)

    unique = np.unique(cluster_id)
    rows_by_cluster = {c: np.nonzero(cluster_id == c)[0] for c in unique}
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        deltas[i] = improvement(np.concatenate([rows_by_cluster[c] for c in picked]))

    lo, hi = (float(v) for v in np.nanpercentile(deltas, [2.5, 97.5]))
    return {
        "metric": getattr(metric, "__name__", str(metric)),
        "lower_is_better": bool(lower_is_better),
        "reference": float(metric(labels, reference)),
        "arm": float(metric(labels, arm)),
        # Named "improvement" and not "delta" on purpose: a reader who sees
        # "delta_brier = +0.007" on a lower-is-better score has to guess the sign
        # convention, and guessing wrong inverts the conclusion.
        "delta_improvement": float(observed),
        "ci95": (lo, hi),
        "n_boot": int(n_boot),
        "n_rows": int(n),
        "n_clusters": int(len(unique)),
        "n_degenerate_resamples": int(np.isnan(deltas).sum()),
        "seed": int(seed),
        "conclusive": bool(lo > 0 or hi < 0),
        "bootstrap_half_width": float(0.5 * (hi - lo)),
    }


def paired_cluster_bootstrap_censored_nll(cluster_id, *, y_log1p: np.ndarray,
                                          is_censored: np.ndarray,
                                          mu_reference: np.ndarray, sigma_reference: float,
                                          mu_arm: np.ndarray, sigma_arm: float,
                                          threshold: float,
                                          n_boot: int = N_BOOTSTRAP,
                                          seed: int = BOOTSTRAP_SEED) -> dict:
    """Paired interval for censored NLL, where each arm carries its **own** scale.

    Separate from :func:`paired_cluster_bootstrap_metric` because this metric is
    not a function of ``(labels, predictions)`` alone: it needs the labels, the
    censoring mask, the mean *and* that arm's fitted ``sigma``. Forcing it through
    the generic interface would mean scoring both arms at one shared scale, which
    is not the comparison -- an arm's scale is part of what its objective chose,
    and stage 7a's disagreement between NLL and Brier is driven by exactly that
    (sigma 0.75 against 0.58).

    Returned delta is improvement-signed, i.e. ``reference - arm``.
    """
    cluster_id = np.asarray(cluster_id).ravel()
    y_log1p = np.asarray(y_log1p, dtype=float).ravel()
    is_censored = np.asarray(is_censored, dtype=bool).ravel()
    mu_reference = validate_finite(np.asarray(mu_reference, float).ravel(), "reference mu")
    mu_arm = validate_finite(np.asarray(mu_arm, float).ravel(), "arm mu")
    n = len(y_log1p)
    if not (len(mu_reference) == len(mu_arm) == len(is_censored) == len(cluster_id) == n):
        raise ValueError("every input must have one entry per row")

    def improvement(idx: np.ndarray) -> float:
        return (censored_nll_mean(mu_reference[idx], y_log1p[idx], is_censored[idx],
                                  sigma_reference, threshold)
                - censored_nll_mean(mu_arm[idx], y_log1p[idx], is_censored[idx],
                                    sigma_arm, threshold))

    full = np.arange(n)
    observed = improvement(full)
    unique = np.unique(cluster_id)
    rows_by_cluster = {c: np.nonzero(cluster_id == c)[0] for c in unique}
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        deltas[i] = improvement(np.concatenate([rows_by_cluster[c] for c in picked]))

    lo, hi = (float(v) for v in np.nanpercentile(deltas, [2.5, 97.5]))
    return {
        "metric": "censored_nll",
        "lower_is_better": True,
        "reference": censored_nll_mean(mu_reference, y_log1p, is_censored,
                                       sigma_reference, threshold),
        "arm": censored_nll_mean(mu_arm, y_log1p, is_censored, sigma_arm, threshold),
        "sigma_reference": float(sigma_reference),
        "sigma_arm": float(sigma_arm),
        "delta_improvement": float(observed),
        "ci95": (lo, hi),
        "n_boot": int(n_boot),
        "n_rows": int(n),
        "n_clusters": int(len(unique)),
        "seed": int(seed),
        "conclusive": bool(lo > 0 or hi < 0),
        "bootstrap_half_width": float(0.5 * (hi - lo)),
    }


def null_half_width(cluster_id, metric, *, predictions: np.ndarray,
                    labels: np.ndarray, lower_is_better: bool = True,
                    n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED,
                    n_splits: int = 8) -> dict:
    """Paired bootstrap half-width under **no true difference**, for setting a bar.

    EVALUATION.md's 0.05 ranking bar was derived this way: resample a comparison
    that has no real effect and see how wide the interval is anyway. There is no
    published analogue for Brier, so Phase B derives one by the same route, and
    this function is that derivation.

    The null comparison is one arm against itself with the rows split in half by
    cluster parity and the halves swapped -- a permutation that leaves the
    marginal distribution of predictions intact while destroying any systematic
    difference. Repeated over ``n_splits`` offsets so the answer does not rest on
    one arbitrary partition; the reported half-width is the **maximum** across
    them, which is the conservative choice for a bar.

    This is a noise-width measurement, not a comparison, so it is legitimate to
    run before the arms exist -- and it must be, because a bar chosen after
    seeing a result is not a bar.
    """
    cluster_id = np.asarray(cluster_id).ravel()
    predictions = np.asarray(predictions, dtype=float).ravel()
    labels = np.asarray(labels).ravel()
    unique = np.unique(cluster_id)
    rng = np.random.default_rng(seed)
    widths, results = [], []
    for s in range(n_splits):
        # A different random half of the clusters each time; within those rows the
        # "arm" is the reference with its predictions shuffled *within cluster*,
        # which keeps the cluster-level signal and removes the row-level one.
        half = set(rng.choice(unique, size=max(1, len(unique) // 2), replace=False).tolist())
        pseudo = predictions.copy()
        for c in half:
            idx = np.nonzero(cluster_id == c)[0]
            pseudo[idx] = predictions[rng.permutation(idx)]
        res = paired_cluster_bootstrap_metric(
            cluster_id, metric, reference=predictions, arm=pseudo, labels=labels,
            lower_is_better=lower_is_better, n_boot=n_boot, seed=seed + s)
        widths.append(res["bootstrap_half_width"])
        results.append({"split": s, "half_width": res["bootstrap_half_width"],
                        "delta_improvement": res["delta_improvement"],
                        "ci_low": res["ci95"][0], "ci_high": res["ci95"][1]})
    return {
        "metric": getattr(metric, "__name__", str(metric)),
        "n_splits": int(n_splits),
        "half_widths": widths,
        "max_half_width": float(np.max(widths)),
        "median_half_width": float(np.median(widths)),
        "per_split": results,
    }
