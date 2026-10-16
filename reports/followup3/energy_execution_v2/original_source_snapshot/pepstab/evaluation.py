"""The shared evaluation contract. Every model is scored through this module.

Predeclared at stage 1, before any model exists, so that nothing here can be
chosen after seeing results. See EVALUATION.md for the rules and the minimum
worthwhile gain.

Primary metric: median per-allele Spearman correlation between predicted and
measured half-life, over a set of eligible alleles shared by all models.
Secondary: MAE on log1p half-life, precision@10 at a 2-hour threshold, and
pooled correlations.

Uncertainty always comes from resampling *peptide clusters*, not rows. Rows
sharing a peptide cluster are not independent -- the same peptide appears on
many alleles -- so a row bootstrap would understate the interval.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

#: Rows an allele needs before it enters the per-allele summary, per split.
#:
#: The test split carries 20% of the data (median 72 rows per allele), so the
#: final claim rests on alleles with >= 50 rows. Validation carries 10% (median
#: 35), where a 50-row bar would leave only 10 eligible alleles covering a
#: quarter of the split -- far too thin to select models on. At 20 rows both
#: splits yield 68 eligible alleles covering >99% of their rows, so selection
#: and reporting look at effectively the same alleles while the final number
#: stays on the better-powered ones.
MIN_ROWS_BY_SPLIT = {"train": 50, "val": 20, "test": 50}

#: Fallback when the split is not stated.
MIN_ROWS_PER_ALLELE = 50

#: "Sufficiently stable" for precision@k, in raw hours.
STABILITY_THRESHOLD_HOURS = 2.0
PRECISION_K = 10

#: The smallest improvement in median per-allele Spearman we will call
#: meaningful. Predeclared; see EVALUATION.md for the derivation.
MIN_WORTHWHILE_DELTA_SPEARMAN = 0.05

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261003


def eligible_alleles(allele: pd.Series, y_true: np.ndarray,
                     min_rows: int | None = None,
                     split: str | None = None) -> list[str]:
    """Alleles with enough held-out rows and enough label spread to rank.

    Computed from the split and the labels alone -- never from predictions -- so
    the eligible set is identical for every model. Pass ``split`` to pick up the
    per-split row threshold from :data:`MIN_ROWS_BY_SPLIT`.
    """
    if min_rows is None:
        min_rows = MIN_ROWS_BY_SPLIT.get(split or "", MIN_ROWS_PER_ALLELE)
    frame = pd.DataFrame({"allele": np.asarray(allele), "y": np.asarray(y_true)})
    grouped = frame.groupby("allele")["y"]
    keep = (grouped.size() >= min_rows) & (grouped.nunique() >= 2)
    return sorted(keep.index[keep].tolist())


#: What an allele contributes to the panel median when the model's predictions
#: are constant within it.
#:
#: **Spearman is mathematically undefined for a constant input** -- its
#: denominator is zero -- and this constant does not pretend otherwise. It is a
#: *scoring rule* we choose, declared here before any model exists:
#:
#:     a constant prediction carries no ranking information, which is worth 0.
#:
#: 0 is the right number because it puts the three cases in the right order: a
#: reversed ranking scores below 0, a model with no ranking information scores
#: exactly 0, and any real skill scores above it. The per-allele table keeps the
#: honest NaN and flags the allele as ``unranked``, so the convention is never
#: read as a measurement; see :meth:`Scores.per_allele_frame`.
#:
#: Dropping those alleles instead would make the scoring panel depend on the
#: predictions: a model could raise its median by going constant on the alleles
#: it finds hard, and a paired comparison could pit a one-allele median against
#: a whole-panel one. The panel is fixed by the split and the labels alone.
#:
#: This applies *only* to an otherwise eligible allele whose predictions are
#: constant. It is not used for the two neighbouring cases: an allele with no
#: label spread on the rows at hand is degenerate and leaves the panel (see
#: :func:`rankable_alleles`), and NaN or infinite predictions are rejected as
#: invalid before scoring rather than scored at all.
UNRANKED_CONTRIBUTION = 0.0


def validate_finite(values, what: str) -> np.ndarray:
    """Return ``values`` as a float array, or raise if any entry is not finite.

    Enforced at the library boundary, not just in the CLI, because the scoring
    functions are the shared contract and later stages call them directly.

    NaN and infinity are **invalid input**, not a model behaviour with a score.
    Left unchecked they corrupt the primary metric silently in opposite
    directions: an infinity outranks every real prediction and lifts Spearman
    toward 1.0, while a NaN makes the allele look constant and is absorbed by the
    unranked convention as a 0. Neither is a measurement. A *finite* constant
    prediction is valid and scores :data:`UNRANKED_CONTRIBUTION`; that
    distinction is the whole point of rejecting these early.
    """
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what}: not numeric ({exc})") from exc
    bad = ~np.isfinite(arr)
    if bad.any():
        n_nan = int(np.isnan(arr).sum())
        where = ", ".join(str(i) for i in np.flatnonzero(bad)[:5])
        raise ValueError(
            f"{what}: {int(bad.sum()):,} invalid value(s) -- {n_nan:,} NaN and "
            f"{int(bad.sum()) - n_nan:,} infinite -- first at index {where}. "
            "Predictions and labels must be finite. A constant prediction is "
            f"valid and scores {UNRANKED_CONTRIBUTION}; NaN and inf are not "
            "scoreable and are rejected before any metric is computed."
        )
    return arr


def rankable_alleles(allele: pd.Series, y_true: np.ndarray,
                     alleles: list[str]) -> list[str]:
    """Of ``alleles``, those whose *labels* can be ranked on these rows.

    Label-driven, so identical for every model. Eligibility already requires
    label spread across the whole split, but a bootstrap resample can draw rows
    on which an allele has a single distinct label; such an allele is undefined
    for every model at once and leaves the panel for that resample.
    """
    ok = set(eligible_alleles(allele, y_true, min_rows=2))
    return [a for a in alleles if a in ok]


def panel_spearman(rho: pd.Series, rankable: list[str]) -> pd.Series:
    """The fixed-panel per-allele series behind every aggregate.

    Restricted to the label-rankable panel, with alleles the model failed to
    rank credited :data:`UNRANKED_CONTRIBUTION` rather than dropped.
    """
    return rho.reindex(rankable).fillna(UNRANKED_CONTRIBUTION)


def per_allele_spearman(allele: pd.Series, y_true: np.ndarray, y_pred: np.ndarray,
                        alleles: list[str] | None = None) -> pd.Series:
    """Spearman rho per allele, indexed by allele.

    Ties get average ranks (scipy's default), which matters here: 20.2% of
    labels are exactly 0, so within-allele ties are common and a tie-blind
    correlation would be optimistic.
    """
    frame = pd.DataFrame({
        "allele": np.asarray(allele),
        "y": np.asarray(y_true, dtype=float),
        "p": np.asarray(y_pred, dtype=float),
    })
    if alleles is not None:
        frame = frame[frame["allele"].isin(alleles)]
    out = {}
    for name, grp in frame.groupby("allele", sort=True):
        if len(grp) < 2 or grp["y"].nunique() < 2 or grp["p"].nunique() < 2:
            out[name] = np.nan  # undefined, reported explicitly rather than dropped
        else:
            out[name] = float(stats.spearmanr(grp["y"], grp["p"]).statistic)
    return pd.Series(out, name="spearman").sort_index()


def precision_at_k(allele: pd.Series, y_true_hours: np.ndarray, y_pred: np.ndarray,
                   alleles: list[str] | None = None, k: int = PRECISION_K,
                   threshold_hours: float = STABILITY_THRESHOLD_HOURS) -> pd.DataFrame:
    """Per-allele precision@k against a raw-hours stability threshold.

    Ties are resolved by expectation, not by row order. Taking the literal first
    k rows after sorting would score a *constant* predictor at whatever the
    first k rows happen to contain, which is an artifact of CSV order. Instead
    rows strictly above the k-th prediction value all count, and the remaining
    slots are credited the positive rate among the rows tied at that value --
    the expected precision under random tie-breaking. A constant predictor then
    scores exactly its base rate, which is the honest answer.

    Also returns ``base_rate`` (the share of that allele's held-out peptides
    above the threshold) and ``ceiling`` (the best precision@k reachable given
    how many positives exist), because precision@k alone is not comparable
    across alleles with different positive rates.
    """
    frame = pd.DataFrame({
        "allele": np.asarray(allele),
        "hours": np.asarray(y_true_hours, dtype=float),
        "p": np.asarray(y_pred, dtype=float),
    })
    if alleles is not None:
        frame = frame[frame["allele"].isin(alleles)]
    rows = []
    for name, grp in frame.groupby("allele", sort=True):
        positives = int((grp["hours"] > threshold_hours).sum())
        slots = min(k, len(grp))
        cutoff = np.sort(grp["p"].to_numpy())[::-1][slots - 1]

        above = grp[grp["p"] > cutoff]
        tied = grp[grp["p"] == cutoff]
        remaining = slots - len(above)
        expected = int((above["hours"] > threshold_hours).sum())
        if remaining > 0 and len(tied):
            expected += remaining * float((tied["hours"] > threshold_hours).mean())

        rows.append({
            "allele": name,
            "n": len(grp),
            "n_top": slots,
            "n_tied_at_cutoff": len(tied),
            "precision_at_k": expected / slots,
            "base_rate": positives / len(grp),
            "ceiling": min(positives, slots) / slots,
        })
    return pd.DataFrame(rows).set_index("allele")


@dataclass
class Scores:
    """A model's scores on one split."""

    name: str
    n_rows: int
    n_alleles_eligible: int
    n_alleles_scored: int
    n_alleles_unranked: int
    median_spearman: float
    iqr_spearman: tuple[float, float]
    mae_log1p: float
    median_precision_at_k: float
    pooled_spearman: float
    pooled_pearson_log1p: float
    #: Mathematical Spearman per allele: NaN where undefined. Never imputed.
    per_allele: pd.Series = field(repr=False, default_factory=pd.Series)
    #: The fixed panel actually aggregated: label-rankable alleles only, with
    #: unranked ones at :data:`UNRANKED_CONTRIBUTION`. ``median_spearman`` is
    #: this series' median.
    panel: pd.Series = field(repr=False, default_factory=pd.Series)
    precision_table: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    def per_allele_frame(self) -> pd.DataFrame:
        """Per-allele detail with the scoring convention made explicit.

        ``spearman`` is the mathematical value (NaN when undefined), ``unranked``
        flags the alleles where it was undefined because the predictions are
        constant, and ``scored`` is what entered the panel median. Keeping all
        three means a reader can never mistake the convention for a measurement.
        """
        detail = self.precision_table.join(self.per_allele)
        detail["scored"] = self.panel
        detail["unranked"] = detail["spearman"].isna() & detail["scored"].notna()
        cols = ["n", "spearman", "unranked", "scored", "precision_at_k",
                "base_rate", "ceiling"]
        return detail[[c for c in cols if c in detail.columns]]

    def as_row(self) -> dict:
        lo, hi = self.iqr_spearman
        return {
            "model": self.name,
            "n_rows": self.n_rows,
            "n_alleles": self.n_alleles_eligible,
            "n_scored": self.n_alleles_scored,
            "n_unranked": self.n_alleles_unranked,
            "median_per_allele_spearman": round(self.median_spearman, 4),
            "iqr_low": round(lo, 4),
            "iqr_high": round(hi, 4),
            "mae_log1p": round(self.mae_log1p, 4),
            "median_precision_at_10": round(self.median_precision_at_k, 4),
            "pooled_spearman": round(self.pooled_spearman, 4),
            "pooled_pearson_log1p": round(self.pooled_pearson_log1p, 4),
        }


def _safe_corr(fn, a: np.ndarray, b: np.ndarray) -> float:
    """Correlation that returns NaN for a constant input instead of warning.

    A constant predictor has no defined correlation. Reporting NaN says so;
    reporting 0 would wrongly imply it had been measured.
    """
    if len(a) < 2 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return float("nan")
    return float(fn(a, b).statistic)


def score(name: str, allele: pd.Series, y_true_log1p: np.ndarray,
          y_pred_log1p: np.ndarray, alleles: list[str] | None = None,
          split: str | None = None) -> Scores:
    """Score one model's predictions. Predictions are on the log1p scale.

    Raises ``ValueError`` if any label or prediction is not finite.
    """
    y_true_log1p = validate_finite(y_true_log1p, f"{name}: labels")
    y_pred_log1p = validate_finite(y_pred_log1p, f"{name}: predictions")
    if alleles is None:
        alleles = eligible_alleles(allele, y_true_log1p, split=split)

    rho = per_allele_spearman(allele, y_true_log1p, y_pred_log1p, alleles)
    hours = np.expm1(y_true_log1p)
    prec = precision_at_k(allele, hours, y_pred_log1p, alleles)

    # The panel is fixed by the labels; alleles this model could not rank count
    # as chance rather than leaving the panel. See UNRANKED_CONTRIBUTION.
    rankable = rankable_alleles(allele, y_true_log1p, alleles)
    panel = panel_spearman(rho, rankable)
    n_unranked = int(rho.reindex(rankable).isna().sum())

    mask = pd.Series(np.asarray(allele)).isin(alleles).to_numpy()
    return Scores(
        name=name,
        n_rows=int(mask.sum()),
        n_alleles_eligible=len(alleles),
        n_alleles_scored=len(panel),
        n_alleles_unranked=n_unranked,
        median_spearman=float(panel.median()) if len(panel) else float("nan"),
        iqr_spearman=((float(panel.quantile(0.25)), float(panel.quantile(0.75)))
                      if len(panel) else (float("nan"), float("nan"))),
        mae_log1p=float(np.abs(y_true_log1p[mask] - y_pred_log1p[mask]).mean()),
        median_precision_at_k=float(prec["precision_at_k"].median()),
        pooled_spearman=_safe_corr(stats.spearmanr, y_pred_log1p[mask], y_true_log1p[mask]),
        pooled_pearson_log1p=_safe_corr(stats.pearsonr, y_pred_log1p[mask], y_true_log1p[mask]),
        per_allele=rho,
        panel=panel,
        precision_table=prec,
    )


#: Distance-to-training strata for stage 6. Measured on the frozen split, the
#: held-out peptides sit at d=4 (57.8% of test rows) or d=5 (42.0%), with only
#: 6 peptides / 12 rows at d>=6 -- far too few to score, so they join d>=5.
#: Two strata is what this split actually supports.
DISTANCE_STRATA = {"d=4": (4, 4), "d>=5": (5, 9)}


#: Rows an allele needs *within a stratum*. A stratum holds roughly half a
#: split, so the split-level bar is too strict here: on the test split it would
#: keep 17 alleles at d=4 but only 7 at d>=5, and a different 7 -- comparing
#: those two numbers would compare allele panels, not distances. At 20 rows both
#: strata keep ~66 alleles covering 98% of their rows.
MIN_ROWS_PER_ALLELE_IN_STRATUM = 20


def score_by_distance(name: str, dist_to_train: pd.Series, allele: pd.Series,
                      y_true_log1p: np.ndarray, y_pred_log1p: np.ndarray,
                      split: str | None = None) -> pd.DataFrame:
    """Score within each distance-to-training stratum, on a common allele set.

    Tests whether residual similarity at the split boundary inflates results:
    if the model is exploiting near-neighbours, d=4 should score better than
    d>=5.

    The alleles are the *intersection* of those eligible in every stratum, so
    the strata differ only in distance. Without that, each stratum would be
    scored on a different allele panel and the comparison would be
    uninterpretable.
    """
    # Up front, not per stratum: the empty-panel branch below reports MAE without
    # going through score(), so it would otherwise skip validation entirely.
    y_true_log1p = validate_finite(y_true_log1p, f"{name}: labels")
    y_pred_log1p = validate_finite(y_pred_log1p, f"{name}: predictions")
    dist = np.asarray(dist_to_train)
    masks = {label: (dist >= lo) & (dist <= hi)
             for label, (lo, hi) in DISTANCE_STRATA.items()}
    masks = {k: v for k, v in masks.items() if v.any()}

    common: set[str] | None = None
    for mask in masks.values():
        here = set(eligible_alleles(pd.Series(np.asarray(allele)[mask]),
                                    np.asarray(y_true_log1p)[mask],
                                    min_rows=MIN_ROWS_PER_ALLELE_IN_STRATUM))
        common = here if common is None else (common & here)
    alleles = sorted(common or [])

    rows = []
    for label, mask in masks.items():
        sub_allele = pd.Series(np.asarray(allele)[mask])
        sub_true = np.asarray(y_true_log1p)[mask]
        sub_pred = np.asarray(y_pred_log1p)[mask]
        if not alleles:
            rows.append({"model": name, "stratum": label, "n_rows": int(mask.sum()),
                         "n_alleles": 0, "median_per_allele_spearman": float("nan"),
                         "mae_log1p": float(np.abs(sub_true - sub_pred).mean())})
            continue
        s = score(f"{name}[{label}]", sub_allele, sub_true, sub_pred, alleles)
        row = s.as_row()
        row["model"], row["stratum"] = name, label
        rows.append(row)
    cols = ["model", "stratum", "n_rows", "n_alleles",
            "median_per_allele_spearman", "iqr_low", "iqr_high", "mae_log1p",
            "median_precision_at_10"]
    frame = pd.DataFrame(rows)
    return frame[[c for c in cols if c in frame.columns]]


#: How a resample's per-allele Spearman series is reduced to one number.
#: ``median`` is the contract (EVALUATION.md); ``mean`` exists only so a
#: diagnostic can put an interval on the same quantity an external paper
#: reports, since NetMHCstabpan's published figure is a mean over allotypes.
#: A median interval does not bound a mean effect, so the two are not
#: interchangeable and the primary metric never changes.
PANEL_STATISTICS = {"median": lambda s: float(s.median()),
                    "mean": lambda s: float(s.mean())}


def paired_cluster_bootstrap(cluster_id: pd.Series, allele: pd.Series,
                             y_true_log1p: np.ndarray, pred_a: np.ndarray,
                             pred_b: np.ndarray, alleles: list[str] | None = None,
                             n_boot: int = N_BOOTSTRAP,
                             seed: int = BOOTSTRAP_SEED,
                             split: str | None = None,
                             statistic: str = "median") -> dict:
    """Paired CI for ``b - a`` on the panel Spearman, median by default.

    Resamples whole peptide clusters with replacement, scoring both models on the
    same resample, so the interval reflects the paired comparison and respects
    the fact that one peptide cluster contributes rows across many alleles.

    A CI that crosses zero means inconclusive, not negative.
    """
    cluster_id = np.asarray(cluster_id)
    allele_arr = np.asarray(allele)
    # Both arms are validated before any resampling starts: 2,000 resamples of a
    # corrupt input is a slow way to reach a wrong interval.
    y_true_log1p = validate_finite(y_true_log1p, "labels")
    pred_a = validate_finite(pred_a, "baseline predictions")
    pred_b = validate_finite(pred_b, "comparison predictions")
    if alleles is None:
        alleles = eligible_alleles(allele_arr, y_true_log1p, split=split)

    unique_clusters = np.unique(cluster_id)
    rows_by_cluster = {c: np.nonzero(cluster_id == c)[0] for c in unique_clusters}
    rng = np.random.default_rng(seed)

    try:
        reduce = PANEL_STATISTICS[statistic]
    except KeyError:
        raise ValueError(f"statistic must be one of {sorted(PANEL_STATISTICS)}, "
                         f"got {statistic!r}") from None

    def median_rho(idx: np.ndarray, pred: np.ndarray, rankable: list[str]) -> float:
        rho = per_allele_spearman(allele_arr[idx], y_true_log1p[idx], pred[idx], alleles)
        panel = panel_spearman(rho, rankable)
        return reduce(panel) if len(panel) else float("nan")

    full = np.arange(len(y_true_log1p))
    full_rankable = rankable_alleles(allele_arr, y_true_log1p, alleles)
    base_rho = median_rho(full, pred_a, full_rankable)
    other_rho = median_rho(full, pred_b, full_rankable)
    if not (np.isfinite(base_rho) and np.isfinite(other_rho)):
        # No eligible allele has two distinct labels on these rows, so there is
        # nothing to rank for *either* model. Undefined is not the same as
        # inconclusive, so say so rather than returning an interval. A model
        # that is merely constant within allele does not land here: it scores
        # UNRANKED_CONTRIBUTION and compares normally.
        return {
            "delta_median_spearman": float("nan"),
            "ci95": (float("nan"), float("nan")),
            "n_boot": 0,
            "undefined": True,
            "reason": ("no eligible allele has two distinct labels on these "
                       "rows, so per-allele Spearman is undefined for both "
                       "models"),
            "conclusive": False,
            "meets_min_worthwhile": False,
            "rules_out_min_worthwhile": False,
            "spans_min_worthwhile": False,
        }

    observed = other_rho - base_rho
    deltas = np.empty(n_boot)
    panel_sizes = np.empty(n_boot, dtype=int)
    for i in range(n_boot):
        picked = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([rows_by_cluster[c] for c in picked])
        # One panel per resample, derived from the resampled labels, so both
        # models are always scored on the same alleles. An allele the resample
        # left without label spread is degenerate *for this resample* and leaves
        # its panel -- it is not scored at UNRANKED_CONTRIBUTION, which would
        # confuse "the draw could not test this allele" with "the model could
        # not rank it".
        rankable = rankable_alleles(allele_arr[idx], y_true_log1p[idx], alleles)
        panel_sizes[i] = len(rankable)
        deltas[i] = (median_rho(idx, pred_b, rankable)
                     - median_rho(idx, pred_a, rankable))

    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    return {
        "delta_median_spearman": observed,
        "statistic": statistic,
        "ci95": (float(lo), float(hi)),
        "n_boot": n_boot,
        # Resamples that could score nothing at all; excluded from the CI.
        "n_degenerate_resamples": int(np.isnan(deltas).sum()),
        # Resamples that lost at least one allele to missing label spread. A
        # large share means the interval rests on shifting panels, so report it
        # rather than hiding it inside the CI.
        "n_resamples_with_dropped_alleles": int((panel_sizes < len(full_rankable)).sum()),
        "panel_size_full": len(full_rankable),
        "undefined": False,
        "conclusive": bool(lo > 0 or hi < 0),
        "meets_min_worthwhile": bool(lo > MIN_WORTHWHILE_DELTA_SPEARMAN),
        "rules_out_min_worthwhile": bool(hi < MIN_WORTHWHILE_DELTA_SPEARMAN),
        # The CI can straddle the bar: a real gain whose size is unresolved.
        "spans_min_worthwhile": bool(lo <= MIN_WORTHWHILE_DELTA_SPEARMAN <= hi),
    }


def describe_delta(result: dict,
                   min_delta: float = MIN_WORTHWHILE_DELTA_SPEARMAN) -> str:
    """The predeclared reading of a paired CI. See EVALUATION.md.

    The six cases are mutually exclusive and cover every interval. The one worth
    naming carefully is a CI that excludes 0 but straddles ``min_delta``: the
    improvement is real and its size is *unresolved*, which is not the same as
    being below the bar.
    """
    if result.get("undefined"):
        return f"undefined -- {result['reason']}"
    lo, hi = result["ci95"]
    if hi < 0:
        return "worse: CI entirely below 0"
    if lo > min_delta:
        return f"improvement, meets the {min_delta} bar"
    if lo > 0 and hi < min_delta:
        return f"real, but rules out a {min_delta} gain"
    if lo > 0:
        return (f"real improvement, but whether it clears the {min_delta} bar is "
                f"unresolved (CI straddles {min_delta})")
    if hi < min_delta:
        return (f"inconclusive at 0, and rules out a {min_delta} gain "
                "-- not a negative result")
    return "inconclusive (CI crosses 0) -- not a negative result"
