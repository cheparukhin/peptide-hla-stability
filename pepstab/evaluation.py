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
    median_spearman: float
    iqr_spearman: tuple[float, float]
    mae_log1p: float
    median_precision_at_k: float
    pooled_spearman: float
    pooled_pearson_log1p: float
    per_allele: pd.Series = field(repr=False, default_factory=pd.Series)
    precision_table: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    def as_row(self) -> dict:
        lo, hi = self.iqr_spearman
        return {
            "model": self.name,
            "n_rows": self.n_rows,
            "n_alleles": self.n_alleles_eligible,
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
    """Score one model's predictions. Predictions are on the log1p scale."""
    y_true_log1p = np.asarray(y_true_log1p, dtype=float)
    y_pred_log1p = np.asarray(y_pred_log1p, dtype=float)
    if alleles is None:
        alleles = eligible_alleles(allele, y_true_log1p, split=split)

    rho = per_allele_spearman(allele, y_true_log1p, y_pred_log1p, alleles)
    hours = np.expm1(y_true_log1p)
    prec = precision_at_k(allele, hours, y_pred_log1p, alleles)
    finite = rho.dropna()

    mask = pd.Series(np.asarray(allele)).isin(alleles).to_numpy()
    return Scores(
        name=name,
        n_rows=int(mask.sum()),
        n_alleles_eligible=len(alleles),
        median_spearman=float(finite.median()) if len(finite) else float("nan"),
        iqr_spearman=((float(finite.quantile(0.25)), float(finite.quantile(0.75)))
                      if len(finite) else (float("nan"), float("nan"))),
        mae_log1p=float(np.abs(y_true_log1p[mask] - y_pred_log1p[mask]).mean()),
        median_precision_at_k=float(prec["precision_at_k"].median()),
        pooled_spearman=_safe_corr(stats.spearmanr, y_pred_log1p[mask], y_true_log1p[mask]),
        pooled_pearson_log1p=_safe_corr(stats.pearsonr, y_pred_log1p[mask], y_true_log1p[mask]),
        per_allele=rho,
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


def paired_cluster_bootstrap(cluster_id: pd.Series, allele: pd.Series,
                             y_true_log1p: np.ndarray, pred_a: np.ndarray,
                             pred_b: np.ndarray, alleles: list[str] | None = None,
                             n_boot: int = N_BOOTSTRAP,
                             seed: int = BOOTSTRAP_SEED,
                             split: str | None = None) -> dict:
    """Paired CI for ``b - a`` on median per-allele Spearman.

    Resamples whole peptide clusters with replacement, scoring both models on the
    same resample, so the interval reflects the paired comparison and respects
    the fact that one peptide cluster contributes rows across many alleles.

    A CI that crosses zero means inconclusive, not negative.
    """
    cluster_id = np.asarray(cluster_id)
    allele_arr = np.asarray(allele)
    y_true_log1p = np.asarray(y_true_log1p, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    if alleles is None:
        alleles = eligible_alleles(allele_arr, y_true_log1p, split=split)

    unique_clusters = np.unique(cluster_id)
    rows_by_cluster = {c: np.nonzero(cluster_id == c)[0] for c in unique_clusters}
    rng = np.random.default_rng(seed)

    def median_rho(idx: np.ndarray, pred: np.ndarray) -> float:
        rho = per_allele_spearman(allele_arr[idx], y_true_log1p[idx], pred[idx], alleles)
        finite = rho.dropna()
        return float(finite.median()) if len(finite) else float("nan")

    full = np.arange(len(y_true_log1p))
    base_rho, other_rho = median_rho(full, pred_a), median_rho(full, pred_b)
    if not (np.isfinite(base_rho) and np.isfinite(other_rho)):
        # One side has no defined per-allele correlation at all -- typically a
        # predictor that is constant within each allele. The difference is
        # undefined, which is not the same as inconclusive, so say so rather
        # than returning an interval.
        return {
            "delta_median_spearman": float("nan"),
            "ci95": (float("nan"), float("nan")),
            "n_boot": 0,
            "undefined": True,
            "reason": ("per-allele Spearman is undefined for "
                       + " and ".join(n for n, r in [("baseline", base_rho),
                                                     ("comparison", other_rho)]
                                      if not np.isfinite(r))
                       + " (constant within allele)"),
            "conclusive": False,
            "meets_min_worthwhile": False,
            "rules_out_min_worthwhile": False,
        }

    observed = other_rho - base_rho
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([rows_by_cluster[c] for c in picked])
        deltas[i] = median_rho(idx, pred_b) - median_rho(idx, pred_a)

    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    return {
        "delta_median_spearman": observed,
        "ci95": (float(lo), float(hi)),
        "n_boot": n_boot,
        "n_degenerate_resamples": int(np.isnan(deltas).sum()),
        "undefined": False,
        "conclusive": bool(lo > 0 or hi < 0),
        "meets_min_worthwhile": bool(lo > MIN_WORTHWHILE_DELTA_SPEARMAN),
        "rules_out_min_worthwhile": bool(hi < MIN_WORTHWHILE_DELTA_SPEARMAN),
    }
