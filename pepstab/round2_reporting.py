"""Paired whole-cluster reporting, identical to the frozen percentile procedure.

Replicated rows are represented by integer weights. Weighted average ranks are
exactly the ranks obtained by materialising the repeated rows, including ties.
This lets all declared comparisons share the same 2,000 resamples without
repeating the expensive grouping/ranking for each pair. The frozen evaluation
module remains the reference, with equivalence tested on degenerate/tied cases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluation as ev


def _weighted_ranks(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Average ranks in the expanded sample (weights are integer counts)."""
    _, inverse = np.unique(values, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(inverse[order])) + 1]
    totals = np.add.reduceat(weights[:, order], starts, axis=1)
    group_ranks = np.cumsum(totals, axis=1) - (totals - 1.0) / 2.0
    return group_ranks[:, inverse]


def _weighted_rho(y, prediction, weights):
    ry = _weighted_ranks(np.asarray(y), weights)
    rp = _weighted_ranks(np.asarray(prediction), weights)
    total = weights.sum(axis=1)
    mean_rank = (total + 1.0) / 2.0
    dy, dp = ry - mean_rank[:, None], rp - mean_rank[:, None]
    vy = np.sum(weights * dy * dy, axis=1)
    vp = np.sum(weights * dp * dp, axis=1)
    cov = np.sum(weights * dy * dp, axis=1)
    result = np.zeros(len(weights))
    good = (vy > 0) & (vp > 0)
    result[good] = cov[good] / np.sqrt(vy[good] * vp[good])
    # Missing label variation removes the allele for all arms. Constant
    # predictions with varying labels contribute the frozen zero convention.
    result[(total < 2) | (vy == 0)] = np.nan
    return result


def cluster_bootstrap_scores(frame: pd.DataFrame, predictions: dict[str, np.ndarray],
                             min_rows: int = 20, n_boot: int = 2000,
                             seed: int = ev.BOOTSTRAP_SEED) -> tuple[pd.DataFrame, dict]:
    """Return each procedure's score on identical peptide-cluster draws.

    ``frame`` is one row per canonical pair, in prediction order. Eligibility
    is fixed using the full labels/counts; a draw may lose label variation.
    """
    required = {"pair_id", "cluster_id", "allele", "y_log1p"}
    if not required.issubset(frame) or frame.pair_id.duplicated().any():
        raise ValueError("unique canonical pair_id and complete metadata required")
    if n_boot < 1 or not predictions or len(frame) == 0:
        raise ValueError("nonempty data/predictions and positive n_boot required")
    y = ev.validate_finite(frame.y_log1p.to_numpy(), "labels")
    predictions = {k: ev.validate_finite(v, k) for k, v in predictions.items()}
    if any(np.asarray(p).shape != y.shape for p in predictions.values()):
        raise ValueError("prediction length/shape mismatch")
    alleles = ev.eligible_alleles(frame.allele, y, min_rows=min_rows)
    if not alleles:
        raise ValueError("no eligible allele for this endpoint")
    cluster = frame.cluster_id.to_numpy()
    unique, inverse = np.unique(cluster, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = np.empty((n_boot, len(unique)), dtype=np.int32)
    # The same call and draw ordering as evaluation.paired_cluster_bootstrap.
    for i in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        counts[i] = np.bincount(np.searchsorted(unique, picked), minlength=len(unique))
    rho = {name: np.empty((n_boot, len(alleles))) for name in predictions}
    allele_vector = frame.allele.to_numpy()
    for ai, allele in enumerate(alleles):
        idx = np.flatnonzero(allele_vector == allele)
        weights = counts[:, inverse[idx]].astype(float)
        for name, pred in predictions.items():
            rho[name][:, ai] = _weighted_rho(y[idx], pred[idx], weights)
    scores = pd.DataFrame({name: np.nanmedian(r, axis=1) for name, r in rho.items()})
    panel_sizes = np.isfinite(next(iter(rho.values()))).sum(axis=1)
    meta = {"n_boot": n_boot, "bootstrap_seed": seed,
            "resampling_unit": "whole peptide cluster", "panel_min_rows": min_rows,
            "n_rows": len(frame), "n_clusters": len(unique),
            "n_alleles": len(alleles), "alleles": alleles,
            "n_rows_eligible": int(np.isin(allele_vector, alleles).sum()),
            "n_resamples_with_dropped_alleles": int((panel_sizes < len(alleles)).sum()),
            "n_degenerate_resamples": int((panel_sizes == 0).sum())}
    return scores, meta


def compare_predictions(frame: pd.DataFrame, predictions: dict[str, np.ndarray],
                        contrasts: list[tuple[str, str]], min_rows: int = 20,
                        n_boot: int = 2000, seed: int = ev.BOOTSTRAP_SEED):
    """All declared contrasts; intervals condition on the fitted predictions."""
    boot, meta = cluster_bootstrap_scores(frame, predictions, min_rows, n_boot, seed)
    point = {name: ev.score(name, frame.allele, frame.y_log1p.to_numpy(), pred,
                           alleles=meta["alleles"]).median_spearman
             for name, pred in predictions.items()}
    rows = []
    for baseline, candidate in contrasts:
        delta = point[candidate] - point[baseline]
        sample = (boot[candidate] - boot[baseline]).to_numpy()
        lo, hi = np.nanpercentile(sample, [2.5, 97.5])
        rows.append({"baseline": baseline, "candidate": candidate,
                     "statistic": "median per-allele Spearman on pooled held predictions",
                     "baseline_score": point[baseline], "candidate_score": point[candidate],
                     "delta": delta, "ci_low": float(lo), "ci_high": float(hi),
                     "point_inside_interval": bool(lo <= delta <= hi),
                     "bootstrap_mean_minus_point": float(np.nanmean(sample) - delta),
                     "positive_lower_bound": bool(lo > 0),
                     "lower_bound_exceeds_0_05": bool(lo > 0.05),
                     "uncertainty_scope": "conditional on fitted predictions; not independent confirmation",
                     **{k: v for k, v in meta.items() if k != "alleles"}})
    return pd.DataFrame(rows), boot, meta
