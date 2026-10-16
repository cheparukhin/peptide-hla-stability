"""Label-isolated assessment and stacking for the round-two development study.

Unlike the historical Phase A helper, *every* prediction here excludes its
cluster from fitting, stopping, and selection.  Predictors never receive the
assessment target column.  The original split and metric modules are reused.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from . import evaluation, nested

TARGET_COLUMNS = ("y_log1p", "thalf_hours")


def unlabelled(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=list(TARGET_COLUMNS), errors="ignore").copy()


def validate_partition(fit: pd.DataFrame, held: pd.DataFrame) -> None:
    for name, frame in (("fit", fit), ("held", held)):
        if frame.pair_id.duplicated().any():
            raise ValueError(f"duplicate pair_id in {name}")
        if frame.cluster_id.isna().any():
            raise ValueError(f"missing cluster_id in {name}")
        if "split" in frame and not frame.split.eq("train").all():
            raise ValueError("round-two development accepts original training rows only")
    if set(fit.pair_id) & set(held.pair_id):
        raise ValueError("overlapping row identities")
    if set(fit.cluster_id) & set(held.cluster_id):
        raise ValueError("overlapping peptide clusters")


def align_predictions(frame: pd.DataFrame, predictions: pd.DataFrame) -> np.ndarray:
    """One-to-one identity join that refuses missing, extra, or altered rows."""
    if frame.pair_id.duplicated().any() or predictions.pair_id.duplicated().any():
        raise ValueError("duplicate prediction identity")
    if set(frame.pair_id) != set(predictions.pair_id):
        raise ValueError("prediction coverage differs from requested population")
    joined = frame[["pair_id", "cluster_id"]].merge(
        predictions[["pair_id", "cluster_id", "y_pred"]], on="pair_id",
        how="left", suffixes=("", "_pred"), validate="one_to_one", sort=False)
    if not joined.cluster_id.eq(joined.cluster_id_pred).all():
        raise ValueError("prediction cluster membership changed")
    return evaluation.validate_finite(joined.y_pred, "predictions")


class Ancestry:
    """Content-addressed row sets keep complete fitting ancestry compact."""

    def __init__(self):
        self.rowsets: dict[str, list[int]] = {}
        self.records: list[dict] = []

    def rows(self, frame: pd.DataFrame) -> str:
        ids = sorted(int(x) for x in frame.pair_id)
        key = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
        self.rowsets.setdefault(key, ids)
        return key

    def add(self, *, role: str, fit: pd.DataFrame, held: pd.DataFrame,
            stopping: pd.DataFrame | None = None,
            selection: pd.DataFrame | None = None, **metadata) -> dict:
        validate_partition(fit, held)
        if stopping is not None:
            validate_partition(stopping, held)
            validate_partition(fit, stopping)
        if selection is not None:
            validate_partition(selection, held)
        record = {"role": role, "fit_rows": self.rows(fit),
                  "prediction_rows": self.rows(held), **metadata}
        for name, frame in (("stopping_rows", stopping), ("selection_rows", selection)):
            if frame is not None:
                record[name] = self.rows(frame)
        self.records.append(record)
        return record

    def audit(self, population: pd.DataFrame, outer_held: pd.DataFrame) -> dict:
        cluster = population.set_index("pair_id").cluster_id.to_dict()
        forbidden = set(outer_held.cluster_id)
        for record in self.records:
            predicted = {cluster[i] for i in self.rowsets[record["prediction_rows"]]}
            for role in ("fit_rows", "stopping_rows", "selection_rows"):
                if role in record:
                    used = {cluster[i] for i in self.rowsets[record[role]]}
                    if used & (predicted | forbidden):
                        raise AssertionError(f"{record['role']}: {role} ancestry violation")
        return {"records": len(self.records), "rowsets": len(self.rowsets),
                "outer_cluster_exclusion": True, "predicted_cluster_exclusion": True}

    def as_dict(self) -> dict:
        return {"rowsets": self.rowsets, "records": self.records}


@dataclass
class Selection:
    candidate: dict
    scores: list[dict]
    rule: str


def select(fit: pd.DataFrame, candidates: Sequence[dict], predictor: Callable,
           *, n_inner: int = 3, min_rows: int = 20,
           ancestry: Ancestry | None = None, context: str = "") -> Selection:
    """Select within exactly this fitting partition; outer outcomes are absent.

    Candidates arrive in frozen parsimony order.  Take the first candidate
    within one SE of the largest mean of the three inner-fold panel medians.
    SE uses fold variation, not ensemble members as independent observations.
    """
    if not candidates:
        raise ValueError("empty candidate set")
    folds = nested.inner_folds(fit, n_inner)
    if set(folds) != set(range(n_inner)):
        raise ValueError("insufficient clusters for declared inner folds")
    panels = {fold: evaluation.eligible_alleles(
        fit.loc[folds == fold, "allele"], fit.loc[folds == fold, "y_log1p"],
        min_rows=min_rows) for fold in range(n_inner)}
    if any(not panel for panel in panels.values()):
        raise ValueError("inner fold has no eligible allele panel")
    rows = []
    for candidate in candidates:
        values = []
        for fold in range(n_inner):
            train, held = fit.loc[folds != fold], fit.loc[folds == fold]
            validate_partition(train, held)
            pred = predictor(train, unlabelled(held), candidate,
                             f"{context}/select/{candidate['id']}/{fold}")
            pred = evaluation.validate_finite(pred, "inner predictions")
            score = evaluation.score("inner", held.allele, held.y_log1p,
                                     pred, panels[fold]).median_spearman
            if not np.isfinite(score):
                raise ValueError("inner fold has no eligible allele panel")
            values.append(float(score))
            if ancestry is not None:
                ancestry.add(role="inner_selection_prediction", fit=train,
                             held=held, candidate=candidate["id"], context=context,
                             inner_fold=fold)
        rows.append({"candidate": candidate["id"], "fold_scores": values,
                     "mean": float(np.mean(values)),
                     "se": float(np.std(values, ddof=1) / np.sqrt(n_inner))})
    best = max(rows, key=lambda x: x["mean"])
    cutoff = best["mean"] - best["se"]
    chosen = next(c for c, r in zip(candidates, rows) if r["mean"] >= cutoff)
    return Selection(dict(chosen), rows, "first in frozen parsimony order within 1 fold-SE of best")


def selected_predict(fit: pd.DataFrame, held: pd.DataFrame,
                     candidates: Sequence[dict], predictor: Callable, *,
                     n_inner: int = 3, min_rows: int = 20,
                     ancestry: Ancestry | None = None, context: str = ""):
    validate_partition(fit, held)
    choice = select(fit, candidates, predictor, n_inner=n_inner,
                    min_rows=min_rows, ancestry=ancestry, context=context)
    values = predictor(fit, unlabelled(held), choice.candidate, context + "/refit")
    if ancestry is not None:
        ancestry.add(role="selected_prediction", fit=fit, held=held, selection=fit,
                     candidate=choice.candidate["id"], context=context)
    return evaluation.validate_finite(values, "selected predictions"), choice


def clean_crossfit(fit: pd.DataFrame, candidates: Sequence[dict], predictor: Callable,
                  *, n_inner: int = 3, min_rows: int = 20,
                  ancestry: Ancestry | None = None, context: str = ""):
    """Nested OOF inputs: reselect the base recipe without each predicted fold."""
    folds = nested.inner_folds(fit, n_inner)
    pred = np.full(len(fit), np.nan)
    selections = []
    for fold in range(n_inner):
        train, held = fit.loc[folds != fold], fit.loc[folds == fold]
        pred[folds == fold], choice = selected_predict(
            train, held, candidates, predictor, n_inner=n_inner, min_rows=min_rows,
            ancestry=ancestry, context=f"{context}/crossfit/{fold}")
        selections.append({"fold": fold, "candidate": choice.candidate,
                           "scores": choice.scores})
    return evaluation.validate_finite(pred, "clean crossfit"), selections


def positive_affine(raw, target, *, minimum_slope: float = 1e-8) -> dict:
    """Positive affine OLS common-scale map; never inserts isotonic ties."""
    x = evaluation.validate_finite(raw, "affine input")
    y = evaluation.validate_finite(target, "affine target")
    if x.shape != y.shape or not len(x):
        raise ValueError("affine inputs must be nonempty aligned vectors")
    variance = float(np.mean((x - x.mean()) ** 2))
    slope = (float(np.mean((x - x.mean()) * (y - y.mean()))) / variance
             if variance > 0 else minimum_slope)
    slope = max(minimum_slope, slope)
    return {"slope": slope, "intercept": float(y.mean() - slope * x.mean()),
            "minimum_slope": minimum_slope, "n_rows": len(x)}


def apply_affine(raw, mapping: dict) -> np.ndarray:
    if mapping["slope"] <= 0:
        raise ValueError("rank mapping slope must be positive")
    return mapping["slope"] * np.asarray(raw) + mapping["intercept"]


def blend(baseline, alternative, weight: float) -> np.ndarray:
    if not 0 <= weight <= 1:
        raise ValueError("blend weight outside simplex")
    base = evaluation.validate_finite(baseline, "baseline")
    alt = evaluation.validate_finite(alternative, "alternative")
    if base.shape != alt.shape:
        raise ValueError("blend inputs are not aligned")
    if weight == 0:
        return base.copy()
    if weight == 1:
        return alt.copy()
    return (1 - weight) * base + weight * alt


def config_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
