"""Outcome-blind transfer infrastructure; never reuse peptide-only split helpers.

The bounded selector is a DRAFT until the main assessment completes. HLA and
joint ancestry is enforced at every supervised boundary, including stopping.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from . import evaluation as ev
from .features import build_features
from .mlp import MLPConfig, MLPRegressor
from .round2_nested import config_digest, unlabelled
from .round2_reporting import _weighted_rho

ORIGINAL = {"id": "original", "learner": "original"}
CANDIDATES = (ORIGINAL,) + tuple(
    {"id": f"{rep}_ridge_{alpha:g}", "learner": "ridge", "representation": rep,
     "regularization": alpha}
    for rep in ("R2", "R4") for alpha in (10000., 100., 1.))
ARMS = ("original", "R2_ridge", "R4_ridge", "transfer_selector")


def validate_frame(frame):
    required = {"pair_id", "cluster_id", "hla_group", "allele", "split"}
    if not required.issubset(frame) or frame[list(required)].isna().any().any():
        raise ValueError("complete training identity/group metadata required")
    if frame.pair_id.duplicated().any() or not frame.split.eq("train").all():
        raise ValueError("unique original training rows required")
    if (frame.groupby("allele").hla_group.nunique() != 1).any():
        raise ValueError("allele maps to multiple HLA groups")


def validate_partition(fit, held, regime):
    if regime not in ("hla", "joint"):
        raise ValueError("unknown transfer regime")
    for frame in (fit, held):
        validate_frame(frame)
    for column in ("pair_id", "hla_group") + (("cluster_id",) if regime == "joint" else ()):
        if set(fit[column]) & set(held[column]):
            raise ValueError(f"{regime}: overlapping {column}")


def balanced_folds(frame, column, n_folds):
    counts = frame.groupby(column).size()
    if len(counts) < n_folds:
        raise ValueError(f"insufficient {column} groups for {n_folds} folds")
    loads = np.zeros(n_folds, dtype=int)
    mapping = {}
    for group in sorted(counts.index, key=lambda g: (-counts[g], g)):
        fold = int(np.argmin(loads))
        mapping[group] = fold
        loads[fold] += int(counts[group])
    return frame[column].map(mapping).to_numpy(dtype=int)


def inner_partitions(frame, regime, n_folds=3, *, stopping=False):
    """Joint selection exhausts n² cells; stopping uses n diagonal cells.

    Diagonal stopping retains the original five stopping members, each fitted
    outside both its held groups. Off-diagonal quadrants are unused for that
    member. It intentionally does not guarantee every row is a stopping row.
    """
    validate_frame(frame)
    if regime not in ("hla", "joint"):
        raise ValueError("unknown transfer regime")
    h = balanced_folds(frame, "hla_group", n_folds)
    p = balanced_folds(frame, "cluster_id", n_folds) if regime == "joint" else None
    for hf in range(n_folds):
        for pf in ([None] if regime == "hla" else ([hf] if stopping else range(n_folds))):
            hm = h == hf
            pm = p == pf if p is not None else None
            fit_mask = ~hm if pm is None else ~hm & ~pm
            held_mask = hm if pm is None else hm & pm
            fit, held = frame.loc[fit_mask], frame.loc[held_mask]
            if not len(fit) or not len(held):
                raise ValueError(f"empty frozen {regime} inner cell {hf}/{pf}; no cell may be dropped")
            validate_partition(fit, held, regime)
            yield f"h{hf}" + ("" if pf is None else f"_p{pf}"), fit, held


def outer_cells(frame):
    validate_frame(frame)
    for column, group in (("hla_fold", "hla_group"), ("peptide_fold", "cluster_id")):
        if set(frame[column]) != set(range(5)) or (frame.groupby(group)[column].nunique() != 1).any():
            raise ValueError("invalid frozen outer fold assignment")
    for regime in ("hla", "joint"):
        for hf in range(5):
            for pf in ([None] if regime == "hla" else range(5)):
                hm = frame.hla_fold.eq(hf)
                pm = frame.peptide_fold.eq(pf) if pf is not None else None
                fit = frame.loc[~hm if pm is None else ~hm & ~pm]
                held = frame.loc[hm if pm is None else hm & pm]
                if not len(fit) or not len(held):
                    raise ValueError("empty frozen outer cell; no cell may be dropped")
                validate_partition(fit, held, regime)
                yield regime, f"h{hf}" + ("" if pf is None else f"_p{pf}"), fit, held


class TransferAncestry:
    def __init__(self, regime):
        self.regime, self.rowsets, self.records = regime, {}, []

    def rows(self, frame):
        ids = sorted(int(x) for x in frame.pair_id)
        key = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
        self.rowsets[key] = ids
        return key

    def add(self, role, fit, held, *, stopping=None, selection=None, scaler=None, **metadata):
        validate_partition(fit, held, self.regime)
        record = {"role": role, "fit_rows": self.rows(fit), "prediction_rows": self.rows(held), **metadata}
        for name, frame in (("stopping_rows", stopping), ("selection_rows", selection), ("scaler_rows", scaler)):
            if frame is not None:
                validate_partition(frame, held, self.regime)
                record[name] = self.rows(frame)
        if stopping is not None:
            validate_partition(fit, stopping, self.regime)
        if scaler is not None and not set(scaler.pair_id) <= set(fit.pair_id):
            raise ValueError("scaler ancestry outside gradient fitting rows")
        self.records.append(record)
        return record

    def audit(self, population, outer_fit, outer_held):
        validate_partition(outer_fit, outer_held, self.regime)
        lookup = population.set_index("pair_id", drop=False)
        permitted = set(outer_fit.pair_id)
        rowsets = {key:set(ids) for key,ids in self.rowsets.items()}
        for key, ids in self.rowsets.items():
            if key != hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest():
                raise ValueError("rowset hash mismatch")
        outer_key = self.rows(outer_held)
        checked, checked_outer, frames = set(), set(), {}
        def check(a,b):
            # Thirty ensemble members share rowsets. Validate each exact pair
            # once; metadata is immutable during this audit.
            key = (a,b)
            if key not in checked:
                for value in key:
                    if value not in frames:
                        frames[value] = lookup.loc[self.rowsets[value]]
                validate_partition(frames[a],frames[b],self.regime)
                checked.add(key)
        for record in self.records:
            held_key = record["prediction_rows"]
            for name in ("fit_rows", "stopping_rows", "selection_rows", "scaler_rows"):
                if name not in record:
                    continue
                used_key = record[name]
                if used_key not in checked_outer:
                    if not rowsets[used_key] <= permitted:
                        raise ValueError("outer or cross-quadrant row entered supervised ancestry")
                    check(used_key,outer_key)
                    checked_outer.add(used_key)
                check(used_key,held_key)
            if "stopping_rows" in record:
                check(record["fit_rows"],record["stopping_rows"])
            if "scaler_rows" in record and not rowsets[record["scaler_rows"]] <= rowsets[record["fit_rows"]]:
                raise ValueError("scaler ancestry outside gradient fitting rows")
        return {"regime": self.regime, "records": len(self.records), "rowsets": len(self.rowsets),
                "outer_exclusion": True, "predicted_group_exclusion": True,
                "cross_quadrant_exclusion": self.regime == "joint"}

    def as_dict(self):
        return {"regime": self.regime, "rowsets": self.rowsets, "records": self.records}


class TransferPredictor:
    """Local original recipe (always 30 members) and deterministic ridge."""
    def __init__(self, population, features, ancestry, *, callback=None):
        validate_frame(population)
        self.position = {int(pid): i for i, pid in enumerate(population.pair_id)}
        self.features = {key: np.asarray(value, dtype=np.float32) for key, value in features.items()}
        for key, value in self.features.items():
            if value.ndim != 2 or len(value) != len(population) or not np.isfinite(value).all():
                raise ValueError(f"invalid features {key}")
        self.feature_hashes = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in self.features.items()}
        self.ancestry, self.callback, self.members = ancestry, callback, []

    def X(self, frame, key):
        return self.features[key][[self.position[int(pid)] for pid in frame.pair_id]]

    def __call__(self, fit, held, candidate, context):
        validate_partition(fit, held, self.ancestry.regime)
        if any(k in held for k in ("y_log1p", "thalf_hours")):
            raise ValueError("predictor received assessment labels")
        if candidate not in CANDIDATES:
            raise ValueError("candidate outside bounded scope")
        y = ev.validate_finite(fit.y_log1p, "fitting labels")
        values = []
        if candidate["learner"] == "ridge":
            start = time.perf_counter()
            X = self.X(fit, candidate["representation"])
            scaler = StandardScaler().fit(X)
            model = Ridge(alpha=candidate["regularization"], solver="lsqr", tol=1e-5, max_iter=2000)
            model.fit(scaler.transform(X), y)
            values.append(model.predict(scaler.transform(self.X(held, candidate["representation"]))))
            self.ancestry.add("ridge", fit, held, scaler=fit, context=context, candidate=candidate["id"])
            self._member(context, candidate, start, len(fit), 0, seed=None, n_source=len(fit), excluded_cross_quadrant_rows=0)
        else:
            for stop_cell, gradient, stopping in inner_partitions(fit, self.ancestry.regime, 5, stopping=True):
                for encoding in ("onehot", "blosum"):
                    X, Xs, Xh = [build_features(f, "pep_pseudo", encoding).astype(np.float32)
                                  for f in (gradient, stopping, held)]
                    for seed in range(3):
                        start = time.perf_counter()
                        cfg = MLPConfig(hidden=(256, 64), l2=1e-5 if encoding == "onehot" else 1e-3,
                                        max_epochs=300, patience=15, seed=seed)
                        model = MLPRegressor(cfg).fit(X, gradient.y_log1p.to_numpy(), Xs, stopping.y_log1p.to_numpy())
                        values.append(model.predict(Xh))
                        self.ancestry.add("original_member", gradient, held, stopping=stopping,
                                          context=context, seed=seed, encoding=encoding, stopping_cell=stop_cell)
                        self._member(context, candidate, start, len(gradient), len(stopping), seed=seed,
                                     encoding=encoding, stopping_cell=stop_cell, config=asdict(cfg),
                                     n_source=len(fit), excluded_cross_quadrant_rows=len(fit)-len(gradient)-len(stopping),
                                     best_epoch=model.best_epoch_, epochs=model.n_epochs_)
        result = np.mean(values, axis=0)
        return ev.validate_finite(result, "transfer predictions")

    def _member(self, context, candidate, start, n_fit, n_stop, **details):
        member = {"context": context, "candidate": candidate["id"], "fit_seconds": time.perf_counter()-start,
                  "n_fit": n_fit, "n_stop": n_stop, **details}
        self.members.append(member)
        if self.callback:
            self.callback(member)


def pooled_median(allele, labels, prediction, panel):
    """Ranking endpoint only; empty diagnostic panels are honestly undefined."""
    y = ev.validate_finite(labels, "labels")
    p = ev.validate_finite(prediction, "predictions")
    rankable = ev.rankable_alleles(allele, y, panel)
    if not rankable:
        return float("nan")
    return float(ev.panel_spearman(ev.per_allele_spearman(allele,y,p,panel),rankable).median())


def assess_cell(fit, held, regime, predictor, ancestry, *, min_rows=20):
    """Selection uses pooled inner predictions: all 3 or 9 cells are retained.

    No cellwise eligibility threshold is used for selecting a pooled endpoint.
    Max pooled score wins, frozen order resolves exact ties (baseline first).
    """
    validate_partition(fit, held, regime)
    panel = ev.eligible_alleles(fit.allele, fit.y_log1p, min_rows=min_rows)
    if not panel:
        raise ValueError("no eligible inner pooled panel; no outcome-driven fallback")
    index = {pid: i for i, pid in enumerate(fit.pair_id)}
    scores = []
    for candidate in CANDIDATES:
        pred = np.full(len(fit), np.nan)
        for cell, train, inner_held in inner_partitions(fit, regime):
            out = predictor(train, unlabelled(inner_held), candidate, f"select/{candidate['id']}/{cell}")
            idx = [index[pid] for pid in inner_held.pair_id]
            if np.asarray(out).shape != (len(idx),):
                raise ValueError("prediction shape mismatch")
            pred[idx] = ev.validate_finite(out, "inner prediction")
            ancestry.add("selection_prediction", train, inner_held, candidate=candidate["id"], cell=cell)
        ev.validate_finite(pred, "complete inner predictions")
        score = pooled_median(fit.allele, fit.y_log1p, pred, panel)
        scores.append({"candidate": candidate["id"], "pooled_inner_score": float(score)})
    best = lambda indices: max(indices, key=lambda i: scores[i]["pooled_inner_score"])
    chosen = {"original": 0, "R2_ridge": best(range(1, 4)), "R4_ridge": best(range(4, 7))}
    chosen["transfer_selector"] = best(range(len(CANDIDATES)))
    final = {}
    for arm in ARMS[:3]:
        candidate = CANDIDATES[chosen[arm]]
        final[arm] = predictor(fit, unlabelled(held), candidate, f"final/{arm}")
        ancestry.add("selected_prediction", fit, held, selection=fit, candidate=candidate["id"], arm=arm)
    chosen_id = CANDIDATES[chosen["transfer_selector"]]["id"]
    selected_arm = next(arm for arm in ARMS[:3] if chosen[arm] == chosen["transfer_selector"])
    final["transfer_selector"] = np.asarray(final[selected_arm]).copy()
    ancestry.add("selector_prediction", fit, held, selection=fit, candidate=chosen_id)
    return final, {"scores": scores, "selected": {arm: CANDIDATES[i]["id"] for arm, i in chosen.items()},
                   "inner_panel": panel, "selection_rule": "largest pooled inner score; frozen-order exact ties"}


def fit_graph():
    regimes = {}
    for regime, cells, inner in (("hla", 5, 3), ("joint", 25, 9)):
        regimes[regime] = {"outer_cells": cells, "inner_cells": inner,
            "original_members_per_cell": 30 * (inner + 1),
            "ridge_fits_per_cell": 6 * inner + 2,
            "total_fits": cells * (30 * (inner + 1) + 6 * inner + 2)}
    return {"regimes": regimes, "total_fits": sum(r["total_fits"] for r in regimes.values()),
            "original_mlp_fits": 8100, "deterministic_ridge_fits": 1500,
            "selector_extra_fits": 0, "cloud_spend_usd": 0,
            "repeats": "not included; main locked execution repeats are separate"}


def _frequency_median(values, multiplicity):
    """Median after integer replication, with exact even-sample midpoint."""
    keep = np.isfinite(values) & (multiplicity > 0)
    x, w = np.asarray(values)[keep], np.asarray(multiplicity, dtype=int)[keep]
    if not len(x):
        return float("nan")
    order = np.argsort(x, kind="stable")
    x, c = x[order], np.cumsum(w[order])
    n = int(c[-1])
    return float((x[np.searchsorted(c, (n-1)//2 + 1)] + x[np.searchsorted(c, n//2 + 1)]) / 2)


def transfer_bootstrap(frame, predictions, *, min_rows=20, n_boot=2000,
                       seed=ev.BOOTSTRAP_SEED, mode="crossed"):
    """Paired cluster/group/pigeonhole crossed bootstrap conditional on fits.

    Group multiplicity replicates each allele in the PANEL median as well as
    its rows. Merely multiplying row weights would cancel within an allele and
    silently omit HLA-population uncertainty. Each clone has the same global
    peptide draw. Label-degenerate alleles are removed identically for all arms.
    """
    validate_frame(frame)
    if mode not in ("peptide", "hla", "crossed") or n_boot < 1 or not len(frame) or not predictions:
        raise ValueError("invalid bootstrap request")
    y = ev.validate_finite(frame.y_log1p, "labels")
    predictions = {name: ev.validate_finite(p, name) for name, p in predictions.items()}
    if any(v.shape != y.shape for v in predictions.values()):
        raise ValueError("prediction shape mismatch")
    panel = ev.eligible_alleles(frame.allele, y, min_rows=min_rows)
    if not panel:
        raise ValueError("no eligible pooled panel")
    clusters, ci = np.unique(frame.cluster_id, return_inverse=True)
    groups, gi = np.unique(frame.hla_group, return_inverse=True)
    rng = np.random.default_rng(seed)
    pc = np.ones((n_boot, len(clusters)), dtype=np.int32)
    hc = np.ones((n_boot, len(groups)), dtype=np.int32)
    for i in range(n_boot):
        if mode in ("peptide", "crossed"):
            pc[i] = np.bincount(rng.choice(len(clusters), len(clusters), replace=True), minlength=len(clusters))
        if mode in ("hla", "crossed"):
            hc[i] = np.bincount(rng.choice(len(groups), len(groups), replace=True), minlength=len(groups))
    rhos = {name: np.empty((n_boot, len(panel))) for name in predictions}
    panel_weights = np.empty((n_boot, len(panel)), dtype=np.int32)
    for ai, allele in enumerate(panel):
        idx = np.flatnonzero(frame.allele.to_numpy() == allele)
        group = gi[idx[0]]
        panel_weights[:, ai] = hc[:, group]
        weights = pc[:, ci[idx]].astype(float) * hc[:, group, None]
        for name, p in predictions.items():
            rhos[name][:, ai] = _weighted_rho(y[idx], p[idx], weights)
    scores = pd.DataFrame({name: [_frequency_median(r[i], panel_weights[i]) for i in range(n_boot)]
                           for name, r in rhos.items()})
    rankable = np.isfinite(next(iter(rhos.values()))) & (panel_weights > 0)
    meta = {"mode": mode, "seed": seed, "n_boot": n_boot, "panel_min_rows": min_rows,
            "panel": panel, "n_hla_groups": len(groups), "n_peptide_clusters": len(clusters),
            "n_alleles": len(panel), "eligible_hla_groups": int(frame[frame.allele.isin(panel)].hla_group.nunique()),
            "n_rows_eligible": int(frame.allele.isin(panel).sum()),
            "n_degenerate_resamples": int((rankable.sum(axis=1) == 0).sum()),
            "n_resamples_with_dropped_alleles": int((rankable.sum(axis=1) < len(panel)).sum()),
            "panel_multiplicity": "integer HLA-group replication of each allele; exact expanded-sample median",
            "scope": "conditional on fitted predictions; crossed bootstrap is a pigeonhole sensitivity, not refitting"}
    return scores, meta


def transfer_comparisons(frame, predictions, *, min_rows=20, n_boot=2000):
    point_panel = ev.eligible_alleles(frame.allele, frame.y_log1p, min_rows=min_rows)
    point = {name: pooled_median(frame.allele, frame.y_log1p, p, point_panel)
             for name, p in predictions.items()}
    rows, bootstraps = [], {}
    for mode in ("peptide", "hla", "crossed"):
        boot, meta = transfer_bootstrap(frame, predictions, min_rows=min_rows, n_boot=n_boot, mode=mode)
        bootstraps[mode] = boot
        for arm in ARMS[1:]:
            sample = boot[arm] - boot.original
            valid = sample[np.isfinite(sample)]
            low, high = np.percentile(valid, [2.5, 97.5]) if len(valid) else (np.nan, np.nan)
            delta = point[arm] - point["original"]
            rows.append({"baseline": "original", "candidate": arm, "delta": delta,
                "baseline_score": point["original"], "candidate_score": point[arm],
                "ci_low": low, "ci_high": high, "point_inside_interval": bool(low <= delta <= high),
                "bootstrap_mean_minus_point": float(valid.mean()-delta) if len(valid) else np.nan,
                "positive_lower_bound": bool(low > 0), "lower_bound_exceeds_0_05": bool(low > .05),
                "statistic": "pooled median per-allele Spearman", **{k: v for k,v in meta.items() if k != "panel"}})
    # All-group deletion diagnostic: no sparsity cutoff can select favourable groups.
    sensitivity = []
    for group in sorted(frame.hla_group.unique()):
        mask = frame.hla_group.ne(group).to_numpy()
        panel = [a for a in point_panel if a in set(frame.loc[mask, "allele"])]
        scores = {name: pooled_median(frame.loc[mask, "allele"], frame.loc[mask, "y_log1p"], p[mask], panel)
                  for name, p in predictions.items()}
        for arm in ARMS[1:]:
            sensitivity.append({"dropped_hla_group": int(group), "dropped_rows": int((~mask).sum()),
                "dropped_alleles": int(frame.loc[~mask, "allele"].nunique()), "panel_min_rows": min_rows,
                "candidate": arm, "delta": scores[arm]-scores["original"]})
    return pd.DataFrame(rows), bootstraps, pd.DataFrame(sensitivity)
