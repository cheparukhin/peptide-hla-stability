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
from .round2_nested import unlabelled
from .round2_reporting import _weighted_rho
from .round2_embedding import EMBEDDING_PRIMARY, head_candidates
from .phase_a import allele_offsets, shrunk_allele_means

ORIGINAL = {"id": "original", "learner": "original"}
SEQUENCE_CANDIDATES = (ORIGINAL,) + tuple(
    {"id": f"mlp_centred_k{k}", "learner": "mlp", "target": "centred", "k": k}
    for k in (100, 20))
SEQUENCE_SELECTOR = "bounded_MLP_sequence_selector"
EMBEDDING_SELECTOR = "embedding_selector"


def candidate_registry(nominations=()):
    """Preparation registry only; nominees are explicit after the main gate.

    A nomination identifies a representation/head PROCEDURE, not an alpha
    chosen from outer outcomes. Its entire existing main regularization grid
    is screened anew inside each transfer outer cell.
    """
    nominations = tuple(nominations)
    permitted = tuple(f"{rep}_{learner}" for rep in EMBEDDING_PRIMARY
                      for learner in ("ridge", "mlp"))
    if len(nominations) > 2 or len(set(nominations)) != len(nominations):
        raise ValueError("at most two distinct main-nominated procedures required")
    if any(name not in permitted for name in nominations):
        raise ValueError("nomination outside existing primary main procedures")
    # The main parsimony order resolves ties, independently of CLI ordering.
    ordered = [name for name in permitted if name in nominations]
    return SEQUENCE_CANDIDATES + tuple(c for name in ordered
        for rep, learner in [name.rsplit("_", 1)] for c in head_candidates(rep, learner))


def validate_candidates(candidates):
    candidates = tuple(candidates)
    procedures = tuple(dict.fromkeys(f"{c['representation']}_{c['learner']}"
                                    for c in candidates if "representation" in c))
    if candidates != candidate_registry(procedures):
        raise ValueError("candidate grid outside existing bounded registry/order")
    return candidates


def procedure_order(candidates):
    validate_candidates(candidates)
    return ("original", "mlp_centred") + tuple(dict.fromkeys(
        f"{c['representation']}_{c['learner']}" for c in candidates if "representation" in c))


def candidate_procedure(candidate):
    if "representation" in candidate:
        return f"{candidate['representation']}_{candidate['learner']}"
    return "original" if candidate == ORIGINAL else "mlp_centred"


def reported_arms(candidates):
    return procedure_order(candidates) + (SEQUENCE_SELECTOR, EMBEDDING_SELECTOR)


def replay_selection(scores, candidates):
    """Max pooled endpoint; exact ties use frozen parsimony order, no one-SE."""
    candidates = validate_candidates(candidates)
    if [s['candidate'] for s in scores] != [c['id'] for c in candidates]:
        raise ValueError("selection candidate grid changed")
    values = ev.validate_finite([s['pooled_inner_score'] for s in scores], "selection scores")
    best = lambda indices: max(indices, key=lambda i: values[i])
    chosen = {arm: best([i for i,c in enumerate(candidates) if candidate_procedure(c) == arm])
              for arm in procedure_order(candidates)}
    chosen[SEQUENCE_SELECTOR] = best(range(len(SEQUENCE_CANDIDATES)))
    chosen[EMBEDDING_SELECTOR] = best(range(len(candidates)))
    return chosen


def member_schedule(fit, regime, candidate, *, final):
    """Exact supervised-member loops shared by fitting and evidence replay."""
    if candidate['learner'] == 'ridge':
        yield dict(gradient=fit, stopping=None, seed=None)
        return
    feature_head = 'representation' in candidate
    n_stop = 5 if final or feature_head else 3
    partitions = list(inner_partitions(fit, regime, n_stop, stopping=True))
    if feature_head:
        axes = [(stop, 'features', seed) for stop in range(5) for seed in range(6)] if final else [(0, 'features', 0), (1, 'features', 1)]
    else:
        axes = [(stop, enc, seed) for stop in range(n_stop)
                for enc in ('onehot', 'blosum') for seed in range(3 if final else 2)]
    for stop, encoding, seed in axes:
        cell, gradient, stopping = partitions[stop]
        cfg = MLPConfig(hidden=(64,) if feature_head else (256,64),
            l2=candidate['regularization'] if feature_head else (1e-5 if encoding == 'onehot' else 1e-3),
            max_epochs=200 if feature_head else 300, patience=15, seed=seed)
        yield dict(gradient=gradient, stopping=stopping, stopping_cell=cell,
                   encoding=encoding, seed=seed, config=cfg)


def members_per_call(candidate, *, final):
    if candidate['learner'] == 'ridge':
        return 1
    if 'representation' in candidate:
        return 30 if final else 2
    return 30 if final else 12


# No implicit Boltz shortlist: final scope is intentionally unfrozen.
CANDIDATES = candidate_registry()
ARMS = reported_arms(CANDIDATES)


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
    """Bounded sequence families and main-nominated ridge/neural feature heads."""
    def __init__(self, population, features, ancestry, *, candidates=CANDIDATES, callback=None):
        validate_frame(population)
        self.candidates = validate_candidates(candidates)
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
        return self.predict(fit, held, candidate, context, final=context.startswith('final/'))

    def predict(self, fit, held, candidate, context, *, final=False):
        validate_partition(fit, held, self.ancestry.regime)
        if any(k in held for k in ("y_log1p", "thalf_hours")):
            raise ValueError("predictor received assessment labels")
        if candidate not in self.candidates:
            raise ValueError("candidate outside bounded scope")
        ev.validate_finite(fit.y_log1p, "fitting labels")
        values, prepared = [], {}
        for specification in member_schedule(fit, self.ancestry.regime, candidate, final=final):
            start = time.perf_counter()
            gradient, stopping = specification['gradient'], specification['stopping']
            feature_head = 'representation' in candidate
            encoding = specification.get('encoding')
            cache_key = (specification.get('stopping_cell'), encoding)
            if cache_key not in prepared:
                frames = (gradient, held) if stopping is None else (gradient, stopping, held)
                arrays = [self.X(f, candidate['representation']) if feature_head else
                          build_features(f, 'pep_pseudo', encoding).astype(np.float32) for f in frames]
                scaler = StandardScaler().fit(arrays[0]) if feature_head else None
                if scaler is not None:
                    arrays = [scaler.transform(x).astype(np.float32) for x in arrays]
                prepared[cache_key] = (arrays, scaler)
            arrays, scaler = prepared[cache_key]
            means, global_mean = {}, None
            if candidate.get('target') == 'centred':
                means, global_mean = shrunk_allele_means(gradient.allele, gradient.y_log1p, candidate['k'])
            offsets = lambda f: allele_offsets(f.allele, means, global_mean) if global_mean is not None else np.zeros(len(f))
            y = gradient.y_log1p.to_numpy() - offsets(gradient)
            details = dict(seed=specification['seed'], final_recipe=final,
                n_source=len(fit), excluded_cross_quadrant_rows=len(fit)-len(gradient)-(0 if stopping is None else len(stopping)),
                centering_means=means, centering_global=global_mean)
            if stopping is None:
                model = Ridge(alpha=candidate['regularization'], solver='lsqr', tol=1e-5, max_iter=2000).fit(arrays[0], y)
                prediction, role = model.predict(arrays[1]), 'ridge'
            else:
                cfg = specification['config']
                model = MLPRegressor(cfg).fit(arrays[0], y, arrays[1], stopping.y_log1p.to_numpy()-offsets(stopping))
                prediction, role = model.predict(arrays[2])+offsets(held), 'mlp_member'
                details.update(encoding=encoding, stopping_cell=specification['stopping_cell'],
                    config=asdict(cfg), best_epoch=model.best_epoch_, epochs=model.n_epochs_)
            if scaler is not None:
                details.update(scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
                               feature_hash=self.feature_hashes[candidate['representation']])
            record = self.ancestry.add(role, gradient, held, stopping=stopping,
                scaler=gradient if scaler is not None else None, context=context,
                candidate=candidate['id'], **{k:details[k] for k in ('seed','encoding','stopping_cell') if k in details})
            details.update({k:record[k] for k in ('fit_rows','stopping_rows','prediction_rows','scaler_rows') if k in record})
            values.append(prediction)
            self._member(context, candidate, start, len(gradient), 0 if stopping is None else len(stopping), **details)
        if len(values) != members_per_call(candidate, final=final):
            raise ValueError('actual member loop differs from declared recipe')
        return ev.validate_finite(np.mean(values, axis=0), 'transfer predictions')

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


def assess_cell(fit, held, regime, predictor, ancestry, *, min_rows=20, candidates=CANDIDATES):
    """Pooled transfer selection on all inner cells; no full-main selector claim."""
    candidates = validate_candidates(candidates)
    validate_partition(fit, held, regime)
    panel = ev.eligible_alleles(fit.allele, fit.y_log1p, min_rows=min_rows)
    if not panel:
        raise ValueError("no eligible inner pooled panel; no outcome-driven fallback")
    index = {pid: i for i, pid in enumerate(fit.pair_id)}
    scores = []
    for candidate in candidates:
        pred = np.full(len(fit), np.nan)
        for cell, train, inner_held in inner_partitions(fit, regime):
            out = predictor(train, unlabelled(inner_held), candidate, f"select/{candidate['id']}/{cell}")
            idx = [index[pid] for pid in inner_held.pair_id]
            if np.asarray(out).shape != (len(idx),):
                raise ValueError("prediction shape mismatch")
            pred[idx] = ev.validate_finite(out, "inner prediction")
            ancestry.add("selection_prediction", train, inner_held, candidate=candidate['id'], cell=cell)
        ev.validate_finite(pred, "complete inner predictions")
        scores.append(dict(candidate=candidate['id'], pooled_inner_score=pooled_median(fit.allele, fit.y_log1p, pred, panel)))
    chosen = replay_selection(scores, candidates)
    final = {}
    for arm in procedure_order(candidates):
        candidate = candidates[chosen[arm]]
        final[arm] = predictor(fit, unlabelled(held), candidate, f"final/{arm}")
        ancestry.add("selected_prediction", fit, held, selection=fit, candidate=candidate['id'], arm=arm)
    for selector in (SEQUENCE_SELECTOR, EMBEDDING_SELECTOR):
        source = candidate_procedure(candidates[chosen[selector]])
        final[selector] = np.asarray(final[source]).copy()
        ancestry.add("selector_prediction", fit, held, selection=fit, candidate=candidates[chosen[selector]]['id'], arm=selector)
    return final, dict(scores=scores, selected={arm:candidates[i]['id'] for arm,i in chosen.items()},
        inner_panel=panel, selection_rule="largest pooled inner score; frozen parsimony exact ties")


def fit_graph(candidates=CANDIDATES):
    candidates = validate_candidates(candidates)
    regimes, totals = {}, dict(sequence_mlp_fits=0, embedding_mlp_fits=0, deterministic_ridge_fits=0)
    for regime, cells, inner in (("hla", 5, 3), ("joint", 25, 9)):
        # One final fit of each PROCEDURE, including one selected centred k.
        finals = [next(c for c in candidates if candidate_procedure(c) == arm) for arm in procedure_order(candidates)]
        by_type = dict(sequence_mlp_fits=0, embedding_mlp_fits=0, deterministic_ridge_fits=0)
        for candidate, final, repetitions in [(c,False,inner) for c in candidates]+[(c,True,1) for c in finals]:
            category = 'deterministic_ridge_fits' if candidate['learner'] == 'ridge' else ('embedding_mlp_fits' if 'representation' in candidate else 'sequence_mlp_fits')
            by_type[category] += repetitions*members_per_call(candidate, final=final)
        regimes[regime] = dict(outer_cells=cells, inner_cells=inner, **by_type,
            fits_per_cell=sum(by_type.values()), total_fits=cells*sum(by_type.values()))
        for category in totals:
            totals[category] += cells*by_type[category]
    if totals['sequence_mlp_fits'] != 10440:
        raise ValueError('sequence fit graph discrepancy')
    for learner, contribution in (('ridge',750), ('mlp',2340)):
        procedures = [name for name in procedure_order(candidates) if name.endswith('_'+learner)]
        category = 'deterministic_ridge_fits' if learner == 'ridge' else 'embedding_mlp_fits'
        if totals[category] != len(procedures)*contribution:
            raise ValueError('embedding fit graph discrepancy')
    return dict(regimes=regimes, total_fits=sum(totals.values()), **totals,
        selector_extra_fits=0, cloud_spend_usd=0, repeats="excluded; locked execution repeats separate")


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
    references = [name for name in ("original", SEQUENCE_SELECTOR) if name in predictions]
    if not references:
        raise ValueError("transfer comparisons require a declared reference")
    contrasts = [(baseline, arm) for baseline in references for arm in predictions if arm != baseline and not (arm == "original" and baseline == SEQUENCE_SELECTOR)]
    rows, bootstraps = [], {}
    for mode in ("peptide", "hla", "crossed"):
        boot, meta = transfer_bootstrap(frame, predictions, min_rows=min_rows, n_boot=n_boot, mode=mode)
        bootstraps[mode] = boot
        for baseline, arm in contrasts:
            sample = boot[arm] - boot[baseline]
            valid = sample[np.isfinite(sample)]
            low, high = np.percentile(valid, [2.5, 97.5]) if len(valid) else (np.nan, np.nan)
            delta = point[arm] - point[baseline]
            rows.append({"baseline": baseline, "candidate": arm, "delta": delta,
                "baseline_score": point[baseline], "candidate_score": point[arm],
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
        for baseline, arm in contrasts:
            sensitivity.append({"dropped_hla_group": int(group), "dropped_rows": int((~mask).sum()),
                "dropped_alleles": int(frame.loc[~mask, "allele"].nunique()), "panel_min_rows": min_rows,
                "baseline": baseline, "candidate": arm, "delta": scores[arm]-scores[baseline]})
    return pd.DataFrame(rows), bootstraps, pd.DataFrame(sensitivity)
