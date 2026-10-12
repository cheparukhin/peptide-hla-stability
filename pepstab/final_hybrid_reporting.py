"""Independent selection replay and paired statistics for the hybrid study."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluation, nested
from .round2_nested import align_predictions, positive_affine

COMPONENTS = ('standard', 'tree_rmse', 'tree_rank', 'leaf_mlp', 'quantile_mlp', 'pca_mlp')
SUFFIXES = COMPONENTS + ('blend', 'hybrid_selector')
ORDER = ('standard', 'tree_rmse', 'tree_rank', 'blend', 'quantile_mlp', 'pca_mlp', 'leaf_mlp')
WEIGHTS = (0., .25, .5, .75, 1.)


def choose_first_one_se(rows):
    if not rows or any(not np.isfinite([r['mean'], r['se']]).all() for r in rows):
        raise ValueError('finite inner scores required')
    best = max(rows, key=lambda r: r['mean'])
    return next(r for r in rows if r['mean'] >= best['mean'] - best['se'])


def summarize_inner(values, **identity):
    if len(values) != 3 or not np.isfinite(values).all():
        raise ValueError('exactly three finite inner fold scores required')
    return dict(**identity, fold_scores=list(map(float, values)),
                mean=float(np.mean(values)), se=float(np.std(values, ddof=1) / np.sqrt(3)))


def replay_selection(fit, table):
    """Score only clean inner-held vectors; never accept outer outcomes."""
    required = {'pair_id', 'cluster_id', 'allele', 'inner_fold', 'component', 'y_pred'}
    if not required.issubset(table) or set(table.component) != set(COMPONENTS):
        raise ValueError('incomplete inner component table')
    if len(table) != len(fit) * len(COMPONENTS):
        raise ValueError('inner row count changed')
    folds = nested.inner_folds(fit, 3)
    scores = {name: [] for name in COMPONENTS}
    weights = {w: [] for w in WEIGHTS}
    aligned = {name: np.empty(len(fit)) for name in COMPONENTS}
    for inner in range(3):
        mask = folds == inner
        held = fit.loc[mask]
        panel = evaluation.eligible_alleles(held.allele, held.y_log1p, min_rows=20)
        predictions = {}
        for name in COMPONENTS:
            rows = table.loc[(table.component == name) & (table.inner_fold == inner)]
            if rows.allele.tolist() != held.allele.tolist():
                raise ValueError('inner allele/order changed')
            pred = align_predictions(held, rows)
            predictions[name] = pred
            aligned[name][mask] = pred
            scores[name].append(float(evaluation.score(name, held.allele, held.y_log1p, pred, panel).median_spearman))
        for w in WEIGHTS:
            pred = (1-w)*predictions['standard'] + w*predictions['tree_rmse']
            weights[w].append(float(evaluation.score('blend', held.allele, held.y_log1p, pred, panel).median_spearman))
    blend_scores = [summarize_inner(weights[w], weight=w) for w in WEIGHTS]
    blend = choose_first_one_se(blend_scores)
    component_scores = {name: summarize_inner(scores[name], component=name) for name in COMPONENTS}
    component_scores['blend'] = summarize_inner(blend['fold_scores'], component='blend')
    ordered = [component_scores[name] for name in ORDER]
    selector = choose_first_one_se(ordered)
    mapping = positive_affine(aligned['tree_rank'], fit.y_log1p.to_numpy(), minimum_slope=1e-8)
    return dict(blend_weight=blend['weight'], blend_scores=blend_scores,
                hybrid_component=selector['component'], component_scores=ordered,
                ranker_mapping=mapping)


def contrasts():
    rows = [('primary', 'R5_standard', 'R5_hybrid_selector'),
            ('secondary', 'R0_standard', 'R0_hybrid_selector')]
    for rep in ('R0', 'R5'):
        rows += [('secondary', f'{rep}_standard', f'{rep}_{name}')
                 for name in ('tree_rmse', 'tree_rank', 'leaf_mlp', 'quantile_mlp', 'pca_mlp', 'blend')]
        rows.append(('secondary', f'{rep}_tree_rmse', f'{rep}_tree_rank'))
    rows += [('secondary', f'R0_{name}', f'R5_{name}') for name in SUFFIXES]
    assert len(rows) == 24 and len(set((b, c) for _, b, c in rows)) == 24
    return rows


def paired_statistics(point, boot, min_rows):
    """All contrasts use the same draws; interaction retains covariance."""
    result = []
    for role, baseline, candidate in contrasts():
        if role == 'primary' and min_rows != 20:
            role = 'sensitivity'
        delta = point[candidate] - point[baseline]
        draws = boot[candidate] - boot[baseline]
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])
        result.append(dict(role=role, contrast=f'{candidate}-{baseline}', min_rows=min_rows,
            delta=float(delta), ci_low=float(lo), ci_high=float(hi),
            point_inside_interval=bool(lo <= delta <= hi),
            bootstrap_mean_minus_point=float(np.nanmean(draws)-delta)))
    delta = (point['R5_tree_rank']-point['R5_tree_rmse'])-(point['R0_tree_rank']-point['R0_tree_rmse'])
    draws = (boot.R5_tree_rank-boot.R5_tree_rmse)-(boot.R0_tree_rank-boot.R0_tree_rmse)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    result.append(dict(role='secondary_interaction', contrast='tree_ranking_gain_R5-minus-R0',
        min_rows=min_rows, delta=float(delta), ci_low=float(lo), ci_high=float(hi),
        point_inside_interval=bool(lo <= delta <= hi),
        bootstrap_mean_minus_point=float(np.nanmean(draws)-delta)))
    return result
