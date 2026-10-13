"""Frozen few-shot residual adaptation; metadata plans never inspect outcomes."""
from __future__ import annotations
import hashlib
import json
import numpy as np
import pandas as pd
from .round2_transfer import validate_frame, validate_partition

DRAWS = (0, 1, 2)
BUDGETS = (0, 5, 10, 20)
ARMS = ('R0', 'R5', 'sequence_anchor')
SALT = 'pepstab-fewshot-v1'
SOURCE_ALPHA = 100.
ADAPTER_ALPHA = 1.


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def make_plan(frame):
    """One support label per allele/cluster, fixed union exclusion at every k."""
    validate_frame(frame)
    for column, group in [('hla_fold', 'hla_group'), ('peptide_fold', 'cluster_id')]:
        if set(frame[column]) != set(range(5)) or frame.groupby(group)[column].nunique().ne(1).any():
            raise ValueError('five canonical fold assignments required')
    counts = frame.groupby('allele').cluster_id.nunique()
    candidates = sorted(counts[counts >= 40].index.tolist())
    if not candidates:
        raise ValueError('no label-blind support candidates')
    cells = []
    support_tables = []
    queries = []
    for draw in DRAWS:
        for hf in range(5):
            target = frame.loc[frame.hla_fold.eq(hf) & frame.allele.isin(candidates)]
            supports = {}
            excluded = set()
            for allele, part in target.groupby('allele', sort=True):
                representative = part.sort_values('pair_id').drop_duplicates('cluster_id').set_index('cluster_id')
                ordered = sorted(representative.index, key=lambda c: (
                    hashlib.sha256(f'{SALT}|draw={draw}|allele={allele}|cluster={int(c)}'.encode()).hexdigest(), int(c)))[:20]
                supports[allele] = [int(representative.loc[c, 'pair_id']) for c in ordered]
                excluded.update(map(int, ordered))
                support_tables.extend(dict(draw=draw, hla_fold=hf, allele=allele, support_order=i + 1,
                                           pair_id=pid, cluster_id=int(c)) for i, (pid, c) in enumerate(zip(supports[allele], ordered)))
            query = target.loc[~target.cluster_id.isin(excluded)]
            queries.extend(dict(draw=draw, pair_id=int(row.pair_id)) for row in query.itertuples())
            for pf in range(5):
                source = frame.loc[~frame.hla_fold.eq(hf) & ~frame.peptide_fold.eq(pf) & ~frame.cluster_id.isin(excluded)]
                held = query.loc[query.peptide_fold.eq(pf)]
                support = frame.loc[frame.pair_id.isin([p for ids in supports.values() for p in ids])]
                validate_roles(source, support, held)
                if source.empty or held.empty:
                    raise ValueError('empty frozen cell; no dropping or redraw')
                cells.append(dict(index=len(cells), draw=draw, hla_fold=hf, peptide_fold=pf,
                                  source_ids=source.pair_id.astype(int).tolist(), query_ids=held.pair_id.astype(int).tolist(),
                                  support_ids=supports, excluded_support_clusters=sorted(excluded)))
    return dict(schema='followup3_fewshot_plan.1', draws=list(DRAWS), budgets=list(BUDGETS),
                support_candidates=candidates, cells=cells, supports=support_tables, queries=queries,
                source_fits=450, source_ridge_fits=150, source_mlp_fits=300,
                adapter_fits=3 * 3 * 5 * len(candidates) * len(BUDGETS[1:]),
                adapter_labels_per_allele=[0, 5, 10, 20], target_tuning_labels=0, target_stopping_labels=0)


def validate_roles(source, support, query):
    """Target HLA overlap is intentional; target peptide-family overlap is not."""
    validate_partition(source, support, 'joint')
    validate_partition(source, query, 'joint')
    validate_frame(support); validate_frame(query)
    if set(support.cluster_id) & set(query.cluster_id):
        raise ValueError('support/query peptide-family overlap')
    if support.groupby('allele').cluster_id.nunique().ne(support.groupby('allele').size()).any():
        raise ValueError('support needs one measured label per allele/cluster')


def fit_adapter(Zsupport, residual, Zquery):
    """Fixed dimension-normalized ridge in its at-most-20-dimensional dual."""
    Zsupport, Zquery = np.asarray(Zsupport, dtype=np.float64), np.asarray(Zquery, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    if Zsupport.ndim != 2 or Zquery.ndim != 2 or Zsupport.shape[1] != Zquery.shape[1] or Zsupport.shape[1] < 1:
        raise ValueError('adapter feature dimensions differ')
    if residual.shape != (len(Zsupport),) or not len(Zsupport):
        raise ValueError('nonempty aligned support residuals required')
    if not all(np.isfinite(x).all() for x in (Zsupport, Zquery, residual)):
        raise ValueError('nonfinite adapter input')
    d = Zsupport.shape[1]
    mean = Zsupport.mean(axis=0); offset = float(residual.mean())
    centered = Zsupport - mean
    kernel = centered @ centered.T / d
    dual = np.linalg.solve(kernel + ADAPTER_ALPHA * np.eye(len(kernel)), residual - offset)
    update = ((Zquery - mean) @ centered.T / d) @ dual + offset
    return update, dict(dual=dual, mean=mean, offset=np.asarray(offset))


def fixed_panels(frame, plan):
    """Conventional variance filter, never model-performance-dependent choices."""
    lookup = frame.set_index('pair_id', drop=False)
    panels = {}
    for threshold in (20, 50):
        panel = set(plan['support_candidates'])
        for draw in DRAWS:
            query = lookup.loc[[r['pair_id'] for r in plan['queries'] if r['draw'] == draw]]
            good = query.groupby('allele').y_log1p.agg(['size', 'nunique'])
            panel &= set(good.index[(good['size'] >= threshold) & (good['nunique'] > 1)])
        if not panel:
            raise ValueError('empty common fixed assessment panel')
        panels[threshold] = sorted(panel)
    return panels


def conditional_scores(frame, tables, panels, *, n_boot=2000, seed=20261003, mode='crossed'):
    """Shared FULL-population cluster/group draws; average subset draw scores."""
    from .round2_reporting import _weighted_rho
    from .round2_transfer import _frequency_median
    if mode not in ('crossed', 'peptide', 'hla') or n_boot < 1:
        raise ValueError('invalid bootstrap scope')
    clusters = np.sort(frame.cluster_id.unique()); groups = np.sort(frame.hla_group.unique())
    rng = np.random.default_rng(seed)
    pc = np.ones((n_boot, len(clusters)), dtype=np.int32)
    hc = np.ones((n_boot, len(groups)), dtype=np.int32)
    for i in range(n_boot):
        if mode in ('crossed', 'peptide'):
            pc[i] = np.bincount(rng.choice(len(clusters), len(clusters), replace=True), minlength=len(clusters))
        if mode in ('crossed', 'hla'):
            hc[i] = np.bincount(rng.choice(len(groups), len(groups), replace=True), minlength=len(groups))
    points = []; results = {}; diagnostics = []
    for threshold, panel in panels.items():
        values = {}
        for draw in DRAWS:
            table = tables[draw]
            for arm in ARMS:
                for budget in BUDGETS:
                    name = f'{arm}_k{budget}'
                    rhos = np.empty((n_boot, len(panel))); point_rhos = []; pw = np.empty_like(rhos, dtype=np.int32)
                    for ai, allele in enumerate(panel):
                        part = table.loc[table.allele.eq(allele)]
                        y = part.y_log1p.to_numpy(); p = part[name].to_numpy()
                        ci = np.searchsorted(clusters, part.cluster_id.to_numpy())
                        gi = int(np.searchsorted(groups, part.hla_group.iloc[0]))
                        weights = pc[:, ci].astype(float) * hc[:, gi, None]
                        rhos[:, ai] = _weighted_rho(y, p, weights)
                        pw[:, ai] = hc[:, gi]
                        point_rhos.append(float(_weighted_rho(y, p, np.ones((1, len(y))))[0]))
                    point = float(np.median(point_rhos))
                    boot = np.asarray([_frequency_median(rhos[i], pw[i]) for i in range(n_boot)])
                    if not np.isfinite(boot).all():
                        raise ValueError('degenerate whole-panel bootstrap; no replacement')
                    points.append(dict(panel_min_rows=threshold, draw=draw, arm=arm, budget=budget, score=point))
                    values[(draw, name)] = boot
                    diagnostics.append(dict(panel_min_rows=threshold, draw=draw, arm=arm, budget=budget,
                        n_resamples_dropped_alleles=int((np.sum(np.isfinite(rhos) & (pw > 0), axis=1) < len(panel)).sum())))
        results[threshold] = {f'{arm}_k{k}': np.mean([values[(draw, f'{arm}_k{k}')] for draw in DRAWS], axis=0)
                              for arm in ARMS for k in BUDGETS}
    return pd.DataFrame(points), results, dict(mode=mode, n_boot=n_boot, seed=seed,
        sampling_population='full original training peptide clusters and all actual HLA groups',
        subset_draw_aggregation='mean of draw scores, never prediction ensemble', diagnostics=diagnostics)
