"""Read-only scientific artifact audit; no fitting or bootstrap generation."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'reports/followup3/fewshot'
ARMS = ('R0', 'R5', 'sequence_anchor')
BUDGETS = (0, 5, 10, 20)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    from scripts.followup3_fewshot import context
    frame, protocol, plan = context(OUT, OUT / 'protocol_replay_v2.json')
    lookup = frame.set_index('pair_id', drop=False)
    tables = {draw: [] for draw in range(3)}
    source_fits = adapters = checked_files = 0
    available, consumed, epochs = [], [], []
    for cell in plan['cells']:
        directory = OUT / 'cells' / f"cell_{cell['index']:03d}"
        done = read(directory / 'completed.json')
        assert done['protocol_hash'] == protocol['protocol_hash'] and done['status'] == 'passed'
        assert done['cell_index'] == cell['index']
        for name, digest in done['files'].items():
            assert sha(directory / name) == digest, str(directory / name)
            checked_files += 1
        source = lookup.loc[cell['source_ids']]
        query = lookup.loc[cell['query_ids']]
        support_ids = [p for ids in cell['support_ids'].values() for p in ids]
        support = lookup.loc[support_ids]
        assert set(source.split) == set(query.split) == set(support.split) == {'train'}
        for column in ('pair_id', 'cluster_id', 'hla_group'):
            assert not set(source[column]) & set(query[column])
            assert not set(source[column]) & set(support[column])
        assert not set(support.cluster_id) & set(query.cluster_id)
        assert not set(source.cluster_id) & set(cell['excluded_support_clusters'])
        assert not set(query.cluster_id) & set(cell['excluded_support_clusters'])
        table = pd.read_csv(directory / 'predictions.csv', float_precision='round_trip')
        assert table.pair_id.tolist() == cell['query_ids']
        for column in ('cluster_id', 'hla_group', 'hla_fold', 'peptide_fold', 'allele', 'split', 'y_log1p'):
            np.testing.assert_array_equal(table[column], query[column])
        members = read(directory / 'members.json')
        assert len(members) == 6
        assert sorted((m['arm'], m.get('encoding', ''), m.get('seed', -1)) for m in members) == sorted([
            ('R0', '', -1), ('R5', '', -1), ('sequence_anchor', 'onehot', 0),
            ('sequence_anchor', 'onehot', 1), ('sequence_anchor', 'blosum', 0), ('sequence_anchor', 'blosum', 1)])
        used = {arm: set() for arm in ARMS}
        for member in members:
            gradient = lookup.loc[member['gradient_ids']]
            stop = lookup.loc[member['stopping_ids']]
            used[member['arm']].update(member['gradient_ids'] + member['stopping_ids'])
            assert used[member['arm']] <= set(cell['source_ids'])
            for column in ('cluster_id', 'hla_group'):
                assert not set(gradient[column]) & set(stop[column])
            assert member['scaler_ids'] == (member['gradient_ids'] if member['kind'] == 'ridge' else [])
            for role, ids in [('gradient', member['gradient_ids']), ('stopping', member['stopping_ids'])]:
                key = role + '_label_sha256'
                if key in member:
                    assert hashlib.sha256(lookup.loc[ids, 'y_log1p'].to_numpy().tobytes()).hexdigest() == member[key]
            if member['kind'] == 'mlp':
                epochs.append(member['epochs'])
                assert member['epochs'] <= 300 and len(member['stopping_curve']) == member['epochs']
        assert {a: len(v) for a, v in used.items()} == done['source_label_union_by_arm']
        base = {}
        for arm in ('R0', 'R5'):
            with np.load(directory / f'{arm}_ridge.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['source_ids'], source.pair_id)
                base[arm] = saved['prediction'].astype(float)
        sequence = []
        for encoding in ('onehot', 'blosum'):
            for seed in (0, 1):
                with np.load(directory / f'sequence_{encoding}_{seed}.npz', allow_pickle=False) as saved:
                    sequence.append(saved['prediction'].astype(float))
        base['sequence_anchor'] = np.mean(sequence, axis=0)
        for arm in ARMS:
            np.testing.assert_array_equal(table[f'{arm}_k0'], base[arm][len(support):])
        adapter_rows = read(directory / 'adapters.json')
        assert len(adapter_rows) == 9 * len(cell['support_ids']) == done['adapter_fits']
        expected_keys = {(a, k, allele) for a in ARMS for k in BUDGETS[1:] for allele in cell['support_ids']}
        assert {(r['arm'], r['budget'], r['allele']) for r in adapter_rows} == expected_keys
        pos = {int(pid): i for i, pid in enumerate(support.pair_id)}
        with np.load(directory / 'adapters.npz', allow_pickle=False) as saved:
            for row in adapter_rows:
                ids = cell['support_ids'][row['allele']][:row['budget']]
                assert row['support_ids'] == ids and row['labels'] == row['budget']
                assert row['query_ids'] == query.loc[query.allele.eq(row['allele']), 'pair_id'].tolist()
                assert row['stopping_labels'] == row['tuning_labels'] == 0
                labels = lookup.loc[ids, 'y_log1p'].to_numpy()
                residual = labels - base[row['arm']][[pos[p] for p in ids]]
                np.testing.assert_array_equal(saved[row['key'] + '_offset'], np.asarray(residual.mean()))
                assert saved[row['key'] + '_dual'].shape == (row['budget'],)
        assert done['saved_models_predictions_replayed_exactly'] and done['saved_adapters_replayed_exactly']
        source_fits += len(members)
        adapters += len(adapter_rows)
        available.append(len(source)); consumed.append(len(used['sequence_anchor']))
        tables[cell['draw']].append(table)
    assert source_fits == 450 and adapters == plan['adapter_fits'] == 9180
    panels = {int(k): v for k, v in protocol['fixed_panels'].items()}
    points = []
    allele_scores = {}
    for draw, parts in tables.items():
        table = pd.concat(parts, ignore_index=True)
        expected = sorted(r['pair_id'] for r in plan['queries'] if r['draw'] == draw)
        assert sorted(table.pair_id) == expected and not table.pair_id.duplicated().any()
        for threshold, panel in panels.items():
            for arm in ARMS:
                for budget in BUDGETS:
                    rhos = []
                    for allele in panel:
                        part = table.loc[table.allele.eq(allele)]
                        assert len(part) >= threshold and part.y_log1p.nunique() > 1
                        p = part[f'{arm}_k{budget}'].to_numpy()
                        rho = float(spearmanr(part.y_log1p, p).statistic) if np.ptp(p) else 0.
                        allele_scores[(threshold, draw, arm, budget, allele)] = rho
                        rhos.append(rho)
                    points.append(dict(panel_min_rows=threshold, draw=draw, arm=arm, budget=budget, score=float(np.median(rhos))))
    independent = pd.DataFrame(points)
    published = pd.read_csv(OUT / 'scores_by_draw.csv')
    keys = ['panel_min_rows', 'draw', 'arm', 'budget']
    aligned = independent.merge(published, on=keys, validate='one_to_one', suffixes=('_audit', '_saved'))
    assert len(aligned) == 72
    np.testing.assert_allclose(aligned.score_audit, aligned.score_saved, atol=2e-14, rtol=0)
    curves = independent.groupby(['panel_min_rows', 'arm', 'budget']).score.mean()
    comparisons = pd.read_csv(OUT / 'comparisons.csv')
    from scripts.followup3_fewshot import comparison_specs
    contrasts = dict(comparison_specs())
    for mode in ('crossed', 'peptide', 'hla'):
        for threshold in panels:
            boot = pd.read_csv(OUT / f'bootstrap_{mode}_min{threshold}.csv', float_precision='round_trip')
            assert len(boot) == 2000 and np.isfinite(boot.to_numpy()).all()
            rows = comparisons.loc[comparisons['mode'].eq(mode) & comparisons.panel_min_rows.eq(threshold)]
            assert len(rows) == len(contrasts) and set(rows.contrast) == set(contrasts)
            for row in rows.itertuples():
                weights = contrasts[row.contrast]
                values = sum(boot[name] * weight for name, weight in weights.items())
                score = sum(curves.loc[(threshold, name.rsplit('_k', 1)[0], int(name.rsplit('_k', 1)[1]))] * weight for name, weight in weights.items())
                np.testing.assert_allclose([score, *np.quantile(values, [.025, .975, .0125, .9875])],
                    [row.delta, row.ci_low, row.ci_high, row.family_ci_low, row.family_ci_high], atol=2e-14, rtol=0)
    deletion_table = pd.read_csv(OUT / 'all_hla_group_deletions.csv')
    allele_group = frame.groupby('allele').hla_group.first().to_dict()
    for threshold, panel in panels.items():
        for group in sorted(frame.hla_group.unique()):
            retained = [a for a in panel if allele_group[a] != group]
            scores = {f'{arm}_k{budget}': float(np.mean([np.median([
                allele_scores[(threshold, draw, arm, budget, allele)] for allele in retained])
                for draw in range(3)])) for arm in ARMS for budget in BUDGETS}
            rows = deletion_table.loc[deletion_table.panel_min_rows.eq(threshold) & deletion_table.dropped_hla_group.eq(group)]
            assert set(rows.contrast) == set(contrasts) and len(rows) == len(contrasts)
            for row in rows.itertuples():
                np.testing.assert_allclose(row.delta, sum(scores[a]*w for a,w in contrasts[row.contrast].items()), atol=2e-14, rtol=0)
    primary_boot = pd.read_csv(OUT / 'bootstrap_crossed_min20.csv', float_precision='round_trip')
    delta = primary_boot.R5_k20 - primary_boot.R5_k0
    family = dict(primary='R5 adaptation20 minus zero-shot', point=float(curves.loc[(20,'R5',20)]-curves.loc[(20,'R5',0)]),
                  interval_level=1-0.05/3, quantiles=[1/120,119/120], ci_low=float(delta.quantile(1/120)),
                  ci_high=float(delta.quantile(119/120)), saved_draws_sha256=sha(OUT/'bootstrap_crossed_min20.csv'),
                  amendment_sha256=sha(ROOT/'reports/followup3/THIRD_PRIMARY_AMENDMENT.md'), original_draws_reused=True)
    result = dict(status='passed', protocol_hash=protocol['protocol_hash'], cells=75, source_fits=source_fits,
                  target_adapters=adapters, cell_payload_hashes_checked=checked_files, point_rows_checked=len(aligned),
                  comparison_rows_checked=len(comparisons), hla_deletion_rows_checked=len(deletion_table), available_source_rows_range=[min(available),max(available)],
                  sequence_consumed_source_rows_range=[min(consumed),max(consumed)], source_mlp_epochs_range=[min(epochs),max(epochs)],
                  primary_three_study_family=family, independent_refits=0, independent_bootstrap_regeneration=False,
                  replay_scope='Verified cloud exact-replay assertions, payload hashes, source predictions, target residual offsets and ancestry; no independent matrix replay or refitting.',
                  audit_source_sha256=sha(__file__))
    for name, value in [('independent_audit.json', result), ('primary_family3_interval.json', family)]:
        (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
