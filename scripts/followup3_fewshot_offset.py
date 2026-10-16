"""Metadata freeze and cloud-only exploratory offset attribution; no fitting."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import followup3_fewshot as original
from pepstab import followup3_fewshot as study
from pepstab.round2_reporting import _weighted_rho
from pepstab.round2_transfer import _frequency_median

OUT = ROOT / 'reports/followup3/fewshot_offset'
BASE = ROOT / 'reports/followup3/fewshot'
NAMES = ('R5_k0', 'R5_k20', 'R5_offset20')
N_BOOT = 2000
SEED = 20261003
OWN_SOURCES = ('scripts/followup3_fewshot_offset.py',
               'tests/test_followup3_fewshot_offset.py',
               'reports/followup3/fewshot_offset/SPECIFICATION.md')


def read(path):
    return json.loads(Path(path).read_text())


def source_closure():
    # Include static local imports, including imports inside load_frame/context.
    pending = list(original.SOURCES + OWN_SOURCES +
                   ('pepstab/__init__.py', 'tests/test_followup3_fewshot.py'))
    found = set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        found.add(name)
        if not name.endswith('.py'):
            continue
        package = name.split('/')[:-1]
        for node in ast.walk(ast.parse((ROOT / name).read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                parent = '.'.join(prefix + ([node.module] if node.module else []))
                modules = [parent] + [parent + '.' + alias.name for alias in node.names]
            for module in modules:
                if module.split('.')[0] not in ('pepstab', 'scripts', 'tests'):
                    continue
                path = module.replace('.', '/')
                for candidate in (path + '.py', path + '/__init__.py'):
                    if (ROOT / candidate).is_file() and candidate not in found:
                        pending.append(candidate)
    return sorted(found)


def input_paths(base, plan, protocol):
    paths = [base / name for name in (
        'protocol.json', 'protocol_replay_v2.json', 'SCALER_REPLAY_AMENDMENT.md',
        'task_manifest.json', 'scores_by_draw.csv', 'bootstrap_crossed_min20.csv',
        'bootstrap_crossed_min50.csv', 'audit.json', 'independent_audit.json',
        'primary_family3_interval.json')]
    paths.extend(ROOT / name for name in protocol['data_sha256'])
    paths.append(ROOT / 'reports/followup3/THIRD_PRIMARY_AMENDMENT.md')
    grouping = ROOT / 'reports/further_research/round2/transfer'
    paths.extend(grouping / name for name in read(grouping / 'protocol.json')['sha256'])
    for cell in plan['cells']:
        directory = base / 'cells' / f"cell_{cell['index']:03d}"
        paths.extend(directory / name for name in (
            'completed.json', 'predictions.csv', 'adapters.json', 'adapters.npz'))
    return sorted(set(paths))


def freeze(out=OUT, base=BASE):
    frame, prior, plan = original.context(base, base / 'protocol_replay_v2.json')
    panels = prior['fixed_panels']
    if panels['20'] != panels['50'] or len(panels['20']) != 68:
        raise ValueError('this diagnostic requires the identical original 68-allele panels')
    if len(plan['cells']) != 75 or len(frame) != 19716:
        raise ValueError('original population/cell coverage differs')
    for name in ('audit.json', 'independent_audit.json'):
        if read(base / name)['status'] != 'passed':
            raise ValueError('original audited result must precede post-hoc freeze')
    protocol = dict(schema='followup3_fewshot_offset_protocol.1',
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        status='frozen_exploratory_posthoc_no_paid_execution_authorization',
        chosen_after_original_positive_result=True, primary_family_changed=False,
        prior_protocol_hash=prior['protocol_hash'], plan_sha256=prior['plan_sha256'],
        frame_sha256=prior['frame_sha256'], panel=panels['20'], panel_aliases=[20, 50],
        arm='R5', budget=20, support_draws=list(study.DRAWS),
        uncertainty=dict(mode='crossed', n_boot=N_BOOT, seed=SEED,
            nominal_interval_level=.95, family_adjusted=False,
            sampling_population='full canonical peptide clusters and HLA groups',
            rng_call_order='each replicate: peptide choice then HLA choice'),
        source_sha256={name:original.sha(ROOT / name) for name in source_closure()},
        input_sha256={str(path.relative_to(ROOT)):original.sha(path)
                      for path in input_paths(base, plan, prior)})
    protocol['protocol_hash'] = study.digest(protocol)
    original.write_once(out / 'protocol.json', protocol)
    return dict(protocol_hash=protocol['protocol_hash'], status=protocol['status'],
                bound_input_files=len(protocol['input_sha256']), independent_fits=0)


def validate_freeze(out, base):
    protocol = read(out / 'protocol.json')
    identity = protocol.pop('protocol_hash')
    if study.digest(protocol) != identity:
        raise ValueError('diagnostic protocol identity changed')
    protocol['protocol_hash'] = identity
    for kind in ('source_sha256', 'input_sha256'):
        for name, expected in protocol[kind].items():
            if original.sha(ROOT / name) != expected:
                raise ValueError(f'frozen diagnostic {kind} changed: {name}')
    frame, prior, plan = original.context(base, base / 'protocol_replay_v2.json')
    if (prior['protocol_hash'] != protocol['prior_protocol_hash'] or
            prior['plan_sha256'] != protocol['plan_sha256'] or
            original.frame_hash(frame) != protocol['frame_sha256'] or
            any(prior['fixed_panels'][str(k)] != protocol['panel'] for k in (20, 50))):
        raise ValueError('original ancestry or fixed panel differs')
    expected_inputs = {str(p.relative_to(ROOT)) for p in input_paths(base, plan, prior)}
    if expected_inputs != set(protocol['input_sha256']):
        raise ValueError('bound input closure incomplete')
    return frame, prior, plan, protocol


def offset_cell(table, cell, adapter_rows, arrays, panel):
    """Add already saved support residual means; validate exact cell identities."""
    if table.pair_id.tolist() != cell['query_ids'] or table.pair_id.duplicated().any():
        raise ValueError('query-row identity/order differs')
    result = table.loc[table.allele.isin(panel), [
        'pair_id', 'cluster_id', 'hla_group', 'allele', 'y_log1p',
        'R5_k0', 'R5_k20']].copy()
    result['cell_index'] = cell['index']
    result['draw'] = cell['draw']
    result['peptide_fold'] = cell['peptide_fold']
    result['R5_offset20'] = np.nan
    ranks, offsets = [], []
    selected = [row for row in adapter_rows if row['arm'] == 'R5' and row['budget'] == 20]
    if len(selected) != len(cell['support_ids']) or len({row['allele'] for row in selected}) != len(selected):
        raise ValueError('R5 k20 adapter coverage/uniqueness differs')
    if {row['allele'] for row in selected} != set(cell['support_ids']):
        raise ValueError('R5 k20 adapter allele identities differ')
    for row in selected:
        allele = row['allele']
        query_ids = table.loc[table.allele.eq(allele), 'pair_id'].tolist()
        if (row['support_ids'] != cell['support_ids'][allele][:20] or
                row['query_ids'] != query_ids or row['labels'] != 20 or
                row['stopping_labels'] != 0 or row['tuning_labels'] != 0):
            raise ValueError('saved adapter support/query ancestry differs')
        saved = np.asarray(arrays[row['key'] + '_offset'])
        if saved.shape != () or not np.isfinite(saved):
            raise ValueError('saved offset must be one finite scalar')
        if allele not in panel:
            continue
        mask = result.allele.eq(allele)
        zero = result.loc[mask, 'R5_k0'].to_numpy()
        offset = zero + float(saved)
        if not np.isfinite(offset).all():
            raise ValueError('offset predictions must be finite')
        unchanged = np.array_equal(rankdata(zero), rankdata(offset))
        if not unchanged:
            raise ValueError('finite-precision offset altered within-cell ranks')
        result.loc[mask, 'R5_offset20'] = offset
        record = dict(cell_index=cell['index'], draw=cell['draw'], allele=allele,
                      peptide_fold=cell['peptide_fold'], query_rows=len(zero))
        ranks.append(dict(record, ranks_identical=True))
        offsets.append(dict(record, adapter_key=row['key'], support_labels=20, offset=float(saved)))
    if result.R5_offset20.isna().any():
        raise ValueError('offset query coverage incomplete')
    return result, ranks, offsets


def score_control(frame, tables, panel, *, n_boot=N_BOOT, seed=SEED):
    """Exact original crossed schedule, restricted to these three predictions."""
    clusters = np.sort(frame.cluster_id.unique())
    groups = np.sort(frame.hla_group.unique())
    rng = np.random.default_rng(seed)
    pc = np.ones((n_boot, len(clusters)), dtype=np.int32)
    hc = np.ones((n_boot, len(groups)), dtype=np.int32)
    for i in range(n_boot):
        pc[i] = np.bincount(rng.choice(len(clusters), len(clusters), replace=True), minlength=len(clusters))
        hc[i] = np.bincount(rng.choice(len(groups), len(groups), replace=True), minlength=len(groups))
    points, allele_points, distributions = [], [], {}
    for draw in study.DRAWS:
        table = tables[draw]
        for name in NAMES:
            rhos = np.empty((n_boot, len(panel)))
            pw = np.empty_like(rhos, dtype=np.int32)
            point_rhos = []
            for ai, allele in enumerate(panel):
                part = table.loc[table.allele.eq(allele)]
                if part.empty or part.hla_group.nunique() != 1:
                    raise ValueError('pooled panel allele coverage/group differs')
                y = part.y_log1p.to_numpy()
                prediction = part[name].to_numpy()
                ci = np.searchsorted(clusters, part.cluster_id.to_numpy())
                gi = int(np.searchsorted(groups, part.hla_group.iloc[0]))
                weights = pc[:, ci].astype(float) * hc[:, gi, None]
                rhos[:, ai] = _weighted_rho(y, prediction, weights)
                pw[:, ai] = hc[:, gi]
                point = float(_weighted_rho(y, prediction, np.ones((1, len(y))))[0])
                point_rhos.append(point)
                allele_points.append(dict(draw=draw, prediction=name, allele=allele,
                                         query_rows=len(part), score=point))
            boot = np.asarray([_frequency_median(rhos[i], pw[i]) for i in range(n_boot)])
            if not np.isfinite(boot).all():
                raise ValueError('degenerate whole-panel bootstrap; no replacement')
            points.append(dict(draw=draw, prediction=name, score=float(np.median(point_rhos))))
            distributions[(draw, name)] = boot
    boot = {name:np.mean([distributions[(draw, name)] for draw in study.DRAWS], axis=0)
            for name in NAMES}
    return pd.DataFrame(points), pd.DataFrame(allele_points), boot


def contrasts(points, boot):
    scores = points.groupby('prediction').score.mean().to_dict()
    pairs = [('full_minus_offset', 'R5_k20', 'R5_offset20'),
             ('offset_minus_zero', 'R5_offset20', 'R5_k0'),
             ('original_full_minus_zero', 'R5_k20', 'R5_k0')]
    rows, differences = [], {}
    for contrast, left, right in pairs:
        delta = scores[left] - scores[right]
        distribution = boot[left] - boot[right]
        low, high = np.quantile(distribution, [.025, .975])
        rows.append(dict(contrast=contrast, delta=delta, ci_low=float(low), ci_high=float(high),
            role='original_reference' if contrast.startswith('original') else 'exploratory_posthoc',
            interval_level=.95, family_adjusted=False))
        differences[contrast] = distribution
    np.testing.assert_allclose(rows[0]['delta'] + rows[1]['delta'], rows[2]['delta'], atol=2e-14, rtol=0)
    np.testing.assert_allclose(differences['full_minus_offset'] + differences['offset_minus_zero'],
                               differences['original_full_minus_zero'], atol=2e-14, rtol=0)
    return pd.DataFrame(rows), differences


def report(out=OUT, base=BASE):
    original.require_cloud()
    started = time.monotonic()
    frame, prior, plan, protocol = validate_freeze(out, base)
    if len(frame) != 19716 or frame.cluster_id.nunique() != 3587 or frame.hla_group.nunique() != 35:
        raise ValueError('full canonical bootstrap population differs')
    if protocol['uncertainty']['n_boot'] != N_BOOT or protocol['uncertainty']['seed'] != SEED:
        raise ValueError('diagnostic bootstrap recipe differs')
    # A write-once attempt prevents accidental repeated calculation/overwriting.
    original.write_once(out / 'attempt.json', dict(protocol_hash=protocol['protocol_hash'],
        started_at_utc=datetime.now(timezone.utc).isoformat()))
    lookup = frame.set_index('pair_id', drop=False)
    parts = {draw:[] for draw in study.DRAWS}
    ranks, offsets = [], []
    for cell in plan['cells']:
        directory = base / 'cells' / f"cell_{cell['index']:03d}"
        done = read(directory / 'completed.json')
        if (done['status'] != 'passed' or done['protocol_hash'] != prior['protocol_hash'] or
                done['cell_index'] != cell['index']):
            raise ValueError('original cell completion ancestry differs')
        for name in ('predictions.csv', 'adapters.json', 'adapters.npz'):
            if original.sha(directory / name) != done['files'][name]:
                raise ValueError('original cell payload identity differs')
        table = pd.read_csv(directory / 'predictions.csv', float_precision='round_trip')
        for column in ('cluster_id', 'hla_group', 'allele', 'y_log1p'):
            np.testing.assert_array_equal(table[column], lookup.loc[cell['query_ids'], column])
        with np.load(directory / 'adapters.npz', allow_pickle=False) as saved:
            part, rank, offset = offset_cell(table, cell, read(directory / 'adapters.json'),
                                              saved, protocol['panel'])
        parts[cell['draw']].append(part)
        ranks.extend(rank)
        offsets.extend(offset)
    tables = {draw:pd.concat(rows).sort_values('pair_id').reset_index(drop=True)
              for draw, rows in parts.items()}
    for draw, table in tables.items():
        expected = sorted(row['pair_id'] for row in plan['queries'] if row['draw'] == draw
                          and lookup.loc[row['pair_id'], 'allele'] in protocol['panel'])
        if table.pair_id.tolist() != expected or table.pair_id.duplicated().any():
            raise ValueError('pooled query-row coverage differs')
        if table.groupby('allele').peptide_fold.nunique().to_dict() != {a:5 for a in protocol['panel']}:
            raise ValueError('each panel allele requires all five query peptide cells')
    points, allele_points, computed = score_control(frame, tables, protocol['panel'])
    if len(ranks) != len(protocol['panel']) * 5 * len(study.DRAWS):
        raise ValueError('cell/allele rank check coverage differs')
    saved_boot = pd.read_csv(base / 'bootstrap_crossed_min20.csv', float_precision='round_trip')
    alias_boot = pd.read_csv(base / 'bootstrap_crossed_min50.csv', float_precision='round_trip')
    saved_points = pd.read_csv(base / 'scores_by_draw.csv', float_precision='round_trip')
    for name in NAMES[:2]:
        if len(saved_boot) != N_BOOT:
            raise ValueError('saved bootstrap length differs')
        np.testing.assert_array_equal(saved_boot[name], alias_boot[name])
        np.testing.assert_allclose(computed[name], saved_boot[name], atol=2e-14, rtol=0)
        budget = int(name.split('_k')[1])
        rows = saved_points.loc[saved_points.panel_min_rows.eq(20) &
                                saved_points.arm.eq('R5') & saved_points.budget.eq(budget)].sort_values('draw')
        np.testing.assert_array_equal(rows.draw, np.asarray(study.DRAWS))
        np.testing.assert_allclose(points.loc[points.prediction.eq(name)].score,
                                   rows.score, atol=2e-14, rtol=0)
    boot = {name:saved_boot[name].to_numpy() for name in NAMES[:2]}
    boot['R5_offset20'] = computed['R5_offset20']
    comparisons, differences = contrasts(points, boot)
    by_draw = points.pivot(index='draw', columns='prediction', values='score')
    by_draw['full_minus_offset'] = by_draw.R5_k20 - by_draw.R5_offset20
    by_draw['offset_minus_zero'] = by_draw.R5_offset20 - by_draw.R5_k0
    by_draw['original_full_minus_zero'] = by_draw.R5_k20 - by_draw.R5_k0
    np.testing.assert_allclose(by_draw.full_minus_offset + by_draw.offset_minus_zero,
                               by_draw.original_full_minus_zero, atol=2e-14, rtol=0)
    pd.concat(list(tables.values()), ignore_index=True).to_csv(out / 'offset_predictions.csv', index=False)
    pd.DataFrame(offsets).to_csv(out / 'offsets_by_cell_allele.csv', index=False)
    pd.DataFrame(ranks).to_csv(out / 'rank_invariance.csv', index=False)
    points.to_csv(out / 'scores_by_draw.csv', index=False)
    by_draw.reset_index().to_csv(out / 'comparisons_by_draw.csv', index=False)
    allele_points.to_csv(out / 'scores_by_allele_draw.csv', index=False)
    pd.DataFrame(dict(bootstrap_index=np.arange(N_BOOT), **boot, **differences)).to_csv(
        out / 'bootstrap_crossed.csv', index=False)
    comparisons.to_csv(out / 'comparisons.csv', index=False)
    audit = dict(status='passed', schema='followup3_fewshot_offset_audit.1',
        protocol_hash=protocol['protocol_hash'], prior_protocol_hash=prior['protocol_hash'],
        original_source_inputs_unchanged=True, cells=75, alleles=len(protocol['panel']),
        support_draws=3, rank_checks=len(ranks), all_within_cell_ranks_identical=True,
        independently_recomputed_original_bootstrap_columns_aligned=True,
        original_bootstrap_columns_reused=True, point_and_all_bootstrap_differences_telescope=True,
        posthoc=True, primary_family_changed=False, independent_fits=0,
        panel_aliases=[20, 50], panel_calculated_once=True, n_boot=N_BOOT, seed=SEED,
        seconds=time.monotonic() - started)
    lines = ['# Exploratory post-hoc offset attribution', '',
        'Chosen after the positive original R5 k20 result. No primary-family change. '
        'Intervals below are nominal unadjusted 95%, conditional on saved fits and support draws.', '']
    for row in comparisons.itertuples():
        lines.append(f'- {row.contrast}: {row.delta:+.6f} [{row.ci_low:+.6f}, {row.ci_high:+.6f}].')
    lines += ['', 'The ≥20 and ≥50 panels are the same 68 alleles and were computed once. '
        'All cell/allele ranks were exactly unchanged by the scalar shift. '
        'Pooled shifts can alter ranks because offsets differ between source cells. '
        'The two components telescope at the point and every paired bootstrap replicate.', '',
        'An offset-only gain represents alignment of cell-specific source predictions using '
        'support residual means. A full-minus-offset gain is attributable to the feature-dependent '
        'residual term within this saved prediction comparison. Neither establishes a biological '
        'mechanism, independent replication, neural fine-tuning benefit, or prospective assay savings.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    outputs = ('offset_predictions.csv', 'offsets_by_cell_allele.csv', 'rank_invariance.csv',
               'scores_by_draw.csv', 'comparisons_by_draw.csv', 'scores_by_allele_draw.csv', 'bootstrap_crossed.csv',
               'comparisons.csv', 'REPORT.md', 'attempt.json')
    # Recheck the freeze after reporting as well, before attesting input stability.
    for kind in ('source_sha256', 'input_sha256'):
        for name, expected in protocol[kind].items():
            if original.sha(ROOT / name) != expected:
                raise ValueError(f'frozen input changed during report: {name}')
    audit['output_sha256'] = {name:original.sha(out / name) for name in outputs}
    audit['protocol_file_sha256'] = original.sha(out / 'protocol.json')
    original.write_once(out / 'audit.json', audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'report'])
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--base', type=Path, default=BASE)
    args = parser.parse_args()
    result = freeze(args.out, args.base) if args.mode == 'freeze' else report(args.out, args.base)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
