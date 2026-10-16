"""Audit saved inverse/energy outputs without fitting or generating resamples."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
KEYS = ['allele', 'peptide', 'cluster_id', 'outer_fold']
INVERSE_ARMS = ('R0', 'R0_P', 'R0_PC', 'R5', 'R5_P', 'R5_PC')
ENERGY_ARMS = ('R0', 'R5', 'R0_E', 'R5_E')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def csv(path):
    return pd.read_csv(path, float_precision='round_trip')


def balanced(frame, n):
    """Independently reconstruct equal-target largest-cluster-first folds."""
    counts = frame.groupby('cluster_id').size().to_dict()
    totals = [0] * n
    assignment = {}
    for group in sorted(counts, key=lambda c: (-counts[c], int(c))):
        fold = min(range(n), key=lambda i: (totals[i], i))
        assignment[group] = fold
        totals[fold] += counts[group]
    return frame.cluster_id.map(assignment).to_numpy()


def scores(frame, values, minimum):
    """Unweighted average-tie Spearman and fixed-zero constant predictions."""
    names = tuple(values)
    matrix = np.column_stack([np.asarray(values[a], dtype=float) for a in names])
    assert matrix.shape == (len(frame), len(names)) and np.isfinite(matrix).all()
    y = frame.y_log1p.to_numpy(float)
    assert np.isfinite(y).all()
    panel, correlations = [], []
    for allele in sorted(frame.allele.unique()):
        idx = np.flatnonzero(frame.allele.to_numpy() == allele)
        if len(idx) < minimum or np.ptp(y[idx]) == 0:
            continue
        yr = rankdata(y[idx]); yr -= yr.mean()
        pr = rankdata(matrix[idx], axis=0); pr -= pr.mean(axis=0)
        denominator = np.sqrt(np.sum(yr*yr) * np.sum(pr*pr, axis=0))
        rho = np.divide(np.sum(yr[:, None]*pr, axis=0), denominator,
                        out=np.zeros(len(names)), where=denominator > 0)
        panel.append(allele); correlations.append(rho)
    assert panel, 'empty eligible panel'
    return dict(zip(names, np.median(correlations, axis=0))), panel


def fitting_roles(frame, fold, inner):
    fitting = frame.loc[frame.outer_fold.ne(fold)]
    if inner is None:
        return fitting, frame.loc[frame.outer_fold.eq(fold)]
    parts = balanced(fitting, 3)
    return fitting.loc[parts != inner], fitting.loc[parts == inner]


def audit_selection(frame, selections, arms, learner, folds):
    expected = {(f, a) for f in folds for a in arms}
    assert len(selections) == len(expected)
    assert {(s['outer_fold'], s['arm']) for s in selections} == expected
    chosen = {}
    for row in selections:
        fit = frame.loc[frame.outer_fold.ne(row['outer_fold'])].set_index('pair_id', drop=False)
        ids = row['fitting_pair_ids']
        assert len(ids) == len(set(ids)) == len(fit) and set(ids) == set(fit.index)
        fit = fit.loc[ids]
        key = 'alpha' if 'alpha' in row['choices'][0] else 'regularization'
        grid = [1., .01, .0001] if learner == 'mlp' else [10000., 100., 1.]
        assert [c[key] for c in row['choices']] == grid
        value, panel = scores(fit, {str(i): c['inner_predictions'] for i, c in enumerate(row['choices'])}, 20)
        replay = np.asarray([value[str(i)] for i in range(3)])
        np.testing.assert_allclose(replay, [c['score'] for c in row['choices']], atol=2e-13, rtol=0)
        assert all(c['n_eligible'] == len(panel) for c in row['choices'])
        selected = row.get('selected_regularization', row.get('selected_l2', row.get('selected_alpha')))
        # Verify exact recorded argmax/tie order; independent rank arithmetic
        # can differ at floating-point roundoff while matching every score.
        assert selected == grid[int(np.argmax([c['score'] for c in row['choices']]))]
        assert replay[grid.index(selected)] >= replay.max()-2e-13
        chosen[(row['outer_fold'], row['arm'])] = float(selected)
    return chosen


def audit_ancestry(frame, saved, arms, learner, folds, chosen):
    expected_keys = Counter()
    grid = [1., .01, .0001] if learner == 'mlp' else [10000., 100., 1.]
    members = [(0, 0), (1, 1)] if learner == 'mlp' else [(None, None)]
    for fold in folds:
        for arm in arms:
            for regularization, inner in [(r, i) for r in grid for i in range(3)] + [(chosen[(fold, arm)], None)]:
                for stop, seed in members:
                    expected_keys[(fold, arm, regularization, inner, stop, seed)] += 1
    observed = Counter()
    if isinstance(saved, list):
        assert learner == 'ridge'
        for row in saved:
            fold, arm, inner = row['outer_fold'], row['arm'], row['inner_fold']
            fit, held = fitting_roles(frame, fold, inner)
            assert set(row['fit_clusters']) == set(fit.cluster_id)
            assert set(row['held_clusters']) == set(held.cluster_id)
            assert row['scaler_fit'] == 'all fitting rows'
            observed[(fold, arm, float(row['alpha']), inner, None, None)] += 1
    else:
        assert len(saved['records']) == len(saved['members'])
        for digest, ids in saved['rowsets'].items():
            assert ids == sorted(set(ids))
            assert hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes()).hexdigest() == digest
            assert set(ids) <= set(frame.pair_id)
        for row in saved['records']:
            parts = row['context'].split('/')
            assert parts[1] == learner
            fold, arm = int(parts[2]), parts[3]
            inner = None if parts[4] == 'final' else int(parts[-1])
            regularization = chosen[(fold, arm)] if inner is None else float(parts[4])
            assert row['candidate'] == f'{arm}_{learner}_{regularization:g}'
            fit, held = fitting_roles(frame, fold, inner)
            stop, seed = (row['stop_fold'], row['seed']) if learner == 'mlp' else (None, None)
            assert (stop, seed) in members
            assert set(saved['rowsets'][row['prediction_rows']]) == set(held.pair_id)
            if learner == 'mlp':
                stop_parts = balanced(fit, 5)
                assert set(saved['rowsets'][row['stopping_rows']]) == set(fit.loc[stop_parts == stop, 'pair_id'])
                fit = fit.loc[stop_parts != stop]
            assert set(saved['rowsets'][row['fit_rows']]) == set(fit.pair_id)
            assert row['scaler_fit_rows'] == row['fit_rows']
            observed[(fold, arm, regularization, inner, stop, seed)] += 1
        by_member = Counter((r['context'], r.get('seed'), r.get('stop_fold')) for r in saved['members'])
        by_record = Counter((r['context'], r.get('seed'), r.get('stop_fold')) for r in saved['records'])
        assert by_member == by_record
        widths = {'R0':860,'R0_P':869,'R0_PC':878,'R5':1024,'R5_P':1033,'R5_PC':1042,'R0_E':889,'R5_E':1053}
        for member in saved['members']:
            assert member['n_features'] == widths[member['context'].split('/')[3]]
            if learner == 'mlp':
                assert 1 <= member['epochs'] <= 200 and 0 <= member['best_epoch'] <= member['epochs']
    assert observed == expected_keys, 'missing, duplicate or changed fit ancestry'
    return sum(observed.values())


def load_prediction_artifacts(directory, frame, study, learner, arms):
    root = directory / 'assessment' / learner
    locations = [root/'all'] if (root/'all/receipt.json').exists() else [root/f'fold_{f}' for f in range(5)]
    fitted_arms = arms if study == 'inverse' else ('R0_E', 'R5_E')
    tables, total_fits, receipt_hashes = [], 0, {}
    for loc in locations:
        receipt = read(loc/'receipt.json')
        folds = receipt['folds'] if study == 'inverse' else [int(loc.name.removeprefix('fold_'))]
        if study == 'inverse':
            assert receipt['learner'] == learner
            for name, extension in [('predictions','csv'),('selection','json'),('ancestry','json')]:
                assert sha(loc/f'{name}.{extension}') == receipt[name+'_sha256']
        else:
            for name, expected in receipt['content_sha256'].items():
                assert sha(loc/name) == expected
        selection = read(loc/'selection.json')
        chosen = audit_selection(frame, selection, fitted_arms, learner, folds)
        count = audit_ancestry(frame, read(loc/'ancestry.json'), fitted_arms, learner, folds, chosen)
        assert count == receipt['fits' if study == 'inverse' else 'new_fits']
        total_fits += count
        receipt_hashes[str(loc.relative_to(directory))] = sha(loc/'receipt.json')
        table = csv(loc/'predictions.csv')
        assert set(table) == set(KEYS) | set(arms)
        held = frame.loc[frame.outer_fold.isin(folds), KEYS]
        merged = held.merge(table, on=KEYS, how='outer', validate='one_to_one', indicator=True)
        assert merged._merge.eq('both').all() and len(merged) == len(held)
        tables.append(table)
    table = pd.concat(tables, ignore_index=True)
    assert len(table) == len(frame) and not table.duplicated(KEYS[:2]).any()
    joined = frame.merge(table, on=KEYS, how='outer', validate='one_to_one', indicator=True)
    assert joined._merge.eq('both').all()
    return joined.drop(columns='_merge'), total_fits, receipt_hashes


def contrast_weights(study):
    if study == 'inverse':
        pairs = [('R5_P','R5_PC'),('R5','R5_PC'),('R5','R5_P'),('R0_P','R0_PC'),('R0','R0_PC'),('R0','R0_P')]
        interaction = ('R0_context_gain','R5_context_gain')
        weights = {'R5_PC':1.,'R5_P':-1.,'R0_PC':-1.,'R0_P':1.}
    else:
        pairs = [('R5','R5_E'),('R0','R0_E')]
        interaction = ('R0_energy_gain','R5_energy_gain')
        weights = {'R5_E':1.,'R5':-1.,'R0_E':-1.,'R0':1.}
    return {**{(b,c):{c:1.,b:-1.} for b,c in pairs}, interaction:weights}


def audit_parser_recovery(inverse, frame):
    """Require the separately bound execution ancestry for this known repair."""
    recovery = ROOT/'reports/followup3/inverse_parser_recovery'
    manifest_path = recovery/'manifest.json'
    receipt_path = inverse/'parser_recovery_receipt.json'
    if not manifest_path.exists() and not receipt_path.exists():
        return {'present':False}
    manifest, receipt = read(manifest_path), read(receipt_path)
    body = dict(manifest); identity = body.pop('manifest_hash')
    digest = lambda value: hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert digest(body)==identity==receipt['manifest_hash']
    assert sha(manifest_path)==receipt['recovery_manifest_sha256']
    assert (ROOT/manifest['inverse_directory']).resolve()==inverse.resolve()
    assert receipt['status']=='passed' and receipt['original_scientific_sources_unchanged']
    assert sha(inverse/'protocol.json')==manifest['original_protocol_file_sha256']==receipt['original_protocol_file_sha256']
    assert receipt['wrapper_source_sha256']==manifest['source_sha256']
    for name,expected in manifest['source_sha256'].items():
        assert sha(ROOT/name)==expected
    inherited=manifest['inherited_ok_row_sha256'];completed=receipt['completed_row_sha256']
    assert inherited==receipt['inherited_ok_row_sha256'] and len(inherited)==4163
    expected_names={digest([a,p])+'.json' for a,p in frame[['allele','peptide']].itertuples(index=False,name=None)}
    assert len(completed)==4449 and set(completed)==expected_names
    missing={digest([r['allele'],r['peptide']])+'.json' for r in manifest['corrected_and_missing_rows']}
    assert len(missing)==286 and missing==set(completed)-set(inherited)
    assert receipt['corrected_and_missing_rows']==manifest['corrected_and_missing_rows']
    assert receipt['completed_valid_scores_recomputed']==0
    for name,expected in completed.items():
        path=inverse/'score_rows'/name
        assert sha(path)==expected
        if name in inherited:
            assert expected==inherited[name]
        row=read(path)
        assert row['status']=='ok' and digest([row['allele'],row['peptide']])+'.json'==name
    archive=recovery/'original_failed_attempt.json'
    assert sha(archive)==manifest['original_failed_sha256']==receipt['original_failed_attempt_sha256']
    failed=read(archive)
    assert (failed['allele'],failed['peptide'])==('HLA-B*56:01','YPAEITLTW') and failed['status']=='failed'
    assert manifest['original_failed_filename'] in missing
    assert sha(inverse/'scores.csv')==receipt['aggregate_scores_sha256']
    assert sha(inverse/'extraction_receipt.json')==receipt['extraction_receipt_sha256']
    check=read(recovery/'parse_check.json')
    assert check['manifest_hash']==identity
    assert check['metadata_and_backbone_checks_passed'] and not check['model_forward_executed']
    assert {c['allele'] for c in check['cases']}=={'HLA-B*56:01','HLA-B*57:02'} and len(check['cases'])==2
    for case in check['cases']:
        assert case['peptide']=='YPAEITLTW' and case['native_peptide_chain']=='C' and len(case['context_chains'])==2
        row=read(inverse/'score_rows'/(digest([case['allele'],case['peptide']])+'.json'))
        assert row['structure_sha256']==case['cif_sha256'] and row['metadata_sha256']==case['metadata_sha256']
    proof=read(recovery/'FAILED_STRUCTURE_INSPECTION.json')
    repaired=read(inverse/'score_rows'/manifest['original_failed_filename'])
    assert repaired['structure_sha256']==proof['cif_sha256'] and repaired['metadata_sha256']==proof['metadata_sha256']
    return dict(present=True,manifest_sha256=sha(manifest_path),receipt_sha256=sha(receipt_path),
                inherited_row_hashes_preserved=4163,corrected_or_missing_rows=286,
                completed_rows=4449,failed_attempt_preserved_sha256=sha(archive),frozen_scientific_sources_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=('inverse','energy'), required=True)
    parser.add_argument('--directory', type=Path)
    args = parser.parse_args()
    directory = args.directory or ROOT/'reports/followup3'/('inverse_execution_v2' if args.study=='inverse' else 'energy')
    from scripts.followup3_inverse import labelled_panel, read_panel
    if args.study == 'inverse':
        _, protocol = read_panel(directory)
        inverse = directory
    else:
        from scripts.followup3_energy import verify
        _, protocol = verify(directory)
        inverse = ROOT/protocol['inverse_directory']
    frame = labelled_panel(inverse)
    assert len(frame) == 4449 and set(frame.split) == {'train'}
    parser_recovery = audit_parser_recovery(inverse,frame)
    arms = INVERSE_ARMS if args.study == 'inverse' else ENERGY_ARMS
    comparisons = csv(directory/'comparisons.csv')
    pooled = csv(directory/'pooled_scores.csv')
    contrasts = contrast_weights(args.study)
    assert len(comparisons) == 4*len(contrasts)
    primary = ('R5_P','R5_PC') if args.study=='inverse' else ('R5','R5_E')
    fits, receipts, families, panels = {}, {}, [], {}
    for learner in ('mlp','ridge'):
        joined, fits[learner], receipts[learner] = load_prediction_artifacts(directory, frame, args.study, learner, arms)
        if args.study == 'energy':
            reference, _, _ = load_prediction_artifacts(inverse, frame, 'inverse', learner, INVERSE_ARMS)
            for arm in ('R0','R5'):
                lookup = reference.set_index(['allele','peptide'])[arm]
                np.testing.assert_array_equal(joined[arm], lookup.loc[list(zip(joined.allele,joined.peptide))])
        for minimum in (20,50):
            point, panel = scores(joined, {a:joined[a] for a in arms}, minimum)
            panels[str(minimum)] = panel
            saved_points = pooled.loc[pooled.learner.eq(learner)&pooled.min_rows.eq(minimum)]
            assert len(saved_points) == len(arms) and set(saved_points.arm) == set(arms)
            for row in saved_points.itertuples():
                np.testing.assert_allclose(row.median_spearman, point[row.arm], atol=2e-13, rtol=0)
            boot_path = directory/f'bootstrap_{learner}_min{minimum}.csv'
            boot = csv(boot_path)
            assert len(boot)==2000 and set(boot)==set(arms) and np.isfinite(boot.to_numpy()).all()
            saved = comparisons.loc[comparisons.learner.eq(learner)&comparisons.panel_min_rows.eq(minimum)]
            assert len(saved)==len(contrasts) and set(zip(saved.baseline,saved.candidate))==set(contrasts)
            for row in saved.itertuples():
                weights = contrasts[(row.baseline,row.candidate)]
                estimate = sum(point[a]*w for a,w in weights.items())
                distribution = sum(boot[a]*w for a,w in weights.items())
                np.testing.assert_allclose([row.delta,row.ci_low,row.ci_high],
                    [estimate,*np.quantile(distribution,[.025,.975])], atol=2e-13,rtol=0)
                if (row.baseline,row.candidate)==primary:
                    old_tail = .0125 if args.study=='inverse' else 1/120
                    np.testing.assert_allclose([row.family_ci_low,row.family_ci_high],np.quantile(distribution,[old_tail,1-old_tail]),atol=2e-13,rtol=0)
                    families.append(dict(learner=learner,panel_min_rows=minimum,primary=learner=='mlp' and minimum==20,
                        delta=estimate,ci_low=float(distribution.quantile(1/120)),ci_high=float(distribution.quantile(119/120)),
                        interval_level=1-.05/3,quantiles=[1/120,119/120],saved_draws_sha256=sha(boot_path)))
    assert fits == ({'mlp':600,'ridge':300} if args.study=='inverse' else {'mlp':200,'ridge':100})
    audit = dict(status='passed',study=args.study,protocol_hash=protocol['protocol_hash'],fit_counts=fits,
        rows=len(frame),arms=list(arms),panels=panels,comparison_rows_checked=len(comparisons),
        receipt_hashes=receipts,three_primary_intervals=families,independent_refits=0,bootstrap_regenerated=False,
        saved_weight_replay=False,parser_recovery=parser_recovery,audit_scope='Exact fold/stopping/scaler ancestry; saved inner selection and outer score arithmetic; shared saved-draw intervals; no matrix or weight replay.',
        amendment_sha256=sha(ROOT/'reports/followup3/THIRD_PRIMARY_AMENDMENT.md'),audit_source_sha256=sha(__file__))
    (directory/'independent_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    (directory/'primary_family3_interval.json').write_text(json.dumps(next(r for r in families if r['primary']),indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('panels','receipt_hashes')},indent=2))


if __name__ == '__main__':
    main()
