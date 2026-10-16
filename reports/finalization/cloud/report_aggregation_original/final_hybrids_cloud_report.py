"""Cloud operational copy: replay unchanged frozen hybrid science after portable phase closure.

No fitting, source amendment, candidate expansion or remote calls occur here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pepstab import evaluation, nested
from pepstab.boltz_embeddings import sha256_file
from pepstab.final_hybrid_reporting import COMPONENTS, SUFFIXES, replay_selection, paired_statistics
from pepstab.round2_nested import Ancestry, align_predictions, apply_affine, unlabelled
from pepstab.round2_reporting import cluster_bootstrap_scores
from scripts.final_ranking import load_features, atomic_csv


def historical_scores(protocol):
    references = protocol['reference_sha256']
    for name, digest in references.items():
        if sha256_file(ROOT/name) != digest:
            raise ValueError('historical descriptive reference changed')
    ranking_path = 'reports/finalization/ranking/scores.csv'
    sequence_path = 'reports/further_research/round2/results/main/pooled_scores.csv'
    if ranking_path not in references or sequence_path not in references:
        raise ValueError('descriptive reference sources must be frozen')
    ranking = pd.read_csv(ROOT/ranking_path)
    sequence = pd.read_csv(ROOT/sequence_path)
    results = []
    for minimum in (20, 50):
        for rep in ('R0', 'R5'):
            rows = ranking.loc[(ranking.arm == f'{rep}_mse') & (ranking.min_rows == minimum)]
            if len(rows) != 1:
                raise ValueError('historical neural reference coverage changed')
            results.append(dict(arm=f'historical_{rep}_mse30', min_rows=minimum,
                median_spearman=float(rows.iloc[0].median_spearman),
                n_alleles=int(rows.iloc[0].n_alleles), role='descriptive_only'))
        rows = sequence.loc[(sequence.procedure == 'sequence_selector') &
                            (sequence.scope == 'pooled_outer_held') & (sequence.panel_min_rows == minimum)]
        if len(rows) != 1 or int(rows.iloc[0].n_rows) != 19716:
            raise ValueError('historical sequence reference coverage changed')
        results.append(dict(arm='historical_sequence_selector', min_rows=minimum,
            median_spearman=float(rows.iloc[0].median_per_allele_spearman),
            n_alleles=int(rows.iloc[0].n_eligible_alleles), role='descriptive_only'))
    return results


def resolve_bundle(directory, name):
    path = Path(name)
    if not path.is_absolute():
        path = directory/path
    path = path.resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError('bundle path leaves completed component directory')
    return path


def bundle_runtime(path, representation, fold, role, inner_fold, stop, seed):
    receipt = json.loads((path/'completed.json').read_text())
    seconds, memory = receipt['total_fit_seconds'], receipt['peak_rss_bytes']
    elapsed, transform = receipt['bundle_seconds'], receipt['transform_seconds']
    if not np.isfinite([seconds,elapsed,transform]).all() or min(seconds,elapsed,transform) < 0 or type(memory) is not int or memory <= 0:
        raise ValueError('invalid measured bundle runtime/memory')
    details = receipt['component_details']
    if set(details) != set(COMPONENTS):
        raise ValueError('missing component runtime diagnostics')
    return dict(representation=representation, outer_fold=fold, role=role,
        inner_fold=inner_fold, stop_fold=stop, seed=seed, total_fit_seconds=seconds,
        bundle_seconds=elapsed, transform_seconds=transform,
        peak_rss_bytes=memory, component_details=details)


def audit_pilots(out, protocol, train, features):
    """Verify twelve engineering fits without computing their held-out scores."""
    from pepstab import final_hybrids as core
    outer = nested.outer_folds(train)
    fit = train.loc[outer != 0]
    inner = nested.inner_folds(fit, 3)
    call_fit, held = fit.loc[inner != 0], fit.loc[inner == 0]
    stops = nested.inner_folds(call_fit, 5)
    gradient, stopping = call_fit.loc[stops != 0], call_fit.loc[stops == 0]
    position = {int(pid): i for i, pid in enumerate(train.pair_id)}
    results = []
    for rep in ('R0','R5'):
        directory = Path(out)/'pilot'/'fold_0'/rep
        done = json.loads((directory/'completed.json').read_text())
        exit_receipt = json.loads((directory/'exit_receipt.json').read_text())
        expected = dict(status='complete', phase='pilot', protocol_hash=protocol['protocol_hash'],
            representation=rep, outer_fold=0, n_fits=6, engineering_only=True,
            held_scores_computed=False, reuse_into_scientific=False)
        if any(done.get(k) != v for k,v in expected.items()):
            raise ValueError('pilot completion identity/scope changed')
        if exit_receipt.get('status') != 'complete' or exit_receipt.get('completed_sha256') != sha256_file(directory/'completed.json'):
            raise ValueError('pilot exit receipt missing or inconsistent')
        for name, key in [('bundle_manifest.json','bundle_manifest_sha256'), ('runtime.json','runtime_sha256')]:
            if sha256_file(directory/name) != done[key]:
                raise ValueError('pilot artifact changed')
        manifest = json.loads((directory/'bundle_manifest.json').read_text())
        if len(manifest) != 1:
            raise ValueError('pilot bundle count changed')
        item = manifest[0]
        context = f'hybrids/pilot/outer/0/{rep}/inner/0/stop/0/seed/0'
        if (item['role'],item['inner_fold'],item['stop_fold'],item['seed'],item['representation'],item['context']) != ('inner',0,0,0,rep,context):
            raise ValueError('pilot member axes changed')
        path = resolve_bundle(directory, item['path'])
        if sha256_file(path/'completed.json') != item['completed_sha256']:
            raise ValueError('pilot bundle completion changed')
        arrays = {name:features[rep][[position[int(pid)] for pid in frame.pair_id]]
                  for name,frame in [('gradient',gradient),('stopping',stopping),('held',held)]}
        identity = core.member_identity(arrays,gradient,stopping,unlabelled(held),
            representation=rep, seed=0, context=context, protocol_hash=protocol['protocol_hash'],
            source_feature_hash=core.array_hash(features[rep]))
        if item['identity'] != identity:
            raise ValueError('pilot data ancestry changed')
        audit = core.audit_member_bundle(arrays,gradient,stopping,unlabelled(held),
            artifact_dir=path, protocol_hash=protocol['protocol_hash'], expected_identity=identity)
        if audit['n_fits'] != 6:
            raise ValueError('pilot actual fit count changed')
        results.append(bundle_runtime(path,rep,0,'engineering_pilot',0,0,0))
    return results


def audit_component(out, protocol, train, features, rep, fold):
    from pepstab import final_hybrids as core
    from scripts.final_hybrids import verify_component
    directory = Path(out)/f'fold_{fold}'/rep
    done = verify_component(directory, protocol, fold, rep)
    outer = nested.outer_folds(train)
    fit, held = train.loc[outer != fold], train.loc[outer == fold]
    inner = pd.read_csv(directory/'inner_predictions.csv')
    table = pd.read_csv(directory/'predictions.csv')
    if len(table) != len(held)*len(SUFFIXES) or set(table.arm) != {f'{rep}_{s}' for s in SUFFIXES}:
        raise ValueError('outer component/row coverage changed')
    for arm in table.arm.unique():
        rows = table.loc[table.arm == arm]
        align_predictions(held, rows)
        if rows.allele.tolist() != held.allele.tolist() or not rows.outer_fold.eq(fold).all():
            raise ValueError('outer metadata changed')
    choice = replay_selection(fit, inner)
    saved = json.loads((directory/'selection.json').read_text())
    if saved['blend_weight'] != choice['blend_weight'] or saved['hybrid_component'] != choice['hybrid_component']:
        raise ValueError('inner blend/selector decision does not replay')
    for key in ('blend_scores', 'component_scores'):
        if len(saved[key]) != len(choice[key]):
            raise ValueError('inner candidate coverage changed')
        for a, b in zip(saved[key], choice[key]):
            for identity in ('component', 'weight'):
                if a.get(identity) != b.get(identity):
                    raise ValueError('inner candidate order changed')
            np.testing.assert_allclose(a['fold_scores'], b['fold_scores'], rtol=0, atol=2e-14)
            np.testing.assert_allclose([a['mean'],a['se']], [b['mean'],b['se']], rtol=0, atol=2e-14)
    for key, value in choice['ranker_mapping'].items():
        np.testing.assert_allclose(saved['ranker_mapping'][key], value, rtol=1e-12, atol=1e-12)
    manifest = json.loads((directory/'bundle_manifest.json').read_text())
    if len(manifest) != 8:
        raise ValueError('expected eight six-fit bundles per component')
    position = {int(pid): i for i, pid in enumerate(train.pair_id)}
    X = lambda f: features[rep][[position[int(pid)] for pid in f.pair_id]]
    inner_folds = nested.inner_folds(fit, 3)
    expected = [(i, fit.loc[inner_folds != i], fit.loc[inner_folds == i]) for i in range(3)] + [(None, fit, held)]
    expected_ancestry = Ancestry()
    cursor, count, runtimes = 0, 0, []
    final = None
    for i, call_fit, call_held in expected:
        stopping_folds = nested.inner_folds(call_fit, 5)
        predictions = {name: [] for name in COMPONENTS}
        for stop, seed in ((0,0), (1,1)):
            item = manifest[cursor]; cursor += 1
            if (item['role'],item.get('inner_fold'),item['stop_fold'],item['seed'],item['representation']) != (
                'final' if i is None else 'inner', i, stop, seed, rep):
                raise ValueError('bundle order/member axes changed')
            gradient, stopping = call_fit.loc[stopping_folds != stop], call_fit.loc[stopping_folds == stop]
            context = f'hybrids/main/outer/{fold}/{rep}/'+('final/refit' if i is None else f'inner/{i}')+f'/stop/{stop}/seed/{seed}'
            if item['context'] != context:
                raise ValueError('bundle context differs from frozen scheduling')
            unlabeled = unlabelled(call_held)
            arrays = dict(gradient=X(gradient), stopping=X(stopping), held=X(call_held))
            path = resolve_bundle(directory, item['path'])
            if sha256_file(path/'completed.json') != item['completed_sha256']:
                raise ValueError('bundle completion changed')
            identity = core.member_identity(arrays, gradient, stopping, unlabeled,
                representation=rep, seed=seed, context=context,
                protocol_hash=protocol['protocol_hash'], source_feature_hash=core.array_hash(features[rep]))
            if item['identity'] != identity:
                raise ValueError('member identity differs from derived data roles')
            audit = core.audit_member_bundle(arrays, gradient, stopping, unlabeled,
                artifact_dir=path, protocol_hash=protocol['protocol_hash'], expected_identity=identity)
            if audit['n_fits'] != 6:
                raise ValueError('bundle fit count changed')
            count += audit['n_fits']
            runtimes.append(bundle_runtime(path,rep,fold,item['role'],i,stop,seed))
            for name in COMPONENTS:
                predictions[name].append(audit['predictions'][name])
                expected_ancestry.add(role='hybrid_member', fit=gradient, held=call_held,
                    stopping=None if name in ('tree_rmse','tree_rank') else stopping,
                    component=name, representation=rep, context=context, stop_fold=stop, seed=seed,
                    scaler_rows=expected_ancestry.rows(gradient), transform_rows=expected_ancestry.rows(gradient))
        aggregate = {name: np.mean(values, axis=0) for name, values in predictions.items()}
        if i is None:
            final = aggregate
        else:
            for name, values in aggregate.items():
                rows = inner.loc[(inner.component == name) & (inner.inner_fold == i)]
                np.testing.assert_allclose(align_predictions(call_held, rows), values, rtol=0, atol=2e-12)
    if count != 48:
        raise ValueError('component supervised fit count differs from 48')
    raw_final = pd.read_csv(directory/'raw_final_predictions.csv')
    if len(raw_final) != len(held)*6 or set(raw_final.component) != set(COMPONENTS):
        raise ValueError('raw final component coverage changed')
    for name, values in final.items():
        rows = raw_final.loc[raw_final.component == name]
        np.testing.assert_allclose(align_predictions(held, rows), values, rtol=0, atol=2e-12)
    for name in ('ranker_mapping','blend','hybrid_selector'):
        expected_ancestry.add(role=name, fit=fit, held=held, selection=fit, representation=rep)
    saved_ancestry = json.loads((directory/'ancestry.json').read_text())
    if saved_ancestry != expected_ancestry.as_dict():
        raise ValueError('complete gradient/stopping/transform/selection ancestry graph changed')
    ancestry_audit = expected_ancestry.audit(train, held)
    if saved['ranker_mapping_input_rows'] != expected_ancestry.rows(fit):
        raise ValueError('ranker mapping label ancestry changed')
    final['tree_rank'] = apply_affine(final['tree_rank'], choice['ranker_mapping'])
    w = choice['blend_weight']
    final['blend'] = (1-w)*final['standard'] + w*final['tree_rmse']
    final['hybrid_selector'] = final[choice['hybrid_component']].copy()
    for name, values in final.items():
        rows = table.loc[table.arm == f'{rep}_{name}']
        np.testing.assert_allclose(align_predictions(held, rows), values, rtol=1e-12, atol=2e-12)
    return table, choice, dict(representation=rep, outer_fold=fold, bundles=8,
                              new_supervised_fits=count, predictions_and_selection_replayed=True,
                              ancestry=ancestry_audit, runtime=runtimes)


def build_report(out, approved):
    # Replay uses the same numerical thread limit as fitting.
    with threadpool_limits(limits=2):
        return _build_report(out, approved)


def _build_report(out, approved):
    from scripts.final_hybrids import validate_frozen
    out = Path(out)
    train, protocol = validate_frozen(out)
    from pepstab.final_cloud_protocol import report_phase_receipts
    cloud_auth, controller_receipts = report_phase_receipts(out, approved)
    features, _ = load_features(train)
    pilot_runtime = audit_pilots(out, protocol, train, features)
    tables, selections, audits = [], [], []
    for rep in ('R0', 'R5'):
        for fold in range(5):
            table, choice, audit = audit_component(out, protocol, train, features, rep, fold)
            tables.append(table); audits.append(audit)
            selections.append(dict(representation=rep, outer_fold=fold,
                blend_weight=choice['blend_weight'], hybrid_component=choice['hybrid_component']))
    table = pd.concat(tables, ignore_index=True)
    predictions = {arm: align_predictions(train, table.loc[table.arm == arm]) for arm in table.arm.unique()}
    comparisons, scores, metadata, allele_tables, fold_scores = [], [], {}, [], []
    outer_folds = nested.outer_folds(train)
    for minimum in (20, 50):
        boot, meta = cluster_bootstrap_scores(train, predictions, min_rows=minimum, n_boot=2000, seed=20261003)
        if meta['n_degenerate_resamples']:
            raise ValueError('a resample loses the entire eligible panel')
        point = {}
        for arm, pred in predictions.items():
            result = evaluation.score(arm, train.allele, train.y_log1p, pred, meta['alleles'])
            point[arm] = float(result.median_spearman)
            allele_tables.append(result.per_allele_frame().rename_axis('allele').reset_index().assign(arm=arm, min_rows=minimum))
            scores.append(dict(arm=arm, min_rows=minimum, median_spearman=point[arm],
                mse_log1p=float(np.mean((pred-train.y_log1p.to_numpy())**2)),
                mae_log1p=float(np.mean(np.abs(pred-train.y_log1p.to_numpy()))), n_alleles=meta['n_alleles']))
            for fold in range(5):
                mask = outer_folds == fold
                held = train.loc[mask]
                panel = evaluation.eligible_alleles(held.allele, held.y_log1p, min_rows=minimum)
                value = float(evaluation.score(arm, held.allele, held.y_log1p, pred[mask], panel).median_spearman) if panel else None
                fold_scores.append(dict(arm=arm, outer_fold=fold, min_rows=minimum,
                    n_rows=len(held), n_alleles=len(panel), score_available=bool(panel), median_spearman=value))
        comparisons.extend(paired_statistics(point, boot, minimum))
        metadata[str(minimum)] = meta
        atomic_csv(out/f'bootstrap_min{minimum}.csv', boot)
    for name, value in [('predictions', table), ('scores', pd.DataFrame(scores)),
            ('comparisons', pd.DataFrame(comparisons)), ('selections', pd.DataFrame(selections)),
            ('per_allele_scores', pd.concat(allele_tables, ignore_index=True)),
            ('per_fold_scores', pd.DataFrame(fold_scores)),
            ('historical_descriptive_scores', pd.DataFrame(historical_scores(protocol)))]:
        atomic_csv(out/f'{name}.csv', value)
    primary = next(r for r in comparisons if r['role']=='primary' and r['min_rows']==20)
    bundle_times = [row for a in audits for row in a['runtime']]
    runtime_summary = dict(scientific_bundles=len(bundle_times), engineering_bundles=len(pilot_runtime),
        summed_scientific_learner_fit_seconds=sum(row['total_fit_seconds'] for row in bundle_times),
        summed_scientific_bundle_elapsed_seconds=sum(row['bundle_seconds'] for row in bundle_times),
        summed_scientific_transform_seconds=sum(row['transform_seconds'] for row in bundle_times),
        maximum_observed_process_peak_rss_bytes=max(row['peak_rss_bytes'] for row in bundle_times),
        engineering_runtime=pilot_runtime, scientific_runtime=bundle_times,
        controller_receipts=controller_receipts, cloud_conservative_accounted_liability_usd=cloud_auth['_accounting']['conservative_spend_usd'],
        cloud_current_phase_reservation_usd=cloud_auth['_accounting']['reserved_phase_usd'],
        cloud_accounting=cloud_auth['cloud_accounting'], backend='modal_cpu_sandbox',
        cloud_cost_interpretation='Conservative preceding-phase accounted liability and current phase reservation at authorization; neither field is actual invoiced spend. Final provider reconciliation is separate.',
        interpretation='Supervised learner fit times exclude transforms. Bundle elapsed times can overlap across workers; their sum is not wall time. RSS is the maximum observed process lifetime peak, not summed concurrent memory.')
    (out/'runtime_summary.json').write_text(json.dumps(runtime_summary,indent=2)+'\n')
    lines = ['# Boltz tree, blending and preprocessing follow-up', '',
        'Original-training development only. The original test was not reopened.', '',
        f"Primary R5 hybrid selector minus matched standard MLP: **{primary['delta']:+.4f}**; paired-cluster 95% interval **[{primary['ci_low']:+.4f}, {primary['ci_high']:+.4f}]**.", '',
        'All 480 supervised fits in 80 bundles were audited and their saved model predictions and inner selections replayed. The separately authorized 12-fit engineering pilot is excluded from assessment and never reused.', '',
        'Both representations use the same two-member schedule for inner and final predictions. Six fitted components cover standard MSE MLP, fixed-budget regression and ranking trees, tree-leaf MLP, quantile MLP and PCA MLP. The RMSE tree is shared with its leaf transform; blending and selection add no supervised fits.', '',
        'The MSE-first hybrid selector and five fixed blend weights are selected from three clean inner folds. The ranking tree uses a positive affine map learned from its own inner predictions before the outer refit; this transports scale between different fitting-set sizes. Raw inner rank scores are never calibrated on their own labels for blend selection.', '',
        'All 24 ordinary contrasts and one paired interaction share 2,000 whole-peptide-cluster draws on each 20/50-row panel. Secondary intervals are nominal exploratory intervals, not multiplicity-adjusted discoveries. Uncertainty conditions on fitted models and selection, excluding repeated selection and independent assay variation.', '',
        'Historical 30-member MLPs and the competitive sequence selector are descriptive references only. Fixed-budget trees and adaptively stopped MLPs differ in training recipe; representation dimensions and parameter counts also differ. These are comparisons of complete procedures, not causal attribution to structure, normalization or a learner alone.', '',
        'See [comparisons.csv](comparisons.csv), [scores.csv](scores.csv), [selections.csv](selections.csv), [runtime_summary.json](runtime_summary.json), [audit.json](audit.json) and [protocol.json](protocol.json).', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    audit = dict(status='passed', completed_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_hash=protocol['protocol_hash'], n_rows=len(train), new_supervised_fits=480,
        engineering_fits=12, original_test_reopened=False, component_audits=audits,
        bootstrap=metadata, conditional_development_only=True)
    (out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    return primary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=ROOT/'reports/finalization/hybrids')
    parser.add_argument('--approved', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_report(args.out_dir, args.approved), indent=2))
