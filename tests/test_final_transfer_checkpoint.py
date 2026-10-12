"""Synthetic populations only; no real data, cloud calls, or worker launches."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pepstab import final_transfer as ft
from pepstab import final_transfer_checkpoint as cp
from pepstab.round2_nested import unlabelled


def population(groups=15, peptides=30):
    return pd.DataFrame([dict(pair_id=g * peptides + p, cluster_id=p, hla_group=g, allele=f'A{g}',
        split='train', hla_fold=g % 5, peptide_fold=p % 5, peptide='A' * 8 + 'ACDEFGHIKLMNPQRSTVWY'[p % 20],
        hla_pseudoseq='A' * 34, y_log1p=float(p % 7 + g / 20), thalf_hours=float(p % 7))
        for g in range(groups) for p in range(peptides)])


def axes(regime='hla'):
    frame = population()
    _, cell, fit, held = next(item for item in ft.outer_cells(frame) if item[0] == regime)
    return frame, cell, fit, held


def predictor(frame, cache, regime='hla', **kwargs):
    return cp.CheckpointPredictor(frame, ft.TransferAncestry(regime), cache=cache,
                                  sources=kwargs.pop('sources', {'synthetic/source.py': 'abc'}), **kwargs)


def without_timing(members):
    result = deepcopy(members)
    for item in result:
        item.pop('fit_seconds')
        if 'config' in item:
            item['config'] = json.loads(json.dumps(item['config']))
    return result


def forbidden(*args, **kwargs):
    raise AssertionError('replay attempted model fitting')


def test_real_synthetic_ridge_fresh_parity_and_replay_without_fitting(tmp_path, monkeypatch):
    frame, _, fit, held = axes()
    candidate = next(c for c in ft.CANDIDATES if c['learner'] == 'ridge')
    original = ft.ControlPredictor(frame, ft.TransferAncestry('hla'))
    expected = original(fit, unlabelled(held), candidate, 'ridge', final=True)
    fresh_callbacks = []
    fresh = predictor(frame, tmp_path / 'cache', callback=lambda _: fresh_callbacks.append(deepcopy(fresh.ancestry.as_dict())))
    actual = fresh(fit, unlabelled(held), candidate, 'ridge', final=True)
    np.testing.assert_array_equal(actual, expected)
    assert without_timing(fresh.members) == without_timing(original.members)
    assert fresh.ancestry.as_dict() == original.ancestry.as_dict()
    calls = fresh.completed_inventory()
    assert len(calls) == 1
    seen = []
    replay_callbacks = []
    def callback(member):
        seen.append(member)
        replay_callbacks.append(deepcopy(replay.ancestry.as_dict()))
    replay = predictor(frame, tmp_path / 'cache', callback=callback)
    monkeypatch.setattr(ft.Ridge, 'fit', forbidden)
    np.testing.assert_array_equal(replay(fit, unlabelled(held), candidate, 'ridge', final=True), expected)
    assert replay.members == fresh.members == seen
    assert replay.ancestry.as_dict() == fresh.ancestry.as_dict()
    assert replay_callbacks == fresh_callbacks
    np.testing.assert_array_equal(replay.member_predictions[0]['predictions'], fresh.member_predictions[0]['predictions'])
    assert replay.ancestry.audit(frame, fit, held)['outer_exclusion']


def test_real_synthetic_tree_fresh_parity_and_replay(tmp_path, monkeypatch):
    # Bounded synthetic learner runtime; both paths receive exactly this same recipe.
    monkeypatch.setitem(ft.TREE_PARAMS, 'iterations', 3)
    monkeypatch.setitem(ft.TREE_PARAMS, 'od_wait', 2)
    frame, _, fit, held = axes()
    candidate = ft.CANDIDATES[0]
    original = ft.ControlPredictor(frame, ft.TransferAncestry('hla'))
    expected = original(fit, unlabelled(held), candidate, 'tree')
    fresh_callbacks = []
    fresh = predictor(frame, tmp_path / 'cache', callback=lambda _: fresh_callbacks.append(deepcopy(fresh.ancestry.as_dict())))
    actual = fresh(fit, unlabelled(held), candidate, 'tree')
    np.testing.assert_array_equal(actual, expected)
    assert without_timing(fresh.members) == without_timing(original.members)
    assert fresh.ancestry.as_dict() == original.ancestry.as_dict()
    replay_callbacks = []
    replay = predictor(frame, tmp_path / 'cache', callback=lambda _: replay_callbacks.append(deepcopy(replay.ancestry.as_dict())))
    monkeypatch.setattr(ft, 'tree_fit', forbidden)
    np.testing.assert_array_equal(replay(fit, unlabelled(held), candidate, 'tree'), expected)
    assert len(replay.members) == 6
    assert replay_callbacks == fresh_callbacks


@pytest.fixture
def cached(tmp_path):
    frame, _, fit, held = axes()
    candidate = next(c for c in ft.CANDIDATES if c['learner'] == 'ridge')
    cache = tmp_path / 'cache'
    first = predictor(frame, cache)
    first(fit, unlabelled(held), candidate, 'call')
    return frame, fit, held, candidate, cache, first


@pytest.mark.parametrize('change', ['fit_label', 'fit_order', 'held_order', 'held_metadata', 'source', 'versions', 'candidate', 'context', 'final'])
def test_identity_changes_rejected_before_any_model_fit(cached, monkeypatch, change):
    frame, fit, held, candidate, cache, _ = cached
    fit, held, candidate = fit.copy(), unlabelled(held), deepcopy(candidate)
    sources, context, final = {'synthetic/source.py': 'abc'}, 'call', False
    if change == 'fit_label':
        fit.loc[fit.index[0], 'y_log1p'] += 1
    elif change == 'fit_order':
        fit = fit.iloc[::-1]
    elif change == 'held_order':
        held = held.iloc[::-1]
    elif change == 'held_metadata':
        held.loc[held.index[0], 'peptide'] = 'C' * 9
    elif change == 'source':
        sources['synthetic/source.py'] = 'changed'
    elif change == 'candidate':
        candidate = next(c for c in ft.CANDIDATES if c['learner'] == 'ridge' and c != candidate)
    elif change == 'context':
        context = 'different'
    elif change == 'final':
        final = True
    replay = predictor(frame, cache, sources=sources)
    if change == 'versions':
        replay.versions['python'] = 'changed'
    monkeypatch.setattr(ft.Ridge, 'fit', forbidden)
    with pytest.raises(ValueError, match='identity changed'):
        replay(fit, held, candidate, context, final=final)
    assert replay.members == [] and replay.ancestry.records == []


def test_held_labels_forbidden_and_population_held_label_invariant(cached, monkeypatch):
    frame, fit, held, candidate, cache, first = cached
    replay = predictor(frame, cache)
    with pytest.raises(ValueError, match='assessment labels forbidden'):
        replay(fit, held, candidate, 'call')
    changed = frame.copy()
    changed.loc[changed.pair_id.isin(held.pair_id), ['y_log1p', 'thalf_hours']] = -999
    replay = predictor(changed, cache)
    monkeypatch.setattr(ft.Ridge, 'fit', forbidden)
    result = replay(fit, unlabelled(held), candidate, 'call')
    np.testing.assert_array_equal(result, first.member_predictions[0]['predictions'])


def test_corrupted_archive_fails_closed(cached):
    frame, _, _, _, cache, _ = cached
    path = cache / 'call_0000/arrays.npz'
    path.write_bytes(path.read_bytes()[:-10])
    with pytest.raises(ValueError, match='payload hash'):
        predictor(frame, cache)


@pytest.mark.parametrize('failure', ['model', 'callback', 'commit'])
def test_uncertain_call_stays_pending_and_cannot_retry(tmp_path, monkeypatch, failure):
    frame, _, fit, held = axes()
    candidate = next(c for c in ft.CANDIDATES if c['learner'] == 'ridge')
    cache = tmp_path / 'cache'
    def fail(*args, **kwargs):
        raise RuntimeError('simulated interruption')
    if failure == 'model':
        monkeypatch.setattr(ft.Ridge, 'fit', fail)
    first = predictor(frame, cache, callback=fail if failure == 'callback' else None)
    if failure == 'commit':
        real_write = cp.exclusive_json
        def write(path, value):
            if Path(path).name == 'completed.json':
                fail()
            return real_write(path, value)
        monkeypatch.setattr(cp, 'exclusive_json', write)
    with pytest.raises(RuntimeError, match='interruption'):
        first(fit, unlabelled(held), candidate, 'call')
    assert (cache / 'call_0000/pending.json').exists()
    with pytest.raises(ValueError, match='cannot continue'):
        first(fit, unlabelled(held), candidate, 'call')
    with pytest.raises(ValueError, match='pending checkpoint'):
        predictor(frame, cache)


def test_resigned_bad_ancestry_rejected_before_restore(cached):
    frame, fit, held, candidate, cache, _ = cached
    directory = cache / 'call_0000'
    path = directory / 'call.json'
    record = json.loads(path.read_text())
    record['ancestry']['records'][0]['fit_rows'] = record['ancestry']['records'][0]['prediction_rows']
    path.write_text(json.dumps(record))
    receipt = json.loads((directory / 'completed.json').read_text())
    receipt['sha256']['call.json'] = cp.digest(path)
    (directory / 'completed.json').write_text(json.dumps(receipt))
    replay = predictor(frame, cache)
    with pytest.raises(ValueError, match='ancestry changed'):
        replay(fit, unlabelled(held), candidate, 'call')
    assert replay.members == [] and replay.ancestry.records == []


def test_replay_preserves_existing_wrapper_ancestry_and_rejects_duplicate_context(cached):
    frame, fit, held, candidate, cache, _ = cached
    replay = predictor(frame, cache)
    replay.ancestry.add('synthetic_wrapper', fit, unlabelled(held))
    original = deepcopy(replay.ancestry.records[0])
    replay(fit, unlabelled(held), candidate, 'call')
    assert replay.ancestry.records[0] == original
    assert len(replay.ancestry.records) == 2
    with pytest.raises(ValueError, match='duplicate checkpoint context'):
        replay(fit, unlabelled(held), candidate, 'call')


@pytest.fixture
def mock_learners(monkeypatch):
    class Model:
        def __init__(self, *args, **kwargs):
            self.best_epoch_, self.n_epochs_ = 2, 3
        def fit(self, *args):
            self.mean = float(np.mean(args[1]))
            return self
        def predict(self, X):
            return X[:, :180] @ np.arange(180, dtype=float) + self.mean
    monkeypatch.setattr(ft, 'Ridge', Model)
    monkeypatch.setattr(ft, 'MLPRegressor', Model)
    def tree(X, y, a, Xstop, ystop, astop, Xheld, candidate, seed):
        loss = 'PairLogit' if candidate['learner'] == 'pairlogit' else 'RMSE'
        return Xheld[:, :180] @ np.arange(180, dtype=float) + float(np.mean(y)), dict(
            config=dict(ft.TREE_PARAMS, depth=candidate['depth'], random_seed=seed, loss_function=loss, eval_metric=loss),
            best_iteration=2, tree_count=3)
    monkeypatch.setattr(ft, 'tree_fit', tree)


def test_full_joint_assessor_fresh_and_replay_parity(tmp_path, monkeypatch, mock_learners):
    frame, _, fit, held = axes('joint')
    reused = dict(predictions={p: held.cluster_id.to_numpy(dtype=float) for p in ft.REUSED_ARMS},
                  scores={'original': .2, 'mlp_centred': .3}, selection={'synthetic': True})
    original = ft.ControlPredictor(frame, ft.TransferAncestry('joint'))
    expected = ft.assess_cell(fit, held, 'joint', original, original.ancestry, reused, min_rows=2)
    fresh = predictor(frame, tmp_path / 'cache', 'joint')
    actual = ft.assess_cell(fit, held, 'joint', fresh, fresh.ancestry, reused, min_rows=2)
    assert actual[1] == expected[1]
    pd.testing.assert_frame_equal(actual[2], expected[2])
    for index in (0, 3):
        for key in expected[index]:
            np.testing.assert_array_equal(actual[index][key], expected[index][key])
    assert without_timing(fresh.members) == without_timing(original.members)
    assert fresh.ancestry.as_dict() == original.ancestry.as_dict()
    assert len(fresh.members) == 400 and len(fresh.completed_inventory()) == 104
    replay = predictor(frame, tmp_path / 'cache', 'joint')
    monkeypatch.setattr(ft, 'tree_fit', forbidden)
    monkeypatch.setattr(ft.Ridge, 'fit', forbidden)
    monkeypatch.setattr(ft.MLPRegressor, 'fit', forbidden)
    saved = ft.assess_cell(fit, held, 'joint', replay, replay.ancestry, reused, min_rows=2)
    assert saved[1] == actual[1]
    pd.testing.assert_frame_equal(saved[2], actual[2])
    assert replay.members == json.loads(json.dumps(fresh.members))
    assert replay.ancestry.as_dict() == fresh.ancestry.as_dict()
    assert replay.ancestry.audit(frame, fit, held)['cross_quadrant_exclusion']


def test_worker_legacy_report_audit_and_cache_tamper_gate(tmp_path, monkeypatch, mock_learners):
    from scripts import final_transfer as runner, final_transfer_report as report
    from scripts import final_transfer_checkpoint_worker as worker
    from scripts import final_transfer_durable as durable
    frame, cell, fit, held = axes('joint')
    protocol = dict(protocol_hash=durable.PROTOCOL_HASH, legacy_directory=str(tmp_path / 'legacy'))
    reused = dict(predictions={p: held.cluster_id.to_numpy(dtype=float) for p in ft.REUSED_ARMS},
        scores={'original': .2, 'mlp_centred': .3}, selection={'synthetic': True}, evidence_sha256={'mock': 'identity'})
    auth = dict(approved=True, cells=[f'joint/{cell}'], sources={'synthetic/source.py': 'abc'})
    approved = tmp_path / 'approved.json'
    approved.write_text(json.dumps(auth))
    monkeypatch.setattr(durable, 'validate_authorization', lambda *a, **k: auth)
    monkeypatch.setattr(durable, 'utc_seconds', lambda: durable.deadline_seconds() - 3600)
    monkeypatch.setattr(runner, 'validate_frozen', lambda out: (frame, protocol))
    monkeypatch.setattr(runner, 'legacy_protocol', lambda path: {})
    monkeypatch.setattr(report, 'legacy_protocol', lambda path: {})
    monkeypatch.setattr(runner, 'reuse_cell', lambda *a: reused)
    monkeypatch.setattr(report, 'reuse_cell', lambda *a: reused)
    timers = []
    monkeypatch.setattr(worker.signal, 'setitimer', lambda *args: timers.append(args))
    worker.run_cell(tmp_path, 'joint', cell, approved)
    assert any(delay == 3600 for _, delay in timers)
    directory = tmp_path / 'joint' / cell
    aligned, done = report.audit_cell(frame, fit, held, 'joint', cell, directory, protocol)
    assert len(aligned) == len(held) and done['n_fits'] == 400
    receipt = json.loads((directory / 'exit_receipt.json').read_text())
    assert receipt['completed_sha256'] == durable.sha(directory / 'completed.json')
    assert receipt['authorization_sha256'] == durable.sha(approved)
    worker.verify_checkpoint_cell(tmp_path, 'joint', cell, auth)
    cache = tmp_path / 'checkpoints/joint' / cell
    call_directory = cache / 'call_0000'
    saved_files = {p.name: p.read_bytes() for p in call_directory.iterdir()}
    runtime_path = directory / 'runtime.json'
    original_runtime = runtime_path.read_bytes()
    completed_path = directory / 'completed.json'
    original_completed = completed_path.read_bytes()
    # Re-sign both checkpoint hashes and the final runtime hash: content checks
    # must still connect checkpoints to the separately audited scientific cell.
    for mutation in ('identity', 'config', 'scaler', 'member_array', 'aggregate', 'ancestry'):
        record = json.loads((call_directory / 'call.json').read_text())
        with np.load(call_directory / 'arrays.npz', allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        if mutation == 'identity':
            record['identity']['context'] = 'forged-context'
        elif mutation == 'config':
            record['members'][0]['config']['learning_rate'] += .01
        elif mutation == 'scaler':
            # A semantically meaningful metadata field not known from ancestry.
            record['members'][0]['best_iteration'] += 1
        elif mutation == 'member_array':
            arrays['m0000_predictions'][0] += 1
            arrays['aggregate'] = np.mean([arrays[f'm{i:04d}_predictions'] for i in range(len(record['members']))], axis=0)
        elif mutation == 'aggregate':
            arrays['aggregate'][0] += 1
        else:
            record['ancestry']['records'][0]['fit_rows'] = record['ancestry']['records'][0]['prediction_rows']
        (call_directory / 'call.json').write_text(json.dumps(record))
        np.savez_compressed(call_directory / 'arrays.npz', **arrays)
        receipt = json.loads((call_directory / 'completed.json').read_text())
        receipt['identity_sha256'] = cp.hashlib.sha256(cp.canonical(record['identity'])).hexdigest()
        receipt['sha256'] = {name: cp.digest(call_directory / name) for name in ('call.json', 'arrays.npz')}
        (call_directory / 'completed.json').write_text(json.dumps(receipt))
        runtime = json.loads(original_runtime)
        runtime['checkpoint_cache']['calls'] = cp.cache_inventory(cache)
        runtime_path.write_text(json.dumps(runtime))
        completed = json.loads(original_completed)
        completed['sha256']['runtime.json'] = cp.digest(runtime_path)
        completed_path.write_text(json.dumps(completed))
        with pytest.raises(ValueError, match='checkpoint'):
            worker.verify_checkpoint_cell(tmp_path, 'joint', cell, auth)
        for name, value in saved_files.items():
            (call_directory / name).write_bytes(value)
        runtime_path.write_bytes(original_runtime)
        completed_path.write_bytes(original_completed)
    # Actual on-disk cache payload is rechecked, beyond the legacy runtime hash.
    cached_array = tmp_path / 'checkpoints/joint' / cell / 'call_0000/arrays.npz'
    cached_array.write_bytes(cached_array.read_bytes()[:-1])
    with pytest.raises(ValueError, match='payload hash'):
        worker.verify_checkpoint_cell(tmp_path, 'joint', cell, auth)
    with pytest.raises(ValueError, match='no automatic retry'):
        worker.run_cell(tmp_path, 'joint', cell, approved)
