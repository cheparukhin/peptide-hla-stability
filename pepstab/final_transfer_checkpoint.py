"""Complete-call local checkpoints around the unchanged transfer predictor.

A pending call is evidence of an uncertain attempt, never permission to refit.
The worker remains single-attempt; replay is a primitive for explicitly reviewed
future reconciliation and does not itself authorize process or cell resumption.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from . import final_transfer as ft
from .round2_nested import unlabelled

PACKAGES = ('numpy', 'pandas', 'scipy', 'scikit-learn', 'catboost', 'threadpoolctl')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def frame_identity(frame):
    """Hash values, row order, column order and dtypes without storing population."""
    return dict(columns=list(frame.columns), dtypes=[str(t) for t in frame.dtypes], n_rows=len(frame),
                sha256=hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest())


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exclusive_json(path, value):
    with Path(path).open('xb') as handle:
        handle.write(canonical(value) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(Path(path).parent)


def checked_call(directory):
    """Validate the complete transaction before consuming any of its payload."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('checkpoint directory missing or symlink')
    if (directory / 'pending.json').exists():
        raise ValueError('pending checkpoint requires explicit reconciliation; no retry')
    expected = {'call.json', 'arrays.npz', 'completed.json'}
    if {p.name for p in directory.iterdir()} != expected or any(p.is_symlink() for p in directory.iterdir()):
        raise ValueError('checkpoint transaction incomplete or unexpected files')
    receipt = json.loads((directory / 'completed.json').read_text())
    if (receipt.get('schema') != 'final_transfer_call_completion.1'
            or receipt.get('status') != 'complete' or set(receipt.get('sha256', {})) != expected - {'completed.json'}
            or any(digest(directory / name) != value for name, value in receipt['sha256'].items())):
        raise ValueError('checkpoint payload hash mismatch')
    record = json.loads((directory / 'call.json').read_text())
    if receipt.get('identity_sha256') != hashlib.sha256(canonical(record['identity'])).hexdigest():
        raise ValueError('checkpoint identity receipt mismatch')
    with np.load(directory / 'arrays.npz', allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    return record, arrays


def cache_inventory(cache):
    cache = Path(cache)
    result = {}
    if cache.is_symlink() or not cache.is_dir():
        raise ValueError('checkpoint cache missing or symlink')
    for index, directory in enumerate(sorted(cache.iterdir())):
        if directory.name != f'call_{index:04d}':
            raise ValueError('checkpoint call coverage/order changed')
        checked_call(directory)
        result[directory.name] = {p.name: digest(p) for p in sorted(directory.iterdir())}
    return result


class CheckpointPredictor(ft.ControlPredictor):
    """Only super().__call__ may perform fitting; replay reconstructs its effects."""
    def __init__(self, population, ancestry, *, cache, sources, callback=None, check=lambda: None):
        super().__init__(population, ancestry, callback=callback, ranker_backend=None)
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        if self.cache.is_symlink():
            raise ValueError('checkpoint cache symlink refused')
        self.sources = deepcopy(sources)
        self.versions = dict(python=sys.version, packages={p: version(p) for p in PACKAGES})
        self.population_identity = frame_identity(unlabelled(population))
        self.index, self.calls, self.poisoned = 0, [], False
        self.replayed_calls = 0
        self.check = check
        # Fail before any fresh call if an older uncertain transaction exists.
        cache_inventory(self.cache)

    def identity(self, fit, held, candidate, context, final):
        return dict(schema='final_transfer_call_identity.1', regime=self.ancestry.regime,
                    candidate=deepcopy(candidate), context=context, final=final, call_index=self.index,
                    fit=frame_identity(fit), held=frame_identity(held),
                    population=self.population_identity, feature_hash=self.feature_hash,
                    sources=self.sources, versions=self.versions)

    def expected_ancestry(self, fit, held, candidate, context, final):
        expected = ft.TransferAncestry(self.ancestry.regime)
        for member in ft.member_schedule(fit, self.ancestry.regime, candidate, final=final):
            gradient, stop = member['gradient'], member['stopping']
            expected.add('control_member', gradient, held, stopping=stop,
                         scaler=gradient if candidate['learner'] in ('ridge', 'mlp') else None,
                         context=context, candidate=candidate['id'], seed=member['seed'],
                         stopping_cell=member.get('stopping_cell'))
        return expected

    def validate_payload(self, record, arrays, identity, fit, held, candidate, context, final):
        if record.get('schema') != 'final_transfer_call.1' or record.get('identity') != identity:
            raise ValueError('checkpoint call identity changed')
        expected = self.expected_ancestry(fit, held, candidate, context, final)
        if record.get('ancestry') != expected.as_dict():
            raise ValueError('checkpoint ancestry changed')
        members = record['members']
        if len(members) != len(expected.records):
            raise ValueError('checkpoint member coverage changed')
        keys = {'aggregate', 'pair_ids'} | {f'm{i:04d}_{key}' for i in range(len(members)) for key in ('pair_ids', 'predictions')}
        if set(arrays) != keys:
            raise ValueError('checkpoint array coverage changed')
        ids = held.pair_id.to_numpy(dtype=np.int64)
        predictions = []
        schedule = list(ft.member_schedule(fit, self.ancestry.regime, candidate, final=final))
        for index, (member, ancestry, spec) in enumerate(zip(members, expected.records, schedule)):
            gradient, stop = spec['gradient'], spec['stopping']
            means, global_mean = (ft.shrunk_allele_means(gradient.allele, gradient.y_log1p, 20)
                                  if candidate.get('target') == 'centred' else ({}, None))
            if candidate['learner'] == 'ridge':
                config = dict(alpha=candidate['regularization'], solver='lsqr', tol=1e-5, max_iter=2000)
            elif candidate['learner'] == 'mlp':
                config = json.loads(canonical(asdict(spec['config'])))
            else:
                loss = 'PairLogit' if candidate['learner'] == 'pairlogit' else 'RMSE'
                config = dict(ft.TREE_PARAMS, depth=candidate['depth'], random_seed=spec['seed'],
                              loss_function=loss, eval_metric=loss)
            fixed = dict(context=context, candidate=candidate['id'], seed=ancestry['seed'],
                         stopping_cell=ancestry['stopping_cell'], final_recipe=final,
                         feature_hash=self.feature_hash, config=config, centering_means=means,
                         centering_global=global_mean, n_fit=len(gradient), n_stop=0 if stop is None else len(stop),
                         n_source=len(fit), excluded_cross_quadrant_rows=len(fit) - len(gradient) - (0 if stop is None else len(stop)),
                         **{k: ancestry[k] for k in ('fit_rows', 'stopping_rows', 'prediction_rows', 'scaler_rows') if k in ancestry})
            if any(member.get(k) != value for k, value in fixed.items()):
                raise ValueError('checkpoint member metadata changed')
            if not np.isfinite(member.get('fit_seconds', np.nan)) or member['fit_seconds'] < 0:
                raise ValueError('checkpoint member telemetry changed')
            values = arrays[f'm{index:04d}_predictions']
            if (not np.array_equal(arrays[f'm{index:04d}_pair_ids'], ids)
                    or values.shape != (len(held),) or not np.isfinite(values).all()):
                raise ValueError('checkpoint member prediction identity changed')
            predictions.append(values)
        if (not np.array_equal(arrays['pair_ids'], ids) or arrays['aggregate'].shape != (len(held),)
                or not np.isfinite(arrays['aggregate']).all()
                or not np.array_equal(arrays['aggregate'], np.mean(predictions, axis=0))):
            raise ValueError('checkpoint aggregate changed')
        for key, rows in expected.rowsets.items():
            if key in self.ancestry.rowsets and self.ancestry.rowsets[key] != rows:
                raise ValueError('checkpoint rowset collides with existing ancestry')

    def __call__(self, fit, held, candidate, context, *, final=False):
        if self.poisoned:
            raise ValueError('failed checkpoint predictor cannot continue')
        self.check()
        ft.validate_partition(fit, held, self.ancestry.regime)
        if any(k in held for k in ('y_log1p', 'thalf_hours')):
            raise ValueError('assessment labels forbidden')
        if context in self.calls:
            raise ValueError('duplicate checkpoint context refused')
        identity = self.identity(fit, held, candidate, context, final)
        directory = self.cache / f'call_{self.index:04d}'
        try:
            if directory.exists():
                record, arrays = checked_call(directory)
                self.validate_payload(record, arrays, identity, fit, held, candidate, context, final)
                self.replayed_calls += 1
                for index, member in enumerate(record['members']):
                    ancestry_record = record['ancestry']['records'][index]
                    for key, value in ancestry_record.items():
                        if key.endswith('_rows'):
                            self.ancestry.rowsets[value] = deepcopy(record['ancestry']['rowsets'][value])
                    self.ancestry.records.append(deepcopy(ancestry_record))
                    self.members.append(deepcopy(member))
                    self.member_predictions.append(dict(pair_ids=arrays[f'm{index:04d}_pair_ids'].copy(),
                                                        predictions=arrays[f'm{index:04d}_predictions'].copy()))
                    if self.callback:
                        self.callback(self.members[-1])
                aggregate = arrays['aggregate'].copy()
            else:
                directory.mkdir()
                fsync_directory(self.cache)
                exclusive_json(directory / 'pending.json', dict(identity=identity, pid=os.getpid()))
                begin_members, begin_records = len(self.members), len(self.ancestry.records)
                aggregate = super().__call__(fit, held, candidate, context, final=final)
                records = self.ancestry.records[begin_records:]
                rowkeys = {v for r in records for k, v in r.items() if k.endswith('_rows')}
                record = dict(schema='final_transfer_call.1', identity=identity,
                              members=deepcopy(self.members[begin_members:]),
                              ancestry=dict(regime=self.ancestry.regime, records=deepcopy(records),
                                            rowsets={k: self.ancestry.rowsets[k] for k in rowkeys}))
                arrays = dict(aggregate=aggregate.copy(), pair_ids=held.pair_id.to_numpy(dtype=np.int64))
                arrays.update({f'm{i:04d}_{key}': value for i, member in enumerate(self.member_predictions[begin_members:]) for key, value in member.items()})
                # Normalize JSON tuples exactly as the unchanged evidence writer does.
                record = json.loads(canonical(record))
                self.validate_payload(record, arrays, identity, fit, held, candidate, context, final)
                with (directory / 'arrays.npz').open('xb') as handle:
                    np.savez_compressed(handle, **arrays)
                    handle.flush()
                    os.fsync(handle.fileno())
                exclusive_json(directory / 'call.json', record)
                exclusive_json(directory / 'completed.json', dict(schema='final_transfer_call_completion.1', status='complete',
                    identity_sha256=hashlib.sha256(canonical(identity)).hexdigest(),
                    sha256={name: digest(directory / name) for name in ('arrays.npz', 'call.json')}))
                (directory / 'pending.json').unlink()
                fsync_directory(directory)
            self.calls.append(context)
            self.index += 1
            self.check()
            return aggregate
        except BaseException:
            self.poisoned = True
            raise

    def completed_inventory(self):
        result = cache_inventory(self.cache)
        if len(result) != self.index:
            raise ValueError('unused checkpoint calls remain')
        return result
