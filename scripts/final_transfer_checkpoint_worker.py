"""Single-attempt local transfer worker with complete-call checkpoints."""
from __future__ import annotations

import argparse
from datetime import datetime
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits
from pepstab import final_transfer as ft
from pepstab.final_transfer_checkpoint import CheckpointPredictor, cache_inventory, checked_call, PACKAGES
from scripts import final_transfer as original
from scripts import final_transfer_durable as durable


def verify_checkpoint_cell(out, regime, cell, authorization, *, frame=None, protocol=None):
    """Map every cached call exactly onto the independently audited cell evidence.

    Identity and ancestry are reconstructed without calling the predictor. The
    final legacy audit supplies config/scaler/selection semantics, and this exact
    mapping extends that proof to the checkpoint payloads and their aggregates.
    """
    from pepstab.round2_nested import unlabelled
    out = Path(out)
    directory, cache = out / regime / cell, out / 'checkpoints' / regime / cell
    runtime = durable.read(directory / 'runtime.json')
    proof = runtime.get('checkpoint_cache', {})
    inventory = cache_inventory(cache)
    if (proof.get('schema') != 'final_transfer_checkpoint_cache.1'
            or proof.get('path') != str(cache.resolve()) or proof.get('calls') != inventory
            or proof.get('n_calls') != len(inventory) or proof.get('replayed_calls') != 0):
        raise ValueError('checkpoint runtime proof changed')
    if frame is None:
        frame, protocol = original.validate_frozen(out)
    matches = [(fit, held) for r, c, fit, held in ft.outer_cells(frame) if (r, c) == (regime, cell)]
    if len(matches) != 1:
        raise ValueError('checkpoint outer cell changed')
    fit, held = matches[0]
    selection = durable.read(directory / 'selection.json')
    if set(selection['selected']) != set(ft.NEW_PROCEDURES):
        raise ValueError('checkpoint final procedure coverage changed')
    registry = {candidate['id']: candidate for candidate in ft.CANDIDATES}
    partitions = list(ft.inner_partitions(fit, regime))
    calls = [(f"select/{candidate['id']}/{name}", candidate, train, target, False)
             for candidate in ft.CANDIDATES for name, train, target in partitions]
    calls += [(f'final/{procedure}', registry[selection['selected'][procedure]], fit, held, True)
              for procedure in ft.NEW_PROCEDURES]
    if len(inventory) != len(calls):
        raise ValueError('checkpoint frozen call coverage changed')
    validator = CheckpointPredictor(frame, ft.TransferAncestry(regime), cache=cache, sources=authorization['sources'])
    final_members = durable.read(directory / 'members.json')
    final_ancestry = durable.read(directory / 'ancestry.json')
    control_records = [record for record in final_ancestry['records'] if record['role'] == 'control_member']
    offset = 0
    with np.load(directory / 'member_predictions.npz', allow_pickle=False) as final_arrays:
        for index, (context, candidate, train, target, final) in enumerate(calls):
            target = unlabelled(target)
            record, arrays = checked_call(cache / f'call_{index:04d}')
            identity = validator.identity(train, target, candidate, context, final)
            validator.validate_payload(record, arrays, identity, train, target, candidate, context, final)
            members = record['members']
            stop = offset + len(members)
            if (members != final_members[offset:stop]
                    or record['ancestry']['records'] != control_records[offset:stop]
                    or any(final_ancestry['rowsets'].get(key) != value
                           for key, value in record['ancestry']['rowsets'].items())):
                raise ValueError('checkpoint member/ancestry differs from audited cell')
            for local_index in range(len(members)):
                for key in ('pair_ids', 'predictions'):
                    if not np.array_equal(arrays[f'm{local_index:04d}_{key}'],
                                          final_arrays[f'm{offset + local_index:04d}_{key}']):
                        raise ValueError('checkpoint member array differs from audited cell')
            offset = stop
            validator.index += 1
    if (offset != len(final_members) or offset != len(control_records)
            or offset != ft.fit_graph()['regimes'][regime]['fits_per_cell']):
        raise ValueError('checkpoint fit graph changed')
    return proof


def run_cell(out, regime, cell, approved):
    out, approved = Path(out).resolve(), Path(approved).resolve()
    authorization = durable.validate_authorization(out, approved)
    if f'{regime}/{cell}' not in authorization['cells']:
        raise ValueError('cell outside exact durable authorization')
    frame, protocol = original.validate_frozen(out)
    matches = [(fit, held) for r, c, fit, held in ft.outer_cells(frame) if (r, c) == (regime, cell)]
    if len(matches) != 1:
        raise ValueError('unknown cell')
    fit, held = matches[0]
    directory, cache = out / regime / cell, out / 'checkpoints' / regime / cell
    if directory.exists() or cache.exists():
        raise ValueError('existing worker or checkpoint attempt; no automatic retry')
    directory.mkdir(parents=True)
    auth_hash = durable.sha(approved)
    durable.durable_json(directory / 'attempt.json', dict(pid=os.getpid(), started_at_utc=durable.now(),
                         protocol_hash=protocol['protocol_hash'], authorization_sha256=auth_hash,
                         schema='final_transfer_checkpoint_attempt.1', cache_root=str(cache)))
    handlers = {}

    def check():
        if durable.utc_seconds() >= durable.deadline_seconds():
            raise TimeoutError('original absolute deadline reached')
        if durable.sha(approved) != auth_hash:
            raise ValueError('worker authorization changed')

    def stopped(signum, _frame):
        raise durable.RecoveryStopped(f'worker stopped by signal {signum}')

    try:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            handlers[sig] = signal.signal(sig, stopped)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, durable.deadline_seconds() - durable.utc_seconds()))
        check()
        reused = original.reuse_cell(frame, fit, held, regime, cell,
                   original.legacy_protocol(protocol['legacy_directory']), protocol['legacy_directory'])
        ancestry = ft.TransferAncestry(regime)

        def callback(member):
            check()
            with (directory / 'progress.jsonl').open('a') as handle:
                handle.write(json.dumps(member) + '\n')
                handle.flush()
                os.fsync(handle.fileno())

        predictor = CheckpointPredictor(frame, ancestry, cache=cache, sources=authorization['sources'],
                                        callback=callback, check=check)
        runtime = dict(platform=platform.platform(), python=sys.version, threadpools=threadpool_info(), threads=2,
                       versions={p: version(p) for p in PACKAGES}, ranker_backend='local')
        original.atomic_json(directory / 'runtime.json', runtime)
        values, selection, inner, raw_final = ft.assess_cell(fit, held, regime, predictor, ancestry, reused)
        audit = ancestry.audit(frame, fit, held)
        if len(predictor.members) != ft.fit_graph()['regimes'][regime]['fits_per_cell']:
            raise ValueError('actual fit graph differs')
        calls = predictor.completed_inventory()
        runtime['checkpoint_cache'] = dict(schema='final_transfer_checkpoint_cache.1', path=str(cache.resolve()),
                                          calls=calls, n_calls=len(calls), replayed_calls=predictor.replayed_calls)
        original.atomic_json(directory / 'runtime.json', runtime)
        identity = ['pair_id', 'cluster_id', 'hla_group', 'allele', 'hla_fold', 'peptide_fold']
        original.atomic_csv(directory / 'predictions.csv', pd.concat([
            held[identity].assign(regime=regime, cell=cell, arm=arm, y_pred=p) for arm, p in values.items()], ignore_index=True))
        original.atomic_csv(directory / 'raw_inner_predictions.csv', inner)
        original.atomic_csv(directory / 'raw_final_predictions.csv', pd.concat([
            held[['pair_id']].assign(procedure=p, raw_prediction=v) for p, v in raw_final.items()], ignore_index=True))
        arrays = {f'm{i:04d}_{key}': value for i, member in enumerate(predictor.member_predictions) for key, value in member.items()}
        temporary = directory / 'member_predictions.tmp.npz'
        with temporary.open('xb') as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(directory / 'member_predictions.npz')
        for name, item in [('selection', selection), ('ancestry', ancestry.as_dict()), ('members', predictor.members)]:
            original.atomic_json(directory / f'{name}.json', item)
        check()
        durable.durable_json(directory / 'completed.json', dict(status='complete', regime=regime, cell=cell,
            protocol_hash=protocol['protocol_hash'], n_fits=len(predictor.members), n_held=len(held), audit=audit,
            reused_evidence_sha256=reused['evidence_sha256'], completed_at_utc=durable.now(),
            sha256={name: durable.sha(directory / name) for name in original.FILES}))
        original.verify_cell(directory, protocol, regime, cell)
        verify_checkpoint_cell(out, regime, cell, authorization, frame=frame, protocol=protocol)
        check()
        durable.durable_json(directory / 'exit_receipt.json', dict(status='complete', exit_code=0,
            schema='final_transfer_checkpoint_exit.1', authorization_sha256=auth_hash,
            completed_sha256=durable.sha(directory / 'completed.json'), finished_at_utc=durable.now()))
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        durable.durable_json(directory / 'exit_receipt.json', dict(status='failed', exit_code=1,
            schema='final_transfer_checkpoint_exit.1', authorization_sha256=auth_hash,
            error=repr(error), finished_at_utc=durable.now()))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run-cell'])
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--approved', type=Path, required=True)
    parser.add_argument('--regime', choices=['hla', 'joint'], required=True)
    parser.add_argument('--cell', required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run_cell(args.out_dir, args.regime, args.cell, args.approved)


if __name__ == '__main__':
    main()
