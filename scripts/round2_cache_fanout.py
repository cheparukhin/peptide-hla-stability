#!/usr/bin/env python3
"""Single-owner LOCAL gate/controller for explicitly authorized cache fanout.

No Modal SDK import. Root supplies --approved and an immutable approval JSON:
{approved, run_id, cache_fingerprint, cohort_manifest_sha256,
 feature_protocol_sha256, sequence_authorization_sha256,
 allocations_usd: {cache:80, sequence:18, engineering:2, total:100}}.
The controller writes only its own status/log; never edits a scientific ledger.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TOTAL_ROWS = 19716
SHARD = 256
RATE = 1.10 + 4 * .0473 + 24 * .008
EXTRACT_SOURCES = ('pepstab/boltz_embeddings.py', 'pepstab/round2_representations.py',
                   'modal_app/round2_boltz_worker.py', 'modal_app/round2_boltz_extract.py')
SEQUENCE_SOURCES = ('pepstab/round2_nested.py', 'pepstab/round2_sequence.py',
                    'pepstab/round2_budget.py', 'scripts/round2_sequence.py', 'modal_app/round2_sequence.py')


class GateError(ValueError):
    pass


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical_digest(value):
    # Existing bundle writer hashes canonical JSON including its terminal LF.
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()).hexdigest()


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise GateError(f'{name} must be finite and nonnegative')
    return value


def check_hash(path, expected):
    if digest(path) != expected:
        raise GateError(f'Hash changed: {path}')


def atomic_status(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as handle:
        json.dump(state, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


@contextmanager
def controller_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GateError('Another fanout controller owns the cohort lock') from exc
        yield


def bundle_pairs(record, expected_pairs, protocol, contact_hash):
    directory = Path(record['directory'])
    complete = read(directory / 'complete.json')
    for key, hash_key in (('array_file', 'npz_sha256'), ('metadata_file', 'metadata_sha256')):
        name = complete[key]
        if Path(name).name != name:
            raise GateError('Unsafe compact feature filename')
        check_hash(directory / name, complete[hash_key])
        if record[hash_key] != complete[hash_key]:
            raise GateError('Ledger compact feature checksum mismatch')
    metadata = read(directory / complete['metadata_file'])
    n = len(expected_pairs)
    if (metadata['n_rows'] != n or record['n_rows'] != n
            or metadata['npz_sha256'] != complete['npz_sha256']
            or metadata['array_file'] != complete['array_file']
            or metadata['feature_protocol'] != protocol
            or metadata['feature_protocol_sha256'] != canonical_digest(protocol)
            or metadata['contact_map_sha256'] != contact_hash
            or metadata['dimensions'] != protocol['dimensions']
            or metadata['schema_version'] != protocol['schema_version']
            or complete['schema_version'] != protocol['schema_version']):
        raise GateError('Compact feature protocol, dimensions or coverage mismatch')
    pairs = [str(row['pair_id']) for row in metadata['rows']]
    if len(pairs) != len(set(pairs)) or set(pairs) != set(expected_pairs):
        raise GateError('Compact feature row identity mismatch')
    with np.load(directory / complete['array_file'], allow_pickle=False) as arrays:
        if set(arrays.files) != set(protocol['dimensions']) | {'pair_id', 'input_hash'}:
            raise GateError('Compact feature array keys mismatch')
        if arrays['pair_id'].shape != (n,) or arrays['input_hash'].shape != (n,):
            raise GateError('Compact provenance array shapes mismatch')
        hashes = [row['input_hash'] for row in metadata['rows']]
        if (arrays['pair_id'].astype(str).tolist() != pairs
                or arrays['input_hash'].astype(str).tolist() != hashes
                or any(not isinstance(h, str) or len(h) != 64
                       or any(ch not in '0123456789abcdef' for ch in h) for h in hashes)):
            raise GateError('Compact array/manifest ordered identity or input hash mismatch')
        for name, width in protocol['dimensions'].items():
            values = arrays[name]
            if values.shape != (n, width) or values.dtype != np.float32 or not np.isfinite(values).all():
                raise GateError(f'Compact feature values invalid: {name}')
    return pairs


def sequence_gate(root, authorization):
    out = root / 'reports/further_research/round2/sequence'
    path = out / 'launch_authorization.json'
    check_hash(path, authorization['sequence_authorization_sha256'])
    approved = read(path)
    protocol = read(out / 'protocol.json')
    hashes = {p: digest(root / p) for p in SEQUENCE_SOURCES}
    forecast = read(out / 'modal_budget_forecast.json')
    smoke = read(out / 'modal_smoke.json')
    if (approved.get('approved') is not True or approved.get('budget_ceiling_usd') != 18
            or approved.get('protocol_hash') != protocol['protocol_hash']
            or hashlib.sha256(json.dumps(protocol['specification'], sort_keys=True).encode()).hexdigest() != protocol['protocol_hash']
            or approved.get('code_sha256') != hashes or smoke.get('passed') is not True
            or forecast.get('scope_ceiling_usd') != 18
            or finite(forecast['model_based_total_with_margin_usd'], 'Sequence forecast') > 18):
        raise GateError('Sequence authorization, protocol, implementation or budget mismatch')


def validate_static(root, cohort, forecast_path, authorization, ledger):
    allocation = authorization.get('allocations_usd')
    if allocation != {'cache': 80, 'sequence': 18, 'engineering': 2, 'total': 100}:
        raise GateError('Require unchanged 80 cache + 18 sequence + 2 engineering allocation <=100')
    for name, value in allocation.items():
        finite(value, name)
    if allocation['cache'] + allocation['sequence'] + allocation['engineering'] > allocation['total']:
        raise GateError('Total allocation exceeds 100')
    fingerprint = ledger['fingerprint']
    reserved = sum(finite(value, 'Outstanding reservation') for value in ledger['reservations'].values())
    if finite(ledger['spent_upper_usd'], 'Ledger spending') + reserved > 80:
        raise GateError('Outstanding commitments exceed cache allocation')
    if fingerprint != authorization['cache_fingerprint'] or fingerprint['run_id'] != authorization['run_id']:
        raise GateError('Extraction fingerprint differs from root authorization')
    batch_path = forecast_path.with_name('batch_forecast_options.json')
    contact_path = root / 'reports/further_research/round2/representations/contact_map.json'
    feature_path = contact_path.with_name('feature_protocol.json')
    manifest_path = cohort.with_name('cohort_manifest.json')
    for path, value in ((cohort, fingerprint['cohort_sha256']), (forecast_path, fingerprint['forecast_sha256']),
                        (batch_path, fingerprint['batch_forecast_sha256']), (contact_path, fingerprint['contact_map_sha256']),
                        (feature_path, authorization['feature_protocol_sha256']),
                        (manifest_path, authorization['cohort_manifest_sha256'])):
        check_hash(path, value)
    if set(fingerprint['source_hashes']) != set(EXTRACT_SOURCES):
        raise GateError('Missing required extraction source hashes')
    for relative, value in fingerprint['source_hashes'].items():
        check_hash(root / relative, value)
    manifest = read(manifest_path)
    if (manifest['cohort_kind'] != 'full_training' or manifest['n_rows'] != TOTAL_ROWS
            or manifest['source_training_rows'] != TOTAL_ROWS
            or manifest['cohort_sha256'] != fingerprint['cohort_sha256']):
        raise GateError('Require the entire 19716-row original-training cohort')
    for relative, value in manifest['source_hashes'].items():
        check_hash(root / relative, value)
    if set(manifest['source_hashes']) != {'data/splits.csv', 'data/rasmussen_et_al_dataset.csv', 'structures/ectodomain_msas/manifest.json'}:
        raise GateError('Missing canonical cohort source hashes')
    with cohort.open() as handle:
        rows = list(csv.DictReader(handle))
    pairs = [str(row['pair_id']) for row in rows]
    if len(rows) != TOTAL_ROWS or len(set(pairs)) != TOTAL_ROWS or any(row['split'] != 'train' for row in rows):
        raise GateError('Full training cohort has invalid row identity or split')
    forecast, batch = read(forecast_path), read(batch_path)
    option = next(o for o in batch['options'] if o['shard_size'] == SHARD)
    predicted = (finite(option['ledger_with_extra60s_per_shard_usd'], 'Batch forecast')
                 + option['shards'] * 2 * RATE / 3600)
    if (forecast['gate'] != 'engineering_passed_forecast_only_full_cohort_not_launched'
            or forecast['training_pairs'] != TOTAL_ROWS or batch['n_cases'] != TOTAL_ROWS
            or fingerprint['n_cases'] != TOTAL_ROWS or fingerprint['shard_size'] != SHARD
            or fingerprint['shards'] != math.ceil(TOTAL_ROWS / SHARD)
            or option['shards'] != fingerprint['shards'] or ledger['limit_usd'] != 80
            or not math.isclose(finite(ledger['conservative_forecast_usd'], 'Ledger forecast'), predicted, abs_tol=1e-9)
            or predicted > 80 or finite(ledger['spent_upper_usd'], 'Ledger spending') > 80):
        raise GateError('Changed cohort, batch forecast or budget')
    sequence_gate(root, authorization)
    return pairs, read(feature_path), read(contact_path)['contact_map_sha256'], predicted


def pilot_gate(cohort, ledger, pairs, protocol, contact_hash, predicted):
    if ledger.get('halted') or any(record.get('status') != 'passed' for record in ledger['completed'].values()):
        raise GateError('Pilot or previous shard failed; no automatic retries')
    first = ledger['completed'].get('0')
    if first is None or ledger['reservations']:
        return None
    if set(ledger['completed']) != {'0'} or first['rows'] != SHARD:
        raise GateError('Fanout requires exactly one successful 256-case pilot')
    report = read(cohort.parent / 'shard_00000.json')
    expected = pairs[:SHARD]
    if (report['status'] != 'passed' or report.get('returncode') != 0
            or len(report['cases']) != SHARD or any(r.get('ok') is not True for r in report['cases'])
            or {str(r['pair_id']) for r in report['cases']} != set(expected)):
        raise GateError('First pilot did not pass every requested case')
    charge = finite(first['charged_upper_usd'], 'Pilot charged upper')
    if not math.isclose(ledger['spent_upper_usd'], charge, abs_tol=1e-9):
        raise GateError('Pilot spending differs from settled charge')
    if charge != finite(report['charged_upper_usd'], 'Pilot report charge'):
        raise GateError('Pilot charge differs from settled ledger')
    measured = charge / SHARD * TOTAL_ROWS * 1.10
    if max(predicted, measured) > 80:
        raise GateError(f'Full-cache forecast exceeds 80: existing={predicted:.4f}, measured={measured:.4f}')
    bundle_pairs(first['bundle'], expected, protocol, contact_hash)
    aggregate = read(cohort.parent / 'feature_bundles.json')
    if (aggregate['status'] != 'partial' or aggregate['expected_rows'] != TOTAL_ROWS
            or aggregate['completed_rows'] != SHARD or len(aggregate['bundles']) != 1
            or aggregate['bundles'][0] != first['bundle']
            or aggregate['cohort_sha256'] != ledger['fingerprint']['cohort_sha256']):
        raise GateError('Pilot aggregate does not match completed bundle')
    return {'existing_forecast_usd': predicted, 'pilot_scaled_forecast_usd': measured,
            'gated_forecast_usd': max(predicted, measured)}


def verify_complete(root, cohort, forecast_path, authorization):
    ledger = read(cohort.parent / 'extraction_budget_ledger.json')
    pairs, protocol, contact, _ = validate_static(root, cohort, forecast_path, authorization, ledger)
    if ledger.get('halted') or ledger['reservations'] or set(ledger['completed']) != {str(i) for i in range(math.ceil(TOTAL_ROWS / SHARD))}:
        raise GateError('Final extraction ledger is incomplete or unresolved')
    seen = []
    for i in range(math.ceil(TOTAL_ROWS / SHARD)):
        record = ledger['completed'][str(i)]
        expected = pairs[i * SHARD:(i + 1) * SHARD]
        if record['status'] != 'passed' or record['rows'] != len(expected):
            raise GateError(f'Final shard {i} incomplete')
        finite(record['charged_upper_usd'], 'Shard charge')
        seen += bundle_pairs(record['bundle'], expected, protocol, contact)
    aggregate = read(cohort.parent / 'feature_bundles.json')
    ledger_bundles = [ledger['completed'][str(i)]['bundle'] for i in range(math.ceil(TOTAL_ROWS / SHARD))]
    settled = sum(finite(r['charged_upper_usd'], 'Shard charge') for r in ledger['completed'].values())
    if (len(seen) != TOTAL_ROWS or len(set(seen)) != TOTAL_ROWS or aggregate['status'] != 'complete'
            or aggregate['completed_rows'] != TOTAL_ROWS or aggregate['expected_rows'] != TOTAL_ROWS
            or aggregate['cohort_sha256'] != ledger['fingerprint']['cohort_sha256']
            or sorted(b['directory'] for b in aggregate['bundles']) != sorted(b['directory'] for b in ledger_bundles)
            or sorted(aggregate['feature_bundles']) != sorted(b['directory'] for b in ledger_bundles)
            or not math.isclose(settled, ledger['spent_upper_usd'], abs_tol=1e-8)):
        raise GateError('Final 19716-row aggregate coverage mismatch')
    return {'completed_rows': TOTAL_ROWS, 'spent_upper_usd': ledger['spent_upper_usd']}


def launch_command(root, cohort, forecast, run_id):
    return [str(root / '.venv/bin/modal'), 'run', 'modal_app/round2_boltz_extract.py',
            '--cohort', str(cohort), '--forecast', str(forecast), '--run-id', run_id,
            '--budget-usd', '80', '--workers', '8']


def run_controller(root, cohort, forecast, approval_path, approved, wait_seconds=5400, poll_seconds=15):
    if not approved or read(approval_path).get('approved') is not True:
        raise GateError('Explicit --approved and approved root authorization JSON are required')
    if not 0 < wait_seconds <= 5400 or not 0 < poll_seconds <= 60:
        raise GateError('Wait must be <=90 minutes; polling must be <=60 seconds')
    authorization = read(approval_path)
    auth_hash = digest(approval_path)
    state_dir = cohort.parent / 'fanout_controller'
    state_path = state_dir / 'status.json'
    with controller_lock(state_dir / 'controller.lock'):
        if state_path.exists() and read(state_path).get('launch_attempted'):
            raise GateError('This controller already attempted fanout; automatic restart/retry is forbidden')
        state = {'status': 'waiting_for_pilot', 'controller_pid': os.getpid(), 'child_pid': None,
                 'launch_attempted': False, 'authorization_sha256': auth_hash,
                 'started_utc': datetime.now(timezone.utc).isoformat()}
        child = None
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'Controller received signal {signum}')
        previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            deadline = time.monotonic() + wait_seconds
            while True:
                check_hash(approval_path, auth_hash)
                ledger = read(cohort.parent / 'extraction_budget_ledger.json')
                pairs, protocol, contact, predicted = validate_static(root, cohort, forecast, authorization, ledger)
                gate = pilot_gate(cohort, ledger, pairs, protocol, contact, predicted)
                atomic_status(state_path, state)
                if time.monotonic() >= deadline:
                    raise GateError('Pilot did not pass and settle all reservations within 90-minute bound')
                if gate is not None:
                    break
                time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))
            # Canonical identity check is performed only after the settled pilot.
            from scripts.round2_boltz_cohort import cases_for_cohort
            cases = cases_for_cohort(cohort)
            if len(cases) != TOTAL_ROWS or [str(c['pair_id']) for c in cases] != pairs:
                raise GateError('Canonical original-training identity mismatch')
            check_hash(approval_path, auth_hash)
            # Recheck every authorization/hash and the settled pilot after the
            # canonical read, immediately before recording the launch intent.
            ledger = read(cohort.parent / 'extraction_budget_ledger.json')
            pairs, protocol, contact, predicted = validate_static(root, cohort, forecast, authorization, ledger)
            gate = pilot_gate(cohort, ledger, pairs, protocol, contact, predicted)
            if gate is None:
                raise GateError('Pilot state became unresolved before launch')
            command = launch_command(root, cohort, forecast, authorization['run_id'])
            state.update(status='launching', launch_attempted=True, command=command, budget_gate=gate,
                         log=str(state_dir / 'child.log'))
            atomic_status(state_path, state)  # launch intent persists even if Popen fails
            with (state_dir / 'child.log').open('ab', buffering=0) as log:
                child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True,
                                         env={**os.environ, 'MODAL_PROFILE': 'a-cheparukhin'})
                state.update(status='running', child_pid=child.pid)
                atomic_status(state_path, state)
                returncode = child.wait()
            state['child_returncode'] = returncode
            if returncode != 0:
                raise GateError(f'Extraction child exited {returncode}; no retry')
            state.update(verify_complete(root, cohort, forecast, authorization), status='complete')
            atomic_status(state_path, state)
            return state
        except BaseException as exc:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
            state.update(status='halted', error=f'{type(exc).__name__}: {exc}',
                         requires_manual_reconciliation=True, automatic_retry=False)
            if child is not None:
                state['child_returncode'] = child.poll()
            atomic_status(state_path, state)
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, default=ROOT / 'reports/further_research/round2/extractor/full_training/cohort.csv')
    parser.add_argument('--forecast', type=Path, default=ROOT / 'reports/further_research/round2/extractor/forecast.json')
    parser.add_argument('--authorization', type=Path, required=True)
    parser.add_argument('--approved', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run_controller(ROOT, args.cohort.resolve(), args.forecast.resolve(),
                                    args.authorization.resolve(), args.approved), indent=2))


if __name__ == '__main__':
    main()
