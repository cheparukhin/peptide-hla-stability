"""One-shot, local-only continuation of the closed interrupted transfer scope.

prepare copies evidence and creates an unapproved template; run never retries.
No cloud API or remote worker is imported by this operational controller.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ORIGINAL = ROOT / 'reports/finalization/transfer_controls'
OUT = ROOT / 'reports/finalization/transfer_recovery'
EVIDENCE = ROOT / 'reports/finalization/resources/transfer_main/interruption_20261012'
DEADLINE = '2026-10-13T15:32:19.924192+00:00'
PROTOCOL_HASH = 'bcda56bc3ae36dd757ef1a3210b167fc926d1604be74bb7487c5e18fedaf412d'
ALL_CELLS = tuple(f'hla/h{h}' for h in range(5)) + tuple(f'joint/h{h}_p{p}' for h in range(5) for p in range(5))
REUSED = ALL_CELLS[:18]
REMAINING = ALL_CELLS[18:]
SCHEMA = 'final_transfer_recovery_authorization.1'
SOURCES = ('scripts/final_transfer_recovery.py', 'tests/test_final_transfer_recovery.py')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def binding(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def now():
    return datetime.now(timezone.utc).isoformat()


def utc_seconds():
    return datetime.now(timezone.utc).timestamp()


def deadline_seconds():
    return datetime.fromisoformat(DEADLINE).timestamp()


def durable_json(path, value, *, exact=False):
    """Exclusive evidence writes, including directory fsync; never replace receipts."""
    path = Path(path)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    durable_bytes(path, payload, exact=exact)


def durable_bytes(path, payload, *, exact=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f'symlink output refused: {path}')
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if exact and path.read_bytes() == payload:
            return
        raise ValueError(f'existing or changed output refused: {path}')
    with os.fdopen(fd, 'wb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def checked_binding(item):
    path = Path(item['path'])
    if path.is_symlink() or sha(path) != item['sha256']:
        raise ValueError(f'bound file changed: {path}')
    return path


def inventory(directory):
    result = {}
    for path in sorted(Path(directory).rglob('*')):
        if path.is_symlink():
            raise ValueError('symlink evidence refused')
        if path.is_file():
            result[str(path.relative_to(directory))] = sha(path)
    return result


def runtime_forecast():
    observations = {}
    for cell in REUSED:
        if not cell.startswith('joint/'):
            continue
        members = read(ORIGINAL / cell / 'members.json')
        total = 0.0
        for member in members:
            # Cloud fit_seconds measures parent delivery, not ranker compute.
            value = (member['remote_meter']['worker_seconds']
                     if member['candidate'] == 'fixed_ranker8'
                     else member['fit_seconds'])
            if not math.isfinite(value) or value < 0:
                raise ValueError('invalid completed timing telemetry')
            total += value
        observations[cell] = total
    mean = sum(observations.values()) / len(observations)
    maximum = max(observations.values())
    return dict(schema='final_transfer_recovery_forecast.1',
                joint_cell_serial_seconds=observations, mean_serial_seconds=mean,
                max_serial_seconds=maximum, nominal_seconds=mean * 12 / 3,
                conservative_seconds=maximum * 4 * 3 + 1800,
                workers=3, threads_each=2, waves=4, platform_factor=3,
                overhead_seconds=1800, deadline_utc=DEADLINE,
                method='Non-ranker fit_seconds plus ranker remote_meter.worker_seconds; 13 completed joint cells.',
                limitation='Platform and contention variance; this forecast is not a runtime guarantee.')


def validate_closure():
    closure = read(EVIDENCE / 'closure.json')
    amendment = read(EVIDENCE / 'implementation_amendment.json')
    audit = read(EVIDENCE / 'preserved_cells_audit.json')
    if (closure.get('status') != 'interrupted_scope_closed_not_scientifically_complete'
            or closure.get('app_stopped') is not True or closure.get('provider_tasks') != 0
            or closure.get('new_cloud_spend_authorized_usd') != 0
            or closure.get('original_spending_deadline_utc') != DEADLINE
            or amendment.get('deadline_utc') != DEADLINE
            or amendment.get('interruption_closure_sha256') != sha(EVIDENCE / 'closure.json')
            or amendment.get('original_protocol_hash') != PROTOCOL_HASH
            or amendment.get('original_protocol_sha256') != sha(ORIGINAL / 'protocol.json')):
        raise ValueError('interrupted-scope closure/amendment mismatch')
    if (audit.get('status') != 'passed' or audit.get('protocol_hash') != PROTOCOL_HASH
            or audit.get('n_preserved_cells') != 18 or audit.get('n_preserved_fits') != 6030
            or [c['cell'] for c in audit['completed_cells']] != list(REUSED)
            or audit.get('audit_source_sha256') != sha(ROOT / 'scripts/final_transfer_report.py')):
        raise ValueError('preserved semantic audit mismatch')
    return closure, amendment, audit


def prepare(out=OUT):
    """No fits or process launches. Existing evidence must match byte-for-byte."""
    out = Path(out).resolve()
    if out == ORIGINAL.resolve() or ORIGINAL.resolve() in out.parents:
        raise ValueError('original evidence is immutable')
    _, _, audit = validate_closure()
    original_protocol = read(ORIGINAL / 'protocol.json')
    if original_protocol['protocol_hash'] != PROTOCOL_HASH:
        raise ValueError('protocol changed')
    copied = {}
    for entry in audit['completed_cells']:
        cell = entry['cell']
        files = inventory(ORIGINAL / cell)
        expected = dict(entry['sha256'], **{'completed.json': entry['completed_sha256']})
        if any(files.get(k) != v for k, v in expected.items()):
            raise ValueError('original preserved evidence changed')
        if (out / cell).exists() and inventory(out / cell) != files:
            raise ValueError('existing copied cell differs')
        copied[cell] = files
        for name in files:
            durable_bytes(out / cell / name, (ORIGINAL / cell / name).read_bytes(), exact=True)
    for cell in REMAINING:
        if (out / cell).exists():
            raise ValueError('unfinished destination already exists; no retry')
    durable_bytes(out / 'protocol.json', (ORIGINAL / 'protocol.json').read_bytes(), exact=True)
    forecast = runtime_forecast()
    durable_json(out / 'runtime_forecast.json', forecast, exact=True)
    manifest = dict(schema='final_transfer_recovery_preparation.1', scope_root=str(out),
                    original_root=str(ORIGINAL.resolve()), protocol_hash=PROTOCOL_HASH,
                    protocol=binding(out / 'protocol.json'), closure=binding(EVIDENCE / 'closure.json'),
                    amendment=binding(EVIDENCE / 'implementation_amendment.json'),
                    preserved_audit=binding(EVIDENCE / 'preserved_cells_audit.json'),
                    runtime_forecast=binding(out / 'runtime_forecast.json'),
                    reused_cell_hashes={c: copied[c]['completed.json'] for c in REUSED},
                    copied_file_hashes=copied, cells=list(REMAINING), deadline_utc=DEADLINE,
                    sources={name: sha(ROOT / name) for name in set(SOURCES) | set(original_protocol['code_sha256'])},
                    n_reused_fits=6030, n_new_fits=4800, n_retained_fits=10830,
                    discarded_partial_completed_fits=699, known_completed_including_discarded=11529,
                    unknown_remote_fits=12, additional_cloud_spend_usd=0)
    durable_json(out / 'preparation.json', manifest, exact=True)
    durable_json(out / 'authorization_template.json', authorization_payload(out), exact=True)
    return manifest


def authorization_payload(out=OUT):
    out = Path(out).resolve()
    manifest = read(out / 'preparation.json')
    return dict(schema=SCHEMA, approved=False, scope_root=str(out), protocol_hash=PROTOCOL_HASH,
                preparation=binding(out / 'preparation.json'), protocol=manifest['protocol'],
                closure=manifest['closure'], preserved_audit=manifest['preserved_audit'],
                runtime_forecast=manifest['runtime_forecast'], sources=manifest['sources'],
                reused_cell_hashes=manifest['reused_cell_hashes'], cells=list(REMAINING),
                max_workers=3, threads_each=2, deadline_utc=DEADLINE, ranker_backend='local',
                n_new_fits=4800, additional_cloud_spend_usd=0, no_retry=True)


def validate_authorization(out, approved, *, launching=False):
    out = Path(out).resolve()
    actual = read(approved)
    expected = authorization_payload(out)
    expected['approved'] = True
    if actual.get('approved') is not True or actual != expected:
        raise ValueError('authorization differs from exact source/artifact-bound local scope')
    manifest = read(checked_binding(actual['preparation']))
    if manifest['scope_root'] != str(out) or manifest['cells'] != list(REMAINING):
        raise ValueError('preparation axes changed')
    for key in ('protocol', 'closure', 'amendment', 'preserved_audit', 'runtime_forecast'):
        checked_binding(manifest[key])
    validate_closure()
    for name, digest in actual['sources'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'authorized source changed: {name}')
    for cell, files in manifest['copied_file_hashes'].items():
        if inventory(out / cell) != files or inventory(ORIGINAL / cell) != files:
            raise ValueError('preserved source/copy changed')
    if set(manifest['copied_file_hashes']) != set(REUSED):
        raise ValueError('preserved cell count changed')
    remaining = deadline_seconds() - utc_seconds()
    if remaining <= 0:
        raise TimeoutError('original absolute deadline passed')
    if launching and remaining < read(out / 'runtime_forecast.json')['conservative_seconds']:
        raise TimeoutError('conservative forecast no longer fits original deadline')
    return actual


class RecoveryStopped(BaseException):
    pass


def assert_old_processes_absent():
    closure = read(EVIDENCE / 'closure.json')
    for pid in closure['processes_absent']:
        if type(pid) is not int or pid <= 0:
            raise ValueError('invalid original ownership PID')
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise ValueError(f'original controller/guard/helper PID still present: {pid}')


def cleanup_owned(children):
    """Signal only unreaped live child leaders whose recorded group still matches.

    Workers are the unchanged numerical CLI, which launches no subprocesses.
    An exited/reaped leader is never used as authority to kill a recycled group.
    """
    def live_owned(child):
        if child.poll() is not None:
            return False
        if getattr(child, '_recovery_pid', None) != child.pid:
            raise ValueError('owned child identity changed; refusing signal')
        try:
            group = os.getpgid(child.pid)
        except ProcessLookupError:
            return False
        if group != child.pid:
            raise ValueError('owned child process group changed; refusing signal')
        return True

    live, errors = [], []
    for child in children:
        try:
            if live_owned(child):
                os.killpg(child.pid, signal.SIGTERM)
                live.append(child)
        except ProcessLookupError:
            pass
        except Exception as error:
            errors.append(repr(error))
    until = time.monotonic() + 5
    for child in live:
        try:
            child.wait(timeout=max(0.01, until - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
        except Exception as error:
            errors.append(repr(error))
    for child in live:
        try:
            if live_owned(child):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
        except ProcessLookupError:
            pass
        except Exception as error:
            errors.append(repr(error))
    if errors:
        raise ValueError('owned cleanup incomplete: ' + '; '.join(errors))


def make_certificate(out, approved, pid, *, check_deadline=lambda: None):
    from scripts.final_transfer import validate_frozen, verify_cell
    from scripts.final_transfer_report import audit_cell
    from pepstab.final_transfer import outer_cells
    out = Path(out)
    authorization = validate_authorization(out, approved)
    frame, protocol = validate_frozen(out)
    all_hashes = {}
    for regime, cell, fit, held in outer_cells(frame):
        check_deadline()
        key = f'{regime}/{cell}'
        verify_cell(out / key, protocol, regime, cell)
        if key in REMAINING:
            audit_cell(frame, fit, held, regime, cell, out / key, protocol)
            runtime = read(out / key / 'runtime.json')
            if runtime['ranker_backend'] != 'local' or runtime['threads'] != 2:
                raise ValueError('new worker backend/thread count changed')
        all_hashes[key] = sha(out / key / 'completed.json')
    if set(all_hashes) != set(ALL_CELLS):
        raise ValueError('incomplete aggregate')
    check_deadline()
    return dict(schema='final_transfer_recovery_completion.1', status='complete', pid=pid,
                protocol_hash=PROTOCOL_HASH, scope_root=str(out.resolve()),
                authorization=binding(approved), authorization_sha256=sha(approved),
                closure=authorization['closure'], closure_sha256=authorization['closure']['sha256'],
                preserved_audit=authorization['preserved_audit'], completed_cell_hashes=all_hashes,
                reused_cell_hashes={c: all_hashes[c] for c in REUSED},
                new_cell_hashes={c: all_hashes[c] for c in REMAINING},
                n_retained_fits=10830, n_new_fits=4800, known_completed_including_discarded=11529,
                unknown_remote_fits=12, deadline_utc=DEADLINE, finished_at_utc=now())


def run(out=OUT, approved=None):
    out = Path(out).resolve()
    approved = Path(approved).resolve()
    authorization = validate_authorization(out, approved, launching=True)
    from scripts.final_transfer import validate_frozen, verify_cell
    _, protocol = validate_frozen(out)
    assert_old_processes_absent()
    directory = out / 'controller'
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / 'ownership.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    base = dict(pid=os.getpid(), scope_root=str(out), protocol_hash=PROTOCOL_HASH,
                authorization=binding(approved), authorization_sha256=sha(approved),
                closure=authorization['closure'], deadline_utc=DEADLINE)
    children, active, pending, logs = [], {}, list(REMAINING), []
    finished = []

    def save_state(status, **extra):
        state = dict(base, status=status, at_utc=now(), active_cells=list(active),
                     finished_cells=list(finished), pending_cells=list(pending), **extra)
        temporary = directory / f'.state-{os.getpid()}-{time.monotonic_ns()}.json'
        durable_json(temporary, state)
        os.replace(temporary, directory / 'state.json')
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def check_authorization_unchanged():
        if sha(approved) != base['authorization_sha256']:
            raise ValueError('authorization changed after attempt ownership')
    handlers = {}
    spawning = False
    stop_requested = None
    stop_signals = {signal.SIGTERM, signal.SIGINT, signal.SIGALRM}
    monotonic_deadline = time.monotonic() + max(0, deadline_seconds() - utc_seconds())

    def check_deadline():
        if utc_seconds() >= deadline_seconds() or time.monotonic() >= monotonic_deadline:
            raise TimeoutError('original absolute deadline reached')

    def stopped(signum, _frame):
        nonlocal stop_requested
        stop_requested = signum
        if not spawning:
            raise RecoveryStopped(f'stopped by signal {signum}')

    try:
        for cell in REMAINING:
            if (out / cell).exists():
                raise ValueError('new cell destination exists; no retry')
        durable_json(directory / 'attempt.json', dict(base, status='attempted', started_at_utc=now()))
    except BaseException:
        lock.close()
        raise
    try:
        save_state('running')
        for sig in stop_signals:
            handlers[sig] = signal.signal(sig, stopped)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline_seconds() - utc_seconds()))
        while active or pending:
            check_deadline()
            # Observe every running child before permitting any queued dispatch.
            for cell, child in list(active.items()):
                code = child.poll()
                if code is None:
                    continue
                durable_json(directory / 'exits' / (cell.replace('/', '_') + '.json'),
                             dict(cell=cell, pid=child.pid, exit_code=code, at_utc=now()))
                if code != 0:
                    raise RuntimeError(f'worker failed: {cell}, exit {code}; queued dispatch frozen')
                regime, name = cell.split('/')
                verify_cell(out / cell, protocol, regime, name)
                receipt = read(out / cell / 'exit_receipt.json')
                if receipt.get('status') != 'complete' or receipt.get('exit_code') != 0:
                    raise ValueError('worker missing successful receipt')
                del active[cell]
                children.remove(child)
                finished.append(cell)
                save_state('running')
            if pending and len(active) < 3:
                check_authorization_unchanged()
                validate_authorization(out, approved)
                assert_old_processes_absent()
                check_deadline()
                # Recheck after potentially lengthy hash verification.
                if any(child.poll() is not None for child in active.values()):
                    continue
                cell = pending.pop(0)
                regime, name = cell.split('/')
                command = [sys.executable, str(ROOT / 'scripts/final_transfer.py'), 'run-cell',
                           '--out-dir', str(out), '--approved', str(approved),
                           '--regime', regime, '--cell', name]
                durable_json(directory / 'launch_intents' / (cell.replace('/', '_') + '.json'),
                             dict(base, cell=cell, command=command, at_utc=now()))
                logfile = (directory / (cell.replace('/', '_') + '.log')).open('xb')
                logs.append(logfile)
                env = os.environ.copy()
                for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                    env[key] = '2'
                # Do not allow a signal between child creation and ownership registration.
                spawning = True
                try:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                             stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
                    child._recovery_pid = child.pid
                    children.append(child)
                    active[cell] = child
                    durable_json(directory / 'launches' / (cell.replace('/', '_') + '.json'),
                                 dict(cell=cell, pid=child.pid, process_group=child.pid, at_utc=now()))
                    save_state('running')
                finally:
                    spawning = False
                if stop_requested is not None:
                    raise RecoveryStopped(f'stopped during child launch: {stop_requested}')
                continue
            if active:
                time.sleep(0.2)
        check_deadline()
        check_authorization_unchanged()
        certificate = make_certificate(out, approved, os.getpid(), check_deadline=check_deadline)
        check_deadline()
        check_authorization_unchanged()
        durable_json(directory / 'completed.json', certificate)
        save_state('complete')
        durable_json(directory / 'exit_receipt.json', dict(base, schema='final_transfer_recovery_exit.1',
                     status='complete', exit_code=0, completed_sha256=sha(directory / 'completed.json'),
                     completed_certificate=binding(directory / 'completed.json'), finished_at_utc=now()))
        return certificate
    except BaseException as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig in stop_signals:
            signal.signal(sig, signal.SIG_IGN)
        cleanup_error = None
        try:
            cleanup_owned(children)
        except BaseException as cleanup:
            cleanup_error = repr(cleanup)
        save_state('failed', error=repr(error), cleanup_error=cleanup_error)
        durable_json(directory / 'exit_receipt.json', dict(base, schema='final_transfer_recovery_exit.1',
                     status='failed', exit_code=1, error=repr(error), cleanup_error=cleanup_error,
                     dispatched_cells=list(REMAINING[:len(REMAINING) - len(pending)]), finished_at_utc=now()))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        for handle in logs:
            handle.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'validate', 'run'])
    parser.add_argument('--out-dir', type=Path, default=OUT)
    parser.add_argument('--approved', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        print(json.dumps(prepare(args.out_dir), indent=2))
    elif args.approved is None:
        parser.error('--approved required')
    elif args.mode == 'validate':
        validate_authorization(args.out_dir, args.approved, launching=True)
        print('Exact recovery authorization validated; no fits launched.')
    else:
        run(args.out_dir, args.approved)


if __name__ == '__main__':
    main()
