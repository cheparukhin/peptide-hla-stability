"""Synthetic evidence and fake workers only: never fit or launch processes."""
import json
from pathlib import Path
import signal
import subprocess
from unittest.mock import Mock

import pytest

from scripts import final_transfer_recovery as recovery


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    original = root / 'original'
    out = root / 'recovery'
    evidence = root / 'closure'
    monkeypatch.setattr(recovery, 'ROOT', root)
    monkeypatch.setattr(recovery, 'ORIGINAL', original)
    monkeypatch.setattr(recovery, 'EVIDENCE', evidence)
    for name in (*recovery.SOURCES, 'scripts/final_transfer_report.py'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('synthetic source')
    write(original / 'protocol.json', dict(protocol_hash=recovery.PROTOCOL_HASH, code_sha256={}))
    records = []
    for cell in recovery.REUSED:
        members = [dict(candidate='tree', fit_seconds=100),
                   dict(candidate='fixed_ranker8', fit_seconds=0.0001,
                        remote_meter=dict(worker_seconds=900))]
        write(original / cell / 'members.json', members)
        write(original / cell / 'completed.json', dict(cell=cell))
        records.append(dict(cell=cell, completed_sha256=recovery.sha(original / cell / 'completed.json'),
                            sha256={'members.json': recovery.sha(original / cell / 'members.json')}))
    write(evidence / 'closure.json', dict(status='interrupted_scope_closed_not_scientifically_complete',
        app_stopped=True, provider_tasks=0, processes_absent=[99999999], new_cloud_spend_authorized_usd=0,
        original_spending_deadline_utc=recovery.DEADLINE))
    write(evidence / 'implementation_amendment.json', dict(deadline_utc=recovery.DEADLINE,
        interruption_closure_sha256=recovery.sha(evidence / 'closure.json'),
        original_protocol_hash=recovery.PROTOCOL_HASH,
        original_protocol_sha256=recovery.sha(original / 'protocol.json')))
    write(evidence / 'preserved_cells_audit.json', dict(status='passed', protocol_hash=recovery.PROTOCOL_HASH,
        n_preserved_cells=18, n_preserved_fits=6030, completed_cells=records,
        audit_source_sha256=recovery.sha(root / 'scripts/final_transfer_report.py')))
    recovery.prepare(out)
    auth = recovery.authorization_payload(out)
    auth['approved'] = True
    approved = out / 'authorization.json'
    write(approved, auth)
    monkeypatch.setattr(recovery, 'utc_seconds', lambda: recovery.deadline_seconds() - 86400)
    return out, approved, root, original


def test_preparation_exact_idempotence_and_immutable_copy(prepared):
    out, approved, root, original = prepared
    before = recovery.inventory(original)
    recovery.prepare(out)
    recovery.validate_authorization(out, approved, launching=True)
    assert recovery.inventory(original) == before
    assert recovery.read(out / 'runtime_forecast.json')['mean_serial_seconds'] == 1000
    assert recovery.read(out / 'runtime_forecast.json')['conservative_seconds'] == 13800
    assert recovery.read(out / 'authorization_template.json')['approved'] is False
    (out / recovery.REUSED[0] / 'members.json').write_text('tamper')
    with pytest.raises(ValueError, match='existing copied cell differs'):
        recovery.prepare(out)
    with pytest.raises(ValueError, match='preserved source/copy'):
        recovery.validate_authorization(out, approved)
    assert recovery.inventory(original) == before


@pytest.mark.parametrize('key,value', [
    ('cells', list(recovery.REMAINING[:-1])), ('cells', list(recovery.REMAINING) + [recovery.REMAINING[0]]),
    ('deadline_utc', '2027-01-01T00:00:00+00:00'), ('max_workers', 4), ('threads_each', 3),
    ('ranker_backend', 'modal'), ('n_new_fits', 4799), ('approved', False),
])
def test_exact_authorization_bounds(prepared, key, value):
    out, approved, *_ = prepared
    auth = recovery.read(approved)
    auth[key] = value
    write(approved, auth)
    with pytest.raises(ValueError, match='exact source/artifact-bound'):
        recovery.validate_authorization(out, approved)


def test_source_and_closure_binding(prepared):
    out, approved, root, _ = prepared
    (root / recovery.SOURCES[0]).write_text('changed implementation')
    with pytest.raises(ValueError, match='authorized source changed'):
        recovery.validate_authorization(out, approved)


def test_deadline_not_extendable_and_forecast_gate(prepared, monkeypatch):
    out, approved, *_ = prepared
    monkeypatch.setattr(recovery, 'utc_seconds', lambda: recovery.deadline_seconds() + 1)
    with pytest.raises(TimeoutError, match='absolute deadline'):
        recovery.validate_authorization(out, approved)
    monkeypatch.setattr(recovery, 'utc_seconds', lambda: recovery.deadline_seconds() - 100)
    with pytest.raises(TimeoutError, match='forecast'):
        recovery.validate_authorization(out, approved, launching=True)


class FakeChild:
    def __init__(self, pid, code=None):
        self.pid, self.code = pid, code
        self._recovery_pid = pid
        self.waited = False

    def poll(self):
        return self.code

    def wait(self, timeout):
        self.waited = True
        self.code = -15 if self.code is None else self.code
        return self.code


@pytest.fixture
def run_mocks(prepared, monkeypatch):
    from scripts import final_transfer
    out, approved, *_ = prepared
    monkeypatch.setattr(final_transfer, 'validate_frozen', lambda _: (None, {'protocol_hash': recovery.PROTOCOL_HASH}))
    monkeypatch.setattr(final_transfer, 'verify_cell', lambda *a: {})
    monkeypatch.setattr(recovery.signal, 'setitimer', lambda *a: None)
    killed = []
    monkeypatch.setattr(recovery.os, 'killpg', lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(recovery.os, 'getpgid', lambda pid: pid)
    monkeypatch.setattr(recovery.time, 'sleep', lambda _: None)
    return out, approved, killed


def test_failed_worker_freezes_queue_and_cleans_only_owned(run_mocks, monkeypatch):
    out, approved, killed = run_mocks
    children = []

    def popen(command, **kwargs):
        assert 'scripts/final_transfer.py' in command[1]
        assert kwargs['start_new_session'] is True
        assert kwargs['env']['OMP_NUM_THREADS'] == '2'
        child = FakeChild(70000 + len(children))
        children.append(child)
        if len(children) == 3:
            children[1].code = 7
        return child

    monkeypatch.setattr(recovery.subprocess, 'Popen', popen)
    with pytest.raises(RuntimeError, match='queued dispatch frozen'):
        recovery.run(out, approved)
    assert len(children) == 3
    assert {pid for pid, _ in killed} == {children[0].pid, children[2].pid}
    assert children[0].waited and children[2].waited
    assert recovery.read(out / 'controller/exit_receipt.json')['status'] == 'failed'
    assert not (out / 'controller/completed.json').exists()
    with pytest.raises(ValueError, match='existing or changed output'):
        recovery.run(out, approved)
    assert len(children) == 3


def test_signal_during_spawn_registers_then_cleans_child(run_mocks, monkeypatch):
    out, approved, killed = run_mocks
    child = FakeChild(71000)

    def popen(*args, **kwargs):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return child

    monkeypatch.setattr(recovery.subprocess, 'Popen', popen)
    with pytest.raises(recovery.RecoveryStopped):
        recovery.run(out, approved)
    assert child.waited
    assert (child.pid, signal.SIGTERM) in killed
    assert recovery.read(out / 'controller/launches/joint_h2_p3.json')['pid'] == child.pid


def test_deadline_stops_running_children(run_mocks, monkeypatch):
    out, approved, killed = run_mocks
    child = FakeChild(72000)

    def popen(*args, **kwargs):
        monkeypatch.setattr(recovery, 'utc_seconds', lambda: recovery.deadline_seconds() + 1)
        return child

    monkeypatch.setattr(recovery.subprocess, 'Popen', popen)
    with pytest.raises(TimeoutError, match='absolute deadline'):
        recovery.run(out, approved)
    assert child.waited
    assert {pid for pid, _ in killed} == {child.pid}


def test_bound_certificate_tamper_is_rejected(tmp_path):
    certificate = tmp_path / 'completed.json'
    write(certificate, {'status': 'complete', 'n_retained_fits': 10830})
    receipt_binding = recovery.binding(certificate)
    recovery.checked_binding(receipt_binding)
    write(certificate, {'status': 'complete', 'n_retained_fits': 10831})
    with pytest.raises(ValueError, match='bound file changed'):
        recovery.checked_binding(receipt_binding)


def test_cleanup_never_signals_exited_historical_pid(monkeypatch):
    child = FakeChild(73000, 1)
    killed = []
    monkeypatch.setattr(recovery.os, 'killpg', lambda pid, sig: killed.append((pid, sig)))
    recovery.cleanup_owned([child])
    assert killed == []


def test_cleanup_refuses_mismatched_group(monkeypatch):
    child = FakeChild(73000)
    killed = []
    monkeypatch.setattr(recovery.os, 'killpg', lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(recovery.os, 'getpgid', lambda pid: 123)
    with pytest.raises(ValueError, match='process group changed'):
        recovery.cleanup_owned([child])
    assert killed == []


def test_old_owner_must_be_absent(prepared, monkeypatch):
    monkeypatch.setattr(recovery.os, 'kill', lambda pid, sig: None)
    with pytest.raises(ValueError, match='still present'):
        recovery.assert_old_processes_absent()


def test_certificate_requires_all_cells_and_audits_exact_new_scope(prepared, monkeypatch):
    from scripts import final_transfer, final_transfer_report
    from pepstab import final_transfer as scientific
    out, approved, *_ = prepared
    for cell in recovery.REMAINING:
        write(out / cell / 'completed.json', {'cell': cell})
        write(out / cell / 'runtime.json', {'ranker_backend': 'local', 'threads': 2})
    axes = [(c.split('/')[0], c.split('/')[1], None, None) for c in recovery.ALL_CELLS]
    monkeypatch.setattr(scientific, 'outer_cells', lambda _: iter(axes))
    monkeypatch.setattr(final_transfer, 'validate_frozen', lambda _: (None, {}))
    verified, audited = [], []
    monkeypatch.setattr(final_transfer, 'verify_cell', lambda path, p, r, c: verified.append(f'{r}/{c}'))
    monkeypatch.setattr(final_transfer_report, 'audit_cell', lambda f, t, h, r, c, d, p: audited.append(f'{r}/{c}'))
    result = recovery.make_certificate(out, approved, 1234)
    assert verified == list(recovery.ALL_CELLS)
    assert audited == list(recovery.REMAINING)
    assert len(result['completed_cell_hashes']) == 30
    assert len(result['reused_cell_hashes']) == 18
    assert len(result['new_cell_hashes']) == 12
    assert result['n_retained_fits'] == 10830
    assert result['n_new_fits'] == 4800
    assert result['known_completed_including_discarded'] == 11529
    write(out / recovery.REMAINING[0] / 'runtime.json', {'ranker_backend': 'modal', 'threads': 2})
    with pytest.raises(ValueError, match='backend/thread'):
        recovery.make_certificate(out, approved, 1234)
    assert not (out / 'controller/completed.json').exists()


def test_cleanup_escalates_unresponsive_owned_worker(monkeypatch):
    class StubbornChild(FakeChild):
        def wait(self, timeout):
            if self.code is None:
                raise subprocess.TimeoutExpired('fake', timeout)
            return self.code
    child = StubbornChild(74000)
    killed = []
    monkeypatch.setattr(recovery.os, 'getpgid', lambda pid: pid)

    def killpg(pid, sig):
        killed.append((pid, sig))
        if sig == signal.SIGKILL:
            child.code = -9
    monkeypatch.setattr(recovery.os, 'killpg', killpg)
    recovery.cleanup_owned([child])
    assert killed == [(74000, signal.SIGTERM), (74000, signal.SIGKILL)]


def test_success_emits_bound_exit_only_after_all_twelve_workers(run_mocks, monkeypatch):
    out, approved, _ = run_mocks
    dispatched = []

    def popen(command, **kwargs):
        regime = command[command.index('--regime') + 1]
        cell = command[command.index('--cell') + 1]
        key = f'{regime}/{cell}'
        dispatched.append(key)
        write(out / key / 'exit_receipt.json', {'status': 'complete', 'exit_code': 0})
        return FakeChild(75000 + len(dispatched), 0)

    def certificate(*args, **kwargs):
        assert dispatched == list(recovery.REMAINING)
        return {'schema': 'final_transfer_recovery_completion.1', 'status': 'complete'}
    monkeypatch.setattr(recovery.subprocess, 'Popen', popen)
    monkeypatch.setattr(recovery, 'make_certificate', certificate)
    recovery.run(out, approved)
    receipt = recovery.read(out / 'controller/exit_receipt.json')
    assert receipt['status'] == 'complete'
    assert receipt['exit_code'] == 0
    assert receipt['completed_sha256'] == recovery.sha(out / 'controller/completed.json')
    assert receipt['authorization_sha256'] == recovery.sha(approved)
    assert len(list((out / 'controller/launch_intents').glob('*.json'))) == 12
    assert len(list((out / 'controller/exits').glob('*.json'))) == 12
    assert recovery.read(out / 'controller/state.json')['finished_cells'] == list(recovery.REMAINING)
    with pytest.raises(ValueError, match='no retry'):
        recovery.run(out, approved)
    assert len(dispatched) == 12
