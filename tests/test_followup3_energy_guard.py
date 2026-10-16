"""Resource-guard fault checks; no network or scientific computation."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import followup3_energy_guard as guard


def setup_guard(tmp_path, monkeypatch, *, rate_factor=1):
    root = tmp_path
    directory = root / 'reports/followup3/resources/energy_extract'
    directory.mkdir(parents=True)
    rates = directory / 'rates.json'
    rates.write_text(json.dumps({'cpu_hour_cost': '.0473', 'mem_gib_hour_cost': '.008'}))
    auth = dict(budget_reservation_id='energy-extract', reservation_usd=40,
                max_containers=100, cpu_per_container=1,
                memory_mib_per_container=2048,
                maximum_compute_usd_per_second=.0633 * 100 / 3600 * rate_factor,
                overhead_usd=3, admission_epoch=time.time(),
                deadline_utc=datetime.fromtimestamp(time.time()+120, timezone.utc).isoformat(),
                rates_path=str(rates.relative_to(root)), rates_sha256=guard.sha(rates))
    authorization = directory / 'authorization.json'
    authorization.write_text(json.dumps(auth))
    ledger = dict(planned_commitment_limit_usd=120, jobs=[dict(
        name='energy-extract', plan_sha256=guard.sha(authorization),
        status='reserved', reservation_usd=40)])
    (root/'reports/followup3/budget.json').write_text(json.dumps(ledger))
    created = directory/'created.json'
    created.write_text(json.dumps(dict(app_id='ap-test-energy',
        root_authorization_sha256=guard.sha(authorization))))
    monkeypatch.setattr(guard, 'ROOT', root)
    monkeypatch.setattr(guard.sys, 'argv', ['guard', '--authorization', str(authorization), '--created', str(created)])
    monkeypatch.setattr(guard.time, 'sleep', lambda _: None)
    return directory


def test_understated_rate_rejected_before_provider_call(tmp_path, monkeypatch):
    setup_guard(tmp_path, monkeypatch, rate_factor=.5)
    monkeypatch.setattr(guard, 'cli', lambda *_: pytest.fail('must reject before network'))
    with pytest.raises(AssertionError):
        guard.main()


def test_poll_failure_stops_bound_app_and_keeps_reconciliation_open(tmp_path, monkeypatch):
    directory = setup_guard(tmp_path, monkeypatch)
    calls = []
    def fake_cli(*args):
        calls.append(args)
        if args[1] == 'stop':
            return 'Stopped'
        if len(calls) == 1:
            return json.dumps([dict(app_id='ap-test-energy', description=guard.APP_NAME, state='ephemeral')])
        raise TimeoutError('simulated provider timeout')
    monkeypatch.setattr(guard, 'cli', fake_cli)
    with pytest.raises(TimeoutError):
        guard.main()
    assert ('app', 'stop', 'ap-test-energy', '--yes') in calls
    failure = guard.read(directory/'guard_failure.json')
    assert failure['stop_command_succeeded']
    assert failure['root_reconciliation_required'] and not failure['provider_terminal']
    assert not (directory/'provider_exit.json').exists()


def test_terminal_receipt_requires_provider_stopped_state(tmp_path, monkeypatch):
    directory = setup_guard(tmp_path, monkeypatch)
    monkeypatch.setattr(guard, 'cli', lambda *_: json.dumps([
        dict(app_id='ap-test-energy', description=guard.APP_NAME, state='stopped')]))
    guard.main()
    receipt = guard.read(directory/'provider_exit.json')
    assert receipt['provider_terminal'] and receipt['app_id'] == 'ap-test-energy'
    assert receipt['overhead_usd'] == 3
