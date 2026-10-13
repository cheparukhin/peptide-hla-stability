"""Independent local deadline guard for the separately owned energy Function app."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = 'pepstab-followup3-repaired-energy'
PROFILE = 'a-cheparukhin'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def cli(*args):
    return subprocess.check_output([sys.executable, '-m', 'modal', *args,
                                    '--profile', PROFILE], text=True, timeout=60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorization', type=Path, required=True)
    parser.add_argument('--created', type=Path, required=True)
    args = parser.parse_args()
    auth = read(args.authorization)
    digest = sha(args.authorization)
    ledger = read(ROOT / 'reports/followup3/budget.json')
    entries = [j for j in ledger['jobs'] if j.get('plan_sha256') == digest]
    assert len(entries) == 1 and entries[0]['status'] == 'reserved'
    assert entries[0]['name'] == auth['budget_reservation_id']
    assert entries[0]['reservation_usd'] == auth['reservation_usd'] == 40
    total = sum(j['reservation_usd'] if j['status'] == 'reserved' else
                j['conservative_charge_usd'] for j in ledger['jobs'])
    assert total <= ledger['planned_commitment_limit_usd'] <= 120
    assert auth['max_containers'] == 100 and auth['cpu_per_container'] == 1
    assert auth['memory_mib_per_container'] == 2048
    rate = auth['maximum_compute_usd_per_second']
    rates_path = ROOT / auth['rates_path']
    assert sha(rates_path) == auth['rates_sha256']
    rates = read(rates_path)
    expected_rate = 100 * (float(rates['cpu_hour_cost']) +
                           2 * float(rates['mem_gib_hour_cost'])) / 3600
    assert abs(rate - expected_rate) < 1e-12
    assert rate * 18000 + auth['overhead_usd'] <= 40
    deadline = datetime.fromisoformat(auth['deadline_utc']).timestamp()
    assert 0 < deadline - time.time() <= 18000
    directory = args.authorization.parent
    start = time.time()
    write(directory / 'guard_started.json', dict(start_epoch=start,
          authorization_sha256=digest, deadline_utc=auth['deadline_utc']))
    app_id = None
    try:
        while True:
            if args.created.exists():
                created = read(args.created)
                assert created['root_authorization_sha256'] == digest
                if app_id is not None:
                    assert created['app_id'] == app_id
                app_id = created['app_id']
            apps = json.loads(cli('app', 'list', '--json'))
            matches = [a for a in apps if a['description'] == APP_NAME and
                       (a['app_id'] == app_id if app_id else a['state'] != 'stopped')]
            if app_id is None and matches:
                assert len(matches) == 1, 'ambiguous owned energy app'
                app_id = matches[0]['app_id']
            current = next((a for a in apps if a['app_id'] == app_id), None)
            if current is not None:
                assert current['description'] == APP_NAME
            expired = time.time() >= deadline
            if expired and app_id is None:
                raise RuntimeError('deadline reached before app creation was reconciled')
            if expired and (current is None or current['state'] != 'stopped'):
                cli('app', 'stop', app_id, '--yes')
                time.sleep(2)
                continue
            if current is not None and current['state'] == 'stopped':
                seconds = min(18000, time.time() - auth['admission_epoch'])
                result = dict(app_id=app_id, provider_terminal=True,
                              deadline_reached=expired, authorization_sha256=digest,
                              seconds_from_admission=seconds,
                              conservative_compute_usd=rate * seconds,
                              overhead_usd=auth['overhead_usd'],
                              accounting='maximum 100-container occupancy; not an invoice')
                write(directory / 'provider_exit.json', result)
                print(json.dumps(result))
                return
            write(directory / 'monitor.json', dict(app_id=app_id,
                  at_utc=datetime.now(timezone.utc).isoformat(),
                  state=current['state'] if current else 'waiting_for_app'))
            time.sleep(min(30, max(1, deadline - time.time())))
    except BaseException as error:
        stopped = False
        stop_error = None
        if app_id is not None:
            try:
                cli('app', 'stop', app_id, '--yes')
                stopped = True
            except BaseException as failure:
                stop_error = repr(failure)
        write(directory / 'guard_failure.json', dict(app_id=app_id,
              authorization_sha256=digest, error=repr(error),
              stop_command_succeeded=stopped, stop_error=stop_error,
              provider_terminal=False, root_reconciliation_required=True))
        raise


if __name__ == '__main__':
    main()
