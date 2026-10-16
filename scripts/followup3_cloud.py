"""Small source-bound Modal Sandbox runner for the focused follow-up tranche.

Root prepares plans and reserves the fixed worst-case lifetime before launch.
Provider hard limits survive loss of this client. No automatic retries.
"""
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCOPE = ROOT / 'reports/followup3'
PROFILE = 'a-cheparukhin'
VOLUME = 'pepstab-followup3-20261013'
APP = 'pepstab-followup3-20261013'
BASE_IMAGE = 'im-Epc3H2RdNAgJSwtXBYMCfy'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def relative(name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or any(x.startswith('.') for x in p.parts):
        raise ValueError('explicit nonhidden repository-relative path required')
    if (ROOT / p).is_symlink():
        raise ValueError('symlink refused')
    return p


def validate(plan, *, files=True):
    if plan['profile'] != PROFILE or plan['volume'] != VOLUME or plan['app'] != APP:
        raise ValueError('wrong owned resources')
    if not 1 <= plan['cpu'] <= 16 or not 1024 <= plan['memory_mib'] <= 32768:
        raise ValueError('resource cap exceeded')
    if not 1 <= plan['workers'] <= plan['cpu'] / plan['threads']:
        raise ValueError('worker/thread oversubscription')
    if not 120 <= plan['timeout_seconds'] <= 21600:
        raise ValueError('bounded provider lifetime required')
    rates = read(ROOT / relative(plan['rates_path']))
    rate = (float(rates['cpu_hour_cost_sandbox']) * plan['cpu']
            + float(rates['mem_gib_hour_cost_sandbox']) * plan['memory_mib'] / 1024) / 3600
    if plan.get('gpu'):
        rate += float(rates['gpu_hour_cost_' + plan['gpu_rate_key']]) / 3600
    if abs(rate - plan['rate_usd_second']) > 1e-12:
        raise ValueError('current fixed resource rate mismatch')
    if not .25 <= plan['overhead_usd'] <= 5:
        raise ValueError('explicit bounded startup/storage/egress reserve required')
    if plan['reservation_usd'] < rate * plan['timeout_seconds'] + plan['overhead_usd']:
        raise ValueError('reservation does not cover hard lifetime')
    if plan['reservation_usd'] > 60 or not plan['commands']:
        raise ValueError('bounded phase cap and commands required')
    relative(plan['output_directory'])
    if not plan['output_directory'].startswith('reports/followup3/'):
        raise ValueError('new output namespace required')
    for name, digest in plan['inputs'].items():
        p = ROOT / relative(name)
        if files and sha(p) != digest:
            raise ValueError(f'changed input: {name}')
    return rate


def worker(plan_path, expected_hash):
    """Remote supervisor: immutable inputs, finite child graph, durable receipts."""
    if sha(plan_path) != expected_hash:
        raise ValueError('launch plan changed')
    plan = read(plan_path)
    validate(plan)
    for name, digest in plan.get('remote_inputs', {}).items():
        if sha(ROOT / relative(name)) != digest:
            raise ValueError(f'changed remote artifact: {name}')
    if sys.platform != 'linux':
        raise ValueError('scientific execution is cloud-only')
    directory = ROOT / plan['output_directory']
    directory.mkdir(parents=True, exist_ok=True)
    write(directory / 'worker_attempt.json', {'at_utc': now(), 'plan_sha256': expected_hash,
          'pid': os.getpid()}, exclusive=True)
    deadline = min(time.time() + plan['timeout_seconds'] - 30,
                   datetime.fromisoformat(plan['deadline_utc']).timestamp() - 15)
    pending = list(enumerate(plan['commands']))
    active = {}
    completed = []
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    try:
        while pending or active:
            if stopped or time.time() >= deadline:
                raise TimeoutError('fixed work deadline or termination')
            while pending and len(active) < plan['workers']:
                index, command = pending.pop(0)
                log = (directory / f'child_{index:03d}.log').open('xb')
                start = time.time()
                proc = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                active[index] = (proc, log, start, command)
                write(directory / f'child_{index:03d}.start.json',
                      {'command': command, 'pid': proc.pid, 'at_utc': now()}, exclusive=True)
            for index, (proc, log, start, command) in list(active.items()):
                result = proc.poll()
                if result is None:
                    continue
                log.close()
                row = {'index': index, 'returncode': result, 'seconds': time.time() - start,
                       'command': command, 'at_utc': now()}
                write(directory / f'child_{index:03d}.exit.json', row, exclusive=True)
                completed.append(row)
                del active[index]
                if result:
                    raise RuntimeError(f'child {index} failed: {result}')
            write(directory / 'progress.json', {'at_utc': now(), 'completed': len(completed),
                  'running': len(active), 'pending': len(pending)})
            time.sleep(1)
        validate(plan)
        write(directory / 'worker_complete.json', {'at_utc': now(), 'status': 'complete',
              'plan_sha256': expected_hash, 'children': completed}, exclusive=True)
    except BaseException as error:
        write(directory / 'worker_failure.json', {'at_utc': now(), 'error': repr(error),
              'plan_sha256': expected_hash, 'completed_children': completed}, exclusive=True)
        raise
    finally:
        for proc, log, *_ in active.values():
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            log.close()
        os.sync()


async def handles():
    os.environ['MODAL_PROFILE'] = PROFILE
    import modal
    app = await modal.App.lookup.aio(APP, create_if_missing=True)
    volume = modal.Volume.from_name(VOLUME, create_if_missing=True, version=2)
    await volume.hydrate.aio()
    return modal, app, volume


async def upload(manifest_path):
    """Upload explicit pinned files only. Never upload credentials or directories."""
    manifest = read(manifest_path)
    entries = manifest['files']
    total = 0
    for name, digest in entries.items():
        p = ROOT / relative(name)
        if sha(p) != digest:
            raise ValueError(f'changed upload: {name}')
        total += p.stat().st_size
    if total > 2 * 1024**3 or len(entries) > 3000:
        raise ValueError('input transport cap exceeded')
    _, app, volume = await handles()
    # force permits root-authorized additions/amendments, but caller must have
    # no active worker reading changed files. Exact bytes remain source-bound.
    for start in range(0, len(entries), 128):
        async with volume.batch_upload.aio(force=True) as batch:
            for name in list(entries)[start:start + 128]:
                batch.put_file(ROOT / name, '/' + name)
    result = {'at_utc': now(), 'app_id': app.app_id, 'volume_id': volume.object_id,
              'manifest_sha256': sha(manifest_path), 'files': len(entries), 'bytes': total}
    write(str(manifest_path) + '.uploaded.json', result)
    print(json.dumps(result))


async def launch(plan_path):
    plan = read(plan_path)
    validate(plan)
    directory = ROOT / plan['output_directory']
    ledger = read(SCOPE / 'budget.json')
    matching = [j for j in ledger['jobs'] if j.get('plan_sha256') == sha(plan_path)]
    if len(matching) != 1 or matching[0]['status'] != 'reserved':
        raise ValueError('root must reserve this exact plan before launch')
    reserved = sum(j['reservation_usd'] for j in ledger['jobs'] if j['status'] in ('reserved', 'running'))
    closed = sum(j['conservative_charge_usd'] for j in ledger['jobs'] if j['status'] == 'closed')
    if (matching[0]['reservation_usd'] < plan['reservation_usd']
            or abs(reserved-ledger['reserved_usd']) > 1e-8
            or abs(closed-ledger['conservative_closed_usd']) > 1e-8):
        raise ValueError('ledger must cover this plan and reconcile all jobs')
    if (ledger['planned_commitment_limit_usd'] > 120
            or ledger['inclusive_new_tranche_ceiling_usd'] > 139
            or closed + reserved > min(120, ledger['planned_commitment_limit_usd'])):
        raise ValueError('tranche cap exceeded')
    deadline = datetime.fromisoformat(plan['deadline_utc']).timestamp()
    if not 60 < deadline - time.time() <= plan['timeout_seconds']:
        raise ValueError('launch deadline invalid/expired')
    attempted_epoch = time.time()
    write(directory / 'launch_intent.json', {'at_utc': now(), 'attempted_epoch': attempted_epoch,
          'plan_sha256': sha(plan_path)}, exclusive=True)
    modal, app, volume = await handles()
    image = modal.Image.from_id(plan.get('image_id', BASE_IMAGE))
    mounts = {str(ROOT): volume}
    if plan.get('structures'):
        mounts['/structures'] = modal.Volume.from_name('pepstab-structures', create_if_missing=False)
    env = {key: str(plan['threads']) for key in
           ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
    env.update(PYTHONUNBUFFERED='1',
               PYTHONPATH=str(ROOT / 'runs/followup3/dependencies') + os.pathsep + str(ROOT),
               MODAL_PROFILE=PROFILE, FOLLOWUP3_AUTHORIZED='1')
    sandbox = await modal.Sandbox.create.aio(
        'python', str(ROOT / 'scripts/followup3_cloud.py'), 'worker', '--plan', str(plan_path),
        '--hash', sha(plan_path), app=app, image=image, name=plan['name'],
        cpu=(plan['cpu'], plan['cpu']), memory=(plan['memory_mib'], plan['memory_mib']),
        timeout=max(1, int(deadline-time.time())), gpu=plan.get('gpu'), workdir=str(ROOT),
        env=env, volumes=mounts, tags={'study': 'followup3', 'phase': plan['name']})
    result = {'at_utc': now(), 'created_epoch': attempted_epoch, 'sandbox_id': sandbox.object_id,
              'app_id': app.app_id, 'volume_id': volume.object_id, 'plan_sha256': sha(plan_path)}
    write(directory / 'created.json', result, exclusive=True)
    print(json.dumps(result))


async def monitor(plan_path):
    plan = read(plan_path)
    rate = validate(plan, files=False)
    directory = ROOT / plan['output_directory']
    record = read(directory / 'created.json')
    if record['plan_sha256'] != sha(plan_path):
        raise ValueError('created plan mismatch')
    os.environ['MODAL_PROFILE'] = PROFILE
    import modal
    sandbox = await modal.Sandbox.from_id.aio(record['sandbox_id'])
    deadline = datetime.fromisoformat(plan['deadline_utc']).timestamp()
    try:
        while True:
            result = await asyncio.wait_for(sandbox.poll.aio(), 30)
            if result is not None:
                break
            if time.time() >= deadline:
                await sandbox.terminate.aio(wait=True)
                result = await sandbox.poll.aio()
                break
            write(directory / 'monitor.json', {'at_utc': now(), 'status': 'running',
                  'conservative_elapsed_usd': (time.time()-record['created_epoch'])*rate})
            await asyncio.sleep(15)
    except BaseException:
        await sandbox.terminate.aio(wait=True)
        raise
    seconds = min(plan['timeout_seconds'], time.time()-record['created_epoch'])
    receipt = dict(record, at_utc=now(), returncode=result, seconds=seconds,
                   conservative_compute_usd=seconds*rate, overhead_usd=plan['overhead_usd'],
                   provider_terminal=True)
    write(directory / 'provider_exit.json', receipt, exclusive=True)
    stdout, stderr = await asyncio.gather(sandbox.stdout.read.aio(), sandbox.stderr.read.aio())
    (directory / 'sandbox.stdout.log').write_text(stdout)
    (directory / 'sandbox.stderr.log').write_text(stderr)
    print(json.dumps(receipt))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('upload', 'launch', 'monitor', 'worker', 'validate'))
    p.add_argument('--plan', type=Path)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--hash')
    args = p.parse_args()
    if args.mode == 'worker':
        worker(args.plan, args.hash)
    elif args.mode == 'validate':
        validate(read(args.plan))
        print('Validated; no cloud API called.')
    else:
        asyncio.run({'upload': upload, 'launch': launch, 'monitor': monitor}[args.mode](
                    args.manifest if args.mode == 'upload' else args.plan))


if __name__ == '__main__':
    main()
