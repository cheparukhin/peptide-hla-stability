"""One-attempt Modal Sandbox control; local operations never fit scientific models.

Run the independent watcher under launchd before create. A new Sandbox waits
for a single stdin release, which is sent only after that watcher names its ID.
Any uncertain create is reconciled and stopped; it is never retried here.
"""
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parents[1]
CPU_RATE = .00003942
GIB_RATE = .00000667
LIMITS = {'transfer_report': (2, 8192, 28800, 5),
          'hybrid_pilot': (2, 8192, 7200, 2),
          'hybrid_main': (4, 16384, 43200, 20),
          'hybrid_report': (2, 8192, 28800, 5)}
OUTER_STOP = '2026-10-14T14:15:17.567539+00:00'


def now():
    return datetime.now(timezone.utc).isoformat()


def stamp(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('explicit UTC offset required')
    return parsed.timestamp()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def status_write(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{time.monotonic_ns()}')
    write_once(temporary, value)
    temporary.replace(path)


def bound(item):
    p = Path(item['path'])
    if not p.is_absolute(): p = ROOT / p
    if '..' in p.parts or not p.is_relative_to(ROOT):
        raise ValueError('bound input leaves repository mount')
    cursor=p
    while cursor!=ROOT:
        if cursor.is_symlink():raise ValueError('bound input has symlink ancestor')
        cursor=cursor.parent
    if p.is_symlink() or sha(p) != item['sha256']:
        raise ValueError(f'bound input changed: {p}')
    return p


def rate(auth):
    r = auth['resources']
    return r['cpu_cores'] * CPU_RATE + r['memory_mib'] / 1024 * GIB_RATE


def validate_plan(path, *, admission=False):
    plan = read(path)
    if plan.get('schema') != 'final_sandbox_launch.1' or plan.get('approved') is not True:
        raise ValueError('exact approved launch plan required')
    if plan.get('profile') != 'a-cheparukhin':
        raise ValueError('explicit authorized Modal profile required')
    auth_path = bound(plan['authorization']); auth = read(auth_path)
    phase = auth['phase']
    if auth.get('schema') != 'final_cloud_phase_authorization.1' or auth.get('approved') is not True or phase not in LIMITS:
        raise ValueError('approved known phase required')
    cpu, memory, wall, cap = LIMITS[phase]
    r = auth['resources']
    expected_workers = 2 if phase == 'hybrid_main' else 1
    if (type(r.get('cpu_cores')) is not int
            or any(type(r.get(k)) is not int for k in ('memory_mib', 'numerical_threads', 'max_workers'))
            or r != dict(cpu_cores=cpu, memory_mib=memory, numerical_threads=2, max_workers=expected_workers)
            or type(auth['fit_wall_seconds']) is not int or not 0 < auth['fit_wall_seconds'] <= wall
            or auth['phase_cap_usd'] != cap or plan['phase_cap_usd'] != cap):
        raise ValueError('frozen phase resource/cost caps changed')
    if not plan['image_id'].startswith('im-') or plan['image_id'] != auth['image_id']:
        raise ValueError('immutable image identity required')
    if (not plan['app_id'].startswith('ap-') or not plan['app_name'] or not plan['volume_name']
            or not plan.get('volume_id','').startswith('vo-') or plan.get('volume_version')!=2):
        raise ValueError('exact named provider resources required')
    stop, work = stamp(plan['provider_stop_deadline_utc']), stamp(auth['deadline_utc'])
    created = stamp(plan['created_at_utc'])
    if not created < work <= stop - 120 or stop > stamp(OUTER_STOP):
        raise ValueError('deadline or shutdown cushion invalid')
    if stop - created > auth['fit_wall_seconds'] + 900:
        raise ValueError('setup/lifecycle allowance exceeds bound')
    upper = (stop - created) * rate(auth)
    if upper + .5 > cap:
        raise ValueError('full continuous resource cost lacks phase headroom')
    if type(plan['max_download_bytes']) is not int or not 0 < plan['max_download_bytes'] <= 20 * 1024 ** 3:
        raise ValueError('finite download bound required')
    directory = Path(plan['phase_directory']).resolve()
    allowed = ROOT / 'reports/finalization/resources/cloud_sandboxes'
    if allowed not in directory.parents or directory.name != auth['phase_nonce']:
        raise ValueError('phase state leaves dedicated scope')
    required = {'scripts/final_sandbox_driver.py', 'tests/test_final_sandbox_driver.py'}
    if not required <= set(plan['source_hashes']):
        raise ValueError('launcher and tests must be source-bound')
    for name, digest in plan['source_hashes'].items():
        p = (ROOT / name).resolve()
        if ROOT not in p.parents or p.is_symlink() or sha(p) != digest:
            raise ValueError('launch source changed')
    accounting = read(bound(auth['cloud_accounting']))
    for key in ('prior_conservative_liability_usd', 'future_envelope_usd', 'total_ceiling_usd',
                'phase_cap_usd', 'reserved_phase_usd', 'conservative_spend_usd'):
        if type(accounting.get(key)) not in (int, float) or not math.isfinite(accounting[key]) or accounting[key] < 0:
            raise ValueError('finite nonnegative accounting required')
    if (accounting.get('schema') != 'final_cloud_accounting.1'
            or accounting['prior_conservative_liability_usd'] < 5.1793820030564595
            or not 32 <= accounting['future_envelope_usd'] <= 35 or accounting['total_ceiling_usd'] > 100
            or accounting['prior_conservative_liability_usd'] + accounting['future_envelope_usd'] > accounting['total_ceiling_usd']
            or accounting['phase'] != phase or accounting['phase_cap_usd'] != cap
            or accounting['reserved_phase_usd'] < cap
            or accounting['conservative_spend_usd'] + accounting['reserved_phase_usd'] > accounting['future_envelope_usd']):
        raise ValueError('phase lacks conservative budget reservation')
    if admission and (time.time() >= work or not 0 <= time.time() - created <= 240):
        raise ValueError('launch window expired; no sliding deadlines')
    return plan, auth, auth_path, directory


def tags(auth, auth_path):
    return dict(phase=auth['phase'], phase_nonce=auth['phase_nonce'], authorization_sha256=sha(auth_path))


def watcher_ready(directory, auth_path, *, plan_path, sandbox_id=None):
    state = read(directory / 'watchdog_status.json')
    attempt = read(directory / 'watchdog_attempt.json')
    expected = {'waiting_for_create'} if sandbox_id is None else {'monitoring'}
    if (state['status'] not in expected or state['authorization_sha256'] != sha(auth_path)
            or state.get('plan_sha256') != sha(plan_path)
            or type(state.get('pid')) is not int or state['pid'] <= 1
            or attempt.get('pid') != state['pid'] or attempt.get('plan_sha256') != sha(plan_path)
            or not 0 <= time.time() - stamp(state['at_utc']) <= 30
            or (sandbox_id is not None and state.get('sandbox_id') != sandbox_id)):
        raise ValueError('independent watchdog is not fresh and exact')
    if state['pid'] == os.getpid(): raise ValueError('watchdog must be independent')
    os.kill(state['pid'], 0)
    return state


async def create(plan_path):
    plan, auth, auth_path, directory = validate_plan(plan_path, admission=True)
    os.environ['MODAL_PROFILE'] = plan['profile']
    import modal
    watcher_ready(directory, auth_path, plan_path=plan_path)
    write_once(directory / 'create_intent.json', dict(at_utc=now(), pid=os.getpid(),
        plan_sha256=sha(plan_path), authorization_sha256=sha(auth_path), phase_nonce=auth['phase_nonce'],
        provider_stop_deadline_utc=plan['provider_stop_deadline_utc']))
    app = await asyncio.wait_for(modal.App.lookup.aio(plan['app_name'], create_if_missing=False), 30)
    if app.app_id != plan['app_id']: raise ValueError('app lookup identity changed')
    active = await asyncio.wait_for(_matching_sandboxes(modal, plan, auth, auth_path), 20)
    if active: raise ValueError('matching Sandbox already exists; no retry')
    volume = modal.Volume.from_name(plan['volume_name'], create_if_missing=False, version=2)
    await asyncio.wait_for(volume.hydrate.aio(),30)
    if volume.object_id!=plan['volume_id']:
        raise ValueError('materialized volume identity differs from launch plan')
    # An existing image ID avoids new unbounded dependency-image builds here.
    image = await asyncio.wait_for(modal.Image.from_id.aio(plan['image_id']), 30)
    validate_plan(plan_path, admission=True)
    watcher_ready(directory, auth_path, plan_path=plan_path)
    remaining = math.floor(stamp(plan['provider_stop_deadline_utc']) - time.time())
    if remaining <= 120 or remaining * rate(auth) + .5 > plan['phase_cap_usd']:
        raise ValueError('actual remaining provider lifetime exceeds guarded reservation')
    environment = {key: '2' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                       'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}
    environment['PYTHONUNBUFFERED'] = '1'
    sandbox = await asyncio.wait_for(modal.Sandbox.create.aio('python', str(ROOT / 'scripts/final_cloud_phase.py'),
        '--approved', str(auth_path), '--startup-wait-seconds', '300', app=app, name=auth['sandbox_name'],
        image=image, cpu=(auth['resources']['cpu_cores'],) * 2, memory=(auth['resources']['memory_mib'],) * 2,
        gpu=None, timeout=remaining, workdir=str(ROOT), env=environment,
        volumes={str(ROOT): volume}, tags=tags(auth, auth_path)), 120)
    record = dict(sandbox_id=sandbox.object_id, app_id=app.app_id, at_utc=now(), phase=auth['phase'],
        authorization_sha256=sha(auth_path), plan_sha256=sha(plan_path), timeout_seconds=remaining,
        volume_id=volume.object_id,volume_version=2,
        native_timeout_is_relative=True,absolute_work_stop_enforced_by_supervisor_and_watchdog=True)
    write_once(directory / 'created.json', record)
    return record


async def watch(plan_path, *, interval=5):
    plan, auth, auth_path, directory = validate_plan(plan_path)
    os.environ['MODAL_PROFILE'] = plan['profile']
    import modal
    write_once(directory / 'watchdog_attempt.json', dict(pid=os.getpid(), at_utc=now(), plan_sha256=sha(plan_path)))
    pinned_plan, pinned_auth = sha(plan_path), sha(auth_path)
    active_id = None; started = time.monotonic(); terminal = None; owned_ids = set()
    lease_sequence=0;last_lease_monotonic=-math.inf;lease_expiry=None
    sandbox=None;stdin_writer=None;release_delivered=False;release_digest=None
    base = dict(pid=os.getpid(), authorization_sha256=pinned_auth, plan_sha256=pinned_plan)
    reason = 'provider_terminal'
    loop=asyncio.get_running_loop();task=asyncio.current_task()
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    for sig in previous:loop.add_signal_handler(sig,task.cancel)
    try:
        while True:
            validate_plan(plan_path)
            if sha(plan_path) != pinned_plan or sha(auth_path) != pinned_auth:
                raise ValueError('authorization changed during monitoring')
            record_path = directory / 'created.json'
            if record_path.exists():
                record = read(record_path)
                if (record['authorization_sha256'] != pinned_auth or record['app_id'] != plan['app_id']
                        or record['plan_sha256'] != pinned_plan):
                    raise ValueError('created receipt identity changed')
                if active_id is not None and record['sandbox_id'] != active_id:
                    raise ValueError('provider identity replaced')
                active_id = record['sandbox_id']
            if active_id is None:
                if (directory / 'create_intent.json').exists():
                    matches = await asyncio.wait_for(_matching_sandboxes(modal, plan, auth, auth_path), 20)
                    owned_ids.update(s.object_id for s in matches)
                    if len(matches) > 1:
                        raise ValueError('duplicate provider objects terminated')
                    if matches and time.time() - stamp(read(directory / 'create_intent.json')['at_utc']) > 150:
                        active_id = matches[0].object_id
                        raise ValueError('ambiguous create reconciled; stop without releasing work')
                if time.time() >= stamp(auth['deadline_utc']) or time.monotonic() - started > 300:
                    raise TimeoutError('unlaunched guard expired')
                status_write(directory / 'watchdog_status.json', dict(base, status='waiting_for_create', at_utc=now()))
            else:
                matches = await asyncio.wait_for(_matching_sandboxes(modal, plan, auth, auth_path), 20)
                owned_ids.update(s.object_id for s in matches)
                if any(s.object_id != active_id for s in matches):
                    raise ValueError('unexpected duplicate provider objects terminated')
                if sandbox is None:
                    # This exact handle owns the stream index/offset for its entire lifetime.
                    sandbox = await asyncio.wait_for(modal.Sandbox.from_id.aio(active_id), 20)
                    stdin_writer = sandbox.stdin
                remote_tags = await asyncio.wait_for(sandbox.get_tags.aio(), 20)
                if any(remote_tags.get(k) != v for k, v in tags(auth, auth_path).items()):
                    raise ValueError('provider tags differ from owned phase')
                owned_ids.add(active_id)
                terminal = await asyncio.wait_for(sandbox.poll.aio(), 20)
                if terminal is not None: break
                if time.time() >= stamp(auth['deadline_utc']):
                    raise TimeoutError('fixed work-stop deadline reached')
                if not release_delivered:
                    if (directory/'released.json').exists() or (directory/'release_send_attempt.json').exists():
                        raise ValueError('prior or uncertain stream write cannot be resumed with a new handle')
                    if (directory/'release_intent.json').exists():
                        value=checked_release_intent(plan_path,plan,auth,auth_path,directory,active_id)
                        release_digest=sha(directory/'release_intent.json')
                        write_once(directory/'release_send_attempt.json',dict(schema='final_sandbox_release_send.1',
                            pid=os.getpid(),sandbox_id=active_id,at_utc=now(),plan_sha256=pinned_plan,
                            authorization_sha256=pinned_auth,release_sha256=release_digest,no_retry=True))
                        # Disk sync and provider reads above may have consumed readiness time.
                        if checked_release_intent(plan_path,plan,auth,auth_path,directory,active_id)!=value:
                            raise ValueError('release intent changed immediately before write')
                        stdin_writer.write((json.dumps(value)+'\n').encode())
                        await asyncio.wait_for(stdin_writer.drain.aio(),20)
                        write_once(directory/'released.json',dict(schema='final_sandbox_release_receipt.1',
                            status='delivered',pid=os.getpid(),at_utc=now(),sandbox_id=active_id,
                            plan_sha256=pinned_plan,authorization_sha256=pinned_auth,release_sha256=release_digest))
                        release_delivered=True
                if release_delivered and time.monotonic()-last_lease_monotonic>=10:
                    issued=read(directory/'released.json');release_record=read(directory/'release_intent.json')
                    if (issued.get('sandbox_id')!=active_id or issued.get('pid')!=os.getpid()
                            or issued.get('release_sha256')!=release_digest
                            or sha(directory/'release_intent.json')!=release_digest
                            or release_record.get('authorization_sha256')!=pinned_auth
                            or release_record.get('sandbox_id')!=active_id):
                        raise ValueError('release evidence changed before guard lease')
                    lease_sequence+=1
                    issued_time=time.time();lease_expiry=min(issued_time+90,stamp(auth['deadline_utc']))
                    lease=dict(schema='final_cloud_guard_lease.1',phase_nonce=auth['phase_nonce'],
                        authorization_sha256=pinned_auth,sandbox_id=active_id,watchdog_pid=os.getpid(),
                        sequence=lease_sequence,issued_at_utc=datetime.fromtimestamp(issued_time,timezone.utc).isoformat(),
                        expires_at_utc=datetime.fromtimestamp(lease_expiry,timezone.utc).isoformat())
                    stdin_writer.write((json.dumps(lease)+'\n').encode())
                    await asyncio.wait_for(stdin_writer.drain.aio(),20)
                    last_lease_monotonic=time.monotonic()
                status_write(directory / 'watchdog_status.json', dict(base, status='monitoring', sandbox_id=active_id,
                    at_utc=now(), guard_lease_sequence=lease_sequence,guard_lease_expiry=lease_expiry,
                    conservative_elapsed_resource_usd=max(0, time.time()-stamp(plan['created_at_utc']))*rate(auth)))
            await asyncio.sleep(interval)
    except BaseException as error:
        reason = repr(error)
        for sig in previous:
            loop.remove_signal_handler(sig);signal.signal(sig,signal.SIG_IGN)
        stops = await asyncio.gather(*[_stop_owned(modal, sid) for sid in sorted(owned_ids)])
        stop_results = {item['sandbox_id']: item for item in stops}
        terminal = stop_results.get(active_id, {}).get('returncode')
    else:
        stop_results = {}
    finally:
        for sig,handler in previous.items():
            loop.remove_signal_handler(sig);signal.signal(sig,handler)
    clean = reason == 'provider_terminal' and terminal is not None
    result = dict(base, status='terminal_observed' if clean else 'failed_stop_observed' if stop_results and all(
                      item['returncode'] is not None for item in stop_results.values()) else 'stop_unconfirmed',
                  sandbox_id=active_id, returncode=terminal, reason=reason, at_utc=now(),
                  owned_sandbox_ids=sorted(owned_ids), stop_results=stop_results,
                  reservation_retained_usd=auth['phase_cap_usd'] * max(1,len(owned_ids)))
    status_write(directory / 'watchdog_status.json', result)
    write_once(directory / 'watchdog_exit.json', result)
    return result


async def _matching_sandboxes(modal, plan, auth, auth_path):
    return [s async for s in modal.Sandbox.list.aio(app_id=plan['app_id'], tags=tags(auth, auth_path))]


async def _stop_owned(modal, sandbox_id):
    """Stop each independently verified owner even if a sibling stop fails."""
    errors=[];code=None
    for attempt in range(3):
        try:
            sandbox=await asyncio.wait_for(modal.Sandbox.from_id.aio(sandbox_id),10)
            await asyncio.wait_for(sandbox.terminate.aio(),10)
            code=await asyncio.wait_for(sandbox.poll.aio(),10)
            if code is not None:break
        except Exception as error:errors.append(repr(error))
        if attempt<2:await asyncio.sleep(2)
    return dict(sandbox_id=sandbox_id,returncode=code,errors=errors,stop_observed=code is not None)


def checked_release_intent(plan_path,plan,auth,auth_path,directory,sandbox_id):
    """The running watcher authenticates its own fresh snapshot before one write."""
    validate_plan(plan_path,admission=True)
    value=read(directory/'release_intent.json')
    expected=dict(schema='final_cloud_release.1',phase_nonce=auth['phase_nonce'],
        authorization_sha256=sha(auth_path),sandbox_id=sandbox_id,app_id=plan['app_id'],watchdog_ready=True)
    if any(value.get(k)!=v for k,v in expected.items()):
        raise ValueError('release intent does not identify this exact owned phase')
    ready=value.get('watchdog_readiness',{})
    readiness_path=directory/'watchdog_readiness.json'
    expected_binding=dict(path=str(readiness_path),sha256=sha(readiness_path))
    canonical=json.dumps(ready,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if value.get('watchdog_binding')!=expected_binding or readiness_path.read_bytes()!=canonical:
        raise ValueError('release readiness binding changed')
    attempt=read(directory/'watchdog_attempt.json')
    exact=dict(ready=True,status='monitoring',phase_nonce=auth['phase_nonce'],pid=os.getpid(),
        sandbox_id=sandbox_id,authorization_sha256=sha(auth_path),plan_sha256=sha(plan_path))
    if (any(ready.get(k)!=v for k,v in exact.items())
            or attempt.get('pid')!=os.getpid() or attempt.get('plan_sha256')!=sha(plan_path)
            or not 0<=time.time()-stamp(ready['at_utc'])<=30):
        raise ValueError('release readiness is stale or belongs to another watchdog')
    return value


async def await_release_receipt(plan_path,sandbox_id,*,timeout=45):
    """Observe the guard's write acknowledgement; never acquire another writer."""
    plan,auth,auth_path,directory=validate_plan(plan_path)
    until=time.monotonic()+timeout
    while True:
        receipt_path=directory/'released.json'
        if receipt_path.exists():
            value=read(receipt_path);intent=read(directory/'release_intent.json')
            expected=dict(schema='final_sandbox_release_receipt.1',status='delivered',sandbox_id=sandbox_id,
                authorization_sha256=sha(auth_path),plan_sha256=sha(plan_path),
                release_sha256=sha(directory/'release_intent.json'),pid=intent['watchdog_readiness']['pid'])
            if any(value.get(k)!=v for k,v in expected.items()):
                raise ValueError('release acknowledgement does not match exact intent/guard')
            if not stamp(intent['watchdog_readiness']['at_utc'])<=stamp(value['at_utc'])<=stamp(auth['deadline_utc']):
                raise ValueError('release acknowledgement chronology changed')
            return value
        if time.monotonic()>=until or time.time()>=stamp(auth['deadline_utc']):
            raise TimeoutError('persistent watchdog did not acknowledge release')
        watcher_ready(directory,auth_path,plan_path=plan_path,sandbox_id=sandbox_id)
        await asyncio.sleep(1)


async def release(plan_path):
    """Prepare one authenticated intent; only the persistent watchdog writes stdin."""
    plan, auth, auth_path, directory = validate_plan(plan_path, admission=True)
    created = read(directory / 'created.json')
    sid = created['sandbox_id']
    if (created['authorization_sha256'] != sha(auth_path) or created['app_id'] != plan['app_id']
            or created.get('plan_sha256') != sha(plan_path)):
        raise ValueError('created receipt does not match launch')
    ready = dict(watcher_ready(directory, auth_path, plan_path=plan_path, sandbox_id=sid), ready=True, phase_nonce=auth['phase_nonce'])
    readiness_path = directory / 'watchdog_readiness.json'
    with readiness_path.open('xb') as stream:
        stream.write(json.dumps(ready, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
        stream.flush(); os.fsync(stream.fileno())
    value = dict(schema='final_cloud_release.1', phase_nonce=auth['phase_nonce'],
        authorization_sha256=sha(auth_path), sandbox_id=sid, app_id=plan['app_id'], watchdog_ready=True,
        watchdog_readiness=ready, watchdog_binding=dict(path=str(directory/'watchdog_readiness.json'),
                                                       sha256=sha(directory/'watchdog_readiness.json')))
    write_once(directory / 'release_intent.json', value)
    return value


async def launch(plan_path):
    """One launchd-owned create/release sequence, never a compute controller."""
    plan, auth, auth_path, directory = validate_plan(plan_path, admission=True)
    write_once(directory/'launcher_attempt.json', dict(schema='final_sandbox_launcher_attempt.1',
        pid=os.getpid(), at_utc=now(), plan_sha256=sha(plan_path), no_retry=True))
    try:
        record=await create(plan_path)
        until=time.monotonic()+45
        while True:
            try:
                watcher_ready(directory,auth_path,plan_path=plan_path,sandbox_id=record['sandbox_id'])
                break
            except (FileNotFoundError,ValueError):
                if time.monotonic()>=until:raise TimeoutError('exact watcher did not acknowledge created Sandbox')
                await asyncio.sleep(2)
        await release(plan_path)
        released=await await_release_receipt(plan_path,record['sandbox_id'])
        result=dict(schema='final_sandbox_launcher_exit.1',status='released',exit_code=0,
            sandbox_id=record['sandbox_id'],plan_sha256=sha(plan_path),at_utc=now(),pid=os.getpid(),
            scientific_completion_claimed=False,release_sha256=sha(directory/'release_intent.json'))
        write_once(directory/'launcher_exit.json',result)
        return result
    except BaseException as error:
        write_once(directory/'launcher_exit.json',dict(schema='final_sandbox_launcher_exit.1',
            status='failed',exit_code=1,at_utc=now(),pid=os.getpid(),plan_sha256=sha(plan_path),
            error=repr(error),no_retry=True,scientific_completion_claimed=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['validate', 'create', 'watch', 'release', 'launch'])
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'validate':
        validate_plan(args.plan, admission=True); print('Valid bounded plan; no cloud call made.')
    else:
        print(json.dumps(asyncio.run(globals()[args.mode](args.plan)), indent=2))


if __name__ == '__main__':
    main()
