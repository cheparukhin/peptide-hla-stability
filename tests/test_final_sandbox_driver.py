"""Lightweight control-plane tests: no provider calls or scientific fitting."""
from datetime import datetime, timezone
import json
import asyncio
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import pytest
from scripts import final_sandbox_driver as d


def utc(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


@pytest.fixture
def planned(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'ROOT', tmp_path)
    clock = d.stamp('2026-10-12T17:00:00+00:00')
    monkeypatch.setattr(d.time, 'time', lambda: clock)
    monkeypatch.setattr(d, 'now', lambda: utc(d.time.time()))
    for name in ('scripts/final_sandbox_driver.py', 'tests/test_final_sandbox_driver.py'):
        p = tmp_path/name; p.parent.mkdir(exist_ok=True, parents=True); p.write_text('synthetic source')
    accounting = dict(schema='final_cloud_accounting.1',prior_conservative_liability_usd=5.1793820030564595,
        future_envelope_usd=35, total_ceiling_usd=100, phase='transfer_report',
        phase_cap_usd=5, reserved_phase_usd=5, conservative_spend_usd=0)
    a = tmp_path/'accounting.json'; a.write_text(json.dumps(accounting))
    auth = dict(schema='final_cloud_phase_authorization.1', approved=True, phase='transfer_report',
        phase_nonce='synthetic-once', resources=dict(cpu_cores=2, memory_mib=8192, numerical_threads=2,max_workers=1),
        fit_wall_seconds=3600, deadline_utc=utc(clock+3600), phase_cap_usd=5,
        image_id='im-synthetic', sandbox_name='synthetic-once',
        cloud_accounting=dict(path=str(a),sha256=d.sha(a)))
    p = tmp_path/'approved.json'; p.write_text(json.dumps(auth))
    directory=tmp_path/'reports/finalization/resources/cloud_sandboxes/synthetic-once'
    plan = dict(schema='final_sandbox_launch.1',approved=True,profile='a-cheparukhin',authorization=dict(path=str(p),sha256=d.sha(p)),
        app_id='ap-synthetic',app_name='synthetic',volume_name='synthetic',volume_id='vo-synthetic',volume_version=2,image_id='im-synthetic',
        phase_directory=str(directory),provider_stop_deadline_utc=utc(clock+3720),phase_cap_usd=5,
        created_at_utc=utc(clock),max_download_bytes=1024**3,
        source_hashes={n:d.sha(tmp_path/n) for n in ('scripts/final_sandbox_driver.py','tests/test_final_sandbox_driver.py')})
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    return path,plan,auth,a,clock


def rewrite(path, value):
    path.write_text(json.dumps(value))


def test_exact_resources_and_bound_arithmetic(planned):
    path,plan,auth,_,_=planned
    got=d.validate_plan(path,admission=True)
    assert got[1]==auth
    assert d.rate(auth)==pytest.approx(.0001322)
    assert (d.stamp(plan['provider_stop_deadline_utc'])-d.stamp(plan['created_at_utc']))*d.rate(auth) <5


@pytest.mark.parametrize('change', ['phase','worker','threads','memory','cpu','bool_worker','wall','phase_cap','image'])
def test_modified_phase_authorization_rejected_even_if_rebound(planned,change):
    path,plan,auth,_,_=planned
    if change=='phase':auth['phase']='arbitrary_grid'
    elif change in ('worker','threads','memory','cpu','bool_worker'):
        key={'worker':'max_workers','threads':'numerical_threads','memory':'memory_mib','cpu':'cpu_cores','bool_worker':'max_workers'}[change]
        auth['resources'][key]=True if change=='bool_worker' else 999
    elif change=='wall':auth['fit_wall_seconds']=999999
    elif change=='phase_cap':auth['phase_cap_usd']=99
    elif change=='image':auth['image_id']='im-different'
    auth_path=Path(plan['authorization']['path']);rewrite(auth_path,auth)
    plan['authorization']['sha256']=d.sha(auth_path);rewrite(path,plan)
    with pytest.raises((ValueError,KeyError)):d.validate_plan(path)


@pytest.mark.parametrize('change', ['expired','future_start','outer_stop','cushion','setup','download','path_escape','source'])
def test_launch_plan_deadlines_paths_and_sources_fail_closed(planned,change):
    path,plan,auth,_,clock=planned
    if change=='expired':plan['created_at_utc']=utc(clock-241)
    elif change=='future_start':plan['created_at_utc']=utc(clock+60)
    elif change=='outer_stop':plan['provider_stop_deadline_utc']='2026-10-15T00:00:00+00:00'
    elif change=='cushion':plan['provider_stop_deadline_utc']=auth['deadline_utc']
    elif change=='setup':plan['created_at_utc']=utc(clock-901)
    elif change=='download':plan['max_download_bytes']=21*1024**3
    elif change=='path_escape':plan['phase_directory']='/tmp/synthetic-once'
    elif change=='source':plan['source_hashes']['scripts/final_sandbox_driver.py']='bad'
    rewrite(path,plan)
    with pytest.raises(ValueError):d.validate_plan(path,admission=True)


@pytest.mark.parametrize('key,value', [
    ('prior_conservative_liability_usd',0),('future_envelope_usd',36),
    ('total_ceiling_usd',101),('reserved_phase_usd',4),
    ('conservative_spend_usd',float('nan')),('future_envelope_usd',-1),
    ('prior_conservative_liability_usd',float('inf')),('conservative_spend_usd',31)])
def test_budget_accounting_cannot_hide_uncertainty(planned,key,value):
    path,plan,auth,a,_=planned
    account=d.read(a);account[key]=value;rewrite(a,account)
    auth['cloud_accounting']['sha256']=d.sha(a)
    auth_path=Path(plan['authorization']['path']);rewrite(auth_path,auth)
    plan['authorization']['sha256']=d.sha(auth_path);rewrite(path,plan)
    with pytest.raises(ValueError):d.validate_plan(path)


def test_immutable_intent_cannot_be_created_twice(tmp_path):
    p=tmp_path/'intent.json';d.write_once(p,{'first':True})
    with pytest.raises(FileExistsError):d.write_once(p,{'first':False})
    assert d.read(p)=={'first':True}


def test_release_requires_fresh_independent_exact_live_guard(planned,monkeypatch):
    path,plan,_,_,clock=planned;directory=Path(plan['phase_directory']);directory.mkdir(parents=True)
    auth_path=Path(plan['authorization']['path'])
    state=dict(status='monitoring',authorization_sha256=d.sha(auth_path),at_utc=utc(clock),pid=12345,sandbox_id='sb-exact',plan_sha256=d.sha(path))
    p=directory/'watchdog_status.json';rewrite(p,state)
    rewrite(directory/'watchdog_attempt.json',dict(pid=12345,plan_sha256=d.sha(path)))
    checked=[];monkeypatch.setattr(d.os,'kill',lambda pid,sig:checked.append((pid,sig)))
    assert d.watcher_ready(directory,auth_path,plan_path=path,sandbox_id='sb-exact')==state
    assert checked==[(12345,0)]
    with pytest.raises(ValueError):d.watcher_ready(directory,auth_path,plan_path=path,sandbox_id='sb-wrong')
    state['at_utc']=utc(clock-31);rewrite(p,state)
    with pytest.raises(ValueError):d.watcher_ready(directory,auth_path,plan_path=path,sandbox_id='sb-exact')
    state['at_utc']=utc(clock);state['pid']=os.getpid();rewrite(p,state)
    with pytest.raises(ValueError):d.watcher_ready(directory,auth_path,plan_path=path,sandbox_id='sb-exact')


def test_dead_guard_never_releases(planned,monkeypatch):
    path,plan,_,_,clock=planned;directory=Path(plan['phase_directory']);directory.mkdir(parents=True)
    auth_path=Path(plan['authorization']['path'])
    rewrite(directory/'watchdog_status.json',dict(status='waiting_for_create',authorization_sha256=d.sha(auth_path),at_utc=utc(clock),pid=12345,plan_sha256=d.sha(path)))
    rewrite(directory/'watchdog_attempt.json',dict(pid=12345,plan_sha256=d.sha(path)))
    def dead(*_):raise ProcessLookupError()
    monkeypatch.setattr(d.os,'kill',dead)
    with pytest.raises(ProcessLookupError):d.watcher_ready(directory,auth_path,plan_path=path)


def test_unbound_authorization_edits_fail_before_provider_calls(planned):
    path,plan,auth,_,_=planned
    auth['fit_wall_seconds']=1;rewrite(Path(plan['authorization']['path']),auth)
    with pytest.raises(ValueError,match='bound input changed'):d.validate_plan(path)


@pytest.fixture
def provider(planned,monkeypatch):
    path,plan,auth,_,clock=planned;directory=Path(plan['phase_directory']);directory.mkdir(parents=True)
    counters=dict(create=0,terminate=0,drain=0,from_id=0);wire=[]
    api_state=dict(active=[],returncode=None,create_error=None,drain_error=None)
    def aio(function):return SimpleNamespace(aio=function)
    async def lookup(*args,**kwargs):return SimpleNamespace(app_id=plan['app_id'])
    async def image(*args,**kwargs):return 'im-synthetic'
    async def hydrate():return None
    volume=SimpleNamespace(object_id='vo-synthetic',hydrate=aio(hydrate))
    async def get_tags():return d.tags(auth,Path(plan['authorization']['path']))
    async def poll():return api_state['returncode']
    async def terminate():counters['terminate']+=1;api_state['returncode']=-15
    async def drain():
        counters['drain']+=1
        if api_state['drain_error']:raise api_state['drain_error']
    sandbox=SimpleNamespace(object_id='sb-exact',get_tags=aio(get_tags),poll=aio(poll),terminate=aio(terminate),
        stdin=SimpleNamespace(write=wire.append,drain=aio(drain)))
    async def create(*args,**kwargs):
        counters['create']+=1
        assert kwargs['cpu']==(2,2) and kwargs['memory']==(8192,8192) and kwargs['gpu'] is None
        assert kwargs['timeout']<=3720
        api_state['active']=[sandbox]
        if api_state['create_error']:raise api_state['create_error']
        return sandbox
    async def from_id(sid):
        counters['from_id']+=1
        assert sid=='sb-exact';return sandbox
    async def listing(**kwargs):
        assert kwargs['app_id']==plan['app_id'] and kwargs['tags']['phase_nonce']==auth['phase_nonce']
        for value in api_state['active']:yield value
    fake=SimpleNamespace(App=SimpleNamespace(lookup=aio(lookup)),Image=SimpleNamespace(from_id=aio(image)),
        Volume=SimpleNamespace(from_name=lambda *a,**kw:volume),
        Sandbox=SimpleNamespace(create=aio(create),from_id=aio(from_id),list=aio(listing)))
    monkeypatch.setitem(sys.modules,'modal',fake)
    monkeypatch.setattr(d.os,'kill',lambda *args:None)
    def ready(status='waiting_for_create'):
        state=dict(status=status,authorization_sha256=d.sha(plan['authorization']['path']),
                   at_utc=utc(clock),pid=12345,plan_sha256=d.sha(path))
        if status=='monitoring':state['sandbox_id']='sb-exact'
        rewrite(directory/'watchdog_status.json',state)
        rewrite(directory/'watchdog_attempt.json',dict(pid=12345,plan_sha256=d.sha(path)))
    ready()
    return counters,api_state,wire,ready,directory


def test_create_is_once_and_never_starts_fit_without_release(planned,provider):
    path,*_=planned;c,state,wire,ready,directory=provider
    result=asyncio.run(d.create(path))
    assert result['sandbox_id']=='sb-exact' and c['create']==1 and wire==[]
    assert (directory/'create_intent.json').exists() and (directory/'created.json').exists()
    with pytest.raises(FileExistsError):asyncio.run(d.create(path))
    assert c['create']==1


def test_uncertain_create_cannot_be_retried(planned,provider):
    path,*_=planned;c,state,wire,_,directory=provider
    state['create_error']=TimeoutError('provider result lost')
    with pytest.raises(TimeoutError):asyncio.run(d.create(path))
    assert (directory/'create_intent.json').exists() and not (directory/'created.json').exists()
    with pytest.raises(FileExistsError):asyncio.run(d.create(path))
    assert c['create']==1 and wire==[]


def test_exact_release_prepares_snapshot_without_acquiring_or_writing_stdin(planned,provider):
    path,*_=planned;c,state,wire,ready,directory=provider
    asyncio.run(d.create(path));ready('monitoring')
    before=c['from_id'];result=asyncio.run(d.release(path))
    sent=d.read(directory/'release_intent.json');assert sent==result
    assert sent['watchdog_readiness']['ready'] is True
    canonical=json.dumps(sent['watchdog_readiness'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    import hashlib
    assert sent['watchdog_binding']['sha256']==hashlib.sha256(canonical).hexdigest()
    with pytest.raises(FileExistsError):asyncio.run(d.release(path))
    assert wire==[] and c['drain']==0 and c['from_id']==before
    assert not (directory/'released.json').exists()


def test_uncertain_guard_release_is_not_repeated(planned,provider,monkeypatch):
    path,*_=planned;c,state,wire,ready,directory=provider
    asyncio.run(d.create(path));ready('monitoring');asyncio.run(d.release(path))
    state['drain_error']=TimeoutError('unknown stdin delivery')
    (directory/'watchdog_attempt.json').unlink()
    monkeypatch.setattr(d.os,'getpid',lambda:12345)
    result=asyncio.run(d.watch(path,interval=0))
    assert result['status']=='failed_stop_observed' and c['terminate']==1
    assert (directory/'release_send_attempt.json').exists() and not (directory/'released.json').exists()
    assert len(wire)==1 and c['drain']==1
    with pytest.raises(FileExistsError):asyncio.run(d.watch(path,interval=0))
    assert len(wire)==1


def test_watch_observes_real_terminal_without_inventing_science(planned,provider):
    path,*_=planned;c,state,wire,ready,directory=provider
    asyncio.run(d.create(path));state['returncode']=0
    (directory/'watchdog_attempt.json').unlink()
    result=asyncio.run(d.watch(path,interval=0))
    assert result['status']=='terminal_observed' and result['returncode']==0
    assert result['reservation_retained_usd']==5 and c['terminate']==0
    assert 'scientific_complete' not in result


def test_watch_deadline_stops_exact_owned_sandbox(planned,provider,monkeypatch):
    path,_,_,_,clock=planned;c,state,wire,ready,directory=provider
    asyncio.run(d.create(path));monkeypatch.setattr(d.time,'time',lambda:clock+3601)
    (directory/'watchdog_attempt.json').unlink()
    result=asyncio.run(d.watch(path,interval=0))
    assert c['terminate']==1 and result['returncode']==-15
    assert 'deadline' in result['reason']


def test_watch_reconciles_ambiguous_create_and_never_releases(planned,provider,monkeypatch):
    path,_,_,_,clock=planned;c,state,wire,ready,directory=provider
    state['create_error']=TimeoutError('lost reply')
    with pytest.raises(TimeoutError):asyncio.run(d.create(path))
    monkeypatch.setattr(d.time,'time',lambda:clock+151)
    (directory/'watchdog_attempt.json').unlink()
    result=asyncio.run(d.watch(path,interval=0))
    assert c['create']==1 and c['terminate']==1 and not wire
    assert 'ambiguous create' in result['reason']


@pytest.mark.parametrize('before_created',[False,True])
def test_duplicate_stop_failure_does_not_skip_other_owned_sandbox(planned,provider,monkeypatch,before_created):
    path,*_=planned;c,state,wire,ready,directory=provider
    asyncio.run(d.create(path))
    if before_created:(directory/'created.json').unlink()
    (directory/'watchdog_attempt.json').unlink()
    attempts=[];codes={'sb-exact':None,'sb-extra':None}
    def make(sid):
        async def terminate():
            attempts.append(sid)
            if sid=='sb-exact':raise RuntimeError('first stop failed')
            codes[sid]=0
        async def poll():return codes[sid]
        return SimpleNamespace(object_id=sid,terminate=SimpleNamespace(aio=terminate),poll=SimpleNamespace(aio=poll))
    objects={sid:make(sid) for sid in codes};state['active']=list(objects.values())
    async def from_id(sid):return objects[sid]
    async def no_sleep(*args):pass
    monkeypatch.setattr(sys.modules['modal'].Sandbox.from_id,'aio',from_id)
    monkeypatch.setattr(d.asyncio,'sleep',no_sleep)
    result=asyncio.run(d.watch(path,interval=0))
    assert attempts.count('sb-exact')==3 and attempts.count('sb-extra')==1
    assert set(result['stop_results'])=={'sb-exact','sb-extra'}
    assert result['stop_results']['sb-exact']['returncode'] is None
    assert result['stop_results']['sb-extra']['returncode']==0
    assert result['status']=='stop_unconfirmed' and result['reservation_retained_usd']==10
    assert wire==[]


def test_stale_guard_plan_and_wrong_release_tags_are_rejected(planned,provider,monkeypatch):
    path,plan,auth,_,_=planned;c,state,wire,ready,directory=provider
    value=d.read(directory/'watchdog_status.json');value['plan_sha256']='0'*64
    rewrite(directory/'watchdog_status.json',value)
    with pytest.raises(ValueError):asyncio.run(d.create(path))
    assert c['create']==0
    ready();asyncio.run(d.create(path));ready('monitoring')
    async def wrong():return {'phase_nonce':'not-owned'}
    state['active'][0].get_tags.aio=wrong
    asyncio.run(d.release(path))
    (directory/'watchdog_attempt.json').unlink()
    monkeypatch.setattr(d.os,'getpid',lambda:12345)
    result=asyncio.run(d.watch(path,interval=0))
    assert 'provider tags differ' in result['reason'] and c['terminate']==1
    assert wire==[]


def test_watch_renews_only_released_exact_phase_lease(planned,provider,monkeypatch):
    path,plan,auth,_,clock=planned;c,state,wire,ready,directory=provider
    asyncio.run(d.create(path));ready('monitoring');asyncio.run(d.release(path))
    (directory/'watchdog_attempt.json').unlink()
    monkeypatch.setattr(d.os,'getpid',lambda:12345)
    async def completed_after_one(*_):state['returncode']=0
    monkeypatch.setattr(d.asyncio,'sleep',completed_after_one)
    result=asyncio.run(d.watch(path,interval=0))
    assert result['status']=='terminal_observed' and len(wire)==2
    lease=json.loads(wire[1])
    assert lease['schema']=='final_cloud_guard_lease.1' and lease['phase_nonce']==auth['phase_nonce']
    assert lease['sequence']==1 and lease['sandbox_id']=='sb-exact'
    assert d.stamp(lease['expires_at_utc'])-d.stamp(lease['issued_at_utc'])==90


def test_guard_task_cancellation_cleans_owned_provider_and_restores_handlers(planned,provider):
    path,*_=planned;c,state,wire,ready,directory=provider
    asyncio.run(d.create(path));(directory/'watchdog_attempt.json').unlink()
    original={sig:d.signal.getsignal(sig) for sig in (d.signal.SIGTERM,d.signal.SIGINT)}
    first=[True]
    async def interrupt_poll():
        if first[0]:
            first[0]=False
            # The installed SIGTERM/SIGINT callbacks invoke exactly Task.cancel.
            asyncio.current_task().cancel()
            await asyncio.sleep(0)
        return state['returncode']
    state['active'][0].poll.aio=interrupt_poll
    result=asyncio.run(d.watch(path,interval=0))
    assert result['status']=='failed_stop_observed' and result['returncode']==-15
    assert c['terminate']==1
    assert all(d.signal.getsignal(sig)==handler for sig,handler in original.items())


@pytest.mark.parametrize('failed', [False, True])
def test_one_shot_launcher_records_release_not_scientific_completion(planned,monkeypatch,failed):
    path,plan,*_=planned;directory=Path(plan['phase_directory']);calls=[]
    async def create(_):
        calls.append('create')
        if failed:raise TimeoutError('unknown provider response')
        return {'sandbox_id':'sb-exact'}
    async def release(_):
        calls.append('release');d.write_once(directory/'release_intent.json',{'sandbox_id':'sb-exact'})
    async def acknowledgement(*_args,**_kwargs):
        calls.append('acknowledgement');return {'status':'delivered'}
    monkeypatch.setattr(d,'create',create);monkeypatch.setattr(d,'release',release)
    monkeypatch.setattr(d,'await_release_receipt',acknowledgement)
    monkeypatch.setattr(d,'watcher_ready',lambda *a,**k:None)
    if failed:
        with pytest.raises(TimeoutError):asyncio.run(d.launch(path))
    else:
        assert asyncio.run(d.launch(path))['status']=='released'
    receipt=d.read(directory/'launcher_exit.json')
    assert receipt['scientific_completion_claimed'] is False
    assert receipt['exit_code']==int(failed)
    original=list(calls)
    with pytest.raises(FileExistsError):asyncio.run(d.launch(path))
    assert calls==original


@pytest.mark.parametrize('stream_mode',['v1_index','v2_offset'])
def test_release_and_renewals_share_one_persistent_stream_writer(planned,provider,monkeypatch,stream_mode):
    path,plan,auth,_,clock=planned;c,state,wire,ready,directory=provider
    asyncio.run(d.create(path));ready('monitoring');asyncio.run(d.release(path))
    assert wire==[] and c['from_id']==0
    (directory/'watchdog_attempt.json').unlink()
    monkeypatch.setattr(d.os,'getpid',lambda:12345)
    clocks={'wall':clock,'mono':1000.};positions=[]
    monkeypatch.setattr(d.time,'time',lambda:clocks['wall'])
    monkeypatch.setattr(d.time,'monotonic',lambda:clocks['mono'])
    server_position=[1 if stream_mode=='v1_index' else 0]
    original=state['active'][0]
    class Writer:
        def __init__(self):
            self.position=1 if stream_mode=='v1_index' else 0;self.buffer=b''
            self.drain=SimpleNamespace(aio=self.flush)
        def write(self,data):self.buffer+=data
        async def flush(self):
            if self.position!=server_position[0]:raise ValueError('new handle resets stream index/offset')
            positions.append(self.position);wire.append(self.buffer)
            advance=1 if stream_mode=='v1_index' else len(self.buffer)
            self.position+=advance;server_position[0]+=advance;self.buffer=b''
    handles=[]
    async def fresh_handle(sid):
        c['from_id']+=1
        assert sid=='sb-exact'
        handle=SimpleNamespace(object_id=sid,stdin=Writer(),get_tags=original.get_tags,poll=original.poll,terminate=original.terminate)
        handles.append(handle);return handle
    monkeypatch.setattr(sys.modules['modal'].Sandbox.from_id,'aio',fresh_handle)
    async def tick(*_):
        clocks['wall']+=11;clocks['mono']+=11
        if len(wire)>=3:state['returncode']=0
    monkeypatch.setattr(d.asyncio,'sleep',tick)
    result=asyncio.run(d.watch(path,interval=0))
    assert result['status']=='terminal_observed' and c['terminate']==0
    assert c['from_id']==1 and len(handles)==1
    messages=[json.loads(chunk) for chunk in wire]
    assert [m['schema'] for m in messages]==['final_cloud_release.1','final_cloud_guard_lease.1','final_cloud_guard_lease.1']
    assert [m['sequence'] for m in messages[1:]]==[1,2]
    assert positions==sorted(set(positions))
    receipt=d.read(directory/'released.json')
    assert receipt['pid']==12345 and receipt['release_sha256']==d.sha(directory/'release_intent.json')
    assert asyncio.run(d.await_release_receipt(path,'sb-exact'))==receipt
    # The mock explicitly reproduces the SDK hazard that the old loop triggered.
    replacement=Writer();replacement.write(b'new handle must not resume this stream\n')
    with pytest.raises(ValueError,match='resets stream'):
        asyncio.run(replacement.flush())


@pytest.mark.parametrize('tamper',['pid','stale','binding','authorization'])
def test_guard_refuses_bad_release_intent_before_any_stdin_write(planned,provider,monkeypatch,tamper):
    path,plan,auth,_,clock=planned;c,state,wire,ready,directory=provider
    asyncio.run(d.create(path));ready('monitoring');asyncio.run(d.release(path))
    value=d.read(directory/'release_intent.json')
    if tamper=='pid':value['watchdog_readiness']['pid']=99999
    elif tamper=='stale':value['watchdog_readiness']['at_utc']=utc(clock-31)
    elif tamper=='binding':value['watchdog_binding']['sha256']='0'*64
    else:value['authorization_sha256']='0'*64
    if tamper in ('pid','stale'):
        data=json.dumps(value['watchdog_readiness'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        (directory/'watchdog_readiness.json').write_bytes(data)
        value['watchdog_binding']['sha256']=d.sha(directory/'watchdog_readiness.json')
    rewrite(directory/'release_intent.json',value)
    (directory/'watchdog_attempt.json').unlink();monkeypatch.setattr(d.os,'getpid',lambda:12345)
    result=asyncio.run(d.watch(path,interval=0))
    assert result['status']=='failed_stop_observed' and c['terminate']==1
    assert wire==[] and c['drain']==0
    assert not (directory/'released.json').exists()


def test_guard_rechecks_readiness_after_durable_send_intent(planned,provider,monkeypatch):
    path,plan,auth,_,clock=planned;c,state,wire,ready,directory=provider
    asyncio.run(d.create(path));ready('monitoring');asyncio.run(d.release(path))
    (directory/'watchdog_attempt.json').unlink();monkeypatch.setattr(d.os,'getpid',lambda:12345)
    wall=[clock];monkeypatch.setattr(d.time,'time',lambda:wall[0])
    original=d.write_once
    def delayed(path,value):
        original(path,value)
        if Path(path).name=='release_send_attempt.json':wall[0]+=31
    monkeypatch.setattr(d,'write_once',delayed)
    result=asyncio.run(d.watch(path,interval=0))
    assert result['status']=='failed_stop_observed' and c['terminate']==1 and wire==[]
    assert (directory/'release_send_attempt.json').exists() and not (directory/'released.json').exists()


def test_launcher_rejects_release_receipt_from_other_writer(planned,provider):
    path,plan,auth,_,clock=planned;c,state,wire,ready,directory=provider
    asyncio.run(d.create(path));ready('monitoring');asyncio.run(d.release(path))
    d.write_once(directory/'released.json',dict(schema='final_sandbox_release_receipt.1',status='delivered',
        pid=99999,at_utc=utc(clock),sandbox_id='sb-exact',plan_sha256=d.sha(path),
        authorization_sha256=d.sha(plan['authorization']['path']),release_sha256=d.sha(directory/'release_intent.json')))
    with pytest.raises(ValueError,match='acknowledgement'):
        asyncio.run(d.await_release_receipt(path,'sb-exact'))
    assert wire==[]


def test_recovery_phase_preserves_original_deadline_and_separate_35_reserve(planned):
    path,plan,auth,account_path,clock=planned
    auth.update(phase='transfer_recovery',phase_cap_usd=10,fit_wall_seconds=21600,
        resources=dict(cpu_cores=4,memory_mib=16384,numerical_threads=2,max_workers=2))
    account=d.read(account_path);account.update(phase='transfer_recovery',phase_cap_usd=10,reserved_phase_usd=10,
        future_envelope_usd=10,other_scopes_reserved_usd=35)
    rewrite(account_path,account);auth['cloud_accounting']['sha256']=d.sha(account_path)
    auth_path=Path(plan['authorization']['path']);rewrite(auth_path,auth)
    plan['authorization']['sha256']=d.sha(auth_path);plan['phase_cap_usd']=10;rewrite(path,plan)
    assert d.validate_plan(path)[1]['phase']=='transfer_recovery'
    plan['provider_stop_deadline_utc']='2026-10-13T15:32:20+00:00';rewrite(path,plan)
    with pytest.raises(ValueError,match='deadline'):d.validate_plan(path)


@pytest.mark.parametrize('observed_code',[0,7,None])
def test_shutdown_notfound_requires_fresh_exact_provider_terminal(planned,provider,monkeypatch,observed_code):
    path,*_=planned;counters,state,wire,ready,directory=provider
    asyncio.run(d.create(path));(directory/'watchdog_attempt.json').unlink()
    class NotFoundError(Exception):pass
    monkeypatch.setattr(sys.modules['modal'],'exception',SimpleNamespace(NotFoundError=NotFoundError),raising=False)
    sandbox=state['active'][0];original_tags=sandbox.get_tags.aio;calls=[0]
    async def racing_tags():
        calls[0]+=1
        if calls[0]>1:
            state['returncode']=observed_code
            raise NotFoundError('exact Sandbox already shut down during tag read')
        return await original_tags()
    sandbox.get_tags.aio=racing_tags
    async def next_loop(*_):pass
    monkeypatch.setattr(d.asyncio,'sleep',next_loop)
    result=asyncio.run(d.watch(path,interval=0))
    if observed_code is not None:
        assert result['status']=='terminal_observed' and result['returncode']==observed_code
        assert counters['terminate']==0 and counters['from_id']==2
        assert 'NotFoundError' in result['shutdown_race_error']
        assert result['shutdown_terminal_observation']['sandbox_id']=='sb-exact'
        assert result['shutdown_terminal_observation']['returncode']==observed_code
        assert result['stop_results']=={}
    else:
        assert result['status']=='failed_stop_observed' and counters['terminate']==1
        assert 'shutdown_terminal_observation' not in result
        assert 'NotFoundError' in result['reason']
    assert not wire and 'scientific_complete' not in result


def test_notfound_without_verified_owner_does_not_infer_completion(planned,provider,monkeypatch):
    path,*_=planned;counters,state,wire,ready,directory=provider
    asyncio.run(d.create(path));(directory/'watchdog_attempt.json').unlink()
    class NotFoundError(Exception):pass
    monkeypatch.setattr(sys.modules['modal'],'exception',SimpleNamespace(NotFoundError=NotFoundError),raising=False)
    async def absent_tags():raise NotFoundError('not verified')
    state['active'][0].get_tags.aio=absent_tags
    result=asyncio.run(d.watch(path,interval=0))
    assert result['status']=='failed_stop_observed' and counters['terminate']==1
    assert 'shutdown_terminal_observation' not in result and not wire
