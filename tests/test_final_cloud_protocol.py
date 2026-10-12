"""Small contract/source-equivalence checks. No scientific fits or cloud calls."""
import ast
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import copy
import pytest
from pepstab import final_cloud_protocol as c
from scripts import final_cloud_phase as runner

REPO=Path(__file__).resolve().parents[1]


def save(root,path,value):
    target=root/path;target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(value,sort_keys=True))
    return {'path':path,'sha256':c.sha(target)}


def test_scientific_worker_functions_are_exact_ast_copies():
    original=ast.parse((REPO/'scripts/final_hybrids.py').read_text())
    cloud=ast.parse((REPO/'scripts/final_hybrids_cloud.py').read_text())
    functions=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    a,b=functions(original),functions(cloud)
    assert set(a)==set(b)
    for name in set(a)-{'fit_bundle','runtime','require_authorization','main'}:
        assert a[name]==b[name],name


def test_scientific_report_audits_and_whole_science_tail_unchanged():
    a=(REPO/'scripts/final_hybrids_report.py').read_text()
    b=(REPO/'scripts/final_hybrids_cloud_report.py').read_text()
    funcs=lambda s:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
    af,bf=funcs(a),funcs(b)
    assert set(af)==set(bf)
    for name in set(af)-{'build_report','_build_report'}:assert af[name]==bf[name],name
    tail=lambda s:s[s.index('    features, _ = load_features(train)',s.index('def _build_report')):s.index('\n    return primary')]
    a=tail(a)
    b=tail(b)
    replacement="""controller_receipts=controller_receipts, cloud_conservative_accounted_liability_usd=cloud_auth['_accounting']['conservative_spend_usd'],
        cloud_current_phase_reservation_usd=cloud_auth['_accounting']['reserved_phase_usd'],
        cloud_accounting=cloud_auth['cloud_accounting'], backend='modal_cpu_sandbox',
        cloud_cost_interpretation='Conservative preceding-phase accounted liability and current phase reservation at authorization; neither field is actual invoiced spend. Final provider reconciliation is separate.',"""
    assert a.replace('controller_receipts=controller_receipts, cloud_spend_usd=0,',replacement)==b


def test_paths_duplicate_json_and_bound_bytes_reject(tmp_path):
    reference=save(tmp_path,'safe.json',{'ok':1})
    assert c.read(c.bound(reference,tmp_path))=={'ok':1}
    (tmp_path/'safe.json').write_text('{"ok":2}')
    with pytest.raises(ValueError,match='changed'):c.bound(reference,tmp_path)
    for bad in ['../outside','/etc/passwd']:
        with pytest.raises(ValueError):c.path_in_root(bad,tmp_path)
    (tmp_path/'link').symlink_to(tmp_path/'safe.json')
    with pytest.raises(ValueError,match='symlink'):c.path_in_root('link',tmp_path)
    (tmp_path/'duplicate.json').write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError,match='duplicate'):c.read(tmp_path/'duplicate.json')
    (tmp_path/'nan.json').write_text('{"a":NaN}')
    with pytest.raises(ValueError,match='non-finite'):c.read(tmp_path/'nan.json')


def phase_certificate(tmp_path,phase='hybrid_pilot'):
    auth_hash='a'*64;nonce='abcdefghijkl';phase_dir=f'reports/finalization/cloud/remote/{nonce}'
    files={}
    if phase=='hybrid_pilot':
        ref=save(tmp_path,f'{phase_dir}/pilot_audit.json',dict(schema='final_cloud_pilot_audit.1',status='passed',n_fits=12,
            held_scores_computed=False,reuse_into_scientific=False,saved_model_predictions_replayed=True,
            protocol_hash=c.SCIENCE_HASH,authorization_sha256=auth_hash))
        files[ref['path']]=ref['sha256']
    else:
        ref=save(tmp_path,'result.json',{'ok':True});files[ref['path']]=ref['sha256']
    output=save(tmp_path,f'{phase_dir}/output_manifest.json',dict(schema='final_cloud_output_manifest.1',phase=phase,files=files))
    common=dict(phase=phase,phase_nonce=nonce,sandbox_id='sb-example',authorization_sha256=auth_hash,input_manifest_sha256='b'*64,
        protocol_hash=c.SCIENCE_HASH,n_fits=c.PHASES[phase][-1],started_at_utc='2026-10-12T18:00:00+00:00',finished_at_utc='2026-10-12T19:00:00+00:00',elapsed_fit_seconds=3600)
    receipt=save(tmp_path,f'{phase_dir}/exit_receipt.json',dict(common,schema='final_cloud_phase_exit.1',status='complete',exit_code=0,
        cleanup_complete=True,real_process_owner={'pid':17,'hostname':'sandbox','pid_namespace':'pid:[123]'},output_manifest=output))
    item=dict(common,schema='finalization_sandbox_phase_completion.1',status='complete',app_id='ap-example',runner_exit=receipt,
        output_manifest=output,provider_terminal={'sandbox_id':'sb-example','returncode':0,'observed_at_utc':'2026-10-12T19:00:01+00:00'})
    return save(tmp_path,'certificate.json',item),item


def test_true_cloud_phase_certificate_and_unscored_replay(tmp_path):
    ref,item=phase_certificate(tmp_path)
    assert c.verify_phase_certificate(ref,'hybrid_pilot',tmp_path)['n_fits']==12
    for key,value in [('n_fits',480),('status','failed'),('phase','hybrid_main')]:
        changed=dict(item,**{key:value});bad=save(tmp_path,'bad.json',changed)
        with pytest.raises(ValueError):c.verify_phase_certificate(bad,'hybrid_pilot',tmp_path)
    changed=copy.deepcopy(item);changed['provider_terminal']['returncode']=1
    with pytest.raises(ValueError,match='provider'):c.verify_phase_certificate(save(tmp_path,'bad.json',changed),'hybrid_pilot',tmp_path)
    changed=copy.deepcopy(item);changed['provider_terminal']['sandbox_id']='sb-wrong'
    with pytest.raises(ValueError,match='provider'):c.verify_phase_certificate(save(tmp_path,'bad.json',changed),'hybrid_pilot',tmp_path)
    (tmp_path/item['output_manifest']['path']).write_text('{}')
    with pytest.raises(ValueError,match='changed'):c.verify_phase_certificate(ref,'hybrid_pilot',tmp_path)


def authorization_fixture(tmp_path,monkeypatch):
    sources={}
    for name in c.NEW_SOURCES+('historical.py',):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# source\n');sources[name]=c.sha(path)
    historical=save(tmp_path,'historical_authorization.json',{'sources':{'historical.py':sources['historical.py']}})
    predecessor=save(tmp_path,'predecessor.json',{'authorization':historical})
    monkeypatch.setattr(c,'verify_predecessor',lambda ref,root:{'authorization':historical})
    account=save(tmp_path,'account.json',dict(schema='final_cloud_accounting.1',phase='transfer_report',
        prior_conservative_liability_usd=5.1793820030564595,future_envelope_usd=35,total_ceiling_usd=100,
        phase_cap_usd=5,reserved_phase_usd=5,conservative_spend_usd=0))
    files=dict(sources)
    for ref in (historical,predecessor,account):files[ref['path']]=ref['sha256']
    manifest=save(tmp_path,'manifest.json',dict(schema='final_cloud_input_manifest.1',repository_root=c.REPOSITORY_ROOT,files=files))
    auth=dict(schema='final_cloud_phase_authorization.1',approved=True,phase='transfer_report',scope_root=c.SCOPE,
        resources={'cpu_cores':2,'memory_mib':8192,'numerical_threads':2,'max_workers':1},fit_wall_seconds=28800,
        deadline_utc='2026-10-13T02:00:00+00:00',sandbox_name='bounded-report',phase_nonce='abcdefghijkl',
        run_directory='reports/finalization/cloud/remote/abcdefghijkl',sources=sources,input_manifest=manifest,
        portable_predecessor=predecessor,phase_certificates={},cloud_accounting=account,phase_cap_usd=5,
        image_id='im-exact',durability={'volume_version':2,'method':'os.sync'})
    save(tmp_path,'authorization.json',auth)
    return auth


def test_authorization_resource_deadline_and_manifest_gates(tmp_path,monkeypatch):
    auth=authorization_fixture(tmp_path,monkeypatch)
    check=lambda:c.validate_authorization('authorization.json',root=tmp_path,clock=lambda:c.timestamp('2026-10-12T18:00:00+00:00'))
    assert check()['phase']=='transfer_report'
    mutations=[('approved',False),('fit_wall_seconds',28801),('fit_wall_seconds',float('inf')),('phase_cap_usd',6),
        ('deadline_utc','2026-10-15T00:00:00+00:00'),('deadline_utc','2026-10-12T17:00:00+00:00'),
        ('run_directory','reports/wrong'),('phase_nonce','tiny'),('scope_root','elsewhere'),
        ('resources',dict(auth['resources'],numerical_threads=4)),('durability',{'volume_version':1,'method':'os.sync'})]
    for key,value in mutations:
        save(tmp_path,'authorization.json',dict(auth,**{key:value}))
        with pytest.raises(ValueError):check()
    save(tmp_path,'authorization.json',auth)
    (tmp_path/'historical.py').write_text('changed')
    with pytest.raises(ValueError,match='changed'):check()


def test_source_coverage_is_not_optional(tmp_path,monkeypatch):
    auth=authorization_fixture(tmp_path,monkeypatch)
    auth['sources'].pop(c.NEW_SOURCES[0]);save(tmp_path,'authorization.json',auth)
    with pytest.raises(ValueError,match='all cloud execution'):
        c.validate_authorization('authorization.json',root=tmp_path,clock=lambda:c.timestamp('2026-10-12T18:00:00+00:00'))


def release_payload(auth):
    ready={'ready':True,'phase_nonce':auth['phase_nonce'],'authorization_sha256':auth['_sha256'],'sandbox_id':'sb-real',
           'status':'monitoring','pid':42,'at_utc':'2026-10-12T18:00:00+00:00'}
    digest=hashlib.sha256(json.dumps(ready,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return dict(schema='final_cloud_release.1',phase_nonce=auth['phase_nonce'],authorization_sha256=auth['_sha256'],
        sandbox_id='sb-real',app_id='ap-real',watchdog_ready=True,watchdog_readiness=ready,
        watchdog_binding={'path':'local/watchdog.json','sha256':digest})


def test_exact_stdin_release_without_foreign_pid_probing(monkeypatch):
    auth={'phase_nonce':'abcdefghijkl','_sha256':'a'*64,'deadline_utc':'2026-10-13T02:00:00+00:00'}
    monkeypatch.setattr(runner.time,'time',lambda:c.timestamp('2026-10-12T18:00:00+00:00'))
    release=release_payload(auth)
    for value,valid in [(release,True),(dict(release,phase_nonce='wrong'),False),(dict(release,watchdog_ready=False),False)]:
        read_fd,write_fd=os.pipe()
        with os.fdopen(write_fd,'w') as writer:writer.write(json.dumps(value)+'\n')
        with os.fdopen(read_fd,'r') as reader:
            if valid:assert runner.wait_release(auth,reader,1)['sandbox_id']=='sb-real'
            else:
                with pytest.raises(ValueError):runner.wait_release(auth,reader,1)
    with pytest.raises(ValueError):runner.wait_release(auth,None,301)


def test_original_transfer_and_bounded_worker_commands():
    assert len(runner.phase_jobs('hybrid_main'))==10
    assert runner.phase_jobs('hybrid_pilot')==[(0,'R0'),(0,'R5')]
    command=runner.child_command({'phase':'transfer_report'},Path('/auth'),(None,None))
    assert command[1]==str(REPO/'scripts/final_transfer_report.py')
    command=runner.child_command({'phase':'hybrid_main'},Path('/auth'),(3,'R5'))
    assert command[2]=='run-component' and command[-4:]==['--fold','3','--representation','R5']


def test_failed_historical_closure_cannot_be_inferred_from_missing_pid(tmp_path,monkeypatch):
    monkeypatch.setattr(os,'kill',lambda *_:pytest.fail('portable validation must never probe local PIDs'))
    ref=save(tmp_path,'closure.json',dict(schema='finalization_portable_transfer_closure.1',status='failed'))
    with pytest.raises(ValueError,match='successful'):c.verify_predecessor(ref,tmp_path)
    ref=save(tmp_path,'closure.json',dict(schema='finalization_portable_transfer_closure.1',status='complete',owner={'pid':59308,'absent':True}))
    with pytest.raises(ValueError,match='exact Mac owner'):c.verify_predecessor(ref,tmp_path)


def test_process_cleanup_reaps_true_owned_children(monkeypatch):
    events=[]
    class Child:
        pid=456
        _cloud_owned_pid=456
        status=None
        def poll(self):return self.status
        def wait(self,timeout):self.status=-15;return self.status
    monkeypatch.setattr(runner.os,'killpg',lambda pid,sig:events.append((pid,sig)))
    monkeypatch.setattr(runner.os,'getpgid',lambda pid:pid)
    child=Child()
    assert runner.cleanup_children([child])
    assert events==[(456,runner.signal.SIGTERM)]


def test_no_global_scientific_or_runtime_monkeypatch():
    for name in c.NEW_SOURCES:
        text=(REPO/name).read_text()
        for forbidden in ('os.kill =','sys.version =','platform.python_version =','check_authorization ='):
            assert forbidden not in text


def test_supervisor_failed_child_stops_dispatch_and_preserves_one_attempt(tmp_path,monkeypatch):
    class FakeRaw:
        def __init__(self,stream):pass
        def drain(self):return []
        def restore(self):pass
    monkeypatch.setattr(runner,'RawJsonLines',FakeRaw)
    auth=authorization_fixture(tmp_path,monkeypatch)
    auth['_sha256']=c.sha(tmp_path/'authorization.json')
    release=release_payload(auth)
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'verify_environment',lambda:{'python':'3.12.12'})
    monkeypatch.setattr(runner,'wait_release',lambda *args:release)
    monkeypatch.setattr(c,'path_in_root',lambda name,root=None:tmp_path/name)
    monkeypatch.setattr(c,'validate_authorization',lambda *args:auth)
    monkeypatch.setattr(c,'durable_sync',lambda:None)
    monkeypatch.setattr(runner.time,'time',lambda:c.timestamp('2026-10-12T18:00:00+00:00'))
    launches=[]
    class Child:
        pid=17
        def poll(self):return 9
        def wait(self,timeout):return 9
    def popen(command,**kwargs):launches.append((command,kwargs));return Child()
    with pytest.raises(RuntimeError,match='failed; no retry'):
        runner.run_phase(tmp_path/'authorization.json',popen=popen,sleep=lambda _:None)
    directory=tmp_path/auth['run_directory']
    assert len(launches)==1 and launches[0][1]['start_new_session'] is True
    receipt=c.read(directory/'exit_receipt.json')
    assert receipt['status']=='failed' and receipt['n_fits']==0 and receipt['cleanup_complete'] is True
    with pytest.raises(FileExistsError):
        runner.run_phase(tmp_path/'authorization.json',popen=popen,sleep=lambda _:None)
    assert len(launches)==1


def test_supervisor_never_launches_before_exact_release(tmp_path,monkeypatch):
    class FakeRaw:
        def __init__(self,stream):pass
        def drain(self):return []
        def restore(self):pass
    monkeypatch.setattr(runner,'RawJsonLines',FakeRaw)
    auth=authorization_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(c,'path_in_root',lambda name,root=None:tmp_path/name)
    monkeypatch.setattr(c,'durable_sync',lambda:None)
    monkeypatch.setattr(runner,'verify_environment',lambda:{})
    def release(*args):raise TimeoutError('no watchdog ready')
    monkeypatch.setattr(runner,'wait_release',release)
    with pytest.raises(TimeoutError):
        runner.run_phase(tmp_path/'authorization.json',popen=lambda *_a,**_k:pytest.fail('must never launch'))
    receipt=c.read(tmp_path/auth['run_directory']/'exit_receipt.json')
    assert receipt['status']=='failed' and receipt['no_retry'] is True
    assert not (tmp_path/auth['run_directory']/'attempt.json').exists()


def test_existing_outputs_refuse_new_nonce_retry(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    directory=tmp_path/c.TRANSFER_ROOT/'results';directory.mkdir(parents=True)
    with pytest.raises(ValueError,match='existing'):
        runner.require_fresh_outputs({'phase':'transfer_report'})
    directory=tmp_path/c.SCOPE/'fold_0';directory.mkdir(parents=True)
    with pytest.raises(ValueError,match='existing'):
        runner.require_fresh_outputs({'phase':'hybrid_main'})


def test_bundle_hook_commits_intent_and_model_before_acknowledgement(tmp_path,monkeypatch):
    from scripts import final_hybrids_cloud as cloud
    from pepstab import final_hybrids as core
    events=[]
    directory=tmp_path/'bundle';directory.mkdir()
    def fit(*args,**kwargs):
        events.append(('fit',args,kwargs));(directory/'completed.json').write_text('{}');return {'n_fits':6}
    monkeypatch.setattr(core,'fit_member_bundle',fit)
    monkeypatch.setattr(c,'durable_sync',lambda:events.append('sync'))
    monkeypatch.setattr(c,'binding',lambda p:{'path':'bundle/completed.json','sha256':c.sha(p)})
    result=cloud.fit_bundle('features','gradient','stopping','held',artifact_dir=directory,seed=1)
    assert result=={'n_fits':6}
    assert events[0]=='sync' and events[1][0]=='fit' and events[2:] == ['sync','sync']
    assert events[1][1]==('features','gradient','stopping','held')
    assert events[1][2]==dict(artifact_dir=directory,seed=1)
    assert c.read(directory/'cloud_commit.json')['volume_version']==2


def test_commit_failure_does_not_start_a_fit(tmp_path,monkeypatch):
    from scripts import final_hybrids_cloud as cloud
    from pepstab import final_hybrids as core
    def fail():raise OSError('durability unavailable')
    monkeypatch.setattr(c,'durable_sync',fail)
    monkeypatch.setattr(core,'fit_member_bundle',lambda *args,**kwargs:pytest.fail('must not fit'))
    with pytest.raises(OSError,match='durability'):
        cloud.fit_bundle(artifact_dir=tmp_path)


def test_portable_transfer_real_receipt_linkage_and_foreign_runtime(tmp_path,monkeypatch):
    # One fixture cell exercises receipt binding without building a scientific dataset.
    monkeypatch.setattr(c,'CELLS',('hla/h0',))
    source=tmp_path/'original.py';source.write_text('# frozen\n')
    history=save(tmp_path,'historical_closure.json',{'status':'historical'})
    frame={name:c.sha(source) for name in ['original.py']}
    directory=f'{c.TRANSFER_ROOT}/hla/h0';files={}
    for name in ('predictions.csv','selection.json','ancestry.json','members.json','runtime.json','raw_inner_predictions.csv','raw_final_predictions.csv','member_predictions.npz'):
        ref=save(tmp_path,f'{directory}/{name}',{'fixture':True});files[name]=ref['sha256']
    cell=save(tmp_path,f'{directory}/completed.json',dict(status='complete',protocol_hash=c.TRANSFER_HASH,regime='hla',cell='h0',n_fits=166,sha256=files))
    histkeys=('closure','local_failure_closure','local_failure_exit','local_failure_state','preserved_audit')
    authorization=dict(approved=True,sources=frame,reused_cell_hashes={'hla/h0':cell['sha256']},**{k:history for k in histkeys})
    auth=save(tmp_path,'original_auth.json',authorization)
    monkeypatch.setattr(c,'TRANSFER_AUTH_HASH',auth['sha256'])
    attempt=save(tmp_path,'attempt.json',dict(pid=59308,authorization_sha256=auth['sha256'],started_at_utc='2026-10-12T16:09:00+00:00'))
    complete=dict(schema='final_transfer_durable_completion.1',status='complete',pid=59308,protocol_hash=c.TRANSFER_HASH,
        deadline_utc=c.TRANSFER_DEADLINE,n_retained_fits=10830,n_new_fits=4800,discarded_partial_completed_fits=1785,
        known_completed_including_discarded=12615,unknown_remote_fits=12,unknown_local_fits=True,authorization_sha256=auth['sha256'],
        finished_at_utc='2026-10-12T20:00:00+00:00',scope_root=f'{c.REPOSITORY_ROOT}/{c.TRANSFER_ROOT}',
        completed_cell_hashes={'hla/h0':cell['sha256']},reused_cell_hashes={'hla/h0':cell['sha256']},new_cell_hashes={},**{k:history for k in histkeys})
    completion=save(tmp_path,'complete.json',complete)
    exit_payload=dict(schema='final_transfer_durable_exit.1',status='complete',pid=59308,exit_code=0,
        authorization_sha256=auth['sha256'],completed_sha256=completion['sha256'],finished_at_utc='2026-10-12T20:00:01+00:00')
    receipt=save(tmp_path,'exit.json',exit_payload)
    cert=dict(schema='finalization_portable_transfer_closure.1',status='complete',authorization=auth,attempt=attempt,completion=completion,exit=receipt,
        checked_at_utc='2026-10-12T20:01:00+00:00',evidence_files=frame,
        owner=dict(pid=59308,host='originalMac',boot_id='actual-boot',command='original exact command',started_at_utc='2026-10-12T16:09:00+00:00',
            launchd_label='org.pepstab.transfer-recovery-v2-20261012',absent=True,children_absent=True))
    ref=save(tmp_path,'portable.json',cert)
    monkeypatch.setattr(os,'kill',lambda *_:pytest.fail('never probe Mac PID from this runtime'))
    assert c.verify_predecessor(ref,tmp_path)['owner']['host']=='originalMac'
    for key,value in [('status','failed'),('pid',123),('exit_code',1),('completed_sha256','0'*64)]:
        bad=save(tmp_path,'bad_exit.json',dict(exit_payload,**{key:value}))
        bad_ref=save(tmp_path,'bad_portable.json',dict(cert,exit=bad))
        with pytest.raises(ValueError,match='successful exit'):c.verify_predecessor(bad_ref,tmp_path)
    bad=save(tmp_path,'bad_exit.json',dict(exit_payload,finished_at_utc='2026-10-14T00:00:00+00:00'))
    with pytest.raises(ValueError,match='chronology'):
        c.verify_predecessor(save(tmp_path,'bad_portable.json',dict(cert,exit=bad)),tmp_path)


@pytest.fixture
def guard_pipe():
    start=c.timestamp('2026-10-12T18:00:00+00:00');clock=[start];mono=[0.]
    auth={'phase_nonce':'abcdefghijkl','_sha256':'a'*64,'deadline_utc':'2026-10-12T18:20:00+00:00'}
    release=release_payload(auth)
    read_fd,write_fd=os.pipe()
    with os.fdopen(read_fd,'rb',buffering=0) as stream:
        reader=runner.RawJsonLines(stream)
        lease=runner.GuardLease(auth,release,reader,clock=lambda:clock[0],monotonic=lambda:mono[0])
        state={'auth':auth,'release':release,'reader':reader,'lease':lease,'write_fd':write_fd,'clock':clock,'mono':mono,'start':start}
        yield state
        reader.restore()
    try:os.close(write_fd)
    except OSError:pass


def lease_message(state,sequence=1,seconds=90):
    issued=state['clock'][0]
    return dict(schema='final_cloud_guard_lease.1',phase_nonce=state['auth']['phase_nonce'],
        authorization_sha256=state['auth']['_sha256'],sandbox_id='sb-real',watchdog_pid=42,sequence=sequence,
        issued_at_utc=datetime.fromtimestamp(issued,timezone.utc).isoformat(),
        expires_at_utc=datetime.fromtimestamp(issued+seconds,timezone.utc).isoformat())


def send_guard(state,message):
    os.write(state['write_fd'],(json.dumps(message)+'\n').encode())


def test_real_pipe_guard_lease_expires_and_cannot_be_revived(guard_pipe):
    state=guard_pipe;state['lease'].check()
    state['clock'][0]+=90;state['mono'][0]+=90
    send_guard(state,lease_message(state))
    with pytest.raises(TimeoutError,match='expired'):state['lease'].check()


def test_real_pipe_guard_renewal_sequence_and_partial_lines(guard_pipe):
    state=guard_pipe;state['clock'][0]+=10;state['mono'][0]+=10
    message=(json.dumps(lease_message(state))+'\n').encode()
    os.write(state['write_fd'],message[:30]);state['lease'].check()
    assert state['lease'].sequence==0
    os.write(state['write_fd'],message[30:]);state['lease'].check()
    assert state['lease'].sequence==1 and state['lease'].expires==state['start']+100
    state['clock'][0]+=80;state['mono'][0]+=80
    send_guard(state,lease_message(state,sequence=2));state['lease'].check()
    assert state['lease'].sequence==2 and state['lease'].expires==state['start']+180
    send_guard(state,lease_message(state,sequence=2))
    with pytest.raises(ValueError,match='sequence'):state['lease'].check()


@pytest.mark.parametrize('mutation',[{'sandbox_id':'sb-wrong'},{'phase_nonce':'wrong'},
    {'authorization_sha256':'b'*64},{'watchdog_pid':99},{'sequence':0},{'sequence':True},
    {'issued_at_utc':'2026-10-12T18:00:01+00:00'},
    {'expires_at_utc':'2026-10-12T18:01:31+00:00'},
    {'expires_at_utc':'2026-10-12T18:21:00+00:00'}])
def test_real_pipe_bad_guard_leases_fail_closed(guard_pipe,mutation):
    state=guard_pipe;message=lease_message(state);message.update(mutation);send_guard(state,message)
    with pytest.raises(ValueError):state['lease'].check()


def test_real_pipe_guard_eof_is_failure_even_with_buffered_renewal(guard_pipe):
    state=guard_pipe;send_guard(state,lease_message(state));os.close(state['write_fd'])
    with pytest.raises(EOFError,match='after release'):state['lease'].check()


def test_raw_release_readahead_keeps_first_guard_lease(guard_pipe,monkeypatch):
    state=guard_pipe;monkeypatch.setattr(runner.time,'time',lambda:state['clock'][0])
    data=json.dumps(state['release'])+'\n'+json.dumps(lease_message(state))+'\n'
    os.write(state['write_fd'],data.encode())
    result=runner.wait_release(state['auth'],state['reader'],1)
    assert result['sandbox_id']=='sb-real'
    state['lease'].check();assert state['lease'].sequence==1


def test_guard_monotonic_expiry_survives_wall_clock_reversal(guard_pipe):
    state=guard_pipe;state['clock'][0]-=300;state['mono'][0]+=90
    with pytest.raises(TimeoutError,match='expired'):state['lease'].check()


def test_guard_input_buffer_is_bounded(guard_pipe):
    state=guard_pipe
    state['reader'].buffer=b'x'*(state['reader'].MAX_LINE+1)+b'\n'
    with pytest.raises(ValueError,match='too long'):state['lease'].check()


@pytest.mark.parametrize('stop_kind',['signal','deadline'])
def test_stop_after_durable_intent_prevents_popen(tmp_path,monkeypatch,stop_kind):
    class FakeRaw:
        def __init__(self,stream):pass
        def drain(self):return []
        def restore(self):pass
    auth=authorization_fixture(tmp_path,monkeypatch);auth['_sha256']=c.sha(tmp_path/'authorization.json')
    release=release_payload(auth);clock=[c.timestamp('2026-10-12T18:00:00+00:00')]
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'RawJsonLines',FakeRaw)
    monkeypatch.setattr(runner,'verify_environment',lambda:{})
    monkeypatch.setattr(runner,'wait_release',lambda *args:release)
    monkeypatch.setattr(c,'path_in_root',lambda name,root=None:tmp_path/name)
    monkeypatch.setattr(c,'validate_authorization',lambda *args:auth)
    monkeypatch.setattr(c,'durable_sync',lambda:None)
    monkeypatch.setattr(runner.time,'time',lambda:clock[0])
    original=c.exclusive_json
    def write(path,item):
        original(path,item)
        if str(path).endswith('child_000.intent.json'):
            if stop_kind=='signal':runner.signal.getsignal(runner.signal.SIGTERM)(runner.signal.SIGTERM,None)
            else:clock[0]=c.timestamp(auth['deadline_utc'])
    monkeypatch.setattr(c,'exclusive_json',write)
    with pytest.raises((RuntimeError,TimeoutError)):
        runner.run_phase(tmp_path/'authorization.json',popen=lambda *_a,**_k:pytest.fail('late stop must prevent Popen'))
    receipt=c.read(tmp_path/auth['run_directory']/'exit_receipt.json')
    assert receipt['status']=='failed' and receipt['cleanup_complete'] is True


def test_guard_monitor_services_renewals_during_blocked_parent_preflight(guard_pipe):
    # Waiting on an event keeps this bounded and avoids timing-dependent sleeps.
    import threading
    state=guard_pipe;accepted=threading.Event();original=state['lease'].accept
    def accept(message):
        original(message);accepted.set()
    state['lease'].accept=accept
    state['lease'].start_monitor()
    try:
        state['clock'][0]+=10;state['mono'][0]+=10
        send_guard(state,lease_message(state))
        assert accepted.wait(2)
        assert state['lease'].sequence==1
    finally:state['lease'].stop_monitor()


def test_guard_monitor_eof_error_is_sticky_for_parent(guard_pipe):
    import threading
    state=guard_pipe;finished=threading.Event();original=state['lease']._check_unlocked
    def checked():
        try:original()
        except BaseException:finished.set();raise
    state['lease']._check_unlocked=checked
    state['lease'].start_monitor()
    try:
        os.close(state['write_fd']);assert finished.wait(2)
        with pytest.raises(EOFError):state['lease'].check()
    finally:state['lease'].stop_monitor()


def test_expired_lease_stops_registered_child_while_parent_volume_sync_blocks(tmp_path,monkeypatch):
    import threading
    class FakeRaw:
        def __init__(self,stream):pass
        def drain(self):return []
        def restore(self):pass
    class Child:
        pid=456
        status=None
        def poll(self):return self.status
        def wait(self,timeout):
            assert self.status is not None,'monitor must signal before reaping'
            return self.status
    auth=authorization_fixture(tmp_path,monkeypatch);auth['_sha256']=c.sha(tmp_path/'authorization.json')
    release=release_payload(auth);clock=[c.timestamp('2026-10-12T18:00:00+00:00')]
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'RawJsonLines',FakeRaw)
    monkeypatch.setattr(runner,'verify_environment',lambda:{})
    monkeypatch.setattr(runner,'wait_release',lambda *args:release)
    monkeypatch.setattr(c,'path_in_root',lambda name,root=None:tmp_path/name)
    monkeypatch.setattr(c,'validate_authorization',lambda *args:auth)
    monkeypatch.setattr(runner.time,'time',lambda:clock[0])
    entered=threading.Event();stopped=threading.Event();child=Child();launches=[];signals=[]
    sync_should_block=[False]
    original=c.exclusive_json
    def write(path,item):
        original(path,item)
        if str(path).endswith('child_000.owner.json'):sync_should_block[0]=True
    monkeypatch.setattr(c,'exclusive_json',write)
    def sync():
        if sync_should_block[0]:
            sync_should_block[0]=False;entered.set()
            assert stopped.wait(3),'lease monitor must stop child independently of blocked parent'
    monkeypatch.setattr(c,'durable_sync',sync)
    monkeypatch.setattr(runner.os,'getpgid',lambda pid:pid)
    def kill(pid,sig):
        signals.append((pid,sig));child.status=-15;stopped.set()
    monkeypatch.setattr(runner.os,'killpg',kill)
    def popen(*args,**kwargs):launches.append(args);return child
    def expire():
        assert entered.wait(3)
        clock[0]+=91
    timer=threading.Thread(target=expire);timer.start()
    try:
        with pytest.raises(TimeoutError,match='lease expired'):
            runner.run_phase(tmp_path/'authorization.json',popen=popen,sleep=lambda _:None)
    finally:timer.join(timeout=3)
    assert len(launches)==1 and signals==[(456,runner.signal.SIGTERM)]
    receipt=c.read(tmp_path/auth['run_directory']/'exit_receipt.json')
    assert receipt['status']=='failed' and receipt['cleanup_complete'] is True


def test_child_cleanup_refuses_unowned_process_group_without_signalling(monkeypatch):
    class Child:
        pid=456
        _cloud_owned_pid=456
        def poll(self):return None
        def wait(self,timeout):return 0
    signals=[]
    monkeypatch.setattr(runner.os,'getpgid',lambda pid:999)
    monkeypatch.setattr(runner.os,'killpg',lambda *args:signals.append(args))
    with pytest.raises(RuntimeError,match='owned cleanup incomplete'):
        runner.cleanup_children([Child()],grace_seconds=0)
    assert signals==[]
