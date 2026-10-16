from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import numpy as np
import pandas as pd
import pytest
from scripts import final_hybrids as worker
from scripts import final_hybrids_pipeline as pipe


def population():
    return pd.DataFrame([dict(pair_id=g*30+p,cluster_id=p,allele='A',split='train',y_log1p=float(p%7),thalf_hours=float(p%7),peptide='A'*9,hla_pseudoseq='A'*34) for g in range(15) for p in range(30)])


def save(path,value):
    path.write_text(json.dumps(value));return pipe.sha256_file(path)


def authorized(tmp_path,phase='pilot'):
    created=datetime.now(timezone.utc)-timedelta(seconds=2);expiry=created+timedelta(hours=48);protocol={'protocol_hash':'new','source_sha256':{'bound':'hash'}}
    guard_auth=tmp_path/'old_guard_auth.json';guard_hash=save(guard_auth,{'app_id':'ap-old','controller_pid':2147483646,'prior_charges_usd':1.,'total_allocation_usd':65.})
    transfer=tmp_path/'old_transfer_exit.json';driver=tmp_path/'old_driver_exit.json';guard=tmp_path/'old_guard_status.json';reconciled=tmp_path/'old_reconciliation.json'
    th=save(transfer,{'status':'complete','pid':2147483646,'protocol_hash':'old'});dh=save(driver,{'status':'complete','app_context_closed':True,'app_id':'ap-old','pid':2147483646});gh=save(guard,{'phase':'finished','app_stopped':True,'controller_state':'done','guard_pid':2147483645,'authorization_sha256':guard_hash,'budget_verified':True,'effective_commitment_usd':1.})
    rh=save(reconciled,{'schema':'final_scope_reconciliation.1','authorization_sha256':guard_hash,'guard_status_sha256':gh,'cumulative_prior_charges_usd':2.})
    gate_path=tmp_path/'gate.json';path=tmp_path/'approved.json'
    auth={'schema':pipe.SCHEMA,'approved':True,'phase':phase,'protocol_hash':'new','source_sha256':protocol['source_sha256'],'controller_sha256':pipe.sha256_file(Path(pipe.__file__)),'worker_sha256':pipe.sha256_file(Path(worker.__file__)),'jobs':[f'fold_{f}/{r}' for f,r in pipe.phase_jobs(phase)],'max_workers':1 if phase=='pilot' else 2,'threads_each':2,'cloud_spend_usd':0,'n_fits':12 if phase=='pilot' else 480,'created_at_utc':created.isoformat(),'queue_deadline_utc':expiry.isoformat(),'fit_wall_seconds':7200,'launch_gate_path':str(gate_path),'predecessor':{'controller_pid':2147483646,'guard_pid':2147483645,'app_id':'ap-old','protocol_hash':'old','guard_authorization_path':str(guard_auth),'guard_authorization_sha256':guard_hash}}
    save(path,auth)
    gate={'schema':pipe.GATE_SCHEMA,'approved':True,'phase':phase,'authorization_sha256':pipe.sha256_file(path),'created_at_utc':datetime.now(timezone.utc).isoformat(),'expires_at_utc':expiry.isoformat(),'predecessor':{'transfer_exit_path':str(transfer),'transfer_exit_sha256':th,'driver_exit_path':str(driver),'driver_exit_sha256':dh,'guard_status_path':str(guard),'guard_status_sha256':gh,'reconciliation_path':str(reconciled),'reconciliation_sha256':rh}}
    save(gate_path,gate)
    return path,auth,gate,protocol


def test_exact_scientific_schedule_count_and_nested_exclusion():
    frame=population();manifest=worker.task_manifest(frame)
    assert len(manifest['scientific'])==80 and sum(x['n_fits'] for x in manifest['scientific'])==480
    assert len(manifest['engineering_pilot'])==2 and sum(x['n_fits'] for x in manifest['engineering_pilot'])==12
    for item in manifest['scientific']:
        sets=[set(frame.set_index('pair_id').loc[item[k]].cluster_id) for k in ('gradient_rows','stopping_rows','held_rows')]
        assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    assert worker.fit_graph()['additional_leaf_tree_fits']==0
    assert len(worker.preflight(frame))==40


def test_authorization_refuses_changed_caps_counts_and_sources(tmp_path):
    path,auth,gate,protocol=authorized(tmp_path)
    assert pipe.check_authorization(path,protocol,'pilot')['n_fits']==12
    for key,value in [('n_fits',13),('max_workers',2),('cloud_spend_usd',1),('worker_sha256','changed')]:
        changed=dict(auth,**{key:value});save(path,changed)
        with pytest.raises(ValueError,match='mismatch'):pipe.check_authorization(path,protocol,'pilot',require_gate=False)
    save(path,auth);changed=dict(protocol,source_sha256={'different':'hash'})
    with pytest.raises(ValueError,match='every frozen source'):pipe.check_authorization(path,changed,'pilot',require_gate=False)


def test_gate_absent_failed_stale_and_unclosed_predecessor_refuse(tmp_path):
    path,auth,gate,protocol=authorized(tmp_path);gate_path=Path(auth['launch_gate_path'])
    gate_path.unlink()
    with pytest.raises(FileNotFoundError):pipe.check_authorization(path,protocol,'pilot')
    save(gate_path,dict(gate,approved=False))
    with pytest.raises(ValueError,match='approved exact phase'):pipe.check_authorization(path,protocol,'pilot')
    save(gate_path,dict(gate,expires_at_utc=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()))
    with pytest.raises(ValueError,match='stale'):pipe.check_authorization(path,protocol,'pilot')
    driver=Path(gate['predecessor']['driver_exit_path']);dh=save(driver,{'status':'failed','app_context_closed':False,'app_id':'ap-old','pid':2147483646});gate['predecessor']['driver_exit_sha256']=dh;save(gate_path,gate)
    with pytest.raises(ValueError,match='driver has not closed'):pipe.check_authorization(path,protocol,'pilot')


def test_reused_predecessor_pid_blocks_without_signalling(tmp_path,monkeypatch):
    path,auth,gate,protocol=authorized(tmp_path);signals=[]
    monkeypatch.setattr(pipe.os,'kill',lambda pid,sig:signals.append((pid,sig)))
    with pytest.raises(ValueError,match='still present'):pipe.check_authorization(path,protocol,'pilot')
    assert signals==[(2147483646,0)]


def test_queue_waits_for_explicit_root_gate_then_exits_pilot_only(tmp_path,monkeypatch):
    path,auth,gate,protocol=authorized(tmp_path);gate_path=Path(auth['launch_gate_path']);gate_path.unlink();events=[]
    monkeypatch.setattr(pipe,'validate_frozen',lambda out:(population(),protocol))
    def sleep(seconds):events.append(('wait',seconds));save(gate_path,gate)
    def phase(out,approved,phase):
        events.append(('run',phase));directory=Path(out)/'controller'/phase;save(directory/'exit_receipt.json',{'status':'complete'});return {'n_fits':12}
    monkeypatch.setattr(pipe,'run_phase',phase)
    result=pipe.queue_phase(tmp_path,path,'pilot',sleep=sleep)
    assert result['n_fits']==12 and events==[('wait',30),('run','pilot')]
    assert json.loads((tmp_path/'controller/pilot/queue_exit_receipt.json').read_text())['status']=='complete'
    with pytest.raises(FileExistsError):pipe.queue_phase(tmp_path,path,'pilot',sleep=sleep)


def test_failed_child_stops_dispatch_and_receipt_bars_retry(tmp_path,monkeypatch):
    path,auth,gate,protocol=authorized(tmp_path);monkeypatch.setattr(pipe,'validate_frozen',lambda out:(population(),protocol));launches=[]
    class Child:
        pid=123
        def poll(self):return 9
    def popen(command,**kwargs):launches.append((command,kwargs));return Child()
    with pytest.raises(RuntimeError,match='failed'):pipe.run_phase(tmp_path,path,'pilot',popen=popen,sleep=lambda _:None)
    assert len(launches)==1 and launches[0][1]['start_new_session'] is True
    receipt=json.loads((tmp_path/'controller/pilot/exit_receipt.json').read_text());assert receipt['status']=='failed'
    with pytest.raises(FileExistsError):pipe.run_phase(tmp_path,path,'pilot',popen=popen,sleep=lambda _:None)


def test_owned_deadline_kills_only_launched_process_groups(tmp_path,monkeypatch):
    path,auth,gate,protocol=authorized(tmp_path);monkeypatch.setattr(pipe,'validate_frozen',lambda out:(population(),protocol));ticks=iter([0,1,7201,7202]);signals=[]
    class Child:
        pid=123
        dead=False
        def poll(self):return -15 if self.dead else None
        def wait(self,timeout):self.dead=True;return -15
    monkeypatch.setattr(pipe.os,'getpgid',lambda pid:pid);monkeypatch.setattr(pipe.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    with pytest.raises(TimeoutError,match='deadline'):pipe.run_phase(tmp_path,path,'pilot',popen=lambda *a,**k:Child(),monotonic=lambda:next(ticks),sleep=lambda _:None)
    assert signals==[(123,signal.SIGTERM)]
    receipt=json.loads((tmp_path/'controller/pilot/exit_receipt.json').read_text());assert receipt['active_children']==[] and receipt['status']=='failed'


def test_owned_group_identity_mismatch_refuses_signal(monkeypatch):
    class Child:
        pid=123
        def poll(self):return None
        def wait(self,timeout):raise subprocess.TimeoutExpired('fake',timeout)
    signals=[];monkeypatch.setattr(pipe.os,'getpgid',lambda pid:999);monkeypatch.setattr(pipe.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    state={'finished_children':[]};pipe.stop_owned_children({Child():{'pid':123,'process_group':123}},state)
    assert signals==[] and state['cleanup_errors'] and state['active_children'][0]['cleanup_status']=='process_still_uncertain'


def test_mock_component_artifacts_and_held_label_invariance(tmp_path,monkeypatch):
    frame=population();features={r:np.zeros((len(frame),860 if r=='R0' else 1024),dtype=np.float32) for r in worker.REPRESENTATIONS};protocol={'protocol_hash':'synthetic','specification':{'feature_hashes':{r:worker.array_hash(x) for r,x in features.items()}}};calls=[]
    monkeypatch.setattr(worker,'require_authorization',lambda *args:(frame,protocol,{}));monkeypatch.setattr(worker,'validated_features',lambda *args:features)
    approved=tmp_path/'approved.json';approved.write_text('{}')
    def bundle(arrays,gradient,stopping,held,**kwargs):
        assert 'y_log1p' not in held and 'thalf_hours' not in held
        for a,b in ((gradient,stopping),(gradient,held),(stopping,held)):worker.validate_partition(a,b)
        path=Path(kwargs['artifact_dir']);path.mkdir(parents=True);worker.write_once(path/'completed.json',{'n_fits':6});calls.append(kwargs['context'])
        return {'n_fits':6,'predictions':{c:held.cluster_id.to_numpy(dtype=float)+gradient.y_log1p.mean() for c in worker.COMPONENTS},'identity':{'synthetic':True}}
    monkeypatch.setattr(worker,'fit_bundle',bundle)
    first=tmp_path/'first';worker.component_job(first,approved,'R0',0)
    assert len(calls)==8 and worker.verify_component(first/'fold_0/R0',protocol,0,'R0')['n_fits']==48
    predictions=pd.read_csv(first/'fold_0/R0/predictions.csv');assert set(predictions.arm)=={f'R0_{s}' for s in worker.SUFFIXES}
    outer=worker.nested.outer_folds(frame);changed=frame.copy();changed.loc[outer==0,'y_log1p']=-999
    monkeypatch.setattr(worker,'require_authorization',lambda *args:(changed,protocol,{}));second=tmp_path/'second';worker.component_job(second,approved,'R0',0)
    pd.testing.assert_frame_equal(predictions,pd.read_csv(second/'fold_0/R0/predictions.csv'))
    assert json.loads((first/'fold_0/R0/selection.json').read_text())==json.loads((second/'fold_0/R0/selection.json').read_text())
    with pytest.raises(FileExistsError):worker.component_job(first,approved,'R0',0)


def test_sigterm_cleans_owned_children_and_restores_handlers(tmp_path,monkeypatch):
    path,auth,gate,protocol=authorized(tmp_path);monkeypatch.setattr(pipe,'validate_frozen',lambda out:(population(),protocol));signals=[];previous=signal.getsignal(signal.SIGTERM)
    class Child:
        pid=123
        dead=False
        def poll(self):return -15 if self.dead else None
        def wait(self,timeout):self.dead=True;return -15
    monkeypatch.setattr(pipe.os,'getpgid',lambda pid:pid);monkeypatch.setattr(pipe.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    def interrupt(_):signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
    with pytest.raises(RuntimeError,match='interrupted'):pipe.run_phase(tmp_path,path,'pilot',popen=lambda *a,**k:Child(),sleep=interrupt)
    assert signals==[(123,signal.SIGTERM)] and signal.getsignal(signal.SIGTERM)==previous
    assert json.loads((tmp_path/'controller/pilot/exit_receipt.json').read_text())['active_children']==[]


def test_main_failure_terminates_active_sibling_and_never_dispatches_queue(tmp_path,monkeypatch):
    path,auth,gate,protocol=authorized(tmp_path,'main');monkeypatch.setattr(pipe,'validate_frozen',lambda out:(population(),protocol));monkeypatch.setattr(pipe,'check_launch_gate',lambda *args:{});launches=[];signals=[]
    class Child:
        def __init__(self,pid,failed):self.pid=pid;self.code=9 if failed else None
        def poll(self):return self.code
        def wait(self,timeout):self.code=-15;return self.code
    def launch(*args,**kwargs):
        child=Child(123+len(launches),False);launches.append(child)
        if len(launches)==2:launches[0].code=9
        return child
    monkeypatch.setattr(pipe.os,'getpgid',lambda pid:pid);monkeypatch.setattr(pipe.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    with pytest.raises(RuntimeError,match='failed'):pipe.run_phase(tmp_path,path,'main',popen=launch,sleep=lambda _:None)
    assert len(launches)==2 and signals==[(124,signal.SIGTERM)]
    assert json.loads((tmp_path/'controller/main/exit_receipt.json').read_text())['status']=='failed'


def mocked_bundle_integration(tmp_path,monkeypatch):
    from pepstab import final_hybrids as core
    frame=population();features={r:np.zeros((len(frame),860 if r=='R0' else 1024),dtype=np.float32) for r in worker.REPRESENTATIONS};protocol={'protocol_hash':'synthetic','specification':{'feature_hashes':{r:worker.array_hash(x) for r,x in features.items()}}};saved={}
    approved=tmp_path/'approved.json';approved.write_text('{}')
    monkeypatch.setattr(worker,'require_authorization',lambda *args:(frame,protocol,{}));monkeypatch.setattr(worker,'validated_features',lambda *args:features)
    def bundle(arrays,gradient,stopping,held,**kwargs):
        path=Path(kwargs['artifact_dir']);path.mkdir(parents=True)
        identity=core.member_identity(arrays,gradient,stopping,held,representation=kwargs['representation'],seed=kwargs['seed'],context=kwargs['context'],protocol_hash=kwargs['protocol_hash'],source_feature_hash=kwargs['source_feature_hash'])
        predictions={c:held.cluster_id.to_numpy(dtype=float)+float(gradient.y_log1p.mean())+index*.01 for index,c in enumerate(worker.COMPONENTS)}
        result={'n_fits':6,'predictions':predictions,'identity':identity};saved[str(path.resolve())]=result
        worker.write_once(path/'completed.json',{'n_fits':6,'total_fit_seconds':.6,'bundle_seconds':.7,'transform_seconds':.1,'peak_rss_bytes':1024,'component_details':{c:{'fit_seconds':.1} for c in worker.COMPONENTS}})
        return result
    def audit(arrays,gradient,stopping,held,artifact_dir,protocol_hash,expected_identity):
        result=saved[str(Path(artifact_dir).resolve())]
        assert result['identity']==expected_identity
        return result
    monkeypatch.setattr(worker,'fit_bundle',bundle);monkeypatch.setattr(core,'audit_member_bundle',audit)
    return frame,features,protocol,approved


def test_runner_artifacts_feed_independent_report_audit_and_reject_corrupted_aggregate(tmp_path,monkeypatch):
    from scripts import final_hybrids_report as report
    frame,features,protocol,approved=mocked_bundle_integration(tmp_path,monkeypatch)
    worker.component_job(tmp_path,approved,'R0',0)
    table,choice,audit=report.audit_component(tmp_path,protocol,frame,features,'R0',0)
    assert audit['new_supervised_fits']==48 and audit['bundles']==8 and len(audit['runtime'])==8
    assert choice['hybrid_component'] in worker.ORDER and len(table)==90*8
    path=tmp_path/'fold_0/R0/raw_final_predictions.csv';raw=pd.read_csv(path);raw.loc[0,'y_pred']+=1;raw.to_csv(path,index=False)
    # Re-sign the altered aggregate to exercise semantic member replay.
    completed=tmp_path/'fold_0/R0/completed.json';done=json.loads(completed.read_text());done['sha256']['raw_final_predictions.csv']=worker.sha256_file(path);completed.write_text(json.dumps(done))
    with pytest.raises(AssertionError):report.audit_component(tmp_path,protocol,frame,features,'R0',0)


def test_pilot_artifacts_feed_independent_unscored_audit(tmp_path,monkeypatch):
    from scripts import final_hybrids_report as report
    frame,features,protocol,approved=mocked_bundle_integration(tmp_path,monkeypatch)
    for rep in worker.REPRESENTATIONS:worker.component_job(tmp_path,approved,rep,0,pilot=True)
    runtime=report.audit_pilots(tmp_path,protocol,frame,features)
    assert len(runtime)==2 and {r['representation'] for r in runtime}=={'R0','R5'}
    assert all(r['role']=='engineering_pilot' for r in runtime)
    for rep in worker.REPRESENTATIONS:
        directory=tmp_path/'pilot/fold_0'/rep;done=json.loads((directory/'completed.json').read_text())
        assert done['n_fits']==6 and done['held_scores_computed'] is False and done['reuse_into_scientific'] is False
        assert not (directory/'selection.json').exists() and not (directory/'predictions.csv').exists()


def recovery_gate_fixture(tmp_path,monkeypatch):
    path,auth,gate,protocol=authorized(tmp_path)
    root=tmp_path/'recovery';(root/'controller').mkdir(parents=True)
    hashes={}
    for cell in pipe.TRANSFER_CELLS:
        directory=root/cell;directory.mkdir(parents=True)
        payload={}
        for name in pipe.TRANSFER_FILES:
            target=directory/name;target.write_text(cell+'/'+name);payload[name]=pipe.sha256_file(target)
        regime,name=cell.split('/')
        hashes[cell]=save(directory/'completed.json',dict(status='complete',protocol_hash=pipe.TRANSFER_PROTOCOL,
            regime=regime,cell=name,n_fits=166 if regime=='hla' else 400,sha256=payload))
    audit_path=tmp_path/'preserved_audit.json'
    audit_hash=save(audit_path,dict(status='passed',protocol_hash=pipe.TRANSFER_PROTOCOL,n_preserved_cells=18,n_preserved_fits=6030,
        completed_cells=[{'cell':c,'completed_sha256':hashes[c]} for c in pipe.TRANSFER_CELLS[:18]]))
    guard_path=tmp_path/'interrupted_guard_auth.json';deadline='2026-10-13T15:32:19.924192+00:00'
    transfer_protocol_path=root/'protocol.json'
    original_sources={name:pipe.sha256_file(pipe.ROOT/name) for name in ('scripts/final_transfer.py','scripts/final_transfer_report.py')}
    transfer_protocol_hash=save(transfer_protocol_path,dict(protocol_hash=pipe.TRANSFER_PROTOCOL,code_sha256=original_sources))
    guard_hash=save(guard_path,dict(app_id='ap-interrupted',controller_pid=2147483646,deadline_utc=deadline,
        protocol_hashes={str(pipe.ROOT/'reports/finalization/transfer_controls/protocol.json'):transfer_protocol_hash}))
    apps=tmp_path/'stopped_apps.json';apps_hash=save(apps,[dict(app_id='ap-interrupted',state='stopped',tasks='0')])
    closure_path=tmp_path/'closure.json'
    closure=dict(schema='final_interrupted_scope_closure.1',status='interrupted_scope_closed_not_scientifically_complete',
        app_stopped=True,provider_tasks=0,new_cloud_spend_authorized_usd=0,guard_authorization_path=str(guard_path),
        authorization_sha256=guard_hash,app_id='ap-interrupted',controller_pid=2147483646,original_spending_deadline_utc=deadline,
        provider_apps_path=str(apps),provider_apps_sha256=apps_hash,cumulative_prior_charges_usd=2.3,
        prior_pilot_conservative_carry_usd=1.,journal_settled_usd=1.,uncertain_reservations_fully_carried_usd=.1,
        preemption_liability_usd=.1,unobserved_guard_gap_max_concurrency_liability_usd=.1,processes_absent=[2147483646,2147483645])
    closure_hash=save(closure_path,closure)
    recovery_auth_path=tmp_path/'recovery_authorization.json'
    recovery_auth=dict(schema='final_transfer_recovery_authorization.1',approved=True,scope_root=str(root),protocol_hash=pipe.TRANSFER_PROTOCOL,
        max_workers=3,threads_each=2,deadline_utc=deadline,ranker_backend='local',n_new_fits=4800,additional_cloud_spend_usd=0,no_retry=True,
        cells=list(pipe.TRANSFER_CELLS[18:]),closure={'path':str(closure_path),'sha256':closure_hash},
        sources={**original_sources,**{name:pipe.sha256_file(pipe.ROOT/name) for name in ('scripts/final_transfer_recovery.py','tests/test_final_transfer_recovery.py')}},
        preserved_audit={'path':str(audit_path),'sha256':audit_hash},reused_cell_hashes={c:hashes[c] for c in pipe.TRANSFER_CELLS[:18]})
    forecast_path=tmp_path/'runtime_forecast.json';forecast_hash=save(forecast_path,{'conservative_seconds':1000})
    recovery_auth.update(protocol={'path':str(transfer_protocol_path),'sha256':transfer_protocol_hash},runtime_forecast={'path':str(forecast_path),'sha256':forecast_hash})
    preparation_path=tmp_path/'preparation.json'
    preparation=dict(schema='final_transfer_recovery_preparation.1',scope_root=str(root),protocol_hash=pipe.TRANSFER_PROTOCOL,
        **{key:recovery_auth[key] for key in ('protocol','closure','preserved_audit','runtime_forecast','sources','reused_cell_hashes','cells')})
    recovery_auth['preparation']={'path':str(preparation_path),'sha256':save(preparation_path,preparation)}
    recovery_auth_hash=save(recovery_auth_path,recovery_auth)
    attempt_path=root/'controller/attempt.json'
    attempt_hash=save(attempt_path,dict(status='attempted',pid=2147483644,scope_root=str(root),protocol_hash=pipe.TRANSFER_PROTOCOL,
        authorization={'path':str(recovery_auth_path),'sha256':recovery_auth_hash},authorization_sha256=recovery_auth_hash,
        closure=recovery_auth['closure'],deadline_utc=deadline))
    completed_path=root/'controller/completed.json';exit_path=root/'controller/exit_receipt.json'
    completed=dict(schema='final_transfer_recovery_completion.1',status='complete',pid=2147483644,protocol_hash=pipe.TRANSFER_PROTOCOL,
        scope_root=str(root),n_retained_fits=10830,n_new_fits=4800,known_completed_including_discarded=11529,unknown_remote_fits=12,deadline_utc=deadline,
        authorization={'path':str(recovery_auth_path),'sha256':recovery_auth_hash},closure=recovery_auth['closure'],preserved_audit=recovery_auth['preserved_audit'],
        completed_cell_hashes=hashes,reused_cell_hashes=recovery_auth['reused_cell_hashes'],new_cell_hashes={c:hashes[c] for c in pipe.TRANSFER_CELLS[18:]})
    completed_hash=save(completed_path,completed)
    receipt=dict(schema='final_transfer_recovery_exit.1',status='complete',exit_code=0,pid=completed['pid'],protocol_hash=pipe.TRANSFER_PROTOCOL,
        authorization_sha256=recovery_auth_hash,completed_sha256=completed_hash)
    exit_hash=save(exit_path,receipt)
    auth.update(schema=pipe.RECOVERY_SCHEMA,predecessor=dict(kind=pipe.RECOVERY_KIND,closure_path=str(closure_path),closure_sha256=closure_hash,
        recovery_root=str(root),recovery_protocol_hash=pipe.TRANSFER_PROTOCOL,recovery_authorization_path=str(recovery_auth_path),recovery_authorization_sha256=recovery_auth_hash,
        recovery_controller_pid=2147483644,recovery_attempt_path=str(attempt_path),recovery_attempt_sha256=attempt_hash))
    save(path,auth)
    gate.update(schema=pipe.RECOVERY_GATE_SCHEMA,authorization_sha256=pipe.sha256_file(path),predecessor=dict(recovery_completed_path=str(completed_path),
        recovery_completed_sha256=completed_hash,recovery_exit_path=str(exit_path),recovery_exit_sha256=exit_hash))
    save(Path(auth['launch_gate_path']),gate)
    monkeypatch.setattr(pipe,'assert_process_absent',lambda pid:None)
    return path,auth,gate,protocol,completed,receipt,closure


def test_recovery_gate_accepts_closed_interruption_and_complete_thirty_cells(tmp_path,monkeypatch):
    path,auth,gate,protocol,_,_,_=recovery_gate_fixture(tmp_path,monkeypatch)
    assert pipe.check_authorization(path,protocol,'pilot')['schema']==pipe.RECOVERY_SCHEMA
    Path(auth['launch_gate_path']).unlink()
    assert pipe.check_authorization(path,protocol,'pilot',require_gate=False)['n_fits']==12
    with pytest.raises(FileNotFoundError):pipe.check_authorization(path,protocol,'pilot')


def test_recovery_pilot_cannot_extend_fixed_fit_cap(tmp_path,monkeypatch):
    path,auth,_,protocol,_,_,_=recovery_gate_fixture(tmp_path,monkeypatch)
    auth['fit_wall_seconds']=7201;save(path,auth)
    with pytest.raises(ValueError,match='two-hour cap'):
        pipe.check_authorization(path,protocol,'pilot',require_gate=False)


@pytest.mark.parametrize('alteration',['partial_certificate','wrong_fit_count','changed_preserved','failed_exit','missing_exit','mutated_artifact','active_controller','underaccounted_closure','different_receipt_pid','incomplete_sources'])
def test_recovery_gate_refuses_incomplete_mutated_or_active_predecessor(tmp_path,monkeypatch,alteration):
    _,auth,gate,_,completed,receipt,closure=recovery_gate_fixture(tmp_path,monkeypatch)
    closed=gate['predecessor'];root=Path(auth['predecessor']['recovery_root'])
    if alteration=='partial_certificate':completed['completed_cell_hashes'].pop(pipe.TRANSFER_CELLS[-1])
    if alteration=='wrong_fit_count':completed['n_new_fits']=4801
    if alteration=='different_receipt_pid':
        completed['pid']+=1;receipt['pid']=completed['pid']
    if alteration=='changed_preserved':
        completed['reused_cell_hashes'][pipe.TRANSFER_CELLS[0]]='0'*64
        completed['completed_cell_hashes'][pipe.TRANSFER_CELLS[0]]='0'*64
    if alteration in ('partial_certificate','wrong_fit_count','changed_preserved','different_receipt_pid'):
        closed['recovery_completed_sha256']=save(Path(closed['recovery_completed_path']),completed)
        receipt['completed_sha256']=closed['recovery_completed_sha256']
        closed['recovery_exit_sha256']=save(Path(closed['recovery_exit_path']),receipt)
    if alteration=='failed_exit':
        receipt['status']='failed';closed['recovery_exit_sha256']=save(Path(closed['recovery_exit_path']),receipt)
    if alteration=='missing_exit':Path(closed['recovery_exit_path']).unlink()
    if alteration=='mutated_artifact':(root/pipe.TRANSFER_CELLS[-1]/'predictions.csv').write_text('changed after audit')
    if alteration=='active_controller':
        def absent(pid):
            if pid==completed['pid']:raise ValueError('active recovery controller')
        monkeypatch.setattr(pipe,'assert_process_absent',absent)
    if alteration=='underaccounted_closure':
        closure['cumulative_prior_charges_usd']=1.
        auth['predecessor']['closure_sha256']=save(Path(auth['predecessor']['closure_path']),closure)
    if alteration=='incomplete_sources':
        prior=auth['predecessor'];approved_path=Path(prior['recovery_authorization_path']);approved=json.loads(approved_path.read_text())
        approved['sources'].pop('scripts/final_transfer_recovery.py')
        prior['recovery_authorization_sha256']=save(approved_path,approved)
    with pytest.raises((ValueError,FileNotFoundError)):
        pipe.check_recovery_predecessor(auth,gate)


def test_signal_during_spawn_registers_and_cleans_child(tmp_path,monkeypatch):
    path,_,_,protocol=authorized(tmp_path);monkeypatch.setattr(pipe,'validate_frozen',lambda out:(population(),protocol));signals=[]
    class Child:
        pid=123
        dead=False
        def poll(self):return -15 if self.dead else None
        def wait(self,timeout):self.dead=True;return -15
    def spawn(*args,**kwargs):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
        return Child()
    monkeypatch.setattr(pipe.os,'getpgid',lambda pid:pid);monkeypatch.setattr(pipe.os,'killpg',lambda pid,sig:signals.append((pid,sig)))
    with pytest.raises(RuntimeError,match='during spawn'):
        pipe.run_phase(tmp_path,path,'pilot',popen=spawn,sleep=lambda _:None)
    assert signals==[(123,signal.SIGTERM)]
    receipt=json.loads((tmp_path/'controller/pilot/exit_receipt.json').read_text())
    assert receipt['active_children']==[] and len(receipt['finished_children'])==1


def test_known_main_failure_prevents_next_dispatch(tmp_path,monkeypatch):
    path,_,_,protocol=authorized(tmp_path,'main');monkeypatch.setattr(pipe,'validate_frozen',lambda out:(population(),protocol));monkeypatch.setattr(pipe,'check_launch_gate',lambda *args:{});launches=[]
    class Child:
        pid=123
        def poll(self):return 9
    def spawn(*args,**kwargs):launches.append(args);return Child()
    with pytest.raises(RuntimeError,match='failed'):
        pipe.run_phase(tmp_path,path,'main',popen=spawn,sleep=lambda _:None)
    assert len(launches)==1


def test_report_replay_holds_numerical_thread_limit(tmp_path,monkeypatch):
    from scripts import final_hybrids_report as report
    from threadpoolctl import threadpool_info
    observed=[]
    def inner(out):
        observed.extend(threadpool_info())
        assert observed and all(pool['num_threads']<=2 for pool in observed)
        return {'replayed':True}
    monkeypatch.setattr(report,'_build_report',inner)
    assert report.build_report(tmp_path)=={'replayed':True}


def durable_gate_fixture(tmp_path,monkeypatch):
    from scripts import final_transfer_checkpoint_worker
    # The worker's integration tests validate real cache payloads; this fixture isolates gate bindings.
    monkeypatch.setattr(final_transfer_checkpoint_worker,'verify_checkpoint_cell',lambda *args:None)
    # Isolate source bindings as well as receipts; no running experiment or source build is required.
    fixture_root=tmp_path/'repository'
    names=pipe.DURABLE_SOURCES|{'scripts/final_transfer.py','scripts/final_transfer_report.py',
        'scripts/final_transfer_recovery.py','tests/test_final_transfer_recovery.py','scripts/final_hybrids.py'}
    for name in names:
        destination=fixture_root/name;destination.parent.mkdir(parents=True,exist_ok=True)
        source=pipe.ROOT/name
        destination.write_bytes(source.read_bytes() if source.exists() else ('fixture source '+name).encode())
    monkeypatch.setattr(pipe,'ROOT',fixture_root)
    path,auth,gate,protocol,completed,receipt,closure=recovery_gate_fixture(tmp_path,monkeypatch)
    prior=auth['predecessor'];root=Path(prior['recovery_root']);approved_path=Path(prior['recovery_authorization_path'])
    approved=json.loads(approved_path.read_text());preparation_path=Path(approved['preparation']['path'])
    preparation=json.loads(preparation_path.read_text())
    def bound(path,value):return {'path':str(path),'sha256':save(path,value)}
    old=tmp_path/'old_local';old.mkdir()
    failed_exit=bound(old/'exit_receipt.json',dict(schema='final_transfer_recovery_exit.1',status='failed',exit_code=1,
        cleanup_error=None,pid=2147483643,deadline_utc=pipe.TRANSFER_DEADLINE,protocol_hash=pipe.TRANSFER_PROTOCOL,closure=approved['closure']))
    failed_state=bound(old/'state.json',dict(status='failed',cleanup_error=None,pid=2147483643,finished_cells=[]))
    failed_closure=bound(tmp_path/'local_closure.json',dict(schema='final_local_recovery_interruption.1',
        status='failed_stopped_no_new_complete_cells',cleanup_error=None,controller_pid=2147483643,hybrid_queue_pid=2147483642,
        additional_discarded_recorded_completed_fits=1086,total_discarded_recorded_completed_fits=1785,
        known_completed_after_eventual_success_including_discarded=12615,remaining_scientific_fits=4800,
        retained_complete_cells=18,retained_complete_fits=6030,uncertain_remote_attempts_from_original_scope=12,
        additional_cloud_spend_usd=0,controller_exit_path=failed_exit['path'],controller_exit_sha256=failed_exit['sha256'],
        controller_state_sha256=failed_state['sha256']))
    failure=dict(local_failure_closure=failed_closure,local_failure_exit=failed_exit,local_failure_state=failed_state)
    approved.update(schema='final_transfer_durable_authorization.1',**failure,old_local_pids=[2147483642,2147483643])
    original_sources=json.loads(Path(approved['protocol']['path']).read_text())['code_sha256']
    approved['sources']={**original_sources,**{n:pipe.sha256_file(pipe.ROOT/n) for n in pipe.DURABLE_SOURCES}}
    preparation.update(schema='final_transfer_durable_preparation.1',**failure,old_local_pids=approved['old_local_pids'],
        sources=approved['sources'],n_reused_fits=6030,n_new_fits=4800,n_retained_fits=10830,
        discarded_partial_completed_fits=1785,known_completed_including_discarded=12615,unknown_remote_fits=12,
        unknown_local_fits=True,additional_cloud_spend_usd=0,previous_local_exit_status='failed',
        local_failure_evidence={item['path']:item['sha256'] for item in (failed_exit,failed_state)},
        amendment=bound(tmp_path/'amendment.json',{'original_protocol_hash':pipe.TRANSFER_PROTOCOL}),
        local_failure_amendment=bound(tmp_path/'local_amendment.json',{'scientific_protocol_hash':pipe.TRANSFER_PROTOCOL}),
        copied_file_hashes={c:{str(f.relative_to(root/c)):pipe.sha256_file(f) for f in (root/c).rglob('*') if f.is_file()}
                           for c in pipe.TRANSFER_CELLS[:18]})
    approved['preparation']['sha256']=save(preparation_path,preparation)
    prior.update(kind=pipe.DURABLE_KIND,**failure,recovery_authorization_sha256=save(approved_path,approved))
    exact_auth={'path':str(approved_path),'sha256':prior['recovery_authorization_sha256']}
    attempt_path=Path(prior['recovery_attempt_path']);attempt=json.loads(attempt_path.read_text())
    attempt.update(started_at_utc='2026-10-12T16:00:00+00:00',authorization=exact_auth,authorization_sha256=exact_auth['sha256'],
                   local_failure_closure=failed_closure,local_failure_exit=failed_exit)
    prior['recovery_attempt_sha256']=save(attempt_path,attempt)
    completed.update(schema='final_transfer_durable_completion.1',**failure,previous_local_exit_status='failed',
        discarded_partial_completed_fits=1785,known_completed_including_discarded=12615,unknown_local_fits=True,
        authorization=exact_auth,authorization_sha256=exact_auth['sha256'],closure_sha256=prior['closure_sha256'],
        finished_at_utc='2026-10-13T14:00:00+00:00')
    closed=gate['predecessor'];closed['recovery_completed_sha256']=save(Path(closed['recovery_completed_path']),completed)
    receipt.update(schema='final_transfer_durable_exit.1',scope_root=str(root),authorization=exact_auth,
        authorization_sha256=exact_auth['sha256'],closure=approved['closure'],local_failure_closure=failed_closure,
        local_failure_exit=failed_exit,deadline_utc=pipe.TRANSFER_DEADLINE,completed_sha256=closed['recovery_completed_sha256'],
        completed_certificate={'path':closed['recovery_completed_path'],'sha256':closed['recovery_completed_sha256']},
        finished_at_utc='2026-10-13T14:00:01+00:00')
    closed['recovery_exit_sha256']=save(Path(closed['recovery_exit_path']),receipt)
    monkeypatch.setattr(pipe,'DURABLE_OUT',tmp_path/'hybrids_v3')
    auth.update(schema=pipe.DURABLE_SCHEMA,scope_root=str(pipe.DURABLE_OUT),queue_deadline_utc=pipe.QUEUE_DEADLINE,
                created_at_utc='2026-10-12T16:00:00+00:00')
    save(path,auth)
    gate.update(schema=pipe.DURABLE_GATE_SCHEMA,authorization_sha256=pipe.sha256_file(path),
                created_at_utc='2026-10-13T16:00:00+00:00',expires_at_utc=pipe.QUEUE_DEADLINE)
    save(Path(auth['launch_gate_path']),gate)
    monkeypatch.setattr(pipe.time,'time',lambda:pipe.timestamp('2026-10-13T16:00:01+00:00'))
    return path,auth,gate,protocol,completed,receipt,approved,preparation


def test_durable_gate_accepts_both_closed_attempts_and_exact_success_even_after_transfer_deadline(tmp_path,monkeypatch):
    path,auth,gate,protocol,_,_,_,_=durable_gate_fixture(tmp_path,monkeypatch)
    assert pipe.check_authorization(path,protocol,'pilot')['schema']==pipe.DURABLE_SCHEMA
    Path(auth['launch_gate_path']).unlink()
    assert pipe.check_authorization(path,protocol,'pilot',require_gate=False)['n_fits']==12
    with pytest.raises(FileNotFoundError):pipe.check_authorization(path,protocol,'pilot')


@pytest.mark.parametrize('alteration',[
    'failed_durable_exit','wrong_owner','changed_attempt','source_missing','source_changed','cloud_open',
    'local_success_fabricated','local_cleanup_failed','local_closure_count','local_state_complete','local_evidence_changed',
    'preparation_count','partial_cells','wrong_retained','wrong_discarded','wrong_total','changed_artifact',
    'certificate_auth','certificate_failure','exit_certificate','late_completion','late_exit','active_owner',
    'wrong_auth_schema','wrong_gate_schema','queue_extension','gate_extension','pilot_extension','wrong_output'])
def test_durable_gate_rejects_tampered_evidence(tmp_path,monkeypatch,alteration):
    path,auth,gate,protocol,completed,receipt,approved,preparation=durable_gate_fixture(tmp_path,monkeypatch)
    prior=auth['predecessor'];closed=gate['predecessor']
    if alteration=='failed_durable_exit':receipt.update(status='failed',exit_code=1)
    if alteration=='wrong_owner':completed['pid']+=1;receipt['pid']=completed['pid']
    if alteration=='changed_attempt':
        target=Path(prior['recovery_attempt_path']);value=json.loads(target.read_text());value['pid']+=1;prior['recovery_attempt_sha256']=save(target,value)
    if alteration=='source_missing':approved['sources'].pop(next(iter(pipe.DURABLE_SOURCES)))
    if alteration=='source_changed':approved['sources']['scripts/final_transfer_durable.py']='0'*64
    if alteration=='cloud_open':
        target=Path(prior['closure_path']);value=json.loads(target.read_text());value['app_stopped']=False;prior['closure_sha256']=save(target,value)
    if alteration in ('local_success_fabricated','local_cleanup_failed','local_state_complete','local_closure_count'):
        key='local_failure_exit' if alteration in ('local_success_fabricated','local_cleanup_failed') else 'local_failure_state' if alteration=='local_state_complete' else 'local_failure_closure'
        target=Path(prior[key]['path']);value=json.loads(target.read_text())
        if alteration=='local_success_fabricated':value['status']='complete';value['exit_code']=0
        elif alteration=='local_cleanup_failed':value['cleanup_error']='still running'
        elif alteration=='local_state_complete':value['status']='complete'
        else:value['total_discarded_recorded_completed_fits']=699
        item={'path':str(target),'sha256':save(target,value)}
        prior[key]=approved[key]=preparation[key]=item
        # Rebind outer ownership so semantic checks, rather than stale hashes, reject the mutation.
        if key!='local_failure_closure':
            target=Path(prior['local_failure_closure']['path']);value=json.loads(target.read_text())
            value['controller_exit_sha256' if key=='local_failure_exit' else 'controller_state_sha256']=item['sha256']
            prior['local_failure_closure']=approved['local_failure_closure']=preparation['local_failure_closure']={'path':str(target),'sha256':save(target,value)}
    if alteration=='local_evidence_changed':Path(approved['local_failure_state']['path']).write_text('{}')
    if alteration=='preparation_count':preparation['known_completed_including_discarded']=11529
    if alteration=='partial_cells':completed['completed_cell_hashes'].pop(pipe.TRANSFER_CELLS[-1])
    if alteration=='wrong_retained':completed['n_retained_fits']=10829
    if alteration=='wrong_discarded':completed['discarded_partial_completed_fits']=699
    if alteration=='wrong_total':completed['known_completed_including_discarded']=11529
    if alteration=='changed_artifact':(Path(prior['recovery_root'])/pipe.TRANSFER_CELLS[-1]/'predictions.csv').write_text('changed')
    if alteration=='certificate_auth':completed['authorization_sha256']='0'*64
    if alteration=='certificate_failure':completed['local_failure_exit']={}
    if alteration=='late_completion':completed['finished_at_utc']='2026-10-13T16:00:00+00:00'
    if alteration=='late_exit':receipt['finished_at_utc']='2026-10-13T16:00:00+00:00'
    if alteration=='active_owner':
        def absent(pid):
            if pid==prior['recovery_controller_pid']:raise ValueError('active durable owner')
        monkeypatch.setattr(pipe,'assert_process_absent',absent)
    approved['preparation']['sha256']=save(Path(approved['preparation']['path']),preparation)
    old_auth_hash=prior['recovery_authorization_sha256']
    prior['recovery_authorization_sha256']=save(Path(prior['recovery_authorization_path']),approved)
    if old_auth_hash!=prior['recovery_authorization_sha256']:
        exact={'path':prior['recovery_authorization_path'],'sha256':prior['recovery_authorization_sha256']}
        target=Path(prior['recovery_attempt_path']);attempt=json.loads(target.read_text())
        attempt.update(authorization=exact,authorization_sha256=exact['sha256'],local_failure_closure=approved['local_failure_closure'],local_failure_exit=approved['local_failure_exit'])
        prior['recovery_attempt_sha256']=save(target,attempt)
    closed['recovery_completed_sha256']=save(Path(closed['recovery_completed_path']),completed)
    receipt['completed_sha256']=closed['recovery_completed_sha256'];receipt['completed_certificate']['sha256']=closed['recovery_completed_sha256']
    if alteration=='exit_certificate':receipt['completed_certificate']['sha256']='0'*64
    closed['recovery_exit_sha256']=save(Path(closed['recovery_exit_path']),receipt)
    if alteration=='wrong_output':auth['scope_root']=str(tmp_path/'old_scope')
    if alteration=='wrong_auth_schema':auth['schema']=pipe.RECOVERY_SCHEMA
    if alteration=='wrong_gate_schema':gate['schema']=pipe.RECOVERY_GATE_SCHEMA
    if alteration=='queue_extension':auth['queue_deadline_utc']='2026-10-14T15:00:00+00:00'
    if alteration=='gate_extension':gate['expires_at_utc']='2026-10-14T15:00:00+00:00'
    if alteration=='pilot_extension':auth['fit_wall_seconds']=7201
    gate['authorization_sha256']=save(path,auth);save(Path(auth['launch_gate_path']),gate)
    with pytest.raises(ValueError):pipe.check_authorization(path,protocol,'pilot')


def test_durable_main_needs_separate_runtime_gate_and_fixed_ten_jobs(tmp_path,monkeypatch):
    path,auth,gate,protocol,_,_,_,_=durable_gate_fixture(tmp_path,monkeypatch)
    auth.update(phase='main',max_workers=2,n_fits=480,jobs=[f'fold_{f}/{r}' for f,r in pipe.phase_jobs('main')])
    gate.update(phase='main',authorization_sha256=save(path,auth))
    pilot=dict(status='complete',phase='pilot',n_fits=12,protocol_hash=protocol['protocol_hash'],pid=2147483641)
    pilot_completed=tmp_path/'pilot_completed.json';pilot_exit=tmp_path/'pilot_exit.json';forecast=tmp_path/'pilot_forecast.json'
    gate['pilot']=dict(completed_path=str(pilot_completed),completed_sha256=save(pilot_completed,pilot),
        exit_path=str(pilot_exit),exit_sha256=save(pilot_exit,pilot),runtime_forecast_path=str(forecast),
        runtime_forecast_sha256=save(forecast,{'conservative_seconds':1000}))
    save(Path(auth['launch_gate_path']),gate)
    with pytest.raises(ValueError,match='separate root scientific runtime approval'):
        pipe.check_authorization(path,protocol,'main')
    gate['runtime_approved']=True;save(Path(auth['launch_gate_path']),gate)
    actual=pipe.check_authorization(path,protocol,'main')
    assert len(actual['jobs'])==10 and actual['n_fits']==480 and actual['max_workers']==2
    auth['max_workers']=3;save(path,auth)
    with pytest.raises(ValueError,match='resources mismatch'):pipe.check_authorization(path,protocol,'main',require_gate=False)


@pytest.mark.parametrize('mode',['queue_phase','run_phase'])
def test_durable_controller_refuses_other_output_before_writing_ownership(tmp_path,monkeypatch,mode):
    path,_,_,protocol,_,_,_,_=durable_gate_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(pipe,'validate_frozen',lambda out:(population(),protocol))
    wrong=tmp_path/'wrong_output'
    with pytest.raises(ValueError,match='out-dir differs'):getattr(pipe,mode)(wrong,path,'pilot')
    assert not wrong.exists()


def test_v2_cannot_accept_durable_completion_or_failed_old_controller(tmp_path,monkeypatch):
    path,auth,gate,protocol,completed,receipt,_=recovery_gate_fixture(tmp_path,monkeypatch)
    receipt['status']='failed';receipt['exit_code']=1
    gate['predecessor']['recovery_exit_sha256']=save(Path(gate['predecessor']['recovery_exit_path']),receipt)
    save(Path(auth['launch_gate_path']),gate)
    with pytest.raises(ValueError,match='successful exact recovery controller exit'):
        pipe.check_authorization(path,protocol,'pilot')
    completed.update(schema='final_transfer_durable_completion.1',known_completed_including_discarded=12615)
    gate['predecessor']['recovery_completed_sha256']=save(Path(gate['predecessor']['recovery_completed_path']),completed)
    save(Path(auth['launch_gate_path']),gate)
    with pytest.raises(ValueError,match='complete exact thirty-cell recovery'):
        pipe.check_authorization(path,protocol,'pilot')


def test_durable_gate_revalidates_every_new_checkpoint_cell_and_blocks_corruption(tmp_path,monkeypatch):
    from scripts import final_transfer_checkpoint_worker
    path,auth,_,protocol,_,_,approved,_=durable_gate_fixture(tmp_path,monkeypatch)
    checked=[]
    def verify(root,regime,cell,bound):
        assert root==Path(auth['predecessor']['recovery_root'])
        assert bound==approved
        checked.append(f'{regime}/{cell}')
    monkeypatch.setattr(final_transfer_checkpoint_worker,'verify_checkpoint_cell',verify)
    pipe.check_authorization(path,protocol,'pilot')
    assert checked==list(pipe.TRANSFER_CELLS[18:])
    def corrupt(*args):raise ValueError('checkpoint payload hash mismatch after completion')
    monkeypatch.setattr(final_transfer_checkpoint_worker,'verify_checkpoint_cell',corrupt)
    with pytest.raises(ValueError,match='checkpoint payload hash mismatch'):
        pipe.check_authorization(path,protocol,'pilot')
