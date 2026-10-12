"""Durable LOCAL queue and fixed-deadline subprocess controller; no cloud/retry."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pepstab.boltz_embeddings import sha256_file
from scripts.final_hybrids import OUT,atomic_json,write_once,validate_frozen,verify_component
SCHEMA='final_hybrids_controller_authorization.1'
GATE_SCHEMA='final_hybrids_launch_gate.1'
RECOVERY_SCHEMA='final_hybrids_controller_authorization.2'
RECOVERY_GATE_SCHEMA='final_hybrids_launch_gate.2'
RECOVERY_KIND='interrupted_transfer_local_recovery'
DURABLE_SCHEMA='final_hybrids_controller_authorization.3'
DURABLE_GATE_SCHEMA='final_hybrids_launch_gate.3'
DURABLE_KIND='interrupted_transfer_durable_recovery'
DURABLE_OUT=ROOT/'reports/finalization/hybrids_recovery_v2'
QUEUE_DEADLINE='2026-10-14T14:15:17.567539+00:00'
TRANSFER_DEADLINE='2026-10-13T15:32:19.924192+00:00'
DURABLE_SOURCES={'scripts/final_transfer_durable.py','tests/test_final_transfer_durable.py',
                 'pepstab/final_transfer_checkpoint.py','scripts/final_transfer_checkpoint_worker.py',
                 'tests/test_final_transfer_checkpoint.py'}
TRANSFER_PROTOCOL='bcda56bc3ae36dd757ef1a3210b167fc926d1604be74bb7487c5e18fedaf412d'
TRANSFER_CELLS=tuple(f'hla/h{i}' for i in range(5))+tuple(f'joint/h{i}_p{j}' for i in range(5) for j in range(5))
TRANSFER_FILES={'predictions.csv','selection.json','ancestry.json','members.json','runtime.json','raw_inner_predictions.csv','raw_final_predictions.csv','member_predictions.npz'}


def now():return datetime.now(timezone.utc).isoformat()

def timestamp(value):
    item=datetime.fromisoformat(value.replace('Z','+00:00'))
    if item.tzinfo is None or item.utcoffset().total_seconds()!=0:raise ValueError('explicit UTC deadline required')
    return item.timestamp()

def phase_jobs(phase):
    if phase=='pilot':return tuple((0,rep) for rep in ('R0','R5'))
    if phase=='main':return tuple((fold,rep) for fold in range(5) for rep in ('R0','R5'))
    raise ValueError('phase outside separate pilot/main authorizations')

def assert_process_absent(pid):
    if type(pid)!=int or pid<=1:raise ValueError('valid exact predecessor PID required')
    try:os.kill(pid,0)
    except ProcessLookupError:return
    raise ValueError('prerequisite controller/guard process still present')

def read_bound(path,digest):
    path=Path(path)
    if sha256_file(path)!=digest:raise ValueError('hash-bound prerequisite receipt changed')
    return json.loads(path.read_text())

def check_recovery_closure(prior,*,durable=False):
    """A stopped interrupted attempt is closure, never scientific success."""
    if prior.get('kind')!=(DURABLE_KIND if durable else RECOVERY_KIND) or prior.get('recovery_protocol_hash')!=TRANSFER_PROTOCOL:
        raise ValueError('unknown recovery prerequisite')
    closure=read_bound(prior['closure_path'],prior['closure_sha256'])
    expected=dict(schema='final_interrupted_scope_closure.1',status='interrupted_scope_closed_not_scientifically_complete',app_stopped=True,provider_tasks=0,new_cloud_spend_authorized_usd=0)
    if any(closure.get(k)!=v for k,v in expected.items()) or closure.get('app_stopped') is not True:
        raise ValueError('interrupted cloud scope is not explicitly closed')
    guard=read_bound(closure['guard_authorization_path'],closure['authorization_sha256'])
    if closure['app_id']!=guard['app_id'] or closure['controller_pid']!=guard['controller_pid'] or closure['original_spending_deadline_utc']!=guard['deadline_utc']:
        raise ValueError('closed predecessor ownership/deadline changed')
    apps=read_bound(closure['provider_apps_path'],closure['provider_apps_sha256'])
    matching=[a for a in apps if a.get('app_id')==closure['app_id']]
    if len(matching)!=1 or matching[0].get('state')!='stopped' or int(matching[0].get('tasks',-1))!=0:
        raise ValueError('exact interrupted app is not stopped with zero tasks')
    amount=closure['cumulative_prior_charges_usd']
    liabilities=[closure[k] for k in ('prior_pilot_conservative_carry_usd','journal_settled_usd','uncertain_reservations_fully_carried_usd','preemption_liability_usd','unobserved_guard_gap_max_concurrency_liability_usd')]
    if any(isinstance(v,bool) or not math.isfinite(v) or v<0 for v in liabilities):raise ValueError('invalid closed interruption liability')
    minimum=sum(liabilities)
    if not math.isfinite(amount) or not math.isfinite(minimum) or not 0<=minimum<=amount+1e-10 or amount>100:
        raise ValueError('closed interruption liabilities underaccounted')
    for pid in closure['processes_absent']:assert_process_absent(pid)
    approved=read_bound(prior['recovery_authorization_path'],prior['recovery_authorization_sha256'])
    expected=dict(schema='final_transfer_durable_authorization.1' if durable else 'final_transfer_recovery_authorization.1',approved=True,scope_root=str(Path(prior['recovery_root']).resolve()),protocol_hash=TRANSFER_PROTOCOL,max_workers=3,threads_each=2,deadline_utc=guard['deadline_utc'],ranker_backend='local',n_new_fits=4800,additional_cloud_spend_usd=0,no_retry=True)
    if any(approved.get(k)!=v for k,v in expected.items()) or approved.get('approved') is not True or approved.get('cells')!=list(TRANSFER_CELLS[18:]):
        raise ValueError('recovery authorization changes frozen scope/resources')
    if approved['closure']!={'path':str(Path(prior['closure_path']).resolve()),'sha256':prior['closure_sha256']}:
        raise ValueError('recovery authorization binds another closure')
    preparation=read_bound(approved['preparation']['path'],approved['preparation']['sha256'])
    protocol=read_bound(approved['protocol']['path'],approved['protocol']['sha256'])
    original_protocol=str(ROOT/'reports/finalization/transfer_controls/protocol.json')
    if approved['protocol']['sha256']!=guard['protocol_hashes'][original_protocol] or protocol['protocol_hash']!=TRANSFER_PROTOCOL:
        raise ValueError('recovery protocol differs from original frozen transfer')
    required=set(protocol['code_sha256'])|(DURABLE_SOURCES if durable else {'scripts/final_transfer_recovery.py','tests/test_final_transfer_recovery.py'})
    if set(approved['sources'])!=required or any(approved['sources'][k]!=v for k,v in protocol['code_sha256'].items()):
        raise ValueError('recovery source binding coverage changed')
    if preparation.get('schema')!=('final_transfer_durable_preparation.1' if durable else 'final_transfer_recovery_preparation.1') or preparation.get('scope_root')!=approved['scope_root'] or preparation.get('protocol_hash')!=TRANSFER_PROTOCOL:
        raise ValueError('recovery preparation identity changed')
    for key in ('protocol','closure','preserved_audit','runtime_forecast','sources','reused_cell_hashes','cells'):
        if preparation.get(key)!=approved[key]:raise ValueError('recovery preparation binding changed')
    read_bound(approved['runtime_forecast']['path'],approved['runtime_forecast']['sha256'])
    for name,digest in approved['sources'].items():
        if sha256_file(ROOT/name)!=digest:raise ValueError('recovery source changed')
    attempt_path=Path(prior['recovery_attempt_path'])
    if attempt_path.is_symlink() or attempt_path.resolve()!=Path(prior['recovery_root']).resolve()/'controller/attempt.json':
        raise ValueError('recovery attempt leaves authorized scope')
    attempt=read_bound(attempt_path,prior['recovery_attempt_sha256'])
    expected_auth={'path':str(Path(prior['recovery_authorization_path']).resolve()),'sha256':prior['recovery_authorization_sha256']}
    if any(attempt.get(k)!=v for k,v in dict(status='attempted',pid=prior['recovery_controller_pid'],scope_root=approved['scope_root'],protocol_hash=TRANSFER_PROTOCOL,authorization=expected_auth,authorization_sha256=prior['recovery_authorization_sha256'],closure=approved['closure'],deadline_utc=approved['deadline_utc']).items()):
        raise ValueError('recovery launched attempt identity changed')
    if durable:
        check_durable_failure(prior,approved,preparation,attempt)
    return closure,approved

def check_durable_failure(prior,approved,preparation,attempt):
    """Verify the second closure as failed evidence, never as a successful exit."""
    if approved['deadline_utc']!=TRANSFER_DEADLINE:raise ValueError('original transfer deadline changed')
    accounting=dict(n_reused_fits=6030,n_new_fits=4800,n_retained_fits=10830,
                    discarded_partial_completed_fits=1785,known_completed_including_discarded=12615,
                    unknown_remote_fits=12,unknown_local_fits=True,additional_cloud_spend_usd=0,
                    previous_local_exit_status='failed')
    if any(preparation.get(k)!=v for k,v in accounting.items()):raise ValueError('durable preparation accounting changed')
    for key in ('local_failure_closure','local_failure_exit','local_failure_state'):
        if prior.get(key)!=approved.get(key) or preparation.get(key)!=approved.get(key):
            raise ValueError('local failed predecessor binding changed')
    for key in ('local_failure_closure','local_failure_exit'):
        if attempt.get(key)!=approved[key]:raise ValueError('durable attempt failure provenance changed')
    failure,receipt,state=[read_bound(approved[k]['path'],approved[k]['sha256']) for k in
                           ('local_failure_closure','local_failure_exit','local_failure_state')]
    expected=dict(schema='final_local_recovery_interruption.1',status='failed_stopped_no_new_complete_cells',
                  cleanup_error=None,additional_discarded_recorded_completed_fits=1086,
                  total_discarded_recorded_completed_fits=1785,known_completed_after_eventual_success_including_discarded=12615,
                  remaining_scientific_fits=4800,retained_complete_cells=18,retained_complete_fits=6030,
                  uncertain_remote_attempts_from_original_scope=12,additional_cloud_spend_usd=0,
                  controller_exit_sha256=approved['local_failure_exit']['sha256'],
                  controller_state_sha256=approved['local_failure_state']['sha256'])
    if any(failure.get(k)!=v for k,v in expected.items()):raise ValueError('local failed closure changed')
    if (ROOT/failure['controller_exit_path']).resolve()!=Path(approved['local_failure_exit']['path']).resolve():
        raise ValueError('local failed exit path changed')
    expected=dict(schema='final_transfer_recovery_exit.1',status='failed',exit_code=1,cleanup_error=None,
                  pid=failure['controller_pid'],deadline_utc=TRANSFER_DEADLINE,protocol_hash=TRANSFER_PROTOCOL,
                  closure=approved['closure'])
    if any(receipt.get(k)!=v for k,v in expected.items()):raise ValueError('previous local exit must remain failed with clean shutdown')
    if (state.get('status')!='failed' or state.get('cleanup_error') is not None or state.get('finished_cells')!=[]
            or state.get('pid')!=failure['controller_pid'] or Path(approved['local_failure_exit']['path']).with_name('completed.json').exists()):
        raise ValueError('previous local state must remain failed without completion')
    if preparation.get('old_local_pids')!=approved.get('old_local_pids') or not {failure['controller_pid'],failure['hybrid_queue_pid']}<=set(approved['old_local_pids']):
        raise ValueError('previous local ownership coverage changed')
    if prior['recovery_controller_pid'] in approved['old_local_pids']:raise ValueError('durable owner reuses failed predecessor PID')
    for pid in approved['old_local_pids']:assert_process_absent(pid)
    # These immutable evidence maps were semantically checked by the source-bound controller.
    evidence=preparation['local_failure_evidence']
    if any(evidence.get(approved[key]['path'])!=approved[key]['sha256'] for key in ('local_failure_exit','local_failure_state')):
        raise ValueError('local failure evidence coverage changed')
    for path,digest in evidence.items():
        if sha256_file(path)!=digest:raise ValueError('local failure evidence changed')
    for key in ('amendment','local_failure_amendment'):
        read_bound(preparation[key]['path'],preparation[key]['sha256'])
    copies=preparation['copied_file_hashes']
    if set(copies)!=set(TRANSFER_CELLS[:18]):raise ValueError('durable copied cell coverage changed')
    for cell,files in copies.items():
        if files.get('completed.json')!=approved['reused_cell_hashes'][cell]:raise ValueError('durable preserved completion changed')
        for name,digest in files.items():
            if sha256_file(Path(approved['scope_root'])/cell/name)!=digest:raise ValueError('durable preserved artifact changed')


def check_recovery_predecessor(auth,gate,*,durable=False):
    prior=auth['predecessor'];closed=gate['predecessor'];closure,approved=check_recovery_closure(prior,durable=durable)
    root=Path(prior['recovery_root']).resolve()
    for key,name in [('recovery_completed_path','completed.json'),('recovery_exit_path','exit_receipt.json')]:
        path=Path(closed[key])
        if path.is_symlink() or path.resolve()!=root/'controller'/name:raise ValueError('recovery receipt leaves authorized scope')
    completed=read_bound(closed['recovery_completed_path'],closed['recovery_completed_sha256'])
    receipt=read_bound(closed['recovery_exit_path'],closed['recovery_exit_sha256'])
    expected=dict(schema='final_transfer_durable_completion.1' if durable else 'final_transfer_recovery_completion.1',status='complete',pid=prior['recovery_controller_pid'],protocol_hash=TRANSFER_PROTOCOL,scope_root=str(root),n_retained_fits=10830,n_new_fits=4800,known_completed_including_discarded=12615 if durable else 11529,unknown_remote_fits=12,deadline_utc=approved['deadline_utc'])
    if any(completed.get(k)!=v for k,v in expected.items()):raise ValueError('complete exact thirty-cell recovery required')
    expected_auth={'path':str(Path(prior['recovery_authorization_path']).resolve()),'sha256':prior['recovery_authorization_sha256']}
    if completed.get('authorization')!=expected_auth or completed.get('closure')!=approved['closure'] or completed.get('preserved_audit')!=approved['preserved_audit']:
        raise ValueError('recovery certificate provenance changed')
    if durable:
        expected_extra=dict(previous_local_exit_status='failed',discarded_partial_completed_fits=1785,
                            unknown_local_fits=True,authorization_sha256=prior['recovery_authorization_sha256'],
                            closure_sha256=prior['closure_sha256'])
        expected_extra.update({key:approved[key] for key in ('local_failure_closure','local_failure_exit','local_failure_state')})
        if any(completed.get(k)!=v for k,v in expected_extra.items()):raise ValueError('durable completion provenance/accounting changed')
        attempt=read_bound(prior['recovery_attempt_path'],prior['recovery_attempt_sha256'])
        if not timestamp(attempt['started_at_utc'])<=timestamp(completed['finished_at_utc'])<=timestamp(TRANSFER_DEADLINE):
            raise ValueError('durable completion exceeded original deadline or predates attempt')
    audit=read_bound(approved['preserved_audit']['path'],approved['preserved_audit']['sha256'])
    if audit.get('status')!='passed' or audit.get('n_preserved_cells')!=18 or audit.get('n_preserved_fits')!=6030 or audit.get('protocol_hash')!=TRANSFER_PROTOCOL:
        raise ValueError('preserved semantic audit incomplete')
    all_hashes=completed['completed_cell_hashes'];old=completed['reused_cell_hashes'];new=completed['new_cell_hashes']
    if set(all_hashes)!=set(TRANSFER_CELLS) or set(old)!=set(TRANSFER_CELLS[:18]) or set(new)!=set(TRANSFER_CELLS[18:]) or {**old,**new}!=all_hashes:
        raise ValueError('recovery cell coverage changed')
    if old!=approved['reused_cell_hashes'] or old!={a['cell']:a['completed_sha256'] for a in audit['completed_cells']}:
        raise ValueError('preserved eighteen cells changed')
    for cell,digest in all_hashes.items():
        directory=root/cell;done=read_bound(directory/'completed.json',digest)
        regime,name=cell.split('/')
        expected=dict(status='complete',protocol_hash=TRANSFER_PROTOCOL,regime=regime,cell=name,n_fits=166 if regime=='hla' else 400)
        if any(done.get(k)!=v for k,v in expected.items()) or set(done['sha256'])!=TRANSFER_FILES:raise ValueError('recovered cell identity/count changed')
        for filename,h in done['sha256'].items():
            if sha256_file(directory/filename)!=h:raise ValueError('recovered cell artifact changed after audit')
    if durable:
        # The runtime hashes bind the inventory; recheck the actual checkpoint payloads too.
        from scripts.final_transfer_checkpoint_worker import verify_checkpoint_cell
        for key in TRANSFER_CELLS[18:]:
            regime,cell=key.split('/')
            verify_checkpoint_cell(root,regime,cell,approved)
    expected=dict(schema='final_transfer_durable_exit.1' if durable else 'final_transfer_recovery_exit.1',status='complete',exit_code=0,pid=completed['pid'],protocol_hash=TRANSFER_PROTOCOL,authorization_sha256=prior['recovery_authorization_sha256'],completed_sha256=closed['recovery_completed_sha256'])
    if any(receipt.get(k)!=v for k,v in expected.items()):raise ValueError('successful exact recovery controller exit required')
    if durable:
        expected_exit=dict(scope_root=str(root),authorization=expected_auth,closure=approved['closure'],
                           local_failure_closure=approved['local_failure_closure'],local_failure_exit=approved['local_failure_exit'],
                           deadline_utc=TRANSFER_DEADLINE,
                           completed_certificate={'path':str(root/'controller/completed.json'),'sha256':closed['recovery_completed_sha256']})
        if any(receipt.get(k)!=v for k,v in expected_exit.items()):raise ValueError('durable exit provenance changed')
        if not timestamp(completed['finished_at_utc'])<=timestamp(receipt['finished_at_utc'])<=min(timestamp(TRANSFER_DEADLINE),timestamp(gate['created_at_utc'])):
            raise ValueError('durable successful exit exceeded original deadline')
    assert_process_absent(completed['pid'])

def check_predecessor(auth,gate):
    if auth['predecessor'].get('kind')==DURABLE_KIND:
        return check_recovery_predecessor(auth,gate,durable=True)
    if auth['predecessor'].get('kind')==RECOVERY_KIND:
        return check_recovery_predecessor(auth,gate)
    prior=auth['predecessor'];closed=gate['predecessor'];root_guard=read_bound(prior['guard_authorization_path'],prior['guard_authorization_sha256'])
    if any(root_guard.get(k)!=v for k,v in dict(app_id=prior['app_id'],controller_pid=prior['controller_pid']).items()):raise ValueError('predecessor app/controller ownership changed')
    transfer=read_bound(closed['transfer_exit_path'],closed['transfer_exit_sha256']);driver=read_bound(closed['driver_exit_path'],closed['driver_exit_sha256']);guard=read_bound(closed['guard_status_path'],closed['guard_status_sha256'])
    if (transfer.get('status'),transfer.get('pid'),transfer.get('protocol_hash'))!=('complete',prior['controller_pid'],prior['protocol_hash']):raise ValueError('transfer exit is not exact successful predecessor')
    if driver.get('status')!='complete' or driver.get('app_context_closed') is not True or driver.get('app_id')!=prior['app_id'] or driver.get('pid')!=prior['controller_pid']:raise ValueError('driver has not closed exact successful predecessor app')
    if guard.get('phase')!='finished' or guard.get('app_stopped') is not True or guard.get('controller_state')!='done' or guard.get('guard_pid')!=prior['guard_pid'] or guard.get('authorization_sha256')!=prior['guard_authorization_sha256']:raise ValueError('exact predecessor guard has not finished with stopped app')
    reconciliation=read_bound(closed['reconciliation_path'],closed['reconciliation_sha256'])
    if reconciliation.get('schema')!='final_scope_reconciliation.1' or reconciliation.get('authorization_sha256')!=prior['guard_authorization_sha256'] or reconciliation.get('guard_status_sha256')!=closed['guard_status_sha256']:raise ValueError('prior reconciliation does not bind final guard')
    minimum=root_guard['prior_charges_usd']+(guard['effective_commitment_usd'] if guard.get('budget_verified') else root_guard['total_allocation_usd']-root_guard['prior_charges_usd'])
    if reconciliation.get('cumulative_prior_charges_usd',-1)<minimum-1e-10:raise ValueError('reconciliation underaccounts previous commitment')
    assert_process_absent(prior['controller_pid']);assert_process_absent(prior['guard_pid'])
    for pid in prior.get('child_fitting_pids',[]):assert_process_absent(pid)


def check_launch_gate(path,auth,phase):
    gate=json.loads(Path(path).read_text())
    expected_schema={SCHEMA:GATE_SCHEMA,RECOVERY_SCHEMA:RECOVERY_GATE_SCHEMA,DURABLE_SCHEMA:DURABLE_GATE_SCHEMA}.get(auth.get('schema'))
    if gate.get('schema')!=expected_schema or gate.get('approved') is not True or gate.get('authorization_sha256')!=auth['_sha256'] or gate.get('phase')!=phase:raise ValueError('launch gate does not bind approved exact phase authorization')
    created=timestamp(gate['created_at_utc']);expiry=timestamp(gate['expires_at_utc'])
    if created<timestamp(auth['created_at_utc']) or created>time.time() or expiry<=time.time() or expiry<=created:raise ValueError('launch gate is stale, expired or future-dated')
    if auth.get('schema')==DURABLE_SCHEMA and expiry>min(timestamp(auth['queue_deadline_utc']),timestamp(QUEUE_DEADLINE)):raise ValueError('durable gate exceeds fixed queue deadline')
    check_predecessor(auth,gate)
    if phase=='main':
        pilot=gate['pilot'];completed=read_bound(pilot['completed_path'],pilot['completed_sha256']);receipt=read_bound(pilot['exit_path'],pilot['exit_sha256'])
        if completed.get('status')!='complete' or completed.get('phase')!='pilot' or completed.get('n_fits')!=12 or completed.get('protocol_hash')!=auth['protocol_hash']:raise ValueError('exact12-fit unscored pilot missing')
        if receipt.get('status')!='complete' or receipt.get('phase')!='pilot' or receipt.get('n_fits')!=12 or receipt.get('protocol_hash')!=auth['protocol_hash']:raise ValueError('successful pilot exit missing')
        assert_process_absent(receipt['pid'])
        forecast=Path(pilot['runtime_forecast_path'])
        if sha256_file(forecast)!=pilot['runtime_forecast_sha256']:raise ValueError('root runtime gate does not bind pilot forecast')
        if gate.get('runtime_approved') is not True:raise ValueError('separate root scientific runtime approval required')
    return gate


def check_authorization(path,protocol,phase,*,require_gate=True):
    path=Path(path);auth=json.loads(path.read_text());auth['_sha256']=sha256_file(path)
    durable=auth.get('predecessor',{}).get('kind')==DURABLE_KIND
    recovery=durable or auth.get('predecessor',{}).get('kind')==RECOVERY_KIND
    expected=dict(schema=DURABLE_SCHEMA if durable else RECOVERY_SCHEMA if recovery else SCHEMA,approved=True,phase=phase,protocol_hash=protocol['protocol_hash'],controller_sha256=sha256_file(Path(__file__)),worker_sha256=sha256_file(ROOT/'scripts/final_hybrids.py'),jobs=[f'fold_{fold}/{rep}' for fold,rep in phase_jobs(phase)],max_workers=1 if phase=='pilot' else 2,threads_each=2,cloud_spend_usd=0,n_fits=12 if phase=='pilot' else 480)
    if auth.get('approved') is not True or any(type(auth.get(k)) is not int for k in ('max_workers','threads_each','n_fits')) or any(auth.get(k)!=v for k,v in expected.items()):raise ValueError('authorization axes/source/count/resources mismatch')
    if auth.get('source_sha256')!=protocol['source_sha256']:raise ValueError('authorization must bind every frozen source')
    start=timestamp(auth['created_at_utc']);deadline=timestamp(auth['queue_deadline_utc'])
    if not 0<deadline-start<=48*3600 or time.time()>=deadline:raise ValueError('queue deadline expired or exceeds48hours')
    wall=auth.get('fit_wall_seconds')
    if type(wall) not in (int,float) or not 0<wall<=48*3600:raise ValueError('fixed finite fitting deadline required')
    if recovery and phase=='pilot' and wall!=7200:raise ValueError('recovery pilot retains fixed two-hour cap')
    if durable and auth.get('scope_root')!=str(DURABLE_OUT.resolve()):raise ValueError('durable hybrid output scope changed')
    if durable and deadline>timestamp(QUEUE_DEADLINE):raise ValueError('durable queue exceeds original expiry')
    if recovery:check_recovery_closure(auth['predecessor'],durable=durable)
    if require_gate:check_launch_gate(auth['launch_gate_path'],auth,phase)
    return auth


def stop_owned_children(active,state):
    errors=[];uncertain=[]
    for child,record in active.items():
        if child.poll() is None:
            try:
                if record.get('pid')!=child.pid or record.get('process_group')!=child.pid or os.getpgid(child.pid)!=child.pid:raise RuntimeError('owned child process-group identity differs; refusing signals')
                os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            except BaseException as error:errors.append(repr(error))
    until=time.monotonic()+30
    for child,record in list(active.items()):
        try:child.wait(timeout=max(.01,until-time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                if record.get('pid')!=child.pid or record.get('process_group')!=child.pid or os.getpgid(child.pid)!=child.pid:raise RuntimeError('owned child process-group identity differs; refusing kill')
                os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            except BaseException as error:errors.append(repr(error))
            try:child.wait(timeout=5)
            except BaseException as error:errors.append(repr(error))
        except BaseException as error:errors.append(repr(error))
        record.update(exit_code=child.poll(),disposition='terminated_after_phase_failure_or_deadline')
        if record['exit_code'] is None:uncertain.append(dict(record,cleanup_status='process_still_uncertain'))
        else:state['finished_children'].append(record)
    state['active_children']=uncertain
    if errors:state['cleanup_errors']=errors


def child_command(out,approved,phase,fold,rep):
    return [str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/final_hybrids.py'),'pilot-component' if phase=='pilot' else 'run-component','--out-dir',str(out),'--approved',str(approved),'--fold',str(fold),'--representation',rep]


def run_phase(out,approved,phase,*,popen=subprocess.Popen,monotonic=time.monotonic,sleep=time.sleep):
    out=Path(out);_,protocol=validate_frozen(out);auth=check_authorization(approved,protocol,phase);directory=out/'controller'/phase
    if auth.get('schema')==DURABLE_SCHEMA and out.resolve()!=Path(auth['scope_root']):raise ValueError('durable hybrid out-dir differs from authorization')
    directory.mkdir(parents=True,exist_ok=True)
    owner=(directory/'ownership.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='running',phase=phase,pid=os.getpid(),protocol_hash=protocol['protocol_hash'],authorization_sha256=auth['_sha256'],started_at_utc=now(),n_fits=0,fit_wall_seconds=auth['fit_wall_seconds'],jobs=auth['jobs'],max_workers=auth['max_workers'],launch_intents=[],active_children=[],finished_children=[],peak_concurrent_children=0)
    try:write_once(directory/'attempt.json',state)
    except BaseException:fcntl.flock(owner,fcntl.LOCK_UN);owner.close();raise
    start=monotonic();active={};pending=list(phase_jobs(phase));failure=None;save=lambda:atomic_json(directory/'state.json',state)
    env={**os.environ,**{key:'2' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}}
    previous_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    spawning=False;stop_requested=None
    def interrupted(sig,frame):
        nonlocal stop_requested
        stop_requested=sig
        if not spawning:raise RuntimeError(f'owned hybrid controller interrupted by signal {sig}')
    try:
        for sig in previous_handlers:signal.signal(sig,interrupted)
        while pending or active:
            if monotonic()-start>=auth['fit_wall_seconds']:raise TimeoutError('fixed hybrid fitting deadline reached; no extension')
            if sha256_file(approved)!=auth['_sha256']:raise ValueError('phase authorization changed')
            for child,record in list(active.items()):
                code=child.poll()
                if code is None:continue
                del active[child];fold=int(record['job'].split('/')[0].split('_')[1]);rep=record['job'].split('/')[1];directory_job=out/('pilot' if phase=='pilot' else '')/record['job']
                record.update(exit_code=code,finished_at_utc=now());state['finished_children'].append(record);state['active_children']=list(active.values());save()
                if code:raise RuntimeError(f"hybrid {record['job']} failed; queued dispatch frozen; no retry")
                if phase=='main':done=verify_component(directory_job,protocol,fold,rep)
                else:
                    done=json.loads((directory_job/'completed.json').read_text())
                    if done.get('status')!='complete' or done.get('phase')!='pilot' or done.get('n_fits')!=6 or done.get('held_scores_computed') is not False or done.get('reuse_into_scientific') is not False or done.get('protocol_hash')!=protocol['protocol_hash']:raise ValueError('pilot engineering completion differs')
                    if sha256_file(directory_job/'bundle_manifest.json')!=done['bundle_manifest_sha256'] or sha256_file(directory_job/'runtime.json')!=done['runtime_sha256']:raise ValueError('pilot receipt artifacts changed')
                receipt=json.loads((directory_job/'exit_receipt.json').read_text())
                if receipt.get('status')!='complete' or receipt.get('completed_sha256')!=sha256_file(directory_job/'completed.json') or receipt.get('pid')!=child.pid:raise ValueError('owned child exit receipt mismatch')
                state['n_fits']+=done['n_fits'];record.update(disposition='complete',completed_sha256=sha256_file(directory_job/'completed.json'));save()
            if pending and len(active)<auth['max_workers']:
                # Poll before replacing any finished worker; never ignore a known failure.
                if any(child.poll() is not None for child in active):continue
                fold,rep=pending.pop(0);job=f'fold_{fold}/{rep}';command=child_command(out,approved,phase,fold,rep);record=dict(job=job,command=command,launch_intent_at_utc=now());state['launch_intents'].append(record);save()
                log=directory/(job.replace('/','_')+'.log')
                # Defer signals until ownership is registered, without blocking child signals.
                spawning=True
                try:
                    with log.open('x') as handle:child=popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                    record.update(pid=child.pid,process_group=child.pid,log=str(log),started_at_utc=now());active[child]=record;state['active_children']=list(active.values());state['peak_concurrent_children']=max(state['peak_concurrent_children'],len(active));save()
                finally:spawning=False
                if stop_requested is not None:raise RuntimeError(f'owned hybrid controller interrupted during spawn by signal {stop_requested}')
                continue
            if active:sleep(1)
        if state['n_fits']!=auth['n_fits']:raise ValueError('phase count does not match exact authorized graph')
        state.update(status='complete',finished_at_utc=now(),elapsed_fit_seconds=monotonic()-start);save();write_once(directory/'completed.json',state);write_once(directory/'exit_receipt.json',state)
    except BaseException as error:
        pending=[];stop_owned_children(active,state);state.update(status='failed',error=repr(error),finished_at_utc=now(),elapsed_fit_seconds=monotonic()-start);save();write_once(directory/'exit_receipt.json',state);raise
    finally:
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)
        fcntl.flock(owner,fcntl.LOCK_UN);owner.close()
    return state


def queue_phase(out,approved,phase,*,sleep=time.sleep):
    out=Path(out);_,protocol=validate_frozen(out);auth=check_authorization(approved,protocol,phase,require_gate=False);directory=out/'controller'/phase
    if auth.get('schema')==DURABLE_SCHEMA and out.resolve()!=Path(auth['scope_root']):raise ValueError('durable hybrid out-dir differs from authorization')
    directory.mkdir(parents=True,exist_ok=True)
    write_once(directory/'queue_attempt.json',dict(pid=os.getpid(),phase=phase,authorization_sha256=auth['_sha256'],started_at_utc=now(),queue_deadline_utc=auth['queue_deadline_utc'],poll_seconds=30))
    try:
        while True:
            if time.time()>=timestamp(auth['queue_deadline_utc']):raise TimeoutError('fixed48h-or-shorter queue deadline expired')
            if sha256_file(approved)!=auth['_sha256']:raise ValueError('queue authorization changed')
            gate=Path(auth['launch_gate_path'])
            if gate.exists():
                check_launch_gate(gate,auth,phase);break
            sleep(min(30,max(0,timestamp(auth['queue_deadline_utc'])-time.time())))
        write_once(directory/'queue_gate_receipt.json',dict(status='ready',phase=phase,gate_sha256=sha256_file(gate),at_utc=now()))
        state=run_phase(out,approved,phase)
        write_once(directory/'queue_exit_receipt.json',dict(status='complete',phase=phase,pid=os.getpid(),n_fits=state['n_fits'],finished_at_utc=now(),phase_exit_sha256=sha256_file(directory/'exit_receipt.json')))
        return state
    except BaseException as error:
        write_once(directory/'queue_exit_receipt.json',dict(status='failed',phase=phase,pid=os.getpid(),error=repr(error),finished_at_utc=now(),no_retry=True));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('queue','run'));parser.add_argument('--out-dir',type=Path,default=OUT);parser.add_argument('--approved',type=Path,required=True);parser.add_argument('--phase',choices=('pilot','main'),required=True)
    args=parser.parse_args();(queue_phase if args.mode=='queue' else run_phase)(args.out_dir,args.approved,args.phase)
if __name__=='__main__':main()
