"""Durable LOCAL queue and fixed-deadline subprocess controller; no cloud/retry."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl
import json
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

def check_predecessor(auth,gate):
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
    if gate.get('schema')!=GATE_SCHEMA or gate.get('approved') is not True or gate.get('authorization_sha256')!=auth['_sha256'] or gate.get('phase')!=phase:raise ValueError('launch gate does not bind approved exact phase authorization')
    created=timestamp(gate['created_at_utc']);expiry=timestamp(gate['expires_at_utc'])
    if created<timestamp(auth['created_at_utc']) or created>time.time() or expiry<=time.time() or expiry<=created:raise ValueError('launch gate is stale, expired or future-dated')
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
    expected=dict(schema=SCHEMA,approved=True,phase=phase,protocol_hash=protocol['protocol_hash'],controller_sha256=sha256_file(Path(__file__)),worker_sha256=sha256_file(ROOT/'scripts/final_hybrids.py'),jobs=[f'fold_{fold}/{rep}' for fold,rep in phase_jobs(phase)],max_workers=1 if phase=='pilot' else 2,threads_each=2,cloud_spend_usd=0,n_fits=12 if phase=='pilot' else 480)
    if auth.get('approved') is not True or any(type(auth.get(k)) is not int for k in ('max_workers','threads_each','n_fits')) or any(auth.get(k)!=v for k,v in expected.items()):raise ValueError('authorization axes/source/count/resources mismatch')
    if auth.get('source_sha256')!=protocol['source_sha256']:raise ValueError('authorization must bind every frozen source')
    start=timestamp(auth['created_at_utc']);deadline=timestamp(auth['queue_deadline_utc'])
    if not 0<deadline-start<=48*3600 or time.time()>=deadline:raise ValueError('queue deadline expired or exceeds48hours')
    wall=auth.get('fit_wall_seconds')
    if type(wall) not in (int,float) or not 0<wall<=48*3600:raise ValueError('fixed finite fitting deadline required')
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
    out=Path(out);_,protocol=validate_frozen(out);auth=check_authorization(approved,protocol,phase);directory=out/'controller'/phase;directory.mkdir(parents=True,exist_ok=True)
    owner=(directory/'ownership.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='running',phase=phase,pid=os.getpid(),protocol_hash=protocol['protocol_hash'],authorization_sha256=auth['_sha256'],started_at_utc=now(),n_fits=0,fit_wall_seconds=auth['fit_wall_seconds'],jobs=auth['jobs'],max_workers=auth['max_workers'],launch_intents=[],active_children=[],finished_children=[],peak_concurrent_children=0)
    try:write_once(directory/'attempt.json',state)
    except BaseException:fcntl.flock(owner,fcntl.LOCK_UN);owner.close();raise
    start=monotonic();active={};pending=list(phase_jobs(phase));failure=None;save=lambda:atomic_json(directory/'state.json',state)
    env={**os.environ,**{key:'2' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}}
    previous_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    def interrupted(sig,frame):raise RuntimeError(f'owned hybrid controller interrupted by signal {sig}')
    try:
        for sig in previous_handlers:signal.signal(sig,interrupted)
        while pending or active:
            if monotonic()-start>=auth['fit_wall_seconds']:raise TimeoutError('fixed hybrid fitting deadline reached; no extension')
            if sha256_file(approved)!=auth['_sha256']:raise ValueError('phase authorization changed')
            while pending and len(active)<auth['max_workers']:
                fold,rep=pending.pop(0);job=f'fold_{fold}/{rep}';command=child_command(out,approved,phase,fold,rep);record=dict(job=job,command=command,launch_intent_at_utc=now());state['launch_intents'].append(record);save()
                log=directory/(job.replace('/','_')+'.log')
                with log.open('x') as handle:child=popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                record.update(pid=child.pid,process_group=child.pid,log=str(log),started_at_utc=now());active[child]=record;state['active_children']=list(active.values());state['peak_concurrent_children']=max(state['peak_concurrent_children'],len(active));save()
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
    out=Path(out);_,protocol=validate_frozen(out);auth=check_authorization(approved,protocol,phase,require_gate=False);directory=out/'controller'/phase;directory.mkdir(parents=True,exist_ok=True)
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
