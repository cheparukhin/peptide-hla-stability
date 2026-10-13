"""One-shot CPU Sandbox supervisor, released only through its exact stdin handle."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import select
import signal
import subprocess
import sys
import time
import threading

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pepstab import final_cloud_protocol as contract

DEPENDENCIES={'numpy':'2.5.3','pandas':'3.0.6','scipy':'1.18.1','scikit-learn':'1.9.1','catboost':'1.2.10','threadpoolctl':'3.7.0'}
THREAD_VARIABLES=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')


def verify_environment():
    packages={name:version(name) for name in DEPENDENCIES}
    if platform.python_version()!='3.12.12' or packages!=DEPENDENCIES:
        raise ValueError('exact frozen Python/dependencies required; no fits authorized')
    if sys.platform!='linux' or ROOT.resolve()!=Path(contract.REPOSITORY_ROOT).resolve():
        raise ValueError('exact Linux repository namespace required')
    return dict(python=platform.python_version(),python_build=sys.version,packages=packages,
                platform=platform.platform(),machine=platform.machine(),
                logical_repository_root=contract.REPOSITORY_ROOT,resolved_repository_root=str(ROOT.resolve()))


def phase_jobs(phase):
    if phase=='transfer_recovery':return [('joint',f'h4_p{i}') for i in range(5)]
    if phase=='hybrid_pilot':return [(0,'R0'),(0,'R5')]
    if phase=='hybrid_main':return [(fold,rep) for fold in range(5) for rep in ('R0','R5')]
    return [(None,None)]


def child_command(auth,approved,job):
    phase=auth['phase'];out=ROOT/contract.SCOPE
    if phase=='transfer_recovery':
        regime,cell=job
        return [sys.executable,str(ROOT/'scripts/final_transfer.py'),'run-cell','--out-dir',str(ROOT/contract.CLOUD_TRANSFER_ROOT),
                '--approved',str(approved),'--regime',regime,'--cell',cell]
    if phase=='transfer_report':
        report_root=contract.CLOUD_TRANSFER_ROOT if auth.get('scope_root')==contract.CLOUD_TRANSFER_ROOT else contract.TRANSFER_ROOT
        return [sys.executable,str(ROOT/'scripts/final_transfer_report.py'),'--out-dir',str(ROOT/report_root)]
    if phase=='hybrid_report':
        return [sys.executable,str(ROOT/'scripts/final_hybrids_cloud_report.py'),'--out-dir',str(out),'--approved',str(approved)]
    if phase=='label_efficiency':
        return [sys.executable,str(ROOT/'scripts/final_label_efficiency.py'),'run','--approved',str(approved)]
    fold,rep=job
    return [sys.executable,str(ROOT/'scripts/final_hybrids_cloud.py'),'pilot-component' if phase=='hybrid_pilot' else 'run-component',
            '--out-dir',str(out),'--approved',str(approved),'--fold',str(fold),'--representation',rep]


def scientific_entry_command(command,approved):
    return [command[0],str(ROOT/'scripts/final_cloud_scientific_entry.py'),'--approved',str(approved),
            '--script',command[1],'--',*command[2:]]


class RawJsonLines:
    """Bounded nonblocking bytes; no TextIO read-ahead can hide lease lines."""
    MAX_LINE=65536
    MAX_BUFFER=262144
    def __init__(self,stream):
        self.fd=stream.fileno();self.buffer=b'';self.eof=False
        self.was_blocking=os.get_blocking(self.fd)
        os.set_blocking(self.fd,False)

    def restore(self):
        os.set_blocking(self.fd,self.was_blocking)

    def _read(self):
        try:block=os.read(self.fd,self.MAX_LINE)
        except BlockingIOError:return False
        if not block:self.eof=True;return False
        self.buffer+=block
        if len(self.buffer)>self.MAX_BUFFER:raise ValueError('guard input buffer limit exceeded')
        if b'\n' not in self.buffer and len(self.buffer)>self.MAX_LINE:
            raise ValueError('guard JSON line too long')
        return True

    def _pop(self):
        end=self.buffer.find(b'\n')
        if end<0:return None
        if end+1>self.MAX_LINE:raise ValueError('guard JSON line too long')
        line,self.buffer=self.buffer[:end],self.buffer[end+1:]
        def unique(items):
            result={}
            for key,value in items:
                if key in result:raise ValueError('duplicate guard JSON key')
                result[key]=value
            return result
        def invalid(value):raise ValueError('nonfinite guard JSON value')
        value=json.loads(line.decode('utf-8'),object_pairs_hook=unique,parse_constant=invalid)
        if not isinstance(value,dict):raise ValueError('guard message must be JSON object')
        return value

    def read_line(self,seconds):
        end=time.monotonic()+seconds
        while True:
            value=self._pop()
            if value is not None:return value
            if self.eof:raise EOFError('guard stdin closed before release')
            remaining=end-time.monotonic()
            if remaining<=0 or not select.select([self.fd],[],[],remaining)[0]:
                raise TimeoutError('watchdog release absent before startup deadline')
            self._read()

    def drain(self):
        messages=[];reads=0
        while True:
            value=self._pop()
            if value is not None:
                messages.append(value)
                if len(messages)>64:raise ValueError('too many guard messages in one drain')
                continue
            if self.eof:raise EOFError('guard stdin closed after release')
            if reads>=4:raise ValueError('guard input drain byte limit exceeded')
            if not self._read():
                if self.eof:raise EOFError('guard stdin closed after release')
                return messages
            reads+=1


class GuardLease:
    """A stopped watcher cannot leave a healthy-looking Sandbox working forever."""
    MAX_SECONDS=90
    def __init__(self,auth,release,reader,*,clock=None,monotonic=None):
        self.auth=auth;self.reader=reader;self.clock=clock or time.time
        self.monotonic=monotonic or time.monotonic
        self.sandbox_id=release['sandbox_id'];self.watchdog_pid=release['watchdog_readiness']['pid']
        current=self.clock();self.sequence=0
        self.last_issued=contract.timestamp(release['watchdog_readiness']['at_utc'])
        self.work_deadline=contract.timestamp(auth['deadline_utc'])
        self.expires=min(current+self.MAX_SECONDS,self.work_deadline)
        self.monotonic_expiry=self.monotonic()+max(0,self.expires-current)
        self._lock=threading.RLock();self._error=None;self._cleanup_error=None;self._stop=threading.Event();self._monitor=None

    def accept(self,message):
        current=self.clock();mono=self.monotonic()
        if current>=self.expires or mono>=self.monotonic_expiry:
            raise TimeoutError('watchdog guard lease expired; renewal cannot revive it')
        expected=dict(schema='final_cloud_guard_lease.1',phase_nonce=self.auth['phase_nonce'],
            authorization_sha256=self.auth['_sha256'],sandbox_id=self.sandbox_id,watchdog_pid=self.watchdog_pid)
        if type(message.get('watchdog_pid')) is not int or any(message.get(k)!=v for k,v in expected.items()):
            raise ValueError('guard lease identity mismatch')
        sequence=message.get('sequence')
        if type(sequence) is not int or sequence<=self.sequence:
            raise ValueError('guard lease sequence must increase')
        issued=contract.timestamp(message['issued_at_utc']);expires=contract.timestamp(message['expires_at_utc'])
        if issued>current or issued<self.last_issued or not current<expires<=min(issued+self.MAX_SECONDS,self.work_deadline):
            raise ValueError('guard lease timestamp, duration or deadline invalid')
        self.sequence=sequence;self.last_issued=issued;self.expires=expires
        self.monotonic_expiry=mono+min(self.MAX_SECONDS,expires-current)

    def _check_unlocked(self):
        if self.clock()>=self.expires or self.monotonic()>=self.monotonic_expiry:
            raise TimeoutError('watchdog guard lease expired')
        for message in self.reader.drain():self.accept(message)
        if self.clock()>=self.expires or self.monotonic()>=self.monotonic_expiry:
            raise TimeoutError('watchdog guard lease expired')

    def check(self):
        with self._lock:
            if self._error is not None:raise self._error
            self._check_unlocked()

    def start_monitor(self,*,on_failure=None):
        """Service leases during input hashing too; numerical work stays in children."""
        if self._monitor is not None:raise RuntimeError('guard monitor already started')
        def monitor():
            while not self._stop.is_set():
                try:self.check()
                except BaseException as error:
                    with self._lock:self._error=error
                    # Never hold the lease lock while taking the child-ownership lock.
                    # This callback must stop fitting even if the main thread is in Volume sync.
                    if on_failure is not None:
                        try:on_failure()
                        except BaseException as cleanup:
                            with self._lock:self._cleanup_error=repr(cleanup)
                    return
                self._stop.wait(.2)
        self._monitor=threading.Thread(target=monitor,name='cloud-guard-lease',daemon=True)
        self._monitor.start()

    def stop_monitor(self):
        self._stop.set()
        if self._monitor is not None:
            self._monitor.join(timeout=1)
            if self._monitor.is_alive():raise RuntimeError('guard reader did not stop')

    def evidence(self):
        with self._lock:
            return dict(watchdog_pid=self.watchdog_pid,last_sequence=self.sequence,
                expires_at_utc=datetime.fromtimestamp(self.expires,timezone.utc).isoformat(),maximum_seconds=self.MAX_SECONDS,
                monitor_cleanup_error=self._cleanup_error)


def wait_release(auth,stream,seconds):
    if type(seconds) is not int or not 1<=seconds<=300:
        raise ValueError('startup wait must be bounded to 1..300 seconds')
    remaining=min(seconds,contract.timestamp(auth['deadline_utc'])-time.time())
    if remaining<=0:raise TimeoutError('watchdog release absent before startup deadline')
    owned=not isinstance(stream,RawJsonLines)
    reader=RawJsonLines(stream) if owned else stream
    try:release=reader.read_line(remaining)
    finally:
        if owned:reader.restore()
    sandbox_id=release.get('sandbox_id')
    contract.validate_release(release,auth,sandbox_id)
    if os.environ.get('MODAL_SANDBOX_ID') not in (None,sandbox_id):
        raise ValueError('provider sandbox ID differs from release')
    ready=release.get('watchdog_readiness',{})
    if (not release.get('app_id') or not isinstance(release.get('watchdog_binding'),dict)
            or any(ready.get(k)!=v for k,v in dict(phase_nonce=auth['phase_nonce'],authorization_sha256=auth['_sha256'],sandbox_id=sandbox_id).items())
            or ready.get('ready') is not True or ready.get('status') != 'monitoring'
            or type(ready.get('pid')) is not int or ready['pid'] <= 0
            or not 0 <= time.time()-contract.timestamp(ready.get('at_utc','1970-01-01T00:00:00+00:00')) <= 30):
        raise ValueError('exact independent watchdog readiness required')
    # Its path belongs to the local launcher; authenticate the carried snapshot,
    # never pretend that that local-only path is a materialized remote file.
    payload=json.dumps(ready,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    import hashlib
    if release['watchdog_binding'].get('sha256')!=hashlib.sha256(payload).hexdigest():
        raise ValueError('watchdog carried snapshot digest differs')
    return release


def cleanup_children(active,*,grace_seconds=10):
    """Caller serializes polling/reaping with registration before signalling."""
    def signal_owned(child,sig):
        if child.poll() is not None:return
        if getattr(child,'_cloud_owned_pid',None)!=child.pid or os.getpgid(child.pid)!=child.pid:
            raise RuntimeError('owned child process-group identity changed; refusing signal')
        os.killpg(child.pid,sig)
    errors=[]
    for child in active:
        try:signal_owned(child,signal.SIGTERM)
        except ProcessLookupError:pass
        except BaseException as error:errors.append(repr(error))
    deadline=time.monotonic()+grace_seconds
    for child in active:
        remaining=max(0,deadline-time.monotonic())
        try:child.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:signal_owned(child,signal.SIGKILL)
            except ProcessLookupError:pass
            except BaseException as error:errors.append(repr(error))
            try:child.wait(timeout=5)
            except BaseException as error:errors.append(repr(error))
        except BaseException as error:errors.append(repr(error))
    if errors:raise RuntimeError('owned cleanup incomplete: '+'; '.join(errors))
    return all(child.poll() is not None for child in active)


def job_receipt(auth,job,child):
    if auth['phase']=='label_efficiency':
        directory=ROOT/contract.LABEL_SCOPE
        attempt=contract.read(directory/'attempt.json');done=contract.read(directory/'completed.json')
        if (attempt.get('pid')!=child.pid or done.get('pid')!=child.pid
                or attempt.get('authorization_sha256')!=auth['_sha256'] or done.get('authorization_sha256')!=auth['_sha256']
                or done.get('status')!='complete' or done.get('protocol_hash')!=contract.phase_protocol_hash(auth)
                or done.get('new_supervised_fits')!=120 or done.get('reused_full_members')!=20
                or set(done.get('member_completed_sha256',{}))!={str(i) for i in range(120)}
                or set(done.get('outputs',{}))!=set(contract.LABEL_RESULTS)-{'attempt.json','completed.json'}):
            raise ValueError('exact label worker completion/fit count required')
        for i in range(120):
            member=directory/'members'/f'member_{i:03d}';completion=contract.read(member/'completed.json')
            if (contract.sha(member/'completed.json')!=done['member_completed_sha256'][str(i)]
                    or completion.get('status')!='complete' or completion.get('n_fits')!=1
                    or completion.get('saved_model_predictions_replayed_exactly') is not True
                    or set(completion.get('payload_sha256',{}))!={'arrays.npz','predictions.csv'}
                    or any(contract.sha(member/name)!=h for name,h in completion['payload_sha256'].items())):
                raise ValueError('label member count/payload/replay mismatch')
        if any(contract.sha(directory/name)!=h for name,h in done['outputs'].items()):
            raise ValueError('label result payload changed')
        return 120
    if auth['phase']=='transfer_recovery':
        regime,cell=job;directory=ROOT/contract.CLOUD_TRANSFER_ROOT/regime/cell
        attempt=contract.read(directory/'attempt.json');receipt=contract.read(directory/'exit_receipt.json')
        done=contract.read(directory/'completed.json')
        if (attempt.get('pid')!=child.pid or attempt.get('authorization_sha256')!=auth['_sha256']
                or receipt.get('status')!='complete' or receipt.get('exit_code')!=0
                or done.get('n_fits')!=400 or done.get('protocol_hash')!=contract.TRANSFER_HASH):
            raise ValueError('exact original transfer worker completion required')
        contract.verify_transfer_cells(contract.CLOUD_TRANSFER_ROOT,{f'{regime}/{cell}':contract.sha(directory/'completed.json')},[f'{regime}/{cell}'])
        return 400
    if auth['phase'] not in ('hybrid_pilot','hybrid_main'):return 0
    fold,rep=job;pilot=auth['phase']=='hybrid_pilot'
    directory=ROOT/contract.SCOPE/('pilot' if pilot else '')/f'fold_{fold}'/rep
    done=contract.read(directory/'completed.json');receipt=contract.read(directory/'exit_receipt.json')
    if (done.get('status')!='complete' or done.get('n_fits')!=(6 if pilot else 48)
            or done.get('protocol_hash')!=contract.SCIENCE_HASH or receipt.get('status')!='complete'
            or receipt.get('exit_code')!=0 or receipt.get('pid')!=child.pid
            or receipt.get('completed_sha256')!=contract.sha(directory/'completed.json')):
        raise ValueError('real child successful completion/fit count mismatch')
    if pilot and (done.get('held_scores_computed') is not False or done.get('reuse_into_scientific') is not False):
        raise ValueError('pilot may not score or supply scientific predictions')
    return done['n_fits']


def require_fresh_outputs(auth):
    phase=auth['phase'];out=ROOT/contract.SCOPE
    if phase=='transfer_recovery':paths=[ROOT/contract.CLOUD_TRANSFER_ROOT/cell for cell in contract.RECOVERY_CELLS]+[ROOT/contract.CLOUD_TRANSFER_ROOT/'recovery_completion.json']
    elif phase=='transfer_report':paths=[ROOT/(contract.CLOUD_TRANSFER_ROOT if auth.get('scope_root')==contract.CLOUD_TRANSFER_ROOT else contract.TRANSFER_ROOT)/'results']
    elif phase=='hybrid_pilot':paths=[out/'pilot']
    elif phase=='hybrid_main':paths=[out/f'fold_{fold}' for fold in range(5)]
    elif phase=='label_efficiency':paths=[ROOT/contract.LABEL_SCOPE/'members']+[ROOT/contract.LABEL_SCOPE/name for name in contract.LABEL_RESULTS]
    else:paths=[out/name for name in ('predictions.csv','scores.csv','comparisons.csv','audit.json','REPORT.md','runtime_summary.json','bootstrap_min20.csv','bootstrap_min50.csv')]
    if any(p.exists() for p in paths):
        raise ValueError('existing scientific/report attempt; no automatic overwrite or retry')


def output_files(auth,run_dir):
    phase=auth['phase'];out=ROOT/contract.SCOPE
    if phase=='transfer_recovery':directories=[ROOT/contract.CLOUD_TRANSFER_ROOT/cell for cell in contract.RECOVERY_CELLS]
    elif phase=='transfer_report':directories=[ROOT/(contract.CLOUD_TRANSFER_ROOT if auth.get('scope_root')==contract.CLOUD_TRANSFER_ROOT else contract.TRANSFER_ROOT)/'results']
    elif phase=='hybrid_pilot':directories=[out/'pilot']
    elif phase=='hybrid_main':directories=[out/f'fold_{fold}' for fold in range(5)]
    else:directories=[]
    files=[p for directory in directories for p in directory.rglob('*') if p.is_file()]
    if phase=='hybrid_report':
        names=['bootstrap_min20.csv','bootstrap_min50.csv','predictions.csv','scores.csv','comparisons.csv','selections.csv',
               'per_allele_scores.csv','per_fold_scores.csv','historical_descriptive_scores.csv','runtime_summary.json','REPORT.md','audit.json']
        files.extend(out/name for name in names)
    if phase=='label_efficiency':
        directory=ROOT/contract.LABEL_SCOPE
        files.extend(directory/name for name in contract.LABEL_RESULTS)
        files.extend(directory/'members'/f'member_{i:03d}'/name for i in range(120)
                     for name in ('attempt.json','arrays.npz','predictions.csv','completed.json'))
    if phase=='hybrid_pilot':files.append(run_dir/'pilot_audit.json')
    if phase=='transfer_recovery':files.append(ROOT/contract.CLOUD_TRANSFER_ROOT/'recovery_completion.json')
    if not files or any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError('missing or unsafe phase outputs')
    return {str(p.relative_to(ROOT)):contract.sha(p) for p in sorted(files)}


def audit_pilot(approved,destination):
    from scripts.final_hybrids import validate_frozen
    from scripts.final_hybrids_cloud import validated_features
    from scripts.final_hybrids_report import audit_pilots
    from threadpoolctl import threadpool_limits
    auth=contract.validate_authorization(approved,'hybrid_pilot')
    if os.environ.get('FINAL_CLOUD_AUTH_SHA256')!=auth['_sha256']:
        raise ValueError('pilot audit requires released parent context')
    with threadpool_limits(limits=2):
        train,protocol=validate_frozen(ROOT/contract.SCOPE)
        features=validated_features(train,protocol)
        telemetry=audit_pilots(ROOT/contract.SCOPE,protocol,train,features)
    contract.exclusive_json(destination,dict(schema='final_cloud_pilot_audit.1',status='passed',n_fits=12,
        held_scores_computed=False,reuse_into_scientific=False,saved_model_predictions_replayed=True,
        protocol_hash=contract.SCIENCE_HASH,authorization_sha256=auth['_sha256'],runtime=telemetry,finished_at_utc=contract.now()))
    contract.durable_sync()


def audit_transfer_recovery(approved):
    """Unchanged transfer semantic audit, without fitting or bootstrap reporting."""
    from scripts.final_transfer import validate_frozen
    from scripts.final_transfer_report import audit_cell
    from pepstab.final_transfer import outer_cells
    from threadpoolctl import threadpool_limits
    auth=contract.validate_authorization(approved,'transfer_recovery')
    if os.environ.get('FINAL_CLOUD_AUTH_SHA256')!=auth['_sha256']:
        raise ValueError('transfer audit requires released supervisor context')
    out=ROOT/contract.CLOUD_TRANSFER_ROOT;hashes={}
    with threadpool_limits(limits=2):
        frame,protocol=validate_frozen(out)
        for regime,cell,fit,held in outer_cells(frame):
            audit_cell(frame,fit,held,regime,cell,out/regime/cell,protocol)
            hashes[f'{regime}/{cell}']=contract.sha(out/regime/cell/'completed.json')
    if set(hashes)!=set(contract.CELLS) or any(hashes[k]!=v for k,v in auth['_predecessor']['preserved_cell_hashes'].items()):
        raise ValueError('recovered scope or preserved cells changed')
    contract.exclusive_json(out/'recovery_completion.json',dict(schema='final_transfer_cloud_recovery_completion.1',
        status='complete',protocol_hash=contract.TRANSFER_HASH,n_retained_fits=10830,n_new_fits=2000,
        preserved_fits=8830,discarded_partial_completed_fits=2579,known_completed_including_discarded=13409,
        unknown_remote_fits=12,unknown_local_fits=True,semantic_audit_passed=True,
        authorization_sha256=auth['_sha256'],failed_predecessor=auth['portable_predecessor'],
        completed_cell_hashes=hashes,finished_at_utc=contract.now()))
    contract.durable_sync()


def run_phase(approved,*,startup_wait_seconds=300,stream=sys.stdin,popen=subprocess.Popen,sleep=time.sleep,monotonic=time.monotonic):
    approved=Path(approved).resolve()
    bare=contract.read(approved)
    if (bare.get('schema')!='final_cloud_phase_authorization.1' or bare.get('approved') is not True
            or bare.get('phase') not in contract.PHASES or not __import__('re').fullmatch(r'[A-Za-z0-9_-]{12,128}',str(bare.get('phase_nonce','')))):
        raise ValueError('exact phase header required before startup')
    expected_run=f"reports/finalization/cloud/remote/{bare['phase_nonce']}"
    if bare.get('run_directory')!=expected_run:raise ValueError('exact immutable remote run directory required')
    run_dir=contract.path_in_root(expected_run);run_dir.mkdir(parents=True,exist_ok=True)
    owner=contract.process_owner();auth=dict(bare,_sha256=contract.sha(approved));started=contract.now();start=monotonic()
    startup=dict(schema='final_cloud_startup.1',phase=bare['phase'],phase_nonce=bare['phase_nonce'],
        authorization_sha256=auth['_sha256'],real_process_owner=owner,started_at_utc=started,sandbox_id=None)
    contract.exclusive_json(run_dir/'startup.json',startup);contract.durable_sync()
    active={};all_children=[];release=None;finished=[];n_fits=0;cleanup=False;reader=None;lease=None
    ownership_lock=threading.RLock()
    def stop_owned_children():
        with ownership_lock:
            return cleanup_children(all_children)
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    interrupted=[False]
    def stop(sig,frame):interrupted[0]=True
    for sig in previous:signal.signal(sig,stop)
    base=dict(schema='final_cloud_phase_exit.1',phase=bare['phase'],phase_nonce=bare['phase_nonce'],
        authorization_sha256=auth['_sha256'],input_manifest_sha256=bare['input_manifest']['sha256'],
        protocol_hash=contract.phase_protocol_hash(bare),real_process_owner=owner,started_at_utc=started)
    if bare['phase']=='label_efficiency':base['protocol']=bare['protocol']
    try:
        environment=verify_environment()
        reader=RawJsonLines(stream)
        release=wait_release(auth,reader,startup_wait_seconds)
        lease=GuardLease(auth,release,reader)
        lease.start_monitor(on_failure=stop_owned_children)
        contract.exclusive_json(run_dir/'release.json',release);contract.durable_sync()
        auth=contract.validate_authorization(approved)
        lease.check()
        base.update(sandbox_id=release['sandbox_id'],app_id=release['app_id'],sandbox_id_source='root_exact_sandbox_stdin_release',environment=environment)
        contract.exclusive_json(run_dir/'attempt.json',dict(base,status='attempted',no_retry=True));contract.durable_sync()
        require_fresh_outputs(auth)
        work_start=monotonic();deadline=min(contract.timestamp(auth['deadline_utc']),time.time()+auth['fit_wall_seconds'])
        env={**os.environ,**{key:'2' for key in THREAD_VARIABLES},'FINAL_CLOUD_SANDBOX_ID':release['sandbox_id'],'FINAL_CLOUD_AUTH_SHA256':auth['_sha256']}
        pending=phase_jobs(auth['phase']);audit_pending=auth['phase'] in ('hybrid_pilot','transfer_recovery')
        def check_running():
            if interrupted[0]:raise RuntimeError('phase interrupted; no retry')
            if time.time()>=deadline or monotonic()-work_start>=auth['fit_wall_seconds']:
                raise TimeoutError('fixed cloud work deadline reached')
            lease.check()
            if contract.sha(approved)!=auth['_sha256']:raise ValueError('authorization changed during phase')
        while pending or active or audit_pending:
            check_running()
            for child,record in list(active.items()):
                with ownership_lock:
                    code=child.poll()
                    if code is not None:del active[child]
                if code is None:continue
                record.update(exit_code=code,finished_at_utc=contract.now());finished.append(record)
                if code:raise RuntimeError(f"cloud child {record['job']} failed; no retry")
                if record['job'] not in ('pilot_saved_model_audit','transfer_semantic_audit'):n_fits+=job_receipt(auth,record['axes'],child)
                contract.exclusive_json(run_dir/f"child_{record['index']:03d}.exit.json",record);contract.durable_sync()
            if pending and len(active)<auth['resources']['max_workers']:
                with ownership_lock:
                    if any(child.poll() is not None for child in active):continue
                axes=pending.pop(0);job=f'{axes[0]}/{axes[1]}' if auth['phase']=='transfer_recovery' else f'fold_{axes[0]}/{axes[1]}' if axes[0] is not None else auth['phase']
                command=child_command(auth,approved,axes)
            elif not pending and not active and audit_pending:
                axes=None;audit_pending=False
                if auth['phase']=='transfer_recovery':
                    job='transfer_semantic_audit';command=[sys.executable,str(Path(__file__).resolve()),'--approved',str(approved),'--audit-transfer-recovery']
                else:
                    job='pilot_saved_model_audit';command=[sys.executable,str(Path(__file__).resolve()),'--approved',str(approved),'--audit-pilot',str(run_dir/'pilot_audit.json')]
            else:
                if active:sleep(.2)
                continue
            command=scientific_entry_command(command,approved)
            index=len(all_children);record=dict(index=index,job=job,axes=axes,command=command,intent_at_utc=contract.now())
            contract.exclusive_json(run_dir/f'child_{index:03d}.intent.json',record);contract.durable_sync()
            with (run_dir/f'child_{index:03d}.log').open('x') as log:
                # Recheck after durable intent/sync: no dispatch on a late stop or dead watcher.
                with ownership_lock:
                    check_running()
                    child=popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    child._cloud_owned_pid=child.pid
                    all_children.append(child);record.update(pid=child.pid,started_at_utc=contract.now());active[child]=record
            contract.exclusive_json(run_dir/f'child_{index:03d}.owner.json',record);contract.durable_sync()
        check_running()
        if n_fits!=contract.PHASES[auth['phase']][-1]:raise ValueError('exact phase fit count not completed')
        cleanup=stop_owned_children()
        if not cleanup:raise RuntimeError('child cleanup unconfirmed')
        manifest=dict(schema='final_cloud_output_manifest.1',phase=auth['phase'],files=output_files(auth,run_dir))
        contract.exclusive_json(run_dir/'output_manifest.json',manifest);contract.durable_sync()
        check_running()
        receipt=dict(base,status='complete',exit_code=0,n_fits=n_fits,cleanup_complete=True,
            finished_at_utc=contract.now(),elapsed_fit_seconds=monotonic()-work_start,
            output_manifest=contract.binding(run_dir/'output_manifest.json'),finished_children=finished,guard_lease=lease.evidence(),
            durability=dict(volume_version=2,method='os.sync',acknowledged=True))
        contract.exclusive_json(run_dir/'exit_receipt.json',receipt);contract.durable_sync()
        check_running()
        return receipt
    except BaseException as error:
        try:cleanup=stop_owned_children()
        except BaseException:cleanup=False
        receipt=dict(base,status='failed',exit_code=1,n_fits=n_fits,cleanup_complete=cleanup,
            finished_at_utc=contract.now(),elapsed_seconds=monotonic()-start,error=repr(error),no_retry=True,finished_children=finished)
        if release:receipt.update(sandbox_id=release['sandbox_id'],app_id=release.get('app_id'))
        if lease:receipt['guard_lease']=lease.evidence()
        if not (run_dir/'exit_receipt.json').exists():
            contract.exclusive_json(run_dir/'exit_receipt.json',receipt);contract.durable_sync()
        raise
    finally:
        if lease is not None:lease.stop_monitor()
        if reader is not None:reader.restore()
        for sig,handler in previous.items():signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved',type=Path,required=True)
    parser.add_argument('--startup-wait-seconds',type=int,default=300)
    parser.add_argument('--audit-pilot',type=Path)
    parser.add_argument('--audit-transfer-recovery',action='store_true')
    args=parser.parse_args()
    if args.audit_transfer_recovery:audit_transfer_recovery(args.approved)
    elif args.audit_pilot:audit_pilot(args.approved,args.audit_pilot)
    else:run_phase(args.approved,startup_wait_seconds=args.startup_wait_seconds)

if __name__=='__main__':main()
