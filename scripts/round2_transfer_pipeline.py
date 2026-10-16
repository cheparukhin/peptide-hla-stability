"""Explicitly authorized, durable local transfer phases; never retry an attempt.

Timing runs hla/h0 then joint/h0_p0 serially. Remaining runs the other 28 cells
at one to three workers only after timing authorization. This controller reads
completion receipts and hashes, never numerical prediction tables, and does not
run the transfer report. A failed or interrupted attempt requires root recovery.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pepstab.boltz_embeddings import sha256_file
from scripts.round2_transfer import OUT, validate_frozen, verify_cell

SCHEMA = 'round2_transfer_controller_authorization.1'
TIMING_CELLS = ('hla/h0', 'joint/h0_p0')
ALL_CELLS = tuple(f'hla/h{h}' for h in range(5)) + tuple(
    f'joint/h{h}_p{p}' for h in range(5) for p in range(5))


def atomic_json(path, item):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as handle:
        json.dump(item,handle,indent=2)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def phase_cells(phase):
    if phase == 'timing':
        return TIMING_CELLS
    if phase == 'remaining':
        return tuple(cell for cell in ALL_CELLS if cell not in TIMING_CELLS)
    raise ValueError('unknown transfer controller phase')


def cell_command(cell, out, grouping, main_root, feature_manifest):
    regime, name = cell.split('/')
    return [str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/round2_transfer.py'),
        'run-cell', '--out-dir', str(out), '--grouping-dir', str(grouping),
        '--main-root', str(main_root), '--feature-manifest', str(feature_manifest),
        '--regime', regime, '--cell', name]


def check_authorization(path, phase, protocol, workers, out):
    if path is None:
        raise ValueError('explicit --approved authorization JSON required; no launch')
    path = Path(path)
    authorization = json.loads(path.read_text())
    cells = phase_cells(phase)
    script_hash = sha256_file(Path(__file__))
    if (authorization.get('schema') != SCHEMA or authorization.get('approved') is not True
        or authorization.get('phase') != phase
        or authorization.get('protocol_hash') != protocol['protocol_hash']
        or authorization.get('controller_sha256') != script_hash
        or authorization.get('cells') != list(cells)):
        raise ValueError('authorization does not bind exact controller/protocol/phase/cells')
    if (type(workers) is not int or workers not in (1,2,3)
        or type(authorization.get('max_workers')) is not int
        or authorization['max_workers'] != workers or (phase == 'timing' and workers != 1)):
        raise ValueError('authorization worker count mismatch or exceeds local phase cap')
    if phase == 'remaining':
        expected = authorization.get('timing_completed_sha256')
        if not isinstance(expected,dict) or set(expected) != set(TIMING_CELLS):
            raise ValueError('remaining authorization must bind both timing completion hashes')
        receipt_path = Path(out)/'controller/timing_exit_receipt.json'
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get('status') != 'complete' or receipt.get('protocol_hash') != protocol['protocol_hash']
            or receipt.get('controller_sha256') != script_hash or receipt.get('cells') != list(TIMING_CELLS)):
            raise ValueError('timing phase lacks a matching successful exit receipt')
        children = receipt.get('finished_children',[])
        if (receipt.get('max_workers') != 1 or len(children) != 2
            or [child.get('cell') for child in children] != list(TIMING_CELLS)
            or any(child.get('exit_code') != 0 or child.get('disposition') != 'complete'
                   or child.get('completed_sha256') != expected[child['cell']] for child in children)):
            raise ValueError('both timing children must have successful hash-bound exit receipts')
        for cell in TIMING_CELLS:
            regime,name = cell.split('/')
            directory = Path(out)/regime/name
            verify_cell(directory,protocol,regime,name)
            if sha256_file(directory/'completed.json') != expected[cell]:
                raise ValueError('timing completion hash changed after authorization')
    return dict(path=str(path.resolve()),sha256=sha256_file(path),document=authorization,
                controller_sha256=script_hash)


def write_launch_intent(path, state):
    """Exclusive intent precedes every Popen, including uncertain launch windows."""
    descriptor = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    with os.fdopen(descriptor,'w') as handle:
        json.dump(state,handle,indent=2)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())


def stop_children(active, state, save):
    """Best-effort cleanup cannot depend on a writable receipt filesystem."""
    cleanup_errors = []
    for child,record in list(active.items()):
        try:
            running = child.poll() is None
        except BaseException as error:
            running = True
            cleanup_errors.append(dict(pid=record['pid'],action='poll',error=repr(error)))
        if running:
            try:
                child.terminate()
            except BaseException as error:
                cleanup_errors.append(dict(pid=record['pid'],action='terminate',error=repr(error)))
    for child,record in list(active.items()):
        code = None
        try:
            code = child.wait(timeout=30)
        except BaseException as error:
            cleanup_errors.append(dict(pid=record['pid'],action='wait',error=repr(error)))
            try:
                child.kill()
            except BaseException as kill_error:
                cleanup_errors.append(dict(pid=record['pid'],action='kill',error=repr(kill_error)))
            try:
                code = child.wait(timeout=30)
            except BaseException as wait_error:
                cleanup_errors.append(dict(pid=record['pid'],action='wait_after_kill',error=repr(wait_error)))
        if code is not None:
            state['finished_children'].append(dict(record,exit_code=code,
                disposition='aborted_after_controller_interrupt',uncertain=True,finished_at_utc=now()))
            del active[child]
        else:
            active[child] = dict(record,cleanup_status='uncertain',uncertain=True)
    state['active_children'] = list(active.values())
    if cleanup_errors:
        state['cleanup_errors'] = cleanup_errors
    try:
        save()
    except BaseException as error:
        state['cleanup_persistence_error'] = repr(error)



def run_phase_jobs(jobs, state, save, protocol, out, max_workers):
    """Stop queued launches on failure; let in-flight siblings settle normally."""
    pending, active, failure = list(jobs), {}, None
    env = {**os.environ, **{key:'2' for key in (
        'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS')}}
    def halt_dispatch(error):
        nonlocal failure
        if failure is None:
            failure = error
        pending.clear()
        state.update(dispatch_stopped=True,dispatch_error=repr(failure),dispatch_stopped_at_utc=now())
        save()
    try:
        while pending or active:
            while pending and len(active) < max_workers:
                job = pending.pop(0)
                record = dict(job,launch_status='intent_persisted',launch_intent_at_utc=now())
                state['launch_intents'].append(record)
                save()  # crash here is uncertain, so the phase marker bars retry
                try:
                    with Path(job['log']).open('a') as log:
                        child = subprocess.Popen(job['command'],cwd=ROOT,env=env,
                            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                except Exception as error:
                    state.setdefault('uncertain_launches',[]).append(dict(record,error=repr(error)))
                    halt_dispatch(error)
                    break
                record = dict(record,pid=child.pid,launch_status='launched',started_at_utc=now())
                active[child] = record
                state['active_children'] = list(active.values())
                save()
            for child, record in list(active.items()):
                code = child.poll()
                if code is None:
                    continue
                del active[child]
                receipt = dict(record,exit_code=code,finished_at_utc=now())
                state['finished_children'].append(receipt)
                state['active_children'] = list(active.values())
                save()
                if code:
                    receipt['disposition'] = 'failed'
                    halt_dispatch(RuntimeError(f"{record['cell']} exited {code}; root recovery required; see {record['log']}"))
                    continue
                regime,name = record['cell'].split('/')
                directory = Path(out)/regime/name
                try:
                    verify_cell(directory,protocol,regime,name)
                    receipt['completed_sha256'] = sha256_file(directory/'completed.json')
                    receipt['disposition'] = 'complete'
                except Exception as error:
                    receipt.update(disposition='completion_unverified',error=repr(error))
                    halt_dispatch(error)
                save()
            if active:
                time.sleep(1)
        if failure is not None:
            raise failure
    finally:
        # Cleanup runs even when the failure came from persistent save errors.
        # stop_children absorbs individual cleanup/persistence errors so the
        # original dispatch failure or interrupt remains the raised exception.
        if active:
            state['controller_interrupted'] = True
            state['active_children'] = list(active.values())
            stop_children(active,state,save)


def execute_phase(phase, approved, *, out=OUT, grouping=OUT, main_root=OUT.parent,
                  feature_manifest=None, workers=1):
    # Approval is required before any scientific gate or subprocess dispatch.
    if approved is None:
        raise ValueError('explicit --approved authorization JSON required; no launch')
    if feature_manifest is None:
        raise ValueError('explicit feature manifest required')
    out,grouping,main_root = map(Path,(out,grouping,main_root))
    directory = out/'controller';directory.mkdir(parents=True,exist_ok=True)
    with (directory/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        intent_path = directory/f'{phase}_launch_intent.json'
        status_path = directory/f'{phase}_status.json'
        exit_path = directory/f'{phase}_exit_receipt.json'
        if any(path.exists() for path in (intent_path,status_path,exit_path)):
            raise ValueError('phase already attempted or uncertain; root recovery required; no retry')
        _,protocol = validate_frozen(out,grouping,main_root)
        authorization = check_authorization(approved,phase,protocol,workers,out)
        cells = phase_cells(phase)
        for cell in cells:
            cell_directory = out/cell
            if cell_directory.exists() and any(cell_directory.iterdir()):
                raise ValueError('cell already complete or partially written/running/uncertain; root recovery required')
        jobs = [dict(index=index,cell=cell,command=cell_command(cell,out,grouping,main_root,feature_manifest),
                     log=str(directory/f'{phase}_{index:02d}_{cell.replace("/","_")}.log'))
                for index,cell in enumerate(cells)]
        state = dict(schema='round2_transfer_controller_state.1',phase=phase,status='launch_intent',
            pid=os.getpid(),started_at_utc=now(),protocol_hash=protocol['protocol_hash'],
            controller_sha256=authorization['controller_sha256'],authorization=authorization,
            cells=list(cells),max_workers=workers,jobs=jobs,launch_intents=[],
            active_children=[],finished_children=[],cloud_spend_usd=0)
        write_launch_intent(intent_path,state)
        def save():
            state['updated_at_utc'] = now()
            atomic_json(status_path,state)
            atomic_json(directory/'status.json',state)
        try:
            state['status'] = 'running';save()
            run_phase_jobs(jobs,state,save,protocol,out,workers)
            if len(state['finished_children']) != len(cells) or any(
                child.get('disposition') != 'complete' for child in state['finished_children']):
                raise RuntimeError('incomplete phase exit receipts; root recovery required')
            state['status'] = 'complete'
        except BaseException as error:
            state.update(status='failed',error=repr(error),recovery='root inspection required; no automatic retry')
            raise
        finally:
            state['finished_at_utc'] = now()
            original_failure = state['status'] == 'failed'
            persistence_errors = []
            for persist in (save,lambda:atomic_json(exit_path,state)):
                try:
                    persist()
                except BaseException as error:
                    persistence_errors.append(error)
                    state.setdefault('exit_persistence_errors',[]).append(repr(error))
            if persistence_errors and not original_failure:
                raise persistence_errors[0]
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=('timing','remaining'))
    parser.add_argument('--approved',type=Path,required=True)
    parser.add_argument('--workers',type=int,choices=(1,2,3),default=1)
    parser.add_argument('--out-dir',type=Path,default=OUT)
    parser.add_argument('--grouping-dir',type=Path,default=OUT)
    parser.add_argument('--main-root',type=Path,default=OUT.parent)
    parser.add_argument('--feature-manifest',type=Path,required=True)
    args = parser.parse_args()
    def interrupted(number,_frame):
        raise RuntimeError(f'controller interrupted by signal {number}; root recovery required')
    signal.signal(signal.SIGTERM,interrupted)
    execute_phase(args.phase,args.approved,out=args.out_dir,grouping=args.grouping_dir,
        main_root=args.main_root,feature_manifest=args.feature_manifest,workers=args.workers)


if __name__ == '__main__':
    main()
