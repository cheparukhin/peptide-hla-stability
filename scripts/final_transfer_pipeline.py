"""One durable, explicitly authorized controller; no automatic launch or retry.

One root process uses <=3 threads for cells. A supplied batch backend may serve
all cloud calls inside ONE root-owned Modal app context (6 calls/cell batch;
<=18 active). The caller owns paid-call reservations and independent guard.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
from threadpoolctl import threadpool_limits
import sys
import time
import threading
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from pepstab.boltz_embeddings import sha256_file
from scripts.final_transfer import OUT,atomic_json,run_cell,validate_frozen,verify_cell
SCHEMA='final_transfer_controller_authorization.1'
ALL_CELLS=tuple(f'hla/h{h}' for h in range(5))+tuple(f'joint/h{h}_p{p}' for h in range(5) for p in range(5))
TIMING_CELLS=('hla/h0','joint/h0_p0')


def now():return datetime.now(timezone.utc).isoformat()


def assert_process_absent(pid):
    if type(pid)!=int or pid<=0:raise ValueError('explicit valid prerequisite PID required')
    try:os.kill(pid,0)
    except ProcessLookupError:return
    raise ValueError('prerequisite ranking/local controller process still exists')


def check_authorization(path,protocol,cells,workers):
    item=json.loads(Path(path).read_text())
    if item.get('schema')!=SCHEMA or item.get('approved') is not True or item.get('protocol_hash')!=protocol['protocol_hash'] or item.get('controller_sha256')!=sha256_file(Path(__file__)):raise ValueError('authorization does not bind exact controller/protocol')
    if len(set(cells))!=len(cells) or not cells or any(c not in ALL_CELLS for c in cells) or item.get('cells')!=list(cells):raise ValueError('authorization cell axes differ')
    if type(workers)!=int or workers not in (1,2,3) or item.get('max_workers')!=workers:raise ValueError('local worker count exceeds cap or authorization')
    if type(item.get('wall_seconds')) not in (int,float) or item['wall_seconds']<=0:raise ValueError('bounded wall deadline required')
    receipt_path=Path(item['prerequisite_exit_receipt']); receipt=json.loads(receipt_path.read_text())
    if sha256_file(receipt_path)!=item.get('prerequisite_exit_receipt_sha256') or receipt.get('status')!='complete':raise ValueError('prerequisite completion not hash-bound')
    assert_process_absent(item['prerequisite_pid'])
    if item.get('ranker_backend') not in ('local','modal'):raise ValueError('explicit backend required')
    if item['ranker_backend']=='modal' and item.get('cloud_max_active')!=18:raise ValueError('cloud batch concurrency differs from3x6 topology')
    return item


def _run_controller(out,approved,cells,workers=1,*,ranker_backend=None):
    out=Path(out);frame,protocol=validate_frozen(out);authorization=check_authorization(approved,protocol,cells,workers)
    if authorization['ranker_backend']=='modal' and ranker_backend is None:raise ValueError('one root-owned guarded Modal batch backend must be supplied')
    directory=out/'controller';directory.mkdir(parents=True,exist_ok=True);lock=(directory/'ownership.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    attempt=directory/'attempt.json'
    try:fd=os.open(attempt,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except BaseException:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close();raise
    started=time.monotonic();state=dict(status='running',pid=os.getpid(),protocol_hash=protocol['protocol_hash'],controller_sha256=sha256_file(Path(__file__)),authorization_sha256=sha256_file(approved),cells=list(cells),max_workers=workers,started_at_utc=now(),wall_seconds=authorization['wall_seconds'],launch_intents=[],finished_cells=[],active_cells=[])
    with os.fdopen(fd,'w') as handle:json.dump(state,handle);handle.flush();os.fsync(handle.fileno())
    save=lambda:atomic_json(directory/'state.json',state)
    pool=ThreadPoolExecutor(max_workers=workers);active={};pending=list(cells);failure=None;cancel_event=threading.Event()
    def guarded_backend(tasks):
        if cancel_event.is_set():raise RuntimeError('controller froze paid dispatch')
        return ranker_backend(tasks)
    try:
        while active or pending:
            if time.monotonic()-started>=authorization['wall_seconds']:raise TimeoutError('controller wall deadline; no queued dispatch')
            if failure:pending=[]
            while pending and len(active)<workers:
                cell=pending.pop(0);record=dict(cell=cell,launch_intent_at_utc=now());state['launch_intents'].append(record);save()
                regime,name=cell.split('/');future=pool.submit(run_cell,out,regime,name,approved,ranker_backend=guarded_backend if ranker_backend is not None else None,cancel_event=cancel_event);active[future]=cell;state['active_cells']=list(active.values());save()
            finished,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
            for future in finished:
                cell=active.pop(future);regime,name=cell.split('/');receipt=dict(cell=cell,finished_at_utc=now())
                try:
                    future.result();verify_cell(out/cell,protocol,regime,name);receipt.update(status='complete',completed_sha256=sha256_file(out/cell/'completed.json'))
                except BaseException as error:
                    receipt.update(status='failed',error=repr(error));failure=error;pending=[];cancel_event.set()
                state['finished_cells'].append(receipt);state['active_cells']=list(active.values());save()
        if failure:raise failure
        state.update(status='complete',finished_at_utc=now());save();atomic_json(directory/'exit_receipt.json',state)
    except BaseException as error:
        cancel_event.set()
        state.update(status='failed',error=repr(error),finished_at_utc=now(),uncertain_active_cells=list(active.values()));save();atomic_json(directory/'exit_receipt.json',state)
        # Thread fits cannot be killed individually. An owned process-level
        # deadline/watchdog must terminate the root process for hard shutdown.
        # No successful exit receipt is emitted while a thread remains live.
        raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    return state


def run_controller(out,approved,cells,workers=1,*,ranker_backend=None):
    with threadpool_limits(limits=2):
        return _run_controller(out,approved,cells,workers,ranker_backend=ranker_backend)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out-dir',type=Path,default=OUT);parser.add_argument('--approved',type=Path,required=True);parser.add_argument('--cell',action='append',required=True);parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args();run_controller(args.out_dir,args.approved,args.cell,args.workers)
if __name__=='__main__':main()
