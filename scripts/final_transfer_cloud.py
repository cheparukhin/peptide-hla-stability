#!/usr/bin/env python3
"""Explicitly approved one-app smoke/main driver; waits for independent guard.

A prelaunch plan approves app creation only for a frozen scope. After app ID
receipt, root binds/starts the independent guard. No remote call precedes a
fresh healthy handshake. No controller recovery or automatic retry is provided.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
import modal
from modal_app.final_transfer import app, ranker_member, smoke_app, smoke_member
from pepstab.final_modal_backend import GuardedBatchBackend, CALL_UPPER, RATE, OVERHEAD_SECONDS, TIMEOUT_SECONDS, write_once
from pepstab.round2_budget import ReservationBudget
from pepstab.boltz_embeddings import sha256_file
from pepstab import final_transfer as ft
from scripts.final_transfer import validate_frozen
from scripts.final_transfer_pipeline import run_controller, ALL_CELLS

def utcnow():
    return datetime.now(timezone.utc).isoformat()


def make_smoke_tasks(frame):
    """One synthetic compatibility case and one full-size training-only member.

    The real member measures runtime; its predictions are never scored or reused
    by the main assessment. No held-out test rows enter either task.
    """
    _, _, fit, held = next(x for x in ft.outer_cells(frame) if x[:2] == ('hla','h0'))
    candidate = next(c for c in ft.CANDIDATES if c['id']=='fixed_ranker8')
    member = next(ft.member_schedule(fit,'hla',candidate,final=True))
    columns = ['pair_id','cluster_id','hla_group','allele','split','peptide','hla_pseudoseq','y_log1p']
    base = dict(candidate=candidate,seed=0,stopping_cell=member['stopping_cell'],regime='hla',
                gradient=member['gradient'][columns].to_dict('records'),
                stopping=member['stopping'][columns].to_dict('records'),
                held=held[[c for c in columns if c!='y_log1p']].to_dict('records'))
    real = dict(base,context='pilot/full-size-runtime-only')
    synthetic = dict(base,context='pilot/synthetic-platform-compatibility')
    rng = np.random.default_rng(20261012)
    for key in ('gradient','stopping','held'):
        synthetic[key] = [dict(row) for row in base[key][:64]]
        if key != 'held':
            for row, y in zip(synthetic[key],rng.normal(size=len(synthetic[key]))):
                row['y_log1p'] = float(y)
    for task in (synthetic,real):
        task['task_id']=hashlib.sha256(json.dumps(task,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return [synthetic,real]


def load_launch_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get('schema')!='final_transfer_cloud_launch.1' or plan.get('approved') is not True:
        raise ValueError('explicit prelaunch approval required')
    if plan.get('phase') not in ('smoke','main'):
        raise ValueError('unknown scope')
    if not 0 < plan['scope_ceiling_usd'] <= 100 or not 0 <= plan['initial_reserve_usd'] < plan['scope_ceiling_usd']:
        raise ValueError('invalid bounded allocation')
    if plan['max_calls'] != (2 if plan['phase']=='smoke' else 1620):
        raise ValueError('frozen scope call count changed')
    if plan.get('source_hashes',{}).get('scripts/final_transfer_cloud.py') != sha256_file(Path(__file__)):
        raise ValueError('driver source is not bound')
    for file, digest in plan['source_hashes'].items():
        if sha256_file(ROOT/file) != digest:
            raise ValueError(f'prelaunch source changed: {file}')
    frame, protocol = validate_frozen(plan['out_dir'])
    if protocol['protocol_hash'] != plan['protocol_hash']:
        raise ValueError('prelaunch scientific protocol changed')
    if plan['phase']=='smoke':
        if sha256_file(plan['tasks_path']) != plan['tasks_sha256']:
            raise ValueError('smoke task file changed')
        tasks=json.loads(Path(plan['tasks_path']).read_text())
        if tasks != make_smoke_tasks(frame):
            raise ValueError('smoke task identities changed')
    elif plan.get('cells') != list(ALL_CELLS) or plan.get('workers') != 3:
        raise ValueError('main scope must be exactly30 frozen cells and3 local workers')
    return plan


def validate_guard_binding(auth, plan, app_id, controller_pid):
    """Bind resource/wall accounting to the actual app, not just a healthy file."""
    exact = dict(app_id=app_id, controller_pid=controller_pid,
        ledger_path=str(Path(plan['ledger_path']).resolve()), scope_kind=plan['phase'],
        max_concurrency=1 if plan['phase']=='smoke' else 18,
        cpu_cores=2, memory_mib=4096)
    if any(auth.get(k)!=v for k,v in exact.items()):
        raise ValueError('guard app/controller/resource/scope binding disagrees')
    for key, expected in dict(rate_usd_per_second=RATE,call_timeout_seconds=TIMEOUT_SECONDS,
            per_call_overhead_seconds=OVERHEAD_SECONDS,per_call_upper_usd=CALL_UPPER,
            initial_reserve_usd=plan['initial_reserve_usd']).items():
        value=auth.get(key)
        if type(value) not in (int,float) or not math.isclose(value,expected,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('guard resource rate/reservation binding disagrees')
    if not math.isclose(auth['total_allocation_usd']-auth['prior_charges_usd'],plan['scope_ceiling_usd'],rel_tol=0,abs_tol=1e-10):
        raise ValueError('guard scope allocation disagrees with journal')
    expected_sources={str((ROOT/name).resolve()):h for name,h in plan['source_hashes'].items()}
    if any(auth.get('source_hashes',{}).get(name)!=h for name,h in expected_sources.items()):
        raise ValueError('guard does not bind every approved source')
    protocol_path=str((Path(plan['out_dir'])/'protocol.json').resolve())
    if auth.get('protocol_hashes',{}).get(protocol_path)!=sha256_file(protocol_path):
        raise ValueError('guard does not bind frozen scientific protocol file')


def run(plan_path):
    plan = load_launch_plan(plan_path)
    directory=Path(plan['resource_dir']);directory.mkdir(parents=True,exist_ok=True)
    write_once(directory/'driver.launched.json',dict(pid=os.getpid(),at_utc=utcnow(),plan_sha256=sha256_file(plan_path),phase=plan['phase']))
    if Path(plan['ledger_path']).exists():
        raise ValueError('existing ledger requires reconciliation; no automatic recovery')
    budget=ReservationBudget(plan['ledger_path'],ceiling_usd=plan['scope_ceiling_usd'],
        initial_reserve_usd=plan['initial_reserve_usd'],call_upper_usd=CALL_UPPER,max_calls=plan['max_calls'])
    cloud_app, function = (smoke_app,smoke_member) if plan['phase']=='smoke' else (app,ranker_member)
    def startup_timeout(signum,frame):
        raise TimeoutError('fixed600-second app-build/guard handshake deadline; no retry')
    signal.signal(signal.SIGALRM,startup_timeout);signal.alarm(600)
    app_id=None
    try:
        with modal.enable_output(),cloud_app.run():
            app_id=cloud_app.app_id
            write_once(directory/'app_ready.json',dict(app_id=app_id,pid=os.getpid(),at_utc=utcnow(),phase=plan['phase'],calls_dispatched=0))
            wait_until=time.monotonic()+300
            auth_path=Path(plan['guard_authorization_path'])
            while not auth_path.exists():
                if time.monotonic()>=wait_until:raise TimeoutError('guard authorization handshake expired')
                time.sleep(.5)
            backend=GuardedBatchBackend(function,budget,auth_path,directory/'calls',max_batch=1 if plan['phase']=='smoke' else 6)
            auth=backend.authorization
            validate_guard_binding(auth,plan,app_id,os.getpid())
            while True:
                try:
                    backend.check_guard();break
                except (FileNotFoundError,RuntimeError):
                    if Path(auth['stop_flag_path']).exists() or time.monotonic()>=wait_until:raise
                    time.sleep(.5)
            signal.alarm(0)
            if plan['phase']=='smoke':
                tasks=json.loads(Path(plan['tasks_path']).read_text())
                with threadpool_limits(limits=2):
                    local=ft.ranker_task(tasks[0])
                remote=backend([tasks[0]])[0]
                np.testing.assert_allclose(local['predictions'],remote['predictions'],rtol=1e-8,atol=1e-8)
                if local['details']['tree_count']!=remote['details']['tree_count']:
                    raise ValueError('platform compatibility changed stopping')
                real=backend([tasks[1]])[0]
                write_once(directory/'smoke.completed.json',dict(status='complete',protocol_hash=plan['protocol_hash'],
                    synthetic_max_abs_difference=float(np.max(np.abs(np.asarray(local['predictions'])-remote['predictions']))),
                    local_synthetic_versions=local['versions'],remote_synthetic_versions=remote['versions'],
                    full_size_worker_seconds=real['worker_seconds'],full_size_tree_count=real['details']['tree_count'],
                    full_size_rows={k:len(tasks[1][k]) for k in ('gradient','stopping','held')},
                    cloud_calls=2,actual_local_synthetic_fits=1,full_size_predictions_scored=False,
                    budget=budget.status(),at_utc=utcnow()))
            else:
                run_controller(plan['out_dir'],plan['controller_authorization'],plan['cells'],3,ranker_backend=backend)
        write_once(directory/'driver.exit.json',dict(status='complete',pid=os.getpid(),app_id=app_id,app_context_closed=True,
                   at_utc=utcnow(),budget=budget.status()))
    except BaseException as error:
        write_once(directory/'driver.exit.json',dict(status='failed',pid=os.getpid(),app_id=app_id,error=repr(error),
                   at_utc=utcnow(),budget=budget.status(),requires_app_and_ledger_reconciliation=True))
        raise
    finally:
        signal.alarm(0);budget.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True)
    run(parser.parse_args().plan)
