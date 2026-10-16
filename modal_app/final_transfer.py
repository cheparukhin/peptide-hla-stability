"""Explicit fixed-depth transfer PairLogit worker; importing does not launch.

A root-owned driver opens exactly one app context, records appID before calls,
reserves all timeout-bound calls, checks an independent guard before every six
member batch, and closes the app context on every exit. No local entrypoint is
provided: this module cannot launch paid work by a generic CLI invocation.
"""
from __future__ import annotations
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1]
TASK_TIMEOUT=1000
STARTUP_TIMEOUT=60
SCALEDOWN_SECONDS=2
MAX_CALLS=1620
MAX_CONTAINERS=18
EXPECTED_ACTIVE=18
app=modal.App('pepstab-final-transfer-controls')
image=(modal.Image.debian_slim(python_version='3.12')
    .pip_install('numpy==2.5.3','pandas==3.0.6','scipy==1.18.1','scikit-learn==1.9.1','catboost==1.2.10','threadpoolctl==3.7.0')
    .env({k:'2' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}))
for module in ('__init__.py','augment.py','data.py','features.py','mlp.py','evaluation.py','splits.py','nested.py','phase_a.py','round2_nested.py','round2_sequence.py','round2_reporting.py','round2_embedding.py','round2_transfer.py','final_transfer.py'):
    image=image.add_local_file(ROOT/'pepstab'/module,f'/root/pepstab/{module}',copy=True)

@app.function(image=image,cpu=(2.,2.),memory=(4096,4096),timeout=TASK_TIMEOUT,startup_timeout=STARTUP_TIMEOUT,
    max_containers=MAX_CONTAINERS,min_containers=0,buffer_containers=0,scaledown_window=SCALEDOWN_SECONDS,retries=0)
def ranker_member(task:dict)->dict:
    from pepstab.final_transfer import ranker_task
    return ranker_task(task)


def timeout_cost_plan(cpu_hour_usd,gib_hour_usd):
    """Caller supplies fresh rates: plan is a bound, never launch approval."""
    if cpu_hour_usd<=0 or gib_hour_usd<=0:raise ValueError('fresh positive rates required')
    rate=2*cpu_hour_usd+4*gib_hour_usd
    return dict(member_count=MAX_CALLS,max_active=EXPECTED_ACTIVE,hard_container_cap=MAX_CONTAINERS,
        requested_cpu=2,requested_gib=4,task_timeout_seconds=TASK_TIMEOUT,startup_timeout_seconds=STARTUP_TIMEOUT,
        scaledown_seconds=SCALEDOWN_SECONDS,per_call_upper_usd=(TASK_TIMEOUT+STARTUP_TIMEOUT+SCALEDOWN_SECONDS)*rate/3600,
        total_call_upper_usd=MAX_CALLS*(TASK_TIMEOUT+STARTUP_TIMEOUT+SCALEDOWN_SECONDS)*rate/3600,
        full_task_timeout_wall_hours=MAX_CALLS*TASK_TIMEOUT/EXPECTED_ACTIVE/3600,
        automatic_retries=0,preemption_restart_policy='independent wall/resource watchdog required; retries0 does not bound provider restart billing',
        status='PLAN_ONLY_REQUIRES_MEASURED_PILOT_FRESH_RATES_AND_ROOT_GUARD')


smoke_app=modal.App('pepstab-final-transfer-smoke')


@smoke_app.function(image=image,cpu=(2.,2.),memory=(4096,4096),timeout=TASK_TIMEOUT,startup_timeout=STARTUP_TIMEOUT,
    max_containers=1,min_containers=0,buffer_containers=0,scaledown_window=SCALEDOWN_SECONDS,retries=0)
def smoke_member(task:dict)->dict:
    from pepstab.final_transfer import ranker_task
    return ranker_task(task)
