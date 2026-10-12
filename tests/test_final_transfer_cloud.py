import hashlib
import json
from pathlib import Path

import pytest

from scripts import final_transfer_cloud as driver


def test_prelaunch_requires_scope_sources_and_quota(tmp_path):
    path=tmp_path/'plan.json';path.write_text('{}')
    with pytest.raises(ValueError,match='approval'):driver.load_launch_plan(path)
    plan=dict(schema='final_transfer_cloud_launch.1',approved=True,phase='smoke',scope_ceiling_usd=2.,initial_reserve_usd=1.35,max_calls=1620)
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='call count'):driver.load_launch_plan(path)
    plan['max_calls']=2;path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='source'):driver.load_launch_plan(path)


def test_prelaunch_rejects_task_change_before_app_creation(tmp_path,monkeypatch):
    taskpath=tmp_path/'tasks.json';taskpath.write_text('[]')
    plan=dict(schema='final_transfer_cloud_launch.1',approved=True,phase='smoke',scope_ceiling_usd=2.,initial_reserve_usd=1.35,max_calls=2,
        source_hashes={'scripts/final_transfer_cloud.py':driver.sha256_file(Path(driver.__file__))},out_dir=str(tmp_path),protocol_hash='p',
        tasks_path=str(taskpath),tasks_sha256='0'*64)
    monkeypatch.setattr(driver,'validate_frozen',lambda out:(None,dict(protocol_hash='p')))
    p=tmp_path/'plan.json';p.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='task file'):driver.load_launch_plan(p)
    plan['tasks_sha256']=driver.sha256_file(taskpath);p.write_text(json.dumps(plan))
    monkeypatch.setattr(driver,'make_smoke_tasks',lambda frame:[{'expected':'exact'}])
    with pytest.raises(ValueError,match='identities'):driver.load_launch_plan(p)


def test_main_requires_full_frozen_cells_and_three_workers(tmp_path,monkeypatch):
    plan=dict(schema='final_transfer_cloud_launch.1',approved=True,phase='main',scope_ceiling_usd=63.,initial_reserve_usd=0.,max_calls=1620,
        source_hashes={'scripts/final_transfer_cloud.py':driver.sha256_file(Path(driver.__file__))},out_dir=str(tmp_path),protocol_hash='p',cells=['hla/h0'],workers=3)
    monkeypatch.setattr(driver,'validate_frozen',lambda out:(None,dict(protocol_hash='p')))
    p=tmp_path/'plan.json';p.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='exactly30'):driver.load_launch_plan(p)


def test_guard_cannot_understate_actual_app_concurrency(tmp_path):
    protocol=tmp_path/'protocol.json';protocol.write_text('{}')
    plan=dict(phase='main',ledger_path=str(tmp_path/'ledger'),out_dir=str(tmp_path),scope_ceiling_usd=63.,initial_reserve_usd=0.,source_hashes={})
    auth=dict(app_id='ap-example',controller_pid=123,ledger_path=plan['ledger_path'],scope_kind='main',
        max_concurrency=1,cpu_cores=2,memory_mib=4096,rate_usd_per_second=driver.RATE,
        call_timeout_seconds=1000,per_call_overhead_seconds=62,per_call_upper_usd=driver.CALL_UPPER,
        initial_reserve_usd=0.,total_allocation_usd=65.,prior_charges_usd=2.,source_hashes={},
        protocol_hashes={str(protocol):driver.sha256_file(protocol)})
    with pytest.raises(ValueError,match='resource/scope'):driver.validate_guard_binding(auth,plan,'ap-example',123)
    auth['max_concurrency']=18
    driver.validate_guard_binding(auth,plan,'ap-example',123)
    auth['scope_kind']='smoke'
    with pytest.raises(ValueError,match='resource/scope'):driver.validate_guard_binding(auth,plan,'ap-example',123)
