"""Resource ownership/admission tests; no provider contact or model fitting."""
import asyncio
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from scripts import final_cloud_resources as r


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    monkeypatch.setattr(r.c, 'path_in_root', lambda p: tmp_path / p if not Path(p).is_absolute() else Path(p))
    monkeypatch.setattr(r.c, 'bound', lambda ref: tmp_path / ref['path'])
    monkeypatch.setattr(r.c, 'verify_files', lambda *_: None)
    auth = dict(phase_nonce='report-20261012-test', deadline_utc='2026-10-12T23:00:00+00:00',
        sources={'scripts/final_cloud_resources.py':'a'*64,'tests/test_final_cloud_resources.py':'b'*64},
        input_manifest={'path':'manifest.json','sha256':'c'*64},_sha256='d'*64)
    monkeypatch.setattr(r.c, 'validate_authorization', lambda *a,**kw:copy.deepcopy(auth))
    monkeypatch.setattr(r.time,'time',lambda:datetime.fromisoformat('2026-10-12T17:00:00+00:00').timestamp())
    for name,text in [('auth.json','{}'),('manifest.json',json.dumps({'files':{'input.bin':'e'*64}})),('input.bin','bytes')]:
        (tmp_path/name).write_text(text)
    plan=dict(schema='final_cloud_resource_plan.1',approved=True,profile='a-cheparukhin',
        authorization={'path':'auth.json','sha256':'d'*64}, app_name='pepstab-final-report-20261012-test',
        volume_name='pepstab-final-report-20261012-test',volume_version=2,
        directory='reports/finalization/resources/cloud_sandboxes/report-20261012-test',
        sources=copy.deepcopy(auth['sources']),created_at_utc='2026-10-12T17:00:00+00:00',
        storage_cleanup_deadline_utc='2026-10-15T14:15:17.567539+00:00',
        max_upload_bytes=3*1024**3,max_download_bytes=8*1024**3,
        overhead_envelope_usd=3,simultaneously_retained_volumes=1)
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    return path,plan,auth


def test_valid_preparation_has_no_provider_import(tmp_path,monkeypatch):
    path,plan,auth=fixture(tmp_path,monkeypatch)
    assert r.validate(path)[0]==plan


@pytest.mark.parametrize('field,value',[
    ('approved',False),('profile','other'),('volume_version',1),('app_name','shared-app'),
    ('volume_name','shared-volume'),('directory','elsewhere'),('sources',{}),
    ('created_at_utc','2026-10-12T16:00:00+00:00'),('created_at_utc','2026-10-12T18:00:00+00:00'),
    ('max_upload_bytes',1),('max_download_bytes',21*1024**3),('max_download_bytes',20*1024**3),
    ('overhead_envelope_usd',4),('simultaneously_retained_volumes',3),
    ('simultaneously_retained_volumes',True),('simultaneously_retained_volumes',1.0),
    ('simultaneously_retained_volumes',2.0),('simultaneously_retained_volumes',0),
    ('storage_cleanup_deadline_utc','2026-10-16T00:00:00+00:00'),
    ('storage_cleanup_deadline_utc','2026-10-12T18:00:00+00:00')])
def test_invalid_resource_admission(tmp_path,monkeypatch,field,value):
    path,plan,_=fixture(tmp_path,monkeypatch)
    plan[field]=value;path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):r.validate(path)


def mock_provider(monkeypatch, *, app_exists=False, fail_create=False):
    class Missing(Exception):pass
    calls=[]
    async def lookup(name,*,create_if_missing):
        calls.append(('app',create_if_missing))
        if not create_if_missing and not app_exists:raise Missing('absent')
        if create_if_missing and fail_create:raise TimeoutError('uncertain create')
        return NS(app_id='ap-exact')
    volumes=[]
    def from_name(name,**kwargs):
        index=len(volumes)
        async def hydrate():
            if index==0:raise Missing('absent')
        volume=NS(object_id='vo-exact',hydrate=NS(aio=hydrate));volumes.append(volume)
        return volume
    create=AsyncMock()
    monkeypatch.setitem(sys.modules,'modal',NS(App=NS(lookup=NS(aio=lookup)),Volume=NS(from_name=from_name,objects=NS(create=NS(aio=create)))))
    monkeypatch.setitem(sys.modules,'modal.exception',NS(NotFoundError=Missing))
    return calls,create


def test_unique_creation_and_exclusive_attempt(tmp_path,monkeypatch):
    path,_,_=fixture(tmp_path,monkeypatch);calls,create=mock_provider(monkeypatch)
    result=asyncio.run(r.create(path))
    assert result['volume_id']=='vo-exact' and result['compute_started'] is False
    assert calls==[('app',False),('app',True)]
    create.assert_awaited_once()
    with pytest.raises(FileExistsError):asyncio.run(r.create(path))
    assert len(calls)==2


@pytest.mark.parametrize('app_exists,fail_create',[(True,False),(False,True)])
def test_existing_or_uncertain_resources_never_retried(tmp_path,monkeypatch,app_exists,fail_create):
    path,_,_=fixture(tmp_path,monkeypatch);calls,create=mock_provider(monkeypatch,app_exists=app_exists,fail_create=fail_create)
    with pytest.raises((ValueError,TimeoutError)):asyncio.run(r.create(path))
    original=list(calls)
    with pytest.raises(FileExistsError):asyncio.run(r.create(path))
    assert calls==original
    create.assert_not_awaited()


def test_recovery_resource_budget_counts_fifth_download(tmp_path,monkeypatch):
    path,plan,auth=fixture(tmp_path,monkeypatch)
    auth['phase']='transfer_recovery'
    assert r.validate(path)[0]==plan
    # With five downloads this cap exhausts the $3 overhead reservation;
    # accepting the old four-download arithmetic would incorrectly pass.
    plan['max_download_bytes']=12*1024**3;path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='overhead'):r.validate(path)


@pytest.mark.parametrize('slots',[1,2])
def test_one_or_two_retained_volumes_with_eight_gib_results(tmp_path,monkeypatch,slots):
    path,plan,_=fixture(tmp_path,monkeypatch)
    plan['simultaneously_retained_volumes']=slots;path.write_text(json.dumps(plan))
    assert r.validate(path)[0]==plan


def test_storage_upper_bound_counts_both_retained_volumes(tmp_path,monkeypatch):
    path,plan,_=fixture(tmp_path,monkeypatch)
    plan['max_download_bytes']=14*1024**3
    path.write_text(json.dumps(plan))
    assert r.validate(path)[0]==plan
    plan['simultaneously_retained_volumes']=2;path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='overhead'):r.validate(path)
