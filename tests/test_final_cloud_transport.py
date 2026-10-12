"""Transport checks use synthetic files and fake providers only; no fits/API calls."""
import asyncio
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
from scripts import final_cloud_transport as t

NOW='2026-10-12T18:00:00+00:00'


def put(root,name,value):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(value if isinstance(value,bytes) else json.dumps(value,sort_keys=True).encode())
    return {'path':name,'sha256':t.c.sha(path)}


def setup(tmp_path,monkeypatch):
    nonce='report-20261012-test';directory=f'reports/finalization/resources/cloud_sandboxes/{nonce}'
    sources={}
    for name in sorted(t.REQUIRED_SOURCES):
        sources[name]=put(tmp_path,name,b'# test frozen source\n')['sha256']
    predecessor=put(tmp_path,'portable.json',{'synthetic':'predecessor'})
    account=put(tmp_path,'account.json',{'synthetic':'account'})
    payload=put(tmp_path,'input.bin',b'opaque exact bytes')
    manifest=put(tmp_path,'manifest.json',dict(schema='final_cloud_input_manifest.1',repository_root=t.c.REPOSITORY_ROOT,
        files={**sources,predecessor['path']:predecessor['sha256'],account['path']:account['sha256'],payload['path']:payload['sha256']}))
    auth=dict(schema='final_cloud_phase_authorization.1',approved=True,phase='transfer_report',
        phase_nonce=nonce,sources=copy.deepcopy(sources),input_manifest=manifest,
        portable_predecessor=predecessor,cloud_accounting=account,
        phase_certificates={},deadline_utc='2026-10-12T23:00:00+00:00',
        run_directory=f'reports/finalization/cloud/remote/{nonce}')
    reference=put(tmp_path,'auth.json',auth)
    auth['_sha256']=reference['sha256']
    # Full scientific predecessor validation is covered by the protocol tests;
    # this fixture substitutes only that gate, retaining real transport hashes.
    monkeypatch.setattr(t.c,'validate_authorization',lambda *a,**kw:copy.deepcopy(auth))
    monkeypatch.setattr(t.c,'now',lambda:NOW)
    resources=put(tmp_path,directory+'/resources.json',dict(schema='final_cloud_resources.1',
        authorization_sha256=auth['_sha256'],volume_version=2,compute_started=False,
        volume_id='vo-exact',app_id='ap-exact',volume_name=f'pepstab-final-{nonce}',app_name=f'pepstab-final-{nonce}',
        max_upload_bytes=1024**2,max_download_bytes=1024**2))
    plan=dict(schema='final_cloud_transport_plan.1',approved=True,profile='a-cheparukhin',
        authorization=reference,resources=resources,directory=directory,sources=copy.deepcopy(sources),
        max_upload_bytes=1024**2,max_download_bytes=1024**2,max_file_bytes=1024,
        max_metadata_bytes=16384,max_files=100,upload_timeout_seconds=5,download_timeout_seconds=5)
    path=tmp_path/'transport.json';path.write_text(json.dumps(plan))
    return path,plan,auth,tmp_path/directory


def provider(monkeypatch,*,remote=None,fail_upload=False,volume_id='vo-exact',code=0,tags=None):
    calls=[];uploaded={};remote=remote or {}
    class Batch:
        async def __aenter__(self):calls.append(('batch',));return self
        async def __aexit__(self,*args):
            if fail_upload:raise TimeoutError('uncertain upload')
        def put_file(self,local,remote,mode):
            calls.append(('put',remote,mode));uploaded[remote]=Path(local).read_bytes()
    def from_name(name,**kwargs):
        calls.append(('volume',name,kwargs))
        async def hydrate():calls.append(('hydrate',))
        def batch_upload(**kwargs):
            assert kwargs=={'force':False}
            return Batch()
        async def read_file(name):
            calls.append(('read',name))
            for i in range(0,len(remote[name]),17):yield remote[name][i:i+17]
        return NS(object_id=volume_id,hydrate=NS(aio=hydrate),batch_upload=NS(aio=batch_upload),read_file=NS(aio=read_file))
    async def from_id(sandbox_id):
        calls.append(('sandbox',sandbox_id))
        async def get_tags():return tags or {}
        async def poll():calls.append(('poll',));return code
        return NS(object_id=sandbox_id,get_tags=NS(aio=get_tags),poll=NS(aio=poll))
    monkeypatch.setitem(sys.modules,'modal',NS(Volume=NS(from_name=from_name),Sandbox=NS(from_id=NS(aio=from_id))))
    return calls,uploaded


def remote_success(root,plan,auth,directory):
    launch=dict(authorization=plan['authorization'],app_id='ap-exact',volume_id='vo-exact',volume_version=2)
    launch_ref=put(root,str(directory.relative_to(root))+'/launch_plan.json',launch)
    base=dict(authorization_sha256=auth['_sha256'],plan_sha256=launch_ref['sha256'],sandbox_id='sb-exact')
    put(root,str(directory.relative_to(root))+'/created.json',dict(base,app_id='ap-exact',volume_id='vo-exact'))
    terminal_path=directory/'watchdog_exit.json'
    put(root,str(terminal_path.relative_to(root)),dict(base,status='terminal_observed',returncode=0,reason='provider_terminal',at_utc=NOW))
    output=t.c.TRANSFER_ROOT+'/results/REPORT.md';data=b'Synthetic report\n'
    manifest=dict(schema='final_cloud_output_manifest.1',phase=auth['phase'],files={output:hashlib.sha256(data).hexdigest()})
    encoded=json.dumps(manifest,sort_keys=True).encode()
    receipt=dict(schema='final_cloud_phase_exit.1',status='complete',phase=auth['phase'],phase_nonce=auth['phase_nonce'],
        sandbox_id='sb-exact',app_id='ap-exact',authorization_sha256=auth['_sha256'],input_manifest_sha256=auth['input_manifest']['sha256'],
        protocol_hash=t.c.TRANSFER_HASH,n_fits=0,exit_code=0,cleanup_complete=True,
        real_process_owner={'pid_namespace':'pid:[synthetic]'},started_at_utc='2026-10-12T17:00:00+00:00',
        finished_at_utc='2026-10-12T17:30:00+00:00',elapsed_fit_seconds=1800,
        output_manifest={'path':auth['run_directory']+'/output_manifest.json','sha256':hashlib.sha256(encoded).hexdigest()},
        durability={'volume_version':2,'method':'os.sync','acknowledged':True})
    remote={'/'+output:data,'/'+receipt['output_manifest']['path']:encoded,
        '/'+auth['run_directory']+'/exit_receipt.json':json.dumps(receipt).encode()}
    tags=dict(phase=auth['phase'],phase_nonce=auth['phase_nonce'],authorization_sha256=auth['_sha256'])
    return terminal_path,remote,tags,output,receipt,manifest


def replace_remote(remote,auth,receipt,manifest):
    encoded=json.dumps(manifest).encode()
    receipt['output_manifest']['sha256']=hashlib.sha256(encoded).hexdigest()
    remote['/'+receipt['output_manifest']['path']]=encoded
    remote['/'+auth['run_directory']+'/exit_receipt.json']=json.dumps(receipt).encode()


def test_pure_validate_and_exact_separate_upload(tmp_path,monkeypatch):
    path,plan,auth,_=setup(tmp_path,monkeypatch)
    _,_,_,_,entries,sizes=t.validate(path,root=tmp_path)
    assert entries['auth.json']==auth['_sha256'] and 'manifest.json' in entries
    assert sum(sizes.values())<plan['max_upload_bytes']


@pytest.mark.parametrize('field,value',[
    ('approved',False),('profile','other'),('sources',{}),('directory','escape'),
    ('max_upload_bytes',True),('max_upload_bytes',1),('max_download_bytes',0),
    ('max_download_bytes',21*1024**3),('max_file_bytes',1024**2+1),('max_metadata_bytes',float('inf')),
    ('max_files',1),('upload_timeout_seconds',3601),('download_timeout_seconds',0)])
def test_plan_fails_closed(tmp_path,monkeypatch,field,value):
    path,plan,_,_=setup(tmp_path,monkeypatch);plan[field]=value;path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):t.validate(path,root=tmp_path)


@pytest.mark.parametrize('name',['../escape','/absolute','a/../b','a//b','./a','a/./b','.env','a\\b','secrets/key','a/'])
def test_paths_fail_closed(tmp_path,name):
    with pytest.raises(ValueError):t.safe_path(name,tmp_path)


def test_symlink_ancestor_and_changed_input(tmp_path,monkeypatch):
    path,_,_,_=setup(tmp_path,monkeypatch)
    (tmp_path/'link').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError):t.safe_path('link/input.bin',tmp_path)
    (tmp_path/'input.bin').write_bytes(b'changed')
    with pytest.raises(ValueError):t.validate(path,root=tmp_path)


@pytest.mark.parametrize('fail_upload',[False,True])
def test_once_only_upload_including_uncertainty(tmp_path,monkeypatch,fail_upload):
    path,plan,_,directory=setup(tmp_path,monkeypatch)
    calls,uploaded=provider(monkeypatch,fail_upload=fail_upload)
    if fail_upload:
        with pytest.raises(TimeoutError):asyncio.run(t.upload(path,root=tmp_path))
        assert not (directory/'upload_receipt.json').exists()
    else:
        result=asyncio.run(t.upload(path,root=tmp_path))
        assert result['status']=='uploaded'
        assert uploaded['/auth.json']==(tmp_path/'auth.json').read_bytes()
        assert uploaded['/manifest.json']==(tmp_path/'manifest.json').read_bytes()
    before=len(calls)
    with pytest.raises(FileExistsError):asyncio.run(t.upload(path,root=tmp_path))
    assert len(calls)==before
    assert calls[0][2]=={'create_if_missing':False,'version':2}
    assert (directory/'upload_intent.json').exists()


def test_wrong_volume_id_prevents_write(tmp_path,monkeypatch):
    path,_,_,_=setup(tmp_path,monkeypatch);calls,_=provider(monkeypatch,volume_id='vo-wrong')
    with pytest.raises(ValueError):asyncio.run(t.upload(path,root=tmp_path))
    assert not any(c[0]=='put' for c in calls)


def test_verified_download_emits_real_completion(tmp_path,monkeypatch):
    path,plan,auth,directory=setup(tmp_path,monkeypatch)
    terminal,remote,tags,output,_,_=remote_success(tmp_path,plan,auth,directory)
    calls,_=provider(monkeypatch,remote=remote,tags=tags)
    destination=tmp_path/'certificate.json'
    cert=asyncio.run(t.download(path,terminal,certificate_path=destination,root=tmp_path))
    assert cert['provider_terminal']=={'sandbox_id':'sb-exact','returncode':0,'observed_at_utc':NOW}
    assert (tmp_path/output).read_bytes()==remote['/'+output]
    assert destination.is_file() and ('poll',) in calls
    assert t.c.verify_phase_certificate({'path':'certificate.json','sha256':t.c.sha(destination)},auth['phase'],tmp_path)==cert


@pytest.mark.parametrize('change',['hash','unsafe','phase','cleanup','exit','binding','existing','cap','metadata_cap','total_cap','provider','tags','watchdog','chronology','nonfinite','durability'])
def test_download_no_certificate_on_failure(tmp_path,monkeypatch,change):
    path,plan,auth,directory=setup(tmp_path,monkeypatch)
    terminal,remote,tags,output,receipt,manifest=remote_success(tmp_path,plan,auth,directory)
    code=0
    if change=='hash':remote['/'+output]=b'changed'
    elif change=='unsafe':manifest['files']={'../escape':'a'*64}
    elif change=='phase':manifest['files']={t.c.SCOPE+'/pilot/model.bin':'a'*64}
    elif change=='cleanup':receipt['cleanup_complete']=False
    elif change=='exit':receipt['exit_code']=1
    elif change=='binding':receipt['sandbox_id']='sb-other'
    elif change=='existing':put(tmp_path,output,b'original local content')
    elif change=='cap':plan['max_file_bytes']=1
    elif change=='metadata_cap':plan['max_metadata_bytes']=1
    elif change=='total_cap':plan['max_download_bytes']=plan['max_file_bytes']=1
    elif change=='provider':code=None
    elif change=='tags':tags['phase_nonce']='different'
    elif change=='watchdog':
        data=t.c.read(terminal);data['returncode']=1;terminal.write_text(json.dumps(data))
    elif change=='chronology':receipt['finished_at_utc']='2026-10-13T00:00:00+00:00'
    elif change=='nonfinite':receipt['elapsed_fit_seconds']=float('nan')
    elif change=='durability':receipt['durability']['acknowledged']=False
    path.write_text(json.dumps(plan));replace_remote(remote,auth,receipt,manifest)
    provider(monkeypatch,remote=remote,tags=tags,code=code)
    cert=tmp_path/'certificate.json'
    with pytest.raises(ValueError):asyncio.run(t.download(path,terminal,certificate_path=cert,root=tmp_path))
    assert not cert.exists()
    if change=='existing':assert (tmp_path/output).read_bytes()==b'original local content'
    else:assert not (tmp_path/output).exists()


def test_duplicate_json_manifest_refused(tmp_path,monkeypatch):
    path,plan,auth,directory=setup(tmp_path,monkeypatch)
    terminal,remote,tags,_,receipt,_=remote_success(tmp_path,plan,auth,directory)
    data=b'{"schema":"final_cloud_output_manifest.1","phase":"transfer_report","files":{},"files":{}}'
    receipt['output_manifest']['sha256']=hashlib.sha256(data).hexdigest()
    remote['/'+receipt['output_manifest']['path']]=data
    remote['/'+auth['run_directory']+'/exit_receipt.json']=json.dumps(receipt).encode()
    provider(monkeypatch,remote=remote,tags=tags)
    with pytest.raises(ValueError):asyncio.run(t.download(path,terminal,root=tmp_path))


@pytest.mark.parametrize('name,allowed',[
    ('joint/h4_p0/completed.json',True),('joint/h4_p4/member_predictions.npz',True),
    ('recovery_completion.json',True),('joint/h3_p4/completed.json',False),
    ('joint/h4_p5/completed.json',False),('results/REPORT.md',False),('checkpoints/joint/h4_p0/call.json',False)])
def test_recovery_outputs_have_exact_new_scope(tmp_path,name,allowed):
    auth={'phase':'transfer_recovery'}
    assert t.allowed_output(t.c.CLOUD_TRANSFER_ROOT+'/'+name,auth,tmp_path)==allowed


def test_successor_report_outputs_use_completed_cloud_scope(tmp_path):
    auth={'phase':'transfer_report','scope_root':t.c.CLOUD_TRANSFER_ROOT}
    assert t.allowed_output(t.c.CLOUD_TRANSFER_ROOT+'/results/REPORT.md',auth,tmp_path)
    assert not t.allowed_output(t.c.TRANSFER_ROOT+'/results/REPORT.md',auth,tmp_path)


def test_legacy_report_auth_scope_keeps_original_transfer_results(tmp_path):
    auth={'phase':'transfer_report','scope_root':t.c.SCOPE}
    assert t.allowed_output(t.c.TRANSFER_ROOT+'/results/REPORT.md',auth,tmp_path)
    assert not t.allowed_output(t.c.CLOUD_TRANSFER_ROOT+'/results/REPORT.md',auth,tmp_path)
