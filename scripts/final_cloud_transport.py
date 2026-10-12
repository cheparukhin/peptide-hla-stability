"""Bounded, one-attempt input upload and verified result download; never start compute.

Validation is local only. Explicit upload/download use an existing, exact v2
Volume. An upload intent survives uncertainty and forbids another upload call.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pepstab import final_cloud_protocol as c
from scripts.final_sandbox_driver import write_once

REQUIRED_SOURCES = {'scripts/final_cloud_transport.py', 'tests/test_final_cloud_transport.py'}
REPORT_FILES = {'bootstrap_min20.csv','bootstrap_min50.csv','predictions.csv','scores.csv',
    'comparisons.csv','selections.csv','per_allele_scores.csv','per_fold_scores.csv',
    'historical_descriptive_scores.csv','runtime_summary.json','REPORT.md','audit.json'}


def safe_path(name, root=ROOT):
    """Canonical repository-relative files only; check every existing ancestor."""
    if (not isinstance(name, str) or not name or '\\' in name or '\x00' in name
            or PurePosixPath(name).is_absolute() or PurePosixPath(name).as_posix() != name
            or any(part in ('', '.', '..') or part.startswith('.') for part in name.split('/'))):
        raise ValueError('canonical nonhidden repository-relative path required')
    if any(part.lower() in {'secrets', 'credentials'} for part in name.split('/')):
        raise ValueError('credential path refused')
    root = Path(root).resolve(); path = root / name
    cursor = root
    for part in name.split('/'):
        cursor /= part
        if cursor.is_symlink():
            raise ValueError('symlink path refused')
    if not path.resolve().is_relative_to(root):
        raise ValueError('path escapes repository')
    if path.exists() and not path.is_file():
        raise ValueError('explicit regular file required')
    return path


def binding_path(reference, root=ROOT):
    if (not isinstance(reference, dict) or set(reference) != {'path','sha256'}
            or not c.HEX.fullmatch(str(reference['sha256']))):
        raise ValueError('exact file/hash binding required')
    path = safe_path(reference['path'], root)
    if c.sha(path) != reference['sha256']:
        raise ValueError('bound content changed')
    return path


def positive_int(value, maximum):
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError('finite positive integer cap required')
    return value


def validate(path, *, root=ROOT, for_download=False):
    root = Path(root).resolve(); path = Path(path)
    path = safe_path(str(path.relative_to(root)) if path.is_absolute() else str(path), root)
    plan = c.read(path)
    if (plan.get('schema') != 'final_cloud_transport_plan.1' or plan.get('approved') is not True
            or plan.get('profile') != 'a-cheparukhin'):
        raise ValueError('approved exact transport profile required')
    auth_path = binding_path(plan['authorization'], root)
    bare = c.read(auth_path)
    # Downloading completed work after admission expiry is allowed; it never
    # authorizes fresh compute. All other authorization checks still run.
    clock = (lambda: c.timestamp(bare['deadline_utc']) - 1) if for_download else None
    auth = c.validate_authorization(plan['authorization']['path'], root=root, clock=clock, verify_inputs=False)
    nonce = auth['phase_nonce']
    expected_directory = f'reports/finalization/resources/cloud_sandboxes/{nonce}'
    if plan.get('directory') != expected_directory:
        raise ValueError('exact phase transport directory required')
    directory = root / expected_directory
    safe_path(expected_directory + '/transport.marker', root)
    resources = c.read(binding_path(plan['resources'], root))
    if (resources.get('schema') != 'final_cloud_resources.1'
            or resources.get('authorization_sha256') != c.sha(auth_path)
            or resources.get('volume_version') != 2 or resources.get('compute_started') is not False
            or resources.get('volume_name') != f'pepstab-final-{nonce}'
            or resources.get('app_name') != f'pepstab-final-{nonce}'
            or not re.fullmatch(r'vo-[A-Za-z0-9]+', str(resources.get('volume_id','')))
            or not re.fullmatch(r'ap-[A-Za-z0-9]+', str(resources.get('app_id','')))):
        raise ValueError('exact existing owned v2 resource receipt required')
    for key, maximum in [('max_upload_bytes',20*1024**3),('max_download_bytes',20*1024**3),
            ('max_file_bytes',20*1024**3),('max_metadata_bytes',16*1024**2),('max_files',20000),
            ('upload_timeout_seconds',3600),('download_timeout_seconds',3600)]:
        positive_int(plan.get(key), maximum)
    if (plan['max_upload_bytes'] > resources['max_upload_bytes']
            or plan['max_download_bytes'] > resources['max_download_bytes']
            or plan['max_file_bytes'] > plan['max_download_bytes']):
        raise ValueError('transport exceeds reserved byte bounds')
    sources = plan.get('sources',{})
    if not REQUIRED_SOURCES <= set(sources):
        raise ValueError('transport and tests must be source-bound')
    manifest_path = binding_path(auth['input_manifest'], root)
    manifest = c.read(manifest_path)
    if (set(manifest) != {'schema','repository_root','files'}
            or manifest['schema'] != 'final_cloud_input_manifest.1'
            or manifest['repository_root'] != c.REPOSITORY_ROOT
            or not isinstance(manifest['files'],dict) or not manifest['files']
            or len(manifest['files']) + 2 > plan['max_files']):
        raise ValueError('finite exact input hash manifest required')
    for name, digest in sources.items():
        binding_path(dict(path=name,sha256=digest),root)
        if auth['sources'].get(name) != digest or manifest['files'].get(name) != digest:
            raise ValueError('transport source absent from authorization/manifest')
    entries = dict(manifest['files'])
    # Authorization is separate from the manifest to avoid self-hash circularity.
    for ref in (auth['input_manifest'], plan['authorization']):
        if ref['path'] in entries:
            raise ValueError('authorization/manifest must be uploaded separately')
        entries[ref['path']] = ref['sha256']
    for key in ('portable_predecessor','cloud_accounting','runtime_forecast'):
        if key in auth and entries.get(auth[key]['path']) != auth[key]['sha256']:
            raise ValueError('mandatory authorization evidence absent from upload')
    for ref in auth.get('phase_certificates',{}).values():
        if entries.get(ref['path']) != ref['sha256']:
            raise ValueError('preceding certificate absent from upload')
    sizes = {}; total = 0
    for name, digest in entries.items():
        local = binding_path(dict(path=name,sha256=digest),root)
        sizes[name] = local.stat().st_size; total += sizes[name]
        if sizes[name] > plan['max_upload_bytes'] or total > plan['max_upload_bytes']:
            raise ValueError('input upload exceeds exact byte reservation')
    return plan, auth, resources, directory, entries, sizes


async def volume_handle(plan, resources):
    os.environ['MODAL_PROFILE'] = plan['profile']
    import modal
    volume = modal.Volume.from_name(resources['volume_name'], create_if_missing=False, version=2)
    await asyncio.wait_for(volume.hydrate.aio(),30)
    if volume.object_id != resources['volume_id']:
        raise ValueError('existing Volume identity changed')
    return volume


def snapshot(source, target, digest, size):
    """Copy bounded bytes to private staging before handing bytes to SDK."""
    target.parent.mkdir(parents=True,exist_ok=True)
    h=hashlib.sha256(); count=0
    with source.open('rb') as src, target.open('xb') as dst:
        for block in iter(lambda:src.read(1024*1024),b''):
            count+=len(block)
            if count > size:raise ValueError('input grew during staging')
            h.update(block);dst.write(block)
    if count != size or h.hexdigest() != digest:
        raise ValueError('input changed during staging')


async def upload(path, *, root=ROOT):
    plan,auth,resources,directory,entries,sizes=validate(path,root=root)
    intent=dict(schema='final_cloud_upload_intent.1',at_utc=c.now(),no_retry=True,
        plan_sha256=c.sha(path),authorization_sha256=auth['_sha256'],
        input_manifest_sha256=auth['input_manifest']['sha256'],volume_id=resources['volume_id'],
        files=len(entries),bytes=sum(sizes.values()))
    write_once(directory/'upload_intent.json',intent)
    with tempfile.TemporaryDirectory(prefix='upload-',dir=directory) as staging:
        for name,digest in entries.items():
            snapshot(safe_path(name,root),Path(staging)/name,digest,sizes[name])
        volume=await volume_handle(plan,resources)
        async def send():
            # Small batches bound SDK checksum task fanout. Partial/uncertain
            # success still leaves one intent, never authorizing a repeat.
            names=sorted(entries)
            for offset in range(0,len(names),128):
                async with volume.batch_upload.aio(force=False) as batch:
                    for name in names[offset:offset+128]:
                        batch.put_file(Path(staging)/name,'/'+name,mode=0o644)
        await asyncio.wait_for(send(),plan['upload_timeout_seconds'])
    result=dict(intent,schema='final_cloud_upload_receipt.1',status='uploaded',finished_at_utc=c.now())
    write_once(directory/'upload_receipt.json',result)
    return result


def allowed_output(name, auth, root=ROOT):
    safe_path(name,root)  # lexical validation also applies before remote reads
    phase=auth['phase'];scope=c.SCOPE
    if phase=='transfer_recovery':
        return (name==c.CLOUD_TRANSFER_ROOT+'/recovery_completion.json'
            or any(name.startswith(c.CLOUD_TRANSFER_ROOT+'/'+cell+'/') for cell in c.RECOVERY_CELLS))
    if phase=='transfer_report':
        transfer=c.CLOUD_TRANSFER_ROOT if auth.get('scope_root')==c.CLOUD_TRANSFER_ROOT else c.TRANSFER_ROOT
        return name.startswith(transfer+'/results/')
    if phase=='hybrid_pilot':
        return name.startswith(scope+'/pilot/') or name==auth['run_directory']+'/pilot_audit.json'
    if phase=='hybrid_main':return any(name.startswith(f'{scope}/fold_{fold}/') for fold in range(5))
    return name in {scope+'/'+filename for filename in REPORT_FILES}


async def read_remote(volume,name,target,limit,budget,*,digest=None):
    h=hashlib.sha256();size=0
    with target.open('xb') as handle:
        async for chunk in volume.read_file.aio('/'+name):
            if not isinstance(chunk,bytes):raise ValueError('provider returned nonbyte payload')
            size+=len(chunk);budget[0]+=len(chunk)
            if size>limit or budget[0]>budget[1]:raise ValueError('remote download byte cap exceeded')
            h.update(chunk);handle.write(chunk)
        handle.flush();os.fsync(handle.fileno())
    if digest is not None and h.hexdigest()!=digest:raise ValueError('remote output hash mismatch')
    return h.hexdigest()


def install(staged,name,root):
    target=safe_path(name,root)
    target.parent.mkdir(parents=True,exist_ok=True)
    safe_path(name,root)
    if target.exists():
        if c.sha(target)!=c.sha(staged):raise ValueError('different local output already exists')
        return
    # Atomic creation without replacing a concurrently created destination.
    temporary=target.parent/('.transport-'+uuid.uuid4().hex)
    try:
        with staged.open('rb') as source, temporary.open('xb') as dest:
            shutil.copyfileobj(source,dest,1024*1024);dest.flush();os.fsync(dest.fileno())
        try:os.link(temporary,target)
        except FileExistsError:
            if c.sha(target)!=c.sha(staged):raise ValueError('local output changed during install') from None
        fd=os.open(target.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:temporary.unlink(missing_ok=True)


async def download(path, terminal_path, *, certificate_path=None, root=ROOT):
    plan,auth,resources,directory,_,_=validate(path,root=root,for_download=True)
    # Bind watchdog evidence to the exact launch and creation, never launcher_exit.
    terminal_path=Path(terminal_path)
    if terminal_path!=directory/'watchdog_exit.json':raise ValueError('exact watchdog terminal path required')
    terminal=c.read(terminal_path);created=c.read(directory/'created.json')
    launch_path=directory/'launch_plan.json'
    launch=c.read(launch_path);launch_hash=c.sha(launch_path)
    expected=dict(authorization_sha256=auth['_sha256'],plan_sha256=launch_hash)
    if (any(terminal.get(k)!=v or created.get(k)!=v for k,v in expected.items())
            or terminal.get('status')!='terminal_observed' or type(terminal.get('returncode')) is not int
            or terminal['returncode']!=0 or terminal.get('reason')!='provider_terminal'
            or terminal.get('sandbox_id')!=created.get('sandbox_id')
            or not str(created.get('sandbox_id','')).startswith('sb-')
            or created.get('app_id')!=resources['app_id'] or created.get('volume_id')!=resources['volume_id']
            or launch.get('authorization')!=plan['authorization'] or launch.get('app_id')!=resources['app_id']
            or launch.get('volume_id')!=resources['volume_id'] or launch.get('volume_version')!=2):
        raise ValueError('successful exact provider terminal/creation linkage required')
    volume=await volume_handle(plan,resources)
    import modal
    sandbox=await asyncio.wait_for(modal.Sandbox.from_id.aio(created['sandbox_id']),30)
    tags=await asyncio.wait_for(sandbox.get_tags.aio(),30)
    code=await asyncio.wait_for(sandbox.poll.aio(),30)
    if (sandbox.object_id!=created['sandbox_id'] or type(code) is not int or code!=0
            or any(tags.get(k)!=v for k,v in dict(phase=auth['phase'],phase_nonce=auth['phase_nonce'],authorization_sha256=auth['_sha256']).items())):
        raise ValueError('actual provider successful terminal and exact tags required')
    observed=c.now();budget=[0,plan['max_download_bytes']]
    names={};run=auth['run_directory']
    with tempfile.TemporaryDirectory(prefix='download-',dir=directory) as stage:
        stage=Path(stage)
        async def receive():
            exit_name=run+'/exit_receipt.json'; manifest_name=run+'/output_manifest.json'
            safe_path(exit_name,root);safe_path(manifest_name,root)
            exit_temp=stage/'exit';manifest_temp=stage/'manifest'
            exit_digest=await read_remote(volume,exit_name,exit_temp,plan['max_metadata_bytes'],budget)
            receipt=c.read(exit_temp)
            expected=dict(schema='final_cloud_phase_exit.1',status='complete',phase=auth['phase'],phase_nonce=auth['phase_nonce'],
                sandbox_id=created['sandbox_id'],app_id=resources['app_id'],authorization_sha256=auth['_sha256'],
                input_manifest_sha256=auth['input_manifest']['sha256'],exit_code=0,cleanup_complete=True,
                protocol_hash=c.TRANSFER_HASH if auth['phase'] in ('transfer_recovery','transfer_report') else c.SCIENCE_HASH,n_fits=c.PHASES[auth['phase']][-1])
            if (any(receipt.get(k)!=v for k,v in expected.items())
                    or type(receipt.get('exit_code')) is not int
                    or not receipt.get('real_process_owner',{}).get('pid_namespace')
                    or receipt.get('durability')!={'volume_version':2,'method':'os.sync','acknowledged':True}):
                raise ValueError('successful exact remote exit/cleanup/durability required')
            ref=receipt.get('output_manifest',{})
            if set(ref)!={'path','sha256'} or ref['path']!=manifest_name or not c.HEX.fullmatch(str(ref['sha256'])):
                raise ValueError('exact remote manifest binding required')
            await read_remote(volume,manifest_name,manifest_temp,plan['max_metadata_bytes'],budget,digest=ref['sha256'])
            manifest=c.read(manifest_temp);files=manifest.get('files')
            if (set(manifest)!={'schema','phase','files'} or manifest['schema']!='final_cloud_output_manifest.1'
                    or manifest['phase']!=auth['phase'] or not isinstance(files,dict) or not files
                    or len(files)+2>plan['max_files']):raise ValueError('finite exact phase output manifest required')
            for name,digest in files.items():
                safe_path(name,root)
                if not allowed_output(name,auth,root) or not c.HEX.fullmatch(str(digest)):
                    raise ValueError('output outside expected phase prefixes/hash manifest')
            names.update({exit_name:exit_temp,manifest_name:manifest_temp})
            for index,(name,digest) in enumerate(sorted(files.items())):
                target=stage/f'output-{index}'
                await read_remote(volume,name,target,plan['max_file_bytes'],budget,digest=digest)
                names[name]=target
            c.finite_number(receipt['elapsed_fit_seconds'])
            if not (c.timestamp(receipt['started_at_utc'])<=c.timestamp(receipt['finished_at_utc'])
                    <=c.timestamp(auth['deadline_utc']) and c.timestamp(receipt['finished_at_utc'])
                    <=c.timestamp(terminal['at_utc'])<=c.timestamp(observed)):
                raise ValueError('remote/provider success chronology mismatch')
            return receipt,exit_digest,ref
        receipt,exit_digest,ref=await asyncio.wait_for(receive(),plan['download_timeout_seconds'])
        # Check every destination before installing any staged output.
        for name,temp in names.items():
            local=safe_path(name,root)
            if local.exists() and c.sha(local)!=c.sha(temp):raise ValueError('different local output already exists')
        for name,temp in names.items():install(temp,name,root)
        certificate={key:receipt[key] for key in ('phase','phase_nonce','sandbox_id','app_id','authorization_sha256',
            'input_manifest_sha256','protocol_hash','n_fits','started_at_utc','finished_at_utc','elapsed_fit_seconds')}
        certificate.update(schema='finalization_sandbox_phase_completion.1',status='complete',
            runner_exit={'path':run+'/exit_receipt.json','sha256':exit_digest},output_manifest=ref,
            provider_terminal=dict(sandbox_id=created['sandbox_id'],returncode=code,observed_at_utc=observed))
        checked=stage/'certificate.json';write_once(checked,certificate)
        # Includes pilot replay audit checks; no certificate can bypass them.
        check_name=str((directory/f'certificate-check-{uuid.uuid4().hex}.json').relative_to(root))
        install(checked,check_name,root)
        try:c.verify_phase_certificate(dict(path=check_name,sha256=c.sha(checked)),auth['phase'],root)
        finally:(root/check_name).unlink()
        if certificate_path is not None:
            destination=Path(certificate_path)
            name=str(destination.relative_to(root)) if destination.is_absolute() else str(destination)
            install(checked,name,root)
        return certificate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['validate','upload','download'])
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--terminal',type=Path)
    parser.add_argument('--certificate',type=Path)
    args=parser.parse_args()
    if args.mode=='validate':validate(args.plan);print('Validated; no provider API called.')
    elif args.mode=='upload':print(json.dumps(asyncio.run(upload(args.plan)),indent=2))
    else:
        if args.terminal is None or args.certificate is None:parser.error('download requires --terminal and --certificate')
        print(json.dumps(asyncio.run(download(args.plan,args.terminal,certificate_path=args.certificate)),indent=2))

if __name__=='__main__':main()
