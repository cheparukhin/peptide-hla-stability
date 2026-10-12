"""Prepare uniquely named empty cloud resources once; never start compute.

The input manifest and authenticated closed predecessor must already be frozen.
An uncertain provider response preserves its intent and forbids a second call.
Use read-only provider reconciliation, not rerunning this command, after error.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pepstab import final_cloud_protocol as c
from scripts.final_sandbox_driver import write_once


def validate(path, *, verify_inputs=True):
    path = c.path_in_root(path)
    plan = c.read(path)
    if (plan.get('schema') != 'final_cloud_resource_plan.1' or plan.get('approved') is not True
            or plan.get('profile') != 'a-cheparukhin'):
        raise ValueError('approved exact profile resource plan required')
    auth_path = c.bound(plan['authorization'])
    auth = c.validate_authorization(auth_path, verify_inputs=verify_inputs)
    nonce = auth['phase_nonce']
    expected_name = f'pepstab-final-{nonce}'
    if (plan.get('app_name') != expected_name or plan.get('volume_name') != expected_name
            or not re.fullmatch(r'[a-z0-9-]{12,100}', nonce) or plan.get('volume_version') != 2):
        raise ValueError('unique exact phase resource names/version required')
    directory = c.path_in_root(plan['directory'])
    if directory != ROOT / 'reports/finalization/resources/cloud_sandboxes' / nonce:
        raise ValueError('resource ownership directory mismatch')
    sources = plan.get('sources', {})
    if not {'scripts/final_cloud_resources.py', 'tests/test_final_cloud_resources.py'} <= set(sources):
        raise ValueError('resource preparation sources must be bound')
    c.verify_files(sources)
    if any(auth['sources'].get(name) != digest for name, digest in sources.items()):
        raise ValueError('resource sources absent from phase authorization')
    now = time.time()
    if not 0 <= now - c.timestamp(plan['created_at_utc']) <= 900:
        raise ValueError('resource creation window expired; no automatic reauthorization')
    cleanup = c.timestamp(plan['storage_cleanup_deadline_utc'])
    if not c.timestamp(auth['deadline_utc']) < cleanup <= c.timestamp('2026-10-15T14:15:17.567539+00:00'):
        raise ValueError('bounded storage cleanup deadline required')
    if (type(plan.get('max_upload_bytes')) is not int or not 0 < plan['max_upload_bytes'] <= 20 * 1024**3
            or type(plan.get('max_download_bytes')) is not int or not 0 < plan['max_download_bytes'] <= 20 * 1024**3
            or plan.get('overhead_envelope_usd') != 3 or plan.get('simultaneously_retained_volumes') != 1):
        raise ValueError('finite storage/egress caps required')
    manifest_path = c.bound(auth['input_manifest'])
    manifest = c.read(manifest_path)
    total = sum(c.path_in_root(name).stat().st_size for name in manifest['files'])
    total += manifest_path.stat().st_size + auth_path.stat().st_size
    if total > plan['max_upload_bytes']:
        raise ValueError('exact upload manifest exceeds byte reservation')
    gib = 1024**3
    storage = (plan['max_upload_bytes'] + plan['max_download_bytes']) / gib * .09 * (cleanup-now) / (30*86400)
    # Recovery plus four finalization downloads, with no free-tier assumption.
    downloads = 5 if auth.get('phase') == 'transfer_recovery' else 4
    egress = downloads * plan['max_download_bytes'] / gib * .04
    if storage + egress + .5 > plan['overhead_envelope_usd']:
        raise ValueError('storage, all-phase egress and cleanup reserve exceed overhead envelope')
    return plan, auth, directory


async def create(path):
    plan, auth, directory = validate(path)
    os.environ['MODAL_PROFILE'] = plan['profile']
    import modal
    write_once(directory / 'resource_intent.json', dict(schema='final_cloud_resource_intent.1',
        at_utc=c.now(), pid=os.getpid(), plan_sha256=c.sha(path), authorization_sha256=auth['_sha256'],
        app_name=plan['app_name'], volume_name=plan['volume_name'], no_retry=True))
    # Both existing names must be absent before either creation. Authentication
    # or transport errors are not interpreted as absence.
    from modal.exception import NotFoundError
    try:
        await asyncio.wait_for(modal.App.lookup.aio(plan['app_name'], create_if_missing=False), 30)
    except NotFoundError:
        pass
    else:
        raise ValueError('named App already exists; reconcile ownership without creation')
    existing = modal.Volume.from_name(plan['volume_name'], create_if_missing=False, version=2)
    try:
        await asyncio.wait_for(existing.hydrate.aio(), 30)
    except NotFoundError:
        pass
    else:
        raise ValueError('named Volume already exists; reconcile ownership without creation')
    app = await asyncio.wait_for(modal.App.lookup.aio(plan['app_name'], create_if_missing=True), 60)
    write_once(directory / 'app_created.json', dict(at_utc=c.now(), app_id=app.app_id,
        app_name=plan['app_name'], plan_sha256=c.sha(path)))
    await asyncio.wait_for(modal.Volume.objects.create.aio(plan['volume_name'], version=2, allow_existing=False), 60)
    volume = modal.Volume.from_name(plan['volume_name'], create_if_missing=False, version=2)
    await asyncio.wait_for(volume.hydrate.aio(), 30)
    result = dict(schema='final_cloud_resources.1', at_utc=c.now(), app_id=app.app_id,
        app_name=plan['app_name'], volume_id=volume.object_id, volume_name=plan['volume_name'], volume_version=2,
        plan_sha256=c.sha(path), authorization_sha256=auth['_sha256'], compute_started=False,
        max_upload_bytes=plan['max_upload_bytes'], max_download_bytes=plan['max_download_bytes'],
        storage_cleanup_deadline_utc=plan['storage_cleanup_deadline_utc'])
    write_once(directory / 'resources.json', result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['validate', 'create'])
    parser.add_argument('--plan', required=True, type=Path)
    args=parser.parse_args()
    if args.mode=='validate':
        validate(args.plan);print('Validated; no cloud API called.')
    else:
        print(json.dumps(asyncio.run(create(args.plan)), indent=2))


if __name__=='__main__': main()
