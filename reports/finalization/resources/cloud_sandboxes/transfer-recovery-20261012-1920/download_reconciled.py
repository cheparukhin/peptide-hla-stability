"""One-off retrieval after a successful Sandbox exit raced its watchdog stream.

Preserves the original watchdog receipt. Accepts only its exact zero-return
shutdown race plus fresh provider success and the unchanged scientific checks.
No compute, fitting, retry or source/protocol change is performed.
"""
import asyncio,json,os,shutil,sys,tempfile
from pathlib import Path
ROOT=Path('/Users/cheparukhin/ai_science_hack')
sys.path.insert(0,str(ROOT))
from pepstab import final_cloud_protocol as c
from scripts import final_cloud_transport as t
from scripts.final_sandbox_driver import write_once
P=ROOT/'reports/finalization/resources/cloud_sandboxes/transfer-recovery-20261012-1920'
async def main():
 plan,auth,res,directory,_,_=t.validate(P/'transport_plan.json',for_download=True)
 assert directory==P
 terminal=c.read(P/'watchdog_exit.json');created=c.read(P/'created.json');launch=c.read(P/'launch_plan.json')
 sid=created['sandbox_id'];expected={'authorization_sha256':auth['_sha256'],'plan_sha256':c.sha(P/'launch_plan.json')}
 assert all(terminal[k]==created[k]==v for k,v in expected.items())
 assert terminal['status']=='failed_stop_observed' and terminal['returncode']==0
 assert terminal['sandbox_id']==sid and terminal['owned_sandbox_ids']==[sid]
 assert terminal['reason'].startswith("NotFoundError('Modal Sandbox with container ID ") and 'already shut down' in terminal['reason']
 assert terminal['stop_results']=={sid:dict(sandbox_id=sid,returncode=0,errors=[],stop_observed=True)}
 assert created['app_id']==res['app_id'] and created['volume_id']==res['volume_id']
 assert launch['authorization']==plan['authorization'] and launch['volume_id']==res['volume_id'] and launch['app_id']==res['app_id']
 assert shutil.disk_usage(ROOT).free>2*plan['max_download_bytes']
 volume=await t.volume_handle(plan,res)
 import modal
 sandbox=await modal.Sandbox.from_id.aio(sid)
 assert await sandbox.poll.aio()==0
 tags=await sandbox.get_tags.aio()
 assert all(tags.get(k)==v for k,v in dict(phase=auth['phase'],phase_nonce=auth['phase_nonce'],authorization_sha256=auth['_sha256']).items())
 observed=c.now();run=auth['run_directory'];budget=[0,plan['max_download_bytes']]
 reconciliation=dict(schema='final_cloud_successful_terminal_reconciliation.1',status='fresh_provider_success_after_guard_shutdown_race',at_utc=observed,sandbox_id=sid,app_id=res['app_id'],provider_returncode=0,provider_tags=tags,original_watchdog_exit=c.binding(P/'watchdog_exit.json'),created=c.binding(P/'created.json'),launch_plan=c.binding(P/'launch_plan.json'),authorization=plan['authorization'],retrieval_source=c.binding(Path(__file__)),rule='Preserve original failed_stop_observed receipt; independently require provider0 and complete scientific exit before retrieving. No success is inferred from guard error alone.')
 write_once(P/'terminal_reconciliation.json',reconciliation)
 with tempfile.TemporaryDirectory(prefix='reconciled-download-',dir=P) as stage:
  stage=Path(stage);names={};exit_name=run+'/exit_receipt.json';manifest_name=run+'/output_manifest.json'
  exit_digest=await t.read_remote(volume,exit_name,stage/'exit',plan['max_metadata_bytes'],budget)
  receipt=c.read(stage/'exit')
  expected=dict(schema='final_cloud_phase_exit.1',status='complete',phase='transfer_recovery',phase_nonce=auth['phase_nonce'],sandbox_id=sid,app_id=res['app_id'],authorization_sha256=auth['_sha256'],input_manifest_sha256=auth['input_manifest']['sha256'],exit_code=0,n_fits=2000,cleanup_complete=True,protocol_hash=c.TRANSFER_HASH)
  assert all(receipt.get(k)==v for k,v in expected.items())
  assert receipt['durability']==dict(volume_version=2,method='os.sync',acknowledged=True)
  assert receipt['real_process_owner']['pid_namespace']
  assert c.timestamp(receipt['started_at_utc'])<=c.timestamp(receipt['finished_at_utc'])<=c.timestamp(terminal['at_utc'])<=c.timestamp(observed)
  assert c.timestamp(receipt['finished_at_utc'])<=c.timestamp(auth['deadline_utc'])
  ref=receipt['output_manifest'];assert ref['path']==manifest_name
  await t.read_remote(volume,manifest_name,stage/'manifest',plan['max_metadata_bytes'],budget,digest=ref['sha256'])
  manifest=c.read(stage/'manifest');assert set(manifest)=={'schema','phase','files'} and manifest['schema']=='final_cloud_output_manifest.1' and manifest['phase']=='transfer_recovery'
  assert 0<len(manifest['files'])+2<=plan['max_files']
  names[exit_name]=stage/'exit';names[manifest_name]=stage/'manifest'
  for i,(name,digest) in enumerate(sorted(manifest['files'].items())):
   assert t.allowed_output(name,auth) and c.HEX.fullmatch(digest)
   target=stage/f'output-{i}'
   await t.read_remote(volume,name,target,plan['max_file_bytes'],budget,digest=digest)
   names[name]=target
  for name,temp in names.items():
   target=t.safe_path(name)
   assert not target.exists() or c.sha(target)==c.sha(temp)
  for name,temp in names.items():t.install(temp,name,ROOT)
  certificate={key:receipt[key] for key in ('phase','phase_nonce','sandbox_id','app_id','authorization_sha256','input_manifest_sha256','protocol_hash','n_fits','started_at_utc','finished_at_utc','elapsed_fit_seconds')}
  certificate.update(schema='finalization_sandbox_phase_completion.1',status='complete',runner_exit=dict(path=exit_name,sha256=exit_digest),output_manifest=ref,provider_terminal=dict(sandbox_id=sid,returncode=0,observed_at_utc=observed),terminal_reconciliation=c.binding(P/'terminal_reconciliation.json'))
  check=stage/'certificate.json';write_once(check,certificate)
  c.verify_phase_certificate(c.binding(check),'transfer_recovery')
  t.install(check,str((P/'phase_certificate.json').relative_to(ROOT)),ROOT)
  write_once(P/'download_receipt.json',dict(schema='final_cloud_reconciled_download.1',status='complete',at_utc=c.now(),bytes=budget[0],files=len(names),certificate=c.binding(P/'phase_certificate.json'),terminal_reconciliation=c.binding(P/'terminal_reconciliation.json')))
  print(json.dumps(dict(status='complete',bytes=budget[0],files=len(names),certificate=c.binding(P/'phase_certificate.json')),indent=2))
asyncio.run(asyncio.wait_for(main(),3600))
