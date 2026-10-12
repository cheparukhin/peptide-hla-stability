"""Pure, fail-closed contracts for the four bounded cloud finalization phases.

No Modal API, process probing, fitting, or historical-runtime substitution lives
here. A certificate is trusted only through a binding in the phase authorization.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = '/Users/cheparukhin/ai_science_hack'
SCOPE = 'reports/finalization/hybrids_cloud_v1'
TRANSFER_ROOT = 'reports/finalization/transfer_recovery_v2'
CLOUD_TRANSFER_ROOT = 'reports/finalization/transfer_cloud_recovery'
RECOVERY_CELLS = tuple(f'joint/h4_p{i}' for i in range(5))
SCIENCE_HASH = 'c5fd5501d8b640428305a1eb5b2d9fd83c3fb47139faa7de055550cb02de3955'
TRANSFER_HASH = 'bcda56bc3ae36dd757ef1a3210b167fc926d1604be74bb7487c5e18fedaf412d'
TRANSFER_AUTH_HASH = 'd00771c31eba4b3f4d60db72e27af7ec07dc2b842113a1dfde06c2038830465c'
TRANSFER_DEADLINE = '2026-10-13T15:32:19.924192+00:00'
OUTER_DEADLINE = '2026-10-14T14:15:17.567539+00:00'
PHASES = {'transfer_recovery': (4,16384,2,21600,10,2000),
          'transfer_report': (2,8192,1,28800,5,0),
          'hybrid_pilot': (2,8192,1,7200,2,12),
          'hybrid_main': (4,16384,2,43200,20,480),
          'hybrid_report': (2,8192,1,28800,5,0)}
NEW_SOURCES = ('pepstab/final_cloud_protocol.py','scripts/final_hybrids_cloud.py',
               'scripts/final_hybrids_cloud_report.py','scripts/final_cloud_phase.py','scripts/final_sandbox_driver.py')
CELLS = tuple(f'hla/h{i}' for i in range(5)) + tuple(f'joint/h{i}_p{j}' for i in range(5) for j in range(5))
HEX = re.compile(r'^[0-9a-f]{64}$')


def now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError('timezone-aware timestamp required')
    return parsed.timestamp()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    def reject(value):
        raise ValueError(f'non-finite JSON constant: {value}')
    def unique(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('duplicate JSON key')
            out[key] = value
        return out
    return json.loads(Path(path).read_text(), parse_constant=reject, object_pairs_hook=unique)


def path_in_root(value, root=ROOT):
    root = Path(root).resolve()
    given = Path(value)
    # Absolute evidence paths retain the original repository namespace in tests.
    if given.is_absolute():
        try:
            relative = given.relative_to(REPOSITORY_ROOT)
        except ValueError:
            raise ValueError('path leaves original repository') from None
    else:
        relative = given
    if '..' in relative.parts or not relative.parts:
        raise ValueError('path traversal refused')
    path = root / relative
    cursor = path
    while cursor != root:
        if cursor.is_symlink():
            raise ValueError('symlink evidence refused')
        cursor = cursor.parent
    if not path.resolve().is_relative_to(root):
        raise ValueError('path leaves repository')
    return path


def bound(binding, root=ROOT):
    if not isinstance(binding, dict) or set(binding) != {'path','sha256'} or not HEX.fullmatch(str(binding['sha256'])):
        raise ValueError('exact path/SHA256 binding required')
    path = path_in_root(binding['path'], root)
    if sha(path) != binding['sha256']:
        raise ValueError(f'bound artifact changed: {binding["path"]}')
    return path


def binding(path, root=ROOT):
    path = Path(path)
    return {'path': str(path.resolve().relative_to(Path(root).resolve())), 'sha256': sha(path)}


def finite_number(value, *, positive=False):
    if type(value) not in (int,float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError('finite nonnegative numeric resource/accounting value required')
    return value


def verify_files(files, root=ROOT):
    if not isinstance(files, dict) or not files:
        raise ValueError('nonempty file manifest required')
    for name, digest in files.items():
        if Path(name).is_absolute():
            raise ValueError('manifest entries must be repository relative')
        bound({'path':name,'sha256':digest}, root)


def verify_input_manifest(reference, root=ROOT):
    item = read(bound(reference, root))
    if item.get('schema') != 'final_cloud_input_manifest.1' or item.get('repository_root') != REPOSITORY_ROOT:
        raise ValueError('input manifest namespace/schema changed')
    verify_files(item.get('files'), root)
    return item


def verify_transfer_cells(scope,hashes,expected_cells,root=ROOT):
    if set(hashes)!=set(expected_cells):raise ValueError('transfer cell coverage differs')
    total=0
    expected_files={'predictions.csv','selection.json','ancestry.json','members.json','runtime.json','raw_inner_predictions.csv','raw_final_predictions.csv','member_predictions.npz'}
    for cell,digest in hashes.items():
        done=read(bound({'path':f'{scope}/{cell}/completed.json','sha256':digest},root))
        regime,name=cell.split('/');count=166 if regime=='hla' else 400
        if any(done.get(k)!=v for k,v in dict(status='complete',regime=regime,cell=name,protocol_hash=TRANSFER_HASH,n_fits=count).items()) or set(done.get('sha256',{}))!=expected_files:
            raise ValueError('retained transfer cell identity/count changed')
        for name,digest in done['sha256'].items():bound({'path':f'{scope}/{cell}/{name}','sha256':digest},root)
        total+=count
    return total


def verify_transfer_failure(reference,root=ROOT):
    """A failed owner with 25 retained cells authorizes only five fresh cells."""
    item=read(bound(reference,root))
    expected=dict(schema='finalization_portable_transfer_failure.1',status='failed_scope_closed',
        preserved_fits=8830,new_discarded_completed_fits=794,discarded_partial_completed_fits=2579,
        remaining_cells=list(RECOVERY_CELLS))
    if any(item.get(k)!=v for k,v in expected.items()):raise ValueError('exact failed recovery closure/counts required')
    owner=item.get('owner',{})
    if (owner.get('pid')!=59308 or owner.get('absent') is not True or owner.get('children_absent') is not True
            or owner.get('launchd_label')!='org.pepstab.transfer-recovery-v2-20261012'
            or any(not owner.get(k) for k in ('host','boot_id','command','started_at_utc'))):
        raise ValueError('failed Mac owner and child absence required')
    old=read(bound(item['authorization'],root));attempt=read(bound(item['attempt'],root))
    failed=read(bound(item['exit'],root));state=read(bound(item['state'],root))
    if item['authorization']['sha256']!=TRANSFER_AUTH_HASH or old.get('approved') is not True:
        raise ValueError('original transfer authorization changed')
    for document in (failed,state):
        if any(document.get(k)!=v for k,v in dict(status='failed',pid=59308,protocol_hash=TRANSFER_HASH,
            authorization_sha256=TRANSFER_AUTH_HASH,deadline_utc=TRANSFER_DEADLINE,cleanup_error=None).items()):
            raise ValueError('truthful failed transfer exit/state required')
    if failed.get('schema')!='final_transfer_durable_exit.1' or failed.get('exit_code')!=1:
        raise ValueError('failed exact owner exit required')
    if attempt.get('pid')!=59308 or attempt.get('authorization_sha256')!=TRANSFER_AUTH_HASH:
        raise ValueError('exact failed transfer attempt required')
    if not (timestamp(owner['started_at_utc'])<=timestamp(attempt['started_at_utc'])<=timestamp(owner['started_at_utc'])+300
            and timestamp(attempt['started_at_utc'])<=timestamp(failed['finished_at_utc'])<=timestamp(TRANSFER_DEADLINE)
            and timestamp(failed['finished_at_utc'])<=timestamp(item['checked_at_utc'])):
        raise ValueError('failed owner closure chronology changed')
    verify_files(old['sources'],root);verify_files(item['evidence_files'],root)
    if verify_transfer_cells(TRANSFER_ROOT,item['preserved_cell_hashes'],CELLS[:25],root)!=8830:
        raise ValueError('exact preserved count required')
    # New root receives complete cell files only; partial checkpoints never enter it.
    verify_transfer_cells(CLOUD_TRANSFER_ROOT,item['preserved_cell_hashes'],CELLS[:25],root)
    if sha(path_in_root(f'{CLOUD_TRANSFER_ROOT}/protocol.json',root))!=sha(path_in_root(f'{TRANSFER_ROOT}/protocol.json',root)):
        raise ValueError('original transfer scientific protocol changed')
    return item


def verify_cloud_transfer_closure(reference,root=ROOT):
    item=read(bound(reference,root))
    expected=dict(schema='finalization_portable_cloud_transfer_closure.1',status='complete',scope_root=CLOUD_TRANSFER_ROOT,
        protocol_hash=TRANSFER_HASH,retained_fits=10830,preserved_fits=8830,new_fits=2000,
        discarded_partial_completed_fits=2579,known_completed_including_discarded=13409,unknown_remote_fits=12,unknown_local_fits=True)
    if any(item.get(k)!=v for k,v in expected.items()):raise ValueError('exact completed cloud transfer closure required')
    previous=verify_transfer_failure(item['failed_predecessor'],root)
    if item['authorization']!=previous['authorization']:raise ValueError('original failed owner provenance changed')
    recovery=read(bound(item['recovery_authorization'],root))
    certificate=verify_phase_certificate(item['phase_certificate'],'transfer_recovery',root)
    if (certificate['authorization_sha256']!=item['recovery_authorization']['sha256']
            or recovery.get('portable_predecessor')!=item['failed_predecessor'] or recovery.get('cells')!=list(RECOVERY_CELLS)
            or recovery.get('scope_root')!=CLOUD_TRANSFER_ROOT or recovery.get('n_fits')!=2000):
        raise ValueError('cloud recovery authorization/phase linkage changed')
    if any(item['completed_cell_hashes'].get(cell)!=digest for cell,digest in previous['preserved_cell_hashes'].items()):
        raise ValueError('preserved cells changed during cloud recovery')
    if verify_transfer_cells(CLOUD_TRANSFER_ROOT,item['completed_cell_hashes'],CELLS,root)!=10830:
        raise ValueError('all thirty completed transfer cells required')
    aggregate=read(path_in_root(f'{CLOUD_TRANSFER_ROOT}/recovery_completion.json',root))
    if aggregate.get('completed_cell_hashes')!=item['completed_cell_hashes']:
        raise ValueError('cloud aggregate cell linkage changed')
    return item


def verify_predecessor(reference, root=ROOT):
    """Authenticate Mac closure and hash-check its already-audited evidence.

    Certificate fields: schema/status; authorization/attempt/completion/exit
    bindings; checked_at_utc; owner {pid,host,boot_id,launchd_label,command,
    started_at_utc,absent,children_absent}; evidence_files {path:hash}. The
    completion itself binds all 30 cells and their original historical audits.
    """
    item = read(bound(reference, root))
    if item.get('schema') == 'finalization_portable_cloud_transfer_closure.1':
        return verify_cloud_transfer_closure(reference,root)
    if item.get('schema') != 'finalization_portable_transfer_closure.1' or item.get('status') != 'complete':
        raise ValueError('successful portable transfer closure required')
    owner = item.get('owner', {})
    if (owner.get('pid') != 59308 or owner.get('absent') is not True or owner.get('children_absent') is not True
            or owner.get('launchd_label') != 'org.pepstab.transfer-recovery-v2-20261012'
            or any(not owner.get(k) for k in ('host','boot_id','command','started_at_utc'))):
        raise ValueError('exact Mac owner absence evidence required')
    authorization = read(bound(item['authorization'], root))
    if item['authorization']['sha256'] != TRANSFER_AUTH_HASH or authorization.get('approved') is not True:
        raise ValueError('exact original transfer authorization required')
    attempt = read(bound(item['attempt'], root))
    complete = read(bound(item['completion'], root))
    receipt = read(bound(item['exit'], root))
    expected = dict(schema='final_transfer_durable_completion.1',status='complete',pid=59308,
        protocol_hash=TRANSFER_HASH,deadline_utc=TRANSFER_DEADLINE,n_retained_fits=10830,n_new_fits=4800,
        discarded_partial_completed_fits=1785,known_completed_including_discarded=12615,
        unknown_remote_fits=12,unknown_local_fits=True,authorization_sha256=TRANSFER_AUTH_HASH)
    if any(complete.get(k) != v for k,v in expected.items()):
        raise ValueError('transfer completion identity/count mismatch')
    if (attempt.get('pid') != 59308 or attempt.get('authorization_sha256') != TRANSFER_AUTH_HASH
            or receipt.get('schema') != 'final_transfer_durable_exit.1' or receipt.get('status') != 'complete'
            or receipt.get('pid') != 59308 or receipt.get('exit_code') != 0
            or receipt.get('authorization_sha256') != TRANSFER_AUTH_HASH
            or receipt.get('completed_sha256') != item['completion']['sha256']):
        raise ValueError('exact transfer attempt/successful exit required')
    if not (timestamp(owner['started_at_utc']) <= timestamp(attempt['started_at_utc']) <= timestamp(owner['started_at_utc'])+300):
        raise ValueError('Mac owner start does not match exact transfer attempt')
    if not (timestamp(attempt['started_at_utc']) <= timestamp(complete['finished_at_utc']) <=
            timestamp(receipt['finished_at_utc']) <= timestamp(TRANSFER_DEADLINE)
            and timestamp(receipt['finished_at_utc']) <= timestamp(item['checked_at_utc'])):
        raise ValueError('transfer closure chronology/deadline changed')
    for key in ('closure','local_failure_closure','local_failure_exit','local_failure_state','preserved_audit'):
        if complete.get(key) != authorization.get(key):
            raise ValueError('historical transfer provenance mismatch')
        bound(authorization[key], root)
    verify_files(authorization['sources'], root)
    verify_files(item.get('evidence_files'), root)
    if (set(complete.get('reused_cell_hashes',{})) != set(CELLS[:18])
            or set(complete.get('new_cell_hashes',{})) != set(CELLS[18:])
            or {**complete['reused_cell_hashes'],**complete['new_cell_hashes']} != complete['completed_cell_hashes']
            or complete['reused_cell_hashes'] != authorization.get('reused_cell_hashes')):
        raise ValueError('preserved versus new transfer cell linkage changed')
    expected_root = f'{REPOSITORY_ROOT}/{TRANSFER_ROOT}'
    if complete.get('scope_root') != expected_root or set(complete['completed_cell_hashes']) != set(CELLS):
        raise ValueError('all thirty original cells required')
    for cell, digest in complete['completed_cell_hashes'].items():
        directory = path_in_root(f'{TRANSFER_ROOT}/{cell}', root)
        done = read(bound({'path':f'{TRANSFER_ROOT}/{cell}/completed.json','sha256':digest}, root))
        regime, name = cell.split('/')
        if any(done.get(k) != v for k,v in dict(status='complete',protocol_hash=TRANSFER_HASH,regime=regime,cell=name,n_fits=166 if regime=='hla' else 400).items()):
            raise ValueError('transfer cell identity/count changed')
        expected_files={'predictions.csv','selection.json','ancestry.json','members.json','runtime.json','raw_inner_predictions.csv','raw_final_predictions.csv','member_predictions.npz'}
        if set(done.get('sha256',{})) != expected_files:
            raise ValueError('exact transfer artifact set required')
        for filename,h in done['sha256'].items():
            bound({'path':f'{TRANSFER_ROOT}/{cell}/{filename}','sha256':h}, root)
        if cell in CELLS[18:]:
            runtime = read(directory/'runtime.json')
            cache = runtime.get('checkpoint_cache', {})
            expected_cache = f'{REPOSITORY_ROOT}/{TRANSFER_ROOT}/checkpoints/{cell}'
            calls = cache.get('calls', {})
            if cache.get('path') != expected_cache or cache.get('n_calls') != 104 or len(calls) != 104 or cache.get('replayed_calls') != 0:
                raise ValueError('checkpoint cache provenance changed')
            cache_directory=path_in_root(f'{TRANSFER_ROOT}/checkpoints/{cell}',root)
            if {p.name for p in cache_directory.iterdir()} != set(calls):
                raise ValueError('unexpected checkpoint transaction')
            for index,(call,files) in enumerate(sorted(calls.items())):
                if call != f'call_{index:04d}' or set(files) != {'call.json','arrays.npz','completed.json'}:
                    raise ValueError('checkpoint inventory changed')
                call_directory=path_in_root(f'{TRANSFER_ROOT}/checkpoints/{cell}/{call}',root)
                if {p.name for p in call_directory.iterdir()} != set(files):
                    raise ValueError('pending or unexpected checkpoint file')
                for filename,h in files.items():
                    bound({'path':f'{TRANSFER_ROOT}/checkpoints/{cell}/{call}/{filename}','sha256':h}, root)
                record = read(path_in_root(f'{TRANSFER_ROOT}/checkpoints/{cell}/{call}/call.json',root))
                if record['identity'].get('versions') != dict(python=runtime['python'],packages=runtime['versions']):
                    raise ValueError('historical cache runtime identity mismatch')
    return item


def verify_phase_certificate(reference, phase, root=ROOT):
    item = read(bound(reference, root))
    count = PHASES[phase][-1]
    expected = dict(schema='finalization_sandbox_phase_completion.1',status='complete',phase=phase,
                    protocol_hash=TRANSFER_HASH if phase in ('transfer_report','transfer_recovery') else SCIENCE_HASH,n_fits=count)
    if any(item.get(k) != v for k,v in expected.items()):
        raise ValueError('exact successful cloud phase certificate required')
    terminal = item.get('provider_terminal', {})
    if (not str(item.get('sandbox_id','')).startswith('sb-') or not item.get('app_id')
            or terminal.get('sandbox_id') != item['sandbox_id'] or type(terminal.get('returncode')) is not int or terminal.get('returncode') != 0
            or timestamp(terminal['observed_at_utc']) < timestamp(item['finished_at_utc'])):
        raise ValueError('actual successful provider terminal observation required')
    receipt = read(bound(item['runner_exit'], root))
    shared = ('phase','phase_nonce','sandbox_id','authorization_sha256','input_manifest_sha256',
              'protocol_hash','n_fits','started_at_utc','finished_at_utc','elapsed_fit_seconds')
    if (receipt.get('schema') != 'final_cloud_phase_exit.1' or receipt.get('status') != 'complete'
            or receipt.get('exit_code') != 0 or receipt.get('cleanup_complete') is not True
            or any(receipt.get(k) != item.get(k) for k in shared)
            or not receipt.get('real_process_owner',{}).get('pid_namespace')):
        raise ValueError('phase receipt/real namespace mismatch')
    if receipt.get('output_manifest') != item['output_manifest']:
        raise ValueError('phase output manifest linkage changed')
    manifest = read(bound(item['output_manifest'],root))
    if manifest.get('schema') != 'final_cloud_output_manifest.1' or manifest.get('phase') != phase:
        raise ValueError('output manifest phase mismatch')
    verify_files(manifest.get('files'), root)
    if phase == 'hybrid_pilot':
        audits = [p for p in manifest['files'] if p.endswith('/pilot_audit.json')]
        if len(audits) != 1:
            raise ValueError('pilot actual saved-model replay audit required')
        audit = read(path_in_root(audits[0],root))
        if any(audit.get(k) != v for k,v in dict(schema='final_cloud_pilot_audit.1',status='passed',n_fits=12,held_scores_computed=False,reuse_into_scientific=False,saved_model_predictions_replayed=True,protocol_hash=SCIENCE_HASH,authorization_sha256=item['authorization_sha256']).items()):
            raise ValueError('pilot unscored saved-model audit changed')
    if phase=='transfer_recovery':
        name=f'{CLOUD_TRANSFER_ROOT}/recovery_completion.json'
        if name not in manifest['files']:raise ValueError('cloud recovery semantic audit missing')
        aggregate=read(path_in_root(name,root))
        expected_aggregate=dict(schema='final_transfer_cloud_recovery_completion.1',status='complete',protocol_hash=TRANSFER_HASH,
            n_retained_fits=10830,n_new_fits=2000,preserved_fits=8830,discarded_partial_completed_fits=2579,
            known_completed_including_discarded=13409,semantic_audit_passed=True,authorization_sha256=item['authorization_sha256'])
        if any(aggregate.get(k)!=v for k,v in expected_aggregate.items()):raise ValueError('cloud recovery audit/counts changed')
        verify_transfer_cells(CLOUD_TRANSFER_ROOT,aggregate['completed_cell_hashes'],CELLS,root)
        if timestamp(item['finished_at_utc'])>timestamp(TRANSFER_DEADLINE):raise ValueError('original transfer deadline exceeded')
    finite_number(item['elapsed_fit_seconds'])
    if not timestamp(item['started_at_utc']) <= timestamp(item['finished_at_utc']) <= timestamp(OUTER_DEADLINE):
        raise ValueError('phase chronology exceeds fixed expiry')
    return item


def validate_authorization(path, phase=None, *, root=ROOT, clock=None, verify_inputs=True):
    path = path_in_root(path,root)
    auth = read(path)
    phase = phase or auth.get('phase')
    expected_scope=CLOUD_TRANSFER_ROOT if phase=='transfer_recovery' else auth.get('scope_root') if phase=='transfer_report' and auth.get('scope_root')==CLOUD_TRANSFER_ROOT else SCOPE
    if phase not in PHASES or auth.get('schema') != 'final_cloud_phase_authorization.1' or auth.get('approved') is not True or auth.get('phase') != phase or auth.get('scope_root') != expected_scope:
        raise ValueError('approved exact cloud phase/scope required')
    cpu,memory,workers,max_wall,max_cap,_ = PHASES[phase]
    resources = dict(cpu_cores=cpu,memory_mib=memory,numerical_threads=2,max_workers=workers)
    if (auth.get('resources') != resources or any(type(auth['resources'].get(k)) is not int for k in resources)):
        raise ValueError('phase resources differ from fixed caps')
    finite_number(auth['fit_wall_seconds'],positive=True)
    finite_number(auth['phase_cap_usd'],positive=True)
    if type(auth['fit_wall_seconds']) is not int or auth['fit_wall_seconds'] > max_wall or auth['phase_cap_usd'] != max_cap:
        raise ValueError('phase wall/budget cap exceeded')
    current = datetime.now(timezone.utc).timestamp() if clock is None else clock()
    phase_deadline=TRANSFER_DEADLINE if phase=='transfer_recovery' else OUTER_DEADLINE
    if not current < timestamp(auth['deadline_utc']) <= timestamp(phase_deadline):
        raise ValueError('cloud deadline expired or extended')
    if (auth.get('run_directory') != f"reports/finalization/cloud/remote/{auth.get('phase_nonce')}"
            or not re.fullmatch(r'[A-Za-z0-9_-]{12,128}',str(auth.get('phase_nonce','')))
            or not auth.get('sandbox_name') or not str(auth.get('image_id','')).startswith('im-')
            or auth.get('durability') != {'volume_version':2,'method':'os.sync'}):
        raise ValueError('unique phase/image/durability identity required')
    sources = auth.get('sources',{})
    if not set(NEW_SOURCES).issubset(sources):
        raise ValueError('all cloud execution sources must be bound')
    verify_files(sources,root)
    manifest = verify_input_manifest(auth['input_manifest'],root) if verify_inputs else read(bound(auth['input_manifest'],root))
    if any(manifest['files'].get(k) != v for k,v in sources.items()):
        raise ValueError('executed source absent from immutable input manifest')
    predecessor = verify_transfer_failure(auth['portable_predecessor'],root) if phase=='transfer_recovery' else verify_predecessor(auth['portable_predecessor'],root)
    if phase=='transfer_recovery':
        if any(auth.get(k)!=v for k,v in dict(protocol_hash=TRANSFER_HASH,cells=list(RECOVERY_CELLS),ranker_backend='local',n_fits=2000).items()):
            raise ValueError('exact five-cell original scientific authorization required')
    if phase=='transfer_report' and (expected_scope==CLOUD_TRANSFER_ROOT)!=(predecessor.get('schema')=='finalization_portable_cloud_transfer_closure.1'):
        raise ValueError('report root differs from exact completed transfer predecessor')
    historical_auth = read(bound(predecessor['authorization'],root))
    if any(sources.get(k) != h for k,h in historical_auth['sources'].items()):
        raise ValueError('all historical transfer sources must be execution-bound')
    if phase not in ('transfer_report','transfer_recovery'):
        original_protocol = read(path_in_root('reports/finalization/hybrids_recovery_v2/protocol.json',root))
        staged_protocol = path_in_root(f'{SCOPE}/protocol.json',root)
        if (sha(staged_protocol) != sha(path_in_root('reports/finalization/hybrids_recovery_v2/protocol.json',root))
                or original_protocol.get('protocol_hash') != SCIENCE_HASH
                or any(sources.get(k) != h for k,h in original_protocol['source_sha256'].items())):
            raise ValueError('cloud scientific protocol/source bytes changed')
    account = read(bound(auth['cloud_accounting'],root))
    if account.get('schema') != 'final_cloud_accounting.1' or account.get('phase') != phase:
        raise ValueError('phase accounting schema/identity changed')
    for key in ('prior_conservative_liability_usd','future_envelope_usd','total_ceiling_usd','phase_cap_usd','reserved_phase_usd','conservative_spend_usd'):
        finite_number(account[key])
    other=account.get('other_scopes_reserved_usd',0)
    finite_number(other)
    if phase=='transfer_recovery' and (account['future_envelope_usd']!=10 or other!=35):
        raise ValueError('recovery10 and future35 reservations must remain separate')
    if (account['prior_conservative_liability_usd'] < 5.1793820030564595 or account['future_envelope_usd'] > 35
            or account['total_ceiling_usd'] > 100 or account['prior_conservative_liability_usd'] + account['future_envelope_usd'] + other > account['total_ceiling_usd']
            or account['phase_cap_usd'] != auth['phase_cap_usd'] or account['reserved_phase_usd'] < auth['phase_cap_usd']
            or account['conservative_spend_usd'] + account['reserved_phase_usd'] > account['future_envelope_usd']):
        raise ValueError('accounting under-reserved or exceeds authorized ceiling')
    certificates = auth.get('phase_certificates',{})
    required = () if phase in ('transfer_report','transfer_recovery','hybrid_pilot') else ('hybrid_pilot',) if phase=='hybrid_main' else ('hybrid_pilot','hybrid_main')
    if set(certificates) != set(required):
        raise ValueError('exact preceding phase certificates required')
    checked = {key:verify_phase_certificate(certificates[key],key,root) for key in required}
    if phase == 'hybrid_main':
        forecast = read(bound(auth['runtime_forecast'],root))
        if (forecast.get('schema') != 'final_cloud_hybrid_runtime_forecast.1' or forecast.get('held_scores_used') is not False
                or forecast.get('pilot_certificate_sha256') != certificates['hybrid_pilot']['sha256']
                or forecast.get('fit_wall_seconds') != auth['fit_wall_seconds'] or forecast.get('max_workers') != workers
                or forecast.get('saved_model_replay_passed') is not True):
            raise ValueError('separate unscored successful pilot runtime gate required')
    return dict(auth,_sha256=sha(path),_path=str(path),_predecessor=predecessor,_phase_certificates=checked,_accounting=account)


def validate_release(release, auth, sandbox_id):
    if (release.get('schema') != 'final_cloud_release.1' or release.get('phase_nonce') != auth['phase_nonce']
            or release.get('authorization_sha256') != auth['_sha256'] or release.get('sandbox_id') != sandbox_id
            or release.get('watchdog_ready') is not True or not str(sandbox_id).startswith('sb-')):
        raise ValueError('exact watchdog-ready sandbox release required')
    return release


def process_owner():
    namespace = Path('/proc/self/ns/pid')
    return dict(pid=os.getpid(),hostname=socket.gethostname(),
                pid_namespace=os.readlink(namespace) if namespace.exists() else f'host:{socket.gethostname()}')


def durable_sync():
    # The launch contract requires a v2 Volume; this is never an API retry.
    os.sync()


def exclusive_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:
        json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n');handle.flush();os.fsync(handle.fileno())


def scientific_authorization(out,approved,phase):
    from scripts.final_hybrids import validate_frozen
    if Path(out).resolve() != ROOT/SCOPE:
        raise ValueError('cloud output scope mismatch')
    auth=validate_authorization(approved, f'hybrid_{phase}')
    frame,protocol=validate_frozen(out)
    if protocol['protocol_hash'] != SCIENCE_HASH or any(auth['sources'].get(k) != h for k,h in protocol['source_sha256'].items()):
        raise ValueError('unchanged scientific source set must be execution-bound')
    if os.environ.get('FINAL_CLOUD_AUTH_SHA256') != auth['_sha256'] or not os.environ.get('FINAL_CLOUD_SANDBOX_ID','').startswith('sb-'):
        raise ValueError('worker requires released supervisor context')
    return frame,protocol,auth


def report_phase_receipts(out,approved):
    if Path(out).resolve() != ROOT/SCOPE:
        raise ValueError('report output scope mismatch')
    auth=validate_authorization(approved,'hybrid_report')
    if os.environ.get('FINAL_CLOUD_AUTH_SHA256') != auth['_sha256'] or not os.environ.get('FINAL_CLOUD_SANDBOX_ID','').startswith('sb-'):
        raise ValueError('report requires released supervisor context')
    return auth,{key.removeprefix('hybrid_'):dict(path=value['runner_exit']['path'],
        sha256=value['runner_exit']['sha256'],elapsed_fit_seconds=value['elapsed_fit_seconds'],
        started_at_utc=value['started_at_utc'],finished_at_utc=value['finished_at_utc'],
        sandbox_id=value['sandbox_id'],backend='modal_cpu_sandbox') for key,value in auth['_phase_certificates'].items()}
