"""Finite metadata-only cloud input planner; hashing/freezing is an explicit later step.

Never imports scientific code, reads outcome tables/arrays, stages bytes, fits,
probes processes or contacts Modal. A template is not an upload authorization.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

REPOSITORY_ROOT = '/Users/cheparukhin/ai_science_hack'
ROOT = Path(__file__).resolve().parents[1]
TRANSFER = 'reports/finalization/transfer_recovery_v2'
HYBRID = 'reports/finalization/hybrids_recovery_v2'
PORTABLE = 'reports/finalization/cloud/portable_transfer_closure.json'
CLOUD_SOURCES = ('pepstab/final_cloud_protocol.py', 'scripts/final_hybrids_cloud.py',
                 'scripts/final_hybrids_cloud_report.py', 'scripts/final_cloud_phase.py')
CELLS = tuple(f'hla/h{i}' for i in range(5)) + tuple(f'joint/h{i}_p{j}' for i in range(5) for j in range(5))
CELL_FILES = {'predictions.csv','selection.json','ancestry.json','members.json','runtime.json',
              'raw_inner_predictions.csv','raw_final_predictions.csv','member_predictions.npz'}
TRANSFER_AUTH_HASH = 'd00771c31eba4b3f4d60db72e27af7ec07dc2b842113a1dfde06c2038830465c'
TRANSFER_PROTOCOL_HASH = 'bcda56bc3ae36dd757ef1a3210b167fc926d1604be74bb7487c5e18fedaf412d'
HEX = re.compile(r'^[0-9a-f]{64}$')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def relative_path(value, root=ROOT):
    root = Path(root).resolve()
    path = Path(value)
    if path.is_absolute():
        # Relocate only the physical fixture root; logical absolute evidence stays unchanged.
        try:
            path = path.relative_to(REPOSITORY_ROOT)
        except ValueError:
            try:
                path = path.relative_to(root)
            except ValueError:
                raise ValueError('input outside exact repository root') from None
    if not path.parts or '..' in path.parts or any(p.startswith('.') for p in path.parts):
        raise ValueError('unsafe or hidden input path')
    if any(part.lower() in {'secrets','credentials'} for part in path.parts):raise ValueError('credential input refused')
    old_partial_roots=('reports/finalization/transfer_recovery/joint/',
                       'reports/finalization/transfer_controls/joint/h2_p3/',
                       'reports/finalization/transfer_controls/joint/h2_p4/',
                       'reports/finalization/transfer_controls/joint/h3_p0/')
    if path.as_posix()==f'{TRANSFER}/controller/state.json' or path.as_posix().startswith(old_partial_roots):
        raise ValueError('active state or old partial input refused')
    if path.suffix in {'.log','.lock','.pem','.key'} or path.name in {'progress.jsonl','state.json.tmp','pending.json'}:
        raise ValueError('live/log/credential input refused')
    cursor = root
    for part in path.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError('symlink input refused')
    if cursor.exists() and not cursor.is_file():
        raise ValueError('input must be an explicit file')
    return path.as_posix()


def python_import_closure(root, seeds):
    """Conservative local AST import closure, including imports inside functions.

    Only existing pepstab/scripts/modal_app modules are followed. Data paths
    are separately enumerated below; no package is imported or executed.
    """
    root = Path(root); pending = list(seeds); seen = set()
    def resolve(parts):
        if not parts or parts[0] not in ('pepstab','scripts','modal_app'):
            return []
        base = Path(*parts)
        options = [base.with_suffix('.py'), base/'__init__.py']
        parents = [Path(*parts[:i])/'__init__.py' for i in range(1,len(parts))]
        return [p.as_posix() for p in options+parents if (root/p).is_file()]
    while pending:
        name = relative_path(pending.pop(),root)
        if name in seen or not name.endswith('.py') or not (root/name).is_file():
            continue
        seen.add(name)
        parts = list(Path(name).with_suffix('').parts)
        package = parts[:-1]
        for length in range(1,len(parts)):
            init = Path(*parts[:length])/'__init__.py'
            if (root/init).is_file() and init.as_posix() not in seen:
                pending.append(init.as_posix())
        for node in ast.walk(ast.parse((root/name).read_text())):
            if isinstance(node,ast.Import):
                for alias in node.names:pending.extend(resolve(alias.name.split('.')))
            elif isinstance(node,ast.ImportFrom):
                base = package[:len(package)-node.level+1] if node.level else []
                base += node.module.split('.') if node.module else []
                pending.extend(resolve(base))
                for alias in node.names:
                    if alias.name != '*':pending.extend(resolve(base+[alias.name]))
    return sorted(seen)


def plan_inventory(root=ROOT, *, extra_files=(), portable=PORTABLE):
    root = Path(root).resolve(); entries = {}; blockers = []; deferred = []
    def add(value, role, digest=None):
        name = relative_path(value,root); path = root/name
        if digest is not None and not HEX.fullmatch(str(digest)):
            raise ValueError(f'invalid expected digest: {name}')
        item = entries.setdefault(name,dict(roles=[],expected_sha256=digest,exists=path.is_file(),
                                            size_bytes=path.stat().st_size if path.is_file() else None))
        if digest and item['expected_sha256'] not in (None,digest):raise ValueError(f'conflicting frozen hashes: {name}')
        if digest:item['expected_sha256']=digest
        if role not in item['roles']:item['roles'].append(role)
        if not path.is_file():blockers.append(f'missing required input: {name}')
        return path
    def binding(item,role):return add(item['path'],role,item['sha256'])
    tp = read(add(f'{TRANSFER}/protocol.json','transfer_protocol'))
    hp = read(add(f'{HYBRID}/protocol.json','hybrid_protocol'))
    authorization = read(add(f'{TRANSFER}/authorization.json','historical_transfer_authorization'))
    for document, key, role in ((tp,'code_sha256','frozen_transfer_source'),(hp,'source_sha256','frozen_hybrid_source'),
                                (hp,'reference_sha256','historical_descriptive_reference'),(authorization,'sources','protected_transfer_source')):
        for name,digest in document[key].items():add(name,role,digest)
    for name,key in [('data/rasmussen_et_al_dataset.csv','raw_data_sha256'),('data/splits.csv','splits_sha256'),
                     ('reports/finalization/ranking/protocol.json','ranking_protocol_sha256')]:
        add(name,'opaque_canonical_input',hp[key])
    for filename,key in [('SPECIFICATION.md','scientific_specification_sha256'),('fit_graph.json','scientific_graph_sha256')]:
        add(f'{HYBRID}/{filename}','unchanged_hybrid_design',hp['specification'][key])
    for filename in ('task_manifest.json','preflight.json','operational_amendment.json','validation.json'):
        add(f'{HYBRID}/{filename}','hybrid_manifest')
    # Pure cloud copies validate the original protocol in their separate output root.
    # Materialize these original bytes there as an explicit derivative operation,
    # recording its destination in the phase-specific manifest; no byte rewriting.
    legacy = relative_path(tp['legacy_directory']+'/assessment_protocol.json',root)
    lp = read(add(legacy,'legacy_protocol',tp['legacy_protocol_sha256'])); legacy_root = str(Path(legacy).parent)
    for name,digest in lp['code_sha256'].items():add(name,'legacy_source',digest)
    for mapping in (tp['grouping_sha256'],lp['specification']['grouping_hashes']):
        for name,digest in mapping.items():add(f'{legacy_root}/{name}','legacy_grouping',digest)
    grouping = read(root/legacy_root/'protocol.json')
    for name,digest in grouping['sha256'].items():add(f'{legacy_root}/{name}','legacy_grouping',digest)
    for cell,files in tp['legacy_evidence_sha256'].items():
        if cell not in CELLS:raise ValueError('unknown legacy cell')
        for name,digest in files.items():
            if Path(name).name != name:raise ValueError('legacy filename must be basename')
            add(f'{legacy_root}/{cell}/{name}','legacy_scientific_artifact',digest)
    provenance = hp['feature_provenance']; production = provenance['production_manifest']
    fm = read(binding({'path':production['path'],'sha256':production['sha256']},'compact_feature_manifest'))
    declared = fm['feature_bundles']
    bundle_paths = [relative_path((b if isinstance(b,str) else b['path'])+'/complete.json',root) for b in declared]
    if set(bundle_paths) != {relative_path(b['path']+'/complete.json',root) for b in provenance['bundles']}:
        raise ValueError('compact bundle manifest/protocol coverage differs')
    for bundle in provenance['bundles']:
        complete_path = add(bundle['path']+'/complete.json','compact_feature_completion')
        complete = read(complete_path)
        for field,digestkey in [('array_file','npz_sha256'),('metadata_file','metadata_sha256')]:
            name = complete[field]
            if Path(name).name != name or complete[digestkey] != bundle[digestkey]:raise ValueError('compact feature identity changed')
            add(str(complete_path.parent/name),'compact_feature_payload',bundle[digestkey])
    add('reports/further_research/round2/representations/contact_map.json','compact_contact_map')
    for key in ('closure','local_failure_closure','local_failure_exit','local_failure_state','preserved_audit','preparation','runtime_forecast'):
        binding(authorization[key],'historical_transfer_provenance')
    add(f'{TRANSFER}/controller/attempt.json','historical_transfer_attempt')
    completed_path = add(f'{TRANSFER}/controller/completed.json','successful_transfer_closure')
    exit_path = add(f'{TRANSFER}/controller/exit_receipt.json','successful_transfer_closure')
    completion = read(completed_path) if completed_path.is_file() else {}
    receipt = read(exit_path) if exit_path.is_file() else {}
    expected_completion=dict(schema='final_transfer_durable_completion.1',status='complete',pid=59308,
        protocol_hash=TRANSFER_PROTOCOL_HASH,authorization_sha256=TRANSFER_AUTH_HASH,
        n_retained_fits=10830,n_new_fits=4800,discarded_partial_completed_fits=1785,
        known_completed_including_discarded=12615,deadline_utc='2026-10-13T15:32:19.924192+00:00')
    if any(completion.get(k)!=v for k,v in expected_completion.items()):
        blockers.append('exact durable completion identity/count/authorization is unavailable')
    if completed_path.is_file() and receipt.get('completed_sha256')!=sha(completed_path):
        blockers.append('successful exit does not bind exact completed certificate')
    if (completion.get('schema')!='final_transfer_durable_completion.1' or completion.get('status')!='complete'
            or set(completion.get('completed_cell_hashes',{}))!=set(CELLS)
            or receipt.get('schema')!='final_transfer_durable_exit.1' or receipt.get('status')!='complete' or receipt.get('exit_code')!=0):
        blockers.append('exact successful thirty-cell transfer closure not yet available')
    for cell in CELLS:
        path = root/TRANSFER/cell/'completed.json'
        if not path.is_file():
            deferred.extend(f'{TRANSFER}/{cell}/{name}' for name in sorted(CELL_FILES|{'completed.json'}))
            continue
        done = read(path)
        regime,cell_name=cell.split('/')
        expected_cell=dict(status='complete',regime=regime,cell=cell_name,protocol_hash=TRANSFER_PROTOCOL_HASH,n_fits=166 if regime=='hla' else 400)
        if any(done.get(k)!=v for k,v in expected_cell.items()) or set(done.get('sha256',{}))!=CELL_FILES:
            blockers.append(f'incomplete cell receipt: {cell}');continue
        add(str(path),'complete_transfer_artifact',completion.get('completed_cell_hashes',{}).get(cell))
        for name,digest in done['sha256'].items():add(f'{TRANSFER}/{cell}/{name}','complete_transfer_artifact',digest)
        if cell in CELLS[18:]:
            runtime = read(root/TRANSFER/cell/'runtime.json'); cache = runtime.get('checkpoint_cache',{})
            calls = cache.get('calls',{})
            expected_path = f'{REPOSITORY_ROOT}/{TRANSFER}/checkpoints/{cell}'
            if cache.get('path')!=expected_path or len(calls)!=104 or cache.get('n_calls')!=104 or cache.get('replayed_calls')!=0:
                raise ValueError('new cell checkpoint proof differs from fixed104-call scope')
            for index,(call,files) in enumerate(sorted(calls.items())):
                if call!=f'call_{index:04d}' or set(files)!={'call.json','arrays.npz','completed.json'}:raise ValueError('checkpoint coverage changed')
                for name,digest in files.items():add(f'{TRANSFER}/checkpoints/{cell}/{call}/{name}','completed_checkpoint',digest)
            add(f'{TRANSFER}/{cell}/exit_receipt.json','checkpoint_worker_exit')
    for cell in CELLS[18:]:
        if not (root/TRANSFER/cell/'completed.json').is_file():
            deferred.append(dict(checkpoint_root=f'{TRANSFER}/checkpoints/{cell}',exact_calls=104,files_each=['call.json','arrays.npz','completed.json'],admit_only_after_cell_completion=True))
    certificate_path = add(portable,'portable_transfer_closure')
    if certificate_path.is_file():
        certificate = read(certificate_path)
        if certificate.get('schema')!='finalization_portable_transfer_closure.1' or certificate.get('status')!='complete':
            blockers.append('portable closure is not successful')
        for key in ('authorization','attempt','completion','exit'):binding(certificate[key],'portable_bound_provenance')
        for name,digest in certificate.get('evidence_files',{}).items():add(name,'portable_extra_evidence',digest)
    else: blockers.append('root-authenticated portable transfer closure is not yet available')
    seeds = ['scripts/final_transfer_report.py','scripts/final_hybrids.py','scripts/final_hybrids_report.py',*CLOUD_SOURCES]
    for name in CLOUD_SOURCES:add(name,'new_cloud_operational_source')
    for name in python_import_closure(root,seeds):add(name,'runtime_import')
    for name in extra_files:
        path=add(name,'explicit_phase_extra')
        if path.is_file() and path.suffix=='.json':
            document=read(path)
            schema=document.get('schema') if isinstance(document,dict) else None
            if schema in {'final_cloud_phase_authorization.1','final_cloud_input_manifest.1'}:
                raise ValueError('phase authorization/input manifest cannot include itself through extras')
    if deferred:blockers.append('remaining transfer artifacts/checkpoints deferred until successful completion')
    groups = {}
    for name,item in entries.items():
        role=item['roles'][0];group=groups.setdefault(role,dict(files=0,existing_bytes=0,missing=0))
        group['files']+=1;group['existing_bytes']+=item['size_bytes'] or 0;group['missing']+=not item['exists']
    return dict(schema='final_cloud_input_inventory.1',status='blocked_template' if blockers else 'ready_for_hash_freeze',
        repository_root=REPOSITORY_ROOT,at_utc=datetime.now(timezone.utc).isoformat(),
        scope='unchanged transfer report and four-phase hybrid cloud continuation',
        files=dict(sorted(entries.items())),deferred=deferred,blockers=sorted(set(blockers)),summary=groups,
        existing_size_bytes=sum(item['size_bytes'] or 0 for item in entries.values()),
        protected_source_count=len(authorization['sources']),compact_feature_bundles=len(provenance['bundles']),
        launch_authorized_by_this_record=False)


def freeze_manifest(root=ROOT, *, extra_files=(), portable=PORTABLE):
    """Explicit content hashing only after successful closure; no copy/upload/fit."""
    root=Path(root).resolve();plan=plan_inventory(root,extra_files=extra_files,portable=portable)
    if plan['blockers']:raise ValueError('input freeze blocked: '+'; '.join(plan['blockers']))
    files={}
    for name,item in plan['files'].items():
        digest=sha(root/name)
        if item['expected_sha256'] is not None and digest!=item['expected_sha256']:
            raise ValueError(f'frozen input content changed: {name}')
        files[name]=digest
    return dict(schema='final_cloud_input_manifest.1',repository_root=REPOSITORY_ROOT,files=files)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['plan','freeze'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--extra-file',action='append',default=[])
    parser.add_argument('--portable',default=PORTABLE)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('refusing existing inventory/manifest output')
    value=(plan_inventory if args.mode=='plan' else freeze_manifest)(extra_files=args.extra_file,portable=args.portable)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n')
    print(json.dumps({key:value[key] for key in ('schema','status','existing_size_bytes','blockers') if key in value},indent=2))


if __name__=='__main__':main()
