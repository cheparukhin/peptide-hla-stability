"""Small synthetic files only; no imports of fitting code, payload arrays or Modal."""
from pathlib import Path
import pytest
from scripts import final_cloud_inputs as inputs


@pytest.mark.parametrize('path',['../escape','.git/config','.venv/bin/python','x/../../escape',
    '/Users/another/secrets.json','secrets/token.json','x/progress.jsonl','x/output.log',
    inputs.TRANSFER+'/controller/state.json','reports/finalization/transfer_recovery/joint/h2_p3/attempt.json','x/pending.json'])
def test_unsafe_inputs_rejected(tmp_path,path):
    with pytest.raises(ValueError):inputs.relative_path(path,tmp_path)


def test_absolute_logical_paths_keep_exact_repository_namespace(tmp_path):
    assert inputs.relative_path(inputs.REPOSITORY_ROOT+'/data/input.csv',tmp_path)=='data/input.csv'


def test_symlink_parent_and_directory_rejected(tmp_path):
    (tmp_path/'actual').mkdir();(tmp_path/'linked').symlink_to(tmp_path/'actual',target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):inputs.relative_path('linked/file.json',tmp_path)
    with pytest.raises(ValueError,match='explicit file'):inputs.relative_path('actual',tmp_path)


def test_import_closure_includes_package_initializers_without_executing(tmp_path):
    files={'pepstab/__init__.py':'raise RuntimeError("must not import")',
        'scripts/entry.py':'from pepstab import first\nfrom . import second\nimport os\n',
        'pepstab/first.py':'from .second import function\n',
        'pepstab/second.py':'function = None\n','scripts/second.py':'pass\n'}
    for name,text in files.items():
        path=tmp_path/name;path.parent.mkdir(exist_ok=True);path.write_text(text)
    assert set(inputs.python_import_closure(tmp_path,['scripts/entry.py']))==set(files)


def test_blocked_inventory_never_hashes_or_freezes(tmp_path,monkeypatch):
    monkeypatch.setattr(inputs,'plan_inventory',lambda *a,**k:{'blockers':['transfer still active']})
    def forbidden(*args):raise AssertionError('must not hash incomplete inputs')
    monkeypatch.setattr(inputs,'sha',forbidden)
    with pytest.raises(ValueError,match='transfer still active'):inputs.freeze_manifest(tmp_path)


def test_manifest_hashes_only_explicit_files_and_detects_changes(tmp_path,monkeypatch):
    (tmp_path/'allowed.txt').write_text('frozen')
    (tmp_path/'unrelated.txt').write_text('must stay out')
    expected=inputs.sha(tmp_path/'allowed.txt')
    plan={'blockers':[],'files':{'allowed.txt':{'expected_sha256':expected}}}
    monkeypatch.setattr(inputs,'plan_inventory',lambda *a,**k:plan)
    manifest=inputs.freeze_manifest(tmp_path)
    assert manifest==dict(schema='final_cloud_input_manifest.1',repository_root=inputs.REPOSITORY_ROOT,
                         files={'allowed.txt':expected})
    (tmp_path/'allowed.txt').write_text('changed')
    with pytest.raises(ValueError,match='content changed'):inputs.freeze_manifest(tmp_path)


def recovery_fixture(tmp_path,monkeypatch):
    import json
    def put(name,value=b'opaque'):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(value if isinstance(value,bytes) else json.dumps(value).encode())
        return {'path':name,'sha256':inputs.sha(path)}
    def blob(name):return put(name)['sha256']
    sources={name:blob(name) for name in inputs.CLOUD_SOURCES}
    legacy='legacy';group={name:blob(legacy+'/'+name) for name in ('protocol.json','group.json')}
    # protocol.json is parsed rather than treated as opaque.
    group['protocol.json']=put(legacy+'/protocol.json',{'sha256':{'group.json':group['group.json']}})['sha256']
    lp=put(legacy+'/assessment_protocol.json',dict(code_sha256=sources,specification={'grouping_hashes':group}))
    tp=put(inputs.TRANSFER+'/protocol.json',dict(code_sha256=sources,legacy_directory=legacy,
        legacy_protocol_sha256=lp['sha256'],grouping_sha256=group,legacy_evidence_sha256={}))
    put(inputs.CLOUD_TRANSFER+'/protocol.json',(tmp_path/tp['path']).read_bytes())
    raw=blob('data/rasmussen_et_al_dataset.csv');splits=blob('data/splits.csv');ranking=blob('reports/finalization/ranking/protocol.json')
    spec=blob(inputs.HYBRID+'/SPECIFICATION.md');graph=blob(inputs.HYBRID+'/fit_graph.json')
    production=put('features/manifest.json',{'feature_bundles':[]})
    put(inputs.HYBRID+'/protocol.json',dict(source_sha256=sources,reference_sha256={},raw_data_sha256=raw,
        splits_sha256=splits,ranking_protocol_sha256=ranking,
        specification={'scientific_specification_sha256':spec,'scientific_graph_sha256':graph},
        feature_provenance={'production_manifest':production,'bundles':[]}))
    for name in ('task_manifest.json','preflight.json','operational_amendment.json','validation.json'):
        blob(inputs.HYBRID+'/'+name)
    blob('reports/further_research/round2/representations/contact_map.json')
    auth={'sources':sources,'protocol':tp}
    for key in ('closure','local_failure_closure','local_failure_exit','local_failure_state','preserved_audit','preparation','runtime_forecast'):
        auth[key]=put('provenance/'+key+'.json',{})
    auth_ref=put(inputs.TRANSFER+'/authorization.json',auth)
    monkeypatch.setattr(inputs,'TRANSFER_AUTH_HASH',auth_ref['sha256'])
    attempt=put(inputs.TRANSFER+'/controller/attempt.json',{})
    exit_ref=put(inputs.TRANSFER+'/controller/exit_receipt.json',{'status':'failed','exit_code':1})
    state=put(inputs.TRANSFER+'/controller/state.json',{'status':'failed'})
    preserved={}
    for cell in inputs.PRESERVED_CELLS:
        runtime={}
        if cell in inputs.CELLS[18:]:
            calls={}
            for i in range(104):
                call=f'call_{i:04d}'
                calls[call]={name:blob(inputs.TRANSFER+'/checkpoints/'+cell+'/'+call+'/'+name)
                    for name in ('call.json','arrays.npz','completed.json')}
            runtime={'checkpoint_cache':{'path':inputs.REPOSITORY_ROOT+'/'+inputs.TRANSFER+'/checkpoints/'+cell,
                'n_calls':104,'replayed_calls':0,'calls':calls}}
            put(inputs.TRANSFER+'/'+cell+'/exit_receipt.json',{})
        files={name:(put(inputs.TRANSFER+'/'+cell+'/'+name,runtime)['sha256'] if name=='runtime.json'
            else blob(inputs.TRANSFER+'/'+cell+'/'+name)) for name in inputs.CELL_FILES}
        regime,name=cell.split('/')
        done=put(inputs.TRANSFER+'/'+cell+'/completed.json',dict(status='complete',regime=regime,cell=name,
            protocol_hash=inputs.TRANSFER_PROTOCOL_HASH,n_fits=166 if regime=='hla' else 400,sha256=files))
        preserved[cell]=done['sha256']
        for name in inputs.CELL_FILES|{'completed.json'}:
            put(inputs.CLOUD_TRANSFER+'/'+cell+'/'+name,(tmp_path/inputs.TRANSFER/cell/name).read_bytes())
    failure=dict(schema='finalization_portable_transfer_failure.1',status='failed_scope_closed',
        preserved_fits=8830,new_discarded_completed_fits=794,discarded_partial_completed_fits=2579,
        remaining_cells=list(inputs.RECOVERY_CELLS),preserved_cell_hashes=preserved,
        owner=dict(pid=59308,absent=True,children_absent=True,launchd_label='org.pepstab.transfer-recovery-v2-20261012',
            host='test',boot_id='test',command='test',started_at_utc='2026-10-12T17:00:00+00:00'),
        authorization=auth_ref,attempt=attempt,exit=exit_ref,state=state,evidence_files={state['path']:state['sha256']})
    put(inputs.FAILURE,failure)
    # Partial originals exist, but recovery manifests must exclude them.
    put(inputs.TRANSFER+'/joint/h4_p0/attempt.json',{})
    put(inputs.TRANSFER+'/checkpoints/joint/h4_p0/call_0103/pending.json',{})
    return put,failure


def test_recovery_inventory_requires_failed_closure_and_only_whole_copies(tmp_path,monkeypatch):
    put,failure=recovery_fixture(tmp_path,monkeypatch)
    inventory=inputs.plan_inventory(tmp_path,transfer_mode='recovery')
    assert inventory['blockers']==[] and inventory['deferred']==[]
    assert inputs.TRANSFER+'/controller/completed.json' not in inventory['files']
    assert inputs.PORTABLE not in inventory['files']
    assert inputs.TRANSFER+'/controller/state.json' in inventory['files']
    assert not any(name.startswith(inputs.TRANSFER+'/joint/h4_') or name.startswith(inputs.TRANSFER+'/checkpoints/joint/h4_') for name in inventory['files'])
    assert all(inputs.CLOUD_TRANSFER+'/'+cell+'/completed.json' in inventory['files'] for cell in inputs.PRESERVED_CELLS)
    manifest=inputs.freeze_manifest(tmp_path,transfer_mode='recovery')
    assert inputs.FAILURE in manifest['files']
    (tmp_path/inputs.CLOUD_TRANSFER/inputs.PRESERVED_CELLS[0]/'predictions.csv').write_bytes(b'changed copy')
    with pytest.raises(ValueError,match='content changed'):inputs.freeze_manifest(tmp_path,transfer_mode='recovery')


def test_recovery_rejects_missing_retained_cell_and_prior_new_attempt(tmp_path,monkeypatch):
    put,failure=recovery_fixture(tmp_path,monkeypatch)
    del failure['preserved_cell_hashes'][inputs.PRESERVED_CELLS[0]];put(inputs.FAILURE,failure)
    put(inputs.CLOUD_TRANSFER+'/joint/h4_p0/attempt.json',{})
    blockers=inputs.plan_inventory(tmp_path,transfer_mode='recovery')['blockers']
    assert any('twenty-five' in text for text in blockers)
    assert any('fresh output directory' in text for text in blockers)


def test_successor_inventory_follows_actual_recovery_completion(tmp_path,monkeypatch):
    import json
    put,failure=recovery_fixture(tmp_path,monkeypatch)
    completed=dict(failure['preserved_cell_hashes'])
    for cell in inputs.RECOVERY_CELLS:
        files={name:put(inputs.CLOUD_TRANSFER+'/'+cell+'/'+name)['sha256'] for name in inputs.CELL_FILES}
        done=put(inputs.CLOUD_TRANSFER+'/'+cell+'/completed.json',dict(status='complete',regime='joint',
            cell=cell.split('/')[1],protocol_hash=inputs.TRANSFER_PROTOCOL_HASH,n_fits=400,sha256=files))
        completed[cell]=done['sha256']
    aggregate=put(inputs.CLOUD_TRANSFER+'/recovery_completion.json',{'completed_cell_hashes':completed})
    output=put('remote/output_manifest.json',{'files':{aggregate['path']:aggregate['sha256']}})
    phase=put('phase_certificate.json',dict(runner_exit=put('remote/exit.json',{}),output_manifest=output))
    failure_ref={'path':inputs.FAILURE,'sha256':inputs.sha(tmp_path/inputs.FAILURE)}
    successor=dict(schema='finalization_portable_cloud_transfer_closure.1',status='complete',scope_root=inputs.CLOUD_TRANSFER,
        protocol_hash=inputs.TRANSFER_PROTOCOL_HASH,retained_fits=10830,preserved_fits=8830,new_fits=2000,
        discarded_partial_completed_fits=2579,completed_cell_hashes=completed,authorization=failure['authorization'],
        failed_predecessor=failure_ref,recovery_authorization=put('recovery_authorization.json',{}),phase_certificate=phase)
    put(inputs.SUCCESSOR,successor)
    inventory=inputs.plan_inventory(tmp_path,transfer_mode='successor')
    assert inventory['blockers']==[]
    assert aggregate['path'] in inventory['files'] and phase['path'] in inventory['files']
    assert all(inputs.CLOUD_TRANSFER+'/'+cell+'/completed.json' in inventory['files'] for cell in inputs.CELLS)
    assert inputs.TRANSFER+'/controller/completed.json' not in inventory['files']


def test_only_exact_authenticated_closed_controller_logs_can_be_input(tmp_path):
    name=inputs.TRANSFER+'/controller/joint_h4_p0.log'
    with pytest.raises(ValueError):inputs.relative_path(name,tmp_path)
    assert inputs.relative_path(name,tmp_path,allow_closed_log=True)==name
    with pytest.raises(ValueError):inputs.relative_path(inputs.TRANSFER+'/controller/other.log',tmp_path,allow_closed_log=True)
    with pytest.raises(ValueError):inputs.relative_path('secrets/key.pem',tmp_path,allow_closed_log=True)
