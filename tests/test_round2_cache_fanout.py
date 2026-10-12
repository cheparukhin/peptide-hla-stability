"""Only synthetic manifests and local fake processes; never invoke Modal."""
import hashlib
import json

import numpy as np
import pytest
from scripts import round2_cache_fanout as c


def test_bundle_protocol_hash_matches_existing_writer_convention():
    from pepstab.boltz_embeddings import json_bytes
    protocol = {'dimensions': {'R2': 1152}, 'schema_version': 'synthetic'}
    assert c.canonical_digest(protocol) == hashlib.sha256(json_bytes(protocol)).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':')))


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'TOTAL_ROWS', 512)
    root = tmp_path
    cohort = root/'full_training/cohort.csv'
    cohort.parent.mkdir()
    cohort.write_text('pair_id,split\n'+''.join(f'{i},train\n' for i in range(512)))
    forecast = root/'extractor/forecast.json'
    put(forecast, {'gate':'engineering_passed_forecast_only_full_cohort_not_launched','training_pairs':512})
    put(forecast.with_name('batch_forecast_options.json'), {'n_cases':512,'options':[{'shard_size':256,'shards':2,'ledger_with_extra60s_per_shard_usd':2.0}]})
    contact = root/'reports/further_research/round2/representations/contact_map.json'
    put(contact, {'contact_map_sha256':'internal-contact'})
    protocol = {'schema_version':'synthetic','dimensions':{'F':3}}
    put(contact.with_name('feature_protocol.json'), protocol)
    for relative in c.EXTRACT_SOURCES+c.SEQUENCE_SOURCES+('data/splits.csv','data/rasmussen_et_al_dataset.csv','structures/ectodomain_msas/manifest.json'):
        path=root/relative; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('synthetic source')
    manifest={'cohort_kind':'full_training','n_rows':512,'source_training_rows':512,'cohort_sha256':c.digest(cohort),'source_hashes':{p:c.digest(root/p) for p in ('data/splits.csv','data/rasmussen_et_al_dataset.csv','structures/ectodomain_msas/manifest.json')}}
    put(cohort.with_name('cohort_manifest.json'),manifest)
    fp={'run_id':'synthetic-run','cohort_sha256':c.digest(cohort),'forecast_sha256':c.digest(forecast),'batch_forecast_sha256':c.digest(forecast.with_name('batch_forecast_options.json')),'contact_map_sha256':c.digest(contact),'n_cases':512,'shard_size':256,'shards':2,'source_hashes':{p:c.digest(root/p) for p in c.EXTRACT_SOURCES}}
    ledger={'fingerprint':fp,'limit_usd':80,'conservative_forecast_usd':2+2*2*c.RATE/3600,'spent_upper_usd':0,'reservations':{},'completed':{}}
    seq=root/'reports/further_research/round2/sequence'
    specification={'synthetic':True}
    ph=hashlib.sha256(json.dumps(specification,sort_keys=True).encode()).hexdigest()
    put(seq/'protocol.json',{'protocol_hash':ph,'specification':specification})
    put(seq/'launch_authorization.json',{'approved':True,'budget_ceiling_usd':18,'protocol_hash':ph,'code_sha256':{p:c.digest(root/p) for p in c.SEQUENCE_SOURCES}})
    put(seq/'modal_smoke.json',{'passed':True})
    put(seq/'modal_budget_forecast.json',{'scope_ceiling_usd':18,'model_based_total_with_margin_usd':17})
    auth={'approved':True,'run_id':'synthetic-run','cache_fingerprint':fp,'cohort_manifest_sha256':c.digest(cohort.with_name('cohort_manifest.json')),'feature_protocol_sha256':c.digest(contact.with_name('feature_protocol.json')),'sequence_authorization_sha256':c.digest(seq/'launch_authorization.json'),'allocations_usd':{'cache':80,'sequence':18,'engineering':2,'total':100}}
    approval=root/'approval.json'; put(approval,auth)
    put(cohort.parent/'extraction_budget_ledger.json',ledger)
    return root,cohort,forecast,approval,auth,ledger,protocol


def add_shard(study, shard, charge=.5):
    root,cohort,forecast,approval,auth,ledger,protocol=study
    directory=root/f'bundle_{shard}'; directory.mkdir()
    pairs=list(range(shard*256,(shard+1)*256))
    input_hashes=[hashlib.sha256(str(i).encode()).hexdigest() for i in pairs]
    np.savez_compressed(directory/'features.npz',F=np.zeros((256,3),dtype=np.float32),
                        pair_id=np.asarray(pairs).astype(str),input_hash=np.asarray(input_hashes))
    ah=c.digest(directory/'features.npz')
    metadata={'n_rows':256,'rows':[{'pair_id':str(i),'input_hash':h} for i,h in zip(pairs,input_hashes)],'array_file':'features.npz','npz_sha256':ah,'feature_protocol':protocol,'feature_protocol_sha256':c.canonical_digest(protocol),'contact_map_sha256':'internal-contact','dimensions':protocol['dimensions'],'schema_version':'synthetic'}
    put(directory/'metadata.json',metadata)
    mh=c.digest(directory/'metadata.json')
    put(directory/'complete.json',{'array_file':'features.npz','metadata_file':'metadata.json','npz_sha256':ah,'metadata_sha256':mh,'schema_version':'synthetic'})
    bundle={'directory':str(directory),'n_rows':256,'npz_sha256':ah,'metadata_sha256':mh}
    ledger['completed'][str(shard)]={'status':'passed','rows':256,'charged_upper_usd':charge,'bundle':bundle}
    ledger['spent_upper_usd']+=charge
    put(cohort.parent/f'shard_{shard:05d}.json',{'status':'passed','returncode':0,'cases':[{'pair_id':i,'ok':True} for i in pairs],'charged_upper_usd':charge})
    bundles=[r['bundle'] for r in ledger['completed'].values()]
    put(cohort.parent/'feature_bundles.json',{'status':'complete' if len(bundles)==2 else 'partial','expected_rows':512,'completed_rows':256*len(bundles),'cohort_sha256':ledger['fingerprint']['cohort_sha256'],'bundles':bundles,'feature_bundles':[b['directory'] for b in bundles]})
    put(cohort.parent/'extraction_budget_ledger.json',ledger)


def ready(study):
    root,cohort,forecast,_,auth,ledger,_=study
    return c.validate_static(root,cohort,forecast,auth,ledger)


def test_pilot_wait_success_and_failure_are_gated(study):
    pairs,protocol,contact,predicted=ready(study)
    cohort,ledger=study[1],study[5]
    assert c.pilot_gate(cohort,ledger,pairs,protocol,contact,predicted) is None
    add_shard(study,0)
    assert c.pilot_gate(cohort,ledger,pairs,protocol,contact,predicted)['gated_forecast_usd']==predicted
    ledger['reservations']['1']=1
    assert c.pilot_gate(cohort,ledger,pairs,protocol,contact,predicted) is None
    ledger['completed']['0']['status']='failed'
    with pytest.raises(c.GateError,match='failed'):
        c.pilot_gate(cohort,ledger,pairs,protocol,contact,predicted)


@pytest.mark.parametrize('bad', [float('nan'),float('inf'),81.])
def test_cost_and_fingerprint_rejection(study,bad):
    study[5]['conservative_forecast_usd']=bad
    with pytest.raises(c.GateError):
        ready(study)


def test_stale_sources_and_sequence_authorization_rejected(study):
    source=study[0]/c.SEQUENCE_SOURCES[0]
    source.write_text('changed')
    with pytest.raises(c.GateError,match='Sequence'):
        ready(study)


def test_pilot_all_cases_compact_checksums_and_forecast(study):
    add_shard(study,0)
    pairs,protocol,contact,predicted=ready(study)
    cohort,ledger=study[1],study[5]
    ledger['completed']['0']['charged_upper_usd']=50
    ledger['spent_upper_usd']=50
    report=c.read(cohort.parent/'shard_00000.json'); report['charged_upper_usd']=50
    put(cohort.parent/'shard_00000.json',report)
    with pytest.raises(c.GateError,match='exceeds 80'):
        c.pilot_gate(cohort,ledger,pairs,protocol,contact,predicted)
    report['cases'][0]['ok']=False; put(cohort.parent/'shard_00000.json',report)
    with pytest.raises(c.GateError,match='every requested'):
        c.pilot_gate(cohort,ledger,pairs,protocol,contact,predicted)


def test_final_exact_coverage_and_compact_bundle_corruption(study):
    add_shard(study,0); add_shard(study,1)
    assert c.verify_complete(study[0],study[1],study[2],study[4])['completed_rows']==512
    (study[0]/'bundle_1/features.npz').write_bytes(b'corrupt')
    with pytest.raises(c.GateError,match='Hash changed'):
        c.verify_complete(study[0],study[1],study[2],study[4])


def test_explicit_approval_lock_and_no_pilot_flag(study):
    root,cohort,forecast,approval=study[:4]
    with pytest.raises(c.GateError,match='Explicit'):
        c.run_controller(root,cohort,forecast,approval,False)
    lock=cohort.parent/'fanout_controller/controller.lock'
    with c.controller_lock(lock):
        with pytest.raises(c.GateError,match='Another'):
            with c.controller_lock(lock):
                pass
    command=c.launch_command(root,cohort,forecast,'synthetic-run')
    assert '--pilot-only' not in command and command[-4:]==['--budget-usd','80','--workers','8']


def test_controller_fake_child_success_and_durable_no_retry(study,monkeypatch):
    add_shard(study,0)
    import scripts.round2_boltz_cohort as cohort_module
    monkeypatch.setattr(cohort_module,'cases_for_cohort',lambda _: [{'pair_id':i} for i in range(512)])
    calls=[]
    class Child:
        pid=12345
        def __init__(self,command,**kwargs):
            calls.append(command)
            assert kwargs['stdin'] is c.subprocess.DEVNULL and kwargs['start_new_session']
        def wait(self,**kwargs):
            add_shard(study,1)
            return 0
        def poll(self):
            return 0
    monkeypatch.setattr(c.subprocess,'Popen',Child)
    assert c.run_controller(*study[:4],True,wait_seconds=1,poll_seconds=.1)['status']=='complete'
    assert len(calls)==1
    with pytest.raises(c.GateError,match='already attempted'):
        c.run_controller(*study[:4],True,wait_seconds=1,poll_seconds=.1)


def test_pilot_timeout_never_starts_child_and_records_halt(study,monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('No child may start while the pilot is unresolved')
    monkeypatch.setattr(c.subprocess,'Popen',forbidden)
    with pytest.raises(c.GateError,match='within 90-minute'):
        c.run_controller(*study[:4],True,wait_seconds=.01,poll_seconds=.01)
    state=c.read(study[1].parent/'fanout_controller/status.json')
    assert state['status']=='halted' and not state['launch_attempted']


def test_child_failure_persists_attempt_and_prevents_retry(study,monkeypatch):
    add_shard(study,0)
    import scripts.round2_boltz_cohort as cohort_module
    monkeypatch.setattr(cohort_module,'cases_for_cohort',lambda _: [{'pair_id':i} for i in range(512)])
    class FailedChild:
        pid=12345
        def __init__(self,*args,**kwargs):
            state=c.read(study[1].parent/'fanout_controller/status.json')
            assert state['launch_attempted'] and state['status']=='launching'
        def wait(self,**kwargs):
            return 1
        def poll(self):
            return 1
    monkeypatch.setattr(c.subprocess,'Popen',FailedChild)
    with pytest.raises(c.GateError,match='exited 1'):
        c.run_controller(*study[:4],True,wait_seconds=1,poll_seconds=.1)
    state=c.read(study[1].parent/'fanout_controller/status.json')
    assert state['status']=='halted' and state['launch_attempted'] and state['child_returncode']==1
    with pytest.raises(c.GateError,match='already attempted'):
        c.run_controller(*study[:4],True,wait_seconds=1,poll_seconds=.1)


def test_real_writer_layout_with_all_representations_and_provenance(tmp_path):
    from pepstab.round2_representations import feature_protocol
    from pepstab.boltz_embeddings import json_bytes
    protocol=feature_protocol()
    ids=['1','7']; hashes=[hashlib.sha256(x.encode()).hexdigest() for x in ids]
    arrays={name:np.zeros((2,width),dtype=np.float32) for name,width in protocol['dimensions'].items()}
    arrays.update(pair_id=np.asarray(ids),input_hash=np.asarray(hashes))
    np.savez_compressed(tmp_path/'features.npz',**arrays)
    ah=c.digest(tmp_path/'features.npz')
    metadata={'schema_version':protocol['schema_version'],'rows':[{'pair_id':i,'input_hash':h,'cache_path':'synthetic','cache_npz_sha256':h,'cache_metadata_sha256':h} for i,h in zip(ids,hashes)],'n_rows':2,'dimensions':protocol['dimensions'],'feature_protocol':protocol,'feature_protocol_sha256':hashlib.sha256(json_bytes(protocol)).hexdigest(),'contact_map_sha256':'synthetic-contact','npz_sha256':ah,'array_file':'features.npz'}
    put(tmp_path/'metadata.json',metadata)
    mh=c.digest(tmp_path/'metadata.json')
    complete={'schema_version':protocol['schema_version'],'array_file':'features.npz','metadata_file':'metadata.json','npz_sha256':ah,'metadata_sha256':mh}
    put(tmp_path/'complete.json',complete)
    record={'directory':str(tmp_path),'npz_sha256':ah,'metadata_sha256':mh,'n_rows':2}
    assert c.bundle_pairs(record,ids,protocol,'synthetic-contact')==ids
    # Keep byte hashes valid but change provenance order: semantic checks must
    # detect this, rather than succeeding merely because content hashes agree.
    arrays['input_hash']=np.asarray(hashes[::-1])
    np.savez_compressed(tmp_path/'features.npz',**arrays)
    ah=c.digest(tmp_path/'features.npz')
    metadata['npz_sha256']=ah; put(tmp_path/'metadata.json',metadata)
    mh=c.digest(tmp_path/'metadata.json')
    complete.update(npz_sha256=ah,metadata_sha256=mh); put(tmp_path/'complete.json',complete)
    record.update(npz_sha256=ah,metadata_sha256=mh)
    with pytest.raises(c.GateError,match='ordered identity or input hash'):
        c.bundle_pairs(record,ids,protocol,'synthetic-contact')
