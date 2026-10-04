"""CPU-only verification of actual model preprocessing for matched stage 4c inputs."""
import json,time
from pathlib import Path
import modal
from boltz_common import boltz_image,weights_vol,msa_vol,out_vol,CACHE_DIR,MSA_DIR,OUT_DIR
from esmfold_common import esmfold_image,esm_weights_vol,ESM_CACHE
from ectodomain_common import *
app=modal.App('pepstab-ectodomain-preflight')
bimage=boltz_image.add_local_python_source('ectodomain_common','esmfold_common')
eimage=esmfold_image.add_local_python_source('ectodomain_common')

@app.function(image=bimage,volumes={CACHE_DIR.parent:weights_vol,MSA_DIR:msa_vol,OUT_DIR:out_vol},cpu=2,memory=16384,timeout=1200,retries=0,max_containers=1)
def boltz_preflight(run_id:str):
    import numpy as np
    from boltz.main import process_inputs
    from boltz.data.module.inferencev2 import PredictionDataset
    from boltz.data.types import Manifest
    import boltz.data.const as const
    start=time.monotonic(); cases=load_cases(); work=Path('/tmp/preflight'); inputs=work/'inputs'; inputs.mkdir(parents=True,exist_ok=True)
    for case in cases:
        input_manifest(case); (inputs/f'{case_id(case)}.yaml').write_text(yaml_for(case))
    process_inputs(list(inputs.glob('*.yaml')),out_dir=work/'parsed',ccd_path=CACHE_DIR/'ccd.pkl',mol_dir=CACHE_DIR/'mols',msa_server_url='https://api.colabfold.com',msa_pairing_strategy='greedy',max_msa_seqs=CAP,use_msa_server=False,boltz2=True,preprocessing_threads=1)
    p=work/'parsed'/'processed'; manifest=Manifest.load(p/'manifest.json')
    assert len(manifest.records)==len(cases),(len(manifest.records),len(cases))
    ds=PredictionDataset(manifest=manifest,target_dir=p/'structures',msa_dir=p/'msa',mol_dir=CACHE_DIR/'mols',constraints_dir=p/'constraints',template_dir=p/'templates',extra_mols_dir=p/'extra_mols')
    by_id={case_id(c):c for c in cases}; captured={}; shape=[]
    for i,record in enumerate(manifest.records):
        f=ds[i]; assert f['record'].id==record.id,'Dataset silently substituted another case'
        case=by_id[record.id]; total=sum(len(c['sequence']) for c in case['chains']); assert len(f['asym_id'])==total
        assert f['msa'].shape[0]==case['chains'][0]['msa']['depth']
        captured[(case['complex_id'],case['arm'])]=capture_groove(f,'boltz2')
        shape.append({'id':record.id,'tokens':total,'msa_shape':list(f['msa'].shape),'asym_ids':np.unique(numpy(f['asym_id'])).tolist()})
    verdict=verify_control(captured)
    token_map={int(const.token_ids[token]):const.prot_token_to_letter[token] for token in set(const.prot_letter_to_token.values())}
    canonical={}
    for (cid,arm),features in captured.items():
        residues=features['msa']; chars=np.array([[token_map[int(x)] for x in row] for row in residues],dtype='U1'); canonical[f'{cid}_{arm}_msa']=chars
        canonical[f'{cid}_{arm}_deletion']=features['deletion_value']
    dest=OUT_ROOT/run_id/'preflight'; dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'boltz2_groove_features.npz',**canonical)
    result={'model':'boltz2','ok':True,'cap':CAP,'seconds':time.monotonic()-start,'cases':shape,'control_checks':verdict,'cgroup_peak_gb':cgroup_peak()}
    (dest/'boltz2.json').write_text(json.dumps(result,indent=2)); out_vol.commit(); return result

@app.function(image=eimage,volumes={ESM_CACHE:esm_weights_vol,MSA_DIR:msa_vol,OUT_DIR:out_vol},cpu=2,memory=16384,timeout=1200,retries=0,max_containers=1)
def esm_preflight(run_id:str):
    import numpy as np
    from esm.models.esmfold2 import ESMFold2InputBuilder
    from esm.models.esmfold2.paired_msa import protein_letter_to_res_type
    start=time.monotonic(); builder=ESMFold2InputBuilder(ccd_cache=ESM_CACHE/'ESMFold2'); captured={}; shape=[]
    for case in load_cases():
        input_manifest(case); f,chains=builder.prepare_input(esm_input(case),seed=0,device='cpu')
        total=sum(len(c['sequence']) for c in case['chains']); assert f['msa'].shape[-1]==total
        assert f['msa'].shape[1]==case['chains'][0]['msa']['depth']
        captured[(case['complex_id'],case['arm'])]=capture_groove(f,'esmfold2')
        shape.append({'id':case_id(case),'tokens':total,'msa_shape':list(f['msa'].shape),'chains':[{'id':c.chain_id,'asym_id':c.asym_id,'entity_id':c.entity_id,'tokens':len(c.tokens)} for c in chains]})
    verdict=verify_control(captured); mapping={v:k for k,v in protein_letter_to_res_type().items()}; canonical={}
    for (cid,arm),features in captured.items():
        residues=features['msa']; canonical[f'{cid}_{arm}_msa']=np.array([[mapping[int(x)] for x in row] for row in residues],dtype='U1')
        canonical[f'{cid}_{arm}_deletion']=features['deletion_value']
    dest=OUT_ROOT/run_id/'preflight'; dest.mkdir(parents=True,exist_ok=True); np.savez_compressed(dest/'esmfold2_groove_features.npz',**canonical)
    meta=ESM_CACHE/'ESMFold2'/'.cache/huggingface/download/config.json.metadata'
    revision=meta.read_text().splitlines()[0] if meta.exists() else None
    assert revision and len(revision)==40,'Cached weight revision could not be identified'
    result={'model':'esmfold2','ok':True,'cap':CAP,'weight_revision':revision,'seconds':time.monotonic()-start,'cases':shape,'control_checks':verdict,'cgroup_peak_gb':cgroup_peak()}
    (dest/'esmfold2.json').write_text(json.dumps(result,indent=2)); out_vol.commit(); return result

@app.local_entrypoint()
def preflight(run_id:str='ectodomain-20261004',model:str='both'):
    handles=[]
    if model in ('both','boltz2'):handles.append(('boltz2',boltz_preflight.spawn(run_id)))
    if model in ('both','esmfold2'):handles.append(('esmfold2',esm_preflight.spawn(run_id)))
    dest=Path(__file__).resolve().parent.parent/'reports'/run_id; dest.mkdir(exist_ok=True)
    for model,handle in handles:
        result=handle.get(); (dest/f'preflight_{model}.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2),flush=True)
