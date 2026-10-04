"""Bounded stage 4c pilot only: matched inputs, separate model/seed outputs."""
from pathlib import Path
import hashlib,json,shutil,subprocess,time
import modal
from boltz_common import boltz_image,weights_vol,msa_vol,out_vol,CACHE_DIR,MSA_DIR,OUT_DIR
from esmfold_common import esmfold_image,esm_weights_vol,ESM_CACHE
from ectodomain_common import *
app=modal.App('pepstab-ectodomain-pilot')
bimage=boltz_image.add_local_python_source('ectodomain_common','esmfold_common','boltz_bench')
eimage=esmfold_image.add_local_python_source('ectodomain_common','boltz_bench')


def save_metadata(dest,case,model,seed,settings,extra):
    metadata={'model':model,'seed':seed,'inputs':input_manifest(case),'settings':settings,**extra}
    metadata['output_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.is_file() and p.name!='metadata.json'}
    (dest/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n'); return metadata

def output_path(run_id,model,seed,case): return OUT_ROOT/run_id/model/f'seed_{seed}'/case_id(case)

@app.function(image=bimage,volumes={CACHE_DIR.parent:weights_vol,MSA_DIR:msa_vol,OUT_DIR:out_vol},gpu='A10G',cpu=4,memory=24576,timeout=1200,scaledown_window=10,max_containers=3,retries=0,single_use_containers=True)
def fold_boltz(run_id:str,seed:int,cases:list[dict]):
    import numpy as np
    from boltz_bench import _GpuMemoryProbe,_CompletionProbe,_child_peak_rss_gb
    start=time.monotonic(); msa_vol.reload(); out_vol.reload()
    work=Path('/tmp')/f'{run_id}_boltz_seed_{seed}'; shutil.rmtree(work,ignore_errors=True); inputs=work/'inputs'; inputs.mkdir(parents=True); case_out=work/'out'
    cases=sorted(cases,key=case_id)
    for case in cases: input_manifest(case); (inputs/f'{case_id(case)}.yaml').write_text(yaml_for(case))
    cmd=['boltz','predict',str(inputs),'--out_dir',str(case_out),'--cache',str(CACHE_DIR),'--model','boltz2','--accelerator','gpu','--devices','1','--diffusion_samples','1','--recycling_steps','3','--sampling_steps','200','--output_format','mmcif','--write_full_pae','--max_msa_seqs',str(CAP),'--seed',str(seed),'--override']
    settings={'package':'boltz==2.1.1','weight_revision':'6fdef46d763fee7fbb83ca5501ccceff43b85607','diffusion_samples':1,'recycling_steps':3,'sampling_steps':200,'max_msa_seqs':CAP,'subsample_msa':False,'requested_input_order':[case_id(c) for c in cases],'seed_scope':'process','command':cmd}
    invoke=time.monotonic()
    with _GpuMemoryProbe() as memory,_CompletionProbe(case_out) as done:
        proc=subprocess.run(cmd,capture_output=True,text=True,timeout=1100)
    out=[]; times=sorted(done.seen.items(),key=lambda x:x[1]); previous=invoke; deltas={}
    settings['actual_prediction_order']=[cid for cid,_ in times]
    settings['preprocessing_threads']='boltz default'
    for rank,(cid,at) in enumerate(times):deltas[cid]=(at-previous,rank==0);previous=at
    for case in cases:
        cid=case_id(case); dest=output_path(run_id,'boltz2',seed,case); dest.mkdir(parents=True,exist_ok=True)
        source=next(iter(case_out.rglob(f'{cid}_model_0.cif')),None)
        rec={'model':'boltz2','seed':seed,'complex_id':case['complex_id'],'arm':case['arm'],'ok':False,'peak_gpu_gb':memory.peak_gb,'host_peak_rss_gb':_child_peak_rss_gb(),'cgroup_peak_gb':cgroup_peak()}
        if source:
            for p in source.parent.iterdir():
                if p.is_file() and (p.name.endswith('.cif') or p.name.startswith(('confidence_','pae_','plddt_'))):shutil.copy2(p,dest/p.name)
            pae_files=list(dest.glob('pae_*.npz')); plddt_files=list(dest.glob('plddt_*.npz'))
            assert pae_files and plddt_files,(cid,'missing confidence arrays')
            with np.load(pae_files[0]) as f:pae=f['pae']
            with np.load(plddt_files[0]) as f:plddt=f['plddt']
            n=sum(len(ch['sequence']) for ch in case['chains']); assert pae.shape==(n,n) and plddt.shape==(n,),(cid,pae.shape,plddt.shape)
            assert np.isfinite(pae).all() and np.isfinite(plddt).all()
            dt,warm=deltas[cid]; rec.update(ok=True,fold_s=dt,is_warmup=warm,pae_shape=list(pae.shape),plddt_shape=list(plddt.shape))
        else:rec['error']=(proc.stderr or proc.stdout)[-2500:]
        save_metadata(dest,case,'boltz2',seed,settings,rec);out.append(rec)
    batch={'model':'boltz2','seed':seed,'returncode':proc.returncode,'container_s':time.monotonic()-start,'batch_s':time.monotonic()-invoke,'peak_gpu_gb':memory.peak_gb,'cgroup_peak_gb':cgroup_peak(),'results':out,'stdout_tail':proc.stdout[-5000:],'stderr_tail':proc.stderr[-5000:]}
    base=OUT_ROOT/run_id/'boltz2'/f'seed_{seed}'; (base/f'batch_{len(cases)}.json').write_text(json.dumps(batch,indent=2));out_vol.commit(); return batch

@app.function(image=eimage,volumes={ESM_CACHE:esm_weights_vol,MSA_DIR:msa_vol,OUT_DIR:out_vol},gpu='L40S',cpu=4,memory=65536,timeout=1200,scaledown_window=10,max_containers=3,retries=0,single_use_containers=True)
def fold_esm(run_id:str,seed:int,cases:list[dict]):
    import numpy as np, torch
    from esm.models.esmfold2 import EsmFold2Model,ESMFold2InputBuilder
    from boltz_bench import _GpuMemoryProbe
    start=time.monotonic(); msa_vol.reload();out_vol.reload();local=ESM_CACHE/'ESMFold2'
    meta=local/'.cache/huggingface/download/config.json.metadata'; revision=meta.read_text().splitlines()[0]; assert revision=='69869f737beffec5294845ede23db5fc0b4f509e'
    for weight_meta in (local/'.cache/huggingface/download').glob('model-*.safetensors.metadata'):
        assert weight_meta.read_text().splitlines()[0]==revision,str(weight_meta)
    settings={'package':'esm==3.4.1.post1','weight_revision':revision,'diffusion_samples':1,'num_loops':20,'sampling_steps':200,'msa_max_depth':CAP,'msa_column_mask_rate':0.0,'lm_dropout':0.3,'seed_scope':'per-case','input_order':[case_id(c) for c in sorted(cases,key=case_id)]}
    load=time.monotonic()
    with _GpuMemoryProbe() as memory:
        model=EsmFold2Model.from_pretrained(str(local),device='cuda').eval();builder=ESMFold2InputBuilder(ccd_cache=local)
        load_s=time.monotonic()-load;out=[]
        for i,case in enumerate(sorted(cases,key=case_id)):
            input_manifest(case);dest=output_path(run_id,'esmfold2',seed,case);dest.mkdir(parents=True,exist_ok=True);t=time.monotonic()
            rec={'model':'esmfold2','seed':seed,'complex_id':case['complex_id'],'arm':case['arm'],'ok':False,'is_warmup':i==0}
            try:
                with torch.inference_mode():
                    res=builder.fold(model,esm_input(case),num_loops=20,num_sampling_steps=200,num_diffusion_samples=1,seed=seed,msa_max_depth=CAP,msa_column_mask_rate=0.0,lm_dropout=0.3,complex_id=case_id(case))
                if isinstance(res,list):res=res[0]
                cif=res.complex.to_mmcif();(dest/f'{case_id(case)}.cif').write_text(cif)
                pae=numpy(res.pae);plddt=numpy(res.plddt);n=sum(len(ch['sequence']) for ch in case['chains'])
                assert pae.shape==(n,n) and plddt.shape==(n,),(pae.shape,plddt.shape,n)
                assert np.isfinite(pae).all() and np.isfinite(plddt).all()
                np.save(dest/'pae.npy',pae);np.save(dest/'plddt.npy',plddt)
                conf={'ptm':res.ptm,'iptm':res.iptm,'pair_chains_iptm':numpy(res.pair_chains_iptm).tolist() if res.pair_chains_iptm is not None else None}
                (dest/'confidence.json').write_text(json.dumps(conf,indent=2));rec.update(ok=True,fold_s=time.monotonic()-t,pae_shape=list(pae.shape),plddt_shape=list(plddt.shape))
            except Exception as exc:
                rec.update(error=f'{type(exc).__name__}: {exc}',fold_s=time.monotonic()-t)
            rec.update(peak_gpu_gb=memory.peak_gb,cgroup_peak_gb=cgroup_peak());save_metadata(dest,case,'esmfold2',seed,settings,rec);out.append(rec);out_vol.commit()
            print(json.dumps(rec),flush=True)
            if not rec['ok']:break
    batch={'model':'esmfold2','seed':seed,'weight_load_s':load_s,'container_s':time.monotonic()-start,'peak_gpu_gb':memory.peak_gb,'cgroup_peak_gb':cgroup_peak(),'results':out}
    base=OUT_ROOT/run_id/'esmfold2'/f'seed_{seed}'; (base/f'batch_{len(cases)}.json').write_text(json.dumps(batch,indent=2));out_vol.commit();return batch

@app.local_entrypoint()
def pilot(run_id:str='ectodomain-20261004',phase:str='smoke',model:str='both'):
    root=Path(__file__).resolve().parent.parent;cases=json.loads((root/'structures/ectodomain_msas/pilot_cases.json').read_text())
    assert len(cases)==15;dest=root/'reports'/run_id;dest.mkdir(exist_ok=True)
    for name in ['boltz2','esmfold2']:
        assert json.loads((dest/f'preflight_{name}.json').read_text())['ok']
    assert json.loads((dest/'cross_model_preflight.json').read_text())['ok'],'Run shared-content verification before GPU'
    if phase=='rest':
        assert json.loads((dest/'smoke_verdict.json').read_text())['ok'],'First validate both smoke outputs and poses'
    selected=[c for c in cases if c['arm']=='B' and c['complex_id']=='B0702_IPRRNVATL'] if phase=='smoke' else cases
    assert phase in ('smoke','rest')
    handles=[]
    seeds=[0] if phase=='smoke' else [0,1,2]
    for seed in seeds:
        batch=selected if phase=='smoke' or seed!=0 else [c for c in selected if not(c['arm']=='B' and c['complex_id']=='B0702_IPRRNVATL')]
        if model in ('both','boltz2'): handles.append(('boltz2',seed,fold_boltz.spawn(run_id,seed,batch)))
        if model in ('both','esmfold2'): handles.append(('esmfold2',seed,fold_esm.spawn(run_id,seed,batch)))
    for name,seed,handle in handles:
        result=handle.get();(dest/f'{phase}_{name}_seed_{seed}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
