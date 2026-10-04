"""Shared matched-input contract for the stage 4c pilot (no cloud side effects)."""
from pathlib import Path
import hashlib, json
MSA_ROOT=Path('/msa/ectodomain_stage4c')
OUT_ROOT=Path('/structures/stage4c')
CAP=1024

def case_id(case): return f"{case['complex_id']}_arm_{case['arm']}"
def load_cases(): return json.loads((MSA_ROOT/'pilot_cases.json').read_text())
def yaml_for(case):
    lines=['version: 1','sequences:']
    for chain in case['chains']:
        msa=str(MSA_ROOT/chain['msa']['csv']) if chain['msa'] else 'empty'
        lines+=['  - protein:',f"      id: {chain['id']}",f"      sequence: {chain['sequence']}",f'      msa: {msa}']
    return '\n'.join(lines)+'\n'
def esm_input(case):
    from esm.models.esmfold2 import MSA, ProteinInput, StructurePredictionInput
    proteins=[]
    for chain in case['chains']:
        msa=MSA.from_a3m(str(MSA_ROOT/chain['msa']['a3m'])) if chain['msa'] else None
        proteins.append(ProteinInput(id=chain['id'],sequence=chain['sequence'],msa=msa))
    return StructurePredictionInput(sequences=proteins)
def input_manifest(case):
    for ch in case['chains']:
        if ch['msa']:
            for ext in ['csv','a3m']:
                p=MSA_ROOT/ch['msa'][ext]
                assert hashlib.sha256(p.read_bytes()).hexdigest()==ch['msa'][ext+'_sha256'],str(p)
    return {'complex_id':case['complex_id'],'allele':case['allele'],'peptide':case['peptide'],'arm':case['arm'],'pdb_id':case['pdb_id'],'chains':case['chains'],'cap':CAP}

def numpy(value):
    if hasattr(value,'detach'): value=value.detach().cpu().numpy()
    import numpy as np
    return np.asarray(value)

def capture_groove(features,engine):
    result={}
    # ESMFold2 exposes a batch dimension; Boltz dataset yields unbatched tensors.
    for key in ['msa','deletion_value','has_deletion','deletion_mean','profile','msa_paired','msa_mask','msa_attention_mask']:
        if key not in features:continue
        a=numpy(features[key]); a=a[0] if engine=='esmfold2' else a
        result[key]=a[:182] if key in ('deletion_mean','profile') else a[:,:182]
    return result

def verify_control(captured):
    import numpy as np
    verdict=[]
    complexes=sorted({cid for cid,arm in captured})
    for cid in complexes:
        b=captured[(cid,'B')]; c=captured[(cid,'C')]
        checks={key:bool(np.array_equal(b[key],c[key])) for key in b}
        assert all(checks.values()),(cid,checks,{k:(b[k].shape,c[k].shape) for k in b})
        verdict.append({'complex_id':cid,'checks':checks,'groove_msa_shape':list(b['msa'].shape)})
    return verdict

def cgroup_peak():
    for name in ['/sys/fs/cgroup/memory.peak','/sys/fs/cgroup/memory/memory.max_usage_in_bytes']:
        p=Path(name)
        if p.exists():return int(p.read_text())/1e9
    return None
