"""Fixed-crystal pose/feature checks in the shared HLA 1–182 coordinate frame."""
from __future__ import annotations
import argparse,csv,json,sys
from functools import lru_cache
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'scripts'))
from boltz_pose_check import ca_by_chain,find_subsequence,load_prediction,load_crystal
load_crystal=lru_cache(maxsize=5)(load_crystal)

SYMMETRIES={'ASP':[('OD1','OD2')],'GLU':[('OE1','OE2')],'ARG':[('NH1','NH2')],'VAL':[('CG1','CG2')],'LEU':[('CD1','CD2')],'PHE':[('CD1','CD2'),('CE1','CE2')],'TYR':[('CD1','CD2'),('CE1','CE2')]}
def residues(atoms,chain):
    import biotite.structure as struc
    a=atoms[atoms.chain_id==chain]
    starts=struc.get_residue_starts(a,add_exclusive_stop=True)
    return [a[start:end] for start,end in zip(starts[:-1],starts[1:]) if np.any(a.atom_name[start:end]=='CA')]

def load_arrays(folder):
    pae_files=list(folder.glob('pae*.npy'))+list(folder.glob('pae*.npz'));pl=list(folder.glob('plddt*.npy'))+list(folder.glob('plddt*.npz'))
    assert len(pae_files)==len(pl)==1,(folder,pae_files,pl)
    def read(p,key):
        if p.suffix=='.npy':return np.load(p)
        with np.load(p) as a:return a[key]
    return read(pae_files[0],'pae'),read(pl[0],'plddt')

def score(folder,cache):
    import biotite.structure as struc
    meta=json.loads((folder/'metadata.json').read_text());case=meta['inputs'];cifs=list(folder.glob('*.cif'));assert len(cifs)==1
    pred=load_prediction(cifs[0]);xtal=load_crystal(case['pdb_id'],cache)
    pc=ca_by_chain(pred);xc=ca_by_chain(xtal);groove=case['chains'][0]['sequence'][:182];pep=case['peptide']
    ph=find_subsequence(pc,groove);xh=find_subsequence(xc,groove);pp=find_subsequence(pc,pep);xp=find_subsequence(xc,pep)
    assert all(x is not None for x in (ph,xh,pp,xp)),(folder,ph,xh,pp,xp)
    assert len(pc[pp[0]][0])==9 and len(xc[xp[0]][0])==9
    assert len(pc)==len(case['chains']) and sorted(len(v[0]) for v in pc.values())==sorted(len(c['sequence']) for c in case['chains'])
    pca=pc[ph[0]][1][ph[1]:ph[1]+182];xca=xc[xh[0]][1][xh[1]:xh[1]+182]
    _,trans=struc.superimpose(xca,pca);moved=trans.apply(pc[pp[0]][1]);dev=np.linalg.norm(moved-xc[xp[0]][1],axis=1)
    pr=residues(pred,pp[0]);xr=residues(xtal,xp[0]);assert len(pr)==len(xr)==9
    total_sq=0.;natoms=0;reference_atoms=0;counts=[]
    for p,x in zip(pr,xr):
        pd={n:trans.apply(c[None])[0] for n,c,el in zip(p.atom_name,p.coord,p.element) if el not in ('H','D') and n!='OXT'}
        xd={n:c for n,c,el in zip(x.atom_name,x.coord,x.element) if el not in ('H','D') and n!='OXT'}
        common=sorted(pd.keys()&xd.keys());assert {'N','CA','C','O'}.issubset(common),(folder,common)
        reference_atoms+=len(xd);mapping={n:n for n in common}
        alternatives=[mapping]
        pairs=SYMMETRIES.get(str(x.res_name[0]),[])
        if pairs and all(a in common and b in common for a,b in pairs):
            swapped=dict(mapping)
            for a,b in pairs:swapped[a]=b;swapped[b]=a
            alternatives.append(swapped)
        sq=min(sum(float(np.sum((pd[n]-xd[m[n]])**2)) for n in common) for m in alternatives)
        total_sq+=sq;natoms+=len(common);counts.append(len(common))
    pae,plddt=load_arrays(folder);n=sum(len(c['sequence']) for c in case['chains']);assert pae.shape==(n,n) and plddt.shape==(n,)
    assert np.isfinite(pae).all() and np.isfinite(plddt).all()
    start=n-9;pep_pae=pae[start:,:182];groove_pae=pae[:182,start:]
    # Same groove-only geometry definition for all arms and models.
    receptor=pred[(pred.chain_id==ph[0]) & ~np.isin(pred.element,['H','D'])]
    rg=residues(pred,ph[0])[ph[1]:ph[1]+182]; receptor_coords=np.concatenate([r.coord[~np.isin(r.element,['H','D'])] for r in rg])
    contacts=[]
    for r in pr:
        coord=r.coord[~np.isin(r.element,['H','D'])];d=np.linalg.norm(coord[:,None,:]-receptor_coords[None,:,:],axis=-1);contacts.append(int(np.sum(d<4.5)))
    result={'model':meta['model'],'seed':meta['seed'],'complex_id':case['complex_id'],'allele':case['allele'],'arm':case['arm'],'pdb_id':case['pdb_id'],'pred_hla_chain':ph[0],'pred_peptide_chain':pp[0],'crystal_hla_chain':xh[0],'crystal_peptide_chain':xp[0],'hla_ca_rmsd':float(np.sqrt(np.mean(np.sum((trans.apply(pca)-xca)**2,axis=1)))),'peptide_ca_rmsd':float(np.sqrt(np.mean(dev**2))),'peptide_heavy_rmsd':float(np.sqrt(total_sq/natoms)),'heavy_atom_count':natoms,'reference_heavy_atoms':reference_atoms,'heavy_atom_coverage':natoms/reference_atoms,'dev_p2':float(dev[1]),'dev_p9':float(dev[-1]),'per_position_ca':dev.tolist(),'heavy_atom_counts':counts,'peptide_plddt':plddt[start:].tolist(),'peptide_to_groove_pae_mean':float(pep_pae.mean()),'groove_to_peptide_pae_mean':float(groove_pae.mean()),'groove_contacts_4p5A':contacts,'pae_shape':list(pae.shape),'plddt_shape':list(plddt.shape),'status':'ok','peak_gpu_gb':meta.get('peak_gpu_gb'),'fold_s':meta.get('fold_s')}
    assert np.isfinite(result['peptide_heavy_rmsd']) and any(contacts)
    return result

def gate(rows):
    models={}
    for model in ['boltz2','esmfold2']:
        rs=[r for r in rows if r.get('model')==model]; b=[r for r in rs if r.get('arm')=='B' and r.get('status')=='ok'];a=[r for r in rs if r.get('arm')=='A' and r.get('status')=='ok'];checks={}
        checks['complete_45_predictions']=len(rs)==45 and all(r.get('status')=='ok' for r in rs)
        checks['all_B_backbone_and_anchors']=len(b)==15 and all(r['peptide_ca_rmsd']<=2 and r['dev_p2']<=1 and r['dev_p9']<=1 for r in b)
        med={}
        for cid in sorted({r['complex_id'] for r in b}):
            bm=np.median([r['peptide_heavy_rmsd'] for r in b if r['complex_id']==cid]);aa=[r['peptide_heavy_rmsd'] for r in a if r['complex_id']==cid];am=float(np.median(aa)) if len(aa)==3 else None
            med[cid]={'B':float(bm),'A':am,'B_minus_A':float(bm-am) if am is not None else None}
        sentinel=med.get('B0702_IPRRNVATL',{});checks['sentinel_absolute']=sentinel.get('B',999)<=1
        checks['sentinel_improvement']=sentinel.get('B_minus_A') is not None and sentinel['B_minus_A']<=-.5
        checks['other_complex_no_regression']=len(med)==5 and all(r['B_minus_A'] is not None and r['B_minus_A']<=.5 for cid,r in med.items() if cid!='B0702_IPRRNVATL')
        models[model]={'quality_pass':all(checks.values()),'checks':checks,'median_heavy_rmsd':med,'scored':len(rs)}
    return {'models':models,'paired_quality_pass':all(m['quality_pass'] for m in models.values()),'resource_gate':'pending measured production forecast and deadline'}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--structures',type=Path,default=ROOT/'structures/ectodomain_pilot');ap.add_argument('--run-id',default='ectodomain-20261004');ap.add_argument('--smoke',action='store_true');args=ap.parse_args();rows=[]
    for p in sorted(args.structures.rglob('metadata.json')):
        try:r=score(p.parent,ROOT/'structures/crystal_cache')
        except Exception as e:
            m=json.loads(p.read_text());r={'model':m['model'],'seed':m['seed'],'arm':m['inputs']['arm'],'complex_id':m['inputs']['complex_id'],'status':f'{type(e).__name__}: {e}'}
        rows.append(r);print(json.dumps({k:v for k,v in r.items() if k in ['model','seed','arm','complex_id','status','peptide_ca_rmsd','peptide_heavy_rmsd','dev_p2','dev_p9']}),flush=True)
    out=ROOT/'reports'/args.run_id;out.mkdir(exist_ok=True);suffix='smoke' if args.smoke else 'pilot'
    (out/f'{suffix}_pose_scores.json').write_text(json.dumps(rows,indent=2)+'\n')
    if rows:
        cols=list(dict.fromkeys(k for r in rows for k in r))
        with (out/f'{suffix}_pose_scores.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    if args.smoke:
        verdict={'ok':len(rows)==2 and all(r['status']=='ok' and r['peptide_ca_rmsd']<=2 and r['dev_p2']<=1 and r['dev_p9']<=1 for r in rows),'predictions':len(rows)}
    else:verdict=gate(rows)
    (out/f'{suffix}_verdict.json').write_text(json.dumps(verdict,indent=2)+'\n');print(json.dumps(verdict,indent=2),flush=True)
    if args.smoke and not verdict['ok']:sys.exit(1)
