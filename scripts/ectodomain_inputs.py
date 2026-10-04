"""Prepare matched A/B/C MSA inputs; generate missing ectodomain searches on CPU."""
from __future__ import annotations
import argparse, csv, hashlib, json, re, tarfile, time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
DEST=ROOT/'structures'/'ectodomain_msas'
CAP=1024
PDBS={'A1101_KTFPPTEPK':'1X7Q','A0201_LLWNGPMAV':'5N6B','B1501_ILGPPGSVY':'1XR9','B0702_IPRRNVATL':'7LFZ','B0801_ELRRKMMYM':'4QRU'}
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def slug(a): return a.replace('*','_').replace(':','_').replace('(','_').replace(')','')
def query_columns(row): return ''.join(c for c in row if not c.islower() and c!='.')
def crop(row,n):
    # Insertions precede the aligned residue whose deletion count they affect.
    out=[]; col=0
    for c in row:
        if col>=n: break
        out.append(c)
        if not c.islower() and c!='.': col+=1
    assert col==n
    return ''.join(out)
def rows_a3m(text):
    rows=[]; current=[]
    for line in text.splitlines():
        if line.startswith('#'): continue
        if line.startswith('>'):
            if current: rows.append(''.join(current)); current=[]
        elif line.strip(): current.append(line.strip().replace('.',''))
    if current: rows.append(''.join(current))
    assert rows
    return rows

def select(rows,n,cap=CAP,also_crop=None):
    full_seen=set(); short_seen=set(); result=[]; ids=[]
    for i,row in enumerate(rows):
        assert len(query_columns(row))==n, (i,n,len(query_columns(row)))
        key=row.replace('-','').upper()
        short=crop(row,also_crop) if also_crop else row
        skey=short.replace('-','').upper()
        if key in full_seen or skey in short_seen: continue
        full_seen.add(key); short_seen.add(skey); result.append(row); ids.append(i)
        if len(result)==cap: break
    return result,ids

def write_rows(stem,rows):
    a3m=DEST/f'{stem}.a3m'; csvp=DEST/f'{stem}.csv'
    a3m.write_text(''.join(f'>row_{i} key=-1\n{r}\n' for i,r in enumerate(rows)))
    with csvp.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['sequence','key']); w.writerows((r,-1) for r in rows)
    assert rows_a3m(a3m.read_text())==[r['sequence'] for r in csv.DictReader(csvp.open())]
    return {'a3m':a3m.name,'csv':csvp.name,'a3m_sha256':digest(a3m),'csv_sha256':digest(csvp),'depth':len(rows)}

def prepare():
    DEST.mkdir(parents=True,exist_ok=True); rawdir=DEST/'raw'; rawdir.mkdir(exist_ok=True)
    table=list(csv.DictReader((ROOT/'hla75_ectodomain_b2m.csv').open()))
    raw=list(csv.DictReader((ROOT/'data/rasmussen_et_al_dataset.csv').open()))
    grooves={r['allele']:r['hla_seq'] for r in raw}
    assert len(table)==len(grooves)==75
    assert len({r['b2m'] for r in table})==1
    by_ecto={r['ecto']:r for r in table}; by_groove={r['hla_seq']:r['allele'] for r in raw}
    archive_grooves={}; b2rows=None; sources={}
    with tarfile.open(ROOT/'phla_msa_a3m.tar.gz') as tar:
        for m in tar.getmembers():
            if not m.name.endswith('.a3m'): continue
            text=tar.extractfile(m).read().decode(); rows=rows_a3m(text); q=query_columns(rows[0])
            if len(q)==182:
                assert q in by_groove; archive_grooves[by_groove[q]]=rows
            elif len(q) in (275,276):
                assert q[:275] in by_ecto
                r=by_ecto[q[:275]]; rows=[crop(x,275) for x in rows]
                path=rawdir/f"{slug(r['allele'])}.a3m"
                if not path.exists(): path.write_text(''.join(f'>source_row_{i}\n{x}\n' for i,x in enumerate(rows)))
                sources[r['allele']]={'source':m.name,'source_query_length':len(q),'normalization':'first 275 query columns'}
            elif len(q)==99:
                assert q==table[0]['b2m']; b2rows=rows
            else: raise ValueError((m.name,len(q)))
    assert len(archive_grooves)==75 and b2rows
    beta,_=select(b2rows,99); beta_info=write_rows('beta2m',beta)
    manifest=[]; pending=[]
    for r in table:
        allele=r['allele']; stem=slug(allele)
        assert len(r['ecto'])==275 and r['ecto'][:182]==grooves[allele] and len(r['b2m'])==99
        a,aid=select(archive_grooves[allele],182); ainfo=write_rows(stem+'_A',a)
        path=rawdir/f'{stem}.a3m'
        if not path.exists(): pending.append({'allele':allele,'sequence':r['ecto'],'msa_id':stem}); continue
        source=rows_a3m(path.read_text()); assert query_columns(source[0])==r['ecto']
        b,ids=select(source,275,also_crop=182); c=[crop(x,182) for x in b]
        assert query_columns(c[0])==grooves[allele]
        binfo=write_rows(stem+'_B',b); cinfo=write_rows(stem+'_C',c)
        # One beta2m file per HLA depth guarantees total row depth is unchanged.
        beta_info=write_rows(stem+'_beta2m',beta[:len(b)])
        manifest.append({'allele':allele,'ecto':r['ecto'],'groove':grooves[allele],'b2m':r['b2m'],'note':r['note'],'a3_source':r['a3_source'],'A':ainfo,'B':binfo,'C':cinfo,'beta2m':beta_info,'source_rows':ids,'A_source_rows':aid,'source_raw_sha256':digest(path),'source_provenance':sources.get(allele,{'source':'new ColabFold search; raw file retained'}),'cap':CAP})
    provenance={'created_utc':datetime.now(timezone.utc).isoformat(),'cap':CAP,'files':{str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'phla_msa_a3m.tar.gz',ROOT/'hla75_ectodomain_b2m.csv',ROOT/'hla_prot.fasta',ROOT/'hla75_ectodomain_crystal_validation.csv',ROOT/'data/rasmussen_et_al_dataset.csv',ROOT/'data/splits.csv']},'complete_alleles':len(manifest),'pending_alleles':len(pending),'msas':manifest}
    (DEST/'manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (DEST/'pending.json').write_text(json.dumps(pending,indent=2)+'\n')
    pilot=list(csv.DictReader((ROOT/'reports/boltz_pilot.csv').open())); by_a={r['allele']:r for r in manifest}; cases=[]
    for r in pilot:
        if r['allele'] not in by_a: continue
        t=by_a[r['allele']]; assert r['split']=='train' and len(r['peptide'])==9
        for arm in ['A','B','C']:
            chains=[{'id':'A','sequence':t['ecto'] if arm=='B' else t['groove'],'msa':t[arm]}]
            if arm=='B': chains.append({'id':'B','sequence':t['b2m'],'msa':t['beta2m']})
            chains.append({'id':'C','sequence':r['peptide'],'msa':None})
            cases.append({'complex_id':r['complex_id'],'allele':r['allele'],'peptide':r['peptide'],'groove':t['groove'],'arm':arm,'pdb_id':PDBS[r['complex_id']],'chains':chains,'cap':CAP})
    (DEST/'pilot_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    print(f'Matched MSAs prepared: {len(manifest)}/75 ectodomains; {len(pending)} pending; {len(cases)} pilot inputs; cap={CAP}',flush=True)
    return pending

def generate():
    from boltz.data.msa.mmseqs2 import run_mmseqs2
    from importlib.metadata import version
    pending=prepare()
    if not pending:return
    t=time.monotonic(); print(f'Submitting {len(pending)} missing ectodomain queries in one CPU-only, unpaired ColabFold request',flush=True)
    result=run_mmseqs2([r['sequence'] for r in pending],prefix=str(DEST/'search'),use_env=True,use_filter=True,use_pairing=False)
    alignments=result[0] if isinstance(result,tuple) else result
    assert len(alignments)==len(pending)
    for r,text in zip(pending,alignments):
        assert query_columns(rows_a3m(text)[0])==r['sequence']
        (DEST/'raw'/f"{r['msa_id']}.a3m").write_text(text)
    (DEST/'generation.json').write_text(json.dumps({'engine':'boltz run_mmseqs2','version':version('boltz'),'server':'https://api.colabfold.com','use_env':True,'use_filter':True,'use_pairing':False,'queries':[r['allele'] for r in pending],'seconds':time.monotonic()-t,'finished_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
    prepare()

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--generate-missing',action='store_true'); args=ap.parse_args()
    generate() if args.generate_missing else prepare()
