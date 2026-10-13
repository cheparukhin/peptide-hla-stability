"""Independent saved-output arithmetic for the selected full-train MPNN study.

No fitting, feature extraction, bootstrap generation, or model-weight replay.
Refuses to assess an unfrozen draft. Does not modify experiment artifacts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import followup3_representation_independent_audit as audit

ARMS = ('R5', 'R5_P', 'R5_PC')
CONTRASTS = {('R5','R5_PC'):'net', ('R5_P','R5_PC'):'context', ('R5','R5_P'):'peptide'}
INPUT_NAMES = {'protocol.json','panel.csv','scores.csv','base_features.npz','base_features.provenance.json'}
OUTPUT_NAMES = {'predictions.csv','selection.json','ancestry.json','runtime.json'}


def check_comparisons(frame, predictions, saved, boot, minimum):
    point, panel = audit.scores(frame, predictions, minimum)
    assert len(boot)==2000 and set(boot)==set(ARMS) and np.isfinite(boot.to_numpy()).all()
    assert len(saved)==3 and set(zip(saved.baseline,saved.candidate))==set(CONTRASTS)
    for row in saved.itertuples():
        delta = point[row.candidate]-point[row.baseline]
        draws = boot[row.candidate]-boot[row.baseline]
        np.testing.assert_allclose([row.baseline_score,row.candidate_score,row.delta,row.ci_low,row.ci_high],
            [point[row.baseline],point[row.candidate],delta,*np.quantile(draws,[.025,.975])],atol=2e-13,rtol=0)
        assert row.panel_min_rows==minimum and row.n_rows==len(frame) and row.n_alleles==len(panel)
        assert row.n_clusters==frame.cluster_id.nunique() and row.n_boot==2000 and row.bootstrap_seed==20261003
    return point, panel


def check_fixed_query(frame, old, new, saved, learner):
    """Independently replay all points and differences at fixed old queries."""
    assert len(saved)==12 and set(saved.learner)=={learner}
    panels={}
    for minimum in (20,50):
        old_point,panel=audit.scores(frame,old,minimum)
        new_point,new_panel=audit.scores(frame,new,minimum)
        assert panel==new_panel
        rows=saved.loc[saved.min_rows.eq(minimum)]
        expected={('point',a):(old_point[a],new_point[a]) for a in ARMS}
        expected.update({('increment',name):(old_point[c]-old_point[b],new_point[c]-new_point[b]) for (b,c),name in CONTRASTS.items()})
        assert len(rows)==6 and set(zip(rows.kind,rows.comparison))==set(expected)
        for row in rows.itertuples():
            before,after=expected[(row.kind,row.comparison)]
            np.testing.assert_allclose([row.original_subset,row.fulltrained_same_queries,row.old_to_new_change],
                [before,after,after-before],atol=2e-13,rtol=0)
            assert row.query_rows==len(frame) and row.eligible_alleles==len(panel) and row.panel_alleles=='|'.join(panel)
        panels[str(minimum)]=panel
    return panels


def load_heads(directory, frame, learner):
    tables=[]; fits=0; hashes={}; feature_hashes=None
    for fold in range(5):
        loc=directory/'assessment'/learner/f'fold_{fold}'
        receipt=audit.read(loc/'receipt.json')
        assert receipt['learner']==learner and receipt['fold']==fold
        assert set(receipt['inputs'])==INPUT_NAMES and set(receipt['outputs'])==OUTPUT_NAMES
        for name,value in receipt['inputs'].items(): assert audit.sha(directory/name)==value
        for name,value in receipt['outputs'].items(): assert audit.sha(loc/name)==value
        selections=audit.read(loc/'selection.json'); saved=audit.read(loc/'ancestry.json')
        chosen=audit.audit_selection(frame,selections,ARMS,learner,[fold])
        count=audit.audit_ancestry(frame,saved,ARMS,learner,[fold],chosen)
        assert count==receipt['fits']==(60 if learner=='mlp' else 30)
        assert set(saved['feature_sha256'])==set(ARMS)
        if feature_hashes is not None: assert feature_hashes==saved['feature_sha256']
        feature_hashes=saved['feature_sha256']
        prediction=audit.csv(loc/'predictions.csv')
        assert set(prediction)==set(audit.KEYS)|set(ARMS)
        matched=frame.loc[frame.outer_fold.eq(fold),audit.KEYS].merge(prediction,on=audit.KEYS,how='outer',validate='one_to_one',indicator=True)
        assert matched._merge.eq('both').all()
        tables.append(prediction);fits+=count;hashes[str(loc.relative_to(directory))]=audit.sha(loc/'receipt.json')
    table=pd.concat(tables,ignore_index=True)
    assert len(table)==len(frame) and not table.duplicated(audit.KEYS[:2]).any()
    joined=frame.merge(table,on=audit.KEYS,how='outer',validate='one_to_one',indicator=True)
    assert joined._merge.eq('both').all()
    return joined.drop(columns='_merge'),fits,hashes


def check_extraction(directory,frame,cache,manifest):
    """Check exact cache and disjoint fresh row coverage without running a model."""
    key=lambda table:set(table[['allele','peptide']].itertuples(index=False,name=None))
    digest=lambda a,p:hashlib.sha256(json.dumps([a,p],sort_keys=True,separators=(',',':')).encode()).hexdigest()+'.json'
    old_panel=audit.csv(cache.parent/'panel.csv'); cached=key(old_panel)
    assert len(cached)==len(manifest['row_sha256'])==4449 and cached<=key(frame)
    old_receipt=audit.read(cache.parent/'parser_recovery_receipt.json')
    assert manifest['row_sha256']==old_receipt['completed_row_sha256']
    batches=[audit.csv(directory/f'batch_{i}.csv') for i in (0,1)]
    keys=[key(b) for b in batches]
    assert list(map(len,batches))==[7634,7633] and all(len(b)==len(k) for b,k in zip(batches,keys))
    assert not keys[0]&keys[1] and not cached&(keys[0]|keys[1])
    assert cached|keys[0]|keys[1]==key(frame)
    rowfiles={cache/name:value for name,value in manifest['row_sha256'].items()}
    new_hashes={}
    for i,batch in enumerate(batches):
        receipt=audit.read(directory/f'batch_{i}.receipt.json')
        assert receipt['batch']==i and receipt['rows']==len(batch)
        assert set(receipt['row_sha256'])=={digest(a,p) for a,p in keys[i]}
        assert 1<=receipt['max_workers']<=8 and receipt['threads_per_worker']==2
        assert receipt['worker_start_method']==('serial' if receipt['max_workers']==1 else 'spawn')
        new_hashes.update(receipt['row_sha256'])
        rowfiles.update({directory/'score_rows'/name:value for name,value in receipt['row_sha256'].items()})
    extraction=audit.read(directory/'extraction_receipt.json')
    assert extraction['cached_rows']==4449 and extraction['new_rows']==15267 and extraction['n_rows']==19716
    assert extraction['cached_bytes_unchanged'] and extraction['new_row_sha256']==new_hashes
    assert audit.sha(directory/'scores.csv')==extraction['scores_sha256']
    aggregate=audit.csv(directory/'scores.csv').set_index(['allele','peptide'])
    assert len(aggregate)==19716 and aggregate.index.is_unique and set(aggregate.index)==key(frame)
    numeric=[f'{c}_pos_{i}' for c in ('C','P') for i in range(1,10)]+['C_total','P_total','delta_total']
    for path,value in rowfiles.items():
        assert audit.sha(path)==value
        record=audit.read(path); pair=(record['allele'],record['peptide'])
        assert record['status']=='ok' and path.name==digest(*pair)
        saved=aggregate.loc[pair]
        for field in ('checkpoint_sha256','metadata_sha256','structure_sha256','order_protocol','n_orders','decoding_seed'):
            assert saved[field]==record[field]
        np.testing.assert_array_equal(saved[numeric].to_numpy(float),[record[n] for n in numeric])
    return old_panel,extraction


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--directory',type=Path,default=ROOT/'reports/followup3/inverse_fulltrain_candidate')
    args=cli.parse_args();directory=args.directory
    from scripts.followup3_inverse_fulltrain import verify
    from scripts.followup3_inverse import labelled_panel
    panel,protocol,cache,manifest=verify(directory)
    frame=labelled_panel(directory)
    assert len(frame)==19716 and frame.allele.nunique()==75 and frame.cluster_id.nunique()==3587 and set(frame.split)=={'train'}
    old_panel,extraction=check_extraction(directory,frame,cache,manifest)
    assert extraction['protocol_hash']==protocol['protocol_hash']
    old_frame=frame.merge(old_panel[audit.KEYS],on=audit.KEYS,validate='one_to_one')
    assert len(old_frame)==4449
    parser_recovery=audit.audit_parser_recovery(cache.parent,old_frame)
    comparisons=audit.csv(directory/'comparisons.csv');fixed=audit.csv(directory/'fixed_query_descriptive.csv')
    assert len(comparisons)==12 and len(fixed)==24
    complete=audit.read(directory/'completion_audit.json')
    assert complete['status']=='passed' and complete['rows']==19716 and complete['mlp_fits']==300 and complete['ridge_fits']==150
    assert complete['saved_weight_replay'] is False and complete['tranche_primary_success_claim'] is False
    for name,value in complete['outputs'].items(): assert audit.sha(directory/name)==value
    fits={};receipts={};panels={};fixed_panels={};points={}
    for learner in ('mlp','ridge'):
        joined,fits[learner],receipts[learner]=load_heads(directory,frame,learner)
        values={a:joined[a].to_numpy(float) for a in ARMS}
        for minimum in (20,50):
            saved=comparisons.loc[comparisons.learner.eq(learner)&comparisons.panel_min_rows.eq(minimum)]
            point,population=check_comparisons(joined,values,saved,audit.csv(directory/f'bootstrap_{learner}_min{minimum}.csv'),minimum)
            points[f'{learner}_min{minimum}']=point;panels[str(minimum)]=population
        original,_,_=audit.load_prediction_artifacts(cache.parent,old_frame,'inverse',learner,audit.INVERSE_ARMS)
        query=original[audit.KEYS+list(ARMS)].merge(joined,on=audit.KEYS,validate='one_to_one',suffixes=('_old','_new'))
        assert len(query)==4449
        fixed_panels=check_fixed_query(query,{a:query[a+'_old'] for a in ARMS},{a:query[a+'_new'] for a in ARMS},fixed.loc[fixed.learner.eq(learner)],learner)
    assert fits=={'mlp':300,'ridge':150}
    result=dict(status='passed',study='inverse_fulltrain_selected_development',protocol_hash=protocol['protocol_hash'],rows=19716,arms=list(ARMS),fit_counts=fits,panels=panels,absolute_scores=points,receipt_hashes=receipts,parser_recovery=parser_recovery,cached_rows_unchanged=4449,new_rows=15267,fixed_query_rows=4449,fixed_query_panels=fixed_panels,comparison_rows_checked=12,fixed_query_entries_checked=24,independent_refits=0,bootstrap_regenerated=False,saved_weight_replay=False,tranche_primary=False,scope='Saved prediction, selection, exact ancestry, feature-file/receipt hashes and percentile arithmetic; conditional selected-development evidence only.',audit_source_sha256=audit.sha(__file__),shared_audit_source_sha256=audit.sha(audit.__file__))
    (directory/'independent_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('panels','fixed_query_panels','receipt_hashes','parser_recovery')},indent=2))

if __name__=='__main__':main()
