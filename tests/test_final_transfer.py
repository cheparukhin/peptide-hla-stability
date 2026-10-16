import hashlib
import json
import numpy as np
import pandas as pd
import pytest
from pepstab import final_transfer as ft
from pepstab.round2_nested import unlabelled


def population(groups=15,peptides=30):
    return pd.DataFrame([dict(pair_id=g*peptides+p,cluster_id=p,hla_group=g,allele=f'A{g}',split='train',hla_fold=g%5,peptide_fold=p%5,
        peptide='A'*8+'ACDEFGHIKLMNPQRSTVWY'[p%20],hla_pseudoseq='A'*34,y_log1p=float(p%7+g/20),thalf_hours=float(p%7)) for g in range(groups) for p in range(peptides)])


def tree_config(candidate,seed):
    loss='PairLogit' if candidate['learner']=='pairlogit' else 'RMSE'
    return dict(ft.TREE_PARAMS,depth=candidate['depth'],random_seed=seed,loss_function=loss,eval_metric=loss)


@pytest.mark.parametrize('regime',['hla','joint'])
def test_exact_graph_and_all_nested_ancestry(regime):
    frame=population();graph=ft.fit_graph();assert graph['total_fits']==10830
    cells=[(c,f,h) for r,c,f,h in ft.outer_cells(frame) if r==regime]
    assert len(cells)==(5 if regime=='hla' else 25)
    cell,fit,held=cells[0];ancestry=ft.TransferAncestry(regime);count=0
    for _,train,target in ft.inner_partitions(fit,regime):
        for candidate in ft.CANDIDATES:
            for member in ft.member_schedule(train,regime,candidate):
                ancestry.add('synthetic',member['gradient'],target,stopping=member['stopping'],scaler=member['gradient'] if candidate['learner'] in ('ridge','mlp') else None);count+=1
    for procedure in ft.NEW_PROCEDURES:
        candidate=next(c for c in ft.CANDIDATES if c['procedure']==procedure)
        for member in ft.member_schedule(fit,regime,candidate,final=True):
            ancestry.add('synthetic',member['gradient'],held,stopping=member['stopping']);count+=1
    assert count==graph['regimes'][regime]['fits_per_cell'];assert ancestry.audit(frame,fit,held)['outer_exclusion']
    assert graph['by_procedure']['fixed_ranker8']==1620


@pytest.mark.parametrize('regime',['hla','joint'])
def test_raw_fixed_ranker_oof_map_and_held_label_invariance(regime):
    frame=population();_,cell,fit,held=next(x for x in ft.outer_cells(frame) if x[0]==regime)
    def evaluate(target):
        ancestry=ft.TransferAncestry(regime)
        def predictor(train,pred,candidate,context,final=False):
            assert 'y_log1p' not in pred and 'thalf_hours' not in pred
            ft.validate_partition(train,pred,regime)
            return pred.cluster_id.to_numpy(dtype=float)+train.y_log1p.mean()
        reused=dict(predictions={p:held.cluster_id.to_numpy(dtype=float) for p in ft.REUSED_ARMS},scores={'original':1.,'mlp_centred':1.},selection={'synthetic':True})
        values,choice,raw,finalraw=ft.assess_cell(fit,target,regime,predictor,ancestry,reused,min_rows=2)
        ancestry.audit(frame,fit,target)
        return values,choice,raw,finalraw,ancestry
    first=evaluate(held);altered=held.copy();altered.y_log1p=-999;altered.thalf_hours=999
    second=evaluate(altered)
    assert first[1]==second[1];assert first[1]['expanded_selected']=='original'
    assert len(first[2])==len(ft.CANDIDATES)*len(fit)
    for arm in ft.ARMS:np.testing.assert_array_equal(first[0][arm],second[0][arm])
    fixed=first[2].query("candidate=='fixed_ranker8'").set_index('pair_id').loc[fit.pair_id]
    expected=ft.positive_affine(fixed.raw_prediction,fit.y_log1p,minimum_slope=1e-8)
    assert first[1]['ranker_mapping']==expected
    np.testing.assert_array_equal(first[0]['fixed_ranker8'],ft.apply_affine(first[3]['fixed_ranker8'],expected))
    record=next(r for r in first[4].records if r['role']=='fixed_ranker_affine_map')
    assert record['selection_rows']==first[4].rows(fit)


def test_scope_no_grid_and_first_ties():
    fixed=[c for c in ft.CANDIDATES if c['learner']=='pairlogit'];assert len(fixed)==1 and fixed[0]['depth']==8
    assert ft.SEQUENCE_ORDER==('original','mlp_centred','tree_log','tree_centred','fixed_ranker8')
    chosen=ft.select_scores([dict(candidate=c['id'],pooled_inner_score=.5) for c in ft.CANDIDATES])
    assert ft.CANDIDATES[chosen['tree_log']]['depth']==4
    assert ft.CANDIDATES[chosen['R0_ridge']]['regularization']==10000
    with pytest.raises(ValueError,match='order'):ft.select_scores([])


def test_batched_remote_identity_has_no_outer_cell_collision(monkeypatch):
    frame=population();tasks=[]
    def backend(batch):
        assert len(batch)==6;tasks.extend(batch)
        return [dict(task_id=t['task_id'],pair_ids=[r['pair_id'] for r in t['held']],predictions=list(range(len(t['held']))),details=dict(config=tree_config(t['candidate'],t['seed']),best_iteration=1,tree_count=2),worker_seconds=1.,platform='mock',max_rss_mib=1.,versions={}) for t in batch]
    candidate=ft.CANDIDATES[-1]
    for _,cell,fit,held in list(ft.outer_cells(frame))[:2]:
        ancestry=ft.TransferAncestry('hla');predictor=ft.ControlPredictor(frame,ancestry,ranker_backend=backend)
        predictor(fit,unlabelled(held),candidate,'final/fixed_ranker8',final=True)
        assert len(predictor.members)==6 and len(predictor.member_predictions)==6
        ancestry.audit(frame,fit,held)
    assert len({t['task_id'] for t in tasks})==12
    for t in tasks:
        payload={k:v for k,v in t.items() if k!='task_id'}
        assert t['task_id']==hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert all('y_log1p' not in row for row in t['held'])
    corrupted=tasks[0].copy();corrupted['task_id']='bad'
    with pytest.raises(ValueError,match='identity'):ft.ranker_task(corrupted)


def test_remote_order_mismatch_fails_before_member_receipt():
    frame=population();_,_,fit,held=next(ft.outer_cells(frame));ancestry=ft.TransferAncestry('hla')
    def backend(tasks):return [dict(task_id='bad',pair_ids=held.pair_id.tolist()) for t in tasks]
    predictor=ft.ControlPredictor(frame,ancestry,ranker_backend=backend)
    with pytest.raises(ValueError,match='identity'):predictor(fit,unlabelled(held),ft.CANDIDATES[-1],'final/fixed_ranker8',final=True)
    assert predictor.members==[]


def test_r0_scaler_and_mlp_schedule(monkeypatch):
    seen=[]
    class MockMLP:
        def __init__(self,cfg):self.cfg=cfg;self.best_epoch_=2;self.n_epochs_=3
        def fit(self,X,y,Xstop,ystop):seen.append((self.cfg,X.copy(),Xstop.copy()));return self
        def predict(self,X):return np.arange(len(X),dtype=float)
    monkeypatch.setattr(ft,'MLPRegressor',MockMLP)
    frame=population();_,_,fit,held=next(ft.outer_cells(frame));ancestry=ft.TransferAncestry('hla');predictor=ft.ControlPredictor(frame,ancestry)
    candidate=next(c for c in ft.CANDIDATES if c['learner']=='mlp');prediction=predictor(fit,unlabelled(held),candidate,'synthetic')
    assert len(seen)==2 and len(prediction)==len(held)
    assert all(cfg.hidden==(64,) and cfg.max_epochs==200 and cfg.patience==15 for cfg,_,_ in seen)
    assert all(np.max(np.abs(X.mean(axis=0)))<1e-6 for _,X,_ in seen)
    assert all('scaler_var' in m and len(m['scaler_var'])==860 for m in predictor.members)
    ancestry.audit(frame,fit,held)


def test_all_bootstrap_modes_panels_and_group_deletions():
    frame=population(groups=35,peptides=60)
    predictions={p:frame.y_log1p.to_numpy()+i*.01 for i,p in enumerate(ft.ARMS)}
    for minimum in (20,50):
        comparisons,boots,deletions=ft.comparisons(frame,predictions,min_rows=minimum,n_boot=5)
        assert set(boots)=={'peptide','hla','crossed'}
        assert all(len(b)==5 for b in boots.values())
        assert deletions.dropped_hla_group.nunique()==35
        assert (('R0_ridge','R4_ridge') in set(zip(comparisons.baseline,comparisons.candidate)))
        assert (('R0_mlp','R5_mlp') in set(zip(comparisons.baseline,comparisons.candidate)))
        assert set(comparisons.panel_min_rows)=={minimum}


@pytest.mark.parametrize('regime',['hla','joint'])
def test_full_synthetic_cell_replay_and_tampering(tmp_path,monkeypatch,regime):
    from scripts import final_transfer as runner
    from scripts import final_transfer_report as reporting
    frame=population();_,cell,fit,held=next(x for x in ft.outer_cells(frame) if x[0]==regime)
    protocol=dict(protocol_hash='synthetic_protocol',legacy_directory=str(tmp_path/'legacy'))
    reused=dict(predictions={p:held.cluster_id.to_numpy(dtype=float) for p in ft.REUSED_ARMS},scores={'original':.2,'mlp_centred':.3},selection={'synthetic':True},evidence_sha256={'mock':'identity'})
    monkeypatch.setattr(runner,'validate_frozen',lambda out:(frame,protocol));monkeypatch.setattr(runner,'legacy_protocol',lambda path:{});monkeypatch.setattr(reporting,'legacy_protocol',lambda path:{})
    monkeypatch.setattr(runner,'reuse_cell',lambda *args:reused);monkeypatch.setattr(reporting,'reuse_cell',lambda *args:reused)
    class MockModel:
        def __init__(self,*args,**kwargs):self.best_epoch_=2;self.n_epochs_=3
        def fit(self,*args):self.mean=float(np.mean(args[1]));return self
        def predict(self,X):return X[:,:180]@np.arange(180,dtype=float)+self.mean
    monkeypatch.setattr(ft,'Ridge',MockModel);monkeypatch.setattr(ft,'MLPRegressor',MockModel)
    def mock_tree(X,y,a,Xstop,ystop,astop,Xheld,candidate,seed):
        return Xheld[:,:180]@np.arange(180,dtype=float)+float(np.mean(y)),dict(config=tree_config(candidate,seed),best_iteration=2,tree_count=3)
    monkeypatch.setattr(ft,'tree_fit',mock_tree)
    approved=tmp_path/'approved.json';approved.write_text(json.dumps(dict(approved=True,protocol_hash='synthetic_protocol',cells=[f'{regime}/{cell}'],ranker_backend='local')))
    runner.run_cell(tmp_path,regime,cell,approved)
    directory=tmp_path/regime/cell
    aligned,done=reporting.audit_cell(frame,fit,held,regime,cell,directory,protocol)
    assert len(aligned)==len(held) and done['n_fits']==ft.fit_graph()['regimes'][regime]['fits_per_cell']
    # Re-sign corruption to prove audit checks content, beyond artifact hashes.
    selection_path=directory/'selection.json';original=selection_path.read_text();selection=json.loads(original);selection['ranker_mapping']['intercept']+=1;selection_path.write_text(json.dumps(selection))
    def resign(name):
        done=json.loads((directory/'completed.json').read_text());done['sha256'][name]=runner.sha256_file(directory/name);(directory/'completed.json').write_text(json.dumps(done))
    resign('selection.json')
    with pytest.raises(ValueError,match='affine map'):reporting.audit_cell(frame,fit,held,regime,cell,directory,protocol)
    selection_path.write_text(original);resign('selection.json')
    archive_path=directory/'member_predictions.npz'
    with np.load(archive_path,allow_pickle=False) as arrays:changed={key:arrays[key].copy() for key in arrays.files}
    changed['m0000_predictions'][0]+=1
    np.savez_compressed(archive_path,**changed);resign('member_predictions.npz')
    with pytest.raises(ValueError,match='member archive'):reporting.audit_cell(frame,fit,held,regime,cell,directory,protocol)
    with pytest.raises(FileExistsError):runner.run_cell(tmp_path,regime,cell,approved)
