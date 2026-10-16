"""Tiny synthetic metadata checks only; no scientific data, fits or providers."""
import numpy as np
import pandas as pd
import pytest
from scripts import final_label_efficiency as study
from pepstab import nested


def frame():
    clusters=np.repeat(np.arange(100),3)
    return pd.DataFrame({'pair_id':np.arange(len(clusters)), 'cluster_id':clusters,
        'split':'train','y_log1p':np.arange(len(clusters))/100})


def test_exact_counts_label_blind_ordering_and_nested_shared_subsets():
    data=frame();plan=study.make_tasks(data)
    assert len(plan['new_members'])==120 and len(plan['reused_full_members'])==20
    assert plan['new_supervised_fits']==120 and plan['new_full_budget_fits']==0
    altered=data.copy();altered['y_log1p']=-altered.y_log1p*7
    assert study.make_tasks(altered)==plan
    for fold in range(5):
        for draw in range(3):
            choices={}
            for fraction in (25,50):
                tasks=[t for t in plan['new_members'] if (t['outer_fold'],t['draw'],t['fraction'])==(fold,draw,fraction)]
                assert len(tasks)==4
                assert len({tuple(t['selected_clusters']) for t in tasks})==1
                assert len({tuple(t['selected_rows']) for t in tasks})==1
                choices[fraction]=set(tasks[0]['selected_clusters'])
            assert choices[25]<choices[50]


def test_original_stopping_assignments_and_all_labels_count():
    data=frame();outer=nested.outer_folds(data);plan=study.make_tasks(data)
    for task in plan['new_members']:
        fit=data.loc[outer!=task['outer_fold']];held=data.loc[outer==task['outer_fold']]
        original_stops=nested.inner_folds(fit,5);mask=fit.cluster_id.isin(task['selected_clusters'])
        assert task['gradient_rows']==fit.loc[mask&(original_stops!=task['stop_fold'])].pair_id.tolist()
        assert task['stopping_rows']==fit.loc[mask&(original_stops==task['stop_fold'])].pair_id.tolist()
        assert set(task['gradient_rows'])|set(task['stopping_rows'])==set(task['selected_rows'])
        assert len(task['gradient_rows'])+len(task['stopping_rows'])==task['selected_rows_count']
        gradient=data.set_index('pair_id').loc[task['gradient_rows']]
        stopping=data.set_index('pair_id').loc[task['stopping_rows']]
        assert not set(gradient.cluster_id)&set(stopping.cluster_id)
        assert not set(task['selected_clusters'])&set(held.cluster_id)
        assert task['held_rows']==held.pair_id.tolist()


def test_nontraining_or_duplicate_rows_are_refused():
    data=frame();data.loc[0,'split']='test'
    with pytest.raises(ValueError):study.make_tasks(data)
    data=frame();data.loc[0,'pair_id']=1
    with pytest.raises(ValueError):study.make_tasks(data)


def test_float64_member_average_preserves_midpoint_rank_without_float32_tie():
    low=np.float32(1);high=np.nextafter(low,np.float32(np.inf))
    vectors=[np.array([low,low],dtype=np.float32),np.array([high,low],dtype=np.float32)]
    averaged=study.average_members(vectors)
    assert averaged.dtype==np.float64
    assert averaged[0]==(float(low)+float(high))/2
    assert averaged[0]>averaged[1]
    assert np.mean(vectors,axis=0)[0]==np.mean(vectors,axis=0)[1]


def test_saved_member_reload_exact_check_with_mocked_predict_only(tmp_path,monkeypatch):
    from pepstab import final_hybrids as core
    pred=np.array([1,2],dtype=np.float32);features=np.ones((2,2))
    arrays={name:np.zeros((1,),dtype=np.float32) for name in ('w0','w1','b0','b1')}
    arrays.update(offset=np.asarray(0.),curve=np.array([1.]),scaler_mean=np.zeros(2),
        scaler_var=np.ones(2),scaler_scale=np.ones(2),prediction=pred)
    np.savez_compressed(tmp_path/'arrays.npz',**arrays)
    pd.DataFrame({'y_pred':pred.astype(np.float64)}).to_csv(tmp_path/'predictions.csv',index=False)
    class Scaler:
        def transform(self,value):return value
    class Model:
        def predict(self,value):return pred.copy()
    monkeypatch.setattr(core,'_restore_scaler',lambda state,prefix,n:Scaler())
    monkeypatch.setattr(core,'_restore_mlp',lambda state,prefix,seed:Model())
    result=study.verify_saved_member(tmp_path,arrays,features,features,pred,0,10)
    assert result['saved_model_predictions_replayed_exactly'] is True
    altered=pred.copy();altered[0]=np.nextafter(altered[0],np.float32(np.inf))
    with pytest.raises(AssertionError):study.verify_saved_member(tmp_path,arrays,features,features,altered,0,10)
