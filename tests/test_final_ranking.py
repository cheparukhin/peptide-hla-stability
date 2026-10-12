"""Synthetic acceptance checks before any new biological fits."""
from dataclasses import asdict
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from pepstab import nested
from pepstab.final_ranking import (RankedHead, array_hash, audit_cached_call,
    candidates, canonical_cache_key, deploy_choice, member_config, one_se)
from pepstab.mlp import MLPConfig, MLPRegressor
from pepstab.phase_a import RankingMLPRegressor, ranking_pairs, pairwise_ranking_loss
from pepstab.round2_embedding import FeatureHead, head_candidates
from pepstab.round2_nested import Ancestry, select, unlabelled
from scripts.final_ranking import fit_graph, require_authorization


@pytest.fixture
def synthetic():
    rng=np.random.default_rng(82)
    n=180
    X=rng.normal(size=(n,6)).astype(np.float32)
    population=pd.DataFrame(dict(pair_id=np.arange(n),cluster_id=np.arange(n)//2,
        allele=np.tile(["A","B","C"],n//3),split="train",y_log1p=X[:,0]+.2*X[:,1]))
    return population,{"R0":X,"R5":X.copy()}


@pytest.mark.parametrize("seed",[0,1,5])
@pytest.mark.parametrize("l2",[1.,.01,.0001])
def test_lambda0_exact_canonical_full_recipe_synthetic(seed,l2):
    rng=np.random.default_rng(90)
    X=rng.normal(size=(70,9)).astype(np.float32)
    y=rng.normal(size=70).astype(np.float32)
    config=MLPConfig(hidden=(64,),l2=l2,lr=.001,batch_size=256,max_epochs=200,patience=15,seed=seed)
    with threadpool_limits(limits=1):
        old=MLPRegressor(config).fit(X[:50],y[:50],X[50:],y[50:])
        new=RankingMLPRegressor(config,rank_weight=0,dev_metric="mse").fit(X[:50],y[:50],X[50:],y[50:])
    assert (new.best_epoch_,new.n_epochs_)==(old.best_epoch_,old.n_epochs_)
    assert new.dev_curve_==old.dev_curve_
    for actual,expected in zip(new.weights_+new.biases_,old.weights_+old.biases_):
        np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(new.predict(X),old.predict(X))


def test_frozen_grid_and_fit_graph(synthetic):
    train,_=synthetic
    grid=candidates("R5")
    assert [(c["rank_weight"],c["regularization"]) for c in grid]==[(w,l) for w in (.1,.3,1.) for l in (1.,.01,.0001)]
    config=member_config(grid[0],0)
    assert config==MLPConfig(hidden=(64,),l2=1.,lr=.001,batch_size=256,max_epochs=200,patience=15,seed=0)
    graph=fit_graph(train)
    assert len(graph["fits"])==840
    assert sum(r["phase"]=="inner" for r in graph["fits"])==540
    assert sum(r["phase"]=="final" for r in graph["fits"])==300
    cluster=train.set_index("pair_id").cluster_id.to_dict()
    for fit in graph["fits"]:
        for a,b in (("gradient","stopping"),("gradient","prediction"),("stopping","prediction"),("selection","prediction")):
            if a not in fit:
                continue
            left={cluster[i] for i in graph["rowsets"][fit[a]]}
            right={cluster[i] for i in graph["rowsets"][fit[b]]}
            assert not left & right


def test_ranknet_pairs_and_gradient():
    y=np.array([2.,2.,1.,3.,0.])
    allele=np.array(["a","a","a","b","b"])
    pairs=ranking_pairs(allele,y)
    hi,lo,weight=pairs
    assert (y[hi]>y[lo]).all() and (allele[hi]==allele[lo]).all()
    assert len(hi)==3
    assert weight[allele[hi]=="a"].sum()==.5
    pred=np.array([.1,.5,-.2,.4,.9])
    _,gradient=pairwise_ranking_loss(pred,pairs)
    eps=1e-6
    finite=[]
    for i in range(len(pred)):
        shift=np.eye(len(pred))[i]*eps
        finite.append((pairwise_ranking_loss(pred+shift,pairs)[0]-pairwise_ranking_loss(pred-shift,pairs)[0])/(2*eps))
    np.testing.assert_allclose(gradient,finite,atol=1e-9)


def test_one_se_positive_order_and_mse_first():
    rows=[dict(candidate="small",mean=.49,se=.01),dict(candidate="large",mean=.5,se=.03)]
    assert one_se(rows)["candidate"]=="small"
    def choice(name,mean,se):
        return dict(candidate=dict(id=name),scores=[dict(candidate=name,mean=mean,se=se,fold_scores=[mean]*3)])
    assert deploy_choice(choice("mse",.49,.01),choice("rank",.5,.03))["procedure"]=="mse"
    assert deploy_choice(choice("mse",.4,.01),choice("rank",.5,.03))["procedure"]=="ranked"


def test_exact_old_cache_signature():
    fit=pd.DataFrame(dict(pair_id=[4,8],y_log1p=[1.,2.]))
    held=pd.DataFrame(dict(pair_id=[9]))
    c=head_candidates("R0","mlp")[0]
    expected=dict(fit=[4,8],y=[1.,2.],held=[9],candidate=c,context="x",final=False,
        feature_hash="f",max_epochs=200,final_seeds=tuple(range(6)),code="s")
    digest=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
    assert canonical_cache_key(fit,held,c,"x","f","s")==digest


def test_cache_ancestry_tamper_rejected(synthetic,tmp_path):
    train,features=synthetic
    outer=nested.outer_folds(train)
    fit,held=train.loc[outer!=0],train.loc[outer==0]
    c=head_candidates("R0","mlp")[0]
    with threadpool_limits(limits=1):
        head=FeatureHead(train,features,Ancestry(),cache_dir=tmp_path,max_epochs=1)
        head.predict(fit,unlabelled(held),c,"test")
    item=json.loads(next(tmp_path.glob("*.json")).read_text())
    audit_cached_call(item,fit,held,c,"test",train)
    item["ancestry"][0]["scaler_fit_rows"]=item["ancestry"][0]["prediction_rows"]
    with pytest.raises(ValueError,match="ancestry"):
        audit_cached_call(item,fit,held,c,"test",train)


def test_ranked_receipts_scaler_and_exact_resume(synthetic,tmp_path):
    train,features=synthetic
    folds=nested.outer_folds(train)
    fit,held=train.loc[folds!=0],train.loc[folds==0]
    c=candidates("R0")[0]
    with threadpool_limits(limits=1):
        first=RankedHead(train,features,Ancestry(),tmp_path,"synthetic",max_epochs=2)
        pred=first.predict(fit,unlabelled(held),c,"x")
        second=RankedHead(train,features,Ancestry(),tmp_path,"synthetic",max_epochs=2)
        repeated=second.predict(fit,unlabelled(held),c,"x")
    np.testing.assert_array_equal(pred,repeated)
    assert len(first.members)==len(second.members)==2
    assert second.ancestry.audit(train,held)["outer_cluster_exclusion"]
    for record in first.ancestry.records:
        assert record["scaler_fit_rows"]==record["fit_rows"]
    arrays=next(tmp_path.rglob("*.npz"))
    with np.load(arrays) as data:
        assert set(data.files)=={"predictions","scaler_mean","scaler_scale","scaler_var","dev_mse_curve"}
    arrays.write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="hash"):
        second.predict(fit,unlabelled(held),c,"x")


def test_ranked_no_assessment_labels_or_unknown_candidate(synthetic,tmp_path):
    train,features=synthetic
    folds=nested.outer_folds(train)
    fit,held=train.loc[folds!=0],train.loc[folds==0]
    head=RankedHead(train,features,Ancestry(),tmp_path,"test",max_epochs=1)
    with pytest.raises(ValueError,match="assessment labels"):
        head.predict(fit,held,candidates("R0")[0],"x")
    changed={**candidates("R0")[0],"rank_weight":3.}
    with pytest.raises(ValueError,match="outside frozen"):
        head.predict(fit,unlabelled(held),changed,"x")


def test_ranked_final30_and_outer_label_invariance(synthetic,tmp_path):
    train,features=synthetic
    outer=nested.outer_folds(train)
    fit,held=train.loc[outer!=0],train.loc[outer==0]
    c=candidates("R0")[0]
    ancestry=Ancestry()
    head=RankedHead(train,features,ancestry,tmp_path,"synthetic",max_epochs=1)
    with threadpool_limits(limits=1):
        values=head.predict(fit,unlabelled(held),c,"final",final=True)
        other=head.predict(fit,unlabelled(held.assign(y_log1p=-9000)),c,"final",final=True)
    np.testing.assert_array_equal(values,other)
    assert len(head.members)==60
    assert {(m["stop_fold"],m["seed"]) for m in head.members}=={(stop,seed) for stop in range(5) for seed in range(6)}
    assert ancestry.audit(train,held)["predicted_cluster_exclusion"]


def test_unfinished_member_cannot_blind_retry(synthetic,tmp_path,monkeypatch):
    train,features=synthetic
    outer=nested.outer_folds(train)
    fit,held=train.loc[outer!=0],train.loc[outer==0]
    head=RankedHead(train,features,Ancestry(),tmp_path,"synthetic",max_epochs=1)
    def fail(*args,**kwargs):
        raise RuntimeError("simulated interrupted fit")
    monkeypatch.setattr(RankingMLPRegressor,"fit",fail)
    with pytest.raises(RuntimeError,match="interrupted"):
        head.predict(fit,unlabelled(held),candidates("R0")[0],"x")
    with pytest.raises(FileExistsError):
        head.predict(fit,unlabelled(held),candidates("R0")[0],"x")
