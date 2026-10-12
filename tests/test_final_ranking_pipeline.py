"""Pure reporting/controller gate checks, without biological fits."""
import json
from dataclasses import asdict,replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from scripts import final_ranking, final_ranking_pipeline
from scripts.final_ranking_report import paired_interaction


def test_interaction_preserves_shared_draw_covariance():
    boot=pd.DataFrame(dict(R5_ranked=[.8,.6,.9],R5_mse=[.7,.5,.8],
        R0_ranked=[.6,.7,.8],R0_mse=[.5,.6,.7]))
    point=dict(R5_ranked=.72,R5_mse=.63,R0_ranked=.65,R0_mse=.57)
    delta,draw=paired_interaction(point,boot)
    assert np.isclose(delta,.01)
    np.testing.assert_allclose(draw,np.zeros(3),atol=1e-15)
    # Identical correlated deltas cancel; independent interval subtraction
    # would discard precisely the covariance the matched design protects.
    boot.R0_ranked += [.01,-.01,.02]
    _,draw=paired_interaction(point,boot)
    np.testing.assert_allclose(draw,[-.01,.01,-.02],atol=1e-15)


def test_scientific_gate_requires_both_validation_members(tmp_path,monkeypatch):
    runtime=final_ranking.runtime_identity()
    protocol=dict(protocol_hash="frozen",specification=dict(runtime=runtime))
    monkeypatch.setattr(final_ranking,"verify_freeze",lambda out:protocol)
    (tmp_path/"gate.json").write_text(json.dumps(dict(protocol_hash="frozen",scientific_authorized=True,
        max_workers=2,threads_per_worker=2)))
    (tmp_path/"lambda0_validation.json").write_text(json.dumps(dict(protocol_hash="frozen",status="passed",members=[{}])))
    with pytest.raises(ValueError,match="both real"):
        final_ranking.require_authorization(tmp_path,"scientific")
    (tmp_path/"lambda0_validation.json").write_text(json.dumps(dict(protocol_hash="frozen",status="passed",members=[{},{}])))
    assert final_ranking.require_authorization(tmp_path,"scientific")==protocol


def test_runtime_change_blocks_gate(tmp_path,monkeypatch):
    monkeypatch.setattr(final_ranking,"verify_freeze",lambda out:dict(protocol_hash="frozen",specification=dict(runtime={})))
    with pytest.raises(ValueError,match="runtime"):
        final_ranking.require_authorization(tmp_path,"validation")


def test_expired_deadline_never_launches_workers(tmp_path,monkeypatch):
    monkeypatch.setattr(final_ranking_pipeline,"require_authorization",lambda out,phase:dict(protocol_hash="frozen"))
    (tmp_path/"gate.json").write_text(json.dumps(dict(scientific_deadline_utc=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat())))
    with pytest.raises(ValueError,match="deadline expired"):
        final_ranking_pipeline.run(tmp_path)
    assert not (tmp_path/"pipeline.launched.json").exists()


def test_single_owner_bounded_controller_and_receipts(tmp_path,monkeypatch):
    monkeypatch.setattr(final_ranking_pipeline,"require_authorization",lambda out,phase:dict(protocol_hash="frozen"))
    (tmp_path/"gate.json").write_text(json.dumps(dict(max_workers=2,
        scientific_deadline_utc=(datetime.now(timezone.utc)+timedelta(minutes=1)).isoformat())))
    calls=[]
    class FinishedWorker:
        returncode=0
        def __init__(self,*args,**kwargs):
            calls.append((args,kwargs))
            self.pid=1000+len(calls)
        def poll(self):
            return 0
    monkeypatch.setattr(final_ranking_pipeline.subprocess,"Popen",FinishedWorker)
    monkeypatch.setattr(final_ranking_pipeline.time,"sleep",lambda seconds:None)
    final_ranking_pipeline.run(tmp_path)
    assert len(calls)==10
    assert all(k["env"]["OPENBLAS_NUM_THREADS"]=="2" for a,k in calls)
    assert len(list(tmp_path.glob("worker*.exit.json")))==10
    assert json.loads((tmp_path/"pipeline.completed.json").read_text())["new_scientific_fits"]==840
    with pytest.raises(FileExistsError):
        final_ranking_pipeline.run(tmp_path)


def test_fresh_component_report_replays_every_member_and_scaler(tmp_path,monkeypatch):
    from threadpoolctl import threadpool_limits
    from pepstab import nested
    from pepstab.boltz_embeddings import sha256_file
    from pepstab.final_ranking import RankedHead,candidates,atomic_json,deploy_choice
    from pepstab.round2_embedding import head_candidates
    from pepstab.round2_nested import Ancestry,select,unlabelled
    from scripts.final_ranking import atomic_csv
    from scripts import final_ranking_report
    import pepstab.final_ranking as head_module

    rng=np.random.default_rng(6)
    n=900
    X=rng.normal(size=(n,5)).astype(np.float32)
    train=pd.DataFrame(dict(pair_id=np.arange(n),cluster_id=np.arange(n)//2,
        allele=np.tile(["A","B","C"],n//3),split="train",y_log1p=X[:,0]+.1*X[:,1]))
    features={"R0":X}
    outer=nested.outer_folds(train)
    fit,held=train.loc[outer!=0],train.loc[outer==0]
    ancestry=Ancestry()
    directory=tmp_path/"fold_0/R0"
    head=RankedHead(train,features,ancestry,directory/"members","synthetic")
    # Speed up only synthetic fits while retaining the real audited interface.
    config=head_module.member_config
    monkeypatch.setattr(head_module,"member_config",lambda c,seed,epochs:replace(config(c,seed,epochs),max_epochs=1))
    with threadpool_limits(limits=1):
        choice=select(fit,candidates("R0"),head,ancestry=ancestry,context="outer/0/R0_ranked")
        raw=[]
        for call in head.calls:
            ids=call["prediction_pair_ids"]
            rows=fit.set_index("pair_id").loc[ids].reset_index()[["pair_id","cluster_id","allele"]]
            raw.append(rows.assign(outer_fold=0,inner_fold=int(call["signature"]["context"].rsplit("/",1)[1]),
                representation="R0",candidate=call["signature"]["candidate"]["id"],y_pred=call["predictions"]))
        pred=head.predict(fit,unlabelled(held),choice.candidate,"outer/0/R0_ranked/outer_refit",final=True)
    ancestry.add(role="outer_ranked_selection",fit=fit,held=held,selection=fit,
        candidate=choice.candidate["id"],context="outer/0/R0_ranked")
    mse=dict(candidate=head_candidates("R0","mlp")[0],scores=[dict(candidate="R0_mlp_1",fold_scores=[.1]*3,mean=.1,se=0.)],rule="synthetic baseline")
    baseline=held[["pair_id","cluster_id","allele"]].assign(y_pred=np.zeros(len(held)))
    reused=pd.DataFrame(dict(pair_id=[1],cluster_id=[0],allele=["A"],y_pred=[0.]))
    deployment=deploy_choice(mse,asdict(choice))
    frames=[]
    for arm,values,candidate in (("R0_mse",np.zeros(len(held)),mse["candidate"]["id"]),
        ("R0_ranked",pred,choice.candidate["id"]),("R0_deploy",np.zeros(len(held)) if deployment["procedure"]=="mse" else pred,
        mse["candidate"]["id"] if deployment["procedure"]=="mse" else choice.candidate["id"])):
        frames.append(held[["pair_id","cluster_id","allele"]].assign(outer_fold=0,arm=arm,
            y_pred=np.asarray(values,dtype=np.float64),selected_candidate=candidate))
    atomic_csv(directory/"predictions.csv",pd.concat(frames,ignore_index=True))
    atomic_csv(directory/"inner_predictions.csv",pd.concat(raw,ignore_index=True))
    atomic_csv(directory/"reused_inner_predictions.csv",reused)
    atomic_json(directory/"selection.json",dict(mse=mse,ranked=asdict(choice),deployment=deployment))
    atomic_json(directory/"ancestry.json",ancestry.as_dict())
    atomic_json(directory/"member_manifest.json",head.members)
    atomic_json(directory/"completed.json",dict(protocol_hash="synthetic",status="complete",new_scientific_fits=84,
        files={name:sha256_file(directory/name) for name in ("predictions.csv","inner_predictions.csv",
            "reused_inner_predictions.csv","selection.json","ancestry.json","member_manifest.json")}))
    monkeypatch.setattr(final_ranking_report,"read_baseline",lambda *args:(mse,baseline,reused,[]))
    table,saved,audit=final_ranking_report.audit_component(tmp_path,dict(protocol_hash="synthetic"),train,features,"R0",0)
    assert len(table)==3*len(held)
    assert audit["records"]==112
    member=head.members[0]
    signature=member["identity"]["signature"]
    from pepstab.round2_nested import config_digest
    path=directory/"members"/config_digest(signature)/"stop_0_seed_0.npz"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="arrays changed"):
        final_ranking_report.audit_component(tmp_path,dict(protocol_hash="synthetic"),train,features,"R0",0)
