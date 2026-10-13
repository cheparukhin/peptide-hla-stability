import numpy as np
import pandas as pd
import pytest
from pepstab import followup3_inverse_fulltrain as full


def fixture():
    return pd.DataFrame([dict(pair_id=i*4+a*2+d, allele=f"A{a}", peptide=f"P{i}_{d}", cluster_id=i,
        outer_fold=i%5, split="train", y_log1p=float(i%7+d), complex_id=f"case{i}_{a}_{d}", folder=f"folder{i}_{a}_{d}")
        for i in range(50) for a in range(2) for d in range(2)])


def test_full_panel_preserves_only_original_train_and_cluster_folds():
    frame = fixture()
    splits = frame[["allele", "peptide", "pair_id", "cluster_id", "split"]]
    folds = frame[["pair_id", "cluster_id", "outer_fold"]]
    cohort = frame[["allele", "peptide", "complex_id", "split"]]
    panel = full.full_panel(splits, folds, cohort)
    assert len(panel) == len(frame)
    assert panel.groupby("cluster_id").outer_fold.nunique().max() == 1
    with pytest.raises(ValueError, match="folds must cover"):
        full.full_panel(splits, folds.iloc[:-1], cohort)
    changed = cohort.copy(); changed.loc[0,"split"] = "test"
    with pytest.raises(ValueError, match="ORIGINAL train"):
        full.full_panel(splits, folds, changed)


def test_new_batches_are_complete_disjoint_and_never_recompute_cache():
    frame = fixture().loc[:, ["allele","peptide","cluster_id","outer_fold","complex_id","folder"]]
    cached = set(frame.iloc[:17][["allele","peptide"]].itertuples(index=False,name=None))
    batches = full.extraction_batches(frame, cached)
    keys = [set(b[["allele","peptide"]].itertuples(index=False,name=None)) for b in batches]
    assert not keys[0] & keys[1] and not (keys[0] | keys[1]) & cached
    assert len(keys[0] | keys[1]) == len(frame)-17
    assert abs(len(batches[0])-len(batches[1])) <= 1
    for a,b in zip(batches,full.extraction_batches(frame.sample(frac=1,random_state=4),cached)):
        assert a.equals(b)


@pytest.mark.parametrize("learner,expected", [("mlp",60),("ridge",30)])
def test_three_arm_graph_no_held_labels_ties_and_held_target_invariance(learner,expected):
    frame = fixture()
    class FakeHead:
        def __init__(self,population,features,ancestry):
            self.members=[]
            self.ancestry=ancestry
            assert tuple(features)==full.ARMS
        def predict(self,fit,held,candidate,context,final):
            assert not final and "y_log1p" not in held
            assert not set(fit.cluster_id)&set(held.cluster_id)
            for seed in range(2 if learner=="mlp" else 1):
                stop=fit.loc[fit.cluster_id.eq(fit.cluster_id.unique()[seed])] if learner=="mlp" else None
                gradient=fit.loc[~fit.cluster_id.isin(stop.cluster_id)] if stop is not None else fit
                self.ancestry.add(role="mlp_head" if learner=="mlp" else "ridge_head",fit=gradient,held=held,stopping=stop,
                    context=context,scaler_fit_rows=self.ancestry.rows(gradient))
                self.members.append(dict(seed=seed))
            return held.cluster_id.to_numpy()%7
    arms={a:np.zeros((len(frame),3)) for a in full.ARMS}
    _,pred,choices,ancestry=full.assess_heads(frame,arms,learner,0,FakeHead)
    changed=frame.copy();changed.loc[changed.outer_fold.eq(0),"y_log1p"]+=1000
    _,pred2,choices2,_=full.assess_heads(changed,arms,learner,0,FakeHead)
    assert len(ancestry["members"])==expected and len(choices)==3
    assert choices==choices2 and all(np.array_equal(pred[a],pred2[a]) for a in full.ARMS)
    assert all(c["selected_regularization"]==(1. if learner=="mlp" else 10000.) for c in choices)
    assert full.audit_heads(frame,ancestry,choices,learner,0)
    import copy
    changed_ancestry=copy.deepcopy(ancestry)
    changed_ancestry["records"][0]["scaler_fit_rows"]="bad"
    with pytest.raises(ValueError,match="scaler cluster leakage"):
        full.audit_heads(frame,changed_ancestry,choices,learner,0)
    changed_choices=copy.deepcopy(choices);changed_choices[0]["choices"][0]["score"]+=1
    with pytest.raises(ValueError,match="score/selection replay"):
        full.audit_heads(frame,ancestry,changed_choices,learner,0)


def test_candidate_cannot_execute_without_explicit_freeze(tmp_path):
    from scripts.followup3_inverse_fulltrain import verify
    with pytest.raises(FileNotFoundError): verify(tmp_path)


def test_only_position_features_and_float64_ridge_inputs():
    scored=pd.DataFrame({f"{ctx}_pos_{i}":[float(i),float(i+1)] for ctx in ("P","C") for i in range(1,10)})
    arms=full.feature_arms({"R5":np.zeros((2,1024),dtype=np.float32)},scored)
    assert [x.shape[1] for x in arms.values()]==[1024,1033,1042]
    assert all(x.dtype==np.float64 for x in arms.values())


def test_fixed_query_points_and_changes_share_original_panels():
    frame=fixture()
    old={a:(frame.cluster_id.to_numpy()%7).astype(float) for a in full.ARMS}
    new={a:values.copy() for a,values in old.items()}
    table=full.fixed_query_table(frame,old,new,"mlp")
    assert len(table)==12 and set(table.kind)=={"point","increment"}
    assert table.old_to_new_change.eq(0).all()
    assert table.query_rows.eq(len(frame)).all() and table.eligible_alleles.eq(2).all()
    assert table.groupby("min_rows").panel_alleles.nunique().eq(1).all()
    assert not any("ci" in column for column in table.columns)
    new["R5_PC"]=-new["R5_PC"]
    changed=full.fixed_query_table(frame,old,new,"mlp").query("min_rows==20").set_index("comparison")
    assert changed.loc["net","fulltrained_same_queries"] == changed.loc["R5_PC","fulltrained_same_queries"]-changed.loc["R5","fulltrained_same_queries"]
    assert changed.loc["context","old_to_new_change"] == changed.loc["net","old_to_new_change"]
    assert changed.loc["peptide","old_to_new_change"] == 0
    short=dict(new);short["R5"]=short["R5"][:-1]
    with pytest.raises(ValueError,match="matched fixed-query vectors"):
        full.fixed_query_table(frame,old,short,"mlp")
