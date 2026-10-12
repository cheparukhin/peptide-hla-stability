import json

import numpy as np
import pandas as pd
import pytest

from pepstab.round2_nested import unlabelled
from pepstab.round2_transfer import (ARMS, CANDIDATES, TransferAncestry, TransferPredictor,
    assess_cell, balanced_folds, fit_graph, inner_partitions, outer_cells, validate_partition)
from scripts.round2_transfer import main_gate, verify_cell


def population(groups=15, peptides=30):
    rows = []
    for group in range(groups):
        for peptide in range(peptides):
            rows.append({"pair_id":group*peptides+peptide,"cluster_id":peptide,
                "hla_group":group,"allele":f"A{group}","split":"train","hla_fold":group%5,
                "peptide_fold":peptide%5,"peptide":"A"*9,"hla_pseudoseq":"A"*34,
                "y_log1p":float((peptide+group)%7),"thalf_hours":float(peptide%7)})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("regime",["hla","joint"])
def test_all_outer_inner_and_stopping_exclusions(regime):
    frame = population()
    cells = [(c,f,h) for r,c,f,h in outer_cells(frame) if r == regime]
    assert len(cells) == (5 if regime == "hla" else 25)
    assert sorted(pid for _,_,h in cells for pid in h.pair_id) == sorted(frame.pair_id)
    for _,fit,held in cells:
        ancestry = TransferAncestry(regime)
        inner_rows = []
        for cell, train, predicted in inner_partitions(fit,regime):
            inner_rows.extend(predicted.pair_id)
            for _,gradient,stop in inner_partitions(train,regime,5,stopping=True):
                ancestry.add("mock",gradient,predicted,stopping=stop,scaler=gradient)
                assert set(gradient.pair_id)|set(stop.pair_id) <= set(fit.pair_id)
                if regime == "joint":
                    assert len(gradient)+len(stop) < len(train)
        assert sorted(inner_rows) == sorted(fit.pair_id)
        assert ancestry.audit(frame,fit,held)["outer_exclusion"]


def test_hla_allows_peptide_overlap_joint_rejects_it():
    frame = population()
    fit, held = frame[frame.hla_group>0],frame[frame.hla_group==0]
    validate_partition(fit,held,"hla")
    with pytest.raises(ValueError,match="cluster_id"):
        validate_partition(fit,held,"joint")
    altered = fit.copy()
    altered.loc[altered.index[0],"split"] = "test"
    with pytest.raises(ValueError,match="training"):
        validate_partition(altered,held,"hla")


def test_partitions_invariant_to_labels_order_and_missing_cells_fail():
    frame = population()
    permuted = frame.sample(frac=1,random_state=9)
    permuted.y_log1p = np.nan
    for (_,f,h),(_,ff,hh) in zip(inner_partitions(frame,"joint"),inner_partitions(permuted,"joint")):
        assert set(f.pair_id) == set(ff.pair_id)
        assert set(h.pair_id) == set(hh.pair_id)
    with pytest.raises(ValueError,match="insufficient"):
        list(inner_partitions(frame[frame.hla_group<2],"hla"))


@pytest.mark.parametrize("regime",["hla","joint"])
def test_outer_label_perturbation_cannot_change_choices_or_predictions(regime):
    frame = population()
    _,_,fit,held = next(item for item in outer_cells(frame) if item[0] == regime)
    def evaluate(target):
        ancestry = TransferAncestry(regime)
        def predictor(train,predict,candidate,context):
            assert "y_log1p" not in predict and "thalf_hours" not in predict
            validate_partition(train,predict,regime)
            # Same scores for all candidates exercise baseline-first ties.
            return predict.cluster_id.to_numpy(dtype=float) + train.y_log1p.mean()
        predictions, selection = assess_cell(fit,target,regime,predictor,ancestry,min_rows=2)
        ancestry.audit(frame,fit,target)
        return predictions,selection
    first, choice = evaluate(held)
    corrupted = held.copy()
    corrupted.y_log1p = np.arange(len(held))*1000
    corrupted.thalf_hours = -99
    second, other = evaluate(corrupted)
    assert choice == other
    assert choice["selected"]["transfer_selector"] == "original"
    for arm in ARMS:
        np.testing.assert_array_equal(first[arm],second[arm])


def test_cross_quadrants_and_scaler_cannot_enter_audit():
    frame = population()
    _,_,fit,held = next(item for item in outer_cells(frame) if item[0] == "joint")
    ancestry = TransferAncestry("joint")
    ancestry.add("valid",fit,held,scaler=fit)
    # Tamper an ancestry rowset with an outer cross-quadrant row; the hash is
    # recomputed to ensure this tests actual exclusion, not just corruption.
    cross = frame[frame.hla_fold.eq(0) & frame.peptide_fold.ne(0)].iloc[:1]
    ancestry.records[0]["scaler_rows"] = ancestry.rows(cross)
    with pytest.raises(ValueError,match="cross-quadrant"):
        ancestry.audit(frame,fit,held)


def test_original_members_exact_configuration_and_joint_counts(monkeypatch):
    import pepstab.round2_transfer as transfer
    fitted = []
    class MockMLP:
        def __init__(self,cfg):
            fitted.append(cfg)
            self.best_epoch_,self.n_epochs_ = 3,5
        def fit(self,X,y,Xs,ys):
            return self
        def predict(self,X):
            return np.arange(len(X),dtype=float)
    monkeypatch.setattr(transfer,"MLPRegressor",MockMLP)
    frame = population()
    _,_,fit,held = next(item for item in outer_cells(frame) if item[0] == "joint")
    ancestry = TransferAncestry("joint")
    predictor = TransferPredictor(frame,{},ancestry)
    predictor(fit,unlabelled(held),CANDIDATES[0],"synthetic")
    assert len(fitted) == 30
    assert {cfg.seed for cfg in fitted} == {0,1,2}
    assert {cfg.l2 for cfg in fitted} == {1e-5,1e-3}
    assert all(cfg.hidden == (256,64) and cfg.max_epochs == 300 and cfg.patience == 15 for cfg in fitted)
    assert all(m["n_fit"]+m["n_stop"]+m["excluded_cross_quadrant_rows"] == len(fit) for m in predictor.members)
    assert all(m["excluded_cross_quadrant_rows"] > 0 for m in predictor.members)
    ancestry.audit(frame,fit,held)


def test_ridge_scaler_uses_fitting_rows_only_and_forbids_labels():
    frame = population()
    _,_,fit,held = next(outer_cells(frame))
    features = np.arange(len(frame)*3,dtype=np.float32).reshape(len(frame),3)
    ancestry = TransferAncestry("hla")
    predictor = TransferPredictor(frame,{"R2":features},ancestry)
    with pytest.raises(ValueError,match="assessment labels"):
        predictor(fit,held,CANDIDATES[1],"synthetic")
    pred = predictor(fit,unlabelled(held),CANDIDATES[1],"synthetic")
    assert np.isfinite(pred).all()
    assert ancestry.records[0]["scaler_rows"] == ancestry.rows(fit)
    ancestry.audit(frame,fit,held)


def test_graph_and_main_gate_fail_closed(tmp_path):
    graph = fit_graph()
    assert graph["total_fits"] == 9600
    assert graph["total_fits"] == graph["original_mlp_fits"]+graph["deterministic_ridge_fits"]
    for kind in ("sequence","embedding"):
        (tmp_path/kind).mkdir()
        (tmp_path/kind/"protocol.json").write_text(json.dumps({"protocol_hash":"x"}))
    with pytest.raises(ValueError,match="incomplete"):
        main_gate(tmp_path)


def test_main_gate_requires_audited_current_report(tmp_path):
    from pepstab.boltz_embeddings import sha256_file
    for kind in ("sequence","embedding"):
        directory = tmp_path/kind
        directory.mkdir()
        (directory/"protocol.json").write_text(json.dumps({"protocol_hash":kind}))
        for fold in range(5):
            cell = directory/f"fold_{fold}"
            cell.mkdir()
            (cell/"predictions.csv").write_text("synthetic\n")
            done = {"protocol_hash":kind,"prediction_sha256":sha256_file(cell/"predictions.csv")}
            if kind == "embedding":
                done.update(status="complete",sequence_protocol_hash="sequence",sequence_prediction_sha256=done["prediction_sha256"])
            (cell/"completed.json").write_text(json.dumps(done))
    with pytest.raises(ValueError,match="audited report incomplete"):
        main_gate(tmp_path)
    results = tmp_path/"results/main"
    results.mkdir(parents=True)
    (results/"synthetic.csv").write_text("synthetic\n")
    source = tmp_path/"sequence/protocol.json"
    manifest = {"schema":"round2_results.1","scope":"sequence_and_embedding","all_five_folds_completed":True,
        "protocol_hashes":{"sequence":"sequence","embedding":"embedding"},
        "source_sha256":{str(source):sha256_file(source)},
        "output_sha256":{"synthetic.csv":sha256_file(results/"synthetic.csv")}}
    (results/"manifest.json").write_text(json.dumps(manifest))
    assert len(main_gate(tmp_path)) == 13
    (results/"synthetic.csv").write_text("changed\n")
    with pytest.raises(ValueError,match="output changed"):
        main_gate(tmp_path)
