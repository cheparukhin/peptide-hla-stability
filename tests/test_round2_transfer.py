import json

import numpy as np
import pandas as pd
import pytest

from pepstab.round2_nested import unlabelled
from pepstab.round2_transfer import (ARMS, CANDIDATES, TransferAncestry, TransferPredictor,
    assess_cell, balanced_folds, fit_graph, inner_partitions, outer_cells, validate_partition, candidate_registry, SEQUENCE_SELECTOR, EMBEDDING_SELECTOR, replay_selection)
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
    assert choice["selected"][SEQUENCE_SELECTOR] == "original"
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
    predictor(fit,unlabelled(held),CANDIDATES[0],"final/synthetic")
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
    grid = candidate_registry(("R2_ridge",))
    predictor = TransferPredictor(frame,{"R2":features},ancestry,candidates=grid)
    with pytest.raises(ValueError,match="assessment labels"):
        predictor(fit,held,grid[3],"synthetic")
    pred = predictor(fit,unlabelled(held),grid[3],"synthetic")
    assert np.isfinite(pred).all()
    assert ancestry.records[0]["scaler_rows"] == ancestry.rows(fit)
    ancestry.audit(frame,fit,held)


def test_graph_and_main_gate_fail_closed(tmp_path):
    graph = fit_graph()
    assert graph["total_fits"] == 10440
    assert graph["total_fits"] == graph["sequence_mlp_fits"]+graph["deterministic_ridge_fits"]
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


def test_nomination_cap_existing_grids_and_counts():
    from pepstab.round2_transfer import validate_candidates
    for names,total in ((('R2_ridge','R4_ridge'),11940), (('R3_mlp','R5_mlp'),15120), (('R3_ridge','R5_mlp'),13530)):
        registry = candidate_registry(names)
        assert fit_graph(registry)['total_fits'] == total
        assert len(registry) == 9
        validate_candidates(registry)
    for names in (('R2_ridge','R3_mlp','R4_ridge'),('R3_pooled_mlp',),('R0_ridge',),('R5_mlp','R5_mlp')):
        with pytest.raises(ValueError):
            candidate_registry(names)
    altered = list(candidate_registry(('R3_mlp',)))
    altered[-1] = dict(altered[-1],regularization=.002)
    with pytest.raises(ValueError,match='grid'):
        validate_candidates(altered)


def test_selector_replay_both_sequence_fallbacks_and_parsimony():
    grid = candidate_registry(('R3_mlp','R5_ridge'))
    scores = [dict(candidate=c['id'],pooled_inner_score=.1) for c in grid]
    choice = replay_selection(scores,grid)
    assert choice[SEQUENCE_SELECTOR] == choice[EMBEDDING_SELECTOR] == 0
    scores[1]['pooled_inner_score'] = .3
    choice = replay_selection(scores,grid)
    assert choice[SEQUENCE_SELECTOR] == choice[EMBEDDING_SELECTOR] == 1
    scores[3]['pooled_inner_score'] = .4
    assert replay_selection(scores,grid)[EMBEDDING_SELECTOR] == 3
    assert replay_selection(scores,grid)[SEQUENCE_SELECTOR] == 1


@pytest.mark.parametrize('regime',['hla','joint'])
def test_centering_is_member_gradient_only_and_unseen_fallback(monkeypatch,regime):
    import pepstab.round2_transfer as transfer
    from pepstab.phase_a import shrunk_allele_means,allele_offsets
    calls = []
    class ZeroMLP:
        def __init__(self,cfg): self.best_epoch_,self.n_epochs_ = 0,1
        def fit(self,X,y,Xs,ys):
            calls.append((y.copy(),ys.copy()))
            return self
        def predict(self,X): return np.zeros(len(X))
    monkeypatch.setattr(transfer,'MLPRegressor',ZeroMLP)
    frame = population()
    _,_,fit,held = next(i for i in outer_cells(frame) if i[0] == regime)
    candidate = CANDIDATES[1]
    ancestry = TransferAncestry(regime)
    predictor = TransferPredictor(frame,{},ancestry)
    actual = predictor(fit,unlabelled(held),candidate,'select/centred')
    expected = []
    for specification, (y,ys),member in zip(transfer.member_schedule(fit,regime,candidate,final=False),calls,predictor.members):
        g,stop = specification['gradient'],specification['stopping']
        means,global_mean = shrunk_allele_means(g.allele,g.y_log1p,100)
        np.testing.assert_allclose(y,g.y_log1p-allele_offsets(g.allele,means,global_mean))
        # Matched-HLA stopping and held alleles are unseen in gradient rows.
        assert set(stop.allele).isdisjoint(means) and set(held.allele).isdisjoint(means)
        np.testing.assert_allclose(ys,stop.y_log1p-global_mean)
        assert member['centering_means'] == means and member['centering_global'] == global_mean
        expected.append(np.full(len(held),global_mean))
    assert len(calls) == 12
    np.testing.assert_allclose(actual,np.mean(expected,axis=0))
    ancestry.audit(frame,fit,held)
    # Corrupt one member's stopping labels only: its own centring stays fixed.
    _,g,stop = next(inner_partitions(fit,regime,3,stopping=True))
    altered = fit.copy();altered.loc[stop.index,'y_log1p'] += 10000
    before = shrunk_allele_means(g.allele,g.y_log1p,100)
    _,gg,_ = next(inner_partitions(altered,regime,3,stopping=True))
    assert before == shrunk_allele_means(gg.allele,gg.y_log1p,100)


@pytest.mark.parametrize('regime',['hla','joint'])
def test_neural_feature_scaling_gradient_only_and_recipes(monkeypatch,regime):
    import pepstab.round2_transfer as transfer
    calls = []
    class FakeMLP:
        def __init__(self,cfg): self.best_epoch_,self.n_epochs_ = 0,1
        def fit(self,X,y,Xs,ys): calls.append((X.copy(),Xs.copy())); return self
        def predict(self,X): return np.zeros(len(X))
    monkeypatch.setattr(transfer,'MLPRegressor',FakeMLP)
    frame = population()
    _,_,fit,held = next(i for i in outer_cells(frame) if i[0] == regime)
    features = np.c_[frame.pair_id,frame.cluster_id].astype(np.float32)
    grid = candidate_registry(('R3_mlp',))
    ancestry = TransferAncestry(regime)
    predictor = TransferPredictor(frame,{'R3':features},ancestry,candidates=grid)
    for final,count in ((False,2),(True,30)):
        start = len(predictor.members)
        predictor.predict(fit,unlabelled(held),grid[3],'final/head' if final else 'select/head',final=final)
        assert len(predictor.members)-start == count
        for m in predictor.members[start:]:
            ids = ancestry.rowsets[m['fit_rows']]
            rows = frame.set_index('pair_id').loc[ids].index.to_numpy()
            np.testing.assert_allclose(m['scaler_mean'],features[rows].mean(axis=0),rtol=1e-6)
            assert m['scaler_rows'] == m['fit_rows']
            assert m['config']['hidden'] == (64,) and m['config']['max_epochs'] == 200
    assert all(np.allclose(X.mean(axis=0),0,atol=1e-5) for X,_ in calls)
    ancestry.audit(frame,fit,held)


def test_freeze_requires_explicit_nominees_after_main_gate(tmp_path,monkeypatch):
    import scripts.round2_transfer as runner
    monkeypatch.setattr(runner,'main_gate',lambda _: {'gated':'main'})
    with pytest.raises(ValueError,match='explicit one or two'):
        runner.freeze(tmp_path,tmp_path,tmp_path,None,'documented amendment')


def test_frozen_main_nomination_grid_cannot_expand(tmp_path):
    from scripts.round2_transfer import validate_main_nominations
    registry = candidate_registry(('R3_mlp',))
    directory = tmp_path/'embedding';directory.mkdir()
    data = dict(specification=dict(selector_order=['original','R3_mlp'],candidate_grids={'R3_mlp':json.loads(json.dumps(registry[3:]))}))
    (directory/'protocol.json').write_text(json.dumps(data))
    validate_main_nominations(tmp_path,registry)
    data['specification']['candidate_grids']['R3_mlp'][-1]['regularization'] = .003
    (directory/'protocol.json').write_text(json.dumps(data))
    with pytest.raises(ValueError,match='frozen main'):
        validate_main_nominations(tmp_path,registry)


def test_saved_frozen_spec_rejects_valid_head_grid_swap_with_old_hash(tmp_path,monkeypatch):
    import scripts.round2_transfer as runner
    from pepstab.round2_nested import config_digest
    frame = population()
    ridge = candidate_registry(('R3_ridge',))
    neural = candidate_registry(('R3_mlp',))
    specification = runner.specification(frame,runner.OUT,('R3_ridge',))
    protocol = dict(status='FROZEN',specification=specification,
        protocol_hash=config_digest(specification),main_completion_hashes={'main':'complete'},
        feature_hashes={'R3':'same-feature-key'},code_sha256={})
    directory = tmp_path/'embedding';directory.mkdir()
    (directory/'protocol.json').write_text(json.dumps(dict(specification=dict(
        selector_order=['original','R3_ridge','R3_mlp'],
        candidate_grids={'R3_ridge':list(ridge[3:]),'R3_mlp':list(neural[3:])}))))
    monkeypatch.setattr(runner,'training_frame',lambda _:frame)
    monkeypatch.setattr(runner,'main_gate',lambda _: {'main':'complete'})
    path = tmp_path/'assessment_protocol.json'
    path.write_text(json.dumps(protocol))
    runner.validate_frozen(tmp_path,runner.OUT,tmp_path)
    old_hash = protocol['protocol_hash']
    old_nominations = specification['nominations'].copy()
    # This is another complete, permitted registry using the SAME R3 array.
    # Nominations and the signed specification digest deliberately stay old.
    protocol['specification']['candidate_order'] = list(neural)
    assert protocol['protocol_hash'] == old_hash
    assert protocol['specification']['nominations'] == old_nominations
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError,match='saved frozen transfer specification hash mismatch'):
        runner.validate_frozen(tmp_path,runner.OUT,tmp_path)
    # Rehashing the altered saved spec must still fail regeneration equality.
    protocol['protocol_hash'] = config_digest(protocol['specification'])
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError,match='data/specification changed'):
        runner.validate_frozen(tmp_path,runner.OUT,tmp_path)


def test_freeze_reuse_rejects_tampered_saved_specification(tmp_path,monkeypatch):
    import scripts.round2_transfer as runner
    frame = population()
    ridge = candidate_registry(('R3_ridge',))
    neural = candidate_registry(('R3_mlp',))
    directory = tmp_path/'embedding';directory.mkdir()
    (directory/'protocol.json').write_text(json.dumps(dict(specification=dict(
        selector_order=['original','R3_ridge','R3_mlp'],
        candidate_grids={'R3_ridge':list(ridge[3:]),'R3_mlp':list(neural[3:])}))))
    monkeypatch.setattr(runner,'main_gate',lambda _: {'main':'complete'})
    monkeypatch.setattr(runner,'training_frame',lambda _:frame)
    monkeypatch.setattr(runner,'preflight',lambda _: [])
    monkeypatch.setattr(runner,'load_compact_features',lambda *_:
        ({'R3':np.zeros((len(frame),2),dtype=np.float32)}, {'synthetic':True}))
    arguments = (tmp_path,runner.OUT,tmp_path,None,'synthetic amendment',('R3_ridge',))
    original = runner.freeze(*arguments)
    assert runner.freeze(*arguments) == original
    path = tmp_path/'assessment_protocol.json'
    changed = json.loads(path.read_text())
    changed['specification']['candidate_order'] = list(neural)
    assert changed['protocol_hash'] == original['protocol_hash']
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='saved frozen transfer specification changed'):
        runner.freeze(*arguments)
