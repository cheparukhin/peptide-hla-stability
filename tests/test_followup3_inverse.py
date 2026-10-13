import numpy as np
import pandas as pd
import pytest

from pepstab import followup3_inverse as study


def synthetic():
    rows = []
    for cluster in range(50):
        for allele in ("HLA-A*01:01", "HLA-B*07:02"):
            for duplicate in range(2):
                rows.append(dict(pair_id=len(rows), allele=allele, peptide=f"P{cluster:06}{duplicate}X", cluster_id=cluster,
                                 split="train", outer_fold=cluster % 5, y_log1p=float(cluster % 7 + duplicate)))
    return pd.DataFrame(rows)


def test_panel_is_label_blind_natural_key_sample_with_inherited_folds():
    frame = synthetic()
    splits = frame.drop(columns=["outer_fold", "y_log1p"])
    extra = splits.iloc[[0]].assign(pair_id=999, peptide="ZZZZZZZZZ", cluster_id=999, split="test")
    splits = pd.concat([splits, extra], ignore_index=True)
    cohort = splits[["allele", "peptide", "split"]].assign(complex_id=lambda x: x.peptide)
    folds = frame[["pair_id", "cluster_id", "outer_fold"]]
    panel = study.select_panel(splits, folds, cohort)
    assert len(panel) == 128
    assert not panel.peptide.eq("ZZZZZZZZZ").any()
    assert set(panel) == set(study.PANEL_COLUMNS)
    assert panel.groupby("cluster_id").outer_fold.nunique().max() == 1
    assert panel.folder.str.contains("HLA-A_01_01").any()
    shuffled = study.select_panel(splits.sample(frac=1, random_state=2), folds.sample(frac=1, random_state=3), cohort)
    pd.testing.assert_frame_equal(panel, shuffled)


def test_panel_rejects_cluster_leakage():
    frame = synthetic()
    folds = frame[["pair_id", "cluster_id", "outer_fold"]].copy()
    folds.loc[1, "outer_fold"] = 4
    cohort = frame[["allele", "peptide", "split"]].assign(complex_id=lambda x: x.peptide)
    with pytest.raises(ValueError, match="cluster leakage"):
        study.select_panel(frame, folds, cohort)


def test_exact_common_peptide_orders_and_visible_context_first():
    permutations = study.peptide_orders()
    positions = np.arange(10, 19)
    full = study.decoding_orders(positions, 25, permutations)
    peptide = study.decoding_orders(np.arange(9), 9, permutations)
    assert np.array_equal(full[:, -9:] - 10, peptide)
    assert all(not set(row[:-9]) & set(positions) for row in full)
    assert np.array_equal(study.peptide_orders(), permutations)
    with pytest.raises(ValueError):
        study.peptide_orders(n_orders=8)


def score_fixture():
    panel = pd.DataFrame([dict(allele="HLA-A*01:01", peptide="AAAAAAAAA", cluster_id=1, outer_fold=0,
                              complex_id="x", folder="somewhere")])
    row = dict(panel.iloc[0], status="ok", n_orders=16, decoding_seed=0,
               checkpoint_sha256=study.CHECKPOINT_SHA256, order_protocol="explicit-common-peptide-permutations-v1")
    for prefix in ("P", "C"):
        row.update({f"{prefix}_pos_{i}": -float(i) for i in range(1, 10)})
        row[f"{prefix}_total"] = -45.
    return panel, pd.DataFrame([row])


def test_scores_reject_missing_failed_protocol_drift_and_wrong_identity():
    panel, scores = score_fixture()
    assert len(study.validate_scores(panel, scores)) == 1
    for column, value in (("status", "failed"), ("n_orders", 8), ("outer_fold", 3), ("C_total", -50.)):
        invalid = scores.copy()
        invalid[column] = value
        with pytest.raises(ValueError):
            study.validate_scores(panel, invalid)
    with pytest.raises(ValueError):
        study.validate_scores(panel, scores.iloc[:0])


def test_only_nine_position_scores_enter_each_control_block():
    panel, scores = score_fixture()
    scored = study.validate_scores(panel, scores)
    arms = study.feature_arms(dict(R0=np.zeros((1, 4)), R5=np.zeros((1, 5))), scored)
    assert tuple(arms) == study.ARMS
    assert arms["R0_P"].shape == (1, 13)
    assert arms["R5_PC"].shape == (1, 23)


def test_ridge_fixed_graph_and_ties_use_stronger_regularization():
    frame = synthetic()
    x = np.column_stack([frame.y_log1p, np.arange(len(frame))])
    arms = {a: x.copy() for a in study.ARMS}
    calls = []
    def predict(x_fit, y_fit, x_held, alpha):
        calls.append((set(x_fit[:, 1]), set(x_held[:, 1])))
        assert not calls[-1][0] & calls[-1][1]
        return x_held[:, 0]
    predictions, selection, ancestry = study.nested_assessment(frame, arms, fit_predict=predict)
    assert len(calls) == len(ancestry) == 300
    assert len(selection) == 30
    assert all(s["selected_alpha"] == 10000 for s in selection)
    assert all(np.isfinite(x).all() for x in predictions.values())
    for record in ancestry:
        assert not set(record["fit_clusters"]) & set(record["held_clusters"])


def test_neural_head_never_receives_held_labels_and_exactly600members():
    frame = synthetic()
    arms = {a: np.zeros((len(frame), 2)) for a in study.ARMS}
    class FakeHead:
        def __init__(self, population, features, ancestry):
            self.members = []
        def predict(self, fit, held, candidate, context, final):
            assert not final
            assert not {"y_log1p", "thalf_hours"} & set(held)
            assert not set(fit.cluster_id) & set(held.cluster_id)
            self.members.extend([dict(seed=0), dict(seed=1)])
            return (held.cluster_id.to_numpy() % 7).astype(float)
    predictions, selection, ancestry = study.neural_assessment(frame, arms, head_factory=FakeHead)
    assert len(ancestry["members"]) == 600
    assert len(selection) == 30
    assert all(s["selected_l2"] == 1 for s in selection)
    assert all(np.isfinite(x).all() for x in predictions.values())


def test_nontrain_assessment_rejected():
    frame = synthetic()
    frame.loc[0, "split"] = "test"
    arms = {a: np.zeros((len(frame), 1)) for a in study.ARMS}
    with pytest.raises(ValueError, match="original training"):
        study.nested_assessment(frame, arms)


def test_single_worker_extract_runs_directly_without_pool(monkeypatch):
    from scripts import followup3_inverse as cli
    monkeypatch.setattr(cli, "_extract_one", lambda task: task * 2)
    def forbidden_pool(**kwargs):
        raise AssertionError("one-worker pilot must never create a process pool")
    monkeypatch.setattr(cli, "ProcessPoolExecutor", forbidden_pool)
    assert cli.run_extraction_tasks([1, 2], 1) == [2, 4]


def test_parallel_extract_explicitly_uses_clean_spawn_context(monkeypatch):
    from scripts import followup3_inverse as cli
    observed = {}
    class FakePool:
        def __init__(self, *, max_workers, mp_context):
            observed.update(workers=max_workers, method=mp_context.get_start_method())
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def map(self, function, tasks): return ["mock_result" for _ in tasks]
    monkeypatch.setattr(cli, "ProcessPoolExecutor", FakePool)
    assert cli.run_extraction_tasks([1, 2], 2) == ["mock_result", "mock_result"]
    assert observed == dict(workers=2, method="spawn")
