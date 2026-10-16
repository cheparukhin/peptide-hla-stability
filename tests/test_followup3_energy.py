import ast
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest
from pepstab import followup3_energy as energy


def test_global_and_case_deadlines_are_both_binding(monkeypatch):
    monkeypatch.setattr(energy.time, "time", lambda: 1000.)
    monkeypatch.setattr(energy.time, "monotonic", lambda: 200.)
    assert energy.remaining_seconds("1970-01-01T00:17:30Z", 100.) == 50.
    assert energy.remaining_seconds("1970-01-01T00:30:00Z", 100.) == 500.
    with pytest.raises(TimeoutError):
        energy.remaining_seconds("1970-01-01T00:10:00Z", 100.)
    with pytest.raises(ValueError):
        energy.utc_deadline("2026-10-13T12:00:00")


def pilot_fixture():
    panel = pd.DataFrame([dict(allele=f"A{i}", peptide=f"P{i}", folder=f"f{i}") for i in range(32)])
    cases = [dict(row, status="ok", binary_sha256=energy.BINARY_SHA256, structure_sha256="source", metadata_sha256="metadata", repaired_pdb_sha256="repaired",
                  repaired=dict(terms=dict(foldx_group1="C", foldx_group2="A", foldx_interaction_energy=-12., foldx_van_der_waals_clashes=1.))) for row in panel.to_dict(orient="records")]
    return panel, cases


def test_all32_gate_and_complete_finite_schema():
    panel, cases = pilot_fixture()
    assert energy.pilot_schema(cases, panel) == ["foldx_interaction_energy", "foldx_van_der_waals_clashes"]
    with pytest.raises(ValueError): energy.pilot_schema(cases[:31], panel)
    cases[0]["repaired"]["terms"]["foldx_van_der_waals_clashes"] = "nan"
    with pytest.raises(ValueError, match="nonfinite/non-numeric"): energy.pilot_schema(cases, panel)


@pytest.mark.parametrize("learner,expected", [("mlp", 40), ("ridge", 20)])
def test_only_two_new_arms_and_exact_bounded_member_counts(learner, expected):
    rows = [dict(pair_id=i * 4 + a * 2 + d, allele=f"A{a}", peptide=f"P{i}_{d}", cluster_id=i, outer_fold=i % 5,
                 split="train", y_log1p=float(i % 7 + d)) for i in range(50) for a in range(2) for d in range(2)]
    frame = pd.DataFrame(rows)
    class FakeHead:
        def __init__(self, population, features, ancestry):
            self.members = []
            assert tuple(features) == energy.ARMS
        def predict(self, fit, held, candidate, context, final):
            assert not final and "y_log1p" not in held
            assert not set(fit.cluster_id) & set(held.cluster_id)
            self.members.extend([dict(seed=s) for s in range(2 if candidate["learner"] == "mlp" else 1)])
            return held.cluster_id.to_numpy() % 7
    base = dict(R0=np.zeros((len(frame), 2)), R5=np.zeros((len(frame), 3)))
    _, predictions, choices, ancestry = energy.assess_heads(frame, base, np.zeros((len(frame), 4)), learner, 0, head_factory=FakeHead)
    assert set(predictions) == set(energy.ARMS)
    assert len(ancestry["members"]) == expected
    assert len(choices) == 2
    assert all(choice["selected_regularization"] == (1. if learner == "mlp" else 10000.) for choice in choices)


def test_worker_hard_resource_caps_and_start_before_compute_commit():
    path = Path(__file__).resolve().parents[1] / "modal_app/followup3_energy.py"
    source = path.read_text()
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "score_one")
    kwargs = {keyword.arg:ast.literal_eval(keyword.value) for keyword in function.decorator_list[0].keywords if keyword.arg in ("cpu", "memory", "max_containers", "timeout", "retries")}
    assert kwargs == dict(cpu=(1., 1.), memory=(2048, 2048), max_containers=100, timeout=660, retries=0)
    assert source.index('with attempted.open("x")') < source.index("outputs.commit()") < source.index("result = core.score_repaired")
    assert 'status="interrupted_attempt"' in source
    assert source.index("if attempted.exists()") < source.index("result = core.score_repaired")


def test_ridge_keeps_float64_and_matches_original_scaler_lsqr_recipe():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from pepstab.round2_nested import Ancestry, unlabelled
    frame = pd.DataFrame(dict(pair_id=np.arange(8), cluster_id=np.arange(8), split="train", y_log1p=np.arange(8, dtype=float)))
    x = np.column_stack([1e8 + np.arange(8) * .001, np.arange(8) ** 2]).astype(np.float64)
    fit, held = frame.iloc[:6], unlabelled(frame.iloc[6:])
    head = energy.Float64RidgeHead(frame, {"R5_E":x}, Ancestry())
    candidate = dict(id="test", representation="R5_E", regularization=100.)
    actual = head.predict(fit, held, candidate, "test")
    expected = make_pipeline(StandardScaler(), Ridge(alpha=100., solver="lsqr", tol=1e-5, max_iter=2000)).fit(x[:6], np.arange(6, dtype=float)).predict(x[6:])
    assert head.features["R5_E"].dtype == np.float64
    assert np.array_equal(actual, expected)


def test_reused_csv_predictions_round_trip_exactly(tmp_path):
    values = np.asarray([.12345678901234567, 1.0000000000000002, 1.8765432109876543])
    path = tmp_path / "predictions.csv"
    pd.DataFrame(dict(R5=values)).to_csv(path, index=False)
    assert np.array_equal(pd.read_csv(path, float_precision="round_trip").R5, values)


def test_shared_access_gate_stops_before_metadata_or_executable(tmp_path):
    def denied():
        raise PermissionError("shared access gate failed")
    result = energy.score_repaired(dict(allele="A", peptide="P"), tmp_path / "missing", {"sha256":energy.BINARY_SHA256},
                                   "2099-01-01T00:00:00Z", [], access_stop_check=denied)
    assert result["status"] == "access_rejected"
    assert result["phases"] == []
