import numpy as np
import pandas as pd
import pytest

from pepstab import evaluation as ev
from pepstab.round2_reporting import _weighted_rho, compare_predictions
from pepstab.legacy_assessment import refuse_legacy_assessment


def test_weighted_ranks_equal_expanded_rows_with_ties_and_missing_groups():
    y = np.array([0, 0, 1, 2, 2, 5.])
    p = np.array([3, 1, 1, 2, 0, 2.])
    w = np.array([[2, 0, 1, 4, 2, 1], [0, 0, 2, 0, 1, 0], [2, 0, 0, 0, 0, 0]])
    got = _weighted_rho(y, p, w)
    from scipy.stats import spearmanr
    for i in range(2):
        idx = np.repeat(np.arange(len(y)), w[i])
        assert got[i] == pytest.approx(spearmanr(y[idx], p[idx]).statistic, abs=1e-14)
    assert np.isnan(got[2])
    assert _weighted_rho(y, np.ones(6), w)[0] == 0


def test_batched_bootstrap_matches_frozen_procedure():
    rng = np.random.default_rng(47)
    n = 160
    frame = pd.DataFrame({"pair_id": np.arange(n), "cluster_id": np.arange(n) % 31,
                          "allele": np.array(["a", "b", "c", "d"])[np.arange(n) % 4],
                          "y_log1p": rng.integers(0, 4, n).astype(float)})
    a, b = rng.integers(0, 5, n).astype(float), rng.normal(size=n)
    b[frame.allele == "c"] = 0
    out, _, meta = compare_predictions(frame, {"a": a, "b": b}, [("a", "b")], n_boot=41)
    reference = ev.paired_cluster_bootstrap(frame.cluster_id, frame.allele,
                                           frame.y_log1p, a, b,
                                           alleles=meta["alleles"], n_boot=41)
    assert out.iloc[0].delta == pytest.approx(reference["delta_median_spearman"], abs=1e-14)
    assert out.iloc[0].ci_low == pytest.approx(reference["ci95"][0], abs=1e-14)
    assert out.iloc[0].ci_high == pytest.approx(reference["ci95"][1], abs=1e-14)
    assert meta["n_resamples_with_dropped_alleles"] == reference["n_resamples_with_dropped_alleles"]


def test_retired_generator_cannot_publish():
    with pytest.raises(SystemExit, match="retired historical assessment"):
        refuse_legacy_assessment("legacy.py")


@pytest.mark.parametrize("script", ["phase_a_select", "phase_b_analyse",
                                   "phase_c_blend_gate_check", "phase_c_frozen_comparison",
                                   "phase_d_structural_residual"])
def test_historical_main_stops_before_loading_or_writing(script):
    from importlib import import_module
    with pytest.raises(SystemExit, match="retired historical assessment"):
        import_module(f"scripts.{script}").main()


def test_rejects_duplicate_identity():
    frame = pd.DataFrame({"pair_id": [1, 1], "cluster_id": [1, 2],
                          "allele": ["a", "a"], "y_log1p": [0, 1]})
    with pytest.raises(ValueError, match="unique canonical"):
        compare_predictions(frame, {"a": np.array([0, 1])}, [("a", "a")])
