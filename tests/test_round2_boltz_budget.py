import pytest

from scripts.round2_boltz_budget import BudgetLedger


def test_reservations_prevent_parallel_overspend_and_release_unused_budget():
    ledger = BudgetLedger(1.)
    ledger.reserve("0", .4)
    ledger.reserve("1", .4)
    with pytest.raises(ValueError, match="remaining budget"):
        ledger.reserve("2", .4)
    ledger.settle("0", .15, {"status": "passed"})
    ledger.reserve("2", .4)
    assert ledger.state["spent_upper_usd"] + sum(ledger.state["reservations"].values()) <= 1.
    ledger.fail("1", "timeout")
    assert ledger.state["spent_upper_usd"] == .55
    with pytest.raises(ValueError, match="remaining budget"):
        ledger.reserve("3", .4)


def test_unknown_failure_debits_full_reservation_and_no_silent_budget_increase():
    ledger = BudgetLedger(.5)
    ledger.reserve("x", .4)
    ledger.fail("x", "connection lost")
    with pytest.raises(ValueError, match="cannot change"):
        BudgetLedger(1., ledger.state)
    with pytest.raises(ValueError, match="duplicate"):
        ledger.reserve("x", .05)


@pytest.mark.parametrize("charge", [float("nan"), float("inf"), -.1, .6])
def test_invalid_accounting_durably_stops_dispatch(charge):
    ledger = BudgetLedger(1.)
    ledger.reserve("0", .5)
    with pytest.raises(ValueError, match="measured charge"):
        ledger.settle("0", charge, {"status": "passed"})
    resumed = BudgetLedger(1., ledger.state)
    with pytest.raises(ValueError, match="halted"):
        resumed.reserve("1", .1)
    assert ledger.state["completed"]["0"]["status"] == "failed"


def test_reduced_cohort_keeps_whole_frozen_training_clusters(tmp_path):
    from scripts.round2_boltz_cohort import freeze_cohort, cases_for_cohort
    from pepstab.data import load_with_splits
    manifest = freeze_cohort(tmp_path, max_rows=100)
    cases = cases_for_cohort(tmp_path / "cohort.csv")
    assert 0 < len(cases) <= 100 and len(cases) == manifest["n_rows"]
    ids = {c["pair_id"] for c in cases}
    clusters = {c["cluster_id"] for c in cases}
    data = load_with_splits()
    assert set(data.loc[data.cluster_id.isin(clusters) & data.split.eq("train"), "pair_id"]) == ids
    assert all(c["split"] == "train" for c in cases)
    assert not any("thalf_hours" in c or "y_log1p" in c for c in cases)
    with pytest.raises(FileExistsError, match="already frozen"):
        freeze_cohort(tmp_path, max_rows=100)
