import threading
import time

import pytest

from pepstab.round2_budget import BudgetExhausted, ReservationBudget


def test_reservation_settlement_releases_only_unused_cost(tmp_path):
    budget = ReservationBudget(tmp_path/"ledger.jsonl", ceiling_usd=18, call_upper_usd=1,
                               initial_reserve_usd=1.35, max_calls=30)
    first = budget.reserve(6)
    assert budget.status()["committed_usd"] == pytest.approx(7.35)
    budget.settle(first, 2)
    assert budget.status()["committed_usd"] == pytest.approx(3.35)
    second = budget.reserve(6)
    budget.burn(second)
    assert budget.status()["settled_usd"] == pytest.approx(9.35)
    assert budget.status()["calls"] == 12
    with pytest.raises(BudgetExhausted, match="quota"):
        budget.reserve(19)
    with pytest.raises(BudgetExhausted, match="worst-case"):
        budget.reserve(9)
    budget.close()


def test_threads_wait_for_release_and_never_exceed_cap(tmp_path):
    budget = ReservationBudget(tmp_path/"ledger.jsonl", ceiling_usd=10, call_upper_usd=1)
    first = budget.reserve(6)
    result = []
    def wait_for_budget():
        token = budget.reserve(6)
        result.append(budget.status()["committed_usd"])
        budget.settle(token, 2)
    worker = threading.Thread(target=wait_for_budget)
    worker.start()
    time.sleep(.02)
    assert not result
    budget.settle(first, 3)
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert result == [9]
    assert budget.status()["settled_usd"] == 5
    budget.close()


def test_restart_burns_unacknowledged_work_and_single_owner(tmp_path):
    path = tmp_path/"ledger.jsonl"
    budget = ReservationBudget(path, ceiling_usd=10, call_upper_usd=1)
    a = budget.reserve(2)
    budget.settle(a, .2)
    budget.reserve(3)
    with pytest.raises(RuntimeError, match="another controller"):
        ReservationBudget(path, ceiling_usd=10, call_upper_usd=1)
    budget.close()
    resumed = ReservationBudget(path, ceiling_usd=10, call_upper_usd=1)
    assert resumed.status()["calls"] == 5
    assert resumed.status()["settled_usd"] == pytest.approx(3.2)
    assert resumed.status()["inflight_batches"] == 0
    resumed.close()


def test_bad_telemetry_does_not_release_reservation(tmp_path):
    budget = ReservationBudget(tmp_path/"ledger.jsonl", ceiling_usd=10, call_upper_usd=1)
    token = budget.reserve(3)
    for bad in (-1, 4, float("nan")):
        with pytest.raises(ValueError, match="settlement"):
            budget.settle(token, bad)
        assert budget.status()["committed_usd"] == 3
    budget.burn(token)
    budget.close()


def test_backend_settles_success_and_burns_failed_members(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import modal_app.round2_sequence as app
    tasks = [{"context": "test", "seed": i, "stop_fold": i} for i in range(6)]
    outcomes = [{"worker_seconds": 100., "max_rss_mib": 200.} for _ in tasks]
    monkeypatch.setattr(app, "ranker_member", SimpleNamespace(map=lambda *a, **k: iter(outcomes)))
    backend = app.BoundedBackend(tmp_path)
    backend(tasks)
    per_success = 162*app.RATE_HOUR/3600
    assert backend.budget.status()["settled_usd"] == pytest.approx(1.35+6*per_success)
    outcomes[0] = RuntimeError("worker failed")
    with pytest.raises(RuntimeError, match="remote members failed"):
        backend(tasks)
    assert backend.budget.status()["settled_usd"] == pytest.approx(1.35+11*per_success+backend.call_upper)
    assert backend.budget.status()["calls"] == 12
    assert backend.budget.status()["inflight_batches"] == 0
    backend.budget.close()


def test_backend_unexpected_controller_error_releases_no_credit(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import modal_app.round2_sequence as app
    tasks = [{"context": "test", "seed": i, "stop_fold": i} for i in range(6)]
    monkeypatch.setattr(app, "ranker_member", SimpleNamespace(map=lambda *a, **k: iter(
        [{"worker_seconds": 100.} for _ in tasks])))  # Missing memory metadata.
    backend = app.BoundedBackend(tmp_path)
    with pytest.raises(KeyError):
        backend(tasks)
    assert backend.budget.status()["settled_usd"] == pytest.approx(1.35+6*backend.call_upper)
    assert backend.budget.status()["inflight_batches"] == 0
    backend.budget.close()
