"""Mock-only tests. No Modal subprocesses, process lookups or real signals."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

from scripts import final_runtime_guard as guard

NOW = 1_800_000_000.
START = "Fri Jan 15 08:00:00 2027"


def utc(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


class FakeSystem:
    def __init__(self):
        self.actions = []
        self.controller_status = "running"
        self.apps = [{"app_id": "ap-target", "state": "ephemeral"},
                     {"app_id": "ap-unrelated", "state": "ephemeral"}]
        self.fail_stops = 0
        self.fail_list = False
        self.before_signal = None
        self.before_stop = None
        self.ignore_successful_stop = False

    def run(self, argv, **kwargs):
        assert kwargs["timeout"] <= 30
        self.actions.append(tuple(argv))
        if argv[0] == "ps":
            if self.controller_status == "done":
                return SimpleNamespace(returncode=1, stdout="", stderr="")
            pid, pgid, start, command = 100001, 100001, START, "study --locked"
            if self.controller_status == "wrong_group":
                pgid = 55
            if self.controller_status == "wrong_command":
                command = "unrelated"
            if self.controller_status == "reused_same_command":
                start = "Fri Jan 15 09:00:00 2027"
            return SimpleNamespace(returncode=0, stdout=f"{pid} {pgid} {start} {command}\n", stderr="")
        if argv[1:3] == ["app", "list"]:
            if self.fail_list:
                raise TimeoutError("list unavailable")
            return SimpleNamespace(returncode=0, stdout=json.dumps(self.apps), stderr="")
        assert argv[1:3] == ["app", "stop"] and argv[3] in ("ap-target", "ap-main")
        if self.before_stop:
            self.before_stop()
        if self.fail_stops:
            self.fail_stops -= 1
            raise TimeoutError("stop unavailable")
        if not self.ignore_successful_stop:
            for app in self.apps:
                if app.get("app_id", app.get("App ID")) == argv[3]:
                    app["state"] = "stopped"
        return SimpleNamespace(returncode=0, stdout="stopped", stderr="")

    def killpg(self, pid, sig):
        assert pid == 100001
        if self.before_signal:
            self.before_signal(sig)
        self.actions.append(("signal", pid, sig))
        if sig == signal.SIGCONT:
            self.controller_status = "done"


@pytest.fixture
def setup(tmp_path):
    source = tmp_path / "controller.py"
    protocol = tmp_path / "protocol.json"
    source.write_text("# locked controller\n")
    protocol.write_text('{"locked": true}\n')
    auth = {"schema": guard.SCHEMA, "phase_id": "final100", "scope_id": "smoke1", "scope_kind": "smoke",
            "phase_ceiling_usd": 100., "total_allocation_usd": 65., "prior_charges_usd": 0.,
            "modal_cli": str(tmp_path / "modal"), "app_id": "ap-target", "controller_pid": 100001,
            "controller_command": "study --locked", "controller_started_at": START,
            "ledger_path": str(tmp_path / "ledger.jsonl"), "execution_log_paths": [str(tmp_path / "execution.log")],
            "poll_seconds": 5., "command_timeout_seconds": 5., "max_concurrency": 30,
            "cpu_cores": 2, "memory_mib": 4096, "rate_usd_per_second": guard.RATE_USD_PER_SECOND,
            "call_timeout_seconds": 1800., "per_call_overhead_seconds": 60., "per_call_upper_usd": .1,
            "initial_reserve_usd": 1., "stop_commitment_usd": 55., "shutdown_cushion_usd": 2.,
            "max_app_stop_attempts": 3, "authorized_at_utc": utc(NOW),
            "startup_deadline_utc": utc(NOW + 30), "deadline_utc": utc(NOW + 3600),
            "source_hashes": {str(source): guard.file_hash(source), str(Path(guard.__file__).resolve()): guard.file_hash(guard.__file__)},
            "protocol_hashes": {str(protocol): guard.file_hash(protocol)}}
    for key in ("status_path", "event_log_path", "stop_flag_path", "scope_lock_path", "global_lock_path", "global_status_path"):
        auth[key] = str(tmp_path / key)
    Path(auth["execution_log_paths"][0]).write_text("")
    write_ledger(auth)
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(auth))
    return auth, path, FakeSystem()


def write_ledger(auth, records=(), tail=b""):
    config = {"event": "configuration", "schema": "round2_budget.1", "ceiling_usd": guard.scope_limit(auth),
              "call_upper_usd": auth["per_call_upper_usd"], "initial_reserve_usd": auth["initial_reserve_usd"], "max_calls": 10000}
    payload = "".join(json.dumps(r) + "\n" for r in (config, *records)).encode() + tail
    Path(auth["ledger_path"]).write_bytes(payload)


def reserve(n, token="a"):
    return {"event": "reservation", "token": token, "calls": n, "reserved_usd": n * .1}


def build(setup, now=lambda: NOW):
    _, path, system = setup
    return guard.RuntimeGuard(path, runner=system.run, signal_group=system.killpg, now=now, sleep=lambda _: None)


@pytest.mark.parametrize("value", [True, False, None, float("nan"), float("inf"), -1, "NaN", "bad"])
def test_amount_rejects_nonfinite_and_bad_inputs(value):
    with pytest.raises(ValueError, match="invalid"):
        guard.amount(value, "money")


@pytest.mark.parametrize("key,value", [("max_concurrency", 31), ("max_concurrency", True), ("cpu_cores", 4),
    ("memory_mib", 2048), ("phase_ceiling_usd", 101), ("shutdown_cushion_usd", 0), ("per_call_upper_usd", .00001),
    ("stop_commitment_usd", 64), ("rate_usd_per_second", .001), ("deadline_utc", utc(NOW + 100000))])
def test_invalid_authorization_budget_and_resources(setup, key, value):
    auth = deepcopy(setup[0])
    auth[key] = value
    with pytest.raises(ValueError):
        guard.validate_authorization(auth)


def test_scope_allocation_debits_shared_prior_and_accepts18_workers(setup):
    auth = deepcopy(setup[0])
    auth.update(max_concurrency=18, prior_charges_usd=1.35)
    assert guard.scope_limit(guard.validate_authorization(auth)) == 63.65
    auth["prior_charges_usd"] = 66.
    with pytest.raises(ValueError, match="global phase ceiling"):
        guard.validate_authorization(auth)


def test_replay_retains_partial_settlement_and_full_failed_call_reserve(setup):
    auth = setup[0]
    write_ledger(auth, [reserve(3)], b'{"event":"settlement","token":"a"')
    value = guard.replay_commitments(auth["ledger_path"], auth)
    assert value["ordinary_commitment_usd"] == pytest.approx(1.3)
    assert value["inflight_calls"] == 3 and value["partial_tail_bytes"] > 0
    write_ledger(auth, [reserve(3), {"event": "settlement", "token": "a", "calls": 3, "charged_usd": .3, "outcome": "preempted"}])
    assert guard.replay_commitments(auth["ledger_path"], auth)["ordinary_commitment_usd"] == pytest.approx(1.3)
    write_ledger(auth, [reserve(3), {"event": "settlement", "token": "a", "calls": 3, "charged_usd": .1, "outcome": "failure"}])
    with pytest.raises(ValueError, match="full reservation"):
        guard.replay_commitments(auth["ledger_path"], auth)


def test_preemption_counts_prefixed_and_partial_warning_and_triggers_threshold(setup):
    auth, _, system = setup
    write_ledger(auth, [reserve(538)])
    log = Path(auth["execution_log_paths"][0])
    log.write_text("[worker] " + guard.PREEMPTION_LINE + "\n" + guard.PREEMPTION_LINE)
    runtime = build(setup)
    try:
        assert runtime.tick()
        assert runtime.state["ordinary_commitment_usd"] == pytest.approx(54.8)
        assert runtime.state["preemption_count"] == 2
        assert runtime.state["effective_commitment_usd"] == pytest.approx(55.)
        assert runtime.state["stop_intent"]["reason"] == "commitment_burst_cushion_threshold"
        assert system.apps[1]["state"] == "ephemeral"
    finally:
        runtime.close()


def test_stopflag_and_intent_precede_freeze_and_cloud_mutations(setup):
    auth, _, system = setup
    write_ledger(auth, [reserve(540)])
    ledger_before = Path(auth["ledger_path"]).read_bytes()
    def check(*_):
        flag = json.loads(Path(auth["stop_flag_path"]).read_text())
        status = json.loads(Path(auth["status_path"]).read_text())
        assert flag["authorization_sha256"] == status["authorization_sha256"]
        assert status["stop_intent"]["app_id"] == "ap-target"
    system.before_signal = check
    system.before_stop = check
    runtime = build(setup)
    try:
        assert runtime.tick()
        mutations = [a for a in system.actions if a[0] == "signal" or a[1:3] == ("app", "stop")]
        assert mutations == [("signal", 100001, signal.SIGSTOP),
            (auth["modal_cli"], "app", "stop", "ap-target", "--yes"),
            ("signal", 100001, signal.SIGTERM), ("signal", 100001, signal.SIGCONT)]
        assert Path(auth["ledger_path"]).read_bytes() == ledger_before
    finally:
        runtime.close()


@pytest.mark.parametrize("identity", ["wrong_group", "wrong_command", "reused_same_command"])
def test_invalid_process_identity_never_signaled_exact_app_still_stopped(setup, identity):
    setup[2].controller_status = identity
    runtime = build(setup)
    try:
        assert not runtime.tick()
        assert not any(a[0] == "signal" for a in setup[2].actions)
        assert runtime.state["app_stop_command_succeeded"]
        assert runtime.state["stop_intent"]["reason"] == "controller_identity_unknown"
    finally:
        runtime.close()


def test_fixed_deadline_stops_before_app_list_despite_broken_all_telemetry(setup):
    auth, path, system = setup
    runtime = build(setup, now=lambda: NOW + 3601)
    path.unlink()
    Path(auth["ledger_path"]).write_text("broken")
    system.fail_list = True
    system.fail_stops = 100
    try:
        assert not runtime.tick()
        assert runtime.state["stop_intent"]["reason"] == "immutable_wall_deadline"
        assert runtime.state["phase"] == "stop_failed_manual_action"
        actions = system.actions
        first_stop = next(i for i, a in enumerate(actions) if a[1:3] == ("app", "stop"))
        first_list = next(i for i, a in enumerate(actions) if a[1:3] == ("app", "list"))
        assert first_stop < first_list and runtime.state["app_stop_attempts"] == 3
        assert runtime.state["stop_result"]["terminated"]
    finally:
        runtime.close()


def test_single_owner_global_lock_prevents_different_parallel_scope(setup):
    runtime = build(setup)
    auth, path, _ = setup
    try:
        replacement = deepcopy(auth)
        replacement["scope_id"] = "other"
        replacement["scope_lock_path"] += ".other"
        other = path.with_name("other-auth.json")
        other.write_text(json.dumps(replacement))
        with pytest.raises(RuntimeError, match="parallel scopes"):
            guard.RuntimeGuard(other, runner=setup[2].run, signal_group=setup[2].killpg, now=lambda: NOW)
    finally:
        runtime.close()


def test_restart_authorization_change_cannot_extend_deadline(setup):
    runtime = build(setup)
    runtime.close()
    auth, path, _ = setup
    auth["deadline_utc"] = utc(NOW + 7200)
    path.write_text(json.dumps(auth))
    with pytest.raises(ValueError, match="stopped/reconciled predecessor"):
        build(setup)


def test_authorization_and_source_drift_failclosed_using_original_scope(setup):
    runtime = build(setup)
    replacement = deepcopy(setup[0])
    replacement["app_id"] = "ap-unrelated"
    replacement["deadline_utc"] = utc(NOW + 7200)
    setup[1].write_text(json.dumps(replacement))
    try:
        assert runtime.tick()
        assert runtime.state["stop_intent"]["reason"] == "immutable_authorization_or_source_drift"
        assert setup[2].apps[1]["state"] == "ephemeral"
    finally:
        runtime.close()


def test_frozen_source_content_drift_is_detected_without_authorization_edit(setup):
    runtime = build(setup)
    source = next(path for path in setup[0]["source_hashes"] if path != str(Path(guard.__file__).resolve()))
    Path(source).write_text("# changed controller\n")
    try:
        assert runtime.tick()
        assert runtime.state["stop_intent"]["reason"] == "immutable_authorization_or_source_drift"
    finally:
        runtime.close()


def test_stop_retries_bounded_and_attempt_count_survives_restart(setup):
    setup[2].fail_stops = 100
    write_ledger(setup[0], [reserve(540)])
    runtime = build(setup)
    try:
        assert runtime.run() == 2
        assert runtime.state["app_stop_attempts"] == 3
    finally:
        runtime.close()
    restarted = build(setup)
    try:
        assert not restarted.tick()
        assert restarted.state["app_stop_attempts"] == 3
        assert len([a for a in setup[2].actions if a[1:3] == ("app", "stop")]) == 3
        assert Path(setup[0]["stop_flag_path"]).exists()
    finally:
        restarted.close()


def test_transient_stop_failures_recover_only_within_bound(setup):
    setup[2].fail_stops = 2
    write_ledger(setup[0], [reserve(540)])
    runtime = build(setup)
    try:
        assert runtime.tick()
        assert runtime.state["app_stop_attempts"] == 3
        assert runtime.state["app_stop_command_succeeded"]
    finally:
        runtime.close()


def test_success_exit_without_observed_stop_retries_bounded_then_requires_manual_action(setup):
    setup[2].ignore_successful_stop = True
    write_ledger(setup[0], [reserve(540)])
    runtime = build(setup)
    try:
        for _ in range(3):
            assert not runtime.tick()
        assert runtime.state["app_stop_attempts"] == 3
        assert runtime.state["app_stop_command_succeeded"]
        assert runtime.state["phase"] == "stop_failed_manual_action"
        runtime.tick()
        assert len([a for a in setup[2].actions if a[1:3] == ("app", "stop")]) == 3
    finally:
        runtime.close()


@pytest.mark.parametrize("failure", ["corrupt_complete", "truncate", "log_truncate", "ledger_remove", "unknown_state", "missing_seen_app"])
def test_unverifiable_telemetry_failclosed(setup, failure):
    auth, _, system = setup
    runtime = build(setup)
    try:
        assert not runtime.tick()
        if failure == "corrupt_complete":
            with Path(auth["ledger_path"]).open("a") as handle:
                handle.write("broken\n")
        elif failure == "truncate":
            Path(auth["ledger_path"]).write_text("")
        elif failure == "log_truncate":
            log = Path(auth["execution_log_paths"][0])
            log.write_text(guard.PREEMPTION_LINE + "\n")
            runtime.tick()
            log.write_text("")
        elif failure == "ledger_remove":
            Path(auth["ledger_path"]).unlink()
        elif failure == "unknown_state":
            system.apps[0]["state"] = "unrecognized"
        elif failure == "missing_seen_app":
            system.apps = system.apps[1:]
        runtime.tick()
        assert runtime.state["stop_intent"] is not None
        assert Path(auth["stop_flag_path"]).exists()
    finally:
        runtime.close()


def test_fixed_startup_window_cannot_be_reset_by_polling(setup):
    setup[2].apps = setup[2].apps[1:]
    clock = [NOW]
    runtime = build(setup, now=lambda: clock[0])
    try:
        assert not runtime.tick() and runtime.state["stop_intent"] is None
        clock[0] = NOW + 31
        runtime.tick()
        assert runtime.state["stop_intent"] is not None
    finally:
        runtime.close()


def test_smoke_to_main_requires_stopped_status_hash_and_shared_prior(setup):
    auth, path, system = setup
    runtime = build(setup)
    system.apps[0]["state"] = "stopped"
    system.controller_status = "done"
    assert runtime.tick()
    runtime.close()
    old_status = Path(auth["status_path"])
    record = {"schema": "final_scope_reconciliation.1", "authorization_sha256": guard.file_hash(path),
              "guard_status_sha256": guard.file_hash(old_status), "cumulative_prior_charges_usd": 1.35}
    reconciliation = path.with_name("reconciled.json")
    reconciliation.write_text(json.dumps(record))
    main = deepcopy(auth)
    main.update(scope_kind="main", scope_id="main1", app_id="ap-main", max_concurrency=18, prior_charges_usd=1.35)
    for key in ("status_path", "event_log_path", "stop_flag_path", "scope_lock_path", "ledger_path"):
        main[key] += ".main"
    main["execution_log_paths"] = [auth["execution_log_paths"][0] + ".main"]
    Path(main["execution_log_paths"][0]).write_text("")
    main["prior_reconciliation"] = {"path": str(reconciliation), "sha256": guard.file_hash(reconciliation)}
    write_ledger(main)
    main_path = path.with_name("main.json")
    main_path.write_text(json.dumps(main))
    next_guard = guard.RuntimeGuard(main_path, runner=system.run, signal_group=system.killpg, now=lambda: NOW)
    try:
        assert guard.scope_limit(next_guard.auth) == pytest.approx(63.65)
    finally:
        next_guard.close()


def test_main_without_prior_reconciliation_rejected_and_validate_only_has_no_system_calls(setup):
    auth, path, system = setup
    wrong = deepcopy(auth)
    wrong["scope_kind"] = "main"
    with pytest.raises(ValueError, match="requires stopped/reconciled"):
        guard.validate_authorization(wrong)
    assert guard.main(["--authorization", str(path), "--validate-only"]) == 0
    assert not system.actions and not Path(auth["status_path"]).exists()
