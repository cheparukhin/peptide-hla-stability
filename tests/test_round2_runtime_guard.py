"""Mock-only tests: never invoke Modal, ps, or real process-group signals."""
from datetime import datetime, timezone
import json
import signal
from types import SimpleNamespace

import pytest

from scripts import round2_runtime_guard as guard


NOW = 1_800_000_000.


class FakeSystem:
    def __init__(self):
        self.actions = []
        self.controllers = {100001: "running", 100002: "running"}
        self.apps = [{"App ID": "ap-cache", "State": "ephemeral"},
                     {"App ID": "ap-sequence", "State": "ephemeral"},
                     {"App ID": "ap-unrelated", "State": "ephemeral"}]
        self.fail_list = False
        self.fail_stop = False
        self.before_signal = None

    def run(self, argv, **kwargs):
        assert 0 < kwargs["timeout"] <= 30
        self.actions.append(tuple(argv))
        if argv[0] == "ps":
            pid = int(argv[2])
            state = self.controllers[pid]
            if state == "done":
                return SimpleNamespace(returncode=1, stdout="", stderr="")
            if state == "wrong_group":
                stdout = f"{pid} 9 study-{pid} --approved\n"
            elif state == "wrong_command":
                stdout = f"{pid} {pid} unrelated-process\n"
            else:
                stdout = f"{pid} {pid} study-{pid} --approved\n"
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        if argv[1:3] == ["app", "list"]:
            if self.fail_list:
                raise TimeoutError("list unavailable")
            return SimpleNamespace(returncode=0, stdout=json.dumps(self.apps), stderr="")
        assert argv[1:3] == ["app", "stop"]
        assert argv[3] in {"ap-cache", "ap-sequence"}
        assert argv[4:] == ["--yes"]
        if self.fail_stop:
            raise TimeoutError("stop unavailable")
        for app in self.apps:
            if app["App ID"] == argv[3]:
                app["State"] = "stopped"
        return SimpleNamespace(returncode=0, stdout="stopped", stderr="")

    def killpg(self, pid, sig):
        assert pid in {100001, 100002}
        if self.before_signal:
            self.before_signal(pid, sig)
        self.actions.append(("signal", pid, sig))
        if sig == signal.SIGCONT:
            self.controllers[pid] = "done"


@pytest.fixture
def setup(tmp_path):
    config = {"schema": guard.SCHEMA, "modal_cli": str(tmp_path / "modal"),
              "status_path": str(tmp_path / "guard-status.json"),
              "event_log_path": str(tmp_path / "guard-events.json"),
              "lock_path": str(tmp_path / "guard.lock"),
              "poll_seconds": 30, "command_timeout_seconds": 5, "studies": {}}
    for name, pid, bound, cap, threshold in [
            ("cache", 100001, 1., 80., 70.), ("sequence", 100002, .1, 18., 17.)]:
        ledger = tmp_path / f"{name}-ledger"
        log = tmp_path / f"{name}.log"
        log.write_text("")
        study = {"app_id": f"ap-{name}", "controller_pid": pid,
                 "controller_command": f"study-{pid} --approved",
                 "ledger_path": str(ledger), "execution_log_paths": [str(log)],
                 "limit_usd": cap, "rate_usd_per_second": .001,
                 "max_concurrency": 8, "per_call_upper_usd": bound,
                 "stop_commitment_usd": threshold,
                 "deadline_utc": datetime.fromtimestamp(NOW + 3600, timezone.utc).isoformat()}
        config["studies"][name] = study
        if name == "cache":
            ledger.write_text(json.dumps({"limit_usd": cap, "reservation_per_call_usd": bound,
                                          "spent_upper_usd": 1, "reservations": {"one": bound}}))
        else:
            ledger.write_text(json.dumps({"event": "configuration", "schema": "round2_budget.1",
                                          "ceiling_usd": cap, "call_upper_usd": bound,
                                          "initial_reserve_usd": 1.35}) + "\n")
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(config))
    system = FakeSystem()
    return config, path, system


def build(setup, now=lambda: NOW):
    config, path, system = setup
    return guard.RuntimeGuard(path, runner=system.run, signal_group=system.killpg, now=now)


def cache_spent(setup, spent):
    path = setup[0]["studies"]["cache"]["ledger_path"]
    value = json.loads(open(path).read())
    value["spent_upper_usd"] = spent
    with open(path, "w") as handle:
        json.dump(value, handle)


def test_threshold_counts_reservations_and_exact_preemption_lines(setup):
    config, path, system = setup
    cache_spent(setup, 67)
    log = config["studies"]["cache"]["execution_log_paths"][0]
    with open(log, "w") as handle:
        handle.write(guard.PREEMPTION_LINE + " extra\n" + guard.PREEMPTION_LINE + "\n")
    runtime = build(setup)
    try:
        assert not runtime.tick()
        state = runtime.state["studies"]["cache"]
        assert state["effective_commitment_usd"] == 69
        assert state["preemption_count"] == 1
        assert not any(action[1:3] == ("app", "stop") for action in system.actions)
        with open(log, "a") as handle:
            handle.write(guard.PREEMPTION_LINE + "\n")
        runtime.tick()
        assert state["effective_commitment_usd"] == 70
        assert state["stop_intent"]["reason"] == "commitment_cushion_threshold"
        assert state["stop_result"]["app_stop_succeeded"]
    finally:
        runtime.close()


def test_sequence_replay_keeps_partial_settlement_reserved(setup):
    config, _, _ = setup
    study = config["studies"]["sequence"]
    records = [
        {"event": "reservation", "token": "a", "calls": 6, "reserved_usd": .6},
        {"event": "reservation", "token": "b", "calls": 6, "reserved_usd": .6},
        {"event": "settlement", "token": "a", "charged_usd": .2},
    ]
    with open(study["ledger_path"], "a") as handle:
        handle.write("".join(json.dumps(record) + "\n" for record in records))
        handle.write('{"event": "settlement", "token": "b"')
    assert guard.sequence_commitments(study["ledger_path"], study) == pytest.approx(2.15)
    records.append({"event": "settlement", "token": "a", "charged_usd": .1})
    with open(study["ledger_path"], "w") as handle:
        handle.write(json.dumps({"event": "configuration", "schema": "round2_budget.1",
                                 "ceiling_usd": 18, "call_upper_usd": .1,
                                 "initial_reserve_usd": 1.35}) + "\n")
        handle.write("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(KeyError):
        guard.sequence_commitments(study["ledger_path"], study)


def test_single_owner_and_bound_authorization(setup):
    runtime = build(setup)
    try:
        with pytest.raises(RuntimeError, match="another runtime guard"):
            build(setup)
    finally:
        runtime.close()
    setup[0]["studies"]["cache"]["deadline_utc"] = "2027-01-01T00:00:00Z"
    setup[1].write_text(json.dumps(setup[0]))
    with pytest.raises(ValueError, match="different authorization"):
        build(setup)


def test_stop_persists_intent_then_freezes_cloud_stops_and_terminates_only_scope(setup):
    config, _, system = setup
    cache_spent(setup, 69)
    runtime = build(setup)
    ledger_before = open(config["studies"]["cache"]["ledger_path"], "rb").read()
    def check_intent(pid, sig):
        state = json.loads(open(config["status_path"]).read())
        assert state["studies"]["cache"]["stop_intent"]["app_id"] == "ap-cache"
    system.before_signal = check_intent
    try:
        runtime.tick()
        actions = [action for action in system.actions if action[0] != "ps"]
        assert actions[:4] == [("signal", 100001, signal.SIGSTOP),
                               (config["modal_cli"], "app", "stop", "ap-cache", "--yes"),
                               ("signal", 100001, signal.SIGTERM),
                               ("signal", 100001, signal.SIGCONT)]
        assert open(config["studies"]["cache"]["ledger_path"], "rb").read() == ledger_before
        assert system.apps[-1]["State"] == "ephemeral"
        assert system.controllers[100002] == "running"
        events = json.loads(open(config["event_log_path"]).read())
        assert [event["event"] for event in events] == ["stop_intent", "stop_result"]
    finally:
        runtime.close()


@pytest.mark.parametrize("identity", ["wrong_group", "wrong_command"])
def test_invalid_controller_identity_never_receives_signals_but_exact_app_stops(setup, identity):
    cache_spent(setup, 69)
    setup[2].controllers[100001] = identity
    runtime = build(setup)
    try:
        runtime.tick()
        assert not any(action[0] == "signal" for action in setup[2].actions)
        state = runtime.state["studies"]["cache"]
        assert state["stop_result"]["app_stop_succeeded"]
        assert state["stop_result"]["errors"]
        assert runtime.state["phase"] == "degraded"
    finally:
        runtime.close()


def test_deadline_stops_despite_failed_ledger_api_and_deleted_authorization(setup):
    config, path, system = setup
    runtime = build(setup, now=lambda: NOW + 3601)
    path.unlink()
    for study in config["studies"].values():
        with open(study["ledger_path"], "w") as handle:
            handle.write("broken")
    system.fail_list = True
    system.fail_stop = True
    try:
        assert not runtime.tick()
        assert runtime.state["phase"] == "degraded"
        for state in runtime.state["studies"].values():
            assert state["stop_intent"]["reason"] == "immutable_wall_deadline"
            assert state["stop_result"]["terminated"]
            assert not state["stop_result"]["app_stop_succeeded"]
        assert [action[3] for action in system.actions if action[1:3] == ("app", "stop")] == ["ap-cache", "ap-sequence"]
        system.fail_stop = False
        runtime.tick()
        assert all(state["stop_result"]["app_stop_succeeded"] for state in runtime.state["studies"].values())
    finally:
        runtime.close()


def test_both_apps_must_be_observed_stopped_and_both_controllers_done(setup):
    system = setup[2]
    system.controllers = {100001: "done", 100002: "done"}
    system.apps = []
    runtime = build(setup)
    try:
        assert not runtime.tick()
        assert not any(state["seen_app"] for state in runtime.state["studies"].values())
        system.apps = [{"App ID": "ap-cache", "State": "stopped"},
                       {"App ID": "ap-sequence", "State": "ephemeral"}]
        assert not runtime.tick()
        system.apps[1]["State"] = "stopped"
        assert runtime.tick()
        assert runtime.state["phase"] == "finished"
    finally:
        runtime.close()


def test_durable_preemption_counts_cannot_shrink(setup):
    log = setup[0]["studies"]["cache"]["execution_log_paths"][0]
    with open(log, "w") as handle:
        handle.write(guard.PREEMPTION_LINE + "\n")
    runtime = build(setup)
    try:
        runtime.tick()
        with open(log, "w") as handle:
            handle.write("")
        runtime.tick()
        assert runtime.state["phase"] == "degraded"
        assert runtime.state["studies"]["cache"]["preemptions_by_log"][log] == 1
    finally:
        runtime.close()


def test_actual_lowercase_cli_schema_and_malformed_list_are_not_completion(setup):
    system = setup[2]
    system.controllers = {100001: "done", 100002: "done"}
    system.apps = [{"app_id": "ap-cache", "state": "stopped", "tasks": 0},
                   {"app_id": "ap-sequence", "state": "ephemeral", "tasks": 30}]
    runtime = build(setup)
    try:
        assert not runtime.tick()
        assert runtime.state["studies"]["cache"]["app_stopped"]
        system.apps[1]["state"] = "stopped"
        assert runtime.tick()
        system.apps = [{"error": "provider unavailable"}]
        assert not runtime.tick()
        assert runtime.state["phase"] == "degraded"
        assert not any(state["app_stopped"] for state in runtime.state["studies"].values())
    finally:
        runtime.close()


def test_immutable_deadline_survives_authorization_edit(setup):
    config, path, system = setup
    runtime = build(setup, now=lambda: NOW + 3601)
    replacement = json.loads(path.read_text())
    for study in replacement["studies"].values():
        study["deadline_utc"] = "2099-01-01T00:00:00Z"
        study["app_id"] = "ap-unrelated"
    path.write_text(json.dumps(replacement))
    try:
        assert runtime.tick()
        assert runtime.state["errors"]
        assert [action[3] for action in system.actions if action[1:3] == ("app", "stop")] == ["ap-cache", "ap-sequence"]
        assert system.apps[-1]["State"] == "ephemeral"
    finally:
        runtime.close()


def test_failed_stop_intent_persistence_prevents_mutation_and_remains_degraded(setup, monkeypatch):
    runtime = build(setup, now=lambda: NOW + 3601)
    original = runtime.persist
    attempts = []
    def cannot_persist():
        attempts.append(1)
        if len(attempts) <= 2:
            raise OSError("disk unavailable")
        original()
    monkeypatch.setattr(runtime, "persist", cannot_persist)
    try:
        assert not runtime.tick()
        assert len(attempts) == 3  # Both deadline stops tried, then status saved.
        assert not any(action[0] == "signal" or action[1:3] == ("app", "stop") for action in setup[2].actions)
        assert runtime.state["phase"] == "degraded"
        assert runtime.tick()
    finally:
        runtime.close()


def test_approval_and_full_burst_cushion_required(setup):
    with pytest.raises(SystemExit):
        guard.main(["--authorization", str(setup[1])])
    setup[0]["studies"]["cache"]["stop_commitment_usd"] = 73
    with pytest.raises(ValueError, match="burst cushion"):
        guard.validate_authorization(setup[0])
