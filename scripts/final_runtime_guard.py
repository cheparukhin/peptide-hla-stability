#!/usr/bin/env python3
"""Independent single-scope Modal budget watchdog; never launches paid work.

ReservationBudget journals and execution logs are read-only. Root supplies an
immutable authorization after identifying one app and its process-group leader.
All real stop operations require --approved; tests inject mock operations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

SCHEMA = "final_runtime_guard.1"
PHASE_CEILING_USD = 100.
RATE_USD_PER_SECOND = .1266 / 3600  # Authorized 2 CPU, 4 GiB container.
STOPPED = {"stopped", "terminated"}
# Exact Modal 1.6.1 CLI labels, plus previously supported canonical labels.
# Never infer lifecycle from an unknown label or remove arbitrary punctuation.
APP_STATE_LABELS = {
    "deployed": "deployed", "ephemeral": "ephemeral",
    "ephemeral (detached)": "ephemeral", "initializing...": "initializing",
    "stopping...": "stopping", "disabled": "disabled", "stopped": "stopped",
    "terminated": "terminated", "initializing": "initializing", "pending": "pending", "stopping": "stopping",
}
PREEMPTION_LINE = (
    "Container terminated due to preemption. Your Function will be restarted with the same input. "
    "For more details, see https://modal.com/docs/guide/preemption"
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    return sha256(Path(path).read_bytes())


def amount(value, label):
    if isinstance(value, bool):
        raise ValueError(f"invalid {label}")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        raise ValueError(f"invalid {label}") from None
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"invalid {label}")
    return result


def utc_timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("deadline must use explicit UTC")
    return parsed.timestamp()


def absolute(value):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("authorization paths must be absolute")
    return value


def _positive_int(value, maximum, label):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"invalid {label}; expected integer in 1..{maximum}")
    return value


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise ValueError("invalid SHA256")


def scope_limit(auth):
    return amount(auth["total_allocation_usd"], "total allocation") - amount(auth["prior_charges_usd"], "prior charges")


def validate_authorization(auth):
    auth = dict(auth)
    if auth.get("schema") != SCHEMA:
        raise ValueError("unknown authorization schema")
    for key in ("phase_ceiling_usd", "total_allocation_usd", "prior_charges_usd", "poll_seconds",
                "command_timeout_seconds", "rate_usd_per_second", "call_timeout_seconds",
                "per_call_overhead_seconds", "per_call_upper_usd", "initial_reserve_usd",
                "stop_commitment_usd", "shutdown_cushion_usd"):
        auth[key] = amount(auth[key], key)
    for key in ("phase_id", "scope_id"):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", auth[key]):
            raise ValueError("invalid phase/scope ID")
    if auth["scope_kind"] not in ("smoke", "main"):
        raise ValueError("scope_kind must be smoke or main")
    if amount(auth["phase_ceiling_usd"], "phase ceiling") != PHASE_CEILING_USD:
        raise ValueError("new phase ceiling must remain 100 USD")
    allocation = amount(auth["total_allocation_usd"], "total allocation")
    prior = amount(auth["prior_charges_usd"], "prior charges")
    if not 0 <= prior < allocation <= PHASE_CEILING_USD:
        raise ValueError("prior charges and total allocation exceed the global phase ceiling")
    outputs = ("status_path", "event_log_path", "stop_flag_path", "scope_lock_path", "global_lock_path", "global_status_path")
    paths = [absolute(auth[key]) for key in outputs]
    absolute(auth["modal_cli"])
    inputs = [absolute(auth["ledger_path"])]
    logs = auth["execution_log_paths"]
    if not isinstance(logs, list) or not logs:
        raise ValueError("durable execution logs required")
    inputs.extend(absolute(path) for path in logs)
    if len(set(str(Path(p).resolve()) for p in paths + inputs)) != len(paths + inputs):
        raise ValueError("guard output/input paths must differ, including resolved aliases")
    for key in ("poll_seconds", "command_timeout_seconds"):
        cap = 10 if key == "poll_seconds" else 30
        if not 0 < amount(auth[key], key) <= cap:
            raise ValueError(f"{key} must be positive and <= {cap}")
    concurrency = _positive_int(auth["max_concurrency"], 30, "maximum cloud concurrency")
    if auth["cpu_cores"] != 2 or auth["memory_mib"] != 4096:
        raise ValueError("authorized resource shape must be 2 CPU and 4 GiB")
    if not math.isclose(amount(auth["rate_usd_per_second"], "rate"), RATE_USD_PER_SECOND, rel_tol=1e-9, abs_tol=0):
        raise ValueError("rate differs from the locked 0.1266 USD/hour container rate")
    timeout = amount(auth["call_timeout_seconds"], "call timeout")
    overhead = amount(auth["per_call_overhead_seconds"], "per-call overhead")
    reserve = amount(auth["per_call_upper_usd"], "per-call reservation")
    if timeout <= 0 or reserve <= 0 or reserve + 1e-10 < (timeout + overhead) * RATE_USD_PER_SECOND:
        raise ValueError("full per-call reservation does not cover timeout plus overhead")
    initial = amount(auth["initial_reserve_usd"], "initial reserve")
    threshold = amount(auth["stop_commitment_usd"], "stop threshold")
    cushion = amount(auth["shutdown_cushion_usd"], "shutdown cushion")
    attempts = _positive_int(auth["max_app_stop_attempts"], 3, "app-stop attempts")
    burst = concurrency * reserve
    command_bound = auth["command_timeout_seconds"] * (attempts + 10)
    if cushion + 1e-10 < concurrency * RATE_USD_PER_SECOND * command_bound or cushion <= 0:
        raise ValueError("shutdown cushion cannot cover bounded stop-command latency")
    if not 0 <= initial < threshold or threshold + burst + cushion > scope_limit(auth) + 1e-10:
        raise ValueError("stop threshold must leave a full concurrency burst and shutdown cushion")
    authorized = utc_timestamp(auth["authorized_at_utc"])
    startup = utc_timestamp(auth["startup_deadline_utc"])
    deadline = utc_timestamp(auth["deadline_utc"])
    if not authorized < startup < deadline or startup - authorized > 300:
        raise ValueError("startup deadline must be fixed within five minutes and precede wall deadline")
    if initial + concurrency * RATE_USD_PER_SECOND * (deadline - authorized) + cushion > scope_limit(auth) + 1e-10:
        raise ValueError("immutable wall deadline exceeds the continuous runtime allocation")
    if not re.fullmatch(r"ap-[A-Za-z0-9]+", auth["app_id"]):
        raise ValueError("invalid exact app ID")
    pid = auth["controller_pid"]
    if type(pid) is not int or pid <= 1 or pid == os.getpid():
        raise ValueError("invalid exact controller PID")
    if not isinstance(auth["controller_command"], str) or not auth["controller_command"].strip():
        raise ValueError("exact controller command required")
    if not isinstance(auth["controller_started_at"], str) or len(auth["controller_started_at"].split()) != 5:
        raise ValueError("exact ps lstart controller fingerprint required")
    for key in ("source_hashes", "protocol_hashes"):
        mapping = auth[key]
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"nonempty {key} required")
        for path, value in mapping.items():
            inputs.append(absolute(path))
            _hash(value)
    if str(Path(__file__).resolve()) not in auth["source_hashes"]:
        raise ValueError("source_hashes must bind this watchdog's exact source path")
    if any(str(Path(p).resolve()) in {str(Path(i).resolve()) for i in inputs} for p in paths):
        raise ValueError("guard outputs cannot overwrite source/protocol inputs")
    previous = auth.get("prior_reconciliation")
    if previous is not None:
        absolute(previous["path"])
        _hash(previous["sha256"])
        if str(Path(previous["path"]).resolve()) in {str(Path(p).resolve()) for p in paths}:
            raise ValueError("guard outputs cannot overwrite prior reconciliation")
    if auth["scope_kind"] == "main" and previous is None:
        raise ValueError("main authorization requires stopped/reconciled previous scope")
    return auth


def verify_sources(auth):
    for key in ("source_hashes", "protocol_hashes"):
        for path, expected in auth[key].items():
            if file_hash(path) != expected:
                raise ValueError(f"immutable {key} drift: {path}")


def replay_commitments(path, auth):
    """Strict read-only ReservationBudget replay; incomplete tail cannot settle.

    Dispatch requires the controller's fsynced, newline-terminated reservation.
    A partial tail is ignored; complete corrupt records fail closed. The guard
    separately binds every observed complete byte prefix against truncation.
    """
    data = Path(path).read_bytes()
    complete_size = data.rfind(b"\n") + 1
    prefix = data[:complete_size]
    lines = prefix.splitlines()
    if not lines:
        raise ValueError("missing complete budget configuration")
    first = json.loads(lines[0])
    if first.get("event") != "configuration" or first.get("schema") != "round2_budget.1":
        raise ValueError("unknown ledger schema")
    if not math.isclose(amount(first["ceiling_usd"], "ledger ceiling"), scope_limit(auth), rel_tol=0, abs_tol=1e-10):
        raise ValueError("ledger ceiling differs from current scope allocation")
    if not math.isclose(amount(first["call_upper_usd"], "ledger call bound"), auth["per_call_upper_usd"], rel_tol=1e-9):
        raise ValueError("ledger per-call bound differs from authorization")
    spent = amount(first["initial_reserve_usd"], "ledger initial reserve")
    if not math.isclose(spent, auth["initial_reserve_usd"], rel_tol=0, abs_tol=1e-10):
        raise ValueError("ledger initial reserve differs from authorization")
    max_calls = first["max_calls"]
    if type(max_calls) is not int or max_calls <= 0:
        raise ValueError("invalid ledger call quota")
    active, seen, calls = {}, set(), 0
    for line in lines[1:]:
        record = json.loads(line)
        token = record["token"]
        if not isinstance(token, str) or not token:
            raise ValueError("invalid ledger token")
        if record["event"] == "reservation":
            n = record["calls"]
            if token in seen or type(n) is not int or n <= 0:
                raise ValueError("duplicate/invalid reservation")
            value = amount(record["reserved_usd"], "reservation")
            if not math.isclose(value, n * auth["per_call_upper_usd"], rel_tol=1e-9):
                raise ValueError("reservation does not use full per-call bound")
            active[token] = (value, n)
            seen.add(token)
            calls += n
        elif record["event"] == "settlement":
            if token not in active:
                raise ValueError("settlement without an active reservation")
            reserved, n = active.pop(token)
            charged = amount(record["charged_usd"], "settlement")
            if charged > reserved + 1e-10 or record.get("calls", n) != n:
                raise ValueError("invalid settlement")
            # Failed or interrupted calls burn their full bound, matching the
            # controller contract. A lower failed charge is under-accounting.
            if record.get("outcome") != "success" and not math.isclose(charged, reserved, rel_tol=1e-9):
                raise ValueError("failed/interrupted call did not burn its full reservation")
            spent += charged
        else:
            raise ValueError("unknown journal event")
        if spent + sum(v for v, _ in active.values()) > scope_limit(auth) + 1e-10 or calls > max_calls:
            raise ValueError("ledger exceeds authorized bounds")
    return {"ordinary_commitment_usd": spent + sum(v for v, _ in active.values()),
            "settled_usd": spent, "inflight_reserved_usd": sum(v for v, _ in active.values()),
            "calls": calls, "inflight_calls": sum(n for _, n in active.values()),
            "complete_bytes": complete_size, "complete_prefix_sha256": sha256(prefix),
            "partial_tail_bytes": len(data) - complete_size}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


class RuntimeGuard:
    def __init__(self, authorization_path, *, runner=subprocess.run, signal_group=os.killpg,
                 now=time.time, sleep=time.sleep):
        self.authorization_path = Path(authorization_path)
        raw = self.authorization_path.read_bytes()
        self.auth = validate_authorization(json.loads(raw))
        self.digest = sha256(raw)
        self.runner, self.signal_group, self.now, self.sleep = runner, signal_group, now, sleep
        self.owners = []
        try:
            for key in ("global_lock_path", "scope_lock_path"):
                path = Path(self.auth[key])
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = path.open("a+")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    raise RuntimeError("another guard owns the phase or scope; parallel scopes forbidden") from None
                self.owners.append(handle)
            output_paths = [self.auth[k] for k in ("status_path", "event_log_path", "stop_flag_path", "global_status_path")]
            if str(self.authorization_path.resolve()) in {str(Path(p).resolve()) for p in output_paths}:
                raise ValueError("guard output cannot overwrite authorization")
            global_path = Path(self.auth["global_status_path"])
            previous = json.loads(global_path.read_text()) if global_path.exists() else None
            if previous is not None:
                if previous.get("schema") != SCHEMA or previous.get("phase_id") != self.auth["phase_id"]:
                    raise ValueError("global phase identity mismatch")
                if previous["authorization_sha256"] != self.digest:
                    self._validate_transition(previous)
            elif self.auth["scope_kind"] != "smoke" or self.auth["prior_charges_usd"] != 0:
                raise ValueError("first phase scope must be zero-prior smoke; explicit transition required")
            path = Path(self.auth["status_path"])
            self.state = json.loads(path.read_text()) if path.exists() else {
                "schema": SCHEMA, "authorization_sha256": self.digest, "scope_id": self.auth["scope_id"],
                "seen_app": False, "seen_ledger": False, "preemptions_by_log": {}, "log_prefixes": {},
                "ledger_complete_bytes": 0, "ledger_complete_prefix_sha256": sha256(b""),
                "stop_intent": None, "app_stop_attempts": 0, "app_stop_command_succeeded": False,
                "app_stopped": False, "controller_state": "unknown", "phase": "starting", "errors": []}
            if self.state.get("schema") != SCHEMA or self.state.get("authorization_sha256") != self.digest:
                raise ValueError("existing guard state belongs to different authorization; deadline cannot be extended")
            if (type(self.state.get("app_stop_attempts")) is not int
                    or not 0 <= self.state["app_stop_attempts"] <= self.auth["max_app_stop_attempts"]
                    or type(self.state.get("ledger_complete_bytes")) is not int
                    or self.state["ledger_complete_bytes"] < 0):
                raise ValueError("invalid persistent guard counters")
            _hash(self.state["ledger_complete_prefix_sha256"])
            if self.state["stop_intent"] is not None and self.state["stop_intent"].get("authorization_sha256") != self.digest:
                raise ValueError("persistent stop intent identity mismatch")
            flag = Path(self.auth["stop_flag_path"])
            if flag.exists() and json.loads(flag.read_text()).get("authorization_sha256") != self.digest:
                raise ValueError("stop flag belongs to another authorization")
            self.events = json.loads(Path(self.auth["event_log_path"]).read_text()) if Path(self.auth["event_log_path"]).exists() else []
            if not isinstance(self.events, list):
                raise ValueError("invalid event log")
            self.persist()
        except BaseException:
            self.close()
            raise

    def _validate_transition(self, previous):
        reconciliation = self.auth.get("prior_reconciliation")
        if reconciliation is None:
            raise ValueError("new scope requires stopped/reconciled predecessor")
        if file_hash(reconciliation["path"]) != reconciliation["sha256"]:
            raise ValueError("prior reconciliation hash mismatch")
        record = json.loads(Path(reconciliation["path"]).read_text())
        old_status = Path(previous["status_path"])
        old = json.loads(old_status.read_text())
        if not old.get("app_stopped") or old.get("controller_state") != "done":
            raise ValueError("previous scope not observed stopped with controller done")
        if (record.get("schema") != "final_scope_reconciliation.1"
                or record.get("authorization_sha256") != previous["authorization_sha256"]
                or record.get("guard_status_sha256") != file_hash(old_status)
                or old.get("authorization_sha256") != previous["authorization_sha256"]):
            raise ValueError("reconciliation does not bind previous authorization and stopped status")
        liability = old.get("effective_commitment_usd", previous["scope_limit_usd"]) if old.get("budget_verified") else previous["scope_limit_usd"]
        minimum = previous["prior_charges_usd"] + liability
        reconciled = amount(record["cumulative_prior_charges_usd"], "reconciled cumulative prior charges")
        if reconciled + 1e-10 < minimum or not math.isclose(reconciled, self.auth["prior_charges_usd"], rel_tol=0, abs_tol=1e-10):
            raise ValueError("new prior charges under-account previous scope")
        if self.auth["scope_id"] == previous["scope_id"] or self.auth["app_id"] == previous["app_id"]:
            raise ValueError("new scope must have distinct scope and app IDs; no automatic recovery")

    def close(self):
        for handle in reversed(self.owners):
            if not handle.closed:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()

    def persist(self):
        self.state["updated_at_utc"] = datetime.fromtimestamp(self.now(), timezone.utc).isoformat()
        self.state["guard_pid"] = os.getpid()
        atomic_json(self.auth["status_path"], self.state)
        atomic_json(self.auth["global_status_path"], {"schema": SCHEMA, "phase_id": self.auth["phase_id"],
            "scope_id": self.auth["scope_id"], "app_id": self.auth["app_id"], "authorization_sha256": self.digest,
            "status_path": self.auth["status_path"], "phase": self.state["phase"],
            "prior_charges_usd": self.auth["prior_charges_usd"], "scope_limit_usd": scope_limit(self.auth),
            "phase_ceiling_usd": PHASE_CEILING_USD})

    def event(self, kind, **details):
        self.events.append({"time": self.now(), "event": kind, **details})
        atomic_json(self.auth["event_log_path"], self.events)

    def command(self, argv):
        return self.runner(argv, text=True, capture_output=True, timeout=self.auth["command_timeout_seconds"], check=False)

    def controller(self):
        pid = self.auth["controller_pid"]
        result = self.command(["ps", "-p", str(pid), "-o", "pid=,pgid=,lstart=,command="])
        if result.returncode == 1 and not result.stdout.strip():
            return "done"
        if result.returncode:
            raise RuntimeError(f"controller lookup failed: {result.stderr.strip()}")
        fields = result.stdout.strip().split(None, 7)
        if (len(fields) != 8 or int(fields[0]) != pid or int(fields[1]) != pid
                or " ".join(fields[2:7]) != " ".join(self.auth["controller_started_at"].split())
                or fields[7] != self.auth["controller_command"]):
            raise RuntimeError("controller PID/group/start-time/command identity differs; refusing signals")
        return "running"

    def signal_controller(self, sig):
        # Verify immediately before each signal, including SIGCONT.
        if self.controller() == "running":
            self.signal_group(self.auth["controller_pid"], sig)
            return True
        return False

    def app_state(self):
        result = self.command([self.auth["modal_cli"], "app", "list", "--json"])
        if result.returncode:
            raise RuntimeError(f"Modal list failed: {result.stderr.strip()}")
        records = json.loads(result.stdout)
        if not isinstance(records, list):
            raise ValueError("Modal app list must be a JSON list")
        states = {}
        for record in records:
            app = record.get("app_id", record.get("App ID"))
            state = record.get("state", record.get("State"))
            if not isinstance(app, str) or not isinstance(state, str) or app in states:
                raise ValueError("malformed/duplicate app telemetry")
            states[app] = state
        raw = states.get(self.auth["app_id"])
        self.state["app_state_raw"] = raw
        if raw is None:
            return None
        value = APP_STATE_LABELS.get(raw.strip().lower())
        if value is None:
            raise ValueError(f"unknown authorized app state: {raw!r}")
        return value

    def budget(self):
        path = Path(self.auth["ledger_path"])
        raw = path.read_bytes()
        n = self.state["ledger_complete_bytes"]
        if len(raw) < n or sha256(raw[:n]) != self.state["ledger_complete_prefix_sha256"]:
            raise ValueError("durable ledger prefix shrank or changed")
        value = replay_commitments(path, self.auth)
        self.state.update(ledger_complete_bytes=value.pop("complete_bytes"),
                          ledger_complete_prefix_sha256=value.pop("complete_prefix_sha256"), seen_ledger=True)
        for log in self.auth["execution_log_paths"]:
            data = Path(log).read_bytes()
            previous = self.state["log_prefixes"].get(log, {"bytes": 0, "sha256": sha256(b"")})
            if len(data) < previous["bytes"] or sha256(data[:previous["bytes"]]) != previous["sha256"]:
                raise ValueError("durable preemption log shrank or changed")
            # Recognize complete warning text even before its terminating newline.
            count = data.count(PREEMPTION_LINE.encode())
            self.state["preemptions_by_log"][log] = count
            self.state["log_prefixes"][log] = {"bytes": len(data), "sha256": sha256(data)}
        preemptions = sum(self.state["preemptions_by_log"].values())
        effective = value["ordinary_commitment_usd"] + preemptions * self.auth["per_call_upper_usd"]
        self.state.update(value, preemption_count=preemptions, effective_commitment_usd=effective,
                          cumulative_commitment_usd=self.auth["prior_charges_usd"] + effective)
        return effective

    def stop(self, reason):
        if self.state["stop_intent"] is None:
            self.state["stop_intent"] = {"reason": reason, "time": self.now(), "app_id": self.auth["app_id"],
                                         "controller_pid": self.auth["controller_pid"], "authorization_sha256": self.digest}
        # Stop flag precedes process or cloud mutations. Controllers MUST check
        # it immediately before every submission, including retries.
        atomic_json(self.auth["stop_flag_path"], self.state["stop_intent"])
        self.persist()
        self.event("stop_intent", **self.state["stop_intent"])
        result = {"frozen": False, "terminated": False, "errors": []}
        try:
            result["frozen"] = self.signal_controller(signal.SIGSTOP)
        except Exception as exc:
            result["errors"].append(f"freeze: {exc}")
        while (not self.state["app_stopped"]
               and self.state["app_stop_attempts"] < self.auth["max_app_stop_attempts"]):
            self.state["app_stop_attempts"] += 1
            self.persist()  # Attempt is charged durably before it is made.
            try:
                stopped = self.command([self.auth["modal_cli"], "app", "stop", self.auth["app_id"], "--yes"])
                if stopped.returncode:
                    raise RuntimeError(stopped.stderr.strip() or f"exit {stopped.returncode}")
                self.state["app_stop_command_succeeded"] = True
                self.persist()
                break  # Observe provider state before another successful retry.
            except Exception as exc:
                result["errors"].append(f"app stop attempt {self.state['app_stop_attempts']}: {exc}")
        try:
            if self.signal_controller(signal.SIGTERM):
                result["terminated"] = True
                self.signal_controller(signal.SIGCONT)
        except Exception as exc:
            result["errors"].append(f"terminate: {exc}")
        self.state["stop_result"] = result
        self.persist()
        self.event("stop_result", **result)

    def tick(self):
        errors, reason = [], None
        stopped_this_tick = False
        # Clock checks happen before all filesystem telemetry and network calls.
        if self.now() >= utc_timestamp(self.auth["deadline_utc"]):
            reason = "immutable_wall_deadline"
        if not (self.state["app_stopped"] and self.state["controller_state"] == "done") and (reason or self.state["stop_intent"]):
            self.stop(reason or self.state["stop_intent"]["reason"])
            stopped_this_tick = True
        try:
            if file_hash(self.authorization_path) != self.digest:
                raise ValueError("authorization bytes changed; retaining loaded scope/deadlines")
            verify_sources(self.auth)
        except Exception as exc:
            errors.append(str(exc))
            reason = reason or "immutable_authorization_or_source_drift"
        try:
            effective = self.budget()
            self.state["budget_verified"] = True
            if effective >= self.auth["stop_commitment_usd"]:
                reason = reason or "commitment_burst_cushion_threshold"
        except FileNotFoundError as exc:
            self.state["budget_verified"] = False
            errors.append(f"budget telemetry: {exc}")
            if self.state["seen_ledger"] or self.now() >= utc_timestamp(self.auth["startup_deadline_utc"]):
                reason = reason or "unverifiable_budget"
        except Exception as exc:
            self.state["budget_verified"] = False
            errors.append(f"budget telemetry: {exc}")
            reason = reason or "unverifiable_budget"
        try:
            self.state["controller_state"] = self.controller()
        except Exception as exc:
            self.state["controller_state"] = "unknown"
            errors.append(f"controller telemetry: {exc}")
            reason = reason or "controller_identity_unknown"
        try:
            state = self.app_state()
            self.state["app_state"] = state or "unobserved"
            if state is not None:
                self.state["seen_app"] = True
            self.state["app_stopped"] = state in STOPPED
            if state is None and (self.state["seen_app"] or self.now() >= utc_timestamp(self.auth["startup_deadline_utc"])):
                reason = reason or "authorized_app_unobserved"
        except Exception as exc:
            self.state["app_state"], self.state["app_stopped"] = "unknown", False
            errors.append(f"app telemetry: {exc}")
            reason = reason or "app_telemetry_unknown"
        finished = self.state["app_stopped"] and self.state["controller_state"] == "done"
        if self.now() >= utc_timestamp(self.auth["startup_deadline_utc"]) and not (self.state["seen_app"] and self.state["seen_ledger"]):
            reason = reason or "immutable_startup_deadline"
        if (self.state["controller_state"] == "done" and not self.state["app_stopped"]
                and self.state["app_state"] != "stopping"):
            reason = reason or "controller_done_app_not_stopped"
        if not finished and not stopped_this_tick and (reason or self.state["stop_intent"] is not None):
            self.stop(reason or self.state["stop_intent"]["reason"])
            # Completion must be provider-observed, never inferred from a stop
            # command exit code or from absence in app list.
            try:
                self.state["app_state"] = self.app_state() or "unobserved"
                self.state["app_stopped"] = self.state["app_state"] in STOPPED
                self.state["controller_state"] = self.controller()
            except Exception as exc:
                errors.append(f"post-stop observation: {exc}")
            finished = self.state["app_stopped"] and self.state["controller_state"] == "done"
        if finished:
            # The controller may append its last settlement after the initial
            # budget read, while the provider is closing the app. Refresh only
            # after stopped+absent observation; never rewrite historical ledgers.
            try:
                self.budget()
                self.state["budget_verified"] = True
            except Exception as exc:
                self.state["budget_verified"] = False
                errors.append(f"terminal budget telemetry: {exc}")
        exhausted = (self.state["stop_intent"] is not None and not self.state["app_stopped"]
                     and self.state["app_stop_attempts"] >= self.auth["max_app_stop_attempts"])
        self.state["errors"] = errors
        ready = (self.state["seen_app"] and self.state["seen_ledger"] and self.state.get("budget_verified")
                 and self.state["controller_state"] == "running" and self.state["app_state"] not in ("stopping", "disabled"))
        self.state["phase"] = "finished" if finished else "stop_failed_manual_action" if exhausted else "stopping" if self.state["stop_intent"] else "degraded" if errors else "finishing" if self.state["app_state"] == "stopping" else "monitoring" if ready else "starting"
        self.persist()
        return finished

    def run(self):
        while True:
            try:
                if self.tick():
                    return 0
                if self.state["phase"] == "stop_failed_manual_action":
                    print("runtime guard: bounded app-stop attempts exhausted; stop flag remains; manual app stop/reconciliation required", file=sys.stderr, flush=True)
                    return 2
            except Exception as exc:
                # Do not launch or thaw. A failed durable stop write prevents
                # mutations; print the failure and retry only persistence.
                print(f"runtime guard persistence failure: {exc}", file=sys.stderr, flush=True)
            boundaries = [utc_timestamp(self.auth[k]) - self.now() for k in ("startup_deadline_utc", "deadline_utc")]
            future = [delay for delay in boundaries if delay > 0]
            self.sleep(max(.1, min([self.auth["poll_seconds"], *future])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--approved", action="store_true", help="explicit root-supplied stop authorization")
    parser.add_argument("--validate-only", action="store_true", help="read-only authorization/source checks; no CLI/process calls")
    args = parser.parse_args(argv)
    if args.validate_only:
        auth = validate_authorization(json.loads(args.authorization.read_text()))
        verify_sources(auth)
        print(json.dumps({"scope_id": auth["scope_id"], "current_scope_limit_usd": scope_limit(auth),
                          "max_concurrency": auth["max_concurrency"], "deadline_utc": auth["deadline_utc"]}))
        return 0
    if not args.approved:
        parser.error("--approved required before this watchdog may stop its authorized scope")
    runtime = RuntimeGuard(args.authorization)
    try:
        return runtime.run()
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
