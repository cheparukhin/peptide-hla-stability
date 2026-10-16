#!/usr/bin/env python3
"""Independent local guard for two explicitly authorized round-two Modal apps.

This never launches cloud work, imports Modal, or edits scientific ledgers.
Authorization is loaded once and bound to durable state by its SHA256. Monetary
commitments are conservative runtime estimates, not provider billing telemetry.
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


PREEMPTION_LINE = (
    "Container terminated due to preemption. Your Function will be restarted with the same input. "
    "For more details, see https://modal.com/docs/guide/preemption"
)
SCHEMA = "round2_runtime_guard.1"
STOPPED = {"stopped", "terminated"}


def amount(value, label):
    if isinstance(value, bool):
        raise ValueError(f"invalid {label}")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"invalid {label}")
    return result


def utc_timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("deadline must have an explicit UTC offset")
    return parsed.timestamp()


def absolute(value):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("authorization paths must be absolute")
    return value


def validate_authorization(auth):
    if auth.get("schema") != SCHEMA:
        raise ValueError("unknown authorization schema")
    for key in ("modal_cli", "status_path", "event_log_path", "lock_path"):
        absolute(auth[key])
    if len({auth[k] for k in ("status_path", "event_log_path", "lock_path")}) != 3:
        raise ValueError("guard output paths must differ")
    if not 0 < amount(auth["poll_seconds"], "poll interval") <= 60:
        raise ValueError("poll interval must be within 60 seconds")
    if not 0 < amount(auth["command_timeout_seconds"], "command timeout") <= 30:
        raise ValueError("command timeout must be within 30 seconds")
    studies = auth["studies"]
    if not isinstance(studies, dict) or set(studies) != {"cache", "sequence"}:
        raise ValueError("exactly cache and sequence authorizations required")
    ids, pids, inputs = [], [], []
    for name, config in studies.items():
        if not re.fullmatch(r"ap-[A-Za-z0-9]+", config["app_id"]):
            raise ValueError("invalid exact app ID")
        ids.append(config["app_id"])
        pid = config["controller_pid"]
        if type(pid) is not int or pid <= 1 or pid == os.getpid():
            raise ValueError("invalid controller PID")
        pids.append(pid)
        if not isinstance(config["controller_command"], str) or not config["controller_command"].strip():
            raise ValueError("exact ps controller command required")
        inputs.append(absolute(config["ledger_path"]))
        logs = config["execution_log_paths"]
        if not isinstance(logs, list) or not logs or len(set(logs)) != len(logs):
            raise ValueError("unique durable execution log paths required")
        inputs.extend(absolute(path) for path in logs)
        if amount(config["limit_usd"], "allocation") != {"cache": 80., "sequence": 18.}[name]:
            raise ValueError("study allocation must remain cache 80 / sequence 18")
        if type(config["max_concurrency"]) is not int or config["max_concurrency"] <= 0:
            raise ValueError("positive maximum concurrency required")
        if amount(config["rate_usd_per_second"], "rate") <= 0:
            raise ValueError("positive rate required")
        bound = amount(config["per_call_upper_usd"], "per-call reservation")
        threshold = amount(config["stop_commitment_usd"], "stop threshold")
        if bound <= 0 or threshold <= 0 or threshold + config["max_concurrency"] * bound > config["limit_usd"] + 1e-10:
            raise ValueError("stop threshold must leave a full maximum-concurrency burst cushion")
        utc_timestamp(config["deadline_utc"])
    if len(set(ids)) != 2 or len(set(pids)) != 2:
        raise ValueError("study app IDs and controller PIDs must be distinct")
    if any(auth[k] in inputs for k in ("status_path", "event_log_path", "lock_path")):
        raise ValueError("guard outputs cannot overwrite study inputs")
    return auth


def cache_commitments(path, config):
    ledger = json.loads(Path(path).read_text())
    if amount(ledger["limit_usd"], "ledger ceiling") != config["limit_usd"]:
        raise ValueError("cache ceiling differs from authorization")
    if not math.isclose(amount(ledger["reservation_per_call_usd"], "ledger per-call bound"),
                        config["per_call_upper_usd"], rel_tol=1e-9):
        raise ValueError("cache per-call bound differs from authorization")
    return amount(ledger["spent_upper_usd"], "spent") + sum(
        amount(value, "reservation") for value in ledger["reservations"].values())


def sequence_commitments(path, config):
    data = Path(path).read_bytes()
    # An append in progress cannot release a reservation. Replay only complete
    # records; a partial settlement retains its reservation. A new reservation
    # cannot dispatch until its durable newline has been written.
    lines = data.split(b"\n")[:-1]
    if not lines:
        raise ValueError("missing sequence configuration")
    first = json.loads(lines[0])
    if first.get("event") != "configuration" or first.get("schema") != "round2_budget.1":
        raise ValueError("unknown sequence ledger configuration")
    if amount(first["ceiling_usd"], "ledger ceiling") != config["limit_usd"]:
        raise ValueError("sequence ceiling differs from authorization")
    if not math.isclose(amount(first["call_upper_usd"], "ledger per-call bound"),
                        config["per_call_upper_usd"], rel_tol=1e-9):
        raise ValueError("sequence per-call bound differs from authorization")
    spent = amount(first["initial_reserve_usd"], "initial reserve")
    active, seen = {}, set()
    for line in lines[1:]:
        record = json.loads(line)
        token = record["token"]
        if record["event"] == "reservation":
            if token in seen or type(record["calls"]) is not int or record["calls"] <= 0:
                raise ValueError("invalid or duplicate sequence reservation")
            value = amount(record["reserved_usd"], "reservation")
            if not math.isclose(value, record["calls"] * config["per_call_upper_usd"], rel_tol=1e-9):
                raise ValueError("sequence reservation differs from bound")
            active[token] = value
            seen.add(token)
        elif record["event"] == "settlement":
            reserved = active.pop(token)
            charged = amount(record["charged_usd"], "settlement")
            if charged > reserved + 1e-10:
                raise ValueError("settlement exceeds reservation")
            spent += charged
        else:
            raise ValueError("unknown sequence journal event")
    return spent + sum(active.values())


def preemption_count(path):
    return sum(line == PREEMPTION_LINE for line in Path(path).read_text().splitlines())


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
    def __init__(self, authorization_path, *, runner=subprocess.run,
                 signal_group=os.killpg, now=time.time, sleep=time.sleep):
        self.authorization_path = Path(authorization_path)
        raw = self.authorization_path.read_bytes()
        self.auth = validate_authorization(json.loads(raw))
        self.digest = hashlib.sha256(raw).hexdigest()
        self.runner, self.signal_group, self.now, self.sleep = runner, signal_group, now, sleep
        lock = Path(self.auth["lock_path"])
        lock.parent.mkdir(parents=True, exist_ok=True)
        self.owner = lock.open("a+")
        try:
            fcntl.flock(self.owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.owner.close()
            raise RuntimeError("another runtime guard owns this authorization") from None
        try:
            status = Path(self.auth["status_path"])
            self.state = json.loads(status.read_text()) if status.exists() else {
                "schema": SCHEMA, "authorization_sha256": self.digest, "studies": {
                    name: {"seen_app": False, "preemptions_by_log": {}, "stop_intent": None,
                           "stop_result": None, "errors": []} for name in self.auth["studies"]}}
            if self.state.get("authorization_sha256") != self.digest or self.state.get("schema") != SCHEMA:
                raise ValueError("existing guard state belongs to a different authorization")
            log = Path(self.auth["event_log_path"])
            self.events = json.loads(log.read_text()) if log.exists() else []
            self.persist()
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self.owner.closed:
            fcntl.flock(self.owner.fileno(), fcntl.LOCK_UN)
            self.owner.close()

    def persist(self):
        self.state["updated_at_utc"] = datetime.fromtimestamp(self.now(), timezone.utc).isoformat()
        self.state["guard_pid"] = os.getpid()
        atomic_json(self.auth["status_path"], self.state)

    def event(self, name, kind, **details):
        self.events.append({"time": self.now(), "study": name, "event": kind, **details})
        atomic_json(self.auth["event_log_path"], self.events)

    def run_command(self, argv):
        return self.runner(argv, text=True, capture_output=True,
                           timeout=self.auth["command_timeout_seconds"], check=False)

    def controller(self, config):
        result = self.run_command(["ps", "-p", str(config["controller_pid"]), "-o", "pid=,pgid=,command="])
        if result.returncode == 1 and not result.stdout.strip():
            return "done"
        if result.returncode != 0:
            raise RuntimeError(f"ps failed: {result.stderr.strip()}")
        fields = result.stdout.strip().split(None, 2)
        if len(fields) != 3:
            raise RuntimeError("unparseable controller identity")
        if int(fields[0]) != config["controller_pid"] or int(fields[1]) != config["controller_pid"]:
            raise RuntimeError("controller is not its authorized process-group leader")
        if fields[2] != config["controller_command"]:
            raise RuntimeError("controller command identity differs; refusing group signals")
        return "running"

    def app_states(self):
        argv = [self.auth["modal_cli"], "app", "list", "--json"]
        result = self.run_command(argv)
        if result.returncode:
            raise RuntimeError(f"Modal app list failed: {result.stderr.strip()}")
        records = json.loads(result.stdout)
        if not isinstance(records, list):
            raise ValueError("Modal app list did not return a JSON list")
        apps = {}
        for record in records:
            # Current CLI emits lower-case keys; older versions emitted table
            # labels. Accept both explicitly, never infer state from absence.
            app_id = record["app_id"] if "app_id" in record else record["App ID"]
            state = record["state"] if "state" in record else record["State"]
            if app_id in apps:
                raise ValueError("duplicate Modal app ID")
            apps[app_id] = str(state).strip().lower()
        return apps

    def stop(self, name, reason):
        config, state = self.auth["studies"][name], self.state["studies"][name]
        if state["stop_intent"] is None:
            state["stop_intent"] = {"reason": reason, "time": self.now(), "app_id": config["app_id"],
                                    "controller_pid": config["controller_pid"]}
        # Durable intent precedes every mutation, including recovery attempts.
        self.persist()
        self.event(name, "stop_intent", **state["stop_intent"])
        result = {"frozen": False, "app_stop_succeeded": False, "terminated": False, "errors": []}
        controller_valid = False
        try:
            controller_valid = self.controller(config) == "running"
            if controller_valid:
                self.signal_group(config["controller_pid"], signal.SIGSTOP)
                result["frozen"] = True
        except Exception as error:
            result["errors"].append(f"freeze: {error}")
        # Cloud stop remains necessary even when the local controller is absent
        # or its identity cannot safely be validated.
        try:
            stopped = self.run_command([self.auth["modal_cli"], "app", "stop", config["app_id"], "--yes"])
            if stopped.returncode:
                raise RuntimeError(stopped.stderr.strip() or f"exit {stopped.returncode}")
            result["app_stop_succeeded"] = True
        except Exception as error:
            result["errors"].append(f"app stop: {error}")
        if controller_valid:
            try:
                # Validate again after the network call in case the PID exited
                # and was reused. Never signal a newly unrelated process.
                if self.controller(config) == "running":
                    self.signal_group(config["controller_pid"], signal.SIGTERM)
                    self.signal_group(config["controller_pid"], signal.SIGCONT)
                    result["terminated"] = True
            except Exception as error:
                result["errors"].append(f"terminate: {error}")
        state["stop_result"] = result
        self.persist()
        self.event(name, "stop_result", **result)

    def tick(self):
        self.state["errors"] = []
        try:
            if hashlib.sha256(self.authorization_path.read_bytes()).hexdigest() != self.digest:
                self.state["errors"].append("authorization file changed; using immutable loaded deadlines and scope")
        except Exception as error:
            self.state["errors"].append(f"authorization read failed; using immutable loaded scope: {error}")
        # Deadlines are checked before telemetry and all network requests.
        for name, config in self.auth["studies"].items():
            state = self.state["studies"][name]
            state["errors"] = []
            stop_success = (state["stop_result"] or {}).get("app_stop_succeeded", False)
            settled = state.get("app_stopped") and state.get("controller_state") == "done"
            try:
                if state["stop_intent"] is not None and not stop_success and not settled:
                    self.stop(name, state["stop_intent"]["reason"])
                elif self.now() >= utc_timestamp(config["deadline_utc"]) and not stop_success and not settled:
                    self.stop(name, "immutable_wall_deadline")
            except Exception as error:
                state["errors"].append(f"durable deadline stop: {error}")
            try:
                ordinary = (cache_commitments if name == "cache" else sequence_commitments)(config["ledger_path"], config)
                for path in config["execution_log_paths"]:
                    count = preemption_count(path)
                    previous = state["preemptions_by_log"].get(path, 0)
                    if count < previous:
                        raise ValueError("durable preemption log count decreased")
                    state["preemptions_by_log"][path] = count
                total = sum(state["preemptions_by_log"].values())
                effective = ordinary + total * config["per_call_upper_usd"]
                state.update(ordinary_commitment_usd=ordinary, preemption_count=total,
                             effective_commitment_usd=effective)
                if effective >= config["stop_commitment_usd"] and state["stop_intent"] is None and not settled:
                    self.stop(name, "commitment_cushion_threshold")
            except Exception as error:
                state["errors"].append(f"budget telemetry: {error}")
            try:
                state["controller_state"] = self.controller(config)
            except Exception as error:
                state["controller_state"] = "unknown"
                state["errors"].append(f"controller telemetry: {error}")
        try:
            apps = self.app_states()
            for name, config in self.auth["studies"].items():
                state = self.state["studies"][name]
                app_state = apps.get(config["app_id"])
                if app_state is not None:
                    state["seen_app"] = True
                state["app_state"] = app_state or "unobserved"
                state["app_stopped"] = state["seen_app"] and app_state in STOPPED
        except Exception as error:
            self.state["errors"].append(f"app telemetry: {error}")
            for state in self.state["studies"].values():
                state["app_stopped"] = False
                state["app_state"] = "unknown"
        finished = all(s.get("app_stopped") and s.get("controller_state") == "done"
                       for s in self.state["studies"].values())
        errors = self.state["errors"] + [error for s in self.state["studies"].values()
                                         for error in s["errors"] + (s["stop_result"] or {}).get("errors", [])]
        self.state["phase"] = "finished" if finished else "degraded" if errors else "monitoring"
        self.persist()
        return finished

    def run(self):
        while True:
            try:
                if self.tick():
                    return
            except Exception as error:
                # Persistence failures are visible on stderr; deadlines remain
                # active on subsequent polls. A failed intent write prevents
                # mutations until durable recovery information can be saved.
                print(f"runtime guard degraded: {error}", file=sys.stderr, flush=True)
            deadlines = [utc_timestamp(s["deadline_utc"]) for s in self.auth["studies"].values()]
            future = [deadline - self.now() for deadline in deadlines if deadline > self.now()]
            delay = min([self.auth["poll_seconds"], *future])
            self.sleep(max(.1, delay))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True, help="immutable root-supplied authorization JSON")
    parser.add_argument("--approved", action="store_true", help="required explicit launch authorization")
    args = parser.parse_args(argv)
    if not args.approved:
        parser.error("--approved is required")
    guard = RuntimeGuard(args.authorization)
    try:
        guard.run()
    finally:
        guard.close()


if __name__ == "__main__":
    main()
