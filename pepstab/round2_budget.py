"""Durable shared reservations for bounded paid round-two work.

Reserve worst-case cost before dispatch. Settle successful calls at measured
runtime plus the declared overhead; failed or interrupted calls burn their full
reservation. One controller owns a journal, shared by its concurrent threads.
This is a runtime-derived guard, not a reconciled provider invoice.
"""
from __future__ import annotations

import fcntl
import json
import math
from pathlib import Path
import threading
import time
import uuid


class BudgetExhausted(RuntimeError):
    pass


class ReservationBudget:
    def __init__(self, path, *, ceiling_usd, call_upper_usd,
                 initial_reserve_usd=0., max_calls=900):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.config = {"schema": "round2_budget.1", "ceiling_usd": float(ceiling_usd),
                       "call_upper_usd": float(call_upper_usd),
                       "initial_reserve_usd": float(initial_reserve_usd), "max_calls": int(max_calls)}
        if not (0 <= initial_reserve_usd <= ceiling_usd and call_upper_usd > 0 and max_calls > 0):
            raise ValueError("invalid budget configuration")
        if not all(math.isfinite(float(x)) for x in (ceiling_usd, call_upper_usd, initial_reserve_usd)):
            raise ValueError("nonfinite budget configuration")
        self.condition = threading.Condition()
        self.calls = 0
        self.spent = float(initial_reserve_usd)
        self.active = {}
        self._owner = self.path.with_suffix(self.path.suffix+".lock").open("a+")
        try:
            fcntl.flock(self._owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._owner.close()
            raise RuntimeError("another controller owns this paid-work budget") from None
        try:
            if self.path.exists() and self.path.stat().st_size:
                records = [json.loads(line) for line in self.path.read_text().splitlines()]
                if records[0] != {"event": "configuration", **self.config}:
                    raise ValueError("existing budget configuration differs; explicit reconciliation required")
                for record in records[1:]:
                    if record["event"] == "reservation":
                        self.active[record["token"]] = record
                        self.calls += record["calls"]
                    elif record["event"] == "settlement":
                        reserved = self.active.pop(record["token"])
                        if not 0 <= record["charged_usd"] <= reserved["reserved_usd"] + 1e-12:
                            raise ValueError("invalid settlement in saved budget")
                        self.spent += record["charged_usd"]
                    else:
                        raise ValueError("unknown budget journal event")
                # A dead controller's unacknowledged work might have run remotely.
                # Never assume absence of a result means absence of a provider bill.
                for token in list(self.active):
                    self.settle(token, self.active[token]["reserved_usd"], outcome="interrupted_full_reservation")
            else:
                self._append({"event": "configuration", **self.config})
            self._assert_bounds()
        except BaseException:
            self.close()
            raise

    def _append(self, record):
        import os
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False)+"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _assert_bounds(self):
        committed = self.spent + sum(x["reserved_usd"] for x in self.active.values())
        if committed > self.config["ceiling_usd"] + 1e-10 or self.calls > self.config["max_calls"]:
            raise ValueError("saved paid-work ledger exceeds its bound")

    def reserve(self, calls, *, context=""):
        if not isinstance(calls, int) or calls <= 0:
            raise ValueError("reservation must contain a positive integer call count")
        amount = calls*self.config["call_upper_usd"]
        with self.condition:
            if self.calls+calls > self.config["max_calls"]:
                raise BudgetExhausted("paid member-call quota exhausted; no automatic retries")
            while self.spent+sum(x["reserved_usd"] for x in self.active.values())+amount > self.config["ceiling_usd"]+1e-12:
                if not self.active:
                    raise BudgetExhausted("remaining credit cannot cover this complete ensemble's worst-case reservation")
                self.condition.wait(timeout=30.)
                if self.calls+calls > self.config["max_calls"]:
                    raise BudgetExhausted("paid member-call quota exhausted while waiting")
            token = uuid.uuid4().hex
            record = {"event": "reservation", "token": token, "calls": calls,
                      "reserved_usd": amount, "context": context, "time": time.time()}
            self._append(record)
            self.active[token] = record
            self.calls += calls
            self._assert_bounds()
            return token

    def settle(self, token, charged_usd, *, outcome="success", details=None):
        with self.condition:
            reserved = self.active[token]
            charged = float(charged_usd)
            if not math.isfinite(charged) or not 0 <= charged <= reserved["reserved_usd"]+1e-12:
                raise ValueError("settlement must be finite and within its reservation")
            record = {"event": "settlement", "token": token, "calls": reserved["calls"],
                      "charged_usd": charged, "outcome": outcome, "details": details,
                      "time": time.time()}
            self._append(record)
            del self.active[token]
            self.spent += charged
            self._assert_bounds()
            self.condition.notify_all()

    def burn(self, token, *, outcome="failure", details=None):
        self.settle(token, self.active[token]["reserved_usd"], outcome=outcome, details=details)

    def status(self):
        with self.condition:
            reserved = sum(x["reserved_usd"] for x in self.active.values())
            return {**self.config, "calls": self.calls, "settled_usd": self.spent,
                    "inflight_reserved_usd": reserved, "committed_usd": self.spent+reserved,
                    "remaining_uncommitted_usd": self.config["ceiling_usd"]-self.spent-reserved,
                    "inflight_batches": len(self.active)}

    def close(self):
        if not self._owner.closed:
            fcntl.flock(self._owner.fileno(), fcntl.LOCK_UN)
            self._owner.close()
