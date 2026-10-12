"""Pure budget reservation accounting used by the bounded extraction controller."""
from __future__ import annotations

import math


class BudgetLedger:
    def __init__(self, limit_usd: float, state: dict | None = None):
        if not math.isfinite(limit_usd) or not 0 < limit_usd <= 80:
            raise ValueError("full-cache allocation must be positive and at most $80")
        self.state = state or {"limit_usd": limit_usd, "spent_upper_usd": 0., "reservations": {}, "completed": {}}
        if self.state["limit_usd"] != limit_usd:
            raise ValueError("cannot change a resumed ledger budget")

    def reserve(self, key: str, bound: float):
        if self.state.get("halted"):
            raise ValueError("budget ledger halted; manual reconciliation required")
        if not math.isfinite(bound) or bound <= 0 or key in self.state["completed"] or key in self.state["reservations"]:
            raise ValueError("invalid or duplicate budget reservation")
        if self.state["spent_upper_usd"] + sum(self.state["reservations"].values()) + bound > self.state["limit_usd"] + 1e-9:
            raise ValueError("remaining budget cannot cover another bounded shard")
        self.state["reservations"][key] = bound

    def settle(self, key: str, charged_upper: float, record: dict):
        reserved = self.state["reservations"].pop(key)
        if not math.isfinite(charged_upper) or charged_upper < 0 or charged_upper > reserved + 1e-9:
            # Keep finite, conservative accounting and make the stop durable.
            self.state["spent_upper_usd"] += max(reserved, charged_upper) if math.isfinite(charged_upper) else reserved
            self.state["completed"][key] = {"status": "failed", "charged_upper_usd": reserved,
                                            "error": "invalid or excess measured charge"}
            self.state["halted"] = True
            raise ValueError("measured charge exceeded reservation or was invalid; stop all further dispatch")
        self.state["spent_upper_usd"] += charged_upper
        self.state["completed"][key] = {**record, "charged_upper_usd": charged_upper, "reserved_usd": reserved}

    def fail(self, key: str, error: str):
        # If a container did not return measurements, spend its entire reservation.
        reserved = self.state["reservations"].pop(key)
        self.state["spent_upper_usd"] += reserved
        self.state["completed"][key] = {"status": "failed", "charged_upper_usd": reserved, "error": error}
