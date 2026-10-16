"""Bounded round-two PairLogit workers; local fitting/selection stays on host.

Smoke: MODAL_PROFILE=a-cheparukhin .venv/bin/modal run modal_app/round2_sequence.py::smoke
Full run is separately gated by a measured forecast and the root cost decision.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import time
import threading

import modal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/further_research/round2/sequence"
RATE_HOUR = .0473*2 + .008*4
TASK_TIMEOUT = 1000
STARTUP_TIMEOUT = 60
SCALEDOWN_SECONDS = 2
MAX_MEMBERS = 900
CEILING_USD = 18.
SMOKE_RESERVE_USD = 1.
BUILD_RESERVE_USD = .35

app = modal.App("pepstab-round2-sequence")
image = (modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.3", "pandas==3.0.6", "scipy==1.18.1",
                 "catboost==1.2.10", "threadpoolctl==3.7.0")
    .env({key: "2" for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
         "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}))
for module in ("__init__.py", "augment.py", "data.py", "features.py", "mlp.py", "evaluation.py",
               "splits.py", "nested.py", "phase_a.py", "round2_nested.py", "round2_sequence.py"):
    image = image.add_local_file(ROOT/"pepstab"/module, f"/root/pepstab/{module}", copy=True)


@app.function(image=image, cpu=(2., 2.), memory=(4096, 4096),
              timeout=TASK_TIMEOUT, startup_timeout=STARTUP_TIMEOUT,
              max_containers=30, min_containers=0, buffer_containers=0,
              scaledown_window=SCALEDOWN_SECONDS, retries=0)
def ranker_member(task: dict) -> dict:
    import importlib.metadata
    import platform
    import resource
    from pepstab.round2_sequence import ranker_task
    started = time.perf_counter()
    result = ranker_task(task)
    result.update({"worker_seconds": time.perf_counter()-started,
                   "backend": "Modal Linux CPU 2 cores / 4 GiB hard cap",
                   "platform": platform.platform(),
                   "catboost_version": importlib.metadata.version("catboost"),
                   "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
    return result


def worst_cost(members=MAX_MEMBERS):
    return {"member_count": members, "requested_cpu_cores": 2, "requested_gib": 4,
            "cpu_hour_usd": .0473, "gib_hour_usd": .008,
            "task_timeout_seconds": TASK_TIMEOUT, "startup_seconds_per_member": STARTUP_TIMEOUT,
            "scaledown_seconds_per_member": SCALEDOWN_SECONDS,
            "task_and_per_call_overhead_upper_usd": members*(TASK_TIMEOUT+STARTUP_TIMEOUT+SCALEDOWN_SECONDS)*RATE_HOUR/3600,
            "smoke_reserve_usd": SMOKE_RESERVE_USD, "image_build_reserve_usd": BUILD_RESERVE_USD,
            "total_upper_usd": members*(TASK_TIMEOUT+STARTUP_TIMEOUT+SCALEDOWN_SECONDS)*RATE_HOUR/3600+SMOKE_RESERVE_USD+BUILD_RESERVE_USD,
            "scope_ceiling_usd": CEILING_USD, "automatic_retries": 0,
            "reservation_policy": "dynamic worst-case reservation and measured-runtime settlement; full-graph worst case exceeds allocated cap, so completion is not guaranteed by the ceiling",
            "rates_source": "https://modal.com/pricing", "rates_checked_date": "2026-10-12"}


class BoundedBackend:
    def __init__(self, out=OUT):
        self.out = Path(out)
        self.lock = threading.Lock()
        self.ledger = self.out/"modal_ledger.jsonl"
        from pepstab.round2_budget import ReservationBudget
        budget_path = self.out/"modal_budget.jsonl"
        if self.ledger.exists() and self.ledger.stat().st_size and not budget_path.exists():
            raise ValueError("historical paid-call ledger needs reconciliation before dynamic budgeting")
        self.call_upper = (TASK_TIMEOUT+STARTUP_TIMEOUT+SCALEDOWN_SECONDS)*RATE_HOUR/3600
        self.budget = ReservationBudget(budget_path, ceiling_usd=CEILING_USD,
            call_upper_usd=self.call_upper, initial_reserve_usd=SMOKE_RESERVE_USD+BUILD_RESERVE_USD,
            max_calls=MAX_MEMBERS)

    def __call__(self, tasks):
        token = self.budget.reserve(len(tasks), context=tasks[0]["context"])
        try:
            return self._run_reserved(tasks, token)
        finally:
            # Unexpected parsing, file-I/O, or caller failures must not leave a
            # phantom in-flight batch making other worker threads wait forever.
            with self.budget.condition:
                if token in self.budget.active:
                    self.budget.burn(token, outcome="unhandled_controller_failure")

    def _run_reserved(self, tasks, token):
        try:
            results = list(ranker_member.map(tasks, return_exceptions=True))
            if len(results) != len(tasks):
                raise RuntimeError("remote member result count differs from submitted calls")
        except BaseException as exc:
            self.budget.burn(token, outcome="map_failure", details={"error": repr(exc)})
            raise
        failures = []
        charge = 0.
        with self.lock, self.ledger.open("a") as handle:
            for task, result in zip(tasks, results):
                if not isinstance(result, BaseException):
                    try:
                        seconds = float(result["worker_seconds"])
                        if not math.isfinite(seconds) or not 0 <= seconds <= TASK_TIMEOUT:
                            raise ValueError("runtime telemetry outside declared worker timeout")
                    except (KeyError, TypeError, ValueError) as exc:
                        result = exc
                if isinstance(result, BaseException):
                    record = {"event": "failure", "context": task["context"],
                              "seed": task["seed"], "stop_fold": task["stop_fold"], "error": repr(result),
                              "charged_usd": self.call_upper, "reservation_token": token}
                    charge += self.call_upper
                    failures.append(record)
                else:
                    cost = (seconds+STARTUP_TIMEOUT+SCALEDOWN_SECONDS)*RATE_HOUR/3600
                    charge += cost
                    record = {"event": "completed", "context": task["context"],
                              "seed": task["seed"], "stop_fold": task["stop_fold"],
                              "worker_seconds": seconds, "charged_usd": cost,
                              "reservation_token": token,
                              "runtime_estimate_usd": seconds*RATE_HOUR/3600,
                              "max_rss_mib": result["max_rss_mib"]}
                handle.write(json.dumps(record)+"\n")
        self.budget.settle(token, charge, outcome="member_failures" if failures else "success",
                           details={"n_failures": len(failures), "context": tasks[0]["context"]})
        with self.lock:
            temporary = self.out/"modal_budget_status.tmp"
            temporary.write_text(json.dumps(self.budget.status(), indent=2)+"\n")
            temporary.replace(self.out/"modal_budget_status.json")
        if failures:
            raise RuntimeError(f"{len(failures)} remote members failed; complete ensemble refused; see ledger")
        return results


@app.local_entrypoint()
def smoke():
    import numpy as np
    import pandas as pd
    from pepstab.data import load_with_splits
    from pepstab import nested
    from pepstab.round2_nested import unlabelled
    from pepstab.round2_sequence import candidates, ranker_task
    if os.environ.get("MODAL_PROFILE") != "a-cheparukhin":
        raise ValueError("set MODAL_PROFILE=a-cheparukhin to keep billing on the agreed workspace")
    OUT.mkdir(parents=True, exist_ok=True)
    train = load_with_splits().query("split=='train'").reset_index(drop=True)
    outer = nested.outer_folds(train)
    fit, held = train[outer != 0], train[outer == 0]
    fold = nested.inner_folds(fit, 3)
    columns = ["pair_id", "cluster_id", "allele", "peptide", "hla_pseudoseq", "y_log1p"]
    real = {"context": "smoke/full_size_depth8", "stop_fold": 0, "seed": 0,
            "candidate": candidates()["pairlogit"][1], "iterations": 2000,
            "fit": fit.loc[fold != 0, columns].to_dict("records"),
            "stop": fit.loc[fold == 0, columns].to_dict("records"),
            "held": unlabelled(held[columns]).to_dict("records")}
    # Synthetic sequences make the platform-equivalence check cheap and independent
    # of downstream labels/outcomes; each predicted cluster is disjoint.
    rng = np.random.default_rng(19)
    aa = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    tiny = pd.DataFrame({"pair_id": range(180), "cluster_id": range(180),
                        "allele": ["A", "B", "C"]*60,
                        "peptide": ["".join(x) for x in rng.choice(aa, (180, 9))],
                        "hla_pseudoseq": ["A"*34, "C"*34, "D"*34]*60,
                        "y_log1p": rng.uniform(0, 3, 180)})
    small = {"context": "smoke/platform_equivalence", "stop_fold": 0, "seed": 0,
             "candidate": candidates()["pairlogit"][1], "iterations": 20,
             "fit": tiny.iloc[:100].to_dict("records"), "stop": tiny.iloc[100:140].to_dict("records"),
             "held": unlabelled(tiny.iloc[140:]).to_dict("records")}
    local = ranker_task(small)
    started = time.perf_counter()
    outputs = list(ranker_member.map([small, small, real], return_exceptions=True))
    if any(isinstance(x, BaseException) for x in outputs):
        failures = [repr(x) for x in outputs if isinstance(x, BaseException)]
        (OUT/"modal_smoke_failure.json").write_text(json.dumps({"failures": failures}, indent=2))
        raise RuntimeError(str(failures))
    error = float(np.max(np.abs(np.asarray(local["predictions"])-outputs[0]["predictions"])))
    repeat_error = float(np.max(np.abs(np.asarray(outputs[0]["predictions"])-outputs[1]["predictions"])))
    # Tree floating arithmetic can differ across architectures; declare a strict
    # implementation tolerance, not a universal performance noise floor.
    tolerance = 1e-6
    passed = error <= tolerance and repeat_error <= tolerance
    result = {"passed": passed, "local_vs_remote_max_abs": error,
              "remote_repeat_max_abs": repeat_error, "tolerance": tolerance,
              "total_wall_seconds": time.perf_counter()-started,
              "outputs": [{k:v for k,v in x.items() if k not in ("predictions", "pair_ids")} for x in outputs],
              "successful_worker_runtime_estimate_usd": sum(x["worker_seconds"] for x in outputs)*RATE_HOUR/3600,
              "smoke_upper_usd": 3*(TASK_TIMEOUT+STARTUP_TIMEOUT+SCALEDOWN_SECONDS)*RATE_HOUR/3600+BUILD_RESERVE_USD,
              "full_graph_worst_case": worst_cost(),
              "forecast_1_25_all_at_full_depth8_runtime_usd": 1.25*MAX_MEMBERS*(outputs[2]["worker_seconds"]+STARTUP_TIMEOUT+SCALEDOWN_SECONDS)*RATE_HOUR/3600,
              "full_cohort_launch_authorized": False}
    (OUT/"modal_smoke.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise RuntimeError("platform equivalence tolerance failed; no full launch")


@app.local_entrypoint()
def run(approved: bool = False):
    if not approved:
        raise ValueError("root must approve measured forecast and matching cohort before full launch")
    if os.environ.get("MODAL_PROFILE") != "a-cheparukhin":
        raise ValueError("set MODAL_PROFILE=a-cheparukhin")
    authorization = json.loads((OUT/"launch_authorization.json").read_text())
    if not authorization.get("approved"):
        raise ValueError("root launch authorization is not approved")
    protocol = json.loads((OUT/"protocol.json").read_text())
    if authorization["protocol_hash"] != protocol["protocol_hash"]:
        raise ValueError("authorized cohort/protocol differs from current protocol")
    code_paths = ["pepstab/round2_nested.py", "pepstab/round2_sequence.py",
                  "pepstab/round2_budget.py", "scripts/round2_sequence.py", "modal_app/round2_sequence.py"]
    current_hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in code_paths}
    if authorization.get("code_sha256") != current_hashes:
        raise ValueError("implementation changed after reviewed launch authorization")
    smoke_result = json.loads((OUT/"modal_smoke.json").read_text())
    forecast = json.loads((OUT/"modal_budget_forecast.json").read_text())
    if (not smoke_result["passed"] or authorization.get("budget_ceiling_usd") != CEILING_USD
            or forecast["model_based_total_with_margin_usd"] > CEILING_USD):
        raise ValueError("successful smoke, reviewed forecast, and matching dynamic budget required")
    from threadpoolctl import threadpool_limits
    from scripts.round2_sequence import run_fold
    backend = BoundedBackend()
    with threadpool_limits(limits=2), ThreadPoolExecutor(max_workers=5) as executor:
        jobs = {executor.submit(run_fold, OUT, fold, repeats=2, ranker_backend=backend,
                                ranker_backend_name="ModalLinuxCPU2x4GiB-v1"): fold for fold in range(5)}
        for job in as_completed(jobs):
            fold = jobs[job]
            try:
                job.result()
                print(f"fold {fold} completed", flush=True)
            except Exception as exc:
                (OUT/f"fold_{fold}"/"failure.json").write_text(json.dumps({"error": repr(exc)}, indent=2))
                raise
