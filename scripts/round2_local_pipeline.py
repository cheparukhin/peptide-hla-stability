"""Durable local-only continuation of the frozen round-two assessment.

Waits for the authorized cloud jobs' completed inputs. Never launches Modal or
retries a failed child. Each scientific runner validates its own frozen inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/further_research/round2"


def input_state(out=OUT):
    out = Path(out)
    failures = list((out / "sequence").glob("fold_*/failure.json"))
    if failures:
        return "failed", [str(path) for path in failures]
    cache_path = out / "extractor/full_training/feature_bundles.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    ledger_path = out / "extractor/full_training/extraction_budget_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    if ledger.get("halted") or any(row.get("status") == "failed" for row in ledger.get("completed", {}).values()):
        return "failed", [str(ledger_path)]
    complete_cache = (cache.get("status") == "complete" and cache.get("completed_rows") == 19716
                      and cache.get("expected_rows") == 19716)
    sequence = all((out / f"sequence/fold_{fold}/completed.json").exists() and
                   (out / f"sequence/fold_{fold}/repeats_completed.json").exists() for fold in range(5))
    return ("ready" if complete_cache and sequence else "waiting"), {
        "cached_rows": cache.get("completed_rows", 0), "cache_complete": complete_cache,
        "sequence_with_repeats_complete": sequence}


def atomic_json(path, item):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(item, indent=2)+"\n")
    temp.replace(path)


def run_stage(name, commands, state, save, max_workers=3):
    pending = list(enumerate(commands))
    active = {}
    state.update(stage=name, stage_status="running", active_children=[])
    save()
    env = {**os.environ, **{key: "2" for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")}}
    try:
        while pending or active:
            while pending and len(active) < max_workers:
                index, command = pending.pop(0)
                log_path = OUT / "pipeline" / f"{name}_{index}.log"
                with log_path.open("a") as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT)
                active[child] = {"index": index, "pid": child.pid, "command": command,
                                 "log": str(log_path), "started_at": datetime.now(timezone.utc).isoformat()}
                state["active_children"] = list(active.values())
                save()
            for child, record in list(active.items()):
                code = child.poll()
                if code is None:
                    continue
                del active[child]
                state.setdefault("finished_children", []).append({**record, "stage": name, "exit_code": code})
                state["active_children"] = list(active.values())
                save()
                if code:
                    raise RuntimeError(f"{name} child {record['index']} failed with exit {code}; see {record['log']}")
            if active:
                time.sleep(10)
    finally:
        # These are this controller's children only. A failed stage cannot leave
        # unobserved local fits consuming resources while new work is dispatched.
        for child in active:
            child.terminate()
        for child in active:
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    state["stage_status"] = "complete"
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-hours", type=float, default=12.)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    directory = OUT / "pipeline"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"pid": os.getpid(), "started_at_utc": datetime.now(timezone.utc).isoformat(),
                 "status": "waiting_for_inputs", "cloud_spend_usd": 0,
                 "workers": args.workers, "canonical_plan": "reports/NEXT_RESEARCH_PLAN.md"}
        def save():
            state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(directory / "status.json", state)
        deadline = time.monotonic() + args.wait_hours * 3600
        try:
            while True:
                condition, detail = input_state()
                state["input_state"] = detail
                save()
                if condition == "failed":
                    raise RuntimeError(f"upstream failure requires diagnosis: {detail}")
                if condition == "ready":
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("completed inputs did not arrive before declared wait deadline")
                time.sleep(30)
            state["status"] = "running"
            python = str(ROOT / ".venv/bin/python")
            manifest = str(OUT / "extractor/full_training/feature_bundles.json")
            run_stage("embedding", [[python, "scripts/round2_embedding_screen.py", "--fold", str(fold),
                "--production-manifest", manifest] for fold in range(5)], state, save, args.workers)
            run_stage("main_report", [[python, "scripts/round2_results.py", "--embedding-directory", str(OUT / "embedding"),
                "--output-directory", str(OUT / "results/main")]], state, save, 1)
            run_stage("complementarity", [[python, "scripts/round2_complementarity.py", "--fold", str(fold),
                "--production-manifest", manifest] for fold in range(5)], state, save, args.workers)
            run_stage("complementarity_report", [[python, "scripts/round2_complementarity_report.py"]], state, save, 1)
            state.update(status="complete", scope="frozen main and complementarity development assessments",
                         remaining="transfer selection/refits and independent confirmation gates remain with orchestrator")
        except Exception as exc:
            state.update(status="failed", error=repr(exc))
            raise
        finally:
            save()


if __name__ == "__main__":
    main()
