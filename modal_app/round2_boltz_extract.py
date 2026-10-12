"""Frozen-cohort extraction with eight bounded workers and a spend ledger.

Requires the successful 64-case engineering forecast and an explicit frozen
study cohort. It cannot dispatch more than the separately allocated $80.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import modal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pepstab.boltz_embeddings import _atomic_bytes, json_bytes, sha256_bytes, sha256_file

RATE = 1.10 + 4 * .0473 + 24 * .008
TIMEOUT = 2700
SHARD_SIZE = 256
STARTUP_ALLOWANCE = 60
SCALEDOWN_ALLOWANCE = 2
RESERVATION = (TIMEOUT + STARTUP_ALLOWANCE + SCALEDOWN_ALLOWANCE) * RATE / 3600
CONTACT_PATH = ROOT / "reports/further_research/round2/representations/contact_map.json"
app = modal.App("pepstab-round2-boltz-extraction")
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("boltz==2.1.1")
         .add_local_python_source("pepstab", "round2_boltz_worker")
         .add_local_file(str(CONTACT_PATH), "/opt/round2/contact_map.json"))
weights = modal.Volume.from_name("pepstab-boltz-weights")
msas = modal.Volume.from_name("pepstab-hla-msa")
outputs = modal.Volume.from_name("pepstab-round2-boltz-embeddings")


@app.function(image=image, gpu="A10G", cpu=(4, 4), memory=(24576, 24576),
              volumes={"/weights": weights.with_mount_options(read_only=True),
                       "/msa": msas.with_mount_options(read_only=True), "/embeddings": outputs},
              timeout=TIMEOUT, startup_timeout=STARTUP_ALLOWANCE,
              scaledown_window=SCALEDOWN_ALLOWANCE, max_containers=8, retries=0)
def extract_shard(cases: list[dict], run_id: str, shard: int) -> dict:
    if not 1 <= len(cases) <= SHARD_SIZE or any(c["split"] != "train" for c in cases):
        raise ValueError("1-256 frozen training cases per shard required")
    if not run_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("unsafe run id")
    start = time.monotonic()
    destination = Path("/embeddings") / run_id / f"shard_{shard:05d}"
    destination.mkdir(parents=True, exist_ok=True)
    work = Path("/tmp") / f"{run_id}-{shard:05d}"
    work.mkdir(parents=True, exist_ok=True)
    config = {"cases": cases, "mode": "extract", "output": str(destination), "work": str(work),
              "cache": "/weights/boltz", "msa_root": "/msa/ectodomain_stage4c",
              "contact_map_path": "/opt/round2/contact_map.json"}
    config_path = work / "config.json"
    config_path.write_bytes(json_bytes(config))
    returncode, stdout, stderr = -1, "", ""
    try:
        process = subprocess.run([sys.executable, "-m", "round2_boltz_worker", str(config_path)],
                                 capture_output=True, text=True, timeout=TIMEOUT - 60)
        returncode, stdout, stderr = process.returncode, process.stdout, process.stderr
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
        stderr = "Hard worker process timeout (840 seconds)."
    finally:
        (destination / "stdout.txt").write_text(stdout)
        (destination / "stderr.txt").write_text(stderr)
        report_path = destination / "report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {"status": "failed", "cases": []}
        elapsed = time.monotonic() - start
        if returncode != 0:
            report["status"] = "failed"
        report.update(returncode=returncode, container_wall_s=elapsed,
                      runtime_estimate_usd=elapsed * RATE / 3600,
                      charged_upper_usd=(elapsed + STARTUP_ALLOWANCE + SCALEDOWN_ALLOWANCE) * RATE / 3600,
                      shard=shard, requested_case_ids=[c["complex_id"] for c in cases],
                      output_prefix=f"{run_id}/shard_{shard:05d}", app_id=os.environ.get("MODAL_APP_ID"),
                      stdout_tail=stdout[-2000:], stderr_tail=stderr[-4000:])
        report_path.write_bytes(json_bytes(report))
        outputs.commit()
    return report


def _download_bundle(report: dict, directory: Path) -> dict:
    prefix = report["output_prefix"] + "/features"
    directory.mkdir(parents=True, exist_ok=True)
    complete_bytes = b"".join(outputs.read_file(prefix + "/complete.json"))
    complete = json.loads(complete_bytes)
    for key, hash_key in (("array_file", "npz_sha256"), ("metadata_file", "metadata_sha256")):
        name = complete[key]
        if Path(name).name != name:
            raise ValueError("unsafe feature filename")
        payload = b"".join(outputs.read_file(prefix + "/" + name))
        if sha256_bytes(payload) != complete[hash_key]:
            raise ValueError("downloaded compact feature checksum mismatch")
        _atomic_bytes(directory / name, payload)
    _atomic_bytes(directory / "complete.json", complete_bytes)
    manifest = json.loads((directory / complete["metadata_file"]).read_text())
    if manifest["npz_sha256"] != complete["npz_sha256"]:
        raise ValueError("feature manifest mismatch")
    return {"directory": str(directory.resolve()), "metadata_sha256": complete["metadata_sha256"],
            "npz_sha256": complete["npz_sha256"], "n_rows": manifest["n_rows"]}


@app.local_entrypoint()
def main(cohort: str, forecast: str, run_id: str, budget_usd: float = 80., workers: int = 8, pilot_only: bool = False):
    from scripts.round2_boltz_budget import BudgetLedger
    from scripts.round2_boltz_cohort import cases_for_cohort
    if not 1 <= workers <= 8:
        raise ValueError("worker cap is eight")
    cohort_path, forecast_path = Path(cohort).resolve(), Path(forecast).resolve()
    engineering = json.loads(forecast_path.read_text())
    if engineering["gate"] != "engineering_passed_forecast_only_full_cohort_not_launched":
        raise ValueError("successful equivalence and measured64-case forecast required")
    cases = cases_for_cohort(cohort_path)
    batch_forecast_path = forecast_path.with_name("batch_forecast_options.json")
    batch_forecast = json.loads(batch_forecast_path.read_text())
    if len(cases) != engineering["training_pairs"] or len(cases) != batch_forecast["n_cases"]:
        raise ValueError("this approved run must preserve the full original-training cohort")
    selected_forecast = next(o for o in batch_forecast["options"] if o["shard_size"] == SHARD_SIZE)
    predicted = selected_forecast["ledger_with_extra60s_per_shard_usd"] + selected_forecast["shards"] * SCALEDOWN_ALLOWANCE * RATE / 3600
    if predicted > budget_usd:
        raise ValueError(f"frozen cohort conservative forecast ${predicted:.2f} exceeds allocation ${budget_usd:.2f}")
    shards = [cases[i:i + SHARD_SIZE] for i in range(0, len(cases), SHARD_SIZE)]
    out = cohort_path.parent
    ledger_path = out / "extraction_budget_ledger.json"
    state = json.loads(ledger_path.read_text()) if ledger_path.exists() else None
    ledger = BudgetLedger(budget_usd, state)
    fingerprint = {"run_id": run_id, "cohort_sha256": sha256_file(cohort_path), "forecast_sha256": sha256_file(forecast_path),
                   "batch_forecast_sha256": sha256_file(batch_forecast_path), "shard_size": SHARD_SIZE,
                   "contact_map_sha256": sha256_file(CONTACT_PATH), "n_cases": len(cases), "shards": len(shards),
                   "source_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                                     (ROOT / "pepstab/boltz_embeddings.py", ROOT / "pepstab/round2_representations.py",
                                      ROOT / "modal_app/round2_boltz_worker.py", Path(__file__).resolve())}}
    if state is not None and state.get("fingerprint") != fingerprint:
        raise ValueError("resume input/code fingerprint changed")
    if ledger.state["reservations"]:
        raise ValueError("unresolved prior reservations: reconcile running apps before resume")
    ledger.state["fingerprint"] = fingerprint
    ledger.state["conservative_forecast_usd"] = predicted
    ledger.state["reservation_per_call_usd"] = RESERVATION
    def save_ledger():
        _atomic_bytes(ledger_path, json_bytes(ledger.state))
    save_ledger()
    aggregate_path = out / "feature_bundles.json"
    aggregate = json.loads(aggregate_path.read_text()) if aggregate_path.exists() else {
        "schema_version": "pepstab-round2-feature-bundles-v1", "cohort_sha256": fingerprint["cohort_sha256"],
        "expected_rows": len(cases), "completed_rows": 0, "status": "partial", "feature_bundles": [], "bundles": []}
    pending = [i for i in range(len(shards)) if ledger.state["completed"].get(str(i), {}).get("status") != "passed"]
    if pilot_only:
        pending = [i for i in pending if i == 0]
        workers = 1
    elif ledger.state["completed"].get("0", {}).get("status") != "passed":
        raise ValueError("first256-case shard must pass using --pilot-only before fanout")
    if any(r.get("status") == "failed" for r in ledger.state["completed"].values()):
        raise ValueError("previous failed shards need explicit diagnosis; automatic retries are disabled")
    futures = {}
    halted = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or futures:
            while pending and len(futures) < workers and not halted:
                shard = pending[0]
                try:
                    ledger.reserve(str(shard), RESERVATION)
                except ValueError:
                    if not futures:
                        halted = True
                    break
                pending.pop(0)
                save_ledger()
                futures[pool.submit(extract_shard.remote, shards[shard], run_id, shard)] = shard
            if not futures:
                break
            done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                shard = futures.pop(future)
                try:
                    report = future.result()
                    _atomic_bytes(out / f"shard_{shard:05d}.json", json_bytes(report))
                    if report["status"] != "passed" or len(report["cases"]) != len(shards[shard]):
                        ledger.fail(str(shard), report.get("error", report.get("stderr_tail", "failed shard")))
                        halted = True
                    else:
                        bundle = _download_bundle(report, ROOT / "features/round2_boltz" / run_id / f"shard_{shard:05d}")
                        ledger.settle(str(shard), report["charged_upper_usd"], {"status": "passed", "rows": len(shards[shard]), "bundle": bundle})
                        aggregate["bundles"].append(bundle)
                        aggregate["feature_bundles"].append(bundle["directory"])
                        aggregate["completed_rows"] += bundle["n_rows"]
                        aggregate["status"] = "complete" if aggregate["completed_rows"] == len(cases) else "partial"
                        _atomic_bytes(aggregate_path, json_bytes(aggregate))
                        print(json.dumps({"shard": shard, "rows": aggregate["completed_rows"], "expected": len(cases),
                                          "charged_upper_usd": ledger.state["spent_upper_usd"]}), flush=True)
                except Exception as exc:
                    if str(shard) in ledger.state["reservations"]:
                        ledger.fail(str(shard), f"{type(exc).__name__}: {exc}")
                    halted = True
                    print(f"Dispatch halted after shard {shard}: {exc}", flush=True)
                save_ledger()
    if pilot_only and ledger.state["completed"].get("0", {}).get("status") == "passed":
        print("First production shard passed; fanout remains gated on orchestrator budget approval.", flush=True)
        return
    if aggregate["status"] != "complete":
        raise RuntimeError("extraction incomplete; budget preserved and completed shard artifacts retained")
