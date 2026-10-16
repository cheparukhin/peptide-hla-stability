"""Hard-bounded milestone-1 verification/profile. Never launches the full cohort.

Run only with an orchestration budget approval and a frozen panel JSON:
  modal run modal_app/round2_boltz.py --panel ... --run-id ... --budget-usd 2
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import modal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pepstab.boltz_embeddings import json_bytes, sha256_bytes, sha256_file

app = modal.App("pepstab-round2-boltz-audit")
image = (modal.Image.debian_slim(python_version="3.12").uv_pip_install("boltz==2.1.1")
         .add_local_python_source("pepstab", "round2_boltz_worker"))
weights = modal.Volume.from_name("pepstab-boltz-weights")
msas = modal.Volume.from_name("pepstab-hla-msa")
outputs = modal.Volume.from_name("pepstab-round2-boltz-embeddings", create_if_missing=True)
RATE_USD_PER_HOUR = 1.10 + 4 * .0473 + 24 * .008
HARD_TIMEOUT_S = 900


@app.function(image=image, gpu="A10G", cpu=(4, 4), memory=(24576, 24576),
              volumes={"/weights": weights.with_mount_options(read_only=True),
                       "/msa": msas.with_mount_options(read_only=True), "/embeddings": outputs},
              timeout=HARD_TIMEOUT_S, scaledown_window=2, max_containers=1, retries=0)
def pilot(panel: dict, run_id: str, mode: str = "verify") -> dict:
    if mode not in ("verify", "profile"):
        raise ValueError("only bounded verification/profile modes allowed")
    limit = 16 if mode == "verify" else 64
    if not 1 <= len(panel["cases"]) <= limit:
        raise ValueError(f"panel exceeds hard {mode} cap {limit}")
    if any(c["split"] != "train" for c in panel["cases"]):
        raise ValueError("original training cases only")
    if not run_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("unsafe run id")
    start = time.monotonic()
    destination = Path("/embeddings") / run_id
    if (destination / "report.json").exists():
        prior = json.loads((destination / "report.json").read_text())
        if prior.get("status") == "passed":
            raise ValueError("completed run exists; use its validated cache rather than rerun")
    destination.mkdir(parents=True, exist_ok=True)
    work = Path("/tmp") / run_id
    work.mkdir(parents=True, exist_ok=True)
    config = {"cases": panel["cases"], "mode": mode, "output": str(destination), "work": str(work),
              "cache": "/weights/boltz", "msa_root": "/msa/ectodomain_stage4c"}
    config_path = work / "config.json"
    config_path.write_bytes(json_bytes(config))
    result = subprocess.run([sys.executable, "-m", "round2_boltz_worker", str(config_path)],
                            capture_output=True, text=True, timeout=HARD_TIMEOUT_S - 60)
    (destination / "stdout.txt").write_text(result.stdout)
    (destination / "stderr.txt").write_text(result.stderr)
    report_path = destination / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"status": "failed", "cases": []}
    report.update(returncode=result.returncode, container_wall_s=time.monotonic() - start,
                  rate_usd_per_hour=RATE_USD_PER_HOUR, runtime_estimate_usd=(time.monotonic() - start + 2) * RATE_USD_PER_HOUR / 3600,
                  hard_runtime_cap_usd=(HARD_TIMEOUT_S + 2) * RATE_USD_PER_HOUR / 3600,
                  output_volume="pepstab-round2-boltz-embeddings", output_prefix=run_id,
                  app_id=os.environ.get("MODAL_APP_ID"), panel_sha256=sha256_bytes(json_bytes(panel)),
                  stdout_tail=result.stdout[-4000:], stderr_tail=result.stderr[-6000:])
    report_path.write_bytes(json_bytes(report))
    outputs.commit()
    return report


@app.local_entrypoint()
def main(panel: str, run_id: str, budget_usd: float, mode: str = "verify"):
    if budget_usd <= 0 or budget_usd > 5:
        raise ValueError("this engineering app only accepts a positive budget <= $5")
    hard_cost = (HARD_TIMEOUT_S + 2) * RATE_USD_PER_HOUR / 3600
    if hard_cost > budget_usd:
        raise ValueError("budget cannot cover the hard runtime bound")
    panel_path = Path(panel).resolve()
    frozen = json.loads(panel_path.read_text())
    source_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in
                    (ROOT / "pepstab/boltz_embeddings.py", Path(__file__).resolve(),
                     ROOT / "modal_app/round2_boltz_worker.py", panel_path)}
    # One remote call, no retries or hidden fanout. Image builds billed separately.
    result = pilot.remote(frozen, run_id, mode)
    result["approved_call_budget_usd"] = budget_usd
    result["source_hashes"] = source_hashes
    target = ROOT / "reports/further_research/round2/extractor" / f"{run_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(json_bytes(result))
    print(json.dumps({"report": str(target), "status": result["status"], "cases": len(result["cases"]),
                      "estimate_usd": result["runtime_estimate_usd"]}))
    if result.get("status") != "passed" or result["returncode"] != 0:
        raise RuntimeError(f"engineering gate failed; inspect {target}")
