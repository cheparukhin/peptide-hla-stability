"""Turn measured Boltz-2 benchmark timings into the stage 4 GPU choice.

Applies HACKATHON_PLAN.md's decision rule -- *pick the cheapest GPU that meets
the deadline and memory requirements* -- to measured per-complex wall time. The
inputs are billed worker seconds per successful complex and the observed failure
rate; everything else is arithmetic:

    cost per success = billed seconds * combined $/s / success rate
    capacity         = folding budget / cost per success
    elapsed hours    = total worker hours / concurrent workers

Rates live in ``RATES`` and must be rechecked against modal.com/pricing before
launch -- the plan says so explicitly, and published rates move.

Reads ``reports/boltz_bench_results.csv`` (written by the Modal benchmark) with
at least the columns ``gpu``, ``complex_id``, ``ok``, ``billed_s``. Writes
``reports/gpu_decision.csv`` and prints the recommendation.

Usage:
    python scripts/gpu_decision.py
    python scripts/gpu_decision.py --budget 330 --panel 2000 --deadline-hours 5 --workers 10
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "reports" / "boltz_bench_results.csv"
OUT = REPO / "reports" / "gpu_decision.csv"

# Published Modal per-second rates, read from modal.com/pricing on 2026-10-03.
# These reproduce HACKATHON_PLAN.md's reference table to the cent once CPU and
# memory are added, so the plan's "recheck rates before launch" step is done --
# but recheck again if the event slips, since published rates move.
GPU_PER_S = {
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "H100": 0.001097,
}
VRAM_GB = {"L40S": 48, "A100-40GB": 40, "A100-80GB": 80, "H100": 80}

# Per the plan's worker assumption: 4 physical cores and 32 GiB host memory.
CPU_PER_CORE_S = 0.0000131
MEM_PER_GIB_S = 0.00000222
WORKER_CORES = 4
WORKER_GIB = 32

_HOST_PER_S = WORKER_CORES * CPU_PER_CORE_S + WORKER_GIB * MEM_PER_GIB_S

RATES = {
    gpu: {
        "gpu_hr": round(per_s * 3600, 4),
        "combined_hr": round((per_s + _HOST_PER_S) * 3600, 4),
        "vram_gb": VRAM_GB[gpu],
    }
    for gpu, per_s in GPU_PER_S.items()
}


def load_results(path: Path) -> dict[str, list[dict]]:
    if not path.exists():
        raise SystemExit(
            f"{path.relative_to(REPO)} not found -- run the Modal benchmark first"
        )
    by_gpu: dict[str, list[dict]] = {}
    for r in csv.DictReader(path.open()):
        by_gpu.setdefault(r["gpu"], []).append(r)
    return by_gpu


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def summarize(gpu: str, runs: list[dict], per_container: int) -> dict:
    """Per-GPU cost, excluding warm-up folds from the steady-state mean.

    The first fold in a container pays weight load and CUDA kernel compilation.
    Averaging it into per-complex cost would overstate a 2,000-complex run,
    where that cost is paid once per container rather than once per complex --
    so it is amortised over ``per_container`` instead.
    """
    steady = [r for r in runs if not _truthy(r.get("is_warmup", ""))]
    pool = steady or runs  # a single-complex batch is all warm-up
    ok = [r for r in pool if _truthy(r["ok"])]
    n, n_ok = len(pool), len(ok)
    if n_ok == 0:
        return {"gpu": gpu, "n": n, "n_ok": 0, "note": "all complexes failed"}

    rate = RATES.get(gpu.replace("!", ""))
    if rate is None:
        raise SystemExit(f"no published rate on file for GPU '{gpu}' -- add it to RATES")

    billed = [float(r["billed_s"]) for r in ok]
    peak = [float(r["peak_mem_gb"]) for r in ok if r.get("peak_mem_gb")]

    # Container overhead: weight load plus the warm-up fold's extra time, paid
    # once per container. Modal bills container load time, so this is real.
    startups = [float(r["startup_s"]) for r in runs if r.get("startup_s")]
    warmups = [
        float(r["fold_s"]) for r in runs if _truthy(r.get("is_warmup", "")) and _truthy(r["ok"])
    ]
    mean_billed = statistics.fmean(billed)
    overhead_s = (statistics.fmean(startups) if startups else 0.0) + (
        max(0.0, statistics.fmean(warmups) - mean_billed) if warmups else 0.0
    )

    combined_per_s = rate["combined_hr"] / 3600.0
    success_rate = n_ok / n
    # Charge failures to the successes: a failed complex still burns worker time.
    per_complex_s = mean_billed + overhead_s / max(1, per_container)
    cost_per_success = per_complex_s * combined_per_s / success_rate

    return {
        "gpu": gpu,
        "n": n,
        "n_ok": n_ok,
        "success_rate": round(success_rate, 4),
        "median_s": round(statistics.median(billed), 1),
        "mean_s": round(mean_billed, 1),
        "p90_s": round(sorted(billed)[int(0.9 * (len(billed) - 1))], 1),
        "overhead_s": round(overhead_s, 1),
        "billed_per_complex_s": round(per_complex_s, 1),
        "peak_mem_gb": round(max(peak), 2) if peak else "",
        "vram_gb": rate["vram_gb"],
        "combined_hr": rate["combined_hr"],
        "cost_per_success": round(cost_per_success, 4),
        "note": "",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--budget", type=float, default=330.0, help="Modal folding budget ceiling, $")
    p.add_argument("--panel", type=int, default=2000, help="target complexes in the production panel")
    p.add_argument("--deadline-hours", type=float, default=5.0, help="wall-clock hours available for folding")
    p.add_argument(
        "--workers",
        type=int,
        default=10,
        help="concurrent Modal workers (Modal's Starter plan caps GPU concurrency at 10)",
    )
    p.add_argument(
        "--per-container",
        type=int,
        default=100,
        help="complexes folded per container in production, over which startup amortises",
    )
    p.add_argument("--results", type=Path, default=RESULTS)
    args = p.parse_args()

    by_gpu = load_results(args.results)
    rows = [summarize(g, r, args.per_container) for g, r in sorted(by_gpu.items())]

    for r in rows:
        if not r.get("cost_per_success"):
            r["panel_cost"] = ""
            r["capacity"] = ""
            r["elapsed_hours"] = ""
            r["meets_budget"] = False
            r["meets_deadline"] = False
            continue
        r["panel_cost"] = round(r["cost_per_success"] * args.panel, 2)
        r["capacity"] = int(args.budget / r["cost_per_success"])
        worker_hours = (
            r["billed_per_complex_s"] * args.panel / r["success_rate"] / 3600.0
        )
        r["elapsed_hours"] = round(worker_hours / args.workers, 2)
        r["meets_budget"] = r["panel_cost"] <= args.budget
        r["meets_deadline"] = r["elapsed_hours"] <= args.deadline_hours

    cols = [
        "gpu", "n", "n_ok", "success_rate", "median_s", "mean_s", "p90_s",
        "overhead_s", "billed_per_complex_s", "peak_mem_gb", "vram_gb",
        "combined_hr", "cost_per_success", "panel_cost", "capacity",
        "elapsed_hours", "meets_budget", "meets_deadline", "note",
    ]
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"wrote {OUT.relative_to(REPO)}\n")

    hdr = f"{'GPU':11} {'ok':>6} {'med s':>7} {'$/cplx':>8} {'panel $':>9} {'cap':>7} {'hours':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if not r.get("cost_per_success"):
            print(f"{r['gpu']:11} {'0/' + str(r['n']):>6}   {r['note']}")
            continue
        print(
            f"{r['gpu']:11} {str(r['n_ok']) + '/' + str(r['n']):>6} "
            f"{r['median_s']:>7} {r['cost_per_success']:>8.3f} "
            f"{r['panel_cost']:>9.2f} {r['capacity']:>7} {r['elapsed_hours']:>7.2f}"
        )

    # The rule: cheapest per success among GPUs that fit the panel in budget,
    # hit the deadline, and never exceeded their own VRAM.
    eligible = [
        r for r in rows
        if r.get("cost_per_success") and r["meets_budget"] and r["meets_deadline"]
        and (not r["peak_mem_gb"] or r["peak_mem_gb"] <= r["vram_gb"])
    ]
    print()
    if not eligible:
        print(
            "No GPU fits both the budget and the deadline at this panel size.\n"
            "Per the hour-5 rule: shrink the panel or report pilot results only."
        )
        for r in rows:
            if not r.get("cost_per_success"):
                continue
            if not r["meets_budget"]:
                print(f"  {r['gpu']}: panel costs ${r['panel_cost']} > ${args.budget} budget")
            if not r["meets_deadline"]:
                print(f"  {r['gpu']}: {r['elapsed_hours']} h > {args.deadline_hours} h deadline at {args.workers} workers")
        return

    best = min(eligible, key=lambda r: r["cost_per_success"])
    print(
        f"CHOOSE {best['gpu']}: ${best['cost_per_success']:.3f} per successful complex, "
        f"${best['panel_cost']:.2f} for {args.panel} complexes "
        f"({100 * best['panel_cost'] / args.budget:.0f}% of the ${args.budget:.0f} ceiling), "
        f"{best['elapsed_hours']:.2f} h at {args.workers} workers."
    )
    others = [r for r in eligible if r["gpu"] != best["gpu"]]
    if others:
        print("Also eligible: " + ", ".join(
            f"{r['gpu']} (${r['cost_per_success']:.3f}/complex)" for r in others
        ))


if __name__ == "__main__":
    main()
