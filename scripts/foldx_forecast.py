#!/usr/bin/env python
"""Stage 8: forecast the FoldX run. **Genuinely free — it never touches Modal.**

This exists because `modal run ...::score --dry-run` is *not* free, and it is
worth being exact about why. `modal run` builds and validates the image before
a single line of the local entrypoint executes, and this app's image carries an
87 MB FoldX layer. No *worker* container starts, nothing scores, but a builder
runs. This project has already been bitten once by a `--dry-run` that quietly
started a container, so the free path is a script that cannot start one: it
does not import `modal` at all.

Everything it needs is committed:

* `data/structural_cohort.csv` for the cohort size and the two halves,
* `reports/ectodomain_rates.json` for the **metered** rates — never published
  list rates; an unmeasured rate assumption cost this project a 4.6x error.

    .venv/bin/python scripts/foldx_forecast.py --seconds-per-structure 30
    .venv/bin/python scripts/foldx_forecast.py \
        --measured-from reports/stage8_foldx_smoke_a-cheparukhin.json --sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COHORT_CSV = REPO / "data" / "structural_cohort.csv"
RATES_PATH = REPO / "reports" / "ectodomain_rates.json"
COHORT_TOTAL = 28166


def load_rates() -> dict[str, float]:
    raw = json.loads(RATES_PATH.read_text())
    return {
        "cpu_hour": float(raw["cpu_hour_cost"]),
        "mem_gib_hour": float(raw["mem_gib_hour_cost"]),
    }


def cohort_sizes() -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in csv.DictReader(COHORT_CSV.open()):
        counts[row["profile"]] = counts.get(row["profile"], 0) + 1
    return counts


def forecast(
    n: int,
    seconds_per_structure: float,
    cpu: float,
    memory_gib: float,
    containers: int,
    procs: int,
    startup_s: float = 90.0,
    margin: float = 1.25,
    chunk: int = 0,
) -> dict:
    """Core-hours and dollars.

    Modal bills the cores and memory a container **reserves**, for as long as
    it lives — not the cores it keeps busy. So cost is
    ``containers x cpu x wall``, and under-filling a container is a real
    expense rather than something the arithmetic hides. ``startup_s`` covers
    image pull and Volume mount; the stage 4c fold measured 67-79 s of
    per-shard startup on a far heavier image, so 90 s is deliberately generous.

    ``chunk`` is not cosmetic, and leaving it out was a real defect in this
    function. ``score_chunk`` runs ``Pool(procs)`` over **one chunk**, so a
    chunk smaller than the pool leaves ``procs - chunk`` reserved cores idle
    while work remains — and reserved cores are billed. This function used to
    assume perfect packing, which priced the repaired arm at
    ``--chunk 32 --procs 64`` at $205 when it actually costs $386: an 88%
    overspend, straight through the stage's $250 stop threshold, reported as
    comfortably inside it. Pass ``chunk`` and the fill factor is applied; the
    default of 0 means "assume the chunk fills the pool" and is only right when
    it does.
    """
    rates = load_rates()
    fill = min(chunk, procs) / procs if chunk else 1.0
    per_container = n / containers
    busy_s = per_container / procs * seconds_per_structure / fill
    wall_s = busy_s + startup_s
    container_hours = containers * wall_s / 3600
    core_hours = container_hours * cpu
    mem_gib_hours = container_hours * memory_gib
    usd = core_hours * rates["cpu_hour"] + mem_gib_hours * rates["mem_gib_hour"]
    return {
        "n": n,
        "seconds_per_structure": seconds_per_structure,
        "cpu": cpu,
        "memory_gib": memory_gib,
        "containers": containers,
        "procs": procs,
        "chunk": chunk,
        "pool_fill": fill,
        "parallel_processes": containers * procs,
        "wall_h": wall_s / 3600,
        "container_hours": container_hours,
        "core_hours": core_hours,
        "mem_gib_hours": mem_gib_hours,
        "usd": usd,
        "usd_with_margin": usd * margin,
        "margin": margin,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds-per-structure", type=float, default=0.0,
                    help="measured, from ::smoke. No default: an assumed per-unit "
                         "time is how this project's last 4.6x cost error happened.")
    ap.add_argument("--measured-from", type=Path, default=None,
                    help="a reports/stage8_foldx_smoke_<profile>.json")
    ap.add_argument("--cpu", type=float, default=32.0)
    ap.add_argument("--memory-gib", type=float, default=16.0)
    ap.add_argument("--containers", type=int, default=10)
    ap.add_argument("--procs", type=int, default=0, help="FoldX processes per container")
    ap.add_argument("--chunk", type=int, default=0,
                    help="structures per chunk. A chunk below --procs leaves "
                         "reserved cores idle and billed; pass the value you will "
                         "actually launch with, or the forecast assumes perfect "
                         "packing and under-reports.")
    ap.add_argument("--n", type=int, default=0, help="structures; default the full cohort")
    ap.add_argument("--repair-multiple", type=float, default=0.0,
                    help="measured RepairPDB runtime multiple, to price that variant")
    ap.add_argument("--sweep", action="store_true", help="price a range of shapes")
    args = ap.parse_args()

    seconds = args.seconds_per_structure
    measured = False
    if args.measured_from:
        data = json.loads(args.measured_from.read_text())
        seconds = float(data["seconds_per_structure_process"])
        measured = True
        if not args.repair_multiple and data.get("repair_runtime_multiple"):
            args.repair_multiple = float(data["repair_runtime_multiple"])
    if not seconds:
        print("error: --seconds-per-structure or --measured-from is required.",
              file=sys.stderr)
        return 2

    procs = args.procs or int(args.cpu)
    sizes = cohort_sizes()
    total = sum(sizes.values())
    assert total == COHORT_TOTAL, f"cohort is {total}, expected {COHORT_TOTAL}"
    n = args.n or total
    rates = load_rates()

    print(f"cohort            {total:,} pairs "
          f"({', '.join(f'{k}={v:,}' for k, v in sorted(sizes.items()))})")
    print(f"per structure     {seconds:.2f} s of FoldX process time  "
          f"[{'MEASURED' if measured else 'ASSUMED'}]")
    print(f"metered rates     ${rates['cpu_hour']:.5f}/core-h, "
          f"${rates['mem_gib_hour']:.5f}/GiB-h  "
          f"[{RATES_PATH.relative_to(REPO)}]")
    print()

    variants = [("AnalyseComplex only", 1.0)]
    if args.repair_multiple:
        variants.append((f"RepairPDB + AnalyseComplex ({args.repair_multiple:.1f}x)",
                         args.repair_multiple))

    shapes = [(args.cpu, args.containers, procs)]
    if args.sweep:
        shapes = [(c, k, int(c)) for c in (8.0, 16.0, 32.0, 64.0) for k in (5, 10, 20)]

    header = (f"{'variant':<34}{'shape':>22}{'fill':>6}{'wall h':>9}{'core-h':>9}"
              f"{'$':>9}{'$+25%':>9}")
    print(header)
    print("-" * len(header))
    underfilled = False
    for label, factor in variants:
        for cpu, containers, p in shapes:
            f = forecast(n, seconds * factor, cpu, args.memory_gib, containers, p,
                         chunk=args.chunk)
            shape = f"{containers}x{cpu:g}c/{p}p"
            print(f"{label:<34}{shape:>22}{f['pool_fill']:>6.0%}"
                  f"{f['wall_h']:>9.2f}{f['core_hours']:>9.1f}"
                  f"{f['usd']:>9.2f}{f['usd_with_margin']:>9.2f}")
            underfilled |= f["pool_fill"] < 1.0
    print()
    if not args.chunk:
        print("No --chunk given, so the pool is assumed to fill perfectly. It only "
              "does when chunk >= procs; pass --chunk to price what you will "
              "actually launch.")
    elif underfilled:
        print(f"WARNING: --chunk {args.chunk} fills only "
              f"{min(args.chunk, procs) / procs:.0%} of a {procs}-process pool. "
              f"score_chunk runs Pool(procs) over one chunk, so the idle cores are "
              f"reserved and billed. Raise --chunk to {procs} or a multiple of it; "
              "modal_app/foldx_scoring.py::score now refuses the under-filled shape.")
    print("Per workspace, each scoring its own half: divide the cost by two and "
          "keep the wall time (the halves run in parallel).")
    print("Modal bills reserved cores for the container's whole life, so a wider "
          "container is only cheaper if it is actually filled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
