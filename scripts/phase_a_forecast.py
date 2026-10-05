#!/usr/bin/env python
"""Phase A: forecast the Modal run. **Genuinely free -- it never touches Modal.**

This exists because `modal run modal_app/phase_a_experiments.py::forecast` is
*not* free, and it is worth being exact about why. `modal run` builds and
validates the image before a single line of the local entrypoint executes, and
this app's image carries a CPU torch wheel and CatBoost. No *worker* container
starts, nothing fits, but a builder runs. This project has already been bitten
once by a `--dry-run` that quietly started a container, so the free path is a
script that cannot start one: it does not import `modal` at all.

Everything it needs is committed:

* `reports/ectodomain_rates.json` for the **metered** rates -- never published
  list rates; an unmeasured rate assumption cost this project a 4.6x error,
* the unit count, which is arithmetic over the prespecified grid sizes in
  `reports/further_research/phase_a_prespecification.md`.

    .venv/bin/python scripts/phase_a_forecast.py --seconds-per-unit 120
    .venv/bin/python scripts/phase_a_forecast.py --seconds-per-unit 120 --sweep
    .venv/bin/python scripts/phase_a_forecast.py \
        --measured-from reports/further_research/phase_a_smoke_a-cheparukhin.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RATES_PATH = REPO / "reports" / "ectodomain_rates.json"

#: Defaults mirrored from modal_app/phase_a_experiments.py. PROVISIONAL until
#: that app's ``::smoke`` measures the shape; priced here so a shape change can
#: be costed before it is launched rather than after.
WORKER_CPU = 2.0
WORKER_MEMORY_GIB = 4.0
MAX_CONTAINERS = 40
DEFAULT_CHUNK = 8
CONTAINER_STARTUP_S = 45.0
COST_MARGIN = 1.25

#: Paired seeds and inner cluster folds for screening, FURTHER_RESEARCH_PLAN.md
#: Sec. 3.3.
SCREEN_SEEDS = 2
SCREEN_INNER_FOLDS = 3

#: Configurations each arm's grid enumerates, mirroring
#: ``pepstab.phase_a.arm_configs``. The Sec. 3.3 ceiling of 12 is **per
#: learner/objective family**, not per arm, and holds: A1 is one family of 12;
#: A2 one of 4; A3 two, MSE (4 = 2 sampling x 2 stopping) and MSE+ranking
#: (12 = 3 lambda x 2 sampling x 2 stopping); A4 two declared objectives,
#: YetiRank and PairLogit, of 12 each. No family exceeds 12.
ARM_CONFIGS = {"A1": 12, "A2": 4, "A3": 16, "A4": 24}

#: Encoding and outer fold are unit axes, not grid axes, and neither is
#: optional. Both encodings run because the deployed reference ensemble spans
#: both and ensembling every arm identically is a standing project rule -- it
#: is worth about +0.074 median Spearman here. All five outer folds run because
#: hyperparameters are chosen independently inside each outer fitting
#: partition: selecting once on fold 0's partition and reusing it would leak,
#: since fold 1's held-out rows sit inside fold 0's fitting partition.
SCREEN_ENCODINGS = 2
OUTER_FOLDS = 5


def enumerated_screen_units() -> int:
    """The screening unit count: 3,360.

    Pinned against ``pepstab.phase_a.enumerate_units`` by
    ``tests/test_phase_a_modal.py`` so this standalone arithmetic cannot drift
    from the enumerator that actually runs.
    """
    return (sum(ARM_CONFIGS.values()) * SCREEN_ENCODINGS * SCREEN_SEEDS
            * OUTER_FOLDS * SCREEN_INNER_FOLDS)


def load_rates() -> dict[str, float]:
    raw = json.loads(RATES_PATH.read_text())
    return {
        "cpu_hour": float(raw["cpu_hour_cost"]),
        "mem_gib_hour": float(raw["mem_gib_hour_cost"]),
    }


def forecast(
    n_units: int,
    seconds_per_unit: float,
    cpu: float = WORKER_CPU,
    memory_gib: float = WORKER_MEMORY_GIB,
    containers: int = MAX_CONTAINERS,
    chunk: int = DEFAULT_CHUNK,
    startup_s: float = CONTAINER_STARTUP_S,
    margin: float = COST_MARGIN,
) -> dict:
    """Container-hours and dollars. Identical arithmetic to the app's.

    Modal bills the cores and memory a container **reserves**, for as long as
    it lives, not the cores it keeps busy -- so cost is
    ``containers x cpu x wall`` and an under-filled container is a real
    expense. Both fill factors are reported rather than divided away:
    ``chunk_fill`` for a tail chunk, ``container_fill`` for containers
    requested beyond the number of chunks, which Modal never starts and which
    therefore buy no wall time however high the number goes.
    """
    assert chunk >= 1, f"--chunk must be at least 1 unit, got {chunk}"
    assert containers >= 1, f"--containers must be at least 1, got {containers}"
    assert n_units >= 1, f"nothing to forecast: {n_units} units"

    rates = load_rates()
    n_chunks = math.ceil(n_units / chunk)
    chunk_fill = n_units / (n_chunks * chunk)
    active = min(containers, n_chunks)
    container_fill = active / containers
    chunks_per_container = math.ceil(n_chunks / active)
    busy_s = chunks_per_container * chunk * seconds_per_unit
    wall_s = busy_s + startup_s
    container_hours = active * wall_s / 3600
    core_hours = container_hours * cpu
    mem_gib_hours = container_hours * memory_gib
    usd = core_hours * rates["cpu_hour"] + mem_gib_hours * rates["mem_gib_hour"]
    return {
        "n_units": n_units,
        "seconds_per_unit": seconds_per_unit,
        "cpu": cpu,
        "memory_gib": memory_gib,
        "containers_requested": containers,
        "containers_active": active,
        "container_fill": container_fill,
        "chunk": chunk,
        "n_chunks": n_chunks,
        "chunk_fill": chunk_fill,
        "chunks_per_container": chunks_per_container,
        "startup_s": startup_s,
        "wall_h": wall_s / 3600,
        "container_hours": container_hours,
        "core_hours": core_hours,
        "mem_gib_hours": mem_gib_hours,
        "usd": usd,
        "usd_with_margin": usd * margin,
        "margin": margin,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seconds-per-unit", type=float, default=0.0,
                    help="measured, from ::smoke. No default: an assumed per-unit "
                         "time is how this project's last 4.6x cost error happened.")
    ap.add_argument("--measured-from", type=Path, default=None,
                    help="a reports/further_research/phase_a_smoke_<profile>.json")
    ap.add_argument("--n", type=int, default=0,
                    help="units; default the enumerated screen "
                         f"({enumerated_screen_units():,}). Pass the assessment's "
                         "count explicitly once the finalists are chosen.")
    ap.add_argument("--cpu", type=float, default=WORKER_CPU)
    ap.add_argument("--memory-gib", type=float, default=WORKER_MEMORY_GIB)
    ap.add_argument("--containers", type=int, default=MAX_CONTAINERS)
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK,
                    help="units per container call. Units run serially inside a "
                         "chunk, so a chunk larger than the whole set just "
                         "serialises the run onto one container.")
    ap.add_argument("--sweep", action="store_true",
                    help="price a range of container counts")
    args = ap.parse_args()

    seconds = args.seconds_per_unit
    measured = False
    if args.measured_from:
        data = json.loads(args.measured_from.read_text())
        seconds = float(data["seconds_per_unit"])
        measured = True
    if not seconds:
        print("error: --seconds-per-unit or --measured-from is required. There is "
              "no default, deliberately: an assumed per-unit time is how this "
              "project's last 4.6x cost error happened. Run "
              "modal_app/phase_a_experiments.py::smoke first.", file=sys.stderr)
        return 2

    n = args.n or enumerated_screen_units()
    rates = load_rates()

    print(f"units             {n:,}"
          + ("" if args.n else
             f"  ({sum(ARM_CONFIGS.values())} configs x {SCREEN_ENCODINGS} "
             f"encodings x {SCREEN_SEEDS} seeds x {OUTER_FOLDS} outer x "
             f"{SCREEN_INNER_FOLDS} inner)"))
    if not args.n:
        print("                  the encoding and outer-fold axes are part of "
              "the count, not overhead: see the module docstring in "
              "modal_app/phase_a_experiments.py")
    print(f"per unit          {seconds:.2f} s  "
          f"[{'MEASURED' if measured else 'ASSUMED -- run ::smoke'}]")
    print(f"metered rates     ${rates['cpu_hour']:.5f}/core-h, "
          f"${rates['mem_gib_hour']:.5f}/GiB-h  [{RATES_PATH.relative_to(REPO)}]")
    print(f"startup charged   {CONTAINER_STARTUP_S:.0f} s per container")
    print()

    shapes = [(args.cpu, args.containers)]
    if args.sweep:
        shapes = [(c, k) for c in (1.0, 2.0, 4.0) for k in (10, 20, 40, 80)]

    header = (f"{'shape':>20}{'chunks':>8}{'fill':>12}{'wall h':>9}"
              f"{'cont-h':>9}{'$':>9}{'$+25%':>9}")
    print(header)
    print("-" * len(header))
    underfilled = False
    for cpu, containers in shapes:
        f = forecast(n, seconds, cpu, args.memory_gib, containers, args.chunk)
        shape = f"{containers}x{cpu:g}c/{args.memory_gib:g}G"
        fill = f"{f['chunk_fill']:.0%}/{f['container_fill']:.0%}"
        print(f"{shape:>20}{f['n_chunks']:>8}{fill:>12}{f['wall_h']:>9.3f}"
              f"{f['container_hours']:>9.3f}{f['usd']:>9.3f}"
              f"{f['usd_with_margin']:>9.3f}")
        underfilled |= f["container_fill"] < 1.0
    print()
    print("fill is chunk/container: chunk is how full the last chunk is, "
          "container the share of requested containers that get any work.")
    if underfilled:
        print(f"At --chunk {args.chunk} there are only "
              f"{math.ceil(n / args.chunk)} chunks, so container counts above "
              "that buy no wall time. Modal never starts them, so they cost "
              "nothing -- but the forecast should not imply a speedup they "
              "cannot deliver.")
    print("Modal bills the cores and memory a container reserves for the whole "
          "of its life, not the ones it keeps busy. More containers buy wall "
          "time, not cost, up to the per-container startup charged above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
