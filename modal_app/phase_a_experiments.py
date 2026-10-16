"""Phase A: bounded CPU experiments A1-A4, fanned out one unit per container.

The science lives in ``pepstab/phase_a.py`` and ``pepstab/nested.py``; this file
only arranges for it to run on many containers instead of one laptop. A work
unit is one ``(arm, config, encoding, seed, outer fold, inner fold)`` cell of
``reports/further_research/phase_a_prespecification.md``. The split is in the
units, not in the containers: every container runs the same ``run_unit``, and
the payload it is handed is row records and sequences, not feature matrices, so
the encoding happens where the fit happens.

**Four arms, not six.** A0 (reference reproduction) runs locally via
``scripts/phase_a_a0.py`` and A5 (blends) is arithmetic over the rows this
batch produces, so neither is a container unit.

**The screen is 3,360 units: 56 configurations x 2 encodings x 2 seeds x 5
outer folds x 3 inner folds.** Two of those axes look like waste and are not,
so neither should be "optimised" away later.

*The outer fold is not optional.* Hyperparameters are chosen independently
inside **each** outer fitting partition. Selecting once on outer fold 0's
fitting partition and reusing the result across all five would leak: the rows
held out in outer fold 1 sit inside outer fold 0's fitting partition, so a
selection made there has already seen them. That is what makes the 5x real
rather than redundant.

*Both encodings run* because the deployed reference ensemble spans both and
ensembling every arm identically is a standing project rule -- it is worth
about +0.074 median Spearman here, larger than most effects Phase A is looking
for, so dropping an encoding would halve the bill and change what is being
measured.

**56 configurations is not a breach of the plan's ceiling.** Sec. 3.3 of
``reports/archive/plans/further-research-2026-10-04.md`` caps "at most 12 prespecified
configurations per new learner/objective family", and the cap is per family,
not per arm. The decomposition, pinned against
``pepstab.phase_a.arm_configs`` by ``tests/test_phase_a_modal.py``:

* **A1, 12 configurations, one family** -- CatBoost RMSE regression,
  depth 3 x learning_rate 2 x l2_leaf_reg 2. Exactly at the ceiling.
* **A2, 4, one family** -- allele-centred targets, k in {1, 5, 20, 100}, with
  the MLP hyperparameters held at the A0 reference values (same-geometry).
* **A3, 16, two families** -- MSE: 4 (2 sampling x 2 stopping). MSE+pairwise
  ranking: 12 (3 lambda x 2 sampling x 2 stopping).
* **A4, 24, two declared objective families** -- YetiRank 12 and PairLogit 12,
  each over A1's 12-point grid.

No family exceeds 12, so no Sec. 7 extension entry is owed.

**CPU only, by construction.** No function in this file declares ``gpu=``, and
``tests/test_phase_a_modal.py`` asserts that by reading the decorators rather
than the prose. Two reasons, and the second is the stronger one. Phase A is
bounded CPU work -- a numpy MLP and CatBoost on 19,716 training rows -- so a GPU
would buy nothing. And per ``modal_app/boltz_common.py:3-8``, Modal validates
*every* function in an app at creation time, so a single ``gpu=`` function here
would make this app undeployable in a workspace whose GPU access is not
provisioned, taking the CPU work down with it. That is a validation liability
for no gain.

**No Volumes, no Secrets.** Nothing is cached between containers: the payload
travels with the call and the results come back as return values, so there is
no shared mutable state and a unit can be retried freely.

    # free: this path imports no modal and starts nothing
    .venv/bin/python modal_app/phase_a_experiments.py --self-check
    .venv/bin/python scripts/phase_a_forecast.py --seconds-per-unit 120 --sweep

    # 3-5 units end to end, before any batch (project invariant)
    MODAL_PROFILE=a-cheparukhin modal run modal_app/phase_a_experiments.py::smoke \
        --profile a-cheparukhin

    # forecast, from the time ::smoke measured
    MODAL_PROFILE=a-cheparukhin modal run modal_app/phase_a_experiments.py::forecast \
        --profile a-cheparukhin \
        --measured-from reports/further_research/phase_a_smoke_a-cheparukhin.json

    # the batch, only after the forecast is approved
    MODAL_PROFILE=a-cheparukhin modal run --detach \
        modal_app/phase_a_experiments.py::run --profile a-cheparukhin \
        --measured-from reports/further_research/phase_a_smoke_a-cheparukhin.json

    # and, once the screen has chosen finalists, the five-fold assessment
    MODAL_PROFILE=a-cheparukhin modal run --detach \
        modal_app/phase_a_experiments.py::run --profile a-cheparukhin \
        --stage assess --selected-from reports/further_research/phase_a_selected.json \
        --measured-from reports/further_research/phase_a_smoke_a-cheparukhin.json

``::run --dry-run`` prints the forecast and stops without a remote call. It is
still not free -- ``modal run`` builds and validates this app's image before
the entrypoint executes a line -- which is why the genuinely free forecast is
``scripts/phase_a_forecast.py``, which does not import ``modal``.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Same two profiles as every other app here; ``colleague`` points at the
#: ``sofyaleyn`` workspace. Unlike stages 4c/5/8, Phase A does **not** split
#: work between them -- there is no Volume and no disjoint half, so one profile
#: runs the whole unit set. The guard is still required, for a different
#: reason: the forecast below is written against one workspace's metered rates
#: and one workspace's reserved credit, and a run that silently lands in the
#: other spends the wrong budget.
PROFILES = ("a-cheparukhin", "colleague")

#: Arm ids this runner dispatches, as prespecified in
#: reports/further_research/phase_a_prespecification.md. A0 (reference
#: reproduction) and A5 (blends over out-of-fold predictions) are **not**
#: container units -- A0 runs locally via ``scripts/phase_a_a0.py`` and A5 is
#: arithmetic over rows this batch produces -- so they are deliberately absent.
#: Unit enumeration is the science layer's job; this is only what ``--arms``
#: accepts, so a typo is caught on the client rather than in a container.
#: ``_parse_arms`` cross-checks it against ``pepstab.phase_a.ARMS`` to catch
#: drift between the two files.
ARMS = ("A1", "A2", "A3", "A4")

# --- worker shape ----------------------------------------------------------
# PROVISIONAL until ``::smoke`` measures them, and deliberately small.
#
# cpu: a unit is one fit, and both learners in Phase A are single-model --
# pepstab.mlp is numpy/BLAS and CatBoost takes its own thread count -- so the
# shape that matters is many containers running one unit each, not one wide
# container. 2.0 cores is the smallest reservation that lets BLAS use a second
# thread without the container thrashing against its own limit.
#
# memory: stage 5's CPU-torch worker peaked at ~1.2 GiB on this repo's
# encodings (modal_app/proteinmpnn_scoring.py), and CatBoost at depth 8 over
# 19,716 x ~760 float32 columns is the new term here. 4 GiB is a guess with
# headroom, and ``::smoke`` writes the measured cgroup peak so the guess can be
# replaced rather than inherited.
WORKER_CPU = 2.0
WORKER_MEMORY = 4096  # MiB

#: BLAS and torch thread count, pinned into the image and recorded on every
#: row. This project has measured that thread count moves a single network by
#: ~0.02 median Spearman at a fixed seed (reference-blas-threads note,
#: prespecification Sec. 5), so the thread count is part of the run record, not
#: a performance knob. Pinned to the reserved core count: more threads than
#: reserved cores oversubscribes a cgroup-limited container.
WORKER_THREADS = int(WORKER_CPU)

#: Container count buys WALL TIME ONLY -- the unit-seconds are fixed by the
#: experiment, so cost is unchanged by this up to the per-container startup
#: that the forecast charges for explicitly. Not independently verified against
#: the workspace CPU quota; Modal queues rather than failing if the real cap is
#: lower, and the run reports what it got.
MAX_CONTAINERS = 40
DEFAULT_CHUNK = 8

#: Stage 5 measured ~30 s to pull a CPU-torch image and import it
#: (modal_app/proteinmpnn_scoring.py). This image adds pandas, scikit-learn and
#: CatBoost, so 45 s until ``::smoke`` measures it.
CONTAINER_STARTUP_S = 45.0

#: 1.25, the same margin stages 4c and 8 forecast with.
COST_MARGIN = 1.25

#: Metered rates for this workspace, read from the repo rather than from a
#: pricing page: an unmeasured rate assumption cost this project a 4.6x error
#: once already.
RATES_PATH = REPO / "reports" / "ectodomain_rates.json"

OUT_DIR = REPO / "reports" / "further_research"
RUNS_CSV = OUT_DIR / "phase_a_runs.csv"

#: What ``run_unit`` must return non-null for a smoke to pass. The names are
#: the ones ``pepstab.phase_a._score_block`` emits. The two coverage keys are
#: here and not only the correlation because prespecification Sec. 2 requires
#: coverage -- eligible alleles and the share of fold rows they cover -- to be
#: reported for every fold alongside every score, and a score whose coverage is
#: missing cannot be read against that rule.
REQUIRED_METRIC_KEYS = (
    "median_per_allele_spearman",
    "n_alleles_eligible",
    "eligible_row_coverage",
)


def load_rates() -> dict[str, float]:
    """Metered CPU and memory rates. Read **lazily**, never at module scope.

    Modal imports this module inside every container, where the repo does not
    exist. A module-scope read of a repo file kills the container on import
    before it executes a line of its own, which surfaces as "0 tasks" and
    nothing else -- the defect documented at
    ``modal_app/proteinmpnn_scoring.py:97-105``, which neither a local forecast
    nor ``--dry-run`` can catch, because both run where the file is present.
    """
    raw = json.loads(RATES_PATH.read_text())
    return {
        "cpu_hour": float(raw["cpu_hour_cost"]),
        "mem_gib_hour": float(raw["mem_gib_hour_cost"]),
    }


def check_profile(profile: str) -> str:
    """Refuse to run if the requested profile and the active one disagree.

    The house assertion, same form as ``modal_app/foldx_scoring.py`` and
    ``modal_app/ectodomain_production.py``. Phase A has no disjoint halves to
    protect; what this protects is the budget -- the forecast is priced at one
    workspace's metered rates against one workspace's reserved credit.
    """
    active = os.environ.get("MODAL_PROFILE")
    assert profile in PROFILES, f"--profile must be one of {PROFILES}, got {profile!r}"
    assert active in (None, profile), (
        f"--profile {profile!r} but MODAL_PROFILE={active!r}. The forecast is "
        f"priced against one workspace's metered rates and reserved credit; set "
        f"both the same."
    )
    if active is None:
        print(
            f"warning: MODAL_PROFILE is unset, so this runs against your active "
            f"profile. Billing the {profile!r} workspace -- confirm that is the "
            f"same one."
        )
    return profile


# --------------------------------------------------------------------------
# cost model
# --------------------------------------------------------------------------


def forecast_numbers(
    n_units: int,
    seconds_per_unit: float,
    cpu: float = WORKER_CPU,
    memory_gib: float = WORKER_MEMORY / 1024,
    containers: int = MAX_CONTAINERS,
    chunk: int = DEFAULT_CHUNK,
    startup_s: float = CONTAINER_STARTUP_S,
    margin: float = COST_MARGIN,
) -> dict:
    """Container-hours and dollars at the repo's metered rates.

    Copied from ``modal_app/foldx_scoring.py:145-192`` because the billing
    model is the same and should not be re-derived: Modal bills the cores and
    memory a container **reserves**, for as long as it lives, not the cores it
    keeps busy. So the cost is ``containers x cpu x wall``, and a container
    that is not full is a real expense rather than something the arithmetic
    hides. ``startup_s`` is charged explicitly because every container pays it.

    Two fill factors, both reported rather than divided away:

    ``chunk_fill`` is how full the last chunk is. ``run_chunk`` runs its units
    serially, so a short tail chunk costs only its own units -- unlike stage
    8's process pool, where a short chunk left reserved cores idle and turned a
    $205 forecast into a $386 bill. It is reported so that a chunk size chosen
    to be larger than the whole unit set is visible.

    ``container_fill`` is the share of requested containers that have any work.
    Requesting more containers than there are chunks does not cost anything --
    Modal never starts them -- but it does mean the wall time will not improve
    however high the number goes, which is worth seeing in the forecast rather
    than discovering in the run.
    """
    assert chunk >= 1, f"--chunk must be at least 1 unit, got {chunk}"
    assert containers >= 1, f"--containers must be at least 1, got {containers}"
    assert n_units >= 1, f"nothing to forecast: {n_units} units"

    rates = load_rates()
    n_chunks = math.ceil(n_units / chunk)
    chunk_fill = n_units / (n_chunks * chunk)
    active = min(containers, n_chunks)
    container_fill = active / containers
    # The bill is set by the container that lives longest, times the number of
    # containers: Modal keeps a mapped container alive until the map drains, so
    # charging the slowest container's wall to all of them is the conservative
    # reading and the one stage 8 used.
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
        "cpu_per_container": cpu,
        "memory_gib_per_container": memory_gib,
        "threads": WORKER_THREADS,
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
        "cpu_hour_rate_usd": rates["cpu_hour"],
        "mem_gib_hour_rate_usd": rates["mem_gib_hour"],
        "usd": usd,
        "usd_with_margin": usd * margin,
        "margin": margin,
        "rates_source": str(RATES_PATH.relative_to(REPO)),
    }


def print_forecast(f: dict, title: str, measured: bool = False) -> None:
    print(f"\n{title}")
    print(f"  units               {f['n_units']:,} in {f['n_chunks']:,} chunks of "
          f"{f['chunk']}")
    print(f"  s/unit              {f['seconds_per_unit']:.2f}  "
          f"({'MEASURED' if measured else 'ASSUMED -- run ::smoke'})")
    print(f"  shape               {f['containers_active']} of "
          f"{f['containers_requested']} containers x "
          f"{f['cpu_per_container']:g} cores / "
          f"{f['memory_gib_per_container']:g} GiB, "
          f"{f['threads']} BLAS threads each")
    print(f"  fill                chunk {f['chunk_fill']:.0%}, "
          f"containers {f['container_fill']:.0%} "
          f"({f['chunks_per_container']} chunks per container)")
    print(f"  wall                {f['wall_h']:.3f} h  "
          f"(incl. {f['startup_s']:.0f} s startup per container)")
    print(f"  container-hours     {f['container_hours']:.3f}  -> "
          f"{f['core_hours']:.2f} core-h + {f['mem_gib_hours']:.2f} GiB-h")
    print(f"  metered rates       ${f['cpu_hour_rate_usd']:.5f}/core-h, "
          f"${f['mem_gib_hour_rate_usd']:.5f}/GiB-h  [{f['rates_source']}]")
    print(f"  cost                ${f['usd']:.3f}   with "
          f"{f['margin']:.2f}x margin ${f['usd_with_margin']:.3f}")
    if f["container_fill"] < 1.0:
        print(f"  note                {f['containers_requested'] - f['containers_active']} "
              f"requested containers have no chunk to run; wall time will not "
              f"improve above {f['n_chunks']} containers at --chunk {f['chunk']}.")


def seconds_from_smoke(path: Path) -> float:
    """The per-unit time ``::smoke`` measured. The only non-flag source."""
    data = json.loads(Path(path).read_text())
    return float(data["seconds_per_unit"])


# --------------------------------------------------------------------------
# free self-check
# --------------------------------------------------------------------------


def self_check() -> int:
    """Everything checkable without Modal and without the science layer.

    Deliberately placed **above** ``import modal`` so that ``--self-check``
    cannot import the SDK at all. ``scripts/foldx_forecast.py:3-10`` records
    why that distinction is worth the awkward import order: ``modal run
    --dry-run`` still builds and validates the image, so "imports modal" and
    "free" are not the same claim, and this project has already described a
    ``--dry-run`` as free while it quietly started a container.
    """
    print(f"repo                {REPO}")
    print(f"rates               {load_rates()}  [{RATES_PATH.relative_to(REPO)}]")
    print(f"worker shape        {WORKER_CPU:g} cores / {WORKER_MEMORY} MiB, "
          f"{WORKER_THREADS} BLAS threads (PROVISIONAL until ::smoke)")
    print(f"arms                {', '.join(ARMS)}")
    for module in PEPSTAB_MODULES:
        path = REPO / "pepstab" / module
        print(f"  pepstab/{module:<16} exists={path.exists()}")
    missing = [m for m in PEPSTAB_MODULES if not (REPO / "pepstab" / m).exists()]
    if missing:
        print(f"\nnot yet written by the science layer: {missing}. The image "
              f"declaration is lazy, so this is only a problem at build time.")

    # The enumerated screen, not an arithmetic sketch of it: the only unit
    # count worth pricing is the one enumerate_units emits. 120 s/unit is a
    # placeholder and is labelled as such -- there is no measured per-unit time
    # until ::smoke runs.
    try:
        sys.path.insert(0, str(REPO))
        from pepstab import phase_a
    except ImportError as exc:
        print(f"\npepstab.phase_a not importable ({exc}); the unit count and "
              f"therefore the forecast are unavailable. Nothing here guesses "
              f"one: the batch size is whatever enumerate_units emits.")
        return 0
    units = phase_a.enumerate_units(list(ARMS))
    by_arm = {arm: sum(1 for u in units if u["arm"] == arm) for arm in ARMS}
    print_forecast(
        forecast_numbers(len(units), seconds_per_unit=120.0),
        f"enumerated screen: {len(units):,} units  {by_arm}",
        measured=False,
    )
    print("\nThis spends nothing and imports no modal. Next: ::smoke.")
    return 0


#: Repo modules copied into the image, mirroring the repo tree so that
#: ``import pepstab.phase_a`` resolves the same way it does locally. ``augment``
#: is here because ``pepstab/__init__.py`` imports it, so omitting it makes
#: every ``import pepstab.*`` fail on import -- the same class of failure as a
#: module-scope repo read, and just as silent.
PEPSTAB_MODULES = (
    "__init__.py",
    "augment.py",
    "data.py",
    "evaluation.py",
    "features.py",
    "mlp.py",
    "nested.py",
    "phase_a.py",
    "splits.py",
)


if __name__ == "__main__" and "--self-check" in sys.argv:
    raise SystemExit(self_check())


import modal  # noqa: E402 -- below the self-check on purpose; see self_check()

app = modal.App("pepstab-phasea-experiments")

# Exact pins, every one matching the local venv (python 3.12.12, numpy 2.5.3,
# pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, torch 2.14.1, catboost
# 1.2.10). A float would make a Phase A result unreproducible, and this project
# has already established that even the BLAS thread count moves a single
# network's score.
#
# torch comes from the CPU index: torch 2.14.1 on PyPI is the CUDA build, which
# pulls ~2.5 GB of nvidia wheels into a CPU-only image. `==2.14.1` matches
# `2.14.1+cpu` under PEP 440, which is how modal_app/proteinmpnn_scoring.py
# pins it too.
#
# catboost 1.2.10 is the version the local arms layer runs, verified executing
# against exactly the numpy 2.5.3 and pandas 3.0.6 pinned above:
# ``CatBoostRegressor`` fits and predicts, and ``CatBoostRanker`` with
# ``loss_function="YetiRank"`` and ``group_id`` fits. The image matches that
# version deliberately. Shipping a different minor version to the cloud than
# the local arms run on would make the two a different learner -- the same
# drift the pepstab modules are copied in to prevent ("the feature definitions
# live in the repo module so the local pilot run and the cloud run cannot drift
# apart"), one layer down, and invisible until two runs disagreed. It publishes
# a cp312-manylinux2014_x86_64 wheel, so nothing is built from source here.
#
# catboost also drags in matplotlib, plotly, graphviz and six. They are
# unused and add image weight; pinning them out would risk an import-time
# failure inside catboost for a saving that one CPU image pull does not
# justify.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.5.3",
        "pandas==3.0.6",
        "scipy==1.18.1",
        "scikit-learn==1.9.1",
        "torch==2.14.1",
        "catboost==1.2.10",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .env({
        # Pinned, not tuned. Recorded on every returned row: the same seed at a
        # different thread count moves a single network by ~0.02 median
        # Spearman, so a run record without the thread count cannot be
        # reproduced.
        #
        # Set in the image rather than by calling
        # ``pepstab.phase_a.pin_threads``, because that function has to run
        # before numpy is imported to take effect and importing the science
        # layer imports numpy. The image environment is already in place when
        # the interpreter starts, which is the only point early enough. The
        # list mirrors ``pepstab.phase_a.THREAD_ENV_VARS``; ``run_chunk``
        # asserts the two agree rather than trusting that they still do.
        "OMP_NUM_THREADS": str(WORKER_THREADS),
        "MKL_NUM_THREADS": str(WORKER_THREADS),
        "OPENBLAS_NUM_THREADS": str(WORKER_THREADS),
        "NUMEXPR_NUM_THREADS": str(WORKER_THREADS),
        "VECLIB_MAXIMUM_THREADS": str(WORKER_THREADS),
    })
)
for _module in PEPSTAB_MODULES:
    image = image.add_local_file(
        REPO / "pepstab" / _module, f"/root/pepstab/{_module}", copy=True
    )


def _container_memory() -> dict:
    """Peak memory this container reached. Telemetry must never fail a run.

    ``WORKER_MEMORY`` is a guess with headroom, and an OOM partway through a
    paid batch is the failure this exists to prevent. Same reader as
    ``modal_app/foldx_scoring.py``.
    """
    import resource

    out: dict = {}
    for path in ("/sys/fs/cgroup/memory.peak",
                 "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            out["cgroup_peak_gib"] = round(
                int(Path(path).read_text().strip()) / 1024**3, 3
            )
            break
        except Exception:  # noqa: BLE001 -- telemetry must never fail a run
            continue
    try:
        out["self_max_rss_gib"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3
        )
    except Exception:  # noqa: BLE001
        pass
    return out


@app.function(
    image=image,
    cpu=WORKER_CPU,
    memory=WORKER_MEMORY,
    timeout=60 * 60,
    max_containers=MAX_CONTAINERS,
    # retries=1 because a unit is idempotent: ``run_unit`` is pure, writes
    # nothing and reads nothing, so a retry recomputes rather than
    # double-counting. One retry covers a preempted container without masking a
    # unit that fails deterministically -- that one returns ok=False twice and
    # shows up in the table.
    retries=1,
)
def run_chunk(units: list[dict], payload: dict) -> list[dict]:
    """Run one chunk of units serially and return one result row each.

    The decorator's ``cpu`` / ``memory`` / ``max_containers`` are **defaults**;
    the entrypoints override them with ``Function.with_options``, per
    ``modal_app/foldx_scoring.py:532-534``, so the shape is a run parameter
    rather than something frozen into the source.

    A unit that raises becomes a recorded ``ok=False`` row. One arm's bad
    configuration must not discard the other units that shared its container
    and have already been paid for.
    """
    sys.path.insert(0, "/root")
    from pepstab import phase_a

    # The image's .env is the only thing pinning the thread count, and a
    # dropped environment variable would silently change every float32 result
    # in the batch (~0.02 median Spearman on a single network) without failing
    # anything. Recorded per row so ::smoke can refuse to pass on a mismatch.
    observed_threads = phase_a.n_threads()

    started = time.time()
    rows: list[dict] = []
    for unit in units:
        t0 = time.time()
        try:
            result = phase_a.run_unit(unit, payload)
            if not isinstance(result, dict):
                result = {"ok": False,
                          "error": f"run_unit returned {type(result).__name__}"}
        except Exception as exc:  # noqa: BLE001 -- recorded, not raised
            import traceback

            result = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2000:],
            }
        row = dict(result)
        row.setdefault("key", unit.get("key"))
        row.setdefault("arm", unit.get("arm"))
        row.setdefault("ok", True)
        # The thread count is part of the run record, not a performance note.
        row["threads"] = WORKER_THREADS
        row["threads_observed"] = observed_threads
        row["thread_pin_ok"] = observed_threads == WORKER_THREADS
        row["units_schema_version"] = getattr(phase_a, "UNITS_SCHEMA_VERSION", None)
        row["worker_cpu"] = WORKER_CPU
        row["worker_memory_mib"] = WORKER_MEMORY
        row["unit_wall_s"] = round(time.time() - t0, 3)
        rows.append(row)

    wall = time.time() - started
    memory = _container_memory()
    ok = sum(1 for r in rows if r.get("ok"))
    print(f"{len(rows)} units, {ok} ok, {len(rows) - ok} failed, {wall:.1f}s wall "
          f"({wall / max(len(rows), 1):.2f}s/unit); memory {memory}")
    for row in rows:
        row["chunk_wall_s"] = round(wall, 1)
        row.update(memory)
    return rows


# --------------------------------------------------------------------------
# entrypoints
# --------------------------------------------------------------------------


def _parse_arms(arms: str, phase_a=None) -> list[str]:
    """Validate ``--arms`` on the client, not in a container.

    When the science layer is at hand its own ``ARMS`` is the authority and
    this file's copy is checked against it: two lists of arm ids is two chances
    for them to disagree, and the disagreement would surface as a container
    raising ``unknown arm`` after the batch had already been launched.
    """
    wanted = [a.strip() for a in arms.split(",") if a.strip()] or list(ARMS)
    bad = [a for a in wanted if a not in ARMS]
    assert not bad, f"unknown arm(s) {bad}; --arms must be from {ARMS}"
    if phase_a is not None:
        assert tuple(ARMS) == tuple(phase_a.ARMS), (
            f"this runner dispatches {ARMS} but pepstab.phase_a enumerates "
            f"{tuple(phase_a.ARMS)}; the two have drifted apart"
        )
    return wanted


def _shape(cpu: float, memory_gib: float, containers: int):
    """``run_chunk`` with this run's shape applied at call time."""
    return run_chunk.with_options(
        cpu=cpu, memory=int(memory_gib * 1024), max_containers=containers
    )


def _enumerate(phase_a, wanted: list[str], stage: str, selected_from: str) -> list[dict]:
    """``enumerate_units`` for one stage of the prespecified protocol.

    Screening sweeps each arm's grid at 2 seeds x 3 inner folds; assessment
    runs the finalists at 3 seeds x 5 inner folds, which is what makes an arm's
    per-outer-fold ensemble 30 members -- the same arity as the deployed
    reference. Ensembling is worth about +0.074 median Spearman here, which is
    larger than most effects Phase A is looking for, so assessing an arm at the
    screening arity would be measuring ensemble size.

    ``--selected-from`` is required for ``--stage assess`` because the finalist
    configurations are chosen on the screen's inner folds; there is no default
    selection, and inventing one would be a selection made by this file.
    """
    if stage == "screen":
        assert not selected_from, "--selected-from applies to --stage assess only"
        return phase_a.enumerate_units(wanted)
    if stage != "assess":
        raise SystemExit(f"--stage must be 'screen' or 'assess', got {stage!r}")
    if not selected_from:
        raise SystemExit(
            "--stage assess needs --selected-from <json>: the finalist "
            "configurations come from the screen's inner folds, and this runner "
            "must not pick them."
        )
    selected = json.loads(Path(selected_from).read_text())
    return phase_a.enumerate_units(wanted, stage="assess", selected=selected)


@app.local_entrypoint()
def smoke(
    profile: str = PROFILES[0],
    n: int = 0,
    arms: str = ",".join(ARMS),
    cpu: float = WORKER_CPU,
    memory_gib: float = WORKER_MEMORY / 1024,
):
    """3-5 real units end to end, before any batch. Project invariant.

    Runs the **same** ``run_chunk`` the batch does, on the same image, so what
    it measures is the thing that will be launched. Writes the measured
    seconds-per-unit and the peak memory, which are the two numbers
    ``::forecast`` and ``WORKER_MEMORY`` are currently guessing at.

    ``--n`` defaults to one unit per arm, which is the subset
    ``pepstab.phase_a.smoke_units`` declares.
    """
    check_profile(profile)

    from pepstab import phase_a

    wanted = _parse_arms(arms, phase_a)
    # The predeclared subset -- one unit per arm, first config, first encoding,
    # seed 0, outer fold 0, inner fold 0 -- not "the first n the enumerator
    # happened to emit", which would be several cells of A1 and nothing else.
    picked = phase_a.smoke_units(wanted)
    if n:
        picked = picked[:n]
    n = len(picked)
    assert 3 <= n <= 5, (
        f"{n} units for arms {wanted}: the invariant asks for 3-5 examples"
    )

    # Priced before anything remote happens, so an unexpected number is caught
    # by reading rather than by the invoice.
    rates = load_rates()
    budget_s = 60 * 60
    worst = (cpu * rates["cpu_hour"] + memory_gib * rates["mem_gib_hour"]) * budget_s / 3600
    print(f"smoke: {n} units on 1 container x {cpu:g} cores / {memory_gib:g} GiB, "
          f"<= {budget_s // 60:.0f} min => at most ${worst:.4f}")
    for unit in picked:
        print(f"  {unit['key']}")

    payload = phase_a.load_payload()

    started = time.time()
    rows = _shape(cpu, memory_gib, 1).remote(picked, payload)
    wall = time.time() - started

    bad = [r for r in rows if not r.get("ok")]
    for row in bad:
        print(f"  FAILED {row.get('key')}: {row.get('error')}")
    nulls = [
        (r.get("key"), k)
        for r in rows
        if r.get("ok")
        for k in REQUIRED_METRIC_KEYS
        if r.get(k) is None
    ]
    for key, missing in nulls:
        print(f"  MISSING {key}: {missing} is null")
    unpinned = [r.get("key") for r in rows if not r.get("thread_pin_ok")]
    for key in unpinned:
        print(f"  UNPINNED {key}: container ran at "
              f"{[r.get('threads_observed') for r in rows if r.get('key') == key]} "
              f"threads, not {WORKER_THREADS}")

    # ``run_unit`` times itself; ``unit_wall_s`` is this runner's fallback for a
    # unit that died before it could.
    per_unit = [float(r.get("unit_seconds") or r.get("unit_wall_s") or 0.0)
                for r in rows]
    per_unit = [s for s in per_unit if s > 0]
    # Two readings of peak memory: the cgroup's, which covers the container,
    # and run_unit's own RSS, which covers the fit. The reservation has to clear
    # the larger of them.
    peak = max(
        [r.get("cgroup_peak_gib") or 0.0 for r in rows]
        + [(r.get("peak_rss_mb") or 0.0) / 1024 for r in rows]
    ) if rows else 0.0
    summary = {
        "profile": profile,
        "n": len(rows),
        "ok": len(rows) - len(bad),
        "failed": len(bad),
        "arms": wanted,
        "keys": [r.get("key") for r in rows],
        "units_schema_version": rows[0].get("units_schema_version") if rows else None,
        "payload_digest": payload.get("digest"),
        # The number ::forecast extrapolates. Max, not mean: the batch's wall
        # time is set by its slowest container, and a 5-unit sample has no
        # business claiming a tighter statistic than that.
        "seconds_per_unit": round(max(per_unit), 3) if per_unit else None,
        "seconds_per_unit_median": (
            round(sorted(per_unit)[len(per_unit) // 2], 3) if per_unit else None
        ),
        "chunk_wall_s": rows[0].get("chunk_wall_s") if rows else None,
        "client_wall_s": round(wall, 1),
        "peak_gib": peak,
        "worker_memory_mib": WORKER_MEMORY,
        "threads": WORKER_THREADS,
        "shape": {"cpu": cpu, "memory_gib": memory_gib, "containers": 1},
        "all_ok": not bad and not nulls and not unpinned,
        "rows": rows,
    }
    out = OUT_DIR / f"phase_a_smoke_{profile}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "rows"},
                            indent=2, default=str))
    print(f"wrote {out.relative_to(REPO)}")

    if not summary["all_ok"]:
        raise SystemExit("smoke failed; do not launch the batch")
    if peak and peak > 0.85 * WORKER_MEMORY / 1024:
        print(f"WARNING: peak {peak:.2f} GiB against a {WORKER_MEMORY / 1024:.1f} GiB "
              f"reservation. Raise --memory-gib before the batch.")
    print(f"smoke passed: {len(rows)}/{len(rows)} units ok, "
          f"{summary['seconds_per_unit']:.1f}s/unit measured, {peak:.2f} GiB peak.")
    print(f"Next: ::forecast --measured-from {out.relative_to(REPO)}")


@app.local_entrypoint()
def forecast(
    profile: str = PROFILES[0],
    seconds_per_unit: float = 0.0,
    measured_from: str = "",
    arms: str = ",".join(ARMS),
    n: int = 0,
    cpu: float = WORKER_CPU,
    memory_gib: float = WORKER_MEMORY / 1024,
    containers: int = MAX_CONTAINERS,
    chunk: int = DEFAULT_CHUNK,
    stage: str = "screen",
    selected_from: str = "",
):
    """Container-hours and dollars. Makes no remote call.

    ``--seconds-per-unit`` has **no default**, deliberately. For a forecast
    that cannot start anything at all, use ``scripts/phase_a_forecast.py``:
    ``modal run`` builds and validates this app's image before this function
    executes a line, so this path is cheap but not free.
    """
    check_profile(profile)
    measured = False
    if measured_from:
        seconds_per_unit = seconds_from_smoke(Path(measured_from))
        measured = True
        print(f"per-unit time read from {measured_from} (MEASURED)")
    if not seconds_per_unit:
        raise SystemExit(
            "--seconds-per-unit is required (or --measured-from a ::smoke "
            "result). There is no default, deliberately: an assumed per-unit "
            "time is how this project's last 4.6x cost error happened. Run "
            "::smoke first."
        )

    if n:
        n_units, wanted = n, _parse_arms(arms)
    else:
        from pepstab import phase_a

        wanted = _parse_arms(arms, phase_a)
        n_units = len(_enumerate(phase_a, wanted, stage, selected_from))
    f = forecast_numbers(n_units, seconds_per_unit, cpu, memory_gib, containers, chunk)
    print_forecast(f, f"Phase A {stage}, arms {wanted}", measured=measured)
    print("\nModal bills the cores and memory a container reserves for as long "
          "as it lives, not the ones it keeps busy. More containers buy wall "
          "time, not cost, up to the startup term charged above.")


@app.local_entrypoint()
def run(
    profile: str = PROFILES[0],
    arms: str = ",".join(ARMS),
    seconds_per_unit: float = 0.0,
    measured_from: str = "",
    cpu: float = WORKER_CPU,
    memory_gib: float = WORKER_MEMORY / 1024,
    containers: int = MAX_CONTAINERS,
    chunk: int = DEFAULT_CHUNK,
    stage: str = "screen",
    selected_from: str = "",
    config_filter: str = "",
    limit: int = 0,
    dry_run: bool = False,
    resume: bool = False,
):
    """The batch: every enumerated unit, one chunk per container.

    Every completed chunk is journalled to
    ``reports/further_research/phase_a_runs_<profile>.partial.jsonl`` with
    ``write`` / ``flush`` / ``os.fsync`` before anything else happens, because
    rows accumulate on the **client**: without the journal a dropped connection
    discards work that has already been billed, and ``--detach`` does not help
    -- it keeps the containers alive, but nothing is collecting their results.

    An existing journal without ``--resume`` is a hard stop, not a silent
    rescore.
    """
    check_profile(profile)
    assert chunk >= 1, f"--chunk must be at least 1 unit, got {chunk}"

    from pepstab import phase_a

    wanted = _parse_arms(arms, phase_a)
    payload = phase_a.load_payload()
    units = _enumerate(phase_a, wanted, stage, selected_from)
    if config_filter:
        # §6c runs A4 as a same-geometry ablation at A1's own selected and argmax
        # configs with only the objective changed, which is 120 units rather than
        # the 1,440 the full A4 grid emits -- and half of that grid is YetiRank,
        # withdrawn in §6a because it cannot honour the declared pair rules. A
        # substring filter on the config label is enough because the label carries
        # the objective (`...,objective=PairLogit`). Filtering here rather than in
        # pepstab.phase_a keeps the prespecified grid itself untouched.
        wants = tuple(s.strip() for s in config_filter.split(";") if s.strip())
        kept = [u for u in units
                if any(w in u["config"]["label"] for w in wants)]
        assert kept, (f"--config-filter {config_filter!r} matched none of "
                      f"{len(units)} units; labels look like "
                      f"{units[0]['config']['label']!r}")
        print(f"--config-filter kept {len(kept)} of {len(units)} units "
              f"({len(wants)} pattern(s))")
        units = kept
    if limit:
        units = units[:limit]
    assert units, f"no units enumerated for arms {wanted}"

    measured = False
    if measured_from:
        seconds_per_unit = seconds_from_smoke(Path(measured_from))
        measured = True
    if seconds_per_unit:
        f = forecast_numbers(len(units), seconds_per_unit, cpu, memory_gib,
                             containers, chunk)
        print_forecast(f, f"Phase A {stage} batch, arms {wanted}", measured=measured)
    else:
        print("no --seconds-per-unit and no --measured-from: running unforecast. "
              "::smoke writes the measured time; there is no default on purpose.")

    if dry_run:
        print(f"\ndry run: {len(units):,} units over {len(wanted)} arms, planned "
              f"from pepstab.phase_a.enumerate_units on the client. No remote "
              f"call was made and no container started.")
        print("  Not free, though: `modal run` built and validated this app's "
              "image before this entrypoint ran. For a forecast that cannot "
              "start anything, use scripts/phase_a_forecast.py, which does not "
              "import modal.")
        return

    # The stage is in the *filename*, not only in a column. Assessment reruns
    # the finalists at a larger ensemble arity, so one name for both stages
    # would let the cheap screen overwrite the expensive assessment -- the same
    # defect ``modal_app/foldx_scoring.py`` avoids by putting the repair arm in
    # its output name. ``screen`` keeps the unsuffixed name because it is the
    # stage the plan's deliverable table is built from.
    suffix = "" if stage == "screen" else f"_{stage}"
    partial = OUT_DIR / f"phase_a_runs{suffix}_{profile}.partial.jsonl"
    runs_csv = RUNS_CSV if stage == "screen" else OUT_DIR / f"phase_a_runs{suffix}.csv"
    done: dict[str, dict] = {}
    if resume and partial.exists():
        for line in partial.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row.get("key", "")] = row
        units = [u for u in units if u["key"] not in done]
        print(f"resuming: {len(done):,} units already in "
              f"{partial.relative_to(REPO)}, {len(units):,} left")
    elif partial.exists():
        raise SystemExit(
            f"{partial.relative_to(REPO)} already exists, holding "
            f"{sum(1 for ln in partial.read_text().splitlines() if ln.strip()):,} "
            "units from an interrupted run. Pass --resume to continue it, or "
            "delete the file to recompute from scratch -- silently rescoring "
            "paid-for units is not a safe default."
        )

    chunks = [units[i : i + chunk] for i in range(0, len(units), chunk)]
    fn = _shape(cpu, memory_gib, containers)
    print(f"{len(units):,} units in {len(chunks)} chunks of up to {chunk}, "
          f"{containers} containers x {cpu:g} cores x {WORKER_THREADS} threads, "
          f"CPU only")

    started = time.time()
    rows: list[dict] = list(done.values())
    if chunks:
        partial.parent.mkdir(parents=True, exist_ok=True)
        with partial.open("a") as journal:
            for part in fn.map(chunks, kwargs={"payload": payload}):
                for row in part:
                    journal.write(json.dumps(row, default=str) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                rows.extend(part)
                print(f"  {len(rows):,}/{len(units) + len(done):,} "
                      f"[{(time.time() - started) / 60:.1f} min]", flush=True)

    ok = sum(1 for r in rows if r.get("ok"))
    unpinned = sum(1 for r in rows if not r.get("thread_pin_ok"))
    for row in rows:
        row["profile"] = profile
        row["stage"] = row.get("stage") or stage
    _write_csv(rows, runs_csv)
    print(json.dumps({"profile": profile, "stage": stage, "units": len(rows),
                      "ok": ok, "failed": len(rows) - ok,
                      "unpinned_threads": unpinned,
                      "wall_min": round((time.time() - started) / 60, 1)}, indent=2))
    if unpinned:
        print(f"WARNING: {unpinned} units ran at a thread count other than "
              f"{WORKER_THREADS}. Their float32 results are not reproducible "
              f"against the rest; see the threads_observed column.")
    print(f"wrote {runs_csv.relative_to(REPO)}")
    print(f"  journal {partial.relative_to(REPO)} kept. Delete it once the CSV "
          "is safe; leaving it makes a later run refuse to start rather than "
          "silently recompute.")


def _write_csv(rows: list[dict], path: Path) -> None:
    """One column per key seen anywhere, in first-seen order.

    Arms return different metric sets -- a ranker's calibrated MAE has no
    counterpart in A0 -- so the header is the union rather than the first row's
    keys, and ``extrasaction="ignore"`` would silently drop a column that only
    a later arm emits.
    """
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
