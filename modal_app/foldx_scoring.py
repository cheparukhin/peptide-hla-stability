"""Stage 8: FoldX interaction energy over the 28,166 stage 4c folds, on CPU.

FoldX is proprietary. The binary and its embedded licence are **not** in this
repository -- ``foldx5_Linux_0/`` is gitignored -- so this app reads them from
the working copy at image-build time and records the binary's SHA-256 on every
row. Nothing here downloads FoldX or works around its licence.

**CPU only, by construction.** No function in this file declares ``gpu=``;
``tests/test_foldx.py`` asserts that by reading the source. FoldX is a
single-threaded C++ program, so the shape that matters is *many cores per
container running many FoldX processes side by side*, not a big machine running
one. Core count, memory, container count and per-container process count are
**parameters** (``--cpu``, ``--memory-gib``, ``--containers``, ``--procs``),
applied through ``Function.with_options`` at call time rather than frozen into
the decorator.

**Two profiles, two disjoint halves.** ``a-cheparukhin`` and ``colleague``
(workspace ``sofyaleyn``) each hold their own half of the cohort on their own
``pepstab-structures`` Volume. ``--profile`` must equal ``MODAL_PROFILE``;
:func:`check_profile` asserts it. Two runs, concatenated locally, with an
explicit row-count assert.

    # genuinely free -- neither imports modal, so neither can start anything
    python modal_app/foldx_scoring.py --self-check
    python scripts/foldx_forecast.py --seconds-per-structure 30 --sweep

    # 3-5 structures per workspace, the project invariant, before any batch
    MODAL_PROFILE=a-cheparukhin modal run modal_app/foldx_scoring.py::smoke \
        --profile a-cheparukhin
    MODAL_PROFILE=colleague     modal run modal_app/foldx_scoring.py::smoke \
        --profile colleague

    # the batch, only after the forecast is approved
    MODAL_PROFILE=a-cheparukhin modal run --detach \
        modal_app/foldx_scoring.py::score --profile a-cheparukhin --cpu 32

``--dry-run`` on ``score`` makes **no remote call and starts no worker**: it
plans from ``data/structural_cohort.csv``, a committed file, rather than by
listing the Volume. It is still not *zero*, and the distinction is the point --
``modal run`` builds and validates this app's image, 87 MB FoldX layer
included, before the entrypoint executes a line. The last ``--dry-run`` in this
project was described as free and quietly started a container, so the free path
here is ``scripts/foldx_forecast.py``, which does not import ``modal`` and
therefore cannot start anything.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
RUN_ID = "ectodomain-20261004"

# Must match modal_app/ectodomain_production.py and
# scripts/freeze_structural_cohort.py: `colleague` points at `sofyaleyn`.
PROFILES = ("a-cheparukhin", "colleague")

OUT_DIR = Path("/structures")
PRODUCTION_ROOT = OUT_DIR / "stage4c" / RUN_ID / "boltz2" / "production"

#: Where the gitignored FoldX distribution sits in the working copy, and where
#: it lands in the image. Not a Volume: a Volume would need a `modal volume
#: put` in each workspace and would carry no content hash, while an image layer
#: is content-addressed and uploaded once per workspace.
FOLDX_LOCAL = REPO / "foldx5_Linux_0"
FOLDX_REMOTE = "/opt/foldx"

#: Metered rates, not published list rates. This project has already caught a
#: 4.6x cost error from an unmeasured assumption, so the forecast reads the
#: same file the fold's cost was reconciled against.
RATES_PATH = REPO / "reports" / "ectodomain_rates.json"

#: Defaults, overridable per run. FoldX is single-threaded; `procs` is how many
#: of it run per container, and should not exceed `cpu`.
DEFAULT_CPU = 32.0
DEFAULT_MEMORY_GIB = 16
DEFAULT_CONTAINERS = 10
DEFAULT_CHUNK = 64

COHORT_CSV = REPO / "data" / "structural_cohort.csv"
COHORT_TOTAL = 28166

out_vol = modal.Volume.from_name("pepstab-structures", create_if_missing=False)

app = modal.App("pepstab-stage8-foldx")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.3", "biotite==1.7.1")
    # The feature definitions and the verified chain mapping live in the repo
    # modules, so the cloud run and any local check cannot drift apart.
    .add_local_file(REPO / "pepstab" / "structural_features.py",
                    "/root/pepstab/structural_features.py", copy=True)
    .add_local_file(REPO / "pepstab" / "foldx.py", "/root/pepstab/foldx.py", copy=True)
    .add_local_dir(str(FOLDX_LOCAL), FOLDX_REMOTE, copy=True)
)


def check_profile(profile: str) -> str:
    """Refuse to run if the cohort half and the active Modal profile disagree.

    Same assertion as the fold runner and the 4c.5 extractor, and required by
    the project invariant: the halves are disjoint and each Volume lives in
    exactly one workspace, so a mismatch scores the wrong half or an empty tree.
    """
    active = os.environ.get("MODAL_PROFILE")
    assert profile in PROFILES, f"--profile must be one of {PROFILES}, got {profile!r}"
    assert active in (None, profile), (
        f"--profile {profile!r} but MODAL_PROFILE={active!r}. Each profile holds "
        f"its own disjoint half of the cohort on its own Volume; set both the same."
    )
    if active is None:
        print(
            f"warning: MODAL_PROFILE is unset, so this runs against your active "
            f"profile. Scoring the {profile!r} half -- confirm that is the same "
            f"workspace."
        )
    return profile


def cohort_half(profile: str) -> list[dict]:
    """This profile's rows of the frozen cohort. Local file, no Modal call."""
    rows = [r for r in csv.DictReader(COHORT_CSV.open()) if r["profile"] == profile]
    assert rows, f"no cohort rows for profile {profile!r} in {COHORT_CSV}"
    return rows


def load_rates() -> dict[str, float]:
    raw = json.loads(RATES_PATH.read_text())
    return {
        "cpu_hour": float(raw["cpu_hour_cost"]),
        "mem_gib_hour": float(raw["mem_gib_hour_cost"]),
        "volume_gib_month": float(raw["volume_storage_gib_month_cost"]),
    }


def forecast_numbers(
    n_structures: int,
    seconds_per_structure: float,
    cpu: float,
    memory_gib: float,
    containers: int,
    procs: int,
    startup_s: float = 30.0,
    margin: float = 1.25,
) -> dict:
    """Core-hours and dollars at the repo's metered rates.

    Modal bills reserved cores and reserved memory for the container's whole
    lifetime, not per busy core -- so a container with ``cpu`` cores running
    ``procs`` FoldX processes bills ``cpu`` core-hours per wall hour whatever
    ``procs`` is. Under-filling a container is therefore a real cost, and the
    forecast makes it visible instead of quietly dividing it away.
    """
    rates = load_rates()
    # Wall time: structures split over containers, each running `procs` in
    # parallel, plus one container startup (image pull + Volume mount).
    per_container = n_structures / containers
    busy_s = per_container / procs * seconds_per_structure
    wall_s = busy_s + startup_s
    container_hours = containers * wall_s / 3600
    core_hours = container_hours * cpu
    mem_gib_hours = container_hours * memory_gib
    usd = core_hours * rates["cpu_hour"] + mem_gib_hours * rates["mem_gib_hour"]
    return {
        "n_structures": n_structures,
        "seconds_per_structure": seconds_per_structure,
        "cpu_per_container": cpu,
        "memory_gib_per_container": memory_gib,
        "containers": containers,
        "procs_per_container": procs,
        "parallel_foldx_processes": containers * procs,
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


def print_forecast(f: dict, title: str) -> None:
    print(f"\n{title}")
    print(f"  structures          {f['n_structures']:,}")
    print(f"  s/structure         {f['seconds_per_structure']:.2f}  "
          f"({'MEASURED' if f.get('measured') else 'ASSUMED -- run ::smoke'})")
    print(f"  shape               {f['containers']} containers x "
          f"{f['cpu_per_container']:g} cores / {f['memory_gib_per_container']:g} GiB, "
          f"{f['procs_per_container']} FoldX processes each "
          f"({f['parallel_foldx_processes']} in parallel)")
    print(f"  wall                {f['wall_h']:.2f} h")
    print(f"  core-hours          {f['core_hours']:.1f}  "
          f"(+ {f['mem_gib_hours']:.1f} GiB-hours of memory)")
    print(f"  metered rates       ${f['cpu_hour_rate_usd']:.5f}/core-h, "
          f"${f['mem_gib_hour_rate_usd']:.5f}/GiB-h  [{f['rates_source']}]")
    print(f"  cost                ${f['usd']:.2f}   "
          f"with {f['margin']:.2f}x margin ${f['usd_with_margin']:.2f}")


#: The six alleles ``reports/stage8_foldx.md`` section 6 flags as awkward for
#: FoldX specifically: three engineered C67S constructs and three whose alpha3
#: domain was borrowed from a close relative. The pilot forces them in rather
#: than sampling and hoping, because they are the rows most likely to behave
#: oddly and the cheapest place to find that out.
AWKWARD_SLUGS = (
    "HLA-B_14_01_C67S", "HLA-B_14_02_C67S", "HLA-B_39_06_C67S",
    "HLA-A_02_50", "HLA-A_24_19", "HLA-B_08_03",
)


def _container_memory() -> dict:
    """Peak memory this container reached, for the shape the pilot is testing.

    The smoke ran 5 FoldX processes on 12 GiB -- about 2.4 GiB each. Production
    puts 32 processes on 16 GiB, about 0.5 GiB each, and ``RepairPDB`` is the
    memory-hungry command. That 4.8x reduction was never measured, and an OOM
    six hours into a paid run is the failure this telemetry exists to prevent.
    """
    import resource

    out: dict = {}
    for path, key in (
        ("/sys/fs/cgroup/memory.peak", "cgroup_peak_gib"),
        ("/sys/fs/cgroup/memory/memory.max_usage_in_bytes", "cgroup_peak_gib"),
    ):
        try:
            out[key] = round(int(Path(path).read_text().strip()) / 1024**3, 3)
            break
        except Exception:  # noqa: BLE001 -- telemetry must never fail a run
            continue
    try:
        # ru_maxrss is the largest RSS of any single reaped child, in KiB on
        # Linux: per-FoldX-process peak, which is what the 0.5 GiB/process
        # question is actually about.
        out["max_child_rss_gib"] = round(
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2, 3
        )
    except Exception:  # noqa: BLE001
        pass
    return out


def pick_pilot_folds(folders: list[str], n: int, seed: int = 20261004) -> list[str]:
    """``n`` folds spread across alleles, deterministically, awkward ones first.

    Not "whichever structures happened to finish", and not five of one allele:
    the metric is a *per-allele* Spearman, so a pilot that cannot see across
    alleles cannot inform a decision about one. Round-robins over allele slugs
    in sorted order after forcing in :data:`AWKWARD_SLUGS`.
    """
    import random

    by_allele: dict[str, list[str]] = {}
    for folder in folders:
        by_allele.setdefault(Path(folder).parent.name, []).append(folder)
    rng = random.Random(seed)
    for group in by_allele.values():
        group.sort()
        rng.shuffle(group)

    ordered = [s for s in AWKWARD_SLUGS if s in by_allele]
    ordered += [s for s in sorted(by_allele) if s not in set(AWKWARD_SLUGS)]
    picked: list[str] = []
    depth = 0
    while len(picked) < n:
        added = False
        for slug in ordered:
            if depth < len(by_allele[slug]):
                picked.append(by_allele[slug][depth])
                added = True
                if len(picked) == n:
                    break
        if not added:
            break
        depth += 1
    return picked


# --------------------------------------------------------------------------
# remote functions
# --------------------------------------------------------------------------


@app.function(image=image, timeout=10 * 60, max_containers=1, retries=0, cpu=1.0, memory=2048)
def probe_binary() -> dict:
    """Does this binary run here at all? First execution of it, ever.

    It is a Linux x86-64 static build and the development machine is arm64
    Darwin, so nothing about it has been exercised before this call. A clean
    exit is not evidence of a correct result, so this reports what it saw
    rather than asserting success.
    """
    import subprocess

    sys.path.insert(0, "/root")
    from pepstab.foldx import locate_foldx

    found = locate_foldx(FOLDX_REMOTE)
    binary = Path(found["binary"])
    os.chmod(binary, 0o755)

    def run(cmd, cwd=None):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=cwd)
            return {"rc": p.returncode, "stdout": p.stdout[-1500:], "stderr": p.stderr[-1500:]}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}

    return {
        "found": {k: v for k, v in found.items() if k != "root"},
        "uname": run(["uname", "-m"]),
        "ldd": run(["ldd", str(binary)]),
        "version": run([str(binary), "--version"], cwd=FOLDX_REMOTE),
        "help_head": run([str(binary), "-h"], cwd=FOLDX_REMOTE),
        "molecules": sorted(p.name for p in Path(found["molecules"]).iterdir())
        if found.get("molecules")
        else [],
    }


@app.function(
    image=image,
    volumes={OUT_DIR: out_vol},
    cpu=0.5,
    memory=2048,
    timeout=30 * 60,
    max_containers=1,
    retries=0,
)
def list_folds(root: str = str(PRODUCTION_ROOT)) -> list[str]:
    """Cohort fold directories on this workspace's Volume.

    Delegates the ``_smoke`` / ``_shards`` / ``_failed`` exclusion to
    ``pepstab.structural_features``. A second implementation of that rule would
    be a second chance to score harness folds as cohort rows.
    """
    sys.path.insert(0, "/root")
    from pepstab.structural_features import EXCLUDED_DIR_NAMES, is_excluded

    out_vol.reload()
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(f"{base} does not exist on this workspace's Volume")
    folders, excluded = [], 0
    for meta in base.rglob("metadata.json"):
        if is_excluded(meta.parent, base):
            excluded += 1
        else:
            folders.append(str(meta.parent))
    folders.sort()
    print(f"{len(folders)} cohort folds; {excluded} skipped in {sorted(EXCLUDED_DIR_NAMES)}")
    return folders


@app.function(
    image=image,
    volumes={OUT_DIR: out_vol},
    cpu=DEFAULT_CPU,
    memory=DEFAULT_MEMORY_GIB * 1024,
    timeout=60 * 60,
    max_containers=DEFAULT_CONTAINERS,
    retries=1,
)
def score_chunk(
    folders: list[str], repair: bool = False, procs: int = 0, timeout_s: int = 3600
) -> list[dict]:
    """Score one chunk, ``procs`` FoldX processes in parallel on this container.

    The decorator's ``cpu`` / ``memory`` / ``max_containers`` are defaults that
    the entrypoints override with ``Function.with_options``; ``procs`` is passed
    per call because it is a property of the work, not of the container.
    """
    import multiprocessing
    import subprocess
    import tempfile

    sys.path.insert(0, "/root")
    from pepstab.foldx import locate_foldx, score_fold

    out_vol.reload()
    found = locate_foldx(FOLDX_REMOTE)
    os.chmod(found["binary"], 0o755)
    procs = procs or max(1, (os.cpu_count() or 1))

    work_root = tempfile.mkdtemp(prefix="foldx_")
    started = time.time()

    if procs == 1:
        rows = [_score_one(f, work_root, found, repair, timeout_s) for f in folders]
    else:
        with multiprocessing.get_context("fork").Pool(procs) as pool:
            rows = pool.starmap(
                _score_one,
                [(f, work_root, found, repair, timeout_s) for f in folders],
            )

    ok = sum(1 for r in rows if r.get("status") == "ok")
    wall = time.time() - started
    print(
        f"{len(rows)} structures, {ok} ok, {len(rows) - ok} failed, {wall:.1f}s wall "
        f"on {procs} processes ({wall / max(len(rows), 1):.2f}s/structure wall, "
        f"{procs * wall / max(len(rows), 1):.2f}s/structure of process time)"
    )
    memory = _container_memory()
    print(f"container memory peak: {memory}")
    for row in rows:
        row["chunk_wall_s"] = round(wall, 1)
        row["chunk_procs"] = procs
        row.update(memory)
    return rows


def _score_one(
    folder: str, work_root: str, found: dict, repair: bool, timeout_s: int = 3600
) -> dict:
    """Top-level so ``multiprocessing`` can pickle it."""
    sys.path.insert(0, "/root")
    from pepstab.foldx import score_fold

    return score_fold(folder, work_root, found, repair=repair, timeout_s=timeout_s)


# --------------------------------------------------------------------------
# entrypoints
# --------------------------------------------------------------------------


@app.local_entrypoint()
def forecast(
    seconds_per_structure: float = 0.0,
    cpu: float = DEFAULT_CPU,
    memory_gib: float = DEFAULT_MEMORY_GIB,
    containers: int = DEFAULT_CONTAINERS,
    procs: int = 0,
    n: int = COHORT_TOTAL,
    measured_from: str = "",
):
    """Core-hours and dollars at the repo's metered rates. Free: no Modal call.

    ``--seconds-per-structure`` should come from ``::smoke``. Pass
    ``--measured-from reports/stage8_foldx_smoke_<profile>.json`` to read it
    from a smoke result instead, which also labels the output MEASURED rather
    than ASSUMED.
    """
    procs = procs or int(cpu)
    measured = False
    if measured_from:
        data = json.loads(Path(measured_from).read_text())
        seconds_per_structure = float(data["seconds_per_structure_process"])
        measured = True
        print(f"per-structure time read from {measured_from} (MEASURED)")
    if not seconds_per_structure:
        raise SystemExit(
            "--seconds-per-structure is required (or --measured-from a smoke "
            "result). There is no default, deliberately: an assumed per-unit "
            "time is how this project's last 4.6x cost error happened."
        )

    for repair_label, factor in (("AnalyseComplex only", 1.0),):
        f = forecast_numbers(n, seconds_per_structure * factor, cpu, memory_gib,
                             containers, procs)
        f["measured"] = measured
        print_forecast(f, f"full cohort, {repair_label}")
        print(f"  per workspace       ${f['usd'] / 2:.2f} "
              f"(${f['usd_with_margin'] / 2:.2f} with margin), half the cohort each")
    print(
        "\nNote: Modal bills the cores a container reserves for as long as it "
        "lives, not the cores it keeps busy. Fewer, fuller containers cost less "
        "than many idle ones at the same throughput."
    )


@app.local_entrypoint()
def probe():
    """Run the binary once in a container and report what it did. ~$0.001."""
    result = probe_binary.remote()
    print(json.dumps(result, indent=2)[:6000])


@app.local_entrypoint()
def smoke(
    profile: str = PROFILES[0],
    n: int = 5,
    root: str = str(PRODUCTION_ROOT),
    cpu: float = 6.0,
    memory_gib: float = 12.0,
    compare_repair: bool = True,
):
    """3-5 real structures end to end, before any batch. Project invariant.

    Runs each structure **twice** -- once with ``AnalyseComplex`` alone and once
    with ``RepairPDB`` first -- because whether repair is needed is the open
    question, it is the step that dominates runtime, and the answer has to be
    measured on these structures rather than assumed from what crystal
    structures usually need.
    """
    assert 3 <= n <= 5, "the invariant asks for 3-5 examples"
    check_profile(profile)

    rates = load_rates()
    budget_s = 15 * 60
    worst = (cpu * rates["cpu_hour"] + memory_gib * rates["mem_gib_hour"]) * budget_s / 3600
    print(f"smoke: {n} structures, {cpu:g} cores / {memory_gib:g} GiB, "
          f"<= {budget_s // 60:.0f} min => at most ${worst * (2 if compare_repair else 1):.3f}")

    print("\nprobing the binary (first execution of this build anywhere)...")
    info = probe_binary.remote()
    print(json.dumps(info, indent=2)[:4000])

    folders = list_folds.remote(root)[:n]
    if len(folders) < n:
        raise SystemExit(f"only {len(folders)} folds under {root}")
    for f in folders:
        print(f"  {Path(f).name}")

    # The five run side by side on their own cores, so wall time is one
    # structure's, not five -- and RepairPDB, if it is slow, cannot run the
    # container into its timeout five times over.
    fn = score_chunk.with_options(
        cpu=cpu, memory=int(memory_gib * 1024), max_containers=1, timeout=45 * 60
    )
    results: dict[str, list[dict]] = {}
    arms = [("analyse_only", False, 1800)]
    if compare_repair:
        # RepairPDB is the expensive command and has never run here. A shorter
        # per-fold cap means a slow repair returns a recorded timeout instead
        # of burning the container's whole budget; the repair arm is
        # informational, so a timeout is an answer, not a smoke failure.
        arms.append(("repair_then_analyse", True, 1500))
    for label, repair, per_fold in arms:
        print(f"\n--- {label} ---")
        rows = fn.remote(folders, repair=repair, procs=n, timeout_s=per_fold)
        results[label] = rows
        for r in rows:
            mark = "ok" if r.get("status") == "ok" else f"FAILED {r.get('status')}"
            print(f"  {r['complex_id']:30} {r.get('total_s', 0):8.2f}s  "
                  f"dG={r.get('foldx_interaction_energy')}  {mark}")

    summary = _summarise_smoke(profile, folders, results)
    summary["probe"] = info
    out = REPO / "reports" / f"stage8_foldx_smoke_{profile}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print("\n" + json.dumps(
        {k: v for k, v in summary.items() if k not in {"rows", "probe"}}, indent=2))
    print(f"wrote {out.relative_to(REPO)}")
    if summary["all_ok"]:
        print("\nsmoke passed. Next: scripts/foldx_forecast.py --measured-from "
              f"{out.relative_to(REPO)}, then wait for approval before ::score.")
    else:
        raise SystemExit("smoke failed; do not launch the batch")


def _summarise_smoke(profile: str, folders: list[str], results: dict[str, list[dict]]) -> dict:
    import statistics

    summary: dict = {"profile": profile, "n": len(folders),
                     "folds": [Path(f).name for f in folders], "arms": {}, "rows": results}
    all_ok = "analyse_only" in results
    for label, rows in results.items():
        ok = [r for r in rows if r.get("status") == "ok"]
        # Only the arm production would actually run gates the smoke. The
        # repair arm is a measurement of whether repair is needed; it failing
        # or timing out is part of that answer, not a reason to block.
        if label == "analyse_only":
            all_ok = all_ok and len(ok) == len(rows)
        energies = [r["foldx_interaction_energy"] for r in ok
                    if isinstance(r.get("foldx_interaction_energy"), float)]
        summary["arms"][label] = {
            "ok": len(ok),
            "failed": len(rows) - len(ok),
            "median_total_s": statistics.median([r["total_s"] for r in rows]) if rows else None,
            "median_convert_s": statistics.median([r.get("convert_s", 0) for r in ok]) if ok else None,
            "median_repair_s": statistics.median([r.get("repair_s", 0) for r in ok]) if ok else None,
            "median_analyse_s": statistics.median([r.get("analyse_s", 0) for r in ok]) if ok else None,
            "interaction_energy": energies,
        }
    summary["all_ok"] = all_ok
    base = summary["arms"].get("analyse_only", {})
    summary["seconds_per_structure_process"] = base.get("median_total_s")
    rep = summary["arms"].get("repair_then_analyse")
    if rep and base.get("median_total_s"):
        summary["repair_runtime_multiple"] = (
            rep["median_total_s"] / base["median_total_s"] if rep["median_total_s"] else None
        )
        a, b = base.get("interaction_energy") or [], rep.get("interaction_energy") or []
        if len(a) == len(b) and len(a) >= 3:
            summary["repair_energy_shift"] = [round(y - x, 4) for x, y in zip(a, b)]
            # Does repair change the *ranking*? That, not the absolute shift,
            # is what matters for a rank-correlation metric.
            summary["repair_preserves_order"] = (
                [i for i, _ in sorted(enumerate(a), key=lambda t: t[1])]
                == [i for i, _ in sorted(enumerate(b), key=lambda t: t[1])]
            )
    return summary


@app.local_entrypoint()
def pilot(
    profile: str = PROFILES[0],
    n: int = 48,
    root: str = str(PRODUCTION_ROOT),
    cpu: float = DEFAULT_CPU,
    memory_gib: float = DEFAULT_MEMORY_GIB,
    procs: int = 0,
    compare_repair: bool = True,
):
    """Widen the repair decision, at the **production container shape**.

    ``::smoke`` answered "does the binary run" on 5 structures of one allele at
    5 processes on 6 cores. Three things it could not answer, and this does,
    for about a quarter of a dollar against the batch's hundred:

    1. **Per-process memory at production concurrency.** 32 processes on
       16 GiB is 0.5 GiB each against the smoke's 2.4 GiB. Unmeasured, and
       ``RepairPDB`` is the hungry command.
    2. **Throughput at production concurrency.** The smoke had a spare core per
       process; production has none, so the measured seconds-per-structure may
       not survive the shape it is being extrapolated to.
    3. **Whether repair reorders across alleles.** The metric is a per-allele
       Spearman and the smoke saw one allele, so its reordering result cannot
       speak to the metric that decides this arm.

    Deliberately not a scored result: it writes no feature table and selects
    its folds by :func:`pick_pilot_folds`, declared in advance, rather than by
    what came back first.
    """
    check_profile(profile)
    procs = procs or int(cpu)
    assert procs <= cpu * 2, f"{procs} processes on {cpu:g} reserved cores would thrash"

    rates = load_rates()
    per_hour = cpu * rates["cpu_hour"] + memory_gib * rates["mem_gib_hour"]
    budget_h = 1.5 if compare_repair else 0.25
    print(f"pilot: {n} structures, {cpu:g} cores / {memory_gib:g} GiB, {procs} "
          f"processes; at most {budget_h:g} h => ${per_hour * budget_h:.3f}")

    folders = pick_pilot_folds(list_folds.remote(root), n)
    if len(folders) < n:
        raise SystemExit(f"only {len(folders)} folds available under {root}")
    alleles = sorted({Path(f).parent.name for f in folders})
    print(f"{len(folders)} folds over {len(alleles)} alleles; awkward ones present: "
          f"{[a for a in AWKWARD_SLUGS if a in alleles]}")

    fn = score_chunk.with_options(
        cpu=cpu, memory=int(memory_gib * 1024), max_containers=1, timeout=90 * 60
    )
    arms = [("analyse_only", False, 900)]
    if compare_repair:
        arms.append(("repair_then_analyse", True, 1800))

    results: dict[str, list[dict]] = {}
    for label, repair, per_fold in arms:
        print(f"\n--- {label} ---", flush=True)
        results[label] = fn.remote(folders, repair=repair, procs=procs, timeout_s=per_fold)

    summary = _summarise_pilot(profile, folders, results, cpu, memory_gib, procs)
    out = REPO / "reports" / f"stage8_foldx_pilot_{profile}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print(f"wrote {out.relative_to(REPO)}")


def _summarise_pilot(profile, folders, results, cpu, memory_gib, procs) -> dict:
    import statistics

    summary: dict = {
        "profile": profile, "n": len(folders),
        "shape": {"cpu": cpu, "memory_gib": memory_gib, "procs": procs,
                  "gib_per_proc": round(memory_gib / procs, 3)},
        "n_alleles": len({Path(f).parent.name for f in folders}),
        "arms": {}, "rows": results,
    }
    energies: dict[str, dict[str, float]] = {}
    for label, rows in results.items():
        ok = [r for r in rows if r.get("status") == "ok"]
        # Process time per structure at this concurrency -- the number the
        # forecast actually extrapolates, measured in the shape it extrapolates
        # to rather than in the smoke's roomier one.
        times = [r["total_s"] for r in ok]
        wall = max((r.get("chunk_wall_s") or 0) for r in rows) if rows else 0
        summary["arms"][label] = {
            "ok": len(ok), "failed": len(rows) - len(ok),
            "median_total_s": round(statistics.median(times), 3) if times else None,
            "p90_total_s": round(sorted(times)[int(0.9 * (len(times) - 1))], 3) if times else None,
            "max_total_s": round(max(times), 3) if times else None,
            "median_repair_s": round(statistics.median([r.get("repair_s", 0) for r in ok]), 3) if ok else None,
            "chunk_wall_s": wall,
            # Wall time per structure at full concurrency is what sets the bill;
            # process time divided by procs would assume perfect scaling.
            "effective_s_per_structure": round(wall / len(rows), 3) if rows else None,
            "cgroup_peak_gib": max((r.get("cgroup_peak_gib") or 0) for r in rows) if rows else None,
            "max_child_rss_gib": max((r.get("max_child_rss_gib") or 0) for r in rows) if rows else None,
            "statuses": sorted({r["status"][:70] for r in rows if r.get("status") != "ok"}),
        }
        energies[label] = {
            r["complex_id"]: r["foldx_interaction_energy"] for r in ok
            if isinstance(r.get("foldx_interaction_energy"), float)
        }

    base, rep = energies.get("analyse_only", {}), energies.get("repair_then_analyse", {})
    both = sorted(set(base) & set(rep))
    if len(both) >= 8:
        a = [base[k] for k in both]
        b = [rep[k] for k in both]
        summary["repair_vs_raw"] = {
            "n_common": len(both),
            "spearman": round(_spearman(a, b), 4),
            "pearson": round(_pearson(a, b), 4),
            "preserves_order": [i for i, _ in sorted(enumerate(a), key=lambda t: t[1])]
            == [i for i, _ in sorted(enumerate(b), key=lambda t: t[1])],
            "raw_range": [round(min(a), 3), round(max(a), 3)],
            "repaired_range": [round(min(b), 3), round(max(b), 3)],
            "shift_range": [round(min(y - x for x, y in zip(a, b)), 3),
                            round(max(y - x for x, y in zip(a, b)), 3)],
        }
        # Does the raw energy rank structures by strain inside the predicted
        # HLA model rather than by the interface? That is the mechanism the
        # smoke suggested on 5 structures and the reason repair matters here.
        strain = {r["complex_id"]: r.get("foldx_intraclashesgroup2")
                  for r in results["analyse_only"] if r.get("status") == "ok"}
        if all(isinstance(strain.get(k), float) for k in both):
            summary["repair_vs_raw"]["spearman_raw_dG_vs_intra_hla_clash"] = round(
                _spearman(a, [strain[k] for k in both]), 4)
    if rep:
        summary["repaired_energy_by_allele"] = {
            slug: round(statistics.median(v), 3)
            for slug, v in sorted(_group_by_allele(results, rep).items())
        }
    return summary


def _group_by_allele(results, energies) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for row in results.get("repair_then_analyse", []):
        cid = row.get("complex_id")
        if cid in energies:
            out.setdefault(Path(row["fold_dir"]).parent.name, []).append(energies[cid])
    return out


def _ranks(values: list[float]) -> list[float]:
    """Average ranks, so ties do not shift a correlation."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
    return num / den if den else float("nan")


def _spearman(x: list[float], y: list[float]) -> float:
    return _pearson(_ranks(x), _ranks(y))


@app.function(
    image=image,
    volumes={OUT_DIR: out_vol},
    cpu=2.0,
    memory=4096,
    timeout=20 * 60,
    max_containers=1,
    retries=0,
)
def raw_output(folder: str, repair: bool = False) -> dict:
    """Return the raw ``.fxout`` text for one structure, unparsed.

    Section 7 of ``reports/stage8_foldx.md`` requires the real header to be
    compared against what the parser extracted, once, by eye. A parser that
    agrees with itself is not evidence.
    """
    import subprocess
    import tempfile

    sys.path.insert(0, "/root")
    from pepstab.foldx import (
        analyse_complex_command,
        locate_foldx,
        parse_fxout,
        prepare_pdb,
    )

    out_vol.reload()
    found = locate_foldx(FOLDX_REMOTE)
    os.chmod(found["binary"], 0o755)
    work = Path(tempfile.mkdtemp(prefix="foldx_raw_"))
    (work / "in").mkdir()
    (work / "out").mkdir()
    Path(work / "molecules").symlink_to(found["molecules"])

    prepared = prepare_pdb(Path(folder), work / "in")
    name = Path(prepared["pdb"]).name
    groups = (prepared["peptide_chain"], prepared["hla_chain"])
    proc = subprocess.run(
        analyse_complex_command(found["binary"], name, str(work / "in"),
                                str(work / "out"), groups),
        capture_output=True, text=True, timeout=900, cwd=str(work),
    )
    files = {p.name: p.read_text()[:8000] for p in sorted(Path(work / "out").iterdir())}
    interaction = next((v for k, v in files.items() if k.startswith("Interaction_")), "")
    return {
        "complex_id": Path(folder).name,
        "groups": list(groups),
        "returncode": proc.returncode,
        "output_files": sorted(files),
        "interaction_fxout": interaction,
        "parsed": parse_fxout(interaction) if interaction else None,
        "pdb_head": Path(prepared["pdb"]).read_text()[:400],
    }


@app.local_entrypoint()
def inspect_output(profile: str = PROFILES[0], root: str = str(PRODUCTION_ROOT),
                   repair: bool = False):
    """Dump one raw .fxout and the parse beside it. ~$0.002."""
    check_profile(profile)
    folder = list_folds.remote(root)[0]
    result = raw_output.remote(folder, repair=repair)
    out = REPO / "reports" / f"stage8_foldx_raw_output_{profile}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(result["interaction_fxout"][:4000])
    print("\nparsed columns:")
    for k, v in (result["parsed"] or [{}])[0].items():
        print(f"  {k:40} {v}")
    print(f"\nwrote {out.relative_to(REPO)}")


@app.local_entrypoint()
def score(
    profile: str = PROFILES[0],
    root: str = str(PRODUCTION_ROOT),
    repair: bool = False,
    cpu: float = DEFAULT_CPU,
    memory_gib: float = DEFAULT_MEMORY_GIB,
    containers: int = DEFAULT_CONTAINERS,
    procs: int = 0,
    chunk: int = DEFAULT_CHUNK,
    limit: int = 0,
    seconds_per_structure: float = 0.0,
    dry_run: bool = False,
):
    """Score this profile's half of the cohort.

    ``--dry-run`` makes **no Modal call of any kind** -- it plans from the
    committed ``data/structural_cohort.csv``, prints the forecast, and returns.
    It cannot start a container, and ``tests/test_foldx.py`` asserts that by
    checking the code path.
    """
    check_profile(profile)
    procs = procs or int(cpu)
    assert procs <= cpu * 2, (
        f"{procs} FoldX processes on {cpu:g} reserved cores would thrash; FoldX "
        "is single-threaded, so procs should be <= cpu."
    )

    rows = cohort_half(profile)
    n = limit or len(rows)
    if seconds_per_structure:
        f = forecast_numbers(n, seconds_per_structure, cpu, memory_gib, containers, procs)
        f["measured"] = False
        print_forecast(f, f"{profile}: {n:,} structures")

    if dry_run:
        print(f"\ndry run: planned from {COHORT_CSV.relative_to(REPO)} "
              f"({len(rows):,} rows for {profile!r}). No remote call was made and "
              f"no worker container started.")
        print("  Not zero, though, and the difference matters: `modal run` builds "
              "and validates this app's image -- including the 87 MB FoldX layer "
              "-- before the entrypoint runs at all. For a forecast that cannot "
              "start anything, use `scripts/foldx_forecast.py`, which does not "
              "import modal.")
        return

    folders = list_folds.remote(root)
    if limit:
        folders = folders[:limit]
    assert folders, f"no folds under {root}"
    if not limit and len(folders) != len(rows):
        print(f"warning: {len(folders)} folds on the Volume against {len(rows)} "
              f"cohort rows for {profile!r}")

    chunks = [folders[i : i + chunk] for i in range(0, len(folders), chunk)]
    fn = score_chunk.with_options(
        cpu=cpu, memory=int(memory_gib * 1024), max_containers=containers
    )
    print(f"{len(folders):,} structures in {len(chunks)} chunks of {chunk}, "
          f"{containers} containers x {cpu:g} cores x {procs} processes, CPU only")

    started = time.time()
    out_rows: list[dict] = []
    for part in fn.starmap([(c, repair, procs) for c in chunks]):
        out_rows.extend(part)
        print(f"  {len(out_rows):,}/{len(folders):,} "
              f"[{(time.time() - started) / 3600:.2f} h]", flush=True)

    ok = sum(1 for r in out_rows if r.get("status") == "ok")
    for r in out_rows:
        r["profile"] = profile
        r["cohort_coverage"] = "half"
    local = REPO / "reports" / f"stage8_foldx_scores_{profile}.csv"
    _write_csv(out_rows, local)
    print(json.dumps({"profile": profile, "rows": len(out_rows), "ok": ok,
                      "failed": len(out_rows) - ok,
                      "wall_h": round((time.time() - started) / 3600, 2)}, indent=2))
    print(f"wrote {local.relative_to(REPO)} -- ONE HALF of the cohort. Concatenate "
          f"both halves with scripts/foldx_concat.py, which asserts {COHORT_TOTAL:,}.")


def _write_csv(rows: list[dict], path: Path) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# free self-check: no modal import side effects, no network
# --------------------------------------------------------------------------

if __name__ == "__main__":
    if "--self-check" in sys.argv:
        print(f"FoldX distribution: {FOLDX_LOCAL}  exists={FOLDX_LOCAL.exists()}")
        if FOLDX_LOCAL.exists():
            binary = sorted(FOLDX_LOCAL.glob("foldx*"))
            for b in binary:
                digest = hashlib.sha256(b.read_bytes()).hexdigest()
                print(f"  {b.name}  {b.stat().st_size / 1e6:.1f} MB  sha256={digest}")
        for p in PROFILES:
            print(f"  cohort half {p:16} {len(cohort_half(p)):,} pairs")
        print(f"  rates: {load_rates()}")
