"""Phase B on Modal: three censored arms, one container per (arm, outer fold).

    python modal_app/phase_b_calibration.py --self-check
    modal run modal_app/phase_b_calibration.py::forecast --profile a-cheparukhin
    modal run modal_app/phase_b_calibration.py::smoke    --profile a-cheparukhin
    modal run modal_app/phase_b_calibration.py::run      --profile a-cheparukhin

The science lives in ``pepstab/phase_b.py``; this file only arranges for it to
run on 15 containers instead of one contended laptop. A work unit is one
**(arm, outer fold)** cell of
``reports/further_research/phase_b_declaration.md`` §3: three arms x five outer
folds, each fitting ``5 inner x 2 encodings x 3 seeds = 30`` members, so 450
networks in 15 units.

**Why the cloud helps here beyond wall time.** Locally, four concurrent jobs made
BLAS thread pinning load-bearing for reproducibility -- this project has recorded
that the same seed at a different thread count moves a single network by about
0.02 median Spearman, and under contention that becomes nondeterministic rather
than a constant offset, which is the worst case for a *paired* comparison. One
container per unit with threads pinned in the image removes that class of problem
rather than mitigating it.

House conventions this file follows, each for a reason the comments give:
``check_profile()`` first in every entrypoint; **no repo file read at module
scope**; ``pepstab`` modules shipped with ``add_local_file(..., copy=True)``
mirroring the repo tree; CSVs read client-side and passed as plain records;
results returned as dicts and written to ``reports/`` client-side; BLAS threads
set in the image ``.env``, which is the only point early enough to bind.

Nothing about the protocol changes by moving: the same frozen folds, the same
arity, the same ``>=20`` panel with a ``>=50`` sensitivity, the same
``pepstab.evaluation`` paths, the same declared arms and decision rules. Only
where the arithmetic happens.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

PROFILES = ("a-cheparukhin",)
#: One core per container. A unit is a sequence of single-threaded network fits,
#: so a second core would be reserved and billed while idle -- Modal bills
#: reserved cores for the container's whole lifetime, not busy ones.
WORKER_CPU = 1.0
WORKER_THREADS = 1
#: 4 GiB. The two feature matrices for a 15,773-row fitting partition are about
#: 54 MB each at float32 (43 positions x 20 amino acids), so this is mostly
#: headroom: an OOM partway through a paid batch is the failure the headroom
#: exists to prevent, and 4 GiB costs $0.008/GiB-hour.
WORKER_MEMORY = 4 * 1024
#: 15 units, so 15 containers runs the whole batch in one wave.
MAX_CONTAINERS = 15

# The full intra-package import closure of ``pepstab.phase_b``, derived rather
# than guessed. ``augment.py`` is in here because ``pepstab/__init__.py`` imports
# it eagerly, so omitting it makes *any* ``from pepstab import ...`` fail inside a
# container with a misleading "circular import" error -- which is exactly what the
# first smoke run hit. Nothing in phase_b uses augment directly.
PEPSTAB_MODULES = (
    "__init__.py",
    "augment.py",
    "calibration.py",
    "censored.py",
    "data.py",
    "evaluation.py",
    "features.py",
    "mlp.py",
    "nested.py",
    "phase_b.py",
    "splits.py",
)

RATES_PATH = REPO / "reports" / "ectodomain_rates.json"
FOLDS_CSV = REPO / "reports" / "further_research" / "phase_a_fold_assignments.csv"
OUT_DIR = REPO / "reports" / "further_research"


# --------------------------------------------------------------------------
# things that must work without modal installed


def self_check() -> int:
    """Check the local side without importing modal or touching the network.

    Deliberately runnable with no Modal credentials: it catches a missing
    ``pepstab`` module or an unreadable rates file, which are the two failures
    that would otherwise surface as a container crash at import time with no
    useful error.
    """
    print(f"repo        : {REPO}")
    ok = True
    for module in PEPSTAB_MODULES:
        path = REPO / "pepstab" / module
        print(f"  pepstab/{module:<16} exists={path.exists()}")
        ok &= path.exists()
    for path in (RATES_PATH, FOLDS_CSV):
        print(f"  {path.relative_to(REPO)}  exists={path.exists()}")
        ok &= path.exists()
    try:
        sys.path.insert(0, str(REPO))
        from pepstab import phase_b

        print(f"  pepstab.phase_b arms: {[a[0] for a in phase_b.ARMS]}")
        assert set(phase_b.THREAD_ENV_VARS) == set(IMAGE_THREAD_VARS), (
            "pepstab.phase_b.THREAD_ENV_VARS and the image's .env have drifted; "
            "a run record's thread count would then be wrong")
        print("  thread variables agree between the image and the science layer")
    except Exception as exc:  # noqa: BLE001
        print(f"  pepstab.phase_b import FAILED: {exc}")
        ok = False
    print("\nself-check", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


#: The variables the image sets. Kept as a module constant so ``self_check``
#: can assert it matches ``pepstab.phase_b.THREAD_ENV_VARS`` rather than the two
#: silently drifting apart.
IMAGE_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def read_rates() -> dict:
    """Metered rates from the repo, never a pricing page.

    Read inside a function, not at module scope: a module-scope read kills every
    container on import with an unhelpful traceback, and neither ``--self-check``
    nor a dry run would catch it because both run where the file is present.
    """
    raw = json.loads(RATES_PATH.read_text())
    return {"cpu_hour": float(raw["cpu_hour_cost"]),
            "mem_gib_hour": float(raw["mem_gib_hour_cost"])}


def check_profile(profile: str) -> str:
    """Refuse to run if the requested profile and the active one disagree.

    The house assertion. What it protects is the budget: the forecast is priced
    at one workspace's metered rates against one workspace's reserved credit.
    """
    active = os.environ.get("MODAL_PROFILE")
    assert profile in PROFILES, f"--profile must be one of {PROFILES}, got {profile!r}"
    assert active in (None, profile), (
        f"--profile {profile!r} but MODAL_PROFILE={active!r}. The forecast is "
        f"priced against one workspace's metered rates and reserved credit; set "
        f"both the same.")
    if active is None:
        print(f"warning: MODAL_PROFILE is unset, so this runs against your active "
              f"profile. Billing the {profile!r} workspace -- confirm that is the "
              f"same one.")
    return profile


def price(n_units: int, seconds_per_unit: float, *, cpu: float = WORKER_CPU,
          memory_gib: float = WORKER_MEMORY / 1024,
          containers: int = MAX_CONTAINERS,
          startup_seconds: float = 30.0) -> dict:
    """Cost of a batch, surfacing reserved-resource waste rather than dividing it away.

    Modal bills **reserved** cores and memory for the container's whole lifetime,
    so container startup and any under-filled tail are real money. They are
    itemised here for the same reason ``modal_app/foldx_scoring.py`` itemises
    them: a forecast that silently amortises startup across units understates a
    short-unit batch, and these units are short.
    """
    rates = read_rates()
    waves = -(-n_units // max(1, containers))        # ceiling division
    busy_hours = n_units * seconds_per_unit / 3600.0
    startup_hours = n_units * startup_seconds / 3600.0
    reserved_hours = busy_hours + startup_hours
    cpu_usd = reserved_hours * cpu * rates["cpu_hour"]
    mem_usd = reserved_hours * memory_gib * rates["mem_gib_hour"]
    return {
        "n_units": n_units, "seconds_per_unit": seconds_per_unit,
        "containers": containers, "waves": waves,
        "wall_clock_minutes": round(waves * (seconds_per_unit + startup_seconds) / 60, 1),
        "reserved_core_hours": round(reserved_hours * cpu, 4),
        "reserved_gib_hours": round(reserved_hours * memory_gib, 3),
        "startup_share_of_reserved": round(startup_hours / reserved_hours, 3)
        if reserved_hours else 0.0,
        "cpu_usd": round(cpu_usd, 4), "mem_usd": round(mem_usd, 4),
        "total_usd": round(cpu_usd + mem_usd, 4),
        "rates": rates,
    }


if __name__ == "__main__" and "--self-check" in sys.argv:
    raise SystemExit(self_check())


import modal  # noqa: E402 -- below the self-check on purpose; see self_check()

app = modal.App("pepstab-phaseb-calibration")

# Exact pins matching the local venv, so a Phase B member fitted in the cloud is
# the same network as one fitted locally. torch is NOT needed -- the MLP is
# hand-rolled numpy -- so it is left out, which keeps this image small.
# scikit-learn is in because the post-hoc calibrators use it.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy==2.5.3",
        "pandas==3.0.6",
        "scipy==1.18.1",
        "scikit-learn==1.9.1",
    )
    .env({
        # Pinned, not tuned, and set here rather than by calling
        # ``pepstab.phase_b.pin_threads`` because that has to run before numpy is
        # imported and importing the science layer imports numpy. The image
        # environment is already in place when the interpreter starts, which is
        # the only point early enough.
        var: str(WORKER_THREADS) for var in IMAGE_THREAD_VARS
    })
)
for _module in PEPSTAB_MODULES:
    image = image.add_local_file(
        REPO / "pepstab" / _module, f"/root/pepstab/{_module}", copy=True)

# The two read-only data files, baked in once rather than shipped as a payload
# every run. The first version of this file passed ~15,800 row records per unit,
# which is 2.08 MB serialised per unit and 31 MB for the batch; the container now
# gets pair_id lists (~160 KB) and loads the canonical CSVs itself.
#
# They land at /root/data/ because ``pepstab/data.py`` resolves
# ``REPO_ROOT = Path(__file__).resolve().parent.parent``, which is ``/root`` when
# the package sits at /root/pepstab -- so the container's notion of the repo root
# is /root and the data directory beside it is /root/data. ``run_unit`` verifies
# the raw file's SHA-256 against the client's, so a container can never quietly
# fit on different data -- the same guarantee ``data/SHA256SUMS`` gives locally.
for _name in ("rasmussen_et_al_dataset.csv", "splits.csv"):
    image = image.add_local_file(
        REPO / "data" / _name, f"/root/data/{_name}", copy=True)


def _container_memory() -> dict:
    """Peak memory this container reached. Telemetry must never fail a run."""
    import resource

    out: dict = {}
    for path in ("/sys/fs/cgroup/memory.peak",
                 "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            out["cgroup_peak_gib"] = round(int(Path(path).read_text().strip())
                                           / 1024**3, 3)
            break
        except Exception:  # noqa: BLE001 -- telemetry must never fail a run
            continue
    try:
        out["self_max_rss_gib"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3)
    except Exception:  # noqa: BLE001
        pass
    return out


@app.function(
    image=image,
    cpu=WORKER_CPU,
    memory=WORKER_MEMORY,
    timeout=60 * 60,
    max_containers=MAX_CONTAINERS,
    # retries=1 because a unit is pure: given the same payload it recomputes the
    # same members from fixed seeds and writes nothing, so a retry cannot
    # double-count or corrupt anything.
    retries=1,
)
def run_unit(unit: dict) -> dict:
    """Fit one (arm, outer fold) unit. Pure: no repo reads, no writes."""
    import hashlib
    import time

    sys.path.insert(0, "/root")
    from pepstab import phase_b
    from pepstab.data import RAW_CSV, load_with_splits

    # The dataset is read-only and the whole study rests on it being the same
    # bytes everywhere. Checked here rather than trusted, because a stale image
    # layer would otherwise produce plausible numbers from the wrong data.
    digest = hashlib.sha256(RAW_CSV.read_bytes()).hexdigest()
    if digest != unit["raw_sha256"]:
        raise AssertionError(
            f"dataset digest mismatch: container {digest[:16]}... but client "
            f"{unit['raw_sha256'][:16]}...; the image's data layer is stale")

    df = load_with_splits().set_index("pair_id", drop=False)
    cols = list(phase_b.UNIT_COLUMNS)
    fit_df = df.loc[unit["fit_pair_id"], cols].reset_index(drop=True)
    held_df = df.loc[unit["held_pair_id"], cols].reset_index(drop=True)

    started = time.perf_counter()
    got = phase_b.fit_arm_unit(
        fit_df.to_dict("records"), held_df.to_dict("records"), unit["inner"],
        kind=unit["kind"], stop_on=unit["stop_on"],
        encodings=tuple(unit["encodings"]), seeds=tuple(unit["seeds"]),
        selected={k: (tuple(v[0]), v[1]) for k, v in unit["selected"].items()},
        input_set=unit["input_set"], floor_hours=unit["floor_hours"])
    got.update({
        "arm": unit["arm"], "outer_fold": unit["outer_fold"],
        "unit_seconds": round(time.perf_counter() - started, 2),
        "threads_env": {v: os.environ.get(v) for v in IMAGE_THREAD_VARS},
        "memory": _container_memory(),
    })
    return got


@app.function(
    image=image,
    cpu=WORKER_CPU,
    memory=WORKER_MEMORY,
    timeout=90 * 60,
    max_containers=MAX_CONTAINERS,
    retries=1,
)
def run_two_part_unit(unit: dict) -> dict:
    """One outer fold of the two-part floor model: both components, 60 networks.

    Twice the networks of a ``run_unit`` because each component carries the same
    inner/encoding/seed budget as a censored arm, so the two halves are each
    comparable to the reference rather than splitting one budget between them.
    """
    import hashlib
    import time

    sys.path.insert(0, "/root")
    from pepstab import phase_b
    from pepstab.data import RAW_CSV, load_with_splits

    digest = hashlib.sha256(RAW_CSV.read_bytes()).hexdigest()
    if digest != unit["raw_sha256"]:
        raise AssertionError(
            f"dataset digest mismatch: container {digest[:16]}... but client "
            f"{unit['raw_sha256'][:16]}...; the image's data layer is stale")

    df = load_with_splits().set_index("pair_id", drop=False)
    cols = list(phase_b.UNIT_COLUMNS)
    fit_df = df.loc[unit["fit_pair_id"], cols].reset_index(drop=True)
    held_df = df.loc[unit["held_pair_id"], cols].reset_index(drop=True)

    started = time.perf_counter()
    got = phase_b.fit_two_part_unit(
        fit_df.to_dict("records"), held_df.to_dict("records"), unit["inner"],
        encodings=tuple(unit["encodings"]), seeds=tuple(unit["seeds"]),
        selected={k: (tuple(v[0]), v[1]) for k, v in unit["selected"].items()},
        input_set=unit["input_set"])
    got.update({
        "outer_fold": unit["outer_fold"],
        "unit_seconds": round(time.perf_counter() - started, 2),
        "threads_env": {v: os.environ.get(v) for v in IMAGE_THREAD_VARS},
        "memory": _container_memory(),
    })
    return got


# --------------------------------------------------------------------------
# client side


def _load_units(folds: int | None = None, arms=None, n_inner: int = 5,
                seeds=None) -> list[dict]:
    """Build every unit payload client-side. Reads the repo; containers do not."""
    sys.path.insert(0, str(REPO))
    import pandas as pd

    from pepstab import nested, phase_b
    from pepstab.censored import DEFAULT_FLOOR_HOURS
    from pepstab.data import load_with_splits

    seeds = phase_b.SEEDS if seeds is None else tuple(seeds)
    arms = phase_b.ARMS if arms is None else arms

    import hashlib

    from pepstab.data import RAW_CSV

    raw_sha256 = hashlib.sha256(RAW_CSV.read_bytes()).hexdigest()
    df = load_with_splits()
    train = df[df.split == "train"].reset_index(drop=True)
    fa = pd.read_csv(FOLDS_CSV)
    pair_ids = train.pair_id.to_numpy()

    # Frozen assignment, asserted against the harness's own cut. If these ever
    # disagree the frozen file wins, but it means cv_folds is not deterministic
    # and three phases are relying on it -- worth failing loudly.
    import numpy as np

    frozen_outer = (fa[fa.inner_fold == -1].set_index("pair_id").outer_fold
                    .reindex(pair_ids).to_numpy())
    assert np.array_equal(frozen_outer, nested.outer_folds(train)), (
        "frozen outer folds disagree with nested.outer_folds()")

    splits = list(nested.iter_outer(train, n_inner))
    if folds is not None:
        splits = splits[:folds]

    units = []
    for split in splits:
        fit_df = train[split.fit_mask].reset_index(drop=True)
        held_df = train[split.held_mask].reset_index(drop=True)
        frozen_inner = (fa[(fa.outer_fold == split.fold) & (fa.inner_fold >= 0)]
                        .set_index("pair_id").inner_fold
                        .reindex(fit_df.pair_id.to_numpy()).to_numpy())
        if n_inner == 5:
            assert np.array_equal(frozen_inner, split.inner), (
                f"frozen inner folds disagree on outer fold {split.fold}")
        for arm_id, kind, stop_on, _label in arms:
            units.append({
                "arm": arm_id, "kind": kind, "stop_on": stop_on,
                "outer_fold": int(split.fold),
                "fit_pair_id": fit_df.pair_id.astype(int).tolist(),
                "held_pair_id": held_df.pair_id.astype(int).tolist(),
                "raw_sha256": raw_sha256,
                "inner": split.inner.astype(int).tolist(),
                "encodings": list(phase_b.ENCODINGS), "seeds": list(seeds),
                "selected": {k: [list(v[0]), v[1]]
                             for k, v in phase_b.SELECTED.items()},
                "input_set": phase_b.INPUT_SET,
                "floor_hours": DEFAULT_FLOOR_HOURS,
            })
    return units


def _write_results(results: list[dict], suffix: str) -> None:
    """Write members and run records client-side, as the house pattern requires."""
    import numpy as np
    import pandas as pd

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    store, run_rows = {}, []
    for res in results:
        key = f"{res['arm']}__{res['outer_fold']}"
        store[f"{key}__mus"] = np.asarray(res["mus"], dtype=float)
        store[f"{key}__sigmas"] = np.asarray(res["sigmas"], dtype=float)
        store[f"{key}__sigmas_from_dev_mse"] = np.asarray(
            res["sigmas_from_dev_mse"], dtype=float)
        store[f"{key}__held_pair_id"] = np.asarray(res["held_pair_id"], dtype=np.int64)
        store[f"{key}__fit_pair_id"] = np.asarray(res["fit_pair_id"], dtype=np.int64)
        store[f"{key}__inner_oof"] = np.asarray(res["inner_oof"], dtype=float)
        for rec in res["records"]:
            run_rows.append({"arm": res["arm"], "outer_fold": res["outer_fold"],
                             **rec})
    np.savez_compressed(OUT_DIR / f"phase_b_nested_members{suffix}.npz", **store)
    pd.DataFrame(run_rows).to_csv(
        OUT_DIR / f"phase_b_nested_runs{suffix}.csv", index=False)
    cost = pd.DataFrame([{
        "arm": r["arm"], "outer_fold": r["outer_fold"],
        "n_members": r["n_members"], "unit_seconds": r["unit_seconds"],
        "cgroup_peak_gib": r["memory"].get("cgroup_peak_gib"),
        "self_max_rss_gib": r["memory"].get("self_max_rss_gib"),
        "threads": r["threads_env"].get("OMP_NUM_THREADS"),
    } for r in results])
    cost.to_csv(OUT_DIR / f"phase_b_modal_units{suffix}.csv", index=False)
    print(f"\nwrote members, runs and unit telemetry to {OUT_DIR} (suffix "
          f"{suffix!r})")
    print(cost.to_string(index=False))


@app.local_entrypoint()
def forecast(profile: str = PROFILES[0], seconds_per_unit: float = 480.0):
    """Price the batch before launching it. Reads the repo, spends nothing.

    ``seconds_per_unit`` defaults to a placeholder and is labelled as such: the
    only forecast worth logging is one priced from ``::smoke``'s measured time.
    """
    check_profile(profile)
    units = _load_units()
    est = price(len(units), seconds_per_unit)
    print(f"Phase B batch: {est['n_units']} units "
          f"({len(set(u['arm'] for u in units))} arms x "
          f"{len(set(u['outer_fold'] for u in units))} outer folds), "
          f"{est['n_units'] * 30} networks")
    print(f"  assumed {seconds_per_unit:.0f} s/unit  "
          f"(PLACEHOLDER unless taken from ::smoke)")
    print(f"  {est['containers']} containers, {est['waves']} wave(s), "
          f"~{est['wall_clock_minutes']:.0f} min wall clock")
    print(f"  reserved {est['reserved_core_hours']:.4f} core-hours and "
          f"{est['reserved_gib_hours']:.3f} GiB-hours")
    print(f"  startup is {est['startup_share_of_reserved']:.1%} of reserved time "
          f"-- short units pay for it")
    print(f"  CPU ${est['cpu_usd']:.4f} + memory ${est['mem_usd']:.4f} = "
          f"TOTAL ${est['total_usd']:.4f}")
    print(f"  rates from reports/ectodomain_rates.json: "
          f"${est['rates']['cpu_hour']}/core-hour, "
          f"${est['rates']['mem_gib_hour']}/GiB-hour")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "phase_b_modal_forecast.json").write_text(json.dumps(est, indent=2))
    return est


@app.local_entrypoint()
def smoke(profile: str = PROFILES[0]):
    """End-to-end on 3 units (one per arm, outer fold 0, 2 inner folds, 1 seed).

    Required by CLAUDE.md before any batch job: no batch without a passing
    end-to-end pilot on 3-5 examples. It exercises the real image, the real
    payload shape and the real science layer, and it measures the per-unit time
    the forecast needs.
    """
    check_profile(profile)
    units = _load_units(folds=1, n_inner=2, seeds=(0,))
    print(f"smoke: {len(units)} units, "
          f"{len(units) * 2 * len(units[0]['encodings'])} networks total")
    results = list(run_unit.map(units))
    for res in results:
        print(f"  {res['arm']} fold {res['outer_fold']}: {res['n_members']} members, "
              f"{res['unit_seconds']:.1f}s, peak "
              f"{res['memory'].get('cgroup_peak_gib')} GiB, threads "
              f"{res['threads_env'].get('OMP_NUM_THREADS')}")
    _write_results(results, "_smoke")
    per_net = sum(r["unit_seconds"] for r in results) / sum(
        r["n_members"] for r in results)
    full_unit_seconds = per_net * 30
    print(f"\nmeasured {per_net:.2f} s/network -> {full_unit_seconds:.0f} s for a "
          f"30-member unit")
    est = price(15, full_unit_seconds)
    print(f"forecast for the full 15-unit batch from MEASURED time: "
          f"${est['total_usd']:.4f}, ~{est['wall_clock_minutes']:.0f} min wall")
    (OUT_DIR / "phase_b_modal_smoke.json").write_text(json.dumps(
        {"per_network_seconds": per_net, "unit_seconds_30_members": full_unit_seconds,
         "forecast_from_measured": est,
         "units": [{"arm": r["arm"], "outer_fold": r["outer_fold"],
                    "n_members": r["n_members"], "unit_seconds": r["unit_seconds"],
                    "memory": r["memory"]} for r in results]}, indent=2))
    return est


@app.local_entrypoint()
def run(profile: str = PROFILES[0]):
    """The full batch: 15 units, 450 networks."""
    check_profile(profile)
    units = _load_units()
    print(f"Phase B batch: {len(units)} units, {len(units) * 30} networks")
    results = list(run_unit.map(units))
    for res in sorted(results, key=lambda r: (r["arm"], r["outer_fold"])):
        print(f"  {res['arm']} fold {res['outer_fold']}: {res['n_members']} members, "
              f"{res['unit_seconds']:.1f}s, peak "
              f"{res['memory'].get('cgroup_peak_gib')} GiB")
    _write_results(results, "")
    total = sum(r["unit_seconds"] for r in results)
    est = price(len(units), total / len(units))
    print(f"\nrealised: {total / 60:.1f} container-minutes busy, "
          f"priced ${est['total_usd']:.4f} (upper bound: reserved time includes "
          f"startup)")
    (OUT_DIR / "phase_b_modal_realised.json").write_text(json.dumps(est, indent=2))
    return est


@app.local_entrypoint()
def two_part_smoke(profile: str = PROFILES[0]):
    """End-to-end on one outer fold, 2 inner folds, 1 seed = 8 networks."""
    check_profile(profile)
    units = [u for u in _load_units(folds=1, n_inner=2, seeds=(0,))
             if u["arm"] == "B0"]
    for u in units:
        u.pop("arm", None), u.pop("kind", None), u.pop("stop_on", None)
    print(f"two-part smoke: {len(units)} unit(s)")
    results = list(run_two_part_unit.map(units))
    for res in results:
        print(f"  fold {res['outer_fold']}: {res['n_members']} members, "
              f"{res['unit_seconds']:.1f}s, peak "
              f"{res['memory'].get('cgroup_peak_gib')} GiB")
    _write_two_part(results, "_smoke")
    per_net = sum(r["unit_seconds"] for r in results) / sum(
        r["n_members"] for r in results)
    est = price(5, per_net * 60)
    print(f"\nmeasured {per_net:.2f} s/network -> 5 units x 60 members = "
          f"${est['total_usd']:.4f}, ~{est['wall_clock_minutes']:.0f} min wall")
    (OUT_DIR / "phase_b_two_part_smoke.json").write_text(json.dumps(
        {"per_network_seconds": per_net, "forecast_from_measured": est}, indent=2))
    return est


@app.local_entrypoint()
def two_part(profile: str = PROFILES[0]):
    """The two-part model on all five outer folds: 5 units, 300 networks."""
    check_profile(profile)
    units = [u for u in _load_units() if u["arm"] == "B0"]
    for u in units:
        u.pop("arm", None), u.pop("kind", None), u.pop("stop_on", None)
    print(f"two-part batch: {len(units)} units, {len(units) * 60} networks")
    results = list(run_two_part_unit.map(units))
    for res in sorted(results, key=lambda r: r["outer_fold"]):
        print(f"  fold {res['outer_fold']}: {res['n_members']} members, "
              f"{res['unit_seconds']:.1f}s")
    _write_two_part(results, "")
    total = sum(r["unit_seconds"] for r in results)
    est = price(len(units), total / len(units))
    print(f"\nrealised: {total / 60:.1f} container-minutes busy, priced "
          f"${est['total_usd']:.4f}")
    (OUT_DIR / "phase_b_two_part_realised.json").write_text(json.dumps(est, indent=2))
    return est


def _write_two_part(results: list[dict], suffix: str) -> None:
    """Members and run records for the two-part model, written client-side."""
    import numpy as np
    import pandas as pd

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    store, rows = {}, []
    for res in results:
        k = res["outer_fold"]
        for component in ("above", "magnitude"):
            store[f"{component}__{k}__preds"] = np.asarray(
                res[f"{component}_preds"], dtype=float)
            store[f"{component}__{k}__inner_oof"] = np.asarray(
                res[f"{component}_inner_oof"], dtype=float)
        store[f"meta__{k}__held_pair_id"] = np.asarray(res["held_pair_id"],
                                                       dtype=np.int64)
        store[f"meta__{k}__fit_pair_id"] = np.asarray(res["fit_pair_id"],
                                                      dtype=np.int64)
        for rec in res["records"]:
            rows.append({"outer_fold": k, **rec})
    np.savez_compressed(OUT_DIR / f"phase_b_two_part_members{suffix}.npz", **store)
    pd.DataFrame(rows).to_csv(
        OUT_DIR / f"phase_b_two_part_runs{suffix}.csv", index=False)
    print(f"wrote two-part members and runs to {OUT_DIR} (suffix {suffix!r})")
