"""Stage 4c production: arm B, Boltz-2, full 28,166-pair cohort, two workspaces.

ESMFold2 is not run here. It failed its stage 4c pilot gate on the sentinel
complex and costs 4.55x more per fold; the matched cross-model comparison is
already delivered by the 90-fold pilot. See reports/stage4c_ectodomain_pilot.md.

The cohort and its shard schedule are frozen in ``data/structural_cohort.csv``
(written once by ``scripts/freeze_structural_cohort.py``). Each Modal profile
folds its own half; the halves are interleaved pair-by-pair, so either half
alone is still balanced across alleles and splits.

    # once per workspace, CPU only
    MODAL_PROFILE=a-cheparukhin modal run modal_app/ectodomain_production.py::setup
    MODAL_PROFILE=colleague     modal run modal_app/ectodomain_production.py::setup

    # end-to-end check on one short shard before committing the night
    MODAL_PROFILE=a-cheparukhin modal run modal_app/ectodomain_production.py::production --limit 5

    # the run itself
    MODAL_PROFILE=a-cheparukhin modal run --detach modal_app/ectodomain_production.py::production
    MODAL_PROFILE=colleague     modal run --detach modal_app/ectodomain_production.py::production

Re-running ``production`` is the resume path: shards with a committed marker on
the output Volume are skipped, so an interrupted run continues where it stopped.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import modal

from boltz_common import (
    CACHE_DIR,
    HF_REPO,
    HF_REVISION,
    MSA_DIR,
    OUT_DIR,
    boltz_image,
    download_image,
    msa_vol,
    out_vol,
    weights_vol,
)
from ectodomain_common import CAP, MSA_ROOT, OUT_ROOT, cgroup_peak

app = modal.App("pepstab-ectodomain-production")

# Modal imports this whole module in *every* container, so each image must
# carry the local modules imported at module scope -- including the small
# CPU-only download image, which otherwise dies on `import ectodomain_common`
# before it runs a line of its own. See the same warning in boltz_common.
_LOCAL = ("ectodomain_common", "esmfold_common", "boltz_bench")
image = boltz_image.add_local_python_source(*_LOCAL)
setup_image = download_image.add_local_python_source(*_LOCAL)

REPO = Path(__file__).resolve().parent.parent
RUN_ID = "ectodomain-20261004"

# Measured on the stage 4c pilot (A10G, 383 residues, 1,024 MSA rows), which is
# the run that gated this configuration: 16.76 s median steady fold, with model
# load and preprocessing adding ~57 s to a shard's first fold.
STEADY_S = 16.76
SHARD_STARTUP_S = 57.0
# $1.10 A10G + 4 CPU + 24 GiB host, the shape the pilot actually measured.
RATE_USD_PER_HOUR = 1.4812
WORKERS_PER_PROFILE = 10

# Must match scripts/freeze_structural_cohort.py: these are the local Modal
# profile names, and `colleague` points at the `sofyaleyn` workspace.
PROFILES = ("a-cheparukhin", "colleague")


def prod_root(run_id: str) -> Path:
    return OUT_ROOT / run_id / "boltz2" / "production"


def allele_slug(allele: str) -> str:
    return allele.replace("*", "_").replace(":", "_").replace("(", "_").replace(")", "")


def case_dir(run_id: str, case: dict, smoke: bool = False) -> Path:
    base = prod_root(run_id) / "_smoke" if smoke else prod_root(run_id)
    return base / allele_slug(case["allele"]) / case["complex_id"]


def shard_marker(run_id: str, shard: int) -> Path:
    return prod_root(run_id) / "_shards" / f"shard_{shard:04d}.json"


def check_profile(profile: str) -> str:
    """Refuse to run if the cohort half and the active Modal profile disagree.

    The two halves are disjoint, so a mismatch is the one mistake here that
    both wastes money and silently leaves pairs unfolded: both workspaces would
    fold the same half and neither would fold the other.
    """
    import os

    active = os.environ.get("MODAL_PROFILE")
    assert profile in PROFILES, f"--profile must be one of {PROFILES}, got {profile!r}"
    assert active in (None, profile), (
        f"--profile {profile!r} but MODAL_PROFILE={active!r}. "
        f"Each profile folds its own half of the cohort; set both to the same value."
    )
    if active is None:
        print(
            f"warning: MODAL_PROFILE is unset, so this runs against your active "
            f"profile. Folding the {profile!r} half -- confirm that is the same "
            f"workspace."
        )
    return profile


def build_cases(profile: str) -> list[dict]:
    """Arm-B cases for one profile, in frozen shard order.

    Reads the frozen cohort and the prepared-MSA manifest. Nothing here is
    recomputed: both files are committed inputs.
    """
    manifest = json.loads(
        (REPO / "structures/ectodomain_msas/manifest.json").read_text()
    )
    by_allele = {m["allele"]: m for m in manifest["msas"]}
    rows = [
        r
        for r in csv.DictReader((REPO / "data/structural_cohort.csv").open())
        if r["profile"] == profile
    ]
    assert rows, f"no cohort rows for profile {profile!r}"

    cases = []
    for r in rows:
        t = by_allele[r["allele"]]
        cases.append(
            {
                "complex_id": r["complex_id"],
                "allele": r["allele"],
                "peptide": r["peptide"],
                "split": r["split"],
                "pair_id": r["pair_id"],
                "arm": "B",
                "shard": int(r["shard"]),
                "chains": [
                    {"id": "A", "sequence": t["ecto"], "msa": t["B"]},
                    {"id": "B", "sequence": t["b2m"], "msa": t["beta2m"]},
                    {"id": "C", "sequence": r["peptide"], "msa": None},
                ],
            }
        )
    return cases


def yaml_for(case: dict) -> str:
    lines = ["version: 1", "sequences:"]
    for chain in case["chains"]:
        msa = str(MSA_ROOT / chain["msa"]["csv"]) if chain["msa"] else "empty"
        lines += [
            "  - protein:",
            f"      id: {chain['id']}",
            f"      sequence: {chain['sequence']}",
            f"      msa: {msa}",
        ]
    return "\n".join(lines) + "\n"


def verify_inputs(case: dict) -> dict:
    """Hash-check the MSA files this fold will actually read.

    Boltz consumes the CSV form only; the a3m siblings are the ESMFold2 path
    and are not uploaded for production, so they are not checked here.
    """
    for chain in case["chains"]:
        if not chain["msa"]:
            continue
        p = MSA_ROOT / chain["msa"]["csv"]
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        assert got == chain["msa"]["csv_sha256"], f"{p} changed on the Volume"
    return {
        "complex_id": case["complex_id"],
        "allele": case["allele"],
        "peptide": case["peptide"],
        "split": case["split"],
        "pair_id": case["pair_id"],
        "arm": "B",
        "chains": case["chains"],
        "cap": CAP,
    }


# --------------------------------------------------------------------------
# setup: pinned weights + the arm-B MSA slice this workspace needs
# --------------------------------------------------------------------------


@app.function(
    image=setup_image,
    volumes={CACHE_DIR.parent: weights_vol},
    timeout=45 * 60,
    max_containers=1,
    retries=0,
)
def download_weights(force: bool = False) -> dict:
    """Pull the pinned Boltz-2 snapshot, or confirm it is already here.

    Both checks are deliberately shallow: a top-level glob and one stat. Walking
    the 6.2 GB tree to report a size is pure latency on a Volume, and the point
    of this function is to get out of the way when there is nothing to do.
    """
    from huggingface_hub import snapshot_download

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    weights_vol.reload()
    present = sorted(p.name for p in CACHE_DIR.glob("*.ckpt"))
    if present and (CACHE_DIR / "mols").exists() and not force:
        return {
            "revision": HF_REVISION,
            "status": "cached",
            "checkpoints": present,
            "seconds": 0.0,
        }

    t0 = time.monotonic()
    snapshot_download(repo_id=HF_REPO, revision=HF_REVISION, local_dir=str(CACHE_DIR))
    weights_vol.commit()
    return {
        "revision": HF_REVISION,
        "status": "downloaded",
        "checkpoints": sorted(p.name for p in CACHE_DIR.glob("*.ckpt")),
        "seconds": round(time.monotonic() - t0, 1),
    }


@app.function(
    image=image,
    volumes={MSA_DIR: msa_vol, CACHE_DIR.parent: weights_vol},
    cpu=1,
    memory=4096,
    timeout=20 * 60,
    max_containers=1,
    retries=0,
)
def verify_workspace(profile: str, expected: list[dict]) -> dict:
    """Confirm this workspace can actually run the frozen cohort before GPUs.

    ``expected`` comes from the committed manifest on the caller's side, so
    this checks the Volume against the repository rather than against a copy
    of the manifest that was uploaded alongside the files it describes.
    """
    msa_vol.reload()
    weights_vol.reload()

    checked = 0
    for m in expected:
        for key in ("B", "beta2m"):
            p = MSA_ROOT / m[key]["csv"]
            assert p.exists(), f"missing {p}"
            got = hashlib.sha256(p.read_bytes()).hexdigest()
            assert got == m[key]["csv_sha256"], f"hash mismatch for {p}"
            checked += 1

    ckpt = sorted(p.name for p in CACHE_DIR.glob("*.ckpt"))
    assert ckpt, f"no Boltz checkpoint under {CACHE_DIR}"
    assert (CACHE_DIR / "mols").exists(), "CCD mols directory missing"
    return {
        "profile": profile,
        "alleles": len(expected),
        "msa_files_verified": checked,
        "checkpoints": ckpt,
        "ok": True,
    }


# --------------------------------------------------------------------------
# fold
# --------------------------------------------------------------------------


@app.function(
    image=image,
    volumes={CACHE_DIR.parent: weights_vol, MSA_DIR: msa_vol, OUT_DIR: out_vol},
    gpu="A10G",
    # The shape the pilot measured at $1.4812/h. 24 GiB is generous against a
    # ~10 GB observed child RSS, but the container could not read a cgroup peak
    # for the 383-residue construct, so this is not the place to shave $8.
    cpu=4,
    memory=24576,
    # ~29 min expected for a 100-case shard; 60 gives room for a slow start
    # without letting a wedged shard hold a GPU all night.
    timeout=60 * 60,
    scaledown_window=60,
    max_containers=WORKERS_PER_PROFILE,
    retries=0,
)
def fold_shard(run_id: str, shard: int, cases: list[dict], smoke: bool = False) -> dict:
    import numpy as np
    from boltz_bench import _GpuMemoryProbe, _CompletionProbe, _child_peak_rss_gb

    start = time.monotonic()
    msa_vol.reload()
    weights_vol.reload()
    out_vol.reload()

    cases = sorted(cases, key=lambda c: c["complex_id"])
    work = Path("/tmp") / f"{run_id}_shard_{shard:04d}"
    shutil.rmtree(work, ignore_errors=True)
    inputs = work / "inputs"
    inputs.mkdir(parents=True)
    shard_out = work / "out"

    for case in cases:
        verify_inputs(case)
        (inputs / f"{case['complex_id']}.yaml").write_text(yaml_for(case))

    cmd = [
        "boltz", "predict", str(inputs),
        "--out_dir", str(shard_out),
        "--cache", str(CACHE_DIR),
        "--model", "boltz2",
        "--accelerator", "gpu",
        "--devices", "1",
        "--diffusion_samples", "1",
        "--recycling_steps", "3",
        "--sampling_steps", "200",
        "--output_format", "mmcif",
        "--write_full_pae",
        "--max_msa_seqs", str(CAP),
        "--seed", "0",
        "--override",
    ]
    settings = {
        "package": "boltz==2.1.1",
        "weight_revision": HF_REVISION,
        "diffusion_samples": 1,
        "recycling_steps": 3,
        "sampling_steps": 200,
        "max_msa_seqs": CAP,
        "subsample_msa": False,
        "seed": 0,
        # Deliberately absent: --preprocessing-threads. The pilot that gated
        # this configuration ran without it, so production does too.
        "preprocessing_threads": "boltz default",
        "seed_scope": "process",
        "shard": shard,
        "requested_input_order": [c["complex_id"] for c in cases],
        "command": cmd,
    }

    invoke = time.monotonic()
    with _GpuMemoryProbe() as memory, _CompletionProbe(shard_out) as done:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3500)

    times = sorted(done.seen.items(), key=lambda x: x[1])
    settings["actual_prediction_order"] = [cid for cid, _ in times]
    deltas = {}
    previous = invoke
    for rank, (cid, at) in enumerate(times):
        deltas[cid] = (at - previous, rank == 0)
        previous = at

    results = []
    for case in cases:
        cid = case["complex_id"]
        dest = case_dir(run_id, case, smoke)
        dest.mkdir(parents=True, exist_ok=True)
        source = next(iter(shard_out.rglob(f"{cid}_model_0.cif")), None)
        rec = {
            "complex_id": cid,
            "allele": case["allele"],
            "peptide": case["peptide"],
            "split": case["split"],
            "pair_id": case["pair_id"],
            "shard": shard,
            "ok": False,
            "peak_gpu_gb": memory.peak_gb,
        }
        if source is None:
            rec["error"] = (proc.stderr or proc.stdout)[-2000:]
            results.append(rec)
            continue

        for p in source.parent.iterdir():
            if p.is_file() and (
                p.name.endswith(".cif")
                or p.name.startswith(("confidence_", "pae_", "plddt_"))
            ):
                shutil.copy2(p, dest / p.name)

        pae_files = list(dest.glob("pae_*.npz"))
        plddt_files = list(dest.glob("plddt_*.npz"))
        n = sum(len(ch["sequence"]) for ch in case["chains"])
        try:
            assert pae_files and plddt_files, "missing confidence arrays"
            with np.load(pae_files[0]) as f:
                pae = f["pae"]
            with np.load(plddt_files[0]) as f:
                plddt = f["plddt"]
            assert pae.shape == (n, n) and plddt.shape == (n,), (pae.shape, plddt.shape)
            assert np.isfinite(pae).all() and np.isfinite(plddt).all()
        except AssertionError as exc:
            rec["error"] = f"validation failed: {exc}"
            results.append(rec)
            continue

        dt, warm = deltas.get(cid, (None, False))
        rec.update(
            ok=True,
            fold_s=dt,
            is_shard_first=warm,
            pae_shape=list(pae.shape),
            plddt_shape=list(plddt.shape),
        )

        metadata = {
            "model": "boltz2",
            "run_id": run_id,
            "inputs": verify_inputs(case),
            "settings": settings,
            **rec,
        }
        metadata["output_hashes"] = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in dest.iterdir()
            if p.is_file() and p.name != "metadata.json"
        }
        (dest / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        results.append(rec)

    ok = sum(1 for r in results if r["ok"])
    batch = {
        "run_id": run_id,
        "shard": shard,
        "model": "boltz2",
        "arm": "B",
        "n": len(cases),
        "ok": ok,
        "failed": len(cases) - ok,
        "returncode": proc.returncode,
        "container_s": time.monotonic() - start,
        "batch_s": time.monotonic() - invoke,
        "peak_gpu_gb": memory.peak_gb,
        "host_peak_rss_gb": _child_peak_rss_gb(),
        "cgroup_peak_gb": cgroup_peak(),
        "results": results,
        "stderr_tail": proc.stderr[-3000:] if ok < len(cases) else "",
    }

    base = prod_root(run_id) / "_smoke" if smoke else prod_root(run_id)
    batch["smoke"] = smoke
    # The marker is written last and only for a fully successful shard, so a
    # resume re-runs anything that failed or died mid-way. A smoke shard never
    # writes one: it is a harness check, not cohort progress.
    if ok == len(cases) and not smoke:
        marker = shard_marker(run_id, shard)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(batch, indent=2) + "\n")
    else:
        (base / "_failed").mkdir(parents=True, exist_ok=True)
        (base / "_failed" / f"shard_{shard:04d}_{int(time.time())}.json").write_text(
            json.dumps(batch, indent=2) + "\n"
        )
    out_vol.commit()
    return batch


@app.function(
    image=image,
    volumes={OUT_DIR: out_vol},
    cpu=0.25,
    memory=1024,
    timeout=10 * 60,
    max_containers=1,
    retries=0,
)
def completed_shards(run_id: str) -> list[int]:
    out_vol.reload()
    d = prod_root(run_id) / "_shards"
    if not d.exists():
        return []
    return sorted(int(p.stem.split("_")[1]) for p in d.glob("shard_*.json"))


# --------------------------------------------------------------------------
# entrypoints
# --------------------------------------------------------------------------


@app.local_entrypoint()
def setup(profile: str = "a-cheparukhin", force_weights: bool = False):
    """CPU only. Pull pinned weights, then verify this workspace end to end.

    A workspace that already holds the arm-B MSAs (a-cheparukhin does, from the
    pilot) needs nothing uploaded; this just checks them. For a fresh workspace
    see docs/ECTODOMAIN_FOLDING_PLAN.md: stage the 29.6 MB slice and
    `modal volume put` it before running this.
    """
    check_profile(profile)
    manifest = json.loads(
        (REPO / "structures/ectodomain_msas/manifest.json").read_text()
    )
    expected = [
        {k: m[k] for k in ("allele", "B", "beta2m")} for m in manifest["msas"]
    ]
    w = download_weights.remote(force=force_weights)
    print(
        f"weights {w['revision'][:8]}: {w['status']}, "
        f"{', '.join(w['checkpoints'])} ({w['seconds']} s)"
    )
    v = verify_workspace.remote(profile, expected)
    print(json.dumps(v, indent=2))
    print("\nworkspace ready. Next: ::smoke")


@app.local_entrypoint()
def smoke(profile: str = "a-cheparukhin", run_id: str = RUN_ID, n: int = 5):
    """End-to-end GPU check on n real cases before committing to the cohort.

    Required by the project invariant: no batch GPU job without a passing
    end-to-end pilot on 3-5 examples. Outputs go to a `_smoke` subtree and no
    shard marker is written, so this never counts as cohort progress.
    """
    check_profile(profile)
    cases = build_cases(profile)[:n]
    usd = (SHARD_STARTUP_S + n * STEADY_S) / 3600 * RATE_USD_PER_HOUR
    print(f"{n} cases from the {profile!r} half, ~${usd:.2f}, ~{(SHARD_STARTUP_S + n * STEADY_S) / 60:.0f} min")
    for c in cases:
        print(f"  {c['complex_id']:28} {c['split']}")

    batch = fold_shard.remote(run_id, 0, cases, smoke=True)
    print(f"\n{batch['ok']}/{batch['n']} ok in {batch['container_s'] / 60:.1f} min, "
          f"peak GPU {batch['peak_gpu_gb']} GiB")
    for r in batch["results"]:
        flag = "ok" if r["ok"] else f"FAILED: {r.get('error', '')[:200]}"
        fold = f"{r['fold_s']:6.1f} s" if r.get("fold_s") else "     - "
        print(f"  {r['complex_id']:28} {fold}  {flag}")
    assert batch["ok"] == batch["n"], "smoke check failed; do not launch production"
    print("\nsmoke passed. Next: ::production")


@app.local_entrypoint()
def production(
    profile: str = "a-cheparukhin",
    run_id: str = RUN_ID,
    limit: int = 0,
    dry_run: bool = False,
):
    """Fold this profile's half of the frozen cohort, resuming what is missing.

    ``limit`` caps the number of shards. The per-workspace budget ceiling is in
    HACKATHON_PLAN.md; the forecast printed below is what to check against it.
    """
    check_profile(profile)
    cases = build_cases(profile)
    by_shard: dict[int, list[dict]] = {}
    for c in cases:
        by_shard.setdefault(c["shard"], []).append(c)

    done = set(completed_shards.remote(run_id))
    todo = sorted(s for s in by_shard if s not in done)
    if limit:
        todo = todo[:limit]

    n_cases = sum(len(by_shard[s]) for s in todo)
    gpu_s = len(todo) * SHARD_STARTUP_S + n_cases * STEADY_S
    usd = gpu_s / 3600 * RATE_USD_PER_HOUR
    wall_h = gpu_s / 3600 / WORKERS_PER_PROFILE

    print(f"profile          {profile}")
    print(f"cohort half      {len(cases)} pairs in {len(by_shard)} shards")
    print(f"already complete {len(done & set(by_shard))} shards")
    print(f"this run         {len(todo)} shards / {n_cases} folds")
    print(f"forecast         {gpu_s / 3600:.1f} GPU-h, ${usd:.2f}, "
          f"~{wall_h:.1f} h wall at {WORKERS_PER_PROFILE} workers")
    print(f"with 25% margin  ${usd * 1.25:.2f}, ~{wall_h * 1.25:.1f} h")

    if not todo:
        print("\nnothing to do: every shard for this profile is complete.")
        return
    if dry_run:
        print(f"\ndry run: would spawn shards {todo[0]}..{todo[-1]}")
        return

    dest = REPO / "reports" / run_id
    dest.mkdir(parents=True, exist_ok=True)
    log = dest / f"production_{profile}.jsonl"

    started = time.monotonic()
    folded = failed = 0
    with log.open("a") as fh:
        for batch in fold_shard.starmap(
            [(run_id, s, by_shard[s]) for s in todo],
            order_outputs=False,
            return_exceptions=True,
        ):
            if isinstance(batch, Exception):
                failed += 1
                print(f"  shard raised: {type(batch).__name__}: {batch}", flush=True)
                continue
            folded += batch["ok"]
            failed += batch["failed"]
            fh.write(json.dumps(batch) + "\n")
            fh.flush()
            elapsed = (time.monotonic() - started) / 3600
            print(
                f"  shard {batch['shard']:4d}  {batch['ok']:3d}/{batch['n']} ok  "
                f"{batch['container_s'] / 60:5.1f} min  "
                f"[{folded}/{n_cases} folded, {failed} failed, {elapsed:.2f} h]",
                flush=True,
            )

    print(f"\n{folded} folded, {failed} failed in {(time.monotonic() - started) / 3600:.2f} h")
    print(f"log: {log.relative_to(REPO)}")
    if failed:
        print("re-run the same command to retry only the incomplete shards.")
