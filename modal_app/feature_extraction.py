"""Stage 4c.5: extract stage-5 structural features next to the folds, on CPU.

The production output is ~22 GB spread over two workspace Volumes, so pulling
it down to score locally is slower than running the scorer there. This app
mounts ``pepstab-structures`` read-only and writes one small feature CSV back.

**CPU only, deliberately.** The production fold is using up to 10 concurrent
A10G GPUs per workspace. A ``gpu=`` function declared in this app would contend
for those slots and slow the fold down, so no function here declares one.

**One profile at a time.** The two workspaces hold disjoint, pair-by-pair
interleaved halves of ``data/structural_cohort.csv``. A single profile's half
is a first-class result, not a partial failure: it is already balanced across
alleles and splits, so ~14,083 pairs from one workspace is an unbiased
diagnostic. Concatenating the halves is a separate, local step
(``scripts/extract_structural_features.py concat``), which asserts the row
count -- nothing in the output path makes a half-sized table obvious.

    # 3-5 predictions end to end, before any full pass (project invariant)
    MODAL_PROFILE=a-cheparukhin modal run modal_app/feature_extraction.py::smoke \
        --profile a-cheparukhin

    # one workspace's half
    MODAL_PROFILE=a-cheparukhin modal run modal_app/feature_extraction.py::extract \
        --profile a-cheparukhin
    MODAL_PROFILE=colleague     modal run modal_app/feature_extraction.py::extract \
        --profile colleague

    # then, locally
    python scripts/extract_structural_features.py concat \
        reports/stage4c5_features_a-cheparukhin.csv \
        reports/stage4c5_features_colleague.csv \
        --out reports/stage4c5_features_production.csv --require-full

``extract`` is re-runnable: it writes a fresh table each time and does not
mutate the fold output, which it mounts read-only.
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
RUN_ID = "ectodomain-20261004"

# Must match scripts/freeze_structural_cohort.py and
# modal_app/ectodomain_production.py: `colleague` points at the `sofyaleyn`
# workspace.
PROFILES = ("a-cheparukhin", "colleague")

OUT_DIR = Path("/structures")
# modal_app/ectodomain_production.py: OUT_ROOT / run_id / "boltz2" / "production"
PRODUCTION_ROOT = OUT_DIR / "stage4c" / RUN_ID / "boltz2" / "production"
FEATURES_DIR = OUT_DIR / "stage4c" / RUN_ID / "boltz2" / "features"

out_vol = modal.Volume.from_name("pepstab-structures", create_if_missing=False)

app = modal.App("pepstab-stage4c5-features")

# CPU-only image. biotite parses the mmCIFs; numpy does the rest. The feature
# definitions live in the repo module so the local pilot run and the cloud run
# cannot drift apart -- that is the whole point of validating on the pilot
# first.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.3", "biotite==1.7.1")
    .add_local_file(
        REPO / "pepstab" / "structural_features.py",
        "/root/structural_features.py",
        copy=True,
    )
)

# Measured on the first 1,000 production folds: 1.57 s per fold of wall time
# against 0.096 s of CPU, so extraction is Volume-read-bound, not CPU-bound,
# and reserving more than one core per container just bills idle cores.
# Peak RSS was 641 MB locally, so 2 GiB is ample.
WORKER_CPU = 1.0
WORKER_MEMORY = 2048
CHUNK = 400
MAX_CONTAINERS = 30


def check_profile(profile: str) -> str:
    """Refuse to run if the cohort half and the active Modal profile disagree.

    Same assertion as the production runner, and required by the project
    invariant: the halves are disjoint, and each Volume lives in exactly one
    workspace, so a mismatch reads the wrong half or an empty tree.
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
            f"profile. Extracting the {profile!r} half -- confirm that is the "
            f"same workspace."
        )
    return profile


# --------------------------------------------------------------------------
# listing: find the per-pair directories, skipping the harness siblings
# --------------------------------------------------------------------------


@app.function(
    image=image,
    volumes={OUT_DIR: out_vol},
    cpu=0.5,
    memory=2048,
    timeout=20 * 60,
    max_containers=1,
    retries=0,
)
def list_folds(root: str = str(PRODUCTION_ROOT)) -> list[str]:
    """Per-pair directories under ``root``.

    ``_smoke``, ``_shards`` and ``_failed`` live next to the cohort output and
    are skipped. Ingesting them would not error and would not look wrong: it
    would quietly add pre-launch harness folds to the feature table.

    **This is a live tree while the fold runs.** Directories appear in batches
    of 100 as each shard commits, so the walk tolerates entries appearing and
    disappearing underneath it. ``metadata.json`` is the completion signal --
    the production runner writes it last, after the confidence arrays are
    validated -- so a directory without one is mid-write, not broken, and is
    simply not yet ready to extract. Discovery keys on ``metadata.json`` for
    exactly that reason.

    The returned count is **not** a coverage figure. While the run is live a
    missing pair means "not yet folded", never "failed"; the authoritative
    progress records are ``_shards/shard_NNNN.json`` and the local
    ``production_<profile>.jsonl``.
    """
    import sys

    sys.path.insert(0, "/root")
    import structural_features as sf

    out_vol.reload()
    base = Path(root)
    if not base.exists():
        raise FileNotFoundError(f"{base} does not exist on the Volume")
    folders, excluded = [], 0
    try:
        for meta in base.rglob("metadata.json"):
            if sf.is_excluded(meta.parent, base):
                excluded += 1
            else:
                folders.append(str(meta.parent))
    except (FileNotFoundError, OSError) as exc:
        # A shard committing mid-walk can remove a directory entry under us.
        # Whatever was enumerated is still valid; the rest arrives next pass.
        print(f"walk interrupted by a concurrent commit ({exc}); using what was listed")
    folders.sort()
    markers = base / "_shards"
    n_shards = len(list(markers.glob("shard_*.json"))) if markers.exists() else 0
    print(
        f"{len(folders)} folds ready (metadata.json present); {excluded} skipped in "
        f"{sorted(sf.EXCLUDED_DIR_NAMES)}; "
        f"{n_shards} committed shard markers -- markers, not this count, are the "
        f"authoritative progress record"
    )
    return folders


# --------------------------------------------------------------------------
# extraction: one chunk of folds per container
# --------------------------------------------------------------------------


@app.function(
    image=image,
    volumes={OUT_DIR: out_vol},
    cpu=WORKER_CPU,
    memory=WORKER_MEMORY,
    timeout=30 * 60,
    max_containers=MAX_CONTAINERS,
    retries=1,
)
def extract_chunk(folders: list[str]) -> list[dict]:
    """Feature rows for one chunk. A failed fold becomes a recorded row."""
    import sys

    sys.path.insert(0, "/root")
    import structural_features as sf

    # This container may hold an older Volume snapshot than the lister did.
    out_vol.reload()
    started = time.time()
    rows = [sf.extract_path(Path(f)) for f in folders]
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(
        f"{len(rows)} folds, {ok} ok, {len(rows) - ok} failed, "
        f"{time.time() - started:.1f}s"
    )
    return rows


def _rows_to_csv(rows: list[dict]) -> str:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _run(profile: str, root: str, limit: int | None, chunk: int, label: str) -> list[dict]:
    check_profile(profile)
    folders = list_folds.remote(root)
    if limit:
        folders = folders[:limit]
    if not folders:
        raise SystemExit(f"no folds found under {root}")
    chunks = [folders[i : i + chunk] for i in range(0, len(folders), chunk)]
    print(f"{len(folders)} folds in {len(chunks)} chunks of up to {chunk}, CPU only")

    started = time.time()
    rows: list[dict] = []
    for part in extract_chunk.map(chunks):
        rows.extend(part)
    elapsed = time.time() - started

    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(
        json.dumps(
            {
                "profile": profile,
                "label": label,
                "rows": len(rows),
                "ok": ok,
                "failed": len(rows) - ok,
                "wall_s": round(elapsed, 1),
                "cohort_coverage": "half",
            },
            indent=2,
        )
    )
    return rows


def _write(rows: list[dict], profile: str, label: str) -> Path:
    local = REPO / "reports" / f"stage4c5_features_{label}_{profile}.csv"
    local.parent.mkdir(parents=True, exist_ok=True)
    # Stamp the half onto every row. Without it the concatenated table cannot
    # say which workspace a row came from, and "which half is this?" is
    # exactly the question the concat assertion exists to answer.
    for row in rows:
        row["profile"] = profile
        row["cohort_coverage"] = "half"
    text = _rows_to_csv(rows)
    local.write_text(text)
    print(f"wrote {local} ({len(rows)} rows)")
    print(
        "This is ONE HALF of the cohort. Join to splits and concatenate with "
        "`scripts/extract_structural_features.py concat --require-full`, which "
        "asserts the 28,166-pair total."
    )
    return local


@app.local_entrypoint()
def smoke(profile: str = PROFILES[0], root: str = str(PRODUCTION_ROOT), n: int = 5):
    """End-to-end check on 3-5 real predictions, before any full pass.

    Required by the project invariant: no batch job without a passing
    end-to-end pilot on 3-5 examples, for a new runner as well as a new model.
    """
    assert 3 <= n <= 5, "the invariant asks for 3-5 examples"
    rows = _run(profile, root, limit=n, chunk=n, label="smoke")
    bad = [r for r in rows if r.get("status") != "ok"]
    if bad:
        for row in bad:
            print(f"  FAILED {row.get('complex_id') or '?'}: {row.get('status')}")
        raise SystemExit(f"smoke failed: {len(bad)} of {len(rows)} folds did not extract")
    first = rows[0]
    for key in (
        "pep_plddt_mean",
        "pae_pep_rows_groove_cols_mean",
        "pae_groove_rows_pep_cols_mean",
        "groove_contacts_4p5A_total",
        "pep_buried_sasa_frac",
        "global_iptm",
    ):
        assert first.get(key) is not None, f"smoke row is missing {key}"
    print(f"smoke ok: {len(rows)}/{len(rows)} extracted, {len(first)} columns")
    _write(rows, profile, "smoke")


@app.local_entrypoint()
def extract(
    profile: str = PROFILES[0],
    root: str = str(PRODUCTION_ROOT),
    limit: int = 0,
    chunk: int = CHUNK,
):
    """Extract one workspace's half of the cohort."""
    rows = _run(profile, root, limit=limit or None, chunk=chunk, label="production")
    _write(rows, profile, "production")
