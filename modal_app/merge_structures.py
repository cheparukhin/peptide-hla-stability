"""Copy the `colleague` half of the stage 4c production folds into `a-cheparukhin`.

A Modal Volume belongs to one workspace, so the production fold left two
disjoint halves on two ``pepstab-structures`` Volumes. This app merges them
into the ``a-cheparukhin`` Volume, under the same tree, without the data
passing through a laptop:

* It runs **in the `colleague` workspace**, with that workspace's Volume
  mounted read-only at ``/src``.
* It opens the ``a-cheparukhin`` Volume through a second Modal client built
  from that profile's token, and writes with ``batch_upload``. The token is
  read from the local ``~/.modal.toml`` at launch and passed as an ephemeral
  ``Secret.from_dict``, so it is never saved as a named Secret in the
  `colleague` workspace.

CPU only. The source Volume is never written to.

**What is copied.** Exactly the colleague-half complex IDs from the frozen
cohort (``data/structural_cohort.csv``) -- not whatever happens to be on the
source Volume. They are disjoint from the a-cheparukhin half, so they land
beside it in the same allele folders. The two bookkeeping folders do collide
(both profiles numbered their shard markers from 0), so ``_shards`` and
``_smoke`` go to ``_shards_colleague`` and ``_smoke_colleague``.

**Never overwrites.** ``batch_upload(force=False)`` fails on an existing path,
so a name clash stops the copy instead of replacing a file. Pairs already on
the destination are skipped, which makes ``merge`` the resume path.

    # 5 pairs into a scratch prefix, verified by sha256, then removed
    MODAL_PROFILE=colleague modal run modal_app/merge_structures.py::smoke

    # the full colleague half (the forecast printed by smoke prices it)
    MODAL_PROFILE=colleague modal run --detach modal_app/merge_structures.py::merge

    # free: compares the two Volumes' listings path by path, from here
    modal run modal_app/merge_structures.py::verify
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
import tomllib
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
RUN_ID = "ectodomain-20261004"

#: Local Modal profile names. `colleague` points at the `sofyaleyn` workspace.
SOURCE_PROFILE = "colleague"
DEST_PROFILE = "a-cheparukhin"
VOLUME = "pepstab-structures"

#: Volume-relative paths, so the same strings work for the mounted source and
#: the SDK-addressed destination. Must match ectodomain_production.prod_root.
PROD = f"stage4c/{RUN_ID}/boltz2/production"
SMOKE_PREFIX = f"stage4c/{RUN_ID}/boltz2/_merge_smoke"
#: Bookkeeping folders whose names collide across the two halves.
RENAMED = {"_shards": "_shards_colleague", "_smoke": "_smoke_colleague"}
FILES_PER_PAIR = 5

SRC = Path("/src")
src_vol = modal.Volume.from_name(VOLUME, create_if_missing=False)

#: One batch_upload per chunk: a failure loses at most one chunk, and a chunk
#: is committed whole, so a pair is never left half-copied on the destination.
CHUNK = 100
MAX_CONTAINERS = 10
WORKER_CPU = 1.0
WORKER_MEMORY = 1024
RATES_PATH = REPO / "reports" / "ectodomain_rates.json"

app = modal.App("pepstab-merge-structures")
image = modal.Image.debian_slim(python_version="3.12")


def check_profile() -> None:
    """The source Volume is resolved in whichever workspace runs the app."""
    active = os.environ.get("MODAL_PROFILE")
    assert active == SOURCE_PROFILE, (
        f"MODAL_PROFILE={active!r}: run this as MODAL_PROFILE={SOURCE_PROFILE}, "
        f"which mounts the colleague half as the source."
    )


def dest_secret() -> modal.Secret:
    """Ephemeral: lives for this app run only, never saved by name."""
    cfg = tomllib.loads(Path("~/.modal.toml").expanduser().read_text())[DEST_PROFILE]
    return modal.Secret.from_dict(
        {"DEST_TOKEN_ID": cfg["token_id"], "DEST_TOKEN_SECRET": cfg["token_secret"]}
    )


def profile_client(profile: str) -> modal.Client:
    cfg = tomllib.loads(Path("~/.modal.toml").expanduser().read_text())[profile]
    return modal.Client.from_credentials(cfg["token_id"], cfg["token_secret"])


def dest_volume() -> modal.Volume:
    client = modal.Client.from_credentials(
        os.environ["DEST_TOKEN_ID"], os.environ["DEST_TOKEN_SECRET"]
    )
    return modal.Volume.from_name(VOLUME, client=client)


def allele_slug(allele: str) -> str:
    """Same as ectodomain_production.allele_slug, which this must not import:
    that module pulls in the Boltz image definitions."""
    return allele.replace("*", "_").replace(":", "_").replace("(", "_").replace(")", "")


def colleague_pairs() -> dict[str, list[str]]:
    """Colleague-half complex IDs by allele folder, from the frozen cohort."""
    by_allele: dict[str, list[str]] = {}
    with open(REPO / "data" / "structural_cohort.csv") as f:
        for row in csv.DictReader(f):
            if row["profile"] == SOURCE_PROFILE:
                by_allele.setdefault(allele_slug(row["allele"]), []).append(row["complex_id"])
    assert sum(map(len, by_allele.values())) == 14_083
    return {k: sorted(v) for k, v in sorted(by_allele.items())}


# --------------------------------------------------------------------------
# remote
# --------------------------------------------------------------------------


@app.function(
    image=image,
    volumes={SRC: src_vol.read_only()},
    cpu=WORKER_CPU,
    memory=WORKER_MEMORY,
    timeout=60 * 60,
    max_containers=MAX_CONTAINERS,
    retries=1,
)
def copy_pairs(allele: str, complex_ids: list[str], dest_prefix: str = PROD) -> dict:
    """Copy one allele's colleague pairs; skip any already on the destination."""
    t0 = time.monotonic()
    dest = dest_volume()
    try:
        existing = {e.path.rsplit("/", 1)[-1] for e in dest.listdir(f"{dest_prefix}/{allele}")}
    except Exception:  # allele folder absent on the destination yet
        existing = set()

    todo, nbytes = [], 0
    for cid in complex_ids:
        src = SRC / PROD / allele / cid
        files = sorted(p for p in src.iterdir() if p.is_file())
        assert len(files) == FILES_PER_PAIR, (str(src), [p.name for p in files])
        if cid in existing:
            continue
        todo.append(src)
        nbytes += sum(p.stat().st_size for p in files)

    for i in range(0, len(todo), CHUNK):
        with dest.batch_upload(force=False) as batch:
            for src in todo[i : i + CHUNK]:
                batch.put_directory(src, f"/{dest_prefix}/{allele}/{src.name}")
    return {
        "allele": allele,
        "copied": len(todo),
        "skipped": len(complex_ids) - len(todo),
        "bytes": nbytes,
        "seconds": time.monotonic() - t0,
    }


@app.function(
    image=image,
    volumes={SRC: src_vol.read_only()},
    cpu=WORKER_CPU,
    memory=WORKER_MEMORY,
    timeout=20 * 60,
    retries=1,
)
def copy_bookkeeping() -> dict:
    """``_shards`` and ``_smoke`` under non-colliding names. Idempotent."""
    dest = dest_volume()
    done = {}
    for src_name, dest_name in RENAMED.items():
        try:
            if list(dest.listdir(f"{PROD}/{dest_name}")):
                done[dest_name] = "already present"
                continue
        except Exception:
            pass
        with dest.batch_upload(force=False) as batch:
            batch.put_directory(SRC / PROD / src_name, f"/{PROD}/{dest_name}")
        done[dest_name] = sum(1 for p in (SRC / PROD / src_name).rglob("*") if p.is_file())
    return done


# --------------------------------------------------------------------------
# entrypoints
# --------------------------------------------------------------------------


def sha256_tree(vol: modal.Volume, prefix: str) -> dict[str, str]:
    out = {}
    for e in vol.listdir(prefix, recursive=True):
        if e.type == modal.volume.FileEntryType.FILE:
            rel = e.path.split(prefix.strip("/") + "/", 1)[1]
            out[rel] = hashlib.sha256(b"".join(vol.read_file(e.path))).hexdigest()
    return out


@app.local_entrypoint()
def smoke(n: int = 5):
    """n real pairs into a scratch prefix, byte-compared, timed, then removed."""
    check_profile()
    pairs = colleague_pairs()
    allele = "HLA-A_02_01"
    ids = pairs[allele][:n]
    print(f"copying {n} pairs of {allele} to {SMOKE_PREFIX}/")
    r = copy_pairs.with_options(secrets=[dest_secret()]).remote(allele, ids, SMOKE_PREFIX)
    print(json.dumps(r, indent=2))
    assert r["copied"] == n, r

    src, dest = (modal.Volume.from_name(VOLUME, client=profile_client(p))
                 for p in (SOURCE_PROFILE, DEST_PROFILE))
    for cid in ids:
        a = sha256_tree(src, f"{PROD}/{allele}/{cid}")
        b = sha256_tree(dest, f"{SMOKE_PREFIX}/{allele}/{cid}")
        assert len(a) == FILES_PER_PAIR and a == b, (cid, a, b)
    print(f"sha256 identical on all {n * FILES_PER_PAIR} files")

    dest.remove_file(SMOKE_PREFIX, recursive=True)
    print(f"removed {SMOKE_PREFIX}")

    rates = {k: float(v) for k, v in json.loads(RATES_PATH.read_text()).items()
             if k in ("cpu_hour_cost", "mem_gib_hour_cost")}
    per_pair = r["seconds"] / n  # includes listing and client setup: an upper bound
    total = sum(map(len, pairs.values()))
    container_h = total * per_pair / 3600
    usd = container_h * (WORKER_CPU * rates["cpu_hour_cost"]
                         + WORKER_MEMORY / 1024 * rates["mem_gib_hour_cost"])
    print(f"\nforecast for {total:,} pairs: {per_pair:.2f} s/pair (upper bound), "
          f"{container_h:.2f} container-h, ~{container_h / MAX_CONTAINERS * 60:.0f} min "
          f"wall at {MAX_CONTAINERS} containers, ~${usd:.2f}")


@app.local_entrypoint()
def merge():
    check_profile()
    pairs = colleague_pairs()
    secret = dest_secret()
    print(json.dumps(copy_bookkeeping.with_options(secrets=[secret]).remote()))
    totals = {"copied": 0, "skipped": 0, "bytes": 0, "seconds": 0.0}
    for r in copy_pairs.with_options(secrets=[secret]).starmap(
        list(pairs.items()), order_outputs=False
    ):
        print(f"{r['allele']:20} copied {r['copied']:4}  skipped {r['skipped']:4}  "
              f"{r['bytes'] / 1e6:7.1f} MB  {r['seconds']:6.1f} s")
        for k in totals:
            totals[k] += r[k]
    print(json.dumps(totals))
    print("\nnext: modal run modal_app/merge_structures.py::verify")


@app.local_entrypoint()
def verify():
    """Every colleague file is on the destination with the same size. No containers."""
    pairs = colleague_pairs()
    src, dest = (modal.Volume.from_name(VOLUME, client=profile_client(p))
                 for p in (SOURCE_PROFILE, DEST_PROFILE))

    def sizes(vol):
        return {e.path: e.size for e in vol.listdir(PROD, recursive=True)
                if e.type == modal.volume.FileEntryType.FILE}

    s, d = sizes(src), sizes(dest)
    expected = {}
    for path, size in s.items():
        parts = path.split("/")
        top = parts[len(PROD.split("/"))]
        if top in RENAMED:
            parts[len(PROD.split("/"))] = RENAMED[top]
        expected["/".join(parts)] = size

    missing = [p for p in expected if p not in d]
    differ = [p for p in expected if p in d and d[p] != expected[p]]
    n_pairs = len({p.rsplit("/", 1)[0] for p in d
                   if p.split("/")[len(PROD.split("/"))] not in ("_shards", "_smoke", *RENAMED.values())})
    print(f"source files {len(s):,}; destination files {len(d):,}; "
          f"missing {len(missing)}; size mismatches {len(differ)}")
    print(f"pair folders on destination: {n_pairs:,} (expected 28,166)")
    print(f"destination total {sum(d.values()) / 1e9:.2f} GB")
    assert sum(map(len, pairs.values())) == 14_083
    assert not missing and not differ, (missing[:5], differ[:5])
    assert n_pairs == 28_166, n_pairs
    print("merge verified")
