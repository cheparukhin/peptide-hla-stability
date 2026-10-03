"""CPU-only setup for the Boltz-2 pipeline: model weights and HLA MSAs.

    modal run modal_app/boltz_setup.py::setup

No GPU is declared in this app, for two reasons:

1. **It would be wrong.** `boltz predict --use_msa_server` builds alignments
   synchronously inside the predict process, before the trainer starts, so an
   accelerator would be billed while blocking on HTTP to the MSA server.
2. **It would make this unreachable.** Modal validates every function in an app
   at creation time and refuses a `gpu=` declaration on a workspace with no
   payment method, which would block the MSA step on GPU billing it does not
   need.

MSAs are per unique chain sequence, so this runs once and the whole project
reuses the cache: 8 alignments for the pilot and benchmark panels, 6 for the
~2,000-complex production panel, 75 for the entire dataset.
"""

from __future__ import annotations

import time
from pathlib import Path

import modal

from boltz_common import (
    CACHE_DIR,
    HF_REPO,
    HF_REVISION,
    MINUTES,
    MSA_DIR,
    WORKER_CPU,
    WORKER_MEM,
    boltz_image,
    download_image,
    msa_vol,
    read_msa_targets,
    weights_vol,
)

app = modal.App("pepstab-boltz-setup")


@app.function(
    image=download_image,
    volumes={CACHE_DIR.parent: weights_vol},
    timeout=30 * MINUTES,
    max_containers=1,
    retries=0,
)
def download_weights(force: bool = False) -> dict:
    """Pull the pinned Boltz-2 snapshot (~6.2 GB) onto the weights Volume."""
    from huggingface_hub import snapshot_download

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    snapshot_download(
        repo_id=HF_REPO,
        revision=HF_REVISION,
        local_dir=str(CACHE_DIR),
        force_download=force,
    )
    weights_vol.commit()
    files = {
        f.name: f.stat().st_size for f in CACHE_DIR.rglob("*") if f.is_file()
    }
    return {
        "seconds": round(time.monotonic() - t0, 1),
        "gb": round(sum(files.values()) / 1e9, 3),
        "files": sorted(files.items(), key=lambda kv: -kv[1])[:6],
    }


@app.function(
    image=boltz_image,
    volumes={MSA_DIR: msa_vol},
    # Deliberately NOT the fold worker's 4 cores / 32 GiB. This function spends
    # almost all its life in a polling loop waiting on the MSA server's queue
    # (observed: ~22 min of wall time for 8 sequences), so it is I/O-bound, and
    # Modal bills the greater of requested and used. Requesting the fold shape
    # here would cost ~10x for the same wait.
    cpu=0.25,
    memory=2048,
    timeout=90 * MINUTES,
    max_containers=1,
    retries=0,
)
def build_msas(targets: list[dict], force: bool = False) -> list[dict]:
    """Build one ``.a3m`` per unique HLA domain sequence.

    All sequences go to the MSA server in a single batched request: it is a
    free shared community service (ColabFold/MMseqs2), and one ticket for eight
    sequences is both the polite and the fast way to ask.
    """
    from boltz.data.msa.mmseqs2 import run_mmseqs2

    MSA_DIR.mkdir(parents=True, exist_ok=True)
    pending = [
        t for t in targets if force or not (MSA_DIR / f"{t['msa_id']}.a3m").exists()
    ]
    if not pending:
        return [
            {"msa_id": t["msa_id"], "status": "cached",
             "depth": _depth(MSA_DIR / f"{t['msa_id']}.a3m")}
            for t in targets
        ]

    t0 = time.monotonic()
    result = run_mmseqs2(
        [t["hla_seq"] for t in pending],
        prefix=str(MSA_DIR / "_tmp"),
        use_env=True,
        use_filter=True,
        use_pairing=False,
    )
    # Signature is annotated as a tuple but the single-chain path returns just
    # the a3m list; accept either so a boltz point release cannot break setup.
    a3m_lines = result[0] if isinstance(result, tuple) else result
    elapsed = time.monotonic() - t0

    if len(a3m_lines) != len(pending):
        raise RuntimeError(
            f"MSA server returned {len(a3m_lines)} alignments for {len(pending)} queries"
        )

    out = []
    for target, lines in zip(pending, a3m_lines):
        path = MSA_DIR / f"{target['msa_id']}.a3m"
        path.write_text(lines)
        out.append(
            {
                "msa_id": target["msa_id"],
                "alleles": target["alleles"],
                "status": "built",
                "depth": _depth(path),
                "bytes": path.stat().st_size,
                "total_seconds": round(elapsed, 1),
            }
        )
    msa_vol.commit()
    return out


def _depth(path: Path) -> int:
    """Sequence count in an a3m -- the query plus its homologues."""
    try:
        return sum(1 for ln in path.read_text().splitlines() if ln.startswith(">"))
    except OSError:
        return -1


@app.function(image=boltz_image, volumes={MSA_DIR: msa_vol}, timeout=5 * MINUTES)
def verify_msas(targets: list[dict]) -> list[dict]:
    """Confirm each a3m exists and that its first sequence is the query.

    Boltz expects the query as the first record. A misordered a3m would still
    fold, silently, against the wrong reference -- so check rather than assume.
    """
    out = []
    for t in targets:
        path = MSA_DIR / f"{t['msa_id']}.a3m"
        rec = {"msa_id": t["msa_id"], "alleles": t["alleles"], "exists": path.exists()}
        if path.exists():
            lines = path.read_text().splitlines()
            seqs = [
                lines[i + 1] for i, ln in enumerate(lines[:-1]) if ln.startswith(">")
            ]
            rec["depth"] = len(seqs)
            rec["query_matches"] = bool(seqs) and seqs[0].replace("-", "").upper() == t[
                "hla_seq"
            ].upper()
            rec["bytes"] = path.stat().st_size
        out.append(rec)
    return out


@app.local_entrypoint()
def setup(force: bool = False):
    """Download weights and build the HLA MSAs. CPU only -- no GPU is billed."""
    targets = read_msa_targets()

    print(f"weights: pulling {HF_REPO}@{HF_REVISION[:8]} ...")
    w = download_weights.remote(force=force)
    print(f"  {w['gb']} GB in {w['seconds']} s")
    for name, size in w["files"]:
        print(f"    {name:24} {size / 1e9:6.3f} GB")

    print(f"\nMSAs: {len(targets)} unique HLA domain sequences ...")
    built = build_msas.remote(targets, force=force)
    for rec in built:
        extra = f", depth {rec['depth']}" if rec.get("depth", -1) >= 0 else ""
        print(f"  {rec['msa_id']} {rec.get('alleles', ''):14} {rec['status']}{extra}")
    if built and built[0].get("total_seconds"):
        print(f"  built in {built[0]['total_seconds']} s total")

    print("\nverifying ...")
    bad = []
    for rec in verify_msas.remote(targets):
        ok = rec["exists"] and rec.get("query_matches")
        flag = "ok" if ok else "PROBLEM"
        print(
            f"  {rec['msa_id']} {rec.get('alleles', ''):14} depth "
            f"{rec.get('depth', '?'):>6}  query_first={rec.get('query_matches')}  {flag}"
        )
        if not ok:
            bad.append(rec["msa_id"])
    if bad:
        raise SystemExit(f"MSA verification failed for: {', '.join(bad)}")
    print("\nall MSAs verified. Next: modal run modal_app/boltz_bench.py::pilot")
