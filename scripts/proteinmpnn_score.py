#!/usr/bin/env python
"""Score peptide chains with ProteinMPNN over a directory of predictions.

Takes any directory containing prediction folders (one ``metadata.json`` plus
one mmCIF each -- the stage 4c pilot layout, and the same layout the Modal
production volume writes) and emits one tidy row per prediction keyed by
``(model, allele, peptide, arm, seed)``.  Nothing under ``structures/`` is
written or modified.

    python scripts/proteinmpnn_score.py \
        --structures structures/ectodomain_pilot \
        --out reports/stage5_inverse_folding_pilot.csv \
        --cross-peptide --workers 2

CPU only.  ``--workers`` is capped at 2 by default because this machine is
shared; each worker is also pinned to ``--threads`` torch threads.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _init(threads: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)
    import torch

    torch.set_num_threads(threads)


def _one(job: tuple) -> dict:
    folder, decoys, n_orders, checkpoint, seed, threads, batch_rows = job
    _init(threads)
    from pepstab.inverse_folding import score_folder

    t0 = time.time()
    try:
        row = score_folder(
            Path(folder),
            decoys=decoys,
            n_orders=n_orders,
            checkpoint=checkpoint,
            seed=seed,
            batch_rows=batch_rows,
        )
    except Exception as exc:  # a failed structure must not kill the sweep
        row = {"folder": str(folder), "status": f"{type(exc).__name__}: {exc}"}
    row["folder"] = str(Path(folder).relative_to(REPO)) if str(folder).startswith(str(REPO)) else str(folder)
    row["score_s"] = round(time.time() - t0, 2)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--structures", type=Path, required=True,
                    help="directory searched recursively for prediction folders")
    ap.add_argument("--out", type=Path, required=True, help="output CSV")
    ap.add_argument("--n-orders", type=int, default=16,
                    help="autoregressive decoding orders averaged per sequence")
    ap.add_argument("--checkpoint", default="v_48_020.pt")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for decoding orders")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    ap.add_argument("--batch-rows", type=int, default=8,
                    help="max (orders x candidates) rows per forward pass; "
                         "this is the memory knob on a shared machine")
    ap.add_argument("--cross-peptide", action="store_true",
                    help="also score every other peptide in the sweep on each "
                         "backbone (the circularity control); requires all "
                         "peptides to share one length")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N (smoke)")
    ap.add_argument("--filter", default="",
                    help="comma-separated substrings; keep folders whose path "
                         "contains ALL of them (e.g. '/boltz2/,_arm_B')")
    ap.add_argument("--exclude", default="",
                    help="comma-separated substrings; drop folders whose path "
                         "contains ANY of them (e.g. '_smoke,_shards,_failed')")
    args = ap.parse_args()

    from pepstab.inverse_folding import find_predictions, proteinmpnn_provenance

    folders = find_predictions(args.structures)
    if args.filter:
        want = [t for t in args.filter.split(",") if t]
        folders = [f for f in folders if all(t in str(f) for t in want)]
    if args.exclude:
        drop = [t for t in args.exclude.split(",") if t]
        folders = [f for f in folders if not any(t in str(f) for t in drop)]
    if args.limit:
        folders = folders[: args.limit]
    if not folders:
        print(f"no prediction folders under {args.structures}", file=sys.stderr)
        return 1

    decoys: dict[str, str] | None = None
    if args.cross_peptide:
        peptides = sorted(
            {json.loads((f / "metadata.json").read_text())["inputs"]["peptide"]
             for f in folders}
        )
        lengths = {len(p) for p in peptides}
        if len(lengths) != 1:
            print(f"--cross-peptide needs one peptide length, saw {sorted(lengths)}",
                  file=sys.stderr)
            return 1
        decoys = {p: p for p in peptides}  # score_folder drops the native match

    prov = proteinmpnn_provenance(args.checkpoint)
    print(json.dumps(prov), flush=True)
    print(f"{len(folders)} predictions, n_orders={args.n_orders}, "
          f"candidates={1 if not decoys else len(decoys)}, workers={args.workers}",
          flush=True)

    jobs = [(str(f), decoys, args.n_orders, args.checkpoint, args.seed,
             args.threads, args.batch_rows) for f in folders]
    rows: list[dict] = []
    t0 = time.time()
    workers = max(1, min(args.workers, 2))
    if workers == 1:
        results = map(_one, jobs)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(_one, jobs)
    for i, row in enumerate(results, 1):
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in
                          ("model", "complex_id", "arm", "seed", "status",
                           "pep_ll_total", "native_margin", "score_s")}), flush=True)
        if i % 10 == 0:
            print(f"# {i}/{len(jobs)}  {time.time() - t0:.0f}s elapsed", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    meta = args.out.with_suffix(".provenance.json")
    meta.write_text(json.dumps({
        "provenance": prov,
        "structures": str(args.structures),
        "n_orders": args.n_orders,
        "decoding_order_seed": args.seed,
        "cross_peptide": bool(decoys),
        "candidates": sorted(decoys) if decoys else [],
        "n_predictions": len(rows),
        "n_ok": sum(r.get("status") == "ok" for r in rows),
        "wall_s": round(time.time() - t0, 1),
        "workers": workers,
        "threads_per_worker": args.threads,
    }, indent=2) + "\n")
    bad = [r for r in rows if r.get("status") != "ok"]
    print(f"wrote {args.out} ({len(rows)} rows, {len(bad)} failed) "
          f"in {time.time() - t0:.0f}s", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
