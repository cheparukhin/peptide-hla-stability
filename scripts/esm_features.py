"""Stage 3: build the frozen ESM-2 embedding cache, once per unique sequence.

    .venv/bin/python scripts/esm_features.py --verify-index
    .venv/bin/python scripts/esm_features.py --checkpoint esm2_t12_35M_UR50D
    .venv/bin/python scripts/esm_features.py --checkpoint esm2_t30_150M_UR50D

Embeds the 5,633 distinct peptides and 75 distinct HLA domain sequences in the
dataset -- not the 28,166 rows -- and writes ``features/esm/<checkpoint>/``.
Two layers per checkpoint (a middle one and the last), so the layer comparison
costs one forward pass rather than two.

Reads no labels and no splits, so this step cannot leak. It is also idempotent:
re-running with the same sequence set is a no-op unless ``--force`` is given.
"""

from __future__ import annotations

# --- BLAS thread pinning, before numpy/torch are imported --------------------
# Six workstreams share 8 cores on this machine. Every one of them links numpy
# against Accelerate, which sizes its thread pool *at import time* and defaults
# to one thread per core -- so six processes ask for 48 threads on 8 cores and
# the measured load average hit 114. Under that much oversubscription each
# process runs slower than it would with a single thread, because the cores are
# spent on context switching rather than on arithmetic.
#
# VECLIB_MAXIMUM_THREADS is the one that matters on macOS (numpy here links
# against Accelerate, not OpenBLAS or MKL); the others are set so the same file
# behaves on a Linux box. An operator who knows the machine is idle can
# override any of them in the environment -- these are defaults, not overrides.
import os as _os  # noqa: E402

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab import esm as pesm  # noqa: E402
from pepstab.data import load_raw  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "reports"


def unique_sequences(df: pd.DataFrame) -> dict[str, list[str]]:
    """The deduplicated, sorted sequence sets. Sorting makes the hash stable."""
    return {
        "peptide": sorted(df["peptide"].astype(str).unique().tolist()),
        "hla": sorted(df["hla_seq"].astype(str).unique().tolist()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=pesm.DEFAULT_CHECKPOINT,
                    choices=sorted(pesm.CHECKPOINTS))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify-index", action="store_true",
                    help="only check that the 34 contact positions reproduce "
                         "hla_pseudoseq, then exit")
    args = ap.parse_args()

    df = load_raw()
    seqs = unique_sequences(df)

    # --- indexing verification, always run before any embedding -------------
    pairs = df[["hla_seq", "hla_pseudoseq"]].drop_duplicates()
    report = pesm.verify_contact_index(pairs.hla_seq.astype(str).tolist(),
                                       pairs.hla_pseudoseq.astype(str).tolist())
    print(f"contact-position check over {report['n_alleles']} unique HLA domains: "
          f"exact match = {report['exact_match']} "
          f"({report['n_mismatched_alleles']} mismatched alleles)")
    for a in report["ambiguous"]:
        print(f"  pseudo position {a['pseudo_position']:>2} -> hla_seq column "
              f"{a['assigned']:>3} ({a['residue']}); "
              f"{len(a['indistinguishable_columns'])} columns are "
              f"indistinguishable from sequence alone: "
              f"{a['indistinguishable_columns']}")
    if not report["exact_match"]:
        raise SystemExit("contact index does not reproduce hla_pseudoseq; "
                         "fix the index before embedding")
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "stage3_contact_index.json").write_text(
        json.dumps(report, indent=2) + "\n")
    if args.verify_index:
        return 0

    rows = []
    started = time.perf_counter()
    for kind in ("peptide", "hla"):
        s = seqs[kind]
        print(f"\n{kind}: {len(s):,} unique sequences of length {len(s[0])} "
              f"(from {len(df):,} rows)")

        def progress(done, total, kind=kind):
            print(f"\r  {kind}: {done:,}/{total:,}", end="", flush=True)

        res = pesm.build_cache(s, checkpoint=args.checkpoint, kind=kind,
                               device=args.device, force=args.force,
                               progress=progress)
        print(f"\r  device={res.device} load={res.load_seconds:.1f}s "
              f"embed={res.embed_seconds:.1f}s cache={res.cache_mb:.1f} MB "
              f"peak RSS={res.peak_rss_mb:.0f} MB      ")
        rows.append({
            "checkpoint": res.checkpoint, "kind": res.kind,
            "n_unique_sequences": res.n_sequences, "n_rows_covered": len(df),
            "seq_length": res.seq_length, "dim": res.dim, "device": res.device,
            "load_seconds": round(res.load_seconds, 2),
            "embed_seconds": round(res.embed_seconds, 2),
            "seconds_per_1k_sequences": round(
                1000 * res.embed_seconds / max(res.n_sequences, 1), 2),
            "peak_rss_mb": round(res.peak_rss_mb, 1),
            "cache_mb": round(res.cache_mb, 1),
            "content_hash": res.content_hash,
        })
    wall = time.perf_counter() - started

    cost = pd.DataFrame(rows)
    dest = REPORT_DIR / "stage3_embedding_cost.csv"
    if dest.exists():
        old = pd.read_csv(dest)
        old = old[~old.checkpoint.isin(cost.checkpoint.unique())
                  | ~old.kind.isin(cost.kind.unique())]
        cost = pd.concat([old, cost], ignore_index=True)
    cost.to_csv(dest, index=False)
    print(f"\ntotal wall {wall:.1f}s; wrote {dest.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
