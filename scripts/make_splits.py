"""Build the frozen 70/10/20 splits. Run once; the output is committed.

    python scripts/make_splits.py

Writes ``data/splits.csv`` and prints the checks that justify it. Re-running
with the same read-only input reproduces the same file byte for byte -- the
clustering and the water-fill are both deterministic. Do not call this from
training code: load the committed result with
``pepstab.data.load_with_splits()``.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from pepstab.data import SPLITS_CSV, load_raw  # noqa: E402
from pepstab.evaluation import MIN_ROWS_PER_ALLELE  # noqa: E402
from pepstab.splits import (  # noqa: E402
    HAMMING_THRESHOLD,
    SPLIT_FRACTIONS,
    SPLIT_NAMES,
    build_splits,
    min_cross_split_distance,
)


def main() -> int:
    df = load_raw()
    print(f"loaded {len(df):,} pairs, {df.peptide.nunique():,} peptides, "
          f"{df.allele.nunique()} alleles")

    out = build_splits(df)
    n_clusters = out.cluster_id.nunique()
    print(f"single-linkage at Hamming <= {HAMMING_THRESHOLD}: {n_clusters:,} clusters")

    sizes = out.groupby("cluster_id").peptide.nunique()
    weights = out.groupby("cluster_id").size()
    print(f"  largest cluster: {sizes.max()} peptides, {weights.max()} rows "
          f"({weights.max() / len(out):.2%} of all pairs)")
    print(f"  singleton clusters: {(sizes == 1).sum():,}")

    print("\nsplit sizes (target 70/10/20 by row count):")
    for name, frac in zip(SPLIT_NAMES, SPLIT_FRACTIONS):
        n = int((out.split == name).sum())
        print(f"  {name:5s} {n:6,d} rows  {n / len(out):6.2%}  (target {frac:.0%})"
              f"  {int((out.split == name).groupby(out.cluster_id).any().sum()):,} clusters"
              f"  {out.loc[out.split == name, 'peptide'].nunique():,} peptides")

    print("\nleakage check -- minimum peptide Hamming distance between splits:")
    dist = min_cross_split_distance(out)
    ok = True
    for (a, b), d in sorted(dist.items()):
        flag = "OK" if d > HAMMING_THRESHOLD else "FAIL"
        ok &= d > HAMMING_THRESHOLD
        print(f"  {a:5s} vs {b:5s}: {d}  [{flag}]")
    print(f"  guarantee: no held-out peptide within {HAMMING_THRESHOLD} "
          f"substitutions of any training peptide -- {'holds' if ok else 'VIOLATED'}")
    if not ok:
        print("refusing to write splits.csv", file=sys.stderr)
        return 1

    counts = out.pivot_table(index="allele", columns="split", aggfunc="size",
                             fill_value=0)
    counts = counts.reindex(columns=list(SPLIT_NAMES), fill_value=0)
    counts["total"] = counts.sum(axis=1)
    print(f"\nper-allele coverage ({len(counts)} alleles):")
    print(f"  alleles present in all three splits: "
          f"{int((counts[list(SPLIT_NAMES)] > 0).all(axis=1).sum())}")
    print(f"  alleles with >= {MIN_ROWS_PER_ALLELE} test rows (evaluation-eligible): "
          f"{int((counts['test'] >= MIN_ROWS_PER_ALLELE).sum())}")
    print(f"  alleles with >= 100 test rows: {int((counts['test'] >= 100).sum())}")
    print("  alleles missing from at least one split:")
    thin = counts[(counts[list(SPLIT_NAMES)] == 0).any(axis=1)].sort_values("total")
    print("    " + thin.to_string().replace("\n", "\n    "))

    print("\ndistance from each held-out peptide to its nearest training peptide")
    print("  (frozen into splits.csv so stage 6 can stratify without recomputing)")
    for name in ("val", "test"):
        s = out[out.split == name]
        by_row = s.dist_to_train.value_counts().sort_index()
        n_pep = s.groupby("dist_to_train").peptide.nunique()
        print(f"  {name}:")
        for d in by_row.index:
            print(f"    d={d}: {n_pep[d]:5,d} peptides, {by_row[d]:5,d} rows "
                  f"({by_row[d] / len(s):6.2%} of split)")

    out = out.sort_values("pair_id")
    out[["pair_id", "allele", "peptide", "cluster_id", "split",
         "dist_to_train"]].to_csv(SPLITS_CSV, index=False)
    print(f"\nwrote {SPLITS_CSV.relative_to(SPLITS_CSV.parent.parent)} "
          f"({len(out):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
