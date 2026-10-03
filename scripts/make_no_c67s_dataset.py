#!/usr/bin/env python3
"""Split the raw dataset into a modelling set and a held-out C67S benchmark.

The three C67S entries are engineered assay constructs carrying serine at
position 67 of the mature heavy chain in place of the cysteine found at that
site in some natural HLA-B allotypes. Position 67 is not part of the structural
C101-C164 disulfide, which is intact in all three. The substitution is visible
in hla_seq, not just in the allele name. They are pulled out of train/eval
because:

  1. they are not natural allotypes, so fitting them teaches the model about an
     assay stabilisation artefact rather than about HLA biology;
  2. none of their wild-type counterparts (HLA-B*14:01, B*14:02, B*39:06) is in
     the dataset, and the nearest natural alleles differ at 3-7 of 182 residues,
     so they cannot serve as a clean mutation-effect test either; and
  3. they barely form stable complexes - 74.9%, 89.0% and 92.1% of their
     measurements are exactly zero - so they would inflate the zero spike
     without being informative.

Excluding them costs no coverage: all 5,633 peptides survive, and the six
natural alleles that retain Cys67 stay in the modelling set. See
docs/DATASETS.md for the full rationale and before/after numbers.

    python3 scripts/make_no_c67s_dataset.py
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "rasmussen_et_al_dataset.csv"
DEFAULT_MAIN_OUT = REPO_ROOT / "data" / "c67s_cleanup" / "rasmussen_no_C67S.csv"
DEFAULT_BENCHMARK_OUT = REPO_ROOT / "data" / "c67s_cleanup" / "benchmark_C67S.csv"

C67S_ALLELES = [
    "HLA-B*14:01(C67S)",
    "HLA-B*14:02(C67S)",
    "HLA-B*39:06(C67S)",
]
EXPECTED_EXCLUDED_ROWS = 1135


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--main-out", type=Path, default=DEFAULT_MAIN_OUT)
    parser.add_argument("--benchmark-out", type=Path, default=DEFAULT_BENCHMARK_OUT)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    present = set(df["allele"])
    missing = [a for a in C67S_ALLELES if a not in present]
    if missing:
        raise SystemExit(f"{args.input} does not contain: {missing}")

    # Guard against a silent miss if the source ever adds another construct.
    unlisted = sorted(a for a in present if "C67S" in a and a not in C67S_ALLELES)
    if unlisted:
        raise SystemExit(f"unlisted C67S constructs found, update C67S_ALLELES: {unlisted}")

    is_c67s = df["allele"].isin(C67S_ALLELES)
    if is_c67s.sum() != EXPECTED_EXCLUDED_ROWS:
        raise SystemExit(
            f"expected {EXPECTED_EXCLUDED_ROWS} C67S rows, found {is_c67s.sum()} — "
            "the source dataset changed, re-check the exclusion before proceeding"
        )

    main_df = df.loc[~is_c67s].reset_index(drop=True)
    bench_df = df.loc[is_c67s].reset_index(drop=True)

    for out_path, out_df in ((args.main_out, main_df), (args.benchmark_out, bench_df)):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)

    pct = 100 * len(bench_df) / len(df)
    print(f"input       {args.input.name}: {len(df)} rows, {df.allele.nunique()} alleles")
    print(
        f"excluded    {len(bench_df)} rows ({pct:.2f}%) across "
        f"{bench_df.allele.nunique()} C67S constructs"
    )
    for allele in C67S_ALLELES:
        vals = bench_df.loc[bench_df.allele == allele, "thalf_hours"]
        print(
            f"            {allele}: n={len(vals)}, "
            f"{100 * (vals == 0).mean():.1f}% zeros, p95={vals.quantile(0.95):.2f} h"
        )
    print(
        f"kept        {len(main_df)} rows, {main_df.allele.nunique()} alleles, "
        f"{main_df.peptide.nunique()} peptides"
    )
    for path in (args.main_out, args.benchmark_out):
        print(f"wrote       {path}  sha256={sha256(path)}")


if __name__ == "__main__":
    main()
