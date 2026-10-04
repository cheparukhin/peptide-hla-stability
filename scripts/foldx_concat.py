#!/usr/bin/env python
"""Stage 8: concatenate the two workspaces' FoldX halves into one table.

**Free — no Modal import, no network.** It reads the two per-profile CSVs that
`modal_app/foldx_scoring.py::score` writes and assembles the cohort table the
evaluation arm consumes.

The asserts are the point. Each workspace scores its own disjoint half of the
frozen cohort, and a half-sized output is not obvious from the file name: it is
a well-formed CSV with plausible numbers in every column. So this refuses to
write unless

* the union is exactly 28,166 rows,
* no ``(allele, peptide)`` pair appears twice — the halves are disjoint, and an
  overlap means one profile scored the other's half,
* every row matches an ``(allele, peptide)`` in ``data/structural_cohort.csv``
  and every cohort pair is present,
* both halves agree on the FoldX binary SHA-256 and on ``foldx_repair``.

That last one is the silent confound ``reports/stage8_foldx.md`` §4 forbids:
repaired and unrepaired rows mixed in one feature column would be undetectable
downstream, so it is caught here instead.

Splits are joined from ``data/splits.csv`` on ``(allele, peptide)``, never
positionally on ``pair_id`` — the project invariant. No label column is read:
this script handles features only, and the test split's labels are scored once,
at stage 6, by the orchestrator.

    .venv/bin/python scripts/foldx_concat.py
    .venv/bin/python scripts/foldx_concat.py --repair   # the repaired variant
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

COHORT_CSV = REPO / "data" / "structural_cohort.csv"
SPLITS_CSV = REPO / "data" / "splits.csv"
COHORT_TOTAL = 28166
PROFILES = ("a-cheparukhin", "colleague")

#: Recorded in the Results section per §6 of the protocol, because each one
#: carries a systematically different energy decomposition and saying so after
#: seeing the scores would be an excuse rather than a caveat.
#: Exactly as spelled in data/structural_cohort.csv -- the cohort carries the
#: "(C67S)" suffix, and an `isin` against the bare allele name would silently
#: match nothing and report the engineered constructs as absent.
C67S_ALLELES = ("HLA-B*14:01(C67S)", "HLA-B*14:02(C67S)", "HLA-B*39:06(C67S)")
BORROWED_ALPHA3 = ("HLA-A*02:50", "HLA-A*24:19", "HLA-B*08:03")


def half_path(profile: str, repair: bool) -> Path:
    stem = "stage8_foldx_scores_repair" if repair else "stage8_foldx_scores"
    return REPO / "reports" / f"{stem}_{profile}.csv"


def load_halves(repair: bool) -> pd.DataFrame:
    frames = []
    for profile in PROFILES:
        path = half_path(profile, repair)
        if not path.exists():
            raise SystemExit(
                f"{path.relative_to(REPO)} is missing. Each workspace writes its "
                f"own half; run ::score under MODAL_PROFILE={profile} before "
                "concatenating."
            )
        frame = pd.read_csv(path)
        if "profile" not in frame:
            frame["profile"] = profile
        got = sorted(frame["profile"].dropna().unique())
        if got != [profile]:
            raise SystemExit(
                f"{path.relative_to(REPO)} carries profile(s) {got} but is named "
                f"for {profile!r}. --profile must equal MODAL_PROFILE; one of "
                "these runs scored the wrong half."
            )
        print(f"  {path.relative_to(REPO)}  {len(frame):,} rows")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def check_disjoint_and_complete(table: pd.DataFrame) -> pd.DataFrame:
    cohort = pd.read_csv(COHORT_CSV)
    assert len(cohort) == COHORT_TOTAL, f"cohort is {len(cohort)}, expected {COHORT_TOTAL}"

    dupes = table.duplicated(subset=["allele", "peptide"], keep=False)
    if dupes.any():
        sample = table.loc[dupes, ["allele", "peptide", "profile"]].head(5)
        raise SystemExit(
            f"{int(dupes.sum())} rows share an (allele, peptide) across the two "
            f"halves, which are supposed to be disjoint:\n{sample}"
        )
    if len(table) != COHORT_TOTAL:
        raise SystemExit(
            f"{len(table):,} rows, expected {COHORT_TOTAL:,}. This is one half, or "
            "a partial run — it is a well-formed CSV either way, which is why "
            "this assert exists."
        )

    keys = set(map(tuple, table[["allele", "peptide"]].to_numpy()))
    want = set(map(tuple, cohort[["allele", "peptide"]].to_numpy()))
    if keys != want:
        missing, extra = want - keys, keys - want
        raise SystemExit(
            f"the scored table does not cover the frozen cohort: {len(missing)} "
            f"cohort pairs missing, {len(extra)} pairs not in the cohort. "
            f"Examples missing: {sorted(missing)[:3]}"
        )
    return cohort


def check_one_provenance(table: pd.DataFrame) -> dict:
    """One binary, one repair setting, across both halves."""
    out = {}
    for column, label in (("foldx_binary_sha256", "FoldX binary"),
                          ("foldx_repair", "repair setting")):
        if column not in table:
            raise SystemExit(f"no {column!r} column; provenance is not optional")
        values = sorted({str(v) for v in table[column].dropna().unique()})
        if len(values) != 1:
            raise SystemExit(
                f"{len(values)} distinct {label} values across the halves: "
                f"{values}. Mixed rows in one feature column are the silent "
                "confound reports/stage8_foldx.md §4 forbids — rescore the odd "
                "half rather than concatenating these."
            )
        out[column] = values[0]
    return out


def attach_splits(table: pd.DataFrame) -> pd.DataFrame:
    """Join the frozen splits on ``(allele, peptide)``. No labels are read."""
    splits = pd.read_csv(SPLITS_CSV, usecols=["pair_id", "allele", "peptide",
                                              "cluster_id", "split"])
    merged = table.merge(splits, on=["allele", "peptide"], how="left",
                         validate="one_to_one")
    if merged["split"].isna().any():
        raise SystemExit(
            f"{int(merged['split'].isna().sum())} scored rows match no "
            "(allele, peptide) in data/splits.csv"
        )
    return merged


def report(table: pd.DataFrame, provenance: dict) -> None:
    ok = table["status"].eq("ok")
    print(f"\nrows                {len(table):,}")
    print(f"scored ok           {int(ok.sum()):,} ({100 * ok.mean():.2f}%)")
    failed = int((~ok).sum())
    print(f"failed              {failed:,}"
          + ("  -> these take the declared sequence fallback" if failed else ""))
    if failed:
        counts = table.loc[~ok, "status"].str.slice(0, 60).value_counts()
        for status, n in counts.head(10).items():
            print(f"    {n:>6,}  {status}")

    print("\nby split (features only; no label was read)")
    for split, group in table.groupby("split"):
        print(f"    {split:<6} {len(group):>7,}   ok {int(group['status'].eq('ok').sum()):>7,}")

    energy = pd.to_numeric(table.get("foldx_interaction_energy"), errors="coerce")
    valid = energy.dropna()
    if len(valid):
        print(f"\nfoldx_interaction_energy over {len(valid):,} scored rows")
        print(f"    median {valid.median():8.3f}   IQR [{valid.quantile(.25):.3f}, "
              f"{valid.quantile(.75):.3f}]   range [{valid.min():.3f}, {valid.max():.3f}]")
        positive = int((valid > 0).sum())
        if positive:
            print(f"    {positive:,} rows are positive — check the group orientation "
                  "before trusting those")

    for label, alleles in (("C67S engineered", C67S_ALLELES),
                           ("borrowed alpha3", BORROWED_ALPHA3)):
        sub = table[table["allele"].isin(alleles)]
        if len(sub):
            e = pd.to_numeric(sub["foldx_interaction_energy"], errors="coerce").dropna()
            print(f"\n{label}: {len(sub):,} rows, {sub['allele'].nunique()} alleles"
                  + (f", median dG {e.median():.3f}" if len(e) else ""))
            for allele, g in sub.groupby("allele"):
                ge = pd.to_numeric(g["foldx_interaction_energy"], errors="coerce").dropna()
                print(f"    {allele:<16} {len(g):>5,} rows"
                      + (f"   median {ge.median():8.3f}" if len(ge) else ""))

    print(f"\nprovenance: binary sha256 {provenance['foldx_binary_sha256'][:16]}..., "
          f"repair={provenance['foldx_repair']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repair", action="store_true",
                    help="concatenate the RepairPDB variant's halves")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print("reading the two halves")
    table = load_halves(args.repair)
    check_disjoint_and_complete(table)
    provenance = check_one_provenance(table)
    table = attach_splits(table)
    report(table, provenance)

    out = args.out or (REPO / "reports" /
                       ("stage8_foldx_features_repair.csv" if args.repair
                        else "stage8_foldx_features.csv"))
    table.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(REPO)}  "
          f"{len(table):,} rows x {len(table.columns)} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
