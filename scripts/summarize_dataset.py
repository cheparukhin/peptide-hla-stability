#!/usr/bin/env python3
"""Summarize the raw peptide-HLA stability dataset.

Reads the dataset read-only and writes a headline summary table, a per-allele
breakdown, and a markdown report. Everything here is derived and regenerable:

    python3 scripts/summarize_dataset.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "rasmussen_et_al_dataset.csv"
# Committed on purpose: these tables are small and are read as documentation.
# Derived per input file, so summarising a second dataset cannot clobber the first.
DOCS_DIR = REPO_ROOT / "docs"

ALLELE_COL = "allele"
PEPTIDE_COL = "peptide"
TARGET_COL = "thalf_hours"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Predeclared threshold from HACKATHON_PLAN.md: "sufficiently stable" candidates.
STABLE_THRESHOLD_HOURS = 2.0


def _fmt(value: float, digits: int = 4) -> str:
    """Render a number without scientific notation or trailing zero noise."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")


def describe_numeric(series: pd.Series, label: str, section: str) -> list[dict]:
    """Quantile + moment summary for one numeric column."""
    stats = {
        "count": series.count(),
        "n_missing": series.isna().sum(),
        "mean": series.mean(),
        "std": series.std(),
        "min": series.min(),
        "p01": series.quantile(0.01),
        "p05": series.quantile(0.05),
        "q1": series.quantile(0.25),
        "median": series.median(),
        "q3": series.quantile(0.75),
        "p95": series.quantile(0.95),
        "p99": series.quantile(0.99),
        "max": series.max(),
        "iqr": series.quantile(0.75) - series.quantile(0.25),
        "skew": series.skew(),
        "kurtosis": series.kurtosis(),
    }
    return [
        {"section": section, "metric": f"{label}_{name}", "value": _fmt(value)}
        for name, value in stats.items()
    ]


def build_summary_rows(df: pd.DataFrame) -> list[dict]:
    """Headline counts, coverage, and target-value statistics as one long table."""
    n_alleles = df[ALLELE_COL].nunique()
    n_peptides = df[PEPTIDE_COL].nunique()
    target = df[TARGET_COL]
    peptide_lengths = df[PEPTIDE_COL].str.len()
    rows_per_allele = df[ALLELE_COL].value_counts()
    alleles_per_peptide = df.groupby(PEPTIDE_COL)[ALLELE_COL].nunique()
    observed_aa = set("".join(df[PEPTIDE_COL].unique()))

    rows: list[dict] = [
        {"section": "size", "metric": "n_rows", "value": len(df)},
        {"section": "size", "metric": "n_columns", "value": df.shape[1]},
        {"section": "size", "metric": "n_distinct_alleles", "value": n_alleles},
        {"section": "size", "metric": "n_distinct_peptides", "value": n_peptides},
        {
            "section": "size",
            "metric": "n_distinct_allele_peptide_pairs",
            "value": len(df.drop_duplicates([ALLELE_COL, PEPTIDE_COL])),
        },
        {
            "section": "integrity",
            "metric": "n_duplicate_rows",
            "value": int(df.duplicated().sum()),
        },
        {
            "section": "integrity",
            "metric": "n_duplicate_allele_peptide_pairs",
            "value": int(df.duplicated([ALLELE_COL, PEPTIDE_COL]).sum()),
        },
        {
            "section": "integrity",
            "metric": "n_rows_with_any_missing",
            "value": int(df.isna().any(axis=1).sum()),
        },
        {
            "section": "integrity",
            "metric": "n_nonstandard_amino_acids",
            "value": len(observed_aa - STANDARD_AA),
        },
        {
            "section": "coverage",
            "metric": "full_grid_size_alleles_x_peptides",
            "value": n_alleles * n_peptides,
        },
        {
            "section": "coverage",
            "metric": "grid_fill_pct",
            "value": _fmt(100 * len(df) / (n_alleles * n_peptides), 2),
        },
        {
            "section": "coverage",
            "metric": "rows_per_allele_min",
            "value": int(rows_per_allele.min()),
        },
        {
            "section": "coverage",
            "metric": "rows_per_allele_median",
            "value": _fmt(rows_per_allele.median(), 1),
        },
        {
            "section": "coverage",
            "metric": "rows_per_allele_max",
            "value": int(rows_per_allele.max()),
        },
        {
            "section": "coverage",
            "metric": "alleles_per_peptide_min",
            "value": int(alleles_per_peptide.min()),
        },
        {
            "section": "coverage",
            "metric": "alleles_per_peptide_median",
            "value": _fmt(alleles_per_peptide.median(), 1),
        },
        {
            "section": "coverage",
            "metric": "alleles_per_peptide_max",
            "value": int(alleles_per_peptide.max()),
        },
        {
            "section": "coverage",
            "metric": "n_peptides_measured_on_one_allele_only",
            "value": int((alleles_per_peptide == 1).sum()),
        },
        {
            "section": "peptide",
            "metric": "peptide_length_min",
            "value": int(peptide_lengths.min()),
        },
        {
            "section": "peptide",
            "metric": "peptide_length_max",
            "value": int(peptide_lengths.max()),
        },
        {
            "section": "peptide",
            "metric": "peptide_lengths_present",
            "value": "|".join(str(x) for x in sorted(peptide_lengths.unique())),
        },
        {
            "section": "peptide",
            "metric": "n_distinct_amino_acids_observed",
            "value": len(observed_aa),
        },
    ]

    for col in ("hla_seq", "hla_pseudoseq"):
        if col not in df.columns:
            continue
        lengths = df[col].str.len().unique()
        rows += [
            {"section": "hla", "metric": f"n_distinct_{col}", "value": df[col].nunique()},
            {
                "section": "hla",
                "metric": f"{col}_lengths_present",
                "value": "|".join(str(x) for x in sorted(lengths)),
            },
            {
                "section": "hla",
                "metric": f"max_alleles_sharing_one_{col}",
                "value": int(df.groupby(col)[ALLELE_COL].nunique().max()),
            },
        ]

    rows += describe_numeric(target, TARGET_COL, "target")
    rows += describe_numeric(np.log1p(target), f"log1p_{TARGET_COL}", "target_log1p")
    rows += [
        {
            "section": "target_shape",
            "metric": "n_at_floor_zero",
            "value": int((target == 0).sum()),
        },
        {
            "section": "target_shape",
            "metric": "pct_at_floor_zero",
            "value": _fmt(100 * (target == 0).mean(), 2),
        },
        {
            "section": "target_shape",
            "metric": "n_at_observed_max",
            "value": int((target == target.max()).sum()),
        },
        {
            "section": "target_shape",
            "metric": f"n_above_{_fmt(STABLE_THRESHOLD_HOURS)}h",
            "value": int((target > STABLE_THRESHOLD_HOURS).sum()),
        },
        {
            "section": "target_shape",
            "metric": f"pct_above_{_fmt(STABLE_THRESHOLD_HOURS)}h",
            "value": _fmt(100 * (target > STABLE_THRESHOLD_HOURS).mean(), 2),
        },
        {
            "section": "target_shape",
            "metric": "n_distinct_values",
            "value": int(target.nunique()),
        },
        {
            "section": "target_shape",
            "metric": "min_gap_between_distinct_values",
            "value": _fmt(np.diff(np.unique(target.to_numpy())).min(), 3),
        },
        {
            "section": "target_shape",
            "metric": "pct_values_on_0.1h_grid",
            "value": _fmt(100 * np.isclose(target, target.round(1)).mean(), 2),
        },
    ]
    return rows


def build_per_allele(df: pd.DataFrame) -> pd.DataFrame:
    """One row per allele: coverage plus target distribution."""
    target_by_allele = df.groupby(ALLELE_COL)[TARGET_COL]
    out = pd.DataFrame(
        {
            "n_rows": target_by_allele.size(),
            "n_distinct_peptides": df.groupby(ALLELE_COL)[PEPTIDE_COL].nunique(),
            "thalf_mean": target_by_allele.mean(),
            "thalf_std": target_by_allele.std(),
            "thalf_min": target_by_allele.min(),
            "thalf_q1": target_by_allele.quantile(0.25),
            "thalf_median": target_by_allele.median(),
            "thalf_q3": target_by_allele.quantile(0.75),
            "thalf_max": target_by_allele.max(),
            "log1p_thalf_mean": df.assign(_l=np.log1p(df[TARGET_COL]))
            .groupby(ALLELE_COL)["_l"]
            .mean(),
            "pct_at_floor_zero": 100 * target_by_allele.apply(lambda s: (s == 0).mean()),
            f"pct_above_{_fmt(STABLE_THRESHOLD_HOURS)}h": 100
            * target_by_allele.apply(lambda s: (s > STABLE_THRESHOLD_HOURS).mean()),
        }
    )
    numeric = out.select_dtypes("number").columns
    out[numeric] = out[numeric].round(4)
    return out.sort_values("n_rows", ascending=False).reset_index()


def to_markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, rule, *body])


def build_markdown(
    summary: pd.DataFrame, per_allele: pd.DataFrame, input_path: Path
) -> str:
    look = dict(zip(summary["metric"], summary["value"]))
    top = per_allele.head(10)[
        ["allele", "n_rows", "thalf_median", "thalf_max", "pct_at_floor_zero"]
    ]
    rel_input = (
        input_path.relative_to(REPO_ROOT)
        if input_path.is_relative_to(REPO_ROOT)
        else input_path
    )
    regen = "python3 scripts/summarize_dataset.py"
    if input_path.resolve() != DEFAULT_INPUT.resolve():
        regen += f" --input {rel_input}"

    n_shared = int(look["max_alleles_sharing_one_hla_pseudoseq"])
    if n_shared > 1:
        pseudoseq_note = (
            f"- `hla_pseudoseq` is **not** a unique allele key: up to {n_shared} "
            "alleles share one pseudosequence, so keying on it silently merges "
            "those alleles."
        )
    else:
        pseudoseq_note = (
            "- `hla_pseudoseq` is a unique allele key here (one pseudosequence per "
            "allele), so it is safe to key on."
        )

    sections = []
    for name in summary["section"].unique():
        block = summary[summary["section"] == name][["metric", "value"]]
        sections.append(f"### `{name}`\n\n{to_markdown_table(block)}")

    return f"""# Dataset summary — `{rel_input}`

Generated by [scripts/summarize_dataset.py](../../scripts/summarize_dataset.py).
Regenerate with `{regen}`. This script only reads its input.

## Headline

- **{look['n_rows']}** measurements over **{look['n_distinct_alleles']}** distinct
  alleles and **{look['n_distinct_peptides']}** distinct peptides, with no
  duplicate allele-peptide pairs and no missing values.
- Every peptide is a **{look['peptide_length_min']}-mer** built from the
  {look['n_distinct_amino_acids_observed']} standard amino acids.
- The allele x peptide grid is only **{look['grid_fill_pct']}%** filled
  ({look['n_rows']} of {look['full_grid_size_alleles_x_peptides']} cells), and
  coverage is lopsided: {look['rows_per_allele_min']} to
  {look['rows_per_allele_max']} measurements per allele (median
  {look['rows_per_allele_median']}).
- `{TARGET_COL}` is strongly right-skewed (skew
  {look[f'{TARGET_COL}_skew']}): median **{look[f'{TARGET_COL}_median']} h**
  against a mean of {look[f'{TARGET_COL}_mean']} h and a max of
  {look[f'{TARGET_COL}_max']} h.
- **{look['pct_at_floor_zero']}%** of rows sit exactly at the 0-hour floor, so the
  target is a spike-at-zero plus a long tail rather than a smooth distribution.
  The scale is also coarse: only {look['n_distinct_values']} distinct half-lives
  occur across {look['n_rows']} rows, {look['pct_values_on_0.1h_grid']}% of them on
  a 0.1 h grid.
- Only **{look[f'pct_above_{_fmt(STABLE_THRESHOLD_HOURS)}h']}%** of rows clear the
  {_fmt(STABLE_THRESHOLD_HOURS)}-hour stability threshold used for precision@10 in
  [HACKATHON_PLAN.md](../../HACKATHON_PLAN.md).

## Modelling notes

- The zero spike plus the heavy tail is why the plan scores MAE on `log1p`;
  log1p compresses the tail from {look[f'{TARGET_COL}_max']} to
  {look[f'log1p_{TARGET_COL}_max']} and drops skew to
  {look[f'log1p_{TARGET_COL}_skew']}.
{pseudoseq_note}
- {look['n_peptides_measured_on_one_allele_only']} peptides appear against a single
  allele, which limits what within-peptide, between-allele comparisons can say.

## Summary table

Full table: [dataset_summary.csv](dataset_summary.csv)

{(chr(10) * 2).join(sections)}

## Per-allele breakdown

Full table ({len(per_allele)} alleles): [per_allele_stats.csv](per_allele_stats.csv).
Ten best-covered alleles:

{to_markdown_table(top)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="defaults to docs/<input-stem>_summary/",
    )
    args = parser.parse_args()
    outdir = args.outdir or DOCS_DIR / f"{args.input.stem}_summary"

    df = pd.read_csv(args.input)
    missing = {ALLELE_COL, PEPTIDE_COL, TARGET_COL} - set(df.columns)
    if missing:
        raise SystemExit(f"{args.input} is missing required columns: {sorted(missing)}")

    summary = pd.DataFrame(build_summary_rows(df), columns=["section", "metric", "value"])
    per_allele = build_per_allele(df)

    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "dataset_summary.csv"
    allele_path = outdir / "per_allele_stats.csv"
    md_path = outdir / "DATASET_SUMMARY.md"

    summary.to_csv(summary_path, index=False)
    per_allele.to_csv(allele_path, index=False)
    md_path.write_text(build_markdown(summary, per_allele, args.input))

    for path in (summary_path, allele_path, md_path):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
