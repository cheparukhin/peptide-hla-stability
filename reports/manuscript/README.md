# Manuscript figure sources

The [manuscript](../MANUSCRIPT.md) contains the main text, and [supplementary results](../REPORT.md) provide additional numerical detail. This directory contains figure source data, rendering code, and export files.

## Figure provenance

| Figure | Saved evidence | Inspectable values |
|---|---|---|
| 1: main and transfer comparisons | [Main contrasts](../further_research/round2/results/main/paired_comparisons.csv), [transfer contrasts](../finalization/transfer_cloud_recovery/results/comparisons.csv) | [Exact selected rows and source hashes](figure_data.csv) |
| 2: allele-level variation | [Per-allele scores](../further_research/round2/results/main/per_allele_scores.csv) | [Paired scores, counts and source hash](figure2_data.json) |
| 3: representation controls | [Main contrasts](../further_research/round2/results/main/paired_comparisons.csv) | [Exact selected rows and source hashes](figure_data.csv) |
| 4: retrospective adaptation | [Canonical figure record](../followup3/figures/fewshot_curves.json), [absolute curves](../followup3/fewshot/absolute_curves.csv), [three-primary interval](../followup3/fewshot/primary_family3_interval.json) | [Curve values, interval and source hash](figure4_data.json) |

Figures 1, 3, and 4 copy estimates and bounds directly from saved artifacts, while Figure 2 reads saved per-allele scores and derives simple paired differences and counts. The renderer verifies unique rows, matching allele identities and sample counts (69 alleles across 19,647 rows), panel medians against the saved primary comparison, score-difference arithmetic, and adaptation input hashes, and it records source hashes for all exported records; it does not refit models, regenerate bootstrap draws, or acquire measurements. Contrasts use unrounded source values, while displayed values are rounded to four decimal places. The 98.3333% three-primary interval supersedes the older 97.5% fields retained in the original few-shot comparison table.

Figures are supplied as SVG and PDF for vector export and as PNG for preview. With Python 3.11+ and `matplotlib==3.10.8` available, run from the repository root:

```bash
python3 reports/manuscript/build_figures.py
```

## Publication preparation

Before external submission, the authors must finalize authorship, affiliations, funding and conflict statements, data-use and licensing statements, repository release details, and venue formatting. The extraction-wrapper provenance limitation is noted in the manuscript's availability statement and the linked cache audit. Do not fill missing declarations with invented details.
