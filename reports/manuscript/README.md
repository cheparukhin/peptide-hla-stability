# Manuscript figure sources

The [manuscript](../MANUSCRIPT.md) is the primary scientific account, and [supplementary results](../REPORT.md) provide additional numerical detail. This directory contains figure source data, rendering code, and export files.

## Figure provenance

| Figure | Saved evidence | Inspectable values |
|---|---|---|
| 1: main and transfer comparisons | [Main contrasts](../further_research/round2/results/main/paired_comparisons.csv), [transfer contrasts](../finalization/transfer_cloud_recovery/results/comparisons.csv) | [Exact selected rows and source hashes](figure_data.csv) |
| 2: representation controls | [Main contrasts](../further_research/round2/results/main/paired_comparisons.csv) | [Exact selected rows and source hashes](figure_data.csv) |
| 3: retrospective adaptation | [Canonical figure record](../followup3/figures/fewshot_curves.json), [absolute curves](../followup3/fewshot/absolute_curves.csv), [three-primary interval](../followup3/fewshot/primary_family3_interval.json) | [Curve values, interval and source hash](figure3_data.json) |

All estimates and bounds are copied directly from saved artifacts. The renderer verifies unique comparison rows, score-difference arithmetic, and adaptation source hashes; it does not refit models, regenerate bootstrap draws, or acquire measurements. Contrasts use unrounded source values, while displayed values are rounded to four decimal places. The 98.3333% three-primary interval supersedes the older 97.5% fields retained in the original few-shot comparison table.

Figures are supplied as SVG and PDF for vector export and as PNG for preview. With Python 3.11+ and `matplotlib==3.10.8` available, run from the repository root:

```bash
python3 reports/manuscript/build_figures.py
```

This lightweight rendering step leaves the statistical and computational scope of the underlying studies unchanged.

## Publication preparation

Before external submission, the authors must finalize authorship, affiliations, funding and conflict statements, data-use and licensing statements, repository release details, and venue formatting. The extraction-wrapper provenance limitation is disclosed in the manuscript's availability statement and the linked cache audit. Missing declarations must not be filled with invented assertions.
