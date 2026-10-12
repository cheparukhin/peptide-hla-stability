# Historical arithmetic reconciled — 12 October 2026

Current scope remains [NEXT_RESEARCH_PLAN.md](../../../NEXT_RESEARCH_PLAN.md). This report recomputes saved predictions, not model fits. It does not repair their selection history or provide independent confirmation. The frozen source labels/splits and earlier artifacts are unchanged.

All four procedures cover 19,716 original training rows. At the 20-row threshold, 69 alleles are eligible. A0 pooled median per-allele Spearman is **0.661522**; the median of its fold medians is **0.649299**. These are distinct statistics.

| Procedure | Pooled median per-allele Spearman |
|---|---:|
| A0_historical | 0.661522 |
| C_joint_historical | 0.404414 |
| C_separate_historical | 0.385088 |
| C_esm2_historical | 0.465080 |

All six declared contrasts and the 50-row sensitivity are in `historical_pooled_comparisons.csv`, using 2,000 paired whole-peptide-cluster resamples. This implementation was tested against the frozen resampling/ranking procedure, including tied values and constant predictions. Intervals condition on fitted predictions; search and execution variability are separate.

B2 floor Brier changes from 0.109946 to 0.093452, a **15.0% reduction**. Fixed-bin ECE falls 69.6%, a different metric. Its ranking delta is -0.0108 [-0.0242, +0.0036]; the interval does not establish the −0.02 tolerance.

A4 versus its matched control is +0.0295 [+0.0150, +0.0596]. The interval straddles 0.05 and does not account for screening selection. Composition/additivity and a deployable gain remain unestablished.

The old blend has **6/15 zero weights**. Other outer-fold models could have used an assessed fold's labels; those inputs are not clean stacking data. Fold-specific scaling also changes the pooled baseline endpoint. This blend cannot settle complementarity; replacement fitting is required.

A percentile interval missing its point estimate is flagged with the bootstrap mean-minus-point diagnostic, not declared an artefact. Per-allele signed-rank tests are not used as primary evidence because peptides couple the allele measurements. Fold spread and execution variation do not identify an assay ceiling.

## What can be repaired from saved artifacts

- Pooled/fold aggregation, numerical claims, weight counts and conditional cluster intervals: rebuilt here.
- Global configuration selection, selected control ancestry, residual/blend/calibration inputs: require fresh fitting.
- Independent confirmation and assay reliability: require reserved measurements/replicates.

Unsafe legacy report entrypoints now stop with a pointer to the new study. Their importable helpers and historical numeric files remain available for audits. No original test rescore was performed.
