# Scientific results and supporting evidence

The [scientific report](REPORT.md) is the complete account of the results.
[Methods](../EVALUATION.md) defines the assessment and uncertainty, and
[Data](../docs/DATASETS.md) describes the assay and populations.
The [submission abstract](finalization/SUBMISSION_BRIEF.md) is a concise summary.

## Result tables

Each assessment has its own population, comparator and fitting procedure.
Scores from different rows of this index should not be treated as matched
comparisons. Column names retain the representation identifiers defined in
Methods.

| Assessment | Numerical results | Saved predictions or uncertainty |
|---|---|---|
| Main peptide-cluster assessment | [Scores](further_research/round2/results/main/pooled_scores.csv), [comparisons](further_research/round2/results/main/paired_comparisons.csv) | [Predictions](further_research/round2/results/main/matched_predictions.csv), [bootstrap scores](further_research/round2/results/main/bootstrap_scores_min20.csv) |
| Sequence and embedding complementarity | [Comparisons](further_research/round2/complementarity/results/paired_comparisons.csv) | [Assessment manifest](further_research/round2/complementarity/results/manifest.json) |
| HLA and joint transfer | [Comparisons](finalization/transfer_cloud_recovery/results/comparisons.csv), [group-deletion sensitivities](finalization/transfer_cloud_recovery/results/all_group_deletions.csv) | [HLA predictions](finalization/transfer_cloud_recovery/results/hla_pooled_predictions.csv), [joint predictions](finalization/transfer_cloud_recovery/results/joint_pooled_predictions.csv) |
| Ranking objective | [Comparisons](finalization/ranking/comparisons.csv) | [Completion audit](finalization/ranking/completion_audit.json) |
| Trees, blending and preprocessing | [Comparisons](finalization/hybrids_cloud_v1/comparisons.csv) | [Coverage audit](finalization/hybrids_cloud_v1/coverage_audit.json) |
| Training-label budget | [Absolute scores](finalization/label_efficiency/absolute_curves.csv), [comparisons](finalization/label_efficiency/comparisons.csv) | [Actual label budgets](finalization/label_efficiency/actual_budgets.csv) |
| Few-shot HLA adaptation | [Absolute scores](followup3/fewshot/absolute_curves.csv), [comparisons](followup3/fewshot/comparisons.csv) | [Three-primary interval](followup3/fewshot/primary_family3_interval.json), [bootstrap scores](followup3/fewshot/bootstrap_crossed_min20.csv) |
| ProteinMPNN on the balanced panel | [Scores](followup3/inverse_execution_v2/pooled_scores.csv), [comparisons](followup3/inverse_execution_v2/comparisons.csv) | [Three-primary interval](followup3/inverse_execution_v2/primary_family3_interval.json) |
| Repaired energy descriptors | [Scores](followup3/energy_execution_v2/pooled_scores.csv), [comparisons](followup3/energy_execution_v2/comparisons.csv) | [Three-primary interval](followup3/energy_execution_v2/primary_family3_interval.json) |
| ProteinMPNN on the full training cohort | [Scores](followup3/inverse_fulltrain_candidate/scores.csv), [comparisons](followup3/inverse_fulltrain_candidate/comparisons.csv) | [Fixed-query sensitivity](followup3/inverse_fulltrain_candidate/fixed_query_descriptive.csv) |
| Original geometry and language-model assessment | [Scores](stage6_test_summary.csv), [comparisons](stage6_test_paired_ci.csv) | [Scoring provenance](test_scoring_provenance.json) |

The separate three-primary interval files supersede the two-primary interval
columns retained in the original comparison tables. The source tables remain
unchanged; Methods defines which interval supports each current claim.

## Reproducibility sources

CSV and JSON artifacts retain predictions, decisions, fitting ancestry and
numerical evidence. The [protocol source bundle](protocol_sources.zip) preserves
original documents needed to interpret or replay hash-bound experiment records;
its [manifest](protocol_sources.json) records the source commit and each member's
SHA-256. These are immutable experimental sources. Use the current report and
Methods for the scientific interpretation.

[Development instructions](../docs/DEVELOPMENT.md) describe the code and source
versions. [Independent validation](finalization/INDEPENDENT_ASSAY_BRIEF.md)
specifies the next experiment needed to test assay generalisation.
