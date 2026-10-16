# Supplementary results

This supplement reports procedure scores, sensitivity checks and historical benchmarks accompanying the [manuscript](MANUSCRIPT.md). The [evaluation specification](../EVALUATION.md) defines endpoints and uncertainty, the [representation guide](../docs/STRUCTURES.md) defines feature families, and the [evidence index](README.md) lists source files.

Unless stated otherwise, scores are median within-allele Spearman correlations. Intervals are pointwise 95% paired estimates conditional on saved fits and selections, unless marked as family-adjusted. Comparisons apply only within their stated population and fitting procedure; scores from different assessments are not matched contrasts.

## Main procedure scores and sensitivities

Alongside the R0–R5 matched-MLP ladder in Table 1 of the [manuscript](MANUSCRIPT.md), the table below reports individual procedures on the 69-allele development panel. These scores describe individual procedures and do not replace the nested selection rule with the best component.

| Main assessment procedure | Median within-allele Spearman |
|---|---:|
| Original sequence MLP ensemble | 0.6615 |
| Per-position interface ridge | 0.6845 |
| Same MLP with shuffled contact membership | 0.7203 |
| Full 9 × 34 positional interface MLP | 0.7315 |

Full outputs are in [all scores](further_research/round2/results/main/pooled_scores.csv) and [paired comparisons](further_research/round2/results/main/paired_comparisons.csv). In the stricter ≥50-row panel (68 alleles), the embedding selector exceeded the sequence selector by +0.0564 [+0.0317, +0.0791]. In the main ≥20-row panel (69 alleles), the competitive sequence selector differed from the original ensemble by +0.0053 [−0.0093, +0.0123].

In complementarity tests, an inner-selected sequence/R4-ridge blend scored 0.7238 and exceeded R4 ridge (per-position interface ridge) alone by +0.0393 [+0.0220, +0.0567]. Residual correction scored 0.7114, exceeding the competitive sequence selector by +0.0446 [+0.0304, +0.0671] ([complementarity comparisons](further_research/round2/complementarity/results/paired_comparisons.csv)).

## Transfer and adaptation sensitivities

The matched nonlinear transfer contrasts (manuscript Figure 1) kept positive lower bounds in the ≥50-row panel and positive point differences after dropping any single HLA group. The original sequence ensemble scored 0.1709 under HLA-group holdout and 0.1350 under joint holdout; these scores are reference baselines rather than the expanded selector used in the selected-procedure comparisons ([all transfer contrasts](finalization/transfer_cloud_recovery/results/comparisons.csv); [group-deletion results](finalization/transfer_cloud_recovery/results/all_group_deletions.csv)).

In few-shot adaptation, the ≥20-row and ≥50-row eligibility panels are identical, so their agreement does not provide an extra robustness check ([absolute curves](followup3/fewshot/absolute_curves.csv)). Exploratory estimates for the fitted-cell offset are in [offset comparisons](followup3/fewshot_offset/comparisons.csv). This offset adds an allele-specific scalar within each separately fitted joint-holdout cell, so pooled evaluation can change ordering across cells without changing ranks within a cell.

## Training-label budgets

The label-budget experiments compare fixed two-member MLP recipes using a sequence baseline weaker than the main competitive selector. Scores are averaged across predefined fitting-cluster subsets, and counts include labels used for both fitting and early stopping. Because this benchmark is retrospective, it does not measure prospective assay savings.

| Fitting-cluster budget | Sequence MLP | Interface MLP | Difference |
|---|---:|---:|---:|
| 25% | 0.2640 | 0.5877 | +0.3237 |
| 50% | 0.4010 | 0.6504 | +0.2494 |
| 100% | 0.5373 | 0.6995 | +0.1622 |

Full series and sample counts are in [absolute curves](finalization/label_efficiency/absolute_curves.csv), [paired contrasts](finalization/label_efficiency/comparisons.csv) and [actual label budgets](finalization/label_efficiency/actual_budgets.csv).

## Descriptor and head comparisons

The balanced descriptor panel contains 4,449 pairs, 2,082 peptide clusters and 75 allele/construct labels, with at most 64 rows per allele and 69 eligible scoring alleles. The table below reports matched two-member MLPs; ridge regressions serve as sensitivity checks.

| Balanced-panel MLP | Sequence features | Interface features |
|---|---:|---:|
| Base representation | 0.2952 | 0.6007 |
| + Peptide-only ProteinMPNN scores | 0.2892 | 0.5956 |
| + Peptide-only and full-complex ProteinMPNN scores | 0.3406 | 0.6195 |
| + Repaired FoldX descriptors | 0.3276 | 0.6157 |

The primary ProteinMPNN comparison adds full-complex ProteinMPNN scores (C) to a model already using base Boltz R5 interface features plus peptide-only ProteinMPNN scores (P), comparing R5+P+C versus R5+P. Adding all ProteinMPNN scores to base Boltz R5 interface features (R5+P+C versus R5) changed the score by +0.0188 (secondary 95% interval [−0.0163, +0.0566]). Ridge and ≥50-row sensitivity checks also showed no established descriptor gain beyond interface features ([ProteinMPNN scores](followup3/inverse_execution_v2/pooled_scores.csv), [contrasts](followup3/inverse_execution_v2/comparisons.csv) and [family interval](followup3/inverse_execution_v2/primary_family3_interval.json); [FoldX scores](followup3/energy_execution_v2/pooled_scores.csv), [contrasts](followup3/energy_execution_v2/comparisons.csv) and [family interval](followup3/energy_execution_v2/primary_family3_interval.json)).

In the full-cohort ProteinMPNN follow-up, the base interface MLP scored 0.6955 and the fully augmented MLP scored 0.6885. The ridge difference was −0.0016 [−0.0151, +0.0073], and the peptide-only, complex-context and ≥50-row sensitivity intervals also spanned zero. This fixed two-member design differs from the main selector, and its intervals do not adjust for selecting the follow-up after seeing the balanced-panel results ([full-cohort contrasts](followup3/inverse_fulltrain_candidate/comparisons.csv)).

In head and hybrid comparisons, ranking-loss selection chose mean squared error in every interface fold ([ranking contrasts](finalization/ranking/comparisons.csv)). The hybrid selector scored 0.7039 versus 0.6995 for the matched standard MLP, choosing the standard MLP in three folds and a 75% MLP/25% regression-tree blend in two. Regression trees, ranking trees, tree-leaf preprocessing and principal components performed worse, while quantile preprocessing remained inconclusive ([hybrid comparisons](finalization/hybrids_cloud_v1/comparisons.csv)).

## Earlier sequence and pretrained-feature comparisons

The original test partition has 5,633 rows, with 5,565 rows across 67 eligible alleles in this scoring panel. These earlier experiments tested frozen ESM-2 features ([Lin et al., 2023](https://pubmed.ncbi.nlm.nih.gov/36927031/)) and sequence features combined with Boltz geometry and confidence summaries. Because this partition was used in earlier diagnostic checks, it is not an untouched confirmation set. Its population and feature definitions also differ from the nested development study, so these scores do not provide a matched comparison of ESM-2 and internal Boltz pair features.

| Original test procedure | Median Spearman | Difference from sequence ensemble, 95% interval |
|---|---:|---:|
| Peptide + HLA pseudosequence ensemble | 0.7064 | Reference |
| Single sequence network | 0.6181 | −0.0883 [−0.1201, −0.0555] |
| Peptide + full HLA-domain ensemble | 0.6904 | −0.0160 [−0.0388, +0.0043] |
| ESM-2 ensemble | 0.6979 | −0.0085 [−0.0291, +0.0233] |
| Sequence + ESM-2 ensemble | 0.6816 | −0.0248 [−0.0460, +0.0073] |
| Sequence + Boltz geometry/confidence | 0.5947 | −0.1117 [−0.1339, −0.0575] |

On this panel, ESM-2 did not establish a ranking gain over the sequence ensemble, and the older geometry-and-confidence pipeline performed worse ([test scores](stage6_test_summary.csv); [paired comparisons](stage6_test_paired_ci.csv)). Because that pipeline also changed sequence encoding, its lower score cannot be attributed solely to the geometry features. Similarly, the difference between the single network and the ensemble combines model averaging with encoding and recipe changes rather than isolating the effect of ensembling.

## Earlier allele-transfer and affinity analyses

An earlier leave-allele-out sequence study across 22,533 combined training and validation rows (68 eligible alleles; six-member heads; no original test evaluation) provides baseline context rather than evidence for a Boltz advantage ([allele-holdout summary](stage7_allele_holdout_headline_seq_pep_pseudo.json)). Median within-allele Spearman was 0.7415 for 25 alleles at nearest training-pseudosequence distance ≤1 versus 0.3390 for 20 alleles at distance ≥4. Unlike joint peptide/HLA holdout, this evaluation had a median held-out-allele peptide-seen share of 1.0 (largely seen-peptide, unseen-allele prediction) and differing assay panel composition across distance strata, precluding a causal distance interpretation.

Across 5,135 dual-measured pairs (33 eligible of 55 alleles), binding-affinity rankings correlated with stability at a median within-allele Spearman of 0.5801 ([affinity association](stage2c_affinity_ceiling.csv)). This correlation is an empirical association, not an accuracy ceiling or proof that auxiliary affinity data cannot help. Weak-binder augmentation ([augmentation deltas](stage2b_deltas.csv)) and auxiliary-affinity training ([auxiliary deltas](stage2c_deltas_ensemble_pep_pseudo.csv)) produced no established ranking gain under the tested data sources, weights and recipes, though this does not rule out gains under other designs.

## Calibration of recorded zero labels

A secondary sequence-model study evaluated probability prediction and calibration for recorded zero half-lives (`thalf_hours == 0`) across five outer peptide-cluster folds of the 19,716 original training rows (30 ensemble members per arm per fold; no original test evaluation). Probability metrics pool all 19,716 out-of-fold rows, whereas ranking uses the median within-allele Spearman correlation across the 69 eligible alleles.

The target event is the recorded zero label itself, not a verified assay detection limit or physiological threshold, so calibrating zero-label probabilities does not calibrate physical half-lives. Arms B1 and B2 optimized a Gaussian censored negative log-likelihood (NLL) assuming left-censoring below 0.1 h as a modelling convention, whereas B0 optimized `log1p` mean squared error (MSE). All three arms mapped ensemble outputs to zero-label probabilities using the same mixture-of-members formula ([pooled scores](further_research/phase_b_pooled.csv); [paired contrasts](further_research/phase_b_paired.csv); [run table](further_research/phase_b_run_table.csv); [historical protocol](https://github.com/cheparukhin/peptide-hla-stability/blob/83d5327ea726812aae592621bcf7643af08e8c8b/reports/further_research/phase_b_declaration.md)).

| Model | Median within-allele Spearman | Pooled zero-label Brier | Fixed-bin ECE |
|---|---:|---:|---:|
| B0: `log1p`-MSE regression, stopped on MSE | 0.6604 | 0.1099 | 0.0643 |
| B1: censored Gaussian NLL, stopped on NLL | 0.6089 | 0.1016 | 0.0454 |
| B2: censored Gaussian NLL, stopped on MSE | 0.6497 | 0.0935 | 0.0196 |

Lower Brier scores indicate better zero-label probability predictions, whereas lower fixed-bin expected calibration error (ECE) indicates closer binned calibration. Compared with B0, B2 reduced pooled Brier error by +0.0165 (paired 95% interval [+0.0151, +0.0179]) and changed median within-allele Spearman by −0.0108 [−0.0242, +0.0036], an interval compatible with a modest ranking loss. B2 also achieved lower pooled Brier error than fold-fitted Platt (0.1035) and isotonic (0.1011) post-hoc calibration, reducing Brier error by 0.0100 [0.0091, 0.0109] and 0.0077 [0.0068, 0.0086].

However, B2 worsened `log1p` mean absolute error from 0.4865 to 0.5442 and censored NLL from 1.0714 to 1.1120, showing a trade-off across metrics rather than a uniform improvement. Holding the censored loss fixed while switching early stopping from NLL (B1) to MSE (B2) raised ranking from 0.6089 to 0.6497 and lowered Brier error from 0.1016 to 0.0935. The effect of the training loss therefore depends on the stopping rule and evaluation metric, and MSE stopping is not necessarily optimal in general. Intervals come from 2,000 paired peptide-cluster resamples conditional on saved fits, without project-wide multiplicity adjustment, and apply only to the sequence models rather than the main Boltz selector.

## Sequence predictions transfer to ligand presentation

A separate benchmark tested the sequence stability ensemble on an MHC Motif Atlas snapshot ([Tadros et al., 2023](https://pubmed.ncbi.nlm.nih.gov/36318236/)) without training on elution labels. The dataset contained 81,600 eluted nine-mers and 816,000 human-proteome decoys (897,600 pairs across 51 alleles). Both classes excluded all 5,633 unique peptide sequences from the full stability dataset (training, validation and test; this peptide count is distinct from the 5,633 pair rows in the original test split). Median per-allele AUROC was 0.9656 and AUPRC was 0.7804 ([results](stage3c_summary.csv); [provenance](stage3c_provenance.json)).

In control checks, median AUROC was 0.9157 against ligands eluted from other alleles and 0.6965 when HLA pseudosequences were permuted. These controls test for generic peptide presentability and allele specificity, but they do not remove all confounding or isolate causal contributions to AUROC. Because this benchmark tests the sequence model on ligand elution, it does not confirm the Boltz stability advantage or measure immunogenicity prediction.

## Compute and inference scope

The recorded compute figures cover different populations, extraction pipelines and accounting scopes, so they do not support matched cost ratios or performance-per-dollar claims. Earlier full coordinate generation across 28,166 pairs completed with zero execution failures and an estimated runtime cost of $212.71 based on job durations and hourly rates, without a final provider invoice ([coordinate verification](ectodomain-20261004/production_verification.json)).

For the 19,716 development pairs in the pair-feature study, trunk-only extraction stopped before the coordinate-diffusion and confidence modules and reused existing MSAs and checkpoints. This shared feature cache was extracted once and reused across downstream heads rather than repeated per arm. It cost $66.84 in runtime accounting (including successful runs, worker execution, serialization and reload overhead, and startup/scale-down allowances) and $65.08 in an October 12 provider-meter snapshot that included pilot runs ([runtime cost snapshot](further_research/round2/resources/runtime_cost_summary.json)). These two figures use different accounting definitions and should not be summed; both exclude storage, egress and new-sequence MSA generation, and neither is a final invoice or total project spend.

| Exported head (CPU, 1 numerical thread, 25 repeats) | Population and scope | Single-pair median (ms) | 64-pair batch median (ms) |
|---|---|---:|---:|
| R4 per-position interface ridge | Warm cached development inputs | 1.56 | 1.65 |
| R5 interface MLP (30 members) | Warm cached development inputs | 2.40 | 3.87 |

The warm CPU head-call timings measure model and batch validation, feature scaling, ensemble member inference and output-table construction on cached development rows (timing only, without accuracy evaluation). These head-call benchmarks exclude cache file reads, process startup, MSA search and Boltz trunk extraction ([inference manifest](finalization/inference/distribution_manifest.json); [inference guide](../docs/ROUND2_INFERENCE.md)). The exported packages run fixed R4 and R5 heads rather than the nested selector and do not inherit its 0.7212 development score. End-to-end latency and cost for uncached new peptide–HLA sequences were not measured.
