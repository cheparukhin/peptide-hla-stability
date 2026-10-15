# Supplementary results

The [manuscript](MANUSCRIPT.md) presents the scientific argument, main results and interpretation, while this supplement provides additional procedure scores, sensitivity checks and historical benchmarks. The [evaluation specification](../EVALUATION.md) defines endpoints and uncertainty, the [representation guide](../docs/STRUCTURES.md) defines feature families, and the [evidence index](README.md) lists source artifacts.

Unless stated otherwise, scores are medians of within-allele Spearman correlations, and intervals are pointwise 95% paired estimates conditional on saved fitted predictions and selections unless a family-adjusted level is identified. Comparisons apply only within their stated population and fitting procedure; scores from different assessments are not matched contrasts.

## Main procedure scores and sensitivities

While Table 1 in the [manuscript](MANUSCRIPT.md) gives the R0–R5 matched-MLP ladder, the component scores below describe additional main-study procedures on the 69-allele development panel. These components describe the individual assessed procedures and do not replace the nested selection rule with the best component.

| Main assessment procedure | Median within-allele Spearman |
|---|---:|
| Original sequence MLP ensemble | 0.6615 |
| Per-position interface ridge | 0.6845 |
| Same MLP with shuffled contact membership | 0.7203 |
| Full 9 × 34 positional interface MLP | 0.7315 |

Full outputs are provided in [all scores](further_research/round2/results/main/pooled_scores.csv) and [paired comparisons](further_research/round2/results/main/paired_comparisons.csv). In the stricter ≥50-row panel of 68 alleles, the embedding-versus-sequence selector difference was +0.0564 [+0.0317, +0.0791]. The competitive sequence selector differed from the original ensemble by +0.0053 [−0.0093, +0.0123].

In complementarity analyses, an inner-selected sequence/R4-ridge blend scored 0.7238 and exceeded R4 ridge (per-position interface ridge) alone by +0.0393 [+0.0220, +0.0567]. Residual correction scored 0.7114, a difference of +0.0446 [+0.0304, +0.0671] from the competitive sequence selector ([complementarity comparisons](further_research/round2/complementarity/results/paired_comparisons.csv)).

## Transfer and adaptation sensitivities

The matched nonlinear transfer contrasts in manuscript Figure 1 retained positive lower bounds in the ≥50-row panel and positive point differences after deleting any single HLA group. The original sequence ensemble scored 0.1709 under HLA-group holdout and 0.1350 under joint holdout; these values serve as additional references rather than the expanded selector used in the selected-procedure comparisons ([all transfer contrasts](finalization/transfer_cloud_recovery/results/comparisons.csv); [group-deletion results](finalization/transfer_cloud_recovery/results/all_group_deletions.csv)).

For few-shot adaptation, the ≥20-row and ≥50-row eligibility panels are identical, so agreement between them is not an additional robustness result ([absolute curves](followup3/fewshot/absolute_curves.csv)). The primary adaptation result appears in the manuscript, while [offset comparisons](followup3/fewshot_offset/comparisons.csv) retain the exploratory estimates for the fitted-cell offset, defined as an allele-specific scalar correction within each separately fitted joint-holdout cell so that pooling can change cross-cell ordering without changing within-cell ranks.

## Training-label budgets

The label-budget experiments evaluate separate fixed two-member MLP recipes using a sequence reference that is weaker than the main competitive selector. The displayed scores average scores across predefined fitting-cluster subset designs, and counts include labels used for both fitting and stopping. Because this evaluation is retrospective, no prospective assay savings were measured.

| Fitting-cluster budget | Sequence MLP | Interface MLP | Difference |
|---|---:|---:|---:|
| 25% | 0.2640 | 0.5877 | +0.3237 |
| 50% | 0.4010 | 0.6504 | +0.2494 |
| 100% | 0.5373 | 0.6995 | +0.1622 |

Full series and sample counts are provided in [absolute curves](finalization/label_efficiency/absolute_curves.csv), [paired contrasts](finalization/label_efficiency/comparisons.csv) and [actual label budgets](finalization/label_efficiency/actual_budgets.csv).

## Descriptor and head comparisons

The balanced descriptor panel contains 4,449 pairs, 2,082 peptide clusters and 75 allele/construct labels, with at most 64 rows per allele and 69 eligible scoring alleles. The table below reports matched two-member MLPs, whereas ridge results serve as sensitivity analyses.

| Balanced-panel MLP | Sequence features | Interface features |
|---|---:|---:|
| Base representation | 0.2952 | 0.6007 |
| + Peptide-only ProteinMPNN scores | 0.2892 | 0.5956 |
| + Peptide-only and full-complex ProteinMPNN scores | 0.3406 | 0.6195 |
| + Repaired FoldX descriptors | 0.3276 | 0.6157 |

The ProteinMPNN net increment over base interface features was +0.0188 (secondary 95% interval [−0.0163, +0.0566]), which differs from the primary complex-context contrast beyond peptide-only scoring reported in the manuscript. Ridge and ≥50-row sensitivities likewise showed no established descriptor benefit beyond interface features ([ProteinMPNN scores](followup3/inverse_execution_v2/pooled_scores.csv), [contrasts](followup3/inverse_execution_v2/comparisons.csv) and [family interval](followup3/inverse_execution_v2/primary_family3_interval.json); [FoldX scores](followup3/energy_execution_v2/pooled_scores.csv), [contrasts](followup3/energy_execution_v2/comparisons.csv) and [family interval](followup3/energy_execution_v2/primary_family3_interval.json)).

In the selected full-cohort ProteinMPNN extension, the interface MLP scored 0.6955 and the fully augmented MLP scored 0.6885. Ridge regression gave a difference of −0.0016 [−0.0151, +0.0073], and the peptide-only, context and ≥50-row sensitivity intervals also spanned zero. This two-member design is distinct from the main selector, and its intervals do not account for selecting the extension after observing the panel results ([full-cohort contrasts](followup3/inverse_fulltrain_candidate/comparisons.csv)).

Among prediction heads and hybrid architectures, ranking-loss selection retained mean squared error in every interface fold ([ranking contrasts](finalization/ranking/comparisons.csv)). The hybrid selector scored 0.7039 versus 0.6995 for its matched standard MLP, selecting the standard MLP in three folds and a 75% MLP/25% regression-tree blend in two. Regression trees, ranking trees, tree-leaf preprocessing and principal components performed worse, whereas quantile preprocessing remained inconclusive ([hybrid comparisons](finalization/hybrids_cloud_v1/comparisons.csv)).

## Earlier sequence and pretrained-feature comparisons

The original test partition contains 5,633 rows, with 5,565 rows across 67 eligible alleles in this scoring panel. These historical comparisons evaluated frozen ESM-2 features ([Lin et al., 2023](https://pubmed.ncbi.nlm.nih.gov/36927031/)) and sequence inputs augmented with Boltz geometry and confidence summaries. Prior diagnostic test exposure prevents treating this partition as an untouched confirmation set, and because its population and representations differ from those in the nested development studies, these scores do not provide a matched numerical comparison of ESM-2 and internal Boltz pair features.

| Original test procedure | Median Spearman | Difference from sequence ensemble, 95% interval |
|---|---:|---:|
| Peptide + HLA pseudosequence ensemble | 0.7064 | Reference |
| Single sequence network | 0.6181 | −0.0883 [−0.1201, −0.0555] |
| Peptide + full HLA-domain ensemble | 0.6904 | −0.0160 [−0.0388, +0.0043] |
| ESM-2 ensemble | 0.6979 | −0.0085 [−0.0291, +0.0233] |
| Sequence + ESM-2 ensemble | 0.6816 | −0.0248 [−0.0460, +0.0073] |
| Sequence + Boltz geometry/confidence | 0.5947 | −0.1117 [−0.1339, −0.0575] |

On this panel, ESM-2 established no ranking gain over the sequence ensemble, while the implemented geometry-and-confidence pipeline performed worse ([test scores](stage6_test_summary.csv); [paired comparisons](stage6_test_paired_ci.csv)). Because that older pipeline also changed sequence encoding, its decline is not attributable solely to adding geometry. Likewise, the single-network versus ensemble difference bundles model averaging with encoding and recipe changes rather than isolating a causal ensembling effect, motivating the competitive sequence baselines used in the main comparisons.

## Earlier allele-transfer and affinity analyses

An earlier leave-allele-out sequence study across 22,533 combined training and validation rows (68 eligible alleles; six-member heads; no original test evaluation) provides baseline context rather than evidence for a Boltz advantage ([allele-holdout summary](stage7_allele_holdout_headline_seq_pep_pseudo.json)). Median within-allele Spearman was 0.7415 for 25 alleles at nearest training-pseudosequence distance ≤1 versus 0.3390 for 20 alleles at distance ≥4. Unlike the main group and joint holdouts, this evaluation had a median held-out-allele peptide-seen share of 1.0 (largely seen-peptide, unseen-allele prediction) and differing assay panel composition across distance strata, precluding a causal distance interpretation.

Across 5,135 dual-measured pairs (33 eligible of 55 alleles), affinity-derived rankings correlated with stability at a median within-allele Spearman of 0.5801, reflecting empirical association rather than an accuracy ceiling or evidence that auxiliary affinity data cannot help ([affinity association](stage2c_affinity_ceiling.csv)). Weak-binder augmentation ([augmentation deltas](stage2b_deltas.csv)) and auxiliary-affinity training ([auxiliary deltas](stage2c_deltas_ensemble_pep_pseudo.csv)) produced no established ranking gain at the tested sources, weights and recipes, without implying equivalence or universal failure.

## Calibration of recorded zero labels

A secondary sequence-model study evaluated zero-label probability prediction and calibration for recorded zero half-lives (`thalf_hours == 0`) across five outer peptide-cluster folds of the 19,716 original training rows (30 ensemble members per arm per outer fold; no original test evaluation). Probability metrics pool all 19,716 out-of-fold rows, whereas ranking is summarized as the median within-allele Spearman correlation across the 69 eligible alleles.

The observed target event is the recorded zero label itself rather than an established physiological threshold or verified assay detection limit, and zero-label calibration does not establish physical half-life calibration. Arms B1 and B2 optimized a Gaussian censored negative log-likelihood (NLL) that assumed censoring below 0.1 h as a historical modelling convention, whereas B0 optimized `log1p` mean squared error (MSE). All three arms used a shared mixture-of-members construction to map ensemble outputs to zero-event probabilities ([pooled scores](further_research/phase_b_pooled.csv); [paired contrasts](further_research/phase_b_paired.csv); [run table](further_research/phase_b_run_table.csv); [historical protocol](https://github.com/cheparukhin/peptide-hla-stability/blob/83d5327ea726812aae592621bcf7643af08e8c8b/reports/further_research/phase_b_declaration.md)).

| Model | Median within-allele Spearman | Pooled zero-label Brier | Fixed-bin ECE |
|---|---:|---:|---:|
| B0: `log1p`-MSE regression, stopped on MSE | 0.6604 | 0.1099 | 0.0643 |
| B1: censored Gaussian NLL, stopped on NLL | 0.6089 | 0.1016 | 0.0454 |
| B2: censored Gaussian NLL, stopped on MSE | 0.6497 | 0.0935 | 0.0196 |

Lower Brier scores reflect improved zero-label probability prediction, whereas lower fixed-bin expected calibration error (ECE) serves as a diagnostic of closer binned calibration. Compared with B0, B2 reduced pooled Brier error by +0.0165 (paired 95% interval [+0.0151, +0.0179]) while changing median within-allele Spearman by −0.0108 [−0.0242, +0.0036], an interval compatible with modest ranking loss rather than zero-cost equivalence. B2 also outperformed fold-fitted Platt (Brier 0.1035) and isotonic (0.1011) post-hoc maps, reducing pooled Brier error by 0.0100 [0.0091, 0.0109] and 0.0077 [0.0068, 0.0086], respectively.

However, B2 worsened `log1p` mean absolute error from 0.4865 to 0.5442 and censored NLL from 1.0714 to 1.1120, indicating a metric trade-off rather than a general predictive improvement. Holding the censored objective fixed while changing early stopping from NLL (B1) to MSE (B2) shifted ranking from 0.6089 to 0.6497 and Brier error from 0.1016 to 0.0935; thus, apparent objective effects depend on the stopping criterion and chosen endpoint, without implying that MSE stopping is universally optimal. Intervals reflect 2,000 paired peptide-cluster resamples conditional on saved fits, without project-wide multiplicity adjustment, and apply to sequence models rather than the main Boltz selector.

## Sequence predictions transfer to ligand presentation

A separate benchmark evaluated the sequence stability ensemble on an MHC Motif Atlas snapshot ([Tadros et al., 2023](https://pubmed.ncbi.nlm.nih.gov/36318236/)) without fitting to elution labels. The dataset included 81,600 eluted nine-mers and 816,000 human-proteome decoys (897,600 pairs across 51 alleles), excluding from both elution classes all 5,633 unique peptide sequences across the entire stability dataset (including training, validation, and test; this count happens to equal the 5,633 pair rows in the original test partition, though those quantities are distinct). Median per-allele AUROC was 0.9656 and AUPRC was 0.7804 ([results](stage3c_summary.csv); [provenance](stage3c_provenance.json)).

In control evaluations, AUROC was 0.9157 against ligands observed on other alleles and 0.6965 with permuted HLA pseudosequences. These controls address generic presentability and allele specificity but do not eliminate all confounding or identify causal contributions to AUROC. Because the endpoint and model differ from those in the main study, this benchmark is neither independent confirmation of the Boltz kinetic-stability advantage nor evidence of immunogenicity prediction.

## Compute and inference scope

Recorded compute figures span distinct populations, extraction recipes and scopes, precluding matched cost ratios or performance-per-dollar claims. Historical full coordinate generation across 28,166 pairs had zero execution failures and a runtime-derived cost of $212.71 from recorded durations and rates, with no final provider bill captured ([coordinate verification](ectodomain-20261004/production_verification.json)).

For the 19,716 development pairs in the pair-feature study, trunk-only extraction stopped before coordinate diffusion and confidence heads, reusing existing MSAs and checkpoints. This shared cache—reused across heads rather than charged per arm—recorded $66.84 in runtime accounting (covering successful attempts, worker processing, recorded serialization/reload overhead, and startup and scale-down allowances) and $65.08 in an earlier October 12 provider-meter snapshot including pilot runs ([runtime cost snapshot](further_research/round2/resources/runtime_cost_summary.json)). These differently defined measures from an earlier completed-scope snapshot are not additive, exclude storage, egress and new-input MSAs, and do not represent a final invoice or all-project spend.

| Exported head (CPU, 1 numerical thread, 25 repeats) | Population and scope | Single-pair median (ms) | 64-pair batch median (ms) |
|---|---|---:|---:|
| R4 per-position interface ridge | Warm cached development inputs | 1.56 | 1.65 |
| R5 interface MLP (30 members) | Warm cached development inputs | 2.40 | 3.87 |

Separate warm CPU head-call timings cover model and batch validation, feature scaling, member inference and output-frame construction on identified training-cache rows (performance only, without accuracy testing), and exclude cache file reads, process startup, MSA search and Boltz extraction ([inference manifest](finalization/inference/distribution_manifest.json); [inference guide](../docs/ROUND2_INFERENCE.md)). These portable exports represent fixed R4 and R5 recipes rather than the nested selector and do not inherit its 0.7212 development score. End-to-end latency and cost for uncached new peptide–HLA sequences remain unmeasured.
