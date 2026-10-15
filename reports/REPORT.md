# Learned interface representations improve peptide–HLA stability ranking

## Abstract

Peptide–HLA complex stability is a kinetic property that cannot be inferred from a static bound structure or equilibrium affinity alone. We tested whether frozen Boltz-2 interface representations improve the ranking of assay-derived peptide–HLA stability over supervised sequence models. Nested evaluation covered 19,716 peptide–HLA pairs grouped by sequence-related peptide clusters. Median within-allele Spearman correlation was 0.6668 for the competitive sequence selector and 0.7212 for the prespecified Boltz-2 embedding selector, a paired gain of **+0.0543, with a conditional 95% resampling interval of [+0.0314, +0.0808]**. The lower interval bound fell below +0.05, leaving the declared decision threshold unmet.

Matched nonlinear comparisons also favored interface embeddings when HLA groups or both peptide clusters and HLA groups were held out, and embeddings lost less ranking accuracy under reduced label budgets. Position-preserving token representations outperformed their pooled control, and full 9×34 pair interface representations outperformed a contact-restricted pooled control that jointly changes canonical-groove coverage and residue resolution. By contrast, a crystal-derived contact mask showed no established benefit over shuffled contacts. Twenty target-allele measurements improved ridge adaptation, whereas adding ProteinMPNN scores, repaired FoldX descriptors, ranking loss, or hybrid heads showed no established incremental gain. Although the sequence ensemble transferred to external ligand-elution data, these results do not establish a structural mechanism, calibrated half-lives, or independent experimental confirmation.

## Biological question and measurement

HLA class I molecules present short endogenous peptides at the cell surface for recognition by T cells. Effective presentation depends not only on whether a peptide binds the HLA groove at equilibrium, but also on how long the assembled ternary complex persists before dissociating. Complex stability, equilibrium binding affinity, and T-cell immunogenicity are related yet distinct biological properties. Because an equilibrium binding free energy reflects the balance between association and dissociation without determining kinetic rates, a plausible bound geometry or strong equilibrium affinity does not specify how rapidly a complex dissociates. Here we study how accurately learned representations rank nine-residue peptides by measured stability with a specified HLA molecule.

The benchmark dataset, attributed to [Rasmussen et al. (2016)](https://doi.org/10.4049/jimmunol.1600582), contains 28,166 peptide–HLA pairs spanning 5,633 peptides and 75 HLA alleles or constructs. The source study used a scintillation proximity assay and reported geometric means across two independent experiments ([Harndahl et al., assay method](https://doi.org/10.1016/j.jim.2010.10.012)). Because this assay monitors labelled β2-microglobulin dissociation as a proxy for ternary complex breakdown, the labels represent assay-derived complex-stability summaries rather than direct measurements of each peptide's microscopic unbinding rate.

Candidate peptides in the source study were selected using predicted binding, so this cohort does not represent a random peptide repertoire. Moreover, the supplied table omits individual replicate observations, raw dissociation time courses, and measurement-level failure or censoring flags. About 20.2% of the labels equal zero; these entries may reflect very low or unresolved stability, but the distribution alone neither establishes a common detection limit nor proves that every zero is left-censored.

We retain the supplied labels for within-allele ranking while noting that their structure limits claims regarding physical calibration, censoring models, and replicate reliability. Three engineered C67S constructs also remain in the assessed population, and effects of this substitution cannot be separated without matched wild-type measurements ([data description](../docs/DATASETS.md)).

## Methods

### Cohorts and held-out assessment

The original benchmark split contains 19,716 training, 2,817 validation, and 5,633 test pairs. Nine-mers connected by a Hamming distance of three or fewer substitutions belong to the same peptide cluster and remain in a single partition. All embedding, transfer, label-budget, adaptation, and descriptor evaluations reported below use the original training partition of 3,587 peptide clusters across 75 HLA alleles or constructs. Five outer peptide-cluster folds generate held-out predictions for every training pair while restricting regularization, preprocessing, early stopping, and procedure selection to inner data.

Because of prior exploration, these analyses represent development assessments rather than independent confirmation. An earlier diagnostic study repartitioned the complete dataset and used 3,350 original test rows for model fitting, 448 for stopping, and 585 for evaluation. The subsequent embedding, transfer, adaptation, and descriptor studies used the original training partition without reusing the original test split, but those earlier exposures and aggregate results informed the research process. Neither nested cross-validation nor new analyses of the training cohort restore independent confirmation ([evaluation specification](../EVALUATION.md)).

Transfer experiments use five HLA-group folds and their 25 intersections with the peptide-cluster folds. Related HLA sequences are grouped under a fixed Hamming-distance rule, which serves as a sequence convention rather than a validated functional distance. Joint holdout excludes both the target HLA group and the target peptide cluster from fitting. Because predictions pooled across joint cells come from separately fitted models, their numerical scales may differ; no outcome-based recalibration was applied.

### Representations and predictors

Each primary procedure includes inner-loop selection, so outer-fold results evaluate the complete prespecified rule rather than determining a new global winner. The competitive sequence selector chooses among neural networks, regression trees, and a pairwise ranker using inner data. The main embedding selector chooses among frozen Boltz-2 representations paired with ridge or multilayer perceptron (MLP) heads while retaining the original sequence reference as a fallback. Main neural heads average 30 ensemble members.

Boltz-2 features describe a predicted complex containing the HLA heavy chain, β2-microglobulin, and peptide. We compare chain averages, position-preserving token features, and peptide–HLA interface pair features. The principal contact-pooled representation retains separate peptide positions 1, 2, and 9 while averaging positions 3–8 across a fixed heavy-atom contact mask derived from five crystal structures. Dimension-matched controls preserve feature dimensions and label-independent channel projections. Specifically, the token control replaces positional tokens with tiled averages of the same underlying tokens. The pair comparison contrasts the full 9×34 pair interface—retaining the 34 canonical groove positions—against crystal-mask-selected contacts averaged at each peptide position and tiled across those 34 positions, which changes coverage and resolution together. Finally, the contact-mask control shuffles HLA contact membership ([Feature definitions and saved protocol](further_research/round2/embedding/protocol.json)).

| Description used here | Identifier in source tables |
|---|---|
| Positional peptide + 34-residue HLA one-hot sequence | R0 |
| Boltz chain averages | R2 |
| Projected positional token features | R3 |
| Interface features at all nine peptide positions | R4 |
| Contact-pooled interface features | R5 |
| Native-peptide compatibility on peptide backbone alone | P |
| Native-peptide compatibility with HLA/β2m context | C |
| Repaired FoldX energy and interface descriptors | E |

Matched transfer comparisons hold the learner recipe, parameter grid, selection rule, and stopping criteria fixed across feature sets, although feature dimensions and parameter counts still differ. Moreover, a ridge model on concatenated peptide and HLA one-hot features is additive: within a fitted model, HLA identity shifts the intercept without changing peptide ordering. This linear ridge model cannot represent allele-specific peptide motifs. Nonlinear sequence comparisons therefore carry more weight than the ridge comparison. Boltz-2's fuller molecular context and pretraining also prevent attributing differences solely to three-dimensional structure.

ProteinMPNN evaluates the native peptide using the full-backbone `v_48_020.pt` checkpoint with zero added coordinate noise and 16 shared decoding orders across contexts, yielding nine position-wise mean log probabilities per block ([ProteinMPNN](followup3/inverse_execution_v2/protocol.json)). Because the peptide backbone was predicted in the full complex, removing the other chains does not create a context-free structural control. FoldX applies `RepairPDB` and `AnalyseComplex` with β2-microglobulin retained to produce 29 descriptors ([FoldX](followup3/foldx/protocol.json)). Both methods are tested as empirical predictors rather than direct kinetic measurements.

### Estimand and uncertainty

For each eligible allele, we compute the within-allele Spearman correlation on pooled held-out predictions. The primary endpoint is the median of these correlations. This median gives each allele equal weight. The main panel contains 69 alleles with at least 20 observations and varying labels, covering 19,647 rows. A sensitivity panel requires at least 50 observations and contains 68 alleles.

Each reported difference subtracts the two panel medians rather than taking a median of allele differences or fold scores. This endpoint addresses within-allele ranking rather than error in hours.

Intervals use 2,000 paired bootstrap draws of whole peptide clusters. Transfer and adaptation analyses additionally resample HLA groups independently, carrying their multiplicities into the allele median as a dependence sensitivity check. All intervals are **conditional on the saved fitted predictions and selection decisions**. They omit retraining, repeated selection, support-design variation, and independent assay variation, and should not be read as demonstrated coverage for a newly trained or deployed procedure.

The main study's declared decision criterion requires the lower interval bound to exceed +0.05, which is a study convention rather than a detection limit or demonstrated biological utility threshold. The adaptation, ProteinMPNN-context, and repaired-FoldX primary questions form a three-comparison family reported with Bonferroni-adjusted 98.3333% intervals. Other intervals are pointwise 95% summaries that do not cover the project's adaptive research history ([evaluation specification](../EVALUATION.md)).

## Results

### Interface embeddings improve ranking within the development cohort

Across the 69-allele panel, the prespecified embedding selector outperformed the competitive sequence selector by **+0.0543 [+0.0314, +0.0808]**. Because the lower interval bound of +0.0314 remained below +0.05, the declared worthwhile-gain criterion was not met. Requiring at least 50 observations per allele across 68 alleles gave +0.0564 [+0.0317, +0.0791]. Inner selection chose the contact-pooled interface MLP in four outer folds and the per-position interface ridge model in one.

| Main assessment procedure | Median within-allele Spearman |
|---|---:|
| Original sequence MLP ensemble | 0.6615 |
| Competitive sequence selector | 0.6668 |
| Chain-average Boltz MLP | 0.4219 |
| Per-position interface ridge | 0.6845 |
| Per-position interface MLP | 0.7136 |
| Contact-pooled interface MLP | 0.7210 |
| Same MLP with shuffled contact membership | 0.7203 |
| Full 9 × 34 positional interface MLP | 0.7315 |
| Prespecified embedding selector | 0.7212 |

Component scores describe the assessed procedures rather than defining a replacement selection rule. In paired contrasts, the full positional interface model's difference from the contact-pooled model was uncertain at +0.0106 [−0.0142, +0.0220], as was the competitive sequence selector's difference from the original ensemble at +0.0053 [−0.0093, +0.0123] ([All absolute scores](further_research/round2/results/main/pooled_scores.csv); [paired comparisons](further_research/round2/results/main/paired_comparisons.csv)).

Dimension-matched controls showed clear gains over their pooled counterparts: +0.3387 [+0.3029, +0.3773] for projected tokens over pooled tokens and +0.1663 [+0.1378, +0.1992] for full 9×34 pair interface features over the contact-restricted pooled pair control. By contrast, comparing the designated versus shuffled contact mask gave +0.0007 [−0.0096, +0.0236]. The token control therefore isolates useful positional indexing, whereas the 9×34 pair contrast changes canonical-groove coverage and residue resolution jointly. The specified contact map has no established benefit.

Sequence and embedding predictions also contain complementary information. An inner-selected blend of the per-position interface ridge model and original sequence ensemble scored 0.7238, exceeding that ridge model alone by +0.0393 [+0.0220, +0.0567] and the competitive sequence selector by +0.0569 [+0.0426, +0.0838]. Residual correction scored 0.7114, or +0.0446 [+0.0304, +0.0671] over the sequence selector ([Complementarity comparisons](further_research/round2/complementarity/results/paired_comparisons.csv)). Both positive secondary comparisons also had lower interval bounds below +0.05.

### Nonlinear transfer comparisons favor interface embeddings

Holding out entire HLA groups or jointly holding out peptide clusters and HLA groups produced large differences between matched nonlinear predictors.

| Groups held out | Matched sequence MLP | Matched interface MLP | Difference, crossed 95% interval |
|---|---:|---:|---:|
| HLA groups | 0.1605 | 0.5667 | +0.4062 [+0.3073, +0.4733] |
| Peptide clusters and HLA groups | 0.1242 | 0.4980 | +0.3738 [+0.3092, +0.4638] |

Full selection procedures followed the same pattern in these two regimes. The original sequence ensemble scored 0.1709 and 0.1350, while the expanded sequence selector scored 0.1921 and 0.1085; the expanded selector's lower joint score is retained rather than treated as an improvement over the original reference. The embedding selector scored 0.5463 and 0.5067, exceeding the expanded sequence selector by +0.3542 [+0.2762, +0.4563] and +0.3982 [+0.3143, +0.4613].

Matched ridge gains were also positive, though their additive sequence comparator has the limitation described in Methods. The nonlinear gains retained positive lower bounds in the ≥50-row panel and positive point differences after deleting any one HLA group ([Transfer scores, all contrasts and resampling modes](finalization/transfer_cloud_recovery/results/comparisons.csv)). These checks strengthen the internal comparison without measuring transfer to a new assay or establishing structural causality.

### Interface embeddings lose less performance under smaller label budgets

A separate fixed-recipe comparison reduced the number of outer-fitting peptide clusters across three predetermined nested subsets, averaging scores across subsets and counting labels reserved for stopping.

| Fitting-cluster budget | Sequence MLP | Interface MLP | Difference |
|---|---:|---:|---:|
| 25% | 0.2640 | 0.5877 | +0.3237 |
| 50% | 0.4010 | 0.6504 | +0.2494 |
| 100% | 0.5373 | 0.6995 | +0.1622 |

Both predictors deteriorated as training clusters were removed, but the interface model deteriorated less, widening its advantage by +0.1615 [+0.1333, +0.2041] at the smallest budget relative to the full budget ([Absolute curves](finalization/label_efficiency/absolute_curves.csv); [budget contrasts](finalization/label_efficiency/comparisons.csv)). This reflects relative robustness under fixed two-member recipes whose sequence baseline is weaker than the competitive main-study comparator, and does not estimate prospective assay savings.

### Twenty target-allele measurements improve ridge adaptation

A source ridge model was adapted using 20 measured peptides per target allele after excluding target HLA groups and support/query peptide families from source fitting. Three fixed support designs were averaged as scores; each used 1,360 target measurements across 68 alleles with no target tuning or stopping labels, while source stopping labels were counted separately.

Target adaptation improved the interface model from **0.5296 to 0.5763**, a gain of **+0.0468** with a crossed 95% interval of [+0.0227, +0.1035] and a three-primary **98.3333% interval of [+0.0140, +0.1195]** ([Adaptation curves](followup3/fewshot/absolute_curves.csv); [comparisons](followup3/fewshot/comparisons.csv); [family interval](followup3/fewshot/primary_family3_interval.json)). The ≥20- and ≥50-row panels are identical here, so their agreement is not an additional robustness result. A reduced four-member sequence anchor improved from 0.0830 to 0.2058; it is not the competitive main-study comparator.

![Ranking performance as the target-allele support set grows.](followup3/figures/fewshot_curves.svg)

*Figure 1. Ranking performance as the target-allele support set grows across three fixed support designs on 68 alleles with a reduced sequence reference and no curve uncertainty bands; interval estimates are conditional on the fixed designs.*

A post hoc control retained only the adapter's scalar offset. Full adaptation exceeded this control by +0.0444 [+0.0165, +0.0928], whereas offset alone versus no adaptation was uncertain at +0.0024 [−0.0062, +0.0239] ([Offset comparisons](followup3/fewshot_offset/comparisons.csv)). This exploratory result supports a contribution from feature-dependent correction beyond aligning fitted-cell offsets, though its unadjusted nominal interval does not establish a biological mechanism.

### Supplemental descriptors and complex heads show no established benefit

The descriptor comparison used a balanced subset of 4,449 pairs, 2,082 peptide clusters, and all 75 alleles (at most 64 rows per allele; 69 eligible alleles). Matched two-member MLPs and ridge sensitivities assessed peptide-only ProteinMPNN compatibility, full-complex compatibility, and repaired FoldX descriptors.

| Balanced-panel MLP | Sequence features | Interface features |
|---|---:|---:|
| Base representation | 0.2952 | 0.6007 |
| + Peptide-only ProteinMPNN scores | 0.2892 | 0.5956 |
| + Peptide-only and full-complex ProteinMPNN scores | 0.3406 | 0.6195 |
| + Repaired FoldX descriptors | 0.3276 | 0.6157 |

The primary ProteinMPNN context increment was +0.0240 (98.3333% family interval [−0.0219, +0.0659]), and its net increment over base interface features was +0.0188 (secondary 95% interval [−0.0163, +0.0566]). FoldX's primary increment was +0.0150 (98.3333% family interval [−0.0524, +0.0450]). Ridge and ≥50-row sensitivities likewise did not establish benefit beyond interface features, and positive exploratory sequence-control gains do not replace these primary questions ([ProteinMPNN scores](followup3/inverse_execution_v2/pooled_scores.csv); [contrasts](followup3/inverse_execution_v2/comparisons.csv); [family interval](followup3/inverse_execution_v2/primary_family3_interval.json); [FoldX scores](followup3/energy_execution_v2/pooled_scores.csv); [contrasts](followup3/energy_execution_v2/comparisons.csv); [family interval](followup3/energy_execution_v2/primary_family3_interval.json)).

A ProteinMPNN extension selected after these results assessed all 19,716 training pairs. The interface MLP scored 0.6955 and its fully augmented counterpart scored 0.6885: **−0.0070 [−0.0220, +0.0131]**; ridge gave −0.0016 [−0.0151, +0.0073], and peptide-only, context, and ≥50-row sensitivities also spanned zero ([All full-cohort contrasts](followup3/inverse_fulltrain_candidate/comparisons.csv)). These pointwise 95% intervals do not account for selecting the extension, which is not a fourth primary test. Reusing overlapping training pairs does not provide independent confirmation, and the intervals establish neither benefit, harm, nor equivalence.

More complex heads showed the same pattern. Adding within-allele RankNet loss to mean squared error gave +0.0071 [−0.0051, +0.0098], and inner selection retained mean squared error in every interface fold ([Ranking-loss contrasts](finalization/ranking/comparisons.csv)). A selector over trees, neural preprocessing, and blends scored 0.7039 against 0.6995 for its matched standard MLP (+0.0044 [−0.0055, +0.0100]), choosing the standard MLP in three folds and a 75% MLP/25% regression-tree blend in two ([hybrid contrasts](finalization/hybrids_cloud_v1/comparisons.csv)). Regression trees, ranking trees, tree-leaf preprocessing, and principal components performed worse, while quantile preprocessing remained inconclusive.

### Earlier sequence and pretrained-feature comparisons

The full original test partition contains 5,633 rows, of which 5,565 rows across 67 eligible alleles form the evaluated panel. On this panel, a separate comparison evaluated frozen ESM-2 features and sequence inputs augmented with Boltz-2 geometry and confidence summaries rather than internal interface embeddings. Earlier test exposure described in Methods prevents treating this partition as wholly untouched.

| Original test procedure | Median Spearman | Difference from sequence ensemble, 95% interval |
|---|---:|---:|
| Peptide + HLA pseudosequence ensemble | 0.7064 | Reference |
| Single sequence network | 0.6181 | −0.0883 [−0.1201, −0.0555] |
| Peptide + full HLA-domain ensemble | 0.6904 | −0.0160 [−0.0388, +0.0043] |
| ESM-2 ensemble | 0.6979 | −0.0085 [−0.0291, +0.0233] |
| Sequence + ESM-2 ensemble | 0.6816 | −0.0248 [−0.0460, +0.0073] |
| Sequence + Boltz geometry/confidence | 0.5947 | −0.1117 [−0.1339, −0.0575] |

On this test set, coarse geometry augmentation reduced ranking performance, and ESM-2 did not establish improvement ([Test scores](stage6_test_summary.csv); [paired comparisons](stage6_test_paired_ci.csv)). Because both representations and evaluation populations differ from the nested training-cohort studies, these findings remain separate from the positive interface-embedding result and do not imply that protein foundation models uniformly help or fail.

### Sequence predictions transfer to ligand presentation

An external ligand-elution evaluation assessed the sequence stability ensemble on the MHC Motif Atlas without fitting to elution labels. Across 81,600 eluted nine-mers and 816,000 human-proteome decoys (897,600 pairs across 51 alleles, with all 5,633 stability-measured peptide sequences excluded from both classes), the sequence ensemble achieved a median per-allele AUROC of 0.9656 and AUPRC of 0.7804 ([Elution results](stage3c_summary.csv); [Elution benchmark provenance](stage3c_provenance.json)).

When decoys were sampled from ligands observed on other alleles, median AUROC was 0.9157, whereas permuted HLA pseudosequences gave 0.6965. Switched-allele decoys address generic presentability, but do not eliminate all confounding or permit causal decomposition of AUROC. This assessment evaluates sequence-model transfer to ligand presentation, not independent confirmation of the Boltz-2 kinetic-stability advantage or immunogenicity.

## Discussion

The most defensible conclusion is that learned, position-aware interface representations contain information useful for stability ranking in this benchmark. That evidence combines a competitive sequence comparison, matched nonlinear transfer controls, a matched token-position ablation, and a comparison of the full 9×34 pair interface against a contact-restricted pooled pair control that jointly varies canonical-groove coverage and residue resolution. It remains a comparison of complete predictive procedures whose pretraining, molecular context, representation dimensions, and fitting choices differ. Moreover, the contact-map control provides no evidence for a specific structural explanation, and supervised HLA holdout does not establish that related structures were absent from foundation-model pretraining.

Conditional intervals quantify variation in the saved assessments rather than the full uncertainty of the research program. Nested selection limits leakage within each declared procedure, but repeated use of the same population informed which procedures were studied. Furthermore, the positive main result does not establish a lower bound above +0.05, and even such a gain would require a separate connection to practical benefit. Rank correlation alone does not demonstrate calibrated half-lives, better immunogenicity prediction, or more stable candidates recovered within an experimental budget.

Measurement uncertainty is a second unresolved constraint. The aggregated labels cannot identify replicate reliability, the origin of zero values, or a noise ceiling. Candidate public datasets have unresolved construct and measurement lineage and mix assay types, so they do not currently supply representative independent confirmation ([Independent-candidate provenance](followup3/independent_data/study_method_status.csv)).

Finally, the descriptor and head comparisons favor retaining a simpler interface model for the next fixed comparison while leaving small incremental effects unresolved. Decisive evidence should come from independent kinetic measurements, with candidate and reference procedures locked before outcomes are opened. A suitable study must define its intended peptide and HLA population, retain raw curves and independent repeats, record measurement failure separately from censoring, and predefine both ranking and an experimental decision endpoint ([independent assay design](finalization/INDEPENDENT_ASSAY_BRIEF.md); its proposed sample sizes are planning assumptions, not demonstrated power).

## Data and model availability

The [dataset documentation](../docs/DATASETS.md) and [evaluation specification](../EVALUATION.md) define the scientific inputs and analysis. Linked CSV and JSON tables provide all assessed arms, predictions, contrasts, sensitivities, and frozen protocols. Portable ridge and neural models accept compatible, verified cached Boltz-2 features and output uncalibrated log-scale predictions. The [inference guide](../docs/ROUND2_INFERENCE.md) describes the supported input contract; generating features for new sequences is a separate computational step.

Feature and cohort hashes bind the stored evidence. A remaining extraction-provenance limitation is that the fanout wrapper's recorded launch digest differs from its retained source digest, so its exact launch-time bytes cannot be certified, although the frozen extraction sources and feature checks agree ([Cache provenance](further_research/round2/resources/cache_completion_audit.json)). Reproducing saved arithmetic or execution does not establish experimental reproducibility.

## Method references

- Rasmussen et al. (2016). *Pan-Specific Prediction of Peptide–MHC Class I Complex Stability, a Correlate of T Cell Immunogenicity.* [The Journal of Immunology](https://doi.org/10.4049/jimmunol.1600582).
- Harndahl et al. (2011). *Real-time, high-throughput measurements of peptide–MHC-I dissociation using a scintillation proximity assay.* [Journal of Immunological Methods](https://doi.org/10.1016/j.jim.2010.10.012).
- Passaro et al. (2025). *Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction.* [Technical preprint](https://doi.org/10.1101/2025.06.14.659707).
- Dauparas et al. (2022). *Robust deep learning–based protein sequence design using ProteinMPNN.* [Science](https://doi.org/10.1126/science.add2187).
- Delgado et al. (2019). *FoldX 5.0: working with RNA, small molecules and a new graphical interface.* [Bioinformatics](https://doi.org/10.1093/bioinformatics/btz184).
- Cawley and Talbot (2010). *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation.* [Journal of Machine Learning Research](https://www.jmlr.org/papers/v11/cawley10a.html).
