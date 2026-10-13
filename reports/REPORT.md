# Learned interface representations improve peptide–HLA stability ranking

## Abstract

Peptide–HLA complex stability is a kinetic property that cannot be inferred from a plausible bound structure alone. We tested whether frozen Boltz-2 representations improve prediction of assay-derived stability over supervised sequence models. Nested assessment of 19,716 peptide–HLA pairs, with related peptides held together, increased median within-allele Spearman correlation from 0.6668 to 0.7212: a difference of **+0.0543, with a conditional 95% resampling interval of [+0.0314, +0.0808]**. Matched nonlinear comparisons also favoured embeddings when HLA groups, or both peptide and HLA groups, were held out. Position-preserving features outperformed pooled controls, whereas the specified contact map had no established advantage over shuffled contacts. Twenty measurements per target allele improved adaptation; additional ProteinMPNN scores, repaired FoldX descriptors, ranking loss and hybrid heads did not establish further gains. These results support useful predictive information in learned interface representations. They do not isolate a structural mechanism, demonstrate calibrated physical half-lives or provide independent experimental confirmation.

## Biological question and measurement

HLA class I molecules present short peptides for immune recognition. We study the ranking of nine-residue peptides by the stability of their complex with a specified HLA molecule. Stability, binding affinity and immunogenicity are related but distinct quantities. In particular, an equilibrium binding free energy does not determine a dissociation rate without kinetic information.

The supplied benchmark, attributed to [Rasmussen et al. (2016)](https://doi.org/10.4049/jimmunol.1600582), contains 28,166 pairs, 5,633 peptides and 75 HLA alleles or constructs. The source study used a scintillation proximity assay and reported geometric means of two independent experiments. This assay follows labelled β2-microglobulin dissociation as a proxy for peptide–HLA dissociation; the labels are therefore assay-derived complex-stability measurements, rather than direct measurements of every peptide's microscopic unbinding rate. [Harndahl et al., assay method](https://doi.org/10.1016/j.jim.2010.10.012).

The source study selected peptides using predicted binding, so this population does not represent a random peptide repertoire. The supplied file lacks the separate replicate observations, raw time courses and measurement-level failure or censoring flags. About 20.2% of labels are zero. They may represent very low or unresolved stability, but the distribution alone does not establish a common detection limit or prove that every zero is left-censored. We retain the supplied labels for the ranking analyses. Their interpretation limits claims about calibration, censoring models and assay reliability. Three engineered C67S constructs remain in the assessed population; effects of this substitution cannot be isolated without matched wild-type measurements. The [data description](../docs/DATASETS.md) specifies the source, constructs and exclusions.

## Methods

### Cohorts and held-out assessment

The original split contains 19,716 training, 2,817 validation and 5,633 test pairs. Peptides connected by Hamming distance of three or fewer substitutions belong to the same cluster and remain in one partition. All embedding, transfer, label-budget and descriptor assessments reported below use the original training partition: 3,587 peptide clusters across 75 alleles or constructs. Five outer peptide-cluster folds produce held-out predictions for every training pair; regularisation, preprocessing, stopping and procedure selection use only the permitted inner data.

These are development assessments. An earlier diagnostic repartitioned the complete dataset and used 3,350 original test rows for fitting, 448 for stopping and 585 for evaluation. Although the final benchmark models were fitted on the original training partition, those exposures and aggregate results influenced the research process. The original test was not reused for the subsequent studies, but neither nested assessment nor a new analysis of the training cohort restores independent confirmation. Exact split and exposure details are in the [evaluation specification](../EVALUATION.md).

Transfer uses five HLA-group folds and their 25 intersections with peptide-cluster folds. Related HLA sequences are grouped by a fixed Hamming-distance rule; this is a sequence convention, not a validated functional distance. Joint holdout excludes both the target HLA group and target peptide cluster from fitting. Predictions pooled across joint cells may differ in scale because they come from different fitted models. No outcome-based recalibration was applied.

### Representations and predictors

The competitive sequence procedure selects among neural models, regression trees and a pairwise ranker using inner data. The main embedding procedure selects among frozen representations and ridge or multilayer perceptron (MLP) heads, retaining the original sequence reference as a fallback. Main neural heads average 30 members. Selection is part of each assessed procedure; outer results do not determine a new global winner.

Boltz-2 features describe a complex containing HLA, β2-microglobulin and peptide. We compare chain averages, position-preserving token features and interface pair features. The principal contact-pooled representation retains separate peptide positions 1, 2 and 9 and averages positions 3–8. Its fixed contact mask uses heavy-atom proximity across five crystal structures. Matched controls shuffle HLA contact membership or replace positional features with tiled pooled values while preserving dimensions and projection. Fixed channel projections are independent of labels. [Feature definitions and saved protocol](further_research/round2/embedding/protocol.json).

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

Matched transfer comparisons hold the learner recipe, parameter grid, selection and stopping rules fixed across feature sets. Feature dimensions and model parameter counts still differ. Moreover, a ridge model of concatenated peptide and HLA one-hot features is additive: within a fitted model, HLA identity changes the intercept but cannot change peptide ordering. It cannot represent allele-specific peptide motifs. Nonlinear sequence comparisons therefore carry more weight than the ridge comparison. Boltz's fuller molecular context and pretraining also prevent attributing the difference solely to three-dimensional structure.

ProteinMPNN evaluates the native peptide using the full-backbone `v_48_020.pt` checkpoint, zero added coordinate noise and the same 16 decoding orders across contexts. Each descriptor block contains nine position-wise mean log probabilities. The peptide-only backbone was predicted in the full native complex, so removing its other chains does not create a context-free structural control. FoldX applies `RepairPDB` and `AnalyseComplex`, retaining β2-microglobulin, to produce 29 descriptors. Both methods are tested as predictors of the measured endpoint; neither directly measures dissociation kinetics. Exact settings are in the [ProteinMPNN](followup3/inverse_execution_v2/protocol.json) and [FoldX](followup3/foldx/protocol.json) protocols.

### Estimand and uncertainty

The primary endpoint is the median of within-allele Spearman correlations on pooled held-out predictions. The main panel contains 69 alleles with at least 20 observations and varying labels, covering 19,647 rows. A sensitivity panel requires at least 50 observations and contains 68 alleles. A reported difference subtracts the two panel medians; it is not a median of allele differences or of fold scores. This estimand gives equal weight to eligible alleles and addresses ranking, rather than error in hours.

Intervals use 2,000 paired bootstrap draws of whole peptide clusters. Transfer and adaptation additionally resample HLA groups independently, carrying their multiplicities into the allele median. These crossed intervals are dependence sensitivities. All intervals are **conditional on the saved fitted predictions and selection decisions**. They omit retraining, repeated selection, support-design variation and independent assay variation; they should not be read as demonstrated coverage for a newly trained or deployed procedure.

The main study's declared decision criterion requires the lower interval bound to exceed +0.05. That threshold is a study decision, not a universal detection limit or a demonstrated biological utility threshold. The adaptation, ProteinMPNN-context and repaired-FoldX primary questions form a three-comparison family, reported with 98.3333% intervals using Bonferroni tail probabilities. Other intervals are pointwise 95% unless stated otherwise. This correction covers the specified family, not the project's adaptive research history. The [evaluation specification](../EVALUATION.md) contains the complete analysis rules.

## Results

### Interface embeddings improve ranking within the development cohort

The embedding selector achieved 0.721163 against 0.666841 for the competitive sequence selector: **Δ +0.054322 [+0.031395, +0.080805]**. The lower bound is positive but below +0.05, so the declared worthwhile-gain criterion was not met. Requiring 50 observations per allele gave +0.056352 [+0.031732, +0.079137]. Inner selection chose the contact-pooled MLP in four outer folds and the per-position interface ridge model in one.

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

Component scores describe the assessed procedures and are not a replacement selection rule. The full positional model's difference from the contact-pooled model was uncertain, +0.0106 [−0.0142, +0.0220]. The competitive sequence selector's improvement over the original ensemble was also uncertain, +0.0053 [−0.0093, +0.0123]. [All absolute scores](further_research/round2/results/main/pooled_scores.csv) and [paired comparisons](further_research/round2/results/main/paired_comparisons.csv).

Matched positional controls show clear gains over their pooled counterparts: +0.3387 [+0.3029, +0.3773] for projected tokens and +0.1663 [+0.1378, +0.1992] for full interface positions. By contrast, the designated versus shuffled contact mask gave +0.0007 [−0.0096, +0.0236]. Position information is useful in these representations; the particular contact map has no established benefit.

Sequence and embedding predictions also contain complementary information. An inner-selected blend of the per-position interface ridge model and original sequence ensemble scored 0.7238: +0.0393 [+0.0220, +0.0567] over that ridge alone and +0.0569 [+0.0426, +0.0838] over the competitive sequence selector. Residual correction scored 0.7114, +0.0446 [+0.0304, +0.0671] over the selector. These positive secondary comparisons also fall short of a lower bound above +0.05. [Complementarity comparisons](further_research/round2/complementarity/results/paired_comparisons.csv).

### Nonlinear transfer comparisons favour embeddings

| Groups held out | Matched sequence MLP | Matched interface MLP | Difference, crossed 95% interval |
|---|---:|---:|---:|
| HLA groups | 0.1605 | 0.5667 | +0.4062 [+0.3073, +0.4733] |
| Peptide clusters and HLA groups | 0.1242 | 0.4980 | +0.3738 [+0.3092, +0.4638] |

The original sequence ensemble scored 0.1709 and 0.1350 in these regimes; the expanded sequence selector scored 0.1921 and 0.1085. The embedding selector scored 0.5463 and 0.5067, exceeding the expanded selector by +0.3542 [+0.2762, +0.4563] and +0.3982 [+0.3143, +0.4613]. The expanded sequence selector's lower joint score is retained in the comparison rather than treated as an improvement over the original reference.

Matched ridge gains were also positive, but their additive sequence comparator has the limitation described above. The nonlinear gains retained positive lower bounds in the ≥50-row panel, and positive point differences after deleting any one HLA group. These checks strengthen the development comparison without measuring transfer to a new assay or establishing structural causality. [Transfer scores, all contrasts and resampling modes](finalization/transfer_cloud_recovery/results/comparisons.csv).

### Embeddings lose less ranking performance when training labels are reduced

A separate fixed-recipe comparison reduced the number of outer-fitting peptide clusters. Three predetermined nested subsets were averaged as scores, counting labels reserved for stopping.

| Fitting-cluster budget | Sequence MLP | Interface MLP | Difference |
|---|---:|---:|---:|
| 25% | 0.2640 | 0.5877 | +0.3237 |
| 50% | 0.4010 | 0.6504 | +0.2494 |
| 100% | 0.5373 | 0.6995 | +0.1622 |

The embedding advantage increased by +0.1615 [+0.1333, +0.2041] at 25% relative to 100%. Both models deteriorated; the embedding model deteriorated less. This is relative robustness under these fixed, two-member recipes, whose sequence baseline is weaker than the competitive main-study comparator. It does not estimate prospective assay savings. [Absolute curves](finalization/label_efficiency/absolute_curves.csv) and [budget contrasts](finalization/label_efficiency/comparisons.csv).

### Twenty target measurements improve adaptation

A source ridge model was adapted using 20 measured peptides per target allele. Target HLA groups and support/query peptide families were excluded from source fitting. Three fixed support designs were averaged as scores; each used 1,360 target measurements across 68 alleles, with no target tuning or stopping labels. Source stopping labels were counted separately.

The interface model improved from **0.529583 to 0.576337**, a gain of **+0.046754**. Its crossed 95% interval was [+0.022657, +0.103467] and its three-primary **98.3333% interval was [+0.014042, +0.119457]**. The ≥20- and ≥50-row panels are identical here, so their agreement is not an additional robustness result. A reduced four-member sequence anchor improved from 0.0830 to 0.2058; it is not the competitive main-study comparator. [Adaptation curves](followup3/fewshot/absolute_curves.csv), [comparisons](followup3/fewshot/comparisons.csv) and [family interval](followup3/fewshot/primary_family3_interval.json).

A post hoc control retained only the adapter's scalar offset. Full adaptation exceeded this control by +0.0444 [+0.0165, +0.0928]; offset alone versus no adaptation was uncertain, +0.0024 [−0.0062, +0.0239]. This exploratory result supports a contribution from feature-dependent correction beyond alignment of fitted-cell offsets. Its nominal interval is unadjusted and does not establish a biological mechanism. [Offset comparisons](followup3/fewshot_offset/comparisons.csv).

### Additional descriptors and more complex heads have no established incremental benefit

The descriptor comparison used a balanced subset of 4,449 pairs, 2,082 peptide clusters and all 75 alleles, with at most 64 rows per allele. Its primary panel contained 69 eligible alleles. Matched two-member MLPs and ridge sensitivity analyses assessed peptide-only ProteinMPNN compatibility, full-complex compatibility and repaired FoldX descriptors.

| Balanced-panel MLP | Sequence features | Interface features |
|---|---:|---:|
| Base representation | 0.2952 | 0.6007 |
| + Peptide-only ProteinMPNN scores | 0.2892 | 0.5956 |
| + Peptide-only and full-complex ProteinMPNN scores | 0.3406 | 0.6195 |
| + Repaired FoldX descriptors | 0.3276 | 0.6157 |

The primary ProteinMPNN context increment was +0.0240, with a 98.3333% family interval of [−0.0219, +0.0659]. Its net increment over interface features alone was +0.0188 [−0.0163, +0.0566] under a secondary 95% interval. FoldX's primary increment was +0.0150, with a 98.3333% family interval of [−0.0524, +0.0450]. Ridge and ≥50-row sensitivities likewise did not establish additional benefit beyond interface features. Positive exploratory sequence-control comparisons do not replace these primary questions. [ProteinMPNN scores](followup3/inverse_execution_v2/pooled_scores.csv), [contrasts](followup3/inverse_execution_v2/comparisons.csv) and [family interval](followup3/inverse_execution_v2/primary_family3_interval.json); [FoldX scores](followup3/energy_execution_v2/pooled_scores.csv), [contrasts](followup3/energy_execution_v2/comparisons.csv) and [family interval](followup3/energy_execution_v2/primary_family3_interval.json).

A ProteinMPNN extension selected after these results assessed all 19,716 training pairs. The interface MLP scored 0.6955 and its fully augmented counterpart 0.6885: **−0.0070 [−0.0220, +0.0131]**. Ridge gave −0.0016 [−0.0151, +0.0073]; peptide-only, context and ≥50-row sensitivities also spanned zero. These pointwise 95% intervals do not account for selection of the extension, which is not a fourth primary in the original family. Adding overlapping training pairs does not create independent confirmation. The extension establishes neither added benefit, harm nor equivalence. [All full-cohort contrasts](followup3/inverse_fulltrain_candidate/comparisons.csv).

The same pattern held for more complex predictive heads. Adding within-allele RankNet to mean squared error gave +0.0071 [−0.0051, +0.0098]; inner selection retained mean squared error in every interface-model fold. A selector over trees, neural preprocessing and blends scored 0.7039 against 0.6995 for its matched standard MLP: +0.0044 [−0.0055, +0.0100]. It selected the standard MLP in three folds and a 75% MLP/25% regression-tree blend in two. Regression trees, ranking trees, tree-leaf preprocessing and principal components performed worse under the tested settings; quantile preprocessing remained inconclusive. These secondary comparisons are unadjusted. [Ranking-loss contrasts](finalization/ranking/comparisons.csv) and [hybrid contrasts](finalization/hybrids_cloud_v1/comparisons.csv).

### Simpler pretrained features do not show the same benefit

The separate original test comparison evaluated frozen ESM-2 features and sequence augmentation with Boltz geometry/confidence summaries, rather than internal interface embeddings. Its 67-allele panel covered 5,565 rows. The earlier exposure described in Methods prevents treating this test as wholly untouched.

| Original test procedure | Median Spearman | Difference from sequence ensemble, 95% interval |
|---|---:|---:|
| Peptide + HLA pseudosequence ensemble | 0.7064 | Reference |
| Single sequence network | 0.6181 | −0.0883 [−0.1201, −0.0555] |
| Peptide + full HLA-domain ensemble | 0.6904 | −0.0160 [−0.0388, +0.0043] |
| ESM-2 ensemble | 0.6979 | −0.0085 [−0.0291, +0.0233] |
| Sequence + ESM-2 ensemble | 0.6816 | −0.0248 [−0.0460, +0.0073] |
| Sequence + Boltz geometry/confidence | 0.5947 | −0.1117 [−0.1339, −0.0575] |

The tested geometry augmentation reduced ranking performance; ESM-2 did not establish improvement. Different representations and evaluation populations explain why these findings must remain separate from the positive interface-embedding result. They do not support a claim that protein foundation models uniformly help or fail. [Test scores](stage6_test_summary.csv) and [paired comparisons](stage6_test_paired_ci.csv).

## Discussion

The most defensible conclusion is that learned, position-aware interface representations contain information useful for stability ranking in this benchmark. The evidence combines a competitive sequence comparison, matched nonlinear transfer controls and matched positional ablations. It remains a comparison of complete predictive procedures. Pretraining, molecular context, representation dimensions and fitting choices differ, while the contact-map control provides no evidence for a specific structural explanation. Supervised HLA holdout also does not establish that related structures were absent from foundation-model pretraining.

The conditional intervals quantify variation in the saved assessments, rather than the full uncertainty of the research programme. Nested selection limits leakage within each declared procedure, but repeated use of the same population informed which procedures were studied. The positive main result does not establish a gain above +0.05, and even such a gain would require a separate connection to practical benefit. Rank correlation alone does not demonstrate calibrated half-lives, better immunogenicity prediction or more stable candidates recovered within an experimental budget.

Measurement uncertainty is a second unresolved constraint. The aggregated labels cannot identify replicate reliability, the origin of zero values or a noise ceiling. Candidate public datasets have unresolved construct and measurement lineage and mix assay types; they do not currently supply representative independent confirmation. [Independent-candidate provenance](followup3/independent_data/study_method_status.csv).

The descriptor and head comparisons favour retaining a simpler interface model for the next fixed comparison, while leaving small incremental effects unresolved. The next decisive evidence should come from independent kinetic measurements, with candidate and reference procedures locked before outcomes are opened. A suitable study must define its intended peptide/HLA population, retain raw curves and independent repeats, record measurement failure separately from censoring, and predefine both ranking and an experimental decision endpoint. The [independent assay design](finalization/INDEPENDENT_ASSAY_BRIEF.md) specifies these requirements; its proposed sample sizes are planning assumptions, not demonstrated power.

## Data and model availability

The [dataset documentation](../docs/DATASETS.md) and [evaluation specification](../EVALUATION.md) define the scientific inputs and analysis. Linked CSV/JSON tables provide all assessed arms, predictions, contrasts, sensitivities and frozen protocols. Portable ridge and neural models accept compatible, verified cached Boltz features; their output is an uncalibrated log-scale prediction. The [inference guide](../docs/ROUND2_INFERENCE.md) describes the supported input contract. Generating features for new sequences is a separate computational step.

Feature and cohort hashes bind the stored evidence. A remaining extraction-provenance limitation is that the fanout wrapper's recorded launch digest differs from its retained source digest, so its exact launch-time bytes cannot be certified; the frozen extraction sources and feature checks agree. [Cache provenance](further_research/round2/resources/cache_completion_audit.json). Reproducing saved arithmetic or execution does not establish experimental reproducibility.

## Method references

- Rasmussen et al. (2016). *Pan-Specific Prediction of Peptide–MHC Class I Complex Stability, a Correlate of T Cell Immunogenicity.* [The Journal of Immunology](https://doi.org/10.4049/jimmunol.1600582).
- Harndahl et al. (2011). *Real-time, high-throughput measurements of peptide–MHC-I dissociation using a scintillation proximity assay.* [Journal of Immunological Methods](https://doi.org/10.1016/j.jim.2010.10.012).
- Passaro et al. (2025). *Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction.* [Technical preprint](https://doi.org/10.1101/2025.06.14.659707).
- Dauparas et al. (2022). *Robust deep learning–based protein sequence design using ProteinMPNN.* [Science](https://doi.org/10.1126/science.add2187).
- Delgado et al. (2019). *FoldX 5.0: working with RNA, small molecules and a new graphical interface.* [Bioinformatics](https://doi.org/10.1093/bioinformatics/btz184).
- Cawley and Talbot (2010). *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation.* [Journal of Machine Learning Research](https://www.jmlr.org/papers/v11/cawley10a.html).
