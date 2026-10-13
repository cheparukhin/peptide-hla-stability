# Do protein foundation models help predict peptide–HLA stability?

13 October 2026. The finalization studies below are complete. The subsequently
authorized [third follow-up tranche](../followup3/REPORT.md) is assessed
separately. The few-shot result is complete and independently audited;
ProteinMPNN is complete and independently audited; its context increment
beyond Boltz is inconclusive. Repaired-descriptor complementarity is complete and inconclusive beyond R5;
one selected full-cohort ProteinMPNN extension remains pending.
The [third-primary amendment](../followup3/THIRD_PRIMARY_AMENDMENT.md)
records the pre-efficacy extension and three-primary family control.
Linked protocols and tables retain the methods and every frozen comparison.

**Boltz interface embeddings improve ranking on the examined development data.
Adding ranking loss, trees or preprocessing has not established a further gain.**
The practical recommendation is to retain a simple embedding head and seek
independent kinetic measurements before making deployment claims.

## Question and assessment

The task is to predict how long a peptide remains bound to an HLA molecule.
Dissociation half-life is distinct from binding affinity or immunogenicity.
Following the challenge brief, we ask whether pretrained representations add
useful information beyond supervised sequence models, with fair comparisons
and explicit computation costs. NetMHCstabpan is unsuitable as an independent
benchmark here because its training data overlap the supplied corpus.

Later studies use only the original training partition: **19,716 pairs, 3,587
peptide clusters and 75 alleles**. The endpoint is median per-allele Spearman
correlation on pooled held predictions. The primary panel contains 69 alleles
with at least 20 measurements and varying labels; the ≥50-row sensitivity has
68. Differences subtract two medians on the same panel. Fitting, stopping,
scaling and inner model selection respect the declared holdout groups.

The [original hackathon report](../REPORT.md) evaluated different, simpler
sequence-plus-geometry/confidence augmentation against the sequence ensemble:
its test difference was −0.112
[−0.134,−0.057]. An earlier diagnostic had exposed some original test rows, so
that historical assessment was not wholly untouched. **The later studies did
not reopen the original test**; they remain development work informed by the
project's history. Richer embeddings do not erase the earlier negative result.

## What the representations contribute

The main nested embedding procedure scores **0.7212**, versus **0.6668** for a
competitive sequence selector: **+0.0543 [0.0314,0.0808]**. The positive interval
supports improvement on this cohort. Its lower bound is below the predeclared
+0.05 worthwhile-gain threshold, so that stronger criterion was not met. The
≥50-row sensitivity agrees. [Main assessment](../further_research/round2/results/main/REPORT.md).

Position-preserving representations outperform matched tiled pooling, supporting
useful positional information. However, the designated contact map versus its
shuffled control gives only **+0.0007 [−0.0096,0.0236]**. This does not establish
a benefit from that particular contact map or a causal structural mechanism.
Clean sequence/embedding blending also improves development scores, without
clearing the worthwhile-gain threshold. [Representation and complementarity
results](../further_research/round2/REPORT.md).

Stronger transfer controls address whether earlier advantages came from unequal
supervised learners. Across 30 cells and 10,830 retained fits, the matched MLP
comparisons remain positive:

| Held-out groups, ≥20 panel | R5−R0 MLP difference [crossed 95% interval] | Original / expanded sequence scores |
|---|---:|---:|
| HLA groups | +0.4062 [0.3073,0.4733] | 0.1709 / 0.1921 |
| Peptide clusters and HLA groups | +0.3738 [0.3092,0.4638] | 0.1350 / 0.1085 |

Matched ridge comparisons, ≥50-row sensitivities and every single-HLA-group
removal also remain positive. Both sequence references are shown because the
expanded selector has a lower joint point estimate. These results strengthen the
representation comparison, but neither establish structural causality nor
retrospectively pass the main study's separate practical threshold.
[Complete transfer results and audit](transfer_cloud_recovery/results/REPORT.md).

## Does a more complicated head help?

The **840-fit ranking-loss study** adds within-allele RankNet to a matched MLP.
R5's difference is **+0.0071 [−0.0051,0.0098]**; the representation-by-loss
interaction also spans zero. The frozen inner deployment rule retains MSE in
all five R5 folds. A small benefit remains unresolved; this is not equivalence.
[Ranking report](ranking/REPORT.md).

The **480-fit hybrid study** tests regression trees, ranking trees, tree-leaf
MLPs, quantile preprocessing, PCA and selected blends on both sequence and
Boltz features. Its primary selector scores **0.7039 versus 0.6995** for the
matched two-member R5 MLP: **+0.0044 [−0.0055,0.0100]**. The ≥50-row result is
also inconclusive. It selects the standard MLP in three folds and a 25% tree
blend in two. Selector and blend predictions coincide; they are one finding.

Regression/ranking trees, tree-leaf preprocessing and PCA reduce R5 scores in
both panels; quantile preprocessing is inconclusive. Secondary intervals are
exploratory and unadjusted. These findings concern the frozen recipes, not all possible trees
or preprocessing methods. Trees consumed about 95% of summed learner fitting
time. The fixed two-member sequence head is weaker than the main study's
competitive selector, so the hybrid representation contrast must not replace
the main comparison. [All hybrid results](hybrids_cloud_v1/REPORT.md),
[coverage audit](hybrids_cloud_v1/coverage_audit.json).

## What changes when fewer labels are available?

The 120-fit label-budget study shows a larger matched Boltz advantage at lower
budgets. Three predetermined nested subsets were averaged **as scores**, with
stopping labels included. Full-budget members were reused after identity checks.

| Outer-fitting peptide-cluster budget | Sequence MLP | Boltz MLP | Gap |
|---|---:|---:|---:|
| 25% | 0.2640 | 0.5877 | +0.3237 |
| 50% | 0.4010 | 0.6504 | +0.2494 |
| 100% | 0.5373 | 0.6995 | +0.1622 |

The primary increase in the gap at 25% versus 100% is **+0.1615 [0.1333,0.2041]**;
the 50% budget and ≥50-row sensitivity agree. Both procedures worsen with fewer
labels, but R5 loses less performance. This supports relative robustness under
previously nominated fixed recipes, not prospective assay savings or a comparison
against the stronger main sequence selector. [Label-budget results and protocol](label_efficiency/REPORT.md).

## Can a few target-allele measurements improve transfer?

The [new few-shot study](../followup3/REPORT.md#completed-few-shot-result) gives
20 support measurements per target allele to fixed source ridge predictors,
with support/query peptide families and target HLA groups excluded from source
fitting. R5 improves from **0.529583 to 0.576337**: **+0.046754
[0.022657,0.103467]** under the individual crossed 95% interval. Its
**98.333333% three-primary family interval is [0.014042,0.119457]**. This
supports useful conditional adaptation on this development design.

Three predetermined support draws are averaged as scores. Each uses 1,360
target labels across 68 alleles at k=20, with no target tuning/stopping labels;
source stopping labels count too. The ≥20 and ≥50 panels are identical, so
their equality is no separate robustness result. Every HLA-group deletion
retains a positive primary point difference. R0 and the reduced four-member
historical sequence anchor adapt more in nominal secondary comparisons, but
R5 remains stronger in absolute score. The sequence anchor is distinct from
the full main competitive selector. Different source exclusions, queries and
label access prevent direct comparison with the main score or label-budget
study; no prospective assay savings are established.

An exploratory [offset diagnostic](../followup3/fewshot_offset/REPORT.md),
chosen after the positive primary, uses the saved adapter's scalar offset
without its feature term. R5 scores 0.531987 with offsets alone, versus
0.529583 zero-shot and 0.576337 with full adaptation. Full minus offset is
+0.044350 [0.016478,0.092778]; offset minus zero is +0.002404
[−0.006172,0.023880] under nominal unadjusted 95% intervals. Feature-dependent
correction adds ranking beyond this cell-alignment control on saved predictions;
offset-only benefit is unresolved. No new fitting or labels are used, and the
three-primary family is unchanged. This does not establish a biological mechanism.

All 450 source fits and 9,180 adapters passed cloud exact saved-state replay.
The [independent numerical audit](../followup3/fewshot/independent_audit.json)
checks predictions, arithmetic, coverage, residual offsets and ancestry without
independent refitting or bootstrap regeneration. ProteinMPNN is complete with an inconclusive increment beyond Boltz;
repaired-descriptor complementarity is complete and inconclusive; the pre-efficacy three-primary amendment governs
tranche-wide claims. Independent assay confirmation remains open.

## Does native-sequence compatibility add to Boltz?

The completed [ProteinMPNN assessment](../followup3/REPORT.md#completed-proteinmpnn-result)
uses a balanced 4,449-pair original-training panel and six matched arms.
Its primary MLP context increment, R5+P+C minus R5+P, is **+0.023960
[−0.014736,0.058359]**; the required **98.333333% family interval is
[−0.021942,0.065942]**. The net R5+P+C minus R5 increment is **+0.018816
[−0.016250,0.056637]** under its exploratory 95% interval, from 0.600708
to 0.619524; R5+P alone scores 0.595564. Neither comparison establishes an
additional gain beyond the matched Boltz MLP, nor equivalence.

R0 context controls are positive exploratory comparisons, while R5 ridge
increments remain inconclusive. The ≥20 and ≥50 panels contain 69 and 68
alleles respectively. All 900 fits and parser-recovery lineage passed
independent arithmetic, selection and ancestry auditing; these heads do not
save fitted states, so fitted-weight replay is not claimed. All arms and
contrasts remain visible in the source-linked report. The balanced panel and
bounded learner differ from the main selector and few-shot ridge study;
these scores cannot directly replace or be subtracted from their results.

## Do repaired FoldX descriptors add to Boltz?

The [completed repaired-descriptor assessment](../followup3/REPORT.md#completed-repaired-descriptor-result)
uses the same balanced panel, reusing R0/R5 baselines exactly. MLP R5+E
minus R5 is **+0.014991 [−0.043512,0.036368]**, with a **98.333333%
family interval [−0.052436,0.045026]**. Scores rise from 0.600708 to 0.615698;
ridge adds +0.006510 [−0.023760,0.020296]. Both learners and ≥50 sensitivities
remain inconclusive, without establishing equivalence. The exploratory weak
R0 ridge benefit does not establish extra incumbent R5 value. All 4,449
extraction records, 200 new MLP members and 100 ridge fits pass independent
recovery, exact feature-matrix and assessment audits; 4,376 inherited success
hashes are preserved and 73 interrupted cases recovered. No saved-weight replay
is claimed. All arms and contrasts remain source-linked in the report.

The [balanced-panel figure](../followup3/figures/descriptor_comparison.svg)
shows all eight absolute MLP scores and both descriptor primary family intervals.
Few-shot adaptation is the sole positive primary under the original three-question
family correction. These conditional development results do not establish
kinetic mechanism or independent assay confirmation.

After both audits, the [frozen full-cohort decision](../followup3/FULL_COHORT_DECISION.json)
admits one final ProteinMPNN extension over all 19,716 original-training pairs:
R5, R5+P and R5+P+C, with 300 MLP members and 150 ridge fits. The main
comparison is MLP R5+P+C minus R5; results remain pending. A descriptive
old/new prediction comparison on the original 4,449 query rows adds no fit or
confidence interval. Source coverage, inner selection and stopping change
together, so it does not isolate a causal sample-size effect. This selected
development transport check is bounded by $40 and is neither independent
confirmation nor a fourth primary in the original family. No full FoldX
extension, combined arm or further grid is admitted. The
[label-free cohort comparison](../followup3/inverse_fulltrain_metadata/cohort_metadata_comparison.json)
keeps the same 75 alleles: rows grow 4,449 → 19,716 (4.43×), while inherited
clusters grow 2,082 → 3,587 (1.72×). Coverage and within-cluster density both
change; row expansion is not proportional independent evidence. Label-free
≥20/≥50 row counts do not establish assay-rankability eligibility.

## Cost, reproducibility and remaining evidence

Portable R4 ridge and R5 MLP packages predict from verified cached features.
Warm single-row medians are 1.56 and 2.40 ms on the recorded host; these exclude
feature extraction, startup and cache reads. They are not new-sequence latency
or calibrated physical half-life guarantees. [Model guide](../../docs/ROUND2_INFERENCE.md).

The original full-cohort folding cost was **$212.71 derived from container-hours**,
not a reconciled invoice. Later round-two cloud work carries **$76.84**
conservatively. Finalization's closed compute carries **$8.61**, plus a retained
$3 overhead allowance within the additional $100 ceiling. All compute and
temporary finalization storage are closed; historical caches remain. Cached-feature experiments are cheap
relative to obtaining representations for new pairs. [Original costs](../REPORT.md#6-compute-cost)
and [current accounting](execution_manifest.json).

The successful hybrid audit reproduced every held prediction and all 328
stopping-MSE scalars exactly under the recorded numerical backend; failed
attempts and report-only corrections remain disclosed. The inherited extraction
wrapper has a separate launch-source provenance gap. [Replay details](cloud/BACKEND_REPLAY.md)
and [round-two provenance](../further_research/round2/REPORT.md).

Intervals use 2,000 paired cluster draws; transfer also crosses peptide and HLA
groups. They condition on fitted models, selections and fixed label subsets,
excluding refitting, subset/selection repetition and independent assay variability. No adequate reserved
independent half-life cohort or assay-noise ceiling is available. The completed
[metadata/methods audit](../followup3/independent_data/PROVENANCE.md) preserves
129 candidates across nine alleles and 36 cited studies. Primary methods reveal
heterogeneous endpoints, temperatures, physical constructs and unresolved
measurement mappings. The largest single-study/allele stratum has only 23 rows
(Strønen 2016, A*02:01), with original methods and measurement lineage still
unverified. Its possible narrow external assessment remains gated; it cannot
provide representative multi-allele confirmation. External candidate outcome columns remain
reserved. Incidental visibility of public paper results is disclosed in that audit. The
[unsent assay brief](INDEPENDENT_ASSAY_BRIEF.md) specifies what confirmation
requires. The evidence warrants further controlled evaluation of interface
embeddings, while favoring the simpler head for the next frozen comparison.
