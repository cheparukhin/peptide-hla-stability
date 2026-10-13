# Do protein foundation models help predict peptide–HLA stability?

13 October 2026. Core studies are complete; the bounded label-budget follow-up
is running. This report summarizes the evidence; linked protocols and tables
retain the full methods and every frozen comparison.

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

| Frozen R5 challenger versus standard MLP, ≥20 | Difference [95% interval] |
|---|---:|
| Regression tree | −0.0571 [−0.0862,−0.0373] |
| Ranking tree | −0.0500 [−0.0761,−0.0251] |
| Tree-leaf MLP | −0.1343 [−0.1606,−0.1055] |
| PCA MLP | −0.0520 [−0.0660,−0.0252] |
| Quantile MLP | −0.0093 [−0.0159,0.0168] |

Both panels show the same pattern; secondary intervals are exploratory and
unadjusted. These findings concern the frozen recipes, not all possible trees
or preprocessing methods. Trees consumed about 95% of summed learner fitting
time. The fixed two-member sequence head is weaker than the main study's
competitive selector, so the hybrid representation contrast must not replace
the main comparison. [All hybrid results](hybrids_cloud_v1/REPORT.md),
[coverage audit](hybrids_cloud_v1/coverage_audit.json).

A separate, fixed 120-fit study is assessing how the R5−R0 gap changes with
25%, 50% and 100% of fitting peptide clusters. Its subsets and recipes are
frozen, stopping labels count, and the full-budget models are reused only after
exact identity checks. No result is available yet. It cannot establish
prospective assay savings. [Protocol](label_efficiency/SPECIFICATION.md).

## Cost, reproducibility and remaining evidence

Portable R4 ridge and R5 MLP packages predict from verified cached features.
Warm single-row medians are 1.56 and 2.40 ms on the recorded host; these exclude
feature extraction, startup and cache reads. They are not new-sequence latency
or calibrated physical half-life guarantees. [Model guide](../../docs/ROUND2_INFERENCE.md).

The original full-cohort folding cost was **$212.71 derived from container-hours**,
not a reconciled invoice. Later round-two cloud work carries **$76.84**
conservatively. Finalization's closed compute currently carries **$8.51**, with
$3 overhead reserved and a separate inclusive $10 label-study cap, all within
the unchanged $100 additional ceiling. Cached-feature experiments are cheap
relative to obtaining representations for new pairs. [Original costs](../REPORT.md#6-compute-cost)
and [current accounting](execution_manifest.json).

The successful hybrid audit reproduced every held prediction and all 328
stopping-MSE scalars exactly under the recorded numerical backend; failed
attempts and report-only corrections remain disclosed. The inherited extraction
wrapper has a separate launch-source provenance gap. [Replay details](cloud/BACKEND_REPLAY.md)
and [round-two provenance](../further_research/round2/REPORT.md).

Intervals use 2,000 paired cluster draws; transfer also crosses peptide and HLA
groups. They condition on fitted models and selection, excluding refitting,
selection repetition and independent assay variability. No adequate reserved
independent half-life cohort or assay-noise ceiling is available. The
[unsent assay brief](INDEPENDENT_ASSAY_BRIEF.md) specifies what confirmation
requires. The evidence warrants further controlled evaluation of interface
embeddings, while favoring the simpler head for the next frozen comparison.
