# Statistical methods

The [research report](reports/REPORT.md) presents the findings. This document
defines their populations, comparisons and uncertainty. It describes the
completed assessments; it does not create a new validation study or change
saved protocols, split assignments or predictions.

## Population and separation

The source contains 28,166 peptide–HLA measurements. The fixed
[`data/splits.csv`](data/splits.csv) assigns 19,716 rows to training, 2,817 to
validation and 5,633 to test. Peptides are grouped across all alleles by
single-linkage clustering of nine-residue sequences at Hamming distance ≤3;
whole connected components stay together. Thus no peptide pair separated by
three or fewer substitutions crosses a split. Transitive cluster membership
need not imply that every member is within three substitutions of every other
member. Sequence distance is a leakage-control convention, not a validated
measure of biological similarity.

The current main, transfer and adaptation studies use only the original
19,716 training rows, representing 3,587 peptide clusters. Original validation
and test outcomes do not enter these assessments. The same development
population has informed earlier experiments and subsequent research choices;
clean inner and outer partitions protect the declared fitting procedure but
do not remove that project-wide adaptation.

An earlier diagnostic repartitioned the full source and used 3,350 original-test
rows for diagnostic fitting, 448 for stopping and 585 for scoring. Aggregate
performance was inspected and influenced subsequent development. Those
diagnostic models were separate from the benchmark models. The original test
must therefore not be described as wholly untouched or as a fresh independent
confirmation. New evidence requires independently acquired measurements.

Use `pepstab.data.load_with_splits()` and preserve the supplied assignments.
`pair_id` is the source CSV's zero-based row index; verify it against the exact
`(allele, peptide)` identity, including construct suffixes. Derived tables join
by identity rather than assuming their row order is unchanged. Assay provenance
and the interpretation of recorded zeros are in [DATASETS.md](docs/DATASETS.md).

## Endpoint

Models fit `y_log1p = log(1 + thalf_hours)` unless their protocol specifies a
ranking loss. For each eligible allele, compute Spearman correlation between
predictions and recorded half-life, using average ranks for ties. The primary
statistic is the **median of these within-allele correlations**, calculated
after pooling all outer-held predictions. Alleles receive equal weight. A
comparison subtracts the candidate and reference panel medians; it is neither
the median of allele-specific differences nor the median of fold scores.

Eligibility depends only on assessment labels and row counts, and is shared by
all arms. The main and transfer primary panels require at least 20 rows and two
distinct labels per allele: 69 alleles and 19,647 rows. The ≥50-row sensitivity
panel includes 68 alleles and 19,619 rows. Subsample and few-shot panels use the
same rule on their own assessment populations. In few-shot analyses the panel
is the intersection of eligibility across the three fixed query designs.

A constant prediction has undefined mathematical correlation but contributes
zero under the declared scoring rule; the allele remains in the panel. Labels
with no variation cannot be ranked and are excluded. Nonfinite labels or
predictions are rejected. Bootstrap draws that lose label variation omit that
allele from every arm for that draw and record the resulting degeneracy.

The original validation/test evaluation used ≥20/≥50 rows respectively. These
thresholds belong to that split-based assessment; they do not override the
current development primary panel. Secondary measures include mean absolute
error against recorded `log1p` labels, pooled correlations and precision among
the ten highest predictions at a recorded half-life threshold above two hours.
Prediction ties at the selection boundary receive expected credit. These
endpoints address different questions: ranking alone establishes neither
calibrated half-lives nor prospective experimental savings.

## Main peptide-cluster assessment

Five outer folds retain whole peptide clusters. Within each outer fitting
partition, three inner folds select configurations and procedures. Early
stopping uses further partitions inside the inner fitting data. Outer-held
labels are excluded from fitting, stopping, scaling, centring and selection.
All component procedures and the declared selectors are reported.

The sequence selector considers original and allele-centred MLPs, original and
centred tree regressors, and a pairwise ranker. MLP inputs encode peptide and
the 34-residue HLA pseudosequence using one-hot and BLOSUM features. The MLP
has hidden layers of 256 and 64 units; its screening recipe has 12 members and
its final recipe has 30. Tree procedures use six members. The embedding screen
uses ridge regression and a 64-unit MLP on frozen Boltz representations and
matched sequence controls. Feature MLP screening uses two members; final
assessment uses 30. Regularization grids are fixed before assessment. These
are comparisons of complete procedures, with different dimensions and
training effort; equal hidden-layer width does not imply equal capacity.

For the main study, selection uses the mean of three inner-fold median
correlations and a one-standard-error rule with a fixed preference order.
Standard errors across inner folds serve as a selection heuristic, not an
independent uncertainty estimate. The embedding selector includes the original
sequence MLP as a fallback. It is not the single best component selected from
the outer results.

Scaling and allele-centred targets use fitting data only; neural scalers and
centres use each member's gradient rows. Unseen alleles receive the
fitting-derived global offset. Ranker outputs are mapped to a common scale by
a positive affine transformation fitted to nested crossfit predictions within
the outer fitting population. Its transport from smaller inner fits to final
fits is an additional limitation when pooling predictions across folds.

Exact candidate grids, ensemble recipes and selection order are preserved in
the [sequence protocol](reports/further_research/round2/sequence/protocol.json)
and [embedding protocol](reports/further_research/round2/embedding/protocol.json).
The [main manifest](reports/further_research/round2/results/main/manifest.json)
binds the saved predictions, selections and ancestry.

## Transfer and few-shot adaptation

HLA groups are connected components at Hamming distance ≤3 over the 34-residue
pseudosequence, with engineered and corresponding wild-type names assigned to
the same group. The 35 groups form five balanced HLA folds. HLA holdout assesses
five folds. Joint holdout assesses all 25 HLA-fold × peptide-fold intersections;
fitting excludes both held groups, including the two cross quadrants. Inner
selection and stopping respect the same grouping axes. Final joint neural
members use five diagonal stopping intersections within fitting data and
exclude their cross quadrants. This differs from the 25 outer assessment cells.

Transfer compares pre-nominated R4 ridge and R5 MLP against their matched R0
heads and reports an expanded sequence selector. R0 ridge is additive in
peptide and HLA features: within a fitted model HLA changes only an offset and
cannot change peptide ordering. Nonlinear sequence comparisons therefore carry
more information about allele-specific ranking. Boltz also receives fuller
molecular context and pretrained information. HLA holdout removes supervised
stability labels; it does not demonstrate that a structural family was absent
from foundation-model pretraining. The
[transfer manifest](reports/finalization/transfer_cloud_recovery/results/manifest.json)
identifies the complete final comparison.

Few-shot adaptation uses the 25 joint cells and three fixed support designs.
Within each target allele, 20 distinct peptide clusters define a maximum
support set; budgets of 0, 5, 10 and 20 labels use nested prefixes. The union of
maximum-support clusters across target alleles is removed from both source
fitting and query sets. Query identities are fixed across budgets and arms
within each design. Source fitting also excludes the target HLA and query
peptide folds.

Fixed R0 and R5 source ridge models use alpha 100. An allele-specific residual
ridge adapter uses only that allele's support labels, source-scaled features
normalised by the square root of dimension, alpha 1 and an intercept. No target
label selects hyperparameters or stopping. The sequence anchor is a separate
four-member MLP; it is not the main 30-member selector. Source stopping labels
count toward label access. The primary contrast averages the three
**design-specific score differences** at 20 versus zero labels; predictions
are not pooled across repeated query designs. See the
[few-shot protocol](reports/followup3/fewshot/protocol.json) and
[support manifest](reports/followup3/fewshot/support_manifest.csv).

## Descriptor comparisons

ProteinMPNN and repaired FoldX use the same identity-selected, allele-balanced
panel of 4,449 training pairs, 75 alleles and 2,082 inherited peptide clusters.
Five outer and three inner cluster folds compare matched two-member MLPs;
ridge results are sensitivities. Regularization is selected by the median
within-allele correlation on pooled clean inner-held predictions, with stronger
regularization breaking exact ties. This differs from the main study's
one-standard-error rule and 30-member final MLP.

ProteinMPNN supplies nine position-wise native-residue scores for the peptide
backbone alone (P) and for the full complex (C), averaged over 16 common peptide
decoding orders. The primary contrast is R5+P+C minus R5+P. The key secondary
contrast is R5+P+C minus R5. Repaired FoldX supplies 29 energy/interface
features (E); its primary is R5+E minus R5. The methods are defined in the
[ProteinMPNN protocol](reports/followup3/inverse_execution_v2/protocol.json) and
[energy protocol](reports/followup3/energy/protocol.json).

The full-training ProteinMPNN extension uses 19,716 pairs and the same bounded
head design, with R5+P+C minus R5 as its main comparison. It was chosen after
the balanced-panel results and remains a selected development follow-up,
not independent replication or a fourth prespecified primary. Its
[protocol](reports/followup3/inverse_fulltrain_candidate/protocol.json) also
defines a descriptive comparison on fixed query identities; changes in source
coverage, tuning and stopping prevent attributing that difference solely to
sample size.

## Uncertainty and interpretation

All reported resampling comparisons pair candidate and reference predictions
on the same 2,000 draws, seed 20261003. Main and descriptor assessments resample
whole peptide clusters. Transfer and few-shot primary intervals independently
resample peptide clusters and actual HLA groups; group multiplicities affect
both rows and the allele median. Peptide-only, HLA-only and group-deletion
analyses are sensitivities. Five HLA folds are not five biological sampling
units. Sparse crossed intersections can make these intervals conservative.

These are **conditional resampling intervals**: fitted predictions, selection
outcomes and—where relevant—the three support designs remain fixed. They
exclude uncertainty from refitting the full procedure, repeated model search,
new support acquisition, feature extraction and independent assay measurements.
Nominal 95% coverage for a newly trained or deployed procedure has not been
established. Seed and execution-repeat diagnostics address narrower sources
of variation and cannot supply an assay noise ceiling.

The ProteinMPNN context, 20-label adaptation and repaired-energy primaries form
a three-question family. Their 98.333333% intervals use percentile tails 1/120
and 119/120 for Bonferroni family control at 5%, alongside individual 95%
intervals. The saved earlier 97.5% two-question fields are not the current
family criterion. This adjustment addresses only these three questions,
conditional on the interval procedure; it does not control the project's
entire adaptive search. Other component, transfer and extension comparisons
are exploratory with nominal, unadjusted intervals.

The declared main-study reference gain is **0.05** in median per-allele
Spearman. Preserve it when interpreting that comparison, but do not call it a
universal detection limit, an assay noise floor or an established threshold of
biological value. Detectability depends on the paired predictions and sampling
design; practical importance requires a decision-specific criterion. A positive
lower interval bound supports improvement within the stated uncertainty scope;
a lower bound above 0.05 is required to support a gain exceeding that reference.
An interval crossing zero establishes neither benefit nor equivalence.
