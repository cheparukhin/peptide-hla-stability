# Bounded follow-up studies: scientific contract

13 October 2026. **Proposed scientific contract, to be source-bound before
execution.** Root records the final arm-specific implementation manifests,
resource admissions and any amendment made before assessment. Link status and
findings back to the [canonical plan](../NEXT_RESEARCH_PLAN.md). This document
does not reopen any completed experiment or the original validation/test.

## Common population, outcome and resource boundary

Use only the canonical **19,716 original-training pairs**, with the unchanged
3,587 peptide clusters, five peptide outer folds, 35 HLA groups and five HLA
folds. Identity joins use exact peptide and allele including construct suffixes;
row identifiers are verified against those identities. Original validation/test
outcomes and reserved external outcomes are excluded. Pretrained feature
extraction is label-blind; transformations and supervised steps are fitted only
within their declared source/inner/support partitions.

The outcome is canonical log1p half-life for fitting. Evaluation is the median
of per-allele Spearman correlations on pooled held predictions. Each difference
subtracts two medians on the same eligible panel, rather than taking the median
of allele-specific differences. Use the existing tie/constant-prediction rules.
Primary eligibility is at least 20 query/assessment rows and varying labels;
the at-least-50 panel is a sensitivity. Eligibility is frozen before comparing
arms. Report numbers of rows, clusters, alleles and HLA groups and all exclusions.

The new tranche has a **$139 inclusive maximum**, separately authorized after
the earlier finalization budget closed. Root allocates smaller study caps and
retains operational reserve. Heavy work, feature extraction, fitting and full
bootstrap run on Modal. Local work is orchestration, editing and lightweight
checks. Bound workers, CPU threads, memory, deadlines, startup/storage/reporting
costs and total reservations before launch. No automatic retries, full energy
sweep, new coordinate production, checkpoint search or encoder fine-tuning.

## A. ProteinMPNN complementarity

Freeze up to **64 pairs per allele**, selected by a recorded SHA-256 ordering
of original-training pair identities and no target or feature values. Retain
all pairs when an allele has fewer than 64. This is at most 4,800 rows; it is
an allele-balanced development panel rather than the complete training cohort.
Inherit the existing cluster outer folds even when sampling only part of a
cluster; every occurrence of a retained cluster stays in its existing fold.
Freeze identities before checking new scores. Technical failures are retained
in a failure table and stop the primary assessment; no outcome-based replacement.

Use saved production Boltz structures, the pinned existing ProteinMPNN
checkpoint `v_48_020.pt`, teacher-forced native sequence scoring, zero coordinate
noise and 16 fixed-seed decoding orders. Produce two nine-position feature
blocks: **C**, native-peptide log likelihoods with full HLA/β2m context; and
**P**, likelihoods on the same peptide backbone with all other chains removed.
Record checkpoint/source/coordinate hashes and exact decoding protocol. Use
exactly paired peptide decoding permutations across contexts; equal seeds alone
do not imply equal orders across different lengths.
Totals, means and C−P are diagnostics, not redundant fitted columns.

The new paired-order protocol differs from historical conditional scoring, so
do not reuse historical score values. Reuse the existing coordinates. A new
small extraction pilot (at most 16 predetermined panel rows) is label-free and
checks sequences, chain masks, finite outputs, timing and control construction.
It may authorize the already frozen panel through a cost/validity gate only.
Valid pilot features can be retained; do not duplicate them gratuitously.

Assess exactly six arms: **R0, R0+P, R0+P+C, R5, R5+P, R5+P+C**. R0 is frozen
positional peptide/HLA one-hot and R5 the verified cached contact-pooled Boltz
representation. The primary learner is the existing hidden-64 ReLU MLP recipe,
with linear output, MSE, Adam .001, natural batches 256, at most 200 epochs,
stopping-MSE patience 15 and best weights restored. For both inner and final
fits use exactly two members `(stop_fold=0, seed=0)` and
`(stop_fold=1, seed=1)`, with five cluster-respecting stopping partitions created
within the supplied fitting partition. Fit StandardScaler on gradient rows
only. Average the two member predictions. Each arm receives the same L2 grid
**1, .01, .0001**, three group-respecting inner folds and five unchanged outer
peptide folds. Select L2 by pooled inner median per-allele Spearman; exact ties
choose stronger regularization. This is **600 MLP fits**: six arms × five outer
folds × (three L2 values × three inner folds × two members + two final members).

The ridge sensitivity fits a StandardScaler on all supplied fitting rows and
ridge with an intercept, using the same three inner folds and alpha grid
**1, 100, 10,000**, with the same selection statistic and strongest-alpha tie
rule. It adds **300 ridge fits**: six arms × five outer folds ×
(three alphas × three inner folds + one final fit). Gradient plus stopping
labels count toward each MLP's label budget; ridge uses their full source union.
Save all inner/final predictions, member telemetry, selection tables and
content-bound ancestry for independent prediction/scoring/selection audits.
The reused FeatureHead does not persist fitted scaler/model states; this study
therefore does not promise a weight-level replay audit. No extra supervised
selection, grid expansion or outcome-gated second panel is authorized.

The single primary comparison is **MLP R5+P+C minus MLP R5+P** on the ≥20 panel.
Prespecified secondaries are R5+P+C minus R5, R5+P minus R5, the analogous
three R0 contrasts, and the representation-by-context-gain interaction. The
ridge versions are sensitivity results, not a second primary opportunity.
Show every arm's absolute score; standalone C/P correlations are descriptive.
Use 2,000 shared paired peptide-cluster draws, seed 20261003, for pointwise
95% intervals, the primary 97.5% interval and ≥50 sensitivity. Freeze panel
eligibility on complete assessment data. These intervals condition on fitted
models and selections. They do not include refitting or extraction uncertainty.

## B. Few-shot adaptation to held-out HLA groups and peptide families

Use all **25 existing HLA-fold × peptide-fold joint cells** and three
predetermined support draws. For each target allele and draw, choose one
representative pair per peptide cluster by minimum canonical pair ID, then order
clusters by a separate frozen hash. The first **20 distinct clusters** define
the maximum support set. Budgets **0, 5, 10 and 20** use nested prefixes.
No selection uses outcome values. Require at least **40 distinct clusters**
per target allele before drawing support (20 support clusters plus a minimum
20-cluster query opportunity); retain all otherwise valid source alleles.

Within each held-out HLA fold/draw, remove the union of maximum-support clusters
across its target alleles from **all source fitting and all query rows**.
For cell `(h,p)`, source rows also lie outside HLA fold h and peptide fold p;
query rows lie inside both held folds. Thus support and query peptide clusters
never overlap, even across different target alleles, and the source has never
seen a support family. Queries are identical for every budget and arm within
a draw. Fix one eligible allele panel from the intersection of eligibility
across the three draw-specific queries; do not pool duplicated queries across
draws. Record exact source/support/query row and cluster lists and hashes.

Primary matched source models are fixed **R0 and R5 ridge, alpha 100**, with
source-only StandardScaler and intercept. There are **150 source ridge fits**
(25 cells × three draws × two representations), reused across budgets.
For each target allele, fit a fixed residual ridge adapter using only its k
support measurements and their residuals relative to that cell's source model.
Use source-scaled features normalized by the square root of representation
dimension, fixed alpha **1**, and an intercept. At k=0 return the source
prediction exactly. No target validation, early stopping, hyperparameter
selection, source-model refit or support-label replenishment is permitted.
Fit each allele's adapter only on its own k labels; receiving other target
alleles' labels would change the stated per-allele budget and is excluded.

A bounded strong historical sequence-recipe anchor is included with exactly
the same source/support/query identities: the established original one-hot/BLOSUM
two-layer 256/64 MLP recipe with two fixed source stopping/seed members per
encoding: four members, **300 source MLP fits**, giving **450 total source fits**
with the 150 ridge fits. Its stopping partitions
must respect both grouping axes and the support exclusion. Source stopping
labels count in the source budget. All models receive the same available source
pool; report both this available-label budget and the actual gradient/stopping
union used across source MLP members, which can be smaller because of excluded
joint cross-quadrants. Ridge fits the full available source pool. It uses the
same fixed R0 residual adapter
and target budgets. The arm manifest must pin the existing source MLP recipe;
it is a reduced ensemble historical anchor, **not** the unchanged full main
competitive selector. Do not replace or drop this arm after seeing outcomes.

The primary contrast is **mean_draw(score R5 at k=20 − score R5 at k=0)**.
Secondaries are gains at k=5/10, corresponding R0/sequence gains, absolute
R5−R0 and R5−sequence differences at each budget, and R5−R0 adaptation-gain
interaction at k=20. Average draw-specific scores and contrasts, never their
predictions. Report every draw, absolute learning curves, actual support labels
per allele and summed across target alleles, source fitting/stopping labels,
and the same shared k=0 baseline for each budget within a draw.

Use 2,000 shared draws over the canonical peptide-cluster and HLA-group
universes, seed 20261003. Apply each resampled set of group multiplicities to
each draw-specific query subset, then compute its score before averaging.
The primary interval uses independent crossed peptide/HLA resampling; paired
peptide-only and HLA-only intervals, ≥50 panel and leave-one-actual-HLA-group
deletions are sensitivities. Weight both within-allele observations and allele
aggregation consistently with the existing crossed bootstrap convention.
Keep finite/degenerate-draw diagnostics. Uncertainty conditions on these fixed
support draws, source fits and adapters; it does not establish prospective
assay savings or capture repeated assay/acquisition variation.

## C. Optional repaired FoldX technical pilot

At most **32** predetermined original-training structures, selected by identity
without outcomes, may receive paired unrepaired and RepairPDB-then-interaction
scoring. Pin licensed installed binary, conversion recipe, repair flags and
chain partition. Preserve sequence/construct identities; inspect coordinate
validity, expected interfaces, finite outputs and repair/clash changes. Record
per-case runtimes and failures. No cherry-picked energy window is proof of
biological validity, and energy rank changes after repair are not half-life
rank changes. No half-life assessment, full sweep or new structures follow
automatically. If existing binary/access or bounded costs fail, report the
gate as unresolved without purchasing access.

## D. Independent-data feasibility and common interpretation

Resolve study references, assay type, temperature, physical constructs,
replicates/censoring metadata and exact measurement lineage for the already
reserved public candidates. Use metadata-only projections where files also
contain outcomes. Do not read, summarize, rank, calibrate on or fit to reserved
labels. No external contact or purchase is authorized. Report whether remaining
coherent strata meet the existing assay brief, and the precise missing fields;
small heterogeneous candidate counts do not become adequate by pooling them.

All modelling remains conditional development evidence because original-training
outcomes and prior results influenced these questions. Report two individual
primary 95% intervals; a tranche-wide success statement additionally uses the
two primary 97.5% intervals. Secondary intervals are exploratory. A null does
not prove equivalence or close an entire model class. A positive does not
demonstrate structural causality or independent assay confirmation. Preserve
failed attempts and all declared comparisons, independently recompute scores
and selections from saved predictions, and audit split ancestry before updating
the scientific write-up. Distinguish these checks from fitted-weight replay;
only claim the latter for arms whose states were actually saved and replayed.
