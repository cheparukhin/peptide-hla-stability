# Finalisation readout: peptide–HLA stability

12 October 2026. **Portable inference, matched ranking loss and stronger transfer
controls are complete; the 480-fit hybrid study is running.** This report combines the closed
round-two record with completed finalisation deliverables. It does not replace
the frozen protocols or historical reports.

Boltz interface embeddings improve ranking on the examined development cohort
against a selected sequence comparator. The additional within-allele ranking
loss produces a small, uncertain gain and is not selected for R5 deployment.
The transfer advantage survives stronger sequence controls and matched ridge/MLP
recipes. Two portable cached-feature prediction recipes are ready. Representative
independent assay confirmation remains open.

## Assessment and historical distinction

The later studies use the **original training partition only**: 19,716
peptide–HLA pairs, 3,587 peptide clusters and 75 alleles. The primary panel has
69 alleles with at least 20 measurements and varying labels; the ≥50-row
sensitivity has 68 alleles. Peptide-cluster, HLA-group and joint holdouts
answer different questions; their scores cannot be subtracted to identify a
causal representation effect.

The endpoint is the median of per-allele Spearman correlations on pooled held
predictions. A difference subtracts two such medians on the same eligible
panel, rather than taking a median of allele differences or fold scores.
Target ties receive average ranks; otherwise eligible alleles with constant
predictions contribute zero.

The [4 October hackathon report](../REPORT.md) retains its original held-out
test pass. Its simple geometry features and assessment differ from the later
interface embeddings and development holdouts. **The original test was not
reopened for these follow-ups.** Earlier development and benchmark/test
results influenced the research history; preserving that test does not make
the later training-cohort studies independent confirmation.

## Completed round-two findings

| Completed assessment | Embedding procedure | Sequence comparator | Difference [95% interval] |
|---|---:|---:|---:|
| Main peptide-cluster holdout | 0.721163 | 0.666841 | +0.054322 [+0.031395, +0.080805] |
| HLA-group holdout, prior bounded comparator | 0.546303 | 0.173961 | +0.372342 [+0.290338, +0.453269] |
| Joint peptide/HLA holdout, prior bounded comparator | 0.506675 | 0.110831 | +0.395844 [+0.295075, +0.445743] |

The main procedure selects against competitive sequence procedures inside
fitting partitions and preserves the original-sequence fallback. Its 2,000
paired whole-peptide-cluster draws support a positive conditional development
gain. The lower bound is below +0.05, so the predeclared worthwhile-gain
criterion is not met. The ≥50-row sensitivity is +0.056352
[+0.031732, +0.079137]. [Main report](../further_research/round2/results/main/REPORT.md)
and [saved comparisons](../further_research/round2/results/main/paired_comparisons.csv).

Position-preserving controls outperform matched tiled pooling: the R3 MLP
difference is +0.338713 [+0.302942, +0.377253], and the full 9×34 positional MLP
difference is +0.166349 [+0.137791, +0.199165]. R5 versus shuffled contact
membership is only +0.000676 [−0.009556, +0.023590]. These support useful
positional information within the frozen representations, without establishing
a benefit from the particular crystal contact map or a causal structural mechanism.

Clean nested complementarity also supports conditional gains. The R4 ridge
blend with the exact original sequence reference scores 0.723789, a difference
of +0.056948 [+0.042591, +0.083811] against the competitive sequence selector.
R4 residual correction scores 0.711429, with +0.044588
[+0.030379, +0.067056]. Neither lower bound exceeds +0.05. Proximity between
component scores is not a tested superiority claim.
[Complementarity report](../further_research/round2/complementarity/results/REPORT.md).

The prior transfer assessments cover all five HLA folds and 25 joint cells.
Their crossed intervals resample peptide clusters and HLA groups. Positive
lower bounds above +0.05 apply to these complete procedures against the
**narrow prior sequence comparators**, which exclude the main screen's stronger
tree/ranker families. Representation and training-recipe effects are combined.
Raw inner vectors and numerical scaler statistics were not independently
reconstructed in that prior transfer audit.
[Prior transfer report](../further_research/round2/transfer/results/README.md).

## Completed matched ranking-loss follow-up

The frozen experiment compares the same 64-unit MLP recipe on R0 onehot860
and R5 Boltz1024 features, with MSE versus MSE plus within-allele RankNet.
Ranking weights 0.1, 0.3 and 1 cross L2 values 1, 0.01 and 0.0001. Selection
uses three inner peptide-cluster folds and two members per candidate/fold;
selected final procedures average 30 members. Natural batches, Adam 0.001,
200-epoch cap, patience15 and **MSE stopping** are fixed. Small ranking weights
then stronger L2 define the one-standard-error preference order.

| Frozen comparison, ≥20 rows | Difference | Paired 95% interval |
|---|---:|---:|
| R5 mixed ranking loss − R5 MSE, primary | +0.007072 | [−0.005051, +0.009829] |
| R0 mixed ranking loss − R0 MSE | +0.002891 | [−0.007704, +0.012037] |
| Loss-by-representation interaction, ΔR5 − ΔR0 | +0.004181 | [−0.012532, +0.011938] |

R5's pooled score rises from 0.720986 to 0.728058. The ≥50-row difference is
+0.005523 [−0.003946, +0.008494]. **Neither panel establishes an additional
ranking-loss gain, and an inconclusive interval does not establish equivalence.**
The deployment selector preferred MSE within one fold-standard-error using
inner scores only; it retained MSE in all five R5 folds. R0 selected ranking
only in fold2, and its pooled deployment difference was −0.001794
[−0.006063, +0.004763].

All 840 new scientific fits and 480 reused MSE logical members passed the
source, cohort, feature, identity and ancestry audit. Both separately
authorized real zero-weight validation members reproduced the cached MSE
predictions exactly. The completed report reconstructed every new member's
prediction, gradient-only scaler mean/variance/scale and stopping curve/epoch,
then replayed inner selections. Independent arithmetic checks covered both
panels, all contrasts and every fallback. The interaction uses the same
2,000 paired draws across all four arms before forming its percentile interval.
[Protocol](ranking/protocol.json), [result report](ranking/REPORT.md),
[comparisons](ranking/comparisons.csv) and [completion audit](ranking/completion_audit.json).

MSE anchors the log1p half-life scale; no affine mapping or pure-ranking claim
is made. Feature dimensions and parameter counts differ, and two-member
screening differs from the final 30-member recipe. Intervals condition on the
fitted predictions and inner selections, excluding selection repetition and
independent assay variation. Historical package versions were not recorded;
exact sources/features and both exact real member checks provide bounded
compatibility evidence. No outcome changed the grid or stopping rule, and the
ranking result authorizes no automatic extension.

## Completed portable inference

Two recipes were locked before new follow-up outcomes and refit on original
training rows only: **R4 ridge alpha100** and **R5 MLP L2 0.01**, with 30 members.
This used 31 local fits and no cloud calls. They are deployment demonstrations,
without a new predictive assessment or a universal-winner claim.

Each model directory contains a pickle-free `parameters.npz`, `model.json`
and `complete.json`; all three travel together. Both exports passed exact
save/reload prediction equality and finite-output checks. Real novel
peptide/HLA inputs require compatible Boltz extraction; no sequence-to-prediction
output is simulated in the project page.

| Recorded-host warm performance | Single row median / p95 | Batch64 median / p95 |
|---|---:|---:|
| R4 ridge alpha100 | 1.559 / 1.799 ms | 1.651 / 1.843 ms |
| R5 MLP L2 0.01 | 2.401 / 2.565 ms | 3.870 / 4.079 ms |

This is a performance-only in-sample benchmark on identified training-cache
inputs. Calls include package/feature validation, scaling, inference and
output construction. They exclude cache-file reads, startup, MSA generation,
Boltz extraction and network transfer; they are not new-sequence end-to-end
latency or deployment-cost estimates. Compatible feature caches remain
external to Git. [Usage and input contract](../../docs/ROUND2_INFERENCE.md),
[exports and timing](inference/README.md),
[distribution manifest](inference/distribution_manifest.json).

## Completed stronger transfer controls

The frozen extension adds sequence regression/centred-target trees, a fixed
sequence ranker and matched R0 ridge/MLP heads to five HLA cells and 25 joint
cells. Selection, stopping and scaling respect each regime's inner groups.
All 30 cells and 10,830 retained scientific fits passed the full semantic audit;
the once-only Modal report and an independent saved-table audit are complete.
The experiment reuses previously nominated R4/R5 procedures; these are
conditional development comparisons, not a new independent confirmation.

| Regime / panel | Original sequence | Expanded sequence selector | R4 ridge | R5 MLP | Embedding selector |
|---|---:|---:|---:|---:|---:|
| HLA / ≥20 | 0.170935 | 0.192134 | 0.549781 | 0.566719 | 0.546303 |
| HLA / ≥50 | 0.163120 | 0.191836 | 0.548667 | 0.565128 | 0.543417 |
| Joint / ≥20 | 0.134985 | 0.108454 | 0.512678 | 0.497998 | 0.506675 |
| Joint / ≥50 | 0.130702 | 0.107051 | 0.510709 | 0.492374 | 0.504977 |

Both original and expanded sequence procedures are reported. The expanded
selector's ≥20 difference from the original is +0.021199 [−0.064630,+0.073387]
in HLA and −0.026531 [−0.088797,+0.018463] in joint holdout, using crossed
95% intervals. Its joint point estimate is worse, and every single-group
removal leaves that difference negative, although the crossed interval spans
zero. The outer results do not authorize choosing a different selector.

The matched heads address the earlier training-recipe concern directly:

| Regime / panel | Matched contrast | Difference [crossed 95% interval] |
|---|---|---:|
| HLA / ≥20 | R4 − R0 ridge | +0.465509 [+0.395953, +0.553410] |
| HLA / ≥50 | R4 − R0 ridge | +0.467516 [+0.395092, +0.553011] |
| Joint / ≥20 | R4 − R0 ridge | +0.463260 [+0.380234, +0.524616] |
| Joint / ≥50 | R4 − R0 ridge | +0.461998 [+0.378981, +0.524035] |
| HLA / ≥20 | R5 − R0 MLP | +0.406234 [+0.307286, +0.473273] |
| HLA / ≥50 | R5 − R0 MLP | +0.405422 [+0.304430, +0.473766] |
| Joint / ≥20 | R5 − R0 MLP | +0.373784 [+0.309165, +0.463816] |
| Joint / ≥50 | R5 − R0 MLP | +0.386285 [+0.307505, +0.463886] |

Every matched contrast also has positive peptide-only and HLA-only intervals,
and remains positive after removing any one of the 35 HLA groups. R4, R5 and
the embedding selector also exceed both original and expanded sequence
references in each regime/panel, with positive crossed intervals. Thus the
advantage is not explained solely by the earlier unmatched head recipes or a
single influential HLA group. Different feature dimensions, pretraining and
representation construction still prevent causal attribution to structure.
The transfer findings do not retrospectively satisfy the main study's separate
worthwhile-gain criterion.

The report contains 264 comparisons (22 contrasts × two regimes × two panels
× three uncertainty modes), twelve 2,000-draw paired bootstrap tables and
3,080 group-deletion rows. All output hashes, pooled cohort coverage and delta
arithmetic passed independent lightweight checks. Intervals condition on fitted
predictions and inner choices; they omit repeated fitting/selection and assay
variation. Crossed resampling is a dependence sensitivity and can be conservative.
Group deletion does not retrain models. HLA groups are a declared sequence
convention, not validated functional distance. The fixed ranker's affine
inner-to-final scale transport remains a procedure limitation.

The final 2,000 fits ran on Modal, preserving 8,830 earlier fits byte-identically.
Known discarded completed fits are 2,579, giving 13,409 known completed fits
including the retained study. Twelve uncertain original remote attempts and
any unrecorded interrupted local fits are separate; engineering fits are excluded.
The successful recovery and report compute/storage are closed.
[Full transfer readout](transfer_cloud_recovery/results/REPORT.md),
[all comparisons](transfer_cloud_recovery/results/comparisons.csv),
[coverage audit](transfer_cloud_recovery/results/coverage_audit.json),
[frozen specification](transfer_controls/SPECIFICATION.md) and
[recovery record](RECOVERY_OPERATIONS.md).

## Newly authorized hybrid follow-up

**The user has now authorized bounded execution of the hybrid experiments; no Boltz tree result is available yet.**
The completed transfer trees and ranker consume sequence features; they do not
test boosted trees or tree ranking on Boltz embeddings. The small, uncertain
MLP ranking result does not rule out a benefit from a different learner.

The frozen study asks whether any of these additions improves a matched Boltz
MLP on held peptide clusters. Its **single primary comparison** is the
inner-selected R5 hybrid procedure minus the R5 standard MLP. Both use the same
two-member schedule; the earlier 30-member MLP is a descriptive reference,
not the matched control.

| Scientific question | Frozen comparison |
|---|---|
| Do trees exploit the embeddings better? | Regression and ranking trees versus the matched MLP, on both R0 sequence and R5 Boltz features. |
| Does ranking help a tree specifically on Boltz features? | Ranking tree minus regression tree within each representation, plus their paired interaction. |
| Do trees and the MLP make complementary predictions? | A blend selected from clean inner predictions versus the standard MLP. |
| Does preprocessing help the neural head? | Quantile, PCA and supervised tree-leaf inputs versus the standard MLP. |

All six fitted components are evaluated even if the inner selector rejects
them. The study has 480 scientific fits and a separate 12-fit unscored runtime
pilot; pilot models never enter the comparison. Secondary intervals are
exploratory, and every result remains development evidence. The separate 12-fit unscored pilot and saved-model replay passed on Modal.
Its runtime and memory measurements alone set a four-hour main window. The
480 scientific fits launched once at 20:51 UTC on 12 October with two workers,
two numerical threads each and four Modal CPUs. The fixed work stop is
13 October 00:55 UTC; completion is not guaranteed. No main outcome is claimed
until the saved-model, ancestry and statistical report passes. See the [scientific specification](hybrids_cloud_v1/SPECIFICATION.md)
and [operational continuation](cloud/CONTINUATION.md).

The existing MLP already standardizes features using gradient-fitting rows
only. Additional normalization or tree-derived preprocessing is a separate
change, not an explanation established by the current results. Any learned
tree predictions, leaf encoding, scaler and downstream head must preserve
cluster-separated fitting, stopping and selection ancestry; outcome-fitted
tree features cannot be generated once on the full cohort and reused as clean
assessment inputs. Tree leaves are supervised features, not a generic way to
normalize a distribution. The separate quantile and PCA controls test simpler
transformations directly. A positive result would support that complete
procedure, rather than proving why its preprocessing works.

## Independent confirmation and provenance limits

The public stability release overlaps the canonical corpus. Conservative
exclusions retain only 129 candidate rows across heterogeneous assays and nine
alleles, with only one allele above 20 rows and unresolved construct/endpoint
details. Their outcomes remain reserved. This is not a representative
independent confirmation cohort; supplied aggregate measurements cannot
establish an assay-noise ceiling.

An [unsent assay request brief](INDEPENDENT_ASSAY_BRIEF.md) specifies kinetic
dissociation half-life, physical constructs, raw time courses, censoring,
measurement lineage and independent repeats. It proposes separately quoted
12-allele panels of 200/400/800 distinct pairs per allele, with repeats on a
stratified 20% subset. These are conditional planning alternatives, not a
power guarantee. No assay quote, partner commitment, funded measurements or
adequate independent cohort is secured. Candidate procedures and analysis
must be frozen before opening confirmation outcomes.

Locked identical-setting execution repeats matched on the recorded platform;
they do not establish cross-platform equivalence, assay reproducibility or
model-selection reproducibility. The inherited cache fanout wrapper's
launch-authorized and current SHA-256 digests differ. Frozen extraction
sources, feature protocol and local feature audits passed, but exact
launch-time wrapper bytes cannot be certified. The portable exports and new
ranking audit do not repair that historical gap.
[Closed round-two assessment](../further_research/round2/REPORT.md).

The closed round-two phase conservatively accounted for $76.8388 against its
$100 ceiling, including the retained $2 engineering allowance. This is not a
provider invoice or live balance. Finalisation has a separately authorized
additional ceiling. Its closed compute scopes carry $6.809382 conservatively, including uncertain
calls, preemptions, startup diagnostics and the completed recovery. The recovery
provider snapshot is $0.878851; $1 is retained conservatively for that run.
The completed pilot and transfer report carry $0.10 and $0.15 within the shared
$35 finalization envelope. Remaining main/report caps are $20/$5, plus $3
reserved for storage, egress and build overhead. The combined current maximum
commitment is $34.809382,
within the user's $100 new-phase ceiling. This new-phase liability is separate
from the old accounting and is not a live balance.
[New-phase closure](resources/transfer_main/interruption_20261012/closure.json)
and [prior resource closeout](../further_research/round2/resources/runtime_cost_summary.json).

## Local delivery and verification

The [project page](../../site/index.html) and [README](../../README.md) separate
the original film/test archive from later development findings. Their update
is local; this work does not publish the site or submit externally. Site
numbers and source hashes are generated from saved evidence:

```sh
python3 site/round2-build.py --check
.venv/bin/python -m pytest -q tests/test_round2_inference.py
```

The ranking [report exit receipt](ranking/report.exit.json) records the single
frozen report invocation; its completion audit retains independent arithmetic
and artifact checks. Completed expensive scopes must not be blindly relaunched.
This readout remains provisional for the running hybrid comparison and
independent confirmation. The transfer extension is complete.
