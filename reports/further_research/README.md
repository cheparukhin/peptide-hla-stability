> **Historical study record.** The protocols, results and future-tense instructions below describe the completed October study. The [current project review and research plan](../NEXT_RESEARCH_PLAN.md) is the sole forward work queue; its [dated correction register](../NEXT_RESEARCH_PLAN.md#dated-correction-register--12-october-2026) supersedes conflicting interpretation, selection and blend-gate claims here. No old protocol should be relaunched as a new study without the current evaluation gates.

# Further research: namespace index and decision log

Follow-up study to the frozen benchmark, executing
[FURTHER_RESEARCH_PLAN.md](../archive/plans/further-research-2026-10-04.md). Opened 4 October 2026.

**This study does not change the frozen benchmark, its splits, or its published
conclusions.** It scores neither the test split nor the validation split. All
four phases assess on training-only nested cluster validation. Original reports
and prediction files are preserved; nothing here overwrites them.

## Shared protocol

One document binds every phase:
[`phase_a_prespecification.md`](phase_a_prespecification.md). It records every
grid, budget, eligibility rule and selection rule **before** the first fit, as
the plan's §3.3 requires, and its §7 logs each amendment with a date and a
reason. Phases B, C and D inherit from it rather than declaring their own
versions of the same thing.

The points all four phases hold in common:

| Decision | Value |
|---|---|
| Outer assessment | 5 whole-cluster folds over the training split, `scripts.baseline_ensemble.cv_folds(train, 5)` |
| Inner selection | cluster folds inside each outer fitting partition; 3 for screening, 5 at assessment |
| Allele eligibility | ≥ 20 rows and ≥ 2 distinct labels per fold, with ≥ 50 reported as a declared sensitivity check |
| Primary metric | median per-allele Spearman, frozen tie and unranked rules, via `pepstab.evaluation` verbatim |
| Intervals | paired peptide-cluster bootstrap, 2,000 resamples, seed `20261003`, read with EVALUATION.md's six verdicts |
| Floor event | `thalf_hours == 0` (the recorded assay-floor code) |
| Ensemble parity | every arm at the reference's arity — 5 inner folds × 2 encodings × 3 seeds = 30 members per outer fold |

## File ownership

Four workstreams write into this directory at once. Each owns its own prefix and
appends to `../compute_ledger.csv` without reformatting it.

| Prefix | Phase | Scope |
|---|---|---|
| `phase_a_*` | A — bounded CPU experiments | boosted trees, allele-centred targets, ranking objectives, blends |
| `phase_b_*` | B — assay-floor modelling | censored-loss stopping comparison, floor-event calibration, two-part model |
| `phase_c_*` | C — joint embeddings (**gate only**) | checkpoint leakage audit, access verification, priced smoke benchmark |
| `phase_d_*` | D — structural reuse | **COMPLETE** — [`phase_d_structural_residual.md`](phase_d_structural_residual.md); residual screening on features already on disk, no new folding, 25 core-minutes at zero cost |

Phase A additionally owns `pepstab/nested.py`, `pepstab/phase_a.py`,
`scripts/phase_a_*.py` and `modal_app/phase_a_experiments.py`. Phase B owns
`pepstab/calibration.py`. Phase C owns `modal_app/mint_*.py`. No phase edits
`pepstab/mlp.py`, `pepstab/evaluation.py` or `pepstab/censored.py` — 540 existing
tests rest on them and several phases read them concurrently.

## Decision log

Decisions taken during execution that are not derivable from the plan. Each is
dated and names the evidence.

**4 Oct — outer folds are `cv_folds`, not `stage6.nested_folds`.** Both exist.
`nested_folds` partitions by peptide and deliberately places mutant pairs in
*different* folds, which its stage 6 mutant-ranking diagnostic needs and which
is wrong here: it would let a model early-stop or be assessed on a peptide one
or two substitutions from one it fitted. Cluster-grouped folds are the stricter
choice and match the plan's "existing five training-cluster folds". Reached
independently by the Phase A and Phase D workstreams. Consequence:
`preds/seq_ensemble_pep_pseudo_oof.csv` and `preds/esm_ensemble_oof.csv` are
built on the peptide scheme and **cannot be reused**; A0 regenerates.

**4 Oct — A0 emits a 30-member-equivalent inner out-of-fold set; budget 150 →
750 networks.** Raised by Phase D. Fitting a residual or blend model on a
6-member inner out-of-fold error distribution and applying it to a 30-member
`outer_held` prediction mismatches ensemble arity, which was believed worth
roughly +0.07 to +0.09 — **corrected 4 Oct: that figure is
single-network-versus-ensemble; measured, 12 members against 30 is +0.0156** —
and which was at the time thought worth roughly +0.07
to +0.09 median Spearman here and inflates the fitted correction's scale. This
is a confound in A5's blend weights as much as in Phase D's residual model. Cost
to remove it is about 25 minutes of one core and $0. The residual *training-size*
mismatch it leaves (10,094 against 12,618 fit rows) is the irreducible remainder
of nested CV and is reported, not claimed away.

**4 Oct — floor event is `thalf_hours == 0`; two supporting arguments withdrawn
the same day.** The change stands on the recorded assay-floor code being an
exact 0, which is what EVALUATION.md's "20.2%" names (20.1626% reproduces to the
digit), so the calibration target needs no assay constant. Two arguments
originally offered for it were wrong, were challenged by Phase B, were verified
against the source, and are recorded as withdrawn in §7 rather than deleted:
the "443 rows / 1.57 pp" figure compared `<=` against `<` when the definitions
actually on the table differ by exactly one row (`pair_id` 1189), and the claimed
float-comparison hazard does not exist on this path because
`pepstab.censored.censored_mask` compares hours (`pepstab/censored.py:89`). The
loss's own masking rule stays at `< 0.1 h` so the censored arms remain
comparable to the stage 7 result Phase B exists to resolve; the resulting
one-row disagreement between loss mask and calibration event is carried
deliberately and checked.

**4 Oct — grid-boundary triggers and their remedies declared in advance, in both
phases that select on a ladder.** A truncated search space at the
weak-regularisation end cost the ESM arm 0.093 median Spearman earlier in this
project, so neither Phase A nor Phase D decides the remedy after seeing which
arm it would help. Each declares a trigger (a parsimonious selection landing on
a ladder boundary) and a fixed remedy: the axis extends **once**, by a declared
step; **every arm in the family is re-selected** on the extended ladder, not
only the arm that triggered it; both ladders are reported; no second extension
is available, and a boundary selection on the extended ladder is reported as a
handicap on that arm. Raised by Phase D for its ridge ladder and mirrored into
Phase A's grids before any screen ran.

**4 Oct — SPEARMINT stability checkpoints are disqualified by measurement, not
by inference.** Phase C's audit finds the two fine-tuned stability checkpoints
trained on the NetMHCstab-pan corpus, which is our corpus, covering 100% of our
validation and test peptide clusters. The affinity-stage checkpoint is clean of
stability labels but shows extensive affinity-cluster overlap, measured against
our own curated affinity table as a proxy donor set — a lower bound if that
table is a subset of NetMHCpan 4.1's, and so usable only as a declared
affinity-exposed arm. Only the base interaction-pretrained checkpoint is a clean
upstream. Consequence: the plan's §6.4 affinity-then-half-life comparison cannot
borrow a released affinity stage and becomes a separate, unbudgeted
sub-project; §6.2 and §6.3 proceed without it.

**4 Oct — a calibration map must never sit in the path of the primary metric.**
Raised by Phase D, which caught it by asserting the rank-invariance it had
assumed rather than relying on it; the assertion fired on every arm.
`IsotonicRegression` is only **weakly** monotone — a step function that
collapses distinct predictions into equal values, and those new ties change
Spearman, because ties take average ranks under the frozen rule EVALUATION.md
adopts for the assay floor. A non-decreasing map preserves rank *order* while
destroying rank *resolution*. Measured on a 40-row allele, the scale our ≥20-row
bar admits: isotonic collapsed **40 distinct predictions to 7** and moved
Spearman by 0.23, while a strictly monotone affine rescale moved it by exactly
0.0 and kept all 40 distinct values. The sign is not predictable — it rises when
the tie-collapse aligns with the labels and falls otherwise — so it cannot be
treated as a bounded, conservative bias. Phase A's §2 was defective on this
point and is amended: the primary metric is always scored on **uncalibrated**
predictions; maps serve magnitude metrics only; a reported map's measured effect
on ranking is reported, never assumed zero; and a provably rank-preserving
rescale uses a strictly monotone affine fit with an assertion to 1e-8. Relayed
to Phase B, whose §5 post-hoc calibrated baseline would otherwise have read a
ranking change as a calibration result.

**4 Oct — the plan's §3.3 twelve-configuration ceiling does not transfer to a
GPU arm.** Noted by Phase C. §3.3 caps a new learner family at 12 prespecified
configurations, which is a *search-discipline* rule written with CPU learners in
mind; applied to an encoder-adaptation arm it silently becomes a budget ceiling
and is unaffordable. Phase C's §6.3 screen was cut to 3 unfreeze depths × 2
seeds for that reason. This is a fact about the plan, not about any checkpoint,
and any future GPU arm should declare its own reduced configuration count with
the reduction recorded rather than inheriting 12.

**4 Oct — Phase C gate outcome: §6.2 go, §6.3 unfunded, §6.4 descoped.** §6.2
(frozen separate-chain versus frozen joint-chain representations) is affordable
at about $1.10 and proceeds. §6.3 (frozen versus limited adaptation) forecasts
$238 with margin, which exceeds the free credit in *either* workspace
individually — roughly $110 of ~$300 is realised in each, and credit cannot
cross workspaces — so it is **unfunded, not refused**. The forecast rests on a
batch-1 fp32 measurement, so two cents-scale throughput probes (batched bf16 at
batch 8/16/32; epochs-to-early-stop on one inner fold) were authorised, which
could plausibly land §6.3 near $40–60 and make it affordable. Gate realised
≤$0.05 against a $0.13 compute forecast.

**4 Oct — the published MINT wrapper cannot be used for §6.2.**
`MINTWrapper.forward` mean-pools every chain, which would silently destroy the
position-preserving peptide pooling that §6.2's comparison is built on while
still returning plausible numbers. §6.2 must call `ESM2.forward(...,
repr_layers=[33])` directly. Recorded here because this is a correctness trap
that a later reader could easily re-introduce.

**4 Oct — a reported ranking-versus-MSE tradeoff in the structural correction
was RETRACTED before it reached a conclusion.** Phase D initially reported that
the geometry correction improved within-allele ranking while making squared
error worse, on the evidence that its two selection criteria disagreed — the
ranking criterion kept a finite shrinkage, the MSE criterion selected no
correction. A requested within/between-allele decomposition disproved it: MSE
**improves** for every arm (−0.0046 to −0.0480), almost entirely within-allele.
The apparent conflict came from reading the *selection* rather than the ladder
profile: D1's inner MSE improves by a real 0.0048 at its argmin, but that is
smaller than one inner-fold standard error, so the declared 1-SE parsimony rule
correctly pushed the selection to "no correction", while the ranking
improvement is larger relative to *its* standard error and kept a finite value.
The criteria agree on direction and differ only on whether the MSE gain is
resolvable at five inner folds — the rule working as declared, not objectives
in conflict. Recorded because a retracted claim should stay visible: the
instruction it implied (that a structural term must sit behind a calibration
map to recover MSE) is wrong, since there is no MSE to recover. What survives is
smaller: the **between**-allele MSE component worsens consistently (+0.0006 to
+0.006) while within-allele improves 8–9× more, so the correction displaces
alleles slightly relative to each other while ordering better inside them. The
primary metric is blind to that by construction, so it changes no verdict, but
anyone reporting absolute accuracy should know the sign. Affine slopes of
0.905–0.962 size the overshoot at 4–10%, and recalibrating recovers essentially
nothing.

**4 Oct — the affine rank-preservation guarantee holds at fold scope only.**
Phase D's 1e-8 assertion fired on a *pooled* prediction vector, correctly: each
fold gets its own `(a, b)`, so a pooled vector is several affine transforms
stitched together and is not a single monotone map — within one allele, rows
from different folds can reorder. Measured pooled shift ≤ 0.0012. Phase A's §2
rule 4 was incomplete on this point and now specifies fitting and asserting
within one fold.

**4 Oct — all compute moves to Modal.** User directive, mid-study: run compute
in the cloud rather than on the laptop. At the time local CPU was carrying four
concurrent jobs at ~98% each. Arm A0's local fit was allowed to finish, because
it was 20 minutes into 900 networks and blocks Phases B and D, and discarding
billed progress to rebuild it in the cloud would have been a net loss; the ESM
component's full pass, the A1–A4 screens, and Phase B's and Phase D's remaining
fits all go to Modal. Nothing about the protocol changes — same folds, same
arity, same panels, same `pepstab.evaluation` paths — only where the arithmetic
happens. One incidental gain: one container per unit with BLAS threads pinned in
the image removes thread contention as a reproducibility hazard, which local
concurrency had made load-bearing.

**4 Oct — PHASE D COMPLETE: the structural residual correction does not help,
and the interval rules out a worthwhile gain.** On the pooled outer-held
69-allele panel, 2,000 resamples at seed 20261003, against a D0 reference that
**reproduces A0 element-wise to four decimals** (two independently written
harnesses on the same folds — the strongest provenance check in the study):

| arm | features | median SCC | Δ vs D0 | 95% CI | folds selecting no correction |
|---|---:|---:|---:|---|---:|
| D0 none | 0 | 0.66152 | — | — | — |
| D1 geometry | 36 | 0.66208 | +0.00056 | [−0.0035, +0.0021] | 4/5 |
| D2 confidence | 73 | 0.66799 | +0.00647 | [−0.0055, +0.0122] | 1/5 |
| D3 anchor P2/P9 | 15 | 0.66152 | +0.00000 | [0, 0] | **5/5** |
| D3b anchor P1/P2/P9 | 18 | 0.66152 | +0.00000 | [0, 0] | **5/5** |
| D4 geometry+confidence | 109 | 0.66994 | +0.00842 | [−0.0055, +0.0124] | 2/5 |
| D3c central P3–P8 | 27 | 0.66284 | +0.00132 | [−0.0049, +0.0035] | 4/5 |
| DX shuffled control | 36 | 0.66152 | +0.00000 | [0, 0] | 5/5 |

Every arm reads as EVALUATION.md's row 5 — "inconclusive at 0, and rules out a
worthwhile gain; not a negative result". The best arm's upper bound is 0.0124, a
quarter of the 0.05 bar, so this is a reasonably precise **exclusion** rather
than a failure to detect.

Sharpest findings: the declared parsimony rule selected **no correction in all
five folds** for both anchor arms, so P2/P9 geometry adds nothing on top of a
30-member sequence ensemble; and D3c reaches no-correction in 4/5, so the plan's
hypothesis that central peptide positions carry biology the anchors miss is
**also** unsupported. Neither subset helps and nor does the full block. The
ladder-extension trigger did not fire — selections went to the *strong*
shrinkage end (`inf`, 1e5, 1e6), confirming that the earlier low-edge selection
was an artifact of the weak 0.236 plumbing reference. Affine slopes flipped to
1.03–1.05, so against A0 the correction undershoots rather than overshoots.

**Reading note on the [0, 0] intervals:** those arms selected no correction in
every fold, so their predictions are *identical* to D0 and the delta is
identically zero by construction. The degenerate interval means "the same
predictions", not "a zero effect measured with extraordinary precision".

**Consequence for the plan:** §7's gate ("commission more structures only
after...") is answered in the negative on existing data, which is the cheapest
"no" available in the whole further-research plan. No new folding is justified.

**Sensitivity check agrees and is slightly more negative.** Re-run on
`inner_oof_6`, D1 and D3c join the anchor arms at no-correction in 5/5 folds, D4
halves to +0.00427 [−0.0042, +0.0091] and D2 is unchanged at +0.00632 [−0.0055,
+0.0114]. Every arm returns the same reading under both surrogates, so the
verdict does not depend on which inner surrogate was chosen. The primary stays
`inner_oof_30` because that preference was declared before the ratio was known.

**4 Oct — the `inner_oof_30` amendment was counterproductive for its stated
purpose.** Measured independently by Phase D, agreeing with the A0 harness:
`inner_oof_6` sits at residual-sd ratio **1.0193** to `outer_held` against
`inner_oof_30`'s **1.0342**, so the higher-arity surrogate is *further* from the
target — by 1.8× on the only quantity that measures it. The mechanism is visible
in A0's own dev MSE: 0.5543 at level 2 against 0.5903 at level 3. A 6-member
ensemble of better-trained members is a closer surrogate for a 30-member
ensemble of well-trained members than a 30-member ensemble of worse-trained
ones. The arity confound was real and the reasoning sound as far as it went; it
was not weighed against the training-size cost of the remedy, and the side
effect exceeds the problem. Neither session could have known the sign without
measuring, which argues for measuring before paying rather than for the
reasoning being wrong. The 750 extra networks still bought the measurement
itself — the training-size remainder is **0.0229** in residual sd (`outer_held`
0.6674 against `inner_oof_30` 0.6903), or 3.4% of `outer_held`, rather than an
argument. (An earlier draft of this entry said 0.0149, which came from nowhere
and is withdrawn; the `inner_oof_6` gap is 0.0129, which is the figure behind
"1.8× closer" and which stands.) **A5 fits its blend weights on `inner_oof_6`**, after checking the
ratio for its own arm, since blend weights are a different functional of the
residuals than a ridge correction is. Phase D keeps `inner_oof_30` as its
pre-declared primary with `inner_oof_6` as a declared sensitivity check, because
switching after seeing the ratio would be post-hoc.

**4 Oct — a long run must checkpoint its partial result, not only its final
one.** Adopted as a convention for this repository, alongside the
`esmfold_common.py` note about `modal run` buffering a traceback out of sight.
Established by a measured contrast during the outage below: Phase C's feature
extraction kept **45%** of its work because it commits to a Volume
incrementally, while its epoch probe lost **100% of 42 minutes** because it
returned its history only as a return value — the per-epoch numbers existed in
memory the whole time. Cost: $0.78, logged `measured_wasted`. On a flaky network
this is the difference between a $0.78 loss and a $0.78 answer. The same shape
appeared in Phase A, where A0's 900-network run printed per-fold progress to a
block-buffered stdout, so a 34-minute job was indistinguishable from a hung one
from outside. Any long run here should emit and persist partial state as it
goes.

**4 Oct — UNRESOLVED: four detached Modal apps across three workstreams were
cancelled in a 7-second window.** From `modal app list --json`:
`mint-gate` 19:41:29, `phasea-experiments` 19:41:31, `phaseb-calibration`
19:41:34, `mint-features` 19:41:36. Every session on the machine independently
observed network failures in that window — `StreamTerminatedError`,
`SSL connection is closed`, `[Errno 8] nodename nor servname provided`, and an
`ENOTFOUND` that killed a subagent mid-task.

Two explanations were tested and eliminated. **A mis-targeted `modal app stop`
is ruled out**: the only explicitly named stop landed at 19:37:33, 3m56s before
the cluster, and a per-app command cannot stop four apps. (The pipeline used to
obtain that app ID — `modal app list | grep -B4 <time>` — *is* unsafe, because
`modal app list` wraps every column across lines so an ID and its timestamp sit
on different rows; use `modal app list --json` and match on the name field. It
happened to target correctly, but the pattern should not be reused.)
**Insufficient detachment is also ruled out**: `mint-gate` was observed in state
`ephemeral (detached)` while still running, so the flag applied.

So **why a genuinely detached app received "Received a cancellation signal"
following a client-side network failure is not established**, and is recorded as
unresolved rather than given a plausible-sounding mechanism. The operational
consequence is the one that matters: on this machine, `--detach` did not in fact
guarantee survival of a client-side outage, so a long run should not rely on it
in place of incremental checkpointing.

**4 Oct — A5's ESM component is complete, and it is a null.** Refitted under the
cluster protocol at A0's exact arity (5 inner × 2 ESM layers × 3 seeds = 30
members per outer fold, 900 networks), on `colleague`, $0.64 realised:

| | five-fold median | per fold | spread |
|---|---:|---|---:|
| A5/ESM | 0.6534 | 0.6769 / 0.6227 / 0.6676 / 0.6534 / 0.6421 | 0.0542 |
| A0 | 0.6493 | 0.6765 / 0.6281 / 0.6418 / 0.6493 / 0.6515 | 0.0484 |
| delta | **+0.0041** | +0.0004 / −0.0054 / +0.0258 / +0.0041 / −0.0094 | — |

The delta is an order of magnitude inside each arm's own fold spread, the
per-fold signs are mixed 3 up / 2 down, and the whole range sits below the 0.05
bar. This reproduces stage 3's validation verdict — "inconclusive at 0, rules
out a +0.05 gain" — under the stricter cluster protocol, so it is corroboration
rather than news. **The ESM arm's role in A5 is complementarity only**: if a
blend beats A0, that is a blend result and must not be reported as an ESM gain.
Fold 2's +0.0258 is exactly the single-fold gap the spread caveat forbids
reading into.

Its `inner_oof` scales match A0's closely (×1.017 and ×1.033 against ×1.019 and
×1.034), confirming that **both inner roles are pessimistic one-sided bounds on
a second arm too** — the no-bracket finding generalises beyond A0.

**4 Oct — ORCHESTRATION ERROR: "15,772 rows" was given to two sessions and is
wrong for four folds in five.** The fitting-partition sizes are
`[15772, 15773, 15773, 15773, 15773]` and the outer-held sizes
`[3944, 3943, 3943, 3943, 3943]`, because `assign_clusters` water-fills whole
clusters and fold 0 lands one row larger. 15,772 is fold 0's complement alone.
It was circulated as "15,772 rows each", and worse as a suggested assertion
("assert the fit row count is 15,772"), which would make a *correct* run fail on
four folds. It did: the ESM arm hardcoded it and **four of its thirty units
failed**, and its unit test passed because it checked only `sizes[0]`.

The remedy adopted, and recommended wherever a fit-row count is guarded:
replace the literal with a **partition identity** — assert the fit rows and the
predicted rows are disjoint *and* together reconstitute the frame they were cut
from. Correct for every fold with no magic constant, and strictly stronger than
a count because it also catches a dropped row. The anti-pooling intent survives:
a pooled fit makes the fit set 19,716 and breaks the identity at once.

**4 Oct — PHASE A SCREEN COMPLETE: 1,920/1,920 units, and nothing clears the
bar.** Zero failures, zero duplicates, one payload digest, every unit
`thread_pin_ok` at 2 threads. Two independently built assembly paths agree to
**5.55e-17**, and A0's declared per-fold medians reproduce exactly.

| arm | best config | 5-fold median | span | vs matched control |
|---|---|---:|---:|---:|
| A1 boosted trees | depth=8, l2=10, lr=0.06 | 0.6611 | 0.0363 | **+0.0273** |
| A3 ranking objective | mse+rank, natural, rank-stop, λ=1.0 | 0.6590 | 0.0718 | **+0.0253** |
| A2 allele-centred | k=20 best, k=100 selected | 0.6298 | 0.0630 | −0.0039 |
| *matched 12-member control* | *MSE, natural, MSE-stop* | *0.6337* | *0.0463* | — |
| *A0 (30 members, macOS)* | — | *0.6493* | *0.0484* | — |

**Every effect measured is smaller than the arms' own fold spans** (0.036–0.072
against effects of ≤0.027). That does **not** make them unresolvable, and an
earlier draft of this entry wrongly said it did: a *paired* bootstrap on shared
resamples cancels fold variance, which is precisely what it exists to do. The
first intervals show the effects are resolvable and small.

**The reference choice, not the arm, decides the verdict.** Same arm, same
predictions, same 2,000 resamples at seed 20261003:

| reference | Δ | 95% CI | verdict |
|---|---:|---|---|
| matched 12-member control | **+0.0178** | [+0.0033, +0.0459] | **real, but rules out a 0.05 gain** |
| A0 (30 members, macOS) | +0.0072 | [−0.0113, +0.0304] | inconclusive at 0, rules out a 0.05 gain |
| strongest new control (A3-c, λ=1.0, rank-stop) | −0.0031 | [−0.0150, +0.0240] | inconclusive at 0, rules out a 0.05 gain |

Three readings, and **both reference rules in §2 earned their place**:

- Against the **matched control** the interval **excludes 0**: the boosted tree
  genuinely beats the plain MSE recipe at equal arity, inner-fold count and
  platform, while ruling out a worthwhile-sized gain.
- Against **A0** it does not. The point estimate drops 0.0106 — close to the
  +0.0156 separating A0 from the matched control — and the interval widens,
  because the A0 pairing carries arity, inner-fold-count and cross-platform
  variance the matched pairing does not. **Had A0 been the only reference, a
  real effect would have been reported as a null.**
- Against the **strongest new sequence control** A1 and the best A3 cell are
  **indistinguishable**. So A1's win is *not* evidence that boosted trees beat
  the MLP family: A3 reaches the same place using the reference MLP with only a
  changed objective and stopping rule. §2 requires this second reference exactly
  so that a learner-family win is not claimed when a cheaper change gets there
  too — **and here it does**.

**The resulting headline is not "trees win".** It is that the plain MSE recipe
leaves roughly +0.018 on the table, and that gain is reachable either by
switching learner family to boosted trees *or* by changing the objective and
early-stopping rule on the MLP already in use. The second route is far cheaper —
no new library, no new tuning surface — and the stopping-rule decomposition
below says most of it comes from the stopping surface rather than the gradient.

Note the two aggregations also differ: A1 is +0.0118 above A0 on five-fold
median-of-medians but +0.0072 on the pooled paired quantity, and the pooled
panel median at low resample count put it *below* A0. **The pooled paired figure
is the one the interval attaches to** and is the inferential result; the
median-of-medians difference is descriptive only.

**"Does any arm beat A0?" has two answers that disagree in sign.** A1 is +0.0118
above A0 on five-fold median-of-medians and *below* A0 on the pooled panel
median. Both differences are smaller than A0's own span, and every A0
comparison additionally carries the cross-platform term (A0 on
macOS/Accelerate, the screen on Linux/OpenBLAS). **The matched 12-member
control is therefore the headline reference, not A0.**

**RETRACTED — the stopping rule has no stand-alone effect.** An earlier version
of this entry said rank-based stopping was the screen's sharpest and most
transferable finding, on the strength of its winning all 8 matched cells on
median-of-medians (+0.0085 and +0.0172 in the two pure-MSE cells). **The paired
intervals do not support that:**

| cell | ranking gradient? | Δ (rank − MSE stopping) | 95% CI | verdict |
|---|---|---:|---|---|
| A3-a MSE, natural | **no** | −0.0032 | [−0.0073, +0.0175] | inconclusive |
| A3-b MSE, balanced | **no** | +0.0058 | [−0.0048, +0.0204] | inconclusive |
| A3-c mse+rank, natural | yes | +0.0288 | [+0.0012, +0.0279] | real, sub-bar |
| A3-d mse+rank, balanced | yes | +0.0085 | [+0.0004, +0.0254] | real, sub-bar |

Where there is **no ranking gradient** the effect is centred near zero and
inconclusive, and A3-a flips sign. It resolves only in the two cells that also
carry the ranking term. **The corrected claim: rank-based stopping does not help
on its own; it interacts with the ranking gradient.** The decomposition's
apparent additivity (stopping +0.0085 / loss +0.0138 / interaction +0.0030) was
an aggregation artifact — median-of-medians and the pooled paired quantity
disagree in sign on the decisive cell.

So the transferable part is the **ranking loss**, with rank-based stopping
useful only alongside it. This is a weaker and narrower finding than the one
first reported, and the retraction is recorded rather than the text quietly
replaced.

**Anomaly on the largest stopping-axis effect:** A3-c's point estimate
(+0.0288) lies **outside its own percentile CI upper bound** (+0.0279).
Legitimate for a percentile bootstrap whose resample distribution is shifted,
but it signals bootstrap bias, and that interval should not be quoted without a
BCa check.

Still consistent across all 8 matched pairs: natural sampling beats
allele-balanced, and λ is monotone 0.1 < 0.3 < 1.0 in all four cell × stopping
combinations.

**A2 is the screen's one conclusively negative arm**: −0.0132 [−0.0310,
−0.0003] against A0 and −0.0235 [−0.0379, −0.0037] against the best A3 cell —
the only two "worse" verdicts in the screen. §A2 asked a mover to identify a
training-side mechanism; it moved *downward*, so the arm is unsupported.

**Blend surrogates disagree.** A0 + A1 weights fitted on `inner_oof_6` give a
mean A0 weight of 0.732 against `inner_oof_30`'s 0.710, max |Δw| 0.05 across
folds — five grid steps, beyond the grid's resolution. Both agree the blend is
A0-heavy. Checking it for the blend specifically, rather than inheriting Phase
D's ridge finding, was therefore worth doing.

**Repository hygiene:** the screen produced 1.1 GB of untracked intermediates —
a 496 MB journal and its 496 MB CSV twin, plus a 97 MB assembled file — none
gitignored and all one `git add` from the repository. Now ignored with a
regeneration note, leaving 27.5 MB of genuine record.

**4 Oct — STUDY-WIDE METRIC FINDING: the difference of medians is not the
median of differences, and a CI excluding its own point estimate is a stop
sign.** Found by Phase C, and it applies to every comparison in this study
because the primary metric is a *median over a fixed allele panel*.

Phase C's joint-vs-esm2 contrast produced a bootstrap CI of [−0.0582, +0.0190]
around a point estimate of **−0.0607** — the interval **excluded its own
estimate**. The cause is not a bug: `median(arm) − median(ref)` is sensitive to
*which allele sits at the median* in each arm, while the per-allele effect was
only −0.0097. So the −0.0607 was almost entirely median placement, and
`describe_delta`'s verdict could not be read as a per-allele claim.

**Adopted for the whole study:** report a **paired sign test and the median of
per-allele differences** alongside every difference of medians, and treat a CI
that excludes its point estimate as a stop sign rather than a curiosity.

**Phase A's headline was checked against this and passes, strengthened:**

| statistic | A1 vs matched control |
|---|---:|
| difference of medians (what the CI is on) | +0.0178 |
| **median of per-allele differences** | **+0.0200** |
| mean of differences | +0.0234 |
| alleles improved | **54 / 69** |
| Wilcoxon paired | p = 5.2e-07 |
| sign test | p = 2.6e-06 |

The two aggregations agree in sign and magnitude, so the boosted-tree effect is
a genuine per-allele improvement rather than a panel-placement artefact — a
materially stronger statement than the bootstrap interval alone supports.

**4 Oct — PHASE C §6.2 COMPLETE: negative, with a clean mechanistic
decomposition.** All frozen arms sit 0.18–0.26 below the BLOSUM sequence
ensemble (A0 0.6493; esm2 0.4651; joint MINT 0.4044; separate MINT 0.3851):

- **separate → joint +0.0193**, Wilcoxon p=5.0e-06, 52/69 alleles —
  cross-chain context at inference genuinely helps; size against the 0.05 bar
  unresolved.
- **esm2 → separate −0.0800**, p=1.0e-05, 15/69 — MINT's STRING PPI pretraining
  *cost* single-chain representation quality relative to the ESM-2 it was built
  from. Identical feature contract, so it is the weights.
- **esm2 → joint: no evidence of a difference** (p=0.33, 28/69).

Net: cross-chain attention recovers roughly what the pretraining gave up,
leaving joint MINT indistinguishable from plain frozen ESM-2. Consistent with
stage 3. Phase C declined to recommend stop on §6.2 alone, correctly: §6's
criterion is a conjunction and the blend half is untested.

Phase C also found its 1-SE L2 rule pinned at the ladder boundary on 12/15
arm-folds, **unevenly** across arms (5/5, 4/5, 3/5), which could have inflated
both significant contrasts. It extended the ladder as a declared §3.3
extension with the prespecified run preserved as primary; **no arm improved**,
so 1e-2 is a genuine optimum of a flat region and every conclusion holds.

**4 Oct — RETRACTED: "§6.3 costs a measured $30.59" and "adaptation tops out at
0.47–0.48".** Both figures were circulated, including to the user, and both
rested on a misread epoch curve. The probe was inspected from a Volume
checkpoint at epoch 10, which showed one worse epoch after a monotone decline
and was called a turnover. Epochs 11 and 12 did not exist yet:

```
epoch  9  val_mse 0.66737
epoch 10  val_mse 0.68753   <- the single-epoch blip that was misread
epoch 11  val_mse 0.67050
epoch 12  val_mse 0.65843   <- NEW MINIMUM, at the cap, still improving
```

`best_epoch_by_mse = 12`, `still_improving_at_cap = True`. A patience of 1
would also have stopped at 10 and been wrong; this curve needs patience ≥ 3.

Consequences: **§6.3's cost is unresolved with a floor of $39.72** (not
$30.59), and $61.01 at the gate's original 20-epoch assumption is **not
refuted**. The throughput correction (4.2× from batching and bf16) is where the
genuine fall from $238 came from; the epoch half was never measured. And
**0.4890 at the cap while still climbing is a floor at 12 epochs, not a
ceiling** — the adaptation trajectory is upward and its real ceiling is
unmeasured.

**The recommendation to stop Phase C is unaffected**, and both corrections cut
slightly *in favour* of §6.3 — it is dearer than stated and its ceiling unknown
rather than low. The case never rested on either figure: it rests on §6.2's
frozen comparison and the blend check below, both independent of epoch count.

**4 Oct — PHASE C COMPLETE: stop, on the plan's own conjunction.** The blend
half of §6's stop-scaling criterion returned **no signal**: given a free
nonnegative weight chosen on held-out folds, the procedure selected **zero on 11
of 15 arm-folds** and never exceeded 0.1, preferring the sequence model
essentially alone. Every CI crosses zero (joint −0.0016, separate +0.0002, esm2
+0.0020).

The new sign-test rule earned its place immediately here: `joint`'s Wilcoxon is
p = 4.0e-06, which reads as a strong effect, but only **17/69** alleles improved
and the median paired delta is −0.0010 — significant in the *wrong* direction.
Adding a little of a weaker arm degrades most alleles slightly, exactly as a
near-zero optimal weight predicts. The difference of medians alone would have
hidden the direction; the sign test alone would have overstated the magnitude.

Phase C also noted its own scope boundary: what it ran is the minimal
gate-condition test (one weight on a 0.05 grid, chosen on the other four folds),
**not §6.5** — no inner-OOF weight fitting, no residual variant, no tuning.
Labelled as such so it cannot be cited as Phase C's blend finding.

**Reconciliation note:** Phase C's sequence reference reads 0.6615 against this
document's 0.6493. Same predictions, different aggregation — 0.6615 is the panel
median computed once over all 19,716 pooled rows, 0.6493 is the across-fold
figure from per-fold medians. Neither is wrong and they must not be presented as
contradicting.

Total realised Phase C spend **$3.14**, of which $0.78 was the external
cancellation.

**5 Oct — THE MOST CONSEQUENTIAL FINDING IN THE STUDY: the run-to-run
reproducibility floor (0.0043) is LARGER than the platform effect it was
commissioned to measure.** The A0 Linux re-run was asked to quantify a
macOS-versus-Linux artifact. It found something bigger.

| comparison | ΔMoM | max abs | mean abs | corr | bit-identical folds |
|---|---:|---:|---:|---:|---:|
| **Platform** @ threads=1 | −0.0020 | 2.51e-01 | 3.28e-02 | 0.998646 | 0/5 |
| **Threads** 1→2 on Linux | +0.0034 | 2.44e-01 | 3.23e-02 | 0.998671 | 0/5 |
| **Compound** mac/t1 → linux/t2 | +0.0013 | 2.03e-01 | 3.29e-02 | 0.998663 | 0/5 |
| **Run-to-run, IDENTICAL settings** | **+0.0043** | 1.91e-01 | 1.44e-02 | 0.999348 | **3/5** |

Two runs of the same five units at the same settings on the same platform
disagree by **0.0043** on median-of-medians — more than the platform shift, and
more than either of its components.

**Mechanism, identified rather than inferred.** Per-container CPU fingerprints
show the containers split across two microarchitectures: 3 at `avx,avx2` and 2
at `avx,avx2,avx512dq,avx512f`. OpenBLAS dispatches a different kernel per
microarchitecture, changing float32 accumulation order. Units landing on the
same host class reproduce bit-for-bit (3 of 5 folds); others do not. **This is
not fixable by pinning threads or seeds — only by replication.**

**Why the decomposition mattered.** Platform is −0.0020 and threads is +0.0034,
and they nearly cancel into the +0.0013 compound. A single threads=2 run
differenced against the macOS A0 would have reported "+0.0013, platform barely
matters" while two effects of twice that size pointed in opposite directions.
The platform term also reproduces Phase B's independent fold-0 measurement of
0.0020 exactly.

**Consequences applied across the study:**

- **No effect of this size is resolvable from one run per arm.** Phase A's
  headline (+0.0178) is about 4× the floor and survives; the ESM arm's +0.0041
  over macOS A0 and +0.0027 over Linux A0 are both *inside* it, so **the ESM
  result is parity, full stop**.
- Any future arm should be **replicated and reported with spread**, not run once.
- The floor is **unmeasured for the CatBoost arms**, which do not use the same
  BLAS paths.
- One replicate gives one difference, not a standard error: 0.0043 is an order
  of magnitude, not a bound.

**Three defects found in the same run, two of them the runner's own:** the image
`.env` thread pin **silently failed** (all five containers reported
`OMP_NUM_THREADS=2` while dispatched as threads=1) and was caught only by the
`thread_pin_ok` field — the same mechanism `phase_a_esm.py` and
`phase_a_experiments.py` rely on, so both should carry that check; a copied
payload column (`hla_seq` instead of `hla_pseudoseq`) killed all three smoke
units in 2 seconds for ~$0.001, now fixed by deriving columns from
`features.INPUT_SETS` rather than by hand; and `to_csv` under pandas 3.0.6 drops
the final ULP (up to 4.44e-16), which means **Phase B's "bit-identical, max|diff|
4.44e-16" was measuring exactly that CSV precision — the agreement was *exact*,
not merely close.**

**5 Oct — A4 RUN: boosted trees plus the ranking objective is the study's best
arm, and the combination does add.** 120/120 units, 0 failures, 0 unpinned
threads, 51.5 min on `colleague`, at 364–373 s/unit against a 526 s forecast.
Run as §6c's same-geometry ablation: A1's selected and argmax configs with only
`RMSE → PairLogit` changed, screen arity so it is directly comparable to A1's
existing 12-member result.

| arm | pooled rho | vs matched control |
|---|---:|---:|
| **A4 trees + PairLogit** (l2=10) | **0.6805** | **+0.0295** |
| A4 trees + PairLogit (l2=3) | 0.6726 | +0.0216 |
| A1 trees alone (l2=3) | 0.6731 | +0.0221 |
| A3-c MLP + ranking | 0.6719 | +0.0209 |
| A1 trees alone (l2=10) | 0.6687 | +0.0177 |
| matched 12-member control | 0.6510 | — |

**The decisive comparison is A4 against A1 at identical tree geometry**, where
the objective is the only difference:

| comparison | Δ(med) | med(Δ) | improved | Wilcoxon | sign test |
|---|---:|---:|---:|---:|---:|
| A4 vs A1, l2=10 | **+0.0118** | **+0.0175** | **53/69** | 1.1e-07 | 9.1e-06 |
| A4 vs A1, l2=3 | −0.0005 | **+0.0116** | 45/69 | 3.9e-05 | 0.015 |
| A4 vs matched control | +0.0295 | +0.0319 | **62/69** | 3.3e-11 | 4.1e-12 |
| A4 vs A3-c | +0.0086 | +0.0182 | 49/69 | 3.8e-06 | 0.00064 |

**Paired intervals, 2,000 resamples at seed 20261003**
([phase_a_a4_comparisons.csv](phase_a_a4_comparisons.csv)):

| comparison | Δ | 95% CI | verdict |
|---|---:|---|---|
| **A4 vs matched control** | **+0.0295** | **[+0.0150, +0.0596]** | **real, size unresolved — straddles 0.05** |
| A4 vs A1, l2=10 | +0.0118 | [−0.0032, +0.0300] | inconclusive at 0, rules out a 0.05 gain |
| A4 vs A1, l2=3 | −0.0005 | [−0.0094, +0.0240] | inconclusive at 0, rules out a 0.05 gain |
| A4 vs A3-c | +0.0086 | [−0.0027, +0.0395] | inconclusive at 0, rules out a 0.05 gain |

**This is the only interval in the entire study that does not rule out the 0.05
bar**, and it must not be read as clearing it either — EVALUATION.md's row 4,
"real, size unresolved", is explicit that "below the bar" would be unsupported.

**An earlier draft of this entry overstated it, and the overstatement is
instructive.** It led with the per-allele sign tests (A4 over A1: 53/69 alleles,
Wilcoxon p = 1.1e-07) and treated the aggregate as corroboration, concluding
"the two levers compose". The panel-median intervals say A4-over-A1 is
**inconclusive at both ladder rungs**, and the panel median is this project's
primary metric. A sign test answers "does this help most alleles"; the panel
median answers "does the median allele's score move". Both are true here and
they are different questions.

The supportable statement: **both levers together reach an effect of unresolved
magnitude, and neither lever's individual increment is separately resolvable.**
That is the expected signature of two small composing effects at this
measurement precision — and it strengthens rather than weakens the study's
conclusion that the binding constraint is measurement precision, not model
choice.

**The sign-test rule rescued this result from being misread, hours after it was
adopted.** The difference of medians disagrees across the two configs (+0.0118
and −0.0005), which alone reads as "one gains, one does not — inconsistent,
probably noise". The per-allele statistic shows **both** configs improving
(+0.0175 and +0.0116, each with a majority of alleles), so the l2=3 null was a
median-placement illusion — the exact artefact Phase C identified. Reporting
only the difference of medians would have produced a wrong conclusion here.

**A prediction recorded in §6c was wrong, and the user's challenge to it was
right.** §6c originally argued the combination would not add, because A1 and
A3-c are statistically indistinguishable and must therefore capture the same
signal. That inference is invalid — same aggregate score does not imply same
mechanism, and this study's own A0+A1 blend (+0.0128 real, from two arms
differing by an inconclusive +0.0072) was already the counterexample. The
mechanistic case that survived is the opposite one: `PairLogit` with
`group_id = allele` optimises the metric's exact grouping, and a
piecewise-constant tree ensemble is precisely what most lacks within-group
ordering pressure. §6c was amended to remove the prejudicial expectation
*before* the arm ran.

**Caveat carried:** +0.0118 sits near a run-to-run floor of 0.0043 that is
**unmeasured for CatBoost arms**, which do not share the MLP arms' BLAS paths.
The per-allele evidence (53/69 at p = 1.1e-07) is far harder to attribute to run
noise than the aggregate is, but a CatBoost replicate is now worth its ~$2 —
reversing the earlier judgement that replication bought nothing, which held only
while every effect was indistinguishable from zero.

**5 Oct — LATENT TRAP: the batch's final CSV filename does not carry the
profile, though its journal does.** A4's run wrote
`reports/further_research/phase_a_runs.csv` — the same path the A1–A3 screen had
used — silently replacing 1,920 rows with 120. **No data was lost**: the
1,920-row journal was intact, the CSV is a pure projection of it, and all three
derived screen outputs survived. Regenerated and split into
`phase_a_runs_screen.csv` (1,920 rows) and `phase_a_runs_a4.csv` (120). The
regenerated screen CSV is **2.0 MB** rather than 496 MB, because the per-unit
prediction arrays belong in the journal and not in a metrics table. Any future
second run on a different profile hits the same trap until the output name
carries the profile.

## Status

Phase A's protocol is frozen and its harness is built; fitting is under way.
Phases B, C and D are executing their own scopes against the shared protocol
above. No phase has reported a scored arm yet, so this directory currently holds
protocol, provenance and diagnostics rather than results. Nothing in it should
be read as a performance claim until a phase's own report states one with a
paired interval.
