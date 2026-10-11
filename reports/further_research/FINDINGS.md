> **Historical study record.** The protocols, results and future-tense instructions below describe the completed October study. The [current project review and research plan](../NEXT_RESEARCH_PLAN.md) is the sole forward work queue; its [dated correction register](../NEXT_RESEARCH_PLAN.md#dated-correction-register--12-october-2026) supersedes conflicting interpretation, selection and blend-gate claims here. No old protocol should be relaunched as a new study without the current evaluation gates.

# Further research: findings

Date: 5 October 2026. Status: **synthesis of a completed four-phase study**, for
handoff and external review. It reports what the study found; it is not a plan or
a protocol. The plan it executed is
[FURTHER_RESEARCH_PLAN.md](../archive/plans/further-research-2026-10-04.md); the dated decision log and
every retraction live in [README.md](README.md); the binding protocol is
[phase_a_prespecification.md](phase_a_prespecification.md). Verdict vocabulary and
the 0.05 worthwhile bar are from [EVALUATION.md](../../EVALUATION.md).

A reader who wants one document instead of five should read this one. Every figure
is traced to a phase report or a CSV. Where two phase reports quote different
numbers for the same quantity, the reason is stated rather than hidden (§8).

## What was and was not scored

**No split of the frozen benchmark was scored.** The test split was not read, and
the validation split was not scored either. All four phases assess on
**training-only nested cluster validation**: five whole-cluster outer folds over
the 19,716 training rows, inner cluster folds for every selection, scored through
`pepstab.evaluation` verbatim with the frozen tie and eligibility rules. The
published validation (0.6931) and test (0.7064) numbers for the sequence reference
are *not* reproduced here and are not involved in any claim below. A reviewer who
assumes the frozen benchmark numbers are in play will misread the whole study.

The one exception is a **retrospective correction** of the stage 7a record, which
Phase B scored on the frozen validation split because that is where the stage 7a
numbers already lived. It selected no new model.

## 1. Headline

No intervention tested is **established** as worth deploying for ranking. Of
every ranking comparison in the study, exactly one does not rule out a 0.05
median per-allele Spearman gain — the project's worthwhile bar — and that one
does not establish it either:

> **A4, a boosted tree with a groupwise ranking objective, against the matched
> sequence control: +0.0295 [+0.0150, +0.0596].** The interval excludes 0 and
> **straddles 0.05**. EVALUATION.md's verdict: *"Real, size unresolved."*

That is the contract's own tricky row, and it must not be read as "below the
bar": the improvement is real and its magnitude is not established. Every other
comparison across all four phases returned "inconclusive, rules out a
worthwhile gain" or "worse".

Two single changes produce **real but sub-bar** ranking gains and are
statistically indistinguishable from each other: boosted trees (A1) and a
within-allele ranking loss with rank-based early stopping (A3-c). Against a
matched sequence control each is worth roughly +0.02; against each other,
inconclusive.

**Their combination is the study's best arm.** A4 — a boosted tree with
CatBoost's groupwise `PairLogit` objective at `group_id = allele` — scores a
pooled 0.6805, the highest single figure anywhere in the study, **+0.0295
[+0.0150, +0.0596]** over the matched control with 62 of 69 alleles improving.
That is the one interval in the study that does not rule out the bar.

**What the increment from each lever is, is a separate question and is not
resolved.** A4 against the tree alone at identical geometry — the objective as
the only difference — is **inconclusive at both ladder rungs** (+0.0118
[−0.0032, +0.0300] and −0.0005 [−0.0094, +0.0240]), and A4 against the best MLP
ranking cell is inconclusive too (+0.0086 [−0.0027, +0.0395]). The per-allele
evidence for the objective's addition is strong (53/69 alleles, Wilcoxon
p = 1.1e-07), but it does not carry the panel median, which is this project's
primary metric.

So: **both levers together reach an effect of unresolved magnitude; neither
lever's individual contribution is separately resolvable.** That is what two
small composing effects look like at this measurement precision, and it is the
sharpest statement the data supports. Claiming "the two levers compose" on the
strength of the sign test alone would overstate it.

One change is a **clear win on a different metric**: the censored likelihood under
MSE early stopping (Phase B's B2) more than halves floor-probability Brier and
cuts calibration error by two-thirds, at no established ranking cost. It is a
calibration result, not a ranking result, and the study never lets it count toward
the ranking bar.

Everything else is null or worse: allele-centred targets are worse; the ESM blend
arm is parity; frozen joint peptide–HLA (MINT) embeddings are indistinguishable
from frozen ESM-2 and both sit ~0.18–0.26 below the sequence baseline; the
structural residual correction adds nothing; the two-part floor model is dominated.

The study's own conclusion is that **the binding constraint is measurement
precision, not model choice**. Several independent interventions land on real
effects clustered near +0.02, against a fold-to-fold spread of ~0.05 with the
model held fixed and a run-to-run reproducibility floor of 0.0043 that the paired
intervals do not even cover (§4).

## 2. Ranking effects, ranked

All deltas are median per-allele Spearman on the pooled outer-held panel (69
alleles, ≥20 rows and ≥2 distinct labels, 2,000-resample paired peptide-cluster
bootstrap at seed `20261003`). The reference is named on every row, because the
verdict depends on it as much as on the arm.

### 2.1 Against the matched sequence control

The matched control is A3-a: the reference MLP recipe — MSE loss, natural
sampling, MSE-based early stopping, stage-2 hyperparameters — run inside the same
screen at the same 12-member arity, inner-fold count and platform. An
arm-against-control delta therefore isolates what the arm changed and nothing
else. Source: [phase_a_screen.md](phase_a_screen.md) §"Comparison against three
references", pooled control 0.6510.

| Arm | Δ vs matched control | 95% CI | Verdict |
|---|---:|---|---|
| **A4 boosted trees + PairLogit** (`depth=8,l2=10,lr=0.06`) | **+0.0295** | **[+0.0150, +0.0596]** | **Real, size unresolved** (straddles 0.05) |
| A3-c ranking loss + rank stopping (λ=1.0) | **+0.0209** | [+0.0040, +0.0354] | Real but too small to matter |
| A1 boosted trees (`depth=8,l2=10,lr=0.06`) | **+0.0178** | [+0.0033, +0.0459] | Real but too small to matter |
| A2 allele-centred targets (`k=100`) | −0.0026 | [−0.0153, +0.0139] | Inconclusive, rules out a worthwhile gain |

**A4 — the two levers compose.** Run as §6c's same-geometry ablation (A1's own
config, `RMSE → PairLogit`, nothing else changed), A4 scores a pooled 0.6805 —
the highest single number anywhere in the study — and improves **62 of 69
alleles** over the control. Against A1 at *identical tree geometry*, so the
objective is the only difference: **+0.0118 aggregate, +0.0175 median
per-allele, 53/69 alleles improved, Wilcoxon p = 1.1e-07.**

**The two statistics disagree here, and both readings belong in the record.**
On the per-allele measure the objective's addition to the tree is clear: +0.0175
and +0.0116 median per-allele difference at the two rungs, each improving a
majority of alleles, Wilcoxon p = 1.1e-07 and 3.9e-05. On the panel median — the
primary metric — it is inconclusive at both rungs. A sign test answers "does this
help most alleles"; the panel median answers "does the median allele's score
move", and those are different questions. Reporting only the first would
overstate the result; reporting only the second would hide a consistent
per-allele effect. §4 is why both are now computed for every contrast.

So the earlier framing — that A1 and A3-c being indistinguishable (−0.0031
[−0.0150, +0.0240]) meant the learner family was not doing the work — was
**half right and half wrong**. Right that trees alone are not better than a
modified MLP. Wrong to infer that the two interventions are alternative routes
to one ceiling: they are orthogonal (A1 changes the learner with the objective
fixed; A3-c changes the objective with the learner fixed) and they **compose**.
The mechanistic reason is that `PairLogit` with `group_id = allele` optimises the
primary metric's exact grouping, and a piecewise-constant tree ensemble is
precisely the model class that most lacks within-group ordering pressure.

The Phase A headline is therefore: **the plain MSE recipe leaves roughly +0.03 on
the table, best reached by combining a boosted tree with a groupwise ranking
objective** — and even that is below the 0.05 bar.

**Caveat.** The +0.0118 A4-over-A1 margin sits near a run-to-run reproducibility
floor of 0.0043 that is **unmeasured for CatBoost arms** (§4). The per-allele
evidence is far harder to attribute to run noise than the aggregate, but a
CatBoost replicate is outstanding and is listed in §6.

### 2.2 Other sequence-internal ranking effects (different references, same family)

These are real sub-bar effects too, each stated against its own reference so the
figures are comparable only within a row.

| Effect | Reference | Δ | 95% CI | Verdict |
|---|---|---:|---|---|
| A0+A1 blend (A5, `inner_oof_30` weights) | A0 (30 members) | +0.0132 | [+0.0014, +0.0258] | Real but too small to matter |
| Rank vs MSE early stopping, A3-c λ=1.0 | MSE-stopping, same cell | +0.0288 | [+0.0012, +0.0279] † | Real but too small to matter |

† The point estimate (+0.0288) lies **outside its own percentile CI upper bound**
(+0.0279). That is legitimate for a shifted percentile bootstrap but signals
bootstrap bias; it should not be quoted without a BCa check, which was not run.

Against **A0 itself** (the 30-member reference) rather than the matched control,
A1 is +0.0072 [−0.0113, +0.0304] — **inconclusive, rules out a worthwhile gain** —
because the A0 pairing additionally carries ensemble arity (12 vs 30 members),
inner-fold count and a cross-platform term (A0 on macOS/Accelerate, the screen on
Linux/OpenBLAS). Had A0 been the only reference, a real effect would have been
reported as a null. This is why §2 of the protocol required two references.

### 2.3 Calibration and representation are not rankable against §2.1

The results below sit on **different metrics or different references** and must not
be placed in the ranking table above.

**Phase B — floor-probability calibration.** Scored on Brier for the floor event
(`thalf_hours == 0`), improvement-signed, against a pre-derived worthwhile bar of
**0.0030** ([phase_b_calibration.md](phase_b_calibration.md) §5.2). Lower Brier is
better; this is not median Spearman.

| Arm | vs B0 (declared σ) | vs Platt baseline | vs isotonic | Clears bar? |
|---|---:|---:|---:|---|
| B2 censored, MSE stopping | +0.0165 [+0.0151, +0.0179] | +0.0100 [+0.0091, +0.0109] | +0.0077 [+0.0068, +0.0086] | **yes, every comparator** |
| B1 censored, NLL stopping | +0.0084 [+0.0067, +0.0100] | +0.0019 [+0.0007, +0.0030] | −0.0004 [−0.0018, +0.0009] | **no — below the bar vs Platt, inconclusive vs isotonic** |

B2 also keeps its ranking: Δ median ρ −0.0108 [−0.0242, +0.0036] against B0 —
**inconclusive, rules out a worthwhile gain** in either direction. B1 loses ranking
conclusively: −0.0515 [−0.0677, −0.0290], **worse**.

**Phase C — frozen representations.** These contrast protein-language-model arms
with each other and with the sequence ensemble; none is a deployable ranking gain,
because the best pLM arm is far below the baseline. Pooled panel; sources
[phase_c_frozen_comparison.md](phase_c_frozen_comparison.md) §1–§2.

| Contrast | Δ median | 95% CI | Reading |
|---|---:|---|---|
| separate → joint (MINT) | +0.0193 | [+0.0038, +0.0830] | Real, size unresolved — but a contrast between two arms both ~0.18–0.26 below baseline |
| esm2 → separate (MINT) | −0.0800 | [−0.1017, −0.0259] | Worse |
| esm2 → joint (MINT) | −0.0607 | [−0.0582, +0.0190] | **median-placement artefact** — CI excludes its own point estimate; sign test says indistinguishable (§4) |

**Phase D — structural residual correction.** Best arm (D4,
geometry+confidence, 109 features) +0.0084 [−0.0055, +0.0124] against the A0
sequence ensemble — **inconclusive, rules out a worthwhile gain**, upper bound a
quarter of the bar. Every other arm is the same reading or a `[0, 0]` degenerate
(no correction selected in all folds). Source
[phase_d_structural_residual.md](phase_d_structural_residual.md) §5.1.

## 3. What each phase established

### Phase A — bounded CPU experiments (complete)

[phase_a_screen.md](phase_a_screen.md),
[phase_a_platform_artifact.md](phase_a_platform_artifact.md).

The plain MSE recipe leaves roughly +0.02 median ρ on the table, reachable either
by switching to boosted trees (A1, +0.0178 vs the matched control) or by adding a
within-allele ranking loss with rank-based early stopping to the reference MLP
(A3-c, +0.0209). The two are indistinguishable from each other, so this is one
gain by two routes, not two gains. Both rule out a 0.05 improvement. Allele-centred
targets (A2) are the screen's one conclusively negative arm — −0.0132 [−0.0310,
−0.0003] against A0, **worse** — so the parameterisation moved ranking downward.
The ESM blend component (A5/ESM), refit at A0's exact 30-member arity, is a null:
+0.0041 over macOS A0 and +0.0027 over Linux A0, both inside the reproducibility
floor, reproducing stage 3's verdict under the stricter protocol. Blend weights are
A0-heavy (mean A0 weight ~0.65–0.73); the A0+A1 blend is +0.0132 [+0.0014, +0.0258]
against A0, real but sub-bar. 1,920/1,920 screen units completed with zero failures
and two independently built assembly paths agreeing to 5.55e-17.

### Phase B — assay-floor modelling and calibration (complete)

[phase_b_calibration.md](phase_b_calibration.md),
[phase_b_declaration.md](phase_b_declaration.md).

The censored likelihood under MSE early stopping (B2) is a real, baseline-beating
improvement in the calibrated probability of being at the assay floor — Brier
0.1099 → 0.0935, fixed-bin ECE 0.0643 → 0.0196, floor AUROC 0.8821 → 0.8938 — and it
clears every comparator, including an oracle σ and a post-hoc Platt map, by five to
fifty times the pre-derived bar. It costs no ranking (−0.0108, worthwhile loss ruled
out). Stage 7a's *declared* arm (censored, NLL stopping; B1) is **rejected**: it
loses ranking conclusively (−0.0515) and its calibration gain does not survive a
two-parameter logistic map. The two-part floor model is dominated on every axis by
B2. A bolted-on Gaussian tail is the wrong functional form for a floor probability
regardless of how its scale is chosen. The phase also corrected the stage 7a
record (§5 retractions).

### Phase C — joint peptide–HLA embeddings (stopped at the gate's conjunction)

[phase_c_gate.md](phase_c_gate.md),
[phase_c_frozen_comparison.md](phase_c_frozen_comparison.md).

Only the base MINT checkpoint (STRING protein–protein interactions, no
peptide–MHC supervision) is a clean upstream. SPEARMINT's two fine-tuned stability
checkpoints trained on the NetMHCstab-pan corpus — our corpus — covering 100% of
our validation and test peptide clusters; its affinity checkpoint covers ≥80.4%
(val) / ≥77.9% (test). A SPEARMINT score on our splits would measure memorisation.
On the clean upstream, frozen joint-chain MINT is statistically indistinguishable
from frozen ESM-2, and all three frozen pLM arms sit 0.18–0.26 below the BLOSUM
sequence ensemble on the same folds. Cross-chain attention does help (+0.0193,
52/69 alleles, Wilcoxon p=5.0e-06) but only recovers what MINT's own pretraining
gave up relative to the ESM-2 it was built from. No frozen arm adds blend signal:
given a free nonnegative weight on held-out folds, the procedure chose zero on 11 of
15 arm-folds. Both halves of §6's stop-scaling conjunction are met, so Phase C
stops. §6.3 (limited adaptation) and §6.4 (affinity-then-half-life) were not run;
§6.4 was descoped to a separate unbudgeted sub-project because no released affinity
stage is leakage-clean.

### Phase D — reuse structures before expanding compute (complete, $0)

[phase_d_structural_residual.md](phase_d_structural_residual.md),
[phase_d_protocol.md](phase_d_protocol.md).

A strongly shrunk structural residual correction fitted on sequence out-of-fold
errors does not improve cluster-held-out ranking, and a worthwhile gain is ruled
out: best arm +0.0084 [−0.0055, +0.0124]. Both anchor-focused arms (P2/P9, and the
P1 extension) selected **no correction in all five folds**, and the central-position
control (P3–P8) reached no-correction in four of five — so neither the anchors nor
the centre nor the whole 109-feature block carries ranking signal a 30-member
sequence ensemble has not already extracted. Both directions of peptide–HLA pair
ipTM are already present in all 28,166 production pairs (Spearman 0.726 between
them, so not redundant), so no folding is justified to acquire them. A single
predicted pose already recovers a median 99.1% of each feature's usable
within-allele signal, so more poses are not worth generating for feature stability.
Phase D's D0 reproduces Phase A's A0 element-wise to four decimals on all five
folds — two independently written harnesses on the same folds, the strongest
provenance check in the study. This closes the plan's §7 gate in the negative on
existing data, at zero compute cost.

## 4. Methodological findings

These are arguably the study's most durable output, because each is a property of
the task or the evaluation machinery rather than of any one arm.

**The run-to-run reproducibility floor (0.0043) exceeds the platform effect it was
commissioned to measure.** The A0 Linux re-run was asked to quantify a
macOS-versus-Linux artifact. Two runs of the same five units at the same settings
on the same platform disagree by **0.0043** on median-of-medians — larger than the
macOS→Linux shift (+0.0013) and larger than either of its separated components
(platform −0.0020, threads +0.0034, which nearly cancel). The mechanism was
identified, not inferred: Modal CPU containers split across two microarchitectures
(`avx,avx2` vs `avx,avx2,avx512dq,avx512f`), and OpenBLAS dispatches a different
kernel per microarchitecture, changing float32 accumulation order. Units on the same
host class reproduce bit-for-bit (3 of 5 folds); others do not. It is not fixable by
pinning threads or seeds — only by replication. The consequence the whole study
inherits: the paired peptide-cluster bootstrap **treats each arm's predictions as
fixed**, so its intervals do not cover this floor. An effect smaller than ~0.004 is
not distinguishable from a re-run of the same arm, and the floor is **unmeasured for
the CatBoost arms**, which do not use the same BLAS paths.
([phase_a_platform_artifact.md](phase_a_platform_artifact.md).)

**The difference of medians is not the median of differences.** The primary metric
is a median over a fixed allele panel, so a large gap in medians can come entirely
from which allele sits at the median in each arm. The tell is a bootstrap CI that
**excludes its own point estimate**: Phase C's esm2 → joint contrast gave −0.0607
against an interval of [−0.0582, +0.0190], while the median *per-allele* difference
was only −0.0097 and the sign test gave p=0.33. `describe_delta`'s verdict could not
be read as a per-allele effect. Adopted study-wide: report a paired sign test and
the median of per-allele differences alongside every difference of medians, and
treat a CI that excludes its point estimate as a stop sign. Phase A's headline was
re-checked against this and strengthened — difference of medians +0.0178, median of
per-allele differences +0.0200, 54/69 alleles improved, sign test p=2.6e-06.
([phase_c_frozen_comparison.md](phase_c_frozen_comparison.md) §2.1.)

**Isotonic calibration destroys rank resolution while preserving rank order.**
Isotonic regression is only weakly monotone — a step function that maps strict
inequalities to equalities — so it never inverts a pair but collapses distinct
predictions into ties, and those ties move Spearman because the frozen contract
gives ties average ranks. Measured on this project's own predictions, isotonic
collapsed 2,776 distinct predictions to 57 and shifted per-allele Spearman between
−0.0431 and +0.1109, exceeding 0.05 on 3 of 68 alleles; on a 40-row allele it
collapsed 40 predictions to 7 and moved Spearman by 0.23. Both signs occur within
one fit, so this is a confound, not a conservative bias. The fix is not to find a
better monotone map — every weakly monotone map does this — but to score the primary
metric on **uncalibrated** predictions and confine any map to magnitude metrics. A
strictly monotone affine rescale is provably rank-safe, but only at fold scope: a
pooled vector is several affine transforms stitched together, so an assertion placed
at pooled scope correctly fires (measured pooled shift ≤0.0012).
([README.md](README.md) decision log, 4 Oct; Phase D §5.7.)

**Machinery silently substitutes its own rule for the declared one.** Two instances.
CatBoost resolves its early-stopping metric from the objective, so the withdrawn
A4/YetiRank arm would have stopped on **NDCG** — a top-heavy, position-discounted
metric nobody declared — reproducing exactly the stopping-surface confound A3 exists
to prevent; the resolved `eval_metric` is now recorded on every tree row. And the
Modal image `.env` thread pin **reported success while failing**: all five A0
containers reported `OMP_NUM_THREADS=2` while dispatched as threads=1, caught only by
the `thread_pin_ok` field, which is why any arm reporting a thread count must carry
and check that field. ([phase_a_prespecification.md](phase_a_prespecification.md)
§6a; [phase_a_platform_artifact.md](phase_a_platform_artifact.md) §6.)

**A leak can be invisible in the metric and visible only in a row count.** Phase B's
Platt baseline was first fitted once on the pooled `inner_oof_30` rows across all
five outer folds and applied to the pooled outer-held predictions. Because a row is
outer-held in one fold and inner-OOF in the other four, the pooled fit used every
row's own label to build the map that then predicted it. The leaked and corrected
versions gave the same Brier to four decimal places; the only tell was a log line
reporting 19,716 fit rows where a per-fold fit can see at most 15,772. A metric check
cannot see this class of error — a disjointness assertion can.
([phase_b_calibration.md](phase_b_calibration.md) §8.)

**Seed and fold variation on this dataset match the effects being claimed.** A0's
five outer folds span 0.048 with the model held fixed — essentially the whole 0.05
bar — and stage 7a's single-seed sensitivity sweep varied by 0.0095 against a
per-seed spread of 0.028–0.043. So single-seed sweeps and unpaired fold comparisons
carry no information here; only paired comparisons on shared folds and shared
resamples do. A related orchestration lesson: a fit-row guard should assert a
**partition identity** (fit and predicted rows disjoint and together reconstituting
the frame), not a magic constant — a circulated "assert 15,772 rows" was correct for
only one of five water-filled folds and failed four of thirty ESM units.
([phase_b_calibration.md](phase_b_calibration.md) §1; [README.md](README.md), 4 Oct
orchestration-error entry.)

## 5. Retractions

The study corrected itself repeatedly, and the audit trail is part of the
deliverable. Each entry states what was claimed, what was measured, and what stands.

**The stopping rule has no stand-alone ranking effect.** *Claimed:* rank-based early
stopping was the screen's sharpest, most transferable finding, winning all 8 matched
cells on median-of-medians. *Measured:* paired intervals resolve the effect only in
the two cells that also carry a ranking gradient (A3-c +0.0288, A3-d +0.0085, both
real sub-bar); in the two pure-MSE cells it is centred near zero and inconclusive,
and A3-a flips sign. The apparent additivity of the decomposition was an aggregation
artifact — median-of-medians and the pooled paired quantity disagree in sign on the
decisive cell. *Stands:* the transferable part is the **ranking loss**, with
rank-based stopping useful only alongside it. This is an interaction, not a
stand-alone effect. ([README.md](README.md), 4 Oct "RETRACTED — the stopping rule
has no stand-alone effect".)

**The `inner_oof_30` amendment was counterproductive for its stated purpose.**
*Claimed:* emitting a 30-member-equivalent inner out-of-fold set at 5× the networks
(150 → 900) would better match the outer-held ensemble's arity for fitting residual
and blend models. *Measured:* `inner_oof_30` sits at residual-sd ratio 1.0342 to
`outer_held` against `inner_oof_6`'s 1.0193 — it is **further** from the target, by
1.8×, because the extra nesting level cuts fit rows (12,618 → 10,095) by more than
the arity gain is worth. *Stands:* the arity confound was real, but the remedy was
not weighed against its training-size cost. A5 fits its blend weights on
`inner_oof_6`; Phase D keeps `inner_oof_30` as its pre-declared primary with
`inner_oof_6` as a sensitivity check (and the verdict is unchanged, if anything more
negative). The 750 extra networks bought the measurement that showed the remedy was
wrong. ([README.md](README.md), 4 Oct "the `inner_oof_30` amendment was
counterproductive"; Phase D §5.4.)

**Phase C's epoch turnover, and the figures that rested on it.** *Claimed:* §6.3
adaptation costs a measured $30.59 and "tops out at 0.47–0.48", because early
stopping bit at epoch 9–10. *Measured:* the read was taken from a mid-run Volume
checkpoint before epochs 11 and 12 existed; epoch 10 was a single-epoch blip and
epoch 12 set a new minimum, still improving at the 12-epoch cap. *Stands:* §6.3's
cost is unresolved with a floor of $39.72 (not $30.59) and the 20-epoch assumption is
not refuted; 0.4890 at the cap is a **floor at 12 epochs, not a ceiling**, and the
adaptation trajectory is upward and unmeasured. Both corrections cut slightly in
favour of §6.3, and the recommendation to stop Phase C is unaffected because it rests
on the frozen comparison and the blend check, not on epoch count. The $30.59 and the
ceiling claim had both been circulated, including to the user.
([phase_c_gate.md](phase_c_gate.md) §5.4.)

**The arity figure was overstated roughly fivefold.** *Claimed:* ensembling at the
reference's arity is worth +0.07 to +0.09 median Spearman, used to justify the parity
requirement and the `inner_oof_30` cost. *Measured:* +0.07–0.09 is the
*single-network-versus-ensemble* comparison, a different quantity. The arity change
the argument needed — 12 members against 30 — is worth **+0.0156** (matched control
0.6337 against A0's 0.6493), and Phase B independently measured 10 → 30 at +0.0081.
*Stands:* ensemble parity is still the right requirement, but the magnitude cited for
it was about five times too large, and the overstatement is part of why the
`inner_oof_30` remedy looked worth 5× the networks. Corrected in §1 and §A0 of the
prespecification rather than deleted. ([phase_a_prespecification.md](phase_a_prespecification.md)
§1; [phase_a_screen.md](phase_a_screen.md).)

Three further retractions are recorded in the phase reports and summarised here so
the trail is complete: Phase B withdrew stage 7a's "gives back almost all of the
ranking" (the matched-budget estimate is −0.0160 [−0.0333, +0.0248], inconclusive,
not "recovered") while "roughly halving the calibration error" survived and
strengthened; Phase D retracted a reported ranking-versus-MSE tradeoff in the
structural correction (its two selection criteria agreed on direction and differed
only on whether a sub-SE MSE gain was resolvable — the 1-SE rule working as declared,
not objectives in conflict); and the floor-event amendment withdrew two wrong
supporting arguments the same day it was made (a "443 rows / 1.57 pp" figure that
compared `<=` against `<` when the real gap is one row, and a float-equality hazard
that does not exist on the `pepstab.censored.censored_mask` path).

## 6. What to do next

The recommendations formerly listed here are **superseded**. Use the
[current research plan](../NEXT_RESEARCH_PLAN.md) for the reviewed evidence,
prioritised work queue, stopping rules and deferred proposals. The original
recommendations are preserved in the
[proposal archive](../archive/plans/proposals-retired-2026-10-12.md).

## 7. Cost

About **$7 realised across all four phases**, Modal plus free laptop CPU, every run
priced before launch and logged append-only in
[../compute_ledger.csv](../compute_ledger.csv). Phase D cost **$0.00** and still
closed a gate (all structural numbers read off the committed stage 4c5 extraction).

| Phase | Realised | Notes |
|---|---:|---|
| A | ~$4.26 | screen $2.45 (priced before launch; realised not separately logged), A5/ESM batch $0.64, A0 Linux refit $0.70, smokes $0.03, plus $0.44 wasted |
| B | $0.216 | 450 + 300 networks and two smokes |
| C | $3.14 | gate $0.05, probes $0.06 + $1.00, extraction $0.45 + $0.21 partial, plus $0.78 wasted |
| D | $0.00 | no GPU, no Modal, no new folding |

Two losses are recorded honestly rather than buried. **$0.78** was lost to an
external cancellation of four detached Modal apps across three workstreams inside a
7-second window during a machine-level network outage; why a genuinely detached app
received a cancellation signal is recorded as **unresolved** rather than given a
plausible mechanism, and the operational lesson — a long run must checkpoint partial
state, not rely on `--detach` — was adopted (the extraction kept 45% of its work, the
epoch probe lost 100% of 42 minutes). **~$0.44** was lost to three self-inflicted
Modal launch errors (a builder-only run, a chunk that would exceed the function
timeout, and a `ConflictError` from relaunching an app before it reached `stopped`).

A note on the Phase A screen figure: the ledger carries its $2.45 as priced before
launch; the screen completed (1,920/1,920 units) but its realised Modal cost is not
logged as a separate measured row, so the per-phase total above uses the pre-launch
price. The realised grand total is therefore approximately $7.6, of which about
$1.22 is accounted loss or pre-launch estimate.

## 8. Two numbers a reviewer will see disagree, and why they do not

- **The sequence reference reads 0.6493 and 0.6615 in different places.** 0.6493 is
  the across-fold median of per-fold medians; 0.6615 is the panel median computed
  once over all 19,716 pooled out-of-fold rows. Phase C quotes 0.6615, Phase A and
  Phase D quote 0.6493 as the headline and 0.6615 (0.66152) in their pooled tables.
  Same predictions, different aggregation; the pooled figure is the one the bootstrap
  intervals attach to, and the across-fold figure is the one comparable to the
  published work. Neither is wrong.
- **The matched control reads 0.6337 and 0.6510.** 0.6337 is its five-fold
  median-of-medians; 0.6510 is its pooled panel median, which is the one the §2.1
  intervals are taken against.

**Which aggregation each interval belongs to, stated explicitly**, because this is
the version that can actually mislead. The first case above is two documents
quoting different aggregations of the same quantity — visible and self-correcting.
The second is a single table **attaching an interval to one aggregation while
reporting another nearby**, which a reader has no way to detect:

| quantity | aggregation | used for |
|---|---|---|
| 0.6493 / 0.6337 / 0.6534 | five-fold median of per-fold medians | headline scores, fold-spread statements, comparison with published work |
| 0.6615 / 0.6510 | panel median over all 19,716 pooled rows | **every Δ and every 95% CI in §2.1**, and every `describe_delta` verdict |

So a Δ of +0.0178 is `pooled(A1) − pooled(control)`, not `0.6611 − 0.6337`. The
difference-of-medians arithmetic on the across-fold figures gives +0.0274 for the
same arm, and that number appears nowhere in the interval tables. Any future table
mixing the two would produce a Δ whose interval belongs to a different statistic.

Carrying both aggregations, with the reference named on every figure, is the house
convention and is deliberate — not a contradiction to reconcile. What is **not**
acceptable is quoting a Δ from one aggregation beside a CI from the other, which is
the failure this section exists to prevent.
