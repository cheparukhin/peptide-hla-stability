# Phase B declaration — assay-floor modelling and calibration

Date: 4 October 2026. Status: **arms, objectives and decision rules recorded
before any Phase B model was fitted.** Nothing in §1–§8 is edited after the
first fit; extensions are appended to §9 with date and reason.

This file exists because Phase B's job is to resolve a comparison that stage 7a
left unresolved, and a resolution chosen after seeing the numbers would be worth
nothing. [FURTHER_RESEARCH_PLAN.md](../FURTHER_RESEARCH_PLAN.md) §5 requires the
decision rule and the optional fourth arm to be fixed in advance; this is that
record.

## 1. What stage 7a actually established, verified against source

Every number below was recomputed from `preds/stage7_{mse,censored,censored_devmse}.csv`
and the scales in `reports/stage7_censored_headline.json`, not copied from the
report prose. Floor probabilities use the plug-in rule `Phi((c - mu)/sigma)` for
both arms, `c = log1p(0.1)`, on all 2,817 validation rows; ranking uses the
68-allele eligible panel (2,802 rows).

| arm | members | median per-allele ρ | Brier | ECE (equal-count deciles) | censored NLL |
|---|---:|---:|---:|---:|---:|
| `log1p`-MSE | 30 | 0.6931 | 0.1050 | 0.0560 | 1.0452 |
| censored, stopped on dev NLL | 30 | 0.6518 | 0.0984 | 0.0423 | 1.0764 |
| censored, stopped on dev MSE | **10** | 0.6910 | 0.0930 | 0.0237 | 1.1134 |

The headline Δ median per-allele ρ = **−0.04136, 95% CI [−0.07801, −0.00619]**
reproduces exactly, and the verdict **"worse: CI entirely below 0"** stands. The
reduced-budget sensitivity line (0.6975 / 0.6657 / 0.6910) reproduces from
`reports/stage7_censored_sensitivity.csv`.

### 1.1 Three defects in the stage 7a record that Phase B must not inherit

These are corrections to a published report, so they are stated here rather than
quietly fixed. All three were confirmed against
`reports/stage7_censored_runs.csv`.

1. **§6.4's table is budget-mismatched.** Its ranking column is the reduced
   10-network budget (seed 0 only, 5 inner folds × 2 encodings); its Brier, ECE
   and censored-NLL columns for the first two rows are the **30-network**
   values. Only the dev-MSE row is internally consistent. So the claim that the
   dev-MSE stopping rule "roughly halv[es] the calibration error" compares a
   10-network arm against a 30-network arm, and ensembling is worth roughly
   +0.07–0.09 median Spearman in this project — larger than most effects being
   looked for. The direction of the calibration difference may well survive; the
   magnitude as published does not follow from a matched comparison.
2. **`reports/stage7_censored_calibration.csv` is stale.** It lacks the
   `p_floor_brier` and `p_floor_ece` columns that the current
   `scripts/stage7_censored.py` writes, so the report's headline calibration
   numbers are not reproducible from the shipped artifact. They *are*
   reproducible from the prediction files, which is how they were verified here.
3. **The two arms used different `P(floor)` constructions.** The script gives
   the MSE arm the plug-in `Phi((c - mean_k mu_k)/sigma_bar)` and the censored
   arm the mixture `mean_k Phi((c - mu_k)/sigma_k)`. The shipped reliability
   table holds the mixture for the censored arm, so §6.2 narrates mixture
   deciles (0.816 predicted against 0.790 observed in the top decile) in the
   same breath as a plug-in ECE. Under the mixture the censored arm's
   equal-count ECE is **0.0474**, not 0.0423 — which turns the narrated
   "25% ECE improvement" into **15%**. Phase B declares one construction (§4)
   and reports what the other would have given.

Nothing here is a leak: the MSE comparator's scale is
`sqrt(mean(inner-dev MSE))`, estimated on training folds only.

## 2. The floor event

> **The floor event is `thalf_hours == 0`** — the recorded assay-floor code.
> 5,679 rows, **20.1626%** of the dataset, which is exactly the "20.2% of labels
> sit at the assay floor" that [EVALUATION.md](../../EVALUATION.md) names.

A probability target should be an observable, and the recorded 0 is one. Defining
the event this way means no part of the calibration metric depends on where the
detection limit is *believed* to be, which matters because stage 1 locates that
limit in a range (0.1–0.2 h) rather than at a point.

**The loss's censoring rule is unchanged** at stage 7a's declared
`thalf_hours < 0.1` (`pepstab.censored.censored_mask`). The two definitions
therefore disagree, on **exactly one row of 28,166**: `pair_id` 1189,
HLA-A*24:02 / WYGMEIRPL, measured at 0.05 h, `cluster_id` 4704, in the training
split. Validation and test contain no rows in `(0, 0.1)`, so the two rules
coincide there exactly.

This disagreement is a deliberate carry, not an oversight. Changing the *loss* to
match would make these arms non-comparable to the stage 7a result Phase B exists
to resolve. A one-row sensitivity check is reported for every calibration number
so the reader can see the gap is immaterial rather than being asked to assume it.

Two things that are **not** reasons for this choice, recorded so they are not
later mistaken for reasons: there is no floating-point hazard at the boundary
(`censored_mask` compares hours, the stored 0.1 is the exact double `0.1`, and
`0.1 < 0.1` is deterministically `False`), and the 443 rows recorded at exactly
0.1 h are measurements under either rule, as stage 7a §3 declares. The gap
between the two candidate definitions is one row, not 443.

## 3. Arms — exactly three, plus two baselines

| ID | Arm | Differs from the reference in |
|---|---|---|
| B0 | `log1p`-MSE reference | — (the reference) |
| B1 | censored loss, stopped on dev **NLL** | objective + its consistent stopping rule |
| B2 | censored loss, stopped on dev **MSE** | objective only |

B1 is stage 7a's declared arm. B2 isolates the stopping rule from the gradient:
stage 7a §6.4 located the ranking loss there, with the censored arm's own dev
objective turning over at about epoch 10 against the MSE arm's 27.

Two calibration baselines, so that any distributional model has to beat the
simpler solution:

| ID | Baseline | Fitted on |
|---|---|---|
| C1 | **Platt** scaling of the floor indicator on B0's `mu` | inner out-of-fold rows, training partition only |
| C2 | Gaussian scale, `P = Phi((c - mu)/sigma)` | inner out-of-fold residuals, training partition only |

C1 is the declared baseline of the pair; C2 is stage 7a's comparator
construction, retained for continuity. Isotonic regression is reported as a
secondary variant of C1 — more flexible, correspondingly easier to overfit an
inner fold.

### 3.1 A ranking-stopped censored arm is DECLINED

Recorded now, before the phase starts, as plan §5 requires — not deferred.

1. It needs a new stopping criterion in `pepstab/censored.py`, whose
   `STOPPING_RULES` is `("nll", "mse")`. Adding one breaks the property that
   makes the stage 7a comparison literal rather than approximate: that the arms
   differ in the objective and nothing else.
2. It has little room to act. Stage 7a found precision@10 **identical at 0.700
   for every arm and every threshold**, and floor AUROC flat at 0.885–0.886.
   Ranking *within* the tied floor block is unidentifiable under any objective,
   because those labels carry no order.
3. It would add a third selection axis to a ranking claim Phase B is explicitly
   **not permitted to make** (§7).

## 4. `P(floor)` construction — one rule for every arm

> **Declared: the mixture, `P(floor) = mean_k Phi((c - mu_k)/sigma_k)`** over
> ensemble members (`pepstab.calibration.prob_floor_mixture`).

It is the predictive probability of the mixture the ensemble actually is;
averaging member probabilities keeps between-member disagreement instead of
discarding it, where the plug-in collapses the ensemble to one Gaussian and is
consequently more confident in both tails.

Where an arm has no per-member scale — the MSE reference, and baselines C1/C2 —
the member-level probability is formed first and then averaged, so every arm is
averaged at the same level. The plug-in value is reported alongside for every arm
as a declared sensitivity, because stage 7a's numbers were plug-in and the two
differ materially (0.0474 against 0.0423 on the same predictions).

## 5. Metrics

**Primary calibration objective: Brier score for the floor event.** Lower is
better; the comparison statistic is the paired Δ against the matched reference.

Brier because it is **strictly proper**: the only way to improve it is to be
better calibrated without being worse at discrimination. ECE is not proper — a
predictor that returns the base rate on every row has an ECE near zero and no
discrimination at all (pinned in
`tests/test_calibration.py::test_ece_cannot_see_a_base_rate_predictor_but_brier_can`),
so ECE cannot be an objective.

| Role | Metric |
|---|---|
| **Primary** | Brier score, floor event |
| Diagnostic | **fixed-bin** ECE (10 equal-width bins, occupancy reported), reliability table with Wilson intervals |
| Diagnostic | floor AUROC — discrimination, the identifiable part of floor behaviour |
| Secondary | censored NLL; log loss |
| Reported, never used to select | equal-count-decile ECE, for continuity with stage 7a |
| Ranking | median per-allele Spearman, through `pepstab.evaluation` verbatim |

**Fixed-width bins, not equal-count.** Stage 7a used equal-count deciles, whose
edges depend on each arm's own predictions, so two arms are binned differently
and part of any ECE difference is the binning. Fixed edges remove that. The cost
is sparse bins when an arm concentrates its predictions, so bin occupancy `n` is
reported next to every observed rate — an observed rate over nine rows is not
evidence of miscalibration.

**Censored NLL is secondary, and reported *because* it disagreed with Brier and
ECE at stage 7a.** The disagreement is structural, not a bug: censored NLL sums a
Gaussian **density** term over measured rows with a **probability** term over
floor rows, so an arm that sharpens its floor probabilities by shrinking `sigma`
is charged for the sharper density on every measured row. Brier scores only the
floor event and is not subject to that trade. The two are minimised at different
scales on the same predictions
(`tests/test_calibration.py::test_nll_and_brier_are_minimised_at_different_scales`).
Naming that mechanism is the useful output here, not the number.

## 6. Assessment protocol

Training-only nested cluster validation, per plan §3.2. **The test split is not
read. The frozen validation split is not used to select anything**; it appears
only in the retrospective correction of the stage 7a record (§8), which scores no
new model selection.

- **Folds:** loaded from `reports/further_research/phase_a_fold_assignments.csv`,
  never recomputed. Independently verified here: 98,580 rows, every `pair_id`
  exactly five times, `pair_id` set exactly the 19,716 training rows,
  `inner_fold == -1` outer-held counts 3944/3943/3943/3943/3943, **zero clusters
  spanning more than one outer fold and zero spanning more than one inner fold**
  within any outer fitting partition.
- **Budget:** `pepstab.nested.fit_mlp_ensemble`, 5 inner folds × 2 encodings × 3
  seeds = 30 members per outer fold, `selected = {"onehot": ((256,64), 1e-5),
  "blosum": ((256,64), 1e-3)}` — A0's exact budget, inherited rather than
  transplanted. Five outer folds × 30 members × 3 arms = **450 networks**.
- **Eligibility, frozen before fitting:** an allele is eligible on a fold with
  **≥ 20 rows and ≥ 2 distinct labels**, from labels and counts alone. A **≥ 50**
  bar is reported for every arm as a declared sensitivity check. Measured on the
  outer-held folds: at ≥ 20, 68 alleles per fold covering 99.3–99.7% of rows; at
  ≥ 50, 41–49 alleles covering 69.4–78.5%. Coverage is reported per fold.
- **Intervals:** paired peptide-cluster bootstrap, **2,000 resamples, seed
  20261003**, both arms scored on the same resample. Ranking goes through
  `pepstab.evaluation.paired_cluster_bootstrap`; calibration metrics through
  `pepstab.calibration.paired_cluster_bootstrap_metric`, which uses the same
  resampling contract and returns the delta **signed as an improvement**, so
  `describe_delta`'s six verdicts read off it with no sign juggling.
- **Floor-event share per outer-held fold:** 0.1910–0.2077, so no fold is
  unrepresentative of the 20.2% overall.
- **Scales:** the post-hoc baseline's `sigma` comes from `inner_oof_30`
  residuals, **not** from the per-member `dev_mse`. `dev_mse` is each member's
  early-stopping-fold MSE at its best epoch — the quantity early stopping
  selected on — so it is optimistic as an estimate of out-of-sample spread, which
  is the same selection-optimism argument this whole phase turns on. The
  `dev_mse`-derived scale is reported alongside as the stage-7-comparable
  variant.
- **Threads:** BLAS pinned to one thread before `numpy` is imported. The project
  has recorded that thread count moves a single network's result by about 0.02
  median Spearman at fixed seed; under concurrent load that becomes
  nondeterministic rather than a constant offset, which is the worst case for a
  paired comparison.

## 7. Decision rule — fixed before fitting

Phase B keeps this branch only if **both** limbs are satisfied.

### Limb 1 — the calibration objective improves

The paired Δ Brier (improvement-signed) against the matched reference must have
its **95% CI lower bound above 0**, and the point estimate must reach the
worthwhile bar defined below.

**The worthwhile-Δ-Brier bar is declared as a rule, not a number**, because no
analogue of EVALUATION.md's 0.05 exists for Brier and inventing one after seeing
a result would be exactly the post-hoc selection this phase exists to avoid. The
rule, by the same route EVALUATION.md derived 0.05 — the width of the interval
under no true difference:

> The bar is the **maximum** 95% paired cluster-bootstrap half-width under no
> true difference, computed by `pepstab.calibration.null_half_width` on the
> harness's own outer-held row set, 2,000 resamples, seed 20261003, 8 null
> splits, rounded **up** to two significant figures. **Computed and recorded
> before any censored arm is scored.**

### Limb 2 — the ranking tradeoff is acceptable

Chosen in advance, not after seeing it:

> Δ median per-allele Spearman against the matched reference must have a **point
> estimate ≥ −0.02** *and* a **95% CI lower bound > −0.05**, i.e. a worthwhile
> ranking loss is ruled out.

The −0.05 limb is the mirror of EVALUATION.md's "rules out a worthwhile gain",
using the project's own worthwhile unit in the harmful direction. The point-estimate
limb stops an arm being accepted merely because its interval is too wide to
resolve a visible loss.

**Sanity check that the rule has teeth:** stage 7a's declared censored arm fails
both limbs (point −0.0414, lower bound −0.0780). A decision rule the status quo
passes trivially would be worthless.

### What may not be claimed

> **Calibration improvement does not count toward the +0.05 ranking claim.**

The two live in separate sections of the write-up. An arm that improves Brier and
loses ranking is reported as exactly that, and the +0.05 worthwhile-gain bar is a
statement about median per-allele Spearman only.

## 8. The two-part floor model

Probability of exceeding the floor, times conditional magnitude above it.

The arithmetic to be careful about, stated before fitting: for predicting the
**recorded** `log1p` labels,

```
E[log1p(t)] = p(above floor) * E[log1p(t) | above floor]
```

treats the floor-coded component as contributing zero. That is correct as an
expectation **of the recorded label**, because the recorded label *is* 0 there.
It is **not** an estimate of true sub-floor lifetime, and it is **not** generally
the optimal ranking score — multiplying a probability by a conditional magnitude
can reorder pairs that the conditional magnitude alone ranks correctly.

Declared consequences:

- Its ranking is compared against the ordinary regressor's and reported
  explicitly, whichever way it falls.
- Any statement about physical lifetime below the floor retains a **censored
  latent component**; sub-floor lifetime is never equated to zero.

## 9. Amendments

None yet. Any extension of an arm, a grid, a seed count or a budget is appended
here with its date and reason.

## 10. Retrospective correction of the stage 7a record

Separate from the Phase B assessment, and scored on the frozen validation split
because that is where the stage 7a numbers live. **No new model is selected on
it.** Its purpose is to give the dev-MSE control the paired interval it never
had, at a matched budget.

Procedure: refit **B2 at the full 30-member budget** under stage 7a's protocol
exactly, so that all three arms sit at 30 members and the existing 30-member B0
and B1 predictions can be reused. Then report paired intervals for Brier,
fixed-bin ECE, censored NLL, floor AUROC and median per-allele Spearman, and
state separately what the published mismatched comparison showed — so a reader
can see how much of "roughly halving the calibration error" was budget rather
than loss.
