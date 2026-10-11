> **Historical study record.** The protocols, results and future-tense instructions below describe the completed October study. The [current project review and research plan](../NEXT_RESEARCH_PLAN.md) is the sole forward work queue; its [dated correction register](../NEXT_RESEARCH_PLAN.md#dated-correction-register--12-october-2026) supersedes conflicting interpretation, selection and blend-gate claims here. No old protocol should be relaunched as a new study without the current evaluation gates.

# Phase B — assay-floor modelling and calibration

Date: 4 October 2026. Status: **complete.** Protocol, arms and decision rules
were fixed in [phase_b_declaration.md](phase_b_declaration.md) before any Phase B
model was fitted; nothing in this report chooses anything.

Scope: the training split only, assessed on the five frozen outer
training-cluster folds, plus a retrospective correction of the stage 7a record
on the frozen validation split. **The test split was not read.**

Two claims are kept strictly apart throughout, as the declaration requires:
calibration improvement is reported on its own and **never counted toward the
project's +0.05 ranking bar**.

---

## 1. The finding that generalises beyond this phase

Before anything about censored likelihoods: **stage 7a's reduced-budget
sensitivity analysis was run at seed 0 only, and seed 0 is the best of the three
seeds for all three arms.**

Per-seed median per-allele Spearman on the validation panel, each a 10-member
ensemble (5 inner folds × 2 encodings × 1 seed):

| arm | seed 0 | seed 1 | seed 2 | spread | all 30 members |
|---|---:|---:|---:|---:|---:|
| `log1p`-MSE | **0.6975** | 0.6798 | 0.6542 | 0.0433 | 0.6931 |
| censored, dev-NLL stop | **0.6657** | 0.6387 | 0.6283 | 0.0374 | 0.6518 |
| censored, dev-MSE stop | **0.6910** | 0.6745 | 0.6633 | 0.0277 | 0.6771 |

Source: [phase_b_stage7_seed_spread.csv](phase_b_stage7_seed_spread.csv).

Stage 7a §6.3 reports that "across a 6-fold range of the detection limit the
median rho spans **0.0095**" and concludes the threshold is not load-bearing.
That span was measured at one seed whose own spread is **0.0277–0.0433**, three
to four times larger. EVALUATION.md's own rule applies directly: *"A gap smaller
than seed spread is not a result."*

**This is a statement about the method, not about that table.** Any single-seed
sensitivity sweep in this project inherits the same defect, because the seed
spread here is comparable to every effect the project is trying to measure. The
claim being made is **not** that stage 7a's threshold conclusion is false — it
may well hold — but that it was established on a basis that cannot support it.
Those are different claims and only the second is defensible from this evidence.

It compounds with the partition-level analogue Phase A measured: with the model
held fixed, A0's five outer folds span **0.6281 to 0.6765, a range of 0.048** —
essentially the entire 0.05 worthwhile bar, from fold assignment alone. Taken
together, **on this dataset both seed and fold variation match or exceed the
effects being claimed, so only paired comparisons on shared folds and shared
resamples carry information.** Every comparison below is of that form.

---

## 2. Corrections to the stage 7a record

Stage 7a's conclusion §6.5.5 — "the likely fix is the stopping rule, not the
loss" — rested on a control it correctly labelled as having no confidence
interval. It had a second problem it did not label.

### 2.1 The defect

[phase_b_declaration.md §1.1](phase_b_declaration.md) documents three, all
confirmed against `reports/stage7_censored_runs.csv`. The consequential one:

> **§6.4's table is budget-mismatched.** Its ranking column is the reduced
> 10-network budget (seed 0 only); its Brier, ECE and censored-NLL columns for
> the first two rows are the **30-network** values. Only the dev-MSE row is
> internally consistent.

So the published claim that dev-MSE stopping "gives back almost all of the
ranking ... while roughly halving the calibration error" compares a 10-network
arm against a 30-network arm on the calibration side.

How much does arity actually move things? Measured on these same members,
10 → 30 is worth **+0.0159 (B0), +0.0075 (B1), +0.0008 (B2)**, mean +0.0081
([phase_b_arity_effect.csv](phase_b_arity_effect.csv)), and Phase A
independently measures +0.0156 for 12 → 30 on its own screen. That is **smaller
than the seed spread in every arm**, so the budget mismatch is not large in
itself. It is still a defect: a comparison between arms of different size is not
a comparison, and §2.3 shows it moved the headline ranking estimate by a factor
of 2.4.

Two smaller ones: `stage7_censored_calibration.csv` is stale (it lacks the
`p_floor_brier`/`p_floor_ece` columns the current script writes, so the report's
calibration headline is not reproducible from the shipped artifact), and the two
arms used **different `P(floor)` constructions** — plug-in for the MSE arm,
mixture for the censored arm — so §6.2 narrates mixture reliability deciles
alongside a plug-in ECE. Under the mixture the censored arm's equal-count ECE is
**0.0474**, not 0.0423, which turns the narrated "25% ECE improvement" into
**15%**.

No leak was found: the MSE comparator's scale is `sqrt(mean(inner-dev MSE))`,
training-only.

### 2.2 The corrected comparison

All three arms refitted at the **same 30-member budget** under stage 7a's
protocol, every member's `mu` and `sigma` retained so one declared `P(floor)`
construction applies to all of them, and every comparison given a paired
peptide-cluster bootstrap interval (2,000 resamples, seed 20261003).

**Fidelity check first.** The refit reproduces the shipped prediction files to
max |diff| **2.2e-07** (`preds/stage7_mse.csv`) and **1.4e-07**
(`preds/stage7_censored.csv`), so nothing below is a protocol difference.

Validation split, 2,817 rows, 553 at the floor (19.63%), 68 eligible alleles,
declared mixture `P(floor)`:

| arm | members | median ρ | Brier | ECE (fixed bin) | censored NLL | mean σ |
|---|---:|---:|---:|---:|---:|---:|
| `log1p`-MSE | 30 | 0.6931 | 0.1058 | 0.0587 | 1.0452 | 0.724 |
| censored, dev-NLL stop | 30 | 0.6518 | 0.0992 | 0.0440 | 1.0764 | 0.755 |
| censored, dev-MSE stop | 30 | 0.6771 | **0.0924** | **0.0217** | 1.0953 | 0.588 |

Source: [phase_b_stage7_retrospective_metrics.csv](phase_b_stage7_retrospective_metrics.csv),
[_paired.csv](phase_b_stage7_retrospective_paired.csv).

### 2.3 "Gives back almost all of the ranking" is withdrawn

Dev-MSE arm against the MSE reference, Δ median per-allele Spearman, both sides
at the same budget:

| budget | Δ median ρ | 95% CI | verdict |
|---|---:|---|---|
| stage 7a as published | −0.0065 | **none** | — |
| 10 members, matched | −0.0066 | [−0.0411, +0.0246] | inconclusive, rules out a 0.05 gain |
| **30 members, matched (finalist)** | **−0.0160** | [−0.0333, +0.0248] | inconclusive, rules out a 0.05 gain |

Matching the budget moves the point estimate by a factor of **2.4** while leaving
the verdict unchanged. That is more informative than either number alone: the
defect mattered for the magnitude and not for the conclusion. Under
EVALUATION.md's vocabulary the honest reading at both budgets is **"inconclusive
at 0, and rules out a worthwhile gain"**, which is not the same claim as
"recovered". Stage 7a read a point estimate with no interval.

### 2.4 "Roughly halving the calibration error" survives, and strengthens

At the matched 30-member budget, dev-MSE arm against the MSE reference:

| metric | reference | arm | Δ (improvement) | 95% CI | verdict |
|---|---:|---:|---:|---|---|
| Brier | 0.1058 | 0.0924 | **+0.0133** | [+0.0098, +0.0167] | real |
| ECE, fixed bin | 0.0587 | 0.0217 | **+0.0370** | [+0.0200, +0.0457] | real |
| log loss | 0.3438 | 0.3110 | +0.0328 | [+0.0226, +0.0421] | real |
| floor AUROC | 0.8873 | 0.8956 | +0.0083 | [+0.0045, +0.0123] | real |
| censored NLL | 1.0452 | 1.0953 | −0.0500 | [−0.0834, −0.0210] | worse |

A 63% reduction in fixed-bin ECE, with an interval. **The budget defect
invalidated the comparison, not the direction.** A correction that only removed
claims would be less useful than one that also establishes which survived.

### 2.5 Two further corrections

**The dev-NLL arm's ECE gain does not survive fixed-width binning.** Fixed-bin
ECE Δ **+0.0147 [−0.0018, +0.0260]**, inconclusive; the equal-count-decile
version is +0.0168 [+0.0014, +0.0332], real. Its Brier gain does survive
(+0.0066 [+0.0032, +0.0103]). So stage 7a's headline "ECE 0.056 → 0.042" for its
*declared* arm is binning-dependent. This is precisely the hazard that motivated
declaring fixed-width bins before fitting — equal-count edges are a function of
each arm's own predictions, so two arms are binned differently and part of any
difference is the binning — and it fired on the published arm.

**Floor discrimination does improve, for one arm.** Stage 7a's conclusion 3,
"floor discrimination does not improve at all" (0.886 vs 0.885), holds for the
dev-NLL arm — measured here as −0.0011 [−0.0071, +0.0048], inconclusive — but it
was generalised to the loss, and it is false for the dev-MSE arm (+0.0083
[+0.0045, +0.0123], conclusive). It is a generalisation from one arm to an
objective.

---

## 3. Declarations, and why each one was made

Recorded in full in [phase_b_declaration.md](phase_b_declaration.md) before
fitting. The four that did work:

**The floor event is `thalf_hours == 0`** — the recorded assay-floor code, 5,679
rows, 20.1626%, exactly EVALUATION.md's "20.2%". A probability target should be
an observable, and defining it on the recorded code means no part of the
calibration metric depends on where the detection limit is *believed* to be. The
loss's censoring rule stays at stage 7a's `thalf_hours < 0.1`, so the two
disagree on **exactly one row of 28,166** — `pair_id` 1189, HLA-A*24:02 /
WYGMEIRPL, 0.05 h, in the training split. Carried deliberately rather than
reconciled: changing the loss would make these arms non-comparable to the result
Phase B exists to resolve.

**Brier is primary because it is strictly proper.** ECE is not: a predictor
returning the base rate on every row has an ECE near zero and no discrimination
at all, so it cannot be an objective. Pinned as a test
(`test_ece_cannot_see_a_base_rate_predictor_but_brier_can`). ECE and reliability
are diagnostics; fixed-width bins, with occupancy reported next to every observed
rate, because an observed rate over nine rows is not evidence of miscalibration.

**Censored NLL is secondary, and reported because it disagreed.** The
disagreement is structural: censored NLL sums a Gaussian **density** term over
measured rows with a **probability** term over floor rows, so an arm that
sharpens its floor probabilities by shrinking `sigma` is charged for the sharper
density on every measured row. Brier scores only the floor event and is not
subject to that trade. The two are minimised at different scales on the same
predictions (`test_nll_and_brier_are_minimised_at_different_scales`). Naming the
mechanism is the useful output, not the number.

**A ranking-stopped fourth arm was declined**, in advance and with reasons
(§3.1): it needs a new stopping rule in `censored.py`, breaking the property that
the arms differ in the objective and nothing else; stage 7a found precision@10
identical at 0.700 for every arm and every threshold and floor AUROC flat at
0.885–0.886, so it has little room to act; and it would add a third selection
axis to a ranking claim Phase B is not permitted to make.

### 3.1 The worthwhile-ΔBrier bar

No analogue of EVALUATION.md's 0.05 exists for Brier, so the declaration fixed a
**rule** rather than a number: the maximum 95% paired cluster-bootstrap
half-width under no true difference, on the harness's own rows, 2,000 resamples,
seed 20261003, 8 null splits, rounded up to two significant figures.

Derived by [`scripts/phase_b_derive_bar.py`](../../scripts/phase_b_derive_bar.py),
a **separate script and a separate invocation that loads only the reference arm's
members** — so the ordering is a property of the artifact trail rather than a
claim about line order inside one program. Per-split half-widths 0.00258,
0.00263, 0.00296, 0.00271, 0.00266, 0.00239, 0.00258, 0.00265; max 0.00296.

> **Worthwhile Δ Brier = 0.0030.** Recorded in
> [phase_b_brier_bar.json](phase_b_brier_bar.json) before any censored arm was
> scored.

---

## 4. Method

| | |
|---|---|
| Assessment | five frozen outer training-cluster folds, loaded from `phase_a_fold_assignments.csv`, never recomputed |
| Verified | 98,580 rows, every `pair_id` exactly five times, `pair_id` set exactly the 19,716 training rows, outer-held counts 3944/3943/3943/3943/3943, **zero clusters spanning more than one outer fold or more than one inner fold** |
| Budget | 5 inner folds × 2 encodings × 3 seeds = **30 members per outer fold per arm**; 3 arms × 5 folds = **450 networks** |
| Parity | `selected = {"onehot": ((256,64), 1e-5), "blosum": ((256,64), 1e-3)}`, A0's exact values; the member loop mirrors `nested.fit_mlp_ensemble` and `--parity-check` asserts the MSE arm's ensemble mean matches it to 1e-9 |
| Eligibility | ≥20 rows and ≥2 distinct labels per fold (68 alleles, 99.3–99.7% coverage), with ≥50 as a declared sensitivity check (41–49 alleles, 69.4–78.5% coverage) |
| Intervals | paired peptide-cluster bootstrap, 2,000 resamples, seed 20261003, both arms on the same resample |
| Threads | BLAS pinned to 1, in the Modal image's `.env` — the only point early enough to bind |
| Floor share | 0.1910–0.2077 per outer-held fold, so no fold is unrepresentative |

Compute ran on Modal (app `pepstab-phaseb-calibration`), priced before launch
against `reports/ectodomain_rates.json` and logged append-only in
`reports/compute_ledger.csv`. The science is in `pepstab/phase_b.py` so the local
and cloud paths cannot drift.

### 4.1 Why the reference needed a scale ladder

The MSE reference has no fitted scale, so its floor probability needs one bolted
on — and **no training-only estimator of that scale is unbiased**. Measured
against A0's own output, RMSE against `y_log1p` on measured rows:

| estimator | σ | relative to the target |
|---|---:|---|
| `outer_held` — the quantity being estimated | 0.6944 | — |
| `inner_oof_6` | 0.7086 | +2.0% |
| `inner_oof_30` | 0.7176 | +3.3% |

Both inner estimators are **pessimistic**, not optimistic. Early-stopping
optimism is real but swamped by the fit-size effect: an inner out-of-fold
prediction comes from an ensemble fitted on 10,094 or 12,618 rows against the
outer-held ensemble's 15,773. They bound the out-of-sample scale **from one side
only** and do not bracket it; `inner_oof_6` is the tighter bound.

That bias is not neutral, and it runs **in favour of the censored arms**: a σ
that is too large makes the reference under-confident, which worsens its Brier
and ECE. So the reference is scored at a ladder of scales including an **oracle**
σ chosen to minimise its own Brier *on the assessment rows*.

> **The oracle σ is deliberately not a legitimate model.** It reads the held-out
> label and can predict nothing. It is reported only as an upper bound on how
> good the reference's Gaussian floor probability could possibly be.

A censored arm that beats the oracle has a result robust to the scale question
entirely; one that beats only the training-only σ has a comparator artifact.

[phase_b_reference_scale_ladder.csv](phase_b_reference_scale_ladder.csv) carries
a `legitimate` column that is `False` for the oracle σ and for the
`a0_outer_held_NOT_USABLE` row, so neither can be lifted out of the table and
read as a candidate model. Both use the held-out label.

This is also why **Platt (C1) was declared the primary baseline rather than the
Gaussian scale (C2)**: Platt fits the `mu → P(floor)` map directly on inner
out-of-fold rows and never needs a σ, so it is immune to all of the above. That
turned out to be load-bearing rather than stylistic. **If the censored arms beat
C2 but not C1, the honest conclusion is Gaussian-tail misspecification in the
comparator, not a benefit of the censored likelihood.**

---

## 5. Results — the three arms on the nested harness

Pooled over the five outer-held folds: **19,716 rows, 3,977 at the floor
(20.17%), 69 eligible alleles** at the declared ≥20 bar. Declared mixture
`P(floor)`. The two floor definitions differ on one row, as §3 says they must.

| arm | median ρ | MAE¹ | Brier | ECE (fixed bin) | log loss | floor AUROC | censored NLL | mean σ | share μ ≤ c |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 `log1p`-MSE | **0.6604** | **0.4865** | 0.1099 | 0.0643 | 0.3535 | 0.8821 | **1.0714** | 0.775 | 0.068 |
| B1 censored, dev-NLL stop | 0.6089 | 0.5931 | 0.1016 | 0.0454 | 0.3344 | 0.8813 | 1.0911 | 0.775 | 0.169 |
| B2 censored, dev-MSE stop | 0.6497 | 0.5442 | **0.0935** | **0.0196** | **0.3112** | **0.8938** | 1.1120 | 0.586 | 0.146 |

Observed floor share 0.2017. Sources: [phase_b_pooled.csv](phase_b_pooled.csv),
[phase_b_paired.csv](phase_b_paired.csv).

¹ MAE here is over the **eligible-allele** rows (19,647), as
`nested.score_fold` computes it alongside the ranking metric. §6.2 reports MAE
over **all** 19,716 pooled rows, which is why B0 appears there as 0.4870 rather
than 0.4865. Two row sets, both correct; the 69-allele panel drops 69 rows.

The censored arms' mean best epoch reproduces stage 7a's mechanism on the nested
harness: B0 **25.5**, B1 **10.7**, B2 **27.0**. B1's own dev objective turns over
early and it stops systematically undertrained.

### 5.1 Ranking

Against B0, paired peptide-cluster bootstrap, 2,000 resamples:

| arm | Δ median ρ | 95% CI | verdict |
|---|---:|---|---|
| B1 | −0.0515 | [−0.0677, −0.0290] | **worse: CI entirely below 0** |
| B2 | −0.0108 | [−0.0242, +0.0036] | inconclusive at 0, rules out a 0.05 gain |

**The per-fold signs say more than the pooled estimate, and a reader given only
the pooled number cannot tell these two situations apart:**

| arm | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | pattern |
|---|---:|---:|---:|---:|---:|---|
| B1 | −0.0504 | −0.0473 | −0.0530 | −0.0388 | −0.0570 | **5/5 negative — a consistent penalty** |
| B2 | −0.0043 | −0.0110 | **+0.0041** | **+0.0134** | −0.0181 | **3 down, 2 up — sign flips** |

B1 loses ranking on every fold by a similar amount: that is a real effect, not a
wide interval. B2's delta changes sign across folds, which is what an
inconclusive interval looks like from underneath and is better evidence for "no
ranking effect" than the point estimate alone. Source:
[phase_b_fold_consistency.csv](phase_b_fold_consistency.csv).

**Eligibility sensitivity, as declared.** At the ≥50 bar the deltas are −0.0509
(B1) and −0.0091 (B2), against −0.0515 and −0.0107 at ≥20. Unchanged.
[phase_b_eligibility_sensitivity.csv](phase_b_eligibility_sensitivity.csv).

These are **pooled** figures, where ≥50 barely bites — a pooled allele averages
286 rows, so the panel only drops from 69 alleles to 68. Applied *per fold* the
same bar keeps a median of about 48 alleles at roughly 75% coverage and widens
the fold span substantially, so at fold scope ≥50 is a **noisier instrument, not
a stricter one**. A panel disagreement at fold scope should be read as ≥50's
variance before it is read as a finding.

### 5.2 Floor-probability calibration

Against **every** comparator, Δ Brier (improvement-signed), with the
pre-derived bar of **0.0030**:

| comparator | B1 Δ Brier [95% CI] | B2 Δ Brier [95% CI] |
|---|---|---|
| B0, declared σ | +0.0084 [+0.0067, +0.0100] ✓ | +0.0165 [+0.0151, +0.0179] ✓ |
| B0 at `dev_mse` σ | +0.0074 [+0.0056, +0.0091] ✓ | +0.0155 [+0.0141, +0.0169] ✓ |
| B0 at `inner_oof_6` σ | +0.0071 [+0.0052, +0.0088] ✓ | +0.0152 [+0.0137, +0.0166] ✓ |
| **B0 at its oracle σ** | +0.0066 [+0.0046, +0.0086] ✓ | +0.0148 [+0.0132, +0.0162] ✓ |
| **C1 Platt** | +0.0019 [+0.0007, +0.0030] **✗ below the bar** | +0.0100 [+0.0091, +0.0109] ✓ |
| **C2 isotonic** | −0.0004 [−0.0018, +0.0009] **✗ inconclusive** | +0.0077 [+0.0068, +0.0086] ✓ |

**B2 clears every comparator conclusively and by five to fifty times the bar.
B1 does not clear the baselines at all** — against a two-parameter logistic map
fitted post hoc on the reference's own predictions it gains +0.0019, *below* the
bar derived before it was scored, and against isotonic it is inconclusive. The
conditional written into the declaration — "if the censored arms beat C2 but not
C1, the honest conclusion is Gaussian-tail misspecification in the comparator" —
fired, and for B1 it fired in the harsher form: it beats neither.

### 5.3 ECE is not a proper score, demonstrated on this data

The declaration's reason for making Brier primary turned up as a concrete
instance rather than a hypothetical:

| arm / baseline | ECE (fixed bin) | Brier | log loss | floor AUROC |
|---|---:|---:|---:|---:|
| **C2 isotonic** | **0.0073** | 0.1011 | 0.3309 | 0.8792 |
| B2 censored, dev-MSE | 0.0196 | **0.0935** | **0.3112** | **0.8938** |
| C1 Platt | 0.0346 | 0.1035 | 0.3391 | 0.8799 |
| B0 at oracle σ | 0.0422 | 0.1082 | 0.3567 | 0.8803 |
| B1 censored, dev-NLL | 0.0454 | 0.1016 | 0.3344 | 0.8813 |
| B0, declared σ | 0.0643 | 0.1099 | 0.3535 | 0.8821 |

**Isotonic has the best ECE of anything measured — nearly three times better
than the winning arm — while being beaten on every proper score and on
discrimination.** B2 against isotonic: ECE −0.0123 [−0.0145, −0.0062]
(conclusively worse) but Brier +0.0077, log loss +0.0197 and AUROC +0.0146 (all
conclusively better).

That is the base-rate failure mode in a real instance. Isotonic regression fits
a monotone step function to the empirical floor rate, so it is calibrated
*by construction* — and it buys that by collapsing 19,501 distinct `mu` values
into 288, destroying the resolution that discrimination depends on. **An arm
selected on ECE would have chosen the worst of the six on Brier, log loss and
AUROC.** This is why ECE is a diagnostic here and never an objective, and the
point generalises past this phase.

### 5.4 Where the reference actually fails — the reliability tables

Fixed-width bins, so all three arms are binned identically. Occupancy `n` is
shown because an observed rate over ten rows is not evidence.

**B0 cannot express high confidence at all.** Its two top bins are *empty* and
its eighth holds ten rows — it never predicts a floor probability above 0.73 —
while where it does predict 0.54 the observed rate is 0.81, and at 0.63 it is
0.88:

| bin | B0 predicted / observed (n) | B1 (n) | B2 (n) |
|---|---|---|---|
| 0.0–0.1 | 0.034 / 0.021 (7,692) | 0.034 / 0.027 (7,970) | 0.028 / 0.030 (9,964) |
| 0.4–0.5 | 0.448 / **0.590** (1,980) | 0.448 / 0.336 (1,338) | 0.449 / 0.438 (973) |
| 0.5–0.6 | 0.541 / **0.813** (1,147) | 0.549 / 0.494 (956) | 0.547 / 0.574 (753) |
| 0.6–0.7 | 0.632 / **0.883** (188) | 0.649 / 0.628 (767) | 0.647 / 0.695 (671) |
| 0.8–0.9 | — (**0**) | 0.850 / 0.876 (485) | 0.850 / 0.899 (463) |
| 0.9–1.0 | — (**0**) | 0.953 / 0.959 (506) | 0.948 / 0.963 (461) |

Full table: [phase_b_reliability.csv](phase_b_reliability.csv). B2 tracks the
diagonal across all ten bins; B1 fills them but is over-confident in the middle
(0.448 predicted against 0.336 observed). **This is the qualitative fix the
censored likelihood was built for, and it is far more visible on the training
folds than it was on validation:** a bolted-on Gaussian scale on a regressor
fitted to recorded zeros simply has no mass to put above 0.73.

### 5.5 Censored NLL still disagrees, for the declared reason

Against B0, paired, from
[phase_b_censored_nll_paired.csv](phase_b_censored_nll_paired.csv): B1 **−0.0197
[−0.0265, −0.0132]** and B2 **−0.0406 [−0.0557, −0.0273]**, both *worse* with the
interval entirely below zero. The arm that wins every
proper score for the floor event is the **worst** on censored NLL, because its
tighter scale (σ 0.586 against 0.775) is charged on the Gaussian density term at
every one of the 15,739 measured rows while sharpening the probability term at
the 3,977 floor rows. The two quantities genuinely measure different things, and
this is why the declaration made Brier primary and NLL secondary rather than
treating the disagreement as something to resolve.

### 5.6 Discrimination

Floor AUROC: B1 −0.0008 [−0.0032, +0.0015], **inconclusive**; B2 +0.0117
[+0.0101, +0.0134], **real**. Stage 7a's "floor discrimination does not improve
at all" holds for the arm it declared and is false for the other one.

---

## 6. The two-part floor model

Two components at the same budget as a censored arm each — `p_above`, the floor
indicator under squared error, and `mu_cond = E[log1p(t) | above floor]` fitted
on above-floor rows only — 300 networks over the same five outer folds.

`p_above` mean 0.7994 against an observed above-floor share of 0.7983, so the
probability is well-centred. 186 predictions clipped at 0 and 2,740 at 1.

### 6.1 Its ranking is not better than the ordinary regressor

This was the specific question the declaration said to ask explicitly, and the
answer is no:

| score | median ρ | Δ vs B0 | 95% CI | verdict |
|---|---:|---:|---|---|
| B0 `mu` (reference) | 0.6604 | — | — | — |
| `p_above × mu_cond` | 0.6541 | −0.0063 | [−0.0255, +0.0102] | inconclusive, rules out a 0.05 gain |
| `mu_cond` alone | 0.6039 | −0.0566 | [−0.0713, −0.0195] | **worse** |
| `−p_above` alone | −0.4384 | −1.0989 | [−1.1177, −1.0327] | **worse** |

The product's ranking is statistically indistinguishable from the ordinary
regressor's, with a slightly worse point estimate. As predicted, multiplying a
probability by a conditional magnitude is not a ranking improvement.

`−p_above` at **−0.4384** is worth keeping as the illustration it is: a
near-binary score with 2,740 predictions saturated at one value cannot rank
within an allele, and under the frozen average-rank tie rule a huge tied block
drives the correlation sharply negative. It is not a competitive score and was
never proposed as one.

### 6.2 It does not beat the recorded-label MAE it is the correct expectation of

| score | MAE (all) | floor rows | measured rows |
|---|---:|---:|---:|
| B0 `mu` | **0.4870** | 0.3714 | **0.5162** |
| `p_above × mu_cond` | 0.4985 | 0.3772 | 0.5291 |
| `mu_cond` alone | 0.5643 | 0.6883 | 0.5329 |

`p_above × mu_cond` **is** the correct expectation of the recorded `log1p` label
— the recorded value at a floor row is literally 0, so the floor-coded component
contributes zero — and it still loses to the plain regressor on that very
quantity. Floor-row MAE is error against the recorded 0, which EVALUATION.md
records as biased in **both** directions, so it is read alongside the other
columns and never alone.

**None of this estimates true sub-floor lifetime.** A peptide below the detection
limit has a positive half-life the assay could not resolve. Nothing in the
two-part model estimates it, and the product must not be read as doing so.

### 6.3 As a floor probability it is good, and dominated

`P(floor) = 1 − p_above`: Brier **0.0960**, fixed-bin ECE **0.0188**, AUROC
0.8855.

| against | Δ Brier | 95% CI | verdict |
|---|---:|---|---|
| B0 | +0.0139 | [+0.0120, +0.0158] | real |
| B1 | +0.0055 | [+0.0039, +0.0071] | real |
| **B2** | **−0.0026** | [−0.0040, −0.0012] | **worse** |

So the two-part model beats the reference and beats stage 7a's declared censored
arm, and **loses to the single censored arm Phase B keeps** — on the one axis
where it was competitive, while also losing on ranking and on recorded-label MAE.

**The branch does not earn a place.** It is dominated by a model that is simpler
to deploy (one network family, one objective) and needs no second component.

---

## 7. Verdict

### 7.1 Against the declared decision rule

Both limbs were fixed before fitting; the bar of 0.0030 was derived from B0 alone
by a separate invocation before any censored arm was scored.

| | limb 1 — calibration | limb 2 — ranking tradeoff | keep? |
|---|---|---|---|
| **B1** censored, dev-NLL stop | **FAIL** — +0.0019 against Platt is below the bar; inconclusive against isotonic | **FAIL** — −0.0515, CI entirely below 0 | **no** |
| **B2** censored, dev-MSE stop | **PASS** — clears every comparator including the oracle σ and both baselines | **PASS** — −0.0108 (≥ −0.02), CI lower bound −0.0242 (> −0.05) | **yes** |

> **Phase B keeps the censored loss under MSE early stopping, as a
> floor-probability model only.** It delivers a real, baseline-beating
> improvement in calibrated probability of being at the assay floor — Brier
> 0.1099 → 0.0935, fixed-bin ECE 0.0643 → 0.0196, floor AUROC 0.8821 → 0.8938 —
> at no established ranking cost, with a worthwhile ranking *loss* ruled out.
>
> **It buys no ranking, and the calibration result is not evidence for any
> ranking claim.** Δ median per-allele Spearman is −0.0108 [−0.0242, +0.0036]:
> inconclusive at zero and ruling out a 0.05 gain in either direction.
>
> **Stage 7a's declared arm (dev-NLL stopping) is rejected.** It loses ranking
> conclusively and its calibration gain does not survive a two-parameter
> post-hoc logistic map.

**Calibration improvement is not counted toward the +0.05 ranking bar**, as the
declaration requires. Nothing in Phase B supports a ranking claim of any size.

### 7.2 What a reader should take from this beyond the arms

1. **The stopping rule, not the gradient, was the whole story** — and the
   magnitude only becomes visible at a matched budget. B1 and B2 share an
   objective, differ only in early stopping, and differ by **+0.0407 median ρ**
   and **+0.0081 Brier**. Stage 7a located this correctly and could not size it.

   **This is not a quirk of the censored loss.** Phase A's A3 screen, on
   entirely fresh data, finds rank-based early stopping winning in **all eight**
   matched cells (mean +0.011) *including the two pure-MSE-loss cells where
   there is no ranking gradient at all*, and decomposes the effect as stopping
   +0.0085, loss +0.0138, interaction +0.0030. A loss-only comparison would
   have credited the objective with roughly a third of what the stopping
   surface did. Two independent arms, two different objectives, the same
   confound — so the finding is better stated as a property of **how early
   stopping on this task interacts with any objective whose dev curve turns
   over at a different epoch than MSE's**, and the remedy is to make the
   stopping rule an explicit axis rather than letting it ride along with the
   loss.
2. **A bolted-on Gaussian tail is the wrong functional form for a floor
   probability**, independent of how well its scale is chosen: both post-hoc
   fitted maps beat every Gaussian scale including the oracle (Platt 0.1035,
   isotonic 0.1011 against 0.1082–0.1094). A regressor that needs a floor
   probability should be given a fitted map, not a scaled normal.
3. **ECE cannot be an objective.** Selecting on it here would have picked the
   worst of six candidates on Brier, log loss and AUROC (§5.3).
4. **Seed and fold variation on this dataset match the effects being claimed**
   (§1), so single-seed sensitivity sweeps and unpaired fold comparisons cannot
   support conclusions. Only paired comparisons on shared folds and shared
   resamples carry information.

### 7.3 Reproducibility note — the reference is platform-dependent

Phase B's B0 does not reproduce Phase A's A0 bit-for-bit, and the cause is
measured rather than assumed
([phase_b_parity.json](phase_b_parity.json), outer fold 0, 30 members, threads=1
throughout):

| comparison | max abs diff | reading |
|---|---:|---|
| `nested.fit_mlp_ensemble` (local) vs A0 shipped | **4.4e-16** | bit-identical |
| `phase_b.fit_arm_unit` (local) vs `fit_mlp_ensemble` (local) | **1.2e-06** | the loops are equivalent |
| `fit_arm_unit` (local, macOS) vs `fit_arm_unit` (Modal, Linux) | **1.8e-01** | the entire discrepancy |

The budget **is** inherited: the implementation term is five orders of magnitude
below the platform term. The discrepancy is macOS/Accelerate arm64 against
Linux/OpenBLAS x86-64 at matched thread count — float32 accumulation order — and
amounts to **0.0020 of median-of-medians** (0.6473 against A0's 0.6493).

**It affects no comparison in this report.** B0, B1 and B2 were fitted in the
same containers, on the same platform, at the same thread count, so every delta
is internally paired and measured against this study's own B0. The figure is
recorded because any future arm fitted in a container and compared against a
locally-fitted reference inherits it.

---

## 8. What was built, and what it caught

`pepstab/calibration.py` with 38 tests. Three defects in this phase's own work
were caught by machinery rather than by inspection, which is the argument for
building it:

**A leak in the Platt baseline, invisible in the metric and visible in a row
count.** One calibration map was fitted on the pooled `inner_oof_30` rows across
all five outer folds and applied to the pooled outer-held predictions. A row is
outer-held in exactly one fold and inner-OOF in the other four, so the pooled fit
used **every row's own label to build the map that then predicted it**. The
leaked and corrected versions give the *same* Brier to four decimal places; the
giveaway was a log line reporting 19,716 fit rows where a per-fold fit can only
ever see 15,772. It is now one calibrator per outer fold with an assertion that
the fit rows and the scored rows are disjoint. **A metric check cannot see this
class of error.**

**An over-strict range check and an arity trap, both caught by a 6-network
pilot** before a 90-network run: `floor_auroc` rejected `-mu` as an input because
it demanded a probability, when AUROC is scale-free and `-mu` is exactly what
stage 7a reported; and a reproduction check compared a 6-member pilot mean
against a 30-member shipped file. The second is the arity hazard in miniature,
found by a reproduction check rather than reasoned about.

**An interval I claimed without computing it, caught by checking the report
against its own artifacts.** A draft of §5.5 stated the censored-NLL deltas as
"B1 −0.0197 and B2 −0.0406 against B0, both conclusive". The point estimates were
right — they are differences of two numbers in the pooled table — but no paired
bootstrap had been run, so "both conclusive" was an assertion about an interval
that did not exist. A script that re-derives all 37 quoted numbers from the CSVs
found it, because the figures were in no artifact. The intervals are now measured
([−0.0265, −0.0132] and [−0.0557, −0.0273], both conclusive as it happens) and
written to their own file. Being right by luck is not the same as being
supported, and a prose claim about a statistic nobody computed is the easiest
kind of error to make at the writing-up stage rather than the analysis stage.

**A misleading import error, caught by a 3-unit Modal smoke.**
`pepstab/__init__.py` imports `augment` eagerly, so an unshipped module presents
as `ImportError: cannot import name 'augment' ... (most likely due to a circular
import)`. The real cause is an incomplete module set, and the shipped list is now
derived from the import closure rather than guessed. The same smoke showed the
first payload was 2.08 MB per unit (31 MB for the batch) because it passed row
records; shipping the two read-only CSVs into the image and passing `pair_id`
lists cut it to 0.09 MB, with a SHA-256 check in the container so a stale image
layer cannot quietly fit on different data.

---

## 9. Artifacts

| file | contents |
|---|---|
| [phase_b_declaration.md](phase_b_declaration.md) | arms, objectives and decision rules, fixed before fitting |
| [phase_b_run_table.csv](phase_b_run_table.csv) | **the run table** — one row per arm with every metric, paired delta, fold-sign pattern, cost, coverage and provenance |
| [phase_b_pooled.csv](phase_b_pooled.csv) | pooled metrics, both `P(floor)` constructions |
| [phase_b_paired.csv](phase_b_paired.csv) | every paired interval, pooled and per fold, against every comparator |
| [phase_b_censored_nll_paired.csv](phase_b_censored_nll_paired.csv) | censored-NLL intervals (each arm at its own σ) |
| [phase_b_fold_consistency.csv](phase_b_fold_consistency.csv) | per-fold deltas and sign agreement |
| [phase_b_reliability.csv](phase_b_reliability.csv) | fixed-bin reliability with Wilson intervals and occupancy |
| [phase_b_reference_scale_ladder.csv](phase_b_reference_scale_ladder.csv) | the reference's σ ladder, with a `legitimate` flag |
| [phase_b_brier_bar.json](phase_b_brier_bar.json) | the bar, derived from B0 alone before any censored arm was scored |
| [phase_b_eligibility_sensitivity.csv](phase_b_eligibility_sensitivity.csv) | ≥20 and ≥50 panels, pooled and per fold |
| [phase_b_two_part_results.csv](phase_b_two_part_results.csv), [_oof.csv](phase_b_two_part_oof.csv) | two-part model scores and per-row predictions |
| [phase_b_parity.json](phase_b_parity.json) | the B0-vs-A0 investigation |
| [phase_b_stage7_retrospective_*.csv](phase_b_stage7_retrospective_metrics.csv), [phase_b_stage7_seed_spread.csv](phase_b_stage7_seed_spread.csv) | the §1–§2 correction of the stage 7a record |
| [phase_b_nested_runs.csv](phase_b_nested_runs.csv), [phase_b_two_part_runs.csv](phase_b_two_part_runs.csv) | run records for all 750 networks: best epoch, σ, `dev_mse`, fit seconds, thread count |
| `phase_b_nested_members.npz`, `phase_b_two_part_members.npz` | per-member `mu` and `σ`. **Not in the repository** — `*.npz` is gitignored project-wide, so these stay local; regenerate with `modal run modal_app/phase_b_calibration.py::run` and `::two_part` |

Code: `pepstab/calibration.py` (39 tests), `pepstab/phase_b.py`,
`modal_app/phase_b_calibration.py`, `scripts/phase_b_{derive_bar,analyse,two_part,stage7_retrospective,censored_nested}.py`.

Compute: **$0.216 realised** over four Modal runs (450 + 300 networks plus two
smokes), every run priced before launch and logged append-only in
[compute_ledger.csv](../compute_ledger.csv).

---

## 10. Bounds

One input representation (`pep_pseudo`), one architecture, one dataset, one
detection-limit point estimate treated as known, a homoscedastic Gaussian latent
with a single global scale per member. A heteroscedastic scale and a per-allele
floor remain untested. The assessment is on training-cluster folds; the frozen
validation split appears only in the §2 retrospective, and **the test split was
not read**. Conclusions do not extend to it.
