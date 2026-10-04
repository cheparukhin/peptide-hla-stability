# Stage 3b — does auxiliary affinity training help the ESM-2 arm?

Status: **harness built and tested, leakage audit re-verified, no training
run.** Sections 1–5 were written before any comparison was fitted. Section 6 is
pending and is gated on `esm-arm` reporting its headline comparison and its
selected representation.

Scope: validation split only. The test split is not read; a structural test
(`tests/test_stage3b.py::test_harness_scores_validation_only`) checks that.

Code: `scripts/stage3b_esm_multitask.py`, `tests/test_stage3b.py` (22 guards).

---

## 1. The question, and why stage 2c does not already answer it

Stage 2c asked whether auxiliary binding-affinity labels help the **sequence**
baseline. The answer was a *bounded null*: 20 paired comparisons, every 95% CI
crossing zero and every upper bound below the predeclared 0.05 bar. Its
explanation was redundancy — measured affinity used directly as a stability
predictor ranks at median per-allele ρ 0.580, **below** the 0.610 a single
sequence network already reaches from stability labels alone.

**Redundancy with a sequence model does not establish redundancy with ESM-2
features.** `HACKATHON_PLAN.md` §3b is explicit about why this is worth
running: *"If multi-task training closes the gap between the sequence baseline
and ESM-2, that's worth reporting — it would mean cheap extra labels substitute
for expensive pretrained features on this task."*

It is live now because ESM-2 35M came out roughly level with the sequence
baseline once its regularisation ladder was fixed, not behind it.

So the quantity of interest is not one delta but **two, compared**:

```
Δ_esm = ρ(ESM arm,  λ>0) − ρ(ESM arm,  λ=0)
Δ_seq = ρ(sequence, λ>0) − ρ(sequence, λ=0)
difference-in-differences = Δ_esm − Δ_seq
```

Reading that off two separately-bootstrapped intervals would lose the pairing
and overstate the uncertainty, so `difference_in_differences()` scores all four
arms on the **same** resampled peptide clusters and returns one interval.

## 2. Arms, and what is held fixed

| Arm | Features | Ensemble axis | Question it answers |
|---|---|---|---|
| `additive` | ESM-2 block **+** stage 2 raw encoding (1,190 cols) | {one-hot, BLOSUM62} | the decision-relevant one: do ESM features *improve* on sequence features, and does affinity change that |
| `seq` | stage 2 raw encoding only (860 cols) | {one-hot, BLOSUM62} | the stage 2c comparator, re-run on its **extended** ladder |
| `esm` | ESM-2 block only (330 cols) | {mid layer, final layer} | can ESM features *replace* sequence features |

ESM representation, matching what `esm-arm` selected:
`esm2_t12_35M_UR50D`, peptide `pos`, HLA `contact`, layer `mid`.
Features are built by importing `assemble()` from `scripts/esm_arm.py`, not
reimplemented, so the blocks are byte-for-byte the ones stage 3 scored.

Held fixed across every arm and every λ:

| | Value |
|---|---|
| Model class | `pepstab.multitask.MultiTaskMLPRegressor` |
| Ensemble | 5 CV folds × 2 groups × 3 seeds = **30 networks** |
| Fold assignment | `cv_folds()` from `scripts/baseline_ensemble.py` — identical to stage 2 |
| Seeds | (0, 1, 2) |
| Stopping | stability dev MSE, both arms, so the auxiliary task cannot buy itself epochs |
| Optimiser | Adam, lr 1e-3, batch 256, max 300 epochs, patience 15 |
| λ grid | (0.0, 0.1, 0.3, 1.0, 3.0) — identical to stage 2c, so the curves are comparable |
| Tuning | **none.** Every arm inherits the L2 its own stage 3 tuning selected |

**At λ = 0 the network is bit-identical to `pepstab.mlp.MLPRegressor`**, so each
arm's single-task comparator *is* that arm's stage 3 model rather than a
near-replica. This is asserted **on the real ESM feature grid**
(`test_lambda_zero_is_bit_identical_on_the_real_esm_feature_grid`), not only on
a toy — a 40-column toy and a 1,190-column block mixing PCA-reduced dense ESM
features with sparse one-hot accumulate float differently, so a toy-only check
would not be evidence.

## 3. Tuning parity is budget parity, not value parity

This is the failure mode that bit hardest tonight. `esm-arm` measured that
transplanting stage 2's L2 ladder onto the ESM arm cost **0.109** median ρ
(`reports/stage3_tuning_sensitivity.csv`) — more than twice the worthwhile-gain
bar — and nearly manufactured a false negative.

Every arm therefore gets the same *number* of grid points with *ranges
appropriate to its feature scale*, and `check_interior()` **refuses to run** an
arm whose selected L2 sits at the edge of its own ladder. A boundary hit means
the ladder was truncated, so the selected value is not the best available and
the arm is being handicapped.

Verified interior before any fit (`reports/stage3b_tuning_parity.csv`):

| Arm | Group | hidden | selected L2 | ladder | interior |
|---|---|---|---|---:|---|
| additive | one-hot | 256×64 | 0.01 | 0.001 \| 0.01 \| 0.1 | yes |
| additive | BLOSUM62 | 256×64 | 0.01 | 0.001 \| 0.01 \| 0.1 | yes |
| seq | one-hot | 256×64 | 0.01 | 1e-07 … 0.1 (5 pts) | yes |
| seq | BLOSUM62 | 256×64 | 1e-05 | 1e-07 … 0.1 (5 pts) | yes |
| esm | mid | 256×64 | 0.01 | 1e-05 … 10 (6 pts) | yes |
| esm | final | 256×64 | 0.01 | 1e-05 … 10 (6 pts) | yes |

### 3.1 A two-point ladder cannot satisfy the gate — budget parity needs ≥ 3 values

`check_interior()` requires `min(ladder) < selected < max(ladder)`. With two
values that condition is **unsatisfiable**: whichever one is selected is also an
endpoint. So a two-point ladder is not merely weak evidence of an interior
selection — it is structurally incapable of producing any.

This is not hypothetical. Two of the ladders a stage-3b arm would naturally
inherit are two-point:

| Source | Ladder | Points | Can pass the gate? |
|---|---|---:|---|
| `scripts/baseline_sequence.L2_GRID` | 1e-05, 1e-03 | 2 | **no** |
| `scripts/esm_arm.ESM_L2_GRID` (as of 07:45) | 1e-03, 1e-01 | 2 | **no** |
| this report's `additive` | 1e-03, 1e-02, 1e-01 | 3 | yes |
| this report's `seq` | 1e-07 … 1e-01 | 5 | yes |
| this report's `esm` | 1e-05 … 10 | 6 | yes |

**Update.** `esm_arm.ESM_L2_GRID` was extended to `(1e-3, 1e-2, 1e-1)` in commit
`b58c9d6`, so it is now a three-point ladder and no longer an example of the
trap. `baseline_sequence.L2_GRID` is still two-point. The rule below is what
made the extension happen, so the row is kept rather than deleted; the ladders
this report actually uses were revised again in §3.2.

Any arm that adopts "the same ladder as the comparator" therefore inherits an
un-checkable one, and the parity claim quietly becomes unverifiable at exactly
the point it is being asserted. `struct-features` hit the same wall building
the stage 5 ablation.

The generalisable rule, which is worth more than this one comparison:

> **Tuning-budget parity must be spent on at least three values per axis.**
> Equal *counts* at two points per arm look like parity and satisfy a
> grid-size audit, but no boundary check can pass, so a truncated ladder — the
> failure that cost 0.109 median ρ on the ESM arm tonight — remains invisible.

The three stage 3b ladders above were sized accordingly, and
`tests/test_stage3b.py::test_every_ladder_has_at_least_three_points` enforces
it so a later arm cannot reintroduce a two-point ladder and still appear to
pass the gate.

### 3.2 Interiority is claimed over the **union** of ladders, which exposed two errors

`check_interior()` is per-invocation: it proves the selection is interior to
the tuple it is handed. `esm_arm`'s convention for
`reports/stage3_tuning_sensitivity.csv` is stronger — interior to the **union**
of every ladder the arm was run on — because `--l2-grid` lets a later
invocation extend a ladder, so a truncation is only visible against the union.
The two claims coincide only if the tuple *is* the union. Checking that, row by
row against `reports/stage3_runs.csv`, found it was not:

| Arm / group | Ladder as declared | L2 values stage 3 actually ran | Problem |
|---|---|---|---|
| `esm` mid, final | 1e-05, 1e-03 … 10 | 1e-03, 1e-02, 1e-01, 1, 10 | 1e-05 **borrowed** |
| `additive` both | 1e-03, 1e-02, 1e-01 | 1e-03, 1e-02, 1e-01 | union correct, but 3 points with the middle selected |

The borrowed point is the sharper of the two. Every `1e-05` run on
`peppos_hlacontact_mid` in `reports/stage3_runs.csv` was at `--pep-pca 0` — the
**uncompressed control**, a different feature matrix — so the ESM arms were
claiming interiority partly against a point their own representation was never
tuned on. That is parity theatre in the opposite direction from a truncated
ladder, and it inflates the apparent budget without widening anything real.

Both are now corrected, and the λ=3 probe (§3.3) supplied the missing points:

| Arm / group | Ladder used here | Points | Selected | Interior |
|---|---|---:|---|---|
| `additive` onehot, blosum | 1e-04, 1e-03, 1e-02, 1e-01, 1 | 5 | 1e-02 | yes |
| `esm` mid, final | 1e-04, 1e-03, 1e-02, 1e-01, 1, 10 | 6 | 1e-02 | yes |
| `seq` onehot, blosum | 1e-07, 1e-06, 1e-05, 1e-02, 1e-01 | 5 | 1e-02 / 1e-05 | yes |

`tests/test_stage3b.py::test_each_ladder_contains_every_l2_stage_3_ran_on_that_representation`
re-derives the union from `reports/stage3_runs.csv` and fails if a ladder omits
a point the arm was tuned on, and
`::test_the_esm_ladders_do_not_borrow_the_uncompressed_control_point` fails if
`1e-05` is reintroduced. Neither is vacuous: the first matches 9–15 stage 3 grid
rows per group.

### 3.3 The selection was re-checked **with the auxiliary head present**

`check_interior()` runs **pre-fit**, on the L2 that stage 3 selected for a
*single-headed* network. Adding a second head changes the effective
regularisation, so a value that was interior single-task is not guaranteed to
stay interior multi-task — and the ESM arm is roughly **50x more sensitive to
the regularisation range than the baseline** (0.109 against +0.002). The
additive ladder as inherited was three points with the middle one selected, so
it had no headroom to absorb a shift. Two independent probes were run.

**Probe 1, in-harness** (`--probe-l2`, `reports/stage3b_l2_probe.csv`): the
stage 2/3 *selection* protocol — one permanent `inner_folds` fit/dev cut, 3
seeds — over the inherited three-point ladder, at λ=0 and λ=3.

| group | λ | 1e-03 | **1e-02** | 1e-01 | selected | interior |
|---|---:|---:|---:|---:|---|---|
| onehot | 0 | 0.5551 | **0.6113** | 0.5791 | 1e-02 | yes |
| onehot | 3 | 0.4772 | **0.6212** | 0.5855 | 1e-02 | yes |
| blosum | 0 | 0.5650 | **0.5959** | 0.5651 | 1e-02 | yes |
| blosum | 3 | 0.5018 | **0.6158** | 0.5767 | 1e-02 | yes |

The sensitivity is real and large — 0.477 at 1e-03 against 0.621 at 1e-02 —
which **corroborates the 0.109 transplanted-ladder cost rather than the interim
0.093**. The λ=0 column also reproduces stage 3's own grid medians for this arm
to four decimals at all six points (§3.4), so this is the same feature matrix.

Carried verbatim, because it is precise and should not be rounded off:

> **Interiority with the head present was verified only over the inner three
> values** (1e-3, 1e-2, 1e-1), since that was the ladder at the time. The curve
> is clearly unimodal and peaked at 1e-2 with both neighbours far worse, so
> extending to 1e-4 and 1.0 is very unlikely to change the selection — but
> those two points have **not** been measured with the head present, and the
> report must not imply they have.

**Probe 2 closes that gap at λ=3, and only at λ=3.** An independent probe on a
different protocol — `cv_folds` fold 0 as the stopping fold (one real ensemble
member's setup), 2 seeds — measured the two outer points with the head present:

| group | 1e-04 | 1e-03 | **1e-02** | 1e-01 | 1.0 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| additive onehot | 0.4345 | 0.4388 | **0.5875** | 0.5896 | 0.3589 | — |
| additive blosum | 0.4872 | 0.4607 | **0.5996** | 0.5817 | 0.3755 | — |
| esm mid | 0.3747 | 0.5036 | **0.5986** | 0.5935 | 0.3924 | 0.1347 |
| esm final | 0.3463 | 0.4417 | **0.5730** | 0.5767 | 0.3582 | 0.0949 |

So at λ=3 the optimum is now **bracketed**, not merely not-at-an-edge: 1e-04 and
1.0 cost 0.10–0.24 median ρ against the selection. What remains unmeasured with
the head present is **λ=0 at those two outer points**; no probe has covered
that cell.

The two protocols agree to within seed noise and disagree on nothing that
matters. The one difference worth recording: on probe 2, additive/onehot at λ=3
marginally preferred 1e-01 (0.5896) over 1e-02 (0.5875). That is **+0.0021**, an
order of magnitude inside that cell's own seed spread (0.0222 at 1e-02), and
probe 1 on 3 seeds puts 1e-02 ahead by 0.0357. L2 is held fixed across λ by
design — re-tuning per λ would break the controlled comparison exactly as stage
2c's did not — so the λ=3 additive point is at most ~0.002 below its own λ=3
optimum under one of the two protocols. That is **conservative in the direction
that matters**: it can only understate Δ_esm, never inflate it, and 0.002 is two
orders of magnitude below the 0.109 handicap that motivated the concern.

### 3.4 The λ=0 comparator is the stage 3 model, verified at two levels

| Level | Check | Result |
|---|---|---|
| network | `test_lambda_zero_is_bit_identical_on_the_real_esm_feature_grid` | bit-identical to `MLPRegressor` on the 1,190-column additive grid |
| grid point | probe 1's λ=0 column vs `reports/stage3_runs.csv` | all 6 points match to 4 dp (0.5551 / 0.6113 / 0.5791, 0.5650 / 0.5959 / 0.5651) |
| ensemble | additive λ=0, 30 networks | **0.6761**, matching the declared stage 3 additive headline to 4 dp |

The ensemble-level match is the one that was missing. Bit-identity at the
network level does not by itself prove the 30-member ensemble reassembles the
same way; reproducing 0.6761 does. The difference-in-differences therefore
differences against the stage 3 arm itself, not a near-replica of it.

Note the `seq` arm is **re-run**, not reused from stage 2c: stage 2c fitted it
on stage 2's truncated ladder, where one-hot selected 1e-05. On the extended
ladder it selects 0.01. Reusing the old numbers would make the
difference-in-differences compare an untruncated ESM arm against a truncated
sequence arm.

## 4. Leakage — re-verified from the raw reference, not quoted

Re-derived in this session from
`data/data_augmentation_iedb/affinity_reference_75alleles.csv`
(110,346 rows after dropping the C67S construct mismatches), not read from the
stage 2c write-up. Reproduced by
`test_audit_expansion_reproduces_the_stage_2c_counts`.

### 4.1 The probe (what stage 3b actually trains on)

Affinity labels attached to pairs that **already carry a measured half-life**:

| | |
|---|---:|
| dual-labelled rows, all splits | 7,281 / 28,166 |
| dual-labelled rows in train | **5,135 / 19,716 (26.0%)** |
| training alleles covered | 55 |
| new peptides introduced | **0** |

Because no peptide is added, none can move across a split boundary. This is the
only data the comparison trains on, so **the probe has no new leakage surface**.

### 4.2 The expansion surface (audited, not trained on)

Peptides absent from the stability set entirely — 66,214 rows on 20,836
peptides — against the 2,076 held-out peptides (validation + test + **the inner
stopping fold**, which belongs in the set because under `cv_folds()` every
training peptide is some ensemble member's stopping peptide).

| min Hamming to any held-out peptide | peptides | rows | excluded |
|---:|---:|---:|---|
| 1 | 190 | 620 | yes |
| 2 | 64 | 301 | yes |
| 3 | 387 | 1,067 | yes |
| 4 | 4,899 | 14,372 | no |
| 5 | 13,269 | 42,807 | no |
| 6 | 2,027 | 7,047 | no |

- **Plan §3b's stated minimum** (> 1 substitution) would keep 20,646 peptides /
  65,594 rows.
- **The stricter rule actually applied** (> 3, matching the frozen splits'
  clustering threshold) keeps **20,195 peptides / 64,226 rows**, excluding 641
  peptides / 1,988 rows.

There is **no distance-0 row**, confirming the absence test worked.

Every stage 2c count reproduces exactly: 66,214 / 20,836 before filtering,
64,226 / 20,195 after, 2,076 held-out peptides. **Audit re-verified.**

### 4.3 The `padding_eligible` trap, re-confirmed

The reference table's own `padding_eligible` flag tests the
`(allele, peptide)` **pair**, while the frozen splits group peptides across
every allele. Measured: of 46,157 flagged rows, **9,123 rows on 805 peptides
carry a peptide that is held out** in validation, test or the stopping fold.

Absence must be tested on `peptide` alone. `test_padding_eligible_would_leak_and_is_not_used`
asserts both halves: that the flag leaks, and that the function the harness
actually uses does not.

## 5. Minimum detectable effect — measured, declared before results

Following the precedent set tonight by `eval-harness` and by stage 7b. A point
estimate with an interval that spans everything is not a finding; knowing what
this comparison *can* resolve is.

`measure_mde()` blends an arm's predictions toward a **within-allele** shuffle
of themselves. Within-allele shuffling preserves each allele's marginal
distribution and the between-allele offsets, so the only thing destroyed is
within-allele ranking — exactly what the primary metric measures. For each
blend level it reports the realised Δ median per-allele Spearman and the paired
cluster bootstrap CI, on the same 2,817 validation rows, the same 68-allele
panel and the same peptide clusters the real comparisons use.

The MDE is the smallest realised |Δ| whose CI excludes 0. If that floor sits
above the predeclared **0.05** worthwhile-gain bar, this comparison cannot
resolve the effect we care about — and *that* is the finding.

---

## 6. Results — sequence arm (the ESM arms are not yet run)

Run: `scripts/stage3b_esm_multitask.py --arms seq --mde --mde-boot 500
--n-boot 2000`. 150 networks, 14.2 min, one worker, BLAS pinned.
Artifacts: `reports/stage3b_{runs,lambda_sweep,deltas,mde,tuning_parity}.csv`,
predictions in `preds/stage3b_seq_lam*.csv`.

The ESM arms are deliberately **not** run yet: their selected representation is
not final until `esm-arm` reports, and 300 networks fitted on a superseded
representation would have to be thrown away.

### 6.1 The sequence arm reproduces stage 2c's bounded null on the extended ladder

| λ | median per-allele ρ | IQR | MAE | p@10 | affinity-head ρ |
|---:|---:|---|---:|---:|---:|
| **0 (single-task)** | **0.6876** | [0.576, 0.770] | 0.4774 | 0.70 | **+0.009** |
| 0.1 | 0.6844 | [0.583, 0.762] | 0.4764 | 0.70 | +0.437 |
| 0.3 | 0.6833 | [0.567, 0.771] | 0.4738 | 0.70 | +0.482 |
| 1.0 | 0.6822 | [0.556, 0.772] | 0.4767 | 0.70 | +0.595 |
| 3.0 | 0.6702 | [0.572, 0.772] | 0.4774 | 0.70 | +0.616 |

Paired cluster bootstrap against λ = 0 (2,000 resamples, 68 alleles):

| λ | Δ median ρ | 95% CI | Verdict |
|---:|---:|---|---|
| 0.1 | −0.0031 | [−0.0215, +0.0254] | inconclusive at 0, **rules out a 0.05 gain** |
| 0.3 | −0.0043 | [−0.0239, +0.0202] | inconclusive at 0, **rules out a 0.05 gain** |
| 1.0 | −0.0054 | [−0.0255, +0.0232] | inconclusive at 0, **rules out a 0.05 gain** |
| 3.0 | −0.0174 | [−0.0281, +0.0230] | inconclusive at 0, **rules out a 0.05 gain** |

Every point estimate is slightly negative, every CI crosses zero, and **every
upper bound sits below the predeclared 0.05 bar**. The worthwhile gain is ruled
out, not merely undetected.

**This matters as more than a repeat.** Stage 2c reached the same conclusion on
stage 2's *truncated* L2 ladder, where the one-hot group selected 1e-05. On the
extended ladder the one-hot group selects 0.01, which is a genuinely different
model — and the null survives it. The ladder correction did not flip the
sequence result, so the sequence half of stage 3b rests on a properly tuned arm.

### 6.2 The auxiliary task is learned — so this is a clean null

| λ | affinity-head ρ against held-out affinity |
|---:|---:|
| 0 | **+0.009** |
| 0.1 → 3.0 | +0.437 → +0.616 |

At λ = 0 the head receives only L2 decay and reads as **never trained**
(+0.009), which is the control this diagnostic is interpreted against. At λ ≥
0.1 the shared trunk learns affinity about as well as it learns stability, and
the stability predictions still do not move.

So the finding is *"affinity is redundant with what a sequence model already
extracts"*, not *"the affinity head failed to train"*. Mean stopping epoch is
flat across λ (31.5–36.5), so the auxiliary task is not buying or losing epochs
either.

### 6.3 Measured minimum detectable effect

Blend level → realised Δ median ρ and its paired CI, on the same 2,817
validation rows, 68-allele panel and peptide clusters the comparisons use (500
resamples per level):

| blend *t* | realised Δ | 95% CI | detected |
|---:|---:|---|---|
| 0.05 | +0.0067 | [−0.0146, +0.0175] | no |
| 0.10 | −0.0075 | [−0.0185, +0.0218] | no |
| 0.15 | −0.0016 | [−0.0189, +0.0350] | no |
| 0.20 | +0.0184 | [−0.0099, +0.0527] | no |
| **0.30** | **+0.0373** | **[+0.0086, +0.0800]** | **yes** |
| 0.45 | +0.1766 | [+0.1231, +0.2149] | yes |

**MDE ≈ 0.037, bracketed in (0.018, 0.037].** Nothing was tested between the
largest undetected effect (+0.0184) and the smallest detected one (+0.0373), so
the floor is an interval, not a point.

**The design can resolve the 0.05 bar.** That is the load-bearing consequence:
a null from either arm is informative rather than merely underpowered, and the
"rules out a 0.05 gain" verdicts in §6.1 are earned rather than a formality.

At small *t* the realised Δ is non-monotone and occasionally negative — at that
blend level the median-of-per-allele-ρ statistic moves less than its own
sampling noise. That is the honest shape of the floor, not a defect in the
degradation model, which `tests/test_stage3b.py` checks is monotone in
expectation.

### 6.4 What is still open

The question stage 3b exists to answer — whether affinity helps the **ESM** arm
*differently* — is untouched by the above. The sequence result bounds only the
sequence arm, and stage 2c's redundancy explanation (affinity is redundant with
what a *sequence* model extracts) makes no prediction about ESM-2 features.

Pending, once the representation is confirmed: the `additive` and `esm` arms at
the same five λ, then the difference-in-differences Δ_esm − Δ_seq with a single
paired interval. The sequence predictions above will be **reused, not refitted**
(`--reuse-arms seq`), so the difference-in-differences is computed against
exactly the numbers reported here.
