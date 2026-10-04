# Stage 7a — censored (Tobit) likelihood for the assay floor

Status: **protocol predeclared, results pending.** Sections 1–4 were written and
committed to disk before any censored model was fitted. Nothing in them is
chosen from a result.

Scope: validation split only. The test split is not read, scored, or reported
anywhere in this document.

---

## 1. The problem

20.2% of `thalf_hours` labels are exactly 0 (5,679 / 28,166). Stage 1 concluded
on three independent lines of evidence that these are **left-censored at the
assay's detection floor**, not measurements of zero
(`reports/audit_summary.md` §3):

1. a trough at 0.1 h — a fit over 0.2–1.0 h predicts ~1,651 rows at 0.1 h, only
   443 are observed;
2. per-allele zero share anticorrelates with that allele's median non-zero
   label (Spearman −0.608, p = 3.7e-8);
3. zeros are spread across peptides, not concentrated in a few.

The project's 16-hour substitute is `log1p` plus rank-led metrics.
`EVALUATION.md` already records why MAE at the floor is biased in **both**
directions: a latent 0.05 h recorded as 0 charges a 0.10 h prediction 2× too
much and a 0.01 h prediction 4× too little.

A censored likelihood states the correct thing instead. For a floor row the
likelihood contribution is not "the value is 0" but "the latent value is at or
below the detection limit".

## 2. The loss

Per-row negative log-likelihood of a Gaussian with a learned homoscedastic
scale `sigma`, left-censored at a threshold `c` on the `log1p` scale. `mu` is
the network output.

| Row | Contribution |
|---|---|
| uncensored (`y > c` in hours) | `0.5 * ((y - mu)/sigma)^2 + log sigma + 0.5*log(2*pi)` |
| censored (recorded at the floor) | `-log Phi((c - mu)/sigma)` |

`sigma` is a single scalar trained jointly with the network by the same Adam
optimiser at the same learning rate, parameterised as `log sigma`.

Implementation: `pepstab/censored.py`, `CensoredMLPRegressor`, a subclass of the
existing `pepstab.mlp.MLPRegressor`. It inherits initialisation, forward pass,
backward pass, Adam, L2, target centring and best-epoch restore unchanged, and
overrides only the objective. That is what makes "the loss is the only
difference" literally true rather than approximately true.

## 3. The censoring threshold — chosen before training

**This is the one free parameter the method introduces, and tuning it on
validation results would make the whole exercise circular.** It is therefore
fixed here, from the assay description and the label distribution alone.

### Declared value

> **`c = log1p(0.1) = 0.095310`**, i.e. a detection limit of **0.1 hours**.
> A row is treated as left-censored iff its recorded `thalf_hours` is **strictly
> below 0.1 h**.

### Justification (no validation result is involved)

| Evidence | Value |
|---|---|
| Recording grid of the published labels | 0.1 h (98.74% of labels sit on it) |
| Smallest positive half-life anywhere in the dataset | 0.05 h — **1 row** |
| Second smallest | 0.1 h — 443 rows |
| Rows strictly between 0 and 0.1 h | **1** |
| Stage 1's located floor (`audit_summary.md` §3) | "below ~0.1–0.2 h" |

0.1 h is the smallest half-life the assay actually reports at any volume, so a
recorded 0 means "shorter than the shortest value this assay resolves", which
is exactly `latent <= log1p(0.1)`. Setting `c` any higher would declare rows
that *were* reported as measurements to be unobservable; setting it lower would
place the floor below the reporting grid's resolution.

The censored set under this rule is 5,680 rows (the 5,679 zeros plus the single
0.05 h row). The threshold is never re-chosen after seeing a score.

### Predeclared sensitivity grid

Because stage 1 locates the floor in a *range* (0.1–0.2 h) rather than at a
point, the headline is reported alongside a sweep. One rule throughout: the
detection limit is `c_hours`, and every recorded value strictly below it is a
floor code.

| `c_hours` | `c` on log1p scale | Censored rows (all splits) | Reading |
|---:|---:|---:|---|
| 0.05 | 0.048790 | 5,679 | floor below the reporting grid |
| **0.10** | **0.095310** | **5,680** | **declared primary** |
| 0.15 | 0.139762 | 6,125 | the depleted 0.1 h bin is also floor |
| 0.20 | 0.182322 | 6,139 | stage 1's upper bound |
| 0.30 | 0.262364 | 7,448 | deliberately over-censored control |

0.30 is included as an over-censoring control, not as a candidate: it discards
1,297 genuine measurements at 0.2 h. If the conclusion only holds there, it is
an artifact.

## 4. What is held fixed between the two arms

Both arms are the stage 2 `pep_pseudo` input set (peptide + 34-residue HLA
contact pseudosequence), fitted under the stage 2b ensemble protocol.

| Held fixed | Value |
|---|---|
| Features | `build_features(df, "pep_pseudo", enc)`, identical matrices |
| Architecture | `(256, 64)` ReLU, per-encoding L2 from `scripts/baseline_ensemble.py` `SELECTED` |
| Optimiser | Adam, lr 1e-3, batch 256, `max_epochs` 300, `patience` 15 |
| Ensemble | 5 inner CV folds × 2 encodings × 3 seeds = **30 networks**, mean-averaged |
| Inner folds | cut along whole Hamming ≤ 3 peptide clusters, same assignment for both arms |
| Seeds | `(0, 1, 2)` |
| Tuning budget | **zero** — configs are inherited from stage 2, nothing is re-tuned in either arm |
| Target centring | mean of recorded `y_log1p` on the fit rows, both arms |
| Fit rows | `split == "train"` only |
| Scoring | `pepstab.evaluation.score` on `split == "val"`, 68 eligible alleles |

**The only difference is the objective**, including the early-stopping
criterion: each arm stops on its own training objective evaluated on the held-out
inner fold (MSE for the MSE arm, censored NLL for the censored arm). That is the
consistent analogue, not a second change — but a dev-MSE-stopped censored arm is
reported in the sensitivity table so the choice is visible.

### Declared prediction heads

The censored model's `mu` estimates the **latent** half-life, which is not the
same quantity as the recorded label. Three uses, fixed in advance:

| Metric | Prediction used | Why |
|---|---|---|
| Spearman, precision@10 | raw `mu` (both arms) | strictly monotone, no artificial ties |
| MAE (headline) | raw output (both arms) | like-for-like against the recorded label |
| MAE (floor-aware) | `max(pred, 0)` (both arms) | available to both arms, so it is not a censored-arm privilege |

## 5. Predeclared expectation

A censored loss should mainly improve **calibration near the floor**, not
ranking. Ranking *within* the tied floor block is unidentifiable under either
objective — the labels there carry no order — so the primary metric has little
room to move. The predeclared minimum worthwhile gain of **Δ median per-allele
Spearman = 0.05** applies, and an interval that crosses zero is **inconclusive**,
not negative. Only an interval whose upper end is below +0.05 rules a worthwhile
ranking gain out.

---

## 6. Results

Run: `scripts/stage7_censored.py`, 130 networks, 29.6 min CPU on a heavily
contended 8-core machine. Artifacts: `reports/stage7_censored_runs.csv`,
`_sensitivity.csv`, `_calibration.csv`, `_reliability.csv`, `_headline.json`;
predictions in `preds/stage7_{mse,censored,censored_devmse}.csv`.

Validation split, 2,802 scored rows, 68 eligible alleles, 553 floor rows
(19.6%). **The test split was not read.**

### 6.1 Headline — the censored arm loses on ranking

Both arms at the full 30-network budget, identical in everything but the loss.

| | median per-allele rho | IQR | MAE (log1p) | median p@10 | pooled Pearson |
|---|---:|---:|---:|---:|---:|
| `log1p`-MSE | **0.6931** | [0.574, 0.753] | **0.4734** | 0.700 | 0.810 |
| censored @ 0.1 h | 0.6518 | [0.516, 0.728] | 0.5763 | 0.700 | 0.745 |

**Δ median per-allele Spearman = −0.0414, paired cluster bootstrap 95% CI
[−0.0780, −0.0062]** (2,000 resamples, seed 20261003, 68-allele panel).

Under the predeclared reading in `EVALUATION.md` the verdict is
**"worse: CI entirely below 0"**. This is a *negative* result on ranking, not an
inconclusive one — the interval excludes zero, so it is not merely that we
failed to resolve a difference.

It is also not a case of optimising the right thing and measuring the wrong
one: the censored arm is worse **on its own objective**, on held-out data —
mean validation censored NLL **1.0765** against **1.0452** for the MSE arm
scored with a post-hoc scale. §6.4 explains why.

Precision@10 is identical at 0.700 for every arm and every threshold, so it
does not separate anything here.

### 6.2 Calibration at the floor — a real but modest gain

This is where the predeclared expectation said to look.

| | MSE | censored | observed |
|---|---:|---:|---:|
| share of predictions at or below the limit | 0.070 | **0.168** | 0.196 |
| mean predicted `P(floor)` | 0.183 | 0.231 | 0.196 |
| Brier score for `P(floor)` | 0.1050 | **0.0984** | — |
| expected calibration error (10 equal-count bins) | 0.0561 | **0.0422** | — |
| **floor AUROC** | **0.8862** | 0.8853 | — |
| mean prediction on floor rows | +0.326 | −0.086 | — |
| MAE, floor rows only | 0.355 | 0.615 | — |
| MAE, measured rows only | 0.502 | 0.566 | — |
| MAE, all rows, clipped at 0 | 0.470 | 0.494 | — |

Read carefully:

- **The censored arm puts the right amount of mass below the detection limit.**
  The MSE arm places 7.0% of validation predictions at or below the floor
  against an observed 19.6%; the censored arm places 16.8%. That is the
  qualitative fix working as designed, and it is the one claim here that is not
  contestable.
- **Floor probabilities are better calibrated**, by 12% on Brier and 25% on
  ECE. The reliability table (`_reliability.csv`) shows where: in the top
  decile the MSE arm predicts 0.539 against an observed 0.772 — badly
  underconfident — while the censored arm predicts 0.816 against 0.790. In the
  middle deciles the censored arm is *over*confident (e.g. 0.180 predicted
  against 0.067 observed), so the gain is not uniform.
- **Floor discrimination does not improve at all.** AUROC 0.8862 vs 0.8853 —
  the MSE arm is nominally ahead. The two arms order floor rows against
  measured rows equally well; the censored arm only places them better on the
  scale. This is exactly what §5 predicted, and it is the reason the ranking
  metric had little room to move.
- **MAE gets worse, and only part of that is the known bias.** Floor-row MAE
  (0.355 → 0.615) is error against the recorded 0, which `EVALUATION.md`
  records as biased in both directions; the censored arm's floor predictions
  are deliberately *below* 0, so this number is close to meaningless on its own.
  But measured-row MAE also worsens (0.502 → 0.566), and that is not explained
  by the floor bias. Even the floor-aware MAE, clipped at 0 and available to
  both arms, is worse (0.470 → 0.494). The pooled Pearson drop (0.810 → 0.745)
  is partly scale: `mu` estimates a latent value that the recorded labels
  censor.

### 6.3 Threshold sensitivity — the free parameter is not driving anything

All points at a reduced, internally matched budget (10 networks each), so they
are comparable to one another and not to §6.1.

| detection limit | censored train rows | median rho | MAE | floor AUROC | mean sigma |
|---|---:|---:|---:|---:|---:|
| MSE reference (no censoring) | 0 | 0.6975 | 0.4764 | 0.8825 | — |
| 0.05 h | 3,977 | 0.6743 | 0.5893 | 0.8865 | 0.753 |
| **0.10 h (declared)** | 3,978 | **0.6657** | 0.5799 | 0.8848 | 0.750 |
| 0.15 h | 4,271 | 0.6700 | 0.5855 | 0.8835 | 0.746 |
| 0.20 h | 4,276 | 0.6726 | 0.5851 | 0.8820 | 0.755 |
| 0.30 h (over-censoring control) | 5,196 | 0.6648 | 0.5840 | 0.8771 | 0.766 |

Across a 6-fold range of the detection limit the median rho spans **0.0095**
and floor AUROC spans 0.0094 — both far below the 0.05 bar and below the
seed-to-seed spread. The declared 0.1 h is unremarkable within that range: it
is neither the best nor the worst point, which is what a threshold fixed before
training should look like. **The conclusion in §6.1 does not depend on the
threshold choice**, including at the deliberately over-censored 0.3 h control.

### 6.4 Where the ranking loss actually comes from — the stopping rule, not the gradient

Mean epoch kept, over each arm's networks:

| arm | mean best epoch | mean epochs run | mean sigma |
|---|---:|---:|---:|
| MSE | 27.1 | 42.1 | — |
| censored, stopped on dev NLL (declared) | **10.3** | 25.3 | 0.750 |
| censored, stopped on dev MSE (control) | 31.0 | 46.0 | 0.579 |

**The censored arm's own objective bottoms out on the dev fold at about epoch
10 and then rises, so the declared arm is systematically undertrained relative
to the MSE arm.** Swapping only the stopping rule — same censored loss, same
gradients, same everything else — changes the picture:

| arm (reduced budget) | median rho | Brier | ECE | val censored NLL |
|---|---:|---:|---:|---:|
| MSE reference | 0.6975 | 0.1050 | 0.0561 | 1.0452 |
| censored, dev-NLL stop | 0.6657 | 0.0984 | 0.0422 | 1.0765 |
| censored, dev-MSE stop | **0.6910** | **0.0930** | **0.0237** | 1.1134 |

The dev-MSE-stopped censored arm gives back almost all of the ranking (−0.0065
against its reference, versus −0.0318 for the declared arm) while roughly
halving the calibration error. Its validation censored NLL is nonetheless the
worst of the three, because its much tighter scale (sigma 0.58) is penalised on
the Gaussian density term even as it sharpens the floor probabilities — the two
quantities genuinely disagree.

**This is a control, not a result.** It was predeclared as a control in §4, it
was run at the reduced budget, and it has **no confidence interval**. It
localises the mechanism and points at a fix; it does not establish one. Calling
it an improvement would be exactly the post-hoc selection this stage exists to
avoid.

### 6.5 Conclusion, bounded to what was tested

1. **A left-censored Gaussian likelihood, as predeclared, does not improve this
   model and measurably harms its ranking.** Δ median per-allele Spearman
   −0.0414, 95% CI [−0.0780, −0.0062] — worse, conclusively. The frozen
   `log1p`-MSE baseline stands; nothing downstream should switch loss on this
   evidence.
2. **It does deliver the floor behaviour it was built for**, modestly: the
   right amount of predictive mass below the detection limit (16.8% vs 7.0%
   against an observed 19.6%) and better-calibrated floor probabilities (ECE
   0.056 → 0.042). If a downstream use needs a probability that a pair is
   below the assay floor, the censored arm is the only one of the two that
   produces it from its own fit rather than from a scale bolted on afterwards.
3. **It does not improve ranking at the floor, as predicted.** Floor AUROC is
   unchanged (0.886 vs 0.885). Ranking *within* the tied floor block is
   unidentifiable under either objective, so there was never much room here.
4. **The threshold is not load-bearing.** The result is flat across 0.05–0.3 h.
5. **The likely fix is the stopping rule, not the loss.** The censored arm is
   undertrained because its own dev objective turns over early. A control that
   changes only the stopping rule recovers the ranking and improves calibration
   further — untested at full budget and without an interval.

**Bounds.** One input representation (`pep_pseudo`), one architecture, one
dataset, validation only, a homoscedastic Gaussian latent with a single global
scale, and a point estimate of the detection limit treated as known. A
heteroscedastic scale, a per-allele floor, or a loss that keeps the MSE
stopping rule are all untested. Conclusions do not extend to the test split,
which was not read.

