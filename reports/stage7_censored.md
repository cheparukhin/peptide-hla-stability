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

*Pending — filled in after the run.*
