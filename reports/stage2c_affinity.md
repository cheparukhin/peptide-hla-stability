# Stage 2c: auxiliary affinity training

Does training on **binding affinity** — how strongly a peptide binds — as a
second task improve prediction of **stability**, how long it stays bound?

**Answer: no, and the bound is tight.** Across 20 paired comparisons — 5 λ
settings, two encodings, two training protocols and a censoring-robustness
variant — every 95% CI crosses zero and every upper bound sits below the
predeclared 0.05 worthwhile gain. The largest is +0.034. This is a bounded
negative result, not an inconclusive one.

The mechanism is visible: the auxiliary head *does* learn affinity (ρ ≈ 0.58 on
held-out affinity labels), but measured affinity used directly as a stability
predictor ranks at ρ 0.580 — **below** the 0.610 the stability labels alone
already deliver. The signal is redundant, not absent.

All numbers are **validation**. Metrics and the 0.05 bar are predeclared in
[EVALUATION.md](../EVALUATION.md). Test is untouched.

Regenerate:

```bash
.venv/bin/python scripts/affinity_multitask.py                    # probe, ~10 min
.venv/bin/python scripts/affinity_multitask.py --protocol ensemble # ~25 min
.venv/bin/python scripts/affinity_multitask.py --drop-censored     # robustness
```

Raw runs in `stage2c_runs_*.csv`, paired CIs in `stage2c_deltas_*.csv`, the
ceiling diagnostic in `stage2c_affinity_ceiling*.csv`, the leakage audit in
`stage2c_expansion_audit.csv`. 17 guards in `tests/test_multitask.py`.

## Why this was worth testing

The stability assay's peptides were pre-selected for strong *predicted* affinity,
so the dataset's peptide diversity is narrow by construction. Public affinity
corpora cover far more peptides and alleles. Rasmussen et al. found that
combining affinity and stability data improved epitope prediction beyond either
alone (p<0.001), attributing the gain to complementary signal rather than row
count. If cheap affinity labels could substitute for expensive pretrained
features, that would be a result worth reporting at stage 3.

## Setup

The model is the stage 2 network with one extra output unit:

```
X ─→ [shared ReLU trunk 256×64] ─→ stability head  ←  y_log1p
                                └─→ affinity head   ←  y_affinity
```

Loss: `MSE_stability + λ · MSE_affinity`. λ is swept; **λ = 0 is the
single-task control.**

**The comparison is controlled by construction.** Both arms are the same class
(`pepstab/multitask.py`), differing only in λ. The affinity head is always
allocated and always drawn from its own random generator, so the main stream —
trunk init, stability head init, then every minibatch permutation — is
untouched. Consequence, asserted in `tests/test_multitask.py` on both a toy
problem and the real 860-column feature grid:

> At λ = 0 the multi-task network is **bit-identical** to
> `pepstab.mlp.MLPRegressor`.

So the single-task comparator is not a near-replica of the stage 2 baseline; it
*is* the stage 2 baseline. Verified two ways on the real data, not just asserted:

- The λ=0 single-network rows below reproduce `reports/stage2_summary.csv`
  exactly — 0.6096 (one-hot) and 0.6027 (BLOSUM), seed ranges included.
- The λ=0 **ensemble** is bit-identical to `scripts/baseline_ensemble.py`'s
  output: `preds/seq_ensemble_pep_pseudo.csv` and
  `preds/stage2c_ensemble_pep_pseudo_ensemble_lam0.csv` agree to 0.0 on all
  2,817 validation predictions.

Three further decisions that determine whether the result means anything:

- **The auxiliary loss is masked.** Only 5,135 of 19,716 training rows (26.0%)
  carry an affinity measurement. The auxiliary MSE averages over the *labelled*
  rows in each minibatch. Averaging over all rows would shrink the effective λ
  ~4× and make it drift batch to batch, so a swept value would not mean what it
  says.
- **Stopping is on stability alone.** Both arms stop on dev MSE on `y_log1p`,
  so the auxiliary task cannot win by buying itself extra epochs.
- **λ acts where the gradients meet.** Adam normalises each parameter's update
  by its own gradient RMS, so scaling the affinity head's gradient barely
  changes that head's step size. What λ changes is the *ratio* of the two
  gradients at the last shared activation — the mixing weight. The affinity
  head trains at roughly full speed at any non-zero λ, which is why it reaches
  ρ ≈ 0.58 even at λ = 0.1.

### Affinity labels

From the committed [affinity reference](../docs/AFFINITY_REFERENCE.md),
deduplicated to the strongest measurement per `(allele, peptide)`:

| | count |
|---|---:|
| dual-labelled pairs (affinity **and** half-life) | 7,281 |
| …alleles | 58 |
| in the **training** split | 5,135 rows (26.0%), 55 alleles, 3,011 peptides |
| …of which in the inner fit fold | 4,592 |
| in validation (diagnostic only, never trained on) | 732 rows |

Target transform: `1 − log10(nM) / log10(50,000)`, clipped to [0, 1] — the
NetMHC-family convention. It puts the auxiliary target on roughly the same scale
as the centred `log1p` stability target, so one λ means something comparable
across the two losses.

**C67S constructs carry no affinity label.** The three constructs match the
affinity corpora under their wild-type names, but C67S substitutes position 67 —
one of the 34 peptide-contact positions — so wild-type affinity describes a
different groove from the one whose half-life was measured. 245 such rows are
excluded, asserted in the tests, not left to a flag.

**No new leakage surface.** The probe trains only on affinity labels attached to
pairs that already carry a measured half-life. It adds no peptide and moves none
across a split boundary, so the frozen splits are untouched. The expansion to
peptides *outside* the stability set is a different matter, audited below.

## The ceiling: what an affinity label could buy at best

Before fitting anything, ask what the auxiliary label could contribute even in
the best case. Use **measured** affinity — perfect knowledge, no model error —
directly as a stability predictor, scored through the project's own per-allele
Spearman:

| Predictor | Median per-allele ρ | Panel |
|---|---:|---|
| **Measured affinity, used directly** | **0.580** [IQR 0.486, 0.696] | 33 training alleles with ≥ 30 dual pairs |
| Stage 2 single network (stability labels only) | 0.610 | 68 val alleles |
| Stage 2 30-network ensemble | 0.693 | 68 val alleles |

Pooled ρ between the affinity target and `y_log1p` is 0.494 on the 5,135
dual-labelled training pairs, consistent with the −0.491 recorded in
`docs/AFFINITY_REFERENCE.md` for nM against hours across the whole table (the
sign flips because the transform inverts the scale).

The panel is the **training split**: that is the ceiling on what the auxiliary
labels can teach this model, and it keeps the diagnostic clear of held-out
labels. On all 28,166 rows the figure is 0.581 instead of 0.580, so nothing
rests on the choice.

**The auxiliary label is a weaker ranker of our target than the model it is
being asked to improve.** Affinity carries real stability signal — 0.581 is far
from zero — but it is signal the stability labels have already supplied. On this
reading a flat result is the expected outcome, not a surprise, and the
hypothesis was still worth testing: nothing about the ceiling was knowable
without computing it.

## Result 1 — the probe (single network)

`inner_folds()`: one permanent 10% stopping fold, 17,744 fit rows, 1,972 dev
rows, the same cut stage 2 used. Config pinned per encoding to the stage 2
selection. 3 seeds per cell; ρ is the seed mean, range in brackets.

| Encoding | λ | ρ (mean) | ρ range | MAE log1p | P@10 | Affinity-head ρ | Best epoch |
|---|---:|---:|---|---:|---:|---:|---:|
| one-hot | **0** | **0.610** | [0.594, 0.625] | 0.517 | 0.70 | −0.07 | 22 |
| one-hot | 0.1 | 0.606 | [0.592, 0.618] | 0.517 | 0.70 | +0.57 | 22 |
| one-hot | 0.3 | 0.614 | [0.612, 0.615] | 0.521 | 0.72 | +0.58 | 17 |
| one-hot | 1 | 0.601 | [0.587, 0.622] | 0.527 | 0.70 | +0.57 | 18 |
| one-hot | 3 | 0.609 | [0.598, 0.628] | 0.525 | 0.70 | +0.58 | 16 |
| BLOSUM | **0** | **0.603** | [0.589, 0.615] | 0.533 | 0.60 | −0.17 | 35 |
| BLOSUM | 0.1 | 0.616 | [0.596, 0.633] | 0.540 | 0.70 | +0.55 | 31 |
| BLOSUM | 0.3 | 0.580 | [0.570, 0.596] | 0.547 | 0.65 | +0.59 | 30 |
| BLOSUM | 1 | 0.600 | [0.589, 0.610] | 0.543 | 0.63 | +0.62 | 27 |
| BLOSUM | 3 | 0.594 | [0.577, 0.604] | 0.540 | 0.70 | +0.62 | 39 |

Paired cluster bootstrap against the λ=0 control, **both arms averaged over the
same 3 seeds** so neither side gets a free ensemble:

| Encoding | λ | Δ median ρ | 95% CI | Verdict |
|---|---:|---:|---|---|
| one-hot | 0.1 | +0.003 | [−0.037, +0.033] | rules out a 0.05 gain |
| one-hot | 0.3 | −0.001 | [−0.041, +0.027] | rules out a 0.05 gain |
| one-hot | 1 | +0.005 | [−0.044, +0.031] | rules out a 0.05 gain |
| one-hot | 3 | +0.011 | [−0.044, +0.034] | rules out a 0.05 gain |
| BLOSUM | 0.1 | −0.008 | [−0.043, +0.027] | rules out a 0.05 gain |
| BLOSUM | 0.3 | −0.035 | [−0.067, +0.008] | rules out a 0.05 gain |
| BLOSUM | 1 | −0.036 | [−0.055, +0.022] | rules out a 0.05 gain |
| BLOSUM | 3 | −0.011 | [−0.048, +0.028] | rules out a 0.05 gain |

Every interval crosses 0 and every upper bound sits below 0.05. Under the
predeclared reading in EVALUATION.md that is "inconclusive at 0, **and rules out
a 0.05 gain** — not a negative result": we cannot say affinity hurts, and we can
say it does not help by the margin that would matter.

Three observations that make this a clean null rather than an ambiguous one:

1. **The auxiliary task was genuinely learned.** The affinity head reaches ρ
   0.55–0.62 against held-out affinity labels at every λ > 0, against ≈ 0 at
   λ = 0. The shared trunk learns affinity about as well as it learns stability
   (0.58 vs 0.61) — and the stability predictions do not move. This is not a
   "did the head train?" failure; it is redundancy.
2. **There is no dose-response.** The two encodings disagree about which λ is
   best (0.3 for one-hot, 0.1 for BLOSUM) and λ=0.3 is simultaneously the best
   one-hot cell and the worst BLOSUM cell. The variation is noise around the
   control, not a curve with an optimum.
3. **MAE and P@10 agree.** MAE is flat to slightly worse at every λ in both
   encodings; P@10 moves within its own granularity. No secondary metric rescues
   the primary one.

**A note on selection honesty.** λ was swept on validation, so the best λ's
validation score is optimistically biased, and the single-task arm had no
equivalent freedom. That bias is exactly why the whole sweep is tabulated rather
than its maximum: the peak one-hot cell (λ=0.3, 0.614) beats the control by
0.004, which is a sixth of the seed spread. Read the curve, not the peak.

## Result 2 — the ensemble (the decisive comparison)

Stage 2 established that **ensembling alone is worth +0.090 mean SCC from no new
information**, so a result measured only on single networks could be an artifact
of the weaker base. This is the protocol stage 3 must be compared against, and
both arms are ensembled identically — an ensembled multi-task arm against a
single-network control would manufacture a result.

`cv_folds()`: 5 folds × 2 encodings × 3 seeds = **30 networks per arm**, folds
cut along whole Hamming ≤ 3 clusters, each member stopping on its own fold so
the ensemble collectively covers all 19,716 training rows.

| λ | Ensemble ρ | Mean member ρ | MAE log1p | P@10 | Affinity-head ρ | Fit (s) |
|---:|---:|---:|---:|---:|---:|---:|
| **0** | **0.693** | 0.586 | 0.473 | 0.70 | −0.03 | 102 |
| 0.1 | 0.689 | 0.589 | 0.471 | 0.70 | +0.55 | 92 |
| 0.3 | 0.684 | 0.586 | 0.473 | 0.70 | +0.58 | 86 |
| 1 | 0.701 | 0.588 | 0.469 | 0.75 | +0.60 | 90 |
| 3 | 0.686 | 0.577 | 0.476 | 0.70 | +0.59 | 87 |

The λ=0 row is not merely equal to the stage 2 ensemble baseline (0.693) — its
predictions are **bit-identical** to `preds/seq_ensemble_pep_pseudo.csv` on all
2,817 validation rows. The control is the stage 2 model, verified rather than
assumed.

| λ | Δ median ρ | 95% CI | Verdict |
|---:|---:|---|---|
| 0.1 | −0.004 | [−0.013, +0.027] | rules out a 0.05 gain |
| 0.3 | −0.009 | [−0.016, +0.029] | rules out a 0.05 gain |
| 1 | +0.008 | [−0.017, +0.030] | rules out a 0.05 gain |
| 3 | −0.007 | [−0.030, +0.025] | rules out a 0.05 gain |

Same verdict, **tighter intervals** — ensembling cuts the CI half-width from
~0.037 to ~0.022, so the ensemble protocol rules out the 0.05 gain more firmly
than the probe did, not less.

One detail worth naming: **mean member ρ is 0.577–0.589 at every λ**, including
the control. The auxiliary task does not change individual member quality, and
the ensembling gain (≈ +0.107) is the same in every arm. So affinity is not even
acting as a diversity source, which would have been the remaining way for it to
help an ensemble.

## Result 3 — censoring robustness

6.0% of dual-labelled affinity measurements carry a `<` or `>` inequality: the
recorded value bounds the true affinity rather than measuring it. The default
keeps them, following stage 1's decision to keep the censored stability labels
rather than discard a fifth of the data. Dropping them instead:

| | Default | Censored rows dropped |
|---|---:|---:|
| Training rows with an affinity label | 5,135 (26.0%) | 4,830 (24.5%) |
| Alleles | 55 | 53 |
| Affinity ceiling (median per-allele ρ) | 0.580 | 0.561 |
| λ=0 control, one-hot / BLOSUM | 0.610 / 0.603 | 0.610 / 0.603 |
| Best λ > 0, one-hot / BLOSUM | 0.614 / 0.616 | 0.608 / 0.603 |
| Comparisons ruling out a 0.05 gain | 8 / 8 | 8 / 8 |

Dropping the censored rows lowers the ceiling and leaves the conclusion
unchanged. The result is not an artifact of treating a bound as a measurement.

## What was not run, and why

**The expansion is not run.** Stage 2c's plan gates it explicitly: *"Expand
(only if probe helps)"*. The probe does not help under either protocol, and the
bound is not merely inconclusive — it rules out the predeclared gain in all 20
comparisons. Training on 3× more affinity rows would be testing a stronger
version of a hypothesis whose weak version has a measured ceiling below the
baseline.

The leakage audit was done anyway, because it is the part a reader has to check
and it is needed by stage 2b regardless:

| Minimum Hamming to any held-out peptide | Peptides | Rows | Admitted |
|---:|---:|---:|---|
| 1 | 190 | 620 | excluded |
| 2 | 64 | 301 | excluded |
| 3 | 387 | 1,067 | excluded |
| 4 | 4,899 | 14,372 | kept |
| 5 | 13,269 | 42,807 | kept |
| ≥ 6 | 2,027 | 7,047 | kept |

66,214 affinity rows sit on 20,836 peptides absent from the stability set
entirely; after excluding everything within 3 substitutions of any validation,
test, **or inner stopping-fold** peptide (2,076 peptides in all), 64,226 rows on
20,195 peptides remain admissible. The expansion is therefore available and
leakage-clean — it is declined on evidence, not blocked by a problem.

Two points on that filter, both load-bearing:

- **Absence is tested on the peptide, never on the `(allele, peptide)` pair.**
  The reference table's `padding_eligible` flag tests the pair, which leaks: a
  peptide held out in val or test reappears as "not in the stability dataset"
  whenever it was measured on a different allele.
- **The stopping fold is in the exclusion set.** Under `cv_folds()` every
  training peptide is some ensemble member's stopping peptide, so an auxiliary
  peptide near one of them would tune that member's epoch count on a
  near-duplicate.

**The ESM-2 arm is blocked, not declined.** Stage 2c's third step — "extend to
ESM-2 once stage 3 features are ready" — cannot run because stage 3 has not
produced embeddings. It remains open, and the interesting version of the
question lives there: affinity is redundant *with the stability labels a
sequence model already exploits*, which does not establish that it is redundant
with ESM-2 features. If stage 3 reports a gap, re-running this probe on the ESM
arm is cheap (the machinery is protocol-agnostic) and would test whether cheap
labels substitute for expensive pretraining.

## Conclusion

**Auxiliary affinity training does not improve stability prediction here.**
Across 20 paired comparisons — 5 λ settings, two encodings, two training
protocols, and a censoring-robustness variant — every 95% CI crosses zero and
every upper bound sits below 0.05. The largest upper bound is **+0.034**, so the
predeclared worthwhile gain is ruled out, not merely undetected.

The result is clean rather than ambiguous because the mechanism is visible:

1. **The auxiliary task was learned.** The affinity head reaches ρ 0.55–0.62 on
   held-out affinity labels, against ≈ 0 at λ=0. The shared trunk learns affinity
   about as well as it learns stability.
2. **The label's ceiling is below the baseline.** Measured affinity used
   directly as a stability predictor ranks at median per-allele ρ 0.580 — under
   the 0.610 a single network already reaches from stability labels alone, and
   well under the ensemble's 0.693.
3. **Nothing downstream moves.** MAE, P@10, mean member quality and the
   ensembling gain are all flat across λ.

Together these say the auxiliary signal is **redundant, not absent**. Affinity
genuinely predicts stability — 0.581 is far from chance — but it predicts it
through the same within-allele groove chemistry the stability labels already
teach. That is a specific, bounded finding, and it was not knowable in advance:
the ceiling had to be computed, and Rasmussen et al.'s reported benefit from
combining the two label types made the opposite outcome plausible.

**Cost: CPU only, $0.** 30 networks for the probe (92 s of fitting), 150 for the
ensemble sweep (12.1 min), 30 for the robustness variant. The bootstrap
dominates wall time, not the fitting.

## Limitations

- **Bounded to this representation and this label set.** The finding is about a
  shared-trunk MLP on one-hot/BLOSUM peptide + pseudosequence features, with
  affinity labels on 26% of training rows. It does not rule out gains from a
  different sharing scheme (separate towers, gated fusion), a different auxiliary
  target encoding, or the broader affinity corpus.
- **The ESM-2 arm is untested** and is where the hypothesis has its strongest
  remaining form. See above.
- **λ was selected on validation**, so the per-λ peaks are optimistically biased.
  This cuts against the hypothesis's own best case, which is why the sweep is
  reported in full rather than its maximum.
- **The ceiling is computed on 33 training alleles** with ≥ 30 dual-labelled
  pairs, a narrower panel than the 68-allele validation panel. It is an estimate
  of what affinity can rank, not a score comparable row-for-row with the model.
- **Only 58 of 75 alleles have any affinity data**, and 5 have none at all, so
  the auxiliary task is unevenly distributed across the allele panel. A gain
  concentrated on well-covered alleles could in principle hide in a panel median;
  the per-allele tables in `stage2c_runs_*.csv` show no such pattern, but the
  experiment was not designed to detect it.
- **Affinity is not stability.** Neither label measures the energy barrier to
  unbinding. Both are selected assays on a peptide panel pre-filtered by
  predicted affinity, which narrows what either can teach.
