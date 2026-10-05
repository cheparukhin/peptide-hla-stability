# Phase C §6.2: frozen joint-chain vs separate-chain representations

Date: 4 October 2026. Gated by [`phase_c_gate.md`](phase_c_gate.md); user
approved. Scope: §6.2 only. §6.3 and §6.4 are not started.

## 1. Result

**Frozen joint-chain MINT does not beat a matched frozen ESM-2 control, and all
three frozen protein-language-model arms sit far below the sequence ensemble.**
Cross-chain attention does help — measurably and consistently — but it recovers
roughly what MINT's own pretraining gave up relative to the ESM-2 it started
from, netting out to no advantage on our primary metric.

| Arm | Median per-allele Spearman | IQR | MAE(log1p) | Pooled Spearman |
|---|---:|---|---:|---:|
| A0 sequence ensemble *(same folds, Phase A)* | **0.6493** | — | — | — |
| `esm2` — ESM-2 650M, chains separate | 0.4651 | 0.308–0.536 | 0.6348 | 0.6671 |
| `joint` — **MINT, chains attend** | 0.4044 | 0.291–0.536 | 0.6415 | 0.6638 |
| `separate` — MINT, chains separate | 0.3851 | 0.261–0.472 | 0.6632 | 0.6340 |

69-allele panel fixed by the labels alone (≥20 rows, ≥2 distinct labels), 19,647
rows, 0 unranked alleles, five outer folds inherited from Phase A.

The gap to the sequence ensemble is the headline finding and it is large: the
best frozen pLM arm is **0.184 below** a BLOSUM-encoded MLP ensemble on the same
partitions. This is consistent with stage 3, which already found ESM-2 did not
beat the sequence baseline, and Phase C does not overturn it.

## 2. The three contrasts, and one metric artefact

§6.2 asks for joint against separate under one head and one pooling. Adding the
project's already-cached ESM-2 650M as a third arm splits the confound §6.2 was
worried about, because MINT *is* ESM-2 650M plus cross-chain attention plus
STRING PPI pretraining.

| Contrast | Isolates | Δ median | 95% CI | Median paired Δ | Alleles favouring | Wilcoxon |
|---|---|---:|---|---:|---:|---:|
| separate → joint | cross-chain context at inference, **weights fixed** | **+0.0193** | 0.0038 – 0.0830 | +0.0458 | 52/69 | **5.0e-06** |
| esm2 → separate | STRING PPI **pretraining**, chain treatment fixed | **−0.0800** | −0.1017 – −0.0259 | −0.0506 | 15/69 | **1.0e-05** |
| esm2 → joint | headline — **confounded by both** | −0.0607 | −0.0582 – +0.0190 | −0.0097 | 28/69 | 0.33 |

**Cross-chain context genuinely helps.** 52 of 69 alleles improve, Wilcoxon
p = 5.0e-06, and the paired cluster bootstrap CI excludes zero. Its *size* is
unresolved against the project's 0.05 bar — the CI straddles it — so this is a
real improvement of undetermined magnitude, not a worthwhile one.

**MINT's pretraining cost single-chain representation quality.** Using the same
weights without letting the chains attend is *worse* than base ESM-2 by 0.080,
CI entirely below zero, 15 of 69 alleles favouring MINT. Both arms use an
identical feature contract, so this is a property of the weights, not of pooling:
adapting ESM-2 on 96M STRING interactions appears to have degraded what a single
chain's residues encode on their own.

**The two effects roughly cancel, and the third row needs care.** Its difference
of medians is −0.0607, but the median *paired* per-allele difference is only
−0.0097, 28 of 69 alleles favour joint, and Wilcoxon gives p = 0.33. The two
functionals disagree about whether there is an effect at all.

### 2.1 A median-placement artefact, and how to spot it

The primary metric is a **median over a fixed allele panel**, and the difference
of medians is not the median of differences. When two arms have nearly identical
per-allele distributions, a large median gap can come entirely from *which allele
sits at the median* in each arm rather than from any consistent per-allele
difference.

**The tell is a bootstrap CI that excludes its own point estimate.** For
esm2 → joint the observed −0.0607 lies outside [−0.0582, +0.0190]. That cannot
happen when the statistic is describing a stable effect, and it is flagged
automatically in
[`phase_c_frozen_paired_diagnostic.csv`](phase_c_frozen_paired_diagnostic.csv)
as `median_placement_artefact`. The other two contrasts contain their point
estimates and their sign tests agree with their medians, so they are unaffected.

Consequence: **`describe_delta`'s verdict on esm2 → joint must not be read as an
effect.** Its text ("inconclusive at 0, and rules out a 0.05 gain") describes an
interval that is not centred on the quantity it is meant to summarise. The honest
statement is that joint MINT and frozen ESM-2 are **statistically
indistinguishable** on this panel, by the sign test that makes no distributional
assumption.

This is a property of the project's primary metric, not of Phase C, and it is
worth carrying: any paired comparison here should report the sign test alongside
the difference of medians, and should treat a CI excluding its point estimate as
a signal to stop trusting the median gap.

## 3. Per fold

A0's fold medians span 0.048 with the model held fixed — essentially the whole
0.05 worthwhile bar — so per-fold numbers are reported and single-fold gaps are
not read as results.

| Outer fold | Rows | Clusters | Panel | `joint` | `separate` | `esm2` |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 3,944 | 715 | 68 | 0.4314 | 0.3850 | 0.4428 |
| 1 | 3,943 | 718 | 68 | 0.4552 | 0.3514 | 0.4221 |
| 2 | 3,943 | 718 | 68 | 0.4579 | 0.3963 | 0.4682 |
| 3 | 3,943 | 718 | 68 | 0.4192 | 0.3741 | 0.4454 |
| 4 | 3,943 | 718 | 68 | 0.4503 | 0.3869 | 0.4428 |
| **span** | | | | **0.0387** | **0.0449** | **0.0461** |

`joint` beats `esm2` on fold 1 and loses on the other four. Each arm's own
fold-to-fold span (0.039–0.046) is larger than the 0.0193 joint-vs-separate
effect, which is exactly why the inference rests on the paired cluster bootstrap
and the per-allele sign test rather than on fold means.

## 4. Protocol

Inherited, not declared, per [`phase_c_gate.md`](phase_c_gate.md) §8.1.

* **Outer folds** loaded from
  [`phase_a_outer_folds.csv`](phase_a_outer_folds.csv), never recomputed. Joined
  on `pair_id`, which is safe *only* because that file and `data/splits.csv` are
  both keyed positionally into the same raw CSV; asserted, not assumed.
* **Nested**: five inner whole-cluster folds inside each outer fitting partition.
  Three of them score the L2 ladder; inner fold 0 stops the epoch. No outer-fold
  label chooses an epoch, an L2, or a feature transform.
* **Standardisation** fitted on the fitting partition only.
* **Equal tuning budget per arm**, selected per arm — not a transplanted ladder.
  L2 ∈ {1e-5, 1e-4, 1e-3, 1e-2}, with a declared 1-SE parsimony rule because a
  ladder across three same-geometry arms is expected to be flat. It was: the
  argmax and the 1-SE choice differed on 4 of 15 arm-folds (all of them in
  `esm2` or `separate`, none in `joint`).
* **Two paired seeds**, averaged identically in every arm.
* **Same head everywhere**: one hidden layer of 64, 819,329 parameters.
* **Same feature contract everywhere**: 9 per-position peptide vectors plus the
  mean-pooled partner chain, 10 × 1,280 = 12,800 dims.
* **Primary metric on uncalibrated predictions.** No calibration map is fitted
  anywhere in this script, so none can reach the ranking path.
* **Scoring** from `pepstab.evaluation` verbatim — 2,000 resamples at seed
  20261003, read through `describe_delta` except where §2.1 says not to.
* **fp32 in every arm.** The MINT features were computed in fp32 and stored in
  fp16 to halve egress; the head sees float32 in all three arms, so dtype is not
  a difference between them.
* `OMP_NUM_THREADS=4` for every arm: BLAS thread count moves a single network's
  result by ~0.02 in this project, so it is held fixed across arms.

### 4.1 What this does not establish

* **Nothing about adaptation.** All three arms are frozen. §6.3 asks whether
  limited adaptation of the joint encoder helps, and these numbers do not
  address it. A frozen null is weak evidence about a fine-tuned model.
* **Not a causal test of cross-attention.** joint vs separate changes whether
  chains attend *at inference* with weights fixed, which is informative but is
  not the same as showing how the pretraining helped — the plan is already right
  about that.
* **No comparison to the published SPEARMINT numbers is possible.** Their
  stability checkpoints trained on our corpus; see the gate's §2.1.
* **Head capacity is not tuned per arm beyond L2.** A single declared
  architecture across arms is the parity choice. Only the regularisation was
  searched; a different architecture might suit 12,800-dim embeddings better than
  it suits all three arms equally.

### 4.2 Declared sensitivity: the L2 ladder was extended, and it changed nothing

The prespecified ladder's 1-SE rule selected the boundary value 1e-2 on **12 of
15** arm-folds, and unevenly — `separate` 5/5, `esm2` 4/5, `joint` 3/5. A
parsimony rule pinned to the grid edge usually means the optimum lies outside it,
and because `separate` was both the worst arm and the most pinned, the truncation
plausibly inflated *both* significant contrasts. That is a directional bias, so
it was **tested rather than caveated**. §3.3 requires logging an extension rather
than silently expanding, which is what this section is.

Extended ladder: L2 ∈ {1e-3, 1e-2, 1e-1, 1, 10}, every arm, same budget, same
everything else. Outputs carry an `_extended` suffix; the prespecified run is
preserved as primary.

| Contrast | Prespecified Δ | Extended Δ | Prespecified Wilcoxon | Extended Wilcoxon |
|---|---:|---:|---:|---:|
| separate → joint | +0.0193 | +0.0333 | 5.0e-06 | 8.0e-06 |
| esm2 → separate | −0.0800 | −0.0764 | 1.0e-05 | 4.0e-06 |
| esm2 → joint | −0.0607 | −0.0431 | 0.33 | 0.35 |

| Arm | Prespecified | Extended |
|---|---:|---:|
| `esm2` | 0.4651 | 0.4475 |
| `joint` | 0.4044 | 0.4044 |
| `separate` | 0.3851 | 0.3711 |

**No arm improved.** One arm-fold moved off the boundary (`separate` fold 0, to
1e-1) and scored slightly worse; `joint` selected identically and is unchanged to
four decimals. So 1e-2 is the genuine optimum of a flat region, not a grid edge
cutting off a better value, and the boundary preference was the 1-SE rule working
as designed rather than a truncation artefact. **Every conclusion in §2 holds
with the same sign and comparable magnitude.** The absolute 0.37–0.47 figures are
not a floor imposed by the grid.

One incidental validation: on the extended ladder the esm2 → joint point estimate
*does* fall inside its CI, yet the sign test still shows no effect (p = 0.35,
30/69 alleles). The artefact flag has two triggers and the second one caught it,
which is the behaviour §2.1 wanted — a CI containing its point estimate is not by
itself evidence that a median gap describes a per-allele effect.

## 5. Against the plan's stop-scaling criterion

§6 says: *"Stop scaling if the matched outer assessment excludes a worthwhile
increment **and** the representation adds no useful blend signal."*

The first condition is **met**. The matched outer assessment excludes a
worthwhile increment: the best MINT arm is indistinguishable from frozen ESM-2,
both are ~0.18–0.26 below the sequence ensemble, and the only real effect found
(+0.0193 for cross-chain context) does not clear the 0.05 bar.

The second condition is now **also met**, by a minimal gate-input test
([`phase_c_blend_gate_check.py`](../../scripts/phase_c_blend_gate_check.py) →
[`phase_c_blend_gate_check.csv`](phase_c_blend_gate_check.csv)). A representation
can add useful blend signal while losing standalone, so this had to be checked
rather than assumed.

Each arm was blended with the A0 sequence reference under one nonnegative weight
on a 0.05 grid, with the weight for each outer fold chosen on the **other four
folds** only, so the scored fold's labels never touch the weight.

| Arm | Mean weight | Seq alone | Blend | Δ | 95% CI | Alleles improved | Wilcoxon |
|---|---:|---:|---:|---:|---|---:|---:|
| `joint` | 0.02 | 0.6615 | 0.6600 | −0.0016 | −0.0067 – 0.0036 | 17/69 | 4.0e-06 |
| `separate` | 0.03 | 0.6615 | 0.6617 | +0.0002 | −0.0057 – 0.0046 | 27/69 | 0.0073 |
| `esm2` | 0.07 | 0.6615 | 0.6635 | +0.0020 | −0.0087 – 0.0076 | 31/69 | 0.0924 |

**No arm adds blend signal.** The weight-selection procedure chose **zero** on 11
of 15 arm-folds and never exceeded 0.1 — given a free choice on held-out data it
preferred the sequence model essentially alone. Every CI crosses zero. Where the
sign test is significant it is significant in the *wrong* direction: `joint`'s
p = 4.0e-06 comes with 17 of 69 alleles improved and a median paired delta of
−0.0010, i.e. adding a small amount of a weaker arm makes most alleles slightly
worse, which is what a near-zero optimal weight predicts.

**Both conditions are therefore met and the plan's conjunction to stop scaling
Phase C is satisfied.**

One number to reconcile: the sequence reference reads 0.6615 here against the
0.6493 quoted for A0. These are different aggregations of the same predictions —
0.6615 is the panel median computed once over all 19,716 pooled out-of-fold rows,
0.6493 is the across-fold figure from per-fold medians. Neither is wrong and the
blend comparison is internally consistent, since every arm in the table above is
scored the same pooled way against the same reference.

### 5.1 What this blend check is not

It is a gate input, deliberately minimal, and it is **not §6.5**. §6.5 asks for a
blend *or* a separately regularised residual correction, against both the best
Phase A learner and the original ensemble, with weights fitted on inner
out-of-fold predictions per §3.6. This test uses outer-fold predictions to choose
one weight — clean with respect to the fold being scored, but weaker than §3.6's
procedure — and a single coarse grid with no tuning. It is sufficient to decide
the gate's conjunction and should not be reported as Phase C's blend finding.

## 6. Recommendation

**Stop scaling Phase C.** Both halves of the plan's own criterion are met: the
matched outer assessment excludes a worthwhile increment (§2), and no frozen arm
adds useful blend signal (§5). That is the plan's stated condition, reached on its
own terms rather than on a judgement call about whether the numbers look
promising.

Concretely:

1. **Do not run §6.3.** Its frozen form is indistinguishable from a control we
   already had cached for free, and §2's decomposition points at the backbone
   rather than at the amount of adaptation. Note that the case against it is *not*
   its price: the gate's §5.4 could not pin the epoch count — the adaptation curve
   was still improving at its 12-epoch cap — so §6.3 costs ≥$39.72 with margin and
   its adaptation ceiling is unmeasured, not the ~0.47 an earlier draft of this
   section claimed. Both of those cut slightly *in favour* of §6.3 and it is still
   not recommended, because the frozen and blend evidence does not depend on
   either. If anyone revisits it, the reason should be new evidence, not the
   SPEARMINT headline, which was measured on our own training corpus.
2. **Do not run §6.4.** Separate unbudgeted sub-project, per the gate's §2.4.
3. **Keep the out-of-fold predictions.** `phase_c_oof_*.csv` cost real money to
   produce and are the input any future blend or residual experiment would need.
4. **Delete the `pepstab-mint-weights` and `pepstab-mint-features` Volumes** and
   ledger the storage as not incurred, rather than letting a month accrue against
   a stopped direction.
5. **Carry §2.1's metric finding** into the shared evaluation notes: report the
   paired sign test alongside the difference of medians, and treat a CI excluding
   its own point estimate as a stop sign. It is not a Phase C finding.

What would change this verdict: a result showing the *backbone* is not the
constraint — a different interaction-pretrained family, or evidence that adapting
MINT recovers more than its pretraining cost. Neither is in hand.

## 7. Cost

Extraction $0.45 realised (23.8 min L4, forecast 24.3 min; 0.05997 s/pair
measured against 0.0568 forecast, 5.6% over). All head fitting was laptop CPU at
zero cost: 539 s, 990 s and 868 s for the three arms, ~40 minutes total.

## 8. Files

| Path | Contents |
|---|---|
| [`phase_c_frozen_summary.csv`](phase_c_frozen_summary.csv) | Per-arm scores on the fixed panel |
| [`phase_c_frozen_deltas.csv`](phase_c_frozen_deltas.csv) | The three paired cluster bootstraps |
| [`phase_c_frozen_paired_diagnostic.csv`](phase_c_frozen_paired_diagnostic.csv) | Sign tests and the artefact flag — read with the deltas |
| [`phase_c_frozen_coverage.csv`](phase_c_frozen_coverage.csv) | Per-fold scores and coverage |
| `phase_c_oof_{joint,separate,esm2}.csv` | Nested out-of-fold predictions, for §6.5 |
| `phase_c_frozen_*_extended.csv`, `phase_c_oof_*_extended.csv` | §4.2's declared L2-ladder sensitivity |
| `phase_c_oof_*.manifest.json` | Per-fold L2 selection, epochs, timings |
| [`phase_c_features_manifest.json`](phase_c_features_manifest.json) | Extraction provenance |
| [`../../scripts/phase_c_frozen_comparison.py`](../../scripts/phase_c_frozen_comparison.py) | `time`, `fit`, `compare` |
| [`../../modal_app/mint_features.py`](../../modal_app/mint_features.py) | `::forecast`, `::smoke`, `::extract` |
