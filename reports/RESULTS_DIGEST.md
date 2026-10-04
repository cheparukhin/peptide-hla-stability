# Peptide–HLA stability: results digest

*One-page summary of everything measured, in plain language. Written 4 October
2026, ~13:15 BST. All numbers below are **validation** unless marked otherwise.
The full detail lives in the per-stage reports under `reports/`; this file just
collects the headlines and says what they mean.*

## The question and the answer

**Question.** HLA molecules sit on cell surfaces holding short protein fragments
(9-residue peptides) up for the immune system to inspect. We predict the
*residence time* — how long a peptide stays stuck once bound, measured as a
dissociation half-life. The research question is whether **pretrained
protein-AI features** (language-model embeddings, predicted 3D structure, energy
scores) predict this half-life **better than a cheap supervised model trained on
the raw sequences** — and whether any gain is worth the compute.

**Answer.** On this dataset, under a pre-registered bar, **no.** A small
supervised sequence model is the best approach we found. Every more expensive
source of information we added — language-model embeddings, Boltz-2 structures,
FoldX energies, extra affinity labels — **failed to beat it**, and most are
measurably worse. This is a *well-supported negative result*, which the
challenge brief explicitly values, and it comes with uncertainty intervals, cost
figures, and an external-validation check that the model is genuinely learning
biology rather than memorising.

## The dataset

| | |
|---|---:|
| Peptide–HLA pairs | 28,166 |
| Unique peptides | 5,633 |
| HLA alleles | 75 |
| Peptide length | 9 residues (exact) |
| Labels at exactly 0 h | 20.2% (a detection floor, not a true zero) |

Splits are frozen 70/10/20, grouped so no peptide within 3 substitutions appears
in two splits — this prevents the model from being scored on near-copies of its
training data. The headline metric is **median per-allele Spearman** (how well
the model ranks peptides *within* each allele). The pre-registered bar for a
"worthwhile" gain is **Δ 0.05**, derived from the test set's own statistical
power — below it, this benchmark cannot tell a real difference from noise.

## What each approach scored (validation, median per-allele Spearman)

| Approach | Score | Verdict vs. the 0.693 baseline |
|---|---:|---|
| Allele-mean (sanity floor) | 0.000 | — |
| Single sequence MLP (peptide + HLA contact residues) | 0.610 | reference point |
| **Sequence ensemble, 30 networks** | **0.693** | **best model in the project** |
| ESM-2 35M embeddings, alone | 0.683 | no gain (interval crosses 0, rules out 0.05) |
| ESM-2 150M embeddings, alone | 0.674 | no gain |
| ESM-2 + sequence features | 0.676 | no gain |
| Boltz-2 structure: geometry only | 0.268 | far worse |
| Boltz-2 structure: confidence only | 0.393 | far worse |
| Boltz-2 structure: seq + geometry + confidence | 0.622 | **worse: −0.071 [−0.123, −0.025]** |
| FoldX energies + sequence | — | no gain (worse in cross-validation) |

**Reading these.** The one *conclusive* comparison is that the best structural
arm is **worse** than sequence alone — the confidence interval sits entirely
below zero, and a sequence-only control through the identical pipeline wins at
every tuning setting, so the loss is the features, not the fit. The ESM-2 arms
are *inconclusive at zero*: they neither replace nor improve the baseline, and
the intervals are tight enough to rule out a worthwhile gain. We are careful
**not** to claim "pretraining helps" — ESM-2 only ties the cheap 34-residue
pseudosequence baseline; both beat a weaker *full-domain* encoding by similar
margins, which is a fact about that encoding, not about pretraining.

## The secondary experiments

- **Weak-binder augmentation (stage 2b).** Padding training with assumed-zero
  weak binders, from two sources, two weights, three seeds: **negative**, best
  arm +0.024, all intervals exclude the 0.05 bar.
- **Auxiliary affinity labels (stage 2c).** Adding a second "how strongly does it
  bind" prediction head: **negative and bounded** — the affinity signal is
  *redundant*, not absent (affinity alone ranks stability at ρ 0.580, below what
  the stability labels already give).
- **ESM-2 + affinity (stage 3b).** The sharpest form of the affinity question on
  the language-model arm: **unresolved** — the measurement's own power straddles
  the 0.05 bar, so we report it as "could not resolve", not as a null. This is
  flagged honestly as the one place a "rules out 0.05" claim is *not* backed by
  demonstrated sensitivity.
- **Censored (Tobit) loss for the 20% floor (stage 7a).** Improves calibration
  near the floor as predicted, but **loses on ranking**: Δ −0.0414 [−0.078,
  −0.006], the only conclusively-worse result in the project. The frozen
  `log1p` baseline stands.
- **ProteinMPNN inverse folding (stage 5).** The peptide "fits the backbone"
  score is **not** a half-life predictor. It turns out to be a useful
  *crystal-free detector of a bad predicted pose* — but this rests on a single
  failing complex (n = 1), stated wherever the claim travels.
- **FoldX empirical energy (stage 8).** A third extraction from the same
  structures; adds nothing over the sequence baseline. A genuine side-finding:
  these predicted structures need repair before FoldX gives sane energies
  (raw Boltz-2 side chains are strained).

## The strong positive result

**Elution as external validation (stage 3c).** Without any retraining, the
sequence model was asked to rank 81,600 naturally-presented peptides (from mass
spectrometry) above 816,000 matched decoys — a *different assay measuring a
different biological event*. It reaches **median AUROC 0.9656**, while random
scores through the same code return chance. This is the strongest evidence that
the model learned real binding/stability biology, not dataset artefacts.

**New-allele headroom (stage 7b).** Holding out whole alleles, the model ranks
well for alleles similar to training ones (ρ 0.741) and poorly for distant ones
(ρ 0.339) — real headroom exists for unseen alleles, though this is partly
confounded by distant alleles having harder data. Reported under its own
separate contract and never quoted beside the main numbers.

## What it cost

- **Sequence baseline:** CPU only, $0 cloud, under 1 ms per 1,000 predictions.
  This is the floor everything else had to justify itself against.
- **ESM-2 embeddings:** ~$0.0005 per 1,000 predictions, no measurable accuracy
  gain.
- **Boltz-2 structures:** all 28,166 complexes folded, **zero failures,
  $212.71, 7 h 32 min** across two cloud workspaces — delivered no predictive
  gain over the free baseline.
- FoldX added ~$0.33 of pilots; the full ~$190 energy pass was bounded by two
  prior negatives before spending.

## Status of the final test scoring

Every number above is **validation**. The held-out **test set is scored once,
last** — a rule fixed in writing before any test number existed. That single
pass is **running now** (test predictions for all five arms were written
12:56–13:14 BST; the orchestrator is computing metrics with a reproduction guard
and has not yet published them). Any arm not frozen by the cutoff is reported on
validation only, and labelled as such — a missing test number is a scheduling
fact, not a result.

## Is this enough for a good submission?

**Yes — this is a strong, submittable result**, with two honest caveats.

**Why it's strong.** It answers the exact question posed, cleanly and in the
negative, which the brief says it values. It covers all three model classes the
brief names — embeddings, structure/confidence, and inverse folding — plus
energy functions. Every claim carries an uncertainty interval and a cost. The
methodology is unusually disciplined: a pre-registered improvement bar, grouped
splits that block leakage, a documented and quarantined prior test-exposure
incident, and power analyses that let us separate a true "null" from an
"unresolved". The elution AUROC of 0.966 gives it a genuine positive headline so
the story is not purely "nothing worked".

**Caveat 1 — it's a negative result, so the framing is everything.** The
submission must lead with *"a cheap sequence model is the right tool here, and
here is the rigorous evidence that expensive pretrained features don't beat it,
and what they cost"* — not with an apology. The value is the rigor and the cost
accounting, not a leaderboard number.

**Caveat 2 — the test pass must land (or be clearly marked pending).** All
headline numbers are validation. If the single test pass completes, the
conclusions almost certainly hold (validation and test are well-aligned by
design). If it doesn't finish, the submission is still valid *as a
validation-only study* provided it says so plainly — which the project's own
rules already require.

**Bottom line:** the science is done and defensible. The remaining work
(finishing the one test pass, and compressing the 118 KB `SUBMISSION.md` into a
presentation) is packaging, not research.
