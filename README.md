# Antigen Presentation Stability: Sequence Is All You Need?

**logic binders team · London AI × Science, Protein Engineering Track ·
3–4 October 2026**

**TL;DR:** We tested language-model embeddings and predicted structures against
a supervised model trained on sequences. None of them won.

### Start here

- **[The one-pager →](https://claude.ai/artifact/VDSMa967oxqnZNH6TnU6Cg)** — the
  whole result in one scrolling page, with the intro animation. The fastest way
  to see what we found.
- **[pMHC-I Stability Explorer →](https://claude.ai/artifact/HzCqEa1HBXMKywcoNiN2tT)**
  — browse the measured peptides for any of the 75 alleles, ordered by
  half-life, with a rotatable 3D view of the peptide in the HLA groove.
- **[reports/REPORT.md](reports/REPORT.md)** — the write-up. Every claim sourced
  to the stage report behind it. Everything below is the summary; that is the
  long form.

## The task

Your cells constantly shred their own proteins and display the fragments on the
surface, held in a groove by an HLA molecule, so passing immune cells can check
what is being made inside. How *long* a fragment stays in that groove — its
**dissociation half-life** — is a large part of what makes it visible to the
immune system, and it is what a vaccine or cancer-neoantigen designer wants to
predict. We predict the half-life of a 9-residue peptide in a given HLA groove.

**Stability is different from affinity.** One is a rate, the other an
equilibrium, and a plausible pose or a favourable interaction energy is evidence
for neither until it is tested against half-life labels.

| | |
|---|---|
| **Stability** — how long it stays bound | *t*<sub>½</sub> = ln 2 / *k*<sub>off</sub> |
| **Affinity** — how readily it binds | *K*<sub>d</sub> = *k*<sub>off</sub> / *k*<sub>on</sub> |

Pretrained protein models are the obvious thing to reach for. **We tested
whether they actually help here, against a predeclared bar, on a split that was
scored once.** They did not.

## Our approach

Ten arms, each scored through the same harness:

- Allele mean (sanity floor)
- Single sequence MLP — peptide + 34 HLA contact residues
- Sequence ensemble, 30 networks — contact-residue and full-domain encodings
- ESM-2 35M embeddings, alone
- ESM-2 150M embeddings, alone
- ESM-2 + sequence features
- Boltz-2 structure — geometry only
- Boltz-2 structure — confidence only
- Boltz-2 structure — sequence + geometry + confidence
- FoldX energy + sequence

Six of them reached the single test pass. ESM-2 150M, the geometry-only and
confidence-only structural ablations, the allele-mean floor and FoldX stopped at
validation or at pilot, and are reported as such.

## The result

![Paired test comparisons against the sequence ensemble](reports/figures/report_test_comparisons.png)

**Held-out test split**, median per-allele Spearman, higher is better. Scored
once, 4 October 2026, after every modelling decision was frozen: 5,565 rows
across 67 eligible alleles. Deltas are against the sequence ensemble, with
paired 95% intervals from 2,000 whole-peptide-cluster resamples.

| Approach | Test Spearman | Δ vs. reference [95% CI] | Verdict |
|---|---:|---|---|
| **Sequence ensemble: peptide + contact residues** | **0.7064** | — | **best in project** |
| ESM-2 35M ensemble | 0.6979 | −0.009 [−0.029, +0.023] | inconclusive; rules out +0.05 |
| Sequence ensemble: full HLA domain | 0.6904 | −0.016 [−0.039, +0.004] | inconclusive; rules out +0.05 |
| Sequence + ESM-2 ensemble | 0.6816 | −0.025 [−0.046, +0.007] | inconclusive; rules out +0.05 |
| Single sequence network | 0.6181 | −0.088 [−0.120, −0.055] | worse |
| Sequence + Boltz-2 geometry + confidence | 0.5947 | −0.112 [−0.134, −0.057] | **worse** |

The winner is 30 small networks reading the peptide plus 34 HLA contact
residues, trained on CPU in about a minute. Validation agrees with test on every
one of these verdicts — see
[§3 of the report](reports/REPORT.md#3-main-result-validation-comparisons).

**The structural arm is conclusively worse.** Its interval sits entirely below
zero, after folding all 28,166 complexes for **$212.71** — roughly **7 million
times** the cost per prediction of the winner. A sequence-only control through
the *same* pipeline has lower development MSE at five of six tested L2 settings,
including the one the structural arm selected, so the loss is the structural
features rather than the fit. This is the result we would defend hardest: we
spent the money, folded the full cohort with zero failures, and the features
still subtracted accuracy. It measures *this* structural arm, whose sequence
encoding also differs from the reference, not structure-based prediction in
general ([why that matters](reports/REPORT.md#3-main-result-validation-comparisons)).

**The ESM-2 arms are inconclusive at zero.** They neither replace nor improve
the baseline, and the intervals are tight enough to rule out a worthwhile gain:
ESM-2 35M lands at −0.009 [−0.029, +0.023], so the *direction* is unresolved
while the upper bound sits below the +0.05 we predeclared.

**We do not claim "pretraining helps."** ESM-2 only *ties* the cheap
34-residue contact-sequence baseline. Both beat a weaker full-domain encoding by
similar margins — a fact about that encoding, not evidence for pretraining.

Full cost accounting, including what each arm costs per 1,000 new predictions:
[compute_ledger.md](reports/compute_ledger.md).

## One strong positive

With **no retraining**, the sequence ensemble separates 81,600 naturally
presented peptides from 816,000 matched decoys across 51 alleles at median
**AUROC 0.9656** — a different assay and a different biological event. Two
controls make that meaningful: it holds at 0.9157 when the decoys are real
ligands from *other* alleles, and collapses to 0.6965 when the model is handed
the wrong HLA sequence. So it has learned allele-specific recognition rather
than a generic "looks like a peptide" prior, and real biology rather than
dataset artefacts.

This is a different assay, not a second measurement of half-life accuracy.
Protocol and caveats: [stage 3c](reports/stage3c_elution_validation.md).

## Methods

**The rules were fixed before any modelling.** Splits are frozen 70/10/20 with
peptides grouped by similarity, so no peptide within 3 substitutions appears in
two splits; without this, a model scores well by recognising near-copies of its
training data. The primary metric is median per-allele Spearman — ranking
peptides *within* each allele. The **+0.05** bar is not a preference: below it,
this benchmark cannot separate a real difference from noise. It was written into
[EVALUATION.md](EVALUATION.md) at stage 1, before any model existed.

Every arm uses the same rows, the same 30-network ensembling and an equal tuning
*budget*. Ensembling alone is worth **+0.074** mean Spearman from no new
information, so comparing an ensembled ESM arm against a single sequence network
would have manufactured a win. Intervals resample whole peptide clusters, not
rows, so correlated near-duplicate peptides cannot narrow one. An interval
crossing zero is reported as **inconclusive, not negative**.

**NetMHCstabpan is not used as a comparator.** The published model for this task
trained on our entire dataset, including our test peptides, so any score it
posts here is memorisation rather than prediction. It appears only as a
calibration check.

The test set was scored in one pass over six arms, under
[a runbook](reports/TEST_SCORING_RUNBOOK.md) fixed in advance, with a
[completion manifest](reports/stage6_test_manifest.json) carrying per-arm
prediction hashes. One earlier diagnostic re-partitioned the data and exposed
3,350 frozen test rows to fitting; we retracted its conclusion, refit the
benchmark on the frozen training split, and
[wrote down exactly what leaked](EVALUATION.md#disclosed-test-exposure) rather
than quietly reusing the split. The full contract is
[EVALUATION.md](EVALUATION.md); everything to discount the result by is in
[limitations.md](reports/limitations.md).

| Property | Value |
|---|---:|
| Measured peptide–HLA pairs | 28,166 |
| Unique 9-mer peptides | 5,633 |
| HLA alleles | 75 |
| Peptide length | 9 exact |
| Labels at 0 h (assay floor) | 20.2% |
| Train / validation / test rows | 19,716 / 2,817 / 5,633 |
| Worthwhile bar | Δρ ≥ 0.05 |

Measured half-lives from Rasmussen et al.; see [docs/DATASETS.md](docs/DATASETS.md).

## Next steps

| Proposed experiment | Explanation |
|---|---|
| **ESM-C with concatenated sequences** | Feed peptide–linker–MHC α1α2 into ESM-C as one sequence, so each partner is represented in the context of the other. Compare against encoding them separately. |
| **Separate embeddings + cross-attention** | Keep peptide and MHC embeddings separate, then add a trainable cross-attention module connecting individual peptide positions with groove positions. Compare against concatenated embeddings + a small MLP, keeping the embeddings unchanged. |
| **Inverse-folding and energy features** | Score each predicted complex with ProteinMPNN peptide likelihoods and FoldX interaction energies, and test them as half-life features. Compare against the sequence ensemble, keeping the same folds and tuning budget. |

## Where to look

| | |
|---|---|
| **[reports/REPORT.md](reports/REPORT.md)** | **The write-up.** Every claim sourced to the stage report behind it. |
| [reports/README.md](reports/README.md) | Stage-by-stage index: what each stage concluded and which report holds it. |
| [EVALUATION.md](EVALUATION.md) | The evaluation contract, frozen at stage 1. |
| [reports/limitations.md](reports/limitations.md) | Everything to discount the result by. |
| [reports/compute_ledger.md](reports/compute_ledger.md) | Every dollar and GPU-hour, and cost per 1,000 predictions. |
| [docs/README.md](docs/README.md) | The data: dataset stats, splits, structures, augmentation. |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Setup, reproduction commands, repo rules. |
| [HACKATHON_PLAN.md](HACKATHON_PLAN.md) | Scope and stage order. |
| [site/](site/README.md) | Local source for the one-pager, with the intro animation. |

## Status

Project complete; the Boltz-2 production fold finished 28,166 folds with 0
failures. Three items are genuinely open — provider-bill reconciliation of the
$212.71, the elution transfer pass on the ESM-2 and structural arms, and stage
3d ESM-2 likelihood features. Each is tracked in
[the report's open-work section](reports/REPORT.md#provenance-and-open-work).

To run any of it: **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)**.
