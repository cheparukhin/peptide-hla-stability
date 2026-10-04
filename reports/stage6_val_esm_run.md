# Stage 6 validation run: the ESM-2 arms

What the stage 6 machinery adds on top of the stage 3 headline. The headline
comparison itself is `reports/stage3_esm.md` and
`reports/stage3_headline.json`; this document covers the four analyses that
headline does not contain, and it is the artefact the submission should cite
for them.

**Validation only. The test split was not touched.** Method, definitions and
the verification of every count are in
[`stage6_evaluation_machinery.md`](stage6_evaluation_machinery.md).

Command (one invocation, every arm, `seq_ensemble_pep_pseudo` first so every
paired interval is taken against it):

```
.venv/bin/python scripts/stage6_report.py --split val \
    --n-boot 2000 --stratum-min-rows 10 --prefix stage6_val_esm \
    seq_ensemble_pep_pseudo=preds/seq_ensemble_pep_pseudo.csv \
    esm_ensemble=preds/esm_ensemble.csv \
    esm_plus_seq_ensemble=preds/esm_plus_seq_ensemble.csv \
    esm_ensemble_150m=preds/esm_ensemble_150m.csv \
    seq_ensemble_pep_domain=preds/seq_ensemble_pep_domain.csv
```

`--stratum-min-rows 10` is the validation-only bar, chosen from row counts and
panel coverage before any model was scored in a stratum; test keeps the frozen
20. The baseline's SHA-256 matches the digest recorded in the earlier run's
manifest, so every delta below is against the same file stage 3 used.

## Headline, reproduced independently

| Arm | Median per-allele rho | IQR | MAE log1p | Pooled rho |
|---|---:|---|---:|---:|
| `seq_ensemble_pep_pseudo` | **0.6931** | 0.574–0.753 | 0.4734 | 0.8015 |
| `esm_ensemble` (35M, selected) | 0.6830 | 0.559–0.768 | 0.4793 | 0.8003 |
| `esm_plus_seq_ensemble` | 0.6761 | 0.560–0.770 | 0.4875 | 0.7934 |
| `esm_ensemble_150m` | 0.6737 | 0.566–0.750 | 0.5058 | 0.7856 |
| `seq_ensemble_pep_domain` | 0.6529 | 0.503–0.750 | 0.4948 | 0.7856 |

Matches `reports/stage3_headline.json` exactly, from an independent code path.

### Paired intervals against the sequence baseline

2,000 cluster resamples for the panel statistics, 500 for the differential and
precision medians, seed 20261003, whole peptide clusters.

| Model vs `seq_ensemble_pep_pseudo` | Median rho | 95% CI | Mean rho | 95% CI |
|---|---:|---|---:|---|
| `esm_ensemble` | −0.0101 | [−0.0382, +0.0353] | −0.0036 | [−0.0248, +0.0165] |
| `esm_plus_seq_ensemble` | −0.0170 | [−0.0468, +0.0320] | −0.0112 | [−0.0332, +0.0111] |
| `esm_ensemble_150m` | −0.0194 | [−0.0599, +0.0195] | **−0.0199** | **[−0.0392, −0.0017]** |
| `seq_ensemble_pep_domain` | −0.0402 | [−0.0727, +0.0009] | **−0.0348** | **[−0.0502, −0.0197]** |

Every median interval reads *inconclusive at 0, and rules out a 0.05 gain* —
no arm beats the sequence baseline and none is shown to be worse on the primary
metric. 103 of the 2,000 resamples lost at least one of the 68 alleles to
missing label spread; those alleles leave that resample's panel for both arms
at once.

The **mean** over the same panel is sharper than the median here and
conclusively places two arms below the baseline. It is reported because it is
the statistic stage 2 quotes, and because two panel statistics disagreeing
about conclusiveness is itself worth seeing: the median is the contract's
metric and the predeclared verdict rule applies to it alone.

## 1. The differential target

394 validation peptides sit on two or more eligible alleles, giving **10,365**
allele-pair comparisons, of which 755 are undecidable (equal labels) and 268
allele pairs clear the 10-peptide bar.

| Arm | Concordance | Peptide-weighted | Pooled rho(delta) | Median allele-pair rho | MAE(delta) |
|---|---:|---:|---:|---:|---:|
| `seq_ensemble_pep_pseudo` | 0.8262 | 0.8318 | 0.7864 | 0.6854 | 0.5771 |
| `esm_ensemble` | **0.8289** | 0.8290 | **0.7906** | 0.6802 | 0.5775 |
| `esm_plus_seq_ensemble` | 0.8249 | 0.8303 | 0.7847 | 0.6743 | 0.5868 |
| `esm_ensemble_150m` | 0.8174 | 0.8216 | 0.7688 | 0.6548 | 0.6044 |
| `seq_ensemble_pep_domain` | 0.8137 | 0.8226 | 0.7657 | 0.6454 | 0.6002 |

Paired, 500 cluster resamples, against `seq_ensemble_pep_pseudo`:

| Model | Delta concordance | 95% CI | Verdict |
|---|---:|---|---|
| `esm_ensemble` | +0.0028 | [−0.0052, +0.0102] | inconclusive at 0 |
| `esm_plus_seq_ensemble` | −0.0013 | [−0.0091, +0.0058] | inconclusive at 0 |
| `esm_ensemble_150m` | −0.0088 | [−0.0149, −0.0025] | **conclusive, worse** |
| `seq_ensemble_pep_domain` | −0.0124 | [−0.0186, −0.0061] | **conclusive, worse** |

**This is the sharpest result in the run, and it is a measured equivalence.**

On cross-allele ranking — where the peptide's own contribution cancels exactly,
so what remains is groove chemistry — ESM-2 and the sequence baseline are
equivalent **to within one point of concordance**: +0.003 [−0.005, +0.010] for
ESM-2 alone and −0.001 [−0.009, +0.006] for sequence plus ESM-2. The intervals
have half-widths of 0.0077 and 0.0074, four to six times tighter than the
primary metric's on the same rows.

A tight interval around zero invites the objection that the metric simply
cannot tell arms apart. **The same measurement, on the same 10,365 comparisons
with the same bootstrap, conclusively separates two other arms** — the 150M
checkpoint at −0.009 and the full-domain ensemble at −0.012, both with
intervals excluding zero, both of which the primary metric calls inconclusive.
The differential resolves differences of about 0.009; the ESM arms sit inside
that. The equivalence is a measured null, not a blind metric.

This is also the clearest demonstration of why the differential target was
worth building: it is the only analysis in the stage 6 set that returns a
conclusive verdict on any comparison in this run.

## 2. Distance stratification

| Arm | d=4 | d>=5 | Drop |
|---|---:|---:|---:|
| `seq_ensemble_pep_pseudo` | 0.7360 | 0.6539 | 0.0821 |
| `esm_ensemble` | 0.7114 | 0.6490 | 0.0624 |
| `esm_plus_seq_ensemble` | 0.7021 | 0.6154 | 0.0867 |
| `esm_ensemble_150m` | 0.6894 | **0.6758** | **0.0136** |
| `seq_ensemble_pep_domain` | 0.6963 | 0.6233 | 0.0730 |

55-allele shared panel at the validation 10-row bar; 1,377 rows at d=4 and 992
at d>=5.

The point estimates look like a size effect: the 150M arm barely degrades and
has the highest d>=5 score of any arm. **The confidence intervals do not
support it.** Paired cluster bootstrap on the stratum gap itself (d=4 minus
d>=5, differenced against the baseline; negative means the arm loses *less*):

| Model vs `seq_ensemble_pep_pseudo` | Gap delta | 95% CI | Verdict |
|---|---:|---|---|
| `esm_ensemble` | −0.0197 | [−0.0973, +0.0800] | inconclusive |
| `esm_plus_seq_ensemble` | +0.0046 | [−0.0803, +0.1235] | inconclusive |
| `esm_ensemble_150m` | −0.0685 | [−0.1188, +0.0625] | inconclusive |
| `seq_ensemble_pep_domain` | −0.0091 | [−0.0888, +0.0847] | inconclusive |

All four cross zero. Two point estimates ordered the way a size effect would
order them, with intervals that each admit the opposite ordering, is not a
trend — it is two draws from a noisy statistic. See the two sections below on
why this split cannot resolve the question and why the obvious alternative
explanation cannot be ruled out.

### The discriminating test, and the complete null

The gap statistic is a difference of differences, which is why it is noisy. The
better-powered question is a **single** difference: does an arm's score inside
the far stratum genuinely beat the baseline's? Same 55-allele shared panel,
2,000 cluster resamples, each stratum scored on its own rows.

| Model vs `seq_ensemble_pep_pseudo` | within d=4 | 95% CI | within d>=5 | 95% CI |
|---|---:|---|---:|---|
| `esm_ensemble` | −0.0246 | [−0.0706, +0.0312] | −0.0049 | [−0.0804, +0.0632] |
| `esm_plus_seq_ensemble` | −0.0339 | [−0.0707, +0.0383] | −0.0385 | [−0.1167, +0.0542] |
| `esm_ensemble_150m` | −0.0466 | [−0.0906, +0.0166] | **+0.0219** | **[−0.0776, +0.0619]** |
| `seq_ensemble_pep_domain` | −0.0397 | [−0.0821, +0.0141] | −0.0305 | [−0.1067, +0.0406] |

The bolded row is the decisive one. The 150M arm's apparent far-stratum
advantage — 0.6758 against the baseline's 0.6539 — is +0.0219 with an interval
spanning [−0.078, +0.062]. It does not survive.

**Twelve intervals in total bear on the distance question: four gap tests and
eight within-stratum tests. Every one crosses zero.** The question was asked
three different ways — gap, near stratum, far stratum — and this split answered
none of them. That is a thorough null, not a single underpowered look.

A note for anyone reading `reports/stage6_val_esm_stratum_ci.csv` directly: the
150M d>=5 row has a point estimate of +0.0219 against an interval of
[−0.0776, +0.0619], so the point estimate sits in the upper part of its own
interval rather than at the centre. That is expected. The statistic is a median
over a panel that is rebuilt on every resample, and the bootstrap distribution
of a resampled-panel median is skewed. It is one more reason not to read a
point estimate from this statistic without its interval attached.

## 3. Precision@10 at 2 hours

| Arm | Median P@10 | Median base rate | Median lift | Median ceiling share | At ceiling | Below base rate |
|---|---:|---:|---:|---:|---:|---:|
| `seq_ensemble_pep_pseudo` | 0.700 | 0.359 | 0.2910 | 0.900 | 29 | 0 |
| `esm_ensemble` | 0.700 | 0.359 | 0.2919 | 0.900 | **31** | 1 |
| `esm_plus_seq_ensemble` | 0.700 | 0.359 | 0.2767 | 0.900 | 27 | 1 |
| `esm_ensemble_150m` | 0.700 | 0.359 | 0.2939 | 0.900 | 29 | 2 |
| `seq_ensemble_pep_domain` | 0.700 | 0.359 | 0.2767 | 0.900 | 25 | 1 |

All five arms share a median precision of 0.700 and a median ceiling share of
0.900. Lift spans 0.277 to 0.294 — a range of 0.017 across arms that differ by
0.04 on the primary metric. The one validation allele with no peptide above 2 h
(`HLA-B*08:01`) is excluded from `ceiling_share` and counted separately in
every row, as it is everywhere else.

**Precision@10 does not separate these arms.** It separated the single network
from the ensemble only on lift and alleles-at-ceiling, and here even those
columns are within noise of each other. Reported for completeness and because
it is the predeclared secondary metric; no claim rests on it.

## 4. Nested near-neighbour evaluation (mutant ranking inside training)

**What this is and is not.** The shipped prediction files cover validation rows
only, and this analysis lives inside the training split, so it cannot use the
30-member ensembles directly. The arms below are **ridge heads on the same
cached features**, fit out-of-fold on
`pepstab.stage6.nested_folds` (5 folds by peptide, seed 20261004), with PCA
refit inside each fold on that fold's training rows only. It measures the
**representation under a matched linear head**, not the shipped arm.

Out-of-fold concordance on the 490 scoreable same-allele mutant comparisons
(Hamming 1-2) that sit on 96 independent peptide clusters:

| Arm (ridge head, out-of-fold) | Concordance | d=1 | d=2 | Minus chance | 95% CI |
|---|---:|---:|---:|---:|---|
| BLOSUM62, peptide+pseudosequence | 0.5224 | 0.5188 | 0.5276 | +0.0224 | [−0.0448, +0.0939] |
| ESM-2 35M, **selected representation** | 0.4652 | 0.4770 | 0.4479 | −0.0348 | [−0.1096, +0.0459] |
| ESM-2 650M, mean-pooled (**ablation, see below**) | 0.3905 | 0.3933 | 0.3865 | −0.1095 | [−0.1712, −0.0384] |
| constant (control) | **0.5000** | 0.5000 | 0.5000 | — | — |

**Paired, ESM-2 35M selected representation vs BLOSUM62: −0.0572,
95% CI [−0.1456, +0.0378] — inconclusive.** The half-width is 0.092, wider
than the 0.055 recorded in the machinery report, because that figure was
measured on a more correlated pair of arms; two arms that disagree more give a
wider paired interval. Neither arm's own interval against the 0.5 chance floor
excludes it either. **No claim about ESM-2 and mutant ranking is supported by
this analysis** — in either direction.

The selected representation uses the 35M checkpoint, middle layer, per-position
peptide embeddings and the 34 HLA contact positions, reduced by PCA to 256 and
74 components, matching `reports/stage3_headline.json`. PCA is refit inside
each fold on that fold's training rows only, so a held-out peptide never
contributes to the basis.

### The mean-pooled ablation, and why it is in the table

`ridge_esm650m_meanpool` is **an ablation, not an ESM arm.** It uses
mean-pooled peptide and mean-pooled HLA embeddings — a representation stage 3
considered and did not select. It is close to **blind by construction** to the
question this analysis asks: mean-pooling over nine residues dilutes a single
substitution to a ninth of the signal, and the HLA block is identical for both
members of a same-allele mutant pair, so almost nothing distinguishing the two
peptides survives into the features.

It is kept because it is the harness's own control: a representation that
cannot see point mutations should score badly on a point-mutation metric, and
it does, conclusively. That is evidence the nested evaluation measures what it
claims. It is **not** evidence about ESM-2, and the first version of this
analysis used it by mistake before the selected representation was checked
against `reports/stage3_headline.json`.

## Provenance of the distance-profile hypothesis

**Written 4 October 2026, before the test split was scored, so it cannot be
reconstructed favourably afterwards.**

The distance stratification itself is **predeclared**. It is specified in
`EVALUATION.md` and in the plan's stage 6, with the frozen 20-row stratum bar
and the 65-allele shared panel, and it runs in the single test pass whether or
not anything below is true. Nothing is being added to that pass.

The **specific hypothesis** below is not predeclared. It was generated on the
validation split, by this workstream, on the morning of 4 October 2026, by
looking at the validation stratum table:

> A larger ESM-2 checkpoint has a flatter distance profile — it loses less
> going from d=4 to d>=5 than the sequence baseline does.

It arose because `esm_ensemble_150m` dropped 0.0136 across the strata where the
sequence baseline dropped 0.0821, and because the 35M arm sat between them at
0.0624, which is the ordering a size effect would produce. **Every interval
testing it on validation crosses zero** (table above). It is a pattern in point
estimates that the validation split cannot resolve.

Consequences for how the test numbers may be described, whatever they are:

- If the test results show the same ordering, that is **a hypothesis generated
  on validation and tested once on test** — suggestive, not confirmatory. It is
  a different epistemic object from the predeclared comparisons in the same
  table and must be labelled differently wherever it appears.
- A hypothesis that survives one look at better-powered data is **worth a
  proper test on a dataset that can carry it**, which is the honest conclusion
  available; it is not an established finding.
- If the test results do not show it, that is simply the end of it.

The same labelling applies to any other pattern this document notices in point
estimates whose intervals cross zero.

## Why this split cannot answer the distance-profile question

Worth recording independently of the answer, because it tells a reader why a
claim the point estimates appear to support is not being made.

The stratum gap is a **difference of differences of medians**: a median over 55
alleles within d=4, differenced against the same within d>=5, then differenced
between two arms. Each stratum holds roughly 1,377 and 992 validation rows, so
each median rests on about half the data the primary metric uses, and the
four-way difference compounds the noise of all of them.

The resulting 95% CI half-width is near **0.09**, against a largest observed
gap difference of **0.069**. **This split cannot resolve distance-profile
differences between arms at all.** That is a statement about the size of the
validation split, not about the arms.

The test split is better powered on both axes: twice the rows, and the frozen
20-row stratum bar yields a 65-allele shared panel covering ~96% of each
stratum's rows, against 55 alleles covering 80% and 91% here. It is the right
place to ask the question — once.

## The weak-model confound is unresolvable from this data

`esm_ensemble_150m` is last or second-to-last of the four non-domain arms on
**every** metric here: median per-allele rho 0.6737, differential concordance
0.8174, MAE 0.5058. An arm that is uniformly weaker and also flatter across
distance is exactly what the "nothing to lose" explanation predicts — a model
that was never exploiting near-neighbour similarity has none to surrender when
the near neighbours are removed.

Nothing in these intervals separates that from genuine distance-robustness, and
no further re-analysis of validation will, because the two explanations make
the same prediction for every quantity this split can measure. Distinguishing
them needs either a better-powered split or an arm that is distance-flat
*without* being weaker overall. Stated here rather than left implicit.

## What the machinery adds over the headline

<!--SUMMARY-->

## Limitations

- Validation only; nothing here was tuned on test and nothing was measured on
  test.
- The nested arms are ridge heads on the selected features, not the shipped
  30-member MLP ensembles. Treat them as a representation comparison.
- `esm_ensemble_150m` is the weakest arm on the primary metric. Any claim that
  it degrades less with distance has to clear the obvious alternative
  explanation — a model that was never exploiting near-neighbours has less to
  lose — which is why the discriminating test below is its *absolute* score in
  the far stratum, not the size of its gap.
- Every delta is against specific prediction files; the digests are in
  `reports/stage6_val_esm_manifest.json`.
