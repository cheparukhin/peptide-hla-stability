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

### Against the full-domain ensemble

The plan added the full-domain sequence arm at stage 1 so that "more input"
could not be mistaken for a pretraining benefit: it feeds a one-hot/BLOSUM head
the same 182 HLA residues ESM-2 sees. That makes it the controlled comparison
for what pretraining extracts from a fixed input, and it is the one comparison
in the headline that is not flat.

| Model vs `seq_ensemble_pep_domain` | Median rho | 95% CI | Mean rho | 95% CI | Diff. concordance | 95% CI |
|---|---:|---|---:|---|---:|---|
| `esm_ensemble` | +0.0301 | [−0.0084, +0.0757] | **+0.0312** | **[+0.0082, +0.0540]** | **+0.0152** | **[+0.0046, +0.0241]** |
| `esm_plus_seq_ensemble` | +0.0231 | [−0.0163, +0.0746] | +0.0236 | [−0.0013, +0.0507] | **+0.0111** | **[+0.0012, +0.0207]** |
| `esm_ensemble_150m` | +0.0208 | [−0.0259, +0.0576] | +0.0149 | [−0.0037, +0.0349] | +0.0036 | [−0.0039, +0.0112] |
| `seq_ensemble_pep_pseudo` | +0.0402 | [−0.0009, +0.0727] | **+0.0348** | **[+0.0197, +0.0502]** | **+0.0124** | **[+0.0061, +0.0186]** |

The median row for `esm_ensemble` reproduces `reports/stage3_headline.json`
exactly (+0.0301 [−0.0084, +0.0757]) from an independent code path.

**ESM-2 conclusively beats the full-domain sequence ensemble** on two of the
three statistics — the mean panel statistic and the differential concordance —
while the contract's median calls it inconclusive. The bounded claim this
supports: *given the same 182 HLA residues, ESM-2 extracts more usable signal
from them than a one-hot/BLOSUM encoding does.*

Three things keep that claim honest:

- It is **not** a claim that ESM-2 beats the best sequence arm. The
  pseudosequence ensemble beats the full-domain arm by a similar margin on the
  same statistics (+0.0348 mean, +0.0124 concordance), and ESM-2 does not beat
  *it*. What pretraining buys here is recovered by hand-picking the 34 contact
  residues instead.
- The **mean** interval runs to +0.0540, so it does not establish the 0.05 bar
  either. "Conclusively better than the full-domain arm" and "worth 0.05" are
  different statements and only the first is supported.
- The median — the contract's primary metric, and the only one the predeclared
  verdict rule applies to — remains inconclusive at [−0.0084, +0.0757]. The
  conclusive verdicts come from secondary statistics, and that must travel with
  the claim.

The 150M checkpoint shows the same sign on all three and reaches conclusiveness
on none.

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
(Hamming 1-2) that sit on **96 independent peptide clusters**.

The two ESM arms are the **shipped ensembles**, scored out-of-fold on
`pepstab.stage6.nested_folds` (imported, not reimplemented; 5 folds by peptide,
seed 20261004). For each outer fold the full 30-member ensemble was rebuilt on
the other four and predicted the held-out one — 150 networks per arm — at the
selected configuration, with PCA refit inside each fold on that fold's training
rows only. The arm that produced the predictions did not score them.

| Arm | Concordance | d=1 | d=2 | Minus chance | 95% CI |
|---|---:|---:|---:|---:|---|
| `esm_ensemble` (shipped, 150 nets) | 0.4851 | 0.4854 | 0.4847 | −0.0149 | [−0.0896, +0.0622] |
| `esm_plus_seq_ensemble` (shipped, 150 nets) | 0.4801 | 0.4644 | 0.5031 | −0.0199 | [−0.1000, +0.0616] |
| `ridge_blosum_pep_pseudo` (reference, ridge head) | 0.5224 | 0.5188 | 0.5276 | +0.0224 | [−0.0448, +0.0939] |
| `ridge_esm650m_meanpool` (**ablation**, see below) | 0.3905 | 0.3933 | 0.3865 | −0.1095 | [−0.1712, −0.0384] |
| constant (control) | **0.5000** | 0.5000 | 0.5000 | — | — |

**Paired, `esm_plus_seq_ensemble` vs `esm_ensemble`: −0.0050,
95% CI [−0.0508, +0.0415] — inconclusive, half-width 0.046.**

Neither shipped arm is distinguishable from chance, both sit slightly below 0.5
in point estimate, and the two are indistinguishable from each other. Adding
the raw sequence encoding to ESM-2 does not measurably change mutant ranking.

The 0.046 half-width is **tighter than the 0.092 recorded in the machinery
report**, because that figure was measured on an ESM-versus-BLOSUM ridge pair
that disagree substantially, whereas these two shipped arms both contain ESM-2
and are highly correlated — pairing cancels their shared noise. On this
particular comparison the harness would have resolved a difference of about
0.05 concordance, and there is none. The half-width is a property of the arm
pair, not of the harness, and must be quoted per comparison.

**What is still missing.** The headline nested question is ESM-2 against the
*sequence* baseline, and it cannot be asked properly yet: the only sequence arm
available out-of-fold is the ridge reference, which uses a different head from
the shipped ESM ensembles. Comparing 0.5224 against 0.4851 across that pair
would confound representation with head and ensemble size, so it is not
reported as that comparison. It needs
`preds/seq_ensemble_pep_pseudo_oof.csv` on the same folds at the selected
configuration; scoring it afterwards takes under two minutes.

### The mean-pooled ablation, and why it is in the table

`ridge_esm650m_meanpool` is **an ablation, not an ESM arm.** It uses
mean-pooled peptide and mean-pooled HLA embeddings — a representation stage 3
considered and did not select. It is close to **blind by construction** to the
question this analysis asks: mean-pooling over nine residues dilutes a single
substitution to a ninth of the signal, and the HLA block is identical for both
members of a same-allele mutant pair, so almost nothing distinguishing the two
peptides survives into the features.

It is kept because it is the harness's own control, and the completed table
makes the point sharply: **it is the only row with an interval excluding
chance.** Every representation that can see a single substitution lands in a
band around 0.5 that this dataset cannot resolve; the one representation that
cannot see it scores conclusively below chance. That is the nested evaluation
demonstrating it measures what it claims. It is **not** evidence about ESM-2,
and the first version of this analysis used it by mistake before the selected
representation was checked against `reports/stage3_headline.json`.

### Two near-misses worth recording together

Both would have produced a plausible number with nothing visibly wrong:

1. **The mean-pooled representation.** A single substitution is one ninth of a
   mean-pooled 9-mer, and the HLA block is identical within a same-allele
   mutant pair, so the features are nearly blind to the comparison being
   scored. It yields a dramatic, conclusive, worse-than-chance result that
   reads as a finding about ESM-2.
2. **PCA fitted across folds.** Fitting the basis on all training rows rather
   than per fold leaves the *labels* untouched, so nothing about the setup
   looks dishonest from the outside — but the basis has then seen rows it will
   later predict. Both the shipped arms and the ridge reference refit PCA
   inside each fold.

Neither is caught by a leakage check that only watches labels. They are caught
by asking what the representation can physically express about the question
being asked.

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

Of every comparison in this run, **the differential target is the only
analysis that returns a conclusive verdict on anything.** The primary metric,
the distance strata, precision@10 and the nested mutant evaluation are
inconclusive on every arm.

Counted over the 8 paired comparisons in
`stage6_val_esm_paired_ci.csv` and `stage6_val_esm_vs_domain_ci.csv` combined
(4 arms against each of two baselines), the 12 stratum intervals, and the
nested paired row.

| Analysis | Comparisons | Conclusive |
|---|---:|---:|
| Median per-allele rho (contract primary) | 8 | **0** |
| Mean per-allele rho | 8 | 4 |
| Differential concordance | 8 | **5** |
| Distance strata (gap + within-stratum) | 12 | **0** |
| Precision@10 median | 4 | 0 |
| Nested mutant ranking | 1 paired | 0 |

What the machinery establishes that the headline does not:

1. **A measured equivalence rather than a shrug.** ESM-2 and the sequence
   baseline are equal to within one point of concordance on cross-allele
   ranking (+0.003 [−0.005, +0.010]), and the same measurement conclusively
   separates two other arms, so the equivalence is a null with power behind it.
2. **ESM-2 conclusively beats the full-domain sequence ensemble** on the mean
   panel statistic and on the differential, which the median cannot resolve.
   Given the same 182 HLA residues, pretraining extracts more from them — but
   hand-picking the 34 contact residues recovers the same ground.
3. **The distance-robustness hypothesis is dead**, across twelve intervals and
   three different ways of asking.
4. **Mutant ranking is unresolved for every arm**, and the dataset, not the
   method, is why.
5. **Precision@10 is quantised past usefulness here** — all four deltas are
   exactly 0.0000 with intervals of [−0.100, +0.100], because a median over 68
   alleles of a ten-slot statistic sits on a lattice point and stays there
   under resampling. It is reported as the predeclared secondary metric; no
   claim rests on it, and only its lift and ceiling-share columns carry
   information.

## Artefacts and cost

| File | Contents |
|---|---|
| `stage6_val_esm_summary.csv` | headline, 5 arms |
| `stage6_val_esm_per_allele.csv` | per-allele Spearman and precision, 340 rows |
| `stage6_val_esm_distance_census.csv` | rows/peptides per distance |
| `stage6_val_esm_distance_strata.csv` | stratum scores, shared panel |
| `stage6_val_esm_stratum_ci.csv` | 12 stratum intervals (gap + within) |
| `stage6_val_esm_differential.csv` | differential, 5 arms |
| `stage6_val_esm_differential_allele_pairs.csv` | per-allele-pair detail |
| `stage6_val_esm_precision_summary.csv`, `..._per_allele.csv` | precision@10 |
| `stage6_val_esm_paired_ci.csv` | 16 paired intervals vs the baseline |
| `stage6_val_esm_vs_domain_ci.csv` | 12 paired intervals vs the full-domain arm |
| `stage6_val_esm_resampling_units.csv` | cluster/peptide/row widths, 3 seeds |
| `stage6_nested_shipped_nested_mutant.csv`, `..._paired.csv` | nested, shipped arms |
| `stage6_val_esm_manifest.json`, `stage6_nested_shipped_manifest.json` | SHA-256 of every prediction file read |

Runtime, one core, BLAS pinned to one thread:

| Run | Wall time |
|---|---:|
| Main five-arm run (2,000 resamples) | 1,969 s (32.8 min) |
| Stratum and vs-domain intervals | 2,443 s (40.7 min) |
| Nested on the shipped arms | 103 s |
| **Total** | **~75 min** |

The resampling-unit comparison reproduced the machinery report's ratios on a
different arm pair: row/cluster 0.888 on the paired delta and 0.841 on a single
arm, against 0.898 and 0.832 previously.

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
