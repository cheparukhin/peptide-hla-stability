# Stage 2: supervised sequence baselines

What labelled sequences alone teach a small model, and the reference point every
later stage has to beat. All numbers are **validation**; the test split is
untouched. Metrics and the 0.05 minimum worthwhile gain are the predeclared ones
in [EVALUATION.md](../EVALUATION.md).

Regenerate with `.venv/bin/python scripts/baseline_sequence.py`
(~10 min, CPU only). Every run lands in `stage2_runs.csv`; the selected model
per arm in `stage2_summary.csv`; the chosen config in `stage2_headline.json`.

## What was run

Six arms: `{one-hot, BLOSUM62} x {peptide, peptide + 34-residue contact
pseudosequence, peptide + 182-residue domain}`. Each residue becomes a 20-vector
and the vectors are concatenated in sequence order, so position is never pooled
away.

Each arm got the same budget: a 4-point MLP grid (hidden `(64,)` or `(256, 64)`,
L2 `1e-5` or `1e-3`) at 3 seeds, plus a 5-point ridge alpha sweep as the linear
reference. 72 MLP fits and 36 ridge fits, ~10 minutes of wall time on one
laptop core.

**Every model fits on the same 17,744 rows.** The MLP needs a held-out fold to
stop on, so 10% of the training split (1,972 rows, 357 whole peptide clusters)
is cut off as an inner `dev` fold, and ridge picks alpha on that same fold
instead of refitting on all of train. Giving one family 10% more training rows
than another would confound every comparison in the table.

The inner cut moves **whole Hamming ≤ 3 peptide clusters**, reusing the stage 1
water-fill. A random fold would put near-duplicate peptides on both sides and
tune the epoch count on leaked rows. Measured: the nearest fit/dev peptide pair
is 4 substitutions apart (3,557 fit peptides against 385 dev peptides) — the
same guarantee the frozen splits give between train and test, asserted in
`tests/test_baselines.py`.

Validation is used for one thing: choosing a config per arm. Stage 6 scores test
once.

## Results

Reference baselines, from the whole training split (they have no
hyperparameters and nothing to stop):

| Model | Median per-allele ρ | MAE log1p | P@10 (2 h) | Pooled ρ |
|---|---:|---:|---:|---:|
| Training global mean | 0.000 | 0.928 | 0.359 | — |
| Training allele mean | 0.000 | 0.734 | 0.359 | 0.554 |

Both are constant within an allele, so every per-allele Spearman is undefined
and the panel median is exactly 0 by the predeclared rule. That is the point:
the primary metric is measured against chance. The allele mean still reaches
pooled ρ = 0.554 purely from between-allele offsets — which is why pooled
correlation is a secondary metric here and not the headline.

Selected model per arm. ρ is the median per-allele Spearman over 68 eligible
validation alleles, **averaged over 3 seeds**, with the seed range in brackets:

| Arm | Family | Config | Features | ρ (mean) | ρ range | MAE log1p | P@10 | Fit (s) |
|---|---|---|---:|---:|---|---:|---:|---:|
| peptide + pseudoseq, one-hot | MLP | 256×64, L2 1e-5 | 860 | **0.610** | [0.594, 0.625] | 0.517 | 0.70 | 2.3 |
| peptide + pseudoseq, BLOSUM | MLP | 256×64, L2 1e-3 | 860 | 0.603 | [0.589, 0.615] | 0.533 | 0.60 | 3.2 |
| peptide + domain, one-hot | MLP | 256×64, L2 1e-3 | 3,820 | 0.574 | [0.561, 0.594] | 0.532 | 0.67 | 12.0 |
| peptide + domain, BLOSUM | MLP | 256×64, L2 1e-5 | 3,820 | 0.521 | [0.494, 0.545] | 0.562 | 0.63 | 27.8 |
| peptide only, BLOSUM | MLP | 64, L2 1e-3 | 180 | 0.244 | [0.217, 0.267] | 0.805 | 0.47 | 0.4 |
| peptide only, one-hot | MLP | 64, L2 1e-3 | 180 | 0.202 | [0.196, 0.206] | 0.817 | 0.40 | 0.4 |
| peptide + pseudoseq, one-hot | ridge | α = 0.1 | 860 | 0.278 | — | 0.700 | 0.50 | 1.3 |
| peptide + domain, one-hot | ridge | α = 1 | 3,820 | 0.274 | — | 0.697 | 0.50 | 27.8 |
| peptide + pseudoseq, BLOSUM | ridge | α = 0.1 | 860 | 0.270 | — | 0.700 | 0.50 | 1.1 |
| peptide + domain, BLOSUM | ridge | α = 1 | 3,820 | 0.259 | — | 0.696 | 0.50 | 29.9 |
| peptide only, one-hot | ridge | α = 100 | 180 | 0.170 | — | 0.834 | 0.40 | <0.1 |
| peptide only, BLOSUM | ridge | α = 1 | 180 | 0.169 | — | 0.829 | 0.40 | <0.1 |

Seed spread for the selected MLP configs is 0.010–0.051. **Any gap below ~0.05
is inside seed noise and is not a result** — which is the same order as the
predeclared 0.05 bar, and not a coincidence: both reflect what this data can
resolve.

No run stopped on the epoch ceiling (max best epoch 217 of 300), so patience
ended every fit and no arm's score is an artifact of a truncated budget. At an
earlier 120-epoch ceiling, 6 of 72 runs did hit the cap — all in
`BLOSUM + domain`, the weakest arm. Raising the ceiling changed that arm's score
by 0.002 (0.523 → 0.521), so it is genuinely weaker, not undertrained.

## The four questions, with paired CIs

Paired cluster bootstrap on validation, 2,000 resamples, whole peptide clusters
resampled together, both models scored on the same resample. Run on the per-arm
prediction files, which are seed 0 of each arm's selected config — fixed in
advance, not picked by score.

| Question | Comparison | Δ median ρ | 95% CI | Verdict |
|---|---|---:|---|---|
| Does the model rank at all? | allele mean → MLP pep+pseudo | +0.610 | [+0.557, +0.656] | meets the 0.05 bar |
| Does nonlinearity matter? | ridge → MLP, pep+pseudo one-hot | +0.331 | [+0.256, +0.414] | meets the 0.05 bar |
| Does HLA identity matter? | MLP peptide-only → MLP pep+pseudo | +0.404 | [+0.300, +0.474] | meets the 0.05 bar |
| Pseudosequence or full domain? | MLP pep+domain → MLP pep+pseudo | +0.016 | [−0.031, +0.084] | inconclusive |
| One-hot or BLOSUM62? | MLP one-hot → BLOSUM, pep+pseudo | +0.005 | [−0.049, +0.061] | inconclusive |

Read in order:

1. **A trained sequence model ranks peptides within an allele; the allele mean
   cannot.** ρ = 0.61 against a floor of 0, CI nowhere near zero. MAE also
   improves (0.734 → 0.517) and precision@10 at 2 hours nearly doubles
   (0.359 → 0.70).
2. **Nonlinearity is most of the model.** Ridge on identical features reaches
   only 0.278. The gap is +0.331 [+0.256, +0.414] — the largest single effect in
   the table. A linear model on one-hot residues is a position-weight matrix; it
   cannot represent the interaction between a peptide residue and the pocket it
   sits in.
3. **The HLA input is the other half.** Peptide alone reaches 0.202; adding 34
   contact residues takes it to 0.610. So the model is not just learning "some
   peptides are stable everywhere" — it is using which allele it is being asked
   about.
4. **34 contact residues versus 182 domain residues is not resolved.** The
   pseudosequence arm is nominally ahead (+0.016 on the seed-0 files, +0.035 on
   seed means), but the CI crosses zero and the gap is inside seed noise. Both
   framings sit below the 0.05 bar. This matters for stage 3: the full-domain
   arm is a *legitimate* matching baseline, so if domain ESM-2 embeddings beat
   the pseudosequence baseline, the win has to be checked against the
   full-domain raw-sequence baseline before it can be attributed to pretraining.
5. **The encoding does not matter here.** +0.005 [−0.049, +0.061] for
   pep+pseudo. BLOSUM62's substitution structure should help most when training
   data is thin; with 17.7k rows the model apparently learns comparable
   structure from one-hot. The encodings do separate on the other two arms, and
   in *opposite* directions — one-hot ahead at pep+domain (0.574 vs 0.521),
   BLOSUM ahead at peptide-only (0.244 vs 0.202) — with both gaps near the seed
   spread. Neither was tested with a paired CI. No encoding preference is
   established.

## Cost

CPU only, one laptop core, no GPU and no credits spent.

| Arm | Feature build (s) | Fit (s, mean) | Inference (s / 1,000 rows) | Parameters |
|---|---:|---:|---:|---:|
| peptide + pseudoseq | 0.05 | 1.7–3.2 | 0.0005 | 236,929 |
| peptide + domain | 0.27 | 7.8–19.4 | 0.0020 | 994,689 |
| peptide only | 0.01 | 0.5 | 0.0002 | 11,649 |

Feature building is cached per unique sequence (28,166 rows carry 5,633
peptides and 75 HLA sequences), so it is a rounding error. The whole grid runs
in ~10 wall minutes, of which 7.5 CPU-minutes is recorded model fitting and the
rest is the ridge alpha sweep, feature building and scoring. **Cost per 1,000
new predictions is under a millisecond of CPU**, which is the floor stage 3 and stage 5 will be measured against: an
ESM-2 arm has to justify embedding extraction, and a structural arm has to
justify GPU hours, against a baseline that is effectively free.

Ridge is *slower* to fit than the MLP on the widest arm (27.8 s vs 12.0 s) —
closed-form solution of a 3,820-column system versus minibatch Adam — while
scoring 0.30 lower. Nothing recommends the linear arm here.

## Two checks worth recording

**The C67S pseudosequence collision is real and measurable.** Stage 1 flagged
that `HLA-B*14:01(C67S)` and `HLA-B*14:02(C67S)` share one contact
pseudosequence (756 rows, 2.7%). Both are eligible on validation (41 and 43
rows), and 41 peptides are measured on both. Confirmed:

- the pseudosequence arm predicts the two alleles **bit-identically** — max
  absolute difference 0.00 across those 41 shared peptides;
- the domain arm does separate them (max difference 0.31);
- the measured labels on those peptides correlate at only ρ = 0.708 between the
  two alleles, so there is allele-specific signal the pseudosequence arm
  structurally cannot reach.

But the consequence for the *primary metric* is small: per-allele Spearman is
computed within an allele, and an identical peptide ordering can still rank each
allele's own labels well (pseudo: 0.487 / 0.613; domain: 0.508 / 0.498). So the
cap is real for cross-allele and shared-peptide discrimination, and largely
invisible to within-allele ranking. It is not the reason the two arms tie.

**Per-allele failures.** For the headline arm, 1 of 68 validation alleles ranks
backwards (`HLA-A*24:19`, ρ = −0.12) and 4 fall below 0.20 (`HLA-A*24:19`,
`HLA-A*25:01`, `HLA-A*01:01`, `HLA-B*39:06(C67S)`). The domain arm fails on
nearly the same set, so these look like hard alleles rather than an artifact of
one input representation. `HLA-A*01:01` is already known from stage 1 to be
thin (220 pairs, 43 test rows).

## What this does not show

- **Distance stratification is not answerable on validation.** Splitting the
  2,817 validation rows into d=4 and d≥5 and applying the 20-row-per-allele bar
  leaves 6 common alleles and 499 rows. The d=4 / d≥5 gap we see (0.715 vs
  0.610 for the headline arm) rests on 6 alleles and should not be read as
  evidence either way. Stage 6 does this on the 5,633-row test split, where both
  strata keep ~65 alleles.
- **Validation scores are selection scores.** The config for each arm was chosen
  on the number reported next to it, so these are optimistic as estimates of
  held-out performance. They are valid for *comparing* arms given the same
  budget, which is what stage 2 is for.
- **The encoding and input-width questions are unresolved, not settled.** Two of
  the five CIs cross zero. Per EVALUATION.md that is inconclusive, not negative.
- **No censoring model.** 20.2% of labels sit at the assay floor and are trained
  on as exact zeros under `log1p`. A Tobit-style censored loss remains a
  recorded limitation.
- Bounded to these representations, this split, and this budget.

## Against NetMHCstabpan (Rasmussen et al. 2016)

Regenerate with `.venv/bin/python scripts/compare_to_paper.py` (~3 min);
table in `compare_to_paper.csv`.

The paper's figure 1 reports, for the released configuration (global rescaling
t0 = 1 h), **average per-allotype SCC ≈ 0.69 and PCC 0.676** — from 5-fold
cross-validation on this same 28,166-row dataset. PCC is stated in the text;
SCC is read off figure 1, so quote it as approximate.

SCC is the comparable metric: it is invariant to the target transform, so it
rewards neither side's choice of scale. Our headline arm reaches **mean
per-allele SCC 0.573** (median 0.610 — the paper aggregates by mean, we report
median, so both are given). That is ~0.12 below NetMHCstabpan.

That gap is not a like-for-like model comparison. Three things differ besides
the model; changing one at a time, holding arm, config, metric and allele panel
fixed:

| Change | Mean SCC | Δ |
|---|---:|---:|
| Stage 2 baseline — frozen split, log1p, single network | 0.573 | — |
| Identity-grouped split (as the paper describes) | 0.591 | +0.018 |
| 3-seed ensemble | 0.615 | +0.042 |
| Both | 0.625 | +0.052 |
| The paper's `2^(-1/th)` target, single network | 0.551 | −0.022 |

1. **Split grouping is worth +0.018, less than expected.** The paper groups "all
   peptide-HLA-I stability data for a given peptide" into one CV group — peptide
   *identity*, so a held-out peptide may sit 1 substitution from a training
   peptide (measured: minimum distance 1, against 4 on our frozen split). That
   should flatter it, and does, but only slightly: stage 1 found just 15.5% of
   peptides have any neighbour within 3 substitutions, so there is limited
   leakage available. **Our split is harder, but it is not why we score lower.**
2. **Ensembling is worth +0.042 — the single largest explained factor.**
   NetMHC-family training fits a network per CV fold per architecture (2
   encodings × 3 hidden sizes × 5 folds ≈ 30 networks) and predicts with the
   ensemble. Stage 2 reports single networks. Averaging just 3 seeds recovers
   +0.042, nearly the whole 0.05 worthwhile-gain bar, **from no new information
   at all.**
3. **Their target transform does not explain anything.** Trained on
   `s = 2^(-t0/th)` at t0 ∈ {0.5, 1, 2}, our model scores 0.022–0.034 *below*
   log1p. The paper's own t0 sweep moved PCC from 0.633 to 0.676, so the choice
   matters for them; it does not transfer to this setup.

That leaves **~0.07 SCC unexplained**, against three factors this comparison
cannot isolate:

- they train each network on 4/5 of the data (~22,500 rows) against our 17,744;
- they ensemble ~30 networks across 6 architectures, not 3 seeds;
- their reported score is measured on the same 1/5 fold used for early stopping
  ("the remaining 1/5 was left for testing and early stop"), which is optimistic
  by an unknown amount.

So the single-network baseline is a *weakened* form of the paper's method —
1 network against ~30. That is a problem for stage 3: ESM-2 beating a hobbled
NetMHCstabpan is not the claim we want to make.

### The strong form: a 30-network ensemble

`scripts/baseline_ensemble.py` builds the method properly — **5 inner CV folds ×
2 encodings × 3 seeds = 30 networks**, averaged, matching the paper's count.
Each network stops on its own fold, so the ensemble collectively trains on all
19,716 training rows rather than the 17,744 a single fit/dev cut leaves. Folds
are cut along whole Hamming ≤ 3 clusters, so the frozen split's guarantee holds
inside the ensemble too. Config per encoding is taken from
`stage2_summary.csv`, not re-tuned — ensembling is the only thing that changed.

| Model | Mean SCC | Median per-allele ρ | Mean PCC (paper scale) | MAE log1p |
|---|---:|---:|---:|---:|
| Single network, pep + pseudoseq | 0.573 | 0.610 | 0.562 | 0.517 |
| **30-network ensemble, pep + pseudoseq** | **0.645** | **0.693** | **0.639** | **0.473** |
| 30-network ensemble, pep + domain | 0.610 | 0.653 | 0.605 | 0.495 |
| *NetMHCstabpan (their 5-fold CV)* | *~0.69* | *—* | *0.676* | *—* |

Ensembling is worth **+0.090 mean SCC** on the pseudosequence arm and +0.100 on
the domain arm — far more than the +0.042 three seeds alone bought, because CV
folds add training-data coverage on top of seed averaging. Individual members
score 0.511–0.555 mean SCC; the ensemble reaches 0.645. On the project's primary
metric the paired cluster bootstrap gives **Δ median per-allele ρ = +0.083
[+0.029, +0.124]** over the single network: a real improvement whose size
against the 0.05 bar is unresolved.

**This is method parity, on a harder benchmark.** The remaining gap is 0.045
mean SCC and 0.037 PCC — and the paper's number carries +0.018 of split
advantage plus an unknown amount of optimism from scoring on its own
early-stopping fold. Adjusting for the split alone puts the two within ~0.02.
We did not reproduce their *number*, and should not try to: part of it is
measurement protocol, not model quality. We reproduced their *method* and
measured it honestly.

The pseudosequence-vs-domain tie survives ensembling: **+0.040 [−0.001,
+0.073]**, still inconclusive. Stage 3 still owes both comparisons.

### Why NetMHCstabpan can never be our comparator

It was trained on all 28,166 rows, **including every peptide in our test
split**. Any score it posts on our data is memorisation, not generalisation. So
there is no "beat NetMHCstabpan" result available from this dataset at any
stage — the 0.69 above is a cross-validation score on its own training data,
quoted for calibration only. The baseline that stage 3 must beat is the
ensemble in the table above: the same method, trained on our train split,
scored on data it has never seen.

(An honest comparison would need peptide–HLA stability measurements published
after 2016 and absent from its training set. Out of scope here.)

None of the above was scored on test.

## Carried into stage 3

- **The baseline to beat is the 30-network ensemble**, `preds/seq_ensemble_pep_pseudo.csv`
  — median per-allele ρ = 0.693, mean SCC 0.645 on validation. Not the
  single-network `preds/seq_baseline.csv`, which is 0.083 lower and would hand
  stage 3 a gap it did not earn.
- The matching **full-domain** ensemble (`preds/seq_ensemble_pep_domain.csv`,
  ρ = 0.653) is the comparator for any HLA-domain embedding result. Report both;
  the two arms are still statistically tied.
- The single-network table above stays valid as the *arm and encoding*
  comparison, because every arm in it is single-network.
- Compare against seed means, not single seeds, and treat anything under ~0.05
  as noise.
- **Ensemble both arms identically, or neither.** Seed-averaging alone is worth
  +0.042 mean SCC — nearly the whole worthwhile-gain bar, from no new
  information. An ensembled ESM arm against a single-network sequence arm (or
  the reverse) would manufacture a result. Stage 2's single-network table stays
  valid as an *arm* comparison because every arm is single-network.
- Reuse `inner_folds()` for the ESM heads so every arm still trains on the same
  17,744 rows and stops on the same 1,972.
