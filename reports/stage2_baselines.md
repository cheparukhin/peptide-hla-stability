# Stage 2: supervised sequence baselines

How well can a small model predict stability from labelled sequences alone?
This report establishes the reference point every later stage must beat.

All numbers are **validation**; the test split is untouched. Metrics and the
0.05 minimum worthwhile gain are predeclared in
[EVALUATION.md](../EVALUATION.md).

Regenerate: `.venv/bin/python scripts/baseline_sequence.py` (~10 min, CPU only).
Raw runs in `stage2_runs.csv`; selected configs in `stage2_summary.csv`;
headline config in `stage2_headline.json`.

## Setup

Six arms cross two encodings with three HLA input widths:

- **Encodings:** one-hot, BLOSUM62
- **HLA inputs:** peptide only, peptide + 34-residue contact pseudosequence,
  peptide + 182-residue domain

Each residue becomes a 20-dimensional vector. Vectors are concatenated in
sequence order, preserving position information.

Each arm got the same budget: a 4-point MLP grid (hidden `(64,)` or
`(256, 64)`, L2 `1e-5` or `1e-3`) at 3 seeds, plus a 5-point ridge alpha sweep
as a linear reference. 72 MLP fits and 36 ridge fits, ~10 minutes on one laptop
core.

**All models train on the same 17,744 rows.** The MLP needs a stopping fold, so
10% of train (1,972 rows, 357 whole peptide clusters) is held out as an inner
dev fold. Ridge picks its alpha on that same fold rather than refitting on all of
train. Without this, one model family would see 10% more data and confound every
comparison.

The inner fold moves **whole Hamming ≤ 3 peptide clusters**, reusing the stage 1
water-fill procedure. A random split would put near-duplicate peptides on both
sides and leak labels into early-stopping. Verified: the closest fit–dev peptide
pair is 4 substitutions apart (3,557 fit vs 385 dev peptides), the same
guarantee the frozen splits provide between train and test. Asserted in
`tests/test_baselines.py`.

Validation serves one purpose here: choosing a config per arm. The test split is
scored once, at stage 6.

## Results

### Reference baselines

These use the whole training split and have no hyperparameters:

| Model | Median per-allele ρ | MAE log1p | P@10 (2 h) | Pooled ρ |
|---|---:|---:|---:|---:|
| Training global mean | 0.000 | 0.928 | 0.359 | — |
| Training allele mean | 0.000 | 0.734 | 0.359 | 0.554 |

Both predict a constant within each allele, so per-allele Spearman is undefined
and defaults to 0 by the predeclared rule. The primary metric measures ranking
ability against chance. The allele mean still reaches pooled ρ = 0.554 from
between-allele offsets alone, which is why pooled correlation is secondary.

### Selected models

ρ = median per-allele Spearman over 68 eligible validation alleles,
**averaged over 3 seeds** (seed range in brackets):

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

Seed spread for the selected MLP configs is 0.010–0.051. **Gaps below ~0.05 are
within seed noise** and match the predeclared bar — both reflect what this
dataset can resolve.

No run hit the epoch ceiling (max best epoch 217 of 300), so patience ended
every fit naturally. An earlier 120-epoch cap was hit by 6 of 72 runs, all in
BLOSUM + domain (the weakest arm). Raising the cap changed that arm's score by
0.002 (0.523 → 0.521), confirming it is genuinely weaker, not undertrained.

## Paired comparisons

Paired cluster bootstrap on validation: 2,000 resamples, whole peptide clusters
resampled together, both models scored on each resample. Comparisons use seed 0
of each arm's selected config, fixed in advance (not picked by score).

| Question | Comparison | Δ median ρ | 95% CI | Verdict |
|---|---|---:|---|---|
| Does the model rank at all? | allele mean → MLP pep+pseudo | +0.610 | [+0.557, +0.656] | meets the 0.05 bar |
| Does nonlinearity matter? | ridge → MLP, pep+pseudo one-hot | +0.331 | [+0.256, +0.414] | meets the 0.05 bar |
| Does HLA identity matter? | MLP peptide-only → MLP pep+pseudo | +0.404 | [+0.300, +0.474] | meets the 0.05 bar |
| Pseudosequence or full domain? | MLP pep+domain → MLP pep+pseudo | +0.016 | [−0.031, +0.084] | inconclusive |
| One-hot or BLOSUM62? | MLP one-hot → BLOSUM, pep+pseudo | +0.005 | [−0.049, +0.061] | inconclusive |

### Interpretation

1. **The model ranks peptides within an allele; the allele mean cannot.**
   ρ = 0.61 against a floor of 0, CI far from zero. MAE drops from 0.734 to
   0.517; precision@10 at 2 hours nearly doubles (0.359 → 0.70).

2. **Nonlinearity accounts for most of the model's power.** Ridge on the same
   features reaches only 0.278 — a gap of +0.331 [+0.256, +0.414], the largest
   effect in the table. A linear model on one-hot residues is a position-weight
   matrix: it cannot capture interactions between a peptide residue and the HLA
   pocket it sits in.

3. **The HLA input contributes the other half.** Peptide alone reaches 0.202;
   adding 34 contact residues brings it to 0.610. The model is not just learning
   "some peptides are stable everywhere" — it uses allele identity.

4. **34 contact residues vs 182 domain residues: unresolved.** The
   pseudosequence arm leads nominally (+0.016 on seed-0 files, +0.035 on seed
   means), but the CI crosses zero and the gap is within seed noise.
   Consequence for stage 3: the full-domain arm is a legitimate baseline, so any
   win from domain ESM-2 embeddings must be checked against the full-domain
   raw-sequence arm, not only the pseudosequence arm.

5. **One-hot vs BLOSUM62: unresolved.** +0.005 [−0.049, +0.061] for pep+pseudo.
   With 17.7k training rows the model apparently learns substitution structure
   from one-hot alone. The other two arms separate in opposite directions
   (one-hot ahead at pep+domain, BLOSUM ahead at peptide-only), both within seed
   spread. No encoding preference is established.

## Cost

CPU only, one laptop core, no GPU, no credits.

| Arm | Feature build (s) | Fit (s, mean) | Inference (s / 1,000 rows) | Parameters |
|---|---:|---:|---:|---:|
| peptide + pseudoseq | 0.05 | 1.7–3.2 | 0.0005 | 236,929 |
| peptide + domain | 0.27 | 7.8–19.4 | 0.0020 | 994,689 |
| peptide only | 0.01 | 0.5 | 0.0002 | 11,649 |

Features are cached per unique sequence (28,166 rows carry only 5,633 peptides
and 75 HLA sequences). The full grid runs in ~10 wall minutes, of which 7.5
CPU-minutes is model fitting and the rest is the ridge sweep, feature building,
and scoring. **Inference costs under 1 ms per 1,000 predictions** — the floor
that ESM-2 extraction and GPU folding must justify themselves against.

Ridge is actually *slower* to fit than the MLP on the widest arm (27.8 s vs
12.0 s) — closed-form solution of a 3,820-column system costs more than
minibatch Adam — while scoring 0.30 lower.

## Checks

**The C67S pseudosequence collision is confirmed.** Stage 1 flagged that
`HLA-B*14:01(C67S)` and `HLA-B*14:02(C67S)` share one contact pseudosequence
(756 rows, 2.7%). Both are eligible on validation (41 and 43 rows), with 41
peptides measured on both alleles.

- The pseudosequence arm predicts the two alleles **identically** (max absolute
  difference 0.00 across 41 shared peptides).
- The domain arm separates them (max difference 0.31).
- Measured labels correlate at only ρ = 0.708 between the two alleles — there is
  allele-specific signal the pseudosequence arm cannot reach.

The impact on the primary metric is small, though. Per-allele Spearman is
computed within each allele, so identical predictions can still rank each
allele's labels well (pseudo: 0.487 / 0.613; domain: 0.508 / 0.498). The
collision caps cross-allele discrimination, not within-allele ranking, and is
not why the two arms tie.

**Per-allele failures.** For the headline arm, 1 of 68 validation alleles ranks
backwards (`HLA-A*24:19`, ρ = −0.12) and 4 fall below 0.20 (`HLA-A*24:19`,
`HLA-A*25:01`, `HLA-A*01:01`, `HLA-B*39:06(C67S)`). The domain arm fails on
nearly the same set, suggesting these are hard alleles rather than an
input-representation artifact. `HLA-A*01:01` is already known from stage 1 to
be thin (220 pairs, 43 test rows).

## Limitations

- **Distance stratification cannot be assessed on validation.** Splitting 2,817
  validation rows into d=4 and d≥5 strata and applying the 20-row-per-allele
  bar leaves only 6 alleles and 499 rows. The gap we see (0.715 vs 0.610)
  rests on too few alleles to interpret. Stage 6 runs this on the 5,633-row
  test split, where both strata keep ~65 alleles.
- **Validation scores are selection scores.** Each arm's config was chosen on
  the number reported next to it, making these optimistic as held-out estimates.
  They remain valid for *comparing* arms under the same budget, which is what
  stage 2 is for.
- **Encoding and input-width questions are unresolved, not settled.** Two of
  five CIs cross zero — inconclusive per EVALUATION.md, not negative.
- **No censoring model.** 20.2% of labels sit at the assay floor and are treated
  as exact zeros under `log1p`. A Tobit-style censored loss is a recorded
  limitation.
- Bounded to these representations, this split, and this budget.

## Calibration against NetMHCstabpan (Rasmussen et al. 2016)

Regenerate: `.venv/bin/python scripts/compare_to_paper.py` (~3 min); tables in
`compare_to_paper.csv` and `compare_to_paper_grouping.csv`.

The paper reports **mean per-allotype SCC ≈ 0.69 and PCC 0.676** from 5-fold CV
on this same 28,166-row dataset (figure 1, global rescaling t0 = 1 h; PCC stated
in the text, SCC read off the figure). SCC is the metric to compare on: it is
invariant to the target transform, so it favours neither side's scale.

### This is a calibration, not a reproduction

Our model is **not** NetMHCstabpan reimplemented. It differs on every axis:

| Axis | Rasmussen et al. | Ours |
|---|---|---|
| **Training rows** | **103,166** — 28,166 measured **plus 1,000 assumed-zero weak binders per allele (75,000 rows, 73% of the set)** | **19,716 measured** |
| Encoding A | BLOSUM50 / 5 | BLOSUM62 / 5 |
| Encoding B | smoothed sparse (0.9 / 0.05) | plain one-hot (1 / 0) |
| Architecture | one hidden layer, 40 / 50 / 60 units | two layers, 256×64 |
| Ensemble diversity | 2 encodings × 3 hidden sizes × 5 folds | 2 encodings × 5 folds × 3 seeds |
| Target | `s = 2^(-t0/th)`, t0 tuned | `log1p(th)` |
| Held-out grouping | peptide identity | Hamming ≤ 3 cluster |
| Scored on | the same 1/5 fold used for early stopping | a split never seen during fitting |

The first row dominates. **Their training set is 5.2× ours**, and nearly
three-quarters of it is augmentation we do not have: random natural 9-mers with
predicted affinity weaker than 20,000 nM, assigned `thalf = 0`. Our
`data/rasmussen_et_al_dataset.csv` is the 28,166 *measured* rows; the 75,000
augmented rows were never released with it. Adding them is stage 2b.

So the gap below is **not a model-quality comparison**, and no claim of method
parity is supported. Matching their network count is not matching their method.

### What we can measure

| Factor | Effect on mean SCC | Status |
|---|---|---|
| Split grouping, at equal training rows | Δ median ρ **−0.005 [−0.035, +0.046]** | **inconclusive; rules out a 0.05 gain** |
| Ensembling (3 seeds) | +0.042 | measured |
| Ensembling (30-network CV) | +0.090 | measured |
| The paper's `2^(-t0/th)` target | −0.022 to −0.034 | measured, worse here |

**Split grouping explains nothing — this corrects an earlier claim of +0.018.**
That figure came from re-partitioning the whole dataset by peptide identity and
comparing the result against the frozen score. That experiment was invalid twice
over: it changed the training, stopping *and* scored rows together (only 310 of
2,817 validation rows survived into it), and it consumed frozen test rows (see
EVALUATION.md, "Disclosed test exposure").

The controlled version fixes a common evaluation set carved from the frozen
training split — 2,894 rows, 592 peptides, 68 eligible alleles, 18.8% of rows
within 3 substitutions of a training peptide under identity grouping — and
varies only which training rows are available, at **equal row count**. Result:
**−0.005 [−0.035, +0.046]**. Inconclusive, and it rules out a 0.05 advantage.
The identity-grouped arm *does* score +0.007 higher when it keeps its extra 830
near-neighbour rows, but that is the row count, not the neighbours.

So our harder split is not why we score below the paper, and **the gap cannot be
discounted for it.** Given the grouping and target-transform results are null or
negative, the training-set difference is the leading remaining explanation —
which stage 2b tests directly.

**Ensembling is the one factor that clearly moves us.** NetMHC-family training
fits one network per CV fold per architecture and predicts with the ensemble;
stage 2 reported single networks, a weakened form of the same idea.

### The strong form: a 30-network ensemble

`scripts/baseline_ensemble.py`: **5 inner CV folds × 2 encodings × 3 seeds = 30
networks**, averaged. Each network stops on its own fold, so the ensemble
collectively trains on all 19,716 training rows rather than the 17,744 a single
fit/dev cut leaves. Folds are cut along whole Hamming ≤ 3 clusters. Configs come
from `stage2_summary.csv`, not re-tuned — ensembling is the only change.

| Model | Mean SCC | Median per-allele ρ | Mean PCC (paper scale) | MAE log1p |
|---|---:|---:|---:|---:|
| Single network, pep + pseudoseq | 0.573 | 0.610 | 0.568 | 0.517 |
| **30-network ensemble, pep + pseudoseq** | **0.645** | **0.693** | **0.649** | **0.473** |
| 30-network ensemble, pep + domain | 0.610 | 0.653 | 0.621 | 0.495 |
| *NetMHCstabpan (their 5-fold CV, different training set)* | *~0.69* | *—* | *0.676* | *—* |

Ensembling adds **+0.090 mean SCC** on the pseudosequence arm (+0.100 on
domain) — more than the +0.042 from 3 seeds alone, because CV folds add
training-data coverage on top of seed averaging. Members score 0.511–0.555; the
ensemble reaches 0.645. Paired cluster bootstrap vs the single network:
**Δ median per-allele ρ = +0.083 [+0.029, +0.124]**.

PCC here puts **both** predictions and labels on the paper's `2^(-1/th)` scale.
An earlier version correlated log1p predictions against paper-scale labels,
which is neither metric and read 0.639 where the paper-scale value is 0.649.

**Reading the remaining 0.045 SCC.** It is not discountable for the split, and
not explained by the target transform. It is consistent with the training-set
difference (5.2×, mostly augmented negatives), the richer architecture diversity
in their ensemble, and their scoring on the same fold each network stopped on.
We cannot separate those here. **What we have is a strong sequence baseline in
the NetMHCstabpan family, not a reproduction of it.**

The pseudosequence-vs-domain tie survives ensembling: +0.040 [−0.001, +0.073],
still inconclusive.

### Why NetMHCstabpan cannot be a comparator

It was trained on all 28,166 rows, **including every peptide in our test
split**. Any score it produces on our data reflects memorisation, not
generalisation. There is no valid "beat NetMHCstabpan" result available from
this dataset at any stage. The 0.69 is a cross-validation score on its own
training data, useful for calibration only. The baseline stage 3 must beat is
the ensemble above: our method, trained on our train split, scored on data it
has never seen.

(An honest comparison would require stability measurements published after 2016
and absent from its training set. Out of scope.)

Apart from the disclosed exposure in the superseded calibration script, none of
the above was scored on test.


## Rules for stage 3

- **Beat the 30-network ensemble**, not the single network. The ensemble
  (`preds/seq_ensemble_pep_pseudo.csv`) scores median per-allele ρ = 0.693,
  mean SCC 0.645. The single network (`preds/seq_baseline.csv`) is 0.083 lower
  and would give stage 3 a free gap.
- The **full-domain ensemble** (`preds/seq_ensemble_pep_domain.csv`, ρ = 0.653)
  is the comparator for any HLA-domain embedding result. Report both; the two
  arms remain tied.
- The single-network table remains the valid *arm and encoding* comparison (all
  arms are single-network).
- Compare seed means, not single seeds. Treat gaps under ~0.05 as noise.
- **Ensemble both arms the same way, or neither.** Ensembling alone is worth
  +0.090 mean SCC from no new information. An ensembled ESM arm against a
  single-network sequence arm (or vice versa) would manufacture a result.
- **Use `cv_folds()` from `scripts/baseline_ensemble.py`, not `inner_folds()`.**
  The ensemble comparator fits each member on 15,772–15,773 rows across 5 folds,
  collectively covering all 19,716 training rows. `inner_folds()` is the
  single-network protocol: one permanent stopping fold, 17,744 fit rows. An ESM
  arm built on `inner_folds()` would be compared against a baseline trained a
  different way, confounding the result. Match the folds **and** the member
  count (5 folds × encodings/representations × 3 seeds).
- Score with `scripts/evaluate.py --split val preds/seq_ensemble_pep_pseudo.csv
  preds/esm_ensemble.csv` — the ensemble is the baseline argument, so the paired
  CI is measured against it.
