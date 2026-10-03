# Stage 2b: weak-binder augmentation

Does adding assumed-zero training rows help the sequence baseline, and does
their source matter?

**No. Neither source helps, and the intervals rule out a gain worth having.**
The best arm moves the primary metric by +0.024 against a predeclared 0.05 bar.
All eight paired intervals cross zero and exclude 0.05 — the strongest negative
this validation set can produce.

The useful finding is *why*. The two sources supply different kinds of negative,
and the model already handles both. Measured weak binders have the right anchor
residues — they look like binders — and the model already scores them near the
assay floor. Predicted weak binders from random human peptides have the *wrong*
anchors, and the model already scores them *below* the floor. Training data that
confirms what the model already knows doesn't teach it anything new.

All numbers are **validation**, on measured labels only. Assumed zeros never
enter validation or test. The 0.05 bar is predeclared in
[EVALUATION.md](../EVALUATION.md).

Regenerate:

```bash
.venv/bin/python scripts/augment_affinity.py    # manifests, ~40 s
.venv/bin/python scripts/baseline_augmented.py  # arms + intervals, ~13 min
.venv/bin/python scripts/stage2b_negatives.py   # pool characterisation, ~10 s
```

`scripts/augment_affinity.py --sweep-predictor` reproduces the predictor
diagnosis below; `--skip-predicted` builds only the measured arm.

Artifacts: `data/augmentation/` (three manifests + `provenance.json`),
`stage2b_arms.csv`, `stage2b_runs.csv`, `stage2b_deltas.csv`,
`stage2b_coverage.csv`, `stage2b_negatives.csv`, `stage2b_composition.csv`,
`stage2b_predictor_sweep.csv`, `stage2b_headline.json`.
Guards: `tests/test_augmentation.py` (47).
Prediction files: `preds/stage2b_<arm>.csv`, rescorable with the shared CLI.

## What is held fixed

Every arm uses the stage 2 selected config unchanged: one-hot
peptide+pseudosequence, hidden (256, 64), L2 1e-5, seeds 0/1/2, the same 17,744
measured fit rows, the same 1,972-row stopping fold, and the same 68-allele
validation panel. Augmentation is the only variable.

Two design choices make the comparison exact:

1. **Augmented rows enter the fit set only.** The stopping fold stays
   measured-only and unweighted. Otherwise the stopping criterion would change
   per arm, confounding the comparison.
2. **Uniform weights reduce to no weights.** The weighted loss normalises by row
   count, not weight sum, so weight 1 everywhere is bit-identical to the
   unweighted path. The `measured_only` arm reproduces stage 2 exactly: seed-0
   rho 0.6098, 3-seed mean 0.6096.

## The two sources

| | measured affinity | predicted affinity |
|---|---|---|
| Evidence | lab measurement ≥ 20,000 nM | in-house MLP prediction > 20,000 nM |
| Peptide universe | assay panels (IEDB/MHCflurry) | random 9-mers from human proteins |
| Matched-arm rows | 2,910 | 2,910 |
| Alleles | 51 of 75 | 51 matched, 75 unconstrained |
| Candidate pool after filtering | 36,426 | 468,781 |

Both arms add identical per-allele counts on the same 51 alleles, capped at
25% of each allele's fit rows. The cap binds on 45 alleles; measured
availability limits the other 6. Totals: 2,910 rows, 16.4% of the fit set.

### Where the measured pool comes from

Starting from the 110,591-row affinity reference:

| step | rows |
|---|---:|
| at or above 20,000 nM | 46,479 |
| after dropping C67S constructs | 46,418 |
| after dropping pairs with a measured half-life | 46,157 |
| after Hamming ≥ 4 exclusion against held-out peptides | **36,426** |

Inequalities are respected: `>` at or above the threshold is a lower bound that
establishes weak binding; the code raises if any row carries `<` at the
threshold. Of the survivors, 35,774 are corroborated by the independent BD2013
snapshot.

### The leakage filter

The affinity reference's `padding_eligible` flag is not safe to use directly. It
checks `(allele, peptide)` pairs, but our splits group peptides across *all*
alleles — so a validation peptide reappears as "absent" whenever it was measured
under a different allele. Proper filtering removes 9,731 rows:

| Hamming to nearest held-out peptide | rows excluded |
|---:|---:|
| 0 | 9,123 |
| 1 | 166 |
| 2 | 131 |
| 3 | 311 |

94% of the exclusions are at distance **zero** — exact held-out peptides offered
under another allele. A pair-wise check would have let every one through.

For the predicted arm, the same rule applied to 40,000 sampled natural 9-mers
excludes only 259 (none at distance 0 or 1). Random peptides rarely land near
the assay panel.

### The ensemble exclusion tax

Both manifests carry an `ok_cv_folds` flag. Under `cv_folds()`, every training
peptide is some ensemble member's stopping peptide, so candidates must clear the
Hamming threshold against *all* 19,716 training peptides. That admits 1,640 of
2,910 measured rows but 2,861 of 2,910 predicted ones — an ensembled rerun would
lose 44% of the measured arm.

### The predictor

Built in-house rather than using NetMHCpan or MHCflurry, because both were
trained on affinity corpora containing our validation and test peptides — their
knowledge of those peptides would be a leakage path into the augmentation labels.
An in-house predictor is held to the same exclusion rule as the augmented rows.

| | |
|---|---|
| Architecture | one-hot peptide+pseudosequence, hidden (256, 64), L2 1e-5 |
| Fit / dev rows | 82,795 / 9,200 |
| Held-out Spearman | +0.706 |
| Weak-call precision / recall | **0.823 / 0.263** |
| Natural 9-mers scored | 2,980,575; 468,781 (15.7%) called weak |

**Why recall is only 0.26.** 26.8% of the dev set sits exactly on the 20,000 nM
boundary — the IEDB convention records competitive-assay non-binders as `>20000`,
right on the classification threshold. An MSE regressor shrinks toward the
conditional mean, pushing most of those borderline rows to the non-weak side.
Architecture sweeps confirm this is structural:

| hidden | L2 | dev Spearman | weak precision | weak recall |
|---|---:|---:|---:|---:|
| 256×64 (shipped) | 1e-5 | 0.706 | 0.823 | 0.263 |
| 256×64 | 1e-4 | 0.712 | 0.825 | 0.264 |
| 512×128 | 1e-5 | 0.694 | 0.832 | 0.195 |
| 256×128×64 | 1e-4 | 0.718 | 0.870 | 0.206 |

The fix is a censored (Tobit) likelihood — out of scope.

High precision is what matters for assigning an assumed-zero label. But the
consequence is that the predicted pool is not a random sample of peptide space —
it is the sixth the predictor is most confident about, which turns out to be
mostly anchor violators (see below).

### Does the assumed-zero label hold up?

Checked on the 413 training pairs that carry both an affinity measurement and a
half-life:

| subset | n | at assay floor | > 2 h | median t½ |
|---|---:|---:|---:|---:|
| all dual-measured | 413 | 8.5% | 50.1% | 2.10 h |
| measured-affinity calls weak | 16 | **75.0%** | 0.0% | 0.00 h |
| predictor calls weak | 0 | — | — | — |

The measured flag is right three quarters of the time and never picks a peptide
above 2 hours. The predictor calls none of the 413 weak (they were pre-selected
for strong binding), so its label quality rests on the 0.823 affinity precision,
not a stability comparison.

## Results

Median per-allele Spearman over 68 eligible validation alleles, mean over 3
seeds. The last two columns split the panel into 46 alleles that receive
augmentation and 22 that do not.

| Arm | Weight | Added | rho | seed range | MAE | P@10 | rho (46 aug.) | rho (22 other) |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| measured_only | — | 0 | 0.610 | [.594, .625] | 0.517 | 0.700 | 0.626 | 0.577 |
| measured_affinity | 0.1 | 2,910 | 0.610 | [.599, .616] | 0.526 | 0.700 | 0.624 | 0.572 |
| measured_affinity | 0.25 | 2,910 | **0.621** | [.599, .638] | 0.520 | 0.700 | **0.631** | 0.570 |
| predicted_affinity | 0.1 | 2,910 | 0.608 | [.603, .618] | 0.517 | 0.700 | 0.626 | 0.570 |
| predicted_affinity | 0.25 | 2,910 | 0.608 | [.604, .614] | 0.518 | 0.700 | 0.626 | 0.573 |
| pred._aff._full | 0.1 | 4,407 | 0.613 | [.606, .625] | 0.522 | 0.700 | 0.629 | 0.592 |
| pred._aff._full | 0.25 | 4,407 | 0.618 | [.604, .638] | **0.511** | **0.733** | 0.634 | **0.601** |

Every arm sits inside the baseline's seed range of [0.594, 0.625]. The largest
movement, +0.011, is a third of that range.

### Paired intervals

Paired cluster bootstrap, 2,000 resamples, both models scored on the same
resample ([EVALUATION.md](../EVALUATION.md)). Computed on the **3-seed-mean
prediction** — a single deterministic predictor that can be paired. The seed-mean
baseline scores rho 0.649 (a 3-network ensemble), so these deltas differ from
the table above.

| Comparison | delta rho | 95% CI | Verdict |
|---|---:|---|---|
| measured_affinity w=0.1 | +0.005 | [−0.041, +0.030] | inconclusive, rules out 0.05 |
| measured_affinity w=0.25 | +0.023 | [−0.033, +0.039] | inconclusive, rules out 0.05 |
| predicted_affinity w=0.1 | +0.004 | [−0.035, +0.040] | inconclusive, rules out 0.05 |
| predicted_affinity w=0.25 | +0.006 | [−0.032, +0.037] | inconclusive, rules out 0.05 |
| pred._aff._full w=0.1 | +0.009 | [−0.039, +0.036] | inconclusive, rules out 0.05 |
| pred._aff._full w=0.25 | **+0.024** | [−0.026, **+0.048**] | inconclusive, rules out 0.05 |

**Source comparison** (predicted minus measured at matched per-allele counts):

| Weight | delta rho | 95% CI | Verdict |
|---:|---:|---|---|
| 0.1 | −0.002 | [−0.025, +0.040] | inconclusive, rules out 0.05 |
| 0.25 | −0.017 | [−0.033, +0.031] | inconclusive, rules out 0.05 |

All eight intervals cross zero and exclude 0.05. The source comparison — the
question this stage exists to answer — is inconclusive and bounded below the bar
in both directions.

## Why: the two pools are different kinds of thing

HLA class I pins a peptide by two anchor residues: P2 (position 2) and PΩ (the
C-terminal end). These anchors largely decide whether a peptide can enter the
binding groove at all. What varies among peptides that *do* bind is how long they
stay — and that's what our target measures.

| Pool | PΩ hydrophobic | PΩ charged | baseline prediction |
|---|---:|---:|---:|
| measured_affinity | 77.6% | 14.3% | +0.126 |
| predicted_affinity | 49.1% | 25.9% | **−0.071** |
| *val rows at assay floor* | 84.2% | 14.3% | +0.218 |
| *val rows above floor* | 84.0% | 13.6% | +1.103 |

Three observations:

1. **Real zero-stability peptides have normal anchors.** Floor and non-floor
   validation rows look the same at the anchor positions (84.2% vs 84.0%
   hydrophobic at PΩ). The assay panel was selected for predicted binding, and
   these peptides do bind — what separates a 0-hour peptide from a 38-hour one
   is not whether it enters the groove.

2. **Measured weak binders also have normal anchors** (77.6%) and the baseline
   scores them near the floor (+0.126 vs +0.218). They are hard negatives —
   peptides that look like binders and were measured not to be. Adding them
   tells the model something it mostly already knows.

3. **Predicted weak binders from random peptides are anchor violators** (49.1%
   hydrophobic, 25.9% charged at PΩ) and the baseline scores them **below** the
   floor (−0.071). They are easier than the easiest real data. Adding them tells
   the model nothing.

The composition tells the same story. Against training peptides, the predicted
pool is depleted in anchor-preferred residues (Y −3.5pp, F −3.4pp, L −1.9pp)
and enriched in ones the groove rejects (E +5.2pp, K +2.7pp, D +1.5pp).

A model trained on 17,744 measured rows has already learned anchor preferences.
2,910 rows restating them don't help it rank peptides that *do* bind.

## Allele coverage

The predicted source can label any allele (it only needs a pseudosequence); the
measured source covers 51 of 75. The 24 it misses: 16 have affinity data but no
weak binders, 5 have no public affinity data at all, and 3 are C67S constructs
whose wild-type affinity describes a different groove. 22 of the 24 are eligible
validation alleles.

`predicted_affinity_full` reaches all 75 alleles with 4,407 rows. On the 22
unaugmented alleles it moves rho from 0.577 to 0.601 at weight 0.25, and it
carries the best MAE and P@10 in the table. That's a median over 22 alleles at
one weight with no interval — a hypothesis for stage 6, not a result.

## Cost

CPU only, $0.

| step | time |
|---|---|
| Manifest build (predictor fit + 3M affinity predictions) | 39 s |
| 21 network fits (7 arms × 3 seeds) | 1.3 min |
| 8 paired cluster bootstraps (2,000 resamples each) | ~11 min |
| Pool characterisation | 10 s |

## Limitations

- **Volume untested.** The 25% cap gives 0.16 augmented rows per measured row;
  Rasmussen et al. used ~2.7 (16x more). This tests the zero-label *mechanism*,
  not their volume. The anchor analysis suggests volume wouldn't help — more easy
  negatives are still easy negatives — but that's a prediction, not a
  measurement.
- **Single networks, not the ensemble.** The stage 2 ensemble is 30 networks.
  This pilot uses 3-seed single networks by design, matching the stage 2
  per-config comparison. An ensembled rerun would lose 44% of the measured arm
  to the stricter `cv_folds` exclusion.
- **One architecture.** Only the selected one-hot peptide+pseudosequence config
  was tested. Augmentation could matter more for a weaker model; that's the
  stage 3 question.
- **Two weights only.** 0.1 and 0.25, as predeclared. Weight 0.25 did better in
  most arms, so the useful range may be above it — but testing that after seeing
  these results would be selection on the outcome.
- **The predicted pool is biased.** At a 15.7% weak-call rate it is the
  high-confidence tail, not a random sample of natural peptide space.
- **The assumed-zero label is sometimes wrong.** A quarter of checkable
  measured-weak calls have a non-zero half-life, though none exceeds 2 hours.

## Consequences for later stages

- **Stage 3 runs unaugmented.** The plan allows carrying augmentation forward
  only if it helps the sequence arm. It doesn't.
- **Stage 6 scores no augmented model.** Nothing here earned a test-set
  evaluation.
- **Stage 2c is unaffected.** It tests affinity as an auxiliary *target* (a
  second prediction head), not as an assumed-zero stability label. The
  dual-measured pairs it trains on are near-canonical binders, not the
  anchor-violating pool that made the predicted arm uninformative here.
- **The filtering code is reusable.** `pepstab.augment` and the exclusion ledger
  apply to any future augmentation, including stage 2c's IEDB expansion.
