# Phase D protocol — prespecified before any arm was scored

Date: 4 October 2026. Phase D of
[FURTHER_RESEARCH_PLAN.md](../FURTHER_RESEARCH_PLAN.md) §7: reuse existing
structures before expanding compute.

The points this phase holds in common with A, B and C — the outer fold
definition, the inner selection scheme, the eligibility bar, the primary
metric, the interval procedure and the floor event — are inherited from the
shared protocol, [`phase_a_prespecification.md`](phase_a_prespecification.md),
and indexed in [`README.md`](README.md). This file declares only what is
specific to Phase D: the arms, the residual model, the shrinkage ladder and
the selection rule. Where it restates a shared decision it is for readability,
and the shared document governs.

This file is the frozen search space and selection rule. It was written and
committed before any residual arm was fitted or scored, so that a disappointing
arm cannot quietly acquire a wider ladder and a flat ladder cannot acquire a
post-hoc tie-break. Extensions get logged in §7 below with a date, never
substituted silently.

**No new folding and no GPU spend in this phase.** Every structural number
comes off disk from the stage 4c5 production extraction.

## 1. What is being tested

Whether a strongly shrunk correction fitted on the *sequence arm's own
out-of-fold errors* and driven by **already-computed** structural features
improves median per-allele Spearman on held-out peptide clusters.

The question is deliberately not "do structures help" in general. It is the
narrower, cheaper one: given features we already paid for, does a residual
correction survive cluster-held-out assessment.

## 2. Partition and fold contract

| Item | Value |
|---|---|
| Rows | `split == "train"` only, 19,716 rows. Validation is untouched by this phase; the test split is not scored. |
| Outer folds | `scripts.baseline_ensemble.cv_folds(train, 5)` — whole peptide clusters, water-filled. Sizes 3944/3943/3943/3943/3943; 715/718/718/718/718 clusters; verified pairwise cluster-disjoint. |
| Outer fold fingerprint | `sha256` of the int64 assignment vector, first 16 hex: `50fdca745c9e8e99`. Asserted at run time. |
| Inner folds | `cv_folds(fitting_partition, 5)` recomputed **inside** each outer fitting partition, again whole clusters. |
| What an outer held-out fold may never choose | the shrinkage strength, the feature standardisation, any feature transform, the eligibility rule, or the arm. |

`pepstab.stage6.nested_folds` is **not** used here. It folds by peptide and
deliberately splits near-neighbour mutant families across folds, which is right
for stage 6's mutant-ranking question and wrong for a cluster-held-out
assessment. The existing `preds/*_oof.csv` files are built on it and are
therefore not valid inputs to this phase.

## 3. Arms

All arms share one ladder, one fold assignment, one standardisation procedure
and one selection rule. They differ **only** in which structural columns enter
the correction. Feature groups come from `scripts/stage5_structural_arm.py`, so
the grouping is the one the structural arm already committed to rather than a
new one invented for this phase.

| ID | Arm | Columns |
|---|---|---|
| D0 | `none` | no correction — the sequence outer prediction, unmodified. The reference every delta is taken against. |
| D1 | `geometry` | `GEOMETRY_PREFIXES`: contacts, min/max distances, SASA and burial, aggregate and per-position. |
| D2 | `confidence` | `CONFIDENCE_PREFIXES`: pLDDT, PAE in both directions, pTM/ipTM, including `iptm_pair_` in both peptide–HLA directions. |
| D3 | `anchor_geometry` | geometry at P2 and P9 only, plus the aggregate (non-positional) geometry columns. |
| D3b | `anchor_geometry_p1p2p9` | the prespecified P1 extension: geometry at P1, P2 and P9, plus aggregates. |
| D4 | `geometry+confidence` | the combined block. |

Two further arms are declared now as controls, not as candidates:

| ID | Control | Purpose |
|---|---|---|
| D3c | `central_geometry` | geometry at P3–P8 only, plus aggregates. The plan warns that central peptide positions may still carry biology; D1/D4 are the full-feature controls that keep them, and D3c is the direct test of whether they carry anything on their own. |
| DX | `geometry_shuffled` | D1's columns, row-permuted within the fitting partition under a fixed seed. **Sign test only.** A width-matched shuffled block is not a capacity control: it breaks the honest relationship between features and labels while leaving the ladder intact, so its *effect size* is not interpretable. The only reading taken from it is directional — if DX appears to improve on held-out clusters, the harness is leaking and no other arm's number may be reported. |

## 4. Residual model

For outer fold *f*, let `ŷ_seq` be the sequence arm's prediction for a row from
a model that never saw that row, and `r = y_log1p − ŷ_seq` the residual.

A ridge fitted on the fitting partition's residuals predicts `r` from the
standardised structural block; the corrected prediction is
`ŷ = ŷ_seq + r̂`. Ridge, not a network: "strongly shrunk" is the point of the
arm, the features are standardised dense columns, and a linear correction is
the cheapest model that can be shrunk continuously to exactly zero.

**Residuals used for fitting are themselves out-of-sample.** Fitting the
correction on in-sample residuals of the outer fitting partition would shrink
the apparent correction toward zero and understate the arm, so the fitting
partition's residuals come from inner out-of-fold sequence predictions.

**Zero correction is an explicit model choice, not an assumption.** The ladder
contains an exact no-correction point, so inner selection is free to choose it,
and the report states for every arm and fold whether it did. Zero
initialisation preserves the baseline only initially; a fitted correction can
still damage generalisation, which is why every arm is judged on held-out
clusters rather than on fit.

Transforms fitted inside the fitting partition only: column mean and standard
deviation for z-scoring, and the fitting-partition mean as the fill value for a
non-finite cell. (Coverage is 100% — all 28,166 pairs extracted with
`status == "ok"` — so no fill is expected; the rule is declared so that a
future gap cannot be handled after seeing a result.)

## 5. Shrinkage ladder — the whole search space

Seven points per arm, identical across arms:

```
alpha ∈ (1e1, 1e2, 1e3, 1e4, 1e5, 1e6, inf)
```

`inf` is the exact no-correction point. The ladder is shifted far above stage
5's `(1, 10, 100, 1000, 10000)` because this phase asks for a *strongly* shrunk
correction, and it keeps the same budget order (7 points, versus stage 5's 5
ridge alphas and 4 MLP L2 values).

Tuning parity here is equal budget with scale-appropriate ranges, not
transplanted values. Alpha is selected per arm from this common ladder; the
arms differ in column count, so forcing one alpha across arms would handicap
the wider ones. Every selection is recorded.

6 arms × 7 alphas × 5 outer folds × 5 inner folds = 1,050 ridge fits, plus the
DX control. Well inside the plan's bound of 12 prespecified configurations per
family.

## 6. Selection rule

Fitted inside each outer fitting partition, on its inner cluster folds.

1. **Criterion:** inner-fold median per-allele Spearman of the *corrected*
   prediction. The primary metric is a ranking metric, so selection uses a
   ranking criterion. Inner-fold MSE of the corrected prediction is computed
   and reported alongside as a declared secondary; where the two selections
   disagree the disagreement is reported, not resolved after the fact.
2. **Parsimony (1-SE rule).** The ladder may be flat, and an argmax on a flat
   ladder chases boundaries. Among alphas whose mean inner-fold criterion is
   within one standard error (across the 5 inner folds) of the best alpha,
   take the **largest** alpha — the strongest shrinkage, nearest to no
   correction. Parsimony points toward no correction, declared now.
3. **Boundary report, not a gate.** A selection landing at `1e1` or at `inf` is
   reported with its ladder profile. It is not fatal: `inf` is a meaningful
   endpoint (the model choosing not to correct), and the criterion here is
   expected to be flat rather than to have an interior optimum, so stage 5's
   `check_interior` gate would be a category error.

## 7. Assessment and verdict

| Item | Value |
|---|---|
| Primary metric | median per-allele Spearman, on the pooled outer-held predictions (all 19,716 training rows, each predicted by a model that never saw its outer fold). Ties get average ranks; a constant prediction on an eligible allele scores 0. |
| Eligibility panel | frozen before fitting, from splits and labels alone: an allele needs **≥ 20 rows** and ≥ 2 distinct labels. On the pooled training split this gives **69 of 75** alleles, **99.65%** row coverage. Same panel for every arm. This is `MIN_ROWS_BY_SPLIT["val"]`, and it is the bar Phase A froze for all of its arms — see the amendment in §9. |
| Declared sensitivity check | the pooled primary recomputed on the stricter **≥ 50-row** panel (68 of 75 alleles, 99.51% coverage), reported for every arm. Declared now so that neither panel can be chosen after seeing a result. |
| Per-fold secondary | the same metric within each outer fold at the same **≥ 20-row** bar (fold size ~3,943, against validation's 2,817), with coverage reported per fold. |
| Secondary metrics | log1p MAE, precision@10 at 2 h with base rate, per-allele distribution. |
| Intervals | paired peptide-cluster bootstrap, **2,000** resamples, seed **20261003**, resampling whole peptide clusters, each arm paired against D0 on the same resample. |
| Verdict | the six mutually exclusive readings of `describe_delta()` from EVALUATION.md. +0.05 is this project's worthwhile-effect threshold, not a universal detection limit: an interval that excludes 0 below 0.05 is a real but too-small effect, and an interval crossing 0 is inconclusive, not a negative result. |

A bootstrap interval on a selected development winner does not remove
model-selection optimism. Every prespecified arm is reported, with its
selection history, whether or not it won.

## 8. Multi-pose reproducibility — what may and may not be claimed

Measured from existing records only: `reports/stage4c5_pilot_features.csv`
(2 models × 3 arms × 3 seeds × 5 complexes = 90 folds),
`reports/stage4c5_feature_variation.csv`, and the crystal-agreement records
`reports/boltz_pose_check.csv` / `reports/esmfold_pose_check.csv`.
**No new poses are generated.**

Reproducibility is reported as a reliability ratio and as the attenuation it
implies for a feature's correlation with any outcome, which is what "sample
sufficiency" means operationally: how many seeds would have to be averaged to
recover a given fraction of a feature's usable between-complex signal.

Across-seed variance measures prediction ambiguity and the sampler's
behaviour. It is **not** a thermodynamic ensemble, **not** a conformational
population, and **not** an unbinding rate. No energetic or kinetic
interpretation is drawn from it anywhere in this phase.

## 9. Logged amendments

**4 October 2026, before any arm was fitted — eligibility bar lowered from
≥ 50 to ≥ 20 rows, for cross-phase panel consistency.** The first version of
this file froze ≥ 50 rows on the pooled training split, reasoning by size from
the test split's bar. Phase A had independently frozen ≥ 20 rows and ≥ 2
distinct labels for all of its arms, reasoning by size from validation's bar.

The two could not both stand. Phase D's D0 reference *is* Phase A's A0
sequence arm, so a panel differing by one allele would make the two reports'
medians differ for a reason that has nothing to do with any model. Phase D
adopts Phase A's bar. The stricter ≥ 50 panel is retained as a declared
sensitivity check reported for every arm, so the looser bar cannot flatter a
result unnoticed and neither panel can be picked after the fact.

No arm had been fitted or scored when this was changed.

**4 October 2026, after the ridge-reference plumbing run, before the Phase A
run — a declared trigger and fixed extension for a low-edge ladder hit.** The
harness validation on the ridge reference arm selected the ladder's lowest
alpha (`1e1`) for the confidence block in three of five outer folds. That is a
truncated ladder at the weak-shrinkage end: the correction wants *less*
shrinkage than the ladder offers, so the arm is handicapped by its own search
space. On this project that failure mode has a price tag — a transplanted
ladder cost the ESM arm 0.093 median SCC and nearly produced a false negative.

The trigger and the remedy are both fixed here, in advance of the Phase A run:
**if any candidate arm's selected alpha lands on `1e1` in the Phase A run, the
ladder is extended downward once to `(1e-1, 1e0, 1e1, …, 1e6, inf)` — nine
points — every arm is re-selected on the extended ladder, and both ladders'
results are reported.** One extension, a declared size, applied to every arm
rather than to the ones it would flatter.

The reference run that exposed this is a plumbing check on a deliberately weak
sequence arm, not the comparator, and none of its numbers are a result. What is
being carried forward from it is the knowledge that the ladder's low end is
reachable — which is information about the search space, not about any arm's
effect.

**4 October 2026, before any arm was fitted — added a near-constant-column
diagnostic.** `iptm_pair_hla_chain_rows_peptide_cols` sits at mean 0.9926 with
sd 0.0028, close to its ceiling. Z-scoring rescales every column to unit
variance, so a near-constant column enters the ridge with its noise amplified
to the same scale as a column carrying real spread. For the confidence block
the report therefore includes, as a diagnostic and **not** as a filter, each
column's within-allele variance share. Filtering columns on it would be
post-hoc feature selection; reading a coefficient without it would be
overconfident.
