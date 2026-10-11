> **Historical study record.** The protocols, results and future-tense instructions below describe the completed October study. The [current project review and research plan](../NEXT_RESEARCH_PLAN.md) is the sole forward work queue; its [dated correction register](../NEXT_RESEARCH_PLAN.md#dated-correction-register--12-october-2026) supersedes conflicting interpretation, selection and blend-gate claims here. No old protocol should be relaunched as a new study without the current evaluation gates.

# Phase A prespecification

Date: 4 October 2026. Status: **search spaces and decision rules recorded before
any Phase A fit.** This file exists to satisfy
[FURTHER_RESEARCH_PLAN.md](../archive/plans/further-research-2026-10-04.md) §3.3 — "Record the
search spaces before execution; log extensions rather than silently expanding a
disappointing arm's budget."

Any later extension of a grid, a seed count or a fold count is appended to §7
below with its date and reason. Nothing in §1–§6 is edited after the first fit.

## 1. What is being assessed, and on what

**Assessment partition: the training split only.** Phase A uses training-only
nested cluster validation. The frozen validation split is a development resource
that has been consulted repeatedly across stages 2–7; the frozen test split is
spent. Neither is used to choose anything in Phase A, and neither is scored
here.

- **Outer assessment:** five folds over training peptide clusters —
  `scripts/baseline_ensemble.py::cv_folds(train, 5)`, which water-fills whole
  `cluster_id` groups via `pepstab.splits.assign_clusters` and is deterministic
  with no seed. Training is 19,716 rows over 3,587 clusters, so each outer fold
  holds out roughly 3,943 rows. Clusters, not rows, are the partition unit, and
  a cluster groups peptides within Hamming distance 3, so no outer-held-out
  peptide sits within 3 substitutions of a peptide its model trained on.
- **Inner selection:** cluster folds built *inside each outer fitting partition*
  by the same `assign_clusters` primitive. Early stopping, hyperparameters,
  feature transforms, allele target means, calibration maps and blend weights are
  all chosen on inner folds only.
- **Screening budget:** **three** inner folds × two paired seeds, per plan §3.3.
- **Outer assessment (A0 and every finalist):** **five** inner folds, so that an
  arm's ensemble is `5 inner folds × 2 encodings × 3 seeds = 30 members` per
  outer fold — bit-for-bit the same arity, encoding axis and seed protocol as the
  deployed reference ensemble. Five outer folds × 30 members = **150 networks**
  per MLP arm. Inner folds are three during screening and five at assessment
  *because* ensemble parity has to hold where the comparison is made: ensembling
  is worth roughly +0.074 median Spearman here, which is larger than most
  effects Phase A is looking for, so an arm assessed at 18 members against a
  reference at 30 would be measuring ensemble size.
  **[CORRECTED 4 Oct: +0.074 is single-network-versus-ensemble, a different
  comparison, and does not describe a change in ensemble *size*. Measured on
  this screen, 12 members against 30 is worth +0.0156 — the matched 12-member
  control scores 0.6337 against A0's 0.6493 at 30. The parity argument stands;
  the magnitude cited for it was about five times too large.]**

**Why `cv_folds` and not `pepstab.stage6.nested_folds`.** Both exist. The plan
§3.2 asks for "the existing five training-cluster folds", which is `cv_folds`.
`stage6.nested_folds` partitions by *peptide* and is deliberately built so that
a mutant pair lands in **different** folds — that is what its mutant-ranking
diagnostic needs, and it is the wrong property here, because it would let a
model early-stop or be assessed on a peptide one or two substitutions from one
it fitted. Cluster-grouped outer folds are the stricter choice and they match
the frozen split's own construction. The existing `scripts/esm_arm.py::mode_oof`
nested recipe uses `nested_folds` for the outer level; Phase A departs from it
on this point on purpose, and that departure is recorded here rather than left
implicit.

An outer held-out label must not choose anything. Existing
early-stopping-fold predictions from earlier stages are **not** assumed to be
clean stacking inputs and are regenerated here.

## 2. Metrics and verdicts

Primary: **median per-allele Spearman**, with the frozen tie rule (average
ranks) and the frozen constant-prediction rule (a constant prediction on an
eligible allele scores 0, and the allele stays in the panel) from
[EVALUATION.md](../../EVALUATION.md).

**Eligibility on inner and outer folds is frozen here, before fitting.** The
frozen split-level bars (50 rows on test, 20 on val) are calibrated to those
splits' sizes and do not transfer to a fold of the training split. Phase A
rule, from labels and counts alone and never from predictions:

> An allele is eligible on a fold if it has **≥ 20 rows on that fold and ≥ 2
> distinct labels**. Coverage — eligible alleles and the share of fold rows they
> cover — is reported for every fold alongside every score.

A **≥ 50-row panel is also reported for every arm, as a declared sensitivity
check**, so that the looser bar cannot flatter a result unnoticed and neither
panel can be chosen after the fact. On the pooled training split, ≥20 gives
69/75 alleles at 99.65% row coverage and ≥50 gives 68/75 at 99.51% (measured by
the Phase D session; Phase B and D report the same two panels, so the three
reports stay comparable).

**Correction: those figures are pooled, and the ≥50 bar is a much weaker
instrument at fold scope.** Applied per outer fold — which is where Phase A
scores — ≥50 keeps a median of **48 alleles at 75.1% row coverage**, and A0's
fold-to-fold span widens from 0.0484 to **0.0910**. So at fold scope ≥50 is
*noisier* than the primary panel, not stricter: it discards a quarter of the
rows and nearly doubles the spread against which any effect must be read. It is
still reported for every arm as declared, but it must be presented as a
weaker-powered check rather than a more conservative one, and a disagreement
between the two panels at fold scope is more likely to be ≥50's variance than a
finding.

Twenty is `MIN_ROWS_BY_SPLIT["val"]` from `pepstab/evaluation.py`, and it
transfers here on size: an outer fold holds ~3,943 rows against validation's
2,817, and an inner fold ~5,258, so both are at least as large as the split that
bar was calibrated for. The test split's 50-row bar does not transfer. The rest
of the frozen contract — average ranks for ties, `UNRANKED_CONTRIBUTION = 0.0`
for a constant prediction on an eligible allele, alleles with no label spread
leaving the panel, `validate_finite` rejecting NaN/inf before scoring — is reused
unchanged from `pepstab.evaluation`, not reimplemented.

Secondary, reported but never used to select: log1p MAE, precision@10 at 2 h
with base rate and reachable ceiling, per-allele distribution, distance strata
(`dist_to_train`), and training/inference cost.

Pure rankers have no calibrated magnitude. Any ranker that reports log1p MAE or
hour-scale error must first be given a calibration map fitted inside the
training partition, and that map is part of the arm.

### A calibration map must never sit in the path of the primary metric

**State this precisely, because the loose version points at the wrong fix.**
Isotonic regression **preserves rank order** — it is non-decreasing, so it never
inverts a pair. What it destroys is rank **resolution**: being a step function,
it maps strict inequalities to equalities, and Spearman moves only because the
frozen contract gives ties average ranks, a rule EVALUATION.md adopts precisely
because the assay floor creates a large tied block.

So the remedy is **not** "find a monotone map that does not do this" — *every*
weakly monotone map does. The remedy is rule 1 below: do not put any map in
front of the ranking metric. A reader told only "isotonic is not
rank-preserving" could reasonably infer that it reorders predictions and go
hunting for a map that does not, which is a search with no answer.

Measured on this project's own predictions by the Phase B session, over the
68-allele validation panel of `preds/stage7_mse.csv`: isotonic collapsed **2,776
distinct predictions to 57**, and the per-allele Spearman shift ran from
**−0.0431 to +0.1109**, exceeding 0.05 in absolute value on **3 of 68 alleles** —
reaching 2.2× the project's worthwhile bar on a single allele. Worst case was
`HLA-B*39:06(C67S)`: 38 rows, 38 distinct predictions collapsed to 6, rho
0.1561 → 0.2670. A pooled median hides this, because only a few alleles move
far; the pooled figure was −0.016.

**Both signs occur within one panel on one fit**, so this is not a known-sign
bias that a reader could treat as conservative. A strictly monotone affine
rescale on the same data (slope 1.0424) shifted Spearman by exactly 0.0 and
preserved all 2,776 distinct values. Platt scaling likewise preserved all 2,776
and scored Spearman 1.0000000000 against its input — strictly monotone, so it
cannot create a tie. Isotonic applied to the floor event collapsed 2,776 to
**19** and scored only 0.9853, so it does not even preserve resolution on the
quantity it is calibrating.

Therefore, in Phase A:

1. **Primary metric is always scored on uncalibrated predictions.** Median
   per-allele Spearman for every arm, including A4, is computed on the arm's raw
   output. No calibration map, isotonic or otherwise, is applied before it.
2. Calibration maps exist **only** to put magnitude metrics on a comparable
   scale, and are applied only to log1p MAE and hour-scale error.
3. Where a map is reported, its **measured effect on the ranking metric is
   reported alongside it**, not asserted to be zero.
4. Where a monotone rescale is wanted that provably cannot move ranks, use a
   **strictly monotone affine** fit (`y ~ a + b·x` by least squares on the
   fitting partition, refusing any fit with slope `b ≤ 0`) and assert the
   ranking metric is unchanged to within 1e-8. A strictly positive slope cannot
   create ties, so this is checkable rather than hoped for.

   **The guarantee holds at fold scope only — fit and assert within one fold.**
   Each fold gets its own `(a, b)`, so a *pooled* prediction vector is several
   different affine transforms stitched together and is **not** a single
   monotone map: within one allele, rows from different outer folds can
   reorder. The Phase D session's 1e-8 assertion fired for exactly this reason,
   correctly, with a measured pooled shift of ≤ 0.0012 — small, but not zero.
   An assertion placed at pooled scope will fail and should; place it per fold,
   where it passes exactly, and report the pooled shift as a measured quantity.

Credit: raised by the Phase D session, which caught it by asserting the
rank-invariance it had assumed rather than relying on it — the assertion fired
on every arm. The affine slope is additionally the better probe for scale
overshoot, since `b < 1` measures the overshoot directly and at zero cost to
ranking.

Intervals: paired peptide-cluster bootstrap, 2,000 resamples, seed `20261003`,
both arms scored on the same resample, panel rebuilt per resample. Comparisons
are read with EVALUATION.md's six verdicts. **0.05 is this project's
worthwhile-effect threshold, not a detection limit** — an interval excluding 0
with an upper bound below 0.05 is a real but small effect, and is reported as
such rather than as a null.

Every arm is compared against **two** references: the frozen sequence reference
(peptide + pseudosequence ensemble) and the strongest *new* sequence-only
control from A0/A2/A3, so a better learner is not mistaken for a benefit from
embeddings or structure.

## 3. Selection rule inside a screen

Flat ladders are the norm in this project, and plain argmax on a flat ladder
chases grid boundaries. Phase A reuses the house convention — the
one-standard-error parsimony rule already implemented in
`scripts/stage3e_coupling.py::_parsimonious` — stated for each family as:

> Among configurations whose mean inner-fold median Spearman is within one
> standard error of the best, take the **most regularised** one. Report argmax
> alongside selected, the seed spread at each, and whether the selection is
> interior to the grid. A selection at a grid boundary is flagged, not quietly
> accepted.

### Grid-boundary trigger and its remedy, declared in advance

A truncated search space at the weak-regularisation end has already cost this
project a real result once — the ESM arm lost 0.093 median Spearman to exactly
that. So the remedy is fixed here rather than decided after seeing which arm it
would help:

> **Trigger.** Any arm whose parsimonious selection lands on the
> weakest-regularisation boundary of its grid — `l2_leaf_reg == 3` for the tree
> families, `k == 1` for A2's shrinkage, `λ == 1.0` for A3's ranking weight.
>
> **Remedy.** The affected axis extends **once**, by one declared step in that
> direction: `l2_leaf_reg → {1, 3, 10, 30}`, `k → {0.2, 1, 5, 20, 100}`,
> `λ → {0.03, 0.1, 0.3, 1.0}`. **Every arm in that family is then re-selected on
> the extended ladder**, not only the arm that triggered it, and **both ladders
> are reported**. One extension, of declared size, applied to all arms rather
> than to the ones it flatters. A second extension is not available and a
> boundary selection on the extended ladder is reported as a handicap on that
> arm's result.

**DEFECT, found when the trigger fired (see §6b): the declared extensions for
`λ` and `k` run the wrong way.** A3 selected at `λ=1.0` and A2 at `k=100` —
both the *top* of their ladders — but the declared sets add values at the
bottom (`0.03` and `0.2`). A fired trigger therefore has no step past it for two
of the three families. Only the tree ladder `{1, 3, 10, 30}` reaches both ends.
The corrected remedies are `λ → {0.1, 0.3, 1.0, 3.0}` and
`k → {5, 20, 100, 400}`; §6b states the reasoning. The trigger itself, the
one-extension-only discipline, and the re-select-every-arm rule are unchanged
and were all declared before any fit.

**Second defect: the trigger does not name every axis that can hit an edge.**
A1's selection sits at `depth=8` of `{4,6,8}` and `learning_rate=0.06` of
`{0.03,0.06}`, neither of which the trigger lists. `depth` has three values so
it *could* have been interior — this is a genuine edge, not §6a's
unextendable-by-construction case — and the argument for extending applies
identically. No remedy was declared for it in advance, so none is applied, and
the selection is reported as sitting at an undeclared edge rather than
retrofitted with a rule.

The same rule applies symmetrically at the strongest-regularisation boundary,
where a selection means the parsimony rule wanted more shrinkage than the grid
offered.

This mirrors the trigger the Phase D session declared for its own ridge ladder,
so the two phases handle boundary selections the same way.

"Most regularised" is defined per family in §4.

**Same-geometry ablations hold hyperparameters common.** Where two arms differ
only in an objective, a sampling scheme or a target parameterisation — not in
feature geometry or capacity — they are both run at one common hyperparameter
value chosen on the pooled inner ladder, because letting them sit at different
values adds a second difference and the gap stops isolating the intended one.
Where two arms are different learner families, each gets its **own
equal-budget** search: tuning parity in this project means equal budget, not
transplanted values, and a transplanted ladder has already cost a real result
here once.

## 4. Arms, grids and budgets

### A0 — reference reproduction and nested out-of-fold predictions

Not a new model. Refits the peptide + pseudosequence sequence reference
(`seq_ensemble_pep_pseudo`) under the §1 nested protocol and emits per-row
out-of-fold predictions with fold provenance. The published reference numbers
(validation median per-allele Spearman 0.6931, test 0.7064) are *not* reproduced
here — those are frozen-split numbers and Phase A does not score those splits.
What must reconcile is the machinery: row counts, fold membership, ensemble
size, seed protocol and metric code.

Member recipe, held identical to the deployed reference: `MLPConfig(hidden=(256,
64), lr=1e-3, batch_size=256, max_epochs=300, patience=15)` with `l2=1e-5` on
the one-hot encoding and `l2=1e-3` on BLOSUM — the per-encoding values selected
at stage 2 and never re-tuned — seeds `(0, 1, 2)`, and each member early-stopping
on its own inner fold.

**The existing OOF prediction files cannot be reused.**
`preds/seq_ensemble_pep_pseudo_oof.csv` and `preds/esm_ensemble_oof.csv` were
built on `stage6.nested_folds` (peptide folds, seed 20261004, mutant families
deliberately split apart). They are correct for the stage 6 mutant-ranking
question and wrong for a cluster-held-out assessment. A0 regenerates on cluster
outer folds.

**Output contract** (Phase B and D consume these, so it is fixed here):

| File | Columns | Rows |
|---|---|---|
| `phase_a_outer_folds.csv` | `pair_id, outer_fold` | 19,716 |
| `phase_a_fold_assignments.csv` | `pair_id, outer_fold, inner_fold` | 98,580 |
| `phase_a_seq_oof_cluster.csv` | `pair_id, outer_fold, role, y_pred` | 177,444 |
| `phase_a_a0_runs.csv` | per-member: outer_fold, inner_fold, encoding, seed, n_fit_rows, best_epoch, **dev_mse**, fit_seconds | 750 |
| `phase_a_a0_provenance.json` | the full member → (outer, inner, encoding, seed) map, thread count, package versions, input digests | — |

In `phase_a_fold_assignments.csv`, `inner_fold == -1` marks the rows that are
*outer-held* in that outer fold; the other rows carry the inner fold they
occupied inside that outer fitting partition. Every `pair_id` appears five
times, once per outer fold.

`role ∈ {outer_held, inner_oof_6, inner_oof_30}`:

- `outer_held` — 19,716 rows, each `pair_id` exactly once. The 30-member mean
  for rows in outer fold *f*, from members fitted on the other four outer folds.
- `inner_oof_6` — 78,864 rows, each `pair_id` exactly four times. The mean of
  the 6 members (2 encodings × 3 seeds) that early-stopped on the inner fold
  holding that row. Free: these are the same members' own held-out-inner-fold
  predictions.
- `inner_oof_30` — 78,864 rows, each `pair_id` exactly four times. A full
  30-member mean for each inner fold, requiring a third nesting level (five
  sub-folds of the inner fitting partition × 2 encodings × 3 seeds).

### Why `inner_oof_30` is emitted, at 5× the fits

A consumer that fits anything on inner out-of-fold residuals — Phase D's
structural residual model, and **A5's own blend weights** — and then applies it
to `outer_held` predictions would be fitting on a 6-member error distribution
and applying to a 30-member one. Ensemble arity is worth roughly +0.07 to +0.09
median Spearman here, so that gap is larger than most effects Phase A is
looking for; it inflates the fitted correction's scale roughly uniformly, which
a ridge partly absorbs into shrinkage but not cleanly. This is a confound in
A5, not only a courtesy to Phase D.
**[CORRECTED 4 Oct: the +0.07 to +0.09 figure is single-network-versus-ensemble
and does not describe a change in ensemble size. Measured, 12 members against
30 is +0.0156. The confound is real — and the measured residual-scale ratios
later showed the remedy made it *worse*, not better (see "Measured — and it
does not support the reasoning above") — but the magnitude quoted here was
roughly five times too large and that overstatement is part of why the remedy
looked worth 5× the networks.]**

The cost is 150 → **900 networks**: 5 outer folds × 30 members = 150 at level 2
(which yields both `outer_held` and `inner_oof_6`), plus 5 outer × 5 inner × 30
= 750 at level 3 (`inner_oof_30`). An earlier draft of this section said "750",
which was the level-3 count mistaken for the total. At the deployed ensemble's
measured rate of about 30 members per minute that is roughly half an hour of one
CPU core. At that price the confound is not worth carrying.

**What this does not fix.** `inner_oof_30` members are fitted on about 10,094
rows against `outer_held` members' 12,618, because each extra nesting level
removes another fifth. So a large, systematic *arity* mismatch is converted
into a small *training-size* mismatch, which is the irreducible remainder of
nested cross-validation and is reported as a limitation rather than claimed
away. `inner_oof_6` is shipped alongside precisely so the ratio between the two
inner residual scales can be measured and reported instead of argued about.

### Measured — and it does not support the reasoning above

That ratio has now been measured, paired per row, and **the arity correction
overshoots**:

| role | RMSE | ratio to `outer_held` | median-of-medians rho |
|---|---:|---:|---:|
| `outer_held` | 0.6678 | 1.000 | 0.6493 |
| `inner_oof_6` | 0.6805 | **1.019** | 0.6369 |
| `inner_oof_30` | 0.6904 | **1.034** | 0.6277 |

`inner_oof_30` is *further* from `outer_held` than `inner_oof_6` is, on both
residual scale and ranking. The 6→30 arity gain is smaller than the
12,618→10,095 training-size penalty a 20% row cut imposes. On this measurement
alone, `inner_oof_6` is the better-matched stacking input — the opposite of the
argument that justified building `inner_oof_30` at 5× the networks.

**The early-stopping-optimism objection is real but is swamped.**
`inner_oof_6` predictions *are* mildly optimistic by construction, since those 6
members chose their stopping epoch on the rows they predict, while
`inner_oof_30` is clean of that. But measured against `outer_held` on measured
rows only, **both inner roles are pessimistic — they sit on the same side of the
target and do not bracket it**:

| role | RMSE | vs `outer_held` |
|---|---:|---:|
| `outer_held` | 0.6944 | target |
| `inner_oof_6` | 0.7086 | **+2.0%** |
| `inner_oof_30` | 0.7176 | **+3.3%** |

The fit-size effect dominates: an inner prediction comes from an ensemble fitted
on 10,094 or 12,618 rows against `outer_held`'s 15,773, and less training data
means larger residuals. So the two estimators bound the out-of-sample scale from
**one side only**, and `inner_oof_6` is the tighter bound. Any consumer
estimating a residual scale from these should say "upper bound", not "estimate",
and must not claim the two bracket the truth.

**Compute this un-averaged.** Each `pair_id` has four inner replicates, one per
outer fold. Averaging them before taking the RMSE gives 0.6845 for
`inner_oof_6` — *below* `outer_held`, reversing the sign and fabricating the
appearance of a bracket. That average is an ensemble-of-ensembles no stacker
ever sees: within one outer fold a stacking model gets exactly one inner
prediction per row. Score every row as the stacker sees it.

**Consequence for A5:** the arity argument is **not settled**, and this document
no longer presents it as settled. Both roles are shipped. A5 selects its
stacking input on a **paired blend-outcome comparison**, not on either scale
proxy, and reports the choice with the table above. The 5× network cost bought a
second input and a measurement that complicated the question rather than
answering it — which is a worse outcome than predicted and is recorded as such.

**Next-step criterion:** predictions, metrics and fold provenance reconcile;
out-of-fold coverage is exactly the training split; the row-count identities
above hold exactly; and no row is scored by a model that saw it.

### A1 — boosted-tree regression

Learner family test. CatBoost, `RMSE` objective, on the same
position-preserving peptide + pseudosequence encodings the reference uses
(one-hot / BLOSUM positional). No allele ID feature in this pan-allele arm;
allele-specific models belong to a separately labelled seen-allele analysis.

**Grid — 12 configurations** (the §3.3 ceiling), screened at 2 seeds × 3 inner
folds:

| Axis | Values |
|---|---|
| `depth` | 4, 6, 8 |
| `learning_rate` | 0.03, 0.06 |
| `l2_leaf_reg` | 3, 10 |

Held fixed: `iterations = 2000` with early stopping on the inner fold
(`od_type="Iter"`, `od_wait=100`), `bootstrap_type="Bernoulli"`,
`subsample=0.8`, `rsm=1.0`, `random_strength=1.0`, `border_count=128`, single
thread count recorded per run.

"Most regularised" for the parsimony rule, in priority order: larger
`l2_leaf_reg`, then smaller `depth`, then smaller `learning_rate`.

A categorical residue-per-position encoding is a **separate optional ablation**,
run only if A1 is promoted, and is not part of this grid. Arbitrary numeric
amino-acid IDs are never used as ordered measurements.

**Next-step criterion:** improves outer-fold median Spearman against the A0
reference, *or* adds complementary signal that improves the A5 blend when its
weights are fitted on inner predictions only.

**Cost reporting:** 30 boosted fits do not imply the same capacity as 30 MLPs.
A1 reports tuning effort, fit wall time, inference time and model size next to
accuracy, and no tree-count cap is given the meaning of architectural fairness.

### A2 — allele-centred targets

Target parameterisation test, same MLP architecture as A0. Per fitting fold,
compute each allele's mean of `y_log1p` **inside that fitting fold only**, shrink
it toward the fitting-fold global mean, subtract for training, and add back on
the log1p scale for prediction. Unseen alleles fall back to the shared estimate.

Shrinkage is James–Stein style with a single prespecified grid:

> `shrunk_mean(a) = (n_a * mean_a + k * global_mean) / (n_a + k)`,
> with **k ∈ {1, 5, 20, 100}** chosen on inner folds.

Same-geometry ablation against A0, so the MLP hyperparameters are held common at
the A0 reference values and only `k` varies.

Adding back an allele mean cannot by itself change within-allele ranks, and the
primary metric is within-allele. Any benefit must therefore come from changed
*training* — this arm tests training efficiency, not a guaranteed removal of
harmful variance. If A2 moves the primary metric, the write-up must identify the
training-side mechanism or report the result as unexplained.

**Next-step criterion:** consistent outer-fold gain across folds and seeds, with
no validation-derived or outer-derived means anywhere.

### A3 — objective versus sampling

Four cells, same MLP architecture, hyperparameters held common (same-geometry):

| Cell | Loss | Sampling |
|---|---|---|
| A3-a | MSE | natural |
| A3-b | MSE | allele-balanced |
| A3-c | MSE + pairwise ranking | natural |
| A3-d | MSE + pairwise ranking | allele-balanced |

This 2×2 separates the loss change from the sampling change, which the plan
requires be tested separately.

Pairwise ranking rules, fixed here:

- Pairs are formed **only within the same allele**.
- **Tied targets are skipped**, including floor–floor pairs. No ordering is
  claimed for a pair whose assay intervals overlap.
- Pair contributions are **normalised per allele**, so an allele with many
  measurements does not dominate quadratically.
- MSE is retained as an auxiliary term, weight `λ ∈ {0.1, 0.3, 1.0}` on the
  ranking term, chosen on inner folds, common across the two ranking cells.

**Stopping rule is reported as its own axis, not folded into the loss.** Every
cell is run with MSE-based early stopping and with ranking-based early stopping,
and the two are compared explicitly. Stage 7's censored-loss result is the
precedent: there, the apparent ranking recovery traced to the stopping rule
rather than the gradient, and that confound is not repeated here.

**Next-step criterion:** a ranking gain that survives the paired controls and
fold/seed variation, with the stopping rule held fixed.

### A4 — boosted-tree ranker

CatBoost groupwise ranking with **allele as the ranking group**, on the A1
features. This is a learner-family × objective change relative to A3, so it gets
its own equal-budget search rather than A1's selected values: **12
configurations**, 2 seeds × 3 inner folds, same three axes and ranges as §A1,
with the objective fixed per sub-arm.

Two objectives are declared: `YetiRank` and `PairLogit`. Groupwise ranking
objectives are not identical to median Spearman — they are surrogates, and the
write-up says so rather than treating the objective as the metric. The same
within-allele, tie-skipping, per-allele-normalisation rules from §A3 apply to
pair construction.

A4's **median per-allele Spearman is scored on its raw ranker output**, with no
calibration map in the path — see §2. A map is fitted inside the training
partition for the magnitude metrics only, and its measured effect on the ranking
metric is reported rather than assumed to be zero. A4 is the arm most exposed to
this, because a groupwise ranker's output scale is arbitrary and the temptation
to calibrate before scoring is strongest exactly where it would do most damage.

**Next-step criterion:** beats the matched A1 tree regressor on outer-fold
ranking, or improves the A5 blend.

### A5 — blends

Simple blends with **non-negative weights summing to one**, fitted inside the
outer training partition using inner out-of-fold predictions only:

- A0 + A1 (reference + tree)
- A0 + ESM (existing cached embeddings arm)
- A0 + A1 + ESM (small combined blend)

Finalists receive a declared bagging budget comparable to the reference, with
diminishing returns checked inside training only.

Component-removal ablations are run on any blend that is promoted. Phase A does
**not** begin by stacking every change: each change is first shown to help
independently.

**Next-step criterion:** improves the outer assessment with weights fitted only
on inner predictions.

## 5. Compute and cost

Phase A is CPU-only. Execution is on Modal CPU containers, one container per
(arm, config, seed, fold) unit, so the screens run in parallel.

Rates from [`reports/ectodomain_rates.json`](../ectodomain_rates.json):
**$0.04730 per CPU-core-hour** and **$0.00800 per GiB-hour**. The forecast is
computed and recorded in the run manifest **before** launch, and actuals are
logged in [`reports/compute_ledger.csv`](../compute_ledger.csv). Per CLAUDE.md
no batch job runs without a passing end-to-end smoke on 3–5 units first, and the
Modal app carries its own `::smoke` entrypoint for that.

Floating-point note for reproducibility: BLAS thread count changes float32
results, moving a single network by roughly 0.02 median Spearman at fixed seed.
Thread count is pinned and recorded per run; conclusions rest on ensembles,
which are stable to it.

## 6. What Phase A will not claim

- No test-split score, and no new validation-split score.
- No claim that a calibration improvement counts toward the ranking bar.
- No effect size attributed to an uninformative or width-matched control block;
  such a control supports a sign test only.
- No budgeting around an assumed recoverable headroom. Concordance between
  distinct alleles is not replicate reliability, and the gap between our median
  and the best cross-allele correlation does not establish attainable
  improvement. Replicates would be needed.

## 6a. Amendments forced by the arms implementation

Recorded on 4 October 2026, after the arms layer was built and timed but
**before** any screen was scored. Each is a defect in §4 as written.

**A4/YetiRank is withdrawn from the screen.** Two reasons, both protocol
failures rather than results:

1. **It cannot honour §A3's pair rules.** CatBoost generates YetiRank's pairs
   internally from labels inside each group, and that generation is not
   overridable. So tie-skipping — including the floor–floor pairs §A3 explicitly
   forbids claiming an ordering for — and per-allele normalisation cannot be
   applied. PairLogit accepts explicit pairs (812,462 on one fitting partition)
   and *can* honour them.
2. **Its stopping surface is NDCG, which this document never declared.**
   CatBoost resolves the eval metric from the objective, giving `RMSE` for A1,
   `PairLogit` for A4/PairLogit and **`NDCG`** for A4/YetiRank. NDCG is
   position-discounted and top-heavy; it is not the panel median. YetiRank stops
   at 377–440 iterations and scores 0.17–0.41 against PairLogit's 0.52–0.55, so
   the gap may be the *stopping surface* rather than the gradient — precisely
   the stage 7 confound A3 exists to avoid. Comparing an NDCG-stopped arm with
   an RMSE-stopped one would reproduce that error inside the arm built to
   prevent it.

The resolved `eval_metric` is now recorded on every tree row so this is visible
rather than inherited silently.

**A1 was not early-stopped, and the write-up must not say it was.** With
`iterations=2000` and `od_wait=100`, **11 of 12 A1 configs ran all 2,000
iterations** with eval RMSE still falling at the cap. `iterations=2000` is
therefore A1's binding budget and its cost is deterministic. The grid is
unchanged, since this document is binding, but the claim "early stopping on the
inner fold" is false for A1 as configured.

**Parsimony for A3's λ is a declared interpretation, not a derivation.** λ is an
objective weight, not a capacity control, so "most regularised" has no intrinsic
meaning. It is defined as the **smallest departure from the reference MSE
objective** — smaller λ. Consequence, stated rather than discovered later: a
flat λ ladder always selects 0.1 and is always flagged at a boundary.

**`interior` is structurally unreachable for the tree grids.**
`learning_rate` and `l2_leaf_reg` carry two values each, so any selection is at
an endpoint on both axes by construction. Selection reports `boundary_axes`,
naming which axes are unextendable-by-construction versus genuinely at an edge,
so `interior=False` is not misread as a verdict on the grid.

**Allele-balanced sampling is a weighted loss, not resampling.**
`MLPRegressor` normalises by row count, so all-ones weights reproduce the
unweighted path bit-identically, whereas resampling would introduce draw noise
as a second difference between A3's cells. The ranking term is already
per-allele normalised, so the sampling axis reaches only the MSE term — which is
the separation §A3 asks for, and is stated here rather than left implicit.

**Encoding is a unit axis, not a grid axis.** §A1 names both encodings and caps
the grid at 12. Making encoding a grid axis would give 24 and breach §3.3;
dropping an encoding would break ensemble parity. It is therefore a unit axis
alongside seed, with `l2` resolved per encoding from the stage-2 values.

**Screening runs inside every outer fitting partition.** §3.3's "two paired
seeds and three inner folds" does not say per outer fold. It is 5 outer × 3
inner × 2 seeds per config, because selecting on one outer partition and reusing
it leaks: the rows held out in outer fold 1 sit inside outer fold 0's fitting
partition, so a selection made there has already seen them. This makes the
screen 5× larger and the 5× is load-bearing, not waste.

## 6b. The boundary trigger fired, and its declared remedy is wrong for the case that occurred

Recorded 4 October 2026, immediately after the A1–A3 screen completed and
before any finalist was assessed.

**The trigger fired on two arms.** A1's parsimonious selection landed on
`depth=8` *and* `l2_leaf_reg=10`, both grid maxima; A3's landed on `λ=1.0`, also
a grid edge. §3's trigger is therefore active and the screen's selections are
flagged rather than accepted.

**But §3's declared remedy for λ points the wrong way.** §3 lists `λ == 1.0` as
a *weakest-regularisation* boundary and prescribes extending to
`{0.03, 0.1, 0.3, 1.0}` — downward. That is incoherent for a selection sitting
at the **top** of the λ ladder: adding smaller values cannot change a selection
that chose the largest one. The underlying reasoning was sound and is worth
restating, because it is what makes the direction confusing: §6a defines
parsimony for λ as the *smallest departure from the reference MSE objective*, so
a selection at `λ=1.0` means the rule picked the least parsimonious rung in
spite of that preference — evidence the ranking term is helping enough to
overcome it, and that the ladder may be truncating the benefit. The remedy that
follows is to extend **upward**.

**Corrected remedy, declared now:** a λ selection at the top of the ladder
extends it **once** to `{0.1, 0.3, 1.0, 3.0}`; all A3 ranking cells are
re-selected on the extended ladder, both ladders are reported, and no second
extension is available. The A1 tree selection at `l2_leaf_reg=10` and `depth=8`
takes §3's symmetric strong-end remedy as written, extending `l2_leaf_reg` to
`{1, 3, 10, 30}` and `depth` to `{4, 6, 8, 10}`.

**This is a defect in §3, not a result-driven change**, and the distinction
matters: the trigger and the one-extension-only discipline were declared before
any fit, and only the *direction* of one remedy was wrong. Recording it rather
than silently extending the other way, because a reader checking whether the
declared rule was followed should be able to see that it was not followable as
written.

## 6c. A4 promoted — boosted trees with the ranking objective

Declared 5 October 2026, **before any A4 unit has been fitted or scored**, after
the A1–A3 screen completed. Plan §9 made A4 conditional on A1 or A3 looking
promising; both returned real sub-bar gains against the matched control, so the
condition is met.

**Objective: PairLogit only. YetiRank stays withdrawn** for the two reasons in
§6a — it generates its pairs internally so it cannot honour the declared
tie-skipping, and its stopping metric silently resolves to NDCG.

**Design: a same-geometry ablation, not a new grid search.** The question is
whether adding the ranking objective to the boosted tree helps, so A4 runs at
**A1's own selected hyperparameters** (`depth=8, l2_leaf_reg=10,
learning_rate=0.06`) with only the objective changed from `RMSE` to
`PairLogit`, plus the same at A1's argmax (`l2_leaf_reg=3`) so the conclusion
does not hinge on the parsimony choice. That is §3's same-geometry rule — the
one A3 followed when it held the MLP hyperparameters common across its 2×2 —
and it costs **120 units rather than 1,440**.

**Declared limitation, stated now rather than discovered later:** RMSE-optimal
tree hyperparameters need not be PairLogit-optimal. This arm therefore answers
*"does the ranking objective help at A1's geometry"*, **not** *"what is the best
achievable boosted-tree ranker"*. The second question needs its own
equal-budget 12-configuration search under §3's tuning-parity rule and is
explicitly out of scope here. A null result on this arm does not license
"boosted-tree rankers do not work".

**Pair construction**, unchanged from §A3: within-allele pairs only; tied
targets skipped including floor–floor pairs; per-allele normalisation, with the
weights rescaled to mean 1 because CatBoost's pair weights are absolute and a
`1/(A·P_a)` scheme summing to 1 over ~800k pairs would shrink every leaf update
to nothing. `pairs_explicit` and `n_pairs` recorded per row, and the resolved
`eval_metric` recorded so no stopping surface is inherited silently.

**Scoring:** median per-allele Spearman on the **raw** ranker output, with no
calibration map in the path (§2). A map is fitted inside the training partition
for magnitude metrics only, and its measured effect on ranking is reported.

**The pre-declared question, and what each answer means.** A4 is compared
against three references: the matched 12-member control, A0, and — the decisive
one — **A1 itself**.

- **A4 > A1 conclusively:** the ranking objective adds on top of the learner
  family. The two levers are complementary.
- **A4 indistinguishable from A1 *and* from A3-c:** the three may be reaching a
  common ceiling by different routes — but see the caution below before reading
  it that way.
- **A4 < A1:** the explicit-pair objective costs something at this geometry,
  most plausibly through the stopping surface, and the §A3 stopping-rule
  retraction is the precedent for suspecting that first.

**No outcome is declared as expected, and an earlier draft of this section was
wrong to do so.** It argued that because A1 and A3-c are statistically
indistinguishable (−0.0031 [−0.0150, +0.0240]), they must be capturing the same
signal, so A4 should land in the same place. That inference does not hold, and
**this study contains its own counterexample**: A0 and A1 differ by only +0.0072
(inconclusive, nearly indistinguishable) yet the A0+A1 blend scores **+0.0128
[+0.0016, +0.0246], real** — two arms with indistinguishable aggregate scores
making complementary errors. Equal effect size is not equal mechanism.

The two interventions are also **orthogonal**, not alternative routes to one
place: A1 changes the learner family with the objective held at squared error,
A3-c changes the objective with the learner held at the reference MLP. Nothing
about their having similar magnitudes bears on whether they compose.

**If anything the mechanistic prior favours a gain.** The primary metric is
within-allele ranking. An MSE-trained MLP already produces a smooth global
ordering, so an auxiliary ranking term has less to repair; a tree ensemble
produces piecewise-constant predictions with real tie structure and no incentive
to order *within* a group. `PairLogit` with `group_id = allele` optimises
exactly the metric's grouping, which is a closer match to the target than A3's
ranking term added to MSE. That is a reason to think trees have more to gain
from the objective than the MLP did — stated as a hypothesis, not a forecast.

**Interpretation caveat carried from §6b and the platform work:** the run-to-run
reproducibility floor on this task is **0.0043** on median-of-medians, and the
paired bootstrap treats predictions as fixed so its intervals do not cover it.
Any A4 effect smaller than about 0.004 is not distinguishable from a re-run of
the same arm. The floor is additionally **unmeasured for CatBoost arms**, which
do not use the same BLAS paths as the MLP arms.

**Cost: $2.83 with margin**, 120 units, ~36 min wall at 30 containers × 2
cores / 4 GiB, chunk 4 (a 35-min worst-case chunk, inside the 60-min function
timeout — chunk 6 would not be). Ledgered as `FORECAST_priced_before_launch`.

Per-unit time is **526 s**, derived from the smoke's directly measured
`chunk_wall_s` of 552.4 s for 5 units, less 4 cheap units at the 6.5 s median.
Locally the same arm ran 380 s, so Modal is 1.39× slower here — it fits on
812,462 explicit pairs, against A1's 10.8 s local.

**The smoke's own `seconds_per_unit` field reports 520.9 and was not used.**
That figure implies 2,604 s of work in a chunk whose measured wall was 552.4 s,
sequentially in one container — inconsistent by 4.7×, so whatever it aggregates
it is not a mean of unit times. Had it been fed to the forecast it would have
inflated the price 4.7-fold. Recorded here because the project's rule is to
price from a measured quantity, and this is a case where the field *labelled*
as the measurement was the wrong one to trust and the raw wall clock was right.

## 7. Logged extensions

Each entry records a date, what changed, and why. No arm had been fitted when
either amendment below was made.

**4 October 2026 — A0 emits `inner_oof_30`; A0 budget 150 → 750 networks.**
Raised by the Phase D session. Fitting a residual or blend model on
`inner_oof_6` and applying it to `outer_held` mismatches ensemble arity (6
members against 30), which inflates the fitted correction's scale. This affects
A5's blend weights as much as Phase D's residual model, so it is a Phase A
confound. 750 networks is about 25 minutes of one core and $0, so the confound
is not worth carrying. `inner_oof_6` is still emitted so the residual-scale
ratio can be measured. Residual training-size mismatch (10,094 vs 12,618 fit
rows) is documented as the irreducible remainder, not claimed away. §A0's output
contract was rewritten; no grid, metric or selection rule changed.

**4 October 2026 — A4 deferred; first batch is A1–A3 only.** Not a change to the
plan, but an application of it: [FURTHER_RESEARCH_PLAN.md](../archive/plans/further-research-2026-10-04.md)
§9 says "First batch: A0–A3, **then** A4/A5 for promising candidates". Measured
per-unit times make that ordering decisive rather than merely tidy.
A4/PairLogit costs **380 s per unit** against 2.6–11.2 s for every other arm,
because it fits on 812,462 explicit pairs. Priced at the metered rates:

| arm | units | s/unit | cost with margin |
|---|---:|---:|---:|
| A1 | 720 | 10.8 | $0.34 |
| A2 | 240 | 2.6 | $0.03 |
| A3 | 960 | 5.1 | $0.22 |
| A4/YetiRank | 720 | 11.2 | $0.35 (withdrawn, §6a) |
| A4/PairLogit | 720 | 380.0 | **$12.03** |

A1+A2+A3 is $0.59 — **4.5%** of the full screen — and A4/PairLogit alone is
**95.5%**. So A1–A3 runs now and A4/PairLogit is commissioned only if A1 or A3
earns it, which is exactly the condition §9 already set. The 150× cost gap
between the two A4 objectives traces directly to the explicit-pair asymmetry in
§6a.

**4 October 2026 — floor event defined as `thalf_hours == 0`, not `< 0.1 h`.**
Raised while reviewing the Phase B declaration; **rationale corrected the same
day after Phase B challenged it**, and the corrected reason is the one that
stands.

The change: the calibration target's floor event is `thalf_hours == 0`. The
reason: the recorded assay-floor code *is* an exact 0, that is the quantity
EVALUATION.md's "20.2% of labels sit at the assay floor" names (20.1626%
reproduces to the digit), and defining the event on the recorded code rather
than on a threshold means the calibration target needs no assay constant at
all. This is a definitional fix; no arm, grid or selection rule changes.

**Two arguments originally given here were wrong and are withdrawn.** They are
recorded rather than deleted, because a prespecification amendment justified by
a hazard that does not exist is the same class of error as the budget-mismatched
stage 7 §6.4 table that Phase B exists to correct.

1. *Withdrawn: "a 0.1 h threshold swings the floor class by 443 rows / 1.57
   percentage points."* That is the gap between `<= 0.1` and `< 0.1`. Nobody
   proposed `<=`, and stage 7 §3 declares the opposite. The two definitions
   actually on the table, `== 0` and `< 0.1`, differ by **exactly one row** —
   `pair_id` 1189, `HLA-A*24:02` / `WYGMEIRPL`, 0.05 h, `cluster_id` 4704. Train
   3,977 against 3,978; val and test identical. The original claim overstated
   the effect by a factor of 443.
2. *Withdrawn: "`c = log1p(0.1)` is a float equality in disguise."* It is not on
   this code path. `pepstab.censored.censored_mask` compares **hours** —
   `return hours < floor_hours` (`pepstab/censored.py:89`) — and the stored 0.1
   is the exact double 0.1, so `0.1 < 0.1` is deterministically False on every
   platform. `log1p(0.1)` appears only inside `Phi((c - mu)/sigma)`, where `mu`
   is a prediction, and is never compared against a label.

**Consequence, flagged by Phase B and carried deliberately.** The calibration
event (`== 0`) and the censored loss's own masking rule (`< 0.1 h`, unchanged
from stage 7) now disagree on that one row: the loss censors it, the calibration
target calls it measured. The loss mask stays as it is on purpose — altering it
would make the censored arms non-comparable to the stage 7 result Phase B exists
to resolve — and the one-row disagreement is documented with a sensitivity check
rather than left as two silent definitions of "floor" in one report.
