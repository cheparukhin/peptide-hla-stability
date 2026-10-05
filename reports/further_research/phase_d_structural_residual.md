# Phase D — reuse structures before expanding compute

Date: 4 October 2026. Phase D of
[FURTHER_RESEARCH_PLAN.md](../FURTHER_RESEARCH_PLAN.md) §7.
Protocol frozen before any arm was fitted:
[`phase_d_protocol.md`](phase_d_protocol.md).
Shared cross-phase protocol: [`phase_a_prespecification.md`](phase_a_prespecification.md);
namespace index: [`README.md`](README.md).

**No new folding and no GPU spend.** Every structural number in this report
comes off disk from the stage 4c5 production extraction. The phase exists to
answer whether features already paid for are worth anything before more are
commissioned.

Scope: the training split only. Validation is not scored by this phase and the
test split is not touched.

---

## 1. Verdict

**All three questions are answered, and all three answers are negative for
further spending.**

| Question | Answer |
|---|---|
| Does a strongly shrunk structural residual correction improve cluster-held-out ranking? | **No, and a worthwhile gain is ruled out.** Best arm +0.0084 median per-allele Spearman, 95% CI [−0.0055, +0.0124] — an upper bound a quarter of the project's 0.05 bar. Both anchor-focused arms selected **no correction in 5 of 5 folds**. |
| Does Boltz-2 already expose peptide–HLA pair ipTM in both directions? | **Yes**, measured, in all 28,166 production pairs, and the two directions are not redundant. No folding is justified to obtain it. |
| Are more predicted poses worth generating for feature stability? | **No.** A single seed already recovers a median **99.1%** of each feature's usable within-allele signal. More poses buy ~1% relative on a typical feature. |

This closes the plan's §7 gate in the negative on existing data. The result is a
reasonably precise **exclusion**, not a failure to detect: the structural
features already on disk do not carry ranking signal that a 30-member sequence
ensemble has not already extracted from sequence alone.

Two things the result does *not* say. It does not say structure is irrelevant
to peptide–HLA stability — it says these particular summary features, added as
a shrunk linear correction on top of this particular sequence arm, add nothing
measurable. And it does not close the plan's §7 follow-up: an interface
pair-embedding pilot remains a distinct proposal, now with the additional
burden of explaining what it would capture that 109 geometry and confidence
summaries do not.

### 1.1 The strongest provenance check in the study

Phase D's D0 arm is Phase A's A0 sequence ensemble, consumed through an
independently written harness. The two agree **element-wise to four decimals
on all five outer folds**:

| outer fold | 0 | 1 | 2 | 3 | 4 | five-fold median of medians |
|---|---:|---:|---:|---:|---:|---:|
| Phase A's A0 | 0.6765 | 0.6281 | 0.6418 | 0.6493 | 0.6515 | 0.6493 |
| Phase D's D0 | 0.6765 | 0.6281 | 0.6418 | 0.6493 | 0.6515 | 0.6493 |

Two harnesses, written separately, reaching the same numbers on the same folds
retires a class of doubt about the fold machinery for every phase in this
study, not just this one. The outer-fold assignment itself was also verified
identical element-by-element between a local recomputation of
`scripts.baseline_ensemble.cv_folds` and Phase A's independently written
`phase_a_outer_folds.csv`, same `sha256` prefix `50fdca745c9e8e99` — which
closes the question of whether `assign_clusters` is deterministic.

**Fold spread is the number to keep in mind while reading everything below.**
D0 alone ranges 0.6281 to 0.6765 across the five outer folds — a span of
**0.048**, essentially the whole of the project's 0.05 worthwhile bar, arising
from fold assignment with the model held fixed. Any arm evaluated on a single
fold would be uninterpretable, and the paired peptide-cluster bootstrap is not
a formality here: it is the only thing separating an effect from fold
assignment.

## 2. Inventory: what is already on disk

### 2.1 Coverage is complete

`reports/stage4c5_features_production.csv`: **28,166 rows**, one per
`(allele, peptide)` pair, every one `status == "ok"`. 14,083 rows per Modal
profile, Boltz-2 only, **seed 0 and arm B only** — the production
configuration.

| Split | Rows | With a structure |
|---|---:|---:|
| train | 19,716 | 19,716 (100%) |
| val | 2,817 | 2,817 (100%) |
| test | 5,633 | 5,633 (100%) |

All 3,587 training peptide clusters and all 75 alleles are covered. This
matters for the design: stage 5's structural arm needed a declared **sequence
fallback** for pairs without a usable structure, and that fallback creates a
covered/uncovered asymmetry which has to be reasoned about. Phase D has no such
rows. The residual model fits on the full training split with no subset
selection of any kind.

Every peptide in the dataset is a **9-mer** (28,166 of 28,166), so the
positional features `_p1`…`_p9` are complete and the anchor arms below are
well defined without a length caveat.

### 2.2 Both directions of peptide–HLA pair ipTM are present

The plan records the claim that Boltz lacks a pair-specific ipTM as stale, and
asks for ablation rather than acquisition. Confirmed by measurement, not by
reading the code:

| Column | Non-null | mean | sd |
|---|---:|---:|---:|
| `iptm_pair_hla_chain_rows_peptide_cols` | 100% | 0.99257 | 0.00278 |
| `iptm_pair_peptide_rows_hla_chain_cols` | 100% | 0.94947 | 0.01769 |

Plus the four b2m-involving directions, also 100% non-null.
`CONFIDENCE_PREFIXES` in `scripts/stage5_structural_arm.py` includes
`iptm_pair_`, so these columns were already entering the structural arm's
confidence block.

**They are not redundant with each other.** Spearman between the two
peptide–HLA directions is **0.726**, with a maximum absolute difference of
**0.155** across the cohort. This is consistent with stage 4c5 §1.3's finding
that PAE is asymmetric on all 90 pilot folds: the two directions ask different
questions — how well the HLA's tokens are placed relative to the peptide's,
and the converse — and keeping both is the right call.

**Consequence: no structures are commissioned to obtain this feature.** It is
the cheapest "no" available in the further-research plan, and it is now
evidenced rather than asserted.

### 2.3 One caveat on reading an ipTM coefficient

`iptm_pair_hla_chain_rows_peptide_cols` sits at mean 0.9926 with sd 0.0028 —
close to its ceiling with very little spread. This bites specifically for a
ridge: standardisation maps every column to unit variance, so a near-constant
column's noise arrives at the penalty on exactly the same footing as a column
carrying real spread, and the shrinkage cannot tell them apart.

Reported as a **diagnostic, never as a filter** — dropping columns on it after
seeing outcomes would be post-hoc feature selection. The full table is
[`phase_d_feature_variance_partition.csv`](phase_d_feature_variance_partition.csv).

What the metric can actually use is *within-allele* spread, because median
per-allele Spearman ranks peptides inside an allele and is blind to any
between-allele offset. Both ipTM directions survive that test:

| Column | sd (total) | sd (within allele) | within-allele share of variance |
|---|---:|---:|---:|
| `iptm_pair_hla_chain_rows_peptide_cols` | 0.00278 | 0.00224 | 0.647 |
| `iptm_pair_peptide_rows_hla_chain_cols` | 0.01769 | 0.01479 | 0.699 |

So the near-constant direction is near-constant in *absolute* terms but not
degenerate within allele. The orchestration session's expectation that the
higher-spread direction is the more plausible signal carrier is reasonable on
these numbers, but it is a prior, not a result — only the ablation can say.

Across all 109 feature columns the within-allele share runs 0.32 to 0.96
(median 0.82 for geometry, 0.69 for confidence). The weakest are PAE blocks
involving the alpha3 domain (0.32), which is expected: alpha3 is largely
allele-determined and borrowed in some constructs, so most of its spread is
between alleles and invisible to the primary metric.

**One column is exactly constant**: `global_ligand_iptm` is identically 0 in
all 28,166 rows, because there is no ligand. It is retained rather than
dropped, so the feature contract stays identical to stage 5's, and it
contributes nothing to any fit.

---

## 3. Multi-pose reproducibility, from existing records only

### 3.1 What this is and is not

Across-seed spread in a predicted structure's features measures **prediction
ambiguity and the sampler's behaviour**. It is **not** a thermodynamic
ensemble, **not** a conformational population, and **not** an unbinding rate.
No energetic or kinetic quantity is inferred from it anywhere in this phase.
Genuine dynamics would be a separate and far more expensive study.

What across-seed spread *is* good for is **reliability**: how much of a
feature's usable spread survives being measured once. A feature measured with
reliability *R* correlates with any outcome at `sqrt(R)` times its true
correlation. That makes "sample sufficiency" an answerable question — how many
seeds must be averaged to recover a given fraction of the signal — without
making any claim about what the seeds physically represent.

### 3.2 Records used

No new poses. Three existing sources:

| Source | Content |
|---|---|
| `reports/stage4c5_pilot_features.csv` | the three-seed pilot: 2 models × 3 arms × 3 seeds × 5 complexes = 90 folds, full feature table |
| `reports/stage4c5_feature_variation.csv` | stage 4c5's own seed / arm / between-complex spread per feature |
| `reports/boltz_pose_check.csv`, `reports/esmfold_pose_check.csv` | crystal-structure agreement (RMSD against an experimental PDB) for the same 5 complexes — a different question from seed stability, and not mixed with it here |

Seed variance comes from the pilot (3 seeds, 5 complexes: 10 pooled degrees of
freedom per model/arm). Signal variance comes from the **production** cohort
(28,166 pairs, 28,091 within-allele degrees of freedom), which is a far better
estimate of between-peptide spread than 5 complexes can give. The transfer
assumption is stated plainly: **the pilot's 5 complexes are assumed to have
representative seed noise.** It cannot be checked without new poses, which
this phase does not generate.

### 3.3 Result: one seed is already sufficient

Boltz-2, arm B — the production configuration. 109 feature columns.

| Numerator | median sd ratio (signal / seed) | features with ratio < 1 | median 1-seed reliability | median 1-seed attenuation | median seeds for 99% of signal |
|---|---:|---:|---:|---:|---:|
| between-complex | 8.87 | **0 of 109** | 0.9875 | **0.9937** | 1 |
| **within-allele** | 7.28 | **0 of 109** | 0.9814 | **0.9906** | 1 |

The within-allele row is the one that bears on the primary metric. Reading it:
a single predicted pose recovers a median **99.1%** of each feature's
correlation with any outcome. The worst column is
`pep_min_dist_to_groove_p6` at 0.9485 — a 5% relative loss — and it needs 6
seeds to reach 99%. No column anywhere has seed noise exceeding its
within-allele signal.

**Robustness to the pilot's small degrees of freedom.** With 10 df, the
chi-square 95% upper bound on the seed variance is 3.08× the point estimate.
Pushing every seed variance to that bound:

| | median attenuation | worst attenuation | median seeds for 99% | max seeds for 99% | features with ratio < 1 |
|---|---:|---:|---:|---:|---:|
| point estimate | 0.9906 | 0.9485 | 1 | 6 | 0 |
| seed variance at its 95% upper bound | 0.9720 | 0.8627 | 3 | 17 | 0 |

Even in the pessimistic case a single pose keeps 97% of the median feature's
signal, and no feature becomes seed-dominated. **The verdict does not depend
on the uncertainty.**

### 3.4 Agreement with stage 4c5, and the one definitional axis

Stage 4c5 reports a median between/seed ratio of **5.05** for Boltz-2.
Recomputed here it is **5.95** for arm B. The difference is one documented
choice and not a discrepancy: stage 4c5 treats each fold as a unit, while the
calculation here averages the three seeds into a complex mean before measuring
between-complex spread, which removes seed noise from the numerator and raises
the ratio. Stage 4c5 §8.1 identifies exactly this axis as the bulk of a
similar gap in the ProteinMPNN control, and its own guidance is to **quote the
envelope, not a point estimate**. Both values sit comfortably inside it and
both return the same verdict.

### 3.5 Decision

**Do not generate more poses for feature stability.** The plan's gate requires
defining the feature, matched comparator, panel coverage and budget before
commissioning structures. On these numbers the *feature* side of that gate
already fails: there is no appreciable signal to recover. Buying a ~1% relative
improvement in a feature's correlation — against a project bar of +0.05 median
Spearman — is not a defensible use of GPU credit.

This does **not** say multi-pose features are worthless in principle. It says
that *seed-averaging to reduce measurement noise* is not worth paying for
here. A genuinely different question — whether the *dispersion* across poses
is itself a predictive feature, rather than noise to be averaged away — is not
answered by this analysis. It would need a prespecified multi-allele panel
drawn within training, separate cluster-held-out evaluation, and a single-pose
control matched on the same rows, exactly as the plan's §7 requires. With
production at one seed, that experiment has no existing records to run on and
is therefore a new-pose proposal subject to the full gate.

Run tables:
[`phase_d_pose_seed_variance.csv`](phase_d_pose_seed_variance.csv),
[`phase_d_pose_reliability.csv`](phase_d_pose_reliability.csv),
[`phase_d_feature_variance_partition.csv`](phase_d_feature_variance_partition.csv).
Code: `scripts/phase_d_pose_reproducibility.py`.

---

## 4. The residual screen: design and harness validation

### 4.1 Arms

Prespecified in [`phase_d_protocol.md`](phase_d_protocol.md) §3, before any
arm was fitted. Feature groups are **imported** from
`scripts/stage5_structural_arm.py` rather than redefined, so Phase D cannot
regroup a feature into whichever block it helps.

| ID | Arm | Columns |
|---|---|---:|
| D0 | `none` — the sequence prediction unmodified; the reference for every delta | 0 |
| D1 | `geometry` — contacts, distances, SASA, burial | 36 |
| D2 | `confidence` — pLDDT, PAE both directions, pTM/ipTM incl. both `iptm_pair_` peptide–HLA directions | 73 |
| D3 | `anchor_geometry` — P2/P9 positional geometry + aggregates | 15 |
| D3b | `anchor_geometry_p1p2p9` — the prespecified P1 extension | 18 |
| D4 | `geometry+confidence` | 109 |
| D3c | `central_geometry` — P3–P8 + aggregates (control) | 27 |
| DX | `geometry_shuffled` — D1's columns row-permuted inside the fitting partition (**sign test only**) | 36 |

D1 and D4 are the **full-feature controls** the plan requires: the
anchor-focused arms restrict which positions are resolved, they do not replace
the full block. Aggregate (non-positional) geometry is kept in every geometry
subset, so an anchor arm differs from D1 by position selection alone and not
by also losing the peptide's overall burial and contact count. D3c then asks
the direct question the plan raises — whether central peptide positions carry
anything on their own.

### 4.2 Model and selection

A ridge fitted on the fitting partition's **out-of-sample** sequence residuals
predicts those residuals from the standardised structural block; the corrected
prediction is `ŷ_seq + r̂`.

Three choices worth stating explicitly:

- **The residuals used for fitting are themselves out-of-sample.** Fitting on
  in-sample residuals would shrink the apparent correction toward zero and
  understate the arm.
- **Zero correction is an explicit model choice.** The ladder's top point is
  `alpha = inf`, which returns exactly zero correction, so "do not correct" is
  reachable by selection rather than assumed away. Zero initialisation
  preserves the baseline only *initially*; a fitted correction can still
  damage generalisation, which is why every arm is judged on held-out clusters.
- **All transforms are fitted inside the fitting partition** — column means
  and standard deviations for z-scoring, and the fitting-partition mean as the
  fill value for a non-finite cell (none occur, at 100% coverage).

Ladder: `alpha ∈ (1e1, 1e2, 1e3, 1e4, 1e5, 1e6, inf)`, identical across arms,
selected on inner cluster folds inside each outer fitting partition.
Selection uses a **1-SE rule toward stronger shrinkage**: among alphas within
one standard error of the best, take the largest. The ladder is expected to be
flat, an argmax on a flat ladder chases boundaries, and the parsimony
direction — toward *not* correcting — was declared before fitting.

### 4.3 The harness passes its own leak test

The thing most likely to go wrong in a residual screen is not the model, it is
the plumbing: a harness that accidentally fits on rows it later scores reports
a large, clean, entirely fake improvement. Four defences, all asserted at run
time rather than documented:

1. **Training split only.** Asserted on the frame.
2. **Outer folds are whole peptide clusters**, from
   `scripts.baseline_ensemble.cv_folds` imported rather than reimplemented,
   fingerprinted `50fdca745c9e8e99`, with cluster-disjointness between every
   pair of folds checked explicitly. The reconstruction was verified
   **identical element-by-element** to Phase A's independently written
   `phase_a_outer_folds.csv`, so the two phases rest on the same folds and
   `cv_folds` is confirmed deterministic.
3. **The sequence OOF contract is asserted, not trusted**: every training
   `pair_id` exactly once as `outer_held` and exactly four times as each inner
   role, all predictions finite, and the fitting role chosen by declared
   preference rather than guessed.
4. **The DX shuffled control.** Its columns are row-permuted inside the
   fitting partition under a fixed seed.

On the harness-validation run, **DX selected `alpha = inf` — exact no
correction — in all five outer folds**, for a delta of precisely +0.0000. That
is the selection rule correctly refusing to correct on a block carrying no
information, and it is the single most useful thing the control could have
done. Informative arms selected a finite alpha in 28 of 30 arm-folds.

DX's reading is **directional only**. A width-matched shuffled block is not a
capacity control: it breaks the honest relationship between features and
labels while leaving the ladder intact, so its *effect size* is not
interpretable and is never quoted as one. The only inference drawn is the sign
— had DX improved on held-out clusters, the harness would be leaking and no
other arm's number could be reported.

### 4.4 The residual-scale mismatch, measured rather than assumed

The correction is fitted on inner out-of-fold errors and applied to
outer-held errors. Where those two differ, the fitting residuals are
systematically larger than the ones being corrected, which biases the
correction's scale upward.

This was raised with the Phase A workstream, which found it to be a confound
in its own blend-weight arm (A5) as well and chose to emit a
**30-member-equivalent** inner out-of-fold set, matching the outer-held
ensemble's arity, at a cost of 750 networks instead of 150. The cheap
6-member variant ships alongside it so both can be measured.

**The mismatch is reduced, not eliminated, and nothing can eliminate it.**
`inner_oof_30` members are fitted on 10,095 rows against `outer_held`
members' 12,618, because each nesting level removes another fifth. What
remains is a training-size mismatch rather than an arity mismatch — the
irreducible remainder of nested cross-validation. This report states the
measured residual-scale ratio for every role so a reader can size both the
gap that was closed and the one that is left.

**Measured, the remedy turned out to be worse than the problem — see §5.4.**
The expectation when this was specified was that `inner_oof_30` would sit
closer to `outer_held` than `inner_oof_6`. It does not. That is recorded here
rather than only in the results section because the prediction belongs with
the design that made it.

On the ridge reference arm the ratio is **1.0027**, which is the floor case:
there the inner and outer fits differ only in training size, so ~0.3% is
roughly what nesting alone costs at this depth.

---

## 5. Results

### 5.1 The residual screen against the Phase A sequence ensemble

Pooled outer-held predictions, all 19,716 training rows, each predicted by a
30-member ensemble that never saw its outer fold. 69-allele panel at the ≥20
row bar, 99.65% coverage. Paired peptide-cluster bootstrap, 2,000 resamples,
seed 20261003.

| Arm | n feat | median SCC | Δ vs D0 | 95% CI | Verdict | no-correction folds |
|---|---:|---:|---:|---|---|---:|
| D0 `none` | 0 | 0.66152 | — | — | reference | — |
| D1 `geometry` | 36 | 0.66208 | +0.00056 | [−0.0035, +0.0021] | inconclusive, rules out 0.05 | 4/5 |
| D2 `confidence` | 73 | 0.66799 | +0.00647 | [−0.0055, +0.0122] | inconclusive, rules out 0.05 | 1/5 |
| D3 `anchor_geometry` | 15 | 0.66152 | +0.00000 | degenerate † | inconclusive, rules out 0.05 | **5/5** |
| D3b `anchor_p1p2p9` | 18 | 0.66152 | +0.00000 | degenerate † | inconclusive, rules out 0.05 | **5/5** |
| D4 `geometry+confidence` | 109 | 0.66994 | +0.00842 | [−0.0055, +0.0124] | inconclusive, rules out 0.05 | 2/5 |
| D3c `central_geometry` | 27 | 0.66284 | +0.00132 | [−0.0049, +0.0035] | inconclusive, rules out 0.05 | 4/5 |
| DX `geometry_shuffled` | 36 | 0.66152 | +0.00000 | degenerate † | **sign test only** | 5/5 |

† **Degenerate by construction, not extraordinary precision.** These arms
selected no correction in every fold, so their predictions are *identical* to
D0's, the delta is identically zero, and the bootstrap has nothing to
resample. A `[0, 0]` interval here means "the same predictions", not a
tighter measurement than D2's.

**Every arm returns the same verdict: inconclusive at 0, and rules out a
worthwhile gain** (EVALUATION.md's fifth reading). The best arm's upper bound
is 0.0124 — a quarter of the 0.05 bar. Strict-panel (≥50 rows, 68 alleles)
figures move nothing: D0 0.66047, best arm D4 0.66313.

### 5.2 What the two sharpest findings are

**The anchor hypothesis is dead, and so is its hedge.** D3 (P2/P9) and D3b
(the prespecified P1 extension) selected **no correction in all five folds**.
The parsimony rule, declared before fitting and deliberately pointing toward
not correcting, found nothing in anchor geometry worth any weight at all on
top of a 30-member sequence ensemble.

The plan asked for a full-feature control precisely so that a null on anchors
could not be mistaken for a null on geometry. Both were run, and **both are
null**: D1 (all 36 geometry columns) reaches no-correction in 4 of 5 folds,
and D3c — central positions P3–P8, the direct test of the plan's hedge that
central residues might carry biology the anchors miss — also reaches
no-correction in 4 of 5. Neither the anchors, nor the centre, nor the whole
block.

**Confidence features are the only block with any signal, and it is too
small.** D2 and D4 are the only arms that consistently select a finite alpha,
and they are the only arms whose MSE improves (−0.0043 and −0.0036). Their
point estimates are positive (+0.0065, +0.0084) and their intervals exclude
nothing at zero. Read honestly: there is probably a small real effect in the
confidence block, and it is far too small to matter. That is a complementary
signal of the kind the plan says can justify a blend experiment without being
a standalone win — but at this magnitude, against this baseline, it does not
justify the blend either.

### 5.3 The prespecified ladder extension did **not** fire

No candidate arm selected `1e1`. Selections went to the *strong*-shrinkage
end — `inf`, `1e5`, `1e6` — so the ladder's low edge was never reached and the
declared downward extension was not triggered. The plumbing run's low-edge hit
(§5.5) was an artifact of its deliberately weak reference, as expected.

Confirming this from the other direction: the **affine recalibration slopes**
flipped sign against the real comparator. They run 1.032–1.054, so the
correction *undershoots* rather than overshoots — exactly what heavy shrinkage
predicts, and the opposite of the plumbing run's 0.905–0.962.

### 5.4 The arity fix made the residual-scale mismatch worse, not better

This phase asked the Phase A workstream to emit a 30-member-equivalent inner
out-of-fold set, at a cost of 750 networks instead of 150, on the argument
that fitting a correction on a 6-member error distribution and applying it to
a 30-member one mismatches ensemble arity. **Measured, the remedy is worse
than the problem it fixed:**

| role | residual sd | ratio vs `outer_held` | members | mean fit rows | member dev MSE |
|---|---:|---:|---:|---:|---:|
| `outer_held` | 0.6674 | 1.0000 | 30 | 12,618 | 0.5543 |
| `inner_oof_30` | 0.6903 | **1.0342** | 30 | 10,095 | 0.5903 |
| `inner_oof_6` | 0.6803 | **1.0193** | 6 | 12,618 | 0.5543 |

`inner_oof_6` is **1.8× closer** to the quantity it stands in for. The
training-size penalty of the extra nesting level (12,618 → 10,095 fit rows,
dev MSE 0.5543 → 0.5903) outweighs the arity gain. A 6-member ensemble of
better-trained members is a closer surrogate for a 30-member ensemble of
well-trained members than a 30-member ensemble of worse-trained ones.

The confound identified was real; what was missing was the training-size cost
of the remedy, which nobody had measured. The sign could not have been known
in advance — which is an argument for measuring before paying, not for the
reasoning having been wrong. The 750 networks did buy the measurement: the
irreducible remainder of nested CV — what is left once ensemble arity is
matched and only training size differs — is now **0.0229 in residual sd, or
3.4% of `outer_held`'s**, rather than a caveat. `inner_oof_30` remains the
right input for anything that needs matched arity for its own sake.

**The primary run keeps `inner_oof_30`**, because that preference was declared
in `phase_d_protocol.md` before any of this was visible, and switching after
seeing the ratio would be post-hoc. `inner_oof_6` is reported as a declared
sensitivity check:

| Arm | Δ on `inner_oof_30` (primary) | Δ on `inner_oof_6` (sensitivity) |
|---|---|---|
| D1 `geometry` | +0.00056 [−0.0035, +0.0021] | +0.00000 (degenerate; 5/5 no-correction) |
| D2 `confidence` | +0.00647 [−0.0055, +0.0122] | +0.00632 [−0.0055, +0.0114] |
| D3 / D3b anchor | +0.00000 (5/5 no-correction) | +0.00000 (5/5 no-correction) |
| D4 `geometry+confidence` | +0.00842 [−0.0055, +0.0124] | +0.00427 [−0.0042, +0.0091] |
| D3c `central_geometry` | +0.00132 [−0.0049, +0.0035] | +0.00000 (degenerate; 5/5 no-correction) |

**The verdict is robust and the sensitivity check is, if anything, more
negative**: on the closer surrogate, D1 and D3c also go to no-correction in
every fold and D4's point estimate halves. Every arm returns the same reading
under both.

Run tables: [`phase_d_residual_runs_phase_a.csv`](phase_d_residual_runs_phase_a.csv),
[`phase_d_residual_selection_phase_a.csv`](phase_d_residual_selection_phase_a.csv),
[`phase_d_residual_by_fold_phase_a.csv`](phase_d_residual_by_fold_phase_a.csv),
[`phase_d_residual_per_allele_phase_a.csv`](phase_d_residual_per_allele_phase_a.csv),
[`phase_d_residual_manifest_phase_a.json`](phase_d_residual_manifest_phase_a.json),
and the `_phase_a_sens_oof6` set for the sensitivity check.
Code: `scripts/phase_d_structural_residual.py`. Wall 674 s on one pinned core.

### 5.5 The harness-validation run — a plumbing check, not a result

Run first against a deliberately cheap **ridge** sequence reference (BLOSUM
pep+pseudosequence, alpha 10), in the role
`pepstab.stage6.ridge_oof_predictions` plays for the nested near-neighbour
harness: it exercises the pipeline end to end without waiting on another
workstream.

That reference scores median per-allele Spearman **0.2358**, against A0's
0.6615 on the same folds. A baseline that weak leaves roughly three times the
residual headroom, so a correction fitted against it is largely fitting error
the real ensemble does not make. **Its apparent gains were not a forecast and
did not survive**: the confidence block moved +0.111 there against +0.006
against A0, and geometry +0.036 against +0.001. The gap between those two
columns is the clearest illustration in this study of why a comparator must be
the real one.

| Arm | Δ vs ridge D0 (0.2358) | Δ vs A0 D0 (0.6615) |
|---|---|---|
| D1 `geometry` | +0.0355 [+0.011, +0.054] | +0.0006 [−0.004, +0.002] |
| D2 `confidence` | +0.1111 [+0.095, +0.153] | +0.0065 [−0.006, +0.012] |
| D4 `geometry+confidence` | +0.1136 [+0.099, +0.160] | +0.0084 [−0.005, +0.012] |

200 resamples on the plumbing run only. What it legitimately established,
independent of effect size: the harness does not leak, the selection rule
declines to correct on noise and corrects on signal, and the ladder's low edge
was reachable — which is what prompted the extension trigger in §5.3 that then
did not fire.

### 5.6 A retracted claim, kept visible

An earlier reading of the plumbing run reported that the geometry correction
improves ranking while making squared error worse, on the evidence that the two
parallel selection criteria disagreed: inner median per-allele Spearman
selected a finite alpha for every geometry arm, inner MSE selected `inf`.

**That reading was wrong.** Squared error did not get worse on that run; it
improved for every arm, concentrated within-allele. The error was reading the
*selection* instead of the ladder profile, which was already in the run table:
D1's inner MSE over `(1e1 … 1e6, inf)` was
`0.83284 | 0.83281 | 0.83219 | 0.83134 | 0.83453 | 0.83655 | 0.83618`, so the
argmin at `1e4` beat no-correction by 0.0048 — a real improvement, just
smaller than one inner-fold standard error, which is what sent the **1-SE
parsimony rule** to `inf`.

The two criteria never disagreed about direction. They disagreed about whether
the MSE improvement was resolvable at five inner folds, and the parsimony rule
answered no. That is the rule working as designed. It is recorded here rather
than quietly dropped because this project selects on 1-SE ladders throughout,
and "the parsimony rule declined to correct" is a result about the rule that
is easy to misread as a result about the data.

On the real comparator the question is moot: MSE changes by at most −0.0043
(D2) and +0.00007 (D3c), and the between-allele component changes by at most
0.0003 in either direction. There is no ranking-versus-calibration tradeoff to
report.

### 5.7 Two calibration maps that are not interchangeable

Both facts below were found by asserting a claim rather than assuming it, and
both matter beyond this phase.

**Isotonic calibration is not rank-preserving.** Isotonic regression is
*weakly* monotone — a step function — so it collapses ranges of distinct
predictions into equal values, and those new ties move per-allele Spearman,
because ties take average ranks under the frozen scoring rule. Measured here:
up to **−0.016** median SCC, a third of the project's 0.05 bar. The
orchestration session reproduced the mechanism independently on a 40-row
allele — the scale the ≥20-row bar admits — where `IsotonicRegression`
collapsed 40 distinct predictions to 7 and moved Spearman by 0.23.

**The sign is not predictable.** That session's 0.23 was *upward*, from an
in-sample fit whose tie-collapse happened to align with the labels; the
out-of-sample fits here move down. So this is a confound, not a conservatism:
"isotonic costs up to 0.016" must not be read as "then my result is a lower
bound."

**A strictly monotone affine rescale is rank-safe — but only at the scope it
is fitted.** `y ≈ a + b·x` with `b > 0` cannot reorder anything, and the
harness asserts this to 1e-8. The assertion initially *fired* on the pooled
prediction vector, which was correct behaviour rather than a bug: each outer
fold gets its own `(a, b)`, so the pooled vector is five affine transforms
stitched together and is not a single monotone map — within one allele, rows
from different outer folds can reorder. The measured pooled shift is small
(≤ 0.0012) but not zero. The assertion now sits at fold scope, where the
guarantee holds, and passes exactly there.

Consequence for Phase B, which owns the calibration work: a post-hoc
calibrated baseline must be scored on **uncalibrated** predictions for the
primary metric, with any map confined to magnitude metrics and its measured
effect on ranking reported rather than assumed to be zero.

## 6. What this phase does not establish

- **No validation or test number.** Every figure here is training-only nested
  cluster validation. Nothing in this report is a performance claim against
  the frozen benchmark.
- **The plumbing run is not a comparator.** §5.5's deltas are measured against
  a ridge reference 0.43 median SCC below the Phase A arm, and none of its
  apparent gains survived the real one.
- **A null against this sequence arm is not a null against structure.** The
  arms tested are 109 summary features entering as a shrunk *linear*
  correction on top of a 30-member ensemble. A non-linear reader of the same
  features, or an interface representation rather than a summary of one, is
  untested here. What the result does establish is that the cheap version of
  the idea does not work, which is what the plan's §7 gate asked.
- **The exclusion is of a worthwhile gain, not of any gain.** D2 and D4 have
  positive point estimates whose intervals exclude nothing at zero; there is
  plausibly a small real effect in the confidence block. The claim is that its
  size is bounded well below the bar, not that it is zero.
- **+0.05 is this project's worthwhile-effect threshold, not a detection
  limit.** Precision depends on the paired errors and the actual interval; an
  interval that excludes 0 below 0.05 is a real but too-small effect, and an
  interval crossing 0 is inconclusive rather than negative.
- **Seed reliability is not physical flexibility.** §3's reliability numbers
  say a feature is reproducible, not that the pose is correct. Accuracy
  against crystal structures is a separate question, measured in stage 4c5,
  and the two are not used to explain each other.
- **One configuration only.** Production is Boltz-2, seed 0, arm B. Nothing
  here speaks to other folding models, other construct arms, or other seeds
  beyond the 3-seed pilot.
- **The transfer assumption in §3** — that the pilot's 5 complexes have
  representative seed noise — cannot be checked without new poses.
