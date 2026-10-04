# Next steps

What remains worth doing after stage 5, and what does **not** — because the
repo has already settled it.

[HACKATHON_PLAN.md](HACKATHON_PLAN.md) and [reports/](reports/SUBMISSION.md)
are ground truth. This document adds only items they do not already carry, and
records explicitly where a plausible-sounding next step is already closed, so
nobody re-runs a finished experiment.

## What stage 5 settled

The structural arm is **not** an untested question, and the negative is not a
smoke-run artefact:

- **The fold succeeded completely** — 28,166 of 28,166 pairs, 100% structural
  coverage, zero sequence fallback, poses reproducing the pilot within 0.04 Å.
- **Structural features do not help, and adding them hurts.** Against the
  sequence ensemble at 0.6931: geometry −0.4254, confidence −0.3006, both
  −0.2722; added on top of the baseline, seq+geometry −0.0788,
  seq+confidence −0.0973, seq+geometry+confidence −0.0715. Every interval sits
  entirely below zero — **negative, not inconclusive**.
- **It is attributable to the features**, not to tuning: a `seq_only` control
  through the same pipeline, ladder and folds beats seq+geometry+confidence at
  all six L2 values tested.
- **Direct geometry was included.** Burial, contacts and the other aggregates
  are in the 109 structural features, and **none of the 109 isolates the
  failure**.

This is a well-powered negative, consistent with a static co-folded pose
estimating well depth against a barrier-height label.

## Already closed — do not re-run

| Plausible next step | Status |
| --- | --- |
| Direct geometric features — pocket burial, anchor depth, H-bonds, apolar contacts | **Tested, negative.** Inside the 109 features; seq+geometry −0.0788 |
| Confidence channels (pLDDT/PAE) as predictors | **Tested, negative.** seq+confidence −0.0973 |
| Seed-ensemble spread as a flexibility proxy | **Measured.** Between-complex over within-seed spread 3.1–9.3 (Boltz-2): signal-dominated, already characterised |
| Per-pocket energy decomposition (A/B/F ↔ P1/P2/PΩ) | **Rejected on two grounds** — premised on FoldX, which is licence-blocked, and ΔG is thermodynamically mismatched to a kinetic label |
| Molecular-dynamics unbinding | **Deferred in the register**, same kinetic-mismatch caution; short MD does not give k_off |
| Template threading | **Rejected** — cannot represent non-canonical bulges, which may be precisely the unstable complexes |
| Elution as training augmentation | **Rejected** — scale mismatch; it is external validation only ([stage 3c](reports/stage3c_elution_validation.md)) |
| Auxiliary affinity head | **Tested, inconclusive, rules out 0.05** ([stage 2c](reports/stage2c_affinity.md)) |

## Genuinely open

### 1. The pairformer trunk pair representation

The **one** structural item stage 5 did not test. Stage 5 used 109 aggregate
features derived from the *output* pose and its confidence; it never read the
trunk's internal peptide × groove pair representation. Extract it from the
final one or two pairformer blocks and evaluate it as its own arm.

Why it survives the stage 5 negative: coarse aggregates over a static pose
losing the signal does not establish that the learned representation carries
none. The folds are on disk and co-folding is already paid for, so this is
extraction plus a head — not another $6.90-per-1,000 production fold.

**It must clear a high bar to be believed.** Stage 5 is a well-powered negative
on the same structures; a positive here would be the single dissenting result,
so it needs the same comparator (the 30-network ensemble at 0.693), the same
folds, matched head architecture, ensemble size and tuning budget, plus a
`seq_only` control through the identical pipeline. Report against the
predeclared 0.05 bar in [EVALUATION.md](EVALUATION.md).

### 2. Items the register already names as untested, now unblocked

These are recorded in the plan's out-of-scope register as *promoted but never
run*, and the stage 3 null is explicitly bounded because of them:

- **Chimeric peptide–linker–α1α2 ESM input.** The register calls this "the
  sharpest *untested* version of the question". The original objection stands
  (a linkered 9-mer sits outside ESM-2's distribution), so it is a labelled
  diagnostic, not a headline arm.
- **A second pLM family (ESM-C / ProtT5).** Gated on RAM, never cleared. One
  family is a thin basis for "foundation models don't help"; a second says
  whether the stage 3 null is family-specific. **Sweep layers** rather than
  assuming the final layer, which stage 3 did not vary.
- **Stage 3d, ESM-2 likelihood features.** Not run — untested scope, not a
  null.

Holding features fixed when comparing concatenation against cross-attention
remains the right design, so coupling gains stay separable from
representation gains.

### 3. Reframe the target

Neither of these is in the plan, and neither needs new data:

- **Between-allele Δlog(t½) for shared peptides** — differences out the
  peptide-intrinsic component and isolates groove-dependent effects. Stage 6
  already carries a differential target; this is the allele-axis version.
- **Stability residuals conditional on affinity** on the 7,281 matched pairs.
  Stage 2c tested affinity as an auxiliary *input* and found it redundant;
  predicting the residual asks the different question of what stability carries
  *beyond* affinity. A conditional target, not a pure measure of kinetics.

### 4. Commission the missing experiment

The one item no amount of compute substitutes for. **Pilot:** a few hundred
peptides across 8–10 deliberately divergent alleles; get feasibility and price
quotes before committing. Prefer stability measurement over more elution data —
we hold 487,343 elution rows already and
[stage 3c](reports/stage3c_elution_validation.md) is why they do not substitute.

1. **Divergent grooves**, chosen by distance from the measured allele set.
   Median nearest-neighbour pseudosequence identity across the current 75
   alleles is **0.941** — that homogeneity is the gap this targets, and it is
   also the confound behind the allele-axis hold-out. Confirm assay feasibility
   for rare, non-classical or non-human candidates.
2. **Two or three independently retained replicates.** The current dataset
   ships none, so **no noise ceiling can be estimated from it** — we cannot
   currently say how much of the residual is irreducible.
3. **Peptides sampled uniformly or across score strata**, not only predicted
   binders.
4. **Full dissociation time courses**, early sampling, explicit censoring and
   detection limits. About **20.2%** of current labels are exactly zero; stage
   7's censored-likelihood arm was conclusively worse, so the fix is better
   measurement, not a better loss.
5. **Matched WT/pocket-mutant panels** on one or two allele backgrounds.
6. **Matched affinity and stability** for the same pairs.

**Vendor shortlist from the supplied notes:** immunAware and ProImmune first;
Creative BioMart/Biolabs as alternatives. Verify allotype coverage, curve
export, replicate design, detection limits and pricing.
**Vendor capabilities and pricing are not independently verified here.**

### 5. Publish the bounded result

Mostly already satisfied — [reports/SUBMISSION.md](reports/SUBMISSION.md),
[limitations.md](reports/limitations.md) and
[compute_ledger.md](reports/compute_ledger.md) carry Δρ with CIs, ablations and
cost per arm. What remains:

- **Report the structural negative with its exact scope**: confidence *and*
  coarse geometric aggregates over Boltz-2 co-folded poses fail to beat
  sequence on this dataset. Not "structure is uninformative" — the trunk
  representation (§1) is untested, and that boundary is the honest one.
- **Extend elution validation to the ESM-2 and structural arms** — the
  follow-on the submission flags as still open.
- Release the tiered PDB template table, matched-affinity reference and elution
  evaluation as standalone resources.

## Figures

Verified against this repo:

| figure | value | source |
| --- | ---: | --- |
| stability measurements | 28,166 | [docs/DATASETS.md](docs/DATASETS.md) |
| alleles | 75 | [reports/audit_summary.md](reports/audit_summary.md) |
| median NN pseudosequence identity | 0.941 | computed from `data/rasmussen_et_al_dataset.csv` |
| zero-valued labels | 20.2% | computed; [reports/audit_summary.md](reports/audit_summary.md) |
| matched-affinity pairs | 7,281 (25.8%, 58 alleles) | [docs/AFFINITY_REFERENCE.md](docs/AFFINITY_REFERENCE.md) |
| affinity–stability pooled Spearman | −0.491 | [docs/AFFINITY_REFERENCE.md](docs/AFFINITY_REFERENCE.md) |
| elution rows | 487,343 | [reports/stage3c_elution_validation.md](reports/stage3c_elution_validation.md) |
| structural coverage | 28,166 / 28,166, zero fallback | stage 5 |

The affinity–stability sign is **settled, not open**: Spearman between
`affinity_nM` (untransformed, lower = tighter) and `stability_thalf_hours`, so
negative is the expected direction. Per allele, across the 36 alleles with ≥ 30
dual-measured pairs, median −0.578 (range −0.839 to 0.002).

Every number above is a **validation** number. The test split is scored once,
at stage 6, under
[reports/TEST_SCORING_RUNBOOK.md](reports/TEST_SCORING_RUNBOOK.md).
