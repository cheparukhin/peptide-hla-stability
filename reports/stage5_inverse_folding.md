# Stage 5 stretch arm: ProteinMPNN inverse folding

4 October 2026. **Status: pilot complete on all 90 stage 4c folds. The seed
control passes for Boltz-2 and fails for ESMFold2. The circularity control came
back with a result I did not expect, and it changes what this feature is for:
the ProteinMPNN peptide log-likelihood behaves as a crystal-free triage signal
for Boltz-2 pose failure, and there is no evidence from this pilot that it
predicts half-life. A ~$1 QC sample is approved; full production scoring as a
regression feature is not, and nothing has been launched.**

**The single most important limit, stated before anything else:** the pilot
contains exactly **one** complex that folded badly (B*07:02/IPRRNVATL, in arms
A and C). Every claim below about detecting pose failure rests on that one
complex, measured three ways. It is suggestive and it is well controlled, but
it is not a general property of the method, and no number in this report should
be quoted without that caveat attached.

This closes the third of the three protein-foundation-model classes named in
the challenge brief. The project already tests structure prediction (Boltz-2,
stage 4c) and protein language models (ESM-2, stage 3). Inverse folding was the
untested one, listed out of scope in `HACKATHON_PLAN.md` because it "requires
structures first". The stage 4c pilot left 90 folds on disk, which unblocked it.

## Read this first: three caveats, up front

1. **The score inherits every error in the predicted backbone.** It is as much
   a feature of Boltz-2's output as of biology. Section "What this feature
   actually measures" shows that this is not a hedge — it is the dominant
   effect, and in the end it is what makes the feature useful.
2. **Sequence–structure compatibility is thermodynamic-flavoured; half-life is
   kinetic.** It is governed by the barrier to unbinding, not by how well a
   sequence fits a groove. This project rejected FoldX for exactly this
   mismatch (`HACKATHON_PLAN.md` out-of-scope register). The same mismatch
   applies here in softer form, and nothing below overturns it.
3. **Circularity risk.** Boltz-2 was *given* the peptide sequence and built a
   backbone to suit it; ProteinMPNN then reports that the sequence suits the
   backbone. This was the stated gate on whether to spend anything on
   production, and it is measured directly below rather than argued about.

## Provenance

Without the checkpoint identity the score is meaningless, so:

| | |
|---|---|
| Repository | `https://github.com/dauparas/ProteinMPNN` (MIT) |
| Commit | `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57` |
| Checkpoint | `vanilla_model_weights/v_48_020.pt` |
| Checkpoint sha256 | `c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd` |
| Mode | `ProteinMPNN.forward` — teacher-forced **scoring**, not `.sample` |
| Backbone noise | `augment_eps = 0.0` (deterministic given a decoding order) |
| Hardware | laptop CPU, no GPU, no cloud, **$0** |

`v_48_020.pt` is ProteinMPNN's default published checkpoint: 48 neighbours,
0.20 Å training backbone noise. Cloned into `external/` (gitignored).

## The feature definition, declared before it was computed

This definition is the module docstring of `pepstab/inverse_folding.py` and was
written before any number in this report existed.

Let a complex have chains H (HLA), optionally M (beta2m), and P (the 9-mer
peptide). The peptide chain is the **only masked (designed) chain**; H and M
are **visible**, so their sequences are given to the decoder. ProteinMPNN's
decoding order is `argsort((chain_M + 1e-4) * |randn|)` and visible positions
carry `chain_M = 0`, so **every HLA and beta2m position is decoded before any
peptide position** — the HLA really is held fixed, by the model's own rule
rather than by assertion. `tests/test_inverse_folding.py` checks this directly.

Within the peptide the order is a uniform random permutation, so one forward
pass gives one autoregressive factorisation of
`log P(peptide | backbone, HLA sequence)`. We average over 16 permutations from
a fixed seed.

Reported per prediction:

| Field | Meaning |
|---|---|
| `pep_ll_total` | mean over decoding orders of `sum_i log p(s_i | ...)`, in nats; **higher = more compatible** |
| `pep_ll_mean` | `pep_ll_total / 9` |
| `mpnn_score` | `-pep_ll_mean`, ProteinMPNN's own convention (lower = better) |
| `ll_pos_1..9` | per-position mean log-likelihood, P1..P9 |
| `order_sd` | s.d. across decoding orders — the feature's pure numerical noise floor |

Measured decoding-order noise: `order_sd` mean 0.47 nats, max 0.77, so the
standard error of `pep_ll_total` at 16 orders is about 0.12 nats. Every
difference discussed below is far larger than that.

**Scoring, not sampling.** These are different ProteinMPNN modes and only the
former is a feature. `score_complex` calls `ProteinMPNN.forward` with the true
sequence tensor `S` and reads log-probabilities back; it never calls
`.sample`. A test asserts that the designable positions hold exactly the native
peptide before any score is taken.

**Chain identification is inherited, not reinvented.** `load_complex` uses
`scripts/boltz_pose_check.{load_prediction, ca_by_chain, find_subsequence}` —
the same parsing the stage 4c pose checker uses — and finds the peptide chain by
exact sequence match rather than hard-coding chain `C`. It then cross-checks
every chain sequence against `metadata.json`'s `inputs.chains` and raises on any
disagreement. This reproduces the pose checker's layout exactly: arms A and C
are HLA 182 + peptide 9; arm B is HLA ectodomain 275 + beta2m 99 + peptide 9.
No disagreement with the pose checker was found.

## Pilot: all 90 stage 4c folds

5 complexes x 3 arms x 3 seeds x 2 models. 90/90 scored, 0 failures, 481 s wall
on 2 laptop cores.

Mean `pep_ll_total` over the 3 seeds (nats, higher = better):

| Complex | t½ (h) | Boltz A | Boltz B | Boltz C | ESM A | ESM B | ESM C |
|---|---:|---:|---:|---:|---:|---:|---:|
| A*02:01 LLWNGPMAV | 38.2 | -20.96 | -21.14 | -20.88 | -20.26 | -21.91 | -20.95 |
| A*11:01 KTFPPTEPK | 59.3 | -21.81 | -22.29 | -21.79 | -21.73 | -21.91 | -21.75 |
| B*07:02 IPRRNVATL | 4.0 | **-27.55** | **-21.49** | **-27.68** | -21.90 | -22.00 | -21.89 |
| B*08:01 ELRRKMMYM | 0.75 | -21.68 | -22.26 | -21.76 | -23.69 | -27.67 | -23.83 |
| B*15:01 ILGPPGSVY | 11.0 | -17.13 | -17.51 | -17.46 | -19.43 | -18.37 | -19.59 |

All five are **training** rows of `data/splits.csv`. No validation or test row
was scored. Per `HACKATHON_PLAN.md` these five cannot support a generalization
claim, and none is made.

### Seed control — the critical gate

If the score moves more across seeds of one complex than it does between
complexes, the feature is noise.

**Definition of the quoted ratio.** Between-complex s.d. divided by the mean
within-complex seed s.d., computed **per arm**, on the **`pep_ll_total`**
column, with `ddof=1`. Stating this matters because the same control can be
computed several defensible ways and this report is not the only place it
appears:

| Aggregation (all on `pep_ll_total`, = `mpnn_score` — see below) | Boltz-2 | ESMFold2 |
|---|---:|---:|
| **Per arm, `ddof=1` (quoted throughout this report)** | **10.0 / 8.0 / 6.8** | **1.1 / 2.4 / 1.2** |
| Pooled, (complex, arm) as unit, mean of SDs, `ddof=1` | 7.81 | 1.56 |
| Pooled, (complex, arm) as unit, mean of SDs, `ddof=0` | 9.24 | 1.84 |
| Pooled, (complex, arm) as unit, RMS of SDs, `ddof=1` | 6.73 | 1.23 |
| Pooled, between-SD over all rows, `ddof=1` | 7.69 | 1.85 |
| Pooled ignoring arm entirely, `ddof=1` | 3.13 | 1.33 |

**The column provably cannot matter.** `mpnn_score = -pep_ll_total / 9` is an
affine transform, and a ratio of two standard deviations is invariant under
affine rescaling, so the two columns give identical ratios *to every digit*.
This is not a reconciliation of two conventions — it is a proof that one axis
of apparent disagreement cannot exist. Confirmed numerically as well as
analytically.

**The aggregation does matter, and I report the envelope rather than a single
number.** Across the defensible choices above, ESMFold2 lands between **1.2 and
1.9** and Boltz-2 between **3.1 and 9.3**. The stage 4c.5 workstream quotes
**8.10 / 1.83** (`ddof=1`, pooled), inside this envelope.

**The difference between the two implementations is mostly one definitional
choice, with a small remainder.** The axis is what counts as a unit in the
numerator: each *fold* (45 units), or each *(complex, arm)* after averaging its
three seeds (15 units). Holding the denominator fixed and varying only that,
on my own data:

| Numerator unit | Boltz-2 | ESMFold2 |
|---|---:|---:|
| Each fold a unit | 7.69 | **1.85** |
| Seeds averaged first | 7.81 | **1.56** |
| *(stage 4c.5, same two conventions)* | *8.10 → 8.07* | *1.83 → 1.54* |

**The shape of the effect confirms the explanation rather than merely fitting
it.** Averaging seeds first strips seed noise out of the numerator, so it must
pull the ratio down hard for the *noisier* model and barely move the *quieter*
one. That is exactly what happens, in both implementations independently:
ESMFold2 falls by **-0.29** in mine and **-0.29** in theirs — identical to two
decimals — while Boltz-2 moves by +0.12 and -0.03 respectively. A directional,
magnitude-asymmetric prediction borne out on two separately-written pipelines
is an explanation; a pair of numbers that merely land near each other is not.

What remains unexplained is small and confined to Boltz-2: our ESMFold2 figures
agree to 0.02 under both conventions, our Boltz-2 figures differ by 0.3–0.4.
Deliberately not chased — every aggregation either implementation has tried
returns the same verdict (ESMFold2 marginal, Boltz-2 comfortable), production
is Boltz-2 only regardless, and the residual touches no conclusion. **Neither
set of figures was reproduced to the digit by the other**, and this report does
not claim otherwise.

An earlier draft claimed its 1.83 reconciled with the `ddof=0` row above
(1.84); **that was wrong** — its figure is `ddof=1`, and `ddof=0` on its side
gives 2.00. Two different conventions had landed near the same number, which
is precisely the coincidence the asymmetry test above is designed to
distinguish from a real explanation.

Per arm, `ddof=1`:

| Model | Arm A | Arm B | Arm C |
|---|---:|---:|---:|
| Boltz-2 | **10.0** | **8.0** | **6.8** |
| ESMFold2 | 1.1 | 2.4 | 1.2 |

**Boltz-2 passes.** Arm B — the frozen production configuration — is 8.0x, with
mean seed s.d. 0.248 nats against a between-complex s.d. of 1.98 nats. Worst
seed range on any Boltz-2 arm-B complex is 0.97 nats.

**ESMFold2 fails this control**, sitting near the noise floor (1.1–2.4x per
arm; 1.56–1.84 pooled), with seed ranges up to 7.3 nats.

Two precisions on how far that goes.

**It is marginal, not inverted.** ESMFold2's between-complex spread still
*exceeds* its seed spread — by roughly 1.6–1.8x pooled rather than Boltz-2's
~8x. The correct statement is that the margin collapses to the point where the
feature is not usable, not that the ordering reverses.

**What it does and does not corroborate.** It *does* independently corroborate
the stage 4c production decision: stage 4c rejected ESMFold2 on peptide
heavy-atom RMSD against crystal structures, whereas this measurement uses **no
crystal structure at all** — it asks a third-party model whether the predicted
backbone is self-consistent with the sequence that produced it — and reaches the
same verdict. Two unrelated routes to one rejection is worth more than either
alone.

It does **not** corroborate the stage 4c.5 structural-feature seed control, and
this report should not be read as claiming it does. That control reports
**5.87** between/seed on ESMFold2 against **5.29** on Boltz-2 — marginally
*more* seed-stable on ESMFold2 — while ProteinMPNN collapses to 1.83 on the same
folds. The two controls agree on Boltz-2 and **diverge on ESMFold2**.

**The divergence is the more interesting result, and it is mechanistically
explicable.** ProteinMPNN reads **backbone geometry only**: its featuriser takes
N, CA, C and O, plus a *virtual* CB computed from N/CA/C
(`protein_mpnn_utils.py:970`). It never sees a real side-chain coordinate. So
the explanation cannot be side-chain sensitivity. What it is instead: the two
measurements differ in *granularity*. ProteinMPNN is a fine-grained readout of
local backbone geometry — inter-atomic distances to 48 neighbours, with the
virtual-CB direction set by backbone dihedrals — whereas the 109 structural
features are coarse aggregates (contacts, burial, confidence means) that are
robust to sub-Ångström jitter.

And ESMFold2's backbone genuinely is seed-unstable at that scale. Mean
within-complex seed s.d. of peptide RMSD, from `pilot_pose_scores.csv`:

| Model | arm | CA seed s.d. (Å) | heavy seed s.d. (Å) |
|---|---|---:|---:|
| ESMFold2 | A / B / C | 0.114 / 0.085 / 0.119 | 0.117 / 0.096 / 0.111 |
| Boltz-2 | A / B / C | 0.046 / 0.016 / 0.037 | 0.113 / 0.044 / 0.135 |

ESMFold2's **backbone** moves 2–7x more between seeds than Boltz-2's, and its
seed variation is backbone-dominated (heavy/CA ratio ≈ 1.1) where Boltz-2's is
side-chain-dominated (≈ 2.8 on arm B). A backbone-only, fine-grained reader is
exactly the instrument that would see this and a coarse aggregate is exactly the
one that would not. The two controls are therefore **complementary rather than
redundant**, and their disagreement localises where ESMFold2's instability
lives: in fine backbone detail, not in the coarse pose descriptors.

One caveat on that explanation: the granularity account is consistent with every
number above, but it is an interpretation of five complexes, not a controlled
experiment. Stage 4c separately attributed ESMFold2's *accuracy* failure to
side-chain placement (its backbone passes at max 1.57 Å CA RMSD while
heavy-atom RMSD fails); that is a different question from *seed stability*, and
the two should not be conflated.

### Arm control, and the sentinel

Boltz-2 B*07:02/IPRRNVATL scores **-27.55 on arm A and -21.49 on arm B**. This
is the exact complex whose arm-A pose carries the 2.3 Å central-bulge error that
arm B fixes to 0.32 Å. Arm C (-27.68), which uses arm B's alignment cropped to
the groove, tracks arm A — the same pattern stage 4c found in RMSD.

The per-position breakdown localises it. Mean `ll_pos` for that complex:

| | I1 | P2 | R3 | R4 | N5 | V6 | A7 | T8 | L9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Arm A (bad pose) | -4.15 | -0.92 | -4.75 | -3.14 | -5.52 | -3.84 | -3.06 | -2.06 | -0.12 |
| Arm B (good pose) | -4.24 | -1.32 | -4.94 | -2.85 | -2.21 | -3.75 | -1.33 | -0.74 | -0.11 |
| Delta | -0.09 | -0.40 | -0.19 | +0.29 | **+3.31** | +0.09 | **+1.73** | **+1.32** | +0.01 |

The gain is concentrated at N5, A7 and T8 — the bulge. Stage 4c's arm-A
per-position CA deviation for the same complex is `N5 1.84, V6 3.39, A7 0.60`
against ≤0.2 Å elsewhere. Two independent measurements point at the same three
residues. The score reads backbone geometry, and it reads it *locally*.

Anchor biology also shows up unprompted: over the five Boltz-2 arm-B complexes
the mean per-position likelihood is highest at P2 (-1.15) and second-highest at
P9 (-1.72), against -2.4 to -3.8 at the non-anchor positions. P2 and PΩ are the
pockets that hold the peptide. Nothing in the setup told the model that.

## The circularity control — the gate on spending anything

This was the declared gate: if the native sequence wins only because the folder
built that backbone *from* that sequence, the feature is an expensive
restatement of the input and production scoring is not worth buying.

**Test.** On each Boltz-2 arm-B backbone, score all five pilot peptides (all
9-mers) and ask whether the native one wins. 15 backbones x 5 sequences x 16
decoding orders, 1,107 s on one laptop core.

Mean `pep_ll_total` over 3 seeds. Rows are backbones, columns are the sequence
scored on them; the diagonal is native.

| Backbone | LLWNGPMAV | KTFPPTEPK | IPRRNVATL | ELRRKMMYM | ILGPPGSVY |
|---|---:|---:|---:|---:|---:|
| A*02:01 LLWNGPMAV | **-21.08** | -30.81 | -29.11 | -31.13 | -32.05 |
| A*11:01 KTFPPTEPK | -33.26 | **-22.42** | -35.18 | -36.55 | -24.21 |
| B*07:02 IPRRNVATL | -38.08 | -34.79 | **-21.64** | -37.76 | -37.64 |
| B*08:01 ELRRKMMYM | -30.33 | -32.51 | -25.19 | **-22.21** | -35.09 |
| B*15:01 ILGPPGSVY | -36.27 | -29.27 | -26.63 | -37.83 | **-17.51** |

The native sequence ranks **first on 15 of 15 backbones**, by a mean margin of
7.01 nats over the best foreign peptide (min 0.40) and 11.71 nats over the mean
foreign peptide. The matrix is overwhelmingly diagonal.

**Taken alone this is ambiguous**, which is worth saying plainly: a real crystal
structure would also be diagonal, because the peptide's conformation genuinely
*is* determined by its sequence and the pocket. Diagonal dominance is consistent
with circularity and with real structural signal, and does not separate them.

### The control that does separate them

Run the same test on a backbone that is **known to be wrong**: Boltz-2 arm A for
B*07:02/IPRRNVATL, the 2.3 Å bulge error. Boltz-2 was given the identical
peptide sequence in arm A as in arm B. If the diagonal dominance were pure
circularity, arm A would show it too.

| Backbone for IPRRNVATL | Pose error | Native margin over best foreign |
|---|---|---:|
| Arm B (correct, 0.32 Å) | good | **+13.15** (s.d. 0.11 over 3 seeds) |
| Arm A (wrong, 2.3 Å) | bulge error | **-0.45, -0.17, +0.08** (per seed) |

On the wrong backbone the native peptide's advantage **collapses to zero and it
loses outright on two of three seeds**, despite the folder having been handed
that very sequence. Same model, same sequence, same complex, same seeds; the
only thing that changed is whether the backbone is right.

**Conclusion: the signal is not a tautological readback of the input.**
ProteinMPNN recovers the sequence only when the backbone is actually correct.

## What this feature actually measures

The controls together say something more specific — and more useful — than the
framing the plan started from.

**Units first, because three columns are one quantity.** `mpnn_score` =
−`pep_ll_mean`, and `pep_ll_total` = `pep_ll_mean` × 9. This section quotes
**`pep_ll_mean`** (nats per residue) throughout, because that is the unit the
stage 4c.5 workstream uses and a figure without its unit is not checkable.

**The scope of the evidence, stated before the result.** Across all 45 Boltz-2
folds, six have peptide heavy-atom RMSD above 2.0 Å. **All six are the same
complex** — B*07:02/IPRRNVATL, arms A and C, three seeds each. The effective
sample is therefore **one failing complex out of five**, not six independent
failures. Any sentence of the form "the six lowest of 45" reads as six
successes and would overstate this; it is one.

### The within-complex control

This is the argument, and it is the wrong-backbone control from the previous
section made a second way. Hold the allele and the peptide fixed and vary only
the construct:

| B*07:02 IPRRNVATL | mean heavy RMSD | `pep_ll_mean` |
|---|---:|---:|
| Arm A | 2.272 Å | **-3.061** |
| Arm C | 2.251 Å | **-3.076** |
| **Arm B** | **0.299 Å** | **-2.388** |
| (the other four complexes, all 36 folds) | — | -2.513 to -1.881 |

When the *same complex* folds correctly in arm B, its score moves **inside the
good range**. Same allele, same peptide, same model, same seeds. **The score
tracks the pose, not the complex.** That is stronger than any gap statistic
because it holds identity fixed, and it is the same shape of evidence as the
+13.15 → −0.45 margin collapse above: two independent routes to the same
conclusion.

### The claim is specificity, not separation

The naive claim — "the bad folds separate cleanly, with a 0.471 nats/residue
gap" — is **not the right claim, and on its own it would be unsound.** The
stage 4c.5 workstream tested exactly that on its 109 structural features and
found a multiple-comparisons artefact: 2 of 109 features appear to isolate this
failure, but repeating the test on the A+C folds of *every* complex yields
**7, 9, 2, 13 and 2** "fully separating" features respectively. Every complex
is separated by a handful of features, and the genuinely failing one by the
**fewest**. Searching ~100 features for one that splits a six-fold group will
always succeed, so clean separation of that group measures **complex identity,
not pose quality**. Their conclusion: **none of the 109 structural features
isolates the failure**, which extends stage 4c's two-feature finding to all of
them.

Against that backdrop the defensible claim is specificity:

> `pep_ll_mean` flags the one complex that actually folded badly, **with zero
> false positives among the 36 folds of the other four complexes**. It is the
> only readout tested that does so.

Two things make this not the same artefact. First, `pep_ll_mean` was **a single
predeclared quantity** — fixed in the module docstring before any score
existed — not one survivor of a search over ~100 candidates. Second, the
within-complex control above shows the score moving *within* the failing
complex when its pose is fixed, which complex identity cannot explain.

For comparison, on the same 45 folds Boltz-2's own confidence outputs do not
separate the failing folds at all: peptide↔groove PAE bad 1.376–1.552 sits
inside good 1.136–2.196, and peptide pLDDT bad 0.973–0.976 inside good
0.969–0.990. Stage 4c recorded this as an open problem — confidence "does not
isolate the bulge failure... No PAE threshold catches the failure without
flagging accurate predictions."

So the honest description of this feature is:

> **An unsupervised, crystal-free indicator of peptide pose failure in a
> co-folded pMHC complex — demonstrated on one failing complex out of five.**

That is a structural-QC quantity, not a kinetic one. Caveat 2 above still
stands in full. And the limit is load-bearing, not decorative: **one failing
complex is not a general property of the method.** The same limit applies to
the stage 4c.5 negative result, and both write-ups say so.

### How much of the signal could still be artefact

Quantitatively: the between-complex spread of the native score on correct
backbones is small (s.d. 1.98 nats, four of five complexes within -21.1 to
-22.4) compared with the 7.01-nat mean native-vs-foreign margin that the
circularity test exposes. The part that would serve as a *per-pair regression
feature* — how the diagonal value varies from complex to complex — is the small
residual sitting on top of a large, largely constant sequence-recovery effect.
It is 8x the seed noise, so it is real, but it is a thin signal.

And on the label itself: over the five Boltz-2 arm-B complexes,
Spearman(`pep_ll_total`, t½) = **-0.100 (p = 0.87, n = 5, all training rows)**.
That is zero. It carries no inferential weight whatsoever at n = 5 and is
recorded only so that nobody later mistakes its absence for an oversight. The
feature's demonstrated competence is pose QC; its value as a half-life
predictor is **untested**, and this pilot cannot test it.

## Recommendation

1. **Report the ESMFold2 seed-control result and the pose-triage specificity
   result as results in their own right.** Both are verified, both are negative or
   methodological rather than predictive, and the brief values that. The
   ESMFold2 result corroborates stage 4c's crystal-based verdict without using
   a crystal; it does **not** corroborate the stage 4c.5 structural-feature
   seed control, which diverges from it — see that subsection for why the
   divergence is the more informative finding.
2. **Treat `pep_ll_mean` as a structural-QC triage signal**, not as a kinetic
   feature and not as a classifier — a way to rank folds for inspection, which
   stage 4c and the 109 structural features both lack. The caution is
   load-bearing: it is demonstrated on **one failing complex out of five**, so
   it is an encouraging observation and not a calibrated threshold. It must not
   filter the production cohort, and it must not be described as a failure
   detector until it has been checked on more than one failing complex.
3. **Do not buy it as a regression feature.** Decided and recorded above: the
   n=5 label correlation is zero and the usable variation is thin. If it is
   ever revisited, it must be on validation at the same ensemble size and
   tuning budget as every other arm, per `EVALUATION.md`.

## Production scoring: prepared, forecast, not launched

Production structures are **not available** as of this writing:
`reports/ectodomain-20261004/production_{a-cheparukhin,colleague}.jsonl` are
both empty, so no shard has committed, and per stage 4c the outputs stay on the
Modal `pepstab-structures` Volume and are not downloaded.

`modal_app/proteinmpnn_scoring.py` is a **new** file written for this; no
existing file under `modal_app/` was modified. It is **CPU-only by
construction** — no function declares `gpu=` — because the production fold is
using up to 10 concurrent A10Gs per workspace and a GPU container here would
contend for those slots.

**Discovery is shared, not duplicated.** `pepstab.structural_features` already
owns `discover_folds` / `is_excluded`, which skip the `_smoke`, `_shards` and
`_failed` trees that production writes beside the cohort output. Ingesting one
of those raises nothing and produces a row that looks entirely normal, so there
is exactly one implementation of that rule and both this app and the stage 4c.5
extractor use it. `pepstab.inverse_folding.find_predictions` delegates to it,
and `tests/test_inverse_folding.py` asserts all three trees are skipped.

### Forecast

Basis: the pilot's **measured** 14.70 s mean (14.18 s median) per Boltz-2 arm-B
fold at `n_orders=16` on 2 threads — the same 383-residue construct production
folds. Peak RSS 1.2 GiB at `batch_rows=5`. Chunks of 200 folds, `cpu=2.0`,
`memory=4096`, up to 20 containers.

**Rates are the repo's metered figures**, read at runtime from
`reports/ectodomain_rates.json` — the same source the stage 4c forecast used —
not from a pricing page: **$0.04730/core-hour** and **$0.00800/GiB-hour**.

**On checking rates rather than assuming them.** My first forecast used Modal's
published list prices. Checking them against the workspace's metered figures
found they agreed to **0.3%** (CPU 1.0030x, memory 1.0010x) — the check
confirmed the estimate and changed nothing. That is the point worth recording,
because earlier in this project the same check caught an estimate that was off
by **4.6x**. The takeaway a reader should draw is **"check it"**, not "never
trust published rates": the check is nearly free, it usually confirms, and the
one time it did not it caught a 4.6x error. The figures below are the metered
ones regardless, because the cost of using them is zero.

What is measured and what is derived: the **14.70 s/fold and the core-hours are
measured**; the **dollar figures are derived** from measured seconds times
metered rates. Nothing here is a list-price estimate.

| | folds | core-hours | wall | cost |
|---|---:|---:|---:|---:|
| **QC sample, `n_orders=16` (the approved buy)** | **2,000** | **16.5** | **0.8 h** | **$1.04** |
| QC sample, `n_orders=8` | 2,000 | 8.3 | 0.4 h | $0.53 |
| Per profile, `n_orders=16` | 14,083 | 116.0 | 2.9 h | $7.34 |
| Both profiles, `n_orders=16` | 28,166 | 232.0 | 2.9 h in parallel | $14.68 |
| Both profiles, `n_orders=8` | 28,166 | 117.0 | 1.5 h in parallel | $7.39 |

Cost is linear in `n_orders`. The QC sample uses **16** rather than 8 so that
its scores are directly comparable to the pilot reference points, which were
measured at 16; the extra $0.51 buys that comparability.

**Parallelism was raised for wall time only; core-hours and cost are
unchanged.** `CHUNK` 200 → **25** and `MAX_CONTAINERS` 20 → **100**, taking the
QC sample from 10 chunks to 80 and its wall time from ~50 min to ~4–7 min.
Core-hours are fixed by the job, so the only cost change is the extra
per-container startup the forecast charges explicitly: $1.04 → $1.12 on the
conservative 14.70 s/fold basis, or **$0.61 at the 7.5 s/fold actually
measured** in the smokes. **The experiment is untouched** — same 2,000 folds,
same seed, same 16 orders, same draw. `CHUNK` was the binding constraint, not
the container cap: at `CHUNK=200` a 2,000-fold sample is only 10 chunks and
could never have used more than 10 containers however high the cap was set.
`MAX_CONTAINERS=100` has **not** been verified against the workspace CPU quota;
Modal queues rather than failing if the real cap is lower, and the run reports
what it got.

**Verified so far.** `modal 1.6.1` installed (torch 2.14.1 and biotite 1.7.1
unaffected — the ESM-2 arm shares this `.venv`).

| Check | Workspace | Result |
|---|---|---|
| `::forecast` | a-cheparukhin | app deploys, image builds, 5 files uploaded inc. the 6.7 MB checkpoint, both functions created (`ap-H7756iQvG0uxpjoLC1tyBM`) |
| `::score --dry-run --profile a-cheparukhin` | a-cheparukhin | frozen 14,083 and $7.34 confirmed, nothing spawned (`ap-EP5FawBKhdYD3IZhRgS6Ov`) |
| `::forecast --profile colleague` | **sofyaleyn** | **both `list_folds` and `score_chunk` created** (`ap-GUceVVHdiPHrNHQlCz3fFY`) |
| `MODAL_PROFILE`/`--profile` mismatch | — | refuses with `AssertionError`, as designed |

The `sofyaleyn` deploy was the one worth doing early. Stage 4c found that Modal
validates every function at creation time and that a workspace without a
verified payment method cannot declare one at all. That had been ruled out
there for GPU functions but never for this app, and it would otherwise have
surfaced at fold completion with everyone waiting. It passes.

### The `--dry-run` check was assumed free, and was not

Worth stating as a finding rather than describing the fixed behaviour, because
the assumption is the interesting part. `--dry-run` was written, reviewed and
scheduled as a zero-cost check, and both I and the orchestrator referred to it
that way. It was not: it called `list_folds.remote()`, which **starts a
container**, and against a production root that does not yet exist it would
have raised `FileNotFoundError` instead of printing a plan. The "free check we
can run any time" would have failed at exactly the moment it was needed.

Nobody had run it. It was fixed only because it was finally executed, hours
before the window rather than inside it. This is the project's own standing
lesson — *verify a harness measures what it claims* — applied to a harness
written in this report, and it is the second time that discipline has paid out
here (the first being the rate check below). `--dry-run` is now purely local:
it forecasts against the frozen cohort size and never lists the Volume.

### `::smoke` passed in both workspaces — and found two real defects first

Both smokes: **5/5 ok, 0 failed**, 5 real production folds each.
`reports/stage5_inverse_folding_smoke_{a-cheparukhin,colleague}.csv`. Each
workspace lists **14,083 cohort folds and 5 skipped** in `_smoke`/`_shards`/
`_failed` — the shared exclusion logic working on real production output, and
the fold count matching the frozen half exactly.

Measured **7.0–8.0 s per fold**, against the 14.70 s laptop-derived forecast: a
Modal container at `cpu=2.0` is about **2x faster** than my contended laptop,
so the forecast is conservative and the real cost is roughly half.

The invariant earned its keep. `::forecast` and `--dry-run` both passed while
the remote function was **unrunnable**, and only executing it exposed that:

1. **The container died on import.** `_RATES = json.loads((REPO /
   "reports" / ...).read_text())` ran at *module scope*, and Modal imports this
   module inside every container, where the repo does not exist. The container
   raised `FileNotFoundError: '/reports/ectodomain_rates.json'` before running a
   line of its own. Externally this looks like **an app with 0 tasks and no
   other symptom**. This is the same defect stage 4c hit with
   `download_weights`, reproduced independently in a new file — and neither
   local check can catch it, because both run where the file exists. Rates are
   now read lazily. `tests/test_inverse_folding.py` has an AST regression test
   that fails on any module-scope repo read in this file.
2. **Production and pilot metadata have different schemas.** The pilot writes
   `seed` at the top level; production writes it under `settings`. `load_complex`
   raised `KeyError: 'seed'` on all five folds. It now reads both, and production's
   `split` and `shard` are carried through as provenance. (`pair_id` is
   deliberately **not** carried: it is positional into the raw CSV and the
   project invariant forbids joining on it. Joins are on `(allele, peptide)`.)

Neither would have been found by inspection, and both would have fired on the
first chunk of the real run.

`--profile` must match `MODAL_PROFILE`; the runner asserts it, because the two
halves are disjoint. Each half is scored separately and concatenated locally
with a row-count assertion — a half-sized table looks like nothing is wrong.

### Decision: no regression feature; buy the QC sample

**Agreed outcome: do not score the production cohort as a regression feature.**
Spearman(`pep_ll_total`, t½) = -0.100 at p = 0.87 on n = 5 is no evidence of a
half-life feature, and the usable per-pair variation is thin next to the
sequence-recovery effect it rides on. Spending $7–15 to add a feature with no
label correlation would be buying a number to put in a table.

**Agreed outcome: buy the QC sample, ~$1.** Stage 4c recorded that no PAE or
pLDDT threshold catches the pose failure without flagging accurate predictions,
and the project currently has **no cohort-wide way to estimate the pose-failure
rate at all**. Stage 5 requires reporting coverage and outright failures.

### Predeclared QC sample protocol

**Everything in this subsection was fixed and written down before any
production fold was scored, and before any production score was looked at.** It
is mirrored in code as module constants in `modal_app/proteinmpnn_scoring.py`
(`QC_SAMPLE_N`, `QC_SAMPLE_SEED`, `QC_SAMPLE_ORDERS`, `QC_BAD_MAX`,
`QC_GAP_MIDPOINT`, `qc_sample()`), so the draw is reproducible from the repo
rather than from this prose.

| Parameter | Value |
|---|---|
| Sample size | **2,000 folds** |
| Draw | simple random sample **without replacement** over the **sorted** fold paths of one half |
| RNG | `numpy.random.default_rng(20261004)`, `rng.choice(..., replace=False)` |
| Decoding orders | **16** (matches the pilot, so the reference range transfers) |
| Primary reference | `pep_ll_mean <= -2.984` (= `pep_ll_total <= -26.858`) — the *best-scoring* fold of the one known-bad complex |
| Secondary reference | `pep_ll_mean <= -2.749` (= `pep_ll_total <= -24.740`) — midpoint of the observed 0.471 nats/residue gap |

These are **reference points from a single complex, not calibrated
thresholds**, and the report will not call them thresholds.

**The sample must be drawn from a COMPLETE half, not a partial run.** Shards are
ordered by allele, so ten committed shards is five of seventy-five alleles and
any mid-run sample is allele-biased. `qc_sample()` is called only after the
fold count for that profile is checked against 14,083.

**There is no early-diagnostic window.** The two profiles run in parallel at
the same shard count (141 each) and finish within minutes of each other, so one
complete half is not meaningfully earlier than both. The plan is therefore a
single window after completion, sequenced: `::smoke` in each workspace, then
the QC sample, then this report. `::forecast` and `--dry-run` are free and have
already been run (below).

**What will be reported: a distribution and a triage list — never a failure
count or a failure rate.** The deliverable is the distribution of
`pep_ll_mean` over the 2,000 sampled folds. Folds sitting far below the pilot's
good range are reported as **a triage list warranting inspection**, not as
failures: nothing in this pilot can tell a bad pose from an unusual-but-correct
one at the level of an individual production fold.

The one sanctioned sentence form is: *"N of 2,000 sampled folds score in the
range where the one known-bad complex sat, from a single-complex reference."*
The words "failure rate" must not appear. The single-complex provenance travels
with the number wherever it is quoted.

Realised allele coverage will be reported as a diagnostic.

**No row is dropped on this score.** The reference points are descriptive. They
must not filter the production cohort, and nothing downstream may condition on
them.

## Runtime and cost

Everything in this report ran on laptop CPU. **Total cloud spend: $0.**

| Run | Folds | Settings | Workers | Wall |
|---|---:|---|---:|---:|
| Pilot, native only | 90 | 16 orders | 2 | 481 s |
| Cross-peptide, Boltz-2 arm B | 15 | 16 orders, 5 candidates | 1 | 1,107 s |
| Wrong-backbone control, B*07:02 arm A | 3 | 16 orders, 5 candidates | 1 | 107 s |
| Aborted cross-peptide sweep (all Boltz-2 arms) | 7 of 45 | 16 orders, 5 candidates | 2 | ~780 s, discarded |

The aborted sweep is recorded rather than quietly dropped: it was killed under
machine-wide memory pressure and its partial output was not used. The memory
cause was a batch of `orders x candidates = 40` rows on a 383-residue complex;
`batch_rows` now caps total rows per forward pass, and the default was lowered.
`scripts/proteinmpnn_score.py` now also pins all five BLAS thread-pool
environment variables (including `VECLIB_MAXIMUM_THREADS`, which is the one that
matters where numpy links against Accelerate) before importing numpy or torch,
and defaults to 1 worker and 1 thread.

## Artifacts

| Path | Contents |
|---|---|
| `pepstab/inverse_folding.py` | feature definition, mmCIF → ProteinMPNN, scoring |
| `scripts/proteinmpnn_score.py` | CLI: a directory of structures → one tidy table |
| `tests/test_inverse_folding.py` | 13 passing (the two `modal`-gated tests now run) |
| `reports/stage5_inverse_folding_pilot.csv` | 90 rows, native scores |
| `reports/stage5_inverse_folding_crosspeptide_armB.csv` | 15 rows, 5x5 matrix |
| `reports/stage5_inverse_folding_crosspeptide_armA_B0702.csv` | 3 rows, wrong-backbone control |
| `reports/stage5_inverse_folding_*.provenance.json` | commit, checkpoint sha256, settings, wall |
| `modal_app/proteinmpnn_scoring.py` | production entrypoint — **not launched** |
| `external/ProteinMPNN/` | clone at `8907e667` (gitignored) |

Every output table is keyed by `(model, allele, peptide, arm, seed)` and the CLI
takes `--structures <dir>`, so pointing it at production output needs no change
beyond `--exclude _smoke,_shards,_failed`, which `find_predictions` now applies
by default.

## Not done

- **ESM-IF** (`esm.inverse_folding`), the brief's second inverse-folding model.
  Not attempted: the machine was under heavy contention from five other
  workstreams, and the ESM-2 arm has priority on both cores and the shared
  `.venv`. ProteinMPNN alone is sufficient to cover the inverse-folding class.
- **Any validation-split evaluation.** No production structures exist yet, and
  five training complexes cannot support one. The feature has not been tested
  against the sequence baseline.
