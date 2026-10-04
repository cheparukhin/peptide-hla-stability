# Stage 5 stretch arm: ProteinMPNN inverse folding

4 October 2026. **Status: pilot complete on all 90 stage 4c folds. The seed
control passes for Boltz-2 and fails for ESMFold2. The circularity control came
back with a result I did not expect, and it changes what this feature is for:
the ProteinMPNN peptide log-likelihood works as a crystal-free detector of
Boltz-2 pose failure, and there is no evidence from this pilot that it predicts
half-life. Production scoring is prepared but not launched.**

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
complexes, the feature is noise. Ratio of between-complex s.d. to mean
within-complex seed s.d. of `pep_ll_total`:

| Model | Arm A | Arm B | Arm C |
|---|---:|---:|---:|
| Boltz-2 | **10.0** | **8.0** | **6.8** |
| ESMFold2 | 1.1 | 2.4 | 1.2 |

**Boltz-2 passes.** Arm B — the frozen production configuration — is 8.0x, with
mean seed s.d. 0.248 nats against a between-complex s.d. of 1.98 nats. Worst
seed range on any Boltz-2 arm-B complex is 0.97 nats.

**ESMFold2 fails**, sitting at the noise floor (1.1–2.4x), with seed ranges up
to 7.3 nats. It is worth being explicit about what this is: **an independent
corroboration of the stage 4c production decision, reached by a completely
different measurement.** Stage 4c rejected ESMFold2 on peptide heavy-atom RMSD
against crystal structures. This measurement uses no crystal structure at all —
it asks a third-party model whether the predicted backbone is self-consistent
with the sequence that produced it — and reaches the same verdict. A rejection
reached twice by unrelated routes is much stronger than either route alone.

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

Across all 45 Boltz-2 folds, six have peptide heavy-atom RMSD above 2.0 Å: the
three arm-A and three arm-C folds of B*07:02/IPRRNVATL. Ranking all 45 by
`pep_ll_total`, **the six lowest are exactly those six folds**:

| | range of `pep_ll_total` | clean separation? |
|---|---|---|
| 6 bad folds (RMSD > 2 Å) | -28.62 to -26.86 | — |
| 39 good folds | -22.62 to -16.93 | **yes, 4.24-nat gap** |
| Boltz-2 peptide↔groove PAE | bad 1.376–1.552 inside good 1.136–2.196 | **no, fully overlapping** |
| Boltz-2 peptide pLDDT | bad 0.973–0.976 inside good 0.969–0.990 | **no, fully overlapping** |

This lands exactly on an open problem stage 4c recorded. That report concluded:
confidence "does not isolate the bulge failure... No PAE threshold catches the
failure without flagging accurate predictions", and recommended keeping
confidence features but *not* using them as a per-prediction failure filter.
ProteinMPNN's peptide log-likelihood does what PAE and pLDDT could not, on the
same 45 folds, and it needs no crystal structure to do it.

So the honest description of this feature is:

> **An unsupervised, crystal-free detector of peptide pose failure in a
> co-folded pMHC complex.**

That is a structural-QC quantity, not a kinetic one. Caveat 2 above still
stands in full.

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

1. **Report the ESMFold2 corroboration and the pose-failure filter as results
   in their own right.** Both are independently verified, both are negative or
   methodological rather than predictive, and the brief values that.
2. **Treat `pep_ll_total` as a structural-QC covariate**, not as a kinetic
   feature — a per-prediction flag for "Boltz-2 probably got this peptide
   wrong", which stage 4c explicitly lacked. One caution: it is validated
   against a *single* failure mode on *five* training complexes. A 4.24-nat gap
   on 45 folds of one failure is an encouraging observation, not a calibrated
   threshold, and it must not be used to filter the production cohort until it
   has been checked on more than one kind of failure.
3. **Only then consider it as a regression feature**, on validation, at the same
   ensemble size and tuning budget as every other arm, per `EVALUATION.md`.

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

| | folds | core-hours | wall at 20 containers | cost |
|---|---:|---:|---:|---:|
| Per profile, `n_orders=16` | 14,083 | 116 | 2.9 h | **$7.32** |
| Both profiles, `n_orders=16` | 28,166 | 232 | 2.9 h in parallel | **$14.65** |
| Both profiles, `n_orders=8` | 28,166 | 117 | 1.5 h in parallel | **$7.39** |

Cost is linear in `n_orders`. At 8 orders the decoding-order standard error
rises from 0.12 to 0.17 nats, still an order of magnitude below the 1.98-nat
between-complex spread, so **8 orders is the better buy** if this runs at all.

Two honest qualifications on these numbers:

- **The dollar figures use Modal's published list rates**
  ($0.0000131/core·s CPU, $0.00000222/GiB·s memory), which this repo has not
  measured and which are not verified here. The core-hours are measured; the
  dollars are not. Confirm the rates before launching.
- **The `modal` package is not installed in this `.venv`**, so this entrypoint
  has not been dry-run, let alone smoke-tested. Per the project invariant it
  must pass its `::smoke` on 5 real folds in each workspace before any full
  pass, and `::forecast` and `--dry-run` spawn nothing and should be run first.

`--profile` must match `MODAL_PROFILE`; the runner asserts it, because the two
halves are disjoint. Each half is scored separately and concatenated locally
with a row-count assertion — a half-sized table looks like nothing is wrong.

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
| `tests/test_inverse_folding.py` | 11 passing, 1 skipped (needs `modal`) |
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
