# Stage 3d, the full-width ESM control, and the 650M check — status

Two long CPU runs completed and are reported here with their confidence
intervals. Stage 3d and the 650M arm are coded but unrun; §3 and §4 record why
and what it costs to finish them.

## 1. Two pipeline defects, both real, neither sufficient alone

An earlier draft of this branch reported ESM-only median per-allele
rho = 0.3866 and seq+ESM 0.5387 against a 0.6904 sequence baseline, and
attributed the gap to the 256-component peptide PCA. A later note on this
branch then retracted that and blamed the head's L2 instead. **Both of those
single-cause stories are wrong.** The full-width run has now finished, and
with upstream's tuned run it fills in three corners of a two-factor grid:

| peptide block | head L2 | ESM-only | ESM+sequence | source |
|---|---|---|---|---|
| 256 PCA components | 1e-5 | 0.3866 | 0.5387 | earlier draft, branch `stage3-esm` |
| **full width (4,394 cols)** | 1e-5 | **0.6087** | **0.6261** | `reports/stage3_summary_esm2_35M_fullwidth.csv` |
| 256 PCA components | 1e-2 | 0.683 | 0.6761 | `reports/stage3_headline.json` (upstream) |
| full width | 1e-2 | not run | not run | the one cheap test left |

Read down the first column: removing the reduction at the bad L2 is worth
**+0.222**, and fixing the L2 at 256 components is worth **+0.296**. Both
knobs were genuinely mis-set, the L2 is the larger of the two, and neither
correction alone reaches the baseline — only upstream's tuned run lands
inconclusive at zero.

With CIs against the 0.6904 baseline, the full-width arms are still
conclusively worse:

| arm | rho | delta | 95% CI | verdict |
|---|---|---|---|---|
| ESM-only, 30 nets | 0.6087 | −0.0817 | [−0.1486, −0.0512] | worse: CI entirely below 0 |
| ESM+sequence, 30 nets | 0.6261 | −0.0643 | [−0.1194, −0.0262] | worse: CI entirely below 0 |

Retained variance is 1.0 by construction (no projection), so this arm has no
information-loss defence left: at `1e-5` the head simply cannot use 4,394
dense columns well, and it cost **19,896 s of fitting** (5.5 CPU-hours for 60
networks) to establish that — against upstream's tuned run, which is both
cheaper and better. Width was not the fix; tuning was.

The transferable lesson, and the reason the remaining grid cell matters less
than it looks: on dense features the per-arm head grid is not optional
politeness, it is what separates a real negative from a manufactured one.
`scripts/esm_arm.py` already says so in its tuning-parity note — dense
standardised features need shrinkage orders of magnitude stronger than sparse
one-hot columns, and transplanting stage 2's ladder onto them "would be a
handicap wearing the costume of fairness". Two independent arms on this branch
proceeded to demonstrate it.

## 1b. BLOSUM62 positional cross-features — the control, with a caveat

`scripts/stage3_blosum_cross.py` finished: 9 x 34 = 306 substitution scores
per pair, no network, 0.69 s to build.

| arm | features | rho | delta vs baseline | 95% CI | verdict |
|---|---|---|---|---|---|
| `blosum_cross_alone` | 306 | 0.3717 | −0.3187 | [−0.3683, −0.2347] | worse: CI entirely below 0 |
| `blosum_cross_stacked` | 1,166 | 0.6277 | −0.0627 | [−0.0908, −0.0088] | worse: CI entirely below 0 |

**This arm is under-regularised in the same way, and its numbers are floors
rather than estimates.** The ladder tried was stage 2's `(1e-5, 1e-3)`;
performance rose monotonically with L2 (alone: 0.292 → 0.336 member median;
stacked: 0.541 → 0.563) and selection landed on `1e-3`, the largest value
tried. A selection at the ladder edge means the ladder was truncated, and
`best_epoch` of 14 and 20 against the baseline's 27 says the heads were still
stopping early. Before this arm is quoted as a control it should be re-run on
the dense ladder `(1e-3, 1e-2, 1e-1)`.

One figure in `reports/stage3e_summary_blosum_cross.csv` should be ignored:
the `delta_vs_esm_stacked` column (+0.0890) compares against the defective
PCA-256 / L2=1e-5 ESM arm from the first row of the table above, so it
measures that arm's mis-tuning, not BLOSUM's merit.

## 2. Stage 3d: code ready, not run

`scripts/stage3d_likelihood.py` is complete and committed. It asks a different
question of the same checkpoint than stage 3 did — not what ESM-2 represents
but what it finds probable — on the hypothesis that a peptide ESM-2 finds
surprising sits in a thinly populated region of sequence space.

Features, cached once per unique peptide by masked marginals (each position
replaced by `<mask>` in turn, 9 forward passes per peptide, 50,697 per
checkpoint over the 5,633 unique peptides):

| feature | count | what it asks |
|---|---|---|
| `pll`, `pll_mean`, `perplexity` | 3 | is a surprising peptide less stable? |
| `logp_P1..P9` | 9 | does position-specific naturalness predict stability? |
| `entropy_P1..P9` | 9 | does residue ambiguity at a position matter? |
| `margin_P1..P9` | 9 | how far is the true residue from ESM-2's preferred one? |

Entropy and margin are computed over the 20 canonical residues after
renormalising, not over ESM-2's full 33-token vocabulary: an entropy that
moved when the model shifted mass onto `<pad>` would not be a statement about
residue preference.

Protocol is inherited from stage 3 unchanged — the same `cv_folds` assignment,
the same seeds, the same boundary-checked L2 ladder `{1e-3, 1e-2, 1e-1}`, the
same paired cluster bootstrap against the stage 2 sequence ensemble, and the
same 30-network count, with the **checkpoint** as the ensemble axis (5 folds x
2 checkpoints x 3 seeds) in place of stage 3's layer pair.

Two correctness guards worth knowing about, because both would otherwise fail
silently and still produce plausible numbers: the token layout is asserted
(token `i+1` must be peptide residue `i`, checked on the first five peptides
before any scoring), and the selected L2 is reported as interior or at the
ladder boundary.

**Predeclared limitation.** ESM-2 scores the peptide without its HLA, so a
null here bounds only *context-free peptide likelihood*. There is also a
specific reason to expect that null: ESM-2 is trained on UniRef, where a 9-mer
is never an entity. Every peptide here is an interior fragment of a longer
protein presented without its flanks, so the masked-marginal distribution is
conditioned on eight neighbours and nothing else.

To run:

```bash
python scripts/stage3d_likelihood.py extract --device cuda --batch-size 2048
python scripts/stage3d_likelihood.py arm
```

## 3. Why the 650M arm is still unrun

Stage 3 dropped ESM-2 650M for memory, and it remains the first question a
reader will ask. It is still unanswered, for two reasons that are both
infrastructure rather than science.

**The rented GPU was refused.** The Modal submit for an A10G returned
`Please add a payment method to use A10G GPU sandboxes` — an account-level
refusal, independent of credit balance. No GPU tier will start on the
connected workspace until that is resolved.

**The local attempt was mis-scheduled, and that is on me.** `pick_device` now
prefers CUDA under `auto` and the extraction CLI accepts `--device cuda`, so
the code is GPU-ready. The local run produced no cache file in four hours and
was killed at 93 MB free memory and a load average of 11.9 — but it had been
sharing 8 cores and 8 GiB with the BLOSUM cross-feature ensemble (4,461 s of
fitting) and the tail of the full-width run (19,896 s) for most of that
window. So this is **not** evidence that 650M cannot extract on this machine;
it is evidence that three CPU-saturating jobs cannot. The 35M checkpoint peaks
at 726 MB (`reports/stage3_embedding_cost.csv`) and 650M was measured at
5.06 GB on a 16 GiB machine, so a serial attempt here would be tight but is
untested. A GPU remains the right answer; a quiet machine is the fallback.

This is a genuine resource bound, not a result. The extraction itself is
small — 5,633 peptides and 75 HLA domains, under two minutes on an A10G — so
the arm is roughly fifteen minutes of GPU time whenever one is available:

```bash
python scripts/esm_features.py --checkpoint esm2_t33_650M_UR50D --device cuda
python scripts/esm_arm.py grid --checkpoint esm2_t33_650M_UR50D \
    --pep-rep pos --hla-rep contact --layers mid final
python scripts/esm_arm.py ensemble --checkpoint esm2_t33_650M_UR50D \
    --pep-rep pos --hla-rep contact --axis layer --layers mid final \
    --name esm650_ensemble
```

`run_gpu.sh` (workspace root, not committed) runs the 650M arm and both
likelihood checkpoints in sequence, writing each step's results as it
finishes so a deadline returns partial output rather than nothing.

## 4. Cross-embeddings: not started

Feeding peptide and HLA through ESM-2 together, so self-attention can see the
interaction, is the one genuinely novel arm and no code exists for it yet. It
is also the most expensive: one forward pass per *measured pair* rather than
per unique sequence, so 28,166 passes of a 46-residue concatenation against
5,708 short ones — roughly a 40x increase in extraction cost over stage 3.
That is minutes on a GPU and hours here, which is why it was not attempted.

If it is attempted, the BLOSUM62 positional cross-feature control
(`scripts/stage3_blosum_cross.py`, 9 x 34 = 306 features, no network) is the
comparison that matters, not the sequence baseline alone: if ESM-2
cross-features do not beat substitution compatibility, the attention learned
nothing beyond it.
