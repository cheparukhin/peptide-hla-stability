# Stage 3d and the 650M completeness check — status

**No new validation numbers are reported here.** This file records one
correction to the stage 3 interpretation, the code that is ready to run, and
the resource wall that stopped it, so that whoever next has a GPU can produce
the results in minutes rather than re-deriving the setup.

## 1. Correction: stage 3's ESM arm was not hurt by the PCA

An earlier draft of this branch reported ESM-only median per-allele
rho = 0.387 and seq+ESM 0.539, and attributed the gap to the 256-component
peptide PCA being lossy. **That attribution was wrong**, and the headline
numbers in it should not be quoted.

`reports/stage3_headline.json` records the team's stage 3 run at the *same*
representation — peptide per-position, 256 PCA components, middle layer, 34
HLA contact positions, 74 HLA components — and scores **0.683**. The two runs
differ in one place: the L2 on the MLP head. The earlier draft inherited
stage 2's `1e-5`; stage 3 selected `1e-2` from a boundary-checked ladder.

So the defect was **under-regularisation, not dimensionality reduction**. The
supporting evidence is internally consistent: at `1e-5` the ESM heads stopped
at `best_epoch` around 5 against the baseline's 27, and the seed spread was
0.061 median and 0.184 maximum against stage 2's 0.010–0.051. Those are the
signatures of a head diverging early on dense inputs, which is exactly what
`scripts/esm_arm.py` documents in its tuning-parity note: dense standardised
features need shrinkage orders of magnitude stronger than sparse one-hot
columns, and transplanting stage 2's ladder onto them "would be a handicap
wearing the costume of fairness".

The practical lesson for any further arm on dense features: the per-arm head
grid is not optional politeness, it is what separates a real negative from a
manufactured one.

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

**The local CPU cannot carry it.** `pepstab.esm.pick_device` now prefers CUDA
under `auto` and the extraction CLI accepts `--device cuda`, so the code is
GPU-ready. But run on this machine — 8 cores, 8 GiB — the 650M extraction
produced no cache file in four hours and was killed: 93 MB of free memory and
a load average of 11.9 means it was paging, not computing. The 35M checkpoint
peaks at 726 MB (`reports/stage3_embedding_cost.csv`); 650M was measured at
5.06 GB on the 16 GiB laptop that originally rejected it, which does not fit
here alongside anything else.

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
