# Stage 3: frozen ESM-2 features

Answers the track's question on the peptide axis: **frozen ESM-2 features do not
reach the stage 2 sequence baseline, and do not add to it.** Branched on `main`,
independent of the companion `analysis/biology-probes` PR.

## Result

Validation split, 68 eligible alleles, scored through `pepstab.evaluation`
against the stage 2 30-network ensemble:

| model | median per-allele rho | delta | 95% CI | verdict |
|---|---:|---:|---|---|
| sequence ensemble (stage 2) | **0.6904** | — | — | — |
| ESM alone, 30 nets | 0.3866 | −0.3038 | [−0.359, −0.220] | worse |
| ESM stacked on baseline, 30 nets | 0.5387 | −0.1517 | [−0.205, −0.089] | worse |

ESM-2 35M, peptide and HLA embedded separately, matched to stage 2 on splits,
folds, seeds, architecture, tuning budget and ensemble size (5 CV folds x 2
layers x 3 seeds = 30 networks, the two layers playing the role the two
encodings play in the baseline). Layers 6 and 12 are indistinguishable at member
level: 0.244 vs 0.266 standalone, 0.399 vs 0.395 stacked, against stage 2's
recorded seed spread of 0.010–0.051.

## What is new in `pepstab/`

- `attn.py` — cross-attention head: the 9 peptide positions as queries over the
  34 HLA contact residues. The ablation is the same module with attention
  replaced by a uniform average, at an **identical parameter count** (94,913
  both arms), so a gap isolates coupling from capacity.

## Reviewer notes

**Run `scripts/baseline_ensemble.py` first.** `preds/` is gitignored, so the
comparator is not in the repo and every delta here is measured against a file a
fresh clone does not have. Rebuilding it reproduced stage 2 within rounding —
median per-allele rho 0.6904 against the reported 0.693, ensembling gain +0.093
against +0.090 — which is also an independent check on the stage 2 numbers.

**Residue indexing is tested, not assumed.** Recovering the pseudosequence-to-domain
mapping from the data alone is underdetermined: with only 75 alleles, several
conserved domain columns match a given pseudosequence column by chance, and the
first implementation of this correctly refused to guess. The published NetMHCpan
positions are tested instead, and land at index p−1, exact on all 75 alleles x
34 columns.

**Weights are not committed.** `models/` is gitignored; pass a Hugging Face id
(`--model facebook/esm2_t12_35M_UR50D`).

**Layout follows stages 1 and 2**, not the `results/esm_stacking/` layout in the
task brief: `results/` is gitignored here and `scripts/evaluate.py` resolves
prediction paths relative to `preds/`, so a split tree would break the shared
evaluator.

## What this does not claim

- **The stacked arm's −0.15 is not shown to be damage.** 330 extra columns under
  a fixed budget, and the ESM arms stop after ~5 epochs against the baseline's
  ~27, which points at capacity displacement rather than contradictory
  information. A narrowed-block run separates the two and is not done.
- **A larger checkpoint is not the obvious next step.** The failure mode is
  displacement under a fixed budget, so a 1,280-dimensional embedding built the
  same way displaces more. Worth paying for only if the coupling or
  cross-feature arm shows representation quality is the binding constraint.
- **Scope is new peptides on familiar grooves.** All 75 alleles are in training.
  Nothing here bounds what a pretrained model does on an unseen allele.
- **The benchmark is not saturated**, so this is a statement about the features
  rather than about a ceiling — see the companion PR.

## Follow-up commit on this branch

`scripts/stage3_coupling.py` (coupling vs the parameter-identical ablation) and
`scripts/stage3_blosum_cross.py` (BLOSUM 9x34 positional cross-features, the
no-network control) are committed and running. They are the two tests that could
overturn the conclusion above: the first asks whether concatenation, rather than
ESM, is the limitation; the second asks whether *any* positional cross-encoding
helps this baseline, at ten CPU-minutes. Per-layer ensemble rows land with them.
