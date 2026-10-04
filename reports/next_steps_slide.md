# Deck slide — next steps

Slide source for reporting. One slide, version 1. Reconciled against stage 5:
the original draft of this slide was written before the production fold landed
and framed the structural result as a smoke run with geometry untested. It is
not. See [../NEXT_STEPS.md](../NEXT_STEPS.md) for the full reasoning and for
the register of steps that are **already closed**.

---

## pMHC-I stability: the structural negative is well-powered, and bounded

**Finding.** Boltz-2 co-folded structures do not carry half-life signal beyond
sequence, and adding them **hurts**. The fold succeeded completely — 28,166 of
28,166 pairs, 100% coverage, zero fallback — so this is a result about
structure, not about failed folding. Against the sequence ensemble at **0.693**:
seq+geometry **−0.0788**, seq+confidence **−0.0973**,
seq+geometry+confidence **−0.0715**; every interval entirely below zero. A
`seq_only` control through the same pipeline pins it to the features, not
tuning. **Neither confidence nor coarse geometry is the culprit alone — none of
the 109 structural features isolates the failure.**

**What it does not establish.** That structure is uninformative. The pairformer
trunk's internal pair representation was never read; stage 5 used aggregates
over the *output* pose. That is the one remaining structural question.

| Priority | Next step | Question answered |
|---|---|---|
| **1. Read the trunk, not the pose** | Extract peptide × groove pair representations from the final 1–2 pairformer blocks; evaluate as its own arm. Folds are on disk — extraction plus a head, no new production fold. Must clear the same comparator, folds, matched head/ensemble/tuning budget and a `seq_only` control. | Does the learned representation carry what 109 pose aggregates lost? |
| **2. Close the pLM holes the register already names** | Chimeric peptide–linker–α1α2 ESM input (promoted, never run); a second family (ESM-C / ProtT5) with a **layer sweep**; stage 3d likelihood features. Features fixed when comparing concatenation vs cross-attention. | Is the stage 3 null family-specific, or about separate encoding? |
| **3. Change the target, not the model** | Between-allele Δlog(t½) for shared peptides; stability **residuals conditional on affinity** on the 7,281 matched pairs. No new data needed. | Can we isolate groove-dependent effects and signal beyond affinity? |
| **4. Buy the missing experiment** | Stability pilot: a few hundred peptides, 8–10 divergent alleles. Replicates retained, broad peptide sampling, full curves with explicit censoring, matched affinity, WT/pocket-mutant panels. | Does this generalise to genuinely unfamiliar grooves? |
| **5. Publish the bounded result** | Report Δρ with CIs, ablations and cost per arm; extend elution validation to the ESM-2 and structural arms; release the template table, affinity reference and elution evaluation. | What stays useful regardless of model outcome? |

**First move: the trunk pair representation** — the only structural feature
class whose failure mode is still untested, with co-folding already paid for.
It must clear a high bar: stage 5 is a well-powered negative on the same
structures, so a positive here would be the lone dissenting result.

**Data strategy: stability over more elution.** We hold 487,343 elution rows
and stage 3c shows they do not substitute for kinetics. Two gaps no compute
closes: median nearest-neighbour pseudosequence identity is **0.941** (the
alleles are near-duplicates, and the allele-hold-out confound), and the dataset
ships **no replicates**, so **no noise ceiling can be estimated** — we cannot
say how much residual error is irreducible. Shortlist immunAware and ProImmune
for quotes; verify coverage, curve export, replicate handling and assay limits.
Vendor capabilities and pricing are not independently verified.

---

> **Every number on this slide is a validation number.** The frozen test split
> is scored once, at stage 6, under
> [TEST_SCORING_RUNBOOK.md](TEST_SCORING_RUNBOOK.md).

**Sources.** Stage 5 figures from the stage 5 structural tables
(`stage5_structural_bootstrap.csv`, `stage5_structural_interior.csv`,
`stage5_structural_selected.json`); baseline 0.693 from
[stage2_baselines.md](stage2_baselines.md); 0.941 and 20.2% computed from
`data/rasmussen_et_al_dataset.csv`; 7,281 matched pairs from
[../docs/AFFINITY_REFERENCE.md](../docs/AFFINITY_REFERENCE.md); elution rows
from [stage3c_elution_validation.md](stage3c_elution_validation.md).
