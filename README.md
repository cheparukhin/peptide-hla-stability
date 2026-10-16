# Frozen Boltz-2 pair features improve peptide–HLA stability ranking

**logic binders · London AI × Science · Protein Engineering Track**

Human leukocyte antigen (HLA) class I proteins display short peptides on the cell surface for T-cell recognition. Kinetic stability—how long an assembled peptide–HLA complex remains intact—differs from equilibrium binding affinity and shapes cell-surface persistence. Focusing on nine-residue peptides, we rank candidate peptides by stability within each HLA type (allele).

We reuse internal features from Boltz-2, a biomolecular structure-prediction model, that describe pairs of amino acids in the peptide–HLA complex. Keeping Boltz-2 unchanged, we train small supervised predictors on these features and compare them with sequence-only predictors.

Across **19,716** development pairs and **69** eligible alleles, the selected Boltz-2 pair-feature predictors ranked peptide stability within each allele more accurately than the selected sequence-only predictors. Median within-allele Spearman correlation rose from **0.6668** to **0.7212** (paired difference **+0.0543**, conditional 95% interval **[+0.0314, +0.0808]**). Spearman correlation measures agreement with measured stability orderings, not calibrated physical half-life. Under nested cross-validation, model choice and final assessment used separate data splits and evaluated held-out groups of similar peptides.

## Supporting results

Transfer comparison: median within-allele Spearman correlation across 69 alleles for matched neural predictors.

| Excluded from predictor training | Sequence predictors | Boltz-feature predictors |
| :--- | :---: | :---: |
| Related HLA groups | 0.1605 | **0.5667** |
| Related HLA groups and peptide groups | 0.1242 | **0.4980** |

- **Few-shot adaptation:** Adding **20** existing stability measurements per target allele raised the median within-allele Spearman correlation from **0.5296** to **0.5763** (panel medians averaged across 3 fixed support-set designs). Unlike the transfer comparison above, this retrospective test used a Boltz-feature ridge model and a different query panel across 68 alleles under joint peptide/HLA holdout.
- **Representations and extensions:** Learned pair features outperformed tested chain-average and residue-feature summaries, whereas extra ProteinMPNN/FoldX descriptors and ranking/hybrid heads had no established added benefit.

These retrospective assessments of one existing assay cohort rank stability measurements rather than calibrated physical half-lives or immunogenicity; independent kinetic validation remains needed.

## Secondary result: sequence-model transfer to ligand presentation

We also tested whether stability predictions distinguish naturally presented peptides (ligands) from comparison peptides (decoys). Using the sequence-only stability ensemble on the MHC Motif Atlas without retraining, we evaluated **81,600** observed nine-residue peptide–HLA pairs across **51** alleles. All stability-dataset peptide sequences were excluded from both the presented-peptide and decoy sets. The ensemble reached a median per-allele AUROC of **0.9656** (AUPRC **0.7804**) against human-proteome decoys and **0.9157** in a harder comparison using ligands observed on other alleles as decoys.

Decoys are not experimentally confirmed nonbinding or non-presented peptides. Only the sequence ensemble was tested, so the elution result does not establish a Boltz-2 stability advantage or independent kinetic validation ([Benchmark details](reports/REPORT.md#sequence-predictions-transfer-to-ligand-presentation)).

## Repository guide

The repository contains source code, saved predictions, numerical result tables, and portable model weights. Large precomputed feature caches are stored outside Git.

- [Manuscript](reports/MANUSCRIPT.md): Methods, benchmark results, and four summary figures.
- [Supplementary report](reports/REPORT.md): Supplementary scores, sensitivity analyses, and historical benchmarks.
- [Evidence index](reports/README.md): Index of numerical results and source tables.
- [Project page](https://cheparukhin.github.io/peptide-hla-stability/) ([page source](site/index.html)): Visual summary of the findings.
- [Evaluation protocol](EVALUATION.md) and [Dataset and assay](docs/DATASETS.md): Statistical specification, dataset sources, and split construction.
- [Structural representations](docs/STRUCTURES.md): Boltz-2 pair-feature definitions, structure provenance, and structural baselines.
- [Cached-feature inference guide](docs/ROUND2_INFERENCE.md): How to run the portable prediction heads on cached Boltz-2 features.
- [Independent assay design](reports/finalization/INDEPENDENT_ASSAY_BRIEF.md): Draft protocol for proposed experimental validation.

## Quick verification

Using only the Python standard library and Git (no feature-cache downloads), this command verifies that the generated website, its evidence snapshot, and the copied figure match the saved results:

```bash
python3 site/round2-build.py --check
```

To run predictions on cached Boltz-2 trunk features using the released model weights, see [Cached-feature inference guide](docs/ROUND2_INFERENCE.md).
