# Frozen Boltz-2 pair features improve peptide–HLA stability ranking

**logic binders · London AI × Science · Protein Engineering Track**

Human leukocyte antigen (HLA) class I proteins display short peptides on the cell surface for T-cell recognition. Kinetic stability—how long an assembled peptide–HLA complex remains intact—differs from equilibrium binding affinity and shapes cell-surface persistence. Focusing on nine-residue peptides, we rank candidate peptides by stability within each HLA type (allele).

We reuse internal features from Boltz-2, a biomolecular structure-prediction model, that describe pairs of amino acids in the peptide–HLA complex. Keeping Boltz-2 unchanged, we train small supervised predictors on these features and compare them with sequence-only predictors.

Across **19,716** development pairs and **69** eligible alleles, the selected Boltz-2 pair-feature predictors ranked peptide stability within each allele more accurately than the selected sequence-only predictors. Median within-allele Spearman correlation rose from **0.6668** to **0.7212** (paired difference **+0.0543**, conditional 95% interval **[+0.0314, +0.0808]**). Spearman correlation measures agreement with the measured stability ordering within each allele, not calibrated physical half-life. Under nested cross-validation, model choice and final assessment used separate data splits and evaluated predictions on held-out groups of similar peptides.

## Supporting results

Transfer comparison: median within-allele Spearman correlation across 69 alleles for matched neural predictors.

| Excluded from predictor training | Sequence predictors | Boltz-feature predictors |
| :--- | :---: | :---: |
| Related HLA groups | 0.1605 | **0.5667** |
| Related HLA groups and peptide groups | 0.1242 | **0.4980** |

- **Few-shot adaptation:** Using **20** existing stability measurements per target allele raised the median within-allele Spearman correlation from **0.5296** to **0.5763**, averaged over 3 fixed support-set designs. This separate retrospective analysis used a Boltz-feature ridge model across 68 alleles under joint peptide/HLA holdout. This evaluation uses a different model and query panel than the transfer comparison.
- **Representations and extensions:** Learned pair features outperformed tested chain-average and residue-feature summaries, whereas extra ProteinMPNN/FoldX descriptors and ranking/hybrid heads had no established added benefit.

These retrospective development assessments of one existing assay cohort rank stability measurements rather than calibrated physical half-lives or immunogenicity; independent kinetic validation remains needed.

## Secondary result: sequence-model transfer to ligand presentation

We also tested whether stability predictions distinguish naturally presented peptides (ligands) from comparison peptides (decoys). Using the sequence-only stability ensemble on the MHC Motif Atlas without retraining, we evaluated **81,600** observed nine-residue peptide–HLA pairs across **51** alleles. All stability-dataset peptide sequences were excluded from both the presented-peptide and decoy sets. The ensemble reached a median per-allele AUROC of **0.9656** (AUPRC **0.7804**) against human-proteome decoys and **0.9157** in a harder comparison using ligands observed on other alleles as decoys.

Here, decoys are comparison peptides rather than confirmed nonbinders or non-presented peptides. This benchmark evaluates only the sequence ensemble, not Boltz-2 improvement or independent kinetic validation ([Benchmark details](reports/REPORT.md#sequence-predictions-transfer-to-ligand-presentation)).

## Repository guide

The repository contains source code, saved predictions, numerical result tables, and portable model weights. Large precomputed feature caches are stored externally to Git.

- [Manuscript](reports/MANUSCRIPT.md): Primary scientific narrative and four figures presenting the completed computational benchmark results.
- [Supplementary report](reports/REPORT.md): Supplementary scores, sensitivity analyses, and historical benchmarks.
- [Evidence index](reports/README.md): Numerical evidence index and source table directory.
- [Project page](https://cheparukhin.github.io/peptide-hla-stability/) ([page source](site/index.html)): Visual overview of the findings.
- [Evaluation protocol](EVALUATION.md) and [Dataset and assay](docs/DATASETS.md): Detailed statistical specification, dataset provenance, and split construction.
- [Structural representations](docs/STRUCTURES.md): Construction of Boltz-2 interface representations and structural baselines.
- [Cached-feature inference guide](docs/ROUND2_INFERENCE.md): Guide to running portable prediction heads on compatible cached Boltz-2 features.
- [Independent assay design](reports/finalization/INDEPENDENT_ASSAY_BRIEF.md): Draft design for proposed experimental validation.

## Quick verification

To verify the integrity of all saved full-cohort comparison tables and fixed-query evaluation records locally using only committed result tables, the Python standard library, and Git (without model fitting or feature-cache downloads):

```bash
python3 site/round2-build.py --check-full-cohort
```

To run predictions on cached Boltz-2 trunk features using the released model weights, see [Cached-feature inference guide](docs/ROUND2_INFERENCE.md).
