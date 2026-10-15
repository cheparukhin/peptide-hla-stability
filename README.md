# Frozen Boltz-2 pair features improve peptide–HLA stability ranking

**logic binders · London AI × Science · Protein Engineering Track**

Human leukocyte antigen (HLA) class I proteins present short peptides at the cell surface for T-cell recognition. Kinetic stability—how long an assembled complex remains intact—differs from equilibrium affinity and shapes cell-surface persistence. Focusing on nine-residue peptides, we compare small supervised predictors using frozen Boltz-2 residue-pair features against sequence models for stability ranking.

Across **19,716** development pairs and **69** eligible alleles, our Boltz-2 pair-feature selector raised median within-allele Spearman correlation from **0.6668** (competitive sequence selection) to **0.7212** (paired difference **+0.0543**, conditional 95% interval **[+0.0314, +0.0808]**). Under nested cross-validation, inner folds selected models while held-out peptide clusters were assessed separately; Spearman measures agreement with measured ordering within each allele, not calibrated half-life.

## Supporting results

Using matched nonlinear predictors (MLPs) across 69 alleles, Boltz-2 pair features generalized better than sequence inputs when related HLA groups were excluded from supervised fitting:

| Supervised holdout (69 alleles, matched MLPs) | Sequence MLP | Boltz-2 pair MLP |
| :--- | :---: | :---: |
| HLA-group holdout | 0.1605 | **0.5667** |
| Joint HLA-group/peptide-cluster holdout | 0.1242 | **0.4980** |

- **Few-shot adaptation:** In a separate retrospective linear ridge study across 68 alleles under joint holdout, using **20** existing target labels per allele raised the median within-allele Spearman correlation from **0.5296** to **0.5763**, averaged over 3 fixed support-set designs. This evaluation uses a different model and query panel than the transfer comparison.
- **Representations and extensions:** Learned pair features outperformed tested chain-average and residue-feature summaries, whereas extra ProteinMPNN/FoldX descriptors and ranking/hybrid heads had no established added benefit.

These retrospective development assessments of one existing assay cohort rank stability measurements rather than calibrated physical half-lives or immunogenicity; independent kinetic validation remains needed.

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
