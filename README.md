# Predicting Peptide–HLA Class I Kinetic Stability with Structural Foundation Model Representations

[**Project Page**](https://cheparukhin.github.io/peptide-hla-stability/) · [**Full Paper (`reports/REPORT.md`)**](reports/REPORT.md) · [**Evaluation Protocol (`EVALUATION.md`)**](EVALUATION.md) · [**Dataset & Assay (`docs/DATASETS.md`)**](docs/DATASETS.md) · [**Model Inference (`docs/ROUND2_INFERENCE.md`)**](docs/ROUND2_INFERENCE.md)

---

## Overview

Effective T-cell recognition requires a peptide–HLA class I (pMHC-I) complex not only to bind in equilibrium ($K_d = k_{\text{off}}/k_{\text{on}}$), but to remain bound at the cell surface long enough to engage surveying T-cell receptors. Complex dissociation half-life ($t_{1/2} = \ln 2 / k_{\text{off}}$) is therefore a distinct kinetic determinant of immunogenicity. Across 7,281 peptide–HLA pairs with both measurements, experimental binding affinity correlates with kinetic half-life at a median within-allele Spearman $\rho$ of only $0.580$.

This repository investigates a central question in AI for structural immunology: **when and why do biological foundation models outperform sequence-only predictors for peptide–HLA kinetic stability?**

Using the Rasmussen et al. (2016) scintillation proximity assay dataset (**28,166** 9-mer peptide–HLA class I complexes across **75** HLA alleles and **5,633** peptides) and an external immunopeptidomics benchmark from the MHC Motif Atlas (**897,600** evaluated pairs across **51** alleles), we establish four principal findings:

1. **Sequence ensembles provide a strong baseline and transfer zero-shot to natural ligand presentation, whereas naive foundation-model summaries fail.**  
   On the held-out peptide-cluster test split (5,565 pairs, 67 alleles), a 30-model multilayer perceptron (MLP) ensemble taking the 9-mer peptide and 34-residue HLA contact pseudosequence achieves a median within-allele Spearman $\rho = \mathbf{0.7064}$, outperforming a single network by $+0.0883$ ($\rho = 0.6181$) and matching a full 182-residue HLA domain ensemble ($\rho = 0.6904$). Evaluated zero-shot on 81,600 MS-eluted ligands against 816,000 human proteome decoys across 51 alleles, this stability predictor reaches a median **AUROC of 0.9656** (AUPRC $0.7804$), retains **0.9157 AUROC** against allele-swapped eluted-ligand decoys that control for abundance, proteasomal cleavage, and TAP transport, and collapses to **0.6965** when HLA pseudosequences are permuted. Conversely, frozen **ESM-2** embeddings tie the sequence baseline ($\rho = 0.6979$, $\Delta\rho = -0.0085$), and compressing 28,166 predicted **Boltz-2** 3D structures (143.6 A10G GPU-hours) into hand-engineered pocket geometry and confidence summaries degrades ranking to $\rho = \mathbf{0.5947}$ ($\Delta\rho = -0.1117$).

2. **Internal position-aware pair representations from the frozen Boltz-2 trunk outperform competitive sequence predictors.**  
   The failure of coarse structural summaries stems from pooling away residue-level interface organization. Extracting position-preserving peptide–groove pair representations directly from the frozen Boltz-2 trunk (`R4` and `R5`) substantially improves ranking. In nested 5-fold peptide-cluster cross-validation across the 19,716-pair training cohort (3,587 peptide clusters, 69 eligible alleles), the pre-specified Boltz-2 embedding selector achieves $\rho = \mathbf{0.7212}$ compared with $\rho = \mathbf{0.6668}$ for a competitive sequence selector spanning standard and allele-centered MLP ensembles, gradient-boosted trees, and pairwise rankers (**$\Delta\rho = +0.0543$**, 95% CI $[+0.0314, +0.0808]$). Retaining all $9 \times 34$ peptide–groove pair tokens (`full_interface_pos_mlp`) reaches **0.7315**, and blending a linear `R4` ridge head with the sequence ensemble reaches **0.7238** ($\Delta\rho = +0.0569$).

3. **Positional anchor resolution—not a specific crystal contact mask—drives the structural representation gain.**  
   Controlled ablations show that preserving individual peptide positions outperforms dimension-matched pooled trunk representations by **$+0.3387$** ($[+0.3029, +0.3773]$) in projected token space (`R3`) and **$+0.1663$** ($[+0.1378, +0.1992]$) across the full $9 \times 34$ interface. By contrast, pooling HLA residues via a crystal-derived peptide–groove contact mask (`R5`, $\rho = 0.7210$) provides no advantage over randomly shuffled HLA contact subsets ($\rho = 0.7203$, $\Delta\rho = +0.0007$ $[-0.0096, +0.0236]$).

4. **Boltz-2 interface representations generalize across unseen HLA groups, reduce label requirements, and support 20-shot allele calibration.**  
   When entire HLA sequence groups ($\le 3$ Hamming distance on the 34-residue contact pseudosequence) are held out from supervised fitting, sequence predictors collapse ($\rho = 0.1605$ for matched MLPs; $0.1921$ for the sequence selector), whereas Boltz-2 `R5` representations achieve $\rho = \mathbf{0.5667}$ (**$\Delta\rho = +0.4062$**, 95% CI $[+0.3073, +0.4733]$). Under simultaneous peptide-cluster and HLA-group holdout, matched MLPs improve from $0.1242$ to $\mathbf{0.4980}$ (**$\Delta\rho = +0.3738$** $[+0.3092, +0.4638]$). Calibrating a frozen `R5` ridge head with just **20 measured peptides** from a held-out allele lifts ranking from $\mathbf{0.5296}$ to $\mathbf{0.5763}$ (**$\Delta\rho = +0.0468$**, 95% CI $[+0.0227, +0.1035]$, 98.33% Bonferroni CI $[+0.0140, +0.1195]$), driven by feature-weight rotation ($+0.0444$) rather than an additive allele offset ($+0.0024$).

In secondary investigations, adding inverse-folding log-likelihoods (**ProteinMPNN**; full-cohort $\Delta\rho = -0.0070$), empirical energy terms (**FoldX**; panel $\Delta\rho = +0.0150$), within-allele **RankNet** objectives ($\Delta\rho = +0.0071$), or hybrid tree/neural selectors ($\Delta\rho = +0.0044$) provided no established improvement beyond `R5` trunk features. On the sequence baseline, left-censored **Tobit regression** with MSE early stopping halved the Brier calibration error for the 20.2% assay-floor observations ($\Delta\text{Brier} = +0.0165$) without degrading rank correlation.

---

## Summary of Principal Results

| Experimental regime | Evaluation cohort | Baseline | Candidate | Primary metric | Effect [95% CI] |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Test-split baseline & naive FMs** | 5,565 pairs · 67 alleles (test split) | 30-MLP sequence ensemble ($\rho = 0.7064$) | Single MLP ($\rho = 0.6181$)<br>ESM-2 35M ($\rho = 0.6979$)<br>Boltz-2 coarse geometry ($\rho = 0.5947$) | Median within-allele Spearman $\rho$ | $-0.0883$ $[-0.1201, -0.0555]$<br>$-0.0085$ $[-0.0291, +0.0233]$<br>$-0.1117$ $[-0.1339, -0.0575]$ |
| **Zero-shot ligand elution transfer** | 81,600 eluted ligands vs 816,000 decoys · 51 alleles | Random / permuted-HLA ($0.4974$ / $0.6965$) | 30-MLP stability ensemble (proteome / allele-swapped decoys) | Median per-allele AUROC | **0.9656** $[0.9435, 0.9779]$ (IQR)<br>**0.9157** $[0.8598, 0.9521]$ (IQR) |
| **Nested peptide-cluster holdout** | 19,716 pairs · 3,587 clusters · 69 alleles | Competitive sequence selector ($\rho = 0.6668$) | Boltz-2 trunk embedding selector ($\rho = 0.7212$) | Median within-allele Spearman $\rho$ | **$+0.0543$** $[+0.0314, +0.0808]$ |
| **Sequence + trunk complementarity** | 19,716 pairs · 3,587 clusters · 69 alleles | Competitive sequence selector ($\rho = 0.6668$) | Linear blend: sequence ensemble + `R4` ridge ($\rho = 0.7238$) | Median within-allele Spearman $\rho$ | **$+0.0569$** $[+0.0426, +0.0838]$ |
| **Positional vs pooled ablation** | 19,716 pairs · 3,587 clusters · 69 alleles | Pooled trunk tokens (`R3` pooled $\rho = 0.3165$) | Position-preserving trunk tokens (`R3` $\rho = 0.6552$) | Median within-allele Spearman $\rho$ | **$+0.3387$** $[+0.3029, +0.3773]$ |
| **Unseen HLA-group transfer** | 5 HLA folds $\times$ 5 peptide folds · 69 alleles | Matched sequence MLP `R0` ($\rho = 0.1605$) | Boltz-2 trunk MLP `R5` ($\rho = 0.5667$) | Median within-allele Spearman $\rho$ | **$+0.4062$** $[+0.3073, +0.4733]$ |
| **Joint peptide $\times$ HLA holdout** | 5 HLA folds $\times$ 5 peptide folds · 69 alleles | Matched sequence MLP `R0` ($\rho = 0.1242$) | Boltz-2 trunk MLP `R5` ($\rho = 0.4980$) | Median within-allele Spearman $\rho$ | **$+0.3738$** $[+0.3092, +0.4638]$ |
| **20-shot allele adaptation** | 68 alleles · 3 support designs (joint holdout) | Zero-shot `R5` ridge ($\rho = 0.5296$) | 20-peptide adapted `R5` ridge ($\rho = 0.5763$) | Median within-allele Spearman $\rho$ | **$+0.0468$** $[+0.0227, +0.1035]$ |

---

## Repository Guide

| Path | Contents |
| :--- | :--- |
| [`reports/REPORT.md`](reports/REPORT.md) | Complete scientific paper covering experimental design, main results, ablations, biological transfer, secondary studies, and limitations |
| [`site/index.html`](site/index.html) | Interactive project page and visual summary ([live page](https://cheparukhin.github.io/peptide-hla-stability/)) |
| [`EVALUATION.md`](EVALUATION.md) | Split construction, primary ranking metric, eligibility rules, and paired cluster/crossed bootstrap protocols |
| [`docs/DATASETS.md`](docs/DATASETS.md) | Provenance and curation of the Rasmussen (2016) SPA dataset, MHCflurry/BD2013 affinity reference, and MHC Motif Atlas elution benchmark |
| [`docs/STRUCTURES.md`](docs/STRUCTURES.md) | Extraction of Boltz-2 3D geometry (`R1`), trunk pair representations (`R2`–`R5`, `full_interface_pos`), ProteinMPNN scores (`P`, `C`), and FoldX energies (`E`) |
| [`docs/AFFINITY_REFERENCE.md`](docs/AFFINITY_REFERENCE.md) | Comparison between equilibrium binding affinity ($K_d$) and kinetic complex stability ($t_{1/2}$) |
| [`docs/ROUND2_INFERENCE.md`](docs/ROUND2_INFERENCE.md) | Instructions for running inference with the released `R4` ridge and `R5` MLP model weights (`pMHC-stability-round2-models-v1`) |
| [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) | Environment setup, data verification, and build checks |
| [`reports/finalization/SUBMISSION_BRIEF.md`](reports/finalization/SUBMISSION_BRIEF.md) | Concise executive summary of all experimental findings |
| [`reports/finalization/INDEPENDENT_ASSAY_BRIEF.md`](reports/finalization/INDEPENDENT_ASSAY_BRIEF.md) | Prospective experimental protocol for blinded wet-lab validation |

---

## Quick Verification

To verify the integrity of the saved experimental tables, source hashes, and project page locally:

```bash
python3 site/round2-build.py --check-full-cohort
```

To score precomputed Boltz-2 trunk representations (`R4` or `R5`) using the released 5-fold cross-validation model ensembles (`pMHC-stability-round2-models-v1`), follow [`docs/ROUND2_INFERENCE.md`](docs/ROUND2_INFERENCE.md).
