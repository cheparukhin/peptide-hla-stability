# Predicting Peptide–HLA Class I Kinetic Stability with Structural Foundation Model Representations

[**Project Page**](https://cheparukhin.github.io/peptide-hla-stability/) · [**Full Paper (`reports/REPORT.md`)**](reports/REPORT.md) · [**Evaluation Protocol (`EVALUATION.md`)**](EVALUATION.md) · [**Datasets & Assays (`docs/DATASETS.md`)**](docs/DATASETS.md) · [**Structural Representations (`docs/STRUCTURES.md`)**](docs/STRUCTURES.md) · [**Model Inference (`docs/ROUND2_INFERENCE.md`)**](docs/ROUND2_INFERENCE.md)

---

## Overview

For a viral or tumor neoantigen peptide to trigger a $\text{CD8}^+$ T-cell response, the assembled peptide–HLA class I (pMHC-I) complex must not merely bind in equilibrium ($K_d = k_{\text{off}}/k_{\text{on}}$), but persist at the cell surface for hours ($t_{1/2} = \ln 2 / k_{\text{off}}$) to engage surveying T-cell receptors. Across $7,281$ peptide–HLA complexes measured for both binding affinity and dissociation half-life, equilibrium affinity ranks kinetic stability with a median within-allele Spearman correlation of only $\rho = 0.580$, leaving substantial variation in complex longevity unexplained by $K_d$ alone.

This study addresses a central question in AI for structural immunology: **when and why do biomolecular foundation models outperform sequence-only predictors for peptide–HLA kinetic stability?** Using the Rasmussen et al. (2016) scintillation proximity assay dataset (**28,166** 9-mer pMHC-I complexes across **75** HLA alleles and **5,633** peptides) together with an external immunopeptidomics benchmark from the MHC Motif Atlas (**897,600** evaluated complexes across **51** alleles), we establish four principal findings:

### 1. Sequence ensembles capture within-allele stability and transfer zero-shot to natural ligand presentation
Within any single HLA allele, a one-way variance decomposition ($\eta^2$) across peptide positions P1–P9 shows that kinetic stability is heavily concentrated at primary anchor positions P2 and P9 (supplemented by allele-specific auxiliary anchors at P1 and P3), whereas the solvent-exposed central bulge (P4–P8) carries high sequence entropy and low marginal variance. On the held-out peptide-cluster test split ($5,565$ complexes across $67$ eligible alleles), a 30-model multilayer perceptron (MLP) ensemble taking the 9-mer peptide and 34-residue HLA contact pseudosequence achieves a median within-allele Spearman $\rho$ of **0.7064**, outperforming a single network by $+0.0883$ ($\rho = 0.6181$) and matching an ensemble trained on the full 182-residue HLA $\alpha_1\alpha_2$ groove domain ($\rho = 0.6904$).

Applied zero-shot to $897,600$ disjoint complexes from the MHC Motif Atlas across $51$ alleles, this sequence stability predictor discriminates $81,600$ naturally MS-eluted ligands from $816,000$ human proteome decoys at a median per-allele **AUROC of 0.9656** (AUPRC $0.7804$; $93.9\%$ precision in the top $1\%$). When proteome decoys are replaced with real eluted ligands swapped from other HLA alleles—controlling for source-protein abundance, proteasomal cleavage, TAP transport, and MS ionizability—the predictor retains a median **AUROC of 0.9157** ($0.9562$ when donor grooves differ at $>14$ of $34$ contact positions vs $0.8906$ for closely related grooves) and drops to **0.6965** when HLA pseudosequences are cyclically permuted.

### 2. Naive protein language model embeddings and 3D coordinate summaries fail to beat sequence baselines
Replacing or augmenting sequence inputs with frozen **ESM-2** embeddings (35M, 150M, and 650M) yields no improvement over the sequence ensemble ($\rho = 0.6979$ on test, $\Delta\rho = -0.0085$). Controlled evaluation of **MINT** (an ESM-2 650M model adapted on $96\text{M}$ protein–protein interaction pairs) reveals that enabling cross-chain self-attention between the peptide and HLA chains adds a statistically significant $+0.0193$ ($p = 5.0 \times 10^{-6}$) over independent chain encoding, but generic PPI pre-training degrades per-residue anchor specificity by $-0.0800$ relative to base ESM-2 650M.

More strikingly, folding all $28,166$ three-chain pMHC-I heterotrimers with **Boltz-2** ($143.6$ A10G GPU-hours) and summarizing each predicted 3D structure via $50$ pocket contact counts, minimum distances, buried surface area, and pLDDT/ipTM/PAE confidence metrics (`R1`) reduces test ranking to $\rho = \mathbf{0.5947}$ ($\Delta\rho = -0.1117$). Multi-seed variance partitioning confirms that this degradation is not caused by diffusion coordinate noise (single-seed intraclass reliability $= 99.1\%$) or neural network co-adaptation (L2-shrunk linear residual models on top of frozen sequence predictions add at most $\Delta\rho = +0.0084$). Rather, because nearly all bound 9-mers adopt a canonical extended backbone trace inside the MHC cleft, coarse distance cutoffs and chain-pooled embeddings (`R2`, $\rho = 0.4219$) discard the side-chain chemistry and electrostatic compatibility of the anchor pockets.

### 3. Position-resolved pair representations from the frozen Boltz-2 trunk outperform competitive sequence models
Prior to coordinate diffusion, the deterministic Boltz-2 trunk maintains a directed 128-channel pairwise representation $z \in \mathbb{R}^{L \times L \times 128}$ across all residue pairs. Extracting position-resolved peptide–groove pair features directly from this frozen tensor—either averaging over the 182 groove residues for each peptide position (`R4`, $2,304\text{-d}$) or averaging over crystal-derived pocket contacts while keeping anchors P1, P2, and P9 distinct from the central core P3–P8 (`R5`, $1,024\text{-d}$)—recovers the physical interface signal. In nested 5-fold peptide-cluster cross-validation across the $19,716$-pair development cohort ($3,587$ peptide clusters, $69$ alleles), the pre-specified Boltz-2 trunk selector achieves a median within-allele Spearman $\rho$ of **0.7212** compared with **0.6668** for a competitive sequence selector spanning standard and allele-centered 30-MLP ensembles, CatBoost gradient-boosted trees, and pairwise rankers (**$\Delta\rho = +0.0543$**, 95% CI $[+0.0314, +0.0808]$). Blending a linear `R4` ridge head with the sequence ensemble reaches **0.7238** ($\Delta\rho = +0.0569$ $[+0.0426, +0.0838]$), and retaining all $9 \times 34$ projected peptide–contact-groove pairs (`full_interface_pos`, $2,448\text{-d}$) reaches **0.7315** ($\Delta\rho = +0.0647$ $[+0.0398, +0.0950]$).

Controlled ablations isolate the mechanism: preserving individual residue positions outperforms dimension-matched pooled trunk controls by **$+0.3387$** ($[+0.3029, +0.3773]$) in projected single-token space (`R3`: $0.6552$ vs $0.3165$) and by **$+0.1663$** ($[+0.1378, +0.1992]$) across the $9 \times 34$ pair interface ($0.7315$ vs $0.5653$). Conversely, replacing the crystal-derived HLA contact mask in `R5` ($0.7210$) with randomly permuted subsets of the 34 groove residues yields **0.7203** ($\Delta\rho = +0.0007$ $[-0.0096, +0.0236]$), showing that trunk self-attention has already propagated pocket context across the groove tokens.

### 4. Trunk representations generalize across unseen HLA sequence groups and calibrate from 20 peptides
In leave-one-allele-out cross-validation across $68$ alleles, sequence-ensemble accuracy decays steeply with binding-groove divergence—from $\rho = 0.7415$ when a training allele lies within $d \le 1$ contact-residue substitution to $0.5758$ at $d = 2\text{–}3$ and $0.3390$ at $d \ge 4$. When entire HLA sequence groups ($d \le 3$ on the 34-residue contact pseudosequence, forcing $d \ge 4$ to all training alleles) are held out in a crossed $5 \times 5$ design, sequence predictors collapse ($\rho = 0.1605$ for matched MLPs; $0.1921$ for the sequence selector), whereas Boltz-2 `R5` representations maintain $\rho = \mathbf{0.5667}$ (**$\Delta\rho = +0.4062$**, 95% CI $[+0.3073, +0.4733]$). Under simultaneous peptide-cluster and HLA-group holdout, matched MLPs improve from $0.1242$ to **0.4980** (**$\Delta\rho = +0.3738$** $[+0.3092, +0.4638]$). Calibrating a frozen `R5` ridge head with just **20 measured peptides** from a held-out allele lifts joint-holdout ranking from **0.5296** to **0.5763** (**$\Delta\rho = +0.0468$**, 95% CI $[+0.0227, +0.1035]$, 98.33% Bonferroni CI $[+0.0140, +0.1195]$), driven by feature-weight rotation ($+0.0444$) rather than an additive allele offset ($+0.0024$).

In secondary studies, augmenting `R5` with **ProteinMPNN** inverse-folding log-likelihoods (full-cohort $\Delta\rho = -0.0070$), repaired **FoldX** energy terms (panel $\Delta\rho = +0.0150$), within-allele **RankNet** losses ($\Delta\rho = +0.0071$), or hybrid tree/neural selectors ($\Delta\rho = +0.0044$) provided no established improvement. On the sequence baseline, left-censored **Tobit regression** at the $0.1\text{ h}$ detection floor with above-floor MSE early stopping significantly reduced floor-probability calibration error ($\Delta\text{Brier} = +0.0165$ vs Gaussian $\sigma$) without sacrificing rank order ($\Delta\rho = -0.0108$). Operationally, these results support a **two-tier screening workflow**: rapid proteome-wide pre-screening on CPU with the 30-MLP sequence ensemble ($0.073\text{ s}$ per $1,000$ complexes), followed by Boltz-2 `R4`/`R5` trunk scoring for lead prioritization and under-characterized HLA alleles.

---

## Summary of Principal Results

| Experimental regime | Evaluation cohort | Baseline | Candidate | Primary metric | Effect [95% CI] |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Test-split baseline & naive FMs** | 5,565 pairs · 67 alleles (test split) | 30-MLP sequence ensemble ($\rho = 0.7064$) | Single MLP ($\rho = 0.6181$)<br>ESM-2 35M ($\rho = 0.6979$)<br>Boltz-2 coarse geometry `R1` ($\rho = 0.5947$) | Median within-allele Spearman $\rho$ | $-0.0883$ $[-0.1201, -0.0555]$<br>$-0.0085$ $[-0.0291, +0.0233]$<br>$-0.1117$ $[-0.1339, -0.0575]$ |
| **Zero-shot ligand elution transfer** | 81,600 eluted ligands vs 816,000 decoys · 51 alleles | Random / permuted-HLA ($0.4974$ / $0.6965$) | 30-MLP stability ensemble (proteome / allele-swapped decoys) | Median per-allele AUROC | **0.9656** $[0.9435, 0.9779]$ (IQR)<br>**0.9157** $[0.8598, 0.9521]$ (IQR) |
| **Nested peptide-cluster holdout** | 19,716 pairs · 3,587 clusters · 69 alleles | Competitive sequence selector ($\rho = 0.6668$) | Boltz-2 trunk embedding selector ($\rho = 0.7212$) | Median within-allele Spearman $\rho$ | **$+0.0543$** $[+0.0314, +0.0808]$ |
| **Sequence + trunk complementarity** | 19,716 pairs · 3,587 clusters · 69 alleles | Competitive sequence selector ($\rho = 0.6668$) | Linear blend: sequence ensemble + `R4` ridge ($\rho = 0.7238$) | Median within-allele Spearman $\rho$ | **$+0.0569$** $[+0.0426, +0.0838]$ |
| **Positional vs pooled ablation** | 19,716 pairs · 3,587 clusters · 69 alleles | Pooled trunk tokens (`R3_pooled_matched` $\rho = 0.3165$) | Position-preserving trunk tokens (`R3` $\rho = 0.6552$) | Median within-allele Spearman $\rho$ | **$+0.3387$** $[+0.3029, +0.3773]$ |
| **Unseen HLA-group transfer** | 5 HLA folds $\times$ 5 peptide folds · 69 alleles | Matched sequence MLP `R0` ($\rho = 0.1605$) | Boltz-2 trunk MLP `R5` ($\rho = 0.5667$) | Median within-allele Spearman $\rho$ | **$+0.4062$** $[+0.3073, +0.4733]$ |
| **Joint peptide $\times$ HLA holdout** | 5 HLA folds $\times$ 5 peptide folds · 69 alleles | Matched sequence MLP `R0` ($\rho = 0.1242$) | Boltz-2 trunk MLP `R5` ($\rho = 0.4980$) | Median within-allele Spearman $\rho$ | **$+0.3738$** $[+0.3092, +0.4638]$ |
| **20-shot allele adaptation** | 68 alleles · 3 support designs (joint holdout) | Zero-shot `R5` ridge ($\rho = 0.5296$) | 20-peptide adapted `R5` ridge ($\rho = 0.5763$) | Median within-allele Spearman $\rho$ | **$+0.0468$** $[+0.0227, +0.1035]$ |

---

## Repository Guide

| Path | Contents |
| :--- | :--- |
| [`reports/REPORT.md`](reports/REPORT.md) | Complete scientific paper covering experimental design, biophysical signal structure, main results, ablations, biological transfer, secondary studies, and limitations |
| [`site/index.html`](site/index.html) | Narrative project page and visual summary ([live page](https://cheparukhin.github.io/peptide-hla-stability/)) |
| [`EVALUATION.md`](EVALUATION.md) | Split construction, primary ranking metric, eligibility rules, and paired cluster/crossed bootstrap protocols |
| [`docs/DATASETS.md`](docs/DATASETS.md) | Provenance and curation of the Rasmussen (2016) SPA dataset, MHCflurry/BD2013 affinity reference, and MHC Motif Atlas elution benchmark |
| [`docs/STRUCTURES.md`](docs/STRUCTURES.md) | Mathematical construction of Boltz-2 3D geometry (`R1`), trunk representations (`R2`–`R5`, `full_interface_pos`), ProteinMPNN scores (`P`, `C`), and FoldX energies (`E`) |
| [`docs/AFFINITY_REFERENCE.md`](docs/AFFINITY_REFERENCE.md) | Comparison between equilibrium binding affinity ($K_d$) and kinetic complex stability ($t_{1/2}$) |
| [`docs/ROUND2_INFERENCE.md`](docs/ROUND2_INFERENCE.md) | Instructions for running inference with the released `R4` ridge and `R5` MLP model weights (`pMHC-stability-round2-models-v1`) |
| [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) | Environment setup, data verification, and reproducibility checks |
| [`reports/finalization/SUBMISSION_BRIEF.md`](reports/finalization/SUBMISSION_BRIEF.md) | Concise executive summary of all experimental findings |
| [`reports/finalization/INDEPENDENT_ASSAY_BRIEF.md`](reports/finalization/INDEPENDENT_ASSAY_BRIEF.md) | Prospective experimental protocol for blinded wet-lab validation |

---

## Quick Verification

To verify the integrity of all saved full-cohort comparison tables and fixed-query evaluation records locally:

```bash
python3 site/round2-build.py --check-full-cohort
```

To inspect and run predictions on precomputed Boltz-2 trunk representations (`R4` or `R5`) using the released 5-fold cross-validation model ensembles (`pMHC-stability-round2-models-v1`), follow [`docs/ROUND2_INFERENCE.md`](docs/ROUND2_INFERENCE.md).
