# Executive Summary: Structural Foundation Model Representations for Peptide–HLA Stability

This brief summarizes the principal scientific findings of the study. For the complete paper, see [`reports/REPORT.md`](../REPORT.md); for the interactive summary, see the [project page](https://cheparukhin.github.io/peptide-hla-stability/).

---

## 1. Scientific Question and Benchmark Design

Peptide–HLA class I dissociation half-life ($t_{1/2} = \ln 2 / k_{\text{off}}$) governs cell-surface antigen persistence and is biophysically distinct from equilibrium binding affinity ($K_d = k_{\text{off}} / k_{\text{on}}$; within-allele Spearman $\rho = 0.580$ across $7,281$ dual-labeled pairs). Using the Rasmussen et al. (2016) scintillation proximity assay dataset (**28,166** 9-mer complexes across **75** HLA alleles and **5,633** peptides) and an external immunopeptidomics benchmark from the MHC Motif Atlas (**897,600** evaluated pairs across **51** alleles), we evaluated when and why protein foundation models improve within-allele stability ranking over sequence-only predictors.

---

## 2. Four Principal Findings

1. **Sequence ensembles provide a strong in-panel baseline and transfer zero-shot to natural ligand elution, while naive foundation-model summaries fail.**  
   On the held-out peptide-cluster test split ($5,565$ pairs, $67$ alleles), a 30-model sequence MLP ensemble using the 9-mer peptide and 34-residue HLA contact pseudosequence achieves a median within-allele Spearman $\rho$ of **0.7064**, beating a single network by $+0.0883$ ($\rho = 0.6181$) and matching a full 182-residue HLA domain ensemble ($\rho = 0.6904$). Evaluated zero-shot on $81,600$ MS-eluted ligands vs $816,000$ proteome decoys across $51$ alleles, this stability predictor reaches **0.9656 median AUROC** (AUPRC $0.7804$), retains **0.9157 AUROC** against allele-swapped eluted-ligand decoys that control for generic presentability, and drops to **0.6965** under permuted HLA pseudosequences. Conversely, frozen **ESM-2 35M** embeddings tie the sequence baseline ($\rho = 0.6979$, $\Delta\rho = -0.0085$), and compressing $28,166$ predicted **Boltz-2** structures ($143.6$ A10G GPU-hours) into hand-engineered pocket geometry and confidence scalars (`R1`) degrades ranking to **0.5947** ($\Delta\rho = -0.1117$).

2. **Position-preserving Boltz-2 trunk pair representations outperform competitive sequence predictors.**  
   In nested 5-fold peptide-cluster cross-validation across the $19,716$-pair training cohort ($3,587$ peptide clusters, $69$ eligible alleles), extracting position-aware peptide–groove pair features from the frozen Boltz-2 trunk (`R4`, `R5`) lifts median within-allele Spearman $\rho$ from **0.6668** (competitive sequence selector spanning standard and allele-centered MLP ensembles, boosted trees, and pairwise rankers) to **0.7212** (pre-specified Boltz-2 embedding selector; **$\Delta\rho = +0.0543$**, $95\%$ CI $[+0.0314, +0.0808]$). Full $9 \times 34$ positional interface features reach **0.7315**, and blending `R4` ridge predictions with the sequence ensemble reaches **0.7238** ($\Delta\rho = +0.0569$ $[+0.0426, +0.0838]$).

3. **Positional anchor resolution—not a specific crystal contact mask—drives the structural representation gain.**  
   Preserving individual peptide positions outperforms dimension-matched pooled trunk tokens by **$+0.3387$** ($[+0.3029, +0.3773]$) in projected space (`R3`: $0.6552$ vs $0.3165$) and **$+0.1663$** ($[+0.1378, +0.1992]$) across the full $9 \times 34$ interface ($0.7315$ vs $0.5653$). By contrast, pooling HLA residues via a crystal contact mask (`R5`, $0.7210$) provides no advantage over randomly shuffled HLA contact subsets ($0.7203$, $\Delta\rho = +0.0007$ $[-0.0096, +0.0236]$).

4. **Boltz-2 interface representations generalize across unseen HLA sequence groups and support 20-shot allele calibration.**  
   Under leave-one-HLA-group-out cross-validation ($5 \times 5$ crossed folds), matched MLPs improve from **0.1605** (`R0` sequence) to **0.5667** (`R5` trunk; **$\Delta\rho = +0.4062$**, $95\%$ CI $[+0.3073, +0.4733]$), and under simultaneous peptide-cluster and HLA-group holdout from **0.1242** to **0.4980** (**$\Delta\rho = +0.3738$** $[+0.3092, +0.4638]$). Calibrating a frozen `R5` ridge head with **20 measured peptides** per held-out allele lifts ranking from **0.5296** to **0.5763** (**$\Delta\rho = +0.0468$**, $95\%$ CI $[+0.0227, +0.1035]$, $98.33\%$ Bonferroni CI $[+0.0140, +0.1195]$), driven by feature-weight rotation ($+0.0444$) rather than a scalar offset ($+0.0024$).

---

## 3. Secondary Findings

- **Inverse folding and empirical energies**: Adding ProteinMPNN complex log-likelihoods (`R5+P+C` vs `R5+P`: $+0.0240$ $[-0.0219, +0.0659]$ on the 4,449-pair subset; `R5+P+C` vs `R5`: $-0.0070$ $[-0.0220, +0.0131]$ across all 19,716 pairs) or 29 repaired FoldX energy terms ($+0.0150$ $[-0.0524, +0.0450]$) yields no established improvement over `R5`.
- **Objectives and hybrids**: Within-allele RankNet loss ($+0.0071$ $[-0.0051, +0.0098]$) and hybrid neural/tree selectors ($+0.0044$ $[-0.0055, +0.0100]$) do not materially improve upon standard MSE training on `R5`.
- **Assay-floor calibration**: Decoupled Tobit left-censored regression (MSE early stopping) improves floor-event Brier skill by **$+0.0165$** ($[+0.0138, +0.0205]$) without degrading rank correlation ($\Delta\rho = -0.0108$).
