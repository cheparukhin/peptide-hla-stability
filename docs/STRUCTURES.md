# Structural Predictions and Representation Hierarchy

This document details how 3D complex predictions, internal foundation-model trunk tensors, inverse-folding likelihoods, and physics-based energy descriptors were generated and assembled into the representation hierarchy (`R0`–`R5`, `full_interface_pos`, `P`, `C`, and `E`).

---

## 1. Boltz-2 Complex Structure Prediction

### 1.1 Input Construction and Execution
Every unique `(allele_raw, peptide)` record in [`data/rasmussen_et_al_dataset.csv`](../data/rasmussen_et_al_dataset.csv) ($28,166$ complexes across $75$ HLA alleles and $5,633$ 9-mer peptides) was modeled as a three-chain heterotrimer:
- **Chain A (HLA heavy chain)**: 182-residue $\alpha_1\alpha_2$ antigen-binding domain (`domain_seq`, preserving the engineered `C67S` substitution for the three mutated constructs).
- **Chain B ($\beta_2$-microglobulin)**: Canonical 99-residue human $\beta_2\text{m}$ sequence.
- **Chain C (Peptide)**: 9-mer ligand sequence.

Precomputed multiple sequence alignments (MSAs) were generated once for the 75 unique heavy-chain sequences and $\beta_2\text{m}$ (`reports/msa_manifest.csv`), while the 9-mer peptide chain was run in single-sequence mode (`msa: empty`) so that cross-chain conditioning reflects the specific peptide sequence. All $28,166$ complexes were folded with **Boltz-2** (`boltz==2.2.1`, 1 recycling iteration, 25 diffusion steps, 1 diffusion sample per complex) across 25 parallel Modal A10G GPU containers (`143.6` total A10G GPU-hours, $6.2\text{ h}$ wall-clock time; [`reports/gpu_decision.csv`](../reports/gpu_decision.csv)).

---

## 2. The Representation Hierarchy (`R0`–`R5` and `full_interface_pos`)

A central finding of our study is that *where* and *at what positional resolution* representations are extracted from a structure predictor determines whether they underperform or outperform sequence baselines.

### 2.1 Sequence and Coarse 3D Summary Baselines (`R0`, `R1`, `R2`)
- **`R0` (One-hot sequence baseline, 860-d)**: Flattened one-hot encoding of the 9 peptide residues and 34 polymorphic HLA contact groove residues ($43 \times 20 = 860$). Used for matched-architecture neural and ridge comparisons against structural representations. (Note: our `original_seq_mlp` baseline uses $43 \times 41 = 1,763$-d one-hot + BLOSUM62 features.)
- **`R1` (Coarse 3D geometry and confidence summaries, 50-d; `stage4c5` / `stage5`)**: Extracted from the final diffused 3D coordinates and confidence heads:
  - Per-peptide-position heavy-atom contact counts ($<4.0\,\text{\AA}$ to HLA heavy chain, 9 features) and minimum heavy-atom distances to the HLA groove (9 features).
  - Global interface contact count, peptide radius of gyration, peptide end-to-end distance, and buried surface proxy.
  - Boltz-2 confidence metrics: per-peptide-position pLDDT (9 features), mean peptide pLDDT, complex pLDDT, pTM, ipTM, and peptide–groove interface PAE summaries.
  - *Outcome*: On the held-out test split, adding `R1` to sequence degraded within-allele Spearman $\rho$ from $0.7064$ to **$0.5947$** ($\Delta\rho = -0.1117$), because all 9-mers share the same canonical backbone trace inside the groove and hard distance cutoffs amplify stochastic diffusion-seed coordinate jitter ([`reports/further_research/phase_d_pose_seed_variance.csv`](../reports/further_research/phase_d_pose_seed_variance.csv)).
- **`R2` (Chain-pooled Boltz-2 trunk states, 512-d)**: Global mean-pooling of the frozen Boltz-2 trunk single representation ($s \in \mathbb{R}^{L \times 384}$) and pair representation ($z \in \mathbb{R}^{L \times L \times 128}$) across peptide and HLA chains.
  - *Outcome*: Collapses to **0.4219** (`R2_mlp`) and **0.3694** (`R2_ridge`) in nested 5-fold CV because averaging across the 9 peptide residues erases which amino acid occupies primary anchor pockets P2 and P9.

### 2.2 Position-Preserving Boltz-2 Trunk Representations (`R3`, `R4`, `R5`, `full_interface_pos`)
To preserve residue-level interface organization while bypassing coordinate diffusion noise, we extract deterministic internal representations directly from the frozen Boltz-2 trunk:
- **`R3` (Projected positional single + groove-averaged pair tokens, 580-d)**: Retains individual peptide positions (and groove contact positions) via compact linear projections of the trunk single states $s$ and groove-averaged pair states $z$.
  - *Controlled pooling ablation*: `R3_mlp` achieves **0.6552** vs **0.3165** for a dimension-matched pooled control (`R3_pooled_matched_mlp`), a paired gain of **$\Delta\rho = +0.3387$** ($[+0.3029, +0.3773]$).
- **`R4` (Full peptide-position groove-averaged pair representation, 2,304-d)**: For each of the 9 peptide positions $i \in \{1,\dots,9\}$, we extract both directed 128-channel trunk pair vectors between peptide position $i$ and the 34 polymorphic HLA contact residues $j \in \mathcal{G}_{34}$, averaged over the 34 groove positions:
  $$z_{i,\text{pep}\to\text{HLA}} = \frac{1}{34}\sum_{j \in \mathcal{G}_{34}} z_{i,j} \in \mathbb{R}^{128}, \qquad z_{i,\text{HLA}\to\text{pep}} = \frac{1}{34}\sum_{j \in \mathcal{G}_{34}} z_{j,i} \in \mathbb{R}^{128}.$$
  Concatenating both directions across all 9 peptide positions yields $9 \times 2 \times 128 = \mathbf{2,304}$ features.
  - *Outcome*: `R4_ridge` reaches **0.6845**, `R4_mlp` reaches **0.7136**, and blending `R4_ridge` with the sequence ensemble (`blend_seq_R4`) reaches **0.7238** in nested 5-fold CV.
- **`R5` (Anchor-resolved contact-pooled pair representation, 1,024-d)**: Partitions the 9-mer peptide into four functional segments—primary/auxiliary anchors **P1**, **P2**, and **P9** kept at single-residue resolution, and the central bulge **P3–P8** pooled together—each averaged in both pair directions ($z_{\text{pep}\to\text{HLA}}, z_{\text{HLA}\to\text{pep}}$) over the subset of HLA groove residues designated as contacting that segment in reference pMHC crystal structures ($4 \times 2 \times 128 = \mathbf{1,024}$ features).
  - *Outcome & shuffled-contact ablation*: `R5_mlp` reaches **0.7210** in nested 5-fold CV, **0.5667** on unseen HLA groups ($\Delta\rho = +0.4062$ over `R0_mlp`), and **0.4980** on joint HLA-group $\times$ peptide-cluster holdouts ($\Delta\rho = +0.3738$). Replacing the crystal contact groove subsets with randomly permuted subsets of the 34 groove residues (`R5_shuffled_contact_mlp`) yields **0.7203** ($\Delta\rho = +0.0007$ $[-0.0096, +0.0236]$), proving that anchor position resolution—not the specific crystal contact mask—drives the gain.
- **`full_interface_pos` (Unpooled $9 \times 34$ peptide–groove pair interface, 2,448-d)**: Retains all $9 \times 34 = 306$ peptide–groove residue pairs by projecting the 256-d bidirectional pair state $(z_{i,j}, z_{j,i})$ at each $(i,j)$ pair to 8 channels ($9 \times 34 \times 8 = \mathbf{2,448}$ features).
  - *Outcome*: Achieves the highest single-model score in nested 5-fold CV (**0.7315**), outperforming a dimension-matched pooled interface control (`full_interface_pooled_mlp`, **0.5653**) by **$\Delta\rho = +0.1663$** ($[+0.1378, +0.1992]$).

---

## 3. Secondary Structural Descriptors: ProteinMPNN (`P`, `C`) and FoldX (`E`)

To test whether explicit inverse-folding likelihoods or empirical force-field energies add complementary information beyond `R5` trunk pair features, we extracted two additional descriptor families ([`reports/followup3/`](../reports/followup3/)):

1. **ProteinMPNN native-peptide log-likelihoods (`P` and `C`, 13 + 13 features)**:
   - Using fixed-backbone ProteinMPNN (`v_48_020`), we scored the log-likelihood of the native 9-mer peptide sequence in two structural contexts:
     - **Peptide-only context (`P`, 13 features)**: Conditioned solely on the predicted 9-mer peptide backbone coordinates (total, mean, core P3–P8, anchor P1/P2/P9, and all 9 individual positional log-likelihoods).
     - **Full complex context (`C`, 13 features)**: Conditioned on the complete pMHC-I heterotrimer backbone and fixed HLA + $\beta_2\text{m}$ sequences, capturing pocket-specific inverse-folding compatibility.
   - Evaluated on a balanced 4,449-pair subset ($69$ alleles; [`reports/followup3/inverse_execution_v2/comparisons.csv`](../reports/followup3/inverse_execution_v2/comparisons.csv)) and scaled to all **19,716** development complexes ([`reports/followup3/inverse_fulltrain_candidate/comparisons.csv`](../reports/followup3/inverse_fulltrain_candidate/comparisons.csv)), `R5+P+C` showed no gain over `R5` alone on the full cohort (**0.6885** vs **0.6955**, $\Delta\rho = -0.0070$ $[-0.0220, +0.0131]$).

2. **Repaired FoldX empirical energy descriptors (`E`, 29 features)**:
   - On the balanced 4,449-pair subset, each predicted complex was relaxed with `FoldX RepairPDB` and scored with `AnalyseComplex` and `Stability` to extract 29 thermodynamic terms ($\Delta G_{\text{interaction}}$, van der Waals, electrostatics, polar and apolar solvation, hydrogen-bond energetics, side-chain and main-chain entropy, electrostatic kon, and intra-/inter-chain clash penalties).
   - Adding `E` to `R5` (`R5+E` vs `R5`; [`reports/followup3/energy_execution_v2/comparisons.csv`](../reports/followup3/energy_execution_v2/comparisons.csv)) yielded $0.6157$ vs $0.6007$ ($\Delta\rho = +0.0150$, $98.33\%$ Bonferroni CI $[-0.0524, +0.0450]$), providing no established improvement over `R5` alone.
