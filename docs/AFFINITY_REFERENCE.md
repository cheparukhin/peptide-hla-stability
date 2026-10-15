# Equilibrium Binding Affinity Reference vs. Kinetic Stability

This document describes the curated equilibrium binding affinity reference datasets ([`data/data_augmentation_iedb/affinity_reference_75alleles.csv`](../data/data_augmentation_iedb/affinity_reference_75alleles.csv) and [`data/augmentation/measured_affinity.csv`](../data/augmentation/measured_affinity.csv)) and quantifies the empirical divergence between equilibrium affinity ($K_d$) and kinetic complex stability ($t_{1/2}$).

---

## 1. Biophysical Distinction: $K_d$ vs. $t_{1/2}$

Equilibrium binding affinity measures the ratio of dissociation and association rate constants:
$$K_d = \frac{k_{\text{off}}}{k_{\text{on}}},$$
typically approximated by radioligand or fluorescence competition $\text{IC}_{50}$ (nM). By contrast, the Rasmussen et al. (2016) scintillation proximity assay measures **kinetic dissociation half-life** at $37\,^\circ\text{C}$:
$$t_{1/2} = \frac{\ln 2}{k_{\text{off}}}.$$
Because peptide association rates ($k_{\text{on}}$) vary across sequences and HLA grooves—reflecting conformational selection, groove breathing, and electrostatic steering during loading—two peptides with similar equilibrium $\text{IC}_{50}$ values can exhibit markedly different off-rates and cell-surface persistence times.

---

## 2. Provenance and Curation of the Equilibrium Affinity Reference

Quantitative binding affinity records were curated from the MHCflurry 2.0 / IEDB BD2013 training compendium (`curated_training_data.affinity.csv.bz2`, SHA-256 verified in [`data/data_augmentation_iedb/affinity_reference_75alleles.manifest.json`](../data/data_augmentation_iedb/affinity_reference_75alleles.manifest.json)) using deterministic filtering:
1. Restrict to human four-digit canonical `HLA-A`, `HLA-B`, `HLA-C`, and `HLA-E` alleles and canonical 9-mer peptides.
2. Transform measured affinities (nM, capped to $[0.01, 10^5]\text{ nM}$) onto the unit interval:
   $$y_{\text{aff}} = 1 - \frac{\log_{10}(\text{nM})}{\log_{10}(50{,}000)} \in [0, 1],$$
   where higher $y_{\text{aff}}$ corresponds to tighter binding ($500\text{ nM} \mapsto 0.426$).
3. Deduplicate repeated `(allele_canonical, peptide)` records using the censorship-aware geometric mean (or exact replicate median when present).

The resulting reference tables contain **140,855** unique `(allele_canonical, peptide)` records across **105** HLA class I alleles (`data/external/mhcflurry_ba/curated_9mer_ba.parquet`), with **121,150** records across the **64** overlapping wild-type benchmark alleles tracked in [`data/data_augmentation_iedb/affinity_reference_75alleles.csv`](../data/data_augmentation_iedb/affinity_reference_75alleles.csv) and [`data/augmentation/measured_affinity.csv`](../data/augmentation/measured_affinity.csv).

---

## 3. Empirical Concordance Between Experimental $K_d$ and $t_{1/2}$

Intersecting `curated_9mer_ba.parquet` with wild-type records in `stability.parquet` (excluding the three engineered `C67S` disulfide constructs so that experimental constructs match) yields **7,281** dual-labeled peptide–HLA pairs across **64** alleles (**42** alleles with $\ge 20$ shared observations; [`reports/stage2c_affinity_ceiling.csv`](../reports/stage2c_affinity_ceiling.csv)):
- **Median within-allele Spearman correlation**: **$\rho = 0.580$** (mean $\rho = 0.573$, IQR $[0.485, 0.672]$ across the 42 eligible alleles).
- **Pooled Pearson correlation**: **$r = 0.562$** between $y_{\text{aff}}$ and $y = \log_{10}(\max(t_{1/2}, 0.01))$.

Notice that our sequence-only stability ensemble ($\rho = 0.7064$ on the held-out test split; $0.6615$ in nested CV) and our Boltz-2 trunk predictors ($\rho = 0.7212\text{–}0.7315$ in nested CV) rank kinetic stability substantially better than *experimental equilibrium binding affinity measurements on the exact same complexes* ($\rho = 0.580$).

---

## 4. Auxiliary Affinity Padding and Multitask Experiments (`stage2b`, `stage2c`, `stage3b`)

We tested two ways of incorporating `curated_9mer_ba.parquet` into stability training after removing all validation and test peptide clusters (Hamming distance $\le 3$) to prevent homology leakage:
1. **Weak-binder negative padding (`stage2b`, [`reports/stage2b_deltas.csv`](../reports/stage2b_deltas.csv))**: Augmenting under-sampled stability alleles with verified non-binders ($\text{IC}_{50} > 20{,}000\text{ nM}$) mapped to the $y = -2.0$ stability floor yielded $\Delta\rho = +0.0055$ ($[-0.0185, +0.0226]$) on validation.
2. **Auxiliary affinity multitask training (`stage2c` and `stage3b`, [`reports/stage2c_deltas_ensemble_pep_pseudo.csv`](../reports/stage2c_deltas_ensemble_pep_pseudo.csv), [`reports/stage3b_deltas.csv`](../reports/stage3b_deltas.csv))**: Jointly training shared sequence or ESM-2 representations with a dual stability + affinity head across $62,557$ leak-free affinity pairs shifted validation median Spearman $\rho$ by **$+0.0055$** ($[-0.0149, +0.0211]$) for the sequence ensemble and **$+0.0044$** ($[-0.0191, +0.0239]$) for ESM-2.

Because $K_d$ and $t_{1/2}$ diverge in their dependence on $k_{\text{on}}$ versus $k_{\text{off}}$, supervising on equilibrium affinity does not improve kinetic stability ranking on alleles with direct kinetic training data.
