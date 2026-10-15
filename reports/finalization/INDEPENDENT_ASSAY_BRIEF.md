# Prospective Experimental Validation Protocol: Peptide–HLA Kinetic Dissociation

This document specifies a prospective wet-lab validation design to test the predictions of our frozen sequence and Boltz-2 trunk models on newly synthesized peptide–HLA class I complexes.

---

## 1. Scientific Objectives

All supervised training and cross-validation in [`reports/REPORT.md`](../REPORT.md) used the historical scintillation proximity assay (SPA) corpus of Rasmussen et al. (2016). A prospective experimental study should test three specific predictions while addressing the measurement limitations of the historical dataset:
1. **In-panel peptide ranking**: Whether the Boltz-2 trunk predictors (`R5_mlp`, `R4_ridge`, and the sequence + `R4` blend) improve within-allele Spearman rank correlation over the 30-MLP sequence baseline on newly assayed 9-mer peptides from novel peptide clusters (Hamming distance $> 3$ from all 5,633 peptides in `stability.parquet`).
2. **Cross-allele generalization**: Whether `R5_mlp` and `R5_ridge` maintain substantial ranking accuracy ($\rho \approx 0.50\text{–}0.57$) on HLA class I alleles whose contact pseudosequences differ by $> 3$ substitutions from all 75 training alleles, where sequence models collapse ($\rho < 0.20$).
3. **20-shot target-allele calibration**: Whether assaying a fixed 20-peptide support panel for a newly targeted HLA allele and updating the frozen `R5_ridge` head improves ranking on the remaining query peptides for that allele.

---

## 2. Recommended Panel Design

| Stratum | Target Alleles | Candidate 9-mer Peptides per Allele | Experimental Purpose |
| :--- | :---: | :---: | :--- |
| **Stratum A: In-panel alleles** | 6–8 common alleles from the 69-allele panel | 40 de novo peptides (Hamming distance $> 3$ from training set), spanning predicted deciles and sequence–trunk disagreement pairs | Test in-panel ranking gain ($\Delta\rho$) and sequence + trunk complementarity |
| **Stratum B: Unseen HLA sequence groups** | 6–8 alleles at contact-pseudosequence Hamming distance $> 3$ from all 75 training alleles | 20 support peptides + 40 query peptides per allele (in disjoint peptide clusters) | Test zero-shot cross-allele transfer and 20-shot `R5_ridge` calibration |

To maximize statistical power for distinguishing sequence models from Boltz-2 trunk representations in Stratum A, half of the 40 peptides per allele should be sampled stratified across predicted stability deciles, and half should be **discordant pairs** where `original_seq_mlp` and `R5_mlp` disagree by at least 3 within-allele rank deciles.

---

## 3. Assay Requirements and Blinding Protocol

To resolve the measurement ambiguities present in the Rasmussen et al. (2016) summary table (where $20.21\%$ of records are reported as $t_{1/2} = 0\text{ h}$ without raw decay curves):
1. **Pre-registration of frozen predictions**: Prior to wet-lab execution, all candidate `(allele, peptide)` pairs should be scored with the frozen [`pMHC-stability-round2-models-v1`](../../docs/ROUND2_INFERENCE.md) weights (`R5_mlp`, `R4_ridge`) and the frozen 30-MLP sequence ensemble, and the prediction table should be hashed (SHA-256) and timestamped.
2. **Separation of complex refolding yield from dissociation kinetics**: Measure initial complex formation / monomer yield prior to initiating the $37\,^\circ\text{C}$ dissociation chase (e.g., via size-exclusion chromatography, native ELISA, or initial time-zero signal amplitude) so that failed assembly is explicitly distinguished from rapid dissociation ($t_{1/2} < 0.1\text{ h}$).
3. **Replicate time-resolved decay curves**: Record full time-course dissociation traces across at least **3 independent technical/biological replicates** per peptide–HLA pair, fitting $k_{\text{off}}$ and $t_{1/2} = \ln 2 / k_{\text{off}}$ with explicit lower-limit-of-quantification (LLOQ) censoring flags.
4. **Primary analysis**: Evaluate paired differences in median within-allele Spearman $\rho$ using the exact cluster and crossed bootstrap procedures defined in [`EVALUATION.md`](../../EVALUATION.md).
