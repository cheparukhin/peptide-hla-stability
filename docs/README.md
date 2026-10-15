# Documentation Index

This directory documents the datasets, structural representations, equilibrium affinity reference, model inference pipeline, and reproducibility checks supporting [`reports/REPORT.md`](../reports/REPORT.md).

| Document | Scope |
| :--- | :--- |
| [`DATASETS.md`](DATASETS.md) | Provenance, curation, and schema of the Rasmussen et al. (2016) kinetic stability dataset (`data/processed/stability.parquet`), HLA sequence mappings, and the MHC Motif Atlas zero-shot elution benchmark (`stage3c`) |
| [`STRUCTURES.md`](STRUCTURES.md) | Construction of the structural representation hierarchy: coarse Boltz-2 3D geometry and confidence summaries (`R1`), internal Boltz-2 trunk pair representations (`R2`–`R5`, `full_interface_pos`), ProteinMPNN inverse-folding scores (`P`, `C`), and repaired FoldX energy terms (`E`) |
| [`AFFINITY_REFERENCE.md`](AFFINITY_REFERENCE.md) | Curated equilibrium binding affinity reference (`data/external/mhcflurry_ba/`, 140,855 pairs) and empirical comparison between $K_d$ and kinetic $t_{1/2}$ ($\rho = 0.580$ across 7,281 shared pairs) |
| [`ROUND2_INFERENCE.md`](ROUND2_INFERENCE.md) | Step-by-step guide to scoring precomputed Boltz-2 trunk features (`R4`, `R5`) using the released 5-fold cross-validation model weights (`pMHC-stability-round2-models-v1`) |
| [`DEVELOPMENT.md`](DEVELOPMENT.md) | Environment setup, data verification scripts, and static site integrity checks |
