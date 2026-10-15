# Development and Reproducibility Guide

This guide describes the repository layout, local environment setup, and verification commands for checking the saved experimental artifacts and static project page.

---

## 1. Environment Setup

Local orchestration, data verification, and static site checks require Python $\ge 3.11$ managed via `uv` (or standard `venv`):

```bash
uv sync
```

Compute-intensive structure predictions (`Boltz-2`), protein language model extractions (`ESM-2`), inverse-folding scoring (`ProteinMPNN`), and force-field calculations (`FoldX`) were executed on Modal cloud workers (`pepstab/modal_*.py`) and their resulting features, out-of-fold predictions, and bootstrap intervals are committed directly under `reports/`.

---

## 2. Repository Structure

| Directory / File | Role |
| :--- | :--- |
| [`reports/REPORT.md`](../reports/REPORT.md) | Main scientific paper |
| [`site/`](../site/) | Static project page (`index.html`), figure assets (`figures/`), and interactive biology animation (`intro/`) |
| [`EVALUATION.md`](../EVALUATION.md) | Specification of splits, primary ranking metric, and bootstrap inference |
| [`data/`](../data/) | Curated Rasmussen et al. (2016) dataset ([`rasmussen_et_al_dataset.csv`](../data/rasmussen_et_al_dataset.csv)), homology splits ([`splits.csv`](../data/splits.csv)), HLA constructs ([`hla75_constructs.fasta`](../data/hla75_constructs.fasta)), and IEDB/MHCflurry affinity references ([`data_augmentation_iedb/`](../data/data_augmentation_iedb/)) |
| [`pepstab/`](../pepstab/) | Core Python package for data ingestion, homology splitting, sequence/trunk modeling, and bootstrap evaluation |
| [`models/finalization/`](../models/finalization/) | Exported `R4_ridge_100` and `R5_mlp_0.01` model heads and SHA-256 checksum manifests |
| [`scripts/`](../scripts/) | Standalone CLI utilities, including [`scripts/round2_inference.py`](../scripts/round2_inference.py) for model verification and cached-feature inference |
| [`reports/`](../reports/) | Saved experimental tables, out-of-fold predictions, bootstrap distributions, and provenance manifests |
| [`tests/`](../tests/) | Automated verification suite (including [`tests/test_round2_inference.py`](../tests/test_round2_inference.py)) |

---

## 3. Integrity and Reproducibility Checks

To verify that all saved full-cohort comparison tables and fixed-query evaluation records pass arithmetic and schema checks:

```bash
python3 site/round2-build.py --check-full-cohort
```

To serve the project page (`site/index.html`) and biology animation (`site/intro/index.html`) locally:

```bash
python3 -m http.server 8000 --directory site
```
Then open `http://localhost:8000/` in a browser.
