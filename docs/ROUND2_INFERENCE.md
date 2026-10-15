# Inference with Exported Boltz-2 Trunk Models (`R4_ridge_100` and `R5_mlp_0.01`)

This guide documents how to inspect, verify, and run stability predictions using the exported Stage II Boltz-2 trunk heads tracked in [`models/finalization/`](../models/finalization/) via [`scripts/round2_inference.py`](../scripts/round2_inference.py). Full benchmark methodology and empirical comparisons are presented in [`reports/REPORT.md`](../reports/REPORT.md).

---

## 1. Exported Model Heads

The repository exports two pre-specified development refits trained on the $19,716$-pair original training partition ([`reports/finalization/inference/distribution_manifest.json`](../reports/finalization/inference/distribution_manifest.json)):

| Model Directory | Representation | Input Dimension | Architecture & Training Recipe | Nested 5-Fold CV Median $\rho$ (69 Alleles) |
| :--- | :--- | :---: | :--- | :---: |
| **[`models/finalization/R4_ridge_100`](../models/finalization/R4_ridge_100)** | `R4` (9-position groove-averaged Boltz-2 trunk pair states) | $2,304$ ($9 \times 2 \times 128$) | Single `Ridge(alpha=100, solver='lsqr', tol=1e-5, max_iter=2000)` with training-cohort feature standardization | **0.6845** (**0.7238** when blended with sequence ensemble) |
| **[`models/finalization/R5_mlp_0.01`](../models/finalization/R5_mlp_0.01)** | `R5` (anchor-resolved contact-pooled Boltz-2 trunk pair states) | $1,024$ ($4 \times 2 \times 128$) | 30-member ensemble (5 peptide-cluster early-stopping folds $\times$ 6 seeds `0..5`) of 64-unit ReLU MLPs ($\text{L2}=0.01$, Adam $\text{lr}=10^{-3}$, batch size $256$, max $200$ epochs, patience $15$) | **0.7210** (**0.5667** on unseen HLA groups) |

Each model directory contains three immutable files verified by SHA-256 checksums:
- `parameters.npz` (weights, biases, target offsets, and per-member feature standardization statistics, loaded safely with `allow_pickle=False`).
- `model.json` ([`R4_ridge_100/model.json`](../models/finalization/R4_ridge_100/model.json), [`R5_mlp_0.01/model.json`](../models/finalization/R5_mlp_0.01/model.json): hyperparameter recipe, stopping-fold ancestry, feature/contact-map schema, and data provenance hashes).
- `complete.json` (integrity checksum manifest).

---

## 2. Environment Setup

From the repository root on macOS or Linux, create a Python 3.12 virtual environment matching the recorded runtime ([`reports/finalization/transfer_controls/local_timing/runtime.json`](../reports/finalization/transfer_controls/local_timing/runtime.json)):

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install \
  numpy==2.5.3 pandas==3.0.6 scipy==1.18.1 scikit-learn==1.9.1 \
  threadpoolctl==3.7.0 joblib==1.6.0 python-dateutil==2.9.0.post0 six==1.17.0 \
  pytest==9.1.1
```

---

## 3. Input Feature Contract

The inference CLI consumes verified compact Boltz-2 trunk feature bundles (`schema: pepstab-round2-representations-v1`; see [`docs/STRUCTURES.md`](STRUCTURES.md) for how `R4` and `R5` are constructed from the Boltz-2 pair state tensor $z \in \mathbb{R}^{L \times L \times 128}$):
- **`R4` ($2,304\text{-d}$)**: Ordered as $[\text{P1}\dots\text{P9}] \times [\text{peptide}\to\text{groove},\, \text{groove}\to\text{peptide}] \times 128\text{ channels}$, averaged over the 34 polymorphic HLA contact positions.
- **`R5` ($1,024\text{-d}$)**: Ordered as $[\text{P1},\, \text{P2},\, \text{P9},\, \operatorname{mean}(\text{P3}\dots\text{P8})] \times [\text{peptide}\to\text{groove},\, \text{groove}\to\text{peptide}] \times 128\text{ channels}$, averaged over the crystal-derived HLA contact subsets.

Predictions are written as `prediction_y_log1p` on the stabilized logarithmic target scale $y = \log_{10}(1 + t_{1/2})$ alongside `pair_id` and `input_hash`, intended for **within-allele peptide ranking**.

---

## 4. Command-Line Usage (`scripts/round2_inference.py`)

### Inspect a Model Configuration or Verify Saved Weights
```bash
# Display the locked architecture and fit count for a candidate
.venv/bin/python scripts/round2_inference.py plan --candidate R5_mlp_0.01

# Verify checksums, dimensions, and metadata of an exported model directory
.venv/bin/python scripts/round2_inference.py inspect --model models/finalization/R4_ridge_100
.venv/bin/python scripts/round2_inference.py inspect --model models/finalization/R5_mlp_0.01
```

### Run Predictions on a Feature Bundle
```bash
.venv/bin/python scripts/round2_inference.py predict \
  --model models/finalization/R4_ridge_100 \
  --features reports/further_research/round2/extractor/full_training/feature_bundles.json \
  --pair-ids reports/finalization/inference/benchmark_pair_ids.txt \
  --out /tmp/r4_cached_predictions.csv
```

### Run the Verification Test Suite
[`tests/test_round2_inference.py`](../tests/test_round2_inference.py) verifies exact numerical parity between serialized models and `FeatureHead` across all 30 MLP ensemble members, tests corruption/hash rejection, and validates input contracts:

```bash
.venv/bin/python -m pytest -q tests/test_round2_inference.py
```
