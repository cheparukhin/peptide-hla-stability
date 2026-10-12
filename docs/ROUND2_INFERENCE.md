# Cached-feature round-two inference

This package exports reusable **R4 ridge** and **R5 MLP** heads. It consumes
verified Boltz compact feature caches. It does not create features from a novel
peptide/HLA sequence, call Boltz, produce structures, or run an MSA search.
New sequence inference still requires the pinned extraction workflow and its
unmeasured new-case costs. See the [canonical research report](../reports/further_research/round2/REPORT.md).

The two demonstration recipes are explicitly locked development refits:
`R4_ridge_100` (one fit) and `R5_mlp_0.01` (30 fits). They are not selected from
follow-up outer outcomes and neither is designated a global winner. These
exports are fit only to the original training partition. They provide a usable
artifact, not independent confirmation or a new held-out accuracy estimate.

## Input contract

Use a compact bundle directory containing `complete.json`, its content-addressed
NPZ and manifest, or a complete production JSON manifest listing
`feature_bundles`. Existing full training input is
`reports/further_research/round2/extractor/full_training/feature_bundles.json`.
Bare feature matrices and sequence strings are intentionally unsupported.

Each bundle must have schema `pepstab-round2-representations-v1`, finite float32
vectors, unique ordered `pair_id` strings and a nonempty `input_hash` per row.
Array identities must match the manifest. File hashes, feature protocol, contact
map and dimensions are verified before prediction. R4 has 2,304 columns in
`[P1..P9, peptide-to-groove/groove-to-peptide, 128 channels]` order. R5 has 1,024
columns in `[P1,P2,P9,mean(P3..P8), direction,128 channels]` order using the frozen
crystal contact map. Both use pooling; no channel projection is applied.

`--pair-ids` optionally selects a subset and sets output order; one ID per line.
Missing or duplicate requested IDs fail. Prediction outputs preserve `pair_id`
and `input_hash` and return `prediction_y_log1p`, an uncalibrated prediction on
the `log1p(thalf_hours)` scale. Outputs may be negative and are not guaranteed
physical half-life estimates. The tool does not silently clamp or calibrate.

## Portable artifacts

Each model directory contains `parameters.npz`, `model.json` and `complete.json`.
NPZ is read with `allow_pickle=False`; no executable serialized objects are used.
The completion record checksums both files. Parameters contain weights, biases,
target offsets and exact per-member scaler means, scales and variances.
Metadata records the complete candidate/configuration, training row/label/feature
hashes, source hashes, versions, stopping-fold ancestry, authorization and the
feature/contact-map contract. Fixed token/pair projection matrices are preserved
for provenance although these two representations do not use them. Model and
output paths are immutable; a new run requires a new path.

R4 matches the frozen `Ridge(alpha=100, solver='lsqr', tol=1e-5,
max_iter=2000)` recipe with a scaler fitted on all training rows. R5 matches
the main study's 64-unit ReLU MLP: L2 0.01, Adam learning rate 0.001, batch 256,
epoch cap 200 and patience 15. It averages five deterministic peptide-cluster
stopping folds times seeds 0..5. Each scaler sees only that member's gradient
rows. This is the main recipe; the export does not reproduce HLA/joint outer
assessment procedures or their selection. Predictions use the stored float32
scaler arithmetic and member order. Exact save/load equality is verified on
the recorded platform; cross-platform numerical identity is not promised.

## Commands

Show the cost bound without fitting:

```sh
.venv/bin/python scripts/round2_inference.py plan --candidate R5_mlp_0.01
```

Inspect a model and verify its files:

```sh
.venv/bin/python scripts/round2_inference.py inspect --model models/finalization/R4_ridge_100
```

Predict from an existing cached bundle:

```sh
.venv/bin/python scripts/round2_inference.py predict \
  --model models/finalization/R4_ridge_100 \
  --features reports/further_research/round2/extractor/full_training/feature_bundles.json \
  --pair-ids reports/finalization/inference/benchmark_pair_ids.txt \
  --out /tmp/r4_cached_predictions.csv
```

Refit only with an explicit authorized budget and locked candidate:

```sh
.venv/bin/python scripts/round2_inference.py fit \
  --candidate R4_ridge_100 --max-fits 1 \
  --features reports/further_research/round2/extractor/full_training/feature_bundles.json \
  --training-csv /tmp/original_training_rows.csv \
  --authorization reports/finalization/inference/training_authorization.json \
  --out /tmp/r4_development_refit
```

The training CSV must contain `pair_id`, `cluster_id`, `split` and `y_log1p`.
The CLI verifies these against canonical original training rows before any fit;
validation/test rows or altered labels/clusters fail. The Python API supports
synthetic developer checks and relies on its caller to supply original-train
membership; it enforces the marked training split, finite labels, row identity
and five nonempty grouped stopping folds. It never chooses a candidate.
The CLI limits numerical work to two threads and writes a prefit authorization
record with data/feature/source hashes before fitting. One candidate is exported
per call; 1/30 fits are the full recipes, not grid-search budgets.

## Verification and measured scope

`tests/test_round2_inference.py` checks exact prediction parity with the existing
frozen `FeatureHead` on synthetic data, all 30 MLP members, exact serialization,
budget rejection, training-scope/identity/provenance errors, corrupted files and
path traversal rejection. Run:

```sh
.venv/bin/python -m pytest -q tests/test_round2_inference.py
```

The finalization distribution manifest and benchmark report live under
`reports/finalization/inference/`. The benchmark is a **performance-only,
in-sample cached-head benchmark** on identified training cache rows. It includes
package validation, scaling, member inference and output-frame construction,
but excludes cache reading, feature extraction, startup and MSA generation.
No predictive score is calculated. The two final model NPZ files are intentionally versioned alongside their JSON
manifests. Other large feature caches remain external to Git. Distribute all
three model files together; prediction also requires compatible feature bundles.
