# Cached-feature inference

This package exports reusable **R4 ridge** and **R5 MLP** prediction heads that consume verified Boltz compact feature caches. It does not create features from a novel peptide/HLA sequence, call Boltz, produce structures, or run an MSA search; new-sequence inference still requires the separate pinned extraction workflow and its unmeasured new-case costs (see the [research report](../reports/REPORT.md)).

The two demonstration recipes—`R4_ridge_100` (one fit) and `R5_mlp_0.01` (30 fits)—are explicitly locked development refits trained only on the original training partition. Neither recipe is selected from follow-up outer outcomes or designated a global winner. Because these exports do not reproduce the nested embedding selector, they do not inherit its reported 0.7212 score and carry no new held-out accuracy estimate.

## Environment setup

From the repository root on Linux or macOS, use **Python 3.12.12** to recreate the local environment used by the commands below. The core pins match the [saved head metadata](../models/finalization/R5_mlp_0.01/model.json) and [recorded runtime](../reports/finalization/transfer_controls/local_timing/runtime.json), while supporting dependency pins come from the recorded environment package inventory.

```sh
python3.12 -m venv .venv
.venv/bin/python -c 'import sys; assert sys.version_info[:3] == (3, 12, 12)'
.venv/bin/python -m pip install \
  numpy==2.5.3 pandas==3.0.6 scipy==1.18.1 scikit-learn==1.9.1 \
  threadpoolctl==3.7.0 joblib==1.6.0 python-dateutil==2.9.0.post0 six==1.17.0

# Optional: the synthetic verification suite documented below.
.venv/bin/python -m pip install \
  pytest==9.1.1 iniconfig==2.3.0 packaging==26.3 pluggy==1.6.0 pygments==2.21.0
```

This environment covers the cached-head CLI and its synthetic checks, whereas feature extraction uses separate pinned Boltz/ESM runtimes. Large feature caches remain external to Git and must be supplied as verified compact bundles; installing packages does not recreate them. Keep local work to lightweight inspection and checks, and run refits or large inference jobs on Modal within an explicitly authorized resource and credit budget. Matching package versions alone does not guarantee identical numerical results across platforms or BLAS libraries.

## Input contract

Pass either a compact bundle directory containing `complete.json`, its content-addressed NPZ, and its manifest, or a complete production JSON manifest listing `feature_bundles`. The existing full training input is `reports/further_research/round2/extractor/full_training/feature_bundles.json`. Bare feature matrices and sequence strings are intentionally unsupported.

Before prediction, the tool verifies file hashes, the feature protocol, the contact map, and array dimensions, and confirms that array identities match the manifest. Each bundle must declare schema `pepstab-round2-representations-v1` and contain finite float32 vectors, unique ordered `pair_id` strings, and a nonempty `input_hash` per row. Both representations use pooling without channel projection: R4 has 2,304 columns in `[P1..P9, peptide-to-groove/groove-to-peptide, 128 channels]` order, whereas R5 has 1,024 columns in `[P1,P2,P9,mean(P3..P8), direction,128 channels]` order using the frozen crystal contact map.

The optional `--pair-ids` file (one ID per line) selects a row subset and sets the output order; missing or duplicate requested IDs fail. Prediction outputs preserve `pair_id` and `input_hash` and return `prediction_y_log1p`, an uncalibrated prediction on the `log1p(thalf_hours)` scale. Because the tool does not silently clamp or calibrate outputs, values may be negative and are not guaranteed physical half-life estimates.

## Portable artifacts

Each model directory contains `parameters.npz`, `model.json`, and `complete.json`, which checksums both other files. Model and output paths are immutable, so each new run requires a new path. The NPZ archive is read with `allow_pickle=False` so no executable serialized objects are used; it stores weights, biases, target offsets, and exact per-member scaler means, scales, and variances. Fixed token/pair projection matrices are also preserved for provenance, although these two representations do not use them. The `model.json` metadata records the complete candidate configuration, training row/label/feature hashes, source hashes, versions, stopping-fold ancestry, authorization, and the feature/contact-map contract.

R4 matches the frozen `Ridge(alpha=100, solver='lsqr', tol=1e-5, max_iter=2000)` recipe with a scaler fitted on all training rows. R5 matches the main study's 64-unit ReLU MLP (L2 0.01, Adam learning rate 0.001, batch 256, epoch cap 200, and patience 15), averaging five deterministic peptide-cluster stopping folds times seeds 0..5. Each MLP member's scaler sees only that member's gradient rows. This export implements the main study recipe and does not reproduce the HLA or joint outer assessment procedures or their selection. Predictions use the stored float32 scaler arithmetic and member order. Exact save/load equality is verified on the recorded platform, though cross-platform numerical identity is not promised.

## Commands

Show the fixed configuration and fit count without training:

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

For an authorised refit on Modal, the worker invocation is:

```sh
.venv/bin/python scripts/round2_inference.py fit \
  --candidate R4_ridge_100 --max-fits 1 \
  --features reports/further_research/round2/extractor/full_training/feature_bundles.json \
  --training-csv /tmp/original_training_rows.csv \
  --authorization reports/finalization/inference/training_authorization.json \
  --out /tmp/r4_development_refit
```

The training CSV must contain `pair_id`, `cluster_id`, `split`, and `y_log1p`. Before any fit, the CLI verifies these columns against canonical original training rows; validation/test rows or altered labels/clusters fail. The CLI also limits numerical work to two threads and writes a prefit authorization record with data/feature/source hashes before fitting. One candidate is exported per call (1/30 fits are the full recipes, not grid-search budgets), and the tool never chooses a candidate. For synthetic developer checks, the Python API relies on its caller to supply original-train membership while enforcing the marked training split, finite labels, row identity, and five nonempty grouped stopping folds.

## Verification and measured scope

`tests/test_round2_inference.py` checks exact prediction parity with the existing frozen `FeatureHead` on synthetic data, all 30 MLP members, exact serialization, budget rejection, training-scope/identity/provenance errors, corrupted files, and path traversal rejection. Run:

```sh
.venv/bin/python -m pytest -q tests/test_round2_inference.py
```

The [distribution manifest](../reports/finalization/inference/distribution_manifest.json) records the packaged models and timing measurements. This timing check is a **performance-only, in-sample cached-head benchmark** on identified training cache rows: it includes package validation, scaling, member inference, and output-frame construction, but excludes cache reading, feature extraction, startup, and MSA generation, and calculates no predictive score. The two final model NPZ files are intentionally versioned alongside their JSON manifests, whereas other large feature caches remain external to Git. Distribute all three model files together; prediction also requires compatible feature bundles.
