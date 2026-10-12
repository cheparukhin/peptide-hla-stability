# Development inference exports and cached-head timing

Two explicitly locked recipes are now usable from verified compact feature
caches: **R4 ridge, alpha 100** and **R5 MLP, L2 0.01**. Both were refit to all
19,716 original training rows. These are development/deployment demonstrations;
neither is declared a global winner or independently confirmed. No validation
or test predictions were produced and no new accuracy score was calculated.

The [prefit authorization](training_authorization.json) locked the 31-fit budget,
configurations, training data, source and feature hashes before any fit. One
local process (PID 58866) ran at two numerical threads with zero cloud calls.
All 30 neural members retained the frozen 200-epoch cap, grouped five-fold
stopping schedule and six seeds. Completed epochs ranged from 24 to 74;
best epochs ranged from 9 to 59. No cap adjustment or new candidate search was
performed. Fit timing was 4.889 seconds for ridge and 34.940 seconds for the
30-member MLP; total worker time was 42.646 seconds including loading, export
and checks.

## Artifact locations

| Recipe | Model directory | Compressed parameter size |
|---|---|---:|
| R4 ridge 100 | `models/finalization/R4_ridge_100` | 76,594 bytes |
| R5 MLP 0.01 | `models/finalization/R5_mlp_0.01` | 8,212,758 bytes |

Each directory contains `parameters.npz`, `model.json` and `complete.json`.
Both models passed exact save/reload prediction equality and finite-output
checks. All files must travel together. These two final parameter NPZ files are intentionally versioned alongside
their JSON manifests. Compatible feature bundles remain external to Git. The [distribution manifest](distribution_manifest.json)
records exact local paths, file hashes, training counts and measured timing.
See [usage and input contract](../../../docs/ROUND2_INFERENCE.md).

## Performance-only benchmark

This is an **in-sample, cached-head performance benchmark**, not a predictive
assessment. The first 64 original training IDs in sorted canonical order are
recorded in [benchmark_pair_ids.txt](benchmark_pair_ids.txt); the manifest also
binds their input and feature hashes. The API was warmed with three 64-row
calls, then measured for 25 calls per batch size at two numerical threads.
Single-row measurements used the first of those IDs. No labels were supplied
to prediction or benchmark scoring.

| Recipe | Single-row median / p95 | 64-row batch median / p95 |
|---|---:|---:|
| R4 ridge 100 | 1.559 / 1.799 ms | 1.651 / 1.843 ms |
| R5 MLP 0.01 | 2.401 / 2.565 ms | 3.870 / 4.079 ms |

These calls include package and feature validation, scaling, member inference
and output-dataframe construction. They exclude cache-file reading, interpreter
startup, MSA generation, Boltz extraction, coordinate prediction and network
transfer. They therefore do not establish new-sequence end-to-end latency or
deployment compute cost. Timings are recorded-platform observations rather
than a cross-platform guarantee. The inherited extraction-wrapper provenance
gap described in the [round-two report](../../further_research/round2/REPORT.md)
also applies to the cached features consumed by these exports.

No further training, tuning or assessment was triggered by the benchmark.
