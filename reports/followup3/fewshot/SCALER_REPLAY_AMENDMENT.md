# Operational amendment: exact saved StandardScaler replay

The first cloud cell-0 pilot stopped during saved-scaler replay after one R0 ridge fit, before efficacy reporting. Its exclusive attempt and NPZ artifact are preserved by Root in the failed-pilot archive; they are not accepted scientific results or reused fits. Original `protocol.json`, task identities, support identities and scientific specification remain intact.

The initial replay manually subtracted the saved float64 mean and divided by the saved float64 scale in a float32 array. The installed scikit-learn1.9.1 `StandardScaler.transform` first casts both fitted statistics to the input dtype. These two arithmetic orders produce different float32 rounding. The trained model already used authoritative `StandardScaler.transform`; the failure was in the handwritten replay verification, not a new scientific preprocessing procedure.

The corrected serialization adds fitted feature/sample metadata to the saved scaler state. Replay restores a StandardScaler with the saved mean, scale, variance, feature count and sample count, verifies exact source-fitting row identity, and calls the estimator's authoritative transform. Saved statistics are checked against the fitted source scaler, and both source and support/query transforms must replay exactly. Ridge predictions still require exact equality. No tolerance has been widened.

A small synthetic regression uses float32 inputs and float64 fitted statistics, reproduces the erroneous arithmetic mismatch, verifies exact authoritative replay after a pickle-free NPZ round trip, and rejects altered source-fitting IDs. No canonical models were fitted locally.

This amendment changes only serialization/replay code. Cohort, grouping, supports, queries, budgets, source recipes, fitted scaling, target adapter, uncertainty, endpoints and grids are unchanged. The metadata-only `amend-replay` command writes a NEW `protocol_replay_v2.json`, binding the original protocol and this amendment, revised source hash and regression-test hash. It refuses changes to other frozen scientific sources, inputs, population, plan or panels. There is no overwrite of the original protocol or automatic scientific retry.

Root must preserve the failed cloud directory before a separately source-bound new admission. Run/report must explicitly select `--protocol reports/followup3/fewshot/protocol_replay_v2.json`. Report evidence cannot mix protocol versions. Paid launch ownership and spending bounds remain with Root.
