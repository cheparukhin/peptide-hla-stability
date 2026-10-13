# Boltz tree, blending and preprocessing follow-up

Original-training development only. The original test was not reopened.

Primary R5 hybrid selector minus matched standard MLP: **+0.0044**; paired-cluster 95% interval **[-0.0055, +0.0100]**.

All 480 supervised fits in 80 bundles were audited and their saved model predictions and inner selections replayed. The separately authorized 12-fit engineering pilot is excluded from assessment and never reused.

Both representations use the same two-member schedule for inner and final predictions. Six fitted components cover standard MSE MLP, fixed-budget regression and ranking trees, tree-leaf MLP, quantile MLP and PCA MLP. The RMSE tree is shared with its leaf transform; blending and selection add no supervised fits.

The MSE-first hybrid selector and five fixed blend weights are selected from three clean inner folds. The ranking tree uses a positive affine map learned from its own inner predictions before the outer refit; this transports scale between different fitting-set sizes. Raw inner rank scores are never calibrated on their own labels for blend selection.

All 24 ordinary contrasts and one paired interaction share 2,000 whole-peptide-cluster draws on each 20/50-row panel. Secondary intervals are nominal exploratory intervals, not multiplicity-adjusted discoveries. Uncertainty conditions on fitted models and selection, excluding repeated selection and independent assay variation.

Historical 30-member MLPs and the competitive sequence selector are descriptive references only. Fixed-budget trees and adaptively stopped MLPs differ in training recipe; representation dimensions and parameter counts also differ. These are comparisons of complete procedures, not causal attribution to structure, normalization or a learner alone.

Replay promotes member outputs to float64 before ensemble averaging, matching the actual frozen worker. The earlier report-only float32 aggregation bug was corrected without changing any saved prediction, fit, selection or comparison tolerance.

See [comparisons.csv](comparisons.csv), [scores.csv](scores.csv), [selections.csv](selections.csv), [runtime_summary.json](runtime_summary.json), [audit.json](audit.json) and [protocol.json](protocol.json).

Report replay used the separately authorized stopping-MSE amendment: 328 MLP scalar checks across 82 bundles, 0 nonzero differences, maximum 0 float32 ULP. Only recomputed stopping MSE may differ by one ULP; persisted best-MSE/curve identity, held-prediction equality and all other frozen audit checks remain unchanged. No fitting, selection or statistical output was changed. The amendment binding and every scalar diagnostic are preserved in audit.json.

Saved-model replay used the recorded SkylakeX OpenBLAS backend with two numerical threads. CPU capabilities and actual library identities are in audit.json. All held predictions still had to reproduce exactly; no prediction tolerance was introduced. The cause of the earlier unlogged-container mismatch remains unproven.
