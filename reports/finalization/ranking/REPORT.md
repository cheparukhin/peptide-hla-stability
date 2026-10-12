# Matched Boltz + ranking follow-up

Previously examined original-training development evidence; original test not reopened.

Primary R5 mixed-loss minus R5 MSE: **+0.0071** median per-allele Spearman; paired-cluster 95% interval **[-0.0051, +0.0098]** (min20).

The source/cohort/features, all 480 reused MSE logical members, all 840 fresh scientific fits, raw inner predictions, scaler statistics, row hashes, and complete fitting/stopping/selection ancestry were audited before scoring. Two separately receipted lambda0 validation fits passed.

Intervals use 2,000 shared whole-cluster draws and condition on the fitted predictions and inner selection. They do not include selection repetition, independent assay variation, or independent confirmation. Min50 and the matched sequence contrast, representation contrasts, and loss-by-representation interaction are reported alongside the primary result.

The positive ranking grid was lambda .1/.3/1 × L2 1/.01/.0001; MSE stopping, 64-unit MLP, natural batches256, Adam .001, max200 epochs, patience15. Each representation/fold used three inner cluster folds and two members per candidate/fold, then the selected candidate's 30-member final ensemble. The deployment selector preferred MSE within one fold-SE using inner scores only.

MSE anchors the log1p half-life scale; no affine mapping or pure-ranking claim is made. R0 and R5 feature dimensions/parameter counts differ. This result does not authorize transfer expansion, additional lambda values, stopping changes, or repeats.

See [comparisons.csv](comparisons.csv), [scores.csv](scores.csv), [selections.csv](selections.csv), [protocol.json](protocol.json), and [audit.json](audit.json).
