# Completed round-two development results

Scope: **sequence_and_embedding**. All five outer folds cover exactly **19,716 original training rows** and **3,587 peptide clusters**. Saved predictions, source/fold contracts and ancestry were validated before scoring. Labels are joined through the original training partition only.

All declared comparisons use **2,000 paired whole-peptide-cluster bootstrap draws**, seed 20261003, shared across every procedure. The 20-row panel is primary; the 50-row panel is reported separately. These intervals condition on fitted predictions and do not include search/execution uncertainty or constitute independent confirmation.

| Procedure | Pooled median per-allele Spearman (min20) |
|---|---:|
| mlp_log | 0.661522 |
| mlp_centred | 0.665390 |
| tree_log | 0.624802 |
| tree_centred | 0.624516 |
| pairlogit | 0.651190 |
| sequence_selector | 0.666841 |
| R0_ridge | 0.214862 |
| R0_mlp | 0.591367 |
| R1_ridge | 0.403057 |
| R1_mlp | 0.609964 |
| R2_ridge | 0.297055 |
| R2_mlp | 0.421852 |
| R3_ridge | 0.198800 |
| R3_mlp | 0.386197 |
| R4_ridge | 0.684510 |
| R4_mlp | 0.713592 |
| R5_ridge | 0.666315 |
| R5_mlp | 0.720986 |
| R5_shuffled_ridge | 0.664301 |
| R5_shuffled_mlp | 0.720310 |
| R5_position9x34_ridge | 0.654638 |
| R5_position9x34_mlp | 0.731541 |
| R3_pooled_ridge | 0.021923 |
| R3_pooled_mlp | 0.047484 |
| R5_position9x34_pooled_ridge | 0.393860 |
| R5_position9x34_pooled_mlp | 0.565192 |
| embedding_selector | 0.721163 |

Prespecified primary comparison, embedding selector versus competitive sequence selector: **+0.054322 [+0.031395, +0.080805]**.

Every component/reference contrast is retained in `paired_comparisons.csv`. Matched-head and isolated positional controls have explicit comparison labels; `confounded_component` contrasts change chain context, channels or pooling and cannot isolate positional information. No arm is selected from these report outcomes.

`fold_scores.csv` contains fold-specific medians and panels. Those statistics are separate from the pooled primary statistic; their median is not substituted for it. `coverage.csv` and `per_allele_scores.csv` retain allele eligibility, tied/constant predictions and row coverage.

Sequence paired seed0/1 diagnostics complete: **True**. Sequence identical-seed execution repeats complete: **True**. Embedding paired seed0/1 diagnostics complete: **True**. Available complete diagnostics are in the paired/pooled tables and `execution_differences.csv`, with file hashes verified against completion markers. Incomplete diagnostics receive no partial-cohort comparison.

Protocols, all per-fold inner selections, source/output hashes and audited ancestry are retained in `manifest.json` and `selections.csv`. Seed variation, identical-seed execution variation, paired-cluster uncertainty and assay variation remain distinct; no assay noise ceiling is inferred.
