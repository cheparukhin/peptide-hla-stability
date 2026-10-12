# Completed round-two development results

Scope: **sequence_only**. All five outer folds cover exactly **19,716 original training rows** and **3,587 peptide clusters**. Saved predictions, source/fold contracts and ancestry were validated before scoring. Labels are joined through the original training partition only.

All declared comparisons use **2,000 paired whole-peptide-cluster bootstrap draws**, seed 20261003, shared across every procedure. The 20-row panel is primary; the 50-row panel is reported separately. These intervals condition on fitted predictions and do not include search/execution uncertainty or constitute independent confirmation.

| Procedure | Pooled median per-allele Spearman (min20) |
|---|---:|
| mlp_log | 0.661522 |
| mlp_centred | 0.665390 |
| tree_log | 0.624802 |
| tree_centred | 0.624516 |
| pairlogit | 0.651190 |
| sequence_selector | 0.666841 |

Embedding results were not included. The embedding comparison remains pending; no incomplete embedding-fold outcomes were examined.

Every component/reference contrast is retained in `paired_comparisons.csv`. Matched-head and isolated positional controls have explicit comparison labels; `confounded_component` contrasts change chain context, channels or pooling and cannot isolate positional information. No arm is selected from these report outcomes.

`fold_scores.csv` contains fold-specific medians and panels. Those statistics are separate from the pooled primary statistic; their median is not substituted for it. `coverage.csv` and `per_allele_scores.csv` retain allele eligibility, tied/constant predictions and row coverage.

Sequence paired seed0/1 diagnostics complete: **True**. Sequence identical-seed execution repeats complete: **True**. Embedding paired seed0/1 diagnostics complete: **False**. Available complete diagnostics are in the paired/pooled tables and `execution_differences.csv`, with file hashes verified against completion markers. Incomplete diagnostics receive no partial-cohort comparison.

Protocols, all per-fold inner selections, source/output hashes and audited ancestry are retained in `manifest.json` and `selections.csv`. Seed variation, identical-seed execution variation, paired-cluster uncertainty and assay variation remain distinct; no assay noise ceiling is inferred.
