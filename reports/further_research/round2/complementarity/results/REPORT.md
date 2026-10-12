# Completed complementarity development results

All five completed outer folds cover exactly **19,716 original training rows**. Protocols, completion hashes, source references, row/cluster/allele identity and ancestry passed validation. Fixed original baseline members were checked against the 30-member recipe. Stored blend/correction choices reproduce their frozen inner-fold rule; zero endpoints reproduce the baseline exactly.

All 16 declared contrasts use **2,000 paired whole-peptide-cluster bootstrap draws**, seed 20261003, shared across every procedure. The 20-row panel is primary; min50 sensitivity and fold-specific statistics are separate files.

| Procedure | Pooled median per-allele Spearman (min20) |
|---|---:|
| mlp_log | 0.661522 |
| sequence_selector | 0.666841 |
| R2_ridge | 0.297055 |
| R4_ridge | 0.684510 |
| R2_blend | 0.661980 |
| R4_blend | 0.723789 |
| R2_residual | 0.655992 |
| R4_residual | 0.711429 |

`paired_comparisons.csv` reports every blend/residual procedure and R2/R4 embedding alone against both original and competitive sequence references, plus each complementary procedure against its embedding-alone arm. Standalone and complementarity gains answer different questions. No arm is selected from these outcomes.

Each assessed cluster was excluded from base fitting, stopping and ridge selection. Residual targets came from fresh clean original cross-fits inside the residual fitting partition. Other outer-fold predictions never supplied stacking inputs. Inner-to-outer training-size differences remain a declared limitation.

Intervals condition on saved fitted predictions and exclude search/execution uncertainty. These previously examined training data provide development evidence, not independent confirmation. `selections.csv`, `coverage.csv`, `per_allele_scores.csv` and `manifest.json` retain the selections, coverage and provenance.
