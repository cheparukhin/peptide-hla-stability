# Experimental Reports and Evidence Directory

This directory contains the main scientific paper ([`REPORT.md`](REPORT.md)), executive briefs ([`finalization/`](finalization/)), and all underlying CSV/JSON experimental outputs across Stage I and Stage II of the project.

---

## 1. Narrative Documents

- **[`REPORT.md`](REPORT.md)**: Full scientific paper covering the biological problem, dataset and split construction, representation hierarchy, main experimental findings, zero-shot immunopeptidomics transfer, cross-allele transfer, 20-shot adaptation, secondary/negative extensions, and limitations.
- **[`finalization/SUBMISSION_BRIEF.md`](finalization/SUBMISSION_BRIEF.md)**: Concise executive summary of all experimental findings and quantitative tables.
- **[`finalization/INDEPENDENT_ASSAY_BRIEF.md`](finalization/INDEPENDENT_ASSAY_BRIEF.md)**: Prospective experimental protocol for blinded wet-lab dissociation testing.

---

## 2. Key Empirical Result Tables by Study Section

### Stage I: Test-Split Baselines, Naive Foundation Models, and Zero-Shot Elution Transfer
| Study Component | Primary Artifacts |
| :--- | :--- |
| **Sequence baselines (`stage2`)** | [`stage2_summary.csv`](stage2_summary.csv), [`stage2_headline.json`](stage2_headline.json), [`stage2_ensemble_pep_pseudo.csv`](stage2_ensemble_pep_pseudo.csv), [`stage2_ensemble_pep_domain.csv`](stage2_ensemble_pep_domain.csv) |
| **Frozen ESM-2 35M & 150M (`stage3`, `stage3e`, `stage3f`)** | [`stage3_comparisons.csv`](stage3_comparisons.csv), [`stage3_embedding_cost.csv`](stage3_embedding_cost.csv), [`stage3e_comparisons.csv`](stage3e_comparisons.csv), [`stage3f_comparisons.csv`](stage3f_comparisons.csv) |
| **Coarse Boltz-2 3D geometry & confidence (`stage4c5`, `stage5`)** | [`stage5_structural_bootstrap.csv`](stage5_structural_bootstrap.csv), [`stage5_structural_val.csv`](stage5_structural_val.csv), [`gpu_decision.csv`](gpu_decision.csv) |
| **Zero-shot MHC Motif Atlas elution transfer (`stage3c`)** | [`stage3c_summary.csv`](stage3c_summary.csv), [`stage3c_per_allele_seq_ensemble.csv`](stage3c_per_allele_seq_ensemble.csv), [`stage3c_per_allele_seq_ensemble__swapped_decoys.csv`](stage3c_per_allele_seq_ensemble__swapped_decoys.csv), [`stage3c_per_allele_seq_ensemble__wrong_allele.csv`](stage3c_per_allele_seq_ensemble__wrong_allele.csv), [`stage3c_donor_distance_seq_ensemble.csv`](stage3c_donor_distance_seq_ensemble.csv) |
| **Censored Tobit calibration (`stage7_censored` & `Phase B`)** | [`stage7_censored_calibration.csv`](stage7_censored_calibration.csv), [`stage7_censored_reliability.csv`](stage7_censored_reliability.csv), [`further_research/phase_b_paired.csv`](further_research/phase_b_paired.csv) |
| **Equilibrium affinity reference & multitask (`stage2b`, `stage2c`, `stage3b`)** | [`stage2c_affinity_ceiling.csv`](stage2c_affinity_ceiling.csv), [`stage2b_deltas.csv`](stage2b_deltas.csv), [`stage2c_deltas_ensemble_pep_pseudo.csv`](stage2c_deltas_ensemble_pep_pseudo.csv), [`stage3b_deltas.csv`](stage3b_deltas.csv) |

### Stage II: Internal Boltz-2 Trunk Representations, Transfer, Adaptation, and Extensions (19,716-Pair Training Cohort)
| Study Component | Primary Artifacts |
| :--- | :--- |
| **Nested 5-fold peptide-cluster CV (`R2`–`R5`, `full_interface_pos`, blends, ablations)** | [`further_research/round2/results/main/paired_comparisons.csv`](further_research/round2/results/main/paired_comparisons.csv), [`further_research/round2/results/main/declared_contrasts.csv`](further_research/round2/results/main/declared_contrasts.csv), [`further_research/round2/results/main/pooled_scores.csv`](further_research/round2/results/main/pooled_scores.csv), [`further_research/round2/complementarity/results/paired_comparisons.csv`](further_research/round2/complementarity/results/paired_comparisons.csv) |
| **Unseen HLA-group & joint holdout transfer ($5 \times 5$ crossed CV)** | [`further_research/round2/transfer/results/comparisons.csv`](further_research/round2/transfer/results/comparisons.csv), [`further_research/round2/transfer/results/leave_one_hla_group_out.csv`](further_research/round2/transfer/results/leave_one_hla_group_out.csv), [`finalization/transfer_cloud_recovery/results/comparisons.csv`](finalization/transfer_cloud_recovery/results/comparisons.csv) |
| **Label-budget efficiency (25%, 50%, 100% peptide clusters)** | [`finalization/label_efficiency/comparisons.csv`](finalization/label_efficiency/comparisons.csv), [`finalization/label_efficiency/absolute_curves.csv`](finalization/label_efficiency/absolute_curves.csv) |
| **Few-shot target-allele adaptation ($k \in \{0, 5, 10, 20\}$)** | [`followup3/fewshot/absolute_curves.csv`](followup3/fewshot/absolute_curves.csv), [`followup3/fewshot/comparisons.csv`](followup3/fewshot/comparisons.csv), [`followup3/fewshot/primary_family3_interval.json`](followup3/fewshot/primary_family3_interval.json) |
| **ProteinMPNN (`P`, `C`) and FoldX (`E`) descriptors** | [`followup3/inverse_execution_v2/comparisons.csv`](followup3/inverse_execution_v2/comparisons.csv), [`followup3/energy_execution_v2/comparisons.csv`](followup3/energy_execution_v2/comparisons.csv), [`followup3/inverse_fulltrain_candidate/comparisons.csv`](followup3/inverse_fulltrain_candidate/comparisons.csv) |
| **Within-allele RankNet & hybrid tree/MLP selectors** | [`finalization/ranking/comparisons.csv`](finalization/ranking/comparisons.csv), [`finalization/hybrids_cloud_v1/comparisons.csv`](finalization/hybrids_cloud_v1/comparisons.csv) |
| **Diffusion-seed pose variance vs trunk stability (`Phase D`)** | [`further_research/phase_d_pose_seed_variance.csv`](further_research/phase_d_pose_seed_variance.csv), [`further_research/phase_d_feature_variance_partition.csv`](further_research/phase_d_feature_variance_partition.csv) |
