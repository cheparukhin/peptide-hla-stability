# reports

Results for the peptide–HLA stability project. [REPORT.md](REPORT.md) is the
authoritative write-up; the stage reports below hold the supporting evidence,
and the scored `.csv` / `.json` tables beside them hold the numbers each report
cites. [../HACKATHON_PLAN.md](../HACKATHON_PLAN.md) is the source of truth for
scope and stage order; [../docs/README.md](../docs/README.md) documents the data.

| document | covers |
| --- | --- |
| [REPORT.md](REPORT.md) | the write-up — problem, evaluation contract, validation and test comparisons, transfer, cost, limitations |
| [limitations.md](limitations.md) | the limitations register: everything to discount the result by |
| [compute_ledger.md](compute_ledger.md) | every dollar, CPU-minute and GPU-hour, plus cost per 1,000 new predictions |
| [TEST_SCORING_RUNBOOK.md](TEST_SCORING_RUNBOOK.md) | the single-pass test procedure, fixed before the test set was scored |

## Stage index

Outcomes are **validation** unless a row says otherwise, and deltas are against
the sequence ensemble.

| Stage | Outcome | Report |
|---|---|---|
| 1 data audit + frozen splits | Alphabets, construct, zero-label pile-up, split design; evaluation contract frozen | [audit_summary](audit_summary.md), [EVALUATION](../EVALUATION.md) |
| 2 sequence baselines | Sequence ensemble **0.693**; single network 0.610; ensembling alone worth +0.074 mean SCC | [stage2_baselines](stage2_baselines.md) |
| 2b weak-binder augmentation | No gain; best arm +0.024, all intervals cross zero and exclude +0.05 | [stage2b_augmentation](stage2b_augmentation.md) |
| 2c auxiliary affinity labels | No gain; affinity alone ranks stability at ρ 0.580, below labels | [stage2c_affinity](stage2c_affinity.md) |
| 3 frozen ESM-2 embeddings | No worthwhile gain; 35M 0.683 (−0.010), seq+ESM 0.676 (−0.017); both exclude +0.05 | [stage3_esm](stage3_esm.md) |
| 3b ESM-2 × affinity | Unresolved; difference-in-differences power at the bar unestablished | [stage3b_esm_multitask](stage3b_esm_multitask.md) |
| 3c elution transfer (sequence arm) | Median AUROC **0.9656** over 51 alleles; ESM-2/structural pass not run | [stage3c_elution_validation](stage3c_elution_validation.md) |
| 3d ESM-2 likelihood features | **Not run** — stage 3 conclusion bounded to embeddings | — |
| 4a/4b GPU benchmark + two-chain pilot | A10 selected; three folding lessons carry forward | [stage4_benchmark](stage4_benchmark.md) |
| 4b.1 groove MSA cache | All 75 alleles cached, 182-residue groove, $0 | [stage4b1_msa_cache](stage4b1_msa_cache.md) |
| 4c ectodomain fold | Boltz-2 passed the pilot gate, ESMFold2 failed; production complete, 28,166 folds, **$212.71** | [stage4c_ectodomain_pilot](stage4c_ectodomain_pilot.md) |
| 4c.5 structural features | Full pass complete, 28,166 rows, 0 failures, **109 numeric features**, ~$1.18 | [stage4c5_features](stage4c5_features.md) |
| 5 inverse folding (ProteinMPNN) | Structural arm conclusively worse (test −0.112); peptide log-likelihood is a pose-failure triage signal, not a predictor; QC sample run, full scoring declined | [stage5_inverse_folding](stage5_inverse_folding.md) |
| 6 evaluation machinery + test scoring | Machinery exercised, then test scored once; test agrees with validation | [stage6_evaluation_machinery](stage6_evaluation_machinery.md), [stage6_val_esm_run](stage6_val_esm_run.md) |
| 7a censored (Tobit) loss | Conclusively worse ranking, Δ −0.0414 [−0.0780, −0.0062]; better floor calibration | [stage7_censored](stage7_censored.md) |
| 7b leave-allele-out | Near 0.741 / distant 0.339, near − distant +0.403 [+0.223, +0.478]; separate contract, not comparable to frozen-split numbers | [stage7_allele_holdout](stage7_allele_holdout.md) |
| 8 FoldX empirical energy | Pilot only; RepairPDB required; the arm was not scored for half-life | [stage8_foldx](stage8_foldx.md) |

## Figures

| figure | built by | used in |
| --- | --- | --- |
| [`figures/report_test_comparisons.png`](figures/report_test_comparisons.png) | [`figures/make_report_comparison.py`](figures/make_report_comparison.py) | [REPORT.md §7](REPORT.md#7-test-results), the top-level README |
| [`figures/fig2_cost_vs_accuracy.png`](figures/fig2_cost_vs_accuracy.png) | [`figures/make_figures.py`](figures/make_figures.py) | [compute_ledger.md §3](compute_ledger.md#3-cost-per-1000-new-predictions) |
| [`figures/fig1_per_allele_spearman.png`](figures/fig1_per_allele_spearman.png) | [`figures/make_figures.py`](figures/make_figures.py) | per-allele spread; not currently cited |
