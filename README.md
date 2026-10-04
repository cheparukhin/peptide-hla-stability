# Peptide–HLA stability

Predicting peptide–HLA dissociation half-life (London AI × Science protein
engineering track, 3–4 October 2026). HLA molecules hold short protein fragments
at the cell surface for immune inspection; we predict how long a 9-residue
peptide stays bound, measured as its dissociation half-life.

**Read [reports/REPORT.md](reports/REPORT.md) first.** It is the written-up
answer, with every claim sourced to the stage report behind it.
[HACKATHON_PLAN.md](HACKATHON_PLAN.md) has scope and stage order;
[EVALUATION.md](EVALUATION.md) is the evaluation contract, frozen at stage 1;
[reports/limitations.md](reports/limitations.md) is everything to discount the
result by; [reports/compute_ledger.md](reports/compute_ledger.md) is every
dollar and GPU-hour.

## Result

All deltas below are against the **sequence ensemble** reference (30 networks on
the peptide plus 34 HLA contact residues). The sequence ensemble leads: median
per-allele Spearman **0.693 on validation** and **0.7064 on test**. Frozen ESM-2
does not establish a worthwhile gain — on test, ESM-2 35M is −0.009 [−0.029,
+0.023] and sequence + ESM-2 is −0.025 [−0.046, +0.007], both intervals crossing
zero and both excluding the predeclared +0.05 bar. The Boltz-2 structural arm is
conclusively worse (test −0.112 [−0.134, −0.057]) despite dominating project
compute at **$212.71** for the full-cohort fold. Test agrees with validation.

## Status

Project complete. The frozen test split was scored **once**, on 4 October 2026
at 13:35 BST ([runbook](reports/TEST_SCORING_RUNBOOK.md); completion manifest
`reports/stage6_test_manifest.json`). The Boltz-2 production fold is done:
28,166 folds, 0 failures. Three items are genuinely open:

- Provider-bill reconciliation of the $212.71 fold cost (the figure is derived
  from recorded container-hours × the measured worker rate, not a vendor bill).
- The elution transfer pass (stage 3c) on the ESM-2 and structural arms; only
  the sequence arm has been scored.
- Stage 3d ESM-2 likelihood features — not run, so the stage 3 conclusion is
  bounded to embeddings.

## Dataset and evaluation

| Property | Value |
|---|---:|
| Peptide–HLA pairs | 28,166 |
| Unique 9-mer peptides | 5,633 |
| HLA alleles | 75 |
| Labels at 0 h (assay floor) | 20.2% |
| Train / validation / test rows | 19,716 / 2,817 / 5,633 |

Peptides are grouped by Hamming ≤ 3 single-linkage before splitting; the minimum
cross-split peptide Hamming distance is 4. Scoring covers eligible alleles only:
68 alleles / 2,802 rows on validation, 67 / 5,565 on test. The primary metric is
median per-allele Spearman; the predeclared worthwhile-gain bar is Δ +0.05. An
interval crossing zero is **inconclusive, not negative**. See
[EVALUATION.md](EVALUATION.md) for the frozen contract and the disclosed earlier
test-exposure diagnostic.

## Stage index

Per-stage detail lives in each report; this table replaces the long prose.
Validation unless a row says otherwise. Deltas use the sequence ensemble.

| Stage | Outcome | Report |
|---|---|---|
| 1 data audit + frozen splits | Alphabets, construct, zero-label pile-up, split design; evaluation contract frozen | [audit_summary](reports/audit_summary.md), [EVALUATION](EVALUATION.md) |
| 2 sequence baselines | Sequence ensemble **0.693**; single network 0.610; ensembling alone worth +0.074 mean SCC | [stage2_baselines](reports/stage2_baselines.md) |
| 2b weak-binder augmentation | No gain; best arm +0.024, all intervals cross zero and exclude +0.05 | [stage2b_augmentation](reports/stage2b_augmentation.md) |
| 2c auxiliary affinity labels | No gain; affinity alone ranks stability at ρ 0.580, below labels | [stage2c_affinity](reports/stage2c_affinity.md) |
| 3 frozen ESM-2 embeddings | No worthwhile gain; 35M 0.683 (−0.010), seq+ESM 0.676 (−0.017); both exclude +0.05 | [stage3_esm](reports/stage3_esm.md) |
| 3b ESM-2 × affinity | Unresolved; difference-in-differences power at the bar unestablished | [stage3b_esm_multitask](reports/stage3b_esm_multitask.md) |
| 3c elution transfer (sequence arm) | Median AUROC **0.9656** over 51 alleles; ESM-2/structural pass not run | [stage3c_elution_validation](reports/stage3c_elution_validation.md) |
| 3d ESM-2 likelihood features | **Not run** — stage 3 conclusion bounded to embeddings | — |
| 4a/4b GPU benchmark + two-chain pilot | A10 selected; three folding lessons carry forward | [stage4_benchmark](reports/stage4_benchmark.md) |
| 4b.1 groove MSA cache | All 75 alleles cached, 182-residue groove, $0 | [stage4b1_msa_cache](reports/stage4b1_msa_cache.md) |
| 4c ectodomain fold | Boltz-2 passed the pilot gate, ESMFold2 failed; production complete, 28,166 folds, **$212.71** | [stage4c_ectodomain_pilot](reports/stage4c_ectodomain_pilot.md) |
| 4c.5 structural features | Full pass complete, 28,166 rows, 0 failures, **109 numeric features**, ~$1.18 | [stage4c5_features](reports/stage4c5_features.md) |
| 5 inverse folding (ProteinMPNN) | Structural arm conclusively worse (test −0.112); pep log-likelihood is a pose-failure triage signal, not a predictor; QC sample run, full scoring declined | [stage5_inverse_folding](reports/stage5_inverse_folding.md) |
| 6 evaluation machinery + test scoring | Machinery exercised, then test scored once; test agrees with validation | [stage6_evaluation_machinery](reports/stage6_evaluation_machinery.md), [stage6_val_esm_run](reports/stage6_val_esm_run.md) |
| 7a censored (Tobit) loss | Conclusively worse ranking, Δ −0.0414 [−0.0780, −0.0062]; better floor calibration | [stage7_censored](reports/stage7_censored.md) |
| 7b leave-allele-out | Near 0.741 / distant 0.339, near − distant +0.403 [+0.223, +0.478]; separate contract, not comparable to frozen-split numbers | [stage7_allele_holdout](reports/stage7_allele_holdout.md) |
| 8 FoldX empirical energy | Pilot only; RepairPDB required; the arm was not scored for half-life | [stage8_foldx](reports/stage8_foldx.md) |

NetMHCstabpan is calibration only, never a comparator: it trained on our test
rows.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy pandas scipy scikit-learn pyarrow pytest
```

## Quick start

```python
from pepstab.data import load_with_splits

df = load_with_splits()                 # never recompute the splits
train = df[df.split == "train"]
# inputs: peptide, hla_seq, hla_pseudoseq   target: y_log1p
```

Write predictions as `pair_id,y_pred` on the `log1p` scale, then:

```bash
.venv/bin/python scripts/evaluate.py --split val \
    preds/seq_ensemble_pep_pseudo.csv preds/esm_ensemble.csv
```

The first file is the baseline; the rest get paired cluster-bootstrap CIs
against it. For the full table set — distance strata, differential concordance,
nested mutant ranking, precision@10 — use the stage 6 CLI:

```bash
.venv/bin/python scripts/stage6_report.py --split val \
    --n-boot 2000 --stratum-min-rows 10 --nested \
    seq_ensemble_pep_pseudo=preds/seq_ensemble_pep_pseudo.csv \
    esm_ensemble=preds/esm_ensemble.csv \
    esm_plus_seq_ensemble=preds/esm_plus_seq_ensemble.csv
```

The first arm is the baseline every interval is taken against; a structural
prediction file drops straight in. `--stratum-min-rows 10` is the validation
bar; test keeps the frozen 20.

## Rules

- `data/rasmussen_et_al_dataset.csv` is read-only;
  `(cd data && shasum -a 256 -c SHA256SUMS)` must pass. Derived data goes to a
  new file.
- Load splits from `data/splits.csv`, never recompute them. Regenerating drops
  the peptide-cluster grouping and leaks training data into test. Join on
  `(allele, peptide)`, not on `pair_id`.
- Select on validation. The test set is scored once, at stage 6.
- Ensemble every arm identically, or ensemble none of them. Ensembling alone is
  worth +0.074 mean SCC against the deployed single network, from no new
  information. Always state which reference a delta uses.
- Augmented rows in `data/augmentation/` are assumed labels (`thalf_hours = 0`),
  not measurements. They enter the fit set only, never validation or test, and
  never overwrite a measured value. Verify any manifest with
  `pepstab.augment.verify_manifest` first.
- Load the frozen production cohort from `data/structural_cohort.csv`, never
  recompute it. Each Modal profile folds its own half; `--profile` must match
  `MODAL_PROFILE`.
- No batch GPU job without a passing pilot on 3–5 examples. That applies to a
  new runner as well as a new model.

## Regenerating

```bash
.venv/bin/python scripts/make_splits.py        # deterministic; rewrites data/splits.csv
.venv/bin/python scripts/audit_data.py         # rewrites reports/audit_summary.md
.venv/bin/python scripts/baseline_constant.py  # constant reference baselines
.venv/bin/python scripts/baseline_sequence.py  # stage 2 grid, ~10 CPU-minutes
.venv/bin/python scripts/baseline_ensemble.py  # 30-network ensemble baseline, ~1 min
.venv/bin/python scripts/compare_to_paper.py   # NetMHCstabpan calibration, ~3 min
.venv/bin/python scripts/augment_affinity.py   # stage 2b manifests, ~40 s (downloads a proteome)
.venv/bin/python scripts/baseline_augmented.py # stage 2b arms + intervals, ~13 min
.venv/bin/python scripts/stage2b_negatives.py  # stage 2b pool characterisation, ~10 s
.venv/bin/python scripts/affinity_multitask.py # stage 2c probe, ~10 CPU-minutes
.venv/bin/python scripts/esm_features.py --verify-index              # stage 3 contact indexing
.venv/bin/python scripts/esm_features.py --checkpoint esm2_t12_35M_UR50D
.venv/bin/python scripts/esm_arm.py sweep --checkpoint esm2_t12_35M_UR50D
.venv/bin/python scripts/stage3b_esm_multitask.py      # stage 3b; exact flags in its report
.venv/bin/python scripts/stage3c_elution_validation.py # stage 3c external pass
.venv/bin/python scripts/stage6_report.py --split val ...   # stage 6 tables
.venv/bin/python scripts/stage7_censored.py            # stage 7a, 130 networks, ~30 CPU-minutes
.venv/bin/python scripts/stage7_allele_holdout.py      # stage 7b, 68 folds × 6 networks
.venv/bin/python reports/figures/make_figures.py       # figures, CPU, no network
.venv/bin/python -m pytest tests/ -q                   # 540 guards collected
```

Several stages have more flags than fit here; each report gives the exact
commands that produced the committed artifacts.

The MSA cache needs boltz, which pulls torch, so it is kept out of `.venv`:

```bash
uv venv /tmp/boltzenv --python 3.12
uv pip install --python /tmp/boltzenv/bin/python boltz
/tmp/boltzenv/bin/python scripts/make_msas.py --all-alleles   # ~2.5 min, $0
```

Structural production and feature extraction run on Modal, not locally — see
[the stage 4c report](reports/stage4c_ectodomain_pilot.md) for the launch
sequence. The fold is complete.
