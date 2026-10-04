# Development

Setup, reproduction and repo rules for the peptide–HLA stability project. For
what the project found, see [README.md](../README.md) and
[reports/REPORT.md](../reports/REPORT.md).

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
commands that produced the committed artifacts. Data-side and historical
regeneration commands are in [docs/README.md](README.md#regenerating).

The MSA cache needs boltz, which pulls torch, so it is kept out of `.venv`:

```bash
uv venv /tmp/boltzenv --python 3.12
uv pip install --python /tmp/boltzenv/bin/python boltz
/tmp/boltzenv/bin/python scripts/make_msas.py --all-alleles   # ~2.5 min, $0
```

Structural production and feature extraction run on Modal, not locally — see
[the stage 4c report](../reports/stage4c_ectodomain_pilot.md) for the launch
sequence. The fold is complete.
