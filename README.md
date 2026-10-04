# Peptide–HLA stability

Predicting peptide–HLA dissociation half-life.
[HACKATHON_PLAN.md](HACKATHON_PLAN.md) has scope and stages;
[EVALUATION.md](EVALUATION.md) has the evaluation rules.

Structural work folds a **383-residue, three-chain construct**: 275-residue HLA
ectodomain + 99-residue beta2m + 9-residue peptide. The matched 90-fold pilot
(45 per model) is **complete**: Boltz-2 passed its gate, ESMFold2 failed on the
sentinel complex, and production is frozen as **Boltz-2 over all 28,166 pairs,
split across two Modal workspaces**. See
[stage 4c of the main plan](HACKATHON_PLAN.md#4c-ectodomain--beta-2-microglobulin-folding),
the [execution checklist](docs/ECTODOMAIN_FOLDING_PLAN.md), and the
[pilot report](reports/stage4c_ectodomain_pilot.md). The production run has not
been launched; earlier two-chain measurements remain historical evidence.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy pandas scipy scikit-learn pyarrow pytest
```

## Stage 1 output (done — read before writing a model)

| Artifact | Description |
|---|---|
| [reports/audit_summary.md](reports/audit_summary.md) | Data audit: alphabets, constructs, zero-label pile-up, split design |
| `data/splits.csv` | Frozen splits: `pair_id, allele, peptide, cluster_id, split, dist_to_train` |
| [EVALUATION.md](EVALUATION.md) | Predeclared metrics, eligible alleles, 0.05 improvement threshold |
| `pepstab/` | Shared library: data loading, splits, scoring |
| `tests/test_contract.py` | 51 guards on the above |

## Stage 2 output (done — the baseline stage 3 must beat)

| Artifact | Description |
|---|---|
| [reports/stage2_baselines.md](reports/stage2_baselines.md) | Six sequence arms, paired CIs, cost, limitations |
| `reports/stage2_runs.csv` | Every run in the grid; `stage2_summary.csv` is the selection |
| `preds/seq_ensemble_pep_pseudo.csv` | **The baseline stage 3 must beat**: 30-network ensemble, median per-allele rho 0.693 |
| `preds/seq_baseline.csv` | Single-network headline arm (rho 0.610); the arm/encoding comparison |
| `pepstab/features.py` | One-hot and BLOSUM62 encodings, cached per unique sequence |
| `pepstab/mlp.py` | Small numpy MLP; stops on a caller-supplied fold |
| `tests/test_baselines.py` | 25 guards, including the fit/dev leakage check |
| `reports/compare_to_paper.csv` | Calibration against NetMHCstabpan, factor by factor |

Validation median per-allele Spearman: **0.693** (30-network ensemble, the
NetMHCstabpan method under our splits), **0.610** (single network), **0.278**
(ridge on identical features), **0.000** (training allele mean). Full-domain
input ties the pseudosequence within noise, so stage 3 must compare domain
embeddings against *both*. Ensembling alone is worth +0.090 mean SCC, so
stage 3 must ensemble its arm the same way.

## Stage 2b output (done — augmentation does not help)

| Artifact | Description |
|---|---|
| [reports/stage2b_augmentation.md](reports/stage2b_augmentation.md) | Measured vs predicted weak-binder negatives, paired intervals, and the mechanism |
| `data/augmentation/*.csv` | Three candidate manifests with per-row provenance; `provenance.json` carries seeds, the leakage ledger and digests |
| `reports/stage2b_arms.csv` | The seven arms; `stage2b_deltas.csv` the eight paired intervals |
| `reports/stage2b_negatives.csv` | Anchor composition and what the baseline already predicts per pool |
| `pepstab/augment.py` | Candidate filtering, the Hamming ≥ 4 exclusion, manifest verification |
| `tests/test_augmentation.py` | 47 guards, including the weighted-loss and distance-rule checks |

Neither source helps: best arm +0.024, **all eight paired intervals cross zero
and exclude 0.05**. The reason is that the two sources supply different kinds of
negative. Measured weak binders have near-canonical anchors and the model already
scores them near the floor. Predicted ones from random peptides have the wrong
anchors and the model already scores them *below* the floor — they're easier than
the easiest real data. Stage 3 runs unaugmented; stage 6 scores no augmented
model.
## Stage 2c output (done — negative, and bounded)

| Artifact | Description |
|---|---|
| [reports/stage2c_affinity.md](reports/stage2c_affinity.md) | Auxiliary affinity head: λ sweep, paired CIs, the ceiling diagnostic, the leakage audit, limitations |
| `reports/stage2c_runs_*.csv` | Every run; `stage2c_deltas_*.csv` are the paired CIs |
| `reports/stage2c_affinity_ceiling.csv` | What measured affinity can rank on its own |
| `reports/stage2c_expansion_audit.csv` | Hamming-distance exclusion table for the declined expansion |
| `pepstab/multitask.py` | Two-headed MLP; at λ=0 **bit-identical** to `pepstab/mlp.py`, and its ensemble to `preds/seq_ensemble_pep_pseudo.csv` |
| `pepstab/affinity.py` | Affinity target transform, dual-labelled join, Hamming ≤ 3 leakage filter |
| `tests/test_multitask.py` | 17 guards, including the λ=0 parity check and the C67S exclusion |

Auxiliary affinity labels **do not help**. Across 20 paired comparisons (5 λ ×
2 encodings × {single network, 30-network ensemble}, plus a censoring variant),
every 95% CI crosses zero and every upper bound is below 0.05 — largest +0.034,
so the worthwhile gain is ruled out, not undetected. The null is clean because
the mechanism is visible: the affinity head genuinely learns (ρ ≈ 0.58 on
held-out affinity), but measured affinity used *directly* as a stability
predictor ranks at ρ **0.580** — below the 0.610 stability labels alone already
give. Redundant signal, not absent signal. The expansion to 64,226
leakage-filtered IEDB rows is declined on this evidence; the **ESM-2 arm is
still open** and needs stage 3.

## Completed stage 4b.1 output — 182-residue groove MSA cache

| Artifact | Description |
|---|---|
| [reports/stage4b1_msa_cache.md](reports/stage4b1_msa_cache.md) | Route, verified Boltz constraints, parse-cost benchmark |
| `reports/msa_manifest.csv` | Committed record: 75 alleles, `sha256` per HLA sequence and per MSA file |
| `reports/msa_parse_benchmark.json` | `parse_csv` cost vs `--max_msa_seqs` |
| `scripts/make_msas.py` | Regenerates the cache (needs boltz; see the report) |
| `structures/msa/<stem>.csv` | The MSAs — **gitignored**, 141.3 MB, regenerable |
| `tests/test_msa_cache.py` | 14 guards on manifest/cache consistency |

All **75** 182-residue grooves cached, not just the six panel alleles: 137 s of
CPU, **$0**, no GPU booked. Measured findings for that cache — the parse cost at the
default `--max_msa_seqs 8192` is ~$0.25 across 2,000 complexes, so **do not trim
for cost**; `--subsample_msa` defaults to *False* despite its help text;
and the C67S pseudosequence collision does not reach this arm, because the full
domains differ. This cache supplies pilot arm A. Stage 4c still needs complete
275-residue ectodomain MSAs and the verified cropped-MSA control; the groove
cache does not clear those dependencies.

## Quick start

```python
from pepstab.data import load_with_splits

df = load_with_splits()                 # never recompute the splits
train = df[df.split == "train"]
# inputs: peptide, hla_seq, hla_pseudoseq   target: y_log1p
```

Write predictions as `pair_id,y_pred` on the `log1p` scale, then:

```bash
.venv/bin/python scripts/evaluate.py --split val preds/seq_ensemble_pep_pseudo.csv preds/esm.csv
```

First file is the baseline; the rest get paired cluster-bootstrap CIs against
it. `--per-allele` for the full table, `--by-distance` for distance strata.

## Rules

- `data/rasmussen_et_al_dataset.csv` is read-only;
  `(cd data && shasum -a 256 -c SHA256SUMS)` must pass.
- Load splits from disk. Regenerating breaks peptide-cluster grouping and leaks
  training data into test.
- Select on validation. **Test is scored once, at stage 6.**
- Augmented rows are assumptions, not measurements: they carry `thalf_hours = 0`
  in `data/augmentation/`, enter the fit set only, and never touch validation or
  test. Verify any manifest with `pepstab.augment.verify_manifest` before
  training on it.
- No batch GPU job without a passing pilot on 3–5 examples.

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
.venv/bin/python scripts/affinity_multitask.py --protocol ensemble   # ~25 min
.venv/bin/python -m pytest tests/ -q
```

The MSA cache needs boltz, which pulls torch, so it is kept out of `.venv`:

```bash
uv venv /tmp/boltzenv --python 3.12
uv pip install --python /tmp/boltzenv/bin/python boltz
/tmp/boltzenv/bin/python scripts/make_msas.py --all-alleles   # ~2.5 min, $0
```
