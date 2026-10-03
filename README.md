# Peptide–HLA stability

Predicting peptide–HLA dissociation half-life.
[HACKATHON_PLAN.md](HACKATHON_PLAN.md) has scope and stages;
[EVALUATION.md](EVALUATION.md) has the evaluation rules.

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
| `tests/test_contract.py` | 30 guards on the above |

## Quick start

```python
from pepstab.data import load_with_splits

df = load_with_splits()                 # never recompute the splits
train = df[df.split == "train"]
# inputs: peptide, hla_seq, hla_pseudoseq   target: y_log1p
```

Write predictions as `pair_id,y_pred` on the `log1p` scale, then:

```bash
.venv/bin/python scripts/evaluate.py --split val preds/seq_baseline.csv preds/esm.csv
```

First file is the baseline; the rest get paired cluster-bootstrap CIs against
it. `--per-allele` for the full table, `--by-distance` for distance strata.

## Rules

- `data/rasmussen_et_al_dataset.csv` is read-only;
  `(cd data && shasum -a 256 -c SHA256SUMS)` must pass.
- Load splits from disk. Regenerating breaks peptide-cluster grouping and leaks
  training data into test.
- Select on validation. **Test is scored once, at stage 6.**
- No batch GPU job without a passing pilot on 3–5 examples.

## Regenerating stage 1

```bash
.venv/bin/python scripts/make_splits.py   # deterministic; rewrites data/splits.csv
.venv/bin/python scripts/audit_data.py    # rewrites reports/audit_summary.md
.venv/bin/python -m pytest tests/ -q
```
