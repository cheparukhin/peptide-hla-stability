# Peptide-HLA stability

Predicting peptide-HLA dissociation half-life. See
[HACKATHON_PLAN.md](HACKATHON_PLAN.md) for scope and stage order, and
[EVALUATION.md](EVALUATION.md) for the frozen evaluation contract.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy pandas scipy scikit-learn pyarrow pytest
```

## Stage 1 output (done — read this before writing a model)

| Artifact | What it is |
|---|---|
| [reports/audit_summary.md](reports/audit_summary.md) | data audit: alphabets, constructs, the censored zeros, split justification |
| `data/splits.csv` | **frozen** splits, committed. `pair_id, allele, peptide, cluster_id, split, dist_to_train` |
| [EVALUATION.md](EVALUATION.md) | predeclared metrics, eligible alleles, the 0.05 improvement bar |
| `pepstab/` | shared library: loading, splits, scoring |
| `tests/test_contract.py` | 30 guards on the above |

## Using it

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

The first file is the baseline; later ones get paired cluster-bootstrap CIs
against it. Add `--per-allele` for the per-allele table, `--by-distance` for
distance-stratified scores.

## Rules that are not negotiable

- `data/rasmussen_et_al_dataset.csv` is read-only;
  `(cd data && shasum -a 256 -c SHA256SUMS)` must pass.
- Load splits from disk. Regenerating them drops the peptide-cluster grouping
  and leaks training peptides into the test set.
- Select on validation. **The test set is scored once, at stage 6.**
- No batch GPU job without a passing end-to-end pilot on 3-5 examples.

## Regenerating stage 1

```bash
.venv/bin/python scripts/make_splits.py   # deterministic; rewrites data/splits.csv
.venv/bin/python scripts/audit_data.py    # rewrites reports/audit_summary.md
.venv/bin/python -m pytest tests/ -q
```
