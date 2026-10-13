# Reproduction and verification

The [research report](../reports/REPORT.md) contains the conclusions,
[EVALUATION.md](../EVALUATION.md) the statistical methods and
[reports/README.md](../reports/README.md) the numerical evidence index.
Source data, split identities and frozen experiment artifacts are distinct
from the current scientific narrative.

## Environment and data

Use Python 3.12.12 and the pinned package instructions in
[ROUND2_INFERENCE.md](ROUND2_INFERENCE.md#environment-setup) for the portable
cached-head interface and its lightweight checks. Feature extraction, trees,
structural scoring and full assessment have separate recorded environments;
the inference environment alone does not reproduce them.

Load canonical identities and splits through the supplied interface:

```python
from pepstab.data import load_with_splits

frame = load_with_splits()
training = frame.loc[frame.split.eq("train")].copy()
```

The raw CSV is read-only. From the repository's `data/` directory, verify it
with `shasum -a 256 -c SHA256SUMS`. Preserve `data/splits.csv` and the supplied
cluster/fold assignments. Derived files join by exact allele–peptide identity;
do not use subset row positions as source `pair_id` values. Do not regenerate
splits for an existing assessment.

## Saved evidence

CSV predictions, comparison tables, bootstrap draws and JSON manifests provide
the reproducible numerical record. Check row coverage, identity alignment,
metric eligibility and paired differences against those artifacts. A saved
prediction audit does not refit a model, repeat its selection procedure or
establish independent biological replication. Some descriptor studies do not
save fitted weights; their scoring and ancestry are auditable, but an exact
weight-level replay is unavailable.

Original hash-bound protocol text is preserved in
[`reports/protocol_sources.zip`](../reports/protocol_sources.zip) at its original
repository-relative member paths. For a complete reconstruction with the
original experiment scripts and prose inputs, use source commit `f021bbac` in a
separate checkout. If needed, materialise archive members there. Do not unpack
obsolete narrative files over the current documentation: some frozen audits
verify their original bytes, whereas the current documents reconcile the
scientific interpretation. External feature caches and coordinates must also
be available for workflows that depend on them.

## Lightweight checks and inference

Inspect a packaged head without training or accessing a feature cache:

```sh
.venv/bin/python scripts/round2_inference.py inspect \
  --model models/finalization/R4_ridge_100
```

The supported synthetic interface checks are:

```sh
.venv/bin/python -m pytest -q tests/test_round2_inference.py
```

The [inference guide](ROUND2_INFERENCE.md) defines the verified feature-bundle
contract and prediction command. Predictions from the exported fixed recipes
must not inherit the main nested selector's reported accuracy. Cache timing
excludes feature extraction and is not end-to-end latency for a novel complex.

## New computation

Keep local work to editing, orchestration and lightweight checks. Run substantial
fitting, feature extraction, resampling and structural calculations on Modal
only within an explicitly authorised credit budget, with bounded resources and
spending guards. Independent jobs may run concurrently within that budget;
reuse valid artifacts rather than duplicate experiments. Do not treat a stored
execution authorisation or an old budget as permission for a new paid run.

New research must specify its population, question, comparator, partitions and
claim before inspecting new assessment outcomes. The reused training cohort
provides development evidence. Independent confirmation requires a separate
measurement population and the assay design described in the
[independent assay brief](../reports/finalization/INDEPENDENT_ASSAY_BRIEF.md).
