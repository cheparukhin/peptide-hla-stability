# Reproduction and verification

The [research report](../reports/REPORT.md) presents the study conclusions, the
[evaluation guide](../EVALUATION.md) defines the statistical methods, and the
[evidence index](../reports/README.md) catalogues the numerical records.
Source data, split identities, and frozen experiment artifacts remain distinct
from the current scientific narrative.

## Environment and data

Use Python 3.12.12 and the pinned package instructions in the
[inference setup guide](ROUND2_INFERENCE.md#environment-setup) for the portable
cached-head interface and its lightweight checks. Because feature extraction,
trees, structural scoring, and full assessment rely on separate recorded
environments, the inference environment alone cannot reproduce them.

Load canonical identities and splits through the supplied interface:

```python
from pepstab.data import load_with_splits

frame = load_with_splits()
training = frame.loc[frame.split.eq("train")].copy()
```

The raw CSV is read-only; verify it from the repository's `data/` directory
with `shasum -a 256 -c SHA256SUMS`. Preserve `data/splits.csv` and the supplied
cluster/fold assignments, and do not regenerate splits for an existing
assessment. Derived files must join by exact allele–peptide identity rather
than using subset row positions as source `pair_id` values.

## Saved evidence

CSV predictions, comparison tables, bootstrap draws, and JSON manifests provide
the reproducible numerical record. Check row coverage, identity alignment,
metric eligibility, and paired differences against these artifacts. Auditing
saved predictions does not refit a model, repeat its selection procedure, or
establish independent biological replication. Some descriptor studies omit
fitted weights; their scoring and ancestry remain auditable, though an exact
weight-level replay is unavailable.

The [protocol source archive](../reports/protocol_sources.zip) preserves
original hash-bound protocol text at its repository-relative member paths. For
a complete reconstruction with the original experiment scripts and prose
inputs, check out source commit `f021bbac` in a separate working tree and
materialise archive members there if needed. Do not unpack obsolete narrative
files over the current documentation: frozen audits verify the original bytes,
whereas the current documents reconcile the scientific interpretation.
Workflows that depend on external feature caches and coordinates also require
those inputs to be available.

## Lightweight checks and inference

Inspect a packaged head locally without training or accessing a feature cache:

```sh
.venv/bin/python scripts/round2_inference.py inspect \
  --model models/finalization/R4_ridge_100
```

Run the supported synthetic interface checks with:

```sh
.venv/bin/python -m pytest -q tests/test_round2_inference.py
```

The [inference guide](ROUND2_INFERENCE.md) defines the verified feature-bundle
contract and prediction command. Predictions from the exported fixed recipes
must not inherit the main nested selector's reported accuracy. In addition,
cache timing excludes feature extraction and is not end-to-end latency for a
novel complex.

## New computation

Keep local work to editing, orchestration, and lightweight checks. Run
substantial fitting, feature extraction, resampling, and structural
calculations on Modal only within an explicitly authorised credit budget, with
bounded resources and spending guards. Independent jobs may run concurrently
within that budget, but reuse valid artifacts rather than duplicating
experiments. Never treat a stored execution authorisation or an old budget as
permission for a new paid run.

New research must specify its population, question, comparator, partitions, and
claim before inspecting new assessment outcomes. Whereas the reused training
cohort provides development evidence, independent confirmation requires a
separate measurement population; the
[independent assay brief](../reports/finalization/INDEPENDENT_ASSAY_BRIEF.md)
outlines one possible draft validation design rather than a sole mandated or
externally given requirement.