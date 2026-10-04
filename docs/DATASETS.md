# Datasets

| file | rows | alleles | peptides | role | summary |
| --- | --- | --- | --- | --- | --- |
| [`data/rasmussen_et_al_dataset.csv`](../data/rasmussen_et_al_dataset.csv) | 28,166 | 75 | 5,633 | raw source, read-only | [summary](rasmussen_et_al_dataset_summary/DATASET_SUMMARY.md) |
| [`data/c67s_cleanup/rasmussen_no_C67S.csv`](../data/c67s_cleanup/rasmussen_no_C67S.csv) | 27,031 | 72 | 5,633 | available, not in use this round | [summary](rasmussen_no_C67S_summary/DATASET_SUMMARY.md) |
| [`data/c67s_cleanup/benchmark_C67S.csv`](../data/c67s_cleanup/benchmark_C67S.csv) | 1,135 | 3 | 663 | available, not in use this round | — |

Split assignments for every row of all three files live in
[`data/splits.csv`](../data/splits.csv) and are documented in
[EVALUATION.md](../EVALUATION.md). Join them on `(allele, peptide)`. The earlier
`data/c67s_cleanup/peptide_splits.csv` is superseded — see [SPLITS.md](SPLITS.md)
for why it is kept.

Public binding-affinity measurements for the same 75 alleles — the auxiliary
labels for stage 2c — are in
[`data/data_augmentation_iedb/`](../data/data_augmentation_iedb/) and described
in [AFFINITY_REFERENCE.md](AFFINITY_REFERENCE.md). They are not stability data
and are not part of train/eval; read that document before training on them.

Structural coverage for these alleles — deposited PDB entries, threading
templates, and the 32 complexes with both a measured half-life and an
experimental structure — is described in [STRUCTURES.md](STRUCTURES.md).

The two derived files are regenerated with
`python3 scripts/make_no_c67s_dataset.py`; the summaries with
`python3 scripts/summarize_dataset.py [--input <csv>]`. The split is lossless —
concatenating them reproduces the raw CSV row for row. The raw CSV is never
modified; `(cd data && shasum -a 256 -c SHA256SUMS)` still passes.

## The C67S constructs: available but kept in training

**Decision:** all 75 alleles, including the three C67S constructs, remain in
train/val/test. This matches `data/splits.csv` (which distributes all 1,135 C67S
rows: train 798 / val 122 / test 215) and `EVALUATION.md`'s 67-of-75
eligible-allele panel. The files below exist for reference but are not the
canonical training path.

The original rationale for building them follows — it explains why the split
*exists*, not why it is *used*.

### Why these files were built (historical)

`HLA-B*14:01(C67S)`, `HLA-B*14:02(C67S)` and `HLA-B*39:06(C67S)` — 1,135 rows,
4.03% of the dataset — are pulled out of train/eval and promoted to a dedicated
benchmark.

All three carry **serine at position 67** of the mature heavy chain, verifiable
directly in the `hla_seq` column (`hla_seq[66] == "S"` for each). Position 67 is
an unpaired cysteine site in some natural HLA-B allotypes; it is not part of the
structural disulfide, which is C101–C164 and is intact in all three constructs
(their only cysteines are at 101 and 164). So the substitution is visible to any
sequence-based model rather than hidden behind the allele name.

### 1. They are engineered assay constructs, not natural allotypes

Fitting them teaches the model about an assay stabilisation artefact rather than
about HLA biology.

### 2. No wild-type counterpart is available

`HLA-B*14:01`, `HLA-B*14:02` and `HLA-B*39:06` are all absent from the dataset,
so the constructs cannot serve as a mutation-effect test either. The nearest
natural alleles are several substitutions away — too far to attribute a
half-life difference to position 67:

| construct | nearest natural allele(s) | residues differing (of 182) |
| --- | --- | --- |
| HLA-B*14:01(C67S) | B*39:01, B*39:02, B*39:10 (tied) | 6 |
| HLA-B*14:02(C67S) | B*39:01, B*39:02, B*39:10 (tied) | 7 |
| HLA-B*39:06(C67S) | B*39:01, B*39:02, B*39:10 (tied) | 3 |

### 3. They barely form stable complexes

They would inflate the zero spike without being informative:

| construct | rows | % exactly 0 h | median | p95 half-life | max |
| --- | --- | --- | --- | --- | --- |
| HLA-B*14:01(C67S) | 374 | 89.0% | 0.0 h | 0.60 h | 108.4 h |
| HLA-B*14:02(C67S) | 382 | 74.9% | 0.0 h | 4.39 h | 66.3 h |
| HLA-B*39:06(C67S) | 379 | 92.1% | 0.0 h | 0.50 h | 8.5 h |

Note on provenance: the challenge brief reports the same zero fractions and
95th-percentile half-lives but pairs the B\*14:01 and B\*14:02 values the
opposite way round. The table above is recomputed from the raw CSV.

## Effect of the exclusion

- Zero-inflation drops from **20.16% to 17.43%** of rows.
- Rows above the 2-hour stability threshold rise from **39.88% to 41.37%**.
- `hla_pseudoseq` becomes a **unique allele key** (72 alleles, 72 pseudosequences).
  The only collision in the raw data was `HLA-B*14:01(C67S)` /
  `HLA-B*14:02(C67S)` sharing one pseudosequence.
- **No peptides are lost** — all 5,633 remain, because each of the 663 peptides
  measured against a C67S construct is also measured against at least one
  natural allele.
- **No Cys67 signal is lost.** Six natural alleles retain cysteine at position 67
  (B\*15:10, B\*27:02, B\*27:03, B\*27:05, B\*27:20, B\*39:01), together 2,504
  rows (8.9%), and all stay in the training set.
