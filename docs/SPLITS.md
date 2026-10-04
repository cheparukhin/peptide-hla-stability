# Peptide-grouped splits (superseded)

> **Superseded by [`data/splits.csv`](../data/splits.csv)**, frozen in stage 1
> by `scripts/make_splits.py` with single-linkage clustering at Hamming ≤ 3 and
> documented in [EVALUATION.md](../EVALUATION.md) and
> [reports/audit_summary.md](../reports/audit_summary.md). Load that file. This
> document describes the earlier BLOSUM62 grouping at 0.70 and is kept for the
> method comparison and the threshold argument below; the two assign peptides
> differently, so do not mix them.


[`data/c67s_cleanup/peptide_splits.csv`](../data/c67s_cleanup/peptide_splits.csv) — 28,166
rows, one per measurement, carrying the frozen 70/10/20 partition.
[`peptide_splits_manifest.json`](../data/c67s_cleanup/peptide_splits_manifest.json)
records the parameters and the verification numbers. Regenerate with
`python3 scripts/split_peptides.py`; the procedure has no randomness, so it
reproduces byte for byte.

| column | meaning |
| --- | --- |
| `row_id` | positional index into `data/rasmussen_et_al_dataset.csv` |
| `allele`, `peptide` | the join key |
| `peptide_group` | similarity component; indivisible across folds |
| `fold` | 0–9 |
| `split` | `train` (folds 0–6), `val` (fold 7), `test` (folds 8–9) |

**Filter on `split`, and join on `(allele, peptide)`.** The pair is unique in all
three datasets. Do not join on `row_id` — it is positional into the raw CSV and
silently misaligns against `rasmussen_no_C67S.csv`, which has 1,135 fewer rows.

The split is built on the **raw** CSV so one table covers the raw set, the
no-C67S modelling set, and the C67S benchmark. The proportions survive the C67S
removal intact:

| | rows (raw) | | rows (no-C67S) | |
| --- | ---: | ---: | ---: | ---: |
| train | 19,718 | 70.0% | 18,929 | 70.0% |
| val | 2,816 | 10.0% | 2,700 | 10.0% |
| test | 5,632 | 20.0% | 5,402 | 20.0% |

Ten folds rather than a direct three-way cut, so folds 0–6 also support
cross-validation during model selection without touching fold 7 or 8–9.

## Method

1. Normalised BLOSUM62 similarity between all 5,633 unique peptides, scaled so
   identical sequences score 1.0.
2. An edge wherever similarity ≥ 0.70; **connected components** of that graph,
   so if A resembles B and B resembles C all three travel together.
3. Components weighted by rows owned, not peptides owned — a peptide screened on
   36 alleles brings 36 measurements with it. Largest component first into the
   currently emptiest fold (LPT), which balances without a seed to defend.
4. Verification, below.

The 5,633 peptides collapse into **5,374 components**, the largest holding 5
peptides and 37 rows. Folds come out at 2,816–2,817 rows each — a 0.04% spread,
as even as integer component sizes allow.

## Why 0.70

Stage 1 of [HACKATHON_PLAN.md](../HACKATHON_PLAN.md) requires that identical
peptides **and one-residue neighbours** stay together. The weakest
one-substitution pair in this dataset scores **0.7767**, so any threshold below
that captures all 146 of them. 0.70 does, with margin:

| substitution count | pairs | weakest similarity | pairs below 0.70 |
| --- | ---: | ---: | ---: |
| 1 | 146 | 0.7767 | 0 |
| 2 | 114 | 0.5943 | 10 |
| 3 | 324 | 0.3907 | 279 |

Raising the threshold toward 1.0 approaches exact-identity grouping; at 0.78 it
would start breaking one-residue clusters apart.

## What the verification does and does not prove

The manifest reports a maximum test-to-train similarity of **0.6988**, just
under the threshold. Read that as a **consistency check, not a discovery**:
components are the connected components of the ≥ 0.70 graph, so a cross-boundary
pair at or above 0.70 would mean the grouping code is broken. It would catch a
bug; it cannot independently justify the threshold.

The informative figure is measured on a scale that did not define the groups —
substitution count:

| | held-out peptides | within 1 substitution | within 2 substitutions |
| --- | ---: | ---: | ---: |
| test vs train+val | 1,134 | **0** | 2 |
| val vs train | 557 | **0** | 3 |

No one-residue neighbour crosses either boundary. Five peptides in total sit two
substitutions from something the model saw — the two-change pairs whose
substitutions BLOSUM62 scores as chemically dissimilar enough to fall under
0.70. That is the honest residual, and it is stricter than the plan asked for.
The manifest also carries the same check per fold, for cross-validation use.

Assertions fail loudly if a component straddles folds, if `(allele, peptide)` is
not unique, if peptides are not all the same length, or if the input exceeds
20,000 peptides — where the dense similarity matrix stops being the right
approach.

## Per-allele test coverage

On the no-C67S modelling set, across its 72 alleles: median **74** test rows per
allele, **64 of 72** alleles clear the plan's 50-row floor, 9 clear 100. Eight
alleles fall short, the thinnest being `HLA-B*13:02` and `HLA-A*69:01` at 2 test
rows each. Per-allele Spearman is undefined or meaningless for those — stage 6
should fix one eligible-allele list across all models and report the excluded
ones explicitly rather than letting each model choose its own denominator.
