# Affinity reference

[`data/data_augmentation_iedb/affinity_reference_75alleles.csv`](../data/data_augmentation_iedb/affinity_reference_75alleles.csv)
— 110,591 rows, one per `(allele, 9-mer)` binding-affinity measurement on the
75 stability alleles.
[`affinity_reference_75alleles.manifest.json`](../data/data_augmentation_iedb/affinity_reference_75alleles.manifest.json)
records the sources, filters and counts. Regenerate with
`python3 scripts/fetch_affinity_reference.py`; the download is cached in
`external/` (93 MB, gitignored) and the output reproduces byte for byte.

This table supports the auxiliary affinity experiment in
[HACKATHON_PLAN.md](../HACKATHON_PLAN.md) stage 2c, and the measured-affinity
augmentation arm in stage 2b — which has now run; see
[reports/stage2b_augmentation.md](../reports/stage2b_augmentation.md) for the
result and `pepstab.augment` for the filtering described below, implemented. It carries **affinity**, how
strongly a peptide binds — not **stability**, how long it stays. The two are
related but distinct, which is the whole reason the experiment is worth running.

## Provenance

Despite the folder name, nothing here comes from a live IEDB query. Both sources
are MHCflurry GitHub release assets, so the fetch needs no IEDB or DTU host:

| source | role | snapshot |
| --- | --- | --- |
| MHCflurry curated affinity (`data_curated.20231023`) | primary measurements | 2023-10-23 |
| Kim et al. 2014 benchmark BD2013 (`bdata.20130222`) | independent cross-check | 2013-02-22 |

Filters: human HLA, 9-mers only, restricted to the 75 stability alleles, then
deduplicated to the **lowest** `affinity_nM` per `(allele, peptide)`. Keeping the
strongest measurement is the conservative choice for a weak-binder flag — it
under-calls weak rather than over-calls it.

## Columns

| column | meaning |
| --- | --- |
| `dataset_allele` | allele name as it appears in the stability dataset, `(C67S)` suffix restored |
| `allele`, `peptide` | the join key, normalised to `HLA-A*02:01` form |
| `affinity_nM`, `inequality`, `assay_source` | the curated measurement and its censoring flag |
| `bd2013_affinity_nM`, `bd2013_inequality` | the BD2013 measurement for the same pair, if any |
| `bd2013_agrees_weak` | BD2013 also places the pair at or above 20,000 nM |
| `is_weak_binder` | `affinity_nM >= 20,000` |
| `engineered_construct_mismatch` | the affinity is wild-type but the stability allele is a C67S construct — never join these |
| `in_stability_dataset` | this exact `(allele, peptide)` pair has a measured half-life |
| `stability_thalf_hours` | that half-life, else empty |
| `padding_eligible` | `is_weak_binder & ~in_stability_dataset & ~engineered_construct_mismatch`; not a safe training filter on its own — see "Before training on it" |

## What it contains

| | count |
| --- | ---: |
| rows | 110,591 |
| stability alleles with any affinity row | 70 of 75 |
| weak binders (≥ 20,000 nM) | 46,479 |
| pairs with both affinity and half-life | 7,281 (58 alleles) |
| pairs also present in BD2013 | 97,965 |
| peptides not in the stability set at all | 20,879 |
| rows flagged `engineered_construct_mismatch` | 245 |

## Allele coverage

The 75 stability alleles fall into four groups, not two:

| group | alleles |
| --- | ---: |
| affinity data including weak binders | 54 |
| affinity data but nothing at or above 20,000 nM | 16 |
| no affinity data at all | 5 |
| C67S constructs — matchable by name, not by biology | 3 |

The five with nothing are `HLA-A*24:07`, `HLA-A*24:19`, `HLA-A*43:01`,
`HLA-B*39:02` and `HLA-B*41:01`. All are natural allotypes carrying 352–378 rows
each in the stability set, so they are well measured for stability and simply
absent from the public affinity corpora. Nothing about them needs fixing; they
just cannot take an auxiliary affinity label.

> The manifest's `alleles_without_weak_binders` field lists 21 names — the 16
> and the 5 combined. Separate them on whether the allele appears in the
> `allele` column at all.

## Why the C67S constructs are flagged, not matched

The three constructs match the affinity corpora under their wild-type names:
`B*14:02` contributes 172 rows (61 of them weak across all three), `B*14:01` 40,
`B*39:06` 33. Matching them anyway would be a category error. C67S substitutes
position 67, which is **one of the 34 peptide-contact positions** in the
pseudosequence — so wild-type affinity describes a different groove from the one
whose half-life was measured.

Those 245 rows keep their affinity values and carry
`engineered_construct_mismatch = True`; they are excluded from the stability join
and from `padding_eligible`, with the reasoning recorded in the manifest rather
than applied as a silent filter. The correction is small — 4 of the former 7,285
overlap pairs and 61 padding rows — but wrong in kind, and it would have
contaminated the one comparison where the constructs matter most, the
pseudosequence-ceiling test. The pooled affinity–stability Spearman is unchanged
at −0.491.

## The signal it carries

Affinity and half-life agree in direction and disagree in detail, which is what
makes the auxiliary head worth testing rather than redundant:

- Pooled Spearman between `affinity_nM` and `stability_thalf_hours` on the 7,281
  dual-measured pairs: **−0.491**. Per allele, across the 36 alleles with ≥ 30
  dual-measured pairs: median **−0.578**, range −0.839 to 0.002.
- The weak-binder flag predicts the stability floor well. Of the 261
  dual-measured pairs flagged weak, **78.9%** have a half-life of exactly zero
  and only 4.6% exceed 2 hours; among the 7,020 non-weak pairs, 7.2% are zero
  and the median is 2.1 hours.

So weak-affinity rows are a defensible source of near-zero stability labels —
the padding idea behind `padding_eligible` is empirically supported.

## Before training on it

### 1. `padding_eligible` leaks held-out peptides

`in_stability_dataset` matches on the `(allele, peptide)` **pair**, but the
frozen splits group peptides across *all* alleles. A peptide held out in val or
test therefore reappears as "not in the stability dataset" whenever it was
measured on a different allele. Of the 46,157 `padding_eligible` rows:

| | rows | peptides |
| --- | ---: | ---: |
| peptide genuinely absent from the stability set | 21,728 | 4,728 |
| peptide present under a different allele | 24,429 | 2,189 |
| …of those, peptide assigned to **val or test** | **7,502** | **654** |

Those 7,502 rows are distance-zero leakage, not near-neighbour leakage. Filter
`padding_eligible` against [`data/splits.csv`](../data/splits.csv) on `peptide`
before using it, not on `(allele, peptide)`.

Stage 2b measured this on the weak-binder pool it actually used. Against the
inner-dev, validation and test peptides combined, the distance filter removes
9,731 of 46,157 candidate rows (917 peptides) — and **9,123 of those 9,731 sit
at distance 0**, exact held-out peptides offered under another allele. They are
94% of the exclusions, and a pair-wise check would have let every one of them
through.

### 2. Near neighbours of held-out peptides

For the 20,879 peptides absent from the stability set, minimum Hamming distance
to any val/test peptide:

| distance | peptides | share | padding-eligible subset |
| ---: | ---: | ---: | ---: |
| 1 | 177 | 0.85% | 32 |
| 2 | 54 | 0.26% | 11 |
| 3 | 327 | 1.57% | 53 |
| 4 | 4,229 | 20.25% | 827 |
| 5 | 13,344 | 63.91% | 3,050 |
| ≥ 6 | 2,748 | 13.16% | 755 |

Stages 2b and 2c require excluding any candidate within Hamming ≤ 3 of an
inner-dev, validation, or test peptide, across all alleles. For the validation/test
part of this rule, the table above excludes **558** peptides (2.7%), of which 96
are padding-eligible. Apply the same distance check against the inner stopping
fold as well; its additional exclusions are not included in those counts.

Augmented peptides are outside the frozen clustering, so `dist_to_train` in
`splits.csv` does not describe them — these distances are measured here against
the val/test peptides of that file and must be recomputed if the split is
refrozen.

### 3. Censoring and missing values

- `is_weak_binder` compares `affinity_nM` to 20,000 nM and ignores `inequality`.
  No row carries `<` at or above the threshold, so nothing is wrongly flagged
  weak. But 17,967 rows are `>` *below* the threshold — their true affinity may
  well exceed it — so 46,479 is a conservative lower bound on weak binders.
- `bd2013_agrees_weak` is `False` both when BD2013 disagrees and when BD2013 has
  no measurement for the pair. Among padding-eligible rows: 45,453 corroborated,
  47 contradicted, 657 simply absent. Test `bd2013_affinity_nM.notna()` first.
