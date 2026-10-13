# Auxiliary binding-affinity data

[`affinity_reference_75alleles.csv`](../data/data_augmentation_iedb/affinity_reference_75alleles.csv)
contains **110,591 deduplicated allele–nine-mer records** derived from public
binding-affinity measurements. These are auxiliary data, not additional
measured dissociation half-lives. Source versions, filtering and counts are
recorded in the [manifest](../data/data_augmentation_iedb/affinity_reference_75alleles.manifest.json).

## Provenance and fields

Despite the directory name, the table was obtained from MHCflurry release
assets rather than a live IEDB query. The primary source is the curated affinity
snapshot `data_curated.20231023`; the Kim et al. BD2013 dataset
`bdata.20130222` supplies a cross-check. These corpora can share measurements;
agreement between them is not an independent biological replication.

Filtering retained human HLA nine-mers associated with the 75 stability
molecules. Duplicate allele–peptide records were reduced to the lowest reported
`affinity_nM`. This selection is conservative for flagging weak binding, but
is not an unbiased summary of assay variability. Units, inequality flags and
source fields must accompany any interpretation.

| Field | Meaning |
|---|---|
| `dataset_allele` | Name in the stability dataset, including construct suffix |
| `allele`, `peptide` | Normalised affinity identity |
| `affinity_nM`, `inequality`, `assay_source` | Retained affinity value, bound and source |
| `bd2013_affinity_nM`, `bd2013_inequality` | Cross-check value and bound, when present |
| `is_weak_binder` | Recorded affinity ≥20,000 nM |
| `engineered_construct_mismatch` | Wild-type affinity cannot be assigned to a C67S stability construct |
| `in_stability_dataset`, `stability_thalf_hours` | Exact compatible pair overlap and its recorded stability |
| `padding_eligible` | Candidate weak-affinity row; further split filtering is required |

The table contains affinity records for 70 of the 75 stability names, with
7,281 construct-compatible pairs carrying both outcomes. The 245 records
associated by name with C67S constructs are explicitly flagged: wild-type
measurements describe a different molecule and must not be treated as matching
construct labels.

## Leakage and endpoint constraints

`padding_eligible` is insufficient for training admission. Pairwise absence
from the stability table does not imply that a peptide is absent on every
other allele. Auxiliary fitting must exclude exact peptides and Hamming-distance
≤3 neighbours of held-out peptides across **all alleles**, including inner
selection and stopping partitions. The relevant frozen assignments come from
[`data/splits.csv`](../data/splits.csv), not the older C67S-cleanup split.
Candidate manifests and `pepstab.augment.verify_manifest` implement the
augmentation-specific checks; a new assessment requires its own source and
held-partition exclusions.

A weak-affinity observation does not establish zero-hour stability. Any
zero-hour label assigned from affinity is a modelling assumption and belongs
only in the fitting partition. It must remain distinguishable from a measured
outcome, never overwrite one and never enter evaluation as experimental truth.

`is_weak_binder` is based on the stored value. No `<` bound occurs at or above
the threshold in this snapshot; `>` bounds below it can conceal additional
weak binders. `bd2013_agrees_weak=False` can mean either disagreement or a
missing cross-check; inspect value availability before counting disagreement.
Assay heterogeneity, selection by affinity and measurement overlap limit the
interpretation of affinity–stability associations. The scientific conclusions
belong in the [research report](../reports/REPORT.md).
