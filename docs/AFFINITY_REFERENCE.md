# Auxiliary binding-affinity data

The [affinity reference table](../data/data_augmentation_iedb/affinity_reference_75alleles.csv) contains **110,591 deduplicated allele–nine-mer records** derived from public binding-affinity measurements. Equilibrium binding affinity and kinetic half-life describe different properties, so these records serve as auxiliary data rather than additional measured dissociation half-lives. The [manifest](../data/data_augmentation_iedb/affinity_reference_75alleles.manifest.json) records source versions, filtering, and counts.

## Provenance and fields

Despite the directory name, this table comes from MHCflurry release assets rather than a live IEDB query. The curated affinity snapshot `data_curated.20231023` serves as the primary source, while the Kim et al. BD2013 dataset `bdata.20130222` supplies a cross-check. Because these corpora can share measurements, agreement between them is not an independent biological replication.

Filtering retained human HLA nine-mers associated with the 75 stability molecules and reduced duplicate allele–peptide records to the lowest reported `affinity_nM`. This selection is conservative for flagging weak binding, not an unbiased summary of assay variability, so units, inequality flags, and source fields must accompany any interpretation.

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

The table covers 70 of the 75 stability names, with 7,281 construct-compatible pairs carrying both outcomes. By contrast, the 245 records associated by name with C67S constructs are explicitly flagged: wild-type measurements describe a different molecule and must not be treated as matching construct labels.

## Leakage and endpoint constraints

`padding_eligible` is insufficient for training admission. Because pairwise absence from the stability table does not imply that a peptide is absent on every other allele, auxiliary fitting must exclude exact peptides and Hamming-distance ≤3 neighbours of held-out peptides across **all alleles**, including inner selection and stopping partitions. The relevant frozen assignments come from the [split table](../data/splits.csv), not the older C67S-cleanup split. Candidate manifests and `pepstab.augment.verify_manifest` implement these augmentation-specific checks; a new assessment requires its own source and held-partition exclusions.

Similarly, a weak-affinity observation does not establish zero-hour stability. Any zero-hour label assigned from affinity is a modelling assumption and belongs only in the fitting partition. It must remain distinguishable from a measured outcome, never overwrite one, and never enter evaluation as experimental truth.

Flag interpretation depends on recorded bounds and cross-check coverage. `is_weak_binder` is based on the stored value: no `<` bound occurs at or above the threshold in this snapshot, whereas `>` bounds below it can conceal additional weak binders. Likewise, `bd2013_agrees_weak=False` can mean either disagreement or a missing cross-check, so inspect value availability before counting disagreement. Assay heterogeneity, selection by affinity, and measurement overlap limit the interpretation of affinity–stability associations. The scientific conclusions belong in the [research report](../reports/REPORT.md).
