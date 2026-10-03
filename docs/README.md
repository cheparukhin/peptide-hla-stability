# docs

Reference material for the peptide–HLA stability project. Start with
[HACKATHON_PLAN.md](../HACKATHON_PLAN.md) — it is the source of truth for scope
and stage order. These documents describe the data behind it.

## Data

| document | covers |
| --- | --- |
| [DATASETS.md](DATASETS.md) | the three CSVs in `data/`, the C67S exclusion and its effects, which file to train on |
| [rasmussen_et_al_dataset_summary/](rasmussen_et_al_dataset_summary/DATASET_SUMMARY.md) | full stats for the raw dataset — counts, coverage, target distribution, per-allele breakdown |
| [rasmussen_no_C67S_summary/](rasmussen_no_C67S_summary/DATASET_SUMMARY.md) | the same stats for the train/eval dataset |

## Splits

| document | covers |
| --- | --- |
| [SPLITS.md](SPLITS.md) | the frozen peptide-grouped folds — how components are built, why the threshold is 0.70, what the leakage check proves, and how to join them |

## Augmentation

| document | covers |
| --- | --- |
| [AFFINITY_REFERENCE.md](AFFINITY_REFERENCE.md) | the public binding-affinity table behind stage 3b — provenance, allele coverage, why the C67S constructs are flagged rather than matched, and the leakage filters to apply before training on it |

## Structures

| document | covers |
| --- | --- |
| [STRUCTURES.md](STRUCTURES.md) | what the three structural tables contain, the 32-pair PDB overlap and how to use it, the memorisation question, and whether to add an affinity column |
| [pdb_rasmussen_overlap.md](pdb_rasmussen_overlap.md) | the 32 complexes with both a measured half-life and a deposited structure, plus 12 near misses — browsable, with RCSB links |
| [allele_pdb_templates.md](allele_pdb_templates.md) | one threading template per allele, tiered by how well the groove matches |

## Regenerating

| artifact | command |
| --- | --- |
| `data/c67s_cleanup/rasmussen_no_C67S.csv`, `data/c67s_cleanup/benchmark_C67S.csv` | `python3 scripts/make_no_c67s_dataset.py` |
| either dataset summary | `python3 scripts/summarize_dataset.py [--input <csv>]` |
| `data/c67s_cleanup/peptide_splits.csv` and its manifest | `python3 scripts/split_peptides.py` |
| `data/data_augmentation_iedb/affinity_reference_75alleles.csv` and its manifest | `python3 scripts/fetch_affinity_reference.py` |

The structural tables were assembled from RCSB and are checked in as-is; no
script in this repo regenerates them.
