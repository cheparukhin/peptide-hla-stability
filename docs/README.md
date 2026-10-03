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
| [SPLITS.md](SPLITS.md) | **superseded** by `data/splits.csv` and [EVALUATION.md](../EVALUATION.md) — kept for the BLOSUM62-vs-Hamming method comparison and the threshold argument |

## Augmentation

| document | covers |
| --- | --- |
| [AFFINITY_REFERENCE.md](AFFINITY_REFERENCE.md) | the public binding-affinity table behind stages 2b and 2c — provenance, allele coverage, why the C67S constructs are flagged rather than matched, and the leakage filters to apply before training on it |
| [../reports/stage2c_affinity.md](../reports/stage2c_affinity.md) | **the stage 2c result** — auxiliary affinity training does not help, bounded below the 0.05 bar in 20 paired comparisons; why the signal is redundant rather than absent; the declined expansion and its leakage audit |

## Structures

| document | covers |
| --- | --- |
| [STRUCTURES.md](STRUCTURES.md) | what the three structural tables contain, the 32-pair PDB overlap and how to use it, the memorisation question, and whether to add an affinity column |
| [pdb_rasmussen_overlap.md](pdb_rasmussen_overlap.md) | the 32 complexes with both a measured half-life and a deposited structure, plus 12 near misses — browsable, with RCSB links |
| [allele_pdb_templates.md](allele_pdb_templates.md) | one threading template per allele, tiered by how well the groove matches |

## Folding (stage 4)

| document | covers |
| --- | --- |
| [BOLTZ_PIPELINE.md](BOLTZ_PIPELINE.md) | why MSAs are per-sequence and cost almost nothing, the pilot and benchmark panels, the Modal cost traps (`H100!`, billed startup, oversized host requests), the verified rate table, and the GPU decision rule |
| [../reports/stage4_benchmark.md](../reports/stage4_benchmark.md) | **the measured result** — throughput and cost per complex across 5 GPUs, why H100 is a trap at this complex size, the A100 anomaly, pose validation, spend, and limitations |

## Regenerating

| artifact | command |
| --- | --- |
| `data/c67s_cleanup/rasmussen_no_C67S.csv`, `data/c67s_cleanup/benchmark_C67S.csv` | `python3 scripts/make_no_c67s_dataset.py` |
| either dataset summary | `python3 scripts/summarize_dataset.py [--input <csv>]` |
| `data/c67s_cleanup/peptide_splits.csv` and its manifest | `python3 scripts/split_peptides.py` |
| `data/data_augmentation_iedb/affinity_reference_75alleles.csv` and its manifest | `python3 scripts/fetch_affinity_reference.py` |
| `reports/stage2c_*.csv` | `.venv/bin/python scripts/affinity_multitask.py` (add `--protocol ensemble`, `--drop-censored`) |
| `reports/boltz_pilot.csv`, `reports/boltz_bench_panel.csv`, `reports/boltz_msa_targets.csv` | `python3 scripts/boltz_panel.py` |
| `reports/gpu_decision.csv` | `python3 scripts/gpu_decision.py` (needs `reports/boltz_bench_results.csv`) |

The structural tables were assembled from RCSB and are checked in as-is; no
script in this repo regenerates them.
