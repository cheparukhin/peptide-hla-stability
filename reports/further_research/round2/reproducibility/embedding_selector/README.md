# Locked embedding-selector execution repeats

all five locked per-outer-fold embedding_selector decisions; final refits only, no reselection or outer-best winner.

Two identical-seed/settings executions replay the frozen selected recipe for each fold. The 30-member neural or single deterministic ridge final recipe is refitted in fresh cache namespaces; selected mlp_log branches reuse validated separately recorded reference repeats. Selection is not refitted. This differs from paired-seed variation and is not assay noise.

All original training rows have primary, repeat1 and repeat2 predictions. execution_differences.csv records primary-versus-each-repeat and repeat1-versus-repeat2 maximum/mean absolute prediction differences and candidate-minus-baseline pooled metric differences at frozen 20/50-row allele panels. No partial-cohort results or outer-outcome winner are reported.

Additional fits: 242 (absolute upper bound 300); zero selection fits and zero new sequence reference fits. Hardware, libraries, threads, precision, ancestry, member tables, primary sources and fresh fit caches are hash-bound by the protocol and atomic completion markers. Within-machine only.

| Comparison | Allele panel | Max absolute prediction difference | Mean absolute prediction difference | Median per-allele Spearman delta | MAE delta |
|---|---:|---:|---:|---:|---:|
| primary versus repeat1 | 20 | 0 | 0 | 0 | 0 |
| primary versus repeat2 | 20 | 0 | 0 | 0 | 0 |
| repeat1 versus repeat2 | 20 | 0 | 0 | 0 | 0 |
| primary versus repeat1 | 50 | 0 | 0 | 0 | 0 |
| primary versus repeat2 | 50 | 0 | 0 | 0 | 0 |
| repeat1 versus repeat2 | 50 | 0 | 0 | 0 | 0 |
