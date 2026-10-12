# Locked embedding-selector execution repeats — implementation draft

Scope fixed before repeat fitting: replay **every saved per-outer-fold embedding_selector decision**, twice at the original seeds and settings. There is no outer-outcome winner and no reselection. This measures execution of the selector's locked final decisions rather than reproducibility of fitting the whole selection procedure. Development evidence only; no transfer, cross-platform, or assay-noise claim.

The runner is local CPU only, with two numerical-library threads and zero cloud spending. It does not launch work by default. Both main studies must have all five final completion markers before any prediction table is read; predictions, scientific population/folds, saved inner-only selections, main ancestry, component completion/seed hashes and exact feature identities are checked before the repeat protocol is frozen. Sequence implementation authorization follows its reviewed launch hashes and recorded protocol/hardware amendment chain while preserving the frozen scientific specification. Embedding implementation hashes must match the current frozen protocol.

For a selected embedding arm, reuse its exact selected candidate dictionary and exact loaded feature matrix, then call FeatureHead.predict(final=True) with the original outer-refit context, five stopping folds, six seeds 0–5 and max_epochs 200. The ridge recipe is a single lsqr fit with its frozen alpha, tolerance and iteration cap. Every embedding repeat attempt receives a new empty cache namespace under the ignored runs/round2_embedding_repeats/<repeat_protocol_hash>/fold_<n>/repeat_<n>/<unique_attempt>/ directory. Interrupted-attempt caches and main caches cannot supply the repeated fits. Completed, hash-validated repeat outputs may be resumed without fitting again.

If a fold selected mlp_log, reuse only that fold's separately recorded sequence reference execution repeats. Validate their marker/protocol/primary binding, prediction/member/ancestry hashes, exclusion audit, exact selected candidate, and complete five stopping folds × two encodings × three seeds 0–2 per repeat. Paired-seed files never substitute for execution repeats.

Additional model fits are exactly twice the sum across locked folds of 30 for an embedding MLP, 1 for an embedding ridge and 0 for a reused sequence reference. The absolute bound is **300 new fits**; no selection fits or new sequence reference fits are scheduled. Reused references already received their separately declared sequence-repeat fitting budget.

Each atomic repeat completion marker binds its predictions, member table, complete ancestry, hardware/library/thread/precision manifest and fresh fit-cache content hashes to the frozen repeat protocol and its main-source hashes. Baseline reuse preserves the original recorded reference runtime manifest separately. There is no partial-cohort repeat comparison.

After all ten fold-repeat markers validate, the runner writes complete primary/repeat1/repeat2 selector predictions, pooled metrics and execution differences. The three comparisons are primary versus repeat1, primary versus repeat2, and repeat1 versus repeat2. Each includes maximum/mean absolute prediction difference on the exact full cohort and candidate-minus-baseline pooled metric differences on the frozen 20/50-row allele panels. Existing paired-seed diagnostics remain a separate measurement.

Run these commands from /Users/cheparukhin/ai_science_hack only after the root confirms the main studies and baseline repeat gates are complete. The first command validates and freezes without fitting; the second explicitly executes the bounded local refits.

```sh
.venv/bin/python scripts/round2_embedding_repeats.py --production-manifest reports/further_research/round2/extractor/full_training/feature_bundles.json --freeze
.venv/bin/python scripts/round2_embedding_repeats.py --production-manifest reports/further_research/round2/extractor/full_training/feature_bundles.json --execute
```

Default sources are reports/further_research/round2/embedding and reports/further_research/round2/sequence. Default output is reports/further_research/round2/reproducibility/embedding_selector. This document is an implementation draft, not a claim that main outputs, repeat fits or repeat results have completed. Only synthetic tests and read-only implementation-provenance checks have run for this addition.
