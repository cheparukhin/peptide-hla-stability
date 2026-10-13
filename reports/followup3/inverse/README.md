# ProteinMPNN complementarity: bounded training development study

Implementation: `pepstab/followup3_inverse.py` and
`scripts/followup3_inverse.py`. The shared followup3 protocol governs scientific
interpretation. Root controls every paid launch and its cumulative guard.

The panel is selected without labels or features: at most 64 SHA-ordered pairs
per allele from ORIGINAL train, inheriting the existing five whole-cluster
outer folds. Sampling part of a cluster does not permit its retained
occurrences to cross folds. No old validation/test pair or reserved external
label enters extraction or assessment. Production structures are joined using
`(allele, peptide)`; positional production `pair_id` is never a structure join.
Canonical IDs are used only after the original dataset label/feature join.

Both contexts score the native peptide with checkpoint `v_48_020.pt`, zero
backbone noise and exactly 16 common peptide decoding permutations from seed 0.
All visible HLA/beta2m residues precede peptide decoding. C retains the full
complex; P strips the other chains from the same backbone. Nine positions per
context enter fitting. Totals and C-minus-P remain diagnostics. The old
2,000-row QC cache has 1,346 original-training rows, 230 validation rows and 424
test rows; its unpaired order protocol differs, so none of its scores is reused.

Six matched arms use R0, R0+P, R0+P+C, R5, R5+P and R5+P+C. The primary is
the two-member hidden 64 MLP context increment R5+P+C minus R5+P; the same
ridge comparisons are sensitivity results. There are 600 MLP members and 300
ridge fits. Every arm selects its own regularization using pooled clean inner
OOF predictions, three inner whole-cluster folds and exact stronger-tie rules.
Both inner and outer MLP calls deliberately use the existing FeatureHead
`final=False` two-member recipe, not its 30-member full-cohort final recipe.
The canonical target transport restores the frozen training log1p bits on
Linux. Saved inner vectors support independent scoring and selection replay;
complete neural gradient/stopping/scaler ancestry and member telemetry are
retained. Fitted model weights are not saved, so weight replay is not claimed.

The report uses shared 2,000 peptide-cluster draws for all declared contrasts,
the representation-by-context interaction, pointwise 95% intervals and the
primary 97.5% interval. The primary panel has at least 20 rows per allele;
at least 50 is sensitivity. All findings remain conditional development evidence.
The context ablation also changes graph size/neighbors and cannot establish
causal geometry, kinetics, pose validity or independent confirmation.

## Commands and cloud inputs

`prepare` is lightweight and does not import torch/biotite. Freeze only after
source review, then send the same source bytes and generated artifacts to the
cloud. Preparation itself does not authorize paid execution.

```
python scripts/followup3_inverse.py prepare
python scripts/followup3_inverse.py pilot --volume-root /structures --workers 1
python scripts/followup3_inverse.py extract --volume-root /structures --workers 8
python scripts/followup3_inverse.py stage --bundle-root /path/to/round2-fulltrain-20261012
python scripts/followup3_inverse.py assess --learner mlp --outer-fold 0
python scripts/followup3_inverse.py assess --learner ridge --outer-fold 0
python scripts/followup3_inverse.py report
```

The five-row pilot is engineering-only, selected deterministically across the
frozen panel. Main extraction retains identical verified pilot features.
Failures persist as per-row JSON and a score table and stop complete-panel
assessment; no complete-case replacement is supported. Root sets absolute
deadlines, cost reservations and process concurrency outside this scientific CLI.

The merged primary `pepstab-structures` volume contains both historical halves
under `stage4c/ectodomain-20261004/boltz2/production/HLA-A_01_01/...` and analogous
allele directories. `modal_app/merge_structures.py` documents this provenance.
No colleague-profile launch is needed if root confirms the merged inventory.
Required cloud runtime dependencies: numpy/pandas/scipy/scikit-learn,
threadpoolctl, CPU torch and biotite. Pin the scientific runtime before fitting.
Stage the existing ProteinMPNN source/checkpoint under `external/ProteinMPNN`,
the original dataset/splits/cohort and sequence fold manifest, the canonical
training-target JSON/NPZ, contact-map files, and cached R5 compact shards plus
their completion/metadata files. No encoder or structure prediction reruns.

Each extraction worker reserves two torch threads; at most eight workers are
accepted. Each assessment process caps numerical libraries to two threads.
The ten learner/fold assessment jobs may be scheduled separately by root.

## Feasibility forecast, not a current measurement

Historical QC telemetry was 9.83 seconds per conditional score on two CPUs and
4 GiB at 16 orders. The earlier smoke used 14.70 seconds. At 4,800 rows, using
the observed QC time separately for both contexts conservatively gives 52.4 core-hours
and 104.9 GiB-hours, excluding startup. Applying recorded ordinary Function
rates ($0.04730/core-hour, $0.008/GiB-hour) gives about $3.32; the historical
Sandbox CPU rate is three times higher, giving about $8.27. This is a derived
forecast; it does not measure the explicit-order implementation or peptide-only
runtime. The actual five-row pilot must gate throughput/memory and the fixed
main reservation. Fitting/reporting and startup need separate root reservations.

## FoldX inventory only

Local `foldx5_Linux_0/foldx_20261231` exists, is executable, 86,717,072 bytes,
and is a static Linux x86-64 ELF. Its SHA256 is
`faed5e54e47744ab6ab75f8f9ad1a1f2e2cdbeac4f0f98d4e36c97049eb14268`, matching
historical pilot records. The adjacent `molecules/` directory exists. No binary
was executed in this inventory and no license strings or credentials were read.
Binary presence does not establish entitlement for a new cloud run.

Stage8 is a diagnostic pilot record; no trained FoldX half-life arm exists.
Repair historically changed energy ordering and cost far more than raw scoring;
the repaired contention factor was not established by the aborted larger pilot.
A tiny repaired technical check would need root to resolve access and authorize
its bounded spend first. It would assess parsing/coordinates/runtime only, with
no half-life result or performance-ceiling inference.
