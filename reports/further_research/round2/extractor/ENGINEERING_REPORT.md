# Boltz trunk extraction engineering gate

12 October 2026. Milestone 1 of the [canonical research plan](../../../NEXT_RESEARCH_PLAN.md).
**Technical equivalence and the 64-case cost profile passed.** This is engineering evidence; no supervised outcomes or labels were inspected. A full original-training cohort has been frozen but no full-cohort extraction was launched as part of this report.

## Exact extraction point and inputs

The extractor requires `boltz==2.1.1` and verifies the byte-for-byte upstream `Boltz2.forward` hash before compiling its original prefix with a dedicated return of `s` and `z` immediately before the distogram head. Three requested recycling steps execute four trunk updates. The original class/instance forward and original folding pipeline remain unchanged. A separate process temporarily supplies an embedding prediction step and writer; the normal coordinate writer is not asked to consume missing coordinates.

The checkpoint is pinned to Hugging Face revision `6fdef46d763fee7fbb83ca5501ccceff43b85607`; its actual bytes must match SHA-256 `090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1`, verified against the revision's LFS digest. See [source audit](source_audit.json) for the wheel, module and forward hashes.

Inputs retain HLA ectodomain A (275 residues), beta2m B (99), and peptide C (9), with production cached MSAs, seed 0, no MSA subsampling, 1024 maximum rows, kernels enabled, and Lightning `bf16-mixed`. Actual final trunk arrays were float32. Runtime: NVIDIA A10, Torch 2.14.1+cu130, CUDA 13.0. This records the tested numerical environment; it does not claim the unspecified winning pipeline has been reproduced.

## Technical panel and acceptance

Eight distinct original-training rows span engineered C67S and borrowed-alpha3 constructs. Selection uses allele/construct coverage and smallest canonical pair ID, with labels removed first. All production HLA MSA caches have depth 1024; two explicitly labelled technical variants use depths 32 and 128. Full and extraction paths receive the same preprocessed batch and restored RNG state. Token features must exactly reconstruct every chain sequence and mature residue index; padding and unmapped tokens cannot pass.

A final-pairformer hook captures full-forward tensors, then the original forward continues through all 200 diffusion steps and confidence. Extraction forbids distogram, diffusion conditioning, diffusion sampling, and confidence calls. Both native tensors and repeated extraction were **bitwise identical on all 10 cases**, with maximum absolute error 0. Declared float32 tolerance was `atol=rtol=1e-5`; lower-precision tolerances were frozen in the source before execution. The source boundary also excludes affinity. Full arrays had shapes `s=(1,383,384)`, `z=(1,383,383,128)`.

| pair_id | Allele | MSA depth | Both tensors exact | Extraction seconds |
|---|---|---:|---|---:|
| 7294 | HLA-B*14:01(C67S) | 1024 | yes | 7.097 |
| 0 | HLA-A*02:01 | 1024 | yes | 7.115 |
| 6343 | HLA-A*02:50 | 1024 | yes | 7.130 |
| 12 | HLA-B*07:02 | 1024 | yes | 7.171 |
| 7343 | HLA-A*24:19 | 1024 | yes | 7.176 |
| 7277 | HLA-B*39:06(C67S) | 1024 | yes | 7.200 |
| 3289 | HLA-B*08:03 | 1024 | yes | 7.203 |
| 137 | HLA-A*11:01 | 1024 | yes | 7.209 |
| 0 | HLA-A*02:01 | 32 | yes | 6.755 |
| 7294 | HLA-B*14:01(C67S) | 128 | yes | 6.795 |

Every case passed finite checks, full token identity, both directional indexing, float16 storage conversion (`atol=rtol=1e-3`), and exact serialization/reload of stored blocks. Float16 conversion can have sizable absolute differences for large activations; its relative tolerance is explicit and is not evidence that every downstream predictor is insensitive to quantisation. Full tensors are retained on two audit cases only.

## Cache contract

Each entry uses content-addressed NPZ/metadata files and an atomic `complete.json` published last. Validation checks identity/settings hashes, file hashes, block names/shapes/dtypes and finite values. A CIF or partial NPZ cannot count as a completed embedding. Existing input-independent structure files never suppress extraction.

| v2 block | Shape | Meaning |
|---|---|---|
| `peptide_s` | 9 × 384 | P1–P9 |
| `groove_s` | 182 × 384 | Mature HLA residues 1–182 |
| `peptide_to_groove_z` | 9 × 182 × 128 | Actual z[peptide, groove] |
| `groove_to_peptide_z` | 9 × 182 × 128 | Actual z[groove, peptide], transposed to common layout |
| `chain_means_s` | 3 × 384 | Means over full A275, B99, C9 chains, in that order |

The technical audit first wrote v1 (the four required interface blocks). V2 adds means computed from the unchanged full native `s`; the 64-case profile validates v2. The validator still reads v1 audit caches. Every metadata file contains the complete token map, source case/chains, actual MSA hashes, preprocessed tensor hash, checkpoint/input hashes, extraction AST hash and settings.

[Compact feature audit](compact_feature_audit.json) verifies all eight frozen representation/control arrays from an actual downloaded v2 cache. Production bundles derive features from **reloaded float16 blocks** and record contact-map/projection protocol and source hashes. Raw activations remain on the Volume; compact bundles are downloaded and verified per shard.

## Measured profile and cost

The profile covered 64 distinct training peptides/alleles, all with original-depth MSAs. It passed 64/64. Warm trunk median was **7.236s**, p95 **7.249s**; median case processing was 7.419s. Container total was **572.924s**, including 67.443s for imports/hash/input preparation and model/preprocessing startup. Between-batch work raises amortised warm cost to 7.898s/case. Peak GPU allocated memory was 4.458GB and worker peak host RSS 10.173GB. Requested/hard host allocation remains 24 GiB and four cores.

Compressed embedding arrays average 922,412 bytes/case. Full original-training arrays project to 18.19GB; including a 64KiB/entry metadata allowance gives 19.48GB. Compact arrays add approximately 53,824 uncompressed bytes/case; the actual single-case compressed audit was 36,316 bytes. Full audit tensors are a fixed two-case overhead.

Current verified runtime rate is $1.4812/hour: A10G $1.10 + four CPU cores $0.1892 + 24 GiB $0.192. Verification estimated $0.1615 and the profile $0.2365. Provider-reported verification cost was $0.16292828 at the first completed billing snapshot; reports can lag. Storage list pricing is $0.09/GiB/month with 1 TiB/month free, but unused workspace allowance is not assumed. [Modal pricing](https://modal.com/pricing).

Repeating 64-case shards for all 19,716 rows forecasts $72.62 (or $90.77 with a 25% planning allowance), about 6.13 hours at eight workers. [Larger-batch sensitivity](batch_forecast_options.json) prices 256-case shards at $69.48 including feature/input preparation allowances, or $71.40 including a further 60 seconds reserved per shard. These larger-batch values are extrapolations; the first bounded production shard must validate them before fanout. The 25% margin is a planning allowance, not a statistical confidence interval.

## Artifacts and next gate

- [Frozen technical panel](technical_panel.json), [64-case profile panel](profile_panel.json), [full training cohort manifest](full_training/cohort_manifest.json).
- [Raw equivalence report](round2-verify-20261012-a.json), [raw profile report](round2-profile-20261012-a.json), [reproducible forecast](forecast.json).
- [Verification app](https://modal.com/apps/a-cheparukhin/main/ap-LWIdYOzFfzkVvT0pU8o7zp), [profile app](https://modal.com/apps/a-cheparukhin/main/ap-ailqSkkRms7DNOJe0S66Qa). Output Volume: `pepstab-round2-boltz-embeddings`.

Nine focused tests cover the pinned extraction boundary, token mapping failures, asymmetry, serialization/corruption, backward-compatible schemas, whole-cluster cohort selection and parallel budget reservations. The repository-wide assessment gate and downstream learning remain separate responsibilities. Before launch, the orchestrator must approve the full-cohort budget/batch amendment and first-shard cost gate. The 19,716-row cohort has not been silently reduced.
