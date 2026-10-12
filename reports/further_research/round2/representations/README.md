# Frozen Boltz representation feature protocol

This implements the bounded representation screen in [the canonical plan](../../../NEXT_RESEARCH_PLAN.md), sections 4–5. It derives label-independent features only. It does not fit predictors, choose representations, or commission structures. The full HLA ectodomain + beta2m + nine-residue peptide remains the Boltz input.

The extraction cache must use `pepstab-boltz-interface-v2`: peptide token channels `9×384`, groove token channels `182×384`, full-chain means `3×384` in A/HLA, B/beta2m, C/peptide order, and both actual directional interface tensors `9×182×128`. The reverse tensor stores `z[groove,peptide]`, transposed into the same peptide/groove layout. R2 requires all three full-chain means; a mean of the cropped groove would answer a different question.

| Arm | Frozen construction | Dimensions |
|---|---|---:|
| R2 | Concatenated full-chain means A, B, C | 1,152 |
| R3 | P1–P9 then groove residues 1–182; shared fixed 384-to-8 channel projection at each position | 1,528 |
| R4 | Separate groove means for each peptide position and each direction | 2,304 |
| R5 | Separate contact means at P1, P2, P9, plus the equal mean of six P3–P8 per-position contact means; both directions | 1,024 |
| R5_shuffled | Same pooling and per-position contact counts as R5, with frozen shuffled HLA membership before pooling | 1,024 |
| R5_position9x34 | All nine peptide positions × all 34 canonical contact positions, both directions; shared fixed 128-to-4 channel projection | 2,448 |
| R3_pooled | Same R3 tokens/channels/dimensions, with peptide mean tiled over P1–P9 and groove mean tiled over 1–182 before projection | 1,528 |
| R5_position9x34_pooled | Same position-indexed channels/dimensions; projected pocket mean per peptide/direction tiled over all 34 HLA positions | 2,448 |

All features use float32 and C-order flattening. R4 orders peptide, direction, channel. R5 orders P1, P2, P9, central block, then direction and channel. The position-indexed arm orders peptide, HLA contact position, direction, channel. The eight embedding arms together require 53,824 uncompressed bytes per case, before identity metadata. R0 and R1 integration belongs to the assessment runner.

The two additional pooled controls hold feature dimensions and projected channel space constant with their positional counterparts, and receive independently selected regularisation. They are secondary controls excluded from the primary embedding selector. The primary positional contrasts are R3 versus R3_pooled, R5 versus R5_shuffled, and R5_position9x34 versus R5_position9x34_pooled, separately by learner. R3 versus R2 also changes full-chain/beta2m context, and position9x34 versus R5 also changes projected channels and central pooling; those component contrasts alone cannot isolate positional information.

The fixed random projections are a declared scope amendment to bound fitting and transfer cost. NumPy PCG64 seed **20261012** supplies an independent standard Gaussian matrix for each declared shape, followed by reduced QR and signs chosen to make the diagonal of R positive. The resulting orthonormal-column matrices are cast to float32. They never see inputs or labels. Their byte hashes and dimensions are frozen in [feature_protocol.json](feature_protocol.json). This screen tests compressed positional channels; it cannot rule out benefits from full channels or dimensionality reduction fitted inside each fitting partition. Head capacity and regularisation still require the declared matched assessment.

The [contact map](contact_map.json) uses **all five existing crystals: 1X7Q, 1XR9, 4QRU, 5N6B and 7LFZ**. Model 1, first alternate conformer, non-hetero heavy atoms excluding H and D, chain A/HLA and chain C/peptide are used. Contact membership is the union of minimum heavy-atom distances ≤4.5 Å across the five structures, restricted to `pepstab.esm.CONTACT_POSITIONS_1BASED`. That existing canonical NetMHCpan list was already verified against the supplied pseudosequences for all 75 alleles.

For every crystal, the CIF polymer sequence scheme verifies author, PDB and label numbers equal mature positions 1–182 for A and 1–9 for C, without insertion codes. The mature HLA N-terminal motif is checked. The artifact freezes the groove/peptide sequences, CIF SHA256 hashes, per-crystal distances, map hash and pocket membership. The pocket sizes from P1 to P9 are **9, 12, 11, 6, 10, 6, 5, 5 and 11**. The predeclared nearest-contact fallback is recorded but **not applied**. No predicted coordinates or half-life outcomes define this map.

The shuffle uses a fixed PCG64(20261012) permutation of the 34 columns of the membership mask. It changes the HLA residues selected from each raw directional tensor before constrained means are computed, preserving each peptide position's pocket size. It does not permute the output feature columns. Tests verify both changed membership and changed synthetic features; anchor and central positions remain present in both arms.

The stable API is `representation_features(blocks, contact_map) -> dict[str, np.ndarray]`. `load_contact_map()` loads and validates the frozen artifact. `build_feature_bundle(entries, output_directory, contact_map)` accepts ordered manifest entries containing `pair_id`, `input_hash` and `cache_path`. Each cache is validated against its own hashes and requested input hash; its pair identity must match (the nested `case.pair_id` location is supported, with consistent top-level identity if also supplied). Duplicate pair IDs, incomplete caches, malformed shapes and non-finite features fail closed. The builder writes content-addressed compact NPZ/manifest files and publishes `complete.json` atomically last; it leaves raw caches untouched.

To regenerate and verify the frozen map/projections locally:

```sh
.venv/bin/python scripts/round2_representation_features.py --freeze-protocol
```

To build compact features:

```sh
.venv/bin/python scripts/round2_representation_features.py \
  --manifest ordered-cache-manifest.json --output-directory features/round2
```

The manifest is either an entry list or an object with an `entries` list. Paths relative to the manifest are supported. An optional `--cache-list` supplies one cache path per manifest row in matching order. The compact NPZ preserves ordered `pair_id` and `input_hash` arrays; its manifest records raw cache hashes, the contact-map hash and the full feature protocol. Production workers may call the per-case API and combine its results without transferring the raw cache.

Validation: `tests/test_round2_representations.py` passes 11 tests covering map reproducibility/numbering, orthonormal projections, full-chain pooling, exact positional/directional layouts, central-block weighting, actual shuffle changes, matched projected controls, finite checks, stale identity rejection and raw-cache preservation. These engineering checks do not establish extraction equivalence, downstream performance or statistical validity.

The prepared `scripts/round2_embedding_screen.py` runner freezes three regularisations per representation/learner, three inner cluster folds, two inner MLP seeds and 30 final MLP members (five stopping folds × six seeds). Ridge is deterministic and fitted once per call. There are 10 representations including R0 and R1, giving **580 actual model fits per outer fold and 2,900 over five folds**, before the separately budgeted original/competitive sequence programme. All scaling is fitted inside the relevant fitting role. The runner stores per-component selection, predictions, members and full ancestry and resumes partial fitting from identity-specific cache entries.

The embedding selector may select R2–R5 ridge/MLP or baseline-only original `mlp_log`, with original first under the parsimony rule. It waits for the sequence fold's completed outputs before finalizing. Every component is retained. The prespecified primary comparison is embedding selector versus sequence selector; all component comparisons against original and competitive sequence references and all positional contrasts are frozen in the runner specification. Its input is the extraction controller's complete `feature_bundles.json`, with exact training-row coverage and validated content/map/projection hashes. Synthetic real-head tests cover outer-label invariance, fitting-role scaling, exact 30-member final MLP, feature-cache identity and preserved compact row identity. No real candidate fit was executed during implementation.
