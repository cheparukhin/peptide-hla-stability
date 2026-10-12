# Finite cloud input inventory

This is a metadata-only dependency inventory prepared on 12 October 2026. It is not a completed input manifest, a launch authorization or an upload. Transfer controller PID59308 is still the authorized local owner. No source in its 35-file binding has been changed, and no dataset table or feature array was opened for this inventory.

`input_inventory.json` lists explicit repository-relative file paths, their known frozen hashes, roles, existence and byte sizes. A null expected hash means the final freeze must compute and pin it; it does not mean the bytes were verified. `scripts/final_cloud_inputs.py` is a standard-library-only planner and manifest builder. Planning reads JSON receipts/protocol metadata, Python source ASTs and filesystem statistics. It does not import scientific modules, evaluate outcomes, fit models, copy files, compress archives, inspect process ownership or call Modal.

## Paths and measured size

The container must materialize `/Users/cheparukhin/ai_science_hack` exactly. The transfer protocol uses the absolute legacy directory `reports/further_research/round2/transfer` under this root. The compact production manifest and all 78 feature bundle paths also contain this absolute prefix. Rewriting those strings changes their hashes and feature provenance.

At the inventory snapshot, 21 transfer cells were complete, including three new durable cells. The explicit existing files occupy **1,640,966,621 bytes (1.528 GiB)**. Nine cells and their checkpoint caches remain deferred. Each of the three completed new joint cells plus its cache occupies 35.96–36.35 MB; extrapolating the largest gives about **1.97 GB (1.83 GiB)** for the completed base input set. This is an estimate, not an upload cap or a completion claim. Allow additional measured bytes for the separate cloud-scope copies, closure/accounting certificates and phase-specific outputs. A proposed 1.5 GB initial upload bound is insufficient. Freeze and sum actual complete sizes before fixing the transport/storage reservation; do not stage or compress a broad checkout on the Mac.

| Required input group | Existing bytes | Inclusion reason |
|---|---:|---|
| 78 compact feature bundles | 673,860,208 | Exactly three referenced files per bundle: completion JSON, metadata JSON, NPZ |
| All 30 legacy transfer cells | 414,783,905 | 180 explicitly hash-bound scientific evidence files used by unchanged `reuse_cell`/legacy audit |
| 21 current completed transfer cells | 475,786,487 | Completion plus eight scientific files per cell; remaining nine deferred |
| Three completed new-cell checkpoint caches | 42,776,420 | 104 calls per joint cell, three transaction files per call; remaining nine deferred |
| Canonical raw CSV | 6,912,310 | Opaque exact bytes required by frozen hash checks and the unchanged training reader |
| Canonical splits CSV | 1,132,751 | Frozen split identity and training-only selection |
| Hybrid task manifest | 20,058,388 | Exact frozen supervised-member schedule |

The raw canonical CSV physically contains all original splits. It is copied as an opaque hash-bound artifact because the unchanged reader needs it; this grants no original-test scoring or inspection. The scientific entrypoints continue to select original training rows only.

## Dependency closure

The planner includes all 35 protected transfer authorization sources, the 30-source original transfer map, the 23-source hybrid map, all legacy source bindings, and the four explicit new cloud operational entrypoints. Overlapping paths are deduplicated and contradictory expected hashes fail. New operational modules are separately pinned at manifest freeze and in phase authorization; historical scientific maps are not rewritten.

Python imports are discovered statically from the actual reporting/worker entrypoints, including imports inside functions and package initializers. Eight supporting modules outside the frozen source maps were found: `pepstab/__init__.py`, `pepstab/augment.py`, `pepstab/esm.py`, `pepstab/legacy_assessment.py`, `scripts/baseline_ensemble.py`, `scripts/baseline_sequence.py`, `scripts/phase_d_structural_residual.py`, and `scripts/stage5_structural_arm.py`. Including these small source files does not authorize executing their unrelated workflows. External packages remain governed by the separately pinned Python 3.12.12 image and dependency versions.

For transfer reporting, the closure includes the exact new protocol, legacy `assessment_protocol.json`, legacy source hashes, grouping `protocol.json` and every declared grouping file, and all files in `legacy_evidence_sha256`. `final_transfer.validate_frozen`, `legacy_protocol` and `reuse_cell` can therefore operate at their original paths. Each of the 30 new completion records contributes exactly its eight bound scientific files.

For hybrid work, it includes the unchanged protocol, specification, fit graph, task manifest, preflight, population and feature identity references, frozen ranking protocol and five descriptive historical reference files. Root must additionally materialize byte-identical `protocol.json`, `SPECIFICATION.md`, `fit_graph.json`, `task_manifest.json` and `preflight.json` under `reports/finalization/hybrids_cloud_v1`, then pass those five explicit paths as phase extras. No existing local protocol is changed.

`final_ranking.load_features` rebuilds R0 from canonical sequence fields and calls `load_compact_features` for the compact bundle manifest. That loader opens every declared representation array in each NPZ, not only R5. The inventory therefore includes each entire referenced NPZ, metadata file and `complete.json`, plus `reports/further_research/round2/representations/contact_map.json` and the production feature manifest. Raw Boltz activations, PDBs, projection scratch files and unrelated extractor shards are not required. Fixed projections are defined by source; no separate learned projection file is read.

Historical nomination-only pointers inside the legacy assessment and ranking protocols are preserved inside their immutable JSON bytes. They are not expanded into unrelated historical fit directories: the selected entrypoints do not call the old main-screen gate or cached ranking fitter. This is execution dependency closure, not a recursive export of every historical experiment mentioned in a document.

## Closure and deferred artifacts

The final manifest cannot be produced until the exact durable 30-cell completion and successful exit receipts exist, all 30 cells are complete, and all twelve new cells have their exact 104-call checkpoint inventories. Only the recorded `call.json`, `arrays.npz` and `completed.json` files are admitted per call. Pending transactions, directory glob discoveries and additional cache calls are not admitted.

Root must first create `reports/finalization/cloud/portable_transfer_closure.json` using the reviewed portable certificate contract. The inventory includes its original transfer authorization, attempt, completion, exit, both interrupted closures, previous failed local exit/state, preserved-cell audit and explicit `evidence_files`. Historical failed state is immutable evidence; active controller state and progress/log files are excluded. The original Mac checkpoint Python build string remains historical metadata. The cloud verifier compares it with authenticated recorded runtime identity, not the Linux build string. No local heavyweight checkpoint replay is performed here.

The portable certificate must be authenticated by its digest in the separate cloud phase authorization. A manifest hash proves exact bytes, not actual process absence or successful execution; Root's owner/receipt checks and the pure cloud certificate validator remain necessary. The cloud phase authorization itself is deliberately excluded from its input manifest to avoid circularity. The historical transfer authorization is an ordinary immutable input.

## Safe use

Run the inexpensive planner into a new snapshot path:

```sh
.venv/bin/python scripts/final_cloud_inputs.py plan --output /tmp/cloud-input-plan.json
```

After authenticated transfer closure, immutable source freeze and explicit phase extras are ready, Root may run the explicit content-hashing step. This reads bytes but performs no scientific audit, copy or upload:

```sh
.venv/bin/python scripts/final_cloud_inputs.py freeze \
  --output reports/finalization/cloud/input_manifest.json \
  --extra-file reports/finalization/hybrids_cloud_v1/protocol.json \
  --extra-file reports/finalization/hybrids_cloud_v1/SPECIFICATION.md \
  --extra-file reports/finalization/hybrids_cloud_v1/fit_graph.json \
  --extra-file reports/finalization/hybrids_cloud_v1/task_manifest.json \
  --extra-file reports/finalization/hybrids_cloud_v1/preflight.json
```

Add separately enumerated accounting, protocol-wrapper and preceding phase certificate/output files required by the exact phase authorization. Later phases consume the previous phase's explicit output manifest and certificates through the durable Volume; their files must be pinned before that successor is authorized. Never replace a failed launch with a new manifest or automatically continue incomplete fitting.

The output schema is `final_cloud_input_manifest.1`, with `repository_root` equal to the exact original absolute path and `files` mapping relative paths to SHA256. Missing inputs, incomplete closure, changed expected bytes, conflicting source hashes, hidden paths, path escape, symlinks and directory inputs fail closed. The helper creates no archive and refuses overwriting an existing output. Cloud phase authorizations and input manifests are rejected as explicit extras.

Excluded by construction: `.git`, `.venv`, home/config credentials, hidden files, unrelated reports/data, active states/logs/locks, original partial cell outputs and old discarded progress logs. Do not supplement this allowlist with `add_local_dir('.')`, recursive repository upload or a broad tar glob.

Validation: 16 lightweight synthetic-file tests cover safe path handling, symlink rejection, AST import traversal without execution, blocked-freeze refusal and exact allowlist hashing. No real fits, array loads, local scientific audits, staging, compression, cloud mutation or commit was performed for this inventory.
