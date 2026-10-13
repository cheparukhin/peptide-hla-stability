# Static input and command checklist

This is operational advice only, not a scientific amendment, budget admission
or launch. Original frozen sources and protocols remain unchanged.

Energy v2 collect/heads/report

- Finish the complete row download and source-bound manual73-case recovery.
  Preserve the original4,376 successes and73 interrupted starts. No subset
  assessment. Use the recovery's verified combined_rows directory, and pin its
  manifest/recovery receipt in subsequent operational and independent audits.
- Use scripts/followup3_energy_v2.py throughout; its default output is
  reports/followup3/energy_execution_v2. Do not call freeze again. The original
  scoring directory is energy_scoring/03808264b86e217945bdde190d77d71699f46c6818b1b98cf267ccec95ac0315.
- Collect explicitly with --rows-root pointing to
  reports/followup3/energy_recovery_v3/combined_rows. The collector requires all
  4,449 successful identities with the original protocol hash, RepairALL and
  complete finite29-term schema. It produces energy_features.npz/provenance;
  preserve these hashes before assessment. Run collect once, with fresh output
  artifacts: it writes the NPZ before its write-once provenance check.
- Launch exactly10 assessment commands: learner mlp/ridge ×outer-fold0..4.
  Always specify --outer-fold: the energy CLI defaults to0, not all folds.
  Each command fits only40 MLP members or20 ridge fits and reuses exact R0/R5
  predictions after old ancestry/inner-selection replay. No baseline refits.
- Report only after all10 receipts succeed. It verifies exact reused baseline
  vectors, replays new inner selections, audits ancestry, and produces the
  original2000 bootstrap draws,95% intervals and three-primary98.3333% family
  interval. Do not rerun completed head cells: prediction CSVs are written
  before write-once selection/ancestry receipts, and a rerun need not be benign.
- For every phase, include the union of energy-v2 source_sha256/canonical_inputs
  and inverse-v2 source_sha256/input_sha256, plus the v2 protocol/panel/pilot
  files, original energy protocol/panel/pilot files (v2 verifier opens them),
  inverse-v2 protocol/panel/base_features.npz/base_features.provenance.json.
  Also bind pepstab/splits.py and package initialization files if present as
  operational inputs; inner-fold helpers import splits independently.
- Collection needs the4449 combined JSONs. Heads additionally need E NPZ and
  provenance, and old inverse assessment/<learner>/fold_<i>/{receipt.json,
  predictions.csv,selection.json,ancestry.json}. Reporting needs all10 old
  baseline cells and all10 energy cells, including baseline_reuse.json and
  every artifact named by each content_sha256 receipt.
- All statistical phases use the existing pinned numerical runtime, canonical
  ROOT mount /Users/cheparukhin/ai_science_hack and two numerical threads.
  No /structures mount, FoldX binary, torch checkpoint, source mmCIFs or raw
  FoldX process output is needed after verified collection. Sources checked by
  protocol hashes remain present even when their extraction code is not called.
- Bind existing remote JSON/feature files with remote_inputs rather than upload
  >3000 entries. The existing runner verifies remote_inputs only at startup;
  independently recheck immutable hashes at phase completion. Use a fresh
  resource namespace and admission for each bounded job. Sequential collect,
  heads and report cannot share one concurrently launched command graph.

Full ORIGINAL-train ProteinMPNN candidate

- Reviewed draft SHA is2b466ce664f0e30a6f75073e91fff210f5abe0b2ae864a1038eaf15657f5b414.
  Wait for completed energy audit and explicit root decision. Decision fields
  are approve_fulltrain_inverse=true, energy_final_audit_complete=true and
  candidate_draft_sha256 equal to that SHA. Freeze creates protocol.json only
  after checking unchanged23 source/37 input bindings. Freeze is not admission.
- Each extraction plan includes new protocol/panel/cache_manifest/batch files,
  all source/input bindings and the4,449 inherited JSONs via remote_inputs at
  their existing inverse_execution_v2/score_rows paths. Do not copy or rescore
  those4,449 cases. Batch0/1 contain disjoint7,634/7,633 new rows. Use separate
  supervisors/resource namespaces and one command per Sandbox, with eight
  spawned child workers×two threads,16CPU/32GiB, five-hour/$17 caps each.
- Extraction requires the existing pepstab-structures volume mounted at
  /structures (primary merged volume includes both historical halves), and
  existing external/ProteinMPNN/protein_mpnn_utils.py plus
  vanilla_model_weights/v_48_020.pt at the repository path. Pin the checkpoint
  separately in the execution plan even though the scorer rechecks its SHA.
  Use corrected exact-unique-chain selector and all unchanged native checks.
- Commands are extract-batch --batch0/1 --workers8, then collect. Both batch
  receipts must pass; partial or failed coverage blocks all assessment.
  Do not run the original inverse prepare/extract commands against the new
  namespace: they enforce the original64-per-allele/six-arm scope.
- Stage uses the existing full19716 R5 cache, not a new Boltz extraction:
  reports/further_research/round2/extractor/full_training/feature_bundles.json
  plus78 bundle directories, each complete.json and its named NPZ/metadata JSON
  (234 existing remote files). Default bundle paths are absolute canonical
  ROOT paths. --bundle-root alternatively denotes their common parent; the
  stager appends each shard basename, not the entire stored absolute path.
- Staging also needs pepstab/features.py, round2_representations.py, esm.py
  (contact-position constants), boltz_embeddings.py, and
  reports/further_research/round2/representations/contact_map.json. Add esm.py
  and the contact JSON to the operational hash closure; they are imported/read
  by the reused stager beyond the reviewed draft's direct binding list.
  Include canonical raw/splits/training-target JSON+NPZ and inherited fold
  manifest. No Boltz weights, GPU, raw trunk caches or crystal coordinates
  are opened by this stage path. /structures is unnecessary for staging.
- Fresh base_features.npz/provenance are required; the old4,449-row base NPZ
  cannot be reused for19716 rows. The reused stager also writes R0 for transport
  consistency, but only R5/R5_P/R5_PC enter the new450-fit assessment. Stage is
  write-once and refuses an existing base NPZ.
- Run10 fresh assessment cells (mlp/ridge ×fold0..4), then report. All three
  arms refit; old subset predictions are used only for the prospective
  fixed-query descriptive table. Its exact old predictions/receipts/audits and
  the possible saved main-selector reference are already draft-bound inputs.
- Independent final audit must verify complete19716 native cases, disjoint new
  batches, cached-byte preservation,450 fits, held/fitting/stopping/scaler
  cluster exclusion, all saved inner vectors/choices, matching feature hashes,
  and fixed-query arithmetic. This remains selected development evidence, not
  an independent confirmation or frozen-tranche primary.
