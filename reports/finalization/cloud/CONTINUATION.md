# Modal continuation after the durable transfer

12 October 2026. This is an operational runbook, not a launch authorization.
Read the actual processes and immutable receipts before each action. The only
current training owner is the existing launchd transfer recovery described in
`../RECOVERY_OPERATIONS.md`; never duplicate or migrate it. Future heavy phases
run on Modal. The Mac only orchestrates, edits and checks hashes/small fixtures.

## Scientific and budget boundaries

Keep all 35 transfer-bound sources and all original hybrid-bound sources
unchanged. The hybrid scientific protocol remains
`c5fd5501d8b640428305a1eb5b2d9fd83c3fb47139faa7de055550cb02de3955`.
There are exactly 12 unscored pilot fits and 480 scientific fits across ten jobs.
The original test remains excluded. No pilot result changes the scientific
scope, recipes, selection, stopping or uncertainty calculations.

Four sequential CPU Sandboxes perform transfer reporting, the hybrid pilot,
hybrid main fitting, and hybrid saved-model audit/reporting. See
`OPERATIONS_DESIGN.md` for the exact resource and time caps. The four phase caps
are $5, $2, $20 and $5, with $3 for storage, egress and engineering overhead:
at most **$35**, including all these future phases. Carry the previous
**$5.1793820030564595** conservatively within the user's unchanged $100 new-phase
ceiling. This is a ceiling, not a target. Do not release reservations on lagging
provider usage alone. No additional cloud allocation exists until an exact
reviewed phase authorization is written.

Use the existing immutable image `im-b7zNuOIkXIwp4tdIyIggdN` only if the remote
startup check confirms Python 3.12.12 and all six frozen package versions.
Missing compatibility stops before fitting; it never authorizes a silent
upgrade or an automatic second Sandbox. Any bounded image repair needs a
separate recorded operational decision within the same overhead envelope.

## Close the existing transfer truthfully

1. Require all 30 cells, the real durable completion certificate, successful
   controller exit, and exact owner/child absence. The original deadline stays
   `2026-10-13T15:32:19.924192+00:00`. The completion certificate already binds
   its full semantic and checkpoint audit; do not rerun that heavy audit locally.
2. Bind the real owner observation in `transfer_owner_observation.json`, the
   exact authorization, attempt, completion and exit, and all required evidence
   into `portable_transfer_closure.json` using schema
   `finalization_portable_transfer_closure.1`. Validate with
   `pepstab.final_cloud_protocol.verify_predecessor`. Never invent a successful
   receipt for either interrupted attempt or probe Mac PIDs inside Linux.
3. After absence is verified, remove only the two recorded one-shot launchd
   registrations. Do not send signals to historical PIDs without checking their
   current identity.
4. Preserve retained 10,830 scientific fits, 1,785 known discarded completed
   fits, 12 uncertain original remote attempts, and any unrecorded interrupted
   local fits as distinct categories. Eventual known completed total 12,615
   excludes engineering pilots and uncertain attempts.

## Freeze and materialize each phase

Copy `protocol.json`, `SPECIFICATION.md`, `fit_graph.json`, `task_manifest.json`
and `preflight.json` from `hybrids_recovery_v2` into `hybrids_cloud_v1`
byte-for-byte. Do not issue any old local hybrid authorization or gate.

Use `scripts/final_cloud_inputs.py` to freeze the finite input manifest only
after the transfer closure exists. Include explicit extra files for all new
operational scripts and tests, the cloud scope's five metadata files, accounting,
portable closure, and all prerequisite phase certificates and their bound
artifacts. The existing `input_inventory.json` is a planning snapshot, not a
complete execution manifest. It intentionally reports missing unfinished cells.
Do not upload the entire repository, credentials, logs or live/partial outputs.
The canonical raw CSV and splits are copied as opaque inputs; only the frozen
training partition is opened by scientific code.

Input manifests do not contain themselves or their own authorizations. Hash and
upload those two files separately after freeze. Use exact original repository
paths inside a dedicated Volume v2 mounted at
`/Users/cheparukhin/ai_science_hack`. Input size is approximately 2 GiB and must
be measured after closure; do not use the obsolete 1.5 GiB estimate. A 3 GiB
input cap and 10 GiB output cap fit the proposed storage/egress reservation.

Each phase gets a separately named App and Volume, its own exclusive attempt,
source-bound `final_cloud_phase_authorization.1`, accounting reservation and
`final_cloud_resource_plan.1`. `scripts/final_cloud_resources.py` refuses
pre-existing resource names, uncertain retries, oversized bytes and missing
closure. It creates no compute. The resource plan expires 15 minutes after
creation. Finish verified download and delete the preceding phase's temporary
Volume before retaining another; keep authentic local evidence. Record actual
deletion and provider stop before releasing storage reservations.

## Launch exactly once and observe real completion

The immutable `final_sandbox_launch.1` plan binds the exact App, Volume v2 and
image IDs, resource request and hard limit, phase source/input hashes, fixed
work deadline and provider-stop deadline with at least 120 seconds of cushion.
Its admission window is four minutes. The outer provider stop must be no later
than `2026-10-14T14:15:17.567539+00:00`; restarting a client never slides it.

Register an independent watcher under launchd with `KeepAlive: false` and no
recurrence. Wait for its exact fresh waiting status before launching the
one-shot launcher, also under launchd. Do not use `launchctl submit`.
`scripts/final_sandbox_driver.py launch` is the intended create/release sequence;
its exit acknowledges release only, never scientific completion. The watcher
must own one persistent stdin writer for the initial authenticated release and
every renewal; fresh Modal handles reset their stream offset. It grants at most
90 seconds per lease. Missing supervision, EOF, wrong identity, duplicate
provider objects or expired time closes work and stops exact owned processes.

Unknown create/release/upload results keep their exclusive intents. Reconcile
exact resource IDs and actual provider state; never rerun a command speculatively.
On unexpected infrastructure replacement or interruption, preserve partials and
stop for a separately bounded operational recovery rather than retrying fits.

Accept completion only with actual provider return code zero, the exact remote
supervisor exit, cleanup confirmation, bound output manifest, verified downloaded
bytes and source/input identity. A stopped Sandbox alone is insufficient. Produce
the portable cloud phase certificate only after these checks. Download only
allowlisted phase output/receipt paths; refuse differing existing local content.

Use only the completed unscored pilot's runtime and memory, including a larger
final-partition allowance, to freeze the separate main forecast and authorization.
The main cap remains two fit workers with two numerical threads each, within
four physical cores and 16 GiB. Pilot fits are never reused scientifically.

Run each heavy report once on Modal. Validate all saved-model, ancestry,
selection and statistical audits there. Then update `FINAL_REPORT.md`, README,
the submission brief and the source-backed local site; regenerate its snapshot
after final-report changes. Claims remain conditional development evidence;
independent assay confirmation remains open. Commit completed validated changes,
excluding partial/live/unrelated output. Keep the existing heartbeat active
until all achievable delivery and cloud/storage closeout work is complete.
