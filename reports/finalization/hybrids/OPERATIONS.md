# Hybrid experiment execution

The implementation and data protocol were frozen on 12 October 2026 after 37
synthetic, integration and artifact-replay checks and an Astra scientific review.
Protocol: `c5fd5501d8b640428305a1eb5b2d9fd83c3fb47139faa7de055550cb02de3955`.
See `implementation_readiness.json`, `SPECIFICATION.md`, `protocol.json` and
`task_manifest.json`. No hybrid assessment outcomes exist at this checkpoint.

The single waiting pilot queue launched at **14:15:46 UTC**, PID **11628**,
keep-awake helper **11629**. Its command and ownership are in
`../resources/hybrid_pilot/root_launch.json`. The immutable authorization is
`../resources/hybrid_pilot/controller_authorization.json`, SHA256
`0348033e89cbb78a3fc8a43ebf359e769f6bfd401a116cc5671c31763176cb41`.
The queue expires on **14 October at 14:15:17 UTC**; do not extend it or launch
another queue. Its status starts with `controller/pilot/queue_attempt.json`.
The absence of `attempt.json` means fitting has not started.

## Release the pilot after the current transfer study

1. Inspect the exact transfer controller PID89292, watchdog PID89617 and Modal
   app `ap-RrFfbDvriD4rehaba5x7CA`. Preserve their frozen source and deadline.
   Require all30 cell completions, matching successful controller and driver
   exit receipts, stopped app with zero tasks, no fitting processes, finished
   watchdog and released cloud-guard ownership. Verify every cell using the
   frozen transfer artifact checks. Do not treat process absence as success.
2. Reconcile the journal without instantiating its mutating budget class. Include
   provider metering, reservations, preemption charges, initial allowances and
   the $1.3895622678414123 pilot carry. Save an immutable
   `final_scope_reconciliation.1` document binding the exact final guard status
   and authorization. Its conservative cumulative prior charge must meet the
   guard minimum; uncertain accounting carries the whole $65 allocation.
3. Create `../resources/hybrid_pilot/launch_gate.json` once, using the exact
   `final_hybrids_launch_gate.1` contract in `OPERATIONS_DESIGN.md`. Bind the
   pilot authorization and final transfer controller exit, driver exit, guard
   status and reconciliation paths/hashes. Give the gate explicit creation and
   expiry timestamps, with expiry no later than the queue deadline. The queue
   and worker independently check those prerequisites before fitting.
4. The existing queue then runs exactly two six-fit bundles, one representation
   at a time, with two numerical threads and a fixed two-hour phase limit.
   It saves predictions for replay but computes no held-out scores. The
   engineering pilot is never reused in assessment. A failure stops dispatch
   and preserves its attempt; diagnose before any separately documented repair.

The completed transfer report can use the third local worker slot only when
root records a two-thread report limit. Check there is no existing report
process or receipt, then launch the frozen transfer report once and audit its
full results. If report ownership/resources cannot be verified, keep the
hybrid launch gate absent until that report exits.

## Authorize the full hybrid study from runtime only

After the pilot controller exits successfully, audit both six-fit bundles and
their source/ancestry receipts. Use only runtime, memory and completion telemetry
to forecast the ten main jobs. Record a conservative forecast, available disk
and memory, fixed wall limit and the main authorization; never use pilot held
outcomes to alter scope. No automatic main launch is permitted by the pilot.

The main authorization contains exactly ten jobs, `fold_0/R0`, `fold_0/R5`,
through `fold_4/R0`, `fold_4/R5`: **480 supervised fits in80 bundles**, maximum
two worker processes and two numerical threads each, **zero cloud spend**.
Its separate launch gate binds the successful pilot completion/exit receipts
and the runtime forecast, plus the same closed transfer predecessor. Use the
tested queue/controller contract. Preserve all grids, preprocessing recipes,
member schedules and selection rules. No uncertain attempt is blindly retried.

After all ten jobs and the controller complete and exit, check for an existing
hybrid report process/receipt before launching `scripts/final_hybrids_report.py`
once with durable launch and exit records. It replays saved models, transforms,
inner decisions and ancestry before calculating the frozen paired-cluster
intervals. Independently review its completion audit, then integrate both new
studies into the final report, README, submission brief and source-backed local
site. Regenerate the site snapshot after changing the final report.

All results remain original-training development evidence. The original test
stays excluded, and independent assay confirmation remains open. Current
hybrid cloud allocation is zero; any later cloud fallback requires a separately
frozen operational gate and must fit at most $35 additional within the same
$100 finalization ceiling. Pause the heartbeat only after all achievable
authorized reporting is complete and all cloud jobs are stopped.
