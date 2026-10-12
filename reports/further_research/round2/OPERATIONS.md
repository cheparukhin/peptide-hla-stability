# Round-two execution operations

Updated 12 October 2026. The scientific plan and priority order remain in
[the canonical plan](../../NEXT_RESEARCH_PLAN.md). This file records execution
and recovery details, not a second research agenda. All new evidence is
**development-only**. The original test and historical numerical outputs remain
unchanged.

## Budget and ownership

The user reported $213 remaining Modal credits and authorized autonomous work.
The orchestrator fixed a $100 incremental ceiling: $80 full Boltz cache,
$18 complete sequence procedure including a $1.35 smoke/build debit, and $2
extractor engineering/overhead. This preserves $113 of the reported balance,
subject to unrelated usage and delayed provider metering. Runtime-derived
reservations are conservative accounting, not a provider invoice or a live
credit-balance API. No retry, extra arm, cohort reduction or budget expansion is
automatically authorized.

### Preemption accounting amendment, 12 October 2026

Modal infrastructure preemptions restart an input even with `retries=0`.
Successful-worker runtime therefore omits a lost attempt. The original ledgers
remain intact; the independent [runtime watchdog authorization](resources/runtime_guard/authorization.json)
adds a full per-call reservation for each exact preemption warning and stops
the affected scope before its adjusted commitments consume the remaining
maximum-concurrency burst cushion. This is an operational budget repair; no
scientific grid, source hash, model setting or evaluated row changes.

The watchdog polls every 30 seconds. Cache commitments stop at $70.808724,
leaving $9.191276 for an eight-call burst and shutdown; sequence commitments
stop at $16.779590, leaving $1.220410. Independent fixed wall deadlines are
**07:39:25 UTC / 08:39:25 London for cache** and **05:42:28 UTC / 06:42:28
London for sequence**. They conservatively charge every permitted container
continuously from the recorded start of paid work, add the prior pilot/smoke
debit, and leave a further $1/$0.50 shutdown/overlap allowance. These are latest
stop deadlines, not completion forecasts, and must not be extended automatically.

While authorized cloud work is active, `resources/runtime_guard/status.json`
must remain fresh; the launch receipt records its PID and keep-awake helper.
All authorized cloud work is now complete, and the verified watchdog shutdown
is recorded in `resources/runtime_guard/shutdown.json`. Its final status remains
a historical snapshot; do not restart it for the local-only assessment. The single-owner watchdog verifies exact
process commands and process groups, persists stop intent, freezes submissions,
stops only the two authorized app IDs, then terminates the matching local
controllers. For cache, it targets Modal child PID 16698 rather than its wrapper
PID 15111. Reservations survive a stop and require reconciliation. The watchdog
never launches replacement work. Thirteen mock-only tests cover accounting,
partial journals, ownership, process identity, immutable deadlines and failures.

The [01:43 UTC provider snapshot](resources/modal_billing_20261012T014323Z.json)
shows $10.8964 additional metered ephemeral-app usage against the pre-study
snapshot; this can lag actual activity and is not a remaining-credit balance.
Provider metering and live app state must be reconciled before any recovery.
Deadline enforcement depends on the local watcher remaining alive and the
provider accepting shutdown, so check both on every heartbeat.

References: [Modal preemption behavior](https://modal.com/docs/guide/preemption)
and [resource billing](https://modal.com/pricing).

Exactly one controller owns each active scope. Each launch receipt records its
PID, command, log, start time and (where applicable) idle-sleep guard. Do not
start a second controller merely because a log is quiet. The extraction worker
captures its case log until the shard returns.

| Scope | Durable state | Behavior |
|---|---|---|
| First production shard | `extractor/full_training/controller_launch.json`, `extraction_budget_ledger.json` | Completed 256/256 cases; app `ap-CTPosnxjWl1PutPuL6XHcK` stopped. Saved charge $0.8734; measured full-cache forecast plus 10% is $73.99. |
| Conditional cache continuation | `extractor/full_training/fanout_controller/status.json` | Pilot/hash/cost gate passed. Eight-worker app `ap-wYa6yW8KQunK4VNxPInsPd` completed all 19,716 rows/78 shards and stopped at 06:46:25 UTC. Settled runtime accounting $66.8431 against $80; zero unresolved reservations and no preemptions. See `resources/cache_completion_audit.json`. |
| Full sequence assessment | `sequence/controller_launch.json`, `modal_budget.jsonl`, `fold_*/completed.json`, `fold_*/repeats_completed.json` | Complete: five primary folds and both fixed-setting repeats passed audits (19,716 rows; 2,100 primary + 420 repeat members). App `ap-vgwn2eHeZngfofbxH6SMsf` stopped normally at 03:14 UTC; all 900 calls settled, no reservations. [Validated report](results/sequence/REPORT.md) and [cost closeout](resources/sequence_cost_closeout.json): $7.9957 conservative estimate including preemptions, against $18. |
| Main local pipeline | `pipeline/status.json` | Complete input gate passed; embedding folds 0–2 started at 06:46:38 UTC. The same controller continues all five folds, main report, complementarity folds and report at three local workers. No cloud spend. |
| Chat continuation | Codex heartbeat `continue-round-two-research` | Every 30 minutes; quiet while unchanged. Diagnose failures, complete remaining gated work, report meaningful milestones and commit validated results. |
| Independent spending watchdog | `resources/runtime_guard/status.json` | Stopped deliberately at 06:56 UTC after both cloud scopes were terminal and settled. `resources/runtime_guard/shutdown.json` preserves exact identity and shutdown evidence; no deadline or study-ledger change. |

Launch authorizations bind the exact code/protocol/source hashes. Do not edit
active sequence or extraction implementation files. An implementation repair
requires a documented amendment, a stopped/reconciled run, validation, and
explicitly compatible resumption. A new Git commit does not change file hashes.

## Reading progress safely

Read controller status, ledger totals/reservations, failure files and completion
markers before opening numerical outcomes. Cache completion must cover exactly
19,716 original-training rows. All scientific report scripts verify per-fold
markers, ancestry and source hashes before publishing. The main pipeline does
not perform transfer fitting; its completion is a milestone, not completion of
the whole plan.

Persistent artifacts and cache locations:

- Full raw interface arrays: separate Modal volume
  `pepstab-round2-boltz-embeddings`, run `round2-fulltrain-20261012`.
- Local compact feature bundles: ignored `features/round2_boltz/` paths in the
  final `extractor/full_training/feature_bundles.json` inventory.
- Per-fit local caches: ignored `runs/`; final predictions, selections, member
  counts, ancestry and hashes remain under this report directory.
- Logs, locks, transient journals and partial files are ignored by Git. Commit
  complete validated evidence; do not stage active ledgers or partial folds.

The user expects a progress check around **11 AM Europe/London on 12 October**.
The existing heartbeat includes this checkpoint and continues unfinished work
afterward. It is not a promise that every transfer fit will finish overnight.

## Failure and recovery

A failed, timed-out or interrupted cloud call retains its conservative charge.
An unresolved reservation is not free budget. First inspect the corresponding
Modal app and local controller; stop only this study's orphaned work if needed,
then reconcile against saved receipts and provider usage. Preserve failure
records. Never erase reservations or rerun an uncertain shard automatically.
Stopped Modal apps can age out of the CLI listing. The sequence [terminal-state receipt](resources/sequence_cost_closeout.json) preserves the provider-observed normal stop and zero tasks. An absent old app is not evidence of a restart. After sequence aged out, the watchdog's unchanged deadline caused redundant stop calls; the provider replied that it had already stopped at 03:14 UTC. Cache protection remained active. Both scopes are now terminal, all reservations settled and cloud controllers absent. The orchestrator verified and stopped only watchdog PID 25226 after cache completed; its keep-awake helper exited too. The [shutdown receipt](resources/runtime_guard/shutdown.json) preserves this evidence. Do not relaunch a completed cloud scope or the retired watchdog.

The fanout controller persists launch intent before starting its child and
refuses an automatic second attempt. The local pipeline stops its own remaining
children if a stage fails.

A forecast can be wrong: the $18 sequence ledger may stop before all fits
finish. If that happens, report the incomplete procedure honestly, diagnose
runtime and decide scope with the remaining total allocation before any new
spending. Do not publish a silently truncated arm.

## Cloud closeout, 06:56 UTC

The [closeout receipt](resources/cloud_closeout.json) combines $66.8431 cache
accounting, $7.9957 sequence accounting including lost-attempt reserves, and the
full $2 engineering allowance: **$76.8388 against $100**. Retaining the full
engineering allowance does not claim it was spent. No unused scope allocation
is reassigned and no new cloud work is queued. Retained-volume storage remains
separate.

The final audit found that the saved fanout wrapper's current SHA-256 does
not match `controller_sha256` in its authorization. The mismatch already
exists in the first implementation commit; no archived wrapper source matching
the recorded digest has been established. The wrapper does not validate its
own recorded digest. All four extraction source hashes and the frozen cohort,
feature protocol and cache fingerprint still match. The audit preserves both
wrapper digests as an unresolved launch-provenance limitation; it does not
rewrite the authorization or infer the exact launch-time wrapper bytes from
completed output. Strict local cache validation remains required before
acceptance; this discrepancy alone is not evidence of different extracted
features and does not justify a paid rerun.

The [provider snapshot](resources/modal_billing_20261012T065508Z.json) records
$70.2872 in named study apps and a $73.4426 monthly account increase since the
before snapshot. These differently timed metering views do not exactly
reconcile; the $3.1554 difference remains unattributed. Metering can lag, and
neither figure is a remaining-credit balance or a final invoice.

## Remaining scientific gates

Transfer implementation and execution-repeat tooling may be prepared while
main jobs run. Freeze the bounded transfer candidate set after the main screen
is complete and before any transfer outcomes. Preserve all five HLA folds and
25 joint intersections; inner selection and stopping must mirror the regime.
Report peptide, HLA and crossed dependence separately, including sparse-group
sensitivity. Repeat locked reference/selector decisions twice at identical
settings and report execution variation separately from paired seeds.

The public-data audit and simulation are complete, but no adequately powered
independent confirmation cohort or assay quote exists. External outcomes stay
reserved. Do not turn a planning simulation, historical test or sparse external
stratum into a confirmation result. Optional calibration and encoder adaptation
remain behind the core gates and are not part of automatic scope expansion.

Implementation readiness for the next continuation:

- `reproducibility/embedding_execution_protocol_draft.md` gives the two exact
  freeze/execute commands for locked embedding-selector repeats. Maximum 300
  fresh local fits; selected original-reference branches reuse the already
  budgeted sequence execution repeats. All main folds and feature/provenance
  checks must pass first. No repeat fitting has run yet.
- `transfer/assessment_protocol_draft.json` is a refreshed, unfrozen draft with
  no Boltz nominees. The reviewed implementation supports original and centred
  MLP sequence families plus one or two explicitly nominated existing R2–R5
  ridge/MLP procedures, retaining their full main regularization grids. See
  [implementation readiness](transfer/implementation_readiness.md) for the
  member recipes, tests, review and limitations. No transfer fits have run.
- The separately named `bounded_MLP_sequence_selector` chooses between original
  and centred MLP using regime-matched inner assessment. It excludes trees and
  PairLogit from the full main sequence selector. After the complete main report,
  review this comparator limitation, document a scope amendment in the canonical
  plan, then nominate and freeze before any transfer outcomes. Do not imply that
  the bounded comparisons establish superiority over the full sequence selector.
  The embedding selector includes both already fitted sequence families as
  fallback candidates, without extra fits.
- All five HLA and 25 joint cells require 10,440 sequence fits, plus 750 for each
  nominated ridge procedure or 2,340 for each neural feature procedure; repeats
  are additional. The draft currently records only the sequence component.
  Split feasibility passes all 30 cells and their deeper stopping schedules.
  Total elapsed runtime is not established by fit counts or historical profile
  extrapolations. Use timing-only telemetry from deterministic first HLA/joint
  cells to validate scheduling after the scope gate. Preserve all 30 cells and
  never nominate or discard cells using their held outcomes.

The bounded implementation passed 32 focused tests and an independent complex
review. A saved-specification integrity issue found in that review was fixed:
validation and existing-freeze reuse require the actual saved content to match
its digest and regenerated specification. Four focused agent regressions and
five independently rerun freeze/nomination checks passed afterward. The current
unfrozen draft was regenerated through the CLI and has empty nominations; a
real freeze requires one or two nominees and complete audited main evidence.
