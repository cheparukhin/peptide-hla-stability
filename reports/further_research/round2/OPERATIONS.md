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

Exactly one controller owns each active scope. Each launch receipt records its
PID, command, log, start time and (where applicable) idle-sleep guard. Do not
start a second controller merely because a log is quiet. The extraction worker
captures its case log until the shard returns.

| Scope | Durable state | Behavior |
|---|---|---|
| First production shard | `extractor/full_training/controller_launch.json`, `extraction_budget_ledger.json` | Exactly 256 rows; app `ap-CTPosnxjWl1PutPuL6XHcK`; 2,700-second worker timeout and no retries. |
| Conditional cache continuation | `extractor/full_training/fanout_controller/status.json` | Waits at most 90 minutes for the first shard, verifies all hashes, features and settled reservations, and launches the existing run once at eight workers only if both measured and frozen forecasts fit $80. |
| Full sequence assessment | `sequence/controller_launch.json`, `modal_budget.jsonl`, `fold_*/completed.json`, `fold_*/repeats_completed.json` | All five outer folds plus two fixed-setting repeats; app `ap-vgwn2eHeZngfofbxH6SMsf`; $18 persistent reservation guard and 900 remote-call quota. |
| Main local pipeline | `pipeline/status.json` | Waits up to 12 hours for the complete 19,716-row cache and all sequence markers, then runs embedding folds, main report, complementarity folds and report at three local workers. No cloud spend. |
| Chat continuation | Codex heartbeat `continue-round-two-research` | Every 30 minutes; quiet while unchanged. Diagnose failures, complete remaining gated work, report meaningful milestones and commit validated results. |

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

## Failure and recovery

A failed, timed-out or interrupted cloud call retains its conservative charge.
An unresolved reservation is not free budget. First inspect the corresponding
Modal app and local controller; stop only this study's orphaned work if needed,
then reconcile against saved receipts and provider usage. Preserve failure
records. Never erase reservations or rerun an uncertain shard automatically.
The fanout controller persists launch intent before starting its child and
refuses an automatic second attempt. The local pipeline stops its own remaining
children if a stage fails.

A forecast can be wrong: the $18 sequence ledger may stop before all fits
finish. If that happens, report the incomplete procedure honestly, diagnose
runtime and decide scope with the remaining total allocation before any new
spending. Do not publish a silently truncated arm.

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
