# Independent finalization budget watchdog

`scripts/final_runtime_guard.py` monitors exactly one authorized Modal app and
controller. It does not launch apps/functions, import Modal, write reservation
journals, restart controllers, extend deadlines or automatically resume work.
All tests use mocked app commands and process signals; no real workload was
stopped while implementing this guard.

The new phase has a **$100 global ceiling**. Each authorization declares a total
allocation **including prior charges**. The current scope's ReservationBudget
ceiling is `total_allocation_usd - prior_charges_usd`. For example, a $65 total
ranker allocation and $1.35 reconciled smoke debit leave a $63.65 main ledger.
The authorization can specify 1–30 cloud containers; the current planned main
topology uses 18. Each container is exactly 2 CPU cores / 4 GiB, with the locked
authorized runtime rate `$0.1266 / 3600` per second. These are conservative
runtime commitments, not a provider invoice or remaining account-credit claim.

## Immutable authorization fields

Root writes an authorization JSON only after identifying the app and its local
controller, then binds its complete bytes to guard state using SHA256. Fields:

| Field | Required value/meaning |
|---|---|
| `schema` | `final_runtime_guard.1` |
| `phase_id`, `scope_id` | Stable phase and unique scope identifiers |
| `scope_kind` | `smoke` or `main` |
| `phase_ceiling_usd` | Exactly `100` |
| `total_allocation_usd`, `prior_charges_usd` | Inclusive allocation ≤100; finite nonnegative prior debit below allocation |
| `modal_cli` | Absolute path to the pinned Modal CLI |
| `app_id` | Exact authorized `ap-...` ID; no name lookup or wildcard |
| `controller_pid` | Exact process-group leader PID |
| `controller_command` | Exact `ps` command text, preserving internal whitespace |
| `controller_started_at` | Exact five-token `ps lstart` fingerprint, with whitespace normalized |
| `ledger_path` | Absolute read-only `ReservationBudget` JSONL journal |
| `execution_log_paths` | Nonempty list of absolute append-only scope log paths |
| `status_path`, `event_log_path`, `stop_flag_path` | Absolute durable guard outputs |
| `scope_lock_path` | Absolute lock unique to this authorization |
| `global_lock_path`, `global_status_path` | Same absolute phase-wide paths for every smoke/main scope |
| `poll_seconds` | Positive, ≤10 |
| `command_timeout_seconds` | Positive, ≤30 |
| `max_concurrency` | Positive integer ≤30, locked to the actual cloud worker bound |
| `cpu_cores`, `memory_mib` | Exactly `2`, `4096` |
| `rate_usd_per_second` | Exactly `.1266 / 3600` within floating precision |
| `call_timeout_seconds`, `per_call_overhead_seconds` | Positive timeout and nonnegative paid overhead |
| `per_call_upper_usd` | Full timeout+overhead reserve; never a mean observed call cost |
| `initial_reserve_usd` | Must equal the journal's startup/build reserve |
| `stop_commitment_usd` | Fixed stopping threshold above the initial reserve |
| `shutdown_cushion_usd` | Positive paid shutdown allowance, covering bounded command latency |
| `max_app_stop_attempts` | Integer 1–3, total across guard restarts |
| `authorized_at_utc` | Fixed explicit-UTC authorization time |
| `startup_deadline_utc` | Fixed explicit-UTC time, within five minutes of authorization |
| `deadline_utc` | Fixed explicit-UTC wall deadline after the startup deadline |
| `source_hashes`, `protocol_hashes` | Nonempty absolute-file-path→SHA256 mappings; sources must include this guard itself |
| `prior_reconciliation` | Required for main: `{ "path": absolute_file, "sha256": exact_digest }` |

Outputs, inputs and locks must be distinct even after resolving path aliases.
They cannot overwrite the authorization or frozen source/protocol files. The
wall bound is checked before launch authorization is accepted:

```text
initial_reserve
  + max_concurrency × rate × (deadline - authorized_at)
  + shutdown_cushion
  ≤ total_allocation - prior_charges

stop_commitment
  + max_concurrency × per_call_upper
  + shutdown_cushion
  ≤ total_allocation - prior_charges
```

The second inequality leaves a complete maximum-concurrency burst beyond the
stopping threshold. The cushion additionally covers the configured bounded
stop command attempts and process/provider observation latency. It must not be
substituted for per-call reservations.

## Controller handshake and stop behavior

1. Freeze source/protocol bytes. Start one controller in its own process group.
   Prepare its complete journal configuration and durable scope logs. Create
   the app context with no function calls, obtaining the exact app ID.
2. Record PID, PGID, start time and command using
   `ps -p PID -o pid=,pgid=,lstart=,command=`. PID must equal PGID. Write immutable
   authorization with fixed startup and wall deadlines.
3. Start the watchdog in an **independent process group**, so freezing the
   controller does not freeze its watchdog. Check authorization read-only with
   `--validate-only` first if useful.
4. Before the first dispatch, wait for fresh status with matching
   `authorization_sha256`, `phase="monitoring"`, `budget_verified=true`,
   observed app and verified ledger. Before **every** submission or retry,
   require a fresh healthy status and absence of `stop_flag_path`.
5. The controller must durably write and fsync its newline-terminated full
   ReservationBudget reservation before dispatching. Failed/interrupted calls
   burn their full reservation; successes settle at the declared upper cost.

Read-only validation:

```sh
.venv/bin/python scripts/final_runtime_guard.py \
  --authorization /absolute/path/authorization.json --validate-only
```

Explicitly authorized watchdog process (can stop its exact app/controller):

```sh
.venv/bin/python scripts/final_runtime_guard.py \
  --authorization /absolute/path/authorization.json --approved
```

A stop writes the submission-freeze flag and durable stop intent **before**
any process or cloud mutation. It validates PID, process group, creation time
and command immediately before each signal. It sends SIGSTOP, attempts the
exact app stop, then SIGTERM/SIGCONT only to a still-matching controller group.
An unrelated or reused PID is never signaled; the exact authorized app still
receives its bounded stop attempts. A successful CLI exit is not completion:
the app must be observed stopped/terminated and the controller absent.

The fixed wall deadline is checked before budget telemetry and provider-list
requests. Authorization deletion/editing or frozen source/protocol drift causes
stopping using the original loaded scope; it cannot enlarge the deadline or
replace the app ID. Unknown app states, failed provider telemetry, disappearing
previously observed apps, lost verified ledgers, corrupt complete records and
changed/truncated journal or log prefixes also cause stopping. An initially
unobserved app/absent ledger is tolerated only until the fixed startup deadline.
This startup allowance is never reset by polling or restarting the guard.

The guard replays complete journal records without changing them. An incomplete
trailing settlement cannot release its reservation; a new reservation cannot
dispatch before its durable newline, so an incomplete tail is ignored. Every
complete preemption warning found in the scope logs, including prefixed warnings
or one without a final newline, adds **one full per-call reserve**. Append-only
byte-prefix checks prevent log rotation or ledger rewriting from erasing these
charges. Multiple captured warnings may overcount retries conservatively.

App-stop retries have a durable attempt counter and a total maximum of three.
After exhaustion the stop flag remains and the guard exits with code 2 and
`phase="stop_failed_manual_action"`; manual app stopping and reconciliation
are required. It never creates a replacement app or automatically restores
submission. Filesystem persistence errors are reported while the guard retries
durable persistence; no mutation precedes successful stop-flag/intent writes.

## Smoke → stopped/reconciled → main

The phase-wide flock prevents simultaneous scopes even if their per-scope
paths differ. Persistent global state blocks silently replacing an unfinished
scope after its guard exits. Changing authorization bytes, including a later
deadline, cannot restart the same scope under its saved state.

Before main starts, the smoke app must be provider-observed stopped and its
controller absent, the guard must release ownership, and root must write an
immutable reconciliation file:

```json
{
  "schema": "final_scope_reconciliation.1",
  "authorization_sha256": "PREVIOUS_AUTHORIZATION_SHA256",
  "guard_status_sha256": "EXACT_STOPPED_PREVIOUS_STATUS_SHA256",
  "cumulative_prior_charges_usd": 1.35
}
```

This is an example schema, not a launch authorization or an asserted smoke
cost. The cumulative debit must cover the previous scope's final conservative
commitment plus earlier prior charges and equal the new authorization's
`prior_charges_usd`. If the previous budget cannot be verified, the guard
requires its full allocation as liability. Main gets new scope/app IDs and
fresh scope status/log/ledger/stop-flag paths, reusing the same phase lock and
global status. No automatic recovery or overlapping smoke/main calls occur.

## Mock verification

```sh
.venv/bin/python -m pytest -q tests/test_final_runtime_guard.py
```

Tests cover finite amounts, locked resource/rate bounds, full burst cushions,
partial settlement replay, failed-call full reserves, preemption counting,
fixed startup/wall deadlines, single global owner, source/authorization drift,
restart identity, same-command PID reuse, corrupt/truncated ledgers/logs,
unknown apps, stop-flag ordering, bounded retries surviving restart and the
stopped/reconciled smoke-to-main transition. No real Modal commands or signals
are invoked by this suite.
