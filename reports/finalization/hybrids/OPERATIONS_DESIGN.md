# Hybrid experiment operations design

Prepared 2026-10-12T13:47:10.781785+00:00 from local source and operational records. This document is a feasibility and queue design; it does not authorize fitting, app creation, dispatch, stopping, retry, or changes to an existing study. Astra owns the bounded scientific protocol, candidates, split ancestry, fit graph and reuse decisions.

## Active inventory and fixed limits

The existing transfer controller is PID **89292**, watchdog PID **89617**, and app **ap-RrFfbDvriD4rehaba5x7CA**. Its frozen scope remains exactly 10,830 scientific fits, including 1,620 cloud members, three local workers with two numerical threads each, and at most 18 cloud containers. Its cumulative allocation is **$65**, with a fixed 26-hour deadline of **2026-10-13 15:32:19.924192 UTC**. Preserve its sources, authorizations, app, journal, deadlines and artifact namespace.

The local guard snapshot at `2026-10-12T13:47:07.782278+00:00` reports phase `monitoring`, controller `running`, app `ephemeral`, app_stopped `False`, and cumulative commitment `$2.750589399`. This is a changing operational snapshot, not final settlement, spare credit, or a new launch gate. No provider or process-control API was called for this inventory.

The overall new-phase ceiling remains **$100**. Reserve at most **$35 additional for hybrids**, including their cloud pilot, engineering diagnostics, builds, shutdown cushions, failed or lost attempts and main fits. The $65 transfer allocation remains intact while it runs, regardless of its currently low recorded usage. Do not infer a larger hybrid allowance from an incomplete transfer ledger or user-reported balance.

Let `P` be the conservatively reconciled cumulative prior charge after transfer and `H` the complete hybrid incremental allocation. Require `0 <= H <= 35` and `P + H <= 100`. If transfer accounting cannot be verified, carry its full authorized cumulative $65 until root resolves the liability. Hybrid pilot and main are parts of the same `H`; main cannot add another $35 after the pilot.

## Exact queue gates

1. **Prepare while transfer runs.** Author only new hybrid modules, synthetic tests, source inventories, candidate/count manifests and ancestry preflight checks. Do not perform real hybrid fits, cloud pilots, feature extraction or model-dependent timing. Cheap inspection of cached metadata is permitted. Preparation must leave all files hashed by the active transfer protocol and guard unchanged.
2. **Freeze scientific scope before outcomes.** Astra specifies every hybrid, baseline, selector/blend recipe, regularization grid, member/stopping schedule, calibration/stacking inputs, split regimes, uncertainty and fit counts. Freeze the new protocol, source hashes, cached feature identities and permitted reuse ancestry. A favorable old or new outcome cannot silently change the grid or trigger additional scope.
3. **Close the predecessor operationally.** Require the transfer controller exit receipt and cloud driver exit receipt, hash-bound to their existing protocol and authorization; all requested cell completion hashes and member counts must pass the established audit. Verify controller PID 89292 and any child fitting processes are absent. Require watchdog status `app_stopped=true`, `controller_state=done`, and finished guard ownership; confirm PID 89617 is absent and the shared global guard lock has been released. An app-close intent or old launch receipt alone is insufficient.
4. **Reconcile liabilities.** Save an immutable `final_scope_reconciliation.1` document binding the predecessor authorization SHA256, final guard-status SHA256 and conservative cumulative prior charge. Require its charge to be at least the guard's verified cumulative effective commitment; if its budget was not verified, carry the full prior scope limit as the existing watchdog requires. Match accounting for provider preemption/lost attempts and all outstanding reservations. The predecessor global phase identity and paths must be retained for any later paid transition.
5. **Authorize queued local fitting separately.** Root records the new protocol/source hashes, exact task axes, fit count, predecessor receipt hashes, verified process absence, workers 1–3, two threads each, fixed local deadline, owned process identity and immutable attempt/exit paths. A queue must wake only after gates 3–4 are satisfied; it must not poll its way into an unapproved run. A failed predecessor needs explicit root reconciliation and disposition, never automatic retry or a successful receipt fabricated from process absence. If predecessor artifacts are incomplete, no reuse-dependent task can run until the new scientific protocol explicitly addresses that missing dependency.
6. **Launch only the approved next scope.** One owner acquires the new local-controller lock and creates an exclusive durable attempt/launch intent before any fitting or process launch. Keep the global three-local-worker cap across hybrid jobs, packaging and any remaining local work. Set numerical limits once around the whole controller. Freeze pending local work at the first failed/uncertain task; retain successful receipts and all ownership files. A separate process-level watchdog must enforce a hard local deadline because in-process fitting threads cannot be killed individually.

These gates favor a completed, audited transfer predecessor. Root may authorize a new study after an explicitly reconciled failed predecessor, but that requires a distinct documented disposition and validated scientific dependencies; it is not the current successful-prerequisite path.

## Initial local implementation

A local-only implementation is feasible for the proposed tree/MLP/ridge heads and clean OOF blending because the original-training sequence and Boltz representations are already cached and CatBoost/scikit-learn/neural CPU dependencies exist. It needs no new Boltz extraction, GPU, cloud deployment or simulated features. Its actual duration and memory remain unmeasured until a separately frozen engineering timing gate after transfer has released the local worker cap.

Use independent new modules, proposed names `pepstab/final_hybrids.py`, `scripts/final_hybrids.py`, `scripts/final_hybrids_pipeline.py`, `scripts/final_hybrids_report.py` and `tests/test_final_hybrids*.py`. These names are an implementation recommendation, not created runners. Keep outputs under `reports/finalization/hybrids/`; future resource records belong under new `resources/hybrid_*` directories. Do not import and launch the existing transfer app or use its member worker, fixed candidate registry, controller attempt, paid journal or live task queue.

Prefer a new controller with an injected prediction backend and declarative task graph. The default backend is local. Use one owned root process, up to three task workers, two numerical threads each, exclusive per-task intent, durable member/progress/completion receipts, exact row/feature/scaler/target hashes and no automatic retry. Record individual member predictions so raw inner/final aggregates, selected candidates, maps and hybrids can be replayed without refitting. Tasks must reject held-out labels. Preserve the original test exclusion.

Any reuse requires exact population, regime, outer/inner fit, stopping, seed, candidate, feature, transform and scaler identities. Five peptide-fold, HLA and joint predictions are not interchangeable simply because pair IDs coincide. Previously selected outer predictions are not clean inner predictions for learning a blend or stacking head. If required clean inner outputs are absent, count and freeze the needed fits; do not reconstruct their values from scores. Pure deterministic arithmetic on already audited predictions can add zero model fits, while fitted blending/calibration must have its own clean ancestry and explicitly counted dependencies.

This approach can implement all Astra-approved hybrids locally, including PairLogit if runtime permits. A cloud backend is an optional operational alternative for frozen tasks, not a reason to shrink or expand the scientific set based on outcomes. Engineering timing predictions are never scored or automatically reused as assessment fits.

## Existing API compatibility

| Existing component | Read-only capability and constraint | Recommended use |
|---|---|---|
| `scripts.final_transfer_pipeline.run_controller(out, approved, cells, workers, ranker_backend=...)` | Hard-codes the 30 transfer cells, old protocol, controller schema and transfer output paths | Copy the durable design into a new hybrid controller; do not call it with hybrid tasks |
| `scripts.final_transfer_cloud.load_launch_plan` / `validate_guard_binding` | Hard-codes two pilot calls or 1,620 main calls, transfer source/protocol and 3×6 main topology | Implement a new source-bound hybrid launch-plan schema and binding checks |
| `modal_app.final_transfer.ranker_member` | Builds positional sequence onehot and accepts only fixed depth8 transfer ranker candidate | New hybrid worker/module required; preserve this worker and app |
| `pepstab.final_modal_backend.GuardedBatchBackend(function, budget, authorization_path, receipts, max_batch=...)` | Callable takes an ordered list of tasks; requires `task_id`, `candidate`, `seed`, `regime`, gradient/stopping/held rows; checks guard every batch; fixed 1,000-second timeout, 62-second overhead, 2CPU/4GiB rate | Reuse unchanged only if the new worker matches every contract and locked resource/rate assumption; otherwise create a new tested backend module |
| `pepstab.round2_budget.ReservationBudget` | One locked journal, complete batch reservation before dispatch, failed/uncertain calls burned at their full reservation | Reuse with a fresh hybrid journal; never reopen the live transfer journal or invoke its automatic interrupted-ledger settlement during inventory |
| `scripts.final_runtime_guard.RuntimeGuard` | Supports only scope kinds `smoke` and `main`; locks the shared global phase; transitions require a stopped/reconciled predecessor; resource shape fixed at 2CPU/4GiB and rate .1266/hour | Reuse unchanged through new hybrid-prefixed scope IDs, with scope_kind `smoke` for pilot and `main` for assessment |

The watchdog transition does **not** require that the predecessor was a smoke scope. Its `_validate_transition` checks the shared previous global record, stopped app, done controller, distinct scope/app identities and a reconciliation that conservatively carries liability. Thus `transfer-main → hybrid-smoke → hybrid-main` is structurally supported without editing the running watchdog, provided the same phase/global lock/status paths are used. Do not introduce a `hybrid` scope_kind or create another global phase path to bypass the existing lock.

If fresh rates, resource shape, timeout accounting or required transition semantics differ from the locked guard contract, preserve that module and develop a separate tested guard version with an explicit phase transition after predecessor closure. Importing/copying a design is allowed preparation; launching another active guard against the current scope is not.

## Future cloud pilot and main gates

Cloud work remains queued until the transfer predecessor is settled and the scientific task/count/source freeze is approved. Obtain fresh rate/provider state then save a new launch plan and exact task manifest. A paid pilot must have a small explicit call quota, separate build reserve, wall deadline and full worst-case reservation, within `H <= 35`. If parity requires a local synthetic fit, count that engineering fit separately and schedule it under the local cap. Measure a representative expensive training-only task for each materially different worker family; no held-out scoring and no automatic reuse as scientific assessment fits.

One new root-owned app context must write its exact app ID immediately and wait for the healthy independent guard before any remote call. Bind exact app, controller PID/process fingerprint, shared phase identity, source/protocol hashes, scope kind, resource shape, provider hard concurrency, timeout/overhead rates, journal ceiling and deadline. Use fresh scope-specific status, events, stop flag, receipts and app IDs. The global guard lock must forbid parallel paid scopes. Call identities must hash the full explicit candidate/config, ordered gradient/stopping/held identities, labels only for supervised fitting/stopping rows, feature-array/bundle hashes, seeds and representation. The worker returns exact IDs, predictions, model config/stopping information, versions, wall time and memory; reject incomplete or reordered results.

After pilot closure, verify the app stopped, controller exited, reservations reconciled and guard released ownership, then carry its charge into the hybrid-main predecessor reconciliation. Main launch binds the frozen complete call count and fit graph, representative pilot receipt hashes, local timing, the measured wall/cost forecast and a fixed shutdown-cushioned deadline. Keep no retries, no scope expansion and no deadline extensions. Stop and preserve the incomplete attempt if the budget/deadline/guard is exhausted.

At the currently locked resource rate, one 1,000+62-second call reserves **$0.037347**. Use this only as a planning reference; pilot plus all main calls, builds, shutdown and lost-attempt accounting must fit the hybrid incremental allocation. For allocation `H`, available main budget after pilot/build/shutdown commitments `R` is `H-R`; main batch reservations require `N × call_upper <= H-R` for a full timeout-bound count guarantee. Independently require `initial_build + C × .1266 × wall_hours + shutdown <= hybrid_current_scope_budget`, where `C` is the actual provider hard container cap. Preemption can escape successful-call telemetry, so the continuous resource/deadline guard is mandatory even when call reservations fit.

## Evidence inventory

Source/protocol/immutable authorization hashes read for this design:

| Path | SHA256 |
|---|---|
| `scripts/final_transfer_cloud.py` | `f488562a2cc3ba1583a7fbd1adac137552917fe19782a3c2e918e7f64f32c458` |
| `scripts/final_transfer_pipeline.py` | `e98e5e1953d52745fcca5da7cdd164275ed27500575bdb8d56638a04facc31ff` |
| `scripts/final_runtime_guard.py` | `ea05ea977fab7ec274942d71b521e931117912408b7150ad9d0fb5203e1a6627` |
| `pepstab/final_modal_backend.py` | `98c08da2b52c35c7d891e6126df8f5da3cd072f117dcc4b999f3de6216d5c3b3` |
| `pepstab/round2_budget.py` | `57f54e788e24093bafe1e97c5e7ec63d0e200114283bb9489161733fccea4d91` |
| `modal_app/final_transfer.py` | `8603bf6989aa92a553df751aba28673cfc424efca3979205c978126c100e48c4` |
| `reports/finalization/transfer_controls/protocol.json` | `3f4c0eacd46a432a66ed81d453dbae4d96bdedce2dc957cd0701889bd93bef23` |
| `reports/finalization/resources/transfer_main/guard_authorization.json` | `79423a9d60e8559049ac50cb3263fb1d898c1c230e21e925f330088c55f41e45` |
| `reports/finalization/resources/transfer_main/controller_authorization.json` | `72fe8656cad7770018a4d293ef03e859450ca29ca82cf01b2de7954331293b74` |

The active guard status and execution manifest were used only as changing inventory snapshots. Their values are not frozen into a launch approval. Before the queue opens, root must capture and hash the final predecessor receipts, final guard status, provider reconciliation and process-absence evidence. This design added no fits, provider calls, process stops, retries or source changes.
