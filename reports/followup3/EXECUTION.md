# Focused follow-up studies

Started 13 October 2026. The user requested autonomous selection and execution
of worthwhile follow-up studies, Sol/Astra delegation, and updates to the
writeup and submission. They explicitly reported $139 of remaining Modal
credits and authorized using those credits. This is the **inclusive ceiling
for this new tranche**, not a spending target or a fresh grant of credits.

Root owns resource admission, launches, accounting and final delivery. Astra
owns scientific design and independent review. Separate Sol agents implement
the inverse-folding and few-shot HLA studies. All heavy extraction, fitting
and statistical resampling run on Modal; local work is editing, orchestration
and lightweight checks. No vendor messages, assay purchases or public
submission/deployment are authorized by this execution record.

## Scope under review

1. ProteinMPNN context/complementarity on a frozen, label-blind original-training
   panel, with peptide-only recovery controls and matched sequence/Boltz heads.
2. Few-shot adaptation to HLA groups excluded from source fitting, with support
   and query peptide clusters disjoint and all target labels counted.
3. FoldX repaired-structure feasibility only if access and current resource
   estimates can be established; no automatic full-cohort expansion.
4. Resolve public independent-assay provenance from metadata and publications
   while retaining external outcomes for later confirmation.

The inherited test, original validation partition and reserved external
outcomes will not select these studies or be used for their performance
assessment. Completed historical studies and their outputs remain immutable.
Every new procedure remains development work informed by prior findings.

## Resource rules

- One root-owned ledger covers all new jobs, including pilots, failures,
  storage and egress. Reserve worst-case fixed resources and wall time before
  each launch; release reservations only after provider closure.
- Use provider-enforced runtime/resource limits and a separate root monitor.
  No unlimited retries, open-ended jobs or expansion after looking at outcomes.
- Freeze input/source hashes, endpoints, fit graph and cohorts before opening
  study outcomes. Timing pilots are engineering gates, not efficacy filters.
- Reuse verified features and historical structures. No new folding cohort.
- Retain $19 as an operational cushion; planned active commitments must stay
  below $120, with the absolute new-tranche ceiling at $139.
- Record actual/provider costs when available and conservative runtime-derived
  charges separately. The user-reported balance is not a verified invoice.

## Initial inspection

The tracked working tree was clean at start (HEAD `b40c715`). Numerous unrelated
untracked historical model/log artifacts already existed and will be excluded
from commits. Modal profile `a-cheparukhin` returned no active/recent apps on
13 October at approximately 11:35 UTC; historical structure and embedding
volumes remain available. The lightweight control runtime is
`/private/tmp/pepstab-modal-control-20261012/bin/python`, Modal 1.6.1; the former
repository `.venv` is absent. New work will use separate output namespaces.

## Before-outcome expansion decision

At 12:35 UTC the manager recorded [the third-primary amendment](THIRD_PRIMARY_AMENDMENT.md)
(SHA-256 `dd09a9d31a1e5c655ec01550394c989c0207aa14fb55c83c337990234fc872c3`).
Astra's review supported an energy-complementarity hypothesis on the already
frozen 4,449-pair panel, conditional on all 32 technical cases passing and a
separately admitted bounded resource plan. At that time no new efficacy result
had been inspected by the manager; few-shot reporting had completed but its
outputs had not been downloaded/read. ProteinMPNN extraction was still active.
The manager then downloaded the few-shot and FoldX outputs. FoldX case hashes
matched the receipt and all 32 cases passed. Few-shot metrics were first read
after the amendment. No efficacy result determines energy admission, term
selection, resource settings or the addition of a third primary question.

## Energy execution recovery

The first ordinary-Function launch was stopped before new scoring because
Volume reload conflicted with native libraries loaded from that same mount.
Provider closure, 32 reused records and zero attempt markers are recorded in
[the failure receipt](resources/energy_extract/failure.json). Its conservative
closed allowance is $3.248455, including $3 overhead; this is not a provider
invoice. The unchanged scientific source is preserved in commit `bc625c5`.

The reviewed [operational amendment](energy_execution_v2/EXECUTION_AMENDMENT.md)
puts pinned native libraries in the image while retaining documented Volume
reload/commit semantics. A predetermined fresh case must pass the exact
production path before fanout, and its feature is reused. Original panel,
descriptor schema, learners, baselines and inference contract are unchanged.
The new protocol hash is
`03808264b86e217945bdde190d77d71699f46c6818b1b98cf267ccec95ac0315`.
A fresh $40 reservation covers the small image build, pilot and remaining work
under one fixed deadline, 18:04:45 UTC. The independent guard was verified live
before dispatch to app `ap-nvIT2aCnYh2xc3Ea32CotH`. The old app is stopped.

## Exploratory few-shot attribution diagnostic

After the positive primary few-shot result was inspected, root and Astra
authorized one narrow diagnostic of its interpretation: R5 at 20 support
labels, comparing the saved full adapter with a saved-offset-only correction
and the unchanged zero-shot prediction. It adds no fitting, target labels or
primary hypothesis. The frozen queries, 68-allele panel, three support draws
and original 2,000 crossed resamples remain unchanged. Two nominal paired
contrasts must telescope exactly to the original gain; within-cell ranks must
be invariant to the scalar offset.

[The specification](fewshot_offset/SPECIFICATION.md) explicitly discloses the
post-hoc timing. Astra approved the source and ten synthetic tests passed
before freezing protocol
`65da2dec138effa7572cd32242b77b45dec5870b762d0ed62a5c512b527d6dee`.
Commit `2fe57b8` preserves the pre-control-results implementation. Root admitted
a single 2-CPU/8-GiB Modal report with a one-hour hard limit and $1.50 maximum
reservation; no new control metrics were calculated locally before launch.

## ProteinMPNN deadline and parser diagnosis

The original extraction stopped at its fixed deadline on 13 October,
14:22:25 UTC; provider termination was confirmed at 14:22:38. The conservative
closed charge is $7.067744 including the $1 overhead allowance. The downloaded
checkpoint contains 4,163 successful records, one failed record and 285
unwritten panel identities. No supervised inverse assessment had run. The
planned continuation was held before admission when its complete-cache audit
detected the failure; no incomplete-panel assessment was permitted.

The failed identity is HLA-B*56:01 / YPAEITLTW. Independent source review found
that the original loader selects the first chain containing the peptide as a
substring, then rejects it unless that entire chain is the peptide. A peptide
also present inside the HLA chain therefore triggers a false rejection before
the actual peptide chain is considered. The existing length/offset guard fails
closed: it prevents accepting the long-chain match. A narrow, separately bound
whole-chain identification amendment was reviewed and frozen before further scoring.
The failed attempt is retained; no pair, coordinate, native sequence, decoding
order or comparison is replaced. FoldX uses a separate, explicit chain-boundary
validation and is unaffected by this specific loader defect.

The reviewed recovery is preserved in commit `a32c1e5`; its standalone
[manifest](inverse_parser_recovery/manifest.json) has identity
`e4da04548340803b3777139f459c4a7268e8cee186468deecabd5853565cc5c6`.
The one-hour continuation was admitted with a $4.50 cap. Both affected
YPAEITLTW complexes passed cloud parsing before any scoring. It completed in
552 seconds with a conservative $1.465857 charge including overhead. All
4,449 records now pass the unchanged acceptance rules. The separately saved
[recovery receipt](inverse_execution_v2/parser_recovery_receipt.json) binds the
4,163 unchanged successful records, the exact archived failure, 286 corrected
or missing cases and final aggregate scores. [Head admission](inverse_execution_v2/HEAD_ADMISSION_AUDIT.json)
independently checked these hashes and full coverage before supervised fitting.

All ten learner/fold commands subsequently completed the frozen 900-fit graph.
Their provider phase lasted 79 seconds ($1.066922 including overhead). The
cloud report completed in 68 seconds ($0.509007 including overhead), with all
declared comparisons and shared resamples. These phase allowances are not
provider invoices. Independent completed-output review subsequently passed
all 900 fits and 28 comparisons. The current three-primary family intervals
are derived from the saved resamples; the historical two-primary intervals
remain explicitly superseded.

## Energy preemption diagnosis

The v2 Function app stopped on 13 October at approximately 16:08 UTC with
4,376 successful statuses and 73 `interrupted_attempt` statuses. Its independent
guard confirmed provider termination before the fixed global deadline. The
conservative closed allowance is $22.344642, including $3 overhead and assuming
maximum 100-container occupancy throughout; it is not an invoice. The tranche
total conservatively closed is now $43.523274, with no live reservation.

Bounded provider system logs explicitly report worker preemptions followed by
same-input restarts. Existing start markers correctly prevented automatic
recomputation. No energy efficacy was calculated. Root and the independent
reviewer approved preparation of one manually admitted operational recovery
of the unresolved identities only. It must preserve every successful record
and original start marker, use the same frozen coordinates, binary, repair
settings and descriptor schema, and complete the original panel before any
fitting. A separate amendment, exact source-bound manifest and fresh resource
reservation are required before launch; no complete-case fallback is allowed.

The final [source review](energy_recovery_v3/PRELAUNCH_REVIEW.json) passed six
synthetic checks and verified all 73 native structure/metadata hash pairs
against the independently audited inverse cache. Commit `20c3d19c` preserves
the reviewed implementation. Root froze recovery manifest
`d2aa04836f99311851700772b63e583dd11cb2752bccfdbf4bc7c37f5ab564fd`
and admitted a separate $3.50 phase. Sandbox
`sb-01M4V17YJGZWCVSR1478SKWA59` launched at 16:24 UTC with aggregate hard
16-CPU/32-GiB limits, 16 spawned single-thread workers and deadline 17:08:22 UTC.
No per-process memory cgroup is claimed. Recovery records and the verified
combined collection are written separately from the original partial run.

Recovery completed in 1,093 seconds; provider termination with return code 0
was confirmed at 16:42:13 UTC. Its conservative allowance is $1.422684,
including overhead. All 73 recovered cases passed. The
[independent recovery audit](energy_recovery_v3/independent_recovery_audit.json)
verified 4,376 unchanged successful rows, 73 unchanged original starts, every
new native-input/attempt record and all 4,449 combined rows. No efficacy labels
or fitted predictions entered that audit. Root then admitted the frozen cloud
collector under a separate $0.75 cap; matrix verification precedes head fitting.

## Full-cohort option authorized by the user

During execution the user explicitly permitted expansion to all complexes if
the initial study is scientifically promising. The existing $139 inclusive
ceiling remains unchanged. This authorizes a decision after reviewing the
frozen results, rather than an automatic expansion or a spending target.
Any additional assessment must use the eligible original-training cohort,
reuse accepted extraction records, and have a new prospective resource and
analysis specification. It would remain development evidence informed by
these results, not independent confirmation. Original validation/test and
reserved external outcomes remain outside the authorized research assessment.

## Completed energy comparison and one selected extension

The complete 4,449×29 energy matrix passed independent value-by-value review
against the accepted raw records. All ten new assessment commands completed
200 MLP members and 100 ridge fits, reusing the exact prior R0/R5 baselines.
The heads phase lasted 128 seconds ($0.608186 conservative allowance); the
cloud report lasted 83 seconds ($0.510931). The independent final audit passed
all 12 contrasts and their saved-draw uncertainty calculations, with recovery
and feature audits bound into the execution-plan chain. Fitted-weight replay
is not claimed. The primary neural increment is +0.014991 with family interval
[-0.052436, 0.045026]; the ridge increment is also inconclusive.

The [root decision](FULL_COHORT_DECISION.json), made after both completed
descriptor audits, selects one final full-training-cohort ProteinMPNN study
and declines the more expensive full FoldX extension. The modest neural signal
on the capped panel merits a bounded coverage/fitting-density check; its
uncertainty and ridge disagreement remain material. This is selected
development evidence, never independent confirmation or a fourth primary.
No further descriptor combination or tuning search follows its outcomes.

Commit `9124db99` preserves the reviewed four-batch resource amendment and the
earlier unexecuted two-batch proposal. Commit `4b78bd66` freezes the selected
scope and decision. Four 16-CPU/32-GiB Sandboxes launched at 17:02 UTC, each
running eight scoring workers with two numerical threads. They partition the
15,267 new cases into 3,817/3,817/3,817/3,816 disjoint identities; the 4,449
completed native scores remain byte-exact cached inputs. Each job has a
$8.50 reservation and fixed deadline 19:30:55 UTC. The $34 extraction envelope
plus $6 for staging/fitting/reporting bounds the full extension at $40. With
$46.571781 conservatively closed before this extension, the maximum planned
commitment is $86.571781, below the $120 planning limit and $139 user ceiling.
These are resource bounds, not invoices or guaranteed completion times.

The complete existing Boltz feature cache is staged separately under a $0.75
cap while native scoring proceeds. All 19,716 scores must pass before the
frozen 300-MLP/150-ridge fit graph is admitted. Original validation/test and
reserved external outcomes remain untouched.
