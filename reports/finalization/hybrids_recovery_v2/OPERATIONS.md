# Hybrid operational version three

This isolated output root preserves the frozen scientific protocol
`c5fd5501d8b640428305a1eb5b2d9fd83c3fb47139faa7de055550cb02de3955`: ten jobs and 480 scientific fits, preceded by a separate twelve-fit unscored engineering pilot. The specification, fit graph, task manifest, population and feature references are unchanged. Only the pipeline and its tests have new source hashes.

Both earlier waiting queues stopped before any hybrid fit. Their records and protocols remain historical. Operational-v2 sources and tests are preserved in `../hybrids/operational_v2_archive/`; its gate bound to failed transfer controller PID37822 must remain closed.

Version three requires both transfer interruption closures, exact durable preparation and authorization, its launched attempt/PID, all thirty audited scientific cell artifacts, the twelve new checkpoint caches, a successful durable exit and process absence. It retains 10,830 scientific fits and accounts separately for 1,785 discarded recorded fits, giving 12,615 known completed fits after success. Unknown original attempts remain separate.

The transfer deadline remains 13 October 2026 at 15:32:19.924192 UTC. Completion and successful exit must precede that deadline; later gate review is permitted. The hybrid queue and gate expiry cannot exceed 14 October 2026 at 14:15:17.567539 UTC. The pilot has a fixed 7,200-second limit and at most one worker with two numerical threads. Main fitting requires a separate runtime-only authorization after pilot success and saved-model audit, with at most two workers and two threads each. Cloud allocation is zero.

No authorization, launch gate, queue attempt or fitting process is created by this protocol freeze. Root will launch the pilot only after durable transfer closure and will record any exact authorized process ownership separately. Read `../RECOVERY_OPERATIONS.md` and `operational_amendment.json` before continuation.
