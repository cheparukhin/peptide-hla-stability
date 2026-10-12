# Hybrid operational version two

This directory is the current output root for the unchanged480-fit hybrid
assessment and separate12-fit unscored pilot. The original interrupted queue
and operational source snapshots remain under `../hybrids/`.

Read [recovery operations](../RECOVERY_OPERATIONS.md) before acting. The pilot
waiting queue is PID44698/helper44699; its fit gate is absent until the exact
local transfer recovery PID37822 completes all30 cells, passes audit and exits.
Never launch a duplicate controller. The pilot authorization is under
`../resources/hybrid_recovery_pilot/`; its queue expiry remains14 October
14:15:17.567539 UTC and its fitting limit remains two hours. Cloud allocation
is zero. A separate runtime-only authorization is required for the main phase.

The scientific protocol hash remains
`c5fd5501d8b640428305a1eb5b2d9fd83c3fb47139faa7de055550cb02de3955`.
The source mapping differs only for the pipeline and its tests. See
`operational_amendment.json` and `protocol.json` for exact provenance.
