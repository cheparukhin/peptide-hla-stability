# Presentation design handoff

Three bounded design briefs were tried on Claude Code 2.1.296 with the exact
model `claude-opus-5-5` and effort `xhigh` on 16 October 2026:

- [Website redesign](briefs/website.md)
- [Interactive research figures](briefs/figures.md)
- [90-second project film](briefs/film.md)

Each brief uses the same compact [project facts](briefs/FACTS.md), reflecting
research at commit `be0197ce`. Biological context and material scientific
qualifications remain explicit. The former October 4 film was not supplied:
its main narrative no longer describes the current results.

## Attempt and boundaries

The requests ran in three separate temporary directories outside this Git
checkout. Claude's restricted mode confined file tools to each directory.
Only Read, Write, Edit, Glob, and Grep were enabled; there was no shell,
browser, MCP, or research-repository access. Safe mode disabled unrelated
local customizations. Each request had a $3 CLI budget and a 15-minute
orchestrator timeout. No fallback model was requested.

No design files were produced. All three completion records report two
safety stops; website and film report `stop_reason: refusal`. They also
report repeated failures creating an authentication-refresh lock in
Claude's home-directory state. The requests were stopped after roughly
7.5–8 minutes. These records do not establish the specific safety category
or isolate how much the authentication problem affected completion.
The figures request's usage includes `claude-opus-5` as well as 5.5 despite
the explicit model selection. Total reported list cost was about $0.85.
See the [compact attempt record](claude-attempt.json).

This handoff narrows a legitimate design task while retaining its real
subject. It is not a prompt-obfuscation or safety-bypass procedure. No
content-stripping retries were made after the reported stops.

The temporary input packets and full local completion records were saved
under `/private/tmp/logic-binders-design-20261016/`; these may be removed by
system cleanup. A durable [input-packet archive](input-packets.zip) contains
the exact ten input files in separate website, figures, and film folders.
The briefs, facts, and compact record here are also durable.

The website and published film route were not changed. Preparatory chart
integration and video-export code were removed when no design deliverables
were returned. No Modal job or deployment was launched.
