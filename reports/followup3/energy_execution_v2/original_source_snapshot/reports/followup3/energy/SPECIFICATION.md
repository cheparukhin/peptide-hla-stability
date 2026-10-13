# Repaired FoldX complementarity, separately frozen

The pre-efficacy decision and root's THIRD_PRIMARY_AMENDMENT govern this study.
It uses the unchanged 4,449-pair original-training inverse panel and inherited
five peptide-cluster outer folds. It adds only R0+E and R5+E. The primary is
MLP R5+E minus matched MLP R5; R0 increment, the representation-by-energy-gain
interaction and ridge versions are prespecified secondary/sensitivity results.

Technical admission passed all 32 distinct-allele cases, with identical finite
repaired output schemas and unchanged native chains. Checks bind every case
artifact to the technical receipt. The complete E block comprises all 29
numeric repaired terms after excluding group identifiers. Include energy,
clash and residue-count terms; no term is chosen using assay associations.
Every new row must have the identical complete schema, finite terms and exact
native/construct/source identity. Missing, failed or interrupted rows stop the
primary assessment; no complete-case replacement or posthoc reduced schema.

Current pilot timing: repaired-process mean 188.654 seconds, median 187.009,
maximum 225.582 across 32 cases. The maximum repaired backbone RMS displacement
was 0.0 Å. These are technical observations, not biological efficacy evidence.
The proposed 600-second repaired-case timeout exceeds the observed maximum
but does not establish the unmeasured full-panel tail. Reuse all 32 technical
repaired rows after binary/structure/protocol/native/term checks, leaving 4,417
new repaired-only cases. The runtime-derived expected ordinary compute cost is
about $14.65 before cold-start/storage/reporting at recorded rates; this forecast
is not a production measurement.

Ordinary Functions have hard limits of one CPU, 2,048 MiB and 100 simultaneous
containers. No requested retries; one logical attempt per case. A fixed UTC
global deadline is supplied through source-bound root authorization, at most
five hours after launch. Every command uses the smaller of time remaining to
that deadline and the case's cumulative 600-second remaining budget. A completed
record persists before acknowledgment and can be reused. A committed started
record without completion is an interrupted attempt and is not rerun after
preemption. Failed rows remain failures. Root's independent app-stop guard is
required to bound queued starts, container startup/cleanup and lost work.

At ordinary rates, 100 concurrent containers × five hours ×
($0.04730 per CPU-hour + 2 × $0.008 per GiB-hour) gives $31.65 of bounded
compute before overhead. A $40 root reservation covers overhead and failure
allowance, subject to current tranche commitments. This is a global admission
bound, not a guarantee that every 600-second tail will complete: per-case worst
aggregate cost alone would exceed that allocation. Incomplete work stops the
comparison rather than extending a deadline or consuming an extra budget.

The source-bound inverse R0/R5 baseline features, inner vectors, choices,
predictions and ancestry are verified before exact reuse. Both new arms use
the existing FeatureHead recipe: hidden64 ReLU, MSE, Adam .001, batch256,
200 epochs, stopping-MSE patience15, two members stop0/seed0 and stop1/seed1
at inner and final levels; five cluster stopping partitions; gradient-only
StandardScaler. L2 grid 1, .01, .0001 and three inner cluster folds. Ridge
uses fitting-only StandardScaler and alpha10000,100,1. Select by pooled inner
OOF median per-allele Spearman, ≥20 varying-label eligibility and stronger
regularization on exact ties. There are 200 new neural members and 100 new
ridge fits. No baseline fit repeats or extra selector are needed.
The ridge path keeps float64 features/scaling/LSQR exactly as its inverse
baseline; MLP retains the existing FeatureHead float32 recipe. Read saved CSV
predictions with round-trip precision so reused endpoint identity stays exact.

Use shared 2,000 paired whole-peptide-cluster draws, seed20261003, primary ≥20
and sensitivity ≥50 allele panels. Report absolute scores, every increment,
interaction and conditional pointwise95% intervals. The third-primary family
amendment fixes tails1/120 and119/120 (98.333333% intervals) for any claim that
at least one of the three tranche primaries succeeds. Report failed technical
or completeness gates without removing this hypothesis from the family.

Save complete inner/final predictions, selected grids and fitting/stopping/
scaler ancestry; replay scores and choices independently. Model weights are
not serialized and no weight-level replay is claimed. Findings remain
conditional development evidence from repeatedly used training labels and
a balanced subset with a bounded two-member comparator. Equilibrium energies
do not establish dissociation kinetics, structural causality or pose validity.
No ProteinMPNN+E combination, transfer assessment, full19,716-row sweep,
checkpoint search or outcome-driven expansion follows from this protocol.

Commands after review and technical/source admission:

```
python scripts/followup3_energy.py freeze
MODAL_PROFILE=a-cheparukhin modal run modal_app/followup3_energy.py::run --protocol-path reports/followup3/energy/protocol.json --authorization-path <root-bound-authorization.json>
python scripts/followup3_energy.py collect --rows-root <downloaded-persisted-rows>
python scripts/followup3_energy.py assess --learner mlp --outer-fold 0
python scripts/followup3_energy.py assess --learner ridge --outer-fold 0
python scripts/followup3_energy.py report
```

Only root launches paid work. Root supplies the independent guard, absolute
deadline, resource reservation, completion downloads and cloud head jobs.
Workers reuse image `im-Epc3H2RdNAgJSwtXBYMCfy` and the already staged
`runs/followup3/dependencies` on the shared volume, mounted at the canonical
repository path. No fresh pip installation or binary image layer is needed.
Source digests are checked before worker imports. A shared persisted access
stop marker is checked before each new command; the client also stops dispatch
on the first returned access rejection, and root's guard stops that app.
The client exclusively records `function_app_created.json` before any pilot
upload or fanout, binding app ID, protocol and root authorization to the guard.
Idle scale-down is two seconds and startup timeout120 seconds.
