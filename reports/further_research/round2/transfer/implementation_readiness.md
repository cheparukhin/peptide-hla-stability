# Transfer implementation preparation (12 October 2026)

**Preparation only. The assessment candidate scope remains unfrozen. No real
transfer cell was fitted, no transfer outcomes were opened, and no cloud work
was launched.** The refreshed `assessment_protocol_draft.json` contains no
Boltz nominations and records the 10,440-fit sequence component only. It replaces
the earlier unfrozen 9,600-fit draft; the earlier version remains in Git history.
The main embedding assessment and audited full-main manifest must complete
before the assessment scope can be amended and frozen.

The three transfer code files now support a bounded MLP sequence comparator
and one or two explicitly main-nominated primary Boltz representation/head
procedures. A sequence-only registry is available for draft inspection; real
freeze rejects zero nominations. Valid nominations are R2, R3, R4, or R5 with
ridge or MLP, using the complete existing main regularization grid for each
procedure. Secondary controls and altered or expanded grids are rejected.
The nomination grids are checked against the frozen main embedding protocol.
No actual nominees have been chosen.

The sequence comparison retains the original log-target MLP reference and
screens centred MLP k100 and k20. Both families use 12-member inner calls
(three stopping partitions × two encodings × seeds 0,1); the original and one
selected centred procedure each receive a 30-member final call (five stopping
partitions × two encodings × seeds 0,1,2). The separately named
`bounded_MLP_sequence_selector` selects these sequence families using pooled
inner median per-allele Spearman. Exact ties use frozen parsimony order, with
original first. It is intentionally **not the full five-family main sequence
selector**: trees and PairLogit, including their pooled joint calibration, are
outside this implementation scope. Development transfer comparisons cannot
settle superiority over that excluded comparator.

The transfer `embedding_selector` can fall back to either already fitted
sequence family, followed by the nominated embedding procedures in frozen
main parsimony order. It adds no fits. The original-only main embedding selector
is unchanged. Reports retain every component and the embedding selector against
both the original reference and `bounded_MLP_sequence_selector`, plus the
bounded sequence selector against original, on both 20- and 50-row panels.

All five HLA cells and all 25 joint intersections remain required. Inner
selection retains all three HLA or nine joint cells and uses the declared
pooled endpoint, rather than the main one-SE rule. Every gradient, stopping,
selection, scaler and prediction boundary respects the transfer regime. Joint
gradients exclude both held HLA and peptide groups and all cross quadrants.
Final neural stopping retains five paired diagonal intersections: those stops
cover only a subset of source rows. Gradient, stopping and excluded-quadrant
counts are logged for each member. This limitation is separate from the complete
25-cell joint assessment coverage.

Centring means and the shared unseen-HLA fallback use each member's gradient
labels only. Those offsets transform both gradient and stopping targets and
are restored on predictions. Neural feature heads standardize using gradient
rows only. Neural feature screening retains the main two members (stop0/seed0,
stop1/seed1 of five regime-matched partitions); final heads use five stops × six
seeds = 30. Their hidden layer, regularization and epoch settings match the
frozen main head recipe. No peptide-only stopping helper is reused.

| Fits across all 30 cells | HLA: 5 cells | Joint: 25 cells | Total |
| --- | ---: | ---: | ---: |
| Sequence procedures | 5 × 168 = 840 | 25 × 384 = 9,600 | **10,440** |
| Each nominated ridge procedure | 5 × 10 = 50 | 25 × 28 = 700 | **750** |
| Each nominated MLP procedure | 5 × 48 = 240 | 25 × 84 = 2,100 | **2,340** |

Two ridge nominees therefore total **11,940** fits; two MLP nominees total
**15,120**; one of each totals **13,530**. Execution repeats are additional.
The graph derives these totals from the actual declared candidate and member
loops, asserts the expected contributions, and refuses completion if realized
member counts differ. Shared member schedules drive fitting and report replay.
Reports also reconstruct every member identity/configuration, all row ancestry,
centring values, frozen feature hashes, inner selections and exact selector
prediction copies. They require hash-verified completion on all 30 cells before
writing pooled results; a 29-cell assessment cannot generate a result claim.

The paired peptide-cluster, HLA-group and independently crossed bootstrap
sensitivity logic is retained, including HLA multiplicity in the equal-allele
median, every-group deletion sensitivity and undefined sparse cell panels.
These intervals remain conditional on fitted predictions and do not quantify
execution variability, assay noise or independent confirmation.

Validation: the initial preparation passes **32 transfer-focused tests**
(185.86 seconds). After review, the saved-specification integrity fix passes
**four focused regressions** (1.13 seconds), including two new cases for a
complete valid ridge-to-neural grid swap under an unchanged digest and for
reusing a tampered existing freeze. The actual saved specification must match
its digest and the entire regenerated specification; rehashing altered saved
content cannot bypass regeneration equality. Compilation and whitespace checks
pass. The orchestrator independently reran the five freeze/nomination-focused
tests: **5 passed, 16 deselected in 1.14 seconds**. The complete suite was not
rerun after this isolated integrity repair. The single existing all-NaN bootstrap
warning is from the reference equivalence test for label-degenerate resamples.
Validation uses only transfer-focused synthetic tests. It covers centred-target
leakage and unseen-HLA fallback, gradient-only neural scaling, nested group
exclusions, nominee cap and exact frozen grids, pooled selection replay,
both-reference contrasts, realized fit counts and complete 30-cell reporting.
The count/label-only preflight on the frozen real training manifest passes all
30 cells, including both three- and five-partition deeper stopping schedules:
minimum gradient rows 2,352, minimum stopping rows 163, and minimum inner
eligible alleles 33. No supervised fitting was performed for this preflight.
No full repository or heavy model suite is required for this preparation.

Remaining gates: complete and audit the full main assessment; review and document
the bounded comparator limitation; explicitly nominate one or two procedures;
freeze the candidate registry, selector orders and provenance before transfer
outcomes; and validate scheduling from deterministic first-HLA/joint timing-only
telemetry. Fit counts and historical sequence-member extrapolations are not a
validated wall-time bound. No total runtime or launch approval is established
by this note.
