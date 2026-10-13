# Scientific review of the third follow-up tranche

13 October 2026. Design review only; no new outcome has been inspected for this
review. The [canonical plan](../NEXT_RESEARCH_PLAN.md) remains the status index;
the [shared protocol](PROTOCOL.md) specifies this newly authorized scope.

The user subsequently authorized a **$139 ceiling for this new tranche**.
It replaces the review's initial assumption that this work shared the prior
finalization ceiling. Historical $11.609382 accounting/reservation is separate;
credits are a ceiling, never a spending target. Root owns all spending decisions.

## Studies worth doing

| Study | Question not answered by completed work | Minimal informative comparison |
|---|---|---|
| ProteinMPNN | Does inverse-folding compatibility add half-life signal beyond cached Boltz features and peptide recovery? | Matched two-member R5 MLP plus peptide-only and full-complex ProteinMPNN positions versus R5 MLP plus peptide-only positions, on a frozen balanced original-training panel; matched ridge sensitivity |
| Few-shot adaptation | Can a small number of target-HLA measurements improve ranking for new peptide families? | Fixed residual adaptation of R5 at 20 target labels per allele versus its matched zero-shot source predictor, with R0 and a bounded historical sequence-recipe anchor |
| Independent-data metadata | Can already identified reserved candidates support a coherent confirmation cohort? | Resolve publication, protocol, construct and data-lineage metadata; preserve reserved outcomes |
| Repaired FoldX pilot | Does repaired energy extraction have a reproducible, affordable implementation? | At most 32 label-blind training structures, paired unrepaired/repaired extraction and runtime/structural checks; no supervised claim |

The first two warrant bounded end-to-end development assessments. The metadata
audit remains the most direct step toward stronger evidence. FoldX is optional
and last: it closes an implementation uncertainty, not the half-life hypothesis.
It must not crowd out the two informative comparisons or reserve for reporting.

## Evidence and interpretation corrections

The [ProteinMPNN pilot](../stage5_inverse_folding.md) contains 90 folds but only
five underlying complexes. The 2,000-fold sample characterized scores and the
failure of a pilot QC threshold to transport. Neither fit or assessed a
half-life predictor. Their results cannot reject incremental predictive value.
The production QC sample also spans original partitions, so its reuse requires
an explicit original-training identity filter before any target access.

The [completed label-budget study](../finalization/label_efficiency/REPORT.md)
subsampled source fitting clusters. It did not give a held-out HLA a small support
set. The adaptation experiment is distinct only if source fitting, support and
query roles are explicit, peptide families cannot cross those roles, and target
label use includes tuning and stopping. The proposed closed-form adapter needs
neither target tuning nor stopping. A target-specific intercept alone cannot
improve within-allele Spearman; feature-dependent adaptation is necessary.

The [historical FoldX document](../stage8_foldx.md) contains an invalid argument
that observed affinity correlation bounds the performance of every computed
energy feature. Different errors and complementary information prevent that
conclusion. Conversely, predicted-structure energy and kinetic half-life are
different quantities, and a small successful extraction pilot cannot establish
predictive utility. Retain both qualifications without repeating an energy sweep.

The latest ranking and hybrid comparisons did not establish further gain.
Repeating those grids, adding seeds until a favorable result appears, or
enlarging them after looking at follow-up outcomes adds little trustworthy
evidence. Existing main/transfer findings are development-selected; additional
within-training folds do not create independent confirmation.

## Review gates and reporting requirements

Freeze identities, exclusions, support lists, candidates, primary contrasts,
source hashes and resource ceilings before outcomes. Technical pilots may
gate on valid extraction, completeness, memory and runtime only. A failed
pilot is a valid outcome; it does not authorize substitution or repeated search.

Each assessed comparison must have shared rows, matched label access and
recorded ancestry. Save absolute scores and every frozen secondary comparison,
not only a winning difference. Average support-draw **scores**, never predictions.
Use actual peptide/HLA groups for conditional uncertainty, not five folds as
five independent biological units. Maintain fixed eligible panels across arms
and budgets. A simple comparator is a matched recipe comparison, not the main
competitive selector.

Assess ProteinMPNN's added full-context block against peptide-only compatibility
as the primary comparison, and against R5 alone as an essential secondary
comparison. This distinguishes context complementarity from an arbitrary extra
sequence-recovery feature. The peptide-only backbone still came from a complex
predicted using the native sequence and HLA: it is not an unconditional sequence
model, and the control does not isolate a causal structural mechanism.

The few-shot primary is R5 adaptation gain at 20 labels. The representation-by-
adaptation interaction is secondary: a useful representation may already be
strong zero-shot and consequently show a smaller adaptation gain. Show absolute
scores so a difference-in-gains cannot obscure that distinction.

There are two primary development questions in this tranche. Report individual
95% conditional intervals and, for any joint statement that either new study
succeeded, also provide 97.5% per-comparison intervals (Bonferroni family control).
Do not elevate a nominal exploratory secondary result into the headline. Neither
study retrospectively passes the main report's separate +0.05 criterion.

Before release, independently check exact feature/target alignment, fold and
support exclusions, saved predictions and inner selection, all declared arms
and budgets, comparison arithmetic, and cloud/resource closure. ProteinMPNN's
reused FeatureHead does not save fitted weights; its prediction/selection/
ancestry audit must not be described as fitted-weight replay. A protocol document alone
does not certify implementation or completion.

## Initial implementation review

The initial code review checked the minimal cloud runner against installed
Modal 1.6.1, including hard CPU/memory maxima, provider lifetime, deadline,
process-group cleanup and exclusive launch intents. Two findings were repaired:
the ledger now reconciles individual reservations with its aggregate and checks
coverage of the exact plan; runtime accounting now starts at launch intent,
before Sandbox creation. Root still owns immutable source uploads, manual
admission, monitoring, provider reconciliation and temporary-storage closure.

An independent lightweight run of the inverse/few-shot synthetic suites passed
**14 tests**. These cover label-blind support selection, cross-allele family
exclusions, centered adapter equivalence to ordinary ridge, shared uncertainty
draws, identical-arm zero differences, explicit common peptide decoding orders,
and bounded fitting graphs. No canonical models were fitted locally.

The initial few-shot source review found the source/support/query separation,
source-only scaling, exact target budgets, crossed HLA multiplicity and
score-level support-draw averaging consistent with the protocol. Its frozen
metadata has 68 target alleles and 9,180 tiny adapter solves in addition to
450 source fits. Target labels are reused across the five query cells and are
counted once per experimental draw/budget. This is an implementation review,
not certification of completed cloud outputs.

The final prelaunch source review verified all three inverse-folding repairs:
the canonical target adapter and its input files are pinned; inner OOF vectors
and fitting IDs are saved for independent score/selection recomputation; and
the report includes the interaction and 97.5% family interval. The main
extraction reuses the five same-protocol pilot rows. Source ancestry audits
check held/gradient/stopping exclusions and scaler ownership. The repeated
lightweight synthetic run passed **14 tests in 4.22 seconds**.

Both scientific implementations are suitable for source-bound freeze and
bounded execution after Root's resource admission. The inverse label-free
pilot must still establish real parser/model compatibility and runtime. This
review does not certify cloud completion, measured costs or numerical results;
those require the subsequent output audit. The cloud supervisor additionally
verifies named remote artifacts and enables the explicit cloud-execution guard.
