# Project review and prioritised research plan

Reviewed and reconciled: **12 October 2026**. Execution updated: **12 October 2026**. Status: **bounded round-two development execution complete: evaluation repair, extraction, sequence/representation screens, complementarity, locked execution repeats and all 30 transfer cells with full reporting; independent confirmation remains open**. The original review is preserved below; §12 records subsequent execution, scope amendments and outstanding gates. The original test has not been rescored.

This is the **single current source of truth for project status, research priorities and execution gates**. It consolidates the original follow-up plan, completed Phase A–D findings, the [8 October Boltz strategy](archive/plans/boltz-strategy-2026-10-08.md), and the remaining hackathon proposals. Superseded proposals are in the [plan archive](archive/plans/README.md). Stage reports, saved tables and dated prespecifications remain evidence about completed work; they are not additional work queues. The frozen [original evaluation contract](../EVALUATION.md) remains unchanged. Section 11 records the reconciliation decisions.

**Current decision after execution:** frozen Boltz procedures improve the specified development comparisons, including the bounded HLA/joint transfer comparisons below. The main peptide comparison does not establish the original 0.05 worthwhile-gain bar; transfer exceeds that bar only for its narrower, conditional comparison. Independent half-life confirmation is the outstanding evidence gate. Keep calibration/adaptation separate; no additional fitting or cloud scope is authorized automatically. No assay noise ceiling is established. The original sequencing rationale below is retained as the study design history.

Maintain priorities and completion status here only. New experiment protocols may specify a milestone's implementation, but must link back here and cannot silently change its scope, comparator or gate. Record amendments and their evidence in this document; archive superseded proposals instead of keeping parallel plans.

The archived Boltz strategy records a report that winning teams used frozen Boltz embeddings; this motivates the representation priority. Their exact extraction point, construct, pooling, head and split protocol have not yet been independently inspected. Record those details if they become available; until then, describe the work as testing the frozen trunk-embedding hypothesis, not reproducing the winning pipeline.

## 1. Objective and order

Determine whether frozen Boltz-2 representations improve peptide–HLA half-life ranking, whether preserving peptide–groove position information adds value, and where any gain generalises. Compare with both the original sequence recipe and a competitive new sequence control. Treat assay-floor calibration as a separate, lower-priority objective.

| Order | Milestone | Required output | Gate |
|---|---|---|---|
| 0 | Repair evaluation; start independent-data feasibility | Tested nested assessment, ancestry contract, corrections; data/access/assay inventory | Assessed labels cannot influence fitting, selection or stacking; record confirmation feasibility separately |
| 1 | Verify embedding extraction; establish sequence controls after milestone 0 | Training-panel equivalence checks, measured throughput/storage, corrected A0/A1/A4 and learner × target controls | Correct tensors and indexing; priced cache plan; credible sequence comparators |
| 2 | Run the bounded representation screen | All declared arms, nested predictions, cost and coverage | Compare representations with both sequence references; no selection on outer outcomes inside a claimed nested procedure |
| 3 | Test positional/interface information and complementarity | Matched ablations and clean blends/residuals | Identify whether the added information earns its cost |
| 4 | Assess transfer and execution reproducibility | Separate peptide/HLA/joint holdout results and repeats | Claims limited to the regimes actually supported |
| 5 | Freeze and independently confirm | Locked shortlist, reserved measurements and analysis | New performance claims rest on independent evidence |

Extractor engineering and independent-data feasibility can begin while the evaluation is repaired. No supervised assessment should proceed through the old selection or blending paths.

**First execution tranche:** milestone 0, the small extractor verification/profile in milestone 1, and the data inventory in milestone 5. After the evaluation gate, run corrected sequence controls and their execution repeats while preparing the embedding cache. Stop for a documented scope decision if extraction is infeasible; continue the sequence/data programme. A4 replication alone cannot repair its selection dependence.

**Why this order:** trustworthy comparisons are a dependency for every model; A4 is a cheap, already motivated challenger; Boltz internal representations test the largest untested hypothesis without commissioning new coordinate production; independent data are required to turn development results into stronger claims. Transfer and calibration answer different questions and should not expand the first screen indiscriminately.

## 2. Starting evidence and its limits

| Existing finding | Use in the next study |
|---|---|
| Original sequence ensemble: validation 0.6931, test 0.7064 | Best arm on the original benchmark; historical reference for new work. Refit the same recipe under every new regime. Previously disclosed test exposure remains a limitation. |
| Original ESM-2 35M: test Δ −0.0085 [−0.0291, +0.0233]; sequence + structural summaries: −0.1117 [−0.1339, −0.0575] | No established improvement for the tested ESM arm; the tested structural pipeline is worse. Neither result evaluates internal Boltz representations or all foundation-model approaches. |
| A4 tree ranker: +0.0295 [0.0150, 0.0596] against the matched screen control (pooled primary statistic) | A promising sequence-control candidate. Global selection and screening budgets prevent treating it as a fully nested finalist result. |
| B2 floor-event Brier: 0.1099 → 0.0935 | A calibration candidate; approximately 15% relative improvement on this comparator. It is not a ranking gain. |
| Frozen MINT versus frozen ESM-2 | Evidence about the tested frozen setup. The previous blend gate needs repair, and adaptation was not assessed. |
| Structural residual D4: +0.0084 [−0.0055, 0.0124] | Weak motivation for more folding to regenerate these geometry/confidence features. This does not answer the internal Boltz-representation question. |

Sources: [original report](REPORT.md), [follow-up findings](further_research/FINDINGS.md), [A4 comparisons](further_research/phase_a_a4_comparisons.csv), [calibration results](further_research/phase_b_pooled.csv), and [structural residual report](further_research/phase_d_structural_residual.md).

The numerical anchors above were checked against saved tables during reconciliation; this is not a fresh validation of their statistical procedures. The primary sources still contain claims requiring the correction register below.

The existing outer-fold results have already informed study design. Repaired assessment on these training data remains **development evidence**. Correctly separating folds does not make the previously examined data prospectively untouched.

### Current implementation and evidence status

- The original benchmark and Phase A–D runs are complete. Completed runs do not imply all their interpretations or selection procedures are sound. Round-two implementation and audit artifacts now live in [further_research/round2/](further_research/round2/); see §12 for current completion status.
- Reusable assets include the frozen splits, original predictions, 28,166 reported production structures, their 109-feature summary table, cached MSAs and follow-up prediction tables. Remote structure availability and provider bills were not reverified in this review.
- The original production runner persists coordinates/confidence. The new embedding-only extractor has passed pinned full-forward equivalence and technical repeat checks; full-cohort extraction completed within its fixed cost allocation (§12).
- This review inspected the current working tree, including pre-existing uncommitted planning edits, and checked headline values against saved tables. The source dataset checksum passes. Software test results are recorded under §10; they do not certify the statistical design.
- The source corpus contains no supplied assay replicates. Fold spread, execution variation and correlations between different alleles cannot establish irreducible assay noise or recoverable performance headroom.

Original numerical sources: [test scores](stage6_test_summary.csv), [paired test intervals](stage6_test_paired_ci.csv). Follow-up numerical sources and the interpretation corrections are recorded below.

## 3. Milestone 0 — evaluation repair and corrections

### Selection inside each fitting partition

Preserve `data/splits.csv`, source labels, original prediction files and [EVALUATION.md](../EVALUATION.md). Load rows through `pepstab.data.load_with_splits()` and restrict development to the original training partition. `docs/SPLITS.md` is superseded and must not supply the split.

Reuse the frozen training-cluster outer folds for continuity. For each outer fold K, choose hyperparameters, stopping rules, transformations, learner families and blend weights using only K's fitting partition. Do not pool selection scores across outer folds, and do not choose a representative configuration or stopping cell by its outer score.

Report each declared learner procedure with per-outer-fold selection. If reporting a single selected sequence control or selected embedding procedure, define a selector that chooses among the permitted procedures using inner scores only. Freeze its candidates and tie-breaking rule before execution; baseline-only must remain available. Report all declared component procedures as well as the selector.

### Clean stacking and calibration inputs

For assessed fold K, generate every residual, blend and calibration input entirely within the fitting partition excluding K. Each inner-held cluster must be excluded from base-model fitting, hyperparameter selection and early stopping before its prediction is generated. Fit the second-stage model on these predictions and apply it to base predictions for K.

Other outer folds' predictions are not substitutes: their models may have used K's labels. Predictions on a member's own early-stopping rows are not clean stacking inputs either. Record fitting/selection ancestry so both exclusions can be verified.

Match the base recipes for stacking and outer application where practical and report remaining training-size differences. Do not assume that increasing inner ensemble size necessarily improves that match.

### Focused tests and acceptance checks

- Changing outer-held targets leaves that fold's selected configuration, transforms, weights and predictions unchanged.
- Every stacking prediction has ancestry excluding its predicted cluster and the assessed outer fold.
- Selection is per outer fold; no outer score enters the selector.
- Joins preserve row identity, coverage and cluster membership.
- A zero-weight blend reproduces the corresponding baseline exactly, including pooled scores. Fold-specific normalisation must not introduce a change by itself.
- Numerical report claims reproduce from saved tables, including interval bounds, counts and relative improvements.

During implementation, run the existing suite plus these targeted tests. Existing passing tests and the source-data checksum do not validate the missing selection/ancestry guarantees.

### Dated correction register — 12 October 2026

These corrections govern interpretation now. Notices in the historical reports point here; numerical artifacts remain intact. **Documented corrections, repaired assessment code and saved-prediction reconciliation are complete; new bounded development assessments are complete (§12); independent confirmation remains open.**

| Historical claim or issue | Current interpretation and required action | Evidence |
|---|---|---|
| Phase A is fully nested | `select_arms` pools selection runs over outer folds; an A3 representative is then chosen by outer scores. These are screening results. Select anew inside each assessed fold's fitting partition. | [selection code](../scripts/phase_a_select.py), [selection table](further_research/phase_a_selection.csv) |
| A4 proves the two changes compose, or its gain is below 0.05 | Reported Δ +0.0295 [0.0150, 0.0596] straddles 0.05. Its primary contrasts with matched A1 and A3-c cross zero. Neither a deployable gain, additivity nor “below the bar” is established; the intervals also omit screening selection uncertainty. | [A4 comparisons](further_research/phase_a_a4_comparisons.csv) |
| All inner-OOF predictions are clean stacking inputs | `fit_mlp_ensemble` predicts inner rows with members that early-stopped on those rows. Exclusion from gradient fitting is insufficient. Audit each role, including deeper `inner_oof_30`, rather than assuming every role has the same defect. | [nested helper](../pepstab/nested.py), [Phase A baseline generator](../scripts/phase_a_a0.py) |
| Phase C's blend gate is clean; 11/15 weights are zero | Models generating other outer folds' predictions can have fitted the assessed fold. The weight's input ancestry is contaminated. Saved weights contain **6/15 zeros**. Fold-specific standardisation also changes the pooled baseline endpoint. Retract the clean-gate claim; complementary signal remains unresolved. | [blend code](../scripts/phase_c_blend_gate_check.py), [saved weights](further_research/phase_c_blend_gate_check.csv) |
| B2 “more than halves” floor Brier and has no ranking cost | Brier 0.109946 → 0.093452 is **15.0% lower**; fixed-bin ECE falls about 69.6%, a different metric. Ranking Δ −0.0108 [−0.0242, +0.0036] does not establish the proposed −0.02 tolerance. Retain B2 as a calibration candidate with an unresolved ranking tradeoff. | [pooled B results](further_research/phase_b_pooled.csv), [paired results](further_research/phase_b_paired.csv) |
| Phase C scores and A0 0.6493 are directly comparable | A0 0.6493 is the median of fold medians; pooled A0 is approximately 0.6615. Attach deltas/intervals only to the same population and aggregation. Rebuild mixed tables from saved predictions. | [Phase A predictions](further_research/phase_a_seq_oof_cluster.csv), [C summary](further_research/phase_c_frozen_summary.csv) |
| A CI missing its point estimate proves an artefact; allele Wilcoxon is an assumption-free sign test | Investigate bootstrap bias/sensitivity; this observation alone does not invalidate the statistic. Alleles share peptides, and Wilcoxon is a signed-rank test. Keep cluster-aware primary inference. | [C diagnostic](further_research/phase_c_frozen_paired_diagnostic.csv), [blend code](../scripts/phase_c_blend_gate_check.py) |
| Corpus self-overlap measures exact checkpoint exposure | The stability audit uses our corpus as a proxy for an unavailable upstream file. It establishes a serious overlap concern, not checkpoint-specific train-partition membership. Keep overlapping supervised checkpoints out of the clean comparison until provenance is resolved; do not call their score pure memorisation. | [overlap audit](../scripts/phase_c_leakage_audit.py) |
| Measurement precision is proved to be the binding constraint; 0.0043 is a universal floor | There are no supplied assay replicates. The measured MLP execution difference does not transfer to CatBoost or Boltz. Report each uncertainty separately and acquire replicates. | [platform experiment](further_research/phase_a_platform_artifact.md), [evaluation contract](../EVALUATION.md) |
| Structural residuals or stable summary features close further structural research | D4 is weak evidence for more of the tested summaries, subject to ancestry review. It does not exclude internal embeddings, adaptation, dynamics or all structural predictors. Pilot pose repeatability does not establish general half-life signal or an assay ceiling. | [D report](further_research/phase_d_structural_residual.md) |

These are specific interpretation and implementation defects, not grounds for discarding every saved measurement. Fix generators as well as prose so rerunning a legacy script cannot silently republish a superseded claim. Record which replacement tables can be rebuilt from existing predictions and which require refitting. Do not present either as independent confirmation.

**Gate:** tested assessment harness, ancestry manifests and corrected report generators before supervised candidate assessment. The original review completed the written register; subsequent tested implementation and reconciliation are recorded in §12.

## 4. Milestone 1 — audited Boltz embedding extraction

### Existing cache and extraction boundary

- The original Boltz experiment used geometry and confidence summaries, not internal embeddings. Its negative result does not answer the representation question.
- Production used `boltz==2.1.1`, a pinned checkpoint, cached MSAs, three requested recycling steps, and the HLA ectodomain + beta2m + peptide construct. The launch did not request embeddings, and the persistence code copied only CIF, confidence, PAE and pLDDT outputs. See [production pipeline](../modal_app/ectodomain_production.py).
- The archived strategy reports that inspection of the 2.1.1 source established that the final token representation `s` and pair representation `z` are computed before diffusion. A custom extraction path should therefore be able to omit coordinate generation and confidence scoring. Numerical equivalence and performance have not been tested. Current [upstream source](https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/models/boltz2.py) also shows this ordering; implement against our pinned version, not an assumed equivalent latest version.
- Saved coordinates and confidence cannot reconstruct the missing internal activations. Cached MSAs avoid repeating sequence searches, but the trunk must be run again.
- The exact winning extraction layer, construct, recycling count, pooling, predictor, data exclusions and splits remain unknown. If code or a methods description becomes available, record these details before calling an experiment a reproduction. Otherwise proceed with the explicitly labelled trunk-embedding hypothesis.

### 4.1 Verify the exact representation

Start from final post-recycling trunk `s` and `z`, frozen weights in evaluation mode, and the original three-chain construct. Preserve checkpoint, MSA handling, chain order, recycling count and numerical settings. Do not replace the full input with a cropped groove during this first experiment; crop the output representation. Input cropping is a later ablation because it changes the context seen by the model.

Implement a dedicated extraction return after the final trunk update. Bypass diffusion, confidence and affinity heads. The normal prediction/writer path may expect coordinates, so simply setting a skip flag is not a complete extractor. Keep the original folding path intact and version the extraction patch.

### 4.2 Technical panel and acceptance checks

Use 8–16 training complexes spanning several alleles, including an engineered C67S construct and a borrowed-alpha3 case where available. Include varied MSA depths. Select the panel for engineering coverage, not for favourable model errors.

For the same preprocessed input and checkpoint, compare a full forward pass instrumented at the trunk boundary against the extraction-only path:

- Exact row/sequence identity and token-to-chain/residue mapping; exclude padding and special/non-residue tokens correctly.
- Shape and finite-value checks; retain both directions of pair representations instead of assuming symmetry.
- Numerical equivalence with tolerances declared by dtype and repeat-run numerical variability, before inspecting downstream label performance.
- Confirm that diffusion and confidence code do not execute in extraction-only mode.
- Verify that serialization, reloading, indexing and any storage precision conversion preserve the intended features.
- Measure peak GPU/host memory, cold-start time, warm throughput, transfer time and compressed bytes per complex.

If tensors differ, resolve the cause before scaling. A successful process exit is not sufficient verification.

### 4.3 Cache only what the experiments need

Retain peptide token embeddings in positions P1–P9, groove token embeddings for HLA residues 1–182, and both peptide–groove pair blocks. Store the complete 9×182 interface blocks so the 34-contact-residue subset and other position-aware summaries can be derived without rerunning Boltz. Keep full-complex tensors on a few audit cases only unless the winning method requires them. Store token maps, shapes, dtype, checkpoint/input hashes and extraction settings alongside every shard.

Use resumable, atomic shard writes and a completion manifest; existing structure caches must not cause embedding jobs to be skipped merely because a CIF exists. A cache entry is complete only when the expected embedding blocks and their metadata validate.

### 4.4 Budget gate

After correctness, profile a larger training panel (approximately 64 complexes). Forecast full training extraction from measured throughput, cloud runtime rates, memory and storage. Record model-load/initialisation overhead separately. Do not reuse the old full-fold cost as an embedding estimate or promise a speedup without measurement.

Proceed to training-cohort extraction after this engineering/cost gate; do not require the old geometry features to improve first. A small technical panel is not powered to reject the embedding hypothesis. If cost is restrictive, declare a cluster-preserving reduced training study and compare every arm on exactly that cohort, labelling its limited power.

**Gate:** equivalence report, complete cache schema, measured forecast and explicit study-cohort manifest. Full extraction requires this gate; supervised assessment additionally requires milestone 0. Extraction engineering can proceed while assessment is repaired.

## 5. Milestones 2–3 — bounded predictors and positional tests

### Core representation screen

Use the following declared representations. Start with ridge and a small MLP, each with its own bounded regularisation search. Do not stack all representations, target changes and learner changes into a single first experiment.

| ID | Representation | Question |
|---|---|---|
| R0 | Position-preserving peptide + HLA pseudosequence | Matched sequence control |
| R1 | Existing geometry/confidence, with a matched sequence control | Historical bridge; limited diagnostic scope because the summary-feature residual has already been tested |
| R2 | Frozen Boltz token embeddings, chain-wise mean pooling | Coarse learned representation |
| R3 | Token embeddings retaining P1–P9 and fixed groove positions | Value of preserving position |
| R4 | Directional pair embeddings pooled over the groove separately for each peptide position | Value of explicit interface representation |
| R5 | Position/contact-aware pair summaries | Value of peptide–pocket correspondence |

R1 is not a prerequisite for R2–R5 and does not justify regenerating coordinates. Fix the small R1 diagnostic budget before fitting rather than rerunning the entire earlier structural programme.

Freeze at most 12 configurations per learner/representation, three inner cluster folds and two paired seeds. Within each outer fitting partition, apply a recorded parsimony rule and preserve every declared outcome. Fit scaling, dimensionality reduction and supervised feature choices inside the appropriate fitting partition. Frozen label-independent extraction may be cached across folds.

Keep positional comparisons interpretable:

- Compare pooling versus retained positions at matched dimension/head capacity where practical.
- Keep central peptide positions as well as anchors.
- Fix the contact-map source and geometric rule before fitting; report any dependence on predicted coordinates.
- Include a fixed shuffled position-to-pocket mapping control that changes membership in the constrained summaries. Merely permuting columns of an unrestricted ridge/MLP leaves the available information unchanged and does not test biological correspondence.
- Select regularisation separately for information-rich and control representations; a transplanted grid must not handicap one arm silently.

For R5, retain P1/P2/P9 summaries and a separate P3–P8 block. Compare against a position-indexed 9×34 peptide/contact-residue representation, reducing its channel dimension within fitting folds when needed. The 34-residue list alone does not define a position-specific contact map: choose either a documented label-independent structural map or a fixed geometric rule on our saved predicted structures before outcomes. Keep both directions of the pair block; do not assume symmetry.

### Competitive sequence controls

Complete the learner × target 2×2 under corrected assessment; reuse existing fitting artifacts only if their full ancestry passes milestone 0.

| Learner | Original log1p target | Shrunk allele-centred target |
|---|---|---|
| Original MLP | Matched reference | Reassessment of the earlier neural centring candidate |
| Boosted-tree regression | Reassessment of A1 | Required combined learner/centring ablation |

Do not infer that a negative MLP centring result settles centred boosted trees. Fit means inside the fitting fold only; unseen alleles use the fitting-derived shared fallback. Treat centring as a hypothesis despite the previous negative neural result, rather than assuming it improves training.

Include the existing promising A4/PairLogit tree-ranker recipe in the first corrected sequence-control candidate set, with per-fold selection and a declared budget; it need not wait for a repeat of completed discovery work. If extending the neural ranking arm, limit lambda to {0.1, 0.3, 1.0, 3.0} and declare the stopping comparison before fitting. Do not reopen balanced sampling or broader objectives without a distinct recorded question.

Compare every embedding finalist with both the fixed original sequence recipe and the inner-selected competitive sequence procedure. Report encoding, seed/member counts, fitting effort and inference cost. Equal numbers of tree and neural fits do not imply equal capacity. Final neural arms should receive matched ensemble recipes; tree bagging effort is declared and costed separately.

Use position-preserving one-hot/BLOSUM encodings initially and one tree library. Do not encode amino acids as arbitrary ordered numbers. Add allele means back for absolute predictions, recognising that the constant cannot itself improve within-allele ranking. Exclude raw allele-identity shortcuts from the pan-allele main arm. Pure rankers require a separately evaluated training-only calibration map for magnitude metrics.

### Complementarity

For a small candidate set, assess embeddings alone, a strongly regularised residual correction, and a simplex prediction blend. Fit residual targets and weights through the clean ancestry procedure from milestone 0. Keep zero correction/baseline-only available. Compare with both sequence references.

A blend gain and a standalone gain are different findings. Report both, with all selection history. If standalone and complementarity results are negative, limit the conclusion to the tested representations, predictors and regime.

**Gate:** complete candidate table, predictions, provenance, costs and matched positional ablations. Nominate at most two ranking procedures for independent confirmation. Any shortlist chosen after viewing these development results remains development-selected.

**Scale control:** the upper bound for R2–R5 alone is 4 representations × 2 learners × 12 configurations × 3 inner folds × 2 seeds × 5 outer folds = **2,880 inner fits**, before refits, controls, selectors and clean stacking. Freeze the actual smaller grids and forecast the entire fit graph before execution. “Twelve configurations” is not the total study cost. Do not expand to all transfer regimes until the main screen is complete; freeze the transfer selector/candidate set before inspecting transfer outcomes.

## 6. Milestone 4 — reproducibility and transfer

Repeat the locked reference and leading candidate recipes twice at identical seeds/settings to measure execution reproducibility. Record hardware, numerical libraries, precision and thread counts. Report that variation separately from paired-seed variation. Do not assign CatBoost or Boltz the historical MLP difference of 0.0043 as a universal noise floor.

Refit and assess each comparator under these separate regimes:

| Regime | Main interpretation | Additional uncertainty to report |
|---|---|---|
| Peptide-cluster-held-out | New peptides, with familiar HLAs permitted | Paired peptide-cluster resampling |
| HLA-group-held-out | New HLA groups, with peptide overlap quantified | HLA-group uncertainty as well as peptide dependence |
| Joint-held-out | Both peptide cluster and HLA group new to fitting | Crossed peptide/HLA-group uncertainty; small-group sensitivity |

Freeze sequence-based HLA grouping, engineered-construct treatment, fold combinations and eligibility before outcomes. For joint holdout, select evaluation peptide clusters P and HLA groups A. Fit only rows outside P AND outside A, and assess P×A. Neither cross-quadrant may enter fitting, selection or early stopping. Mirror the regime at the inner level. Keep identical HLA pseudosequences and closely related constructs in the same HLA group. Select grouping rules using sequence distances and coverage before outcomes, not model scores. Auxiliary manifests live inside the original training partition and do not overwrite the frozen benchmark split.

Publish row/allele/group coverage, censoring rates, nearest-fitting-sequence distances and sparse-intersection limitations. Do not weaken holdout rules or select favourable fold combinations after results. A peptide-only interval does not establish generalisation uncertainty over an unseen HLA population.

**Gate:** each claim identifies its regime, comparator, uncertainty and added inference cost. A gain on familiar HLAs is not an unseen-HLA result.

## 7. Optional calibration branch

Keep this behind the core representation/sequence work. The completed calibration screen informs candidate selection; it is not an untried branch. The first candidate is B2, censored loss with MSE stopping. Compare it with Platt and isotonic floor-probability maps fitted on clean inner sequence predictions; retain the historical Gaussian construction as a secondary comparator.

The observable floor event remains `thalf_hours == 0`, distinct from the likelihood's censoring mask. Primary calibration metric: improvement-signed Brier delta versus the inner-selected simple calibration baseline. Carry 0.003 as a provisional study decision threshold, not a universal detection limit or practical-value guarantee. Report fixed-bin ECE, reliability, floor AUROC and censored NLL as secondary diagnostics.

Proposed ranking tolerance: a loss of no more than 0.02 median per-allele Spearman. Establish acceptable loss only if the ranking-delta interval's lower bound exceeds −0.02; otherwise leave the tradeoff unresolved. Freeze this tolerance before new fitting. Report floor probability and ranking score separately where appropriate, and never evaluate ranking through an isotonic probability map.

Do not repeat B1 or the two-part model without new evidence. Nominate at most one calibration procedure for independent confirmation. Its gain does not count toward the 0.05 ranking bar.

## 8. Statistical reporting rules

- Primary continuity statistic: median per-allele Spearman over pooled outer-held predictions, using the original tie and constant-prediction rules. Report per-fold medians separately and never mix their estimates/deltas with pooled intervals.
- Pure rankers can emit arbitrary scales across folds, which can alter pooled within-allele ranking. Prespecify a common-scale mapping (for example, a positive affine map fitted on clean inner predictions to log1p labels) as part of the assessed procedure; report raw per-fold ranking separately. No isotonic map goes into the ranking endpoint. Preserve the original sequence endpoint exactly at zero blend weight; arbitrary fold-wise z-scores are not a substitute.
- Determine allele eligibility from labels/counts before candidate fitting: at least 20 rows and two distinct labels for the main training-fold panel. Report the 50-row sensitivity and coverage without selecting the favourable panel.
- Use the existing 2,000 paired whole-peptide-cluster bootstrap procedure for the peptide-held-out endpoint. These intervals condition on fitted predictions; report execution/seed variability separately. Transfer requires the additional group treatment above.
- Declare the inner-selected embedding procedure versus the inner-selected sequence procedure as the primary development ranking comparison. Original-reference and component comparisons are secondary; retain every declared comparison. This is a comparison of selectors, not an outer-outcome winner comparison.
- A positive lower interval bound supports an improvement on the specified statistic. A lower bound above 0.05 supports the original worthwhile ranking bar. Failure to exclude zero does not prove equivalence.
- A percentile interval excluding its observed point estimate prompts a bias/sensitivity investigation, not an automatic artefact diagnosis. If BCa is added, jackknife/resample whole clusters and report both methods without choosing the favourable interval.
- Per-allele paired deltas are useful diagnostics. Do not substitute unadjusted Wilcoxon tests over dependent allele results for cluster-aware inference or describe them as assumption-free sign tests.
- Repeated-search optimism remains after fold repair. Do not claim that fold spread or numerical variability identifies an assay reliability ceiling; suitable assay replicates are required.

## 9. Milestone 5 — independent data and confirmation

Begin feasibility work during milestone 0. Inventory accessible independent half-life datasets or assay partners, with access conditions, exact peptide–allele identities, temperature, construct, protocol, assay floor and measurement uncertainty. Distinguish new measurements from republished source-corpus rows. Elution/presentation data remain transfer checks, not substitutes for half-life confirmation.

Design a representative cohort across alleles, peptide families and lifetime ranges, with a replicate subset and poorly covered alleles. Declare unseen-HLA transfer as a separate endpoint. Disagreement-selected active-learning samples may form a separate acquisition arm; they must not be the only confirmation population. Once assessment is repaired, bounded training-cluster learning curves for the reference and leading candidate can inform acquisition; they cannot substitute for assay replicates or identify an assay ceiling.

For peptide-novel confirmation, audit exact and Hamming-distance-3 cluster overlap against all development data and available supervised upstream donor data. Report coverage lost to exclusions and overlap strata separately. Audit structural-pretraining exposure separately from supervised half-life leakage; unknown provenance must remain labelled unknown.

Select panel size through simulation of the paired primary endpoint, peptide/HLA dependence and per-allele coverage. Report sensitivity near both a small gain of 0.02 and the original 0.05 bar. Obtain assay quotes before assigning a cost or calendar date. Do not estimate recoverable headroom from correlations between different alleles.

Reserve confirmation labels before using external data for adaptation or calibration. Freeze at most two ranking procedures and one calibration procedure, including preprocessing, model/data hashes, ancestry, endpoints and analysis code. Designate one primary ranking comparison and report the other as secondary, or prespecify multiplicity handling if both will support confirmatory claims. Fit on development data only, then evaluate the reserved cohort once. The original test is not a replacement. If independent data cannot be obtained, publish development-only results and leave confirmation open.

**Gate:** a prospective or independently reserved confirmation result, including null or worse outcomes. Update performance claims only at this gate.

## 10. Resources, artifacts and deferred work

| Work | Resource decision |
|---|---|
| Evaluation repair and corrections | Local CPU; first implementation task |
| Extractor equivalence and throughput | Small successful technical panel before any full cache job |
| Training cache | Forecast runtime, transfer and storage from the measured profile; set the ceiling before launch |
| Predictor screens and repeats | Forecast CPU cost from successful smokes; keep grids and finalist counts bounded |
| Independent data | Panel design, assay quotes and turnaround determine budget and schedule |
| Calibration or encoder adaptation | Separate scope after the core milestones; no automatic expansion |

No cloud-cost or completion-date promise is made before the extraction benchmark. Record rates, overhead and storage explicitly. If a forecast exceeds its declared ceiling, revise scope before running rather than silently truncating or expanding an arm.

Use `reports/further_research/round2/` for this study, leaving earlier phase artifacts intact, and a separate versioned embedding-cache namespace. Required artifacts are:

- Dated protocol/config manifest and code revision.
- Data, fold, feature and checkpoint hashes plus fitting/selection ancestry.
- Extractor equivalence report, throughput/storage forecast and cache inventory.
- Per-fold selections, seed/member counts, hardware/runtime records and failures.
- Predictions keyed by row identity, with role, regime and fold.
- All comparisons, coverage, calibration diagnostics, execution repeats and cost provenance.
- Decision log distinguishing unchanged rules, amendments and newly generated hypotheses.

Generate numerical report text from the result table where practical. Each comparison records population, reference, statistic, estimate, delta, interval, coverage, selection history and cost provenance.

Defer new coordinate production, dynamics, energy production, broad checkpoint sweeps and affinity pretraining. Limited encoder adaptation remains an open hypothesis; reconsider it only with a specific question left by the corrected frozen/Boltz results, matched controls and a priced pilot. A frozen null cannot scientifically exclude adaptation.

### Disposition of remaining proposals

| Proposal or outstanding item | Final priority and reopening condition |
|---|---|
| A4/A1 replication | Included in milestone 1 corrected controls; record within-setting execution repeats and paired-seed variation separately. The old approximately $2 estimate is not a new quote. |
| Neural ranking λ extension to 3.0 | Bounded secondary control in milestones 2–3 if neural ranking is included in the declared selector. Use {0.1, 0.3, 1.0, 3.0}; do not execute the stale ladder that extends the wrong end. |
| BCa sensitivity / pooled-table reconciliation | Statistical repair and sensitivity work under milestone 0/§8; not a new modelling branch and not a licence to pick a favourable interval. |
| ESM-C/ProtT5, peptide–linker–HLA inputs, additional cross-attention heads | Deferred. The README's original three-item list is superseded. Reopen only for a concrete representation question left by the bounded Boltz study, with matched position-aware controls and a costed pilot. |
| MINT limited adaptation | Scientifically open, lower priority than frozen Boltz. Its old blend gate cannot establish lack of complementarity. Reopen with clean upstream provenance, a matched baseline and a separately budgeted learning-curve pilot; the old ≥$39.72 forecast is not an executed adaptation study or current price. |
| Affinity-then-half-life pretraining | Separate deferred project requiring a demonstrably clean donor corpus and matched controls; not implied by this plan. |
| ESM-2 likelihood features (stage 3d) | Not run, still untested, deferred behind representation and confirmation work. Do not label this a negative result. |
| ESM-2/structural elution pass | Optional transfer endpoint for locked finalists, after overlap and decoy controls. Do not generate a large new structural cache merely to fill an old checklist. Presentation discrimination is not half-life validation. |
| ProteinMPNN full-cohort scoring | Existing pilot/QC work does not establish incremental half-life value; full scoring remains deferred. Reopen only with a distinct, matched training-only feature pilot that addresses sequence-recovery confounding. |
| FoldX full-cohort energy features | Pilot only; no full half-life arm was scored. Deferred for weak current motivation and unresolved access/cost details. Equilibrium energy differs from a kinetic target, but this does not mathematically prove a performance ceiling or rule out complementary information. |
| More poses, new folding engines, molecular dynamics | Deferred. Existing summary-feature negatives do not settle these questions, but they provide little motivation to spend now. Require a distinct measurable hypothesis, matched controls and a priced feasibility study. |
| Provider-bill reconciliation | Open administrative task. Reconcile the historical $212.71 runtime-derived estimate when provider records are available; do not use it as an embedding-cost forecast. |

### Review checks and maintenance

The 12 October review checked source-data integrity, the current test suite, the saved headline tables, selection/stacking source paths, and plan/report links. **799 tests passed**, with 17 structure-parser warnings about falling back from `auth_atom_id` to `label_atom_id`; the canonical source-data SHA-256 check passed. Passing tests do not cover the methodological defects identified in §3. This review did not independently reproduce every experiment, audit remote volumes, obtain a provider bill, verify winning-team methods or run a systematic literature/data search. The [current upstream Boltz implementation](https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/models/boltz2.py) was inspected and supports the trunk-before-diffusion boundary; equivalence against pinned 2.1.1 remains an implementation gate.

Keep the [research index](README.md), [project README](../README.md) and site source as entry points to this plan, without separate priority lists. The submission film is a dated historical artifact; its proposals are not current instructions. Local site-source cleanup does not publish a website update.

**Immediate next evidence gate:** obtain an adequate independently reserved half-life cohort and lock its one-time confirmation analysis before opening outcomes. The bounded development execution and consolidated reporting are complete; no suitable confirmation cohort or assay quote exists. Both cloud jobs are stopped and reservations settled. No new fitting, grid expansion, calibration/adaptation or paid acquisition follows automatically. See §12 and the [consolidated report](further_research/round2/REPORT.md).


## 11. Reconciliation decisions (8 October 2026)

| Difference between the plans | Resolved instruction |
|---|---|
| Separate strategy and execution documents could diverge | This file is canonical; earlier plans are preserved in archive/plans/ with historical notices |
| Earlier plan starts with extractor/protocol; review requires statistical repairs | Milestone 0 blocks supervised assessment; extractor engineering and independent-data feasibility can proceed concurrently |
| Earlier plan lists trees, ranking, MINT, censoring and structural residuals as future experiments | Use completed Phase A–D evidence with its limitations; repair/reassess promising candidates instead of repeating discovery blindly |
| Boltz cache/extraction details lived only in the earlier plan | Preserve pinned construct/settings, both directional 9×182 pair blocks, residue maps, equivalence tests and resumable storage in §4 |
| Structural-summary negatives risk delaying internal representations | Frozen Boltz R2–R5 are the first new representation study; R1 is only a bounded diagnostic |
| “Choose the best arm” and stacking language were underspecified | All assessed selection occurs inside each outer fitting partition, including selector families and base-model ancestry for meta-models |
| Ordinary versus centred trees could be omitted after a neural centring null | Retain the complete 2×2 plus A4 as an early competitive control |
| Earlier plan reopens balancing and two-part floor models | Defer them absent a new question; optional calibration prioritises B2 versus clean simple maps |
| Output paths risk mixing old and repaired results | All new assessment artifacts go under further_research/round2/; original phase outputs remain intact |
| Confirmation and provisional calibration thresholds differ in detail | Preserve NEXT's explicit confirmation design and provisional 0.003 Brier / −0.02 ranking-tolerance rules, separately from the original 0.05 ranking bar |

The report index points to this document for future scope. Historical phase prespecifications remain the records of what earlier experiments intended; the current plan must not be retroactively described as their original protocol. The 8 October consolidation established one planning document; the 12 October review adds the corrections and final dispositions below.

### 12 October review and final disposition

At the initial review, only the written correction register was complete. The following execution record supersedes that initial pending status without changing the historical findings.

| Additional conflict found in review | Resolution |
|---|---|
| Findings rank independent data first while Boltz strategy ranks embeddings first | Start data feasibility now; execute modelling only after assessment repair. Confirmation is mandatory for new confirmatory claims, but waiting for an assay is not a prerequisite for bounded development. |
| Findings call measurement precision the established bottleneck | Treat it as an untested explanation. No assay replicates are available; promising representations remain untested. |
| README and website still promote ESM-C, new attention heads and energy production | Replace the live lists with a link here; preserve their former wording in the archive and label the submission film historical. |
| Phase C's stopping decision is treated as a scientific closure of adaptation/blends | Retract the clean blend-gate rationale. Keep adaptation deferred on opportunity cost and scope, rather than claiming the flawed gate has excluded it. |
| Proposals omit total search cost or mistake a feature permutation for a biological control | Count the full fitting graph; require pocket-membership controls that actually change the tested summaries. |
| Historical reports and methods contain future-tense instructions | Label them historical at entry, preserve results/protocols, and route all future scope changes here. |


## 12. Autonomous execution record — 12 October 2026

The user authorized autonomous execution and reported **$213 remaining Modal credit**. The orchestration ceiling is **$100 additional spend**, preserving at least $113 of that reported balance, subject to unrelated account activity and delayed provider metering. No automatic expansion, retry loop, new coordinate production or encoder adaptation is authorized by this execution record. Bounded delegated implementation/reporting uses 6.1 Sol; complex extractor/statistical judgments remain with the orchestrator and the existing specialist work.

| Gate | Current evidence and disposition |
|---|---|
| Assessment repair | Implemented per-partition selection, target-stripped prediction inputs, clean reselected cross-fitting, positive ranker scaling, exact zero-weight endpoint and content-addressed ancestry. The latest full integration passed **878 tests** with the 17 existing structure-parser warnings; subsequent focused seed-persistence, budget and pipeline tests also pass. Tests are evidence about implementation, not independent scientific confirmation. |
| Historical reconciliation | [Generated reconciliation](further_research/round2/corrections/REPORT.md) and saved paired comparisons rebuild compatible pooled endpoints, correct Brier/ECE/weight-count claims and preserve historical artifacts. Five superseded assessment CLIs now stop before publishing obsolete analyses; importable historical helpers remain available. |
| Extractor correctness | Ten technical cases passed full-forward versus extraction and repeated-input **bitwise equality** of final native s/z tensors. Construct, token indexing, checkpoint/source hashes, MSA provenance, directionality, finite fp16 storage and reload were checked. [Engineering forecast](further_research/round2/extractor/forecast.json) records 64 additional successful cases. |
| Full cache | All **19,716 original training rows**, **3,587 peptide clusters**, **75 alleles** are preserved in the [frozen cohort](further_research/round2/extractor/full_training/cohort.csv). The 64-case batch forecast was $72.62 before contingency. A 256-case scheduling amendment reduces replicated startup; projected budget-accounted spend including startup/idle allowances is $71.47. A single bounded production shard precedes fanout. The first 256-case production shard passed all checks in 2,060.71 seconds, with a $0.8734 budget-accounted charge. Its measured whole-cohort forecast with 10% contingency is $73.99, below the fixed $80 allocation; eight-worker extraction completed all 78 shards and 19,716 rows at 06:46 UTC. All reservations are settled, with $66.8431 runtime-budget accounting against $80 and no cache preemptions. The cloud app stopped normally. The [completion audit](further_research/round2/resources/cache_completion_audit.json) records local feature and provenance checks. Full allocations remain $80 cache + $18 sequence + $2 engineering = $100. |
| Sequence controls | [Frozen protocol](further_research/round2/sequence/protocol.json) covers the full MLP/tree × log/centred target grid plus PairLogit, per-fold selection, original 30-member neural reference, six-member tree procedure, clean positive affine ranker mapping and two identical-setting execution repeats. Initial remote packaging failure was repaired and retained in the failure record. The remote smoke passed: full-size depth-8 fit 576.11 seconds, synthetic local/remote maximum difference 9.7e-17 and repeated remote predictions identical. The complete 900-remote-call fit-graph forecast is $17.50 including margins and the smoke/build debit; the $18 reservation ledger stops before exceeding commitments. The full five-fold assessment and two repeats launched at 00:47 UTC. All five primary folds finished and passed the [completion audit](further_research/round2/resources/sequence_primary_completion_audit.json) at 02:56 UTC: all 19,716 training rows, 2,100 members, 53 artifact hashes, identity, ancestry and inner-selector replay checked. Both execution repeats and the [sequence-only results report](further_research/round2/results/sequence/REPORT.md) are complete. All 900 remote calls settled and the app stopped normally at 03:14 UTC. The conservative cost including smoke/build and seven lost-attempt reserves is $7.9957 against $18; provider metering remains separately recorded. Tiny-platform agreement is not a full-cohort cross-platform equivalence claim. |
| Representation screen | [Frozen protocol](further_research/round2/embedding/protocol.json) covers R0–R5, shuffled memberships and matched positional controls, with three regularization settings per learner and three inner folds. The full graph is **2,900 actual head fits**, including all refits and secondary controls, on local CPU. No real head fitting preceded the gate. The durable local controller passed the complete cache/sequence gate and started the first three folds at 06:46:38 UTC; all five folds and the [main report](further_research/round2/results/main/REPORT.md) completed by 07:19 UTC. The [completion audit](further_research/round2/resources/embedding_completion_audit.json) passed all 100 component procedures, 2,900 members, identity/ancestry/scaler/selector checks, source/output hashes and both panels' endpoint/interval arithmetic. The cache wrapper provenance limitation remains inherited. |
| Independent data | [Identity-only provenance audit](further_research/round2/independent_data/README.md) found all 27,031 released source stability pairs in the canonical corpus. Strict donor/canonical exclusions leave only 129 potential rows across heterogeneous assay strata, insufficient for the planned confirmation endpoint. External labels were not opened. Metadata, cross-length similarity and checkpoint provenance gaps remain. |
| Complementarity | [Frozen protocol](further_research/round2/complementarity/protocol.json) covers R2/R4 ridge blends and strongly regularized residuals against the exact original reference, with baseline-only options and fresh nested stacking ancestry. All five folds and the [report](further_research/round2/complementarity/results/REPORT.md) completed by 07:54 UTC. The [completion audit](further_research/round2/resources/complementarity_completion_audit.json) passed identities, clean ancestry, all choices, exact zero endpoints and both panels' arithmetic. The recorded graph contains 2,140 logical model fits including compatible cache reuse; this is not an independent count of equally many new hardware executions. |
| Transfer | [Auxiliary manifest](further_research/round2/transfer/protocol.json) freezes 35 HLA sequence groups, five HLA folds and all 25 joint intersections. Coverage and nearest-sequence distances are recorded before transfer outcomes. The main completion audit has passed; the dated post-main amendment below nominates R4 ridge and R5 MLP before transfer fitting. This is not a transfer result. The reviewed [local implementation](further_research/round2/transfer/implementation_readiness.md) supports the original reference, a separately named bounded MLP sequence selector, and one or two explicitly nominated R2–R5 ridge/MLP procedures with unchanged main grids. All 30 cells require 10,440 sequence fits plus 750 per ridge or 2,340 per neural nominee. The earlier empty-nominee draft remains a preparation record. The dated amendment records the reduced comparator, exact nominees and 13,530-fit graph. All 30 cells completed at 09:06 UTC; the complete report finished at 09:10 UTC. The [completion audit](further_research/round2/resources/transfer_completion_audit.json) checks all members, source/artifact hashes, ancestry, selections, endpoints, uncertainty and group deletions. Final findings and limitations are recorded below; neither the nominees nor grids changed after transfer outcomes. |
| Confirmation | [Bounded acquisition design](further_research/round2/independent_data/design/README.md) completed 108 synthetic scenarios, 162,000 endpoint draws and 3,240 exact paired-bootstrap diagnostics. Balanced 12×200/400/800 panels and a separate 20% independent-replicate subset are options for quotes; rates are conditional planning sensitivity, not a guaranteed real-world power result. No assay quote, reserved adequately powered cohort or confirmatory result exists. Performance claims remain development-only. |

### Completed sequence development finding

The repaired inner-selected sequence procedure scores **0.666841** versus **0.661522** for the original reference: paired difference **+0.005319 [−0.009283, +0.012299]** on the primary 69-allele panel. The 50-row sensitivity is **+0.003421 [−0.008264, +0.010427]** across 68 alleles. Neither interval establishes improvement. The inner selector chose centred MLP in three folds and the original recipe in two; the tree ranker scores 0.651190, with difference **−0.010332 [−0.036280, +0.009983]** against the original reference. These comparisons use 2,000 shared paired peptide-cluster draws and remain conditional development evidence. They do not independently confirm or refute the earlier differently assessed screening claim.

Both locked repeats of the original MLP, ordinary tree and PairLogit exactly match their primary saved predictions across all rows. This establishes repeatability for the recorded settings/platforms; it does not measure assay noise or universal cross-platform variation. [Repeat and report validation](further_research/round2/resources/sequence_repeat_completion_audit.json) covers all 420 additional members, ancestry, hashes, endpoint/interval arithmetic and both complete panels. The completed main screen and dated transfer amendment below now govern nomination.

### Completed representation development finding

The [audited main screen](further_research/round2/results/main/REPORT.md)
scores **0.721163** for the inner-selected embedding procedure versus
**0.666841** for the competitive sequence selector. The prespecified primary
difference is **+0.054322 [0.031395, 0.080805]** on the 69-allele, 20-row
panel; the 68-allele, 50-row sensitivity gives **+0.056352 [0.031732,
0.079137]**. Against the original reference, the differences are **+0.059642
[0.031711, 0.081264]** and **+0.059773 [0.031921, 0.079272]** respectively.
All intervals use 2,000 shared whole-peptide-cluster draws and condition on
fitted predictions. Positive lower bounds support a development improvement;
the lower bounds remain below 0.05, so the original worthwhile-gain bar is
**not established**. Reused development data, unseen-HLA transfer and
independent confirmation remain separate limitations.

The matched positional controls support retaining position in the tested
representations: full 9×34 MLP versus its same-dimension pooled control gives
**+0.166349 [0.137791, 0.199165]** on the primary panel, and R3 MLP versus its
matched pooled control gives **+0.338713 [0.302942, 0.377253]**. Both learners
and both panels retain positive intervals for these controls. R3's absolute
performance remains below the sequence references; its positional benefit
alone does not make it competitive. R5 contact memberships versus shuffled
memberships remain unresolved: the MLP primary difference is **+0.000676
[−0.009556, 0.023590]**. Do not infer a biological contact-assignment advantage
or equivalence. Comparisons changing channels, context or pooling together
remain confounded. Every component and both panels remain in the saved tables.

The audit recomputed every saved pooled endpoint and all interval arithmetic;
it also checked three fresh deterministic cluster draws for every procedure
and panel. It did not repeat all 2,000 bootstrap computations. All 658 audited
evidence hashes match. These checks preserve the inherited cache wrapper
launch-provenance limitation and the explicitly amended sequence source
bindings; they do not establish independent scientific confirmation.

Both locked main-selector execution repeats are complete: **242 fresh fits**,
all ten fold/repeat cells, and all 19,716 predictions per execution. Primary,
repeat 1 and repeat 2 match exactly, including every reported endpoint on both
panels. This is recorded-setting, within-machine repeatability, not a claim
about selection reproducibility, transfer repeats, cross-platform equality or
assay noise. The [repeat audit](further_research/round2/resources/embedding_repeat_completion_audit.json)
records exact float32 cache/CSV reconstruction and distinguishes decimal
serialization rounding from execution differences.

### Completed complementarity development finding

The [audited complementarity report](further_research/round2/complementarity/results/REPORT.md)
retains all eight procedures and all 16 declared contrasts on both panels.
R4 blend scores **0.723789** on the primary panel, with difference **+0.056948
[0.042591, 0.083811]** against the competitive sequence selector, **+0.062268
[0.044337, 0.083869]** against the original reference, and **+0.039279
[0.022009, 0.056708]** against R4 ridge alone. The 50-row differences are
**+0.059101 [0.042081, 0.081825]**, **+0.062522 [0.044237, 0.081917]** and
**+0.040167 [0.024428, 0.056107]** respectively. The weight 0.5 was chosen
independently within each fitting fold.

R4 residual correction also improves these conditional endpoints: its primary
difference versus competitive sequence is **+0.044588 [0.030379, 0.067056]**,
and versus R4 ridge alone **+0.026919 [0.001338, 0.046971]**. Both sequence
comparisons and the embedding-alone comparison retain positive intervals in
the 50-row sensitivity. R2 blend and residual intervals cross zero against
both sequence references in both panels. Their large improvements over the
weak R2 ridge alone do not establish improvement over sequence.

No complementarity lower bound exceeds 0.05 against either sequence reference.
These secondary contrasts condition on fitted predictions and do not include
multiplicity/search uncertainty, execution variation or independent confirmation.
No direct R4-blend-versus-R5 or main-embedding-selector contrast was declared;
the point scores cannot establish that comparison. Complementarity procedures
are not added to the already frozen transfer nominations.

The audit checked the 2,140-member logical graph, all 30 nested ridge choices,
20 blend/correction choices and three exact zero endpoints. It recomputed all
pooled/per-allele/fold scores and 32 paired interval rows, checked every saved
reference-bootstrap row against the main report, and checked three fresh draws
for all eight arms and both panels. All 224 evidence hashes match, including
bootstrap/matched-prediction/report files omitted from the original report
manifest's seven-table output map. No frozen generator or numerical output was
rewritten. The inherited cache provenance limitation remains explicit.

### Declared implementation amendments

- A separate runtime-budget watchdog accounts for Modal infrastructure preemptions, which can restart inputs despite `retries=0` and omit lost-attempt compute from successful-worker telemetry. It adds a full reservation per observed preemption, leaves a maximum-concurrency burst cushion, and fixes latest stop deadlines at 06:42:28 London for sequence and 08:39:25 London for cache. The original $80/$18/$2 allocations, scientific files and study ledgers remain unchanged. [Operational amendment and immutable authorization](further_research/round2/OPERATIONS.md#preemption-accounting-amendment-12-october-2026) record the derivation and failure behavior; 13 mock-only watchdog tests pass. These safeguards can stop an incomplete study rather than expand spending.
- Original assay rows, splits, labels and all original/Phase A–D numerical artifacts are unchanged. All new outputs are isolated under `further_research/round2/` and a separate Modal volume.
- R3 uses a fixed, label-independent orthonormal Gaussian projection to eight channels, preserving every peptide/groove position. R5's 9×34 control uses four channels. The frozen cache retains full cropped channels, so a projected-head null cannot exclude richer channel use.
- Contact membership is the union of heavy-atom contacts within 4.5 Å from five existing crystal structures, restricted to the canonical 34 HLA positions. Both pair directions are retained. R5 shuffled membership changes the constrained information, rather than only permuting unrestricted columns.
- `R3_pooled` and `R5_position9x34_pooled` have exactly the same dimension, channel projection, context and head as their respective positional arms, with within-block pooling tiled back to positions. These secondary controls are excluded from the primary embedding selector. R3-versus-R2 and full-position-versus-R5 contrasts remain confounded by other representation differences and are labelled accordingly.
- Each embedding learner selects its own regularization. Ridge is deterministic and fitted once per cell; duplicating it for nominal seeds would not add execution evidence. Final neural heads use 30 members. The primary embedding selector retains the original baseline-only option; it never chooses by outer outcomes.
- Ranking procedures are compared using the exact original pooled per-allele statistic and paired whole-cluster resampling. The optimized count-weighted bootstrap has been checked against expanded-row resampling, including ties and constant predictions.
- The source publication describes geometric means from two independent experiments, but replicate-level observations and uncertainty are absent from the supplied corpus. This does not establish an assay noise ceiling.

The cache closeout found one provenance limitation: the current fanout wrapper digest differs from its recorded authorization digest, already in the first implementation commit. Its exact launch-time bytes cannot be certified from that record. The four frozen extraction sources, cohort and feature protocol hashes match; both wrapper digests are preserved in the audit without rewriting authorization. This limits the wrapper launch-provenance claim, while direct feature/artifact validation remains the cache acceptance check.

Both authorized cloud scopes have stopped; the local watchdog was deliberately shut down after terminal-state and settlement checks. The [cloud closeout](further_research/round2/resources/cloud_closeout.json) records **$76.8388 conservative accounting**, including the full $2 engineering allowance, against the unchanged $100 ceiling. The provider snapshot shows $70.2872 for named study apps and a $73.4426 account-wide increase; these different metering snapshots do not exactly reconcile and are not a final invoice or remaining-credit balance. The difference is retained as unattributed. Unused allocations are not reassigned. All bounded local assessment and reporting are now complete, with no new Modal compute. The later [app inventory](further_research/round2/resources/modal_final_app_inventory.json) lists no apps; historical shutdown receipts remain the evidence for the earlier normal stops.

Budget records and environment/source hashes live under [round-two resources](further_research/round2/resources/). The provider's historical October compute records total approximately $116.65 before the new study; the older $212.71 runtime-derived estimate is not a provider invoice and must not be treated as credit consumption. New usage is tracked separately.

[Execution operations and recovery](further_research/round2/OPERATIONS.md) record durable controller paths and the 30-minute quiet-unless-actionable chat continuation. Transfer/refit preparation has passed 32 focused tests and independent statistical review; the subsequent saved-specification integrity repair passed four agent regressions and five independently rerun freeze/nomination checks. The saved specification must match both its digest and a regenerated specification, including on reuse. No transfer outcomes were opened before the candidate decision. The completed main audit now permits the dated amendment and hash-bound freeze; the separate controller still gates timing and remaining-cell launches.

A separate [execution-repeat implementation draft](further_research/round2/reproducibility/embedding_execution_protocol_draft.md) replays every locked per-fold embedding-selector decision twice, with fresh fit caches and at most 300 new local fits. The completed main gate passed; its [frozen protocol](further_research/round2/reproducibility/embedding_selector/protocol.json) locks 242 fresh fits. Execution ran from 07:27 to 07:39 UTC; the [repeat audit](further_research/round2/resources/embedding_repeat_completion_audit.json) passed all ten completed fold/repeat cells, fresh caches, member/ancestry/scaler checks and saved metrics. Both complete-cohort repeats exactly match the primary predictions; all reported metric differences are zero on both panels. The process and keep-awake helper have exited. It does not choose an outer-outcome winner or repeat selection.

### Transfer scope amendment — 12 October 2026, after the main screen

The complete main report was generated at 07:19 UTC. Subject to its independent
completion audit passing, nominate **R4 ridge and R5 MLP** for the bounded
transfer assessment. They are the two procedures actually chosen by the frozen
main inner selector: R5 MLP in folds 0, 1, 2 and 4, and R4 ridge in fold 3. This
preserves every realised selector branch and both linear and neural heads.
It is a post-main, development-selected shortlist, not a prospectively
independent finalist selection or the two largest pooled outer scores.
R4 MLP is omitted to keep the two-procedure bound, not because it was shown
inferior in transfer. The full 9×34 secondary control is outside the main
primary selector and is not promoted into transfer after viewing its score.

Retain each nominee's **entire frozen main regularization grid**: ridge alpha
{10000, 100, 1} and neural L2 {1, 0.01, 0.0001}. Reselect inside each transfer
fitting partition; do not carry a favourable main-fold regularization into a
new regime. All five HLA folds and all 25 joint intersections remain required.
The complete graph is **13,530 local fits**: 10,440 sequence fits, 750 R4 ridge
fits and 2,340 R5 neural fits. No cloud work, feature recomputation or grid
expansion is added.

The transfer primary comparison is the **bounded embedding-or-MLP selector
versus the bounded MLP sequence selector**, separately in HLA and joint
holdout. The latter chooses between the original reference and centred MLP
with k {100, 20}. It excludes both tree regressors and PairLogit from the
full main sequence selector. This retains the two sequence families actually
selected in the main screen, but does **not** establish equivalence to the
full sequence selector or superiority over that full selector on unseen HLAs.
The transfer embedding selector may choose either sequence family or either
nominee. This differs from the main embedding selector's original-only
fallback and is explicitly part of this bounded transfer procedure.

Use regime-matched inner holdout, stopping, gradient-only centring and feature
scaling. Choose by pooled inner per-allele median Spearman with frozen
parsimony order for exact ties. This replaces the main mean-fold/one-SE rule
because sparse joint inner cells may lack adequate per-cell panels. Joint
gradient rows exclude both held dimensions and cross-quadrants; the five
diagonal stopping intersections do not cover every fitting row, a retained
limitation. Publish every component and both selectors against the original
reference, both 20/50-row panels, all declared peptide/HLA/crossed uncertainty
and every HLA-group deletion sensitivity. Do not drop an unfavourable cell,
panel or uncertainty mode.

Before any transfer fitting, freeze the amended specification and source,
data, feature and main-completion hashes. Authorize a deterministic timing
phase consisting of **HLA h0, then joint h0_p0, with one worker**. These cells
are chosen by index, before any transfer outcome. Inspect only runtime and
completion telemetry to forecast the remaining 28 cells. A separate immutable
authorization must bind the frozen protocol, controller source, successful
timing-cell markers, exact remaining cells and at most three workers. The
timing phase and remaining phase share a single owner lock; failed or uncertain
attempts require diagnosis rather than automatic retries. The user receives
an honest 11 AM London checkpoint even if local fitting is still running.

The main selector's two identical-setting execution repeats are separately
frozen at **242 fresh local fits**, replaying its locked per-fold decisions.
They measure execution of those fitted recipes, not reproducibility of
selection, every standalone nominee on every fold, the centred comparator in
every regime, or the new transfer procedures. Sequence reference repeats were
already completed. No assay-repeat or independent-confirmation claim follows
from numerical repeatability.

The amended transfer protocol froze at 07:37 UTC, after the main audit and
before any transfer fitting. The independently reviewed controller passed
37 mocked authorization/failure tests, including cleanup when status writes
fail. Its one-worker timing phase launched at 07:42 UTC; that authorization
covered only the two timing cells. [Timing authorization](further_research/round2/transfer/timing_authorization.json)
and [launch receipt](further_research/round2/transfer/controller/timing_controller_launch.json)
bind exact controller/protocol hashes and the two fixed cells.

Both deterministic transfer timing cells completed successfully: 226 HLA fits
in 485.8 seconds and 496 joint fits in 300.6 seconds of observed cell wall time.
A [timing-only forecast](further_research/round2/transfer/timing_forecast.json)
uses row-count scaling, a 2× timing contingency and a separate 15-minute report
allowance; it is a scheduling estimate rather than a guaranteed deadline.
All remaining 28 cells (12,808 fits) launched at 07:58 UTC under the
[immutable remaining authorization](further_research/round2/transfer/remaining_authorization.json),
with three workers and no new Modal work. The previous main local pipeline and
timing controller have exited. Transfer outcomes have not been used to alter
the scope. The 11 AM London checkpoint remains an honest progress report if
any fitting or final reporting is unfinished.

The [resource summary](further_research/round2/resources/runtime_cost_summary.json)
records shared extraction, fitting and reporting costs from 337 hashed sources.
The full cache's $66.8431 conservative accounting is shared across arms, about
$0.00339 per cached training row. Its recorded synchronized extraction span
averages 7.21 seconds per case; aggregate parallel wall time and those per-case
spans are different quantities. Cached MSAs and checkpoint reuse, startup,
storage and downstream head prediction prevent interpreting this as a measured
end-to-end latency or price for a new input. No isolated serving benchmark or
live credit balance is claimed. Recorded model-training timings and logical
cache-reused fits are explicitly separated from inference and new hardware work.


### Completed transfer development finding and execution closeout — 12 October 2026

All **five HLA-group and 25 joint peptide/HLA cells** completed without retries
or exclusions at 09:06 UTC. All **13,530 members** and cell artifact hashes
passed the completion gate; the frozen report ran once and finished at 09:10 UTC.
The [transfer audit](further_research/round2/resources/transfer_completion_audit.json)
covers every member's identity/configuration/ancestry, gradient-only centring and
scaler ancestry, saved inner decision replay, full row coverage, all endpoints,
108 comparison rows, 12 saved 2,000-draw tables and 1,260 group-deletion rows.
It verifies interval arithmetic and selected fresh draw probes, not a full new
bootstrap or refit. Raw inner prediction vectors were not retained, so saved
selection scores cannot be independently recomputed without new fits; numerical
scaler values were checked for dimensions/validity and exact fitting ancestry,
not recalculated from every training matrix. These audit limits remain explicit.

The primary transfer comparison is the **bounded embedding-or-MLP selector
versus the bounded MLP sequence selector**; this is a different comparator and
selector scope from the main screen. The primary ≥20-row panel gives:

| Regime | Bounded MLP selector | Embedding selector | Difference | Conditional crossed 95% interval |
|---|---:|---:|---:|---:|
| HLA-group holdout | 0.173961 | 0.546303 | +0.372342 | [0.290338, 0.453269] |
| Joint peptide/HLA holdout | 0.110831 | 0.506675 | +0.395844 | [0.295075, 0.445743] |

The ≥50-row sensitivity gives HLA **+0.371665 [0.288564, 0.455018]**
and joint **+0.394712 [0.292501, 0.444975]**. Against the original
sequence reference, ≥20-row differences are HLA **+0.375368
[0.283444, 0.455040]** and joint **+0.371689 [0.278298, 0.436636]**.
All these conditional lower bounds exceed 0.05. This does **not** revise the
main peptide result's failure to establish its own worthwhile-gain bar.

Every declared component/contrast, both panels and peptide/HLA/crossed intervals
are retained in [the transfer tables](further_research/round2/transfer/results/comparisons.csv).
All 35 group deletions preserve positive primary differences: ≥20-row ranges
are +0.354967 to +0.380237 for HLA and +0.372950 to +0.404739 for joint.
All 108 point estimates lie inside their intervals; no bootstrap draw loses the
entire eligible panel. Cell diagnostics favor the embedding selector over its
bounded sequence comparator in all five HLA and all 25 joint cells on both
panels. These are sensitivity diagnostics, not independent tests or a reason
to change the pooled endpoint. Inner choices select R4 ridge in four of five
HLA and 23 of 25 joint cells, with R5 MLP in the others. No global outer-score
winner is substituted for the frozen selector.

These large gains compare **complete procedures**. Ridge uses all permitted
fitting rows; neural members reserve stopping rows and, for joint stopping,
exclude cross quadrants. Representation, training recipe and training size can
all contribute. The transfer comparator omits trees/PairLogit; this is not a
claim of beating the full main sequence selector on unseen HLA groups. Joint
pooled within-allele rankings combine five different models, including their
between-fold scale behavior. The cell diagnostics argue against pooling alone
explaining the advantage; no outcome-driven recalibration was performed.
The grouping is a declared Hamming-distance convention, not a validated
functional distance. The 69/68 eligible alleles cover 34 of 35 represented HLA
groups; sparse coverage and conditional crossed resampling do not establish
performance across arbitrary future HLA populations. Main locked repeats do
not measure transfer-regime execution variation. Development reuse and the
inherited cache wrapper provenance gap remain limitations.

The [transfer runtime closeout](further_research/round2/resources/transfer_runtime_closeout.json)
records 84 minutes 2 seconds of fitting operational wall span, peak concurrency
three and 137.6 seconds for reporting, with **zero additional Modal compute**.
The earlier resource summary remains a timing-phase snapshot; the new closeout
adds final actuals rather than rewriting that historical record. All local
controllers, reporting processes and keep-awake helpers have exited.

The [consolidated development report](further_research/round2/REPORT.md) is the
finished readout. All achievable bounded development work is complete;
**independent confirmation is not complete**, because no adequate reserved
cohort or assay arrangement exists. External candidate outcomes remain
reserved. Optional calibration/adaptation and new acquisition are not silently
substituted. The original labels, splits, test and historical numerical
artifacts remain unchanged.


## 13. Newly authorized hackathon finalisation — 12 October 2026

After round two closed in commit `3cf7443`, the user requested autonomous
project finalisation, then reported **$143 remaining Modal credit** and
authorized **up to $100 additional** for this new phase. This is a separate
ceiling; old studies, authorizations and accounting remain closed. It is not a
spending target. The completed cloud pilot carries $1.3895623 conservatively,
including its retained build allowance and the higher historical guard snapshot.
The app is stopped; its reconciled journal is $1.3589062, with zero reservations.
Provider metering can lag and is not an account balance.

The user proposed combining Boltz embeddings with ranking loss. Freeze a
bounded, matched sequence/Boltz representation by regression/ranking-loss
comparison before fitting. Also price a defensible extension of the narrower
transfer controls. Design, implementation tests, fit counts and measured cost
gates precede any paid fanout. Neither a favorable development outcome nor
remaining budget automatically authorizes a larger grid or another transfer
phase. Original test data remain excluded.

Final delivery includes a portable prediction package, real cached-head timing,
updated project site, current root documentation and submission brief. Assay
feasibility will be packaged as a concrete request, without contacting vendors
or claiming unavailable confirmation. Bounded implementation uses 6.1 Sol;
complex scientific scope/review uses Astra. The [phase execution plan](finalization/PLAN.md)
and [durable manifest](finalization/execution_manifest.json) record operational
ownership, frozen follow-up protocols, budget guards and completion. Earlier
closeout source hashes remain historical snapshots of commit `3cf7443`; this
dated new authorization intentionally updates the living canonical plan.

### Completed ranking follow-up and pilot gates

All 840 new ranking fits, 480 logical reused MSE members and two exact lambda-zero compatibility checks are audited. Boltz R5 mixed ranking minus MSE improves the development endpoint by +0.00707, with paired 95% interval [−0.00505,+0.00983]; the gain is not established. The matched representation-by-loss interaction is also uncertain. The inner deployment rule retains MSE in all five R5 folds. See [ranking report](finalization/ranking/REPORT.md) and [completion audit](finalization/ranking/completion_audit.json).

The stronger transfer extension remains pending: exactly 30 cells and 10,830 new fits, including 1,620 bounded cloud ranker members. Two cloud pilot calls and two local tree timing fits completed without scoring held outcomes. A documented watchdog lifecycle repair preserves the original pilot protocol and scientific scope. Main launch requires the recorded runtime forecast and fixed resource authorization; no repeated pilot or enlarged grid is implied. Portable development heads and the unsent independent-assay brief are complete.

The main transfer extension launched once at 13:32 UTC on 12 October under the fixed $65 cumulative allocation and independent watchdog. Its immutable deadline is 15:32:19 UTC on 13 October; forecast completion is roughly ten hours, not guaranteed. Current exact ownership is in the phase manifest. No source changes, grid expansion, deadline extension or blind retries are authorized. The later Boltz/tree-ranking and tree/MLP hybrid suggestion is recorded as an unexecuted follow-up, separate from these sequence-input transfer controls.

### User-authorized hybrid follow-up

The user subsequently asked to orchestrate all proposed Boltz tree/ranking, tree–MLP blend, tree-leaf preprocessing and alternative preprocessing experiments. Their status is now authorized for bounded design and execution under a separate frozen protocol and runtime gate, replacing the earlier unexecuted-only proposal status. The active transfer scope is unchanged. The phase still has the same $100 incremental Modal ceiling: $65 reserved for transfer and at most $35 additional for hybrids. Prefer queued local execution after current fitting exits. See [hybrid authorization](finalization/hybrids/user_authorization.json) and the phase plan.

### Operational interruption and recovery amendment — 12 October

The transfer controller, watchdog and waiting hybrid queue disappeared without
exit receipts at about14:36 UTC; the cause is unknown. The cloud app was
confirmed stopped at14:39:08 UTC, with zero tasks. Eighteen complete transfer
cells are preserved. The [reviewed amendment](finalization/resources/transfer_main/interruption_20261012/implementation_amendment.json)
permits one separately tested local recovery of the twelve unfinished cells,
using unchanged scientific recipes and the original absolute deadline, with
zero additional cloud spend. All partial attempts remain archived and their
discarded/uncertain fit counts must be reported. The hybrid scientific design
is unchanged; its operational gate must explicitly bind the interrupted-scope
closure and successful recovered30-cell assessment, without fabricating
missing original success receipts. See the dated recovery section of the
[phase plan](finalization/PLAN.md).

The local recovery subsequently received SIGTERM at15:30:54 UTC during a
confirmed Codex restart/update. It cleaned up its workers and recorded a failed
exit; no fresh cell completed. An additional1,086 completed partial fits are
discarded, bringing the known discarded count to1,785. The separately recorded
second amendment requires a one-time operating-system-managed controller and
validated complete-call checkpoints before another attempt. Scientific scope,
original deadline and zero additional cloud allocation remain unchanged.
Both hybrid waiting queues stopped before fitting. Current recovery ownership
and its revised gates are recorded in the phase manifest and recovery operations.

### Scientific priority and future compute placement — 12 October

The user requests Modal for future heavy work and emphasizes scientific results
over further execution infrastructure. The existing bounded transfer continues;
subsequent reporting and the frozen 12-pilot/480-scientific-fit hybrid comparison
move to four sequential, capped Modal phases. The reviewed operational amendment
preserves all scientific sources and comparisons, with at most $35 inclusive
future spend inside the unchanged $100 ceiling. Exact phase authorization still
requires successful transfer closure. Finish these experiments and interpretation
without expanding either the model grid or the execution framework.
See the [continuation](finalization/cloud/CONTINUATION.md) and
[reviewed amendment](finalization/cloud/implementation_amendment.json).

### Transfer completion on Modal — 12 October 18:35 UTC

Disk cleanup removed the active local environment and interrupted recovery
after 25 complete cells. Preserve those 8,830 fits; the final five cells run
unchanged on Modal under a separate $10 cap. Two zero-fit startup failures
were reconciled before the current launch. The existing $35 finalization
envelope and the user's $100 additional ceiling remain unchanged. Exact
ownership and fixed deadlines are in the [recovery operations](finalization/RECOVERY_OPERATIONS.md).
This is an operational amendment only; finish the frozen scientific comparisons
without expanding the experiment grid or orchestration framework.
