# Further research plan: improving peptide–HLA half-life prediction

Date: 4 October 2026. Status: proposed follow-up protocol; no experiments launched.

This plan integrates Claude's supplied suggestions, the existing experiment records, and the joint peptide–HLA embedding discussion. It does not change the frozen benchmark, its splits, or its published conclusions. Priorities reflect expected information gained per unit of effort; they are not measured probabilities of success.

## 1. Recommendation

Run a bounded CPU study of **boosted trees, allele-centred targets, and within-allele ranking objectives**, then test prediction blends. In parallel research planning, make **joint interaction-pretrained peptide–HLA representations** the principal GPU direction. Revisit censored models as a separate calibration question. Reuse existing structural features before commissioning more structures.

The original reference remains the peptide+pseudosequence ensemble: validation median per-allele Spearman 0.6931, test 0.7064. New arms must also be compared against the strongest new sequence-only control, so a better learner is not mistaken for a benefit from embeddings or structure.

## 2. What to adopt, qualify, or defer

| Suggestion | Decision and evidence |
|---|---|
| Gradient-boosted trees | High priority. No implementation was found in the current model scripts. A different learner could improve either standalone prediction or ensemble complementarity. General tabular benchmarks motivate this, but do not establish superiority on biological sequence encodings. |
| Per-allele target centring | High priority as a cheap, controlled ablation. Current MLP centring is global. Fit allele means exclusively inside each training fold, with shrinkage for sparse alleles. Adding back an allele mean cannot itself change within-allele ranks; any benefit must come from changed training. |
| Ranking objective | High priority. Pairwise ranking is a surrogate, not an exact optimisation of median Spearman. Test allele balancing separately from changing the loss. |
| Censored loss with a different stopping rule | Worth following up for calibration. The reduced-budget control recovered ranking to 0.6910 versus its matched MSE reference at 0.6975, and improved Brier/ECE. It has no paired interval and worsened censored NLL; this is not an established ranking improvement or universal calibration improvement. |
| Two-part floor model | Worth a bounded follow-up after the simpler objective experiments. Distinguish the probability of an assay-floor code from the latent physical lifetime. |
| Joint interface embeddings | Principal GPU direction. Our failed shallow attention head did not test an interaction-pretrained backbone with cross-chain attention throughout its layers. |
| Likelihood features / larger or different pLMs | Secondary ablations. Cached unmasked embeddings do not provide masked pseudo-likelihood for free. Size, family, adaptation and interaction modelling should not all change in one experiment. |
| Chimeric peptide–linker–HLA sequence | Low-priority control. A synthetic linker creates an artificial single-chain context; it does not establish a physical interface. Prefer an encoder trained to represent separate interacting chains. |
| Anchor-focused geometry | Cheap ablation on existing features; include it in structural residual screening. Preserve a full-feature control because central peptide positions can still carry biology. |
| Multiple predicted poses | Exploratory uncertainty features, not validated molecular flexibility or dissociation kinetics. Start with existing multi-seed pilots; any new pilot must include both fitting and held-out training-cluster subsets. |
| Peptide–HLA pair ipTM | Already available and selected by the structural pipeline. The claim that Boltz lacks it is stale: production CSVs contain both directions, and `CONFIDENCE_PREFIXES` includes `iptm_pair_`. No new folding is justified merely to obtain it. |
| FoldX/Rosetta/energy features | Defer on cost and uncertain incremental value, not because equilibrium features are mathematically incapable of predicting kinetic labels. FoldX predictive performance remains unmeasured here. |

Three interpretation corrections apply throughout:

- **+0.05 is the project's worthwhile-effect threshold, not a universal detection limit.** Precision depends on paired errors and the actual interval; smaller effects can be statistically distinguishable. The existing verdict table already allows this.
- **Zero initialisation preserves baseline predictions only initially.** A fitted residual can still damage generalisation. Include zero correction as an explicit model choice and evaluate on held-out clusters.
- **Do not budget around an assumed 0.2 of recoverable headroom.** Concordance between distinct alleles is not replicate reliability, and subtracting our median score from the best cross-allele correlation does not establish attainable improvement. Replicates are needed.

Evidence: [main report](REPORT.md), [censored control §6.4](stage7_censored.md#64-where-the-ranking-loss-actually-comes-from--the-stopping-rule-not-the-gradient), [structural feature provenance](stage4c5_features.md), [attention comparison](stage3e_comparisons.csv), [limitations](limitations.md), [feature implementation](../pepstab/structural_features.py), [structural predictor](../scripts/stage5_structural_arm.py).

## 3. Evaluation protocol for all phases

1. **Keep the original train/validation/test assignments and peptide clusters.** The old test is already exposed and will not be used for development. The repeatedly consulted validation set is a development resource, not fresh confirmation. Leave the original evaluation contract unchanged; this document defines a follow-up study.
2. **Use training-only nested cluster validation.** Use the existing five training-cluster folds for outer assessment. Early stopping and tuning must use inner cluster folds inside the outer fitting partition. An outer held-out label must not choose epochs, hyperparameters, feature transforms, target means, or blend weights. Existing early-stopping-fold predictions are not automatically clean stacking inputs.
3. **Keep search bounded.** Start with at most 12 prespecified configurations per new learner/objective family, two paired seeds and three inner cluster folds for screening. Run the full outer assessment for finalists. Record the search spaces before execution; log extensions rather than silently expanding a disappointing arm's budget.
4. **Separate screening from final ensembles.** Use matched seed/fold budgets for neural ablations. For tree-versus-MLP comparisons report both tuning effort and training/inference cost; 30 boosted fits do not imply the same capacity as 30 MLPs. Finalists should receive a declared bagging budget comparable to the reference, with diminishing returns checked inside training only. Do not give an arbitrary tree-count cap the meaning of architectural fairness.
5. **Evaluate ranking and absolute predictions separately.** Primary: median per-allele Spearman, with the existing tie and eligibility rules where applicable. On smaller internal folds, freeze an eligibility rule based on labels/counts before fitting and report coverage. Secondary: log1p MAE, precision@10, per-allele distribution, distance strata and inference cost. Pure rankers need a training-only calibration map before reporting hour-scale or log1p prediction error.
6. **Use paired peptide-cluster bootstrap intervals.** Retain the existing 2,000-resample reporting procedure. Report every prespecified finalist and its selection history; a bootstrap interval on a selected development winner does not remove model-selection optimism. Use a simple blend with nonnegative weights summing to one, selected inside the outer training partition using inner out-of-fold predictions.
7. **Freeze one shortlist before reading external validation again.** Report comparisons to the frozen baseline and the best new sequence control. Distinguish positive point estimates, evidence of any gain (lower interval bound >0), and evidence meeting the existing worthwhile bar (lower bound >0.05). A small complementary gain can justify a blend experiment without being declared a meaningful standalone win.
8. **Final confirmation needs new evidence.** Obtain independent measurements with cluster-overlap checks or a prospectively reserved external cohort. Repartitioning previously examined rows cannot create an untouched test. A second old-test pass, if later requested, must be labelled exploratory.

## 4. Phase A — bounded CPU experiments

| ID | Experiment | What it isolates | Next-step criterion |
|---|---|---|---|
| A0 | Reproduce the sequence reference and generate nested out-of-fold predictions | Reliable reference, not a new model | Predictions, metrics and fold provenance reconcile |
| A1 | Boosted-tree regression on position-preserving peptide+pseudosequence features | Learner family | Improves outer-fold ranking or adds useful complementary predictions |
| A2 | Sequence MLP on allele-centred targets versus original targets | Target parameterisation | Consistent outer-fold gain; no validation-derived means |
| A3 | MLP MSE, MSE+pairwise ranking; each with and without allele-balanced sampling | Objective versus sampling | Ranking gain survives paired controls and fold/seed variation |
| A4 | Boosted-tree ranker on the A1 features, with allele as the ranking group | Whether objective gains transfer across learners | Beats matched tree regressor or improves the blend |
| A5 | Baseline+tree, baseline+ESM, and a small combined blend | Complementarity | Improves outer assessment with weights fitted only on inner predictions |

Choose one tree library initially (CatBoost is a practical candidate for both regression and ranking). Start with the existing one-hot/BLOSUM positional encodings. A categorical residue-per-position encoding is a separate optional ablation; do not use arbitrary numeric amino-acid IDs as ordered measurements. Avoid allele IDs in the pan-allele main arm; allele-specific models belong in a separately labelled seen-allele analysis.

For A2, shrink each fitting-fold allele mean toward the fitting-fold global mean. Use a prespecified small shrinkage grid in inner validation; unseen alleles fall back to the shared estimate. Add the mean back for prediction on the log1p scale. This tests training efficiency, not a guaranteed removal of harmful variance.

For A3/A4, form ranking pairs only within the same allele. Skip tied targets, including floor–floor pairs; avoid claiming a known ordering for pairs whose assay intervals overlap. Normalise pair contributions per allele so alleles with many measurements do not dominate quadratically. Retain MSE as an auxiliary objective when calibrated magnitudes matter. Compare MSE-based and ranking-based early stopping explicitly rather than attributing a changed stopping rule to the loss.

**Deliverable:** a single experiment table with all arms, paired deltas, cost, coverage, and prediction provenance. Do not start by stacking every change: first identify which changes help independently, then test their combination with component-removal ablations.

## 5. Phase B — assay-floor modelling and calibration

Complete the existing censored-loss stopping comparison at a matched finalist budget: MSE reference; censored loss stopped on NLL; censored loss stopped on MSE. A ranking-stopped censored arm is optional only if declared before this phase starts.

Report probability calibration specifically for the floor event, with paired Brier intervals, fixed-bin ECE and reliability plots. Also report censored NLL and ranking, since these previously disagreed. Evaluate a training-only post-hoc calibrated sequence baseline so the added distributional model must beat a simpler calibration solution.

Then test a two-part model: probability of exceeding the fixed assay floor plus conditional magnitude above it. For prediction of recorded log1p labels, `p(above floor) * E[log1p(t) | above floor]` treats the floor-coded component as zero. This is an observed-label expectation, not an estimate of true sub-floor lifetime, and is not generally the optimal ranking score. Compare its ranking with the ordinary regressor. If estimating physical lifetime below the floor, retain a censored latent component rather than equating that lifetime to zero.

**Decision:** keep this branch if it improves a prespecified calibration objective with an acceptable ranking tradeoff chosen before fitting. Do not count calibration improvement toward the +0.05 ranking claim.

## 6. Phase C — joint peptide–HLA embeddings

Prioritise an interaction-pretrained, chain-aware backbone over adding another shallow attention layer to independently frozen embeddings.

1. Audit checkpoint training data and obtain an upstream checkpoint without supervised exposure to our held-out stability labels. Audit affinity training overlap at the peptide-cluster level as well. Released SPEARMINT stability weights are not a clean competitor on our source corpus.
2. Compare frozen separate-chain and frozen joint-chain representations with the same small downstream head and position-preserving peptide pooling. Record parameter count and pretraining differences; do not ascribe all differences to cross-attention when the pretraining also changes.
3. Compare frozen versus limited adaptation of the joint encoder. A matched masking/architecture ablation can test cross-chain attention, but inference-time masking alone is not a complete causal test of how pretraining helped.
4. Compare direct half-life adaptation with affinity-then-half-life adaptation under leakage-controlled upstream data. Our earlier auxiliary-affinity null does not settle sequential representation pretraining.
5. Test a baseline prediction blend or separately regularised residual correction. Compare against the best Phase A sequence learner as well as the original ensemble.

The SPEARMINT preprint supplies direct motivation: it reports better discovery-set pooled Spearman for MINT Transfer than ESM Transfer. That is not a forecast for our stricter splits or median per-allele metric. Use its architecture as a hypothesis, not its headline score as our expected result.

Alternative checkpoint families, larger ESM, chimeric inputs, and masked pseudo-likelihood are deferred until this comparison answers whether interaction-aware adaptation helps. Proper masked peptide pseudo-likelihood requires masked forward passes; any joint conditional score must use an encoder capable of conditioning on the partner chain.

**Gate:** perform an extraction/training smoke benchmark for memory, time and checkpoint access before specifying a full GPU budget. Stop scaling if the matched outer assessment excludes a worthwhile increment and the representation adds no useful blend signal.

## 7. Phase D — reuse structures before expanding compute

Fit a strongly shrunk residual model on sequence out-of-fold errors. Compare: no correction; existing geometry; existing confidence; anchor-focused geometry (P2/P9, with a prespecified P1 extension); and the combined block. Fit all feature transforms inside training. Both directions of peptide–HLA pair ipTM are already present; test ablation, not acquisition.

If this fails, an interface pair-embedding pilot is a distinct follow-up, conditional on verifying that the chosen structural model exposes those representations and what extraction costs. Do not assume intermediate embeddings were cached with coordinates.

For multi-pose features, first use existing three-seed pilot records to measure reproducibility and sample sufficiency. If expanding, prespecify a multi-allele panel drawn within training with separate cluster-held-out evaluation, and match the single-pose control on the same rows. Across-seed variance may reflect prediction ambiguity or sampling behaviour. It is not a thermodynamic ensemble or an unbinding rate. Genuine dynamics simulations remain a separate, more expensive study.

**Gate:** commission more structures only after defining the feature, matched comparator, panel coverage and budget. Small panels provide feasibility evidence; they may not have power to adjudicate the original primary metric.

## 8. Longer-term data and transfer work

Acquire independent half-life measurements across poorly covered alleles and peptide families, with a replicate subset and assay temperature/protocol metadata. Build learning curves using existing training clusters to guide acquisition. Reserve a representative prospective confirmation panel; disagreement-selected active-learning examples should not be the sole evaluation population.

If unseen-allele transfer becomes the main deployment objective, promote pseudosequence-conditioned hierarchical adjustments and distant-allele evaluation to a dedicated track. Improvements on the present seen-allele benchmark do not establish transfer gains.

## 9. Execution order and outputs

**First batch:** A0–A3, then A4/A5 for promising candidates. **Next:** censored calibration follow-up and existing-feature residual tests. **Main GPU investment:** Phase C. **Deferred:** new pose generation, broader model-family sweeps, and empirical energy production.

Save follow-up outputs under a new `reports/further_research/` namespace when execution begins: protocol/config manifest, split and checkpoint hashes, run table, nested predictions, paired comparisons, calibration diagnostics, cost measurements, and an updated decision log. Preserve original reports and prediction files.

External sources used to motivate the plan:

- [Grinsztajn et al., tabular learner benchmarks](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html): tree-model motivation, not task-specific evidence.
- [CatBoost ranking objectives](https://catboost.ai/docs/en/concepts/loss-functions-ranking): groupwise ranking capabilities; these objectives are not identical to median Spearman.
- [SPEARMINT preprint](https://www.biorxiv.org/content/10.64898/2026.06.28.735023v1.full): joint interaction modelling and sequential transfer for half-life prediction; different evaluation and overlapping source corpus.
- [Differentiable sorting and ranking](https://proceedings.mlr.press/v119/blondel20a.html): optional rank-surrogate methods after simple pairwise loss.
