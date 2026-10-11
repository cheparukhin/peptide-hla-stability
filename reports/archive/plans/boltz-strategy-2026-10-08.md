> **Superseded for execution — 8 October 2026.** The authoritative forward plan is [NEXT_RESEARCH_PLAN.md](../../NEXT_RESEARCH_PLAN.md). It incorporates this document's Boltz extraction, positional representations, centred-tree control and transfer design, together with completed follow-up evidence and evaluation repairs. The text below is retained as the earlier planning snapshot; its future-tense descriptions and ordering are not the current work queue. Historical experiments are governed by their dated phase prespecifications, not this later revision.

# Research plan: frozen Boltz-2 representations for peptide–HLA stability

Updated: 8 October 2026. Status: planning only; no new training, extraction or cloud jobs launched in this chat.

This revision supersedes the 4 October priorities. Frozen Boltz-2 embeddings are now the first representation experiment. The user's report that winning teams benefited from these embeddings and positional interface information motivates this change; their exact methods, splits and effect sizes have not yet been independently inspected. Reproduction and independent extension are separate objectives.

## 1. Objective and order of work

Determine whether frozen Boltz-2 representations improve half-life ranking over our sequence ensemble, whether explicit peptide–groove position information adds value, and under which peptide/HLA generalisation regimes those gains hold.

The primary sequence reference remains peptide+pseudosequence: validation median per-allele Spearman 0.6931, test 0.7064. These historical scores are context, not scores to compare against newly constructed split regimes. Every new regime needs its own refitted, matched baseline.

| Order | Work package | Output and completion gate |
|---|---|---|
| 0 | Freeze hypotheses, candidate representations and evaluation manifests | Versioned protocol with bounded search spaces and no evaluation-based split selection |
| 1 | Implement and verify an embedding-only Boltz extractor | Matching tensors against a full forward pass on a small panel; measured runtime, memory and storage |
| 2 | Cache training embeddings and fit simple predictors | Paired assessment against sequence controls and hand-crafted structural features |
| 3 | Test peptide-position and groove-contact organisation | Matched ablations identifying whether biological position information helps |
| 4 | Test peptide-held-out, HLA-held-out and joint-held-out transfer | Separate results and coverage for each regime; no mixing of headline scores |
| 5 | Combine independently useful improvements and confirm | Frozen final model, cost report, and a genuinely independent confirmation cohort |

CPU controls can run alongside engineering and extraction once execution begins. They are not prerequisites for testing Boltz embeddings. Censoring, dynamics, energy scoring and broad checkpoint sweeps are deferred.

## 2. What is known and what remains open

- The original Boltz experiment used geometry and confidence summaries, not internal embeddings. Its negative result does not answer the representation question.
- Production used `boltz==2.1.1`, a pinned checkpoint, cached MSAs, three requested recycling steps, and the HLA ectodomain + beta2m + peptide construct. The launch did not request embeddings, and the persistence code copied only CIF, confidence, PAE and pLDDT outputs. See [production pipeline](../../../modal_app/ectodomain_production.py).
- Inspection of the 2.1.1 source established that the final token representation `s` and pair representation `z` are computed before diffusion. A custom extraction path should therefore be able to omit coordinate generation and confidence scoring. Numerical equivalence and performance have not been tested. Current [upstream source](https://github.com/jwohlwend/boltz/blob/main/src/boltz/model/models/boltz2.py) also shows this ordering; implement against our pinned version, not an assumed equivalent latest version.
- Saved coordinates and confidence cannot reconstruct the missing internal activations. Cached MSAs avoid repeating sequence searches, but the trunk must be run again.
- The exact winning extraction layer, construct, recycling count, pooling, predictor, data exclusions and splits remain unknown. If code or a methods description becomes available, record these details before calling an experiment a reproduction. Otherwise proceed with the explicitly labelled trunk-embedding hypothesis.

## 3. Work package 0 — evaluation and provenance

Preserve `data/splits.csv`, the original labels, original prediction files and [EVALUATION.md](../../../EVALUATION.md). Use `pepstab.data.load_with_splits()`; `docs/SPLITS.md` describes a superseded split and must not drive this work.

Use the existing training partition for development. Construct nested training-cluster folds: outer folds assess generalisation; inner folds select hyperparameters, epoch counts, centring, dimensionality reduction and blend weights. An outer row must be held out from early stopping as well as fitting. Fit all supervised preprocessing inside the appropriate fitting partition. Frozen label-independent extraction may be cached once across these folds.

Before fitting, save the row IDs, peptide clusters, HLA groups, allele eligibility rules, input/checkpoint hashes, search spaces and seed list. Freeze the main peptide-held-out analysis first; the other regimes are prespecified transfer questions. The previously inspected validation and test data cannot become untouched evidence by repartitioning them.

Primary outcome: median per-allele Spearman, with average ranks for ties. Secondary outcomes: per-allele distribution, log1p MAE, precision@10, coverage, nearest-training-sequence distances and inference cost. Pure ranking models require training-only calibration before absolute-error reporting.

Retain +0.05 as the original worthwhile-effect threshold, not a universal detectability limit. Report paired differences and intervals. Distinguish a positive point estimate, evidence of some improvement (lower bound >0), and evidence clearing the original bar (lower bound >0.05). Smaller complementary gains remain useful candidates for blending.

## 4. Work package 1 — extraction feasibility before production

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

## 5. Work packages 2–3 — representation and positional ablations

Start with ridge and a small MLP, each with its own bounded regularisation search. The frozen representation is the variable of interest. Use the same inner/outer folds, seeds, target and tuning opportunities for matched comparisons. Record feature dimensions and head capacity; dimensionality reduction must be fit on training features only.

| ID | Representation | Question |
|---|---|---|
| R0 | Existing position-preserving peptide + HLA pseudosequence | Matched sequence control |
| R1 | Existing geometry/confidence features, with a matched sequence control | Link to the earlier structural result |
| R2 | Frozen Boltz token embeddings with simple chain-wise mean pooling | Does a coarse learned representation carry signal? |
| R3 | Frozen Boltz token embeddings retaining P1–P9 separately and fixed groove positions | Does keeping residue positions improve on pooling? |
| R4 | Frozen pair embeddings pooled over the groove separately for each peptide position | Does explicit interface representation add to token embeddings? |
| R5 | Position/contact-aware pair embedding summaries | Do specific peptide–pocket correspondences help beyond R4? |

For R5, begin with P1, P2 and P9 anchor-region summaries plus a separate P3–P8 block; do not discard the central peptide. Compare these with a position-indexed 9×34 interface representation reduced on its channel axis inside each fitting fold. Index membership alone does not specify which residues contact each peptide position: define the position-specific contact mask from a documented structural map independent of stability labels, or from our already predicted structures using one fixed geometric rule. Treat those as different masking methods and choose one before execution.

Use a matched-dimension control where practical so more columns are not confused with better positional organisation. Add a fixed shuffled position-to-pocket mapping control to test whether the selected correspondence matters. Predicted-contact masks consume coordinate information, so report that dependency rather than attributing the gain solely to trunk embeddings.

For a small shortlist, compare embeddings alone, a separately regularised correction to sequence predictions, and prediction blending. Residual targets and blend weights must use inner out-of-fold baseline predictions; zero correction/baseline-only must remain selectable. Zero initialisation is not a generalisation guarantee.

First-pass search budget: at most 12 prespecified configurations per learner/representation, two paired seeds, three inner folds; assess shortlisted configurations through five outer peptide-cluster folds. Keep all prespecified results, including failures. Broaden the search only for a documented unresolved optimisation issue. Do not test every possible feature/learner/target combination at once.

## 6. CPU control: learner × target formulation

Run the following 2×2 study on the same sequence features and cluster folds:

| Learner | Original log1p target | Allele-centred log1p target |
|---|---|---|
| Existing MLP | Reference | Centring ablation |
| Boosted-tree regression | Learner ablation | Combined learner + centring |

Fit shrunk allele means using fitting labels only; tune any shrinkage on inner folds. Add the mean back for absolute prediction. For unseen alleles, use a fitting-derived shared fallback, never held-out allele labels. Adding a constant does not change within-allele ranking; gains must arise from changed training. Centring will not automatically remove allele-dependent target spreads or censoring.

Use one tree library initially, position-preserving one-hot/BLOSUM features, and no numeric ordinal coding of amino acids. Compare the best Boltz arm against both the original ensemble and the strongest new sequence control. Report tuning compute and ensemble size; equal numbers of neural and boosted models do not imply equal capacity or cost.

Only after this study, test within-allele pairwise loss and allele-balanced sampling as separate ablations on a promising learner. Skip tied pairs, normalise contributions by allele, and retain a magnitude objective if calibrated lifetime matters. Revisit sequence–ESM and tree–MLP blending using proper nested predictions.

## 7. Work package 4 — peptide and HLA generalisation

Create auxiliary evaluation manifests inside the original training partition; do not overwrite the frozen split. Define HLA similarity groups from contact pseudosequences, selecting and recording a grouping rule from sequence distances and sample counts before model outcomes. Keep identical pseudosequences and closely related constructs together in group-held-out transfer analyses. Report engineered constructs explicitly.

| Regime | Fitting/evaluation separation | Interpretation |
|---|---|---|
| Peptide-held-out | Whole existing peptide clusters held out; HLA may be shared | New peptides for assayed alleles; main continuity benchmark |
| HLA-group-held-out | HLA groups held out; peptide clusters may be shared, with overlap quantified | Transfer to new HLA groups, potentially on familiar peptides |
| Joint-held-out | Both peptide cluster and HLA group absent from fitting | New peptides on new HLA groups |

For joint holdout, choose evaluation peptide groups P and HLA groups A. Fit only rows outside P AND outside A; assess the intersection P×A. The cross-quadrant rows must not re-enter fitting, tuning or early stopping for that evaluation. Mirror each regime in its inner splits. Use a limited prespecified set of fold combinations instead of choosing favourable partitions.

Audit cross-boundary sequence identities, nearest-neighbour distances, counts, censoring rates and eligible alleles. Group definitions use sequences/counts, not model scores. Keep eligibility identical across models. Sparse intersections may be underpowered: publish coverage and descriptive results rather than weakening holdout rules after seeing scores.

Refit every comparator under each regime. Use paired peptide-cluster bootstrap for the peptide-held-out panel. For transfer to an HLA population, also assess HLA-group uncertainty; the joint design needs a crossed peptide/HLA-group resampling analysis, with sensitivity reporting when groups are few. A row bootstrap or peptide-only interval does not establish uncertainty over unseen HLA groups.

## 8. Work package 5 — selection and confirmation

Freeze a shortlist and weights using training-only assessment, then consult the old validation set once for this follow-up package, explicitly as additional development evidence. Selected-model intervals do not erase repeated-search optimism. Preserve the original test result; any later old-test rerun is exploratory and must be labelled accordingly.

Advance an embedding method when its outer-fold performance or blend contribution is reproducible enough to justify further evaluation; report the actual effect and interval rather than requiring every exploratory step to clear +0.05. Conversely, consistent underperformance across reasonably tuned heads and positional summaries is evidence against this tested representation under that regime, not against all structural embeddings.

Final confirmation requires an independent, prospectively reserved or external half-life cohort with peptide/HLA overlap audits and assay metadata. Record possible upstream structural-pretraining exposure separately from supervised half-life leakage; do not claim none without an audit. Do not use released stability-trained checkpoints on overlapping benchmark data as clean comparators.

Success reporting must state which regime improved, against which baseline, with what uncertainty and added inference cost. A ranking gain confined to familiar HLAs is still useful, but is not evidence for unseen-HLA transfer.

## 9. Deliverables and deferred work

Save new outputs under `reports/further_research/` and embedding cache manifests under a separate versioned feature namespace. Required deliverables: protocol and split manifests; extractor plus equivalence report; cost/storage forecast; cache inventory; all run configurations; nested predictions; paired comparisons; position/contact ablations; transfer results; final decision log.

First implementation milestone: **an audited extractor, 8–16-case equivalence report, approximately 64-case throughput profile, and locked evaluation manifests**. These precede a full extraction launch. Wall-clock and monetary estimates follow that benchmark; this plan does not invent them.

Retain, but defer: censored-loss stopping-rule follow-up and two-part floor models for calibration; alternative interaction-pretrained encoders and limited fine-tuning if Boltz results leave a clear question; larger pLMs; multi-pose uncertainty features; FoldX/Rosetta; dynamics simulation. Pose-to-pose variation is not a measured kinetic ensemble. Energy features are deferred for priority/cost reasons, not declared incapable of predicting lifetime.

Longer term, collect diverse independent half-life labels with replicates and assay metadata. Do not infer recoverable headroom by subtracting our score from the best correlation between different alleles.

Supporting records: [main report](../../REPORT.md), [structural feature provenance](../../stage4c5_features.md), [censored follow-up evidence](../../stage7_censored.md), [previous HLA holdout](../../stage7_allele_holdout.md), [existing cross-attention comparison](../../stage3e_comparisons.csv).
