# Peptide-HLA stability: hackathon execution plan

**Team:** 3 participants
**Event:** London AI × Science, protein engineering track, October 3-4, 2026
**Research question:** can pretrained protein language model features predict how long a peptide stays bound to an HLA molecule better than raw sequence features — and is the improvement worth the compute cost?

Our main claim is about **unseen peptides on HLA alleles we trained on**. Whether the model works on entirely new alleles is a separate test, not the headline.

## Recommended scope

Commit to comparing a supervised sequence baseline against frozen ESM-2 features across the full dataset. Only add a structural experiment (predicted 3D shapes, confidence scores) if a small end-to-end pilot works first.

Do not commit to folding the whole dataset. A clear negative result ("pretrained features didn't help") is a successful submission. Structural work must not block the core comparison.

## Scientific rationale

HLA molecules sit on cell surfaces and display short protein fragments (peptides) so the immune system can inspect them. Our target is **residence time** — how long a peptide stays bound once the complex has formed, measured as a dissociation half-life.

This is different from binding affinity (how readily a peptide binds in the first place). A peptide might bind easily but leave quickly, or vice versa. So a good-looking predicted structure or a favourable energy score doesn't automatically mean a long half-life. We test these quantities as candidate *predictors* of the measured half-life, not as direct measurements of it.

The experiment ladder asks: does each additional source of information add value **beyond what a trained sequence baseline already captures**? We start with cheap full-dataset comparisons, only pay for expensive structure predictions after a small pilot works end to end, and then reuse those same structures to test geometry, confidence, sequence compatibility, and energy separately.

The baseline, ESM extraction, and folding pilot can run in parallel. The ordering below describes what evidence is needed before expanding scope, not a strict serial schedule.

| Experiment | Priority | Question | What goes in | What we measure |
|---|---|---|---|---|
| Sequence baseline | Core | What can labelled sequences alone tell us? This is the cheap reference point. | Peptide amino acids and HLA amino acids, preserving which position each residue sits at. | Peptide ranking accuracy, prediction error, training and inference cost. |
| Weak-binder augmentation | Optional, after stage 2 | Does broader negative coverage improve stability prediction, and does the source of negatives matter? | Measured weak-affinity pairs or random natural peptides with predicted weak affinity, assigned assumed zero-hour stability labels. | Gain over the measured-only baseline on unchanged validation examples; label-source effects, allele coverage, and cost. |
| Auxiliary affinity | Optional, after stage 2 | Does training on binding-affinity labels alongside stability labels improve the sequence baseline? The initial probe needs no ESM features. | A shared encoder with a second head predicting affinity (how strongly a peptide binds, not how long it stays). Tested first on dual-labelled pairs, then optionally on broader IEDB data and the ESM arm. | Gain over single-task training; whether the benefit differs between the sequence and ESM arms. |
| Frozen ESM-2 | Core | Does a pretrained protein language model add useful signal? Testable across the full dataset, no structures needed. | Learned vector representations of each peptide position and HLA sequence (from ESM-2), used alone and combined with raw sequence features. | Improvement over the sequence baseline; embedding extraction cost; sensitivity to which internal layer and model size we use. |
| ESMFold2 geometry + confidence | Pilot-gated, no MSA | Does a cheap single-sequence co-folded structure explain stability? Runs first because it needs no MSA. | Contacts and burial per peptide position; per-position pLDDT; the peptide-HLA PAE block; peptide-HLA ipTM. | Added accuracy over the sequence and ESM-2 arms; cost per structure; failure rate; whether the peptide is even placed in the groove. |
| Boltz-2 geometry | Pilot-gated, MSA panel | Does the predicted physical fit in the HLA groove explain stability? Small panel, because folding is expensive. | Contacts and burial (how deeply each peptide position sits in the groove); hydrogen bonds at the peptide ends; clashes at anchor pockets (positions where the peptide is pinned down). | Added accuracy on matched examples; whether the predicted poses look physically reasonable; cost and failure rate. |
| Boltz-2 confidence | Same structures | Does the model's own uncertainty carry signal? Tested separately from geometry, reusing the same predicted structures. | Per-position pLDDT plus its peptide mean and minimum (pLDDT = per-residue confidence in the local structure around that residue); peptide-HLA PAE (predicted alignment error — how sure the model is about the relative placement of the two chains); pairwise ipTM (interface predicted TM-score — overall interface quality). | Gain from confidence alone and beyond geometry. |
| ProteinMPNN | Out of scope this round | Is the peptide sequence "compatible" with the predicted backbone shape? Uses another pretrained model, no refolding needed. | Peptide-only overall likelihood and per-position scores from ProteinMPNN (an inverse-folding model that asks: given this 3D backbone, how probable is this amino acid sequence?), with HLA held fixed. | Not run this round — see the out-of-scope register. It needs the structures first, so it is the last thing to add, not a core arm. |

Each experiment must earn its place by improving prediction on the same held-out examples, with uncertainty and compute cost reported alongside accuracy. A useful negative result tells us which approach didn't help under these conditions — it doesn't rule out every use of that model family.

## Dataset and known facts

**Canonical input:** the corrected CSV committed under `data/`. Keep it unmodified alongside its checksum so every workstream uses identical data.

| Property | Value |
|---|---:|
| Unique peptide-HLA pairs | 28,166 |
| Unique peptides | 5,633 |
| Unique HLA domain sequences | 75 |
| Peptide length | 9 residues |
| Supplied HLA domain length | 182 residues |
| HLA contact pseudosequence length | 34 residues |
| Duplicate peptide-allele pairs | 0 |
| Zero-hour labels | 5,679 (20.2%) |
| Median nearest-neighbour distance between peptides | 4 substitutions |
| Peptides with any neighbour within 3 substitutions | 871 (15.5%) |
| Largest single-linkage cluster at Hamming ≤ 3 | 22 peptides (157 pairs, 0.56%) |

Inputs are `peptide`, `hla_seq`, and `hla_pseudoseq`; the target `thalf_hours` is dissociation half-life.

The 34-position contact pseudosequence identifies which HLA positions are likely to touch the peptide, but gives no 3D geometry for any specific peptide-HLA pair. Individual experimental replicates aren't included, so we can't estimate a noise ceiling from the data alone. The released NetMHCstabpan model was trained on this same dataset, so it's not a fair held-out comparison.

## Stages, deliverables, and justification

### 1. Audit the data and freeze evaluation

**Work**

- Check sequence alphabets, pseudosequence lengths, engineered constructs, missing values, and label distributions.
- Start with `y = log1p(thalf_hours)` as the prediction target — this handles zero-valued labels cleanly. Keep original labels for reporting.
- Investigate what zero values and any apparent assay limits actually mean. Don't assume they're censored based on repeated values alone, and don't automatically discard them.
- Freeze approximately **70/10/20** train/validation/test splits. Group peptides by **single-linkage clustering at Hamming distance ≤ 3** and keep every cluster wholly within one split, across all alleles. All peptides are 9 residues, so Hamming distance is exact and needs no alignment. Check per-allele counts before fitting.
- Use plain Hamming distance, not a BLOSUM-weighted or embedding-based metric. At a matched threshold BLOSUM62 reproduces the same partition (5,493 vs 5,494 clusters at Hamming ≤ 1) while adding a threshold to defend, and a protein-language-model distance would make the split depend on ESM-2 — the model under test at stage 3. The split must stay model-independent.
- Agree on metrics, model-selection rules, **a predeclared threshold for what counts as a meaningful improvement**, and shared example IDs. Don't touch the test set until the final comparison.

**Deliverable:** audit summary, saved split assignments, and a shared evaluation script.

**Status: done.** `reports/audit_summary.md` (audit), `data/splits.csv` (frozen splits, committed; carries `cluster_id`, `split`, and `dist_to_train` per `pair_id`), `EVALUATION.md` (the predeclared contract), `pepstab/` + `scripts/evaluate.py` (shared scoring), `tests/test_contract.py` (51 guards). Load splits with `pepstab.data.load_with_splits()`; never recompute them.

Two audit findings constrain later stages:

- **`HLA-B*14:01(C67S)` and `HLA-B*14:02(C67S)` share one contact pseudosequence** (756 pairs, 2.7%). Pseudosequence-only models cannot separate them, which caps the stage 2 pseudosequence arm. The full-domain baseline can.
- **The 20.2% of labels at exactly 0 are left-censored at the assay floor**, on three independent lines of evidence (a trough at 0.1 h holding 27% of the extrapolated count; per-allele zero share anticorrelated with that allele's median non-zero label, Spearman -0.608, p=3.7e-8; zeros spread across peptides). Keep the rows, train on `log1p`, and lead with rank metrics. MAE at the floor is error against the *recorded* censored label, with uncertain sign relative to the latent half-life — a prediction above the latent value is charged too much, one below it too little — so it is neither an upper nor a lower bound. A censored (Tobit-style) loss is the principled alternative and is recorded as a limitation, not scope.

**Predeclared minimum worthwhile gain: Δ median per-allele Spearman = 0.05.** Derived from the frozen test set, not preference: within-allele label permutation gives 0.000 ± 0.018, and a paired cluster bootstrap of a true-zero difference has a 95% CI half-width of 0.037–0.053. Below ~0.05 this test set cannot separate a difference from zero.

**Why:** every model must solve the same generalisation problem without data leakage. The larger test partition gives more statistical power for within-allele ranking.

Hamming ≤ 3 is the most conservative feasible threshold. Single-linkage clustering percolates sharply between 3 and 4: at ≤ 3 the largest component has 22 peptides (0.6% of pairs); at ≤ 4 one component swallows 2,807 peptides (54.8% of pairs), making balanced splitting impossible. The split hits 70/10/20 exactly, with **67 of 75 alleles clearing 50 test rows**.

The resulting minimum distance from any validation or test peptide to any training peptide is 4 substitutions (verified). That separation is real but not absolute — same-allele label similarity decays smoothly with distance, not in a step:

| Distance | Same-allele label comparisons | Mean \|Δy\| | Spearman |
|---:|---:|---:|---:|
| 1 | 513 | 0.548 | 0.725 |
| 2 | 384 | 0.483 | 0.764 |
| 3 | 1,187 | 0.622 | 0.651 |
| 4 (min cross-split) | 11,275 | 0.737 | 0.592 |
| 5 | 91,636 | 0.815 | 0.512 |
| unrelated background | 81,455 | 0.968 | 0.302 |

At d=4 the Spearman rank correlation is 0.592, roughly halfway between unrelated pairs (0.302) and d=1 (0.725). No feasible threshold eliminates this residual similarity — it sits right at the percolation edge. Grouping reduces label leakage; it doesn't remove it. Stage 6 reporting accounts for this by stratifying test metrics by distance (see below).

**Trade-off:** the test set contains no peptide pairs within 3 substitutions of each other across splits. This means the benchmark measures generalisation to distant sequences but cannot assess mutant ranking — scoring point mutants of a known binder, which is often the practically relevant question. A nested cross-validation inside the training split partially recovers this (see stage 6).

Known residual: 8 alleles fall below 50 test rows, 7 of them because they hold ≤ 32 pairs in the whole dataset (`HLA-A*01:01` is the exception: 220 pairs, 43 test rows). Only `HLA-B*40:02` (19 pairs) is missing from a split entirely. That reflects allele rarity, not the threshold, and bounds which alleles support per-allele claims.

**The water-fill must rank splits by deficit *relative to target*, not absolute deficit.** Each allele was assayed on its own peptide panel, so cluster placement decides per-allele held-out coverage. Ranking by absolute deficit degenerates — once the three absolute deficits equalise they stay equal, so clusters go round-robin into thirds and every heavy cluster lands in whichever split led early. That still reaches 70/10/20 overall, but it left `HLA-B*42:01`, `HLA-B*51:01` and `HLA-B*81:01` (~350 pairs each) with **zero** test rows and only 60 of 75 alleles present in all three splits. The proportional criterion fixes this at no cost to the totals or the distance separation. See `reports/audit_summary.md` §5.

### 2. Establish supervised baselines

**Work**

- Train a small MLP on position-preserving one-hot or BLOSUM encodings of the peptide and HLA contact pseudosequence. (BLOSUM encodes amino acids using a substitution-probability matrix, so similar amino acids get similar vectors.)
- When comparing against domain-based embeddings later, add a matching baseline that uses the full HLA domain sequence as input — so we don't mistake "more input sequence" for a benefit of pretraining.
- Include training-set allele means as a simple sanity-check baseline for error and pooled metrics.
- Use a modest, comparable tuning budget across approaches. Save predictions, configuration, validation performance, and training/inference time.

**Deliverable:** reproducible sequence baselines and a first results table.

**Status: done.** `reports/stage2_baselines.md` (results and reasoning), `scripts/baseline_sequence.py` (the full grid, ~10 CPU-minutes), `pepstab/features.py` + `pepstab/mlp.py`, `reports/stage2_runs.csv` (every run), `tests/test_baselines.py` (25 guards). Headline baseline in `preds/seq_baseline.csv`.

Six arms — {one-hot, BLOSUM62} × {peptide, peptide+pseudosequence, peptide+domain} — each given the same budget: a 4-point MLP grid at 3 seeds plus a ridge alpha sweep. All models train on the same 17,744 rows (10% of train is held out as a stopping fold). Validation median per-allele Spearman, mean over seeds:

| Arm | MLP | Ridge |
|---|---:|---:|
| peptide + pseudosequence (one-hot) | **0.610** | 0.278 |
| peptide + pseudosequence (BLOSUM) | 0.603 | 0.270 |
| peptide + domain (one-hot) | 0.574 | 0.274 |
| peptide + domain (BLOSUM) | 0.521 | 0.259 |
| peptide only | 0.202–0.244 | 0.169–0.170 |
| training allele mean | 0.000 | — |

Three results clear the 0.05 bar (paired cluster bootstrap), two are inconclusive:

- **The model ranks within allele; the allele mean cannot.** +0.610 [+0.557, +0.656]. MAE 0.734 → 0.517, precision@10 at 2 h 0.359 → 0.70.
- **Nonlinearity is most of the model.** Ridge on the same features: 0.278; MLP − ridge = +0.331 [+0.256, +0.414]. A linear model cannot capture interactions between a peptide residue and the HLA pocket it sits in.
- **The HLA input contributes the other half.** Peptide alone 0.202; adding 34 contact residues: +0.404 [+0.300, +0.474].
- **34 contact residues vs 182 domain residues: unresolved.** +0.016 [−0.031, +0.084]. The full-domain arm is a legitimate matching baseline — any stage 3 win for domain embeddings must be checked against it, not only the pseudosequence arm.
- **One-hot vs BLOSUM62: unresolved.** +0.005 [−0.049, +0.061]. The other two arms separate in opposite directions, both within seed spread.

Two constraints on later stages:

- **Seed spread is 0.010–0.051.** Compare seed means, not single seeds; gaps under ~0.05 are noise.
- **Every model trains on the same rows as its comparator.** Single-network arms use `inner_folds()`: 10% of train (1,972 rows, 357 whole Hamming ≤ 3 clusters) held out as one permanent stopping fold, 17,744 fit rows, minimum fit/dev peptide distance 4. The ensemble comparator uses `cv_folds()`: 5 folds, each member fitting 15,772–15,773 rows, collectively covering all 19,716. **Stage 3 must match the ensemble protocol — `cv_folds()` and the same member count — not `inner_folds()`**, or the comparison is confounded by training-set size and ensembling together.

The stage 1 C67S finding is confirmed: the pseudosequence arm predicts `HLA-B*14:01(C67S)` and `HLA-B*14:02(C67S)` identically (max difference 0.00 across 41 shared validation peptides), while their measured labels correlate at only ρ = 0.708. The domain arm separates them (max difference 0.31). The impact on the primary metric is small — per-allele Spearman is computed within each allele, so identical predictions can still rank each allele's labels well (pseudo: 0.487/0.613 vs domain: 0.508/0.498). The collision caps cross-allele discrimination, not within-allele ranking.

Distance stratification is not answerable on validation: the two strata share only 6 alleles and 499 of 2,817 rows at the 20-row bar. Deferred to stage 6.

**Calibration against NetMHCstabpan** (`scripts/compare_to_paper.py`). The paper reports mean per-allotype SCC ≈ 0.69 / PCC 0.676 from 5-fold CV on this dataset; our 30-network ensemble reaches mean SCC 0.645 / PCC 0.649 (median per-allele ρ 0.693). **This is calibration, not reproduction, and no method-parity claim is supported.** Their training set is **5.2× ours and 73% augmentation we do not have** — 28,166 measured rows plus 1,000 assumed-zero weak binders per allele (75,000 rows); our CSV is the measured rows only. They also use BLOSUM50 not 62, smoothed sparse (0.9/0.05) not one-hot, and single hidden layers of 40/50/60 not 256×64. Matching their network count is not matching their method. Of the factors we can test: **split grouping explains nothing** — measured properly at equal row count on a common evaluation set inside train, Δ median ρ = **−0.005 [−0.035, +0.046]**, inconclusive and ruling out a 0.05 gain. The paper's `2^(-t0/th)` target is 0.022–0.034 *worse* here than `log1p`. Only ensembling clearly moves us (+0.090). The remaining 0.045 **cannot be discounted for the split**; the training-set difference is the leading explanation, which stage 2b tests directly. NetMHCstabpan can never be a comparator — it trained on every peptide in our test split.

**The baseline stage 3 must beat is the 30-network ensemble, not the single network** (`scripts/baseline_ensemble.py`, `preds/seq_ensemble_pep_pseudo.csv`). The paper uses one network per CV fold per architecture; our single networks are a weakened version. The strong form — 5 inner CV folds × 2 encodings × 3 seeds = 30 networks, folds cut along whole Hamming ≤ 3 clusters, configs from `stage2_summary.csv` — reaches **median per-allele ρ 0.693 / mean SCC 0.645** (single network: 0.610 / 0.573; paired Δ median ρ **+0.083 [+0.029, +0.124]**). The full-domain ensemble reaches 0.653; the two arms remain tied (+0.040 [−0.001, +0.073]).

A **superseded version of that calibration consumed frozen test rows** (3,350 into fitting, 448 into early stopping, 585 into scoring) and produced the since-retracted +0.018 grouping figure. Disclosed in EVALUATION.md, "Disclosed test exposure"; no model that will be scored at stage 6 saw a test row.

**Ensemble both arms identically at stage 3, or neither.** Ensembling alone is worth +0.090 mean SCC from no new information. An ensembled ESM arm against a single-network sequence arm would manufacture a result.

**NetMHCstabpan cannot be a comparator.** It was trained on all 28,166 rows, including every peptide in our test split, so any score on our data is memorisation. Its 0.69 is a CV score on its own training data, useful for calibration only. An honest comparison would need post-2016 stability measurements absent from its training set.

**Cost:** CPU only, no credits. Feature build under 0.3 s per arm (cached per unique sequence), fit 0.4–28 s, **inference under 1 ms per 1,000 predictions**. This is the floor ESM-2 extraction and GPU folding must justify themselves against.

**Why:** this shows what the task's labelled data can teach a small model on its own. It's not a reproduction of NetMHCstabpan's training and shouldn't be described as one.

### 2b. Compare weak-binder augmentation sources (optional)

**Work**

- **Pilot (~1-2 hours, only while the core comparison is on track):** compare two augmentation arms against the stage 2 measured-only baseline, using its selected peptide+pseudosequence MLP config, the same fit rows, stopping fold, and seeds. Run separately from stage 2c's auxiliary affinity head so each effect can be identified.
- **Measured-affinity arm:** use the committed [affinity reference](docs/AFFINITY_REFERENCE.md) (MHCflurry-curated and BD2013 snapshots). Select 9-mer pairs whose measurement or lower bound establishes affinity ≥ **20,000 nM**. Respect measurement inequalities; a generic negative assay result does not count.
- **Predicted-affinity arm:** sample 9-mers from natural protein sequences and keep pairs with predicted affinity **weaker than 20,000 nM**, following Rasmussen's recipe. Record the protein source, sampling seed, predictor version, and training-data provenance. Use affinity predictions, not stability predictions; reject candidates contradicted by measured affinity. If predictor setup threatens the time box, run only the measured arm.
- **Assumed stability labels:** assign `thalf_hours = 0` (`y_log1p = 0`) in a separate training table with label provenance. Never overwrite measured stability. Do not substitute wild-type affinity for a C67S construct; skip constructs without a matching measurement or supported predictor input.
- **Matched comparison:** use equal added counts per allele (capped at **25% of that allele's real fit rows**) on alleles supported by both sources. Keep unsupported alleles in the evaluation panel. Try assumed-label sample weights **0.1 and 0.25** (measured weight 1). Save the sampled rows and weights for reuse across model arms.
- **Leakage protection:** load `data/splits.csv` unchanged, and whichever stopping-fold scheme the arm under test uses (`inner_folds()` for single networks, `cv_folds()` for the ensemble — under `cv_folds()` every training peptide is some member's stopping peptide, so augmented candidates must clear the threshold against *all* training peptides, not just one dev fold). Exclude candidates within **Hamming distance ≤ 3 of any inner-dev, validation, or test peptide, across all alleles**. The reference's `padding_eligible` flag is insufficient (it checks exact pairs, so held-out peptides can reappear under other alleles). Save distance checks and exclusion counts.
- **Selection on measured validation only:** compare median per-allele Spearman over seeds, MAE on `log1p`, and precision@10 at 2 hours. Assumed zeros never enter validation or test. Test the selected arms at stage 6 under the existing **Δ Spearman = 0.05** contract. If extending to ESM-2, give both model families the same augmentation rows, targets, and weights.

**Deliverable:** measured-only vs. measured-affinity-padding vs. predicted-affinity-padding validation table, candidate manifests, leakage checks, seed variation, and costs. Record a skipped or inconclusive source comparison explicitly.

**Why:** the assay peptides were pre-selected for predicted binding, so augmentation tests whether broader negative coverage helps. Rasmussen et al. added 1,000 predicted weak binders per allele but did not compare them with measured affinity negatives. Earlier [NetMHCpan work](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0000796) motivates random negatives to reduce selection bias. Measured affinity gives stronger evidence per label; prediction gives control over diversity and allele coverage. Neither actually measures half-life (8 of 181 measured weak binders in the training overlap exceed 2 hours). This pilot tests the zero-label assumption rather than adopting the paper's full augmentation volume. Stage 2c separately tests affinity as an auxiliary target without assigning zero stability.

### 2c. Test auxiliary affinity training (optional)

**Work**

- **Probe after stage 2 (~1 hour):** ~7,600 peptides have both affinity and stability measurements across 58 allotypes (Rasmussen et al.). Add a second prediction head for affinity to the sequence baseline. Compare single-task vs. multi-task training on dual-labelled training pairs, using the frozen splits and stopping fold. This CPU probe can run in parallel with stage 3 embedding extraction.
- **Expand (only if probe helps, ~2-3 hours):** bring in the broader IEDB affinity dataset (~136K measurements, 152 alleles), starting with the sequence model. Apply stage 2b's exclusion rule: no auxiliary peptide within Hamming distance ≤ 3 of any inner-dev, validation, or test peptide, across alleles.
- **Extend to ESM-2 once stage 3 features are ready:** use the same affinity data and comparable tuning budgets for both arms. If multi-task training closes the gap between the sequence baseline and ESM-2, report it — cheap extra labels substituting for expensive pretrained features would be a noteworthy finding.

**Deliverable:** single-task vs. multi-task sequence validation results first, then an optional matched ESM comparison, with the leakage audit documented.

**Why:** the stability dataset's peptides were pre-selected for strong predicted affinity, limiting peptide diversity. IEDB affinity data covers far more peptides and alleles. Multi-task training lets the shared encoder see that diversity without changing the stability evaluation. Rasmussen et al. found that combining affinity and stability data improved epitope prediction beyond either alone (p<0.001), from complementary signal rather than row count.

### 3. Test frozen ESM-2 representations

**Work**

- Start with **ESM-2 35M** (the smallest checkpoint); try a larger one only if time, throughput, and validation results justify it.
- Cache embeddings once per unique peptide and HLA sequence, not once per measurement row.
- Embed peptide and HLA separately. Keep per-position information from the peptide; don't only use the mean across positions.
- Embed the supplied HLA domain. Verify residue indexing before selecting the 34 contact-position embeddings.
- Compare a middle layer against the final layer. Train small prediction heads on embeddings alone, and on embeddings plus raw sequence features.
- Select checkpoint, layer, representation, and head architecture using validation data. Check how much results vary across random seeds.

**Deliverable:** full-dataset baseline-versus-ESM comparison, cached features, and measured compute costs.

**Why:** this is the cheapest direct test of the foundation-model question and needs no structures. Embedding peptide and HLA separately means the prediction head has to learn peptide-HLA interactions on its own, so useful performance is a hypothesis, not a guarantee.

### 3b. Test auxiliary affinity training (optional, pre-structural)

**Work**

- **Probe (~1 hour):** Rasmussen et al. report ~7,600 peptides with both affinity (how strongly the peptide binds) and stability (how long it stays) measurements across 58 allotypes. Add a second prediction head to the sequence baseline for affinity. Train on these dual-labelled peptides only — they're already inside the frozen splits, so no new leakage risk. Compare single-task vs. multi-task validation performance.
- **Expand (only if probe helps, ~2-3 hours):** Bring in the broader IEDB affinity dataset (~136K measurements, 152 alleles). Before training, verify that no IEDB peptide is within one substitution of any stability test-set peptide; exclude any that are. Train multi-task models for both the sequence baseline and the ESM-2 arm.
- Report whether auxiliary affinity data helps each arm differently. If multi-task training closes the gap between the sequence baseline and ESM-2, that's worth reporting — it would mean cheap extra labels substitute for expensive pretrained features on this task.

**Deliverable:** multi-task vs. single-task comparison on the same frozen validation set, with the leakage audit documented.

**Why:** the stability dataset's peptides were pre-selected for strong predicted affinity, so peptide diversity is limited. IEDB affinity data covers far more peptides and alleles. Multi-task training lets the shared encoder see that broader diversity during training without changing the stability evaluation. Rasmussen et al. showed that combining affinity and stability data improved epitope prediction beyond either alone (p<0.001), with the gain coming from complementary signal, not just more rows.

**Data note:** the affinity reference ([`data/data_augmentation_iedb/affinity_reference_75alleles.csv`](data/data_augmentation_iedb/affinity_reference_75alleles.csv), 110,591 rows) carries both strong binders (64,112 rows with affinity <20,000 nM) and weak binders (46,479 rows at ≥20,000 nM). Both are needed: weak binders supply negative examples for classification-style augmentation; strong binders are the positives. The test of whether augmentation helps requires both — a dataset of only negatives cannot teach affinity prediction. See [docs/AFFINITY_REFERENCE.md](docs/AFFINITY_REFERENCE.md) for leakage filtering and the C67S construct exclusion.

### 3c. Elution as external validation (non-training)

**Work**

- Use the MHC Motif Atlas class I ligands (`data_classI_all_peptides.txt`, 487,343 rows) as an independent validation set, not training augmentation.
- Join to the stability dataset on `(allele, peptide)`. Only 140 of 151,170 9-mers on shared alleles overlap — this is a near-disjoint peptide universe, making it a genuine external test rather than circular validation.
- Score atlas ligands against length- and allele-matched proteome decoys. If the trained model ranks true ligands above decoys, that demonstrates transfer to a different assay measuring a different biological event.

**The finding.** Elution implies high stability probabilistically: among the 140 overlapping pairs, eluted ligands have **~7× higher median half-life** (7.90 h vs 1.10 h), and the fraction with zero measured stability drops **tenfold** (2.1% vs 21.2%). Mann–Whitney p = 6.0 × 10⁻³¹. The effect is not carried by HLA-A\*02:01 — among 18 alleles with ≥3 measured eluted ligands, the eluted median exceeds the non-eluted median in 14, including HLA-A\*01:01 (15.00 vs 1.30 h), HLA-B\*27:05 (5.20 vs 0.80 h), and HLA-A\*02:01 (17.50 vs 3.80 h). Common-language effect size: **0.78** — a random eluted ligand outlasts a random non-eluted peptide 78% of the time.

Six counterexamples have measured half-life <0.5 h, three at exactly zero. `FPEHIFPAL` appears twice on different alleles (`HLA-B*51:01` and `HLA-B*08:01`) with zero stability on both — a pattern more consistent with motif-deconvolution error (peptide assigned to alleles it doesn't bind) than with biology. This puts the atlas error rate at roughly 2% for stability predictions.

**Why not training augmentation.** Four reasons against using the atlas as training data:

1. **Scale mismatch.** 151,170 atlas 9-mers against 28,166 stability measurements. An auxiliary task 5.4× the size of the target dominates training; the result is an antigen-presentation model with a stability side-effect.
2. **The threshold is unknowable.** The honest encoding of an eluted ligand is "t½ > τ" for some protocol-dependent τ. Nothing in the data determines τ, and conclusions move with whatever value is chosen.
3. **Confounding.** Elution is confounded with source-protein abundance, proteasomal cleavage, TAP transport and MS ionisation efficiency. Stability is one of several filters, and the data cannot separate them.
4. **It changes the question.** Adding auxiliary data benefits a small BLOSUM network and a frozen protein language model to different degrees. Any difference measured afterwards is partly about which architecture absorbs mass-spec data, not about whether foundation models encode stability.

**Deliverable:** external validation pass on the trained model, measuring how well it ranks atlas ligands above decoys. The finding and reproducible analysis are in [`elution_stability_finding.md`](elution_stability_finding.md) and [`04_elution_stability_test.py`](04_elution_stability_test.py).

**Why:** a single scoring pass with no retraining demonstrates transfer to a different assay measuring a different biological event — a stronger claim than any within-dataset correlation.

### 4a. Structure arm A — ESMFold2

**The model is ESMFold2 (`biohub/ESMFold2`), not ESMFold v1.** Name the checkpoint explicitly everywhere; the two differ on exactly the properties that matter here.

| | ESMFold v1 (`facebook/esmfold_v1`) | **ESMFold2 (`biohub/ESMFold2`)** |
|---|---|---|
| Backbone | ESM-2 3B, frozen | **ESM-C** |
| Multi-chain | No — needs a poly-glycine linker hack | **Native, via a per-token `asym_id`** |
| Interface confidence | None (no ipTM) | **`iptm`, `pair_chains_iptm`, `complex_iplddt`** |
| Structure head | Direct regression | Diffusion, `num_diffusion_samples` |
| Parameters | ~3.7B | ~6.6-7B |

Arm A runs first because it has **no MSA step at all**, so it is producing structures while arm B's MSAs are still generating. Neither arm blocks the other.

**Why the version matters.** Published benchmarks put ESMFold **v1** at roughly 6-7 Å median peptide RMSD on peptide-HLA class I: the HLA fold itself comes out near 0.85 Å while the peptide is placed *outside* the groove, against ~0.7 Å for AlphaFold-based peptide-HLA methods. That is the worst possible error shape for this project — global confidence looks fine while the one thing we measure is wrong. ESMFold2's native multi-chain support is designed to fix precisely that failure mode, **but no published peptide-HLA benchmark of ESMFold2 exists.** Arm A is genuinely unvalidated on this task, which is why 4a.1 is a hard gate rather than a formality.

**4a.1 — Smoke pilot (3-5 complexes).** Use the same five crystal-matched complexes as stage 4b.2, so both arms are measured against identical ground truth from the first hour.

- **Peptide RMSD is go / no-go here, not a diagnostic.** Predeclare a pass threshold. If the peptide lands outside the groove on most of the five, arm A reports that as its result and does not proceed to a batch. Cutting arm A at the pilot is a legitimate and cheap outcome.
- Confirm the `asym_id` to output-index mapping, so feature code slices the right 9 residues.
- Confirm `pair_chains_iptm` indexes chain 0 as the HLA and chain 1 as the peptide.
- Push one complex all the way to a scored prediction through `scripts/evaluate.py` before trusting the pipeline.

**4a.2 — Throughput measurement and the scale decision.** Arm A folds one of two row sets, and the measurement decides which:

- **Option 1 — the full dataset (all 28,166 rows).** Take this if the measured cost per structure fits the arm A budget. It is much the stronger result: arm A then scores on the *same* rows, the *same* frozen splits and the *same* 67-allele panel as the sequence and ESM-2 arms, through the same evaluation call. No subsetting and no caveat about which rows were compared.
- **Option 2 — the same ~2,000-complex panel as arm B (`data/structural_panel.csv`).** Take this if the full dataset does not fit the budget. Arm A then folds exactly the complexes arm B folds, so ESMFold2 and Boltz-2 are compared structure for structure, with neither model having seen a larger or easier set of rows.

The cost driver to measure is `num_diffusion_samples` — it defaults to 32 and is roughly linear in runtime. Benchmark at 1 and at 8 before assuming arm A is the cheap arm; at the default it may well cost more per structure than Boltz-2. Decide on measured dollars per successful structure, and predeclare the threshold before measuring so the choice is arithmetic.

**4a.3 — Batch fold.** Shard by worker; keep a manifest of successes and failures keyed by `pair_id`. The diffusion head is stochastic, so fix and record the seed. If `num_diffusion_samples > 1`, sample spread is a free uncertainty feature that arm B will not have.

**4a.4 — Extract features** into `features/esmfold2_<scope>.parquet`, keyed by `pair_id`, **sharing one schema with arm B**: per-position peptide pLDDT, the peptide-HLA PAE block, peptide-HLA ipTM, and contacts and burial per peptide position.

**Infrastructure.** Use the HuggingFace `transformers` route. ESMFold2 runs in bf16 (~13 GB of weights), wants 24 GB or more of VRAM, and has a fused-kernel path. Modal publishes an ESMFold2 example to start from. Pin the revision.

**Deliverable:** pilot RMSDs against crystal structures, a throughput and cost measurement, the scale decision with its arithmetic, the structures, and a success/failure manifest.

**Why:** it is the cheaper structural integration and it removes the MSA dependency from the critical path, so a structural result survives even if arm B is cut entirely.

### 4b. Structure arm B — Boltz-2

Input: the supplied 182-residue HLA domain plus a separate 9-residue peptide chain. Keep Chai-1 as a fallback rather than running both.

**4b.1 — Precompute the MSAs (CPU, ~$0, before any GPU is booked).**

One MSA per unique HLA sequence, reused across every complex on that allele — six panel alleles means six MSA generations for ~2,000 folds. Alleles and HLA domain sequences are 1:1, so there is never a reason to compute an MSA per complex.

Boltz has no MSA-only subcommand; `predict` is the only command. The harvest route:

1. Write one pilot YAML per allele: the HLA chain with **no `msa:` key**, the peptide with **`msa: empty`**. Only one entity then needs generating, so Boltz takes the unpaired branch and produces a clean single-chain MSA.
2. Run `boltz predict pilots/<allele>.yaml --out_dir ./msa_gen --use_msa_server --accelerator cpu`. MSA generation happens before the model loads.
3. Harvest `msa_gen/boltz_results_<stem>/msa/<stem>_0.csv` — two columns, `key,sequence`. Copy to `structures/msa/<allele>.csv`, sanitising the `(C67S)` suffix out of the filename.
4. Record a manifest: allele, `sha256(hla_seq)`, sequence count, server URL, timestamp.

Every batch YAML then looks like this:

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: <182-residue hla_seq>
      msa: /abs/path/structures/msa/<allele>.csv
  - protein:
      id: B
      sequence: <9-mer peptide>
      msa: empty
```

Four constraints that otherwise cost hours:

- **`msa: empty` on the peptide is mandatory, not optional.** Boltz refuses to mix custom and auto-generated MSAs in one input; omitting the key means "auto" and fails the run.
- **Only `.a3m` and `.csv` are accepted.** `.a3m.gz` fails the suffix check.
- **Omit `--use_msa_server` on the batch run.** Boltz then errors if any chain lacks an MSA — exactly the guard that stops 2,000 jobs silently calling the ColabFold server.
- **Trim the MSA.** Boltz re-parses the MSA file once per complex, so a deep MSA is parsed ~2,000 times on CPU. Measure that cost in the pilot and cap `--max_msa_seqs` (default 8192) accordingly.

**4b.2 — Smoke pilot on 5 complexes with crystal ground truth (1 GPU, <$5).**

`data/pdb_rasmussen_overlap.csv` holds 32 pairs with both a measured half-life and a deposited structure, 19 of them TCR-free, and 17 of those 19 are in the training split — so a pilot drawn from them touches no held-out data. Proposed set — all training rows, all TCR-free, four alleles that are also on the structural panel so their MSAs get reused:

| Allele | Peptide | Half-life | PDB | Resolution (Å) | Released |
|---|---|---:|---|---:|---|
| `HLA-A*02:01` | VVPYEPPEV | 0.5 h | 21EX | 2.02 | **2026-09-09** |
| `HLA-B*07:02` | IPRRNVATL | 4.0 h | 7LFZ | 1.90 | 2021 |
| `HLA-B*35:01` | LPFERATVM | 5.5 h | 3LKR | 2.00 | 2010 |
| `HLA-B*15:01` | ILGPPGSVY | 11.0 h | 1XR9 | 1.79 | 2005 |
| `HLA-A*02:01` | LLWNGPMAV | 38.2 h | 5N6B | 1.60 | 2017 |

**The first row is a free recall-versus-prediction control.** `21EX` was deposited 2025-12-10 and released **2026-09-09**, after the training cutoff of every model under consideration, and it is the only pair among the 32 clearly post-cutoff for Boltz-2. Compare its peptide RMSD against the pre-2020 rows, which the model could have memorised: comparable RMSD means the model is predicting, markedly worse RMSD on `21EX` means it was recalling. Report it with the obvious limit — n=1, so it is directional, not a measurement.

Six of the 32 are 2021 or later (`21EX`, `8T7R`, `7PBC`, `7LG2`, `7LG3`, `7LFZ`). Whether those count as post-cutoff for ESMFold2 depends on its training snapshot; check the model card before claiming them as controls.

Exit criteria — **all** must pass before any batch launches:

1. Peptide backbone RMSD to the crystal peptide, after superposing on the HLA domain, below a predeclared threshold.
2. Canonical register: P2 and PΩ seated in the B and F pockets.
3. **Token-to-chain mapping verified.** The PAE matrix should be 191×191 with indices 0-181 the HLA and 182-190 the peptide. Boltz does not document this boundary — confirm it from the CIF residue order before any code slices the peptide-HLA PAE block.
4. **`pair_chains_iptm["0"]["1"]` confirmed to be HLA-to-peptide.** Chain index follows YAML entity order, also undocumented.
5. `--write_full_pae` produced `pae_*.npz`, and `plddt_*.npz` is present.
6. One feature row per complex, joined on `(allele, peptide)` to a `pair_id`, scored end to end through `scripts/evaluate.py` without error.

These five complexes are the ones *most* likely to have been memorised. The pilot is a **pipeline-correctness gate, not an accuracy estimate** — do not quote its RMSDs as evidence that Boltz predicts peptide-HLA structures well.

**4b.3 — Hardware and throughput benchmark (20-30 complexes).**

- Compare L40S 48 GB, A100 40 GB and H100 80 GB on identical settings. The metric is **measured billed dollars per successful complex**, not dollars per hour.
- Pre-stage weights in a persistent volume and set `BOLTZ_CACHE` to an absolute path. Boltz downloads all three files unconditionally — `boltz2_conf.ckpt` 2.29 GB, `boltz2_aff.ckpt` 2.06 GB, `mols.tar` 1.86 GB, ~6.2 GB total — even though the affinity head is never used. Pin the revision.
- Pin settings: `--diffusion_samples 1 --recycling_steps 3 --sampling_steps 200 --write_full_pae --output_format mmcif`. Pin `--subsample_msa` explicitly; its CLI default and its help text disagree.
- Batch by **directory**: `boltz predict <dir>` loads weights once for every YAML inside, which satisfies the "keep models loaded" requirement natively.
- Capacity = the arm B budget divided by measured cost per successful complex.

**4b.4 — Freeze the structural panel (CPU; labels never consulted).**

Selected from `data/splits.csv` and allele row counts alone — spread across the top 6 alleles proportionally, ~2,000 complexes, preserving 70/10/20 within each allele.

| Allele | In panel | Allele total | Test rows |
|---|---:|---:|---:|
| `HLA-B*15:01` | 434 | 1,070 | ~87 |
| `HLA-A*02:01` | 415 | 1,023 | ~83 |
| `HLA-A*03:01` | 349 | 861 | ~70 |
| `HLA-B*39:01` | 276 | 680 | ~55 |
| `HLA-B*35:01` | 264 | 650 | ~53 |
| `HLA-B*07:02` | 262 | 647 | ~52 |

Every allele clears the 50-row test bar in [EVALUATION.md](EVALUATION.md), so the primary metric has six alleles to take a median over. The deep alternative — three alleles folded in full, 2,954 complexes — gives tighter per-allele rho, but a median over three alleles is uninformative, and it runs ~950 folds over budget.

Freeze to `data/structural_panel.csv` (`pair_id, allele, peptide, split`), committed like `data/splits.csv`, selection deterministic and seeded.

Tiers, predeclared against measured throughput so the hour-5 call is arithmetic:

| Tier | Complexes | Alleles | When |
|---|---:|---|---|
| 0 | 5 | pilot only | pipeline works but throughput can't finish by hour 10 |
| 1 | ~700 | 3 | reduced |
| 2 | ~2,000 | 6 | target |
| 3 | +subset | 6 | stretch: a second diffusion sample on a subset, for seed spread |

**4b.5 — Batch fold.** Shard the panel YAMLs into one subdirectory per worker. Restart is free — Boltz skips any target with an existing `predictions/<id>/` directory unless `--override`. Keep a manifest of successes and failures keyed by `pair_id`.

**4b.6 — Extract features** from the CIF, `plddt_*.npz`, `pae_*.npz` and `confidence_*.json` into `features/boltz_panel.parquet`, keyed by `pair_id`, on the schema shared with arm A.

Moving to full-length HLA with beta-2-microglobulin (the additional chain that stabilises the HLA structure in vivo) would require a new pilot and runtime benchmark.

**Deliverable:** MSA manifest, hardware and runtime benchmark, the frozen structural panel, the structures themselves, and a manifest of successes and failures.

**Why:** folding is the biggest compute and integration risk. Cheaper-per-hour hardware may be slower per structure, so the metric that matters is measured cost per completed prediction. A smaller, interpretable experiment with adequate test coverage is worth more than many structures that can't support a comparison.

### 5. Test additional feature groups

**Work**

- Start with geometry and confidence features, which come from the existing structure predictions. Add each group separately, then test combinations that validation supports.
- **Test every feature group per arm**, so the ESMFold2-versus-Boltz-2 comparison lands in the ablation table rather than in prose. Report the two arms on the rows **both** folded; if arm A took the full dataset, report its unrestricted numbers separately, clearly labelled, never against a panel-restricted arm B.
- Train sequence, ESM, and augmented models on **exactly the same structural training rows**, with identical validation and test rows. Separately report full-training-data sequence models on that same test set as practical comparators.
- Keep low-confidence but usable predictions. Predefine how to handle outright failures, report coverage, and show a sequence-model fallback for structures that fail.
- Try multiple poses only on a small diagnostic subset if budget remains.

**Deliverable:** ablation table showing the incremental predictive value and cost of each feature group, per arm.

**Why:** a feature can look useful alone but add nothing on top of the sequence baseline. Important conceptual guardrails: pLDDT is the model's confidence in its prediction, not a measure of physical stability; seed disagreement captures model uncertainty, not real molecular motion; inverse-folding likelihood tells you if the sequence fits the backbone, not how long the complex lasts; interaction energy estimates the strength of binding, not the energy barrier to unbinding.

One guardrail specific to arm A: ESMFold2 is built on **ESM-C** and stage 3 uses **ESM-2**, so the two are related by model family but not by weights. Stack arm A's features on the ESM-2 arm as well as on the sequence baseline — it is cheap and it closes the question instead of leaving it open in the write-up. (Had arm A been ESMFold v1 this would have been a hard confound rather than a check: v1's trunk consumes frozen ESM-2 3B hidden states directly.)

### 6. Evaluate and prepare the submission

**Work**

- Evaluate the validation-selected models **once** on the held-out test set.
- Report per-allele Spearman correlation (how well the model ranks peptides within each allele), with test-set sizes and a median/IQR summary across alleles. Use a common set of eligible alleles across models and report small or undefined cases explicitly.
- Report MAE on `log1p` half-life for numerical error. Add **precision@10 at a predeclared 2-hour threshold** — of the top 10 predictions per allele, how many actually have a half-life above 2 hours? This directly measures whether the model identifies sufficiently stable peptides. Treat pooled metrics as secondary. Distinguish within-allele ranking from cross-allele effects.
- **Stratify test metrics by nearest-neighbour distance to training.** For each test peptide, compute the Hamming distance to its closest training peptide and report metrics in **two strata: d=4 (57.8% of test rows) and d≥5 (42.2%)**. A separate d≥6 stratum is not viable — measured on the frozen split only 6 test peptides (12 rows) sit that far from training. Score both strata on the *same* allele set (the intersection of those eligible in each, 65 alleles at a 20-row bar), or the comparison measures allele panels rather than distance. If label similarity decays as expected, performance should visibly differ across strata. If it doesn't, that's a strong signal the model genuinely generalises rather than exploiting residual similarity at the split boundary.
- **Nested near-neighbour evaluation inside training.** Cross-validate mutant ranking on the d≤2 peptide clusters that live entirely within the training split. This recovers the question the grouped split cannot answer — can the model rank point mutants of a known binder? — without touching the test set or the frozen split assignments.
- **The differential target.** For peptides measured on two or more alleles, evaluate Δ log half-life *between* alleles. This subtracts out whatever is intrinsic to the peptide and tests groove chemistry directly, which is the sharpest available version of "distinguish within-allele ranking from cross-allele effects". It is abundant — **3,941 peptides sit on ≥2 alleles, covering 26,474 rows (94% of the dataset), up to 36 alleles for a single peptide** — and it costs no new compute, being a re-aggregation of predictions already made.
- Use paired uncertainty estimates that keep peptide clusters together across alleles.
- Report accuracy gains alongside extraction/training cost, **cost per 1,000 new predictions**, runtime, and prediction failures.
- If a confidence interval crosses zero, the result is inconclusive — not negative. A strong negative result should rule out the predeclared minimum worthwhile gain. Bound every conclusion to the specific representation, data, split, and budget tested.
- If time remains, add an allele-held-out evaluation (train without some alleles, test on them), stratified by how similar the held-out alleles are to training alleles. Only then make any new-allele generalisation claim — and report the confound described in the out-of-scope register alongside it.
- The original assay panel was partly selected by predicted affinity, so broader biological or clinical claims need additional evidence.

**Deliverable:** reproducible code and configs, final comparison table, limitations section, and a concise presentation.

**Why:** the submission should show what helped, where it helped, and what it cost. A gain on unseen peptides for familiar alleles is useful even without a new-allele result.

## Budget and GPU decision rule

Team credits: **$450 Modal + approximately $60 Hugging Face**. Keep provider budgets separate. The figures below are **ceilings, not spending targets**.

| Allocation | Ceiling | Note |
|---|---:|---|
| HF: embedding extraction and regression experiments | $60 | |
| Modal: end-to-end and hardware pilot | $15 | covers both arms' pilots |
| Modal: structure arm A (ESMFold2) | $90 | scales with the 4a.2 decision; `num_diffusion_samples` is the cost driver |
| Modal: structure arm B (Boltz-2) | $240 | |
| Modal: contingency | $105 | includes the $45 formerly held for inverse-folding and energy scoring, both now out of scope |

If arm A's measured cost comes in well under its ceiling, the remainder moves to arm B's panel tier, not to a third model.

Claude, Devin, Antigravity, and AMASS credits support implementation and research; don't assume they fund folding GPUs. Recheck rates and credit availability before launch.

Reference worker rates assume standard Functions at base rates, four physical CPU cores, and 32 GiB host memory per GPU worker. These exclude startup, retries, extra poses, and storage/egress; region and non-preemptible options change pricing.

| GPU | GPU-only $/hour | Including CPU/memory $/hour |
|---|---:|---:|
| L40S, 48 GB | 1.95 | 2.40 |
| A100, 40 GB | 2.10 | 2.54 |
| A100, 80 GB | 2.50 | 2.94 |
| H100, 80 GB | 3.95 | 4.39 |

```text
cost = total billed worker seconds * combined GPU/CPU/memory rate
capacity = folding budget / measured cost per successful complex
elapsed time ~ total worker hours / concurrent workers
```

**Any seconds-per-complex figure is a budget threshold, not measured throughput.** Choose hardware based on measured dollars per successful prediction and the deadline. Running more workers in parallel shortens wall-clock time but doesn't reduce total compute cost. Pick the cheapest GPU that meets the deadline and memory requirements.

## Team workstreams and checkpoints

| Participant | Primary responsibility | Handoff |
|---|---|---|
| 1 | Data audit, splits, sequence baselines, evaluation | Shared split IDs and prediction/evaluation format |
| 2 | ESM-2 embeddings, regression heads, representation comparisons | Cached features and validation-selected models |
| 3 | GPU pilots, both structure arms, additional feature extraction | Structure and feature manifests keyed by complex ID, with success and failure records |

Participant 3 now owns two arms. They are not serial: **arm A has no MSA step, so it starts immediately, while arm B's 4b.1 MSA generation is CPU work that runs concurrently with arm A's pilot.** Neither blocks the other.

- **First 3 hours:** freeze data and evaluation, establish a baseline, begin ESM extraction, complete both arms' pilots on the same five crystal-matched complexes.
- **By hour 5:** review validation results. Make **two** structural decisions — arm A's scale decision (full dataset versus panel, from 4a.2) and arm B's go / reduce / stop against the 4b.4 tier table. Arm A proceeding while arm B is cut to tier 0 is a legitimate and reasonably likely outcome, and it still leaves a structural result. Run the auxiliary affinity probe if the core comparison is on track.
- **Hours 5-12:** finish the selected workload and ablations. Stop adding features at hour 10; freeze configurations by hour 12.
- **Final 4 hours:** evaluate on held-out data, compute uncertainty, prepare figures and the presentation, save deliverables.
- Adjust these cutoffs if the official deadline requires it.
- Save code, data, splits, features, checkpoints, structures, and results locally before the **Antigravity event resources are deleted after the event on Sunday, October 4**. Modal and Hugging Face credits are separate per-participant offers and aren't affected by that deletion. Keep credentials out of shared manifests and exported artifacts.

## Stretch work and stopping rules

- The auxiliary affinity experiment (stage 3b) may interact with the ESM-2 comparison: if multi-task training helps the sequence arm more than the ESM arm, report that result rather than burying it.
- Same-peptide / different-HLA diagnostics are possible with this data. **Matched C67S / wild-type comparisons are not** — the corresponding wild-type alleles aren't in the dataset.
- Only expand a feature pipeline when validation evidence or useful uncertainty information supports the extra cost. Don't claim an unbinding mechanism from improved prediction alone.

**Minimum successful submission:** reliable splits, a trained sequence baseline, an ESM-2 comparison, uncertainty estimates, and measured costs. Structural experiments strengthen this result but must not prevent completing the core study.

## Out of scope this round

Nothing correct is deleted from this plan — it is listed here with its reason. An entry means "decided against for this round", not "wrong".

| Item | Why out of scope this round |
|---|---|
| **Allele-axis hold-out** | Long-form rationale below |
| Template threading | We co-fold instead. Threading assumes the canonical register and cannot represent the non-canonical bulges that may be exactly the unstable complexes. `data/allele_pdb_templates.csv` (33 tier-A exact-groove templates) makes it cheap if this is ever revisited |
| FoldX, Rosetta, any empirical energy layer | Estimates equilibrium ΔG, not the ΔG‡ barrier to unbinding — a known mismatch with a kinetic label. Also a setup cost we would not recover in 16 hours |
| Per-pocket energy decomposition (A/B/F ↔ P1/P2/PΩ) | A good idea, but premised on FoldX `AnalyseComplex`; it falls with FoldX |
| OpenMM minimisation energy | Not a drop-in replacement for interface scoring, and the same thermodynamic/kinetic mismatch applies |
| ProteinMPNN | Correct and cheap per structure, but it needs the structures first, so it is the last thing to add rather than a core arm. Revisit after both folding arms report |
| ESMFold v1 (`facebook/esmfold_v1`) | Not the model in arm A and not a substitute for it: no native multi-chain support (it needs a poly-glycine linker), no ipTM, and published peptide-HLA benchmarks put its peptide ~6-7 Å outside the groove. Its trunk is frozen ESM-2 3B, which would also confound it with stage 3 |
| Chai-1 | Fallback for Boltz-2 only. Running both doubles integration cost to answer a question nobody asked |
| β2-microglobulin / full-length HLA | Would require a fresh pilot and runtime benchmark; the α1/α2 groove is what contacts the peptide |
| Chimeric peptide-linker-groove ESM input | The sharpest version of the stage 3 question and a real risk to a negative result, but a linkered 9-mer construct sits far outside ESM-2's distribution. Listed so that "ESM-2 didn't help" is reported as bounded by the separate-embedding representation actually tested |
| Tobit / censored likelihood | Principled for the 20.2% floor; `log1p` plus rank-led metrics is the 16-hour substitute. Already recorded as a stage 1 limitation |
| Source-protein / UniProt mapping, gene-level splits | The splits are frozen and cannot be rebuilt |
| SaProt, cross-attention, folding-trunk features, geometry-aware GNNs, extensive interpretability probes, molecular-dynamics unbinding | Defer until the core comparison is secure |

### Why the allele-axis hold-out is not the headline

Leave-allele-out evaluation, stratified by pseudosequence distance from the held-out allele to its nearest training allele, stays where stage 6 already puts it — "if time remains". Four reasons, in order of force:

1. **It is confounded by design in this dataset.** Each allele was assayed on its own peptide panel (`reports/audit_summary.md` §5 — this is what broke the first naive split). Holding out an allele simultaneously holds out its peptide panel, so an apparent allele-distance effect is inseparable from a peptide-panel effect. This reason is specific to this data and is the one that matters.
2. **It answers a different question from our headline claim.** The claim is unseen peptides on alleles we trained on. New-allele generalisation is a separate and weaker-powered question.
3. **The frozen split is peptide-grouped, not allele-grouped.** Running it properly means a second split, a second evaluation contract, and every number reported twice, under a 16-hour clock.
4. **Allele coverage is too skewed to stratify.** 8 of 75 alleles hold fewer than 50 test rows, 7 of them because they hold ≤32 pairs in the entire dataset. A near / intermediate / distant stratification would leave very few alleles per bin.

**Why it is still worth doing, and what it would buy.** This is the stratum where pretrained features have the strongest prior of winning — pan-allele generalisation is precisely what large-scale pretraining should provide — so a gain confined to distant alleles would be a real, reportable result even if the pooled comparison came out flat. If it is run, reason 1 must be reported alongside it: the confound does not disappear by being measured.

## Sources and shared context

- Corrected dataset CSV — canonical modelling input, committed to this repository under `data/`
- [Challenge brief](./Serova%20Protein%20Engineering%20Track%20Challenge.pdf)
- [Rasmussen et al. — assay, dataset selection, and NetMHCstabpan](https://pmc.ncbi.nlm.nih.gov/articles/PMC4976001/)
- [ESM-2 — checkpoints and representation extraction](https://github.com/facebookresearch/esm)
- [Boltz-2 — inputs, MSA formats, confidence outputs, and PAE saving](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md)
- [ProteinMPNN — conditional sequence scoring](https://github.com/dauparas/ProteinMPNN)
- FoldX: [RepairPDB](https://foldxsuite.crg.eu/command/RepairPDB) and [AnalyseComplex](https://foldxsuite.crg.eu/command/AnalyseComplex)
- [Modal pricing](https://modal.com/pricing) and [Boltz deployment example](https://modal.com/docs/examples/boltz_predict)
- [Hugging Face Jobs pricing](https://huggingface.co/docs/hub/en/jobs-pricing)
- [pLDDT interpretation](https://www.ebi.ac.uk/training/online/courses/alphafold/inputs-and-outputs/evaluating-alphafolds-predicted-structures-using-confidence-scores/plddt-understanding-local-confidence/)
