# Peptide-HLA stability: hackathon execution plan

**Team:** 3 participants
**Event:** London AI × Science, protein engineering track, October 3-4, 2026
**Research question:** can pretrained protein language model features predict how long a peptide stays bound to an HLA molecule better than raw sequence features — and is the improvement worth the compute cost?

Our main claim is about **unseen peptides on HLA alleles we trained on**. Whether the model works on entirely new alleles is a separate test, not the headline.

## Live execution status — 4 October 2026, final push

Five workstreams run concurrently. Each owns a disjoint set of files; **only the
orchestrator commits**, to avoid index contention in the shared worktree. No
workstream edits this file, `CLAUDE.md`, or `EVALUATION.md`.

| Workstream | Stage | Owns | State |
|---|---|---|---|
| `esm-arm` | 3 | `pepstab/esm.py`, `scripts/esm_*.py`, `features/esm/`, `reports/stage3_*`, `preds/esm_*` | running |
| `eval-harness` | 6 machinery | `pepstab/stage6.py`, `scripts/stage6_report.py`, `reports/stage6_*` | running |
| `submission` | 6 deliverable | `reports/SUBMISSION.md`, `reports/compute_ledger.*`, `reports/limitations.md`, `reports/figures/` | running |
| `elution` | 3c scoring pass | `scripts/stage3c_*`, `pepstab/elution.py`, `reports/stage3c_*`, `external/` | running |
| Boltz-2 production fold | 4c | `modal_app/`, `structures/`, `data/structural_cohort.csv`, Modal volumes | running (separate session) |

Added later the same morning: `inverse-folding` (ProteinMPNN, stage 5's third
model class), `stats-stretch` (stage 7 censored likelihood and allele hold-out),
and `struct-features` (stage 4c.5 extraction). FoldX and Rosetta were requested
as stretch goals and declined — see the out-of-scope register.

**Two remaining items are blocked, not unowned.** Both are assigned in advance
so neither falls through when its blocker clears:

- **Stage 3b's ESM-2 arm** (multi-task affinity) goes to `esm-arm`, behind the
  core stage 3 comparison and stage 3d. The sequence half is already done and
  negative at stage 2c.
- **Stage 5's ablations** — training heads on structural features — go to
  `struct-features` as a third phase, since it will know the feature semantics
  best. Its phases are: (1) build and validate the extractor against the 90
  local pilot folds, free; (2) extract from the production Volumes on Modal,
  CPU-only, on explicit go-ahead; (3) the stage 5 ablations.

Dependency order for what remains: stage 3 unblocks the ESM half of stage 3b;
stage 4c production unblocks stage 4c.5, which unblocks stage 5; stages 3, 5
and the stage 6 machinery all feed the single test scoring at stage 6, which
happens **once**, last, under the single-pass rule predeclared in that stage.

**Resource note, 4 October 04:30 BST.** Seven concurrent workstreams on an
8-core / 16 GB laptop drove load average to 59 and swap to 289 MB free. Agents
are throttled to one worker each and ESM-2 650M is dropped. The binding
constraint is not CPU but the two client processes driving the production fold:
they must survive ~6.5 more hours, and an OOM that kills them costs the
structural arm and the GPU spend. Scope is cut before footprint is grown.

## Recommended scope

Commit to comparing a supervised sequence baseline against frozen ESM-2 features across the full dataset. Only add a structural experiment (predicted 3D shapes, confidence scores) if a small end-to-end pilot works first.

The structural path folds a **275-residue HLA ectodomain + 99-residue beta2m + 9-residue peptide** three-chain construct. The matched 90-fold pilot ran **both Boltz-2 and ESMFold2** on the same sequences, prepared MSAs, pilot arms, seeds, and evaluation; it is complete. **Boltz-2 passed its gate and ESMFold2 failed it**, so production is Boltz-2 only, over all 28,166 pairs, split across two Modal workspaces — a labelled revision of the agreed model comparison, with the cross-model result delivered by the pilot rather than by production. A clear negative result ("pretrained features didn't help") remains a successful submission, and structural work must not block the core comparison.

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
| ESMFold2 geometry + confidence | Stage 4c pilot; same production cohort as Boltz-2 | Does changing the folding model change predictive value at fixed inputs? | Same full construct, prepared MSAs, shared geometry/confidence definitions; verified pair-specific ipTM is a separate feature. | Matched-row gain, pose quality, resource cost, and failure rate. |
| Boltz-2 geometry | Stage 4c pilot; full dataset preferred if feasible | Do full-construct structural features add predictive value? | Three-chain ectodomain + beta2m + peptide predictions; comparable groove contacts and burial per peptide position. | Added accuracy on matched rows, peptide pose quality, measured cost, runtime, and failure rate. |
| Boltz-2 confidence | Same full-construct structures | Does model confidence carry signal? | Peptide pLDDT; peptide/groove PAE in both directions; global ipTM labeled as global. Pair-specific scores require verified availability and chain mapping. | Gain from confidence alone and beyond geometry; global three-chain confidence is not peptide-interface confidence. |
| ProteinMPNN | Out of scope this round | Is the peptide sequence "compatible" with the predicted backbone shape? Uses another pretrained model, no refolding needed. | Peptide-only overall likelihood and per-position scores from ProteinMPNN (an inverse-folding model that asks: given this 3D backbone, how probable is this amino acid sequence?), with HLA held fixed. | **Promoted 4 October 2026.** The structures blocker cleared: 90 stage 4c pilot folds sit on local disk, so the pilot runs now and the harness scales to production unchanged. Gated on a seed-variation control — if the score moves more across seeds of one complex than between complexes, the feature is noise and does not proceed. |

Each experiment must earn its place by improving prediction on the same held-out examples, with uncertainty and compute cost reported alongside accuracy. A useful negative result tells us which approach didn't help under these conditions — it doesn't rule out every use of that model family.

## Dataset and known facts

**Canonical input:** the corrected CSV committed under `data/`. Keep it unmodified alongside its checksum so every workstream uses identical data.

| Property | Value |
|---|---:|
| Unique peptide-HLA pairs | 28,166 |
| Unique peptides | 5,633 |
| Unique HLA domain sequences | 75 |
| Peptide length | 9 residues |
| HLA domain length in the raw dataset | 182 residues |
| HLA ectodomain length for structural production | 275 residues |
| Mature beta2m length for structural production | 99 residues |
| Full structural complex length | 383 residues across three chains |
| HLA contact pseudosequence length | 34 residues |
| Duplicate peptide-allele pairs | 0 |
| Zero-hour labels | 5,679 (20.2%) |
| Median nearest-neighbour distance between peptides | 4 substitutions |
| Peptides with any neighbour within 3 substitutions | 871 (15.5%) |
| Largest single-linkage cluster at Hamming ≤ 3 | 22 peptides (157 pairs, 0.56%) |

Inputs are `peptide`, `hla_seq`, and `hla_pseudoseq`; the target `thalf_hours` is dissociation half-life. Sequence and ESM-2 comparisons continue to use the original dataset inputs. Structural production joins the supplied ectodomain table on the exact allele identifier; it preserves the 182-residue groove prefix and all C67S constructs. Input provenance and MSA preparation are in stage 4c.

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

**Calibration against NetMHCstabpan** (`scripts/compare_to_paper.py`). The paper reports mean per-allotype SCC ≈ 0.69 / PCC 0.676 from 5-fold CV on this dataset; our 30-network ensemble reaches mean SCC 0.645 / PCC 0.649 (median per-allele ρ 0.693). **This is calibration, not reproduction, and no method-parity claim is supported.** Their training set is **5.2× ours and 73% augmentation we do not have** — 28,166 measured rows plus 1,000 assumed-zero weak binders per allele (75,000 rows); our CSV is the measured rows only. They also use BLOSUM50 not 62, smoothed sparse (0.9/0.05) not one-hot, and single hidden layers of 40/50/60 not 256×64. Matching their network count is not matching their method. Of the factors we can test: **split grouping shows no conclusive advantage** — measured at equal row count on a common evaluation set inside train, Δ median ρ = **−0.005 [−0.035, +0.046]** and Δ mean ρ = **−0.000 [−0.024, +0.024]**. Both intervals straddle zero and both still admit a modest positive effect (the retracted +0.018 sits inside each), so this is *no conclusive advantage*, not *no effect*; what it does bound is grouping below ~0.024 on the mean — about half the 0.045 gap. The paper's `2^(-t0/th)` target is 0.022–0.034 *worse* here than `log1p`. Only ensembling clearly moves us (+0.074 against the deployed single network; the +0.090 sometimes quoted is against the mean of the ensemble's own 30 members, a different reference — see `reports/stage2_baselines.md`). So at most half the remaining 0.045 is attributable to the split; the training-set difference is the leading explanation. Stage 2b tested its mechanism and found nothing: assumed-zero padding from either source gives Δ median ρ of at most +0.024, every interval excluding 0.05. That was run at 0.16 augmented rows per measured row against the paper's ~2.7, so it weakens the training-set explanation without eliminating it — but the negatives it adds are ones the baseline already places below the assay floor, so volume is unlikely to change the answer. NetMHCstabpan can never be a comparator — it trained on every peptide in our test split.

**The baseline stage 3 must beat is the 30-network ensemble, not the single network** (`scripts/baseline_ensemble.py`, `preds/seq_ensemble_pep_pseudo.csv`). The paper uses one network per CV fold per architecture; our single networks are a weakened version. The strong form — 5 inner CV folds × 2 encodings × 3 seeds = 30 networks, folds cut along whole Hamming ≤ 3 clusters, configs from `stage2_summary.csv` — reaches **median per-allele ρ 0.693 / mean SCC 0.645** (single network: 0.610 / 0.573; paired Δ median ρ **+0.083 [+0.029, +0.124]**). The full-domain ensemble reaches 0.653; the two arms remain tied (+0.040 [−0.001, +0.073]).

A **superseded version of that calibration consumed frozen test rows** (3,350 into fitting, 448 into early stopping, 585 into scoring) and produced the since-retracted +0.018 grouping figure. Disclosed in EVALUATION.md, "Disclosed test exposure"; no model that will be scored at stage 6 saw a test row.

**Ensemble both arms identically at stage 3, or neither.** Ensembling alone is worth **+0.074 mean SCC / +0.083 median per-allele ρ [+0.029, +0.124]** over the matched single network, from no new information. An ensembled ESM arm against a single-network sequence arm would manufacture a result. (Quote +0.090 only with its reference stated: that figure is against the mean of the ensemble's own members, not against the network we would otherwise ship, and it additionally bundles in that members fit 15,773 rows to the single network's 17,744.)

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

**Status: done. Negative result, and the intervals rule out the bar.** `reports/stage2b_augmentation.md` (results and mechanism), `scripts/augment_affinity.py` (manifests and the in-house affinity predictor), `scripts/baseline_augmented.py` (the arms and intervals), `scripts/stage2b_negatives.py` (pool characterisation), `pepstab/augment.py` (candidate filtering and the leakage rules), `data/augmentation/` (three committed manifests plus `provenance.json` with per-manifest digests), `tests/test_augmentation.py` (47 guards). CPU only, $0.

Both arms ran at **matched per-allele counts** — 2,910 rows each, 51 of 75 alleles, capped at 25% of that allele's fit rows — at assumed weights 0.1 and 0.25, three seeds, on the stage 2 selected peptide+pseudosequence config with the same fit rows and the same measured-only stopping fold. The `measured_only` arm reproduces stage 2 exactly (ρ 0.6096 over 3 seeds), because uniform weights reduce to the unweighted loss bit-identically.

**All eight paired intervals cross zero and exclude 0.05.** Best arms: predicted at full coverage, w=0.25 (**+0.024 [−0.026, +0.048]**) and measured, w=0.25 (**+0.023 [−0.033, +0.039]**). Source comparison at matched counts: **−0.002** (w=0.1) and **−0.017** (w=0.25), both inconclusive and bounded below the bar.

**The mechanism matters more than the number.** The two sources supply different kinds of negative. Measured weak binders have near-canonical anchor residues (77.6% hydrophobic at PΩ, matching the 84.2% of real floor rows) — the baseline already scores them near the floor (+0.13 vs +0.22). Predicted weak binders from random human peptides have the *wrong* anchors (49.1% hydrophobic, 25.9% charged at PΩ) — the baseline scores them *below* the floor (−0.07). More easy negatives are still easy negatives.

Constraints:

- **Stage 3 runs unaugmented.** The plan requires augmentation to help the sequence arm first. It doesn't. Manifests stay committed if revisited.
- **Stage 6 scores no augmented model.**
- **An ensembled rerun would lose 44% of the measured arm** to the stricter `cv_folds` exclusion (1,640 of 2,910 rows survive vs 2,861 of 2,910 predicted).
- **Volume is untested.** 0.16 augmented rows per measured row vs Rasmussen's ~2.7. The anchor analysis suggests volume wouldn't help, but that's a prediction.

One hypothesis for stage 6: on the 22 alleles with no public weak-affinity data, the full-coverage predicted arm moves the sub-panel median from 0.577 to 0.601 with the best MAE (0.511) and P@10 (0.733). Underpowered — 22 alleles, one weight, no interval.

**Why:** Rasmussen et al. padded their training set with ~1,000 predicted weak binders per allele, labelled 0 h, but never compared against measured-affinity negatives. This pilot tests that mechanism at lower volume. Stage 2c separately tests affinity as an auxiliary *target* without assigning zero stability.

### 2c. Test auxiliary affinity training (optional)

**Work**

- **Probe after stage 2 (~1 hour):** ~7,600 peptides have both affinity and stability measurements across 58 allotypes (Rasmussen et al.). Add a second prediction head for affinity to the sequence baseline. Compare single-task vs. multi-task training on dual-labelled training pairs, using the frozen splits and stopping fold. This CPU probe can run in parallel with stage 3 embedding extraction.
- **Expand (only if probe helps, ~2-3 hours):** bring in the broader IEDB affinity dataset (~136K measurements, 152 alleles), starting with the sequence model. Apply stage 2b's exclusion rule: no auxiliary peptide within Hamming distance ≤ 3 of any inner-dev, validation, or test peptide, across alleles. The rule is implemented in `pepstab.augment` (`annotate_candidates`, `verify_manifest`) and needs no new code. Stage 2b's negative result does not bear on this stage: it ruled out *assumed-zero stability labels*, while 2c keeps affinity as a separate target, and the 7,281 dual-measured pairs are near-canonical binders rather than the anchor-violating pool that made 2b's predicted arm uninformative.
- **Extend to ESM-2 once stage 3 features are ready:** use the same affinity data and comparable tuning budgets for both arms. If multi-task training closes the gap between the sequence baseline and ESM-2, report it — cheap extra labels substituting for expensive pretrained features would be a noteworthy finding.

**Deliverable:** single-task vs. multi-task sequence validation results first, then an optional matched ESM comparison, with the leakage audit documented.

**Status: done — negative, and bounded.** `reports/stage2c_affinity.md`
(results and reasoning), `scripts/affinity_multitask.py` (regenerates),
`pepstab/multitask.py` + `pepstab/affinity.py`, `reports/stage2c_runs_*.csv`,
`reports/stage2c_deltas_*.csv`, `tests/test_multitask.py` (17 guards).

Across **20 paired comparisons** — 5 λ settings × {one-hot, BLOSUM} × {single
network, 30-network ensemble}, plus a censoring-robustness variant — every 95%
CI crosses zero and **every upper bound sits below 0.05**. Largest upper bound
+0.034. The predeclared worthwhile gain is ruled out, not merely undetected.

The comparison is controlled by construction: at **λ = 0 the multi-task network
is bit-identical to `pepstab.mlp.MLPRegressor`** (asserted on the real feature
grid, not only a toy), so the single-task arm *is* the stage 2 baseline. The
λ=0 single network reproduces 0.610 exactly, and the λ=0 ensemble is
**bit-identical** to `scripts/baseline_ensemble.py`'s predictions on all 2,817
validation rows. Both arms were ensembled identically, per the stage 3 parity
rule.

Two diagnostics make this a clean null rather than an ambiguous one:

- **The auxiliary task was genuinely learned** — the affinity head reaches
  ρ 0.55–0.62 against held-out affinity labels, against ≈ 0 at λ = 0. The shared
  trunk learns affinity about as well as it learns stability, and the stability
  predictions still do not move. Mean ensemble-member quality is flat at every
  λ, so affinity is not acting as a diversity source either.
- **The label's ceiling is below the baseline.** Measured affinity used
  *directly* as a stability predictor ranks at median per-allele ρ **0.580**
  [IQR 0.486, 0.696] over the 33 training alleles with ≥ 30 dual-labelled pairs
  — under the 0.610 a single network already reaches from stability labels
  alone. The auxiliary signal is **redundant, not absent**.

**The expansion is declined on evidence, not blocked.** The plan gates it on the
probe helping; it does not. The leakage audit was still completed because stage
2b needs it: of 66,214 affinity rows on 20,836 peptides absent from the
stability set, **64,226 rows on 20,195 peptides** clear Hamming > 3 from every
validation, test *and inner stopping-fold* peptide (2,076 in all). Absence is
tested on `peptide`, never on `(allele, peptide)` — the reference table's
`padding_eligible` flag tests the pair and therefore leaks.

**The ESM-2 extension is blocked, not declined**, and is where the hypothesis
keeps its strongest form: affinity is redundant with what a *sequence* model
already extracts, which does not establish redundancy with ESM-2 features. The
machinery is protocol-agnostic, so re-running it on the stage 3 arm is cheap.

**Cost: CPU only, $0.** 210 networks in total; the paired bootstrap dominates
wall time, not the fitting.

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

**Status: the sequence half is done at stage 2c** (negative and bounded — see
above). What remains here is only the ESM-2 arm, which needs stage 3 features.

**Why:** the stability dataset's peptides were pre-selected for strong predicted affinity, so peptide diversity is limited. IEDB affinity data covers far more peptides and alleles. Multi-task training lets the shared encoder see that broader diversity during training without changing the stability evaluation. Rasmussen et al. showed that combining affinity and stability data improved epitope prediction beyond either alone (p<0.001), with the gain coming from complementary signal, not just more rows.

**Data note:** the affinity reference ([`data/data_augmentation_iedb/affinity_reference_75alleles.csv`](data/data_augmentation_iedb/affinity_reference_75alleles.csv), 110,591 rows) carries both strong binders (64,112 rows with affinity <20,000 nM) and weak binders (46,479 rows at ≥20,000 nM). Both are needed: weak binders supply negative examples for classification-style augmentation; strong binders are the positives. The test of whether augmentation helps requires both — a dataset of only negatives cannot teach affinity prediction. See [docs/AFFINITY_REFERENCE.md](docs/AFFINITY_REFERENCE.md) for leakage filtering and the C67S construct exclusion.

### 3d. ESM-2 likelihood features (added 4 October 2026)

**Work**

- The challenge brief names three ways to use a protein foundation model:
  embeddings, **log-likelihoods/perplexities**, and confidence metrics, and
  invites running them "with different inputs masked". Stage 3 covers
  embeddings, stage 4c covers structural confidence, and seed variation is
  tested in both. Likelihoods were not covered anywhere, so this stage closes
  that gap.
- Compute, from the same ESM-2 checkpoint stage 3 selects: peptide
  pseudo-log-likelihood (sum and per-position masked marginals), peptide
  perplexity, and per-position entropy. Cache once per unique peptide, as in
  stage 3.
- Test them as a feature group on their own and appended to the stage 3 and
  stage 2 feature sets, selected on validation, with the same ensemble size and
  tuning budget as every other arm.

**Deliverable:** likelihood feature cache, a validation comparison on matched
rows with paired uncertainty, and the marginal compute cost over stage 3.

**Why:** it is nearly free once the stage 3 cache and weights exist, and it
tests a different claim from embeddings. An embedding asks "where does this
sequence sit in representation space"; a likelihood asks "how surprising is
this sequence to a model of natural protein sequence space". Neither is
obviously the right prior for a *kinetic* property, which is what makes the
comparison informative either way.

**Known limitation, stated up front.** ESM-2 scores the peptide without its
HLA. A 9-mer's likelihood under a model of whole natural proteins is a weak,
context-free signal, and the HLA-conditioned version of this feature would need
the chimeric peptide-linker-groove input that the out-of-scope register rejects
as far outside ESM-2's training distribution. So a null result here bounds to
context-free peptide likelihood, and must be reported that way rather than as
"likelihoods don't work".

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

### 4a. Completed two-chain ESMFold2 pilot and benchmark

**Status: historical diagnostic complete; matched three-chain production planned in 4c.** ESMFold2
(`biohub/ESMFold2`) was tested on the same two-chain groove/peptide workload as
Boltz-2. Its measured peak GPU memory was 26.0 GB, its speed overlapped
Boltz-2's between-container variation, and median peptide CA RMSD on five
complexes was 0.68 A versus Boltz-2's 0.29 A.

The measured results and limitations remain in
[reports/stage4_benchmark.md](reports/stage4_benchmark.md), with the original
implementation in `modal_app/esmfold_*.py`. The earlier recommendation to
reject ESMFold2 is superseded: both models will run the same experiment under
stage 4c. The measurements remain useful for hardware planning; they do not
validate the new three-chain construct or justify changing one model's inputs.

### 4b. Completed two-chain Boltz-2 pilot and benchmark

**Status: groove MSA cache, pose pilot, and hardware benchmark complete.**
These outputs describe **182-residue HLA groove + 9-residue peptide**, two
chains and 191 residues; an ectodomain production batch has not run.

| Completed artifact | What it establishes |
|---|---|
| [reports/stage4b1_msa_cache.md](reports/stage4b1_msa_cache.md), `reports/msa_manifest.csv` | 75 groove MSAs cached in 137 s on CPU at $0; depths 9,505-10,454. They supply pilot arm A, not the 275-residue HLA input. |
| [reports/stage4_benchmark.md](reports/stage4_benchmark.md), `reports/gpu_decision.csv` | Two-chain A10 measurement: $0.004/complex, projected $8 for 2,000 pairs or about $113 for 28,166. The two-chain panel projection was 0.56 h at ten workers. |
| `reports/boltz_pose_check.csv`, `reports/boltz_pilot.csv` | Five training-split peptides were in the groove; median CA RMSD 0.29 A. B*07:02/IPRRNVATL had 1.296 A CA RMSD and up to 3.329 A deviation at P6 despite correctly placed anchors. |
| `modal_app/boltz_*.py`, `scripts/boltz_pose_check.py`, [docs/BOLTZ_PIPELINE.md](docs/BOLTZ_PIPELINE.md) | Historical two-chain harness, pose scorer, and measurement protocol to adapt for stage 4c. |

The benchmark recorded 68 Boltz folds with no failures and approximately $2.64
spend. It established three implementation lessons that carry forward:

- Keep weights resident across a batch. Reloading per complex initially
  overstated cost by about 4.6x.
- Measure separate allocations: L40S container timings ranged from 6.3 to
  10.9 s, while variation within a container was small.
- Verify peptide geometry directly. High confidence did not flag the central
  bulge error, so the earlier coarse groove-placement gate is insufficient for
  accepting the new construct.

The old cost, memory, 191 x 191 PAE boundaries, engine arm names, and automatic
restart assumptions are historical. They do not establish stage 4c feasibility
or replace its output-completeness and pose checks.

### 4c. Ectodomain + beta-2-microglobulin folding

**Current execution plan, 4 October 2026. Status: shared MSAs, CPU preflight,
and the matched 90-fold GPU pilot are complete. Boltz-2 passed its gate;
ESMFold2 failed on both sentinel criteria. Production scope is frozen as
Boltz-2 / arm B / all 28,166 pairs across two workspaces, and has not been
launched.** Results and the full verdict are in
[`reports/stage4c_ectodomain_pilot.md`](reports/stage4c_ectodomain_pilot.md).
This section governs structural scope, pilot gates, and rollout; that report
holds the executed procedure, the launch sequence, and the stage 5 feature
contract. **The pilot
ran both Boltz-2 and ESMFold2 with the same constructs, MSA content, unpaired
policy, complexes, and seeds.** Model was a separate variable from pilot arm,
and neither model was an optional fallback: ESMFold2 is dropped from production
by its gate result, recorded in 4c.4, not by preference.

**4c.1 — Pin inputs and prepare MSAs on CPU.** Preserve the four colleague-supplied
files with their hashes. Track the ectodomain table as a modeling input and
retain the exact IPD-IMGT/HLA FASTA. Record the release if known; keep unknown
release metadata explicit without delaying use of the pinned bytes. Preserve
all C67S constructs and flag the three borrowed alpha3 alleles.

The archive contains **75 groove MSAs, five ectodomain MSAs, and one beta2m
MSA**. Validate and normalize the five ectodomain alignments to 275 query
columns and generate the missing 70. Use the supplied 99-residue beta2m
alignment if valid. Prepare B and C from a common ordered row set that survives
both pinned parsers' selection rules in full and cropped forms. Serialize the
same canonical rows as Boltz CSV and ESMFold2 A3M, including the same arm A
groove and beta2m alignments. Respect lowercase A3M insertions and verify
identical B/C processed groove rows and applicable deletion, pairing, and
profile features within each model; verify shared biological MSA content across
models. Use the same effective depth cap if either parser needs a lower limit. Boltz independent MSAs use CSV keys `-1`; validate the equivalent
unpaired ESMFold2 input. The peptide is single-sequence for both models.
Keep server requests off GPU workers.

**4c.2 — Complete CPU preflight.** Adapt the input builder, pose checker, feature
mapping, and resumable launchers for three chains in both models. Pin
`boltz==2.1.1`, weight
revision `6fdef46d763fee7fbb83ca5501ccceff43b85607`, one diffusion sample,
three recycling steps, 200 sampling steps, mmCIF, full PAE, and an MSA parse
cap of 8,192. Omit `--subsample_msa` to select this version's false CLI default
and record the effective setting and retained depth. Pin `esm==3.4.1.post1`
and the exact `biohub/ESMFold2` snapshot revision and hashes; the existing
ESMFold2 downloader needs a revision pin. Use native multi-chain inputs with
MSAs enabled, one diffusion sample, 200 sampling steps, and `num_loops=20`.
Fix the current ESMFold2 single-sequence default and hard-coded seed 0.
Boltz recycling and ESMFold2 loops are different internal algorithms; hold
these model configurations fixed across arms and report them. Both save mmCIF,
full PAE, pLDDT, and available confidence scores in a common schema. Record
model/run identities, seeds, shard ordering, and input hashes; verify mappings.

**Executed preprocessing configuration:** shared 1,024-row cap; both model
adapters and B/C groove-feature checks passed, including cross-model decoded
rows and insertion/deletion features. ESMFold2 uses `msa_max_depth=1024` and
`msa_column_mask_rate=0.0` to avoid its default row sampling and column masking;
Boltz subsampling is disabled. Full raw alignments are retained. All 75
ectodomain MSAs are now prepared. ESMFold2 weight revision is
`69869f737beffec5294845ede23db5fc0b4f509e`. Evidence and live pilot results
are under `reports/ectodomain-20261004/`.

**4c.3 — Run the matched 90-fold pilot.** Use three seeds (0, 1, 2) on each of
the five training-split complexes in `reports/boltz_pilot.csv`, for all three
inputs with each model: **45 Boltz-2 + 45 ESMFold2 predictions**:

| Pilot arm | Construct | HLA alignment |
|---|---|---|
| A | Groove + peptide | Existing groove MSA |
| B | Ectodomain + beta2m + peptide | Selected ectodomain MSA |
| C | Groove + peptide | The same selected rows as B, cropped to 182 query columns |

Both models run all three arms with the same prepared inputs. The arm letters
identify constructs and alignments; model identity is recorded separately.
Pin reference crystals before scoring:
A*11:01/KTFPPTEPK -> 1X7Q; A*02:01/LLWNGPMAV -> 5N6B;
B*15:01/ILGPPGSVY -> 1XR9; B*07:02/IPRRNVATL -> 7LFZ;
B*08:01/ELRRKMMYM -> 4QRU. The supplied 6JOZ sequence-coherence check concerns
a different A*11:01 peptide and does not replace the KTFPPTEPK reference.

Score new A/B/C predictions in the same groove reference frame: fit on common
HLA CA atoms at residues 1-182, then measure the peptide. Report CA and
heavy-atom RMSD plus per-position deviation with one atom-matching policy.
Start with one B fold per model to validate outputs and extraction, then finish
the pilot.
Measure full-construct steady-state throughput, startup, peak VRAM, host memory,
and billed cost; the old two-chain multiplier is not a resource measurement.

Apply the same operational gate separately to each model, fixed before the
new pilot. B-versus-A comparisons are within-model; also report cross-model
results on identical arm/complex/seed inputs. Seed numbers do not imply identical
random draws across architectures:

- Every B prediction has valid structures, full PAE, pLDDT, a finite mapped
  feature row, and verified peptide register.
- All five B complexes at all three seeds have peptide CA RMSD <= 2.0 A and
  P2/P9 CA deviations <= 1.0 A.
- B*07:02/IPRRNVATL has median B heavy-atom RMSD <= 1.0 A and improves by at
  least 0.5 A over the new A median scored under the same protocol.
- Each other complex has median B heavy-atom RMSD no more than 0.5 A worse
  than A. Inspect per-position errors as well as means.
- Measured memory headroom and the resource forecast permit the chosen scope.

C succeeding is **not** a reason to reject B: it suggests the new groove MSA
may suffice and identifies a cheaper candidate for later work. B versus C
measures the added construct and its extra evolutionary information together;
it does not isolate alpha3 from beta2m or prove a physical mechanism. If either
model's B fails, resolve or document the failure before scaling the paired
experiment. Do not silently drop one model, switch its MSA policy, or substitute
C for it. Label any revised matched plan or threshold/pilot explicitly.

**4c.4 — Production scope, frozen 4 October 2026 from measured feasibility.**
Construct B: three separate chains, **275 + 99 + 9 = 383 residues**. A and C
were pilot controls and are not folded in production.

**Model: Boltz-2 only.** ESMFold2 failed the 4c.3 gate — median arm-B heavy
RMSD 2.564 A on the B*07:02 sentinel against a 1.0 A bar, improving 0.008 A
over arm A against a 0.5 A bar, with no construct effect in any arm or seed.
This is a **labelled revision of the matched plan, not a silent drop**: the
cross-model comparison it was designed to produce has already been delivered by
the pilot on identical inputs, and ESMFold2 costs 4.55x more per fold
($0.0314 vs $0.0069), so its full-cohort forecast of $884 never fitted
alongside Boltz-2 in the first place. Recorded limits: pose accuracy is not
feature utility, and a five-complex gate is an operational rule rather than a
general claim about ESMFold2 on peptide-MHC. Both belong in the write-up.

**Cohort: all 28,166 pairs, one prediction each at seed 0**, frozen with its
shard schedule in `data/structural_cohort.csv` by
`scripts/freeze_structural_cohort.py`. Join on `(allele, peptide)` and attach
`pair_id` afterwards; splits are loaded, never recomputed.

**Execution: two Modal workspaces in parallel, 10 workers each.** The cohort is
interleaved pair-by-pair across the `a-cheparukhin` and `colleague`
(workspace `sofyaleyn`) profiles, so each half is balanced across alleles and
splits and either half alone stays unbiased. 141 shards of 100 pairs per
profile, A10G at the pilot-measured $1.4812/h shape.

| | Per profile | Total |
|---|---:|---:|
| Pairs | 14,083 | 28,166 |
| Forecast | 67.8 GPU-h, $100.4 | 135.6 GPU-h, $200.8 |
| With 25% margin | $125.5 (ceiling $150) | **$251** |
| Wall at 10 workers | 6.8 h | **~6.8 h in parallel** (8.5 h with margin) |
| Output | ~11 GB | ~22 GB |

Forecasts use the pilot's measured 16.76 s steady fold and 57 s shard startup.
**Production runs the pilot's exact `boltz predict` command**, and the 5-case
smoke reproduces the pilot's fold time and GPU peak on it. Reserve export,
feature extraction,
and evaluation time before the deadline; for parallel runs use the later
finish, for serial runs sum elapsed times.

**Before launching:** the production runner is new code, so run its 5-case
end-to-end check (`::smoke`) in each workspace first, per the project
invariant. Re-running `::production` resumes: only shards without a committed
success marker are folded again.

**Retained fallback.** Full scope now fits for Boltz-2 alone, so the panel
below is no longer the planned scope. Keep it as the contingency if the
overnight run cannot complete or must be abandoned: freeze this approximately
**2,000-pair, six-allele panel (2,000 Boltz-2 predictions)** from existing
splits without consulting labels, in advance and never by keeping whichever
pairs happened to finish. Reuse an already valid frozen panel if present;
otherwise use these quotas:

| Allele | Panel pairs |
|---|---:|
| HLA-B*15:01 | 434 |
| HLA-A*02:01 | 415 |
| HLA-A*03:01 | 349 |
| HLA-B*39:01 | 276 |
| HLA-B*35:01 | 264 |
| HLA-B*07:02 | 262 |

Preserve split proportions, record selection seed/algorithm, verify >= 50 test
and >= 20 validation rows per allele, and save `data/structural_panel.csv`
before folding. Never recompute `data/splits.csv`. If neither scope fits,
record the same reduced scope for both models in advance or retain the pilot
result; unfinished output must not become a convenience evaluation subset.

**4c.5 — Batch, export, and extract.** Use bounded shards within function timeouts,
configured concurrency, resident weights, and durable checkpoints. Validate the
mmCIF and all required arrays before declaring success; a directory alone is
not sufficient. Allow at most one retry per failed pair within the resource
forecast, record seed/order changes, retain unresolved failures, and stop new
dispatches if forecasts no longer fit.

Join inputs on `(allele, peptide)` and then attach the raw `pair_id`. Key output
records by `(model, allele, peptide)` and share the frozen pair/order/seed
schedule across model queues. Save model, run, construct, MSA, seed/shard/order,
output hashes, attempts, cost, and failures.
Verify the anticipated 383 x 383 PAE and token mapping against actual output.
Core features use peptide pLDDT, peptide/groove PAE in both directions, and
contacts/burial against HLA residues 1-182. Additional ectodomain/beta2m
features need their own definitions. Global ipTM remains global; use a pair
score only if the pinned model emits it with a verified mapping.

**Deliverable:** pinned input and shared MSA manifests, both CPU preflights,
90 pilot predictions and per-model gate verdicts, shared frozen scope and
separate plus combined resource forecasts, durable production outputs with
success/failure records, and validated feature tables.
Stage 5 compares features on matched rows; stage 6 retains the frozen evaluation
contract and one final test scoring. Export artifacts before event resources
are removed.

### 5. Test additional feature groups

**Work**

- Use both models' stage 4c full-construct predictions. Test geometry and confidence
  separately, then combinations supported by validation, on top of both the
  sequence baseline and ESM-2 features.
- Train comparators on **exactly the same structural training rows**, with
  identical validation and test rows. Reuse compatible full-data baselines at
  full scope. At panel scope, refit baseline heads on panel training rows using
  cached inputs/embeddings; also report full-training-data models on the same
  panel held-out rows as practical comparators. No new embedding extraction is
  required solely by the structural construct change.
- Keep usable low-confidence predictions, report coverage and outright failures,
  and show a sequence-model fallback for unresolved structural failures.
- Compare Boltz-2 and ESMFold2 on the same production rows using shared
  feature definitions, head architecture, ensemble size, and tuning budget.
  Label model-specific extra confidence features separately; test combinations
  only if validation supports them. Report the frozen cohort with the declared
  sequence fallback and common successful rows as an additional diagnostic.
- Limit A/B/C construct comparisons to the shared pilot rows. Their 90 folds
  do not provide a dataset-wide structural ablation.
- Try multiple poses or extra feature groups only on a small diagnostic subset
  if time and budget remain and validation supports the work.

**Deliverable:** matched-row geometry/confidence ablations showing incremental
predictive value, uncertainty, coverage, and compute cost.

**Why:** a structural feature can be useful alone yet add nothing beyond the
sequence baseline. Model confidence is not physical stability; seed variation
is model uncertainty rather than molecular motion. Better crystal agreement
is a pipeline/pose result and must earn its predictive value on validation.
Global three-chain ipTM must not be presented as peptide-interface confidence.

### 6. Evaluate and prepare the submission

**Work**

- Evaluate the validation-selected models **once** on the held-out test set.
- Report per-allele Spearman correlation (how well the model ranks peptides within each allele), with test-set sizes and a median/IQR summary across alleles. Use a common set of eligible alleles across models and report small or undefined cases explicitly.
- Report MAE on `log1p` half-life for numerical error. Add **precision@10 at a predeclared 2-hour threshold** — of the top 10 predictions per allele, how many actually have a half-life above 2 hours? This directly measures whether the model identifies sufficiently stable peptides. Treat pooled metrics as secondary. Distinguish within-allele ranking from cross-allele effects.
- **Stratify test metrics by nearest-neighbour distance to training.** For each test peptide, compute the Hamming distance to its closest training peptide and report metrics in **two strata: d=4 (57.8% of test rows) and d≥5 (42.2%)**. A separate d≥6 stratum is not viable — measured on the frozen split only 6 test peptides (12 rows) sit that far from training. Score both strata on the *same* allele set (the intersection of those eligible in each, 65 alleles at a 20-row bar), or the comparison measures allele panels rather than distance. If label similarity decays as expected, performance should visibly differ across strata. If it doesn't, that's a strong signal the model genuinely generalises rather than exploiting residual similarity at the split boundary.
- **The distance-stratum figures above are test-specific.** Verified 4 October
  against the frozen splits: test is d=4 3,256 rows (57.80%) / d>=5 2,377
  (42.20%), and d>=6 holds 12 rows across 6 peptides, confirming it is not
  viable as a third stratum. **Validation sits at 61.41% / 38.59%** (d=4 1,730
  rows; d=5 1,072; d=6 15 rows across 3 peptides). The two splits differ, so
  57.8/42.2 must never be restated as a dataset-wide fact.
- **The 20-row stratum bar holds on test but not on validation.** On test it
  leaves 67 eligible alleles at d=4 and 66 at d>=5, intersecting at 65, as
  stated. Validation is half the size, so each stratum holds roughly 1.4k rows
  and the same bar leaves a 6-allele shared panel covering only 15.2% / 21.7%
  of the two strata — too thin to compare. Validation therefore uses a
  **10-row bar**, giving 55 alleles at 79.6% / 91.3% coverage. The frozen
  constant in `pepstab/evaluation.py` is unchanged and test keeps 20; the bar
  is a CLI argument with a coverage warning. **The validation bar was chosen
  from row counts and coverage alone, before any model was scored** — a
  stratum bar chosen after seeing performance would be a selection effect that
  no reader could detect from the resulting number.
- **Nested near-neighbour evaluation inside training.** Cross-validate mutant ranking on the d≤2 peptide clusters that live entirely within the training split. This recovers the question the grouped split cannot answer — can the model rank point mutants of a known binder? — without touching the test set or the frozen split assignments.
- **The differential target.** For peptides measured on two or more alleles, evaluate Δ log half-life *between* alleles. This subtracts out whatever is intrinsic to the peptide and tests groove chemistry directly, which is the sharpest available version of "distinguish within-allele ranking from cross-allele effects". It is abundant — **3,941 peptides sit on ≥2 alleles, covering 26,474 rows (94% of the dataset), up to 36 alleles for a single peptide** — and it costs no new compute, being a re-aggregation of predictions already made.
- Use paired uncertainty estimates that keep peptide clusters together across alleles.
- Report accuracy gains alongside extraction/training cost, **cost per 1,000 new predictions**, runtime, and prediction failures.
- If a confidence interval crosses zero, the result is inconclusive — not negative. A strong negative result should rule out the predeclared minimum worthwhile gain. Bound every conclusion to the specific representation, data, split, and budget tested.
- If time remains, add an allele-held-out evaluation (train without some alleles, test on them), stratified by how similar the held-out alleles are to training alleles. Only then make any new-allele generalisation claim — and report the confound described in the out-of-scope register alongside it.
- The original assay panel was partly selected by predicted affinity, so broader biological or clinical claims need additional evidence.

**The single-pass rule, predeclared 4 October 2026 before any test number existed.**
The test set is scored **once**, in one pass, covering every arm whose
validation-selected configuration is frozen at the cutoff. This creates a real
scheduling tension worth stating plainly: the structural fold lands ~10:50 BST
and feature extraction follows it, while the sequence and ESM-2 arms are ready
much earlier. Scoring test early would forfeit any structural test number;
waiting indefinitely risks scoring nothing. The rule resolves it in advance:

1. **Cutoff.** Freeze arms at a cutoff set **three hours before the submission
   deadline**, leaving time for the pass itself, uncertainty estimates, figures,
   and the write-up. Nothing is scored on test before the cutoff, and nothing
   is added after it.
2. **Any arm not frozen by the cutoff is reported on validation only**, and the
   report says so explicitly rather than omitting the arm. A missing test number
   is a scheduling fact, not a result, and must not be presented as one.
3. **The cutoff does not move because an arm is nearly ready.** That is the
   failure mode this rule exists to prevent: an arm that slips past the cutoff
   and is then waited for is an arm selected by its own convenience.
4. **Partial structural coverage is admissible; convenience subsets are not.**
   The two Modal halves are interleaved pair-by-pair, so a single completed half
   is balanced across alleles and splits and may be scored as a labelled,
   pre-declared half-cohort diagnostic. Whichever pairs merely *happened to
   finish* by the cutoff is **not** an admissible subset, and the declared
   sequence-model fallback covers unresolved structural rows.
5. **Arms are matched or they are not compared.** Any arm entering the pass
   carries the same ensemble size, seed protocol and tuning budget as the arms
   it is compared against. An unmatched arm is reported, but separately, and
   labelled as not comparable.

**Deliverable:** reproducible code and configs, final comparison table, limitations section, and a concise presentation.

**Why:** the submission should show what helped, where it helped, and what it cost. A gain on unseen peptides for familiar alleles is useful even without a new-allele result.

## Budget and GPU decision rule

**Revised 4 October 2026.** Modal credit now spans **two workspaces holding
approximately $300 each** — `a-cheparukhin` and `sofyaleyn` (local profile
`colleague`) — plus approximately **$60 Hugging Face**. Keep providers and
workspaces separate: credit in one workspace cannot pay for work in the other,
so each half of the cohort must fit its own workspace's balance on its own.
These are ceilings, not spending targets or a claim about the remaining
balance.

| Allocation | Ceiling | Current use |
|---|---:|---|
| HF: embedding extraction and regression experiments | $60 | Core sequence/ESM-2 comparison. |
| Modal `a-cheparukhin`: pilots to date | $15 total | Stage 4c pilot spent **$1.49** of this; prior benchmark spend included. |
| Modal `a-cheparukhin`: production half | $150 maximum | 14,083 pairs; forecast $100.4, $125.5 with margin. |
| Modal `colleague`/`sofyaleyn`: production half | $150 maximum | 14,083 pairs; forecast $100.4, $125.5 with margin. |
| Modal: contingency | remainder of each balance | Reserve; not automatically available to the folding launcher. |

The per-profile $150 ceiling is checked by hand: the production entrypoint
prints its forecast with margin (`--dry-run` prints it without spawning
anything), and that figure is what to compare against the ceiling before
launching. ESMFold2 production is **not** funded: it
failed its gate (4c.4), and its $884 full-cohort forecast exceeded the
combined balance regardless.

At launch, verify actual credit, prior spend, other commitments, and current
GPU/CPU/memory rates in **each** workspace. A budget table does not authorize
spending unavailable credit.

Historical worker rates, two-chain timings, and right-sizing measurements are
in `reports/stage4_benchmark.md`. Measure the 383-residue construct and choose
the cheapest worker shape that meets memory and deadline requirements. Keep
weights resident and use prepared MSAs; concurrency shortens elapsed time but
does not reduce total compute cost.

```text
forecast_cost = 1.25 * (N * steady_cost_per_success
                       + total_startup_cost + export_storage_cost)
forecast_time = 1.25 * ((N * steady_seconds_per_success
                        + total_startup_seconds) / available_workers)
                + export_and_feature_time
```

Include failed attempts in per-success measurements, account for expected
restarts, and avoid adding startup twice if it is already in a quoted cost.
Plan from observed allocation variability, verify actual concurrency on the
first production shards, and refresh cost and completion forecasts at shard
boundaries. The 25% margin is an operational allowance, not a statistical
guarantee. Calculate costs per model and sum them. Allocate shared GPU capacity
explicitly across both queues; for parallel runs use the later finish, and for
serial runs sum elapsed times. Stop dispatching new work if the paired scope
no longer fits.

## Team workstreams and checkpoints

| Participant | Primary responsibility | Handoff |
|---|---|---|
| 1 | Frozen data contract, sequence baselines, matched-row evaluation | Validated joins, comparator predictions, evaluation format, and final uncertainty estimates. |
| 2 | ESM-2 embeddings, regression heads, representation comparison | Cached embeddings and validation-selected models; matched panel heads if needed. |
| 3 | Stage 4c shared inputs/MSAs, both model pilots and production, structural features | Pilot verdict, frozen scope/resource forecast, exported outputs, and feature/coverage manifests. |

The original engine pilots and groove MSA cache are complete. Participant 3's
remaining work is the full-construct input/MSA preparation, corrected scoring
and feature mapping for both models, matched 90-fold pilot, and resumable
production workflows on the same cohort. The core sequence/ESM-2 comparison
can continue while this preparation runs.

- **Before GPU work:** pin supplied inputs, complete MSAs and CPU preflight,
  and record the absolute completion deadline in Europe/London.
- **After the new pilot:** apply the quality/resource gates, verify credit and
  available concurrency, and freeze full scope or the six-allele panel.
- **During production:** check forecasts and output completeness at shard
  boundaries; export durable checkpoints while the core comparison progresses.
- **Before final scoring:** finish features, freeze validation-selected model
  configurations, and reserve time for the single test evaluation, uncertainty,
  figures, and presentation.
- Save code, data, MSAs, splits, features, checkpoints, and results locally
  before temporary Antigravity event resources are deleted after the event on
  Sunday, October 4. Keep credentials out of exported artifacts.
- **Stage 4c structures are the exception: they stay on Modal.** 22 GB lives on
  the `pepstab-structures` Volume in each of the two workspaces, which are
  personal accounts and are not deleted with the event resources. Analysis runs
  on Modal with the Volume mounted rather than against a local copy; share
  access by inviting people to both workspaces. A Volume cannot span
  workspaces, so there is no single shared Volume for the whole cohort.

## Stretch work and stopping rules

- The auxiliary affinity experiment (stage 3b) may interact with the ESM-2 comparison: if multi-task training helps the sequence arm more than the ESM arm, report that result rather than burying it.
- Same-peptide / different-HLA diagnostics are possible with this data. **Matched C67S / wild-type comparisons are not** — the corresponding wild-type alleles aren't in the dataset.
- Only expand a feature pipeline when validation evidence or useful uncertainty information supports the extra cost. Don't claim an unbinding mechanism from improved prediction alone.

**Minimum successful submission:** reliable splits, a trained sequence baseline, an ESM-2 comparison, uncertainty estimates, and measured costs. Structural experiments strengthen this result but must not prevent completing the core study.

## Out of scope this round

The following work is deferred or rejected for this round. Completed engine diagnostics and the active ectodomain workflow are recorded in stage 4.

| Item | Why out of scope this round |
|---|---|
| **Allele-axis hold-out** | **Promoted 4 October 2026** to `stage7_allele_holdout`. The long-form rationale below still stands in full and is reported alongside every number it produces — reason 1 is a confound, not a caveat |
| Template threading | We co-fold instead. Threading assumes the canonical register and cannot represent the non-canonical bulges that may be exactly the unstable complexes. `data/allele_pdb_templates.csv` (33 tier-A exact-groove templates) makes it cheap if this is ever revisited |
| FoldX, Rosetta, any empirical energy layer | **Still out, and now also blocked by access.** The scientific objection is unchanged: these estimate equilibrium ΔG, not the ΔG‡ barrier to unbinding, which is a known mismatch with a kinetic label. Independently, FoldX is proprietary and gated behind registration and a per-user licence file, and Rosetta/PyRosetta likewise require a licence; neither is installed and neither can be obtained without the user's own account. Requested as a stretch goal on 4 October and declined on both grounds |
| Per-pocket energy decomposition (A/B/F ↔ P1/P2/PΩ) | A good idea, but premised on FoldX `AnalyseComplex`; it falls with FoldX, which is now licence-blocked as well as scientifically mismatched |
| OpenMM minimisation energy | Not a drop-in replacement for interface scoring, and the same thermodynamic/kinetic mismatch applies |
| ProteinMPNN | **Promoted 4 October 2026** to `stage5_inverse_folding`. The blocker was structures; the stage 4c pilot left 90 folds on local disk, so the pilot and the scale-ready harness no longer wait on production. Inverse folding is the third of the three model classes the challenge brief names, so covering it turns a two-class answer into a three-class one |
| ESMFold v1 (`facebook/esmfold_v1`) | No native multi-chain support or ipTM; its frozen ESM-2 trunk also overlaps the model family under test at stage 3. |
| Different input policies for the two folding models | Primary comparison fixes constructs and MSA content. Single-sequence ESMFold2 versus MSA-assisted Boltz-2 is deferred as a separate experiment. |
| Chai-1 | A third folding engine adds integration cost beyond the agreed Boltz-2/ESMFold2 comparison. |
| Separate alpha3/beta2m mechanism ablations | Stage 4c tests the full input pipeline. Additional mechanistic controls are deferred until its predictive value and resource feasibility are established. |
| Chimeric peptide-linker-groove ESM input | **Promoted 4 October 2026** to a labelled stage 3 diagnostic, not a headline arm. The original objection stands — a linkered 9-mer sits far outside ESM-2's distribution — but leaving it untested leaves the obvious hole in a negative result: that we never let the model see the complex. Running it lets us report *why* it fails with evidence rather than with an argument |
| Tobit / censored likelihood | **Promoted 4 October 2026** to `stage7_censored`. It is the principled treatment of the 20.2% floor, which is a detection limit rather than a measurement; `log1p` plus rank-led metrics was the time-pressured substitute. Expect it to improve calibration near the floor rather than rank correlation, since ranking inside the tied floor block is unidentifiable either way |
| Source-protein / UniProt mapping, gene-level splits | The splits are frozen and cannot be rebuilt |
| ESMC / ProtT5 as a second pLM family | **Conditionally promoted 4 October 2026**, queued behind stage 3 and stage 3d. One model family is a thin basis for "foundation models don't help"; a second says whether the stage 3 conclusion is family-specific. Runs only if RAM allows — six workstreams share 16 GB |
| SaProt, cross-attention, folding-trunk features, geometry-aware GNNs, extensive interpretability probes, molecular-dynamics unbinding | Defer until the core comparison is secure. SaProt's blocker is softening now that structures exist, but it stays deferred behind the three promoted items above |

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
