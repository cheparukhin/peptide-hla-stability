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
| Frozen ESM-2 | Core | Does a pretrained protein language model add useful signal? Testable across the full dataset, no structures needed. | Learned vector representations of each peptide position and HLA sequence (from ESM-2), used alone and combined with raw sequence features. | Improvement over the sequence baseline; embedding extraction cost; sensitivity to which internal layer and model size we use. |
| Auxiliary affinity | Optional, pre-structural | Does training on binding-affinity labels alongside stability labels improve the model? No extra GPU cost. | A shared encoder with a second output head predicting affinity (how strongly a peptide binds, rather than how long it stays). Tested first on peptides that have both measurements, then optionally on broader IEDB data. | Gain over single-task baseline and over ESM-2; whether the benefit differs between the sequence and ESM arms. |
| Boltz geometry | Pilot-gated | Does the predicted physical fit in the HLA groove explain stability? Small panel, because folding is expensive. | Contacts and burial (how deeply each peptide position sits in the groove); hydrogen bonds at the peptide ends; clashes at anchor pockets (positions where the peptide is pinned down). | Added accuracy on matched examples; whether the predicted poses look physically reasonable; cost and failure rate. |
| Boltz confidence | Same structures | Does the model's own uncertainty carry signal? Tested separately from geometry, reusing the same predicted structures. | Per-position pLDDT plus its peptide mean and minimum (pLDDT = per-residue confidence in the local structure around that residue); peptide-HLA PAE (predicted alignment error — how sure the model is about the relative placement of the two chains); pairwise ipTM (interface predicted TM-score — overall interface quality). | Gain from confidence alone and beyond geometry. |
| ProteinMPNN | Optional next | Is the peptide sequence "compatible" with the predicted backbone shape? Uses another pretrained model, no refolding needed. | Peptide-only overall likelihood and per-position scores from ProteinMPNN (an inverse-folding model that asks: given this 3D backbone, how probable is this amino acid sequence?), with HLA held fixed. | Predictive value beyond sequence and structure features, versus scoring cost. |
| FoldX | Optional last | Do estimated binding energetics add information? Only if FoldX is already available. | Peptide-HLA interaction energy and its component terms (van der Waals, electrostatics, etc.), computed after repairing the predicted structure. | Incremental accuracy versus setup and scoring cost. |

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

**Status: done.** `reports/audit_summary.md` (audit), `data/splits.csv` (frozen splits, committed; carries `cluster_id`, `split`, and `dist_to_train` per `pair_id`), `EVALUATION.md` (the predeclared contract), `pepstab/` + `scripts/evaluate.py` (shared scoring), `tests/test_contract.py` (30 guards). Load splits with `pepstab.data.load_with_splits()`; never recompute them.

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

**Why:** this shows what the task's labelled data can teach a small model on its own. It's not a reproduction of NetMHCstabpan's training and shouldn't be described as one.

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
- **Expand (only if probe helps, ~2-3 hours):** Bring in the broader IEDB affinity dataset (~136K measurements, 152 alleles). Before training, verify that no IEDB peptide is within one residue change of any stability test-set peptide; exclude any that are. Train multi-task models for both the sequence baseline and the ESM-2 arm.
- Report whether auxiliary affinity data helps each arm differently. If multi-task training closes the gap between the sequence baseline and ESM-2, that's worth reporting — it would mean cheap extra labels substitute for expensive pretrained features on this task.

**Deliverable:** multi-task vs. single-task comparison on the same frozen validation set, with the leakage audit documented.

**Why:** the stability dataset's peptides were pre-selected for strong predicted affinity, so peptide diversity is limited. IEDB affinity data covers far more peptides and alleles. Multi-task training lets the shared encoder see that broader diversity during training without changing the stability evaluation. Rasmussen et al. showed that combining affinity and stability data improved epitope prediction beyond either alone (p<0.001), with the gain coming from complementary signal, not just more rows.

### 4. Pilot structure prediction and choose scale

**Work**

- Use Boltz-2 for structure prediction; keep Chai-1 as a fallback rather than running both. Input: the supplied 182-residue HLA domain plus a separate 9-residue peptide chain.
- **Push 3-5 training/validation examples all the way through folding, feature extraction, and prediction before launching any batch.** Visually inspect the predicted poses to check the peptide sits in the HLA groove, and verify residue/chain mapping.
- Benchmark 20-30 representative complexes with identical settings, comparing L40S, A100 40 GB, and H100 GPUs. Measure billed cost per successful complex, throughput, peak memory, startup overhead, and failure rate.
- Cache model weights and HLA MSAs. Use single-sequence peptide input. Keep models loaded across complexes to avoid reload overhead. Don't spend GPU time waiting on MSA generation. Start with one pose per complex and standard settings. **Save full PAE** (the matrix of predicted alignment errors — needed for feature extraction). Test reduced sampling only against pilot quality.
- Choose roughly **2,000 complexes across 4-6 well-represented alleles**, independently of model performance and test labels. Preserve the frozen splits. Target **at least 50 held-out examples per included allele, ideally closer to 100**.
- Moving to full-length HLA with beta-2-microglobulin (the additional chain that stabilises the HLA structure in vivo) would require a new pilot and runtime benchmark.
- **At hour 5, expand only if the complete pipeline works and measured throughput supports finishing by hour 10.** Otherwise shrink the panel or report just the pilot results.

**Deliverable:** hardware and runtime benchmark, a fixed structural panel, the structures themselves, and a manifest of successes and failures.

**Why:** folding is the biggest compute and integration risk. Cheaper-per-hour hardware may be slower per structure, so the metric that matters is measured cost per completed prediction. A smaller, interpretable experiment with adequate test coverage is worth more than many structures that can't support a comparison.

### 5. Test additional feature groups

**Work**

- Start with geometry and confidence features, which come from the existing structure predictions. Add each group separately, then test combinations that validation supports.
- Score only the peptide chain with ProteinMPNN (holding HLA fixed), so the much larger HLA chain doesn't dominate the score. Do this once the core pipeline is complete.
- If FoldX is already available, run `RepairPDB` (fixes common structural artifacts) then `AnalyseComplex` (decomposes binding energy into physical terms) for peptide versus HLA. Use interaction energy, not total complex energy.
- Train sequence, ESM, and augmented models on **exactly the same structural training rows**, with identical validation and test rows. Separately report full-training-data sequence models on that same test set as practical comparators.
- Keep low-confidence but usable predictions. Predefine how to handle outright failures, report coverage, and show a sequence-model fallback for structures that fail.
- Try multiple poses only on a small diagnostic subset if budget remains.

**Deliverable:** ablation table showing the incremental predictive value and cost of each feature group.

**Why:** a feature can look useful alone but add nothing on top of the sequence baseline. Important conceptual guardrails: pLDDT is the model's confidence in its prediction, not a measure of physical stability; seed disagreement captures model uncertainty, not real molecular motion; inverse-folding likelihood tells you if the sequence fits the backbone, not how long the complex lasts; interaction energy estimates the strength of binding, not the energy barrier to unbinding. OpenMM minimisation energy is not a drop-in replacement for FoldX interface scoring.

### 6. Evaluate and prepare the submission

**Work**

- Evaluate the validation-selected models **once** on the held-out test set.
- Report per-allele Spearman correlation (how well the model ranks peptides within each allele), with test-set sizes and a median/IQR summary across alleles. Use a common set of eligible alleles across models and report small or undefined cases explicitly.
- Report MAE on `log1p` half-life for numerical error. Add **precision@10 at a predeclared 2-hour threshold** — of the top 10 predictions per allele, how many actually have a half-life above 2 hours? This directly measures whether the model identifies sufficiently stable peptides. Treat pooled metrics as secondary. Distinguish within-allele ranking from cross-allele effects.
- **Stratify test metrics by nearest-neighbour distance to training.** For each test peptide, compute the Hamming distance to its closest training peptide and report metrics in **two strata: d=4 (57.8% of test rows) and d≥5 (42.2%)**. A separate d≥6 stratum is not viable — measured on the frozen split only 6 test peptides (12 rows) sit that far from training. Score both strata on the *same* allele set (the intersection of those eligible in each, 65 alleles at a 20-row bar), or the comparison measures allele panels rather than distance. If label similarity decays as expected, performance should visibly differ across strata. If it doesn't, that's a strong signal the model genuinely generalises rather than exploiting residual similarity at the split boundary.
- **Nested near-neighbour evaluation inside training.** Cross-validate mutant ranking on the d≤2 peptide clusters that live entirely within the training split. This recovers the question the grouped split cannot answer — can the model rank point mutants of a known binder? — without touching the test set or the frozen split assignments.
- Use paired uncertainty estimates that keep peptide clusters together across alleles.
- Report accuracy gains alongside extraction/training cost, **cost per 1,000 new predictions**, runtime, and prediction failures.
- If a confidence interval crosses zero, the result is inconclusive — not negative. A strong negative result should rule out the predeclared minimum worthwhile gain. Bound every conclusion to the specific representation, data, split, and budget tested.
- If time remains, add an allele-held-out evaluation (train without some alleles, test on them), stratified by how similar the held-out alleles are to training alleles. Only then make any new-allele generalisation claim.
- The original assay panel was partly selected by predicted affinity, so broader biological or clinical claims need additional evidence.

**Deliverable:** reproducible code and configs, final comparison table, limitations section, and a concise presentation.

**Why:** the submission should show what helped, where it helped, and what it cost. A gain on unseen peptides for familiar alleles is useful even without a new-allele result.

## Budget and GPU decision rule

Team credits: **$450 Modal + approximately $60 Hugging Face**. Keep provider budgets separate. The figures below are **ceilings, not spending targets**.

| Allocation | Ceiling |
|---|---:|
| HF: embedding extraction and regression experiments | $60 |
| Modal: end-to-end and hardware pilot | $15 |
| Modal: structure predictions | $330 |
| Modal: inverse-folding and energy scoring | $45 |
| Modal: contingency | $60 |

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
| 3 | GPU pilot, structures, additional feature extraction | Structure and feature manifests keyed by complex ID, with success and failure records |

- **First 3 hours:** freeze data and evaluation, establish a baseline, begin ESM extraction, complete the small structural pilot.
- **By hour 5:** review validation results. Make the structural go / reduce / stop decision. Run the auxiliary affinity probe if the core comparison is on track.
- **Hours 5-12:** finish the selected workload and ablations. Stop adding features at hour 10; freeze configurations by hour 12.
- **Final 4 hours:** evaluate on held-out data, compute uncertainty, prepare figures and the presentation, save deliverables.
- Adjust these cutoffs if the official deadline requires it.
- Save code, data, splits, features, checkpoints, structures, and results locally before the **Antigravity event resources are deleted after the event on Sunday, October 4**. Modal and Hugging Face credits are separate per-participant offers and aren't affected by that deletion. Keep credentials out of shared manifests and exported artifacts.

## Stretch work and stopping rules

- The auxiliary affinity experiment (stage 3b) may interact with the ESM-2 comparison: if multi-task training helps the sequence arm more than the ESM arm, report that result rather than burying it.
- Defer until the core comparison is secure: SaProt, chimeric inputs, cross-attention, folding-trunk features, template threading, new geometry-aware GNNs, extensive interpretability probes, source-protein mapping, and molecular-dynamics unbinding simulations.
- Same-peptide / different-HLA diagnostics are possible with this data. **Matched C67S / wild-type comparisons are not** — the corresponding wild-type alleles aren't in the dataset.
- Only expand a feature pipeline when validation evidence or useful uncertainty information supports the extra cost. Don't claim an unbinding mechanism from improved prediction alone.

**Minimum successful submission:** reliable splits, a trained sequence baseline, an ESM-2 comparison, uncertainty estimates, and measured costs. Structural experiments strengthen this result but must not prevent completing the core study.

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
