# Peptide-HLA stability: hackathon execution plan

**Team:** 3 participants
**Event:** London AI × Science, protein engineering track, October 3-4, 2026
**Research question:** do frozen protein foundation-model features improve prediction of how long a peptide remains bound to HLA, and does any improvement justify its compute cost?

The primary claim concerns **unseen peptides on familiar HLA alleles**. New-allele generalisation is a separate test, not the headline result.

## Recommended scope

Commit to a full-dataset comparison of a supervised sequence baseline against frozen ESM-2 features. Add a bounded structural experiment using geometry and confidence **only if an end-to-end pilot passes**.

Do not commit to folding the whole dataset. A precise negative result is a successful submission; structural work must not prevent completion of the core comparison.

## Scientific rationale

HLA molecules display short protein fragments for immune surveillance. The target here is **residence time**: how long a peptide remains bound once a complex has formed. Binding affinity also depends on how readily binding occurs, so a good-looking predicted structure or a favourable interaction energy is not automatically evidence of a long half-life. We test these quantities as candidate predictors of the measured half-life, not as measurements of it.

The experiment ladder asks whether each source of information adds value **beyond a trained sequence baseline**. Start with inexpensive full-dataset comparisons; pay for structures only after a small pilot works end to end; then reuse those structures to test geometry, confidence, sequence compatibility, and energy separately.

The baseline, ESM extraction, and the folding pilot can run in parallel. The ordering below describes the evidence needed before expanding scope, not a strict serial schedule.

| Experiment | Priority | Scientific question | Features and their meaning | What we track |
|---|---|---|---|---|
| Sequence baseline | Core | What can we learn from the labelled sequences alone? Establishes the inexpensive reference. | Peptide and HLA amino-acid encodings, preserving residue positions. | Peptide ranking, prediction error, training/inference cost. |
| Frozen ESM-2 | Core | Does protein pretraining add useful information? Testable across the dataset without structures. | Learned representations of peptide positions and HLA sequence; alone and added to raw sequence features. | Improvement over the sequence baseline; embedding cost; sensitivity to layer and model size. |
| Auxiliary affinity | Optional, pre-structural | Does multi-task training with binding-affinity labels improve the stability encoder? No GPU cost. | Shared encoder with a second head predicting affinity; tested on dual-labelled peptides first, then optionally on broader IEDB data. | Gain over single-task stability baseline and ESM-2; whether the benefit differs between sequence and ESM arms. |
| Boltz geometry | Pilot-gated | Does the predicted fit in the HLA groove explain stability? Small panel, because folding is costly. | Contacts and burial per peptide position; terminal hydrogen bonds; anchor-pocket clashes. | Added accuracy on matched examples; credible groove poses; cost and failures. |
| Boltz confidence | Same structures | Does uncertainty in the prediction carry signal? Tested separately from geometry, reusing the same folds. | Per-position and peptide mean/minimum pLDDT (local confidence); peptide-HLA PAE (relative placement); pairwise ipTM (interface). | Gain from confidence alone and beyond geometry. |
| ProteinMPNN | Optional next | Is the peptide sequence compatible with the predicted backbone? Another pretrained model, no refolding. | Peptide-only likelihood and per-position scores, with HLA fixed. | Predictive value beyond sequence/structure features, versus scoring cost. |
| FoldX | Optional last | Do estimated interface energetics add information? Setup is conditional. | Peptide-HLA interaction energy and individual terms, after structure repair. | Incremental accuracy versus setup/scoring cost. |

Each experiment must earn its place by improving prediction on the same held-out examples, with uncertainty and compute cost reported alongside accuracy. A useful negative result tells us which representation did not help under these conditions; it does not rule out every use of that model family.

## Dataset and known facts

**Canonical input:** the corrected raw CSV committed under `data/`. Preserve it unmodified alongside its checksum so every workstream uses identical data.

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
| Largest one-residue-neighbour peptide cluster | 5 peptides |

Inputs are `peptide`, `hla_seq`, and `hla_pseudoseq`; the target `thalf_hours` is dissociation half-life.

The 34-position contact pseudosequence identifies potential peptide-contacting HLA positions but gives no peptide-specific 3D geometry. Individual experimental replicates are absent, so the CSV cannot establish a replicate-based noise ceiling. Released NetMHCstabpan was trained on this dataset and is not a fair held-out comparator.

## Stages, deliverables, and justification

### 1. Audit the data and freeze evaluation

**Work**

- Check sequence alphabets, pseudosequence lengths, engineered constructs, missing values, and label distributions.
- Start with `y = log1p(thalf_hours)`, which accommodates zero labels. Preserve original labels for reporting.
- Investigate what zero values and any apparent assay limits mean. Do not infer censoring from repeated values alone or discard zeros automatically.
- Freeze approximately **70/10/20** train/validation/test partitions. Keep identical peptides **and one-residue-neighbour clusters** together, across all alleles. Inspect per-allele counts before fitting.
- Agree on metrics, model-selection rules, **a predeclared practically meaningful improvement**, and shared example identifiers. Keep test outcomes sealed until the final comparison.

**Deliverable:** audit summary, saved split assignments, and a common evaluation script.

**Why:** every model must solve the same generalisation problem without leakage. The larger test partition gives more evidence for within-allele ranking; grouping near-identical peptides removes the main leakage route, and the clusters are small enough that grouping costs little data.

### 2. Establish supervised baselines

**Work**

- Train a small MLP on position-preserving one-hot or BLOSUM encodings of the peptide and HLA contact pseudosequence.
- Add an input-matched baseline using the HLA domain sequence when comparing against domain-based embeddings, so extra sequence context is not mistaken for a pretraining benefit.
- Include training-set allele means as a simple control for error and pooled metrics.
- Use a modest, comparable tuning budget across approaches. Save predictions, configuration, validation performance, and training/inference time.

**Deliverable:** reproducible sequence baselines and a first results table.

**Why:** this establishes what the task's labelled data can teach a small model. It is not a reproduction of NetMHCstabpan's training procedure and should not be described as one.

### 3. Test frozen ESM-2 representations

**Work**

- Start with **ESM-2 35M**; consider a larger checkpoint only if time, throughput, and validation results justify it.
- Cache embeddings once per unique peptide and HLA sequence, not once per measurement.
- Embed peptide and HLA separately. Preserve peptide-position information; do not rely only on mean pooling.
- Embed the supplied HLA domain. Verify residue mapping before selecting the 34 contact-position embeddings.
- Compare a middle layer against the final layer. Train small heads on embeddings alone and embeddings plus raw sequence features.
- Select checkpoint, layer, representation, and head on validation data, and check variation across head-training seeds.

**Deliverable:** full-dataset baseline-versus-ESM comparison, cached features, and measured compute costs.

**Why:** this is the cheapest direct test of the foundation-model question and needs no structures. Separate embeddings leave peptide-HLA interaction learning to the head, so useful performance is a hypothesis, not a guarantee.

### 3b. Test auxiliary affinity training (optional, pre-structural)

**Work**

- **Probe (∼1 hour):** Rasmussen et al. report ∼7,600 peptides with both affinity and stability measurements across 58 allotypes. Add a second prediction head to the sequence baseline that predicts binding affinity alongside stability. Train on these dual-labelled peptides only — they are already inside the frozen splits, so no new leakage surface. Compare single-task versus multi-task validation performance.
- **Expand (only if probe helps, ∼2–3 hours):** Ingest the broader IEDB affinity dataset (∼136K measurements, 152 alleles). Before training, verify that no IEDB peptide is a Hamming-1 neighbour of any stability-set test peptide; exclude any that are. Train multi-task models for both the sequence baseline and the ESM-2 arm.
- Report whether the auxiliary target helps each arm differently. If multi-task training closes the gap between the sequence baseline and ESM-2, that is a finding worth reporting — it means cheap auxiliary labels substitute for expensive pretrained features on this task.

**Deliverable:** multi-task versus single-task comparison on the same frozen validation set, with the leakage audit documented.

**Why:** The stability dataset's peptides were pre-selected for strong predicted affinity, limiting peptide diversity. The IEDB affinity data covers far more peptides and alleles. Multi-task training lets the shared encoder see that diversity during training without changing the stability evaluation. Rasmussen et al. showed that combining affinity and stability data improved epitope prediction beyond either alone (p<0.001), and that the gain came from complementary signal, not just more rows.

### 4. Pilot structure prediction and choose scale

**Work**

- Use Boltz-2; keep Chai-1 as an alternative rather than running both. Input is the supplied 182-residue HLA domain plus a separate 9-residue peptide chain.
- **Push 3-5 training/validation examples all the way through folding, feature extraction, and prediction before launching any batch.** Inspect groove-bound poses and verify residue/chain mapping.
- Benchmark 20-30 representative complexes with identical settings, comparing L40S, A100 40 GB, and H100 where useful. Measure billed dollars per successful complex, throughput, peak memory, startup overhead, and failures.
- Cache model weights and HLA MSAs; use single-sequence peptide input; keep models loaded across complexes; avoid GPU time spent waiting on MSA generation. Begin with one pose per complex and standard settings. **Save full PAE** for feature extraction. Test reduced sampling only against pilot structural quality.
- Choose roughly **2,000 complexes across 4-6 adequately represented alleles**, independently of model performance and test labels. Preserve the frozen splits and seek **at least 50 held-out examples per included allele, nearer 100 where possible**.
- Moving to full soluble HLA plus beta-2-microglobulin requires a new pilot and runtime benchmark.
- **At hour 5, expand only if the complete pipeline works and measured throughput supports completion by hour 10.** Otherwise reduce the panel or report the pilot alone.

**Deliverable:** hardware/runtime benchmark, a fixed structural panel, the structures, and a manifest of successes and failures.

**Why:** folding is the largest compute and integration risk. Cheaper hourly hardware may be slower, so measured cost per completed prediction decides the choice. A smaller interpretable experiment with adequate test coverage is worth more than many structures that cannot support a comparison.

### 5. Test additional feature groups

**Work**

- Start with geometry and confidence, which come from the existing predictions. Add each group separately, then test combinations supported by validation.
- Score only the peptide with ProteinMPNN, HLA fixed, so the larger chain does not dominate the score. Do this once the core pipeline is complete.
- If FoldX access is already available, run `RepairPDB` then `AnalyseComplex` for peptide versus HLA. Use interaction energy, not total complex folding energy.
- Train sequence, ESM, and augmented models on **exactly the same structural training rows**, with identical validation and test rows. Separately report full-training-data sequence models on that same test set as practical comparators.
- Keep low-confidence but usable predictions. Predefine technical-failure handling, report coverage, and show a sequence-model fallback for failed structures.
- Use repeated poses only on a small diagnostic subset if budget remains.

**Deliverable:** ablation table showing the incremental predictive value and cost of each feature group.

**Why:** a feature can score well alone yet add nothing to the sequence baseline. pLDDT is confidence, not physical stability; seed disagreement is model uncertainty, not measured motion; inverse-folding likelihood is not a half-life; interaction energy is not the unbinding barrier. OpenMM minimisation energy is not a drop-in replacement for FoldX interface scoring.

### 6. Evaluate and prepare the submission

**Work**

- Evaluate validation-selected configurations **once** on the held-out test set.
- Report per-allele Spearman correlation for peptide ranking, with test counts and a median/IQR summary. Keep a common set of eligible alleles across models and report small or undefined cases explicitly.
- Report MAE on `log1p` half-life for numerical error. Add **precision@10 at a predeclared 2-hour threshold** to show how many top-ranked candidates are sufficiently stable. Treat pooled metrics as secondary, and distinguish within-allele ranking from between-allele effects.
- Use paired uncertainty estimates that keep peptide clusters together across alleles.
- Report accuracy gains alongside extraction/training cost, **cost per 1,000 new predictions**, runtime, and prediction failures.
- A confidence interval crossing zero is inconclusive. A strong negative result should exclude the predeclared worthwhile gain. Bound every conclusion to the tested representation, data, split, and budget.
- If time remains, add an allele-held-out evaluation, stratified by distance to training HLA sequences, before making any new-allele generalisation claim.
- The original assay panel was partly selected by predicted affinity, so broader biological or clinical claims need additional evidence.

**Deliverable:** reproducible code/configurations, final comparison table, limitations, and a concise presentation.

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

Claude, Devin, Antigravity, and AMASS credits support implementation and research; do not assume they fund folding GPUs. Recheck rates and credit availability before launch.

Reference worker rates assume standard Functions at base rates, four physical CPU cores, and 32 GiB host memory per GPU worker. They exclude startup, retries, extra poses, and storage/egress; region and non-preemptible options change pricing.

| GPU | GPU-only $/hour | Including CPU/memory $/hour |
|---|---:|---:|
| L40S, 48 GB | 1.95 | 2.40 |
| A100, 40 GB | 2.10 | 2.54 |
| A100, 80 GB | 2.50 | 2.94 |
| H100, 80 GB | 3.95 | 4.39 |

```text
cost = total billed worker seconds * combined GPU/CPU/memory rate
capacity = folding budget / measured cost per successful complex
elapsed time approximately = total worker hours / concurrent workers
```

**Any seconds-per-complex figure is a budget threshold, not measured throughput.** Choose hardware on measured dollars per successful prediction and the deadline. Parallel workers shorten elapsed time but do not inherently reduce total compute cost. Choose the lowest-cost GPU that meets the deadline and memory requirements.

## Team workstreams and checkpoints

| Participant | Primary responsibility | Handoff |
|---|---|---|
| 1 | Data audit, splits, sequence baselines, evaluation | Shared split IDs and prediction/evaluation format |
| 2 | ESM-2 embeddings, regression heads, representation comparisons | Cached features and validation-selected models |
| 3 | GPU pilot, structures, additional feature extraction | Structure/feature manifests keyed by complex ID, with success and failure records |

- **First 3 hours:** freeze data and evaluation, establish a baseline, begin ESM extraction, complete the small structural pilot.
- **By hour 5:** review validation results; make the structural go / reduce / stop decision. Run the auxiliary affinity probe if the core comparison is on track.
- **Hours 5-12:** finish the selected workload and ablations. Stop adding features at hour 10; freeze configurations by hour 12.
- **Final 4 hours:** evaluate on held-out data, compute uncertainty, prepare figures and the presentation, save deliverables.
- Adjust these cutoffs if the official deadline requires it.
- Save code, data and splits, features, checkpoints, structures, and results locally before the **Antigravity event resources are deleted after the event on Sunday, October 4**. Modal and Hugging Face credits are separate per-participant offers and are not affected by that deletion. Keep credentials out of shared manifests and exported artifacts.

## Stretch work and stopping rules

- The auxiliary affinity experiment (stage 3b) may interact with the ESM-2 comparison: if multi-task training helps the sequence arm more than the ESM arm, report that result rather than suppressing it.
- Defer until the core comparison is secure: SaProt, chimeric inputs, cross-attention, folding-trunk features, template threading, new geometry-aware GNNs, extensive interpretability probes, source-protein mapping, and molecular-dynamics unbinding calculations.
- Same-peptide / different-HLA diagnostics are possible on this data. **Matched C67S / wild-type comparisons are not** — the corresponding wild-type alleles are absent.
- Expand a feature pipeline only when validation evidence or useful uncertainty information supports the extra cost. Do not claim an unbinding mechanism from improved prediction alone.

**Minimum successful submission:** reliable splits, a trained sequence baseline, an ESM-2 comparison, uncertainty estimates, and measured costs. Structural experiments strengthen this result but must not prevent completion of the core study.

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
