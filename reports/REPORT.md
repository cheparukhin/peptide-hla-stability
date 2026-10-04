# Does pretrained protein AI improve peptide–HLA stability prediction?

**London AI × Science — Protein Engineering Track, 3–4 October 2026**

> **Main result.** A supervised sequence ensemble leads the six-arm test
> comparison: median per-allele Spearman **0.693 on validation** and **0.706 on test**.
> Frozen ESM-2 embeddings do not establish an improvement, and their test
> intervals exclude the predeclared **+0.05** worthwhile gain. The implemented
> Boltz-2 structural arm reduces test ranking accuracy by
> **−0.112 [−0.134, −0.057]**, after a full-cohort fold costing **$212.71,
> derived from recorded container-hours**.

**Evidence status.** The six-arm test summary and paired intervals are available
and reported in §7; the run's completion manifest is still outstanding in the
reviewed artifacts. Validation comparisons remain separate in §3. ProteinMPNN
and FoldX are diagnostic pilots, with narrower conclusions than the predictive
benchmark.

## 1. The problem

HLA molecules hold short protein fragments at the cell surface for immune
inspection. We predict how long a **9-residue peptide** remains bound to its
HLA molecule, measured as its **dissociation half-life**.

Binding affinity describes equilibrium binding strength; dissociation half-life
describes how quickly a bound complex comes apart. A plausible predicted pose
or favourable interaction energy therefore needs to be tested against half-life
labels before it can be used as a stability predictor.

**Research question.** Do pretrained protein-model features improve half-life
prediction over a supervised sequence model, and is any improvement worth the
additional compute?

## 2. Data and evaluation

| Property | Value |
|---|---:|
| Peptide–HLA pairs | 28,166 |
| Unique peptides | 5,633 |
| HLA alleles | 75 |
| Peptide length | 9 residues |
| Labels recorded at 0 h | 20.2% — treated as an assay floor |
| Train / validation / test rows | 19,716 / 2,817 / 5,633 |

**Splits.** Frozen 70/10/20 partitions group peptides by sequence similarity:
peptides within three substitutions cannot appear in different splits. This
reduces inflation from training on near-copies of evaluation peptides.

**Primary metric.** Median per-allele Spearman measures peptide ranking within
each allele. Validation covers **68 eligible alleles / 2,802 scored rows**;
test covers **67 / 5,565**. Eligibility is fixed from row counts and labels,
independently of predictions. Secondary metrics are MAE on `log1p(half-life
in hours)` and precision@10 for half-life greater than 2 h.

**Decision rule.** The frozen [evaluation contract](../EVALUATION.md) sets a
minimum worthwhile gain of **Δ median per-allele Spearman = +0.05**, motivated
by the benchmark's estimated uncertainty. This is a decision threshold rather
than a universal detection limit. Paired 95% intervals resample whole peptide
clusters, using 2,000 draws. An interval crossing zero leaves the direction
unresolved; an upper bound below +0.05 excludes that gain under this procedure.
An improvement clears the bar only if its lower bound exceeds +0.05.

**Comparison controls.** The headline sequence, ESM and structural arms each
use 30-network ensembles on the same validation rows. Arm-specific tuning
ranges and sequence controls are recorded in the stage reports. The single
network and feature-only ablations below provide context. NetMHCstabpan is
excluded as a held-out comparator because its training data overlap this
benchmark, including test peptides.

**Test exposure disclosure.** An earlier diagnostic re-partitioned the full
dataset and exposed frozen test rows to diagnostic fitting and aggregate
evaluation. Its conclusion was retracted; the final benchmark models were
fitted on the frozen training split. The exposure partly influenced the project
narrative and motivation for ensembling, so this study does not claim a wholly
untouched test set. The diagnostic used **3,350 frozen test rows for fitting,
448 for early stopping and 585 in its evaluation**, exposing eight aggregate
statistics. Observed aggregates and the fix are recorded in
[the evaluation contract](../EVALUATION.md#disclosed-test-exposure).

## 3. Main result: validation comparisons

Higher Spearman is better. All differences are against the **0.6931 sequence
ensemble**, and every interval in this table is **validation**, not test.

| Approach | Median Spearman | Δ vs. baseline [paired 95% CI] | Interpretation |
|---|---:|---|---|
| Allele mean | 0.000 | — | Sanity floor |
| Single sequence MLP: peptide + 34 HLA contact residues | 0.610 | — | Single-network reference |
| **Sequence ensemble, 30 networks** | **0.693** | — | **Reference; highest point estimate** |
| ESM-2 35M embeddings | 0.683 | −0.010 [−0.038, +0.035] | Direction unresolved; excludes +0.05 |
| ESM-2 150M embeddings | 0.674 | −0.019 [−0.060, +0.020] | Direction unresolved; excludes +0.05 |
| Sequence + ESM-2 embeddings | 0.676 | −0.017 [−0.047, +0.032] | Direction unresolved; excludes +0.05 |
| Boltz-2 geometry only | 0.268 | −0.425 [−0.499, −0.352] | Worse |
| Boltz-2 confidence only | 0.393 | −0.301 [−0.369, −0.227] | Worse |
| Sequence + Boltz-2 geometry + confidence | 0.622 | **−0.071 [−0.123, −0.025]** | **Worse** |

Sources: [ESM comparisons](stage3_comparisons.csv),
[structural scores](stage5_structural_val.csv) and
[structural paired intervals](stage5_structural_bootstrap.csv).

**Interpretation.** ESM's results allow small gains or losses; they do not
establish equivalence to the baseline or a zero effect. The tested structural
arm performs worse on validation. Its sequence encoding mix also differs from
the reference ensemble, so the headline delta measures the implemented arm
rather than isolating the effect of structural features. A sequence-only
control in the structural pipeline has lower development MSE at **five of six**
tested L2 settings, including the structural arm's selected setting: **0.5458
vs. 0.5503 at L2 0.01**. At L2 0.1 the ordering reverses, **0.8176 vs. 0.7082**.
That supports a detrimental feature addition within this implementation, but is not
a paired validation comparison on the primary ranking metric. See
[the structural report](stage4c5_features.md#93-the-control-that-makes-the-result-attributable).

## 4. Additional experiments and diagnostic pilots

These experiments use validation or their own separately stated protocols.
They are not additional held-out test comparisons.

| Experiment | Supported outcome |
|---|---|
| [Weak-binder augmentation](stage2b_augmentation.md) | Best Δ +0.024; all paired intervals cross zero and exclude +0.05 under the tested sources and weights. |
| [Auxiliary affinity labels](stage2c_affinity.md) | No demonstrated worthwhile increment. Affinity alone ranks stability at ρ 0.580 on the dual-labelled subset; this is not an upper bound on an energy model or its incremental value. |
| [ESM-2 + affinity](stage3b_esm_multitask.md) | **Unresolved:** sensitivity at the +0.05 bar was not established for the difference-of-differences analysis. |
| [Censored (Tobit) loss](stage7_censored.md) | Better floor calibration, worse ranking: Δ −0.0414 [−0.078, −0.006]. |
| [ProteinMPNN inverse folding](stage5_inverse_folding.md) | A five-complex pilot showed no half-life association and separated one known pose failure. A subsequent **[2,000-fold QC sample](stage5_inverse_folding_qcsample_a-cheparukhin.csv)** showed that the pilot's absolute score thresholds did **not transfer** ([run record](stage5_inverse_folding_qcsample_a-cheparukhin.provenance.json)). No full-cohort regression comparison was performed. |
| [FoldX empirical energy](stage8_foldx.md#results) | Pilot repair changed implausible raw energies. **Neither standalone FoldX nor sequence + FoldX has been scored for half-life prediction**; incremental predictive value remains open. |

The geometry/confidence benchmark supports a negative result for the tested
structural feature block. ProteinMPNN and FoldX supply engineering diagnostics;
they do not provide two further independent predictive nulls.

## 5. Transfer and unseen-allele analysis

**Elution transfer.** Without retraining on elution labels, the sequence model
discriminates **81,600 naturally presented peptides** from **816,000 matched
proteome decoys** across **51 alleles**, with median **AUROC 0.9656**. Random
scores return 0.4974. More informative controls preserve AUROC **0.9157** when
decoys are ligands from other alleles and reduce it to **0.6965** when the model
receives the wrong HLA pseudosequence.

These controls support transfer of allele-dependent peptide recognition to a
different assay. Elution is not a half-life measurement, and proteome decoys are
assumed negatives. This is evidence of biological transfer, rather than a
second measurement of stability accuracy or proof that dataset artefacts are
absent. Protocol and caveats: [elution report](stage3c_elution_validation.md).

**Unseen alleles.** A separate leave-allele-out analysis gives median ρ **0.741**
for near alleles and **0.339** for distant alleles: near − distant **+0.403
[+0.223, +0.478]**. Panel composition partly confounds the distance association.
These models are refitted under a different protocol, so these values are not
comparable to §3 or §7. See [the allele hold-out report](stage7_allele_holdout.md).

## 6. Compute cost

| Approach | Project cloud spend | Compute per 1,000 new pairs | Cost basis |
|---|---:|---|---|
| Sequence ensemble | $0; local CPU | **~0.073 s**, including feature assembly | Measured local timing |
| ESM-2 35M ensemble | $0; local GPU/CPU | **~1.14 s**; illustrative ~$0.00047 | Local timing priced at a rented A10G rate; not an incurred bill |
| Boltz-2 production folds | **$212.71 derived** | **~$7.55**, fold only at realised cohort rate | 28,166 folds, 0 execution failures, 143.6 A10G-hours; 7 h 32 min wall |
| Structural feature extraction | ~$1.18 derived | Reported separately from folding | Full cohort, 0 extraction failures |
| ProteinMPNN QC sample | ~$1.15 derived, including a discarded attempt | Diagnostic sample; no deployed predictor | 2,000 successful sampled scores |
| FoldX pilots | ~$0.33 derived | Predictive production pass not run | Pilot timing and rates |

Production folding dominates project compute. Its **$212.71** figure is
recorded container-hours multiplied by the measured worker rate; a final
provider-bill reconciliation was not captured. The ledger also records **$4.41
metered pre-production spend**. Costs here are useful comparisons of the tested
implementations, rather than deployment quotes. Zero execution failures do not
establish pose accuracy across the cohort. The
[production verification](ectodomain-20261004/production_verification.json)
records folding totals, and the
[QC run record](stage5_inverse_folding_qcsample_a-cheparukhin.provenance.json)
records the successful sample and discarded attempt. See
[the compute ledger](compute_ledger.md) for the broader accounting.

## 7. Test results

The [test provenance record](test_scoring_provenance.json), captured
**4 October 2026 at 13:35:40 BST**, identifies **six arms**, each with predictions
for all **5,633 test rows**. The written summary scores **5,565 rows across 67
eligible alleles**. The estimates below come directly from
[the existing test summary](stage6_test_summary.csv); no test scoring was rerun
for this report.

| Approach | Test median Spearman | Δ vs. sequence ensemble [paired 95% CI] | Test MAE, log1p |
|---|---:|---|---:|
| **Sequence ensemble: peptide + contact residues** | **0.7064** | — | **0.4657** |
| Single sequence MLP | 0.6181 | −0.088 [−0.120, −0.055] | 0.5322 |
| Sequence ensemble: full HLA domain | 0.6904 | −0.016 [−0.039, +0.004] | 0.4804 |
| ESM-2 35M ensemble | 0.6979 | −0.009 [−0.029, +0.023] | 0.4735 |
| Sequence + ESM-2 ensemble | 0.6816 | −0.025 [−0.046, +0.007] | 0.4828 |
| Sequence + Boltz-2 geometry + confidence | 0.5947 | **−0.112 [−0.134, −0.057]** | 0.5174 |

![Paired test comparisons against the sequence ensemble](figures/report_test_comparisons.png)

The [test paired intervals](stage6_test_paired_ci.csv) use 2,000 whole-cluster
resamples for the primary metric. Both ESM comparisons cross zero and exclude
the +0.05 gain. The implemented structural arm is worse, with its interval
entirely below zero. These test verdicts agree with validation. Median
precision@10 is 0.9 for the sequence and ESM arms and 0.8 for the structural
arm; the paired precision intervals cross or touch zero, so that secondary
metric does not establish a difference.

**The completion manifest remains outstanding in the reviewed artifacts**;
the results above are taken from the written summary and paired-output files.
The final benchmark procedure is specified in the
[test-scoring runbook](TEST_SCORING_RUNBOOK.md), with the earlier exposure
disclosed in §2.

## 8. Limitations

- Conclusions cover the tested frozen ESM-2 representations and Boltz-2
  single-pose feature extraction. Fine-tuning, other model families and other
  structural representations remain open.
- ProteinMPNN's pose-failure evidence rests on one failing complex; its
  absolute thresholds failed to transfer to the QC cohort. FoldX predictive
  performance remains unmeasured.
- ESM-2 + affinity is unresolved at the worthwhile-gain threshold. Missing
  or underpowered evidence is not treated as a predictive negative.
- The assay floor creates tied labels, and no assay replicates are available
  to estimate a noise ceiling. The panel was partly selected by predicted
  affinity, limiting broader biological generalisation.
- Earlier test exposure, validation model selection, pending completion audit
  and incomplete provider-bill reconciliation limit the strength of the claims.

Full register: [limitations.md](limitations.md).

## 9. Conclusion

Within the six-arm benchmark, the supervised sequence ensemble leads the test
point estimates. Frozen ESM-2 embeddings do not establish a worthwhile gain,
while the implemented Boltz-2 arm performs worse despite substantially greater
compute. Paired validation and test intervals support those conclusions.

The submission's contribution is a controlled comparison of predictive value
and compute, supported by bounded validation negatives and a separate
allele-specific transfer result. Inverse folding and empirical energy pilots
identify useful implementation limits while leaving their broader predictive
value open.

## Submission checks

This report collects the submission results, interpretation and outstanding
checks. Linked stage reports and scored tables provide the supporting evidence.

- **Completion record:** archive the original pass's
  `stage6_test_manifest.json` when available and check it against the prediction
  hashes and scored outputs. The paired intervals are already included here.
  Do not relaunch test scoring solely to finish the prose.
- **Frozen configurations:** the current provenance records the dirty-tree
  flag, hashes and arm list, but omits the per-arm frozen configurations and
  stage-report references requested by the runbook. Preserve the original
  capture; date any supplemental record and use pre-score evidence for freezes.
- **Companion documents:** reconcile [SUBMISSION.md](SUBMISSION.md),
  [README.md](../README.md) and [the compute ledger](compute_ledger.md) with the
  results here. They still describe an unscored test set, a missing structural
  result or an unrun ProteinMPNN QC sample. Their historical figures also need
  updating: the old distribution plot omits the structural arm and the cost
  plot shows its accuracy as pending. The figure in §7 uses the published test
  tables directly; its generator is
  [make_report_comparison.py](figures/make_report_comparison.py).
- **Structural source correction:** change the source report's claim that
  sequence-only wins at all six L2 settings to five of six, and identify the
  quantity as development MSE. The exact values and limits are in §3 above.

Review verification: the reported primary scores and intervals, all six test
prediction hashes, both pinned input hashes and document links were checked
against saved artifacts. No models were fitted and no test scores were
recomputed during the review.
