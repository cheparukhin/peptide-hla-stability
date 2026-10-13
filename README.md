# Predicting peptide–HLA stability

**logic binders · London AI × Science · Protein Engineering Track**

Can pretrained protein representations improve predictions of how long a
9-residue peptide remains bound to an HLA molecule? Our completed development
study finds that **Boltz interface embeddings improve ranking over a selected
sequence baseline**, while simple geometry features did not help in the earlier
hackathon test. These are different representations and different assessments.

The current development gain is **+0.054 median per-allele Spearman**
(95% paired interval **+0.031 to +0.081**). This supports improvement on the
examined development cohort; it does not establish that the gain exceeds the
predeclared worthwhile threshold of +0.05. Independent assay confirmation remains
open. The original test has not been reopened for these follow-ups; its
[historical report](reports/REPORT.md#2-data-and-evaluation) discloses earlier
diagnostic exposure of test rows and aggregate outcomes.

## Explore the project

- [Current research report](reports/further_research/round2/REPORT.md): complete
  main, transfer, complementarity, reproducibility and resource findings.
- [Combined finalisation readout](reports/finalization/FINAL_REPORT.md): completed
  portable inference, ranking, stronger transfer controls, hybrids and label-budget
  sensitivity.
- [Matched ranking-loss report](reports/finalization/ranking/REPORT.md): frozen
  outcome, source tables and the completed artifact/arithmetic audit.
- [Label-budget report](reports/finalization/label_efficiency/REPORT.md): fixed-head
  response to fewer training labels, with shared subsets and uncertainty.
- [Hybrid report](reports/finalization/hybrids_cloud_v1/REPORT.md): matched tree,
  blending and preprocessing comparisons, with saved-model replay and an
  [independent table audit](reports/finalization/hybrids_cloud_v1/coverage_audit.json).
- [Project page](site/index.html): locally updated evidence and a clearly dated
  archive of the original film/test. Serve with `python3 -m http.server 8000 --bind 127.0.0.1` and open `http://localhost:8000/site/`. This branch has not been
  deployed to the public site.
- [Portable model guide](docs/ROUND2_INFERENCE.md): use the trained ridge and
  neural heads with verified, compatible cached Boltz features.
- [Canonical research plan](reports/NEXT_RESEARCH_PLAN.md) and
  [finalisation status](reports/finalization/execution_manifest.json): frozen
  decisions, remaining work and spending controls.
- [Third follow-up protocol](reports/followup3/PROTOCOL.md): frozen ProteinMPNN
  context-complementarity and few-shot HLA adaptation studies. A
  [pre-efficacy amendment](reports/followup3/THIRD_PRIMARY_AMENDMENT.md) adds
  conditional repaired-energy complementarity, with three-primary family control.
  The [few-shot result](reports/followup3/REPORT.md#completed-few-shot-result)
  is complete and audited. ProteinMPNN context augmentation is audited and
  inconclusive beyond the matched Boltz head, as is repaired-descriptor
  complementarity. The selected full-cohort ProteinMPNN extension is complete
  and independently audited; computational search stops here.
- [Independent metadata/methods audit](reports/followup3/independent_data/PROVENANCE.md):
  preserved 129 candidates, heterogeneous protocols and unresolved construct/
  measurement mapping; external candidate outcomes remain reserved.
- [Independent assay brief](reports/finalization/INDEPENDENT_ASSAY_BRIEF.md):
  concrete, unsent acquisition requirements and remaining confirmation gaps.
- [Original 4 October report](reports/REPORT.md): preserved historical study,
  including its recorded test assessment and earlier exposure disclosure.

## Completed development results

All rows below use only the **original training partition**: 19,716 pairs,
3,587 peptide clusters and 75 alleles. The main eligible panel contains 69
alleles with at least 20 measurements. Scores are medians of per-allele
Spearman correlations; differences are between those medians.

| Assessment | Embedding procedure | Sequence comparator | Difference [95% interval] |
|---|---:|---:|---:|
| Main peptide-cluster assessment | 0.7212 | 0.6668 | +0.0543 [+0.0314, +0.0808] |

The main comparator selects among five sequence procedures inside fitting
partitions. Its interval uses paired peptide-cluster resampling. The separate
HLA-group and joint holdouts below assess transfer, with scaling, stopping and
selection respecting their corresponding groups. Their results cannot be
subtracted from the main score to identify a causal representation effect.

The stronger transfer study covers all 30 cells and 10,830 retained fits.
With matched MLP recipes, R5 minus R0 is **+0.4062 [+0.3073,+0.4733]** in HLA
holdout and **+0.3738 [+0.3092,+0.4638]** in joint holdout (≥20 panel, crossed
95% intervals). Matched ridge comparisons, the ≥50 panel and all single-HLA-group
deletions also remain positive. Both original and expanded sequence references
are shown: the expanded selector has a lower joint point estimate than the original,
so it must not be the only reference. These remain conditional development
results and do not isolate a causal structural effect.
[Completed transfer report](reports/finalization/transfer_cloud_recovery/results/REPORT.md).

The **matched ranking-loss follow-up** adds MSE + RankNet to the same R5 MLP:
**+0.0071 [−0.0051, +0.0098]** against MSE alone. The ≥50-row sensitivity and
loss-by-representation interaction also span zero; the inner deployment selector
retains MSE in all five R5 folds. All 840 new fits, 480 reused MSE members and
both zero-weight validation members passed the
[completion audit](reports/finalization/ranking/completion_audit.json).

The **completed hybrid follow-up** asks whether trees, blending or preprocessing
improve a matched two-member R5 standard MLP. Its primary hybrid-selector gain
is **+0.004384 [−0.005541, +0.009984]**; the ≥50-row sensitivity is
**+0.002640 [−0.003962, +0.009132]**. The inner selector retains the standard
R5 MLP in three folds and selects a 75% MLP / 25% regression-tree blend in two.
The frozen exploratory comparisons show lower scores for regression/ranking
trees, tree-leaf MLP and PCA MLP on both representations; quantile preprocessing
is inconclusive. These nominal secondary intervals are not multiplicity-adjusted.
The primary result establishes neither a hybrid gain nor equivalence. All 480
scientific fits and the separate 12-fit unscored pilot passed saved-model replay;
the [table audit](reports/finalization/hybrids_cloud_v1/coverage_audit.json)
verified all 16 prediction arms and 50 comparison rows. These matched recipes
are distinct from the competitive main-study sequence selector.

The [audited few-shot HLA adaptation study](reports/followup3/REPORT.md#completed-few-shot-result)
raises R5 ridge from **0.529583 to 0.576337** with 20 target labels per allele:
**+0.046754 [0.022657,0.103467]**, with the **98.333333% three-primary family
interval [0.014042,0.119457]**. Three fixed support draws are averaged as
scores, with 1,360 target measurements per draw across 68 alleles and no target
tuning/stopping. Source stopping labels count, and target HLA groups plus
support/query families are excluded from source fitting. The ≥20 and ≥50
panels are identical. All 450 source fits and 9,180 adapters pass cloud exact
saved-state replay and independent numerical/ancestry checks. Its reduced
sequence anchor and strict joint exclusions differ from the main assessment;
no prospective assay savings or independent confirmation is established.

The [audited ProteinMPNN study](reports/followup3/REPORT.md#completed-proteinmpnn-result)
on the balanced 4,449-pair panel finds an inconclusive MLP context increment:
**R5+P+C minus R5+P = +0.023960 [−0.014736,0.058359]**; its required
**98.333333% family interval is [−0.021942,0.065942]**. The net increment
over R5 is **+0.018816 [−0.016250,0.056637]**, from 0.600708 to 0.619524.
All six arms, 600 MLP members and 300 ridge fits passed arithmetic/ancestry
and parser-recovery audits; fitted-weight replay is not claimed. R0 controls
are positive exploratory comparisons, while R5 ridge increments remain
inconclusive. This does not establish equivalence or replace the main result.

The [audited repaired-descriptor study](reports/followup3/REPORT.md#completed-repaired-descriptor-result)
finds **R5+E minus R5 = +0.014991 [−0.043512,0.036368]**, with
**98.333333% family interval [−0.052436,0.045026]**. Ridge and ≥50-panel
sensitivities are also inconclusive. All 4,449 descriptors and 300 new fits
pass independent recovery, matrix and assessment audits; R0/R5 baselines are
reused exactly. The [comparison figure](reports/followup3/figures/descriptor_comparison.svg)
shows all eight balanced-panel MLP arms and both descriptor family intervals.
Few-shot adaptation is the only positive primary in the original family.

The final [selected full-cohort ProteinMPNN extension](reports/followup3/REPORT.md#completed-selected-full-cohort-extension)
is complete on all **19,716 original-training pairs**, with 300 MLP members
and 150 ridge fits. MLP R5+P+C minus R5 is **−0.007026 [−0.022047,+0.013082]**,
from 0.695498 to 0.688472; ridge, context, P-only and ≥50 sensitivities remain
inconclusive. These are conditional pointwise 95% intervals after evidence-based
study selection, not independent confirmation or a fourth original-family
primary. All arms and fixed-query descriptive results remain visible.
No added ProteinMPNN or FoldX benefit beyond R5 under the tested recipes is
established; neither harm nor equivalence is established.
**Stop computational search; retain the simpler embedding head and seek
independent kinetic-assay confirmation.** The [unsent assay brief](reports/finalization/INDEPENDENT_ASSAY_BRIEF.md)
keeps locked-model confirmation primary with an optional future few-shot arm.
No new grid, combined arm, full FoldX expansion, acquisition or outreach is admitted.

Intervals throughout condition on fitted predictions and inner choices;
they omit repeated fitting/selection and independent assay variation.

Position-aware interface representations substantially outperform global
pooling in matched neural comparisons. The designated contact-map representation
has no established advantage over its shuffled-contact control. These controls
support useful positional information, not a demonstrated causal structural
mechanism. Exact execution repeats matched on the recorded platform; they do
not measure assay reproducibility or independent model-selection reproducibility.

## Use the trained heads

Create the documented [Python environment](docs/ROUND2_INFERENCE.md#environment-setup)
before running the commands below.

Two locked demonstration recipes are included under `models/finalization/`:
R4 ridge with alpha 100 and a 30-member R5 MLP with L2 0.01. They were refit on
original training rows only and are not designated universal winners. Portable
NPZ/JSON artifacts contain weights, scalers and provenance; executable pickle
objects are not used.

```sh
.venv/bin/python scripts/round2_inference.py inspect \
  --model models/finalization/R5_mlp_0.01
```

Prediction requires a compatible verified compact feature bundle; large caches
are external to Git. Follow the [input contract and prediction example](docs/ROUND2_INFERENCE.md).
This is cached-feature inference, not a sequence-to-prediction service.
Warm single-row cached-head medians were 1.56 ms for ridge and 2.40 ms for the
MLP on the recorded host. These measurements exclude Boltz extraction, MSA,
cache loading and startup, and are not end-to-end new-sequence costs.

## Reproduce and inspect

The environment and experiment-specific commands are documented in
[the completed study](reports/further_research/round2/REPORT.md),
[operations](reports/further_research/round2/OPERATIONS.md), and individual stage
reports. Source, cohort, feature and completion hashes bind the saved evidence.
Completed expensive scopes must not be blindly relaunched.

```sh
# Check the portable head implementation without cloud work.
.venv/bin/python -m pytest -q tests/test_round2_inference.py

# Verify that the project page still matches its audited sources.
python3 site/round2-build.py --check
```

The inherited extraction-wrapper hash discrepancy is disclosed in the report:
scientific source and feature audits passed, but exact launch-wrapper bytes
cannot be certified. No new result repairs that historical provenance gap.

## Completed follow-up and remaining work

Portable inference, ranking loss, stronger transfer controls and the hybrid
study are complete and audited. Their uncertain ranking/hybrid gains do not
authorize changing the completed candidate grids or stopping rules.
The [120-fit label-budget study](reports/finalization/label_efficiency/REPORT.md)
also passed. At 25% versus 100% of outer-fitting peptide clusters, the matched
Boltz−sequence MLP gap grows by **+0.1615 [0.1333,0.2041]**. Both models worsen,
but Boltz loses less performance. The fixed recipes, label-blind subsets and
counted stopping labels support relative robustness; they do not establish
prospective assay savings or a gain against the main competitive selector.
The finalization computational work and its temporary cloud resources are
closed. The separately authorized third follow-up tranche is closed at
**$67.302731 conservatively, with $0 reserved**, against its $139 inclusive
ceiling. Failed attempts are included; these are not reconciled provider invoices
or a live credit balance. [Ledger](reports/followup3/budget.json).


The completed round-two phase conservatively accounted for $76.84 of its $100
ceiling, including a retained engineering allowance. Its cloud jobs are stopped.
Finalisation has a separately authorized additional $100 ceiling and reuses
existing embeddings. [Current accounting](reports/finalization/FINAL_REPORT.md)
separates completed costs and reservations; these are conservative operational
figures, not provider invoices or a live balance.

## Scope

Stability is a kinetic endpoint; affinity, thermal melting and structural
plausibility cannot silently substitute for dissociation half-life. The public
candidate cohort does not yet provide representative independent confirmation,
and available aggregate labels cannot establish an assay-noise ceiling.
This project provides development evidence and reproducible research artifacts,
not a validated clinical or vaccine-design system.
