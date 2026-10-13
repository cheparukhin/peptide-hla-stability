# Peptide–HLA stability: submission brief

Prepared 13 October 2026. Local draft covering completed development, ranking,
transfer, hybrid and label-budget studies, and the independent-data metadata
audit. The few-shot result is complete and independently audited;
ProteinMPNN is complete with an inconclusive context increment beyond Boltz;
repaired-descriptor assessment is also complete and inconclusive beyond R5.
Few-shot adaptation is the sole positive primary in the original family. One
selected full-cohort ProteinMPNN extension is complete and independently audited,
with no established added R5 benefit. Computational search stops here. The
[pre-efficacy amendment](../followup3/THIRD_PRIMARY_AMENDMENT.md) declares three
primaries and 98.333333% family intervals.
Not submitted externally.

## Project in one paragraph

We investigated whether position-aware Boltz interface embeddings help rank
peptide–HLA stability. On the examined original-training cohort, a nested
embedding procedure improves median within-allele Spearman correlation from
0.6668 for a selected sequence comparator to 0.7212: a paired difference of
+0.0543 with a 95% interval of [+0.0314, +0.0808]. The project includes leakage-
controlled grouped assessments, representation controls, execution repeats,
auditable predictions, and portable ridge and neural prediction heads. These
are development findings; independent experimental confirmation remains open.

The supplied kinetic-stability dataset is attributed to **Rasmussen et al.
(2016)**, *Pan-Specific Prediction of Peptide–MHC Class I Complex Stability,
a Correlate of T Cell Immunogenicity*, *The Journal of Immunology* 197(4),
1517–1524. [DOI: 10.4049/jimmunol.1600582](https://doi.org/10.4049/jimmunol.1600582)
([PubMed record](https://pubmed.ncbi.nlm.nih.gov/27402703/)). Later assessments
retain the supplied original-training partition and frozen grouping identities.

## What we built

- Frozen Boltz feature extraction and position-aware interface representations.
- Nested peptide-cluster, HLA-group and joint holdout assessments with fitting,
  stopping and selection kept inside the appropriate partitions.
- Sequence, pooling and shuffled-contact controls, plus a matched ranking-loss
  experiment covering 840 new fits and a 480-fit matched hybrid study.
- Portable, checksummed NPZ/JSON ridge and 30-member MLP heads with documented
  inputs and cached-feature inference.
- A source-backed project page, complete reports and an unsent assay brief.

## What the evidence supports

The main development gain is positive, but its lower confidence bound is below
the predeclared +0.05 worthwhile-gain threshold. Position-aware representations
outperform pooled controls; the designated contact map has no established
advantage over its shuffled counterpart. Adding mixed ranking loss to the
Boltz MLP yields +0.0071 [−0.0051, +0.0098], and the inner deployment selector
retains MSE in every R5 fold. This is an uncertain gain, not evidence of
no possible ranking benefit.

The stronger transfer extension is complete: 30 cells and 10,830 retained fits.
Matched R5−R0 MLP gains are +0.4062 [+0.3073,+0.4733] in HLA and
+0.3738 [+0.3092,+0.4638] in joint holdout (≥20 panel, crossed 95% intervals).
Matched ridge, ≥50-panel and single-HLA-group removal checks also stay positive.
Both original and expanded sequence references remain visible because the
expanded selector has a lower joint point estimate than the original. This supports
useful pretrained representations under matched learners, not causal attribution
to structure or independent confirmation.

The audited hybrid study compares an inner-selected tree/MLP/preprocessing
procedure with the matched two-member R5 standard MLP. Its primary difference
is **+0.004384 [−0.005541, +0.009984]**; the ≥50-row sensitivity is
**+0.002640 [−0.003962, +0.009132]**. The selector keeps the standard R5 MLP in
three folds and a 75% MLP / 25% regression-tree blend in two. Regression/ranking
trees, tree-leaf MLP and PCA MLP score worse in the frozen exploratory comparisons
on both representations; quantile preprocessing remains inconclusive.
Secondary intervals are nominal and not multiplicity-adjusted. The primary
result establishes neither an additional hybrid gain nor equivalence. These
fixed recipes do not replace the stronger competitive sequence comparator used
for the main +0.0543 result. See the [hybrid report](hybrids_cloud_v1/REPORT.md)
and [independent saved-table audit](hybrids_cloud_v1/coverage_audit.json).

The final **120-fit label-budget study** finds a larger matched Boltz advantage
with fewer labels: the R5−R0 gap increases by **+0.1615 [0.1333,0.2041]** at
25% versus 100% of fitting peptide clusters. Both procedures worsen; R5 loses
less performance. Stopping labels count and subset scores are averaged. This
fixed-recipe sensitivity does not estimate prospective assay savings or replace
the stronger main-study sequence comparison. [Complete results](label_efficiency/REPORT.md).

The completed [few-shot follow-up](../followup3/REPORT.md#completed-few-shot-result)
adds a distinct finding: 20 target measurements per allele raise R5 ridge
ranking from 0.529583 to 0.576337, a **+0.046754 [0.022657,0.103467]** gain
with a **98.333333% family interval [0.014042,0.119457]**. Three support draws
are averaged as scores; target HLA groups and support/query peptide families
stay outside source fitting. All 450 source fits and 9,180 adapters pass cloud
saved-state replay and a separate independent numerical/ancestry audit. The
≥20 and ≥50 panels are the same 68 alleles. Per draw, k=20 uses 1,360 target
labels with no target tuning/stopping; source stopping labels count. The
sequence comparator is a reduced historical four-member anchor. These are
conditional development findings with different source/query/label access
from the main study, not prospective assay savings or independent confirmation.

A subordinate [post-hoc offset diagnostic](../followup3/fewshot_offset/REPORT.md),
chosen after the positive few-shot primary, finds full adaptation adds
**+0.044350 [0.016478,0.092778]** beyond the saved scalar-offset control.
Offset-only minus zero-shot is **+0.002404 [−0.006172,0.023880]**. These are
nominal unadjusted 95% intervals: the feature term adds ranking beyond this
control on saved predictions, while offset-only benefit is unresolved. No new
fits or labels are used; it is exploratory and leaves the three-primary family
unchanged.

The [completed ProteinMPNN follow-up](../followup3/REPORT.md#completed-proteinmpnn-result)
has an inconclusive context increment beyond Boltz: **+0.023960
[−0.014736,0.058359]**, with a **98.333333% family interval
[−0.021942,0.065942]**. The net R5+P+C minus R5 gain is **+0.018816
[−0.016250,0.056637]** under its exploratory interval, from 0.600708 to
0.619524. All six arms and 900 fits on the balanced 4,449-pair panel pass
independent prediction/selection/ancestry and parser-recovery audits;
fitted-weight replay is not claimed. R0 controls are exploratory, and the
R5 ridge sensitivities remain inconclusive. This is not equivalence or a
replacement for the main assessment.

[Post-hoc descriptive influence checks](../followup3/inverse_execution_v2/exploratory_influence/summary.json)
keep the MLP's net point positive after removing any one allele or outer-held
prediction fold under fixed original eligibility. Ridge disagreement remains;
these checks add no confidence interval or inference upgrade and leave the
original MLP uncertainty inconclusive.

The [completed repaired-descriptor study](../followup3/REPORT.md#completed-repaired-descriptor-result)
finds MLP R5+E minus R5 **+0.014991 [−0.043512,0.036368]**, with a
**98.333333% family interval [−0.052436,0.045026]**. Scores are 0.600708
and 0.615698; ridge adds +0.006510 [−0.023760,0.020296]. Both learners
and the ≥50-panel sensitivities remain inconclusive. A positive exploratory
R0 ridge increment does not establish additional incumbent R5 value. All
4,449 descriptors, 200 new MLP members and 100 ridge fits pass independent
audits, preserving the 4,376 inherited successes and recovering 73 interrupted
attempts. R0/R5 baselines are reused exactly with no refits; fitted-weight replay
is not claimed. The [comparison figure](../followup3/figures/descriptor_comparison.svg)
shows every balanced-panel MLP arm and both descriptor primary family intervals.

The final [selected full-cohort ProteinMPNN extension](../followup3/REPORT.md#completed-selected-full-cohort-extension)
completes all 19,716 original-training pairs and 450 fits. Its MLP net increment
is **−0.007026 [−0.022047,+0.013082]**, from 0.695498 for R5 to 0.688472
for R5+P+C. Ridge net is −0.001644 [−0.015097,+0.007286]; context, P-only
and ≥50 sensitivities remain inconclusive. These are conditional pointwise
95% intervals in a study selected after balanced outcomes, not independent
confirmation or a fourth original-family primary. The fixed original-query
MLP net changes from +0.018816 to +0.001088 descriptively; source coverage,
selection and stopping change together. No harm or equivalence is established.
All arms and contrasts are retained in the source-backed report.

## Three-minute demonstration

1. Open the [project page](../../site/index.html). Show the main development
   comparison, then the positive few-shot adaptation result and its label
   budget. Show the balanced descriptor figure and the completed full-cohort
   result: no added ProteinMPNN benefit is established, so computational search
   stops and independent assays are next.
2. Show the [portable model contract](../../docs/ROUND2_INFERENCE.md) and inspect
   one saved head. Its files contain weights, scalers, configuration and hashes.
3. Use the documented cached-feature prediction command on the identified
   demonstration rows. State that these are training-cache examples, with no
   accuracy claim attached to the demonstration.
4. Show the evidence boundaries: the historical benchmark is separate, all
   follow-ups use the original training cohort, and independent half-life
   measurements are still needed.

The measured warm cached-head median is 1.56 ms for ridge and 2.40 ms for the
MLP per row on the recorded host. This excludes cache loading, MSA generation,
Boltz extraction and startup. Novel sequence-to-prediction latency is unmeasured.

## Questions we should answer plainly

**Is this independent validation?** No. The later studies reuse the original
training cohort for development. The historical test report is separate and
was not reopened for the follow-ups. Its [original report](../REPORT.md#2-data-and-evaluation)
discloses earlier diagnostic exposure of test rows and aggregate outcomes.
No adequate reserved independent assay cohort is available. A
[metadata/methods audit](../followup3/independent_data/PROVENANCE.md) preserves
129 candidates, but verifies materially different methods and unresolved
construct/measurement mappings. The largest single-study/allele stratum has
23 rows and remains gated; even a valid narrow assessment would not provide
representative multi-allele confirmation. External candidate outcomes remain reserved;
incidental public-result visibility during literature retrieval is disclosed.

**Does structure cause the improvement?** The controls support useful
position-aware learned features. They do not identify a causal structural
mechanism or establish an advantage for the designated contact map.

**Why not deploy the ranking-loss winner?** The pooled outer-held median per-allele Spearman
was slightly higher, but the uncertainty interval spans zero and the frozen
inner selector retained MSE in all R5 folds. We preserve that decision.

**What is next?** Stop computational search and seek independent kinetic-assay
confirmation. The frozen full-cohort extension does not establish additional
ProteinMPNN value; no new grid, combination or FoldX expansion is admitted.
Retain the simpler embedding head and freeze candidate procedures before opening
independent outcomes. The [unsent assay brief](INDEPENDENT_ASSAY_BRIEF.md)
keeps locked-model confirmation primary and describes a clearly optional
few-shot arm with separate calibration supports and unopened query labels.
It makes no acquisition, power or assay-savings claim or authorization. The
new tranche is closed at $67.30 conservatively, with $0 reserved against the
$139 ceiling; accounting includes failures and is not a provider invoice.

**What remains limited?** Conditional uncertainty excludes assay and selection
repetition. The historical extraction-wrapper bytes have a disclosed provenance
gap. Portable heads require compatible verified feature caches and output
uncalibrated log-scale predictions rather than validated physical half-lives.

## Method references

- ProteinMPNN: [Dauparas et al., 2022](https://doi.org/10.1126/science.add2187)
  and [official implementation](https://github.com/dauparas/ProteinMPNN);
  full-backbone vanilla `v_48_020.pt`, with zero inference coordinate noise.
- FoldX: [Delgado et al., 2019, FoldX 5.0](https://doi.org/10.1093/bioinformatics/btz184),
  [Guerois et al., 2002, original force field](https://doi.org/10.1016/S0022-2836(02)00442-4),
  and [Schymkowitz et al., 2005, complex-analysis methods](https://doi.org/10.1093/nar/gki387).
  The recorded Linux binary is hash-pinned; repaired E contains 29 energy and
  interface/clash/residue descriptors.
- Boltz-2: [Passaro et al., 2025, technical preprint](https://doi.org/10.1101/2025.06.14.659707)
  and [official implementation](https://github.com/jwohlwend/boltz). Reused
  features use inference package `boltz==2.1.1`, with a separately pinned
  checkpoint revision and checksum.

The [follow-up report](../followup3/REPORT.md#primary-methods-and-pinned-artifacts)
gives exact artifact provenance. These upstream methods do not establish that
sequence likelihood or equilibrium energy predicts kinetic half-life. The
[pre-results implementation audit](../followup3/ASSESSMENT_PRECHECK.md) passed;
all three balanced/few-shot studies now have completed independent audits.
The selected full-cohort extension also passed independent saved-output auditing.

## Evidence and delivery

Use the [consolidated readout](FINAL_REPORT.md) for exact numbers, protocol links
and limitations; the [README](../../README.md) for navigation; and the
[independent assay brief](INDEPENDENT_ASSAY_BRIEF.md) for the unsent measurement
request. Verify the project-page snapshot with `python3 site/round2-build.py --check`.
No public deployment, external submission, assay purchase or vendor outreach
has been performed by this finalisation work.
