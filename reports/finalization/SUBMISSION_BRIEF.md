# Peptide–HLA stability: submission brief

Prepared 13 October 2026. Local draft covering completed development, ranking,
transfer and hybrid studies. Not submitted externally.

## Project in one paragraph

We investigated whether position-aware Boltz interface embeddings help rank
peptide–HLA stability. On the examined original-training cohort, a nested
embedding procedure improves median within-allele Spearman correlation from
0.6668 for a selected sequence comparator to 0.7212: a paired difference of
+0.0543 with a 95% interval of [+0.0314, +0.0808]. The project includes leakage-
controlled grouped assessments, representation controls, execution repeats,
auditable predictions, and portable ridge and neural prediction heads. These
are development findings; independent experimental confirmation remains open.

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

## Three-minute demonstration

1. Open the [project page](../../site/index.html). Show the main development
   comparison, then the uncertain ranking and hybrid gains with their frozen
   inner-selection fallbacks.
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
No adequate reserved independent assay cohort is available.

**Does structure cause the improvement?** The controls support useful
position-aware learned features. They do not identify a causal structural
mechanism or establish an advantage for the designated contact map.

**Why not deploy the ranking-loss winner?** The pooled outer-held median per-allele Spearman
was slightly higher, but the uncertainty interval spans zero and the frozen
inner selector retained MSE in all R5 folds. We preserve that decision.

**What is next?** Freeze candidate procedures before opening independent assay
outcomes. A separately frozen [120-fit label-budget study](label_efficiency/SPECIFICATION.md)
is running on Modal and has no outcomes yet.

**What remains limited?** Conditional uncertainty excludes assay and selection
repetition. The historical extraction-wrapper bytes have a disclosed provenance
gap. Portable heads require compatible verified feature caches and output
uncalibrated log-scale predictions rather than validated physical half-lives.

## Evidence and delivery

Use the [consolidated readout](FINAL_REPORT.md) for exact numbers, protocol links
and limitations; the [README](../../README.md) for navigation; and the
[independent assay brief](INDEPENDENT_ASSAY_BRIEF.md) for the unsent measurement
request. Verify the project-page snapshot with `python3 site/round2-build.py --check`.
No public deployment, external submission, assay purchase or vendor outreach
has been performed by this finalisation work.
