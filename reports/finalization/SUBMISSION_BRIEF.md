# Peptide–HLA stability: submission brief

Prepared 12 October 2026. Local draft for the completed development findings;
the stronger transfer extension and hybrid comparisons remain unfinished.
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

## What we built

- Frozen Boltz feature extraction and position-aware interface representations.
- Nested peptide-cluster, HLA-group and joint holdout assessments with fitting,
  stopping and selection kept inside the appropriate partitions.
- Sequence, pooling and shuffled-contact controls, plus a matched ranking-loss
  experiment covering 840 new fits.
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

The completed HLA/joint studies use narrower sequence comparators. Their
large differences must not be presented as wins over the full sequence family.
The frozen stronger-comparator extension is pending and must be incorporated
only after all cells and reporting pass their audits.

## Three-minute demonstration

1. Open the [project page](../../site/index.html). Show the main development
   comparison, then the uncertain ranking-loss result and its MSE fallback.
2. Show the [portable model contract](../../docs/ROUND2_INFERENCE.md) and inspect
   one saved head. Its files contain weights, scalers, configuration and hashes.
3. Use the documented cached-feature prediction command on the identified
   demonstration rows. State that these are training-cache examples, with no
   accuracy claim attached to the demonstration.
4. Show the evidence boundaries: original test preserved, stronger transfer
   controls pending, and an independent assay cohort still needed.

The measured warm cached-head median is 1.56 ms for ridge and 2.40 ms for the
MLP per row on the recorded host. This excludes cache loading, MSA generation,
Boltz extraction and startup. Novel sequence-to-prediction latency is unmeasured.

## Questions we should answer plainly

**Is this independent validation?** No. The later studies reuse the original
training cohort for development. The historical test report is separate and
was not reopened. No adequate reserved independent assay cohort is available.

**Does structure cause the improvement?** The controls support useful
position-aware learned features. They do not identify a causal structural
mechanism or establish an advantage for the designated contact map.

**Why not deploy the ranking-loss winner?** The average outer development score
was slightly higher, but the uncertainty interval spans zero and the frozen
inner selector retained MSE in all R5 folds. We preserve that decision.

**What is next?** Finish the existing transfer extension, then freeze candidates
before independent assay outcomes. The user has authorized bounded Boltz tree ranking, tree/MLP blends and
preprocessing comparisons; these remain unexecuted and have no results yet.

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
