# Independent validation of peptide–HLA stability prediction

## Objective

This proposed prospective design outlines how one locked Boltz-based predictor
and one locked sequence reference could be compared on new kinetic measurements;
it is not an externally required or finalized study. The primary question is
whether the ranking advantage reported in the [manuscript](../MANUSCRIPT.md)
persists in an independent assay population. Neither model may be selected or
calibrated using confirmation outcomes.

## Population and measurements

Choose peptide families absent from the development corpus and sample several
HLA sequence groups. Define the intended application population before choosing
peptides, because a panel enriched for predicted binders answers a different
question from an unselected peptide panel. Record inclusion probabilities if
stratified sampling will be used to estimate population performance, and keep
any exploratory subset enriched for model disagreement separate from the primary
panel.

Measure dissociation at 37°C with a consistent protocol, documenting whether the
readout follows peptide release directly or uses β2-microglobulin dissociation
as a proxy. Report exact peptide sequences, modifications, and purity; physical
HLA and β2-microglobulin constructs; and plate layout, reagent lots, and assay
batches. Blind laboratory identifiers to predictions and balance allele and
peptide-family strata across batches.

Retain raw time courses, fitting residuals, measurement limits, replicate-level
estimates, and QC flags. Distinguish insufficient complex formation, measurement
failure, rapid unresolved dissociation, and slow unresolved dissociation rather
than encoding all of these as an exact zero half-life. Finally, repeat a
prespecified subset using independent preparation and measurement.

## Panel size

The candidate panel sizes below are illustrative quotation scenarios rather than
required or recommended design choices. The 12 alleles, pair counts, and 20%
repeat allocations in the table are unvalidated planning examples; they do not
establish power, cost, or a delivery date, and allele and peptide identities
remain to be specified. Final population definition, HLA-group coverage,
independent repeat fraction, and sample size must be justified by the intended
use, required precision or power under realistic dependencies, and assay limits.

| HLA alleles | Pairs per allele | Primary measurements | Additional independent measurements on a 20% subset |
|---:|---:|---:|---:|
| 12 | 200 | 2,400 | 480 |
| 12 | 400 | 4,800 | 960 |
| 12 | 800 | 9,600 | 1,920 |

The [synthetic planning results](../further_research/round2/independent_data/design/simulation_summary.csv)
are conditional on assumed effect sizes and dependence. Consequently, final
sizing should account for assay failures, censoring, peptide families, and the
number of independent HLA groups. Additional peptides within a few alleles
cannot replace allele diversity when the target claim concerns transfer across
HLA molecules.

## Prespecified analysis

Before measuring outcomes, freeze the population, QC and replacement rules,
model and preprocessing artifacts, and prediction hashes. Use the difference
between medians of within-allele Spearman correlations as the primary endpoint,
with common eligibility and tie handling as defined in
[Methods](../../EVALUATION.md). Report absolute scores, per-allele results, and
paired uncertainty that respects peptide families. Generalisation across an HLA
population also requires HLA-group sampling in the analysis.

Report effect estimates with paired uncertainty, and prespecify any
application-relevant decision criteria before new outcomes are observed. If a
minimum meaningful effect is needed, justify it from the intended use and assay
characteristics rather than continuity with an earlier development heuristic. If
experimental prioritisation is the intended application, prespecify a separate
endpoint, such as recovery of stable complexes within a fixed measurement
budget. Report half-life calibration only if the measurement limits and
replicate uncertainty support it.

An optional adaptation comparison may allocate additional support measurements
per target allele, with the support budget chosen for the prospective
application and design rather than carried over automatically from the
retrospective experiment. Support and confirmation queries must have disjoint
peptide families, and support measurements must be counted separately. Freeze
the adaptation rule before opening support outcomes, then freeze adapted query
predictions before opening query outcomes. The locked-model comparison remains
primary.

This document specifies an unperformed study design; the public candidate
measurements described in [Data](../../docs/DATASETS.md) do not yet provide a
coherent independent validation panel.
