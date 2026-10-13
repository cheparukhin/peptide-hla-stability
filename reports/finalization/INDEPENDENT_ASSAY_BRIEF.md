# Independent stability assay: request-for-quote brief

Prepared 12 October 2026; evidence-gap update 13 October 2026. **Draft for review; not sent and not funded.**
This is an acquisition specification, not experimental confirmation.

## Purpose and quantities

Measure peptide–HLA dissociation half-life for an independently acquired panel
that can compare two locked predictive procedures without reuse of development
outcomes. Please quote the following alternatives separately, including setup,
construct production, peptide synthesis/QC, repeats, raw-data delivery and lead
time. Currency, taxes, exclusions and failure/replacement policies should be
explicit; no real assay price or delivery date has been established.

| Option | HLA alleles | Distinct pairs per allele | Primary measurements | Independent repeats on 20% subset |
|---|---:|---:|---:|---:|
| Smaller | 12 | 200 | 2,400 | 480 |
| Starting design | 12 | 400 | 4,800 | 960 |
| Larger | 12 | 800 | 9,600 | 1,920 |

Allele and peptide identities remain to be frozen. The 12×400 option is a
bounded starting design from the [synthetic planning study](../further_research/round2/independent_data/design/README.md),
not a guarantee of statistical power. Detecting a true 0.02 improvement was
substantially harder than detecting 0.05 under strong dependence. At a true
gain of 0.05, detecting improvement does not establish that gain exceeds 0.05.

## Experimental and data requirements

- Kinetic **dissociation half-life**, with units, preferably a consistent 37°C
  protocol. Thermal melting, affinity and denaturant resistance are different
  endpoints and cannot substitute silently.
- Exact peptide sequence, length, modifications, purity, source/family and sample
  identifier; exact physical HLA and β2m constructs, allele names and engineered
  substitutions. Model-input pseudo-sequences alone do not establish constructs.
- Protocol, temperature, batch/plate layout, reagent lots, time points, controls,
  fitting procedure and all deviations. Blind laboratory identifiers to model
  predictions; balance peptide-family and allele strata across batches.
- Raw time courses and replicate-level estimates, fit/QC flags, uncertainty,
  lower/upper measurement limits and censoring indicators. Retain failures and
  floor values with their reasons. Independent repeats require separate
  preparation/measurement, not duplicate file rows or repeated curve fitting.
- Clear measurement lineage, confirmation that measurements are newly acquired
  rather than republished, and rights to use and redistribute the agreed data.

## Panel and analysis freeze before outcomes

An identity/provenance audit must exclude exact pairs and peptide families
exposed during development, assess checkpoint exposure where knowable, and
resolve physical constructs. Allocate across several HLA sequence groups,
including poorly covered alleles; freeze peptide-family and lifetime-proxy
strata before measurement. A disagreement-enriched exploratory subset may be
separate, but must not become the entire confirmation population.

Freeze eligible identities, replacement/QC rules, the candidate and reference
artifacts, all preprocessing, and the prediction file hashes before opening
outcomes. Do not fit an affine map or tune a model on confirmation labels.
The primary endpoint is the difference between the two medians of per-allele
Spearman correlations over the same eligible panel. Use the frozen ≥20-row
eligibility rule, report ≥50-row sensitivity, and retain the prespecified
2,000 paired whole-peptide-cluster resamples. Any claim over an unseen-HLA
population additionally requires adequate independent HLA-group coverage and
an explicitly frozen crossed-group analysis.

Report failures, missingness, floor ties, replicate agreement, both procedures'
absolute scores and the paired interval. Improvement requires a 95% lower
bound above zero; a worthwhile gain requires it above 0.05. An inconclusive
interval does not establish equivalence or a measurement-noise ceiling.

## Optional future few-shot validation arm

The [positive conditional few-shot development result](../followup3/REPORT.md#completed-few-shot-result)
motivates an optional, separately prespecified validation arm alongside the
locked-model confirmation primary above. It is not an executed study or
additional acquisition authorization. The quoted quantities and primary
comparison remain unchanged.

Before any outcomes are opened, define a separate calibration-support panel
and a confirmation query panel whose labels remain unopened throughout
adaptation. Protect peptide families: support and query clusters must not
overlap each other or exposed development families. Freeze support identities,
target alleles, support budgets and reuse rules, preprocessing, source artifacts
and the entire residual-adaptation recipe before opening support labels. The
20-support-label setting is a candidate to prespecify, not a newly selected
validated optimum. Target support labels must not be used for tuning or
stopping. Freeze both zero-shot and adapted query predictions and their hashes
after support adaptation and before opening any query labels.

Count every calibration-support measurement explicitly, separately from query
measurements and independent repeats; report distinct pairs, peptide clusters,
per-allele budgets and any support reuse. Support labels cannot silently reduce
the locked primary's confirmation panel or be counted as held query outcomes.
Any extra support acquisition would need its own quantity/price specification
and authorization. Prespecify the optional paired query comparison, absolute
scores, eligible panel, missingness/QC and uncertainty, keeping it subordinate
to the locked-model primary. This arm makes no power, prospective assay-savings
or independent-confirmation claim before new measurements exist.

## Current evidence gap

The completed [metadata/methods audit](../followup3/independent_data/PROVENANCE.md)
preserves **129 reserved candidate rows across nine alleles**. Primary methods
show heterogeneous endpoints, temperatures and physical constructs, including
an assay-category mismatch; the candidates cannot presently be pooled as one
kinetic validation panel. The largest single-study/allele stratum is **Strønen
2016, HLA-A*02:01, with 23 rows**. Its original methods/supplement, physical
construct, kinetic endpoint and row-to-measurement lineage remain unresolved.
Even a defensible narrow check would remain one study/one allele without a
≥50-row sensitivity, leaving representative independent confirmation open.

External candidate outcomes remain reserved. Incidental public-paper result
visibility during methods retrieval is disclosed in the audit; this is not a
fully blinded literature review. Bibliographic resolution is not measurement-
level provenance. The existing CSV lacks replicate-level measurements
sufficient to estimate an assay reliability ceiling. This brief makes the next
acquisition step concrete; it does not resolve these gaps or authorize provider
contact, extra acquisition or outcome access.
