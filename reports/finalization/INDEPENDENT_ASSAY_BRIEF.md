# Independent stability assay: request-for-quote brief

Prepared 12 October 2026. **Draft for review; not sent and not funded.**
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

## Current evidence gap

The [public-data audit](../further_research/round2/independent_data/README.md)
retained only 129 conservative candidate rows across nine alleles, with only
one allele above 20 rows and unresolved assay/construct details. That is not
representative independent confirmation. External outcomes remain reserved.
The existing CSV lacks replicate-level measurements sufficient to estimate an
assay reliability ceiling. This brief makes the next acquisition step concrete;
it does not resolve those gaps or authorize contacting a provider.
