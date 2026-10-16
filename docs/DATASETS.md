# Data and assay provenance

The analysis uses the measured peptide–HLA stability dataset from
[Rasmussen et al. (2016)](https://doi.org/10.4049/jimmunol.1600582), supplied as
[`data/rasmussen_et_al_dataset.csv`](../data/rasmussen_et_al_dataset.csv)
alongside the [original paper](../data/rasmussen_et_al_2016.pdf). The table
contains **28,166 unique peptide–HLA pairs, 75 HLA-A/B molecules and 5,633
nine-residue peptides**, with no missing values or duplicate pair identities.
The source CSV is read-only, and its checksum is recorded in
[`data/SHA256SUMS`](../data/SHA256SUMS).

The source CSV has five columns:

| Field | Meaning |
|---|---|
| `allele` | HLA molecule name, including any C67S construct suffix |
| `peptide` | Nine-residue peptide sequence |
| `thalf_hours` | Recorded complex dissociation half-life in hours |
| `hla_seq` | 182-residue HLA α1/α2 groove sequence |
| `hla_pseudoseq` | 34-residue HLA contact pseudosequence |

## What the outcome measures

The study reports complex dissociation half-lives at 37 °C from a scintillation
proximity assay. Because the underlying method follows radiolabelled
β2-microglobulin dissociation as a proxy for peptide dissociation, it defines a
complex-stability endpoint rather than a direct observation of every peptide's
microscopic unbinding event. This assay was validated against peptide-based
measurements in [Harndahl et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC4341823/).

Although Rasmussen et al. report geometric means from two independent
experiments, this repository supplies only the aggregated half-life values,
without separate replicate measurements, raw dissociation curves, or
measurement-level fitting and censoring flags. Experimental replication
therefore existed, but its reliability and an assay noise ceiling cannot be
estimated from this file.

Recorded half-lives range from 0 to 256.7 hours (median 1.1 hours), and
exactly 5,679 values (20.16%) are zero. Their concentration at zero is
consistent with low or unresolved stability, but it does not prove that every
zero is a left-censored measurement at one known detection limit; the supplied
fields cannot separate rounding, insufficient complex formation, low signal,
and fitting failure. Analyses that model all zeros as censored are therefore
sensitivity analyses under an assumption, not recovered measurement metadata.
Ranking retains ties, and error against recorded zeros is not necessarily error
against latent physical half-life.

## Sampling and constructs

The assay collection is selected rather than population-representative. To
permit measurement against multiple HLA molecules, Rasmussen et al. grouped
peptides partly using predicted affinity (NetMHCpan 2.8, <500 nM or top 2%
rank), and the published modelling cohort retained molecules with at least one
stable and two nonstable observations using a two-hour threshold. These choices
condition the observed peptide diversity and stability distribution. The
resulting table is sparse: 6.67% of all possible allele–peptide combinations
were measured, and a peptide occurs on as many as 36 alleles.

All 75 molecules, including the engineered constructs `HLA-B*14:01(C67S)`,
`HLA-B*14:02(C67S)` and `HLA-B*39:06(C67S)` (1,135 rows), remain in the
canonical analysis. Their supplied heavy-chain sequences contain serine at
mature residue 67, and no matched wild-type counterpart is measured in this
dataset; consequently, their high zero fractions cannot identify the mutation's
effect without matched controls. In addition, the two B*14 constructs share the
same 34-residue pseudosequence, so a model using only that pseudosequence
cannot distinguish them even though their full 182-residue sequences differ.

In the supplied schema, `hla_seq` covers the 182-residue α1/α2 groove and
`hla_pseudoseq` contains 34 selected residues, whereas structure generation
uses a larger reconstructed ectodomain plus β2-microglobulin. Because these
inputs differ in molecular context, a performance difference cannot be
attributed solely to three-dimensional structure or pretraining. Physical
construct details also matter when comparing this assay with an external study.

## Data references

| Source | Role |
|---|---|
| [Frozen split assignments](../data/splits.csv) | Canonical original train/validation/test membership; load through `pepstab.data.load_with_splits()` |
| [Dataset summary](rasmussen_et_al_dataset_summary/dataset_summary.csv) and [per-allele statistics](rasmussen_et_al_dataset_summary/per_allele_stats.csv) | Machine-readable descriptive counts and distributions |
| [Non-C67S subset](../data/c67s_cleanup/rasmussen_no_C67S.csv) and [C67S subset](../data/c67s_cleanup/benchmark_C67S.csv) | Lossless reference subsets; neither replaces the canonical cohort |
| [Affinity reference](AFFINITY_REFERENCE.md) | Auxiliary binding-affinity provenance and leakage constraints |
| [Structural inputs](STRUCTURES.md) | Deposited structures, Boltz representations and descriptor interpretation |

Join derived data by exact `(allele, peptide)` identity, retaining construct
suffixes. Because `pair_id` is positional in the unmodified source file, it
must not be reinterpreted as row position in a subset. The older split table
under `data/c67s_cleanup/` is not used; current assessment populations and
eligibility are defined in [EVALUATION.md](../EVALUATION.md).

## External comparisons

NetMHCstabpan was trained on this dataset and is therefore not an independent
comparator. Its published training and evaluation design also added 1,000
predicted weak-binding nine-mers per allele with assigned zero-hour outcomes
and used a different target transformation. Consequently, its reported Pearson
correlation of 0.676 cannot be compared directly with median within-allele
Spearman on measured rows; synthetic negatives are assumptions, not
experimental half-lives.

After the recorded overlap exclusions, the
[reserved candidate inventory](../reports/followup3/independent_data/candidate_metadata_audit.json)
contains **129 rows, 126 peptides, 113 peptide clusters and nine HLA alleles
from 36 PMIDs**. The largest single study–allele stratum is 23 HLA-A*02:01 rows
attributed to Strønen et al., for which the original measurement mapping and
methods remain unresolved, while every other study–allele group contains at
most ten rows. These counts use identities and metadata, without checking
outcome variation or completeness.

Across these sources, the
[study-method inventory](../reports/followup3/independent_data/study_method_status.csv)
records heterogeneous purified and cellular endpoints with unresolved or
conflicting temperature and construct annotations. Cellular surface
persistence, thermal stability and purified-complex dissociation are different
endpoints; release assay labels alone do not make them interchangeable. Because
missing measurement identifiers also prevent ruling out reprinted or reused
values, pooling these rows does not establish a coherent multi-allele
confirmation cohort.

Candidate outcome columns remain reserved from assessment. Although incidental
public result text was visible during literature retrieval—so the literature
audit was not fully blinded—those numeric outcomes were not joined to
candidates or used for scoring or selection. Independent confirmation requires
resolved measurement lineage, overlap, endpoint, temperature and physical
constructs, with eligibility fixed before outcome assessment. Requirements for
new measurements are set out in the
[independent assay brief](../reports/finalization/INDEPENDENT_ASSAY_BRIEF.md).
