# Reserved independent-assay provenance audit

13 October 2026. **Metadata/methods audit, not external model assessment.**
The original 129-row candidate inventory is preserved. No donor outcome column
was retained, displayed, summarized, used to choose rows, fitted or scored.
No model, wet-lab work, external contact or paid compute was run.

## Decision

**Representative independent confirmation remains unavailable.** The candidates
cannot presently be pooled into a common kinetic-assay validation panel. The
release's assay category and 37°C annotation are insufficient to establish the
actual measurement method, construct and conditions. Primary methods verify
material protocol differences and one definite assay-category mismatch.
The only single-study/allele group with ≥20 candidate rows is Strønen 2016,
HLA-A*02:01, with 23 rows; its original methods supplement was not accessible
in this bounded audit. A limited frozen-model external check could become
defensible after its measurement mapping, constructs, time-course/QC and model
ancestry are verified. It would be one study/one allele, with no ≥50-row
sensitivity, and would leave the representative multi-allele confirmation gate
open. This audit does not authorize outcome access or establish that those gates
are already met.

## Reproduced identity coverage and source lineage

The [metadata audit](candidate_metadata_audit.json) reproduces **129 rows,
126 peptides, 113 standard-9mer Hamming≤3 components, nine alleles and
36 source PMIDs** under the original conservative exclusion policy. The
[40 source/assay/allele groups](candidate_source_groups.csv) preserve the
release `ref_iri` and PMID join (PMID `.0` suffixes normalized only for citation
lookup). PubMed resolved all 36 cited publications; their titles, DOI/PMCID
and source URLs are recorded in [publication_metadata.json](publication_metadata.json).
That is bibliographic resolution, not measurement-level lineage verification.

| Allele | Retained rows |
|---|---:|
| HLA-A*02:01 | 99 |
| HLA-A*03:01 | 8 |
| HLA-B*14:02 | 7 |
| HLA-C*03:04 | 4 |
| HLA-A*01:01 | 3 |
| HLA-B*08:01 | 3 |
| HLA-B*57:01 | 3 |
| HLA-A*11:01 | 1 |
| HLA-A*24:02 | 1 |

Release strata remain 26 SPA/37°C, 50 Purified_Fluor/37°C and 53
Cellular_Fluor/37°C. Only A*02:01 reaches ≥20 rows when pooled. The
[Strønen Science paper](https://pubmed.ncbi.nlm.nih.gov/27198675/)
([IEDB reference 1037506](https://www.iedb.org/reference/1037506)) contributes
23 purified-fluorescence-tagged rows; each other individual source/allele group
has at most ten. These are counts by identity/metadata, with no check of label
variation, floors, rank ties or missing outcomes.

The release has a reference URI but no original IEDB assay/measurement identifier,
source table/figure/cell identifier, physical-construct field, probe modification,
drug condition, raw time course, replicate identity, fitting flag, floor or
censor flag. Its mapped full HLA input sequence cannot identify the actual
experimental construct. Reference identity alone does not establish independent
measurement, rather than reprinted or reused values.

## Verified primary methods and unresolved row mapping

| Originating source; retained rows | Verified methods | Consequence / unknowns |
|---|---|---|
| [Zappacosta 1997, PMID 9177214](https://pubmed.ncbi.nlm.nih.gov/9177214/); [primary full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC21046/); 4 C*03:04 | Bacterially produced HLA-Cw*0304 heavy chain and radioiodinated β2m; complex formation and β2m dissociation assessed by HPLC gel filtration at time points. | Kinetic stability endpoint is supported, but the release's SPA category is not the actual gel-filtration readout. Methods excerpt did not establish exact time grid, replicate design or floor. Temperature37°C remains release metadata, not independently verified here. |
| [Bassani-Sternberg 2017, PMID 28832583](https://pubmed.ncbi.nlm.nih.gov/28832583/); [primary methods](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005725); 7 B*14:02 | WT and engineered W97R heavy chains; biotinylated β2m; refolding at 4°C for 48h; ELISA time series at 37°C, 0/8/24/48/72h; exponential off-rate fit, half-life ln2/koff; average over two replicates; lower bounds for unresolved slow dissociation. | Release Purified_Fluor is not a precise method description. WT/W97R row mapping is absent. The paper's methods paragraph also switches from B14:02 to B14:01, an unresolved textual inconsistency. Do not infer WT from the nominal release allele or overwrite its input sequence. Replicate independence and row censoring remain unverified. |
| [Blaha 2019, PMID 30425106](https://pubmed.ncbi.nlm.nih.gov/30425106/); [primary methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC6324732/#S2); 3 A*02:01 | Main screen is DSF Tm. A separate kinetic assay couples biotinylated peptide/HLA-A2 to streptavidin beads, incubates at37°C, measures remaining complex by anti-β2m or high-affinity TCR flow cytometry, and fits first-order decay. Methods report at least two replicate experiments and independent time courses. | Genuine kinetic measurements exist, but candidate rows require a source-table/measurement-ID mapping to that endpoint; do not substitute Tm. Exact construct/substitutions, row-specific probe, time grid, limits and replicate-level access remain unresolved. |
| [Lin 2021, PMID 33490292](https://pubmed.ncbi.nlm.nih.gov/33490292/); [primary methods §2.3](https://pmc.ncbi.nlm.nih.gov/articles/PMC7803423/); 6 A*02:01 | TAP-deficient T2A2 cells, peptide plus human β2m loading for 18h, brefeldin A treatment 1h, then 37°C sampling at 0/2/4/6h; BB7.2 fluorescence. DC50 is time to loss of 50% of surface complexes present at 0h. | Cellular persistence is affected by cell-surface processes and is a separate estimand from purified-complex dissociation. Do not pool its absolute lifetime with purified assays. No row-level censoring or replicate definition was verified. Model-input HLA sequence does not establish cellular construct/genotype. |
| [Illing 2021, PMID 34093574](https://pubmed.ncbi.nlm.nih.gov/34093574/); [primary methods](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2021.672737/full); 3 B*57:01 | Multiple assays: cellular pulse-chase/thermal assays and purified fluorescence-polarization dissociation. The latter uses HLA-B*57:01 ER luminal domains with C-terminal Fos zipper, human β2m, TAMRA-labelled peptide, excess unlabelled competitor and conditions with/without abacavir or tapasin complexes; polarization experiments at25°C. | Release Cellular_Fluor/37°C cannot identify which measurement was used. If these are polarization-derived kinetics, temperature/assay annotation is inconsistent; this is not established at row level. Fos tag, fluorescent substitutions, drug and chaperone conditions are absent from release metadata. |
| [Strønen 2016, PMID 27198675](https://pubmed.ncbi.nlm.nih.gov/27198675/); 23 A*02:01 | Bibliographic identity verified. Blaha's primary methods cite this assay lineage for anti-β2m bead-based dissociation, but that is an indirect methodological pointer. | Original Science methods/supplement access failed. Exact original construct, time grid, independent repeats, fit/QC, operational limits and row-to-measurement lineage are not verified. This largest coherent-looking stratum must remain gated. |

For the other 30 source PMIDs, citation metadata are verified; this bounded audit
does not claim that their physical constructs, kinetic endpoint or protocol were
verified. See [study_method_status.csv](study_method_status.csv) for an explicit
verified/unknown inventory. Lack of verified replicate metadata is not evidence
that experiments lacked replication. Neither thermal Tm nor urea resistance is
a replacement for kinetic half-life; no such conversion was attempted.

## Preservation, access limits and next gates

All original 129 candidates stay in the inventory; unresolved or heterogeneous
strata are flagged, not silently deleted. The source audit was chosen from
identity counts and method relevance, never measured outcomes. The new script
uses only the Python standard library and projects explicit allowlists immediately
after CSV parsing. It checks all nine pinned source SHA256s plus canonical source
SHA256, joins candidate components by the same exact six-position-signature rule,
and never calls the canonical loader that eagerly loads outcomes. This is a
metadata-only exception to that loader's normal use, needed to preserve reserved
validation/test outcomes. Original sources, splits, scripts and reports are
unchanged. Row manifest stores source row indices and identity hashes, not labels.

Reservation is an analysis policy, not cryptographic or observer blinding.
During methods retrieval, a generic web find/search unexpectedly returned public
paper result sections for Blaha 2019 and Zappacosta 1997. No published numeric
outcomes were joined to candidate identities, copied into the audit, summarized,
scored or used for selection. Subsequent page retrievals were projected to methods
before displaying to the researcher. This incidental publication-result visibility
precludes describing the literature audit as fully blinded, while the donor
outcome-column reservation remains intact. It is not a reason to alter the129-row
cohort after the fact.

PubMed metadata retrieval succeeded; PubMed full-text retrieval returned an
internal error. PMC access repeatedly returned CAPTCHA; the Science original
paper/supplement and IEDB reference webpage returned errors in this session.
Two tentative IEDB API reference routes returned404 and established no facts.
These access failures are retained as unknowns, not inferred method evidence.
No authenticated workaround or contact was attempted.

Before a limited external assessment: resolve the largest 23-row stratum against
original measurement IDs and methods using outcome-free metadata; identify
physical constructs and assay conditions; freeze row eligibility and QC rules,
paired candidate/reference artifacts, preprocessing and prediction hashes; verify
supervised checkpoint exposure beyond this released donor; predeclare a single
study/allele ranking analysis with honest uncertainty and no calibration. If that
mapping fails, keep outcomes reserved. Even if it succeeds, the experiment would
assess narrow assay transfer and would not establish multi-allele confirmation,
unseen-HLA performance, assay reliability ceiling or equivalence.

Reproduce the inventory with `python3 scripts/followup3_independent_metadata.py`.
Executed successfully on 13 October 2026: pinned hashes passed; counts matched the
original conservative audit (129/126/113/9). No external assessment was run.
