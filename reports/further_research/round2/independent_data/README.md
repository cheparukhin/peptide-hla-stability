# Independent half-life data feasibility and public donor audit

Recorded 12 October 2026. Scope: [canonical research plan §9](../../../NEXT_RESEARCH_PLAN.md#9-milestone-5--independent-data-and-confirmation). This is a sequence/metadata inventory, not a performance experiment or a confirmation result. No external labels were read into the audit, summarized, scored, or used for fitting. Raw CSVs necessarily contain public labels; they remain in ignored `external/round2/spearmint/`. Reservation here is a documented analysis policy, not cryptographic blinding or restricted filesystem access.

## Practical conclusion

The published stability donor cannot provide independent confirmation: **all 27,031 released peptide–allele pairs are canonical pairs, and all 5,633 canonical peptides occur in its partitions**. The assay-conditioned test release is a plausible source of reserved measurements, but only **129 rows, 126 peptides, 113 components and 9 alleles** remain under the conservative available-donor exclusion. Only one retained allele has at least 10 rows (also at least 20 rows). This is insufficient evidence that the plan's representative multi-allele confirmation and paired ranking power requirements are met. Confirmation remains open.

The less conservative exclusion of supervised donor fitting/selection partitions leaves **378 rows, 264 peptides, 248 components and 13 alleles**, including 8 alleles with at least 10 rows and 6 with at least 20. Its retained coverage depends on treating published upstream test partitions as unused in checkpoint development. That assumption has not been established. In both policies, 11 rows have allele names absent from the canonical corpus; sequence-based HLA grouping and adequate unseen-HLA coverage have not been established.

## Pinned sources and access

The authors' [public repository](https://github.com/dhuvik/spearmint) is the target of the paper's `pirl-unc/spearmint` link. Audit data are pinned at [commit d548dee9b04dd0e3a9d7a50a593d2c030e31d2a5](https://github.com/dhuvik/spearmint/tree/d548dee9b04dd0e3a9d7a50a593d2c030e31d2a5). Nine raw CSVs downloaded successfully without login, payment, or external contact. [source_manifest.json](source_manifest.json) records exact raw URLs, SHA-256 hashes, verified Git blob SHA-1 hashes, byte counts, UTC verification time, canonical source/split hashes and the audit script hash. The pinned repository [license](https://github.com/dhuvik/spearmint/blob/d548dee9b04dd0e3a9d7a50a593d2c030e31d2a5/LICENSE) is MIT for software/documentation; original measurement redistribution/access terms must still be checked against IEDB and the originating studies before redistributing raw measurements. No raw source file is committed here.

The [SPEARMINT preprint](https://www.biorxiv.org/content/10.64898/2026.06.28.735023v1.full) describes affinity pretraining, half-life fine-tuning and later assay-conditioned recalibration, and identifies NetMHCstabpan as the stability source. This establishes claimed dataset lineage, not a cryptographic mapping from these partitions to the deployed checkpoint's complete fitting/selection history. This audit downloaded no weights and ran no model.

| Published corpus | Train rows | Validation rows | Test rows | Total rows | Alleles (pooled) | Half-life confirmation use |
|---|---:|---:|---:|---:|---:|---|
| binding_stability | 21,626 | 2,705 | 2,700 | 27,031 | 72 | Republishes canonical pairs; no independent cohort |
| binding_affinity | 126,683 | 21,712 | 21,712 | 170,107 | 109 | Different endpoint; upstream exposure audit only |
| stage3_assay_conditioning | 8,465 | 543 | 1,133 | 10,141 | 40 | Reserved test candidates, subject to overlap and assay checks |

Measured release discrepancies: the stability files have 27,031 rows rather than the preprint's 27,034. The canonical pairs absent from these files are precisely the three explicitly named C67S constructs (1,135 rows); 28,166 − 1,135 = 27,031. Construct suffixes were preserved rather than collapsed into wild-type allele names. The assay-conditioned test file contains 876 9-mer rows rather than the preprint's reported 888; no missing rows were fabricated and no label reconciliation was attempted. These discrepancies must be resolved before treating the release as an exact reconstruction of all paper analyses.

## Exact and component overlap

[overlap_summary.csv](overlap_summary.csv) contains donor-to-canonical results by actual source partition, with separate exact pair, exact peptide, direct Hamming ≤3, component-connected and non-9-mer counts. [canonical_exposure.csv](canonical_exposure.csv) reverses the direction and reports original train/validation/test exposure separately. All canonical data are loaded through `pepstab.data.load_with_splits()`; the frozen splits and source CSV are untouched.

| Released corpus / partition | Rows | Exact canonical pair rows | Exact canonical peptide rows | Direct Hamming ≤3 rows | Component-connected rows | Component-novel standard 9-mer rows |
|---|---:|---:|---:|---:|---:|---:|
| Stability / all | 27,031 | 27,031 | 27,031 | 27,031 | 27,031 | 0 |
| Affinity / all | 170,107 | 7,627 | 54,682 | 60,930 | 63,474 | 62,901 |
| Assay conditioning / all | 10,141 | 6,135 | 6,238 | 6,417 | 6,444 | 3,145 |
| Assay conditioning / test | 1,133 | 0 | 0 | 66 | 84 | 792 |

Pairs use exact `(peptide, allele)` strings. Pair identity does not prove equal physical constructs, repeated experimental measurements, or independent measurement provenance. Peptide matching ignores allele, as required for peptide-novel confirmation. The full threshold graph includes canonical peptides plus **all released partitions of the named donor corpus**; therefore a partition's component results can include bridges through other partitions. Direct-neighbour and exact-match columns use only that named partition. This conservative graph definition is explicit in the result table.

The component algorithm is exact: two standard 9-mers have Hamming distance ≤3 if and only if one of their 84 six-position signatures agrees. Unioning matching signature groups recovers all connected components, including transitive paths. No approximation or nearest-neighbour truncation is used. A synthetic bridge test and exhaustive small-graph comparison verify that direct-neighbour checks alone can miss connected candidates.

The public stability **training** file contains 4,435 exact pairs from the original canonical test and reaches 844/1,023 of its frozen peptide components in the declared graph. The affinity training file contains 940 exact canonical-test pairs and reaches 657/1,023 test components. These are actual released donor-file overlaps rather than the previous self-corpus/IEDB proxy audit. Exposure of a particular deployed checkpoint remains conditional on the upstream fitting/selection provenance.

**Length limit:** Hamming components here apply only to standard-amino-acid 9-mers. Affinity has 43,732 rows at other lengths and stage 3 has 552; they are reported as unassessed, not counted as clean. Unequal-length similarity, shifted alignments, source-protein exposure, structural pretraining and supervised donors absent from this release remain unaudited. Thus retained counts are conditional candidates under the stated 9-mer rule, not a proof of globally unseen biology.

## Reservation and remaining coverage

[confirmation_eligibility.csv](confirmation_eligibility.csv) and [confirmation_assay_strata.csv](confirmation_assay_strata.csv) report aggregate surviving coverage. Neither contains outcome labels or sequence identities. All policies seed exclusions with **the complete canonical corpus**, not just its training partition, because previously examined benchmark rows are not new evidence.

- `train_val`: add affinity/stability train + validation and stage-3 train + validation to the exclusion seeds.
- `all_partitions`: add affinity/stability train + validation + test and stage-3 train + validation. Stage-3 test is the candidate cohort; it is not seeded against itself.

Both graphs include the candidate peptides when computing connectivity, so candidate-to-candidate bridges cannot carry a connected component into the reserved cohort. Stage-3 train and validation yield no eligible confirmation rows by design because they are supervised exposure seeds, even when novel versus the canonical corpus alone.

| Exclusion policy | Candidate rows | Eligible rows | Eligible peptides | Eligible components | Alleles | Rows lost (including non-9-mers) |
|---|---:|---:|---:|---:|---:|---:|
| Canonical component overlap only | 1,133 | 792 | See overlap tables | See overlap tables | See overlap tables | 341 |
| Canonical + available donor train/validation | 1,133 | 378 | 264 | 248 | 13 | 755 |
| Canonical + conservative available donor partitions | 1,133 | 129 | 126 | 113 | 9 | 1,004 |

| Assay and temperature | Eligible rows, train/validation exclusion | Eligible rows, conservative exclusion |
|---|---:|---:|
| SPA, 37°C | 26 | 26 |
| Purified fluorescence, 37°C | 63 | 50 |
| Purified fluorescence, 25°C | 230 | 0 |
| Cellular fluorescence, 37°C | 59 | 53 |

The 378-row cohort contains 230 missing PMIDs, but all have `ref_iri`; all 129 conservative candidates have both fields. Resolving those references and confirming actual assay endpoints is unfinished. Only 26 conservatively retained rows share the SPA/37°C setting of the canonical assay lineage; heterogeneous assay confirmation must declare the endpoint and assess protocol shift separately. No external label ranges, floors, censoring rates or label-dependent eligibility have been inspected.

## Assay and provenance inventory

| Source / lead | Accessible metadata and endpoint | Construct and protocol status | Floor / uncertainty / independent-use status |
|---|---|---|---|
| SPEARMINT stability release | All rows explicitly tagged SPA at 37°C; standard 9-mers | Model-input HLA sequence lengths 181, 362 or 365; these are mappings, not verified experimental construct records | No floor, censor flag, replicate, SD or SE columns. Fully overlaps canonical pairs |
| SPEARMINT stage-3 release | SPA, purified/cellular fluorescence and Other; temperatures 25°C, 37°C or missing; PMID/ref_iri available with gaps | Input sequences are 362/365/366 aa in test. Actual assay constructs/protocols require study-level resolution | Labels remain reserved; missing floor/uncertainty fields prevent assay-ceiling or calibration claims |
| Rasmussen 2016, PMID 27402703 | Primary publication and official [DTU source](https://services.healthtech.dtu.dk/suppl/immunology/NetMHCstabpan-1.0/) accessible; canonical lineage | SPA; includes explicitly engineered C67S constructs | Publication says reported half-life is geometric mean of two independent experiments; canonical CSV does not supply replicate-level measurements, so measurement reliability cannot be estimated from it |
| NetMHCstab 2014, PMID 23927693 | 5,509 measurements over 10 HLA molecules in the [primary abstract](https://pubmed.ncbi.nlm.nih.gov/23927693/) | Earlier same-group kinetic stability lineage; 2016 combines new SPA screening with an earlier in-house dataset | Cannot designate these as independent until actual identity/source lineage is audited; this subtask has not downloaded the 2014 measurement file |
| Blaha 2019, PMID 30425106 | [Full text and supplement access](https://pmc.ncbi.nlm.nih.gov/articles/PMC6324732/) | Main screen uses thermal melting temperature; a smaller subset has bead-based kinetic measurements at 37°C, with HLA-A*02:01 heavy chain + β2m and UV exchange | Kinetic subset reports independent time-course replicates; inspect its source/identity separately. Thermal Tm values are not half-life labels |
| NeoScreen 2023, PMID 37275886 | [Primary methods accessible](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2023.1151659/full); methods inventory only | Urea-resistance readout normalized to an allele reference | Endpoint is relative denaturant resistance, not kinetic half-life; exclude from half-life confirmation. No pathogen-specific sequences or labels were analyzed |

The Rasmussen replicate description comes from [primary Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC4976001/). Absence of supplied replicate-level data must not be rewritten as absence of experimental replication. The publication does not make the aggregated CSV sufficient to estimate an irreducible noise ceiling.

## Feasibility decision and next gates

Keep stage-3 test outcomes reserved while the ranking procedures, preprocessing and analysis are frozen. Do not repurpose the candidate outcomes for adaptation/calibration and then call them independent confirmation. Resolve originating references, establish physical construct, kinetic endpoint, temperature, assay floor and uncertainty without choosing examples by outcomes. Verify full supervised checkpoint ancestry and cross-length exposure where relevant. Only then can eligible rows be frozen in a row manifest and a one-time analysis planned.

The 129-row conservative cohort is a bounded candidate for a limited external assay-transfer check. It does not presently satisfy the representative independent confirmation gate, and the 11 allele-novel rows do not establish unseen-HLA power. This identity audit itself did not size an acquisition panel. A subsequent [bounded synthetic design study](design/README.md) evaluates panel sensitivity and quote options. No assay quote, partner commitment, wet-lab launch, real assay cost forecast or calendar promise exists. A representative new assay cohort with accessible replicate-level results remains the strongest route if public coverage is insufficient. No external communication or paid compute was used.

Reproduce with `.venv/bin/python scripts/round2_independent_data_audit.py`. It validates the nine pinned Git blob hashes before parsing metadata, then regenerates aggregate outputs. Validate with `.venv/bin/python -m pytest tests/test_round2_independent_data_audit.py -q` (4 passed). The tests cover exact graph equivalence, transitive exclusion, invalid-length handling and the external-label allowlist.
