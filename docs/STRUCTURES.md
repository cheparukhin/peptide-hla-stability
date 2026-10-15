# Structural inputs and representations

Boltz-2 supplies predicted coordinates and frozen learned features, whereas experimental PDB entries provide a limited structural reference. Predictive comparisons appear in the [research report](../reports/REPORT.md), with model selection and assessment defined in [EVALUATION.md](../EVALUATION.md).

## Coordinate and feature provenance

Production coordinates contain the reconstructed HLA ectodomain, β2-microglobulin (β2m), and the native nine-residue peptide, generated with prepared HLA alignment information. The production cohort covers all 28,166 source pairs, each receiving one Boltz-2 prediction at seed 0 with three recycling steps and 200 sampling steps. The [input provenance](../reports/ectodomain-20261004/input_provenance.json) and [production verification](../reports/ectodomain-20261004/production_verification.json) records bind the resulting coordinates, confidence values, and residue-to-chain mappings. Although a small matched experimental-structure pilot checked peptide placement, that check cannot establish accuracy for every production structure or across every label range.

The compact representation cache covers only the original 19,716 training rows. Its [completion audit](../reports/further_research/round2/resources/cache_completion_audit.json) records the cohort, identities, feature hashes, and numerical checks, along with its controller-provenance limitation. Provenance distinguishes three identifiers: the **Boltz-2** model family, the `boltz==2.1.1` inference package, and the weight-repository revision `6fdef46d763fee7fbb83ca5501ccceff43b85607`. Upstream details appear in the [Boltz-2 methods](https://doi.org/10.1101/2025.06.14.659707) and [implementation](https://github.com/jwohlwend/boltz).

| Representation | Construction | Dimensions |
|---|---|---:|
| R0 | Positional one-hot peptide and 34-residue HLA pseudosequence | 860 |
| R1 | R0 sequence features plus 109 coordinate and confidence summaries | 969 |
| R2 | Concatenated mean token features for HLA, β2m, and peptide | 1,152 |
| R3 | Position-preserving peptide/groove token features with fixed channel projection | 1,528 |
| R4 | Per-peptide-position means of directed peptide–groove pair features | 2,304 |
| R5 | Contact-pooled pair features at P1, P2, P9, and mean P3–P8 | 1,024 |

Within the round-2 ladder, R1 appends 109 coordinate and confidence values to the 860-dimensional R0 sequence baseline. R3 and the more detailed position-by-contact control compress feature channels through fixed, label-independent random channel projections. By contrast, R4 and R5 retain all 128 pair-feature channels in each direction: R4 averages directed pair features over all 182 groove residues for each peptide position, whereas R5 averages over crystal-contact subsets within the canonical 34 HLA contact positions, keeping P1, P2, P9, and mean P3–P8.

R5 defines its fixed contact mask as the union of heavy-atom contacts within 4.5 Å across five reference crystals, restricted to those canonical 34 HLA positions. Shuffled-contact and dimension-matched pooled controls assess specific positional choices. The [feature protocol](../reports/further_research/round2/representations/feature_protocol.json) is authoritative for tensor order, contact membership, projections, and hashes.

Because these representations combine sequence, evolutionary information, molecular context, and structural pretraining, a supervised advantage does not by itself identify a causal geometric mechanism. Comparisons that alter chain context, channel projection, or pooling together cannot isolate positional information. Cached inference requirements appear in [ROUND2_INFERENCE.md](ROUND2_INFERENCE.md).

## Experimental structure tables

| Table | Contents |
|---|---|
| [PDB peptide–HLA entries](../data/pdb_pmhc_pairs.csv) | 841 records from 840 entries, assigned by exact identity over the 182-residue groove |
| [Allele templates](../data/allele_pdb_templates.csv) | One candidate threading template for each of the 75 molecules, with match quality recorded |
| [Exact dataset overlap](../data/pdb_rasmussen_overlap.csv) | 32 peptide–HLA pairs with both a measured half-life and a deposited structure |

These tables are saved database snapshots rather than a complete audit of any foundation model's training corpus. Because the broad table includes entries without a modelled peptide and entries that are not experimental structures, filter on method, peptide presence, and molecular identity before using it as a reference. The 32-pair overlap contains no recorded zero labels and has a median half-life of six hours, and 13 of its 32 pairs have at least one TCR-bound entry. It therefore favours well-studied, structurally tractable complexes and does not represent the complete assay population. Pose comparisons should prefer a compatible TCR-free entry when available and retain construct and resolution information.

Experimental structures remain subject to coordinate uncertainty, crystal-condition effects, and possible biological-context differences, so they are not error-free kinetic standards. Although exact overlap in this snapshot is small, related peptides, shared HLA grooves, and pretraining exposure remain possible. Supervised HLA-group holdout does not remove earlier structural exposure; claims of pretraining novelty require a separate model-specific training-data audit.

## Descriptor interpretation

ProteinMPNN scores the compatibility of a native sequence with a supplied backbone. Because the peptide-only backbone used here was itself predicted with native peptide and HLA context, removing the other chains changes graph neighbourhoods as well as context. Consequently, the P/C comparison isolates neither native geometry nor a physical dissociation mechanism.

FoldX descriptors follow `RepairPDB` with `repairInterface=ALL` and peptide–HLA `AnalyseComplex`, retaining β2m in the structure; the [extraction protocol](../reports/followup3/foldx/protocol.json) pins the executable and settings. Compatibility scores and computed energy descriptors do not directly measure a dissociation rate, and even exact equilibrium affinity would not uniquely determine off-rate without kinetic information. Nevertheless, such descriptors can serve as useful statistical predictors when evaluated in matched supervised comparisons. An inconclusive result on one selected panel or learner establishes neither equivalence nor failure of all structure-based approaches.
