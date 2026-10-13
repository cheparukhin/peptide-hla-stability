# Structural inputs and representations

Boltz-2 supplies predicted coordinates and frozen learned features; experimental
PDB entries provide a limited structural reference. Predictive comparisons are
reported in the [research report](../reports/REPORT.md), with selection and
assessment defined in [EVALUATION.md](../EVALUATION.md).

## Coordinate and feature provenance

Production coordinates contain the reconstructed HLA ectodomain, β2-microglobulin
and native nine-residue peptide, with prepared HLA alignment information.
The cohort contains all 28,166 source pairs; each receives one Boltz-2 prediction
at seed 0, three recycling steps and 200 sampling steps. Coordinates, confidence
and residue/chain mapping are bound by the
[input provenance](../reports/ectodomain-20261004/input_provenance.json) and
[production verification](../reports/ectodomain-20261004/production_verification.json).
A small matched experimental-structure pilot checked peptide placement; it
cannot establish accuracy for every production structure or every label range.

The current compact representation cache covers the original 19,716 training
rows only. Its [completion audit](../reports/further_research/round2/resources/cache_completion_audit.json)
records the cohort, identities, feature hashes and numerical checks, including
its controller-provenance limitation. The model family is **Boltz-2**, the
inference package is `boltz==2.1.1`, and the weight-repository revision is
`6fdef46d763fee7fbb83ca5501ccceff43b85607`. These are different identifiers.
See the upstream [Boltz-2 methods](https://doi.org/10.1101/2025.06.14.659707)
and [implementation](https://github.com/jwohlwend/boltz).

| Representation | Construction | Dimensions |
|---|---|---:|
| R0 | Positional one-hot peptide and 34-residue HLA pseudosequence | 860 |
| R1 | R0 plus coordinate/confidence summaries | 969 |
| R2 | Concatenated mean token features for HLA, β2m and peptide | 1,152 |
| R3 | Position-preserving peptide/groove token features with fixed channel projection | 1,528 |
| R4 | Per-peptide-position means of directed peptide–groove pair features | 2,304 |
| R5 | Contact-pooled pair features at P1, P2, P9 and mean P3–P8 | 1,024 |

R3 and the more detailed position-by-contact control use fixed, label-independent
random channel projections. R4 and R5 retain all 128 pair-feature channels in
each direction. R5's fixed contact mask is the union of heavy-atom contacts
within 4.5 Å in five reference crystals, restricted to the canonical 34 HLA
positions. Shuffled-contact and dimension-matched pooled controls assess
specific positional choices. The
[feature protocol](../reports/further_research/round2/representations/feature_protocol.json)
is authoritative for tensor order, contact membership, projections and hashes.

These representations combine sequence, evolutionary information, molecular
context and structural pretraining. A supervised advantage does not by itself
identify a causal geometric mechanism. Comparisons that change chain context,
channel projection or pooling together cannot isolate positional information.
Cached inference requirements are in [ROUND2_INFERENCE.md](ROUND2_INFERENCE.md).

## Experimental structure tables

| Table | Contents |
|---|---|
| [PDB peptide–HLA entries](../data/pdb_pmhc_pairs.csv) | 841 records from 840 entries, assigned by exact identity over the 182-residue groove |
| [Allele templates](../data/allele_pdb_templates.csv) | One candidate threading template for each of the 75 molecules, with match quality recorded |
| [Exact dataset overlap](../data/pdb_rasmussen_overlap.csv) | 32 peptide–HLA pairs with both a measured half-life and a deposited structure |

These are saved database snapshots, not a complete audit of any foundation
model's training corpus. The broad table includes entries without a modelled
peptide and entries that are not experimental structures; filter on method,
peptide presence and molecular identity before using it as a reference.
The overlap has no recorded zero labels and a median half-life of six hours;
13 of its 32 pairs have at least one TCR-bound entry. It therefore favours
well-studied, structurally tractable complexes and does not represent the
complete assay population. Prefer a compatible TCR-free entry for a pose
comparison when available, and retain construct and resolution information.

Experimental structures have coordinate uncertainty, crystal-condition effects
and possible biological-context differences. They are not error-free kinetic
standards. Exact overlap in this snapshot is small, but related peptides,
shared HLA grooves and pretraining exposure remain possible. Supervised
HLA-group holdout does not remove earlier structural exposure. Claims of
pretraining novelty require a separate model-specific training-data audit.

## Descriptor interpretation

ProteinMPNN scores compatibility of a native sequence with a supplied backbone.
The peptide-only backbone used here was itself predicted with native peptide
and HLA context; removing the other chains changes graph neighbourhoods as
well as context. Thus the P/C comparison does not isolate native geometry or
a physical dissociation mechanism.

FoldX descriptors follow `RepairPDB` with `repairInterface=ALL` and peptide–HLA
`AnalyseComplex`, retaining β2m in the structure. The
[extraction protocol](../reports/followup3/foldx/protocol.json) pins the executable
and settings. Compatibility scores and computed energy descriptors do not
directly measure a dissociation rate. Even exact equilibrium affinity would
not uniquely determine off-rate without kinetic information. Nevertheless,
such descriptors can be useful statistical predictors; their value is tested
by matched supervised comparisons. An inconclusive result on one selected
panel or learner establishes neither equivalence nor failure of all
structure-based approaches.
