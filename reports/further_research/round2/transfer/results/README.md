# Bounded transfer assessment

Development evidence on original training rows, conditional on fitted predictions.
All five HLA and all 25 joint cells are retained. Primary endpoints pool held
predictions before computing the median per-allele Spearman. Cell medians are
separate diagnostics; empty cell panels remain undefined, never excluded.

The references are the original 30-member sequence recipe and the separately
named bounded_MLP_sequence_selector, comparing original log MLP against shrunk
centred MLP k100/k20. It excludes trees and PairLogit from the full five-family
main selector. At most two explicitly main-nominated primary Boltz head
procedures retain their existing regularization grids. The embedding selector
includes both sequence families as baseline fallbacks and adds no fits. Both 20-row and 50-row panels and every declared contrast
are reported. These are exploratory development comparisons, without a
multiplicity-adjusted confirmatory claim.

Paired peptide, HLA-group, and crossed bootstrap intervals use 2,000 draws.
Crossed draws independently resample both axes; HLA multiplicities also replicate
alleles in the equal-per-allele median. Sparse intersections can make pigeonhole
intervals conservative. Leave-one-HLA-group-out deltas are sensitivity diagnostics,
not independent confidence intervals. Draws losing label variation are counted;
constant predictions contribute the frozen zero score. Intervals omitting their
point estimate carry a bias flag and bootstrap-mean diagnostic.

Joint final neural members stop on five diagonal HLA/peptide intersections and fit
outside both corresponding groups. All cross quadrants are excluded per member;
this stopping design is distinct from the full 25 outer assessment intersections.
Member records give gradient, stopping and cross-quadrant counts. Diagonal
stopping covers only a subset of fitting rows. Sequence inner calls use 12
members; final calls use 30. Feature-head inner calls retain the main two-member
screen and final calls use 30. Centring and feature scaling use each member
gradient rows alone; unseen HLA offsets use that gradient global mean. Execution and
seed variability are not estimated by these conditional bootstrap intervals.
No conclusion about independent confirmation or an assay noise ceiling follows.
