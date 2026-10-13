# Post-hoc R5 offset-only attribution diagnostic

This diagnostic was chosen **after** the original R5 20-label adaptation result
was observed to be positive (point +0.046754; original three-study 98.33% family
interval positive). It is exploratory. It is not a fourth primary study and
does not change or reallocate the original three-primary multiplicity family.
Its two intervals are nominal, unadjusted 95% intervals, conditional on saved
fits and the same three support draws; they omit selection history, refitting,
and independent assay variability.

Only R5 at k=20 is decomposed. Use all 75 existing source/query cells, the
original three support draws, and the identical fixed 68-allele assessment
panel. The original ≥20 and ≥50 panels are identical and are calculated once.
All predictions retain the original query-row identities and order. For each
cell and allele, select the unique adapters.json row with arm R5 and budget 20,
check its support/query identities, and read adapters.npz[key + '_offset'].
That scalar is the already fitted mean of its 20 support residuals. Define
R5_offset20 = R5_k0 + scalar on that cell/allele's query rows. Do not refit,
rescale, tune, inspect new labels, or use feature matrices or source weights.

Within each support draw, pool the five query peptide cells per allele, compute
each allele's Spearman correlation, and take the median across all 68 alleles.
Average the three draw scores. Never average predictions across support draws.
Estimate full R5_k20 minus R5_offset20 and R5_offset20 minus R5_k0. Their points
and every paired bootstrap difference must telescope to full R5_k20 minus
R5_k0. Compare ranks exactly within each cell and allele: adding a scalar must
leave those ranks unchanged. Pooled ranks may change because offsets differ
between source cells; a pooled gain cannot itself establish learned
peptide-dependent biology or prospective assay savings.

Use exactly 2,000 original crossed bootstrap draws, seed 20261003, with the full
canonical 3,587 peptide clusters and all 35 HLA groups. Match the RNG call order
of pepstab.followup3_fewshot.conditional_scores: for each bootstrap replicate,
sample peptide clusters first and HLA groups second. Reuse its weighted-rank
correlation and frequency-weighted median routines. Calculate R5_k0 and R5_k20
again and assert alignment to the saved original bootstrap columns before
using those saved columns for the paired contrasts. No new bootstrap modes,
alternative budgets, sensitivity searches, or alternative panels are allowed.

Before any new control metric is computed, the metadata-only freeze records
SHA-256 identities of this specification, diagnostic source/tests, original
source closure, original protocols and data inputs, manifest, original scores,
saved bootstrap files, prior audit and family result, and all 75 cell completion,
prediction, adapter metadata, and adapter-array files. The report checks those
identities and original protocol ancestry again. Actual scientific reporting
is root-authorized Modal/Linux only. Save query predictions, scalar offsets,
rank checks, per-draw and per-allele points, the offset bootstrap vector, paired
contrasts with nominal 95% intervals, and a hash-bound completion audit. No
original study source, protocol, result, or primary-family file is modified.
