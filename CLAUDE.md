# ai_science_hack

Predicting peptide–HLA dissociation half-life (London AI × Science protein
engineering track, October 3–4, 2026).

**Read [HACKATHON_PLAN.md](HACKATHON_PLAN.md) first.** It is the source of truth
for scope, stage order, budget ceilings, and the hour-5 structural
go/reduce/stop decision. Add detail there, not here.

Data documentation lives in [docs/](docs/README.md): dataset stats, the C67S
exclusion, the frozen splits, the PDB structural overlap, and the public
affinity table used for augmentation.

## Invariants

- `data/rasmussen_et_al_dataset.csv` is read-only. Derived or corrected data
  goes to a new file; `(cd data && shasum -a 256 -c SHA256SUMS)` must still
  pass. The manifest stores a bare filename, so it only verifies from `data/`.
- Load the frozen splits from `data/c67s_cleanup/peptide_splits.csv`, never
  recompute them — regenerating drops the peptide-cluster grouping and leaks
  training data into the test set. Join on `(allele, peptide)` and filter on
  the `split` column; never join on `row_id`, which is positional into the raw
  CSV. See [docs/SPLITS.md](docs/SPLITS.md).
- Validation drives every decision. The test set is scored once, at stage 6.
- No batch GPU job without a passing end-to-end pilot on 3–5 examples.
