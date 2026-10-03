# ai_science_hack

Predicting peptide–HLA dissociation half-life (London AI × Science protein
engineering track, October 3–4, 2026).

**Read [HACKATHON_PLAN.md](HACKATHON_PLAN.md) first.** It is the source of truth
for scope, stage order, budget ceilings, and the hour-5 structural
go/reduce/stop decision. Add detail there, not here.

## Invariants

- `data/rasmussen_et_al_dataset.csv` is read-only. Derived or corrected data
  goes to a new file; `(cd data && shasum -a 256 -c SHA256SUMS)` must still
  pass. The manifest stores a bare filename, so it only verifies from `data/`.
- Load the frozen splits from disk, never recompute them. Regenerating drops the
  peptide-cluster grouping and leaks training data into the test set.
- Validation drives every decision. The test set is scored once, at stage 6.
- No batch GPU job without a passing end-to-end pilot on 3–5 examples.
