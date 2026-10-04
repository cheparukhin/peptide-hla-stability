# ai_science_hack

Predicting peptide–HLA dissociation half-life (London AI × Science protein
engineering track, October 3–4, 2026).

**Read [HACKATHON_PLAN.md](HACKATHON_PLAN.md) first.** It is the source of truth
for scope, stage order, budget ceilings, and the stage 4c structural
pilot/production decision. Stage 4c's results, procedure and feature contract
are in [reports/stage4c_ectodomain_pilot.md](reports/stage4c_ectodomain_pilot.md).
The matched pilot ran both Boltz-2 and ESMFold2 on the same constructs,
prepared MSAs, arms and seeds; Boltz-2 passed its gate and ESMFold2 failed, so
production is Boltz-2 only. Keep this file consistent with the main plan; add
project decisions there, not here.

Data documentation lives in [docs/](docs/README.md): dataset stats, the C67S
exclusion, the frozen splits, the PDB structural overlap, and the public
affinity table used for augmentation.

## Invariants

- `data/rasmussen_et_al_dataset.csv` is read-only. Derived or corrected data
  goes to a new file; `(cd data && shasum -a 256 -c SHA256SUMS)` must still
  pass. The manifest stores a bare filename, so it only verifies from `data/`.
- Load the frozen splits from `data/splits.csv`, never recompute them —
  regenerating drops the peptide-cluster grouping and leaks training data into
  the test set. Join on `(allele, peptide)` and filter on the `split` column;
  never join on `pair_id`, which is positional into the raw CSV. See
  [EVALUATION.md](EVALUATION.md). `data/c67s_cleanup/peptide_splits.csv` is the
  superseded BLOSUM62 version — do not load it.
- Validation drives every decision. The test set is scored once, at stage 6.
- Augmented rows in `data/augmentation/` are **assumed** labels (`thalf_hours =
  0`), not measurements. They enter the fit set only, never validation or test,
  and never overwrite a measured value. Check any manifest with
  `pepstab.augment.verify_manifest` before training on it.
- Load the frozen production cohort from `data/structural_cohort.csv`, never
  recompute it. Each Modal profile folds **its own half**: the halves are
  disjoint, so folding the wrong one both wastes credit and leaves pairs
  unfolded. `--profile` must match `MODAL_PROFILE`; the runner asserts this.
- No batch GPU job without a passing end-to-end pilot on 3–5 examples. This
  applies to a new runner as well as a new model: `ectodomain_production.py`
  has its own `::smoke` entrypoint for exactly that.
