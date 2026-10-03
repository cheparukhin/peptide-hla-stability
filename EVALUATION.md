# Evaluation contract (frozen at stage 1)

Agreed before any model existed. Nothing here may be changed after seeing
results. Implemented in `pepstab/evaluation.py`; run through
`scripts/evaluate.py`.

## The data contract

- **Shared example ID:** `pair_id`, the 0-based row position in the read-only
  `data/rasmussen_et_al_dataset.csv`. Every prediction file, embedding cache,
  structure manifest, and feature table keys on it.
- **Splits:** load `data/splits.csv` from disk via
  `pepstab.data.load_with_splits()`. Never recompute them.
- **Target:** train and predict `y_log1p = log1p(thalf_hours)`. Report raw hours
  alongside. 20.2% of labels sit at the assay floor, where `log1p` is defined
  and a log is not.

## Prediction file format

CSV with exactly two columns, one row per scored pair:

```
pair_id,y_pred
0,1.3842
1,0.0117
```

`y_pred` is on the `log1p` scale. One file per model per split.

## Metrics

**Primary — median per-allele Spearman ρ.** Spearman between predicted and
measured half-life within each allele, then the median across eligible alleles,
reported with the interquartile range and the per-allele table. This is the
headline because the claim is about ranking unseen peptides on alleles we
trained on. Ties take average ranks, which matters: the floor is a large tied
block, and a tie-blind correlation would flatter every model.

**Secondary**

- **MAE on `log1p` half-life** — numerical error. At the floor this is a lower
  bound on true error, because the recorded 0 is a censoring point, not a
  measurement.
- **precision@10 at a 2-hour threshold** — of each allele's top 10 predictions,
  how many really exceed 2 hours. Reported next to that allele's base rate and
  the reachable ceiling, since an allele with three positives cannot score above
  0.3. Ties are credited by expectation, not by row order: rows strictly above
  the 10th predicted value all count, and the remaining slots get the positive
  rate among the rows tied at that value. A constant predictor therefore scores
  exactly its base rate instead of whatever the first ten rows of the CSV happen
  to hold.
- **Pooled Spearman and Pearson** — secondary only. Pooling across alleles mixes
  within-allele ranking with between-allele offsets; always report the two
  separately.

## Eligible alleles

An allele enters the per-allele summary when it clears a row threshold on that
split and has at least two distinct label values. Computed from the split and
labels alone, never from predictions, so the eligible set is identical for every
model.

| Split | Threshold | Eligible alleles | Share of split rows |
|---|---:|---:|---:|
| test | ≥ 50 rows | 67 of 75 | 98.8% |
| val | ≥ 20 rows | 68 of 75 | 99.5% |

The threshold differs because the splits differ in size. Test carries 20% of the
data (median 72 rows per allele), so the final claim can rest on ≥ 50. Val
carries 10% (median 35), where a 50-row bar leaves only **10** eligible alleles
covering a quarter of the split — far too thin to select models on. At 20 rows
both splits land on ~68 alleles and >99% coverage, so selection and reporting
see effectively the same alleles while the headline number stays on the
better-powered ones.

The 8 alleles excluded from the test summary are listed in
`reports/audit_summary.md` §5; 7 of them are rare across the whole dataset.
Alleles whose ρ is undefined are reported as such, not silently dropped, and a
constant-within-allele predictor yields an **undefined** comparison rather than
an inconclusive one.

## Distance-stratified reporting (stage 6)

`splits.csv` carries `dist_to_train`: each peptide's Hamming distance to its
nearest training peptide. Held-out peptides sit at d=4 or d=5 by construction.

| Stratum | Test rows | Share |
|---|---:|---:|
| d=4 | 3,256 | 57.8% |
| d≥5 | 2,377 | 42.2% |

Two strata, not three. Only 6 test peptides (12 rows) reach d≥6, so a separate
far stratum cannot be scored.

Both strata are scored on the **same allele set** — the intersection of those
eligible in each, at a 20-row within-stratum bar (65 alleles on test). At the
split-level 50-row bar the strata would retain 17 and 7 alleles, and not the
same ones, so the gap would partly measure allele panels rather than distance.
On val the intersection is only 6 alleles, so this diagnostic is for the test
split.

If a model is exploiting residual similarity at the split boundary, d=4 should
score better than d≥5.

## Uncertainty

Paired **cluster bootstrap**: resample whole peptide clusters with replacement
and score both models on the same resample. Rows sharing a peptide cluster are
not independent — one peptide appears on up to 36 alleles — so a row bootstrap
would understate the interval. 2,000 resamples, seed `20261003`.

Report the difference and its 95% CI, not two separate intervals.

## Minimum worthwhile gain

**Δ median per-allele Spearman = 0.05.**

Derived from the frozen test set, not chosen by preference:

- Permuting labels within allele gives a median per-allele Spearman of
  0.000 ± 0.018 (sd), 95% range [−0.036, +0.035].
- A paired cluster bootstrap of the difference between two models whose true
  difference is zero has a 95% CI half-width of 0.037–0.053, depending on how
  correlated their errors are.

Below ≈0.05 this test set cannot separate a difference from zero, so a smaller
gain cannot be called meaningful whatever its point estimate.

Reading a comparison:

| Paired 95% CI for Δ | Verdict |
|---|---|
| lower bound > 0.05 | improvement, meets the predeclared bar |
| excludes 0, but upper bound < 0.05 | real but below the bar — report as such |
| crosses 0 | **inconclusive, not negative** |
| upper bound < 0.05 | strong negative: rules out a worthwhile gain |

## Model selection

- Select every checkpoint, layer, representation, head, and hyperparameter on
  **validation** data.
- **The test set is scored once, at stage 6.** One scoring run, all models at
  the same time.
- Report seed-to-seed variation for anything selected on validation. A gap
  smaller than the seed spread is not a result.
- Use a comparable tuning budget across arms, and record training and inference
  cost next to every accuracy number.
- When comparing against HLA-domain embeddings, include a baseline that gets the
  full domain sequence too, so "more input sequence" is not mistaken for a
  benefit of pretraining.

## Standing caveats

- The released NetMHCstabpan model was trained on this dataset, so it is not a
  held-out comparator.
- The assay panel was partly selected by predicted affinity, so peptide
  diversity is limited and broader biological claims need other evidence.
- No replicates are supplied, so no noise ceiling can be estimated from this
  file.
- `HLA-B*14:01(C67S)` and `HLA-B*14:02(C67S)` share a contact pseudosequence
  (756 rows), so pseudosequence-only models cannot separate them.
- Bound every conclusion to the representation, data, split, and budget tested.
