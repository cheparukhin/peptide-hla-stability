# Evaluation contract (frozen at stage 1)

Locked before any model existed. Nothing here changes after seeing results.
Code: `pepstab/evaluation.py`; CLI: `scripts/evaluate.py`.

## Data contract

- **Row key:** `pair_id`, the 0-based row index in `data/rasmussen_et_al_dataset.csv`.
- **Splits:** `pepstab.data.load_with_splits()`. Never recompute.
- **Target:** `y_log1p = log1p(thalf_hours)`. Report raw hours alongside.
  20.2% of labels sit at the assay floor (zero); `log1p` is defined there, `log` is not.

## Prediction file format

CSV, two columns, one row per scored pair. `y_pred` on the `log1p` scale.

```
pair_id,y_pred
0,1.3842
1,0.0117
```

## Metrics

**Primary — median per-allele Spearman rho.** Rank-correlate predicted vs
measured half-life within each HLA allele, then take the median across eligible
alleles. Report IQR and per-allele table. Ties get average ranks — the floor
creates a large tied block and a tie-blind correlation would flatter every model.

**Secondary**

- **MAE on `log1p` half-life.** Error against *recorded* labels. At the floor
  the recorded 0 is a detection limit, not a measurement, so floor MAE is
  biased in both directions and is neither an upper nor a lower bound on true
  error. (Details: a latent 0.05 h recorded as 0 — predicting 0.10 overcharges
  by 2x, predicting 0.01 undercharges by 4x.)

- **Precision@10 at 2 hours.** Of each allele's top 10 predictions, how many
  truly exceed 2 h? Reported with base rate and reachable ceiling. Ties credited
  by expectation, not row order, so a constant predictor scores its base rate.

- **Pooled Spearman and Pearson** — secondary only. Pooling mixes within-allele
  ranking with between-allele offsets; report them separately.

## Eligible alleles

An allele needs enough rows on the split and at least two distinct labels.
Computed from splits and labels alone, never from predictions — same set for
every model.

| Split | Row threshold | Eligible | Coverage |
|---|---:|---:|---:|
| test | >= 50 | 67 / 75 | 98.8% |
| val | >= 20 | 68 / 75 | 99.5% |

Thresholds differ because splits differ in size. Test is 20% of data (median 72
rows/allele); val is 10% (median 35). A 50-row bar on val leaves 10 alleles
covering a quarter of the split — useless for model selection. The 8 excluded
test alleles are in `reports/audit_summary.md` §5; 7 are rare in the dataset.

### Constant predictions

Spearman is undefined when predictions are constant. Scoring rule, declared in
advance:

> Constant prediction on an eligible allele scores **0**.

| Behaviour | Score |
|---|---:|
| Ranks backwards | < 0 |
| Constant (no ranking info) | 0 |
| Ranks with skill | > 0 |

The allele stays in the panel — dropping it would let a model raise its median
by going constant on hard alleles. The per-allele table shows the raw `NaN` in
`spearman` alongside the scored 0 and an `unranked` flag.

Adjacent cases:

| Case | Treatment |
|---|---|
| All labels identical | Degenerate — leaves the panel (untestable) |
| NaN / inf predictions | Rejected before scoring (`ValueError`) |

`validate_finite()` enforces rejection in `score()`, `score_by_distance()`, and
`paired_cluster_bootstrap()`. `+inf` inflates Spearman; `NaN` would be silently
absorbed as a 0. Both are checked before resampling.

Inside bootstrap resamples, an allele can end up with one distinct label. It
leaves that resample's panel (scoring it 0 would confuse "untestable draw" with
"model failed"). The panel rebuilds per resample; both models always share the
same alleles.

## Distance-stratified reporting (stage 6)

`splits.csv` has `dist_to_train`: Hamming distance from each peptide to its
nearest training peptide. Held-out peptides sit at d=4 or d=5 by construction.

| Stratum | Test rows | Share |
|---|---:|---:|
| d=4 | 3,256 | 57.8% |
| d>=5 | 2,377 | 42.2% |

Only 12 rows at d>=6, so two strata, not three. Both scored on the same 65
alleles (intersection of those eligible within each stratum at a 20-row bar).
Using the split-level 50-row bar would leave 17 and 7 alleles — different
ones — and the gap would partly measure panel composition rather than distance.

If a model exploits residual similarity at the split boundary, d=4 will score
higher than d>=5.

## Uncertainty

Paired cluster bootstrap: resample whole peptide clusters with replacement, score
both models on the same resample. Peptide clusters are the independence unit —
one peptide appears on up to 36 alleles, so a row bootstrap understates
uncertainty. 2,000 resamples, seed `20261003`. Report the difference and its
95% CI, not two separate intervals.

## Minimum worthwhile gain

**Delta median per-allele Spearman = 0.05**, derived from the frozen test set:

- Shuffled labels: median rho = 0.000 +/- 0.018, 95% range [-0.036, +0.035].
- Paired cluster bootstrap under no true difference: 95% CI half-width
  0.037–0.053.

Below ~0.05 this test set cannot separate a difference from noise.

### Reading a comparison

Six mutually exclusive verdicts, implemented in `describe_delta()`:

| Paired 95% CI for Delta | Verdict |
|---|---|
| upper < 0 | Worse |
| lower > 0.05 | Meets the bar |
| Excludes 0, upper < 0.05 | Real but too small to matter |
| Excludes 0, straddles 0.05 | Real, size unresolved |
| Crosses 0, upper < 0.05 | Inconclusive, rules out a worthwhile gain |
| Crosses 0, upper >= 0.05 | Inconclusive |

Row 4 is the tricky one: the improvement is real, its magnitude is not
established. "Below the bar" would be wrong — the interval doesn't support that.

## Disclosed test exposure

One diagnostic breached the single-use test rule. Recorded here rather than
quietly fixed, because the audit trail is what makes "scored once" meaningful.

**What happened.** The first version of `scripts/compare_to_paper.py`
(commit 24dfaa6) estimated how much Rasmussen et al.'s looser peptide grouping
flattered their published score. It re-partitioned the **whole dataset** 70/10/20
by peptide identity, which pulled frozen test rows into that experiment:

| Frozen test rows | Used for |
|---:|---|
| 3,350 | fitting a diagnostic model |
| 448 | early stopping that model |
| 585 | the score that model was judged on |

**What was observed.** One aggregate statistic — mean per-allele Spearman 0.591
— computed over 2,817 rows of which 585 were frozen test rows, mixed with
val and train rows. No per-row test prediction, no per-allele test score, and no
test label was inspected individually.

**What it influenced.** A reported +0.018 attribution for split grouping, which
fed the narrative that our split was not the reason for the gap, and partly
motivated building the ensemble baseline. That attribution has since been
measured properly and is **−0.005 [−0.035, +0.046]**, i.e. inconclusive — so the
leaked experiment's conclusion was also wrong.

**What was not affected.** No model that will be scored at stage 6 saw a test
row. `scripts/baseline_sequence.py` and `scripts/baseline_ensemble.py` both fit
on `split == "train"` only; every prediction file in `preds/` covers 2,817
validation rows.

**Assessment.** The exposure is one aggregate number over a 21% test admixture,
used for a diagnostic whose conclusion was then reversed. We judge the test split
still usable and continue to score it once at stage 6. A reader who disagrees has
the numbers above to discount with.

**Fix.** `scripts/compare_to_paper.py` now runs the grouping experiment entirely
inside `split == "train"`, on a common evaluation set, with a paired CI.

## Model selection

- Select on **validation**. Test is scored once, at stage 6, all models together.
- Report seed-to-seed variation. A gap smaller than seed spread is not a result.
- Use comparable tuning budgets across arms. Record cost next to accuracy.
- When comparing against HLA-domain embeddings, include a baseline with the full
  domain sequence, so "more input" is not mistaken for pretraining benefit.

## Caveats

- NetMHCstabpan was trained on this dataset — not a held-out comparator.
- **The published 0.676 PCC is not comparable.** NetMHCstabpan's reported Pearson
  correlation was computed on data padded with ~1,000 synthetic 0 h peptides per
  allele (~75,000 easy negatives on top of ~28,000 measured rows). That inflates
  correlation substantially. Our numbers on measured rows alone will be lower;
  never quote 0.676 as the bar to beat.
- **Early stopping.** Rasmussen et al. used the held-out 1/5 fold as both test
  and early-stopping set, which is mildly optimistic. We use a separate inner
  validation split. This partly explains why our numbers sit lower.
- The assay panel was partly selected by predicted binding affinity, limiting
  peptide diversity.
- No replicates, so no noise ceiling from this file alone.
- `HLA-B*14:01(C67S)` and `HLA-B*14:02(C67S)` share a pseudosequence (756 rows);
  pseudosequence-only models cannot separate them.
- Conclusions are bounded by the representation, data, split, and budget tested.
