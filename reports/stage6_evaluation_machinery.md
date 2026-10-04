# Stage 6 evaluation machinery

Arm-agnostic analysis code for the final comparison, with every count the plan
states verified against the data and a worked example on **validation**.

**The test split is not scored here.** Every number below is validation or
training. Test is scored once, at stage 6, by the orchestrator's decision. The
test code path exists and is exercised on validation and train; it has not been
run on test.

| Artefact | What it is |
|---|---|
| `pepstab/stage6.py` | the analysis library |
| `scripts/stage6_report.py` | CLI: N prediction CSVs in, the whole table set out |
| `tests/test_stage6.py` | 33 guards (contract parity, scoring properties, leakage) |
| `reports/stage6_plan_verification.csv` | every plan claim, stated vs measured |
| `reports/stage6_val_*.csv`, `reports/stage6_val_manifest.json` | the worked example |

Nothing in `pepstab/evaluation.py`, `scripts/evaluate.py`, `EVALUATION.md`,
`HACKATHON_PLAN.md` or `CLAUDE.md` was modified. The frozen contract is reused,
not restated: `tests/test_stage6.py` asserts that the stage 6 distance
stratification and precision tables are **equal**, row for row, to the ones
`pepstab/evaluation.py` already produces.

## Interface

```
.venv/bin/python scripts/stage6_report.py --split val \
    seq_baseline=preds/seq_baseline.csv \
    seq_ensemble_pep_pseudo=preds/seq_ensemble_pep_pseudo.csv \
    allele_mean=preds/allele_mean.csv \
    global_mean=preds/global_mean.csv \
    --n-boot 2000 --stratum-min-rows 10 --nested
```

Each positional argument is a prediction CSV in the frozen two-column format
(`pair_id,y_pred`, `y_pred` on the `log1p` scale), optionally prefixed with
`NAME=`. The first arm is the baseline every paired CI is taken against. A file
may cover more rows than the split; the extras are ignored. **Nothing is
hard-coded about the sequence baseline** — an ESM-2 or structural prediction
file drops straight in.

Joins are on `pair_id` as written by `pepstab.data.load_with_splits()`, which
reads `data/splits.csv`. Splits are never recomputed, and
`data/c67s_cleanup/peptide_splits.csv` is never opened — a test greps both
source files to keep it that way.

## Verification of the plan's counts

Full table in `reports/stage6_plan_verification.csv`. Every figure the plan
states for the **test** split is correct:

| Claim | Source | Stated | Measured |
|---|---|---:|---:|
| test d=4 rows | plan §6, EVALUATION.md | 3,256 | **3,256** (57.80%) |
| test d>=5 rows | plan §6, EVALUATION.md | 2,377 | **2,377** (42.20%) |
| test rows at d>=6 | plan §6, EVALUATION.md | 12 | **12** |
| test peptides at d>=6 | plan §6 | 6 | **6** |
| shared stratum panel, 20-row bar | plan §6, EVALUATION.md | 65 | **65** |
| d=4 eligible at the 50-row bar | EVALUATION.md | 17 | **17** |
| d>=5 eligible at the 50-row bar | EVALUATION.md | 7 | **7** |
| peptides on >= 2 alleles | plan §6 | 3,941 | **3,941** |
| rows they cover | plan §6 | 26,474 | **26,474** (93.99%) |
| most alleles on one peptide | plan §6 | 36 | **36** |

Two additions the plan does not state, both checked here:

- At the 50-row bar the two test strata share only **6** alleles, so the plan's
  "17 and 7 — different ones" is right, and the 20-row bar is doing real work.
- No Hamming <= 2 peptide group straddles a frozen split (**0 of 5,410**),
  which is what lets the nested evaluation stay inside training. The frozen
  clustering radius is 3, so this follows by construction — but it is checked,
  not assumed, by `stage6.verify_groups_within_split()`.

### Where the data disagreed with an assumption

**The 20-row stratum bar does not transfer to validation.** The constant
`MIN_ROWS_PER_ALLELE_IN_STRATUM = 20` was derived from the test split, where it
yields a 65-allele shared panel covering ~96% of each stratum's rows. Validation
is half the size, so each stratum holds ~1.4k rows instead of ~2.8k:

| Split | Bar | d=4 eligible | d>=5 eligible | Shared | Rows covered (d=4 / d>=5) |
|---|---:|---:|---:|---:|---|
| test | 20 | 67 | 66 | **65** | 95.9% / 96.7% |
| val | 20 | 45 | 11 | **6** | 15.2% / 21.7% |
| val | 10 | 68 | 55 | **55** | 79.6% / 91.3% |

The frozen constant is unchanged and **test keeps 20**. `scripts/stage6_report.py`
exposes `--stratum-min-rows` and prints a warning whenever the shared panel
covers under half of a stratum's rows. The validation worked example below uses
10.

> That bar of 10 was chosen from **row counts and panel coverage alone, before
> any model was scored in a stratum**. The table above is the entire basis for
> it. A stratum bar picked after seeing performance would be a selection effect
> that no reader could detect from the resulting number, so the provenance is
> recorded here rather than left implicit.

**The plan's 57.8% / 42.2% split is test-specific.** Validation sits at d=4
1,730 rows (61.4%) and d>=5 1,087 (38.6%), with 15 rows at d=6 across 3
peptides. Nothing is wrong in the plan — it says "test rows" — but the figures
are not a property of the dataset and should not be reused as one.

**One validation allele has a precision@10 ceiling of 0.** `HLA-B*08:01` (55
validation rows) has no peptide above the 2-hour threshold, so every model
scores 0 there and `ceiling_share` is undefined. `precision_summary()` reports
`n_alleles_no_positives` beside `n_alleles_at_ceiling` because such an allele is
counted in the latter: with nothing reachable, 0 *is* the ceiling, and leaving
it unlabelled would let an impossible allele read as a success.

---

## 1. Nearest-neighbour distance stratification

**Definition.** `dist_to_train` is read from the frozen `data/splits.csv`: the
Hamming distance from each held-out peptide to its nearest *training* peptide.
Two strata, `d=4` and `d>=5`, which partition the split (asserted in the tests).
A third `d>=6` stratum is not viable: 12 test rows over 6 peptides, 15
validation rows over 3.

Both strata are scored on **one shared allele panel** — the intersection of the
alleles eligible within each stratum at the stratum row bar. Without the
intersection the two numbers would be computed over different allele panels and
their difference would partly measure panel composition rather than distance.
The panel is derived from the split and the labels only, so it is identical for
every arm.

`stage6.score_strata()` at the frozen 20-row bar is asserted equal to
`evaluation.score_by_distance()`.

**Worked example, validation, 55-allele shared panel at a 10-row bar:**

<!--STRATA-->

Both model arms lose ground moving away from training, by a similar amount, and
the two controls are flat at 0 by construction. A gap of this shape is the
expected consequence of the residual similarity stage 1 measured (same-allele
label Spearman 0.592 at d=4 versus 0.302 for unrelated pairs); it is evidence
that the stratification is sensitive, not that a model is cheating. What would
matter at stage 6 is a gap that is *larger for one arm than another*.

## 2. The differential target

**Definition, written before any number was computed** (`differential_pairs()`
docstring is the authority):

- *Which pairs.* For each peptide measured on two or more **eligible** alleles
  within the scored split, every unordered pair of those alleles, canonicalised
  to `allele_a < allele_b`. A peptide on `k` alleles contributes `k(k-1)/2`
  comparisons; the test asserts the total equals `sum k(k-1)/2`.
- *The quantity.* `delta_true = y_log1p(b) - y_log1p(a)`, and `delta_pred`
  likewise. Both rows carry the same peptide, so everything intrinsic to the
  peptide cancels exactly — including any peptide-only term the model learned.
  A test confirms this: adding an arbitrary per-peptide constant to a
  prediction vector leaves every differential metric unchanged.
- *Ties.* `delta_true == 0` means the two alleles hold the peptide equally long
  (usually both at the assay floor). No prediction can be right or wrong about
  the direction, so the pair is **undecidable**: excluded from concordance,
  reported as a count. A *predicted* tie on a decidable pair is credited 0.5,
  the same tie-by-expectation rule `precision_at_k` uses, which is what makes a
  constant predictor score exactly 0.500.
- *Aggregation,* three ways, because they answer different questions:
  - **concordance** — pair-level fraction of decidable comparisons whose
    direction the model gets right. A peptide on 36 alleles carries 630 of them.
  - **concordance_peptide_weighted** — each peptide's comparisons average to one
    number first, so every peptide counts once regardless of its allele count.
  - **median_allele_pair_spearman** — for each allele pair sharing at least 10
    peptides, Spearman of `delta_true` against `delta_pred` across those
    peptides; median over allele pairs. An allele pair a model ranks constantly
    scores 0, the predeclared `UNRANKED_CONTRIBUTION`, so a model cannot raise
    its median by going flat on the hard pairs.

**Verified scale.** On the whole dataset: 3,941 peptides on >= 2 alleles,
26,474 rows, 93.99%, up to 36 alleles on one peptide — the plan's figures
exactly. Restricted to validation's 68 eligible alleles: 394 peptides, 10,365
comparisons over 1,372 distinct allele pairs, of which 755 are undecidable and
268 allele pairs clear the 10-peptide bar.

**Worked example, validation:**

<!--DIFFERENTIAL-->

The two controls are the reason to trust the metric:

- `global_mean` is constant, and scores **exactly 0.500** concordance — chance,
  as the tie rule guarantees.
- `allele_mean` predicts each allele's mean and nothing else. It scores **0.698**
  concordance, because knowing which groove is stickier on average does get the
  direction right most of the time — but **0.000** on
  `median_allele_pair_spearman`, because within a fixed allele pair its
  predicted delta is the same for every peptide. That second column is therefore
  the one that isolates peptide-specific groove chemistry, and it is the one to
  lead with when comparing arms.

Cost: zero new compute. This is a re-aggregation of predictions already made.

## 3. Paired uncertainty

**The resampling unit is the peptide cluster** (`cluster_id` from
`data/splits.csv`), not the row and not the peptide. One peptide is measured on
up to 36 alleles, and peptides within a cluster are within 3 substitutions of
each other, so neither rows nor peptides are independent. Whole clusters are
drawn with replacement, `len(clusters)` of them, and all of a drawn cluster's
rows enter the resample together — asserted in `test_paired_bootstrap_keeps_clusters_whole`.
Both arms are scored on the *same* resample, so the shared sampling noise
cancels instead of being counted twice. 2,000 resamples, seed `20261003`.

`evaluation.paired_cluster_bootstrap()` remains the contract implementation for
the primary metric. `stage6.paired_cluster_delta()` is the same procedure, same
seed, generalised to any statistic, so the differential concordance, the
precision median and the mutant concordance all carry paired intervals too.

For the differential and mutant tables the bootstrap resamples **comparison
rows** grouped by `cluster_id`. That is still the right unit: a mutant pair lies
wholly inside one cluster, and a differential pair shares one peptide and so one
cluster, so no comparison is ever torn in half by a resample.

**Why clusters, measured rather than asserted.** `resampling_unit_widths()`
re-runs the identical paired comparison under three units:

<!--UNITS-->

The row bootstrap's interval is materially narrower. That is not a better
interval — it is one that counted rows sharing a peptide as independent
evidence. Reporting the measured ratio is the evidence for the contract's
choice.

**Worked example, validation, 2,000 resamples:**

<!--CI-->

### Mean as well as median

The contract's primary statistic is the **median** per-allele Spearman, and the
six-verdict rule in `EVALUATION.md` applies to it. The CLI now reports the
**mean** over the same panel beside it, for one reason: `reports/stage2_baselines.md`
quotes ensembling gains as *mean* SCC, and a stage 6 median delta cannot be
checked against a stage 2 mean delta. Emitting both removes a comparison that
otherwise looks like a contradiction.

**Worked example: the ensemble parity comparison, validation, 2,000 resamples**
(`reports/stage6_val_ensemble_parity.csv`, which keeps the confounded row rather
than deleting it):

| Baseline | Model | Statistic | Delta | 95% CI | Pairing |
|---|---|---|---:|---|---|
| `seq_baseline` | `seq_ensemble_pep_pseudo` | median | **+0.0833** | [+0.0288, +0.1235] | matched |
| `seq_baseline` | `seq_ensemble_pep_pseudo` | mean | **+0.0740** | [+0.0470, +0.1036] | matched |
| `mlp_onehot_pep_domain` | `seq_ensemble_pep_domain` | median | +0.0593 | [+0.0185, +0.1183] | matched |
| `mlp_onehot_pep_domain` | `seq_ensemble_pep_domain` | mean | +0.0643 | [+0.0387, +0.0878] | matched |
| `seq_baseline` | `seq_ensemble_pep_domain` | median | +0.0432 | [−0.0106, +0.0911] | **confounded** |
| `seq_baseline` | `seq_ensemble_pep_domain` | mean | +0.0392 | [+0.0096, +0.0697] | **confounded** |

**Neither interval establishes that the gain clears the 0.05 bar.** Both point
estimates sit above it, the median CI runs from +0.029 and the mean CI from
+0.047 — that lower bound is *below* 0.05, by 0.003. The predeclared verdict is
"real improvement, but whether it clears the 0.05 bar is unresolved". Writing
"clears the bar" here would be the same error we reject on every other arm, and
it does not become acceptable because the number is flattering.

The confounded row is kept deliberately. It is the comparison this workstream
got wrong at first — `preds/seq_baseline.csv` is the pseudosequence single
network, so pairing it against the *domain* ensemble changes the input arm and
the ensembling together. Note that the error did not flatter the result: the
confounded median delta is +0.0432 with an interval that crosses zero, against
+0.0833 excluding zero for the matched pair. Mixing the two changes understated
the effect and turned a conclusive result into an inconclusive one.

Two cautions this surfaced, both recorded for the orchestrator rather than
edited into files this workstream does not own:

1. **`preds/seq_baseline.csv` is the single network on the pseudosequence arm**
   (byte-identical to `preds/mlp_onehot_pep_pseudo.csv`). Comparing it against
   `preds/seq_ensemble_pep_domain.csv` changes the arm *and* the ensembling
   together. The matched partner is `preds/seq_ensemble_pep_pseudo.csv`. The
   `note` column of `reports/stage6_val_ensemble_parity.csv` marks which rows
   are matched and which are confounded.
2. **The "+0.090 mean SCC from ensembling" figure is against the mean of the
   ensemble's own 30 members, not against the single-network baseline.** From
   `reports/stage2_ensemble_pep_pseudo.csv`, the members average 0.5550 and the
   ensemble reaches 0.6452: +0.0902. Against the deployed single network
   (0.5712) the gain is **+0.0740**; on the domain arm the corresponding figures
   are +0.0998 and +0.0643. Both quantities are real and the arithmetic is
   right, but they answer different questions, and each ensemble member fits on
   15,773 rows against the single network's 17,744 — so the
   ensemble-minus-mean-member gap bundles a training-rows difference in with
   averaging. The ensemble-minus-single-network gap does not. Either way the
   gain clears the 0.05 bar and the ensemble-parity rule stands.

## 4. Precision@10 at the predeclared 2-hour threshold

**Reused, not reimplemented.** `evaluation.precision_at_k()` already implements
the per-allele precision, the base rate, the reachable ceiling and the
tie-by-expectation rule; `stage6.precision_report()` calls it unchanged and a
test asserts the three shared columns are identical. Two columns are added:

- **`lift`** — precision@10 minus the allele's base rate. Zero means the top 10
  are no better than a random 10.
- **`ceiling_share`** — precision@10 over the best precision@10 reachable. An
  allele with 4 positives in 60 rows caps at 0.4, so 0.4 there is perfect and
  0.4 elsewhere is not. Left NaN where the ceiling is 0.

Ties are credited by expectation: rows strictly above the 10th prediction value
all count, and the remaining slots are credited the positive rate among the rows
tied at that value. `test_constant_predictor_scores_its_base_rate` asserts the
property this buys — a constant predictor scores its base rate on **every**
allele, to within 1e-12, so `median_lift` is exactly 0.

**Worked example, validation:**

<!--PRECISION-->

Both controls land exactly on the base rate, as designed. `allele_mean` is
constant *within* an allele, so it is indistinguishable from `global_mean` on
this metric — a useful reminder that precision@10 is a within-allele measure and
carries no cross-allele information at all.

## 5. Nested near-neighbour evaluation inside training

The grouped split places every pair of peptides within 3 substitutions on the
same side, so it cannot ask whether a model ranks point mutants of a known
binder. This recovers that question without touching the test set or the frozen
assignments.

**Design.**

- *Groups.* Single-linkage peptide groups at Hamming <= 2, built **inside** the
  training split. Because the frozen split clusters at Hamming <= 3 and 2 is
  strictly smaller, every such group is a subset of one frozen cluster and so of
  one split. Verified over the whole dataset: 0 of 5,410 groups straddle a
  split. `near_neighbour_groups()` refuses a radius >= 3.
- *The comparison unit.* A **mutant pair**: two peptides at Hamming 1 or 2, both
  measured on the **same** allele. `delta_true` is the difference in their
  labels on that allele. Same tie rules as the differential target.
- *Folds.* 5 folds, assigned by **peptide**, balanced by row count,
  deterministic (seed `20261004`). The peptide is the fold unit on purpose: a
  mutant pair must be *split across* folds so the model sees one member while
  predicting the other. Folding by group would hold out whole mutant families
  and reproduce the grouped split's blind spot at smaller scale. All rows of a
  held-out peptide leave together across every allele, so no allele can leak the
  peptide's label.
- *Scoring.* Only pairs whose two rows sit in **different** folds are scored:
  if both were held out the model saw neither, and if both were in training
  neither prediction is out-of-sample. Both counts are reported.
- *Plug-in.* Any arm supplies its own out-of-fold prediction series on the same
  folds. `ridge_oof_predictions()` is a CPU reference arm (BLOSUM62,
  peptide+pseudosequence, ridge) so the harness could be validated end to end
  without waiting on another workstream.

### Finding: this question is underpowered in this dataset

**The nested evaluation is the right design for the question the grouped split
cannot answer, and this dataset does not contain enough independent peptide
clusters to answer it.** That is a dataset limitation, not a method failure,
and it is worth saying plainly because the instinct on reading it is that we
should have tried harder. There is nothing to try harder at: the radius cannot
exceed 2 without the groups straddling a split, and at radius 2 the training
split contains every mutant comparison that exists.

This is the result, not a caveat on one. The plan expects the nested evaluation
to "recover the question the grouped split cannot answer". It recovers the
question; it does not deliver enough evidence to answer it in absolute terms.

| Quantity | Train | Val |
|---|---:|---:|
| same-allele mutant comparisons (Hamming 1–2) | **608** | 96 |
| of which decidable (labels differ) | 505 | 82 |
| at Hamming 1 / Hamming 2 | 342 / 266 | 54 / 42 |
| alleles involved | 68 | 41 |
| **independent peptide clusters behind them** | **115** | 15 |
| scored after the cross-fold filter | 490 (402 decidable) | — |
| independent clusters after that filter | **96** | — |

608 comparisons sound like enough. 96 independent clusters are not: that is the
number the bootstrap actually resamples. Measured minimum detectable effects, at
2,000 resamples:

| Question | 95% CI half-width on concordance |
|---|---:|
| is this arm above the 0.5 chance floor? | **0.069** |
| does arm B rank mutants better than arm A? (two genuinely different arms) | **0.055** |
| ...for two near-identical arms | 0.011 |

So the nested evaluation can detect a jump from 0.50 to about 0.57 concordance,
and nothing smaller. **Report it as a bounded negative, or as a comparison
between arms, never as an absolute claim about mutant ranking.**

**Worked example, validation run, ridge reference arm:**

<!--NESTED-->

Read this as a property of the harness and of a weak reference arm together, not
as a result about mutant ranking. The same ridge out-of-fold predictions score a
median per-allele Spearman of 0.251 within training — far below the 0.693 the
30-network ensemble reaches on validation — so this arm was never going to
resolve a 0.07 effect. The alpha sweep (0.1, 1, 10, 100) moves mutant
concordance only between 0.488 and 0.527, so the reading is not an artifact of
the regularisation strength.

**What to do with it at stage 6.** Pass `--nested` with `--nested-arm
NAME=oof.csv` **twice** and report the paired arm-vs-arm row
(`*_nested_mutant_paired.csv`), with the 0.055 half-width quoted beside every
number. The per-arm "concordance minus chance" column is emitted too, but it is
the weaker question and must never be stated as an absolute claim about mutant
ranking. If no arm clears 0.57, the honest statement is: *on the 490 mutant
comparisons this dataset supports, none of the arms tested ranks point mutants
measurably better than chance, and the comparison cannot resolve differences
below about 0.055 concordance.*

---

## Provenance, and arms that have not arrived yet

Every delta here is a statement about two specific prediction **files**, and
arms get regenerated — a retuned baseline rewrites its ensemble's predictions
and silently invalidates every comparison taken against the old one. So the
manifest records a SHA-256, a byte count and an mtime for each prediction file
it read, under `arms`. A stale comparison is then detectable from the artefact
rather than only by remembering.

The numbers in this document are tied to the digests in
`reports/stage6_val_manifest.json`. If `preds/seq_ensemble_pep_pseudo.csv`
changes, the CLI re-runs in about twelve minutes and every table regenerates;
nothing here is hand-maintained.

The CLI is already general over arms: the first positional argument is the
baseline every paired interval is taken against, so the headline comparison for
an additive arm is

```
scripts/stage6_report.py --split test \
    seq_ensemble_pep_pseudo=preds/seq_ensemble_pep_pseudo.csv \
    esm_ensemble=preds/esm_ensemble.csv \
    esm_plus_seq=preds/esm_plus_seq_ensemble.csv \
    boltz=preds/boltz_structural.csv --nested ...
```

and `--nested-arm` is repeatable so the nested analysis takes a paired
arm-vs-arm interval between the first two out-of-fold files given.

## Guards

`tests/test_stage6.py`, 35 tests, all on validation and train. Grouped by what
they protect:

- **Contract parity.** `score_strata` equals `evaluation.score_by_distance` at
  the frozen bar; `precision_report` equals `evaluation.precision_at_k` on the
  three shared columns; `stage6.read_predictions` returns the identical array to
  `scripts/evaluate.py`'s loader.
- **Scoring properties.** A constant predictor scores 0.500 concordance on the
  differential, 0.500 on mutant ranking, and its base rate on precision@10 for
  every allele; the labels themselves score 1.000 on mutant ranking (a check on
  the sign convention); an allele-only predictor scores 0.000 on
  `median_allele_pair_spearman`; a per-peptide shift leaves every differential
  metric unchanged; precision never exceeds its ceiling.
- **Leakage and joins.** The loaded split equals `data/splits.csv` exactly;
  neither source file mentions `peptide_splits` or `c67s_cleanup`; strata
  partition the split; folds hold out whole peptides and still split most mutant
  pairs apart; no Hamming <= 2 group straddles a split; the bootstrap keeps
  clusters whole and is deterministic; non-finite and incomplete prediction
  files are rejected.
- **CLI.** Every table is emitted; duplicate arm names are refused; the
  manifest carries a SHA-256 of each prediction file; two `--nested-arm` files
  produce a paired arm-vs-arm row, and an arm fed the labels themselves scores
  a perfect 1.000 mutant concordance.

## Limitations

- Everything here is validation and training. Nothing has been tuned on test and
  nothing has been measured on test.
- The differential target's pair-level concordance is dominated by
  high-coverage peptides (36 alleles gives 630 comparisons from one peptide).
  The peptide-weighted column is reported for exactly that reason; where they
  disagree, say which one the claim rests on.
- Undecidable comparisons are a consequence of the 20.2% assay floor, not noise.
  755 of 10,365 on validation. They are excluded from concordance and counted,
  never imputed.
- The nested evaluation's reference arm is a ridge regression, chosen for being
  cheap and CPU-only. It is not a statement about what a strong arm would score.
- The paired bootstrap assumes peptide clusters are exchangeable. They are not
  identically distributed — cluster sizes vary by two orders of magnitude — so
  the interval is an approximation, as every cluster bootstrap is.
