# Test-scoring runbook

Written 4 October 2026 at 04:55 BST, **before any test number existed**, so the
procedure cannot be shaped by the result. The test split is scored **once**.
This file is the whole procedure; follow it mechanically.

The single-pass rule it implements is predeclared in
[HACKATHON_PLAN.md](../HACKATHON_PLAN.md), stage 6. The metric definitions are
frozen in [EVALUATION.md](../EVALUATION.md) and are not reopened here.

## 1. Preconditions — all must hold before the command is typed

- [ ] Every arm entering the pass has a **validation-selected configuration
      frozen**, recorded in its stage report, with the selection made on
      validation only.
- [ ] Every arm's selected hyperparameters are **interior to their ladders**,
      not at a grid edge, and the ladders are recorded. This applies to the
      sequence baseline as much as to any foundation-model arm — see the
      symmetry note in `reports/stage2_baselines.md`.
- [ ] Arms being compared share **ensemble size, seed protocol, tuning budget,
      and rows**. An unmatched arm may enter the pass but is reported
      separately and labelled not comparable.
- [ ] Each arm's prediction file is in the frozen two-column format
      (`pair_id,y_pred`, `y_pred` on the `log1p` scale) and covers the test
      rows.
- [ ] `(cd data && shasum -a 256 -c SHA256SUMS)` passes.
- [ ] The full test suite passes.
- [ ] Any arm **not** frozen by the cutoff is excluded and will be reported on
      validation only, with that stated in the write-up. The cutoff does not
      move for an arm that is nearly ready.

## 2. Record provenance BEFORE running

Capture, into `reports/test_scoring_provenance.json`:

- the git commit SHA of the working tree,
- `shasum -a 256` of every prediction file entering the pass,
- `shasum -a 256` of `data/splits.csv` and `data/rasmussen_et_al_dataset.csv`,
- the arm list with each arm's frozen config and its stage report,
- the UTC and Europe/London timestamp.

This is what makes the claim "selected on validation, scored once on test"
checkable by someone who was not here, rather than something we assert.

## 3. The command

```bash
.venv/bin/python scripts/stage6_report.py --split test --n-boot 2000 \
  --stratum-min-rows 20 \
  <arm_name>=<preds/file.csv> [...] \
  --out-dir reports --prefix stage6_test
```

`--split test` is gated in that script and must be passed exactly once.

The **first positional arm is the baseline** every paired interval is taken
against. `--stratum-min-rows 20` is the frozen default; the 10 used elsewhere
is validation-only, because validation is half the size.

(Flag spelling corrected 4 October 2026 when the pass was run: the arms are
positional and the outputs are `--out-dir`/`--prefix`, not `--arms`/`--out` as
first written. The **procedure** is unchanged — this is a typo fix against
`scripts/stage6_report.py`'s actual interface, made before any test number
existed, not a reopened decision.)

## 4. What is reported

Per `EVALUATION.md`: median per-allele Spearman with IQR and the per-allele
table (primary); MAE on `log1p`; precision@10 at the 2-hour threshold with base
rate and reachable ceiling, with zero-ceiling alleles counted separately; pooled
Spearman/Pearson as secondary only. Plus, from the stage 6 machinery: the d=4
vs d>=5 strata on the **same 65-allele panel** at the 20-row bar, the
differential target, and paired cluster bootstrap intervals on every
arm-vs-arm delta.

Verdict language, applied uniformly and regardless of which direction it
flatters:

- interval excludes 0 and its **lower** bound exceeds +0.05 → gain clears the
  predeclared bar;
- interval excludes 0 but the lower bound is below +0.05 → **real improvement,
  size unresolved against the bar** (this is the honest wording even when the
  point estimate is above 0.05);
- interval crosses 0 → **inconclusive**, not negative;
- interval excludes +0.05 from below → **negative**, the bar is ruled out.

## 5. If something goes wrong

- **The command errors before producing numbers:** fix and re-run. Nothing was
  observed, nothing is compromised.
- **The command succeeds:** that was the one pass. Do not tweak an arm and
  re-run. If a re-run is unavoidable — a genuine bug in the scorer, not a
  disappointing result — **both runs are disclosed in the write-up**, with the
  reason and what changed, in the same spirit as the disclosed test exposure
  already recorded in `EVALUATION.md`.
- **An arm is missing or malformed:** score the others. Do not delay the pass
  past the cutoff to wait for it.

## 6. Afterwards

Fill the holes tagged in `reports/REPORT.md`, regenerate
`reports/figures/make_figures.py` with `--split test`, and update
`reports/compute_ledger.md` with realised spend. The limitations register in
`reports/limitations.md` is reviewed once more **after** the numbers are known,
because some limitations only become material in light of the result — but no
limitation already written is removed because the result made it inconvenient.
