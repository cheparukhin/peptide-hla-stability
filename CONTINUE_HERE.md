# Continuation prompt — final test pass

Written 4 October 2026, 13:05 BST, handing off from a session that ran out of
context. Paste the whole of this file into a fresh session.

---

You are orchestrating the final stretch of a peptide–HLA stability hackathon
submission in `/Users/cheparukhin/ai_science_hack`.

**Read first:** `HACKATHON_PLAN.md`, `EVALUATION.md`, `CLAUDE.md`,
`reports/TEST_SCORING_RUNBOOK.md`, `reports/SUBMISSION.md`.

Git is clean and pushed at `db6fabe`, in sync with origin. No processes are
running — all subagents were stopped deliberately before the handoff.

## What is already done

Every experiment is finished and written up. The Boltz-2 production fold
completed: **28,166 / 28,166, zero failures, $212.71**, poses reproducing the
pilot within 0.04 Å. Stages 3 (ESM-2), 3b, 3c, 5 (structural + inverse
folding), 6 machinery, 7a and 7b are all complete. `reports/SUBMISSION.md` is
written except for test-side holes.

**Every number in the project is a validation number.** The test split has
never been scored.

## The one remaining deliverable: the single test-scoring pass

### Step 1 — regenerate four missing test prediction files

No trained model is persisted in this repo, so each arm must be refit on train
and then predict the 5,633 test rows.

Already on disk and tracked in git (`preds/test/`, force-added despite
`preds/` being gitignored):

- `seq_baseline.csv` — 5,633 rows
- `seq_ensemble_pep_pseudo.csv` — 5,633 rows ← the CI baseline
- `boltz_structural.csv` — 5,633 rows, 100% structural coverage, 0 fallback

**Missing, regenerate these four:**

- `seq_ensemble_pep_domain.csv`
- `esm_ensemble.csv`
- `esm_plus_seq_ensemble.csv`
- `esm_ensemble_150m.csv`

Use the **frozen** configurations in `reports/stage2_headline.json` and
`reports/stage3_headline.json`. **Do not re-select, re-tune, or re-run a
grid.** 30 members per arm, `cv_folds` imported from
`scripts/baseline_ensemble.py`, never reimplemented.

ESM arms: 35M, middle layer (6 of 12), peptide per-position PCA 256, HLA
34-contact PCA 74, MLP (256,64), L2=1e-2. The embedding cache already exists
under `features/esm/` — **do not re-extract**. **Fit PCA inside each fold on
that fold's training rows only**; a cross-fit basis leaks and the leak is
invisible, because the labels are untouched either way and it still looks
honest from outside.

**Correctness check that does not touch test:** each refit must reproduce its
committed *validation* number. `esm_plus_seq_ensemble` ≈ 0.6761,
`esm_ensemble` ≈ 0.6830, and the sequence ensemble was previously verified
**byte-identical** (same md5). If a refit does not reproduce, stop and
investigate rather than proceeding.

**Parallelise across the 8 cores using separate processes each BLAS-pinned to
one thread** — not one process with many threads. Thread count changes float32
reduction order and moves a single network by ~0.02; separate processes keep
member predictions summing in canonical order, so reproduction stays exact.
Total work is small: ~2.4 s per network, so ~180 networks is minutes.

Do **not** put this on Modal. Setup (image build, uploading the 589 MB
embedding cache and the 67 MB feature table, a smoke pass) would cost far more
than the job itself.

### Step 2 — THE RULE THAT MATTERS MOST

**You may PREDICT on test. You may NOT SCORE on test.**

No Spearman, no MAE, no correlation, no "looks reasonable", no eyeballing
predictions against test labels — until the single pass below. The test set is
scored exactly once. A metric computed in passing breaks the single-pass
guarantee and takes the submission's central methodological claim with it.

### Step 3 — capture provenance, then score once

```bash
.venv/bin/python scripts/capture_test_provenance.py \
  --arm seq_ensemble_pep_pseudo=preds/test/seq_ensemble_pep_pseudo.csv \
  --arm seq_baseline=preds/test/seq_baseline.csv \
  --arm seq_ensemble_pep_domain=preds/test/seq_ensemble_pep_domain.csv \
  --arm esm_ensemble=preds/test/esm_ensemble.csv \
  --arm esm_plus_seq_ensemble=preds/test/esm_plus_seq_ensemble.csv \
  --arm esm_ensemble_150m=preds/test/esm_ensemble_150m.csv \
  --arm boltz_structural=preds/test/boltz_structural.csv \
  --out reports/test_scoring_provenance.json
```

Then, **exactly once**:

```bash
.venv/bin/python scripts/stage6_report.py --split test --n-boot 2000 \
  --stratum-min-rows 20 \
  seq_ensemble_pep_pseudo=preds/test/seq_ensemble_pep_pseudo.csv \
  seq_baseline=preds/test/seq_baseline.csv \
  seq_ensemble_pep_domain=preds/test/seq_ensemble_pep_domain.csv \
  esm_ensemble=preds/test/esm_ensemble.csv \
  esm_plus_seq_ensemble=preds/test/esm_plus_seq_ensemble.csv \
  esm_ensemble_150m=preds/test/esm_ensemble_150m.csv \
  boltz_structural=preds/test/boltz_structural.csv \
  --out-dir reports --prefix stage6_test
```

The **first positional is the baseline** every paired interval is taken
against. `--stratum-min-rows 20` is the frozen default; the 10 used elsewhere
is validation-only, because validation is half the size. Expect 25–40 minutes.

**Verdict language is pre-committed** in the runbook and binds in both
directions:

- interval excludes 0 **and** lower bound > +0.05 → clears the predeclared bar
- excludes 0 but lower bound < +0.05 → **real improvement, size unresolved**
- crosses 0 → **inconclusive, not negative**
- excludes +0.05 from below → **negative**, the bar is ruled out

If the command errors before producing numbers, fix and re-run — nothing was
observed. If it **succeeds**, that was the one pass. Any re-run after seeing
results is disclosed in the write-up.

### Step 4 — finish the submission

Fill holes `B2`, `B4`, `S6a–d`, plus table-tracked `B3`, `R1`, `P1`, `A1` in
`reports/SUBMISSION.md`. Regenerate `reports/figures/make_figures.py`. Update
`reports/compute_ledger.md` with realised spend. Commit and push.

## Two findings that must not regress

1. **The one conclusive comparison is about full-domain weakness, not
   pretraining strength.** ESM-2 beats the full-domain ensemble by +0.0152
   differential concordance; the plain pseudosequence ensemble beats it by
   +0.0124; **ESM-2 against pseudosequence is null** (+0.0028). An outside
   reviewer caught this framing and it would be easy to re-flatten while
   summarising.
2. **Stage 3b is an unresolved measurement, not a null.** Its DiD upper bounds
   sit below 0.05, so the frozen rule returns "rules out a worthwhile gain" —
   but the measured DiD floor straddles 0.05, so power at the bar is
   **unestablished**. It must not be cited as having been *able* to find a
   worthwhile differential.

## Other open items

- **ProteinMPNN QC sample** — a Modal run was dispatched and the local client
  was killed at handoff. **Check for
  `reports/stage5_inverse_folding_qcsample_a-cheparukhin.csv` before
  relaunching**, or you pay twice. Relaunch command is in
  `reports/stage5_inverse_folding.md`. ~5 min, ~$1. Report a **distribution and
  a triage list, never a failure rate**; the detector was validated on one
  failure mode in one complex, and that provenance travels with every number.
- **FoldX** — a separate live session, "Run FoldX on 28,166 structures via
  Modal", is holding at ~$0.33 spent. It needs a decision on a ~$105 repaired
  run. RepairPDB **is** required (unrepaired scoring ranks Boltz-2 strain, not
  binding, and returns a positive ΔG on a row with a measured finite
  half-life), and it is 71× slower. Expect a negative: stage 5 showed Boltz-2
  structural features conclusively hurt, and wet-lab affinity itself only ranks
  stability at ρ 0.580 against a 0.693 baseline. The user decides.
- A `demo/` page was merged from another session and already carries the
  corrected framing.

## Standing rules

- Never recompute splits — `pepstab.data.load_with_splits()`, join on
  `(allele, peptide)`, never positionally on `pair_id`.
- `data/rasmussen_et_al_dataset.csv` is read-only;
  `(cd data && shasum -a 256 -c SHA256SUMS)` must pass.
- `EVALUATION.md` is frozen and is not reopened for an inconvenient result.
- Tuning parity means equal **budget** with scale-appropriate **ranges**, not
  transplanted values. A transplanted ladder cost the ESM arm 0.109 SCC and
  nearly produced a false negative. A two-point ladder can never satisfy an
  interior check.
- The `check_interior` gate cannot distinguish a flat objective from a
  truncated ladder. Measure the objective before concluding a ladder is too
  narrow.
- Subagents do not commit; the orchestrator commits.
