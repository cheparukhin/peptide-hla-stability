> **Historical study record.** The protocols, results and future-tense instructions below describe the completed October study. The [current project review and research plan](../NEXT_RESEARCH_PLAN.md) is the sole forward work queue; its [dated correction register](../NEXT_RESEARCH_PLAN.md#dated-correction-register--12-october-2026) supersedes conflicting interpretation, selection and blend-gate claims here. No old protocol should be relaunched as a new study without the current evaluation gates.

# The platform artifact, and the reproducibility floor underneath it

Date: 4 October 2026. **Status: measured on all five outer folds, with the
macOS/Linux and thread-count effects separated.**

A0 is Phase A's reference arm and was the only arm in the study fitted on
macOS. Every other arm — the A1–A3 screen, the A5/ESM arm, Phase B's arms, and
the forthcoming 30-member assessment — runs in Linux containers. This report
measures what that difference is worth, so any cross-platform comparison in the
study can be discounted by a number rather than by intuition.

**The headline is not the platform effect.** Two runs of the *same five units*
at the *same settings* on Modal disagree by **0.0043** on median-of-medians —
more than the macOS→Linux shift of **+0.0013**, and more than either of its
components. Modal's reproducibility floor is the dominant term, and no effect
of this size in this study is resolvable from a single run per arm.

Artifacts: `phase_a_seq_oof_cluster_linux.csv`, `phase_a_a0_runs_linux.csv`,
`phase_a_a0_provenance_linux.json`, and the journals
`phase_a_a0_runs_colleague_t2.partial.jsonl` (full 30 units) and
`phase_a_a0_runs_colleague_t1.partial.jsonl` (5 level-2 units). The macOS A0 is
unchanged; nothing here overwrites it.

---

## 1. The four cells

Identical recipe throughout: `seq_ensemble_pep_pseudo`, the frozen outer folds
from `phase_a_outer_folds.csv`, 5 inner folds × 2 encodings × 3 seeds = 30
members per outer fold, stage 2's per-encoding L2, seeds (0, 1, 2). Only the
platform and the BLAS thread count vary.

| Cell | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | median of medians |
|---|---:|---:|---:|---:|---:|---:|
| macOS / threads=1 *(shipped A0)* | 0.6765 | 0.6281 | 0.6418 | 0.6493 | 0.6515 | **0.6493** |
| Linux / threads=1 | 0.6832 | 0.6302 | 0.6464 | 0.6473 | 0.6567 | **0.6473** |
| Linux / threads=2 *(the new reference)* | 0.6781 | 0.6248 | 0.6378 | 0.6549 | 0.6506 | **0.6506** |
| Linux / threads=2, **replicate** | 0.6781 | 0.6234 | 0.6378 | 0.6549 | 0.6621 | **0.6549** |

All cells: 68 eligible alleles per fold at the ≥ 20-row bar, 99.3–99.7 % row
coverage. Level-3 units feed only `inner_oof_30` and never a fold score, so the
threads=1 cell was run at level 2 only — five units rather than thirty.

## 2. The decomposition

| Comparison | Δ median-of-medians | per-fold Δ | max abs diff | mean abs diff | corr | folds bit-identical |
|---|---:|---|---:|---:|---:|---:|
| **Platform**, at threads=1 (macOS → Linux) | **−0.0020** | +0.0067, +0.0021, +0.0046, −0.0020, +0.0052 | 2.51e-01 | 3.28e-02 | 0.998646 | 0/5 |
| **Thread count**, on Linux (1 → 2) | **+0.0034** | −0.0051, −0.0054, −0.0086, +0.0076, −0.0060 | 2.44e-01 | 3.23e-02 | 0.998671 | 0/5 |
| **Compound** (macOS/t1 → Linux/t2) | **+0.0013** | +0.0016, −0.0033, −0.0040, +0.0056, −0.0008 | 2.03e-01 | 3.29e-02 | 0.998663 | 0/5 |
| **Run-to-run**, identical settings | **+0.0043** | +0.0000, −0.0014, +0.0000, +0.0000, +0.0115 | 1.91e-01 | 1.44e-02 | 0.999348 | **3/5** |

Three things in that table matter more than the individual numbers.

**The platform effect reproduces Phase B's independent measurement.** Phase B
measured macOS against Modal Linux at threads=1 on outer fold 0 and found
max |diff| 1.82e-01, mean 3.21e-02, worth 0.0020 on median-of-medians. This
run, on all five folds, gives max 2.51e-01, mean 3.28e-02, and **−0.0020**.
Two independent measurements agreeing to the digit on the metric and to within
30 % on the tail is as close to confirmation as this project gets.

**The components partially cancel, so the compound understates both.** Platform
is −0.0020 and threads is +0.0034; their sum (+0.0014) is the compound
(+0.0013). Reporting only "macOS A0 versus Linux A0 is +0.0013" would have
suggested the platform barely matters, when in fact two effects roughly twice
that size happened to point in opposite directions. This is the whole reason
the thread count was varied rather than changed silently alongside the
platform.

**Predictions move far more than the metric does.** Every comparison shifts
individual predictions by up to ~0.25 on the log1p target — around a third of
the outer-held RMSE — with correlation 0.9987. The 30-member ensemble and the
rank-based metric absorb almost all of it. That is consistent with the
project's standing note that single networks move ~0.02 median Spearman with
thread count while ensembles are steady, and it is why the arity matters.

## 3. The reproducibility floor, and why it is the real finding

The fourth row is two runs of the same five units at the same settings, on the
same day, from the same journal-writing client. It arose by accident: a
`--threads 1` launch dispatched correctly but its containers reported
`OMP_NUM_THREADS=2`, making it an exact-settings replicate of the threads=2
batch rather than the decomposition arm it was meant to be. It was kept for
that reason.

- **Three of five folds came back bit-identical** — max |diff| exactly 0.0.
- **Two did not**, by 1.6e-01 and 1.9e-01, and fold 4's median moved
  **+0.0115**, which is larger than any systematic effect in this report.
- Median-of-medians moved **+0.0043**.

The mechanism is host heterogeneity. Per-container CPU fingerprints recorded on
the re-run show the five containers split across two microarchitectures:

| SIMD capability | containers |
|---|---:|
| `avx, avx2` | 3 |
| `avx, avx2, avx512dq, avx512f` | 2 |

OpenBLAS dispatches a different kernel per microarchitecture, which changes
float32 accumulation order. A unit that happens to land on the same class of
host twice reproduces bit-for-bit; one that does not, does not. Modal does not
offer a host-type guarantee on CPU containers, so **this is not fixable by
pinning threads or seeds** — only by replication.

### What follows for the study

> A difference of 0.005 or less in median-of-medians between two arms, each run
> once, is not distinguishable from the platform-and-host noise measured here.

That bar is uncomfortably close to the project's 0.05 worthwhile threshold
divided by ten, and it applies to comparisons this study is already making:

- The **A5/ESM arm** scored 0.6534 against the macOS A0's 0.6493 (+0.0041) and
  against the Linux A0's 0.6506 (+0.0028). Both are inside the floor. The ESM
  arm's result should be read as parity — which is what stage 3 concluded on
  validation, and what its per-fold deltas already said by being mixed in sign.
- Any **assessment-versus-A0** comparison should use the Linux A0
  (`*_linux` files) rather than the macOS one, which removes the platform term
  but **not** the run-to-run term.
- Effects near this size need **replicates**, not a tighter protocol. Seeds are
  already averaged over; the variation here is across hosts, beneath the seed
  axis.

## 4. What this does not say

- It does not say the macOS A0 is wrong. Both cells are correct runs of the
  same recipe; neither is a reference the other should be corrected toward.
- It does not establish a *distribution*. One replicate gives one difference,
  not a standard error. The 0.0043 is a single observation of run-to-run
  variation and should be treated as an order of magnitude, not a bound.
- It does not transfer to arms with different learners. CatBoost (A1, A4) does
  not go through the same BLAS paths, and its host sensitivity is unmeasured.
- It does not license re-running an arm until it looks better. The floor is a
  reason to replicate and report the spread, not to select among runs.

## 5. Reproduce

```
MODAL_PROFILE=colleague modal run modal_app/phase_a_a0_modal.py::smoke \
    --profile colleague --threads 2
MODAL_PROFILE=colleague modal run --detach modal_app/phase_a_a0_modal.py::run \
    --profile colleague --threads 2 \
    --measured-from reports/further_research/phase_a_a0_smoke_colleague_t2.json
MODAL_PROFILE=colleague modal run --detach modal_app/phase_a_a0_modal.py::run \
    --profile colleague --threads 1 --levels 2 \
    --measured-from reports/further_research/phase_a_a0_smoke_colleague_t2.json
.venv/bin/python modal_app/phase_a_a0_modal.py --assemble \
    reports/further_research/phase_a_a0_runs_colleague_t2.partial.jsonl --threads 2
```

Cost: $0.28 forecast for the 30-unit batch and $0.05 for the 5-unit threads=1
arm, both on the `colleague`/`sofyaleyn` workspace with the user's
authorisation, priced from a measured 199.7 s/unit and logged in
`reports/compute_ledger.csv`.

## 6. Two measurement notes

**`to_csv` is not bit-exact.** pandas 3.0.6 writes about sixteen significant
digits, dropping the final unit in the last place: a prediction round-tripped
through `phase_a_seq_oof_cluster*.csv` differs from the journal value by up to
**4.44e-16**. Irrelevant to every metric here, but it means bit-identity must
be judged on the journals, not the CSVs — judged on the CSVs, the three
bit-identical folds above read as zero. It also explains Phase B's
"A0 is bit-identical to its own harness, max |diff| 4.44e-16": that figure is
this CSV precision, so the agreement was exact, not merely close.

**The thread pin needs two mechanisms.** Setting `OMP_NUM_THREADS` in the Modal
image `.env` — the method `modal_app/phase_a_esm.py` relies on — did not take
effect for `IMAGES[1]`: the dispatch was correct and every container still
reported 2. `modal_app/phase_a_a0_modal.py` now also sets the variables inside
the worker before the first numpy import, which is valid only because that
module imports numpy nowhere at module scope; the worker asserts that property
at run time rather than assuming it, since otherwise the assignment would be a
no-op that looked like a fix. Any arm that reports a thread count should carry
a `thread_pin_ok` field and check it — this one was caught by that field alone.
