# Stage 8 FoldX — session handoff prompt

*Paste the whole of the next section into a new session. It is self-contained.*

---

You own the FoldX arm (stage 8) of the peptide–HLA stability project in
`/Users/cheparukhin/ai_science_hack`. A previous session built and validated the
pipeline and measured everything needed to launch. **The user has approved
scoring everything: both arms, full cohort.** Your job is to run it and score it.

## Read first

`reports/stage8_foldx.md` — the predeclared protocol, plus a **Results** section
(R1–R8) holding every measurement so far. Nothing above the Results line may
change. Then `HACKATHON_PLAN.md`, `EVALUATION.md`, `CLAUDE.md`,
`reports/stage4c_ectodomain_pilot.md`, `reports/stage2c_affinity.md`, and
`scripts/stage5_structural_arm.py` (your template for the evaluation step).

## State: pipeline validated, nothing scored

Working and tested (64 tests pass, `.venv/bin/python -m pytest tests/test_foldx.py -q`):
`pepstab/foldx.py`, `modal_app/foldx_scoring.py`, `scripts/foldx_convert.py`,
`scripts/foldx_forecast.py`, `scripts/foldx_concat.py`, `scripts/foldx_arm.py`,
`tests/test_foldx.py`. All untracked/uncommitted.

**Read R9 before launching anything.** A later session found a **1.88x cost
defect in the launch command this handoff originally gave** (`--chunk 32` on
`--procs 64` fills half of every container and bills the idle half: $386 rather
than $205, past the $250 stop threshold, and the forecaster reported $205 for
it). The commands below are corrected and `score` now refuses the bad shape.
That session also wrote `scripts/foldx_arm.py` (step 4) and journalled `score`,
but **could not run anything on Modal** — its sandbox denied `api.modal.com` —
so nothing is scored and ~$0.33 is still the stage total.

The binary **works**: FoldX 5.1, x86-64 static, runs in a Modal container,
SHA-256 `faed5e54e47744ab6ab75f8f9ad1a1f2e2cdbeac4f0f98d4e36c97049eb14268`
agreeing across local file, container, and protocol. `.fxout` header matches the
parser exactly, 32 columns. 10/10 structures scored across both workspaces, zero
failures. `foldx5_Linux_0/` is gitignored, proprietary, **never commit it**; it
cannot run on the arm64 host, only in a container.

**RepairPDB is required, and this reverses the protocol's own §4 prediction.**
Unrepaired FoldX energy ranks structures by strain inside the predicted HLA chain
(Spearman +0.90 with `IntraclashesGroup2`) and returns **+0.679 kcal/mol — "does
not bind" — on a peptide with a measured finite half-life**. Repair cuts clashes
8.50→3.01 while the genuine `van_der_waals` term barely moves. §4 reasoned from
"0 of 5,745 residues missing a heavy atom", which is true, but FoldX's clash term
is a *soft* penalty firing well above the 2.2 Å hard-overlap threshold §4
measured: Boltz-2's side chains are complete but strained. Production therefore
uses repair, which satisfies §4's predeclared rule (it reorders, and it fits
budget). Full detail in R2 — do not re-derive it.

Measured: unrepaired **3.62 s**/structure at 5 procs, **6.91 s** at 32 procs
(**1.91× contention**); repaired **246 s** at 5 procs (**~71×** multiple),
contention **unmeasured**. Memory **107 MB peak RSS per process** including
RepairPDB — 4.7× headroom at 32 procs on 16 GiB.

## Run sequence

**Step 1 — measure repair contention (~$0.5, ~9 min).** Converts the derived
cost into a measured one before committing ~$200. Run it at the **production**
shape: a 32-process pilot cannot see 64-process contention any more than the
5-process smoke could see 32 (that mistake is R3's 1.91x correction).

```bash
cd /Users/cheparukhin/ai_science_hack && MODAL_PROFILE=a-cheparukhin .venv/bin/modal run modal_app/foldx_scoring.py::pilot --profile a-cheparukhin --n 64 --cpu 64 --memory-gib 16 --procs 64
```

Read `seconds_per_structure` for `repair_then_analyse` from
`reports/stage8_foldx_pilot_a-cheparukhin.json`, then re-forecast free with
`scripts/foldx_forecast.py` (never imports modal). **Pass `--chunk`** — without
it the forecast assumes a perfectly filled pool, which is what hid the 1.88x
defect:

```bash
.venv/bin/python scripts/foldx_forecast.py --measured-from reports/stage8_foldx_pilot_a-cheparukhin.json --cpu 64 --containers 150 --procs 64 --chunk 64 --n 14083
```

Double the printed figure: it is one half. **If the repaired total exceeds
~$250, stop and report before launching** — that would be >64% of remaining
budget and a materially different decision from the one approved. At 150
containers the headroom is only ~22% ($205 of $250), so a measured contention
worse than the derived 1.91x can cross it; 40x64c costs $188 with ~33% headroom
and 45 min wall if that matters more than speed.

**Step 2 — four production runs.** Both arms × both profiles. The unrepaired
pair is ~$6 and ~3 min; the repaired pair ~$205 at 150×64c (13 min) or ~$188 at
40×64c (45 min). Run the two profiles concurrently; each profile folds **its own
half** and `--profile` must equal `MODAL_PROFILE`.

**`--chunk` must be at least `--procs`.** `score_chunk` runs `Pool(procs)` over
one chunk, so a smaller chunk idles — and bills — the rest of the pool. `score`
now asserts this, so the old `--chunk 32 --procs 64` command fails fast instead
of costing 1.88x its forecast.

```bash
cd /Users/cheparukhin/ai_science_hack && MODAL_PROFILE=a-cheparukhin .venv/bin/modal run --detach modal_app/foldx_scoring.py::score --profile a-cheparukhin --cpu 64 --procs 64 --containers 150 --chunk 64
```

```bash
cd /Users/cheparukhin/ai_science_hack && MODAL_PROFILE=colleague .venv/bin/modal run --detach modal_app/foldx_scoring.py::score --profile colleague --cpu 64 --procs 64 --containers 150 --chunk 64
```

Then the same two with `--repair` added. **Parallelism buys wall time, not
money** — cost is core-hours, fixed by the work; extra containers add startup and
cost slightly *more*. Per half at `--chunk 64`: 40 containers 0.74 h/$94,
150 containers 0.22 h/$102, 220 containers 0.16 h/$108. Confirm your workspaces
actually permit 150 concurrent containers before relying on the wall time.

`score` accumulates rows on the **client**, so a dropped connection used to lose
the whole half (~$92) even with `--detach` — detaching keeps the containers
alive but nothing collects their results. Each chunk is now `fsync`ed to
`<output>.partial.jsonl` as it lands; pass `--resume` to continue an interrupted
run. A leftover journal makes the next run **refuse to start** rather than
silently rescore, so delete it once the CSV is safe.

**Step 3 — concatenate.** `scripts/foldx_concat.py` and `--repair`. It asserts
28,166 rows, zero duplicate `(allele, peptide)`, exact cohort coverage, and one
binary SHA plus one repair setting across both halves. Writes
`reports/stage8_foldx_features{_repair}.csv`.

**Step 4 — score arms A and B.** `scripts/foldx_arm.py` **now exists** and
implements protocol §5 as predeclared; it was verified end to end on a
synthetic table carrying the real 29-column FoldX header. Three modes:

```bash
.venv/bin/python scripts/foldx_arm.py grid     --table reports/stage8_foldx_features_repair.csv
.venv/bin/python scripts/foldx_arm.py ensemble --table reports/stage8_foldx_features_repair.csv --write-preds
.venv/bin/python scripts/foldx_arm.py diagnose --table reports/stage8_foldx_features_repair.csv
```

Run `grid` first — `ensemble` fits from the selections it writes and refuses to
run without them. **`grid` selects on the mean dev MSE over all six ensemble
seeds** (R10/R11: a single-seed argmin on this objective is noise, and two runs
differing only in row order picked different rungs, one interior and one on a
boundary). That means 6x the fits: budget roughly 40 min for `grid` and ~10 min
for `ensemble` on the full table, all local CPU.

If the interior gate still fires, **read the message before widening anything**.
It now reports the ladder spread against the worst within-rung seed sd and says
explicitly whether widening would help; when the objective is flat relative to
its own noise, a wider ladder is the wrong response and R10 has the numbers.

Two things to know about it. It carries a **`seq_only` control** alongside arms
A and B, as stage 5 does: without it an additive arm differs from the committed
baseline in three ways at once (encoding count, L2 selection, the new block) and
a drop cannot be attributed to any of them — in stage 5 that control is what
established the loss was the features rather than the tuning. And the interior
gate writes `reports/stage8_foldx_ladder.csv` **before** it fires, because
`check_interior` cannot tell a truncated ladder from a flat objective and the
per-rung dev MSEs are the only thing that can; it withholds
`stage8_foldx_selected.json` on a boundary hit so `ensemble` cannot fit a
handicapped arm.

R9 records the ladder measured free on real data: the declared `1e-4 … 1e-1`
selects **1e-3, interior**, so the gate should not block the sequence-containing
arms. Arm A's ladder cannot be checked until the FoldX columns exist.

## Hard rules

- **Validation only. Never load or score the test split** — scored once at
  stage 6 by the orchestrator under `reports/TEST_SCORING_RUNBOOK.md`.
- Load frozen splits from `data/splits.csv`, join on `(allele, peptide)`,
  **never** on `pair_id` (positional into the raw CSV). Load the cohort from
  `data/structural_cohort.csv`; never recompute either.
- **Tuning parity is equal budget with scale-appropriate ranges, not
  transplanted values** — a transplanted ladder cost the ESM arm 0.109 SCC and
  nearly faked a negative. Check the selected value is interior; note the known
  caveat that `check_interior` misreads a flat objective as a truncated ladder,
  so measure the objective before concluding the ladder is too narrow.
- **No batch without a passing pilot.** CPU only, no `gpu=` anywhere (a test
  asserts this by reading the source).
- Own only `modal_app/foldx_scoring.py`, `scripts/foldx_*.py`,
  `pepstab/foldx.py`, `tests/test_foldx.py`, `reports/stage8_foldx*`. Do not
  touch other stages' files, `EVALUATION.md`, `HACKATHON_PLAN.md`, `CLAUDE.md`,
  `README.md`.
- `data/rasmussen_et_al_dataset.csv` is read-only;
  `(cd data && shasum -a 256 -c SHA256SUMS)` must still pass.

## Gotchas already paid for — don't rediscover these

- **`--dry-run` is not free.** No worker call (AST-tested), but `modal run`
  builds the 87 MB FoldX image layer first. The free path is
  `scripts/foldx_forecast.py`.
- **A 5-process smoke cannot see 32-process contention.** Measure runtime in the
  shape you extrapolate to.
- **Cohort alleles carry the suffix**: `HLA-B*14:01(C67S)`. An `isin` against the
  bare name silently matches nothing. With the suffix the C67S count is 1,135 and
  borrowed-alpha3 is 1,103, reproducing §6.
- A **module-scope file read kills every Modal container on import**, with the
  symptom of an app with zero tasks and no error. Read lazily inside functions.
  (This app is clean — keep it that way.)
- **Chunk size can bind before container count.** Raising `--containers` alone
  does nothing if `--chunk` leaves too few chunks.
- **And a chunk below `--procs` bills idle cores.** `score_chunk` pools over one
  chunk, so `--chunk 32 --procs 64` costs 1.88x its forecast. `score` asserts
  `chunk >= procs`; `foldx_forecast.py` needs `--chunk` passed or it assumes a
  full pool. These are opposite constraints — chunk large enough to fill the
  pool, small enough to leave more chunks than containers. At 14,083 per half,
  `--chunk 64` gives 221 chunks, which fills up to 221 containers.
- **A forecast is an instrument and can be wrong.** The 1.88x defect was
  invisible because the forecaster modelled perfect packing: the shape and the
  tool that priced it were wrong together, and the tool was the optimistic one.
  Price the command you will actually type, with its real flags.
- Fixed already, don't reintroduce: `score` once wrote one filename for both
  arms, so the cheap run would silently overwrite the expensive one.

## Budget and expectations

~$212.71 of ~$600 spent before this stage; this stage has spent ~$0.33; the plan
commits ~$200 more at 150×64c (or ~$188 at 40×64c), about half of what remains. Metered rates in
`reports/ectodomain_rates.json` — `$0.04730`/core-hour, `$0.00800`/GiB-hour.
**Never published list rates**; this project already caught a 4.6× cost error
from an unmeasured assumption.

**Expect a negative, and report a bounded one with a cost figure.** Three signals
converge: wet-lab affinity used directly ranks stability at **ρ 0.580** against a
sequence baseline of **0.693**; stage 5's structural arm on these same Boltz-2
poses is a conclusive negative at **−0.0715 [−0.1228, −0.0251]** with a
`seq_only` control winning at all six L2 values; and FoldX is a third extraction
from those same structures. FoldX is a genuinely different quantity from pLDDT
and PAE, so §1's open question — does an explicit energy decomposition add
anything *on top of* a trained sequence model — is still open and is bounded by
neither figure. The brief explicitly values well-supported negatives. Append
results to the Results section of `reports/stage8_foldx.md`, which already holds
R1–R8; **nothing above that line changes**. Report cost next to accuracy.
