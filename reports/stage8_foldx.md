# Stage 8: FoldX empirical interaction energy

**Status: protocol predeclared. No FoldX score exists yet.** This document was
written before the binary had executed anywhere, and before any number it
produces had been seen. Everything below — the cohort, the features, the
ladders, the comparisons and the verdict table — is fixed in advance. Results,
when they exist, go in a clearly separated **Results** section appended at the
bottom; nothing above it changes afterwards.

Owned by the `foldx` workstream: `modal_app/foldx_scoring.py`,
`scripts/foldx_*.py`, `pepstab/foldx.py`, `tests/test_foldx.py`,
`reports/stage8_foldx*`. No file belonging to another stage is modified.

---

## 1. Why this is expected to be a negative, and why it is still worth running

FoldX estimates **ΔG**, an equilibrium binding free energy: how favourable the
bound state is relative to the unbound one. The label is **t½**, a dissociation
half-life, which is governed by **ΔG‡**, the height of the barrier the complex
has to climb to come apart. A deep well and a high barrier usually travel
together, but they are not the same quantity, and the dataset is specifically a
kinetic one.

This project has already measured the ceiling on the thermodynamic quantity
**empirically**, which is stronger than the argument from first principles.
From [stage 2c](stage2c_affinity.md): **measured, wet-lab binding affinity used
directly as a stability predictor ranks at median per-allele Spearman 0.580**
[IQR 0.486–0.696] on 5,135 dual-labelled pairs across 33 alleles, against the
sequence baseline's **0.693**. Measured affinity — the real thing, from an
assay — is *below* the sequence baseline as a standalone stability predictor.
FoldX approximates that same quantity computationally and imperfectly, so its
standalone ceiling is bounded above by something already under the baseline.

**So the predicted outcome is a negative.** The challenge brief explicitly
values well-supported negatives, and this is designed to be a *measured,
bounded* one with a cost figure attached, not a fishing expedition for a win.
"FoldX did not help" is worth little; "FoldX standalone reaches ρ X, adds
Δ Y [CI] on top of the sequence baseline, at $Z and W core-hours" is worth
something, and it closes an energy-based objection to the submission with data
instead of an argument.

**The genuinely open question is the second one.** The 0.580 figure bounds
affinity *standing alone*. It says nothing about whether a physics-based
decomposition of the interface adds anything *on top of* a trained sequence
model — a different question, and not bounded by that number. A sequence model
learns which residues suit which pocket from labels; it has never been given an
explicit electrostatics or desolvation term. That increment is what this stage
is actually testing, and it is why the arm is run appended as well as alone.

**Two further things this stage cannot establish**, recorded now so they are
not discovered later as excuses:

- FoldX was parameterised on crystal structures. These are Boltz-2 predictions.
  Pose accuracy is good on the five crystal-referenced complexes (peptide
  heavy-atom RMSD 0.22–0.58 Å, [stage 4c](stage4c_ectodomain_pilot.md)) but
  those are training pairs, and held-out structural accuracy is unmeasured.
- A negative bounds *this* energy function, on *these* predicted structures, at
  *this* budget. It does not retire empirical energy functions generally.

---

## 2. Which structures

**All 28,166 production folds**, from the frozen cohort in
`data/structural_cohort.csv` — the same structures stage 4c.5 extracts
geometry and confidence from. No sampling, no seed, nothing to declare, **if**
the measured forecast fits the budget.

Structures are discovered with
`pepstab.structural_features.discover_folds` / `is_excluded`, imported rather
than reimplemented, so the `_smoke` / `_shards` / `_failed` exclusion has
exactly one implementation in the repository. `tests/test_foldx.py` asserts
that `pepstab.foldx` does not define its own copy.

**Fallback, predeclared, if the forecast does not fit.** Score a sample drawn
**before** any FoldX run, with `numpy.random.default_rng(20261004)`, stratified
by allele and by split so each stratum keeps its cohort proportion, frozen to
`reports/stage8_foldx_sample.csv` and committed before the batch starts. The
sample is never "whichever structures happened to finish" — that rule already
governs the stage 4c contingency panel and it governs this too.

**Validation only.** The test split is never loaded and never scored by this
workstream. Every number in the Results section will be a validation number on
the 2,817 validation rows, under `EVALUATION.md`'s eligible-allele rule
(≥20 rows, 68 of 75 alleles). Test scoring happens once, at stage 6, by the
orchestrator.

### Two workspaces, two halves

The folds live on the `pepstab-structures` Volume in `a-cheparukhin` and in
`colleague` (workspace `sofyaleyn`), 14,083 pairs each, disjoint and
interleaved pair-by-pair. `--profile` must equal `MODAL_PROFILE`;
`modal_app/foldx_scoring.py::check_profile` asserts it, as the fold runner and
the 4c.5 extractor do. The two runs are concatenated locally by
`scripts/foldx_concat.py`, which **asserts 28,166 rows and zero duplicate
`(allele, peptide)` pairs** before writing. A half-sized table is not obvious
from the file name, which is why the assert exists.

---

## 3. The features, fixed before anything is scored

### 3.1 The primary term

**`foldx_interaction_energy`** — FoldX `AnalyseComplex` interaction energy
between the **peptide chain** and the **HLA chain**, in kcal/mol. One number
per complex.

Three definitional points that change what the number means, decided here
rather than later:

1. **Peptide versus the whole 275-residue HLA chain, not the 182-residue
   groove.** `AnalyseComplex` separates *chains*, not residue ranges, so this
   is a genuine difference from the stage 4c.5 geometric features, which are
   all defined on groove residues 1–182. The alpha3 domain contributes almost
   nothing to a peptide interface 40 Å away, but "almost nothing" is an
   expectation, so the column is named for the chain and the difference is
   recorded here rather than glossed.
2. **Beta-2-microglobulin stays in the file.** It is not one of the two
   analysed groups, but deleting it would change the solvation environment of
   the alpha3 domain and, marginally, of the groove. The folded construct is
   the three-chain complex and the scored file is the three-chain complex.
3. **Which chain is the peptide is not inferred from its letter.** It comes
   from `pepstab.structural_features.load_fold`, which re-derives every chain
   boundary from the mmCIF and raises `MappingError` on disagreement with the
   declared inputs — the same verification stage 4c.5 rests on.

### 3.2 The decomposed terms

**Every numeric column FoldX's `Interaction_*.fxout` emits**, prefixed
`foldx_`. On the documented FoldX 5 layout that is roughly 28 columns: the
intra-group clash scores, backbone and side-chain H-bonds, van der Waals,
electrostatics, polar and hydrophobic solvation, van der Waals clashes,
side-chain and main-chain entropy, loop entropies, cis-bond, torsional and
backbone clashes, helix dipole, water bridges, disulfides, electrostatic k-on,
partial covalent bonds, ionisation energy, and the interface residue counts.

The column *set* is taken from the file's own header, not from a remembered
list. `pepstab.foldx.parse_fxout` is header-driven and raises if the table does
not match the header it declares, because FoldX's columns differ between
releases and a positional parser would mislabel every term while producing a
table that looks fine. This is the one part of the pipeline that could not be
validated against a real output before the binary existed — see §7.

### 3.3 Provenance columns, not features

`foldx_binary`, `foldx_binary_sha256`, `foldx_repair`, `foldx_group1`,
`foldx_group2`, `complex_id`, `allele`, `peptide`, `status`, and the timing
columns. These are recorded on every row and **excluded from the regression**.
The binary is proprietary and not in the repository, so its SHA-256 is the only
way someone with their own copy can confirm they ran the same build.

### 3.4 Row failures

A FoldX failure on one complex is data about that complex. `score_fold`
converts any exception into a recorded `status` string rather than losing the
container's whole chunk, matching `structural_features.extract_path`. Rows
whose status is not `ok` take the **declared sequence fallback**
(`preds/seq_ensemble_pep_pseudo.csv`), exactly as the stage 5 structural arm
does, so the arm is scored on the full frozen cohort rather than on a
retrospectively convenient subset of it. A coverage and failure table is
reported alongside every score.

---

## 4. `RepairPDB`: measured, not assumed

`RepairPDB` rebuilds clashing side chains and completes missing ones. It is
standard practice before `AnalyseComplex` on **crystal structures**, where
missing side-chain density and crystallographic contacts are common. These are
**predicted models**, where neither condition holds by construction, and
RepairPDB is also by far the more expensive of the two commands.

So it is treated as an empirical question with a predeclared decision rule.

**What has already been measured locally, on 45 pilot folds** (15 of them the
three-chain arm-B construct), by `scripts/foldx_convert.py`:

| Condition RepairPDB addresses | Measured |
|---|---:|
| Residues short of a heavy atom (missing side chains) | **0 of 5,745** |
| Non-standard residues | **0** |
| Cys SG–SG pairs at ≈2.0 Å (disulfide **bonds**, not clashes) | 75 (1.7/fold) |
| Genuine non-disulfide heavy-atom overlaps < 2.2 Å | **11 (0.24/fold)** |
| …of those, at the peptide–HLA interface | **1**, and it is an arm-A control fold |

Boltz-2 writes complete all-atom models: there is nothing for RepairPDB to
*rebuild*. What remains is a quarter of an overlap per fold, almost all of it
polar side-chain pairs in the HLA chain far from the groove. `tests/test_foldx.py`
re-asserts the zero-missing-atoms figure, so a future construct that does leave
gaps fails loudly instead of inheriting this decision.

**Decision rule, fixed in advance.** `::smoke` runs every smoke structure
twice, once `AnalyseComplex`-only and once `RepairPDB`-then-`AnalyseComplex`,
and records both the runtime multiple and the energy shift. Production uses
**`AnalyseComplex` alone** unless *both* of these hold:

- RepairPDB **changes the ranking** of the smoke structures by interaction
  energy. An absolute offset that preserves order is irrelevant to a
  rank-correlation metric and is not a reason to pay for it.
- The measured runtime multiple leaves the full-cohort forecast inside budget.

If repair is used, it is used on **every** structure in the cohort, recorded in
`foldx_repair`, and the forecast is re-run first. Mixed repaired and unrepaired
rows in one feature column would be a silent confound.

---

## 5. The evaluation protocol

Every parity rule below is the one the stage 5 structural arm already enforces
(`scripts/stage5_structural_arm.py`), for the same reasons, and each has
already cost this project something when it was violated.

### 5.1 Two comparisons, both predeclared

| Arm | Features | Question |
|---|---|---|
| **A. FoldX standalone** | the `foldx_` block alone | Does an empirical interface energy rank peptides at all? Expected to land near or below the 0.580 affinity ceiling. |
| **B. Sequence + FoldX** | the sequence baseline's features **plus** the `foldx_` block | Does the energy term add anything the trained sequence model has not already learned? **This is the open question.** |

Both are compared against the frozen sequence baseline
(`preds/seq_ensemble_pep_pseudo.csv`) on **identical** validation rows.
Arm B is the headline; arm A is reported alongside because a standalone number
is what makes the increment interpretable.

### 5.2 Ensemble parity

**30 networks per arm, asserted at run time, not documented and hoped for.**
The sequence comparator is 5 CV folds × 2 encodings × 3 seeds = 30. The FoldX
block has one natural representation rather than two encodings, so it reaches
30 the way the structural arm does: **5 folds × 6 seeds**, seeds `(0,1,2,3,4,5)`.

This is not bookkeeping. **Ensembling alone is worth +0.090 mean per-allele
Spearman on this task** — the 30-network ensemble against the mean of its own
30 members, [stage 2](stage2_baselines.md) — nearly twice the predeclared 0.05
bar, from no new information. An arm with fewer networks loses on ensemble
size, and the loss gets read as a verdict on the feature.

Folds come from `scripts.baseline_ensemble.cv_folds`, imported rather than
reimplemented, so the assignment is literally the same object.

### 5.3 Tuning parity: equal budget, scale-appropriate ranges

**Tuning parity means an equal number of grid points over ranges appropriate to
this block's scale — not transplanted values.** A transplanted ladder cost the
ESM arm 0.109 median Spearman here and nearly produced a false negative; the
stage 5 structural arm records the same lesson. A sparse-one-hot ladder applied
to standardised dense energy terms is a handicap wearing the costume of
fairness.

The FoldX block is **standardised dense columns, ≈28 of them**, against the
structural block's 109 and the sequence block's sparse one-hot. Ridge
regularisation scales roughly with the number of standardised predictors, so a
block with ~4× fewer columns wants a range about a decade lower. Declared now,
before any fit:

| Hyperparameter | Ladder | Points | Stage 5 structural, for comparison |
|---|---|---:|---|
| MLP L2 | `1e-4, 1e-3, 1e-2, 1e-1` | **4** | `1e-3, 1e-2, 1e-1, 1.0` (4) |
| Ridge alpha | `0.1, 1, 10, 100, 1000` | **5** | `1, 10, 100, 1e3, 1e4` (5) |

Same budget as stage 5 (4 and 5 points), shifted one decade down for the
smaller block. Hidden layer, max epochs and patience are taken unchanged from
`scripts/baseline_sequence.py`, as every other arm does.

**Interior check, enforced.** `check_interior` — the same contract as
`stage3b_esm_multitask` and `stage5_structural_arm` — refuses any arm whose
selected value sits at `min` or `max` of its own ladder. A boundary hit means
the ladder was truncated, so the selected value is not a selection. If it
fires, the ladder is **extended and re-selected**, and the extension is
recorded in the Results section.

**Note on ladder length, stated in advance:** a 2-point ladder can never
satisfy `min < selected < max`, so it can never pass the interior check. Both
ladders above have 4 and 5 points for that reason; neither may be shortened to
two "to save budget".

Selections are written to `reports/stage8_foldx_interior.csv` with the ladder,
its point count, and the interior flag per hyperparameter.

### 5.4 Metrics, uncertainty, verdict

Unchanged from `EVALUATION.md`, which was frozen at stage 1:

- **Primary:** median per-allele Spearman on validation, IQR and per-allele
  table reported.
- **Secondary:** MAE on `log1p` half-life, precision@10 at 2 h, pooled
  Spearman and Pearson.
- **Uncertainty:** paired cluster bootstrap over whole peptide clusters, 2,000
  resamples, seed `20261003`, both arms scored on the same resample. Report the
  **difference and its 95% CI**, not two separate intervals.
- **Verdict:** `describe_delta()`'s six mutually exclusive outcomes against the
  predeclared **0.05** minimum worthwhile gain. In particular, "crosses 0,
  upper < 0.05" is *inconclusive but rules out a worthwhile gain* — a bounded
  negative — and is the outcome this stage expects.
- **Cost reported next to accuracy**, in core-hours and dollars at the repo's
  **metered** rates (`reports/ectodomain_rates.json`), never published list
  rates.

### 5.5 Two diagnostics, labelled as diagnostics

Neither is a headline and neither may be used to select a model:

1. **FoldX versus measured affinity.** Correlate `foldx_interaction_energy`
   against the measured affinity labels on the 7,281 dual-labelled pairs. This
   says how well FoldX reproduces the quantity it is actually estimating, which
   is what separates "FoldX is a bad ΔG estimator here" from "ΔG is the wrong
   quantity for t½". The two failure modes look identical in the headline
   number and have completely different implications.
2. **Per-pocket decomposition, only if arm B is non-null.** `AnalyseComplex`
   does not decompose by pocket, so this would need per-residue energies and
   its own definitions. It is out of scope unless there is something to explain.

---

## 6. Awkward cases in the structures, recorded before they can be excuses

| Case | Count | Why it matters for FoldX, specifically |
|---|---:|---|
| **C67S engineered constructs** — `HLA-B*14:01`, `HLA-B*14:02`, `HLA-B*39:06` | 3 of 75 alleles, **1,135 pairs (4.0%)** | A cysteine is mutated to serine, removing a thiol at a groove position. FoldX's disulfide and solvation terms treat Cys and Ser differently, so these alleles carry a systematically different energy decomposition from their wild-type relatives. **The dataset contains no matched wild types**, so the effect cannot be measured by comparison; it is reported per allele instead, and the three are flagged in the output. |
| `HLA-B*14:01(C67S)` / `HLA-B*14:02(C67S)` | 756 pairs | Share one contact pseudosequence. Unlike a pseudosequence model, FoldX scores the full 275-residue chain and *can* separate them — a point in the arm's favour, worth checking rather than claiming. |
| **Borrowed alpha3** — `HLA-A*02:50`, `HLA-A*24:19`, `HLA-B*08:03` | 3 alleles | IMGT has only a 181-aa groove-only record, so alpha3 was taken from a close relative. Part of the folded chain is a different molecule. `AnalyseComplex` scores peptide against the **whole** chain including that alpha3, so these rows are the ones where the chain-versus-groove definition in §3.1 is least harmless. Flagged by `structural_features.alpha3_provenance` and reported separately. |
| **Disulfides read as clashes** | 1.7 per fold | Cys SG–SG at ≈2.0 Å is a bond. A naive clash filter would reject every fold. Classified separately in `scripts/foldx_convert.py`; FoldX models disulfides explicitly and should handle them, but the count is recorded so a sudden change is visible. |
| **No hydrogens, no OXT, no alternate conformations** | all folds | Boltz-2 writes heavy atoms only and no terminal OXT. FoldX adds polar hydrogens itself. The missing OXT means the C-terminal residue of each chain is formally incomplete; for the peptide's PΩ — which sits in the F pocket and matters — this is a real, if small, difference from a crystal structure. Recorded, not corrected: adding an atom Boltz did not predict would be inventing coordinates. |
| **Side chains are complete** | 0 missing in 5,745 residues | The usual reason to repair does not apply (§4). |

---

## 7. First real output: what must be checked before any number is trusted

The binary is a Linux x86-64 static build; the development machine is arm64
Darwin. **`::smoke` is the first execution of this build anywhere in this
project**, so a clean exit is not evidence of a correct result. Before any
score is reported, confirm on the smoke output:

1. The `Interaction_*.fxout` header matches what `parse_fxout` extracted —
   compare the raw header line to the parsed column names, by eye, once.
2. `Group1`/`Group2` are the peptide and HLA chains, in the orientation the
   row claims.
3. `Interaction Energy` is negative and of plausible magnitude for a 9-mer in
   an MHC-I groove (order −10 to −30 kcal/mol). A positive or near-zero value
   across all five smoke structures means the groups were wrong, not that the
   peptides do not bind.
4. The decomposed terms do not sum to something unrelated to the reported
   interaction energy.
5. `ldd` confirms the static link, and the binary found `molecules/`.

---

## 8. Reproducing this

```bash
# free: no Modal call at all
.venv/bin/python modal_app/foldx_scoring.py --self-check
.venv/bin/python scripts/foldx_convert.py \
    structures/ectodomain_pilot/ectodomain-20261004/boltz2 --out /tmp/foldx_pdb
.venv/bin/python -m pytest tests/test_foldx.py -q

# forecast, from a measured smoke result; still no Modal call
modal run modal_app/foldx_scoring.py::forecast \
    --measured-from reports/stage8_foldx_smoke_a-cheparukhin.json --cpu 32

# 3-5 structures per workspace, the project invariant
MODAL_PROFILE=a-cheparukhin modal run modal_app/foldx_scoring.py::smoke \
    --profile a-cheparukhin
MODAL_PROFILE=colleague     modal run modal_app/foldx_scoring.py::smoke \
    --profile colleague

# the batch, only after the forecast is approved
MODAL_PROFILE=a-cheparukhin modal run --detach \
    modal_app/foldx_scoring.py::score --profile a-cheparukhin --cpu 32
MODAL_PROFILE=colleague     modal run --detach \
    modal_app/foldx_scoring.py::score --profile colleague     --cpu 32
```

**FoldX provenance.** `foldx5_Linux_0/foldx_20261231`, SHA-256
`faed5e54e47744ab6ab75f8f9ad1a1f2e2cdbeac4f0f98d4e36c97049eb14268`, 86.7 MB,
ELF 64-bit x86-64, statically linked, with its `molecules/` directory. It is
**gitignored and not redistributed**: the build is licensed to this account.
Anyone reproducing this obtains their own copy from
<https://foldxsuite.crg.eu/> and checks the hash. The dated build name implies
a licence expiring 31 December 2026.

---

## Results

*Nothing above this line has changed since it was written. Below is what ran.*

**Status: the arm is not scored.** Neither arm A (FoldX standalone) nor arm B
(sequence + FoldX) has been fitted, because the feature table does not exist
yet: the production pass was not run. What follows is the pre-production
measurement sections 4 and 7 required, and one result that reverses a
prediction made above.

### R1. The binary executes, and the output is what the parser claims

First execution of this build anywhere in this project. Section 7's checks:

| Check | Result |
|---|---|
| `uname -m` | `x86_64` |
| `ldd` | "not a dynamic executable" — static link confirmed |
| `--version` | FoldX **5.1** |
| `molecules/` located | yes, 15 fragment definitions |
| SHA-256: local = in-container = section 8 | `faed5e54...4268`, all three agree |
| `Interaction_*.fxout` header vs `parse_fxout` | **exact**, 32 columns, compared by eye against raw text |
| `Group1`/`Group2` | `C` (peptide) vs `A` (HLA), the claimed orientation |
| Terms vs reported energy | plain sum −6.71 against reported −9.99: same sign and order, as expected of a weighted sum |
| Structures scored | 10/10 across both workspaces, zero failures |

Check 3 — the plausibility window — is the one that does not pass, and that is
the subject of R2.

### R2. RepairPDB is required. Section 4 predicted the opposite, and was wrong

Section 4 argued repair was unnecessary because Boltz-2 writes complete
all-atom models: 0 of 5,745 residues short of a heavy atom, and genuine
non-disulfide heavy-atom overlaps below 2.2 Å at 0.24 per fold. **Both
measurements stand.** The inference drawn from them does not.

FoldX's van der Waals clash term is a **soft** penalty that fires well above
the 2.2 Å hard-overlap threshold section 4 measured. Boltz-2's side chains are
complete but sit in slightly strained rotamers, so there is nothing to
*rebuild* and a great deal to *relax*. "Complete side chains" and "relaxed side
chains" are different properties, and section 4 conflated them.

**The unrepaired arm is not a cheaper version of the same measurement — it
measures Boltz-2 strain.** Three lines of evidence, 10 structures, both
workspaces:

1. **It ranks structures by strain inside the predicted HLA chain.** Spearman
   between the unrepaired interaction energy and `IntraclashesGroup2` is
   **+0.90** (exact permutation p=0.083, n=5), and **+0.80** against the
   interface van der Waals clash term (p=0.133). `IntraclashesGroup2` is not an
   interface quantity at all.
2. **It returns a positive binding free energy on a measured binder.**
   `A0101_AMVLDKKLY` scores **+0.679 kcal/mol** unrepaired — "does not bind" —
   on a row with a finite measured half-life. Two more are near zero (−2.08,
   −3.73). The three worst are exactly the three with the highest clash terms;
   the two with the lowest clashes give plausible −13.4 and −13.2.
3. **Repair strips the penalty without changing the packing.** Van der Waals
   clashes fall 8.50 → 3.01 and intra-HLA clashes 27.18 → 12.46, interface
   clashing residues go 1–2 → 0 in every case, while the genuine
   `van_der_waals` term barely moves per structure (−16.63 → −15.89,
   −19.18 → −19.75).

The predeclared plausibility window in section 7 decides it cleanly, on a
criterion fixed before any number existed: unrepaired energies span −13.4 to
+0.7 and fall **outside** the −10 to −30 kcal/mol window; repaired energies
span −16.1 to −24.2 and fall **inside** it.

**Section 4's decision rule is therefore satisfied.** Repair changes the
ranking — `repair_preserves_order` is `False` on both workspaces, and the
Spearman between the two orderings is **0.10** on the `a-cheparukhin` five.
Whether it fits budget is R4.

Caveat, plainly: the paired comparison is 10 structures, 5 per workspace, all
`HLA-A*01:01`. n=5 cannot reach significance — the smallest attainable
two-sided p is 0.017. A 48-structure, 48-allele paired pilot was launched to
remove this limitation and was **aborted before its repair arm finished**, for
a reason unrelated to the science. Its unrepaired arm completed; see R3.

### R3. Runtime, measured, and a contention factor that corrects the forecast

| Quantity | `a-cheparukhin` | `colleague` |
|---|---:|---:|
| Unrepaired, 5 procs on 6 cores | **3.62 s** | **3.41 s** |
| Repaired, 5 procs on 6 cores | **256.95 s** | **235.01 s** |
| Repair runtime multiple | **71.0×** | **68.9×** |

**At the production container shape the unrepaired arm costs 6.91 s of process
time per structure, not 3.62 s** — a measured **1.91×** contention penalty from
running 32 FoldX processes on 32 reserved cores with no headroom, which a
5-process smoke on 6 cores cannot see. Measured on 48 structures over 48
alleles, 48/48 ok, 10.4 s wall.

Any forecast built on the 5-process figure is optimistic by roughly that
factor. **The repair arm's own contention factor was not measured** — the gap
the aborted pilot would have closed — so the repaired forecast in R4 applies
the unrepaired arm's 1.91× and is labelled derived accordingly.

**Memory, measured, and the risk it retires:** **107 MB peak RSS per FoldX
process, including RepairPDB.** 32 processes need about 3.4 GiB against 16 GiB
reserved — 4.7× headroom. The concern that 0.5 GiB per process would OOM the
repair arm is closed.

### R4. Forecast, at the repo's metered rates

`$0.04730`/core-hour and `$0.00800`/GiB-hour from
`reports/ectodomain_rates.json`. Never published list rates.

| Arm | Shape | Wall | Core-hours | Cost | Basis |
|---|---|---:|---:|---:|---|
| Unrepaired | 25×32c/32p | 0.09 h | 74 | **$3.80** | **measured** 32-proc |
| Unrepaired | 40×64c/64p | 0.05 h | 118 | **$5.82** | **measured** 32-proc |
| Repaired | 25×32c/32p | 4.62 h | 3,695 | **$189.54** | derived, 1.91× applied |
| Repaired | 40×64c/64p | 1.46 h | 3,739 | **$184.32** | derived, 1.91× applied |

Two things worth stating because they are easy to get backwards:

- **Parallelism buys wall time, not money.** Cost is total work in core-hours,
  set by 28,166 structures times single-threaded FoldX time. Going from 25×32c
  to 40×64c cuts wall time 4.62 h → 1.46 h and leaves cost essentially flat;
  widening further *raises* it, because Modal bills the cores a container
  reserves for as long as it lives and each extra container adds its own
  startup.
- The repaired arm at ~$185 is **about 48% of the ~$387 remaining**.

### R5. Spent so far

About **$0.33**, derived from observed wall times at the metered rates above,
not read from a billing dashboard: two smoke runs at roughly $0.03 each
(6 cores / 12 GiB, both arms plus the binary probe and a fold listing) and the
aborted 48-structure pilot at roughly $0.27 (32 cores / 16 GiB, ~10 minutes
before abort). An estimate, with its method attached.

### R6. Two defects found before they could cost anything

Both would only ever have surfaced at the worst moment.

- **`score` wrote one output filename for both arms.** `--repair` changed the
  column but not the path, so running both arms — which is the plan — would
  have had the $3.80 unrepaired run **silently overwrite** the ~$185 repaired
  one, leaving a well-formed CSV behind. The repair setting is now in the
  filename, with a regression test.
- **`scripts/foldx_concat.py` did not exist**, although section 2 above and
  `score`'s own closing message both direct the reader to it. Written, with the
  asserts section 2 specifies (28,166 rows, zero duplicate `(allele, peptide)`,
  exact cohort coverage, one binary SHA and one repair setting across both
  halves), and verified end to end on synthetic full halves.

Also confirmed: **`--dry-run` is not free.** It makes no worker call — an AST
test proves no `.remote`/`.map`/`.starmap` precedes its return guard — but
`modal run` builds and validates the 87 MB FoldX image layer before the
entrypoint executes. The genuinely free path is `scripts/foldx_forecast.py`,
which never imports `modal`. Every forecast above came from it.

### R7. The awkward cases

Section 6's six alleles are forced into the pilot's selection rather than
sampled, by `pick_pilot_folds`, which also spreads over alleles
deterministically under seed `20261004` — the metric is a per-allele Spearman,
so a single-allele pilot cannot inform it. All six were present in the
48-structure selection and all converted and scored without error in the
unrepaired arm.

One correction to a detail that would have failed silently: the cohort spells
these alleles **`HLA-B*14:01(C67S)`**, with the suffix. An `isin` against the
bare `HLA-B*14:01` matches nothing and reports the engineered constructs as
absent. With the suffix the counts reproduce section 6 exactly — **1,135 C67S
rows** and 1,103 borrowed-alpha3 rows.

Per-allele energies for these six await the production pass.

### R8. What would close this stage

1. The repair arm's contention factor, from a ~32-structure paired pilot at the
   production shape. About **$0.25**. Converts R4's derived repaired figure
   into a measured one.
2. The production pass, both arms, then `scripts/foldx_concat.py`. About
   **$190** at 40×64c with ~1.5 h wall, on the derived figure.
3. Arms A and B under section 5's protocol, which costs nothing but local CPU.

Step 2 is the decision, and it is not only a budget question. Two independent
results now bound what it can return, both measured after this protocol was
written:

- **Wet-lab affinity used directly ranks stability at ρ 0.580** against the
  sequence baseline's 0.693 (section 1 above). The thermodynamic quantity is
  below the baseline standing alone.
- **Stage 5's structural arm is a conclusive negative on these same Boltz-2
  poses**: seq + geometry + confidence scores **−0.0715 [−0.1228, −0.0251]**
  against the baseline on validation, interval entirely below zero, and a
  `seq_only` control through the identical pipeline beats it at all six L2
  values tested — so the loss is the features, not the tuning.

FoldX is a genuinely different quantity from pLDDT and PAE, so section 1's open
question — whether an explicit energy decomposition adds anything on top of a
trained sequence model — remains formally open and is bounded by neither
figure. But these are two independent failures to extract an increment from
these structures, and the cost of a third attempt rose rather than fell once
contention was measured. R2 is a real and reportable result about empirical
energy functions on predicted models, and it is already in hand.
