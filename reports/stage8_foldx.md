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

*Empty by design. Nothing above this line changes once it is filled.*
