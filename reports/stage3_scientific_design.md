# Stage 3 scientific design: frozen ESM-2 representations

Design document. No source code is modified by this file, and no result is
reported in it. Written against the repository at `main` (`3576c02`), after
stages 1, 2, 2b, 2c, 3c and 4 landed.

---

## 1. Existing evidence

### 1.1 What the repository establishes

**The evaluation is frozen and the splits are reproducible.** `data/splits.csv`
regenerates byte for byte; peptides are grouped by single-linkage Hamming ≤ 3 so
no held-out peptide sits within 3 substitutions of a training peptide; 51 guards
in `tests/test_contract.py` hold the contract. The primary metric (median
per-allele Spearman), the eligible-allele rule (`MIN_ROWS_BY_SPLIT`: val ≥ 20 →
68 alleles, test ≥ 50 → 67), the constant-prediction convention, the paired
cluster bootstrap (2,000 resamples, seed 20261003) and the 0.05 worthwhile-gain
bar were all fixed before any model existed.

**A supervised sequence baseline exists and is strong.** One-hot peptide ⊗ 34
contact residues, MLP (256, 64): **0.610** validation median per-allele
Spearman, single network. The 30-network ensemble — 5 CV folds × 2 encodings ×
3 seeds — reaches **0.693**. Decomposition from `reports/stage2_baselines.md`:
the HLA input is worth +0.404 [+0.300, +0.474] over peptide alone;
non-linearity is worth +0.331 [+0.256, +0.414] over ridge on identical
features; ensembling alone is worth **+0.093 mean SCC** from no new
information.

**Two label-supplementation routes are closed, with bounds.** Stage 2b
(weak-binder augmentation): eight paired intervals, best arm +0.024, every
CI crosses zero and excludes 0.05. Stage 2c (auxiliary affinity head): twenty
paired comparisons, largest +0.034, every CI crosses zero with every upper
bound below 0.05 — and the mechanism is identified, since measured affinity used
directly as a stability predictor ranks at ρ 0.580, *below* the 0.610 the
stability labels already deliver. These are bounded negatives, not inconclusive
ones.

**External transfer is demonstrated once, off-training.** Stage 3c: among 140
overlapping pairs, eluted ligands show ~7× higher median half-life (7.90 h vs
1.10 h), common-language effect size 0.78, p = 6.0 × 10⁻³¹.

**Known contamination is disclosed and bounded.** One diagnostic
(`compare_to_paper.py` at `24dfaa6`) pulled 585 frozen test rows into an
evaluation set; eight panel aggregates were observed, and the +0.018 grouping
attribution it produced has been **retracted** (measured properly: −0.005
[−0.035, +0.046]). No model that will be scored at stage 6 saw a test row.

### 1.2 What the repository does not establish

- **Nothing about foundation-model representations.** No ESM arm exists on
  `main`. The question the track asks is, at this commit, untested.
- **Nothing about unseen alleles.** All 75 alleles appear in `train`.
  `dist_to_train` is *peptide* Hamming distance (d = 4 for 57.8% of test rows,
  d ≥ 5 for 42.2%), not allele novelty. Every result above is scoped to new
  peptides on familiar grooves.
- **No noise ceiling.** `reports/audit_summary.md` records that the file has no
  replicates, so the irreducible error is unknown and no result can currently be
  called "close to the ceiling".
- **Nothing about representation quality.** Stages 2b and 2c supplement
  *labels*. Their failure says the model is not label-starved in a way more rows
  of a correlated target can fix; it says nothing about whether a different
  encoding of the same inputs would help.
- **MAE is not interpretable at the floor.** 20.2% of labels sit at the assay
  detection limit, and error against a censored value has uncertain sign.
  Rank metrics lead; MAE is reported, not interpreted.
- **The test split is unscored** apart from the disclosed exposure, and stays
  that way until stage 6.

### 1.3 Prior art inside this project, not on `main`

A preliminary ESM-2 35M implementation exists on an unmerged branch
(`stage3-esm`). It is named here as a design input, not as evidence: its
headline is **not trustworthy** because the peptide block was PCA-reduced to 256
components before fitting, and an independent implementation of the same arm at
full width scored far higher. That discrepancy is the single strongest reason
the design below treats dimensionality reduction as a variable to be tested
rather than a preprocessing default. Any number from that branch must be
re-established under this design before it is quoted.

---

## 2. Stage 3 hypothesis

**H1 (primary).** Frozen ESM-2 representations of the peptide and the HLA
contact residues carry information about dissociation half-life that the one-hot
peptide ⊗ pseudosequence encoding does not, such that a small head trained on
baseline features **plus** ESM features exceeds the 30-network baseline ensemble
by Δ median per-allele Spearman ≥ 0.05, with the paired 95% CI excluding zero.

**H0.** The ESM arm does not clear the bar. Per `EVALUATION.md`, a CI that
crosses zero is inconclusive; a CI excluding zero with an upper bound below 0.05
is a *bounded negative*, which is the outcome stages 2b and 2c produced and a
legitimate result here.

**Directional prior, stated in advance.** Two facts argue against H1. The assay
panel was pre-selected by predicted affinity, so canonical anchor-motif
knowledge — the part of protein-sequence knowledge a language model holds most
of — is largely already spent in this dataset. And embedding peptide and HLA
separately leaves every peptide×pocket interaction to be learned by the head
from 15,773 fit rows, which is the same burden the one-hot arm carries without
the extra 4,000 columns. Stating the prior before the run is what makes a
negative result credible rather than a post-hoc rationalisation.

**Secondary question (H2), answerable at no extra fitting cost.** Does a
middle layer beat the final layer? ESM-2's last layer is optimised for masked
token prediction, not for interaction. H2 is tested only if the layer gap
exceeds the seed spread (§8); otherwise it is reported as unresolved.

---

## 3. Exact experiment matrix

Every arm uses the same frozen splits, the same fold assignment, the same seeds,
the same head family, the same tuning budget and the same ensemble size. The
only variable is the feature block.

### 3.1 Arms

| # | Arm | Feature block | Purpose |
|---|---|---|---|
| A0 | `seq_ensemble_pep_pseudo` | one-hot peptide + 34 contact residues (860) | the comparator; **must be regenerated**, see §6.4 |
| A1 | `esm_alone` | ESM peptide 9×d + ESM contacts (§4) | does ESM carry stability signal at all |
| A2 | `esm_stacked` | A0 block ⊕ A1 block | **the arm H1 is about**: does ESM add over the baseline |
| A3 | `esm_peptide_only` | ESM peptide 9×d only | which half of the embedding carries anything |
| A4 | `esm_hla_only` | ESM contacts only | as above; expected near-floor, 75 distinct values |
| C1 | `random_matched` | Gaussian block at A1's width, stacked on A0 | what N uninformative columns cost under a fixed budget |
| C2 | `shuffled_esm` | A1 block with the peptide→embedding map permuted | real-minus-shuffled = information ESM contributes |
| C3 | `onehot_pca_matched` | A0 features reduced to A1's width by the same pipeline | positive control: is the pipeline lossy? |

A1–A4 run at **two layers** (middle, final). C1–C3 run at the width A1 settles
on. C3 is the control that decides whether a reduction, if used, is doing
damage — it is the one experiment that can invalidate the whole arm, so it runs
in the first batch, not last.

### 3.2 Why the controls are not optional

A1 and A2 cannot be interpreted alone. If A2 scores below A0, three mechanisms
produce that observation: ESM features are uninformative (the finding), the
extra columns displace capacity the one-hot block was using (C1 measures this),
or the representation pipeline destroyed signal (C3 measures this). Reporting
A2 without C1 and C3 would assert the first while having measured none of them.

### 3.3 Cross-embedding arms: deferred to 3e, not dropped

The task brief for this stage specifies peptide and HLA **embedded separately**,
so arms A1–A4 are what §3.1 must contain. But a separate brief in this project
(`claude_science_esm2.md`, Step 1b) names *cross-embedding* features as its key
contribution and ranks two of them above separate pair embeddings. Omitting them
without saying so would misrepresent the stage. They are specified here as a
defined follow-on track, **3e**, gated on the A-arm result.

| # | Arm | Construction | What only this can show |
|---|---|---|---|
| X1 | `chimeric` | `peptide + linker + α1α2` as **one** sequence; keep the 9 peptide-position states | ESM's own attention makes the peptide representation allele-dependent |
| X2 | `dot_product` | scaled `peptide @ contacts.T` from the **separate** A1 embeddings → 9×34 | explicit position×pocket compatibility, no new forward passes |
| X3 | `cross_attention` | 9 peptide positions as queries over 34 contact residues, one block | coupling learned by the head rather than fixed |
| X0 | `blosum_cross` | BLOSUM62 score for each (peptide position, contact residue) pair → 9×34 | **the control for X1–X3**: no network, no pretraining |

**Why gated rather than run in parallel.** All three consume the same
representation whose validity §4.3 is still testing. If the positive control C3
shows the pipeline is lossy, every cross arm built on it inherits the defect, so
running them first would spend the most compute on the least interpretable
result.

**Pre-registered interpretation.** X0 is a deterministic function of inputs the
baseline already receives, so a gain from it cannot mean new information — it
would mean the MLP was not extracting the peptide×pocket interaction efficiently
from one-hot input, which indicts the *architecture*, not the representation.
X1–X3 are therefore judged against X0, not only against A0: an ESM cross arm
that fails to beat a substitution matrix has not demonstrated that pretraining
contributes anything to the coupling. X3 additionally requires a
**parameter-identical ablation** — the same module with attention replaced by a
uniform average over the 34 contacts — so its gap isolates coupling from
capacity.

**Two construction caveats that must be stated wherever these are reported.** A
glycine linker does **not** signal a chain break: ESM-2 has no chain token, and
the chimera is read as one continuous protein, so X1 tests contextual mixing,
not a modelled two-chain complex. And a dot product on raw ESM vectors is
dominated by their large shared mean component — X2 requires the residue vectors
to be centred (ideally whitened) first, or the 9×34 matrix is near rank-1 and
encodes little beyond vector norms.

**Cost.** X2, X3 and X0 need no new forward passes and are head-level changes.
X1 is the expensive one: it requires one pass per *measured pair* rather than
per unique sequence — 28,166 passes of ~200 tokens against 5,708 short ones —
which is a ~40× increase in extraction cost and must be budgeted against §9
before it runs.

### 3.4 Ensembling

30 networks per arm: 5 CV folds × 2 variants × 3 seeds, where the two variants
are the two ESM layers for A1–A4, two independent draws for C1, two independent
permutations for C2, and the one-hot/BLOSUM encodings for C3 — mirroring how A0
gets its two-encoding diversity axis. Folds come from
`scripts.baseline_ensemble.cv_folds`, cut along whole Hamming ≤ 3 clusters.
**Ensemble both arms identically or neither**: ensembling is worth +0.093 mean
SCC from nothing, so an ensembled ESM arm against a single-network baseline, or
the reverse, would manufacture the result.

---

## 4. Representation definitions

Checkpoint: `facebook/esm2_t12_35M_UR50D` — 12 layers, hidden d = 480. Layers
kept: **6 (middle)** and **12 (final)**.

### 4.1 Peptide

Forward pass on the bare 9-mer. BOS and EOS stripped, so column *j* is residue
*j*. Representation is the **full per-residue block, 9 × 480 = 4,320 columns,
flattened, no pooling**. The plan requires per-position information be
preserved; position is most of the signal in peptide–HLA binding, and the one
place a mean would be actively wrong.

### 4.2 HLA

Forward pass on the supplied **182-residue domain** — not on the 34-residue
pseudosequence as a string, which would present ESM with non-contiguous residues
as though they were adjacent and invite the positional encoding to model an
alignment that does not exist. The 34 contact positions are then **sliced** from
the per-residue output (§5), giving 34 × 480 = 16,320 columns.

**The contacts are not mean-pooled.** The B-pocket and F-pocket residue sets are
disjoint and contact different peptide positions; a mean over all 34 averages
chemically distinct pockets into one vector. Pooling is instead included as an
explicit arm variant so the choice is measured rather than assumed.

### 4.3 Dimensionality — reduce only where reduction is provably lossless

Naively concatenated, A1 is 20,640 columns against 15,773 fit rows. The
resolution is asymmetric, and this is the design's main technical decision:

- **HLA block: PCA to 74 components.** There are only 75 unique domains, so the
  block has rank ≤ 75 and 74 components are **lossless by construction**, not an
  approximation. Verify empirically that retained variance = 1.000 and fail
  loudly otherwise.
- **Peptide block: no reduction in the primary arm.** 4,320 columns, used as-is.
  A lossy PCA here is precisely the step suspected of corrupting the earlier
  attempt (§1.3).

Primary A1 width is therefore **4,394**; A2 is 5,254.

**Reduction is then tested, not assumed**, as a secondary variant: peptide block
at 256 and 1,024 components alongside full width. If full width and 256
components agree within the seed spread, the reduction is harmless and may be
used for cheaper follow-ups; if they disagree, full width is the only admissible
representation and that fact is itself reportable.

### 4.4 Standardisation

ESM components are centred and scaled to unit variance per column using
**fit-fold rows only**. One-hot features are left raw, as `pepstab/features.py`
documents. In the stacked arm the two blocks keep their own treatment so the
baseline half is bit-comparable to A0.

### 4.5 Head

`pepstab.mlp.MLPRegressor` — the same class, loss, optimiser and stopping rule
as stage 2, so no part of the comparison is attributable to a different trainer.
Hidden sizes and L2 come from the grid in §7.

---

## 5. HLA indexing procedure

Slicing 34 positions out of a 182-residue embedding is the step where a silent
off-by-one would corrupt every downstream number while leaving all shapes valid.
It is therefore a **test**, executed before any embedding is sliced, with a hard
failure on mismatch.

**Procedure.**

1. Take the published NetMHCpan pseudosequence positions in HLA heavy-chain
   numbering: 7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81,
   84, 95, 97, 99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159, 163, 167,
   171.
2. Hypothesis: the supplied domain is residues 1–182 of the mature α1/α2, so
   position *p* sits at 0-based index *p* − 1.
3. Assert `hla_seq[p-1] == hla_pseudoseq[j]` for **all 75 alleles × all 34
   columns**. Raise and stop on any mismatch, naming the offending columns.
4. Record the verified index vector alongside the cached embeddings.

**Why the mapping is not inferred from the data.** Searching for domain columns
that reproduce each pseudosequence column is underdetermined: with only 75
alleles, several conserved columns match any given pseudosequence column by
chance. The published definition is the hypothesis and the data is the test —
inverting that would silently select a wrong but self-consistent mapping.

---

## 6. Leakage risks

### 6.1 Not a risk, and why

**Embedding extraction is label-free and split-blind.** ESM sees sequences only,
once per unique sequence, with no access to `thalf_hours` or to split
membership. Caching over all 5,633 peptides and 75 domains therefore leaks
nothing; computing the cache per split would cost 5× more and buy nothing.

### 6.2 Real risks and their controls

| Risk | Control |
|---|---|
| PCA fitted on held-out peptides | Fit on **training-split peptides only**; validation peptides are transformed, never fitted. The HLA PCA uses all 75 alleles, all of which are in `train`, so no held-out information enters. |
| Standardisation statistics from held-out rows | Mean and scale from **fit-fold rows only**, recomputed per fold. |
| Early stopping on the reported split | Stop on the inner CV fold from `cv_folds`, cut along whole peptide clusters. **Never stop on `val`.** This is the error that makes an arm look better than it is, and it is the specific criticism `EVALUATION.md` levels at the published NetMHCstabpan protocol. |
| Selection pressure on validation | Validation is used for selection *and* reporting, which is the repo's accepted design; the protection is that the **test split is scored once at stage 6** and the selection grid is fixed in advance (§7). |
| Peptide clusters straddling fit/dev | Inherited from `cv_folds`; assert minimum cross-fold peptide Hamming ≥ 4 before fitting. |
| Augmented rows leaking in | Stage 3 trains on measured rows only. No manifest from `data/augmentation/` enters any arm. |
| C67S constructs | Left in training, as in stage 2, so the arms stay comparable — but excluded from any claim, because they are the most heavily censored alleles in the panel and a ρ computed on them largely reads tied floor values. |

### 6.3 Test-set discipline

Stage 3 reads `split == "test"` **never**. Given the disclosed exposure in
`EVALUATION.md`, a second breach would end the single-use claim entirely.

### 6.4 The comparator must be rebuilt

`preds/` is gitignored, so `preds/seq_ensemble_pep_pseudo.csv` is absent from a
fresh clone. Every Δ in this stage is measured against it, so
`scripts/baseline_ensemble.py` runs **first**. Treat a regenerated value that
differs from the reported 0.693 by more than seed noise as a blocking
discrepancy, not a rounding difference.

---

## 7. Selection procedure

**Selection happens on validation and on nothing else. The test split is not
read at this stage.**

**The grid is fixed before the run**, and is exactly the grid stage 2 gave each
of its arms — hidden ∈ {(64,), (256, 64)} × L2 ∈ {1e-5, 1e-3}, 3 seeds. Stage 2
gave every arm four configurations; an ESM arm inheriting the baseline's
selected config would be an arm tuned for 860 one-hot columns being judged on
4,394 dense ones. That asymmetry biases **against** ESM, which in a study whose
likely outcome is negative is the dangerous direction: it would let a tuning
artefact be reported as a finding about foundation models.

**Order of selection**, each step on validation, each fixed before the next:

1. **Representation** — full-width vs 256 vs 1,024 peptide components (§4.3),
   at the stage-2 config, single fold, 3 seeds.
2. **Layer** — middle vs final, at the selected representation.
3. **Head** — the 4-point grid, at the selected representation and layer, on a
   single CV fold, scored as the mean over seeds.
4. **Ensemble** — the winner only, rebuilt at the full 30 networks, then scored
   once against A0 with the paired cluster bootstrap.

**Reporting rule.** Every arm in §3.1 is reported whether or not it was
selected, with its verdict from `describe_delta()`. Selection decides what the
headline arm is, not which numbers appear.

---

## 8. Seed procedure

Seeds **0, 1, 2** — the stage 2 set — for every arm, fold and variant.

**Required reporting.** Per-arm seed spread (max − min of member median ρ within
an arm/variant/fold), reported next to every point estimate.

**The decision rule, fixed in advance.** Stage 2 recorded a seed spread of
0.010–0.051 for the sequence arms. Any gap smaller than the measured spread of
the arms being compared is **not a result** and is reported as unresolved. This
applies to the layer comparison (H2) first, and to any arm-versus-arm difference
that is not accompanied by a paired CI.

**A diagnostic, not just a caveat.** If an ESM arm's seed spread materially
exceeds the baseline's, that is evidence the arm is fitting noise rather than
signal, and it should be reported as a finding in its own right — a feature
block that produces unstable models is unusable regardless of its mean.

**Ensembling does not substitute for this.** Averaging 30 networks hides seed
variance; the spread must be measured at member level, where it is visible.

---

## 9. Compute measurements

The track's question is whether foundation models **earn** their compute, so
cost is a reported quantity, not a footnote.

**Measured per arm:**

| Quantity | How |
|---|---|
| Embedding extraction, wall-clock | Timed separately for peptides and domains; recorded in the cache file with the checkpoint id |
| Unique sequences embedded | 5,633 peptides + 75 domains = 5,708 forward passes against 28,166 rows — the caching gain, stated explicitly |
| Cache size | Bytes on disk, and dtype |
| Reduction cost | PCA fit + transform seconds, with retained variance |
| Training | Seconds per network and the total for the 30-network ensemble |
| Inference | Seconds per 1,000 predictions, on the same hardware as the baseline |
| Hardware | CPU/GPU, core count, and whether a GPU was required at all |

**Reported as a Pareto comparison**, not a table of absolutes: accuracy against
total CPU-seconds to build one 30-network ensemble, baseline included as a point
on the same axes. The baseline's cost (73 s of fits, inference under 1 ms per
1,000 predictions) is the floor ESM must justify itself against.

**A machine-speed caveat belongs in the report.** Absolute timings are not
comparable across machines — the same stage 2 ensemble has been measured at 73 s
and at 19.6 min on different hardware. Report **ratios to the baseline measured
on the same machine in the same session**, with absolutes as context.

---

## 10. Criteria for moving beyond ESM-2 35M

A larger checkpoint is **not** the default next step. It is justified only if the
35M results identify representation capacity as the binding constraint. All
three must hold:

1. **The pipeline is exonerated.** C3 (positive control) reaches within one seed
   spread of A0 at matched width. If C3 collapses, the representation pipeline
   is the problem and must be fixed at 35M before any scale-up.
2. **ESM carries measurable signal at 35M.** A1 beats C2 (shuffled) by more than
   the seed spread, with the paired CI excluding zero. If real and shuffled
   embeddings are indistinguishable, the embedding is inert for this task and a
   larger one built the same way is a larger inert block.
3. **The trend points up, not flat.** Either the stacked arm A2 clears zero with
   a CI excluding it, or the middle/final layer comparison shows a gradient
   larger than the seed spread — some evidence that more or better
   representation moves the metric at all.

**Additional gate on cost.** The 650M checkpoint is ~19× the parameters. Before
running it, state the expected gain and the compute budget in advance, and
require that the projected gain still clears 0.05 — a gain of +0.02 at 19× the
cost answers the track's question in the negative regardless of its sign.

**Explicit anti-criterion.** "The 35M arm lost, so try a bigger one" is not a
justification. If the 35M arm loses because a wide dense block displaces
capacity the one-hot features were using (C1 positive), a 1,280-dimensional
block displaces more, and scaling makes the result worse while costing more. In
that case the correct next experiment is architectural — coupling the two
representations rather than concatenating them — not larger.

**Before a larger checkpoint, run 3e.** If the 35M separate-embedding arms fail
while the cross-embedding arms of §3.3 are untested, the open question is
whether concatenation wasted what the embedding holds — not whether the
embedding is too small. 3e costs no new forward passes for X0, X2 and X3, so it
is strictly cheaper than a scale-up and answers the prior question. A 650M run
proposed while 3e is outstanding is answering the second question first.

**If the gates fail**, the stage 3 conclusion is written as a bounded negative in
the form stages 2b and 2c used: the arm, the delta, the interval, the upper
bound against 0.05, and the compute it cost — scoped explicitly to new peptides
on alleles seen in training, frozen features, and this checkpoint.
