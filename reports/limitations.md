# Limitations register

Everything a reader should discount this submission by, collected in one place
and sourced. Companion to [`SUBMISSION.md`](SUBMISSION.md) and
[`compute_ledger.md`](compute_ledger.md).

This file exists because the brief asks for *"a watertight evaluation"* and
says *"negative results are just as good as positive results **where well
supported**"*. A negative result is only well supported if the reader can see
what would have had to be true for it to be wrong. That is what is below.

---

## 0. The distinction everything else depends on

Three different things get called "a negative result", and conflating them is
the single easiest way to overclaim. The predeclared reading in
[`EVALUATION.md`](../EVALUATION.md) keeps them apart:

| Paired 95% CI for Δ median per-allele ρ | Verdict | Plain English |
|---|---|---|
| upper < 0 | **Worse** | The addition actively hurts. |
| lower > 0.05 | **Meets the bar** | Worth the compute. |
| Excludes 0, upper < 0.05 | Real but too small to matter | There is an effect; it is not worth paying for. |
| Excludes 0, straddles 0.05 | Real, size unresolved | Something is there. We cannot say how much. |
| **Crosses 0, upper < 0.05** | **Inconclusive, *and* rules out a worthwhile gain** | We cannot say it helps at all; we *can* say it does not help by the margin that would matter. |
| Crosses 0, upper ≥ 0.05 | **Inconclusive** | We learned nothing. |

Rows 5 and 6 look alike and are not. **This project's negative results are all
row 5, not row 6, and not "the effect is zero".** Stage 2b's best arm is
+0.024 [−0.026, +0.048]: the data are consistent with a real +0.04 improvement.
What is ruled out is +0.05. Saying "augmentation does nothing" would be a
stronger claim than the evidence supports, and it is not the claim made.

The 0.05 bar is itself a property of the test set, not a preference: within-allele
label permutation gives 0.000 ± 0.018, and a paired cluster bootstrap under a
true zero difference has a 95% CI half-width of 0.037–0.053. Below ~0.05 this
test set cannot separate a difference from noise
(`audit_summary.md` §6). **A larger benchmark would move this bar down, and some
of what we call "ruled out" would become detectable.**

---

## 1. The label

### 1.1 One fifth of the labels are not measurements

**5,679 rows (20.2%) sit at exactly 0 hours, and the evidence says this is an
assay detection floor, not a measured zero** (`audit_summary.md` §3). Three
independent lines: a trough at 0.1 h holding 443 observations where a fit over
0.2–1.0 h predicts ~1,651; per-allele zero share anticorrelating with that
allele's median non-zero label (Spearman −0.608, p = 3.7 × 10⁻⁸); and zeros
spread across peptides rather than concentrated.

**What `log1p` fixes.** It is defined at zero where `log` is not, so the floor
rows can be kept rather than discarded, and it compresses a target that spans
0–256.7 h with skewness 5.86 down to skewness 0.91.

**What `log1p` does not fix.** It still treats a censored observation as an
exact value. The honest statement for a floor row is "t½ < some detection
limit", and `log1p` records "t½ = 0". Two consequences:

- **MAE at the floor is uninterpretable in sign.** It measures error against the
  *recorded* label. A latent 0.05 h recorded as 0: predicting 0.10 is charged 2×
  too much, predicting 0.01 is charged 4× too little. Floor MAE is therefore
  neither an upper nor a lower bound on true error (`EVALUATION.md`). This is
  why MAE is secondary and the primary metric is a rank correlation.
- **Rank metrics handle it but do not escape it.** Spearman with average ranks
  treats the 20.2% as one large tied block, which is the right treatment of "all
  of these are below the limit". It cannot recover the ordering *within* that
  block, which is real information the assay did not capture.

**The principled alternative is a censored (Tobit-style) likelihood**, which
models P(t½ < floor) directly. It was recorded as out of scope for a 16-hour
build, not as unnecessary (`HACKATHON_PLAN.md`, out-of-scope register). **It has
now been built and tested, and it is worse** — Δ median per-allele Spearman
**−0.0414, 95% CI [−0.0780, −0.0062]**, an interval entirely below zero, and it
also loses on its own held-out objective (censored NLL 1.0765 vs 1.0452). It
does deliver the calibration it promised (predictive mass below the limit
7.0% → 16.8% against an observed 19.6%; ECE 0.0561 → 0.0422) and does **not**
improve floor discrimination at all (AUROC 0.8862 vs 0.8853), exactly as
predeclared, because ranking inside the tied floor block is unidentifiable under
either objective. The result is flat across detection limits 0.05–0.30 h (median
ρ spans 0.0095), so the free parameter is not driving it. **So the `log1p`
compromise is not merely a 16-hour substitute that was never checked — it was
checked, and it won on the primary metric.** The likely cause of the loss is the
stopping rule rather than the loss itself (the censored arm's dev objective
turns over at epoch ~10 against the MSE arm's ~27, so it is undertrained); that
is reported as a **control without an interval**, not as a result. Full detail
in `stage7_censored.md`; summary at SUBMISSION §4.7. This was previously tracked
as stage 7a — `reports/stage7_censored.md` predeclares the
loss, the detection threshold `c = log1p(0.1)` (justified from the 0.1 h
reporting grid and the single row between 0 and 0.1 h, never from a validation
score), a five-point `c_hours` sensitivity sweep including a deliberate
over-censoring control at 0.30, and the prediction heads — all written before
any censored model was fitted, with §6 still marked "pending". **Hole S7a.** The
predeclared expectation is worth repeating here because it shapes how the result
should be read: a censored loss should improve calibration near the floor, not
ranking, since order *within* the tied floor block is unidentifiable under either
objective. The same
problem bit the stage 2b affinity predictor from the other direction: 26.8% of
its training labels sit exactly on the 20,000 nM boundary, an MSE regressor
shrinks them toward the conditional mean, and weak-call recall collapses to
0.263 at precision 0.823 (`stage2b_augmentation.md`). A censored likelihood is
the recorded fix there too.

### 1.2 No replicates, so no noise ceiling

The supplied file has one value per pair. 98.74% of labels sit on a 0.1-hour
grid; the 354 finer values concentrate in the most-measured alleles and are
likely replicate averages, but the replicates themselves are not supplied
(`audit_summary.md` §3). **We cannot say how much of the residual error is
irreducible assay noise**, so we cannot say how close any model is to the best
achievable score. Every "ruled out" in this project is ruled out against the
0.05 bar, never against a noise ceiling.

### 1.3 Stability is not affinity, and neither is the unbinding barrier

The target is residence time once bound, which is largely a dissociation rate.
Binding affinity is an equilibrium quantity combining on- and off-rates. Neither
label measures the energy barrier to unbinding (ΔG‡) directly. This is the
stated reason FoldX, Rosetta and any empirical ΔG layer are out of scope: they
estimate equilibrium ΔG, a known mismatch with a kinetic label
(`HACKATHON_PLAN.md`). No mechanism claim about unbinding is supported by
improved prediction alone.

---

## 2. The dataset

### 2.1 The peptide panels are allele-confounded by design

**Each allele was assayed on its own peptide panel.** The measured
allele × peptide grid is only 6.7% full; each peptide was tested on a median of
4 alleles and 1,692 peptides appear on a single allele
(`audit_summary.md` §4). This is a property of how the data were generated, and
it has three consequences the submission cannot design away:

1. **It broke the first split.** A water-fill ranked by *absolute* deficit
   degenerates under allele-specific panels: once the three deficits equalise,
   clusters cycle round-robin and heavy clusters all land in whichever split led
   early. That left `HLA-B*42:01`, `HLA-B*51:01` and `HLA-B*81:01` (~350 pairs
   each) with **zero test rows**. Ranking by deficit relative to target fixed it
   at no cost to the 70/10/20 totals (`audit_summary.md` §5).
2. **It is why the allele-held-out evaluation is not the headline — but the
   mechanism we originally wrote down was wrong, and has been corrected.**
   `HACKATHON_PLAN.md` claimed that holding out an allele simultaneously holds
   out its peptide panel, making an allele-distance effect inseparable from a
   panel effect. **Measured, that is not what happens** (`stage7_allele_holdout.md`
   §4): the median peptide is assayed on **4 alleles**, allele-exclusive
   peptides are **6.0% of rows**, and for the median eligible allele **100%** of
   its rows carry a peptide also seen on another allele (minimum 0.565). Peptide
   overlap is uncorrelated with distance (−0.067, p = 0.59) and with per-allele
   performance (+0.034, p = 0.79). So leave-allele-out here is largely *seen
   peptide, unseen allotype* — **easier** than the frozen split in that respect,
   which is exactly why its absolute numbers must never be quoted beside a
   frozen-split number.

   **What the confound actually is: panel composition.** Each allele's panel was
   partly selected by predicted binding affinity for *that* allele, and distant
   alleles carry weaker-binding, more heavily censored panels —
   Spearman(distance, per-allele zero share) = **+0.251, p = 0.039**. Partialling
   the two apart leaves distance at **−0.604** (p = 4.9 × 10⁻⁸) against −0.636
   raw, and zero share at −0.336 (p = 5.1 × 10⁻³). **Both carry independent
   signal: the confound is attenuated, not eliminated**, and zero share is only
   one proxy — anchor-motif composition, peptide diversity and the
   affinity-prediction step that chose each panel are unmeasured. The single
   worst fold makes it concrete: `HLA-B*39:06(C67S)` sits at **d = 3** with a
   **91.8% floor panel**, so it scores badly because of its panel, not its
   distance.

   The plan's other three reasons stand unchanged, except that reason 4
   (skewed allele coverage) describes test-split counts and **does not bind on
   this axis** — in-scope rows per allele jump 30 → 177 with nothing between, so
   68 alleles stratify comfortably into 25/23/20.
3. **Per-allele medians are over an unbalanced panel.** 8 of 75 alleles hold
   fewer than 50 test rows and are excluded from the test panel; 7 of those hold
   ≤ 32 pairs in the entire dataset. Only `HLA-B*40:02` (19 pairs) is missing
   from a split entirely. The eligible panel covers 98.8% of test rows, but
   per-allele claims are bounded to the 67 alleles in it (`audit_summary.md` §5).

### 2.2 The panel was pre-selected by predicted affinity

Rasmussen et al. selected the assayed peptides partly on *predicted* binding
affinity. The peptide diversity in this dataset is therefore narrow by
construction: it is a sample of peptides someone's model already thought would
bind. Two downstream effects, both measured:

- **Real zero-stability peptides have normal anchors.** Validation rows at the
  floor and above it look the same at the anchor positions (84.2% vs 84.0%
  hydrophobic at PΩ, `stage2b_augmentation.md`). What separates a 0-hour peptide
  from a 38-hour one in this dataset is *not* whether it can enter the groove.
  A model trained here learns to rank within the set of peptides that bind, and
  that is the question this benchmark measures.
- **It is why stage 2c was worth running and why it came out flat.** Broader
  affinity corpora cover far more peptides; the hypothesis was that they would
  supply diversity the stability panel lacks. The measured ceiling says the
  signal is redundant with what stability labels already teach
  (`stage2c_affinity.md`).

**Any broader biological or clinical claim needs evidence outside this file.**

### 2.3 Two alleles are indistinguishable to a pseudosequence model

`HLA-B*14:01(C67S)` and `HLA-B*14:02(C67S)` share one 34-residue contact
pseudosequence (756 rows, 2.7%), so a pseudosequence-only model must predict
them identically — verified: max absolute difference 0.00 across 41 shared
validation peptides, while their measured labels correlate at only ρ = 0.708.
The full-domain arm separates them (max difference 0.31)
(`stage2_baselines.md`). **The collision caps cross-allele discrimination, not
within-allele ranking**, which is why it barely moves the primary metric
(pseudo 0.487/0.613 vs domain 0.508/0.498). It does *not* reach the structural
arm: the two 182-residue domains differ at position 10, outside the contact set,
so they receive distinct MSAs (`stage4b1_msa_cache.md`).

### 2.4 Three C67S constructs have no matched wild type

The C67S substitution replaces a free cysteine that caused aggregation in the
assay. It sits at one of the 34 contact positions, so wild-type affinity
describes a different groove — 245 rows are excluded from the stage 2c
auxiliary labels for exactly this reason (`stage2c_affinity.md`). No matched
wild-type alleles exist in the dataset, so **the C67S effect cannot be measured
here**, only observed: these three alleles carry 74.9–92.1% zero labels against
17.4% for the 72 natural alleles (`audit_summary.md` §2).

---

## 3. The split

### 3.1 Grouping reduces label leakage; it does not remove it

Peptides are grouped by single-linkage clustering at Hamming ≤ 3, every cluster
wholly inside one split, so **no validation or test peptide is within 3
substitutions of any training peptide**. But same-allele label similarity decays
smoothly with distance rather than in a step (`HACKATHON_PLAN.md` stage 1):

| Distance | Same-allele comparisons | Spearman |
|---:|---:|---:|
| 1 | 513 | 0.725 |
| 4 (the minimum across splits) | 11,275 | **0.592** |
| 5 | 91,636 | 0.512 |
| unrelated background | 81,455 | 0.302 |

**At d=4 the residual similarity is 0.592 — roughly halfway between unrelated
pairs and near-duplicates.** No feasible threshold eliminates it: single-linkage
percolates sharply between 3 and 4, where one component swallows 2,807 peptides
(54.8% of pairs) and balanced splitting becomes impossible. Hamming ≤ 3 is the
most conservative threshold that still permits a 70/10/20 split.

The stage 6 mitigation is to **stratify test metrics by distance to training**
(d=4, 57.8% of rows; d≥5, 42.2%), scored on the same 65 alleles so the
comparison measures distance and not panel composition. If a model is exploiting
residual similarity, d=4 will score visibly higher than d≥5. A d≥6 stratum is
not viable: only 12 test rows reach it.

### 3.2 The split cannot answer the mutant-ranking question

Because no two peptides within 3 substitutions straddle a split, **the benchmark
measures generalisation to distant sequences and cannot assess mutant ranking** —
scoring point mutants of a known binder, which is often the practically relevant
design question. The partial recovery is a nested cross-validation on the d≤2
clusters that live entirely inside the training split
(`EVALUATION.md`, stage 6). That is a different evaluation on a different panel,
and it is not the headline number.

### 3.3 The split is model-independent on purpose, which costs something

Plain Hamming was chosen over BLOSUM-weighted or embedding-based distance
because an ESM-2 distance would make the split depend on the model under test at
stage 3. At a matched threshold BLOSUM62 reproduces essentially the same
partition (5,493 vs 5,494 clusters at Hamming ≤ 1) while adding a threshold to
defend (`HACKATHON_PLAN.md` stage 1). The cost is that Hamming treats a
conservative substitution and a radical one identically.

### 3.4 Disclosed test exposure

One superseded diagnostic breached the single-use test rule. It re-partitioned
the whole dataset by peptide identity, pulling **3,350 frozen test rows into
fitting, 448 into early stopping and 585 into the scored set**. What was
observed is eight panel aggregates over 68 alleles from two models, on an
evaluation set that was 21% test rows; no per-row prediction, per-allele score
or individual label was inspected. Its conclusion — a +0.018 attribution to
split grouping — **has been retracted**; measured properly it is −0.005 [−0.035,
+0.046] on the median (`EVALUATION.md`, "Disclosed test exposure").

No model that will be scored at stage 6 saw a test row. We judge the split still
usable and score it once. **A reader who disagrees has the eight numbers and can
discount accordingly** — which is the point of disclosing rather than quietly
fixing it.

---

## 4. The comparators

### 4.2 NetMHCstabpan is calibration, never a comparator

**NetMHCstabpan was trained on all 28,166 rows, including every peptide in our
test split.** Any score it produces on our data is memorisation. The brief itself
flags this: *"a competitive baseline for this task, is trained on the entirety
of this dataset, hence is unfair to use as a direct comparator."* **There is no
valid "beat NetMHCstabpan" result available from this dataset at any stage.**

Its published 0.69 mean per-allotype SCC is a 5-fold CV score on its own
training data. Our 30-network ensemble reaches 0.645 mean SCC / 0.693 median
per-allele ρ on data it has never seen. **These are not the same quantity and
the 0.045 gap is not a model-quality comparison.** The differences that matter
(`stage2_baselines.md`):

| Axis | Rasmussen et al. | Ours |
|---|---|---|
| Training rows | **103,166** — 28,166 measured + 75,000 assumed-zero weak binders (73% of the set) | 19,716 measured |
| Encodings | BLOSUM50 + smoothed sparse (0.9/0.05) | BLOSUM62 + plain one-hot |
| Architecture | one hidden layer, 40/50/60 units | two layers, 256×64 |
| Target | `2^(−t0/th)` | `log1p(th)` |
| Scored on | the same 1/5 fold each network stopped on | a split never seen during fitting |

Their published **0.676 PCC must never be quoted as a bar**: it was computed on
data padded with ~75,000 easy synthetic negatives, which inflates correlation
substantially (`EVALUATION.md`, Caveats). Matching their network count is not
matching their method.

**What the gap can and cannot be attributed to.** Split grouping shows *no
conclusive advantage* at equal row count — Δ median ρ −0.005 [−0.035, +0.046],
Δ mean ρ −0.000 [−0.024, +0.024]. Both intervals straddle zero and both still
admit a modest positive effect, including the retracted +0.018. Read as "no
conclusive advantage found", **not** "no effect". What it does bound is grouping
below ~0.024 on the mean, about half the 0.045 gap. The remainder is consistent
with the 5.2× training set, richer architecture diversity, and their scoring on
the stopping fold — and we cannot separate those here.

### 4.3 Validation scores are selection scores

Every number quoted from stages 2, 2b and 2c is a **validation** score, and each
arm's configuration was chosen on the number reported beside it. They are
optimistic as held-out estimates. They remain valid for *comparing* arms under
the same budget, which is what they are for. λ in stage 2c was swept on
validation, which is why the whole sweep is tabulated rather than its maximum:
the peak one-hot cell beats the control by 0.004, a sixth of the seed spread
(`stage2c_affinity.md`).

### 4.4 Seed spread sets a floor on what counts as a difference

Seed-to-seed spread for the selected MLP configs is **0.010–0.051**
(`stage2_baselines.md`). Gaps under ~0.05 are within seed noise, which is the
same order as the predeclared bar — both reflect what this dataset can resolve.
**Compare seed means, never single seeds.**

### 4.5 Arms must be ensembled identically or the comparison is manufactured

**Ensembling alone is worth +0.090 mean SCC from no new information**
(`stage2_baselines.md`). An ensembled ESM-2 arm compared against a
single-network sequence arm would produce a result out of thin air. The rule is
`cv_folds()` with matched member count for every arm, or none. The stage 3
comparator is the 30-network ensemble at median per-allele ρ 0.693, not the
single network at 0.610.

---

## 5. What the negative results do and do not establish

### 5.1 Weak-binder augmentation (stage 2b)

**Established:** across eight paired comparisons, every 95% CI crosses zero and
excludes 0.05. Best arm +0.024 [−0.026, +0.048]. The source comparison —
measured-affinity vs predicted-affinity negatives at matched per-allele counts —
is −0.002 (w=0.1) and −0.017 (w=0.25), inconclusive and bounded below the bar
in both directions.

**Bounded to:** 0.16 augmented rows per measured row against Rasmussen's ~2.7, a
16× volume difference. **Volume is untested.** The anchor analysis suggests
volume would not help — measured weak binders have near-canonical anchors (77.6%
hydrophobic at PΩ) and the baseline already scores them near the floor, while
predicted ones from random peptides have the *wrong* anchors (49.1%) and the
baseline scores them *below* it — but that is a prediction, not a measurement.
Also bounded to one architecture, two weights (0.1, 0.25, predeclared), single
networks rather than the ensemble, and a predicted pool that is the
high-confidence tail rather than a random sample of peptide space.

### 5.2 Auxiliary affinity training (stage 2c)

**Established:** across 20 paired comparisons — 5 λ × 2 encodings × {single
network, 30-network ensemble}, plus a censoring-robustness variant — every CI
crosses zero and every upper bound sits below 0.05, largest +0.034. The
comparison is controlled by construction: at λ=0 the multi-task network is
**bit-identical** to the stage 2 baseline, verified on the real 860-column
feature grid and on all 2,817 validation predictions. The null is clean rather
than ambiguous because the mechanism is visible: the affinity head genuinely
learns (ρ 0.55–0.62 on held-out affinity labels) and the stability predictions
still do not move, while measured affinity used *directly* as a stability
predictor ranks at median per-allele ρ 0.580 — **below** the 0.610 a single
network already reaches from stability labels alone.

**Bounded to:** a shared-trunk MLP on one-hot/BLOSUM peptide + pseudosequence
features, with affinity labels on 26% of training rows. It does **not** rule out
a different sharing scheme (separate towers, gated fusion), a different
auxiliary target encoding, or the broader 64,226-row leakage-filtered corpus —
that expansion is **declined on evidence, not blocked**, and is available.
The ceiling is computed on 33 training alleles with ≥ 30 dual-labelled pairs, a
narrower panel than the 68-allele validation panel. Only 58 of 75 alleles carry
any affinity data.

**Still open:** the ESM-2 arm. Affinity is redundant *with what a sequence model
already extracts from stability labels*, which does not establish redundancy
with ESM-2 features. This is the hypothesis's strongest remaining form.

### 5.3 ESM-2 reaches parity and adds nothing — and what that does not mean

**Established** (`stage3_esm.md`, SUBMISSION §4.0): ESM-2 only, −0.0101
[−0.0382, +0.0352]; sequence + ESM-2, −0.0170 [−0.0468, +0.0320]. Both
**inconclusive at zero, both ruling out +0.05**. Read that precisely — it is
*not* a demonstration that ESM-2 is worse. Both intervals contain zero. What is
established is that every upper bound sits below the predeclared worthwhile
gain.

**The differential target sharpens this from a shrug into an equivalence.** On
cross-allele ranking, where the peptide's own contribution cancels exactly,
ESM-2 is **+0.0028 [−0.0052, +0.0102]** against the baseline and sequence+ESM-2
is **−0.0013 [−0.0091, +0.0058]** — equivalent to within one point of
concordance, on 10,365 allele-pair comparisons. **The same measurement, same
comparisons, same bootstrap, conclusively separates the 150M checkpoint
(−0.0088 [−0.0149, −0.0025]) and the full-domain ensemble (−0.0124 [−0.0186,
−0.0061])**, so the null is measured rather than an artifact of a metric that
cannot discriminate. The mean per-allele Spearman separates the same two arms
(−0.0199 and −0.0348, both excluding zero) while the median calls them
inconclusive — two independent statistics agreeing. Full detail at SUBMISSION
§4.0.

**The one positive result, which must travel with all three of its
constraints.** Against the full-domain sequence ensemble on matching input,
ESM-2 is conclusively ahead on **two of three statistics**: median per-allele ρ
+0.0301 [−0.0084, +0.0757] (inconclusive), mean per-allele ρ +0.0312
**[+0.0082, +0.0540]** (conclusive), differential concordance +0.0152
**[+0.0046, +0.0241]** (conclusive). The additive arm agrees: +0.0111 [+0.0012,
+0.0207] on the differential.

The bounded claim: *given the same 182 HLA residues, ESM-2 extracts more usable
signal from them than a one-hot or BLOSUM encoding does.* It is a **designed**
comparison — the full-domain arm was added at stage 1, before any model existed,
so that "more input" could never be mistaken for "benefit of pretraining".

Three constraints, none of which may be dropped:

1. **It is not a claim that ESM-2 beats the best sequence arm.** The
   pseudosequence ensemble beats the full-domain arm by a *comparable* margin on
   the same two statistics (+0.0348 mean, +0.0124 concordance), and ESM-2 does
   not beat the pseudosequence ensemble (−0.0036 mean, +0.0028 concordance, both
   inconclusive). **Hand-picking the 34 contact residues recovers what
   pretraining buys here** — that is the honest summary of the whole project.
2. **It does not establish the 0.05 bar.** The mean interval runs to +0.0540, so
   it admits values below the bar as well as above it. "Conclusively better than
   the full-domain arm" and "worth 0.05" are different statements.
3. **Both conclusive verdicts come from secondary statistics.** The contract's
   median — the only statistic the predeclared verdict rule governs — stays
   inconclusive. **This caveat travels with the claim wherever it appears**,
   including every summary table. Full detail at SUBMISSION §4.1.

**Bounded to:** *frozen* embeddings of peptide and HLA taken **separately**, at
35M and 150M, with ridge and a small MLP head, under the frozen splits, on
validation. It does **not** establish:

- that **fine-tuned** ESM-2 would not help — nothing here was fine-tuned;
- that a model shown the **complex** would not help. Peptide and HLA are
  embedded independently, so the head must learn the peptide–HLA interaction
  itself from 19,716 rows. The plan predicted this would be the binding
  constraint, and ESM-only reaching parity while adding nothing on top is
  consistent with it. The chimeric peptide-linker-groove input remains the
  sharpest untested version, and is out of scope **specifically so this null is
  reported as bounded**;
- that **likelihood or perplexity** features would not help — embeddings are one
  of the three uses the brief names, and the other two were not tested;
- that a **larger checkpoint** would not help. 150M lands at 0.6737, marginally
  below 35M at 2.6× the cost, so the curve is flat across the two sizes tested —
  which **weakens but does not close** the scaling argument. 650M was excluded
  on a measured memory constraint (5.06 GB peak RSS alongside the production
  fold on a 16 GB machine), not on evidence;
- that **another pLM family** would behave the same way. One family is a thin
  basis for a class-level claim.

**Three controls make this reportable rather than an artifact**, and the first
is a general methodological warning:

1. **Transplanting the baseline's regularisation ladder would have cost 0.109
   SCC** and inflated seed spread from 0.013 to 0.110. At that setting ESM-2
   comes in 0.11 behind and the write-up says so with a straight face. **Tuning
   parity means equal budget, not equal values.** The check was applied
   symmetrically: the baseline on its own extended ladder moves +0.002, so the
   ESM arm is ~50× more sensitive to the range. Anyone comparing a dense
   pretrained representation against a sparse hand-built one should expect this.
2. **The memory-driven PCA helped by +0.033**, so the negative cannot be blamed
   on a compression adopted for resource reasons.
3. **The comparison machinery is verified**: the stage 2 ensemble was rebuilt
   from scratch through the new harness byte-identically (same md5,
   Δ = +0.0000), and stage 2's published prediction files were never regenerated
   in place.

**A separate finding worth carrying:** with only 75 distinct HLA sequences, the
mean-pooled and 34-contact representations are **both rank 74, both lossless at
74 components, and both separate all 75 alleles exactly** — identical
information. So the 0.177-vs-0.499 gap between them is **geometry, not
information**: where you read the embedding from matters more than which model
produced it.

The honest framing the brief asks for: *"Are they useful for this problem"* —
for this representation, on this split, at this budget. Not *"are they useful"*.

**One indexing caveat, already measured.** The plan required verifying residue
indexing before selecting the 34 contact-position embeddings out of the
182-residue domain. `reports/stage3_contact_index.json` records that check:
`exact_match: true`, 0 mismatched alleles — **but 4 of the 34 pseudosequence
positions are flagged ambiguous**, because several domain columns (7, 27, 84,
85, 118, 123, 159) carry the identical residue in all 75 alleles and so cannot
be told apart from the pseudosequence alone. The assignment chosen for those
four is a convention, not a derivation. It affects which column an embedding is
read from, not whether the right residue is read, so the effect should be
small — but if the contact-position ESM-2 representation is the arm that is
reported, this belongs beside it.

### 5.4 ESMFold2's rejection is operational, not scientific

ESMFold2 failed its pre-registered gate on both sentinel criteria: median arm-B
heavy-atom RMSD 2.564 Å on `HLA-B*07:02`/IPRRNVATL against a 1.0 Å bar, and a
0.008 Å improvement over arm A against a 0.5 Å bar, with no construct effect in
any arm or seed (`stage4c_ectodomain_pilot.md`).

**Two limits on reading that as a verdict on ESMFold2:**

1. **Pose accuracy is not feature utility.** Nothing here shows that ESMFold2
   features would predict half-life worse. Worse poses could still carry
   downstream signal; that experiment was not run, and the $884 full-cohort
   forecast is why.
2. **A five-complex gate is an operational decision rule, not a general claim.**
   Five complexes, three seeds, one construct family. It is enough to decide
   where to spend $200; it is not a benchmark of ESMFold2 on peptide–MHC.

The cross-model comparison the matched plan was designed to produce **was
delivered** — by the pilot, on identical constructs, MSA content, complexes and
seeds, with inputs verified identical at the feature level. Dropping ESMFold2
from production is a **labelled revision** of the agreed plan, not a silent drop.

---

## 6. The structural arm

### 6.1 Measured, and currently unfinished

The production fold **has not been launched** at the time of writing. Every
production cost, GPU-hour and wall-clock figure in `compute_ledger.md` is a
forecast from the pilot's measured 16.76 s steady fold and 57 s shard startup,
and is labelled as such. Unverified: whether each workspace is actually granted
10 concurrent A10Gs. If fewer, wall time scales linearly; cost does not move.
`::smoke` has passed on `a-cheparukhin` (5/5 ok, 17.0 s folds, 6.08 GiB peak
against the pilot's 16.76 s and 6.11 GiB) but **has not been run on
`sofyaleyn`**.

### 6.2 The central bulge is several Ångströms uncertain

Both folding engines share one failure mode: **error concentrates at central
peptide positions while the anchors stay tight**. The two-chain Boltz-2 pilot on
`HLA-B*07:02`/IPRRNVATL deviated 1.90 Å at P5 and 3.33 Å at P6 while P1–P2 and
P8–P9 were all under 0.25 Å (`stage4_benchmark.md`). The three-chain construct
flattens this particular case to a 0.28 Å maximum across all three seeds, but
the asymmetry is a property to expect, not one that has been shown to disappear:
**geometry features computed at central positions (burial depth, contact counts)
are noisier than the same features at anchor positions**, and stage 5 should not
treat all nine positions as equally well determined.

### 6.3 Confidence ranks error but cannot filter failures

Across the pilot's 45 Boltz-2 folds, confidence ranks error reasonably —
Spearman(peptide-to-groove PAE, heavy RMSD) = +0.635, Spearman(peptide pLDDT,
heavy RMSD) = −0.540. **It does not isolate the one real failure.** The three
failing arm-A sentinel folds sit at PAE 1.55 and peptide pLDDT 0.973–0.975 while
being 2.3 Å wrong, whereas the corresponding correct arm-B folds sit at *worse*
PAE (1.88–1.99). No PAE threshold catches the failure without flagging accurate
predictions. **Consequence: keep confidence as a feature, never use it as a
per-prediction failure filter** (`stage4c_ectodomain_pilot.md`).

### 6.4 Global ipTM is not peptide-interface confidence

In a three-chain complex, Boltz-2's global ipTM covers interfaces other than
peptide–HLA — notably HLA–β2m, which is large, conserved and easy. **It must be
labelled and reported as global ipTM.** A pair-specific score may be used only
if the pinned model actually emits one with a verified chain mapping. (ESMFold2
exposes `pair_chains_iptm`, the peptide–HLA interface ipTM; Boltz-2 does not.
If that feature proves important, this is a reason to revisit ESMFold2 despite
§5.4.)

### 6.5 B versus C does not isolate a mechanism

Arm B (ectodomain + β2m + peptide, new alignment) improves; arm C (groove +
peptide, the *same* new alignment rows cropped) tracks arm A everywhere
including the sentinel. So the benefit comes from the full construct together
with its extra evolutionary information. **This does not separate α3 from β2m
and is not a physical-mechanism claim.** Separate α3/β2m ablations are
explicitly out of scope this round.

### 6.6 Six of 75 ectodomain constructs are not exact matches

Three alleles have only a 181-aa groove-only IMGT record, so their α3 domain is
borrowed from a close relative: `HLA-A*02:50` ← `HLA-A*02:01` (177/182 α1/α2
identity), `HLA-A*24:19` ← `HLA-A*24:07` (174/182), `HLA-B*08:03` ←
`HLA-B*08:01` (176/182). The three C67S constructs take α3 from their wild type,
keeping α1/α2 from the dataset. Source: `hla75_ectodomain_b2m.csv`, `note`
column. **Structural features for these six alleles rest on a construct that is
partly inferred**, and α3 is outside the peptide-binding groove, so the effect
should be small — but it has not been measured. Worth flagging at stage 5:
`HLA-A*24:19` is also one of the alleles the sequence baseline ranks worst
(validation ρ = −0.12, `stage2_baselines.md`). That is a coincidence to check,
not a claim.

### 6.7 The pilot is a pipeline check, not an accuracy benchmark

Five complexes, from the **training** split, so training-set recall is possible.
It validates the pipeline, the pose scorer and the feature extraction. It does
not estimate generalisation, and A/B/C construct comparisons are limited to
those 90 shared folds. Structural pose accuracy is also a pose result: **better
crystal agreement must still earn its predictive value on validation.**

### 6.8 Hardware measurements carry n=1 container variance

The two-chain GPU sweep used **one container per GPU type against 73% measured
between-container variance** (6.3 s vs 10.9 s for the same model and settings on
the same card). The A10 choice and the H100 verdict survive this because both
are gaps larger than the variance; **the fine-grained ordering among the middle
cards does not** (`stage4_benchmark.md` Finding 2). Those costs are also
*computed* from published per-second rates, not invoiced — though the ~10%
agreement with the metered workspace total is a check on them
(`compute_ledger.md` §1).

---

## 7. Scope not covered

Recorded so that absence is not mistaken for a result. Full rationale in
`HACKATHON_PLAN.md`, "Out of scope this round".

| Not run | Why | Status |
|---|---|---|
| ~~Allele-held-out evaluation~~ | Different question from the headline; confounded by **panel composition**, not panel hold-out (§2.1) | **Run for the sequence arm**: near 0.741 vs distant 0.339, +0.403 [+0.223, +0.478]. Extreme contrast solid, monotone trend not. ESM-2 and structural arms still to run, and **must be refit, not scored from a `preds/*.csv`** |
| ~~ProteinMPNN inverse-folding scores~~ | Was blocked on needing structures; the stage 4c pilot's 90 folds unblocked it | **Run** — pilot complete, §7.2. Kept as a ~$1 QC triage sample; **declined as a regression feature** on a measured n=5 label correlation of −0.100 |
| Chimeric peptide-linker-groove ESM-2 input | Far outside ESM-2's distribution | **Listed specifically so a stage 3 null is reported as bounded** (§5.3) |
| ~~Tobit / censored likelihood~~ | Was the 16-hour substitute's known gap | **Run, and negative**: −0.0414 [−0.0780, −0.0062] on ranking, gains calibration only (§1.1) |
| FoldX / Rosetta / empirical ΔG | Equilibrium ΔG vs a kinetic label (§1.3), **and both are licence-gated behind registration** | Declined for two independent reasons — §7.0 |
| Elution data as training augmentation | 5.4× scale mismatch, unknowable threshold τ, confounded with abundance/cleavage/TAP/ionisation, and it changes the question | Rejected **as training data**; used as external validation instead, which has now run — §7.1 |
| ESMFold2 production folds | Failed its gate; $884 forecast | §5.4 |
| Chai-1, Protenix, SaProt | Integration cost beyond the agreed comparison | Not evaluated |
| **ESM-IF** (the brief's second inverse-folding model) | Machine contention — the ESM-2 arm held priority on the shared environment | Not attempted. **ProteinMPNN alone covers the inverse-folding class**, so the class is tested; this specific model is not |
| Separate α3 / β2m ablations | Deferred until the full pipeline's predictive value is established | §6.5 |

### 7.0 FoldX and Rosetta — declined for two independent reasons

Both were requested as stretch goals and both are declined. **Two reasons, and
the second is a constraint on anyone extending this work, not just on us:**

1. **The scientific objection, which was always the real one.** Both estimate an
   **equilibrium** free energy, ΔG — how favourable the bound state is. Our
   label is **kinetic**: a dissociation half-life, governed by ΔG‡, the height
   of the barrier to unbinding. Two complexes can sit at the same ΔG and come
   apart at completely different rates. This is a known mismatch with the label,
   not a matter of setup time, and it does not go away with more compute. It is
   the same objection recorded against per-pocket energy decomposition (which
   was premised on FoldX `AnalyseComplex` and falls with it) and against OpenMM
   minimisation energy. The softer form of the same mismatch applies to
   ProteinMPNN (§7.2) and is recorded there rather than hidden.
2. **Both are licence-gated behind registration.** Neither can be installed from
   a public package index without an account and an accepted licence. So even if
   reason 1 did not hold, **including them would make this work less
   reproducible**: a reader who wants to re-run the pipeline would hit a
   registration wall that nothing else in this repository has. Every other model
   used here — Boltz-2, ESMFold2, ESM-2, ProteinMPNN — is openly downloadable,
   and the ProteinMPNN checkpoint is pinned by SHA-256 so a reader can verify
   they have the same weights.

Reason 1 alone is sufficient. Reason 2 is recorded because it is the one a
reader is more likely to hit in practice.

### 7.1 The elution external validation has its own error rate

The MHC Motif Atlas comparison is a genuine external check — only 140 of 151,170
9-mers overlap, so the peptide universes are near-disjoint. Eluted ligands show
~7× higher median half-life (7.90 h vs 1.10 h) and a tenfold drop in the
zero-stability fraction (2.1% vs 21.2%), Mann–Whitney p = 6.0 × 10⁻³¹, with the
eluted median exceeding the non-eluted median in 14 of 18 alleles that have ≥ 3
measured eluted ligands. **But it is probabilistic, not deterministic.** Six
eluted ligands have measured half-life < 0.5 h, three at exactly zero, and
`FPEHIFPAL` appears at zero on two different alleles — a pattern more consistent
with motif-deconvolution error than with biology. That puts the atlas error rate
at roughly 2% for stability purposes (`elution_stability_finding.md`).

**The transfer pass itself has now run** (`stage3c_elution_validation.md`, 51
alleles, 81,600 ligands). Five limits attach to its headline AUROC of 0.9656 and
none may be dropped when the number is quoted:

1. **It is not a measurement of stability-prediction accuracy.** Elution is a
   selection effect with **at least four filters besides stability**:
   source-protein abundance, proteasomal cleavage specificity, TAP transport,
   and mass-spec ionisation efficiency. A peptide in the atlas passed all of
   them. A model that had learned only "this peptide is abundant and ionises
   well" would also score well against proteome decoys. The allele-swapped
   control **bounds** this contribution at **0.050 of AUROC** — that is a bound
   on this decoy design, not a decomposition of the biology.
2. **It is not a ranking of stability within presented peptides.** Ligand versus
   non-ligand is a binary discrimination; per-allele Spearman against measured
   half-life is a harder question this pass says nothing about. **0.610 / 0.693
   remain the numbers to quote for accuracy.**
3. **It is not comparable to a trained presentation predictor.** NetMHCpan-4.x
   and friends train on elution data; this model never has. The reading is "a
   stability model transfers", not "competes with a presentation model". That
   comparison was not run and is not claimed.
4. **The positive class carries roughly the 2% error rate above**, which puts a
   soft ceiling just under 1.0 on any AUROC here, and means the weakest alleles
   may be partly measuring motif-deconvolution quality rather than model
   quality.
5. **Decoys are assumed negatives**, since mass spectrometry is positives-only
   and absence carries no information. This contaminates the negative class
   slightly, which **depresses the reported AUROC rather than inflating it** —
   so in that one respect the numbers are conservative. One release, one decoy
   draw, seed sensitivity not swept.

The donor-distance gradient (0.8906 near vs 0.9562 far) is real in direction and
size but **only two bins were measured**, so the shape of that decline is not
established.

### 7.2 ProteinMPNN pose triage rests on n = 1 complex

`stage5_inverse_folding.md`. Four limits, all load-bearing:

- **The sample is one failing complex, not six folds.** All six Boltz-2 folds
  above 2.0 Å peptide heavy RMSD are `HLA-B*07:02`/IPRRNVATL, arms A and C at
  three seeds. "Six of 45" would read as six independent failures and overstate
  it by a factor the evidence cannot carry. **One failing complex is not a
  general property of the method.**
- **It is not a half-life predictor and was not shown to be one.**
  Spearman(`pep_ll_total`, t½) = −0.100 at p = 0.87, n = 5, all training rows.
  That is zero and carries no inferential weight. Its value as a kinetic feature
  is **untested**, and this pilot cannot test it. The full-cohort regression
  scoring is declined on that evidence.
- **The claim is specificity, not separation.** Repeating the separation test on
  every complex yields 7, 9, 2, 13 and 2 "fully separating" structural features,
  with the genuinely failing complex scoring the **fewest** — so clean
  separation of a six-fold group measures *complex identity*, and **none of the
  109 structural features isolates the failure**. `pep_ll_mean` escapes that
  artifact only because it was a single predeclared quantity and because the
  within-complex control shows the score moving when the *same* complex's pose
  is fixed.
- **Thermodynamic-flavoured, kinetic label.** Sequence–backbone compatibility is
  the softer form of the §7.0 mismatch. Nothing in this pilot overturns it.
- **Its seed control disagrees with our other seed control on ESMFold2**, and
  the two must not be cited as confirming each other there. The 109 structural
  features give 5.29 (Boltz-2) / 5.87 (ESMFold2); ProteinMPNN gives 3.1-9.3 /
  1.2-1.9. They agree on Boltz-2, which is what production runs. The divergence
  is attributed to granularity — ProteinMPNN reads **backbone only** (N, CA, C,
  O plus a *virtual* CB; it never sees a real side-chain coordinate) at fine
  resolution, while the structural features are coarse aggregates robust to
  sub-Angstrom jitter, and ESMFold2's CA jitter is 2.5-5.2x Boltz-2's. **An
  earlier attribution of this divergence to side-chain sensitivity was wrong and
  is retracted**, as was a separate claim that the two implementations
  reconciled to two decimal places — a coincidence of conventions. The surviving
  cross-validation is directional: removing seed noise from the numerator drops
  ESMFold2's ratio by -0.29 in **both** independently written pipelines while
  barely moving Boltz-2's. See SUBMISSION §4.6.
- **Quote the ProteinMPNN ratio as an envelope, not a point estimate.** Six
  defensible aggregations span 3.1-9.3 and 1.2-1.9; all return the same verdict,
  so the spread bears on no conclusion, but a single figure quoted without its
  convention is not checkable. The two reports state the Boltz-2 upper bound
  differently (8.1 vs 9.3) depending on whether a `ddof=0` aggregation is
  counted; the ESMFold2 range, which carries the verdict, agrees exactly.

**Operational consequence:** the QC sample's reference points are descriptive,
drawn from a single complex. They **must not filter the production cohort**,
nothing downstream may condition on them, and the deliverable is a distribution
and a triage list — **never a failure count or a failure rate** — until the
signal has been checked on more than one failing complex.

---

### 7.3 Three of our own measurements cannot do the job we gave them

**Including the contract's primary metric.** Across the whole ESM run — 8 paired
comparisons, 12 stratum intervals, 1 nested row — the median per-allele Spearman
returned **0 conclusive verdicts**, while the mean returned 4 and the
differential concordance 5. A median over a 68-allele panel is robust, and
robustness costs power. **We predeclared it and we keep it**, because changing
the primary metric after seeing which one resolves things is the post-hoc
selection this contract exists to prevent — but a future version of this study
should predeclare the differential as primary. Detail at SUBMISSION §6.8.

Recorded because both are failures of *our* design choices, not of the models,
and because a reader who only sees the arms graded would miss that the
instruments were graded too.

### The distance-stratification contrast is underpowered by construction

Stage 1 predicted that if a model exploits residual similarity at the split
boundary, d=4 would score higher than d≥5, and that an arm with a *flatter*
profile would be evidence of better generalisation. On validation, **all four
stratum-gap intervals cross zero** — ESM-2 35M −0.0197 [−0.0973, +0.0800],
sequence+ESM-2 +0.0046 [−0.0803, +0.1235], 150M −0.0685 [−0.1188, +0.0625],
full-domain −0.0091 [−0.0888, +0.0847].

**This is a statement about the split's size, not about the arms.** The stratum
gap is a difference of differences of medians — two medians per arm, differenced,
then differenced again against the baseline — and each step compounds noise. The
half-width lands near **0.09** on a quantity whose largest observed value across
all five arms is **0.069**. The measurement cannot resolve the effect it exists
to detect, whatever that effect's true size, so "pretraining buys generalisation
at the split boundary" is **not supported and also not refuted** here.

**The null is thorough, not a single underpowered look.** The obvious objection
to the above is that only the weakest form of the test was run. It was not. The
question was asked **three different ways** and the split answered none of them:

| Form of the question | Arms | Result |
|---|---:|---|
| Stratum gap (difference of differences) | 4 | all cross zero |
| Within d=4, arm vs baseline directly | 4 | all cross zero |
| Within d≥5, arm vs baseline directly | 3 of 4 | all cross zero |

The decisive one is the **better-powered single difference** that the whole
hypothesis reduces to — does the larger checkpoint actually beat the baseline
where peptides are most distant? ESM-2 150M at d≥5 scores **+0.0219 [−0.0776,
+0.0619]**: inconclusive, the same answer the gap test gives. **Eleven of twelve
intervals are in and not one separates any arm from the baseline in either
stratum**; the twelfth (full-domain at d≥5) is still running and cannot change
the pattern.

Settling this needs a better-powered split, not a re-analysis of this one.

*Technical note, because a reader checking the CSV will notice it:* the 150M
d≥5 point estimate sits in the upper part of its own interval rather than at the
centre. That is expected, not an error — the statistic is a median over a
55-allele panel **rebuilt on every resample**, so its bootstrap distribution is
skewed. One more reason never to read a point estimate from this statistic
without its interval.

A related trap is recorded in the source report and worth repeating: the 150M
arm is the flattest across distance *and* uniformly the weakest on every other
metric. "Nothing to lose" and "genuinely distance-robust" make identical
predictions for every quantity this split can measure, so they are not
separable here.

**Provenance of the flat-profile hypothesis — recorded now, before any test
number exists.** The idea that a *larger checkpoint* might have a flatter
distance profile was **generated on validation**, by looking at the point
estimates above. It was **not predeclared**. `HACKATHON_PLAN.md` now records
this.

The consequence is a rule that binds whichever way the test comes out:

> If a flatter distance profile for the larger checkpoint reappears in the test
> results, it is a **validation-generated hypothesis tested once** — not a
> predeclared prediction confirmed — and it must be labelled that way. If it
> fails to reappear, that is equally a single test of a post-hoc idea, not a
> refutation of a standing prediction.

This matters because the asymmetry is the whole trap: a hypothesis read off one
split and then "confirmed" on another looks exactly like a prediction that was
made in advance, unless the order of events is written down. It is written down
here, **before the test set has been scored**, so the record cannot be
reconstructed favourably afterwards. The same rule applies to any other pattern
first noticed on validation.

### Precision@10 is quantised past the point of usefulness

**This is a criticism of a metric we predeclared.** Against the sequence
baseline, all four ESM comparisons returned a delta of **exactly 0.0000**, with
interval bounds that are themselves lattice points of the statistic — [−0.100,
+0.100] three times, [−0.100, +0.050] once.

It is not that the metric is underpowered; **it cannot express a difference**.
The statistic moves in steps of 0.1 because it is ten slots, and a median over
68 alleles of a 0.1-quantised quantity sits on a lattice point and stays there
under resampling. An underpowered metric gives a wide interval around a non-zero
estimate; this gives an estimate pinned to zero by construction, unable to
represent the 0.04 differences the primary metric reports. Only the lift and
ceiling-share columns carry information, and at this resolution even those are
within noise.

Its only informative columns, lift and ceiling share, span just **0.277–0.294
across arms that differ by 0.04 on the primary metric**, so even they are within
noise here. It separated the single network from the ensemble at stage 2, so it
is not useless in general — it is useless at this resolution. **It stays in the report
because it was predeclared**, and dropping a metric after seeing it is
unflattering to the process is the post-hoc selection the contract exists to
prevent. **No claim in this submission rests on it.**

---

## 8. Open holes in this register

| Tag | What is missing | Who fills it |
|---|---|---|
| ~~E3~~ | ~~Stage 3's outcome and its bounds~~ | **Filled**: §5.3 |
| **B4** | Stage 5 structural feature coverage and failure rate; what the declared sequence fallback covers | blocked on 4c production |
| **S7** | Whether d=4 and d≥5 test strata separate (§3.1), and the nested near-neighbour result (§3.2) | `eval-harness` |
| **S8** | Any allele where the test panel disagrees sharply with validation — hard alleles vs overfitting to the validation panel | `eval-harness` |
| ~~S7a~~ | ~~Censored-likelihood result~~ — **filled**, §1.1 | — |
| **A1** | Per-stratum leave-allele-out comparison for the ESM-2 and structural arms (§2.1). Measured MDE is 0.025–0.030 when per-allele deltas are tight and 0.075–0.115 when loose, so **a stratum may come back inconclusive and must be reported as such, not as a null** | `esm-arm` / stage 5 |
| ~~S3C~~ | ~~Elution external-validation write-up~~ | **Filled**: `stage3c_elution_validation.md`, §7.1. Open as a follow-on: the same pass on the ESM-2 and structural arms |
| **P1** | Whether the ProteinMPNN pose-triage signal survives a second failing complex (§7.2) | blocked on 4c production + the QC sample |
| **A3** | Whether the six partly-inferred ectodomain constructs (§6.6) behave differently at stage 5 | stage 5 |
