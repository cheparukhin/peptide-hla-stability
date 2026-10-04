# Limitations register

Everything a reader should discount this submission by, collected in one place
and sourced. Headline results live in [`REPORT.md`](REPORT.md); the contract,
verdict table and bar live in [`EVALUATION.md`](../EVALUATION.md); per-stage
numbers live in the stage reports named inline. This file carries only the
limitation-specific analysis, not a restatement of those sources. A negative
result is well supported only if the reader can see what would have had to be
true for it to be wrong; that is what is below.

---

## 0. Reading the negative results

The six-verdict rule for a paired 95% CI on Δ median per-allele Spearman is in
[`EVALUATION.md`](../EVALUATION.md), "Reading a comparison", with the 0.05 bar
and its derivation under "Minimum worthwhile gain". Two points about those
verdicts belong here because they are limitations, not contract:

**Every negative result in this project is "crosses 0, upper < 0.05", not "the
effect is zero".** That verdict rules out a worthwhile gain; it does not show the
addition does nothing. Stage 2b's best arm is +0.024 [−0.026, +0.048] — data
consistent with a real +0.04. What is ruled out is +0.05.

**A null is only informative next to a positive control on the same
measurement.** Two analyses have one: the nested mutant evaluation (a
mean-pooled ablation blind to point mutations scores conclusively below chance
while every mutation-aware representation sits in an unresolvable band around
0.5) and the differential target (two conclusive separations license its tight
equivalence). Two do not: the distance strata and precision@10. Their nulls are
correspondingly weaker — measured and found nothing, without showing the
instrument could have found something — and §7.3 records why. The asymmetry is
stated rather than left for the reader, because presenting four inconclusive
results as a uniform block would overstate half of them.

**The 0.05 bar is a property of this test set, not a preference.** A larger
benchmark would move it down, and some of what we call "ruled out" would become
detectable (`audit_summary.md` §6).

---

## 1. The label

### 1.1 One fifth of the labels are not measurements

**5,679 rows (20.2%) sit at exactly 0 hours, and the evidence says this is an
assay detection floor, not a measured zero** (`audit_summary.md` §3). Three
independent lines: a trough at 0.1 h holding 443 observations where a fit over
0.2–1.0 h predicts ~1,651; per-allele zero share anticorrelating with that
allele's median non-zero label (Spearman −0.608, p = 3.7 × 10⁻⁸); and zeros
spread across peptides rather than concentrated.

`log1p` lets the floor rows be kept (it is defined at zero) and compresses a
0–256.7 h target from skewness 5.86 to 0.91. It does not fix the censoring: it
records a "t½ < limit" observation as "t½ = 0". Two consequences:

- **MAE at the floor is uninterpretable in sign.** It measures error against the
  recorded label. For a latent 0.05 h recorded as 0, predicting 0.10 is charged
  2× too much and predicting 0.01 is charged 4× too little, so floor MAE is
  neither an upper nor a lower bound on true error (`EVALUATION.md`). This is why
  MAE is secondary and the primary metric is a rank correlation.
- **Rank metrics handle the floor but do not escape it.** Average-rank Spearman
  treats the 20.2% as one tied block, which is the right treatment of "all below
  the limit", but it cannot recover ordering within that block.

**The principled alternative — a censored (Tobit-style) likelihood that models
P(t½ < floor) directly — was built and tested (stage 7a, fully predeclared), and
it is worse:** Δ median per-allele Spearman −0.0414 [−0.0780, −0.0062], entirely
below zero, and it also loses on its own held-out objective (censored NLL 1.0765
vs 1.0452). It delivers the promised calibration (predictive mass below the limit
7.0% → 16.8% against an observed 19.6%; ECE 0.0561 → 0.0422) and, as predeclared,
does not improve floor discrimination (AUROC 0.8862 vs 0.8853), because ranking
inside the tied block is unidentifiable under either objective. The result is flat
across detection limits 0.05–0.30 h, so the free parameter is not driving it. The
loss likely comes from the stopping rule rather than the objective (the censored
arm's dev objective turns over at epoch ~10 against the MSE arm's ~27, so it is
undertrained) — a control without an interval, not a result. Detail in
`stage7_censored.md`.

The same censoring bit the stage 2b affinity predictor from the other direction:
26.8% of its training labels sit on the 20,000 nM boundary, an MSE regressor
shrinks them toward the conditional mean, and weak-call recall collapses to 0.263
at precision 0.823 (`stage2b_augmentation.md`). A censored likelihood is the
recorded fix there too.

### 1.2 No replicates, so no noise ceiling

The supplied file has one value per pair. 98.74% of labels sit on a 0.1-hour
grid; the 354 finer values concentrate in the most-measured alleles and are
likely replicate averages, but the replicates are not supplied
(`audit_summary.md` §3). **We cannot say how much residual error is irreducible
assay noise**, so we cannot say how close any model is to the best achievable
score. Every "ruled out" is against the 0.05 bar, never against a noise ceiling.

A post-hoc estimate from a separate branch derived a reproducibility floor of
**≥ ~0.90** from allele pairs one contact-residue substitution apart (two such
alleles should rank peptides almost identically, so their observed concordance
bounds assay reproducibility). It is recorded but not adopted, with three
qualifications:

1. **It is the best four of twelve.** There are 12 pairs at Hamming 1, not 4; the
   other eight run 0.657–0.838. The lower-bound logic survives (reproducibility ≥
   max observed concordance, the pair where biology changed least), but "≥ 0.90"
   reads differently once you know it is the top third of the pairs.
2. **It is post hoc and from a separate branch**, computed after results existed.
3. **`EVALUATION.md` is frozen and is not reopened for it.** It still states that
   there are no replicates and therefore no noise ceiling, which is the correct
   statement about this dataset's contents.

No verdict is restated against this ceiling; re-reading verdicts against a
post-hoc estimate from another branch is the moving target the frozen contract
prevents. If the floor holds, the best arm at median per-allele ρ ≈ 0.69 leaves
roughly 0.2 of measured headroom unexplained — larger than every between-arm
difference measured, combined. That is a statement about the task, for next
steps, not a verdict.

### 1.3 Stability is not affinity, and neither is the unbinding barrier

The target is residence time once bound, largely a dissociation rate. Binding
affinity is an equilibrium quantity combining on- and off-rates. Neither label
measures the energy barrier to unbinding (ΔG‡) directly. This is why FoldX,
Rosetta and any empirical ΔG layer are out of scope: they estimate equilibrium
ΔG, a known mismatch with a kinetic label (`HACKATHON_PLAN.md`). No mechanism
claim about unbinding is supported by improved prediction alone.

---

## 2. The dataset

### 2.1 The peptide panels are allele-confounded by design

Each allele was assayed on its own peptide panel. The measured allele × peptide
grid is only 6.7% full; each peptide was tested on a median of 4 alleles and
1,692 peptides appear on a single allele (`audit_summary.md` §4). Three
consequences the submission cannot design away:

1. **It broke the first split.** A water-fill ranked by absolute deficit cycles
   round-robin once the three deficits equalise, so heavy clusters all land in
   whichever split led early, leaving `HLA-B*42:01`, `HLA-B*51:01` and
   `HLA-B*81:01` (~350 pairs each) with zero test rows. Ranking by deficit
   relative to target fixed it at no cost to the 70/10/20 totals
   (`audit_summary.md` §5).
2. **It is why leave-allele-out is not the headline — but the original mechanism
   was wrong and is corrected.** `HACKATHON_PLAN.md` claimed holding out an
   allele also holds out its peptide panel, making allele distance inseparable
   from a panel effect. Measured, that is not what happens
   (`stage7_allele_holdout.md` §4): allele-exclusive peptides are 6.0% of rows,
   and for the median eligible allele 100% of its rows carry a peptide seen on
   another allele. Peptide overlap is uncorrelated with distance (−0.067,
   p = 0.59) and with per-allele performance (+0.034, p = 0.79). So
   leave-allele-out here is largely *seen peptide, unseen allotype* — easier than
   the frozen split, which is exactly why its absolute numbers (stage 7b, near
   0.741 / distant 0.339) must never be quoted beside a frozen-split number.

   **The real confound is panel composition.** Each panel was partly selected by
   predicted binding affinity for that allele, and distant alleles carry
   weaker-binding, more censored panels: Spearman(distance, per-allele zero
   share) = +0.251, p = 0.039. Partialling them apart leaves distance at −0.604
   (p = 4.9 × 10⁻⁸) against −0.636 raw, and zero share at −0.336. Both carry
   independent signal: the confound is attenuated, not eliminated, and zero share
   is only one proxy — anchor-motif composition, peptide diversity and the
   affinity-prediction step are unmeasured. The worst fold:
   `HLA-B*39:06(C67S)` at d = 3 with a 91.8% floor panel scores badly because of
   its panel, not its distance.
3. **Per-allele medians are over an unbalanced panel.** 8 of 75 alleles hold
   fewer than 50 test rows and are excluded; 7 of those hold ≤ 32 pairs total.
   The eligible panel covers 98.8% of test rows, but per-allele claims are
   bounded to the 67 alleles in it (`audit_summary.md` §5).

### 2.2 The panel was pre-selected by predicted affinity

Rasmussen et al. selected assayed peptides partly on predicted binding affinity,
so the peptide diversity is narrow by construction: a sample of peptides a model
already expected to bind. Two measured downstream effects:

- **Real zero-stability peptides have normal anchors.** Validation rows at the
  floor and above it look the same at the anchor positions (84.2% vs 84.0%
  hydrophobic at PΩ, `stage2b_augmentation.md`). What separates a 0-hour peptide
  from a 38-hour one here is not whether it can enter the groove. A model trained
  here ranks within the set of peptides that bind — which is what this benchmark
  measures.
- **It is why stage 2c was worth running and came out flat.** Broader affinity
  corpora cover more peptides; the hypothesis was they would supply diversity the
  stability panel lacks. The measured ceiling says the signal is redundant with
  what stability labels already teach (`stage2c_affinity.md`).

Any broader biological or clinical claim needs evidence outside this dataset.

### 2.3 Two alleles are indistinguishable to a pseudosequence model

`HLA-B*14:01(C67S)` and `HLA-B*14:02(C67S)` share one 34-residue contact
pseudosequence (756 rows, 2.7%), so a pseudosequence-only model must predict them
identically (max absolute difference 0.00 across 41 shared validation peptides),
while their measured labels correlate at only ρ = 0.708. The full-domain arm
separates them (max difference 0.31) (`stage2_baselines.md`). The collision caps
cross-allele discrimination, not within-allele ranking, which is why it barely
moves the primary metric. It does not reach the structural arm: the two
182-residue domains differ at position 10, outside the contact set, so they get
distinct MSAs (`stage4b1_msa_cache.md`).

### 2.4 Three C67S constructs have no matched wild type

The C67S substitution replaces a free cysteine that caused assay aggregation. It
sits at one of the 34 contact positions, so wild-type affinity describes a
different groove — 245 rows are excluded from the stage 2c auxiliary labels for
this reason (`stage2c_affinity.md`). No matched wild-type alleles exist, so the
C67S effect cannot be measured here, only observed: these three alleles carry
74.9–92.1% zero labels against 17.4% for the 72 natural alleles
(`audit_summary.md` §2).

---

## 3. The split

### 3.1 Grouping reduces label leakage; it does not remove it

Peptides are grouped by single-linkage clustering at Hamming ≤ 3, every cluster
wholly inside one split, so no validation or test peptide is within 3
substitutions of any training peptide. But same-allele label similarity decays
smoothly with distance rather than in a step (`HACKATHON_PLAN.md` stage 1):

| Distance | Same-allele comparisons | Spearman |
|---:|---:|---:|
| 1 | 513 | 0.725 |
| 4 (the minimum across splits) | 11,275 | **0.592** |
| 5 | 91,636 | 0.512 |
| unrelated background | 81,455 | 0.302 |

At d=4 the residual similarity is 0.592, roughly halfway between unrelated pairs
and near-duplicates. No feasible threshold eliminates it: single-linkage
percolates sharply between 3 and 4, where one component swallows 2,807 peptides
(54.8% of pairs). Hamming ≤ 3 is the most conservative threshold that still
permits a 70/10/20 split.

The stage 6 mitigation stratifies test metrics by distance to training (d=4,
57.8% of rows; d≥5, 42.2%), on the same 65 alleles so the comparison measures
distance, not panel composition.

**Reviewed after the test pass (4 October 2026): the mitigation returned the
unfavourable answer, so this limitation is stronger than when written.** Every
one of the six arms scored higher at d=4 than at d≥5 — gaps of +0.0040 to
+0.0388, the plain sequence ensemble at +0.0195 and the two ESM-2 arms the
largest. The direction is consistent across arms, which is what exploitation of
boundary similarity predicts. What stops this being quantified is that **no
paired interval was computed on the stratum gap itself** — the pass intervals
arm-vs-arm deltas, not within-arm stratum differences. So the size is unresolved
and the smallest gaps are inside the arm-level noise. Residual similarity at the
boundary is not excluded; it contributes something in every arm, and any number
in this document includes an unknown amount of boundary credit.

### 3.2 The split cannot answer the mutant-ranking question

Because no two peptides within 3 substitutions straddle a split, the benchmark
measures generalisation to distant sequences and cannot assess mutant ranking —
scoring point mutants of a known binder, often the practically relevant design
question. The partial recovery is a nested cross-validation on the d≤2 clusters
inside the training split (`EVALUATION.md`, stage 6). That is a different
evaluation on a different panel, not the headline number.

### 3.3 The split is model-independent on purpose, which costs something

Plain Hamming was chosen over BLOSUM-weighted or embedding-based distance so the
split would not depend on the model under test. At a matched threshold BLOSUM62
reproduces essentially the same partition (5,493 vs 5,494 clusters at Hamming ≤
1) while adding a threshold to defend (`HACKATHON_PLAN.md` stage 1). The cost is
that Hamming treats a conservative and a radical substitution identically.

### 3.4 Disclosed test exposure

One superseded diagnostic breached the single-use test rule, pulling 3,350 frozen
test rows into fitting, 448 into early stopping and 585 into a scored set that was
21% test rows. What was observed is eight panel aggregates over 68 alleles; no
per-row prediction, per-allele score or individual label was inspected, and its
retracted +0.018 attribution measures properly as −0.005 [−0.035, +0.046]. Full
account in [`EVALUATION.md`](../EVALUATION.md), "Disclosed test exposure". No model
scored at stage 6 saw a test row; we judge the split still usable and scored it
once. A reader who disagrees has the eight numbers to discount with — the point of
disclosing rather than quietly fixing.

---

## 4. The comparators

### 4.1 NetMHCstabpan is calibration, never a comparator

**NetMHCstabpan was trained on all 28,166 rows, including every test-split
peptide**, so any score it produces on our data is memorisation. The brief flags
this. There is no valid "beat NetMHCstabpan" result available at any stage. Its
published 0.69 mean per-allotype SCC is a 5-fold CV score on its own training
data, not the same quantity as our 0.645 mean / 0.693 median per-allele ρ on
unseen data; the 0.045 gap is not a model-quality comparison. Its published
0.676 PCC must never be quoted as a bar — it was computed on data padded with
~75,000 easy synthetic negatives, which inflates correlation
(`stage2_baselines.md`; `EVALUATION.md`, Caveats).

**What the gap can be attributed to.** Split grouping shows no conclusive
advantage at equal row count (Δ median ρ −0.005 [−0.035, +0.046], Δ mean ρ
−0.000 [−0.024, +0.024]); both intervals straddle zero but admit a modest
positive effect. Read as "no conclusive advantage found", not "no effect". It
bounds grouping below ~0.024 on the mean, about half the 0.045 gap. The remainder
is consistent with their 5.2× training set, richer architecture diversity and
their scoring on the stopping fold, which cannot be separated here.

### 4.2 Validation scores are selection scores

Every number from stages 2, 2b and 2c is a validation score, and each arm's
configuration was chosen on the number reported beside it, so they are optimistic
as held-out estimates. They remain valid for comparing arms under the same
budget, which is their purpose. λ in stage 2c was swept on validation, which is
why the whole sweep is tabulated rather than its maximum (`stage2c_affinity.md`).

### 4.3 Seed spread sets a floor on what counts as a difference

Seed-to-seed spread for the selected MLP configs is 0.010–0.051
(`stage2_baselines.md`). Gaps under ~0.05 are within seed noise — the same order
as the predeclared bar. Compare seed means, never single seeds.

### 4.4 Single figures are not reproducible to four decimals

**Cause: BLAS thread count, not seed.** Re-running the same diagnostic through a
committed code path moved single-network numbers — a healthy arm by ~0.02, a
collapsed arm by ~0.03 — because a different number of BLAS threads sums float32
products in a different order. Threads were pinned to 1 mid-session for machine
load, so numbers produced before and after the pinning are not bit-comparable.

Every published figure is a 30-member ensemble with a bootstrap interval, and
none of those moved: averaging 30 members and resampling 2,000 times swamps a
0.02 reduction-order effect. The binding consequence: **single-network figures
should not be read to four decimals**, and comparing two that differ by ~0.01 is
reading noise. Two instances:

- The tuning-sensitivity table (`stage3_esm.md`) is single-network. Its
  load-bearing figure — the 0.109 cost of transplanting the baseline's L2 ladder
  onto the ESM arm — is five times the noise scale, so the finding holds. But the
  baseline's −0.002 on the same table is inside the noise, so the contrast is
  stated as two orders of magnitude apart, one indistinguishable from zero,
  rather than as a ratio.
- Stage 2's six-arm table and stage 2c's λ sweep are single-network, but every
  comparison drawn from them is reported with a paired interval far wider than
  0.03, so the conclusions stand.

This surfaced from an independent branch re-running our diagnostics: two runs of
identical code at an identical seed, differing only in thread count, are not
identical in float32, and the defect hides because it is invisible to a seed
check.

**Reviewed after the test pass: one 30-network *ensemble* also failed to
reproduce, which is worse and was not anticipated.** Stage 6 needed every arm
refitted because stage 3 persisted no fitted networks. Two of three ESM-2 arms
reproduced their frozen validation number to four decimals; `esm_ensemble_150m`
did not (0.6676 under one BLAS thread, 0.6690 under default, against a frozen
0.6737). Two refits differing by 0.0014 while both ~0.005 below target rule out
reduction order as a sufficient explanation, and the configs, feature width, fold
assignment, seeds and embedding-cache hash all match the frozen record. The cause
is not identified. Two consequences:

- The arm was **excluded** from the test pass rather than scored unverified, so
  its scaling claim stays validation-only (`stage3_esm.md` §5). The runbook
  working as intended — the check caught something.
- **The frozen record is not sufficient to rebuild an arm bit-for-bit.**
  `stage3_runs.csv` captures configs, folds, seeds, feature widths and per-member
  epochs, and that was still not enough for one arm. "Reproducible from the
  committed artifacts" holds for the verified arms and is unproven in general.

### 4.5 Arms must be ensembled identically or the comparison is manufactured

Ensembling alone is worth **+0.074 mean SCC against the deployed single network**,
from no new information — paired Δ median ρ +0.083 [+0.029, +0.124]
(`stage2_baselines.md`). The +0.090 sometimes quoted is the ensemble minus the
mean of its own 30 members, a different reference; neither figure should be
quoted without saying which, and +0.074 is the one this parity rule needs. An
ensembled ESM-2 arm compared against a single-network sequence arm would produce
a result out of thin air. The rule is matched member count for every arm, or
none. The stage 3 comparator is the 30-network ensemble at median per-allele ρ
0.693, not the single network at 0.610.

---

## 5. What the negative results do and do not establish

### 5.1 Weak-binder augmentation (stage 2b)

**Established:** across eight paired comparisons, every 95% CI crosses zero and
excludes 0.05; best arm +0.024 [−0.026, +0.048]. Measured-affinity vs
predicted-affinity negatives at matched counts is −0.002 (w=0.1) and −0.017
(w=0.25). Detail in `stage2b_augmentation.md`.

**Bounded to:** 0.16 augmented rows per measured row against Rasmussen's ~2.7, a
16× volume difference, so volume is untested. The anchor analysis suggests volume
would not help — measured weak binders have near-canonical anchors (77.6%
hydrophobic at PΩ) and score near the floor, while predicted ones from random
peptides have the wrong anchors (49.1%) and score below it — but that is a
prediction, not a measurement. Also bounded to one architecture, two predeclared
weights, single networks, and a predicted pool that is the high-confidence tail.

### 5.2 Auxiliary affinity training (stage 2c, stage 3b)

**Stage 2c established:** across 20 paired comparisons (5 λ × 2 encodings ×
{single, ensemble}, plus a censoring-robustness variant), every CI crosses zero
and every upper bound sits below 0.05, largest +0.034. The comparison is clean by
construction: at λ=0 the multi-task network is bit-identical to the stage 2
baseline (verified on all 2,817 validation predictions). The affinity head
genuinely learns (ρ 0.55–0.62 on held-out affinity) while stability does not
move, and measured affinity used directly as a stability predictor ranks at
median per-allele ρ 0.580, below the 0.610 a single network reaches from
stability labels alone (`stage2c_affinity.md`).

**Bounded to:** a shared-trunk MLP on one-hot/BLOSUM peptide + pseudosequence
features, affinity labels on 26% of training rows. It does not rule out a
different sharing scheme, auxiliary encoding, or the broader 64,226-row
leakage-filtered corpus — that expansion is declined on evidence, not blocked.
The ceiling is computed on 33 training alleles with ≥ 30 dual-labelled pairs;
only 58 of 75 alleles carry any affinity data.

**The ESM-2 variant (stage 3b) is unresolved, not null.** The
difference-in-differences — does the auxiliary head help the ESM arm more than
the sequence arm — comes out +0.0120 / +0.0122 / +0.0143 / +0.0281 across λ,
every interval crossing zero, the movement almost entirely the sequence arm
degrading rather than ESM improving (`stage3b_esm_multitask.md`). The auxiliary
task was genuinely learned on ESM features (head ρ ≈ 0.57–0.58 at λ ≥ 0.1), and
at λ = 0 the mechanism is recorded exactly: 20 of 30 ESM-only members decay to a
literal constant.

> **The caveat travels with the verdict.** Every DiD upper bound (+0.0285 …
> +0.0422) sits below 0.05, so the frozen rule reads "rules out a worthwhile
> gain" — valid as a property of the intervals obtained. But the measured DiD
> power floor straddles 0.05 (single-delta MDE ≈ 0.037; DiD MDE ≈ 0.071,
> bracketed (0.032, 0.071]), and an injected DiD of −0.0319 was not detected. So
> **power at 0.05 is unestablished**: this is the only place in the project where
> a "rules out 0.05" verdict is not backed by demonstrated sensitivity at 0.05.

### 5.3 ESM-2 reaches parity and adds nothing — and what that does not mean

**Established** (`stage3_esm.md`): ESM-2 only, −0.0101 [−0.0382, +0.0352];
sequence + ESM-2, −0.0170 [−0.0468, +0.0320]. Both inconclusive at zero, both
ruling out +0.05. This is not a demonstration that ESM-2 is worse — both
intervals contain zero.

**The differential target sharpens this into an equivalence.** On cross-allele
ranking, where the peptide's own contribution cancels, ESM-2 is +0.0028 [−0.0052,
+0.0102] and sequence+ESM-2 is −0.0013 [−0.0091, +0.0058] — equivalent to within
one point of concordance over 10,365 allele-pair comparisons. The same
measurement conclusively separates the 150M checkpoint (−0.0088 [−0.0149,
−0.0025]) and the full-domain ensemble (−0.0124 [−0.0186, −0.0061]), so the null
is measured, not an artifact of a metric that cannot discriminate.

**The one positive result, which must travel with all three constraints.**
Against the full-domain sequence ensemble on matching input, ESM-2 is
conclusively ahead on two of three statistics: median per-allele ρ +0.0301
[−0.0084, +0.0757] (inconclusive), mean per-allele ρ +0.0312 [+0.0082, +0.0540]
(conclusive), differential concordance +0.0152 [+0.0046, +0.0241] (conclusive).
The bounded claim: given the same 182 HLA residues, ESM-2 extracts more usable
signal than one-hot or BLOSUM. It is a designed comparison — the full-domain arm
was added at stage 1, before any model existed, so "more input" could not be
mistaken for "benefit of pretraining". Three constraints, none droppable:

1. **Not a claim that ESM-2 beats the best sequence arm, nor a claim about
   pretraining.** The pseudosequence ensemble also conclusively beats the
   full-domain arm on differential concordance (+0.0124 [+0.0061, +0.0186] vs
   ESM-2's +0.0152, overlapping), while ESM-2 does not beat the pseudosequence
   ensemble (−0.0036 mean, +0.0028 concordance, both inconclusive). So the
   conclusive finding is that full-domain one-hot is a weak arm, not that
   pretraining is a strong one: hand-picking the 34 contact residues recovers
   what pretraining buys here.
2. **It does not establish the 0.05 bar.** The mean interval runs to +0.0540, so
   it admits values below the bar as well as above.
3. **Both conclusive verdicts are secondary statistics.** The contract's median —
   the only statistic the predeclared rule governs — stays inconclusive.

**Bounded to:** frozen embeddings of peptide and HLA taken separately, at 35M and
150M, with ridge and a small MLP head, under the frozen splits, on validation. It
does not establish:

- that fine-tuned ESM-2 would not help — nothing was fine-tuned;
- that a model shown the complex would not help. The head-side version is closed:
  a cross-attention arm letting the 34 groove residues condition the 9 peptide
  positions, against a mean-pooling ablation at identical parameter count (94,913,
  asserted at runtime), gives +0.0164 [−0.0178, +0.0437], and both coupled arms
  land 0.11–0.14 below concatenation — so "concatenation cannot use what ESM-2
  carries" is ruled out. The model-side version (a chimeric peptide-linker-groove
  input where ESM-2 itself sees the interaction) stays out of scope, specifically
  so this null is reported as bounded;
- that likelihood or perplexity features would not help — embeddings are one of
  three uses the brief names; the other two were not tested;
- that a larger checkpoint would not help. 150M lands at 0.6737, marginally below
  35M at 2.6× the cost, so the curve is flat across the two sizes tested, which
  weakens but does not close the scaling argument. 650M was excluded on a measured
  memory constraint (5.06 GB peak RSS alongside the production fold on a 16 GB
  machine), not on evidence;
- that another pLM family would behave the same way.

**Three controls make this reportable rather than an artifact.** Transplanting
the baseline's regularisation ladder would have cost 0.109 SCC and inflated seed
spread from 0.013 to 0.110 (tuning parity means equal budget, not equal values;
applied symmetrically, the baseline on its own extended ladder moves only +0.002,
inside float32 noise, §4.4); the memory-driven PCA helped by +0.033, so the
negative is not a compression artifact; and the stage 2 ensemble rebuilt through
the new harness byte-identically (same md5, Δ = +0.0000).

**A separate finding worth carrying:** with only 75 distinct HLA sequences, the
mean-pooled and 34-contact representations are both rank 74, both lossless at 74
components, and both separate all 75 alleles exactly — identical information. So
the 0.177-vs-0.499 gap between them is geometry, not information: where you read
the embedding from matters more than which model produced it.

**One indexing caveat, already measured.** `stage3_contact_index.json` records
the residue-indexing check (`exact_match: true`, 0 mismatched alleles), but flags
4 of the 34 pseudosequence positions ambiguous, because several domain columns
carry the identical residue in all 75 alleles and cannot be told apart from the
pseudosequence alone. The assignment for those four is a convention. It affects
which column an embedding is read from, not whether the right residue is read, so
the effect should be small — but it belongs beside the contact-position ESM-2 arm
if that is the one reported.

### 5.4 The additive null's mechanism remains unsettled

We know the additive arm does not help (§5.3). We do not know why, and one
proposed explanation was withdrawn after its own control refuted it
(`stage3_esm.md`).

**Established.** Permuting the ESM embeddings — destroying only the
sequence-to-embedding correspondence, at identical width, scale, marginals,
covariance and rank — takes the arm from 0.5679 to 0.0453. That is a **sign
test**: the network demonstrably uses the correspondence, so the ESM block is not
inert padding. It is not an effect size, because the permuted arm collapses
rather than degrades.

**What the collapse measures is pipeline fragility, not information content.** A
width-matched uninformative block here is not a capacity control; it is a
memorisation channel that breaks dev-based early stopping. Columns that uniquely
key the peptide or the row let the network drive training loss down without
generalising, so dev loss bottoms at epoch 1–5 instead of 27. Truncation alone
does not explain it: the baseline truncated to epoch 5 still reaches 0.5210 while
the shuffled arm at epoch 5 reaches 0.0453.

**The retraction.** An earlier claim held that ~300 dense columns cost ≈ −0.083
regardless of content, with ESM-2 recovering most of it — which would have made
the additive null a story about width, not ESM-2. The proper controls refute it:
shuffled and random blocks cost 0.45–0.54, not 0.083, nowhere near the
displacement regime. The claim never reached this document; the reviewer ran a
control capable of falsifying their own claim and it did.

**What survives, in narrow form:** ESM-2's 330 columns are more useful on this
baseline than 306 columns of BLOSUM positional cross-encoding. Nothing about what
pure width costs.

**One related question closes in our favour.** The ESM block's mean per-column
standard deviation is 4.788 (max 54.5) against the baseline's 0.084 — a ~57×
disparity. Rescaling components to unit variance makes the arm worse (0.5743 →
0.5049, against a 0.035–0.069 seed spread), because PCA component magnitude is
itself information: unit-rescaling hands the 256th component the same weight as
the 1st. The disparity is doing work, and the additive null is not a scaling
artifact. Run as a probe; the shipped arm was never refit.

### 5.5 ESMFold2's rejection is operational, not scientific

ESMFold2 failed its pre-registered gate on both sentinel criteria: median arm-B
heavy-atom RMSD 2.564 Å on `HLA-B*07:02`/IPRRNVATL against a 1.0 Å bar, and a
0.008 Å improvement over arm A against a 0.5 Å bar, with no construct effect in
any arm or seed (`stage4c_ectodomain_pilot.md`). Two limits on reading that as a
verdict on ESMFold2:

1. **Pose accuracy is not feature utility.** Nothing here shows ESMFold2 features
   would predict half-life worse; worse poses could still carry downstream
   signal. That experiment was not run, and the $884 full-cohort forecast is why.
2. **A five-complex gate is an operational decision rule, not a general claim.**
   Five complexes, three seeds, one construct family. Enough to decide where to
   spend $200; not a benchmark of ESMFold2 on peptide–MHC.

The cross-model comparison the matched plan was designed to produce was delivered
by the pilot, on identical constructs, MSA content, complexes and seeds. Dropping
ESMFold2 from production is a labelled revision of the agreed plan, not a silent
drop.

---

## 6. The structural arm

The Boltz-2 fold and the structural feature arm are **complete**, and the scored
structural arm is **conclusively worse**: sequence + Boltz-2 geometry +
confidence reaches test median 0.5947, Δ −0.112 [−0.134, −0.057] against the
sequence ensemble, interval entirely below zero ([`REPORT.md`](REPORT.md) §7).
That is a reported result, not a pending hole. The limitations below bound how it
was produced and read.

### 6.1 The cost figure is derived arithmetic, not a reconciled bill

The production fold ran 282 of 282 shards, 28,166 of 28,166 pairs, zero failures,
143.6 A10G-hours in 7 h 32 min for **$212.71**. The failure rate of 0.000 is a
final rate measured after the run. The surviving limitation is provenance: the
$212.71 is `derived` (realised container-hours off all 28,166 fold records ×
$1.4812/h), **never reconciled against a provider bill**, and `sofyaleyn` has no
metered figure at any point. The only metered check is the pre-launch one, which
landed within ~10% of the bill; taking a snapshot now would sweep in unrelated
CPU jobs. Tracked as hole B3a in `compute_ledger.md` §5. Discount the cost by
~10%, not by the possibility that the folds did not happen — 282 shard markers and
a whole-run scan of every fold record establish the quantity independently of any
billing figure.

### 6.2 The central bulge is several Ångströms uncertain

Both folding engines share one failure mode: error concentrates at central
peptide positions while anchors stay tight. The two-chain Boltz-2 pilot on
`HLA-B*07:02`/IPRRNVATL deviated 1.90 Å at P5 and 3.33 Å at P6 while P1–P2 and
P8–P9 were under 0.25 Å (`stage4_benchmark.md`). The three-chain construct
flattens this case to a 0.28 Å maximum across seeds, but the asymmetry is a
property to expect, not one shown to disappear: geometry features at central
positions (burial depth, contact counts) are noisier than at anchor positions,
and stage 5 should not treat all nine positions as equally well determined.

### 6.3 Confidence ranks error but cannot filter failures

Across the pilot's 45 Boltz-2 folds, confidence ranks error reasonably
(Spearman(peptide-to-groove PAE, heavy RMSD) = +0.635; Spearman(peptide pLDDT,
heavy RMSD) = −0.540) but does not isolate the one real failure. The three
failing arm-A sentinel folds sit at PAE 1.55 and peptide pLDDT 0.973–0.975 while
2.3 Å wrong, whereas the correct arm-B folds sit at worse PAE (1.88–1.99). No PAE
threshold catches the failure without flagging accurate predictions. Keep
confidence as a feature, never as a per-prediction failure filter
(`stage4c_ectodomain_pilot.md`).

### 6.4 Global ipTM is not peptide-interface confidence

In a three-chain complex, Boltz-2's global ipTM covers interfaces other than
peptide–HLA, notably the large, conserved, easy HLA–β2m. It must be labelled and
reported as global ipTM. A pair-specific score may be used only if the pinned
model emits one with a verified chain mapping. (ESMFold2 exposes
`pair_chains_iptm`, the peptide–HLA interface ipTM; Boltz-2 does not. If that
feature proves important, this is a reason to revisit ESMFold2 despite §5.5.)

### 6.5 B versus C does not isolate a mechanism

Arm B (ectodomain + β2m + peptide, new alignment) improves; arm C (groove +
peptide, the same new alignment rows cropped) tracks arm A including the
sentinel. So the benefit comes from the full construct with its extra
evolutionary information. This does not separate α3 from β2m and is not a
physical-mechanism claim. Separate α3/β2m ablations are out of scope this round.

### 6.6 Six of 75 ectodomain constructs are not exact matches

Three alleles have only a 181-aa groove-only IMGT record, so their α3 domain is
borrowed from a close relative: `HLA-A*02:50` ← `HLA-A*02:01` (177/182 α1/α2
identity), `HLA-A*24:19` ← `HLA-A*24:07` (174/182), `HLA-B*08:03` ←
`HLA-B*08:01` (176/182). The three C67S constructs take α3 from their wild type,
keeping α1/α2 from the dataset (`hla75_ectodomain_b2m.csv`, `note` column).
Structural features for these six rest on a partly inferred construct; α3 is
outside the groove so the effect should be small, but it is unmeasured.
`HLA-A*24:19` is also one of the alleles the sequence baseline ranks worst
(validation ρ = −0.12, `stage2_baselines.md`) — a coincidence to check, not a
claim.

### 6.7 The pilot is a pipeline check, not an accuracy benchmark

Five complexes from the training split, so training-set recall is possible. It
validates the pipeline, the pose scorer and the feature extraction; it does not
estimate generalisation, and A/B/C comparisons are limited to those 90 shared
folds. Better crystal agreement must still earn its predictive value on
validation.

### 6.8 Hardware measurements carry n=1 container variance

The two-chain GPU sweep used one container per GPU type against 73% measured
between-container variance (6.3 s vs 10.9 s for the same model and settings on the
same card). The A10 choice and H100 verdict survive because both are gaps larger
than the variance; the fine-grained ordering among the middle cards does not
(`stage4_benchmark.md` Finding 2). Those costs are computed from published
per-second rates, not invoiced, with the ~10% agreement at the metered workspace
as a check (`compute_ledger.md` §1).

---

## 7. Scope not covered

Recorded so absence is not mistaken for a result. Full rationale in
`HACKATHON_PLAN.md`, "Out of scope this round".

| Not run | Why | Status |
|---|---|---|
| ~~Allele-held-out evaluation~~ | Confounded by panel composition, not panel hold-out (§2.1) | **Run for the sequence arm** (stage 7b): near 0.741 vs distant 0.339, +0.403 [+0.223, +0.478]; separate contract, not comparable to frozen-split numbers. ESM-2 and structural arms not run, and must be refit, not scored from a `preds/*.csv` |
| ~~ProteinMPNN inverse-folding scores~~ | Was blocked on structures; the 4c pilot unblocked it | **Run** ($1.15) as a ~$1 QC triage sample (§7.2); declined as a regression feature on a measured n=5 label correlation of −0.100 |
| Chimeric peptide-linker-groove ESM-2 input | Far outside ESM-2's distribution | Out of scope; listed so the stage 3 null is reported as bounded (§5.3). Model-side coupling only — the head-side version was run and is closed |
| ~~Tobit / censored likelihood~~ | Was the 16-hour substitute's known gap | **Run, negative** (stage 7a): −0.0414 [−0.0780, −0.0062] on ranking, gains calibration only (§1.1) |
| FoldX / Rosetta / empirical ΔG | Equilibrium ΔG vs a kinetic label (§1.3), and both licence-gated | Declined — §7.0. FoldX ran a pilot only; the arm was not scored and its incremental value is open |
| Elution data as training augmentation | 5.4× scale mismatch, unknowable threshold, confounded with abundance/cleavage/TAP/ionisation | Rejected as training data; used as external validation, which ran — §7.1 |
| ESMFold2 production folds | Failed its gate; $884 forecast | §5.5 |
| Chai-1, Protenix, SaProt | Integration cost beyond the agreed comparison | Not evaluated |
| **ESM-IF** (the brief's second inverse-folding model) | Machine contention — ESM-2 held priority | Not attempted. ProteinMPNN covers the inverse-folding class; this specific model is not |
| Separate α3 / β2m ablations | Deferred until the pipeline's predictive value is established | §6.5 |

### 7.0 FoldX and Rosetta — declined for two independent reasons

1. **The scientific objection.** Both estimate an equilibrium free energy, ΔG;
   our label is kinetic, a dissociation half-life governed by ΔG‡. Two complexes
   at the same ΔG can come apart at completely different rates. This is a known
   label mismatch, not a setup-time problem, and does not go away with more
   compute. It is the same objection against per-pocket energy decomposition
   (premised on FoldX `AnalyseComplex`) and OpenMM minimisation energy. The
   softer form applies to ProteinMPNN (§7.2).
2. **Both are licence-gated behind registration.** Neither installs from a public
   package index without an account, so including them would make the pipeline
   less reproducible — every other model here (Boltz-2, ESMFold2, ESM-2,
   ProteinMPNN) is openly downloadable, with the ProteinMPNN checkpoint pinned by
   SHA-256.

Reason 1 alone is sufficient; reason 2 is the one a reader is more likely to hit.
A FoldX pilot ran (`stage8_foldx.md`), but the arm was never scored, so its
incremental predictive value stays open.

### 7.1 The elution external validation has its own error rate

The MHC Motif Atlas comparison is a genuine external check — only 140 of 151,170
9-mers overlap, so the peptide universes are near-disjoint. Eluted ligands show
~7× higher median half-life (7.90 h vs 1.10 h) and a tenfold drop in the
zero-stability fraction (2.1% vs 21.2%), Mann–Whitney p = 6.0 × 10⁻³¹. But it is
probabilistic: six eluted ligands have measured half-life < 0.5 h, three at
exactly zero, and `FPEHIFPAL` appears at zero on two alleles — more consistent
with motif-deconvolution error than biology, putting the atlas error rate at
roughly 2% (`elution_stability_finding.md`).

The transfer pass ran (`stage3c_elution_validation.md`, 51 alleles, 81,600
ligands). Five limits attach to its headline AUROC of 0.9656, none droppable when
the number is quoted:

1. **It is not stability-prediction accuracy.** Elution has at least four filters
   besides stability (source-protein abundance, proteasomal cleavage, TAP
   transport, mass-spec ionisation). The allele-swapped control bounds this
   contribution at 0.050 of AUROC — a bound on this decoy design, not a
   decomposition of the biology.
2. **It is not a ranking of stability within presented peptides.** Ligand vs
   non-ligand is binary discrimination; 0.610 / 0.693 remain the numbers to quote
   for accuracy.
3. **It is not comparable to a trained presentation predictor.** NetMHCpan-4.x
   trains on elution data; this model never has. The reading is "a stability
   model transfers", not "competes with a presentation model".
4. **The positive class carries the ~2% error rate above**, a soft ceiling just
   under 1.0 on any AUROC here, so the weakest alleles may partly measure
   motif-deconvolution quality.
5. **Decoys are assumed negatives** (mass spec is positives-only), which
   contaminates the negative class and depresses rather than inflates the AUROC —
   conservative in that one respect. One release, one decoy draw, seed
   sensitivity not swept.

The donor-distance gradient (0.8906 near vs 0.9562 far) is real in direction and
size, but only two bins were measured, so the shape of the decline is not
established. Open follow-on: the same pass on the ESM-2 and structural arms.

### 7.2 ProteinMPNN pose triage rests on n = 1 complex

`stage5_inverse_folding.md`. Four load-bearing limits:

- **The sample is one failing complex, not six folds.** All six Boltz-2 folds
  above 2.0 Å peptide heavy RMSD are `HLA-B*07:02`/IPRRNVATL, arms A and C at
  three seeds. "Six of 45" would overstate it as six independent failures.
- **It is not a half-life predictor and was not shown to be one.**
  Spearman(`pep_ll_total`, t½) = −0.100 at p = 0.87, n = 5, all training rows —
  zero inferential weight. The full-cohort regression scoring is declined on that
  evidence; its value as a kinetic feature is untested.
- **The claim is specificity, not separation.** Repeating the separation test on
  every complex yields 7, 9, 2, 13 and 2 "fully separating" structural features,
  the genuinely failing complex scoring fewest — so clean separation of a
  six-fold group measures complex identity, and none of the 109 structural
  features isolates the failure. `pep_ll_mean` escapes that artifact only because
  it was a single predeclared quantity with a within-complex control.
- **Thermodynamic-flavoured, kinetic label.** Sequence–backbone compatibility is
  the softer form of the §7.0 mismatch.

Its seed control disagrees with the ESMFold2 seed control (109 structural
features give 5.29 Boltz-2 / 5.87 ESMFold2; ProteinMPNN gives 3.1–9.3 / 1.2–1.9),
so the two must not be cited as confirming each other — though they agree on
Boltz-2, which is what production runs. The divergence is granularity:
ProteinMPNN reads backbone only at fine resolution while the structural features
are coarse aggregates, and ESMFold2's CA jitter is 2.5–5.2× Boltz-2's. An earlier
attribution to side-chain sensitivity, and a claim that the two implementations
reconciled to two decimals, are both retracted. Quote the ProteinMPNN ratio as an
envelope (3.1–9.3, 1.2–1.9), not a point estimate; all aggregations return the
same verdict, and the ESMFold2 range that carries the verdict agrees exactly.

**Operational consequence:** the QC sample's reference points are descriptive,
from a single complex. They must not filter the production cohort, nothing
downstream may condition on them, and the deliverable is a distribution and a
triage list — never a failure count or rate — until the signal is checked on more
than one failing complex.

### 7.3 Three of our own measurements cannot do the job we gave them

These are failures of our design choices, not of the models — the instruments
were graded alongside the arms (`stage3b_esm_multitask.md`).

**The contract's primary metric is underpowered here.** Across the whole ESM run
(8 paired comparisons, 12 stratum intervals, 1 nested row) the median per-allele
Spearman returned 0 conclusive verdicts, while the mean returned 4 and the
differential concordance 5. A median over a 68-allele panel is robust, and
robustness costs power. We predeclared it and keep it — changing the primary
metric after seeing which one resolves things is the post-hoc selection the
contract prevents — but a future study should predeclare the differential as
primary.

**The distance-stratification contrast is underpowered by construction.** On
validation, all four stratum-gap intervals cross zero (ESM-2 35M −0.0197 [−0.0973,
+0.0800]; sequence+ESM-2 +0.0046 [−0.0803, +0.1235]; 150M −0.0685 [−0.1188,
+0.0625]; full-domain −0.0091 [−0.0888, +0.0847]). The stratum gap is a
difference of differences of medians, and each step compounds noise: the
half-width is near 0.09 on a quantity whose largest observed value across all five
arms is 0.069. The measurement cannot resolve the effect it exists to detect, so
"pretraining buys generalisation at the split boundary" is neither supported nor
refuted here. The question was asked three ways and the split answered none:

| Form of the question | Arms | Result |
|---|---:|---|
| Stratum gap (difference of differences) | 4 | all cross zero |
| Within d=4, arm vs baseline directly | 4 | all cross zero |
| Within d≥5, arm vs baseline directly | 3 of 4 | all cross zero |

The decisive single difference — does the larger checkpoint beat the baseline
where peptides are most distant — is ESM-2 150M at d≥5: +0.0219 [−0.0776,
+0.0619], inconclusive. Eleven of twelve intervals are in and not one separates
any arm from the baseline in either stratum. Settling this needs a better-powered
split, not a re-analysis of this one. Two reading traps: the 150M d≥5 point
estimate sits high in its own interval because the statistic is a median over a
55-allele panel rebuilt on every resample (skewed bootstrap — never read it
without its interval); and the 150M arm is both flattest across distance and
uniformly weakest, so "nothing to lose" and "genuinely distance-robust" are not
separable here.

**The flat-profile hypothesis was generated on validation, not predeclared.** The
idea that a larger checkpoint might have a flatter distance profile was read off
the point estimates above, so if it reappears in the test results it is a
validation-generated hypothesis tested once, not a predeclared prediction
confirmed, and must be labelled that way; if it fails to reappear, that is equally
a single test of a post-hoc idea. `HACKATHON_PLAN.md` records this, as it does for
any pattern first noticed on validation.

**The nested table covers four arms, not five — declined deliberately.** The 150M
arm has no out-of-fold prediction file. Generating one was declined, not
overlooked: it is the weakest arm on every measurement, the analysis is
underpowered (490 comparisons on 96 clusters, half-widths near 0.05), and a fifth
inconclusive row would add a number without information. Cost of closing it:
about seven minutes.

**Precision@10 is quantised past the point of usefulness** — a criticism of a
metric we predeclared. Against the sequence baseline, all four ESM comparisons
returned a delta of exactly 0.0000, with interval bounds that are themselves
lattice points. The statistic moves in steps of 0.1 (ten slots), and a median
over 68 alleles of a 0.1-quantised quantity sits on a lattice point and stays
there under resampling, so it cannot express the 0.04 differences the primary
metric reports. Its only informative columns, lift and ceiling share, span
0.277–0.294 across those arms, within noise here. It separated the single network
from the ensemble at stage 2, so it is useless at this resolution, not in general.
It stays in the report because it was predeclared; no claim rests on it.

---

## 8. Open holes in this register

| Tag | What is missing | Who fills it |
|---|---|---|
| ~~E3~~ | ~~Stage 3's outcome and bounds~~ | **Filled**: §5.3 |
| ~~B3a~~ → B3a | Provider-bill reconciliation of the $212.71 derived fold cost | still open; `compute_ledger.md` §5, §6.1 |
| ~~B4~~ | ~~Stage 5 structural feature coverage and the scored arm~~ | **Filled**: §6, `REPORT.md` §7 (scored arm conclusively worse) |
| ~~S7~~ | ~~Whether d=4 and d≥5 test strata separate~~ | **Filled**: §3.1 (all six arms score higher at d=4; no paired interval on the gap, which stays the residual limitation) |
| **S8** | Any allele where the test panel disagrees sharply with validation | `eval-harness` |
| ~~S7a~~ | ~~Censored-likelihood result~~ | **Filled**: §1.1 |
| **A1** | Per-stratum leave-allele-out for the ESM-2 and structural arms (§2.1). MDE is 0.025–0.030 when per-allele deltas are tight and 0.075–0.115 when loose, so a stratum may come back inconclusive and must be reported as such | `esm-arm` / stage 5 |
| ~~S3C~~ | ~~Elution external-validation write-up~~ | **Filled**: `stage3c_elution_validation.md`, §7.1. Open follow-on: the same pass on the ESM-2 and structural arms |
| **P1** | Whether the ProteinMPNN pose-triage signal survives a second failing complex (§7.2) | declined for this round |
| **A3** | Whether the six partly-inferred ectodomain constructs (§6.6) behave differently at stage 5 | stage 5 |
