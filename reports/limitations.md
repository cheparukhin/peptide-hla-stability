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
build, not as unnecessary (`HACKATHON_PLAN.md`, out-of-scope register), and is
**now in flight as stage 7a** — `reports/stage7_censored.md` predeclares the
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
2. **It is why the allele-held-out evaluation is not the headline.** Holding out
   an allele simultaneously holds out its peptide panel, so an apparent
   allele-distance effect is inseparable from a peptide-panel effect. **This
   confound does not disappear by being measured** — if that evaluation is run,
   it must be reported alongside. This reason is specific to this dataset and is
   the one that matters; the other three (different question, second split
   needed, skewed allele coverage) are weaker
   (`HACKATHON_PLAN.md`, "Why the allele-axis hold-out is not the headline").
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

### 4.1 NetMHCstabpan is calibration, never a comparator

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

### 4.2 Validation scores are selection scores

Every number quoted from stages 2, 2b and 2c is a **validation** score, and each
arm's configuration was chosen on the number reported beside it. They are
optimistic as held-out estimates. They remain valid for *comparing* arms under
the same budget, which is what they are for. λ in stage 2c was swept on
validation, which is why the whole sweep is tabulated rather than its maximum:
the peak one-hot cell beats the control by 0.004, a sixth of the seed spread
(`stage2c_affinity.md`).

### 4.3 Seed spread sets a floor on what counts as a difference

Seed-to-seed spread for the selected MLP configs is **0.010–0.051**
(`stage2_baselines.md`). Gaps under ~0.05 are within seed noise, which is the
same order as the predeclared bar — both reflect what this dataset can resolve.
**Compare seed means, never single seeds.**

### 4.4 Arms must be ensembled identically or the comparison is manufactured

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

### 5.3 "ESM-2 didn't help" — what that would and would not mean ‹HOLE E3›

*Stage 3 is in flight. If it returns a null, the following bounds apply and must
be stated with it.*

A null here would be bounded to **the separate-embedding representation actually
tested**: peptide and HLA embedded independently, so the regression head has to
learn the peptide–HLA interaction itself from two unconditioned vectors. It
would **not** be a result about foundation models in general, nor about ESM-2 in
general. The sharpest untested version is a chimeric peptide-linker-groove
input, where ESM-2 sees the interaction directly — explicitly listed as out of
scope *so that a negative result is reported as bounded*, because a linkered
9-mer construct sits far outside ESM-2's training distribution
(`HACKATHON_PLAN.md`, out-of-scope register). Also bounded to the checkpoint and
layer selected, the head architectures tried, and the tuning budget given.

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
| Allele-held-out evaluation | Confounded by allele-specific peptide panels (§2.1); different question from the headline claim | Deferred; the stratum where pretraining has the strongest prior of winning |
| ~~ProteinMPNN inverse-folding scores~~ | Was blocked on needing structures; the stage 4c pilot's 90 folds unblocked it | **Run** — pilot complete, §7.2. Kept as a ~$1 QC triage sample; **declined as a regression feature** on a measured n=5 label correlation of −0.100 |
| Chimeric peptide-linker-groove ESM-2 input | Far outside ESM-2's distribution | **Listed specifically so a stage 3 null is reported as bounded** (§5.3) |
| ~~Tobit / censored likelihood~~ | Was the 16-hour substitute's known gap | **Now in flight as stage 7a**, protocol predeclared, results pending (§1.1, hole S7a) |
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

**Operational consequence:** the QC sample's reference points are descriptive,
drawn from a single complex. They **must not filter the production cohort**,
nothing downstream may condition on them, and the deliverable is a distribution
and a triage list — **never a failure count or a failure rate** — until the
signal has been checked on more than one failing complex.

---

## 8. Open holes in this register

| Tag | What is missing | Who fills it |
|---|---|---|
| **E3** | Stage 3's actual outcome, and the bounds in §5.3 restated against what was tested (checkpoint, layer, head, budget) | `esm-arm` |
| **B4** | Stage 5 structural feature coverage and failure rate; what the declared sequence fallback covers | blocked on 4c production |
| **S7** | Whether d=4 and d≥5 test strata separate (§3.1), and the nested near-neighbour result (§3.2) | `eval-harness` |
| **S8** | Any allele where the test panel disagrees sharply with validation — hard alleles vs overfitting to the validation panel | `eval-harness` |
| **S7a** | The censored-likelihood result and its `c_hours` sensitivity sweep (§1.1) | stage 7a |
| ~~S3C~~ | ~~Elution external-validation write-up~~ | **Filled**: `stage3c_elution_validation.md`, §7.1. Open as a follow-on: the same pass on the ESM-2 and structural arms |
| **P1** | Whether the ProteinMPNN pose-triage signal survives a second failing complex (§7.2) | blocked on 4c production + the QC sample |
| **A3** | Whether the six partly-inferred ectodomain constructs (§6.6) behave differently at stage 5 | stage 5 |
