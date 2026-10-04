# Are protein foundation models useful for predicting peptide–HLA stability?

**London AI × Science Hackathon — Protein Engineering Track, 3–4 October 2026**
Team of 3. Dataset: Rasmussen et al. 2016, 28,166 measured peptide–HLA
dissociation half-lives.

Companion documents: [`compute_ledger.md`](compute_ledger.md) (every dollar and
GPU-hour), [`limitations.md`](limitations.md) (everything to discount this by),
[`EVALUATION.md`](../EVALUATION.md) (the contract, frozen before any model
existed), [`figures/`](figures/).

> **Status at time of writing.** Stages 1, 2, 2b, 2c, 3c, 4a, 4b and the stage
> 4c pilot are complete and reported below with sources. Three results are in
> flight and appear as explicitly marked holes — `‹HOLE …›` — never as estimates.
> Nothing in this document is a predicted number.

**Where the judging criteria are answered.** The brief asks for *"a watertight
evaluation considering ML best practices, engineering & compute requirements
during the timeframe, and incorporation of the biological background of the
problem"*:

| Criterion | Sections |
|---|---|
| Watertight evaluation / ML best practice | §2 (contract frozen before any model), §2.4 (disclosed breach), §6 (five evaluations beyond the headline) |
| Engineering & compute requirements | §7 and [`compute_ledger.md`](compute_ledger.md) — every dollar, GPU-hour and forecast, with the decisions they drove |
| Biological background | §1, §3 (why within-allele ranking is the decision), §4.1 (anchor chemistry), §4.3 (why β2m matters), §4.4 (external transfer to a different assay) |
| Negative results, well supported | §4, and [`limitations.md`](limitations.md) §0 and §5 — what each null does and does not establish |

---

## 1. The question

HLA molecules sit on the surface of nearly every cell and hold up short protein
fragments — peptides — for inspection by T cells. The question is not *whether*
a peptide binds but **how long it stays bound once it has**: the dissociation
half-life, which is the practically relevant quantity for designing a vaccine or
a T-cell therapy, because a complex that falls apart in minutes is never seen by
the immune system.

The challenge asks a narrower and sharper question than "can you predict this
well":

> *"We are asking the question 'Are they useful for this problem', not 'Please
> prove they are useful for this problem'."*
> — challenge brief, §5

So this is a **cost-benefit study with a predeclared decision rule**, not a
leaderboard entry. Every additional information source is tested against a
trained sequence baseline, and asked the same question: does it improve ranking
by enough to be worth what it costs?

| Source | What it adds | Price per 1,000 new predictions |
|---|---|---:|
| Labelled sequences (the baseline) | nothing — this is the reference point | $2.0 × 10⁻⁷ |
| **ESM-2** (a protein language model) | learned representations of peptide and HLA | **$4.4 × 10⁻⁴** extraction (measured); head cost ‹HOLE E1b› |
| **Boltz-2** (a structure predictor) | a predicted 3D complex, plus the model's own confidence | **$6.90** |

The spread in that right-hand column is roughly **seven orders of magnitude**.
That is the entire study in one line: a structural arm does not need to be
*better* than the sequence baseline, it needs to be better by enough that
someone would pay seven orders of magnitude for the difference.

---

## 2. Why this split and this metric — and why they were frozen first

**This is the strongest methodological claim in the submission, so it goes
first.** The evaluation contract in [`EVALUATION.md`](../EVALUATION.md) was
written and committed at stage 1, **before any model existed**. It fixes the
splits, the primary metric, the eligible allele panel, the uncertainty
procedure, the scoring rule for degenerate cases, and — critically — **the
threshold below which a difference does not count**. Nothing in it has changed
after seeing a result.

The brief asked for exactly this: *"We provide no splits in the dataset
specifically to avoid a leaderboard approach; we would strongly recommend
establishing splits early on with a sensible hypothesis."*

### 2.1 The split: group peptides, not rows

Random row splitting would be meaningless here. 871 of 5,633 peptides (15.5%)
have a neighbour within 3 substitutions, and the same peptide appears on up to
36 different alleles. Split rows at random and near-duplicate peptides land on
both sides, so the model is scored partly on memorisation.

So peptides are grouped by **single-linkage clustering at Hamming distance ≤ 3**
(two peptides join a group if they differ in at most 3 of 9 positions, and
groups chain transitively), and every group lands wholly in one split, across
all alleles. All peptides are 9 residues, so Hamming distance is exact and needs
no alignment.

| Split | Rows | Share | Peptide clusters | Peptides |
|---|---:|---:|---:|---:|
| train | 19,716 | 70.00% | 3,587 | 3,942 |
| validation | 2,817 | 10.00% | 510 | 567 |
| test | 5,633 | 20.00% | 1,023 | 1,124 |

Cross-split clusters: **0**. Minimum Hamming distance between any held-out
peptide and any training peptide: **4**. Source:
[`audit_summary.md`](audit_summary.md) §5.

Three choices here are worth defending because each could have gone differently:

- **Hamming ≤ 3 is the most conservative threshold that still works.**
  Single-linkage percolates sharply between 3 and 4: at ≤ 3 the largest
  component holds 22 peptides (0.56% of pairs); at ≤ 4 one component swallows
  2,807 peptides (54.8% of pairs) and a balanced split becomes impossible.
- **Plain Hamming, not a BLOSUM-weighted or embedding distance.** An ESM-2
  distance would make the split depend on the model under test at stage 3. At a
  matched threshold BLOSUM62 reproduces essentially the same partition (5,493 vs
  5,494 clusters at Hamming ≤ 1) while adding a threshold to defend. **The split
  must stay model-independent.**
- **The water-fill ranks splits by deficit *relative to target*, not absolute
  deficit.** This sounds like an implementation detail and is not. Each allele
  was assayed on its own peptide panel, so cluster placement decides per-allele
  held-out coverage. Ranking by absolute deficit degenerates: once the three
  deficits equalise, clusters cycle round-robin and every heavy cluster lands in
  whichever split led early. That reached 70/10/20 overall while leaving
  `HLA-B*42:01`, `HLA-B*51:01` and `HLA-B*81:01` — ~350 pairs each — with **zero
  test rows**. The proportional criterion fixes this at no cost to the totals or
  the distance guarantee.

**What grouping does not do** is remove label leakage; it reduces it. Same-allele
label similarity decays smoothly with distance rather than in a step — at d=4,
the minimum across splits, Spearman is still 0.592, against 0.725 at d=1 and
0.302 for unrelated pairs. No feasible threshold removes this. Stage 6 therefore
**stratifies test metrics by distance to training** (d=4: 57.8% of test rows;
d≥5: 42.2%) so the reader can see whether a model is exploiting it.

### 2.2 The metric: rank within allele, because that is the decision being made

**Primary: median per-allele Spearman ρ.** Rank-correlate predicted against
measured half-life *within each HLA allele*, then take the median across
eligible alleles.

Why within-allele, when a pooled correlation would look better? Because pooling
mixes two different things. Different alleles have different typical half-lives;
a model that knows nothing about peptides but knows which allele is which
already scores pooled ρ = 0.554. The decision a user actually makes is *"given
this patient's HLA, which of these candidate peptides will stick longest"* —
that is a within-allele ranking, and a within-allele metric is the only one that
measures it. **The training allele mean scores 0.000 on the primary metric and
0.554 pooled.** That gap is why pooled correlation is reported as secondary.

Rank correlation, not error, for a second reason: **20.2% of the labels are at
an assay detection floor** (§6.1 and [`limitations.md`](limitations.md) §1.1).
Spearman with average ranks treats them as one honest tied block. A tie-blind
correlation would flatter every model.

**Secondary metrics**, each covering a different failure mode:

- **MAE on `log1p` half-life** — numerical error. Reported with the warning that
  at the floor it is error against a *recorded* label with uncertain sign, and is
  therefore neither an upper nor a lower bound on true error.
- **Precision@10 at 2 hours** — of each allele's 10 top-ranked predictions, how
  many actually exceed 2 hours? This is the metric that matches how a shortlist
  is really used. Ties are credited by expectation, so a constant predictor
  scores its base rate rather than being flattered by row order.
- **Pooled Spearman and Pearson** — secondary, for the reason above.

**Degenerate cases were ruled on in advance**, because ruling on them afterwards
is where leaderboard-chasing hides. A constant prediction on an eligible allele
scores **0**, and the allele stays in the panel — dropping it would let a model
raise its median by going constant on hard alleles. NaN or infinite predictions
are rejected before scoring, not silently absorbed.

### 2.3 The bar: 0.05, derived from the test set rather than chosen

**Predeclared minimum worthwhile gain: Δ median per-allele Spearman = 0.05.**

This is not a preference. It is what this test set can resolve:

- Shuffling labels within each allele gives median ρ = 0.000 ± 0.018, 95% range
  [−0.036, +0.035].
- A paired cluster bootstrap under a true zero difference has a 95% CI
  half-width of 0.037–0.053.

Below ~0.05, this benchmark cannot distinguish a difference from noise
([`audit_summary.md`](audit_summary.md) §6). Uncertainty is a **paired cluster
bootstrap**: resample whole peptide clusters with replacement, score both models
on the same resample, report the *difference* and its 95% CI rather than two
separate intervals. Peptide clusters are the independence unit because one
peptide can appear on 36 alleles, so a row-level bootstrap would understate
uncertainty. 2,000 resamples, seed `20261003`.

And the six verdicts a confidence interval can produce were written down in
advance, so none of them can be chosen after the fact:

| Paired 95% CI for Δ | Verdict |
|---|---|
| upper < 0 | Worse |
| lower > 0.05 | **Meets the bar** |
| Excludes 0, upper < 0.05 | Real but too small to matter |
| Excludes 0, straddles 0.05 | Real, size unresolved |
| Crosses 0, upper < 0.05 | **Inconclusive, and rules out a worthwhile gain** |
| Crosses 0, upper ≥ 0.05 | Inconclusive |

The last two rows are the ones that matter for this submission. **"Inconclusive"
and "ruled out" are different claims and both are weaker than "no effect".** All
of our negative results are row 5: we cannot say the addition helps at all, and
we *can* say it does not help by the margin that would matter. We never claim
the effect is zero.

### 2.4 One disclosed breach of the single-use test rule

A superseded diagnostic re-partitioned the whole dataset and pulled 3,350 frozen
test rows into fitting, 448 into early stopping and 585 into a scored set. What
was observed was eight panel aggregates from two models, on an evaluation set
21% of which was test rows — no per-row prediction, no per-allele test score and
no individual test label. Its conclusion (a +0.018 attribution to split
grouping) **has been retracted**; measured properly it is −0.005 [−0.035,
+0.046].

No model that will be scored at stage 6 saw a test row. This is recorded in
[`EVALUATION.md`](../EVALUATION.md) rather than quietly fixed, **because the
audit trail is what makes "scored once" mean anything.** A reader who judges it
differently has the numbers to discount with.

---

## 3. The baseline

The brief recommends one: *"A simple supervised neural network trained on
peptide and HLA pairs would be a helpful baseline to produce to contextualise
the results on your splits."* We built six, then built the strong form.

**The arms.** {one-hot, BLOSUM62 encoding} × {peptide only, peptide + the
34-residue HLA contact pseudosequence, peptide + the full 182-residue HLA
domain}. Each residue becomes a 20-dimensional vector, concatenated in sequence
order so position information survives. Every arm got the same budget: a 4-point
MLP grid at 3 seeds plus a 5-point ridge sweep. All train on the same 17,744
rows (10% of train held out as a stopping fold, cut along whole peptide
clusters so early stopping cannot leak either).

Validation median per-allele ρ, mean over 3 seeds
([`stage2_baselines.md`](stage2_baselines.md)):

| Arm | MLP | Ridge |
|---|---:|---:|
| peptide + pseudosequence (one-hot) | **0.610** | 0.278 |
| peptide + pseudosequence (BLOSUM62) | 0.603 | 0.270 |
| peptide + domain (one-hot) | 0.574 | 0.274 |
| peptide + domain (BLOSUM62) | 0.521 | 0.259 |
| peptide only | 0.202–0.244 | 0.169–0.170 |
| training allele mean | 0.000 | — |

Four things this establishes, each with a paired cluster-bootstrap interval:

1. **The model ranks within allele; the allele mean cannot.** +0.610
   [+0.557, +0.656]. MAE 0.734 → 0.517; precision@10 at 2 h 0.359 → 0.70.
2. **Nonlinearity is most of the model.** Ridge on identical features reaches
   0.278; MLP − ridge = **+0.331 [+0.256, +0.414]**, the largest effect in the
   study. A linear model on one-hot residues is a position weight matrix — it
   cannot represent the interaction between a peptide residue and the HLA pocket
   that residue sits in. *The single most valuable thing we added was not a
   foundation model; it was a hidden layer.*
3. **The HLA input contributes the other half.** Peptide alone 0.202; adding 34
   contact residues: **+0.404 [+0.300, +0.474]**.
4. **34 contact residues vs 182 domain residues is unresolved.** +0.016
   [−0.031, +0.084]. This matters for stage 3: the full-domain arm is a
   legitimate matching baseline, so **any win for HLA-domain embeddings must be
   checked against it**, not only against the pseudosequence arm — otherwise
   "more input sequence" gets mistaken for "benefit of pretraining".

### 3.1 The baseline a foundation model actually has to beat

Single networks are a weakened form of the NetMHC-family method, which fits one
network per CV fold per architecture and predicts with the ensemble. Built
properly — **5 inner CV folds × 2 encodings × 3 seeds = 30 networks**, folds cut
along whole clusters, configs reused unchanged from the single-network selection
so ensembling is the only variable:

| Model | Median per-allele ρ | Mean SCC | MAE log1p | P@10 |
|---|---:|---:|---:|---:|
| Single network, pep + pseudoseq | 0.610 | 0.573 | 0.517 | 0.70 |
| **30-network ensemble, pep + pseudoseq** | **0.693** | **0.645** | **0.473** | 0.70 |
| 30-network ensemble, pep + domain | 0.653 | 0.610 | 0.495 | 0.70 |

ρ, SCC and MAE from [`stage2_baselines.md`](stage2_baselines.md); the P@10
column is computed from the committed prediction files with
`scripts/evaluate.py --split val`. **Precision@10 does not separate these three
arms** — all reach 0.70 — which is worth saying rather than hiding: the ranking
gain the median ρ reports does not translate into a better top-10 shortlist at
this panel size, and a submission that led with P@10 would have called the
ensemble a wash.

**Ensembling alone is worth +0.090 mean SCC, from no new information at all** —
paired Δ median ρ = +0.083 [+0.029, +0.124]. This produces the project's single
most important procedural rule:

> **Ensemble every arm identically, or ensemble none of them.** An ensembled
> ESM-2 arm compared against a single-network sequence arm would manufacture a
> result worth nearly twice the predeclared bar out of nothing.

So **0.693 is the number to beat**, not 0.610, and every later arm uses the same
`cv_folds()` protocol and the same member count.

### 3.2 NetMHCstabpan is calibration, never a comparator

NetMHCstabpan was trained on **all 28,166 rows, including every peptide in our
test split**. Any score it produces on our data is memorisation. The brief says
so itself. **There is no valid "beat NetMHCstabpan" result available from this
dataset at any stage**, and we do not claim one.

Its published 0.69 mean per-allotype SCC is a 5-fold cross-validation score on
its own training data. We use it to sanity-check that our baseline is in the
right family, nothing more — and the comparison is not like-for-like on any
axis: their training set is **5.2× ours and 73% augmentation we do not have**
(28,166 measured rows plus ~1,000 assumed-zero weak binders per allele), they
use BLOSUM50 and smoothed sparse encoding, single hidden layers of 40/50/60, a
`2^(−t0/th)` target, and they score on the same fold each network stopped on.
Their published **0.676 PCC must never be quoted as a bar**: it was computed on
data padded with ~75,000 easy synthetic negatives, which inflates correlation
substantially.

Of the factors we *can* isolate: split grouping shows **no conclusive
advantage** at equal row count (Δ median ρ −0.005 [−0.035, +0.046]; Δ mean ρ
−0.000 [−0.024, +0.024]) — read as "no conclusive advantage found", not "no
effect", since both intervals still admit a modest positive. Their target
transform is 0.022–0.034 *worse* here. Ensembling is the one factor that clearly
moves us. **What we have is a strong sequence baseline in the NetMHCstabpan
family, not a reproduction of it.**

---

## 4. What we tested, and what we ruled out

Three cheap hypotheses were tested before any GPU was booked. All three came
back negative, **and the intervals are tight enough to say what kind of negative**.

### 4.1 Weak-binder augmentation does not help, and we know why

Rasmussen et al. padded their training set with ~1,000 predicted weak binders
per allele, labelled 0 h. Does that mechanism work, and does the *source* of the
negatives matter? Two arms at matched per-allele counts (2,910 rows each, 51
alleles, capped at 25% of each allele's fit rows): **measured** weak binders from
public affinity data, and **predicted** weak binders from random human peptides.

**All eight paired intervals cross zero and exclude 0.05.** Best arm +0.024
[−0.026, +0.048]. Source comparison at matched counts: −0.002 (w=0.1), −0.017
(w=0.25) — inconclusive and bounded below the bar in both directions.

**The mechanism is the finding.** HLA class I pins a peptide by two anchor
residues — position 2 and the C-terminus — which largely decide whether it can
enter the groove at all.

| Pool | PΩ hydrophobic | PΩ charged | What the baseline already predicts |
|---|---:|---:|---:|
| validation rows at the assay floor | 84.2% | 14.3% | +0.218 |
| validation rows above the floor | 84.0% | 13.6% | +1.103 |
| measured weak binders | 77.6% | 14.3% | +0.126 |
| predicted weak binders (random peptides) | 49.1% | 25.9% | **−0.071** |

Read the first two rows together: **real zero-stability peptides have perfectly
normal anchors.** What separates a 0-hour peptide from a 38-hour one in this
dataset is not whether it can enter the groove. Measured weak binders also look
like binders, and the model already scores them near the floor. Predicted weak
binders from random peptides are *anchor violators*, and the model already
scores them **below** the floor — they are easier than the easiest real data.
**More easy negatives are still easy negatives.**

Bounded to: 0.16 augmented rows per measured row against Rasmussen's ~2.7.
**Volume is untested**; the anchor analysis predicts it would not help, but that
is a prediction. ([`stage2b_augmentation.md`](stage2b_augmentation.md))

### 4.2 Auxiliary affinity training does not help, and we know the ceiling

**7,281 pairs across 58 allotypes** carry both an affinity measurement (how
strongly a peptide binds) and a half-life (how long it stays) — 5,135 of them in
the training split. Does training a second prediction head on affinity improve
the stability head? **20 paired comparisons — 5 λ settings × 2 encodings
× {single network, 30-network ensemble}, plus a censoring-robustness variant.
Every 95% CI crosses zero and every upper bound sits below 0.05. Largest upper
bound +0.034.**

The comparison is controlled *by construction*: at λ = 0 the multi-task network
is **bit-identical** to the stage 2 baseline — asserted on the real 860-column
feature grid, and verified to agree to 0.0 on all 2,817 validation predictions.
The single-task arm is not a near-replica of the baseline; it *is* the baseline.

Two diagnostics turn this from an ambiguous null into a clean one:

- **The auxiliary task was genuinely learned.** The affinity head reaches
  ρ 0.55–0.62 against held-out affinity labels, against ≈ 0 at λ = 0. The shared
  trunk learns affinity about as well as it learns stability — and the stability
  predictions still do not move. Mean ensemble-member quality is flat at every λ,
  so affinity is not acting as a diversity source either.
- **The label's ceiling is below the baseline.** *Measured* affinity — perfect
  knowledge, no model error — used directly as a stability predictor ranks at
  median per-allele ρ **0.580**, under the **0.610** a single network already
  reaches from stability labels alone and far under the ensemble's 0.693.

So the auxiliary signal is **redundant, not absent**. Affinity genuinely predicts
stability; it predicts it through the same groove chemistry the stability labels
already teach. The 64,226-row leakage-filtered expansion is **declined on
evidence, not blocked** — the audit was completed and it is available.
([`stage2c_affinity.md`](stage2c_affinity.md))

### 4.3 The engine comparison was settled by a $1.49 experiment

The structural arm folds a **three-chain, 383-residue construct**: a 275-residue
HLA ectodomain + 99-residue β2-microglobulin (the small partner chain that every
class I HLA needs to be stable) + the 9-residue peptide.

A **matched 90-fold pilot** ran Boltz-2 and ESMFold2 on the same five
training-split complexes × three constructs × three seeds, with identical MSA
content, unpaired policy and inputs verified equal at the feature level. Pose
accuracy is measured by superposing the predicted HLA onto a crystal structure
and *then* measuring the peptide — fitting on the peptide would rotate onto the
answer and hide a mis-docked pose.

| Gate criterion (fixed before the pilot) | Boltz-2 | ESMFold2 |
|---|---|---|
| 45 valid predictions, full PAE and pLDDT, verified register | pass | pass |
| All arm-B peptide CA RMSD ≤ 2.0 Å, anchors ≤ 1.0 Å | pass | pass |
| Sentinel `HLA-B*07:02`/IPRRNVATL median heavy RMSD ≤ 1.0 Å | **pass (0.318)** | **fail (2.564)** |
| Sentinel improves ≥ 0.5 Å over the groove-only arm | **pass (−1.982)** | **fail (−0.008)** |
| **Verdict** | **pass** | **fail** |

The sentinel is the interesting case. The groove-only construct places the
anchors correctly and gets the **middle of the peptide wrong** — per-position CA
deviation runs I1 0.19, P2 0.15 … N5 1.84, **V6 3.39** … L9 0.14. The full
three-chain construct flattens that to a 0.28 Å maximum, reproducibly across all
three seeds. ESMFold2 shows **no construct effect at all**: its sentinel sits at
2.37–2.61 Å in every arm and every seed.

**And the model's own confidence does not catch the failure.** Across the 45
Boltz-2 folds confidence ranks error reasonably (Spearman +0.635 for
peptide–groove PAE, −0.540 for peptide pLDDT) — but the three failing folds sit
at PAE 1.55 and pLDDT 0.973–0.975 while being 2.3 Å wrong, whereas the
*correct* three-chain folds sit at **worse** PAE. No threshold catches the
failure without flagging accurate predictions. Consequence for stage 5: keep
confidence as a feature, **never use it as a per-prediction failure filter**.

ESMFold2 is dropped from production. This is recorded as a **labelled revision
of the agreed matched plan, not a silent drop**: it failed a pre-registered
gate, the cross-model comparison the plan wanted was already delivered by the
pilot on identical inputs, and at 4.55× the cost per fold its full-cohort
forecast of $884 never fitted the budget beside Boltz-2's $194 of fold time
($200.8 including shard startup) in the first place. **What this does not
establish:** pose accuracy is not feature utility,
and a five-complex gate is an operational decision rule, not a general claim
about ESMFold2 on peptide–MHC.
([`stage4c_ectodomain_pilot.md`](stage4c_ectodomain_pilot.md))

### 4.4 A result we did not go looking for

Peptides recovered from real cells by mass spectrometry (the MHC Motif Atlas)
overlap our dataset by only **140 of 151,170** 9-mers — the peptide universes are
near-disjoint, which makes the atlas a genuine external check rather than
circular validation. On those 140 pairs, eluted ligands have **~7× higher median
half-life** (7.90 h vs 1.10 h) and the fraction with zero measured stability
drops **tenfold** (2.1% vs 21.2%). Mann–Whitney p = 6.0 × 10⁻³¹;
common-language effect size 0.78 — a random eluted ligand outlasts a random
non-eluted peptide 78% of the time. The effect is not carried by one well-studied
allele: among 18 alleles with ≥ 3 measured eluted ligands the eluted median
exceeds the non-eluted median in 14.

So a peptide that survived hours of cell lysis and acid elution really does carry
a hidden lower bound on stability. **We deliberately did not use this as training
data** — 151,170 atlas peptides against 28,166 measurements would make
antigen presentation the main task and stability a side effect, the
"t½ > τ" threshold is unknowable, and elution is confounded with protein
abundance, proteasomal cleavage, TAP transport and ionisation efficiency. It is
reserved as an **external validation pass**: score atlas ligands against
allele-matched decoys with no retraining.
([`elution_stability_finding.md`](../elution_stability_finding.md))

**That pass has now run for the sequence baseline** and its artifacts are on
disk — `reports/stage3c_summary.csv`, three per-allele tables (headline plus a
swapped-decoy and a wrong-allele control) and `stage3c_provenance.json`, which
records 51 alleles, 81,600 ligands against a 10× proteome decoy pool, a
`verification` block confirming the scored predictions match
`preds/seq_baseline.csv`, and 48.1 s of CPU. **A narrative report does not exist
yet, so the result is a hole here rather than a claim: ‹HOLE S3C›.** What fills
it: per-allele AUROC/AUPRC with the two controls interpreted, and the same pass
run on whichever arms stage 6 scores — a transfer result for the sequence
baseline alone does not speak to the foundation-model question.

---

## 5. Results

All figures below are **validation**. The test set is scored **once**, at stage
6, with every model together.

### 5.1 The table

| Arm | Information added | Median per-allele ρ | Paired Δ vs its control [95% CI] | Verdict | $ / 1,000 preds |
|---|---|---:|---|---|---:|
| Training allele mean | none | 0.000 | — | floor | — |
| MLP, peptide only | peptide identity | 0.202 | — | — | $2.6 × 10⁻⁹ |
| Ridge, pep + pseudoseq | linear only | 0.278 | — | — | not measured |
| MLP, pep + pseudoseq (single) | + 34 contact residues | 0.610 | −0.083 [−0.124, −0.029] | worse than its own ensemble | $6.6 × 10⁻⁹ |
| 30-net ensemble, pep + domain | + 182 domain residues | 0.653 | −0.040 [−0.073, +0.001] | tied | $7.9 × 10⁻⁷ |
| **30-net ensemble, pep + pseudoseq** | **the baseline to beat** | **0.693** | — | — | **$2.0 × 10⁻⁷** |
| Weak-binder augmentation (best arm) | 2,910–4,407 assumed-zero rows | 0.618 | +0.024 [−0.026, +0.048] | **inconclusive, rules out 0.05** | $2.0 × 10⁻⁷ |
| Auxiliary affinity head (best λ) | 5,135 affinity labels | 0.701 | +0.008 [−0.017, +0.030] | **inconclusive, rules out 0.05** | $2.0 × 10⁻⁷ |
| **ESM-2, frozen representations** | **pretrained sequence embeddings** | **‹HOLE E2›** | **‹HOLE E2›** | **‹HOLE E2›** | **$4.4 × 10⁻⁴** + ‹HOLE E1b› |
| **Boltz-2, structural features** | **predicted 3D complex + confidence** | **‹HOLE B2›** | **‹HOLE B2›** | **‹HOLE B2›** | **$6.90** |

**Notes on reading this table**, because two rows mix aggregations and saying so
is cheaper than a reader discovering it:

- **The control differs by row.** Rows 4–6 are paired against the 30-network
  pseudosequence ensemble (0.693). The auxiliary-affinity row is too — at λ = 0
  that model is *bit-identical* to it, verified on all 2,817 validation rows.
  The augmentation row is paired against its own measured-only control under the
  single-network protocol, which is a different comparator. Exact pairings in
  [`stage2b_augmentation.md`](stage2b_augmentation.md) and
  [`stage2c_affinity.md`](stage2c_affinity.md). **The two in-flight rows must be
  paired against the 30-network ensemble**, per §3.1.
- **The augmentation row's 0.618 and its +0.024 are not the same aggregation.**
  0.618 is the mean of three per-seed scores; +0.024 is measured on the
  3-seed-mean *prediction* (a 3-network ensemble, which scores 0.649 for the
  control). The report states this explicitly; the two columns are not
  additive. The verdict is unaffected — every one of the eight intervals crosses
  zero and excludes 0.05.
- The single network's −0.083 is the ensembling gain read backwards, included so
  the choice of comparator is visible rather than implicit.
- Accuracy figures in rows 1–6 are **means over 3 seeds** from
  [`stage2_baselines.md`](stage2_baselines.md). The figures below are drawn from
  the committed seed-0 prediction files, so they differ by up to the seed spread
  (0.010–0.051).

### 5.2 Figures

Regenerate both with `.venv/bin/python reports/figures/make_figures.py`
(CPU, no network). They redraw unchanged once the in-flight prediction files
land in `preds/`.

**[`figures/fig1_per_allele_spearman.png`](figures/fig1_per_allele_spearman.png)
— the distribution behind the median.** One point per allele, 68 eligible
validation alleles. The headline number is a median over a distribution with a
long left tail: the ensemble's interquartile range is [0.574, 0.753], three
alleles sit below 0.20, and one still ranks backwards (`HLA-A*24:19`,
ρ = −0.04). A single scalar hides that; this panel does not.

**[`figures/fig2_cost_vs_accuracy.png`](figures/fig2_cost_vs_accuracy.png) —
the actual question.** Accuracy against dollars per 1,000 new predictions, log
x-axis spanning nine orders of magnitude between the cheapest and dearest arm. The dash-dot horizontal line is the
best sequence arm plus the predeclared 0.05 bar: **anything that costs more has
to land above that line to have earned it.** ESM-2 and Boltz-2 currently appear
as vertical lines — their costs are measured (ESM-2's extraction half; Boltz-2's
whole fold), their accuracies are not yet known. **That is an honest picture of
where this submission stands**, and it is the figure that answers the brief's
question directly: the two foundation-model arms sit roughly 3 and 7 orders of
magnitude to the right of the baseline, and whether either lands above the
dash-dot line is the entire result.

### 5.3 The open holes, precisely

| Hole | What it is | Owner | What fills it |
|---|---|---|---|
| **E1** | ESM-2 cost per 1,000 new predictions | `esm-arm` | Embedding extraction time and hardware, cache size per unique sequence, head inference seconds per 1,000 rows, ensemble member count |
| **E2** | ESM-2 validation accuracy and its paired CI | `esm-arm` | Median per-allele ρ under `cv_folds()` with matched ensemble size, and the paired cluster-bootstrap Δ against `preds/seq_ensemble_pep_pseudo.csv`. Must also be compared against the **full-domain** ensemble (0.653) if the HLA input is the domain |
| **B2** | Boltz-2 structural accuracy and its paired CI | stage 4c production → stage 5 | Same, on identical train/val/test rows with matched head architecture, ensemble size and tuning budget. Plus coverage and the declared sequence fallback for structural failures |
| **B3** | Realised production spend | Modal production session | Metered before/after snapshots per workspace, realised GPU-hours and wall clock, failure count, actual concurrency granted. `production_<profile>.jsonl` exist for both profiles and are currently **empty** |
| **B4** | Structural coverage and failure rate (§6.5) | stage 5 | Pairs with a valid structure, pairs falling back to the sequence model, and the primary result reported on the **frozen cohort**, not on whatever folded |
| **E1b** | ESM-2 **head** inference cost | `esm-arm` | The extraction half is already measured (`stage3_embedding_cost.csv`: 1.07–8.50 s per 1,000 new peptides across three checkpoints). Still needed: head inference seconds per 1,000 rows, member count, and which checkpoint/layer/representation was selected |
| **S3C** | Elution external-validation write-up | stage 3c | Artifacts exist (`stage3c_summary.csv` + three per-allele tables + provenance). Needed: the narrative interpretation of the two controls, and the same pass run on the arms stage 6 scores |
| **S7a** | Censored (Tobit) likelihood result | stage 7a | `stage7_censored.md` §1–5 are predeclared and committed; §6 is "pending". Needed: the paired Δ against the identical-protocol MSE ensemble, plus the `c_hours` sensitivity sweep |
| **S6** | Stage 6 test results, distance strata, differential target, nested near-neighbour CV | `eval-harness` | §6 below |

---

## 6. Evaluation beyond a single number

A median per-allele Spearman is one number on one panel. Five further
evaluations were specified in the frozen contract, each answering a question the
headline cannot. **They run at stage 6 and cost no new compute** — all are
re-aggregations of predictions already made.

### 6.1 Distance stratification — is the model generalising or remembering?

Every held-out peptide is 4 or 5 substitutions from its nearest training
peptide, and label similarity at d=4 is still Spearman 0.592. So test metrics are
reported in two strata: **d=4 (3,256 rows, 57.8%)** and **d≥5 (2,377 rows,
42.2%)**, scored on the **same 65 alleles** (the intersection of those eligible
within each stratum) — otherwise the comparison measures allele panels rather
than distance. A d≥6 stratum is not viable: 12 test rows reach it.

**If the model is exploiting residual similarity at the split boundary, d=4 will
score visibly higher than d≥5. If it does not, that is a strong positive signal
that the model genuinely generalises.** Result: **‹HOLE S6a›**.

### 6.2 The differential target — groove chemistry, isolated

For peptides measured on two or more alleles, evaluate **Δ log half-life between
alleles**. This subtracts out whatever is intrinsic to the peptide and tests
groove chemistry directly — the sharpest available version of "distinguish
within-allele ranking from cross-allele effects". It is abundant: **3,941
peptides sit on ≥ 2 alleles, covering 26,474 rows (94% of the dataset), up to 36
alleles for a single peptide.** Result: **‹HOLE S6b›**.

### 6.3 Nested near-neighbour evaluation — the question the split cannot ask

Because no two peptides within 3 substitutions straddle a split, this benchmark
**cannot assess mutant ranking** — scoring point mutants of a known binder, which
is often the practically relevant design question. The partial recovery is a
cross-validation on the d≤2 peptide clusters that live entirely inside the
training split, touching neither the test set nor the frozen assignments.
Result: **‹HOLE S6c›**.

### 6.4 Failure analysis, not just aggregate scores

On validation, the single-network headline arm ranks 1 of 68 alleles
**backwards** (`HLA-A*24:19`, ρ = −0.12) and 4 below 0.20; the 30-network
ensemble improves both but does not fix either (`HLA-A*24:19` still at −0.04,
3 alleles below 0.20). The full-domain arm fails on nearly the same set, which
suggests these are hard alleles rather than an artifact of one input
representation. **A model that is right on average and catastrophically wrong on
a specific allele is not safe to deploy on a patient with that allele**, and
reporting only a median would hide it. The per-allele table is published for
every arm. Test-set failure set: **‹HOLE S6d›**.

### 6.5 Coverage and failure, for the structural arm specifically

Structure prediction can fail outright. Stage 5 reports the **frozen cohort with
a declared sequence-model fallback** as the primary result, and common successful
rows only as a diagnostic. **Scoring only the pairs that happened to fold would
be a retrospectively chosen cohort**, which is exactly the kind of quiet
selection a watertight evaluation has to rule out. Result: **‹HOLE B4›**.

---

## 7. Cost

Full detail in [`compute_ledger.md`](compute_ledger.md).

### 7.1 What has been spent

**$4.41 of metered GPU, total, for the whole project to date.** That covers a
five-GPU hardware benchmark, two engine pilots and a 90-fold matched model
comparison. Everything on the sequence side — stages 1, 2, 2b, 2c, 3c, and all
MSA preparation for both constructs — ran on **one laptop core for $0**, about
36 CPU-minutes of fitting and bootstrapping in total.

Production folding of all 28,166 pairs is forecast at **$200.8 (135.6 A10G
hours), $251 with the required 25% margin**, split across two Modal workspaces
in parallel at ~6.8 hours wall clock. **It has not been launched.**

### 7.2 Cost per 1,000 new predictions — the decision number

| Arm | $ / 1,000 | GPU-h / 1,000 | Basis |
|---|---:|---:|---|
| Sequence ensemble, 30 networks | $2.0 × 10⁻⁷ | 0 | 30 × measured 0.0005 s per 1,000 rows, at Modal's published $0.04730/CPU-core-hour |
| **ESM-2, extraction only** | **$4.4 × 10⁻⁴ (35M) – $3.5 × 10⁻³ (650M)** | 0 (laptop `mps`) | measured 1.07–8.50 s per 1,000 new peptides, priced at the A10G rate as a generous upper bound |
| ESM-2, head forward pass | ‹HOLE E1b› | ‹HOLE E1b› | pending |
| **Boltz-2 structural** | **$6.90** | **4.66** | measured 16.76 s fold × measured $1.4812/h A10G |
| *ESMFold2 (rejected)* | *$31.40* | *11.8* | *measured 42.63 s × $2.6512/h L40S; forecast only* |

**The structural arm costs about 3.5 × 10⁷ times more per prediction than the
sequence ensemble it has to beat.** In wall clock rather than dollars: scoring
the entire dataset takes the sequence ensemble **under half a second of one CPU
core**; Boltz-2 takes **131 GPU-hours**.

Two honest caveats. The dollar ratio compares a laptop CPU figure priced at a
cloud CPU rate against a cloud GPU bill — dollars are the only axis that puts
them side by side at all, and they are not the same kind of dollar. And
structures are **reusable**: $6.90 is paid once per *pair*, not once per
experiment, so a fixed cohort studied repeatedly amortises it. For the actual
use case — scoring new candidate peptides — it does not amortise, because a new
peptide needs a new fold. Neither caveat moves the order of magnitude.

### 7.3 Measuring cost changed what we ran

Cost measurement here is not bookkeeping; it is a results-producing activity.

- **A benchmark harness overstated folding cost by 4.6×.** It spawned a fresh
  subprocess per complex, so every "fold time" contained a Python start, a 6.2 GB
  weight load and CUDA init — ~36 s, 86% of each measurement. The shape was
  copied from Modal's own published Boltz example, which runs **one** input per
  call, where the weight load is invisible because there is nothing to amortise
  it over. **A benchmark that measures the wrong thing is worse than no
  benchmark, because it gets quoted with confidence.**
- **H100 does not pay for itself.** It needed to be 3.01× faster than A10 to
  break even on cost per complex; it measured 1.35×. A 191-residue complex at one
  diffusion sample cannot saturate an H100 — the runtime is sequential diffusion
  and recycling, not throughput the extra silicon absorbs.
- **Between-container variance is 73%**, against ±1% within a container. With
  n=1 container per GPU the ordering among middle-ranked cards is not resolvable,
  and we say so rather than quoting it.
- **The plan's MSA-trimming concern was overturned by measurement.** Parse cost
  across 2,000 complexes is ~$0.25 — under 0.2% of the fold bill. Do not trim for
  cost; trim only if accuracy says to.
- **A $1.49 pilot settled the engine choice on measurement rather than
  preference**, and ruled out an $884 alternative before any production compute
  was booked. Cost was only the third of three reasons — the pre-registered
  quality gate came first — but the pilot cost 0.17% of the run it averted.

---

## 8. Limitations

Full register in [`limitations.md`](limitations.md). The six that would change a
reader's interpretation most:

1. **One fifth of the labels are not measurements.** 5,679 rows (20.2%) sit at
   an assay detection floor, on three independent lines of evidence. `log1p`
   makes them trainable and compresses a 5.86-skew target to 0.91; it does **not**
   make them measurements. MAE at the floor is error against a *recorded* label
   with uncertain sign — neither an upper nor a lower bound on true error. A
   censored (Tobit) likelihood is the principled fix; it was recorded as out of
   scope and is now in flight as stage 7a, with its protocol predeclared and
   its results pending (hole **S7a**).
2. **The peptide panels are allele-confounded by design.** Each allele was
   assayed on its own panel; the grid is 6.7% full. This broke the first split,
   and it is the reason leave-allele-out evaluation is not the headline —
   holding out an allele also holds out its peptide panel, so an allele-distance
   effect is inseparable from a panel effect. **That confound does not disappear
   by being measured.**
3. **The assay panel was pre-selected by predicted affinity**, so peptide
   diversity is narrow by construction. The benchmark measures ranking *within
   the set of peptides that bind*. Broader biological or clinical claims need
   evidence from outside this file.
4. **Our negative results are "ruled out at 0.05", never "zero".** Stage 2b's
   best arm is +0.024 [−0.026, +0.048]: the data are consistent with a real
   +0.04. What is excluded is +0.05. A larger benchmark would move that bar down
   and some of what we call ruled out would become detectable.
5. **If ESM-2 does not help, that will be bounded to the representation actually
   tested** — peptide and HLA embedded *separately*, so the head must learn the
   interaction itself from two unconditioned vectors. It would not be a result
   about foundation models in general. The sharpest untested version is a
   chimeric peptide-linker-groove input, listed as out of scope **specifically
   so that a null is reported as bounded**.
6. **ESMFold2's rejection is operational, not scientific.** Pose accuracy is not
   feature utility, and a five-complex gate is a decision rule for spending $200,
   not a benchmark of ESMFold2 on peptide–MHC.

---

## 9. What we would do next

Ordered by expected value per hour, not by appeal.

1. **Finish the in-flight arms and score the test set once** (holes E1b, E2,
   B2–B4, S3C, S6, S7a). Everything else is downstream of knowing whether the
   expensive arms clear 0.05.
2. **Finish the censored (Tobit) likelihood — now in flight as stage 7a.** This
   is the highest-value single change to the modelling. 20.2% of labels are a
   detection limit treated as exact zeros, and the same problem independently
   wrecked the stage 2b affinity predictor's recall (0.263 at 0.823 precision,
   because 26.8% of its labels sit exactly on the threshold). One fix addresses
   both. [`stage7_censored.md`](stage7_censored.md) predeclares the loss, the
   detection threshold `c = log1p(0.1)`, a five-point sensitivity sweep and the
   prediction heads **before any censored model was fitted** — and predeclares
   the expectation that it should improve calibration near the floor rather than
   ranking, since order inside the tied floor block is unidentifiable under
   either objective. Hole **S7a**.
3. **Test the chimeric ESM-2 input.** If the separate-embedding arm is flat, the
   obvious objection is that we never let the language model see the interaction.
   A peptide-linker-groove construct is the sharpest version of the question and
   the biggest risk to a negative result — which is precisely why it is worth
   running rather than avoiding.
4. **Re-run the auxiliary-affinity probe on the ESM-2 arm.** The machinery is
   protocol-agnostic and the leakage audit is already done (64,226 admissible
   rows). Affinity is redundant with what a *sequence* model extracts; that says
   nothing about ESM-2 features. "Cheap labels substitute for expensive
   pretraining" would be a genuinely useful finding.
5. **Run the allele-held-out evaluation, with its confound reported alongside.**
   This is the stratum where pretraining has the strongest prior of winning —
   pan-allele generalisation is exactly what large-scale pretraining should
   provide — so a gain confined to distant alleles would be real and reportable
   even if the pooled comparison came out flat.
6. **ProteinMPNN on the stored structures.** Inverse folding asks "given this
   backbone, how probable is this peptide sequence?" — a different question from
   geometry or confidence, needing no refolding, and the 22 GB of structures will
   already exist. `pepstab/inverse_folding.py` and `scripts/proteinmpnn_score.py`
   exist; the structures do not yet.
7. **Find post-2016 stability measurements.** The only honest route to a
   comparison against NetMHCstabpan is data it could not have trained on. Until
   then, no method-parity claim is available from this dataset at any stage.

---

## Appendix: how to reproduce

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy pandas scipy scikit-learn pyarrow pytest matplotlib

.venv/bin/python scripts/audit_data.py            # stage 1 audit
.venv/bin/python scripts/baseline_sequence.py     # stage 2 grid, ~10 CPU-min
.venv/bin/python scripts/baseline_ensemble.py     # the 30-network comparator
.venv/bin/python scripts/compare_to_paper.py      # NetMHCstabpan calibration
.venv/bin/python scripts/baseline_augmented.py    # stage 2b, ~13 min
.venv/bin/python scripts/affinity_multitask.py --protocol ensemble   # stage 2c
.venv/bin/python reports/figures/make_figures.py  # figures
.venv/bin/python -m pytest tests/ -q              # 327 collected contract guards
                                                  # (some skip when optional
                                                  #  structural inputs are absent)

# score any prediction file against the frozen splits
.venv/bin/python scripts/evaluate.py --split val \
    preds/seq_ensemble_pep_pseudo.csv preds/esm_ensemble.csv --per-allele
```

`data/splits.csv` is loaded, never recomputed — regenerating it drops the
peptide-cluster grouping and leaks training data into the test set.
`data/rasmussen_et_al_dataset.csv` is read-only and checksummed.
