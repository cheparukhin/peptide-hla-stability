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
>
> **Every number in this document is a validation number.** The frozen test set
> has not been scored. It is scored **once**, at stage 6, under
> [`TEST_SCORING_RUNBOOK.md`](TEST_SCORING_RUNBOOK.md), with every model
> together — so nothing here may be read as a held-out test result, including
> the comparisons that return conclusive verdicts.

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
| **All three model classes the brief names** | protein language models **§4.0 (ESM-2)**, structure prediction §4.3 (Boltz-2), inverse folding §4.5 (ProteinMPNN) |

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

| Source | Model class | What it adds | Price per 1,000 new predictions |
|---|---|---|---:|
| Labelled sequences (the baseline) | — | nothing; this is the reference point | $9.6 × 10⁻⁷ |
| **ESM-2** | protein language model | learned representations of peptide and HLA | **$1.5 × 10⁻⁵ – $4.7 × 10⁻⁴** end to end (measured) |
| **Boltz-2** | structure prediction | a predicted 3D complex, plus the model's own confidence | **$6.90** |
| **ProteinMPNN** | inverse folding | sequence–backbone compatibility (§4.5) | $0.52 / 1,000, QC sample only |

That covers **all three model classes the brief names**. The spread in the
right-hand column is roughly **seven orders of magnitude**.
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
  scores its base rate rather than being flattered by row order. **In practice
  it turned out to be too coarse to separate close arms — it moves in steps of
  0.1 — and we report that failure rather than dropping it. See §6.7.**
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

The headline arm first, then three cheap hypotheses tested before any GPU was
booked, then the structural engine comparison. Every one came back negative, and
**the intervals are tight enough to say what kind of negative**.

### 4.0 ESM-2: parity standing alone, and nothing on top

This is the submission's central question, and it is now answered for the
protein-language-model class ([`stage3_esm.md`](stage3_esm.md)). Validation,
2,817 rows, 68 eligible alleles, **30 ensemble members in every arm**, paired
cluster bootstrap at 2,000 resamples.

| Arm | median per-allele ρ | Δ vs baseline | 95% CI |
|---|---:|---:|---|
| **sequence baseline** (30-net, pep + pseudoseq) | **0.6931** | — | — |
| ESM-2 only | 0.6830 | −0.0101 | [−0.0382, +0.0352] |
| sequence + ESM-2 (additive) | 0.6761 | −0.0170 | [−0.0468, +0.0320] |

**Both are inconclusive at zero, and both rule out a +0.05 gain.** That phrasing
is exact and the distinction matters: **this is not a demonstration that ESM-2
is worse.** Both intervals contain zero, so we cannot say the pretrained
representation hurts. What we *can* say is that **every upper bound sits below
the predeclared worthwhile gain** — which is the strongest negative this
evaluation supports, and the one the brief asks for.

The two arms answer different questions and agree:

- **ESM-2 alone reaches parity with the baseline** — from a representation that
  has never seen a stability label, against a model trained on 19,716 of them.
  That is a genuinely interesting positive about what pretraining encodes, and
  it is not a win on this benchmark.
- **ESM-2 adds nothing on top.** Given features the baseline already extracts
  from those labels, the pretrained representation contributes no further signal
  worth the predeclared bar.

#### The sharpest version: a measured equivalence, not a wide shrug

"Inconclusive at zero" is a weak claim when the interval is wide, and on the
primary metric it is: ±0.035 on a quantity whose whole interesting range is
about 0.1. **The differential target makes the same comparison far more
sharply** — and it is the metric this project built specifically to separate
within-allele ranking from cross-allele effects.

For a peptide measured on two or more alleles, evaluate **Δ log half-life
between the alleles**. The peptide's own contribution cancels *exactly*, so what
remains is groove chemistry. 394 validation peptides sit on two or more eligible
alleles, giving **10,365 allele-pair comparisons**; 500 cluster resamples,
paired against the sequence baseline.

| Arm | Δ concordance | 95% CI | Verdict |
|---|---:|---|---|
| ESM-2 35M | **+0.0028** | [−0.0052, +0.0102] | inconclusive at 0 |
| sequence + ESM-2 | **−0.0013** | [−0.0091, +0.0058] | inconclusive at 0 |
| ESM-2 150M | −0.0088 | [−0.0149, −0.0025] | **conclusively worse** |
| full-domain ensemble | −0.0124 | [−0.0186, −0.0061] | **conclusively worse** |

> **On cross-allele ranking — where the peptide's own contribution cancels
> exactly — ESM-2 and the sequence baseline are equivalent to within one point
> of concordance: +0.003 [−0.005, +0.010] for ESM-2 alone and −0.001 [−0.009,
> +0.006] for sequence plus ESM-2. The same measurement conclusively separates
> the 150M checkpoint (−0.009 [−0.015, −0.002]) and the full-domain ensemble
> (−0.012 [−0.019, −0.006]) from the baseline, so the equivalence is a measured
> null and not a metric that cannot tell arms apart.**

That last clause is the whole point. A tight interval around zero invites
exactly one objection — *the metric is degenerate, it cannot resolve anything* —
and **two conclusive separations, on the same 10,365 comparisons under the same
bootstrap, answer it directly.** The differential resolves differences of about
0.009; both ESM arms sit well inside that. Its intervals are four to six times
tighter than the primary metric's on the same rows.

So the headline upgrades from *"inconclusive at zero, rules out +0.05"* to
**"equivalent, to within one point of concordance"** — a far more useful claim
for anyone deciding whether to deploy a language model here, because it says
what the answer *is* rather than only what it is not.

**The mean per-allele Spearman agrees**, which matters because it is an
independent statistic on the same panel:

| Arm | Δ mean per-allele ρ | 95% CI |
|---|---:|---|
| ESM-2 35M | −0.0036 | [−0.0248, +0.0165] |
| sequence + ESM-2 | −0.0112 | [−0.0332, +0.0111] |
| ESM-2 150M | **−0.0199** | **[−0.0392, −0.0017]** |
| full-domain ensemble | **−0.0348** | **[−0.0502, −0.0197]** |

The mean separates **the same two arms** the differential does, while the
median — the contract's metric, to which the predeclared verdict rule applies
alone — calls both inconclusive. Two independent statistics agreeing is more
robust than either on its own, and two panel statistics disagreeing about
*conclusiveness* is itself worth showing rather than hiding.

**This is the differential target earning its place.** It was specified in the
frozen contract to separate within-allele ranking from cross-allele effects, it
costs no new compute — it re-aggregates predictions already made — and it is
**the only metric in the project that resolved what the primary metric could
not.**

#### Three controls, which are what stop this being an artifact

A negative result about a foundation model is only worth reporting if the
obvious ways of manufacturing one have been closed off. Three were, and the
first is the one that nearly produced a confidently wrong answer.

**1. Tuning parity means equal *budget*, not equal *values* — and getting this
wrong costs 0.109 SCC.** The tempting move is to hold the baseline's
hyperparameters fixed "for fairness" and vary only the features. Stage 2's L2
ladder was `{1e-5, 1e-3}`, chosen for sparse one-hot features over 860
dimensions; ESM features are dense standardised components. Transplanted, that
ladder is a handicap wearing the costume of fairness:

| Arm | ρ at stage 2's L2 | seed spread | ρ at its own boundary-checked L2 | seed spread | cost of the transplant |
|---|---:|---:|---:|---:|---:|
| ESM-2, middle layer | 0.4994 | **0.110** | **0.6081** | 0.013 | **−0.109** |
| baseline, one-hot | 0.6096 | 0.032 | 0.6116 | 0.044 | −0.002 |

**The ESM arm is roughly 50× more sensitive to the regularisation range than the
baseline.** Reported at stage 2's setting, frozen ESM-2 would have come in 0.11
behind and we would have shipped a confident negative that was purely an
artifact of the ladder. The tell arrived before any comparison was made: at
L2=1e-5 the ESM arm's three seeds ranged over **0.110**, far outside stage 2's
observed 0.010–0.051. *An effect smaller than seed noise is not an effect; an
instability larger than seed noise is usually under-regularisation* — and here
it was. The check was applied **symmetrically**: every arm's ladder was extended
until its selection was interior, the baseline included, because fixing one
handicap by installing its mirror image is not a fix.

**2. The compression we adopted for memory reasons helped, by +0.033.** PCA on
the embeddings was a resource decision, not a modelling one — so it is exactly
the kind of choice that could be blamed for the negative. Measured, it goes the
other way: the compressed arm *outscores* the uncompressed control. The negative
cannot be attributed to it.

**3. The scaling curve is flat, which is the strongest available answer to "you
just needed a bigger model".** ESM-2 150M lands at **0.6737** — marginally
*below* 35M, at 2.6× the extraction cost. At 35M there is no gap for a larger
checkpoint to close. That weakens the scaling argument without closing it; see
§8 for what it does not establish.

#### The one comparison that is not a null — do not skip it

Against the **full-domain** sequence ensemble (0.6529), on matching input,
ESM-2 scores **+0.0301 [−0.0084, +0.0757]**. Still inconclusive — the interval
crosses zero — but it is **the only interval in this stage whose upper bound
exceeds the 0.05 bar**, so unlike every other comparison here it does not rule
a worthwhile gain out.

Read plainly: **a pretrained representation of the HLA domain beats one-hot
encoding the same domain.** That is a real and unsurprising thing for a protein
language model to do, and it is the strongest result ESM-2 produces here.

It does **not** overturn the headline, and the reason is specific: the
baseline's best configuration **does not use the full domain**. It uses the
34-residue contact pseudosequence — a piece of domain knowledge about which
positions touch the peptide — and that hand-built feature is what ESM-2 has to
beat, not the naive full-domain encoding. **Both halves of that sentence are
load-bearing** and neither should be quoted without the other.

*One further result from this stage is a transferable lesson about **using**
foundation models rather than evidence for or against the verdict above, so it
has its own section: **§9**, on why where you read an embedding from matters
more than which model produced it.*

#### Verification, and one measured engineering constraint

The comparison machinery is **verified, not assumed**: the stage 2 ensemble was
rebuilt from scratch through the new harness and the regenerated file is
**byte-identical to stage 2's published artifact** (same md5), giving
Δ = +0.0000. Stage 2's prediction files were never regenerated in place — the
boundary-extension runs wrote to new stems — so every downstream fingerprint
stays valid.

**ESM-2 650M is cached but excluded**: its 5.06 GB peak RSS would not fit
alongside the concurrent production fold on a 16 GB machine. That is a
**measured engineering constraint**, recorded as one rather than dressed up as a
modelling choice — and engineering and compute requirements within the timeframe
are something the brief explicitly grades.

#### Cost

Head inference is **0.0150 s per 1,000 rows** for the 30-network ESM ensemble,
against **0.0656 s** for the sequence ensemble — the ESM head is *cheaper* at
inference, because 330 PCA components is a narrower input than 860 one-hot
columns. End to end, including embedding 1,000 genuinely new peptides, it is
**1.14 s against the baseline's 0.073 s: about 16×, and both negligible in
absolute terms.** The selected configuration is 35M, middle layer, peptide
per-position (PCA 256), HLA 34-contact (PCA 74, lossless), MLP (256, 64),
L2 = 1e-2. **Total cloud spend for stage 3: $0.**

A 16× cost ratio for parity is a very different proposition from the structural
arm's seven orders of magnitude (§7.2). **ESM-2's problem on this task is not
that it is expensive. It is that the thing it would have to beat is a 34-residue
hand-built feature that already works.**

---

### The three cheap hypotheses

All three came back negative, **and the intervals are tight enough to say what
kind of negative**.

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

All numbers in this section are **validation**. **7,281 pairs across 58
allotypes** carry both an affinity measurement (how
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

**This null survives the stage 3 ladder correction**, which matters because §4.0
showed that a transplanted regularisation range can manufacture a 0.109 swing.
Re-run on the corrected ladder, λ=0 gives 0.6876 and λ = 0.1 / 0.3 / 1 / 3 give
−0.0031 / −0.0043 / −0.0054 / −0.0174 — every CI crossing zero, every upper
bound below 0.05. The measured minimum detectable effect is bracketed in
**(0.018, 0.037]**, so this design *can* resolve the 0.05 bar: the verdicts are
earned, not a formality of an underpowered test.

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

**That pass has now run.** 51 alleles, 81,600 eluted ligands against 816,000
length- and allele-matched human-proteome decoys at a 10:1 ratio fixed in
advance. No retraining: the atlas enters as a scoring target and nothing else.
([`stage3c_elution_validation.md`](stage3c_elution_validation.md))

| Arm | AUROC median [IQR] | AUPRC median | Precision top 1% | Enrichment top 1% |
|---|---|---:|---:|---:|
| **`seq_ensemble` (30 nets)** | **0.9656** [0.9435, 0.9779] | **0.7804** | 0.939 | **10.33×** (of a possible 11.0) |
| `seq_baseline` (single net) | 0.9502 [0.9182, 0.9648] | 0.6676 | 0.877 | 9.65× |
| *random scores, same harness* | *0.4974* | *0.0911* | *0.089* | *0.98×* |

Chance AUPRC is 1/11 = 0.0909 by construction of the ratio. **The random-score
null returns chance on every metric**, which is what establishes that the number
comes from the model and not from the scoring code. And the arm ordering
reproduces the validation ordering (0.693 vs 0.610) — independent corroboration
of the ensembling effect, on a different assay.

#### The three controls, which are what the number means

A headline AUROC against proteome decoys is the least informative part of this
result, because a random proteome 9-mer is a negative on *every* axis at once —
not cleaved, not transported, not presented, not ionised, **and** not stable.
The controls are what separate those.

**1. Allele-swapped decoys — 0.9157 (−0.050).** Replace each allele's decoys
with *other alleles' real eluted ligands*, same ratio, same seed, never
including a peptide the target allele itself presents. The negatives are now
genuinely presented peptides, so abundance, cleavage, transport and ionisation
are held roughly constant on both sides and what remains is allele specificity.
**The 0.050 of AUROC the swap costs is the share of the primary result
attributable to generic presentability rather than allele-specific
discrimination.** That is a bound on this decoy design, not a decomposition of
the biology. AUPRC falls further, 0.780 → 0.557, because precision is more
sensitive to hard negatives than a rank statistic is.

**2. The donor-distance gradient — 0.8906 near vs 0.9562 far.** Each swapped
decoy carries the pseudosequence Hamming distance from the target allele to the
nearest allele known to present it. Split at the pooled median (14 of 34
positions): a ligand of a groove that *resembles* the target is a much harder
negative than a ligand of a distant groove. The model is reading groove
chemistry, not just peptide chemistry, and discrimination degrades as the
grooves converge without collapsing to chance. **Only two bins were measured, so
the direction and size of that decline are established and its shape is not.**

**3. Wrong-allele pseudosequence — 0.6965, and top-1% enrichment collapses to
exactly 1.00×.** Score every row against a different allele's groove under a
fixed-point-free permutation; the peptides are unchanged. Handing the model the
wrong groove costs **0.27 AUROC and all of the top-of-list precision**. So it is
not simply flagging "peptide-shaped sequences". The ~0.70 that survives is the
generic presentability signal control 1 already identified, plus whatever any
two class I grooves share.

#### What this is not

- **Not a measurement of stability-prediction accuracy.** Elution has at least
  four filters besides stability — source-protein abundance, proteasomal
  cleavage, TAP transport, mass-spec ionisation. A peptide in the atlas passed
  all of them, and nothing here separates stability's contribution from the
  others. Control 1 *bounds* it at 0.050; it does not decompose it.
- **Not a ranking of stability within presented peptides.** This is ligand
  versus non-ligand, a binary discrimination. The per-allele Spearman against
  measured half-life asks a harder question, and this pass says nothing about
  it. **0.610 / 0.693 remain the numbers to quote for accuracy.**
- **Not comparable to a trained presentation predictor.** NetMHCpan-4.x and
  friends train on elution data; this model has never seen any. The right
  reading of 0.966 is "a stability model transfers", not "a stability model
  competes with a presentation model". We did not run that comparison.
- **The positive class carries roughly a 2% error rate** (§7.1 of
  [`limitations.md`](limitations.md)), putting a soft ceiling just under 1.0 on
  any AUROC here.
- **Decoys are assumed negatives.** Mass spectrometry is positives-only, so a
  proteome 9-mer absent from the atlas may simply never have been sampled. That
  contaminates the negative class slightly, which **depresses** the reported
  AUROC rather than inflating it — in that one respect these numbers are
  conservative.

**The honest summary:** elution is not a stability assay and this is not a
validation of stability prediction. It is evidence that what the model learned
from half-lives is real biophysics about the peptide–groove interaction rather
than an artifact of the assay panel or the splits — because that knowledge
transfers, allele-specifically, to a measurement nobody trained it on.

Still open: the same pass on the ESM-2 and structural arms. The harness takes an
arbitrary score file (`--emit-scoring-set` then `--scores`), so it is cheap, and
a transfer result for the sequence arm alone does not speak to the
foundation-model question.

### 4.5 Inverse folding — the third model class, and a reframe

The brief names **three classes** of protein foundation model: structure
prediction, protein language models, and **inverse folding**. This project tests
all three — Boltz-2 (stage 4c), ESM-2 (stage 3), and ProteinMPNN here. Inverse
folding was originally out of scope because it "requires structures first"; the
stage 4c pilot left 90 folds on disk, which unblocked it.

Inverse folding runs structure prediction backwards: *given this 3D backbone,
how probable is this amino-acid sequence?* We mask only the 9-residue peptide,
leave the HLA and β2m visible, and read
`log P(peptide | backbone, HLA sequence)` averaged over 16 decoding orders.
([`stage5_inverse_folding.md`](stage5_inverse_folding.md))

**The finding is a reframe, not a feature.** ProteinMPNN's peptide
log-likelihood is **not** a half-life predictor — Spearman(`pep_ll_total`, t½) =
**−0.100, p = 0.87, n = 5**, which is zero and carries no inferential weight
whatsoever. But it is an **unsupervised, crystal-free detector of Boltz-2
peptide-pose failure**, which is something the project otherwise lacks entirely.

Three things travel with that claim and none of them is optional.

**1. n is ONE complex, not six folds.** Across all 45 Boltz-2 folds, six have
peptide heavy-atom RMSD above 2.0 Å — and **all six are the same complex**,
`HLA-B*07:02`/IPRRNVATL, in arms A and C at three seeds each. The effective
sample is one failing complex out of five. Any phrasing like "the six lowest of
45" would read as six independent failures and overstate this.

**2. The within-complex control is the argument.** Hold the allele and the
peptide fixed and vary only the construct:

| `HLA-B*07:02` IPRRNVATL | mean heavy RMSD | `pep_ll_mean` |
|---|---:|---:|
| Arm A | 2.272 Å | −3.061 |
| Arm C | 2.251 Å | −3.076 |
| **Arm B** | **0.299 Å** | **−2.388** |
| *(the other four complexes, all 36 folds)* | — | *−2.513 to −1.881* |

**When the same complex folds correctly in arm B, its score moves inside the
good range.** Same allele, same peptide, same model, same seeds; the pose varies
and the score follows the pose. Pair that with the wrong-backbone control: score
all five pilot peptides on each backbone and the native sequence's advantage is
**+13.15 nats** on the correct arm-B backbone but **collapses to roughly zero**
(−0.45, −0.17, +0.08 per seed) on the arm-A backbone known to be wrong —
*despite Boltz-2 having been handed that very sequence in both arms*. That is
what rules out the obvious objection that this is a tautological readback of the
folder's own input: ProteinMPNN recovers the sequence only when the backbone is
actually correct.

**3. The claim is specificity, not separation.** The tempting claim — "the bad
folds separate cleanly" — would be unsound. Repeating that separation test on
the A+C folds of *every* complex yields **7, 9, 2, 13 and 2** "fully separating"
structural features respectively, with the genuinely failing complex scoring the
**fewest**. Searching ~100 features for one that splits a six-fold group always
succeeds, so clean separation measures **complex identity, not pose quality**,
and **none of the 109 structural features isolates the failure**. What makes
`pep_ll_mean` different is that it was a *single predeclared quantity*, fixed in
the module docstring before any score existed, and it flags the one complex that
actually folded badly **with zero false positives among the 36 folds of the
other four complexes**. It is the only readout tested that does. For contrast,
Boltz-2's own confidence does not separate them at all: bad-fold PAE 1.376–1.552
sits inside good 1.136–2.196.

**Units, since three columns are one quantity:** `mpnn_score` = −`pep_ll_mean`,
and `pep_ll_total` = `pep_ll_mean` × 9. The gap quoted as 4.238 is
`pep_ll_total`; as 0.471 it is per residue.

**Decision: no regression feature, buy the ~$1 QC sample.** Spending $14.68 to
add a feature with no label correlation would be buying a number to put in a
table. The QC sample is bought because stage 4c established that no PAE or pLDDT
threshold catches pose failure without flagging accurate predictions, and the
project currently has **no cohort-wide way to estimate the pose-failure rate at
all**. Its protocol is predeclared: 2,000 folds, seeded draw from a *complete*
half, 16 decoding orders, two reference points from the single known-bad
complex. What will be reported is **a distribution and a triage list, never a
failure count or a failure rate**, the reference points **must not filter the
production cohort**, and the single-complex provenance travels with the number
wherever it is quoted.

One further result worth keeping: ProteinMPNN's seed control **independently
corroborates the stage 4c rejection of ESMFold2 without using a crystal
structure at all.** Between-complex spread over within-complex seed spread sits
at **3.1–9.3 for Boltz-2 and 1.2–1.9 for ESMFold2** — quoted as an envelope over
six defensible aggregations rather than a point estimate, because the
aggregation choice moves the number and every choice returns the same verdict.
Stage 4c rejected ESMFold2 on peptide heavy-atom RMSD *against crystal
structures*; this measurement uses no crystal at all, and asks instead whether
the predicted backbone is self-consistent with the sequence that produced it.
Two unrelated routes to one rejection is worth more than either alone. Note that
1.9 is a **marginal** verdict, not an inverted one — ESMFold2's between-complex
spread still exceeds its seed spread.

#### Two of our own controls disagree, and the disagreement is the finding

Honesty requires reporting this rather than quoting whichever control is
convenient. The stage 4c.5 workstream ran its own seed control on the same 90
folds, over its 109 structural features, and got a **different answer for
ESMFold2**:

| Readout on the same 90 folds | Boltz-2 | ESMFold2 |
|---|---:|---:|
| 109 structural features, median between/seed ratio | 5.29 | **5.87** |
| ProteinMPNN `pep_ll`, envelope over six aggregations | 3.1 – 9.3 | **1.2 – 1.9** |

They agree on Boltz-2 — both comfortably signal-dominated, which is what
matters operationally, since **production is Boltz-2 only**. For ESMFold2 they
diverge: ProteinMPNN falls to the noise floor while the structural features are
if anything *marginally more* seed-stable on ESMFold2 than on Boltz-2. **These
two controls must not be presented as confirming each other on ESMFold2. They do
not.**

**The mechanism is granularity.** ProteinMPNN reads **backbone geometry only** —
its featuriser takes N, CA, C and O plus a *virtual* CB computed from N/CA/C, so
it never sees a real side-chain coordinate. It is a fine-grained reader:
inter-atomic distances to 48 neighbours, with the virtual-CB direction set by
backbone dihedrals. The 109 structural features are coarse aggregates — block
means over hundreds of residue pairs, atom counts, a buried-area fraction — that
sub-Angstrom jitter does not move. And ESMFold2's backbone genuinely is
seed-unstable at that scale: its **CA** jitter is **2.5-5.2x Boltz-2's**, and
its seed variation is *backbone-dominated* (heavy/CA = 1.0) where Boltz-2's is
*side-chain-dominated* (2.45-3.63). A fine-grained backbone reader is exactly the
instrument that sees this; a coarse aggregate is exactly the one that does not.

**The explanation was cross-validated, which is why it is offered as an
explanation rather than a story.** Holding the denominator fixed and varying
only the numerator convention — averaging the three seeds before measuring
between-complex spread, which removes seed noise from the numerator — ESMFold2's
ratio falls by **-0.29 in both independently written pipelines, identical to two
decimals**, while Boltz-2's barely moves (-0.03 in one, +0.12 in the other, and
in *opposite directions*, which is what "barely moves" should look like). A
*directional, magnitude-asymmetric* prediction borne out on two separate
implementations is evidence; two numbers landing near each other is not. This
project has already retracted one claim that failed exactly that test — an
earlier note that the two implementations reconciled to two decimal places,
which was a coincidence of two different conventions landing nearby, and is
withdrawn in both reports. **Neither set of figures has been reproduced to the
digit by the other, and neither report claims otherwise.**

So: **complementary, not redundant.** Each readout integrates over a different
scale, and the disagreement localises where ESMFold2's instability lives — in
fine backbone detail, not in the coarse pose descriptors. Nothing material turns
on it, since production is Boltz-2 only; it governs how ESMFold2 is *described*,
not what runs.

### 4.6 A censored likelihood fixes the floor and loses the ranking

Stage 1 recorded that treating a detection floor as an exact zero is the
project's biggest modelling compromise, and that a **left-censored (Tobit-style)
likelihood** is the principled fix: for a floor row the honest statement is not
"t<sub>1/2</sub> = 0" but "the latent value is at or below the detection limit".
That has now been built and tested
([`stage7_censored.md`](stage7_censored.md)).

Everything was predeclared before a single censored model was fitted — the loss,
the detection limit **c = log1p(0.1 h)** justified from the 0.1 h reporting grid
rather than from any score, a five-point sensitivity sweep, the prediction
heads, and the expectation that a censored loss should improve *calibration near
the floor* rather than *ranking*. Two 30-network ensembles, identical in
everything but the objective.

| | median per-allele ρ | MAE log1p | mass below the limit | ECE | floor AUROC |
|---|---:|---:|---:|---:|---:|
| `log1p`-MSE (the baseline) | **0.6931** | **0.4734** | 0.070 | 0.0561 | 0.8862 |
| censored @ 0.1 h | 0.6518 | 0.5763 | **0.168** | **0.0422** | 0.8853 |
| *observed* | — | — | *0.196* | — | — |

All figures in this section are **validation**, 2,802 scored rows over 68
eligible alleles; the test split was not read.

**Δ median per-allele Spearman = −0.0414, 95% CI [−0.0780, −0.0062].** The
interval lies **entirely below zero**, so under the predeclared reading this is
the one verdict this project has not otherwise produced: **worse, conclusively** —
not inconclusive, and not merely "fails to clear the bar".

Three things make that a real negative rather than a metric mismatch:

- **It loses on its own objective.** Held-out censored NLL is **1.0765** for the
  censored arm against **1.0452** for the MSE arm. We did not optimise one thing
  and measure another.
- **The threshold is not load-bearing.** Across a six-fold range of the
  detection limit, 0.05–0.30 h, median ρ spans **0.0095** — far below the seed
  spread, let alone the 0.05 bar. The declared 0.1 h is neither the best nor the
  worst point in that sweep, which is what a threshold fixed before training
  should look like.
- **It delivered exactly the calibration it promised, and that was not enough.**
  Predictive mass below the limit moves 7.0% → **16.8%** against an observed
  19.6%, and expected calibration error improves 0.0561 → **0.0422**. But floor
  *discrimination* does not move at all (AUROC 0.8862 vs 0.8853) — which is what
  §5 of that report predicted, because **ranking within the tied floor block is
  unidentifiable under either objective**. The censored arm places floor rows
  better on the scale; it does not order them better.

**The mechanism, reported as a control and not as a result.** The censored arm's
own dev objective bottoms out at epoch ~10 against the MSE arm's ~27, so it is
systematically **undertrained**. A control changing only the stopping rule — same
loss, same gradients — recovers almost all of the ranking (−0.0065 against its
reference, versus −0.0318) while roughly halving the calibration error. That
control was predeclared as a control, ran at a reduced budget, and **has no
confidence interval**. It localises a likely fix; it does not establish one.
Promoting it to a result would be precisely the post-hoc selection this stage
exists to avoid, so it stays a control.

**What stands:** the frozen `log1p`-MSE baseline. Nothing downstream switches
loss on this evidence. What the censored arm uniquely offers is a *probability
that a pair sits below the assay floor*, produced by its own fit rather than by
a scale bolted on afterwards — useful if a downstream application needs that
number, which this benchmark does not.

---

## 5. Results

All figures below are **validation**. The test set is scored **once**, at stage
6, with every model together.

Two results above do not appear in the table below because they are not on the
per-allele-Spearman axis: the elution transfer (§4.4) is a binary
ligand-versus-decoy discrimination, and inverse folding (§4.5) is structural QC.
Both are reported in full where they sit rather than forced onto a metric they
do not answer.

### 5.1 The table

| Arm | Information added | Median per-allele ρ | Paired Δ vs its control [95% CI] | Verdict | $ / 1,000 preds |
|---|---|---:|---|---|---:|
| Training allele mean | none | 0.000 | — | floor | — |
| MLP, peptide only | peptide identity | 0.202 | — | — | $2.6 × 10⁻⁹ |
| Ridge, pep + pseudoseq | linear only | 0.278 | — | — | not measured |
| MLP, pep + pseudoseq (single) | + 34 contact residues | 0.610 | −0.083 [−0.124, −0.029] | worse than its own ensemble | $6.6 × 10⁻⁹ |
| 30-net ensemble, pep + domain | + 182 domain residues | 0.653 | −0.040 [−0.073, +0.001] | tied | $7.9 × 10⁻⁷ |
| **30-net ensemble, pep + pseudoseq** | **the baseline to beat** | **0.693** | — | — | **$9.6 × 10⁻⁷** |
| Weak-binder augmentation (best arm) | 2,910–4,407 assumed-zero rows | 0.618 | +0.024 [−0.026, +0.048] | **inconclusive, rules out 0.05** | $9.6 × 10⁻⁷ |
| Auxiliary affinity head (best λ) | 5,135 affinity labels | 0.701 | +0.008 [−0.017, +0.030] | **inconclusive, rules out 0.05** | $9.6 × 10⁻⁷ |
| Censored (Tobit) likelihood | same features, censored loss | 0.6518 | **−0.0414 [−0.0780, −0.0062]** | **worse, conclusively** | $9.6 × 10⁻⁷ |
| **ESM-2 only** (30-net) | pretrained sequence embeddings | **0.6830** | **−0.0101 [−0.0382, +0.0352]** | **inconclusive, rules out 0.05** | **$4.7 × 10⁻⁴** |
| **sequence + ESM-2** (30-net) | both | **0.6761** | **−0.0170 [−0.0468, +0.0320]** | **inconclusive, rules out 0.05** | $4.7 × 10⁻⁴ |
| *ESM-2 only vs the **full-domain** ensemble* | *matching input* | *0.6830* | ***+0.0301 [−0.0084, +0.0757]*** | ***inconclusive — the only upper bound above the bar*** | *$4.7 × 10⁻⁴* |
| ESM-2 150M only | larger checkpoint | 0.6737 | −0.0194 [−0.0599, +0.0195] | inconclusive, rules out 0.05 | 2.6× the 35M extraction |
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
| ~~E2~~ | ~~ESM-2 validation accuracy and its paired CI~~ | — | **Filled**: §4.0. Compared against both the pseudosequence ensemble (−0.0101) and the full-domain ensemble (+0.0301), as required |
| **B2** | Boltz-2 structural accuracy and its paired CI | stage 4c production → stage 5 | Same, on identical train/val/test rows with matched head architecture, ensemble size and tuning budget. Plus coverage and the declared sequence fallback for structural failures |
| **B3** | Realised production spend | Modal production session | Metered before/after snapshots per workspace, realised GPU-hours and wall clock, failure count, actual concurrency granted. `production_<profile>.jsonl` exist for both profiles and are currently **empty** |
| **A1** | Per-stratum leave-allele-out for the ESM-2 and structural arms (§6.6) | `esm-arm` / stage 5 | A **feature matrix plus its `pair_id` index**, refit across all 68 folds at the fixed 6 networks per fold — **not** a `preds/*.csv`, which the runner rejects. A stratum may come back inconclusive and must be reported as such |
| **B4** | Structural coverage and failure rate (§6.5) | stage 5 | Pairs with a valid structure, pairs falling back to the sequence model, and the primary result reported on the **frozen cohort**, not on whatever folded |
| ~~E1b~~ | ~~ESM-2 head inference cost~~ | — | **Filled**: 0.0150 s / 1,000 rows for the 30-network ensemble; 1.14 s end to end including embedding. §4.0 |
| ~~S3C~~ | ~~Elution external validation~~ | — | **Filled**: [`stage3c_elution_validation.md`](stage3c_elution_validation.md), §4.4 above. Still open as a *follow-on*: the same pass on the ESM-2 and structural arms |
| ~~S7a~~ | ~~Censored (Tobit) likelihood result~~ | — | **Filled**: [`stage7_censored.md`](stage7_censored.md), §4.6 above. Negative on ranking, with the sensitivity sweep and the undertraining control |
| **S6** | Stage 6 test results, distance strata, differential target, nested near-neighbour CV | `eval-harness` | §6 below |

---

## 6. Evaluation beyond a single number

A median per-allele Spearman is one number on one panel. Five further
evaluations were specified in the frozen contract, each answering a question the
headline cannot. **They run at stage 6 and cost no new compute** — all are
re-aggregations of predictions already made.

Two more have already been delivered: transfer to an entirely different assay
(§4.4, with its three specificity controls) and the leave-allele-out evaluation
(§6.6), which needed a second contract of its own.

### 6.1 Distance stratification — is the model generalising or remembering?

Every held-out peptide is 4 or 5 substitutions from its nearest training
peptide, and label similarity at d=4 is still Spearman 0.592. So test metrics are
reported in two strata: **d=4 (3,256 rows, 57.8%)** and **d≥5 (2,377 rows,
42.2%)**, scored on the **same 65 alleles** (the intersection of those eligible
within each stratum) — otherwise the comparison measures allele panels rather
than distance. A d≥6 stratum is not viable: 12 test rows reach it.

**If the model is exploiting residual similarity at the split boundary, d=4 will
score visibly higher than d≥5. If it does not, that is a strong positive signal
that the model genuinely generalises.** Test result: **‹HOLE S6a›**.

**On validation, the hypothesis that pretraining buys generalisation at the
split boundary is dead.** Every arm loses something between the strata, and the
point estimates tease — the 150M arm barely degrades (gap 0.014 against the
baseline's 0.082) and posts the highest d≥5 score of any arm. **The intervals do
not support it.** Paired bootstrap on the stratum gap itself, against the
baseline:

| Arm | Δ gap | 95% CI |
|---|---:|---|
| ESM-2 35M | −0.0197 | [−0.0973, +0.0800] |
| sequence + ESM-2 | +0.0046 | [−0.0803, +0.1235] |
| ESM-2 150M | −0.0685 | [−0.1188, +0.0625] |
| full-domain ensemble | −0.0091 | [−0.0888, +0.0847] |

**All four cross zero.** Two point estimates ordered the way a size effect would
order them, with intervals that each admit the opposite ordering, is not a
trend — it is two draws from a noisy statistic.

**And this is a statement about the split's size, not about the arms.** The
stratum gap is a *difference of differences of medians*: two medians per arm,
differenced, then differenced again against the baseline. Each step compounds
the noise, and the resulting half-width is near **0.09** on a quantity whose
largest observed value across all five arms is **0.069**.

**The obvious objection is that we only ran the underpowered version of the
test. We did not.** The gap test is a difference of differences; the
better-powered question is whether an arm's score *within* a stratum simply
beats the baseline's, as a single difference. That was asked too, in both
strata. Within **d=4** (1,377 rows), against the sequence baseline:

| Arm | Δ at d=4 | 95% CI |
|---|---:|---|
| ESM-2 35M | −0.0246 | [−0.0706, +0.0312] |
| sequence + ESM-2 | −0.0339 | [−0.0707, +0.0383] |
| ESM-2 150M | −0.0466 | [−0.0906, +0.0166] |
| full-domain ensemble | −0.0397 | [−0.0821, +0.0141] |

Within **d≥5** (992 rows), where the hypothesis predicted the larger checkpoint
should pull ahead:

| Arm | Δ at d≥5 | 95% CI |
|---|---:|---|
| ESM-2 35M | −0.0049 | [−0.0804, +0.0632] |
| sequence + ESM-2 | −0.0385 | [−0.1167, +0.0542] |
| **ESM-2 150M** | **+0.0219** | **[−0.0776, +0.0619]** |

That bolded row is the decisive one. It is the single comparison the whole
flat-profile hypothesis reduces to — *does the larger checkpoint actually beat
the baseline where peptides are most distant?* — asked in its best-powered form,
and it is **inconclusive**.

**So the question was asked three different ways — the stratum gap, within d=4,
and within d≥5 — and the split answered none of them.** Eleven of the twelve
intervals are in and **not one separates any arm from the baseline in either
stratum**; the twelfth, the full-domain arm at d≥5, is still running and cannot
change that. This is a thorough null, not a single underpowered look. A
better-powered split, not a re-analysis of this one, is what would settle it.

> **A note for anyone checking the numbers against the CSV.** The 150M d≥5
> point estimate (+0.0219) sits in the *upper* part of its own interval rather
> than at the centre. That is expected, not an error: the statistic is a median
> over a 55-allele panel that is **rebuilt on every resample**, so its bootstrap
> distribution is skewed. It is one more reason never to read a point estimate
> from this statistic without its interval attached — which is the same lesson
> the rest of this section teaches.

### 6.2 The differential target — groove chemistry, isolated

For peptides measured on two or more alleles, evaluate **Δ log half-life between
alleles**. This subtracts out whatever is intrinsic to the peptide and tests
groove chemistry directly — the sharpest available version of "distinguish
within-allele ranking from cross-allele effects". It is abundant: **3,941
peptides sit on ≥ 2 alleles, covering 26,474 rows (94% of the dataset), up to 36
alleles for a single peptide.**

**On validation this delivered the project's sharpest result**, and because it
bears directly on the headline question it is reported in full at
**§4.0**: ESM-2 and the sequence baseline are **equivalent to within one point
of concordance**, while the same measurement conclusively separates two other
arms — so the equivalence is a measured null rather than a blind metric. This is
the only metric in the project that returned a conclusive verdict where the
primary metric could not, which is the clearest possible demonstration that it
earned its place in the contract. Test result: **‹HOLE S6b›**.

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

### 6.6 Leave-allele-out — a second contract, and the headroom it exposes

This is the one evaluation that asks a *different* question from the headline:
not "unseen peptides on alleles we trained on" but **"unseen allotype"**. It
needs its own split, its own contract and every number reported twice, which is
why the plan put it at "if time remains". It has now been run for the sequence
arm ([`stage7_allele_holdout.md`](stage7_allele_holdout.md)): 68 folds, one per
eligible allele, each fitting on every *other* allele's rows, 6 networks per
fold, no tuning, `split in {train, val}` only. **The frozen test split was not
read, and `data/splits.csv` is unmodified.**

| Stratum (pseudosequence Hamming to nearest training allele) | Alleles | Median ρ | 95% CI |
|---|---:|---:|---|
| near, d ≤ 1 | 25 | **0.741** | [0.642, 0.803] |
| intermediate, d = 2–3 | 23 | 0.576 | [0.458, 0.726] |
| distant, d ≥ 4 | 20 | **0.339** | [0.298, 0.491] |

**near − distant = +0.403 [+0.223, +0.478].** The sequence baseline degrades
sharply on allotypes unlike anything it trained on.

**The extreme contrast is solid; the monotone three-bin trend is not.** The
adjacent contrasts do not hold up — near − intermediate is **+0.166 [−0.010,
+0.327]**, which crosses zero — and the strata overlap heavily: the distant
bin's *best* allele (0.753) beats the near bin's *worst* (0.419). The medians
separate; the distributions do not. Reporting a monotone trend would overstate
what this design resolves.

**This is where a pretrained model has its strongest prior of winning** —
pan-allele generalisation is exactly what large-scale pretraining should
supply — so a 0.403 deficit in that stratum is a large, measurable headroom.
**Nothing here says ESM-2 will capture it.** A measured deficit is a
precondition for the claim, not evidence for it.

**Two constraints that travel with every number above.**

*The confound is real, but not the one the plan described.* The plan asserted
that holding out an allele also holds out its peptide panel. Measured, that is
**not what happens**: the median peptide is assayed on **4 alleles**,
allele-exclusive peptides are only **6.0% of rows**, and for the median eligible
allele **100%** of its rows carry a peptide seen on some other allele. Peptide
overlap is uncorrelated with distance (−0.067) and with per-allele performance
(+0.034). So this evaluation is largely *seen peptide, unseen allotype* — easier
than the frozen split in one respect, harder in another, and its absolute
numbers **must never be quoted beside a frozen-split number**. What does survive
is **panel composition**: distant alleles carry weaker-binding, more heavily
censored panels (Spearman(distance, zero share) = +0.251, p = 0.039). Partialling
that out leaves distance at **−0.604** (p = 4.9 × 10⁻⁸) against −0.636 raw — so
the confound is **attenuated, not eliminated**, and zero share is only one proxy
for panel composition. The clearest illustration is the single worst fold:
`HLA-B*39:06(C67S)` at **d = 3**, with a 91.8% floor panel — it scores badly
because of its panel, not its distance.

*A prediction file cannot be scored through this contract.* Everything in
`preds/*.csv` is the output of a model fitted on **every** allele. Scoring one
here would score a model on alleles it trained on and **report the leak as
pan-allele generalisation — a number that would look like a win.**
Leave-allele-out refits 68 times, so **each arm must supply features and be
refit**, not hand over predictions. The runner rejects a `preds/*.csv` with that
message and a test guards it. Ensemble size is fixed at 6 networks per fold for
every arm, so no arm can win on ensembling budget.

Per-stratum ESM-2 and structural comparisons are therefore worth running and are
**not guaranteed to conclude**: the measured minimum detectable effect is 0.025–0.030
when per-allele deltas are tight, rising to 0.075–0.115 when they are loose, so
an inconclusive stratum must be reported as inconclusive rather than as a null.

### 6.7 One of our own predeclared metrics does not work, and we are saying so

**Precision@10 at 2 hours cannot express a difference between these arms, and
that is a criticism of a metric we chose in advance.**

Against the sequence baseline, all four comparisons returned a delta of
**exactly 0.0000**, with interval bounds that are themselves lattice points of
the statistic: [−0.100, +0.100] three times and [−0.100, +0.050] once. Every arm
shares a median precision of 0.700 and a median ceiling share of 0.900.

This is not ordinary low power. **The statistic moves in steps of 0.1 because it
is ten slots**, and a median over 68 alleles of a 0.1-quantised quantity lands
on a lattice point and stays there under resampling. An underpowered metric
gives you a wide interval around a non-zero point estimate; this one gives a
point estimate that is *pinned* to zero by construction. It cannot represent the
0.04 differences the primary metric reports.

What that leaves: only the **lift** and **ceiling-share** columns carry
information, and here even they are within noise (lift spans 0.277–0.294 across
arms that differ by 0.04 on the primary metric). It did separate the single
network from the ensemble at stage 2, so it is not useless in general — it is
useless at this resolution.

**It stays in the report**, because it was predeclared and removing a metric
after seeing that it is unflattering to the process would be exactly the
post-hoc selection this contract exists to prevent. But **no claim in this
submission rests on it.** Reporting a weakness in a metric we chose ourselves is
worth more than quietly dropping it; a reader can then judge the contract, not
just the results it produced.

### 6.8 Was the evaluation machinery worth building?

The honest answer, and it has two halves.

**One of the five analyses paid for itself, and we can say exactly which.** Of
everything in the ESM run, **exactly one family of comparisons returns a
conclusive verdict**: the differential target, which places the 150M checkpoint
and the full-domain ensemble conclusively below the baseline on cross-allele
ranking while showing 35M and the additive arm equivalent to within one point of
concordance (§4.0). The primary metric, the distance strata, precision@10 and
nested mutant ranking are **inconclusive on every arm**.

> **The differential target was the only analysis with enough power to resolve
> anything on this split.**

That is worth saying plainly rather than letting five analyses share credit
evenly. It also vindicates a specific design decision: the differential was
built to separate within-allele ranking from cross-allele effects, and
subtracting the peptide's own contribution is precisely what bought the
precision — its intervals are four to six times tighter than the primary
metric's on the same rows.

**The four that returned nothing returned *measured* nothings, and that is not
the same as returning nothing.** Each carries a stated minimum detectable
effect: the distance question was asked three ways and answered none (§6.1),
with the arithmetic showing why — a half-width near 0.09 on a quantity whose
largest observed value is 0.069. Precision@10 cannot express a difference at
all, and we can show that from the lattice structure of its own intervals
(§6.7). The allele hold-out publishes its MDE per stratum (§6.6).

**That distinction is what the whole submission rests on.** Every negative here
is of the form "ruled out at 0.05" or "equivalent to within one point", never
"we looked and saw nothing" — and the only thing separating those two
statements is machinery that knows what it can and cannot detect. An evaluation
that returns four inconclusive results *and can prove they are inconclusive
rather than null* is doing its job. One that returns four inconclusive results
and cannot tell you which is which has told you nothing at all.

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

Two further spends are **approved and not yet made**, both forecast from
measured unit costs:

| Item | Cost | Basis |
|---|---:|---|
| Stage 4c.5 structural feature extraction, both halves | **~$0.78** | CPU-rate × measured container time. Extraction is **Volume-read-bound, not CPU-bound**: 1.57 s of wall per fold against 0.096 s of CPU, about 6% core utilisation — so the container was dropped from 2 cores to 1, since reserving the second was billing an idle one |
| ProteinMPNN QC sample, 2,000 folds | **$1.04** | measured 14.70 s/fold at 16 decoding orders, at the repo's metered rates |

**Both are rounding error against the $200.8 fold.** Worth one line on method:
for these two the *published list rates* agreed with the metered rates to
**0.3%** (CPU 1.0030×, memory 1.0010×). Set against the 4.6× discrepancy this
project hit earlier, the lesson is **"check it", not "never trust published
rates"** — the earlier error was a broken harness measuring model reloads, not a
wrong price list.

### 7.2 Cost per 1,000 new predictions — the decision number

| Arm | $ / 1,000 | GPU-h / 1,000 | Basis |
|---|---:|---:|---|
| Sequence ensemble, 30 networks | **$9.6 × 10⁻⁷** | 0 | **measured** 0.073 s per 1,000 rows end to end, at Modal's published $0.04730/CPU-core-hour |
| **ESM-2, 35M, end to end** | **$1.5 × 10⁻⁵ – $4.7 × 10⁻⁴** | 0 (laptop `mps`) | measured 1.14 s per 1,000 new predictions. The range is the rate, not the time: the low end prices it at the CPU rate, the high end at the A10G rate, because the embedding step ran on a laptop GPU for which no hourly rate exists |
| **Boltz-2 structural** | **$6.90** | **4.66** | measured 16.76 s fold × measured $1.4812/h A10G |
| ProteinMPNN inverse folding | $0.52 | 0 (CPU) | measured 14.70 s/fold at 16 decoding orders × metered CPU + memory rates; **forecast**, QC sample only |
| *ESMFold2 (rejected)* | *$31.40* | *11.8* | *measured 42.63 s × $2.6512/h L40S; forecast only* |

**The structural arm costs about 7 × 10⁶ times more per prediction than the
sequence ensemble it has to beat; ESM-2 costs about 16×.** In wall clock rather
than dollars: scoring the entire dataset takes the sequence ensemble **about two
seconds of one CPU core**, ESM-2 about half a minute, and Boltz-2 **131
GPU-hours**.

*A correction to an earlier figure in this document's history:* the sequence
ensemble's inference cost was previously **derived** as 30 × stage 2's
single-network 0.0005 s per 1,000 rows, giving $2.0 × 10⁻⁷. Stage 3's harness
**measured** it at 0.0656 s per 1,000 rows (0.073 s end to end), so the
derivation understated it by about 4×, and the Boltz-2 ratio falls from
3.5 × 10⁷ to 7 × 10⁶ accordingly. The measured figure supersedes the derived
one. The conclusion does not move — it was never close — but a ledger that
quietly keeps the tidier number is not one worth reading.

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
2. **The peptide panels are allele-confounded by design — though not in the way
   we first wrote down.** Each allele was assayed on its own panel; the grid is
   6.7% full, and this broke the first split. The plan then claimed that holding
   out an allele also holds out its peptide panel. **Measured, that is false**:
   the median peptide sits on 4 alleles, allele-exclusive peptides are 6.0% of
   rows, and for the median eligible allele 100% of its rows carry a peptide
   seen elsewhere (§6.6). The confound survives through **panel composition**,
   not panel hold-out — distant alleles carry more heavily censored panels
   (+0.251, p = 0.039) — and partialling that out leaves the distance effect at
   −0.604 against −0.636 raw. **Attenuated, not eliminated**, and zero share is
   only one proxy for panel composition.
3. **The assay panel was pre-selected by predicted affinity**, so peptide
   diversity is narrow by construction. The benchmark measures ranking *within
   the set of peptides that bind*. Broader biological or clinical claims need
   evidence from outside this file.
4. **Our negative results are "ruled out at 0.05", never "zero".** Stage 2b's
   best arm is +0.024 [−0.026, +0.048]: the data are consistent with a real
   +0.04. What is excluded is +0.05. A larger benchmark would move that bar down
   and some of what we call ruled out would become detectable.
5. **The ESM-2 null is bounded to the representation actually tested**, and the
   bounds are not a formality. It is about *frozen* embeddings of the peptide
   and HLA domain taken **separately**, at 35M and 150M, with a small MLP head.
   It does **not** establish that fine-tuning would not help (nothing was
   fine-tuned); that a model shown the *complex* would not help (the head must
   learn the peptide–HLA interaction itself from 19,716 rows, and the plan
   predicted this would be the binding constraint — the ESM-only arm reaching
   parity while adding nothing on top is consistent with exactly that); that
   log-likelihood or perplexity features would not help (embeddings are one of
   three ways the brief names); that a larger checkpoint would not help (flat
   across the two sizes tested, which weakens but does not close the argument,
   and 650M was excluded for memory); or that another protein-language-model
   family would behave the same way. **One family is a thin basis for a general
   claim.** The chimeric peptide-linker-groove input remains the sharpest
   untested version and is listed as out of scope **specifically so that this
   null is reported as bounded**.
6. **ESMFold2's rejection is operational, not scientific.** Pose accuracy is not
   feature utility, and a five-complex gate is a decision rule for spending $200,
   not a benchmark of ESMFold2 on peptide–MHC.
7. **The two newest results carry the tightest bounds of anything here.** The
   elution transfer is not a measurement of stability-prediction accuracy —
   elution has at least four non-stability filters, and the allele-swapped
   control bounds the generic-presentability share at 0.050 AUROC without
   decomposing the biology ([`limitations.md`](limitations.md) §7.1). The
   ProteinMPNN pose-triage signal rests on **n = 1 failing complex**, is not a
   half-life predictor (ρ = −0.100, p = 0.87, n = 5), and its reference points
   must never filter the production cohort (§7.2). **FoldX and Rosetta are
   declined on the thermodynamic/kinetic mismatch** — and separately because
   both are licence-gated behind registration, which makes them a
   reproducibility cost for anyone extending this work (§7.0).

---

## 9. A finding that transfers: where you read the embedding matters more than which model produced it

On **validation**, the mean-pooled HLA arm scores 0.177–0.270 against the
34-contact arm's 0.469–0.499. The natural explanation — pooling over 182 residues discards
information — is **wrong**, and this dataset can prove it. There are only **75
distinct HLA domain sequences**, so any HLA representation has rank ≤ 74. Both
representations measure rank 74, both are **losslessly** representable in 74
components, and both separate all 75 alleles exactly. **They carry identical
information**; an unconstrained model could not tell them apart.

The entire 0.3 gap is **similarity geometry** — contact-position embeddings
place alleles with similar binding pockets near each other, mean-pooled
embeddings place alleles with similar overall sequence near each other, and only
the first is the right notion of "similar" for this task. For a practitioner
this is the most transferable finding in the stage: **where you read a
foundation model's embedding from matters more than which foundation model
produced it.**

**Why this generalises past this dataset.** The proof above depends on a quirk —
only 75 distinct HLA sequences, so rank is bounded at 74 and "lossless" is
checkable directly. But the *lesson* does not depend on the quirk. Any time a
pretrained embedding is pooled over a long sequence to score a local
interaction, the pooling is choosing a similarity geometry, and that choice can
cost more than the choice of model. Here it cost **0.3 of Spearman** — six times
the predeclared worthwhile gain, and far more than the gap between ESM-2 35M and
150M, or between ESM-2 and the sequence baseline.

For a practitioner picking a foundation model for a binding-site problem, the
ordering of concerns this project measured is: **read the right residues first,
then worry about which checkpoint.**

---

## 10. What we would do next

Ordered by expected value per hour, not by appeal.

1. **Finish the in-flight arms and score the test set once** (holes E1b, E2,
   B2–B4, S3C, S6, S7a). Everything else is downstream of knowing whether the
   expensive arms clear 0.05.
2. **Re-run the censored likelihood with the MSE stopping rule.** §4.6 settled
   the headline question — as predeclared, the censored loss is **worse on
   ranking**, conclusively. But it localised a likely cause: the censored arm's
   own dev objective turns over at epoch ~10, so it is undertrained, and a
   control changing only the stopping rule recovers almost all of the ranking
   *and* halves the calibration error. That control has no interval and was run
   at reduced budget. Running it properly, predeclared, at full budget is the
   obvious next experiment — and it must be predeclared, because selecting it
   now on the strength of the control would be the post-hoc selection the stage
   avoided. A heteroscedastic scale and a per-allele floor are also untested.
3. **Test the chimeric ESM-2 input.** The separate-embedding arm *is* flat
   (§4.0), so the obvious objection now lands: we never let the language model
   see the interaction. A peptide-linker-groove construct is the sharpest
   version of the question and the biggest risk to this negative result — which
   is precisely why it is worth running rather than avoiding. Two cheaper
   follow-ups sit alongside it: **likelihood and perplexity features**, which are
   one of the three uses the brief names and which we did not test at all, and a
   **second pLM family**, since one family is a thin basis for a class-level
   claim.
4. **Re-run the auxiliary-affinity probe on the ESM-2 arm.** The machinery is
   protocol-agnostic and the leakage audit is already done (64,226 admissible
   rows). Affinity is redundant with what a *sequence* model extracts; that says
   nothing about ESM-2 features. "Cheap labels substitute for expensive
   pretraining" would be a genuinely useful finding.
5. **Run the ESM-2 and structural arms through the leave-allele-out contract.**
   §6.6 has already run it for the sequence arm and found a **0.403 deficit** on
   distant allotypes — the stratum where pretraining has the strongest prior of
   winning. A gain confined there would be real and reportable even if the
   pooled comparison came out flat. **Each arm must supply features and be refit
   across all 68 folds**; handing over a `preds/*.csv` would score a model on
   alleles it trained on and report the leak as generalisation.
6. **Check the ProteinMPNN pose-triage signal on more than one failing
   complex.** §4.5 is a promising observation resting on n=1, and it will stay
   that way until production structures exist and the QC sample runs. Until then
   it must not be called a failure detector and must not filter anything.
   **ESM-IF**, the brief's second inverse-folding model, was not attempted —
   machine contention, with the ESM-2 arm holding priority on the shared
   environment — and ProteinMPNN alone covers the class.
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
