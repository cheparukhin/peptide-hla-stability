# Stage 3 reconciliation: two independent ESM-2 implementations, and three experiments neither had run

**What this file is.** `origin/stage3-esm` is a second, independent stage 3
implementation, forked at `3576c02` and never merged — 24 new files, +4,007
lines, no deletions, 45 commits behind `main`. It reaches the **opposite**
headline conclusion from `main` on the same question. This file resolves that
disagreement, records what the branch had that `main` did not, and reports the
four experiments the branch had built but never executed, which are now run.

Nothing here changes which arm ships. `main`'s stage 3 verdict stands exactly as
published in [stage3_esm.md](stage3_esm.md).

| | branch | main | resolution |
|---|---:|---:|---|
| sequence baseline | 0.6904 | 0.6931 | agree (independent rebuild) |
| ESM-2 only, 30 nets | **0.3866** | **0.6830** | **main**; branch arm untuned |
| ESM-2 stacked / additive | 0.5387 | 0.6761 | **main**; same cause |
| verdict | "worse, CI below 0" | "inconclusive at 0, rules out +0.05" | main's |

---

## 1. The disagreement, and why it resolves in main's favour

Everything about the two ESM arms matches **except regularisation**:

- **Representation: identical.** Peptide per-position, HLA 34-contact, layers
  mid/final, peptide PCA to 256, HLA PCA to 74.
- **Ensemble parity: identical.** Both are 30 networks over 5 CV folds × a
  2-way axis × 3 seeds, folds cut along whole peptide clusters.
- **Regularisation: the entire gap.** `scripts/stage3_esm.py:138` on the branch
  reads `hidden, l2 = SELECTED[("pep_pseudo", "onehot")]` — stage 2's selected
  configuration passed straight into the ESM arm. This is not a narrow ladder
  but **no ladder at all**, so "is the selection interior?" is not merely
  unsatisfiable, it is undefined.

[stage3_esm.md](stage3_esm.md) §2 measures that exact transplant at **−0.109**.
The branch independently recorded the same symptoms without connecting them to
a cause: seed spread median 0.061 and max 0.184 against stage 2's observed
0.010–0.051, and a median best epoch of ~5 against the baseline's ~27. Both are
textbook under-regularisation, and both are visible in the branch's own
`stage3_runs_esm2_35M.csv`.

**The honest shape of this is an asymmetry worth stating.** The branch flagged
its own headline as provisional and wrote, verbatim, *"Until that control runs,
do not quote −0.30 as the stage-3 result."* That instinct was right. Its
diagnosis was wrong: it blamed the 256-component peptide PCA, and
[stage3_esm.md](stage3_esm.md) §6 refutes that directly — PCA at 256 components
scored **0.6081** against the uncompressed 4,320-column block's **0.5746**, so
the reduction *helped* by +0.033.

So two independent implementations arrived at the same tuning-parity finding
from opposite directions: `main` from the success side, by measuring what the
transplant cost and avoiding it; the branch from the failure side, by paying the
cost and correctly refusing to publish the number. That is stronger
corroboration of the finding than either run alone, and it is the reason this
branch is worth reading rather than discarding.

**Row counts.** The branch reports 2,802 validation rows and `main`'s
`stage3_comparisons.csv` reports 2,817. Both are correct and both were
ambiguous: 2,817 is all validation rows, 2,802 is the rows on the 68 eligible
alleles (74 alleles in `val`; 68 clear the eligibility bar). The branch agrees
with `main`'s stage 6 harness.

### Not adopted

| branch artifact | why |
|---|---|
| `reports/stage3_esm.md`, `stage3_summary_esm2_35M.csv`, `stage3_runs_esm2_35M.csv`, `stage3_cost_esm2_35M.json` | carry the untuned headline |
| `scripts/stage3_esm.py` | superseded by `scripts/esm_arm.py`, which is wired to `pepstab/esm.py` |
| `scripts/esm_features.py` | collides with `main`'s; `main`'s is the one the cache API uses |
| `scripts/stage3_head_grid.py` | superseded by §2's ladders |

---

## 2. Adopted and verified: the label-side probes

[`scripts/biology_probes.py`](../scripts/biology_probes.py) — ported, re-run,
and reproducing the branch's published numbers to four decimals using **the
repo's own BLOSUM62** (`pepstab.features.residue_rows`) instead of the branch's
pasted-in copy of the matrix. The groove distance is a normalised kernel, so the
repo's 1/5 scaling cancels; `tests/test_biology.py` asserts the four published
pairs survive the swap.

Reproduced exactly: the four near-identical allele pairs
(ρ 0.9205 / 0.9074 / 0.9012 / 0.9007 on n = 369 / 347 / 303 / 245), the
permutation-adjusted per-position variance profile (P1 0.084, P9 0.081, P2
0.074, P3 0.050, with P4 0.010 and P5 0.012 at the floor), the additive identity
oracle at **0.3107**, and the censoring burden of the three C67S constructs
(0.921, 0.890, 0.749 of labels at the assay floor).

### The noise ceiling, with two corrections

The branch's §4 derives a lower bound on assay reproducibility from allele pairs
differing at a single contact residue, which is the one quantity no other table
in this project supplies. The logic is sound — observed concordance =
reproducibility × true similarity, and true similarity is below 1 for two
different molecules, so an observed ρ bounds reproducibility from below. Two
things in the presentation needed fixing.

**There are twelve such pairs, not four.** The branch quotes four at
0.901–0.921 and reads as though those are all of them. The other eight run
0.657–0.838. The bound survives, because the correct bound is the **maximum**,
not the mean — but it has to be stated as the best of twelve.

**And there is a thirteenth pair the branch's binning hid.**
`HLA-B*14:01(C67S)` and `HLA-B*14:02(C67S)` are **identical across all 34
contact residues** — Hamming 0 — and agree at only **ρ 0.6433** on 368 shared
peptides. That pair should be the *tightest* bound available, since true
similarity on the binding surface is 1 by construction, and instead it is the
loosest in the set. The explanation is the branch's own §5 finding: both
constructs are 89% and 75% censored at the assay floor, so their ρ is largely
ranking tied zeros. Across near pairs, concordance against censoring burden is
ρ **−0.46**.

So the defensible statement is:

> Assay reproducibility is **≥ ~0.92 in rank terms on well-populated,
> lightly-censored panels** — the best of 13 near-identical allele pairs — and
> **demonstrably does not transfer to floor-heavy alleles**, where the one pair
> with zero contact-surface difference manages 0.64.

A bound that carries its own limit is worth more than an unqualified one. If it
holds, the best arm at ~0.69 leaves roughly 0.2 of measured headroom
unexplained, which is a statement about the task rather than about any arm.

**This is a post-hoc observation and must be labelled one.**
[EVALUATION.md](../EVALUATION.md) is frozen and should stay frozen — reopening
it after seeing results is the selection the contract exists to prevent. The
recommendation is that `reports/limitations.md` §1.2 ("No replicates, so no
noise ceiling") gain a labelled post-hoc paragraph, and that **no existing
verdict be restated against it**. Those verdicts were made against a predeclared
0.05 bar and stand as made. The amendment is left to whoever integrates this
branch, because `limitations.md` moved on `main` after this branch was cut.

---

## 3. New result: coupling does not rescue frozen ESM-2

[stage3_esm.md](stage3_esm.md) §9 leaves exactly one mechanism open for the
stage 3 null — peptide and HLA are embedded independently, so the head must
learn the interaction itself, and concatenation asks it to do so in its first
layer. *"ESM-2 carries nothing useful here"* and *"concatenation cannot use what
it carries"* predict the same flat result. This separates them.

[`pepstab/attn.py`](../pepstab/attn.py) lets each of the 9 peptide positions
attend over the 34 HLA contact residues.
[`scripts/stage3e_coupling.py`](../scripts/stage3e_coupling.py) runs it against
**the same module with the attention replaced by a uniform average**, at an
identical parameter count — 94,913 in both arms, asserted in-run, so a gap
between them isolates coupling from capacity.

| arm | median ρ | vs ablation | vs baseline 0.6931 | vs `esm_ensemble` 0.6830 |
|---|---:|---|---|---|
| `xattn` (coupling) | 0.5735 | — | −0.1196 [−0.1733, −0.0674] worse | −0.1095 [−0.1724, −0.0676] worse |
| `meanpool` (ablation) | 0.5571 | — | −0.1360 [−0.1861, −0.0782] worse | −0.1259 [−0.1824, −0.0777] worse |

**`xattn` vs `meanpool`: +0.0164 [−0.0178, +0.0437] — inconclusive at zero, and
the upper bound sits below the predeclared +0.05 bar.**

Letting the groove condition the peptide, with capacity held exactly constant,
does not rescue frozen ESM-2. The verdict takes the same form as every other in
this stage: it does not demonstrate that coupling is worthless, but it excludes
the gain that would have made it worth adopting. And both coupling arms land
0.11–0.14 *below* the concatenation arm, so the attention architecture is worse
here, not better. §9's open mechanism can be closed, and the "but you never
tried coupling" objection answered with a number.

60 networks, 13.6 min, 14 s per network, $0 of cloud spend.

### Three implementation changes the port required

1. **The model was not seeded.** `torch.manual_seed` was called inside `fit`,
   after the caller had constructed the module, so initialisation came from
   ambient RNG state — two "seed 0" members could differ and a run could not be
   reproduced. Seeding moved into `__init__`; `tests/test_biology.py` guards it.
2. **Rows were fanned out before training.** One `(34, d)` groove block per
   measurement row is 19,716 copies of 75 distinct domains — 172 MB of pure
   duplication on a machine sharing 16 GB with a live structural fold. A
   `Bank` holding deduplicated blocks and gathering inside the batch costs one
   index op.
3. **L2 reached biases and LayerNorm gains**, via Adam's `weight_decay`.
   `pepstab/mlp.py` deliberately penalises weights only; the arms have to be
   regularised the same way to be comparable with it.

One design improvement: the hidden-axis PCA is fitted **separately per kind**
rather than fitting on training peptide residues and applying that basis to the
groove block too. That is measured, not asserted — a shared peptide-fitted basis
retains **59.4%** of groove residue variance where a separate basis retains
**91.5%**. There is no coordinate system the two blocks need to share, since
each has its own learned input projection.

### On the ladder, and a rule chosen after seeing the data

The coupling arm's L2 ladder **would not terminate under argmax selection**.
Three points: the attention arm selected the bottom. Extended two decades
downward: the ablation selected the new bottom. The reason is that mean
validation ρ moves ~0.04 across five points spanning 1e-6 to 1e-2 while the
*seed* spread at a single point is 0.03–0.07, so most of the ladder is a
statistical tie and the argmax lands wherever the noise peaks. §2's own sentence
applies: an effect smaller than seed noise is not an effect.

So a **one-standard-error parsimony rule** was adopted: among points
statistically indistinguishable from the best, take the most regularised, which
is also the most stable. It terminates, it is applied identically to both arms,
and it independently selects **1e-3** for the attention arm, for the ablation,
and for the pooled ladder — interior to the 5-point union in all three cases.
Both the argmax and the parsimonious choice are recorded in
[stage3e_tuning.csv](stage3e_tuning.csv) so the rule is auditable.

**This rule was chosen after seeing the flatness, not before.** Disclosing that
is what makes it usable: it is a selection rule, not a selection.

**And the two coupling arms run at one common L2, not each at its own.** This
looks like a contradiction of §2 and is not. §2's lesson concerns arms whose
*feature geometry* differs, where a transplanted ladder is a handicap. These two
arms are the same module with one switch flipped — identical geometry, width and
parameter count — so letting them sit at different L2 would introduce a second
difference and the gap would stop isolating coupling. Equal budget means equal
value when the geometry is equal.

---

## 4. New result: a positional cross-encoding bounds the cheap alternatives

[`scripts/stage3f_blosum_cross.py`](../scripts/stage3f_blosum_cross.py) — the
9 × 34 matrix of BLOSUM62 scores between peptide position *i* and contact
residue *j*. 306 columns, no network, no pretrained model, 0.23 s to compute.
These columns carry **no information the baseline lacks**: they are a
deterministic bilinear function of the same residues the one-hot block already
receives, which `tests/test_biology.py` pins. So a gain could only mean the MLP
was failing to extract the interaction from one-hot input — a finding about
architecture, not representation.

| arm | width | median ρ | Δ vs baseline 0.6931 | 95% CI | verdict |
|---|---:|---:|---:|---|---|
| `cross_alone` | 306 | 0.4121 | −0.2811 | [−0.3379, −0.2064] | worse |
| `cross_stacked` | 1,166 | 0.6099 | **−0.0832** | [−0.1084, −0.0235] | **worse** |

Both arms selected their own L2 on a 5-point ladder, both interior (1e-3 and
1e-2). A positional cross-encoding alone is not a model, and stacked it makes
the baseline conclusively worse. **That bounds every cross-feature scheme,
ESM-2's included, for ten CPU-minutes**: whatever a fixed bilinear re-encoding
of these residues can express, the baseline already extracts.

### The matched-width comparison, and the claim it supports

`cross_stacked` at 1,166 columns is within 2.0% of the shipped additive arm's
1,190, with the same `{onehot, blosum}` encoding axis, the same 30 members, the
same folds, head class and seeds.

**`esm_plus_seq_ensemble` beats `cross_stacked` by +0.0662 [+0.0077, +0.1077],
CI entirely above zero.**

The claim this supports is narrow and should be stated narrowly: *ESM-2's 330
columns are more useful on top of this baseline than 306 columns of BLOSUM
positional cross-encoding.* It says nothing about what pure width costs — see
§5, which is where an earlier, wider version of this claim died.

One asymmetry, in the conservative direction: `cross_stacked` received a 5-point
ladder against the additive arm's 3, so the control got *more* search than the
arm that beats it, and +0.0662 is if anything understated.

---

## 5. A dead end, recorded: uninformative blocks are not capacity controls

The obvious next step after §4 is to ask whether ~330 columns cost the baseline
−0.08 *regardless of content*, which would make the additive arm's −0.017 a
mostly-recovered displacement cost. [`scripts/stage3g_displacement.py`](../scripts/stage3g_displacement.py)
builds the two controls that would answer it at exactly 1,190 columns, through
`main`'s own `_fit_block` so the path is identical: a random Gaussian block, and
the real ESM block with the sequence-to-embedding correspondence permuted over
unique peptides and unique alleles.

**It does not work, and the reason is the finding.** Single network, fold 0,
seed 0, l2 = 1e-2 (the shipped arm's selected value):

| arm on the baseline | width | median ρ | best epoch |
|---|---:|---:|---:|
| baseline only | 860 | 0.5833 | 27 |
| + real ESM-2 | 1,190 | 0.5912 | 86 |
| + BLOSUM cross | 1,166 | 0.5668 | 22 |
| + shuffled ESM-2 | 1,190 | **0.0100** | 11 |
| + shuffled, unit-scaled | 1,190 | **0.0755** | 1 |
| + random Gaussian | 1,190 | **0.1281** | 3 |

([stage3g_diagnostic.csv](stage3g_diagnostic.csv). **Read these to one decimal,
not four.** They are single networks, and re-running the same configuration
under a different BLAS thread count moved the healthy arms by ~0.02 and the
collapsed ones by ~0.03 — float32 reduction order, not seed. The pattern is
what carries; the digits do not. The arms that matter to a published number are
all 30-member ensembles with bootstrap intervals.)

The uninformative blocks do not cost 0.08. They cost ~0.5 and take the arm to
near zero, which is nowhere in the displacement regime.

**The mechanism.** A width-matched uninformative block is not an inert consumer
of capacity here — it is a **memorisation channel that breaks dev-fold early
stopping**. 330 columns that uniquely key the peptide (a permutation over 5,633
distinct peptides still *identifies* the peptide; it destroys the geometry, not
the identifiability) or the row (random) let the network drive training loss down
without learning anything that generalises. Dev loss bottoms at epoch 1–5
instead of 27, and stopping fires before the baseline signal is learned.

And it is not only truncation, which is the part that separates two otherwise
confounded explanations. Truncating the **baseline alone** to the same epoch:

| baseline-only, max epochs | 1 | 2 | 3 | 5 | 10 | 27 |
|---|---:|---:|---:|---:|---:|---:|
| median ρ | 0.2175 | 0.3567 | 0.4460 | 0.5210 | 0.5578 | 0.5833 |

Baseline alone stopped at epoch 5 still reaches **0.5210**; the shuffled arm
stops at a comparable epoch and reaches **0.01–0.05**. So the block both terminates stopping
early *and* dominates the fitted function out of sample. Unit-scaling does not
rescue it, so scale is a contributor and not the cause — though the scales are
worth recording, since mean per-column standard deviation is **0.084** for the
baseline's sparse one-hot, 0.411 for the BLOSUM cross block, 1.000 for random,
and **4.788 (max 54.5)** for the real and shuffled ESM blocks.

**What survives.** The `shuffled` arm remains a valid *sign* test, and a strong
one: real ESM-2 at 0.5679 against its own permutation at 0.0453, with identical
width, scale, marginals, covariance and rank, the only difference being which
embedding belongs to which sequence. **The ESM block is not inert padding.** But
because the permuted arm collapses rather than degrades, the size of that gap
measures pipeline fragility, not information content, and must not be quoted as
an effect size.

**What does not survive.** Any statement about what pure width costs. No healthy
uninformative arm exists to anchor it, so capacity displacement stays exactly
what [stage3_esm.md](stage3_esm.md) already calls it — a candidate mechanism,
unsettled. Separating capacity from stopping would need a protocol in which no
arm's model is dev-selected, at a fixed epoch count for every arm. That is not
run here.

This file is kept, with its `diagnose` mode, so the dead end is reproducible
rather than rediscovered.

### Closed: the additive arm's scale disparity is not a handicap

§5 raises an obvious worry about the arm being shipped. A ~57× per-column scale
disparity sits inside it — 0.084 on the baseline's sparse one-hot side against
4.788 on the ESM side — which is the expected consequence of appending PCA
components, since component magnitude tracks explained variance. If this
pipeline is fragile to a dominant appended block, is the additive arm's −0.017
partly that rather than an information result?

`stage3g_displacement.py scaleprobe` answers it: **no, and rescaling makes
things worse.** Both variants on the same 3-point ladder, 2 folds × 2 seeds per
point, single-network protocol ([stage3g_scale_probe.csv](stage3g_scale_probe.csv)):

| variant | best L2 | median ρ | vs baseline-only |
|---|---:|---:|---:|
| baseline only | 1e-3 | 0.6109 | — |
| ESM block **as shipped** | 1e-2 | 0.5743 | −0.0365 |
| ESM block **rescaled to unit variance** | 1e-2 | 0.5049 | **−0.1060** |

Flattening the components to unit variance costs a further ~0.07. That is the
right direction on reflection: PCA component scale *is* information — it says
how much variance a direction carries — and unit-rescaling gives the 256th
component the same weight as the 1st. So the shipped scaling is not an accident
the arm survives; it is better than the obvious alternative, and the −0.017 is
not a scaling artifact.

This is a probe, not a re-selection: the shipped arm's published number stands
as published. Re-tuning it after seeing this would be exactly the post-hoc
selection the frozen contract exists to prevent. Single-network deltas, no
bootstrap — read the sign and whether the variants differ by more than the
0.035–0.069 seed spread, which they do.

---

## 6. What is left open

- **What pure width costs** — §5.
- **Whether coupling helps at a scale this dataset cannot support.** The
  attention arm has 94,913 parameters against 19,716 training rows. A null at
  this scale does not bound a larger one.
- **The paid arm.** The branch re-scored an independent implementation's LLM
  arm — an allele-prompted model given 80 in-context training examples —
  at median ρ **0.5346 for $12.19 and 361 s**, CI entirely below zero. It is
  the only arm in either tree with a dollar cost attached to an accuracy number,
  and for a track that grades compute, a paid arm losing to free one-hot
  features is a real Pareto point. Not reproduced here; the branch's
  `stage3_devin_rescored.csv` carries it, with the caveat that those arms are
  3-seed means rather than 30-network ensembles and early-stop on the split they
  report.

## Reproducing

```bash
.venv/bin/python scripts/biology_probes.py
.venv/bin/python scripts/stage3e_coupling.py smoke
.venv/bin/python scripts/stage3e_coupling.py ladder
.venv/bin/python scripts/stage3e_coupling.py ladder --l2-grid 1e-6 1e-5   # extend on a boundary hit
.venv/bin/python scripts/stage3e_coupling.py ensemble --l2 1e-3
.venv/bin/python scripts/stage3f_blosum_cross.py run
.venv/bin/python scripts/stage3g_displacement.py diagnose
.venv/bin/python -m pytest tests/test_biology.py -q
```

Validation only. The test split is never loaded by any file listed here. All
figures measured on CPU, 8 cores, 16 GB shared with concurrent work, **$0 of
cloud spend**.
