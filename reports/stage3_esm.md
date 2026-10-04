# Stage 3 — frozen ESM-2 representations

**Status: complete at ESM-2 35M and 150M. Result: inconclusive at zero, and
rules out the predeclared worthwhile gain.** Frozen ESM-2 features neither
replace nor improve on the stage 2 sequence baseline on validation. Every
matched interval crosses zero — so this is *not* a demonstration that ESM-2 is
worse — but every interval's upper bound sits below the predeclared
+0.05 bar, which is the strongest negative form this evaluation supports.

Validation only. The test split was never loaded; `tests/test_esm.py` asserts
the string `"test"` does not appear in `scripts/esm_arm.py`.

Reproduce:

```
.venv/bin/python scripts/esm_features.py --verify-index
.venv/bin/python scripts/esm_features.py --checkpoint esm2_t12_35M_UR50D
.venv/bin/python scripts/esm_arm.py sweep     --checkpoint esm2_t12_35M_UR50D
.venv/bin/python scripts/esm_arm.py grid      --checkpoint esm2_t12_35M_UR50D \
    --pep-rep pos --hla-rep contact --layers mid final
.venv/bin/python scripts/esm_arm.py ensemble  --checkpoint esm2_t12_35M_UR50D \
    --pep-rep pos --hla-rep contact --axis layer --layers mid final --name esm_ensemble
```

Artifacts: `reports/stage3_runs.csv` (every run, 180 rows),
`reports/stage3_comparisons.csv` (the matched table),
`reports/stage3_tuning_sensitivity.csv`, `reports/stage3_embedding_cost.csv`,
`reports/stage3_contact_index.json`, `reports/stage3_headline.json`,
`preds/esm_ensemble.csv`, `preds/esm_plus_seq_ensemble.csv`,
`pepstab/esm.py`, `scripts/esm_features.py`, `scripts/esm_arm.py`,
`tests/test_esm.py` (21 guards).

---

## 1. The headline comparison

All arms: **30 networks**, 5 CV folds x a 2-way representation axis x 3 seeds,
on the **same 2,817 validation rows** and the **same 68 eligible alleles**,
with fold assignment imported directly from `scripts/baseline_ensemble.py`
(`cv_folds`) rather than reimplemented. Primary metric is median per-allele
Spearman; uncertainty is the frozen paired cluster bootstrap (2,000 resamples,
seed 20261003).

| Arm | median ρ | IQR | Δ vs sequence baseline | 95% CI | Verdict |
|---|---:|---|---:|---|---|
| **sequence baseline** (stage 2, pep+pseudoseq) | **0.6931** | 0.574–0.753 | — | — | reference |
| ESM-2 only (pep per-position + 34-contact) | 0.6830 | 0.559–0.768 | **−0.0101** | [−0.0382, +0.0352] | inconclusive at 0, rules out +0.05 |
| sequence **+** ESM-2 (additive) | 0.6761 | 0.560–0.770 | **−0.0170** | [−0.0468, +0.0320] | inconclusive at 0, rules out +0.05 |

Against the re-tuned baseline (section 4), the same two arms give −0.0045
[−0.0362, +0.0378] and −0.0115 [−0.0438, +0.0353]. Same verdict either way.

**Reading this.** Both intervals cross zero, so the honest statement is that
this evaluation cannot separate frozen ESM-2 from the sequence baseline in
either direction. What it *can* do is exclude the gain that would have made the
extra machinery worth adopting: the upper bound on the best ESM arm is +0.035,
below the +0.05 predeclared at stage 1. The two arms answer different
questions and agree:

- **Can frozen ESM-2 features replace sequence features?** Essentially yes, at
  parity — 0.683 against 0.693, from a 480-dimensional frozen representation
  the model never saw stability labels for. That is a more interesting result
  than it looks, and it is the one the ESM-only arm establishes.
- **Do they add anything on top?** No. Appending ESM-2 to the baseline's own
  features moves the result *down* by 0.017, well inside noise. Whatever
  ESM-2 encodes about these peptides, the baseline has already extracted it
  from 19,716 labelled rows.

### The one comparison that is not a null

The plan requires that domain *embeddings* be checked against a raw
full-domain sequence baseline, so that "more input sequence" is not mistaken
for a benefit of pretraining (EVALUATION.md, Model selection).

| Comparison | Δ median ρ | 95% CI | Verdict |
|---|---:|---|---|
| ESM-2 only vs **full-domain** sequence ensemble (0.6529) | **+0.0301** | [−0.0084, +0.0757] | inconclusive, CI crosses 0 |

On the matching input — the full 182-residue HLA domain — the ESM-2
representation is ahead by +0.030 and the interval still admits a worthwhile
gain. This is the only arm here whose upper bound exceeds +0.05, and it is
inconclusive rather than positive. It says the pretrained representation of a
domain is a better use of that domain than one-hot encoding it, which is a
weaker and narrower claim than the headline, and it is not enough to overturn
it: the baseline's own best configuration does not use the full domain.

---

## 2. What the tuning range cost, and why it is a result

**This section exists because getting it wrong would have produced a confident,
wrong negative result**, and the mistake is the one the challenge's framing
invites: hold the baseline's hyperparameters fixed "for fairness" and vary only
the features.

Tuning parity has to mean **equal budget**, not equal values. Stage 2's L2
ladder was `{1e-5, 1e-3}`, chosen for sparse one-hot features over 860
dimensions. The ESM features are dense standardised components. Transplanting
the ladder is a handicap wearing the costume of fairness:

| Arm | ρ at stage 2's L2 (1e-5) | seed spread | ρ at its own boundary-checked L2 | seed spread | cost of the transplant |
|---|---:|---:|---:|---:|---:|
| ESM-2 only, middle layer | 0.4994 | 0.110 | **0.6081** (L2=1e-2) | 0.013 | **−0.109** |
| ESM-2 only, final layer | 0.4687 | 0.169 | **0.5970** (L2=1e-2) | 0.032 | **−0.128** |
| baseline pep+pseudoseq, one-hot | 0.6096 | 0.032 | 0.6116 (L2=1e-2) | 0.044 | −0.002 |
| baseline pep+pseudoseq, BLOSUM62 | 0.5946 | 0.009 | 0.5946 (L2=1e-5) | 0.009 | 0.000 |

The ESM arm is **roughly 50x more sensitive to the regularisation range than
the baseline is**. Reported at stage 2's setting, frozen ESM-2 would have come
in 0.11 behind the baseline and the write-up would have said so with a
straight face. The gap was the hyperparameter range, not the features.

**The seed spread is the tell.** At L2=1e-5 the ESM arm's three seeds ranged
over 0.110 — far outside stage 2's observed 0.010–0.051, which is how the
problem announced itself before any comparison was made. At the selected
L2=1e-2 the spread collapses to 0.013. An effect smaller than seed noise is
not an effect; an *instability* larger than seed noise is usually
under-regularisation, and here it was.

**Boundary checks, applied symmetrically.** A configuration selected at the
edge of its ladder means the ladder was truncated and the selection was never
really made. Every arm here was extended until its selection was interior,
including the baseline — otherwise the fix for one handicap just installs the
opposite one.

| Arm | ladder | points | selected | interior? |
|---|---|---:|---:|---|
| ESM-2 only, middle layer | 1e-5, 1e-3, 1e-2, 1e-1, 1, 10 | 6 | 1e-2 | yes |
| ESM-2 only, final layer | 1e-5, 1e-3, 1e-2, 1e-1, 1, 10 | 6 | 1e-2 | yes |
| baseline, one-hot | 1e-7, 1e-6, 1e-5, 1e-2, 1e-1 | 5 | 1e-2 | yes |
| baseline, BLOSUM62 | 1e-7, 1e-6, 1e-5, 1e-2, 1e-1 | 5 | 1e-5 | yes |
| additive, one-hot + ESM | 1e-3, 1e-2, 1e-1 | 3 | 1e-2 | yes |
| additive, BLOSUM + ESM | 1e-3, 1e-2, 1e-1 | 3 | 1e-2 | yes |
| ridge (all ESM arms) | 10 … 1e5 | 5 | 1e4 | yes |

Stage 2's own selection (`stage2_headline.json`: L2=1e-5, grid `[1e-5, 1e-3]`)
sat at the **bottom edge** of its ladder, so it was edge-selected by exactly
the standard applied here. Extending it downward (1e-6, 1e-7) and upward
(1e-2, 1e-1) moves the single network by +0.002 and the 30-network ensemble by
**−0.0055 [−0.0299, +0.0264]** — no material change, so stage 2's published
ensemble is retained as the reference and nothing ripples into that report.

Honest note on budget: the ESM ladders ended up with 6 points against the
baseline's 5 and the additive arm's 3. The additive arm — the one whose
conclusion matters most — therefore received the *smallest* search, which is
conservative against the foundation-model case rather than for it. Its
selection is interior with both flanks falling away, so a wider ladder would
not change which point wins.

---

## 3. Residue indexing: verified, not assumed

The 34 "contact" embeddings are only contact-residue embeddings if the index
into the 182-residue `hla_seq` is right, and a wrong index produces a
plausible-looking feature block and a plausible-looking score. So it is
checked, and the check is itself tested.

**Taking the canonical NetMHCpan positions out of `hla_seq` reproduces the
supplied `hla_pseudoseq` character-for-character for all 75 unique HLA
domains** — 0 mismatched alleles (`reports/stage3_contact_index.json`).

Positions (1-based): 7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77,
80, 81, 84, 95, 97, 99, 114, 116, 118, 143, 147, 150, 152, 156, 158, 159, 163,
167, 171.

**Four positions are not determined by the data alone.** Pseudosequence
positions 1, 18, 24 and 31 carry a tyrosine that is invariant across all 75
alleles, so seven columns of `hla_seq` (7, 27, 84, 85, 118, 123, 159) match
them equally well. They are resolved by the canonical numbering, not by the
reconstruction. This is harmless for a *sequence* check — every candidate is
the same letter — but **not** harmless for a position-dependent embedding,
where column 27 and column 84 carry different contextual vectors. Four of 34
contact embeddings therefore rest on the published numbering rather than on
evidence internal to this dataset. Recorded rather than glossed.

`tests/test_esm.py` includes a negative control: shifting every index by one
must fail the verification, so the check cannot pass vacuously.

---

## 4. Representation sweep

Ten combinations, one shared config and 3 seeds each, single-network
`inner_folds()` protocol. Run at stage 2's L2 (1e-5) — before section 2's
finding — so the absolute numbers are depressed; the *ranking* is what this
table is for, and it is consistent and large compared with the seed spreads.

| peptide | HLA | layer | MLP ρ (mean of 3 seeds) | seed spread | ridge ρ |
|---|---|---|---:|---:|---:|
| per-position | 34-contact | **middle** | **0.4994** | 0.110 | 0.284 |
| per-position | 34-contact | final | 0.4687 | 0.169 | 0.276 |
| mean-pooled | 34-contact | middle | 0.3740 | 0.077 | 0.172 |
| mean-pooled | 34-contact | final | 0.3259 | 0.046 | 0.162 |
| per-position | mean-pooled | final | 0.2699 | 0.179 | 0.274 |
| mean-pooled | mean-pooled | middle | 0.2486 | 0.069 | 0.174 |
| mean-pooled | mean-pooled | final | 0.1986 | 0.029 | 0.166 |
| per-position | mean-pooled | middle | 0.1768 | 0.082 | 0.277 |
| per-position | none | middle | 0.1688 | 0.055 | 0.162 |
| per-position | none | final | 0.1484 | 0.067 | 0.186 |

Selected on validation: **peptide per-position + HLA 34-contact + middle
layer**. Three readings:

- **Keeping the 9 peptide positions beats mean-pooling them** (0.499 vs 0.374
  on the best HLA arm), which is what stage 2 found for one-hot encodings too.
  Which position an anchor residue sits at is most of the signal, and pooling
  destroys it.
- **The middle layer beats the final layer** at every matched setting. ESM-2's
  last block is specialised toward the masked-token objective; the middle of
  the stack transfers better. Consistent with the wider pLM literature, and it
  is why the plan asked for the comparison.
- **Ridge is far below the MLP everywhere** (0.28 at best against 0.61). As at
  stage 2, nonlinearity is most of the model: a linear head cannot represent
  the interaction between a peptide residue and the pocket it sits in, and
  that is true of ESM features as much as of one-hot ones.

### Where you read the HLA embedding matters more than what produced it

The mean-pooled HLA arm scores 0.177–0.270 against the 34-contact arm's
0.469–0.499. The natural explanation — pooling over 182 residues throws
information away — is **wrong here**, and the dataset lets us prove it.

There are only **75 distinct HLA domain sequences** in the whole dataset, so
*any* HLA representation has rank ≤ 74 no matter how many columns it occupies.
Measured on the real cache:

| HLA representation | raw columns | rank (>1e-4) | variance in 74 components | all 75 alleles distinct? | min pairwise distance |
|---|---:|---:|---:|---|---:|
| mean-pooled | 480 | 74 | 1.000000 | yes | 0.273 |
| 34-contact | 16,320 | 74 | 1.000000 | yes | 8.021 |

Both are **losslessly** representable in 74 dimensions, and both separate all
75 alleles exactly. They carry *identical information* about which allele a row
belongs to — an unconstrained model could not tell them apart. The entire gap
is the **similarity geometry** the representation induces: contact-position
embeddings place alleles with similar binding pockets near each other, while
mean-pooled embeddings place alleles with similar overall domain sequence near
each other, and only the first is the right prior for a peptide-binding head
generalising across alleles.

With 75 distinct HLA sequences, choosing *where in the protein to read the
embedding* is the whole decision; choosing the model that produced it is
secondary. That is a cheap, well-controlled methodological finding and it
generalises past this dataset.

---

## 5. Checkpoint size

Started at 35M as the plan directs, and checked 150M because a negative result
at the smallest checkpoint alone is a weak claim.

| checkpoint | best single-network ρ (3 seeds) | 30-network ensemble ρ | Δ vs sequence baseline | 95% CI | extraction |
|---|---:|---:|---:|---|---:|
| **ESM-2 35M** | **0.6081** (middle layer) | **0.6830** | −0.0101 | [−0.0382, +0.0352] | 7.5 s |
| ESM-2 150M | 0.5878 (final layer) | 0.6737 | −0.0194 | [−0.0599, +0.0195] | 19.5 s |
| ESM-2 650M | not run — excluded, see below | — | — | — | (59.5 s, cached) |

**Scaling up did not help.** 150M is 2.6x the extraction cost of 35M and
scores marginally *lower* at both the single-network and ensemble level, and
its interval also excludes the +0.05 bar. Two checkpoints is a short scaling
curve, but it is flat over the range tested, and at 35M the arm is already at
parity with the baseline — so there is no trend suggesting a larger model
would close a gap, because at 35M there is no gap to close.

Per-layer detail at 150M (ladder `1e-3, 1e-2, 1e-1, 1`, all interior): middle
0.5725, final 0.5878. Note the layer preference **reverses** relative to 35M,
where the middle layer won; with seed spreads of 0.015-0.021 that reversal is
marginal and should not be read as a finding.

**650M was cached but excluded from the comparison for resource reasons.**
Its embeddings are on disk (330 MB, extracted in 59.5 s), but the machine runs
six workstreams in 16 GB and extraction alone peaked at 5.06 GB resident while
a $200 structural production run had to survive on the same host. Reported as
an engineering constraint, not as a gap: the challenge grades compute
requirements, and "the largest checkpoint did not fit alongside concurrent
work" is a measurement about deployability. It does bound the conclusion —
see section 8.

---

## 6. Dimensionality reduction, and what it cost

Both feature blocks are standardised (fitted on the unique sequences appearing
in the **training split** only — unsupervised, and never touching validation),
then reduced by PCA.

| block | raw columns | components | variance retained | lossy? |
|---|---:|---:|---:|---|
| HLA (34-contact or mean-pooled) | 16,320 / 480 | 74 | **1.000000** | no — rank-limited by 75 distinct sequences |
| peptide per-position | 4,320 | 256 | **0.9547** | yes — a declared resource decision |

The HLA reduction is lossless by construction. The peptide reduction is not,
and it was adopted because the full block is 346 MB for the training split
alone (before any float64 copy a solver makes) on a machine under memory
pressure. A lossy reduction taken for engineering reasons is only legitimate
if its cost is measured, so it was:

| peptide block | best L2 | median ρ (3 seeds) |
|---|---:|---:|
| PCA, 256 components | 1e-2 | **0.6081** |
| uncompressed, 4,320 columns | 1e-5 | 0.5746 |

Both selections are interior; the uncompressed arm was given a 7-point ladder
(`1e-7 … 1e-1`) against the PCA arm's 6, so it was not short-changed on search.

**The reduction did not handicap the arm — it helped it, by +0.033.** Keeping
all 4,320 peptide columns is worse, not better, and noisier (seed spread 0.067
against 0.013). That is the expected direction: 4,320 dense correlated inputs
against 17,744 training rows is a harder estimation problem than 256
components carrying 95.5% of the variance, and the extra 4.5% is apparently
not where the stability signal lives. So the memory-driven choice costs
nothing scientifically, and the negative result cannot be attributed to it.
The uncompressed block also runs ~6x slower per network (23-101 s against
4-11 s).

---

## 7. Measured compute cost

The challenge grades engineering and compute requirements, so cost is a
deliverable rather than a footnote. All figures measured on this machine
(Apple silicon, 8 cores, 16 GB, no CUDA), **$0 of cloud spend**.

### Embedding extraction — once per unique sequence, not per row

28,166 measurement rows carry only **5,633 distinct peptides and 75 distinct
HLA domains**, so the cache is keyed by sequence content hash and fanned out to
rows at feature-build time. That is a ~5x saving on the peptide side and ~375x
on the HLA side.

| checkpoint | dim | load | embed (5,708 seqs) | s / 1,000 peptides | peak RSS | cache on disk |
|---|---:|---:|---:|---:|---:|---:|
| ESM-2 35M | 480 | 5.1 s | **7.5 s** | 1.07 | 859 MB | 123 MB |
| ESM-2 150M | 640 | 16.5 s | 19.5 s | 2.74 | 1.48 GB | 165 MB |
| ESM-2 650M | 1280 | 73.1 s | 59.5 s | 8.50 | 5.06 GB | 330 MB |

Extraction runs on MPS, two layers per forward pass, float16 on disk (cast to
float32 on load). Install cost: `torch` + `fair-esm` added **582 MB** to the
venv.

### Head training and inference

| arm | networks | total fit | feature assembly / 1k rows | **inference / 1k rows** |
|---|---:|---:|---:|---:|
| sequence baseline (860 features) | 30 | 4.1 min | 0.0065 s | **0.0656 s** |
| ESM-2 only (330 features) | 30 | **2.2 min** | 0.0500 s | **0.0150 s** |
| sequence + ESM-2 (1,190 features) | 30 | 3.7 min | 0.0529 s | 0.0335 s |

The ESM head is *cheaper* to fit and to run than the baseline's, because PCA
leaves it with 330 inputs against the baseline's 860. The cost is upstream.

### Cost per 1,000 new predictions, end to end

| | sequence baseline | ESM-2 arm |
|---|---:|---:|
| embed 1,000 new peptides | — | 1.07 s |
| feature assembly | 0.007 s | 0.050 s |
| 30-network ensemble inference | 0.066 s | 0.015 s |
| **total** | **0.07 s** | **1.14 s** |

**About 16x more expensive per 1,000 predictions, for no measurable accuracy
gain.** Both are negligible in absolute terms — a 9-mer is a short sequence and
35M is a small model — so the honest framing is that cost is not what rules
ESM-2 out here; the absence of a gain is. The ratio would matter at a larger
checkpoint: 650M is 8x the per-sequence extraction cost of 35M.

### A measurement worth passing on

Two engineering findings from this stage, both of which changed wall-clock by
more than an order of magnitude:

- **`sklearn.linear_model.Ridge` rebuilds `X.T @ X` on every call.** Sweeping
  5 alphas on a 17,744 x 4,394 block spent ~12 minutes recomputing one matrix
  six times, single-threaded in float64. Forming the Gram matrix once
  (chunked, to avoid a 600 MB float64 copy) and doing a Cholesky solve per
  alpha gives the same answers — asserted equal to sklearn to 1e-4 in
  `tests/test_esm.py` — in **0.2 s**.
- **BLAS thread oversubscription dominated everything for a while.** Six
  workstreams each defaulting to one thread per core produced a load average
  of **114 on 8 cores** at 631% total CPU. Pinning the pool to one thread
  *before* importing numpy (`VECLIB_MAXIMUM_THREADS` is the one that matters
  on macOS, since numpy here links against Accelerate, and the pool is sized
  at import) brought the load average to ~4 and made every process faster.

---

## 8. What this does and does not establish

**Bounded to what was tested.** This is a result about *frozen* ESM-2
representations of the peptide and the HLA domain **embedded separately**, at
35M and 150M, with ridge and a small MLP head, under the frozen splits, on
validation.

It does **not** establish:

- that fine-tuned ESM-2 would not help — nothing here was fine-tuned;
- that a model shown the *complex* would not help. Peptide and HLA are
  embedded independently, so the head has to learn the peptide–HLA interaction
  itself from 19,716 rows. The plan predicted this would be the binding
  constraint, and the ESM-only arm reaching parity while adding nothing on top
  is consistent with it. The chimeric peptide-linker-groove diagnostic is the
  queued follow-up;
- that log-likelihood or perplexity features would not help — embeddings are
  only one of the three ways the brief names of using a foundation model
  (stage 3d covers likelihoods);
- that a larger checkpoint would not help. 150M did not change the picture
  (section 5), and 650M was excluded for memory. The trend across the two
  sizes tested is flat, which weakens but does not close the scaling argument;
- that another pLM family would behave the same way. One family is a thin
  basis for a general claim; a second is queued.

**Also worth stating plainly:** the comparison is against a strong baseline.
The reference is a 30-network ensemble at median per-allele ρ 0.693, not the
single network at 0.610. Ensembling alone is worth about +0.09 in this setup,
so an un-ensembled ESM arm against it would have manufactured exactly the
negative result reported here. Both arms are ensembled identically — same
folds (imported, not reimplemented), same seeds, same member count, same head
class — and the harness was validated by reproducing
`preds/seq_ensemble_pep_pseudo.csv` through it: **Δ = +0.0000 [+0.0000,
+0.0000]**, median ρ 0.6931, bit-identical to stage 2's published artifact.

Standard caveats from EVALUATION.md apply unchanged: the 20.2% floor is
left-censored, there are no replicates so no noise ceiling, and the assay panel
was partly selected by predicted affinity.
