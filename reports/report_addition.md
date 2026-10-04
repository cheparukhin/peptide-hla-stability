# Report addition — the two experiments that bound the stage 3 null

Slide source. These are the **most important** next steps because they attack
the single largest hole in this study's headline negative: stage 3 concluded
that frozen ESM-2 does not beat a sequence baseline, but it only ever tested
**peptide and HLA encoded separately and concatenated into a small MLP**. The
model was never given a mechanism — at encoding time or downstream — to relate
a peptide position to a groove position.

Both experiments were already identified in
[../HACKATHON_PLAN.md](../HACKATHON_PLAN.md) and **deferred for time, not
rejected on scientific grounds**. Neither needs structures, and neither needs
new data.

---

## Slide: what the foundation-model null does not yet cover

| Proposed experiment | Slide explanation |
|---|---|
| **ESM-C with concatenated sequences** | Feed peptide–linker–MHC α1α2 into ESM-C as **one sequence**, so each partner is represented in the context of the other. Compare against encoding them separately. |
| **Separate embeddings + cross-attention** | Keep peptide and MHC embeddings separate, then add a **trainable cross-attention module** connecting individual peptide positions with groove positions. Compare against concatenated embeddings + a small MLP, **keeping the embeddings unchanged**. |

**Why these two, and why together.** They separate *representation* from
*architecture* — the confound that makes a single combined test uninterpretable:

- **ESM-C changes the representation.** Coupling happens inside a pretrained
  trunk, during encoding, and the model family changes at the same time.
- **Cross-attention changes only the architecture.** Embeddings are held
  fixed, so any gain is attributable to the interaction module alone.

Run only one and a positive result cannot be assigned to either cause. Run both
and the attribution is clean.

---

## What each one answers

### 1. ESM-C on a chimeric peptide–linker–α1α2 sequence

Stage 3 embedded the two chains separately, so ESM-2 never saw the complex. The
plan calls the chimeric input **"the sharpest *untested* version of the
question"** and bounds the stage 3 null to separate embeddings because of it.
ESM-C additionally answers whether that null is **family-specific** — one pLM
family is a thin basis for "foundation models don't help here".

Both halves were promoted on 4 October and neither ran: the chimeric input
"did not reach the clock", and ESM-C was **gated on RAM** (six workstreams
sharing 16 GB — the same constraint that excluded ESM-2 650M). An engineering
constraint, not a modelling choice.

**Carry the standing objection onto the slide:** a linkered 9-mer sits far
outside a pLM's training distribution, so this is a **labelled diagnostic, not
a headline arm**. Sweep layers rather than assuming the final one.

### 2. Cross-attention over fixed embeddings

From the plan's own description of stage 3: *"Embedding peptide and HLA
separately means **the prediction head has to learn peptide-HLA interactions on
its own**, so useful performance is a hypothesis, not a guarantee."*

This is the direct test of that hypothesis. A concatenating MLP must infer
which peptide position contacts which groove residue from pooled vectors, with
no architectural mechanism for it; cross-attention supplies exactly that
mechanism — peptide positions as queries, the 34 contact positions as keys —
and leaves the embeddings untouched.

It is arguably the **safer** of the two: it tests coupling without pushing any
sequence outside the pretrained model's distribution, so it carries no
distribution-shift objection. It is deferred in the register alongside
folding-trunk features, with no argument recorded against it.

---

## Protocol both arms must follow

Non-negotiable, because stage 3 showed what happens otherwise:

- **Comparator:** the 30-network pseudosequence ensemble at **0.693** — not the
  full-domain arm. The one conclusive stage 3 result is about *full-domain
  one-hot being weak*, so beating it establishes nothing.
- **Tuning parity is equal *budget*, not equal values.** Transplanting stage
  2's regularisation ladder cost the ESM arm **0.109 SCC** and would have
  shipped a confident false negative. Extend each ladder until its selection is
  **interior**; the tell is seed spread outside the observed 0.010–0.051.
- Same 2,817 validation rows, 68 eligible alleles, 30 networks under
  `cv_folds()`, paired cluster bootstrap at 2,000 resamples.
- **Hold embeddings fixed** in the cross-attention comparison, or the coupling
  gain and the representation gain are inseparable.
- Report against the predeclared **0.05** bar in
  [../EVALUATION.md](../EVALUATION.md), with cost per 1,000 predictions — the
  baseline costs $9.6 × 10⁻⁷ and ESM-2 $4.7 × 10⁻⁴, so an expensive arm must
  clear the bar to have earned its place.

**Expected outcome is honest uncertainty.** Stage 3's null was *inconclusive at
zero while ruling out 0.05*, so the plausible result here is another bounded
null — which is still worth having, because it would close the hole rather than
leave the headline negative resting on one architecture and one model family.

> **Every number here is a validation number.** The frozen test split is scored
> once, at stage 6, under [TEST_SCORING_RUNBOOK.md](TEST_SCORING_RUNBOOK.md).

**Sources.** Stage 3 design, the 0.109 tuning finding and the null's stated
bounds: [../HACKATHON_PLAN.md](../HACKATHON_PLAN.md) §3 and
[stage3_esm.md](stage3_esm.md). Deferral status for cross-attention and ESM-C:
the out-of-scope register, [../HACKATHON_PLAN.md](../HACKATHON_PLAN.md).
Baseline 0.693 and costs: [stage2_baselines.md](stage2_baselines.md),
[SUBMISSION.md](SUBMISSION.md).
