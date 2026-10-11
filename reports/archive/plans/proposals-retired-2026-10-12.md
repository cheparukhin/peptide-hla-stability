# Retired proposal lists

Archived 12 October 2026. **Historical only.** The [current research plan](../../NEXT_RESEARCH_PLAN.md) is the sole work queue and contains the final disposition of every proposal below. Numerical and methodological claims in these excerpts may be superseded by its correction register.

These excerpts preserve the competing lists removed from the project README and the completed-study findings. The former website table and submission-film closing proposals matched the README list; the film remains a historical artifact.

## Former project README proposals

## Next steps

| Proposed experiment | Explanation |
|---|---|
| **ESM-C with concatenated sequences** | Feed peptide–linker–MHC α1α2 into ESM-C as one sequence, so each partner is represented in the context of the other. Compare against encoding them separately. |
| **Separate embeddings + cross-attention** | Keep peptide and MHC embeddings separate, then add a trainable cross-attention module connecting individual peptide positions with groove positions. Compare against concatenated embeddings + a small MLP, keeping the embeddings unchanged. |
| **Inverse-folding and energy features** | Score each predicted complex with ProteinMPNN peptide likelihoods and FoldX interaction energies, and test them as half-life features. Compare against the sequence ensemble, keeping the same folds and tuning budget. |


## Former findings recommendations (5 October)

## 6. What to do next

Ordered by value. These are the next steps the phases themselves identified; this
section does not introduce new directions.

1. **Acquire independent measurements, with a replicate subset
   ([FURTHER_RESEARCH_PLAN.md](further-research-2026-10-04.md) §8).** This outranks any
   further modelling. The study's convergent result is that the effects on offer are
   real but ~0.02, against a fold spread of ~0.05 and a run-to-run floor of 0.0043
   the intervals cannot see — so the binding constraint is measurement precision, not
   model choice. A replicate subset is the only thing that would establish the
   assay's own noise ceiling; the study has none, so "recoverable headroom" cannot be
   estimated from this data (concordance between distinct alleles is not replicate
   reliability).

2. **Replicate A4 and A1 on CatBoost (~$2).** The run-to-run reproducibility
   floor of 0.0043 was measured on the MLP arms and is **unmeasured for CatBoost**,
   which does not share their BLAS paths. A4's +0.0118 margin over the tree alone
   sits close enough to that figure to matter, and the whole study now rests partly
   on a tree arm whose own reproducibility has never been checked. Two replicates of
   the same config are the cheapest thing on this list and the only one that tests a
   number the study depends on. *(This reverses an earlier judgement in this study
   that replication bought nothing — which held only while every effect was
   indistinguishable from zero. It no longer is.)*

3. **Finish the one bounded CPU item still authorised and not run.** The **λ ladder
   extension** fired its declared trigger (A3 selected at the top of
   `λ ∈ {0.1, 0.3, 1.0}`) and was authorised to extend once to
   `{0.1, 0.3, 1.0, 3.0}`; it was not run. Now more interesting than when it was
   declared: λ is monotone across the existing ladder in all four cell × stopping
   combinations, and A4 has shown the ranking objective composes with the tree, so
   whether the ranking weight is still climbing at the top of its range is a live
   question rather than a formality. A4 itself has now been run (§2.1).

3. **Resolve the A3-c BCa interval.** A3-c's largest stopping-axis effect (+0.0288)
   falls outside its own percentile CI upper bound (+0.0279), signalling bootstrap
   bias. The screen flagged that it should not be quoted without a BCa check, which
   was not performed.

Still open and explicitly *not* recommended on current evidence: Phase C's §6.3
(limited adaptation, cost unresolved ≥$39.72, frozen null and blend null make it
unmotivated) and §6.4 (its own leakage-clean affinity pretraining, a separate
sub-project). More structural folding is closed in the negative by Phase D.
