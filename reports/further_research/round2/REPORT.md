# Round-two development assessment

12 October 2026. Completed evidence: extraction, peptide-held-out screens, complementarity, locked main execution repeats and all HLA/joint transfer cells. **The complete transfer audit passed; independent confirmation remains open.** The original test has not been rescored or used to select these procedures.

## Main finding

Frozen Boltz interface embeddings contain useful ranking information in this corrected **peptide-cluster-held-out development assessment**. The prespecified inner-selected embedding-or-original procedure scores **0.721163**, compared with **0.666841** for the competitive sequence selector: **+0.054322, paired 95% interval [+0.031395, +0.080805]**. The interval supports a positive conditional development difference. Although the point estimate exceeds 0.05, its lower bound does not: **the prespecified worthwhile-gain criterion, lower bound above 0.05, is not met**.

This tests a frozen trunk/interface representation hypothesis. It does not reproduce a winning-team pipeline whose precise extraction and modelling details remain unverified or independently confirm generalisation. [Completed main report and comparisons](results/main/REPORT.md).

## Assessment and coverage

All five outer folds preserve **19,716 original training rows, 3,587 peptide clusters and 75 alleles**. The primary endpoint is the difference between the two pooled medians of per-allele Spearman correlations on a common eligible panel; it is not the median of allele differences or of fold scores. The ≥20-row panel contains 69 eligible alleles/19,647 rows. The ≥50-row sensitivity contains 68 alleles/19,619 rows and gives **+0.056352 [+0.031732, +0.079137]** for the primary comparison.

Regularisation and selector decisions use inner folds only; fitting, stopping and scaling exclude each assessed cluster. The selector retains the original-reference fallback. Its saved choices are R5 MLP in four folds and R4 ridge in one, without choosing a winner from outer outcomes. Intervals use **2,000 shared paired whole-peptide-cluster bootstrap draws** and condition on fitted predictions. They exclude selection/execution uncertainty. These training data and earlier benchmark/test results have already influenced development history; preserving the original test in this execution does not create a newly independent confirmation cohort. [Main audit](resources/embedding_completion_audit.json), [coverage](results/main/coverage.csv).

| Procedure | Primary pooled median Spearman |
|---|---:|
| Original 30-member sequence MLP | 0.661522 |
| Competitive sequence selector | 0.666841 |
| R2 chain-mean MLP / ridge | 0.421852 / 0.297055 |
| R4 per-peptide-position interface MLP / ridge | 0.713592 / 0.684510 |
| R5 contact-pooled MLP / ridge | 0.720986 / 0.666315 |
| R5 shuffled-membership MLP | 0.720310 |
| Full 9×34 positional MLP | 0.731541 |
| Prespecified embedding selector | 0.721163 |

Component scores describe declared procedures, not a new selection rule. The repaired sequence selector improves on the original reference by only **+0.005319 [−0.009283, +0.012299]**; ordinary trees and PairLogit did not establish improvement over the competitive neural sequence control. These results do not confirm earlier differently assessed screening claims. [Sequence report](results/sequence/REPORT.md).

## What the positional controls establish

The matched pooling controls preserve dimension, projection, context and head. For MLPs, retaining R3 positions improves on its tiled pooled control by **+0.338713 [+0.302942, +0.377253]**; retaining full 9×34 positions improves on its matched pooled control by **+0.166349 [+0.137791, +0.199165]**. Corresponding ridge differences are also positive. Position contributes within these frozen compressed representations.

The contact-specific claim is narrower. R5 MLP versus shuffled contact membership gives **+0.000676 [−0.009556, +0.023590]**; ridge gives **+0.002013 [−0.011705, +0.032843]**. Neither establishes a benefit from the particular crystal-derived contact map. The full positional MLP's higher score versus R5 MLP gives **+0.010555 [−0.014244, +0.022029]** and changes other representation details, so it neither establishes superiority nor isolates position. R3 versus R2 likewise changes context/channels/pooling.

R2's weak results show that these tested chain averages are insufficient; they do not reject internal embeddings generally. R3 uses eight projected token channels and the full positional control four pair channels: nulls cannot exclude richer-channel models. [All declared contrasts](results/main/paired_comparisons.csv).

## Complementarity

Clean nested blending and residual correction test information beyond the exact original sequence reference. Assessed clusters never enter base fitting, stopping, ridge selection or residual-target construction; other outer-fold predictions do not supply stacking inputs. Zero-weight/correction endpoints reproduce the original baseline.

R4 ridge alone scores **0.684510**. Its inner-selected blend with the original reference scores **0.723789**, giving **+0.039279 [+0.022009, +0.056708]** over R4 alone and **+0.056948 [+0.042591, +0.083811]** over the competitive sequence selector. R4 residual correction scores **0.711429**, with **+0.044588 [+0.030379, +0.067056]** over that selector. These support conditional complementarity; neither lower bound exceeds 0.05 against the competitive control. R2 blending/residual correction provides no established competitive gain. The blend's numerical proximity to the embedding selector is not a tested superiority claim. Inner-to-outer training-size differences remain a limitation. [Report](complementarity/results/REPORT.md), [audit](resources/complementarity_completion_audit.json).

## Reproducibility and extraction provenance

Both identical-setting sequence executions match their primary predictions for the declared original MLP/tree/PairLogit repeat arms. The embedding selector's ten locked fold/repeat executions use **242 fresh local fits**, distinct caches and the original settings, without repeating selection. Primary versus each repeat and repeat versus repeat have **zero maximum/mean prediction difference across all 19,716 rows** and zero reported metric deltas on both panels. Different-seed diagnostics remain separate. These are recorded-platform execution results, not cross-platform equivalence or an assay-noise ceiling. The source publication describes two independent experiments, but supplied data lack replicate-level observations. [Embedding repeat audit](resources/embedding_repeat_completion_audit.json), [sequence repeat audit](resources/sequence_repeat_completion_audit.json).

Pinned extraction passed full-forward/trunk equivalence and technical repeat checks. All 19,716 rows have validated local compact features. **An inherited provenance gap remains:** the cache fanout wrapper's current/committed SHA-256 differs from its launch-authorization digest; exact launch-time wrapper bytes cannot be certified. Both digests are preserved. The four frozen extraction sources, cohort, feature protocol and local artifact checks match. This limits the wrapper-provenance claim; the discrepancy is not evidence of changed features. [Cache audit](resources/cache_completion_audit.json).

## Cost and added inference burden

The one-time shared full cache has **$66.8431 conservative runtime accounting**, reused across representation arms. Production launch to provider-observed fanout stop spans **6 h 17 min**, about **0.871 rows/s** including startup/gating/scheduling. Recorded synchronized extraction averages **7.206 s/case**; summed extraction work is about **39.5 hours**, overlapping across workers. These are different quantities, not individual end-to-end latency.

Sequence accounting including smoke/build and lost-attempt reserves is **$7.9957**. Retaining the full $2 engineering allowance gives **$76.8388 against the $100 ceiling**. Retention does not mean $2 was spent. Provider metering snapshots lag and differ: named-app metering and account increase leave **$3.1554 unattributed**. None is a final invoice or live credit balance; storage is separate.

The local embedding screen records 2,900 fit members and about 31 minutes of stage wall time. Complementarity records **2,140 logical members including safe cache reuse**, about 33 minutes of stage wall; saved fit-time sums do not independently count fresh hardware executions. Neural final heads have 30 members; ridge one. No isolated head-inference or new-case end-to-end benchmark exists. New-input MSA generation, cold startup, storage/egress and full coordinate generation are outside these latency/cost claims. [Runtime/cost summary](resources/runtime_cost_summary.json).

## Completed bounded transfer assessment

All **five HLA folds and 25 joint intersections** retain the full cohort in each regime. The ≥20/≥50 panels have 69/68 eligible alleles spanning 34 eligible HLA groups, out of 35 total. HLA Hamming grouping is a declared sequence convention, not a validated functional distance. The independent audit passed all 13,530 member records, 236 evidence hashes, 108 comparisons and 1,260 group-deletion rows. Saved inner choices and scaler ancestry were verified; raw inner prediction vectors and numerical scaler statistics were not independently reconstructed. [Transfer audit](resources/transfer_completion_audit.json). [Complete transfer report](transfer/results/README.md).

The primary crossed intervals independently resample peptide clusters and HLA groups, with HLA multiplicities replicating alleles in the median. Peptide-only and HLA-only intervals are also reported; all use 2,000 paired draws conditional on fitted predictions. Crossed pigeonhole intervals and group deletion are dependence sensitivities, not refitting or independent confirmation.

| Regime, ≥20 rows | Embedding selector | Bounded MLP selector | Difference, crossed 95% interval |
|---|---:|---:|---:|
| HLA-held-out | 0.546303 | 0.173961 | +0.372342 [+0.290338, +0.453269] |
| Joint-held-out | 0.506675 | 0.110831 | +0.395844 [+0.295075, +0.445743] |

Against the original reference, gains are **+0.375368 [+0.283444, +0.455040]** for HLA and **+0.371689 [+0.278298, +0.436636]** for joint holdout. Thus the result is not solely relative to the weaker bounded selector. The ≥50-row gains over the bounded selector are +0.371665 and +0.394712, with crossed lower bounds +0.288564 and +0.292501. Unlike the peptide-held-out comparison, these lower bounds exceed 0.05—but **only for these complete procedures and narrow comparators**. [All comparisons](transfer/results/comparisons.csv).

R4 ridge and R5 MLP were nominated from the complete main assessment before transfer outcomes. Inner selection chooses R4 in 4/5 HLA and 23/25 joint cells; neither arm is designated a global outer-outcome winner. All 35 HLA-group deletion sensitivities remain positive, as do all five/25 cell-median selector contrasts against the bounded selector on both panels. No reported comparison interval omits its point estimate. Cell/deletion results are diagnostics, not independent tests. [Deletion sensitivity](transfer/results/leave_one_hla_group_out.csv), [cell diagnostics](transfer/results/per_cell.csv).

The bounded sequence selector excludes main-screen trees/PairLogit; inner selection and stopping change with regime. Transfer gains compare complete procedures: ridge uses all permitted fitting rows, while neural members reserve stopping rows and, in joint stopping, exclude cross quadrants. Representation and training-recipe effects are therefore combined. Joint pooled rankings also reflect predictions from multiple fitted cells and their scales; no outcome-based recalibration was applied. Scores across regimes cannot be subtracted to identify a causal structural effect. Main execution repeats do not measure transfer execution variation.

The resource closeout binds 206 hashes and records **13,530 local fits**, about **84 minutes** of operational fitting and **137.6 seconds** of reporting, with **zero added Modal compute**. Shared extraction is not charged again; no isolated inference benchmark was performed. [Transfer resource closeout](resources/transfer_runtime_closeout.json).

## Independent confirmation remains open

The public stability release overlaps the canonical corpus. Conservative donor/canonical exclusions leave only **129 candidate rows across heterogeneous assays**, insufficient for representative confirmation; their outcomes remain reserved. Construct, endpoint, checkpoint ancestry and cross-length exposure gaps remain. [Identity/provenance audit](independent_data/README.md).

The completed synthetic design examines 12-allele panels of 200/400/800 measurements per allele plus separately quoted repeats for a stratified 20% subset. It supports asking for comparable assay quotes, not a guaranteed power or assay-noise claim. The 12×400 option is a bounded starting design for a true 0.05 gain under tested assumptions; a true 0.02 gain remains difficult under strong dependence. **No assay quote, partner commitment or adequately powered independent cohort is secured.** Final procedure/analysis must be locked before a one-time confirmation; optional calibration/adaptation remains separately gated. [Conditional design study](independent_data/design/README.md).
