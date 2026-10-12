# Matched Boltz procedures retain positive transfer gains in this development assessment

Completed 12 October 2026. These results use the original training cohort under HLA-held-out and joint peptide/HLA-held-out assessment. They are conditional development evidence from previously examined data; the original validation/test partitions were not used for this follow-up.

The matched comparisons remain positive on both declared panels. At min20, R4 ridge exceeds R0 ridge by **+0.465509 [+0.395953, +0.553410]** in HLA holdout and **+0.463260 [+0.380234, +0.524616]** in joint holdout. R5 MLP exceeds R0 MLP by **+0.406234 [+0.307286, +0.473273]** and **+0.373784 [+0.309165, +0.463816]**, respectively. All intervals here are the saved paired crossed peptide-cluster/HLA-group 95% intervals. The min50 sensitivity agrees in sign and retains positive lower bounds.

The embedding selector also exceeds the expanded sequence selector by **+0.354169 [+0.276201, +0.456250]** in HLA holdout and **+0.398220 [+0.314321, +0.461279]** in joint holdout at min20. The positive result therefore persists after adding the declared sequence trees and fixed ranker to the sequence control. This supports these complete frozen procedures under the specified development transfer regimes; it does not establish a causal structural mechanism or independent assay generalisation.

The endpoint is pooled median per-allele Spearman correlation on a common eligible panel. A difference is the difference of the two pooled medians, not the median of allele differences or fold medians. Each regime covers all **19,716 original training rows and 3,587 peptide clusters**. Min20 includes 69 alleles / 19,647 eligible rows; min50 includes 68 alleles / 19,619 rows. Both panels span 34 eligible HLA groups out of 35 actual groups.

| Regime | Panel | Original sequence MLP | Expanded sequence selector | R0 ridge | R4 ridge | R0 MLP | R5 MLP | Embedding selector |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HLA held out | min20 | 0.170935 | 0.192134 | 0.084272 | 0.549781 | 0.160485 | 0.566719 | 0.546303 |
| HLA held out | min50 | 0.163120 | 0.191836 | 0.081152 | 0.548667 | 0.159706 | 0.565128 | 0.543417 |
| Joint held out | min20 | 0.134985 | 0.108454 | 0.049418 | 0.512678 | 0.124214 | 0.497998 | 0.506675 |
| Joint held out | min50 | 0.130702 | 0.107051 | 0.048711 | 0.510709 | 0.106089 | 0.492374 | 0.504977 |

The original sequence arm is the frozen sequence MLP reference. The expanded sequence selector chooses, using inner data only, among original MLP, centred MLP, log-target tree, centred-target tree and the fixed depth-8 PairLogit ranker. Its min20 gain over the original is +0.021199 [−0.064630, +0.073387] in HLA holdout and −0.026531 [−0.088797, +0.018463] in joint holdout. These intervals do not establish that expanding the sequence control improved it, and do not establish equivalence. The selector's joint score is numerically lower; inner selection is not guaranteed to improve outer assessment performance.

The matched R4/R0 ridge and R5/R0 hidden-64 MLP comparisons use the declared matching of head recipe, grids, preprocessing, selection and stopping. Representation dimensions and resulting MLP parameter counts still differ; they are comparisons of complete feature/head procedures. R4 is the position-preserving interface representation, R5 the contact-pooled interface representation, and R0 positional peptide/pseudosequence one-hot. R4 and R5 were nominated before these transfer outcomes; their component scores do not define a new global winner.

**Matched differences with crossed 95% intervals:**

| Regime | Panel | R4 ridge − R0 ridge | R5 MLP − R0 MLP |
|---|---:|---:|---:|
| HLA held out | min20 | +0.465509 [+0.395953, +0.553410] | +0.406234 [+0.307286, +0.473273] |
| HLA held out | min50 | +0.467516 [+0.395092, +0.553011] | +0.405422 [+0.304430, +0.473766] |
| Joint held out | min20 | +0.463260 [+0.380234, +0.524616] | +0.373784 [+0.309165, +0.463816] |
| Joint held out | min50 | +0.461998 [+0.378981, +0.524035] | +0.386285 [+0.307505, +0.463886] |

**Embedding-selector differences with crossed 95% intervals:**

| Regime | Panel | Embedding selector − original sequence MLP | Embedding selector − expanded sequence selector |
|---|---:|---:|---:|
| HLA held out | min20 | +0.375368 [+0.283444, +0.455040] | +0.354169 [+0.276201, +0.456250] |
| HLA held out | min50 | +0.380297 [+0.286483, +0.455424] | +0.351581 [+0.274637, +0.457272] |
| Joint held out | min20 | +0.371689 [+0.278298, +0.436636] | +0.398220 [+0.314321, +0.461279] |
| Joint held out | min50 | +0.374275 [+0.277437, +0.434564] | +0.397926 [+0.314254, +0.465321] |

All 35 leave-one-HLA-group-out deletions retain positive matched R4−R0 ridge, R5−R0 MLP and embedding-selector−expanded-sequence differences in both regimes and both panels. These are dependence sensitivities, not independent tests or model refits. The full [comparison table](comparisons.csv) also preserves peptide-only and HLA-only intervals, every declared arm and all 22 declared contrasts for each regime/panel/mode. [All group deletions](all_group_deletions.csv) preserve every actual group, including groups outside an eligible scoring panel.

Uncertainty uses 2,000 shared paired draws with seed 20261003. Crossed draws independently resample peptide clusters and HLA groups, with integer HLA-group multiplicities replicating alleles in the median. Intervals condition on fitted predictions; they omit retraining, selection repetition, execution variation and independent assay variation. The many reported contrasts are nominal and unadjusted for multiplicity. Previously inspected development evidence and nomination history prevent interpreting these intervals as independent confirmation.

HLA Hamming grouping is a declared sequence convention rather than a validated functional distance. Ridge and neural procedures differ in label access because neural stopping reserves rows, and joint stopping excludes cross quadrants. The ranker uses a positive-affine scale transported from smaller inner fits to final outer fits; pooled joint rankings combine predictions from different fitted cells. No outcome-based recalibration was applied. Neither the matched gains nor differences across regimes identify a biological mechanism, a particular contact-map benefit, deployment latency, an assay-noise ceiling or performance on a new independent cohort. The hybrid study is a separate frozen question and is not assessed by this report.

The downloaded report binds all 30 complete cells (five HLA and 25 joint), representing **10,830 logical fits**. The cloud recovery retained 25 whole completed cells and recomputed only the five remaining joint cells: 2,000 fresh fits, with the failed partial attempts kept separate. The unchanged original semantic audit passed remotely before report delivery. This narrative used only saved outputs; it performed no fitting or bootstrap reruns.

The lightweight [coverage audit](coverage_audit.json) verifies the 17 remote output-file hashes, 16 CSV hashes, all 30 completion bindings, 264 unique comparison rows, 12 complete 2,000-draw bootstrap files, and 3,080 deletion rows (770 for each regime/panel). [The report manifest](manifest.json) and the portable transfer-report [phase certificate](../../resources/cloud_sandboxes/transfer-report-20261012-2025/phase_certificate.json) preserve upstream evidence. This Markdown report and its coverage audit are locally authored delivery artifacts; they do not alter the original remote manifest or claim another model audit.
