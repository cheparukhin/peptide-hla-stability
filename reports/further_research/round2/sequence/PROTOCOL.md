# Corrected sequence controls — frozen 12 October 2026

This is the implementation protocol for milestones 0–1 of [the canonical research plan](../../../NEXT_RESEARCH_PLAN.md), not a separate priority list. This study uses only the original 19,716 training rows and the existing five peptide-cluster folds. It creates development evidence, not independent confirmation.

## Candidates and budget

The machine-readable authority is `protocol.json` and its candidate/grid hash. Original MLP uses the historical hidden layers (256,64), learning rate .001, batch size 256, 300-epoch cap, patience 15, one-hot L2=.00001 and BLOSUM L2=.001. Centred MLP selects shrinkage k in {100,20}. Tree regression selects depth in {4,8}, learning rate .06, L2=10, 2,000 iterations and 100-iteration stopping patience; centred tree regression uses k=20 and the same grid. PairLogit selects the same tree grid, preserving the prior promising depth-8 recipe. Trees use positional one-hot features; MLP uses one-hot and BLOSUM. Allele identity is never an input feature.

Each of five outer fitting partitions has three inner selection folds. Candidate predictions exclude each inner-held cluster from fitting and early stopping by taking a further stopping partition inside the inner fitting rows. MLP screening uses 12 members (three stop folds × two encodings × seeds 0,1). Both final MLP procedures retain the exact original 30-member recipe (five stop folds × two encodings × seeds 0,1,2). Trees use six members (three stop folds × seeds 0,1) throughout. This different ensemble effort is explicit; it is not a capacity-matched neural/tree comparison. The screening/final MLP ensemble-size difference is retained and reported.

Selection uses the mean of the three inner-fold median per-allele Spearman scores. The first candidate in frozen parsimony order within one fold-standard-error of the largest mean is selected. Family selection applies the same rule to each family's selected candidate, in order original MLP, centred MLP, tree regression, centred tree regression, PairLogit. Original-only remains available. No outer score enters either selector.

There are 2,100 initial member fits: 840 MLP, 420 RMSE trees, 840 rankers (including all nested common-scale mapping inputs). Two further fixed-setting refits of original MLP, A1 and A4 on every outer fold add 420 members. Predictions checkpoint after every completed ensemble. Grids do not expand automatically. All fits run locally; cloud cost is $0.

## Clean common-scale mapping and ancestry

Ranker predictions are mapped to log1p scale by a positive affine map (slope floor 1e-8). Its fitting inputs are nested-selected crossfit predictions from within the outer fitting partition. Every input's predicted cluster is excluded from base fitting, early stopping and configuration selection. Candidate selection is repeated within each crossfit fitting partition. The outer-held clusters are absent from every stage. Raw per-fold rank scores are also saved. No isotonic map or fold-wise z-score is used for ranking.

The mapping crossfit models use 2/3 of the outer fitting rows, then their stopping splits; outer-application models use all outer fitting rows, then their stopping splits. Both use the same six-member recipe. This training-size difference can affect scale transport and is a stated limitation.

Each base member records fitting, stopping and predicted row-set hashes; selected predictions also record selection ancestry. Row sets are fully recoverable from the ancestry manifest and row IDs join to the frozen cluster manifest. Automatic ancestry audits reject overlap with either assessed clusters or the currently predicted clusters. Predictor APIs receive no target columns for assessment rows. Label perturbation tests also exercise actual learners, selection and crossfitting. Centring means derive only from gradient-fitting labels; unseen alleles use the fitting-derived global fallback.

## Reporting and repeats

Report all five component procedures and the inner-selected sequence procedure. Primary statistic is the median per-allele Spearman over pooled outer-held predictions, with original ties/constant rules and the ≥20-row panel. The ≥50-row panel is a sensitivity analysis. Fold medians and raw ranker fold results are separate summaries. The original 2,000 paired whole-cluster bootstrap conditions on the saved fitted predictions.

Store per-seed predictions separately. Two extra identical-setting refits assess execution reproducibility of locked A0/A1/A4 configurations; ranker mappings are held fixed for this repeat endpoint. These repeats measure model refit variation and do not measure variation from rerunning the complete nested selection/mapping graph. Hardware, libraries, numerical precision and thread counts accompany the records. No historical MLP difference is imposed as a noise floor.

Completed outputs are keyed by pair_id, cluster_id, outer_fold and arm. The root report generator adds component/reference comparisons, coverage and uncertainty. Original train/validation/test split assignments and historical predictions remain unchanged.

## Local profiling

A successful local two-thread pilot with 10,514 gradient-fitting rows measured 1.16s for one-hot MLP and 2.17s for BLOSUM MLP. One-hot RMSE trees measured 2.35s (depth4) and 5.02s (depth6); BLOSUM trees measured 22.15s and 36.95s. Tree BLOSUM was therefore excluded before candidate assessment. These pilot measurements do not estimate A4 depth8 runtime; the first completed A4 fits provide its updated forecast. Original member/runtime records are in `profile_members.json`.

## Execution amendment before assessment

The full depth-8 PairLogit pilot took 206.22s, invalidating the early provisional runtime estimate based on RMSE trees. `hardware_amendment.json` specifies a bounded Modal-only backend for ranker member fits: at most 900 paid members including repeats, 30 simultaneous containers, hard two-CPU/four-GiB limits, no automatic retries, and a $35 sequence ceiling. The scientific grids and partition logic stay fixed. The full launch awaits the successful measured smoke and the orchestrator's matching-cohort and total-budget decision. MLP and RMSE trees remain local. The original local-only execution proposal above is preserved as the prior state; the amendment controls any subsequently authorized cloud execution.

## Shared budget amendment before assessment

The root allocation is now **$18 for sequence work**. The unconditional timeout-based bound for all 900 ranker calls remains $34.96, so it is not a promise that every possible run can complete within $18. A measured grid-aware forecast is $17.503 including a 25% margin, 62 seconds of overhead per call, and a $1.35 smoke/build reserve. The forecast prices every medium-size member at full-size cost and every selected/repeat recipe as depth 8; pilot wall times overlapped other local work.

The implementation reserves worst-case cost before each complete ensemble, settles successful calls at measured runtime plus the declared overhead, and charges failed or interrupted calls their full reservation. Unused reservations return to the shared pool. A persistent, single-controller ledger governs all five concurrent folds, with a separate immutable maximum of 900 paid member calls. If remaining funds cannot cover the next complete ensemble, the runner stops with checkpoints rather than expanding the allocation or silently dropping work. The root must approve this budget and the final implementation hashes before launch. No candidate, fold, seed, target, or selection rule changed. See `modal_budget_forecast.json` and `hardware_amendment.json`; the prior $35 allocation survives in `hardware_amendment_v1.json`.
