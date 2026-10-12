# Hybrid follow-up: reusable evidence and helper inventory

Prepared 12 October 2026 from completed artifacts and source inspection only.
**No fits, cloud calls, new outcome analysis or commits were performed for this
inventory.** The hybrid protocol is being specified separately. Frozen
round-two, ranking and transfer sources/results remain immutable.

## Population, features and partitions

All reusable peptide-held-out evidence covers the original training partition:
19,716 pairs, 3,587 peptide clusters and 75 alleles. Labels are
`y_log1p = log1p(thalf_hours)`. The original test must not enter these procedures.

| Identity | Value |
|---|---|
| Dataset, `data/rasmussen_et_al_dataset.csv` SHA-256 | `17c574c2c6d2746a238804bd73fa05279ae23e476be5df124dc084e116635658` |
| Original `data/splits.csv` SHA-256 | `6173c8de986c9187601c697597920090d27192e46221558012a88c43e1671877` |
| Canonical sorted training-frame hash | `558ab95517b6f0b7c3e9f60d088ad2afef4ace3a1d085a784ec8b7123c41c84b` |
| Five outer-fold integer-vector hash | `50fdca745c9e8e99e87b50741cfc9bd4c703e400aff8dd6469555ecdeea31b6f` |
| `reports/finalization/ranking/fold_manifest.csv` SHA-256 | `d3a01774b286684dad438b55bb429d6bbd9fea648c9ad6d07b121afe430a9144` |
| R0 float32, shape `(19716,860)`, canonical matrix hash | `d84f1e8244b0a25838f1c9f12fa19f70437e7dd488388306f8fc35ffc3966828` |
| R5 float32, shape `(19716,1024)`, canonical matrix hash | `8c02288566e92de5e0c8a2b87d0d636f25bf01d14eb8711292f78af28c4030ed` |

R0 is position-preserving peptide+pseudosequence onehot, constructed with
`pepstab.features.build_features(train, 'pep_pseudo', 'onehot')`. R5 is the
fixed Boltz contact-pooled representation. The complete compact-feature origin
is [feature_bundles.json](../../further_research/round2/extractor/full_training/feature_bundles.json),
SHA-256 `24d1afeac0640e9b83d24aa75c3f23456c8629d2a1a82452acf3243f66c61799`.
It declares all 19,716 completed rows across 78 directories under
`features/round2_boltz/round2-fulltrain-20261012/shard_00000` through
`shard_00077`; use its declared ordering, not lexical shard ordering. Its
cohort hash is `2cf4ab02a738a3af916003fbc470c07c9c6445234cc084c850148174f01effd0`.
Each directory has its own completion, metadata and array hashes. Large feature
caches are external to Git; do not duplicate or regenerate them for this task.

Use `scripts.round2_embedding_screen.training_frame()` for sorted canonical
training rows, `pepstab.nested.outer_folds(train)` for five outer folds,
`inner_folds(outer_fit,3)` for screening and `inner_folds(call_fit,5)` for member
stopping. Splitting is based on the supplied frame's clusters: inner/stop
assignments cannot be copied from the full population. Exact row and cluster
joins must precede feature slicing and prediction combination.

## Reusable completed MSE evidence

The immutable original embedding protocol hash is
`2251f3e6db0ff8f106f2a0050e7d8930cff5b3054196a40561f10ff42dc5b013`.
Original component directories are
`reports/further_research/round2/embedding/fold_{0..4}/{R0,R5}_mlp/`.
Each has `selection.json`, `predictions.csv`, `seed_predictions.csv`,
`ancestry.json`, `members.csv` and `completed.json`.

| Saved evidence | Coverage and arity | Use |
|---|---|---|
| [ranking/reused_inner_predictions.csv](../ranking/reused_inner_predictions.csv) | 473,184 rows: both representations × five outers × all three fixed L2 candidates × three inner held blocks; each vector averages two members | Fixed-candidate inner predictions, with identity columns |
| `ranking/fold_{f}/{rep}/reused_inner_predictions.csv` | Three candidates over the corresponding outer-fitting rows | Convenient per-cell equivalent of the global file |
| Original `embedding/fold_{f}/{rep}_mlp/predictions.csv` | One selected candidate's outer-held vector, 30 members | Historical final reference; valid only on that outer's held rows |
| Original `seed_predictions.csv` | Six seed vectors, each averaging five stopping members | Seed diagnostics, not individual final-member predictions |
| Original cache calls under `runs/round2_embedding/<old protocol>/fold_{f}/{rep}_mlp/<call hash>.json` | 90 inner calls ×2 +10 final calls ×30 =480 logical members | Canonical exact cache predictions and member ancestry |

The complete **exact 100-path cache index** and each file SHA-256 are in
[ranking/reuse_audit.json](../ranking/reuse_audit.json), `sources` entries with
`logical_members`. These contexts identify each call:

```text
outer/{f}/{rep}_mlp/select/{rep}_mlp_{1|0.01|0.0001}/{inner 0..2}
outer/{f}/{rep}_mlp/outer_refit
```

Each representation/outer uses 18 screening members (3×3×2) and 30 final
members, so 48; each representation totals 240. Inner axes are
stop0/seed0 and stop1/seed1. Final axes are stops0..4 × seeds0..5. All ten
selected final MSE candidates are L2 0.01. `scripts.final_ranking.cached_call`
and `canonical_cache_key` reconstruct the exact content-addressed call path;
`read_baseline` performs the saved selection and ancestry audit without fits.

**Critical arity limitation:** original MSE final caches retain only the overall
mean and six means over five stopping folds. Their individual stop0/seed0 and
stop1/seed1 final vectors cannot be recovered. A newly matched two-member
final MLP baseline must therefore be freshly fitted; a seed mean cannot stand
in for a stopping member. Individual inner member vectors are recoverable
because each inner seed occurs once. Historical MSE CSVs use float32's shortest
roundtrip representation; exact cache equality requires restoring float32.

## Reusable completed mixed-ranking evidence

The immutable ranking protocol hash is
`32d3f8f1908bd5bb3df81ed205c61771b567d2e349f92ebfa199da8111b0ef8d`.
[completion_audit.json](../ranking/completion_audit.json) records the completed
source/member/scaler/selection replay and independent arithmetic checks.

| Saved evidence | Coverage and arity |
|---|---|
| `reports/finalization/ranking/fold_{f}/{R0,R5}/inner_predictions.csv` | All nine fixed positive candidates: λ0.1/0.3/1 × L2 1/0.01/0.0001, all three inner blocks; each mean uses two members |
| Same directory `predictions.csv` | Outer-held `R{rep}_mse`, `R{rep}_ranked`, `R{rep}_deploy`; 30-member MSE and selected-ranking means, then the saved deployment fallback |
| Same directory `selection.json`, `ancestry.json`, `member_manifest.json`, `members.csv`, `completed.json` | Frozen choices, rowsets, individual identities and exact output hashes |
| Same directory `members/<signature hash>/stop_{stop}_seed_{seed}.npz` + `.json` | Individual predictions, scaler mean/scale/variance, MSE stopping curve and completed receipt |
| [ranking/predictions.csv](../ranking/predictions.csv) | 118,296 rows = six arms ×19,716 original train pairs |

Across both representations/five outers there are exactly **840 fresh ranking
member NPZ and 840 completed member receipts**: 540 inner members plus 300
final members. Each component cell has 54 inner members and 30 final members.
Per-cell inner CSV coverage is nine times that cell's outer-fit size; for
fold0/R5 it is 141,948 rows. The files retain `pair_id`, `cluster_id`, `allele`,
`outer_fold`, `inner_fold`, `representation`, `candidate`, `y_pred`.

Example exact individual paths:

```text
reports/finalization/ranking/fold_0/R5/members/22a95879822df11ad06f72512ac6efafca19e51bf519fe56601894492779dffb/stop_0_seed_0.npz
reports/finalization/ranking/fold_0/R5/members/cf4e0702d627fe63b658f5aa64278da2cbc10051f516a54256f4b4769dc772b7/stop_4_seed_5.npz
```

Their prediction shapes are `(5258,)` and `(3944,)`; scaler vectors are
`(1024,)`. The JSON identity gives ordered gradient/stop/prediction IDs,
feature and label hashes, context, seed, stop fold, candidate and protocol.
Resolve all other exact paths from `member_manifest.json` by
`config_digest(member['identity']['signature'])`, then stop/seed basename.
No trained neural weights were saved in these scientific NPZ files, so they
cannot predict additional partitions or yield new transformed features.

Only the **selected positive candidate** has final predictions: R0 λ0.1/L2.01
in all five folds; R5 λ0.1/L2.01 except fold2, which uses L2.0001. Do not treat
final predictions for the other eight candidates as cached. In particular,
R5 λ0.1/L2.01 lacks a final30 vector in fold2.

Global source identities:

| Path | SHA-256 |
|---|---|
| `ranking/predictions.csv` | `8412605c1ff8c607171674c87d753b49fa6d842f507c38f371e40cb15fd96483` |
| `ranking/reused_inner_predictions.csv` | `d0429b741b289008a755458a0efc7c76c67694f1b1e3d6620cf35efa364a4053` |

Each per-cell inner/final hash is already bound in its `completed.json`; do
not replace those historical outputs or their receipts.

## Leakage limits for blend and learned preprocessing

1. Fixed-candidate cached inner vectors exclude their predicted clusters from
   gradient fitting, stopping and scaling. They may support a newly frozen
   **joint candidate/weight choice inside an outer fitting partition**, with
   a separately held outer assessment. They are not independent confirmation.
2. A candidate selected using all three inner blocks has selection ancestry
   through those same labels. Its selected inner vector is **not a nested
   selected-base OOF vector**. A claim of row-clean selected-input stacking
   requires deeper re-selection without each predicted meta fold, or an
   explicitly fixed base recipe. The proposed fresh fixed-candidate hybrid
   inner predictions avoid that old selected-OOF ambiguity.
3. Original outer OOF predictions from other folds are forbidden as inputs
   inside a particular outer fit: their upstream fits can include that
   outer's assessment clusters. Only the matching outer-held final vectors
   are safe historical assessment references for that outer.
4. Blending a two-member tree with a historical 30-member MLP confounds the
   learner/objective question with ensemble arity. The proposed new matched
   two-member standard MLP is an appropriate fresh comparator; historical30
   should remain explicitly descriptive. Fixed weight0 must reproduce its
   designated fresh MSE endpoint exactly.
5. `models/finalization/R5_mlp_0.01/` is a full-training deployment refit.
   Although it stores real weights/scalers, it is forbidden as a clean
   original-training assessment or stacking input. The package's warm latency
   benchmark is not predictive evidence.
6. PCA/quantile/scaler states must fit the exact member gradient rows only,
   then transform stopping/held inputs without their targets. Supervised tree
   preprocessing must exclude downstream held outcomes and record all fitting
   and any stopping/selection ancestry. The new fixed-iteration gradient-only
   tree design avoids a tree early-stopping boundary. Leaf vocabularies and
   encodings must be fixed/gradient-derived; never fit them on all held inputs.
7. A pure tree ranker's score scale is not automatically log1p half-life.
   Any later affine scale-transport map is a separately recorded selection
   operation on clean inner predictions and permitted fitting labels. It
   cannot use outer-held outcomes or be silently inherited from prior transfer.

## Safe existing APIs and wrappers to inspect/reuse read-only

| Source | API | Boundary or limitation |
|---|---|---|
| `pepstab/round2_nested.py` | `unlabelled`, `validate_partition`, `align_predictions`, `Ancestry` | Peptide cluster/row exclusion and content-addressed rowsets; extend audit to transform/leaf roles in new code |
| Same | `select`, `clean_crossfit` | Inner selection and genuinely nested selected-base OOF; these functions **fit** via their predictor when called, so inventory does not execute them |
| Same | `blend`, `positive_affine`, `apply_affine`, `config_digest` | Explicit endpoints and scale map; caller owns clean input ancestry and declared selection protocol |
| `pepstab/round2_complementarity.py` | `choose_meta_weight`, `_assert_exclusion`, `base_crossfit` | Existing one-fold-SE zero-first blend rule and deeper ancestry pattern; original recipe/representation registry is frozen and unsuitable as an unmodified new runner |
| `scripts/final_ranking.py` | `training_frame`, `load_features`, `cached_call`, `read_baseline`, `verify_freeze` | Read-only data/cache reconstruction; `load_features` loads all compact representations before retaining R0/R5, so it has avoidable memory overhead |
| `scripts/round2_embedding_screen.py` | `load_compact_features` | Complete source/metadata/contact/projection/row checks; loads every representation rather than R5 alone |
| `pepstab/round2_inference.py` | `load_feature_batch(...,'R5')`, `FeatureBatch.select` | Validated **R5-only** feature loading saves memory; select by canonical IDs to restore training-frame order. R0 is built separately |
| `pepstab/mlp.py`, `pepstab/phase_a.py` | `MLPConfig`, `MLPRegressor`, `RankingMLPRegressor`, `ranking_pairs` | Existing fixed-width neural fit/MSE stopping and within-allele pair generation; no new learner or stopping change implied |
| `pepstab/final_transfer.py` | `tree_fit` | Generic feature arrays accepted, but frozen transfer params/registry; returns only predictions/details, not fitted models or leaf arrays |
| Same | `ranker_task` | **Do not reuse as a Boltz worker**: validates fixed sequence-ranker8 candidate and constructs sequence onehot itself |
| `pepstab/round2_reporting.py` | `cluster_bootstrap_scores`, `compare_predictions` | Shared 2,000 paired whole-cluster draws; use all declared new arms on common original-training panels |

Current HLA/joint transfer artifacts use different assessment/stopping units
and are being produced under a separate controller. Their trees/ranker use
sequence features and do not constitute Boltz tree evidence. Do not merge
those partially completed cells into the new peptide-cluster reuse graph.
No existing saved tree model is available here for the proposed Boltz leaf
pipeline; a new gradient-only tree and saved CBM artifact are required.

## New implementation boundaries agreed with the root

Use new files only: `pepstab/final_hybrids.py`,
`tests/test_final_hybrids.py`, `scripts/final_hybrids.py`,
`scripts/final_hybrids_pipeline.py`, `tests/test_final_hybrids_pipeline.py`,
and a separately owned `scripts/final_hybrids_report.py` if needed. Outputs
belong under `reports/finalization/hybrids/`. Do not edit frozen old sources.

The proposed core/runner contract is:

```text
fit_member_bundle(
    features={'gradient': Xg, 'stopping': Xs, 'held': Xh},
    gradient, stopping, held_unlabelled,
    representation, seed, context, artifact_dir, protocol_hash,
    source_feature_hash=None,
) -> {
    'predictions': {component: vector}, 'n_fits': 6,
    'identity': identity, 'receipt': path, 'artifacts': artifacts,
}

audit_member_bundle(
    features, gradient, stopping, held_unlabelled,
    artifact_dir, protocol_hash, expected_identity,
) -> same predictions after read-only replay, without refits
```

Role arrays must match the explicit frames in order. Core identity binds IDs,
feature hashes, permitted label hashes, source/protocol and recipe. A durable
exclusive launch receipt prevents automatic retries. The root requires saved
CBM trees, pickle-free MLP weights, transform states, scaler stats, individual
predictions and stopping curves; the report must reconstruct predictions from
these artifacts instead of merely trusting checksums. Sol owns fold loading,
aggregation, frozen affine/blend/selector choices, controller/resource gates
and global fit accounting. This API proposal is not a fitting authorization.
