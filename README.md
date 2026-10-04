# Peptide–HLA stability

Predicting peptide–HLA dissociation half-life.

**Read [reports/SUBMISSION.md](reports/SUBMISSION.md) first** — it is the
written-up answer, with every claim sourced to the stage report behind it.
[HACKATHON_PLAN.md](HACKATHON_PLAN.md) has scope and stages;
[EVALUATION.md](EVALUATION.md) has the evaluation rules, frozen at stage 1 and
unchanged since; [reports/limitations.md](reports/limitations.md) is everything
to discount the result by; [reports/compute_ledger.md](reports/compute_ledger.md)
is every dollar and GPU-hour.

> **Every number in this repository is a validation number.** The frozen test
> split has not been scored. It is scored **once**, at stage 6, with every arm
> together, under
> [reports/TEST_SCORING_RUNBOOK.md](reports/TEST_SCORING_RUNBOOK.md). Nothing
> below may be read as a held-out test result, including the comparisons that
> return conclusive verdicts.

**The short version.** A trained sequence baseline reaches median per-allele
Spearman **0.693** on validation. Frozen ESM-2 neither replaces nor improves on
it: 0.683 alone (Δ −0.0101 [−0.0382, +0.0352]) and 0.676 added on top
(Δ −0.0170 [−0.0468, +0.0320]) — both **inconclusive at zero, both ruling out
the predeclared +0.05 worthwhile gain**. The one conclusive comparison in the
study is about the **weakness of full-domain one-hot encoding**, not the
strength of pretraining: ESM-2 beats the full-domain ensemble by +0.0152
differential concordance and the 34-residue pseudosequence ensemble beats it by
+0.0124, while ESM-2 against the pseudosequence arm stays null. The structural
arm is still running.

## Live status — 4 October 2026

| Workstream | State |
|---|---|
| Stages 1, 2, 2b, 2c | complete |
| Stage 3 (ESM-2), 3b (ESM-2 × affinity), 3c (elution) | complete |
| Stage 3d (ESM-2 likelihood features) | **not run** — untested scope, not a null; the stage 3 conclusion is bounded to embeddings because of it |
| Stage 4a, 4b, 4b.1, 4c pilot | complete |
| **Stage 4c Boltz-2 production fold** | **running** — launched 04:00 BST, **251 of 282 shards committed, 25,100 of 28,166 pairs, zero failures**; tracking completion ~11:30–12:20 BST |
| Stage 4c.5 structural feature extraction | extractor validated on all 90 pilot folds and on 2,000 live production folds (1,000 per half, zero failures); **final full pass waits on the fold** |
| Stage 5 structural ablation | **not started** — blocked on 4c.5 |
| Stage 5 inverse folding (ProteinMPNN) | pilot complete; ~$1 QC sample approved, not launched |
| Stage 6 machinery | complete and exercised twice on validation |
| Stage 7a (censored likelihood), 7b (allele hold-out) | complete |
| Stage 6 test scoring | **not run** |

Structural work folds a **383-residue, three-chain construct**: 275-residue HLA
ectodomain + 99-residue beta2m + 9-residue peptide. The matched 90-fold pilot
(45 per model) is complete: Boltz-2 passed its gate, ESMFold2 failed on the
sentinel complex, and production is frozen as **Boltz-2 over all 28,166 pairs,
split across two Modal workspaces**. See
[stage 4c of the main plan](HACKATHON_PLAN.md#4c-ectodomain--beta-2-microglobulin-folding)
for scope and budget, and the
[stage 4c report](reports/stage4c_ectodomain_pilot.md) for results, how to run
production, and the stage 5 feature contract. Earlier two-chain measurements
remain historical evidence.

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy pandas scipy scikit-learn pyarrow pytest
```

## Stage 1 output (done — read before writing a model)

| Artifact | Description |
|---|---|
| [reports/audit_summary.md](reports/audit_summary.md) | Data audit: alphabets, constructs, zero-label pile-up, split design |
| `data/splits.csv` | Frozen splits: `pair_id, allele, peptide, cluster_id, split, dist_to_train` |
| [EVALUATION.md](EVALUATION.md) | Predeclared metrics, eligible alleles, 0.05 improvement threshold |
| `pepstab/` | Shared library: data loading, splits, scoring |
| `tests/test_contract.py` | 51 guards on the above |

## Stage 2 output (done — the baseline stage 3 must beat)

| Artifact | Description |
|---|---|
| [reports/stage2_baselines.md](reports/stage2_baselines.md) | Six sequence arms, paired CIs, cost, limitations |
| `reports/stage2_runs.csv` | Every run in the grid; `stage2_summary.csv` is the selection |
| `preds/seq_ensemble_pep_pseudo.csv` | **The baseline every later arm must beat**: 30-network ensemble, median per-allele rho 0.693 |
| `preds/seq_baseline.csv` | Single-network headline arm (rho 0.610); the arm/encoding comparison |
| `preds/seq_ensemble_pep_domain.csv` | The full-domain ensemble (rho 0.653) — the matching baseline for any domain-embedding claim |
| `pepstab/features.py` | One-hot and BLOSUM62 encodings, cached per unique sequence |
| `pepstab/mlp.py` | Small numpy MLP; stops on a caller-supplied fold |
| `tests/test_baselines.py` | 25 guards, including the fit/dev leakage check |
| `reports/compare_to_paper.csv`, `tests/test_calibration.py` | Calibration against NetMHCstabpan, factor by factor; 17 guards |

Validation median per-allele Spearman: **0.693** (30-network ensemble, the
NetMHCstabpan method under our splits), **0.610** (single network), **0.278**
(ridge on identical features), **0.000** (training allele mean). Full-domain
input ties the pseudosequence within noise (+0.016 [−0.031, +0.084]), so later
stages compare domain embeddings against *both*.

**Ensembling alone is worth +0.074 mean SCC against the deployed single
network** (median per-allele rho +0.083 [+0.029, +0.124]), from no new
information — so every arm is ensembled identically, or none is. A figure of
+0.090 also appears in the literature of this repo: it is the ensemble minus
**the mean of its own 30 members**, a different reference, and
[reports/stage2_baselines.md](reports/stage2_baselines.md) states which is
which. Quote +0.074 unless you mean the other one and say so.

**NetMHCstabpan is calibration, never a comparator** — it trained on every
peptide in our test split.

## Stage 2b output (done — augmentation does not help)

| Artifact | Description |
|---|---|
| [reports/stage2b_augmentation.md](reports/stage2b_augmentation.md) | Measured vs predicted weak-binder negatives, paired intervals, and the mechanism |
| `data/augmentation/*.csv` | Three candidate manifests with per-row provenance; `provenance.json` carries seeds, the leakage ledger and digests |
| `reports/stage2b_arms.csv` | The seven arms; `stage2b_deltas.csv` the eight paired intervals |
| `reports/stage2b_negatives.csv` | Anchor composition and what the baseline already predicts per pool |
| `pepstab/augment.py` | Candidate filtering, the Hamming ≥ 4 exclusion, manifest verification |
| `tests/test_augmentation.py` | 47 guards, including the weighted-loss and distance-rule checks |

Neither source helps: best arm +0.024, **all eight paired intervals cross zero
and exclude 0.05**. The reason is that the two sources supply different kinds of
negative. Measured weak binders have near-canonical anchors and the model already
scores them near the floor. Predicted ones from random peptides have the wrong
anchors and the model already scores them *below* the floor — they're easier than
the easiest real data. Later stages run unaugmented; stage 6 scores no augmented
model.

## Stage 2c output (done — negative, and bounded)

| Artifact | Description |
|---|---|
| [reports/stage2c_affinity.md](reports/stage2c_affinity.md) | Auxiliary affinity head: λ sweep, paired CIs, the ceiling diagnostic, the leakage audit, limitations |
| `reports/stage2c_runs_*.csv` | Every run; `stage2c_deltas_*.csv` are the paired CIs |
| `reports/stage2c_affinity_ceiling.csv` | What measured affinity can rank on its own |
| `reports/stage2c_expansion_audit.csv` | Hamming-distance exclusion table for the declined expansion |
| `pepstab/multitask.py` | Two-headed MLP; at λ=0 **bit-identical** to `pepstab/mlp.py`, and its ensemble to `preds/seq_ensemble_pep_pseudo.csv` |
| `pepstab/affinity.py` | Affinity target transform, dual-labelled join, Hamming ≤ 3 leakage filter |
| `tests/test_multitask.py` | 17 guards, including the λ=0 parity check and the C67S exclusion |

Auxiliary affinity labels **do not help** the sequence arm. Across 20 paired
comparisons (5 λ × 2 encodings × {single network, 30-network ensemble}, plus a
censoring variant), every 95% CI crosses zero and every upper bound is below
0.05 — largest +0.034, so the worthwhile gain is ruled out, not undetected. The
null is clean because the mechanism is visible: the affinity head genuinely
learns (ρ ≈ 0.58 on held-out affinity), but measured affinity used *directly* as
a stability predictor ranks at ρ **0.580** — below the 0.610 stability labels
alone already give. Redundant signal, not absent signal. The expansion to 64,226
leakage-filtered IEDB rows is declined on this evidence. The ESM-2 half of the
same hypothesis is **stage 3b**, below.

## Stage 3 output (done — ESM-2 neither replaces nor improves the baseline)

| Artifact | Description |
|---|---|
| [reports/stage3_esm.md](reports/stage3_esm.md) | The headline comparison, the three controls, tuning sensitivity, cost |
| `preds/esm_ensemble.csv` | **ESM-2 only**, 30 networks (35M, middle layer, peptide per-position + 34-contact) |
| `preds/esm_plus_seq_ensemble.csv` | **Sequence + ESM-2**, 30 networks |
| `preds/esm_ensemble_150m.csv` | The 150M checkpoint, same protocol |
| `reports/stage3_runs.csv`, `stage3_comparisons.csv`, `stage3_headline.json` | Every run (355 rows, 324 networks, 86 CPU-minutes); the matched table |
| `reports/stage3_tuning_sensitivity.csv`, `stage3_embedding_cost.csv`, `stage3_contact_index.json` | The L2 ladder check, measured extraction cost, the verified contact indexing |
| `pepstab/esm.py`, `scripts/esm_features.py`, `scripts/esm_arm.py` | Feature cache (once per unique sequence) and the arm |
| `tests/test_esm.py` | 21 guards, including one asserting the string `"test"` never appears in `scripts/esm_arm.py` |

All arms: 30 networks, the same `cv_folds()` protocol as the stage 2 ensemble,
the same 2,817 validation rows and 68 eligible alleles, paired cluster bootstrap
at 2,000 resamples.

| Arm | median per-allele ρ | Δ vs baseline | 95% CI | Verdict |
|---|---:|---:|---|---|
| sequence baseline (30-net, pep + pseudoseq) | **0.6931** | — | — | reference |
| ESM-2 only | 0.6830 | −0.0101 | [−0.0382, +0.0352] | inconclusive at 0, rules out +0.05 |
| sequence + ESM-2 | 0.6761 | −0.0170 | [−0.0468, +0.0320] | inconclusive at 0, rules out +0.05 |
| ESM-2 150M | 0.6737 | −0.0194 | [−0.0599, +0.0195] | inconclusive at 0, rules out +0.05 |

**Both intervals cross zero, so this is not a demonstration that ESM-2 is
worse** — and both upper bounds sit below the predeclared bar, which is the
strongest negative this evaluation supports.

**The one conclusive comparison is about full-domain encoding, not about
pretraining.** Against the full-domain ensemble, the contract's primary metric
is inconclusive and two secondaries resolve:

| Comparison against the full-domain ensemble | Differential concordance | Verdict |
|---|---:|---|
| ESM-2, on the same 182 residues | +0.0152 [+0.0046, +0.0241] | conclusive |
| the 34 hand-picked contact residues | +0.0124 [+0.0061, +0.0186] | conclusive |
| *ESM-2 vs those 34 hand-picked residues* | *+0.0028 [−0.0052, +0.0102]* | ***null*** |

Two quite different things beat full-domain one-hot by about the same amount,
and the comparison that would matter stays null. On the primary metric, ESM-2 vs
full-domain is **+0.0301 [−0.0084, +0.0757] — inconclusive**; on the mean
per-allele ρ it is +0.0312 [+0.0082, +0.0540]. Neither establishes the 0.05 bar.

**Three controls make the negative worth reporting.** (1) A transplanted
regularisation ladder — stage 2's `{1e-5, 1e-3}`, chosen for sparse one-hot
features — cost the ESM arm **0.109 SCC** (0.4994 against 0.6081 at its own
boundary-checked L2) and would have shipped a confident false negative; the
check was applied symmetrically to every arm. (2) The PCA compression adopted
for memory reasons *helped*, by +0.033, so the negative cannot be blamed on it.
(3) The scaling curve is flat: 150M lands marginally below 35M at 2.6× the
extraction cost. ESM-2 650M is cached but **excluded on a measured memory
constraint** (5.06 GB peak RSS alongside the concurrent production fold on a
16 GB machine).

**Cost: $0 cloud.** Head inference is 0.0150 s per 1,000 rows for the 30-network
ESM ensemble against 0.0656 s for the sequence ensemble; end to end, including
embedding 1,000 genuinely new peptides, 1.14 s against 0.073 s — about 16×.

## Stage 3b output (done — an unresolved measurement, not a null)

| Artifact | Description |
|---|---|
| [reports/stage3b_esm_multitask.md](reports/stage3b_esm_multitask.md) | The difference-in-differences, its measured power floor, the union-of-ladders audit |
| `preds/stage3b_{seq,esm,additive}_lam*.csv` | All 15 arm × λ prediction files |
| `scripts/stage3b_esm_multitask.py` | The harness; features imported from `scripts/esm_arm.py`, not reimplemented |
| `tests/test_stage3b.py` | 29 guards, including a structural check that the test split is never read |

Does an auxiliary affinity head help the **ESM-2** arm more than it helps the
sequence arm? The quantity is a difference-in-differences. At λ = 0.1 / 0.3 / 1 / 3
it is **+0.0120, +0.0122, +0.0143, +0.0281**, every interval crossing zero — and
the movement is almost entirely the sequence arm degrading rather than ESM
improving (the ESM-only arm's own Δ never leaves ±0.0041).

**This is reported as an unresolved measurement, not a null.** Every DiD upper
bound is below 0.05, so the frozen rule reads *"rules out a worthwhile gain"* —
a valid property of the intervals obtained. But the measured DiD floor
**straddles 0.05** (bracketed in (0.032, 0.071]), and an injected DiD of
**−0.0319 went undetected** while every observed DiD is smaller than that. So
**this design's power at the bar is unestablished**, and stage 3b must not be
cited as having been *able* to find a worthwhile differential. Two objections
are closed off: the auxiliary head genuinely trained on ESM features (head
ρ ≈ 0.57–0.58 at λ ≥ 0.1), and λ = 0 reproduces stage 3's declared headlines to
four decimals (additive 0.6761, ESM-only 0.6830), so the comparator is the
shipped model.

The same audit found a real error: the ESM ladders had been borrowing a 1e-5
point that only ever ran on a different feature matrix. Corrected to the true
union, all arms interior at 1e-2, two new tests enforcing it.

## Stage 3c output (done — external validation on a different assay)

| Artifact | Description |
|---|---|
| [reports/stage3c_elution_validation.md](reports/stage3c_elution_validation.md) | The pass, the decoy protocol declared in advance, three specificity controls |
| [elution_stability_finding.md](elution_stability_finding.md) | The motivating finding on the 140 overlapping pairs |
| `reports/stage3c_provenance.json` | Input hashes recomputed on every run |
| `pepstab/elution.py`, `scripts/stage3c_elution_validation.py` | The harness; takes an arbitrary score file |
| `tests/test_elution.py` | 25 guards |

A scoring pass with no retraining: rank MHC Motif Atlas eluted ligands above
length- and allele-matched human-proteome decoys, 10:1, 51 alleles, 81,600
ligands against 816,000 decoys.

| Arm | AUROC median [IQR] | AUPRC median | Enrichment top 1% |
|---|---|---:|---:|
| `seq_ensemble` (30 nets) | **0.9656** [0.9435, 0.9779] | 0.7804 | 10.33× (of a possible 11.0) |
| `seq_baseline` (single net) | 0.9502 [0.9182, 0.9648] | 0.6676 | 9.65× |
| *random scores, same harness* | *0.4974* | *0.0911* | *0.98×* |

The random-score null returns chance on every metric, which is what establishes
that the number comes from the model and not the scoring code. Three controls
say what it means: allele-swapped decoys cost **0.050 AUROC** (0.9157), bounding
the generic-presentability share; a donor-distance gradient runs 0.8906 near vs
0.9562 far; and handing the model the **wrong allele's pseudosequence** costs
0.27 AUROC (0.6965) with top-1% enrichment collapsing to exactly 1.00×.

**This is not a measurement of stability-prediction accuracy** — elution has at
least four filters besides stability. **0.610 / 0.693 remain the numbers to
quote for accuracy.** Still open: the same pass on the ESM-2 and structural arms.

## Stage 4a / 4b output (done — historical two-chain diagnostics)

| Artifact | Description |
|---|---|
| [reports/stage4_benchmark.md](reports/stage4_benchmark.md), `reports/gpu_decision.csv` | Five-GPU hardware benchmark; A10 at $0.004/complex two-chain; H100 needed 3.01× to break even and measured 1.35× |
| `reports/boltz_pose_check.csv`, `reports/boltz_pilot.csv` | Two-chain pose pilot; median peptide CA RMSD 0.29 Å |
| `modal_app/boltz_*.py`, `modal_app/esmfold_*.py`, [docs/BOLTZ_PIPELINE.md](docs/BOLTZ_PIPELINE.md) | The historical harnesses |
| `tests/test_boltz_pipeline.py` | 40 guards |

Three implementation lessons carry forward: keep weights resident across a batch
(reloading per complex overstated cost by **4.6×**); measure separate container
allocations (between-container variance 73%, within ±1%); and verify peptide
geometry directly, because high confidence did not flag the central-bulge error.

## Stage 4b.1 output (done — 182-residue groove MSA cache)

| Artifact | Description |
|---|---|
| [reports/stage4b1_msa_cache.md](reports/stage4b1_msa_cache.md) | Route, verified Boltz constraints, parse-cost benchmark |
| `reports/msa_manifest.csv` | Committed record: 75 alleles, `sha256` per HLA sequence and per MSA file |
| `reports/msa_parse_benchmark.json` | `parse_csv` cost vs `--max_msa_seqs` |
| `scripts/make_msas.py` | Regenerates the cache (needs boltz; see the report) |
| `structures/msa/<stem>.csv` | The MSAs — **gitignored**, 141.3 MB, regenerable |
| `tests/test_msa_cache.py` | 14 guards on manifest/cache consistency |

All **75** 182-residue grooves cached, not just the six panel alleles: 137 s of
CPU, **$0**, no GPU booked. Measured findings for that cache — the parse cost at the
default `--max_msa_seqs 8192` is ~$0.25 across 2,000 complexes, so **do not trim
for cost**; `--subsample_msa` defaults to *False* despite its help text;
and the C67S pseudosequence collision does not reach this arm, because the full
domains differ. This cache supplies pilot arm A, not the 275-residue HLA input.

## Stage 4c output (pilot done; production **running**)

| Artifact | Description |
|---|---|
| [reports/stage4c_ectodomain_pilot.md](reports/stage4c_ectodomain_pilot.md) | Executed procedure, the per-model gate verdicts, the launch sequence, the stage 5 feature contract |
| `reports/ectodomain-20261004/` | Input provenance, both CPU preflights, pilot pose scores and verdict, the live `production_<profile>.jsonl` shard logs |
| `data/structural_cohort.csv` | **The frozen production cohort** — load it, never recompute it |
| `scripts/ectodomain_inputs.py`, `scripts/ectodomain_pose_check.py`, `scripts/freeze_structural_cohort.py` | Input builder, pose scorer, cohort freeze |
| `modal_app/ectodomain_production.py` | The production runner, with its own `::smoke` entrypoint |
| `tests/test_ectodomain_inputs.py`, `tests/test_structural_cohort.py` | 4 + 8 guards |

The matched 90-fold pilot (3 constructs × 5 complexes × 3 seeds × 2 models) cost
**$1.49** and settled the engine choice on measurement:

| Gate criterion, fixed before the pilot | Boltz-2 | ESMFold2 |
|---|---|---|
| 45 valid predictions, full PAE and pLDDT, verified register | pass | pass |
| All arm-B peptide CA RMSD ≤ 2.0 Å, anchors ≤ 1.0 Å | pass | pass |
| Sentinel `HLA-B*07:02`/IPRRNVATL median heavy RMSD ≤ 1.0 Å | **pass (0.318)** | **fail (2.564)** |
| Sentinel improves ≥ 0.5 Å over the groove-only arm | **pass (−1.982)** | **fail (−0.008)** |

ESMFold2 is dropped from production as a **labelled revision of the matched
plan** — the cross-model comparison it was designed to deliver was already
delivered by the pilot on identical inputs, and at 4.55× the cost per fold its
$884 full-cohort forecast never fitted the budget. **Pose accuracy is not
feature utility, and a five-complex gate is an operational rule, not a general
claim about ESMFold2 on peptide–MHC.**

A separate pilot finding governs stage 5: **Boltz-2's own confidence does not
catch its pose failure.** The three failing folds sit at PAE 1.55 and pLDDT
0.973–0.975 while being 2.3 Å wrong, and the *correct* three-chain folds sit at
worse PAE. Keep confidence as a feature; **never use it as a per-prediction
failure filter.**

**Production, live as this is written.** Boltz-2 only, all 28,166 pairs, one
prediction each at seed 0, interleaved pair-by-pair across the `a-cheparukhin`
and `colleague` Modal profiles so either half alone stays balanced across
alleles and splits. 141 shards of 100 pairs per profile, 10 A10G workers each.
Launched 04:00 BST; **251 of 282 shards committed, 25,100 pairs, zero
failures**; forecast $200.8 ($251 with the required 25% margin). Each profile
folds **its own half** — `--profile` must match `MODAL_PROFILE`, and the runner
asserts it.

## Stage 4c.5 output (extractor done; full pass waits on the fold)

| Artifact | Description |
|---|---|
| [reports/stage4c5_features.md](reports/stage4c5_features.md) | The verified token mapping, feature definitions, the seed/arm variation study |
| `reports/stage4c5_pilot_features.csv` | 90 pilot folds × 129 columns (20 key/provenance, **109 numeric features**) |
| `reports/stage4c5_feature_variation.csv` | Seed / arm / between-complex spread per feature |
| `reports/stage4c5_features_production_<profile>.csv` | The live partial extract, 1,000 rows per half |
| `pepstab/structural_features.py`, `scripts/extract_structural_features.py`, `modal_app/feature_extraction.py` | Parsing and mapping verification; `pilot` / `extract` / `concat` CLI; the CPU-only Modal phase |
| `tests/test_structural_features.py` | 36 guards |

Every chain boundary is **re-derived per fold from the mmCIF** rather than
hard-coded, and `verify_fold` raises on any disagreement — a silently-wrong
slice yields plausible numbers for the wrong molecule, which is the one failure
mode that would survive into a results section. The load-bearing claim (token
index *i* is the *i*-th CA atom in mmCIF order) is checked exactly against the
pLDDT written into the B-factor column: max absolute disagreement **0.005** over
all 90 folds, against a 0.01 tolerance.

**What remains:** the full extraction pass once the fold completes, then the
stage 5 ablations. Neither has run, and no structural accuracy number exists yet.

## Stage 5 output (inverse folding pilot done; structural ablation not started)

| Artifact | Description |
|---|---|
| [reports/stage5_inverse_folding.md](reports/stage5_inverse_folding.md) | The ProteinMPNN pilot, the circularity control, the reframe, the predeclared QC protocol |
| `pepstab/inverse_folding.py`, `scripts/proteinmpnn_score.py`, `modal_app/proteinmpnn_scoring.py` | Scoring (teacher-forced, not sampling), the CLI, the CPU-only Modal app |
| `scripts/stage5_structural_arm.py` | The structural ablation harness — **pre-built and dry-run, waiting on 4c.5** |
| `tests/test_inverse_folding.py` | 13 guards |

ProteinMPNN closes the third of the three model classes the brief names.
**The finding is a reframe, not a feature.** The peptide log-likelihood is *not*
a half-life predictor — Spearman(`pep_ll_total`, t½) = **−0.100, p = 0.87,
n = 5**, which carries no inferential weight at all. It behaves instead as an
unsupervised, **crystal-free detector of Boltz-2 peptide-pose failure**, which
the project otherwise lacks entirely.

**Three things travel with that claim and none is optional.** (1) **n is ONE
complex, not six folds** — all six folds above 2.0 Å heavy RMSD are the same
complex in arms A and C at three seeds each. (2) The within-complex control is
the argument: the same complex folded correctly in arm B moves its score from
−3.06 into the good range at −2.39, and on the known-wrong arm-A backbone the
native sequence's +13.15 nat advantage collapses to roughly zero — so this is
not a tautological readback of the folder's own input. (3) The claim is
**specificity, not separation**: searching ~100 features for one that splits a
six-fold group always succeeds, and **none of the 109 structural features
isolates the failure**; `pep_ll_mean` was a single predeclared quantity fixed
before any score existed.

**Decision: no regression feature, buy the ~$1 QC sample** (2,000 folds, $1.04;
the full $14.68 pass is declined). What will be reported is a distribution and a
triage list, **never a failure count or rate**, and the reference points must
not filter the production cohort.

Two of our own controls disagree about ESMFold2's seed stability — ProteinMPNN
puts it at 1.2–1.9 while the 109 structural features put it at 5.87 — and the
disagreement is reported as a finding about granularity, not resolved by picking
the convenient one. Nothing material turns on it; production is Boltz-2 only.

## Stage 6 output (machinery done and exercised; test scoring **not run**)

| Artifact | Description |
|---|---|
| [reports/stage6_evaluation_machinery.md](reports/stage6_evaluation_machinery.md) | The analysis library, every plan count verified against the data, a worked example on validation |
| [reports/stage6_val_esm_run.md](reports/stage6_val_esm_run.md) | The four analyses the stage 3 headline does not contain, run on all five arms |
| [reports/TEST_SCORING_RUNBOOK.md](reports/TEST_SCORING_RUNBOOK.md) | The single-pass procedure, written before any test number existed |
| `pepstab/stage6.py`, `scripts/stage6_report.py` | N prediction CSVs in, the whole table set out |
| `reports/stage6_plan_verification.csv`, `reports/stage6_val_*.csv` | Every plan claim stated vs measured; the worked example |
| `tests/test_stage6.py` | 35 guards, including row-for-row parity with `pepstab/evaluation.py` |

**What the machinery bought, counted honestly** across the eight paired
comparisons, twelve stratum intervals and the nested row of the ESM run:

| Analysis | Comparisons | Conclusive |
|---|---:|---:|
| median per-allele ρ — **the contract's primary metric** | 8 | **0** |
| mean per-allele ρ | 8 | 4 |
| **differential concordance** | 8 | **5** |
| distance strata (gap + within) | 12 | 0 |
| precision@10 median | 4 | 0 |
| nested mutant ranking | 3 | 0 |

The differential target resolves more comparisons than every other analysis
combined, and the contract's own primary metric resolved nothing — a finding
about the contract, not only about the arms. **We predeclared the median and we
keep it as primary**; the recommendation to predeclare differential concordance
instead is for the next study.

Three further results from the validation run:

- **The distance question was asked three ways and the split answered none.**
  Stratum gap, within d=4 and within d≥5 — eleven of twelve intervals are in and
  not one separates any arm from the baseline. The half-width is near 0.09 on a
  quantity whose largest observed value is 0.069, so this is a fact about the
  validation split's size, not about the arms.
- **Nested mutant ranking is an exact dead heat**, verified pair by pair: the
  sequence baseline and ESM-2 both score 0.485075 on 490 comparisons over 96
  clusters, +0.0000 [−0.0433, +0.0445] — a net of 60 offsetting disagreements
  split exactly 30–30, not two identical vectors. The mean-pooled 650M ablation
  scores **conclusively below chance** (−0.1095 [−0.1712, −0.0384]), which is
  the positive control that makes the null worth anything.
- **One of our own predeclared metrics does not work.** Precision@10 at 2 h
  returned a delta of exactly 0.0000 on all four comparisons, with interval
  bounds that are lattice points of the statistic. It moves in steps of 0.1
  because it is ten slots. **It stays in the report** — removing a metric after
  finding it unflattering is the post-hoc selection the contract exists to
  prevent — but **no claim rests on it.**

**Not all nulls are equal.** The nested ranking and the differential have
positive controls on the same measurement; the distance strata, precision@10 and
stage 3b do not, and stage 3b's partial control *fails*. A reader should
discount those three accordingly.

## Stage 7 output (done — two stretch evaluations)

| Artifact | Description |
|---|---|
| [reports/stage7_censored.md](reports/stage7_censored.md) | Left-censored (Tobit) likelihood for the assay floor, predeclared before any fit |
| [reports/stage7_allele_holdout.md](reports/stage7_allele_holdout.md) | Leave-allele-out under a **second, separate contract** |
| `pepstab/censored.py`, `scripts/stage7_censored.py`, `preds/stage7_{censored,mse,censored_devmse}.csv` | The loss, the sweep, the arms |
| `pepstab/allele_holdout.py`, `scripts/stage7_allele_holdout.py`, `preds/stage7_allele_holdout_seq_pep_pseudo.csv` | 68 folds, 6 networks per fold, features in — not a `preds/*.csv` |
| `tests/test_censored.py`, `tests/test_allele_holdout.py` | 40 + 36 guards |

**7a — the censored likelihood fixes the floor and loses the ranking.**

| | median per-allele ρ | MAE log1p | mass below the limit | ECE |
|---|---:|---:|---:|---:|
| `log1p`-MSE (the baseline) | **0.6931** | **0.4734** | 0.070 | 0.0561 |
| censored @ 0.1 h | 0.6518 | 0.5763 | **0.168** | **0.0422** |
| *observed* | — | — | *0.196* | — |

**Δ = −0.0414 [−0.0780, −0.0062]** — the interval lies entirely below zero, so
this is the one **"worse, conclusively"** verdict in the project. It loses on
its own objective too (held-out censored NLL 1.0765 against 1.0452), the
threshold is not load-bearing (median ρ spans 0.0095 over a six-fold range of
the detection limit), and it delivered exactly the calibration it promised while
floor *discrimination* did not move (AUROC 0.8862 vs 0.8853) — ranking inside
the tied floor block is unidentifiable under either objective. The undertraining
control is reported **as a control**, with no interval; promoting it would be
the post-hoc selection this stage exists to avoid. **What stands is the frozen
`log1p`-MSE baseline.**

**7b — leave-allele-out exposes real headroom, under its own contract.** 68
folds, one per eligible allele, `split in {train, val}` only; the frozen test
split was never read and `data/splits.csv` is unmodified.

| Stratum (pseudosequence Hamming to nearest training allele) | Alleles | Median ρ | 95% CI |
|---|---:|---:|---|
| near, d ≤ 1 | 25 | **0.741** | [0.642, 0.803] |
| intermediate, d = 2–3 | 23 | 0.576 | [0.458, 0.726] |
| distant, d ≥ 4 | 20 | **0.339** | [0.298, 0.491] |

**near − distant = +0.403 [+0.223, +0.478].** The extreme contrast is solid; the
**monotone three-bin trend is not** — near − intermediate is +0.166 [−0.010,
+0.327], crossing zero, and the distant bin's best allele (0.753) beats the near
bin's worst (0.419).

Two constraints travel with every number above. The confound is **panel
composition, not panel hold-out** — the plan's original mechanism was measured
and does not hold (the median peptide sits on 4 alleles; allele-exclusive
peptides are 6.0% of rows) — and partialling zero-share out leaves distance at
−0.604 against −0.636 raw: **attenuated, not eliminated**. And **a `preds/*.csv`
cannot be scored through this contract**: those files come from models fitted on
every allele, so scoring one here would report the leak as pan-allele
generalisation. Each arm must supply features and be refit across all 68 folds;
the runner rejects a prediction file and a test guards it.

These numbers are **not comparable to a frozen-split number** and must never be
quoted beside one.

## Quick start

```python
from pepstab.data import load_with_splits

df = load_with_splits()                 # never recompute the splits
train = df[df.split == "train"]
# inputs: peptide, hla_seq, hla_pseudoseq   target: y_log1p
```

Write predictions as `pair_id,y_pred` on the `log1p` scale, then:

```bash
.venv/bin/python scripts/evaluate.py --split val \
    preds/seq_ensemble_pep_pseudo.csv preds/esm_ensemble.csv
```

First file is the baseline; the rest get paired cluster-bootstrap CIs against
it. `--per-allele` for the full table, `--by-distance` for distance strata.

For the full stage 6 table set — distance strata, the differential target,
nested mutant ranking, precision@10 — use the stage 6 CLI instead:

```bash
.venv/bin/python scripts/stage6_report.py --split val \
    --n-boot 2000 --stratum-min-rows 10 --nested \
    seq_ensemble_pep_pseudo=preds/seq_ensemble_pep_pseudo.csv \
    esm_ensemble=preds/esm_ensemble.csv \
    esm_plus_seq_ensemble=preds/esm_plus_seq_ensemble.csv
```

The first arm is the baseline every paired interval is taken against. Nothing is
hard-coded about the sequence baseline — a structural prediction file drops
straight in. `--stratum-min-rows 10` is the validation-only bar, chosen from row
counts and coverage before any model was scored in a stratum; **test keeps the
frozen 20.**

## Rules

- `data/rasmussen_et_al_dataset.csv` is read-only;
  `(cd data && shasum -a 256 -c SHA256SUMS)` must pass.
- Load splits from disk. Regenerating breaks peptide-cluster grouping and leaks
  training data into test.
- Select on validation. **Test is scored once, at stage 6** — it has not been
  scored yet.
- Ensemble every arm identically, or ensemble none of them. Ensembling alone is
  worth +0.074 mean SCC from no new information.
- Augmented rows are assumptions, not measurements: they carry `thalf_hours = 0`
  in `data/augmentation/`, enter the fit set only, and never touch validation or
  test. Verify any manifest with `pepstab.augment.verify_manifest` before
  training on it.
- Load the frozen production cohort from `data/structural_cohort.csv`, never
  recompute it. Each Modal profile folds **its own half**.
- No batch GPU job without a passing pilot on 3–5 examples. That applies to a
  new runner as well as a new model.
- A confidence interval that crosses zero is **inconclusive, not negative**. A
  null is only worth something next to a positive control on the same
  measurement.

## Regenerating

```bash
.venv/bin/python scripts/make_splits.py        # deterministic; rewrites data/splits.csv
.venv/bin/python scripts/audit_data.py         # rewrites reports/audit_summary.md
.venv/bin/python scripts/baseline_constant.py  # constant reference baselines
.venv/bin/python scripts/baseline_sequence.py  # stage 2 grid, ~10 CPU-minutes
.venv/bin/python scripts/baseline_ensemble.py  # 30-network ensemble baseline, ~1 min
.venv/bin/python scripts/compare_to_paper.py   # NetMHCstabpan calibration, ~3 min
.venv/bin/python scripts/augment_affinity.py   # stage 2b manifests, ~40 s (downloads a proteome)
.venv/bin/python scripts/baseline_augmented.py # stage 2b arms + intervals, ~13 min
.venv/bin/python scripts/stage2b_negatives.py  # stage 2b pool characterisation, ~10 s
.venv/bin/python scripts/affinity_multitask.py # stage 2c probe, ~10 CPU-minutes
.venv/bin/python scripts/affinity_multitask.py --protocol ensemble   # ~25 min
.venv/bin/python scripts/esm_features.py --verify-index              # stage 3 contact indexing
.venv/bin/python scripts/esm_features.py --checkpoint esm2_t12_35M_UR50D
.venv/bin/python scripts/esm_arm.py sweep --checkpoint esm2_t12_35M_UR50D
.venv/bin/python scripts/stage3b_esm_multitask.py      # stage 3b; exact flags in its report
.venv/bin/python scripts/stage3c_elution_validation.py # stage 3c external pass
.venv/bin/python scripts/stage6_report.py --split val ...   # stage 6 tables
.venv/bin/python scripts/stage7_censored.py            # stage 7a, 130 networks, ~30 CPU-minutes
.venv/bin/python scripts/stage7_allele_holdout.py      # stage 7b, 68 folds × 6 networks
.venv/bin/python reports/figures/make_figures.py       # figures, CPU, no network
.venv/bin/python -m pytest tests/ -q                   # 458 guards
```

Stage 3's full arm selection (`sweep` → `grid` → `ensemble`) and stage 3b's two
invocations have more flags than fit here; both reports give the exact commands
that produced the committed artifacts.

The MSA cache needs boltz, which pulls torch, so it is kept out of `.venv`:

```bash
uv venv /tmp/boltzenv --python 3.12
uv pip install --python /tmp/boltzenv/bin/python boltz
/tmp/boltzenv/bin/python scripts/make_msas.py --all-alleles   # ~2.5 min, $0
```

Structural production and feature extraction run on Modal, not locally — see
[reports/stage4c_ectodomain_pilot.md](reports/stage4c_ectodomain_pilot.md) for
the launch sequence. **A production fold is live as this is written; do not
touch `modal_app/`, `structures/`, or `data/structural_cohort.csv`.**
