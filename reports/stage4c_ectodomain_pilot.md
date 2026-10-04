# Stage 4c: matched ectodomain pilot

4 October 2026. **Status: the 90-fold matched pilot is complete. Boltz-2 passed
its gate; ESMFold2 failed. Production is frozen as Boltz-2, arm B, the full
28,166-pair cohort, split across two Modal workspaces. It has not been
launched.**

Both models received the same A/B/C constructs and prepared MSA rows. The pilot
was five training complexes × three arms × three seeds × two models = 90
predictions, all of which completed (`status=ok` on every scored row). A and C
have 191 residues; B has 383.

| Pilot arm | Construct | HLA alignment |
|---|---|---|
| A | Groove + peptide | Existing groove MSA |
| B | Ectodomain + beta2m + peptide | Selected ectodomain MSA |
| C | Groove + peptide | The same selected rows as B, cropped to 182 query columns |

## Inputs and checks completed

- All 75 ectodomain MSAs are prepared: 70 new CPU-only, unpaired ColabFold
  searches plus five validated/normalized supplied alignments. All 75 reach the
  shared 1,024-row cap, so MSA depth does not vary across the cohort.
- All source bytes and sequence/provenance flags are retained.
- Canonical MSAs use a shared 1,024-row cap. CSV and A3M carry the same rows;
  keys are unpaired. Full raw alignments are retained.
- Both actual CPU preprocessors passed the B/C groove-feature equality checks.
  Cross-model decoded residue rows and deletion features also match (30 checks).
- ESMFold2 used `msa_max_depth=1024`, `msa_column_mask_rate=0.0`, 20 loops,
  200 diffusion steps, one sample, and its default LM dropout 0.3. Boltz used
  three recycling steps, 200 diffusion steps, one sample, and no MSA subsampling.
- ESMFold2 cached weight revision is
  `69869f737beffec5294845ede23db5fc0b4f509e`; the downloader is now pinned.
- The raw dataset checksum passed. Existing pipeline/cache tests: 48 passed,
  6 skipped; new A3M input tests: 4 passed.

## Gate verdict

The gate was fixed before the pilot and applied separately to each model.
`reports/ectodomain-20261004/pilot_verdict.json` holds the machine-readable
result; the table below was recomputed independently from
`pilot_pose_scores.csv` and agrees with it.

| Gate criterion | Boltz-2 | ESMFold2 |
|---|---|---|
| 45 valid predictions, full PAE and pLDDT, verified register | pass | pass |
| All B: peptide CA RMSD <= 2.0 A, P2/P9 <= 1.0 A | pass (max 0.37 / 0.17 / 0.21) | pass (max 1.57 / 0.39 / 0.32) |
| Sentinel B*07:02 median B heavy RMSD <= 1.0 A | **pass (0.318)** | **fail (2.564)** |
| Sentinel improves >= 0.5 A over A | **pass (-1.982)** | **fail (-0.008)** |
| No other complex > 0.5 A worse than A | pass | pass (worst +0.333) |
| **Model verdict** | **pass** | **fail** |

Median peptide heavy-atom RMSD (A) over three seeds, scored in the same groove
reference frame (fit on HLA CA residues 1-182, then measure the peptide):

| Complex | Boltz A | Boltz B | Boltz C | ESM A | ESM B | ESM C |
|---|---:|---:|---:|---:|---:|---:|
| A*02:01 LLWNGPMAV | 0.270 | 0.228 | 0.264 | 0.930 | 0.738 | 0.784 |
| A*11:01 KTFPPTEPK | 0.930 | 0.625 | 0.852 | 1.020 | 1.107 | 1.013 |
| B*07:02 IPRRNVATL | 2.300 | **0.318** | 2.263 | 2.572 | **2.564** | 2.573 |
| B*08:01 ELRRKMMYM | 0.701 | 0.368 | 0.634 | 0.672 | 1.004 | 0.667 |
| B*15:01 ILGPPGSVY | 0.496 | 0.424 | 0.517 | 1.051 | 1.235 | 1.085 |

**Boltz-2: B improves and C does not.** C uses the same new alignment rows as B,
cropped to the groove, and tracks A everywhere, including the sentinel
(2.263 vs 2.300). The benefit therefore comes from the full ectodomain + beta2m
construct together with its extra evolutionary information, not from the new
groove alignment. Under the plan's interpretation table this is the
"B improves; C does not" row: proceed with B. B versus C does not isolate
alpha3 from beta2m and is not a physical-mechanism claim.

**ESMFold2 shows no construct effect at all.** The sentinel sits at 2.37-2.61 A
in every arm and every seed, and B is worse than A on three of five complexes.
Its backbone passes (max B CA RMSD 1.57 A) while heavy-atom RMSD fails, so the
error is in side-chain placement, and its peptide confidence is weaker
throughout (median arm-B peptide pLDDT 0.912 vs Boltz's 0.987; peptide-to-groove
PAE 3.98 vs 1.32). This reads as a model limitation on this complex rather than
an input problem: the inputs were verified identical at the feature level.

### The sentinel failure, per seed

B*07:02/IPRRNVATL against 7LFZ, peptide heavy RMSD / CA RMSD (A):

| Model | Arm | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| Boltz-2 | A | 2.30 / 1.31 | 2.31 / 1.30 | 2.21 / 1.28 |
| Boltz-2 | **B** | **0.26 / 0.18** | **0.32 / 0.18** | **0.32 / 0.19** |
| Boltz-2 | C | 2.26 / 1.26 | 2.19 / 1.29 | 2.30 / 1.27 |
| ESMFold2 | A | 2.57 / 1.49 | 2.58 / 1.52 | 2.37 / 1.28 |
| ESMFold2 | B | 2.41 / 1.30 | 2.61 / 1.50 | 2.56 / 1.44 |
| ESMFold2 | C | 2.57 / 1.50 | 2.58 / 1.52 | 2.37 / 1.27 |

The arm-A error is a central-bulge error with correct anchors. Boltz arm A,
seed 0, per-position CA deviation (A):

```
I1 0.19   P2 0.15   R3 0.04   R4 0.15   N5 1.84   V6 3.39   A7 0.60   T8 0.16   L9 0.14
```

Arm B flattens it to a 0.28 A maximum. This reproduces across all three seeds.

### Confidence does not separate this failure

Across all 45 Boltz folds, confidence ranks error reasonably:
Spearman(peptide-to-groove PAE, heavy RMSD) = +0.635, Spearman(peptide pLDDT,
heavy RMSD) = -0.540; within arm A alone, +0.782 and -0.621. But it does not
isolate the bulge failure. The three failing arm-A sentinel folds sit at PAE
1.55 and peptide pLDDT 0.973-0.975 while being 2.3 A wrong, whereas the
corresponding arm-B folds are 0.26-0.32 A at *worse* PAE (1.88-1.99). A*11:01
arm A is more accurate (0.85-1.05 A) at higher PAE still (1.92-2.20). No PAE
threshold catches the failure without flagging accurate predictions.

**Consequence for stage 5:** keep confidence features, since they carry rank
information, but do not use them as a per-prediction failure filter.

An independent figure from a colleague
(`reports/ectodomain-20261004/colleague_boltz2_arm_comparison.jpg`, Boltz-2
only) reproduces the arm A versus arm B comparison, the central-bulge profile,
and the confidence observation on four of these five complexes; its fifth
complex is A*11:01/ATIGTAMYK (6JOZ, t½ 97.1 h) rather than our pinned
A*11:01/KTFPPTEPK (1X7Q, t½ 59.3 h), which is the harder of the two for both
arms. Its panel-c framing is narrower than it looks: confidence does rank error
across our 45 folds, it just cannot isolate this failure. The figure does not
cover ESMFold2.

## Measured resources

Steady-state medians, excluding each shard's first fold (which absorbs model
load). Peak GPU is the maximum over the pilot.

| Model | Worker | Arm A | Arm B | Arm C | Peak GPU | Rate |
|---|---|---:|---:|---:|---:|---:|
| Boltz-2 | A10G / 4 CPU / 24 GiB | 9.88 s | **16.76 s** | 9.89 s | 6.11 GiB | $1.4812/h |
| ESMFold2 | L40S / 4 CPU / 64 GiB | 9.24 s | 42.63 s | 9.05 s | 27.46 GiB | $2.6512/h |

Per arm-B fold that is **$0.0069 for Boltz-2 and $0.0314 for ESMFold2: a 4.55x
ratio**, slower and on a dearer card. Container overhead beyond the sum of fold
times was 2-6%; model load added ~57 s to each shard's first fold. Cgroup peak
memory is unavailable in these containers; live RSS samples are saved as
`host_probe_*.json` and are lower bounds, not whole-run peak estimates.

Pilot spend was **$1.49** metered (billing snapshots at 02:22 and 02:39 BST),
against the agreed $15 pilot allocation. The pilot app ran 02:27:21-02:35:33
BST and is stopped; no app or task remains running. Each batch used separate
containers, a 20-minute timeout, no automatic retries, and no production queue.

## Production decision: Boltz-2 only, as a labelled plan revision

ESMFold2 is **not** carried into production. This is a revision of the agreed
matched plan, recorded here rather than applied silently, and it rests on three
things:

1. **It failed its pre-registered gate** on both sentinel criteria, with no
   construct effect in any arm or seed.
2. **The matched cross-model comparison has already been delivered** by this
   pilot, on identical constructs, MSA content, complexes and seeds. Production
   ESMFold2 folds would answer only the secondary question of whether worse
   poses still carry downstream signal.
3. **Cost.** At 4.55x per fold, ESMFold2 over the full cohort forecasts at $884
   against Boltz-2's $194 — and the two together never fitted the production
   ceiling in the first place.

What this does not establish: pose accuracy is not the same as feature utility,
so this does not prove ESMFold2 features would predict half-life worse; and a
five-complex gate is an operational decision rule, not a general claim about
ESMFold2 on peptide-MHC. Both limits belong in the write-up.

## Frozen production scope

Arm B, Boltz-2, seed 0, one prediction per pair: **all 28,166 pairs** of
`data/splits.csv`, frozen with their shard schedule in
`data/structural_cohort.csv` by `scripts/freeze_structural_cohort.py`. Pairs are
joined on `(allele, peptide)`; `pair_id` is attached afterwards. Splits are
loaded, never recomputed.

The cohort is interleaved pair-by-pair across two Modal profiles, so each half
is balanced across alleles and splits and a partial result from either stays
unbiased:

| Profile | Workspace | Pairs | Shards | Forecast | Wall at 10 workers |
|---|---|---:|---:|---:|---:|
| `a-cheparukhin` | a-cheparukhin | 14,083 | 141 | 67.8 GPU-h, $100.4 | 6.8 h |
| `colleague` | sofyaleyn | 14,083 | 141 | 67.8 GPU-h, $100.4 | 6.8 h |
| **Total** | | **28,166** | **282** | **135.6 GPU-h, $200.8** | **~6.8 h in parallel** |

With the required 25% margin: **$251 total, $125.5 per profile** against the
$150 per-workspace ceiling. Both workspaces hold approximately $300 of credit.
Forecasts use the pilot's measured 16.76 s steady fold and 57 s shard startup
at the $1.4812/h shape — the run that gated this configuration.

**The production `boltz predict` command is byte-for-byte the pilot's**, and
the smoke reproduces the pilot's numbers on it: 5/5 ok, 17.0 s steady folds,
6.08 GiB peak against the pilot's 16.76 s and 6.11 GiB. The configuration that
production runs is the configuration the gate was measured on. A
`--preprocessing-threads 1` option that had never actually executed was
removed rather than carried forward; the observed prediction order is still
recorded by the completion probe, which does not depend on it.

A shard is 100 pairs: one `boltz predict` invocation, ~29 minutes, which
amortises model load to ~3% while staying well inside the 60-minute function
timeout. 140 full shards plus one of 83 per profile.

Expected output is ~792 KB per prediction, so **~22 GB in total** (~11 GB per
workspace).

### Where the outputs live

Structures stay on Modal and are **not** downloaded. Each shard commits to the
`pepstab-structures` Volume in the workspace that folded it, under
`/stage4c/ectodomain-20261004/boltz2/production/<allele>/<complex_id>/`. The
only local artifact is `reports/ectodomain-20261004/production_<profile>.jsonl`,
one small line per shard.

A Modal Volume belongs to one workspace, so there are two: `a-cheparukhin`
holds one half and `sofyaleyn` the other. There is no cross-workspace Volume.
To give the team access, invite them to **both** workspaces — Modal workspaces
support email invites, though a workspace must have a verified payment method
before it can invite members.

Analysis should therefore run **on Modal with the Volume mounted**, not against
a local copy: at 22 GB, pulling everything down to score poses is slower than
running the scorer next to the data. Pull individual files with
`modal volume get` when something needs eyeballing locally. If a single shared
location is later required, `CloudBucketMount` onto S3/R2 can be mounted from
both workspaces, at the cost of an external bucket and a Secret in each.

### Running it

`a-cheparukhin` already holds all 150 production CSVs on `pepstab-hla-msa` from
the pilot, so it needs **no upload**; `::setup` only hash-checks them against
the committed manifest. Only a fresh workspace needs the staged slice.

```bash
# sofyaleyn only: it has an empty pepstab-hla-msa
python scripts/stage_production_msas.py        # 29.6 MB arm-B slice
MODAL_PROFILE=colleague modal volume put pepstab-hla-msa \
    structures/ectodomain_production_msas /ectodomain_stage4c

# both workspaces, CPU only; the weight download is skipped when present
MODAL_PROFILE=<profile> modal run modal_app/ectodomain_production.py::setup  --profile <profile>

# end-to-end check on 5 real cases, ~2 min, ~$0.03 per workspace
MODAL_PROFILE=<profile> modal run modal_app/ectodomain_production.py::smoke  --profile <profile>

# the run
MODAL_PROFILE=<profile> modal run --detach modal_app/ectodomain_production.py::production \
    --profile <profile>
```

Re-running `production` is the resume path: only shards without a committed
success marker are folded again, so an interrupted run continues where it
stopped. `--dry-run` prints the plan and forecast without spawning GPUs; check
that forecast against the $150 per-workspace ceiling before launching.
`--profile` must match `MODAL_PROFILE`; the runner asserts this, because the
halves are disjoint and a mismatch would fold one twice and the other never.

**Verified by dry run in both workspaces** (`ap-zrWEjFduVhc55Ivp7ltGb7` on
a-cheparukhin, `ap-DKtON3889mN3pbGNYmKcyn` on sofyaleyn): the app deploys, all
four functions are created, and each profile resolves its own 141 shards /
14,083 folds. This matters for `sofyaleyn` in particular, because Modal
validates every function at creation time and a workspace without a payment
method cannot declare a `gpu=` function at all — that is now ruled out. Still
unverified: whether each workspace is actually granted 10 concurrent A10Gs. If
fewer are available the forecast scales linearly in wall time, not in cost.

**Smoke status.** `::smoke` passed on `a-cheparukhin`: **5/5 ok, 17.0 s steady
folds, peak GPU 6.08 GiB**, matching the pilot's 16.76 s and 6.11 GiB. It found
two real defects first, both of which would have broken the overnight run at
step one: the stale `--preprocessing_threads` option, and `download_weights`
running on an image that never had `ectodomain_common` added, so the container
died on import before executing anything. Modal imports the whole module in every
container; the earlier dry runs passed because they only exercised
`completed_shards`. Both are fixed.

`::smoke` has **not** yet been run on `sofyaleyn`. Its MSA slice is uploaded
and the app deploys there, but no GPU fold has been executed in that workspace.
Run it before launching that half.

## Feature contract for stage 5

Extraction runs on Modal with the Volume mounted, not against a local copy.

Verify token-to-chain mapping against the CIF. For the A/B/C entity order the
anticipated arm-B PAE shape is 383 x 383, with zero-based slices HLA 0:275,
beta2m 275:374, peptide 374:383. Verify this rather than hard-coding it; the
pose checker's convention is that the peptide is the last 9 tokens and the
groove the first 182.

Core features: peptide per-position pLDDT, the peptide/groove PAE blocks in
both directions, and groove contacts/burial defined over HLA residues 1-182,
using the same receptor atom selection throughout. Any full-HLA or
beta2m-specific feature gets a separate name and definition. Save global ipTM
as global ipTM — in a three-chain complex it covers interfaces other than
peptide-HLA. Use a Boltz pair-specific score only if the pinned model actually
emits it. Label extra confidence outputs separately.

Confidence features are worth keeping but are not a per-prediction failure
filter: across the pilot's 45 Boltz folds they rank error (Spearman +0.635 for
PAE, -0.540 for pLDDT) yet do not isolate the one real failure.

Produce finite feature rows, coverage and failure tables, and a sequence-model
fallback for unresolved structural failures. Check extraction and joins on the
pilot; run an evaluator smoke check on training examples only. Do not train or
estimate generalization from five pilot examples.

Keep the raw dataset and `data/splits.csv` unchanged. Select feature groups and
heads on validation; the frozen test set is scored once at stage 6. Compare
Boltz-2 structural features against the sequence and ESM-2 baselines on
identical train/validation/test rows, matching regression architecture,
ensemble size, and tuning budget across arms. There are no ESMFold2 production
features — that comparison is the pilot's, and is reported above. Report
primary results against the frozen cohort with the declared sequence fallback;
show common successful rows as a diagnostic, not as a retrospectively chosen
cohort. Reuse existing compatible baselines and embeddings; no ESM re-extraction
is required by the construct change.

Follow `EVALUATION.md`: median per-allele Spearman, secondary MAE and
precision@10, paired peptide-cluster bootstrap, and the predeclared 0.05 gain.
Treat the pilot as a pipeline/pose check with possible training-set recall, not
a structural accuracy benchmark, and limit A/B/C comparisons to their 90 shared
folds.

## Artifacts and reproducibility

- `structures/ectodomain_msas/manifest.json`: canonical per-allele row IDs,
  query/content hashes, provenance, and serialized input files.
- `structures/ectodomain_msas/generation.json`: the CPU search configuration,
  query list, generation version, and duration.
- `reports/ectodomain-20261004/preflight_*.json` and
  `cross_model_preflight.json`: actual preprocessing/control checks.
- `reports/ectodomain-20261004/pilot_pose_scores.{csv,json}` and
  `pilot_verdict.json`: all 90 scored predictions and both gate verdicts.
- `structures/ectodomain_pilot/ectodomain-20261004/`: exported outputs and
  per-prediction metadata; Modal volume `pepstab-structures`, path
  `/stage4c/ectodomain-20261004/`.
- `data/structural_cohort.csv`: the frozen production cohort and shard schedule.
- `reports/ectodomain-20261004/production_<profile>.jsonl`: per-shard production
  records, written as the run proceeds.
- `reports/ectodomain-20261004/colleague_boltz2_arm_comparison.jpg`: the
  independent Boltz-2 figure discussed above.

This section records what exists. Production outputs and feature/coverage
tables are listed because they are where those artifacts will land, not as
evidence that they already exist — the run has not been launched.

Generation used the existing local `boltz==2.2.1` CPU search environment;
inference and its preprocessing are pinned to `boltz==2.1.1`, weight revision
`6fdef46d763fee7fbb83ca5501ccceff43b85607`. The search version is recorded and
does not change the matched biological inputs between models.

The Boltz CLI initializes its random stream per batch; ESMFold2 accepts a seed
per case. Matching seed labels is provenance, not identical stochastic draws.
For the pilot, Boltz's `settings.input_order` records YAML creation order; its
parallel parser can produce a different internal record/prediction order, so
exact per-case random-stream replay is not established by that field. Both
runners record the observed prediction order explicitly, which is what makes
the order reproducible in the record; neither constrains preprocessing
threading, so both run Boltz's default and production matches the pilot.
