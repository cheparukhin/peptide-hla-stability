# Final plan: matched Boltz-2 and ESMFold2 ectodomain folding

Date: 4 October 2026. Status: shared MSAs, CPU preflight, and the matched
90-fold GPU pilot are complete. **Boltz-2 passed its gate; ESMFold2 failed on
both sentinel criteria, so production is Boltz-2 only** — see
[`reports/stage4c_ectodomain_pilot.md`](../reports/stage4c_ectodomain_pilot.md)
for results and section 6 below for the runner. Production has not been
launched. Sections 1-4 are kept as executed, and describe both models because
both were run. The authoritative scope, pilot gates,
and rollout decision are in
[HACKATHON_PLAN.md, stage 4c](../HACKATHON_PLAN.md#4c-ectodomain--beta-2-microglobulin-folding),
with allocations in its budget section. This checklist expands that same plan.

## Objective and production choice

Produce peptide-HLA structures for dissociation half-life prediction using three
separate protein chains: **HLA ectodomain, 275 residues; beta-2-microglobulin
(beta2m), 99 residues; peptide, 9 residues. Total: 383 residues per complex.**

Run **both Boltz-2 and ESMFold2 with the same experimental protocol**. Each
model receives the same chain sequences, chain order, prepared MSA rows,
unpaired alignment policy, peptide single-sequence input, complexes, and seeds.
Both run the three-arm pilot and then the same frozen full-construct production
cohort, subject to passing quality checks and a joint resource forecast.
The controls run on the pilot only. Engine-specific input adapters and internal
inference settings are recorded explicitly; they must not change the experiment.
An ESMFold2 single-sequence run against an MSA-assisted Boltz run would change
both the model and its inputs and is outside this primary comparison.

The useful result is a reproducible structural input for the prediction task.
Improved crystal agreement alone does not establish improved half-life
prediction or a physical unbinding mechanism.

## 1. Preserve and validate the supplied inputs

| Supplied file | Role |
|---|---|
| `hla75_ectodomain_b2m.csv` | Authoritative ectodomain and beta2m sequences for the 75 dataset alleles. |
| `phla_msa_a3m.tar.gz` | MSA source archive: 75 groove MSAs, five ectodomain MSAs, one beta2m MSA. |
| `hla_prot.fasta` | IPD-IMGT/HLA source sequences for provenance and sequence reconstruction. |
| `hla75_ectodomain_crystal_validation.csv` | Heavy-chain sequence-coherence check; it does not validate predicted peptide poses. |

Record SHA-256 hashes of all four supplied files and the raw dataset. Preserve
the supplied files unchanged. Track the sequence table and a provenance manifest;
retain the larger source files in durable storage with their exact hashes.
Record the preparation script version, donor rules, and retrieval information.

Record the IPD-IMGT/HLA release and source URL if available. If the release
cannot be identified immediately, record it as unknown and retain the exact
FASTA; an unresolved release label does not prevent using these pinned bytes.
Record beta2m provenance separately: UniProt P61769, mature residues 21-119.

The audit has confirmed that all 75 ectodomain prefixes equal the dataset's
182-residue grooves, that 69 ectodomains match the supplied FASTA at offset 24,
and that the three borrowed alpha3 segments and stated donor identities match.
Repeat these invariants in input preparation so later edits cannot change them.
Also verify one sequence per allele, lengths 275/99/9, identical beta2m across
rows, and complete coverage of the 28,166 unique dataset pairs.

Preserve the three C67S constructs exactly, including their allele identifiers.
Carry explicit sequence-provenance flags for the borrowed alpha3 alleles
`HLA-A*02:50`, `HLA-A*24:19`, and `HLA-B*08:03`. Include them in production and
report their results with those flags; borrowed alpha3 is an approximation to
the actual allele sequence.

## 2. Prepare shared MSAs and a verified control for both models

Reuse the existing groove MSAs for arm A. Validate and normalize the five
supplied ectodomain MSAs, and generate the missing 70 on a CPU worker. Retain
the supplied beta2m MSA if its query matches the final 99-residue sequence.
No MSA server calls run on a GPU worker.

Two supplied ectodomain queries already have 275 columns. Three have 276;
remove their final query column and its associated insertion sequence, then
verify the query against the final table. Count A3M query columns using
uppercase residues and gaps: lowercase letters are insertions. A terminal extra
column alone is not evidence of a shifted alignment or a crash. Regenerate an
alignment only if sequence, column-length, or content validation fails.

Prepare one canonical MSA row set for each allele and use it for both models.
Boltz consumes custom CSV; ESMFold2 consumes A3M through `MSA.from_a3m`.
These are serialization adapters for the same rows, not independent searches.
Also serialize the existing groove MSA for A identically for both models.

Prepare B and C together for each allele:

1. Crop a candidate control row at the first 182 query columns, preserving
   insertions associated with retained columns.
2. Select one ordered row set that survives Boltz's duplicate removal in both
   the full and cropped forms. Preserve the query as row zero. Deduplicate
   using the pinned Boltz parser's rule: remove gaps and uppercase the row.
   Check ESMFold2's parser too; if it imposes additional selection rules,
   choose the common row set before serializing either input. Reject a candidate
   if either its full or cropped form duplicates an accepted row.
3. Keep up to 8,192 common rows, or a smaller shared cap if required by either
   model's preprocessing. Set and verify the same effective cap for both models
   and within B/C. Write the selected full rows for B and their cropped
   counterparts for C; save the original row identifiers and counts.
4. For Boltz, use custom CSV MSAs with `sequence,key` columns. Set keys to `-1` for these
   independently generated, unpaired alignments. Do not invent cross-chain
   pairing from row numbers. Limit beta2m depth to at most the selected HLA
   depth so adding it does not increase the total MSA row count in the control.
   Write equivalent ESMFold2 A3M files from these same selected rows and verify
   its independent-chain assembly does not introduce cross-chain pairing.
5. Run both pinned input adapters and their MSA preprocessing on CPU. Within
   each model, assert identical B/C groove rows, deletion/insertion features,
   pairing indicators, and any derived profiles that model uses. Across models,
   verify the same biological rows, query columns, retained depth, insertion
   information, and unpaired policy are supplied. Architecture-specific tensors
   need not have identical layouts. Fix unexpected filtering, truncation, or
   pairing before the GPU pilot; record every applicable assertion.

This validation matters: applying the pinned parser's selection rule to the
supplied files after ordinary cropping changes 94-189 selected rows per allele,
including 95 for B*07:02. An identical query prefix alone does not make the
control exact. Use the same preparation policy in the pilot and production.

Save an MSA manifest with allele, query hash, source and final file hashes,
source query length, retained row count, crop policy, pairing policy, server and
generation version, and inference version. Store the actual generated files;
a future public-server search need not return identical alignments.

Custom MSA formats and pairing keys are described in the
[official Boltz prediction documentation](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md).
Validate behavior against **boltz 2.1.1**, the inference version used here.

## 3. Complete CPU preflight before GPU use

The current Boltz and ESMFold2 runners build two-chain inputs, and the existing
pose checker fits the whole supplied HLA chain. Adapt both input builders, the
shared pose scorer, feature adapters, and resumable launchers. The existing
benchmark functions are limited to one container and two-hour timeouts; they
are not overnight launchers.

Build and validate this Boltz input and its equivalent native ESMFold2 input
with the same semantic chain IDs and MSA content:

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: <ecto from the allele table>
      msa: /absolute/path/to/selected_ectodomain.csv
  - protein:
      id: B
      sequence: <99-residue beta2m>
      msa: /absolute/path/to/beta2m.csv
  - protein:
      id: C
      sequence: <9-mer peptide from the raw dataset>
      msa: empty
```

For pilot A and C, use HLA ID A and peptide ID C, without beta2m. Infer token
and residue mappings from actual output metadata and sequences, rather than
assuming that chain names equal consecutive array indices.

Pin `boltz==2.1.1` and weight revision
`6fdef46d763fee7fbb83ca5501ccceff43b85607`. Use one diffusion sample, three
recycling steps, 200 sampling steps, mmCIF output, full PAE, and a parse cap of
8,192 MSA rows. Omit `--subsample_msa` in this version to select its false CLI
default; record that effective setting and the featurizer's retained depth.
Omit `--use_msa_server` because the MSAs are already prepared.

For ESMFold2, pin `esm==3.4.1.post1` and the exact `biohub/ESMFold2` weight
snapshot and hashes. The existing downloader does not pin a revision; resolve
and record the cached snapshot revision before launching. Use three separate
`ProteinInput` entries in `StructurePredictionInput`, with the same selected
HLA and beta2m MSAs as Boltz and no peptide MSA. Enable MSA use explicitly;
the current runner defaults to `use_msa=False` and hard-codes seed 0, so both
must change. Probe and validate beta2m MSA handling and independent, unpaired
chains on CPU; do not assume that supplying two MSAs proves equivalent pairing.

Use one diffusion sample and 200 sampling steps for both models. Keep the
current pinned model inference settings: Boltz three recycling steps and
ESMFold2 `num_loops=20`. These are different internal algorithms, not an equal
recycling budget; describe this as a comparison at fixed model configurations.
Record precision and all effective defaults. Keep settings fixed across A/B/C
and production within each model. Save mmCIF, full PAE, pLDDT, and available
confidence scores from both through a shared output schema.

Preserve prior outputs. Identify every new output by model, run, construct,
MSA hash, pair and seed. Record seeds and the complete shard/input ordering: the stock
CLI seed initializes the process, not an independent random stream per pair.
Changing shard contents or skip behavior can change subsequent samples.

### Executed input configuration, 4 October

The pilot uses a shared cap of **1,024 rows**, supplied as the same canonical
CSV/A3M rows to both models. ESMFold2's default 1,024-row inference subsampling
and 10% column masking would otherwise change its conditioning; explicitly set
`msa_max_depth=1024` and `msa_column_mask_rate=0.0`, and leave Boltz MSA
subsampling disabled. Full raw alignments are retained for later depth studies.
Both CPU preflights passed the B/C groove-feature equality checks; decoded
cross-model groove rows and insertion/deletion features also match.
The cached ESMFold2 weights are pinned to revision
`69869f737beffec5294845ede23db5fc0b4f509e`.

Artifacts: `structures/ectodomain_msas/manifest.json`,
`structures/ectodomain_msas/generation.json`, and
`reports/ectodomain-20261004/preflight_*.json` with the cross-model verdict.
Missing 70 ectodomain searches were generated in one unpaired ColabFold request
on CPU; the five supplied ectodomain searches were validated and normalized.

## 4. Run the matched 90-fold pilot

Run five training-split complexes, three arms, and seeds **0, 1, 2** with
**each model: 45 Boltz-2 folds + 45 ESMFold2 folds = 90 predictions**.
The arm letters identify inputs; the model is a separate manifest field.
Pin one reference crystal for each complex before scoring predictions.

| Allele | Peptide | Reference PDB |
|---|---|---|
| HLA-A*11:01 | KTFPPTEPK | 1X7Q |
| HLA-A*02:01 | LLWNGPMAV | 5N6B |
| HLA-B*15:01 | ILGPPGSVY | 1XR9 |
| HLA-B*07:02 | IPRRNVATL | 7LFZ |
| HLA-B*08:01 | ELRRKMMYM | 4QRU |

These are the five complexes in `reports/boltz_pilot.csv`. The supplied
coherence table's 6JOZ contains ATIGTAMYK and is not the pose reference for
KTFPPTEPK. Re-score the new A predictions against the fixed references; the
old checker selected the lowest peptide RMSD among available crystals.

| Pilot arm | Construct | HLA MSA |
|---|---|---|
| A | 182-residue groove + peptide | Existing groove MSA |
| B | 275-residue ectodomain + beta2m + peptide | Selected ectodomain MSA |
| C | 182-residue groove + peptide | Same selected rows as B, cropped to the groove |

Start with one B prediction per model and verify chain placement, PAE/pLDDT
files, and feature extraction before completing the remaining pilot predictions. Use
resident models for batches, deterministic input ordering, and separately
recorded container allocations for the three seeds in each model. Record the combined
effects of diffusion sampling and container variation; this small pilot cannot
estimate each source of variation precisely.

Superpose all arms from both models onto the same available HLA CA atoms in residues 1-182.
Apply that transform to the peptide; never fit the peptide onto itself or fit
B on all 275 HLA residues. Score CA RMSD, heavy-atom RMSD, and per-position CA
deviation. Use identical residue/atom matching, altloc, and side-chain symmetry
policies across arms, and report missing-atom coverage. Verify the canonical
peptide register and placement of P2 and P9 in their pockets.

Record wall time, weight-load/startup overhead, steady-state B fold time, peak
VRAM, peak host memory, failures, output completeness, and billed worker time.
Do not use the old 191-residue cost or a theoretical 2-4x scaling bracket as a
measurement of the 383-residue workload.

### Pilot acceptance criteria

Apply the same operational thresholds separately to each model, chosen before
the new pilot; they are not a general estimate of pHLA prediction accuracy.
All B-versus-A comparisons below are within the same model and matched seeds.
Also report Boltz-versus-ESMFold2 differences on identical arm/complex/seed
inputs. Matching seed numbers supports provenance, not identical random draws.

- Every B prediction produces valid structures, full PAE, pLDDT, and a finite,
  correctly mapped feature row. No unresolved chain/register errors.
- All five B complexes at all three seeds have peptide CA RMSD <= 2.0 A and
  P2/P9 CA deviations <= 1.0 A.
- B*07:02/IPRRNVATL has median B heavy-atom RMSD <= 1.0 A and improves by at
  least 0.5 A over the new A median scored under the same protocol.
- On each of the other four complexes, median B heavy-atom RMSD is no more
  than 0.5 A worse than A. Inspect per-position results as well as averages.
- Resource forecasts pass the production rules below, with measured memory
  headroom on the chosen worker shape.

If either model's B fails a quality criterion, resolve or document the failure
before scaling the paired experiment. Do not silently drop that model, remove
its MSA, or substitute a different construct for it.
Changing a threshold after seeing the result requires a clearly labeled revised
pilot, rather than reporting the original gate as passed. If runtime or memory
is the only failure, choose a smaller scope or benchmark a suitable GPU.

### Interpret the controls without making C failure a launch condition

| Result | Interpretation and next action |
|---|---|
| B improves; C does not | The added construct and its extra evolutionary information help beyond changing the groove alignment. Proceed with B if its gate passes. |
| B and C improve | The new groove alignment may be sufficient. B may still proceed; retain C as a cheaper candidate for later prediction comparisons. |
| B fails; C improves | B has not passed its launch gate for that model. Investigate before scaling; a change to C would require a revised matched plan for both models. |
| Neither improves consistently | Investigate scoring, sampling, inputs, and model limitations before a large run. |

Within either model, B versus C does not separate alpha3 from beta2m, or
additional chain sequence from evolutionary information outside the groove. Report the benefit
of the full input pipeline; reserve physical mechanism claims for a more
controlled experiment. A fourth mechanistic arm is optional later work.

## 5. Choose and freeze the overnight workload

Before launching, save an absolute completion deadline in **Europe/London**, the
verified available GPU concurrency, current account credit, commitments to other
work, pilot spend, and chosen worker resources. Reserve time after folding for
export, feature extraction, and evaluation; do not assume an eight-hour window.

**As frozen, 4 October 2026.** Production allocations are **$150 per Modal
workspace** — `a-cheparukhin` and `sofyaleyn` (local profile `colleague`), each
holding approximately $300 of credit. Credit does not move between workspaces,
so each half must fit its own balance. ESMFold2 production is not funded; it
failed the 4c.3 gate. The 90-fold pilot spent **$1.49** of the **$15 total
pilot allocation**. These ceilings are not evidence of remaining credit, so
recheck rates and balances in each workspace at launch.

Hardware was chosen from measured full-construct throughput and memory: Boltz-2
on **A10G / 4 CPU / 24 GiB** at $1.4812/h, which the pilot measured at 16.76 s
per arm-B fold and a 6.11 GiB GPU peak. The 24 GiB host request is generous
against an observed ~10 GB child RSS, but these containers could not read a
cgroup peak for the 383-residue construct, so it is deliberately not trimmed
for the ~$8 it would save. ESMFold2's measured L40S shape (42.63 s, 27.46 GiB,
$2.6512/h) is recorded for the write-up only.

Still to record before launch: an absolute completion deadline in
**Europe/London**, and the verified GPU concurrency actually available in each
workspace. The plan assumes 10 concurrent A10G workers per workspace; the
forecast scales linearly if fewer are granted.

Estimate each model's production from its measured B throughput and billed
cost per successful fold. Include failures in that cost, plus amortized startup, expected restarts,
and export/storage costs. Use a **25% forecast margin** for both cost and time:

```text
forecast_cost = 1.25 * (N * steady_cost_per_success
                       + total_startup_cost + export_storage_cost)
forecast_time = 1.25 * ((N * steady_seconds_per_success
                        + total_startup_seconds) / available_workers)
                + export_and_feature_time
```

Calculate these forecasts separately for each model, then sum their costs.
For simultaneous runs, completion is the later model's finish time with verified
per-model worker allocations; shared account concurrency applies to their sum.
For sequential runs, sum elapsed times. Do not count the same GPU capacity for
both models at once. Include the shared export/evaluation stage once.

Do not add startup again if it is already included in a quoted per-success
cost. Use observed spread across the pilot allocations when choosing the
planning throughput; the margin is not a statistical guarantee. Verify actual
concurrency and throughput on the first production shards and refresh forecasts.

- **Preferred scope: all 28,166 pairs for each model (56,332 predictions)**,
  if both model forecasts and the combined cost/deadline fit. This preserves
  the full frozen evaluation population.
- **Fallback: the same approximately 2,000 pairs across six alleles for each
  model (4,000 predictions)**, if full scope does not fit jointly and the panel
  does. Reuse a valid frozen structural panel if present;
  otherwise select it deterministically from the existing splits without labels.
  Use the stage 4c.4 allele quotas: B*15:01 434, A*02:01 415, A*03:01 349,
  B*39:01 276, B*35:01 264, B*07:02 262. Preserve split proportions and verify
  >= 50 test and >= 20 validation rows per allele. Save the selection seed,
  algorithm, and manifest before folding; never regenerate the splits.
- If neither joint scope fits, retain the pilot result and record the same
  reduced scope for both models explicitly before doing further folds. Do not
  create a convenience subset afterward from whichever predictions finished first.

For illustration only, 28,166 folds in eight hours require about 20 workers at
20 seconds per fold or 40 at 40 seconds, before margins and overhead. These
figures apply to one model and do not assert available concurrency or measured
ectodomain throughput. The paired workload needs capacity for both models.

## 6. Run resumable production and export

### As implemented: `modal_app/ectodomain_production.py`

One app, Boltz-2, arm B, seed 0. The cohort and its shard schedule are frozen
in `data/structural_cohort.csv`; the runner reads that file and never re-derives
it. 141 shards of 100 pairs per profile, `max_containers=10`, 60-minute
function timeout against a ~29-minute expected shard, `retries=0` so every
failure is recorded rather than silently repaid.

```bash
python scripts/freeze_structural_cohort.py   # committed; re-running is a no-op
python scripts/stage_production_msas.py      # 29.6 MB arm-B CSV slice + manifest

# per workspace: upload inputs, pull pinned weights, verify, smoke, run
MODAL_PROFILE=<profile> modal volume put pepstab-hla-msa \
    structures/ectodomain_production_msas /ectodomain_stage4c
MODAL_PROFILE=<profile> modal run modal_app/ectodomain_production.py::setup      --profile <profile>
MODAL_PROFILE=<profile> modal run modal_app/ectodomain_production.py::smoke      --profile <profile>
MODAL_PROFILE=<profile> modal run --detach \
    modal_app/ectodomain_production.py::production --profile <profile>
```

- `--profile` selects which half of the cohort to fold and **must match
  `MODAL_PROFILE`**; the runner asserts this, because the halves are disjoint
  and a mismatch would fold one half twice and the other never.
- `::smoke` folds 5 real cases into a `_smoke` subtree and writes no shard
  marker. It satisfies the project invariant for this new runner. Do not skip
  it because the 90-fold pilot passed: that validated the inputs and the model
  configuration, not this sharding, resume, and feature-extraction code.
- `::production` is the resume path. Only shards without a committed success
  marker are folded, so re-running after an interruption or a partial failure
  continues where it stopped.
- `--dry-run` prints the plan and forecast without spawning GPUs.
  `--max-usd` (default $150) refuses to dispatch if the 25%-margin forecast for
  the remaining shards exceeds the per-workspace ceiling.
- Each shard writes `_features/shard_NNNN.jsonl` with peptide pLDDT and the
  peptide/groove PAE block means, so section 7 can begin from a few MB while
  the ~22 GB of structures download.

The requirements below are the general contract this implementation satisfies;
keep them if the runner is replaced.

Shard the frozen scope into bounded batches that finish inside function timeouts.
Configure container limits for the chosen concurrency; keep weights resident
across complexes and prepared MSAs on durable storage. Retain one diffusion
sample per model per production pair, with the pilot's model and shared MSA
policies. Use the same frozen pair manifest, deterministic ordering, and
predeclared seed schedule for both models; keep model-specific queues.

Use a durable run manifest keyed by `(model, allele, peptide)`, with the raw
dataset's `pair_id` attached after that join. Record input hashes, construct, MSA IDs,
seed/shard/order, settings, worker allocation, status, attempts, elapsed time,
output hashes, and errors. Keep outputs for each arm and run distinct.

Declare success only after validating the mmCIF, confidence files, and required
array shapes. A prediction directory alone is not proof of success. Checkpoint
after each shard, export completed outputs, and resume only unfinished or invalid
pairs. Record reruns and any changed seed/order provenance. Allow at most one
retry of a failed pair within the remaining resource forecast; retain unresolved
failures in the manifest. Do not drop a usable structure merely for low confidence.

Update completion and spending forecasts after the first shards and thereafter
at shard boundaries. Stop dispatching work if the chosen scope no longer fits;
preserve completed artifacts and report coverage against the frozen scope.
Copy inputs, MSAs, manifests, structures, and confidence outputs to durable local
storage before temporary event resources disappear.

## 7. Extract features and evaluate on frozen splits

Verify token-to-chain mapping against the CIF. For the standard A/B/C entity
order, the anticipated B PAE shape is 383 x 383, with zero-based slices HLA
0:275, beta2m 275:374, peptide 374:383. Verify this rather than hard-coding it.

Preserve comparable core features: peptide per-position pLDDT, the peptide/groove
PAE blocks in both directions, and groove contacts/burial defined over HLA
residues 1-182. Define burial using the same receptor atom selection across arms.
Any full-HLA or beta2m-specific feature receives a separate name and definition.
Save global ipTM as global ipTM; in a three-chain complex it includes interfaces
other than the peptide-HLA interface. ESMFold2's `pair_chains_iptm` is a separate
model-specific feature: verify the HLA/peptide pair mapping, anticipated entity
indices 0 and 2, against actual metadata. Use a Boltz pair-specific score only
if the pinned model actually emits it. Compare shared feature definitions first;
label extra confidence outputs separately so they do not confound the primary
model comparison.

Produce finite feature rows, coverage and failure tables, and a sequence-model
fallback for unresolved structural failures. Check extraction and joins on the
pilot; run an evaluator smoke check on training examples only. Do not attempt
to train or estimate generalization from five pilot examples.

Keep the raw dataset and `data/splits.csv` unchanged. Select feature groups and
heads using validation; the frozen test set is scored once at stage 6.
Compare Boltz-2 and ESMFold2 structural features, separately and optionally
combined if validation supports it, against sequence and ESM-2 baselines on
identical training, validation, and test rows. Match regression architecture,
ensemble size, and tuning budget across model feature arms. Report primary
results against the frozen cohort with the declared sequence fallback; show
common successful rows as a diagnostic, not as a retrospectively chosen cohort.
At full scope, reuse existing compatible baselines and embeddings. At panel scope, refit the inexpensive baseline heads
on the same panel training rows; also report existing full-training-data models
on that panel's held-out rows as practical comparators. No ESM embedding
re-extraction is required solely by the construct change.

Follow `EVALUATION.md`: median per-allele Spearman, secondary MAE and
precision@10, paired peptide-cluster bootstrap, and the predeclared 0.05 gain.
Treat the structural pilot as a pipeline/pose check with possible training-set
recall, not a general structural accuracy benchmark. Limit A/B/C comparisons
to their common folded rows; the pilot does not provide a dataset-wide structural
ablation.

## Required handoff

Save the input/provenance and shared MSA manifests; both CPU preflight results;
all 90 pilot predictions with model/arm/seed pose and resource scores; both gate
verdicts; the shared frozen production scope and separate plus combined
forecasts; production success/failure records;
exported structures and confidence arrays; and validated feature/coverage tables.
Record which steps actually completed. This plan is not evidence that the new
runner, MSAs, pilot, or production outputs already exist.
