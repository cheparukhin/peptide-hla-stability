# Compute-cost ledger

Every dollar, CPU-minute and GPU-hour this project has spent or forecast, with
the report each figure comes from. Machine-readable twin:
[`compute_ledger.csv`](compute_ledger.csv).

The challenge brief asks *"Are they useful for this problem"*, and usefulness
has a denominator. This file is the denominator. It exists so the accuracy table
in [`SUBMISSION.md`](SUBMISSION.md) can be read as a price list rather than a
leaderboard.

**Three labels are used throughout and never mixed:**

| Label | Meaning |
|---|---|
| **measured** | Observed wall time or a provider's metered bill. |
| **derived** | A measured quantity multiplied by a published rate, or by a known repeat count. The arithmetic is shown. |
| **forecast** | Not yet run. A projection from measured unit costs. |

Nothing in this file is an estimate of something nobody measured. Where a stage
did not record its own timing, the cell says so rather than carrying a guess.

---

## 1. Headline: what has actually been spent

| Provider / workspace | Allocation ceiling | Spent | Evidence |
|---|---:|---:|---|
| Modal `a-cheparukhin` | $15 pilots + $150 production | **$4.41 metered** | `reports/ectodomain_billing_after.json` (`metered_cost`), snapshot 02:39 BST 4 Oct |
| Modal `sofyaleyn` (`colleague`) | $150 production | **no GPU fold executed** | stage 4c: `::smoke` has not been run there; the app deploys and dry-runs only |
| Hugging Face | $60 | **pending stage 3** | HACKATHON_PLAN.md, "Budget and GPU decision rule" |
| Laptop CPU | — | **$0** | stages 1, 2, 2b, 2c, 3c, 4b.1 and all MSA preparation |

**$4.41 is the whole GPU bill for this project to date**, covering a five-GPU
hardware benchmark, two engine pilots and a 90-fold matched model comparison.
Production folding — the only large spend — is forecast and **has not been
launched**.

Two reconciliations a reader should be able to perform:

- `stage4_benchmark.md` computes its own two-chain spend as **~$2.64** from
  published per-second rates. The metered workspace total immediately before the
  stage 4c pilot was **$2.92** (`ectodomain_billing_before.json`). The ~$0.28
  difference is setup, CPU and volume work outside that report's table. The
  computed estimate is therefore accurate to roughly 10% against the bill, which
  is the only independent check we have that the computed figures elsewhere in
  this ledger are the right order.
- The 02:39 snapshot predates the production runner's `::smoke`. The stage 4c
  report estimates that at ~$0.03 per workspace; it is not in the $4.41 and is
  not metered anywhere we captured.

---

## 2. Per-stage ledger

CPU minutes are of a single laptop core unless stated. "—" means the stage did
not record that quantity.

### Sequence work (stages 1, 2, 2b, 2c, 3c) — CPU only, $0

| Stage | Item | CPU-min | Wall | $ | Source |
|---|---|---:|---:|---:|---|
| 1 | Data audit, frozen splits, evaluation contract | — | — | 0 | `audit_summary.md` |
| 2 | Baseline grid: 72 MLP + 36 ridge fits, 6 arms | 7.5 | 10 min | 0 | `stage2_baselines.md` §Cost |
| 2 | 30-network ensemble (the stage 3 comparator) | 1.7 | ~1 min | 0 | `stage2c_affinity.md` Result 2, λ=0 row (102 s) |
| 2 | NetMHCstabpan calibration | — | ~3 min | 0 | `README.md` |
| 2b | Manifests: predictor fit + 3.0M affinity predictions | 0.65 | 39 s | 0 | `stage2b_augmentation.md` §Cost |
| 2b | 21 network fits (7 arms × 3 seeds) | 1.3 | 1.3 min | 0 | same |
| 2b | 8 paired cluster bootstraps, 2,000 resamples each | 11 | ~11 min | 0 | same |
| 2b | Negative-pool characterisation | 0.17 | 10 s | 0 | same |
| 2c | Auxiliary-affinity probe, 30 networks | 1.5 | ~10 min | 0 | `stage2c_affinity.md` §Cost (92 s fitting) |
| 2c | Auxiliary-affinity ensemble sweep, 150 networks | 12.1 | ~25 min | 0 | same (12.1 min fitting) |
| 2c | Censoring-robustness variant, 30 networks | — | — | 0 | not separately recorded |
| 3c | Elution external-validation check | — | — | 0 | `elution_stability_finding.md` |

**Total recorded sequence-side compute: about 36 CPU-minutes of fitting and
bootstrapping, and $0.** The external-validation and inverse-folding pilots add
a further 16.4 and 28.3 CPU-minutes respectively, also at $0 — so **every
non-GPU result in this project cost about 95 CPU-minutes on one laptop**. In
both 2b and 2c the *uncertainty estimate* costs more
than the models: 11 of 2b's ~13 minutes are the paired cluster bootstrap. That
is the right way round for a submission judged on evaluation quality, and it is
worth saying out loud — the expensive part of this project's sequence arm is
measuring how confident we are, not fitting.

### Structural work (stages 4a/4b, 4b.1, 4c)

| Stage | Item | Hardware | GPU-h | Wall | $ | Status | Source |
|---|---|---|---:|---:|---:|---|---|
| 4b.1 | 75 groove MSAs (182 aa) | laptop CPU + public ColabFold | — | 137 s | 0 | measured | `stage4b1_msa_cache.md` |
| 4a/4b | Two-chain benchmark: 68 Boltz-2 folds over 5 GPU types, plus the ESMFold2 sweep | A10 / L4 / L40S / A100-40GB / H100 | — | — | **2.64** | computed from published rates | `stage4_benchmark.md` §Spend |
| 4c | 70 ectodomain MSAs (275 aa) | laptop CPU + public ColabFold | — | 126.7 s | 0 | measured | `ectodomain-20261004/input_provenance.json` (`generation.seconds`) |
| 4c | CPU preflight and cross-model input checks | laptop CPU | — | — | 0 | not separately recorded | `stage4c_ectodomain_pilot.md` |
| 4c | **Matched 90-fold pilot** — 45 Boltz-2 (A10G) + 45 ESMFold2 (L40S) | A10G + L40S | — | 8 min 12 s | **1.49** | **metered** | billing snapshots 02:22 / 02:39 |
| 4c | Production runner `::smoke`, 5 folds, `a-cheparukhin` only | A10G | — | ~2 min | ~0.03 | report estimate | `stage4c_ectodomain_pilot.md` |
| 4c | **Production fold, all 28,166 pairs** | A10G × 20 workers | **135.6** | ~6.8 h | **200.8** | **forecast** | `stage4c_ectodomain_pilot.md` |
| 4c | …with the required 25% operational margin | A10G × 20 workers | 169.5 | ~8.5 h | **251** | forecast | same |
| 4c | ESMFold2 production — **rejected at the gate, never funded** | L40S | — | — | 884 | forecast, rejected | same |
| 4c | Structure storage, ~22 GB on two Modal Volumes | Modal Volume | — | — | ~1.98 / month | derived | 22 GB × $0.09/GiB-month, `ectodomain_rates.json` |

The 90-fold pilot cost **$1.49 to settle an engine choice that would otherwise
have been settled by preference**. That is the cheapest decision in the project
and the one with the largest downstream consequence: it ruled out an $884
production run — on a pre-registered quality gate first, with cost as the third
of three supporting reasons. **The pilot cost 0.17% of the run it averted**, and
it was spent before any production compute was booked.

MSA preparation — the step most likely to be assumed expensive — cost **$0 and
under four minutes in total** across both constructs, because it runs on CPU
against a public server and is cached once per allele rather than once per pair.
All 75 alleles are cached, so the per-pair marginal MSA cost of the structural
arm is zero.

### Stage 3: ESM-2 embedding extraction — measured, local GPU, $0

`reports/stage3_embedding_cost.csv`, measured on the laptop's `mps` device.
Embeddings are cached **per unique sequence**, so 28,166 rows cost only 5,633
peptide embeddings and 75 HLA embeddings.

| Checkpoint | Peptide: s / 1,000 unique | HLA: s / 1,000 unique | Load (s) | Peak RSS | Cache |
|---|---:|---:|---:|---:|---:|
| ESM-2 35M (`t12_35M`) | **1.07** | 18.70 | 4.84 + 0.29 | 859 MB | 123.5 MB |
| ESM-2 150M (`t30_150M`) | 2.74 | 53.76 | 14.66 + 1.81 | 1,485 MB | 164.7 MB |
| ESM-2 650M (`t33_650M`) | 8.50 | 154.55 | 67.75 + 5.33 | 5,062 MB | 329.5 MB |

The whole dataset's embeddings cost **7.5 s at 35M and 59.5 s at 650M**
(`embed_seconds` summed), on a laptop, for $0. The 650M checkpoint's 5.06 GB
peak RSS is the only hardware constraint in sight, and it fits a laptop.

### Still to be ledgered

### Stage 3c: elution external validation — measured, CPU, $0

`stage3c_elution_validation.md`. 51 alleles, 897,600 rows scored three times
(proteome decoys, allele-swapped decoys, wrong-allele pseudosequence), plus a
random-score null through the same harness.

| Step | Time |
|---|---|
| Atlas download (8.0 MB) | ~3 s |
| Build scoring set: parse atlas, tile 10.4M proteome 9-mers, draw 816,000 decoys | 80 s |
| Refit `seq_baseline` (1 network) / `seq_ensemble` (30 networks) | 30 s / 394 s |
| Score the ensemble over 897,600 rows, once per decoy set | ~110 s each |
| **Total, three arms and all controls** | **16.4 min** |

Peak memory stays near 350 MB: the 897,600 × 860 feature matrix would be 3.1 GB
in one block, so prediction runs in 100,000-row chunks. Measured with three
other agents on the same 8-core laptop at `OMP_NUM_THREADS=3`; an uncontended
earlier run did the two model arms in 20.1 min including a separate set build,
so the figure is stable to a few minutes either way.

### Stage 5 inverse folding (ProteinMPNN) — pilot measured $0, QC sample forecast

`stage5_inverse_folding.md`. Everything executed so far ran on laptop CPU with
**$0 of cloud spend**.

| Run | Folds | Workers | Wall |
|---|---:|---:|---:|
| Pilot, native scores, all 90 stage 4c folds | 90 | 2 | 481 s |
| Cross-peptide circularity matrix, Boltz-2 arm B | 15 | 1 | 1,107 s |
| Wrong-backbone control, `B*07:02` arm A | 3 | 1 | 107 s |
| *Aborted cross-peptide sweep (killed under memory pressure, output discarded)* | *7 of 45* | *2* | *~780 s* |

The aborted sweep is recorded rather than quietly dropped — it is 780 s of
compute that produced nothing, and a ledger that only lists successful runs is
not a ledger.

**Forecast** from the measured 14.70 s mean per arm-B fold at 16 decoding
orders, priced at the repo's metered rates:

| | folds | core-hours | wall | cost |
|---|---:|---:|---:|---:|
| **QC sample, 16 orders — the approved buy** | **2,000** | **16.5** | **0.8 h** | **$1.04** |
| QC sample, 8 orders | 2,000 | 8.3 | 0.4 h | $0.53 |
| Both profiles, 16 orders — **declined** | 28,166 | 232.0 | 2.9 h parallel | $14.68 |

Cost is linear in decoding orders. The QC sample buys 16 rather than 8 so its
scores are directly comparable to the pilot's reference points, which were
measured at 16; the extra $0.51 buys that comparability. **The full cohort is
declined on evidence, not on price** — the n=5 label correlation is −0.100 at
p=0.87, so $14.68 would buy a feature with no demonstrated relationship to the
target.

### Stage 4c.5 structural feature extraction — forecast, approved, not spent

`stage4c5_features.md` §7.6. **Extraction is Volume-read-bound, not CPU-bound**:
1.57 s of wall per fold against 0.096 s of CPU, about 6% core utilisation. The
container was therefore dropped from `cpu=2.0` to `cpu=1.0` — reserving the
second core was billing an idle one — and throughput moved by only ~8% (326 s vs
352 s per 1,000 folds), confirming the diagnosis.

| Quantity | Per half (14,083 folds) | Both halves |
|---|---:|---:|
| Wall at 30 containers | ~12.3 min | ~12.3 min (parallel) |
| CPU + memory | $0.39 | **$0.78** |

Spend so far on this workstream — two smokes and two 1,000-fold passes — is
**under $0.15**.

### A note on published versus metered rates

Both forecasts above are built from the repo's **metered** rates
(`ectodomain_rates.json`), not from a pricing page, because this project was
once bitten by a cost figure that was 4.6× wrong. For the record, on CPU and
memory the published list rates agreed with the metered ones here to **0.3%**
(CPU 1.0030×, memory 1.0010×).

**The right lesson from the 4.6× error is "check it", not "never trust published
rates".** That error was never a wrong price list — it was a harness that
reloaded 6.2 GB of weights inside every timed fold, so the *seconds* were wrong
and the rate was fine. Distrusting the rate would have fixed nothing; measuring
what the harness actually did is what fixed it.

### Still to be ledgered

| Stage | Item | Owner | What will fill it |
|---|---|---|---|
| 3 | ESM-2 **regression head** fit and inference | `esm-arm` | Head fit seconds, ensemble member count, inference seconds per 1,000 rows (hole **E1b**) |
| 5 | Structural feature **heads** (the extraction forecast is above) | blocked on 4c production | Head fit time, extraction failures, realised extraction spend |
| 6 | Final test scoring and paired bootstraps | `eval-harness` | CPU-minutes for the single test pass and the bootstrap |
| 7a | Censored (Tobit) likelihood | stage 7a | 60 networks (2 arms × 30), CPU; protocol predeclared in `stage7_censored.md`, results pending |

---

## 3. Cost per 1,000 new predictions

This is the number the challenge question actually turns on. It is **inference**
cost: what it costs to score 1,000 peptide–HLA pairs the model has never seen,
once the model exists. Training and feature-cache construction are one-off and
listed separately above.

### Derivation

Sequence arms, from `stage2_baselines.md` §Cost (inference seconds per 1,000
rows, features cached per unique sequence) priced at Modal's published CPU rate
of **$0.04730 per core-hour** (`reports/ectodomain_rates.json`):

| Arm | Networks | s / 1,000 rows | Derivation | $ / 1,000 |
|---|---:|---:|---|---:|
| MLP, peptide only | 1 | 0.0002 | measured | $2.6 × 10⁻⁹ |
| MLP, peptide + pseudosequence | 1 | 0.0005 | measured | $6.6 × 10⁻⁹ |
| MLP, peptide + full domain | 1 | 0.0020 | measured | $2.6 × 10⁻⁸ |
| **Sequence ensemble, pep + pseudoseq** | 30 | 0.015 | 30 × 0.0005 | **$2.0 × 10⁻⁷** |
| Sequence ensemble, pep + domain | 30 | 0.060 | 30 × 0.0020 | $7.9 × 10⁻⁷ |

The ×30 is a derivation, not a measurement: the ensemble is 30 independent
forward passes over one cached feature matrix, so its inference cost is 30× a
single network's. Feature construction is excluded because it is cached per
unique sequence (28,166 rows carry only 5,633 peptides and 75 HLA sequences).

**ESM-2 arm, extraction half only** (`stage3_embedding_cost.csv`). For 1,000 new
pairs on alleles whose HLA embedding is already cached, the marginal work is
1,000 peptide embeddings. The measured seconds are on a laptop `mps` device for
which no published hourly rate exists, so they are priced at Modal's **A10G**
rate ($1.4812/h) as a deliberately generous **upper bound** — a rented A10G
costs far more than the laptop that produced the measurement:

| Checkpoint | s / 1,000 new peptides | $ / 1,000, upper bound |
|---|---:|---:|
| ESM-2 35M | 1.07 | $4.4 × 10⁻⁴ |
| ESM-2 150M | 2.74 | $1.1 × 10⁻³ |
| ESM-2 650M | 8.50 | $3.5 × 10⁻³ |

**The head's forward pass is not included** and is hole **E1b**. Even so, the
extraction half alone already places ESM-2 three to four orders of magnitude
above the sequence ensemble and three to four orders *below* Boltz-2.

Structural arm, from the stage 4c pilot's measured steady-state fold time and
the measured A10G worker rate:

| Quantity | Value | Derivation |
|---|---:|---|
| Boltz-2 arm-B steady fold | 16.76 s | measured, stage 4c |
| A10G + 4 CPU + 24 GiB | $1.4812 / h | measured, stage 4c |
| **Per fold** | **$0.0069** | 16.76 × 1.4812 / 3600 |
| **Per 1,000 predictions, fold only** | **$6.90** | measured unit × 1,000 |
| Per 1,000, including shard startup | $7.13 | $200.8 forecast / 28,166 |
| Per 1,000, at the 25% operational margin | $8.91 | $251 / 28,166 |
| GPU-hours per 1,000 predictions | 4.66 | 16,760 s / 3,600 |

ESMFold2, for comparison only — it failed its gate and was never run in
production:

| Quantity | Value | Derivation |
|---|---:|---|
| Arm-B steady fold | 42.63 s on L40S at $2.6512/h | measured, stage 4c |
| Per fold | $0.0314 | 42.63 × 2.6512 / 3600 |
| **Per 1,000 predictions** | **$31.40** | forecast |

### The table

| Arm | $ / 1,000 new predictions | GPU-h / 1,000 | Status | Accuracy (val median per-allele ρ) |
|---|---:|---:|---|---:|
| Training allele mean | not measured (a table lookup) | 0 | — | 0.000 |
| MLP, peptide only | $2.6 × 10⁻⁹ | 0 | derived | 0.202 |
| MLP, peptide + pseudoseq (single) | $6.6 × 10⁻⁹ | 0 | derived | 0.610 |
| **Sequence ensemble, 30 networks** | **$2.0 × 10⁻⁷** | 0 | derived | **0.693** |
| Sequence ensemble, full domain | $7.9 × 10⁻⁷ | 0 | derived | 0.653 |
| **ESM-2 (frozen representations)** | **$4.4 × 10⁻⁴ – $3.5 × 10⁻³** extraction only, upper bound; **+ ‹HOLE E1b›** for the head | 0 (laptop `mps`) | measured extraction, head pending | **‹HOLE E2›** |
| **Boltz-2 structural (arm B)** | **$6.90** measured / $8.91 with margin | 4.66 | measured unit cost | **‹HOLE B2›** |
| ProteinMPNN inverse folding | $0.52 | 0 (CPU) | forecast, QC sample only | not a half-life predictor (§4.5 of SUBMISSION) |
| *ESMFold2 structural (rejected)* | *$31.40* | *11.8* | *forecast only* | *not run* |

Accuracy figures are **validation** medians from `stage2_baselines.md`; the test
column is filled once at stage 6.

### What the ratio says

The structural arm costs about **3.5 × 10⁷ times** more per prediction than the
sequence ensemble it has to beat ($6.90 against $2.0 × 10⁻⁷). Put in wall-clock
rather than dollars: scoring the whole 28,166-pair dataset takes the sequence
ensemble **under half a second of one CPU core**, and Boltz-2 **131 GPU-hours**
(6.8 hours only because the work is spread over 20 parallel workers).

Two honest caveats on that ratio:

1. **It compares a laptop CPU figure priced at a cloud CPU rate against a cloud
   GPU bill.** Dollars are the only axis that puts them side by side at all, and
   the two sides are not the same kind of dollar. The wall-clock statement above
   is provider-free and carries the same message.
2. **Structures are reusable.** The $6.90 is paid once per *pair*, not once per
   experiment: every stage 5 feature group, head architecture and ensemble
   member reads the same stored mmCIF. For a fixed cohort that is studied
   repeatedly the amortised figure falls; for the actual use case — scoring new
   candidate peptides — it does not, because a new peptide needs a new fold.

Neither caveat changes the order of magnitude, and the order of magnitude is the
finding. **A structural arm does not need to be slightly better than the
sequence ensemble. At seven orders of magnitude it needs to be better by enough
that someone would pay $6.90 per thousand peptides to get it** — which is
exactly what the predeclared 0.05 minimum worthwhile gain exists to adjudicate.

---

## 4. Decisions this ledger drove

Cost measurement is not bookkeeping here; it changed what was run.

| Decision | Evidence | Consequence |
|---|---|---|
| A10 over H100 for two-chain folding | H100 needed to be 3.01× faster to break even; measured 1.35× | `stage4_benchmark.md` Finding 3 |
| Weights kept resident across a batch | A per-complex subprocess charged ~36 s of model load into every fold — **4.6× cost overstatement** | `stage4_benchmark.md` Finding 1 |
| Host request right-sized to 4 cores / 16 GiB | 32 GiB was 3× oversized; the host cost is identical on every GPU, so over-requesting biases the comparison toward the expensive card | `stage4_benchmark.md` Finding 5 |
| `--max_msa_seqs` **not** trimmed for cost | Measured parse cost ~$0.25 across 2,000 complexes — under 0.2% of the fold bill. The plan's concern was overturned by measurement | `stage4b1_msa_cache.md` |
| ESMFold2 dropped from production | 4.55× the cost per fold *and* a failed pose gate; its $884 full-cohort forecast never fitted the combined balance | `stage4c_ectodomain_pilot.md` |
| Production scope set to the full cohort, not the 2,000-pair panel | The measured 16.76 s fold put all 28,166 pairs inside the ceiling for Boltz-2 alone | `stage4c_ectodomain_pilot.md` |
| Feature-extraction container cut from 2 cores to 1 | Extraction is Volume-read-bound: 1.57 s wall against 0.096 s CPU per fold, ~6% core utilisation. The second core was billing idle; dropping it moved throughput by ~8% | `stage4c5_features.md` §7.6 |
| ProteinMPNN bought as a ~$1 QC sample, **not** as a $14.68 regression feature | Spearman against half-life is −0.100 at p=0.87, n=5. The QC sample is bought for a different reason: no PAE or pLDDT threshold catches pose failure, so the project has no cohort-wide way to estimate the pose-failure rate | `stage5_inverse_folding.md` |

The 4.6× correction is worth dwelling on. The original harness was copied from
Modal's own published Boltz example, which runs one input per function call — a
shape in which the weight load is invisible because there is nothing to amortise
it over. Copied into a batch loop it became 86% of every timed fold. **A
benchmark that measures the wrong thing is worse than no benchmark**, because it
is quoted with confidence. This project caught one such error; the rule it
produced — verify that a harness measures what it claims before trusting its
number — is why every figure above carries its source.

---

## 5. Open holes

| Tag | What is missing | Who fills it | What exactly is needed |
|---|---|---|---|
| ~~E1a~~ | ~~ESM-2 embedding extraction cost~~ | — | **Filled**: `reports/stage3_embedding_cost.csv`, three checkpoints, measured |
| **E1b** | ESM-2 **head** inference cost per 1,000 new pairs | `esm-arm` | Head inference seconds per 1,000 rows, ensemble member count, and which checkpoint/layer/representation was selected (per-position vs pooled changes the head width) |
| **E2** | ESM-2 validation accuracy | `esm-arm` | Median per-allele ρ on validation under `cv_folds()` with matched ensemble size, and the paired CI against `preds/seq_ensemble_pep_pseudo.csv` |
| **B1** | Stage 5 feature-extraction cost | blocked on 4c production | Modal CPU-hours with the Volume mounted, per-pair extraction time, coverage and failure counts |
| **B2** | Boltz-2 structural validation accuracy | blocked on 4c production + stage 5 | Median per-allele ρ on validation, and the paired CI against the sequence ensemble |
| **B3** | Actual production spend | the Modal production session | Metered `before`/`after` billing snapshots per workspace, realised GPU-hours, realised wall clock, failure count, and whether 10 concurrent A10Gs were actually granted in each workspace |
| **S6** | Stage 6 scoring cost | `eval-harness` | CPU-minutes for the single test pass plus the paired cluster bootstraps |
| **S7a** | Censored (Tobit) likelihood cost | stage 7a | CPU-minutes for 60 networks (2 arms × 30) plus the paired bootstrap; protocol is predeclared in `stage7_censored.md`, §6 is still "pending" |

`reports/ectodomain-20261004/production_<profile>.jsonl` now exist for both
profiles and are **empty** — the runner has been wired up but no shard has
committed a record yet. That is the first place B3 will appear.

Until **B3** lands, every production figure in this file is a forecast from the
pilot's measured unit cost and is labelled as such. The forecast's own stated
risk is concurrency, not price: if a workspace is granted fewer than 10 A10Gs
the wall clock scales linearly, while the total cost does not move.

---

## 6. Reproduce

```bash
# figures and the accuracy column
.venv/bin/python reports/figures/make_figures.py

# the unit costs the structural forecast is built from
cat reports/ectodomain_rates.json            # Modal published per-hour rates
cat reports/ectodomain_billing_before.json   # metered workspace total, 02:22 BST
cat reports/ectodomain_billing_after.json    # metered workspace total, 02:39 BST
python scripts/gpu_decision.py --panel 2000 --workers 10 --gib 16   # two-chain table

# the production forecast, without spawning any GPU
MODAL_PROFILE=<profile> modal run modal_app/ectodomain_production.py::production \
    --profile <profile> --dry-run
```
