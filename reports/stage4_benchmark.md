# Stage 4: folding engine and GPU choice

Measured 2026-10-03 on Modal. Design rationale in
[docs/BOLTZ_PIPELINE.md](../docs/BOLTZ_PIPELINE.md).

**Historical decision for the 191-residue, two-chain construct: Boltz-2 on A10**
at $0.004 per complex. A 2,000-complex panel costs
$8.00 (2% of the $330 ceiling) and finishes in 0.56 h at 10 workers.

The new 383-residue ectodomain + beta2m + peptide workload follows
[stage 4c](../HACKATHON_PLAN.md#4c-ectodomain--beta-2-microglobulin-folding)
and the [stage 4c report](stage4c_ectodomain_pilot.md), which carries its
measured pilot, memory, throughput, and production forecast.
The values in this report remain the original two-chain results. The earlier
recommendation to use Boltz-2 alone is superseded: stage 4c runs both models
with the same constructs, prepared MSAs, pilot arms, and production cohort.

> **Correction.** An earlier version of this report published $0.018/complex and
> $37.00 per 2,000, measured with a harness that reloaded the model inside every
> timed fold. Those figures were ~4.6x too high. Every number below comes from
> the corrected harness. See [Finding 1](#1-the-first-harness-measured-model-reloading-not-folding).

## Throughput

8 complexes per GPU, one container per GPU, weights preloaded from a Volume,
MSAs precomputed on CPU. Every complex is 191 residues (182 HLA + 9 peptide).
Settings: `--diffusion_samples 1`, `--recycling_steps 3`, `--sampling_steps
200`, `--write_full_pae`, mmCIF.

| GPU | load + 1st fold | steady median | min–max | peak GPU mem | ok |
| --- | ---: | ---: | ---: | ---: | ---: |
| **A10** 24 GB | 57.9 s | **9.6 s** | 9.3–9.6 | 8.4 GB | 8/8 |
| H100 80 GB | 59.5 s | **7.1 s** | 6.8–7.3 | 8.9 GB | 8/8 |
| A100 40 GB | 74.7 s | **10.1 s** | 9.9–10.1 | 8.6 GB | 8/8 |
| L40S 48 GB | 66.3 s | **10.9 s** | 10.9–11.1 | 8.6 GB | 8/8 |
| L4 24 GB | 85.7 s | **15.9 s** | 15.7–16.0 | 8.3 GB | 8/8 |

"load + 1st fold" is the weight load, CUDA init and first fold, paid once per
container and amortised over the production batch size rather than charged to
every complex.

## Cost per complex

Modal per-second rates read 2026-10-03, host request 4 cores / 16 GiB.

| GPU | combined $/hr | $/complex | 2,000 panel | capacity at $330 | 2,000 at 10 workers |
| --- | ---: | ---: | ---: | ---: | ---: |
| **A10** | 1.418 | **$0.0040** | **$8.00** | 82,500 | 0.56 h |
| L4 | 1.116 | $0.0051 | $10.20 | 64,705 | 0.92 h |
| A100-40GB | 2.415 | $0.0072 | $14.40 | 45,833 | 0.59 h |
| L40S | 2.268 | $0.0072 | $14.40 | 45,833 | 0.64 h |
| H100 | 4.266 | $0.0089 | $17.80 | 37,078 | 0.42 h |

At $0.004/complex the **entire 28,166-pair dataset** would cost ~$113, inside
the ceiling. Folding is not the binding constraint on this project.

## Findings

### 1. The first harness measured model reloading, not folding

The original `fold_batch` spawned a fresh `boltz predict` subprocess per
complex, so every timed fold contained a Python start, `import torch` +
Lightning, a 6.2 GB weight load, CUDA context init and teardown. The tell was
`startup_s = 0.0` on all 40 rows: by the harness's own definition nothing
happened before the first fold, because all the overhead lived *inside* it.

Measured on L40S, same panel and settings, weights resident vs reloaded:

| | per-fold time |
| --- | ---: |
| fresh process per complex (original) | 42.2 s |
| one process per batch (corrected) | **6.3 s** |

**~36 s — 86% — of each original measurement was process and weight-load
overhead.** This also violated `HACKATHON_PLAN.md:206` ("keep models loaded
across complexes to avoid reload overhead") while `docs/BOLTZ_PIPELINE.md`
asserted the requirement was met. The container was reused; the process was not.

The root cause is worth recording: the shape was lifted from
[Modal's published Boltz example](https://modal.com/docs/examples/boltz_predict),
which runs `boltz predict` as a subprocess for **one** input per function call.
For a single-input demo the weight load is invisible because there is nothing to
amortise it over. Copied into a batch loop it becomes the dominant term. The
idiomatic Modal pattern is `modal.Cls` with `@modal.enter()` to load once per
container; the example does not demonstrate it.

### 2. Between-container variance is the dominant source of error

The same model, GPU type and settings measured **6.3 s in one container and
10.9 s in another**:

| | spread |
| --- | ---: |
| within one container (8 folds, one host) | **±1%** |
| between containers (same GPU type, different hosts) | **73%** |

Modal schedules each container onto whatever host is free, and hosts differ in
CPU contention, PCIe topology and neighbours. **With n=1 container per GPU, the
10.9 s vs 10.1 s gap between L40S and A100-40GB is not resolvable** — a re-draw
of L40S could land anywhere in that range.

What survives this: A10 is cheapest by a margin larger than the variance, and
the H100 verdict is an absolute gap, not a ratio. What does not: fine-grained
ordering among the middle cards. **Any re-run should use ~3 containers per GPU**
and report a median of container medians. Within-container spread is negligible,
so trade folds-per-container for container count; the weight load is the cost
paid 3x, roughly doubling sweep spend to ~$1.00.

### 3. H100 still does not pay for itself

H100 needs to be **3.01x faster** than A10 to break even on cost per complex
($4.2657 / $1.4181). It measured **1.35x** — 7.1 s against 9.6 s.

This verdict is robust to both corrections above. It held at the old overhead
level (1.03x against a 1.83x bar vs L40S) and holds now, because the gap is
absolute (2.5 s) rather than a ratio that subtracting a constant could rescue. A
191-residue complex at one diffusion sample cannot saturate an H100; runtime is
sequential diffusion and recycling, not throughput the extra silicon absorbs.

### 4. A100-40GB is consistently slower than it should be

A100-40GB came in slower than A10 in **three independent container
allocations**: the original Boltz sweep (64.6 s vs 46.6 s), the corrected Boltz
sweep (10.1 s vs 9.6 s), and the ESMFold2 sweep (9.2 s vs L40S 6.7 s). Three
separate draws agreeing in direction is no longer attributable to a single bad
host, though all three remain n=1 container.

It changes no decision — A100-40GB was never price-competitive — but **do not
cite these figures as a property of A100 hardware** without a multi-container
repeat.

### 5. The host request needed right-sizing

The plan assumed 4 cores / 32 GiB. Measured peak Boltz child RSS: **9.89 GB**.
So 32 GiB was 3x oversized, but 8 GiB would have crashed; the tables above use
4 cores / 16 GiB.

The host cost is identical on every GPU, so over-requesting inflates cheap
cards' bills disproportionately — 29% of an A10 bill against 10% of an H100's —
biasing the comparison toward the fast card.

Caveat: cgroup counters were unreadable under Modal's gVisor sandbox, so 9.89 GB
is the Boltz child process peak, not whole-container demand. Hence 16 GiB rather
than 12.

## ESMFold2 historical benchmark and superseded recommendation

[ESMFold2](https://huggingface.co/biohub/ESMFold2) (Biohub, MIT, `esm==3.4.1.post1`)
was benchmarked as an alternative engine: a diffusion structure predictor on
ESMC embeddings, with native multi-chain input and published DockQ wins over
AlphaFold 3 on protein-protein and antibody-antigen interfaces. It runs
in-process with the model resident, so its numbers were never affected by the
harness bug.

| | Boltz-2 (A10) | ESMFold2 (best card) |
| --- | ---: | ---: |
| 2,000-complex panel | **$8.00** | $8.80 (L40S) |
| steady fold | 9.6 s | 6.7 s (L40S) |
| peak GPU memory | **8.4 GB** | 26.0 GB |
| runs on 24 GB cards | **yes** | **no** |
| peptide CA RMSD, median (n=5) | **0.29 Å** | 0.68 Å |
| poses > 0.5 Å (n=5) | **1/5** | 3/5 |

**Original recommendation: stay on Boltz-2; superseded by stage 4c.**
The historical reasons were:

- **The speed difference is unresolvable.** ESMFold2's 6.7 s sits inside Boltz's
  own 6.3–10.9 s between-container range on the same card. The apparent 5.8x
  win was entirely the harness bug.
- **26 GB costs it the cheap cards.** The 7B ESMC backbone plus diffusion trunk
  does not fit 24 GB even with bf16 weights, so A10 and L4 — the two cheapest
  options — are unavailable. Boltz-2 at 8.4 GB fits everything.
- **Pose quality is worse at matched n.** Median 0.68 Å against 0.29 Å, with 3
  of 5 above 0.5 Å against 1 of 5.

A fourth strike, magnitude unverified: **ESMFold2's measured host RSS is
45.46 GB against Boltz-2's 10.00 GB**, both well above the 16 GiB requested.
Modal bills the greater of requested and used, so ESMFold2's figures above
understate its true cost — at a 48 GiB host request its 2,000-panel goes from
$8.69 to $9.67. The caveat is that safetensors are memory-mapped, so the
26.7 GB of weight file pages count toward RSS without being anonymous
allocations; the real requirement is lower than 45 GB but certainly above
Boltz-2's. The original benchmark did not resolve this with a request-size
sweep. The new matched-model plan requires remeasuring full-construct host
memory and billed cost before production.

Two findings worth keeping anyway:

- **The small two-chain MSA test did not improve ESMFold2 poses.** With the same alignments Boltz-2 used,
  B1501 went 1.35 → 1.72 Å and A1101 was unchanged, while fold time rose 32%
  (7.3 → 9.6 s). These cases favored single-sequence. They do not establish
  an MSA policy for the new three-chain workload; the primary comparison
  supplies the same prepared MSAs to both models.
- **ESMFold2 exposes `pair_chains_iptm`**, the peptide-HLA interface ipTM that
  stage 5 lists as a confidence feature. Boltz-2 only gives a global ipTM. If
  that feature proves important, ESMFold2 is the cheaper source for it.

## Pose validation

Superpose the predicted HLA domain onto the crystal (CA only), then measure the
peptide under that transform. Fitting on the HLA and measuring the peptide is
the point: fitting on the peptide would rotate onto the answer and hide a
mis-docked pose.

| complex | PDB | HLA fit | Boltz-2 peptide | ESMFold2 peptide |
| --- | --- | ---: | ---: | ---: |
| A\*11:01 / KTFPPTEPK | 1X7Q / 7WKJ | 0.33 / 0.88 Å | **0.29 Å** | 0.48 Å |
| A\*02:01 / LLWNGPMAV | 9SL0 | 0.37 / 0.40 Å | **0.14 Å** | 0.20 Å |
| B\*15:01 / ILGPPGSVY | 1XR9 | 0.26 / 0.33 Å | **0.38 Å** | 1.35 Å |
| B\*07:02 / IPRRNVATL | 7LFZ | 0.36 / 0.41 Å | **1.30 Å** | 1.50 Å |
| B\*08:01 / ELRRKMMYM | 4QRU | 0.32 / 0.92 Å | **0.25 Å** | 0.68 Å |

5/5 in groove for both models at the 2.0 Å bar. Boltz-2 ipTM 0.981–0.993.

**Both models share one failure mode, and it matters for stage 5.** The error is
concentrated at central peptide positions while anchors stay tight. Boltz-2 on
B\*07:02 / IPRRNVATL deviates 1.90 Å at P5 and 3.33 Å at P6 while P1–P2 and
P8–P9 are all under 0.25 Å; ESMFold2 shows the same pattern on two complexes.

So the register is reliable in every case — the peptide is pinned correctly at
both ends — but **the central bulge (P5–P7) carries several-angstrom
uncertainty on some complexes in both engines.** Geometry features computed at
central positions (burial depth, contact counts) are therefore noisier than the
same features at anchor positions, and stage 5 should expect that asymmetry
rather than treating all nine positions as equally well determined.

## Spend

| item | cost |
| --- | ---: |
| Boltz-2: original (superseded) sweep, pilot, probes | ~$1.58 |
| ESMFold2: 26.74 GB weights, 2 API probes, 3 pilots, sweep | ~$0.38 |
| Boltz-2: corrected pilot + 5-GPU sweep re-run | ~$0.68 |
| **Total** | **~$2.64** |

18% of the plan's $15 pilot/benchmark ceiling. The $330 folding allocation is
untouched.

## Limitations

- **n=1 container per GPU**, against 73% measured between-container variance.
  This is the largest uncertainty in the report and it limits the middle-card
  ordering (see Finding 2). The A10 choice and the H100 verdict survive it.
- **7 steady-state complexes per GPU** against the plan's 20–30. Runtime
  ranking is reliable at fixed 191 residues; the failure-rate estimate is not.
  0 failures in 68 folds bounds it loosely.
- **Single configuration.** One pose, standard sampling. Reduced sampling,
  multiple poses, or full-length HLA with beta-2-microglobulin would each need
  a fresh benchmark.
- **Computed costs, not invoiced.** Per-second published rates times measured
  wall time. Startup amortised at an assumed 100 complexes per container.
- **Pose validation is n=5, one seed, two-chain prediction against three-chain
  crystals.** Every pilot crystal contains beta-2-microglobulin, which our
  input omits by design; the check superposes on the HLA domain and measures
  only the peptide, so it is valid for the groove. ESMFold2 additionally uses
  `lm_dropout=0.3` by default, so its poses are one sample from a stochastic
  ensemble at `seed=0`.
- **ESMFold2 ran without flash-attn, xformers or Transformer Engine**, which it
  warns costs speed. Its timings are therefore a floor on achievable
  performance, not a ceiling. It would still need to overcome the 26 GB VRAM
  problem to change the decision.

## Reproduce

```bash
# Boltz-2
modal run modal_app/boltz_setup.py::setup                       # CPU: weights + MSAs
modal run modal_app/boltz_bench.py::pilot --gpu L40S --limit 5  # gate
python scripts/boltz_pose_check.py --structures <dir> --out reports/boltz_pose_check.csv
modal run modal_app/boltz_bench.py::benchmark \
    --gpus 'L40S,A100-40GB,H100!,L4,A10' --limit 8
python scripts/gpu_decision.py --panel 2000 --workers 10 --gib 16

# ESMFold2
modal run modal_app/esmfold_setup.py                            # API probe, no GPU
modal run modal_app/esmfold_setup.py --download                 # 26.74 GB to Volume
modal run modal_app/esmfold_bench.py::pilot --limit 5           # add --use-msa for the MSA arm
modal run modal_app/esmfold_bench.py::benchmark --limit 8
python scripts/gpu_decision.py --results reports/esmfold_bench_results.csv --gib 16
```
