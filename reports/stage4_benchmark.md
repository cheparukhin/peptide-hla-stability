# Stage 4: Boltz-2 GPU benchmark and hardware choice

Measured 2026-10-03 on Modal. Method and design rationale in
[docs/BOLTZ_PIPELINE.md](../docs/BOLTZ_PIPELINE.md); this file is the result.

**Decision: fold on A10.** $0.018 per successful complex, $37 for a
2,000-complex panel (11% of the $330 folding ceiling), 2.6 h at 10 concurrent
workers.

## Measured throughput

8 complexes per GPU, identical settings (`--diffusion_samples 1`,
`--recycling_steps 3`, `--sampling_steps 200`, `--write_full_pae`, mmCIF), one
container per GPU, weights preloaded from a Volume, MSAs precomputed. Every
complex is 182 HLA residues + a 9-residue peptide.

| GPU | warm-up | steady median | min–max | peak GPU mem | ok |
| --- | ---: | ---: | ---: | ---: | ---: |
| **A10** 24 GB | 56.1 s | **46.6 s** | 46.2–47.8 | 8.37 GB | 8/8 |
| H100 80 GB | 52.8 s | **41.0 s** | 40.6–43.0 | 8.87 GB | 8/8 |
| L40S 48 GB | 51.9 s | **42.2 s** | 42.1–42.6 | 8.59 GB | 8/8 |
| L4 24 GB | 67.8 s | **60.0 s** | 58.5–64.9 | 8.32 GB | 8/8 |
| A100 40 GB | 77.9 s | **64.6 s** | 60.5–66.1 | 8.59 GB | 8/8 |

**44 folds across the whole stage, zero failures** (40 benchmark + 3 pilot + 1
host probe).

## Cost per successful complex

Published Modal per-second rates, read 2026-10-03. Warm-up is excluded from the
steady-state median and its overhead amortised over 100 complexes per
container. Host request right-sized to 4 cores / 16 GiB (see below).

| GPU | combined $/hr | $/complex | 2,000 panel | capacity at $330 | 2,000 at 10 workers |
| --- | ---: | ---: | ---: | ---: | ---: |
| **A10** | 1.418 | **$0.018** | **$37.0** | 17,837 | 2.6 h |
| L4 | 1.116 | $0.019 | $37.6 | 17,553 | 3.4 h |
| L40S | 2.268 | $0.027 | $53.4 | 12,359 | 2.4 h |
| A100-40GB | 2.415 | $0.043 | $85.8 | 7,692 | 3.6 h |
| H100 | 4.266 | $0.049 | $98.6 | 6,693 | 2.3 h |

A10 and L4 are within 5% of each other on cost; A10 wins because it is **1.29×
faster**, which buys 0.8 h of wall-clock for essentially the same money. A10 is
the choice on cost *and* on deadline among the cheap cards.

## Three findings that matter more than the ranking

### 1. H100 is a trap here: 1.03× faster, 1.88× the price

H100 needed to be **>1.83× faster** than L40S to break even on cost per
complex. It came in at **1.03×** — 41.0 s against 42.2 s. Nearly double the
hourly rate for a 3% speedup.

The reason is the workload, not the hardware: a 191-residue complex at
`diffusion_samples 1` is too small to saturate an H100. Runtime is dominated by
sequential diffusion and recycling steps rather than by throughput the extra
silicon could absorb. **This is the plan's "cheaper per hour is not cheaper per
structure" warning firing in the opposite direction** — the expensive card was
the bad buy, and only a measurement could show it.

### 2. A100-40GB came last, which is anomalous — treat it as unverified

A100-40GB was the **slowest card measured** (64.6 s), slower than A10 (46.6 s),
L4 (60.0 s), L40S and H100. That ordering is physically surprising: A100 should
comfortably beat A10.

The within-container spread was tight (60.5–66.1 s, ±4%), so it was
*consistently* slow inside that container rather than noisy — which points at a
property of the host it landed on, not measurement noise between folds. But
**n = 1 container**, so a degraded or contended host is not ruled out.

This does not change the decision (A10 wins either way, and A100 was never
competitive on price), so it was not worth ~$0.37 to re-test. **Do not cite the
A100 figure as a property of A100 hardware** without a repeat on a fresh
container.

### 3. The host request was 3× oversized — but 8 GiB would have crashed

The worker shape was pinned at 4 cores / 32 GiB only because the plan's rate
table assumes it. Measured on A10: **peak child RSS 9.89 GB**.

So 32 GiB was roughly 3× more than Boltz touches — but the 8 GiB I would have
guessed at would have OOMed. Hence 4 cores / **16 GiB**, which is what the cost
table above uses.

This matters disproportionately for the cheap cards, because the host request
costs the same on every GPU:

| | host share of bill |
| --- | ---: |
| A10 at 4c/32 GiB | 29% |
| H100 at 4c/32 GiB | 10% |

**Over-requesting the host biases the whole comparison toward the expensive
card.** At the plan's 4c/32 GiB the A10 panel costs $40.2; right-sized it costs
$37.0. Both still win, but the bias was real and worth removing.

Caveat: the container-level cgroup counters were unreadable under Modal's gVisor
sandbox (`memory.peak` and `memory.max_usage_in_bytes` both absent), so 9.89 GB
is the peak RSS of the Boltz child process, not the whole container. Real
container demand is somewhat higher — another reason for 16 GiB rather than 12.

## Pose validation

The pilot doubles as the register check
([STRUCTURES.md](../docs/STRUCTURES.md)). Superposing the predicted HLA domain
onto the crystal and then measuring the peptide under that transform:

| complex | PDB | HLA CA fit | peptide CA RMSD | max dev |
| --- | --- | ---: | ---: | ---: |
| A\*11:01 / KTFPPTEPK | 7WKJ | 0.84 Å | **0.39 Å** | 0.68 Å |
| A\*02:01 / LLWNGPMAV | 9SL0 | 0.38 Å | **0.13 Å** | 0.28 Å |
| B\*15:01 / ILGPPGSVY | 1XR9 | 0.28 Å | **0.42 Å** | 0.85 Å |

Essentially crystallographic agreement, with the P2/PΩ anchor positions the
tightest. ipTM 0.982–0.993 across all folds. Geometry features taken from these
structures are measuring the complex, not a docking artefact.

## Spend

| item | cost |
| --- | ---: |
| Setup: 6.204 GB weights + 8 HLA MSAs (CPU only) | ~$0.01 |
| Pilot: 3 complexes on L40S | ~$0.17 |
| Benchmark: 8 complexes × 5 GPUs | ~$1.35 |
| Host-memory probe: 1 complex on A10 | ~$0.03 |
| Wasted on a crash-looping container (my bug) | ~$0.02 |
| **Total** | **~$1.58** |

Against the plan's **$15** ceiling for the end-to-end and hardware pilot: 11%
used. The $330 folding allocation is untouched.

## Limitations

- **n = 1 container per GPU.** Within-container fold spread is tight (±3–9%),
  but between-container variance is unmeasured. The A100 result is the one this
  most plausibly affects.
- **7 steady-state complexes per GPU** against the plan's 20–30. Runtime
  ranking is safe at a fixed 191 residues, but the **failure-rate** estimate is
  weak. 0/44 failures bounds it loosely, not tightly. The panel size was cut
  deliberately to preserve credits for the production run.
- **Single configuration.** One pose, standard sampling. Reduced sampling,
  multiple poses, or full-length HLA with β2-microglobulin would each need a new
  benchmark; the plan says as much for β2m.
- **Published rates, not invoiced amounts.** Costs are computed from
  modal.com/pricing per-second figures and measured seconds, not read off a
  bill. Startup is amortised at an assumed 100 complexes per container.
- The pose check compares a 2-chain prediction against 3-chain crystals; every
  pilot entry contains β2-microglobulin, which our input omits by design.

## Reproduce

```bash
modal run modal_app/boltz_setup.py::setup                       # CPU: weights + MSAs
modal run modal_app/boltz_bench.py::pilot --gpu A10 --limit 3   # gate
python scripts/boltz_pose_check.py --structures <dir>           # register check
modal run modal_app/boltz_bench.py::benchmark \
    --gpus 'L40S,A100-40GB,H100!,L4,A10' --limit 8
python scripts/gpu_decision.py --panel 2000 --workers 10 --gib 16
```
