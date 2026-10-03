# Stage 4: Boltz-2 GPU benchmark and hardware choice

Measured 2026-10-03 on Modal. Design rationale in
[docs/BOLTZ_PIPELINE.md](../docs/BOLTZ_PIPELINE.md).

**Decision: fold on A10** at $0.018 per complex. A 2,000-complex panel costs $37
(11% of the $330 ceiling) and finishes in 2.6 h at 10 workers.

## Throughput

8 complexes per GPU, identical Boltz-2 settings (`--diffusion_samples 1`,
`--recycling_steps 3`, `--sampling_steps 200`, `--write_full_pae`, mmCIF output).
One container per GPU, weights preloaded from a Modal Volume, MSAs precomputed on
CPU. Every complex is 191 residues (182 HLA + 9 peptide).

| GPU | warm-up | steady median | min–max | peak GPU mem | ok |
| --- | ---: | ---: | ---: | ---: | ---: |
| **A10** 24 GB | 56.1 s | **46.6 s** | 46.2–47.8 | 8.37 GB | 8/8 |
| H100 80 GB | 52.8 s | **41.0 s** | 40.6–43.0 | 8.87 GB | 8/8 |
| L40S 48 GB | 51.9 s | **42.2 s** | 42.1–42.6 | 8.59 GB | 8/8 |
| L4 24 GB | 67.8 s | **60.0 s** | 58.5–64.9 | 8.32 GB | 8/8 |
| A100 40 GB | 77.9 s | **64.6 s** | 60.5–66.1 | 8.59 GB | 8/8 |

44 folds total (40 benchmark + 3 pilot + 1 host probe), zero failures.

## Cost per complex

Computed from Modal per-second rates (read 2026-10-03). The warm-up fold is
excluded from the steady-state median; its overhead is amortised over 100
complexes per container. Host request right-sized to 4 cores / 16 GiB after
measuring peak RSS at 9.89 GB.

| GPU | combined $/hr | $/complex | 2,000 panel | capacity at $330 | 2,000 at 10 workers |
| --- | ---: | ---: | ---: | ---: | ---: |
| **A10** | 1.418 | **$0.018** | **$37.0** | 17,837 | 2.6 h |
| L4 | 1.116 | $0.019 | $37.6 | 17,553 | 3.4 h |
| L40S | 2.268 | $0.027 | $53.4 | 12,359 | 2.4 h |
| A100-40GB | 2.415 | $0.043 | $85.8 | 7,692 | 3.6 h |
| H100 | 4.266 | $0.049 | $98.6 | 6,693 | 2.3 h |

A10 and L4 cost nearly the same per complex. A10 wins because it is 1.29x
faster, saving 0.8 h of wall-clock for the same money.

## Key findings

### H100 does not pay for itself at this problem size

H100 needed to be at least 1.83x faster than L40S to break even on cost. It
measured 1.03x — 41.0 s vs 42.2 s. Nearly double the hourly rate for a 3%
speedup.

A 191-residue complex with one diffusion sample is too small to keep an H100
busy. The runtime is dominated by sequential diffusion and recycling steps, not
by parallel computation the extra hardware could absorb. The plan warned that
"cheaper per hour is not cheaper per structure"; here the expensive card was the
worse buy.

### A100-40GB was anomalously slow — treat as unverified

A100-40GB was the slowest card tested (64.6 s median), behind A10 (46.6 s) and
L4 (60.0 s). That is physically implausible: A100 should beat both.

The spread within the container was tight (60.5–66.1 s, ±4%), so it was
consistently slow, not noisy — pointing at a degraded or contended host rather
than measurement error. But with only one container, a bad draw cannot be ruled
out. Re-testing was not worth ~$0.37 since A10 wins regardless. Do not cite this
figure as a property of A100 hardware without repeating on a fresh container.

### The host request needed right-sizing

The plan assumed 4 cores / 32 GiB per worker. Measured peak Boltz child RSS on
A10: 9.89 GB. So 32 GiB was 3x more than needed — but 8 GiB would have
crashed. The cost table uses 4 cores / 16 GiB.

This matters because the host cost is fixed regardless of GPU:

| | host share of bill |
| --- | ---: |
| A10 at 4c/32 GiB | 29% |
| H100 at 4c/32 GiB | 10% |

Over-requesting the host inflates cheap cards' bills more than expensive ones,
biasing the comparison toward the fast GPU. At 4c/32 GiB the A10 panel would
cost $40.2; at 4c/16 GiB it costs $37.0.

Caveat: container-level cgroup counters were unreadable under Modal's gVisor
sandbox, so 9.89 GB is the Boltz child process peak, not the whole container.
Real demand is somewhat higher, which is why 16 GiB rather than 12.

## Pose validation

The pilot also serves as the register check
([STRUCTURES.md](../docs/STRUCTURES.md)). Method: superpose the predicted HLA
domain onto the crystal structure (CA atoms only), then measure how far the
predicted peptide lands from its crystallographic position under the same
transform.

| complex | PDB | HLA CA fit | peptide CA RMSD | max per-residue |
| --- | --- | ---: | ---: | ---: |
| A\*11:01 / KTFPPTEPK | 7WKJ | 0.84 Å | **0.39 Å** | 0.68 Å |
| A\*02:01 / LLWNGPMAV | 9SL0 | 0.38 Å | **0.13 Å** | 0.28 Å |
| B\*15:01 / ILGPPGSVY | 1XR9 | 0.28 Å | **0.42 Å** | 0.85 Å |

Peptide deviations of 0.13–0.42 Å are within crystallographic error. The anchor
positions (P2 and the C-terminal PΩ, where the peptide is pinned into the HLA
groove) are the tightest. ipTM 0.982–0.993 across all folds. Geometry features
extracted from these predicted structures reflect the real complex, not a
mis-docked artefact.

## Spend

| item | cost |
| --- | ---: |
| Setup: 6.204 GB weights + 8 HLA MSAs (CPU only) | ~$0.01 |
| Pilot: 3 complexes on L40S | ~$0.17 |
| Benchmark: 8 complexes × 5 GPUs | ~$1.35 |
| Host-memory probe: 1 complex on A10 | ~$0.03 |
| Wasted on a crash-looping container (bug) | ~$0.02 |
| **Total** | **~$1.58** |

11% of the plan's $15 pilot/benchmark ceiling. The $330 folding allocation is
untouched.

## Limitations

- **One container per GPU.** Fold-to-fold spread within a container is tight
  (±3–9%), but variation between containers on the same GPU type is unmeasured.
  The A100 result is the one most likely affected.
- **7 steady-state complexes per GPU** (the plan called for 20–30). The runtime
  ranking is reliable at a fixed 191 residues, but the failure-rate estimate is
  weak: 0 failures in 44 folds does not tightly bound the true rate. Panel size
  was cut deliberately to save credits for the production run.
- **Single configuration.** One pose, standard sampling. Reduced sampling,
  multiple poses, or full-length HLA with beta-2-microglobulin would each need a
  fresh benchmark.
- **Computed costs, not invoiced.** Costs use modal.com/pricing per-second rates
  and measured wall time, not actual bills. Startup overhead is amortised at 100
  complexes per container.
- **2-chain prediction vs 3-chain crystal.** Every pilot crystal includes
  beta-2-microglobulin (a second protein chain beneath the HLA platform); our
  input omits it by design. The pose check superposes on the HLA domain and
  measures only the peptide, so the comparison is valid for the groove region.

## Reproduce

```bash
modal run modal_app/boltz_setup.py::setup                       # CPU: weights + MSAs
modal run modal_app/boltz_bench.py::pilot --gpu A10 --limit 3   # gate
python scripts/boltz_pose_check.py --structures <dir>           # register check
modal run modal_app/boltz_bench.py::benchmark \
    --gpus 'L40S,A100-40GB,H100!,L4,A10' --limit 8
python scripts/gpu_decision.py --panel 2000 --workers 10 --gib 16
```
