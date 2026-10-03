# Stage 4: Boltz-2 folding pipeline and GPU benchmark

Design and protocol for [HACKATHON_PLAN.md](../HACKATHON_PLAN.md) stage 4. This
document records the decisions and the measurement protocol. **Measured numbers
land in `reports/stage4_benchmark.md`; nothing here is a measurement.**

## MSAs: fewer than you would think, and not on the GPU

The plan says "cache model weights and HLA MSAs, use single-sequence peptide
input, don't spend GPU time waiting on MSA generation." Three facts make that
cheap:

**An MSA is per unique chain sequence, not per complex.** Every row in the
dataset is a 182-residue HLA domain plus a 9-residue peptide. The dataset holds
**75 unique HLA domain sequences** across its 28,166 pairs. So:

| Scope | Complexes | HLA MSAs needed |
| --- | ---: | ---: |
| Pilot | 5 | 5 |
| Pilot + GPU benchmark | 29 | **8** |
| Production panel (6 alleles) | ~2,000 | **6** |
| Entire dataset | 28,166 | **75** |

The production panel needs six alignments. This is a CPU job measured in
minutes, not a workload to budget GPU hours against.

**The peptide gets no MSA.** A 9-residue query cannot produce a meaningful
alignment — there is no reliable homology signal at that length, and the peptide
identity is precisely the variable under study. Peptide chains run
single-sequence.

**Therefore MSA generation must not run on the GPU worker.** This is not a
preference — reading `boltz/main.py` confirms `predict` calls `compute_msa` →
`run_mmseqs2` **synchronously, in-process, before the Lightning trainer starts**.
With `--use_msa_server`, the accelerator sits idle blocking on HTTP to
`api.colabfold.com` and Modal bills it at the GPU rate the whole time. Folding
2,000 complexes that way would issue 2,000 round-trips for six distinct answers
and hammer a free shared community service for results we already have.

Note that **Modal's own published Boltz example does exactly this** — it passes
`--use_msa_server` on an H100. That example is a single-input demo, and copying
its shape into a batch sweep is the trap. Our MSAs are built in a separate
CPU-only Modal function, written to a Volume, and referenced by path.

### Measured: the MSA step costs about a penny

Run 2026-10-03 on Modal, CPU only (`boltz_setup.py::setup`):

| | |
| --- | ---: |
| Weight snapshot cached to Volume | **6.204 GB in 36.5 s** |
| HLA MSAs built (8 unique sequences, one batched request) | **46.1 s** |
| Alignment depth per allele | **9,771 – 10,305** sequences |
| Query sequence first in file (verified, all 8) | yes |
| Approximate cost | **~$0.01** |

Per-allele depth: A\*02:01 9,771 · A\*03:01 9,872 · A\*11:01 10,077 ·
B\*07:02 10,216 · B\*08:01 9,933 · B\*15:01 10,305 · B\*35:01 10,252 ·
B\*39:01 10,187. Class I HLA is a deeply sampled family, so these are rich
alignments — which is the substantive reason to keep the MSA arm rather than
fold single-sequence.

Two notes from the run:

- **The 46 s was a single batched request for all 8 sequences.** The progress
  bar's early estimate said 22 minutes; it finished in 45 s once the server
  dequeued it. Server queue time is the variable here, not compute.
- **`build_msas` requests 0.25 cores / 2 GiB, not the fold worker's 4 cores /
  32 GiB.** It spends its life in a polling loop waiting on the MSA server, so
  it is I/O-bound; Modal bills the greater of requested and used, and asking for
  the fold shape would have cost roughly 10× for the same wait. This matters
  more at 75 alleles than at 8.

### What the pilot checks about MSAs

Boltz's docs say single-sequence mode "reduces accuracy" and is "not
recommended", and our MSAs cost essentially nothing, so **MSA-on is the default
and single-sequence is not a serious contender.** The pilot still runs both arms
on the same 5 complexes, for two reasons that are worth 5 extra folds:

- It quantifies what the MSA buys *on this task*, against crystal structures.
  That number belongs in the writeup either way.
- **It verifies a combination the Boltz docs do not document.** A precomputed
  `.a3m` is only valid when a *single* protein chain carries an MSA; two aligned
  chains would need the paired-CSV format. Our YAML satisfies that by setting
  `msa: empty` on the peptide — but mixing `msa: empty` on one chain with a real
  `.a3m` on another is not shown in any documented example. If it errors, the
  fallback is paired CSV, and the pilot is where that surfaces.

## Panels

Built by [`scripts/boltz_panel.py`](../scripts/boltz_panel.py), deterministic, no
RNG. All rows come from the **train** split: both panels measure runtime and pose
quality, neither of which needs a held-out label, so spending test rows on them
would be waste.

### Pilot — 5 complexes ([`reports/boltz_pilot.csv`](../reports/boltz_pilot.csv))

The plan requires 3–5 examples pushed end to end before any batch job, and
`CLAUDE.md` makes it an invariant. Every pilot complex is **TCR-free, in the
train split, and has a published crystal structure at ≤ 1.9 Å**, so the pilot
doubles as the register check [STRUCTURES.md](STRUCTURES.md) asks for — does the
predicted peptide land in the canonical P2/PΩ-anchored pose where the answer is
already known?

| complex | t½ (h) | PDB entries | best resolution |
| --- | ---: | --- | ---: |
| HLA-A*11:01 / KTFPPTEPK | 59.3 | 1X7Q, 7WKJ | 1.45 Å |
| HLA-A*02:01 / LLWNGPMAV | 38.2 | 5N6B, 9SL0 | 1.60 Å |
| HLA-B*15:01 / ILGPPGSVY | 11.0 | 1XR9 | 1.79 Å |
| HLA-B*07:02 / IPRRNVATL | 4.0 | 7LFZ | 1.90 Å |
| HLA-B*08:01 / ELRRKMMYM | 0.75 | 4QRU | 1.60 Å |

Resolution is the best across the listed entries, as recorded in
`data/pdb_rasmussen_overlap.csv`; pick the specific entry per complex when
running the RMSD comparison.

TCR-free matters because TCR engagement can perturb the peptide conformation,
which would make an RMSD check measure the crystal's binding partner rather than
our prediction. The 0.75–59.3 h spread means a failure confined to one end of
the label range is visible rather than averaged away.

### GPU benchmark — 24 complexes ([`reports/boltz_bench_panel.csv`](../reports/boltz_bench_panel.csv))

Four complexes from each of the six best-represented alleles — B\*15:01,
A\*02:01, A\*03:01, B\*39:01, B\*35:01, B\*07:02 — picked by striding across each
allele's cluster-sorted train rows. Those six are also the production-panel
candidates: each holds 121–222 test rows, clearing the plan's "at least 50
held-out examples per allele, ideally closer to 100" bar by a wide margin.

Every complex is 191 residues (182 + 9), so **per-complex runtime should be
near-constant and the benchmark isolates hardware rather than input size.** If
the spread turns out to be wide, that itself is a finding — it would mean
something other than sequence length drives runtime, and the production cost
estimate needs the p90, not the median.

## Residue and chain mapping: verified

The plan asks to "verify residue/chain mapping" before trusting any structural
feature. Checked against all six deposited entries behind the pilot
([`scripts/boltz_pose_check.py`](../scripts/boltz_pose_check.py) helpers, run
against freshly fetched RCSB mmCIFs):

- **`hla_seq` is exactly residues 1–182 of the deposited heavy chain.** The
  182-mer matches at **offset 0 of chain A** in every entry — 1X7Q, 7WKJ, 5N6B,
  9SL0, 1XR9, 7LFZ, 4QRU. No alignment or offset correction is needed, and the
  same indexing carries to the 34 contact positions used elsewhere.
- **The peptide is chain C**, 9 residues, exact sequence match in every entry.
- 5N6B holds two copies in the asymmetric unit (A–C and D–F); the checker takes
  the first matching pair.

**One caveat this exposes: every pilot crystal contains β2-microglobulin as
chain B (99–100 residues), and our input does not.** We fold a 2-chain complex
against a 3-chain structure. The plan defers full-length HLA with β2m
explicitly — it "would require a new pilot and runtime benchmark" — so this is
a known scope decision, not an oversight. It is defensible for this comparison
because β2m sits beneath the α1/α2 platform rather than in the groove, and the
pose check superposes on the α1/α2 domain and measures only the peptide. It
remains a caveat on absolute RMSD, and belongs in the limitations section.

## Benchmark protocol

[`modal_app/boltz_bench.py`](../modal_app/boltz_bench.py). Identical settings on
every GPU; only the GPU changes, via `fold_batch.with_options(gpu=...)` so there
is exactly one code path.

- **GPUs:** L40S 48 GB, A100 40 GB, A100 80 GB, H100 80 GB.
- **Settings:** `--diffusion_samples 1` (one pose), `--recycling_steps 3` and
  `--sampling_steps 200` (defaults), `--output_format mmcif`, and
  **`--write_full_pae`** because stage 5 feature extraction needs the matrix.
- **Pinned:** `boltz==2.1.1` and the Boltz-2 weight snapshot by git revision. A
  floating version would make the benchmark unreproducible.
- **Weights (~6.2 GB) on a Modal Volume**, mounted at `--cache`, never
  downloaded inside a timed region.
- **A batch runs in one container** so weights load once — the plan's "keep
  models loaded across complexes".
- **Warm-up fold excluded from the steady-state mean.** The first complex pays
  weight load and CUDA kernel compilation. Charging that to every complex would
  overstate a 2,000-complex run, where it is paid once per container. It is
  measured and then *amortised* over the production batch size rather than
  dropped, because Modal explicitly bills container load time.
- **Instrumented per complex:** fold wall time, container startup, peak GPU
  memory (polled from `nvidia-smi` on a thread — Boltz runs as a subprocess, so
  in-process torch counters would see nothing), success/failure with stderr, and
  the output confidence scores (ipTM, pTM, pLDDT) plus whether the PAE actually
  landed.

### Two Modal-specific traps this encodes

- **`gpu="H100!"`, with the bang.** Plain `"H100"` lets Modal substitute an
  H200. The H100 column would then silently be an H200 measurement at H100
  prices in our model. Modal's GPU docs flag this for benchmarking specifically.
- **Startup is billed, and the default container idle tail is 60 s.** At ~1
  minute per fold, a 60 s tail per container is a large fraction of the bill, so
  `scaledown_window` is set explicitly to 10 s. Modal also charges the greater of
  requested and used CPU/memory, and the published Boltz example sets neither —
  defaulting to 0.125 cores and 128 MiB. Ours requests 4 cores / 32 GiB to match
  the plan's rate table, so measured cost and budgeted cost are the same number.

## The decision rule

[`scripts/gpu_decision.py`](../scripts/gpu_decision.py) applies the plan's rule —
*pick the cheapest GPU that meets the deadline and memory requirements* — to the
measured timings:

```text
cost per success = mean billed seconds × combined $/s ÷ success rate
capacity         = folding budget ÷ cost per success
elapsed hours    = total worker hours ÷ concurrent workers
```

Failures are charged to the successes: a complex that crashes still burns worker
time, so dividing by the success rate is what makes "cost per *successful*
complex" the honest number. A GPU is eligible only if it fits ~2,000 complexes
inside the **$330** folding ceiling, hits the wall-clock deadline at the planned
concurrency, and never exceeded its own VRAM.

Two traps this avoids:

- **Cheapest per hour is not cheapest per structure.** A GPU at half the hourly
  rate that takes three times as long costs more per complex. The plan is
  explicit that the metric is dollars per successful prediction.
- **Concurrency buys wall-clock, not cost.** Running 20 workers instead of 10
  halves elapsed time and changes total spend by nothing. Deadline and budget are
  separate constraints and must be checked separately.

### Rates: rechecked

`gpu_decision.py` computes from Modal's published **per-second** rates, read
2026-10-03, rather than hardcoding the plan's rounded hourly figures:

| GPU | $/s (GPU) | $/hr (GPU) | $/hr + 4 cores + 32 GiB | plan's table |
| --- | ---: | ---: | ---: | ---: |
| L40S | 0.000542 | 1.9512 | **2.3956** | 2.40 |
| A100-40GB | 0.000583 | 2.0988 | **2.5432** | 2.54 |
| A100-80GB | 0.000694 | 2.4984 | **2.9428** | 2.94 |
| H100 | 0.001097 | 3.9492 | **4.3936** | 4.39 |

Host rates: CPU $0.0000131/core/s, memory $0.00000222/GiB/s. **All four
reproduce the plan's reference table to the cent**, so the plan's "recheck rates
before launch" step is discharged — recheck again only if the event slips.

One caution: modal.com/pricing carries a *second* CPU/memory table for Sandboxes
and Notebooks at roughly 3× the Function rates. The figures above are the
Function rates, which is what `@app.function` bills at.

## Measured: the pilot passed, and it reframes the whole decision

Run 2026-10-03, 3 complexes on L40S, MSA arm, `--diffusion_samples 1`,
`--write_full_pae`. Cost about **$0.20**.

| complex | fold | peak GPU mem | ipTM | PAE | peptide CA RMSD |
| --- | ---: | ---: | ---: | :---: | ---: |
| A\*11:01 / KTFPPTEPK | 168.1 s *(warm-up)* | 8.59 GB | 0.982 | yes | **0.39 Å** |
| A\*02:01 / LLWNGPMAV | 43.4 s | 8.59 GB | 0.993 | yes | **0.13 Å** |
| B\*15:01 / ILGPPGSVY | 43.8 s | 8.59 GB | 0.990 | yes | **0.42 Å** |

**The register check passes decisively.** Peptide CA RMSD against crystal, after
superposing on the HLA domain only, is 0.13–0.42 Å — essentially
crystallographic agreement. Per-position deviation never exceeds 0.85 Å, and
the P2/PΩ anchors are the tightest positions. The predicted peptides sit in the
canonical pose, in the right register, so geometry features extracted from these
structures are measuring the complex and not a docking artefact.

Also settled: **`msa: empty` on the peptide coexists fine with a precomputed
`.a3m` on the HLA chain** — the combination the Boltz docs never show. No
paired-CSV fallback needed.

### Peak memory is 8.59 GB, and that changes the candidate set

Every GPU in the plan's table has **4.7× to 9.3× headroom**. Two consequences:

- **A100-80GB is strictly dominated by A100-40GB.** Same silicon, 15% higher
  rate, and the extra 40 GB buys nothing at 8.6 GB of demand. It leaves the
  sweep on measurement, not assumption.
- **The plan's candidate set was chosen without knowing this.** Cards the plan
  never considered now fit easily: L4 (24 GB), A10 (24 GB), even T4 (16 GB).
  Whether they are *cheaper per complex* is a different question — see below.

### Budget is not the binding constraint

Projected from the measured steady-state 43.6 s/complex and the 124.5 s
per-container overhead (weight load plus CUDA kernel compilation), at 100
complexes per container:

| | L40S |
| --- | ---: |
| Billed per complex | 44.8 s |
| Cost per complex | $0.0298 |
| **2,000-complex panel** | **$59.7** — 18% of the $330 ceiling |
| Capacity at $330 | ~11,000 complexes |
| Wall-clock at 10 workers | 2.5 h |

**L40S alone already fits both the budget and the deadline with room to spare.**
That reframes the GPU question: it is no longer "can we afford to fold 2,000
complexes" but "is anything meaningfully cheaper or faster than the baseline
card". Two specific open questions:

- **H100 needs to be >1.83× faster than L40S to also be cheaper per complex**
  ($4.3936 / $2.3956). Plausible but unmeasured.
- **Cheap cards are less attractive than their GPU rate suggests**, because the
  host floor (4 cores + 32 GiB = $0.444/hr) is a large share of a budget card's
  total. L4 only wins if it is under ~1.9× slower than L40S. Also worth noting:
  that host request is itself tunable — 8.6 GB of GPU demand does not obviously
  need 32 GiB of host RAM, and dropping to 2 cores / 8 GiB would cut the L40S
  combined rate from $2.396 to $2.109/hr.

## Open question before launch

**Which Modal plan is the workspace on?** Starter caps GPU concurrency at **10
containers**; Team raises it to 50. That cap sets the `--workers` figure in the
deadline arithmetic, and it is the difference between a ~3.5 h and a ~45 min
production run. It changes no cost figure — concurrency buys wall-clock, not
dollars.

## Status

| step | state |
| --- | --- |
| Panels selected | done |
| Rates rechecked against published pricing | done |
| Cost/decision model | done, unit-checked on synthetic timings |
| MSA precompute (8 alignments, verified) | **done**, ~$0.01 |
| Weights cached to Volume (6.204 GB) | **done** |
| Pilot: 3 complexes on L40S | **done**, 3/3, ~$0.20 |
| Register check vs crystal | **done**, 0.13–0.42 Å peptide CA RMSD |
| Hardware sweep across GPUs | pending scope decision |

The `CLAUDE.md` gate — "no batch GPU job without a passing end-to-end pilot on
3–5 examples" — **is cleared**: 3 complexes folded end to end, all three in the
groove, PAE written, chain and residue mapping verified.

Still unmeasured: **every GPU other than L40S.** The A100-40GB, A100-80GB and
H100 figures in the rate table are published prices, not measured throughput,
so no cross-GPU claim can be made yet. Total spend to date is about **$0.21**.
