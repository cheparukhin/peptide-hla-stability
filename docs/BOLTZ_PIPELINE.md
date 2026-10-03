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

- **GPUs tested:** L4 24 GB, A10 24 GB, L40S 48 GB, A100 40 GB, H100 80 GB.
  A100-80GB was dropped after the pilot measured peak GPU memory at 8.59 GB,
  making it strictly dominated by A100-40GB. L4 and A10 were added because
  8.59 GB fits comfortably in a 24 GB card.
- **Settings:** `--diffusion_samples 1` (one pose), `--recycling_steps 3` and
  `--sampling_steps 200` (defaults), `--output_format mmcif`, and
  **`--write_full_pae`** because stage 5 feature extraction needs the matrix.
- **Pinned:** `boltz==2.1.1` and the Boltz-2 weight snapshot by git revision. A
  floating version would make the benchmark unreproducible.
- **Weights (~6.2 GB) on a Modal Volume**, mounted at `--cache`, never
  downloaded inside a timed region.
- **A batch runs as ONE `boltz predict` process over a directory of YAMLs**, so
  weights load once — the plan's "keep models loaded across complexes".
  Per-complex timings are recovered from that single process by watching the
  output tree for each structure to land (`_CompletionProbe`).

  > This was **got wrong the first time and is the most expensive mistake in
  > this stage.** The original harness spawned a fresh `boltz predict` per
  > complex, putting a Python start, torch import, 6.2 GB weight load and CUDA
  > init inside every timed fold — 86% of each measurement. It reported
  > `startup_s = 0.0` because, by its own definition, nothing happened before
  > the first fold. The shape came from Modal's published Boltz example, which
  > folds **one** input per call; there the weight load is invisible because
  > nothing amortises it. The idiomatic fix is `modal.Cls` with
  > `@modal.enter()`. See finding 1 in the benchmark report.

- **Warm-up fold excluded from the steady-state mean.** The first interval pays
  weight load and CUDA kernel compilation. Charging that to every complex would
  overstate a 2,000-complex run, where it is paid once per container. It is
  measured as `load_plus_first_s` and *amortised* over the production batch size
  rather than dropped, because Modal explicitly bills container load time.
- **boltz's stdout is retained** (`stdout_tail`). The original harness discarded
  it on success, which made the overhead above unrecoverable without a re-run.
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
  defaulting to 0.125 cores and 128 MiB. Ours requests 4 cores / 16 GiB,
  right-sized after measuring peak Boltz child RSS at 9.89 GB (the plan's
  original 32 GiB was 3x more than needed and biased the comparison toward
  expensive GPUs — see the benchmark report for the analysis).

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

| GPU | $/s (GPU) | $/hr (GPU) | $/hr + 4c/16 GiB |
| --- | ---: | ---: | ---: |
| L4 | 0.000222 | 0.7992 | **1.1157** |
| A10 | 0.000306 | 1.1016 | **1.4181** |
| L40S | 0.000542 | 1.9512 | **2.2677** |
| A100-40GB | 0.000583 | 2.0988 | **2.4153** |
| H100 | 0.001097 | 3.9492 | **4.2657** |

Host rates: CPU $0.0000131/core/s, memory $0.00000222/GiB/s. The plan's original
four GPUs at 4c/32 GiB reproduce its reference table to the cent, so the
"recheck rates before launch" step is discharged. The table above uses the
right-sized 4c/16 GiB host request. Recheck if the event date slips.

One caution: modal.com/pricing carries a *second* CPU/memory table for Sandboxes
and Notebooks at roughly 3x the Function rates. The figures above are the
Function rates, which is what `@app.function` bills at.

## Measured: the pilot passed, and it reframes the whole decision

Run 2026-10-03 on L40S, MSA arm, `--diffusion_samples 1`, `--write_full_pae`,
5 complexes with the corrected one-process harness (weight load + first fold
53.9 s, then 6.3 s per complex).

| complex | PDB | peptide CA RMSD | max per-residue | ipTM |
| --- | --- | ---: | ---: | ---: |
| A\*11:01 / KTFPPTEPK | 1X7Q | **0.29 Å** | 0.74 Å | 0.981 |
| A\*02:01 / LLWNGPMAV | 9SL0 | **0.14 Å** | 0.31 Å | 0.993 |
| B\*15:01 / ILGPPGSVY | 1XR9 | **0.38 Å** | 0.70 Å | 0.990 |
| B\*07:02 / IPRRNVATL | 7LFZ | **1.30 Å** | 3.33 Å | 0.991 |
| B\*08:01 / ELRRKMMYM | 4QRU | **0.25 Å** | 0.37 Å | 0.990 |

**The register check passes: 5/5 in groove**, median 0.29 Å. Peak GPU memory
8.6 GB throughout. The predicted peptides sit in the canonical pose in the right
register, so geometry features taken from these structures measure the complex
rather than a docking artefact.

**But the error is not uniform along the peptide, and stage 5 needs to know
that.** B\*07:02 / IPRRNVATL deviates 1.90 Å at P5 and 3.33 Å at P6 while
P1–P2 and P8–P9 all stay under 0.25 Å. ESMFold2 shows the same pattern on two
complexes. So the anchors are reliably placed in both engines while **the
central bulge (P5–P7) carries several-angstrom uncertainty on some complexes** —
burial and contact features computed at central positions are intrinsically
noisier than the same features at anchors, and should not be treated as equally
well determined.

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

With the corrected harness, A10 folds at **$0.004/complex**: $8.00 for a 2,000
panel (2% of the $330 ceiling) and ~$113 for the entire 28,166-pair dataset.
Folding is not a budget question on this project at all. Wall-clock and GPU
concurrency are the only real limits.

The benchmark confirmed one prediction from the pilot and refuted the cost
figures:

- **H100 did not break even**, and this survived both corrections. It needs
  3.01x the speed of A10 to justify its rate and measured 1.35x. The gap is
  absolute (2.5 s), not a ratio that subtracting overhead could rescue.
- **Cheap cards win**, but only after right-sizing the host request. At the
  plan's 4c/32 GiB the fixed host cost inflated cheap cards' bills
  disproportionately (29% of A10's bill vs 10% of H100's). At 4c/16 GiB the
  bias is removed.

### Between-container variance is the real measurement limit

The same model, GPU type and settings gave **6.3 s in one container and 10.9 s
in another** — 73% apart, against ±1% *within* a container. Modal places each
container on whatever host is free and those hosts differ.

**The sweep's n=1 container per GPU is therefore too thin to rank the middle
cards.** A10's cost lead and the H100 verdict exceed the variance; the 10.9 vs
10.1 s L40S/A100 ordering does not. Any re-run should use ~3 containers per GPU
and report a median of container medians — and since within-container spread is
negligible, it should trade folds-per-container for container count rather than
adding folds.

## ESMFold2: evaluated, rejected

[ESMFold2](https://huggingface.co/biohub/ESMFold2) (Biohub, MIT) is a diffusion
predictor on ESMC embeddings with native multi-chain input, published DockQ wins
over AlphaFold 3 on protein-protein interfaces, and a richer confidence set
(`pae`, `plddt`, `ptm`, `iptm`, plus `pair_chains_iptm`). Benchmarked under
`modal_app/esmfold_*.py`; it runs in-process with the model resident, so it was
never affected by the harness bug above.

Rejected on three measurements, not on principle:

- **26.0 GB peak** vs Boltz-2's 8.4 GB. A 7B ESMC backbone plus diffusion trunk
  does not fit a 24 GB card even with bf16 weights, which removes A10 and L4 —
  the two cheapest options — from the table entirely.
- **Speed indistinguishable.** Its 6.7 s on L40S sits inside Boltz-2's own
  6.3–10.9 s between-container range on the same card.
- **Pose quality worse at matched n=5**: median peptide CA RMSD 0.68 A against
  Boltz-2's 0.29 A, with 3 of 5 poses over 0.5 A against 1 of 5.

Two results worth retaining:

- **MSAs do not help ESMFold2 on pMHC.** Given the same alignments Boltz-2 uses,
  B1501 degraded 1.35 -> 1.72 A while fold time rose 32%. Single-sequence is
  strictly better here, which contradicts Biohub's general-case guidance and is
  therefore a pMHC-specific finding.
- **`pair_chains_iptm` is the peptide-HLA interface ipTM** that stage 5 wants as
  a confidence feature; Boltz-2 exposes only a global ipTM. If that feature
  earns its place, ESMFold2 is the cheaper way to get it.

## Open question before launch

**Which Modal plan is the workspace on?** Starter caps GPU concurrency at **10
containers**; Team raises it to 50. That cap sets the `--workers` figure in the
deadline arithmetic: 0.56 h at 10 workers against ~7 min at 50. It changes no
cost figure — concurrency buys wall-clock, not dollars.

## Status

| step | state |
| --- | --- |
| Panels selected | done |
| Rates rechecked against published pricing | done |
| Cost/decision model | done, unit-checked on synthetic timings |
| MSA precompute (8 alignments, verified) | **done**, ~$0.01 |
| Weights cached to Volume (6.204 GB) | **done** |
| Host-memory probe (peak RSS 9.89 GB) | **done**, ~$0.03 |
| Harness bug found and fixed (one process per batch) | **done**, invalidated the first sweep |
| Pilot: 5 complexes on L40S | **done**, 5/5 in groove, median 0.29 Å |
| Hardware sweep: 8 complexes x 5 GPUs | **done** (re-run after the fix) |
| ESMFold2 evaluated as an alternative engine | **done**, rejected on VRAM + pose quality |
| **Engine + GPU chosen: Boltz-2 on A10** | $0.004/complex, $8.00 per 2,000, 0.56 h at 10 workers |
| Multi-container re-run (~3 per GPU) | **open** — n=1 cannot rank the middle cards |

The `CLAUDE.md` gate — "no batch GPU job without a passing end-to-end pilot on
3–5 examples" — **is cleared**: 5 complexes folded end to end, all five in the
groove, PAE written, chain and residue mapping verified.

Results in [reports/stage4_benchmark.md](../reports/stage4_benchmark.md).
Headline: **Boltz-2 on A10 at $0.004/complex**, $8.00 per 2,000 complexes.
H100 measured 1.35x faster against the 3.01x it needed to break even.
A100-40GB was slower than A10 in three independent allocations. ESMFold2 is
genuinely competitive on cost but needs 26 GB, which removes the cheap cards,
and is less accurate at matched n=5. Total spend **~$2.64** against the plan's
$15 pilot/benchmark ceiling.

Two numbers in this document were previously wrong and are corrected above: the
per-complex cost (was $0.018, from the harness bug) and the claim that weights
loaded once per batch (they did not, until the fix).

A100-80GB was never run: the pilot's 8.6 GB peak made it strictly dominated by
A100-40GB, and the sweep then showed A100-40GB itself was uncompetitive.
