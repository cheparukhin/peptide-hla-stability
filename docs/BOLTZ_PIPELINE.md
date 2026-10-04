# Completed two-chain Boltz-2 pipeline and GPU benchmark

Historical protocol and how-to for the **182-residue groove + 9-residue
peptide**, two-chain workload completed in stage 4b. It is superseded by stage
4c, which refolded the full ectodomain construct; see
[HACKATHON_PLAN.md, stage 4c](../HACKATHON_PLAN.md#4c-ectodomain--beta-2-microglobulin-folding)
and [the stage 4c report](../reports/stage4c_ectodomain_pilot.md).

The measured two-chain throughput, cost, GPU ranking and pose tables are the
authoritative record in [reports/stage4_benchmark.md](../reports/stage4_benchmark.md);
those numbers are not repeated here. What this document keeps is the input
builders, mapping rules and Modal cost traps — the setup that the benchmark
assumes. The single-engine recommendation it once carried is also superseded:
stage 4c ran both Boltz-2 and ESMFold2 on the same constructs and prepared MSAs,
and selected Boltz-2 on pose quality.

## MSAs: fewer than you would think, and not on the GPU

The plan says "cache model weights and HLA MSAs, use single-sequence peptide
input, don't spend GPU time waiting on MSA generation." Three facts make that
cheap:

**An MSA is per unique chain sequence, not per complex.** Every row in the
dataset is a 182-residue HLA domain plus a 9-residue peptide, and the dataset
holds **75 unique HLA domain sequences** across its 28,166 pairs. So:

| Scope | Complexes | HLA MSAs needed |
| --- | ---: | ---: |
| Pilot | 5 | 5 |
| Pilot + GPU benchmark | 29 | **8** |
| Production panel (6 alleles) | ~2,000 | **6** |
| Entire dataset | 28,166 | **75** |

An MSA (multiple sequence alignment) is the stack of homologous sequences a
folding model conditions on. Building six of them is a CPU job measured in
minutes, not a workload to budget GPU hours against.

**The peptide gets no MSA.** A 9-residue query cannot produce a meaningful
alignment — there is no reliable homology signal at that length, and the peptide
identity is precisely the variable under study. Peptide chains run
single-sequence.

**Therefore MSA generation must not run on the GPU worker.** Reading
`boltz/main.py` confirms `predict` calls `compute_msa` → `run_mmseqs2`
**synchronously, in-process, before the Lightning trainer starts**. With
`--use_msa_server`, the accelerator sits idle blocking on HTTP to
`api.colabfold.com` and Modal bills it at the GPU rate the whole time. Folding
2,000 complexes that way would issue 2,000 round-trips for six distinct answers
and hammer a free shared community service for results we already have.

**Modal's own published Boltz example does exactly this** — it passes
`--use_msa_server` on an H100. That example is a single-input demo, and copying
its shape into a batch sweep is the trap. Our MSAs are built in a separate
CPU-only Modal function, written to a Volume, and referenced by path.

### Measured: the MSA step costs about a penny

Run 2026-10-03 on Modal, CPU only (`boltz_setup.py::setup`): the weight snapshot
cached to the Volume in 36.5 s (6.204 GB); 8 unique HLA MSAs built in **46.1 s**
as one batched request, at alignment depth **9,771–10,305** sequences each, for
roughly **$0.01**. Class I HLA is a deeply sampled family, so these are rich
alignments — the substantive reason to keep the MSA arm rather than fold
single-sequence.

Two notes from the run:

- **The 46 s was a single batched request for all 8 sequences.** The progress
  bar's early estimate said 22 minutes; it finished in 45 s once the server
  dequeued it. Server queue time is the variable here, not compute.
- **`build_msas` requests 0.25 cores / 2 GiB, not the fold worker's 4 cores /
  32 GiB.** It spends its life in a polling loop waiting on the MSA server, so it
  is I/O-bound; Modal bills the greater of requested and used, and asking for the
  fold shape would have cost roughly 10× for the same wait. This matters more at
  75 alleles than at 8.

### What the pilot checks about MSAs

Boltz's docs say single-sequence mode "reduces accuracy" and is "not
recommended", and our MSAs cost essentially nothing, so **MSA-on is the default
and single-sequence is not a serious contender.** The pilot still ran both arms
on the same 5 complexes, for two reasons worth 5 extra folds:

- It quantifies what the MSA buys on this task, against crystal structures.
- **It verifies a combination the Boltz docs do not document.** A precomputed
  `.a3m` is only valid when a *single* protein chain carries an MSA; two aligned
  chains would need the paired-CSV format. Our YAML satisfies that by setting
  `msa: empty` on the peptide — but mixing `msa: empty` on one chain with a real
  `.a3m` on another is not shown in any documented example. The pilot confirmed
  it works (below); the fallback was paired CSV.

## Panels

Built by [`scripts/boltz_panel.py`](../scripts/boltz_panel.py), deterministic, no
RNG. All rows come from the **train** split: both panels measure runtime and pose
quality, neither of which needs a held-out label.

### Pilot — 5 complexes ([`reports/boltz_pilot.csv`](../reports/boltz_pilot.csv))

The plan requires 3–5 examples pushed end to end before any batch job, and
`CLAUDE.md` makes it an invariant. Every pilot complex is **TCR-free, in the
train split, and has a published crystal structure at ≤ 1.9 Å**, so the pilot
doubles as the register check [STRUCTURES.md](STRUCTURES.md) asks for.

| complex | t½ (h) | PDB entries | best resolution |
| --- | ---: | --- | ---: |
| HLA-A*11:01 / KTFPPTEPK | 59.3 | 1X7Q, 7WKJ | 1.45 Å |
| HLA-A*02:01 / LLWNGPMAV | 38.2 | 5N6B, 9SL0 | 1.60 Å |
| HLA-B*15:01 / ILGPPGSVY | 11.0 | 1XR9 | 1.79 Å |
| HLA-B*07:02 / IPRRNVATL | 4.0 | 7LFZ | 1.90 Å |
| HLA-B*08:01 / ELRRKMMYM | 0.75 | 4QRU | 1.60 Å |

Resolution is the best across the listed entries, as recorded in
`data/pdb_rasmussen_overlap.csv`; pick the specific entry per complex when
running the RMSD comparison. TCR-free matters because TCR engagement can perturb
the peptide conformation, which would make an RMSD check measure the crystal's
binding partner rather than our prediction. The 0.75–59.3 h spread means a
failure confined to one end of the label range is visible rather than averaged
away.

### GPU benchmark — 24 complexes ([`reports/boltz_bench_panel.csv`](../reports/boltz_bench_panel.csv))

Four complexes from each of the six best-represented alleles — B\*15:01,
A\*02:01, A\*03:01, B\*39:01, B\*35:01, B\*07:02 — picked by striding across each
allele's cluster-sorted train rows. Those six are also the production-panel
candidates: each holds 121–222 test rows, clearing the plan's per-allele
held-out floor by a wide margin. Every complex is 191 residues (182 + 9), so
per-complex runtime should be near-constant and the benchmark isolates hardware
rather than input size.

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

**Historical caveat: every pilot crystal contains β2-microglobulin as chain B
(99–100 residues), and the two-chain predictions omitted it.** Those pose checks
fit the α1/α2 groove and measure the peptide. Stage 4c tests the full ectodomain
+ β2m + peptide construct; the omission and central-bulge error below remain
limitations of the earlier two-chain benchmark.

## Benchmark protocol

[`modal_app/boltz_bench.py`](../modal_app/boltz_bench.py) applies identical
settings on every GPU; only the GPU changes, via `fold_batch.with_options(gpu=...)`
so there is exactly one code path. The measured throughput, cost and GPU ranking
are in [reports/stage4_benchmark.md](../reports/stage4_benchmark.md). The
engineering choices worth carrying forward:

- **Settings:** `--diffusion_samples 1`, `--recycling_steps 3`,
  `--sampling_steps 200`, `--output_format mmcif`, and **`--write_full_pae`**
  because stage 5 feature extraction needs the PAE matrix.
- **Pinned:** `boltz==2.1.1` and the Boltz-2 weight snapshot by git revision. A
  floating version would make the benchmark unreproducible.
- **Weights (~6.2 GB) on a Modal Volume**, mounted at `--cache`, never
  downloaded inside a timed region.
- **A batch runs as ONE `boltz predict` process over a directory of YAMLs**, so
  weights load once. Per-complex timings are recovered from that single process
  by watching the output tree for each structure to land (`_CompletionProbe`).

  > This was **got wrong the first time and is the most expensive mistake in
  > this stage.** The original harness spawned a fresh `boltz predict` per
  > complex, putting a Python start, torch import, 6.2 GB weight load and CUDA
  > init inside every timed fold — 86% of each measurement. It reported
  > `startup_s = 0.0` because, by its own definition, nothing happened before the
  > first fold. The shape came from Modal's single-input Boltz example, where the
  > weight load is invisible because nothing amortises it. The idiomatic fix is
  > `modal.Cls` with `@modal.enter()`. See finding 1 in the benchmark report.

- **Warm-up fold excluded from the steady-state mean**, but measured as
  `load_plus_first_s` and *amortised* over the production batch size rather than
  dropped, because Modal explicitly bills container load time.
- **boltz's stdout is retained** (`stdout_tail`); the original harness discarded
  it on success, which made the overhead above unrecoverable without a re-run.
- **Per complex:** fold wall time, container startup, peak GPU memory (polled
  from `nvidia-smi` on a thread — Boltz runs as a subprocess, so in-process torch
  counters see nothing), success/failure with stderr, confidence scores (ipTM,
  pTM, pLDDT), and whether the PAE landed.

### Modal-specific traps this encodes

- **`gpu="H100!"`, with the bang.** Plain `"H100"` lets Modal substitute an
  H200, which would silently be an H200 measurement at H100 prices. Modal's GPU
  docs flag this for benchmarking specifically.
- **Startup is billed, and the default container idle tail is 60 s.** At ~1
  minute per fold that tail is a large fraction of the bill, so `scaledown_window`
  is set explicitly to 10 s. Modal also charges the greater of requested and used
  CPU/memory, and the published Boltz example sets neither — defaulting to 0.125
  cores and 128 MiB. Ours requests 4 cores / 16 GiB, right-sized after measuring
  peak Boltz child RSS at 9.89 GB (the plan's original 32 GiB was ~3× more than
  needed and biased the comparison toward expensive GPUs).
- **modal.com/pricing carries a second CPU/memory table** for Sandboxes and
  Notebooks at roughly 3× the Function rates. `@app.function` bills at the
  Function rates; use those.

### The decision rule

[`scripts/gpu_decision.py`](../scripts/gpu_decision.py) applies the plan's rule —
*pick the cheapest GPU that meets the deadline and memory requirements* — to the
measured timings, computing from Modal's published per-second rates rather than
hardcoding rounded hourly figures. Two traps it encodes:

- **Cheapest per hour is not cheapest per structure.** A GPU at half the hourly
  rate that takes three times as long costs more per complex; the metric is
  dollars per *successful* prediction, with failures charged to the successes.
- **Concurrency buys wall-clock, not cost.** Running 20 workers instead of 10
  halves elapsed time and changes total spend by nothing. Deadline and budget are
  separate constraints, checked separately.

The resulting ranking, rates and dollar figures are in
[reports/stage4_benchmark.md](../reports/stage4_benchmark.md).

## The pilot passed, and what it tells stage 5

The 5-complex pilot (L40S, MSA arm, `--diffusion_samples 1`, `--write_full_pae`)
cleared the `CLAUDE.md` gate: **5/5 peptides in the groove, median CA RMSD
0.29 Å**, PAE written, chain and residue mapping verified. The register check
passes, so geometry features taken from these structures measure the complex
rather than a docking artefact. Per-complex RMSDs are in
[reports/stage4_benchmark.md](../reports/stage4_benchmark.md).

Two findings carry into stage 5:

- **The error is not uniform along the peptide.** Anchors (P1–P2, P8–P9) are
  reliably placed in both engines, while the **central bulge (P5–P7) carries
  several-angstrom uncertainty on some complexes** — B\*07:02 / IPRRNVATL
  deviates 3.33 Å at P6. Burial and contact features at central positions are
  intrinsically noisier than the same features at anchors and should not be
  treated as equally well determined.
- **`msa: empty` on the peptide coexists fine with a precomputed `.a3m` on the
  HLA chain** — the combination the Boltz docs never show. No paired-CSV fallback
  needed.

## ESMFold2: historical alternative engine

[ESMFold2](https://huggingface.co/biohub/ESMFold2) (Biohub, MIT) is a diffusion
predictor on ESMC embeddings with native multi-chain input and a richer
confidence set (`pae`, `plddt`, `ptm`, `iptm`, plus `pair_chains_iptm`).
Benchmarked under `modal_app/esmfold_*.py`; it runs in-process with the model
resident, so it was never affected by the harness bug above. Stage 4c settled the
engine choice on a matched experiment and selected Boltz-2; the measured
comparison (peak memory, speed, pose quality) is in
[reports/stage4_benchmark.md](../reports/stage4_benchmark.md). The short version:
ESMFold2 needs 26 GB, which removes the cheapest 24 GB cards, and was less
accurate at matched n=5.

Two results worth retaining:

- **A small two-chain MSA test did not improve ESMFold2 poses.** Given the same
  alignments Boltz-2 used, B1501 degraded 1.35 → 1.72 Å while fold time rose
  32%. Single-sequence was better in these tested cases; this is not evidence to
  omit MSAs from the full-construct comparison.
- **`pair_chains_iptm` contains the peptide-HLA interface ipTM** that stage 5
  wants as a confidence feature. It is a per-chain-pair **matrix**, not a scalar:
  in the three-chain arm-B construct exactly one entry is the peptide-HLA pair,
  and that entry is direction-dependent. Boltz 2.1.1 also emits
  `pair_chains_iptm` on all 45 pilot folds alongside `global_iptm`, verified at
  stage 4c.5, so this score is available from Boltz-2 and is not a reason to
  prefer ESMFold2.

  **Its orientation is transposed relative to the naive reading.** From
  `compute_ptms` in the pinned source, `pair_chains_iptm[a][b]` scores PAE **rows
  in b, columns in a**, via a max over rows. Read in the source orientation it
  correlates −0.98 to −1.00 with pose error across all 8 ordered chain pairs;
  read naively, one pair reaches +0.02. A pooled test would have selected the
  wrong permutation, so any column naming this score must name the pair **and the
  direction**. Global ipTM is `global_iptm` and stays named as such: in a
  three-chain complex it covers interfaces that are not peptide-HLA.

## Historical status

The two-chain stage 4b work (panels selected, rates rechecked, cost model
unit-checked, MSAs precomputed, weights cached, harness bug fixed, pilot passed
5/5, hardware sweep run) is complete. Its one open item — a multi-container
re-run (~3 per GPU), because n=1 cannot rank the middle cards — was not pursued,
because stage 4c moved to the full ectodomain construct and refolded everything
there. The `CLAUDE.md` gate was cleared for the two-chain run; stage 4c ran its
own pilot gate for the production construct.

Everything downstream of this document — engine choice, production fold, and the
scored structural arm — is in
[the stage 4c report](../reports/stage4c_ectodomain_pilot.md) and
[reports/REPORT.md](../reports/REPORT.md).
