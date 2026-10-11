> **Historical study record.** The protocols, results and future-tense instructions below describe the completed October study. The [current project review and research plan](../NEXT_RESEARCH_PLAN.md) is the sole forward work queue; its [dated correction register](../NEXT_RESEARCH_PLAN.md#dated-correction-register--12-october-2026) supersedes conflicting interpretation, selection and blend-gate claims here. No old protocol should be relaunched as a new study without the current evaluation gates.

# Phase C gate: joint peptide–HLA interaction embeddings

Date: 4 October 2026. Scope: step 1 and the gate of
[FURTHER_RESEARCH_PLAN.md §6](../archive/plans/further-research-2026-10-04.md). The full Phase C
comparison has **not** been started. No validation or test label was read; the
test split remains scored once, at stage 6.

## 1. Verdict

**Gate passed on every technical axis. §6.2 was approved, has run, and is a
negative result; the blend condition was then checked and is also negative, so
§6's own stop-scaling criterion is satisfied. Recommendation: stop Phase C.**
§6.3's cost is unresolved with a **$39.72 floor** — the 20-epoch assumption behind
$61.01 is not refuted — and it is unmotivated regardless of price. §6.4 remains a
separate unbudgeted sub-project.

Updated 5 October. Two figures from an earlier version of this report are
**withdrawn**: a "$30.59 measured" cost for §6.3 and a "~0.47" adaptation ceiling.
Both rested on reading a single-epoch blip in a mid-run checkpoint as the point
where early stopping bites; the curve had not turned over. The retraction is
written up in §5.4 rather than quietly replaced, because the figures circulated
before the full run finished. Neither affects the recommendation, which rests on
§6.2 and the blend check. The §6.2 comparison
is reported separately in
[`phase_c_frozen_comparison.md`](phase_c_frozen_comparison.md); its one-line
result is that frozen joint-chain MINT is statistically indistinguishable from a
frozen ESM-2 control, and every frozen arm sits ~0.18–0.26 below the sequence
ensemble on the same partitions.

§6.3 and §6.4 have **not** been started, and the decision to spend on either
belongs to the user.

Three findings drive that, in descending order of consequence:

1. **The plan's caution about SPEARMINT's released stability weights is now a
   measured fact, not an inference.** Its stability checkpoints were fine-tuned
   on the NetMHCstabpan training corpus, which *is* our corpus. 100% of our
   validation clusters and 100% of our test clusters are inside their training
   data. They are not a competitor at any strength.
2. **The affinity checkpoint is not a clean upstream either.** It carries
   supervised affinity labels covering **at least 80.4% of our validation
   clusters and 77.9% of our test clusters**. Clean of stability labels, yes;
   leakage-controlled, no. Only the base MINT checkpoint — STRING
   protein–protein interactions, no peptide–MHC supervision of any kind —
   survives §6.1's requirement.
3. **The gate's own measurement passed with wide margins.** Frozen extraction is
   61 ms per pair at 3.1 GiB peak on a $0.80/h card, so §6.2 costs about a
   dollar. Limited adaptation at batch 1 in fp32 measured 97 ms per pair-step,
   which forecast **$238 with margin** — more than the plausible remaining
   balance in either workspace.
4. **A $0.06 follow-up probe cut that by 4.2× and turned a no-go into a
   decision.** bf16 at batch 8 adapts at 24.9 ms/pair, repricing §6.3 to **$61
   with margin** — inside one workspace's credit. The $238 had rested on the
   least favourable configuration available, which is the failure mode this
   project has an explicit rule about. The 4.2× holds only on this
   padding-free input set (§5.2).
5. **The epoch count is still unmeasured, so §6.3's cost is bounded below, not
   pinned.** Probe 2's curve had not turned over at its 12-epoch cap (§5.4), so
   epochs-to-stop exceeds 12 and §6.3 is **≥$39.72 with margin, plausibly $61 or
   more**. An earlier version of this report said early stopping bit at epoch 9
   and repriced §6.3 at $30.59; that read a single-epoch blip from a mid-run
   checkpoint as a turnover and is retracted. What the probes did establish is the
   throughput correction: batching and bf16 are worth 4.2×, which is where the
   fall from $238 came from.
6. **But §6.2 has since run, and it is a negative result** —
   [`phase_c_frozen_comparison.md`](phase_c_frozen_comparison.md). Frozen
   joint-chain MINT is statistically indistinguishable from frozen ESM-2, and
   every frozen arm sits ~0.18–0.26 below the sequence ensemble. §6.3 is now
   cheap *and* poorly motivated, which is a different decision from cheap and
   promising.

What this does *not* establish: nothing here measures whether interaction-aware
representations help on our metric. The gate measured access, leakage, memory,
time and cost. The plan's stop-scaling criterion — "stop if the matched outer
assessment excludes a worthwhile increment" — still has to be met by §6.2.

## 2. Candidate ladder and leakage audit

The plan names MINT and the SPEARMINT preprint. MINT (Ullanat & Zitnik, *Nature
Communications* 2025) is ESM-2 650M with a per-layer cross-chain attention
module, pretrained without supervision on 96M protein–protein interactions from
STRING. SPEARMINT (Karthikeyan, Vincent & Rubinsteyn, bioRxiv
[10.64898/2026.06.28.735023](https://www.biorxiv.org/content/10.64898/2026.06.28.735023v1))
fine-tunes MINT in three stages and releases a checkpoint after each.

Overlap is measured at **our** clustering rule — single-linkage components at
Hamming ≤ 3, `pepstab.splits.HAMMING_THRESHOLD` — not at exact peptide match. A
cluster counts as exposed if any donor peptide sits within the threshold of any
peptide in it, because such a donor peptide would have joined that cluster had it
been in our dataset. Produced by
[`scripts/phase_c_leakage_audit.py`](../../scripts/phase_c_leakage_audit.py) →
[`phase_c_cluster_overlap.csv`](phase_c_cluster_overlap.csv).

| Checkpoint | Supervised exposure | Val clusters exposed | Test clusters exposed | Usable? |
|---|---|---:|---:|---|
| `varunullanat2012/mint` | STRING PPIs, unsupervised. No pMHC labels. | — | — | **Yes — the only clean upstream** |
| `dkarthikeyan1/mint-stage1-affinity` | + NetMHCpan 4.1 binding affinity | ≥ 80.4% | ≥ 77.9% | Only as a declared affinity-exposed arm |
| `dkarthikeyan1/mint-stage2-stability` | + NetMHCstabpan half-lives | **100%** | **100%** | **No** |
| `dkarthikeyan1/spearmint` | + assay-conditioned recalibration | **100%** | **100%** | **No** |

### 2.1 Why the stability rungs are 100%, exactly

SPEARMINT's stage 2 trains on "the NetMHCstab-pan training corpus", reported as
27,034 measurements across 72 HLA-I alleles after processing. That is the
Rasmussen et al. 2016 corpus — our `data/rasmussen_et_al_dataset.csv`, 28,166
rows across 75 alleles, the difference being their 9-mer and aggregation
filters. Their own similarity-aware splits protect *their* held-out set; they
place no constraint on ours, and their training partition is drawn from the same
rows. Every one of our 510 validation clusters and 1,023 test clusters is
therefore inside their training data, and the audit's self-overlap arm confirms
it rather than discovering it.

This is the same objection already recorded for NetMHCstabpan itself in
[`docs/`](../../docs/README.md) and the project memory: reproduce the method, do
not quote the number. A SPEARMINT score on our test split would measure
memorisation.

### 2.2 Why the affinity rung is a bound, and what kind

We do not hold NetMHCpan 4.1's BA file, so the donor set is proxied by
`data/data_augmentation_iedb/affinity_reference_75alleles.csv` (110,591 rows,
25,690 unique 9-mers; MHCflurry curated 2023-10-23 plus Kim et al. 2014 BD2013,
restricted to our 75 alleles).

That whole table is **not** a subset of NetMHCpan 4.1's BA data: its primary
source is a 2023 IEDB snapshot, three years later than NetMHCpan 4.1 (Reynisson
et al. 2020), so it contains measurements that corpus cannot hold. Against it the
overlap is 89.8% of validation clusters and 88.3% of test clusters, but that
figure is neither a bound nor an estimate of the real overlap — only evidence
that overlap is extensive.

The defensible bound comes from restricting the donor set to the 97,965 rows
carrying a BD2013 measurement (18,442 unique 9-mers). BD2013 entered the
NetMHCpan lineage at NetMHCpan-4.0 (Jurtz et al. 2017), and 4.1's BA partition
is built on that lineage plus later IEDB, so those rows are in the donor corpus.
That is a documented-provenance claim about someone else's training file, not
something measured here, and it is the one link a reader should check rather
than take on trust. On that restricted set:

| Donor set | Val clusters | Test clusters | Val rows affected | Test rows affected |
|---|---:|---:|---:|---:|
| Full curated table (not a bound) | 89.8% | 88.3% | 2,556 / 2,817 | 5,107 / 5,633 |
| BD2013-backed only (**lower bound**) | **80.4%** | **77.9%** | 2,287 / 2,817 | 4,468 / 5,633 |

Affinity is not the target, but it is not independent of it either: the affinity
reference's own manifest records Spearman **−0.4911** between affinity and
stability on the 7,281 pairs present in both. A checkpoint that saw supervised
affinity labels for peptides in four-fifths of our held-out clusters has seen
something informative about our held-out labels. The project already uses this
same affinity table for augmentation, but under a rule that confines it to the
fit set; a checkpoint cannot be confined that way.

### 2.3 What the base checkpoint's exposure actually is

Not audited numerically, and the reason matters. STRING entries are whole
proteins, not 8–11mers, so there is no peptide-level donor set to intersect. The
exposure that does exist is source-protein level — HLA heavy chains and the
proteins our peptides were derived from appear in any large protein corpus — and
it is shared by every protein language model in this project, including the
deployed ESM-2 arm and the sequence baseline's BLOSUM encodings. It is not a new
exposure introduced by Phase C.

### 2.4 Consequence for §6.4 — read this before approving Phase C

§6.4 asks for "affinity-then-half-life adaptation under leakage-controlled
upstream data". We cannot borrow SPEARMINT's affinity stage to supply it: that is
the rung disqualified in 2.2. Doing §6.4 properly now means **our own affinity
pretraining restricted to our training clusters** — building a leakage-filtered
affinity corpus, then a full pretraining run on it, before the half-life stage
starts. That is a materially larger and more expensive job than the plan assumed
when it listed §6.4 alongside four cheap comparisons.

It is deliberately not scoped or costed here. The point is that §6.4 must not be
waved through as part of an approved Phase C. §6.2 and §6.3 do not depend on it.

### 2.5 One thing that got better than the plan expected

Base MINT is ESM-2 650M plus cross-chain attention, and the project already has
ESM-2 650M embeddings cached (`features/esm/esm2_t33_650M_UR50D`). So §6.2's
frozen separate-chain control can hold backbone family and parameter scale fixed
while the representation changes — a cleaner contrast than §6.2 dared assume,
since its stated worry was that pretraining and architecture would move together.

They still move together. STRING PPI pretraining is a second difference from
plain ESM-2, on top of the architecture. The correct claim is that **backbone and
scale are controlled, pretraining corpus and architecture are not**, so a §6.2
difference must not be attributed to cross-attention alone. §6.3's masking
ablation remains the closer test, and the plan is already right that
inference-time masking is not a complete causal account.

## 3. Access record — verified, not assumed

Every row below was returned by a live API call. The HF manifests were re-fetched
from inside the probe container, so the access table is reproducible from
[`phase_c_probe.json`](phase_c_probe.json).

| Artefact | Identifier | Revision / commit | Size | Licence | Access |
|---|---|---|---:|---|---|
| MINT base weights | `varunullanat2012/mint` · `mint.ckpt` | `75b1602c34e9df1a89089a1245f000a51ee3c746` | 3.254 GB | MIT | public, ungated; pulled in 17.0 s |
| MINT code | `github.com/VarunUllanat/mint` | `06694b7606e2d00b76ec58daf5c7aecdaf7cd283` | — | MIT | public |
| SPEARMINT stage 1 | `dkarthikeyan1/mint-stage1-affinity` | `8bf8e51906cf63336706d6cfc4f85e95a8109c86` | 3.256 GB | MIT | public; **disqualified**, not downloaded |
| SPEARMINT stage 2 | `dkarthikeyan1/mint-stage2-stability` | `d3bad29b1e2e9f4313b2187dfa215a748565a58e` | 3.256 GB | MIT | public; **disqualified**, not downloaded |
| SPEARMINT final | `dkarthikeyan1/spearmint` | `93e8e3e640e49e5a373f0dc432b122df19e4b969` | 3.259 GB | MIT | public; **disqualified**, not downloaded |
| SPEARMINT code | `github.com/pirl-unc/spearmint` | pushed 2026-07-01 | — | MIT | public |

Requested revisions matched the returned commit SHAs in all four HF repos, so the
pins are real pins.

### 3.1 Four access facts the probe caught that a plan could not assume

* **The architecture config does not ship in the installed package.** The HF repo
  holds only `mint.ckpt` and `bernett_mlp.pth`; `data/esm2_t33_650M_UR50D.json`
  lives in the git tree, and `setup.py` has no `install_requires` and relies on
  `include_package_data`, which did not carry it. `config_in_package` came back
  empty. `mint_gate.py` falls back to the pinned raw URL.
* **The published environment does not have to be reproduced.** `environment.yml`
  pins python 3.7.12 and pytorch 1.12.1, but the extraction path
  (`mint/__init__` → `model.esm2` → `modules` → attention) imports only torch;
  `apex` and `torchvision` are guarded or lazy, and `lightning`, `wandb` and
  `Bio` are reached only from modules `__init__` never imports. It ran on
  torch 2.5.1 / python 3.11.
* **`torch.load(weights_only=True)` fails on this checkpoint and
  `weights_only=False` succeeds** (1.3 s). It is a Lightning checkpoint carrying
  `hyper_parameters` and `lr_schedulers` alongside `state_dict`. Resolved on a
  CPU worker, not a GPU one, and the gate writes a plain-tensor state dict beside
  it so no later run repeats the risk. 768 tensors, 813.4M parameters, all
  float32, loading into the architecture with **0 missing and 0 unexpected keys**.
* **The cross-chain module is genuinely in the weights.** 198 `multimer_attn.*`
  keys. Worth checking: without them this would be a plain ESM-2 with extra
  steps, and the entire §6.2 contrast would be empty.

### 3.2 Correctness requirement: the published wrapper must not be used for §6.2

This is not a convenience note. **`MINTWrapper` cannot implement §6.2, and using
it would produce plausible numbers that answer a different question.**

`MINTWrapper.forward` returns mean-pooled embeddings only — one vector per chain
with `sep_chains=True`, one vector for the whole complex otherwise. §6.2's design
rests on *position-preserving* peptide pooling, and mean-pooling the peptide
destroys exactly the per-position structure the comparison exists to test. It
would not error. It would return a 1,280- or 2,560-dimensional feature, a head
would fit it, and the resulting Spearman would be reported as a frozen
joint-chain result while silently having discarded peptide position. Anyone
re-introducing the wrapper later would reopen this.

The fix is to bypass it and call `ESM2.forward(tokens, chain_ids,
repr_layers=[33])` directly for the per-residue representation. The probe
confirmed `repr_layers` is in that signature before any GPU time was spent on the
assumption, and the smoke's 12,800-dimensional feature (9 × 1,280 + partner mean)
is the shape §6.2 actually calls for.

Positions are then selected from a special-token mask, not from length
arithmetic — see §3.3.

### 3.3 Position indexing

Positions are selected from a special-token mask, not from length arithmetic.
`CollateFn` wraps *each* chain as `<cls>seq<eos>`, so a chain boundary costs two
tokens and the naive index is off by one for every HLA residue. The smoke asserts
the recovered peptide length matches the input.

### 3.4 A finding about the plan, not about MINT

**§3.3's "at most 12 prespecified configurations per new learner/objective
family, two paired seeds and three inner cluster folds" is unaffordable for any
encoder-adaptation arm.** At the measured adaptation rate that screen is
12 × 2 × 3 = 72 fits, which is roughly $857 before margin — more than the
project's entire two-workspace allocation.

The ceiling was written with CPU learners in mind, where a fit is free and 72 of
them cost minutes. Applied to a GPU arm it silently becomes a budget ceiling
rather than a search-discipline one. §5's §6.3 forecast therefore proposes a
reduced screen (3 unfreeze depths × 2 paired seeds, one finalist, no matched
bagging budget) and says so explicitly rather than quietly shrinking the search.
The plan's *intent* — prespecify the space, log extensions rather than expanding a
disappointing arm — is unaffected and still binds.

## 4. Smoke benchmark — measured

One L4, 5 peptide–HLA pairs drawn from the **train** split of `data/splits.csv`
(joined on `(allele, peptide)`, never `pair_id`), one pair per allele across
5 alleles: clusters 3, 24, 84, 811, 84. Raw output in
[`phase_c_smoke.json`](phase_c_smoke.json).

Contracts timed. `pep_domain` is chain 1 = the 9-mer, chain 2 = the 182-residue
α1/α2 groove domain from `hla_seq` — matching the project's existing
`pep_domain` input set so a Phase C result is comparable to an arm already
measured. `pep_ectodomain_b2m` is the 3-chain 275 + 99 variant, timed only to put
a sensitivity range on the budget.

| Regime | Measurement |
|---|---|
| Card | NVIDIA L4, 22.03 GiB available |
| Weight load | 26.7 s from the converted state dict; 813.4M params, 3.03 GiB resident |
| Frozen extraction, `pep_domain` (195 tokens) | **61.4 ms/pair** median, 62.7 ms max; peak **3.14 GiB** |
| Frozen extraction, `pep_ectodomain_b2m` (389 tokens) | 119.6 ms/pair median; peak 3.22 GiB |
| Feature dimension | 12,800 (9 peptide positions × 1,280 + mean-pooled partner) |
| Head training, 0.82M params | **8.29 ms/step**; peak 3.13 GiB; loss 0.0429 → 1.6e-05 |
| Limited adaptation, top 4 of 33 layers | **486 ms/step** at batch 5 = 97.2 ms/pair-step; 98.4M trainable; peak **4.86 GiB**, no OOM |
| Wall clock, whole smoke | 51.1 s; 32.9 s of measured GPU work |

Two readings that matter for the budget:

* **Memory is a non-issue and L4 is the right card.** The heaviest regime peaked
  at 4.86 GiB of 22.03 — 4.5× headroom. Nothing here justifies a larger card, and
  the cheaper T4 would also fit but has no bf16, which would force a precision
  choice onto a cost decision.
* **The adaptation figure is a batch-1 fp32 upper bound.** The smoke runs each
  pair as its own forward/backward inside a step, so 97.2 ms/pair-step includes
  no batching benefit, and fp32 leaves bf16 on the table. Real batched bf16
  training is very likely several times faster. That is not applied below,
  because the gate's job is to report what was measured.

The loss falling to 1.6e-05 on 5 pairs with zero targets says gradients flow and
the optimiser is wired up. It says nothing about fit, and is reported for that
reason only.

## 5. Full-Phase-C budget forecast

Computed by
[`scripts/phase_c_budget_forecast.py`](../../scripts/phase_c_budget_forecast.py)
→ [`phase_c_budget_forecast.csv`](phase_c_budget_forecast.csv), from the measured
numbers above and HACKATHON_PLAN.md's rule
`forecast_cost = 1.25 × (N × steady + startup)`. L4 container rate **$1.1172/h**
= $0.80 card + 4 CPU + 16 GiB, from `reports/ectodomain_rates.json`. N = 22,533
rows: train + val, since the test split is not in development.

| § | Item | GPU hours | USD | With margin |
|---|---|---:|---:|---:|
| 6.2 | Frozen joint-chain extraction, all dev rows | 0.39 | 0.44 | 0.55 |
| 6.2 | Matched frozen separate-chain control | 0.39 | 0.44 | 0.55 |
| 6.2 | All downstream head fits | 0 | **0.00** | 0.00 |
| | **§6.2 total** | **0.78** | **0.88** | **1.10** |
| 6.3 | One adaptation fit, 20 epochs *(unit cost)* | 10.65 | 11.90 | 14.88 |
| 6.3 | Bounded screen, 6 fits | 63.92 | 71.42 | 89.27 |
| 6.3 | Finalist outer assessment, 10 fits | 106.54 | 119.03 | 148.78 |
| | **§6.3 total** | **170.5** | **190.45** | **238.05** |
| sens | 3-chain ectodomain extraction instead | 0.76 | 0.84 | 1.06 |

Head fits are $0 because they are not a GPU job. The head is 0.82M parameters on
cached features — the same shape of fit the project has run on laptop CPU for
free since stage 2, where a 30-network ensemble cost 1.7 CPU-minutes. Putting it
on a $1.12/h card would invent a cost.

The §6.3 figure rests on two stated assumptions, both of which should be attacked
before the money is spent. **Epochs:** 20, assuming early stopping bites.
SPEARMINT used 200 for its stability stage, which at this rate would be $119 per
*single* fit. **Search budget:** already cut from §3.3's 12-config default to
3 unfreeze depths × 2 paired seeds, with one finalist and no bagging budget
matched to the 30-network reference. §6.3 cannot afford §3.3's default search,
which is itself a finding about the plan.

### 5.1 Why $238 is a no-go rather than a large number

The ledger records **$220.87 realised across both workspaces** before this gate,
against the plan's ~$300 each (`a-cheparukhin` ≈ $105 including its share of the
pilots). So §6.3 with margin
plausibly exceeds what remains in either workspace on its own, and the plan is
explicit that credit in one cannot pay for work in the other, and that a budget
table does not authorise spending unavailable credit. Balances must be verified
at launch, not inferred from this ledger.

### 5.2 Probe 1, run: batching and bf16 are worth 4.2×

Both levers the gate left untested were authorised and measured. Probe 1
(`modal_app/mint_gate.py::throughput_probe`, $0.06 realised against a $0.16
forecast) ran a 16-cell grid: batch 1/8/16/32 × fp32/bf16 × frozen
extraction/top-4 adaptation, 10 timed steps per cell after an untimed warm-up.
Full grid in [`phase_c_throughput.json`](phase_c_throughput.json).

| Regime | fp32 batch 1 | **bf16 batch 8** | bf16 batch 16 | bf16 batch 32 |
|---|---:|---:|---:|---:|
| Frozen extraction, s/pair | 0.06251 | **0.01726** | 0.01934 | 0.02074 |
| Frozen extraction, peak GiB | 3.14 | **3.27** | 3.44 | 3.76 |
| Top-4 adaptation, s/pair | 0.10420 | **0.02486** | 0.02519 | 0.02625 |
| Top-4 adaptation, peak GiB | 4.58 | **4.82** | 5.58 | 7.11 |

**4.2× on adaptation, 3.6× on frozen extraction — on a padding-free input set.**
The $238 no-go rested on the least favourable configuration available, which is
exactly the failure this project has a rule about.

The precondition is not separable from the number. Batching converts almost
fully into throughput **only because every `pep_domain` construct is exactly 195
tokens** — all peptides are 9-mers and `hla_seq` is uniformly 182 residues, so a
batch carries no pad tokens and no pair's cost is diluted by another's padding.
That is a property of *this dataset*, not of batching. Carried to a ragged input
set — variable peptide lengths, or the full ectodomain against a variable
partner — the speed-up will be smaller, and the discovery will come after a
budget has been committed.

### 5.3 Repriced §6.3 — ASSUMED, superseded by §5.4's measurement

| § | Item | GPU hours | USD | With margin |
|---|---|---:|---:|---:|
| 6.3 | One adaptation fit, 20 epochs *(unit cost)* | 2.72 | 3.04 | 3.80 |
| 6.3 | Bounded screen, 6 fits | 16.3 | 18.2 | 22.8 |
| 6.3 | Finalist outer assessment, 10 fits | 27.2 | 30.4 | 38.0 |
| | **§6.3 total, 16 fits** | **43.5** | **48.8** | **61.0** |

That is inside one workspace's plausible remaining credit, where $238 was not.
It inherits §5.2's precondition: the rate holds because every construct is 195
tokens, so a ragged input set would not reprice this way.

This figure was labelled ASSUMED and not approvable, because it is a measured
per-pair-step rate multiplied by an *assumed* 20 epochs. Probe 2 attempted to measure the
epoch count and could not pin it (§5.4): the curve was still improving at its
12-epoch cap, so this table's 20 epochs is a plausible figure rather than a
refuted one, and §6.3's floor is $39.72 with margin. The project's rule about
assumed per-unit quantities still applies — the quantity remains assumed.

### 5.4 Probe 2, run: the curve does NOT turn over — the multiplier is unresolved

**MEASURED, and the answer is that 12 epochs is not enough to find the stopping
point.** An earlier version of this section reported "early stopping bites at
epoch 9" and repriced §6.3 at $30.59. That was wrong, and the error is recorded
here rather than quietly replaced because it changed a cost figure that was
already circulated.

| Epoch | Train loss | Inner-val MSE | Median per-allele SCC |
|---:|---:|---:|---:|
| 0 | — | 2.20548 | 0.0045 |
| 5 | 0.43201 | 0.69723 | 0.4393 |
| 8 | 0.28045 | 0.67066 | 0.4798 |
| 9 | 0.24115 | 0.66737 | 0.4676 |
| 10 | 0.20715 | 0.68753 | 0.4629 |
| 11 | 0.17851 | 0.67050 | 0.4917 |
| **12 (cap)** | 0.15324 | **0.65843** | 0.4890 |

**The minimum is at the cap.** Epoch 10's uptick to 0.68753 was a single-epoch
blip; 11 recovered and 12 set a new minimum. Reading that one bad epoch as a
turnover — from a mid-run checkpoint, before 11 and 12 existed — was the mistake.
A patience of 1 would have stopped there and been wrong too; this curve needs a
patience of at least 3.

**Consequence for §6.3's forecast: it is unresolved and bounded below.**

| Epochs | 16 fits | With margin | |
|---|---:|---:|---|
| 10 | $24.47 | $30.59 | **retracted** — rested on the false turnover |
| ≥13 | $31.77 | $39.72 | floor, since epoch 12 still improves |
| 20 | $48.81 | $61.01 | the gate's original assumption |
| 30 | $73.15 | $91.43 | if it keeps improving |

So the gate's original 20-epoch assumption was **not** shown to be conservative.
It may be about right or still low. The honest statement is that **epochs-to-stop
exceeds 12 and is unmeasured**, and §6.3's cost is therefore ≥$39.72 with margin
and plausibly $61 or more. Measuring it properly needs a longer run with patience
≥3, at roughly $1 per 12 epochs — not run, because §6.2 and the blend check have
since made §6.3 unmotivated regardless of its price.

Measured throughput held up exactly: 399.6 s/epoch median against the 389 s the
throughput probe predicted, 2.7% over, peak 4.83 GiB.

**What this does to the adaptation-ceiling claim: retract it.** An earlier version
said limited adaptation "tops out around 0.47–0.48". It does not top out there —
0.4890 at the cap and still climbing is a **floor at 12 epochs**, not a ceiling.
The gap to the sequence reference is still large and the trajectory is upward and
unresolved. Everything else about that reading stands: one configuration, a
timing fold, a weaker head than §6.2's, no ensembling, no tuning. It is not a
§6.3 result and now it is not a ceiling estimate either.

**What this does not change: the recommendation to stop Phase C.** That rests on
§6.2's frozen comparison and the blend check, both independent of the epoch count.
The epoch number bears only on what §6.3 would *cost*, and the case against §6.3
was never that it was expensive.

### 5.4.1 Original probe 2 specification

One adaptation run at bf16 batch 8 on a 15,772 / 3,944 whole-cluster timing fold
of the training rows, capped at 12 epochs, recording inner-validation MSE and
median per-allele Spearman per epoch.

Forecast **$2.15 with margin, ~92 minutes** — priced from probe 1's measured
rates, logged before launching. This one is **not cents**: a 12-epoch trajectory
is 43% of a real fit's work, which is the price of converting the multiplier from
an assumption into a measurement. It is still under the gate's $5 ceiling and
worth 1/28th of the decision it informs.

What it can and cannot conclude is stated in `epoch_curve`'s docstring: the fold
is built for timing and is **not** the §3 nested protocol fold, so no metric from
it is reportable as a Phase C outcome. It answers one question — is the 20-epoch
multiplier optimistic, about right, or conservative. If the curve is still
improving at the cap, that is a real outcome and the honest answer is that the
multiplier remains unresolved.

### 5.5 Operating guidance for any GPU arm in this project

The 4.2× is a property of this hardware, this model shape **and this
padding-free input set** — not of Phase C, and not transferable to a ragged one
(§5.2). The next person forecasting a GPU job here should find these numbers
rather than re-measure at batch 1 and conclude the work is unaffordable — which
is what nearly happened.

* **Batch 8 is the operating point, and the data is unusually clean about it.**
  It is simultaneously the fastest *and* the lowest-memory batched cell for both
  regimes. Past 8 you pay memory for negative speed: adaptation goes 24.86 ms at
  4.82 GiB → 25.19 at 5.58 → 26.25 at 7.11.
* **bf16 without batching makes things worse.** At batch 1, bf16 is *slower* than
  fp32 — 119.17 vs 104.20 ms/pair adapting, 64.89 vs 62.51 extracting. The
  conversion overhead only amortises once there is enough work per kernel. "Use
  bf16" is not advice on its own.
* **Hold dtype constant within any comparison.** bf16 changes numerics, so a
  bf16-extracted feature matrix is not the fp32 one. If one §6.2 arm were
  extracted in bf16 and another in fp32, the comparison would acquire a second
  difference on top of the one it tests — the same failure mode as §6.2's
  existing pretraining-and-architecture confound. Record dtype in the manifest.
* **Therefore §6.2 stays in fp32.** It costs about $1.10 either way, and there is
  no budget reason to take a numerics risk to save a fraction of a dollar on the
  one arm already funded. Spend the bf16 speed-up on §6.3, which it makes
  possible. Using bf16 extraction anywhere requires first measuring downstream
  head performance on bf16 versus fp32 embeddings on one inner fold and showing
  it does not move.
* **Memory is not the binding constraint at any setting measured.** Worst cell was
  8.73 GiB (fp32 adaptation, batch 32) of 22.03 available.
* **A long run must checkpoint its partial result, not only its final one.** See
  §5.6; this cost $0.78 to learn. It belongs beside the `esmfold_common.py` note
  about `modal run` buffering a traceback out of sight: on a flaky network, the
  difference between a loss and an answer is whether partial work was committed.

### 5.6 Incident: both runs cancelled externally, and what it changed

On 4 October at 19:41 BST both detached runs received
`Received a cancellation signal` — the epoch probe 42 minutes into ~96, the
extraction 11 minutes into ~24. Recorded because the cause is instructive and
because $0.78 of it is logged as `measured_wasted`.

**What the evidence establishes.** `modal app list --json`:

```
ap-FqrQeSyKdgZhkr7ZzTWfi5  phasea-experiments   stopped 19:37:33
ap-POaj7UTcnoE9hDQVgXe3dY  mint-gate            stopped 19:41:29
ap-adoYlVMYDWofjqODaA62Fh  phasea-experiments   stopped 19:41:31
ap-nLW16mg1WjESviyVY21y32  phaseb-calibration   stopped 19:41:34
ap-MWBlfyrn2tzjpChlCKWHyc  mint-features        stopped 19:41:36
```

Four apps across three workstreams inside a 7-second window. A `modal app stop`
targets one app, so the cluster is not a per-app command; the one stop command
issued deliberately in that period hit `…FqrQ` at 19:37:33, nearly four minutes
earlier. `--detach` demonstrably applied — `modal app list` showed
`ephemeral (detached)` while the run was live — so a client disconnect should not
have killed them. It coincided with a machine-level network failure that every
session on the host observed independently (`[Errno 8] nodename nor servname
provided` here; `SSL connection is closed` and a subagent `ENOTFOUND` elsewhere).

**What it does not establish.** Why a *detached* app received a cancellation
signal from a client-side network failure. That is recorded as unresolved rather
than given a plausible mechanism it has not earned.

**What it changed, and this is the part worth keeping.** The two runs lost very
different amounts of work for a structural reason:

| | wrote results | lost |
|---|---|---|
| Extraction | committed to the Volume as it went | 55% — the un-started joint arm |
| Epoch probe | returned its history as a return value | **100% of 42 minutes** |

The epoch probe's per-epoch numbers existed in memory the whole time and were
thrown away. Both are now fixed: `epoch_curve` writes the curve to the Volume
after every epoch and prints it, so a run cancelled at epoch 7 still answers at
epoch 7; the joint extraction is chunked at 4,000 pairs, commits per chunk, and
skips chunks already present. On the same outage the probe would have returned
five epochs of curve instead of nothing, and the re-run cost $0.40 rather than
$0.60 because the separate-chain arms were detected and skipped.

## 6. Spend on this gate

Forecast **before** launching, at the brief's requirement: $0.13 compute plus
$0.59 for a month of Volume storage, $0.90 combined with margin, against a $5
ceiling. Realised **≤ $0.05** of compute: 51.1 s of L4 wall clock, a 17.0 s
weight pull, two CPU containers, and two image builds not separately metered.

Ledger rows appended to `reports/compute_ledger.csv`, all under stage `C-gate`:

| Status | USD | What |
|---|---:|---|
| `FORECAST_superseded_by_split_rows_below` | 0.90 | the original conflated forecast |
| `FORECAST_priced_before_launch` | 0.13 | compute only |
| `derived_forecast_per_month` | 0.59 | 6.6 GiB on the `a-cheparukhin` Volume, recurring |
| `measured_upper_bound` | 0.05 | realised, gate passed |

The storage row is recurring and conditional. **On a no-go for the whole phase,
delete the `pepstab-mint-weights` Volume and ledger the storage as not
incurred** rather than letting a month accrue against a dropped direction. The
§6.2 go means it should be kept for now.

## 7. Against the plan's gate, point by point

§6's gate: "perform an extraction/training smoke benchmark for memory, time and
checkpoint access before specifying a full GPU budget."

| Requirement | Status |
|---|---|
| Checkpoint access | **Passed.** Four repos verified live, pinned by commit SHA, MIT, ungated; base weights pulled and loaded with 0 missing keys. |
| Memory | **Passed.** 4.86 GiB peak of 22.03 in the heaviest regime. |
| Time | **Passed.** 61.4 ms/pair frozen, 97.2 ms/pair-step adapting at batch 1 fp32; 17.3 and 24.9 ms/pair at bf16 batch 8 (§5.2). |
| Per-pair cost | **Passed.** $0.0000191/pair frozen extraction at the L4 rate. |
| §6.1 clean upstream checkpoint | **Passed, with the ladder cut to one rung.** Base MINT only. |
| §6.1 affinity overlap at cluster level | **Audited.** ≥ 80.4% val, ≥ 77.9% test for the affinity rung. |
| Full GPU budget specified | **Yes**, §5: §6.2 at $1.10 (fp32), §6.3 at $61.01 after the throughput probe but ASSUMED pending epochs-to-stop, §6.4 unbudgeted. |
| `CLAUDE.md`: no batch GPU job without a passing pilot | **Satisfied** for a §6.2 extraction run. A §6.3 run needs its own pilot. |

## 8. Recommended next actions, in order

Revised after §6.2 and both probes completed.

**Both halves of §6's stop-scaling criterion are now met**, so the plan's own
condition to stop is satisfied — see
[`phase_c_frozen_comparison.md`](phase_c_frozen_comparison.md) §2 and §5.

1. **Stop scaling Phase C.** The matched outer assessment excludes a worthwhile
   increment, and no frozen arm adds useful blend signal: given a free weight
   choice on held-out folds, the blend selected **zero** on 11 of 15 arm-folds and
   never exceeded 0.1.
2. **Do not run §6.3.** Its cost is unresolved — ≥$39.72 with margin, plausibly
   the $61.01 of §5.3's 20-epoch figure or more, since the adaptation curve was
   still improving at its 12-epoch cap (§5.4). The case against it is not price:
   its frozen form is indistinguishable from a control already cached for free,
   and §6.2's decomposition points at the backbone rather than the amount of
   adaptation. Its adaptation ceiling is **unmeasured** — 0.4890 at the cap while
   still climbing is a floor at 12 epochs. ($30.59 and a ~0.47 ceiling appeared in
   an earlier version of this report and are withdrawn; see §5.4.)
3. **Do not run §6.4.** Separate unbudgeted sub-project, per §2.4.
4. **Carry the two methodological findings out of Phase C**, because neither is
   about MINT: the median-placement artefact on the primary metric
   (`phase_c_frozen_comparison.md` §2.1) and the checkpointing convention (§5.6).
5. **Delete the `pepstab-mint-weights` and `pepstab-mint-features` Volumes** and
   ledger the storage as not incurred rather than letting a month accrue against a
   stopped direction. Keep `phase_c_oof_*.csv` — they cost real money and are what
   any future blend or residual experiment would need.

### 8.1 Execution contract §6.2 must inherit, not declare

§6.2's numbers are only worth something if they are directly comparable to A0's.
That means inheriting the shared protocol rather than defining a parallel one.
All of the following was verified present on disk before being cited here.

* **Outer folds from
  [`phase_a_outer_folds.csv`](phase_a_outer_folds.csv)** — load and compare,
  never recompute, for the same reason `data/splits.csv` is never recomputed.
* **Eligibility panel:** ≥20 rows and ≥2 distinct labels per allele, with ≥50
  as a declared sensitivity check. Coverage reported per fold.
* **Scoring:** `pepstab.evaluation` verbatim — `paired_cluster_bootstrap` at
  `N_BOOTSTRAP = 2000`, `BOOTSTRAP_SEED = 20261003`, reported through
  `describe_delta`.
* **Index:** [`README.md`](README.md) for the follow-up namespace.
* **dtype:** fp32, recorded in the manifest (§5.5).

Two defects found elsewhere in this study apply to §6.2 and are cheaper to
inherit than to rediscover. Both come from
[`phase_a_prespecification.md`](phase_a_prespecification.md) §2 and the Phase D
work; neither is a Phase C finding and both are recorded here only so §6.2 does
not reintroduce them.

* **Score the primary metric on uncalibrated predictions.** `IsotonicRegression`
  is only *weakly* monotone — a step function maps strict inequalities to
  equalities, and ties take average ranks under the frozen contract. Measured on
  this project's own predictions, 2,776 distinct predictions collapsed to 57,
  with per-allele Spearman shifting between −0.0431 and +0.1109 and exceeding
  0.05 in absolute value on 3 of 68 alleles. **Both signs occur within a single
  fit**, so this is a confound, not a bounded conservative bias. If §6.2's head
  reports a magnitude metric, keep every calibration map off the ranking path and
  report the map's measured effect rather than assuming it is zero.
* **Assert rank preservation at fold scope, not pooled scope.** Anything fitted
  per fold and then pooled is several transforms stitched together, not one
  monotone map, so rows from different folds can reorder within an allele. Phase
  D measured ≤0.0012 that way. Assert where it holds exactly and report the
  pooled shift as a measured quantity. Do not loosen a tolerance to make an
  assertion pass.

## 9. Files

| Path | Contents |
|---|---|
| [`phase_c_cluster_overlap.csv`](phase_c_cluster_overlap.csv) | Per-split cluster overlap, three donor corpora |
| [`phase_c_probe.json`](phase_c_probe.json) | API surface, config resolution, four HF manifests |
| [`phase_c_setup.json`](phase_c_setup.json) | Download timing, load-strategy ladder, state-dict census |
| [`phase_c_smoke.json`](phase_c_smoke.json) | The gate measurement, with the 5-pair panel |
| [`phase_c_budget_forecast.csv`](phase_c_budget_forecast.csv) | §5's arithmetic, at the batch-1 fp32 rates |
| [`phase_c_throughput.json`](phase_c_throughput.json) | §5.2's 16-cell batch x dtype grid |
| [`phase_c_epoch_curve.json`](phase_c_epoch_curve.json) | §5.4's inner-validation curve |
| [`../../scripts/phase_c_leakage_audit.py`](../../scripts/phase_c_leakage_audit.py) | The audit, CPU, free |
| [`../../scripts/phase_c_budget_forecast.py`](../../scripts/phase_c_budget_forecast.py) | The forecast, from measured numbers |
| [`../../modal_app/mint_common.py`](../../modal_app/mint_common.py) | Pins, worker shape, construct contract, rates |
| [`../../modal_app/mint_gate.py`](../../modal_app/mint_gate.py) | `::probe`, `::setup`, `::forecast`, `::smoke`, `::throughput_probe`, `::epoch_probe` |

External sources: [MINT](https://www.nature.com/articles/s41467-025-67971-3)
(Ullanat & Zitnik, *Nat Commun* 2025);
[SPEARMINT preprint](https://www.biorxiv.org/content/10.64898/2026.06.28.735023v1)
(Karthikeyan, Vincent & Rubinsteyn, bioRxiv 2026 — not peer reviewed).
