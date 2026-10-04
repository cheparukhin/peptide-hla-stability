# Stage 3 status: what ran, what it showed, what is built but unrun

Written against `main` at `3576c02`. Every number below was produced by a run in
this project and read back from the file it was written to. **Arms that were
implemented but never executed are marked as such and carry no numbers** — that
distinction is the point of this document.

## 1. Status at a glance

| Work item | Implemented | Executed | Result |
|---|---|---|---|
| Baseline regeneration | — (existing) | **yes** | reproduces stage 2 |
| 3a. ESM-2 35M, separate embeddings | yes | **yes** | both arms worse, CIs below 0 |
| 3e-X3. Cross-attention coupling | yes | **no** | — |
| 3e-X0. BLOSUM 9×34 cross-features | yes | **no** | — |
| C1–C3. Positive/negative controls | yes | **no** | — |
| Head-architecture grid for ESM | yes | **no** | — |
| Label-side biology probes | yes | **yes** | noise ceiling, concordance, anchors |
| Re-scoring of a parallel implementation | yes | **yes** | all arms below the ensemble |

Four of the five stage-3 experiments exist as committed, smoke-tested scripts
with no results. They were queued behind each other on one machine and lost to
session restarts. Nothing about them should be reported as a finding.

## 2. What ran, and what it showed

### 2.1 The comparator reproduces

`scripts/baseline_ensemble.py` rebuilt from scratch — `preds/` is gitignored, so
the file every Δ is measured against is absent from a fresh clone.

| quantity | reported in stage 2 | regenerated |
|---|---:|---:|
| median per-allele ρ | 0.693 | **0.6904** |
| mean SCC | 0.645 | 0.649 |
| ensembling gain | +0.090 | +0.093 |

Independent confirmation that the stage-2 ensemble is what it says it is.
Wall-clock differed by 16× (73 s vs 19.6 min) on different hardware, so **cost
comparisons are only valid within one machine and one session**.

### 2.2 Stage 3a — frozen ESM-2 35M, peptide and HLA embedded separately

`facebook/esm2_t12_35M_UR50D`, 12 layers, hidden 480. Matched to stage 2 on
splits, `cv_folds`, seeds, architecture, budget and 30-network ensemble size
(5 folds × 2 layers × 3 seeds), stopping on the inner fold, validation only.

| arm | median per-allele ρ | Δ vs ensemble | 95% CI | verdict |
|---|---:|---:|---|---|
| sequence ensemble (comparator) | **0.6904** | — | — | — |
| ESM alone, 30 nets | 0.3866 | −0.3038 | [−0.359, −0.220] | worse |
| ESM stacked on baseline, 30 nets | 0.5387 | −0.1517 | [−0.205, −0.089] | worse |

Supporting observations, all from `reports/stage3_runs_esm2_35M.csv`:

- **The layer sweep is flat.** Member-level gap between layers 6 and 12 is
  +0.022 standalone and −0.005 stacked.
- **The ESM arms are seed-unstable.** Seed spread median 0.061, max 0.184,
  against 0.010–0.051 for the stage-2 sequence arms — roughly 3×.
- **They stop early.** Best epoch ~5 against the baseline's ~27: the networks
  reach their best dev loss almost immediately and then stop improving.

Cost: 70.3 s to embed all 5,708 unique sequences on CPU (5,633 peptides + 75
domains, against 28,166 rows), 7.8 s PCA, 483 s for 60 networks, 0.005 s
inference per 1,000 predictions. No GPU required at this model size.

**This headline is provisional — see §3.**

### 2.3 Residue indexing, verified

The published NetMHCpan pseudosequence positions sit at index *p* − 1 of the
supplied 182-residue domain, exact across **all 75 alleles × all 34 columns**.
Recovering the mapping from data alone is underdetermined — with 75 alleles,
several conserved columns match any pseudosequence column by chance — so the
published definition is the hypothesis and the data is the test.

### 2.4 Label-side probes (no model involved)

- **A noise ceiling, where the audit records none.** Four allele pairs differing
  at exactly one of the 34 contact residues agree at ρ **0.901–0.921** on
  245–369 shared peptides. Two distinct molecules cannot agree better than the
  assay resolves, so reproducibility is at least ~0.90 and the baseline at 0.69
  has ~0.2 of headroom. **The benchmark is not saturated.**
- **Groove concordance collapses with distance.** Median ρ between allele pairs
  on shared peptides: 0.565 at 1–4 differing contact residues, 0.145 at 5–9,
  ~0.0 beyond 10, with 79% of distant pairs inside |ρ| < 0.2.
- **Within-allele ranking is interaction, not peptide-intrinsic.** An additive
  `allele + peptide` identity oracle — fitted out-of-fold and allowed to read
  each peptide's measured behaviour on *other* alleles, information no model
  gets — reaches only **0.311**.
- **Anchor profile matches the contact geometry.** Permutation-adjusted variance
  explained per peptide position: P1 0.084, P9 0.081, P2 0.074, P3 0.050, with
  P4 0.010 and P5 0.012 at the noise floor. The adjustment matters: anchors
  carry half the residue diversity of solvent positions (P2 11 residues/2.23
  bits, P9 8.5/2.06 vs ~20/4.0) because the panel was affinity-preselected.
- **The C67S benchmark is weaker than it looks.** The three engineered
  constructs are the three most-censored alleles in the dataset — 92%, 89% and
  75% of their labels at the assay floor — so a ρ computed on them largely reads
  tied zeros.

### 2.5 A parallel implementation, re-scored on this contract

An independent stage-3 attempt (forked from `74373ef`, before stage 2 existed)
was re-scored through `pepstab.evaluation` against the 30-network ensemble:

| arm | median ρ | Δ vs ensemble | 95% CI |
|---|---:|---:|---|
| `seq_fulldomain` | 0.6423 | −0.0481 | [−0.110, −0.017] |
| `esm_l6_plus_onehot` | 0.6020 | −0.0884 | [−0.133, −0.038] |
| `esm_l6` | 0.5995 | −0.0909 | [−0.144, −0.047] |
| `seq_baseline` (theirs) | 0.5906 | −0.0998 | [−0.121, −0.030] |
| `esm_l12` | 0.5551 | −0.1353 | [−0.167, −0.058] |
| `claude_opus5` | 0.5346 | −0.1558 | [−0.233, −0.096] |

Three caveats that must travel with this table. Their arms are **3-seed means,
not 30-network ensembles**, and ensembling alone is worth +0.09 — so the fair
comparison is against the single network at 0.610, where their baseline (0.591)
and their best ESM arm (0.600) both land. Their protocol **early-stops on the
split it reports**, which inflates every arm. And their `seq_baseline` is their
own model, not the repo's.

The `claude_opus5` arm is worth keeping regardless: an LLM prompted per allele
with 80 in-context training examples, **0.5346 for $12.19 and 361 s**, CI
entirely below zero. It is the only arm in either tree with a dollar cost
attached, and for a track asking whether models earn their compute, a paid arm
losing to free one-hot features is a real Pareto point. It is few-shot
in-context regression, not zero-shot, and should be labelled that way.

## 3. The open question that makes 3a provisional

Stage 3a PCA-reduced the peptide block to 256 components. The independent
implementation used it at **full width (4,320 columns)** and scored **0.600**
where ours scored **0.387**.

A 0.21 gap is not explained by their early-stopping inflation or their different
head. The most likely cause is that **the reduction destroyed signal** — in
which case 3a measures my preprocessing, not ESM-2.

This is why the design document specifies full width for the peptide block and a
74-component PCA for the HLA block only, where it is lossless by construction
(75 unique domains → rank ≤ 75), and why the positive control runs first rather
than last.

**Until that control runs, do not quote −0.30 as the stage-3 result.**

## 4. Built, smoke-tested, unrun

| Script | What it decides | Status of the check |
|---|---|---|
| `stage3_controls.py` | Is the pipeline or the width to blame rather than ESM? Positive control (baseline features through the identical path), random block at matched width, shuffled-ESM at matched marginals | written; not run |
| `stage3_head_grid.py` | Was ESM under-tuned? Stage 2 gave every arm a 4-point grid; 3a gave ESM zero and inherited a config selected for 860 one-hot columns | written; not run |
| `stage3_blosum_cross.py` | Does *any* positional cross-encoding help? 9×34 BLOSUM62 matrix, no network | feature block verified: 306 columns, spot-checked against `pepstab.features.BLOSUM62` and its scale |
| `pepstab/attn.py` + `stage3_coupling.py` | Is concatenation the limitation rather than ESM? Cross-attention vs a uniform-average ablation | smoke-tested: both arms **94,913 parameters**, attention rows sum to 1.0, non-degenerate predictions |

The parameter-identity check on the coupling arms matters: it is what would let a
gap between them be attributed to coupling rather than capacity.

## 5. What is defensible today

1. The stage-2 ensemble is reproducible at 0.6904 and is the right comparator.
2. The benchmark is **not** saturated; assay reproducibility is ≥ ~0.90.
3. Within-allele ranking is dominated by peptide × groove interaction; an
   identity oracle with privileged information reaches only 0.311.
4. Frozen ESM-2 35M, embedded separately and **PCA-reduced**, does not reach or
   add to the baseline. Whether that survives at full width is **untested**.
5. Everything here is scoped to **new peptides on alleles seen in training**.
   All 75 alleles are in `train`; `dist_to_train` is peptide distance.

## 6. What is not known

- Whether frozen ESM-2 helps at full width — the first thing to run.
- Whether the stacking penalty is capacity displacement or information conflict.
- Whether coupling beats concatenation, and whether either beats a BLOSUM
  substitution matrix.
- Anything about unseen alleles.
- Whether a larger checkpoint would change the answer; per the design document,
  that is not the next experiment.
