# Stage 4c.5: structural feature extraction

4 October 2026. **Status: phase 1 complete.** The extractor is implemented and
validated against all 90 stage-4c pilot folds on local disk, free and local.
Phase 2 (extraction on Modal against the production Volumes) has **not** been
started and is waiting on a go-ahead.

This implements the "Feature contract for stage 5" section of
[`stage4c_ectodomain_pilot.md`](stage4c_ectodomain_pilot.md). Nothing here
invents a feature definition the contract already fixed.

| Artifact | What it is |
|---|---|
| `pepstab/structural_features.py` | parsing, mapping verification, feature definitions |
| `scripts/extract_structural_features.py` | `pilot` / `extract` / `concat` CLI |
| `modal_app/feature_extraction.py` | phase 2, CPU-only, **not yet run** |
| `tests/test_structural_features.py` | 26 tests, all passing |
| `reports/stage4c5_pilot_features.csv` | 90 folds x 129 columns |
| `reports/stage4c5_feature_variation.csv` | seed / arm / between-complex spread per feature |

## 1. The verified mapping, and the evidence that verified it

The contract anticipated an arm-B PAE of 383 x 383 with zero-based slices HLA
`0:275`, beta2m `275:374`, peptide `374:383`, and asked for that to be verified
rather than hard-coded. **It is correct on all 30 arm-B folds** — but the
extractor does not rely on it. Every boundary is re-derived per fold from the
mmCIF, and `verify_fold` raises `MappingError` on any disagreement rather than
slicing on faith. A silently-wrong slice yields plausible numbers for the wrong
molecule, which is the one failure mode that would survive into a results
section.

### 1.1 Token index i is the i-th CA atom in mmCIF atom-site order

This is the load-bearing claim, and it is checkable exactly rather than by
inference. Boltz writes per-token pLDDT into the mmCIF B-factor column, so the
two can be compared element by element:

| Check | Result |
|---|---|
| `max abs(CA b_factor − plddt × 100)` over all 90 folds | **0.005** |
| Pearson r, CA b_factor vs pLDDT (one arm-B fold) | 0.99999992 |

The tolerance is 0.01, enforced per fold in `verify_fold`, so any future fold
whose arrays do not line up with its coordinates fails loudly. A test that
swaps two pLDDT entries confirms the check actually fires.

### 1.2 Chain blocks are contiguous and in input order

Across all 90 folds, the consecutive runs of chain id in CA order equal the
declared input chains exactly:

| Arm | CIF chain runs | tokens | PAE | pLDDT |
|---|---|---:|---|---|
| A (60 folds: 15 Boltz + 15 ESM per arm, A and C) | `A:182, C:9` | 191 | 191 x 191 | 191 |
| B (30 folds) | `A:275, B:99, C:9` | 383 | **383 x 383** | 383 |
| C | `A:182, C:9` | 191 | 191 x 191 | 191 |

Derived slices for arm B: groove `0:182`, alpha3 `182:275`, beta2m `275:374`,
peptide `374:383`. The contract's HLA `0:275` is the whole HLA chain; the
groove is its first 182 residues, and the two are kept as separate features.

### 1.3 PAE rows and columns use the same token axis

Spearman between PAE and CA–CA distance, over ~20,000 sampled token pairs per
fold, 45 Boltz folds:

| Token order | mean | extreme |
|---|---:|---:|
| as read (identity) | **+0.847** | min +0.805 |
| random permutation of the token axis | −0.001 | max +0.055 |

PAE is **not symmetric** on any of the 90 folds, which is why both directions
are kept as separate features rather than one being derived from the other.

### 1.4 Boltz 2.1.1 *does* emit a pair-specific ipTM — with a non-obvious orientation

The contract says to use a pair score "only if the pinned model actually emits
it" with a verified mapping. It does: every Boltz confidence JSON carries
`pair_chains_iptm`. Two things had to be established before using it.

**Chain index → chain.** In `boltz/model/modules/confidence_utils.py`
(`compute_ptms`), the keys are the sorted unique `asym_id` values, and
`asym_id` is assigned by `enumerate` over the input chains in
`data/parse/schema.py:1316`. The mmCIF atom-site loop iterates the same array
(`data/write/mmcif.py:142`). So index 0, 1, 2 = chains A, B, C = HLA, beta2m,
peptide — the same order section 1.2 confirmed empirically.

**Orientation.** The mask is
`(asym_id[:, None, :] == idx1) * (asym_id[:, :, None] == idx2)`, summed over
the last axis and maxed over the first. So `pair_chains_iptm[a][b]` aggregates
**rows in chain b against columns in chain a** — the transpose of the naive
reading — and it is a *max over rows*, not a mean.

Testing that against the arrays, per ordered chain pair, across folds:

| Statistic matched to `pair_chains_iptm[a][b]` | Spearman range over the 8 ordered pairs |
|---|---|
| source orientation: min over rows in **b** of the row-mean PAE to columns in **a** | **−0.98 to −1.00** |
| naive orientation: same statistic on the transposed block | −0.10 to −0.93, one at **+0.02** |

The source orientation is near-perfect on every pair; the naive one is
uninformative on two of them. Exported columns therefore name both the pair and
the direction, e.g. `iptm_pair_peptide_rows_hla_chain_cols`. A pooled
correlation over *all* block types gave an ambiguous answer and would have
picked the wrong permutation — the per-pair test is what settles it.

**ESMFold2 emits `pair_chains_iptm` as a list, not a keyed dict, and its
chain-index mapping has not been verified.** Those columns are therefore left
empty for ESMFold2 rather than filled with a number that might describe the
wrong interface. Production is Boltz-2 only, so this costs nothing.

### 1.5 Global ipTM stays global

`global_iptm` is Boltz's `iptm`. In the arm-B three-chain complex it covers the
HLA/beta2m interface as well as peptide/HLA, so it is never named or presented
as peptide-interface confidence. A test asserts that no column containing
"iptm" is exported without either a `global_` prefix or an explicit
`_rows_…_cols` pair-and-direction suffix.

## 2. Where the real files disagreed with the documentation

Two disagreements, one of which matters.

1. **`docs/BOLTZ_PIPELINE.md:375` is wrong.** It states that
   "`pair_chains_iptm` is the peptide-HLA interface ipTM … Boltz-2 exposes only
   a global ipTM", and uses that as an argument in favour of ESMFold2. The
   pinned `boltz==2.1.1` output contains `pair_chains_iptm` on **every one of
   the 45 Boltz folds**. The line is stale; I have not edited that file. Note
   this does not reopen the production decision — ESMFold2 failed its gate on
   pose, not on confidence outputs — but the claim should not survive into the
   write-up.
2. **The contract's anticipated arm-B PAE shape and slices were right.** No
   change needed. It is verified rather than assumed, so a future construct
   change will fail loudly instead of silently.

## 3. Feature definitions

129 columns: 20 key/provenance, 109 numeric features. Keyed by
`(model, allele, peptide, arm, seed)`; `pair_id`, `split` and `cluster_id` are
attached by joining `pepstab.data.load_with_splits()` on `(allele, peptide)`,
**never positionally on `pair_id`**.

### 3.1 Core features (the contract's list)

| Group | Columns | Definition |
|---|---|---|
| Peptide pLDDT | `pep_plddt_{mean,min,max,std}`, `pep_plddt_p1..p9` | pLDDT of the peptide tokens, 0–1 as Boltz emits it |
| Peptide → groove PAE | `pae_pep_rows_groove_cols_{mean,min,max,std}`, `…_p1..p9` | `pae[peptide tokens, groove tokens]`; per-position values are row means |
| Groove → peptide PAE | `pae_groove_rows_pep_cols_{mean,min,max,std}`, `…_p1..p9` | `pae[groove tokens, peptide tokens]`; per-position values are column means |
| Asymmetry | `pae_pep_groove_asymmetry` | the first mean minus the second |
| Contacts | `groove_contacts_4p5A_total`, `…_mean_per_residue`, `…_p1..p9`, `groove_residues_contacted` | heavy-atom pairs under 4.5 Å between a peptide residue and HLA residues 1–182; the same cutoff and selection as `scripts/ectodomain_pose_check.py` |
| Distance | `pep_min_dist_to_groove_A`, `pep_max_min_dist_to_groove_A`, `pep_min_dist_to_groove_p1..p9` | closest heavy-atom distance to the groove, overall and per position |
| Burial | `pep_sasa_alone_A2`, `pep_sasa_in_groove_A2`, `pep_buried_sasa_A2`, `pep_buried_sasa_frac`, `pep_buried_sasa_frac_p1..p9` | Shrake-Rupley SASA (probe 1.4 Å, 200 points, ProtOr radii) of the peptide alone minus its SASA in the **groove-only** complex |

"Groove" is HLA residues 1–182 everywhere, and **the same receptor atom
selection is used for contacts, distances and burial**, so alpha3 and beta2m
never occlude the peptide in one feature and not another.

### 3.2 Additional ectodomain / beta2m features, separately named and defined

These are not in the contract's core list, so they get their own names and
these definitions. All are `None` on arms A and C, which have no alpha3 or
beta2m.

| Column | Definition |
|---|---|
| `hla_chain_plddt_mean` | mean pLDDT over the **whole** HLA chain (275 residues in arm B), as distinct from `groove_plddt_mean` over residues 1–182 |
| `alpha3_plddt_mean` | mean pLDDT over HLA residues 183–275 |
| `beta2m_plddt_mean` | mean pLDDT over the beta2m chain |
| `pae_pep_rows_alpha3_cols_mean`, `pae_alpha3_rows_pep_cols_mean` | peptide/alpha3 PAE blocks, both directions |
| `pae_pep_rows_b2m_cols_mean`, `pae_b2m_rows_pep_cols_mean` | peptide/beta2m PAE blocks, both directions |
| `pae_groove_rows_{alpha3,b2m}_cols_mean` and the reverse | groove against alpha3 and beta2m, both directions |
| `pae_pep_intra_mean`, `pae_groove_intra_mean` | within-block PAE |
| `complex_plddt_mean_tokens` | mean pLDDT over every token (recomputed from the array, not read from the JSON) |

### 3.3 Confidence scalars

Prefixed `global_` when they describe the whole complex:
`global_confidence_score`, `global_ptm`, `global_iptm`, `global_protein_iptm`,
`global_ligand_iptm`, `global_complex_plddt`, `global_complex_iplddt`,
`global_complex_pde`, `global_complex_ipde`.

Per chain: `chain_ptm_hla_chain`, `chain_ptm_b2m`, `chain_ptm_peptide`.

Pair ipTM, direction in the name, Boltz only (section 1.4):
`iptm_pair_peptide_rows_hla_chain_cols`,
`iptm_pair_hla_chain_rows_peptide_cols`, and the four beta2m pairings.
`hla_chain` is the whole 275-residue chain, **not** the groove — Boltz scores
whole chains and there is no groove-only pair score to be had.

### 3.4 Construct provenance

Three alleles have no full-length IMGT record, so the alpha3 of the folded
275-residue construct was taken from a relative. Every structural feature on
those rows is computed on a partly-synthetic construct, so the flag travels
with the data rather than living only in prose:

| Column | Values |
|---|---|
| `alpha3_provenance` | `exact` (69 alleles) / `borrowed_relative` (3) / `wildtype_of_engineered` (3) |
| `alpha3_borrowed` | `True` only for `borrowed_relative` |
| `alpha3_source_allele` | the donor |

| Allele | alpha3 donor | cohort rows (train / val / test) |
|---|---|---|
| HLA-A\*02:50 | A\*02:01 | 375 (255 / 43 / 77) |
| HLA-A\*24:19 | A\*24:07 | 378 (260 / 41 / 77) |
| HLA-B\*08:03 | B\*08:01 | 350 (243 / 40 / 67) |
| **total** | | **1,103 (758 / 124 / 221)** |

**A correction worth recording.** The obvious rule `a3_source != allele` flags
**all 75** alleles, because `hla75_ectodomain_b2m.csv` drops the `HLA-` prefix
on the exact-match rows and keeps it on the borrowed ones. Both sides must be
normalised. The three engineered C67S alleles (1,135 further cohort rows) take
alpha3 from the wild type of the *same* allele while keeping the measured
alpha1/alpha2, which is a different situation; they get their own label
(`wildtype_of_engineered`) rather than being counted as borrowed.

This column lets stage 5 **test** whether borrowed-alpha3 alleles underperform.
HLA-A\*24:19 is separately the worst-ranked validation allele for the sequence
baseline (`reports/limitations.md` §6.6), but that is one of three borrowed
alleles and is a coincidence until tested. Nothing is concluded from it here.

## 4. Validation against the pilot's 90 folds

**90 of 90 extracted, 0 failures, 0 unmatched `pair_id`, no non-finite value in
any feature column.**

### 4.1 The four numbers the pilot report already published, recomputed

The extractor was written independently of `scripts/ectodomain_pose_check.py`
and derives its boundaries rather than using that script's fixed
`pae[start:, :182]` slices. The two agree exactly:

| Quantity | Max abs difference vs the pose checker, 90 folds |
|---|---|
| peptide → groove PAE mean | 5.9 × 10⁻⁷ |
| groove → peptide PAE mean | 3.0 × 10⁻⁷ |
| per-position peptide pLDDT | 1.1 × 10⁻¹⁶ |
| per-position 4.5 Å contact counts | **0** (exact) |

And the four Spearman correlations the stage-4c report quotes, recomputed from
the feature table against `pilot_pose_scores.csv`:

| Reported in `stage4c_ectodomain_pilot.md` | Recomputed here |
|---|---|
| Spearman(peptide-to-groove PAE, heavy RMSD) = +0.635 | **+0.635** |
| Spearman(peptide pLDDT, heavy RMSD) = −0.540 | **−0.540** |
| within arm A: +0.782 | **+0.782** |
| within arm A: −0.621 | **−0.621** |

Both independently-derived confidence medians also reproduce: arm-B peptide
pLDDT 0.987 (Boltz) vs 0.912 (ESMFold2), peptide-to-groove PAE 1.32 vs 3.98.

### 4.2 The pilot feature table

Boltz-2, arm B, median over three seeds:

| Complex | pep pLDDT | pep→groove PAE | groove→pep PAE | contacts 4.5 Å | groove residues | buried frac | global ipTM | pair ipTM (pep rows) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A\*02:01 LLWNGPMAV | 0.990 | 1.175 | 0.993 | 346 | 31 | 0.780 | 0.986 | 0.970 |
| A\*11:01 KTFPPTEPK | 0.980 | 1.277 | 0.871 | 355 | 33 | 0.782 | 0.989 | 0.968 |
| B\*07:02 IPRRNVATL | 0.969 | 1.959 | 1.055 | 346 | 30 | 0.745 | 0.987 | 0.946 |
| B\*08:01 ELRRKMMYM | 0.988 | 1.292 | 1.101 | 392 | 38 | 0.772 | 0.988 | 0.940 |
| B\*15:01 ILGPPGSVY | 0.987 | 1.394 | 0.828 | 373 | 35 | 0.862 | 0.989 | 0.971 |

Pooled over all five complexes, by model and arm:

| Model | Arm | pep pLDDT | pep→groove PAE | groove→pep PAE | contacts | buried frac | global ipTM |
|---|---|---:|---:|---:|---:|---:|---:|
| Boltz-2 | A | 0.982 | 1.509 | 1.135 | 361 | 0.786 | 0.989 |
| Boltz-2 | **B** | 0.987 | 1.317 | 0.993 | 355 | 0.776 | 0.988 |
| Boltz-2 | C | 0.982 | 1.387 | 1.088 | 366 | 0.785 | 0.990 |
| ESMFold2 | A | 0.912 | 4.137 | 1.816 | 345 | 0.785 | 0.959 |
| ESMFold2 | B | 0.912 | 3.984 | 1.697 | 353 | 0.789 | 0.972 |
| ESMFold2 | C | 0.912 | 4.139 | 1.832 | 347 | 0.788 | 0.961 |

Note the **geometry features barely separate the models** (contacts 345–366,
buried fraction 0.776–0.789 across every model and arm) while the confidence
features separate them sharply. That is consistent with the pilot's finding
that ESMFold2's error is in side-chain placement at an otherwise correct
backbone pose, and it is a reason to carry both kinds of feature into stage 5
rather than only one.

### 4.3 Seed and arm variation

Full table: `reports/stage4c5_feature_variation.csv`. Production is seed 0 only
and arm B only, so what matters is whether a feature is dominated by seed
noise. It is not — **for 0 of 109 numeric features does the within-complex seed
spread exceed the between-complex spread**, and the median seed share of total
variance is 0.195.

Boltz-2 arm B, standard deviation between complexes against standard deviation
across the three seeds of the same complex:

| Feature | between-complex sd | seed sd | ratio |
|---|---:|---:|---:|
| `pep_plddt_mean` | 0.0085 | 0.0005 | 18.8 |
| `pep_plddt_min` | 0.0314 | 0.0018 | 17.2 |
| `pep_buried_sasa_frac` | 0.0440 | 0.0033 | 13.3 |
| `global_confidence_score` | 0.0019 | 0.0001 | 13.0 |
| `pae_pep_groove_asymmetry` | 0.2885 | 0.0266 | 10.8 |
| `pep_buried_sasa_A2` | 96.2 | 8.89 | 10.8 |
| `chain_ptm_peptide` | 0.0021 | 0.0002 | 9.3 |
| `groove_residues_contacted` | 3.17 | 0.35 | 9.1 |
| `iptm_pair_peptide_rows_hla_chain_cols` | 0.0143 | 0.0018 | 7.9 |
| `pae_pep_rows_groove_cols_mean` | 0.3068 | 0.0407 | 7.5 |
| `groove_contacts_4p5A_total` | 21.2 | 4.27 | 5.0 |
| `pae_groove_rows_pep_cols_mean` | 0.1144 | 0.0249 | 4.6 |
| `global_iptm` | 0.0013 | 0.0004 | 3.4 |
| `global_ptm` | 0.0014 | 0.0005 | 2.7 |

The two weakest are the global scores, which is expected: they are dominated by
the HLA fold, which is the same molecule in every fold of a given allele.

Arm effects are consistently about twice the seed effect (mean within-complex
sd across arms 0.0026 vs 0.0006 for `pep_plddt_mean`, 0.178 vs 0.066 for
peptide-to-groove PAE). Arm comparisons are limited to the 90 shared folds, as
the contract requires.

## 5. Confidence is a feature, never a row filter

Per the pilot: across the 45 Boltz folds confidence **ranks** error (section
4.1) but does **not isolate** the one genuine failure. The extractor therefore
has no confidence threshold anywhere, and a row is dropped for no reason other
than a structural or parsing failure, which is recorded rather than discarded.
A test feeds in a fold with pLDDT forced to 0.2, PAE to 25 Å and ipTM to 0.05
and asserts it still extracts with `status == "ok"`.

As a diagnostic only — n = 5 complexes, possible training-set recall, and
emphatically not a result — the strongest rank correlations with peptide heavy
RMSD across the 45 Boltz folds are `global_complex_plddt` (−0.819),
`groove_plddt_mean` (−0.806) and `global_complex_ipde` (+0.782), all stronger
than the two the contract names. Whether any of this survives on 28,166 pairs
is a stage-5 question on validation rows.

## 6. Failure handling

`extract_path` converts any exception into a row with `status` set to the
exception and whatever identity fields `metadata.json` still yields, so a bad
fold becomes a visible row rather than a silent absence or a crashed batch.
Coverage and failure counts are printed by every CLI mode. The contract's
sequence-model fallback for unresolved structural failures is stage 5's to
apply; the feature table's job is to make the unresolved rows visible, which
the `status` column does.

## 7. Phase 2 plan (not started)

`modal_app/feature_extraction.py` is written but **has not been run**. The
project invariant requires a passing 3–5 example end-to-end smoke before any
batch pass, and `::smoke` exists for exactly that.

- **CPU only.** No function in the app declares `gpu=`. The production fold is
  using up to 10 concurrent A10Gs per workspace and a GPU container here would
  contend for those slots.
- **Volume mounted, output not downloaded.** `pepstab-structures` is mounted at
  `/structures`; only the small feature CSV comes back.
- **Discovery skips `_smoke`, `_shards` and `_failed`**, which production
  writes next to the cohort output under
  `/stage4c/ectodomain-20261004/boltz2/production/<allele_slug>/<complex_id>/`.
  Ingesting them would neither error nor look wrong; it would add ~10
  pre-launch harness folds to the cohort table. Tested against a synthetic
  tree that contains all three traps.
- **One profile at a time is a first-class mode, not a failure mode.** The
  halves are interleaved pair-by-pair, so either alone is balanced across
  alleles and splits: 14,083 pairs from one workspace is already an unbiased
  diagnostic, and waiting for both is unnecessary. The output is labelled
  `cohort_coverage = half`.
- **`--profile` must match `MODAL_PROFILE`**, asserted, as the project
  invariant requires. `extract` additionally cross-checks the extracted pairs
  against `data/structural_cohort.csv` and fails if any belong to the other
  half — that is what a wrong Volume mount would look like.
- **Concatenation is a separate, local step** with an explicit row-count
  assertion (`concat --require-full`): it reports rows, cohort pairs covered,
  pairs missing and rows not in the cohort, and exits non-zero unless all
  28,166 are present. Nothing in the output path makes a half-sized table
  obvious.

### Forecast cost and time

Measured locally: **0.096 CPU-seconds per arm-B fold** (15 arm-B pilot folds,
warm, single process), peak RSS 641 MB. The full cohort is therefore
**0.75 core-hours of actual compute**. Modal bills *reserved* cores and memory
against wall time, so at `cpu=2.0`, `memory=4096`, 20 containers and the
project's verified rates ($0.04730/core-hour, $0.00800/GiB-hour from
`reports/ectodomain_rates.json`), a half that takes 10 minutes of wall time
costs 20 × 2 × (1/6) h × $0.0473 + 20 × 4 × (1/6) h × $0.008 ≈ **$0.32 + $0.11
= $0.42**, so **under $1 for both halves**. This is a *forecast*: wall time is
Volume-read-bound, not CPU-bound, and only the smoke run converts it to a
measured figure. That is the number to check before the full pass.

## 8. Limits of what phase 1 establishes

- It validates **extraction and feature definitions**, on the five pilot
  complexes. It says nothing about whether these features predict half-life;
  no model was fitted and nothing was scored on validation or test.
- The five pilot complexes are all in the training split and all have crystal
  structures, so they are a pipeline and pose check with possible training-set
  recall, not a structural accuracy benchmark.
- The pair-ipTM orientation is established from the pinned `boltz==2.1.1`
  source plus near-perfect rank agreement on 45 folds. It is not an exact
  numerical reproduction of `pair_chains_iptm`, which would need the raw PAE
  logits rather than the expected-value array that is written out.
- `modal_app/feature_extraction.py` has not executed. `modal` is not installed
  in the local environment, so it has been syntax-checked only; the `::smoke`
  run is its first real test.
