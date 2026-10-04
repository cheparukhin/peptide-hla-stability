# Stage 3c: elution as external validation

**The pass.** Take a model trained only on peptide–HLA dissociation half-life
and ask it to rank MHC Motif Atlas eluted ligands above length- and
allele-matched human-proteome decoys. No retraining, no fitting of any kind to
the atlas: it enters as a scoring target and nothing else. This is the
project's only genuinely external validation — a different assay (mass
spectrometry of peptides recovered from cells) measuring a different biological
event (antigen presentation) — which makes it a stronger claim than any
within-dataset correlation.

**The result.** The 30-network sequence ensemble separates eluted ligands from
proteome decoys at **median AUROC 0.966** (IQR 0.944–0.978) across 51 alleles,
and still at **0.916** when the decoys are other alleles' eluted ligands rather
than random proteome peptides. The stability signal transfers. What the gap
between those two numbers measures, and what the pass does *not* establish, is
in [Interpretation](#interpretation) — read it before quoting either number.

Reproduce with:

```
.venv/bin/python scripts/stage3c_elution_validation.py
```

Code: [`pepstab/elution.py`](../pepstab/elution.py),
[`scripts/stage3c_elution_validation.py`](../scripts/stage3c_elution_validation.py),
[`tests/test_elution.py`](../tests/test_elution.py). The finding that motivated
this pass — eluted ligands have ~7× higher median half-life among the 140
overlapping pairs — is in
[`elution_stability_finding.md`](../elution_stability_finding.md) and is not
repeated here.

## 1. Data provenance

| | |
|---|---|
| **Atlas file** | `external/data_classI_all_peptides.txt` |
| Source URL | `http://mhcmotifatlas.org/data/classI/all_peptides.txt` |
| Where that link lives | the "Download all peptides" link on the Class I Alleles page of mhcmotifatlas.org. A browser flattens the `download` attribute path to `data_classI_all_peptides.txt`, which is the name `04_elution_stability_test.py` already expected. |
| Retrieved | 2026-10-04T02:59:55Z |
| Server `Last-Modified` | 2025-02-25T14:56:59Z |
| Size | 8,030,388 bytes |
| Rows | 487,344 lines = header + **487,343** allele–peptide rows |
| SHA-256 | `8547c64299905a15aa2f9b79cdd9c1c88e99bdecabe40580a4d0c59604efec87` |

The row count matches the 487,343 recorded in `elution_stability_finding.md`
exactly, so this is the same release that finding was computed on. Two columns,
`Allele` and `Peptide`; no affinity, no assay metadata, no source protein, no
confidence. Positives only.

| | |
|---|---|
| **Decoy source** | `external/uniprot_human_reviewed.fasta.gz` (already local) |
| Content | UniProt reviewed (Swiss-Prot) human proteome, 20,431 proteins |
| SHA-256 | `dc8ef8cbf0fc1b857b3be9a559613533ec49824079fb057e31fdfcf578cb229a` |
| Distinct 9-mer windows | 10,409,122 |

`external/` is gitignored. Both hashes are re-computed into
`reports/stage3c_provenance.json` on every run, so a changed input fails
loudly rather than silently shifting the numbers.

`data/rasmussen_et_al_dataset.csv` is untouched; `(cd data && shasum -a 256 -c
SHA256SUMS)` passes.

## 2. The decoy protocol, declared before any score existed

Every constant below lives in `pepstab/elution.py` and was written down, and
sent to the orchestrator, before a single AUROC was computed. AUROC moves with
the decoy ratio, so a ratio chosen after seeing results is not a result.

| Parameter | Value | Why this value |
|---|---|---|
| Decoy : ligand ratio | **10 : 1** | Fixed in advance. Chance AUPRC is therefore 1/11 = 0.0909 for every allele, which makes AUPRC readable across the panel. |
| Seed | **20261004** | One generator drives the ligand subsample and every decoy draw. The scoring set is byte-identical across runs (`test_scoring_set_is_reproducible_and_seed_dependent`). |
| Peptide length | 9 | The stability dataset is 100% 9-mers. |
| Minimum ligands per allele | **100** | Set from the count distribution alone, before scoring. It is deliberately non-binding: the smallest shared allele has 101 ligands, so **no allele is dropped**, and none can be dropped for scoring badly. |
| Ligand cap per allele | **2,000** | A CPU-budget cap, not a statistical one. At 2,000 positives against 20,000 decoys the standard error on AUROC is under 0.01. Alleles above the cap are subsampled uniformly at the seed. |
| Enrichment fractions | top 1%, top 10% | Fixed in advance. |

### Allele panel: 51 alleles

Atlas allele codes are parsed with the same rule as
`04_elution_stability_test.py` (`A0201` → `HLA-A*02:01`); non-human and
non-A/B/C entries (`H2-Kb`, `G0101`, `E0103`, …) do not parse and are dropped.
The panel is the intersection with the stability dataset's **72 non-C67S**
alleles, which yields 51.

**The three C67S constructs are excluded.** `HLA-B*14:01(C67S)`,
`HLA-B*14:02(C67S)` and `HLA-B*39:06(C67S)` are engineered molecules; residue
67 sits inside the NetMHCpan pseudosequence the model consumes, so the
construct and the wild-type allele are different inputs. Atlas ligands were
eluted from wild-type molecules, and scoring them against a mutated groove
would be a mismatch rather than a measurement. These three alleles have no
wild-type counterpart in the dataset, so they simply leave the panel.

### Exclusions, applied symmetrically to both classes

| Filter | Rows affected |
|---|---|
| Atlas 9-mers with a modified residue (the atlas writes phosphorylated S/T/Y in lower case, e.g. `LSsRtNLQY`) | 2,250 ligand rows dropped. The stability assay has no phosphopeptides and the encoder has no column for a modified residue. Their *unmodified backbone* still blocks the matching proteome 9-mer from being drawn as a decoy. |
| Peptide measured anywhere in `rasmussen_et_al_dataset.csv` | **284 ligand rows dropped** (106 distinct peptides), and all 5,633 measured peptides blocked from the decoy pool. |
| Peptide eluted on *any* allele in the atlas | 182,311 distinct 9-mers blocked from the decoy pool — not just same-allele ligands. |

The second filter is the important one, and it is applied to **both** classes,
not just the decoys. Filtering only the decoys would make "absent from the
stability dataset" a property of the negative class, and a model that had
memorised its training peptides could then score the filter rather than the
biology. `test_exclusion_is_symmetric_across_both_classes` pins this.

For orientation: 140 atlas `(allele, peptide)` pairs match a measured pair
exactly, reproducing the count in `elution_stability_finding.md`; 139 of those
sit on this panel. The peptide-level filter is stricter and removes 284 ligand
rows, because a peptide measured on one allele is dropped from every allele.
Those 106 peptides span all three splits (353 train, 43 val, 116 test rows),
which is precisely why the filter is at peptide level.

Decoy pool after exclusions: **10,241,634** of the 10,409,122 proteome 9-mers
(167,488 removed). Decoys are drawn without replacement within an allele, and
`test_no_decoy_is_an_atlas_ligand_or_a_measured_peptide` checks the resulting
set.

### Resulting scoring set

**51 alleles · 81,600 ligands · 816,000 decoys · 897,600 rows.** Nothing is
fitted to any of it.

## 3. The model under test

No model is persisted in this repository, so the arm is *reproduced* from its
frozen stage 2 configuration rather than loaded: refit on `split == "train"`
with the same cluster-respecting inner folds and the same seeds that
`scripts/baseline_sequence.py` and `scripts/baseline_ensemble.py` use. The
atlas is never in a fit set and the test split is never read.

Two arms, reported together because ensemble parity matters across the project
and an external number from a single network would not be comparable to
anything else we report:

| Arm | What it is | Val median per-allele SCC |
|---|---|---|
| `seq_baseline` | the single network stage 2 selected: one-hot, peptide + pseudosequence, h256×64, L2 1e-5, seed 0 (`reports/stage2_headline.json`) | 0.610 |
| `seq_ensemble` | the strong stage 2 baseline: 5 CV folds × 2 encodings × 3 seeds = 30 networks, averaged | 0.693 |

**Refit verification, quantitative.** Before anything is scored, each refit
predicts the validation split and is compared against the committed prediction
file:

| Arm | Compared against | max abs deviation | Pearson r | Ranking |
|---|---|---|---|---|
| `seq_baseline` | `preds/seq_baseline.csv` | **2.24 × 10⁻⁷** | 1.0000000 | identical |
| `seq_ensemble` | `preds/seq_ensemble_pep_pseudo.csv` | **8.07 × 10⁻⁷** | 1.0000000 | — |

Not bit-identical, because float32 BLAS accumulation order varies, but seven
decimal places and an identical rank order on the single-network arm. These are
the stage 2 models, not approximations of them. The check runs on every
invocation and is recorded in `reports/stage3c_provenance.json`.

## 4. Result: ligands versus proteome decoys

Median and IQR across the 51 alleles. Chance AUROC is 0.5; chance AUPRC is
1/11 = 0.0909. At the 10:1 ratio the enrichment ceiling is 11.0 in the top 1%
and 10.0 in the top 10% (at top 10% the window holds more rows than the allele
has ligands, so precision there cannot exceed 0.909).

| Arm | AUROC median [IQR] | AUROC min–max | AUPRC median [IQR] | Precision top 1% | Enrichment top 1% | Precision top 10% | Enrichment top 10% |
|---|---|---|---|---|---|---|---|
| `seq_ensemble` | **0.9656** [0.9435, 0.9779] | 0.805 – 0.991 | **0.7804** [0.681, 0.858] | 0.939 | 10.33× | 0.701 | 7.71× |
| `seq_baseline` | 0.9502 [0.9182, 0.9648] | 0.730 – 0.982 | 0.6676 [0.571, 0.813] | 0.877 | 9.65× | 0.630 | 6.93× |
| *random scores (harness null)* | 0.4974 [0.4927, 0.5013] | — | 0.0911 | 0.091 | 1.00× | — | — |

The null row is the same harness fed uniform random predictions through the
`--scores` path; it returns chance on both metrics, which is what says the
AUROC above is coming from the model and not from the scoring code.

Every allele scores above chance. Under the ensemble, **no allele falls below
0.80** and only three fall below 0.90: `HLA-A*30:02` (0.805), `HLA-B*13:02`
(0.805) and `HLA-A*30:01` (0.890). The strongest are `HLA-B*15:10` (0.991),
`HLA-A*24:07` (0.990) and `HLA-A*23:01` (0.988).

Full per-allele tables, all columns:

- `reports/stage3c_per_allele_seq_ensemble.csv`
- `reports/stage3c_per_allele_seq_baseline.csv`

The ensemble is better than the single network on every summary metric, by
+0.015 median AUROC and +0.113 median AUPRC. That ordering matches the
validation ordering (0.693 vs 0.610 median per-allele SCC), which is a mild
consistency check — the arm that ranks half-lives better within the stability
assay also ranks ligands better in the elution assay.

## 5. Control: allele-swapped decoys

Proteome decoys are negatives on **every** axis at once. A random proteome
9-mer was not cleaved out by the proteasome, not transported by TAP, not
presented, not ionised in the mass spectrometer — *and* not stable. So part of
the 0.966 is "does this look like a presentable peptide at all" rather than "is
this peptide stable on *this* allele". That is confound reason 3 in the plan's
own list, showing up directly in the decoy design.

The control replaces each allele's decoys with **eluted ligands of other
alleles**, drawn at the same 10:1 ratio, from the same seed, under the same
symmetric exclusions, and never including a peptide the target allele itself
presents. The positives are the identical ligand sample, so only the negatives
change. Those negatives are real presented peptides, so source-protein
abundance, cleavage, transport and ionisation are held roughly constant and
what is left is allele specificity.

| Arm | Proteome decoys | Allele-swapped decoys | Drop |
|---|---|---|---|
| `seq_ensemble` | 0.9656 [0.9435, 0.9779] | **0.9157** [0.8598, 0.9521] | −0.050 |
| `seq_baseline` | 0.9502 [0.9182, 0.9648] | 0.8939 [0.8308, 0.9332] | −0.056 |
| *random scores* | 0.4974 | 0.4976 | — |

AUPRC falls further, from 0.780 to 0.557 for the ensemble, because precision is
more sensitive to hard negatives than a rank statistic is.

This is the expected and honest pattern: **substantially lower, but far above
chance.** Roughly a twentieth of the AUROC is presentability rather than
allele-specific discrimination; the rest survives negatives that are themselves
presented peptides. Only one allele drops near chance — `HLA-A*30:02` at 0.644,
the same allele that is weakest against proteome decoys.

### The donor-distance gradient

Each swapped decoy carries the pseudosequence Hamming distance from the target
allele to the *nearest* allele known to present it. Splitting the decoys at the
pooled median (distance 14 of 34 pseudosequence positions) and re-scoring the
same ligands against each half:

| Arm | Near donors (distance ≤ 14) | Far donors (distance > 14) |
|---|---|---|
| `seq_ensemble` | 0.8906 [0.8154, 0.9326] | 0.9562 [0.9080, 0.9756] |
| `seq_baseline` | 0.8641 [0.7994, 0.9136] | 0.9168 [0.8832, 0.9575] |

The gradient runs the right way and is large: a ligand of a groove that
resembles the target is a much harder negative than a ligand of a distant
groove (−0.066 AUROC for the ensemble). This is a result in its own right —
the model is reading groove chemistry, not just peptide chemistry, and it
degrades smoothly as the grooves converge rather than collapsing. Per-allele
tables: `reports/stage3c_donor_distance_seq_ensemble.csv`,
`reports/stage3c_donor_distance_seq_baseline.csv`.

## 6. Control: wrong-allele pseudosequence

A second specificity check, on the primary proteome-decoy set: score every row
against a *different* allele's pseudosequence, under a fixed-point-free
permutation of the 51 alleles (seed 20261004). The peptides are unchanged, so
whatever survives is the part of the ranking that does not need the right
groove.

| Arm | Correct allele | Wrong allele |
|---|---|---|
| `seq_ensemble` | 0.9656 | **0.6965** |
| `seq_baseline` | 0.9502 | 0.6972 |

Enrichment in the top 1% collapses to 1.00× for the ensemble — exactly chance.
The model is not simply flagging "peptide-shaped sequences": handing it the
wrong groove costs 0.27 AUROC and all of the top-of-list precision. The 0.70
that remains is the generic presentability signal already identified above,
plus whatever is shared between any two class I grooves.

## Interpretation

**What this establishes.** A model that has only ever seen cell-free
dissociation half-lives, trained under splits that keep every held-out peptide
at least four substitutions from every training peptide, ranks
immunopeptidomics-eluted ligands above matched decoys at median AUROC 0.966
over 51 alleles, with 94% precision in its top 1%. The peptide universes are
near-disjoint — 140 of ~147,000 pairs overlap, and even those are filtered out
here — so this is transfer to a different assay measuring a different
biological event, not a restatement of within-dataset correlation. The
discrimination is allele-specific: it survives negatives that are themselves
eluted ligands (0.916), it degrades as the donor groove approaches the target
groove, and it largely disappears when the groove is swapped (0.697).

**What this does not establish.**

1. **It is not a measurement of stability prediction accuracy.** Elution is a
   selection effect with at least four filters besides stability:
   source-protein abundance, proteasomal cleavage specificity, TAP transport,
   and mass-spec ionisation efficiency. A peptide in the atlas passed all of
   them. Nothing here separates the contribution of stability from the others,
   and a model that had learned only "this peptide is abundant and ionises
   well" would also score well against proteome decoys. The allele-swapped
   control bounds this: it holds those four filters roughly constant on both
   sides and costs 0.050 AUROC, so **about a twentieth of the proteome-decoy
   AUROC is attributable to generic presentability** rather than to
   allele-specific discrimination. That is a bound on this decoy design, not a
   decomposition of the biology.

2. **It is not a ranking of stability within presented peptides.** The task
   here is ligand versus non-ligand, a binary discrimination. The project's
   primary metric — per-allele Spearman against measured half-life — asks a
   harder question, and this pass says nothing about it. The validation SCCs
   (0.610 single, 0.693 ensemble) remain the number to quote for accuracy.

3. **AUROC here is not comparable to a published presentation predictor.**
   NetMHCpan-4.x and friends train on elution data; this model has never seen
   any. The right reading of 0.966 is "a stability model transfers", not "a
   stability model competes with a presentation model". We did not run such a
   comparison and this report does not claim one. (Nor could NetMHCstabpan
   serve as a comparator anywhere in this project — it trained on every peptide
   in our test split.)

4. **The atlas has an error rate of roughly 2%.** The existing finding
   estimated this from motif deconvolution artefacts: among 140 eluted ligands
   with a measured half-life, three have a half-life of exactly zero, and
   `FPEHIFPAL` appears with zero stability on *two* different alleles
   (`HLA-B*51:01` and `HLA-B*08:01`) — a pattern more consistent with a peptide
   being assigned to alleles it does not bind than with two independent
   instances of unstable presentation. So roughly 2% of this pass's positives
   are expected to be mislabelled, which puts a soft ceiling just under 1.0 on
   any AUROC reported here and means the weakest alleles may be partly
   measuring deconvolution quality rather than model quality.

5. **Decoys are assumed negatives.** A proteome 9-mer absent from the atlas may
   simply never have been sampled; mass spectrometry is positives-only and
   absence carries no information. At a 10:1 ratio and a realistic presentation
   rate this contaminates the negative class slightly, which depresses the
   reported AUROC rather than inflating it — so the numbers here are, in that
   one respect, conservative.

6. **One release, one decoy draw.** Every number is from the 2025-02-25 atlas
   release at seed 20261004. The seed sensitivity was not swept, though with
   816,000 decoys per arm the draw-to-draw variation on a panel median should
   be far below the differences reported.

**The honest summary.** Elution is not a stability assay, and this is not a
validation of stability prediction. It is evidence that what the model learned
from half-lives is real biophysics about the peptide–groove interaction rather
than an artefact of the assay panel or the splits — because that knowledge
transfers, allele-specifically, to a measurement nobody trained it on.

## Runtime and cost

CPU only, one laptop, no paid cloud compute, $0.

| Stage | Time |
|---|---|
| Atlas download (8.0 MB) | ~3 s |
| Build scoring set (parse atlas, tile 10.4M proteome 9-mers, draw decoys) | 21 s |
| Build allele-swapped control set | ~5 s |
| Refit `seq_baseline` (1 network) | 15 s |
| Refit `seq_ensemble` (30 networks) | 511 s |
| Score `seq_ensemble` over 897,600 rows × 3 sets (primary, swapped, wrong-allele) | ~13 min |
| **Total, all arms and controls** | **~23 min** |

Peak memory stays near 350 MB: the 897,600 × 860 feature matrix would be 3.1 GB
in one block, so prediction runs in 100,000-row chunks.

## Running it against another arm

The model under test is an argument, not a hard-coded assumption, so the ESM-2
arm (or any other) can be measured on identical rows:

```
# 1. write the exact scoring set — both decoy designs, 1,795,200 rows
.venv/bin/python scripts/stage3c_elution_validation.py \
    --emit-scoring-set reports/stage3c_scoring_set.csv

# 2. the other arm scores every distinct (allele, peptide) pair in it and
#    writes allele,peptide,y_pred

# 3. feed it back
.venv/bin/python scripts/stage3c_elution_validation.py \
    --arms esm2 --scores preds/esm2_elution_scores.csv
```

The scoring set is a pure function of the two input files and the seed, so step
1 can be re-run at any time and will produce the same rows. Incomplete coverage
is rejected rather than silently dropped. The wrong-allele pseudosequence
control needs the model itself and is skipped for `--scores` arms; the
allele-swapped control is not, and runs for every arm.

## Files

| Path | Contents |
|---|---|
| `pepstab/elution.py` | atlas parsing, decoy universe, both scoring-set builders, AUROC/AUPRC/enrichment, donor-distance stratification. All protocol constants. |
| `scripts/stage3c_elution_validation.py` | the CLI |
| `tests/test_elution.py` | 25 tests: decoy purity, filter symmetry, ratio and reproducibility, donor-distance correctness, metrics against sklearn |
| `reports/stage3c_summary.csv` | one row per arm × control, median/IQR/min/max for every metric |
| `reports/stage3c_per_allele_<arm>.csv` | per-allele AUROC, AUPRC, precision and enrichment |
| `reports/stage3c_per_allele_<arm>__swapped_decoys.csv` | the same against allele-swapped decoys |
| `reports/stage3c_per_allele_<arm>__wrong_allele.csv` | the same under a permuted pseudosequence |
| `reports/stage3c_donor_distance_<arm>.csv` | per-allele AUROC by donor pseudosequence distance |
| `reports/stage3c_provenance.json` | hashes, URL, every protocol constant, every filter count, the refit verification, runtime |
