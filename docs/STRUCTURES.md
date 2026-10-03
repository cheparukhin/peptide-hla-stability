# Experimental structures for the Rasmussen complexes

Ground-truth pMHC crystal structures matched against the stability dataset,
assembled before any prediction is run. Three tables, from broadest to narrowest:

| file | rows | what it is | readable version |
| --- | --- | --- | --- |
| [`data/pdb_pmhc_pairs.csv`](../data/pdb_pmhc_pairs.csv) | 841 | every peptide entity in every PDB entry whose α1/α2 groove matches a Rasmussen allele exactly | — |
| [`data/allele_pdb_templates.csv`](../data/allele_pdb_templates.csv) | 75 | one threading template per Rasmussen allele, tiered by groove match | [allele_pdb_templates.md](allele_pdb_templates.md) |
| [`data/pdb_rasmussen_overlap.csv`](../data/pdb_rasmussen_overlap.csv) | 32 | (allele, peptide) pairs with **both** a measured half-life and a deposited structure | [pdb_rasmussen_overlap.md](pdb_rasmussen_overlap.md) |

## What is in them

**`pdb_pmhc_pairs.csv`** — 840 PDB entries across 33 of the 75 alleles, allele
assigned by 100% identity over the 182-residue α1/α2 domain. 841 rows rather
than 840 because one entry (`9A0U`, a six-chain class I / class II docking
model) contributes two peptide entities; **rows are entities, not entries**, and
the per-allele counts in `allele_pdb_templates.csv` are entry counts. Of the 840
entries, 808 have an 8–15-residue polypeptide in the groove and **32 have no
peptide modelled at all** — empty or disordered. 813 are X-ray, 25 cryo-EM, 1
NMR, 2 unspecified (including the docking model, which is not experimental).
243 are TCR–pMHC complexes. The peptide entities give **510 distinct
(allele, peptide) pMHC pairs, 329 of them 9-mers**. Coverage is as lopsided as
the stability data: A\*02:01 alone has 393 entries, A\*24:02 has 71, A\*11:01 56.

**`allele_pdb_templates.csv`** — one template per allele, tiered: **33 tier A**
(exact groove, use directly), **24 tier B** (homology, groove intact), **18 tier
C** (homology, groove differs). Note that [HACKATHON_PLAN.md](../HACKATHON_PLAN.md)
defers template threading to the "not in scope" list; this table makes it cheap
if that changes, it does not promote it.

**`pdb_rasmussen_overlap.csv`** — the 32 pairs where a measured half-life and an
experimental structure exist for the *same* complex.

## The 32-pair overlap

**32 pairs, 0.11% of the 28,166 measurements.** That number is the one to carry
forward, and it largely defuses the memorisation worry: for 99.9% of the
dataset, a co-folding model has no deposited structure of that exact complex to
recall. A further **12 are near misses** — the peptide appears in Rasmussen but
the crystal structure is on a different allele.

They span **0.4 to 97.1 h, none censored**, so they cover nearly the full
dynamic range (99.8% of Rasmussen rows fall at or below 97.1 h). The
longest-lived is `HLA-A*11:01 / ATIGTAMYK` at 97.1 h ([6JOZ](https://www.rcsb.org/structure/6JOZ),
1.35 Å); the most-structured is `HLA-A*02:01 / SLLMWITQV` — NY-ESO-1 — with 13
entries; A\*02:01 accounts for 17 of the 32. Resolution runs 1.1–3.84 Å, with 20
of 32 at 2.0 Å or better.

**Six were deposited in 2021 or later** (`21EX`, `8T7R`, `7PBC`, `7LG2`, `7LG3`,
`7LFZ`), but only one is clearly post-cutoff for all models under consideration:
`21EX` ("Wild type p53WT-HLA-A2") was deposited 2025-12-10 and **released
2026-09-09** — under a month ago. It is the only pair that a September 2026
co-folding run could not have memorised. The other five may or may not post-date
ESMFold2's training snapshot (undocumented); check the model card before
claiming them as controls. Stage 4b.2 in [HACKATHON_PLAN.md](../HACKATHON_PLAN.md)
uses these rows as the pilot gate for both structure arms.

### Two uses, not just a note

1. **Pilot gate (stage 4a.1 and 4b.2).** These are the complexes used to validate
   both structure arms before any batch job launches. Co-fold them, measure RMSD
   against the crystal, and verify the peptide lands in the groove before
   committing budget to 28,000 folds. See
   [HACKATHON_PLAN.md](../HACKATHON_PLAN.md) stages 4a.1 and 4b.2 for the exit
   criteria.
2. **Register check.** Verify the predicted peptide sits in the canonical
   P2/PΩ-anchored conformation *here*, where the answer is known, before
   trusting predicted structures for the other 99.9%.

### Caveats on the 32

- **13 of the 32 are TCR-bound.** TCR engagement can perturb peptide
  conformation, so an RMSD control or an energy-function benchmark should
  prefer the **19 TCR-free** pairs (which still span the full 0.4–97.1 h range)
  and treat the TCR-bound ones as a separate stratum.
- **They are not a representative label sample.** Median half-life is 6.0 h
  against 1.1 h for the full dataset, and **none is a censored zero** against
  20.2% zeros overall. The set is biased toward stable, well-studied,
  immunodominant epitopes. Nothing about the zero spike can be validated here.
- **n = 32 is small.** The smallest correlation significant at p < 0.05 is
  r ≈ 0.35, and r = 0.5 would carry a 95% CI of roughly [0.18, 0.72]. Only a
  strong effect or a clean null is readable at this n — which is exactly the
  argument for extending to the 329 nine-mer PDB pairs (there, r ≈ 0.11 is
  detectable and r = 0.5 narrows to [0.41, 0.58]).

### What this check does not cover

Memorisation of the exact pair is rare; memorisation of the **groove** is not.
33 alleles have deposited structures, A\*02:01 alone with 393. A model can know
an allele's groove perfectly without ever having seen your peptide in it. That
residual risk is handled by the **leave-allele-cluster-out split**, not by this
overlap table.

## Affinity is not half-life — and whether to add an affinity column

This is the mismatch that governs the project. Every structure-derived score
under consideration — FoldX, PRODIGY, EvoEF2, an interface-energy decomposition
— estimates **K_d, the depth of the well**. The label is **ln2/k_off, the height
of the barrier out of it**. Two complexes with identical K_d can have very
different off-rates, and dissociation of a class I complex is thought to proceed
by partial unthreading from the anchor pockets rather than by the interface
melting uniformly. A perfect ΔG calculation is therefore predicting an
*adjacent* quantity, not the label. [HACKATHON_PLAN.md:17](../HACKATHON_PLAN.md#L17)
states the same thing from the other direction; this is the quantitative
version of it.

That is what makes the 32-pair overlap valuable rather than merely interesting.
**On those 32, structural error is zero** — the structure is experimentally
determined, so whatever an energy function produces there is the best it will
ever do.

### The test that settles it

1. Run PRODIGY or an EvoEF2-style decomposition on the 32 (ideally the 19
   TCR-free ones first) and correlate against `log t½`.
2. **If there is no correlation on perfect structures, there will be none on the
   28,134 threaded models.** That is established cheaply and decisively, and
   given the kinetic-versus-thermodynamic argument above it is a real result,
   not a null.
3. Extend to all **329 nine-mer PDB pairs** by pulling measured affinities from
   IEDB, where n is large enough for a usable confidence interval.

Everywhere outside the 510 structurally-resolved pairs you are scoring side
chains your repacking protocol invented, so the number partly measures the
protocol rather than the biology.

### Recommendation on the affinity column

Neither dataset carries an affinity column today, and the decision splits three
ways:

- **As the training target — no.** The label is half-life. Substituting or
  blending in affinity changes the task.
- **As an input feature — yes if it is cheap.** A predicted or IEDB-measured
  affinity per (allele, peptide) is one number, costs no GPU time, and is a
  strong baseline covariate precisely because it is adjacent rather than
  identical. The honest framing is that stability prediction exists as a
  separate problem *because* affinity is an incomplete proxy, so the feature
  should help without being sufficient. Treat its marginal gain over the
  sequence baseline as the experiment, same ladder as every other feature.
  One caveat from [HACKATHON_PLAN.md:140](../HACKATHON_PLAN.md#L140): the assay
  panel was partly selected by predicted affinity, so affinity is entangled with
  dataset inclusion and a gain may be partly a selection artefact.
- **As a multi-task auxiliary target — defer.** Plausible, but it needs
  affinities for most of the 28k rows and competes with the core comparison for
  the hours available.

The step-3 result above decides how much weight to put on the feature: if
interface energy does not track `log t½` even on experimental structures, an
affinity feature is unlikely to carry the model either, and the sequence
baseline keeps priority.
