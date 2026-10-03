# Do MS-eluted ligands have measurably higher pMHC stability?

Script: `04_elution_stability_test.py`. Inputs: `rasmussen_et_al_dataset.csv` (28,166 half-lives, 75 alleles) and `data_classI_all_peptides.txt` (MHC Motif Atlas class I, 487,343 allele–peptide rows).

## Question

The MHC Motif Atlas is a list of peptides recovered from MHC molecules by immunopeptidomics. It has exactly two columns — allele and peptide — with no affinity, no assay metadata, no source, no confidence. It is positives-only: absence from a mass-spec run carries no information.

A peptide nonetheless had to *survive* to be in it. Cell lysis, immunoprecipitation and acid elution take hours, and a complex that dissociates in minutes should not reach the mass spectrometer. That reasoning suggests every eluted ligand carries a hidden lower bound on stability — but it is an argument, not a measurement.

**Can we check it against measured half-lives, and does it justify augmenting the stability dataset with elution data?**

## Rationale for the test

The two datasets share 54 alleles, and the atlas contributes 151,170 nine-mers on them. Crucially, the peptide universes are nearly disjoint: only **140** allele–peptide pairs appear in both. That is too few to be useful as training data, but it is exactly what is needed as a *test* — an independent sample of peptides that immunopeptidomics flagged as presented, for which we also hold a measured dissociation half-life.

If the survival argument holds, those 140 should sit high in the half-life distribution relative to the 20,230 measured peptides on the same alleles that the atlas never observed.

## Test

Join on (allele, peptide). Restrict to the 54 shared alleles so allele composition cannot drive the comparison. Compare `thalf_hours` for eluted versus non-eluted peptides with a one-sided Mann–Whitney U test — rank-based, because 21% of the half-lives are censored at zero and the distribution spans four orders of magnitude.

Supporting checks: a per-allele direction test (is the effect general or carried by HLA-A\*02:01?), the common-language effect size, and an inspection of every counterexample.

## Result

| | n | median t½ | mean t½ | frac t½ = 0 | frac ≥ 1 h |
|---|---|---|---|---|---|
| **eluted in atlas** | 140 | **7.90 h** | 12.99 h | **0.021** | **0.921** |
| not in atlas | 20,230 | 1.10 h | 5.36 h | 0.212 | 0.520 |

Mann–Whitney U = 2,210,237, **p = 6.0 × 10⁻³¹** (one-sided, eluted greater). Common-language effect size **0.78** — a randomly chosen eluted ligand outlasts a randomly chosen non-eluted peptide 78% of the time.

The effect is general, not an artefact of one well-studied allele: among the 18 alleles with at least 3 measured eluted ligands, the eluted median exceeds the non-eluted median in **14**. It is visible in the largest ones individually — HLA-A\*02:01 17.50 h versus 3.80 h (n = 31), HLA-A\*01:01 15.00 h versus 1.30 h (n = 9), HLA-B\*27:05 5.20 h versus 0.80 h (n = 5).

The sharpest number is the censored fraction: being eluted drops the probability of an undetectable half-life from 21.2% to **2.1%**, a tenfold reduction.

### Counterexamples

Six eluted ligands have a measured half-life below 0.5 h, three of them exactly zero:

| Allele | Peptide | t½ (h) |
|---|---|---|
| HLA-B\*51:01 | FPEHIFPAL | 0.0 |
| HLA-B\*51:01 | SPSSPGSSL | 0.0 |
| HLA-B\*08:01 | FPEHIFPAL | 0.0 |
| HLA-B\*08:01 | FMKPGKVVL | 0.4 |
| HLA-B\*39:01 | YHEDIHTYL | 0.4 |
| HLA-B\*08:01 | DAYRRIHSL | 0.4 |

`FPEHIFPAL` appears twice, on two different alleles, with a measured half-life of zero on both. That pattern is more consistent with motif-deconvolution error — the peptide assigned to alleles it does not actually bind — than with two independent instances of unstable presentation. The remainder are plausible cases where cellular loading machinery (tapasin, chaperones, local peptide concentration) sustains a complex that falls apart in a cell-free dissociation assay.

## Conclusion

**The inference is correct and quantified: elution implies high stability, probabilistically.** A peptide in the atlas is ~7× higher in median half-life and ten times less likely to be undetectable. But it is not deterministic — roughly 2% of eluted ligands have zero measured stability, and at least one of those is likely a mislabelled allele assignment rather than biology.

**This does not justify augmenting the training data, for four reasons.**

1. **Scale mismatch.** 151,170 atlas 9-mers against 28,166 stability measurements. An auxiliary task 5.4× the size of the target task would dominate training; the result would be an antigen-presentation model with a stability side-effect.
2. **The threshold is unknowable.** The honest encoding of an eluted ligand is "t½ > τ" for some protocol-dependent τ. Nothing in the data determines τ, and conclusions would move with whatever value is chosen.
3. **It is a selection effect, not a measurement.** Elution is confounded with source-protein abundance, proteasomal cleavage, TAP transport and MS ionisation efficiency. Stability is one of several filters a peptide passed, and the dataset cannot separate them.
4. **It changes the question.** Adding auxiliary data benefits a small BLOSUM network and a frozen protein language model to different degrees. Any difference measured afterwards is partly about which architecture absorbs mass-spec data, not about whether foundation models encode stability.

**The high-value use is external validation.** Because only 140 of 151,170 pairs overlap, the atlas is an almost completely independent peptide set on the same alleles. Train on stability alone, then score atlas ligands against length- and allele-matched proteome decoys. If the model ranks true ligands above decoys, that demonstrates transfer to a different assay measuring a different biological event — a stronger claim than any within-dataset correlation, at the cost of a single scoring pass with no retraining and no new assumptions.

If elution data is ever used for training, the established pattern is NetMHCpan-4.x: binding affinity and eluted ligands trained jointly with **separate output heads**, not merged into one target. And in either case the peptide-similarity grouping must span both datasets, or an atlas peptide one substitution from a held-out stability peptide walks straight through the split.
