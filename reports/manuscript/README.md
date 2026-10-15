# Manuscript preparation and figure sources

The [manuscript draft](../MANUSCRIPT.md) develops the paper's argument from the completed assessments. The [scientific report](../REPORT.md) remains the fuller study record, and the [evaluation protocol](../../EVALUATION.md) defines the analyses. This directory contains new presentations of saved results rather than new experiments or model fits.

## Contribution and positioning

The defensible contribution is an empirical evaluation of frozen Boltz-2 representations for assay-derived peptide–HLA stability ranking. It combines a prespecified sequence comparison, dimension-matched representation controls, supervised HLA-group and joint holdouts, and retrospective adaptation. The larger transfer effects do not replace the main comparison or its unmet lower-bound +0.05 criterion.

A focused prior-work check on 15 October 2026 informed the draft. This was not an exhaustive priority review:

| Prior work | Consequence for this manuscript |
|---|---|
| [NetMHCstab](https://pubmed.ncbi.nlm.nih.gov/23927693/) and [NetMHCstabpan](https://pubmed.ncbi.nlm.nih.gov/27402703/) | Stability prediction and allele-dependent anchor motifs are established. The paper does not claim to introduce either or to outperform every published predictor. |
| [AFToolkit](https://pubmed.ncbi.nlm.nih.gov/40622483/) | Frozen structure-model features paired with lightweight adapters are established prior art. Reusing such features is not a sufficient novelty claim. |
| [DeorphaNN](https://pubmed.ncbi.nlm.nih.gov/42492505/) | Pair representations have already been used in peptide–receptor prediction with experimental validation. Because our endpoint, controls, and evaluation population differ, its numerical results do not serve as a matched benchmark. |
| [Chauhan, 2026](https://arxiv.org/abs/2606.04228), preprint | The task dependence of structural-feature utility is already an explicit research question. The manuscript addresses a specific kinetic-assay setting rather than an untouched general question. |

The draft deliberately avoids claiming that positional resolution is universally sufficient, that pair features are universally necessary, or that the contact-mask comparison reveals a physical mechanism. The projected token arm is reported at its saved absolute score of 0.3862—below the sequence selector—despite its large gain over its matched pooled control. Whereas the token control pools the same underlying residues, the full pair interface retains all 34 groove residues while its pooled control uses only mask-selected contacts, changing both coverage and positional detail. The saved label `isolated_position_control` therefore overstates what this pair comparison isolates; historical protocols and numerical tables retain their original labels. This refines the interpretation without changing any fitted results.

## Figure provenance

| Figure | Saved evidence | Inspectable values |
|---|---|---|
| 1: main and transfer comparisons | [Main contrasts](../further_research/round2/results/main/paired_comparisons.csv), [transfer contrasts](../finalization/transfer_cloud_recovery/results/comparisons.csv) | [Exact selected rows and source hashes](figure_data.csv) |
| 2: representation controls | [Main contrasts](../further_research/round2/results/main/paired_comparisons.csv) | [Exact selected rows and source hashes](figure_data.csv) |
| 3: retrospective adaptation | [Canonical figure record](../followup3/figures/fewshot_curves.json), [absolute curves](../followup3/fewshot/absolute_curves.csv), [three-primary interval](../followup3/fewshot/primary_family3_interval.json) | [Curve values, interval and source hash](figure3_data.json) |

All estimates and bounds are copied directly from saved artifacts. The renderer verifies unique comparison rows, score-difference arithmetic, and adaptation source hashes; it does not refit models, regenerate bootstrap draws, or acquire measurements. Contrasts use unrounded source values, while displayed values are rounded to four decimal places. The 98.3333% three-primary interval supersedes the older 97.5% fields retained in the original few-shot comparison table.

The figures are supplied as SVG and PDF for vector export and as PNG for preview. With Python 3.11+ and `matplotlib==3.10.8` available, run from the repository root:

```bash
python3 reports/manuscript/build_figures.py
```

This is a lightweight rendering operation that leaves the statistical and computational scope of the underlying studies unchanged.

## Limits that remain part of the paper

The study reuses a single assay corpus with a research history that includes earlier diagnostic test exposure. Its intervals are conditioned on saved predictions and selection decisions. HLA-group exclusion applies to supervised stability fitting, not to an audited absence of related structures in foundation-model pretraining. The label-budget and adaptation sequence references are reduced recipes and must not be presented as the main competitive selector. Independent kinetic confirmation, calibrated physical half-lives, immunogenicity prediction, and prospective assay savings remain unestablished.

Before external submission, the authors must finalize authorship, affiliations, funding and conflict statements, data-use and licensing statements, repository release details, and venue formatting. The existing extraction-wrapper provenance limitation is disclosed in the manuscript's availability statement and the linked cache audit. No missing declaration should be filled with an invented assertion.
