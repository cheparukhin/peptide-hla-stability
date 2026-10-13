# Few-shot unseen-HLA residual adaptation

With 20 target-allele labels, R5 ranking gain versus 0 labels is **+0.046754 [+0.022657, +0.103467]** on the fixed ≥20 panel.

Three frozen support draws are averaged as scores. Target support and query peptide families are excluded from source fitting. The same query cohort is used at every budget within each draw. All source stopping labels count; no target labels are used for tuning or stopping.

The sequence anchor is a fresh four-member historical two-layer recipe using both one-hot and BLOSUM encodings; it is a reduced ensemble and is not the unchanged 30-member reference or competitive selector. Source ridge heads and residual adapters are fixed. These are cached-feature development comparisons, not neural fine-tuning or prospective assay savings.

Crossed, peptide-only and HLA-only intervals use 2,000 shared full-population draws, conditional on fitted models and the three frozen support designs; they omit refitting, selection history and independent assay variability. Secondary intervals are nominal and unadjusted. All 35 actual HLA-group deletions are retained. See absolute_curves.csv, comparisons.csv, actual_budgets.csv, cell_inventory.csv and audit.json.
