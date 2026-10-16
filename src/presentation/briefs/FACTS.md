# logic binders — design facts, 16 October 2026

This is a public computational immunology research project about peptide–HLA stability ranking. HLA proteins display peptide fragments for immune recognition. The project reuses frozen internal residue-pair features from the Boltz-2 structure-prediction model and trains small predictors to rank existing stability measurements within each HLA type (allele). We are commissioning scientific communication and frontend design for these completed retrospective results.

Primary result: 19,716 development pairs, 3,587 peptide clusters, 69 eligible HLA alleles (75 labels in the cohort). Nested cross-validation separates model selection and evaluation, with related peptides held out together. Median within-allele Spearman correlation rises from 0.6668 (selected sequence models) to 0.7212 (selected Boltz-feature models). Paired difference +0.0543; conditional 95% interval [+0.0314,+0.0808]. Scores improve for 60 alleles and decrease for 9. Higher correlation means better ranking agreement, not longer biological lifetime.

Separate transfer study, matched neural predictors: holding out related HLA groups gives 0.1605 sequence / 0.5667 Boltz; holding out both HLA and peptide groups gives 0.1242 / 0.4980. 69 alleles. This controls supervised training overlap, not novelty relative to Boltz pretraining.

Separate few-shot study: using 20 existing measurements per target allele gives 0.5296 → 0.5763 for a Boltz-feature ridge model, 68 alleles, averaged over three fixed support designs. Gain +0.0468; conditional family-adjusted 98.33% interval [+0.0140,+0.1195]. Its model and query panel differ from transfer, so do not connect their scores as a single learning curve.

Secondary sequence-only ligand-presentation benchmark: 81,600 observed pairs, 51 alleles, AUROC 0.9656 against human-proteome decoys and 0.9157 against other-allele ligand decoys. Stability-dataset peptides excluded. Decoys are comparison peptides, not confirmed nonbinders. This is not a Boltz result or kinetic validation.

Scope: retrospective development evidence from one existing assay cohort; ranking, not calibrated physical half-life or immunogenicity. Independent kinetic validation remains needed. Pair features beat tested summary representations, but a specific physical contact mechanism is not established. Added inverse-folding/energy descriptors and ranking/hybrid learners have no established extra benefit (not equivalence).

Brand: logic binders · London AI × Science · Protein Engineering Track.
Canonical site: https://cheparukhin.github.io/peptide-hla-stability/
Manuscript: https://github.com/cheparukhin/peptide-hla-stability/blob/main/reports/MANUSCRIPT.md
Repository: https://github.com/cheparukhin/peptide-hla-stability

Design scope: work only on supplied presentation files; keep all biological context and qualifications honest. Do not infer new science, generate sequences, run predictors, or change evidence. You have no research repository access because this is a bounded design handoff. If information is missing, say so. No external dependencies, asset downloads or paid calls are needed. Rendered QA and integration will be done by the parent after your output.
