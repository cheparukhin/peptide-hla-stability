# Predicting peptide–HLA stability

**logic binders · London AI × Science · Protein Engineering Track**

We ask whether pretrained protein representations improve the ranking of
nine-residue peptides by the stability of their complex with an HLA molecule.

On 19,716 development pairs, a nested Boltz interface-embedding procedure
increased median within-allele Spearman correlation from **0.6668 to 0.7212**:
**+0.0543, conditional 95% interval [+0.0314, +0.0808]**. The result supports
improved ranking in this dataset. It does not establish a gain above the
predeclared +0.05 threshold or independent experimental confirmation.

- [Scientific report](reports/REPORT.md): complete results, methods, interpretation and source tables.
- [Project page](https://cheparukhin.github.io/peptide-hla-stability/): visual presentation of the findings.
- [Submission abstract](reports/finalization/SUBMISSION_BRIEF.md): concise project description.
- [Dataset](docs/DATASETS.md) and [evaluation specification](EVALUATION.md): measurement provenance, split rules and statistical analysis.
- [Model guide](docs/ROUND2_INFERENCE.md): portable prediction heads for compatible cached Boltz features.
- [Independent assay design](reports/finalization/INDEPENDENT_ASSAY_BRIEF.md): the experimental evidence needed next.

The repository includes source code, saved predictions, numerical result tables
and portable model weights. Large feature caches are external to Git. The
models produce research predictions; calibrated half-lives and performance on
independent measurements remain unestablished.
