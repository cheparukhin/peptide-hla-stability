# Predicting peptide–HLA stability

**logic binders · London AI × Science · Protein Engineering Track**

To trigger a T-cell response, a peptide–HLA complex must persist at the cell surface long enough for immune recognition. Kinetic stability—how long an assembled complex remains intact—is distinct from equilibrium binding affinity. We ask whether internal features from the frozen Boltz-2 structure-prediction model improve the stability ranking of nine-residue peptides over sequence-only predictors.

On 19,716 development pairs, a nested Boltz-2 interface-feature procedure increased median within-allele Spearman correlation from **0.6668 to 0.7212** (**+0.0543**, conditional 95% interval **[+0.0314, +0.0808]**). This supports improved ranking in the development dataset, though the interval lower bound did not meet the predeclared +0.05 criterion and independent kinetic validation is still needed.

## Supporting development findings

Before generating 3D atomic coordinates, the frozen Boltz-2 model computes internal representations that capture pairwise relationships between peptide positions and residues lining the HLA binding groove. Evaluating these learned features across controlled development splits yielded three supporting findings:

- **Preserving position improves ranking.** Retaining position-resolved peptide–groove features from the frozen Boltz-2 trunk outperformed dimension-matched pooled controls, whereas the chosen crystal contact mask had no established benefit over shuffled contacts. This gain reflects useful positional information rather than a demonstrated structural mechanism.
- **Interface representations transfer across held-out HLA allele groups.** When groups of closely related HLA binding grooves are held out during development cross-validation, a matched nonlinear predictor trained on sequence inputs achieves a median within-allele Spearman correlation of **0.1605**, compared with **0.5667** using Boltz-2 interface features. Under simultaneous holdout of both peptide clusters and HLA groups, correlation is **0.1242** for sequence inputs versus **0.4980** for interface features.
- **Few-shot adaptation improves linear readouts on held-out alleles.** In the joint peptide and HLA holdout regime, adapting a linear predictor on frozen Boltz-2 interface features with **20 measured peptides** from a held-out allele improves median within-allele correlation from **0.5296 to 0.5763**.

All comparisons above are development assessments on existing scintillation proximity assay measurements. They evaluate relative peptide ranking within alleles and do not establish physical mechanisms, calibrated dissociation half-lives, or performance on independent experimental assays.

## Repository guide

The repository contains source code, saved predictions, numerical result tables, and portable model weights. Large precomputed feature caches are stored externally to Git.

- [Scientific report](reports/REPORT.md): Complete scientific report, methods, interpretation, and source tables.
- [Project page](https://cheparukhin.github.io/peptide-hla-stability/) ([page source](site/index.html)): Visual presentation of the findings.
- [Evaluation protocol](EVALUATION.md) and [Dataset and assay](docs/DATASETS.md): Dataset provenance, split construction, and statistical analysis specification.
- [Structural representations](docs/STRUCTURES.md): Construction of Boltz-2 interface representations and structural baselines.
- [Cached-feature inference guide](docs/ROUND2_INFERENCE.md): Guide to running portable prediction heads on compatible cached Boltz-2 features.
- [Submission summary](reports/finalization/SUBMISSION_BRIEF.md) and [Independent assay design](reports/finalization/INDEPENDENT_ASSAY_BRIEF.md): Concise project summary and prospective experimental validation protocol.

## Quick verification

To verify the integrity of all saved full-cohort comparison tables and fixed-query evaluation records locally:

```bash
python3 site/round2-build.py --check-full-cohort
```

To run predictions on cached Boltz-2 trunk features using the released model weights, see [Cached-feature inference guide](docs/ROUND2_INFERENCE.md).
