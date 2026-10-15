# Learned interface representations improve peptide–HLA stability ranking

**logic binders · London AI × Science · Protein Engineering Track**

Peptide–HLA stability influences how long a peptide remains available for
immune recognition, yet a plausible bound structure does not by itself specify
a dissociation rate. We therefore tested whether frozen Boltz-2 representations
improve ranking of assay-derived stability over supervised sequence models.

Nested assessment of 19,716 development pairs across 75 HLA alleles or
constructs held related peptide sequences together and kept fitting, stopping,
and model selection within the appropriate partitions. On the 69 eligible
alleles, the embedding procedure increased median within-allele Spearman
correlation from **0.6668 to 0.7212**: **Δ +0.0543, conditional 95% interval
[+0.0314, +0.0808]**. Matched nonlinear comparisons also favoured embeddings
when HLA groups, or both peptide and HLA groups, were held out. Positional
features improved over pooled controls, whereas the specified contact map had
no established advantage over shuffled contacts.

Twenty measurements per target allele improved adaptation from **0.5296 to
0.5763**, with a three-primary 98.3333% interval for the gain of
**[+0.0140, +0.1195]**. By contrast, ProteinMPNN scores, repaired FoldX
descriptors, ranking loss, and hybrid prediction heads did not establish
further improvements under the tested settings, and a full-cohort ProteinMPNN
extension gave **−0.0070 [−0.0220, +0.0131]** beyond the interface model.

These findings support useful predictive information in learned interface
representations without isolating a structural mechanism, and the main interval
does not establish a gain exceeding the declared +0.05 threshold. Moreover, all
later studies reuse development data, an earlier diagnostic exposed part of the
original test set, and resampling intervals condition on saved models,
excluding retraining and independent assay variation. Because the assay tracks
β2-microglobulin dissociation as a proxy for complex stability—and replicate
observations and the measurement-level interpretation of zero labels are
unavailable in the supplied file—independent kinetic measurements are required
before claiming calibrated half-lives or practical assay savings.

The project provides inspectable predictions, source tables, and portable ridge
and neural heads for compatible cached features. The next scientific step is a
fixed-model comparison on independently measured complexes.

[Complete scientific report](../REPORT.md) ·
[Project page](https://cheparukhin.github.io/peptide-hla-stability/) ·
[Model guide](../../docs/ROUND2_INFERENCE.md) ·
[Independent assay design](INDEPENDENT_ASSAY_BRIEF.md)
