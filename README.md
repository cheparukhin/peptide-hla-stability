# Predicting peptide–HLA stability

**logic binders · London AI × Science · Protein Engineering Track**

Can pretrained protein representations improve predictions of how long a
9-residue peptide remains bound to an HLA molecule? Our completed development
study finds that **Boltz interface embeddings improve ranking over a selected
sequence baseline**, while simple geometry features did not help in the earlier
hackathon test. These are different representations and different assessments.

The current development gain is **+0.054 median per-allele Spearman**
(95% paired interval **+0.031 to +0.081**). This supports improvement on the
examined development cohort; it does not establish that the gain exceeds the
predeclared worthwhile threshold of +0.05. Independent assay confirmation remains
open. The original test has not been reopened.

## Explore the project

- [Current research report](reports/further_research/round2/REPORT.md): complete
  main, transfer, complementarity, reproducibility and resource findings.
- [Project page](site/index.html): locally updated evidence and a clearly dated
  archive of the original film/test. Serve with `python3 -m http.server 8000 --directory site` and open `http://localhost:8000`. This branch has not been
  deployed to the public site.
- [Portable model guide](docs/ROUND2_INFERENCE.md): use the trained ridge and
  neural heads with verified, compatible cached Boltz features.
- [Canonical research plan](reports/NEXT_RESEARCH_PLAN.md) and
  [finalisation status](reports/finalization/execution_manifest.json): frozen
  decisions, remaining work and spending controls.
- [Independent assay brief](reports/finalization/INDEPENDENT_ASSAY_BRIEF.md):
  concrete, unsent acquisition requirements and remaining confirmation gaps.
- [Original 4 October report](reports/REPORT.md): preserved historical study,
  including its single held-out test pass.

## Completed development results

All rows below use only the **original training partition**: 19,716 pairs,
3,587 peptide clusters and 75 alleles. The main eligible panel contains 69
alleles with at least 20 measurements. Scores are medians of per-allele
Spearman correlations; differences are between those medians.

| Assessment | Embedding procedure | Sequence comparator | Difference [95% interval] |
|---|---:|---:|---:|
| Main peptide-cluster assessment | 0.7212 | 0.6668 | +0.0543 [+0.0314, +0.0808] |
| HLA-group holdout | 0.5463 | 0.1740 | +0.3723 [+0.2903, +0.4533] |
| Joint peptide/HLA holdout | 0.5067 | 0.1108 | +0.3958 [+0.2951, +0.4457] |

The main comparator selects among five sequence procedures inside fitting
partitions. The completed transfer comparison uses a narrower original/centred
neural comparator; stronger transfer controls are a **new, unfinished follow-up**.
The main interval resamples paired peptide clusters; the transfer intervals
shown use crossed peptide/HLA-group uncertainty. Selection and stopping obey
the corresponding held-out grouping. Reported results remain conditional on
this examined development cohort and frozen candidate sets.

Position-aware interface representations substantially outperform global
pooling in matched neural comparisons. The designated contact-map representation
has no established advantage over its shuffled-contact control. These controls
support useful positional information, not a demonstrated causal structural
mechanism. Exact execution repeats matched on the recorded platform; they do
not measure assay reproducibility or independent model-selection reproducibility.

## Use the trained heads

Two locked demonstration recipes are included under `models/finalization/`:
R4 ridge with alpha 100 and a 30-member R5 MLP with L2 0.01. They were refit on
original training rows only and are not designated universal winners. Portable
NPZ/JSON artifacts contain weights, scalers and provenance; executable pickle
objects are not used.

```sh
.venv/bin/python scripts/round2_inference.py inspect \
  --model models/finalization/R5_mlp_0.01
```

Prediction requires a compatible verified compact feature bundle; large caches
are external to Git. Follow the [input contract and prediction example](docs/ROUND2_INFERENCE.md).
This is cached-feature inference, not a sequence-to-prediction service.
Warm single-row cached-head medians were 1.56 ms for ridge and 2.40 ms for the
MLP on the recorded host. These measurements exclude Boltz extraction, MSA,
cache loading and startup, and are not end-to-end new-sequence costs.

## Reproduce and inspect

The environment and experiment-specific commands are documented in
[the completed study](reports/further_research/round2/REPORT.md),
[operations](reports/further_research/round2/OPERATIONS.md), and individual stage
reports. Source, cohort, feature and completion hashes bind the saved evidence.
Completed expensive scopes must not be blindly relaunched.

```sh
# Check the portable head implementation without cloud work.
.venv/bin/python -m pytest -q tests/test_round2_inference.py

# Verify that the project page still matches its audited sources.
python3 site/round2-build.py --check
```

The inherited extraction-wrapper hash discrepancy is disclosed in the report:
scientific source and feature audits passed, but exact launch-wrapper bytes
cannot be certified. No new result repairs that historical provenance gap.

## Work currently being finalised

The newly authorized follow-up tests **Boltz embeddings plus a within-allele
ranking loss**, with matched sequence inputs and MSE controls. A separate
bounded extension adds stronger sequence transfer comparators and matched
learners. Their protocols and resource gates are being prepared; no outcomes
are claimed here before completion. The model package and current report page
are ready, while the final combined submission readout will include every
completed follow-up and any remaining limitations.

The completed round-two phase conservatively accounted for $76.84 of its $100
ceiling, including a retained engineering allowance. Its cloud jobs are stopped.
The user separately authorized at most $100 additional against a fresh reported
$143 balance. The new phase reuses existing embeddings and requires reservations,
a bounded pilot and an independent deadline watchdog before paid execution.
These figures are operational accounting, not a provider invoice or live balance.

## Scope

Stability is a kinetic endpoint; affinity, thermal melting and structural
plausibility cannot silently substitute for dissociation half-life. The public
candidate cohort does not yet provide representative independent confirmation,
and available aggregate labels cannot establish an assay-noise ceiling.
This project provides development evidence and reproducible research artifacts,
not a validated clinical or vaccine-design system.
