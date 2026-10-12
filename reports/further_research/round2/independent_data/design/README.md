# Independent confirmation: bounded synthetic panel design

Recorded 12 October 2026. Implements the simulation requirement in the [canonical research plan §§8–9](../../../../NEXT_RESEARCH_PLAN.md#8-statistical-reporting-rules), following the [independent-data audit](../README.md). **Design feasibility is conditional; independent confirmation remains open.** This work opened no external outcome labels, fitted no biological predictor, used no cloud service, and sent no external communication.

## Decision

Request comparable assay quotes for **12 alleles × 200, 400 and 800 distinct peptide–allele measurements** (2,400 / 4,800 / 9,600 primary measurements), with independent preparation/measurement repeats for a stratified 20% subset quoted separately. The 12×400 option is the strongest bounded starting design for detecting a true 0.05 improvement under the tested assumptions; 12×200 is a lower-volume alternative with weaker evidence under strong dependence. A true 0.02 gain remains difficult: even 12×800 has only 47% detection in the strong-dependence bootstrap diagnostic. These are acquisition options, not a funded or validated assay programme.

All 12 biological HLA assignments remain pending sequence/construct/provenance review. Include poorly covered alleles and several sequence groups; select peptide-family and lifetime-proxy strata before outcomes. Do not spend the entire budget on a few well-covered alleles, or make disagreement-selected samples the sole confirmation population. The conservative public candidate set has 129 rows with only one allele above 20 rows; this simulation does not turn that coverage into representative confirmation.

## Endpoint, generator and inference

The endpoint is **median(rho_candidate by eligible allele) − median(rho_reference by the same eligible allele)**, not the median of per-allele deltas and not a pooled or unpaired correlation. Labels/counts fix eligibility at ≥20 rows and two distinct labels. The ≥50-row sensitivity is reported. Spearman uses average ranks for ties; finite constant predictions contribute zero, label-degenerate alleles leave both panels together, and invalid values are rejected, matching `pepstab/evaluation.py`. The positive-improvement decision requires the lower bound of a two-sided 95% interval to exceed zero. A worthwhile-gain claim requires its lower bound to exceed 0.05.

Synthetic marginal assumptions: baseline population Spearman 0.66; 20% floor ties; latent normal signal with independent marginal measurement-error SD 0.4; paired prediction-residual correlation 0.8, with 0.5 as a sensitivity. These are planning assumptions, not fitted assay parameters. Labels are nonnegative floor-censored pseudo-log1p values; synthetic predictions are ranking scores. Every allele has the same population skill and target gain. Signal, measurement error and prediction residual fields each combine a shared peptide component, a shared peptide×HLA-group component, and a row-specific component. Groups contain four synthetic allele slots. The variance fractions are weak (0, 0), moderate (0.25, 0.20), or strong (0.60, 0.25). The additional pairing sensitivity uses moderate dependence and residual correlation 0.5.

Peptide pools are fully crossed across alleles; a sparse allele observes the initial subset of the same pool. A cluster contains one row per allele. Thus cluster count equals the largest per-allele count, not total rows. This is a transparent reuse/dependence stress test. Real family clusters, partial overlap, heterogeneous allele skills/gains, batch effects, assay shifts and alternative floor/noise rates may change sensitivity substantially. Do not infer a recoverable headroom or empirical noise ceiling from this generator.

Loadings are calibrated on 160,000 independent synthetic rows to target gains 0, 0.02 and 0.05, then checked on a separate 160,000-row sample. Baseline validation rho is 0.6605; gains for pairing 0.8 are approximately 0.0198 and 0.0496, and for pairing 0.5 are 0.0201 and 0.0499. Null loadings are identical, so population gain is exactly zero by symmetry; finite validation null differences of approximately −0.0016/−0.0021 are sampling variation. See [population_calibration.csv](population_calibration.csv).

The bounded run used 108 scenarios, 1,000 calibration and 500 independent evaluation panels per scenario, seed 20261012, and completed locally in 423.27 seconds. Calibration panels estimate the full paired endpoint-error distribution; its 2.5%/97.5% quantiles invert to a planning interval on each separate evaluation panel. This preserves simulated peptide/HLA dependence, paired predictions, median aggregation, finite-sample bias and ties without assuming independent allele tests or normal Fisher errors. **It knows the assumed data-generating process and target gain, so it is an oracle planning approximation, not the actual confirmation analysis or established bootstrap power.**

The exact paired whole-peptide-cluster percentile bootstrap is evaluated on 30 of the 500 evaluation panels per scenario (3,240 datasets, 399 resamples each). Integer cluster multiplicities give exact average ranks of duplicated rows; both models use the same draws. Tests compare this faster implementation with the frozen scoring and bootstrap contract. This limited diagnostic avoids running a full bootstrap on every synthetic panel. The actual one-time confirmation retains the prespecified **2,000 paired cluster resamples**; the 399-resample planning diagnostic does not amend that rule.

## Actual percentile-bootstrap sensitivity diagnostic

Each percentage below is a simulated positive-lower-bound decision rate over **30 independent datasets**, followed by its Wilson 95% Monte Carlo interval. The intervals describe finite simulation uncertainty conditional on these synthetic assumptions; they are not confidence bounds on performance in a real assay. The small diagnostic cannot prove real-world nominal coverage or lock an unconditional 80% power guarantee.

| Panel | Dependence / pairing | Detect true gain 0.02 | Detect true gain 0.05 |
|---|---|---:|---:|
| balanced_12x200 | weak | 20% [10%, 37%] | 100% [89%, 100%] |
| balanced_12x200 | moderate | 13% [5%, 30%] | 97% [83%, 99%] |
| balanced_12x200 | strong | 30% [17%, 48%] | 83% [66%, 93%] |
| balanced_12x200 | moderate_pair05 | 17% [7%, 34%] | 90% [74%, 97%] |
| balanced_12x400 | weak | 60% [42%, 75%] | 100% [89%, 100%] |
| balanced_12x400 | moderate | 57% [39%, 73%] | 100% [89%, 100%] |
| balanced_12x400 | strong | 27% [14%, 44%] | 100% [89%, 100%] |
| balanced_12x400 | moderate_pair05 | 20% [10%, 37%] | 100% [89%, 100%] |
| balanced_12x800 | weak | 97% [83%, 99%] | 100% [89%, 100%] |
| balanced_12x800 | moderate | 90% [74%, 97%] | 100% [89%, 100%] |
| balanced_12x800 | strong | 47% [30%, 64%] | 100% [89%, 100%] |
| balanced_12x800 | moderate_pair05 | 60% [42%, 75%] | 100% [89%, 100%] |
| balanced_24x200 | weak | 50% [33%, 67%] | 100% [89%, 100%] |
| balanced_24x200 | moderate | 43% [27%, 61%] | 100% [89%, 100%] |
| balanced_24x200 | strong | 23% [12%, 41%] | 90% [74%, 97%] |
| balanced_24x200 | moderate_pair05 | 23% [12%, 41%] | 100% [89%, 100%] |

The known-DGP approximation has higher precision but may be optimistic. For 12×800 and gain 0.02 its rates are 93.8% (weak), 86.6% (moderate), 62.0% (strong) and 62.0% (moderate/pairing 0.5); actual bootstrap diagnostic rates above are 97%, 90%, 47% and 60%. Mean exact bootstrap interval widths are 1.26×, 1.20×, 1.08× and 1.14× the planning widths respectively; diagnostic gain coverage is approximately 99.9%, 99.5%, 97.5% and 98.8%. This conservative behaviour under homogeneous synthetic allele effects is a sensitivity finding, not a reason to substitute the more favourable interval.

Across null scenarios, planning positive rates range from 0.8% to 4.4%, and planning interval coverage across all cells ranges from 91.4% to 97.2%. Exact-bootstrap null positive rates pooled across panels (270 diagnostic datasets per dependence case) are 0.0% weak, 0.37% moderate, 2.22% strong and 0.0% moderate/pairing 0.5. These checks detect gross calibration mistakes, but the pooled rates cannot validate every design or real DGP.

## Full planning comparison and allele coverage

The following rates are the known-DGP planning approximation over 500 independent panels, not actual bootstrap power. Use them to compare allocation and dependence; use the diagnostic table and its uncertainty when interpreting feasible confirmation sensitivity.

| Panel | Primary rows | Unique peptide clusters | Eligible alleles at 20 / 50 | Gain 0.02, moderate / strong | Gain 0.05, moderate / strong |
|---|---:|---:|---:|---:|---:|
| balanced_12x50 | 600 | 50 | 12 / 12 | 12.0% / 4.8% | 53.2% / 27.8% |
| balanced_12x100 | 1,200 | 100 | 12 / 12 | 17.2% / 9.6% | 75.8% / 45.6% |
| balanced_12x200 | 2,400 | 200 | 12 / 12 | 35.2% / 23.0% | 96.2% / 77.0% |
| balanced_12x400 | 4,800 | 400 | 12 / 12 | 59.2% / 32.2% | 100.0% / 97.8% |
| balanced_12x800 | 9,600 | 800 | 12 / 12 | 86.6% / 62.0% | 100.0% / 100.0% |
| balanced_24x200 | 4,800 | 200 | 24 / 24 | 54.0% / 19.6% | 100.0% / 87.4% |
| sparse_2400 | 2,400 | 390 | 6 / 6 | 29.4% / 29.0% | 99.6% / 96.2% |
| near_threshold_2400 | 2,400 | 370 | 12 / 6 | 24.4% / 14.2% | 91.6% / 62.4% |
| repaired_2400 | 2,400 | 350 | 12 / 12 | 22.8% / 14.8% | 94.4% / 77.8% |

At the same 4,800 primary-row budget, 24×200 increases HLA breadth but reuses only 200 clusters, versus 400 clusters for 12×400. Additional alleles do not replace independent peptide information under strong crossed dependence. This fixed-panel simulation does not quantify uncertainty over an unseen-HLA population; a separate HLA/joint-transfer endpoint needs actual sequence groups, adequate group coverage and its own crossed-group analysis.

The sparse 2,400-row panel has six alleles at 10 rows and six at 390: its apparently good 0.05 rate refers to only six eligible alleles. The near-threshold allocation has six at 30 and six at 370: all 12 enter at 20 rows, but the 50-row sensitivity excludes six. Any narrower interval after dropping them changes the population being summarized. The repaired allocation puts six at 50 and six at 350, keeping all 12, but equal total rows still do not match balanced coverage. Freeze both eligibility panels before outcomes; never choose the favourable threshold.

[simulation_summary.csv](simulation_summary.csv) retains every declared cell, sampling mean/bias/SD, separate 50-row results, null behaviour, positive decisions, above-0.05 decisions and interval coverage. Its Wilson intervals condition on estimated calibration quantiles. The additional joint Monte Carlo intervals resample both independent simulation ensembles (400 resamples), capturing finite quantile estimation as well as detection-count uncertainty. These are approximate finite-MC intervals; boundary rates can give degenerate resampling intervals, so the Wilson intervals are retained. No rate of 100% means guaranteed detection.

**Detecting improvement and establishing the worthwhile bar are different.** At a true gain of exactly 0.05, a lower bound above 0.05 should be rare for a calibrated interval. High sensitivity to exclude zero cannot support a claim that the gain exceeds 0.05. Failure to exclude zero also does not establish equivalence.

## Acquisition and metadata requirements

The [panel allocations](panel_allocations.csv) are synthetic slots, not an eligible biological row manifest. Before choosing identities, require the following:

| Requirement | Required evidence / delivery |
|---|---|
| Identity and independent provenance | Exact peptide sequence/length/modifications, physical HLA sequence and allele, engineered changes, β2m construct, source/study/sample identifiers, source proteins/families, new measurement versus republished measurement lineage, acquisition/redistribution rights |
| Exposure exclusions | Exact peptide–allele and peptide matches; Hamming-distance-3 component overlap against all development data and available supervised donors; excluded coverage and overlap strata; separate structural-pretraining exposure; unknown checkpoint provenance remains unknown; audit other lengths separately |
| Kinetic endpoint and protocol | Actual kinetic half-life and units, temperature (prefer protocol-matched 37°C for continuity), construct/loading procedure, assay/readout and version, concentrations/time points/fit model, raw time courses and fitting diagnostics; affinity, melting temperature, elution and denaturant resistance are distinct endpoints |
| Floors and failed measurements | Detection/quantification floor, censoring direction and flags, upper limits, failed-binding/QC/drop rules, per-measurement uncertainty and calibration/controls; freeze exclusions before model outcomes |
| Replicate subset | Independently prepared and measured repeats for a stratified 20% subset, at least 20 pairs per allele in the quoted options; spread across peptide families and prespecified lifetime proxies, floor-adjacent samples, allele coverage, operators/batches/days; retain technical-versus-independent replicate identity and all raw replicates |
| Panel representativeness | Pre-outcome sampling frame across HLA groups, poorly covered alleles and peptide families; lifetime strata use development-derived proxies or frozen predictions, not reserved outcomes; representative core plus separately reported disagreement/acquisition arm |
| Reservation and freeze | Confirmation labels reserved before adaptation/calibration; at most two fully frozen ranking procedures and one calibration procedure; one primary comparison, other ranking comparison secondary or prespecified multiplicity handling; preprocessing/model/data/ancestry hashes, row/group manifest, exclusions, scoring and 2,000-resample analysis code locked before opening labels once |

Replicates can support assay repeatability and a revised noise/dependence model after a separate pilot; they do not automatically identify a universal irreducible ceiling. A protocol-development pilot used for design must be separate from the final reserved confirmation labels. Reassess actual panel coverage and rerun bounded design sensitivity after learning protocol-specific floor rates, clustering and repeatability; do not inspect final candidate errors to choose the design.

## Quote and execution gate

Obtain itemized quotes for 2,400, 4,800 and 9,600 unique peptide–allele kinetic measurements; request one additional independent preparation/measurement on the 20% replicate subset (480, 960 or 1,920 additional half-life estimates) as a separate line. These counts are not well counts: time points, concentration conditions, controls and failed assays may require many more wells. Specify peptide/HLA/β2m production and purity, loading/QC, protocol pilot/optimization, randomization/blinding and batch design, expected success/floor/censor rates, raw time-course and replicate delivery, ownership/access terms, failed-sample rerun policy, physical construct coverage and quote validity. Ask the provider for achievable throughput and delivery window; no amount, calendar date, partner commitment or quote has been invented.

**Gate still open:** actual source/partner availability, quote, biological panel identities, protocol metadata, replicate access, checkpoint exposure provenance and finalist freeze. If independent measurements cannot be obtained, publish development-only results and leave confirmation open. This local synthetic study completes a bounded design analysis, not that gate.

## Reproduction and validation

From the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/round2_confirmation_design.py
.venv/bin/python -m pytest tests/test_round2_confirmation_design.py -q
```

The default run regenerates population calibration, summary, diagnostics, compressed simulation draws and manifest. The narrative and panel allocation inventory are reviewed artifacts for these fixed defaults. Six focused tests pass: endpoint parity including ties/constants/eligibility; finite-input rejection; weighted average-rank equivalence; exact paired bootstrap parity under crossed sparse coverage; synthetic marginal calibration and dependence checks; and identical-model zero deltas on every paired draw. The immutable benchmark data/splits/predictions are untouched.

[manifest.json](manifest.json) records parameters, seeds, software versions, elapsed local runtime, script/endpoint hashes and the aggregate eligibility-audit hash. [bootstrap_diagnostics.csv](bootstrap_diagnostics.csv) preserves all 3,240 exact checks. [simulation_draws.csv.gz](simulation_draws.csv.gz) contains the 162,000 independent calibration/evaluation endpoint draws and planning bounds for inspection. No simulated outcome is represented as a biological observation.
