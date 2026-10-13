# Pre-results assessment and reporting audit

13 October 2026. **No blocking defect found in the reviewed assessment and
report implementations.** This was a read-only source/protocol review before
head fitting. No labels, reserved outcomes or new head results were inspected;
no fits or resamples were run. Existing frozen files were not changed.

Source verification found no mismatches across the inverse protocol's 17
source files or the energy v2 protocol's 25 source files. Protocol-file SHA256s:

- Inverse v2: `063c4aeaaa2685601e27e6aabb37f6a9e07fa2712274d276854635627296ac51`.
- Energy v2: `9f99bf603ffe9a7dd66cd39f5e7a3e197ad495a45f74f334d2eeb8a4865ca926`.

## Matched comparisons and isolation

Both studies retain the same 4,449-pair original-training panel and frozen
peptide outer folds. Inner selection uses three cluster folds confined to
outer fitting rows. MLP stopping splits are reconstructed within each supplied
fitting partition; stopping labels and gradient labels cannot include inner
or outer assessment clusters. Neural scaling uses gradient rows only. Ridge
scaling uses its supplied fitting rows only. Prediction inputs to the neural
and energy head interfaces have their targets removed.

The inverse study contains six arms, 600 neural members and 300 ridge fits.
Each neural candidate uses the same two stopping/seed members at inner and
final levels. Candidate order is strongest regularization first, with the first
maximum retained on exact score ties. Every arm has the same tuning and stopping
opportunities. Stored inner vectors and fitting identities permit score and
choice replay; complete output and ancestry checks precede reporting.

Energy adds only R0+E and R5+E: 200 neural members and 100 ridge fits. Reused
R0/R5 baselines are bound to the inverse protocol and base feature hashes;
their receipt hashes, held identities, ancestry, inner scores and choices are
validated before use. Predictions are copied using round-trip CSV parsing,
and the energy report requires exact equality to the reused vectors. The
energy MLP uses the same float32 FeatureHead recipe; its owned float64 ridge
pipeline matches the inverse ridge scaler, LSQR solver, tolerance and grid.
No baseline refitting or extra model search is introduced.

## Endpoint and uncertainty

Point estimates are medians of per-allele Spearman correlations over pooled
outer-held predictions on fixed eligible panels. Each reported increment is
the difference between two panel medians. The interaction subtracts the R0
increment from the R5 increment at both point and bootstrap levels. Eligibility
uses at least 20 measurements with varying labels; at least 50 is a sensitivity.
Constant predictions retain the frozen zero contribution and tied ranks use
average ranks. No per-fold score averaging replaces the pooled endpoint.

All arms in a study share the same 2,000 whole-peptide-cluster draws and seed.
The reporting helper correctly represents replicated rows with integer
weights, ranks within alleles and uses shared draws for paired differences.
Pointwise intervals use quantiles .025 and .975. The inverse frozen reporter
still emits the original two-primary .0125/.9875 interval: **those fields are
historical 97.5% intervals, not the current three-primary family intervals.**
The separate independent audit derives the required 1/120 and 119/120 quantiles
from the same saved vectors, without refitting or drawing new resamples.
Energy's reporter already uses these three-primary quantiles.

## Interpretation and release gate

Only MLP R5+P+C minus R5+P at minimum 20 is the inverse primary. R5+P+C minus
R5 is an important secondary, with R0 contrasts, interaction, ridge and minimum
50 retained as sensitivities. Only MLP R5+E minus R5 at minimum 20 is the energy
primary. Its R0 increment, interaction, ridge and minimum 50 are secondary.
Do not select whichever learner, panel or secondary comparison looks best.
The pre-efficacy three-question family remains intact if an extraction gate
fails. The separate offset diagnostic stays post-hoc and outside that family.

No result changes the historical main comparator, establishes kinetic or
structural causality, or supplies independent assay confirmation. Uncertainty
conditions on fitted heads and development-selected procedures. These heads
do not persist fitted weights; score/selection/ancestry replay must not be
called weight-level replay.

Before release, run the separate representation audit against complete saved
artifacts. It independently reconstructs inner/stopping partitions and full
fit counts, checks exact energy baseline reuse, recomputes rank-score and
selection arithmetic, verifies every declared contrast and derives the current
family intervals. Passing this source precheck does not substitute for that
completed-output audit or Root's resource/completeness admission.
