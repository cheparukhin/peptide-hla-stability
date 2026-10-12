# Bounded transfer controls — finalisation follow-up

This is a new, separately authorized development extension of the closed round-two study. Existing sources, grouping, protocols, predictions and accounting remain immutable. Original test outcomes are excluded. No fits or cloud launches are authorized by this document or a source freeze alone.

Reuse all 19,716 original training rows and the existing five HLA-group cells plus all 25 joint peptide/HLA cells. Preserve all 35 actual HLA groups, all frozen outcomes and all baseline fallbacks. Inner selection uses three balanced HLA folds, or every joint 3×3 intersection. Joint gradient and stopping rows exclude both held groups; diagonal stopping leaves cross quadrants unused. No cell can be dropped because its result or runtime is inconvenient.

| New procedure | Frozen candidates | Members per inner call / selected final call | Additional fits |
|---|---|---|---:|
| Log-target RMSE tree | Depth 4, 8 | 3 stops × 2 seeds / same 6 | 3,060 |
| Centred-target RMSE tree | Depth 4, 8; shrinkage k20 | 3 stops × 2 seeds / same 6 | 3,060 |
| Matched R0 ridge | Alpha 10,000, 100, 1 | 1 / 1 | 750 |
| Matched R0 hidden64 MLP | L2 1, .01, .0001 | Stop0/seed0 and stop1/seed1 / 5 stops × 6 seeds | 2,340 |
| Fixed sequence PairLogit | One depth8 candidate | 3 stops × 2 seeds / same 6 | 1,620 |
| **Total** | | | **10,830** |

Trees and the fixed ranker use positional peptide plus HLA pseudo-sequence onehot, learning rate .06, L2 10, maximum 2,000 iterations, stopping patience100, Bernoulli .8 subsampling, border count128 and two numerical threads. Ranker pairs stay within allele; inverse pair-count allele weights follow the existing frozen implementation. The ranker has no grid. Centred tree means use each member's gradient labels only; unseen alleles restore the gradient global mean.

R0 ridge and MLP use the same regularization grids, StandardScaler fit scope, hidden64 MLP, training/stopping recipe and pooled selection rule as the reused R4 ridge and R5 MLP transfer heads. Save gradient scaler mean, scale and variance. Report direct R4−R0 ridge and R5−R0 MLP contrasts. The prior R4/R5 and embedding-selector predictions retain their original identities and selection rules.

Reuse original and centred MLP inner scores and selected final predictions after complete legacy protocol, hash, axes, ancestry and recipe audits. Legacy raw inner predictions were not retained; they are neither reconstructed nor claimed as newly verified numerical predictions. Every new inner prediction and every new individual member prediction is saved with its row identity.

The expanded sequence selector uses the fixed family order original, centred MLP, log-target tree, centred-target tree, fixed ranker8. Within-family candidate ties use depth4 before8 and strongest regularization first. The pooled inner median per-allele Spearman uses the outer-fit eligibility panel at min20; argmax resolves exact ties in frozen order. Original remains the first fallback. R0 matched head controls remain outside the expanded sequence selector. The embedding selector is reused unchanged.

The ranker inner endpoint is scored on **raw** clean fixed-candidate inner OOF predictions. One positive affine map (slope floor1e-8) is learned from those predictions and outer-fit labels, then applied to the final outer-fit refit's raw predictions. Mapped inner values are saved as scale-transport diagnostics; they are not independently calibrated inner predictions. This transports score scale from smaller inner training sets to larger final training sets. It differs from the earlier nested ranker procedure, and does not establish independent calibration accuracy.

Before reporting, reconstruct every member, stopping partition, scaler, centring mean, raw inner score, affine map, selected final, selector and reused identity. Member predictions are compressed in a pickle-free NPZ; means must replay the saved raw inner and final aggregate predictions. Publish 2,000 paired peptide-cluster, HLA-group and crossed uncertainty draws for both min20 and min50 panels. Delete each of the 35 actual HLA groups in turn, publishing all deletions. These uncertainties are conditional on fitted predictions; crossed resampling is a sensitivity analysis and may be conservative.

One root process owns the run, one root-owned Modal app context serves all paid calls, and at most three local cell threads each request six remote members in a batch. Numerical thread limits are held once around the entire local controller. Paid peak concurrency and provider hard cap are18. The separate ranking controller must have a hash-bound successful exit receipt and its process must be absent before transfer begins. Ownership files survive failures and refuse automatic retries. A separately bounded pilot, fresh rates, source hashes, a fixed spending deadline, independent watchdog and worst-case reservations are required before paid dispatch. Every new batch must verify the independent guard. No automatic expansion or extension is allowed.
