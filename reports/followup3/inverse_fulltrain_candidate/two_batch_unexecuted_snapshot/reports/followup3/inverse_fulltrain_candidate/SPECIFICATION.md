# Candidate full ORIGINAL-train native compatibility followup

Status: DRAFT, neither frozen nor admitted. The root waits for the final energy
audit before choosing at most one followup. No paid extraction, target-based
fitting, bootstrap or incumbent search is authorized by this draft.

The completed balanced study suggests uncertain neural complementarity; ridge
disagrees. This candidate asks whether full cohort coverage and fitting density
change the incremental signal. Both change together, so this design cannot
attribute any difference to density alone. It is selected development work
after subset outcome inspection, with reused training labels and overlapping
clusters. It is not independent replication, untouched validation, a new
tranche primary, kinetic evidence, or native pose validation.

Population is exactly the 19,716 ORIGINAL-train natural pairs across 75 alleles.
The existing full-training peptide-cluster outer folds are inherited unchanged.
No old validation/test or reserved external labels enter. Canonical training
targets use the existing source-bound cloud adapter. All matching uses
allele/peptide with verified metadata; positional row indexing is not a join.

Reuse all 4,449 complete native ProteinMPNN feature records byte-exactly after
verifying every JSON hash, extraction receipt, parser recovery receipt and
cluster/fold/complex/folder identity. The legacy 2,000-row QC scores are excluded.
The other 15,267 rows are deterministically SHA-ranked within alleles,
interleaved by rank and allele, then alternated into disjoint batches of
7,634 and 7,633. Each batch computes only its own new rows; immutable cached
records are read separately at collection. The jobs never share aggregate
writes. Existing new records are reused without recomputation; failed records
remain failures and cannot trigger automatic retries. No outcome-based changes
to coverage, cases, resources or scoring are allowed.

Use the same v_48_020 checkpoint SHA, finite native backbone, declared sequence
checks, no noise, 16 explicit common peptide decoding permutations seed0,
visible HLA/beta2m first, conditional C and peptide-only P contexts. Apply the
source-bound exact-unique-whole-peptide-chain selector in every spawned worker.
Zero/multiple exact whole-chain matches fail. All original parser and scoring
checks remain, with no frozen source file edited. Per-position9 P and C are
fitted; exact redundant totals/deltas remain diagnostics only.

Fit only R5, R5_P and R5_PC freshly on the enlarged fitting populations.
The main comparison is MLP R5_PC minus R5. Context R5_PC minus R5_P and
peptide-only R5_P minus R5 are secondary, with ridge analogues as sensitivity.
There are no R0, energy, blend, transfer or additional architecture arms.
Old subset predictions and inner choices cannot serve as expanded baselines.

Keep the existing two-member hidden64 MLP (stop0/seed0, stop1/seed1), maximum
200 epochs, patience15, batch256, learning rate.001 and L2 [1,.01,.0001].
Use three clean inner peptide-cluster folds; select highest pooled inner
median per-allele Spearman on the ≥20 panel, exact ties strongest penalty.
Every inner/final call uses FeatureHead final=False. Gradient/stopping/scaler
populations are separated. This gives (3×3×2+2)×5×3 = 300 neural fits.
Float64 StandardScaler/LSQR ridge uses [10000,100,1], tol1e-5, max_iter2000,
with (3×3+1)×5×3 = 150 fits. The reviewed Float64RidgeHead is reused.

Save complete held predictions, each candidate's inner OOF vector in recorded
pair-ID order, selected choices, ancestry rowsets, members, versions, runtime,
seeds and feature/source hashes. Independent auditing replays inner scores and
selection and rejects fit/stop/scaler overlap with assessed or outer clusters.
Model weights are not saved; no fitted-weight replay claim is made.

Report pooled outer-held median per-allele Spearman with ≥20 primary panel and
≥50 sensitivity. Use existing paired whole-cluster 2,000 bootstrap draws seed
20261003 and pointwise95% intervals for all three contrasts and both learners.
These intervals condition on fitted predictions; they omit followup/model
selection uncertainty and do not establish a frozen-tranche family success.
Report ridge disagreement and the prior balanced results without treating
overlapping subset/full results as independent evidence.

Additionally restrict the newly trained held predictions to the original
4,449 query identities, retaining the original fixed ≥20/≥50 eligible allele
panels. For both learners, compare exact saved old R5/R5_P/R5_PC predictions
with the new vectors at these same queries. Save absolute points, the three
net/context/peptide increments and old-to-new differences in
fixed_query_descriptive.csv. Bind the old prediction files, receipts and
independent audit as frozen inputs. No extra CI, bootstrap or model is added.
This removes changing query composition from that descriptive comparison;
source coverage, inner selection and stopping populations still change
together, so it is not a pure training-size causal effect.

The fixed saved practical reference is
reports/further_research/round2/results/main/matched_predictions.csv,
column embedding_selector. Its canonical identity columns cover all19,716
rows and match allele/cluster/inherited-fold metadata. Pair IDs are translated
through frozen ORIGINAL-train allele/peptide metadata before any natural-key
join. The draft binds the entire saved file hash while inspecting identity
columns only. This identifies a possible descriptive practical reference;
the candidate report does not fit/select an incumbent or assert matched
head budgets against that selector. Adding reference metrics requires an
explicit prelaunch report-scope decision, not new model selection.

Historical mean worker time13.548s/pair predicts3.59h with two independent
16CPU/32GiB Sandboxes, eight spawn workers×two threads each. The observed-job
throughput predicts3.66h; mean+25%+five-minute startup predicts4.57h. Propose
five-hour hard lifetime per job and $17 per-job admission ($34 extraction),
plus $6 bounded staging/heads/report allowance. These are projections from
completed telemetry and stored rates, not new measurements or authorization.
Root owns fresh ledger admission, source/volume validation, launches and
independent stop guards. Hard caps can leave incomplete records; preserve
valid partials and stop without automatic extension or complete-case analysis.

The CLI drafts panel/batches/cache/source/input/cohort/recipe provenance only.
A separate freeze command requires an explicit root decision bound to the
draft SHA and completed energy audit. Production commands refuse absent
frozen protocol. Frozen scope does not itself authorize a paid launch.

```
python scripts/followup3_inverse_fulltrain.py draft
# Only after root's energy-audit decision, source review and fresh admission:
python scripts/followup3_inverse_fulltrain.py freeze --decision ROOT_DECISION.json
python scripts/followup3_inverse_fulltrain.py extract-batch --batch 0 --workers 8
python scripts/followup3_inverse_fulltrain.py extract-batch --batch 1 --workers 8
python scripts/followup3_inverse_fulltrain.py collect
python scripts/followup3_inverse_fulltrain.py stage
python scripts/followup3_inverse_fulltrain.py assess --learner mlp --outer-fold 0
# Same for mlp/ridge and folds0..4, then report.
python scripts/followup3_inverse_fulltrain.py report
```
