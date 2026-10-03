# Stage 3: frozen ESM-2 features

Does a pretrained protein language model add anything a supervised sequence
baseline does not already have? Scored through `pepstab.evaluation` on the frozen
validation split, against the stage 2 30-network ensemble.

Regenerate:

```bash
# weights are not committed; a Hugging Face id works in place of a local path
python scripts/esm_features.py --model facebook/esm2_t12_35M_UR50D --out features/esm2_35M.npz
python scripts/stage3_esm.py --features features/esm2_35M.npz        # 3a, separate embeddings
python scripts/stage3_coupling.py --features features/esm2_35M.npz   # 3b, coupling ablation
python scripts/stage3_blosum_cross.py                                # 3c, cross-feature control
python scripts/plot_stage3.py
```

`preds/seq_ensemble_pep_pseudo.csv` is gitignored, so **run
`scripts/baseline_ensemble.py` first** — every delta below is measured against
it. Regenerating it here reproduced stage 2 within rounding: median per-allele
rho **0.6904** (reported 0.693), mean SCC 0.649 (0.645), ensembling gain +0.093
(+0.090). 19.6 min on this machine against the 72.6 s recorded in
`reports/stage2_ensemble_pep_pseudo.csv`, which is a machine difference, not a
code one.

## 3a. Separate embeddings, concatenated — done

ESM-2 35M (`esm2_t12_35M_UR50D`), peptide and HLA embedded separately as stage 3
specifies, so the head has to learn the interaction itself. Per-residue states
kept from the middle layer (6) and the final layer (12); peptide is 9 x 480, HLA
is the 34 contact positions x 480.

Residue indexing was verified before any slice: the published NetMHCpan
pseudosequence positions land at index p-1 of the supplied 182-residue domain,
exact on all 75 alleles x 34 columns. Recovering the mapping from the data alone
is underdetermined — with 75 alleles, several conserved domain columns match any
given pseudosequence column by chance — so the published list is tested rather
than inferred.

Both blocks are PCA-reduced on training peptides only, no labels: peptide to 256
components (90.1% of variance at layer 6, 95.9% at layer 12), HLA to 74. The HLA
reduction is lossless — there are only 75 unique domains, so the block has rank
<= 75.

Ensembling matches stage 2 exactly: 5 CV folds x 2 layers x 3 seeds = 30
networks, folds cut along whole Hamming <= 3 peptide clusters, the stage 2
selected architecture. The two ESM layers play the role the two encodings play
in the baseline, so neither arm gets a bigger ensemble than the other.

`reports/stage3_summary_esm2_35M.csv`:

| model | median per-allele rho | IQR | MAE log1p | P@10 | pooled rho | delta vs baseline | 95% CI | verdict |
|---|---:|---|---:|---:|---:|---:|---|---|
| sequence ensemble (stage 2) | **0.6904** | 0.565–0.759 | 0.473 | 0.70 | 0.801 | — | — | — |
| ESM alone, 30 nets | 0.3866 | 0.231–0.536 | 0.645 | 0.60 | 0.656 | −0.3038 | [−0.359, −0.220] | worse |
| ESM stacked on baseline, 30 nets | 0.5387 | 0.378–0.662 | 0.568 | 0.65 | 0.731 | −0.1517 | [−0.205, −0.089] | worse |

Per-layer ensembles are a follow-up commit; the layer comparison below is at
member level, from `reports/stage3_runs_esm2_35M.csv`.

Three readings.

**Frozen ESM-2 35M features are far below the baseline standalone.** 0.387
against 0.690. The embedding knows a great deal about protein sequence and
little about which peptide stays in which groove.

**Stacking does not merely fail to help — it costs 0.15.** The stacked arm has
1,190 features against the baseline's 860 and the same architecture, budget and
ensemble, so the most likely mechanism is that 330 uninformative columns
displace capacity the one-hot block was using. That distinction matters for how
the result is quoted and is not settled by these runs: a stacked arm with the
ESM block cut to ~32 components and stronger L2 would separate displacement
from damage. Until that is run, the defensible claim is "no gain", not "ESM
features are harmful".

**The layer sweep is flat.** Middle and final layer are indistinguishable at
member level — 0.244 vs 0.266 standalone, 0.399 vs 0.395 stacked, against a
seed spread of 0.010–0.051 recorded in stage 2. The
usual "sweep the layers, middle ones do better on interaction tasks" result does
not appear here, which is consistent with the features carrying little
interaction signal at any depth.

**Cost.** Embedding extraction 70 s on CPU for all 5,633 peptides and 75 domains
(unique sequences, not rows), PCA 8 s, 483 s of fits for 60 networks, inference
0.005 s per 1,000 predictions. No GPU was needed at this model size. Against the
baseline's 73 s of fits, the ESM arms cost ~8x more compute to score 0.15–0.30
lower.

## 3b. Coupling versus concatenation — running

Concatenation hands the head two frozen descriptions and asks it to learn every
peptide-position x pocket-residue interaction in its first layer. `pepstab/attn.py`
builds the interaction into the architecture instead: the 9 peptide positions are
attention queries over the 34 contact residues as keys and values.

The control is the same module with the attention replaced by a uniform average
over the 34 contact residues — identical layer sizes, folds, seeds, budget and
**identical parameter count** (94,913 both arms; the query/key projections exist
in both and are simply unused in the ablation). The only difference is whether
the groove is allowed to condition the peptide, so a gap isolates coupling from
capacity.

This is the test that distinguishes *ESM has nothing to offer here* from
*concatenation cannot use what ESM has*. 3a alone cannot tell them apart.

## 3c. BLOSUM positional cross-features — the control for all of the above

The 9 x 34 matrix of BLOSUM62 scores between peptide position i and contact
residue j: 306 columns, no network, under a second to compute.

These columns contain no information the baseline lacks — they are a fixed
bilinear function of the same residues the one-hot arm already receives. So a
gain cannot mean new information arrived; it can only mean the MLP was not
extracting the interaction efficiently from one-hot input, which is a finding
about architecture rather than representation. A null bounds every cross-feature
scheme, ESM's included, at a cost of ten CPU-minutes. The stacked arm is 1,166
features against the ESM stacked arm's 1,190, so the two cross-representations
are compared at matched width.

## Where the outputs live

`reports/` for tables, figures and prose; `preds/` for `pair_id,y_pred` files;
`features/` for the embedding cache. The last two are gitignored, so both are
rebuilt by the commands above rather than cloned. This follows stages 1 and 2
rather than the `results/esm_stacking/` layout in the task brief — `results/`
is gitignored here, and splitting predictions across two trees would break
`scripts/evaluate.py`, which resolves everything relative to `preds/`.

## Limitations

- **One model size.** ESM-2 35M only. A larger checkpoint is the obvious next
  step, but the 3a failure mode is displacement under a fixed budget, and a
  1,280-dimensional embedding built the same way displaces more. Worth paying
  for only if 3b or 3c shows representation quality is the limiting factor.
- **One reduction.** The PCA retains 90–96% of peptide-block variance. A
  different reduction, or none, could behave differently; the HLA reduction is
  lossless and cannot be the cause.
- **Separate embedding by construction.** 3a embeds peptide and groove in
  isolation, so the peptide's representation is identical in every allele. That
  is the stage 3 specification, and 3b is the response to it.
- **Scope.** All 75 alleles are in training, so every number here describes
  generalisation to novel *peptides* on familiar grooves. Nothing in this report
  bounds what a pretrained model does on an unseen allele — see
  `reports/BIOLOGY_NOTES.md` §6 (companion PR, `analysis/biology-probes`), where the same feature set drops from 0.572 to
  0.325 across allele-distance strata.
- **Not at the noise ceiling.** Four allele pairs differing at one contact
  residue agree at rho 0.90–0.92, so the baseline at 0.69 leaves roughly 0.2 of
  headroom (`BIOLOGY_NOTES.md` §4, companion PR). A flat ESM result is therefore a statement
  about the features, not about a saturated benchmark.

## Constraints on later stages

- **Match the ensemble, or do not ensemble either arm.** Ensembling is worth
  +0.093 mean SCC from no new information. Every arm in this report is 30
  networks for that reason.
- **Quote the regenerated baseline.** 0.6904 on this machine; `preds/` is not in
  the repo, so any comparison starts by rebuilding it.
- **Report cost next to accuracy.** The embedding is cached per unique sequence,
  so extraction is 70 s and inference is free; the expense is in the fits, and
  wider inputs cost both compute and accuracy here.
