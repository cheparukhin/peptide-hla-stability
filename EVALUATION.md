# Evaluation Protocol and Statistical Methodology

This document specifies the dataset splits, evaluation metrics, eligibility criteria, and bootstrap resampling procedures used across the study. Full empirical results are presented in [`reports/REPORT.md`](reports/REPORT.md).

---

## 1. Target Transformation and Assay Floor

All supervised kinetic stability experiments use the Rasmussen et al. (2016) scintillation proximity assay (SPA) dataset ([`data/rasmussen_et_al_dataset.csv`](data/rasmussen_et_al_dataset.csv) and [`data/splits.csv`](data/splits.csv); $28,166$ unique `(allele, peptide)` records across $5,633$ 9-mer peptides and $75$ HLA class I allele strings). Measured dissociation half-lives $t_{1/2}$ (hours, reported as two-replicate geometric means) are transformed to a variance-stabilized natural-logarithm target (`pepstab/data.py`):
$$y = \ln(1 + t_{1/2}) \quad (\text{\texttt{y\_log1p}}).$$
Because $y = \ln(1 + t_{1/2})$ is a strictly monotonic function of $t_{1/2} \ge 0$, within-allele Spearman rank correlations $\rho$ are invariant to the logarithmic base. Exactly **5,693 / 28,166 (20.21%)** observations are recorded at $t_{1/2} = 0.0\text{ h}$ ($y = 0.0$), sitting below the assay's $t_{\text{floor}} = 0.1\text{ h}$ reporting limit ($c = \ln(1 + 0.1) \approx 0.0953$ in Tobit left-censored modeling; [`pepstab/censored.py`](pepstab/censored.py)).

---

## 2. Homology-Aware Splits and Cross-Validation Regimes

### 2.1 Peptide-Cluster Split (`splits.csv`)
To prevent sequence-similar peptides from leaking across training and evaluation partitions, all $5,633$ unique 9-mer peptides are grouped into connected components under single-linkage clustering at **Hamming distance $\le 3$** ($\ge 66.7\%$ sequence identity; implemented in [`pepstab/splits.py`](pepstab/splits.py)). Connected peptide clusters are assigned to three disjoint partitions:
- **Training cohort (`train`)**: **19,716** rows (**70.0%**), **3,587** peptide clusters, **75** alleles (**69** eligible alleles with $\ge 20$ rows).
- **Validation split (`val`)**: **2,817** rows (**10.0%**), **68** eligible alleles ($2,756$ rows).
- **Test split (`test`)**: **5,633** rows (**20.0%**), **67** eligible alleles ($5,565$ rows).

### 2.2 Stage I vs. Stage II Evaluation Partitioning
1. **Stage I (Test-Split Benchmark & External Elution Transfer)**: Initial sequence MLP baselines (`stage2`), frozen ESM-2 embeddings (`stage3`, `stage3e`, `stage3f`), coarse Boltz-2 3D geometry and confidence summaries (`stage5`), and censored Tobit calibration (`stage7_censored`) were tuned on `val` and evaluated once on `test`. Zero-shot immunopeptidomics transfer (`stage3c`) was evaluated on **897,600** disjoint peptide–HLA pairs across **51** alleles from the MHC Motif Atlas.
2. **Stage II (Nested Cross-Validation Within the 19,716-Pair Training Cohort)**: Because the initial `val` and `test` partitions were inspected during post-hoc diagnosis of Stage I structural features, all subsequent development of internal Boltz-2 trunk representations (`R2`–`R5`, `full_interface_pos`), model selectors, cross-allele transfer, few-shot adaptation, and ProteinMPNN/FoldX extensions was restricted strictly to the **19,716-pair `train` cohort**:
   - **Nested 5-fold peptide-cluster CV**: The $3,587$ training peptide clusters are partitioned into 5 outer folds stratified by allele coverage ([`reports/further_research/phase_a_fold_assignments.csv`](reports/further_research/phase_a_fold_assignments.csv)). For each outer fold, hyperparameters, early-stopping epochs, and model-selector choices (`competitive_sequence_selector`, `embedding_or_original_selector`) are determined exclusively on inner 4-fold cross-validation splits before generating out-of-fold predictions on the held-out outer fold.
   - **Crossed $5 \times 5$ HLA-group $\times$ peptide-cluster CV**: All 75 HLA alleles are clustered into connected components at **Hamming distance $\le 3$** on the 34-residue contact pseudosequence (`HLA_groups`) and partitioned into 5 HLA-group folds. Crossing the 5 HLA-group folds with the 5 peptide-cluster folds yields 25 evaluation cells supporting two transfer regimes:
     - *Unseen HLA group*: Evaluated on held-out HLA groups using training peptide clusters.
     - *Joint holdout*: Evaluated on cells where both the HLA sequence group and the peptide cluster are held out from training.

---

## 3. Primary Endpoint and Secondary Diagnostics

### 3.1 Allele Eligibility and Primary Metric
Within any evaluation partition, an HLA allele (`allele`) is **eligible** if it has at least **20 observations** ($n_a \ge 20$) and at least **2 distinct target values** in that partition.
- In the $19,716$-pair training cohort, **69 of 75** alleles meet the $n_a \ge 20$ threshold (and all 69 also satisfy $n_a \ge 50$). The 6 excluded low-count alleles are `HLA-A*02:02`, `HLA-A*02:05`, `HLA-A*02:07`, `HLA-A*02:11`, `HLA-A*68:01`, and `HLA-B*58:01`.
- In the few-shot joint-holdout adaptation study, target alleles require at least **40 joint-holdout observations** so that 20 support peptides can be sampled while leaving $\ge 20$ query peptides in disjoint peptide clusters, yielding **68** eligible alleles (`HLA-A*26:03` has 37 rows and is excluded).

For each eligible allele $a$, we compute the within-allele Spearman rank correlation $\rho_a(\hat{y}, y)$ using average ranks for ties. The **primary evaluation metric** is the **median within-allele Spearman correlation** across eligible alleles:
$$\text{Median }\rho = \operatorname{median}_{a \in \mathcal{A}_{\text{elig}}} \rho_a(\hat{y}, y).$$
Equal allele weighting prevents a small number of densely measured alleles (such as `HLA-A*02:01`) from dominating the benchmark. A pre-specified improvement threshold of **$\Delta\rho \ge +0.05$** was declared prior to representation benchmarking to define a substantial ranking gain over sequence baselines.

### 3.2 Secondary Metrics
Secondary metrics reported across experimental tables include:
- **Mean within-allele Spearman $\rho$** across eligible alleles.
- **Pooled RMSE and MAE** on $y = \ln(1 + t_{1/2})$ across all evaluated pairs.
- **Within-allele top-10% recall**: fraction of an allele's true top-decile most stable peptides recovered within the top decile of predicted scores.
- **Assay-floor discrimination and calibration**: AUROC and Brier score for predicting the censored floor event ($t_{1/2} < 0.1\text{ h}$).
- **Zero-shot ligand elution metrics (`stage3c`)**: Per-allele AUROC, AUPRC, and top-1% precision (and fold enrichment over the $1/11 = 0.0909$ baseline prevalence) across $51$ MHC Motif Atlas alleles.

---

## 4. Bootstrap Confidence Intervals and Multiplicity Adjustment

All model comparisons are evaluated as paired differences on the exact same out-of-fold or held-out predictions:
$$\Delta\rho = \operatorname{median}_{a \in \mathcal{A}_{\text{elig}}} \rho_a(\hat{y}_{\text{cand}}, y) - \operatorname{median}_{a \in \mathcal{A}_{\text{elig}}} \rho_a(\hat{y}_{\text{base}}, y).$$
Non-parametric percentile bootstrap intervals (`2,000` resamples, deterministic seed `20261003` on the test split and pre-specified seeds in Stage II) match the dependency structure of each holdout regime:
1. **Peptide-cluster bootstrap** (used for test-split and nested peptide-cluster CV comparisons): Resamples peptide clusters (`cluster_id`) with replacement, retaining all allele observations belonging to each drawn cluster, and recomputes per-allele Spearman correlations and the paired median difference $\Delta\rho^{(b)}$ on each draw.
2. **Crossed HLA-group $\times$ peptide-cluster bootstrap** (used for unseen HLA-group transfer, joint holdouts, and few-shot adaptation): Simultaneously resamples HLA sequence groups (`HLA_groups`) and peptide clusters (`cluster_id`) with replacement (`*_crossed_draws.csv`), accounting for variance across both allele families and peptide families.
3. **Bonferroni family-wise error control**: For the three pre-declared primary follow-up hypotheses in Stage II (20-shot `R5` ridge adaptation vs zero-shot, ProteinMPNN complex-context increment `R5+P+C` vs `R5+P`, and FoldX energy increment `R5+E` vs `R5`), we report two-sided **$98.3333\%$** ($1 - 0.05/3$, quantiles $0.008333$ and $0.991667$) percentile intervals alongside pointwise $95\%$ intervals.
