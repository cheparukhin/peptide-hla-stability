# Datasets and Biological Provenance

This document describes the experimental datasets, sequence curation, and homology-aware splits used throughout the study.

---

## 1. Primary Kinetic Stability Dataset (Rasmussen et al., 2016)

### 1.1 Experimental Assay
The primary supervised dataset is derived from Rasmussen et al., *J. Immunol.* **197**, 1517–1524 (2016) ([`data/rasmussen_et_al_2016.pdf`](../data/rasmussen_et_al_2016.pdf)), curated in [`data/rasmussen_et_al_dataset.csv`](../data/rasmussen_et_al_dataset.csv) and [`data/splits.csv`](../data/splits.csv) via [`pepstab/data.py`](../pepstab/data.py) (from raw `data/raw/dtu_stability_data.xlsx` / `data/processed/stability.parquet`).

Dissociation half-lives ($t_{1/2}$, in hours) were measured using a high-throughput scintillation proximity assay (SPA). Recombinant biotinylated HLA class I heavy chains were refolded with peptide and radiolabeled $^{125}\text{I}$-$\beta_2$-microglobulin ($\beta_2\text{m}$) in streptavidin-coated FlashPlates, then incubated at **$37\,^\circ\text{C}$** in the presence of excess unlabeled $\beta_2\text{m}$ to prevent reassociation. Because $\beta_2\text{m}$ dissociates rapidly once the peptide leaves the binding groove, the decay of the proximity scintillation signal tracks complex dissociation kinetics ($t_{1/2} = \ln 2 / k_{\text{off}}$). Published values represent the **geometric mean of two independent experimental replicates**; raw time-resolved decay curves and replicate-level variance are not included in the source publication.

### 1.2 Cohort Composition and Censoring Floor
After deterministic ingestion and deduplication, [`data/rasmussen_et_al_dataset.csv`](../data/rasmussen_et_al_dataset.csv) contains **28,166** unique `(allele, peptide)` records:
- **Peptides**: **5,633** unique 9-mer peptides composed exclusively of the 20 canonical amino acids.
- **HLA class I alleles**: **75** distinct `allele` strings ([`data/hla75_constructs.fasta`](../data/hla75_constructs.fasta)) mapping to **72** canonical four-digit alleles (**26** `HLA-A`, **34** `HLA-B`, **11** `HLA-C`, and **1** `HLA-E*01:01`) plus **3** engineered disulfide-modified constructs (`HLA-A*02:01 C67S`, `HLA-A*24:02 C67S`, and `HLA-B*07:02 C67S`, audited separately in [`data/c67s_cleanup/benchmark_C67S.csv`](../data/c67s_cleanup/benchmark_C67S.csv)).
- **Engineered construct handling**: The `C67S` mutation lies outside the 34 canonical peptide-contacting groove positions (`hla_pseudoseq` is identical to the parent wild-type allele), but at position 67 of the 182-residue $\alpha_1\alpha_2$ domain (`hla_seq`). We preserve `allele` as the distinct allele identifier, pass the exact mutated `hla_seq` to ESM-2 and Boltz-2 structure prediction, and exclude engineered rows when matching wild-type equilibrium affinity measurements.
- **Target scale and assay floor**: Half-lives are transformed in [`pepstab/data.py`](../pepstab/data.py) as $y = \ln(1 + t_{1/2})$ (`y_log1p`). Exactly **5,693** records (**20.21%**) have $t_{1/2} = 0.0\text{ h}$ ($y = 0.0$), reflecting complexes below the assay's $0.1\text{ h}$ detection limit ($c = \ln(1 + 0.1) \approx 0.0953$ in Tobit left-censored modeling; [`pepstab/censored.py`](../pepstab/censored.py)) or complexes that failed initial refolding.

### 1.3 Schema of [`data/rasmussen_et_al_dataset.csv`](../data/rasmussen_et_al_dataset.csv) and [`data/splits.csv`](../data/splits.csv)

| Column | Source | Type | Description |
| :--- | :--- | :--- | :--- |
| `pair_id` | `splits.csv` / `data.py` | `int32` | Canonical 0-based row index (`0..28165`) joining all tables, embeddings, and structures |
| `allele` | `rasmussen_et_al_dataset.csv` | `string` | Exact allele identifier from Rasmussen et al. (75 distinct values, preserving `C67S` suffixes) |
| `peptide` | `rasmussen_et_al_dataset.csv` | `string` | 9-mer amino-acid sequence (5,633 unique peptides) |
| `thalf_hours` | `rasmussen_et_al_dataset.csv` | `float64` | Reported dissociation half-life $t_{1/2}$ in hours ($5,693$ rows at $0.0\text{ h}$, $20.21\%$) |
| `hla_seq` | `rasmussen_et_al_dataset.csv` | `string` | 182-residue HLA class I $\alpha_1\alpha_2$ groove domain sequence (reflecting `C67S` where applicable) |
| `hla_pseudoseq` | `rasmussen_et_al_dataset.csv` | `string` | 34-residue polymorphic HLA class I contact pseudosequence |
| `y_log1p` | `pepstab/data.py` | `float64` | Variance-stabilized natural-logarithm target $y = \ln(1 + t_{1/2})$ |
| `cluster_id` | `splits.csv` | `int32` | Connected-component ID under single-linkage peptide Hamming distance $\le 3$ |
| `split` | `splits.csv` | `string` | Homology-aware split assignment: `train` (19,716), `val` (2,817), or `test` (5,633) |
| `dist_to_train` | `splits.csv` | `int8` | Minimum peptide Hamming distance to any training peptide ($\ge 4$ for all `val` and `test` rows) |

---

## 2. External Immunopeptidomics Elution Benchmark (MHC Motif Atlas)

To test whether kinetic stability models trained on the Rasmussen et al. SPA dataset generalize zero-shot to natural cellular antigen presentation, we curated an external benchmark from the **MHC Motif Atlas** (`data/external/mhc_motif_atlas/`):

1. **Direct stability–elution overlap (`data/processed/elution_overlap_pairs.parquet`)**:
   - Intersecting the $21,663$ stability-measured 9-mers across **54** shared canonical HLA alleles with $151,170$ naturally eluted 9-mers in the MHC Motif Atlas yields **140** exact `(allele_canonical, peptide)` matches across $35$ alleles.
   - These 140 eluted ligands have a median measured $t_{1/2}$ of **$7.90\text{ h}$** vs **$1.10\text{ h}$** for the $21,523$ non-eluted pairs ($p = 6.0 \times 10^{-31}$, two-sided Mann–Whitney $U$) and a $10\times$ lower zero-stability floor rate ($2.1\%$ vs $21.2\%$).

2. **Zero-shot elution classification cohort (`stage3c`, [`reports/stage3c_summary.csv`](../reports/stage3c_summary.csv))**:
   - **Symmetric leakage exclusion**: All $5,633$ peptides present in the Rasmussen stability dataset and any peptides containing non-canonical or post-translationally modified residues are removed from both positive and decoy pools prior to sampling.
   - **Allele panel**: **51** shared HLA class I alleles having at least $100$ remaining eluted 9-mer positives (capped at $2,000$ randomly sampled positives per allele; seed `20260307`), totaling **81,600** eluted positive pairs.
   - **Decoy sets (10:1 ratio, 816,000 decoys per condition)**:
     - *Human proteome decoys*: Random 9-mer windows sampled from the reviewed human Swiss-Prot proteome (`uniprot_human_proteome_reviewed.fasta.gz`) disjoint from that allele's positives and from all stability peptides.
     - *Allele-swapped eluted-ligand decoys*: Real MS-eluted 9-mer ligands drawn from *other* HLA alleles in the MHC Motif Atlas (excluding any ligand eluted on the target allele), matching generic source-protein abundance, proteasomal cleavage, TAP transport, and MS ionizability.

---

## 3. External Equilibrium Binding Affinity Reference (MHCflurry / BD2013)

Quantitative equilibrium binding affinity ($K_d$ / $\text{IC}_{50}$) measurements from the MHCflurry 2.0 / IEDB BD2013 curation are tracked in [`data/data_augmentation_iedb/affinity_reference_75alleles.csv`](../data/data_augmentation_iedb/affinity_reference_75alleles.csv) and [`data/augmentation/measured_affinity.csv`](../data/augmentation/measured_affinity.csv) (**140,855** human 9-mer class I records across **105** alleles, with **121,150** records across the **64** overlapping wild-type benchmark alleles and **7,281** exact `(allele, peptide)` pairs overlapping the stability cohort). Full details and the empirical comparison between $K_d$ and $t_{1/2}$ are provided in [`docs/AFFINITY_REFERENCE.md`](AFFINITY_REFERENCE.md).
