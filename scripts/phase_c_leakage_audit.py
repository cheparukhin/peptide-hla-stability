"""Phase C gate: peptide-cluster leakage audit for the MINT checkpoint ladder.

The question the gate has to answer is not "did this checkpoint see our exact
peptides" but "did it see anything inside our held-out clusters". Our clusters
are single-linkage components at Hamming <= 3 (`pepstab.splits`), so a donor
peptide three substitutions from a validation peptide is inside that validation
cluster even though the strings differ.

Three donor corpora are audited, one per rung of the candidate ladder:

* **stability** -- the NetMHCstabpan training corpus that SPEARMINT's stage 2
  and final checkpoints were fine-tuned on. We do not have their file, but it
  is the Rasmussen 2016 corpus, which *is* `data/splits.csv`: 27,034 rows over
  72 alleles against our 28,166 over 75. So the overlap is measured as
  self-overlap -- our own val and test rows -- and the answer is total by
  construction. Reported for completeness, not as a discovery.

* **affinity** -- the NetMHCpan 4.1 BA corpus behind
  `dkarthikeyan1/mint-stage1-affinity`. We do not have NetMHCpan's file either,
  so `data/data_augmentation_iedb/affinity_reference_75alleles.csv` stands in.
  It is **not** a subset of NetMHCpan 4.1's BA data: its primary source is the
  MHCflurry curated 2023-10-23 IEDB snapshot, three years *later* than
  NetMHCpan 4.1 (Reynisson et al. 2020), so it holds measurements that corpus
  cannot contain. The number it yields is therefore neither a bound nor an
  estimate of the real overlap -- only evidence that overlap is extensive.

* **affinity, BD2013-restricted** -- the same table filtered to rows carrying a
  Kim et al. 2014 BD2013 measurement. This one *is* a defensible lower bound.
  BD2013 entered the NetMHCpan lineage at NetMHCpan-4.0 (Jurtz et al. 2017) and
  4.1's BA partition is built on that lineage plus later IEDB, so every
  BD2013 row is in the donor corpus. That is a documented-provenance claim about
  someone else's training file, not something measured here, and it is the one
  link in this chain that a reader should check rather than take on trust.

* **STRING** -- the corpus behind the base `varunullanat2012/mint`. Not audited
  numerically here. STRING entries are whole proteins, not 8-11mers, and the
  checkpoint carries no pMHC label of any kind, so there is no peptide-level
  donor set to intersect. The relevant exposure is source-protein level, which
  is shared by every pLM in the project including the deployed ESM-2 arm.

Writes `reports/further_research/phase_c_cluster_overlap.csv` and prints the
table that goes into the gate report. Reads only; touches no frozen artefact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pepstab.splits import HAMMING_THRESHOLD, PEPTIDE_LENGTH  # noqa: E402

SPLITS = ROOT / "data" / "splits.csv"
AFFINITY = ROOT / "data" / "data_augmentation_iedb" / "affinity_reference_75alleles.csv"
OUT_DIR = ROOT / "reports" / "further_research"


def encode(peptides: list[str]) -> np.ndarray:
    """Peptides as uint8 code matrices, so Hamming is a vectorised compare."""
    return np.frombuffer("".join(peptides).encode(), dtype=np.uint8).reshape(
        len(peptides), PEPTIDE_LENGTH
    )


def min_hamming(query: np.ndarray, donor: np.ndarray, block: int = 256) -> np.ndarray:
    """Minimum Hamming distance from each query row to any donor row.

    Blocked over the query axis: 5.6k x 40k x 9 as one bool array is 2 GB, and
    the whole point of this audit is that it runs on a laptop for free.
    """
    out = np.full(len(query), PEPTIDE_LENGTH, dtype=np.int16)
    for start in range(0, len(query), block):
        chunk = query[start : start + block]
        d = (chunk[:, None, :] != donor[None, :, :]).sum(axis=2)
        out[start : start + block] = d.min(axis=1)
    return out


def audit(name: str, splits: pd.DataFrame, donor_peptides: set[str]) -> list[dict]:
    """Per-split contamination of our clusters by one donor corpus."""
    donor = sorted(p for p in donor_peptides if len(p) == PEPTIDE_LENGTH)
    donor_codes = encode(donor)

    rows = []
    for split in ("train", "val", "test"):
        part = splits[splits.split == split]
        peptides = sorted(part.peptide.unique())
        dist = min_hamming(encode(peptides), donor_codes)
        by_peptide = pd.DataFrame({"peptide": peptides, "dist": dist})

        # A cluster is contaminated if ANY of its peptides sits within the
        # clustering threshold of ANY donor peptide -- that donor peptide would
        # have joined this cluster had it been in the dataset.
        merged = part.merge(by_peptide, on="peptide")
        per_cluster = merged.groupby("cluster_id").dist.min()
        n_clusters = len(per_cluster)
        rows.append(
            {
                "donor_corpus": name,
                "split": split,
                "peptides": len(peptides),
                "clusters": n_clusters,
                "peptides_exact_match": int((dist == 0).sum()),
                "peptides_within_threshold": int((dist <= HAMMING_THRESHOLD).sum()),
                "clusters_exact_match": int((per_cluster == 0).sum()),
                "clusters_within_threshold": int(
                    (per_cluster <= HAMMING_THRESHOLD).sum()
                ),
                "pct_clusters_within_threshold": round(
                    100 * (per_cluster <= HAMMING_THRESHOLD).sum() / n_clusters, 1
                ),
                "rows": len(part),
                "rows_in_contaminated_clusters": int(
                    merged.cluster_id.isin(
                        per_cluster[per_cluster <= HAMMING_THRESHOLD].index
                    ).sum()
                ),
            }
        )
    return rows


def main() -> None:
    splits = pd.read_csv(SPLITS)
    assert set(splits.columns) >= {"allele", "peptide", "cluster_id", "split"}
    print(
        f"{SPLITS.relative_to(ROOT)}: {len(splits):,} rows, "
        f"{splits.peptide.nunique():,} peptides, "
        f"{splits.cluster_id.nunique():,} clusters, threshold Hamming <= {HAMMING_THRESHOLD}"
    )

    results: list[dict] = []

    # Rung 3: SPEARMINT stage-2 / final. Donor corpus == our corpus.
    results += audit("netmhcstabpan_stability", splits, set(splits.peptide))

    # Rung 2: mint-stage1-affinity, proxied by the local curated affinity table.
    aff = pd.read_csv(AFFINITY, usecols=["peptide", "bd2013_affinity_nM"])
    print(
        f"{AFFINITY.relative_to(ROOT)}: {len(aff):,} rows, "
        f"{aff.peptide.nunique():,} unique 9-mers (affinity-corpus proxy)"
    )
    results += audit("netmhcpan41_ba_proxy", splits, set(aff.peptide))

    # Rung 2, lower bound: only the rows the NetMHCpan lineage demonstrably has.
    bd = aff[aff.bd2013_affinity_nM.notna()]
    print(
        f"  of which BD2013-backed: {len(bd):,} rows, "
        f"{bd.peptide.nunique():,} unique 9-mers (lower-bound donor set)"
    )
    results += audit("netmhcpan41_ba_bd2013_lower_bound", splits, set(bd.peptide))

    out = pd.DataFrame(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / "phase_c_cluster_overlap.csv"
    out.to_csv(dest, index=False)

    cols = [
        "donor_corpus",
        "split",
        "clusters",
        "clusters_exact_match",
        "clusters_within_threshold",
        "pct_clusters_within_threshold",
        "rows",
        "rows_in_contaminated_clusters",
    ]
    print("\n" + out[cols].to_string(index=False))
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
