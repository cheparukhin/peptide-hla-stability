"""Canonical dataset loading and shared example IDs.

The raw CSV under ``data/`` is read-only (see CLAUDE.md). Every workstream must
reach the data through this module so that ``pair_id`` means the same thing
everywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_CSV = DATA_DIR / "rasmussen_et_al_dataset.csv"
SPLITS_CSV = DATA_DIR / "splits.csv"

PEPTIDE_LENGTH = 9
HLA_DOMAIN_LENGTH = 182
PSEUDOSEQ_LENGTH = 34

#: The 20 standard amino acids, in a fixed order, so one-hot columns are
#: identical across workstreams.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

#: Target column, and the transform every model trains on. ``log1p`` is defined
#: at 0, which matters because 20.2% of labels are exactly 0 (see
#: reports/audit_summary.md).
TARGET_RAW = "thalf_hours"
TARGET = "y_log1p"


def load_raw() -> pd.DataFrame:
    """Load the canonical CSV and attach the shared ``pair_id``.

    ``pair_id`` is the 0-based row position in the read-only CSV. It is stable
    because the file never changes, and it is the join key for predictions,
    embeddings, structures, and feature tables.
    """
    df = pd.read_csv(RAW_CSV, dtype={
        "allele": "string",
        "peptide": "string",
        "hla_seq": "string",
        "hla_pseudoseq": "string",
    })
    df.insert(0, "pair_id", np.arange(len(df), dtype=np.int32))
    df[TARGET] = np.log1p(df[TARGET_RAW])
    return df


def load_with_splits() -> pd.DataFrame:
    """Load the canonical CSV joined to the frozen split assignments.

    Splits are read from disk, never recomputed -- regenerating them would drop
    the peptide-cluster grouping and leak training peptides into the test set.
    """
    if not SPLITS_CSV.exists():
        raise FileNotFoundError(
            f"{SPLITS_CSV} is missing. It is a committed artifact; restore it "
            "from git rather than regenerating it."
        )
    df = load_raw()
    splits = pd.read_csv(SPLITS_CSV, dtype={"split": "string", "peptide": "string"})
    merged = df.merge(splits[["pair_id", "cluster_id", "split", "dist_to_train"]],
                      on="pair_id", how="left", validate="one_to_one")
    if merged["split"].isna().any():
        raise ValueError("splits.csv does not cover every pair_id")
    return merged


def encode_sequences(seqs: pd.Series | list[str], length: int) -> np.ndarray:
    """Encode equal-length sequences as an ``(n, length)`` int8 index matrix.

    Position is preserved: column ``j`` is sequence position ``j``. Raises on any
    residue outside :data:`AMINO_ACIDS`, so non-standard letters cannot slip
    silently into a model.
    """
    arr = np.asarray(seqs, dtype=str)
    if arr.size and not np.all(np.char.str_len(arr) == length):
        raise ValueError(f"expected every sequence to have length {length}")
    codes = np.full((len(arr), length), -1, dtype=np.int8)
    for i, s in enumerate(arr):
        for j, aa in enumerate(s):
            idx = AA_INDEX.get(aa, -1)
            if idx < 0:
                raise ValueError(f"unknown residue {aa!r} in {s!r}")
            codes[i, j] = idx
    return codes
