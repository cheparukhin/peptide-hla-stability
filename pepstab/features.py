"""Position-preserving sequence encodings for the stage 2 baselines.

Two encodings, one layout. Every residue becomes a 20-vector and the vectors are
concatenated in sequence order, so column ``j * 20 + a`` always means "amino acid
``a`` at position ``j``". Position is never pooled away -- which position an
anchor residue sits at is most of the signal in peptide-HLA binding.

- ``onehot``: 1 at the residue's index, 0 elsewhere.
- ``blosum``: the residue's BLOSUM62 row, divided by 5. Similar amino acids get
  similar vectors, so the model starts with substitution structure instead of
  having to learn it from 19.7k training rows.

Features are **not** standardised. Both encodings already sit in roughly
[-1, 1], and column-wise standardising would divide each one-hot column by its
own frequency -- inflating a residue seen at one position in 0.1% of rows by
~30x purely because it is rare. Keeping the raw scale is also what NetMHC-family
models do.

Encoding is cached per *unique sequence*, not per measurement row: 28,166 rows
carry only 5,633 distinct peptides and 75 distinct HLA sequences.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import (
    AMINO_ACIDS,
    HLA_DOMAIN_LENGTH,
    PEPTIDE_LENGTH,
    PSEUDOSEQ_LENGTH,
    encode_sequences,
)

#: BLOSUM62, rows and columns in :data:`pepstab.data.AMINO_ACIDS` order
#: (``ACDEFGHIKLMNPQRSTVWY``), reordered from the reference matrix at
#: ftp.ncbi.nlm.nih.gov/blast/matrices/BLOSUM62. Symmetric; the diagonal runs
#: from 4 (A) to 11 (W), the rarest residue.
BLOSUM62 = (
    ( 4,  0, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3, -2),  # A
    ( 0,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2),  # C
    (-2, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -3),  # D
    (-1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -2),  # E
    (-2, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1,  3),  # F
    ( 0, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -3),  # G
    (-2, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2,  2),  # H
    (-1, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1),  # I
    (-1, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -2),  # K
    (-1, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1),  # L
    (-1, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1),  # M
    (-2, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -2),  # N
    (-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -3),  # P
    (-1, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1),  # Q
    (-1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -2),  # R
    ( 1, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3, -2),  # S
    ( 0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2, -2),  # T
    ( 0, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1),  # V
    (-3, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11,  2),  # W
    (-2, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2,  7),  # Y
)

#: BLOSUM62 rows are divided by this before use, which puts the encoding on the
#: same rough scale as one-hot. The NetMHC family uses the same convention.
BLOSUM_SCALE = 5.0

_ONEHOT_ROWS = np.eye(len(AMINO_ACIDS), dtype=np.float32)
_BLOSUM_ROWS = np.asarray(BLOSUM62, dtype=np.float32) / BLOSUM_SCALE

ENCODINGS = ("onehot", "blosum")

#: Sequence columns, with the length each one is checked against.
SEQ_LENGTHS = {
    "peptide": PEPTIDE_LENGTH,
    "hla_pseudoseq": PSEUDOSEQ_LENGTH,
    "hla_seq": HLA_DOMAIN_LENGTH,
}

#: The input sets compared at stage 2, each a tuple of sequence columns.
#:
#: ``pep_pseudo`` is the stage 2 baseline proper. ``pep_domain`` exists so the
#: stage 3 ESM comparison has a matching raw-sequence arm with the *same* input
#: sequence -- otherwise a win for domain embeddings could just be a win for
#: feeding the model 182 residues instead of 34. ``pep`` is the reference that
#: says how much of the signal needs the HLA at all.
INPUT_SETS = {
    "pep": ("peptide",),
    "pep_pseudo": ("peptide", "hla_pseudoseq"),
    "pep_domain": ("peptide", "hla_seq"),
}


def residue_rows(encoding: str) -> np.ndarray:
    """The ``(20, 20)`` lookup table one residue index maps through."""
    if encoding == "onehot":
        return _ONEHOT_ROWS
    if encoding == "blosum":
        return _BLOSUM_ROWS
    raise ValueError(f"unknown encoding {encoding!r}; expected one of {ENCODINGS}")


def encode_matrix(seqs, length: int, encoding: str) -> np.ndarray:
    """Encode equal-length sequences as ``(n, length * 20)`` float32.

    Column ``j * 20 + a`` is amino acid ``a`` at position ``j``.
    """
    codes = encode_sequences(seqs, length)
    rows = residue_rows(encoding)
    return rows[codes].reshape(len(codes), length * len(AMINO_ACIDS))


def feature_names(input_set: str, encoding: str) -> list[str]:
    """Column labels matching :func:`build_features`, for debugging a model."""
    del encoding  # layout is the same for both encodings
    names = []
    for col in INPUT_SETS[input_set]:
        short = {"peptide": "P", "hla_pseudoseq": "S", "hla_seq": "D"}[col]
        for pos in range(SEQ_LENGTHS[col]):
            names.extend(f"{short}{pos + 1}:{aa}" for aa in AMINO_ACIDS)
    return names


def build_features(df: pd.DataFrame, input_set: str, encoding: str) -> np.ndarray:
    """Feature matrix for ``df``, one row per measurement, cached per sequence.

    The returned block order follows ``INPUT_SETS[input_set]``, so the same
    ``(input_set, encoding)`` pair always produces the same columns for any
    subset of rows -- which is what lets a model fit on train rows and predict
    val rows.
    """
    if input_set not in INPUT_SETS:
        raise ValueError(f"unknown input set {input_set!r}; "
                         f"expected one of {sorted(INPUT_SETS)}")
    blocks = []
    for col in INPUT_SETS[input_set]:
        values = df[col].to_numpy(dtype=str)
        # Encode each distinct sequence once, then fan out to rows. 28,166 rows
        # hold 5,633 peptides and 75 HLA sequences, so this is a ~5x saving on
        # the peptide block and a ~375x saving on either HLA block.
        uniq, inverse = np.unique(values, return_inverse=True)
        encoded = encode_matrix(uniq, SEQ_LENGTHS[col], encoding)
        blocks.append(encoded[inverse])
    return np.concatenate(blocks, axis=1) if len(blocks) > 1 else blocks[0]


def n_features(input_set: str) -> int:
    """Width of :func:`build_features` output, without building it."""
    return sum(SEQ_LENGTHS[c] for c in INPUT_SETS[input_set]) * len(AMINO_ACIDS)
