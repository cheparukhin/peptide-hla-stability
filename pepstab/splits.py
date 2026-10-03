"""Peptide-cluster grouping and the frozen split assignment.

All peptides are 9 residues, so Hamming distance is exact and needs no
alignment. Peptides are grouped by single-linkage clustering at Hamming
distance <= 3 and every cluster lands wholly in one split, across all alleles.
This guarantees that no validation or test peptide is within 3 substitutions of
any training peptide.

The metric is plain Hamming on purpose. A BLOSUM-weighted or embedding-based
distance would add a threshold to defend, and an ESM-2 distance would make the
split depend on the model under test at stage 3.

This module *builds* the split. Consumers should load the frozen result with
:func:`pepstab.data.load_with_splits` instead of calling anything here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import PEPTIDE_LENGTH, encode_sequences

HAMMING_THRESHOLD = 3
SPLIT_NAMES = ("train", "val", "test")
SPLIT_FRACTIONS = (0.70, 0.10, 0.20)


def pairwise_hamming(codes: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Full Hamming distance matrix for an ``(n, L)`` integer code matrix."""
    n = len(codes)
    out = np.empty((n, n), dtype=np.int8)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = codes[start:stop, None, :] != codes[None, :, :]
        out[start:stop] = block.sum(axis=2).astype(np.int8)
    return out


def single_linkage_clusters(codes: np.ndarray, threshold: int = HAMMING_THRESHOLD,
                            chunk: int = 512) -> np.ndarray:
    """Single-linkage cluster labels at ``Hamming <= threshold``.

    Single linkage over a distance threshold is exactly the connected components
    of the graph that joins every pair within the threshold, so this is a
    union-find over those edges -- no linkage matrix needed.

    Labels are canonical: cluster ids are assigned in order of each cluster's
    smallest member index, so the labelling does not depend on iteration order.
    """
    n = len(codes)
    parent = np.arange(n)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        dist = (codes[start:stop, None, :] != codes[None, :, :]).sum(axis=2)
        rows, cols = np.nonzero(dist <= threshold)
        rows = rows + start
        for i, j in zip(rows.tolist(), cols.tolist()):
            if i == j:
                continue
            ri, rj = find(i), find(j)
            if ri != rj:
                # Union by smaller root keeps the representative canonical.
                if ri < rj:
                    parent[rj] = ri
                else:
                    parent[ri] = rj

    roots = np.array([find(i) for i in range(n)])
    order = {root: rank for rank, root in enumerate(sorted(set(roots.tolist())))}
    return np.array([order[r] for r in roots.tolist()], dtype=np.int32)


def assign_clusters(cluster_weights: pd.Series,
                    fractions: tuple[float, ...] = SPLIT_FRACTIONS,
                    names: tuple[str, ...] = SPLIT_NAMES) -> pd.Series:
    """Water-fill whole clusters into splits to hit ``fractions`` by row count.

    Clusters are placed largest-first into whichever split is furthest below its
    target *as a fraction of that target*, which keeps the three splits growing
    in proportion all the way down the cluster-size distribution.

    The relative criterion is load-bearing. Ranking by *absolute* deficit
    instead degenerates: once the three absolute deficits equalise they stay
    equal, so clusters go round-robin into thirds, and the heavy clusters all
    land in whichever split led early. That still reaches 70/10/20 overall but
    leaves whole alleles with no held-out rows, because each allele was assayed
    on its own peptide panel.

    Deterministic: clusters are ordered by ``(-weight, cluster_id)`` and ties on
    the fill ratio break by split order.

    ``cluster_weights`` maps cluster id -> number of measurement rows.
    """
    total = int(cluster_weights.sum())
    targets = np.array(fractions, dtype=float) * total
    filled = np.zeros(len(names), dtype=float)
    assignment: dict[int, str] = {}

    order = sorted(cluster_weights.index, key=lambda c: (-int(cluster_weights[c]), int(c)))
    for cid in order:
        k = int(np.argmin(filled / targets))
        assignment[cid] = names[k]
        filled[k] += float(cluster_weights[cid])

    return pd.Series(assignment, name="split").sort_index()


def build_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Return a ``pair_id, peptide, allele, cluster_id, split`` frame."""
    peptides = pd.Index(sorted(df["peptide"].unique()))
    codes = encode_sequences(peptides.to_list(), PEPTIDE_LENGTH)
    labels = single_linkage_clusters(codes)
    pep_to_cluster = pd.Series(labels, index=peptides, name="cluster_id")

    out = df[["pair_id", "peptide", "allele"]].copy()
    out["cluster_id"] = out["peptide"].map(pep_to_cluster).astype(np.int32)

    weights = out.groupby("cluster_id").size()
    cluster_split = assign_clusters(weights)
    out["split"] = out["cluster_id"].map(cluster_split).astype("string")
    out["dist_to_train"] = out["peptide"].map(distance_to_train(out)).astype(np.int8)
    return out


def distance_to_train(df: pd.DataFrame, chunk: int = 512) -> pd.Series:
    """Hamming distance from each peptide to its nearest *training* peptide.

    Frozen into ``data/splits.csv`` alongside the split so stage 6 can stratify
    held-out metrics by it without recomputing anything. Training peptides get
    0 (they are their own nearest neighbour).

    Indexed by peptide.
    """
    train_peptides = sorted(df.loc[df["split"] == "train", "peptide"].unique())
    train_codes = encode_sequences(train_peptides, PEPTIDE_LENGTH)

    all_peptides = sorted(df["peptide"].unique())
    codes = encode_sequences(all_peptides, PEPTIDE_LENGTH)
    out = np.empty(len(codes), dtype=np.int8)
    for start in range(0, len(codes), chunk):
        block = codes[start:start + chunk]
        dist = (block[:, None, :] != train_codes[None, :, :]).sum(axis=2)
        out[start:start + len(block)] = dist.min(axis=1)
    return pd.Series(out, index=pd.Index(all_peptides, name="peptide"),
                     name="dist_to_train")


def min_cross_split_distance(df: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Minimum peptide Hamming distance between each pair of splits.

    The grouping guarantee is that every train-vs-held-out entry here is >= 4.
    """
    per_split = {
        name: sorted(df.loc[df["split"] == name, "peptide"].unique())
        for name in SPLIT_NAMES
    }
    result: dict[tuple[str, str], int] = {}
    for i, a in enumerate(SPLIT_NAMES):
        for b in SPLIT_NAMES[i + 1:]:
            ca = encode_sequences(per_split[a], PEPTIDE_LENGTH)
            cb = encode_sequences(per_split[b], PEPTIDE_LENGTH)
            best = PEPTIDE_LENGTH
            for start in range(0, len(ca), 512):
                block = ca[start:start + 512]
                dist = (block[:, None, :] != cb[None, :, :]).sum(axis=2)
                best = min(best, int(dist.min()))
            result[(a, b)] = best
    return result
