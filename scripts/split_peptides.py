#!/usr/bin/env python3
"""Freeze the peptide-grouped fold assignments for the pHLA stability data.

    peptides -> BLOSUM62 similarity graph -> connected components -> balanced folds

A component is indivisible: every measurement of every peptide in it lands in
the same fold, across all alleles. Components, not pairs, so that if A resembles
B and B resembles C all three travel together. Folds are balanced on row count,
not peptide count, because a peptide screened on 36 alleles carries 36
measurements with it. Assignment is largest-component-first into the emptiest
fold, so there is no seed to defend.

Run against the raw CSV, which is a superset of every derived dataset. The
resulting table therefore covers `data/c67s_cleanup/rasmussen_no_C67S.csv` and
`data/c67s_cleanup/benchmark_C67S.csv` as well. Join downstream on **(allele, peptide)**,
which is unique in all three files. `row_id` is a positional index into this
script's input only, so it does not survive row removal -- do not join on it
unless the input is the raw CSV.

Ten folds carry the 70/10/20 train/validation/test partition fixed in stage 1
of HACKATHON_PLAN.md: folds 0-6 train, fold 7 validation, folds 8-9 test. That
assignment is materialised in a `split` column, so downstream code filters on
`split == "train"` rather than re-deriving which fold means what.

    python3 scripts/split_peptides.py
    python3 scripts/split_peptides.py --input data/c67s_cleanup/rasmussen_no_C67S.csv --threshold 0.70
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "data" / "rasmussen_et_al_dataset.csv"
DEFAULT_OUTDIR = REPO_ROOT / "data" / "c67s_cleanup"

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

# HACKATHON_PLAN.md stage 1: approximately 70/10/20, test sealed until stage 6.
DEFAULT_FOLDS = 10
DEFAULT_VAL_FOLD = 7
DEFAULT_TEST_FOLDS = (8, 9)


def encode(peptides: np.ndarray) -> np.ndarray:
    code = {a: i for i, a in enumerate(ALPHABET)}
    return np.array([[code[c] for c in p] for p in peptides], dtype=np.int8)


def similarity_matrix(encoded: np.ndarray) -> np.ndarray:
    """Normalised BLOSUM62 similarity; 1.0 on the diagonal, symmetric."""
    blosum = substitution_matrices.load("BLOSUM62")
    lookup = np.array([[blosum[a][b] for b in ALPHABET] for a in ALPHABET], dtype=np.int16)

    score = np.zeros((len(encoded),) * 2, dtype=np.int32)
    for position in range(encoded.shape[1]):
        column = encoded[:, position]
        score += lookup[column[:, None], column[None, :]]

    self_score = np.diag(score).astype(np.float32)
    return score / np.sqrt(np.outer(self_score, self_score))


def substitution_matrix_counts(encoded: np.ndarray) -> np.ndarray:
    """Pairwise substitution count. All peptides are 9-mers, so this is Hamming."""
    distance = np.zeros((len(encoded),) * 2, dtype=np.int8)
    for position in range(encoded.shape[1]):
        column = encoded[:, position]
        distance += (column[:, None] != column[None, :]).astype(np.int8)
    return distance


def similarity_components(similarity: np.ndarray, threshold: float) -> np.ndarray:
    i, j = np.nonzero(np.triu(similarity >= threshold, 1))
    graph = coo_matrix((np.ones(len(i)), (i, j)), shape=similarity.shape)
    _, labels = connected_components(graph, directed=False)
    return labels


def balance_into_folds(component_rows: pd.Series, n_folds: int) -> dict[int, int]:
    """Largest component first into the currently emptiest fold (LPT heuristic)."""
    load = np.zeros(n_folds, dtype=np.int64)
    assignment: dict[int, int] = {}
    for component, n_rows in component_rows.sort_values(ascending=False).items():
        target = int(np.argmin(load))
        assignment[component] = target
        load[target] += n_rows
    return assignment


def threshold_justification(similarity: np.ndarray, distance: np.ndarray, threshold: float) -> dict:
    """Does `threshold` sit below every one-substitution pair in this dataset?

    This is what picks the threshold. HACKATHON_PLAN.md stage 1 requires that
    one-residue neighbours never straddle a fold boundary; a threshold below the
    weakest one-substitution pair guarantees every such pair becomes an edge.
    """
    upper = np.triu_indices(len(similarity), 1)
    sim, dist = similarity[upper], distance[upper]
    report = {"threshold": threshold}
    for d in (1, 2, 3):
        pairs = dist == d
        if not pairs.any():
            continue
        report[f"{d}_substitution_pairs"] = {
            "n_pairs": int(pairs.sum()),
            "min_similarity": round(float(sim[pairs].min()), 4),
            "n_below_threshold": int((sim[pairs] < threshold).sum()),
        }
    report["captures_all_1_substitution_pairs"] = bool(
        (sim[dist == 1] >= threshold).all() if (dist == 1).any() else True)
    return report


def crossfold_leakage(similarity, distance, peptide_fold, n_folds, threshold) -> dict:
    """How close does a held-out peptide get to the nearest training peptide?

    The similarity figure is a consistency check, not a discovery: components are
    the connected components of the >= threshold graph, so any cross-fold pair at
    or above the threshold would mean the grouping code is broken. The
    substitution counts are the informative numbers -- they are measured on a
    scale that did not define the groups.
    """
    report = {}
    for fold in range(n_folds):
        test = np.flatnonzero(peptide_fold == fold)
        train = np.flatnonzero(peptide_fold != fold)
        block = similarity[np.ix_(test, train)]
        nearest = distance[np.ix_(test, train)].min(axis=1)
        report[fold] = {
            "max_test_to_train_similarity": round(float(block.max()), 4),
            "n_test_peptides_at_or_above_threshold": int((block.max(axis=1) >= threshold).sum()),
            "n_test_peptides_within_1_substitution": int((nearest <= 1).sum()),
            "n_test_peptides_within_2_substitutions": int((nearest <= 2).sum()),
        }
    return report


def partition_leakage(similarity, distance, peptide_split, threshold) -> dict:
    """The same check at the level that actually matters: test against everything
    it was not allowed to see, and validation against train."""
    report = {}
    for held_out, seen in (("test", ("train", "val")), ("val", ("train",))):
        a = np.flatnonzero(peptide_split == held_out)
        b = np.flatnonzero(np.isin(peptide_split, seen))
        if not len(a) or not len(b):
            continue
        report[f"{held_out}_vs_{'+'.join(seen)}"] = {
            "n_held_out_peptides": int(len(a)),
            "max_similarity": round(float(similarity[np.ix_(a, b)].max()), 4),
            "n_at_or_above_threshold": int((similarity[np.ix_(a, b)].max(axis=1) >= threshold).sum()),
            "n_within_1_substitution": int((distance[np.ix_(a, b)].min(axis=1) <= 1).sum()),
            "n_within_2_substitutions": int((distance[np.ix_(a, b)].min(axis=1) <= 2).sum()),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--val-fold", type=int, default=DEFAULT_VAL_FOLD)
    parser.add_argument("--test-folds", type=int, nargs="+", default=list(DEFAULT_TEST_FOLDS))
    args = parser.parse_args()

    roles = {args.val_fold: "val", **{f: "test" for f in args.test_folds}}
    if not set(roles) <= set(range(args.folds)):
        parser.error(f"--val-fold/--test-folds must be in 0..{args.folds - 1}")
    if len(args.test_folds) + 1 != len(roles):
        parser.error("--val-fold cannot also be a test fold")

    path: Path = args.input
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "row_id" not in df:
        df.insert(0, "row_id", np.arange(len(df), dtype=np.int32))
    assert not df.duplicated(["allele", "peptide"]).any(), "(allele, peptide) is not a unique key"

    peptides = np.array(sorted(df.peptide.unique()))
    assert len(peptides) < 20000, "dense similarity matrix is too large; switch to blocked search"
    assert len(set(map(len, peptides))) == 1, "substitution counts assume equal-length peptides"

    encoded = encode(peptides)
    similarity = similarity_matrix(encoded)
    distance = substitution_matrix_counts(encoded)
    component = pd.Series(similarity_components(similarity, args.threshold), index=peptides)

    df["peptide_group"] = df.peptide.map(component).astype(np.int32)
    rows_per_component = df.peptide_group.value_counts()
    fold_of = balance_into_folds(rows_per_component, args.folds)
    df["fold"] = df.peptide_group.map(fold_of).astype(np.int8)

    df["split"] = df.fold.map(roles).fillna("train")

    peptide_fold = component.map(fold_of).to_numpy()
    spanning = df.groupby("peptide_group").fold.nunique()
    assert (spanning == 1).all(), "a peptide group was split across folds"

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    splits = df[["row_id", "allele", "peptide", "peptide_group", "fold", "split"]]
    splits.to_csv(outdir / "peptide_splits.csv", index=False)

    rows_per_fold = df.fold.value_counts()
    group_sizes = component.value_counts()
    manifest = {
        "input": str(path.resolve().relative_to(REPO_ROOT) if path.resolve().is_relative_to(REPO_ROOT)
                     else path.name),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "threshold": args.threshold,
        "n_folds": args.folds,
        "partition": {
            "train_folds": sorted(set(range(args.folds)) - set(roles)),
            "val_fold": args.val_fold,
            "test_folds": sorted(args.test_folds),
            "rows": df.split.value_counts().to_dict(),
            "pct": {k: round(100 * v / len(df), 1) for k, v in df.split.value_counts().items()},
        },
        "n_rows": int(len(df)),
        "n_peptides": int(len(peptides)),
        "n_groups": int(component.nunique()),
        "largest_group_peptides": int(group_sizes.max()),
        "largest_group_rows": int(rows_per_component.max()),
        "rows_per_fold": rows_per_fold.sort_index().to_dict(),
        "peptides_per_fold": pd.Series(peptide_fold).value_counts().sort_index().to_dict(),
        "fold_imbalance_pct": round(
            100 * (rows_per_fold.max() - rows_per_fold.min()) / len(df) * args.folds, 2),
        "threshold_justification": threshold_justification(similarity, distance, args.threshold),
        "partition_leakage_check": partition_leakage(
            similarity, distance,
            np.array([roles.get(f, "train") for f in peptide_fold]), args.threshold),
        "per_fold_leakage_check": crossfold_leakage(
            similarity, distance, peptide_fold, args.folds, args.threshold),
    }
    (outdir / "peptide_splits_manifest.json").write_text(json.dumps(manifest, indent=2, default=int))
    print(json.dumps(manifest, indent=2, default=int))


if __name__ == "__main__":
    main()
