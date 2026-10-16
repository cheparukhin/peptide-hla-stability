"""Outcome-blind auxiliary HLA and joint holdout manifests on original train.

This freezes grouping and coverage, not transfer candidate choices or results.
Fitting remains gated on completion of the peptide-held representation screen.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pepstab import nested
from pepstab.data import encode_sequences, load_with_splits
from pepstab.splits import single_linkage_clusters

OUT = ROOT / "reports/further_research/round2/transfer"


def grouped_hlas(frame, threshold=3):
    """Connect pseudosequences within three substitutions, including constructs."""
    alleles = frame[["allele", "hla_pseudoseq"]].drop_duplicates().sort_values("allele").reset_index(drop=True)
    if alleles.allele.duplicated().any():
        raise ValueError("allele has inconsistent pseudosequence")
    codes = encode_sequences(alleles.hla_pseudoseq.tolist(), 34)
    groups = single_linkage_clusters(codes, threshold=threshold)
    # Explicitly join engineered and wild-type constructs, even if a future
    # construct differs at more positions than the sequence threshold.
    parents = alleles.allele.str.replace(r"\(.*\)$", "", regex=True)
    for parent in parents.unique():
        mask = parents.eq(parent).to_numpy()
        members = np.unique(groups[mask])
        groups[np.isin(groups, members)] = members.min()
    names = {value: index for index, value in enumerate(sorted(set(groups)))}
    alleles["hla_group"] = [names[g] for g in groups]
    alleles["rows"] = alleles.allele.map(frame.allele.value_counts()).astype(int)
    return alleles


def balanced_group_folds(groups, n_folds=5):
    weights = groups.groupby("hla_group").rows.sum()
    if len(weights) < n_folds:
        raise ValueError("insufficient HLA groups for frozen fold count")
    loads = np.zeros(n_folds, dtype=int)
    assignment = {}
    for group in sorted(weights.index, key=lambda g: (-weights[g], g)):
        fold = int(np.argmin(loads))
        assignment[group] = fold
        loads[fold] += int(weights[group])
    return assignment


def regime_masks(frame, regime, hla_fold, peptide_fold=None):
    h = frame.hla_fold.to_numpy() == hla_fold
    if regime == "hla":
        return ~h, h
    if regime != "joint" or peptide_fold is None:
        raise ValueError("joint holdout requires a peptide fold")
    p = frame.peptide_fold.to_numpy() == peptide_fold
    return ~h & ~p, h & p


def build(frame):
    if not frame.split.eq("train").all():
        raise ValueError("original training rows only")
    alleles = grouped_hlas(frame)
    mapping = balanced_group_folds(alleles)
    alleles["hla_fold"] = alleles.hla_group.map(mapping)
    manifest = frame.merge(alleles[["allele", "hla_group", "hla_fold"]], on="allele", validate="many_to_one", sort=False)
    if not np.array_equal(manifest.pair_id, frame.pair_id):
        raise ValueError("row order changed")
    manifest["peptide_fold"] = nested.outer_folds(frame)
    coverage = []
    for regime in ("hla", "joint"):
        for h in range(5):
            for p in ([None] if regime == "hla" else range(5)):
                fit_mask, held_mask = regime_masks(manifest, regime, h, p)
                fit, held = manifest[fit_mask], manifest[held_mask]
                if set(fit.hla_group) & set(held.hla_group):
                    raise AssertionError("HLA group leakage")
                if regime == "joint" and set(fit.cluster_id) & set(held.cluster_id):
                    raise AssertionError("joint peptide leakage")
                peptide_codes = encode_sequences(sorted(set(fit.peptide) | set(held.peptide)), 9)
                peptide_names = sorted(set(fit.peptide) | set(held.peptide))
                pos = {s: i for i, s in enumerate(peptide_names)}
                # Only the held-by-fit matrix is needed; no labels enter grouping.
                a = peptide_codes[[pos[s] for s in held.peptide.unique()]]
                b = peptide_codes[[pos[s] for s in fit.peptide.unique()]]
                minimum = np.concatenate([(a[start:start+128, None] != b[None]).sum(2).min(1)
                                          for start in range(0, len(a), 128)])
                hdist = (encode_sequences(held.hla_pseudoseq.unique(), 34)[:, None] !=
                         encode_sequences(fit.hla_pseudoseq.unique(), 34)[None]).sum(2).min(1)
                counts = held.groupby("allele").agg(rows=("pair_id", "size"), labels=("y_log1p", "nunique"))
                coverage.append({"regime": regime, "hla_fold": h, "peptide_fold": p,
                    "fit_rows": len(fit), "held_rows": len(held), "excluded_cross_quadrant_rows": len(frame)-len(fit)-len(held),
                    "held_hla_groups": held.hla_group.nunique(), "held_alleles": held.allele.nunique(),
                    "eligible_alleles_20": int(((counts.rows >= 20) & (counts.labels >= 2)).sum()),
                    "eligible_alleles_50": int(((counts.rows >= 50) & (counts.labels >= 2)).sum()),
                    "held_floor_fraction": float(held.thalf_hours.eq(0).mean()),
                    "exact_peptide_overlap_rows": int(held.peptide.isin(set(fit.peptide)).sum()),
                    "cluster_overlap_rows": int(held.cluster_id.isin(set(fit.cluster_id)).sum()),
                    "minimum_peptide_hamming": int(minimum.min()), "median_nearest_peptide_hamming": float(np.median(minimum)),
                    "minimum_pseudosequence_hamming": int(hdist.min())})
    return alleles, manifest, pd.DataFrame(coverage)


def main():
    frame = load_with_splits().query("split=='train'").sort_values("pair_id").reset_index(drop=True)
    alleles, manifest, coverage = build(frame)
    OUT.mkdir(parents=True, exist_ok=True)
    alleles.to_csv(OUT / "hla_groups.csv", index=False)
    manifest[["pair_id", "cluster_id", "allele", "hla_group", "hla_fold", "peptide_fold"]].to_csv(OUT / "manifest.csv", index=False)
    coverage.to_csv(OUT / "coverage.csv", index=False)
    spec = {"status": "grouping_frozen; transfer candidates and fitting pending main screen",
        "canonical_plan": "reports/NEXT_RESEARCH_PLAN.md", "population": "original training partition only",
        "hla_groups": int(alleles.hla_group.nunique()), "hla_threshold": "single-linkage Hamming <=3 over 34-residue pseudosequences",
        "construct_rule": "engineered and matching wild-type names share a group",
        "folds": "five HLA folds; largest-group-first balanced row count, deterministic group-id ties",
        "peptide_folds": "unchanged canonical five training-cluster outer folds",
        "joint": "all 25 HLA x peptide fold intersections; fit outside BOTH, exclude cross quadrants",
        "inner_requirement": "mirror HLA or joint exclusions in selection and stopping; peptide-only inner folds are insufficient",
        "uncertainty_requirement": "paired peptide-cluster and HLA-group uncertainty; crossed resampling for joint; sparse-group sensitivity",
        "limitations": ["threshold is a declared sequence convention, not a validated functional HLA distance",
                         "five folds do not imply five biological groups; uncertainty uses actual HLA groups",
                         "manifest and label/count eligibility do not constitute assessed transfer results"],
        "sha256": {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in ("hla_groups.csv", "manifest.csv", "coverage.csv")}}
    (OUT / "protocol.json").write_text(json.dumps(spec, indent=2)+"\n")
    print(json.dumps({"hla_groups": spec["hla_groups"], "regime_cells": len(coverage)}))


if __name__ == "__main__":
    main()
