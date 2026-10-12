"""Freeze a whole-cluster original-training cohort before supervised outcomes."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pepstab.data import load_with_splits
from pepstab.boltz_embeddings import json_bytes, sha256_file


def freeze_cohort(output: Path, max_rows: int = 0) -> dict:
    data = load_with_splits()
    data = data.loc[data.split.eq("train"), ["pair_id", "cluster_id", "allele", "peptide", "split"]]
    counts = data.groupby("cluster_id").size().to_dict()
    ordered = sorted(counts, key=lambda c: hashlib.sha256(f"round2-20261012:{c}".encode()).hexdigest())
    cap = max_rows or len(data)
    selected, used = [], 0
    for cluster in ordered:
        if used + counts[cluster] <= cap:
            selected.append(cluster)
            used += counts[cluster]
    cohort = data[data.cluster_id.isin(selected)].sort_values("pair_id")
    if cohort.empty:
        raise ValueError("row cap cannot accommodate any whole training cluster")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "cohort.csv"
    if path.exists():
        raise FileExistsError("cohort already frozen; use a new named study directory")
    cohort.to_csv(path, index=False)
    manifest = {"plan": "reports/NEXT_RESEARCH_PLAN.md", "cohort_kind": "full_training" if len(cohort) == len(data) else "reduced_whole_training_clusters",
                "selection_rule": "sort whole training clusters by SHA256('round2-20261012:<cluster_id>'); greedily accept a cluster iff all its rows fit the fixed cap",
                "selection_uses_labels": False, "row_cap": cap, "n_rows": len(cohort),
                "n_clusters": int(cohort.cluster_id.nunique()), "n_alleles": int(cohort.allele.nunique()),
                "source_training_rows": len(data), "source_training_clusters": len(counts),
                "cohort_sha256": sha256_file(path), "source_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                    (ROOT / "data/splits.csv", ROOT / "data/rasmussen_et_al_dataset.csv", ROOT / "structures/ectodomain_msas/manifest.json")}}
    (output / "cohort_manifest.json").write_bytes(json_bytes(manifest))
    return manifest


def cases_for_cohort(cohort_csv: Path) -> list[dict]:
    import pandas as pd
    cohort_csv = cohort_csv.resolve()
    frozen = json.loads((cohort_csv.parent / "cohort_manifest.json").read_text())
    if sha256_file(cohort_csv) != frozen["cohort_sha256"]:
        raise ValueError("frozen cohort changed")
    for relative, expected in frozen["source_hashes"].items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError(f"source changed: {relative}")
    requested = pd.read_csv(cohort_csv)
    canonical = load_with_splits()
    canonical = canonical.loc[canonical.split.eq("train"), ["pair_id", "cluster_id", "allele", "peptide", "split", "hla_seq"]]
    selected = canonical[canonical.pair_id.isin(requested.pair_id)].sort_values("pair_id")
    if (len(selected) != len(requested) or set(canonical.loc[canonical.cluster_id.isin(selected.cluster_id), "pair_id"]) != set(selected.pair_id)):
        raise ValueError("cohort must contain complete original-training clusters")
    for col in ("pair_id", "cluster_id", "allele", "peptide", "split"):
        if selected[col].astype(str).tolist() != requested.sort_values("pair_id")[col].astype(str).tolist():
            raise ValueError(f"cohort identity mismatch: {col}")
    msas = {m["allele"]: m for m in json.loads((ROOT / "structures/ectodomain_msas/manifest.json").read_text())["msas"]}
    cases = []
    for row in selected.itertuples(index=False):
        m = msas[row.allele]
        if m["ecto"][:182] != row.hla_seq:
            raise ValueError("construct groove differs from canonical sequence")
        cases.append({"complex_id": f"r2_{row.pair_id:05d}_d1024", "pair_id": int(row.pair_id),
                      "cluster_id": int(row.cluster_id), "allele": row.allele, "peptide": row.peptide,
                      "split": "train", "msa_depth_cap": 1024, "technical_depth_variant": False,
                      "construct_note": m["note"], "alpha3_source": m["a3_source"],
                      "chains": [{"id": "A", "sequence": m["ecto"], "msa": m["B"]},
                                 {"id": "B", "sequence": m["b2m"], "msa": m["beta2m"]},
                                 {"id": "C", "sequence": row.peptide, "msa": None}]})
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()
    if args.max_rows < 0:
        raise ValueError("max rows must be nonnegative")
    print(json.dumps(freeze_cohort(args.output, args.max_rows), indent=2))
