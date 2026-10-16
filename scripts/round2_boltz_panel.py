"""Freeze a label-blind training panel for NEXT_RESEARCH_PLAN milestone 1."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pepstab.data import load_with_splits
from pepstab.boltz_embeddings import json_bytes, sha256_file, sha256_bytes

OUT = ROOT / "reports/further_research/round2/extractor"


def build_panel(count=8, variants=True):
    if not 8 <= count <= 64:
        raise ValueError("engineering panel must have 8-64 unique cases")
    # Labels are deliberately dropped before selecting any row.
    frame = load_with_splits()
    frame = frame.loc[frame.split.eq("train"), ["pair_id", "cluster_id", "allele", "peptide", "hla_seq", "split"]]
    manifest_path = ROOT / "structures/ectodomain_msas/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    by_allele = {r["allele"]: r for r in manifest["msas"]}
    priority = ["HLA-A*02:01", "HLA-B*14:01(C67S)", "HLA-A*02:50", "HLA-B*07:02",
                "HLA-A*24:19", "HLA-B*39:06(C67S)", "HLA-B*08:03", "HLA-A*11:01"]
    priority += sorted(set(frame.allele) - set(priority))
    cases = []
    for allele in priority:
        candidates = frame.loc[frame.allele.eq(allele)].sort_values("pair_id")
        if candidates.empty or allele not in by_allele:
            continue
        row, m = candidates.iloc[0], by_allele[allele]
        if m["ecto"][:182] != row.hla_seq:
            raise ValueError("construct groove differs from canonical source")
        cases.append({"complex_id": f"r2_{int(row.pair_id):05d}_d1024", "pair_id": int(row.pair_id),
                      "cluster_id": int(row.cluster_id), "allele": allele, "peptide": row.peptide,
                      "split": "train", "msa_depth_cap": 1024, "technical_depth_variant": False,
                      "construct_note": m["note"], "alpha3_source": m["a3_source"],
                      "chains": [{"id": "A", "sequence": m["ecto"], "msa": m["B"]},
                                 {"id": "B", "sequence": m["b2m"], "msa": m["beta2m"]},
                                 {"id": "C", "sequence": row.peptide, "msa": None}]})
        if len(cases) == count:
            break
    if len(cases) != count:
        raise ValueError("insufficient eligible alleles")
    if variants:
        for base, depth in zip(cases[:2], (32, 128)):
            case = copy.deepcopy(base)
            case.update(complex_id=base["complex_id"].replace("d1024", f"d{depth}"),
                        msa_depth_cap=depth, technical_depth_variant=True)
            cases.append(case)
    return {"protocol": "reports/NEXT_RESEARCH_PLAN.md milestone 1", "selection": "label-blind allele/construct coverage; smallest canonical pair_id per allele",
            "training_only": True, "unique_pairs": count, "cases": cases,
            "msa_depth_note": "all production HLA MSAs have 1024 rows; smaller depths are explicitly labelled technical perturbations",
            "input_files": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                            (ROOT / "data/rasmussen_et_al_dataset.csv", ROOT / "data/splits.csv", manifest_path)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--no-variants", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT / "technical_panel.json")
    args = parser.parse_args()
    panel = build_panel(args.count, not args.no_variants)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json_bytes(panel))
    print(json.dumps({"path": str(args.output), "sha256": sha256_file(args.output),
                      "unique_pairs": panel["unique_pairs"], "cases": len(panel["cases"])}))


if __name__ == "__main__":
    main()
