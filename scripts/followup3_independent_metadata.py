"""Reproduce conservative candidate provenance without retaining outcome fields.

Standard-library metadata check only; no fitting, scoring or remote compute.
CSV readers immediately project explicit identity/metadata allowlists. Canonical
identity is projected directly because load_with_splits eagerly loads outcomes.
The original audit and its outputs are untouched.
"""
from collections import Counter
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/followup3/independent_data"
ALLOW = {"peptide_sequence", "mhc_sequence", "allele", "assay", "temperature_C", "pmid", "ref_iri"}
AA = set("ACDEFGHIKLMNPQRSTVWY")


def projection(path, allow):
    with path.open() as handle:
        for row in csv.DictReader(handle):
            yield {key: value for key, value in row.items() if key in allow}


def valid(peptide):
    return len(peptide) == 9 and set(peptide) <= AA


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT / "reports/further_research/round2/independent_data/source_manifest.json").read_text())
    canonical = ROOT / "data/rasmussen_et_al_dataset.csv"
    assert digest(canonical) == manifest["canonical_source_sha256"]
    seeds = {r["peptide"] for r in projection(canonical, {"peptide", "allele"}) if valid(r["peptide"])}
    source_records = []
    test_rows = None
    for source in manifest["sources"]:
        path = ROOT / "external/round2/spearmint" / source["source_path"]
        assert digest(path) == source["sha256"], source["source"]
        rows = list(projection(path, ALLOW))
        source_records.append({"source": source["source"], "sha256": source["sha256"], "rows": len(rows)})
        if source["source"] == "stage3_assay_conditioning/test":
            test_rows = rows
        else:
            seeds.update(r["peptide_sequence"] for r in rows if valid(r["peptide_sequence"]))
    assert test_rows is not None
    candidates = sorted({r["peptide_sequence"] for r in test_rows if valid(r["peptide_sequence"])})
    parent = list(range(len(candidates)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    hits = set()
    # All seeds are excluded. Seed-to-seed edges cannot affect candidate
    # eligibility, so only seed adjacency and candidate connectivity are needed.
    for positions in combinations(range(9), 6):
        signatures = {"".join(p[j] for j in positions) for p in seeds}
        first = {}
        for i, peptide in enumerate(candidates):
            signature = "".join(peptide[j] for j in positions)
            if signature in signatures:
                hits.add(i)
            if signature in first:
                a, b = find(i), find(first[signature])
                parent[max(a, b)] = min(a, b)
            else:
                first[signature] = i
    excluded = {find(i) for i in hits}
    eligible = {p for i, p in enumerate(candidates) if find(i) not in excluded}
    retained = [(i, r) for i, r in enumerate(test_rows) if r["peptide_sequence"] in eligible]
    assert len(retained) == 129 and len(eligible) == 126
    counts = Counter((r["pmid"].removesuffix(".0"), r["ref_iri"], r["assay"], r["temperature_C"], r["allele"]) for _, r in retained)
    groups = [dict(zip(["pmid", "ref_iri", "release_assay", "release_temperature_C", "allele"], key), retained_rows=value) for key, value in sorted(counts.items())]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "candidate_source_groups.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(groups[0]))
        writer.writeheader()
        writer.writerows(groups)
    row_manifest = []
    for i, r in retained:
        row_manifest.append({"source_row_zero_based": i, "pair_identity_sha256": hashlib.sha256((r["peptide_sequence"] + "|" + r["allele"]).encode()).hexdigest(), "mhc_input_sha256": hashlib.sha256(r["mhc_sequence"].encode()).hexdigest(), **{k: r[k] for k in ["allele", "assay", "temperature_C", "pmid", "ref_iri"]}})
    summary = {"checked_date": "2026-10-13", "policy": "canonical + all affinity/stability partitions + stage3 train/val; exact standard 9mer Hamming <=3 components", "outcome_policy": "No outcome column retained, displayed, summarized, scored or used for eligibility. CSV raw row objects exist only long enough to allowlist metadata. Literature visibility is documented separately in PROVENANCE.md.", "canonical_sha256": manifest["canonical_source_sha256"], "sources": source_records, "rows": len(retained), "peptides": len(eligible), "components": len({find(i) for i, p in enumerate(candidates) if p in eligible}), "allele_counts": dict(Counter(r["allele"] for _, r in retained)), "source_pmids": sorted({g["pmid"] for g in groups}), "row_manifest": row_manifest}
    assert summary["components"] == 113 and len(summary["allele_counts"]) == 9
    (OUT / "candidate_metadata_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ["rows", "peptides", "components", "allele_counts"]}, indent=2))


if __name__ == "__main__":
    main()
