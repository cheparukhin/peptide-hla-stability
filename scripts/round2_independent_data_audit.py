"""Sequence-only provenance audit; external outcome columns are never loaded.

Run locally with .venv/bin/python scripts/round2_independent_data_audit.py.
Raw downloads are kept in ignored external/round2/. Reports contain aggregates
only. This audit does not establish sample provenance or checkpoint exposure.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pepstab.data import AMINO_ACIDS, RAW_CSV, SPLITS_CSV, load_with_splits

COMMIT = "d548dee9b04dd0e3a9d7a50a593d2c030e31d2a5"
BASE = f"https://raw.githubusercontent.com/dhuvik/spearmint/{COMMIT}/"
GROUPS = ("binding_stability", "binding_affinity", "stage3_assay_conditioning")
PARTITIONS = ("train", "val", "test")
EXPECTED_BLOBS = {
    "binding_stability/train": "3c50678c7aae5d7c0fbdd2523bb17c0b08add4d4",
    "binding_stability/val": "7ba309f000fd14e02b75bc4205604c15af541f41",
    "binding_stability/test": "f1813b5c65e257c546f2af5acf123f35a148dd9c",
    "binding_affinity/train": "cdb957dc8764a6583afbdb272993fb7a6074c3b4",
    "binding_affinity/val": "43ea13da60b26c311f813917e246e24ec000fa3f",
    "binding_affinity/test": "a31da36651b490b0749fa671315143a5b3716c5c",
    "stage3_assay_conditioning/train": "5f2bea5a37ad9b7e3d4c992258df28525ffc8338",
    "stage3_assay_conditioning/val": "0a20138a0ce3a53359603ce6c503e57677a7238f",
    "stage3_assay_conditioning/test": "527cbb7092e3f85d20de06d57fa03983667294a5",
}
METADATA_COLUMNS = {"peptide_sequence", "mhc_sequence", "allele", "assay",
                    "temperature_C", "pmid", "ref_iri"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_metadata(path: Path) -> pd.DataFrame:
    """An explicit allowlist prevents external labels entering the audit."""
    frame = pd.read_csv(path, usecols=lambda c: c in METADATA_COLUMNS,
                        dtype={"peptide_sequence": "string", "allele": "string",
                               "mhc_sequence": "string", "pmid": "string",
                               "ref_iri": "string"})
    if not {"peptide_sequence", "allele", "mhc_sequence"} <= set(frame):
        raise ValueError("source identity columns missing")
    if frame[["peptide_sequence", "allele", "mhc_sequence"]].isna().any().any():
        raise ValueError("missing source identity")
    return frame.rename(columns={"peptide_sequence": "peptide"})


def valid_ninemers(peptides) -> set[str]:
    return {str(p) for p in peptides if len(str(p)) == 9
            and set(str(p)) <= set(AMINO_ACIDS)}


def signature_codes(peptides: list[str]) -> np.ndarray:
    if any(len(p) != 9 or not set(p) <= set(AMINO_ACIDS) for p in peptides):
        raise ValueError("Hamming audit accepts standard-amino-acid 9-mers only")
    return np.frombuffer("".join(peptides).encode("ascii"), dtype=np.uint8).reshape(-1, 9)


def signatures(codes: np.ndarray, positions) -> np.ndarray:
    # Six ASCII bytes fit exactly in uint64; collisions are impossible.
    sig = np.zeros(len(codes), dtype=np.uint64)
    for pos in positions:
        sig = (sig << np.uint64(8)) | codes[:, pos].astype(np.uint64)
    return sig


def within_three(query: list[str], reference: list[str]) -> np.ndarray:
    """Exact Hamming <=3 indicator without a quadratic distance matrix."""
    qc, rc = signature_codes(query), signature_codes(reference)
    near = np.zeros(len(query), dtype=bool)
    if not len(reference):
        return near
    for pos in combinations(range(9), 6):
        near |= np.isin(signatures(qc, pos), signatures(rc, pos))
    return near


def component_map(peptides: set[str]) -> dict[str, int]:
    """Full union-graph components, including transitive donor bridges.

    Two 9-mers differ at <=3 positions iff at least one of the 84 six-position
    signatures agrees. Joining members of each signature group yields exactly
    the Hamming <=3 graph components, with no approximate neighbour search.
    """
    ordered = sorted(peptides)
    codes = signature_codes(ordered)
    parent = list(range(len(ordered)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for pos in combinations(range(9), 6):
        sig = signatures(codes, pos)
        order = np.argsort(sig, kind="stable")
        for k in np.flatnonzero(sig[order][1:] == sig[order][:-1]):
            a, b = find(int(order[k])), find(int(order[k + 1]))
            if a != b:
                parent[max(a, b)] = min(a, b)
    return {p: find(i) for i, p in enumerate(ordered)}


def audit_frame(name: str, donor: pd.DataFrame, canonical: pd.DataFrame,
                components: dict[str, int]) -> tuple[dict, list[dict]]:
    cp = set(canonical.peptide)
    dp = valid_ninemers(donor.peptide)
    pairs = set(zip(canonical.peptide, canonical.allele))
    donor_pairs = list(zip(donor.peptide, donor.allele))
    # Allele strings are exact: engineered suffixes are never collapsed.
    exact = np.array([pair in pairs for pair in donor_pairs])
    near_peptides = set(p for p, hit in zip(sorted(dp), within_three(sorted(dp), sorted(cp))) if hit)
    canonical_roots = {components[p] for p in cp}
    connected_peptides = {p for p in dp if components[p] in canonical_roots}
    result = {
        "source": name, "rows": len(donor), "alleles": donor.allele.nunique(),
        "unique_pairs": len(set(donor_pairs)), "unique_peptides_all_lengths": donor.peptide.nunique(),
        "standard_9mer_rows": int(donor.peptide.isin(dp).sum()), "standard_9mer_peptides": len(dp),
        "exact_pair_overlap_rows": int(exact.sum()),
        "exact_peptide_overlap_rows": int(donor.peptide.isin(cp).sum()),
        "direct_hamming_le3_overlap_rows": int(donor.peptide.isin(near_peptides).sum()),
        "component_overlap_rows": int(donor.peptide.isin(connected_peptides).sum()),
        "component_novel_9mer_rows": int(donor.peptide.isin(dp - connected_peptides).sum()),
        "non_9mer_or_nonstandard_rows": int((~donor.peptide.isin(dp)).sum()),
        "duplicate_pair_rows": len(donor) - len(set(donor_pairs)),
    }
    reverse = []
    for split in PARTITIONS:
        part = canonical[canonical.split == split]
        pep = set(part.peptide)
        roots = {components[p] for p in dp}
        con = {p for p in pep if components[p] in roots}
        near = set(p for p, hit in zip(sorted(pep), within_three(sorted(pep), sorted(dp))) if hit)
        donor_pair_set = set(donor_pairs)
        reverse.append({"source": name, "canonical_split": split, "canonical_rows": len(part),
                        "canonical_peptides": len(pep), "canonical_clusters": part.cluster_id.nunique(),
                        "exact_pair_overlap_rows": sum(pair in donor_pair_set for pair in zip(part.peptide, part.allele)),
                        "exact_peptide_overlap_rows": int(part.peptide.isin(dp).sum()),
                        "direct_hamming_le3_overlap_rows": int(part.peptide.isin(near).sum()),
                        "component_overlap_rows": int(part.peptide.isin(con).sum()),
                        "frozen_clusters_component_overlap": part.loc[part.peptide.isin(con), "cluster_id"].nunique()})
    return result, reverse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-dir", type=Path, default=ROOT / "external/round2/spearmint")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/further_research/round2/independent_data")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    sources, frames, metadata = [], {}, []
    for group in GROUPS:
        for partition in PARTITIONS:
            source_path = f"data/{group}/{partition}.csv"
            dest = args.external_dir / source_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.write_bytes(urllib.request.urlopen(BASE + source_path, timeout=90).read())
            name = f"{group}/{partition}"
            raw = dest.read_bytes()
            blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
            if blob != EXPECTED_BLOBS[name]:
                raise ValueError(f"pinned source blob mismatch: {name}")
            frame = read_metadata(dest)
            frames[name] = frame
            sources.append({"source": name, "source_path": source_path, "url": BASE + source_path,
                            "sha256": sha256(dest), "bytes": dest.stat().st_size, "rows": len(frame),
                            "verified_at_utc": now, "git_blob_sha1": blob,
                            "label_policy": "column label not loaded"})
            item = {"source": name,
                    "peptide_lengths": {str(k): int(v) for k, v in frame.peptide.str.len().value_counts().sort_index().items()},
                    "mhc_sequence_lengths": {str(k): int(v) for k, v in frame.mhc_sequence.str.len().value_counts().sort_index().items()}}
            for col in ("assay", "temperature_C"):
                if col in frame:
                    item[col] = {str(k): int(v) for k, v in frame[col].value_counts(dropna=False).items()}
            for col in ("pmid", "ref_iri"):
                if col in frame:
                    item[col + "_nonmissing_rows"] = int(frame[col].notna().sum())
                    item[col + "_distinct"] = int(frame[col].nunique())
            metadata.append(item)

    canonical = load_with_splits()[["pair_id", "peptide", "allele", "cluster_id", "split", "hla_seq"]]
    summary, reverse = [], []
    for group in GROUPS:
        pooled = pd.concat([frames[f"{group}/{p}"] for p in PARTITIONS], ignore_index=True)
        frames[f"{group}/all"] = pooled
        components = component_map(set(canonical.peptide) | valid_ninemers(pooled.peptide))
        for partition in (*PARTITIONS, "all"):
            name = f"{group}/{partition}"
            one, rev = audit_frame(name, frames[name], canonical, components)
            # Components include all published partitions of this corpus. Direct
            # distances and exact matches use only the stated partition.
            one["component_graph"] = f"canonical + {group}/all"
            summary.append(one)
            reverse.extend(rev)
        print(f"Audited {group}: {len(pooled):,} source rows", flush=True)

    stage3 = frames["stage3_assay_conditioning/all"].copy()
    policy_graphs = {}
    for policy, partitions in (("train_val", ("train", "val")), ("all_partitions", PARTITIONS)):
        # Candidate rows are not seeds. They can bridge into the graph but do
        # not exclude themselves merely because they are present in the graph.
        upstream_names = [f"{g}/{p}" for g in GROUPS[:2] for p in partitions]
        upstream_names += [f"stage3_assay_conditioning/{p}" for p in ("train", "val")]
        upstream = pd.concat([frames[name] for name in upstream_names], ignore_index=True)
        seeds = set(canonical.peptide) | valid_ninemers(upstream.peptide)
        candidates = valid_ninemers(stage3.peptide)
        graph = component_map(seeds | candidates)
        policy_graphs[policy] = graph
        seed_roots = {graph[p] for p in seeds}
        eligible = {p for p in candidates if graph[p] not in seed_roots}
        stage3["eligible_" + policy] = stage3.peptide.isin(eligible)
    eligibility, strata = [], []
    for partition in (*PARTITIONS, "all"):
        if partition == "all":
            selected = stage3
        else:
            start = sum(len(frames[f"stage3_assay_conditioning/{p}"]) for p in PARTITIONS[:PARTITIONS.index(partition)])
            selected = stage3.iloc[start:start + len(frames[f"stage3_assay_conditioning/{partition}"])]
        # Sequence-only counts, never output identities or outcome summaries.
        for policy in ("train_val", "all_partitions"):
            kept = selected[selected["eligible_" + policy]]
            counts = kept.groupby("allele").size()
            eligibility.append({"partition": partition, "upstream_exclusion": policy, "rows": len(selected),
                                "eligible_9mer_rows": len(kept), "eligible_peptides": kept.peptide.nunique(),
                                "eligible_alleles": kept.allele.nunique(),
                                "eligible_components": len({policy_graphs[policy][p] for p in kept.peptide}),
                                "eligible_unique_pairs": len(kept[["peptide", "allele"]].drop_duplicates()),
                                "alleles_at_least_10_rows": int((counts >= 10).sum()),
                                "alleles_at_least_20_rows": int((counts >= 20).sum()),
                                "unseen_canonical_allele_rows": int((~kept.allele.isin(canonical.allele)).sum()),
                                "eligible_missing_pmid_rows": int(kept.pmid.isna().sum()),
                                "eligible_missing_ref_iri_rows": int(kept.ref_iri.isna().sum())})
            for (assay, temp), part in kept.groupby(["assay", "temperature_C"], dropna=False):
                strata.append({"partition": partition, "upstream_exclusion": policy,
                               "assay": assay, "temperature_C": temp, "eligible_rows": len(part),
                               "eligible_peptides": part.peptide.nunique(), "eligible_alleles": part.allele.nunique()})
    pd.DataFrame(summary).to_csv(args.output_dir / "overlap_summary.csv", index=False)
    pd.DataFrame(reverse).to_csv(args.output_dir / "canonical_exposure.csv", index=False)
    pd.DataFrame(eligibility).to_csv(args.output_dir / "confirmation_eligibility.csv", index=False)
    pd.DataFrame(strata).to_csv(args.output_dir / "confirmation_assay_strata.csv", index=False)
    (args.output_dir / "assay_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    manifest = {"verified_at_utc": now, "commit": COMMIT,
                "canonical_source_sha256": sha256(RAW_CSV), "frozen_splits_sha256": sha256(SPLITS_CSV),
                "script_sha256": sha256(Path(__file__)), "sources": sources,
                "component_rule": "exact Hamming <=3 union graph for standard 9-mers only",
                "labels": "external labels stored only in ignored downloaded raw files; not read, summarized, used for fitting or scored"}
    for name in ("README.md", "LICENSE"):
        dest = args.external_dir / name
        if not dest.exists():
            dest.write_bytes(urllib.request.urlopen(BASE + name, timeout=90).read())
        manifest[name] = {"url": BASE + name, "sha256": sha256(dest)}
    (args.output_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(pd.DataFrame(eligibility).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
