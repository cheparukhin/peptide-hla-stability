"""Freeze the stage 4c production cohort and its two-workspace shard schedule.

Writes ``data/structural_cohort.csv``: every (allele, peptide) pair in the
frozen splits, in a deterministic order, with a fixed profile and shard
assignment. Run once; commit the result. The runner reads this file and never
re-derives it, so a resumed or re-launched run folds exactly the same pairs in
exactly the same shards.

Scope is the full dataset, arm B (ectodomain + beta2m + peptide), Boltz-2 only.
ESMFold2 failed its stage 4c pilot gate -- see reports/stage4c_ectodomain_pilot.md.

Splits are loaded, never recomputed, and joined on (allele, peptide); the raw
``pair_id`` is attached afterwards and is positional into the raw CSV only.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits.csv"
MANIFEST = ROOT / "structures" / "ectodomain_msas" / "manifest.json"
DEST = ROOT / "data" / "structural_cohort.csv"

# One shard is one `boltz predict` invocation. Model load is paid once per
# shard and showed up as ~57 s on top of the first fold in the pilot, so 100
# steady folds (~28 min) amortise it to ~3% while staying far inside the
# 60-minute function timeout.
SHARD_SIZE = 100

# Modal profile names as configured locally (`modal profile list`).
# `colleague` points at the `sofyaleyn` workspace.
PROFILES = ("a-cheparukhin", "colleague")


def complex_id(allele: str, peptide: str) -> str:
    short = (
        allele.replace("HLA-", "")
        .replace("*", "")
        .replace(":", "")
        .replace("(", "_")
        .replace(")", "")
    )
    return f"{short}_{peptide}"


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    by_allele = {m["allele"]: m for m in manifest["msas"]}

    rows = list(csv.DictReader(SPLITS.open()))
    missing = sorted({r["allele"] for r in rows} - by_allele.keys())
    assert not missing, f"no prepared ectodomain MSA for: {missing}"

    # Deterministic global order, independent of the splits file's row order.
    rows.sort(key=lambda r: (r["allele"], r["peptide"]))

    out = []
    counts = {p: 0 for p in PROFILES}
    for i, r in enumerate(rows):
        # Interleave pair-by-pair so each profile gets a balanced half of every
        # allele and split: a partial result from either stays unbiased.
        profile = PROFILES[i % len(PROFILES)]
        position = counts[profile]
        counts[profile] += 1
        out.append(
            {
                "complex_id": complex_id(r["allele"], r["peptide"]),
                "allele": r["allele"],
                "peptide": r["peptide"],
                "split": r["split"],
                "pair_id": r["pair_id"],
                "profile": profile,
                "shard": position // SHARD_SIZE,
                "order_in_shard": position % SHARD_SIZE,
            }
        )

    ids = [r["complex_id"] for r in out]
    assert len(set(ids)) == len(ids), "complex_id is not unique across the cohort"

    DEST.parent.mkdir(exist_ok=True)
    with DEST.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    print(f"wrote {DEST.relative_to(ROOT)}: {len(out)} pairs, shard size {SHARD_SIZE}")
    for p in PROFILES:
        sel = [r for r in out if r["profile"] == p]
        shards = {r["shard"] for r in sel}
        print(f"  {p:14} {len(sel):6} pairs  {len(shards):4} shards")
    by_split = {}
    for r in out:
        by_split.setdefault(r["split"], {}).setdefault(r["profile"], 0)
        by_split[r["split"]][r["profile"]] += 1
    for split, d in sorted(by_split.items()):
        print(f"  {split:6} " + "  ".join(f"{p}={d.get(p, 0)}" for p in PROFILES))


if __name__ == "__main__":
    main()
