"""Guard the frozen stage 4c production cohort and its two-workspace schedule.

The expensive mistakes this file is here to catch: folding the same half twice
while leaving the other unfolded, silently dropping pairs, and letting the
cohort drift away from the frozen splits it was derived from.
"""
import csv
import json
from pathlib import Path

from scripts.freeze_structural_cohort import PROFILES, SHARD_SIZE, complex_id

ROOT = Path(__file__).resolve().parent.parent
COHORT = ROOT / "data" / "structural_cohort.csv"
SPLITS = ROOT / "data" / "splits.csv"


def rows():
    return list(csv.DictReader(COHORT.open()))


def test_cohort_is_exactly_the_frozen_splits():
    cohort = {(r["allele"], r["peptide"]) for r in rows()}
    splits = {(r["allele"], r["peptide"]) for r in csv.DictReader(SPLITS.open())}
    assert cohort == splits
    assert len(cohort) == 28166


def test_split_labels_match_the_frozen_file():
    by_pair = {
        (r["allele"], r["peptide"]): r["split"] for r in csv.DictReader(SPLITS.open())
    }
    for r in rows():
        assert r["split"] == by_pair[(r["allele"], r["peptide"])]


def test_halves_are_disjoint_and_complete():
    all_rows = rows()
    halves = {p: {r["complex_id"] for r in all_rows if r["profile"] == p} for p in PROFILES}
    a, b = (halves[p] for p in PROFILES)
    assert not a & b, "a pair is assigned to both profiles"
    assert len(a | b) == len(all_rows), "a pair is assigned to neither profile"
    # Interleaving should keep the halves within one pair of each other.
    assert abs(len(a) - len(b)) <= 1


def test_each_split_is_balanced_across_profiles():
    all_rows = rows()
    for split in {r["split"] for r in all_rows}:
        counts = [
            sum(1 for r in all_rows if r["split"] == split and r["profile"] == p)
            for p in PROFILES
        ]
        # A partial result from one workspace must stay representative.
        assert max(counts) - min(counts) <= 0.05 * max(counts), (split, counts)


def test_complex_ids_are_unique_and_filesystem_safe():
    ids = [r["complex_id"] for r in rows()]
    assert len(set(ids)) == len(ids)
    assert all(not set(i) & set("*:()/ ") for i in ids)
    assert complex_id("HLA-B*14:01(C67S)", "AARGSHGYL") == "B1401_C67S_AARGSHGYL"


def test_shards_are_contiguous_and_correctly_sized():
    all_rows = rows()
    for p in PROFILES:
        sel = [r for r in all_rows if r["profile"] == p]
        by_shard = {}
        for r in sel:
            by_shard.setdefault(int(r["shard"]), []).append(r)
        assert sorted(by_shard) == list(range(len(by_shard))), "shard ids are not contiguous"
        sizes = [len(v) for _, v in sorted(by_shard.items())]
        assert all(s == SHARD_SIZE for s in sizes[:-1]), "only the last shard may be short"
        assert 0 < sizes[-1] <= SHARD_SIZE
        assert sum(sizes) == len(sel)


def test_every_allele_has_a_prepared_production_msa():
    manifest = json.loads(
        (ROOT / "structures/ectodomain_msas/manifest.json").read_text()
    )
    prepared = {m["allele"] for m in manifest["msas"]}
    assert {r["allele"] for r in rows()} <= prepared


def test_staged_production_manifest_matches_the_prepared_msas():
    staged = ROOT / "structures/ectodomain_production_msas/production_manifest.json"
    if not staged.exists():  # produced by scripts/stage_production_msas.py
        return
    full = json.loads(
        (ROOT / "structures/ectodomain_msas/manifest.json").read_text()
    )
    by_allele = {m["allele"]: m for m in full["msas"]}
    for m in json.loads(staged.read_text())["msas"]:
        for key in ("B", "beta2m"):
            assert m[key]["csv_sha256"] == by_allele[m["allele"]][key]["csv_sha256"]
