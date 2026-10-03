"""Guards on the stage 4b.1 MSA cache. Run with: .venv/bin/python -m pytest -q

The manifest is the committed record; the MSA files themselves are gitignored.
These guards check the manifest's internal consistency and its agreement with
the raw dataset, so a stale or partial cache cannot silently feed the fold.
Guards that need the MSA files skip when the cache is absent.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.make_msas import PANEL_ALLELES, slug

MANIFEST = REPO / "reports" / "msa_manifest.csv"


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    if not MANIFEST.exists():
        pytest.skip(f"{MANIFEST} absent; run scripts/make_msas.py --all-alleles")
    return pd.read_csv(MANIFEST)


@pytest.fixture(scope="module")
def alleles() -> pd.DataFrame:
    df = pd.read_csv(REPO / "data" / "rasmussen_et_al_dataset.csv",
                     usecols=["allele", "hla_seq"])
    return df.drop_duplicates("allele").reset_index(drop=True)


def test_slug_sanitises_c67s_constructs():
    assert slug("HLA-A*02:01") == "HLA-A_02_01"
    assert slug("HLA-B*14:01(C67S)") == "HLA-B_14_01_C67S"


def test_slugs_are_unique_across_every_allele(alleles):
    stems = [slug(a) for a in alleles["allele"]]
    assert len(set(stems)) == len(stems), "two alleles sanitise to one filename"


def test_allele_and_hla_sequence_are_one_to_one(alleles):
    # One MSA per allele is only safe while this holds.
    assert alleles["allele"].nunique() == alleles["hla_seq"].nunique() == len(alleles)


def test_manifest_covers_every_allele(manifest, alleles):
    assert set(manifest["allele"]) == set(alleles["allele"])


def test_manifest_rows_are_unique_per_allele(manifest):
    assert not manifest["allele"].duplicated().any()
    assert not manifest["stem"].duplicated().any()


def test_manifest_stems_match_the_slug_rule(manifest):
    assert (manifest["stem"] == manifest["allele"].map(slug)).all()


def test_distinct_hla_sequence_gets_a_distinct_msa(manifest):
    # Identical checksums would mean the cache collapsed two alleles into one MSA.
    assert manifest["hla_seq_sha256"].nunique() == len(manifest)
    assert manifest["msa_sha256"].nunique() == len(manifest)


def test_manifest_hla_checksums_match_the_raw_dataset(manifest, alleles):
    expected = {
        row.allele: hashlib.sha256(row.hla_seq.encode()).hexdigest()
        for row in alleles.itertuples()
    }
    got = dict(zip(manifest["allele"], manifest["hla_seq_sha256"]))
    assert got == expected


def test_domain_length_is_182_everywhere(manifest):
    assert set(manifest["hla_seq_len"]) == {182}


def test_panel_alleles_are_cached(manifest):
    assert set(PANEL_ALLELES) <= set(manifest["allele"])


def test_msa_depth_is_plausible(manifest):
    # A near-empty MSA means the server returned nothing useful.
    assert (manifest["n_rows"] >= 100).all()
    assert (manifest["n_unique_ungapped"] <= manifest["n_rows"]).all()


def test_c67s_pair_is_separated_at_the_domain_level(manifest):
    # Stage 1's pseudosequence collision must not reach the structural arm.
    pair = manifest[manifest["allele"].isin(["HLA-B*14:01(C67S)", "HLA-B*14:02(C67S)"])]
    if len(pair) != 2:
        pytest.skip("C67S pair not in the manifest")
    assert pair["hla_seq_sha256"].nunique() == 2
    assert pair["msa_sha256"].nunique() == 2


def test_cached_files_match_their_recorded_checksums(manifest):
    missing = [r.msa_csv for r in manifest.itertuples() if not (REPO / r.msa_csv).exists()]
    if missing:
        pytest.skip(f"{len(missing)} MSA files absent (gitignored); regenerate to check")
    for row in manifest.itertuples():
        path = REPO / row.msa_csv
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        assert h.hexdigest() == row.msa_sha256, f"{row.msa_csv} changed since the manifest"
        assert path.stat().st_size == row.msa_bytes


def test_cached_files_are_unpaired_with_the_query_first(manifest, alleles):
    seqs = dict(zip(alleles["allele"], alleles["hla_seq"]))
    checked = 0
    for row in manifest.itertuples():
        path = REPO / row.msa_csv
        if not path.exists():
            continue
        msa = pd.read_csv(path)
        assert tuple(sorted(msa.columns)) == ("key", "sequence"), path
        assert (msa["key"] == -1).all(), f"{path}: paired rows in a single-chain MSA"
        first = str(msa["sequence"].iloc[0]).replace("-", "").upper()
        assert first == seqs[row.allele].upper(), f"{path}: first row is not the query"
        checked += 1
    if checked == 0:
        pytest.skip("no MSA files on disk (gitignored); regenerate to check")
