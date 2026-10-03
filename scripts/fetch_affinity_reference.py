"""Download public pMHC-I binding-affinity data and annotate it against the stability set.

Sources (both are MHCflurry GitHub release assets; no IEDB/DTU host required):
  MHCflurry curated affinity, snapshot 2023-10-23  -- primary
  Kim et al. 2014 benchmark BD2013                 -- independent cross-check

Output is every human 9-mer affinity measurement on the 75 stability alleles, with
flags for weak binders, for pairs already measured in the stability set (these carry
an experimental affinity for a complex whose half-life you already have), and for
agreement with BD2013.

    python3 scripts/fetch_affinity_reference.py
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CURATED_URL = ("https://github.com/openvax/mhcflurry/releases/download/"
               "pre-2.1/data_curated.20231023.tar.bz2")
CURATED_MEMBER = "curated_training_data.affinity.csv.bz2"
KIM_URL = "https://github.com/openvax/mhcflurry/releases/download/0.9.1/data_kim2014.tar.bz2"
KIM_MEMBER = "bdata.20130222.mhci.public.1.txt"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STABILITY = REPO_ROOT / "data" / "rasmussen_et_al_dataset.csv"
DEFAULT_CACHE = REPO_ROOT / "external"
DEFAULT_OUT = (REPO_ROOT / "data" / "data_augmentation_iedb"
               / "affinity_reference_75alleles.csv")

WEAK_NM = 20000.0
COMPACT = re.compile(r"^HLA-([ABC])(\d{2})(\d{2,3})$")


def download(url: str, dest: Path) -> Path:
    if not dest.exists():
        urllib.request.urlretrieve(url, dest)
    return dest


def extract_member(archive: Path, member: str, outdir: Path) -> Path:
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extract(member, path=outdir, filter="data")
    return outdir / member


def normalise_allele(name: str) -> str:
    """HLA-A0201 and HLA-A*02:01 both -> HLA-A*02:01."""
    if not isinstance(name, str):
        return ""
    name = name.strip()
    m = COMPACT.match(name)
    return f"HLA-{m.group(1)}*{m.group(2)}:{m.group(3)}" if m else name


def load_curated(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d.allele.str.startswith("HLA-") & (d.peptide.str.len() == 9)].copy()
    d["allele"] = d.allele.map(normalise_allele)
    return d.rename(columns={"measurement_value": "affinity_nM",
                             "measurement_inequality": "inequality",
                             "measurement_source": "assay_source"})[
        ["allele", "peptide", "affinity_nM", "inequality", "assay_source"]]


def load_bd2013(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, sep="\t")
    d = d[(d.species == "human") & (d.peptide_length == 9)].copy()
    d["allele"] = d.mhc.map(normalise_allele)
    return d.rename(columns={"sequence": "peptide", "meas": "bd2013_affinity_nM",
                             "inequality": "bd2013_inequality"})[
        ["allele", "peptide", "bd2013_affinity_nM", "bd2013_inequality"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stability", type=Path, default=DEFAULT_STABILITY)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--weak-nm", type=float, default=WEAK_NM)
    args = parser.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    curated = load_curated(extract_member(
        download(CURATED_URL, cache / "data_curated.tar.bz2"), CURATED_MEMBER, cache))
    bd2013 = load_bd2013(extract_member(
        download(KIM_URL, cache / "data_kim2014.tar.bz2"), KIM_MEMBER, cache))

    stability = pd.read_csv(args.stability)
    alias = {a.replace("(C67S)", ""): a for a in stability.allele.unique()}
    targets = set(alias)

    ref = curated[curated.allele.isin(targets)].copy()
    ref = (ref.sort_values("affinity_nM")
              .drop_duplicates(["allele", "peptide"], keep="first"))

    ref = ref.merge(bd2013.drop_duplicates(["allele", "peptide"]),
                    on=["allele", "peptide"], how="left")

    thalf = (stability.assign(allele=stability.allele.str.replace("(C67S)", "", regex=False))
                      .set_index(["allele", "peptide"]).thalf_hours)
    keys = list(zip(ref.allele, ref.peptide))
    ref["stability_thalf_hours"] = [thalf.get(k, np.nan) for k in keys]
    ref["in_stability_dataset"] = ref.stability_thalf_hours.notna()
    ref["dataset_allele"] = ref.allele.map(alias)
    # Engineered constructs carry a substitution at a peptide-contact position, so
    # wild-type affinity is not a measurement of the same groove. Flag, never join.
    ref["engineered_construct_mismatch"] = ref.dataset_allele.str.contains(r"\(", na=False)
    ref.loc[ref.engineered_construct_mismatch,
            ["in_stability_dataset", "stability_thalf_hours"]] = [False, np.nan]
    ref["is_weak_binder"] = ref.affinity_nM >= args.weak_nm
    ref["padding_eligible"] = (ref.is_weak_binder & ~ref.in_stability_dataset
                               & ~ref.engineered_construct_mismatch)
    ref["bd2013_agrees_weak"] = ref.bd2013_affinity_nM >= args.weak_nm

    ref = ref[["dataset_allele", "allele", "peptide", "affinity_nM", "inequality",
               "assay_source", "bd2013_affinity_nM", "bd2013_inequality",
               "bd2013_agrees_weak", "is_weak_binder", "in_stability_dataset",
               "stability_thalf_hours", "engineered_construct_mismatch", "padding_eligible"]]
    ref.to_csv(out, index=False)

    both = ref[ref.in_stability_dataset]
    manifest = {
        "sources": {"curated": {"url": CURATED_URL, "member": CURATED_MEMBER,
                                "snapshot": "2023-10-23"},
                    "bd2013": {"url": KIM_URL, "member": KIM_MEMBER,
                               "citation": "Kim et al. 2014, BMC Bioinformatics"}},
        "filters": {"species": "human HLA", "peptide_length": 9,
                    "alleles": "the 75 stability alleles, C67S suffix stripped for matching",
                    "weak_binder_threshold_nM": args.weak_nm,
                    "duplicates": "lowest affinity_nM kept per (allele, peptide)"},
        "counts": {
            "rows": int(len(ref)),
            "alleles_with_any_affinity": int(ref.allele.nunique()),
            "alleles_with_weak_binders": int(ref.loc[ref.is_weak_binder, "allele"].nunique()),
            "weak_binders": int(ref.is_weak_binder.sum()),
            "pairs_also_in_stability_set": int(ref.in_stability_dataset.sum()),
            "padding_eligible": int(ref.padding_eligible.sum()),
            "bd2013_overlap": int(ref.bd2013_affinity_nM.notna().sum()),
            "bd2013_agrees_on_weak": int((ref.is_weak_binder & ref.bd2013_agrees_weak).sum()),
            "excluded_engineered_construct_rows": int(ref.engineered_construct_mismatch.sum()),
        },
        "engineered_construct_note": (
            "C67S constructs are substituted at position 67, which is one of the 34 "
            "peptide-contact positions, so wild-type affinity measures a different groove. "
            "These rows are flagged and excluded from both the stability join and padding."),
        "alleles_without_weak_binders": sorted(
            targets - set(ref.loc[ref.is_weak_binder, "allele"])),
    }
    if len(both):
        manifest["counts"]["affinity_vs_stability_spearman"] = round(float(
            both.affinity_nM.rank().corr(both.stability_thalf_hours.rank())), 4)
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
