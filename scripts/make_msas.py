"""Stage 4b.1 — precompute one Boltz MSA per unique HLA sequence (CPU, ~$0).

Alleles and HLA domain sequences are 1:1 in this dataset (75/75, asserted
below), so one MSA per allele is reused across every complex on that allele:
six panel alleles means six MSA generations for ~2,000 folds.

**Route.** HACKATHON_PLAN.md §4b.1 prescribes driving `boltz predict` on a
pilot YAML and harvesting the MSA it drops. That works, but in boltz 2.2.1
`download_boltz2(cache)` runs at `main.py:1141`, *before* `process_inputs` at
`main.py:1162` — so the CLI route downloads all 6.2 GB of weights and CCD
before it generates a single MSA, and then has to be killed before the model
loads. `boltz.main.compute_msa` is the function that CLI route eventually
calls, it is importable, and it needs neither the checkpoints nor the CCD.
Calling it directly is the same code path for the part we want, so the output
format is correct by construction, and it costs no download and no kill hack.

With a single sequence in `data`, compute_msa takes the unpaired branch
(`len(data) > 1` is false), which is exactly what the plan's `msa: empty`
peptide chain was there to force. Every row therefore gets key `-1`.

Labels are never read: only `allele` and `hla_seq` are loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "rasmussen_et_al_dataset.csv"
MSA_SERVER = "https://api.colabfold.com"
PAIRING_STRATEGY = "greedy"

# The stage 4b.4 structural panel: the six deepest alleles. Every allele in the
# 4b.2 crystal pilot (A*02:01, B*07:02, B*35:01, B*15:01) is a subset of these,
# so one pass covers the pilot and the batch.
PANEL_ALLELES = [
    "HLA-B*15:01",
    "HLA-A*02:01",
    "HLA-A*03:01",
    "HLA-B*39:01",
    "HLA-B*35:01",
    "HLA-B*07:02",
]


def slug(allele: str) -> str:
    """Filesystem-safe stem. `HLA-B*14:01(C67S)` -> `HLA-B_14_01_C67S`."""
    out = allele.replace("*", "_").replace(":", "_")
    return out.replace("(", "_").replace(")", "").strip("_")


def rel(path: Path) -> str:
    """Repo-relative when inside the repo, absolute otherwise."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_hla_sequences(alleles: list[str]) -> dict[str, str]:
    df = pd.read_csv(RAW, usecols=["allele", "hla_seq"])

    per_allele = df.groupby("allele")["hla_seq"].nunique()
    if (per_allele > 1).any():
        bad = per_allele[per_allele > 1].index.tolist()
        raise SystemExit(f"allele -> hla_seq is not 1:1 for {bad}; one MSA per allele is unsafe")
    per_seq = df.groupby("hla_seq")["allele"].nunique()
    if (per_seq > 1).any():
        raise SystemExit("one hla_seq maps to several alleles; MSA reuse keys would collide")

    missing = sorted(set(alleles) - set(df["allele"]))
    if missing:
        raise SystemExit(f"alleles absent from the dataset: {missing}")

    first = df.drop_duplicates("allele").set_index("allele")["hla_seq"]
    return {a: first[a] for a in alleles}


def benchmark_parse(path: Path, caps: list[int]) -> list[dict]:
    """Time boltz's own MSA parser, which re-runs once per complex on CPU."""
    from boltz.data.parse.csv import parse_csv

    out = []
    for cap in caps:
        parse_csv(path, max_seqs=cap)  # warm the page cache
        t0 = time.perf_counter()
        msa = parse_csv(path, max_seqs=cap)
        dt = time.perf_counter() - t0
        out.append({"max_msa_seqs": cap, "parsed_seqs": len(msa.sequences), "parse_seconds": round(dt, 3)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alleles", nargs="*", default=None,
                    help="alleles to cache (default: the 4b.4 panel)")
    ap.add_argument("--all-alleles", action="store_true",
                    help="cache every allele in the dataset; the panel is not frozen yet "
                         "and each MSA costs ~2 s, so this removes the dependency")
    ap.add_argument("--delay", type=float, default=1.0,
                    help="seconds between MSA server calls (be polite to the public API)")
    ap.add_argument("--msa-dir", type=Path, default=REPO / "structures" / "msa")
    ap.add_argument("--manifest", type=Path, default=REPO / "reports" / "msa_manifest.csv")
    ap.add_argument("--bench-json", type=Path, default=REPO / "reports" / "msa_parse_benchmark.json")
    ap.add_argument("--force", action="store_true", help="regenerate alleles already on disk")
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()

    from boltz.data import const
    from boltz.main import compute_msa

    if args.all_alleles and args.alleles:
        raise SystemExit("pass --alleles or --all-alleles, not both")
    if args.all_alleles:
        wanted = sorted(pd.read_csv(RAW, usecols=["allele"])["allele"].unique())
    else:
        wanted = list(args.alleles) if args.alleles else list(PANEL_ALLELES)
    seqs = load_hla_sequences(wanted)
    args.msa_dir.mkdir(parents=True, exist_ok=True)
    tmp = args.msa_dir / "_tmp"
    tmp.mkdir(exist_ok=True)

    rows = []
    for allele, hla_seq in seqs.items():
        stem = slug(allele)
        dest = args.msa_dir / f"{stem}.csv"
        if dest.exists() and not args.force:
            print(f"[skip] {allele}: {dest.name} present", flush=True)
            elapsed = None
        else:
            print(f"[gen ] {allele} ({len(hla_seq)} aa) -> {stem}.csv", flush=True)
            t0 = time.perf_counter()
            # compute_msa writes msa_dir / f"{name}.csv" for each name in data
            compute_msa(
                data={stem: hla_seq},
                target_id=stem,
                msa_dir=args.msa_dir,
                msa_server_url=MSA_SERVER,
                msa_pairing_strategy=PAIRING_STRATEGY,
            )
            elapsed = time.perf_counter() - t0
            if not dest.exists():
                raise SystemExit(f"compute_msa did not write {dest}")
            print(f"[done] {allele} in {elapsed:.0f}s", flush=True)
            if args.delay > 0:
                time.sleep(args.delay)

        msa = pd.read_csv(dest)
        if tuple(sorted(msa.columns)) != ("key", "sequence"):
            raise SystemExit(f"{dest}: expected columns key,sequence; got {list(msa.columns)}")
        query = str(msa["sequence"].iloc[0]).replace("-", "").upper()
        if query != hla_seq.upper():
            raise SystemExit(f"{dest}: first row is not the query HLA sequence")
        if not (msa["key"] == -1).all():
            raise SystemExit(f"{dest}: expected all-unpaired keys (-1); found paired rows")
        n_unique = msa["sequence"].str.replace("-", "", regex=False).str.upper().nunique()

        rows.append({
            "allele": allele,
            "stem": stem,
            "hla_seq_len": len(hla_seq),
            "hla_seq_sha256": sha256_text(hla_seq),
            "n_rows": len(msa),
            "n_unique_ungapped": int(n_unique),
            "msa_csv": rel(dest),
            "msa_sha256": sha256_file(dest),
            "msa_bytes": dest.stat().st_size,
            "msa_server_url": MSA_SERVER,
            "msa_pairing_strategy": PAIRING_STRATEGY,
            "gen_cap_max_msa_seqs": const.max_msa_seqs,
            "boltz_version": version("boltz"),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gen_seconds": round(elapsed, 1) if elapsed is not None else "",
        })

    for leftover in tmp.rglob("*"):
        if leftover.is_file():
            leftover.unlink()
    tmp.rmdir() if not any(tmp.iterdir()) else None

    manifest = pd.DataFrame(rows)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False)
    print()
    print(manifest[["allele", "stem", "n_rows", "n_unique_ungapped", "msa_bytes", "gen_seconds"]].to_string(index=False))
    print(f"\nmanifest -> {rel(args.manifest)}")

    if not args.skip_bench:
        deepest = manifest.loc[manifest["n_rows"].idxmax()]
        path = Path(deepest["msa_csv"])
        if not path.is_absolute():
            path = REPO / path
        caps = [256, 512, 1024, 2048, 4096, 8192, 16384]
        bench = benchmark_parse(path, caps)
        payload = {
            "benchmarked_file": deepest["msa_csv"],
            "allele": deepest["allele"],
            "n_rows": int(deepest["n_rows"]),
            "platform": f"{platform.system()} {platform.machine()}",
            "boltz_version": version("boltz"),
            "note": "parse_csv runs once per complex; multiply by the panel size for total CPU cost",
            "results": bench,
        }
        args.bench_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nparse cost on {deepest['allele']} ({deepest['n_rows']} rows):")
        for r in bench:
            print(f"  max_msa_seqs={r['max_msa_seqs']:>6}  parsed={r['parsed_seqs']:>6}  {r['parse_seconds']:.3f}s")
        print(f"benchmark -> {rel(args.bench_json)}")


if __name__ == "__main__":
    main()
