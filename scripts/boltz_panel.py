"""Select the stage 4 pilot and GPU-benchmark panels, and list the MSAs they need.

Three outputs, all deterministic:

* ``reports/boltz_pilot.csv`` -- 5 complexes for the end-to-end pilot. Every one
  is TCR-free, sits in the *train* split, and has a published crystal structure
  at <= 1.9 A, so the pilot doubles as the register check STRUCTURES.md asks for:
  does the predicted peptide land in the canonical P2/POmega-anchored pose where
  the answer is already known?
* ``reports/boltz_bench_panel.csv`` -- 24 complexes (4 each across the 6
  best-represented alleles) for the hardware benchmark.
* ``reports/boltz_msa_targets.csv`` -- the unique HLA domain sequences those
  panels touch. MSAs are per *sequence*, not per complex, so the benchmark's 24
  complexes need only 6 alignments.

Every row is drawn from the train split. The panels measure runtime and pose
quality, neither of which needs a held-out label, so there is no reason to spend
test rows on them.

Usage:
    python scripts/boltz_panel.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATASET = REPO / "data" / "rasmussen_et_al_dataset.csv"
SPLITS = REPO / "data" / "splits.csv"
OVERLAP = REPO / "data" / "pdb_rasmussen_overlap.csv"
REPORTS = REPO / "reports"

# The 6 best-represented alleles by total pair count. Each clears the plan's
# "at least 50 held-out examples, ideally closer to 100" bar by a wide margin
# (121-222 test rows), so the same 6 are the natural production panel.
BENCH_ALLELES = [
    "HLA-B*15:01",
    "HLA-A*02:01",
    "HLA-A*03:01",
    "HLA-B*39:01",
    "HLA-B*35:01",
    "HLA-B*07:02",
]
PER_ALLELE = 4

# Hand-picked pilot: TCR-free, train split, crystal structure <= 1.9 A, and
# spread across the label range (0.75 h to 59.3 h) so a systematic failure at
# one end of the range is visible. TCR-free matters because TCR engagement can
# perturb the peptide conformation, which would muddy an RMSD check.
PILOT_PAIRS = [
    ("HLA-A*11:01", "KTFPPTEPK"),
    ("HLA-A*02:01", "LLWNGPMAV"),
    ("HLA-B*15:01", "ILGPPGSVY"),
    ("HLA-B*07:02", "IPRRNVATL"),
    ("HLA-B*08:01", "ELRRKMMYM"),
]


def load_rows() -> list[dict]:
    """Join the dataset against the frozen splits on (allele, peptide)."""
    splits = {
        (r["allele"], r["peptide"]): r for r in csv.DictReader(SPLITS.open())
    }
    rows = []
    for r in csv.DictReader(DATASET.open()):
        key = (r["allele"], r["peptide"])
        s = splits.get(key)
        if s is None:
            raise SystemExit(f"{key} missing from splits.csv")
        rows.append(
            {
                "allele": r["allele"],
                "peptide": r["peptide"],
                "thalf_hours": float(r["thalf_hours"]),
                "hla_seq": r["hla_seq"],
                "split": s["split"],
                "cluster_id": s["cluster_id"],
            }
        )
    return rows


def complex_id(allele: str, peptide: str) -> str:
    """A filesystem-safe, stable identifier shared by every downstream manifest."""
    safe = allele.replace("HLA-", "").replace("*", "").replace(":", "")
    return f"{safe}_{peptide}"


def select_pilot(rows: list[dict]) -> list[dict]:
    by_key = {(r["allele"], r["peptide"]): r for r in rows}
    overlap = {
        (r["allele"], r["peptide"]): r for r in csv.DictReader(OVERLAP.open())
    }
    out = []
    for key in PILOT_PAIRS:
        row = by_key.get(key)
        if row is None:
            raise SystemExit(f"pilot pair {key} not in dataset")
        if row["split"] != "train":
            raise SystemExit(f"pilot pair {key} is in {row['split']}, expected train")
        ov = overlap.get(key)
        if ov is None:
            raise SystemExit(f"pilot pair {key} has no crystal structure")
        if ov["any_tcr"] != "False":
            raise SystemExit(f"pilot pair {key} is TCR-bound")
        out.append(
            {
                **row,
                "pdb_ids": ov["pdb_ids"],
                "best_resolution": ov["best_resolution"],
            }
        )
    return out


def select_bench(rows: list[dict], exclude: set[tuple[str, str]]) -> list[dict]:
    """Evenly spaced picks from each allele's cluster-sorted train rows.

    Sorting by (cluster_id, peptide) and striding gives a reproducible spread
    across peptide clusters without an RNG, so the panel is identical on every
    machine and in every rerun.
    """
    out = []
    for allele in BENCH_ALLELES:
        pool = [
            r
            for r in rows
            if r["allele"] == allele
            and r["split"] == "train"
            and (r["allele"], r["peptide"]) not in exclude
        ]
        pool.sort(key=lambda r: (int(r["cluster_id"]), r["peptide"]))
        if len(pool) < PER_ALLELE:
            raise SystemExit(f"{allele} has only {len(pool)} eligible train rows")
        stride = len(pool) / PER_ALLELE
        out.extend(pool[int(i * stride)] for i in range(PER_ALLELE))
    return out


def write_panel(path: Path, rows: list[dict], extra_cols: list[str]) -> None:
    cols = ["complex_id", "allele", "peptide", "thalf_hours", "split", "cluster_id"]
    cols += extra_cols
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            rec = {c: r.get(c, "") for c in cols}
            rec["complex_id"] = complex_id(r["allele"], r["peptide"])
            w.writerow(rec)
    print(f"wrote {path.relative_to(REPO)} ({len(rows)} rows)")


def write_msa_targets(path: Path, rows: list[dict]) -> None:
    """One row per unique HLA domain sequence -- the unit an MSA is built for."""
    seen: dict[str, dict] = {}
    for r in rows:
        seq = r["hla_seq"]
        if seq not in seen:
            seen[seq] = {"hla_seq": seq, "alleles": set(), "n_complexes": 0}
        seen[seq]["alleles"].add(r["allele"])
        seen[seq]["n_complexes"] += 1
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["msa_id", "alleles", "n_complexes", "seq_len", "hla_seq"])
        for i, (seq, rec) in enumerate(sorted(seen.items(), key=lambda kv: sorted(kv[1]["alleles"]))):
            alleles = "|".join(sorted(rec["alleles"]))
            w.writerow([f"msa{i:03d}", alleles, rec["n_complexes"], len(seq), seq])
    print(f"wrote {path.relative_to(REPO)} ({len(seen)} unique HLA sequences)")


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    REPORTS.mkdir(exist_ok=True)
    rows = load_rows()

    pilot = select_pilot(rows)
    write_panel(
        REPORTS / "boltz_pilot.csv", pilot, ["pdb_ids", "best_resolution"]
    )

    bench = select_bench(rows, exclude={(r["allele"], r["peptide"]) for r in pilot})
    write_panel(REPORTS / "boltz_bench_panel.csv", bench, [])

    write_msa_targets(REPORTS / "boltz_msa_targets.csv", pilot + bench)

    labels = sorted(r["thalf_hours"] for r in pilot)
    print(f"\npilot: {len(pilot)} complexes, t-half {labels[0]}-{labels[-1]} h")
    print(f"bench: {len(bench)} complexes across {len(BENCH_ALLELES)} alleles")


if __name__ == "__main__":
    main()
