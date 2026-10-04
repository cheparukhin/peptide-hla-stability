"""Download Boltz-2 mmCIF structures from Modal and extract Cα coordinates.

Connects to the ``pepstab-structures`` Modal Volume, downloads the mmCIF for
each complex in the demo's solved-pair set, and writes a compact JSON file
that ``build_payload.py`` merges into the dashboard payload.

Usage:
    python3 demo/fetch_structures.py              # all 32 solved pairs
    python3 demo/fetch_structures.py --limit 5    # quick test
    python3 demo/fetch_structures.py --local DIR  # parse from local mmCIF files

Requires: ``modal`` (pip install modal) and an authenticated Modal profile
with access to the ``pepstab-structures`` Volume.

If Modal auth is not available, use ``modal volume get`` to download files
first, then point ``--local`` at the directory:

    modal volume get pepstab-structures stage4c/ectodomain-20261004/boltz2/production ./structures
    python3 demo/fetch_structures.py --local ./structures

Output: ``demo/structures.json`` — a dict keyed by complex_id, each value an
object with ``atoms`` (array of {x, y, z, chain, resi, el}) for the Cα trace.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO = Path(__file__).resolve().parent
OUT = DEMO / "structures.json"

RUN_ID = "ectodomain-20261004"
OUT_ROOT = f"stage4c/{RUN_ID}/boltz2/production"


def allele_slug(allele: str) -> str:
    return allele.replace("*", "_").replace(":", "_").replace("(", "_").replace(")", "")


def complex_id(allele: str, peptide: str) -> str:
    """Match the format in structural_cohort.csv."""
    return allele.replace("HLA-", "").replace("*", "").replace(":", "") + "_" + peptide


def parse_mmcif_ca(cif_text: str) -> list[dict]:
    """Extract Cα atoms from an mmCIF file, returning [{x, y, z, chain, resi, el}].

    Maps Boltz-2 chain IDs to the dashboard's convention:
      chain A (HLA ectodomain)   → 'H'  (heavy chain)
      chain B (β2-microglobulin) → 'B'
      chain C (peptide)          → 'P'
    """
    chain_map = {"A": "H", "B": "B", "C": "P"}
    atoms = []

    in_atom_site = False
    columns: list[str] = []
    col_idx: dict[str, int] = {}

    for line in cif_text.splitlines():
        stripped = line.strip()

        if stripped == "loop_":
            in_atom_site = False
            columns = []
            col_idx = {}
            continue

        if stripped.startswith("_atom_site."):
            col_name = stripped.split(".")[1].strip()
            columns.append(col_name)
            col_idx[col_name] = len(columns) - 1
            in_atom_site = True
            continue

        if not in_atom_site or not columns or stripped.startswith("_") or stripped.startswith("#") or not stripped:
            if in_atom_site and columns and (stripped.startswith("_") or stripped.startswith("#") or stripped == ""):
                in_atom_site = False
            continue

        # We need these columns to parse an ATOM line
        needed = {"group_PDB", "label_atom_id", "Cartn_x", "Cartn_y", "Cartn_z",
                  "label_asym_id", "label_seq_id", "type_symbol"}
        if not needed.issubset(col_idx.keys()):
            continue

        # Split the line respecting quoted strings
        parts = _split_cif_line(stripped)
        if len(parts) < len(columns):
            continue

        try:
            group = parts[col_idx["group_PDB"]]
            if group != "ATOM":
                continue

            atom_name = parts[col_idx["label_atom_id"]]
            if atom_name != "CA":
                continue

            chain_raw = parts[col_idx["label_asym_id"]]
            chain = chain_map.get(chain_raw, chain_raw)
            resi = int(parts[col_idx["label_seq_id"]])
            x = round(float(parts[col_idx["Cartn_x"]]), 2)
            y = round(float(parts[col_idx["Cartn_y"]]), 2)
            z = round(float(parts[col_idx["Cartn_z"]]), 2)
            el = parts[col_idx["type_symbol"]]

            atoms.append({"x": x, "y": y, "z": z, "chain": chain, "resi": resi, "el": el})
        except (IndexError, ValueError, KeyError):
            continue

    return atoms


def _split_cif_line(line: str) -> list[str]:
    """Split a CIF data line, respecting single- and double-quoted strings."""
    parts = []
    current = ""
    in_quote = False
    quote_char = None
    for ch in line:
        if in_quote:
            if ch == quote_char:
                in_quote = False
            else:
                current += ch
        elif ch in ("'", '"'):
            in_quote = True
            quote_char = ch
        elif ch == " ":
            if current:
                parts.append(current)
            current = ""
        else:
            current += ch
    if current:
        parts.append(current)
    return parts


def centre_and_scale(atoms: list[dict]) -> list[dict]:
    """Centre atoms on the peptide and scale to match the schematic coordinate range."""
    pep = [a for a in atoms if a["chain"] == "P"]
    if not pep:
        pep = atoms
    cx = sum(a["x"] for a in pep) / len(pep)
    cy = sum(a["y"] for a in pep) / len(pep)
    cz = sum(a["z"] for a in pep) / len(pep)
    # Typical pMHC ectodomain Cα trace spans ~80Å; schematic spans ~40 units
    scale = 0.5
    return [
        {
            "x": round((a["x"] - cx) * scale, 2),
            "y": round((a["y"] - cy) * scale, 2),
            "z": round((a["z"] - cz) * scale, 2),
            "chain": a["chain"],
            "resi": a["resi"],
            "el": a["el"],
        }
        for a in atoms
    ]


def fetch_from_modal(targets: list[dict]) -> dict[str, list[dict]]:
    """Download mmCIF files from Modal Volume and parse them."""
    import modal

    vol = modal.Volume.from_name("pepstab-structures")
    results = {}
    failed = []

    for t in targets:
        cid = t["complex_id"]
        allele = t["allele"]
        slug = allele_slug(allele)
        remote_path = f"{OUT_ROOT}/{slug}/{cid}/{cid}_model_0.cif"

        try:
            data = b""
            for chunk in vol.read_file(remote_path):
                data += chunk
            cif_text = data.decode("utf-8")

            atoms = parse_mmcif_ca(cif_text)
            if not atoms:
                failed.append((cid, "no CA atoms found"))
                continue

            atoms = centre_and_scale(atoms)
            n_pep = sum(1 for a in atoms if a["chain"] == "P")
            n_hla = sum(1 for a in atoms if a["chain"] == "H")
            n_b2m = sum(1 for a in atoms if a["chain"] == "B")
            results[cid] = atoms
            print(f"  {cid:30s}  {len(atoms):4d} Cα  (H:{n_hla} B:{n_b2m} P:{n_pep})")

        except Exception as e:
            failed.append((cid, str(e)[:120]))

    if failed:
        print(f"\n{len(failed)} failed:")
        for cid, err in failed:
            print(f"  {cid}: {err}")

    return results


def fetch_from_local(targets: list[dict], local_dir: Path) -> dict[str, list[dict]]:
    """Parse mmCIF files from a local directory tree."""
    results = {}
    failed = []

    for t in targets:
        cid = t["complex_id"]
        allele = t["allele"]
        slug = allele_slug(allele)

        # Try several path patterns
        candidates = [
            local_dir / slug / cid / f"{cid}_model_0.cif",
            local_dir / cid / f"{cid}_model_0.cif",
            local_dir / f"{cid}_model_0.cif",
        ]
        cif_path = None
        for c in candidates:
            if c.exists():
                cif_path = c
                break

        if cif_path is None:
            failed.append((cid, "mmCIF not found"))
            continue

        try:
            cif_text = cif_path.read_text()
            atoms = parse_mmcif_ca(cif_text)
            if not atoms:
                failed.append((cid, "no CA atoms found"))
                continue

            atoms = centre_and_scale(atoms)
            n_pep = sum(1 for a in atoms if a["chain"] == "P")
            n_hla = sum(1 for a in atoms if a["chain"] == "H")
            n_b2m = sum(1 for a in atoms if a["chain"] == "B")
            results[cid] = atoms
            print(f"  {cid:30s}  {len(atoms):4d} Cα  (H:{n_hla} B:{n_b2m} P:{n_pep})")

        except Exception as e:
            failed.append((cid, str(e)[:120]))

    if failed:
        print(f"\n{len(failed)} failed:")
        for cid, err in failed:
            print(f"  {cid}: {err}")

    return results


def build_targets(limit: int = 0) -> list[dict]:
    """Build the target list from solved pairs × structural cohort."""
    overlap = []
    with (ROOT / "data/pdb_rasmussen_overlap.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            overlap.append({"allele": r["allele"], "peptide": r["peptide"]})

    cohort_ids = set()
    with (ROOT / "data/structural_cohort.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            cohort_ids.add(r["complex_id"])

    targets = []
    for o in overlap:
        cid = complex_id(o["allele"], o["peptide"])
        if cid in cohort_ids:
            targets.append({"complex_id": cid, "allele": o["allele"], "peptide": o["peptide"]})

    if limit:
        targets = targets[:limit]
    return targets


def main():
    parser = argparse.ArgumentParser(description="Fetch Boltz-2 structures for the demo")
    parser.add_argument("--limit", type=int, default=0, help="Max structures to fetch (0=all)")
    parser.add_argument("--local", type=str, default="", help="Local directory with mmCIF files")
    parser.add_argument("--dry-run", action="store_true", help="List targets without downloading")
    args = parser.parse_args()

    targets = build_targets(args.limit)
    print(f"Targets: {len(targets)} solved pairs with Boltz-2 structures")

    if args.dry_run:
        for t in targets:
            slug = allele_slug(t["allele"])
            print(f"  {t['complex_id']:30s}  {OUT_ROOT}/{slug}/{t['complex_id']}/")
        return

    if args.local:
        results = fetch_from_local(targets, Path(args.local))
    else:
        results = fetch_from_modal(targets)

    print(f"\nFetched {len(results)}/{len(targets)} structures")

    # Write compact JSON
    OUT.write_text(json.dumps(results, separators=(",", ":")) + "\n")
    size_kb = OUT.stat().st_size / 1024
    print(f"Wrote {OUT.relative_to(ROOT)} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
