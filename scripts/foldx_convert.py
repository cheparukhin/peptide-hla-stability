#!/usr/bin/env python
"""Stage 8: validate mmCIF -> PDB conversion on local folds. Free, no Modal.

FoldX reads PDB and is unforgiving of malformed input, so the conversion is
checked rather than assumed. For every fold this writes the PDB, reads it back
and compares it to the mmCIF atom for atom, then reports the structural
conditions that bear on whether ``RepairPDB`` is needed:

* **missing heavy atoms** -- the usual reason to repair a crystal structure.
  A predicted model has none, but that is a claim to measure, not to assert.
* **non-disulfide steric overlaps** -- the other reason. Cys SG-SG pairs at
  ~2.0 A are disulfide bonds, not clashes, and counting them as clashes would
  make every fold look broken.

    .venv/bin/python scripts/foldx_convert.py structures/ectodomain_pilot --out /tmp/pdb
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pepstab.foldx import ConversionError, prepare_pdb  # noqa: E402
from pepstab.structural_features import discover_folds  # noqa: E402

#: Heavy-atom counts for the standard residues, excluding OXT (a terminal
#: oxygen Boltz does not write). A residue short of its count has a missing
#: side chain, which is what RepairPDB exists to rebuild.
HEAVY_ATOMS = {
    "GLY": 4, "ALA": 5, "SER": 6, "CYS": 6, "THR": 7, "PRO": 7, "VAL": 7,
    "ASN": 8, "ASP": 8, "ILE": 8, "LEU": 8, "MET": 8, "GLN": 9, "GLU": 9,
    "LYS": 9, "HIS": 10, "PHE": 11, "ARG": 11, "TYR": 12, "TRP": 14,
}

#: Heavy-atom pairs closer than this, excluding same and adjacent residues,
#: are overlaps. 2.2 A sits below every real non-bonded contact and above a
#: 2.05 A disulfide, which is why those are classified separately.
CLASH_A = 2.2


def structural_audit(cif: Path) -> dict:
    """Missing side-chain atoms and genuine steric overlaps in one fold."""
    import biotite.structure as struc
    from biotite.structure.io.pdbx import CIFFile, get_structure

    atoms = get_structure(CIFFile.read(str(cif)), model=1)
    atoms = atoms[~np.isin(atoms.element, ("H", "D"))]

    missing, nonstandard = [], []
    starts = struc.get_residue_starts(atoms, add_exclusive_stop=True)
    for a, b in zip(starts[:-1], starts[1:]):
        name = str(atoms.res_name[a])
        if name not in HEAVY_ATOMS:
            nonstandard.append(name)
        elif (b - a) != HEAVY_ATOMS[name]:
            missing.append(f"{atoms.chain_id[a]}{atoms.res_id[a]}{name}:{b - a}")

    cell = struc.CellList(atoms, CLASH_A + 0.2)
    disulfides, overlaps = 0, []
    for i in range(atoms.array_length()):
        for j in cell.get_atoms(atoms.coord[i], CLASH_A):
            j = int(j)
            if j <= i:
                continue
            same_chain = atoms.chain_id[i] == atoms.chain_id[j]
            if same_chain and abs(int(atoms.res_id[i]) - int(atoms.res_id[j])) <= 1:
                continue
            if atoms.atom_name[i] == "SG" and atoms.atom_name[j] == "SG":
                disulfides += 1
                continue
            d = float(np.linalg.norm(atoms.coord[i] - atoms.coord[j]))
            overlaps.append(
                f"{atoms.chain_id[i]}{atoms.res_id[i]}{atoms.res_name[i]}:"
                f"{atoms.atom_name[i]}--{atoms.chain_id[j]}{atoms.res_id[j]}"
                f"{atoms.res_name[j]}:{atoms.atom_name[j]}@{d:.2f}"
            )
    return {
        "n_residues": len(starts) - 1,
        "missing_atom_residues": missing,
        "nonstandard_residues": sorted(set(nonstandard)),
        "disulfides": disulfides,
        "overlaps": overlaps,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="tree containing fold directories")
    ap.add_argument("--out", type=Path, required=True, help="where to write PDBs")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    folders = list(discover_folds(args.root))
    if args.limit:
        folders = folders[: args.limit]
    if not folders:
        print(f"no fold directories under {args.root}", file=sys.stderr)
        return 1

    rows, failures = [], 0
    for folder in folders:
        try:
            prepared = prepare_pdb(folder, args.out)
            audit = structural_audit(next(iter(sorted(folder.glob("*.cif")))))
            rows.append({
                "complex_id": prepared["complex_id"],
                "fold_dir": str(folder),
                "chain_runs": prepared["conversion"]["chain_runs"],
                "n_atoms": prepared["conversion"]["n_atoms"],
                "peptide_chain": prepared["peptide_chain"],
                "peptide": prepared["peptide"],
                "max_coord_dev_A": prepared["validation"]["max_coord_dev_A"],
                "ter_records": prepared["validation"]["ter_records"],
                **audit,
                "status": "ok",
            })
        except (ConversionError, Exception) as exc:  # noqa: BLE001
            failures += 1
            rows.append({"fold_dir": str(folder), "status": f"{type(exc).__name__}: {exc}"})

    ok = [r for r in rows if r["status"] == "ok"]
    missing = sum(len(r["missing_atom_residues"]) for r in ok)
    overlaps = sum(len(r["overlaps"]) for r in ok)
    disulfides = sum(r["disulfides"] for r in ok)
    print(f"{len(ok)}/{len(rows)} converted and validated, {failures} failed")
    print(f"  chain layouts        {sorted({str(r['chain_runs']) for r in ok})}")
    print(f"  max coordinate dev   "
          f"{max((r['max_coord_dev_A'] for r in ok), default=0):.4f} A "
          f"(PDB writes three decimals, so <=0.001 is the format, not a move)")
    print(f"  residues missing a heavy atom   {missing}  "
          f"<- RepairPDB's usual job; 0 means predicted models give it nothing to do")
    print(f"  SG-SG disulfide pairs           {disulfides} "
          f"({disulfides / max(len(ok), 1):.1f} per fold, bonds not clashes)")
    print(f"  genuine steric overlaps <{CLASH_A} A   {overlaps} "
          f"({overlaps / max(len(ok), 1):.2f} per fold)")
    for r in ok:
        for o in r["overlaps"]:
            print(f"      {r['complex_id']}: {o}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
