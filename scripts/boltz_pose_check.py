"""Check the pilot's predicted peptides actually sit in the HLA groove.

HACKATHON_PLAN.md stage 4 requires inspecting the pilot poses and verifying
residue/chain mapping before any batch job; STRUCTURES.md asks for the same
thing as a "register check" -- does the predicted peptide land in the canonical
P2/POmega-anchored conformation *where the answer is already known*?

Every pilot complex has a crystal structure at <= 1.9 A, so this is answerable
numerically rather than by eye:

1. Superpose the predicted HLA domain onto the crystal's, using CA atoms only.
2. Apply that transform to the predicted peptide.
3. Report peptide CA RMSD, and per-position deviation for the anchors.

Superposing on the **HLA** and then measuring the **peptide** is the whole
point. Superposing on the peptide itself would hide a peptide docked in the
wrong place or the wrong register -- it would simply rotate onto the answer.

A pose is called good below 2.0 A peptide CA RMSD. That threshold is the
conventional bar for "same binding mode" in pMHC redocking, not something this
dataset establishes, so the per-position numbers are printed alongside it.

Usage:
    python scripts/boltz_pose_check.py --structures <dir>

``<dir>`` holds one subdirectory per complex_id with the predicted mmCIF, as
written by the Modal benchmark's structures Volume.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PILOT = REPO / "reports" / "boltz_pilot.csv"
OUT = REPO / "reports" / "boltz_pose_check.csv"

GOOD_RMSD = 2.0

# CLAUDE.md gates a batch GPU job on "a passing end-to-end pilot on 3-5
# examples", so three scored complexes is the floor.
MIN_PILOT = 3

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "U", "PYL": "O",
}


def _require_biotite():
    try:
        import biotite.structure as struc  # noqa: F401
        from biotite.database import rcsb  # noqa: F401
        from biotite.structure.io.pdbx import CIFFile, get_structure  # noqa: F401
    except ImportError:
        raise SystemExit(
            "biotite is required: uv pip install --python .venv/bin/python biotite"
        )


def ca_by_chain(atoms) -> dict[str, tuple[str, np.ndarray]]:
    """Per chain: its one-letter sequence and the matching CA coordinates.

    One CA per residue, in residue order, so sequence index i and coordinate
    row i always refer to the same residue. Altloc and insertion-code
    duplicates are dropped by keeping the first CA seen per residue key.
    """
    import biotite.structure as struc

    out: dict[str, tuple[str, np.ndarray]] = {}
    for chain_id in struc.get_chains(atoms):
        chain = atoms[atoms.chain_id == chain_id]
        chain = chain[chain.atom_name == "CA"]
        if chain.array_length() == 0:
            continue
        seq, coords, seen = [], [], set()
        ins = (
            chain.ins_code
            if "ins_code" in chain.get_annotation_categories()
            else [""] * chain.array_length()
        )
        for i in range(chain.array_length()):
            one = THREE_TO_ONE.get(chain.res_name[i].upper())
            if one is None:
                continue
            key = (chain.res_id[i], str(ins[i]))
            if key in seen:
                continue
            seen.add(key)
            seq.append(one)
            coords.append(chain.coord[i])
        if seq:
            out[chain_id] = ("".join(seq), np.asarray(coords, dtype=float))
    return out


def find_subsequence(chains: dict, query: str) -> tuple[str, int] | None:
    """Locate an exact subsequence; returns (chain_id, offset)."""
    for chain_id, (seq, _) in chains.items():
        idx = seq.find(query)
        if idx >= 0:
            return chain_id, idx
    return None


def load_prediction(path: Path):
    from biotite.structure.io.pdbx import CIFFile, get_structure

    atoms = get_structure(CIFFile.read(path), model=1)
    return atoms[~atoms.hetero] if "hetero" in atoms.get_annotation_categories() else atoms


def load_crystal(pdb_id: str, cache: Path):
    from biotite.database import rcsb
    from biotite.structure.io.pdbx import CIFFile, get_structure

    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{pdb_id}.cif"
    if not path.exists():
        rcsb.fetch(pdb_id, "cif", target_path=str(cache))
    atoms = get_structure(CIFFile.read(path), model=1)
    return atoms[~atoms.hetero] if "hetero" in atoms.get_annotation_categories() else atoms


def check_one(row: dict, pred_path: Path, cache: Path) -> dict:
    import biotite.structure as struc

    rec = {
        "complex_id": row["complex_id"],
        "allele": row["allele"],
        "peptide": row["peptide"],
        "thalf_hours": row["thalf_hours"],
        "pdb_id": "",
        "status": "",
    }

    pred = load_prediction(pred_path)
    pred_chains = ca_by_chain(pred)

    # The predicted peptide chain: the one whose sequence is exactly the peptide.
    hit = find_subsequence(pred_chains, row["peptide"])
    if hit is None:
        rec["status"] = "peptide chain not found in prediction"
        return rec
    pep_chain, pep_off = hit
    pep_seq, pep_ca = pred_chains[pep_chain]
    if len(pep_seq) != len(row["peptide"]):
        rec["status"] = f"predicted chain {pep_chain} is not peptide-only"
        return rec
    rec["pred_peptide_chain"] = pep_chain

    # The predicted HLA chain: the longest remaining chain.
    hla_candidates = {k: v for k, v in pred_chains.items() if k != pep_chain}
    if not hla_candidates:
        rec["status"] = "no HLA chain in prediction"
        return rec
    hla_chain = max(hla_candidates, key=lambda k: len(hla_candidates[k][0]))
    hla_seq, hla_ca = pred_chains[hla_chain]
    rec["pred_hla_chain"] = hla_chain
    rec["pred_hla_len"] = len(hla_seq)

    # Try each deposited entry; take the best-resolving match.
    best = None
    for pdb_id in row["pdb_ids"].split():
        try:
            xtal = load_crystal(pdb_id, cache)
        except Exception as exc:
            rec["status"] = f"fetch {pdb_id} failed: {exc}"
            continue
        xtal_chains = ca_by_chain(xtal)

        x_hla = find_subsequence(xtal_chains, hla_seq)
        x_pep = find_subsequence(xtal_chains, row["peptide"])
        if x_hla is None or x_pep is None:
            continue
        xh_chain, xh_off = x_hla
        xp_chain, xp_off = x_pep
        if xh_chain == xp_chain:
            continue  # same chain cannot be both

        xh_ca = xtal_chains[xh_chain][1][xh_off : xh_off + len(hla_seq)]
        xp_ca = xtal_chains[xp_chain][1][xp_off : xp_off + len(row["peptide"])]
        if len(xh_ca) != len(hla_ca) or len(xp_ca) != len(pep_ca):
            continue

        # Fit on the HLA only, then measure the peptide under that transform.
        _, transform = struc.superimpose(xh_ca, hla_ca)
        moved_hla = transform.apply(hla_ca)
        moved_pep = transform.apply(pep_ca)

        hla_rmsd = float(np.sqrt(np.mean(np.sum((moved_hla - xh_ca) ** 2, axis=1))))
        dev = np.sqrt(np.sum((moved_pep - xp_ca) ** 2, axis=1))
        pep_rmsd = float(np.sqrt(np.mean(dev**2)))

        cand = {
            "pdb_id": pdb_id,
            "xtal_hla_chain": xh_chain,
            "xtal_pep_chain": xp_chain,
            "hla_ca_rmsd": round(hla_rmsd, 3),
            "peptide_ca_rmsd": round(pep_rmsd, 3),
            "peptide_max_dev": round(float(dev.max()), 3),
            # P2 and POmega are the anchor positions pinned in the groove.
            "dev_p2": round(float(dev[1]), 3),
            "dev_p9": round(float(dev[-1]), 3),
            "per_position": " ".join(f"{d:.2f}" for d in dev),
            "in_groove": bool(pep_rmsd <= GOOD_RMSD),
            "status": "ok",
        }
        if best is None or cand["peptide_ca_rmsd"] < best["peptide_ca_rmsd"]:
            best = cand

    if best is None:
        rec["status"] = rec["status"] or "no crystal chain pair matched"
        return rec
    rec.update(best)
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--structures",
        type=Path,
        required=True,
        help="directory holding one subdirectory per complex_id with the predicted mmCIF",
    )
    p.add_argument("--cache", type=Path, default=Path("/tmp/rcsb_cache"))
    args = p.parse_args()
    _require_biotite()

    rows = list(csv.DictReader(PILOT.open()))
    results = []
    for row in rows:
        cands = sorted(
            (args.structures / row["complex_id"]).glob("*.cif")
        ) or sorted(args.structures.glob(f"{row['complex_id']}*/**/*.cif"))
        if not cands:
            results.append(
                {**{k: row[k] for k in ("complex_id", "allele", "peptide", "thalf_hours")},
                 "status": "no predicted mmCIF found"}
            )
            continue
        results.append(check_one(row, cands[0], args.cache))

    cols = [
        "complex_id", "allele", "peptide", "thalf_hours", "pdb_id",
        "pred_hla_chain", "pred_peptide_chain", "pred_hla_len",
        "xtal_hla_chain", "xtal_pep_chain", "hla_ca_rmsd", "peptide_ca_rmsd",
        "peptide_max_dev", "dev_p2", "dev_p9", "in_groove", "per_position",
        "status",
    ]
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"wrote {OUT.relative_to(REPO)}\n")

    hdr = f"{'complex':22} {'PDB':6} {'HLA fit':>8} {'pep RMSD':>9} {'max':>6} {'verdict':>9}"
    print(hdr)
    print("-" * len(hdr))
    good = 0
    for r in results:
        if r.get("status") != "ok":
            print(f"{r['complex_id']:22} {'--':6} {r.get('status', 'error')}")
            continue
        verdict = "in groove" if r["in_groove"] else "OFF"
        good += bool(r["in_groove"])
        print(
            f"{r['complex_id']:22} {r['pdb_id']:6} {r['hla_ca_rmsd']:8.2f} "
            f"{r['peptide_ca_rmsd']:9.2f} {r['peptide_max_dev']:6.2f} {verdict:>9}"
        )

    # "Not folded" is not "failed". A cost-reduced pilot deliberately folds a
    # subset, and counting the unfolded rows as failures would block the gate
    # on complexes nobody ran.
    scored = [r for r in results if r.get("status") == "ok"]
    not_run = [r for r in results if r.get("status") == "no predicted mmCIF found"]
    broken = [
        r for r in results
        if r.get("status") not in ("ok", "no predicted mmCIF found")
    ]
    print(f"\n{good}/{len(scored)} folded complexes in groove at <= {GOOD_RMSD} A peptide CA RMSD")
    if not_run:
        print(f"{len(not_run)} not folded (skipped, not failed): "
              + ", ".join(r["complex_id"] for r in not_run))
    if scored:
        vals = [r["peptide_ca_rmsd"] for r in scored]
        print(f"peptide CA RMSD: min {min(vals):.2f}, median "
              f"{sorted(vals)[len(vals) // 2]:.2f}, max {max(vals):.2f} A")
        print("\nper-position CA deviation (P1..P9), angstrom:")
        for r in scored:
            print(f"  {r['complex_id']:22} {r['per_position']}")

    # CLAUDE.md's gate: a passing end-to-end pilot on 3-5 examples. So the bar
    # is at least MIN_PILOT scored complexes, every one of them in the groove,
    # and nothing that errored while being checked.
    if broken:
        print("\nPILOT NOT CLEAN -- these could not be scored:")
        for r in broken:
            print(f"  {r['complex_id']}: {r.get('status')}")
        sys.exit(1)
    if good < len(scored):
        print(
            f"\nPILOT NOT CLEAN -- {len(scored) - good} folded pose(s) sit outside "
            f"the groove at > {GOOD_RMSD} A. Resolve before running ::benchmark."
        )
        sys.exit(1)
    if len(scored) < MIN_PILOT:
        print(
            f"\nPilot poses all check out, but only {len(scored)} complexes were "
            f"folded. CLAUDE.md's gate wants at least {MIN_PILOT} -- fold more "
            "before the batch job."
        )
        sys.exit(1)
    print(
        f"\nPilot poses check out: {good}/{len(scored)} in groove. "
        "The batch gate is clear."
    )


if __name__ == "__main__":
    main()
