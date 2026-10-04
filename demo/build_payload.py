"""Build the demo's data payload from the frozen project data.

Emits `payload.js` (a single `window.__DATA__ = {...}` assignment) next to
`index.html`, so the demo is a static page with no network calls and no build
step beyond this script.

Reads only committed, frozen inputs:

  data/rasmussen_et_al_dataset.csv                     measured half-lives
  data/allele_pdb_templates.csv                        crystallographic template per allele
  data/pdb_rasmussen_overlap.csv                       measured pairs that are also solved
  reports/stage6_val_esm_per_allele.csv                per-allele validation Spearman
  reports/stage7_allele_holdout_per_allele_*.csv       leave-one-allele-out

  demo/structures.json (optional)                      Boltz-2 Cα traces

Nothing here recomputes splits or writes to `data/`. Run from anywhere:

    python3 demo/build_payload.py

To include Boltz-2 structures, run ``fetch_structures.py`` first to produce
``demo/structures.json``, then rebuild the payload.
"""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "payload.js"

# The validation arm whose per-allele scores the demo shows. This is the
# baseline the whole study is measured against (the 30-network ensemble on the
# 34 contact residues), not an ESM arm.
BASELINE_ARM = "seq_ensemble_pep_pseudo"

# Peptides listed per allele, longest half-life first. The full set is 28,166
# rows; the demo shows the top slice plus every structurally solved pair.
CAP = 12


def rows(rel: str) -> list[dict]:
    with (ROOT / rel).open(newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build() -> dict:
    # --- per-allele validation Spearman, baseline arm only ---
    val = {}
    for r in rows("reports/stage6_val_esm_per_allele.csv"):
        if r["model"] == BASELINE_ARM:
            val[r["allele"]] = {
                "rho": fnum(r["spearman"]),
                "n": int(r["n"]),
                "p_at_k": fnum(r["precision_at_k"]),
            }

    # --- leave-one-allele-out: retrain on 67 alleles, predict the 68th ---
    holdout = {}
    for r in rows(
        "reports/stage7_allele_holdout_per_allele_seq_pep_pseudo.csv"
    ):
        holdout[r["held_out_allele"]] = {
            "rho": fnum(r["spearman"]),
            "n": int(r["n_rows"]),
            "dist": int(r["dist_to_nearest_fit_allele"]),
            "stratum": r["stratum"].split(" (")[0],
            "p10": fnum(r["precision_at_10"]),
        }

    # --- crystallographic template per allele ---
    templates = {}
    for r in rows("data/allele_pdb_templates.csv"):
        templates[r["allele"]] = {
            "pdb": r["template"],
            "tier": r["tier"].split(":")[0],
            "tier_text": r["tier"],
            "res": fnum(r["resolution"]),
            "year": r["year"],
            "n_exact": int(r["n_exact_entries"]),
            "tcr": int(r["tcr_complexes"]),
            "mismatch": int(r["pocket_mismatches"]),
            "title": r["title"],
            "link": r["rcsb_link"],
        }

    # --- measured pairs that also have an experimental structure ---
    overlap = []
    for r in rows("data/pdb_rasmussen_overlap.csv"):
        overlap.append(
            {
                "allele": r["allele"],
                "peptide": r["peptide"],
                "thalf": fnum(r["rasmussen_thalf"]),
                "pdbs": r["pdb_ids"].split(),
                "res": fnum(r["best_resolution"]),
                "year": r["earliest_year"],
                "tcr": r["any_tcr"] == "True",
                "link": r["rcsb_link"],
            }
        )
    overlap.sort(key=lambda d: -(d["thalf"] or 0))
    solved = {(d["allele"], d["peptide"]) for d in overlap}

    # --- measured peptides per allele ---
    per = collections.defaultdict(list)
    pseudoseq: dict[str, str] = {}
    domain: dict[str, str] = {}
    for r in rows("data/rasmussen_et_al_dataset.csv"):
        allele, peptide = r["allele"], r["peptide"]
        pseudoseq.setdefault(allele, r["hla_pseudoseq"])
        domain.setdefault(allele, r["hla_seq"])
        per[allele].append((peptide, fnum(r["thalf_hours"])))

    alleles = []
    for allele in sorted(per):
        ranked = sorted(per[allele], key=lambda t: -(t[1] or 0))
        keep = ranked[:CAP]
        have = {p for p, _ in keep}
        for peptide, thalf in ranked:  # always keep solved pairs
            if (allele, peptide) in solved and peptide not in have:
                keep.append((peptide, thalf))
                have.add(peptide)
        alleles.append(
            {
                "allele": allele,
                "pseudoseq": pseudoseq[allele],
                "domain_len": len(domain[allele]),
                "n_pairs": len(ranked),
                "n_floor": sum(1 for _, t in ranked if (t or 0) == 0),
                "thalf_max": ranked[0][1],
                "val": val.get(allele),
                "holdout": holdout.get(allele),
                "tpl": templates.get(allele),
                "peptides": [
                    {"p": p, "t": t, "s": (allele, p) in solved}
                    for p, t in keep
                ],
            }
        )

    # --- Boltz-2 structures (optional, from fetch_structures.py) ---
    structures = {}
    structs_path = Path(__file__).resolve().parent / "structures.json"
    if structs_path.exists():
        structures = json.loads(structs_path.read_text())

    return {
        "alleles": alleles,
        "overlap": overlap,
        "structures": structures,
        "totals": {
            "pairs": sum(len(v) for v in per.values()),
            "alleles": len(per),
            "structure_pairs": len(overlap),
            "structures_loaded": len(structures),
            "templates": len(templates),
            "tier_a": sum(1 for v in templates.values() if v["tier"] == "A"),
        },
    }


def main() -> None:
    payload = build()
    blob = json.dumps(payload, separators=(",", ":"))
    OUT.write_text("window.__DATA__=" + blob + ";\n")

    t = payload["totals"]
    scored = sum(1 for a in payload["alleles"] if a["val"])
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(blob):,} bytes)")
    print(
        f"  {t['pairs']:,} pairs · {t['alleles']} alleles · "
        f"{t['templates']} templates ({t['tier_a']} exact groove) · "
        f"{t['structure_pairs']} measured-and-solved pairs"
    )
    print(f"  {scored} alleles carry a validation score")
    if t["structures_loaded"]:
        print(f"  {t['structures_loaded']} Boltz-2 structures embedded")


if __name__ == "__main__":
    main()
