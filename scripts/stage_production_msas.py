"""Stage the arm-B MSA slice a production workspace needs, and its manifest.

Production folds arm B only, and Boltz reads the CSV form, so a workspace needs
just ``*_B.csv`` + ``*_beta2m.csv`` (~29 MB) rather than the full 745 MB of
prepared alignments. Writes a staging directory plus a trimmed
``production_manifest.json`` that ``verify_workspace`` hash-checks on the
Volume before any GPU is billed.

    python scripts/stage_production_msas.py
    MODAL_PROFILE=<profile> modal volume put pepstab-hla-msa \
        structures/ectodomain_production_msas /ectodomain_stage4c
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "structures" / "ectodomain_msas"
DEST = ROOT / "structures" / "ectodomain_production_msas"


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads((SRC / "manifest.json").read_text())
    DEST.mkdir(parents=True, exist_ok=True)

    msas, total = [], 0
    for m in manifest["msas"]:
        entry = {"allele": m["allele"], "ecto": m["ecto"], "b2m": m["b2m"]}
        for key in ("B", "beta2m"):
            name = m[key]["csv"]
            src = SRC / name
            assert digest(src) == m[key]["csv_sha256"], f"{src} does not match manifest"
            shutil.copy2(src, DEST / name)
            total += src.stat().st_size
            entry[key] = {"csv": name, "csv_sha256": m[key]["csv_sha256"], "depth": m[key]["depth"]}
        msas.append(entry)

    out = {"cap": manifest["cap"], "arm": "B", "alleles": len(msas), "msas": msas}
    (DEST / "production_manifest.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"staged {len(msas)} alleles, {2 * len(msas)} CSVs, {total / 1e6:.1f} MB")
    print(f"  -> {DEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
