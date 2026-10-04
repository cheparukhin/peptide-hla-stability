"""Stage the arm-B MSA slice for a workspace that does not already hold it.

Only needed for a *fresh* workspace. ``a-cheparukhin`` already has all 150
production CSVs on ``pepstab-hla-msa`` from the pilot, so it needs no upload at
all -- ``::setup`` just hash-checks them.

``modal volume put`` takes one local path, and the prepared-MSA directory is
745 MB across 608 files covering all three pilot arms. Production folds arm B
only and Boltz reads the CSV form, so a fresh workspace needs just ``*_B.csv``
+ ``*_beta2m.csv``: 150 files, 29.6 MB. This script is the staging directory
that difference requires, and it re-verifies every file against the committed
manifest while copying.

    python scripts/stage_production_msas.py
    MODAL_PROFILE=colleague modal volume put pepstab-hla-msa \
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

    staged, total = 0, 0
    for m in manifest["msas"]:
        for key in ("B", "beta2m"):
            name = m[key]["csv"]
            src = SRC / name
            assert digest(src) == m[key]["csv_sha256"], f"{src} does not match manifest"
            shutil.copy2(src, DEST / name)
            total += src.stat().st_size
            staged += 1

    # No manifest is written alongside: ``::setup`` hash-checks the Volume
    # against the committed manifest, so a second copy here could only drift.
    print(f"staged {staged} CSVs for {len(manifest['msas'])} alleles, {total / 1e6:.1f} MB")
    print(f"  -> {DEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
