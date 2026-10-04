"""Capture provenance immediately before the single test-scoring pass.

Written 4 October 2026, before any test number existed. Step 2 of
``reports/TEST_SCORING_RUNBOOK.md``: record what is being scored, and from what
tree, *before* scoring it. That is what makes "selected on validation, scored
once on test" checkable by someone who was not here, rather than asserted.

    .venv/bin/python scripts/capture_test_provenance.py \\
        --arm seq_ensemble=preds/seq_ensemble_pep_pseudo.csv \\
        --arm esm_ensemble=preds/esm_ensemble.csv \\
        --out reports/test_scoring_provenance.json

Refuses to overwrite an existing file. A second capture would mean a second
pass, and the runbook requires both to be disclosed rather than one quietly
replacing the other.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parent.parent

#: Hashed alongside the arms: if either changed, no earlier comparison applies.
PINNED_INPUTS = ("data/splits.csv", "data/rasmussen_et_al_dataset.csv")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(path: Path) -> dict:
    if not path.is_file():
        sys.exit(f"missing: {path}")
    return {
        "path": str(path.relative_to(REPO)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "rows": sum(1 for _ in path.open()) - 1,
    }


def git(*args: str) -> str:
    out = subprocess.run(
        ("git", *args), cwd=REPO, capture_output=True, text=True, check=False
    )
    return out.stdout.strip() if out.returncode == 0 else "<unavailable>"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=preds/file.csv",
        help="an arm entering the pass; repeat per arm",
    )
    ap.add_argument("--out", default="reports/test_scoring_provenance.json")
    ap.add_argument(
        "--note",
        default="",
        help="free text, e.g. which arms were excluded at the cutoff and why",
    )
    args = ap.parse_args()

    out = REPO / args.out
    if out.exists():
        sys.exit(
            f"{out} exists. A second capture means a second pass; the runbook "
            "requires both to be disclosed. Move the first aside deliberately."
        )

    arms = {}
    for spec in args.arm:
        if "=" not in spec:
            sys.exit(f"--arm wants NAME=path, got {spec!r}")
        name, _, rel = spec.partition("=")
        arms[name] = describe(REPO / rel)

    now = datetime.now(timezone.utc)
    record = {
        "captured_utc": now.isoformat(timespec="seconds"),
        "captured_london": now.astimezone(ZoneInfo("Europe/London")).isoformat(
            timespec="seconds"
        ),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "arms": arms,
        "pinned_inputs": {p: describe(REPO / p) for p in PINNED_INPUTS},
        "note": args.note,
    }
    out.write_text(json.dumps(record, indent=2) + "\n")

    try:
        shown = out.relative_to(REPO)
    except ValueError:  # an --out outside the repo, e.g. a scratchpad dry run
        shown = out
    print(f"wrote {shown}")
    print(f"  commit {record['git_commit'][:12]}  dirty={record['git_dirty']}")
    for name, meta in arms.items():
        print(f"  {name:28s} {meta['rows']:>6,} rows  {meta['sha256'][:12]}")
    if record["git_dirty"]:
        print(
            "\nNOTE: the working tree is dirty. That is not fatal, but the "
            "commit alone will not reproduce this pass -- say so in the "
            "write-up, or commit first and re-run."
        )


if __name__ == "__main__":
    main()
