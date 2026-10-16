"""Operational namespace wrapper; original frozen energy science stays intact."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pepstab.followup3_energy import digest
from scripts.followup3_energy import sha, write_once

OUT = ROOT / "reports/followup3/energy_execution_v2"
PARENT = ROOT / "reports/followup3/energy"
NEW_SOURCES = ("modal_app/followup3_energy_v2.py", "scripts/followup3_energy_v2.py",
               "reports/followup3/energy_execution_v2/EXECUTION_AMENDMENT.md")


def freeze(out=OUT, parent=PARENT):
    from scripts.followup3_energy import verify as verify_parent
    out, parent = Path(out), Path(parent)
    panel, original = verify_parent(parent)
    out.mkdir(parents=True, exist_ok=True)
    snapshots = out / "original_source_snapshot"
    for name in ("protocol.json", "panel.csv", "pilot_energy_rows.json"):
        source = parent / name
        destination = out / name
        if name != "protocol.json":
            if destination.exists() and destination.read_bytes() != source.read_bytes():
                raise ValueError("same original scientific artifacts required")
            destination.write_bytes(source.read_bytes())
        saved = snapshots / "reports/followup3/energy" / name
        saved.parent.mkdir(parents=True, exist_ok=True)
        if saved.exists() and saved.read_bytes() != source.read_bytes():
            raise ValueError("original protocol snapshot changed")
        saved.write_bytes(source.read_bytes())
    # Preserve all original hash-bound implementation bytes, not just a digest.
    for name, expected in original["source_sha256"].items():
        source = ROOT / name
        if sha(source) != expected:
            raise ValueError("original frozen source changed before snapshot")
        destination = snapshots / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha(destination) != expected:
            raise ValueError("saved original source snapshot changed")
        if not destination.exists():
            shutil.copyfile(source, destination)
    reused = json.loads((out / "pilot_energy_rows.json").read_text())
    keys = {(row["allele"], row["peptide"]) for row in reused}
    first = next(row for row in panel.to_dict(orient="records") if (row["allele"], row["peptide"]) not in keys)
    amended = dict(original)
    amended.pop("protocol_hash")
    amended["source_sha256"] = dict(original["source_sha256"], **{name:sha(ROOT / name) for name in NEW_SOURCES})
    amended["execution_amendment"] = dict(schema="followup3-energy-execution-v2.1", frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                parent_directory=str(parent.resolve().relative_to(ROOT)), parent_protocol_sha256=sha(parent / "protocol.json"),
                parent_protocol_hash=original["protocol_hash"], output_directory=str(out.resolve().relative_to(ROOT)),
                native_runtime="pinned image libraries off mounted volume;documented reload/commit retained",
                pinned_image_packages={"numpy":"2.5.3", "pandas":"3.0.6", "scipy":"1.18.1", "biotite":"1.7.1"},
                first_uncached_case=first, production_path_pilot=1, pilot_reused_before_fanout=True)
    amended["protocol_hash"] = digest(amended)
    write_once(out / "protocol.json", amended)
    print(json.dumps(dict(protocol_hash=amended["protocol_hash"], same_scientific_specification=amended["specification"] == original["specification"],
                         panel_rows=len(panel), first_uncached_case=first)))


def verify(out):
    from scripts.followup3_energy import verify as verify_science
    panel, protocol = verify_science(out)
    amendment = protocol["execution_amendment"]
    parent = ROOT / amendment["parent_directory"]
    if sha(parent / "protocol.json") != amendment["parent_protocol_sha256"]:
        raise ValueError("original failed-run protocol changed")
    original = json.loads((parent / "protocol.json").read_text())
    if original["protocol_hash"] != amendment["parent_protocol_hash"] or original["specification"] != protocol["specification"]:
        raise ValueError("execution amendment changed original scientific scope")
    if sha(parent / "panel.csv") != protocol["panel_sha256"] or sha(parent / "pilot_energy_rows.json") != protocol["pilot_energy_rows_sha256"]:
        raise ValueError("execution amendment changed original panel/pilot reuse")
    return panel, protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "collect", "assess", "report"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--parent-out", type=Path, default=PARENT)
    parser.add_argument("--rows-root", type=Path)
    parser.add_argument("--learner", choices=("mlp", "ridge"), default="mlp")
    parser.add_argument("--outer-fold", type=int, choices=range(5), default=0)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.out, args.parent_out)
        return
    verify(args.out)
    from scripts import followup3_energy as science
    if args.command == "collect": science.collect(args.out, args.rows_root)
    elif args.command == "assess": science.assess(args.out, args.learner, args.outer_fold)
    else: science.report(args.out)


if __name__ == "__main__": main()
