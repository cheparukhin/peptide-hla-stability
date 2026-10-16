#!/usr/bin/env python3
"""Freeze crystal contacts or derive compact features from a validated cache list."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pepstab.boltz_embeddings import _atomic_bytes, json_bytes
from pepstab.round2_representations import (
    REPORT_DIR, build_contact_map, build_feature_bundle, feature_protocol, load_contact_map,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-protocol", action="store_true")
    parser.add_argument("--crystal-directory", type=Path, default=ROOT / "structures/crystal_cache")
    parser.add_argument("--contact-map", type=Path, default=REPORT_DIR / "contact_map.json")
    parser.add_argument("--manifest", type=Path, help="JSON list (or entries list) with pair_id,input_hash,cache_path")
    parser.add_argument("--cache-list", type=Path, help="Optional newline cache paths, in manifest row order")
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    if args.freeze_protocol:
        contact_map = build_contact_map(args.crystal_directory)
        args.contact_map.parent.mkdir(parents=True, exist_ok=True)
        _atomic_bytes(args.contact_map, json_bytes(contact_map))
        _atomic_bytes(args.contact_map.parent / "feature_protocol.json", json_bytes(feature_protocol()))
        print(json.dumps({"contact_map_sha256": contact_map["contact_map_sha256"],
                          "contacts_per_position": [sum(row) for row in contact_map["mask"]]}))
    if args.manifest:
        if args.output_directory is None:
            parser.error("--output-directory is required with --manifest")
        manifest = json.loads(args.manifest.read_text())
        entries = manifest if isinstance(manifest, list) else manifest["entries"]
        entries = [dict(entry) for entry in entries]
        if args.cache_list:
            paths = [x.strip() for x in args.cache_list.read_text().splitlines() if x.strip()]
            if len(paths) != len(entries):
                raise ValueError("cache path list and manifest row counts differ")
            for entry, path in zip(entries, paths):
                entry["cache_path"] = path
        for entry in entries:
            path = Path(entry["cache_path"])
            if not path.is_absolute():
                entry["cache_path"] = str(args.manifest.parent / path)
        result = build_feature_bundle(entries, args.output_directory, load_contact_map(args.contact_map))
        print(json.dumps({"n_rows": result["n_rows"], "dimensions": result["dimensions"],
                          "array_file": result["array_file"]}))
    elif not args.freeze_protocol:
        parser.error("provide --freeze-protocol or --manifest")


if __name__ == "__main__":
    main()
