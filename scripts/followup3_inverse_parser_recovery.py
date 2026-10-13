"""Source-bound parser recovery and one bounded continuation;root launches."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from pepstab import followup3_inverse as science
from pepstab import followup3_inverse_parser as parser_fix
from scripts import followup3_inverse as original_cli

OUT = ROOT / "reports/followup3/inverse_parser_recovery"
INVERSE = ROOT / "reports/followup3/inverse_execution_v2"
FAILED_SHA = "1972c68c52a91a06d846df8910fd0ebdf240fc1320fb0959efa01a704b56bb79"
SOURCE_FILES = ("pepstab/followup3_inverse_parser.py", "scripts/followup3_inverse_parser_recovery.py",
                "reports/followup3/inverse_parser_recovery/SOURCE_BOUND_PARSER_AMENDMENT.md",
                "reports/followup3/inverse_parser_recovery/FAILED_STRUCTURE_INSPECTION.json")


def sha(path):
    return original_cli.sha(path)


def prepare(out=OUT, inverse=INVERSE):
    out, inverse = Path(out), Path(inverse)
    panel, protocol = original_cli.read_panel(inverse)
    records = {p.name:json.loads(p.read_text()) for p in (inverse / "score_rows").glob("*.json")}
    good = {name:r for name, r in records.items() if r.get("status") == "ok"}
    failed = [(name,r) for name,r in records.items() if r.get("status") != "ok"]
    if len(records) != 4164 or len(good) != 4163 or len(failed) != 1:
        raise ValueError("exact reviewed4163OK+oneparserfailure checkpoint required")
    failed_name, failed_row = failed[0]
    if (failed_row["allele"], failed_row["peptide"]) != ("HLA-B*56:01", "YPAEITLTW") or not failed_row["error"].startswith("ValueError:peptide chain A is not peptide-only"):
        raise ValueError("only the demonstrated parser failure is recoverable")
    if sha(inverse / "score_rows" / failed_name) != FAILED_SHA:
        raise ValueError("original failed bytes differ")
    keys = {(r["allele"],r["peptide"]) for r in good.values()}
    subset = panel.loc[[tuple(x) in keys for x in panel[["allele","peptide"]].itertuples(index=False, name=None)]]
    science.validate_scores(subset, pd.DataFrame(good.values()))
    to_complete = [r for r in panel.to_dict(orient="records") if (r["allele"],r["peptide"]) not in keys]
    if len(to_complete) != 286:
        raise ValueError("exact286remaining-or-corrected cases required")
    manifest = dict(schema="inverse_source_bound_parser_recovery.1", status="frozen_no_paid_launch_authorization",
                    frozen_at_utc=datetime.now(timezone.utc).isoformat(), inverse_directory=str(inverse.resolve().relative_to(ROOT)),
                    original_protocol_file_sha256=sha(inverse / "protocol.json"), original_scientific_hash=protocol["protocol_hash"],
                    panel_sha256=protocol["panel_sha256"], selector="unique-whole-chain-exact-native-v1",
                    source_sha256={p:sha(ROOT / p) for p in SOURCE_FILES},
                    inherited_ok_row_sha256={name:sha(inverse / "score_rows" / name) for name in good},
                    original_failed_filename=failed_name, original_failed_sha256=FAILED_SHA,
                    corrected_and_missing_rows=to_complete, threads_per_worker=2, max_workers=8,
                    continuation_limit_hours=1, full_cohort_expansion=False)
    manifest["manifest_hash"] = science.digest(manifest)
    original_cli.write_json(out / "manifest.json", manifest)
    print(json.dumps(dict(manifest_hash=manifest["manifest_hash"], inherited_ok_rows=len(good), new_or_corrected_rows=len(to_complete))))


def verify(out):
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    checked = dict(manifest)
    expected = checked.pop("manifest_hash")
    if science.digest(checked) != expected:
        raise ValueError("parser recovery manifest changed")
    for name, expected in manifest["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("amended parser execution source changed")
    inverse = ROOT / manifest["inverse_directory"]
    panel, protocol = original_cli.read_panel(inverse)
    if sha(inverse / "protocol.json") != manifest["original_protocol_file_sha256"] or protocol["protocol_hash"] != manifest["original_scientific_hash"]:
        raise ValueError("original frozen scientific protocol changed")
    if protocol["panel_sha256"] != manifest["panel_sha256"]:
        raise ValueError("same frozen4449panel required")
    for name, expected in manifest["inherited_ok_row_sha256"].items():
        if sha(inverse / "score_rows" / name) != expected:
            raise ValueError("inherited valid score bytes changed")
    return panel, manifest, inverse


def parse_check(out, volume_root):
    from pepstab.inverse_folding import load_complex
    panel, manifest, _ = verify(out)
    proof = json.loads((ROOT / "reports/followup3/inverse_parser_recovery/FAILED_STRUCTURE_INSPECTION.json").read_text())
    parser_fix.install_exact_selector()
    records = []
    for allele in ("HLA-B*56:01", "HLA-B*57:02"):
        row = next(r for r in manifest["corrected_and_missing_rows"] if r["allele"] == allele and r["peptide"] == "YPAEITLTW")
        folder = Path(volume_root) / row["folder"]
        cifs = list(folder.glob("*.cif"))
        if len(cifs) != 1:
            raise ValueError("one actual structure required in parser preflight")
        if allele == "HLA-B*56:01" and (sha(cifs[0]) != proof["cif_sha256"] or sha(folder / "metadata.json") != proof["metadata_sha256"]):
            raise ValueError("demonstrated failed structure/source changed")
        complex_ = load_complex(folder)
        if complex_.peptide != "YPAEITLTW" or len(complex_.context_chains) != 2:
            raise ValueError("problem-case parser identity failed")
        records.append(dict(allele=allele, peptide=complex_.peptide, native_peptide_chain=complex_.peptide_chain,
                            context_chains=complex_.context_chains, metadata_sha256=sha(folder / "metadata.json"), cif_sha256=sha(cifs[0])))
    original_cli.write_json(Path(out) / "parse_check.json", dict(manifest_hash=manifest["manifest_hash"], cases=records,
                    metadata_and_backbone_checks_passed=True, model_forward_executed=False))


def extract(out, volume_root, workers):
    out = Path(out)
    _, manifest, inverse = verify(out)
    if not 1 <= workers <= 8:
        raise ValueError("one to eight amended spawn workers permitted")
    parse_check(out, volume_root)
    original_cli.write_json(out / "corrected_attempt_started.json", dict(manifest_hash=manifest["manifest_hash"], workers=workers))
    failed_path = inverse / "score_rows" / manifest["original_failed_filename"]
    archive = out / "original_failed_attempt.json"
    if archive.exists() or not failed_path.exists() or sha(failed_path) != manifest["original_failed_sha256"]:
        raise ValueError("failed-row archival preconditions differ;no blind retry")
    # A same-volume atomic move preserves every failed byte, under an explicit
    # hash precondition. No valid score record is moved or overwritten.
    failed_path.rename(archive)
    if sha(archive) != manifest["original_failed_sha256"]:
        raise ValueError("archived failure content differs")
    original_cli.run_extraction_tasks = parser_fix.amended_run_tasks
    original_cli.extract(inverse, volume_root, workers, pilot=False)
    verify(out)
    completed = {p.name:sha(p) for p in (inverse / "score_rows").glob("*.json")}
    if len(completed) != 4449:
        raise ValueError("complete4449score record set required")
    original_cli.write_json(inverse / "parser_recovery_receipt.json", dict(schema="inverse_parser_recovery_receipt.1", status="passed",
                    recovery_manifest_sha256=sha(out / "manifest.json"), manifest_hash=manifest["manifest_hash"],
                    wrapper_source_sha256=manifest["source_sha256"], original_protocol_file_sha256=manifest["original_protocol_file_sha256"],
                    inherited_ok_row_sha256=manifest["inherited_ok_row_sha256"], original_failed_attempt_sha256=sha(archive),
                    corrected_and_missing_rows=manifest["corrected_and_missing_rows"], completed_row_sha256=completed,
                    aggregate_scores_sha256=sha(inverse / "scores.csv"), extraction_receipt_sha256=sha(inverse / "extraction_receipt.json"),
                    original_scientific_sources_unchanged=True, completed_valid_scores_recomputed=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "parse-check", "extract"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--inverse-out", type=Path, default=INVERSE)
    parser.add_argument("--volume-root", type=Path, default=Path("/structures"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "prepare": prepare(args.out, args.inverse_out)
    elif args.command == "parse-check": parse_check(args.out, args.volume_root)
    else: extract(args.out, args.volume_root, args.workers)


if __name__ == "__main__": main()
