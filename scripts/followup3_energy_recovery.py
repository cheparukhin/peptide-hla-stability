"""One explicitly admitted recovery of preempted energy cases; old bytes intact."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.followup3_energy import sha, write_once
from scripts.followup3_energy_v2 import verify as verify_parent
from pepstab import followup3_energy as core

PARENT = ROOT / "reports/followup3/energy_execution_v2"
OUT = ROOT / "reports/followup3/energy_recovery_v3"
SOURCES = ("scripts/followup3_energy_recovery.py", "reports/followup3/energy_recovery_v3/EXECUTION_AMENDMENT.md",
           "reports/followup3/energy_recovery_diagnosis/diagnosis.json")


def inventory(parent):
    panel, protocol = verify_parent(parent)
    directory = Path(parent) / "energy_scoring" / protocol["protocol_hash"]
    if (directory / "STOP_ACCESS.json").exists(): raise PermissionError("original access stop remains binding")
    dispatch = json.loads((Path(parent) / "function_dispatch_receipt.json").read_text())
    if dispatch["protocol_hash"] != protocol["protocol_hash"]: raise ValueError("dispatch protocol differs")
    statuses = {(r["allele"], r["peptide"]):r["status"] for r in dispatch["statuses"]}
    if len(statuses) != 4449 or list(statuses.values()).count("ok") != 4376 or list(statuses.values()).count("interrupted_attempt") != 73:
        raise ValueError("exact4376 successes and73 interrupted attempts required")
    successful, originals, remaining, rows = {}, {}, [], []
    for identity in panel.to_dict(orient="records"):
        key = core.digest([identity["allele"], identity["peptide"]]); path = directory / (key + ".json")
        if statuses[(identity["allele"], identity["peptide"])] == "ok":
            row = json.loads(path.read_text())
            if row["protocol_hash"] != protocol["protocol_hash"] or row.get("repair_interface") != "ALL": raise ValueError("successful row protocol differs")
            rows.append(row); successful[path.name] = sha(path)
        else:
            start = directory / (key + ".started.json")
            original = json.loads(start.read_text())
            if path.exists() or original["protocol_hash"] != protocol["protocol_hash"] or any(str(original[k]) != str(v) for k,v in identity.items()):
                raise ValueError("only exact interrupted missing outputs are recoverable")
            originals[start.name] = sha(start); remaining.append(identity)
    keys = {(r["allele"],r["peptide"]) for r in rows}
    core.validate_energy_rows(panel.loc[[k in keys for k in panel[["allele","peptide"]].itertuples(index=False,name=None)]], rows, protocol["specification"]["energy_columns"])
    return panel, protocol, directory, successful, originals, remaining


def prepare(out=OUT, parent=PARENT):
    out, parent = Path(out), Path(parent)
    panel, protocol, directory, successful, originals, remaining = inventory(parent)
    diagnosis = json.loads((ROOT / SOURCES[2]).read_text())
    if diagnosis["explicit_preemption_events"] != 73 or diagnosis["oom_evidence_in_bounded_system_fetch"] is not False:
        raise ValueError("reviewed label-free preemption diagnosis required")
    log = ROOT / diagnosis["log_path"]
    if sha(log) != diagnosis["log_sha256"]: raise ValueError("diagnostic evidence changed")
    native = {}
    for row in remaining:
        name = core.digest([row["allele"],row["peptide"]]) + ".json"
        path = ROOT / protocol["inverse_directory"] / "score_rows" / name
        score = json.loads(path.read_text())
        if score["status"] != "ok" or any(str(score[k]) != str(v) for k,v in row.items()): raise ValueError("audited inverse native identity differs")
        native[name] = dict(structure_sha256=score["structure_sha256"], metadata_sha256=score["metadata_sha256"],
            inverse_score_path=str(path.relative_to(ROOT)), inverse_score_sha256=sha(path))
    bound = dict(schema="energy_manual_preemption_recovery.1", status="FROZEN_OPERATIONAL_SCOPE_NOT_PAID_LAUNCH_APPROVAL",
        parent_directory=str(parent.relative_to(ROOT)), parent_protocol_file_sha256=sha(parent / "protocol.json"),
        parent_protocol_hash=protocol["protocol_hash"], dispatch_sha256=sha(parent / "function_dispatch_receipt.json"),
        original_rows_directory=str(directory.relative_to(ROOT)), original_success_row_sha256=successful,
        original_interrupted_start_sha256=originals, recovery_rows=remaining, successful_rows=4376, recoverable_rows=73,
        expected_native_inputs=native,
        source_sha256={name:sha(ROOT / name) for name in SOURCES}, log_sha256=sha(log),
        created_at_utc=datetime.now(timezone.utc).isoformat(), recovery_attempt=2, automatic_retries=0,
        numerical_threads=1, max_spawn_workers=16, sandbox_cpu_hard=16, sandbox_memory_hard_mib=32768,
        per_case_timeout_seconds=600, global_limit_seconds=2700, resource_isolation="aggregate Sandbox cap; no per-process2GiB cgroup claim",
        scientific_recipe_unchanged=True, completed_successes_recomputed=0)
    bound["manifest_hash"] = core.digest(bound)
    write_once(out / "manifest.json", bound)
    print(json.dumps(dict(manifest_hash=bound["manifest_hash"], inherited_ok=4376, recovery=73)))


def verify(out):
    out = Path(out); manifest = json.loads((out / "manifest.json").read_text())
    checked = dict(manifest); expected = checked.pop("manifest_hash")
    if core.digest(checked) != expected: raise ValueError("recovery manifest changed")
    for name, expected in manifest["source_sha256"].items():
        if sha(ROOT / name) != expected: raise ValueError("recovery source changed")
    diagnosis = json.loads((ROOT / SOURCES[2]).read_text())
    if sha(ROOT / diagnosis["log_path"]) != manifest["log_sha256"]: raise ValueError("original diagnostic logs changed")
    parent = ROOT / manifest["parent_directory"]
    panel, protocol = verify_parent(parent)
    if sha(parent / "protocol.json") != manifest["parent_protocol_file_sha256"] or protocol["protocol_hash"] != manifest["parent_protocol_hash"] or sha(parent / "function_dispatch_receipt.json") != manifest["dispatch_sha256"]:
        raise ValueError("original scientific/dispatch parent changed")
    old = ROOT / manifest["original_rows_directory"]
    if (old / "STOP_ACCESS.json").exists(): raise PermissionError("original access rejection blocks recovery")
    for name, expected in {**manifest["original_success_row_sha256"], **manifest["original_interrupted_start_sha256"]}.items():
        if sha(old / name) != expected: raise ValueError("original successful/start bytes changed")
    for identity in manifest["recovery_rows"]:
        if (old / (core.digest([identity["allele"],identity["peptide"]]) + ".json")).exists(): raise ValueError("interrupted output appeared after freeze; do not duplicate")
    for expected in manifest["expected_native_inputs"].values():
        if sha(ROOT / expected["inverse_score_path"]) != expected["inverse_score_sha256"]: raise ValueError("expected native inverse provenance changed")
    return panel, protocol, manifest, old


def verify_native(folder, expected):
    folder = Path(folder); cifs = list(folder.glob("*.cif"))
    if len(cifs) != 1 or sha(cifs[0]) != expected["structure_sha256"] or sha(folder / "metadata.json") != expected["metadata_sha256"]:
        raise ValueError("mounted native coordinate/metadata bytes differ")


def recover_one(arguments):
    identity, out, volume_root, protocol, manifest_hash, deadline, expected_native = arguments
    out = Path(out); key = core.digest([identity["allele"],identity["peptide"]])
    directory = out / "retry_rows"; directory.mkdir(exist_ok=True)
    path, start, access = directory / (key + ".json"), directory / (key + ".started.json"), out / "STOP_ACCESS.json"
    if path.exists():
        row = json.loads(path.read_text())
        if row.get("manual_recovery",{}).get("manifest_hash") != manifest_hash or row["protocol_hash"] != protocol["protocol_hash"] or any(str(row[k]) != str(v) for k,v in identity.items()): raise ValueError("recovered cache identity differs")
        return row
    if start.exists(): raise ValueError("manual recovery already started; no second retry")
    if core.utc_deadline(deadline) <= datetime.now(timezone.utc).timestamp(): raise TimeoutError("recovery global deadline reached")
    def check_access():
        if access.exists(): raise PermissionError("shared recovery access stop")
    check_access()
    verify_native(Path(volume_root) / identity["folder"], expected_native)
    write_once(start, dict(identity, manifest_hash=manifest_hash, attempt=2, deadline_utc=deadline))
    os.sync()
    from pepstab.foldx import locate_foldx
    from threadpoolctl import threadpool_limits
    distribution = locate_foldx(ROOT / "foldx5_Linux_0")
    if distribution["sha256"] != core.BINARY_SHA256: raise ValueError("same binary required")
    with threadpool_limits(limits=1):
        row = core.score_repaired(identity, Path(volume_root) / identity["folder"], distribution, deadline,
            protocol["specification"]["energy_columns"], access_stop_check=check_access)
    if row["status"] == "access_rejected":
        try: write_once(access, dict(status="access_rejected", manifest_hash=manifest_hash))
        except FileExistsError: pass
    row["protocol_hash"] = protocol["protocol_hash"]
    if row["status"] == "ok" and any(row[k] != expected_native[k] for k in ("structure_sha256","metadata_sha256")):
        raise ValueError("scored native bytes differ from frozen inverse provenance")
    row["manual_recovery"] = dict(manifest_hash=manifest_hash, attempt=2, numerical_threads=1)
    write_once(path, row); os.sync()
    return row


def run(out, volume_root, workers, deadline):
    out = Path(out); _, protocol, manifest, _ = verify(out)
    remaining = core.utc_deadline(deadline) - datetime.now(timezone.utc).timestamp()
    if not 1 <= workers <= 16 or not 0 < remaining <= 2700: raise ValueError("sixteen workers and45min maximum fixed UTC deadline")
    if sys.platform != "linux": raise ValueError("real recovery is cloud only")
    for variable in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        if os.environ.get(variable) != "1": raise ValueError("one numerical thread must be set before interpreter startup")
    write_once(out / "recovery_execution_started.json", dict(manifest_hash=manifest["manifest_hash"], deadline_utc=deadline, workers=workers))
    arguments = [(row,str(out),str(volume_root),protocol,manifest["manifest_hash"],deadline,
        manifest["expected_native_inputs"][core.digest([row['allele'],row['peptide']])+'.json']) for row in manifest["recovery_rows"]]
    if workers == 1: results = [recover_one(args) for args in arguments]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(recover_one, arguments))
    verify(out)
    write_once(out / "recovery_dispatch_receipt.json", dict(manifest_hash=manifest["manifest_hash"], attempted=73, statuses=[dict(allele=r['allele'],peptide=r['peptide'],status=r['status']) for r in results]))
    combine(out)


def combine(out):
    out = Path(out); panel, protocol, manifest, old = verify(out)
    execution = out / "recovery_execution_started.json"
    settings = json.loads(execution.read_text())
    if settings["manifest_hash"] != manifest["manifest_hash"] or not 1 <= settings["workers"] <= 16:
        raise ValueError("manual execution parent/worker scope differs")
    core.utc_deadline(settings["deadline_utc"])
    starts = {}
    pairs = [(old / name, expected) for name,expected in manifest["original_success_row_sha256"].items()]
    for row in manifest["recovery_rows"]:
        path = out / "retry_rows" / (core.digest([row["allele"],row["peptide"]]) + ".json")
        start = path.with_name(path.stem + ".started.json")
        attempted = json.loads(start.read_text())
        if attempted["manifest_hash"] != manifest["manifest_hash"] or attempted["attempt"] != 2 or attempted["deadline_utc"] != settings["deadline_utc"] or any(str(attempted[k]) != str(v) for k,v in row.items()):
            raise ValueError("recovery start identity/attempt/deadline differs")
        starts[start.name] = sha(start)
        recovered = json.loads(path.read_text())
        if recovered.get("manual_recovery",{}).get("manifest_hash") != manifest["manifest_hash"] or recovered["protocol_hash"] != protocol["protocol_hash"]: raise ValueError("recovery row parent differs")
        expected_native = manifest["expected_native_inputs"][path.name]
        if any(recovered.get(k) != expected_native[k] for k in ("structure_sha256","metadata_sha256")): raise ValueError("recovery row native source differs")
        pairs.append((path,sha(path)))
    core.validate_energy_rows(panel,[json.loads(path.read_text()) for path,_ in pairs],protocol["specification"]["energy_columns"])
    combined = out / "combined_rows"; combined.mkdir(exist_ok=True)
    for path, expected in pairs:
        target = combined / path.name
        if target.exists():
            if sha(target) != expected: raise ValueError("combined artifact changed")
        else:
            with target.open("xb") as handle: handle.write(path.read_bytes())
        if sha(target) != expected: raise ValueError("combined copy bytes differ")
    if len(list(combined.glob("*.json"))) != 4449: raise ValueError("only complete original4449 panel may be collected")
    verify(out)
    write_once(out / "recovery_receipt.json", dict(status="passed", manifest_hash=manifest["manifest_hash"], manifest_file_sha256=sha(out / "manifest.json"),
        parent_protocol_hash=protocol["protocol_hash"], preserved_ok_rows=4376, preserved_original_interrupted_starts=73,
        manual_recovered_rows=73, completed_successes_recomputed=0, scientific_recipe_unchanged=True,
        recovery_execution_started_sha256=sha(execution), retry_start_sha256=starts,
        recovery_dispatch_sha256=sha(out / "recovery_dispatch_receipt.json") if (out / "recovery_dispatch_receipt.json").exists() else None,
        original_success_row_sha256=manifest["original_success_row_sha256"], original_interrupted_start_sha256=manifest["original_interrupted_start_sha256"],
        recovery_row_sha256={path.name:expected for path,expected in pairs if path.parent == out / "retry_rows"},
        combined_row_sha256={path.name:expected for path,expected in pairs}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare","run","combine")); parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--parent-out",type=Path,default=PARENT); parser.add_argument("--volume-root",type=Path,default=Path("/structures"))
    parser.add_argument("--workers",type=int,default=16); parser.add_argument("--deadline-utc")
    args=parser.parse_args()
    if args.command == "prepare": prepare(args.out,args.parent_out)
    elif args.command == "run": run(args.out,args.volume_root,args.workers,args.deadline_utc)
    else: combine(args.out)


if __name__ == "__main__": main()
