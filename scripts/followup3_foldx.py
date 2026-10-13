"""Frozen32-case FoldX technical pilot; no assay outcomes or predictive fitting."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

OUT = ROOT / "reports/followup3/foldx"
BINARY_SHA256 = "faed5e54e47744ab6ab75f8f9ad1a1f2e2cdbeac4f0f98d4e36c97049eb14268"
SEED = "followup3-foldx-technical-20261013"
TIMEOUT_S = 1200


def sha(path):
    from pepstab.boltz_embeddings import sha256_file
    return sha256_file(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise ValueError("refusing changed technical artifact")
        return
    with path.open("x") as handle:
        handle.write(text)


def select_cases(panel, splits):
    keys = ["allele", "peptide"]
    if panel.duplicated(keys).any() or splits.duplicated(keys).any():
        raise ValueError("unique natural keys required")
    joined = panel.merge(splits[keys + ["split"]], on=keys, how="left", validate="one_to_one")
    if not joined.split.eq("train").all():
        raise ValueError("only ORIGINAL-training inverse-panel cases permitted")
    candidates = []
    for allele, rows in joined.groupby("allele", sort=True):
        chosen = min(rows.to_dict(orient="records"), key=lambda row: digest([SEED, row["allele"], row["peptide"]]))
        candidates.append(chosen)
    if len(candidates) < 32:
        raise ValueError("32 distinct alleles required")
    selected = sorted(candidates, key=lambda row: digest([SEED, row["allele"]]))[:32]
    return pd.DataFrame(selected).drop(columns="split").reset_index(drop=True)


def prepare(out=OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    inverse = ROOT / "reports/followup3/inverse"
    inverse_protocol = json.loads((inverse / "protocol.json").read_text())
    if sha(inverse / "panel.csv") != inverse_protocol["panel_sha256"]:
        raise ValueError("inverse frozen panel changed")
    panel = select_cases(pd.read_csv(inverse / "panel.csv"), pd.read_csv(ROOT / "data/splits.csv"))
    path = out / "panel.csv"
    text = panel.to_csv(index=False)
    if path.exists() and path.read_text() != text:
        raise ValueError("refusing changed FoldX technical panel")
    path.write_text(text)
    sources = ("scripts/followup3_foldx.py", "pepstab/foldx.py", "pepstab/structural_features.py", "scripts/boltz_pose_check.py",
               "reports/followup3/foldx/SPECIFICATION.md")
    specification = dict(schema="followup3-foldx-technical.1", technical_only=True, n_cases=32, distinct_alleles=32,
                         selection_seed=SEED, sample="one hash-first frozen inverse-panel pair per allele;32hash-firstalleles",
                         binary_sha256=BINARY_SHA256, repair_interface="ALL", interaction="native peptide versus verified HLA chain;beta2m retained",
                         cases_timeout_seconds=TIMEOUT_S, max_workers=8, attempts_per_case=1, assay_labels_read=False,
                         contrasts="paired raw and repaired energies/runtime/coordinates;no biological-efficacy endpoint",
                         access_handling="explicit license/access rejection stops further commands;no bypass")
    protocol = dict(status="FROZEN_TECHNICAL_SCOPE_NOT_PAID_LAUNCH_APPROVAL", specification=specification,
                    specification_sha256=digest(specification), panel_sha256=sha(path),
                    source_sha256={p:sha(ROOT / p) for p in sources},
                    input_sha256={p:sha(ROOT / p) for p in ("data/splits.csv", "reports/followup3/inverse/panel.csv", "reports/followup3/inverse/protocol.json")})
    protocol["protocol_hash"] = digest(protocol)
    write_once(out / "protocol.json", protocol)
    print(json.dumps(dict(cases=len(panel), alleles=int(panel.allele.nunique()), protocol_hash=protocol["protocol_hash"])))


def verify(out):
    out = Path(out)
    protocol = json.loads((out / "protocol.json").read_text())
    observed = dict(protocol)
    expected = observed.pop("protocol_hash")
    if digest(observed) != expected or digest(protocol["specification"]) != protocol["specification_sha256"]:
        raise ValueError("technical protocol changed")
    if sha(out / "panel.csv") != protocol["panel_sha256"]:
        raise ValueError("technical panel changed")
    for category in ("source_sha256", "input_sha256"):
        for name, expected in protocol[category].items():
            if sha(ROOT / name) != expected:
                raise ValueError(f"frozen technical input changed:{name}")
    return pd.read_csv(out / "panel.csv"), protocol


def access_rejection(stdout, stderr, returncode):
    """Classify denial without returning license-bearing program text."""
    import re
    text = (stdout + "\n" + stderr).lower()
    denial = r"(?:licen[sc]e.{0,80}(?:expired|invalid|denied|not found|failed|error)|(?:expired|invalid|denied).{0,80}licen[sc]e|permission denied|exec format error|cannot execute|cannot open.{0,80}(?:molecules|rotabase))"
    return returncode in (126, 127) or bool(re.search(denial, text))


def run_command(command, work, deadline, access_stop):
    if access_stop.is_set():
        raise PermissionError("technical access gate already failed")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("paired case deadline reached")
    started = time.monotonic()
    result = subprocess.run(command, cwd=work, capture_output=True, text=True, timeout=remaining)
    if access_rejection(result.stdout, result.stderr, result.returncode):
        access_stop.set()
        raise PermissionError("FoldX license/access rejection;no program text retained")
    if result.returncode:
        raise RuntimeError(f"FoldX command returned nonzero status:{result.returncode}")
    return dict(command=[Path(command[0]).name] + command[1:], returncode=result.returncode,
                elapsed_s=time.monotonic() - started)


def pdb_identity(path):
    from biotite.structure.io.pdb import PDBFile
    from pepstab.foldx import _one_letter
    atoms = PDBFile.read(str(path)).get_structure(model=1)
    atoms = atoms[~np.isin(atoms.element, ("H", "D"))]
    if not np.isfinite(atoms.coord).all() or atoms.array_length() == 0:
        raise ValueError("nonfinite or empty PDB coordinates")
    chains = {str(cid): _one_letter(atoms[atoms.chain_id == cid]) for cid in np.unique(atoms.chain_id)}
    if any("X" in seq for seq in chains.values()):
        raise ValueError("nonstandard repaired/native residue")
    return atoms, chains


def coordinate_change(raw, repaired):
    def index(atoms):
        return {(str(c), int(r), str(a)): xyz for c, r, a, xyz in zip(atoms.chain_id, atoms.res_id, atoms.atom_name, atoms.coord)}
    first, second = index(raw), index(repaired)
    common = sorted(set(first) & set(second))
    if not common:
        raise ValueError("no atoms shared after repair")
    squared = np.asarray([np.sum((first[key] - second[key]) ** 2) for key in common])
    backbone = np.asarray([key[2] in ("N", "CA", "C", "O") for key in common])
    return dict(n_matching_heavy_atoms=len(common), atoms_added=len(set(second) - set(first)), atoms_removed=len(set(first) - set(second)),
                all_atom_rms_displacement_A=float(np.sqrt(np.mean(squared))),
                backbone_rms_displacement_A=float(np.sqrt(np.mean(squared[backbone]))))


def validated_interaction(rows, groups):
    """Reject malformed/nonfinite terms BEFORE the shared numeric reduction."""
    from pepstab.foldx import interaction_features
    desired = {str(group).strip().upper() for group in groups}
    selected = [row for row in rows if {str(row.get("group1", "")).strip().upper(),
                                       str(row.get("group2", "")).strip().upper()} == desired]
    if len(selected) != 1:
        raise ValueError("exactly one selected interaction row required")
    for field, value in selected[0].items():
        if field not in ("pdb", "group1", "group2"):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
                raise ValueError("invalid/nonfinite numeric interaction term")
    return interaction_features(rows, *groups)


def score_case(identity, volume_root, distribution, out, access_stop):
    from pepstab import foldx as fx
    key = digest([identity["allele"], identity["peptide"]])
    record_path = Path(out) / "cases" / f"{key}.json"
    if record_path.exists():
        raise ValueError("case already attempted;automatic retry/reuse is not authorized")
    work = Path(out) / "work" / key
    work.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + TIMEOUT_S
    record = dict(identity, status="started", binary_sha256=distribution["sha256"], phases=[], stage="metadata")
    try:
        folder = Path(volume_root) / identity["folder"]
        meta = json.loads((folder / "metadata.json").read_text())
        case = meta["inputs"]
        seed = meta.get("seed", meta.get("settings", {}).get("seed"))
        if (case["allele"], case["peptide"], case["arm"], meta["model"], seed, case.get("split", meta.get("split"))) != (identity["allele"], identity["peptide"], "B", "boltz2", 0, "train"):
            raise ValueError("production/native/construct training identity mismatch")
        cif = list(folder.glob("*.cif"))
        if len(cif) != 1:
            raise ValueError("exactly one source structure required")
        record.update(metadata_sha256=sha(folder / "metadata.json"), structure_sha256=sha(cif[0]))
        for field in ("molecules", "rotabase"):
            if distribution.get(field):
                (work / Path(distribution[field]).name).symlink_to(Path(distribution[field]))
        record["stage"] = "conversion/native_chains"
        prepared = fx.prepare_pdb(folder, work / "input")
        raw = Path(prepared["pdb"])
        raw_atoms, chains = pdb_identity(raw)
        if chains != {str(chain["id"]):chain["sequence"] for chain in case["chains"]}:
            raise ValueError("converted sequence/chain identity differs from metadata")
        record.update(conversion=prepared["conversion"], conversion_validation=prepared["validation"], chain_sequences=chains,
                      peptide_chain=prepared["peptide_chain"], hla_chain=prepared["hla_chain"], beta2m_chain=prepared["beta2m_chain"], raw_pdb_sha256=sha(raw))
        groups = (prepared["peptide_chain"], prepared["hla_chain"])
        for repaired in (False, True):
            directory = work / ("repaired" if repaired else "raw")
            directory.mkdir()
            pdb = raw
            if repaired:
                record["stage"] = "repair"
                command = fx.repair_command(distribution["binary"], raw.name, str(raw.parent), str(directory), repair_interface="ALL")
                record["phases"].append(dict(phase="repair", **run_command(command, str(work), deadline, access_stop)))
                pdb = directory / f"{raw.stem}_Repair.pdb"
                if not pdb.exists():
                    raise RuntimeError("RepairPDB did not produce expected repaired file")
                record["stage"] = "repaired_coordinates/native_chains"
                repaired_atoms, after = pdb_identity(pdb)
                if after != chains:
                    raise ValueError("repair changed native sequences/chain identities")
                record["coordinate_change"] = coordinate_change(raw_atoms, repaired_atoms)
                record["repaired_pdb_sha256"] = sha(pdb)
            record["stage"] = "repaired_analysis" if repaired else "raw_analysis"
            command = fx.analyse_complex_command(distribution["binary"], pdb.name, str(pdb.parent), str(directory), groups)
            record["phases"].append(dict(phase="repaired_analysis" if repaired else "raw_analysis", **run_command(command, str(work), deadline, access_stop)))
            outputs = list(directory.glob("Interaction_*.fxout"))
            if len(outputs) != 1:
                raise RuntimeError("exactly one interaction output required")
            record["stage"] = "repaired_parse" if repaired else "raw_parse"
            parsed = validated_interaction(fx.parse_fxout(outputs[0].read_text()), groups)
            numbers = {k:v for k, v in parsed.items() if isinstance(v, float)}
            if not isinstance(parsed.get("foldx_interaction_energy"), float) or not np.isfinite(list(numbers.values())).all():
                raise ValueError("nonfinite or missing interaction energy terms")
            record["repaired" if repaired else "raw"] = dict(terms=parsed, fxout_sha256=sha(outputs[0]), pdb_sha256=sha(pdb))
        raw_terms, repaired_terms = record["raw"]["terms"], record["repaired"]["terms"]
        record["term_changes"] = {k:repaired_terms[k] - value for k, value in raw_terms.items() if isinstance(value, float) and isinstance(repaired_terms.get(k), float)}
        record["status"] = "ok"
    except Exception as error:
        # Never persist proprietary executable stdout/stderr or arbitrary parser
        # excerpts, which could contain access/license information.
        if isinstance(error, PermissionError):
            access_stop.set()
        record.update(status="access_rejected" if isinstance(error, PermissionError) else "failed",
                      error_type=type(error).__name__, error="technical case failed;no executable output text retained")
    record["elapsed_s"] = time.monotonic() - started
    write_once(record_path, record)
    shutil.rmtree(work)
    return record


def run(out, volume_root, foldx_root, workers):
    from pepstab.foldx import locate_foldx
    out = Path(out).resolve()
    if not 1 <= workers <= 8:
        raise ValueError("one to eight single-threaded FoldX workers permitted")
    panel, protocol = verify(out)
    distribution = locate_foldx(foldx_root)
    if distribution["sha256"] != BINARY_SHA256:
        raise ValueError("installed binary hash differs from frozen build")
    if list((Path(out) / "cases").glob("*.json")):
        raise ValueError("technical pilot already attempted;no automatic relaunch")
    write_once(Path(out) / "attempt.json", dict(protocol_hash=protocol["protocol_hash"], workers=workers, cases=32,
               per_case_timeout_s=TIMEOUT_S, binary_sha256=BINARY_SHA256, license_verified=False))
    started = time.monotonic()
    rows = panel.to_dict(orient="records")
    access_stop = threading.Event()
    results = [score_case(rows[0], volume_root, distribution, out, access_stop)]
    # First paired case gates executable access and output parsing before fanout.
    if results[0]["status"] == "ok":
        pending_rows = iter(rows[1:])
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(score_case, row, volume_root, distribution, out, access_stop) for row in [next(pending_rows, None) for _ in range(min(workers, len(rows) - 1))] if row is not None}
            while futures:
                completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    results.append(future.result())
                if not access_stop.is_set():
                    for _ in completed:
                        row = next(pending_rows, None)
                        if row is not None:
                            futures.add(pool.submit(score_case, row, volume_root, distribution, out, access_stop))
    status = "passed" if len(results) == 32 and all(r["status"] == "ok" for r in results) else "failed"
    summary = dict(status=status, protocol_hash=protocol["protocol_hash"], attempted=len(results),
                   successful=sum(r["status"] == "ok" for r in results), failures=sum(r["status"] != "ok" for r in results),
                   unattempted=32 - len(results), wall_s=time.monotonic() - started, sum_case_s=sum(r["elapsed_s"] for r in results),
                   access_rejected=access_stop.is_set(), license_verified=False,
                   case_artifact_sha256={p.name:sha(p) for p in (Path(out) / "cases").glob("*.json")})
    write_once(Path(out) / "technical_receipt.json", summary)
    flat = []
    for result in results:
        row = {k:result[k] for k in ("allele", "peptide", "status", "elapsed_s")}
        for condition in ("raw", "repaired"):
            row.update({f"{condition}_{k}":v for k, v in result.get(condition, {}).get("terms", {}).items()})
        row.update({f"change_{k}":v for k, v in result.get("term_changes", {}).items()})
        flat.append(row)
    pd.DataFrame(flat).to_csv(Path(out) / "paired_terms.csv", index=False)
    print(json.dumps(summary))
    if status != "passed":
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--volume-root", type=Path, default=Path("/structures"))
    parser.add_argument("--foldx-root", type=Path, default=Path("/opt/foldx"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command == "prepare": prepare(args.out)
    else: run(args.out, args.volume_root, args.foldx_root, args.workers)


if __name__ == "__main__":
    main()
