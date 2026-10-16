"""Prepare, freeze after main completion, and run LOCAL bounded transfer cells.

Draft creation reads training labels/counts and frozen grouping, never main
performance outcomes. Freeze and fitting refuse incomplete main-screen markers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pepstab import evaluation as ev
from pepstab.boltz_embeddings import sha256_file
from pepstab.data import load_with_splits
from pepstab.round2_nested import config_digest
from pepstab.round2_transfer import (ARMS, CANDIDATES, TransferAncestry, TransferPredictor,
    assess_cell, fit_graph, inner_partitions, outer_cells, validate_frame,
    candidate_registry, reported_arms, procedure_order, validate_candidates, SEQUENCE_SELECTOR, EMBEDDING_SELECTOR)
from scripts.round2_embedding_screen import atomic_csv, atomic_json, load_compact_features

OUT = ROOT / "reports/further_research/round2/transfer"
CODE = ("pepstab/round2_transfer.py", "scripts/round2_transfer.py", "scripts/round2_transfer_report.py",
        "pepstab/mlp.py", "pepstab/features.py", "pepstab/evaluation.py", "pepstab/round2_reporting.py",
        "pepstab/round2_nested.py", "scripts/round2_embedding_screen.py", "pepstab/round2_representations.py",
        "pepstab/data.py", "pepstab/boltz_embeddings.py", "pepstab/phase_a.py", "pepstab/round2_embedding.py")


def training_frame(grouping=OUT):
    grouping = Path(grouping)
    protocol = json.loads((grouping / "protocol.json").read_text())
    for name, expected in protocol["sha256"].items():
        if Path(name).name != name or sha256_file(grouping/name) != expected:
            raise ValueError("frozen transfer grouping changed")
    train = load_with_splits().query("split == 'train'").sort_values("pair_id").reset_index(drop=True)
    manifest = pd.read_csv(grouping/"manifest.csv")
    if manifest.pair_id.duplicated().any() or set(manifest.pair_id) != set(train.pair_id):
        raise ValueError("manifest differs from canonical original training population")
    frame = train.merge(manifest, on=["pair_id", "cluster_id", "allele"], validate="one_to_one", how="left", sort=False)
    if frame[["hla_group", "hla_fold", "peptide_fold"]].isna().any().any():
        raise ValueError("manifest identity mismatch")
    validate_frame(frame)
    if frame.hla_group.nunique() != protocol["hla_groups"]:
        raise ValueError("group count differs from frozen manifest")
    return frame


def main_gate(main_root):
    """Content checks only: never parse outcome tables to determine completion."""
    main_root = Path(main_root)
    evidence = {}
    sequence = json.loads((main_root/"sequence/protocol.json").read_text())
    embedding = json.loads((main_root/"embedding/protocol.json").read_text())
    for kind, protocol in (("sequence", sequence), ("embedding", embedding)):
        evidence[f"{kind}/protocol.json"] = sha256_file(main_root/kind/"protocol.json")
        for fold in range(5):
            directory = main_root/kind/f"fold_{fold}"
            marker = directory/"completed.json"
            if not marker.exists():
                raise ValueError(f"main screen incomplete: {kind} fold {fold}")
            done = json.loads(marker.read_text())
            if done.get("protocol_hash") != protocol["protocol_hash"]:
                raise ValueError("main completion protocol mismatch")
            if sha256_file(directory/"predictions.csv") != done["prediction_sha256"]:
                raise ValueError("main completion prediction hash mismatch")
            if kind == "embedding":
                seq_done = json.loads((main_root/"sequence"/f"fold_{fold}"/"completed.json").read_text())
                if done.get("status") != "complete" or done.get("sequence_protocol_hash") != sequence["protocol_hash"] or done.get("sequence_prediction_sha256") != seq_done["prediction_sha256"]:
                    raise ValueError("embedding selector not finalized against sequence")
            evidence[str(marker.relative_to(main_root))] = sha256_file(marker)
    report_path = main_root/"results/main/manifest.json"
    if not report_path.exists():
        raise ValueError("main audited report incomplete")
    report = json.loads(report_path.read_text())
    if report.get("schema") != "round2_results.1" or report.get("scope") != "sequence_and_embedding" or not report.get("all_five_folds_completed"):
        raise ValueError("main audited report has incomplete scope")
    if report.get("protocol_hashes") != {"sequence":sequence["protocol_hash"],"embedding":embedding["protocol_hash"]}:
        raise ValueError("main audited report protocol mismatch")
    if not report.get("source_sha256") or not report.get("output_sha256"):
        raise ValueError("main audited report missing source/output hashes")
    for name, expected in report["source_sha256"].items():
        if sha256_file(Path(name)) != expected:
            raise ValueError("main audited report source changed")
    for name, expected in report["output_sha256"].items():
        if Path(name).name != name or sha256_file(report_path.parent/name) != expected:
            raise ValueError("main audited report output changed")
    evidence["results/main/manifest.json"] = sha256_file(report_path)
    return evidence


def validate_main_nominations(main_root, candidates):
    """Require exact frozen main grids; preparation never creates new settings."""
    embedding = json.loads((Path(main_root)/'embedding/protocol.json').read_text())['specification']
    for procedure in procedure_order(candidates)[2:]:
        nominated = [c for c in candidates if f"{c.get('representation')}_{c['learner']}" == procedure]
        if procedure not in embedding['selector_order'] or embedding['candidate_grids'].get(procedure) != nominated:
            raise ValueError('nomination grid differs from frozen main primary grid')


def specification(frame, grouping, nominations=()):
    candidates = candidate_registry(nominations)
    graph = fit_graph(candidates)
    grouping = Path(grouping)
    timing_path = ROOT/"reports/further_research/round2/sequence/profile_members.json"
    timing = [r for r in json.loads(timing_path.read_text()) if r["learner"] == "mlp"]
    timing_reference = {"source":str(timing_path.relative_to(ROOT)),"sha256":sha256_file(timing_path),
        "members":[{k:r[k] for k in ("encoding","seconds","epochs","fit_rows")} for r in timing],
        "mlp_hours_at_observed_profile":graph["sequence_mlp_fits"]/2*sum(r["seconds"] for r in timing)/3600,
        "mlp_hours_linear_300_epoch_extrapolation":graph["sequence_mlp_fits"]/2*sum(r["seconds"]*300/r["epochs"] for r in timing)/3600,
        "limitations":"extrapolations at profile gradient-row count; exclude nominated embedding heads, row-count changes, feature assembly and other overhead; not a validated runtime bound"}
    return {"schema": "round2_transfer.1", "canonical_plan": "reports/NEXT_RESEARCH_PLAN.md#6",
        "population": "all original training rows only; development evidence; original test untouched",
        "n_rows": len(frame), "n_hla_groups": int(frame.hla_group.nunique()),
        "frozen_pooled_panels": {str(n):ev.eligible_alleles(frame.allele,frame.y_log1p,min_rows=n) for n in (20,50)},
        "source_sha256": sha256_file(ROOT/"data/rasmussen_et_al_dataset.csv"),
        "splits_sha256": sha256_file(ROOT/"data/splits.csv"),
        "data_hash": hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest(),
        "grouping_hashes": {name: sha256_file(grouping/name) for name in ("protocol.json", "manifest.csv", "hla_groups.csv", "coverage.csv")},
        "outer_cells": "all five HLA cells and all 25 joint intersections; no outcome exclusions",
        "nominations": list(procedure_order(candidates)[2:]),
        "candidate_order": list(candidates), "reported_arms": list(reported_arms(candidates)),
        "selector_candidate_orders": {SEQUENCE_SELECTOR:["original","mlp_centred"], EMBEDDING_SELECTOR:list(procedure_order(candidates))},
        "scope_amendment": "bounded MLP sequence comparator and at most two main-nominated embedding procedures; excludes full five-family main sequence selector",
        "rationale": "no Boltz nominees chosen in preparation; freeze explicitly nominates one or two existing main primary representation/head procedures and retains their entire frozen regularization grids",
        "selection": "three balanced group folds; joint all 3x3 intersections; pooled inner median per-allele Spearman; max score with exact ties resolved in frozen order, original first",
        "selection_amendment": "pooled selection replaces mean cell medians/one-SE rule because joint cells can lack eligible per-cell panels and pooled endpoint is the target; inner pooled eligibility min20",
        "stopping": "five balanced HLA folds; joint five paired diagonal HLA/peptide cells; gradients outside BOTH, stop on intersection, cross quadrants excluded; sequence inner calls use 3x2x2=12; sequence finals use 5x2x3=30; neural feature inner calls use stop0/seed0 and stop1/seed1; neural feature finals use 5x6=30",
        "stopping_limitation": "joint diagonal stopping does not cover every fitting row; smaller gradient/stop sets differ from peptide regime and are logged per member",
        "original": {"encodings": ["onehot", "blosum"], "seeds": [0,1,2], "hidden": [256,64], "max_epochs":300, "patience":15, "l2":{"onehot":1e-5,"blosum":1e-3}},
        "ridge": {"solver":"lsqr", "tol":1e-5, "max_iter":2000, "scaling":"fit-only StandardScaler", "alphas":[10000.,100.,1.]},
        "centred": {"k":[100,20], "means":"each member gradient rows only; apply to gradient/stopping targets and restore predictions; unseen allele uses gradient global mean"},
        "neural_embedding": {"hidden":[64],"l2":[1.,.01,.0001],"max_epochs":200,"patience":15,"scaling":"gradient-only StandardScaler", "inner_members":2,"final_members":30},
        "fit_graph": graph, "timing_reference": timing_reference, "local_runtime_gate": "Unmeasured: before execution, validate conservative wall-time forecast from deterministic first HLA/joint timing telemetry; counts alone are not runtime approval", "execution": {"device":"local CPU", "threads":2, "cloud_spend_usd":0},
        "uncertainty": {"draws":2000, "modes":["paired peptide clusters", "paired HLA groups", "crossed independent peptide/HLA draws"],
            "group_multiplicity":"replicate allele contributions to median as well as row weights", "panels":[20,50],
            "sparse_group_sensitivity":"leave EACH actual HLA group out of the pooled endpoint; publish all groups and coverage",
            "interpretation":"conditional on fitted predictions; no refitting; crossed pigeonhole sensitivity can be conservative with sparse intersections; execution repeats separate"},
        "comparisons":"all shortlisted components and embedding selector versus original AND bounded_MLP_sequence_selector; bounded sequence selector versus original; both panels retained",
        "gate":"main sequence and embedding completion required before scope freeze or fitting; explicit documented scope amendment required"}


def preflight(frame):
    """Check all deeper split feasibility, without supervised fitting."""
    rows = []
    for regime, cell, fit, held in outer_cells(frame):
        inner = list(inner_partitions(fit, regime))
        stops = list(inner_partitions(fit, regime, 5, stopping=True))
        stops.extend(inner_partitions(fit, regime, 3, stopping=True))
        for _, train, _ in inner:
            for n_stop in (3,5):
                stops.extend(inner_partitions(train, regime, n_stop, stopping=True))
        rows.append({"regime":regime, "cell":cell, "fit_rows":len(fit), "held_rows":len(held),
            "inner_cells":len(inner), "stopping_partitions":len(stops),
            "minimum_inner_eligible_alleles":min(len(ev.eligible_alleles(t.allele,t.y_log1p,min_rows=20)) for _,t,_ in inner),
            "minimum_gradient_rows":min(len(t) for _,t,_ in stops),
            "minimum_stopping_rows":min(len(h) for _,_,h in stops)})
    return rows


def draft(out, grouping=OUT):
    out = Path(out)
    frame = training_frame(grouping)
    item = {"status":"DRAFT_NOT_FROZEN_NO_FITS", "created_at_utc":datetime.now(timezone.utc).isoformat(),
            "specification":specification(frame, grouping), "preflight":preflight(frame)}
    atomic_json(out/"assessment_protocol_draft.json", item)
    return item


def freeze(out, grouping, main_root, feature_manifest, amendment, nominations=()):
    if not amendment.strip():
        raise ValueError("explicit plan amendment reference required for reduced comparator scope")
    evidence = main_gate(main_root)
    if not nominations:
        raise ValueError("explicit one or two main-nominated Boltz procedures required after completion gate")
    candidates = candidate_registry(nominations)
    validate_main_nominations(main_root, candidates)
    frame = training_frame(grouping)
    preflight(frame)
    features, provenance = load_compact_features(frame, feature_manifest)
    spec = specification(frame, grouping, nominations)
    item = {"status":"FROZEN", "frozen_at_utc":datetime.now(timezone.utc).isoformat(),
        "specification":spec, "protocol_hash":config_digest(spec), "scope_amendment_reference":amendment,
        "main_completion_hashes":evidence, "feature_provenance":provenance,
        "feature_hashes":{k:hashlib.sha256(features[k].tobytes()).hexdigest() for k in sorted({c["representation"] for c in candidates if "representation" in c})},
        "code_sha256":{name:sha256_file(ROOT/name) for name in CODE}}
    path = Path(out)/"assessment_protocol.json"
    if path.exists():
        old = json.loads(path.read_text())
        if config_digest(old["specification"]) != old["protocol_hash"] or old["specification"] != spec:
            raise ValueError("saved frozen transfer specification changed")
        for key in ("protocol_hash", "main_completion_hashes", "feature_hashes", "code_sha256", "scope_amendment_reference"):
            if old[key] != item[key]:
                raise ValueError("refusing changed frozen transfer protocol")
        return old
    atomic_json(path, item)
    return item


def validate_frozen(out, grouping, main_root):
    protocol = json.loads((Path(out)/"assessment_protocol.json").read_text())
    if protocol.get("status") != "FROZEN":
        raise ValueError("transfer scope is not frozen")
    saved_specification = protocol["specification"]
    if config_digest(saved_specification) != protocol["protocol_hash"]:
        raise ValueError("saved frozen transfer specification hash mismatch")
    frame = training_frame(grouping)
    regenerated = specification(frame, grouping, saved_specification["nominations"])
    if saved_specification != regenerated or protocol["protocol_hash"] != config_digest(regenerated):
        raise ValueError("transfer data/specification changed")
    validate_main_nominations(main_root, protocol["specification"]["candidate_order"])
    if main_gate(main_root) != protocol["main_completion_hashes"]:
        raise ValueError("main completion evidence changed after freeze")
    for name, expected in protocol["code_sha256"].items():
        if sha256_file(ROOT/name) != expected:
            raise ValueError(f"frozen transfer code changed: {name}")
    return frame, protocol


def _run_cell(out, grouping, main_root, feature_manifest, regime, cell):
    out = Path(out)
    frame, protocol = validate_frozen(out, grouping, main_root)
    match = [(f,h) for r,c,f,h in outer_cells(frame) if (r,c) == (regime,cell)]
    if len(match) != 1:
        raise ValueError("unknown frozen regime/cell")
    fit, held = match[0]
    directory = out/regime/cell
    if (directory/"completed.json").exists():
        verify_cell(directory, protocol, regime, cell)
        return
    features, provenance = load_compact_features(frame, feature_manifest)
    candidates = validate_candidates(protocol["specification"]["candidate_order"])
    features = {k:features[k] for k in protocol["feature_hashes"]}
    if {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in features.items()} != protocol["feature_hashes"]:
        raise ValueError("features changed after transfer freeze")
    directory.mkdir(parents=True, exist_ok=True)
    ancestry = TransferAncestry(regime)
    def callback(member):
        with (directory/"progress.jsonl").open("a") as handle:
            handle.write(json.dumps(member)+"\n")
    predictor = TransferPredictor(frame, features, ancestry, candidates=candidates, callback=callback)
    with threadpool_limits(limits=2):
        atomic_json(directory/"runtime.json", {"platform":platform.platform(), "python":sys.version,
            "threadpools":threadpool_info(), "threads":2, "package_versions":{p:version(p) for p in ("numpy","pandas","scipy","scikit-learn","threadpoolctl")}, "feature_provenance":provenance,
            "protocol_hash":protocol["protocol_hash"], "regime":regime, "cell":cell})
        values, selection = assess_cell(fit, held, regime, predictor, ancestry, candidates=candidates)
    audit = ancestry.audit(frame, fit, held)
    expected = fit_graph(candidates)["regimes"][regime]
    if len(predictor.members) != expected["fits_per_cell"]:
        raise ValueError("actual fits differ from frozen graph")
    tables = []
    for arm, prediction in values.items():
        table = held[["pair_id","cluster_id","hla_group","allele","hla_fold","peptide_fold"]].copy()
        table["regime"], table["cell"], table["arm"], table["y_pred"] = regime, cell, arm, prediction
        tables.append(table)
    atomic_csv(directory/"predictions.csv", pd.concat(tables, ignore_index=True))
    atomic_json(directory/"selection.json", selection)
    atomic_json(directory/"ancestry.json", ancestry.as_dict())
    atomic_json(directory/"members.json", predictor.members)
    atomic_json(directory/"completed.json", {"status":"complete", "regime":regime, "cell":cell,
        "protocol_hash":protocol["protocol_hash"], "n_held":len(held), "n_fits":len(predictor.members), "audit":audit,
        "completed_at_utc":datetime.now(timezone.utc).isoformat(),
        "sha256":{name:sha256_file(directory/name) for name in ("predictions.csv","selection.json","ancestry.json","members.json","runtime.json")}})



def run_cell(out, grouping, main_root, feature_manifest, regime, cell):
    # Atomic lock prevents two local invocations from overwriting one cell.
    # A stale lock is evidence to inspect after interruption, never bypassed.
    if regime not in ("hla", "joint") or cell not in {f"h{h}"+("" if regime == "hla" else f"_p{p}") for h in range(5) for p in range(5)}:
        raise ValueError("unknown frozen regime/cell")
    directory = Path(out)/regime/cell
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory/"running.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        _run_cell(out, grouping, main_root, feature_manifest, regime, cell)
    finally:
        os.close(descriptor)
        lock.unlink()


def verify_cell(directory, protocol, regime, cell):
    directory = Path(directory)
    done = json.loads((directory/"completed.json").read_text())
    if done.get("status") != "complete" or (done.get("regime"),done.get("cell")) != (regime,cell) or done.get("protocol_hash") != protocol["protocol_hash"]:
        raise ValueError("invalid transfer cell completion marker")
    if set(done["sha256"]) != {"predictions.csv","selection.json","ancestry.json","members.json","runtime.json"}:
        raise ValueError("incomplete transfer cell hash coverage")
    for name, expected in done["sha256"].items():
        if sha256_file(directory/name) != expected:
            raise ValueError("transfer cell artifact hash mismatch")
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("draft","freeze","run-cell"))
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--grouping-dir", type=Path, default=OUT)
    parser.add_argument("--main-root", type=Path, default=OUT.parent)
    parser.add_argument("--feature-manifest", type=Path)
    parser.add_argument("--scope-amendment", default="")
    parser.add_argument("--nomination", action="append", default=[], help="Explicit existing main procedure, at most two; freeze only")
    parser.add_argument("--regime", choices=("hla","joint"))
    parser.add_argument("--cell")
    args = parser.parse_args()
    if args.mode == "draft":
        item = draft(args.out_dir, args.grouping_dir)
        print(json.dumps({"status":item["status"], "fit_graph":item["specification"]["fit_graph"]}, indent=2))
    elif args.mode == "freeze":
        if args.feature_manifest is None:
            parser.error("--feature-manifest required")
        freeze(args.out_dir,args.grouping_dir,args.main_root,args.feature_manifest,args.scope_amendment,args.nomination)
    else:
        if args.feature_manifest is None or args.regime is None or args.cell is None:
            parser.error("--feature-manifest, --regime and --cell required")
        run_cell(args.out_dir,args.grouping_dir,args.main_root,args.feature_manifest,args.regime,args.cell)


if __name__ == "__main__":
    main()
