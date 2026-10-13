"""Candidate only until an explicit root decision freezes this new namespace."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pepstab import followup3_inverse as prior
from pepstab import followup3_inverse_fulltrain as core
from pepstab import followup3_inverse_parser as parser
from scripts import followup3_inverse as existing

OUT = ROOT / "reports/followup3/inverse_fulltrain_candidate"
CACHE = ROOT / "reports/followup3/inverse_execution_v2"
REFERENCE = "reports/further_research/round2/results/main/matched_predictions.csv"
sha, write_json = existing.sha, existing.write_json


def write_csv(path, frame):
    path = Path(path)
    text = frame.to_csv(index=False)
    if path.exists():
        if path.read_text() != text:
            raise ValueError("refusing to overwrite changed artifact")
        return
    with path.open("x") as handle:
        handle.write(text)


def draft(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cached_panel, old_protocol = existing.read_panel(CACHE)
    recovery = json.loads((CACHE / "parser_recovery_receipt.json").read_text())
    if recovery["status"] != "passed":
        raise ValueError("complete audited source cache required")
    sources = dict(old_protocol["source_sha256"])
    for name in ("pepstab/followup3_inverse_fulltrain.py", "scripts/followup3_inverse_fulltrain.py", "pepstab/followup3_inverse_parser.py",
                 "pepstab/followup3_energy.py", "pepstab/splits.py", "reports/followup3/inverse_fulltrain_candidate/SPECIFICATION.md"):
        sources[name] = sha(ROOT / name)
    inputs = dict(old_protocol["input_sha256"])
    for name in ("reports/followup3/inverse_execution_v2/protocol.json", "reports/followup3/inverse_execution_v2/extraction_receipt.json",
                 "reports/followup3/inverse_execution_v2/parser_recovery_receipt.json", "reports/further_research/round2/extractor/full_training/feature_bundles.json"):
        inputs[name] = sha(ROOT / name)
    splits = pd.read_csv(ROOT / "data/splits.csv")
    panel = core.full_panel(splits, pd.read_csv(ROOT / "reports/further_research/round2/sequence/fold_manifest.csv"), pd.read_csv(ROOT / "data/structural_cohort.csv"))
    if len(panel) != 19716 or panel.allele.nunique() != 75:
        raise ValueError("exact full ORIGINAL-train cohort required")
    cached = []
    for name, expected in recovery["completed_row_sha256"].items():
        path = CACHE / "score_rows" / name
        if sha(path) != expected:
            raise ValueError("source cached row bytes changed")
        cached.append(json.loads(path.read_text()))
    prior.validate_scores(cached_panel, pd.DataFrame(cached))
    merged = cached_panel.merge(panel, on=["allele", "peptide"], suffixes=("_old", "_new"), validate="one_to_one")
    if len(cached) != 4449 or len(merged) != 4449 or any(not merged[c + "_old"].equals(merged[c + "_new"]) for c in ("cluster_id", "outer_fold", "complex_id", "folder")):
        raise ValueError("cached feature identities differ in expanded cohort")
    batches = core.extraction_batches(panel, set(zip(cached_panel.allele, cached_panel.peptide)))
    if list(map(len, batches)) != [7634, 7633]:
        raise ValueError("exact disjoint15267 new rows required")
    write_csv(out / "panel.csv", panel)
    for i, batch in enumerate(batches): write_csv(out / f"batch_{i}.csv", batch)
    write_json(out / "cache_manifest.json", dict(source_directory=str(CACHE.relative_to(ROOT)), parent_protocol_file_sha256=sha(CACHE / "protocol.json"),
        parser_receipt_sha256=sha(CACHE / "parser_recovery_receipt.json"), row_sha256=recovery["completed_row_sha256"], cached_rows=4449))
    for name in ("cache_manifest.json", "batch_0.csv", "batch_1.csv"):
        inputs[str((out / name).relative_to(ROOT))] = sha(out / name)
    # Identify the already saved practical reference using identity columns only.
    reference = pd.read_csv(ROOT / REFERENCE, usecols=["pair_id", "cluster_id", "allele", "outer_fold"])
    identity = splits.query("split == 'train'")[["pair_id", "cluster_id", "allele", "peptide"]].merge(reference,
        on=["pair_id", "cluster_id", "allele"], validate="one_to_one")
    identity = identity.merge(panel[["allele", "peptide", "outer_fold"]], on=["allele", "peptide"], validate="one_to_one", suffixes=("_saved", "_new"))
    if len(reference) != 19716 or len(identity) != 19716 or not identity.outer_fold_saved.equals(identity.outer_fold_new):
        raise ValueError("saved practical reference identity differs")
    inputs[REFERENCE] = sha(ROOT / REFERENCE)
    audit = json.loads((CACHE / "independent_audit.json").read_text())
    if audit["status"] != "passed" or audit["rows"] != 4449: raise ValueError("independently audited old query predictions required")
    for name in ("panel.csv", "independent_audit.json", "completion_audit.json"):
        inputs[str((CACHE / name).relative_to(ROOT))] = sha(CACHE / name)
    for learner in ("mlp", "ridge"):
        for fold in range(5):
            location = CACHE / "assessment" / learner / f"fold_{fold}"
            receipt = json.loads((location / "receipt.json").read_text())
            if receipt["learner"] != learner or receipt["folds"] != [fold] or sha(location / "predictions.csv") != receipt["predictions_sha256"]:
                raise ValueError("old fixed-query prediction receipt differs")
            for name in ("predictions.csv", "receipt.json"):
                inputs[str((location / name).relative_to(ROOT))] = sha(location / name)
    spec = dict(schema="followup3-inverse-fulltrain-candidate.1", population="complete ORIGINAL train; selected development followup after balanced outcomes",
        n_rows=19716, n_alleles=75, n_clusters=int(panel.cluster_id.nunique()), arms=list(core.ARMS), ridge_fits=150, mlp_fits=300,
        main_comparison="mlp:R5_PC minus R5", secondary=["mlp:R5_PC minus R5_P", "mlp:R5_P minus R5", "ridge analogues"],
        ridge_alpha=[10000., 100., 1.], mlp_l2=[1., .01, .0001], mlp=old_protocol["specification"]["mlp"],
        scoring=old_protocol["specification"]["scoring"], selection_metric=old_protocol["specification"]["selection_metric"],
        outer_folds=5, inner_folds=3, old_subset_predictions_reused=False, cached_feature_rows=4449, new_feature_rows=15267,
        parser="unique-whole-chain-exact-native-v1; all unchanged metadata/backbone guards", scientific_versions={p:version(p) for p in existing.PACKAGES},
        bootstrap=dict(n_draws=2000, seed=20261003, unit="whole inherited peptide cluster", primary_panel=20, sensitivity_panel=50, interval=95),
        practical_reference=dict(path=REFERENCE, column="embedding_selector", scope="descriptive fixed reference only; no selection or matching-head claim"),
        fixed_query_check=dict(rows=4449, panels=[20, 50], old_arms=list(core.ARMS), learners=["mlp", "ridge"],
            outputs="absolute old/new points; net/context/peptide increments; old-to-new changes; no extra CI/models/bootstrap", interpretation="combined source coverage, tuning and stopping changes at fixed query composition; not pure sample-size causal effect"),
        family="not one of three frozen tranche primaries; no tranche success claim", resource_proposal=dict(extraction_usd=34, remaining_usd=6, total_usd=40, sandboxes=2, hours_each=5),
        limitations=["selected after balanced-panel outcomes; neither independent replication nor untouched validation", "bootstrap conditions on fitted predictions and omits followup/model selection", "density and cohort composition change together", "ridge disagreement must be reported", "no kinetics, causal geometry, native pose validation or deployed-head superiority claim", "no saved-weight replay"])
    write_json(out / "protocol.draft.json", dict(status="DRAFT_NOT_FROZEN_NOT_ADMITTED", drafted_at_utc=datetime.now(timezone.utc).isoformat(),
        specification=spec, protocol_hash=prior.digest(spec), panel_sha256=sha(out / "panel.csv"), source_sha256=sources, input_sha256=inputs))
    print(json.dumps(dict(status="draft", rows=19716, cached=4449, batch_rows=list(map(len, batches)), draft_file_sha256=sha(out / "protocol.draft.json"))))


def freeze(out, decision):
    """Only an explicit root decision after the energy audit can authorize freeze."""
    out = Path(out)
    approval = json.loads(Path(decision).read_text())
    draft = json.loads((out / "protocol.draft.json").read_text())
    if approval.get("approve_fulltrain_inverse") is not True or approval.get("energy_final_audit_complete") is not True or approval.get("candidate_draft_sha256") != sha(out / "protocol.draft.json"):
        raise ValueError("explicit root decision bound to candidate and final energy audit required")
    for name, expected in {**draft["source_sha256"], **draft["input_sha256"]}.items():
        if sha(ROOT / name) != expected: raise ValueError("draft source/input changed; review a new draft")
    draft["status"] = "FROZEN_SCOPE_ONLY_NOT_PAID_LAUNCH_APPROVAL"
    draft["root_decision_sha256"] = sha(decision)
    draft["input_sha256"][str(Path(decision).resolve().relative_to(ROOT))] = sha(decision)
    write_json(out / "protocol.json", draft)


def verify(out):
    panel, protocol = existing.read_panel(out)
    if protocol["status"] != "FROZEN_SCOPE_ONLY_NOT_PAID_LAUNCH_APPROVAL" or len(panel) != 19716:
        raise ValueError("root-approved prospective fulltrain freeze required")
    manifest = json.loads((Path(out) / "cache_manifest.json").read_text())
    cache = ROOT / manifest["source_directory"] / "score_rows"
    for name, expected in manifest["row_sha256"].items():
        if sha(cache / name) != expected: raise ValueError("immutable cached feature row changed")
    return panel, protocol, cache, manifest


def extract_batch(out, batch, volume_root, workers):
    panel, _, _, _ = verify(out)
    if batch not in (0, 1) or not 1 <= workers <= 8: raise ValueError("two batches/eight spawn workers maximum")
    records = pd.read_csv(Path(out) / f"batch_{batch}.csv")
    directory = Path(out) / "score_rows"; directory.mkdir(exist_ok=True)
    tasks = [(str(volume_root), row, str(directory), False) for row in records.to_dict(orient="records")]
    result = parser.amended_run_tasks(tasks, workers)
    prior.validate_scores(records, pd.DataFrame(result))
    verify(out)
    write_json(Path(out) / f"batch_{batch}.receipt.json", dict(batch=batch, rows=len(result), row_sha256={prior.digest([r['allele'], r['peptide']]) + '.json':sha(directory / (prior.digest([r['allele'], r['peptide']]) + '.json')) for r in result},
        max_workers=workers, threads_per_worker=2, worker_start_method="spawn" if workers > 1 else "serial"))


def collect(out):
    panel, protocol, cache, manifest = verify(out)
    rows = [json.loads((cache / name).read_text()) for name in manifest["row_sha256"]]
    for batch in (0, 1):
        receipt = json.loads((Path(out) / f"batch_{batch}.receipt.json").read_text())
        for name, expected in receipt["row_sha256"].items():
            path = Path(out) / "score_rows" / name
            if sha(path) != expected: raise ValueError("new batch row changed")
            rows.append(json.loads(path.read_text()))
    matched = prior.validate_scores(panel, pd.DataFrame(rows))
    write_csv(Path(out) / "scores.csv", matched)
    write_json(Path(out) / "extraction_receipt.json", dict(protocol_hash=protocol["protocol_hash"], n_rows=len(matched), cached_rows=4449, new_rows=15267,
        scores_sha256=sha(Path(out) / "scores.csv"), cached_bytes_unchanged=True, new_row_sha256={p.name:sha(p) for p in (Path(out) / "score_rows").glob("*.json")}))


def assess(out, learner, fold):
    from threadpoolctl import threadpool_limits
    _, protocol, _, _ = verify(out)
    if {p:version(p) for p in existing.PACKAGES} != protocol["specification"]["scientific_versions"]: raise ValueError("scientific versions changed")
    frame = existing.labelled_panel(out)
    receipt = json.loads((Path(out) / "extraction_receipt.json").read_text())
    if sha(Path(out) / "scores.csv") != receipt["scores_sha256"]: raise ValueError("score aggregate changed")
    scored = prior.validate_scores(frame[list(prior.PANEL_COLUMNS)], pd.read_csv(Path(out) / "scores.csv", float_precision="round_trip"))
    provenance = json.loads((Path(out) / "base_features.provenance.json").read_text())
    if sha(Path(out) / "base_features.npz") != provenance["base_sha256"]: raise ValueError("baseline features changed")
    with np.load(Path(out) / "base_features.npz", allow_pickle=False) as data:
        if not np.array_equal(data["allele"], frame.allele) or not np.array_equal(data["peptide"], frame.peptide): raise ValueError("feature natural keys differ")
        arms = core.feature_arms(data, scored)
    with threadpool_limits(limits=2): held, predictions, choices, ancestry = core.assess_heads(frame, arms, learner, fold)
    core.audit_heads(frame, ancestry, choices, learner, fold)
    location = Path(out) / "assessment" / learner / f"fold_{fold}"; location.mkdir(parents=True, exist_ok=True)
    result = held[["allele", "peptide", "cluster_id", "outer_fold"]].copy()
    for arm, values in predictions.items(): result[arm] = values
    write_csv(location / "predictions.csv", result)
    write_json(location / "selection.json", choices); write_json(location / "ancestry.json", ancestry)
    write_json(location / "runtime.json", dict(python=sys.version, platform=platform.platform(), packages={p:version(p) for p in existing.PACKAGES}, threads=2))
    write_json(location / "receipt.json", dict(learner=learner, fold=fold, fits=60 if learner == "mlp" else 30,
        inputs={name:sha(Path(out) / name) for name in ("protocol.json", "panel.csv", "scores.csv", "base_features.npz", "base_features.provenance.json")},
        outputs={name:sha(location / name) for name in ("predictions.csv", "selection.json", "ancestry.json", "runtime.json")}))


def report(out):
    from pepstab.round2_reporting import compare_predictions
    verify(out); frame = existing.labelled_panel(out); comparisons, fixed_queries = [], []
    old_panel = pd.read_csv(CACHE / "panel.csv")
    for learner in ("mlp", "ridge"):
        tables, feature_hashes = [], None
        for fold in range(5):
            location = Path(out) / "assessment" / learner / f"fold_{fold}"
            receipt = json.loads((location / "receipt.json").read_text())
            if receipt["fold"] != fold or receipt["learner"] != learner or receipt["fits"] != (60 if learner == "mlp" else 30): raise ValueError("head receipt scope differs")
            for name, expected in receipt["inputs"].items():
                if sha(Path(out) / name) != expected: raise ValueError("head input features/protocol changed")
            for name, expected in receipt["outputs"].items():
                if sha(location / name) != expected: raise ValueError("head output changed")
            ancestry = json.loads((location / "ancestry.json").read_text())
            core.audit_heads(frame, ancestry, json.loads((location / "selection.json").read_text()), learner, fold)
            if feature_hashes is not None and feature_hashes != ancestry["feature_sha256"]: raise ValueError("fold feature bytes differ")
            feature_hashes = ancestry["feature_sha256"]
            table = pd.read_csv(location / "predictions.csv", float_precision="round_trip")
            if not table.outer_fold.eq(fold).all(): raise ValueError("saved prediction fold differs")
            tables.append(table)
        prediction = pd.concat(tables, ignore_index=True)
        if len(prediction) != len(frame) or prediction.duplicated(["allele", "peptide"]).any(): raise ValueError("complete matched predictions required")
        joined = frame.merge(prediction, on=["allele", "peptide", "cluster_id", "outer_fold"], validate="one_to_one")
        if len(joined) != len(frame): raise ValueError("prediction natural keys differ")
        values = {a:joined[a].to_numpy(float) for a in core.ARMS}
        old_prediction = pd.concat([pd.read_csv(CACHE / "assessment" / learner / f"fold_{fold}" / "predictions.csv", float_precision="round_trip") for fold in range(5)], ignore_index=True)
        keys = ["allele", "peptide", "cluster_id", "outer_fold"]
        if len(old_prediction) != 4449 or old_prediction.duplicated(["allele", "peptide"]).any(): raise ValueError("old query prediction coverage differs")
        query = old_panel[keys].merge(joined, on=keys, validate="one_to_one").merge(old_prediction[keys + list(core.ARMS)],
            on=keys, validate="one_to_one", suffixes=("_full", "_old"))
        if len(query) != 4449: raise ValueError("fixed-query natural keys differ")
        fixed_queries.append(core.fixed_query_table(query, {a:query[a + "_old"].to_numpy(float) for a in core.ARMS},
            {a:query[a + "_full"].to_numpy(float) for a in core.ARMS}, learner))
        for minimum in (20, 50):
            result, boot, _ = compare_predictions(joined, values, [("R5", "R5_PC"), ("R5_P", "R5_PC"), ("R5", "R5_P")], min_rows=minimum, n_boot=2000, seed=20261003)
            result["learner"] = learner; result["scope"] = "selected development followup; conditional95%; not tranche primary"
            comparisons.append(result); write_csv(Path(out) / f"bootstrap_{learner}_min{minimum}.csv", boot)
    write_csv(Path(out) / "comparisons.csv", pd.concat(comparisons, ignore_index=True))
    write_csv(Path(out) / "fixed_query_descriptive.csv", pd.concat(fixed_queries, ignore_index=True))
    write_json(Path(out) / "completion_audit.json", dict(status="passed", rows=19716, mlp_fits=300, ridge_fits=150,
        independent_selection_replay=True, cluster_fit_stop_scaler_audit=True, saved_weight_replay=False, tranche_primary_success_claim=False,
        outputs={p.name:sha(p) for p in Path(out).glob("*.csv")}))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=("draft", "freeze", "extract-batch", "collect", "stage", "assess", "report"))
    cli.add_argument("--out", type=Path, default=OUT); cli.add_argument("--decision", type=Path)
    cli.add_argument("--batch", type=int, choices=(0, 1)); cli.add_argument("--workers", type=int, default=8)
    cli.add_argument("--volume-root", type=Path, default=Path("/structures")); cli.add_argument("--bundle-root", type=Path)
    cli.add_argument("--learner", choices=("mlp", "ridge")); cli.add_argument("--outer-fold", type=int, choices=range(5))
    args = cli.parse_args()
    if args.command == "draft": draft(args.out)
    elif args.command == "freeze": freeze(args.out, args.decision)
    elif args.command == "extract-batch": extract_batch(args.out, args.batch, args.volume_root, args.workers)
    elif args.command == "collect": collect(args.out)
    elif args.command == "stage": verify(args.out); existing.stage_features(args.out, args.bundle_root)
    elif args.command == "assess": assess(args.out, args.learner, args.outer_fold)
    else: report(args.out)


if __name__ == "__main__": main()
