"""Local identical-setting repeats of every locked embedding-selector decision.

Freeze only after every main sequence/embedding fold validates. No outer outcome
is used to choose a recipe: the scope is all five saved selector branches. This
replays final fits, not selection. Incomplete cohorts receive no report.
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
import uuid

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pepstab import nested
from pepstab.boltz_embeddings import sha256_file
from pepstab.round2_embedding import FeatureHead
from pepstab.round2_nested import Ancestry, config_digest, unlabelled
from scripts.round2_embedding_screen import (
    OUT as EMBEDDING, SEQUENCE_OUT as SEQUENCE, atomic_csv, atomic_json,
    choose_embedding_selector, load_features, training_frame,
)
from scripts.round2_results import load_completed_study, population_contract, validate_identity, _score_row

OUT = ROOT / "reports/further_research/round2/reproducibility/embedding_selector"
CACHE_ROOT = ROOT / "runs/round2_embedding_repeats"
SCOPE = "all five locked per-outer-fold embedding_selector decisions; final refits only, no reselection or outer-best winner"
METRICS = ("median_per_allele_spearman", "mae_log1p", "pooled_spearman",
           "pooled_pearson_log1p", "median_precision_at_10")


def read_json(path):
    return json.loads(Path(path).read_text())


def bind(sources, path):
    sources[str(Path(path).resolve())] = sha256_file(path)


def check_sources(sources):
    for path, expected in sources.items():
        if sha256_file(path) != expected:
            raise ValueError(f"locked input content changed: {path}")


def validate_protocol(protocol):
    if config_digest(protocol["specification"]) != protocol["protocol_hash"]:
        raise ValueError("repeat protocol content hash mismatch")


def runtime_manifest():
    return {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                         "processor": platform.processor(), "logical_cpus": os.cpu_count()},
            "python": platform.python_version(),
            "libraries": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "pandas", "threadpoolctl")},
            "threadpools": threadpool_info(), "thread_limit": 2,
            "thread_environment": {key: os.environ.get(key) for key in (
                "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
            "precision": {"features": "float32", "MLP_parameters_activations": "float32",
                          "targets": "original y_log1p dtype", "ridge": "frozen sklearn lsqr; float32 features, original targets"},
            "device": "CPU", "cloud_spend_usd": 0}


def audit_ancestry(path, train, held):
    item = read_json(path)
    ancestry = Ancestry()
    ancestry.rowsets.update(item["rowsets"])
    ancestry.records.extend(item["records"])
    if not ancestry.records:
        raise ValueError("repeat ancestry is empty")
    for key, ids in ancestry.rowsets.items():
        if key != hashlib.sha256(np.asarray(sorted(ids), dtype="<i8").tobytes()).hexdigest():
            raise ValueError("ancestry rowset content hash mismatch")
    ancestry.audit(train, held)
    return ancestry


def sequence_reference(sequence, fold, train, sources):
    """Reuse recorded reference repeats, never the primary fit or paired seeds."""
    directory = sequence["directory"] / f"fold_{fold}"
    marker = directory / "repeats_completed.json"
    complete = read_json(marker)
    if (complete.get("schema") != "round2_sequence_repeats_completion.2"
            or complete.get("repeats") != 2 or complete.get("outer_fold") != fold
            or complete.get("protocol_hash") != sequence["protocol"]["protocol_hash"]
            or complete.get("primary_completion_sha256") != sha256_file(directory / "completed.json")):
        raise ValueError("sequence reference repeat completion mismatch")
    if not all(complete.get("ancestry_audit", {}).get(key) for key in (
            "outer_cluster_exclusion", "predicted_cluster_exclusion")):
        raise ValueError("sequence reference repeat audit not passed")
    for key, name in (("repeat_prediction_sha256", "repeat_predictions.csv"),
                      ("ancestry_sha256", "ancestry_with_repeats.json"),
                      ("member_table_sha256", "repeat_members.csv")):
        path = directory / name
        if complete.get(key) != sha256_file(path):
            raise ValueError(f"sequence reference repeat hash mismatch: {name}")
        bind(sources, path)
    bind(sources, marker)
    bind(sources, directory / "runtime_manifest.json")
    held = train[nested.outer_folds(train) == fold]
    ancestry = audit_ancestry(directory / "ancestry_with_repeats.json", train, held)
    tables = pd.read_csv(directory / "repeat_predictions.csv")
    members = pd.read_csv(directory / "repeat_members.csv")
    candidate_id = sequence["saved_choices"][fold]["mlp_log"]["candidate"]["id"]
    for repeat in (1, 2):
        selected = tables[(tables.arm == "mlp_log") & (tables.repeat == repeat)]
        validate_identity(held, selected, fold)
        context = f"outer/{fold}/mlp_log/repeat/{repeat}"
        current = members[members.context == context]
        if len(current) != 30 or not current.candidate.eq(candidate_id).all():
            raise ValueError("sequence reference must contain 30 locked members per repeat")
        records = [r for r in ancestry.records if r.get("context") == context]
        if len(records) != 30 or any(r.get("candidate") != candidate_id for r in records):
            raise ValueError("sequence reference repeat ancestry/member mismatch")
        axes = {(enc, stop, seed) for enc in ("onehot", "blosum")
                for stop in range(5) for seed in range(3)}
        if {(r["encoding"], r["stop_fold"], r["seed"]) for r in records} != axes:
            raise ValueError("sequence reference repeat ancestry axes mismatch")
        if {(r.encoding, r.stop_fold, r.seed) for r in current.itertuples()} != axes:
            raise ValueError("sequence reference repeat member axes mismatch")
    return directory


def validate_implementation(study, sources):
    protocol = study["protocol"]
    expected = protocol.get("code_sha256", {})
    if study["kind"] == "sequence":
        directory = study["directory"]
        authorization = read_json(directory / "launch_authorization.json")
        reviewed = read_json(directory / "reviewed_implementation_hashes.json")
        hardware = read_json(directory / "hardware_amendment.json")
        amendments = read_json(directory / "protocol_amendments.json")
        initial = read_json(directory / "protocol_v0_pre_execution.json")
        if (authorization.get("approved") is not True
                or authorization.get("protocol_hash") != protocol["protocol_hash"]
                or hardware.get("scientific_protocol_hash") != protocol["protocol_hash"]
                or authorization.get("code_sha256") != reviewed
                or not reviewed or not set(expected).issubset(reviewed)):
            raise ValueError("sequence reviewed implementation authorization mismatch")
        if not amendments or amendments[0].get("prior_protocol_hash") != initial.get("protocol_hash"):
            raise ValueError("sequence protocol amendment ancestry mismatch")
        for name in ("launch_authorization.json", "reviewed_implementation_hashes.json",
                     "hardware_amendment.json", "protocol_amendments.json", "protocol_v0_pre_execution.json", "environment.json"):
            bind(sources, directory / name)
        previous_hardware = directory / "hardware_amendment_v1.json"
        if previous_hardware.exists():
            bind(sources, previous_hardware)
        # protocol.json preserves the original pre-amendment implementation
        # hashes; reviewed launch hashes are the authoritative executable chain.
        expected = reviewed
    for relative, digest in expected.items():
        path = ROOT / relative
        if sha256_file(path) != digest:
            raise ValueError(f"frozen authorized implementation changed: {relative}")
        bind(sources, path)


def prepare(train, embedding_directory, sequence_directory, manifest_path):
    """All completion markers gate any outcome reads and all fresh fitting."""
    for directory in (Path(embedding_directory), Path(sequence_directory)):
        if not all((directory / f"fold_{fold}/completed.json").is_file() for fold in range(5)):
            raise ValueError("all five main embedding and sequence folds must complete before repeats")
    contract = population_contract(train)
    sequence = load_completed_study(sequence_directory, "sequence", train, contract=contract)
    embedding = load_completed_study(embedding_directory, "embedding", train, contract=contract)
    sources = {**sequence["sources"], **embedding["sources"]}
    for reference in ("mlp_log", "sequence_selector"):
        if not np.allclose(embedding["predictions"][reference], sequence["predictions"][reference], rtol=0, atol=1e-12):
            raise ValueError("embedding copied sequence reference predictions mismatch")
    for study in (sequence, embedding):
        validate_implementation(study, sources)
    for path in (Path(__file__), ROOT / "pepstab/mlp.py", ROOT / "pepstab/nested.py",
                 ROOT / "scripts/round2_results.py", ROOT / "pepstab/evaluation.py",
                 ROOT / "pepstab/round2_sequence.py", ROOT / "scripts/round2_embedding_screen.py",
                 ROOT / "pepstab/round2_embedding.py", ROOT / "pepstab/round2_nested.py"):
        bind(sources, path)
    features, provenance = load_features(train, manifest_path)
    if Path(manifest_path).is_file():
        bind(sources, manifest_path)
    hashes = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in features.items()}
    decisions = []
    for fold in range(5):
        emb_path = Path(embedding_directory) / f"fold_{fold}"
        seq_path = Path(sequence_directory) / f"fold_{fold}"
        complete = embedding["completions"][fold]
        if (complete.get("status") != "complete"
                or complete.get("sequence_protocol_hash") != sequence["protocol"]["protocol_hash"]
                or complete.get("sequence_prediction_sha256") != sequence["completions"][fold]["prediction_sha256"]):
            raise ValueError("embedding completion sequence binding mismatch")
        expected = choose_embedding_selector(embedding["saved_choices"][fold], sequence["saved_choices"][fold])
        if embedding["saved_selectors"][fold] != expected:
            raise ValueError("saved embedding selector violates frozen inner-only rule")
        feature_path = emb_path / "feature_identity.json"
        identity = read_json(feature_path)
        if (identity.get("protocol_hash") != embedding["protocol"]["protocol_hash"]
                or identity.get("feature_hashes") != hashes or identity.get("feature_provenance") != provenance):
            raise ValueError("repeat features differ from primary features")
        bind(sources, feature_path)
        bind(sources, seq_path / "runtime_manifest.json")
        # Main component markers do not bind ancestry/members. Snapshot and audit
        # all those files into the new repeat protocol before any repeated fit.
        held = train[nested.outer_folds(train) == fold]
        for arm in embedding["protocol"]["specification"]["component_order"]:
            arm_path = emb_path / arm
            marker = read_json(arm_path / "completed.json")
            if marker["prediction_sha256"] != sha256_file(arm_path / "predictions.csv"):
                raise ValueError("primary component prediction hash mismatch")
            if (marker.get("n_predictions") != len(held)
                    or not all(marker.get("audit", {}).get(k) for k in ("outer_cluster_exclusion", "predicted_cluster_exclusion"))):
                raise ValueError("primary component completion coverage/audit mismatch")
            table = pd.read_csv(arm_path / "predictions.csv")
            raw = validate_identity(held, table, fold)
            if (not table.arm.eq(arm).all()
                    or not table.selected_candidate.eq(embedding["saved_choices"][fold][arm]["candidate"]["id"]).all()
                    or not np.allclose(raw, embedding["predictions"][arm][nested.outer_folds(train) == fold], rtol=0, atol=1e-12)):
                raise ValueError("primary component and finalized predictions differ")
            audit_ancestry(arm_path / "ancestry.json", train, held)
            for name in ("completed.json", "predictions.csv", "ancestry.json", "members.csv"):
                bind(sources, arm_path / name)
            if arm.endswith("_mlp"):
                if marker.get("seed_sha256") != sha256_file(arm_path / "seed_predictions.csv"):
                    raise ValueError("primary component seed hash mismatch")
                bind(sources, arm_path / "seed_predictions.csv")
        arm = expected["arm"]
        candidate = (sequence if arm == "mlp_log" else embedding)["saved_choices"][fold][arm]["candidate"]
        decision = {"outer_fold": fold, "selected_arm": arm, "candidate": candidate,
                    "new_members_per_repeat": 0 if arm == "mlp_log" else (30 if candidate["learner"] == "mlp" else 1)}
        if arm == "mlp_log":
            decision["sequence_reference_directory"] = str(sequence_reference(sequence, fold, train, sources).resolve())
        decisions.append(decision)
    spec = {"schema": "round2_embedding_execution_repeats.1", "scope": SCOPE,
            "population": contract, "embedding_protocol_hash": embedding["protocol"]["protocol_hash"],
            "sequence_protocol_hash": sequence["protocol"]["protocol_hash"],
            "repeat_count": 2, "final": True, "final_seeds": list(range(6)),
            "max_epochs": embedding["protocol"]["specification"]["head"]["mlp"]["max_epochs"],
            "threads": 2, "device": "CPU", "cloud_spend_usd": 0, "decisions": decisions,
            "fit_bound": {"absolute_upper_bound": 300,
                          "locked_additional_model_fits": 2 * sum(d["new_members_per_repeat"] for d in decisions),
                          "selection_fits": 0, "sequence_reference_new_fits": 0},
            "primary_selector_array_sha256": hashlib.sha256(np.asarray(embedding["predictions"]["embedding_selector"], dtype="<f8").tobytes()).hexdigest(),
            "feature_hashes": hashes, "feature_provenance": provenance,
            "production_manifest": str(Path(manifest_path).resolve()), "sources": sources,
            "limitations": ["Measures final execution of the selector's locked decisions; selection is not refitted",
                            "All five selected branches are included without choosing an outer-outcome winner",
                            "Within-machine identical settings; no cross-platform equivalence or assay-noise claim",
                            "Different-seed diagnostics are separate and are not used here"]}
    check_sources(sources)
    return spec, features, embedding["predictions"]["embedding_selector"]


def freeze(out, train, embedding_directory, sequence_directory, manifest_path):
    spec, _, _ = prepare(train, embedding_directory, sequence_directory, manifest_path)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "protocol.json"
    protocol = {"protocol_hash": config_digest(spec), "specification": spec}
    if path.exists() and read_json(path) != protocol:
        raise ValueError("repeat protocol already locked with different inputs")
    atomic_json(path, protocol)
    return protocol


def validate_members(members, ancestry, candidate, context):
    expected = 30 if candidate["learner"] == "mlp" else 1
    if len(members) != expected or len(ancestry.records) != expected:
        raise ValueError("repeat requires complete final member/ancestry recipe")
    if any(m.get("context") != context or m.get("candidate") != candidate["id"] for m in members):
        raise ValueError("repeat member configuration differs from locked recipe")
    if any(r.get("context") != context or r.get("candidate") != candidate["id"] for r in ancestry.records):
        raise ValueError("repeat ancestry configuration differs from locked recipe")
    if candidate["learner"] == "mlp":
        axes = {(stop, seed) for stop in range(5) for seed in range(6)}
        for records in (members, ancestry.records):
            if {(m.get("stop_fold"), m.get("seed")) for m in records} != axes:
                raise ValueError("repeat requires five stopping folds times six identical seeds")


def completed_repeat(directory, protocol, train, held, fold, repeat):
    marker = read_json(directory / "completed.json")
    if (marker.get("status") != "complete" or marker.get("protocol_hash") != protocol["protocol_hash"]
            or marker.get("outer_fold") != fold or marker.get("repeat") != repeat):
        raise ValueError("repeat completion identity mismatch")
    check_sources(marker["output_hashes"])
    check_sources(marker["fit_cache_hashes"])
    decision = protocol["specification"]["decisions"][fold]
    if (marker.get("new_model_fits") != decision["new_members_per_repeat"]
            or not all(marker.get("audit", {}).get(k) for k in ("outer_cluster_exclusion", "predicted_cluster_exclusion"))):
        raise ValueError("repeat completion fitting count or audit mismatch")
    table = pd.read_csv(directory / "predictions.csv")
    validate_identity(held, table, fold)
    if (not table.arm.eq("embedding_selector").all() or not table.repeat.eq(repeat).all()
            or not table.selected_arm.eq(decision["selected_arm"]).all()
            or not table.selected_candidate.eq(decision["candidate"]["id"]).all()):
        raise ValueError("repeat predictions differ from locked selector recipe")
    audit_ancestry(directory / "ancestry.json", train, held)
    return table


def run_repeat(out, protocol, train, features, decision, repeat):
    validate_protocol(protocol)
    check_sources(protocol["specification"]["sources"])
    if decision not in protocol["specification"]["decisions"] or repeat not in (1, 2):
        raise ValueError("repeat decision or index outside frozen protocol")
    fold = decision["outer_fold"]
    outer = nested.outer_folds(train)
    fit, held = train[outer != fold], train[outer == fold]
    directory = Path(out) / f"fold_{fold}" / f"repeat_{repeat}"
    if (directory / "completed.json").exists():
        return completed_repeat(directory, protocol, train, held, fold, repeat)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "runtime_manifest.json", runtime_manifest())
    candidate, arm = decision["candidate"], decision["selected_arm"]
    cache_hashes = {}
    if arm == "mlp_log":
        source = Path(decision["sequence_reference_directory"])
        tables = pd.read_csv(source / "repeat_predictions.csv")
        selected = tables[(tables.arm == arm) & (tables.repeat == repeat)].copy()
        prediction = validate_identity(held, selected, fold)
        ancestry = audit_ancestry(source / "ancestry_with_repeats.json", train, held)
        members = pd.read_csv(source / "repeat_members.csv")
        members = members[members.context == f"outer/{fold}/mlp_log/repeat/{repeat}"]
        mode = "validated separately recorded sequence reference repeat"
        atomic_json(directory / "reference_runtime_manifest.json", {
            "recorded_reference_runtime": read_json(source / "runtime_manifest.json"),
            "recorded_reference_environment": read_json(source.parent / "environment.json"),
            "precision": "reference MLP parameters/activations float32 per hash-bound implementation"})
    else:
        # Each incomplete attempt gets a new, empty namespace. Never read any
        # primary cache or a cache left behind by an interrupted repeat.
        cache = CACHE_ROOT / protocol["protocol_hash"] / f"fold_{fold}" / f"repeat_{repeat}" / uuid.uuid4().hex
        cache.mkdir(parents=True, exist_ok=False)
        ancestry = Ancestry()
        context = f"outer/{fold}/{arm}/outer_refit"  # exact primary context
        head = FeatureHead(train, {candidate["representation"]: features[candidate["representation"]]},
                           ancestry, cache_dir=cache, max_epochs=protocol["specification"]["max_epochs"],
                           final_seeds=tuple(protocol["specification"]["final_seeds"]))
        prediction = head.predict(fit, unlabelled(held), candidate, context, final=True)
        validate_members(head.members, ancestry, candidate, context)
        members = pd.DataFrame(head.members)
        mode = "fresh identical-setting embedding final refit"
        for path in cache.glob("*.json"):
            bind(cache_hashes, path)
        if not cache_hashes:
            raise ValueError("fresh repeat did not persist its fit cache evidence")
    table = held[["pair_id", "cluster_id", "allele"]].copy()
    table = table.assign(outer_fold=fold, arm="embedding_selector", repeat=repeat,
                         selected_arm=arm, selected_candidate=candidate["id"], y_pred=prediction)
    audit = ancestry.audit(train, held)
    atomic_csv(directory / "predictions.csv", table)
    atomic_csv(directory / "members.csv", members)
    atomic_json(directory / "ancestry.json", ancestry.as_dict())
    hashes = {}
    for name in ("predictions.csv", "members.csv", "ancestry.json", "runtime_manifest.json"):
        bind(hashes, directory / name)
    if arm == "mlp_log":
        bind(hashes, directory / "reference_runtime_manifest.json")
    check_sources(protocol["specification"]["sources"])
    atomic_json(directory / "completed.json", {"schema": "round2_embedding_repeat_completion.1",
        "status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": protocol["protocol_hash"], "outer_fold": fold, "repeat": repeat,
        "mode": mode, "new_model_fits": decision["new_members_per_repeat"], "audit": audit,
        "output_hashes": hashes, "fit_cache_hashes": cache_hashes})
    return completed_repeat(directory, protocol, train, held, fold, repeat)


def report(out, protocol, train, primary):
    out = Path(out)
    paths = [out / f"fold_{fold}/repeat_{repeat}/completed.json" for fold in range(5) for repeat in (1, 2)]
    if not all(path.is_file() for path in paths):
        raise ValueError("all ten fold-repeat markers required; no partial-cohort report")
    validate_protocol(protocol)
    check_sources(protocol["specification"]["sources"])
    outer = nested.outer_folds(train)
    if hashlib.sha256(np.asarray(primary, dtype="<f8").tobytes()).hexdigest() != protocol["specification"]["primary_selector_array_sha256"]:
        raise ValueError("report primary predictions differ from frozen selector")
    predictions = {"primary": np.asarray(primary), "repeat1": np.empty(len(train)), "repeat2": np.empty(len(train))}
    tables, sources = [], {}
    for fold in range(5):
        for repeat in (1, 2):
            directory = out / f"fold_{fold}/repeat_{repeat}"
            table = completed_repeat(directory, protocol, train, train[outer == fold], fold, repeat)
            predictions[f"repeat{repeat}"][outer == fold] = validate_identity(train[outer == fold], table, fold)
            tables.append(table)
            bind(sources, directory / "completed.json")
    metric_rows, differences = [], []
    for minimum in (20, 50):
        scores = {name: _score_row(train, name, values, minimum, scope="pooled")[0]
                  for name, values in predictions.items()}
        metric_rows.extend(scores.values())
        for baseline, candidate in (("primary", "repeat1"), ("primary", "repeat2"), ("repeat1", "repeat2")):
            delta = np.abs(predictions[candidate] - predictions[baseline])
            differences.append({"baseline": baseline, "candidate": candidate, "panel_min_rows": minimum,
                "comparison_kind": "identical_setting_execution_repeat", "n_rows": len(train),
                "max_abs_prediction_difference": float(delta.max()),
                "mean_abs_prediction_difference": float(delta.mean()),
                **{f"delta_{metric}": (None if scores[baseline][metric] is None or scores[candidate][metric] is None
                    else float(scores[candidate][metric] - scores[baseline][metric])) for metric in METRICS}})
    primary_table = train[["pair_id", "cluster_id", "allele"]].assign(
        outer_fold=outer, arm="embedding_selector", repeat=0, y_pred=primary)
    atomic_csv(out / "predictions.csv", pd.concat([primary_table] + tables, ignore_index=True))
    atomic_csv(out / "pooled_metrics.csv", pd.DataFrame(metric_rows))
    atomic_csv(out / "execution_differences.csv", pd.DataFrame(differences))
    text = ("# Locked embedding-selector execution repeats\n\n" + SCOPE + ".\n\n"
            "Two identical-seed/settings executions replay the frozen selected recipe for each fold. "
            "The 30-member neural or single deterministic ridge final recipe is refitted in fresh cache namespaces; "
            "selected mlp_log branches reuse validated separately recorded reference repeats. "
            "Selection is not refitted. This differs from paired-seed variation and is not assay noise.\n\n"
            "All original training rows have primary, repeat1 and repeat2 predictions. "
            "execution_differences.csv records primary-versus-each-repeat and repeat1-versus-repeat2 maximum/mean "
            "absolute prediction differences and candidate-minus-baseline pooled metric differences at frozen "
            "20/50-row allele panels. No partial-cohort results or outer-outcome winner are reported.\n\n"
            f"Additional fits: {protocol['specification']['fit_bound']['locked_additional_model_fits']} "
            "(absolute upper bound 300); zero selection fits and zero new sequence reference fits. "
            "Hardware, libraries, threads, precision, ancestry, member tables, primary sources and fresh fit caches "
            "are hash-bound by the protocol and atomic completion markers. Within-machine only.\n")
    text += "\n| Comparison | Allele panel | Max absolute prediction difference | Mean absolute prediction difference | Median per-allele Spearman delta | MAE delta |\n|---|---:|---:|---:|---:|---:|\n"
    def display(value):
        return "unavailable" if value is None else f"{value:.12g}"
    for row in differences:
        text += (f"| {row['baseline']} versus {row['candidate']} | {row['panel_min_rows']} | "
                 f"{display(row['max_abs_prediction_difference'])} | {display(row['mean_abs_prediction_difference'])} | "
                 f"{display(row['delta_median_per_allele_spearman'])} | {display(row['delta_mae_log1p'])} |\n")
    from pepstab.boltz_embeddings import _atomic_bytes
    _atomic_bytes(out / "README.md", text.encode())
    for name in ("protocol.json", "predictions.csv", "pooled_metrics.csv", "execution_differences.csv", "README.md"):
        bind(sources, out / name)
    atomic_json(out / "completed.json", {"schema": "round2_embedding_repeat_report.1", "status": "complete",
        "protocol_hash": protocol["protocol_hash"], "n_rows": len(train), "repeats": 2, "scope": SCOPE,
        "sources": sources, "completed_at_utc": datetime.now(timezone.utc).isoformat()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-directory", type=Path, default=EMBEDDING)
    parser.add_argument("--sequence-directory", type=Path, default=SEQUENCE)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--production-manifest", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true", help="validate and lock only; no fitting")
    parser.add_argument("--execute", action="store_true", help="explicitly authorize local final refits after all gates")
    args = parser.parse_args()
    if args.freeze == args.execute:
        parser.error("choose exactly one of --freeze or --execute")
    train = training_frame()
    with threadpool_limits(limits=2):
        if args.freeze:
            protocol = freeze(args.out_dir, train, args.embedding_directory, args.sequence_directory, args.production_manifest)
            print(json.dumps({"protocol_hash": protocol["protocol_hash"], "fit_bound": protocol["specification"]["fit_bound"]}))
        else:
            protocol = read_json(args.out_dir / "protocol.json")
            spec, features, primary = prepare(train, args.embedding_directory, args.sequence_directory, args.production_manifest)
            if config_digest(spec) != protocol["protocol_hash"] or protocol["specification"] != spec:
                raise ValueError("execution inputs differ from frozen repeat protocol")
            for decision in spec["decisions"]:
                for repeat in (1, 2):
                    run_repeat(args.out_dir, protocol, train, features, decision, repeat)
                    print(f"fold {decision['outer_fold']} repeat {repeat} complete", flush=True)
            report(args.out_dir, protocol, train, primary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
