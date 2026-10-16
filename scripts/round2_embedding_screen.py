"""Freeze and resume the bounded Boltz representation screen on original train.

The freeze step needs no embedding cache and performs no fitting. Fitting must
be invoked explicitly after the extraction and evaluation gates pass.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pepstab import nested
from pepstab.boltz_embeddings import _atomic_bytes, json_bytes, sha256_file
from pepstab.data import load_with_splits
from pepstab.features import build_features
from pepstab.round2_embedding import EMBEDDING_PRIMARY, REPRESENTATIONS, FeatureHead, head_candidates
from pepstab.round2_nested import Ancestry, align_predictions, config_digest, select, unlabelled
from pepstab.round2_representations import FEATURE_DIMENSIONS, FEATURE_SCHEMA, feature_protocol, load_contact_map
from scripts.phase_d_structural_residual import structural_columns

OUT = ROOT / "reports/further_research/round2/embedding"
SEQUENCE_OUT = ROOT / "reports/further_research/round2/sequence"
STRUCTURAL = ROOT / "reports/stage4c5_features_production.csv"
LEARNERS = ("ridge", "mlp")
COMPONENT_ORDER = tuple(f"{rep}_{learner}" for rep in REPRESENTATIONS for learner in LEARNERS)
SELECTOR_ORDER = ("mlp_log",) + tuple(f"{rep}_{learner}" for rep in EMBEDDING_PRIMARY for learner in LEARNERS)


def atomic_json(path, item):
    _atomic_bytes(Path(path), (json.dumps(item, indent=2, allow_nan=False) + "\n").encode())


def atomic_csv(path, frame):
    _atomic_bytes(Path(path), frame.to_csv(index=False).encode())


def training_frame():
    return load_with_splits().query("split == 'train'").sort_values("pair_id").reset_index(drop=True)


def specification(train, structural_path=STRUCTURAL):
    structural = pd.read_csv(structural_path)
    geometry, confidence = structural_columns(structural)
    columns = geometry + confidence
    if len(columns) != 109:
        raise ValueError("R1 requires the original 109 geometry/confidence summary columns")
    # Actual model fits, including ensemble members; deterministic ridge is not duplicated.
    per_representation = {"ridge": {"inner": 3 * 3, "final": 1},
                          "mlp": {"inner": 3 * 3 * 2, "final": 30}}
    per_outer = len(REPRESENTATIONS) * (9 + 1 + 18 + 30)
    return {"schema": "round2_embedding.1", "canonical_plan": "reports/NEXT_RESEARCH_PLAN.md",
            "population": "all original training rows only; previously examined development evidence",
            "n_rows": len(train), "n_clusters": int(train.cluster_id.nunique()),
            "source_sha256": sha256_file(ROOT / "data/rasmussen_et_al_dataset.csv"),
            "splits_sha256": sha256_file(ROOT / "data/splits.csv"),
            "data_hash": hashlib.sha256(pd.util.hash_pandas_object(train, index=False).values.tobytes()).hexdigest(),
            "outer_fold_hash": hashlib.sha256(nested.outer_folds(train).astype("<i8").tobytes()).hexdigest(),
            "feature_protocol": feature_protocol(), "contact_map_sha256": load_contact_map()["contact_map_sha256"],
            "representations": list(REPRESENTATIONS), "component_order": list(COMPONENT_ORDER),
            "feature_dimensions": {"R0": 860, "R1": 969, **FEATURE_DIMENSIONS},
            "R0": "position-preserving peptide+pseudosequence onehot; matched head/scaling",
            "R1": {"representation": "R0 plus existing 109 geometry/confidence summaries; historical diagnostic",
                   "source_sha256": sha256_file(structural_path), "columns": columns},
            "candidate_grids": {arm: head_candidates(rep, learner) for rep in REPRESENTATIONS
                                for learner in LEARNERS for arm in [f"{rep}_{learner}"]},
            "selection": "three inner peptide-cluster folds; first frozen strongest-regularization candidate within one fold-SE of best mean inner median per-allele Spearman; select independently per outer fold, representation and learner",
            "min_rows_inner": 20,
            "scaling": "StandardScaler fitted on ridge fitting rows or MLP gradient-fitting rows only; no feature reduction is fitted",
            "head": {"ridge": {"solver": "lsqr", "tol": 1e-5, "max_iter": 2000,
                                "alpha": [10000., 100., 1.], "members": 1},
                     "mlp": {"hidden": [64], "l2": [1., .01, .0001], "learning_rate": .001,
                             "batch_size": 256, "max_epochs": 200, "patience": 15,
                             "inner_members": "seed0 stop-fold0 and seed1 stop-fold1 of five fitting-cluster stopping folds",
                             "final_members": "five stopping folds x six seeds0-5 =30",
                             "stopping_metric": "MSE log1p half-life"}},
            "fit_graph": {"per_representation_per_outer": per_representation,
                          "per_outer_actual_model_fits": per_outer,
                          "five_outer_actual_model_fits": per_outer * 5,
                          "selector_extra_fits": 0, "sequence_reference_fits": "separate sequence protocol"},
            "selector_order": list(SELECTOR_ORDER),
            "selector": "selected component inner mean scores only; first frozen procedure within one SE of best; original mlp_log baseline-only first and always eligible; delay finalization until sequence outputs complete",
            "comparisons": {"primary": ["embedding_selector", "sequence_selector"],
                            "all_components": "every declared component vs original mlp_log and sequence_selector",
                            "positional_per_learner": [["R3", "R3_pooled"], ["R5", "R5_shuffled"],
                                                       ["R5_position9x34", "R5_position9x34_pooled"]],
                            "confounded_component_contrasts": [["R3", "R2"], ["R5_position9x34", "R5"]],
                            "matched_head_control": "each representation vs R0 with its same learner",
                            "statistic": "pooled outer-held median per-allele Spearman using fixed original tie/constant rules; per-fold medians separately"},
            "execution": {"device": "CPU", "threads_per_worker": 2, "cloud_spend_usd": 0},
            "gate": "do not execute fits before repaired evaluation and extraction equivalence/cost/cohort gates pass",
            "limitations": ["fixed projected channels and unequal feature dimensions do not perfectly match capacity",
                            "R3 vs R2 changes full-chain/beta2m context as well as positions; position9x34 vs R5 changes channels and central pooling",
                            "R0/R1 matched small heads differ from original sequence architecture/encoding ensemble",
                            "inner two-member MLP screening differs from final30-member recipe",
                            "development results are not independent confirmation"]}


def freeze(out, *, train=None):
    train = training_frame() if train is None else train
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    spec = specification(train)
    digest = config_digest(spec)
    path = out / "protocol.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["protocol_hash"] != digest:
            raise ValueError("frozen embedding protocol changed; document a version amendment")
        return existing
    item = {"protocol_hash": digest, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "specification": spec, "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in (
                Path(__file__), ROOT / "pepstab/round2_embedding.py", ROOT / "pepstab/round2_nested.py",
                ROOT / "pepstab/round2_representations.py")}, "platform": platform.platform()}
    atomic_json(path, item)
    atomic_csv(out / "fold_manifest.csv", train[["pair_id", "cluster_id", "allele"]].assign(outer_fold=nested.outer_folds(train)))
    return item


def _bundle_directories(manifest_path):
    path = Path(manifest_path)
    if path.is_dir():
        return [path], {"path": str(path.resolve())}
    item = json.loads(path.read_text())
    bundles = item.get("feature_bundles")
    if bundles is None:
        raise ValueError("production manifest must declare feature_bundles")
    if item.get("status") != "complete" or item.get("completed_rows") != item.get("expected_rows"):
        raise ValueError("production compact feature manifest is incomplete")
    paths = []
    for bundle in bundles:
        value = bundle if isinstance(bundle, str) else bundle["path"]
        directory = Path(value)
        paths.append(directory if directory.is_absolute() else path.parent / directory)
    return paths, {"path": str(path.resolve()), "sha256": sha256_file(path),
                   "expected_rows": item["expected_rows"], "cohort_sha256": item.get("cohort_sha256")}


def load_compact_features(train, manifest_path):
    """Exact one-to-one pair join, content hashes, fixed map/projection provenance."""
    directories, origin = _bundle_directories(manifest_path)
    matrices = {name: [] for name in FEATURE_DIMENSIONS}
    identities, input_hashes, provenance = [], [], []
    expected_protocol = config_digest(feature_protocol())
    expected_map = load_contact_map()["contact_map_sha256"]
    for directory in directories:
        complete = json.loads((directory / "complete.json").read_text())
        if complete["schema_version"] != FEATURE_SCHEMA:
            raise ValueError("unknown compact feature schema")
        for key in ("array_file", "metadata_file"):
            if Path(complete[key]).name != complete[key]:
                raise ValueError("bundle file must be a basename")
        array_path, metadata_path = directory / complete["array_file"], directory / complete["metadata_file"]
        if sha256_file(array_path) != complete["npz_sha256"] or sha256_file(metadata_path) != complete["metadata_sha256"]:
            raise ValueError("compact feature content hash mismatch")
        info = json.loads(metadata_path.read_text())
        if info["schema_version"] != FEATURE_SCHEMA or info["npz_sha256"] != complete["npz_sha256"]:
            raise ValueError("compact feature manifest mismatch")
        if info["contact_map_sha256"] != expected_map or config_digest(info["feature_protocol"]) != expected_protocol:
            raise ValueError("feature protocol/contact map mismatch")
        rows = info["rows"]
        with np.load(array_path, allow_pickle=False) as arrays:
            ids = arrays["pair_id"].astype(str).tolist()
            hashes = arrays["input_hash"].astype(str).tolist()
            if ids != [str(row["pair_id"]) for row in rows] or hashes != [row["input_hash"] for row in rows]:
                raise ValueError("compact array/manifest row identity mismatch")
            if len(rows) != info["n_rows"]:
                raise ValueError("compact manifest count mismatch")
            identities.extend(ids)
            input_hashes.extend(hashes)
            for name, dimension in FEATURE_DIMENSIONS.items():
                a = arrays[name]
                if a.shape != (len(rows), dimension) or a.dtype != np.float32 or not np.isfinite(a).all():
                    raise ValueError(f"invalid compact feature {name}")
                matrices[name].append(a.copy())
        provenance.append({"path": str(directory.resolve()), "npz_sha256": complete["npz_sha256"],
                           "metadata_sha256": complete["metadata_sha256"]})
    target = train.pair_id.astype(str).tolist()
    if "expected_rows" in origin and origin["expected_rows"] != len(target):
        raise ValueError("production manifest expected_rows differs from training cohort")
    if len(set(identities)) != len(identities) or len(set(target)) != len(target):
        raise ValueError("duplicate compact pair_id")
    if set(identities) != set(target):
        raise ValueError("compact cache must cover exactly the original training cohort")
    order = {pid: i for i, pid in enumerate(identities)}
    gather = [order[pid] for pid in target]
    return {name: np.concatenate(values)[gather] for name, values in matrices.items()}, {
        "production_manifest": origin, "bundles": provenance,
        "pair_ids": target, "input_hashes": [input_hashes[i] for i in gather]}


def load_features(train, manifest_path, *, structural_path=STRUCTURAL):
    features, provenance = load_compact_features(train, manifest_path)
    features["R0"] = build_features(train, "pep_pseudo", "onehot").astype(np.float32)
    structural = pd.read_csv(structural_path)
    if structural[["allele", "peptide"]].duplicated().any():
        raise ValueError("duplicate geometry identity")
    geometry, confidence = structural_columns(structural)
    columns = geometry + confidence
    if len(columns) != 109:
        raise ValueError("changed geometry feature list")
    joined = train[["pair_id", "allele", "peptide"]].merge(
        structural[["allele", "peptide"] + columns], on=["allele", "peptide"], how="left", validate="one_to_one", sort=False)
    if not np.array_equal(joined.pair_id, train.pair_id):
        raise ValueError("geometry join changed row order")
    summary = joined[columns].to_numpy(np.float32)
    if not np.isfinite(summary).all():
        raise ValueError("missing or nonfinite geometry summaries")
    features["R1"] = np.concatenate([features["R0"], summary], axis=1)
    provenance["geometry_sha256"] = sha256_file(structural_path)
    return features, provenance


def selected_inner_score(choice):
    return next(row for row in choice["scores"] if row["candidate"] == choice["candidate"]["id"])


def choose_embedding_selector(component_choices, sequence_choices):
    scores = [{"arm": "mlp_log", **selected_inner_score(sequence_choices["mlp_log"])}]
    scores.extend({"arm": arm, **selected_inner_score(component_choices[arm])} for arm in SELECTOR_ORDER[1:])
    best = max(scores, key=lambda row: row["mean"])
    chosen = next(row for row in scores if row["mean"] >= best["mean"] - best["se"])
    return {"arm": chosen["arm"], "scores": scores,
            "rule": "first frozen procedure within one fold-SE of best selected inner mean; baseline-only first"}


def run_component(train, features, fit, held, fold, rep, learner, arm_directory,
                  cache_directory, *, max_epochs=200, min_rows=20):
    arm_directory = Path(arm_directory)
    arm_directory.mkdir(parents=True, exist_ok=True)
    ancestry = Ancestry()
    head = FeatureHead(train, features, ancestry, cache_dir=cache_directory, max_epochs=max_epochs)
    context = f"outer/{fold}/{rep}_{learner}"
    choice = select(fit, head_candidates(rep, learner), head, min_rows=min_rows,
                    ancestry=ancestry, context=context)
    predictions = head.predict(fit, unlabelled(held), choice.candidate, context + "/outer_refit", final=True)
    ancestry.add(role="outer_selected_prediction", fit=fit, held=held, selection=fit,
                 candidate=choice.candidate["id"], context=context)
    audit = ancestry.audit(train, held)
    table = held[["pair_id", "cluster_id", "allele"]].copy()
    table["outer_fold"], table["arm"] = fold, f"{rep}_{learner}"
    table["y_pred"], table["selected_candidate"] = predictions, choice.candidate["id"]
    atomic_json(arm_directory / "selection.json", asdict(choice))
    atomic_json(arm_directory / "ancestry.json", ancestry.as_dict())
    atomic_csv(arm_directory / "predictions.csv", table)
    seed_hash = None
    if head.last_by_seed:
        seed_tables = []
        for seed, values in head.last_by_seed.items():
            seed_tables.append(table.assign(seed=seed, y_pred=values))
        atomic_csv(arm_directory / "seed_predictions.csv", pd.concat(seed_tables))
        seed_hash = sha256_file(arm_directory / "seed_predictions.csv")
    atomic_csv(arm_directory / "members.csv", pd.DataFrame(head.members))
    atomic_json(arm_directory / "completed.json", {"audit": audit, "n_predictions": len(held),
                "actual_model_fits": len(head.members), "sum_member_fit_seconds": sum(m["fit_seconds"] for m in head.members),
                "prediction_sha256": sha256_file(arm_directory / "predictions.csv"),
                "seed_sha256": seed_hash})
    return asdict(choice), table


def finalize_selector(out, fold, sequence_directory=SEQUENCE_OUT):
    fold_path = Path(out) / f"fold_{fold}"
    component_choices, tables, ancestries, seed_tables = {}, [], [], []
    for arm in COMPONENT_ORDER:
        directory = fold_path / arm
        if not (directory / "completed.json").exists():
            return {"status": "waiting", "reason": f"component {arm} incomplete"}
        complete = json.loads((directory / "completed.json").read_text())
        if sha256_file(directory / "predictions.csv") != complete["prediction_sha256"]:
            raise ValueError("saved component prediction hash mismatch")
        component_choices[arm] = json.loads((directory / "selection.json").read_text())
        tables.append(pd.read_csv(directory / "predictions.csv"))
        ancestries.append(json.loads((directory / "ancestry.json").read_text()))
        if arm.endswith("_mlp"):
            if sha256_file(directory / "seed_predictions.csv") != complete.get("seed_sha256"):
                raise ValueError("saved component seed prediction hash mismatch")
            seed_tables.append(pd.read_csv(directory / "seed_predictions.csv"))
    atomic_csv(fold_path / "component_predictions.csv", pd.concat(tables))
    sequence_fold = Path(sequence_directory) / f"fold_{fold}"
    if not (sequence_fold / "completed.json").exists():
        status = {"status": "waiting", "reason": "sequence fold incomplete; baseline-only must remain available"}
        atomic_json(fold_path / "selector_pending.json", status)
        return status
    sequence_completed = json.loads((sequence_fold / "completed.json").read_text())
    if sha256_file(sequence_fold / "predictions.csv") != sequence_completed["prediction_sha256"]:
        raise ValueError("sequence prediction hash mismatch")
    sequence_choices = json.loads((sequence_fold / "selections.json").read_text())
    sequence_ancestry = json.loads((sequence_fold / "ancestry.json").read_text())
    selector = choose_embedding_selector(component_choices, sequence_choices)
    sequence_predictions = pd.read_csv(sequence_fold / "predictions.csv")
    train = training_frame()
    held = train[nested.outer_folds(train) == fold]
    embedding_protocol = json.loads((Path(out) / "protocol.json").read_text())
    sequence_protocol = json.loads((Path(sequence_directory) / "protocol.json").read_text())
    if sequence_completed["protocol_hash"] != sequence_protocol["protocol_hash"]:
        raise ValueError("sequence completion protocol mismatch")
    emb_spec, seq_spec = embedding_protocol["specification"], sequence_protocol["specification"]
    for key in ("source_sha256", "splits_sha256", "data_hash"):
        if emb_spec[key] != seq_spec[key]:
            raise ValueError("sequence and embedding assessment populations differ")
    if emb_spec["outer_fold_hash"] != seq_spec["fold_hash"]:
        raise ValueError("sequence and embedding outer folds differ")
    for table in tables + [sequence_predictions[sequence_predictions.arm == "mlp_log"],
                            sequence_predictions[sequence_predictions.arm == "sequence_selector"]]:
        align_predictions(held, table)
    selected = (sequence_predictions[sequence_predictions.arm == "mlp_log"] if selector["arm"] == "mlp_log"
                else next(table for table in tables if table.arm.iloc[0] == selector["arm"])).copy()
    selected["arm"] = "embedding_selector"
    ancestry = Ancestry()
    for item in ancestries + [sequence_ancestry]:
        ancestry.rowsets.update(item["rowsets"])
        ancestry.records.extend(item["records"])
    ancestry.add(role="embedding_selector_prediction", fit=train[nested.outer_folds(train) != fold],
                 held=held, selection=train[nested.outer_folds(train) != fold],
                 candidate=selector["arm"], context=f"outer/{fold}/embedding_selector")
    atomic_json(fold_path / "selector.json", selector)
    atomic_json(fold_path / "ancestry.json", ancestry.as_dict())
    baseline_tables = sequence_predictions[sequence_predictions.arm.isin(["mlp_log", "sequence_selector"])]
    atomic_csv(fold_path / "predictions.csv", pd.concat(tables + [selected, baseline_tables]))
    atomic_csv(fold_path / "seed_predictions.csv", pd.concat(seed_tables))
    atomic_json(fold_path / "completed.json", {"status": "complete", "audit": ancestry.audit(train, held),
                "protocol_hash": json.loads((Path(out) / "protocol.json").read_text())["protocol_hash"],
                "sequence_protocol_hash": sequence_completed["protocol_hash"],
                "sequence_prediction_sha256": sequence_completed["prediction_sha256"],
                "prediction_sha256": sha256_file(fold_path / "predictions.csv"),
                "seed_sha256": sha256_file(fold_path / "seed_predictions.csv")})
    (fold_path / "selector_pending.json").unlink(missing_ok=True)
    return {"status": "complete", "selected_arm": selector["arm"]}


def run_fold(out, fold, manifest_path, *, representation=None, learner=None,
             sequence_directory=SEQUENCE_OUT):
    train = training_frame()
    out = Path(out)
    protocol = json.loads((out / "protocol.json").read_text())
    if protocol["protocol_hash"] != config_digest(specification(train)):
        raise ValueError("frozen data/feature protocol changed")
    for relative, expected in protocol["code_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError(f"frozen implementation changed: {relative}")
    features, provenance = load_features(train, manifest_path)
    fold_path = out / f"fold_{fold}"
    fold_path.mkdir(parents=True, exist_ok=True)
    identity = {"protocol_hash": protocol["protocol_hash"], "feature_provenance": provenance,
                "feature_hashes": {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in features.items()}}
    identity_path = fold_path / "feature_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("resuming fold with changed compact features")
    atomic_json(identity_path, identity)
    outer = nested.outer_folds(train)
    fit, held = train[outer != fold], train[outer == fold]
    for rep in ([representation] if representation else REPRESENTATIONS):
        for head in ([learner] if learner else LEARNERS):
            arm = f"{rep}_{head}"
            directory = fold_path / arm
            if (directory / "completed.json").exists():
                continue
            started = time.perf_counter()
            run_component(train, {rep: features[rep]}, fit, held, fold, rep, head, directory,
                          ROOT / "runs/round2_embedding" / protocol["protocol_hash"] / f"fold_{fold}" / arm)
            print(f"fold {fold}: {arm} complete, {time.perf_counter()-started:.1f}s", flush=True)
    print(json.dumps(finalize_selector(out, fold, sequence_directory)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--production-manifest", type=Path)
    parser.add_argument("--representation", choices=REPRESENTATIONS)
    parser.add_argument("--learner", choices=LEARNERS)
    parser.add_argument("--sequence-directory", type=Path, default=SEQUENCE_OUT)
    parser.add_argument("--finalize-selector", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        item = freeze(args.out_dir)
        print(json.dumps({"protocol_hash": item["protocol_hash"], "fit_graph": item["specification"]["fit_graph"]}, indent=2))
    if args.fold is not None:
        if args.finalize_selector:
            print(json.dumps(finalize_selector(args.out_dir, args.fold, args.sequence_directory)))
        else:
            if not args.production_manifest:
                parser.error("--production-manifest required for fitting")
            with threadpool_limits(limits=2):
                run_fold(args.out_dir, args.fold, args.production_manifest, representation=args.representation,
                         learner=args.learner, sequence_directory=args.sequence_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
