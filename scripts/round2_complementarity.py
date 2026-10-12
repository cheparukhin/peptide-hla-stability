"""Freeze and explicitly execute clean R2/R4 complementarity; local CPU only."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pepstab import nested
from pepstab.boltz_embeddings import _atomic_bytes, sha256_file
from pepstab.round2_complementarity import (
    BLEND_WEIGHTS, CORRECTION_WEIGHTS, REPRESENTATIONS, RESIDUAL_ALPHA,
    FixedOriginalBase, assess_outer, fitting_graph,
)
from pepstab.round2_embedding import FeatureHead, head_candidates
from pepstab.round2_nested import Ancestry, config_digest
from pepstab.round2_sequence import EnsemblePredictor, candidates
from scripts.round2_embedding_screen import load_compact_features
from scripts.round2_results import load_completed_study, population_contract, training_frame

OUT = ROOT / "reports/further_research/round2/complementarity"
SEQUENCE = ROOT / "reports/further_research/round2/sequence"
EMBEDDING = ROOT / "reports/further_research/round2/embedding"


def atomic_json(path, item):
    _atomic_bytes(Path(path), (json.dumps(item, indent=2, allow_nan=False) + "\n").encode())


def atomic_csv(path, table):
    _atomic_bytes(Path(path), table.to_csv(index=False).encode())


def specification(train, sequence_directory=SEQUENCE, embedding_directory=EMBEDDING):
    sequence = json.loads((Path(sequence_directory) / "protocol.json").read_text())
    embedding = json.loads((Path(embedding_directory) / "protocol.json").read_text())
    for item in (sequence, embedding):
        if item["protocol_hash"] != config_digest(item["specification"]):
            raise ValueError("source frozen protocol hash mismatch")
    return {"schema": "round2_complementarity.1", "canonical_plan": "reports/NEXT_RESEARCH_PLAN.md sections3,5",
            **population_contract(train), "population": "original training only; development evidence",
            "sequence_protocol_hash": sequence["protocol_hash"], "embedding_protocol_hash": embedding["protocol_hash"],
            "representations": list(REPRESENTATIONS),
            "baseline": {"candidate": candidates()["mlp_log"][0],
                         "recipe": "fixed original architecture/encodings; final=True everywhere; 5stoppingfolds x2encodings x3seeds=30 members",
                         "outer_application": "audited exact original30-member outer predictions; no renormalisation"},
            "embedding": {rep: head_candidates(rep, "ridge") for rep in REPRESENTATIONS},
            "blending": {"simplex_weights": list(BLEND_WEIGHTS),
                         "inputs": "fresh original OOF and nested-selected ridge OOF inside outer fitting partition; alpha selection excludes each predicted meta fold",
                         "selection": "three meta-inner cluster panels; first weight within one fold-SE of best mean inner median Spearman; zero first"},
            "residual": {"alpha": RESIDUAL_ALPHA, "correction_weights": list(CORRECTION_WEIGHTS),
                         "targets": "inner-fit labels minus fresh3fold fixed-original cleanOOF inside that innerfit, excluding metaheld from gradient/stopping/selection; final targets from outer-fit cleanOOF",
                         "scaling": "StandardScaler fitted on residual fitting rows only",
                         "selection": "three meta-inner cluster panels; first correction weight within one fold-SE; zero first"},
            "min_rows_inner": 20, "n_meta_inner_folds": 3,
            "no_other_outer_predictions": "other outer folds are never stacking inputs; assessed outer artifacts used only in final application",
            "fit_graph": fitting_graph(), "execution": {"device": "CPU", "threads": 2, "cloud_cost_usd": 0},
            "comparisons": "both blend/residual procedures and embedding-alone versus original and competitive sequence; standalone gain and complementarity gain reported separately",
            "limitations": "conditional development assessment; inner models fit fewer rows than outer application; no independent confirmation or full channel claim"}


def freeze(out, sequence_directory=SEQUENCE, embedding_directory=EMBEDDING):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    train = training_frame()
    spec = specification(train, sequence_directory, embedding_directory)
    digest = config_digest(spec)
    path = out / "protocol.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["protocol_hash"] != digest:
            raise ValueError("frozen complementarity protocol changed; document a scope amendment")
        return previous
    item = {"protocol_hash": digest, "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "specification": spec, "code_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in (
                Path(__file__), ROOT / "pepstab/round2_complementarity.py", ROOT / "pepstab/round2_sequence.py",
                ROOT / "pepstab/round2_embedding.py", ROOT / "pepstab/round2_nested.py")}}
    atomic_json(path, item)
    atomic_csv(out / "fold_manifest.csv", train[["pair_id", "cluster_id", "allele"]].assign(outer_fold=nested.outer_folds(train)))
    return item


def _validate_fixed_original_members(sequence_directory, fold):
    members = pd.read_csv(Path(sequence_directory) / f"fold_{fold}/members.csv")
    context = f"outer/{fold}/mlp_log/outer_refit"
    members = members[members.context.eq(context)]
    expected = {(stop, encoding, seed) for stop in range(5) for encoding in ("onehot", "blosum") for seed in range(3)}
    observed = set(members[["stop_fold", "encoding", "seed"]].itertuples(index=False, name=None))
    if len(members) != 30 or observed != expected or not members.final_recipe.eq(True).all():
        raise ValueError("outer baseline is not the fixed original30-member recipe")
    for member in members.itertuples():
        cfg = ast.literal_eval(member.config) if isinstance(member.config, str) else member.config
        expected_l2 = 1e-5 if member.encoding == "onehot" else 1e-3
        if (cfg["hidden"] != [256, 64] or cfg["l2"] != expected_l2 or cfg["max_epochs"] != 300
                or cfg["patience"] != 15 or cfg["lr"] != .001 or cfg["seed"] != member.seed):
            raise ValueError("outer baseline member configuration differs from fixed original")


def run_fold(out, fold, manifest, sequence_directory=SEQUENCE, embedding_directory=EMBEDDING):
    train = training_frame()
    out = Path(out)
    protocol = json.loads((out / "protocol.json").read_text())
    if protocol["protocol_hash"] != config_digest(specification(train, sequence_directory, embedding_directory)):
        raise ValueError("complementarity protocol or population changed")
    for path, digest in protocol["code_sha256"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError("complementarity code changed after freeze")
    # Full references are checked only at execution, after freeze; never screened.
    sequence = load_completed_study(sequence_directory, "sequence", train)
    embedding = load_completed_study(embedding_directory, "embedding", train)
    if (embedding["completions"][fold]["sequence_protocol_hash"] != sequence["protocol"]["protocol_hash"]
            or embedding["completions"][fold]["sequence_prediction_sha256"] != sequence["completions"][fold]["prediction_sha256"]):
        raise ValueError("outer embedding reference used different sequence provenance")
    if sequence["saved_choices"][fold]["mlp_log"]["candidate"] != candidates()["mlp_log"][0]:
        raise ValueError("outer original baseline recipe is not fixed")
    for name in ("mlp_log", "sequence_selector"):
        if not np.allclose(sequence["predictions"][name], embedding["predictions"][name], rtol=0, atol=1e-12):
            raise ValueError("audited outer sequence references differ")
    _validate_fixed_original_members(sequence_directory, fold)
    compact, feature_identity = load_compact_features(train, manifest)
    features = {rep: compact[rep] for rep in REPRESENTATIONS}
    directory = out / f"fold_{fold}"
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"protocol_hash": protocol["protocol_hash"], "feature_identity": feature_identity,
                "feature_hashes": {rep: hashlib.sha256(array.tobytes()).hexdigest() for rep, array in features.items()},
                "reference_hashes": {**sequence["sources"], **embedding["sources"]}}
    identity_path = directory / "input_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("resuming complementarity with changed inputs")
    atomic_json(identity_path, identity)
    if (directory / "completed.json").exists():
        marker = json.loads((directory / "completed.json").read_text())
        if marker["prediction_sha256"] != sha256_file(directory / "predictions.csv"):
            raise ValueError("completed complementarity prediction hash mismatch")
        for name, digest in marker["artifact_sha256"].items():
            if sha256_file(directory / name) != digest:
                raise ValueError("completed complementarity artifact hash mismatch")
        return marker
    folds = nested.outer_folds(train)
    fit, held = train[folds != fold], train[folds == fold]
    ancestry = Ancestry()
    # Reuse only this outer fold's reference ancestry; other fold records forbid
    # an audit and their predictions are never supplied to the meta procedure.
    for source in (Path(sequence_directory), Path(embedding_directory)):
        item = json.loads((source / f"fold_{fold}/ancestry.json").read_text())
        ancestry.rowsets.update(item["rowsets"])
        ancestry.records.extend(item["records"])
    cache = ROOT / "runs/round2_complementarity" / protocol["protocol_hash"] / f"fold_{fold}"
    last_update = [0.]
    def checkpoint(member):
        with (directory / "members.jsonl").open("a") as handle:
            handle.write(json.dumps(member, allow_nan=False) + "\n")
        if time.perf_counter() - last_update[0] > 45:
            last_update[0] = time.perf_counter()
            print(f"fold {fold}: clean original member {member['context']}, seed{member['seed']}", flush=True)
    ensemble = EnsemblePredictor(train, ancestry, cache_dir=cache / "original", callback=checkpoint)
    head = FeatureHead(train, features, ancestry, cache_dir=cache / "ridge")
    started = time.perf_counter()
    baseline = sequence["predictions"]["mlp_log"][folds == fold]
    alternatives = {rep: embedding["predictions"][f"{rep}_ridge"][folds == fold] for rep in REPRESENTATIONS}
    values, selections, meta = assess_outer(fit, held, FixedOriginalBase(ensemble), head, ancestry,
                                           baseline, alternatives)
    values["sequence_selector"] = sequence["predictions"]["sequence_selector"][folds == fold]
    tables = [held[["pair_id", "cluster_id", "allele"]].assign(outer_fold=fold, arm=name, y_pred=prediction)
              for name, prediction in values.items()]
    atomic_csv(directory / "predictions.csv", pd.concat(tables))
    atomic_json(directory / "selections.json", selections)
    atomic_json(directory / "residual_target_provenance.json", meta["residual_target_provenance"])
    oof = fit[["pair_id", "cluster_id", "allele"]].assign(meta_fold=nested.inner_folds(fit, 3),
                                                       original_clean_oof=meta["base_oof"])
    for rep in REPRESENTATIONS:
        oof[f"{rep}_nested_selected_oof"] = meta["embedding_oof"][rep]
        oof[f"{rep}_nested_residual_oof"] = meta["residual_oof"][rep]
    atomic_csv(directory / "meta_inputs.csv", oof)
    atomic_json(directory / "ancestry.json", ancestry.as_dict())
    atomic_csv(directory / "members.csv", pd.DataFrame(ensemble.members + head.members))
    marker = {"schema": "round2_complementarity_completion.1", "protocol_hash": protocol["protocol_hash"],
              "completed_at_utc": datetime.now(timezone.utc).isoformat(), "n_rows": len(held),
              "audit": ancestry.audit(train, held), "logical_model_fits_including_safe_cache_reuse": len(ensemble.members) + len(head.members),
              "invocation_seconds": time.perf_counter()-started, "cloud_cost_usd": 0,
              "prediction_sha256": sha256_file(directory / "predictions.csv"),
              "artifact_sha256": {name: sha256_file(directory / name) for name in (
                  "selections.json", "residual_target_provenance.json", "meta_inputs.csv", "ancestry.json", "members.csv", "input_identity.json")}}
    atomic_json(directory / "completed.json", marker)
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--sequence-directory", type=Path, default=SEQUENCE)
    parser.add_argument("--embedding-directory", type=Path, default=EMBEDDING)
    parser.add_argument("--production-manifest", type=Path)
    args = parser.parse_args()
    if args.freeze:
        item = freeze(args.out_dir, args.sequence_directory, args.embedding_directory)
        print(json.dumps({"protocol_hash": item["protocol_hash"], "fit_graph": fitting_graph()}, indent=2))
    if args.fold is not None:
        if not args.production_manifest:
            parser.error("--production-manifest required for explicit execution")
        with threadpool_limits(limits=2):
            result = run_fold(args.out_dir, args.fold, args.production_manifest,
                              args.sequence_directory, args.embedding_directory)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
