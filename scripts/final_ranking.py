"""Freeze, audit, validate, and execute the bounded matched ranking follow-up.

Freeze runs no real fits. Validation needs the separate root-owned validation
authorization; scientific execution needs the complete gate and validation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from importlib.metadata import version
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pepstab import evaluation, nested
from pepstab.boltz_embeddings import sha256_file
from pepstab.features import build_features
from pepstab.final_ranking import (REPRESENTATIONS, RankedHead, array_hash, atomic_json,
    audit_cached_call, candidates, canonical_cache_key, deploy_choice, member_config,
    one_se, ordered_rows, FINAL_AXES, INNER_AXES)
from pepstab.phase_a import RankingMLPRegressor
from pepstab.round2_embedding import head_candidates
from pepstab.round2_nested import Ancestry, align_predictions, config_digest, select, unlabelled
from scripts.round2_embedding_screen import load_compact_features, training_frame, atomic_csv

OUT = ROOT / "reports/finalization/ranking"
BASE = ROOT / "reports/further_research/round2/embedding"
FEATURES = ROOT / "reports/further_research/round2/extractor/full_training/feature_bundles.json"
CODE = ("pepstab/final_ranking.py", "scripts/final_ranking.py", "scripts/final_ranking_report.py",
        "scripts/final_ranking_pipeline.py", "pepstab/mlp.py", "pepstab/phase_a.py",
        "pepstab/round2_nested.py", "pepstab/round2_embedding.py", "pepstab/nested.py",
        "pepstab/evaluation.py", "pepstab/round2_reporting.py", "pepstab/splits.py",
        "pepstab/features.py", "pepstab/data.py", "pepstab/round2_representations.py",
        "pepstab/boltz_embeddings.py", "scripts/round2_embedding_screen.py",
        "tests/test_final_ranking.py", "tests/test_final_ranking_pipeline.py")


def runtime_identity():
    with threadpool_limits(limits=2):
        libraries=[{key:item.get(key) for key in ("user_api","internal_api","prefix","version",
            "num_threads","threading_layer","architecture")} for item in threadpool_info()]
    return dict(python=platform.python_version(),platform=platform.platform(),machine=platform.machine(),
        packages={name:version(name) for name in ("numpy","pandas","scipy","scikit-learn","threadpoolctl")},
        numerical_libraries=libraries)


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def load_features(train):
    bundle, provenance = load_compact_features(train, FEATURES)
    features = dict(R0=build_features(train, "pep_pseudo", "onehot").astype(np.float32), R5=bundle["R5"])
    for rep, width in (("R0", 860), ("R5", 1024)):
        if features[rep].shape != (len(train), width):
            raise ValueError("frozen feature dimensions changed")
    return features, provenance


def cached_call(train, features, fold, rep, fit, held, candidate, context, *, final=False):
    original = json.loads((BASE / "protocol.json").read_text())
    key = canonical_cache_key(fit, held, candidate, context, array_hash(features[rep]),
                              original["code_sha256"]["pepstab/round2_embedding.py"], final=final)
    path = ROOT / "runs/round2_embedding" / original["protocol_hash"] / f"fold_{fold}" / f"{rep}_mlp" / f"{key}.json"
    item = json.loads(path.read_text())
    audit_cached_call(item, fit, held, candidate, context, train, final=final)
    if any(m["n_features"] != features[rep].shape[1] for m in item["members"]):
        raise ValueError("canonical feature width changed")
    return path, item


def read_baseline(train, features, fold, rep):
    """Audited saved raw inner predictions and independently replayed selection."""
    outer = nested.outer_folds(train)
    fit, held = train.loc[outer != fold], train.loc[outer == fold]
    directory = BASE / f"fold_{fold}" / f"{rep}_mlp"
    choice = json.loads((directory / "selection.json").read_text())
    complete = json.loads((directory / "completed.json").read_text())
    table = pd.read_csv(directory / "predictions.csv")
    if sha256_file(directory / "predictions.csv") != complete["prediction_sha256"]:
        raise ValueError("canonical prediction content hash changed")
    if sha256_file(directory / "seed_predictions.csv") != complete["seed_sha256"]:
        raise ValueError("canonical seed content hash changed")
    if complete["actual_model_fits"] != 48 or not complete["audit"]["outer_cluster_exclusion"]:
        raise ValueError("canonical baseline is incomplete")
    pred = align_predictions(held, table)
    if not table.allele.tolist() == held.allele.tolist() or not table.outer_fold.eq(fold).all():
        raise ValueError("canonical held metadata changed")
    ancestry = Ancestry()
    saved = json.loads((directory / "ancestry.json").read_text())
    ancestry.records, ancestry.rowsets = saved["records"], saved["rowsets"]
    ancestry.audit(train, held)
    raw, sources, replay = [], [], []
    folds = nested.inner_folds(fit, 3)
    canonical_records = []
    for candidate in head_candidates(rep, "mlp"):
        values = []
        for inner in range(3):
            inner_fit, inner_held = fit.loc[folds != inner], fit.loc[folds == inner]
            context = f"outer/{fold}/{rep}_mlp/select/{candidate['id']}/{inner}"
            path, item = cached_call(train, features, fold, rep, inner_fit, inner_held, candidate, context)
            panel = evaluation.eligible_alleles(inner_held.allele, inner_held.y_log1p, min_rows=20)
            values.append(evaluation.score("inner", inner_held.allele, inner_held.y_log1p,
                                            np.asarray(item["predictions"]), panel).median_spearman)
            raw.append(inner_held[["pair_id", "cluster_id", "allele"]].assign(
                outer_fold=fold, inner_fold=inner, representation=rep,
                candidate=candidate["id"], y_pred=item["predictions"]))
            sources.append(dict(path=relative(path), sha256=sha256_file(path), context=context,
                                logical_members=2, member_axes=list(INNER_AXES)))
            canonical_records.extend(item["ancestry"])
        replay.append(dict(candidate=candidate["id"], fold_scores=list(map(float, values)),
                           mean=float(np.mean(values)), se=float(np.std(values, ddof=1)/np.sqrt(3))))
    if choice["candidate"]["id"] != one_se(replay)["candidate"] or choice["scores"] != replay:
        raise ValueError("canonical MSE inner selection does not replay exactly")
    context = f"outer/{fold}/{rep}_mlp/outer_refit"
    path, item = cached_call(train, features, fold, rep, fit, held, choice["candidate"], context, final=True)
    # Historical CSV uses float32's shortest roundtrip representation. Restore
    # float32 before requiring exact agreement with the canonical cached vector.
    np.testing.assert_array_equal(pred.astype(np.float32), np.asarray(item["predictions"],dtype=np.float32))
    seeds = pd.read_csv(directory / "seed_predictions.csv")
    for seed in range(6):
        rows = seeds.loc[seeds.seed == seed]
        np.testing.assert_array_equal(align_predictions(held,rows).astype(np.float32),
                                      np.asarray(item["by_seed"][str(seed)],dtype=np.float32))
    sources.append(dict(path=relative(path), sha256=sha256_file(path), context=context,
                        logical_members=30, member_axes=list(FINAL_AXES)))
    canonical_records.extend(item["ancestry"])
    if [r for r in ancestry.records if r["role"] == "mlp_head"] != canonical_records:
        raise ValueError("canonical published ancestry differs from cached members")
    expected_wrappers = []
    for candidate in head_candidates(rep, "mlp"):
        for inner in range(3):
            expected_wrappers.append(dict(role="inner_selection_prediction", fit_rows=ancestry.rows(fit.loc[folds != inner]),
                prediction_rows=ancestry.rows(fit.loc[folds == inner]), candidate=candidate["id"],
                context=f"outer/{fold}/{rep}_mlp", inner_fold=inner))
    expected_wrappers.append(dict(role="outer_selected_prediction", fit_rows=ancestry.rows(fit),
        prediction_rows=ancestry.rows(held), selection_rows=ancestry.rows(fit),
        candidate=choice["candidate"]["id"], context=f"outer/{fold}/{rep}_mlp"))
    if [r for r in ancestry.records if r["role"] != "mlp_head"] != expected_wrappers:
        raise ValueError("canonical selection ancestry changed")
    for filename in ("selection.json", "ancestry.json", "predictions.csv", "seed_predictions.csv", "members.csv", "completed.json"):
        sources.append(dict(path=relative(directory / filename), sha256=sha256_file(directory / filename)))
    return choice, table, pd.concat(raw, ignore_index=True), sources


def fit_graph(train):
    graph = []
    outer = nested.outer_folds(train)
    ancestry = Ancestry()
    for rep in REPRESENTATIONS:
        for fold in range(5):
            fit, held = train.loc[outer != fold], train.loc[outer == fold]
            inner_folds = nested.inner_folds(fit, 3)
            for candidate in candidates(rep):
                for inner in range(3):
                    inner_fit, inner_held = fit.loc[inner_folds != inner], fit.loc[inner_folds == inner]
                    stop_folds = nested.inner_folds(inner_fit, 5)
                    for stop, seed in INNER_AXES:
                        graph.append(dict(representation=rep, outer_fold=fold, inner_fold=inner,
                            candidate=candidate["id"], stop_fold=stop, seed=seed, phase="inner",
                            gradient=ancestry.rows(inner_fit.loc[stop_folds != stop]),
                            stopping=ancestry.rows(inner_fit.loc[stop_folds == stop]),
                            prediction=ancestry.rows(inner_held), candidate_grid_frozen=True))
            stop_folds = nested.inner_folds(fit, 5)
            for stop, seed in FINAL_AXES:
                graph.append(dict(representation=rep, outer_fold=fold, candidate="inner-selected positive ranking candidate",
                    stop_fold=stop, seed=seed, phase="final", gradient=ancestry.rows(fit.loc[stop_folds != stop]),
                    stopping=ancestry.rows(fit.loc[stop_folds == stop]), prediction=ancestry.rows(held), selection=ancestry.rows(fit)))
    return dict(fits=graph, rowsets=ancestry.rowsets)


def freeze(out):
    out = Path(out)
    if (out / "protocol.json").exists():
        return verify_freeze(out)
    acceptance=json.loads((out/"synthetic_acceptance.json").read_text())
    if acceptance.get("status") != "passed" or not acceptance.get("lambda0_bit_exact"):
        raise ValueError("synthetic lambda0 acceptance must pass before freeze")
    for path,digest in acceptance["source_hashes"].items():
        if sha256_file(ROOT/path) != digest:
            raise ValueError("source changed after synthetic acceptance")
    train = training_frame()
    features, provenance = load_features(train)
    original = json.loads((BASE / "protocol.json").read_text())
    spec = original["specification"]
    for path, expected in original["code_sha256"].items():
        if sha256_file(ROOT / path) != expected:
            raise ValueError("historical recipe source changed")
    if config_digest(spec) != original["protocol_hash"]:
        raise ValueError("historical protocol hash changed")
    identity = dict(source_sha256=sha256_file(ROOT / "data/rasmussen_et_al_dataset.csv"),
        splits_sha256=sha256_file(ROOT / "data/splits.csv"),
        data_hash=array_hash(pd.util.hash_pandas_object(train, index=False).values),
        outer_fold_hash=array_hash(nested.outer_folds(train).astype("<i8")))
    if any(spec[key] != value for key, value in identity.items()):
        raise ValueError("historical source/cohort/folds changed")
    expected_recipe = dict(hidden=[64], l2=[1., .01, .0001], learning_rate=.001, batch_size=256,
        max_epochs=200, patience=15, inner_members="seed0 stop-fold0 and seed1 stop-fold1 of five fitting-cluster stopping folds",
        final_members="five stopping folds x six seeds0-5 =30", stopping_metric="MSE log1p half-life")
    if spec["head"]["mlp"] != expected_recipe:
        raise ValueError("historical MSE recipe changed")
    all_sources, raw, timings = [], [], []
    for rep in REPRESENTATIONS:
        for fold in range(5):
            saved_identity = json.loads((BASE / f"fold_{fold}/feature_identity.json").read_text())
            for name in REPRESENTATIONS:
                if saved_identity["feature_hashes"][name] != array_hash(features[name]):
                    raise ValueError("historical feature values changed")
            _, _, inner, sources = read_baseline(train, features, fold, rep)
            raw.append(inner)
            all_sources.extend(sources)
            members = pd.read_csv(BASE / f"fold_{fold}/{rep}_mlp/members.csv")
            timings.extend(members.fit_seconds.tolist())
    graph = fit_graph(train)
    if len(graph["fits"]) != 840 or sum(s.get("logical_members", 0) for s in all_sources) != 480:
        raise ValueError("bounded fit accounting changed")
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "fit_graph.json", graph)
    atomic_csv(out / "reused_inner_predictions.csv", pd.concat(raw, ignore_index=True))
    sources = {relative(ROOT / path): sha256_file(ROOT / path) for path in CODE}
    sources.update({s["path"]: s["sha256"] for s in all_sources})
    sources[relative(BASE / "protocol.json")] = sha256_file(BASE / "protocol.json")
    sources[relative(FEATURES)] = sha256_file(FEATURES)
    sources[relative(out/"synthetic_acceptance.json")] = sha256_file(out/"synthetic_acceptance.json")
    for name in ("data/rasmussen_et_al_dataset.csv", "data/splits.csv"):
        sources[name] = sha256_file(ROOT / name)
    for fold in range(5):
        path = BASE / f"fold_{fold}/feature_identity.json"
        sources[relative(path)] = sha256_file(path)
    for bundle in provenance["bundles"]:
        directory = Path(bundle["path"])
        complete = json.loads((directory / "complete.json").read_text())
        for path in (directory / "complete.json", directory / complete["array_file"], directory / complete["metadata_file"]):
            sources[relative(path)] = sha256_file(path)
    protocol = dict(schema="final_ranking.1", population="all original training rows only; examined development evidence",
        n_rows=len(train), n_clusters=int(train.cluster_id.nunique()), identity=identity,
        representations={"R0": "position-preserving peptide+pseudosequence onehot860", "R5": "Boltz contact-pooled1024"},
        feature_hashes={rep: array_hash(x) for rep, x in features.items()},
        candidates={rep: candidates(rep) for rep in REPRESENTATIONS}, recipe=expected_recipe,
        objective="MSE(log1p half-life)+lambda within-allele RankNet; natural batches; target ties omitted; equal mass per contributing allele per batch; L2 weights only",
        stopping="MSE only", positive_selection="existing mean of three inner-fold median per-allele Spearman, min20; first within best fold-SE; small lambda then strong L2",
        deployment_selection="one fold-SE on the independently selected procedures' inner scores, MSE first; no extra fits",
        fit_counts={"scientific_new":840, "logical_MSE_reused":480, "real_lambda0_validation":2,
                    "inner_new_per_rep_outer":54, "final_new_per_rep_outer":30},
        validation={"recipe":"one real lambda0 canonical full200-epoch recipe member per representation: outer0 / inner0 / L2.01 / stop0 / seed0; saved raw member comparison; exact synthetic equivalence first",
                    "rtol":2e-5,"atol":2e-5,"require_same_best_and_stopping_epochs":True},
        contrasts={"primary":"R5_ranked-R5_mse", "secondary":["R0_ranked-R0_mse", "R5_mse-R0_mse", "R5_ranked-R0_ranked", "(R5_ranked-R5_mse)-(R0_ranked-R0_mse)"],
                   "deployment":["R0_deploy-R0_mse", "R5_deploy-R5_mse"]},
        reporting="pooled median per-allele Spearman min20 and min50; 2000 shared paired whole-cluster bootstrap draws; saved selection table; MSE and MAE on anchored log1p scale; no affine mapping",
        uncertainty="conditional on fitted predictions and inner selection; no independent confirmation; fixed two-member screening differs from final30 members",
        limits=["MSE anchors scale; no pure ranking claim", "R0/R5 widths and parameter counts differ", "no transfer extension, expanded lambda, stopping change, repeats, or automatic scope expansion"],
        runtime=runtime_identity(),
        prior_runtime_compatibility={"platform_matches":original["platform"]==platform.platform(),
            "historical_platform":original["platform"],"package_versions":"not recorded by historical embedding screen; exact sources/features and real lambda0 member comparison provide bounded compatibility evidence"},
        artifact_distribution="commit/distribute all finalized member NPZ, receipts and CSV/JSON; NPZ are normally ignored, so explicitly allowlist finalized reports/finalization/ranking/** NPZ; never include partial arrays or launch-only fits",
        execution={"cloud_spend_usd":0, "max_workers":2, "threads_per_worker":2,
                   "single_owner":"exclusive durable controller receipt; interrupted unreceipted members require explicit recovery review"},
        source_hashes=sources, fit_graph_sha256=sha256_file(out / "fit_graph.json"),
        reused_inner_predictions_sha256=sha256_file(out / "reused_inner_predictions.csv"))
    # Baseline runtimes are only planning evidence; ranking overhead is unmeasured.
    planning = dict(baseline_member_seconds_median=float(np.median(timings)),
        baseline_member_seconds_p95=float(np.percentile(timings,95)),
        new840_baseline_equivalent_worker_seconds=840*float(np.mean(timings)),
        provisional_ranking_overhead_multiplier_range=[2,6],
        provisional_two_worker_wall_minutes_range=[840*float(np.mean(timings))*factor/2/60 for factor in (2,6)],
        status="planning estimate only; real two-fit lambda0 timings will measure canonical member cost; ranking pair overhead remains a bounded risk")
    atomic_json(out / "reuse_audit.json", dict(status="passed_read_only", canonical_calls=100,
        logical_members=480, exact_signature_reconstruction=True, cohort_features_ancestry_recipe=True,
        selections_replayed_from_raw_inner_predictions=True, sources=all_sources, runtime_planning=planning))
    item = dict(protocol_hash=config_digest(protocol), frozen_at_utc=datetime.now(timezone.utc).isoformat(), specification=protocol)
    atomic_json(out / "protocol.json", item)
    atomic_csv(out / "fold_manifest.csv", train[["pair_id", "cluster_id", "allele"]].assign(outer_fold=nested.outer_folds(train)))
    return item


def verify_freeze(out):
    out = Path(out)
    item = json.loads((out / "protocol.json").read_text())
    spec = item["specification"]
    if config_digest(spec) != item["protocol_hash"]:
        raise ValueError("ranking protocol changed after freeze")
    for path, expected in spec["source_hashes"].items():
        if sha256_file(ROOT / path) != expected:
            raise ValueError(f"source changed after freeze: {path}")
    for name, key in (("fit_graph.json", "fit_graph_sha256"), ("reused_inner_predictions.csv", "reused_inner_predictions_sha256")):
        if sha256_file(out / name) != spec[key]:
            raise ValueError("frozen prefit artifact changed")
    return item


def require_authorization(out, phase):
    protocol = verify_freeze(out)
    if runtime_identity() != protocol["specification"]["runtime"]:
        raise ValueError("runtime versions/platform/numerical libraries changed after freeze")
    gate = json.loads((Path(out) / "gate.json").read_text())
    if gate.get("protocol_hash") != protocol["protocol_hash"] or not gate.get(f"{phase}_authorized"):
        raise ValueError(f"root-owned {phase} authorization missing")
    if gate.get("max_workers", 2) > 2 or gate.get("threads_per_worker", 2) != 2:
        raise ValueError("frozen two-worker/two-thread allocation exceeded")
    if phase == "scientific":
        validation = json.loads((Path(out) / "lambda0_validation.json").read_text())
        if validation.get("protocol_hash") != protocol["protocol_hash"] or validation.get("status") != "passed" or len(validation["members"]) != 2:
            raise ValueError("both real lambda0 validation members must pass")
    return protocol


def validate_lambda0(out):
    protocol = require_authorization(out, "validation")
    out = Path(out)
    if (out / "lambda0_validation.json").exists():
        raise ValueError("validation already receipted; no repeated validation fits")
    train = training_frame()
    features, _ = load_features(train)
    outer = nested.outer_folds(train)
    fit = train.loc[outer != 0]
    folds = nested.inner_folds(fit, 3)
    inner_fit, held = fit.loc[folds != 0], fit.loc[folds == 0]
    stop_folds = nested.inner_folds(inner_fit, 5)
    gradient, stopping = inner_fit.loc[stop_folds != 0], inner_fit.loc[stop_folds == 0]
    position = {int(pid): i for i, pid in enumerate(train.pair_id)}
    records = []
    with threadpool_limits(limits=2):
        for rep in REPRESENTATIONS:
            candidate = head_candidates(rep, "mlp")[1]
            context = f"outer/0/{rep}_mlp/select/{candidate['id']}/0"
            path, cache = cached_call(train, features, 0, rep, inner_fit, held, candidate, context)
            launch = out / f"validation_{rep}.launched.json"
            with launch.open("x") as handle:
                json.dump(dict(protocol_hash=protocol["protocol_hash"], phase="lambda0_validation", representation=rep), handle)
            X = features[rep][[position[int(pid)] for pid in inner_fit.pair_id]]
            Xh = features[rep][[position[int(pid)] for pid in held.pair_id]]
            mask = stop_folds != 0
            started = time.perf_counter()
            scale = StandardScaler().fit(X[mask])
            model = RankingMLPRegressor(member_config(candidate, 0), rank_weight=0., dev_metric="mse")
            model.fit(scale.transform(X[mask]), gradient.y_log1p.to_numpy(),
                      scale.transform(X[~mask]), stopping.y_log1p.to_numpy())
            actual = model.predict(scale.transform(Xh))
            reference = np.asarray(cache["by_seed"]["0"])
            max_difference = float(np.max(np.abs(actual-reference)))
            # Canonical code is bit exact; two-thread BLAS execution may change
            # low-order bits across machines. Epoch identities must still match.
            np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)
            source_member = cache["members"][0]
            if model.best_epoch_ != source_member["best_epoch"] or model.n_epochs_ != source_member["epochs"]:
                raise ValueError("lambda0 validation changed early-stopping epoch")
            atomic_csv(out / f"validation_{rep}_predictions.csv", held[["pair_id", "cluster_id", "allele"]].assign(
                canonical=reference.astype(np.float64), lambda0=actual.astype(np.float64)))
            np.savez_compressed(out / f"validation_{rep}_scaler.npz", mean=scale.mean_, scale=scale.scale_, var=scale.var_)
            record = dict(representation=rep, source_path=relative(path), source_sha256=sha256_file(path),
                gradient_rows=ordered_rows(gradient), stopping_rows=ordered_rows(stopping), prediction_rows=ordered_rows(held),
                scaler_rows=ordered_rows(gradient), source_member=source_member, max_abs_difference=max_difference,
                tolerance_rtol=2e-5, tolerance_atol=2e-5, best_epoch=model.best_epoch_, epochs=model.n_epochs_,
                seconds=time.perf_counter()-started, threads=threadpool_info())
            records.append(record)
            atomic_json(out / f"validation_{rep}.completed.json", record)
    item = dict(status="passed", protocol_hash=protocol["protocol_hash"], members=records)
    atomic_json(out / "lambda0_validation.json", item)
    return item


def run_component(out, rep, fold):
    protocol = require_authorization(out, "scientific")
    out = Path(out)
    directory = out / f"fold_{fold}/{rep}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "completed.json").exists():
        return json.loads((directory / "completed.json").read_text())
    # Controller receipt survives failures. A duplicate launch cannot overlap.
    with (directory / "controller.launched.json").open("x") as handle:
        json.dump(dict(pid=os.getpid(), protocol_hash=protocol["protocol_hash"],
                       representation=rep, outer_fold=fold, status="launched"), handle)
    try:
        train = training_frame()
        features, _ = load_features(train)
        if {rep: array_hash(x) for rep,x in features.items()} != protocol["specification"]["feature_hashes"]:
            raise ValueError("features changed after freeze")
        outer = nested.outer_folds(train)
        fit, held = train.loc[outer != fold], train.loc[outer == fold]
        mse_choice, mse_table, mse_raw, _ = read_baseline(train, features, fold, rep)
        ancestry = Ancestry()
        head = RankedHead(train, features, ancestry, directory / "members", protocol["protocol_hash"])
        with threadpool_limits(limits=2):
            choice = select(fit, candidates(rep), head, ancestry=ancestry, context=f"outer/{fold}/{rep}_ranked")
            # Fresh raw inner predictions come from the just-receipted calls.
            tables = []
            fit_by_id = fit.set_index("pair_id")
            for call in head.calls:
                ids = call["prediction_pair_ids"]
                rows = fit_by_id.loc[ids].reset_index()[["pair_id", "cluster_id", "allele"]]
                inner = int(call["signature"]["context"].rsplit("/", 1)[1])
                tables.append(rows.assign(outer_fold=fold, inner_fold=inner, representation=rep,
                    candidate=call["signature"]["candidate"]["id"], y_pred=call["predictions"]))
            pred = head.predict(fit, unlabelled(held), choice.candidate,
                                f"outer/{fold}/{rep}_ranked/outer_refit", final=True)
        ancestry.add(role="outer_ranked_selection", fit=fit, held=held, selection=fit,
                     candidate=choice.candidate["id"], context=f"outer/{fold}/{rep}_ranked")
        deployment = deploy_choice(mse_choice, asdict(choice))
        tables_out = []
        for arm, values, candidate in ((f"{rep}_mse", mse_table.y_pred.to_numpy(), mse_choice["candidate"]["id"]),
             (f"{rep}_ranked", pred, choice.candidate["id"]),
             (f"{rep}_deploy", mse_table.y_pred.to_numpy() if deployment["procedure"] == "mse" else pred,
              mse_choice["candidate"]["id"] if deployment["procedure"] == "mse" else choice.candidate["id"])):
            tables_out.append(held[["pair_id", "cluster_id", "allele"]].assign(outer_fold=fold, arm=arm,
                               y_pred=np.asarray(values,dtype=np.float64), selected_candidate=candidate))
        atomic_csv(directory / "predictions.csv", pd.concat(tables_out, ignore_index=True))
        atomic_csv(directory / "inner_predictions.csv", pd.concat(tables, ignore_index=True))
        atomic_csv(directory / "reused_inner_predictions.csv", mse_raw)
        atomic_json(directory / "selection.json", dict(mse=mse_choice, ranked=asdict(choice), deployment=deployment))
        atomic_json(directory / "ancestry.json", ancestry.as_dict())
        atomic_json(directory / "member_manifest.json", head.members)
        atomic_csv(directory / "members.csv", pd.DataFrame([{key: value for key,value in m.items() if key != "identity"} for m in head.members]))
        if len(head.members) != 84:
            raise ValueError("scientific fit count changed")
        files = ("predictions.csv", "inner_predictions.csv", "reused_inner_predictions.csv", "selection.json",
                 "ancestry.json", "member_manifest.json", "members.csv")
        result = dict(status="complete", protocol_hash=protocol["protocol_hash"], representation=rep,
            outer_fold=fold, new_scientific_fits=84, logical_MSE_reused=48,
            ancestry=ancestry.audit(train, held), files={name: sha256_file(directory/name) for name in files})
        atomic_json(directory / "completed.json", result)
        return result
    except BaseException as error:
        atomic_json(directory / "controller.exit.json", dict(status="failed", error=repr(error)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "validate", "fit"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--representation", choices=REPRESENTATIONS)
    parser.add_argument("--fold", type=int, choices=range(5))
    args = parser.parse_args()
    if args.mode == "freeze":
        result = freeze(args.out)
        print(json.dumps(dict(status="frozen", protocol_hash=result["protocol_hash"])))
    elif args.mode == "validate":
        print(json.dumps(validate_lambda0(args.out)))
    else:
        if args.representation is None or args.fold is None:
            parser.error("fit requires --representation and --fold")
        print(json.dumps(run_component(args.out, args.representation, args.fold)))


if __name__ == "__main__":
    main()
