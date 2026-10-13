"""Prepare locally; extract, fit and bootstrap only in bounded cloud workers."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib.metadata import version
import json
import os
import platform
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from pepstab import followup3_inverse as core

OUT = ROOT / "reports/followup3/inverse"
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl")


def sha(path):
    from pepstab.boltz_embeddings import sha256_file
    return sha256_file(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != data:
            raise ValueError(f"refusing to overwrite changed artifact:{path}")
        return
    with path.open("x") as handle:
        handle.write(data)


def read_panel(out):
    protocol = json.loads((Path(out) / "protocol.json").read_text())
    if core.digest(protocol["specification"]) != protocol["protocol_hash"]:
        raise ValueError("protocol identity changed")
    if sha(Path(out) / "panel.csv") != protocol["panel_sha256"]:
        raise ValueError("frozen panel changed")
    for name, expected in protocol["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"frozen implementation changed:{name}")
    for name, expected in protocol["input_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"original input changed:{name}")
    return pd.read_csv(Path(out) / "panel.csv"), protocol


def prepare(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = ("data/splits.csv", "data/structural_cohort.csv", "data/rasmussen_et_al_dataset.csv",
              "reports/further_research/round2/sequence/fold_manifest.csv",
              "reports/finalization/cloud/canonical_training_targets.json", "reports/finalization/cloud/canonical_training_targets.npz")
    panel = core.select_panel(pd.read_csv(ROOT / inputs[0]), pd.read_csv(ROOT / inputs[3]), pd.read_csv(ROOT / inputs[1]))
    text = panel.to_csv(index=False)
    if (out / "panel.csv").exists() and (out / "panel.csv").read_text() != text:
        raise ValueError("refusing changed frozen panel")
    (out / "panel.csv").write_text(text)
    spec = dict(schema="followup3-inverse.1", population="ORIGINAL train only; balanced pair subset; inherited whole-cluster outer folds",
                n_rows=len(panel), n_alleles=int(panel.allele.nunique()), n_clusters=int(panel.cluster_id.nunique()),
                sample_seed=core.PANEL_SEED, per_allele_cap=64, selection="SHA256(seed,allele,peptide) ascending within allele",
                arms=list(core.ARMS), ridge_alpha=list(core.ALPHAS), mlp_l2=[1., .01, .0001],
                mlp=dict(hidden=[64], max_epochs=200, patience=15, learning_rate=.001, batch_size=256, members=[[0, 0], [1, 1]],
                         final_recipe="same bounded two-member as inner;FeatureHead.predict final=False"),
                selection_metric="pooled clean inner OOF median per-allele Spearman,min20;exact ties stronger regularization",
                outer_folds=5, inner_folds=3, ridge_fits=300, mlp_fits=600, primary="mlp:R5_PC minus R5_P",
                key_secondary="mlp:R5_PC minus R5", secondary="all R0 and ridge analogues",
                scoring=dict(checkpoint_sha256=core.CHECKPOINT_SHA256, checkpoint="v_48_020.pt", n_orders=16, seed=0,
                             decoding="explicit-common-peptide-permutations-v1;visible context before native peptide",
                             fitted_features="9 peptide-only positions +9 conditional positions;totals/deltas diagnostic only"),
                legacy_qc_reused=False, legacy_reason="different decoding order protocol and mixed original splits",
                bootstrap=dict(n_draws=2000, seed=20261003, unit="whole inherited peptide cluster", primary_panel=20, sensitivity_panel=50, primary_family_interval=97.5),
                scientific_versions={p:version(p) for p in PACKAGES},
                limitations=["conditional development evidence on repeatedly used training labels", "subset distribution differs from full training cohort",
                             "context ablation changes graph and neighbors;does not establish causal geometry or kinetic mechanism",
                             "neural head is bounded two-member comparator;no claim versus full30-member deployed head",
                             "saved predictions/ancestry support scoring audit;fitted model weights are not serialized"])
    sources = ("pepstab/followup3_inverse.py", "scripts/followup3_inverse.py", "pepstab/inverse_folding.py",
               "scripts/boltz_pose_check.py", "pepstab/round2_embedding.py", "pepstab/mlp.py", "pepstab/nested.py",
               "pepstab/evaluation.py", "pepstab/round2_reporting.py", "pepstab/round2_nested.py", "pepstab/data.py", "pepstab/features.py",
               "pepstab/round2_representations.py", "pepstab/boltz_embeddings.py", "external/ProteinMPNN/protein_mpnn_utils.py",
               "pepstab/final_cloud_targets.py", "pepstab/final_cloud_protocol.py")
    protocol = dict(status="FROZEN_SCIENTIFIC_SCOPE_NOT_PAID_LAUNCH_APPROVAL", specification=spec, protocol_hash=core.digest(spec),
                    panel_sha256=sha(out / "panel.csv"), source_sha256={p: sha(ROOT / p) for p in sources},
                    input_sha256={p: sha(ROOT / p) for p in inputs})
    write_json(out / "protocol.json", protocol)
    coverage = dict(rows_by_allele=panel.groupby("allele").size().to_dict(),
                    outer_fold_counts=panel.groupby("outer_fold").size().to_dict(),
                    per_allele_fold=panel.groupby(["allele", "outer_fold"]).size().unstack(fill_value=0).to_dict(orient="index"))
    write_json(out / "coverage.json", coverage)
    print(json.dumps(dict(panel_rows=len(panel), alleles=int(panel.allele.nunique()), clusters=int(panel.cluster_id.nunique()), protocol_hash=protocol["protocol_hash"])))


def _extract_one(args):
    volume_root, row, output = args
    import torch
    torch.set_num_threads(2)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    path = Path(output) / (core.digest([row["allele"], row["peptide"]]) + ".json")
    if path.exists():
        cached = json.loads(path.read_text())
        if any(str(cached[k]) != str(v) for k, v in row.items()):
            raise ValueError("existing row identity differs")
        return cached
    try:
        result = core.score_pair(Path(volume_root) / row["folder"], row)
    except Exception as error:
        result = dict(row, status="failed", error=f"{type(error).__name__}:{error}")
    write_json(path, result)
    return result


def extract(out, volume_root, workers, pilot=False):
    panel, protocol = read_panel(out)
    if not 1 <= workers <= 8:
        raise ValueError("extraction cap8 workers x2torch threads")
    # Stable SHA-first representatives: engineering pilot reads no assay labels.
    if pilot:
        panel = panel.iloc[np.linspace(0, len(panel) - 1, 5, dtype=int)].copy()
    directory = Path(out) / ("pilot_rows" if pilot else "score_rows")
    directory.mkdir(parents=True, exist_ok=True)
    if not pilot:
        # The five same-protocol engineering rows are verified again by the
        # final panel acceptance check, then reused without duplicate scoring.
        for source in (Path(out) / "pilot_rows").glob("*.json"):
            write_json(directory / source.name, json.loads(source.read_text()))
    tasks = [(str(volume_root), row, str(directory)) for row in panel.to_dict(orient="records")]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_extract_one, tasks))
    result = pd.DataFrame(rows)
    name = "pilot_scores.csv" if pilot else "scores.csv"
    result.to_csv(Path(out) / name, index=False)
    core.validate_scores(panel, result)
    write_json(Path(out) / ("pilot_receipt.json" if pilot else "extraction_receipt.json"),
               dict(protocol_hash=protocol["protocol_hash"], n_rows=len(result), scores_sha256=sha(Path(out) / name),
                    total_pair_seconds=float(result.elapsed_s.sum()), workers=workers, engineering_only=pilot))
    print(json.dumps(dict(rows=len(rows), total_pair_seconds=float(result.elapsed_s.sum()), pilot=pilot)))


def labelled_panel(out):
    from pepstab.final_cloud_targets import install, META
    install(ROOT / META)
    from pepstab.data import load_with_splits
    panel, _ = read_panel(out)
    canonical = load_with_splits().query("split == 'train'")
    frame = panel.merge(canonical, on=["allele", "peptide", "cluster_id"], validate="one_to_one")
    if len(frame) != len(panel) or not frame.split.eq("train").all():
        raise ValueError("panel/ORIGINAL train labels mismatch")
    return frame


def stage_features(out, bundle_root=None):
    """Cloud-only staging; load R5 from checked compact bundles, no refolding."""
    from pepstab.features import build_features
    from pepstab.round2_representations import feature_protocol, load_contact_map
    frame = labelled_panel(out)
    manifest_path = ROOT / "reports/further_research/round2/extractor/full_training/feature_bundles.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete" or manifest["expected_rows"] != 19716 or manifest["completed_rows"] != 19716:
        raise ValueError("full ORIGINAL-train R5 cache must be complete")
    positions = {str(v): i for i, v in enumerate(frame.pair_id)}
    R5 = np.full((len(frame), 1024), np.nan, dtype=np.float32)
    seen, provenance = set(), []
    for value in manifest["feature_bundles"]:
        directory = Path(value) if bundle_root is None else Path(bundle_root) / Path(value).name
        complete = json.loads((directory / "complete.json").read_text())
        array_path, info_path = directory / complete["array_file"], directory / complete["metadata_file"]
        if sha(array_path) != complete["npz_sha256"] or sha(info_path) != complete["metadata_sha256"]:
            raise ValueError("compact feature source hash mismatch")
        info = json.loads(info_path.read_text())
        if info["feature_protocol"] != feature_protocol() or info["contact_map_sha256"] != load_contact_map()["contact_map_sha256"]:
            raise ValueError("R5 feature protocol/contact identity mismatch")
        with np.load(array_path, allow_pickle=False) as data:
            ids = data["pair_id"].astype(str).tolist()
            if ids != [str(row["pair_id"]) for row in info["rows"]] or set(ids) & seen:
                raise ValueError("compact feature row identities differ or duplicate")
            seen.update(ids)
            values = data["R5"]
            if values.shape != (len(ids), 1024) or values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError("invalid R5 source matrix")
            for i, pid in enumerate(ids):
                if pid in positions:
                    R5[positions[pid]] = values[i]
        provenance.append(dict(shard=directory.name, npz_sha256=complete["npz_sha256"], metadata_sha256=complete["metadata_sha256"]))
    canonical_ids = set(pd.read_csv(ROOT / "data/splits.csv").query("split == 'train'").pair_id.astype(str))
    if seen != canonical_ids or not np.isfinite(R5).all():
        raise ValueError("incomplete full source/panel R5 coverage")
    R0 = build_features(frame, "pep_pseudo", "onehot").astype(np.float32)
    if R0.shape != (len(frame), 860) or not np.isfinite(R0).all():
        raise ValueError("canonical R0 feature geometry differs")
    destination = Path(out) / "base_features.npz"
    if destination.exists():
        raise ValueError("base staging already exists")
    np.savez_compressed(destination, R0=R0, R5=R5, allele=frame.allele.to_numpy(str), peptide=frame.peptide.to_numpy(str))
    write_json(Path(out) / "base_features.provenance.json", dict(base_sha256=sha(destination),
               full_source_manifest_sha256=sha(manifest_path), bundles=provenance, feature_protocol=feature_protocol()))


def assess(out, learner, fold=None):
    from threadpoolctl import threadpool_limits
    if {p:version(p) for p in PACKAGES} != read_panel(out)[1]["specification"]["scientific_versions"]:
        raise ValueError("frozen scientific package versions differ")
    frame = labelled_panel(out)
    scored = core.validate_scores(frame[list(core.PANEL_COLUMNS)], pd.read_csv(Path(out) / "scores.csv"))
    path = Path(out) / "base_features.npz"
    provenance = json.loads((Path(out) / "base_features.provenance.json").read_text())
    if sha(path) != provenance["base_sha256"]:
        raise ValueError("staged base features changed")
    with np.load(path, allow_pickle=False) as data:
        if not np.array_equal(data["allele"], frame.allele) or not np.array_equal(data["peptide"], frame.peptide):
            raise ValueError("baseline features natural-key order mismatch")
        arms = core.feature_arms(data, scored)
    folds = tuple(range(5)) if fold is None else (fold,)
    with threadpool_limits(limits=2):
        function = core.neural_assessment if learner == "mlp" else core.nested_assessment
        predictions, selection, ancestry = function(frame, arms, folds=folds)
    where = Path(out) / "assessment" / learner / ("all" if fold is None else f"fold_{fold}")
    where.mkdir(parents=True, exist_ok=True)
    subset = frame.outer_fold.isin(folds).to_numpy()
    result = frame.loc[subset, ["allele", "peptide", "cluster_id", "outer_fold"]].copy()
    for arm, p in predictions.items():
        result[arm] = p[subset]
    result.to_csv(where / "predictions.csv", index=False)
    write_json(where / "selection.json", selection)
    write_json(where / "ancestry.json", ancestry)
    write_json(where / "runtime.json", dict(python=sys.version, platform=platform.platform(), packages={p:version(p) for p in PACKAGES}, numerical_threads=2))
    write_json(where / "receipt.json", dict(learner=learner, folds=list(folds), fits=(120 if learner == "mlp" else 60) * len(folds),
               predictions_sha256=sha(where / "predictions.csv"), selection_sha256=sha(where / "selection.json"), ancestry_sha256=sha(where / "ancestry.json")))


def report(out):
    from pepstab.round2_reporting import compare_predictions
    from pepstab import evaluation as ev
    frame = labelled_panel(out)
    comparisons, points = [], []
    for learner in ("mlp", "ridge"):
        root = Path(out) / "assessment" / learner
        locations = [root / "all"] if (root / "all" / "receipt.json").exists() else [root / f"fold_{i}" for i in range(5)]
        tables = []
        for loc in locations:
            receipt = json.loads((loc / "receipt.json").read_text())
            for name in ("predictions", "selection", "ancestry"):
                extension = "csv" if name == "predictions" else "json"
                if sha(loc / f"{name}.{extension}") != receipt[f"{name}_sha256"]:
                    raise ValueError("assessment receipt content mismatch")
            tables.append(pd.read_csv(loc / "predictions.csv"))
            saved = json.loads((loc / "ancestry.json").read_text())
            core.audit_ancestry(frame, saved, learner, receipt["folds"])
        prediction = pd.concat(tables, ignore_index=True)
        if len(prediction) != len(frame) or prediction.duplicated(["allele", "peptide"]).any():
            raise ValueError("incomplete matched outer predictions")
        joined = frame.merge(prediction, on=["allele", "peptide", "cluster_id", "outer_fold"], validate="one_to_one")
        if len(joined) != len(frame):
            raise ValueError("outer prediction identities differ")
        values = {a: joined[a].to_numpy(float) for a in core.ARMS}
        contrasts = [("R5_P", "R5_PC"), ("R5", "R5_PC"), ("R5", "R5_P"),
                     ("R0_P", "R0_PC"), ("R0", "R0_PC"), ("R0", "R0_P")]
        for minimum in (50, 20):
            result, bootstrap, meta = compare_predictions(joined, values, contrasts, min_rows=minimum, n_boot=2000, seed=20261003)
            result["learner"] = learner
            primary_draws = bootstrap.R5_PC - bootstrap.R5_P
            result["family_ci_low"] = np.nan
            result["family_ci_high"] = np.nan
            primary_row = result.baseline.eq("R5_P") & result.candidate.eq("R5_PC")
            result.loc[primary_row, "family_ci_low"] = float(np.nanquantile(primary_draws, .0125))
            result.loc[primary_row, "family_ci_high"] = float(np.nanquantile(primary_draws, .9875))
            comparisons.append(result)
            bootstrap.to_csv(Path(out) / f"bootstrap_{learner}_min{minimum}.csv", index=False)
            for arm, vector in values.items():
                points.append(dict(learner=learner, arm=arm, min_rows=minimum, eligible_alleles=meta["n_alleles"],
                                   median_spearman=ev.score(arm, joined.allele, joined.y_log1p, vector, alleles=meta["alleles"]).median_spearman))
            sample = (bootstrap.R5_PC - bootstrap.R5_P) - (bootstrap.R0_PC - bootstrap.R0_P)
            context_gains = {rep: core.panel_metric(joined, values[rep + "_PC"], minimum)[0] - core.panel_metric(joined, values[rep + "_P"], minimum)[0] for rep in ("R0", "R5")}
            comparisons.append(pd.DataFrame([dict(learner=learner, baseline="R0_context_gain", candidate="R5_context_gain",
                       delta=context_gains["R5"] - context_gains["R0"], ci_low=float(np.nanquantile(sample, .025)),
                       ci_high=float(np.nanquantile(sample, .975)), panel_min_rows=minimum,
                       uncertainty_scope="conditional representation-by-context-gain interaction")]))
        selection_tables = [json.loads((loc / "selection.json").read_text()) for loc in locations]
        for selection in sum(selection_tables, []):
            expected_fit = joined.loc[joined.outer_fold.ne(selection["outer_fold"])].set_index("pair_id")
            if set(selection["fitting_pair_ids"]) != set(expected_fit.index):
                raise ValueError("saved inner selection population differs")
            fit = expected_fit.loc[selection["fitting_pair_ids"]]
            scores = [core.panel_metric(fit, choice["inner_predictions"], 20)[0] for choice in selection["choices"]]
            if not np.allclose(scores, [choice["score"] for choice in selection["choices"]], atol=1e-12, rtol=0):
                raise ValueError("independent inner score replay differs")
            best = int(np.argmax(scores))
            key = "regularization" if learner == "mlp" else "alpha"
            selected_key = "selected_l2" if learner == "mlp" else "selected_alpha"
            if selection[selected_key] != selection["choices"][best][key]:
                raise ValueError("independent selection replay differs")
    pd.concat(comparisons, ignore_index=True).to_csv(Path(out) / "comparisons.csv", index=False)
    pd.DataFrame(points).to_csv(Path(out) / "pooled_scores.csv", index=False)
    scored = core.validate_scores(frame[list(core.PANEL_COLUMNS)], pd.read_csv(Path(out) / "scores.csv"))
    descriptive = []
    for minimum in (20, 50):
        for col in ("C_total", "P_total", "delta_total"):
            score, panel = core.panel_metric(frame, scored[col], minimum)
            descriptive.append(dict(feature=col, min_rows=minimum, median_spearman=score, n_alleles=len(panel), interpretation="descriptive native compatibility only"))
    pd.DataFrame(descriptive).to_csv(Path(out) / "diagnostic_correlations.csv", index=False)
    write_json(Path(out) / "completion_audit.json", dict(status="passed", protocol_hash=read_panel(out)[1]["protocol_hash"],
               scientific_fits=900, row_count=len(frame), neural_members=600, ridge_fits=300,
               independent_inner_score_and_selection_replay=True, cluster_ancestry_audited=True,
               saved_weight_replay=False, outputs={p.name:sha(p) for p in Path(out).glob("*.csv")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "pilot", "extract", "stage", "assess", "report"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--volume-root", type=Path, default=Path("/structures"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--learner", choices=("mlp", "ridge"), default="mlp")
    parser.add_argument("--outer-fold", type=int, choices=range(5))
    args = parser.parse_args()
    if args.command == "prepare": prepare(args.out)
    elif args.command in ("pilot", "extract"): extract(args.out, args.volume_root, args.workers, pilot=args.command == "pilot")
    elif args.command == "stage": stage_features(args.out, args.bundle_root)
    elif args.command == "assess": assess(args.out, args.learner, args.outer_fold)
    else: report(args.out)


if __name__ == "__main__":
    main()
