"""Freeze label-free energy scope; cloud-only new heads and audited baseline reuse."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from pepstab import followup3_energy as core

OUT = ROOT / "reports/followup3/energy"
INVERSE = ROOT / "reports/followup3/inverse_execution_v2"
TECHNICAL = ROOT / "reports/followup3/foldx"
PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl")
SOURCES = ("pepstab/followup3_energy.py", "scripts/followup3_energy.py", "modal_app/followup3_energy.py",
           "scripts/followup3_foldx.py", "pepstab/foldx.py", "pepstab/structural_features.py", "pepstab/boltz_embeddings.py",
           "pepstab/followup3_inverse.py", "scripts/followup3_inverse.py", "pepstab/round2_embedding.py", "pepstab/mlp.py",
           "pepstab/nested.py", "pepstab/round2_nested.py", "pepstab/evaluation.py", "pepstab/round2_reporting.py",
           "pepstab/final_cloud_targets.py", "pepstab/final_cloud_protocol.py", "pepstab/data.py",
           "scripts/boltz_pose_check.py", "reports/followup3/THIRD_PRIMARY_AMENDMENT.md",
           "reports/followup3/energy/DECISION_DRAFT.md", "reports/followup3/energy/SPECIFICATION.md")


def sha(path):
    from pepstab.boltz_embeddings import sha256_file
    return sha256_file(path)


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise ValueError("refusing changed frozen energy artifact")
        return
    with path.open("x") as handle:
        handle.write(text)


def freeze(out, inverse, technical):
    """Read technical repaired data only;never assay outcomes or head predictions."""
    from scripts.followup3_inverse import read_panel
    from scripts.followup3_foldx import verify as verify_technical
    out, inverse, technical = Path(out), Path(inverse), Path(technical)
    panel, reference = read_panel(inverse)
    technical_panel, technical_protocol = verify_technical(technical)
    receipt_path = technical / "technical_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if (receipt["status"], receipt["attempted"], receipt["successful"], receipt["failures"], receipt["unattempted"]) != ("passed", 32, 32, 0, 0):
        raise ValueError("ALL32 technical QC gate has not passed")
    if receipt["protocol_hash"] != technical_protocol["protocol_hash"]:
        raise ValueError("technical receipt protocol differs")
    cases = []
    for name, expected in receipt["case_artifact_sha256"].items():
        path = technical / "cases" / name
        if sha(path) != expected:
            raise ValueError("technical case provenance differs")
        cases.append(json.loads(path.read_text()))
    if {(c["allele"], c["peptide"]) for c in cases} != set(technical_panel[["allele", "peptide"]].itertuples(index=False, name=None)):
        raise ValueError("technical cases differ from exact predetermined pilot")
    columns = core.pilot_schema(cases, panel)
    out.mkdir(parents=True, exist_ok=True)
    text = panel.to_csv(index=False)
    if (out / "panel.csv").exists() and (out / "panel.csv").read_text() != text:
        raise ValueError("same frozen inverse panel required")
    (out / "panel.csv").write_text(text)
    base_path = inverse / "base_features.npz"
    base_provenance = inverse / "base_features.provenance.json"
    if sha(base_path) != json.loads(base_provenance.read_text())["base_sha256"]:
        raise ValueError("inverse baseline features require intact staging")
    reused = []
    for case in cases:
        identity = panel.loc[panel.allele.eq(case["allele"]) & panel.peptide.eq(case["peptide"])].iloc[0].to_dict()
        reused.append(dict(identity, status="ok", binary_sha256=core.BINARY_SHA256, repair_interface="ALL", terms=case["repaired"]["terms"],
                           structure_sha256=case["structure_sha256"], metadata_sha256=case["metadata_sha256"],
                           repaired_pdb_sha256=case["repaired_pdb_sha256"], fxout_sha256=case["repaired"]["fxout_sha256"],
                           reused_technical_case=True, technical_case_sha256=receipt["case_artifact_sha256"][core.digest([case["allele"], case["peptide"]]) + ".json"],
                           elapsed_s=0.))
    specification = dict(schema="followup3-repaired-energy.1", n_rows=len(panel), same_inverse_panel_sha256=reference["panel_sha256"],
                         energy_columns=columns, feature_terms="complete shared finite repaired numeric schema;no outcome selection",
                         new_arms=list(core.ARMS), reused_baselines=["R0", "R5"], mlp_new_fits=200, ridge_new_fits=100,
                         mlp_recipe=reference["specification"]["mlp"], mlp_l2=[1., .01, .0001], ridge_alpha=[10000., 100., 1.],
                         outer_folds=5, inner_folds=3, selection="pooled clean inner median per-allele Spearman,min20;stronger tie",
                         primary="MLP R5_E minus matched MLP R5", secondary=["R0 increment", "representation-by-energy-gain interaction", "ridge versions"],
                         bootstrap=dict(n_draws=2000, seed=20261003, primary_panel=20, sensitivity_panel=50, pointwise_interval=95.,
                                        three_primary_family_tail=.05 / (2 * 3)), scientific_versions={p:version(p) for p in PACKAGES},
                         extraction=dict(binary_sha256=core.BINARY_SHA256, repair_interface="ALL", cpu_hard_limit=1, memory_hard_mib=2048,
                                         max_containers=100, per_case_timeout_s=600, global_max_hours=5, requested_retries=0,
                                         interrupted_started_attempts="no automatic retry;retain interruption as incomplete evidence"),
                         limitations=["conditional development evidence", "bounded two-member comparator", "no kinetic/causal/pose-validity claim", "no saved-weight replay"])
    protocol = dict(status="FROZEN_SCOPE_NOT_PAID_LAUNCH_APPROVAL", frozen_at_utc=datetime.now(timezone.utc).isoformat(), specification=specification,
                    source_sha256={p:sha(ROOT / p) for p in SOURCES}, panel_sha256=sha(out / "panel.csv"),
                    inverse_directory=str(inverse.resolve().relative_to(ROOT)), inverse_protocol_sha256=sha(inverse / "protocol.json"),
                    base_features_sha256=sha(base_path), base_features_provenance_sha256=sha(base_provenance),
                    technical_receipt_sha256=sha(receipt_path), original_scientific_specification=reference["protocol_hash"],
                    canonical_inputs=reference["input_sha256"])
    protocol["protocol_hash"] = core.digest(protocol)
    write_once(out / "pilot_energy_rows.json", reused)
    protocol["pilot_energy_rows_sha256"] = sha(out / "pilot_energy_rows.json")
    # The protocol hash binds the reuse artifact too.
    protocol.pop("protocol_hash")
    protocol["protocol_hash"] = core.digest(protocol)
    write_once(out / "protocol.json", protocol)
    print(json.dumps(dict(panel_rows=len(panel), energy_terms=len(columns), reused_pilot_rows=len(reused), protocol_hash=protocol["protocol_hash"])))


def verify(out):
    out = Path(out)
    protocol = json.loads((out / "protocol.json").read_text())
    bound = dict(protocol)
    expected = bound.pop("protocol_hash")
    if core.digest(bound) != expected or sha(out / "panel.csv") != protocol["panel_sha256"]:
        raise ValueError("energy protocol/panel changed")
    for name, expected in protocol["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"frozen energy source changed:{name}")
    for name, expected in protocol["canonical_inputs"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("canonical energy inputs changed")
    inverse = ROOT / protocol["inverse_directory"]
    for path, key in ((inverse / "protocol.json", "inverse_protocol_sha256"), (inverse / "base_features.npz", "base_features_sha256"),
                      (inverse / "base_features.provenance.json", "base_features_provenance_sha256"), (out / "pilot_energy_rows.json", "pilot_energy_rows_sha256")):
        if sha(path) != protocol[key]:
            raise ValueError("bound inverse feature/protocol or pilot reuse changed")
    return pd.read_csv(out / "panel.csv"), protocol


def collect(out, rows_root):
    panel, protocol = verify(out)
    rows = []
    for identity in panel.to_dict(orient="records"):
        key = core.digest([identity["allele"], identity["peptide"]])
        path = Path(rows_root) / (key + ".json")
        if not path.exists():
            raise ValueError("missing/interrupted energy rows block assessment")
        row = json.loads(path.read_text())
        if row["protocol_hash"] != protocol["protocol_hash"] or row.get("repair_interface") != "ALL":
            raise ValueError("repaired row frozen protocol differs")
        rows.append(row)
    _, matrix = core.validate_energy_rows(panel, rows, protocol["specification"]["energy_columns"])
    destination = Path(out) / "energy_features.npz"
    np.savez_compressed(destination, E=matrix.astype(np.float64), allele=panel.allele.to_numpy(str), peptide=panel.peptide.to_numpy(str))
    write_once(Path(out) / "energy_features.provenance.json", dict(protocol_hash=protocol["protocol_hash"],
               feature_sha256=sha(destination), row_artifact_sha256={p.name:sha(p) for p in Path(rows_root).glob("*.json") if not p.name.endswith(".started.json")}))


def load_frame_features(out):
    from scripts.followup3_inverse import labelled_panel
    _, protocol = verify(out)
    inverse = ROOT / protocol["inverse_directory"]
    frame = labelled_panel(inverse)
    arrays = {}
    for path, expected in ((inverse / "base_features.npz", protocol["base_features_sha256"]),
                           (Path(out) / "energy_features.npz", json.loads((Path(out) / "energy_features.provenance.json").read_text())["feature_sha256"])):
        if sha(path) != expected:
            raise ValueError("matched finite feature cache changed")
        with np.load(path, allow_pickle=False) as archive:
            if not np.array_equal(archive["allele"], frame.allele) or not np.array_equal(archive["peptide"], frame.peptide):
                raise ValueError("natural-key feature order differs")
            for key in ("R0", "R5", "E"):
                if key in archive:
                    arrays[key] = archive[key].copy()
    return frame, arrays, protocol


def validated_baseline(out, learner, fold, frame):
    """Content/ancestry/choice replay BEFORE using already fitted baseline vectors."""
    from pepstab.followup3_inverse import audit_ancestry, panel_metric
    _, protocol = verify(out)
    inverse = ROOT / protocol["inverse_directory"]
    directory = inverse / "assessment" / learner
    location = directory / "all" if (directory / "all/receipt.json").exists() else directory / f"fold_{fold}"
    receipt = json.loads((location / "receipt.json").read_text())
    for name, extension in (("predictions", "csv"), ("selection", "json"), ("ancestry", "json")):
        if sha(location / f"{name}.{extension}") != receipt[name + "_sha256"]:
            raise ValueError("baseline artifact receipt differs")
    audit_ancestry(frame, json.loads((location / "ancestry.json").read_text()), learner, receipt["folds"])
    all_selection = json.loads((location / "selection.json").read_text())
    chosen = [s for s in all_selection if s["outer_fold"] == fold and s["arm"] in ("R0", "R5")]
    if len(chosen) != 2:
        raise ValueError("two exact matched baseline choices required")
    for selection in chosen:
        fit = frame.loc[frame.outer_fold.ne(fold)].set_index("pair_id")
        if set(selection["fitting_pair_ids"]) != set(fit.index):
            raise ValueError("baseline fitting panel differs")
        fit = fit.loc[selection["fitting_pair_ids"]]
        replay = [panel_metric(fit, c["inner_predictions"], 20)[0] for c in selection["choices"]]
        if not np.allclose(replay, [c["score"] for c in selection["choices"]], atol=1e-12, rtol=0):
            raise ValueError("baseline inner score replay differs")
        key, selected_key = ("regularization", "selected_l2") if learner == "mlp" else ("alpha", "selected_alpha")
        if selection[selected_key] != selection["choices"][int(np.argmax(replay))][key]:
            raise ValueError("baseline inner selection replay differs")
    predictions = pd.read_csv(location / "predictions.csv", float_precision="round_trip").query("outer_fold == @fold")
    held = frame.loc[frame.outer_fold.eq(fold), ["allele", "peptide", "cluster_id", "outer_fold"]]
    matched = held.merge(predictions, on=list(held), how="outer", validate="one_to_one", indicator=True)
    if not matched._merge.eq("both").all() or not np.isfinite(matched[["R0", "R5"]]).all().all():
        raise ValueError("baseline held identity differs")
    return matched[["R0", "R5"]].reset_index(drop=True), dict(receipt_sha256=sha(location / "receipt.json"), reused_choices=chosen)


def assess(out, learner, fold):
    from threadpoolctl import threadpool_limits
    frame, arrays, protocol = load_frame_features(out)
    if {p:version(p) for p in PACKAGES} != protocol["specification"]["scientific_versions"]:
        raise ValueError("scientific versions differ")
    baseline, reuse = validated_baseline(out, learner, fold, frame)
    with threadpool_limits(limits=2):
        held, predictions, selection, ancestry = core.assess_heads(frame, arrays, arrays["E"], learner, fold)
    location = Path(out) / "assessment" / learner / f"fold_{fold}"
    location.mkdir(parents=True, exist_ok=True)
    result = held[["allele", "peptide", "cluster_id", "outer_fold"]].reset_index(drop=True)
    for arm in ("R0", "R5"):
        result[arm] = baseline[arm]
    for arm, values in predictions.items():
        result[arm] = values
    result.to_csv(location / "predictions.csv", index=False)
    write_once(location / "selection.json", selection)
    write_once(location / "ancestry.json", ancestry)
    write_once(location / "baseline_reuse.json", reuse)
    write_once(location / "receipt.json", dict(protocol_hash=protocol["protocol_hash"], new_fits=40 if learner == "mlp" else 20,
               content_sha256={p.name:sha(p) for p in location.iterdir() if p.is_file()}))


def report(out):
    from pepstab.followup3_inverse import panel_metric
    from pepstab.round2_reporting import compare_predictions
    frame, _, protocol = load_frame_features(out)
    comparisons, scores = [], []
    for learner in ("mlp", "ridge"):
        tables = []
        for fold in range(5):
            location = Path(out) / "assessment" / learner / f"fold_{fold}"
            receipt = json.loads((location / "receipt.json").read_text())
            if receipt["protocol_hash"] != protocol["protocol_hash"]:
                raise ValueError("head protocol differs")
            for name, expected in receipt["content_sha256"].items():
                if sha(location / name) != expected:
                    raise ValueError("head artifact changed")
            core.audit_heads(frame, json.loads((location / "ancestry.json").read_text()), learner, fold)
            baseline, _ = validated_baseline(out, learner, fold, frame)
            table = pd.read_csv(location / "predictions.csv", float_precision="round_trip")
            if not np.array_equal(table[["R0", "R5"]].to_numpy(), baseline.to_numpy()):
                raise ValueError("saved baseline endpoint differs from exact reused vectors")
            selection = json.loads((location / "selection.json").read_text())
            if len(selection) != 2:
                raise ValueError("two selected energy arms required")
            for s in selection:
                fitting = frame.loc[frame.outer_fold.ne(fold)].set_index("pair_id")
                if set(fitting.index) != set(s["fitting_pair_ids"]):
                    raise ValueError("new energy inner population differs")
                fitting = fitting.loc[s["fitting_pair_ids"]]
                replay = [panel_metric(fitting, c["inner_predictions"], 20)[0] for c in s["choices"]]
                if not np.allclose(replay, [c["score"] for c in s["choices"]], atol=1e-12, rtol=0) or s["selected_regularization"] != s["choices"][int(np.argmax(replay))]["regularization"]:
                    raise ValueError("independent new energy selection replay differs")
            tables.append(pd.read_csv(location / "predictions.csv", float_precision="round_trip"))
        predictions = pd.concat(tables, ignore_index=True)
        if len(predictions) != len(frame) or predictions.duplicated(["allele", "peptide"]).any():
            raise ValueError("complete held energy predictions required")
        joined = frame.merge(predictions, on=["allele", "peptide", "cluster_id", "outer_fold"], validate="one_to_one")
        if len(joined) != len(frame):
            raise ValueError("held identities differ")
        values = {arm:joined[arm].to_numpy(float) for arm in ("R0", "R5", "R0_E", "R5_E")}
        for minimum in (20, 50):
            result, boot, meta = compare_predictions(joined, values, [("R5", "R5_E"), ("R0", "R0_E")], min_rows=minimum, n_boot=2000, seed=20261003)
            result["learner"] = learner
            tail = protocol["specification"]["bootstrap"]["three_primary_family_tail"]
            result["family_ci_low"] = np.nan
            result["family_ci_high"] = np.nan
            primary = result.baseline.eq("R5") & result.candidate.eq("R5_E")
            result.loc[primary, "family_ci_low"] = np.nanquantile(boot.R5_E - boot.R5, tail)
            result.loc[primary, "family_ci_high"] = np.nanquantile(boot.R5_E - boot.R5, 1 - tail)
            comparisons.append(result)
            point = {arm:panel_metric(joined, vector, minimum)[0] for arm, vector in values.items()}
            draws = (boot.R5_E - boot.R5) - (boot.R0_E - boot.R0)
            comparisons.append(pd.DataFrame([dict(learner=learner, panel_min_rows=minimum, baseline="R0_energy_gain", candidate="R5_energy_gain",
                       delta=(point["R5_E"] - point["R5"]) - (point["R0_E"] - point["R0"]),
                       ci_low=np.nanquantile(draws, .025), ci_high=np.nanquantile(draws, .975))]))
            scores.extend(dict(learner=learner, arm=arm, min_rows=minimum, median_spearman=value, n_alleles=meta["n_alleles"]) for arm, value in point.items())
            boot.to_csv(Path(out) / f"bootstrap_{learner}_min{minimum}.csv", index=False)
    pd.concat(comparisons, ignore_index=True).to_csv(Path(out) / "comparisons.csv", index=False)
    pd.DataFrame(scores).to_csv(Path(out) / "pooled_scores.csv", index=False)
    write_once(Path(out) / "completion_audit.json", dict(status="passed", protocol_hash=protocol["protocol_hash"],
               new_mlp_members=200, new_ridge_fits=100, baseline_fits_repeated=0, independent_selection_replay=True,
               ancestry_and_scaler_audit=True, saved_weight_replay=False, complete_assessment_rows=len(frame),
               output_sha256={p.name:sha(p) for p in Path(out).glob("*.csv")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "collect", "assess", "report"))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--inverse-out", type=Path, default=INVERSE)
    parser.add_argument("--technical-out", type=Path, default=TECHNICAL)
    parser.add_argument("--rows-root", type=Path)
    parser.add_argument("--learner", choices=("mlp", "ridge"), default="mlp")
    parser.add_argument("--outer-fold", type=int, choices=range(5), default=0)
    args = parser.parse_args()
    if args.command == "freeze": freeze(args.out, args.inverse_out, args.technical_out)
    elif args.command == "collect": collect(args.out, args.rows_root)
    elif args.command == "assess": assess(args.out, args.learner, args.outer_fold)
    else: report(args.out)


if __name__ == "__main__": main()
