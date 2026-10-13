"""Repaired FoldX extraction and matched-head complementarity on frozen train."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time

import numpy as np
import pandas as pd

BINARY_SHA256 = "faed5e54e47744ab6ab75f8f9ad1a1f2e2cdbeac4f0f98d4e36c97049eb14268"
ARMS = ("R0_E", "R5_E")
CASE_TIMEOUT_S = 600
MAX_CONTAINERS = 100


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def utc_deadline(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
        raise ValueError("fixed UTC deadline required")
    return result.timestamp()


def remaining_seconds(deadline_utc, case_started):
    remaining = min(utc_deadline(deadline_utc) - time.time(), CASE_TIMEOUT_S - (time.monotonic() - case_started))
    if remaining <= 0:
        raise TimeoutError("fixed global or per-case deadline reached")
    return remaining


def pilot_schema(cases, panel):
    """Accept the complete shared repaired schema using technical evidence only."""
    if len(cases) != 32 or len({row["allele"] for row in cases}) != 32:
        raise ValueError("all32 distinct-allele technical cases required")
    panel_lookup = panel.set_index(["allele", "peptide"])
    schema = None
    for row in cases:
        if row.get("status") != "ok" or row.get("binary_sha256") != BINARY_SHA256:
            raise ValueError("all technical cases must pass pinned binary identity")
        key = (row["allele"], row["peptide"])
        if key not in panel_lookup.index or row["folder"] != panel_lookup.loc[key, "folder"]:
            raise ValueError("technical case lies outside frozen inverse panel")
        if not row.get("structure_sha256") or not row.get("metadata_sha256") or not row.get("repaired_pdb_sha256"):
            raise ValueError("complete technical input/repair provenance required")
        terms = row["repaired"]["terms"]
        columns = sorted(set(terms) - {"foldx_group1", "foldx_group2"})
        if "foldx_interaction_energy" not in columns or not columns:
            raise ValueError("complete repaired numeric schema required")
        if any(not isinstance(terms[c], (int, float)) or isinstance(terms[c], bool) or not np.isfinite(terms[c]) for c in columns):
            raise ValueError("nonfinite/non-numeric repaired component")
        if schema is not None and columns != schema:
            raise ValueError("technical repaired schemas differ")
        schema = columns
    return schema


def validate_energy_rows(panel, rows, columns):
    table = pd.DataFrame(rows)
    keys = ["allele", "peptide"]
    if table.duplicated(keys).any() or len(table) != len(panel):
        raise ValueError("complete energy coverage required;no case deletion")
    joined = panel.merge(table, on=keys, how="outer", validate="one_to_one", suffixes=("", "_energy"), indicator=True)
    if not joined._merge.eq("both").all() or not joined.status.eq("ok").all() or not joined.binary_sha256.eq(BINARY_SHA256).all():
        raise ValueError("missing/extra/failed energy row")
    for col in ("cluster_id", "outer_fold", "complex_id", "folder"):
        if not joined[col].astype(str).eq(joined[col + "_energy"].astype(str)).all():
            raise ValueError("energy row/panel identity differs")
    if any(set(row["terms"]) - {"foldx_group1", "foldx_group2"} != set(columns) for row in rows):
        raise ValueError("production schema differs from complete frozen pilot schema")
    matrix = np.asarray([[row["terms"][c] for c in columns] for row in joined.to_dict(orient="records")], dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite production energy")
    return joined.drop(columns="_merge"), matrix


def score_repaired(identity, folder, distribution, deadline_utc, columns, access_stop_check=None):
    """Exactly one RepairPDB+AnalyseComplex attempt, bounded by both deadlines."""
    from pepstab import foldx as fx
    from scripts.followup3_foldx import pdb_identity, validated_interaction, run_command, sha
    started = time.monotonic()
    record = dict(identity, status="started", binary_sha256=distribution["sha256"], repair_interface="ALL", phases=[])
    try:
        if access_stop_check is not None:
            access_stop_check()
        remaining_seconds(deadline_utc, started)
        folder = Path(folder)
        meta = json.loads((folder / "metadata.json").read_text())
        case = meta["inputs"]
        seed = meta.get("seed", meta.get("settings", {}).get("seed"))
        if (case["allele"], case["peptide"], case["arm"], meta["model"], seed, case.get("split", meta.get("split"))) != (identity["allele"], identity["peptide"], "B", "boltz2", 0, "train"):
            raise ValueError("original-training construct/native identity differs")
        cifs = list(folder.glob("*.cif"))
        if len(cifs) != 1:
            raise ValueError("one source structure required")
        record.update(metadata_sha256=sha(folder / "metadata.json"), structure_sha256=sha(cifs[0]))
        with tempfile.TemporaryDirectory(prefix="energy_repaired_") as temporary:
            work = Path(temporary)
            for field in ("molecules", "rotabase"):
                if distribution.get(field):
                    (work / Path(distribution[field]).name).symlink_to(Path(distribution[field]))
            prepared = fx.prepare_pdb(folder, work / "input")
            raw = Path(prepared["pdb"])
            _, chains = pdb_identity(raw)
            expected = {str(c["id"]):c["sequence"] for c in case["chains"]}
            if chains != expected:
                raise ValueError("conversion/native chain identity differs")
            output = work / "output"
            output.mkdir()
            stop = threading.Event()
            command = fx.repair_command(distribution["binary"], raw.name, str(raw.parent), str(output), "ALL")
            if access_stop_check is not None:
                access_stop_check()
            deadline = time.monotonic() + remaining_seconds(deadline_utc, started)
            record["phases"].append(dict(phase="repair", **run_command(command, str(work), deadline, stop)))
            repaired = output / f"{raw.stem}_Repair.pdb"
            if not repaired.exists() or pdb_identity(repaired)[1] != expected:
                raise ValueError("repair/native chain identity differs")
            groups = (prepared["peptide_chain"], prepared["hla_chain"])
            command = fx.analyse_complex_command(distribution["binary"], repaired.name, str(output), str(output), groups)
            if access_stop_check is not None:
                access_stop_check()
            deadline = time.monotonic() + remaining_seconds(deadline_utc, started)
            record["phases"].append(dict(phase="analysis", **run_command(command, str(work), deadline, stop)))
            outputs = list(output.glob("Interaction_*.fxout"))
            if len(outputs) != 1:
                raise ValueError("one interaction output required")
            terms = validated_interaction(fx.parse_fxout(outputs[0].read_text()), groups)
            if sorted(set(terms) - {"foldx_group1", "foldx_group2"}) != columns:
                raise ValueError("repaired term schema changed")
            record.update(status="ok", terms=terms, fxout_sha256=sha(outputs[0]), repaired_pdb_sha256=sha(repaired),
                          conversion_validation=prepared["validation"], peptide_chain=groups[0], hla_chain=groups[1])
    except Exception as error:
        record.update(status="access_rejected" if isinstance(error, PermissionError) else "failed", error_type=type(error).__name__,
                      error="bounded repaired scoring failed;executable output text not retained")
    record["elapsed_s"] = time.monotonic() - started
    return record


class Float64RidgeHead:
    """Match the inverse baseline's float64 StandardScaler/LSQR recipe exactly."""
    def __init__(self, population, features, ancestry):
        self.position = {int(pid):i for i, pid in enumerate(population.pair_id)}
        self.features = {name:np.asarray(x, dtype=float) for name, x in features.items()}
        self.ancestry, self.members = ancestry, []

    def predict(self, fit, held, candidate, context, final=False):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        from pepstab.round2_nested import validate_partition
        validate_partition(fit, held)
        if "y_log1p" in held or "thalf_hours" in held:
            raise ValueError("ridge received held labels")
        array = self.features[candidate["representation"]]
        x = array[[self.position[int(p)] for p in fit.pair_id]]
        held_x = array[[self.position[int(p)] for p in held.pair_id]]
        started = time.perf_counter()
        model = make_pipeline(StandardScaler(), Ridge(alpha=candidate["regularization"], solver="lsqr", tol=1e-5, max_iter=2000))
        prediction = model.fit(x, fit.y_log1p.to_numpy(float)).predict(held_x)
        self.ancestry.add(role="ridge_head", fit=fit, held=held, context=context,
                          candidate=candidate["id"], scaler_fit_rows=self.ancestry.rows(fit))
        self.members.append(dict(context=context, candidate=candidate["id"], learner="ridge", seed=None,
                                 fit_seconds=time.perf_counter() - started, n_features=x.shape[1], n_fit=len(fit), n_stop=0, dtype="float64"))
        return prediction


def assess_heads(frame, base, energy, learner, fold, *, head_factory=None):
    """Only two new arms:200MLP members+100ridge fits across five folds."""
    from pepstab import nested
    from pepstab.followup3_inverse import panel_metric
    from pepstab.round2_embedding import FeatureHead
    from pepstab.round2_nested import Ancestry, unlabelled
    if learner not in ("mlp", "ridge") or fold not in range(5) or not frame.split.eq("train").all():
        raise ValueError("frozen original-training head scope required")
    if frame.groupby("cluster_id").outer_fold.nunique().max() != 1 or energy.ndim != 2 or len(energy) != len(frame) or not np.isfinite(energy).all():
        raise ValueError("finite matched energy and whole-cluster folds required")
    dtype = np.float32 if learner == "mlp" else np.float64
    features = {rep + "_E":np.column_stack([base[rep], energy]).astype(dtype) for rep in ("R0", "R5")}
    ancestry = Ancestry()
    implementation = FeatureHead if learner == "mlp" else Float64RidgeHead
    head = (implementation if head_factory is None else head_factory)(frame, features, ancestry)
    fitting, held = frame.loc[frame.outer_fold.ne(fold)], frame.loc[frame.outer_fold.eq(fold)]
    inner = nested.inner_folds(fitting, 3)
    selections, predictions = [], {}
    grid = (1., .01, .0001) if learner == "mlp" else (10000., 100., 1.)
    for arm in ARMS:
        choices = []
        for regularization in grid:
            candidate = dict(id=f"{arm}_{learner}_{regularization:g}", representation=arm, learner=learner, regularization=regularization)
            inner_pred = np.full(len(fitting), np.nan)
            for i in range(3):
                inner_pred[inner == i] = head.predict(fitting.loc[inner != i], unlabelled(fitting.loc[inner == i]), candidate,
                                                    f"energy/{learner}/{fold}/{arm}/{regularization:g}/inner/{i}", final=False)
            score, eligible = panel_metric(fitting, inner_pred, 20)
            choices.append(dict(regularization=regularization, score=score, n_eligible=len(eligible), inner_predictions=inner_pred.tolist()))
        chosen = max(choices, key=lambda item:item["score"])
        candidate = dict(id=f"{arm}_{learner}_{chosen['regularization']:g}", representation=arm, learner=learner, regularization=chosen["regularization"])
        predictions[arm] = head.predict(fitting, unlabelled(held), candidate, f"energy/{learner}/{fold}/{arm}/final", final=False)
        selections.append(dict(outer_fold=fold, arm=arm, fitting_pair_ids=fitting.pair_id.tolist(), selected_regularization=chosen["regularization"], choices=choices))
    expected = 40 if learner == "mlp" else 20
    if len(head.members) != expected:
        raise ValueError("bounded new energy-head graph differs")
    return held, predictions, selections, dict(records=ancestry.records, rowsets=ancestry.rowsets, members=head.members)


def audit_heads(frame, saved, learner, fold):
    """Independent rowset/scaler/cluster audit of only the two new arms."""
    expected = 40 if learner == "mlp" else 20
    if len(saved["members"]) != expected or len(saved["records"]) != expected:
        raise ValueError("new energy fit count differs")
    cluster = frame.set_index("pair_id").cluster_id.to_dict()
    forbidden = set(frame.loc[frame.outer_fold.eq(fold), "cluster_id"])
    for record in saved["records"]:
        sets = {}
        for key in ("fit_rows", "prediction_rows", "stopping_rows"):
            if key in record:
                ids = saved["rowsets"][record[key]]
                if hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest() != record[key] or any(i not in cluster for i in ids):
                    raise ValueError("new energy ancestry identity differs")
                sets[key] = {cluster[i] for i in ids}
        used = sets["fit_rows"] | sets.get("stopping_rows", set())
        if used & (sets["prediction_rows"] | forbidden) or sets["fit_rows"] & sets.get("stopping_rows", set()):
            raise ValueError("new energy gradient/stopping cluster leakage")
        if record["scaler_fit_rows"] != record["fit_rows"]:
            raise ValueError("new energy scaler fitted outside fitting rows")
    return True
