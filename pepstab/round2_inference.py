"""Pickle-free, cached-feature inference for explicit round-two head recipes.

This is a head adapter, not a Boltz extractor or a sequence-only predictor.
No outcome-based choice is made here. Training always requires one named
candidate and an explicit fit budget; exported refits are development artifacts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import sklearn

from .boltz_embeddings import json_bytes, sha256_file
from .mlp import MLPConfig, MLPRegressor
from .nested import inner_folds
from .round2_embedding import head_candidates
from .round2_representations import (
    FEATURE_DIMENSIONS, FEATURE_SCHEMA, feature_protocol, fixed_projection,
    load_contact_map, validate_contact_map,
)

MODEL_SCHEMA = "pepstab-round2-cached-head-v1"
PROCEDURES = {"R4": "ridge", "R5": "mlp"}


def digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _basename(value):
    if not isinstance(value, str) or Path(value).name != value or value in ("", ".", ".."):
        raise ValueError("artifact file must be a basename")
    return value


def _finite(value, shape, name, dtype=None):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "fi" or not np.isfinite(array).all():
        raise ValueError(f"invalid {name}: expected finite numeric shape {shape}")
    if dtype is not None and array.dtype != dtype:
        raise ValueError(f"invalid {name}: expected dtype {dtype}")
    return array


def candidate_config(candidate_id):
    for representation, learner in PROCEDURES.items():
        for candidate in head_candidates(representation, learner):
            if candidate["id"] == candidate_id:
                return candidate
    raise ValueError("candidate must be an explicit frozen R4 ridge or R5 MLP grid member")


def fit_count(candidate_id):
    return 1 if candidate_config(candidate_id)["learner"] == "ridge" else 30


@dataclass
class FeatureBatch:
    """Ordered finite float32 vectors plus verified cache identities/provenance."""

    representation: str
    values: np.ndarray
    pair_ids: list[str]
    input_hashes: list[str]
    contract: dict
    provenance: dict

    def validate(self):
        if self.representation not in PROCEDURES:
            raise ValueError("only R4 and R5 cached representations are supported")
        n = len(self.pair_ids)
        if not n or len(set(self.pair_ids)) != n or any(not isinstance(x, str) or not x for x in self.pair_ids):
            raise ValueError("feature pair_id must be nonempty unique strings")
        if len(self.input_hashes) != n or any(not isinstance(x, str) or not x for x in self.input_hashes):
            raise ValueError("feature input_hash must be a nonempty string per row")
        _finite(self.values, (n, FEATURE_DIMENSIONS[self.representation]), "feature matrix", np.dtype("float32"))
        if self.contract.get("representation") != self.representation:
            raise ValueError("feature representation contract mismatch")
        if self.contract.get("feature_protocol") != feature_protocol():
            raise ValueError("feature protocol mismatch")
        validate_contact_map(self.contract.get("contact_map", {}))
        return self

    def select(self, pair_ids):
        self.validate()
        wanted = [str(x) for x in pair_ids]
        if not wanted or len(set(wanted)) != len(wanted):
            raise ValueError("requested pair_id must be nonempty and unique")
        position = {pid: i for i, pid in enumerate(self.pair_ids)}
        if set(wanted) - set(position):
            raise ValueError("requested pair_id missing from feature cache")
        indices = [position[pid] for pid in wanted]
        return FeatureBatch(self.representation, self.values[indices], wanted,
                            [self.input_hashes[i] for i in indices], self.contract, self.provenance).validate()


def load_feature_batch(path, representation, *, expected_contract=None):
    """Read existing verified compact bundle(s); never accept bare feature arrays.

    A complete production manifest or one compact bundle directory is accepted.
    The model's embedded contract permits inference without the repository's
    original contact-map file. Labels are neither requested nor read.
    """
    if representation not in PROCEDURES:
        raise ValueError("unsupported representation")
    path = Path(path)
    if path.is_dir():
        directories, origin = [path], {}
    else:
        item = json.loads(path.read_text())
        if item.get("status") != "complete" or item.get("completed_rows") != item.get("expected_rows"):
            raise ValueError("production feature manifest is incomplete")
        bundles = item.get("feature_bundles")
        if not isinstance(bundles, list) or not bundles:
            raise ValueError("production manifest must contain feature_bundles")
        directories = []
        for bundle in bundles:
            directory = Path(bundle if isinstance(bundle, str) else bundle["path"])
            directories.append(directory if directory.is_absolute() else path.parent / directory)
        origin = {"manifest_sha256": sha256_file(path), "expected_rows": item["expected_rows"]}
    contract = expected_contract or {"representation": representation, "feature_protocol": feature_protocol(),
                                     "contact_map": load_contact_map(), "projection_applied": False}
    arrays, ids, inputs, sources = [], [], [], []
    for directory in directories:
        complete = json.loads((directory / "complete.json").read_text())
        if complete.get("schema_version") != FEATURE_SCHEMA:
            raise ValueError("unknown compact feature schema")
        data = directory / _basename(complete["array_file"])
        metadata = directory / _basename(complete["metadata_file"])
        if sha256_file(data) != complete["npz_sha256"] or sha256_file(metadata) != complete["metadata_sha256"]:
            raise ValueError("compact feature content hash mismatch")
        info = json.loads(metadata.read_text())
        if info.get("schema_version") != FEATURE_SCHEMA or info.get("npz_sha256") != complete["npz_sha256"]:
            raise ValueError("compact manifest mismatch")
        if info.get("feature_protocol") != contract["feature_protocol"]:
            raise ValueError("feature protocol mismatch")
        if info.get("contact_map_sha256") != contract["contact_map"]["contact_map_sha256"]:
            raise ValueError("contact map mismatch")
        rows = info["rows"]
        if len(rows) != info["n_rows"]:
            raise ValueError("compact row count mismatch")
        with np.load(data, allow_pickle=False) as raw:
            row_ids = raw["pair_id"].astype(str).tolist()
            hashes = raw["input_hash"].astype(str).tolist()
            if row_ids != [str(row["pair_id"]) for row in rows] or hashes != [row["input_hash"] for row in rows]:
                raise ValueError("compact row identity mismatch")
            arrays.append(_finite(raw[representation], (len(rows), FEATURE_DIMENSIONS[representation]),
                                  "compact features", np.dtype("float32")).copy())
        ids.extend(row_ids)
        inputs.extend(hashes)
        sources.append({"npz_sha256": complete["npz_sha256"], "metadata_sha256": complete["metadata_sha256"]})
    if "expected_rows" in origin and origin["expected_rows"] != len(ids):
        raise ValueError("production feature row count mismatch")
    return FeatureBatch(representation, np.concatenate(arrays), ids, inputs, contract,
                        {"origin": origin, "bundles": sources}).validate()


def _scale(X, arrays, prefix):
    # Match StandardScaler's two in-place operations on float32 input exactly.
    result = np.array(X, dtype=np.float32, copy=True)
    result -= arrays[prefix + "mean"].astype(np.float32)
    result /= arrays[prefix + "scale"].astype(np.float32)
    return result


@dataclass
class CachedHead:
    metadata: dict
    arrays: dict[str, np.ndarray]

    def validate(self):
        info = self.metadata
        if info.get("schema_version") != MODEL_SCHEMA or info.get("scope") != "development-refit":
            raise ValueError("unknown model schema or scope")
        candidate = candidate_config(info["candidate"]["id"])
        if candidate != info["candidate"] or digest(info["contract"]) != info["contract_sha256"]:
            raise ValueError("model candidate/contract mismatch")
        contract = info["contract"]
        if contract.get("representation") != candidate["representation"] or contract.get("projection_applied") is not False:
            raise ValueError("model representation mismatch")
        if contract.get("feature_protocol") != feature_protocol():
            raise ValueError("model feature protocol mismatch")
        validate_contact_map(contract["contact_map"])
        n_features = FEATURE_DIMENSIONS[candidate["representation"]]
        members = info["members"]
        if len(members) != fit_count(candidate["id"]):
            raise ValueError("model member count mismatch")
        expected = {"token_projection", "pair_projection"}
        for key, shape in (("token_projection", (384, 8)), ("pair_projection", (128, 4))):
            _finite(self.arrays[key], shape, key, np.dtype("float32"))
            if array_digest(self.arrays[key]) != contract["feature_protocol"]["projections"][key.split("_")[0]]["float32_sha256"]:
                raise ValueError("stored projection hash mismatch")
        for i, member in enumerate(members):
            p = f"m{i}_"
            for key in ("mean", "scale", "var"):
                expected.add(p + key)
                _finite(self.arrays[p + key], (n_features,), p + key)
            if np.any(self.arrays[p + "scale"] <= 0) or np.any(self.arrays[p + "var"] < 0):
                raise ValueError("invalid scaler statistics")
            if candidate["learner"] == "ridge":
                shapes = {"coef": (n_features,), "intercept": ()}
            else:
                if (member.get("stop_fold"), member.get("seed")) != (i // 6, i % 6):
                    raise ValueError("MLP member schedule mismatch")
                shapes = {"w0": (n_features, 64), "b0": (64,), "w1": (64, 1), "b1": (1,), "y_offset": ()}
            for key, shape in shapes.items():
                expected.add(p + key)
                _finite(self.arrays[p + key], shape, p + key)
        if set(self.arrays) != expected:
            raise ValueError("model parameter keys mismatch")
        return self

    def predict(self, batch: FeatureBatch):
        self.validate()
        batch.validate()
        if digest(batch.contract) != self.metadata["contract_sha256"]:
            raise ValueError("prediction feature provenance/representation mismatch")
        predictions = []
        for i in range(len(self.metadata["members"])):
            p = f"m{i}_"
            X = _scale(batch.values, self.arrays, p)
            if self.metadata["candidate"]["learner"] == "ridge":
                value = X @ self.arrays[p + "coef"] + self.arrays[p + "intercept"]
            else:
                hidden = np.maximum(X @ self.arrays[p + "w0"] + self.arrays[p + "b0"], 0.)
                value = (hidden @ self.arrays[p + "w1"] + self.arrays[p + "b1"]).ravel() + float(self.arrays[p + "y_offset"])
            predictions.append(value)
        result = np.mean(predictions, axis=0)
        if not np.isfinite(result).all():
            raise ValueError("nonfinite cached head predictions")
        return pd.DataFrame({"pair_id": batch.pair_ids, "input_hash": batch.input_hashes,
                             "prediction_y_log1p": result})

    def save(self, path):
        self.validate()
        path = Path(path)
        # A completed package is immutable; accidental overwrites fail safely.
        path.mkdir(parents=True, exist_ok=False)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **self.arrays)
        payload = buffer.getvalue()
        metadata = json_bytes(self.metadata)
        (path / "parameters.npz").write_bytes(payload)
        (path / "model.json").write_bytes(metadata)
        (path / "complete.json").write_bytes(json_bytes({"schema_version": MODEL_SCHEMA,
            "array_file": "parameters.npz", "metadata_file": "model.json",
            "npz_sha256": hashlib.sha256(payload).hexdigest(), "metadata_sha256": hashlib.sha256(metadata).hexdigest()}))


def load_head(path):
    path = Path(path)
    complete = json.loads((path / "complete.json").read_text())
    if complete.get("schema_version") != MODEL_SCHEMA:
        raise ValueError("unknown model schema")
    arrays = path / _basename(complete["array_file"])
    metadata = path / _basename(complete["metadata_file"])
    if sha256_file(arrays) != complete["npz_sha256"] or sha256_file(metadata) != complete["metadata_sha256"]:
        raise ValueError("model content hash mismatch")
    with np.load(arrays, allow_pickle=False) as raw:
        parameters = {key: raw[key].copy() for key in raw.files}
    return CachedHead(json.loads(metadata.read_text()), parameters).validate()


def fit_head(batch, training, candidate_id, *, max_fits, authorization, max_epochs=200):
    """Fit one explicitly locked recipe; never inspect assessment outcomes.

    ``max_epochs`` may be reduced only for synthetic verification. Production
    CLI always uses the frozen 200-epoch cap. MLP stopping is peptide-grouped;
    this API does not reproduce HLA/joint assessment procedures.
    """
    batch.validate()
    candidate = candidate_config(candidate_id)
    if candidate["representation"] != batch.representation:
        raise ValueError("candidate representation mismatch")
    count = fit_count(candidate_id)
    if max_fits < count:
        raise ValueError(f"recipe requires {count} fits, exceeds max_fits={max_fits}")
    if not authorization or not isinstance(authorization, dict):
        raise ValueError("explicit training authorization metadata is required")
    required = {"pair_id", "cluster_id", "split", "y_log1p"}
    if not required <= set(training) or training.empty or not training["split"].eq("train").all():
        raise ValueError("only explicitly marked original train rows may fit")
    if training[list(required)].isna().any().any() or training.pair_id.astype(str).duplicated().any():
        raise ValueError("invalid training row identity or missing labels/clusters")
    labels = training.y_log1p.to_numpy()
    _finite(labels, (len(training),), "training labels")
    selected = batch.select(training.pair_id)
    X = selected.values
    folds = inner_folds(training, 5) if candidate["learner"] == "mlp" else None
    if folds is not None and set(folds) != set(range(5)):
        raise ValueError("MLP requires five nonempty peptide-cluster stopping folds")
    arrays = {"token_projection": fixed_projection(384, 8).copy(), "pair_projection": fixed_projection(128, 4).copy()}
    members = []
    schedule = [(None, None)] if folds is None else [(stop, seed) for stop in range(5) for seed in range(6)]
    for i, (stop, seed) in enumerate(schedule):
        mask = np.ones(len(X), dtype=bool) if folds is None else folds != stop
        scaler = StandardScaler().fit(X[mask])
        p = f"m{i}_"
        arrays.update({p + "mean": scaler.mean_.copy(), p + "scale": scaler.scale_.copy(), p + "var": scaler.var_.copy()})
        member = {"n_fit": int(mask.sum()), "n_stop": int((~mask).sum()), "seed": seed,
                  "stop_fold": stop, "gradient_pair_ids_sha256": digest(np.asarray(selected.pair_ids)[mask].tolist()),
                  "stopping_pair_ids_sha256": digest(np.asarray(selected.pair_ids)[~mask].tolist()),
                  "scaler_fit_pair_ids_sha256": digest(np.asarray(selected.pair_ids)[mask].tolist())}
        if folds is None:
            model = Ridge(alpha=candidate["regularization"], solver="lsqr", tol=1e-5, max_iter=2000).fit(scaler.transform(X), labels)
            arrays.update({p + "coef": model.coef_.copy(), p + "intercept": np.asarray(model.intercept_)})
            member["config"] = {"solver": "lsqr", "tol": 1e-5, "max_iter": 2000, "alpha": candidate["regularization"]}
        else:
            cfg = MLPConfig(hidden=(64,), l2=candidate["regularization"], max_epochs=max_epochs, patience=15, seed=seed)
            model = MLPRegressor(cfg).fit(scaler.transform(X[mask]), labels[mask], scaler.transform(X[~mask]), labels[~mask])
            for layer, (weights, biases) in enumerate(zip(model.weights_, model.biases_)):
                arrays[p + f"w{layer}"], arrays[p + f"b{layer}"] = weights.copy(), biases.copy()
            arrays[p + "y_offset"] = np.asarray(model.y_offset_)
            member.update(config=asdict(cfg), best_epoch=model.best_epoch_, epochs=model.n_epochs_)
        members.append(member)
    metadata = {"schema_version": MODEL_SCHEMA, "scope": "development-refit", "candidate": candidate,
                "target": "y_log1p = log1p(thalf_hours); uncalibrated regression output",
                "contract": batch.contract, "contract_sha256": digest(batch.contract), "members": members,
                "training": {"n_rows": len(training), "pair_ids_sha256": digest(selected.pair_ids),
                             "input_hashes_sha256": digest(selected.input_hashes),
                             "cluster_ids_sha256": digest(training.cluster_id.astype(str).tolist()),
                             "labels_sha256": array_digest(labels), "features_sha256": array_digest(X),
                             "feature_provenance": batch.provenance, "authorization": authorization},
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
                "source_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in
                                  ("round2_inference.py", "round2_embedding.py", "round2_representations.py", "mlp.py", "nested.py")}}
    return CachedHead(metadata, arrays).validate()
