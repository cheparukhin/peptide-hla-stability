"""Matched, label-isolated ridge/MLP heads for frozen representations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .mlp import MLPConfig, MLPRegressor
from .nested import inner_folds
from .round2_nested import Ancestry, validate_partition


REPRESENTATIONS = ("R0", "R1", "R2", "R3", "R4", "R5", "R5_shuffled", "R5_position9x34",
                   "R3_pooled", "R5_position9x34_pooled")
EMBEDDING_PRIMARY = ("R2", "R3", "R4", "R5")


def head_candidates(representation, learner):
    # Strongest regularisation first is the frozen parsimony order.
    grid = (10000., 100., 1.) if learner == "ridge" else (1., .01, .0001)
    return [{"id": f"{representation}_{learner}_{value:g}", "representation": representation,
             "learner": learner, "regularization": value} for value in grid]


class FeatureHead:
    """Each call fits only on its supplied partition; assessment targets forbidden.

    Inner MLP calls average two seeds, each early-stopped on a different fifth of
    the fitting clusters. Final neural procedures all average 5 stop folds x 6
    seeds = 30 members, including R0/R1 and all positional controls. Ridge is
    deterministic and is fitted once (duplicating seeds would add no variation).
    """
    def __init__(self, population, features, ancestry: Ancestry, *, cache_dir=None,
                 max_epochs=200, final_seeds=tuple(range(6)), callback=None):
        if population.pair_id.duplicated().any():
            raise ValueError("duplicate feature row identity")
        self.position = {int(pid): i for i, pid in enumerate(population.pair_id)}
        self.features = {k: np.asarray(v, dtype=np.float32) for k, v in features.items()}
        for key, array in self.features.items():
            if array.ndim != 2 or len(array) != len(population) or not np.isfinite(array).all():
                raise ValueError(f"invalid feature matrix {key}")
        self.feature_hashes = {key: hashlib.sha256(x.tobytes()).hexdigest() for key,x in self.features.items()}
        self.ancestry = ancestry
        self.members = []
        self.last_by_seed = {}
        self.max_epochs = max_epochs
        self.final_seeds = final_seeds
        self.callback = callback
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def X(self, frame, representation):
        return self.features[representation][[self.position[int(pid)] for pid in frame.pair_id]]

    def __call__(self, fit, held, candidate, context):
        return self.predict(fit, held, candidate, context)

    def predict(self, fit, held, candidate, context, *, final=False):
        validate_partition(fit, held)
        if any(k in held for k in ("y_log1p", "thalf_hours")):
            raise ValueError("head received assessment labels")
        rep = candidate["representation"]
        cache_path = None
        if self.cache_dir:
            signature = {"fit": fit.pair_id.tolist(), "y": fit.y_log1p.tolist(),
                         "held": held.pair_id.tolist(), "candidate": candidate,
                         "context": context, "final": final,
                         "feature_hash": self.feature_hashes[rep],
                         "max_epochs": self.max_epochs, "final_seeds": self.final_seeds,
                         "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
            cache_path = self.cache_dir / f"{key}.json"
            if cache_path.exists():
                item = json.loads(cache_path.read_text())
                self.ancestry.rowsets.update(item["rowsets"])
                self.ancestry.records.extend(item["ancestry"])
                self.members.extend(item["members"])
                self.last_by_seed = {int(k): np.asarray(v) for k, v in item.get("by_seed", {}).items()}
                return np.asarray(item["predictions"])
        a_start, m_start = len(self.ancestry.records), len(self.members)
        X, Xh = self.X(fit, rep), self.X(held, rep)
        y = fit.y_log1p.to_numpy()
        predictions = []
        by_seed = {}
        if candidate["learner"] == "ridge":
            started = time.perf_counter()
            scale = StandardScaler().fit(X)
            model = Ridge(alpha=candidate["regularization"], solver="lsqr", tol=1e-5, max_iter=2000)
            model.fit(scale.transform(X), y)
            predictions.append(model.predict(scale.transform(Xh)))
            self.ancestry.add(role="ridge_head", fit=fit, held=held, context=context,
                              candidate=candidate["id"], scaler_fit_rows=self.ancestry.rows(fit))
            self.members.append({"context": context, "candidate": candidate["id"],
                                 "learner": "ridge", "fit_seconds": time.perf_counter()-started,
                                 "n_fit": len(fit), "n_stop": 0, "n_features": X.shape[1],
                                 "iterations": np.asarray(model.n_iter_).tolist(), "seed": None})
        elif candidate["learner"] == "mlp":
            stop_folds = inner_folds(fit, 5)
            axes = [(stop, seed) for stop in range(5) for seed in self.final_seeds] if final else [(0, 0), (1, 1)]
            for stop, seed in axes:
                gradient, stopping = fit.loc[stop_folds != stop], fit.loc[stop_folds == stop]
                mask = stop_folds != stop
                started = time.perf_counter()
                scale = StandardScaler().fit(X[mask])
                cfg = MLPConfig(hidden=(64,), l2=candidate["regularization"], max_epochs=self.max_epochs,
                                patience=15, seed=seed)
                model = MLPRegressor(cfg).fit(scale.transform(X[mask]), y[mask],
                                             scale.transform(X[~mask]), y[~mask])
                values = model.predict(scale.transform(Xh))
                predictions.append(values)
                by_seed.setdefault(seed, []).append(values)
                self.ancestry.add(role="mlp_head", fit=gradient, held=held, stopping=stopping,
                                  context=context, candidate=candidate["id"], seed=seed, stop_fold=stop,
                                  scaler_fit_rows=self.ancestry.rows(gradient))
                self.members.append({"context": context, "candidate": candidate["id"],
                                     "learner": "mlp", "fit_seconds": time.perf_counter()-started,
                                     "n_fit": len(gradient), "n_stop": len(stopping),
                                     "n_features": X.shape[1], "seed": seed, "stop_fold": stop,
                                     "best_epoch": model.best_epoch_, "epochs": model.n_epochs_})
                if self.callback:
                    self.callback(self.members[-1])
        else:
            raise ValueError("unknown learner")
        result = np.mean(predictions, axis=0)
        self.last_by_seed = {seed: np.mean(values, axis=0) for seed, values in by_seed.items()}
        if not np.isfinite(result).all():
            raise ValueError("nonfinite head output")
        if cache_path:
            records = self.ancestry.records[a_start:]
            rowsets = {r[key] for r in records for key in ("fit_rows", "prediction_rows", "stopping_rows") if key in r}
            item = {"predictions": result.tolist(), "members": self.members[m_start:],
                    "ancestry": records, "rowsets": {key: self.ancestry.rowsets[key] for key in rowsets},
                    "by_seed": {seed: values.tolist() for seed, values in self.last_by_seed.items()}}
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(item, allow_nan=False))
            tmp.replace(cache_path)
        return result
