"""Bounded, fresh sequence controls for NEXT_RESEARCH_PLAN milestone 1."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .features import build_features
from .mlp import MLPConfig, MLPRegressor
from .nested import inner_folds
from .phase_a import allele_offsets, ranking_pairs, shrunk_allele_means
from .round2_nested import Ancestry, unlabelled, validate_partition


SEEDS = (0, 1)
INNER_FOLDS = 3
FAMILY_ORDER = ("mlp_log", "mlp_centred", "tree_log", "tree_centred", "pairlogit")


def candidates() -> dict[str, list[dict]]:
    """Frozen before assessment; order is the parsimony tie-breaking order."""
    out = {"mlp_log": [{"id": "mlp_log", "learner": "mlp", "target": "log"}],
           "mlp_centred": [
               {"id": f"mlp_centred_k{k}", "learner": "mlp", "target": "centred", "k": k}
               for k in (100, 20)]}
    for family in ("tree_log", "tree_centred", "pairlogit"):
        out[family] = [
            {"id": f"{family}_d{depth}", "learner": "pairlogit" if family == "pairlogit" else "tree",
             "target": "centred" if family == "tree_centred" else "log",
             "depth": depth, "learning_rate": .06, "l2_leaf_reg": 10,
             **({"k": 20} if family == "tree_centred" else {})}
            for depth in (4, 8)]
    return out


def fit_graph() -> dict:
    # Inner recipe MLP=3*2*2=12, tree=3*1*2=6 members.
    # Final MLP=5*2*3=30, final trees=6.
    # Ranker map: 3 crossfit blocks, each 3*2*6 selection + 6 refit.
    per_outer = {"mlp_log": 3 * 12 + 30,
                 "mlp_centred": 3 * 2 * 12 + 30,
                 "tree_log": 3 * 2 * 6 + 6,
                 "tree_centred": 3 * 2 * 6 + 6,
                 "pairlogit": 3 * 2 * 6 + 6 + 3 * (3 * 2 * 6 + 6)}
    return {"per_outer": per_outer, "five_outer_total": 5 * sum(per_outer.values()),
            "mlp_fits": 5 * (per_outer["mlp_log"] + per_outer["mlp_centred"]),
            "rmse_fits": 5 * (per_outer["tree_log"] + per_outer["tree_centred"]),
            "ranker_fits": 5 * per_outer["pairlogit"],
            "selection_folds": 3, "screen_seeds": [0, 1],
            "final_mlp_members": 30, "final_tree_members": 6,
            "repeats_additional_fits": 420,
            "repeat_scope": "two additional locked refits of original MLP, A1, and A4 on all five folds; ranker calibration map held fixed"}


class EnsemblePredictor:
    def __init__(self, population: pd.DataFrame, ancestry: Ancestry,
                 *, threads: int = 2, max_epochs: int = 300,
                 tree_iterations: int = 2000, callback=None, cache_dir=None,
                 ranker_backend=None, ranker_backend_name="local"):
        self.ancestry = ancestry
        self.threads = threads
        self.max_epochs = max_epochs
        self.tree_iterations = tree_iterations
        self.callback = callback
        self.ranker_backend = ranker_backend
        self.ranker_backend_name = ranker_backend_name
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.members: list[dict] = []
        self.last_by_seed: dict[int, np.ndarray] = {}
        self.ids = {int(pid): i for i, pid in enumerate(population.pair_id)}
        self.features = {enc: np.ascontiguousarray(
            build_features(population, "pep_pseudo", enc), dtype=np.float32)
            for enc in ("onehot", "blosum")}
        self.feature_hash = hashlib.sha256(b"".join(
            self.features[e].tobytes() for e in sorted(self.features))).hexdigest()
        self.code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def X(self, frame, encoding):
        return self.features[encoding][[self.ids[int(pid)] for pid in frame.pair_id]]

    def __call__(self, fit, held, candidate, context):
        return self.predict(fit, held, candidate, context)

    def predict(self, fit, held, candidate, context, *, final=False):
        validate_partition(fit, held)
        if any(c in held for c in ("y_log1p", "thalf_hours")):
            raise ValueError("predictor received assessment targets")
        cache_path = None
        if self.cache_dir is not None:
            signature = {"fit_ids": fit.pair_id.tolist(), "y": fit.y_log1p.tolist(),
                         "held_ids": held.pair_id.tolist(), "candidate": candidate,
                         "context": context, "final": final, "threads": self.threads,
                         "max_epochs": self.max_epochs, "iterations": self.tree_iterations,
                         "feature_hash": self.feature_hash, "code_hash": self.code_hash}
            signature["ranker_backend"] = self.ranker_backend_name if candidate["learner"] == "pairlogit" else "local"
            key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
            cache_path = self.cache_dir / f"{key}.json"
            if cache_path.exists():
                saved = json.loads(cache_path.read_text())
                self.ancestry.rowsets.update(saved["rowsets"])
                self.ancestry.records.extend(saved["ancestry"])
                self.members.extend(saved["members"])
                self.last_by_seed = {int(k): np.asarray(v) for k, v in saved["by_seed"].items()}
                return np.asarray(saved["predictions"])
        start_member, start_ancestry = len(self.members), len(self.ancestry.records)
        mlp = candidate["learner"] == "mlp"
        encodings = ("onehot", "blosum") if mlp else ("onehot",)
        seeds = (0, 1, 2) if final and mlp else SEEDS
        n_stop = 5 if final and mlp else INNER_FOLDS
        folds = inner_folds(fit, n_stop)
        remote_members = {}
        if candidate["learner"] == "pairlogit" and self.ranker_backend is not None:
            tasks = []
            columns = ["pair_id", "cluster_id", "allele", "peptide", "hla_pseudoseq", "y_log1p"]
            for stop_fold in range(n_stop):
                for seed in seeds:
                    tasks.append({"context": context, "stop_fold": stop_fold, "seed": seed,
                                  "candidate": candidate, "iterations": self.tree_iterations,
                                  "fit": fit.loc[folds != stop_fold, columns].to_dict("records"),
                                  "stop": fit.loc[folds == stop_fold, columns].to_dict("records"),
                                  "held": held[[c for c in columns if c != "y_log1p"]].to_dict("records")})
            outputs = self.ranker_backend(tasks)
            if len(outputs) != len(tasks):
                raise ValueError("remote backend returned incomplete member coverage")
            remote_members = {(r["stop_fold"], r["seed"]): r for r in outputs}
            if set(remote_members) != {(f, s) for f in range(n_stop) for s in seeds}:
                raise ValueError("remote backend changed or duplicated member identities")
        predictions, seed_predictions = [], {s: [] for s in seeds}
        for enc in encodings:
            X = self.X(fit, enc)
            Xh = self.X(held, enc)
            for fold in range(n_stop):
                mask = folds != fold
                gradient, stop = fit.loc[mask], fit.loc[~mask]
                y = fit.y_log1p.to_numpy(dtype=float)
                offsets = np.zeros(len(fit))
                held_offsets = np.zeros(len(held))
                means, global_mean = {}, None
                if candidate["target"] == "centred":
                    means, global_mean = shrunk_allele_means(gradient.allele, y[mask], candidate["k"])
                    offsets = allele_offsets(fit.allele, means, global_mean)
                    held_offsets = allele_offsets(held.allele, means, global_mean)
                for seed in seeds:
                    started = time.perf_counter()
                    if mlp:
                        cfg = MLPConfig(hidden=(256, 64), l2=1e-5 if enc == "onehot" else 1e-3,
                                        max_epochs=self.max_epochs, patience=15, seed=seed)
                        model = MLPRegressor(cfg).fit(X[mask], (y-offsets)[mask],
                                                     X[~mask], (y-offsets)[~mask])
                        values = model.predict(Xh)
                        if candidate["target"] == "centred":
                            values = values + held_offsets
                        details = {"best_epoch": model.best_epoch_, "epochs": model.n_epochs_,
                                   "config": json.loads(json.dumps(asdict(cfg)))}
                    elif remote_members:
                        remote = remote_members[(fold, seed)]
                        values, details = np.asarray(remote["predictions"]), remote["details"]
                        if remote["pair_ids"] != held.pair_id.tolist():
                            raise ValueError("remote predictions changed row order")
                    else:
                        values, details = self._tree(
                            X[mask], (y-offsets)[mask], gradient.allele,
                            X[~mask], (y-offsets)[~mask], stop.allele,
                            Xh, candidate, seed)
                        values = values + held_offsets
                    predictions.append(values)
                    seed_predictions[seed].append(values)
                    member = {"context": context, "candidate": candidate["id"],
                              "encoding": enc, "seed": seed, "stop_fold": fold,
                              "fit_seconds": (remote_members[(fold, seed)]["fit_seconds"]
                                              if remote_members else time.perf_counter()-started),
                              "n_fit": len(gradient), "n_stop": len(stop),
                              "n_predicted": len(held), "final_recipe": final,
                              "centering_means": means, "centering_global": global_mean,
                              **details}
                    ancestry = self.ancestry.add(role="base_member", fit=gradient,
                                                held=held, stopping=stop,
                                                context=context, candidate=candidate["id"],
                                                encoding=enc, seed=seed, stop_fold=fold)
                    member.update({k: ancestry[k] for k in ("fit_rows", "stopping_rows", "prediction_rows")})
                    self.members.append(member)
                    if self.callback:
                        self.callback(member)
        self.last_by_seed = {s: np.mean(p, axis=0) for s, p in seed_predictions.items()}
        result = np.mean(predictions, axis=0)
        if cache_path is not None:
            records = self.ancestry.records[start_ancestry:]
            keys = {r[k] for r in records for k in ("fit_rows", "stopping_rows", "prediction_rows")}
            saved = {"predictions": result.tolist(),
                     "by_seed": {str(k): v.tolist() for k, v in self.last_by_seed.items()},
                     "members": self.members[start_member:], "ancestry": records,
                     "rowsets": {k: self.ancestry.rowsets[k] for k in keys}}
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(saved))
            temporary.replace(cache_path)
        return result

    def _tree(self, X, y, allele, Xdev, ydev, adev, Xheld, candidate, seed):
        from catboost import CatBoost, CatBoostRegressor, Pool
        ranker = candidate["learner"] == "pairlogit"
        params = {"iterations": self.tree_iterations, "od_type": "Iter", "od_wait": 100,
                  "bootstrap_type": "Bernoulli", "subsample": .8, "rsm": 1.,
                  "random_strength": 1., "border_count": 128,
                  "depth": candidate["depth"], "learning_rate": candidate["learning_rate"],
                  "l2_leaf_reg": candidate["l2_leaf_reg"], "random_seed": seed,
                  "thread_count": self.threads, "verbose": False, "allow_writing_files": False,
                  "loss_function": "PairLogit" if ranker else "RMSE",
                  "eval_metric": "PairLogit" if ranker else "RMSE"}
        if ranker:
            def grouped_pool(features, labels, groups):
                groups = np.asarray(groups, dtype=str)
                order = np.argsort(groups, kind="stable")
                groups, labels = groups[order], labels[order]
                hi, lo, weight = ranking_pairs(groups, labels)
                if not len(hi):
                    raise ValueError("no untied within-allele ranking pairs")
                return Pool(features[order], labels, group_id=groups,
                            pairs=np.column_stack([hi, lo]), pairs_weight=weight/weight.mean())
            model = CatBoost(params)
            model.fit(grouped_pool(X, y, allele), eval_set=grouped_pool(Xdev, ydev, adev),
                      use_best_model=True)
        else:
            model = CatBoostRegressor(**params)
            model.fit(X, y, eval_set=(Xdev, ydev), use_best_model=True)
        return np.asarray(model.predict(Xheld)), {
            "best_epoch": int(model.get_best_iteration()), "epochs": int(model.tree_count_),
            "config": params}


def ranker_task(task: dict) -> dict:
    """Pure single-member backend; the local and Modal worker use identical code."""
    import platform
    from threadpoolctl import threadpool_info, threadpool_limits
    started = time.perf_counter()
    fit, stop, held = (pd.DataFrame(task[k]) for k in ("fit", "stop", "held"))
    validate_partition(fit, stop)
    validate_partition(fit, held)
    validate_partition(stop, held)
    if "y_log1p" in held or task["candidate"]["learner"] != "pairlogit":
        raise ValueError("invalid ranker backend payload")
    # No full predictor object is needed: _tree depends only on these two values.
    from types import SimpleNamespace
    settings = SimpleNamespace(threads=2, tree_iterations=task["iterations"])
    X, Xstop, Xheld = [np.ascontiguousarray(build_features(f, "pep_pseudo", "onehot"), dtype=np.float32)
                      for f in (fit, stop, held)]
    with threadpool_limits(limits=2):
        values, details = EnsemblePredictor._tree(
            settings, X, fit.y_log1p.to_numpy(), fit.allele,
            Xstop, stop.y_log1p.to_numpy(), stop.allele, Xheld,
            task["candidate"], task["seed"])
    details.update({"platform": platform.platform(), "threadpools": threadpool_info()})
    return {"stop_fold": task["stop_fold"], "seed": task["seed"],
            "pair_ids": held.pair_id.tolist(), "predictions": values.tolist(),
            "fit_seconds": time.perf_counter()-started, "details": details}
