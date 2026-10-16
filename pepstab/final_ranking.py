"""Frozen matched MSE + within-allele RankNet follow-up on development rows.

No original test access, affine mapping, allele resampling, ranking stopping,
or automatic extension is part of this bounded experiment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.preprocessing import StandardScaler

from .mlp import MLPConfig
from .nested import inner_folds
from .phase_a import RankingMLPRegressor
from .round2_nested import Ancestry, config_digest, validate_partition

REPRESENTATIONS = ("R0", "R5")
LAMBDAS = (.1, .3, 1.)
L2 = (1., .01, .0001)
INNER_AXES = ((0, 0), (1, 1))
FINAL_AXES = tuple((stop, seed) for stop in range(5) for seed in range(6))


def candidates(rep):
    if rep not in REPRESENTATIONS:
        raise ValueError("only frozen R0/R5 representations allowed")
    return [dict(id=f"{rep}_rank_{weight:g}_l2_{l2:g}", representation=rep,
                 learner="mlp", regularization=l2, rank_weight=weight)
            for weight in LAMBDAS for l2 in L2]


def one_se(rows):
    """First in supplied frozen order within one fold-SE of best mean."""
    if not rows or any(not np.isfinite([r["mean"], r["se"]]).all() for r in rows):
        raise ValueError("finite selection scores required")
    best = max(rows, key=lambda r: r["mean"])
    return next(r for r in rows if r["mean"] >= best["mean"] - best["se"])


def deploy_choice(mse, ranked):
    rows = [dict(procedure="mse", **selected_score(mse)),
            dict(procedure="ranked", **selected_score(ranked))]
    return dict(procedure=one_se(rows)["procedure"], scores=rows,
                rule="one fold-SE; MSE first; inner outcomes only")


def selected_score(choice):
    return next(r for r in choice["scores"] if r["candidate"] == choice["candidate"]["id"])


def array_hash(array):
    if isinstance(array, bytes):
        return hashlib.sha256(array).hexdigest()
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def ordered_rows(frame):
    return array_hash(frame.pair_id.to_numpy(dtype="<i8"))


def member_config(candidate, seed, max_epochs=200):
    return MLPConfig(hidden=(64,), l2=candidate["regularization"], lr=.001,
                     batch_size=256, max_epochs=max_epochs, patience=15, seed=seed)


def canonical_cache_key(fit, held, candidate, context, feature_hash, source_hash,
                        *, final=False):
    """Exact historical FeatureHead signature; no legacy file is modified."""
    signature = dict(fit=fit.pair_id.tolist(), y=fit.y_log1p.tolist(),
                     held=held.pair_id.tolist(), candidate=candidate, context=context,
                     final=final, feature_hash=feature_hash, max_epochs=200,
                     final_seeds=tuple(range(6)), code=source_hash)
    return config_digest(signature)


def audit_cached_call(item, fit, held, candidate, context, population, *, final=False):
    """Require every cached member's actual gradient/stop/predict/scaler roles."""
    axes = FINAL_AXES if final else INNER_AXES
    if len(item["members"]) != len(axes) or len(item["ancestry"]) != len(axes):
        raise ValueError("canonical cache member count changed")
    ancestry = Ancestry()
    ancestry.rowsets.update(item["rowsets"])
    ancestry.records.extend(item["ancestry"])
    # Verify rowsets are genuinely content-addressed, not just trusted names.
    for key, ids in ancestry.rowsets.items():
        if ids != sorted(set(ids)) or array_hash(np.asarray(ids, dtype="<i8")) != key:
            raise ValueError("canonical rowset content/hash changed")
    folds = inner_folds(fit, 5)
    for (stop, seed), member, record in zip(axes, item["members"], item["ancestry"]):
        gradient, stopping = fit.loc[folds != stop], fit.loc[folds == stop]
        expected = dict(role="mlp_head", fit_rows=ancestry.rows(gradient),
                        prediction_rows=ancestry.rows(held), stopping_rows=ancestry.rows(stopping),
                        context=context, candidate=candidate["id"], seed=seed, stop_fold=stop,
                        scaler_fit_rows=ancestry.rows(gradient))
        if record != expected:
            raise ValueError("canonical member ancestry/settings changed")
        for key, value in dict(context=context, candidate=candidate["id"], learner="mlp",
                               seed=seed, stop_fold=stop, n_fit=len(gradient),
                               n_stop=len(stopping)).items():
            if member.get(key) != value:
                raise ValueError(f"canonical member identity changed: {key}")
    ancestry.audit(population, held)
    predictions = np.asarray(item["predictions"])
    by_seed = {int(k): np.asarray(v) for k, v in item["by_seed"].items()}
    if predictions.shape != (len(held),) or not np.isfinite(predictions).all():
        raise ValueError("invalid canonical prediction")
    if set(by_seed) != ({0, 1} if not final else set(range(6))):
        raise ValueError("canonical seed coverage changed")
    if any(v.shape != predictions.shape or not np.isfinite(v).all() for v in by_seed.values()):
        raise ValueError("invalid canonical member prediction")
    np.testing.assert_allclose(predictions, np.mean(list(by_seed.values()), axis=0),
                               rtol=2e-6, atol=2e-6)
    return ancestry


def atomic_json(path, item):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(item, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


class RankedHead:
    """Complete member receipts; an unfinished launched fit is never retried.

    New calls save all member predictions and gradient-only scaler statistics,
    not merely an ensemble aggregate. Completed receipts allow safe continuation.
    """
    def __init__(self, population, features, ancestry, directory, protocol_hash,
                 *, max_epochs=200):
        self.population = population
        self.position = {int(pid): i for i, pid in enumerate(population.pair_id)}
        if len(self.position) != len(population):
            raise ValueError("duplicate population identities")
        self.features = {rep: np.asarray(x, dtype=np.float32) for rep, x in features.items()}
        if any(x.ndim != 2 or len(x) != len(population) or not np.isfinite(x).all()
               for x in self.features.values()):
            raise ValueError("invalid features")
        self.feature_hashes = {rep: array_hash(x) for rep, x in self.features.items()}
        self.ancestry, self.directory, self.protocol_hash = ancestry, Path(directory), protocol_hash
        self.max_epochs, self.members, self.calls = max_epochs, [], []

    def X(self, frame, rep):
        return self.features[rep][[self.position[int(pid)] for pid in frame.pair_id]]

    def __call__(self, fit, held, candidate, context):
        return self.predict(fit, held, candidate, context)

    def predict(self, fit, held, candidate, context, *, final=False):
        validate_partition(fit, held)
        if any(key in held for key in ("y_log1p", "thalf_hours")):
            raise ValueError("assessment labels supplied to ranked head")
        if candidate not in candidates(candidate["representation"]):
            raise ValueError("candidate outside frozen ranking grid")
        rep = candidate["representation"]
        signature = dict(protocol_hash=self.protocol_hash, fit=ordered_rows(fit),
                         held=ordered_rows(held), y=array_hash(fit.y_log1p.to_numpy()),
                         candidate=candidate, context=context, final=final,
                         features=self.feature_hashes[rep], max_epochs=self.max_epochs)
        key = config_digest(signature)
        directory = self.directory / key
        directory.mkdir(parents=True, exist_ok=True)
        folds = inner_folds(fit, 5)
        predictions, call_members, records = [], [], []
        X, Xh = self.X(fit, rep), self.X(held, rep)
        for stop, seed in (FINAL_AXES if final else INNER_AXES):
            gradient, stopping = fit.loc[folds != stop], fit.loc[folds == stop]
            stem = directory / f"stop_{stop}_seed_{seed}"
            receipt = stem.with_suffix(".json")
            array_path = stem.with_suffix(".npz")
            launch_path = stem.with_suffix(".launched.json")
            identity = dict(signature=signature, stop_fold=stop, seed=seed,
                            gradient_order=ordered_rows(gradient), stopping_order=ordered_rows(stopping),
                            gradient_targets=array_hash(gradient.y_log1p.to_numpy()),
                            stopping_targets=array_hash(stopping.y_log1p.to_numpy()))
            if receipt.exists():
                member = json.loads(receipt.read_text())
                if member["identity"] != identity or array_hash(array_path.read_bytes()) != member["npz_sha256"]:
                    raise ValueError("completed ranking member identity/hash changed")
                with np.load(array_path, allow_pickle=False) as arrays:
                    values = arrays["predictions"].copy()
            else:
                # Exclusive durable launch receipt is also an individual fit lock.
                with launch_path.open("x") as handle:
                    json.dump(dict(identity=identity, status="launched"), handle)
                started = time.perf_counter()
                mask = folds != stop
                scaler = StandardScaler().fit(X[mask])
                model = RankingMLPRegressor(member_config(candidate, seed, self.max_epochs),
                                            rank_weight=candidate["rank_weight"], dev_metric="mse")
                model.fit(scaler.transform(X[mask]), fit.y_log1p.to_numpy()[mask],
                          scaler.transform(X[~mask]), fit.y_log1p.to_numpy()[~mask],
                          allele=gradient.allele.to_numpy(), allele_dev=stopping.allele.to_numpy())
                values = model.predict(scaler.transform(Xh))
                tmp = array_path.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, predictions=values, scaler_mean=scaler.mean_,
                                    scaler_scale=scaler.scale_, scaler_var=scaler.var_,
                                    dev_mse_curve=np.asarray(model.dev_curve_))
                tmp.replace(array_path)
                member = dict(identity=identity, context=context, candidate=candidate["id"],
                              seed=seed, stop_fold=stop, n_fit=len(gradient), n_stop=len(stopping),
                              n_features=X.shape[1], best_epoch=model.best_epoch_, epochs=model.n_epochs_,
                              fit_seconds=time.perf_counter()-started, npz_sha256=array_hash(array_path.read_bytes()),
                              gradient_rows=self.ancestry.rows(gradient), stopping_rows=self.ancestry.rows(stopping),
                              prediction_rows=self.ancestry.rows(held), scaler_rows=self.ancestry.rows(gradient))
                atomic_json(receipt, member)
            if values.shape != (len(held),) or not np.isfinite(values).all():
                raise ValueError("invalid ranked member predictions")
            record = self.ancestry.add(role="ranking_member", fit=gradient, held=held, stopping=stopping,
                                       context=context, candidate=candidate["id"], seed=seed, stop_fold=stop,
                                       scaler_fit_rows=self.ancestry.rows(gradient))
            records.append(record)
            predictions.append(values)
            self.members.append(member)
            call_members.append(dict(receipt=str(receipt), arrays=str(array_path)))
        result = np.mean(predictions, axis=0)
        call = dict(signature=signature, members=call_members, predictions=result.tolist(),
                    prediction_pair_ids=held.pair_id.tolist(), ancestry=records)
        atomic_json(directory / "completed.json", call)
        self.calls.append(call)
        return result
