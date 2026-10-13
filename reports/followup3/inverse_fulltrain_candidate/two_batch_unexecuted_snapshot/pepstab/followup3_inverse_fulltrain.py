"""Three-arm ORIGINAL-train density followup; independent draft scope."""
from __future__ import annotations

import hashlib
import numpy as np
from . import followup3_inverse as prior

ARMS = ("R5", "R5_P", "R5_PC")


def full_panel(splits, folds, cohort):
    keys = ["allele", "peptide"]
    if splits.duplicated(keys).any() or cohort.duplicated(keys).any() or folds.pair_id.duplicated().any():
        raise ValueError("unique natural keys/fold identities required")
    train = splits.loc[splits.split.eq("train"), keys + ["pair_id", "cluster_id"]]
    if set(train.pair_id) != set(folds.pair_id):
        raise ValueError("folds must cover exactly ORIGINAL train")
    train = train.merge(folds[["pair_id", "cluster_id", "outer_fold"]], on=["pair_id", "cluster_id"], validate="one_to_one")
    train = train.merge(cohort[keys + ["complex_id", "split"]], on=keys, validate="one_to_one")
    if len(train) != int(splits.split.eq("train").sum()) or not train.split.eq("train").all():
        raise ValueError("complete ORIGINAL train structure metadata required")
    if set(train.outer_fold) != set(range(5)) or train.groupby("cluster_id").outer_fold.nunique().max() != 1:
        raise ValueError("five inherited whole-cluster folds required")
    slug = train.allele.str.replace("*", "_", regex=False).str.replace(":", "_", regex=False).str.replace("(", "_", regex=False).str.replace(")", "", regex=False)
    train["folder"] = "stage4c/ectodomain-20261004/boltz2/production/" + slug + "/" + train.complex_id
    return train.sort_values(keys).loc[:, prior.PANEL_COLUMNS].reset_index(drop=True)


def extraction_batches(panel, cached_keys):
    """Interleave SHA-ranked rows within alleles, then alternate disjoint jobs."""
    keys = ["allele", "peptide"]
    panel_keys = set(panel[keys].itertuples(index=False, name=None))
    if not set(cached_keys).issubset(panel_keys):
        raise ValueError("cache has keys outside new cohort")
    remaining = panel.loc[[key not in cached_keys for key in panel[keys].itertuples(index=False, name=None)]].copy()
    remaining["rank_hash"] = [prior.digest([prior.PANEL_SEED, a, p]) for a, p in remaining[keys].itertuples(index=False, name=None)]
    remaining = remaining.sort_values(["allele", "rank_hash"])
    remaining["rank_in_allele"] = remaining.groupby("allele").cumcount()
    remaining = remaining.sort_values(["rank_in_allele", "allele"]).reset_index(drop=True)
    return [remaining.iloc[i::2].loc[:, prior.PANEL_COLUMNS].reset_index(drop=True) for i in (0, 1)]


def feature_arms(base, scored):
    r5 = np.asarray(base["R5"], dtype=float)
    p = scored[[f"P_pos_{i}" for i in range(1, 10)]].to_numpy(float)
    c = scored[[f"C_pos_{i}" for i in range(1, 10)]].to_numpy(float)
    arrays = dict(R5=r5, R5_P=np.column_stack([r5, p]), R5_PC=np.column_stack([r5, p, c]))
    if any(len(x) != len(scored) or not np.isfinite(x).all() for x in arrays.values()):
        raise ValueError("complete finite matched feature rows required")
    return arrays


def assess_heads(frame, arms, learner, fold, head_factory=None):
    from . import nested
    from .round2_embedding import FeatureHead
    from .followup3_energy import Float64RidgeHead
    from .round2_nested import Ancestry, unlabelled
    if tuple(arms) != ARMS or learner not in ("mlp", "ridge") or fold not in range(5):
        raise ValueError("fixed three-arm scope required")
    if not frame.split.eq("train").all() or set(frame.outer_fold) != set(range(5)) or frame.groupby("cluster_id").outer_fold.nunique().max() != 1:
        raise ValueError("ORIGINAL train inherited whole-cluster folds required")
    if not np.isfinite(frame.y_log1p).all():
        raise ValueError("finite fitting labels required")
    ancestry = Ancestry()
    implementation = FeatureHead if learner == "mlp" else Float64RidgeHead
    head = (implementation if head_factory is None else head_factory)(frame, arms, ancestry)
    fitting, held = frame.loc[frame.outer_fold.ne(fold)], frame.loc[frame.outer_fold.eq(fold)]
    inner = nested.inner_folds(fitting, 3)
    grid = (1., .01, .0001) if learner == "mlp" else (10000., 100., 1.)
    predictions, selections = {}, []
    for arm in ARMS:
        choices = []
        for regularization in grid:
            candidate = dict(id=f"{arm}_{learner}_{regularization:g}", representation=arm, learner=learner, regularization=regularization)
            inner_pred = np.full(len(fitting), np.nan)
            for i in range(3):
                inner_pred[inner == i] = head.predict(fitting.loc[inner != i], unlabelled(fitting.loc[inner == i]), candidate,
                    f"inverse_fulltrain/{learner}/{fold}/{arm}/{regularization:g}/inner/{i}", final=False)
            score, eligible = prior.panel_metric(fitting, inner_pred, 20)
            choices.append(dict(regularization=regularization, score=score, n_eligible=len(eligible), inner_predictions=inner_pred.tolist()))
        # Descending grids make exact ties select the strongest penalty.
        chosen = max(choices, key=lambda item: item["score"])
        candidate = dict(id=f"{arm}_{learner}_{chosen['regularization']:g}", representation=arm, learner=learner, regularization=chosen["regularization"])
        predictions[arm] = head.predict(fitting, unlabelled(held), candidate, f"inverse_fulltrain/{learner}/{fold}/{arm}/final", final=False)
        selections.append(dict(outer_fold=fold, arm=arm, fitting_pair_ids=fitting.pair_id.tolist(), selected_regularization=chosen["regularization"], choices=choices))
    if len(head.members) != (60 if learner == "mlp" else 30) or any(not np.isfinite(x).all() for x in predictions.values()):
        raise ValueError("fixed complete fit graph differs")
    dtype = np.float32 if learner == "mlp" else np.float64
    hashes = {a:hashlib.sha256(np.asarray(x, dtype=dtype).tobytes()).hexdigest() for a, x in arms.items()}
    return held, predictions, selections, dict(records=ancestry.records, rowsets=ancestry.rowsets, members=head.members, feature_sha256=hashes)


def audit_heads(frame, saved, selections, learner, fold):
    expected = 60 if learner == "mlp" else 30
    if len(saved["members"]) != expected or len(saved["records"]) != expected or [s["arm"] for s in selections] != list(ARMS):
        raise ValueError("fit count/arm set differs")
    cluster = frame.set_index("pair_id").cluster_id.to_dict()
    forbidden = set(frame.loc[frame.outer_fold.eq(fold), "cluster_id"])
    for record in saved["records"]:
        sets = {}
        if record["role"] != ("mlp_head" if learner == "mlp" else "ridge_head") or (learner == "mlp" and "stopping_rows" not in record):
            raise ValueError("learner/stopping ancestry differs")
        if int(record["context"].split("/")[2]) != fold:
            raise ValueError("ancestry fold differs")
        for key in ("fit_rows", "prediction_rows", "stopping_rows"):
            if key in record:
                ids = saved["rowsets"][record[key]]
                if hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest() != record[key] or any(i not in cluster for i in ids):
                    raise ValueError("rowset identity differs")
                sets[key] = {cluster[i] for i in ids}
        fit, stop = sets["fit_rows"], sets.get("stopping_rows", set())
        if fit & stop or (fit | stop) & (sets["prediction_rows"] | forbidden) or record["scaler_fit_rows"] != record["fit_rows"]:
            raise ValueError("held/stopping/scaler cluster leakage")
    fitting = frame.loc[frame.outer_fold.ne(fold)].set_index("pair_id")
    grid = [1., .01, .0001] if learner == "mlp" else [10000., 100., 1.]
    for selection in selections:
        if selection["outer_fold"] != fold or set(selection["fitting_pair_ids"]) != set(fitting.index):
            raise ValueError("selection fitting population differs")
        if [c["regularization"] for c in selection["choices"]] != grid:
            raise ValueError("selection grid differs")
        fit = fitting.loc[selection["fitting_pair_ids"]]
        scores = [prior.panel_metric(fit, c["inner_predictions"], 20)[0] for c in selection["choices"]]
        if not np.allclose(scores, [c["score"] for c in selection["choices"]], atol=1e-12, rtol=0) or selection["selected_regularization"] != grid[int(np.argmax(scores))]:
            raise ValueError("independent inner score/selection replay differs")
    return True


def fixed_query_table(frame, old_values, full_values, learner):
    """Descriptive points on the identical old queries and fixed allele panels."""
    import pandas as pd
    from . import evaluation as ev
    if frame.duplicated(["allele", "peptide"]).any() or tuple(old_values) != ARMS or tuple(full_values) != ARMS:
        raise ValueError("same unique query keys and three arms required")
    if any(np.asarray(x).shape != (len(frame),) or not np.isfinite(x).all() for x in list(old_values.values()) + list(full_values.values())):
        raise ValueError("complete finite matched fixed-query vectors required")
    rows = []
    for minimum in (20, 50):
        eligible = ev.eligible_alleles(frame.allele, frame.y_log1p, min_rows=minimum)
        if not eligible: raise ValueError("original fixed allele panel empty")
        scores = []
        for vectors in (old_values, full_values):
            scores.append({arm:float(np.median(ev.panel_spearman(ev.per_allele_spearman(frame.allele, frame.y_log1p, vectors[arm], eligible), eligible))) for arm in ARMS})
        entries = [("point", a, scores[0][a], scores[1][a]) for a in ARMS]
        for name, baseline, candidate in (("net", "R5", "R5_PC"), ("context", "R5_P", "R5_PC"), ("peptide", "R5", "R5_P")):
            entries.append(("increment", name, scores[0][candidate]-scores[0][baseline], scores[1][candidate]-scores[1][baseline]))
        for kind, name, old, new in entries:
            rows.append(dict(learner=learner, min_rows=minimum, query_rows=len(frame), eligible_alleles=len(eligible),
                panel_alleles="|".join(eligible), kind=kind, comparison=name, original_subset=old, fulltrained_same_queries=new,
                old_to_new_change=new-old, scope="descriptive fixed queries; combined source coverage/selection/stopping change; no causal size effect or CI"))
    return pd.DataFrame(rows)
