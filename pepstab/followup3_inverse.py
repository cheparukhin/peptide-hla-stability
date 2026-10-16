"""Bounded, train-only inverse-folding complementarity study.

ProteinMPNN measures sequence/backbone compatibility, not kinetics or pose
accuracy. Conditional and peptide-only scores use identical peptide decoding
permutations. All supervised fits and uncertainty belong on Modal.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ARMS = ("R0", "R0_P", "R0_PC", "R5", "R5_P", "R5_PC")
ALPHAS = (10000., 100., 1.)  # exact score ties prefer stronger regularization
CHECKPOINT_SHA256 = "c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd"
N_ORDERS = 16
SEED = 0
PANEL_SEED = "followup3-inverse-20261013"
PANEL_COLUMNS = ("allele", "peptide", "cluster_id", "outer_fold", "complex_id", "folder")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def select_panel(splits, folds, cohort, per_allele=64):
    """Label-blind natural-key sample, inheriting full ORIGINAL-train folds."""
    if per_allele != 64:
        raise ValueError("frozen panel cap is 64 per allele")
    keys = ["allele", "peptide"]
    if splits.duplicated(keys).any() or cohort.duplicated(keys).any():
        raise ValueError("natural keys must be unique")
    train = splits.loc[splits.split.eq("train"), keys + ["pair_id", "cluster_id"]].copy()
    if folds.pair_id.duplicated().any() or set(folds.pair_id) != set(train.pair_id):
        raise ValueError("fold manifest must cover exactly original training rows")
    train = train.merge(folds[["pair_id", "cluster_id", "outer_fold"]], on=["pair_id", "cluster_id"], validate="one_to_one")
    if len(train) != int(splits.split.eq("train").sum()) or not set(train.outer_fold).issubset(set(range(5))):
        raise ValueError("inherited training fold identity mismatch")
    if train.groupby("cluster_id").outer_fold.nunique().max() != 1:
        raise ValueError("outer-fold peptide cluster leakage")
    train = train.merge(cohort[keys + ["complex_id", "split"]], on=keys, validate="one_to_one")
    if not train.split.eq("train").all() or len(train) != int(splits.split.eq("train").sum()):
        raise ValueError("production cohort does not cover original training rows")
    train["sample_hash"] = [digest([PANEL_SEED, a, p]) for a, p in train[keys].itertuples(index=False, name=None)]
    panel = train.sort_values(["allele", "sample_hash"]).groupby("allele", sort=True).head(per_allele).copy()
    slug = panel.allele.str.replace("*", "_", regex=False).str.replace(":", "_", regex=False).str.replace("(", "_", regex=False).str.replace(")", "", regex=False)
    panel["folder"] = "stage4c/ectodomain-20261004/boltz2/production/" + slug + "/" + panel.complex_id
    return panel.sort_values(keys).loc[:, PANEL_COLUMNS].reset_index(drop=True)


def peptide_orders(length=9, n_orders=N_ORDERS, seed=SEED):
    if length != 9 or n_orders != N_ORDERS or seed != SEED:
        raise ValueError("frozen scoring protocol requires nine residues,16 orders,seed0")
    rng = np.random.default_rng(seed)
    return np.asarray([rng.permutation(length) for _ in range(n_orders)], dtype=np.int64)


def decoding_orders(peptide_indices, length, permutations):
    """All visible context first, followed by the SAME peptide permutations."""
    peptide_indices = np.asarray(peptide_indices, dtype=int)
    if len(peptide_indices) != 9 or len(np.unique(peptide_indices)) != 9 or np.any(peptide_indices < 0) or np.any(peptide_indices >= length):
        raise ValueError("exactly nine distinct peptide indices required")
    permutations = np.asarray(permutations, dtype=int)
    if permutations.shape != (N_ORDERS, 9) or any(set(p) != set(range(9)) for p in permutations):
        raise ValueError("sixteen complete peptide permutations required")
    visible = np.setdiff1d(np.arange(length), peptide_indices)
    return np.concatenate([np.tile(visible, (len(permutations), 1)), peptide_indices[permutations]], axis=1)


def _score_context(cx, permutations, batch_rows=5):
    """Teacher-forced forward scoring with explicit common peptide order."""
    import torch
    from . import inverse_folding as inverse
    if str(inverse.MPNN_DIR) not in sys.path:
        sys.path.insert(0, str(inverse.MPNN_DIR))
    from protein_mpnn_utils import tied_featurize
    model, _, _ = inverse.load_model()
    proto = cx.mpnn_dict()
    feats = tied_featurize([proto], torch.device("cpu"), {proto["name"]: ([cx.peptide_chain], cx.context_chains)})
    X, S, mask, chain_M = feats[0], feats[1], feats[2], feats[4]
    cenc, fixed, ridx = feats[5], feats[10], feats[12]
    pep = np.flatnonzero((chain_M * fixed * mask)[0].numpy() > 0)
    if "".join(inverse.ALPHABET[i] for i in S[0, pep].numpy()) != cx.peptide:
        raise ValueError("native peptide sequence differs from scoring tensor")
    orders = torch.as_tensor(decoding_orders(pep, X.shape[1], permutations))
    gathered = []
    with torch.no_grad():
        for start in range(0, len(orders), batch_rows):
            order = orders[start:start + batch_rows]
            size = len(order)
            repeat = lambda x: x.repeat((size,) + (1,) * (x.ndim - 1))
            sb = repeat(S)
            logp = model(repeat(X), sb, repeat(mask), repeat(chain_M * fixed), repeat(ridx), repeat(cenc),
                         torch.zeros(size, X.shape[1]), use_input_decoding_order=True, decoding_order=order)
            gathered.append(torch.gather(logp, 2, sb.unsqueeze(-1)).squeeze(-1)[:, pep].numpy())
    arr = np.concatenate(gathered)
    return dict(positions=arr.mean(axis=0).tolist(), total=float(arr.sum(axis=1).mean()),
                order_sd=float(arr.sum(axis=1).std(ddof=1)))


def score_pair(folder, identity):
    """Extract two contexts from one existing ORIGINAL-train predicted backbone."""
    from . import inverse_folding as inverse
    cx = inverse.load_complex(Path(folder))
    if (cx.allele, cx.peptide, cx.arm, cx.model, cx.seed) != (identity["allele"], identity["peptide"], "B", "boltz2", 0):
        raise ValueError("structure/native sequence identity differs from frozen panel")
    if cx.split != "train":
        raise ValueError("only original training structures may be scored")
    weight = inverse.MPNN_DIR / "vanilla_model_weights" / "v_48_020.pt"
    # Cached after first pair: hash is an input acceptance check, not repeated work.
    if not getattr(score_pair, "_checkpoint_verified", False):
        if hashlib.sha256(weight.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("ProteinMPNN checkpoint differs")
        score_pair._checkpoint_verified = True
    started = time.perf_counter()
    orders = peptide_orders()
    conditional = _score_context(cx, orders)
    only = replace(cx, chains={cx.peptide_chain: cx.chains[cx.peptide_chain]})
    peptide = _score_context(only, orders)
    row = dict(identity, status="ok", n_orders=N_ORDERS, decoding_seed=SEED,
               checkpoint_sha256=CHECKPOINT_SHA256, order_protocol="explicit-common-peptide-permutations-v1",
               context_chains="+".join(cx.context_chains), elapsed_s=time.perf_counter() - started)
    for prefix, result in (("C", conditional), ("P", peptide)):
        row[f"{prefix}_total"] = result["total"]
        row[f"{prefix}_order_sd"] = result["order_sd"]
        row.update({f"{prefix}_pos_{i + 1}": v for i, v in enumerate(result["positions"])})
    row["delta_total"] = conditional["total"] - peptide["total"]
    row["metadata_sha256"] = hashlib.sha256((Path(folder) / "metadata.json").read_bytes()).hexdigest()
    row["structure_sha256"] = hashlib.sha256(next(Path(folder).glob("*.cif")).read_bytes()).hexdigest()
    return row


def validate_scores(panel, scores):
    keys = ["allele", "peptide"]
    if scores.duplicated(keys).any() or len(scores) != len(panel):
        raise ValueError("every frozen pair requires exactly one score")
    joined = panel.merge(scores, on=keys, how="outer", validate="one_to_one", suffixes=("", "_score"), indicator=True)
    if not joined._merge.eq("both").all() or not joined.status.eq("ok").all():
        raise ValueError("missing/extra/failed scoring rows; no complete-case fallback")
    for col in ("cluster_id", "outer_fold", "complex_id", "folder"):
        if f"{col}_score" in joined and not joined[col].astype(str).eq(joined[f"{col}_score"].astype(str)).all():
            raise ValueError("score/panel identity mismatch")
    if not joined.n_orders.eq(N_ORDERS).all() or not joined.decoding_seed.eq(SEED).all() or not joined.checkpoint_sha256.eq(CHECKPOINT_SHA256).all() or not joined.order_protocol.eq("explicit-common-peptide-permutations-v1").all():
        raise ValueError("scoring settings differ from frozen protocol")
    for prefix in ("C", "P"):
        values = joined[[f"{prefix}_pos_{i}" for i in range(1, 10)]].to_numpy(float)
        if not np.isfinite(values).all() or not np.allclose(values.sum(axis=1), joined[f"{prefix}_total"], atol=1e-4):
            raise ValueError("nonfinite or internally inconsistent position scores")
    return joined.drop(columns="_merge")


def feature_arms(base, scored):
    p = scored[[f"P_pos_{i}" for i in range(1, 10)]].to_numpy(float)
    c = scored[[f"C_pos_{i}" for i in range(1, 10)]].to_numpy(float)
    out = {}
    for rep in ("R0", "R5"):
        x = np.asarray(base[rep], dtype=float)
        if x.ndim != 2 or len(x) != len(scored) or not np.isfinite(x).all():
            raise ValueError("matched finite baseline feature rows required")
        out[rep], out[rep + "_P"], out[rep + "_PC"] = x, np.column_stack([x, p]), np.column_stack([x, p, c])
    return out


def panel_metric(frame, prediction, min_rows):
    from . import evaluation as ev
    eligible = ev.eligible_alleles(frame.allele, frame.y_log1p, min_rows=min_rows)
    if not eligible:
        raise ValueError("no eligible allele in predeclared panel")
    rho = ev.per_allele_spearman(frame.allele, frame.y_log1p, prediction, eligible)
    return float(np.median(ev.panel_spearman(rho, eligible))), eligible


def nested_assessment(frame, arms, *, fit_predict=None, folds=tuple(range(5))):
    """300 ridge fits; each alpha selected solely using pooled clean inner OOF."""
    from . import nested
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if not frame.split.eq("train").all() or frame.groupby("cluster_id").outer_fold.nunique().max() != 1:
        raise ValueError("assessment requires original training rows and inherited whole-cluster folds")
    if tuple(arms) != ARMS or set(frame.outer_fold) != set(range(5)):
        raise ValueError("all six arms and five folds required")
    if not np.isfinite(frame.y_log1p).all():
        raise ValueError("finite training labels required")
    def default_fit(x_fit, y_fit, x_held, alpha):
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr", tol=1e-5, max_iter=2000)).fit(x_fit, y_fit).predict(x_held)
    run = default_fit if fit_predict is None else fit_predict
    predictions = {arm: np.full(len(frame), np.nan) for arm in ARMS}
    selections, ancestry = [], []
    labels = frame.y_log1p.to_numpy(float)
    for fold in folds:
        fitting = np.flatnonzero(frame.outer_fold.to_numpy() != fold)
        held = np.flatnonzero(frame.outer_fold.to_numpy() == fold)
        inner = nested.inner_folds(frame.iloc[fitting], 3)
        for arm, x in arms.items():
            choices = []
            for alpha in ALPHAS:
                inner_pred = np.full(len(fitting), np.nan)
                for i in range(3):
                    fit_idx, inner_held = fitting[inner != i], fitting[inner == i]
                    if set(frame.iloc[fit_idx].cluster_id) & set(frame.iloc[inner_held].cluster_id):
                        raise ValueError("inner cluster leakage")
                    started = time.perf_counter()
                    result = np.asarray(run(x[fit_idx], labels[fit_idx], x[inner_held], alpha), float)
                    elapsed = time.perf_counter() - started
                    if result.shape != (len(inner_held),) or not np.isfinite(result).all():
                        raise ValueError("finite complete held predictions required")
                    inner_pred[inner == i] = result
                    ancestry.append(dict(outer_fold=fold, arm=arm, alpha=alpha, inner_fold=i, fit_seconds=elapsed, scaler_fit="all fitting rows",
                                         fit_clusters=sorted(map(int, frame.iloc[fit_idx].cluster_id.unique())),
                                         held_clusters=sorted(map(int, frame.iloc[inner_held].cluster_id.unique()))))
                score, panel = panel_metric(frame.iloc[fitting], inner_pred, min_rows=20)
                choices.append(dict(alpha=alpha, score=score, n_eligible=len(panel), inner_predictions=inner_pred.tolist()))
            chosen = max(choices, key=lambda item: item["score"])
            started = time.perf_counter()
            predictions[arm][held] = run(x[fitting], labels[fitting], x[held], chosen["alpha"])
            ancestry.append(dict(outer_fold=fold, arm=arm, alpha=chosen["alpha"], inner_fold=None, fit_seconds=time.perf_counter() - started, scaler_fit="all fitting rows",
                                 fit_clusters=sorted(map(int, frame.iloc[fitting].cluster_id.unique())),
                                 held_clusters=sorted(map(int, frame.iloc[held].cluster_id.unique()))))
            selections.append(dict(outer_fold=fold, arm=arm, selected_alpha=chosen["alpha"], fitting_pair_ids=frame.iloc[fitting].pair_id.tolist(), choices=choices))
    selected_rows = frame.outer_fold.isin(folds).to_numpy()
    if len(ancestry) != 60 * len(folds) or any(not np.isfinite(p[selected_rows]).all() for p in predictions.values()):
        raise ValueError("incomplete fixed fit graph")
    return predictions, selections, ancestry


def neural_assessment(frame, arms, *, folds=tuple(range(5)), head_factory=None):
    """Same frozen two-member FeatureHead at inner and final levels:600 fits."""
    from . import nested
    from .round2_embedding import FeatureHead
    from .round2_nested import Ancestry, unlabelled
    if tuple(arms) != ARMS or not frame.split.eq("train").all() or set(frame.outer_fold) != set(range(5)):
        raise ValueError("complete frozen original-training panel required")
    if frame.groupby("cluster_id").outer_fold.nunique().max() != 1:
        raise ValueError("outer cluster leakage")
    ancestry = Ancestry()
    head = (FeatureHead if head_factory is None else head_factory)(frame, arms, ancestry)
    predictions = {arm: np.full(len(frame), np.nan) for arm in ARMS}
    selections = []
    for fold in folds:
        fitting = np.flatnonzero(frame.outer_fold.to_numpy() != fold)
        held = np.flatnonzero(frame.outer_fold.to_numpy() == fold)
        fit_frame = frame.iloc[fitting]
        inner = nested.inner_folds(fit_frame, 3)
        for arm in ARMS:
            choices = []
            for l2 in (1., .01, .0001):
                candidate = dict(id=f"{arm}_mlp_{l2:g}", representation=arm, learner="mlp", regularization=l2)
                inner_pred = np.full(len(fitting), np.nan)
                for i in range(3):
                    fit, assess = fit_frame.loc[inner != i], fit_frame.loc[inner == i]
                    inner_pred[inner == i] = head.predict(fit, unlabelled(assess), candidate,
                                                       f"inverse/mlp/{fold}/{arm}/{l2:g}/inner/{i}", final=False)
                score, eligible = panel_metric(fit_frame, inner_pred, 20)
                choices.append(dict(regularization=l2, score=score, n_eligible=len(eligible), inner_predictions=inner_pred.tolist()))
            chosen = max(choices, key=lambda item: item["score"])
            candidate = dict(id=f"{arm}_mlp_{chosen['regularization']:g}", representation=arm,
                             learner="mlp", regularization=chosen["regularization"])
            # final=False deliberately preserves the SAME bounded two-member
            # recipe; FeatureHead.final=True would create30 members instead.
            predictions[arm][held] = head.predict(fit_frame, unlabelled(frame.iloc[held]), candidate,
                                                 f"inverse/mlp/{fold}/{arm}/final", final=False)
            selections.append(dict(outer_fold=fold, arm=arm, selected_l2=chosen["regularization"], fitting_pair_ids=fit_frame.pair_id.tolist(), choices=choices))
    selected_rows = frame.outer_fold.isin(folds).to_numpy()
    if len(head.members) != 120 * len(folds) or any(not np.isfinite(p[selected_rows]).all() for p in predictions.values()):
        raise ValueError("incomplete fixed neural graph")
    return predictions, selections, dict(records=ancestry.records, rowsets=ancestry.rowsets, members=head.members)


def audit_ancestry(frame, saved, learner, folds):
    """Independently reject fitting/stopping/scaling overlap with held clusters."""
    cluster = frame.set_index("pair_id").cluster_id.to_dict()
    if learner == "ridge":
        if len(saved) != 60 * len(folds):
            raise ValueError("ridge ancestry fit count differs")
        for record in saved:
            forbidden = set(frame.loc[frame.outer_fold.eq(record["outer_fold"]), "cluster_id"])
            fitted = set(record["fit_clusters"])
            if fitted & (set(record["held_clusters"]) | forbidden):
                raise ValueError("ridge held cluster entered fitting/scaling")
    else:
        if len(saved["records"]) != 120 * len(folds) or len(saved["members"]) != 120 * len(folds):
            raise ValueError("neural ancestry/member count differs")
        for record in saved["records"]:
            fold = int(record["context"].split("/")[2])
            forbidden = set(frame.loc[frame.outer_fold.eq(fold), "cluster_id"])
            rowsets = saved["rowsets"]
            predict = {cluster[i] for i in rowsets[record["prediction_rows"]]}
            fit = {cluster[i] for i in rowsets[record["fit_rows"]]}
            stop = {cluster[i] for i in rowsets[record["stopping_rows"]]}
            if fit & stop or (fit | stop) & (predict | forbidden):
                raise ValueError("neural held cluster entered gradient/stopping")
            if record["scaler_fit_rows"] != record["fit_rows"]:
                raise ValueError("neural scaler was not fitted on gradient rows only")
            for key in ("fit_rows", "prediction_rows", "stopping_rows"):
                ids = rowsets[record[key]]
                actual = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()
                if actual != record[key] or any(i not in cluster for i in ids):
                    raise ValueError("neural rowset identity differs")
    return True
