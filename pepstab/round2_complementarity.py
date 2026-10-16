"""Clean nested complementarity for fixed original MLP and R2/R4 ridge.

Fresh predictions are generated exclusively inside the supplied outer fitting
partition. Only the final assessed predictions may reuse audited outer artifacts.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib

import numpy as np

from . import evaluation as ev, nested
from .round2_embedding import FeatureHead, head_candidates
from .round2_nested import Ancestry, blend, select, unlabelled, validate_partition
from .round2_sequence import candidates

REPRESENTATIONS = ("R2", "R4")
BLEND_WEIGHTS = (0., .25, .5, .75, 1.)
CORRECTION_WEIGHTS = (0., 1.)
RESIDUAL_ALPHA = 10000.


def fitting_graph():
    return {"per_outer": {"fixed_original_base_members": 360,
                          "blend_ridge_fits": 60, "residual_ridge_fits": 8,
                          "total_actual_model_fits": 428},
            "five_outer_actual_model_fits": 2140,
            "base_breakdown": "3 outer-fit OOF blocks x30 +3 meta folds x3 deeper OOF blocks x30",
            "ridge_breakdown": "2 representations x3 meta folds x(3alpha x3selectionfolds +1 refit) +2x(3meta residualfits+1final)",
            "outer_reference_refits": 0,
            "reuse": "outer references reused only for final assessed rows; fresh crossfits shared across both representations",
            "cost": "local CPU only; execution runtime measured separately; no cloud launch"}


class FixedOriginalBase:
    def __init__(self, ensemble_predictor):
        self.predictor = ensemble_predictor

    def __call__(self, fit, held, context):
        return self.predictor.predict(fit, held, candidates()["mlp_log"][0], context, final=True)


def _partition_key(frame):
    return hashlib.sha256(np.asarray(sorted(frame.pair_id), dtype="<i8").tobytes()).hexdigest()


def base_crossfit(fit, base, ancestry: Ancestry):
    """Three fresh fixed-recipe OOF blocks; canonical contexts permit safe reuse."""
    fold = nested.inner_folds(fit, 3)
    if set(fold) != {0, 1, 2}:
        raise ValueError("insufficient fitting clusters for clean original crossfit")
    result = np.empty(len(fit))
    for k in range(3):
        train, held = fit[fold != k], fit[fold == k]
        validate_partition(train, held)
        context = f"complementarity/base/{_partition_key(fit)}/crossfit/{k}"
        result[fold == k] = base(train, unlabelled(held), context)
        ancestry.add(role="fixed_original_clean_oof", fit=train, held=held,
                     context=context, candidate="mlp_log", recipe_members=30)
    return ev.validate_finite(result, "clean original base OOF")


def _assert_exclusion(ancestry, first_record, forbidden):
    forbidden = set(forbidden.pair_id)
    for record in ancestry.records[first_record:]:
        for role in ("fit_rows", "stopping_rows", "selection_rows", "scaler_fit_rows"):
            if role in record and forbidden.intersection(ancestry.rowsets[record[role]]):
                raise AssertionError(f"meta-held ancestry violation: {record['role']} {role}")


def choose_meta_weight(fit, base_oof, alternative_oof, weights, *, residual=False, min_rows=20):
    """First frozen weight within one fold-SE of best three inner panel medians."""
    fold = nested.inner_folds(fit, 3)
    rows = []
    for weight in weights:
        prediction = apply_correction(base_oof, alternative_oof, weight) if residual else blend(base_oof, alternative_oof, weight)
        scores = []
        for k in range(3):
            keep = fold == k
            panel = ev.eligible_alleles(fit.allele[keep], fit.y_log1p.to_numpy()[keep], min_rows=min_rows)
            if not panel:
                raise ValueError("no eligible meta-inner allele panel")
            value = ev.score("meta_inner", fit.allele[keep], fit.y_log1p.to_numpy()[keep], prediction[keep], alleles=panel).median_spearman
            scores.append(float(value))
        rows.append({"weight": weight, "fold_scores": scores, "mean": float(np.mean(scores)),
                     "se": float(np.std(scores, ddof=1)/np.sqrt(3))})
    best = max(rows, key=lambda row: row["mean"])
    chosen = next(row["weight"] for row in rows if row["mean"] >= best["mean"] - best["se"])
    return {"weight": chosen, "scores": rows, "rule": "first frozen weight within one fold-SE of best inner mean; zero first"}


def apply_correction(baseline, correction, weight):
    baseline = ev.validate_finite(baseline, "baseline")
    correction = ev.validate_finite(correction, "correction")
    if baseline.shape != correction.shape or weight not in CORRECTION_WEIGHTS:
        raise ValueError("invalid residual correction inputs")
    return baseline.copy() if weight == 0 else baseline + correction


def meta_predictions(fit, base, head: FeatureHead, ancestry: Ancestry, *, min_rows=20):
    """Return nested meta-held inputs and residuals for both frozen candidates.

    For each meta-held block, ridge hyperparameters and every residual training
    target are constructed without that block's labels, even for early stopping.
    """
    if head.ancestry is not ancestry:
        raise ValueError("head and complementarity must share complete ancestry")
    fold = nested.inner_folds(fit, 3)
    base_oof = base_crossfit(fit, base, ancestry)
    embedded = {rep: np.empty(len(fit)) for rep in REPRESENTATIONS}
    residual = {rep: np.empty(len(fit)) for rep in REPRESENTATIONS}
    choices, target_records = {rep: [] for rep in REPRESENTATIONS}, []
    for k in range(3):
        inner_fit, inner_held = fit[fold != k], fit[fold == k]
        first = len(ancestry.records)
        # These are fresh crossfits *inside* inner_fit, never other outer folds.
        clean = base_crossfit(inner_fit, base, ancestry)
        residual_fit = inner_fit.copy()
        residual_fit["y_log1p"] = inner_fit.y_log1p.to_numpy() - clean
        target_records.append({"meta_fold": k, "fit_rows": ancestry.rows(inner_fit),
                               "prediction_rows": ancestry.rows(inner_held),
                               "base_oof_sha256": hashlib.sha256(clean.tobytes()).hexdigest(),
                               "residual_target_sha256": hashlib.sha256(residual_fit.y_log1p.to_numpy().tobytes()).hexdigest()})
        for rep in REPRESENTATIONS:
            context = f"complementarity/{_partition_key(fit)}/meta/{k}/{rep}"
            choice = select(inner_fit, head_candidates(rep, "ridge"), head,
                            min_rows=min_rows, ancestry=ancestry, context=context + "/embedding")
            embedded[rep][fold == k] = head.predict(inner_fit, unlabelled(inner_held), choice.candidate,
                                                  context + "/embedding_refit", final=True)
            ancestry.add(role="nested_selected_ridge_oof", fit=inner_fit, held=inner_held, selection=inner_fit,
                         context=context, candidate=choice.candidate["id"])
            choices[rep].append({"meta_fold": k, **asdict(choice)})
            candidate = {"id": f"{rep}_residual_ridge_10000", "representation": rep,
                         "learner": "ridge", "regularization": RESIDUAL_ALPHA}
            residual[rep][fold == k] = head.predict(residual_fit, unlabelled(inner_held), candidate,
                                                  context + "/residual", final=True)
            ancestry.add(role="residual_ridge_meta_prediction", fit=inner_fit, held=inner_held,
                         context=context, candidate=candidate["id"],
                         base_oof_target_hash=target_records[-1]["base_oof_sha256"],
                         residual_target_hash=target_records[-1]["residual_target_sha256"])
        _assert_exclusion(ancestry, first, inner_held)
    return {"base_oof": base_oof, "embedding_oof": embedded, "residual_oof": residual,
            "nested_embedding_choices": choices, "residual_target_provenance": target_records}


def assess_outer(fit, held, base, head: FeatureHead, ancestry: Ancestry,
                 baseline_outer, embedding_outer, *, min_rows=20):
    """Select inside fit; apply only to audited existing predictions for held."""
    validate_partition(fit, held)
    baseline_outer = ev.validate_finite(baseline_outer, "outer original baseline")
    if baseline_outer.shape != (len(held),) or set(embedding_outer) != set(REPRESENTATIONS):
        raise ValueError("outer reference coverage mismatch")
    meta = meta_predictions(fit, base, head, ancestry, min_rows=min_rows)
    predictions = {"mlp_log": baseline_outer.copy()}
    selections = {}
    residual_fit = fit.copy()
    residual_fit["y_log1p"] = fit.y_log1p.to_numpy() - meta["base_oof"]
    for rep in REPRESENTATIONS:
        alternative = ev.validate_finite(embedding_outer[rep], f"outer {rep} ridge")
        if alternative.shape != baseline_outer.shape:
            raise ValueError("outer embedding reference coverage mismatch")
        blend_choice = choose_meta_weight(fit, meta["base_oof"], meta["embedding_oof"][rep], BLEND_WEIGHTS, min_rows=min_rows)
        residual_choice = choose_meta_weight(fit, meta["base_oof"], meta["residual_oof"][rep], CORRECTION_WEIGHTS, residual=True, min_rows=min_rows)
        candidate = {"id": f"{rep}_residual_ridge_10000", "representation": rep,
                     "learner": "ridge", "regularization": RESIDUAL_ALPHA}
        correction = head.predict(residual_fit, unlabelled(held), candidate,
                                  f"complementarity/{_partition_key(fit)}/{rep}/residual_final", final=True)
        predictions[f"{rep}_ridge"] = alternative.copy()
        predictions[f"{rep}_blend"] = blend(baseline_outer, alternative, blend_choice["weight"])
        predictions[f"{rep}_residual"] = apply_correction(baseline_outer, correction, residual_choice["weight"])
        selections[rep] = {"blend": blend_choice, "residual": residual_choice,
                           "nested_embedding_choices": meta["nested_embedding_choices"][rep]}
        ancestry.add(role="complementarity_outer_prediction", fit=fit, held=held, selection=fit,
                     context=f"complementarity/outer/{rep}",
                     blend_weight=blend_choice["weight"], residual_weight=residual_choice["weight"],
                     baseline_oof_hash=hashlib.sha256(meta["base_oof"].tobytes()).hexdigest())
    return predictions, selections, meta
