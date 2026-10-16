"""Audit every frozen ranking outcome before paired development reporting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pepstab import evaluation, nested
from pepstab.boltz_embeddings import sha256_file
from pepstab.final_ranking import (REPRESENTATIONS, FINAL_AXES, INNER_AXES, array_hash,
                                  candidates, atomic_json, deploy_choice, one_se, ordered_rows)
from pepstab.round2_nested import Ancestry, align_predictions, config_digest
from pepstab.round2_reporting import cluster_bootstrap_scores
from scripts.final_ranking import (OUT, verify_freeze, training_frame, load_features,
                                  read_baseline, atomic_csv)


def paired_interaction(point, boot):
    """Subtract per-draw loss deltas, preserving the four-way paired covariance."""
    delta=(point["R5_ranked"]-point["R5_mse"])-(point["R0_ranked"]-point["R0_mse"])
    draw=(boot.R5_ranked-boot.R5_mse)-(boot.R0_ranked-boot.R0_mse)
    return delta,draw


def replay_choice(fit, table, rep):
    folds = nested.inner_folds(fit, 3)
    scores = []
    expected = sum(len(fit.loc[folds == inner]) for inner in range(3)) * 9
    if len(table) != expected:
        raise ValueError("fresh inner prediction coverage changed")
    for candidate in candidates(rep):
        values = []
        for inner in range(3):
            held = fit.loc[folds == inner]
            rows = table.loc[(table.candidate == candidate["id"]) & (table.inner_fold == inner)]
            pred = align_predictions(held, rows)
            if rows.allele.tolist() != held.allele.tolist():
                raise ValueError("inner allele identity changed")
            panel = evaluation.eligible_alleles(held.allele, held.y_log1p, min_rows=20)
            values.append(float(evaluation.score("inner", held.allele, held.y_log1p, pred, panel).median_spearman))
        scores.append(dict(candidate=candidate["id"], fold_scores=values,
                           mean=float(np.mean(values)), se=float(np.std(values,ddof=1)/np.sqrt(3))))
    return dict(candidate=next(c for c in candidates(rep) if c["id"] == one_se(scores)["candidate"]),
                scores=scores, rule="first in frozen parsimony order within 1 fold-SE of best")


def audit_component(out, protocol, train, features, rep, fold):
    directory = Path(out) / f"fold_{fold}/{rep}"
    done = json.loads((directory / "completed.json").read_text())
    if done["protocol_hash"] != protocol["protocol_hash"] or done["status"] != "complete" or done["new_scientific_fits"] != 84:
        raise ValueError("ranking component completion/protocol mismatch")
    for name, digest in done["files"].items():
        if Path(name).name != name or sha256_file(directory / name) != digest:
            raise ValueError("completed ranking output changed")
    outer = nested.outer_folds(train)
    fit, held = train.loc[outer != fold], train.loc[outer == fold]
    table = pd.read_csv(directory / "predictions.csv")
    if set(table.arm) != {f"{rep}_mse", f"{rep}_ranked", f"{rep}_deploy"} or len(table) != 3*len(held):
        raise ValueError("outer outcome coverage changed")
    for arm in table.arm.unique():
        rows = table.loc[table.arm == arm]
        align_predictions(held, rows)
        if rows.allele.tolist() != held.allele.tolist() or not rows.outer_fold.eq(fold).all():
            raise ValueError("outer row/cluster/allele identity changed")
    mse, mse_table, raw, _ = read_baseline(train, features, fold, rep)
    inner = pd.read_csv(directory / "inner_predictions.csv")
    ranked = replay_choice(fit, inner, rep)
    saved = json.loads((directory / "selection.json").read_text())
    # CSV roundtrip changes last bits. The choice must match exactly; stored
    # score values must agree at normal floating-point roundtrip precision.
    if saved["ranked"]["candidate"] != ranked["candidate"] or saved["mse"] != mse:
        raise ValueError("frozen inner selections changed")
    for actual, reference in zip(ranked["scores"], saved["ranked"]["scores"]):
        if actual["candidate"] != reference["candidate"]:
            raise ValueError("frozen candidate order changed")
        np.testing.assert_allclose(actual["fold_scores"], reference["fold_scores"], rtol=0, atol=2e-14)
        np.testing.assert_allclose([actual["mean"], actual["se"]], [reference["mean"],reference["se"]], rtol=0,atol=2e-14)
    if saved["deployment"] != deploy_choice(mse, saved["ranked"]):
        raise ValueError("MSE-first deployment selection changed")
    for arm,candidate in ((f"{rep}_mse",mse["candidate"]["id"]),(f"{rep}_ranked",ranked["candidate"]["id"]),
        (f"{rep}_deploy",saved[saved["deployment"]["procedure"]]["candidate"]["id"])):
        if not table.loc[table.arm==arm,"selected_candidate"].eq(candidate).all():
            raise ValueError("saved selected candidate metadata changed")
    expected_mse = mse_table.y_pred.to_numpy()
    np.testing.assert_allclose(table.loc[table.arm == f"{rep}_mse", "y_pred"], expected_mse, rtol=0,atol=2e-14)
    expected_deploy = table.loc[table.arm == f"{rep}_{saved['deployment']['procedure']}", "y_pred"].to_numpy()
    np.testing.assert_array_equal(table.loc[table.arm == f"{rep}_deploy", "y_pred"], expected_deploy)
    reused = pd.read_csv(directory / "reused_inner_predictions.csv")
    pd.testing.assert_frame_equal(reused.drop(columns="y_pred"), raw.drop(columns="y_pred"),check_dtype=False)
    np.testing.assert_allclose(reused.y_pred,raw.y_pred,rtol=0,atol=2e-14)
    ancestry = Ancestry()
    item = json.loads((directory / "ancestry.json").read_text())
    ancestry.records, ancestry.rowsets = item["records"], item["rowsets"]
    for key, ids in ancestry.rowsets.items():
        if ids != sorted(set(ids)) or array_hash(np.asarray(ids,dtype="<i8")) != key:
            raise ValueError("fresh ancestry rowset changed")
    audit = ancestry.audit(train, held)
    members = json.loads((directory / "member_manifest.json").read_text())
    if len(members) != 84:
        raise ValueError("fresh member count changed")
    position = {int(pid): i for i,pid in enumerate(train.pair_id)}
    scales = {}
    expected_records, aggregate = [], {}
    folds = nested.inner_folds(fit,3)
    schedule = [(c, i, fit.loc[folds != i], fit.loc[folds == i], INNER_AXES,
                  f"outer/{fold}/{rep}_ranked/select/{c['id']}/{i}") for c in candidates(rep) for i in range(3)]
    schedule.append((ranked["candidate"], None, fit, held, FINAL_AXES, f"outer/{fold}/{rep}_ranked/outer_refit"))
    cursor = 0
    for candidate, inner_fold, call_fit, call_held, axes, context in schedule:
        stop_folds = nested.inner_folds(call_fit,5)
        values = []
        for stop, seed in axes:
            member = members[cursor]
            cursor += 1
            gradient, stopping = call_fit.loc[stop_folds != stop], call_fit.loc[stop_folds == stop]
            signature = dict(protocol_hash=protocol["protocol_hash"], fit=ordered_rows(call_fit),
                held=ordered_rows(call_held), y=array_hash(call_fit.y_log1p.to_numpy()),
                candidate=candidate, context=context, final=inner_fold is None,
                features=array_hash(features[rep]), max_epochs=200)
            identity = dict(signature=signature,stop_fold=stop,seed=seed,gradient_order=ordered_rows(gradient),
                stopping_order=ordered_rows(stopping),gradient_targets=array_hash(gradient.y_log1p.to_numpy()),
                stopping_targets=array_hash(stopping.y_log1p.to_numpy()))
            if member["identity"] != identity:
                raise ValueError("fresh member identity/settings changed")
            for name,reference in (("gradient_rows",ancestry.rows(gradient)),("stopping_rows",ancestry.rows(stopping)),
                ("prediction_rows",ancestry.rows(call_held)),("scaler_rows",ancestry.rows(gradient)),
                ("seed",seed),("stop_fold",stop),("n_fit",len(gradient)),("n_stop",len(stopping)),
                ("n_features",features[rep].shape[1])):
                if member[name] != reference:
                    raise ValueError("fresh member audit metadata changed")
            stem = directory / "members" / config_digest(signature) / f"stop_{stop}_seed_{seed}"
            receipt = json.loads(stem.with_suffix(".json").read_text())
            path = stem.with_suffix(".npz")
            if receipt != member or sha256_file(path) != member["npz_sha256"]:
                raise ValueError("fresh member receipt or arrays changed")
            gradient_key = ordered_rows(gradient)
            if gradient_key not in scales:
                X = features[rep][[position[int(pid)] for pid in gradient.pair_id]]
                scales[gradient_key] = StandardScaler().fit(X)
            scaler = scales[gradient_key]
            with np.load(path,allow_pickle=False) as arrays:
                for name, reference in (("scaler_mean",scaler.mean_),("scaler_var",scaler.var_),("scaler_scale",scaler.scale_)):
                    np.testing.assert_array_equal(arrays[name],reference)
                if len(arrays["dev_mse_curve"]) != member["epochs"] or not np.isfinite(arrays["dev_mse_curve"]).all():
                    raise ValueError("fresh stopping curve incomplete")
                if int(np.argmin(arrays["dev_mse_curve"]))+1 != member["best_epoch"]:
                    raise ValueError("fresh MSE stopping epoch changed")
                values.append(arrays["predictions"].copy())
            expected_records.append(dict(role="ranking_member",fit_rows=ancestry.rows(gradient),
                prediction_rows=ancestry.rows(call_held),stopping_rows=ancestry.rows(stopping),
                context=context,candidate=candidate["id"],seed=seed,stop_fold=stop,scaler_fit_rows=ancestry.rows(gradient)))
        prediction = np.mean(values,axis=0)
        if inner_fold is None:
            saved_pred = table.loc[table.arm == f"{rep}_ranked","y_pred"].to_numpy()
        else:
            saved_pred = inner.loc[(inner.candidate == candidate["id"]) & (inner.inner_fold == inner_fold),"y_pred"].to_numpy()
        np.testing.assert_allclose(prediction,saved_pred,rtol=0,atol=2e-14)
    if [r for r in ancestry.records if r["role"] == "ranking_member"] != expected_records:
        raise ValueError("fresh complete member ancestry graph changed")
    wrappers = [dict(role="inner_selection_prediction",fit_rows=ancestry.rows(fit.loc[folds != i]),
        prediction_rows=ancestry.rows(fit.loc[folds == i]),candidate=c["id"],context=f"outer/{fold}/{rep}_ranked",inner_fold=i)
        for c in candidates(rep) for i in range(3)]
    wrappers.append(dict(role="outer_ranked_selection",fit_rows=ancestry.rows(fit),prediction_rows=ancestry.rows(held),
        selection_rows=ancestry.rows(fit),candidate=ranked["candidate"]["id"],context=f"outer/{fold}/{rep}_ranked"))
    if [r for r in ancestry.records if r["role"] != "ranking_member"] != wrappers:
        raise ValueError("fresh complete selection ancestry graph changed")
    return table, saved, audit


def build_report(out=OUT):
    out = Path(out)
    protocol = verify_freeze(out)
    validation = json.loads((out / "lambda0_validation.json").read_text())
    if validation["status"] != "passed" or validation["protocol_hash"] != protocol["protocol_hash"]:
        raise ValueError("lambda0 validation incomplete")
    train = training_frame()
    features, _ = load_features(train)
    tables, selections, audits = [], [], []
    for rep in REPRESENTATIONS:
        for fold in range(5):
            table, choice, audit = audit_component(out,protocol,train,features,rep,fold)
            tables.append(table)
            audits.append(dict(representation=rep,outer_fold=fold,**audit))
            selections.append(dict(representation=rep,outer_fold=fold,mse=choice["mse"]["candidate"]["id"],
                ranked=choice["ranked"]["candidate"]["id"],deployment=choice["deployment"]["procedure"],
                ranked_lambda=choice["ranked"]["candidate"]["rank_weight"],
                ranked_l2=choice["ranked"]["candidate"]["regularization"]))
    table = pd.concat(tables,ignore_index=True)
    predictions = {arm: align_predictions(train,table.loc[table.arm == arm]) for arm in table.arm.unique()}
    results, scores, boot_meta = [], [], {}
    contrasts = [("primary","R5_mse","R5_ranked"),("secondary","R0_mse","R0_ranked"),
        ("secondary","R0_mse","R5_mse"),("secondary","R0_ranked","R5_ranked"),
        ("deployment","R0_mse","R0_deploy"),("deployment","R5_mse","R5_deploy")]
    for min_rows in (20,50):
        boot, meta = cluster_bootstrap_scores(train,predictions,min_rows=min_rows,n_boot=2000)
        point = {arm:evaluation.score(arm,train.allele,train.y_log1p,pred,meta["alleles"]).median_spearman
                 for arm,pred in predictions.items()}
        for arm,pred in predictions.items():
            scores.append(dict(arm=arm,min_rows=min_rows,median_spearman=point[arm],
                mse_log1p=float(np.mean((pred-train.y_log1p.to_numpy())**2)),
                mae_log1p=float(np.mean(np.abs(pred-train.y_log1p.to_numpy()))),
                n_alleles=meta["n_alleles"]))
        for role,baseline,candidate in contrasts:
            draw = boot[candidate]-boot[baseline]
            lo,hi = np.nanpercentile(draw,[2.5,97.5])
            delta=point[candidate]-point[baseline]
            results.append(dict(role=role,contrast=f"{candidate}-{baseline}",min_rows=min_rows,
                delta=delta,ci_low=float(lo),ci_high=float(hi),point_inside_interval=bool(lo<=delta<=hi),
                bootstrap_mean_minus_point=float(np.nanmean(draw)-delta)))
        delta,draw=paired_interaction(point,boot)
        lo,hi=np.nanpercentile(draw,[2.5,97.5])
        results.append(dict(role="secondary_interaction",contrast="deltaR5-deltaR0",min_rows=min_rows,
            delta=delta,ci_low=float(lo),ci_high=float(hi),point_inside_interval=bool(lo<=delta<=hi),
            bootstrap_mean_minus_point=float(np.nanmean(draw)-delta)))
        atomic_csv(out/f"bootstrap_min{min_rows}.csv",boot)
        boot_meta[str(min_rows)]=meta
    atomic_csv(out/"predictions.csv",table)
    atomic_csv(out/"selections.csv",pd.DataFrame(selections))
    atomic_csv(out/"scores.csv",pd.DataFrame(scores))
    atomic_csv(out/"comparisons.csv",pd.DataFrame(results))
    primary=next(r for r in results if r["role"] == "primary" and r["min_rows"]==20)
    lines=["# Matched Boltz + ranking follow-up", "", "Previously examined original-training development evidence; original test not reopened.", "",
        f"Primary R5 mixed-loss minus R5 MSE: **{primary['delta']:+.4f}** median per-allele Spearman; paired-cluster 95% interval **[{primary['ci_low']:+.4f}, {primary['ci_high']:+.4f}]** (min20).", "",
        "The source/cohort/features, all 480 reused MSE logical members, all 840 fresh scientific fits, raw inner predictions, scaler statistics, row hashes, and complete fitting/stopping/selection ancestry were audited before scoring. Two separately receipted lambda0 validation fits passed.", "",
        "Intervals use 2,000 shared whole-cluster draws and condition on the fitted predictions and inner selection. They do not include selection repetition, independent assay variation, or independent confirmation. Min50 and the matched sequence contrast, representation contrasts, and loss-by-representation interaction are reported alongside the primary result.", "",
        "The positive ranking grid was lambda .1/.3/1 × L2 1/.01/.0001; MSE stopping, 64-unit MLP, natural batches256, Adam .001, max200 epochs, patience15. Each representation/fold used three inner cluster folds and two members per candidate/fold, then the selected candidate's 30-member final ensemble. The deployment selector preferred MSE within one fold-SE using inner scores only.", "",
        "MSE anchors the log1p half-life scale; no affine mapping or pure-ranking claim is made. R0 and R5 feature dimensions/parameter counts differ. This result does not authorize transfer expansion, additional lambda values, stopping changes, or repeats.", "",
        "See [comparisons.csv](comparisons.csv), [scores.csv](scores.csv), [selections.csv](selections.csv), [protocol.json](protocol.json), and [audit.json](audit.json).", ""]
    (out/"REPORT.md").write_text("\n".join(lines))
    atomic_json(out/"audit.json",dict(status="passed",protocol_hash=protocol["protocol_hash"],
        n_rows=len(train),new_scientific_fits=840,reused_logical_fits=480,validation_fits=2,
        ancestry=audits,bootstrap=boot_meta,uncertainty=protocol["specification"]["uncertainty"]))
    return primary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=OUT)
    args=parser.parse_args()
    print(json.dumps(build_report(args.out)))
