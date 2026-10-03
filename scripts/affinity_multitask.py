"""Stage 2c: does an auxiliary binding-affinity head help stability prediction?

    .venv/bin/python scripts/affinity_multitask.py                   # probe
    .venv/bin/python scripts/affinity_multitask.py --protocol ensemble
    .venv/bin/python scripts/affinity_multitask.py --drop-censored    # robustness

The target is residence time (how long a peptide stays bound). The auxiliary
label is affinity (how strongly it binds) -- a different measurement of a
different event, correlated at Spearman -0.491 on the pairs carrying both. The
hypothesis is that affinity labels, which cover far more peptides than the
stability assay, teach the shared encoder something the stability labels alone
do not.

**The comparison is controlled by construction.** Both arms are the same class,
:class:`pepstab.multitask.MultiTaskMLPRegressor`, differing only in
``lambda_aff``; at ``lambda_aff = 0`` it is bit-identical to the stage 2 network
(asserted in ``tests/test_multitask.py``). Same fit rows, same stopping fold,
same stopping criterion (stability dev MSE), same seeds, same config. The
auxiliary head is always allocated, so it cannot even change the random stream.

Two protocols, because stage 2 established that ensembling alone is worth +0.090
mean SCC from no new information:

- ``single`` -- the probe. ``inner_folds()``: one permanent 10% stopping fold,
  17,744 fit rows. Sweeps ``lambda_aff`` at 3 seeds per encoding.
- ``ensemble`` -- the decisive form, and the one stage 3 must be compared
  against. ``cv_folds()``: 5 folds x 2 encodings x 3 seeds = 30 networks per
  arm, both arms ensembled identically. An ensembled multi-task arm against a
  single-network single-task arm would manufacture a result.

**Selection honesty.** ``lambda_aff`` is swept on validation, so the best lambda's
validation score is optimistically biased and the single-task arm had no
equivalent freedom. The whole sweep is reported, not just its maximum, and the
paired CI is given for every lambda. Read the curve, not the peak.

Leakage: the probe trains only on affinity labels attached to pairs that already
carry a measured half-life, inside the frozen splits. It adds no peptide and
moves none across a split boundary, so there is no new leakage surface. The
expansion to peptides *outside* the stability set is a separate question and
needs ``pepstab.affinity.filter_by_distance``; this script reports what that
filter would admit but does not train on it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.affinity import (  # noqa: E402
    auxiliary_only,
    dual_labelled,
    filter_by_distance,
    held_out_peptides,
)
from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import (  # noqa: E402
    describe_delta,
    eligible_alleles,
    paired_cluster_bootstrap,
    per_allele_spearman,
    score,
)
from pepstab.features import ENCODINGS, build_features  # noqa: E402
from pepstab.multitask import MultiTaskMLPConfig, MultiTaskMLPRegressor  # noqa: E402
from scripts.baseline_ensemble import SELECTED, cv_folds  # noqa: E402
from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE, SEEDS, inner_folds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = REPO_ROOT / "preds"
REPORT_DIR = REPO_ROOT / "reports"

#: The auxiliary loss weights swept. 0 is the single-task comparator and is
#: always included -- it is the control, not an option.
LAMBDA_GRID = (0.0, 0.1, 0.3, 1.0, 3.0)


def affinity_head_quality(model: MultiTaskMLPRegressor, X: np.ndarray,
                          y_aff: np.ndarray) -> float:
    """Spearman between the auxiliary head and the held-out affinity labels.

    The diagnostic that separates the two ways stage 2c can come out flat: the
    auxiliary task was learned and simply did not transfer, or the auxiliary
    head never trained at all. Without this, a null result is unreadable.
    """
    mask = np.isfinite(y_aff)
    if mask.sum() < 10:
        return float("nan")
    pred = model.predict_affinity(X)[mask]
    if len(np.unique(pred)) < 2:
        return float("nan")  # head never trained -- lambda_aff = 0
    return float(stats.spearmanr(pred, y_aff[mask]).statistic)


#: Minimum dual-labelled pairs an allele needs to enter the ceiling diagnostic.
#: Matches docs/AFFINITY_REFERENCE.md so the two numbers are comparable.
MIN_DUAL_PAIRS_FOR_CEILING = 30


def affinity_ceiling(df: pd.DataFrame, aff: np.ndarray) -> dict:
    """How well *measured* affinity ranks stability, used directly as a predictor.

    This is the diagnostic that makes a null result interpretable rather than
    merely disappointing. It asks what an auxiliary label could contribute at
    best: if measured affinity -- perfect knowledge, no model error -- ranks
    stability worse than the stability model already does, then the auxiliary
    task carries no ranking information the target labels have not already
    supplied, and a flat result is the expected outcome rather than a surprise.

    Scored on the dual-labelled pairs only, through the project's own
    per-allele Spearman, so the number is directly comparable to every other
    rho in the stage 2 and 2c tables.

    The caller passes the **training** split. That is the apt panel -- it is the
    ceiling on what the auxiliary labels can teach *this* model -- and it also
    keeps the diagnostic clear of the held-out labels entirely. Measured on all
    28,166 rows instead the figure is 0.581 rather than 0.580, so nothing rests
    on the choice; see docs/AFFINITY_REFERENCE.md, which characterises the full
    table.
    """
    sub = df.loc[np.isfinite(aff)].copy()
    sub["y_affinity"] = aff[np.isfinite(aff)]
    counts = sub.groupby("allele").size()
    panel = sorted(counts[counts >= MIN_DUAL_PAIRS_FOR_CEILING].index)
    rho = per_allele_spearman(sub.allele, sub.y_log1p.to_numpy(),
                              sub.y_affinity.to_numpy(), panel)
    return {
        "n_dual_pairs": len(sub),
        "n_alleles_total": int(sub.allele.nunique()),
        "n_alleles_in_panel": len(panel),
        "median_per_allele_spearman": float(rho.median()),
        "iqr_low": float(rho.quantile(0.25)),
        "iqr_high": float(rho.quantile(0.75)),
        "pooled_spearman": float(stats.spearmanr(
            sub.y_affinity, sub.y_log1p).statistic),
    }


def fit_one(X_fit, y_fit, aff_fit, X_dev, y_dev, hidden, l2, seed, lam):
    """One network. Everything but ``lam`` is pinned to the stage 2 selection."""
    cfg = MultiTaskMLPConfig(hidden=hidden, l2=l2, seed=seed, lambda_aff=lam,
                             max_epochs=MAX_EPOCHS, patience=PATIENCE)
    return MultiTaskMLPRegressor(cfg).fit(X_fit, y_fit, aff_fit, X_dev, y_dev)


def run_single(df, train, val, aff_train, aff_val, alleles, input_set, lambdas):
    """Probe protocol: one permanent stopping fold, lambda x seed x encoding.

    Returns ``(records, preds)`` where ``preds[(encoding, lam)]`` is the
    seed-averaged validation prediction -- the arm compared in the paired CI, so
    that both sides of every comparison average exactly ``len(SEEDS)`` networks.
    """
    fold = inner_folds(train)
    is_fit = (fold == "fit").to_numpy()
    y_train = train.y_log1p.to_numpy()
    y_val = val.y_log1p.to_numpy()

    records, preds = [], {}
    for enc in ENCODINGS:
        hidden, l2 = SELECTED[(input_set, enc)]
        X_train = build_features(train, input_set, enc)
        X_val = build_features(val, input_set, enc)
        X_fit, X_dev = X_train[is_fit], X_train[~is_fit]
        y_fit, y_dev = y_train[is_fit], y_train[~is_fit]
        aff_fit = aff_train[is_fit]
        print(f"\n  {enc} / {input_set}  hidden={hidden} l2={l2:g}  "
              f"fit={is_fit.sum():,} dev={(~is_fit).sum():,}  "
              f"affinity labels in fit={int(np.isfinite(aff_fit).sum()):,}")
        for lam in lambdas:
            per_seed = []
            for seed in SEEDS:
                m = fit_one(X_fit, y_fit, aff_fit, X_dev, y_dev, hidden, l2, seed, lam)
                p = m.predict(X_val)
                per_seed.append(p)
                s = score(f"mt_{enc}_lam{lam:g}_s{seed}", val.allele, y_val, p, alleles)
                records.append({
                    "protocol": "single", "input_set": input_set, "encoding": enc,
                    "lambda_aff": lam, "seed": seed,
                    "median_per_allele_spearman": s.median_spearman,
                    "mae_log1p": s.mae_log1p,
                    "median_precision_at_10": s.median_precision_at_k,
                    "pooled_spearman": s.pooled_spearman,
                    "best_epoch": m.best_epoch_, "epochs": m.n_epochs_,
                    "dev_mse": round(m.dev_mse_, 5),
                    "fit_seconds": round(m.fit_seconds_, 2),
                    "n_affinity_rows": m.n_affinity_rows_,
                    "affinity_head_spearman": round(
                        affinity_head_quality(m, X_val, aff_val), 4),
                })
            preds[(enc, lam)] = np.mean(per_seed, axis=0)
            rows = [r for r in records if r["encoding"] == enc
                    and r["lambda_aff"] == lam]
            rho = [r["median_per_allele_spearman"] for r in rows]
            ahs = np.nanmean([r["affinity_head_spearman"] for r in rows])
            print(f"    lambda={lam:<5g} rho={np.mean(rho):+.4f} "
                  f"[{min(rho):+.4f},{max(rho):+.4f}]  "
                  f"aff-head rho={ahs:+.3f}  "
                  f"ep={np.mean([r['best_epoch'] for r in rows]):.0f}")
    return records, preds


def run_ensemble(df, train, val, aff_train, aff_val, alleles, input_set, lambdas,
                 n_folds=5):
    """Ensemble protocol: 5 CV folds x 2 encodings x 3 seeds per arm.

    Both arms get the identical ensemble, so the comparison isolates the
    auxiliary loss. Each member stops on its own fold, so the ensemble
    collectively covers all 19,716 training rows.
    """
    fold = cv_folds(train, n_folds)
    y_train = train.y_log1p.to_numpy()
    y_val = val.y_log1p.to_numpy()
    feats = {enc: (build_features(train, input_set, enc),
                   build_features(val, input_set, enc)) for enc in ENCODINGS}

    records, preds = [], {}
    n_members = n_folds * len(ENCODINGS) * len(SEEDS)
    for lam in lambdas:
        members, head_rhos, secs = [], [], 0.0
        for enc in ENCODINGS:
            hidden, l2 = SELECTED[(input_set, enc)]
            X_train, X_val = feats[enc]
            for k in range(n_folds):
                is_fit = fold != k
                for seed in SEEDS:
                    m = fit_one(X_train[is_fit], y_train[is_fit], aff_train[is_fit],
                                X_train[~is_fit], y_train[~is_fit],
                                hidden, l2, seed, lam)
                    members.append(m.predict(X_val))
                    head_rhos.append(affinity_head_quality(m, X_val, aff_val))
                    secs += m.fit_seconds_
        ens = np.mean(members, axis=0)
        preds[("ensemble", lam)] = ens
        s = score(f"mt_ens_lam{lam:g}", val.allele, y_val, ens, alleles)
        member_rho = float(np.mean([
            score("m", val.allele, y_val, p, alleles).median_spearman
            for p in members]))
        records.append({
            "protocol": "ensemble", "input_set": input_set, "encoding": "both",
            "lambda_aff": lam, "seed": -1, "n_members": n_members,
            "median_per_allele_spearman": s.median_spearman,
            "mae_log1p": s.mae_log1p,
            "median_precision_at_10": s.median_precision_at_k,
            "pooled_spearman": s.pooled_spearman,
            "member_mean_median_spearman": member_rho,
            "fit_seconds": round(secs, 1),
            "affinity_head_spearman": round(float(np.nanmean(head_rhos)), 4),
        })
        print(f"    lambda={lam:<5g} {n_members}-net ensemble rho={s.median_spearman:+.4f} "
              f"(mean member {member_rho:+.4f})  "
              f"aff-head rho={np.nanmean(head_rhos):+.3f}  {secs / 60:.1f} min")
    return records, preds


def compare(val, preds, alleles, baseline_key, label: str) -> list[dict]:
    """Paired cluster bootstrap of every arm against its lambda=0 control."""
    y_val = val.y_log1p.to_numpy()
    base = preds[baseline_key]
    rows = []
    for key, pred in sorted(preds.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        if key == baseline_key:
            continue
        if key[0] != baseline_key[0]:
            continue  # only compare within the same encoding / protocol
        r = paired_cluster_bootstrap(val.cluster_id, val.allele, y_val, base, pred,
                                     alleles=alleles)
        lo, hi = r["ci95"]
        rows.append({
            "comparison": f"{label}:{key[0]} lambda={key[1]:g} vs lambda=0",
            "arm": str(key[0]), "lambda_aff": key[1],
            "delta_median_spearman": round(r["delta_median_spearman"], 4),
            "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
            "verdict": describe_delta(r),
        })
        print(f"    {key[0]:9s} lambda={key[1]:<5g} "
              f"delta={r['delta_median_spearman']:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]  {describe_delta(r)}")
    return rows


def report_expansion_surface(df, train) -> pd.DataFrame:
    """What the expansion to non-stability peptides would admit, without training.

    Stage 2c's expansion step is gated on the probe helping. The leakage audit
    is reported either way, because it is the part a reader has to check.
    """
    fold = inner_folds(train)
    dev_peptides = train.loc[(fold == "dev").to_numpy(), "peptide"]
    held = held_out_peptides(df, dev_peptides=dev_peptides)
    cand = auxiliary_only(df)
    kept, audit = filter_by_distance(cand, held)
    print(f"\nexpansion surface (reported, not trained on):")
    print(f"  affinity rows on peptides absent from the stability set: {len(cand):,} "
          f"({cand.peptide.nunique():,} peptides, {cand.dataset_allele.nunique()} alleles)")
    print(f"  held-out peptides to stay away from: {len(held):,} "
          f"(val + test + inner stopping fold)")
    print(f"  after excluding Hamming <= 3 of any of them: {len(kept):,} rows "
          f"({kept.peptide.nunique():,} peptides)")
    print(audit.to_string(index=False))
    return audit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", default="single", choices=["single", "ensemble"])
    ap.add_argument("--input-set", default="pep_pseudo",
                    choices=["pep", "pep_pseudo", "pep_domain"])
    ap.add_argument("--lambdas", type=float, nargs="*", default=None,
                    help=f"auxiliary loss weights (default: {LAMBDA_GRID}). "
                         "0 is always included as the control.")
    ap.add_argument("--drop-censored", action="store_true",
                    help="drop affinity rows carrying a < or > inequality "
                         "(6.0%% of the dual-labelled set). Robustness variant; "
                         "writes *_nocensor reports.")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    lambdas = tuple(args.lambdas) if args.lambdas else LAMBDA_GRID
    if 0.0 not in lambdas:
        lambdas = (0.0,) + lambdas

    df = load_with_splits()
    aff_all = dual_labelled(df, drop_censored=args.drop_censored)
    train = df[df.split == "train"]
    val = df[df.split == "val"].sort_values("pair_id").reset_index(drop=True)
    aff_train = aff_all.loc[train.index].to_numpy()
    aff_val = dual_labelled(val, drop_censored=args.drop_censored).to_numpy()
    alleles = eligible_alleles(val.allele, val.y_log1p, split="val")

    n_tr = int(np.isfinite(aff_train).sum())
    print(f"stage 2c: auxiliary affinity head, protocol={args.protocol}, "
          f"input_set={args.input_set}")
    print(f"censored affinity rows: {'dropped' if args.drop_censored else 'kept'}")
    print(f"train={len(train):,} rows, {n_tr:,} with an affinity label "
          f"({n_tr / len(train):.1%}), "
          f"{train.loc[np.isfinite(aff_train), 'allele'].nunique()} alleles, "
          f"{train.loc[np.isfinite(aff_train), 'peptide'].nunique():,} peptides")
    print(f"val={len(val):,} rows, {int(np.isfinite(aff_val).sum()):,} with an "
          f"affinity label (diagnostic only -- never trained on)")
    print(f"eligible val alleles: {len(alleles)}")
    print(f"lambda grid: {lambdas}")

    # What the auxiliary label could buy at best, before fitting anything.
    ceiling = affinity_ceiling(train, aff_train)
    print(f"\nceiling: measured affinity used *directly* as a stability predictor")
    print(f"  (training split only -- the panel the auxiliary labels act on)")
    print(f"  median per-allele rho {ceiling['median_per_allele_spearman']:+.4f} "
          f"[IQR {ceiling['iqr_low']:+.4f}, {ceiling['iqr_high']:+.4f}] over "
          f"{ceiling['n_alleles_in_panel']} alleles with >= "
          f"{MIN_DUAL_PAIRS_FOR_CEILING} dual-labelled pairs")
    print(f"  pooled rho {ceiling['pooled_spearman']:+.4f} on "
          f"{ceiling['n_dual_pairs']:,} dual-labelled pairs")
    print("  Compare against the stage 2 sequence baseline it is meant to improve.")

    started = time.perf_counter()
    if args.protocol == "single":
        records, preds = run_single(df, train, val, aff_train, aff_val, alleles,
                                    args.input_set, lambdas)
        print(f"\npaired cluster bootstrap vs lambda=0 "
              f"(both arms averaged over {len(SEEDS)} seeds):")
        deltas = []
        for enc in ENCODINGS:
            deltas.extend(compare(val, {k: v for k, v in preds.items()
                                        if k[0] == enc}, alleles, (enc, 0.0),
                                  "single"))
    else:
        records, preds = run_ensemble(df, train, val, aff_train, aff_val, alleles,
                                      args.input_set, lambdas, args.folds)
        print(f"\npaired cluster bootstrap vs lambda=0 "
              f"(both arms {args.folds * len(ENCODINGS) * len(SEEDS)}-network ensembles):")
        deltas = compare(val, preds, alleles, ("ensemble", 0.0), "ensemble")
    wall = time.perf_counter() - started

    audit = report_expansion_surface(df, train)

    REPORT_DIR.mkdir(exist_ok=True)
    PRED_DIR.mkdir(exist_ok=True)
    suffix = "_nocensor" if args.drop_censored else ""
    tag = f"{args.protocol}_{args.input_set}{suffix}"
    pd.DataFrame(records).to_csv(REPORT_DIR / f"stage2c_runs_{tag}.csv", index=False)
    pd.DataFrame(deltas).to_csv(REPORT_DIR / f"stage2c_deltas_{tag}.csv", index=False)
    audit.to_csv(REPORT_DIR / "stage2c_expansion_audit.csv", index=False)
    pd.DataFrame([ceiling]).to_csv(
        REPORT_DIR / f"stage2c_affinity_ceiling{suffix}.csv", index=False)
    for (arm, lam), p in preds.items():
        pd.DataFrame({"pair_id": val.pair_id.to_numpy(), "y_pred": p}).to_csv(
            PRED_DIR / f"stage2c_{tag}_{arm}_lam{lam:g}.csv", index=False)

    print(f"\nwrote reports/stage2c_runs_{tag}.csv, "
          f"reports/stage2c_deltas_{tag}.csv, "
          f"reports/stage2c_expansion_audit.csv, {len(preds)} prediction files")
    print(f"total fit wall time: {wall / 60:.1f} min  (CPU only, $0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
