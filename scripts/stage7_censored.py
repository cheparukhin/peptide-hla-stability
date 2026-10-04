"""Stage 7a: censored (Tobit) likelihood vs log1p-MSE, on validation.

    .venv/bin/python scripts/stage7_censored.py            # headline + sweep
    .venv/bin/python scripts/stage7_censored.py --quick    # 1 fold, 1 seed

The protocol is predeclared in ``reports/stage7_censored.md``, which was written
to disk before this script was first run. In particular the censoring threshold
(0.1 h, the assay's reporting floor) comes from the label distribution, **not**
from any validation score -- a threshold tuned on results is not a result.

Two arms, built with the stage 2b ensemble recipe (5 inner CV folds x 2
encodings x 3 seeds = 30 networks, mean-averaged), on the ``pep_pseudo`` input
set:

- ``mse``       -- ``pepstab.mlp.MLPRegressor``, squared error on ``log1p``
- ``censored``  -- ``pepstab.censored.CensoredMLPRegressor``, same class
                   hierarchy, same initialisation, same minibatch order

Features, architecture, L2, optimiser, seeds, fold assignment, target centring
and tuning budget (zero -- configs are inherited from stage 2) are identical.
The objective is the only difference, including the early-stopping criterion:
each arm stops on its own objective evaluated on the inner dev fold. A
dev-MSE-stopped censored arm is also run, so that choice is visible.

Validation only. The test split is never read.
"""

from __future__ import annotations

import os

# Pin BLAS to one thread **before numpy is imported** -- the thread pool is
# sized at import time, so setting these afterwards does nothing. numpy links
# against Accelerate on this machine (``np.show_config`` reports
# ``name: accelerate``), so ``VECLIB_MAXIMUM_THREADS`` is the one that actually
# binds; the others are set for portability. Without this, each of several
# concurrent agents' processes grabs 8 threads on an 8-core box and everything
# runs slower than it would single-threaded.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.censored import (  # noqa: E402
    DEFAULT_FLOOR_HOURS,
    SENSITIVITY_FLOOR_HOURS,
    CensoredMLPRegressor,
    censored_mask,
    censored_nll,
    floor_threshold,
)
from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import (  # noqa: E402
    MIN_WORTHWHILE_DELTA_SPEARMAN,
    describe_delta,
    eligible_alleles,
    paired_cluster_bootstrap,
    score,
)
from pepstab.features import ENCODINGS, build_features  # noqa: E402
from pepstab.mlp import MLPConfig, MLPRegressor  # noqa: E402
from scripts.baseline_ensemble import SELECTED, cv_folds  # noqa: E402
from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE, SEEDS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "reports"
PRED_DIR = REPO_ROOT / "preds"

INPUT_SET = "pep_pseudo"
N_FOLDS = 5


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

def fit_ensemble(arm: str, X_train: dict, y_train: np.ndarray,
                 cens_train: np.ndarray, X_val: dict, fold: np.ndarray,
                 floor_hours: float, seeds, n_folds: int,
                 stop_on: str = "own") -> tuple[np.ndarray, pd.DataFrame, list]:
    """Fit one arm's ensemble and return mean validation predictions.

    ``arm`` is ``"mse"`` or ``"censored"``. ``stop_on`` is ``"own"`` (each arm
    stops on its own objective) or ``"mse"`` (force the censored arm onto the
    MSE arm's stopping rule, for the sensitivity table).
    """
    members, rows, fitted = [], [], []
    for enc in ENCODINGS:
        hidden, l2 = SELECTED[(INPUT_SET, enc)]
        Xt, Xv = X_train[enc], X_val[enc]
        for k in range(n_folds):
            is_fit = fold != k
            for seed in seeds:
                cfg = MLPConfig(hidden=hidden, l2=l2, seed=seed,
                                max_epochs=MAX_EPOCHS, patience=PATIENCE)
                if arm == "mse":
                    model = MLPRegressor(cfg).fit(Xt[is_fit], y_train[is_fit],
                                                  Xt[~is_fit], y_train[~is_fit])
                    sigma = float("nan")
                elif arm == "censored":
                    model = CensoredMLPRegressor(
                        cfg, floor_hours=floor_hours,
                        stop_on="nll" if stop_on == "own" else "mse")
                    model.fit(Xt[is_fit], y_train[is_fit],
                              Xt[~is_fit], y_train[~is_fit],
                              cens_train[is_fit], cens_train[~is_fit])
                    sigma = model.sigma_
                else:
                    raise ValueError(f"unknown arm {arm!r}")
                members.append(model.predict(Xv))
                fitted.append(model)
                rows.append({
                    "arm": arm, "encoding": enc, "fold": k, "seed": seed,
                    "floor_hours": floor_hours, "stop_on": stop_on,
                    "best_epoch": model.best_epoch_, "epochs": model.n_epochs_,
                    "sigma": sigma, "dev_mse": round(float(model.dev_mse_), 5),
                    "fit_seconds": round(model.fit_seconds_, 2),
                })
    return np.mean(members, axis=0), pd.DataFrame(rows), fitted


# --------------------------------------------------------------------------
# calibration at the floor
# --------------------------------------------------------------------------

def floor_auroc(is_floor: np.ndarray, pred: np.ndarray) -> float:
    """AUROC for 'this row sits at the detection floor', ranking by -prediction.

    Loss-agnostic and available to both arms: it asks only whether the model
    orders floor rows below measured ones, which is the part of floor behaviour
    that *is* identifiable (ranking *within* the tied floor block is not).
    Computed from the Mann-Whitney U so ties get their proper half-credit.
    """
    pos, neg = -pred[is_floor], -pred[~is_floor]
    if not len(pos) or not len(neg):
        return float("nan")
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def calibration_row(name: str, pred: np.ndarray, y_val: np.ndarray,
                    is_floor: np.ndarray, threshold: float,
                    sigma: float | None = None,
                    p_floor: np.ndarray | None = None) -> dict:
    """One arm's behaviour at and above the detection floor.

    MAE is reported separately on floor rows and measured rows. Floor MAE is
    error against the *recorded* 0, which EVALUATION.md records as biased in
    both directions -- it is here to be read alongside the other columns, never
    on its own.
    """
    clipped = np.maximum(pred, 0.0)                       # predeclared
    recorded_median = np.where(pred <= threshold, 0.0, pred)  # added post hoc
    row = {
        "arm": name,
        "mae_all": float(np.abs(pred - y_val).mean()),
        "mae_floor_rows": float(np.abs(pred - y_val)[is_floor].mean()),
        "mae_measured_rows": float(np.abs(pred - y_val)[~is_floor].mean()),
        "mae_all_clipped_at_0": float(np.abs(clipped - y_val).mean()),
        "mae_all_recorded_median": float(np.abs(recorded_median - y_val).mean()),
        "mean_pred_floor_rows": float(pred[is_floor].mean()),
        "mean_pred_measured_rows": float(pred[~is_floor].mean()),
        "share_pred_at_or_below_threshold": float((pred <= threshold).mean()),
        "observed_floor_share": float(is_floor.mean()),
        "floor_auroc": floor_auroc(is_floor, pred),
    }
    if p_floor is not None:
        row["mean_p_floor"] = float(p_floor.mean())
        row["p_floor_calibration_error"] = float(p_floor.mean() - is_floor.mean())
    if sigma is not None:
        row["sigma"] = float(sigma)
    return row


def reliability_table(name: str, p_floor: np.ndarray, is_floor: np.ndarray,
                      n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed floor rate, in equal-count bins of the prediction."""
    order = np.argsort(p_floor)
    chunks = np.array_split(order, n_bins)
    rows = []
    for i, idx in enumerate(chunks):
        rows.append({
            "arm": name, "bin": i + 1, "n": len(idx),
            "mean_predicted_p_floor": float(p_floor[idx].mean()),
            "observed_floor_share": float(is_floor[idx].mean()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="1 fold, 1 seed, no sweep -- pipeline check only")
    ap.add_argument("--no-bootstrap", action="store_true")
    args = ap.parse_args()

    seeds = (SEEDS[0],) if args.quick else SEEDS
    n_folds = 1 if args.quick else N_FOLDS
    sweep = (DEFAULT_FLOOR_HOURS,) if args.quick else SENSITIVITY_FLOOR_HOURS
    suffix = "_quick" if args.quick else ""

    df = load_with_splits()
    train = df[df.split == "train"]
    val = df[df.split == "val"].sort_values("pair_id").reset_index(drop=True)

    fold = cv_folds(train, N_FOLDS)  # always the 5-fold assignment; --quick uses fold 0
    y_train = train.y_log1p.to_numpy()
    y_val = val.y_log1p.to_numpy()
    alleles = eligible_alleles(val.allele, val.y_log1p, split="val")

    X_train = {e: build_features(train, INPUT_SET, e) for e in ENCODINGS}
    X_val = {e: build_features(val, INPUT_SET, e) for e in ENCODINGS}

    c = floor_threshold(DEFAULT_FLOOR_HOURS)
    floor_val = censored_mask(val.thalf_hours, DEFAULT_FLOOR_HOURS)

    n_members = n_folds * len(ENCODINGS) * len(seeds)
    print(f"stage 7a: censored likelihood vs log1p-MSE  (input set {INPUT_SET})")
    print(f"  declared floor      : {DEFAULT_FLOOR_HOURS} h -> c = {c:.6f} on log1p")
    print(f"  censored fit rows   : {int(censored_mask(train.thalf_hours).sum()):,}"
          f" / {len(train):,} ({censored_mask(train.thalf_hours).mean():.1%})")
    print(f"  val rows            : {len(val):,}, {int(floor_val.sum()):,} at the floor"
          f" ({floor_val.mean():.1%}), {len(alleles)} eligible alleles")
    print(f"  ensemble            : {n_folds} folds x {len(ENCODINGS)} encodings"
          f" x {len(seeds)} seeds = {n_members} networks per arm\n")

    runs: list[pd.DataFrame] = []
    calib: list[dict] = []
    reliability: list[pd.DataFrame] = []
    sweep_rows: list[dict] = []
    preds: dict[str, np.ndarray] = {}
    started = time.perf_counter()

    # --- arm A: log1p MSE (threshold-independent, fitted once) --------------
    print("arm mse ...", flush=True)
    cens_train = censored_mask(train.thalf_hours, DEFAULT_FLOOR_HOURS)
    pred_mse, runs_mse, _ = fit_ensemble("mse", X_train, y_train, cens_train,
                                         X_val, fold, DEFAULT_FLOOR_HOURS,
                                         seeds, n_folds)
    runs.append(runs_mse)
    preds["mse"] = pred_mse
    s_mse = score("mse", val.allele, y_val, pred_mse, alleles)
    print(f"  median per-allele rho = {s_mse.median_spearman:+.4f}  "
          f"MAE = {s_mse.mae_log1p:.4f}  p@10 = {s_mse.median_precision_at_k:.4f}")

    # A homoscedastic scale for the MSE arm, so that it can be given the same
    # probabilistic reading. Estimated from its own inner-dev residual RMSE --
    # post hoc and clearly labelled, because the MSE objective does not fit a
    # scale at all.
    sigma_mse = float(np.sqrt(np.mean(runs_mse.dev_mse.to_numpy())))
    p_floor_mse = stats.norm.cdf((c - pred_mse) / sigma_mse)
    calib.append(calibration_row("mse", pred_mse, y_val, floor_val, c,
                                 sigma=sigma_mse, p_floor=p_floor_mse))
    reliability.append(reliability_table("mse (post-hoc sigma)", p_floor_mse, floor_val))

    # --- arm B: censored at the declared threshold, full budget -------------
    print(f"arm censored@{DEFAULT_FLOOR_HOURS:g}h (headline) ...", flush=True)
    cens_tr = censored_mask(train.thalf_hours, DEFAULT_FLOOR_HOURS)
    pred_cens, runs_cens, models_cens = fit_ensemble(
        "censored", X_train, y_train, cens_tr, X_val, fold,
        DEFAULT_FLOOR_HOURS, seeds, n_folds)
    runs.append(runs_cens)
    preds["censored"] = pred_cens
    s_cens = score("censored", val.allele, y_val, pred_cens, alleles)
    sigma_cens = float(runs_cens.sigma.mean())
    p_floor_cens = np.mean([m.prob_censored(X_val[e])
                            for m, e in zip(models_cens, runs_cens.encoding)], axis=0)
    calib.append(calibration_row("censored", pred_cens, y_val, floor_val, c,
                                 sigma=sigma_cens, p_floor=p_floor_cens))
    reliability.append(reliability_table("censored", p_floor_cens, floor_val))
    print(f"  median per-allele rho = {s_cens.median_spearman:+.4f}  "
          f"MAE = {s_cens.mae_log1p:.4f}  p@10 = {s_cens.median_precision_at_k:.4f}  "
          f"sigma = {sigma_cens:.3f}")

    # --- the threshold sweep, at a reduced but internally matched budget ----
    # Every point in the sweep -- including the declared 0.1 h and the MSE
    # reference line -- is fitted at the same reduced size, so the sweep is
    # comparable within itself. The headline above keeps the full budget.
    sweep_seeds = (SEEDS[0],)
    n_sweep = n_folds * len(ENCODINGS) * len(sweep_seeds)
    print(f"\nthreshold sweep at a reduced budget ({n_sweep} networks per point):",
          flush=True)

    pred_ref, runs_ref, _ = fit_ensemble("mse", X_train, y_train, cens_train,
                                         X_val, fold, DEFAULT_FLOOR_HOURS,
                                         sweep_seeds, n_folds)
    runs.append(runs_ref)
    s_ref = score("mse_sweep_ref", val.allele, y_val, pred_ref, alleles)
    sweep_rows.append({
        "floor_hours": np.nan, "threshold_log1p": np.nan, "primary": False,
        "arm": "mse (reference line)", "stop_on": "own", "n_censored_train": 0,
        "median_per_allele_spearman": round(s_ref.median_spearman, 4),
        "mae_log1p": round(s_ref.mae_log1p, 4),
        "median_precision_at_10": round(s_ref.median_precision_at_k, 4),
        "mean_sigma": np.nan,
        "floor_auroc": round(floor_auroc(floor_val, pred_ref), 4),
        "share_pred_at_or_below_c": round(float((pred_ref <= c).mean()), 4),
    })
    print(f"  mse reference          rho = {s_ref.median_spearman:+.4f}  "
          f"MAE = {s_ref.mae_log1p:.4f}")

    for floor_hours in sweep:
        is_primary = floor_hours == DEFAULT_FLOOR_HOURS
        tag = f"censored@{floor_hours:g}h"
        cens_tr = censored_mask(train.thalf_hours, floor_hours)
        pred, runs_c, models = fit_ensemble("censored", X_train, y_train, cens_tr,
                                            X_val, fold, floor_hours, sweep_seeds,
                                            n_folds)
        runs.append(runs_c)
        del models
        s = score(tag, val.allele, y_val, pred, alleles)
        sigma = float(runs_c.sigma.mean())
        sweep_rows.append({
            "floor_hours": floor_hours,
            "threshold_log1p": round(floor_threshold(floor_hours), 6),
            "primary": is_primary,
            "arm": "censored", "stop_on": "own",
            "n_censored_train": int(cens_tr.sum()),
            "median_per_allele_spearman": round(s.median_spearman, 4),
            "mae_log1p": round(s.mae_log1p, 4),
            "median_precision_at_10": round(s.median_precision_at_k, 4),
            "mean_sigma": round(sigma, 4),
            "floor_auroc": round(floor_auroc(floor_val, pred), 4),
            "share_pred_at_or_below_c": round(float((pred <= c).mean()), 4),
        })
        print(f"  {tag:22s} rho = {s.median_spearman:+.4f}  "
              f"MAE = {s.mae_log1p:.4f}  p@10 = {s.median_precision_at_k:.4f}  "
              f"sigma = {sigma:.3f}  floor AUROC = {sweep_rows[-1]['floor_auroc']:.4f}")

    # --- the stopping-rule control, same reduced budget --------------------
    cens_tr = censored_mask(train.thalf_hours, DEFAULT_FLOOR_HOURS)
    pred_sm, runs_sm, _ = fit_ensemble("censored", X_train, y_train, cens_tr,
                                       X_val, fold, DEFAULT_FLOOR_HOURS,
                                       sweep_seeds, n_folds, stop_on="mse")
    runs.append(runs_sm)
    s_sm = score("censored_devmse", val.allele, y_val, pred_sm, alleles)
    sweep_rows.append({
        "floor_hours": DEFAULT_FLOOR_HOURS,
        "threshold_log1p": round(c, 6), "primary": False,
        "arm": "censored", "stop_on": "dev MSE",
        "n_censored_train": int(cens_tr.sum()),
        "median_per_allele_spearman": round(s_sm.median_spearman, 4),
        "mae_log1p": round(s_sm.mae_log1p, 4),
        "median_precision_at_10": round(s_sm.median_precision_at_k, 4),
        "mean_sigma": round(float(runs_sm.sigma.mean()), 4),
        "floor_auroc": round(floor_auroc(floor_val, pred_sm), 4),
        "share_pred_at_or_below_c": round(float((pred_sm <= c).mean()), 4),
    })
    preds["censored_devmse"] = pred_sm
    print(f"  {'censored, dev-MSE stop':22s} rho = {s_sm.median_spearman:+.4f}  "
          f"MAE = {s_sm.mae_log1p:.4f}")

    wall = time.perf_counter() - started

    # --- paired comparison --------------------------------------------------
    boot = None
    if not args.no_bootstrap:
        print("\npaired cluster bootstrap (censored - mse) ...", flush=True)
        t0 = time.perf_counter()
        boot = paired_cluster_bootstrap(val.cluster_id, val.allele, y_val,
                                        preds["mse"], preds["censored"], alleles)
        print(f"  delta median per-allele rho = {boot['delta_median_spearman']:+.4f}  "
              f"95% CI [{boot['ci95'][0]:+.4f}, {boot['ci95'][1]:+.4f}]")
        print(f"  verdict: {describe_delta(boot)}  ({time.perf_counter() - t0:.0f}s)")

    # --- held-out censored NLL ---------------------------------------------
    # The censored arm's own objective, scored on validation. The MSE arm needs a
    # scale to be read this way at all, which is the post-hoc sigma above.
    nll_rows = [
        {"arm": "mse (post-hoc sigma)", "sigma": round(sigma_mse, 4),
         "val_censored_nll": round(float(censored_nll(
             y_val, floor_val, preds["mse"], sigma_mse, c).mean()), 4)},
        {"arm": "censored", "sigma": round(sigma_cens, 4),
         "val_censored_nll": round(float(censored_nll(
             y_val, floor_val, preds["censored"], sigma_cens, c).mean()), 4)},
    ]

    # --- write --------------------------------------------------------------
    REPORT_DIR.mkdir(exist_ok=True)
    PRED_DIR.mkdir(exist_ok=True)
    pd.concat(runs, ignore_index=True).to_csv(
        REPORT_DIR / f"stage7_censored_runs{suffix}.csv", index=False)
    pd.DataFrame(sweep_rows).to_csv(
        REPORT_DIR / f"stage7_censored_sensitivity{suffix}.csv", index=False)
    pd.DataFrame(calib).to_csv(
        REPORT_DIR / f"stage7_censored_calibration{suffix}.csv", index=False)
    pd.concat(reliability, ignore_index=True).to_csv(
        REPORT_DIR / f"stage7_censored_reliability{suffix}.csv", index=False)

    headline = {
        "split": "val",
        "input_set": INPUT_SET,
        "floor_hours_declared": DEFAULT_FLOOR_HOURS,
        "threshold_log1p": c,
        "censoring_rule": "thalf_hours < floor_hours",
        "n_censored_train": int(censored_mask(train.thalf_hours).sum()),
        "n_val_rows": len(val),
        "n_val_floor_rows": int(floor_val.sum()),
        "n_eligible_alleles": len(alleles),
        "ensemble_members_per_arm": n_members,
        "seeds": list(seeds),
        "min_worthwhile_delta_spearman": MIN_WORTHWHILE_DELTA_SPEARMAN,
        "mse": s_mse.as_row(),
        "censored": s_cens.as_row(),
        "val_censored_nll": nll_rows,
        "wall_seconds_fitting": round(wall, 1),
    }
    if boot is not None:
        headline["paired_bootstrap"] = {
            "delta_median_spearman": boot["delta_median_spearman"],
            "ci95": list(boot["ci95"]),
            "n_boot": boot["n_boot"],
            "panel_size_full": boot["panel_size_full"],
            "verdict": describe_delta(boot),
        }
    (REPORT_DIR / f"stage7_censored_headline{suffix}.json").write_text(
        json.dumps(headline, indent=2, default=float) + "\n")

    for name, p in preds.items():
        pd.DataFrame({"pair_id": val.pair_id.to_numpy(), "y_pred": p}).to_csv(
            PRED_DIR / f"stage7_{name}{suffix}.csv", index=False)

    print(f"\nwrote reports/stage7_censored_*{suffix}.csv/.json and "
          f"{len(preds)} prediction files in preds/")
    n_networks = int(sum(len(r) for r in runs))
    print(f"total fit wall time: {wall / 60:.1f} min over {n_networks} networks "
          f"({wall / max(n_networks, 1):.1f} s each)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
