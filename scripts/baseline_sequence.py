"""Stage 2: supervised sequence baselines on one-hot and BLOSUM62 encodings.

    .venv/bin/python scripts/baseline_sequence.py              # val, full grid
    .venv/bin/python scripts/baseline_sequence.py --quick      # one config, 1 seed

Six arms -- {one-hot, BLOSUM62} x {peptide, peptide+pseudosequence,
peptide+domain} -- each given the same tuning budget: a 4-point MLP grid at 3
seeds, plus a 4-point ridge alpha sweep as the linear reference. Writes every
run to ``reports/stage2_runs.csv``, the selected model per arm to
``reports/stage2_summary.csv``, and predictions to ``preds/``. A partial run
(``--quick`` or ``--arms``) writes ``*_partial`` filenames instead and does not
touch ``preds/seq_baseline.csv``: only the whole grid can say which arm won.

**Every model fits on the same rows.** The MLP needs a held-out fold to stop on,
and that fold is cut from the training split along *whole peptide clusters* --
the same grouping the frozen splits use, because a random fold would put
near-duplicate peptides on both sides and tune the epoch count on leaked rows.
Ridge then also fits on the fit fold and picks alpha on the dev fold, rather
than refitting on all of train: a 10% difference in training rows between the
arms would confound every comparison downstream.

The frozen validation split is used for one thing only -- choosing a config per
arm. Test is untouched; stage 6 scores it once.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import eligible_alleles, score  # noqa: E402
from pepstab.features import (  # noqa: E402
    ENCODINGS,
    INPUT_SETS,
    build_features,
    n_features,
)
from pepstab.mlp import MLPConfig, MLPRegressor  # noqa: E402
from pepstab.splits import assign_clusters  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = REPO_ROOT / "preds"
REPORT_DIR = REPO_ROOT / "reports"

#: The inner fit/dev cut inside the training split, by row count. Only the
#: epoch count (MLP) and alpha (ridge) are chosen on ``dev``.
DEV_FRACTION = 0.10

#: The tuning grid. Identical for every arm, so no arm can win on budget.
#: One narrow and one wider network, each at light and heavy weight decay.
HIDDEN_GRID = ((64,), (256, 64))
L2_GRID = (1e-5, 1e-3)
SEEDS = (0, 1, 2)

#: Ridge reference. Spans four orders of magnitude; the selected value is
#: reported so a boundary hit is visible.
ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)

#: Epoch ceiling. Set high enough that patience, not the cap, ends every run --
#: at 120 the slowest arm (BLOSUM + full domain) still had runs stopping on the
#: cap, so its lower score could not be told apart from undertraining.
MAX_EPOCHS = 300
PATIENCE = 15


def inner_folds(train: pd.DataFrame) -> pd.Series:
    """Split the training rows into ``fit`` and ``dev`` by whole clusters.

    Reuses the stage 1 water-fill so the cut is deterministic and respects the
    same peptide-cluster boundaries as the frozen splits. Returns a Series of
    ``"fit"``/``"dev"`` aligned to ``train``'s index.
    """
    weights = train.groupby("cluster_id").size()
    fold_by_cluster = assign_clusters(weights, fractions=(1 - DEV_FRACTION, DEV_FRACTION),
                                      names=("fit", "dev"))
    return train["cluster_id"].map(fold_by_cluster)


def summarise(name: str, truth: pd.DataFrame, y_pred: np.ndarray,
              alleles: list[str]) -> dict:
    """Validation metrics for one fitted model, through the shared contract."""
    s = score(name, truth.allele, truth.y_log1p.to_numpy(), y_pred, alleles)
    row = s.as_row()
    return {k: row[k] for k in ("median_per_allele_spearman", "iqr_low", "iqr_high",
                                "mae_log1p", "median_precision_at_10",
                                "pooled_spearman", "pooled_pearson_log1p")}


def run_arm(input_set: str, encoding: str, df: pd.DataFrame, fold: pd.Series,
            sel: pd.DataFrame, out: pd.DataFrame, sel_alleles: list[str],
            configs: list[tuple[tuple[int, ...], float]], seeds: tuple[int, ...],
            ) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Fit every config x seed for one arm. Returns run records and predictions.

    ``sel`` is the validation split (where configs are chosen); ``out`` is the
    split predictions are written for. They are the same frame unless
    ``--split test``.
    """
    arm = f"{encoding}_{input_set}"
    train = df[df.split == "train"]
    is_fit = (fold == "fit").to_numpy()

    t0 = time.perf_counter()
    X_train = build_features(train, input_set, encoding)
    X_sel = build_features(sel, input_set, encoding)
    X_out = X_sel if out is sel else build_features(out, input_set, encoding)
    feature_seconds = time.perf_counter() - t0

    y_train = train.y_log1p.to_numpy()
    X_fit, y_fit = X_train[is_fit], y_train[is_fit]
    X_dev, y_dev = X_train[~is_fit], y_train[~is_fit]

    records: list[dict] = []
    preds: dict[str, np.ndarray] = {}
    shared = {
        "arm": arm, "input_set": input_set, "encoding": encoding,
        "n_features": n_features(input_set), "n_fit_rows": int(is_fit.sum()),
        "n_dev_rows": int((~is_fit).sum()), "feature_seconds": round(feature_seconds, 3),
    }

    # --- ridge reference: alpha on dev, then scored on the selection split ---
    best_alpha, best_dev = None, np.inf
    for alpha in ALPHA_GRID:
        r = Ridge(alpha=alpha).fit(X_fit, y_fit)
        dev_mse = float(np.mean((r.predict(X_dev) - y_dev) ** 2))
        if dev_mse < best_dev:
            best_alpha, best_dev = alpha, dev_mse
    t0 = time.perf_counter()
    ridge = Ridge(alpha=best_alpha).fit(X_fit, y_fit)
    ridge_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    ridge_sel = ridge.predict(X_sel)
    ridge_infer = time.perf_counter() - t0
    name = f"ridge_{arm}"
    records.append({
        **shared, "model": name, "family": "ridge", "config": f"alpha={best_alpha:g}",
        "seed": 0, "n_parameters": int(X_fit.shape[1] + 1), "epochs": np.nan,
        "best_epoch": np.nan, "dev_mse": round(best_dev, 5),
        "fit_seconds": round(ridge_seconds, 2),
        "infer_seconds_per_1k": round(1000 * ridge_infer / len(X_sel), 4),
        **summarise(name, sel, ridge_sel, sel_alleles),
    })
    preds[name] = ridge.predict(X_out)
    print(f"  {name:34s} alpha={best_alpha:<6g} "
          f"rho={records[-1]['median_per_allele_spearman']:+.4f} "
          f"({records[-1]['fit_seconds']:.1f}s)")

    # --- MLP grid -------------------------------------------------------------
    for (hidden, l2), seed in itertools.product(configs, seeds):
        cfg = MLPConfig(hidden=hidden, l2=l2, seed=seed,
                        max_epochs=MAX_EPOCHS, patience=PATIENCE)
        model = MLPRegressor(cfg).fit(X_fit, y_fit, X_dev, y_dev)
        t0 = time.perf_counter()
        y_sel = model.predict(X_sel)
        infer = time.perf_counter() - t0
        name = f"mlp_{arm}_{cfg.label()}_s{seed}"
        records.append({
            **shared, "model": name, "family": "mlp", "config": cfg.label(),
            "seed": seed, "n_parameters": model.n_parameters(),
            "epochs": model.n_epochs_, "best_epoch": model.best_epoch_,
            "dev_mse": round(model.dev_mse_, 5),
            "fit_seconds": round(model.fit_seconds_, 2),
            "infer_seconds_per_1k": round(1000 * infer / len(X_sel), 4),
            **summarise(name, sel, y_sel, sel_alleles),
        })
        preds[name] = model.predict(X_out) if X_out is not X_sel else y_sel
        print(f"  {name:34s} rho={records[-1]['median_per_allele_spearman']:+.4f} "
              f"dev_mse={model.dev_mse_:.4f} ep={model.best_epoch_}/{model.n_epochs_} "
              f"({model.fit_seconds_:.1f}s)")
    return records, preds


def select(runs: pd.DataFrame) -> pd.DataFrame:
    """Per arm and family, the config with the best mean validation rho.

    Averaged over seeds, then the seed spread is reported next to it: a gap
    smaller than the spread is not a result (EVALUATION.md).
    """
    grouped = runs.groupby(["arm", "input_set", "encoding", "family", "config"])
    agg = grouped.agg(
        n_seeds=("seed", "size"),
        rho_mean=("median_per_allele_spearman", "mean"),
        rho_min=("median_per_allele_spearman", "min"),
        rho_max=("median_per_allele_spearman", "max"),
        mae_mean=("mae_log1p", "mean"),
        p10_mean=("median_precision_at_10", "mean"),
        pooled_mean=("pooled_spearman", "mean"),
        fit_seconds_mean=("fit_seconds", "mean"),
        infer_per_1k=("infer_seconds_per_1k", "mean"),
        n_parameters=("n_parameters", "max"),
        n_features=("n_features", "max"),
    ).reset_index()
    best = agg.sort_values("rho_mean", ascending=False).groupby(
        ["arm", "family"], as_index=False).head(1)
    return best.sort_values(["family", "rho_mean"], ascending=[True, False])


def reference_baselines(df: pd.DataFrame, sel: pd.DataFrame, out: pd.DataFrame,
                        sel_alleles: list[str]) -> tuple[list[dict], dict]:
    """Training global mean and training allele mean, for the same table.

    These have no hyperparameters and nothing to stop, so they use the *whole*
    training split -- matching ``scripts/baseline_constant.py`` exactly. Both are
    constant within an allele, so every per-allele Spearman is undefined and the
    panel median is exactly 0 by the predeclared rule. That is the point: the
    primary metric is measured against chance.
    """
    train = df[df.split == "train"]
    global_mean = float(train.y_log1p.mean())
    allele_mean = train.groupby("allele").y_log1p.mean()
    records, preds = [], {}
    for name, series_sel, series_out in [
        ("global_mean", pd.Series(global_mean, index=sel.index),
         pd.Series(global_mean, index=out.index)),
        ("allele_mean", sel.allele.map(allele_mean).fillna(global_mean),
         out.allele.map(allele_mean).fillna(global_mean)),
    ]:
        records.append({
            "arm": "reference", "input_set": "-", "encoding": "-", "model": name,
            "family": "reference", "config": "train split", "seed": 0,
            "n_features": 0, "n_parameters": 1 if name == "global_mean" else len(allele_mean),
            "n_fit_rows": len(train), "n_dev_rows": 0, "epochs": np.nan,
            "best_epoch": np.nan, "dev_mse": np.nan, "feature_seconds": 0.0,
            "fit_seconds": 0.0, "infer_seconds_per_1k": 0.0,
            **summarise(name, sel, series_sel.to_numpy(dtype=float), sel_alleles),
        })
        preds[name] = series_out.to_numpy(dtype=float)
    return records, preds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="which split to write predictions for (default: val). "
                         "Configs are always selected on val.")
    ap.add_argument("--quick", action="store_true",
                    help="one config, one seed -- for checking the pipeline runs. "
                         "Writes *_partial reports; leaves the canonical ones alone.")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="limit to these arms, e.g. onehot_pep_pseudo. Also "
                         "writes *_partial reports.")
    args = ap.parse_args()

    if args.split == "test":
        print("!! Writing TEST-split predictions. Per EVALUATION.md the test "
              "set is scored once, at stage 6.\n")

    df = load_with_splits()
    train = df[df.split == "train"]
    fold = inner_folds(train)
    sel = df[df.split == "val"].sort_values("pair_id").reset_index(drop=True)
    out = sel if args.split == "val" else (
        df[df.split == args.split].sort_values("pair_id").reset_index(drop=True))
    sel_alleles = eligible_alleles(sel.allele, sel.y_log1p, split="val")

    n_dev = int((fold == "dev").sum())
    n_dev_clusters = train.loc[(fold == "dev").to_numpy(), "cluster_id"].nunique()
    print(f"train={len(train):,} rows -> fit={len(train) - n_dev:,} / dev={n_dev:,} "
          f"({n_dev / len(train):.1%}, {n_dev_clusters:,} whole peptide clusters)")
    print(f"selection split=val  rows={len(sel):,}  eligible alleles={len(sel_alleles)}")
    print(f"prediction split={args.split}  rows={len(out):,}\n")

    configs = [(HIDDEN_GRID[0], L2_GRID[0])] if args.quick else [
        (h, l2) for h in HIDDEN_GRID for l2 in L2_GRID]
    seeds = (0,) if args.quick else SEEDS
    arms = [(i, e) for e in ENCODINGS for i in INPUT_SETS]
    if args.arms:
        arms = [(i, e) for i, e in arms if f"{e}_{i}" in set(args.arms)]
        if not arms:
            raise SystemExit(f"no arm matched {args.arms}")

    records, all_preds = reference_baselines(df, sel, out, sel_alleles)
    print("reference baselines (constant within allele, so panel median is 0 "
          "by the predeclared rule):")
    for r in records:
        print(f"  {r['model']:34s} rho={r['median_per_allele_spearman']:+.4f} "
              f"mae={r['mae_log1p']:.4f} p@10={r['median_precision_at_10']:.3f}")

    started = time.perf_counter()
    for input_set, encoding in arms:
        print(f"\narm {encoding}_{input_set} "
              f"({n_features(input_set):,} features):")
        arm_records, arm_preds = run_arm(input_set, encoding, df, fold, sel, out,
                                         sel_alleles, configs, seeds)
        records.extend(arm_records)
        all_preds.update(arm_preds)
    wall = time.perf_counter() - started

    # A partial run writes to its own filenames. The canonical report is the
    # whole grid, and a one-arm `--quick` check silently overwriting it would
    # leave reports/stage2_baselines.md describing numbers that are no longer
    # on disk.
    partial = args.quick or bool(args.arms)
    suffix = "_partial" if partial else ""
    runs_path = REPORT_DIR / f"stage2_runs{suffix}.csv"
    summary_path = REPORT_DIR / f"stage2_summary{suffix}.csv"

    runs = pd.DataFrame(records)
    REPORT_DIR.mkdir(exist_ok=True)
    runs.to_csv(runs_path, index=False)

    model_runs = runs[runs.family != "reference"]
    best = select(model_runs) if len(model_runs) else pd.DataFrame()
    if len(best):
        best.to_csv(summary_path, index=False)

    PRED_DIR.mkdir(exist_ok=True)
    written = []
    for _, row in best.iterrows():
        # One prediction file per arm and family: seed 0 of the selected config,
        # fixed in advance so the file is not picked by its own score.
        src = (f"ridge_{row.arm}" if row.family == "ridge"
               else f"mlp_{row.arm}_{row.config}_s{seeds[0]}")
        if src not in all_preds:
            continue
        dest = PRED_DIR / f"{row.family}_{row.arm}{suffix}.csv"
        pd.DataFrame({"pair_id": out.pair_id.to_numpy(),
                      "y_pred": all_preds[src]}).to_csv(dest, index=False)
        written.append((f"{row.family}_{row.arm}", src, row.rho_mean))

    print(f"\nselected per arm (mean validation rho over {len(seeds)} seed(s)):")
    cols = ["arm", "family", "config", "rho_mean", "rho_min", "rho_max",
            "mae_mean", "p10_mean", "fit_seconds_mean", "n_parameters"]
    if len(best):
        print(best[cols].to_string(index=False))

    # The stage 2 headline: the best sequence baseline, under a stable name, so
    # stage 3 compares against it without knowing which arm won. Only a full
    # grid gets to claim that name -- a partial run has not seen the other arms.
    if written and not partial:
        head_name, head_src, _ = max(written, key=lambda t: t[2])
        head = PRED_DIR / "seq_baseline.csv"
        pd.DataFrame({"pair_id": out.pair_id.to_numpy(),
                      "y_pred": all_preds[head_src]}).to_csv(head, index=False)
        (REPORT_DIR / "stage2_headline.json").write_text(json.dumps({
            "split": args.split, "model": head_src, "arm_file": head_name,
            "selected_on": "val median per-allele Spearman, mean over seeds",
            "dev_fraction": DEV_FRACTION, "seeds": list(seeds),
            "grid": {"hidden": [list(h) for h in HIDDEN_GRID], "l2": list(L2_GRID),
                     "alpha": list(ALPHA_GRID), "max_epochs": MAX_EPOCHS,
                     "patience": PATIENCE},
        }, indent=2) + "\n")
        print(f"\nheadline sequence baseline: {head_src}")
        print(f"  -> {head.relative_to(REPO_ROOT)} "
              f"(+ {len(written)} per-arm files in preds/)")

    if partial:
        print(f"\npartial run ({len(arms)} of {len(INPUT_SETS) * len(ENCODINGS)} "
              "arms): canonical reports and preds/seq_baseline.csv left alone.")
    print(f"\nwrote {runs_path.relative_to(REPO_ROOT)} ({len(runs)} runs), "
          f"{summary_path.relative_to(REPO_ROOT)}")
    print(f"total fit wall time: {wall / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
