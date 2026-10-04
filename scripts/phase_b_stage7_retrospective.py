"""Phase B: the matched-budget comparison stage 7a should have made.

    .venv/bin/python scripts/phase_b_stage7_retrospective.py
    .venv/bin/python scripts/phase_b_stage7_retrospective.py --analyse-only

Stage 7a reported that swapping only the censored arm's stopping rule "gives back
almost all of the ranking ... while roughly halving the calibration error", and
called it a control rather than a result because it had **no confidence
interval**. It also had something worse, which
``reports/further_research/phase_b_declaration.md`` §1.1 documents: the §6.4
table's ranking column is the reduced 10-member budget while its Brier, ECE and
NLL columns for two of three rows are the 30-member values, so the published
calibration comparison is between arms of different size.

This script fixes both. All three arms are refitted at the **same 30-member
budget** under stage 7a's protocol, every member's ``mu`` and ``sigma`` is kept so
that one declared ``P(floor)`` construction can be applied to all of them, and
every comparison gets a paired peptide-cluster bootstrap interval.

It deliberately changes nothing about the fits: same features, architecture, L2,
optimiser, seeds, inner folds, target centring and zero tuning budget. The
ensemble means are asserted against the shipped ``preds/stage7_*.csv`` files, so a
drift in any of that fails loudly instead of quietly producing new numbers.

**Validation split only. The test split is never read.** No model is selected
here: this is a correction of a published record, and the Phase B assessment
proper runs on the training-only nested harness.
"""

from __future__ import annotations

import os

# Pin BLAS to one thread **before numpy is imported** -- the pool is sized at
# import time. This project has recorded that thread count moves a single
# network's result by about 0.02 median Spearman at a fixed seed; under the
# concurrent load this machine is carrying that becomes nondeterministic rather
# than a constant offset, which is the worst case for a paired comparison.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pepstab import calibration as cal  # noqa: E402
from pepstab.censored import (  # noqa: E402
    DEFAULT_FLOOR_HOURS,
    CensoredMLPRegressor,
    censored_mask,
    floor_threshold,
)
from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import (  # noqa: E402
    describe_delta,
    eligible_alleles,
    paired_cluster_bootstrap,
    score,
)
from pepstab.features import ENCODINGS, build_features  # noqa: E402
from pepstab.mlp import MLPConfig, MLPRegressor  # noqa: E402
from scripts.baseline_ensemble import SELECTED, cv_folds  # noqa: E402
from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE, SEEDS  # noqa: E402

OUT_DIR = REPO_ROOT / "reports" / "further_research"
CACHE_STEM = "phase_b_stage7_members"
INPUT_SET = "pep_pseudo"
N_FOLDS = 5

#: The three arms, at one matched budget. ``stop_on`` is the early-stopping
#: criterion; ``None`` for the MSE arm, which has only one.
ARMS = (
    ("mse", None, "log1p-MSE reference"),
    ("censored", "nll", "censored, stopped on dev NLL"),
    ("censored", "mse", "censored, stopped on dev MSE"),
)

#: Which shipped prediction file each arm's 30-member mean must reproduce.
SHIPPED = {"mse": "stage7_mse", "censored:nll": "stage7_censored"}


def arm_key(arm: str, stop_on: str | None) -> str:
    return arm if stop_on is None else f"{arm}:{stop_on}"


# --------------------------------------------------------------------------
# fitting


def fit_members(arm: str, stop_on: str | None, X_train: dict, y_train: np.ndarray,
                cens_train: np.ndarray, X_val: dict, fold: np.ndarray,
                n_folds: int = N_FOLDS,
                seeds=SEEDS) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Fit one arm's 30 members; return per-member ``mu``, per-member ``sigma``, records.

    Member order is ``encoding -> inner fold -> seed``, matching
    ``scripts/stage7_censored.py::fit_ensemble`` exactly so that the mean over
    members is bit-for-bit the shipped prediction file.

    The MSE arm has no fitted scale, so its per-member ``sigma`` is
    ``sqrt(dev_mse)`` -- that member's own held-out inner-fold residual RMSE.
    Training-only, and the same construction stage 7a used for its comparator,
    which is what keeps this comparison like-for-like. (It is mildly optimistic,
    being the best-epoch value early stopping selected on; the Phase B
    assessment proper uses ``inner_oof_30`` residuals instead, which no stopping
    decision touched.)
    """
    mus, sigmas, rows = [], [], []
    for enc in ENCODINGS:
        hidden, l2 = SELECTED[(INPUT_SET, enc)]
        Xt, Xv = X_train[enc], X_val[enc]
        for k in range(n_folds):
            is_fit = fold != k
            for seed in seeds:
                cfg = MLPConfig(hidden=hidden, l2=l2, seed=seed,
                                max_epochs=MAX_EPOCHS, patience=PATIENCE)
                if arm == "mse":
                    model = MLPRegressor(cfg).fit(
                        Xt[is_fit], y_train[is_fit], Xt[~is_fit], y_train[~is_fit])
                    sigma = float(np.sqrt(model.dev_mse_))
                else:
                    model = CensoredMLPRegressor(cfg, floor_hours=DEFAULT_FLOOR_HOURS,
                                                 stop_on=stop_on)
                    model.fit(Xt[is_fit], y_train[is_fit], Xt[~is_fit],
                              y_train[~is_fit], cens_train[is_fit], cens_train[~is_fit])
                    sigma = float(model.sigma_)
                mus.append(model.predict(Xv))
                sigmas.append(sigma)
                rows.append({
                    "arm": arm, "stop_on": stop_on or "mse (only rule)",
                    "encoding": enc, "inner_fold": k, "seed": seed,
                    "best_epoch": model.best_epoch_, "epochs": model.n_epochs_,
                    "sigma": sigma, "dev_mse": float(model.dev_mse_),
                    "fit_seconds": round(model.fit_seconds_, 2),
                })
                print(f"    {arm_key(arm, stop_on):22s} {enc:7s} fold {k} seed {seed}"
                      f"  epoch {model.best_epoch_:3d}  sigma {sigma:.4f}", flush=True)
    return np.asarray(mus), np.asarray(sigmas), pd.DataFrame(rows)


# --------------------------------------------------------------------------
# metrics


def arm_metrics(label: str, n_members: int, mus: np.ndarray, sigmas: np.ndarray,
                val: pd.DataFrame, is_floor: np.ndarray, is_censored: np.ndarray,
                alleles: list[str], threshold: float, rule: str) -> dict:
    """Every declared metric for one arm, under one ``P(floor)`` construction."""
    mu = mus.mean(axis=0)
    p = (cal.prob_floor_mixture(mus, sigmas, threshold) if rule == "mixture"
         else cal.prob_floor_plugin(mu, float(sigmas.mean()), threshold))
    y_log1p = np.log1p(val.thalf_hours.values)
    sc = score(label, val.allele, y_log1p, mu, alleles=alleles)
    ece_fixed, _ = cal.fixed_bin_ece(is_floor, p)
    return {
        "arm": label, "n_members": n_members, "p_floor_rule": rule,
        "median_per_allele_spearman": round(sc.median_spearman, 4),
        "mae_log1p": round(sc.mae_log1p, 4),
        "median_precision_at_10": round(sc.median_precision_at_k, 4),
        "brier": cal.brier(is_floor, p),
        "ece_fixed_bin": ece_fixed,
        "ece_equal_count": cal.equal_count_ece(is_floor, p),
        "log_loss": cal.log_loss(is_floor, p),
        "floor_auroc_from_p": cal.floor_auroc(is_floor, p),
        "floor_auroc_from_mu": cal.floor_auroc(is_floor, -mu),
        "censored_nll": cal.censored_nll_mean(mu, y_log1p, is_censored,
                                             float(sigmas.mean()), threshold),
        "mean_p_floor": float(p.mean()),
        "observed_floor_share": float(is_floor.mean()),
        "share_mu_at_or_below_threshold": float((mu <= threshold).mean()),
        "mean_sigma": float(sigmas.mean()),
    }


def paired_rows(ref_label: str, ref: dict, arm_label: str, arm: dict,
                val: pd.DataFrame, is_floor: np.ndarray, is_censored: np.ndarray,
                alleles: list[str], threshold: float, rule: str,
                n_members: int, n_boot: int) -> list[dict]:
    """Paired cluster-bootstrap intervals for one arm against the reference."""
    y_log1p = np.log1p(val.thalf_hours.values)
    out = []

    # Ranking goes through the frozen contract's own function, unchanged.
    rank = paired_cluster_bootstrap(val.cluster_id, val.allele, y_log1p,
                                    ref["mu"], arm["mu"], alleles=alleles,
                                    n_boot=n_boot)
    out.append({
        "comparison": f"{arm_label} vs {ref_label}", "n_members": n_members,
        "p_floor_rule": rule, "metric": "median_per_allele_spearman",
        "lower_is_better": False,
        "reference": round(float(ref["scores"]), 4), "arm": round(float(arm["scores"]), 4),
        "delta_improvement": rank["delta_median_spearman"],
        "ci_low": rank["ci95"][0], "ci_high": rank["ci95"][1],
        "verdict": describe_delta(rank),
    })

    for name, metric, pred_ref, pred_arm, lower in (
            ("brier", cal.brier, ref["p"], arm["p"], True),
            ("ece_fixed_bin", lambda y, p: cal.fixed_bin_ece(y, p)[0],
             ref["p"], arm["p"], True),
            ("ece_equal_count", cal.equal_count_ece, ref["p"], arm["p"], True),
            ("log_loss", cal.log_loss, ref["p"], arm["p"], True),
            ("floor_auroc_from_p", cal.floor_auroc, ref["p"], arm["p"], False),
    ):
        res = cal.paired_cluster_bootstrap_metric(
            val.cluster_id, metric, reference=pred_ref, arm=pred_arm,
            labels=is_floor, lower_is_better=lower, n_boot=n_boot)
        out.append({
            "comparison": f"{arm_label} vs {ref_label}", "n_members": n_members,
            "p_floor_rule": rule, "metric": name, "lower_is_better": lower,
            "reference": res["reference"], "arm": res["arm"],
            "delta_improvement": res["delta_improvement"],
            "ci_low": res["ci95"][0], "ci_high": res["ci95"][1],
            # min_delta = 0 because no worthwhile bar is declared for these
            # metrics on validation; the bar is derived on the harness's own rows.
            "verdict": describe_delta(res, min_delta=0.0),
        })

    # Censored NLL carries each arm's own fitted scale, so it gets the dedicated
    # bootstrap rather than the generic one -- scoring both arms at one shared
    # sigma would not be the comparison.
    nll = cal.paired_cluster_bootstrap_censored_nll(
        val.cluster_id, y_log1p=y_log1p, is_censored=is_censored,
        mu_reference=ref["mu"], sigma_reference=ref["sigma"],
        mu_arm=arm["mu"], sigma_arm=arm["sigma"], threshold=threshold,
        n_boot=n_boot)
    out.append({
        "comparison": f"{arm_label} vs {ref_label}", "n_members": n_members,
        "p_floor_rule": rule, "metric": "censored_nll", "lower_is_better": True,
        "reference": nll["reference"], "arm": nll["arm"],
        "delta_improvement": nll["delta_improvement"],
        "ci_low": nll["ci95"][0], "ci_high": nll["ci95"][1],
        "verdict": describe_delta(nll, min_delta=0.0),
    })
    return out


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyse-only", action="store_true",
                    help="reuse the cached member predictions instead of refitting")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--quick", action="store_true",
                    help="pilot: 1 inner fold, 1 seed, 200 resamples -- exercises "
                         "every code path end to end before the real run, per the "
                         "project's no-batch-job-without-a-pilot rule. Its numbers "
                         "are not comparable to anything and are written to "
                         "*_quick.csv so they cannot be mistaken for results.")
    args = ap.parse_args()

    n_folds = 1 if args.quick else N_FOLDS
    seeds = SEEDS[:1] if args.quick else SEEDS
    budgets = (len(ENCODINGS) * len(seeds) * n_folds,) if args.quick else (10, 30)
    suffix = "_quick" if args.quick else ""
    if args.quick:
        args.n_boot = 200

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_with_splits()
    train = df[df.split == "train"].reset_index(drop=True)
    val = df[df.split == "val"].reset_index(drop=True)
    threshold = floor_threshold(DEFAULT_FLOOR_HOURS)

    # Two definitions of "at the floor", declared in phase_b_declaration.md §2.
    # The metric scores the recorded code; the loss censors below the limit. On
    # validation they coincide exactly -- asserted, not assumed.
    is_floor = cal.floor_indicator(val.thalf_hours, "recorded_zero")
    is_censored = cal.floor_indicator(val.thalf_hours, "below_limit")
    assert np.array_equal(is_floor, is_censored), (
        "validation is expected to contain no rows in (0, 0.1); the two floor "
        "definitions must coincide there")

    y_train = train.y_log1p.to_numpy()
    cens_train = censored_mask(train.thalf_hours, DEFAULT_FLOOR_HOURS)
    fold = cv_folds(train, N_FOLDS)
    alleles = eligible_alleles(val.allele, np.log1p(val.thalf_hours.values), split="val")

    print(f"Phase B retrospective{' PILOT' if args.quick else ''}: {len(ARMS)} arms "
          f"x {n_folds} folds x {len(ENCODINGS)} encodings x {len(seeds)} seeds = "
          f"{len(ARMS) * n_folds * len(ENCODINGS) * len(seeds)} networks")
    print(f"  val rows {len(val):,}, floor {int(is_floor.sum()):,} "
          f"({is_floor.mean():.4f}), {len(alleles)} eligible alleles")
    print(f"  train censored rows {int(cens_train.sum()):,} / {len(train):,}\n")

    cache = OUT_DIR / f"{CACHE_STEM}{suffix}.npz"
    if args.analyse_only and cache.exists():
        print(f"reusing {cache.name}")
        blob = np.load(cache, allow_pickle=False)
        keys = [name[: -len("__mus")] for name in blob.files if name.endswith("__mus")]
        members = {k: {"mus": blob[f"{k}__mus"], "sigmas": blob[f"{k}__sigmas"]}
                   for k in keys}
        runs = pd.read_csv(OUT_DIR / f"{CACHE_STEM}{suffix}.csv")
    else:
        X_train = {e: build_features(train, INPUT_SET, e) for e in ENCODINGS}
        X_val = {e: build_features(val, INPUT_SET, e) for e in ENCODINGS}
        members, run_frames = {}, []
        started = time.perf_counter()
        for arm, stop_on, label in ARMS:
            print(f"  fitting {label} ...", flush=True)
            mus, sigmas, rows = fit_members(arm, stop_on, X_train, y_train,
                                            cens_train, X_val, fold,
                                            n_folds, seeds)
            members[arm_key(arm, stop_on)] = {"mus": mus, "sigmas": sigmas}
            run_frames.append(rows)
        runs = pd.concat(run_frames, ignore_index=True)
        runs.to_csv(OUT_DIR / f"{CACHE_STEM}{suffix}.csv", index=False)
        np.savez_compressed(cache, **{f"{k}__{f}": v[f]
                                      for k, v in members.items()
                                      for f in ("mus", "sigmas")})
        print(f"\n  fitting wall time {time.perf_counter() - started:.1f}s")

    # --- reproduction check against the shipped files ---------------------
    # Only meaningful at the full budget: the shipped files are 30-member means,
    # so comparing a 6-member pilot mean against them would always "fail" and
    # would teach nothing.
    repro = []
    if args.quick:
        print("\nreproduction check skipped: the pilot's mean is not a 30-member mean")
    else:
        print("\nreproduction against shipped preds/ (30-member means):")
    for key, stem in ({} if args.quick else SHIPPED).items():
        shipped = pd.read_csv(REPO_ROOT / "preds" / f"{stem}.csv")
        merged = val[["pair_id"]].merge(shipped, on="pair_id", how="left",
                                        validate="one_to_one")
        assert merged.y_pred.notna().all(), stem
        mine = members[key]["mus"].mean(axis=0)
        diff = float(np.abs(mine - merged.y_pred.values).max())
        repro.append({"arm": key, "shipped_file": f"preds/{stem}.csv",
                      "max_abs_diff": diff,
                      "reproduces": bool(diff < 5e-4)})
        print(f"  {key:22s} max |diff| = {diff:.2e}"
              f"  {'OK' if diff < 5e-4 else 'DOES NOT REPRODUCE'}")

    # --- metrics, at both budgets and under both P(floor) rules -----------
    # The 10-member budget is the seed-0 subset of the very same members, so the
    # budget decomposition costs no extra fits and is exactly nested.
    rows_metrics, rows_paired, rel_frames = [], [], []
    for n_members in budgets:
        for rule in cal.P_FLOOR_RULES:
            per_arm = {}
            for arm, stop_on, label in ARMS:
                key = arm_key(arm, stop_on)
                mus, sigmas = members[key]["mus"], members[key]["sigmas"]
                sel = (runs[(runs.arm == arm)
                            & (runs.stop_on == (stop_on or "mse (only rule)"))]
                       .reset_index(drop=True))
                keep = (np.ones(len(sel), dtype=bool) if n_members == len(sel)
                        else (sel.seed == SEEDS[0]).to_numpy())
                assert keep.sum() == n_members, (key, n_members, int(keep.sum()))
                m = arm_metrics(label, n_members, mus[keep], sigmas[keep], val,
                                is_floor, is_censored, alleles, threshold, rule)
                rows_metrics.append(m)
                mu = mus[keep].mean(axis=0)
                p = (cal.prob_floor_mixture(mus[keep], sigmas[keep], threshold)
                     if rule == "mixture"
                     else cal.prob_floor_plugin(mu, float(sigmas[keep].mean()), threshold))
                per_arm[label] = {"mu": mu, "p": p,
                                  "scores": m["median_per_allele_spearman"],
                                  "sigma": float(sigmas[keep].mean())}
                for binning in ("fixed", "equal_count"):
                    t = cal.reliability_table(is_floor, p, binning=binning, arm=label)
                    t.insert(0, "n_members", n_members)
                    t.insert(0, "p_floor_rule", rule)
                    rel_frames.append(t)

            ref_label = ARMS[0][2]
            for _, _, label in ARMS[1:]:
                rows_paired.extend(paired_rows(
                    ref_label, per_arm[ref_label], label, per_arm[label], val,
                    is_floor, is_censored, alleles, threshold, rule, n_members,
                    args.n_boot))

    pd.DataFrame(rows_metrics).to_csv(
        OUT_DIR / f"phase_b_stage7_retrospective_metrics{suffix}.csv", index=False)
    pd.DataFrame(rows_paired).to_csv(
        OUT_DIR / f"phase_b_stage7_retrospective_paired{suffix}.csv", index=False)
    pd.concat(rel_frames, ignore_index=True).to_csv(
        OUT_DIR / f"phase_b_stage7_retrospective_reliability{suffix}.csv", index=False)
    with open(OUT_DIR / f"phase_b_stage7_retrospective{suffix}.json", "w") as fh:
        json.dump({"split": "val", "n_val_rows": len(val),
                   "n_floor_rows": int(is_floor.sum()),
                   "n_eligible_alleles": len(alleles),
                   "floor_hours": DEFAULT_FLOOR_HOURS, "threshold_log1p": threshold,
                   "n_boot": args.n_boot, "seed": 20261003,
                   "reproduction": repro}, fh, indent=2)
    print(f"\nwrote 4 artifacts to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
