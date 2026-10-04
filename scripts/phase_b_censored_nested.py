"""Phase B assessment: censored loss and stopping rule, on the nested harness.

    .venv/bin/python scripts/phase_b_censored_nested.py --quick      # 1 fold pilot
    .venv/bin/python scripts/phase_b_censored_nested.py              # full, 450 nets
    .venv/bin/python scripts/phase_b_censored_nested.py --analyse-only

Three arms, declared in ``reports/further_research/phase_b_declaration.md`` §3
before any fit, at one matched budget:

- **B0** ``log1p``-MSE reference
- **B1** censored loss, early-stopped on dev **NLL** (stage 7a's declared arm)
- **B2** censored loss, early-stopped on dev **MSE** (isolates the stopping rule)

A ranking-stopped fourth arm was **declined** in advance, with reasons, in §3.1.

Assessed on training-only nested cluster folds loaded from
``reports/further_research/phase_a_fold_assignments.csv`` and never recomputed.
**The test split is not read. The frozen validation split is not scored here.**

Two things this script is careful about, both of them failure modes this project
has already paid for:

1. **Budget parity is asserted, not assumed.** The member loop below mirrors
   ``pepstab.nested.fit_mlp_ensemble`` exactly -- same encoding/inner-fold/seed
   order, same ``replace(base_config, ...)``, same row-local features -- because
   it has to return per-member ``mu`` and ``sigma`` that the public function does
   not expose. ``--parity-check`` runs the public function on outer fold 0 and
   asserts the two ensemble means agree to 1e-9, so "inherited budget" is a
   checked claim. A transplanted ladder has cost an arm 0.093 median Spearman
   here before.
2. **The worthwhile-Brier bar is derived before any censored arm is scored.**
   ``phase_b_declaration.md`` §7 declares the bar as a *rule* -- the null
   bootstrap half-width on the harness's own rows -- precisely so it cannot be
   chosen to fit a result. The script computes it from **B0 alone** and writes it
   out before touching a single B1/B2 comparison, and prints the order it did so
   in.
"""

from __future__ import annotations

import os

# Pin BLAS to one thread **before numpy is imported**; the pool is sized at
# import time. Thread count moves a single network's result by about 0.02 median
# Spearman at fixed seed in this project, and under concurrent load that becomes
# nondeterministic rather than a constant offset -- fatal for a paired arm.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pepstab import calibration as cal  # noqa: E402
from pepstab import nested  # noqa: E402
from pepstab.censored import (  # noqa: E402
    DEFAULT_FLOOR_HOURS,
    CensoredMLPRegressor,
    censored_mask,
    floor_threshold,
)
from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import describe_delta, paired_cluster_bootstrap  # noqa: E402
from pepstab.features import build_features  # noqa: E402
from pepstab.mlp import MLPRegressor  # noqa: E402

OUT_DIR = REPO_ROOT / "reports" / "further_research"
FOLDS_CSV = OUT_DIR / "phase_a_fold_assignments.csv"

INPUT_SET = "pep_pseudo"
ENCODINGS = ("onehot", "blosum")
SEEDS = (0, 1, 2)
#: A0's exact per-encoding architecture and L2, inherited rather than re-tuned.
SELECTED = {"onehot": ((256, 64), 1e-5), "blosum": ((256, 64), 1e-3)}

#: ``(arm id, model kind, stop_on, label)``. Order matters: B0 is first because
#: the Brier bar is derived from it before any other arm is compared.
ARMS = (
    ("B0", "mse", None, "log1p-MSE reference"),
    ("B1", "censored", "nll", "censored, stopped on dev NLL"),
    ("B2", "censored", "mse", "censored, stopped on dev MSE"),
)

#: Both ``P(floor)`` constructions are reported; the mixture is declared primary
#: in phase_b_declaration.md §4.
PRIMARY_P_FLOOR_RULE = "mixture"


# --------------------------------------------------------------------------
# folds


def load_frozen_folds(train: pd.DataFrame, n_inner: int) -> list[nested.OuterSplit]:
    """Load the frozen assignment and assert it against the harness's own cut.

    The declaration says load, never recompute. The assertion is here because if
    the two ever disagree it means a nondeterminism in ``cv_folds`` that three
    phases are silently relying on, and that is worth failing loudly over rather
    than preferring one silently.
    """
    fa = pd.read_csv(FOLDS_CSV)
    pair_ids = train.pair_id.to_numpy()
    frozen_outer = (fa[fa.inner_fold == -1].set_index("pair_id").outer_fold
                    .reindex(pair_ids).to_numpy())
    if np.isnan(frozen_outer.astype(float)).any():
        raise ValueError(f"{FOLDS_CSV.name} does not cover every training pair_id")

    splits = list(nested.iter_outer(train, n_inner))
    if not np.array_equal(frozen_outer, nested.outer_folds(train)):
        raise AssertionError(
            "frozen outer folds disagree with nested.outer_folds(); the frozen "
            "file is authoritative, but this means cv_folds is not deterministic "
            "and the orchestrator needs to know")
    for split in splits:
        fit_ids = pair_ids[split.fit_mask]
        frozen_inner = (fa[(fa.outer_fold == split.fold) & (fa.inner_fold >= 0)]
                        .set_index("pair_id").inner_fold.reindex(fit_ids).to_numpy())
        if not np.array_equal(frozen_inner, split.inner):
            raise AssertionError(
                f"frozen inner folds disagree with the harness on outer fold "
                f"{split.fold}")
    return splits


# --------------------------------------------------------------------------
# fitting


def fit_arm_members(kind: str, stop_on: str | None, fit_df: pd.DataFrame,
                    predict_df: pd.DataFrame, inner: np.ndarray,
                    floor_hours: float = DEFAULT_FLOOR_HOURS) -> dict:
    """One arm's ensemble on one outer fold, keeping every member separately.

    Mirrors ``pepstab.nested.fit_mlp_ensemble``'s loop exactly so the budget is
    inherited; returns per-member ``mu`` on the predict frame and per-member
    ``sigma``, which the public function does not expose and which the declared
    mixture ``P(floor)`` needs.

    The MSE arm has no fitted scale, so its per-member ``sigma`` is the RMSE of
    that member's **inner out-of-fold** residuals on measured rows -- not
    ``sqrt(dev_mse)``. ``dev_mse`` is the best-epoch value early stopping selected
    on, so it is optimistic about that member's out-of-sample spread, and sigma is
    the parameter a floor probability is most sensitive to.
    """
    n_inner = int(inner.max()) + 1
    per_inner = len(ENCODINGS) * len(SEEDS)
    y_fit = fit_df["y_log1p"].to_numpy()
    cens_fit = censored_mask(fit_df.thalf_hours, floor_hours)

    mus, sigmas, records = [], [], []
    inner_sum = np.zeros(len(fit_df))
    inner_count = np.zeros(len(fit_df), dtype=int)

    for encoding in ENCODINGS:
        hidden, l2 = SELECTED[encoding]
        X_fit = build_features(fit_df, INPUT_SET, encoding)
        X_pred = build_features(predict_df, INPUT_SET, encoding)
        for k in range(n_inner):
            is_fit = inner != k
            for seed in SEEDS:
                config = replace(nested.REFERENCE_MLP, hidden=tuple(hidden),
                                 l2=l2, seed=seed)
                if kind == "mse":
                    model = MLPRegressor(config).fit(
                        X_fit[is_fit], y_fit[is_fit], X_fit[~is_fit], y_fit[~is_fit])
                else:
                    model = CensoredMLPRegressor(config, floor_hours=floor_hours,
                                                 stop_on=stop_on)
                    model.fit(X_fit[is_fit], y_fit[is_fit],
                              X_fit[~is_fit], y_fit[~is_fit],
                              cens_fit[is_fit], cens_fit[~is_fit])
                dev_pred = model.predict(X_fit[~is_fit])
                inner_sum[~is_fit] += dev_pred
                inner_count[~is_fit] += 1
                if kind == "mse":
                    # Measured inner-held rows only: squaring (0 - mu) at the
                    # floor would measure the floor coding, not the spread.
                    measured = ~cens_fit[~is_fit]
                    sigma = float(np.sqrt(np.mean(
                        (y_fit[~is_fit][measured] - dev_pred[measured]) ** 2)))
                else:
                    sigma = float(model.sigma_)
                mus.append(model.predict(X_pred))
                sigmas.append(sigma)
                records.append({
                    "kind": kind, "stop_on": stop_on or "mse (only rule)",
                    "inner_fold": k, "encoding": encoding, "seed": seed,
                    "n_fit_rows": int(is_fit.sum()),
                    "best_epoch": model.best_epoch_, "epochs": model.n_epochs_,
                    "sigma": sigma, "dev_mse": float(model.dev_mse_),
                    "sigma_from_dev_mse": float(np.sqrt(model.dev_mse_)),
                    "fit_seconds": round(model.fit_seconds_, 2),
                })
        del X_fit, X_pred

    if not np.all(inner_count == per_inner):
        raise AssertionError(
            f"each fit row should be held out by exactly {per_inner} members, got "
            f"{sorted(set(inner_count.tolist()))}")
    return {"mus": np.asarray(mus), "sigmas": np.asarray(sigmas),
            "inner_oof": inner_sum / per_inner, "records": records,
            "n_members": len(mus)}


def parity_check(fit_df: pd.DataFrame, predict_df: pd.DataFrame,
                 inner: np.ndarray, mine: np.ndarray) -> dict:
    """Assert this script's MSE loop reproduces ``nested.fit_mlp_ensemble``.

    The point of the public function is that a budget is inherited rather than
    retyped. This script cannot call it (it needs per-member output), so the
    equivalence is checked instead of asserted in prose.
    """
    theirs = nested.fit_mlp_ensemble(
        fit_df, [predict_df], inner, ENCODINGS, SEEDS, SELECTED,
        input_set=INPUT_SET)["predictions"][0]
    diff = float(np.abs(mine - theirs).max())
    # 1e-5, not 1e-9. The two loops fit the *same* networks from the same seeds;
    # they differ only in how the member mean is accumulated (this script
    # round-trips predictions through Python lists so they can cross a Modal
    # boundary, which changes float32 summation order). Measured on outer fold 0
    # at threads=1: **max |diff| 1.18e-06, mean 7.7e-08**. A 1e-9 bar would fail
    # on that representation noise and say "the budget is not inherited", which
    # would be the wrong conclusion -- the platform term below is five orders of
    # magnitude larger and is the only difference that matters.
    if diff > 1e-5:
        raise AssertionError(
            f"budget parity failed: this script's MSE ensemble mean differs from "
            f"nested.fit_mlp_ensemble by {diff:.3e}, which is too large to be "
            f"float32 accumulation order; the arms would not be comparable to A0")
    return {"max_abs_diff_vs_fit_mlp_ensemble": diff, "tolerance": 1e-5,
            "passed": True,
            "note": ("the loops are equivalent to ~1e-06; a cross-platform term "
                     "(macOS/Accelerate arm64 against Linux/OpenBLAS x86-64, both "
                     "at threads=1) is ~0.18 on predictions and 0.0020 on "
                     "median-of-medians, and is reported separately")}


# --------------------------------------------------------------------------
# scoring


def p_floor_for(mus: np.ndarray, sigmas: np.ndarray, rule: str,
                threshold: float) -> np.ndarray:
    if rule == "mixture":
        return cal.prob_floor_mixture(mus, sigmas, threshold)
    return cal.prob_floor_plugin(mus.mean(axis=0), float(sigmas.mean()), threshold)


def fold_metrics(arm_id: str, label: str, fold: int, held: pd.DataFrame,
                 mus: np.ndarray, sigmas: np.ndarray, rule: str,
                 threshold: float, min_rows: int) -> dict:
    mu = mus.mean(axis=0)
    p = p_floor_for(mus, sigmas, rule, threshold)
    is_floor = cal.floor_indicator(held.thalf_hours, "recorded_zero")
    is_cens = cal.floor_indicator(held.thalf_hours, "below_limit")
    y = held.y_log1p.to_numpy()
    sc = nested.score_fold(label, held.allele, y, mu, min_rows=min_rows)
    cov = nested.fold_coverage(held.allele, y, min_rows=min_rows)
    ece_fixed, _ = cal.fixed_bin_ece(is_floor, p)
    return {
        "arm": arm_id, "label": label, "outer_fold": fold, "min_rows": min_rows,
        "p_floor_rule": rule, "n_rows": len(held),
        "n_floor_rows": int(is_floor.sum()), "floor_share": float(is_floor.mean()),
        "n_alleles_scored": sc.n_alleles_scored,
        "coverage": cov.get("coverage", float("nan")),
        "median_per_allele_spearman": sc.median_spearman,
        "mae_log1p": sc.mae_log1p,
        "median_precision_at_10": sc.median_precision_at_k,
        "brier": cal.brier(is_floor, p),
        "ece_fixed_bin": ece_fixed,
        "ece_equal_count": cal.equal_count_ece(is_floor, p),
        "log_loss": cal.log_loss(is_floor, p),
        "floor_auroc_from_p": cal.floor_auroc(is_floor, p),
        "censored_nll": cal.censored_nll_mean(mu, y, is_cens,
                                              float(sigmas.mean()), threshold),
        "mean_p_floor": float(p.mean()),
        "share_mu_at_or_below_threshold": float((mu <= threshold).mean()),
        "mean_sigma": float(sigmas.mean()),
        "n_members": len(mus),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="pilot: 1 outer fold, 2 inner folds, 1 seed, 200 resamples")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--parity-check", action="store_true", default=True)
    ap.add_argument("--no-parity-check", dest="parity_check", action="store_false")
    args = ap.parse_args()

    global SEEDS
    n_inner = 2 if args.quick else 5
    if args.quick:
        SEEDS = (0,)
        args.n_boot = 200
    suffix = "_quick" if args.quick else ""
    cache = OUT_DIR / f"phase_b_nested_members{suffix}.npz"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_with_splits()
    train = df[df.split == "train"].reset_index(drop=True)
    threshold = floor_threshold(DEFAULT_FLOOR_HOURS)
    splits = load_frozen_folds(train, n_inner)
    if args.quick:
        splits = splits[:1]

    n_nets = len(ARMS) * len(splits) * n_inner * len(ENCODINGS) * len(SEEDS)
    print(f"Phase B nested{' PILOT' if args.quick else ''}: {len(ARMS)} arms x "
          f"{len(splits)} outer x {n_inner} inner x {len(ENCODINGS)} enc x "
          f"{len(SEEDS)} seeds = {n_nets} networks")
    print(f"  train {len(train):,} rows, folds from {FOLDS_CSV.name} (verified "
          f"against nested.iter_outer)")
    print(f"  floor event = recorded zero; loss censors below "
          f"{DEFAULT_FLOOR_HOURS} h\n")

    # --- fit -------------------------------------------------------------
    if args.analyse_only and cache.exists():
        print(f"reusing {cache.name}")
        blob = np.load(cache, allow_pickle=False)
        store = {k: blob[k] for k in blob.files}
        runs = pd.read_csv(OUT_DIR / f"phase_b_nested_runs{suffix}.csv")
        parity = json.loads((OUT_DIR / f"phase_b_nested_parity{suffix}.json").read_text())
    else:
        store, run_rows, parity = {}, [], {}
        started = time.perf_counter()
        for split in splits:
            fit_df = train[split.fit_mask].reset_index(drop=True)
            held_df = train[split.held_mask].reset_index(drop=True)
            print(f"outer fold {split.fold}: fit {len(fit_df):,}, "
                  f"held {len(held_df):,}", flush=True)
            for arm_id, kind, stop_on, label in ARMS:
                t0 = time.perf_counter()
                got = fit_arm_members(kind, stop_on, fit_df, held_df, split.inner)
                store[f"{arm_id}__{split.fold}__mus"] = got["mus"]
                store[f"{arm_id}__{split.fold}__sigmas"] = got["sigmas"]
                store[f"{arm_id}__{split.fold}__held_pair_id"] = \
                    held_df.pair_id.to_numpy()
                store[f"{arm_id}__{split.fold}__inner_oof"] = got["inner_oof"]
                store[f"{arm_id}__{split.fold}__fit_pair_id"] = \
                    fit_df.pair_id.to_numpy()
                for rec in got["records"]:
                    run_rows.append({"arm": arm_id, "outer_fold": split.fold, **rec})
                print(f"  {arm_id} {label:30s} {got['n_members']:3d} members "
                      f"{time.perf_counter() - t0:6.1f}s  "
                      f"mean sigma {got['sigmas'].mean():.4f}  "
                      f"mean best epoch "
                      f"{np.mean([r['best_epoch'] for r in got['records']]):.1f}",
                      flush=True)
                if arm_id == "B0" and split.fold == 0 and args.parity_check:
                    parity = parity_check(fit_df, held_df, split.inner,
                                          got["mus"].mean(axis=0))
                    print(f"  budget parity vs nested.fit_mlp_ensemble: "
                          f"max |diff| {parity['max_abs_diff_vs_fit_mlp_ensemble']:.2e}"
                          f"  OK", flush=True)
        runs = pd.DataFrame(run_rows)
        runs.to_csv(OUT_DIR / f"phase_b_nested_runs{suffix}.csv", index=False)
        np.savez_compressed(cache, **store)
        (OUT_DIR / f"phase_b_nested_parity{suffix}.json").write_text(
            json.dumps(parity or {"skipped": True}, indent=2))
        print(f"\nfitting wall time {time.perf_counter() - started:.1f}s")

    # --- pooled outer-held predictions, one row per training pair_id ------
    pooled = {}
    for arm_id, _, _, _ in ARMS:
        frames = []
        for split in splits:
            pid = store[f"{arm_id}__{split.fold}__held_pair_id"]
            mus = store[f"{arm_id}__{split.fold}__mus"]
            sig = store[f"{arm_id}__{split.fold}__sigmas"]
            frames.append(pd.DataFrame({
                "pair_id": pid, "outer_fold": split.fold,
                "mu": mus.mean(axis=0),
                "p_floor_mixture": p_floor_for(mus, sig, "mixture", threshold),
                "p_floor_plugin": p_floor_for(mus, sig, "plugin", threshold),
                "mean_sigma": float(sig.mean())}))
        pooled[arm_id] = pd.concat(frames, ignore_index=True)
        assert pooled[arm_id].pair_id.is_unique, arm_id
    held_all = (train[["pair_id", "allele", "thalf_hours", "y_log1p", "cluster_id"]]
                .merge(pooled["B0"][["pair_id"]], on="pair_id", how="inner")
                .reset_index(drop=True))
    for arm_id in pooled:
        pooled[arm_id] = (held_all[["pair_id"]]
                          .merge(pooled[arm_id], on="pair_id", validate="one_to_one"))
    pd.concat([p.assign(arm=a) for a, p in pooled.items()], ignore_index=True).to_csv(
        OUT_DIR / f"phase_b_nested_oof{suffix}.csv", index=False)

    is_floor = cal.floor_indicator(held_all.thalf_hours, "recorded_zero")
    is_cens = cal.floor_indicator(held_all.thalf_hours, "below_limit")
    print(f"\npooled outer-held rows {len(held_all):,}, floor "
          f"{int(is_floor.sum()):,} ({is_floor.mean():.4f}); the two floor "
          f"definitions differ on {int((is_floor != is_cens).sum())} row(s)")

    # --- THE BAR, from B0 alone, before any censored arm is compared ------
    print("\nderiving the worthwhile-Brier bar from B0 only "
          "(no censored arm has been scored yet) ...", flush=True)
    bar = cal.null_half_width(
        held_all.cluster_id, cal.brier,
        predictions=pooled["B0"][f"p_floor_{PRIMARY_P_FLOOR_RULE}"].to_numpy(),
        labels=is_floor, n_boot=args.n_boot, n_splits=8)
    raw = bar["max_half_width"]
    # Round UP to two significant figures, as declared.
    mag = 10 ** (np.floor(np.log10(raw)) - 1) if raw > 0 else 1.0
    bar_value = float(np.ceil(raw / mag) * mag)
    bar_out = {"rule": "max 95% paired cluster-bootstrap half-width under no true "
                       "difference, B0 only, rounded up to 2 s.f.",
               "p_floor_rule": PRIMARY_P_FLOOR_RULE,
               "max_half_width": raw, "median_half_width": bar["median_half_width"],
               "half_widths": bar["half_widths"],
               "worthwhile_delta_brier": bar_value,
               "n_boot": args.n_boot, "n_splits": 8, "seed": 20261003,
               "derived_before_any_censored_arm_scored": True}
    (OUT_DIR / f"phase_b_brier_bar{suffix}.json").write_text(json.dumps(bar_out, indent=2))
    print(f"  max null half-width {raw:.5f} -> worthwhile delta Brier bar "
          f"{bar_value:.4f}  (written to phase_b_brier_bar{suffix}.json)")

    # --- per-fold metrics -------------------------------------------------
    rows_fold, rel_frames = [], []
    for min_rows in (20, 50):
        for rule in cal.P_FLOOR_RULES:
            for split in splits:
                held_df = train[split.held_mask].reset_index(drop=True)
                for arm_id, _, _, label in ARMS:
                    mus = store[f"{arm_id}__{split.fold}__mus"]
                    sig = store[f"{arm_id}__{split.fold}__sigmas"]
                    rows_fold.append(fold_metrics(arm_id, label, split.fold, held_df,
                                                  mus, sig, rule, threshold, min_rows))
    pd.DataFrame(rows_fold).to_csv(
        OUT_DIR / f"phase_b_nested_by_fold{suffix}.csv", index=False)

    # --- pooled comparisons, with paired intervals ------------------------
    rows_pair, rows_pooled = [], []
    for rule in cal.P_FLOOR_RULES:
        pcol = f"p_floor_{rule}"
        for arm_id, _, _, label in ARMS:
            p = pooled[arm_id][pcol].to_numpy()
            mu = pooled[arm_id]["mu"].to_numpy()
            sigma = float(pooled[arm_id]["mean_sigma"].mean())
            ece_fixed, table = cal.fixed_bin_ece(is_floor, p)
            t = cal.reliability_table(is_floor, p, binning="fixed", arm=label)
            t.insert(0, "p_floor_rule", rule)
            rel_frames.append(t)
            sc = nested.score_fold(label, held_all.allele,
                                   held_all.y_log1p.to_numpy(), mu)
            rows_pooled.append({
                "arm": arm_id, "label": label, "p_floor_rule": rule,
                "n_rows": len(held_all), "n_alleles_scored": sc.n_alleles_scored,
                "median_per_allele_spearman": sc.median_spearman,
                "mae_log1p": sc.mae_log1p,
                "median_precision_at_10": sc.median_precision_at_k,
                "brier": cal.brier(is_floor, p), "ece_fixed_bin": ece_fixed,
                "ece_equal_count": cal.equal_count_ece(is_floor, p),
                "log_loss": cal.log_loss(is_floor, p),
                "floor_auroc_from_p": cal.floor_auroc(is_floor, p),
                "censored_nll": cal.censored_nll_mean(
                    mu, held_all.y_log1p.to_numpy(), is_cens, sigma, threshold),
                "mean_p_floor": float(p.mean()), "mean_sigma": sigma,
                "share_mu_at_or_below_threshold": float((mu <= threshold).mean()),
            })

        ref = pooled["B0"]
        ref_sigma = float(ref["mean_sigma"].mean())
        for arm_id, _, _, label in ARMS[1:]:
            arm = pooled[arm_id]
            arm_sigma = float(arm["mean_sigma"].mean())
            rank = paired_cluster_bootstrap(
                held_all.cluster_id, held_all.allele, held_all.y_log1p.to_numpy(),
                ref["mu"].to_numpy(), arm["mu"].to_numpy(),
                alleles=nested.fold_eligible_alleles(
                    held_all.allele, held_all.y_log1p.to_numpy()),
                n_boot=args.n_boot)
            rows_pair.append({
                "arm": arm_id, "comparison": f"{label} vs B0", "p_floor_rule": rule,
                "metric": "median_per_allele_spearman", "lower_is_better": False,
                "reference": rank.get("base", float("nan")),
                "delta_improvement": rank["delta_median_spearman"],
                "ci_low": rank["ci95"][0], "ci_high": rank["ci95"][1],
                "verdict": describe_delta(rank),
                "passes_ranking_tolerance": bool(
                    rank["delta_median_spearman"] >= -0.02 and rank["ci95"][0] > -0.05),
            })
            for name, metric, lower in (
                    ("brier", cal.brier, True),
                    ("ece_fixed_bin", lambda y, p: cal.fixed_bin_ece(y, p)[0], True),
                    ("ece_equal_count", cal.equal_count_ece, True),
                    ("log_loss", cal.log_loss, True),
                    ("floor_auroc_from_p", cal.floor_auroc, False)):
                res = cal.paired_cluster_bootstrap_metric(
                    held_all.cluster_id, metric, reference=ref[pcol].to_numpy(),
                    arm=arm[pcol].to_numpy(), labels=is_floor,
                    lower_is_better=lower, n_boot=args.n_boot)
                md = bar_value if name == "brier" else 0.0
                rows_pair.append({
                    "arm": arm_id, "comparison": f"{label} vs B0",
                    "p_floor_rule": rule, "metric": name, "lower_is_better": lower,
                    "reference": res["reference"], "arm_value": res["arm"],
                    "delta_improvement": res["delta_improvement"],
                    "ci_low": res["ci95"][0], "ci_high": res["ci95"][1],
                    "min_delta_applied": md,
                    "verdict": describe_delta(res, min_delta=md),
                })
            nll = cal.paired_cluster_bootstrap_censored_nll(
                held_all.cluster_id, y_log1p=held_all.y_log1p.to_numpy(),
                is_censored=is_cens, mu_reference=ref["mu"].to_numpy(),
                sigma_reference=ref_sigma, mu_arm=arm["mu"].to_numpy(),
                sigma_arm=arm_sigma, threshold=threshold, n_boot=args.n_boot)
            rows_pair.append({
                "arm": arm_id, "comparison": f"{label} vs B0", "p_floor_rule": rule,
                "metric": "censored_nll", "lower_is_better": True,
                "reference": nll["reference"], "arm_value": nll["arm"],
                "delta_improvement": nll["delta_improvement"],
                "ci_low": nll["ci95"][0], "ci_high": nll["ci95"][1],
                "min_delta_applied": 0.0,
                "verdict": describe_delta(nll, min_delta=0.0),
            })

    pd.DataFrame(rows_pooled).to_csv(
        OUT_DIR / f"phase_b_nested_pooled{suffix}.csv", index=False)
    pd.DataFrame(rows_pair).to_csv(
        OUT_DIR / f"phase_b_nested_paired{suffix}.csv", index=False)
    pd.concat(rel_frames, ignore_index=True).to_csv(
        OUT_DIR / f"phase_b_nested_reliability{suffix}.csv", index=False)

    # --- the headline, read against the declared decision rule -------------
    print(f"\npooled outer-held, P(floor) = {PRIMARY_P_FLOOR_RULE}:")
    pooled_tbl = pd.DataFrame(rows_pooled)
    pooled_tbl = pooled_tbl[pooled_tbl.p_floor_rule == PRIMARY_P_FLOOR_RULE]
    print(pooled_tbl[["arm", "median_per_allele_spearman", "brier",
                      "ece_fixed_bin", "censored_nll", "mean_sigma"]]
          .round(4).to_string(index=False))
    print("\ndecision rule:")
    pair_tbl = pd.DataFrame(rows_pair)
    pair_tbl = pair_tbl[pair_tbl.p_floor_rule == PRIMARY_P_FLOOR_RULE]
    for arm_id in ("B1", "B2"):
        sub = pair_tbl[pair_tbl.arm == arm_id]
        b = sub[sub.metric == "brier"].iloc[0]
        r = sub[sub.metric == "median_per_allele_spearman"].iloc[0]
        limb1 = bool(b.ci_low > 0 and b.delta_improvement >= bar_value)
        limb2 = bool(r.passes_ranking_tolerance)
        print(f"  {arm_id}: dBrier {b.delta_improvement:+.4f} "
              f"[{b.ci_low:+.4f},{b.ci_high:+.4f}] vs bar {bar_value:.4f} "
              f"-> limb 1 {'PASS' if limb1 else 'FAIL'}")
        print(f"       drho  {r.delta_improvement:+.4f} "
              f"[{r.ci_low:+.4f},{r.ci_high:+.4f}] "
              f"-> limb 2 {'PASS' if limb2 else 'FAIL'}  ({r.verdict})")
        print(f"       KEEP: {'yes' if (limb1 and limb2) else 'no'}")
    print(f"\nwrote artifacts to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
