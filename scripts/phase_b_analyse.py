"""Phase B analysis: score the three censored arms and the calibration baselines.

    .venv/bin/python scripts/phase_b_analyse.py              # the batch
    .venv/bin/python scripts/phase_b_analyse.py --suffix _smoke

Reads the members that ``modal_app/phase_b_calibration.py`` (or the local
``scripts/phase_b_censored_nested.py``) wrote and does the arithmetic: no
fitting, so this is cheap and rerunnable. Everything it decides was fixed in
``reports/further_research/phase_b_declaration.md`` before any arm was fitted.

**The one thing in here that is not in the declaration, and why it is here.**
The MSE reference has no fitted scale, so its floor probability needs one bolted
on, and no training-only estimator of that scale is unbiased. Measured against
A0's own output (RMSE on measured rows): the out-of-sample target is 0.6944,
``inner_oof_6`` gives 0.7086 and ``inner_oof_30`` gives 0.7176. Both are
**pessimistic** -- larger than the quantity they estimate -- because an inner
out-of-fold prediction comes from an ensemble fitted on 10,094 or 12,618 rows
against the outer-held ensemble's 15,773, and that fit-size penalty outweighs the
early-stopping optimism pulling the other way. They bound the truth from one
side; they do not bracket it.

That bias is not neutral. A scale that is too large makes the reference
**under-confident**, which worsens its Brier and therefore **flatters the
censored arms** -- the direction this phase exists to guard against. So the
reference is scored at a ladder of scales, including an **oracle** scale chosen
to minimise its own Brier on the assessment rows. The oracle is deliberately not
a legitimate model: it uses the held-out label, and it is reported only as an
upper bound on how good the reference's Gaussian floor probability could possibly
be. A censored arm that beats the oracle has a result robust to the scale
question; one that beats only the training-only scale has a comparator artifact,
and is reported as that.

This is also why **Platt** is the declared primary baseline rather than the
Gaussian scale: it fits the ``mu -> P(floor)`` map directly on inner out-of-fold
rows and never needs a sigma, so it is immune to all of the above.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pepstab import calibration as cal  # noqa: E402
from pepstab import nested  # noqa: E402
from pepstab.censored import DEFAULT_FLOOR_HOURS, floor_threshold  # noqa: E402
from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import describe_delta, paired_cluster_bootstrap  # noqa: E402
from pepstab.phase_b import ARMS  # noqa: E402

OUT_DIR = REPO_ROOT / "reports" / "further_research"
PRIMARY_RULE = "mixture"
#: Declared ranking tolerance, phase_b_declaration.md §7 limb 2.
RANK_POINT_TOLERANCE = -0.02
RANK_CI_TOLERANCE = -0.05


def oracle_sigma(mu: np.ndarray, is_floor: np.ndarray, threshold: float,
                 grid=None) -> tuple[float, float]:
    """The sigma that minimises the reference's own Brier. **Not a legitimate model.**

    It reads the assessment label, so it cannot be used to predict anything. It is
    here as an *upper bound* on the reference: whatever a Gaussian floor
    probability on this ``mu`` could achieve with a perfectly chosen scale.
    """
    grid = np.linspace(0.05, 2.0, 196) if grid is None else grid
    briers = [cal.brier(is_floor, cal.prob_floor_plugin(mu, s, threshold))
              for s in grid]
    best = int(np.argmin(briers))
    return float(grid[best]), float(briers[best])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    sx = args.suffix

    members_path = OUT_DIR / f"phase_b_nested_members{sx}.npz"
    if not members_path.exists():
        raise SystemExit(f"{members_path} not found -- run the fitting layer first")
    store = np.load(members_path, allow_pickle=False)
    runs = pd.read_csv(OUT_DIR / f"phase_b_nested_runs{sx}.csv")
    threshold = floor_threshold(DEFAULT_FLOOR_HOURS)

    df = load_with_splits()
    train = df[df.split == "train"].set_index("pair_id", drop=False)
    folds = sorted({int(k.split("__")[1]) for k in store.files if k.endswith("__mus")})
    arms = [(a, lbl) for a, _k, _s, lbl in ARMS
            if f"{a}__{folds[0]}__mus" in store.files]
    print(f"Phase B analysis{sx}: arms {[a for a, _ in arms]}, outer folds {folds}")

    # --- assemble pooled outer-held predictions --------------------------
    pooled, per_fold_pred = {}, {}
    for arm_id, _lbl in arms:
        frames = []
        for k in folds:
            pid = store[f"{arm_id}__{k}__held_pair_id"]
            mus = store[f"{arm_id}__{k}__mus"]
            sig = store[f"{arm_id}__{k}__sigmas"]
            sig_dev = store[f"{arm_id}__{k}__sigmas_from_dev_mse"]
            frames.append(pd.DataFrame({
                "pair_id": pid, "outer_fold": k, "mu": mus.mean(axis=0),
                "p_mixture": cal.prob_floor_mixture(mus, sig, threshold),
                "p_plugin": cal.prob_floor_plugin(mus.mean(axis=0),
                                                  float(sig.mean()), threshold),
                "sigma_member_mean": float(sig.mean()),
                "sigma_dev_mse_mean": float(sig_dev.mean())}))
            per_fold_pred[(arm_id, k)] = frames[-1]
        pooled[arm_id] = pd.concat(frames, ignore_index=True)
        assert pooled[arm_id].pair_id.is_unique, arm_id

    order = pooled[arms[0][0]][["pair_id", "outer_fold"]]
    train_cols = train[["pair_id", "allele", "thalf_hours", "y_log1p",
                        "cluster_id"]].reset_index(drop=True)
    meta = order.merge(train_cols, on="pair_id", validate="one_to_one")
    for arm_id in pooled:
        pooled[arm_id] = order.merge(pooled[arm_id], on=["pair_id", "outer_fold"],
                                     validate="one_to_one")
    is_floor = cal.floor_indicator(meta.thalf_hours, "recorded_zero")
    is_cens = cal.floor_indicator(meta.thalf_hours, "below_limit")
    y = meta.y_log1p.to_numpy()
    print(f"  pooled outer-held {len(meta):,} rows, floor {int(is_floor.sum()):,} "
          f"({is_floor.mean():.4f}); floor definitions differ on "
          f"{int((is_floor != is_cens).sum())} row(s)")

    # --- the reference's scale ladder ------------------------------------
    ref_id = arms[0][0]
    ref_mu = pooled[ref_id]["mu"].to_numpy()
    sigma_member = float(pooled[ref_id]["sigma_member_mean"].mean())
    sigma_dev = float(pooled[ref_id]["sigma_dev_mse_mean"].mean())
    sigma_oracle, brier_oracle = oracle_sigma(ref_mu, is_floor, threshold)
    ladder = {"member_stopping_fold": sigma_member, "dev_mse": sigma_dev,
              "oracle_best_brier": sigma_oracle}
    oof_path = OUT_DIR / "phase_a_seq_oof_cluster.csv"
    if oof_path.exists():
        oof = pd.read_csv(oof_path)
        fl = pd.Series(cal.floor_indicator(train.thalf_hours, "recorded_zero"),
                       index=train.pair_id)
        yy = train.y_log1p
        for role in ("inner_oof_6", "inner_oof_30"):
            s = oof[oof.role == role]
            r = yy.reindex(s.pair_id).to_numpy() - s.y_pred.to_numpy()
            keep = ~fl.reindex(s.pair_id).to_numpy()
            ladder[f"a0_{role}"] = float(np.sqrt(np.mean(r[keep] ** 2)))
        s = oof[oof.role == "outer_held"]
        r = yy.reindex(s.pair_id).to_numpy() - s.y_pred.to_numpy()
        keep = ~fl.reindex(s.pair_id).to_numpy()
        # Reported for context only: this is the quantity the others estimate, and
        # using it to *choose* a scale would be the leak the ladder exists to expose.
        ladder["a0_outer_held_NOT_USABLE"] = float(np.sqrt(np.mean(r[keep] ** 2)))

    print("\n  reference scale ladder (Brier of B0 at each):")
    ladder_rows = []
    for name, s in ladder.items():
        b = cal.brier(is_floor, cal.prob_floor_plugin(ref_mu, s, threshold))
        e, _ = cal.fixed_bin_ece(is_floor, cal.prob_floor_plugin(ref_mu, s, threshold))
        ladder_rows.append({"scale": name, "sigma": s, "b0_brier": b,
                            "b0_ece_fixed_bin": e,
                            "legitimate": not name.endswith("NOT_USABLE")
                            and name != "oracle_best_brier"})
        print(f"    {name:28s} sigma {s:.4f}  B0 Brier {b:.4f}  ECE {e:.4f}"
              f"{'   <- not a legitimate model' if not ladder_rows[-1]['legitimate'] else ''}")
    pd.DataFrame(ladder_rows).to_csv(
        OUT_DIR / f"phase_b_reference_scale_ladder{sx}.csv", index=False)

    # --- Platt / isotonic baselines, fitted on inner out-of-fold only -----
    # **One calibrator per outer fold, fitted only inside that fold's fitting
    # partition.** Pooling the inner out-of-fold rows across all five folds and
    # fitting one map would use every row's own label to build the map that then
    # predicts it -- a row is outer-held in exactly one fold and inner-OOF in the
    # other four, so a pooled fit sees the label of every row it is later applied
    # to. That is precisely the leak the declaration forbids ("an outer held-out
    # label must not choose ... calibration maps"), and it is invisible in the
    # output: the pooled fit produces a *better*-looking baseline, which would
    # have understated the censored arms rather than flattered them, but it would
    # have been the wrong number either way.
    baselines = {}
    if oof_path.exists():
        floor_by_pid = pd.Series(
            cal.floor_indicator(train.thalf_hours, "recorded_zero"),
            index=train.pair_id)
        oof30 = oof[oof.role == "inner_oof_30"]
        platt_p = np.full(len(meta), np.nan)
        iso_p = np.full(len(meta), np.nan)
        slopes, fit_sizes = [], []
        for k in folds:
            held_sel = (meta.outer_fold == k).to_numpy()
            # Only rows in THIS outer fold's fitting partition, i.e. the rows this
            # fold's models were allowed to see.
            fit_rows = oof30[oof30.outer_fold == k]
            fit_mu = fit_rows.y_pred.to_numpy()
            fit_floor = floor_by_pid.reindex(fit_rows.pair_id).to_numpy()
            # A **partition identity**, not a row count: the fit rows and the
            # scored rows must be disjoint *and* together reconstitute exactly
            # this outer fold's training split. Stronger than either check
            # alone -- disjointness alone misses a dropped row, and a literal
            # row count is both brittle (the fitting partitions are 15,772 on
            # fold 0 and 15,773 on the rest, because whole clusters are
            # water-filled) and blind to which rows moved. A pooled fit, the
            # error this guards against, makes the fit set all 19,716 rows and
            # breaks the identity immediately.
            fit_ids, scored_ids = set(fit_rows.pair_id), set(meta.pair_id[held_sel])
            assert not fit_ids & scored_ids, (
                f"outer fold {k}: {len(fit_ids & scored_ids)} calibrator fit rows "
                f"also appear in the rows it scores")
            assert fit_ids | scored_ids == set(train_cols.pair_id), (
                f"outer fold {k}: fit rows + scored rows do not reconstitute the "
                f"training split ({len(fit_ids)} + {len(scored_ids)} = "
                f"{len(fit_ids | scored_ids)} against {len(train_cols)})")
            platt = cal.PlattFloorCalibrator().fit(fit_mu, fit_floor)
            iso = cal.IsotonicFloorCalibrator().fit(fit_mu, fit_floor)
            platt_p[held_sel] = platt.predict_proba(ref_mu[held_sel])
            iso_p[held_sel] = iso.predict_proba(ref_mu[held_sel])
            slopes.append(platt.coef_)
            fit_sizes.append(len(fit_mu))
        assert np.isfinite(platt_p).all() and np.isfinite(iso_p).all()
        baselines["C1_platt"] = platt_p
        baselines["C2_isotonic"] = iso_p
        print(f"\n  C1 Platt   (per outer fold, {min(fit_sizes):,}-{max(fit_sizes):,} "
              f"inner-OOF fit rows each): slopes "
              f"{', '.join(f'{c:+.3f}' for c in slopes)}, Brier "
              f"{cal.brier(is_floor, platt_p):.4f}")
        print(f"  C2 isotonic: Brier {cal.brier(is_floor, iso_p):.4f}, "
              f"{len(np.unique(iso_p))} distinct values from "
              f"{len(np.unique(ref_mu))} mu -- ranking is NOT scored on this")

    # --- pooled table -----------------------------------------------------
    rows_pooled, rel_frames = [], []
    comparators = {}
    for arm_id, lbl in arms:
        for rule, col in (("mixture", "p_mixture"), ("plugin", "p_plugin")):
            p = pooled[arm_id][col].to_numpy()
            mu = pooled[arm_id]["mu"].to_numpy()
            sg = float(pooled[arm_id]["sigma_member_mean"].mean())
            sc = nested.score_fold(lbl, meta.allele, y, mu)
            ece, _ = cal.fixed_bin_ece(is_floor, p)
            rows_pooled.append({
                "arm": arm_id, "label": lbl, "p_floor_rule": rule,
                "n_rows": len(meta), "n_alleles_scored": sc.n_alleles_scored,
                "median_per_allele_spearman": sc.median_spearman,
                "mae_log1p": sc.mae_log1p,
                "median_precision_at_10": sc.median_precision_at_k,
                "brier": cal.brier(is_floor, p), "ece_fixed_bin": ece,
                "ece_equal_count": cal.equal_count_ece(is_floor, p),
                "log_loss": cal.log_loss(is_floor, p),
                "floor_auroc_from_p": cal.floor_auroc(is_floor, p),
                "censored_nll": cal.censored_nll_mean(mu, y, is_cens, sg, threshold),
                "mean_p_floor": float(p.mean()), "mean_sigma": sg,
                "share_mu_at_or_below_threshold": float((mu <= threshold).mean())})
            t = cal.reliability_table(is_floor, p, binning="fixed", arm=lbl)
            t.insert(0, "p_floor_rule", rule)
            rel_frames.append(t)
        comparators[arm_id] = pooled[arm_id][f"p_{PRIMARY_RULE}"].to_numpy()

    # the reference's alternative floor probabilities, as extra comparators
    for name, s in ladder.items():
        if name.endswith("NOT_USABLE"):
            continue
        # The oracle carries its illegitimacy in its name as well as in the
        # ladder's ``legitimate`` column, so a row lifted out of a table cannot be
        # mistaken for a candidate model.
        label = (f"{ref_id}@{name}_NOT_A_MODEL" if name == "oracle_best_brier"
                 else f"{ref_id}@{name}")
        comparators[label] = cal.prob_floor_plugin(ref_mu, s, threshold)
    comparators.update(baselines)

    pd.DataFrame(rows_pooled).to_csv(
        OUT_DIR / f"phase_b_pooled{sx}.csv", index=False)
    pd.concat(rel_frames, ignore_index=True).to_csv(
        OUT_DIR / f"phase_b_reliability{sx}.csv", index=False)

    # --- paired comparisons, pooled --------------------------------------
    rows_pair = []
    alleles = nested.fold_eligible_alleles(meta.allele, y)
    for arm_id, lbl in arms[1:]:
        rank = paired_cluster_bootstrap(meta.cluster_id, meta.allele, y, ref_mu,
                                        pooled[arm_id]["mu"].to_numpy(),
                                        alleles=alleles, n_boot=args.n_boot)
        rows_pair.append({
            "arm": arm_id, "comparator": ref_id, "scope": "pooled",
            "metric": "median_per_allele_spearman", "lower_is_better": False,
            "delta_improvement": rank["delta_median_spearman"],
            "ci_low": rank["ci95"][0], "ci_high": rank["ci95"][1],
            "verdict": describe_delta(rank),
            "passes_ranking_tolerance": bool(
                rank["delta_median_spearman"] >= RANK_POINT_TOLERANCE
                and rank["ci95"][0] > RANK_CI_TOLERANCE)})
        # Brier against every comparator, including the oracle upper bound.
        for comp_name, comp_p in comparators.items():
            if comp_name == arm_id:
                continue
            for metric_name, metric, lower in (
                    ("brier", cal.brier, True),
                    ("ece_fixed_bin", lambda yy, pp: cal.fixed_bin_ece(yy, pp)[0], True),
                    ("log_loss", cal.log_loss, True),
                    ("floor_auroc_from_p", cal.floor_auroc, False)):
                res = cal.paired_cluster_bootstrap_metric(
                    meta.cluster_id, metric, reference=comp_p,
                    arm=comparators[arm_id], labels=is_floor,
                    lower_is_better=lower, n_boot=args.n_boot)
                rows_pair.append({
                    "arm": arm_id, "comparator": comp_name, "scope": "pooled",
                    "metric": metric_name, "lower_is_better": lower,
                    "reference_value": res["reference"], "arm_value": res["arm"],
                    "delta_improvement": res["delta_improvement"],
                    "ci_low": res["ci95"][0], "ci_high": res["ci95"][1],
                    "verdict": describe_delta(res, min_delta=0.0)})

    # --- paired comparisons, per fold, for sign consistency ---------------
    for arm_id, _lbl in arms[1:]:
        for k in folds:
            sel = (meta.outer_fold == k).to_numpy()
            rank = paired_cluster_bootstrap(
                meta.cluster_id[sel], meta.allele[sel], y[sel], ref_mu[sel],
                pooled[arm_id]["mu"].to_numpy()[sel],
                alleles=nested.fold_eligible_alleles(meta.allele[sel], y[sel]),
                n_boot=max(200, args.n_boot // 4))
            br = cal.paired_cluster_bootstrap_metric(
                meta.cluster_id[sel], cal.brier,
                reference=comparators[ref_id][sel], arm=comparators[arm_id][sel],
                labels=is_floor[sel], n_boot=max(200, args.n_boot // 4))
            rows_pair.append({
                "arm": arm_id, "comparator": ref_id, "scope": f"fold_{k}",
                "metric": "median_per_allele_spearman", "lower_is_better": False,
                "delta_improvement": rank["delta_median_spearman"],
                "ci_low": rank["ci95"][0], "ci_high": rank["ci95"][1],
                "verdict": describe_delta(rank)})
            rows_pair.append({
                "arm": arm_id, "comparator": ref_id, "scope": f"fold_{k}",
                "metric": "brier", "lower_is_better": True,
                "reference_value": br["reference"], "arm_value": br["arm"],
                "delta_improvement": br["delta_improvement"],
                "ci_low": br["ci95"][0], "ci_high": br["ci95"][1],
                "verdict": describe_delta(br, min_delta=0.0)})

    paired = pd.DataFrame(rows_pair)
    paired.to_csv(OUT_DIR / f"phase_b_paired{sx}.csv", index=False)

    # --- sign consistency across folds ------------------------------------
    cons = []
    for arm_id, _lbl in arms[1:]:
        for metric in ("median_per_allele_spearman", "brier"):
            s = paired[(paired.arm == arm_id) & (paired.metric == metric)
                       & paired.scope.str.startswith("fold_")]
            signs = np.sign(s.delta_improvement.to_numpy())
            cons.append({"arm": arm_id, "metric": metric,
                         "n_folds": len(s),
                         "n_positive": int((signs > 0).sum()),
                         "n_negative": int((signs < 0).sum()),
                         "all_same_sign": bool(len(set(signs.tolist())) == 1),
                         "fold_deltas": ", ".join(f"{v:+.4f}" for v in
                                                  s.delta_improvement)})
    pd.DataFrame(cons).to_csv(
        OUT_DIR / f"phase_b_fold_consistency{sx}.csv", index=False)

    # --- headline ---------------------------------------------------------
    bar_path = OUT_DIR / f"phase_b_brier_bar{sx}.json"
    bar = json.loads(bar_path.read_text())["worthwhile_delta_brier"] \
        if bar_path.exists() else None
    print(f"\npooled, P(floor) = {PRIMARY_RULE}:")
    pt = pd.DataFrame(rows_pooled)
    print(pt[pt.p_floor_rule == PRIMARY_RULE][
        ["arm", "median_per_allele_spearman", "brier", "ece_fixed_bin",
         "censored_nll", "mean_sigma"]].round(4).to_string(index=False))
    print("\nfold sign consistency:")
    print(pd.DataFrame(cons).to_string(index=False))
    print(f"\ndecision rule (bar = {bar}):")
    for arm_id, _lbl in arms[1:]:
        r = paired[(paired.arm == arm_id) & (paired.scope == "pooled")
                   & (paired.metric == "median_per_allele_spearman")].iloc[0]
        print(f"  {arm_id} ranking: {r.delta_improvement:+.4f} "
              f"[{r.ci_low:+.4f},{r.ci_high:+.4f}] limb2="
              f"{'PASS' if r.passes_ranking_tolerance else 'FAIL'}  ({r.verdict})")
        for comp in [ref_id, f"{ref_id}@oracle_best_brier_NOT_A_MODEL",
                     "C1_platt"]:
            q = paired[(paired.arm == arm_id) & (paired.scope == "pooled")
                       & (paired.metric == "brier") & (paired.comparator == comp)]
            if not len(q):
                continue
            b = q.iloc[0]
            ok = b.ci_low > 0 and (bar is None or b.delta_improvement >= bar)
            print(f"       dBrier vs {comp:28s} {b.delta_improvement:+.4f} "
                  f"[{b.ci_low:+.4f},{b.ci_high:+.4f}] "
                  f"{'PASS' if ok else 'FAIL'}")
    print(f"\nwrote artifacts to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
