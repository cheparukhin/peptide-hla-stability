"""Phase B: score the two-part floor model against the ordinary regressor.

    .venv/bin/python scripts/phase_b_two_part.py

Two components, each at the same inner/encoding/seed budget as a censored arm:

- ``p_above``  -- ``P(recorded label is above the assay floor)``, the floor
  indicator regressed under squared error;
- ``mu_cond``  -- ``E[log1p(t) | above the floor]``, fitted on above-floor rows
  only, so it is a conditional magnitude and not a mixture.

**The arithmetic this phase was asked to be careful about.** For predicting the
*recorded* ``log1p`` labels,

    E[log1p(t)] = p_above * E[log1p(t) | above floor]

treats the floor-coded component as contributing zero. That is **correct as an
expectation of the recorded label**, because the recorded value at a floor row
is literally 0. It is **not** an estimate of true sub-floor lifetime: a peptide
below the detection limit has a positive half-life the assay could not resolve,
and nothing here estimates it. And it is **not** generally the optimal ranking
score -- multiplying a probability by a conditional magnitude can reorder pairs
that the conditional magnitude alone orders correctly. So the product's ranking
is an empirical question, compared here against the ordinary regressor rather
than assumed either way, and reported whichever way it falls.

Four scores are therefore compared, not one:

| score | what it is |
|---|---|
| ``B0`` | the ordinary regressor's ``mu`` -- the reference |
| ``product`` | ``p_above * mu_cond`` -- the recorded-label expectation |
| ``mu_cond`` | the conditional magnitude alone, as a ranking score |
| ``p_above`` | the floor probability alone, as a ranking score |

``p_above`` is also scored as a *floor probability* against the censored arms, on
Brier, since that is what the two-part model produces natively and what the
censored arms produce from their own fit.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
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

OUT_DIR = REPO_ROOT / "reports" / "further_research"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    sx = args.suffix

    tp = np.load(OUT_DIR / f"phase_b_two_part_members{sx}.npz", allow_pickle=False)
    arms = np.load(OUT_DIR / f"phase_b_nested_members{sx}.npz", allow_pickle=False)
    threshold = floor_threshold(DEFAULT_FLOOR_HOURS)
    folds = sorted({int(k.split("__")[1]) for k in tp.files
                    if k.startswith("above__") and k.endswith("__preds")})

    frames = []
    for k in folds:
        pid = tp[f"meta__{k}__held_pair_id"]
        # Clipped to [0, 1] only when read as a probability, never during fitting:
        # a least-squares fit to a 0/1 target can leave the unit interval, and
        # clipping during training would bias the gradient.
        p_above = np.clip(tp[f"above__{k}__preds"].mean(axis=0), 0.0, 1.0)
        mu_cond = tp[f"magnitude__{k}__preds"].mean(axis=0)
        b0 = arms[f"B0__{k}__mus"].mean(axis=0)
        b0_sig = arms[f"B0__{k}__sigmas"]
        frames.append(pd.DataFrame({
            "pair_id": pid, "outer_fold": k, "p_above": p_above,
            "mu_cond": mu_cond, "product": p_above * mu_cond, "b0_mu": b0,
            "b0_p_floor": cal.prob_floor_mixture(arms[f"B0__{k}__mus"],
                                                 b0_sig, threshold)}))
    pred = pd.concat(frames, ignore_index=True)
    assert pred.pair_id.is_unique

    df = load_with_splits()
    train = df[df.split == "train"][["pair_id", "allele", "thalf_hours",
                                     "y_log1p", "cluster_id"]].reset_index(drop=True)
    meta = pred[["pair_id", "outer_fold"]].merge(train, on="pair_id",
                                                 validate="one_to_one")
    is_floor = cal.floor_indicator(meta.thalf_hours, "recorded_zero")
    y = meta.y_log1p.to_numpy()
    alleles = nested.fold_eligible_alleles(meta.allele, y)
    print(f"two-part model: {len(meta):,} outer-held rows, "
          f"{int(is_floor.sum()):,} at the floor ({is_floor.mean():.4f}), "
          f"{len(alleles)} eligible alleles, folds {folds}")
    print(f"  p_above: mean {pred.p_above.mean():.4f} against an observed "
          f"above-floor share of {1 - is_floor.mean():.4f}; "
          f"{int((pred.p_above <= 0).sum()):,} clipped at 0, "
          f"{int((pred.p_above >= 1).sum()):,} at 1")

    # --- ranking: the question the declaration said to ask explicitly ------
    rows = []
    for name in ("product", "mu_cond", "p_above"):
        score_vec = pred[name].to_numpy()
        # p_above ranks the WRONG way round as a half-life score: a high floor
        # probability means a SHORT half-life, so it is negated to be read as a
        # half-life ranking at all. Stated rather than silently flipped.
        if name == "p_above":
            score_vec = -score_vec
        sc = nested.score_fold(name, meta.allele, y, score_vec, min_rows=20)
        res = paired_cluster_bootstrap(meta.cluster_id, meta.allele, y,
                                       pred.b0_mu.to_numpy(), score_vec,
                                       alleles=alleles, n_boot=args.n_boot)
        rows.append({
            "score": name if name != "p_above" else "-p_above",
            "metric": "median_per_allele_spearman",
            "value": sc.median_spearman, "reference_b0": None,
            "delta_improvement": res["delta_median_spearman"],
            "ci_low": res["ci95"][0], "ci_high": res["ci95"][1],
            "verdict": describe_delta(res)})
        print(f"  ranking {rows[-1]['score']:10s} median rho {sc.median_spearman:.4f}"
              f"  vs B0 {res['delta_median_spearman']:+.4f} "
              f"[{res['ci95'][0]:+.4f},{res['ci95'][1]:+.4f}]  {describe_delta(res)}")

    b0_sc = nested.score_fold("B0", meta.allele, y, pred.b0_mu.to_numpy())
    print(f"  reference  B0         median rho {b0_sc.median_spearman:.4f}")

    # --- the floor probability, against every arm's own -------------------
    print("\n  P(floor) = 1 - p_above, Brier against each arm:")
    p_floor_two_part = 1.0 - pred.p_above.to_numpy()
    comparators = {"B0": pred.b0_p_floor.to_numpy()}
    for arm in ("B1", "B2"):
        if f"{arm}__{folds[0]}__mus" not in arms.files:
            continue
        pieces = []
        for k in folds:
            pieces.append(pd.DataFrame({
                "pair_id": arms[f"{arm}__{k}__held_pair_id"],
                "p": cal.prob_floor_mixture(arms[f"{arm}__{k}__mus"],
                                            arms[f"{arm}__{k}__sigmas"], threshold)}))
        comparators[arm] = (meta[["pair_id"]]
                            .merge(pd.concat(pieces, ignore_index=True),
                                   on="pair_id", validate="one_to_one").p.to_numpy())
    print(f"    two-part Brier {cal.brier(is_floor, p_floor_two_part):.4f}, "
          f"fixed-bin ECE {cal.fixed_bin_ece(is_floor, p_floor_two_part)[0]:.4f}, "
          f"AUROC {cal.floor_auroc(is_floor, p_floor_two_part):.4f}")
    for name, comp in comparators.items():
        for metric_name, metric, lower in (
                ("brier", cal.brier, True),
                ("ece_fixed_bin", lambda yy, pp: cal.fixed_bin_ece(yy, pp)[0], True),
                ("floor_auroc_from_p", cal.floor_auroc, False)):
            res = cal.paired_cluster_bootstrap_metric(
                meta.cluster_id, metric, reference=comp, arm=p_floor_two_part,
                labels=is_floor, lower_is_better=lower, n_boot=args.n_boot)
            rows.append({"score": "two_part_p_floor", "metric": metric_name,
                         "value": res["arm"], "reference_b0": res["reference"],
                         "comparator": name,
                         "delta_improvement": res["delta_improvement"],
                         "ci_low": res["ci95"][0], "ci_high": res["ci95"][1],
                         "verdict": describe_delta(res, min_delta=0.0)})
            if metric_name == "brier":
                print(f"    vs {name}: dBrier {res['delta_improvement']:+.4f} "
                      f"[{res['ci95'][0]:+.4f},{res['ci95'][1]:+.4f}]  "
                      f"{describe_delta(res, min_delta=0.0)}")

    # --- magnitude accuracy on the recorded label -------------------------
    print("\n  recorded-label MAE (log1p):")
    for name, vec in (("B0 mu", pred.b0_mu.to_numpy()),
                      ("product", pred["product"].to_numpy()),
                      ("mu_cond", pred.mu_cond.to_numpy())):
        mae = float(np.abs(vec - y).mean())
        mae_floor = float(np.abs(vec - y)[is_floor].mean())
        mae_meas = float(np.abs(vec - y)[~is_floor].mean())
        rows.append({"score": name, "metric": "mae_log1p", "value": mae,
                     "mae_floor_rows": mae_floor, "mae_measured_rows": mae_meas})
        print(f"    {name:10s} all {mae:.4f}  floor rows {mae_floor:.4f}  "
              f"measured rows {mae_meas:.4f}")
    print("    (floor-row MAE is error against the recorded 0, which "
          "EVALUATION.md records as biased in BOTH directions -- not a bound)")

    pred.to_csv(OUT_DIR / f"phase_b_two_part_oof{sx}.csv", index=False)
    pd.DataFrame(rows).to_csv(
        OUT_DIR / f"phase_b_two_part_results{sx}.csv", index=False)
    print(f"\nwrote artifacts to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
