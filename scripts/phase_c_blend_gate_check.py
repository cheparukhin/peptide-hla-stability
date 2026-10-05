"""Phase C gate, second condition: does a frozen MINT arm add blend signal?

    .venv/bin/python scripts/phase_c_blend_gate_check.py

**This is a gate input, not §6.5.** §6's stop-scaling criterion is a conjunction
— stop if the matched outer assessment excludes a worthwhile increment *and* the
representation adds no useful blend signal. §6.2 settled the first condition;
without the second, no go/no-go against the plan's gate is complete. So this
answers exactly one yes/no question: does adding a frozen arm to the sequence
reference improve on the sequence reference alone?

What it deliberately is **not**:

* Not §6.5, which asks for a blend *or* a separately regularised residual
  correction, across the best Phase A learner and the original ensemble, with
  weights fitted on inner out-of-fold predictions. That is a Phase C experiment
  and is out of scope here.
* Not a tuned blend. One nonnegative weight on a 0.05 grid, nothing else.
* Not a result to report as Phase C's blend finding. It is sufficient to decide
  the gate's conjunction and no more.

**Leakage control.** Each arm's predictions are already out-of-fold: fold k's
predictions come from a model fitted without fold k. The blend weight for fold k
is then chosen on the *other four folds*' out-of-fold predictions and labels, so
fold k's labels never touch the weight. That is weaker than §3.6's inner-OOF
procedure — it uses outer-fold predictions to pick a weight — but it is clean with
respect to the fold being scored, which is what this question needs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pepstab import evaluation as ev  # noqa: E402

OUT = ROOT / "reports" / "further_research"
ARMS = ("joint", "separate", "esm2")
#: Nonnegative weights summing to one, per §3.6. One parameter on a coarse grid:
#: a finer grid would be tuning, and this is a gate check.
GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def panel_median(allele, y, pred, panel, rank) -> float:
    rho = ev.per_allele_spearman(allele, y, pred, panel)
    p = ev.panel_spearman(rho, rank)
    return float(p.median()) if len(p) else float("nan")


def main() -> None:
    seq = pd.read_csv(OUT / "phase_a_seq_oof_cluster.csv")
    seq = seq[seq.role == "outer_held"][["pair_id", "y_pred"]].rename(
        columns={"y_pred": "seq"}
    )

    base = pd.read_csv(OUT / "phase_c_oof_joint.csv").sort_values("pair_id").reset_index(drop=True)
    frame = base[["pair_id", "allele", "cluster_id", "outer_fold", "y"]].merge(
        seq, on="pair_id", how="inner", validate="one_to_one"
    )
    assert len(frame) == len(base), f"{len(frame)} rows after the A0 join, expected {len(base)}"
    for arm in ARMS:
        a = pd.read_csv(OUT / f"phase_c_oof_{arm}.csv")[["pair_id", "y_pred"]]
        frame = frame.merge(a.rename(columns={"y_pred": arm}), on="pair_id",
                            validate="one_to_one")

    y = frame.y.to_numpy()
    panel = ev.eligible_alleles(frame.allele, y, split="val")
    rank = ev.rankable_alleles(frame.allele, y, panel)
    folds = sorted(frame.outer_fold.unique())

    ref = panel_median(frame.allele, y, frame.seq.to_numpy(), panel, rank)
    print(f"sequence reference (A0 outer-held): {ref:.4f} on {len(panel)} alleles, "
          f"{len(frame):,} rows\n")

    rows, blended = [], {}
    for arm in ARMS:
        # Per-fold weight, chosen on the other four folds only.
        pred = np.full(len(frame), np.nan)
        chosen = {}
        for k in folds:
            held = (frame.outer_fold == k).to_numpy()
            sel = ~held
            # Standardise both arms inside the selection partition before
            # blending: the two prediction scales differ, and a weight on
            # unstandardised predictions would conflate scale with contribution.
            def z(v, mask=sel):
                return (v - v[mask].mean()) / (v[mask].std() or 1.0)

            s, a = z(frame.seq.to_numpy()), z(frame[arm].to_numpy())
            sel_panel = ev.eligible_alleles(frame.allele[sel], y[sel], split="val")
            sel_rank = ev.rankable_alleles(frame.allele[sel], y[sel], sel_panel)
            best_w, best = 0.0, -np.inf
            for w in GRID:
                sc = panel_median(frame.allele[sel], y[sel],
                                  ((1 - w) * s + w * a)[sel], sel_panel, sel_rank)
                if sc > best:
                    best, best_w = sc, float(w)
            chosen[int(k)] = best_w
            pred[held] = ((1 - best_w) * s + best_w * a)[held]

        blended[arm] = pred
        score = panel_median(frame.allele, y, pred, panel, rank)
        res = ev.paired_cluster_bootstrap(
            frame.cluster_id, frame.allele, y, frame.seq.to_numpy(), pred,
            alleles=panel,
        )
        lo, hi = res["ci95"]
        # The sign test, per the standing rule: a difference of medians alone is
        # not evidence of a per-allele effect on this metric.
        rho_ref = ev.panel_spearman(
            ev.per_allele_spearman(frame.allele, y, frame.seq.to_numpy(), panel), rank)
        rho_bl = ev.panel_spearman(
            ev.per_allele_spearman(frame.allele, y, pred, panel), rank)
        d = (rho_bl - rho_ref).dropna()
        from scipy.stats import wilcoxon
        p = float(wilcoxon(rho_bl.reindex(d.index), rho_ref.reindex(d.index)).pvalue)

        rows.append({
            "arm": arm,
            "weights_per_fold": json.dumps(chosen),
            "weight_mean": round(float(np.mean(list(chosen.values()))), 3),
            "seq_alone": round(ref, 4),
            "blend": round(score, 4),
            "delta": round(res["delta_median_spearman"], 4),
            "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
            "median_paired_delta": round(float(d.median()), 4),
            "alleles_improved": int((d > 0).sum()), "alleles": len(d),
            "wilcoxon_p": round(p, 6),
            "point_estimate_inside_ci": bool(lo <= res["delta_median_spearman"] <= hi),
            "verdict": ev.describe_delta(res),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "phase_c_blend_gate_check.csv", index=False)
    cols = ["arm", "weight_mean", "seq_alone", "blend", "delta", "ci95_low", "ci95_high",
            "median_paired_delta", "alleles_improved", "alleles", "wilcoxon_p"]
    print(out[cols].to_string(index=False))
    print()
    for r in out.itertuples():
        print(f"  {r.arm}: {r.verdict}")
        print(f"    weights per fold {r.weights_per_fold}")

    # The gate's second condition, stated plainly.
    any_signal = bool(((out.ci95_low > 0) & (out.wilcoxon_p < 0.05)).any())
    print(f"\nGATE CONDITION 2 -- does any frozen arm add useful blend signal? "
          f"{'YES' if any_signal else 'NO'}")
    if not any_signal:
        print("  No arm's blend improves on the sequence reference with both a CI")
        print("  excluding zero and a paired sign test below 0.05. Combined with")
        print("  section 6.2 meeting condition 1, the plan's conjunction to STOP SCALING")
        print("  Phase C is satisfied.")
    print("\nwrote phase_c_blend_gate_check.csv")


if __name__ == "__main__":
    main()
