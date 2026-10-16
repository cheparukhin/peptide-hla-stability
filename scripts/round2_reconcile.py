"""Rebuild corrected historical arithmetic without fitting or selecting models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from pepstab import evaluation as ev, nested
from pepstab.data import load_with_splits
from pepstab.round2_reporting import compare_predictions

SOURCE = ROOT / "reports/further_research"
OUT = SOURCE / "round2/corrections"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_with_splits().query("split == 'train'").sort_values("pair_id").reset_index(drop=True)
    folds = pd.read_csv(SOURCE / "phase_a_outer_folds.csv")
    frame = frame.merge(folds, on="pair_id", validate="one_to_one")
    assert np.array_equal(frame.outer_fold, nested.outer_folds(frame))
    seq = pd.read_csv(SOURCE / "phase_a_seq_oof_cluster.csv").query("role == 'outer_held'")
    predictions = {"A0_historical": frame[["pair_id", "outer_fold"]].merge(
        seq, on=["pair_id", "outer_fold"], validate="one_to_one", how="left").y_pred.to_numpy()}
    files = ["phase_a_outer_folds.csv", "phase_a_seq_oof_cluster.csv"]
    for arm in ("joint", "separate", "esm2"):
        name = f"phase_c_oof_{arm}.csv"
        files.append(name)
        source = pd.read_csv(SOURCE / name)
        keys = ["pair_id", "allele", "peptide", "cluster_id", "outer_fold"]
        merged = frame[keys + ["y_log1p"]].merge(source, on=keys, validate="one_to_one", how="left")
        assert len(source) == len(frame) and np.allclose(merged.y, merged.y_log1p, rtol=0, atol=1e-12)
        predictions[f"C_{arm}_historical"] = merged.y_pred.to_numpy()
    contrasts = [("A0_historical", name) for name in predictions if name != "A0_historical"]
    contrasts += [("C_separate_historical", "C_joint_historical"),
                  ("C_esm2_historical", "C_separate_historical"),
                  ("C_esm2_historical", "C_joint_historical")]
    comparisons = []
    points, per_fold = [], []
    for minimum in (20, 50):
        table, _, meta = compare_predictions(frame, predictions, contrasts, min_rows=minimum, n_boot=args.n_boot)
        table["population"] = "original training partition, historical outer-held predictions"
        table["selection_history"] = "historical procedures; no refit or selection repair"
        table["cost_provenance"] = "local CPU arithmetic only; historical fit costs separate"
        comparisons.append(table)
        for name, pred in predictions.items():
            score = ev.score(name, frame.allele, frame.y_log1p.to_numpy(), pred, alleles=meta["alleles"])
            points.append({"procedure": name, "panel_min_rows": minimum,
                           "pooled_median_per_allele_spearman": score.median_spearman,
                           "mae_log1p": score.mae_log1p, "rows": len(frame),
                           "eligible_alleles": len(meta["alleles"]),
                           "eligible_rows": meta["n_rows_eligible"]})
            for fold in sorted(frame.outer_fold.unique()):
                keep = frame.outer_fold.eq(fold).to_numpy()
                score = nested.score_fold(name, frame.allele[keep], frame.y_log1p.to_numpy()[keep],
                                          pred[keep], min_rows=minimum)
                per_fold.append({"procedure": name, "panel_min_rows": minimum, "fold": fold,
                                 "median_per_allele_spearman": score.median_spearman,
                                 "eligible_alleles": score.n_alleles_scored, "rows": int(keep.sum())})
    pd.concat(comparisons).to_csv(OUT / "historical_pooled_comparisons.csv", index=False)
    pd.DataFrame(points).to_csv(OUT / "historical_pooled_scores.csv", index=False)
    pd.DataFrame(per_fold).to_csv(OUT / "historical_fold_scores.csv", index=False)

    # Arithmetic corrections use the saved tables, never rounded prose.
    pooled = pd.read_csv(SOURCE / "phase_b_pooled.csv").query("p_floor_rule == 'mixture'").set_index("arm")
    paired = pd.read_csv(SOURCE / "phase_b_paired.csv")
    rank = paired.query("arm == 'B2' and scope == 'pooled' and metric == 'median_per_allele_spearman'").iloc[0]
    weights = pd.read_csv(SOURCE / "phase_c_blend_gate_check.csv")
    flat = [v for item in weights.weights_per_fold for v in json.loads(item).values()]
    a4 = pd.read_csv(SOURCE / "phase_a_a4_comparisons.csv")
    a4ref = a4.query("comparison == 'A4_l2_10__vs__matched_control'").iloc[0]
    corrected = {"b2_brier_reference": float(pooled.loc["B0", "brier"]),
                 "b2_brier_candidate": float(pooled.loc["B2", "brier"]),
                 "b2_brier_relative_reduction": float(1-pooled.loc["B2", "brier"]/pooled.loc["B0", "brier"]),
                 "b2_ece_relative_reduction": float(1-pooled.loc["B2", "ece_fixed_bin"]/pooled.loc["B0", "ece_fixed_bin"]),
                 "b2_ranking_delta": float(rank.delta_improvement),
                 "b2_ranking_ci": [float(rank.ci_low), float(rank.ci_high)],
                 "b2_ranking_tolerance": -0.02,
                 "b2_ranking_tolerance_established": bool(rank.ci_low > -0.02),
                 "historical_blend_zero_weights": sum(w == 0 for w in flat),
                 "historical_blend_total_weights": len(flat),
                 "a4_matched_delta": float(a4ref.delta_median_spearman),
                 "a4_matched_ci": [float(a4ref.ci95_lo), float(a4ref.ci95_hi)],
                 "a4_lower_bound_exceeds_worthwhile_bar": bool(a4ref.ci95_lo > 0.05),
                 "a4_upper_bound_below_worthwhile_bar": bool(a4ref.ci95_hi < 0.05),
                 "old_blend_ancestry_valid": False,
                 "old_blend_usable_to_close_complementarity": False}
    (OUT / "numerical_corrections.json").write_text(json.dumps(corrected, indent=2) + "\n")
    files += ["phase_b_pooled.csv", "phase_b_paired.csv", "phase_c_blend_gate_check.csv", "phase_a_a4_comparisons.csv"]
    (OUT / "source_hashes.json").write_text(json.dumps({name: hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()
                                                       for name in files}, indent=2) + "\n")
    p = pd.DataFrame(points).query("panel_min_rows == 20").set_index("procedure")
    fold_med = pd.DataFrame(per_fold).query("panel_min_rows == 20 and procedure == 'A0_historical'").median_per_allele_spearman.median()
    lines = ["# Historical arithmetic reconciled — 12 October 2026", "",
             "Current scope remains [NEXT_RESEARCH_PLAN.md](../../../NEXT_RESEARCH_PLAN.md). "
             "This report recomputes saved predictions, not model fits. It does not repair their selection "
             "history or provide independent confirmation. The frozen source labels/splits and earlier artifacts are unchanged.", "",
             f"All four procedures cover {len(frame):,} original training rows. At the 20-row threshold, "
             f"{int(p.iloc[0].eligible_alleles)} alleles are eligible. A0 pooled median per-allele Spearman is "
             f"**{p.loc['A0_historical','pooled_median_per_allele_spearman']:.6f}**; the median of its fold medians "
             f"is **{fold_med:.6f}**. These are distinct statistics.", "",
             "| Procedure | Pooled median per-allele Spearman |", "|---|---:|"]
    lines += [f"| {name} | {row.pooled_median_per_allele_spearman:.6f} |" for name,row in p.iterrows()]
    lines += ["", "All six declared contrasts and the 50-row sensitivity are in `historical_pooled_comparisons.csv`, "
              f"using {args.n_boot:,} paired whole-peptide-cluster resamples. This implementation was tested against "
              "the frozen resampling/ranking procedure, including tied values and constant predictions. "
              "Intervals condition on fitted predictions; search and execution variability are separate.", "",
              f"B2 floor Brier changes from {corrected['b2_brier_reference']:.6f} to "
              f"{corrected['b2_brier_candidate']:.6f}, a **{100*corrected['b2_brier_relative_reduction']:.1f}% reduction**. "
              f"Fixed-bin ECE falls {100*corrected['b2_ece_relative_reduction']:.1f}%, a different metric. "
              f"Its ranking delta is {rank.delta_improvement:+.4f} [{rank.ci_low:+.4f}, {rank.ci_high:+.4f}]; "
              "the interval does not establish the −0.02 tolerance.", "",
              f"A4 versus its matched control is {a4ref.delta_median_spearman:+.4f} "
              f"[{a4ref.ci95_lo:+.4f}, {a4ref.ci95_hi:+.4f}]. The interval straddles 0.05 and does not account "
              "for screening selection. Composition/additivity and a deployable gain remain unestablished.", "",
              f"The old blend has **{corrected['historical_blend_zero_weights']}/{len(flat)} zero weights**. "
              "Other outer-fold models could have used an assessed fold's labels; those inputs are not clean "
              "stacking data. Fold-specific scaling also changes the pooled baseline endpoint. This blend cannot "
              "settle complementarity; replacement fitting is required.", "",
              "A percentile interval missing its point estimate is flagged with the bootstrap mean-minus-point "
              "diagnostic, not declared an artefact. Per-allele signed-rank tests are not used as primary evidence "
              "because peptides couple the allele measurements. Fold spread and execution variation do not "
              "identify an assay ceiling.", "",
              "## What can be repaired from saved artifacts", "",
              "- Pooled/fold aggregation, numerical claims, weight counts and conditional cluster intervals: rebuilt here.",
              "- Global configuration selection, selected control ancestry, residual/blend/calibration inputs: require fresh fitting.",
              "- Independent confirmation and assay reliability: require reserved measurements/replicates.", "",
              "Unsafe legacy report entrypoints now stop with a pointer to the new study. Their importable helpers "
              "and historical numeric files remain available for audits. No original test rescore was performed."]
    (OUT / "REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(corrected, indent=2))


if __name__ == "__main__":
    main()
