"""Score prediction files against the frozen splits. The shared entry point.

    # one model
    python scripts/evaluate.py --split val preds/seq_baseline.csv

    # several, plus paired CIs against the first
    python scripts/evaluate.py --split val preds/seq.csv preds/esm.csv

    # per-allele detail
    python scripts/evaluate.py --split val preds/esm.csv --per-allele

Input files are CSVs with ``pair_id,y_pred`` where ``y_pred`` is on the log1p
scale (see EVALUATION.md). The model name is taken from the filename stem.

``--split test`` is the stage 6 scoring run. Score every model in one command so
the test set is touched once.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import (  # noqa: E402
    MIN_ROWS_BY_SPLIT,
    MIN_WORTHWHILE_DELTA_SPEARMAN,
    N_BOOTSTRAP,
    eligible_alleles,
    paired_cluster_bootstrap,
    score,
    score_by_distance,
)


def read_predictions(path: Path, truth: pd.DataFrame) -> pd.Series:
    """Load one prediction file and align it to the split's ``pair_id`` order."""
    preds = pd.read_csv(path)
    missing = {"pair_id", "y_pred"} - set(preds.columns)
    if missing:
        raise SystemExit(f"{path}: missing column(s) {sorted(missing)}")
    if preds.pair_id.duplicated().any():
        raise SystemExit(f"{path}: duplicate pair_id rows")

    merged = truth[["pair_id"]].merge(preds, on="pair_id", how="left")
    if merged.y_pred.isna().any():
        n = int(merged.y_pred.isna().sum())
        raise SystemExit(
            f"{path}: no prediction for {n:,} of {len(truth):,} rows in this "
            "split. Predict every row, or score a split the file covers."
        )
    extra = len(preds) - len(truth)
    if extra > 0:
        print(f"  note: {path.name} also covers {extra:,} pair_ids outside this "
              "split; they are ignored.")
    return merged.y_pred


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", nargs="+", type=Path)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"],
                    help="which frozen split to score (default: val)")
    ap.add_argument("--per-allele", action="store_true",
                    help="print the per-allele Spearman and precision@10 tables")
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP,
                    help=f"cluster-bootstrap resamples (default {N_BOOTSTRAP})")
    ap.add_argument("--by-distance", action="store_true",
                    help="also score within distance-to-training strata")
    ap.add_argument("--no-ci", action="store_true",
                    help="skip the paired confidence intervals")
    args = ap.parse_args()

    if args.split == "test":
        print("!! Scoring the TEST split. Per EVALUATION.md this happens once, "
              "at stage 6, with every model in one command.\n")

    truth = load_with_splits()
    truth = truth[truth.split == args.split].sort_values("pair_id").reset_index(drop=True)
    alleles = eligible_alleles(truth.allele, truth.y_log1p, split=args.split)
    print(f"split={args.split}  rows={len(truth):,}  "
          f"eligible alleles={len(alleles)} "
          f"(>={MIN_ROWS_BY_SPLIT[args.split]} rows each, "
          f"{truth.allele.isin(alleles).mean():.1%} of split rows)\n")

    scored = []
    for path in args.predictions:
        y_pred = read_predictions(path, truth)
        scored.append((path.stem, y_pred,
                       score(path.stem, truth.allele, truth.y_log1p, y_pred, alleles)))

    table = pd.DataFrame([s.as_row() for _, _, s in scored])
    print(table.to_string(index=False))

    if args.per_allele:
        for name, _, s in scored:
            print(f"\n--- {name}: per-allele Spearman ---")
            detail = s.precision_table.join(s.per_allele)
            print(detail[["n", "spearman", "precision_at_k", "base_rate",
                          "ceiling"]].to_string())
            undefined = s.per_allele[s.per_allele.isna()]
            if len(undefined):
                print(f"  undefined for: {', '.join(undefined.index)}")

    if args.by_distance:
        print("\nby distance from each held-out peptide to its nearest training "
              "peptide:")
        strata = pd.concat([
            score_by_distance(name, truth.dist_to_train, truth.allele,
                              truth.y_log1p, y_pred, split=args.split)
            for name, y_pred, _ in scored
        ])
        print(strata.to_string(index=False))
        print("  If a model is exploiting residual similarity at the split "
              "boundary, d=4 should score better than d>=5.")

    if len(scored) > 1 and not args.no_ci:
        base_name, base_pred, _ = scored[0]
        print(f"\npaired cluster bootstrap vs '{base_name}' "
              f"({args.n_boot:,} resamples, clusters resampled whole):")
        for name, y_pred, _ in scored[1:]:
            r = paired_cluster_bootstrap(truth.cluster_id, truth.allele,
                                         truth.y_log1p, base_pred, y_pred,
                                         alleles, n_boot=args.n_boot)
            lo, hi = r["ci95"]
            if r.get("undefined"):
                print(f"  {name:28s} undefined -- {r['reason']}. Compare the "
                      "primary metric against a model that ranks within allele; "
                      "use MAE for constant references.")
                continue
            if r["meets_min_worthwhile"]:
                verdict = f"improvement, meets the {MIN_WORTHWHILE_DELTA_SPEARMAN} bar"
            elif r["rules_out_min_worthwhile"] and r["conclusive"] and hi < 0:
                verdict = "worse"
            elif r["rules_out_min_worthwhile"]:
                verdict = (f"strong negative: rules out a "
                           f"{MIN_WORTHWHILE_DELTA_SPEARMAN} gain")
            elif r["conclusive"]:
                verdict = f"real but below the {MIN_WORTHWHILE_DELTA_SPEARMAN} bar"
            else:
                verdict = "inconclusive (CI crosses 0) -- not a negative result"
            print(f"  {name:28s} delta={r['delta_median_spearman']:+.4f}  "
                  f"95% CI [{lo:+.4f}, {hi:+.4f}]  {verdict}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
