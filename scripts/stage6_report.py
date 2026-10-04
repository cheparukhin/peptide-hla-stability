"""Stage 6: emit the full analysis table set for one or more prediction files.

    # worked example on validation, two arms, paired CIs against the first
    .venv/bin/python scripts/stage6_report.py --split val \
        preds/seq_baseline.csv preds/seq_ensemble_pep_domain.csv

    # label the arms explicitly, add the nested near-neighbour evaluation
    .venv/bin/python scripts/stage6_report.py --split val \
        baseline=preds/seq_baseline.csv esm2=preds/esm2.csv --nested

Each positional argument is a prediction CSV in the frozen two-column format
(``pair_id,y_pred``, ``y_pred`` on the log1p scale -- see EVALUATION.md),
optionally prefixed with ``NAME=`` to label the arm. Without a label the
filename stem is used. **Nothing here is specific to any arm**: a structural or
ESM-2 prediction file drops in with no change.

Writes a CSV per analysis plus a JSON manifest to ``--out-dir`` (default
``reports/``), and prints the same tables.

``--split test`` is the single stage 6 scoring run and is the orchestrator's
call, not this script's: it prints a warning and expects every arm in one
command.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab import evaluation, stage6  # noqa: E402
from pepstab.data import TARGET  # noqa: E402
from pepstab.evaluation import (  # noqa: E402
    MIN_ROWS_BY_SPLIT,
    MIN_ROWS_PER_ALLELE_IN_STRATUM,
    N_BOOTSTRAP,
    describe_delta,
    eligible_alleles,
    paired_cluster_bootstrap,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_arm(spec: str) -> tuple[str, Path]:
    """``NAME=path/to.csv`` or ``path/to.csv`` -> (name, path)."""
    if "=" in spec:
        name, _, path = spec.partition("=")
        return name.strip(), Path(path.strip())
    p = Path(spec)
    return p.stem, p


def _rel(path: Path) -> str:
    """Repo-relative when the output lives in the repo, absolute otherwise."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _write(frame: pd.DataFrame, out_dir: Path, prefix: str, stem: str,
           written: list[str], index: bool = False) -> None:
    path = out_dir / f"{prefix}_{stem}.csv"
    frame.to_csv(path, index=index)
    written.append(_rel(path))
    print(f"  wrote {_rel(path)}  ({len(frame)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", metavar="[NAME=]PREDICTIONS.csv")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports")
    ap.add_argument("--prefix", default=None,
                    help="output filename prefix (default: stage6_<split>)")
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--stratum-min-rows", type=int,
                    default=MIN_ROWS_PER_ALLELE_IN_STRATUM,
                    help="rows an allele needs inside a distance stratum "
                         f"(frozen default {MIN_ROWS_PER_ALLELE_IN_STRATUM}; "
                         "validation needs a lower bar -- see the report)")
    ap.add_argument("--nested", action="store_true",
                    help="also run the nested near-neighbour evaluation inside "
                         "the training split (analysis 5)")
    ap.add_argument("--nested-arm", metavar="[NAME=]OOF.csv", default=None,
                    help="out-of-fold predictions for the nested evaluation; "
                         "defaults to the built-in ridge reference arm")
    ap.add_argument("--skip-ci", action="store_true")
    args = ap.parse_args()

    prefix = args.prefix or f"stage6_{args.split}"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    started = time.time()

    if args.split == "test":
        print("!! Scoring the TEST split. Per EVALUATION.md this happens once, "
              "at stage 6, with every arm in one command.\n")

    frame = stage6.load_split(args.split)
    y_true = frame[TARGET].to_numpy(dtype=float)
    alleles = eligible_alleles(frame["allele"], y_true, split=args.split)
    print(f"split={args.split}  rows={len(frame):,}  clusters="
          f"{frame.cluster_id.nunique():,}  peptides={frame.peptide.nunique():,}")
    print(f"eligible alleles={len(alleles)} (>= {MIN_ROWS_BY_SPLIT[args.split]} "
          f"rows each, {frame.allele.isin(alleles).mean():.1%} of split rows)\n")

    arms: list[tuple[str, np.ndarray]] = []
    for spec in args.arms:
        name, path = parse_arm(spec)
        arms.append((name, stage6.read_predictions(path, frame).to_numpy()))
    names = [n for n, _ in arms]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate arm names: {names}. Use NAME=path to label.")

    manifest: dict = {
        "split": args.split,
        "n_rows": int(len(frame)),
        "n_clusters": int(frame.cluster_id.nunique()),
        "n_peptides": int(frame.peptide.nunique()),
        "n_eligible_alleles": len(alleles),
        "eligible_row_coverage": float(frame.allele.isin(alleles).mean()),
        "min_rows_per_allele": MIN_ROWS_BY_SPLIT[args.split],
        "arms": {n: str(parse_arm(s)[1]) for n, s in zip(names, args.arms)},
        "n_boot": args.n_boot,
        "bootstrap_seed": evaluation.BOOTSTRAP_SEED,
        "stratum_min_rows": args.stratum_min_rows,
    }

    # ---- headline (the frozen contract, unchanged) -----------------------
    scores = [evaluation.score(n, frame["allele"], y_true, p, alleles)
              for n, p in arms]
    summary = pd.DataFrame([s.as_row() for s in scores])
    print("=== headline (EVALUATION.md contract) ===")
    print(summary.to_string(index=False), "\n")
    _write(summary, out_dir, prefix, "summary", written)

    per_allele = pd.concat(
        [s.per_allele_frame().assign(model=s.name).reset_index() for s in scores])
    _write(per_allele, out_dir, prefix, "per_allele", written)

    # ---- 1. distance stratification --------------------------------------
    print("\n=== 1. nearest-neighbour distance stratification ===")
    census = stage6.distance_census(frame)
    print(census.to_string(index=False))
    _write(census, out_dir, prefix, "distance_census", written)

    panel = stage6.stratum_panel(frame, min_rows=args.stratum_min_rows)
    print(f"\ncommon allele panel at a {args.stratum_min_rows}-row bar: "
          f"{panel['n_common']} alleles "
          f"(eligible per stratum: {panel['n_eligible']})")
    for label, cov in panel["row_coverage"].items():
        print(f"  {label}: panel covers {cov:.1%} of the stratum's rows")
    if panel["n_common"] and min(panel["row_coverage"].values()) < 0.5:
        print("  !! the shared panel covers under half of a stratum's rows; "
              "the stratum comparison on this split is not representative")
    manifest["distance_strata"] = {
        "n_rows": panel["n_rows"], "n_eligible": panel["n_eligible"],
        "n_common_alleles": panel["n_common"],
        "row_coverage": panel["row_coverage"],
        "common_alleles": panel["common"],
        "n_rows_at_d6_plus": int((frame.dist_to_train >= 6).sum()),
        "n_peptides_at_d6_plus": int(
            frame.loc[frame.dist_to_train >= 6, "peptide"].nunique()),
    }

    strata = pd.concat([stage6.score_strata(n, frame, p,
                                            min_rows=args.stratum_min_rows)
                        for n, p in arms])
    print()
    print(strata.to_string(index=False))
    print("  d=4 scoring above d>=5 is the signature of exploiting residual "
          "similarity at the split boundary.")
    _write(strata, out_dir, prefix, "distance_strata", written)

    # ---- 2. the differential target --------------------------------------
    print("\n=== 2. the differential target (delta between alleles) ===")
    pairs = stage6.differential_pairs(frame, alleles)
    print(f"{pairs.peptide.nunique():,} peptides on >= 2 eligible alleles -> "
          f"{len(pairs):,} allele-pair comparisons "
          f"({int((pairs.delta_true == 0).sum()):,} undecidable, equal labels)")
    diff_rows, diff_pairs = [], []
    for n, p in arms:
        lookup = pd.Series(p, index=frame["pair_id"].to_numpy())
        r = stage6.score_differential(n, pairs, lookup)
        diff_pairs.append(r.pop("_per_allele_pair").assign(model=n))
        diff_rows.append(r)
    diff = pd.DataFrame(diff_rows)
    print(diff.drop(columns=["n_peptides", "n_pairs", "n_undecidable"]
                    ).to_string(index=False))
    print("  a constant predictor scores concordance 0.500 by construction; "
          "an allele-only predictor can beat that on concordance but scores "
          "~0 on median_allele_pair_spearman, which needs peptide-specific "
          "groove chemistry.")
    _write(diff, out_dir, prefix, "differential", written)
    _write(pd.concat(diff_pairs), out_dir, prefix, "differential_allele_pairs",
           written)
    manifest["differential"] = {
        "n_peptides": int(pairs.peptide.nunique()),
        "n_pairs": int(len(pairs)),
        "n_undecidable": int((pairs.delta_true == 0).sum()),
        "n_allele_pairs": int(pairs.groupby(["allele_a", "allele_b"]).ngroups),
        "min_peptides_per_allele_pair": stage6.MIN_PEPTIDES_PER_ALLELE_PAIR,
    }

    # ---- 4. precision@10 --------------------------------------------------
    print("\n=== 4. precision@10 at 2 hours ===")
    prec = pd.concat([stage6.precision_report(n, frame, p, alleles)
                      for n, p in arms])
    prec_summary = pd.DataFrame(
        [stage6.precision_summary(prec[prec.model == n]) for n in names])
    print(prec_summary.to_string(index=False))
    print("  ties are credited by expectation, so a constant predictor scores "
          "exactly its base rate and median_lift 0.")
    _write(prec_summary, out_dir, prefix, "precision_summary", written)
    _write(prec.reset_index(), out_dir, prefix, "precision_per_allele", written)

    # ---- 3. paired uncertainty -------------------------------------------
    if len(arms) > 1 and not args.skip_ci:
        base_name, base_pred = arms[0]
        print(f"\n=== 3. paired cluster bootstrap vs '{base_name}' "
              f"({args.n_boot:,} resamples, seed "
              f"{evaluation.BOOTSTRAP_SEED}, whole peptide clusters) ===")
        ci_rows = []
        for n, p in arms[1:]:
            r = paired_cluster_bootstrap(frame.cluster_id, frame["allele"],
                                         y_true, base_pred, p, alleles,
                                         n_boot=args.n_boot)
            lo, hi = r["ci95"]
            ci_rows.append({
                "statistic": "median_per_allele_spearman", "baseline": base_name,
                "model": n, "delta": r["delta_median_spearman"],
                "ci_low": lo, "ci_high": hi, "verdict": describe_delta(r),
                "n_boot": r["n_boot"],
                "n_resamples_with_dropped_alleles":
                    r.get("n_resamples_with_dropped_alleles"),
            })
            # the same paired machinery on the stage 6 statistics
            lk_a = pd.Series(base_pred, index=frame["pair_id"].to_numpy())
            lk_b = pd.Series(p, index=frame["pair_id"].to_numpy())
            da = (lk_a.reindex(pairs.pair_id_b).to_numpy()
                  - lk_a.reindex(pairs.pair_id_a).to_numpy())
            db = (lk_b.reindex(pairs.pair_id_b).to_numpy()
                  - lk_b.reindex(pairs.pair_id_a).to_numpy())
            rd = stage6.paired_cluster_delta(
                pairs.cluster_id.to_numpy(),
                stage6.delta_concordance_statistic(pairs), da, db,
                n_boot=max(200, args.n_boot // 4),
                label="differential_concordance")
            ci_rows.append({"statistic": "differential_concordance",
                            "baseline": base_name, "model": n,
                            "delta": rd["delta"], "ci_low": rd["ci95"][0],
                            "ci_high": rd["ci95"][1],
                            "verdict": ("conclusive" if rd["conclusive"]
                                        else "inconclusive (CI crosses 0)"),
                            "n_boot": rd["n_boot"],
                            "n_resamples_with_dropped_alleles": None})
            rp = stage6.paired_cluster_delta(
                frame.cluster_id.to_numpy(),
                stage6.median_precision(frame, alleles), base_pred, p,
                n_boot=max(200, args.n_boot // 4), label="median_precision_at_10")
            ci_rows.append({"statistic": "median_precision_at_10",
                            "baseline": base_name, "model": n,
                            "delta": rp["delta"], "ci_low": rp["ci95"][0],
                            "ci_high": rp["ci95"][1],
                            "verdict": ("conclusive" if rp["conclusive"]
                                        else "inconclusive (CI crosses 0)"),
                            "n_boot": rp["n_boot"],
                            "n_resamples_with_dropped_alleles": None})
        ci = pd.DataFrame(ci_rows)
        print(ci.to_string(index=False))
        _write(ci, out_dir, prefix, "paired_ci", written)

        units = stage6.resampling_unit_widths(frame, arms[0][1], arms[1][1],
                                              alleles,
                                              n_boot=max(200, args.n_boot // 5))
        print("\nresampling unit (why clusters, not rows):")
        print(units.to_string(index=False))
        print("  a narrower interval is not a better one -- the row bootstrap "
              "counts rows sharing a peptide as independent evidence.")
        _write(units, out_dir, prefix, "resampling_units", written)

    # ---- 5. nested near-neighbour evaluation ------------------------------
    if args.nested:
        print("\n=== 5. nested near-neighbour evaluation inside training ===")
        check = stage6.verify_groups_within_split()
        print(f"Hamming <= {check['radius']} peptide groups: "
              f"{check['n_groups']:,}, spanning more than one frozen split: "
              f"{check['n_groups_spanning_splits']}")
        if check["n_groups_spanning_splits"]:
            raise SystemExit("near-neighbour groups straddle a frozen split; "
                             "the nested evaluation would leak")
        train = stage6.load_split("train")
        mut = stage6.mutant_pairs(train)
        folds = stage6.nested_folds(train)
        fold_map = pd.Series(folds.to_numpy(), index=train["pair_id"].to_numpy())
        print(f"{len(mut):,} same-allele mutant comparisons "
              f"(Hamming 1-{stage6.MUTANT_RADIUS}) over "
              f"{mut.allele.nunique()} alleles and "
              f"{mut.cluster_id.nunique()} peptide clusters; "
              f"{stage6.NESTED_FOLDS} folds by peptide")

        if args.nested_arm:
            nname, npath = parse_arm(args.nested_arm)
            oof_frame = pd.read_csv(npath)
            oof = pd.Series(oof_frame["y_pred"].to_numpy(dtype=float),
                            index=oof_frame["pair_id"].to_numpy())
        else:
            nname = "ridge_pep_pseudo_blosum[reference]"
            oof = stage6.ridge_oof_predictions(train, folds)

        rows = [stage6.score_mutant_ranking(nname, mut, oof, fold_map),
                stage6.score_mutant_ranking(
                    "constant[control]", mut,
                    pd.Series(np.zeros(len(train)),
                              index=train["pair_id"].to_numpy()), fold_map)]
        nested = pd.DataFrame(rows)
        print(nested.to_string(index=False))

        use = mut[(fold_map.reindex(mut.pair_id_a.to_numpy()).to_numpy()
                   != fold_map.reindex(mut.pair_id_b.to_numpy()).to_numpy())
                  ].reset_index(drop=True)
        dp = (oof.reindex(use.pair_id_b.to_numpy()).to_numpy()
              - oof.reindex(use.pair_id_a.to_numpy()).to_numpy())
        rb = stage6.paired_cluster_delta(
            use.cluster_id.to_numpy(), stage6.mutant_concordance_statistic(use),
            np.zeros(len(use)), dp, n_boot=args.n_boot,
            label="mutant_concordance_minus_chance")
        nested.loc[nested.model == nname, "concordance_minus_chance"] = rb["delta"]
        nested.loc[nested.model == nname, "ci_low"] = rb["ci95"][0]
        nested.loc[nested.model == nname, "ci_high"] = rb["ci95"][1]
        print(f"\nconcordance - 0.5 = {rb['delta']:+.4f}  95% CI "
              f"[{rb['ci95'][0]:+.4f}, {rb['ci95'][1]:+.4f}] over "
              f"{rb['n_clusters']} peptide clusters "
              f"-- {'conclusive' if rb['conclusive'] else 'inconclusive'}")
        _write(nested, out_dir, prefix, "nested_mutant", written)
        manifest["nested"] = {
            "radius": stage6.MUTANT_RADIUS, "n_folds": stage6.NESTED_FOLDS,
            "seed": stage6.NESTED_SEED,
            "n_mutant_pairs": int(len(mut)),
            "n_pairs_cross_fold": int(len(use)),
            "n_clusters": int(mut.cluster_id.nunique()),
            "n_alleles": int(mut.allele.nunique()),
            "groups_spanning_splits": check["n_groups_spanning_splits"],
            "reference_arm": nname,
            "ci95_concordance_minus_chance": list(rb["ci95"]),
        }

    manifest["elapsed_seconds"] = round(time.time() - started, 1)
    manifest["files"] = written
    path = out_dir / f"{prefix}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\n  wrote {_rel(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
