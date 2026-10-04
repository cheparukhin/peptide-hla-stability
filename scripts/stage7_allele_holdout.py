"""Stage 7b: leave-allele-out evaluation, stratified by pseudosequence distance.

    .venv/bin/python scripts/stage7_allele_holdout.py                 # sequence arm
    .venv/bin/python scripts/stage7_allele_holdout.py --limit 3       # smoke test
    .venv/bin/python scripts/stage7_allele_holdout.py \
        --arm esm --npy esm=features/esm2.npy:features/esm2_pair_ids.npy --seeds 0 1 2 3 4 5

The protocol is predeclared in ``reports/stage7_allele_holdout.md``, written to
disk before the first fold ran. Strata, the eligibility bar and the ensemble
size all come from the design, not from a score.

**This is a second, separate contract.** ``EVALUATION.md`` and
``data/splits.csv`` are read and left exactly as found; nothing here
reinterprets either. The cohort is ``split in {train, val}``, so the frozen
**test split is never read**.

Adding an arm: pass ``--npy NAME=matrix.npy:pair_ids.npy`` once per
representation. Features are aligned by ``pair_id``, never positionally. The
ensemble size is checked against the contract, so an arm cannot win on budget --
stage 3 measured ensembling alone at nearly twice the worthwhile-gain bar.
"""

from __future__ import annotations

import os

# Pin BLAS to one thread **before numpy is imported** -- the thread pool is
# sized at import time, so setting these afterwards does nothing. numpy links
# against Accelerate on this machine (``np.show_config`` reports
# ``name: accelerate``), so ``VECLIB_MAXIMUM_THREADS`` is the one that actually
# binds; the others are set for portability. This matters twice over here,
# because the worker pool multiplies whatever each process grabs.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.allele_holdout import (  # noqa: E402
    DISTANCE_STRATA,
    ENSEMBLE_SIZE,
    Arm,
    FeatureSource,
    cohort_table,
    eligible_holdout_alleles,
    load_cohort,
    run_fold,
    score_fold,
    sequence_arm,
    stratum_summary,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "reports"
PRED_DIR = REPO_ROOT / "preds"

#: Per-worker state. Built once in the initializer so the 77 MB feature blocks
#: are never pickled per fold.
_STATE: dict = {}


def _init_worker(arm: Arm) -> None:
    cohort = load_cohort()
    _STATE["cohort"] = cohort
    _STATE["arm"] = arm
    _STATE["features"] = {s.name: s.build(cohort) for s in arm.sources}


def _run_one(allele: str) -> tuple[dict, list[dict], np.ndarray, np.ndarray]:
    cohort, arm = _STATE["cohort"], _STATE["arm"]
    pred, records = run_fold(cohort, _STATE["features"], arm, allele)
    scores = score_fold(cohort, allele, pred)
    pair_ids = cohort.loc[cohort["allele"] == allele, "pair_id"].to_numpy()
    return scores, records, pair_ids, pred


def parse_npy(specs: list[str]) -> tuple[FeatureSource, ...]:
    """``NAME=matrix.npy:pair_ids.npy`` -> feature sources.

    Rejects a prediction file outright. A file in ``preds/`` holds the output of
    a model that was fitted on **every** allele, so scoring it here would score
    a model on alleles it trained on and report the leak as pan-allele
    generalisation -- a number that would look like a win. Leave-allele-out
    refits 68 times, so an arm has to supply *features*, not predictions.
    """
    out = []
    for spec in specs:
        if "=" not in spec or ":" not in spec:
            raise SystemExit(f"--npy expects NAME=matrix.npy:pair_ids.npy, got {spec!r}")
        name, paths = spec.split("=", 1)
        matrix_path, pair_id_path = paths.rsplit(":", 1)
        for path in (matrix_path, pair_id_path):
            if path.endswith(".csv") or "preds/" in path:
                raise SystemExit(
                    f"{path!r} looks like a prediction file. Stage 7b refits a "
                    "model per held-out allele, so it needs a feature matrix, "
                    "not predictions: a preds/*.csv was produced by a model "
                    "fitted on every allele, and scoring it here would report "
                    "that leak as pan-allele generalisation. Supply "
                    "NAME=features.npy:pair_ids.npy instead.")
        out.append(FeatureSource(name=name, kind="npy", matrix_path=matrix_path,
                                 pair_id_path=pair_id_path))
    return tuple(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="seq", help="arm name, used in filenames")
    ap.add_argument("--input-set", default="pep_pseudo",
                    choices=["pep", "pep_pseudo", "pep_domain"])
    ap.add_argument("--npy", nargs="*", default=None,
                    help="NAME=matrix.npy:pair_ids.npy, one per representation. "
                         "Replaces the sequence encodings.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--hidden", nargs="+", type=int, default=[256, 64],
                    help="only used with --npy; the sequence arm inherits "
                         "stage 2's selected configs")
    ap.add_argument("--l2", type=float, default=1e-5, help="only used with --npy")
    ap.add_argument("--workers", type=int, default=2,
                    help="worker processes (capped at 2: this machine is shared)")
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N alleles -- smoke test, writes "
                         "*_partial files")
    args = ap.parse_args()

    if args.workers > 2:
        raise SystemExit("--workers is capped at 2; five other agents share this "
                         "8-core machine")

    seeds = tuple(args.seeds)
    if args.npy:
        sources = parse_npy(args.npy)
        configs = {s.name: (tuple(args.hidden), args.l2) for s in sources}
        arm = Arm(name=args.arm, sources=sources, configs=configs, seeds=seeds)
    else:
        arm = sequence_arm(seeds=seeds, input_set=args.input_set)
    arm.validate(ENSEMBLE_SIZE)

    cohort = load_cohort()
    alleles = eligible_holdout_alleles(cohort)
    info = cohort_table(cohort, alleles)
    todo = alleles[:args.limit] if args.limit else alleles
    partial = args.limit is not None
    suffix = "_partial" if partial else ""

    print("stage 7b: leave-allele-out, stratified by pseudosequence distance")
    print(f"  arm            : {arm.name}  "
          f"({len(arm.sources)} representations x {len(seeds)} seeds = "
          f"{arm.n_members()} networks per fold)")
    print(f"  cohort         : {len(cohort):,} rows, splits "
          f"{sorted(cohort['split'].unique())} -- test never read")
    print(f"  folds          : {len(todo)} of {len(alleles)} eligible alleles")
    for name in DISTANCE_STRATA:
        sel = info[info["stratum"] == name]
        print(f"    {name:24s} {len(sel):2d} alleles, {int(sel['n_rows'].sum()):,} rows")
    print(f"  peptide overlap: median {info['peptide_seen_share'].median():.3f} of a "
          f"held-out allele's rows carry a peptide seen on another allele")
    print()

    started = time.perf_counter()
    per_allele, records, preds = [], [], []
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(arm,)) as pool:
        for i, (scores, recs, pair_ids, pred) in enumerate(
                pool.imap(_run_one, todo), start=1):
            per_allele.append(scores)
            records.extend(recs)
            preds.append(pd.DataFrame({"pair_id": pair_ids, "y_pred": pred}))
            row = info.loc[scores["held_out_allele"]]
            print(f"  [{i:2d}/{len(todo)}] {scores['held_out_allele']:18s} "
                  f"d={row['dist_to_nearest_fit_allele']} "
                  f"n={scores['n_rows']:4d}  rho={scores['spearman']:+.4f}  "
                  f"mae={scores['mae_log1p']:.3f}  p@10={scores['precision_at_10']:.2f}",
                  flush=True)
    wall = time.perf_counter() - started

    per_allele = pd.DataFrame(per_allele)
    # ``n_rows`` is in both frames and must agree; keep the scored one and check.
    detail = per_allele.set_index("held_out_allele").join(
        info.drop(columns=["n_rows"])).reset_index()
    assert (detail.set_index("held_out_allele")["n_rows"]
            == info.loc[detail["held_out_allele"], "n_rows"]).all(), \
        "scored row count disagrees with the cohort table"
    summary = stratum_summary(per_allele, info.loc[todo])

    print(f"\nper-stratum median per-allele Spearman "
          f"(allele-level bootstrap, 95% CI):")
    print(summary[["stratum", "n_alleles", "median_spearman", "ci95_low",
                   "ci95_high", "median_mae_log1p"]].to_string(index=False))
    print("\nCONFOUND (HACKATHON_PLAN.md reason 1): each allele's peptide panel was")
    print("partly selected by predicted affinity for that allele, so distance is")
    print("confounded with panel difficulty. See reports/stage7_allele_holdout.md §4.")

    REPORT_DIR.mkdir(exist_ok=True)
    PRED_DIR.mkdir(exist_ok=True)
    detail.to_csv(REPORT_DIR / f"stage7_allele_holdout_per_allele_{arm.name}{suffix}.csv",
                  index=False)
    summary.to_csv(REPORT_DIR / f"stage7_allele_holdout_strata_{arm.name}{suffix}.csv",
                   index=False)
    pd.DataFrame(records).to_csv(
        REPORT_DIR / f"stage7_allele_holdout_runs_{arm.name}{suffix}.csv", index=False)
    pd.concat(preds, ignore_index=True).sort_values("pair_id").to_csv(
        PRED_DIR / f"stage7_allele_holdout_{arm.name}{suffix}.csv", index=False)

    (REPORT_DIR / f"stage7_allele_holdout_headline_{arm.name}{suffix}.json").write_text(
        json.dumps({
            "contract": "stage 7b leave-allele-out -- separate from EVALUATION.md",
            "arm": arm.name,
            "representations": [s.name for s in arm.sources],
            # Recorded explicitly: the sequence arm inherits its L2 from
            # stage 2's selected grid, and that grid is being extended
            # downward. If the baseline moves, this field says whether this run
            # predates the move.
            "configs": {k: {"hidden": list(v[0]), "l2": v[1]}
                        for k, v in arm.configs.items()},
            "seeds": list(seeds),
            "ensemble_members_per_fold": arm.n_members(),
            "cohort_splits": sorted(cohort["split"].unique().tolist()),
            "test_split_read": False,
            "n_cohort_rows": len(cohort),
            "n_folds": len(todo),
            "strata": summary.to_dict(orient="records"),
            "median_peptide_seen_share": float(info["peptide_seen_share"].median()),
            "wall_seconds": round(wall, 1),
        }, indent=2, default=float) + "\n")

    print(f"\nwrote reports/stage7_allele_holdout_*_{arm.name}{suffix}.csv/.json")
    print(f"total wall time: {wall / 60:.1f} min over {len(records)} networks")
    if partial:
        print("partial run: *_partial filenames, canonical reports left alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
