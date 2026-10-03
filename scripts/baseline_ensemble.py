"""The NetMHCstabpan *method*, reimplemented under the frozen splits.

    .venv/bin/python scripts/baseline_ensemble.py                  # pep_pseudo
    .venv/bin/python scripts/baseline_ensemble.py --input-set pep_domain

Stage 2 reported single networks, which is a weakened version of what Rasmussen
et al. actually trained. Their protocol fits one network per cross-validation
fold per architecture and predicts with the whole ensemble (2 encodings x 3
hidden sizes x 5 folds ~ 30 networks). Measured in
``scripts/compare_to_paper.py``, ensembling just 3 seeds is worth +0.042 mean
per-allele SCC -- nearly the entire predeclared worthwhile-gain bar, from no new
information. A single-network baseline would hand stage 3 a gap it did not earn.

So this builds the strong form: **5 inner CV folds x 2 encodings x 3 seeds = 30
networks**, averaged. Each network stops on its own fold, so the ensemble
collectively trains on all 19,716 training rows instead of the 17,744 a single
fit/dev cut leaves.

What is deliberately *not* copied from the paper:

- **Their split.** Folds here are cut along whole Hamming <= 3 peptide clusters,
  the same grouping as the frozen splits, so every held-out peptide stays >= 4
  substitutions from every training peptide. The paper groups by peptide
  identity, which is worth +0.018 and would make our test set meaningless.
- **Their evaluation.** They report performance on the same fold each network
  stopped on. Here the CV folds live entirely inside ``train``; validation is
  never seen during fitting.
- **Their target transform.** ``2^(-t0/th)`` scored 0.022-0.034 *below* log1p
  on this setup at every t0 they swept.

NetMHCstabpan itself cannot serve as a comparator at any stage: it was trained
on all 28,166 rows, including every peptide in our test split.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import eligible_alleles, score  # noqa: E402
from pepstab.features import ENCODINGS, build_features  # noqa: E402
from pepstab.mlp import MLPConfig, MLPRegressor  # noqa: E402
from pepstab.splits import assign_clusters  # noqa: E402
from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE, SEEDS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = REPO_ROOT / "preds"
REPORT_DIR = REPO_ROOT / "reports"

N_FOLDS = 5

#: Per encoding, the config stage 2 selected for the peptide+pseudosequence arm
#: and the peptide+domain arm. Fixed from ``reports/stage2_summary.csv`` rather
#: than re-tuned, so the ensemble is the *only* thing that changed.
SELECTED = {
    ("pep_pseudo", "onehot"): ((256, 64), 1e-5),
    ("pep_pseudo", "blosum"): ((256, 64), 1e-3),
    ("pep_domain", "onehot"): ((256, 64), 1e-3),
    ("pep_domain", "blosum"): ((256, 64), 1e-5),
    ("pep", "onehot"): ((64,), 1e-3),
    ("pep", "blosum"): ((64,), 1e-3),
}


def cv_folds(train: pd.DataFrame, n_folds: int = N_FOLDS) -> np.ndarray:
    """Assign each training row to one of ``n_folds``, whole clusters together.

    Same water-fill as the frozen splits, at equal fractions. Keeping clusters
    whole means a network never stops on a peptide within 3 substitutions of one
    it trained on -- the leak that would make the stopping epoch, and therefore
    the ensemble, quietly optimistic.
    """
    names = tuple(f"f{k}" for k in range(n_folds))
    placement = assign_clusters(train.groupby("cluster_id").size(),
                                fractions=(1.0 / n_folds,) * n_folds, names=names)
    return train["cluster_id"].map(placement).map(
        {name: k for k, name in enumerate(names)}).to_numpy()


def per_allele_scc(allele: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                   alleles: list[str]) -> tuple[float, float]:
    """Mean and median per-allele Spearman.

    Mean is carried because the paper aggregates that way; median is the
    project's primary metric (EVALUATION.md).
    """
    vals = [stats.spearmanr(y_pred[allele == a], y_true[allele == a]).statistic
            for a in alleles
            if len(np.unique(y_pred[allele == a])) > 1
            and len(np.unique(y_true[allele == a])) > 1]
    return float(np.mean(vals)), float(np.median(vals))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-set", default="pep_pseudo",
                    choices=["pep", "pep_pseudo", "pep_domain"])
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="which split to write predictions for (default: val)")
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    args = ap.parse_args()

    if args.split == "test":
        print("!! Writing TEST-split predictions. Per EVALUATION.md the test "
              "set is scored once, at stage 6.\n")

    df = load_with_splits()
    train = df[df.split == "train"]
    val = df[df.split == "val"].sort_values("pair_id").reset_index(drop=True)
    out = val if args.split == "val" else (
        df[df.split == args.split].sort_values("pair_id").reset_index(drop=True))

    fold = cv_folds(train, args.folds)
    alleles = eligible_alleles(val.allele, val.y_log1p, split="val")
    y_train = train.y_log1p.to_numpy()
    y_val = val.y_log1p.to_numpy()
    allele_arr = val.allele.to_numpy()

    n_members = args.folds * len(ENCODINGS) * len(SEEDS)
    print(f"input set: {args.input_set}")
    print(f"ensemble: {args.folds} CV folds x {len(ENCODINGS)} encodings x "
          f"{len(SEEDS)} seeds = {n_members} networks")
    print(f"train={len(train):,} rows, {train.cluster_id.nunique():,} peptide clusters")
    print(f"  fold sizes: {np.bincount(fold).tolist()}")
    print(f"  each network fits on ~{len(train) - len(train) // args.folds:,} rows; "
          f"the ensemble covers all {len(train):,}\n")

    features = {enc: (build_features(train, args.input_set, enc),
                      build_features(val, args.input_set, enc),
                      build_features(out, args.input_set, enc) if out is not val else None)
                for enc in ENCODINGS}

    started = time.perf_counter()
    val_members: list[np.ndarray] = []
    out_members: list[np.ndarray] = []
    rows = []
    for enc in ENCODINGS:
        hidden, l2 = SELECTED[(args.input_set, enc)]
        X_train, X_val, X_out = features[enc]
        for k in range(args.folds):
            is_fit = fold != k
            for seed in SEEDS:
                model = MLPRegressor(MLPConfig(hidden=hidden, l2=l2, seed=seed,
                                               max_epochs=MAX_EPOCHS,
                                               patience=PATIENCE))
                model.fit(X_train[is_fit], y_train[is_fit],
                          X_train[~is_fit], y_train[~is_fit])
                p = model.predict(X_val)
                val_members.append(p)
                out_members.append(p if X_out is None else model.predict(X_out))
                mean_scc, median_scc = per_allele_scc(allele_arr, y_val, p, alleles)
                rows.append({"input_set": args.input_set, "encoding": enc, "fold": k,
                             "seed": seed, "n_fit_rows": int(is_fit.sum()),
                             "best_epoch": model.best_epoch_,
                             "fit_seconds": round(model.fit_seconds_, 2),
                             "scc_mean": mean_scc, "scc_median": median_scc})
        done = [r for r in rows if r["encoding"] == enc]
        print(f"  {enc}: {len(done)} networks, "
              f"member mean SCC {np.mean([r['scc_mean'] for r in done]):.3f} "
              f"(min {min(r['scc_mean'] for r in done):.3f}, "
              f"max {max(r['scc_mean'] for r in done):.3f})")
    wall = time.perf_counter() - started

    members = pd.DataFrame(rows)
    ensemble_val = np.mean(val_members, axis=0)
    member_mean = members.scc_mean.mean()
    ens_mean, ens_median = per_allele_scc(allele_arr, y_val, ensemble_val, alleles)

    print(f"\naverage single member : mean SCC={member_mean:.3f}  "
          f"median SCC={members.scc_median.mean():.3f}")
    print(f"{n_members}-network ensemble : mean SCC={ens_mean:.3f}  "
          f"median SCC={ens_median:.3f}   "
          f"(ensembling gain {ens_mean - member_mean:+.3f} mean SCC)")

    s = score(f"seq_ensemble_{args.input_set}", val.allele, y_val, ensemble_val, alleles)
    print("\nproject metrics on validation (EVALUATION.md):")
    print(pd.DataFrame([s.as_row()]).to_string(index=False))
    print(f"\nNetMHCstabpan, 5-fold CV on all 28,166 rows (Rasmussen et al. "
          f"figure 1, t0=1 h): mean per-allotype SCC ~0.69.")
    print("  Not a comparator: it trained on every peptide in our test split.")

    PRED_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    dest = PRED_DIR / f"seq_ensemble_{args.input_set}.csv"
    pd.DataFrame({"pair_id": out.pair_id.to_numpy(),
                  "y_pred": np.mean(out_members, axis=0)}).to_csv(dest, index=False)
    members.to_csv(REPORT_DIR / f"stage2_ensemble_{args.input_set}.csv", index=False)
    print(f"\nwrote {dest.relative_to(REPO_ROOT)} and "
          f"reports/stage2_ensemble_{args.input_set}.csv")
    print(f"total fit wall time: {wall / 60:.1f} min "
          f"({wall / n_members:.1f} s per network)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
