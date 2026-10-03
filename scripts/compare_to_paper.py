"""Calibrate the stage 2 sequence baseline against NetMHCstabpan's published score.

    .venv/bin/python scripts/compare_to_paper.py     # ~3 min, CPU

Rasmussen et al. (2016) figure 1 reports, for the released configuration
(t0 = 1 h), **average per-allotype SCC ~0.69 and PCC 0.676**, from 5-fold
cross-validation. Our ensemble baseline reaches mean per-allele SCC 0.645 on
the frozen validation split.

**This is a calibration, not a reproduction.** Our model differs from theirs on
every axis listed in :data:`METHOD_DIFFERENCES` -- most importantly, they train
on 103,166 rows (28,166 measured plus 1,000 assumed-zero weak binders per
allele, 73% of their training set) against our 19,716 measured training rows.
The two numbers are not a model-quality comparison and no claim of method parity
is supported.

This script measures the three differences that *are* testable here, each with a
controlled design:

1. **Split grouping** -- the paper groups by peptide identity, so a held-out
   peptide may sit 1 substitution from a training peptide; we hold out whole
   Hamming <= 3 clusters. Measured on a **common evaluation set** carved from
   the frozen training split, varying only which training rows are available.
2. **Ensembling** -- they fit one network per fold per architecture and predict
   with the ensemble; stage 2 reported single networks.
3. **Target transform** -- they train on ``s = 2^(-t0/th)``; we train on
   ``log1p``.

Everything here runs inside the frozen **train** split. See EVALUATION.md
"Disclosed test exposure" for the earlier version of this script, which
re-partitioned the whole dataset and did consume frozen test rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import (  # noqa: E402
    describe_delta,
    eligible_alleles,
    paired_cluster_bootstrap,
)
from pepstab.features import build_features  # noqa: E402
from pepstab.mlp import MLPConfig, MLPRegressor  # noqa: E402
from pepstab.splits import assign_clusters  # noqa: E402
from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE, SEEDS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

INPUT_SET, ENCODING, HIDDEN, L2 = "pep_pseudo", "onehot", (256, 64), 1e-5

#: Rasmussen et al. figure 1, t0 = 1 h. PCC is stated in the text; SCC is read
#: off the figure, so it is approximate.
PAPER_SCC, PAPER_PCC = 0.69, 0.676

T0_GRID = (0.5, 1.0, 2.0)

#: Share of training *peptides* held out as the common evaluation set.
EVAL_PEPTIDE_FRACTION = 0.15
DEV_FRACTION = 0.10

#: Every way our model differs from the one that produced 0.69. Printed with
#: the results so the gap is never read as a model-quality difference.
METHOD_DIFFERENCES = [
    ("Training rows", "103,166 (28,166 measured + 75,000 assumed-zero weak "
                      "binders, 1,000 per allele)", "19,716 measured"),
    ("Encoding A", "BLOSUM50 / 5", "BLOSUM62 / 5"),
    ("Encoding B", "smoothed sparse (0.9 / 0.05)", "plain one-hot (1 / 0)"),
    ("Architecture", "one hidden layer, 40 / 50 / 60 units", "two layers, 256x64"),
    ("Ensemble diversity", "2 encodings x 3 hidden sizes x 5 folds",
     "2 encodings x 5 folds x 3 seeds"),
    ("Target", "s = 2^(-t0/th), t0 tuned over {0.5, 1, 2, 5}", "log1p(th)"),
    ("Held-out grouping", "peptide identity", "Hamming <= 3 cluster"),
    ("Scored on", "the same 1/5 fold used for early stopping",
     "a split never seen during fitting"),
]


def paper_scale(values_log1p: np.ndarray, t0: float = 1.0) -> np.ndarray:
    """Map ``log1p`` half-life onto the paper's target ``s = 2^(-t0/th)``.

    Applied to **both** labels and predictions before computing PCC. Pearson
    correlation is not invariant under a nonlinear transform, so correlating
    log1p predictions against paper-scale labels would be neither metric --
    it reads 0.639 where the paper-scale PCC is 0.649.

    Negative predictions on the log1p scale mean a predicted half-life below
    zero; they clamp to ``s = 0``, the limit as th -> 0.
    """
    hours = np.expm1(np.asarray(values_log1p, dtype=float))
    positive = hours > 0
    return np.where(positive, 2.0 ** (-t0 / np.where(positive, hours, 1.0)), 0.0)


#: The scales a model's predictions can already be on. A model trained on the
#: paper's target emits paper-scale scores directly; one trained on log1p emits
#: log1p half-life. Getting this wrong transforms paper-scale predictions a
#: second time and silently inflates PCC -- it read 0.611 for the t0=1 ensemble
#: where the true paper-scale value is 0.590.
PRED_SCALES = ("log1p", "paper")


def per_allele(allele: np.ndarray, y_log1p: np.ndarray, y_pred: np.ndarray,
               alleles: list[str], pred_scale: str = "log1p") -> dict:
    """Mean and median per-allele SCC, plus PCC with both sides on one scale.

    ``pred_scale`` says what ``y_pred`` already is, so predictions are converted
    at most once. SCC is unaffected either way -- it is rank-based and both
    scales are monotone in half-life -- which is exactly why SCC is the metric
    to compare against the paper on.

    ``pcc_log1p_mean`` is NaN for paper-scale predictions: inverting
    ``s = 2^(-t0/th)`` is undefined at ``s = 0``, where 20.2% of the labels sit.
    """
    if pred_scale not in PRED_SCALES:
        raise ValueError(f"pred_scale must be one of {PRED_SCALES}, got {pred_scale!r}")
    true_paper = paper_scale(y_log1p)
    pred_paper = paper_scale(y_pred) if pred_scale == "log1p" else np.asarray(y_pred, float)

    scc, pcc_log, pcc_paper = [], [], []
    for a in alleles:
        m = allele == a
        if len(np.unique(y_pred[m])) < 2 or len(np.unique(y_log1p[m])) < 2:
            continue
        scc.append(stats.spearmanr(y_pred[m], y_log1p[m]).statistic)
        if pred_scale == "log1p":
            pcc_log.append(stats.pearsonr(y_pred[m], y_log1p[m]).statistic)
        if len(np.unique(pred_paper[m])) > 1 and len(np.unique(true_paper[m])) > 1:
            pcc_paper.append(stats.pearsonr(pred_paper[m], true_paper[m]).statistic)
    return {"n_alleles": len(scc), "scc_mean": float(np.mean(scc)),
            "scc_median": float(np.median(scc)),
            "pcc_log1p_mean": float(np.mean(pcc_log)) if pcc_log else float("nan"),
            "pcc_paper_mean": float(np.mean(pcc_paper)),
            "pred_scale": pred_scale}


def fit_predict(train: pd.DataFrame, evalset: pd.DataFrame, t0: float | None = None,
                seeds=SEEDS) -> list[np.ndarray]:
    """Fit one network per seed on ``train``, predict ``evalset``.

    The stopping fold is cut from ``train`` along whole peptide clusters, so no
    network stops on a near-duplicate of a row it fitted.
    """
    placement = assign_clusters(train.groupby("cluster_id").size(),
                                fractions=(1 - DEV_FRACTION, DEV_FRACTION),
                                names=("fit", "dev"))
    is_fit = (train["cluster_id"].map(placement) == "fit").to_numpy()
    X = build_features(train, INPUT_SET, ENCODING)
    X_eval = build_features(evalset, INPUT_SET, ENCODING)
    y = train.y_log1p.to_numpy()
    target = y if t0 is None else paper_scale(y, t0)

    out = []
    for seed in seeds:
        model = MLPRegressor(MLPConfig(hidden=HIDDEN, l2=L2, seed=seed,
                                       max_epochs=MAX_EPOCHS, patience=PATIENCE))
        model.fit(X[is_fit], target[is_fit], X[~is_fit], target[~is_fit])
        out.append(model.predict(X_eval))
    return out


def grouping_experiment(train: pd.DataFrame) -> pd.DataFrame:
    """Isolate the grouping effect on a **common** evaluation set.

    The earlier version of this script re-partitioned the whole dataset by
    peptide identity and compared the resulting score against the frozen one.
    That changed the training rows, the stopping rows *and* the scored rows at
    once -- only 310 of 2,817 validation rows survived into it -- so the
    difference could not be attributed to grouping. It also moved 3,798 frozen
    test rows into fitting and scoring.

    Here the evaluation set E is fixed, carved from the frozen training split,
    and only the training rows change:

    - **cluster-grouped**: train rows whose peptide's cluster contains no E
      peptide, so every training peptide is >= 4 substitutions from every E
      peptide -- the frozen split's rule.
    - **identity-grouped**: train rows whose *peptide* is not in E. E's
      cluster-mates stay in training, so E peptides can sit 1-3 substitutions
      from a training peptide -- the paper's rule.
    - **identity-grouped, size-matched**: every near-neighbour row, plus enough
      randomly chosen far rows to match the cluster-grouped arm exactly. The
      identity arm is necessarily larger than the cluster arm -- the extra rows
      *are* the neighbours -- and there is no reservoir of unused far rows to
      pad the cluster arm with, so matching has to happen by dropping far rows
      from the identity arm rather than adding to the cluster one.

    ``identity size-matched - cluster-grouped`` is then the grouping effect at
    equal row count: the same number of training rows, with near neighbours
    standing in for unrelated peptides.
    """
    peptides = pd.Index(sorted(train["peptide"].unique()))
    weights = pd.Series(1, index=np.arange(len(peptides)))
    placement = assign_clusters(weights, fractions=(1 - EVAL_PEPTIDE_FRACTION,
                                                   EVAL_PEPTIDE_FRACTION),
                               names=("pool", "eval"))
    eval_peptides = set(peptides[placement.to_numpy() == "eval"])

    in_eval = train["peptide"].isin(eval_peptides).to_numpy()
    evalset = train[in_eval]
    eval_clusters = set(evalset["cluster_id"].unique())

    identity = train[~in_eval]
    cluster = train[~train["cluster_id"].isin(eval_clusters).to_numpy()]
    # The rows identity grouping keeps and cluster grouping drops: E's
    # cluster-mates, i.e. training peptides within 3 substitutions of an
    # evaluation peptide.
    neighbours = identity.index.difference(cluster.index)
    far = identity.index.intersection(cluster.index)
    rng = np.random.default_rng(20261003)
    keep_far = rng.choice(far.to_numpy(), size=len(cluster) - len(neighbours),
                          replace=False)
    matched = identity.loc[np.concatenate([neighbours.to_numpy(), keep_far])]

    y_eval = evalset.y_log1p.to_numpy()
    alleles = eligible_alleles(evalset.allele, y_eval, min_rows=20)
    allele_arr = evalset.allele.to_numpy()
    nearest = evalset["peptide"].map(_nearest_distance(evalset, identity))

    print(f"\ncommon evaluation set carved from train: {len(evalset):,} rows, "
          f"{len(eval_peptides):,} peptides, {len(alleles)} eligible alleles")
    print(f"  nearest training peptide under identity grouping: "
          f"min {int(nearest.min())}, "
          f"{float((nearest <= 3).mean()):.1%} of rows within 3 substitutions")
    print(f"  near-neighbour rows identity grouping keeps and cluster grouping "
          f"drops: {len(neighbours):,}")
    print(f"  training rows -- cluster-grouped {len(cluster):,}, "
          f"identity size-matched {len(matched):,}, "
          f"identity full {len(identity):,}")

    rows, preds = [], {}
    for label, frame in [("cluster-grouped (our rule)", cluster),
                         ("identity-grouped, size-matched", matched),
                         ("identity-grouped, full", identity)]:
        members = fit_predict(frame, evalset)
        preds[label] = np.mean(members, axis=0)
        singles = [per_allele(allele_arr, y_eval, p, alleles) for p in members]
        rows.append({"arm": label, "n_train_rows": len(frame),
                     **{k: float(np.mean([s[k] for s in singles]))
                        for k in singles[0] if k != "pred_scale"}})
        print(f"  {label:32s} mean SCC={rows[-1]['scc_mean']:.3f}  "
              f"median SCC={rows[-1]['scc_median']:.3f}")

    base = "cluster-grouped (our rule)"
    print(f"\n  grouping effect at equal row count (identity size-matched minus "
          f"cluster-grouped),\n  paired cluster bootstrap on the common "
          f"evaluation set:")
    # Both statistics, because the paper's published figure is a *mean* over
    # allotypes while our contract metric is the median -- a median interval
    # does not bound a mean effect, so quoting only one would overreach.
    for statistic in ("median", "mean"):
        boot = paired_cluster_bootstrap(
            evalset.cluster_id, evalset.allele, y_eval,
            preds[base], preds["identity-grouped, size-matched"], alleles,
            statistic=statistic)
        lo, hi = boot["ci95"]
        rows.append({"arm": f"grouping effect ({statistic} per-allele rho)",
                     "n_train_rows": len(cluster),
                     "delta": boot["delta_median_spearman"], "ci_low": lo, "ci_high": hi})
        print(f"    delta {statistic:6s} per-allele rho = "
              f"{boot['delta_median_spearman']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              + (f"   <- contract metric" if statistic == "median" else
                 "   <- comparable to the paper's aggregation"))
    print("    An inconclusive interval is not evidence of no effect: it still "
          "admits\n    a modest positive grouping advantage.")
    return pd.DataFrame(rows)


def _nearest_distance(evalset: pd.DataFrame, training: pd.DataFrame) -> pd.Series:
    """Hamming distance from each evaluation peptide to its nearest training one."""
    from pepstab.data import PEPTIDE_LENGTH, encode_sequences
    ev = sorted(evalset["peptide"].unique())
    tr = encode_sequences(sorted(training["peptide"].unique()), PEPTIDE_LENGTH)
    codes = encode_sequences(ev, PEPTIDE_LENGTH)
    out = np.empty(len(codes), dtype=int)
    for start in range(0, len(codes), 256):
        block = codes[start:start + 256]
        out[start:start + len(block)] = (
            block[:, None, :] != tr[None, :, :]).sum(axis=2).min(axis=1)
    return pd.Series(out, index=pd.Index(ev, name="peptide"))


def main() -> int:
    df = load_with_splits()
    train = df[df.split == "train"]
    val = df[df.split == "val"].sort_values("pair_id").reset_index(drop=True)

    print("CALIBRATION, NOT A REPRODUCTION. How our model differs from the one "
          "that scored 0.69:\n")
    width = max(len(a) for a, _, _ in METHOD_DIFFERENCES)
    for axis, theirs, ours in METHOD_DIFFERENCES:
        print(f"  {axis:<{width}}  paper: {theirs}")
        print(f"  {'':<{width}}  ours : {ours}")
    print(f"\nTheir training set is {103166 / len(train):.1f}x ours and 73% "
          "assumed-zero rows we do not have.")
    print(f"\narm={ENCODING}_{INPUT_SET} hidden={HIDDEN} l2={L2:g} seeds={list(SEEDS)}")

    print("\n" + "=" * 72)
    print("1. SPLIT GROUPING (inside the frozen train split; test untouched)")
    print("=" * 72)
    grouping = grouping_experiment(train)

    print("\n" + "=" * 72)
    print("2. ENSEMBLING and 3. TARGET TRANSFORM (train -> frozen validation)")
    print("=" * 72)
    y_val = val.y_log1p.to_numpy()
    alleles = eligible_alleles(val.allele, y_val, split="val")
    allele_arr = val.allele.to_numpy()

    rows = []
    for label, t0 in [("log1p target", None)] + [
            (f"2^(-{t:g}/th) target", t) for t in T0_GRID]:
        members = fit_predict(train, val, t0=t0)
        # A model trained on the paper's target already emits paper-scale
        # scores; only a log1p-trained model needs converting.
        scale = "log1p" if t0 is None else "paper"
        singles = [per_allele(allele_arr, y_val, p, alleles, scale) for p in members]
        ens = per_allele(allele_arr, y_val, np.mean(members, axis=0), alleles, scale)
        rows.append({"setting": label, "model": "single network (seed mean)",
                     "pred_scale": scale,
                     **{k: float(np.mean([s[k] for s in singles]))
                        for k in singles[0] if k != "pred_scale"}})
        rows.append({"setting": label, "model": f"{len(SEEDS)}-seed ensemble", **ens})
        print(f"  {label:22s} single={rows[-2]['scc_mean']:.3f}  "
              f"{len(SEEDS)}-seed ensemble={ens['scc_mean']:.3f}  "
              f"(PCC paper scale {ens['pcc_paper_mean']:.3f})")

    out = pd.DataFrame(rows)
    out.to_csv(REPO_ROOT / "reports" / "compare_to_paper.csv", index=False)
    grouping.to_csv(REPO_ROOT / "reports" / "compare_to_paper_grouping.csv", index=False)

    print(f"\nNetMHCstabpan (Rasmussen et al. figure 1, t0=1 h): "
          f"mean per-allotype SCC ~{PAPER_SCC}, PCC {PAPER_PCC}.")
    print("Not a comparator at any stage: it trained on all 28,166 rows, "
          "including every peptide in our test split.")
    print("\nwrote reports/compare_to_paper.csv and "
          "reports/compare_to_paper_grouping.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
