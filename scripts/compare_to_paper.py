"""Why our stage 2 baseline scores below NetMHCstabpan, factor by factor.

    .venv/bin/python scripts/compare_to_paper.py     # ~3 min, CPU

Rasmussen et al. (2016) figure 1 reports, for the released NetMHCstabpan
configuration (global rescaling t0 = 1 h), **average per-allotype SCC ~0.69 and
PCC 0.676**, from 5-fold cross-validation on this same 28,166-row dataset. Our
stage 2 headline arm reaches mean per-allele SCC 0.573 on the frozen validation
split -- about 0.12 lower.

That difference is not a like-for-like model comparison, because three things
differ besides the model. This script isolates the ones we can test, changing
one at a time and holding the arm, config, metric and allele panel fixed:

1. **Split grouping.** The paper groups "all peptide-HLA-I stability data for a
   given peptide" into one CV group -- peptide *identity*, so a held-out peptide
   may sit one substitution from a training peptide. Our frozen splits move
   whole Hamming <= 3 clusters, putting every held-out peptide >= 4
   substitutions away.
2. **Ensembling.** NetMHC-family training fits a network per CV fold per
   architecture (2 encodings x 3 hidden sizes x 5 folds) and predicts with the
   ensemble. Stage 2 reports single networks.
3. **Target transform.** The paper trains on ``s = 2^(-t0/th)`` and tuned t0
   over {0.5, 1, 2, 5}, which moved PCC from 0.633 to 0.676. We train on
   ``log1p`` and never tuned it.

What this cannot isolate: they train on 4/5 of the data (~22.5k rows) against
our 17.7k, and their reported score is measured on the same fold used for early
stopping, which is optimistic by an unknown amount.

Nothing here touches ``data/splits.csv`` or the stage 2 results. The frozen
split remains the only one any reported result is scored on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.data import PEPTIDE_LENGTH, encode_sequences, load_with_splits  # noqa: E402
from pepstab.evaluation import eligible_alleles  # noqa: E402
from pepstab.features import build_features  # noqa: E402
from pepstab.mlp import MLPConfig, MLPRegressor  # noqa: E402
from pepstab.splits import SPLIT_FRACTIONS, SPLIT_NAMES, assign_clusters  # noqa: E402
from scripts.baseline_sequence import (  # noqa: E402
    DEV_FRACTION,
    MAX_EPOCHS,
    PATIENCE,
    SEEDS,
    inner_folds,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The stage 2 headline arm and config, held fixed throughout.
INPUT_SET, ENCODING, HIDDEN, L2 = "pep_pseudo", "onehot", (256, 64), 1e-5

#: Rasmussen et al. figure 1, t0 = 1 h. PCC is stated in the text; SCC is read
#: off the figure, so it is quoted as approximate.
PAPER_SCC, PAPER_PCC = 0.69, 0.676

#: t0 values the paper swept for its target transform.
T0_GRID = (0.5, 1.0, 2.0)


def paper_scale(y_log1p: np.ndarray, t0: float) -> np.ndarray:
    """The paper's target: ``s = 2^(-t0/th)``, mapping half-life into (0, 1].

    Monotone in half-life, so it leaves SCC unchanged when applied to *labels*;
    it matters only as the thing a model is trained to fit. A half-life of 0
    maps to 0, which is the limit as th -> 0.
    """
    hours = np.expm1(y_log1p)
    safe = np.where(hours > 0, hours, 1.0)
    return np.where(hours > 0, 2.0 ** (-t0 / safe), 0.0)


def identity_grouped_split(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """70/10/20 grouped by peptide identity alone, as the paper describes.

    Same water-fill as the frozen split, with each distinct peptide its own
    group instead of each Hamming <= 3 cluster. Returns ``(split, group)``.
    """
    group = pd.Series(pd.factorize(df["peptide"])[0], index=df.index, name="group")
    placement = assign_clusters(group.groupby(group).size(),
                                fractions=SPLIT_FRACTIONS, names=SPLIT_NAMES)
    return group.map(placement), group


def min_distance_to_train(df: pd.DataFrame, split: pd.Series, held: str) -> int:
    """Nearest-training-peptide Hamming distance for a held-out split."""
    train_codes = encode_sequences(
        sorted(df.loc[(split == "train").to_numpy(), "peptide"].unique()), PEPTIDE_LENGTH)
    held_codes = encode_sequences(
        sorted(df.loc[(split == held).to_numpy(), "peptide"].unique()), PEPTIDE_LENGTH)
    best = PEPTIDE_LENGTH
    for start in range(0, len(held_codes), 256):
        block = held_codes[start:start + 256]
        best = min(best, int((block[:, None, :] != train_codes[None, :, :])
                             .sum(axis=2).min()))
    return best


def per_allele(allele: np.ndarray, y_log1p: np.ndarray, y_pred: np.ndarray,
               alleles: list[str]) -> dict:
    """Mean and median per-allele SCC, plus PCC on log1p and the paper's scale.

    SCC is the comparable one: it is invariant to the target transform, so it
    does not reward either side's choice of scale. The paper reports a *mean*
    over allotypes, which is why mean is carried alongside our usual median.
    """
    s_paper = paper_scale(y_log1p, 1.0)
    scc, pcc_log, pcc_paper = [], [], []
    for a in alleles:
        m = allele == a
        if len(np.unique(y_pred[m])) < 2 or len(np.unique(y_log1p[m])) < 2:
            continue
        scc.append(stats.spearmanr(y_pred[m], y_log1p[m]).statistic)
        pcc_log.append(stats.pearsonr(y_pred[m], y_log1p[m]).statistic)
        pcc_paper.append(stats.pearsonr(y_pred[m], s_paper[m]).statistic)
    return {"n_alleles": len(scc), "scc_mean": float(np.mean(scc)),
            "scc_median": float(np.median(scc)),
            "pcc_log1p_mean": float(np.mean(pcc_log)),
            "pcc_paper_mean": float(np.mean(pcc_paper))}


def fit_seeds(X_fit, y_fit, X_dev, y_dev, X_val) -> list[np.ndarray]:
    """Validation predictions from one network per seed."""
    out = []
    for seed in SEEDS:
        model = MLPRegressor(MLPConfig(hidden=HIDDEN, l2=L2, seed=seed,
                                       max_epochs=MAX_EPOCHS, patience=PATIENCE))
        model.fit(X_fit, y_fit, X_dev, y_dev)
        out.append(model.predict(X_val))
    return out


def run(df: pd.DataFrame, split: pd.Series, group: pd.Series, label: str,
        t0: float | None = None) -> list[dict]:
    """Fit on ``split``'s train, score its validation, single and ensembled.

    ``t0`` trains on the paper's transform instead of ``log1p``. Scoring is
    always against the ``log1p`` labels, so every row of the output table is
    measured the same way whatever the model was trained on.
    """
    is_train = (split == "train").to_numpy()
    train, val = df[is_train], df[(split == "val").to_numpy()]
    placement = assign_clusters(group[is_train].groupby(group[is_train]).size(),
                                fractions=(1 - DEV_FRACTION, DEV_FRACTION),
                                names=("fit", "dev"))
    is_fit = (group[is_train].map(placement) == "fit").to_numpy()

    X_train = build_features(train, INPUT_SET, ENCODING)
    X_val = build_features(val, INPUT_SET, ENCODING)
    y_train = train.y_log1p.to_numpy()
    y_val = val.y_log1p.to_numpy()
    target = y_train if t0 is None else paper_scale(y_train, t0)
    alleles = eligible_alleles(val.allele, y_val, split="val")
    allele_arr = val.allele.to_numpy()

    preds = fit_seeds(X_train[is_fit], target[is_fit],
                      X_train[~is_fit], target[~is_fit], X_val)
    singles = [per_allele(allele_arr, y_val, p, alleles) for p in preds]
    ensemble = per_allele(allele_arr, y_val, np.mean(preds, axis=0), alleles)

    rows = [{"setting": label, "model": f"single seed (mean of {len(SEEDS)})",
             **{k: float(np.mean([s[k] for s in singles])) for k in singles[0]}},
            {"setting": label, "model": f"{len(SEEDS)}-seed ensemble", **ensemble}]
    print(f"\n{label}")
    print(f"  train={len(train):,} val={len(val):,} alleles={len(alleles)}")
    for r in rows:
        print(f"  {r['model']:24s} mean SCC={r['scc_mean']:.3f}  "
              f"median SCC={r['scc_median']:.3f}  "
              f"mean PCC(paper scale)={r['pcc_paper_mean']:.3f}")
    return rows


def main() -> int:
    df = load_with_splits()
    frozen_split, frozen_group = df["split"], df["cluster_id"]
    identity_split, identity_group = identity_grouped_split(df)

    print(f"arm={ENCODING}_{INPUT_SET} hidden={HIDDEN} l2={L2:g} seeds={list(SEEDS)}")
    print("Scored against log1p labels throughout, on the same allele panel.")
    print(f"min val-to-train peptide distance: "
          f"frozen={min_distance_to_train(df, frozen_split, 'val')}, "
          f"identity-grouped={min_distance_to_train(df, identity_split, 'val')}")

    rows = []
    rows += run(df, frozen_split, frozen_group, "frozen split, log1p target")
    rows += run(df, identity_split, identity_group,
                "identity-grouped split, log1p target")
    for t0 in T0_GRID:
        rows += run(df, frozen_split, frozen_group,
                    f"frozen split, 2^(-{t0:g}/th) target", t0=t0)

    out = pd.DataFrame(rows)
    path = REPO_ROOT / "reports" / "compare_to_paper.csv"
    out.to_csv(path, index=False)

    base = out[(out.setting == "frozen split, log1p target")
               & out.model.str.startswith("single")].iloc[0]
    print("\n--- factor by factor, against the stage 2 baseline "
          f"(mean SCC {base.scc_mean:.3f}) ---")
    for _, r in out.iterrows():
        if r.setting == base.setting and r.model == base.model:
            continue
        print(f"  {r.setting:38s} {r.model:24s} "
              f"{r.scc_mean - base.scc_mean:+.3f}")
    print(f"\nNetMHCstabpan (Rasmussen et al. figure 1, t0=1 h): "
          f"mean per-allotype SCC ~{PAPER_SCC}, PCC {PAPER_PCC}.")
    print(f"Unexplained after split and seed ensembling: "
          f"~{PAPER_SCC - out.scc_mean.max():.2f} SCC.")
    print(f"\nwrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
