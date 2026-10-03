"""Constant reference baselines: training global mean and training allele mean.

    python scripts/baseline_constant.py --split val

Writes ``preds/global_mean.csv`` and ``preds/allele_mean.csv``. These are the
floor for MAE and pooled metrics -- any real model must beat them. They are
deliberately constant within an allele, so they carry no within-allele ranking
information: every per-allele Spearman is undefined and the panel median is
exactly 0, chance level. That is the point, and it makes them the reference the
primary metric is measured against (see EVALUATION.md, "Alleles the model cannot
rank").

Allele means are taken from the training split only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.data import load_with_splits  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "preds"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()

    df = load_with_splits()
    train = df[df.split == "train"]
    target = df[df.split == args.split].sort_values("pair_id")

    OUT_DIR.mkdir(exist_ok=True)
    global_mean = float(train.y_log1p.mean())
    allele_mean = train.groupby("allele").y_log1p.mean()

    # Alleles absent from train fall back to the global mean. With the current
    # splits every allele is present, but a model must not crash if that changes.
    unseen = sorted(set(target.allele) - set(allele_mean.index))
    if unseen:
        print(f"warning: {len(unseen)} allele(s) absent from train, using the "
              f"global mean: {', '.join(unseen)}")

    for name, pred in [
        ("global_mean", pd.Series(global_mean, index=target.index)),
        ("allele_mean", target.allele.map(allele_mean).fillna(global_mean)),
    ]:
        path = OUT_DIR / f"{name}.csv"
        pd.DataFrame({"pair_id": target.pair_id.to_numpy(),
                      "y_pred": pred.to_numpy()}).to_csv(path, index=False)
        print(f"wrote {path.relative_to(OUT_DIR.parent)} ({len(target):,} rows, "
              f"split={args.split})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
