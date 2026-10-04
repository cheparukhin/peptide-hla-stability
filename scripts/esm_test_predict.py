"""Refit the frozen stage-3 ESM arms and write their held-out-split predictions.

    .venv/bin/python scripts/esm_test_predict.py --arm esm_ensemble

Stage 3 selected every configuration on validation and persisted none of the
fitted networks, so the single stage-6 scoring pass needs each arm refitted and
asked for its held-out predictions. That is all this script does.

**Why this is a separate file from ``scripts/esm_arm.py``.**
``tests/test_esm.py::test_no_test_rows_are_ever_loaded`` greps that script's
source and asserts the literal ``"test"`` never appears in it. That guard is
the evidence stage 3's *selection* never saw the held-out split, and it stays
true only if the arm script is left alone. So the held-out-prediction path lives
here instead, where it is a small, auditable file that no selection ever ran
through.

**What this script may and may not do.** It PREDICTS on the held-out split and
writes those predictions to ``preds/test/``. It does **not** score them: no
Spearman, no MAE, nothing. The held-out labels are never read. Scoring happens
exactly once, in ``scripts/stage6_report.py``, per
``reports/TEST_SCORING_RUNBOOK.md``.

**Correctness check that does not touch the held-out split.** Every refit also
predicts validation and scores *that*, so each arm must reproduce the median
per-allele Spearman frozen in ``reports/stage3_headline.json``. The expected
value is declared per arm below and checked automatically; a mismatch beyond
``--tolerance`` is a hard failure, because it means the refit is not the frozen
arm and its held-out predictions mean nothing.

**Configurations are read off the frozen record, not re-selected.** Each arm's
per-group ``(hidden, l2)`` is transcribed from the ``ensemble_member`` rows of
``reports/stage3_runs.csv`` -- the 150M arm genuinely differs by layer
(mid L2=0.1, final L2=0.01), which is why the table is explicit rather than
defaulted. No grid is re-run.

**Feature transforms are fitted exactly as stage 3 fitted them**: once on the
whole training split, outside the fold loop, as ``assemble`` does. This is what
produced the frozen validation numbers, and it is what the reproduction check
above verifies. Fitting the basis per fold instead would be a *different* arm
that does not reproduce them. The transform never sees validation or held-out
rows either way, which is the property that matters for the reported metric;
the inner CV folds only pick early-stopping cuts and spread the ensemble.

Pin BLAS to one thread per process and run arms as separate processes. Thread
count changes float32 reduction order and moves a single network by ~0.02;
one thread per process keeps the member mean summing in canonical order so the
reproduction check is exact.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import eligible_alleles, score  # noqa: E402
from pepstab.features import ENCODINGS  # noqa: E402
from pepstab.mlp import MLPConfig, MLPRegressor  # noqa: E402
from scripts.baseline_ensemble import N_FOLDS, cv_folds  # noqa: E402
from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE, SEEDS  # noqa: E402
from scripts.esm_arm import PEP_PCA, assemble, rep_label  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "preds" / "test"

#: The three stage-3 arms that have no held-out prediction file yet.
#:
#: ``groups`` is the two-way axis the 30 members spread over, as
#: ``(key, layer, raw)`` triples matching ``scripts/esm_arm.py``'s own
#: construction: ``axis=layer`` varies the ESM layer read, ``axis=encoding``
#: varies the raw sequence encoding appended to a fixed layer.
#:
#: ``expected_val_rho`` is the frozen validation median per-allele Spearman
#: from ``reports/stage3_headline.json`` / ``reports/stage3_runs.csv``.
ARMS: dict[str, dict] = {
    "esm_ensemble": {
        "checkpoint": "esm2_t12_35M_UR50D",
        "pep_rep": "pos",
        "hla_rep": "contact",
        "groups": [("mid", "mid", None), ("final", "final", None)],
        "config": {"mid": ((256, 64), 1e-2), "final": ((256, 64), 1e-2)},
        "expected_val_rho": 0.6830,
    },
    "esm_plus_seq_ensemble": {
        "checkpoint": "esm2_t12_35M_UR50D",
        "pep_rep": "pos",
        "hla_rep": "contact",
        "groups": [(enc, "mid", enc) for enc in ENCODINGS],
        "config": {enc: ((256, 64), 1e-2) for enc in ENCODINGS},
        "expected_val_rho": 0.6761,
    },
    "esm_ensemble_150m": {
        "checkpoint": "esm2_t30_150M_UR50D",
        "pep_rep": "pos",
        "hla_rep": "contact",
        "groups": [("mid", "mid", None), ("final", "final", None)],
        "config": {"mid": ((256, 64), 1e-1), "final": ((256, 64), 1e-2)},
        "expected_val_rho": 0.6737,
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--split", default="test", choices=["test"],
                    help="held-out split to predict (predictions only, never scored)")
    ap.add_argument("--tolerance", type=float, default=0.002,
                    help="max |refit - frozen| validation rho before failing")
    args = ap.parse_args()
    spec = ARMS[args.arm]

    df = load_with_splits()
    train = df[df.split == "train"]
    val = df[df.split == "val"].sort_values("pair_id").reset_index(drop=True)
    out = (df[df.split == args.split]
           .sort_values("pair_id").reset_index(drop=True))
    alleles = eligible_alleles(val.allele, val.y_log1p, split="val")

    y_train = train.y_log1p.to_numpy()
    y_val = val.y_log1p.to_numpy()

    n_members = N_FOLDS * len(spec["groups"]) * len(SEEDS)
    print(f"arm {args.arm}  checkpoint {spec['checkpoint']}")
    print(f"ensemble: {N_FOLDS} folds x {len(spec['groups'])} groups x "
          f"{len(SEEDS)} seeds = {n_members} networks")
    print(f"train={len(train):,}  val={len(val):,}  "
          f"{args.split}={len(out):,} rows")
    print(f"reproduction target: validation median rho "
          f"{spec['expected_val_rho']:.4f} +/- {args.tolerance}\n")

    fold = cv_folds(train, N_FOLDS)       # imported, never reimplemented
    print(f"fold sizes: {np.bincount(fold).tolist()}\n")

    val_members: list[np.ndarray] = []
    out_members: list[np.ndarray] = []
    started = time.perf_counter()

    for key, layer, raw in spec["groups"]:
        hidden, l2 = spec["config"][key]
        X_train, (X_val, X_out), info = assemble(
            train, [val, out], spec["checkpoint"], layer,
            spec["pep_rep"], spec["hla_rep"], raw, pep_pca=PEP_PCA)
        label = rep_label(spec["pep_rep"], spec["hla_rep"], layer, raw)
        print(f"{label}: {info['n_features']:,} features, "
              f"config h{hidden} l2={l2:g}")
        for k in range(N_FOLDS):
            is_fit = fold != k
            for seed in SEEDS:
                cfg = MLPConfig(hidden=hidden, l2=l2, seed=seed,
                                max_epochs=MAX_EPOCHS, patience=PATIENCE)
                model = MLPRegressor(cfg).fit(
                    X_train[is_fit], y_train[is_fit],
                    X_train[~is_fit], y_train[~is_fit])
                val_members.append(model.predict(X_val))
                out_members.append(model.predict(X_out))
        print(f"  {N_FOLDS * len(SEEDS)} networks fitted")
        del X_train, X_val, X_out
        gc.collect()

    wall = time.perf_counter() - started

    # --- reproduction check, on VALIDATION only -----------------------------
    ens_val = np.mean(val_members, axis=0)
    s = score(args.arm, val.allele, y_val, ens_val, alleles)
    delta = s.median_spearman - spec["expected_val_rho"]
    print(f"\nvalidation median rho {s.median_spearman:.4f} "
          f"(frozen {spec['expected_val_rho']:.4f}, delta {delta:+.4f})")

    if abs(delta) > args.tolerance:
        print(f"\n!! REPRODUCTION FAILED: |{delta:+.4f}| > {args.tolerance}. "
              f"This refit is not the frozen arm; no predictions written.",
              file=sys.stderr)
        return 1
    print("reproduction OK")

    # --- held-out predictions: written, never scored ------------------------
    ens_out = np.mean(out_members, axis=0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = OUT_DIR / f"{args.arm}.csv"
    pd.DataFrame({"pair_id": out.pair_id.to_numpy(),
                  "y_pred": ens_out}).to_csv(dest, index=False)
    print(f"wrote {dest.relative_to(REPO_ROOT)} "
          f"({len(out):,} rows, unscored)")
    print(f"total fit wall time: {wall / 60:.1f} min "
          f"({wall / n_members:.1f}s per network)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
