"""Stage 3e: give the ESM arm its own head, selected on validation.

    python scripts/stage3_head_grid.py --features features/esm2_35M.npz

Stage 3a handed the ESM arms the architecture stage 2 selected for 860 one-hot
columns: h256x64, l2=1e-5. That is the wrong comparison in the one direction
that matters. Stage 2 gave *every* arm a 4-point MLP grid at 3 seeds before
selecting, and its own rule is "use comparable tuning budgets across arms"; the
ESM arms got zero grid points. The stage 3 plan also asks for head architecture
to be selected on validation. So the 3a result currently describes an
under-tuned arm, and a reviewer is entitled to say ESM lost because it never got
its own head.

This runs the same 4-point grid stage 2 used -- {(64,), (256, 64)} x {1e-5, 1e-3}
-- at 3 seeds, on both ESM input sets, selecting on validation median per-allele
Spearman averaged over seeds. The winner is then ensembled at the full 30
networks and compared to the baseline, so the headline number is a tuned arm
against a tuned arm.

Selection uses a single held-out CV fold rather than the full ensemble, exactly
as stage 2 selected before building its ensemble; the ensemble is rebuilt only
for the winning config.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import (describe_delta, eligible_alleles,  # noqa: E402
                                paired_cluster_bootstrap, score)
from pepstab.features import build_features  # noqa: E402
from pepstab.mlp import MLPConfig, MLPRegressor  # noqa: E402
from scripts.baseline_ensemble import cv_folds  # noqa: E402
from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE, SEEDS  # noqa: E402
from scripts.stage3_esm import HLA_COMPONENTS, reduce_block  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = REPO_ROOT / "preds"
REPORT_DIR = REPO_ROOT / "reports"

#: The stage 2 grid, unchanged, so the budgets are comparable by construction.
GRID = [(h, l2) for h in ((64,), (256, 64)) for l2 in (1e-5, 1e-3)]
SELECTION_FOLD = 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features/esm2_35M.npz")
    ap.add_argument("--pep-components", type=int, default=256)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tag", default="esm2_35M")
    args = ap.parse_args()

    df = load_with_splits()
    train = df[df.split == "train"]
    val = df[df.split == "val"].sort_values("pair_id").reset_index(drop=True)
    fold = cv_folds(train, args.folds)
    alleles = eligible_alleles(val.allele, val.y_log1p, split="val")
    y_train, y_val = train.y_log1p.to_numpy(), val.y_log1p.to_numpy()

    z = np.load(args.features, allow_pickle=True)
    layers = sorted(int(k.split("_L")[1]) for k in z.files if k.startswith("pep_L"))
    peptides, al_names = z["peptides"], z["alleles"]
    pep_pos = {p: i for i, p in enumerate(peptides)}
    al_pos = {a: i for i, a in enumerate(al_names)}
    train_peptides = set(train.peptide.unique())

    blocks = {}
    for L in layers:
        pep, _ = reduce_block(z[f"pep_L{L}"], peptides, train_peptides,
                              args.pep_components)
        hla, _ = reduce_block(z[f"hla_L{L}"], al_names, set(al_names), HLA_COMPONENTS)
        blocks[L] = (pep, hla)

    def esm_rows(frame):
        return {L: np.hstack([blocks[L][0][[pep_pos[p] for p in frame.peptide]],
                              blocks[L][1][[al_pos[a] for a in frame.allele]]])
                for L in layers}

    esm_t, esm_v = esm_rows(train), esm_rows(val)
    base_t = build_features(train, "pep_pseudo", "onehot")
    base_v = build_features(val, "pep_pseudo", "onehot")
    inputs = {
        "esm_alone": (esm_t, esm_v),
        "esm_stacked": ({L: np.hstack([base_t, esm_t[L]]) for L in layers},
                        {L: np.hstack([base_v, esm_v[L]]) for L in layers}),
    }

    rows = []
    for arm, (Xt, Xv) in inputs.items():
        for hidden, l2 in GRID:
            for L in layers:
                scores = []
                for seed in SEEDS:
                    model = MLPRegressor(MLPConfig(hidden=hidden, l2=l2, seed=seed,
                                                   max_epochs=MAX_EPOCHS,
                                                   patience=PATIENCE))
                    is_fit = fold != SELECTION_FOLD
                    model.fit(Xt[L][is_fit], y_train[is_fit],
                              Xt[L][~is_fit], y_train[~is_fit])
                    scores.append(score("m", val.allele, y_val, model.predict(Xv[L]),
                                        alleles).median_spearman)
                rows.append({"arm": arm, "hidden": "x".join(map(str, hidden)),
                             "l2": l2, "layer": L, "n_features": Xt[L].shape[1],
                             "rho_mean": float(np.mean(scores)),
                             "rho_spread": float(np.ptp(scores))})
                print(f"  {arm} h{rows[-1]['hidden']} l2={l2:g} L{L}: "
                      f"{rows[-1]['rho_mean']:.3f} (seed spread {rows[-1]['rho_spread']:.3f})")

    grid = pd.DataFrame(rows)
    grid.to_csv(REPORT_DIR / f"stage3e_grid_{args.tag}.csv", index=False)

    summary = []
    base_pred = pd.read_csv(PRED_DIR / "seq_ensemble_pep_pseudo.csv").set_index(
        "pair_id").loc[val.pair_id].y_pred.to_numpy()
    for arm, (Xt, Xv) in inputs.items():
        pick = (grid[grid.arm == arm].groupby(["hidden", "l2"]).rho_mean.mean()
                .idxmax())
        hidden = tuple(int(x) for x in pick[0].split("x"))
        l2 = pick[1]
        prior = grid[(grid.arm == arm) & (grid.hidden == "256x64")
                     & (grid.l2 == 1e-5)].rho_mean.mean()
        print(f"\n{arm}: selected h{pick[0]} l2={l2:g} "
              f"(stage 3a used h256x64 l2=1e-05, {prior:.3f} on this fold)")
        preds = []
        for L in layers:
            for k in range(args.folds):
                is_fit = fold != k
                for seed in SEEDS:
                    model = MLPRegressor(MLPConfig(hidden=hidden, l2=l2, seed=seed,
                                                   max_epochs=MAX_EPOCHS,
                                                   patience=PATIENCE))
                    model.fit(Xt[L][is_fit], y_train[is_fit],
                              Xt[L][~is_fit], y_train[~is_fit])
                    preds.append(model.predict(Xv[L]))
        pred = np.mean(preds, axis=0)
        row = score(f"{arm}_tuned", val.allele, y_val, pred, alleles).as_row()
        row |= {"selected_hidden": pick[0], "selected_l2": l2,
                "n_networks": len(preds)}
        boot = paired_cluster_bootstrap(val.cluster_id, val.allele, y_val,
                                        base_pred, pred, alleles)
        row |= {"delta_vs_baseline": round(boot["delta_median_spearman"], 4),
                "ci_low": round(boot["ci95"][0], 4),
                "ci_high": round(boot["ci95"][1], 4),
                "verdict": describe_delta(boot)}
        summary.append(row)
        pd.DataFrame({"pair_id": val.pair_id, "y_pred": pred}).to_csv(
            PRED_DIR / f"{arm}_tuned_{args.tag}.csv", index=False)

    out = pd.DataFrame(summary)
    out.to_csv(REPORT_DIR / f"stage3e_summary_tuned_{args.tag}.csv", index=False)
    (REPORT_DIR / f"stage3e_grid_spec_{args.tag}.json").write_text(json.dumps({
        "grid": [{"hidden": "x".join(map(str, h)), "l2": l} for h, l in GRID],
        "seeds": list(SEEDS), "selection_fold": SELECTION_FOLD,
        "selected_on": "validation median per-allele Spearman, mean over seeds and layers",
    }, indent=2))
    print("\n" + out[["model", "selected_hidden", "selected_l2",
                      "median_per_allele_spearman", "delta_vs_baseline",
                      "ci_low", "ci_high", "verdict"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
