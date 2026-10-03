"""Stage 3: frozen ESM-2 features, standalone and stacked on the stage 2 baseline.

    python scripts/stage3_esm.py --features features/esm2_35M.npz

Matched to stage 2 in every respect that could manufacture a result:

- the same frozen splits and the same ``cv_folds`` cut along whole peptide
  clusters, so no network stops on a peptide within 3 substitutions of one it
  trained on;
- the same ensemble size and shape -- 5 CV folds x 2 representations x 3 seeds
  = 30 networks. For the sequence baseline the two representations are one-hot
  and BLOSUM62; here they are the middle and final ESM layer, which is the
  analogous diversity axis. Ensembling alone is worth +0.093 mean SCC, so an
  unmatched ensemble would invent a gain;
- the same architecture and tuning budget (the stage 2 selected config).

Dimensionality. The raw blocks are 9 x hidden for the peptide and 34 x hidden
for the HLA contact positions -- 20,640 columns at hidden=480, which no 15,773-row
fit can use. Two reductions, both unsupervised and both fitted on *training*
peptides only:

- HLA: there are only 75 unique domains, so the contact block has rank <= 75 and
  a 74-component PCA is lossless, not an approximation.
- Peptide: PCA to ``--pep-components``, with retained variance reported.

The layer sweep costs no extra fits: member predictions are kept, so the
middle-layer-only and final-layer-only ensembles are formed by averaging the
relevant 15 members.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import (describe_delta, eligible_alleles,  # noqa: E402
                                paired_cluster_bootstrap, score)
from pepstab.features import build_features  # noqa: E402
from pepstab.mlp import MLPConfig, MLPRegressor  # noqa: E402
from scripts.baseline_ensemble import SELECTED, cv_folds  # noqa: E402
from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE, SEEDS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = REPO_ROOT / "preds"
REPORT_DIR = REPO_ROOT / "reports"
BASELINE_PRED = PRED_DIR / "seq_ensemble_pep_pseudo.csv"
HLA_COMPONENTS = 74  # = 75 unique domains - 1; lossless


def reduce_block(block: np.ndarray, keys: np.ndarray, train_keys: set,
                 n_components: int, seed: int = 0) -> tuple[np.ndarray, float]:
    """Flatten a (n_keys, positions, hidden) block and PCA it on training keys.

    Returns the transformed block (n_keys, n_components), standardised to unit
    variance per component, and the retained variance fraction.
    """
    flat = block.reshape(len(block), -1).astype(np.float32)
    fit_mask = np.array([k in train_keys for k in keys])
    n_components = int(min(n_components, fit_mask.sum() - 1, flat.shape[1]))
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(flat[fit_mask])
    out = pca.transform(flat)
    out /= out[fit_mask].std(axis=0, keepdims=True) + 1e-6
    return out.astype(np.float32), float(pca.explained_variance_ratio_.sum())


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
    print(f"{args.features}: layers {layers}, hidden {int(z['hidden'])}, "
          f"extraction {float(z['seconds_peptides']) + float(z['seconds_hla']):.1f} s")

    t0 = time.perf_counter()
    blocks, retained = {}, {}
    for L in layers:
        pep, r_pep = reduce_block(z[f"pep_L{L}"], peptides, train_peptides,
                                  args.pep_components)
        hla, r_hla = reduce_block(z[f"hla_L{L}"], al_names, set(al_names),
                                  HLA_COMPONENTS)
        blocks[L] = (pep, hla)
        retained[L] = (r_pep, r_hla)
        print(f"  layer {L}: peptide {pep.shape[1]} PCs ({r_pep:.1%} variance), "
              f"HLA {hla.shape[1]} PCs ({r_hla:.1%})")
    pca_seconds = time.perf_counter() - t0

    def esm_rows(frame: pd.DataFrame, L: int) -> np.ndarray:
        pep, hla = blocks[L]
        return np.hstack([pep[[pep_pos[p] for p in frame.peptide]],
                          hla[[al_pos[a] for a in frame.allele]]])

    base_train = {enc: build_features(train, "pep_pseudo", enc) for enc in ("onehot",)}
    base_val = {enc: build_features(val, "pep_pseudo", enc) for enc in ("onehot",)}

    arms = {
        "esm_alone": lambda f, L: esm_rows(f, L),
        "esm_stacked": lambda f, L: np.hstack([
            (base_train["onehot"] if f is train else base_val["onehot"]),
            esm_rows(f, L)]).astype(np.float32),
    }

    hidden, l2 = SELECTED[("pep_pseudo", "onehot")]
    members, rows = {}, []
    for arm, make in arms.items():
        Xt = {L: make(train, L) for L in layers}
        Xv = {L: make(val, L) for L in layers}
        print(f"\n{arm}: {Xt[layers[0]].shape[1]} features")
        for L in layers:
            for k in range(args.folds):
                is_fit = fold != k
                for seed in SEEDS:
                    model = MLPRegressor(MLPConfig(hidden=hidden, l2=l2, seed=seed,
                                                   max_epochs=MAX_EPOCHS,
                                                   patience=PATIENCE))
                    model.fit(Xt[L][is_fit], y_train[is_fit],
                              Xt[L][~is_fit], y_train[~is_fit])
                    t0 = time.perf_counter()
                    p = model.predict(Xv[L])
                    infer = (time.perf_counter() - t0) / len(Xv[L]) * 1000
                    members.setdefault((arm, L), []).append(p)
                    rows.append({"arm": arm, "layer": L, "fold": k, "seed": seed,
                                 "n_features": Xt[L].shape[1],
                                 "best_epoch": model.best_epoch_,
                                 "fit_seconds": round(model.fit_seconds_, 1),
                                 "infer_seconds_per_1k": round(infer, 4),
                                 "member_median_rho": score(
                                     "m", val.allele, y_val, p, alleles).median_spearman})
            done = [r for r in rows if r["arm"] == arm and r["layer"] == L]
            print(f"  layer {L}: {len(done)} networks, member median rho "
                  f"{np.mean([r['member_median_rho'] for r in done]):.3f}, "
                  f"{np.mean([r['fit_seconds'] for r in done]):.0f} s/fit")
        del Xt, Xv

    runs = pd.DataFrame(rows)
    runs.to_csv(REPORT_DIR / f"stage3_runs_{args.tag}.csv", index=False)

    base = pd.read_csv(BASELINE_PRED).set_index("pair_id").loc[val.pair_id].y_pred.to_numpy()
    base_scores = score("seq_ensemble_pep_pseudo", val.allele, y_val, base, alleles)

    ensembles = {"seq_ensemble_pep_pseudo (stage 2)": base}
    for arm in arms:
        ensembles[f"{arm} (30 nets)"] = np.mean(
            [p for L in layers for p in members[(arm, L)]], axis=0)
        for L in layers:
            ensembles[f"{arm} layer {L} (15 nets)"] = np.mean(members[(arm, L)], axis=0)

    summary = []
    for name, pred in ensembles.items():
        s = score(name, val.allele, y_val, pred, alleles)
        row = s.as_row()
        if name != "seq_ensemble_pep_pseudo (stage 2)":
            boot = paired_cluster_bootstrap(val.cluster_id, val.allele, y_val,
                                            base, pred, alleles)
            row |= {"delta_vs_baseline": round(boot["delta_median_spearman"], 4),
                    "ci_low": round(boot["ci95"][0], 4),
                    "ci_high": round(boot["ci95"][1], 4),
                    "verdict": describe_delta(boot)}
            PRED_DIR.mkdir(exist_ok=True)
            if "30 nets" in name:
                pd.DataFrame({"pair_id": val.pair_id, "y_pred": pred}).to_csv(
                    PRED_DIR / f"{name.split()[0]}_{args.tag}.csv", index=False)
        summary.append(row)
    out = pd.DataFrame(summary)
    out.to_csv(REPORT_DIR / f"stage3_summary_{args.tag}.csv", index=False)

    cost = {"pca_seconds": round(pca_seconds, 1),
            "embedding_seconds": round(float(z["seconds_peptides"]) + float(z["seconds_hla"]), 1),
            "total_fit_seconds": float(runs.fit_seconds.sum()),
            "median_infer_seconds_per_1k": float(runs.infer_seconds_per_1k.median()),
            "retained_variance": {str(k): v for k, v in retained.items()},
            "baseline_median_rho": base_scores.median_spearman}
    (REPORT_DIR / f"stage3_cost_{args.tag}.json").write_text(json.dumps(cost, indent=2))

    print("\n" + out[["model", "median_per_allele_spearman", "mae_log1p",
                      "median_precision_at_10"] +
                     [c for c in ("delta_vs_baseline", "ci_low", "ci_high", "verdict")
                      if c in out.columns]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
