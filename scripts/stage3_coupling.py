"""Stage 3b: coupling versus concatenation on identical ESM features.

    python scripts/stage3_coupling.py --features features/esm2_35M.npz

Two arms, same module, same folds, same seeds, same budget, same parameter
count: ``xattn`` lets each peptide position attend over the 34 HLA contact
residues, ``meanpool`` gives every peptide position the same groove summary.
The only difference is whether the groove conditions the peptide, so the gap
between them isolates coupling from capacity. The stage 2 sequence ensemble and
the stage 3 concatenation arms remain the external references.

Residue vectors are PCA-reduced on the hidden axis (480 -> --d-in) before either
arm sees them. The projection is fitted on *training* peptide residues only,
with no labels, and applied identically to both arms, so it cannot favour one.

Ensembling matches stage 2 exactly: 5 CV folds x 2 ESM layers x 3 seeds = 30
networks per arm, folds cut along whole Hamming <= 3 peptide clusters.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.attn import AttnConfig, CrossAttentionHead, fit, predict  # noqa: E402
from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import (describe_delta, eligible_alleles,  # noqa: E402
                                paired_cluster_bootstrap, score)
from scripts.baseline_ensemble import cv_folds  # noqa: E402
from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE, SEEDS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = REPO_ROOT / "preds"
REPORT_DIR = REPO_ROOT / "reports"
BASELINE_PRED = PRED_DIR / "seq_ensemble_pep_pseudo.csv"


def reduce_hidden(pep: np.ndarray, hla: np.ndarray, fit_rows: np.ndarray,
                  d_in: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """PCA the hidden axis on training peptide residues; apply to both blocks.

    Each residue vector is one sample, so position structure is untouched: the
    output is still (n, positions, d_in), just in a smaller basis.
    """
    flat = pep[fit_rows].reshape(-1, pep.shape[-1]).astype(np.float32)
    pca = PCA(n_components=d_in, random_state=seed).fit(flat)
    scale = pca.transform(flat).std(axis=0, keepdims=True) + 1e-6

    def apply(block: np.ndarray) -> np.ndarray:
        n, pos, _ = block.shape
        out = pca.transform(block.reshape(-1, block.shape[-1]).astype(np.float32))
        return (out / scale).reshape(n, pos, -1).astype(np.float32)

    return apply(pep), apply(hla), float(pca.explained_variance_ratio_.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features/esm2_35M.npz")
    ap.add_argument("--d-in", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="esm2_35M")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    df = load_with_splits()
    train = df[df.split == "train"]
    val = df[df.split == "val"].sort_values("pair_id").reset_index(drop=True)
    fold = cv_folds(train, args.folds)
    alleles = eligible_alleles(val.allele, val.y_log1p, split="val")
    y_train, y_val = train.y_log1p.to_numpy(), val.y_log1p.to_numpy()

    z = np.load(args.features, allow_pickle=True)
    layers = sorted(int(k.split("_L")[1]) for k in z.files if k.startswith("pep_L"))
    pep_pos = {p: i for i, p in enumerate(z["peptides"])}
    al_pos = {a: i for i, a in enumerate(z["alleles"])}
    train_pep_rows = np.array(sorted({pep_pos[p] for p in train.peptide.unique()}))
    print(f"{args.features}: layers {layers}, hidden {int(z['hidden'])} -> {args.d_in} PCs")

    t0 = time.perf_counter()
    reduced, retained = {}, {}
    for L in layers:
        p, h, r = reduce_hidden(z[f"pep_L{L}"], z[f"hla_L{L}"], train_pep_rows, args.d_in)
        reduced[L] = (p, h)
        retained[L] = r
        print(f"  layer {L}: {r:.1%} of residue variance retained")
    pca_seconds = time.perf_counter() - t0

    tr_pi = np.array([pep_pos[p] for p in train.peptide])
    tr_ai = np.array([al_pos[a] for a in train.allele])
    va_pi = np.array([pep_pos[p] for p in val.peptide])
    va_ai = np.array([al_pos[a] for a in val.allele])

    members, rows = {}, []
    for couple in (True, False):
        arm = "xattn" if couple else "meanpool"
        print(f"\n{arm}:")
        for L in layers:
            pep_blk, hla_blk = reduced[L]
            tr_pep, tr_hla = pep_blk[tr_pi], hla_blk[tr_ai]
            va_pep, va_hla = pep_blk[va_pi], hla_blk[va_ai]
            for k in range(args.folds):
                is_fit = fold != k
                for seed in SEEDS:
                    cfg = AttnConfig(d_model=args.d_model, n_heads=args.heads,
                                     hidden=args.hidden, seed=seed, couple=couple,
                                     max_epochs=MAX_EPOCHS, patience=PATIENCE)
                    model = CrossAttentionHead(args.d_in, cfg)
                    info = fit(model, tr_pep[is_fit], tr_hla[is_fit], y_train[is_fit],
                               tr_pep[~is_fit], tr_hla[~is_fit], y_train[~is_fit])
                    t0 = time.perf_counter()
                    p = predict(model, va_pep, va_hla, info["offset"])
                    infer = (time.perf_counter() - t0) / len(va_pep) * 1000
                    members.setdefault((arm, L), []).append(p)
                    rows.append({"arm": arm, "layer": L, "fold": k, "seed": seed,
                                 "n_parameters": info["n_parameters"],
                                 "best_epoch": info["best_epoch"],
                                 "fit_seconds": round(info["fit_seconds"], 1),
                                 "infer_seconds_per_1k": round(infer, 4),
                                 "member_median_rho": score("m", val.allele, y_val,
                                                            p, alleles).median_spearman})
            done = [r for r in rows if r["arm"] == arm and r["layer"] == L]
            print(f"  layer {L}: {len(done)} networks, "
                  f"{done[0]['n_parameters']:,} parameters, member median rho "
                  f"{np.mean([r['member_median_rho'] for r in done]):.3f}, "
                  f"{np.mean([r['fit_seconds'] for r in done]):.0f} s/fit")

    runs = pd.DataFrame(rows)
    runs.to_csv(REPORT_DIR / f"stage3b_runs_{args.tag}.csv", index=False)
    assert runs.groupby("arm").n_parameters.nunique().eq(1).all(), "arms differ in size"

    base = pd.read_csv(BASELINE_PRED).set_index("pair_id").loc[val.pair_id].y_pred.to_numpy()
    ens = {"seq_ensemble_pep_pseudo (stage 2)": base}
    for arm in ("xattn", "meanpool"):
        ens[f"{arm} (30 nets)"] = np.mean([p for L in layers for p in members[(arm, L)]],
                                          axis=0)
        for L in layers:
            ens[f"{arm} layer {L} (15 nets)"] = np.mean(members[(arm, L)], axis=0)

    summary = []
    for name, pred in ens.items():
        row = score(name, val.allele, y_val, pred, alleles).as_row()
        for ref_name, ref in (("baseline", base),
                              ("meanpool", ens["meanpool (30 nets)"])):
            if name.startswith(ref_name) or (ref_name == "baseline"
                                             and "stage 2" in name):
                continue
            boot = paired_cluster_bootstrap(val.cluster_id, val.allele, y_val, ref,
                                            pred, alleles)
            row |= {f"delta_vs_{ref_name}": round(boot["delta_median_spearman"], 4),
                    f"ci_low_vs_{ref_name}": round(boot["ci95"][0], 4),
                    f"ci_high_vs_{ref_name}": round(boot["ci95"][1], 4),
                    f"verdict_vs_{ref_name}": describe_delta(boot)}
        summary.append(row)
        if "30 nets" in name:
            PRED_DIR.mkdir(exist_ok=True)
            pd.DataFrame({"pair_id": val.pair_id, "y_pred": pred}).to_csv(
                PRED_DIR / f"{name.split()[0]}_{args.tag}.csv", index=False)

    out = pd.DataFrame(summary)
    out.to_csv(REPORT_DIR / f"stage3b_summary_{args.tag}.csv", index=False)
    (REPORT_DIR / f"stage3b_cost_{args.tag}.json").write_text(json.dumps({
        "pca_seconds": round(pca_seconds, 1),
        "retained_residue_variance": retained,
        "total_fit_seconds": float(runs.fit_seconds.sum()),
        "median_infer_seconds_per_1k": float(runs.infer_seconds_per_1k.median()),
        "n_parameters": {a: int(g.n_parameters.iloc[0])
                         for a, g in runs.groupby("arm")},
    }, indent=2))

    cols = ["model", "median_per_allele_spearman", "mae_log1p",
            "median_precision_at_10"] + [c for c in out.columns
                                         if c.startswith(("delta", "verdict"))]
    print("\n" + out[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
