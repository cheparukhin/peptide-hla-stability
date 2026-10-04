"""Stage 3d: positive and negative control blocks for the ESM ablation.

    python scripts/stage3_controls.py --features features/esm2_35M.npz

Stage 3a found frozen ESM-2 features below the baseline standalone (0.387) and
costly when stacked (0.539 against 0.690). Two explanations survive that result
and they call for different conclusions:

1. the embedding carries no stability signal; or
2. the reduction/standardisation/training path I built around it destroys
   signal, or 330 extra columns displace baseline capacity under a fixed
   budget, and ESM is being blamed for the pipeline.

Three control blocks separate them. Every one is pushed through the *identical*
path as the ESM arms -- same PCA width, same per-component standardisation, same
folds, seeds, architecture, budget and 30-network ensemble.

**Positive control** (``onehot_pca``): the baseline's own one-hot features,
PCA-reduced to the same width. This block is known sufficient -- unreduced it
reaches 0.690. If it survives the pipeline and ESM does not, the pipeline is
exonerated. If it collapses too, stage 3a is an artifact of the reduction and
must be rerun at full width.

**Negative control A** (``random``): Gaussian columns, matched width, zero
information by construction. Standalone it bounds what the metric reports for a
block with nothing in it; stacked it measures what 330 uninformative columns
cost the baseline. That stacked number is the one stage 3a is missing: if noise
also costs ~0.15, the ESM stacking penalty is displacement, not misinformation.

**Negative control B** (``shuffled``): real ESM vectors with the peptide-to-
embedding correspondence permuted. Same marginals, same covariance, no
correspondence. Real-minus-shuffled is the information ESM actually contributes.

Each arm keeps the two-way diversity axis the baseline gets from its two
encodings: two independent random draws, two independent shuffles, and for the
positive control the PCA of the one-hot and BLOSUM encodings respectively.
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

from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import (describe_delta, eligible_alleles,  # noqa: E402
                                paired_cluster_bootstrap, score)
from pepstab.features import build_features  # noqa: E402
from pepstab.mlp import MLPConfig, MLPRegressor  # noqa: E402
from scripts.baseline_ensemble import SELECTED, cv_folds  # noqa: E402
from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE, SEEDS  # noqa: E402
from scripts.stage3_esm import reduce_block, HLA_COMPONENTS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = REPO_ROOT / "preds"
REPORT_DIR = REPO_ROOT / "reports"
BASELINE_PRED = PRED_DIR / "seq_ensemble_pep_pseudo.csv"
CONTROL_SEED = 20261003


def standardise(block: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    """Centre and scale on fit rows only, as :func:`reduce_block` does."""
    mu = block[fit_rows].mean(axis=0, keepdims=True)
    sd = block[fit_rows].std(axis=0, keepdims=True) + 1e-6
    return ((block - mu) / sd).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features/esm2_35M.npz")
    ap.add_argument("--pep-components", type=int, default=256)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tag", default="esm2_35M")
    ap.add_argument("--all-arms", action="store_true",
                    help="also run random_alone and shuffled_stacked; the default "
                         "set is the three arms that decide something")
    args = ap.parse_args()

    df = load_with_splits()
    train = df[df.split == "train"]
    val = df[df.split == "val"].sort_values("pair_id").reset_index(drop=True)
    fold = cv_folds(train, args.folds)
    alleles = eligible_alleles(val.allele, val.y_log1p, split="val")
    y_train, y_val = train.y_log1p.to_numpy(), val.y_log1p.to_numpy()
    n_train, n_val = len(train), len(val)
    rng = np.random.default_rng(CONTROL_SEED)

    z = np.load(args.features, allow_pickle=True)
    layers = sorted(int(k.split("_L")[1]) for k in z.files if k.startswith("pep_L"))
    peptides, al_names = z["peptides"], z["alleles"]
    pep_pos = {p: i for i, p in enumerate(peptides)}
    al_pos = {a: i for i, a in enumerate(al_names)}
    train_peptides = set(train.peptide.unique())

    # The real ESM row block, rebuilt exactly as stage 3a builds it.
    esm_rows = {}
    for L in layers:
        pep, _ = reduce_block(z[f"pep_L{L}"], peptides, train_peptides,
                              args.pep_components)
        hla, _ = reduce_block(z[f"hla_L{L}"], al_names, set(al_names), HLA_COMPONENTS)
        esm_rows[L] = (np.hstack([pep[[pep_pos[p] for p in train.peptide]],
                                  hla[[al_pos[a] for a in train.allele]]]),
                       np.hstack([pep[[pep_pos[p] for p in val.peptide]],
                                  hla[[al_pos[a] for a in val.allele]]]))
    width = esm_rows[layers[0]][0].shape[1]
    base_t = {e: build_features(train, "pep_pseudo", e) for e in ("onehot", "blosum")}
    base_v = {e: build_features(val, "pep_pseudo", e) for e in ("onehot", "blosum")}
    fit_rows = np.arange(n_train)

    variants: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    # Positive control: the baseline's own features, same width, same path.
    variants["onehot_pca"] = []
    for enc in ("onehot", "blosum"):
        stacked = np.vstack([base_t[enc], base_v[enc]])[:, None, :]
        keys = np.arange(len(stacked))
        red, retained = reduce_block(stacked, keys, set(range(n_train)), width)
        print(f"  positive control ({enc}): {red.shape[1]} PCs, {retained:.1%} variance")
        variants["onehot_pca"].append((red[:n_train], red[n_train:]))

    # Negative control A: noise at matched width.
    variants["random"] = [
        (standardise(rng.normal(size=(n_train, width)).astype(np.float32), fit_rows),
         rng.normal(size=(n_val, width)).astype(np.float32))
        for _ in range(2)]

    # Negative control B: real ESM vectors, correspondence destroyed. The
    # permutation is over *unique peptides*, so a peptide keeps one wrong
    # embedding everywhere it appears -- shuffling rows instead would also
    # destroy the repeated-peptide structure and confound the two effects.
    variants["shuffled"] = []
    for L in layers:
        perm = rng.permutation(len(peptides))
        shuffled_pos = {p: perm[i] for p, i in pep_pos.items()}
        pep, _ = reduce_block(z[f"pep_L{L}"], peptides, train_peptides,
                              args.pep_components)
        hla, _ = reduce_block(z[f"hla_L{L}"], al_names, set(al_names), HLA_COMPONENTS)
        variants["shuffled"].append((
            np.hstack([pep[[shuffled_pos[p] for p in train.peptide]],
                       hla[[al_pos[a] for a in train.allele]]]),
            np.hstack([pep[[shuffled_pos[p] for p in val.peptide]],
                       hla[[al_pos[a] for a in val.allele]]])))

    # The decisive three: the positive control (is the pipeline lossy?), shuffled
    # ESM alone (does the embedding carry information at all?), and the random
    # block stacked (what do N uninformative columns cost the baseline?).
    # random_alone measures a block with nothing in it, and shuffled_stacked
    # repeats what shuffled_alone already answers; both are opt-in.
    DEFAULT_ARMS = {"onehot_pca_alone", "shuffled_alone", "random_stacked"}

    arms: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for name, blocks in variants.items():
        candidates = {f"{name}_alone": blocks}
        if name != "onehot_pca":  # stacking the baseline on itself is not a control
            candidates[f"{name}_stacked"] = [
                (np.hstack([base_t["onehot"], bt]), np.hstack([base_v["onehot"], bv]))
                for bt, bv in blocks]
        for arm_name, blks in candidates.items():
            if args.all_arms or arm_name in DEFAULT_ARMS:
                arms[arm_name] = blks

    hidden, l2 = SELECTED[("pep_pseudo", "onehot")]
    members, rows = {}, []
    for arm, blocks in arms.items():
        print(f"\n{arm}: {blocks[0][0].shape[1]} features")
        for v, (Xt, Xv) in enumerate(blocks):
            for k in range(args.folds):
                is_fit = fold != k
                for seed in SEEDS:
                    model = MLPRegressor(MLPConfig(hidden=hidden, l2=l2, seed=seed,
                                                   max_epochs=MAX_EPOCHS,
                                                   patience=PATIENCE))
                    model.fit(Xt[is_fit], y_train[is_fit], Xt[~is_fit], y_train[~is_fit])
                    p = model.predict(Xv)
                    members.setdefault(arm, []).append(p)
                    rows.append({"arm": arm, "variant": v, "fold": k, "seed": seed,
                                 "n_features": Xt.shape[1],
                                 "best_epoch": model.best_epoch_,
                                 "fit_seconds": round(model.fit_seconds_, 1),
                                 "member_median_rho": score("m", val.allele, y_val,
                                                            p, alleles).median_spearman})
        done = [r for r in rows if r["arm"] == arm]
        print(f"  {len(done)} networks, member median rho "
              f"{np.mean([r['member_median_rho'] for r in done]):.3f}, "
              f"best epoch {np.mean([r['best_epoch'] for r in done]):.1f}")

    runs = pd.DataFrame(rows)
    runs.to_csv(REPORT_DIR / f"stage3d_runs_controls_{args.tag}.csv", index=False)

    def load(name):
        path = PRED_DIR / name
        return (pd.read_csv(path).set_index("pair_id").loc[val.pair_id].y_pred.to_numpy()
                if path.exists() else None)

    base = load("seq_ensemble_pep_pseudo.csv")
    refs = {"baseline": base,
            "esm_alone": load(f"esm_alone_{args.tag}.csv"),
            "esm_stacked": load(f"esm_stacked_{args.tag}.csv")}

    summary = []
    for arm in arms:
        pred = np.mean(members[arm], axis=0)
        row = score(arm, val.allele, y_val, pred, alleles).as_row()
        # Compare each control to the reference it is a control *for*.
        targets = ["baseline"] + (["esm_stacked"] if arm.endswith("_stacked")
                                  else ["esm_alone"])
        for ref_name in targets:
            ref = refs.get(ref_name)
            if ref is None:
                continue
            boot = paired_cluster_bootstrap(val.cluster_id, val.allele, y_val,
                                            ref, pred, alleles)
            row |= {f"delta_vs_{ref_name}": round(boot["delta_median_spearman"], 4),
                    f"ci_low_vs_{ref_name}": round(boot["ci95"][0], 4),
                    f"ci_high_vs_{ref_name}": round(boot["ci95"][1], 4),
                    f"verdict_vs_{ref_name}": describe_delta(boot)}
        summary.append(row)
        PRED_DIR.mkdir(exist_ok=True)
        pd.DataFrame({"pair_id": val.pair_id, "y_pred": pred}).to_csv(
            PRED_DIR / f"control_{arm}_{args.tag}.csv", index=False)

    out = pd.DataFrame(summary)
    out.to_csv(REPORT_DIR / f"stage3d_summary_controls_{args.tag}.csv", index=False)
    (REPORT_DIR / f"stage3d_cost_controls_{args.tag}.json").write_text(json.dumps({
        "total_fit_seconds": float(runs.fit_seconds.sum()),
        "control_seed": CONTROL_SEED,
        "block_width": int(width),
    }, indent=2))

    cols = ["model", "median_per_allele_spearman", "mae_log1p"] + [
        c for c in out.columns if c.startswith(("delta", "verdict"))]
    print("\n" + out[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
