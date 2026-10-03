"""Stage 3c: BLOSUM positional cross-features -- the control for every
cross-representation idea.

    python scripts/stage3_blosum_cross.py

The feature is the 9 x 34 matrix of BLOSUM62 scores between peptide position i
and HLA contact residue j: 306 columns, no network, no pretrained model,
computed in under a second.

What it controls for. These 306 numbers contain no information the stage 2
baseline lacks -- they are a fixed bilinear function of the same peptide and
the same 34 contact residues the one-hot arm already receives. So a gain here
cannot mean "new information arrived". It can only mean the MLP was not
extracting the peptide-position x pocket-residue interaction efficiently from
one-hot input, which is a statement about the architecture rather than the
representation. A null means the baseline already captures whatever a
positional cross-encoding can express, which bounds every cross-feature scheme
-- ESM cross-embeddings included -- before any of them is worth a GPU hour.

The stacked arm is 1,166 features against the ESM stacked arm's 1,190, so the
two cross-representations are compared at matched width rather than at matched
name.

Ensembling matches stage 2 at 30 networks. The baseline draws its diversity
from two encodings; there is only one BLOSUM cross-encoding, so the second axis
here is the two L2 settings the baseline's two encodings use (1e-5 and 1e-3),
which is a diversity axis of the same size and the same kind.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.data import AA_INDEX, load_with_splits  # noqa: E402
from pepstab.evaluation import (describe_delta, eligible_alleles,  # noqa: E402
                                paired_cluster_bootstrap, score)
from pepstab.features import BLOSUM62, BLOSUM_SCALE, build_features  # noqa: E402
from pepstab.mlp import MLPConfig, MLPRegressor  # noqa: E402
from scripts.baseline_ensemble import cv_folds  # noqa: E402
from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE, SEEDS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = REPO_ROOT / "preds"
REPORT_DIR = REPO_ROOT / "reports"
L2_GRID = (1e-5, 1e-3)
HIDDEN = (256, 64)


#: The repo's own BLOSUM62, already in AA_INDEX order and on the same scale the
#: stage 2 ``blosum`` encoding uses -- so the cross-features and the baseline's
#: residue encoding cannot disagree about what a substitution score is.
_B62 = np.asarray(BLOSUM62, dtype=np.float32) / BLOSUM_SCALE


def cross_features(frame: pd.DataFrame) -> np.ndarray:
    """(n_rows, 306) BLOSUM62 score between every peptide position and contact residue."""
    pep = np.array([[AA_INDEX[c] for c in s] for s in frame.peptide], dtype=np.int16)
    hla = np.array([[AA_INDEX[c] for c in s] for s in frame.hla_pseudoseq], dtype=np.int16)
    return _B62[pep[:, :, None], hla[:, None, :]].reshape(len(frame), -1).astype(np.float32)


def main() -> int:
    df = load_with_splits()
    train = df[df.split == "train"]
    val = df[df.split == "val"].sort_values("pair_id").reset_index(drop=True)
    fold = cv_folds(train)
    alleles = eligible_alleles(val.allele, val.y_log1p, split="val")
    y_train, y_val = train.y_log1p.to_numpy(), val.y_log1p.to_numpy()

    t0 = time.perf_counter()
    cross_train, cross_val = cross_features(train), cross_features(val)
    base_train = build_features(train, "pep_pseudo", "onehot")
    base_val = build_features(val, "pep_pseudo", "onehot")
    feature_seconds = time.perf_counter() - t0

    arms = {
        "blosum_cross_alone": (cross_train, cross_val),
        "blosum_cross_stacked": (np.hstack([base_train, cross_train]),
                                 np.hstack([base_val, cross_val])),
    }

    members, rows = {}, []
    for arm, (Xt, Xv) in arms.items():
        print(f"\n{arm}: {Xt.shape[1]} features")
        for l2 in L2_GRID:
            for k in range(5):
                is_fit = fold != k
                for seed in SEEDS:
                    model = MLPRegressor(MLPConfig(hidden=HIDDEN, l2=l2, seed=seed,
                                                   max_epochs=MAX_EPOCHS,
                                                   patience=PATIENCE))
                    model.fit(Xt[is_fit], y_train[is_fit],
                              Xt[~is_fit], y_train[~is_fit])
                    p = model.predict(Xv)
                    members.setdefault(arm, []).append(p)
                    rows.append({"arm": arm, "l2": l2, "fold": k, "seed": seed,
                                 "n_features": Xt.shape[1],
                                 "best_epoch": model.best_epoch_,
                                 "fit_seconds": round(model.fit_seconds_, 1),
                                 "member_median_rho": score("m", val.allele, y_val,
                                                            p, alleles).median_spearman})
            done = [r for r in rows if r["arm"] == arm and r["l2"] == l2]
            print(f"  l2={l2:g}: {len(done)} networks, member median rho "
                  f"{np.mean([r['member_median_rho'] for r in done]):.3f}, "
                  f"{np.mean([r['fit_seconds'] for r in done]):.0f} s/fit")

    runs = pd.DataFrame(rows)
    runs.to_csv(REPORT_DIR / "stage3c_runs_blosum_cross.csv", index=False)

    def load(name: str) -> np.ndarray | None:
        path = PRED_DIR / name
        if not path.exists():
            return None
        return pd.read_csv(path).set_index("pair_id").loc[val.pair_id].y_pred.to_numpy()

    base = load("seq_ensemble_pep_pseudo.csv")
    refs = {"baseline": base, "esm_stacked": load("esm_stacked_esm2_35M.csv")}

    summary = []
    for name in ("seq_ensemble_pep_pseudo (stage 2)", *(f"{a} (30 nets)" for a in arms)):
        pred = base if name.startswith("seq_ensemble") else np.mean(
            members[name.split()[0]], axis=0)
        row = score(name, val.allele, y_val, pred, alleles).as_row()
        if not name.startswith("seq_ensemble"):
            for ref_name, ref in refs.items():
                if ref is None:
                    continue
                boot = paired_cluster_bootstrap(val.cluster_id, val.allele, y_val,
                                                ref, pred, alleles)
                row |= {f"delta_vs_{ref_name}": round(boot["delta_median_spearman"], 4),
                        f"ci_low_vs_{ref_name}": round(boot["ci95"][0], 4),
                        f"ci_high_vs_{ref_name}": round(boot["ci95"][1], 4),
                        f"verdict_vs_{ref_name}": describe_delta(boot)}
            PRED_DIR.mkdir(exist_ok=True)
            pd.DataFrame({"pair_id": val.pair_id, "y_pred": pred}).to_csv(
                PRED_DIR / f"{name.split()[0]}.csv", index=False)
        summary.append(row)

    out = pd.DataFrame(summary)
    out.to_csv(REPORT_DIR / "stage3c_summary_blosum_cross.csv", index=False)
    (REPORT_DIR / "stage3c_cost_blosum_cross.json").write_text(json.dumps({
        "feature_seconds": round(feature_seconds, 2),
        "total_fit_seconds": float(runs.fit_seconds.sum()),
        "n_features": {a: int(g.n_features.iloc[0]) for a, g in runs.groupby("arm")},
    }, indent=2))

    cols = ["model", "median_per_allele_spearman", "mae_log1p",
            "median_precision_at_10"] + [c for c in out.columns
                                         if c.startswith(("delta", "verdict"))]
    print("\n" + out[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
