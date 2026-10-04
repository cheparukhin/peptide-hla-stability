"""Stage 3f: BLOSUM positional cross-features -- the cheap control that bounds
every cross-representation scheme, ESM's included.

    .venv/bin/python scripts/stage3f_blosum_cross.py run

The feature is the 9 x 34 matrix of BLOSUM62 substitution scores between peptide
position *i* and HLA contact residue *j*: 306 columns, no network, no pretrained
model, computed in under a second.

**What it controls for.** These 306 numbers contain no information the stage 2
baseline lacks. They are a fixed bilinear function of the same 9 peptide
residues and the same 34 contact residues the one-hot arm already receives. So a
gain here cannot mean "new information arrived" -- it can only mean the MLP was
not extracting the peptide-position x pocket-residue interaction efficiently
from one-hot input, which is a finding about architecture rather than about
representation. A null means the baseline already captures whatever a positional
cross-encoding can express, which bounds every cross-feature scheme before any
of them is worth a GPU hour.

**It is also the matched-width companion to the ESM stacked arm.** The stacked
arm here is 860 + 306 = 1,166 features against the ESM additive arm's 1,190, so
the two cross-representations are compared at matched width rather than at
matched name -- which matters, because ``reports/stage3_esm.md`` raises
capacity displacement as a candidate mechanism for the additive arm's −0.017.

**On tuning.** Both arms select L2 on their own ladder of at least three points
and the selection is checked for interiority, for the reason §2 of
``reports/stage3_esm.md`` gives at length. The ``stage3-esm`` branch, which
wrote this experiment and never ran it, drew its ensemble's second axis from
stage 2's two L2 values (1e-5, 1e-3) directly. That is a milder version of the
same transplant: those two values were selected for 860 sparse one-hot columns,
and this arm's inputs are dense scaled substitution scores. Here the second axis
is instead the **top two points of each arm's own ladder** for the standalone
arm, and stage 2's own ``{onehot, blosum}`` encoding axis for the stacked arm --
where it is not a transplant at all, because the stacked arm genuinely contains
the baseline's encoding and should vary it exactly as the baseline does.
"""
from __future__ import annotations

import os as _os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab.data import AA_INDEX, PSEUDOSEQ_LENGTH, load_with_splits  # noqa: E402
from pepstab.evaluation import (  # noqa: E402
    describe_delta,
    eligible_alleles,
    paired_cluster_bootstrap,
    score,
)
from pepstab.features import ENCODINGS, build_features, residue_rows  # noqa: E402
from pepstab.mlp import MLPConfig, MLPRegressor  # noqa: E402
from scripts.baseline_ensemble import N_FOLDS, cv_folds  # noqa: E402
from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE, SEEDS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = REPO_ROOT / "preds"
REPORT_DIR = REPO_ROOT / "reports"

HIDDEN = (256, 64)
PEPTIDE_LENGTH = 9

#: Five points, so the selection is interior with room on both sides. Spans
#: stage 2's range and stage 3's selected 1e-2, because this block's scale sits
#: between the two: scaled BLOSUM scores are dense like the ESM components but
#: bounded and low-dimensional like the one-hot block.
L2_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)

ARMS = ("cross_alone", "cross_stacked")
BASELINES = ("seq_ensemble_pep_pseudo", "esm_ensemble", "esm_plus_seq_ensemble")


def cross_features(frame: pd.DataFrame) -> np.ndarray:
    """``(n_rows, 9 * 34)`` BLOSUM62 score for every (peptide position, contact) pair.

    Uses the repo's own BLOSUM62 at the repo's own scale, via
    :func:`pepstab.features.residue_rows`, so these columns and the baseline's
    ``blosum`` encoding cannot disagree about what a substitution score is.
    """
    b62 = residue_rows("blosum")
    pep = np.array([[AA_INDEX[c] for c in s] for s in frame.peptide], dtype=np.int16)
    hla = np.array([[AA_INDEX[c] for c in s] for s in frame.hla_pseudoseq],
                   dtype=np.int16)
    if pep.shape[1] != PEPTIDE_LENGTH or hla.shape[1] != PSEUDOSEQ_LENGTH:
        raise ValueError(f"expected 9-mers and {PSEUDOSEQ_LENGTH}-residue "
                         f"pseudosequences, got {pep.shape[1]} and {hla.shape[1]}")
    out = b62[pep[:, :, None], hla[:, None, :]]
    return out.reshape(len(frame), -1).astype(np.float32)


def _blocks(train: pd.DataFrame, val: pd.DataFrame) -> tuple[dict, float]:
    """Feature matrices per arm, per raw encoding. Returns (blocks, seconds)."""
    t0 = time.perf_counter()
    cross = {"train": cross_features(train), "val": cross_features(val)}
    blocks: dict = {}
    for enc in ENCODINGS:
        base_t = build_features(train, "pep_pseudo", enc)
        base_v = build_features(val, "pep_pseudo", enc)
        blocks[("cross_stacked", enc)] = (np.hstack([base_t, cross["train"]]),
                                          np.hstack([base_v, cross["val"]]))
    # The standalone arm has no raw encoding to vary; one entry, reused.
    blocks[("cross_alone", None)] = (cross["train"], cross["val"])
    return blocks, time.perf_counter() - t0


def _fit(X_fit, y_fit, X_dev, y_dev, X_val, l2, seed):
    model = MLPRegressor(MLPConfig(hidden=HIDDEN, l2=l2, seed=seed,
                                   max_epochs=MAX_EPOCHS, patience=PATIENCE))
    model.fit(X_fit, y_fit, X_dev, y_dev)
    return model.predict(X_val), model


def _ladder(blocks, train, val, fold_id, y_train, y_val, alleles, grid, fold):
    """Select L2 per arm on its own ladder; return (selected, sensitivity rows, runs)."""
    records = []
    for arm in ARMS:
        key = (arm, ENCODINGS[0]) if arm == "cross_stacked" else (arm, None)
        Xt, Xv = blocks[key]
        for l2 in grid:
            for seed in SEEDS:
                is_fit = fold_id != fold
                pred, model = _fit(Xt[is_fit], y_train[is_fit], Xt[~is_fit],
                                   y_train[~is_fit], Xv, l2, seed)
                records.append({
                    "stage": "ladder", "arm": arm, "encoding": key[1] or "",
                    "l2": l2, "seed": seed, "fold": fold,
                    "n_features": int(Xt.shape[1]),
                    "best_epoch": model.best_epoch_,
                    "fit_seconds": round(model.fit_seconds_, 1),
                    "member_median_rho": round(score("m", val.allele, y_val, pred,
                                                     alleles).median_spearman, 6)})
            got = [r for r in records if r["arm"] == arm and r["l2"] == l2]
            print(f"  {arm:14s} l2={l2:<7g} rho "
                  f"{np.mean([r['member_median_rho'] for r in got]):.4f} "
                  f"(spread {np.ptp([r['member_median_rho'] for r in got]):.4f})")

    runs = pd.DataFrame(records)
    selected, sens = {}, []
    for arm in ARMS:
        means = runs[runs.arm == arm].groupby("l2").member_median_rho.mean()
        tried = sorted(means.index)
        ranked = means.sort_values(ascending=False)
        best = float(ranked.index[0])
        interior = len(tried) >= 3 and min(tried) < best < max(tried)
        selected[arm] = {"l2": best,
                         "l2_second": float(ranked.index[1]),
                         "interior": bool(interior)}
        sens.append({"arm": arm, "ladder": " ".join(f"{t:g}" for t in tried),
                     "n_points": len(tried), "selected": best,
                     "second_best": float(ranked.index[1]),
                     "interior": bool(interior),
                     "rho_at_selected": round(float(means[best]), 4),
                     "rho_at_min": round(float(means[min(tried)]), 4),
                     "rho_at_max": round(float(means[max(tried)]), 4)})
        print(f"\n{arm}: selected l2={best:g} from {len(tried)} points "
              f"({'interior' if interior else 'AT BOUNDARY'})")
    return selected, sens, runs


def _ensemble_axis(arm: str, selected: dict) -> list[tuple]:
    """The 2-way axis the arm's 30 members are spread over.

    ``cross_stacked`` varies the raw encoding exactly as stage 2's ensemble does,
    because it genuinely contains stage 2's encoding. ``cross_alone`` has no raw
    encoding, so it varies over the two best points of its own ladder -- a
    diversity axis of the same size and kind, selected rather than inherited.
    """
    if arm == "cross_stacked":
        return [((arm, enc), selected[arm]["l2"], enc) for enc in ENCODINGS]
    return [((arm, None), selected[arm][k], f"l2={selected[arm][k]:g}")
            for k in ("l2", "l2_second")]


def mode_run(args) -> int:
    df = load_with_splits()
    train = df[df.split == "train"]
    val = df[df.split == "val"].sort_values("pair_id").reset_index(drop=True)
    fold_id = cv_folds(train, N_FOLDS)
    alleles = eligible_alleles(val.allele, val.y_log1p, split="val")
    y_train, y_val = train.y_log1p.to_numpy(), val.y_log1p.to_numpy()

    blocks, feature_seconds = _blocks(train, val)
    widths = {a: int(blocks[(a, ENCODINGS[0]) if a == "cross_stacked"
                            else (a, None)][0].shape[1]) for a in ARMS}
    print(f"features built in {feature_seconds:.2f}s: "
          + ", ".join(f"{a} {w:,} columns" for a, w in widths.items()))
    print(f"(the ESM additive arm is 1,190 columns, so the stacked arm is "
          f"matched to within {abs(widths['cross_stacked'] - 1190) / 1190:.1%})\n")

    grid = tuple(args.l2_grid) if args.l2_grid else L2_GRID
    print(f"ladder: {[f'{g:g}' for g in grid]}, fold {args.fold}, "
          f"{len(SEEDS)} seeds")
    selected, sens, ladder_runs = _ladder(blocks, train, val, fold_id, y_train,
                                          y_val, alleles, grid, args.fold)
    REPORT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(sens).to_csv(REPORT_DIR / "stage3f_tuning.csv", index=False)
    if not all(s["interior"] for s in sens):
        edge = [s["arm"] for s in sens if not s["interior"]]
        raise SystemExit(
            f"\nSelection is on a ladder edge for {edge}. The ladder was "
            f"truncated and the choice was never really made -- extend it with "
            f"--l2-grid and re-run. Not proceeding to the ensemble, because a "
            f"boundary selection is exactly the failure stage 3 §2 documents.")

    if args.ladder_only:
        print("\n--ladder-only: stopping before the ensemble")
        return 0

    n_members = N_FOLDS * 2 * len(SEEDS)
    print(f"\nensemble: {N_FOLDS} folds x 2-way axis x {len(SEEDS)} seeds = "
          f"{n_members} networks per arm (stage 2 and stage 3 both use 30)\n")

    members: dict[str, list[np.ndarray]] = {}
    records = []
    started = time.perf_counter()
    for arm in ARMS:
        for key, l2, axis_label in _ensemble_axis(arm, selected):
            Xt, Xv = blocks[key]
            for k in range(N_FOLDS):
                is_fit = fold_id != k
                for seed in SEEDS:
                    pred, model = _fit(Xt[is_fit], y_train[is_fit], Xt[~is_fit],
                                       y_train[~is_fit], Xv, l2, seed)
                    members.setdefault(arm, []).append(pred)
                    records.append({
                        "stage": "ensemble_member", "arm": arm,
                        "axis": axis_label, "encoding": key[1] or "", "l2": l2,
                        "seed": seed, "fold": k, "n_features": int(Xt.shape[1]),
                        "best_epoch": model.best_epoch_,
                        "fit_seconds": round(model.fit_seconds_, 1),
                        "member_median_rho": round(score("m", val.allele, y_val,
                                                         pred, alleles
                                                         ).median_spearman, 6)})
            got = [r for r in records if r["arm"] == arm
                   and r["axis"] == axis_label]
            print(f"  {arm:14s} {axis_label:10s}: {len(got)} networks, "
                  f"member rho "
                  f"{np.mean([r['member_median_rho'] for r in got]):.4f}, "
                  f"{np.mean([r['fit_seconds'] for r in got]):.0f}s/net")
    wall = time.perf_counter() - started

    runs = pd.concat([ladder_runs, pd.DataFrame(records)], ignore_index=True)
    runs.to_csv(REPORT_DIR / "stage3f_runs.csv", index=False)

    ens = {a: np.mean(members[a], axis=0) for a in ARMS}
    for arm, pred in ens.items():
        pd.DataFrame({"pair_id": val.pair_id.to_numpy(), "y_pred": pred}).to_csv(
            PRED_DIR / f"stage3f_{arm}.csv", index=False)

    refs = {}
    for base in BASELINES:
        path = PRED_DIR / f"{base}.csv"
        if not path.exists():
            print(f"  (missing {path.name}; skipping that comparison)")
            continue
        merged = val[["pair_id"]].merge(pd.read_csv(path), on="pair_id",
                                        how="left", validate="one_to_one")
        if merged.y_pred.isna().any():
            raise ValueError(f"{path.name} does not cover every validation row")
        refs[base] = merged.y_pred.to_numpy()

    summary, comparisons = [], []
    for arm in ARMS:
        row = score(arm, val.allele, y_val, ens[arm], alleles).as_row()
        row["n_members"] = n_members
        row["n_features"] = widths[arm]
        summary.append(row)
    for base, pred in refs.items():
        row = score(base, val.allele, y_val, pred, alleles).as_row()
        row["n_members"] = 30
        summary.append(row)
    pd.DataFrame(summary).to_csv(REPORT_DIR / "stage3f_summary.csv", index=False)

    for arm in ARMS:
        for base, base_pred in refs.items():
            res = paired_cluster_bootstrap(val.cluster_id, val.allele, y_val,
                                           base_pred, ens[arm], alleles=alleles)
            lo, hi = res["ci95"]
            a = score(arm, val.allele, y_val, ens[arm], alleles)
            b = score(base, val.allele, y_val, base_pred, alleles)
            print(f"\n{arm} vs {base}:")
            print(f"  {base} {b.median_spearman:.4f}  {arm} "
                  f"{a.median_spearman:.4f}  delta "
                  f"{res['delta_median_spearman']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            print(f"  verdict: {describe_delta(res)}")
            comparisons.append({
                "arm": arm, "reference": base, "n_rows": len(val),
                "n_alleles": len(alleles), "n_features": widths[arm],
                "reference_median_rho": round(b.median_spearman, 4),
                "arm_median_rho": round(a.median_spearman, 4),
                "delta": round(res["delta_median_spearman"], 4),
                "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                "verdict": describe_delta(res),
                "n_members_arm": n_members, "n_members_reference": 30,
                "l2_arm": selected[arm]["l2"]})
    pd.DataFrame(comparisons).to_csv(REPORT_DIR / "stage3f_comparisons.csv",
                                     index=False)

    (REPORT_DIR / "stage3f_cost.json").write_text(json.dumps({
        "feature_seconds": round(feature_seconds, 2),
        "n_features": widths,
        "n_networks": int(len(runs)),
        "ladder_networks": int(len(ladder_runs)),
        "ensemble_wall_minutes": round(wall / 60, 1),
        "total_fit_seconds": float(runs.fit_seconds.sum()),
        "median_fit_seconds": float(runs.fit_seconds.median()),
        "selected_l2": {a: selected[a] for a in ARMS},
    }, indent=2))
    print(f"\n{len(records)} ensemble networks in {wall / 60:.1f} min")
    print("wrote reports/stage3f_{tuning,runs,summary,comparisons}.csv, "
          "stage3f_cost.json, preds/stage3f_*.csv")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("run",))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--l2-grid", type=float, nargs="*", default=None)
    ap.add_argument("--ladder-only", action="store_true")
    args = ap.parse_args()
    return mode_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
