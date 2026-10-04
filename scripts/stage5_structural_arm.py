#!/usr/bin/env python
"""Stage 5: the Boltz-2 structural arm, under the frozen splits.

Compares structural features against the sequence baseline on **identical**
train/validation rows, folds, ensemble size, seed protocol and tuning budget.
Validation only -- the test split is scored once, at stage 6, by the
orchestrator under ``reports/TEST_SCORING_RUNBOOK.md``.

    .venv/bin/python scripts/stage5_structural_arm.py grid     --table <csv>
    .venv/bin/python scripts/stage5_structural_arm.py ensemble --table <csv>

Three parity rules, each of which has already cost this project something when
it was violated:

1. **Ensemble parity.** The sequence comparator is 5 CV folds x 2 encodings x
   3 seeds = 30 networks. Ensembling alone is worth about +0.090 median SCC on
   this task -- nearly twice the predeclared 0.05 bar -- so an arm with fewer
   networks loses on ensemble size and the result gets read as a feature
   verdict. :data:`PARITY_NETWORKS` is asserted at run time, not just
   documented. The structural block has one natural representation rather than
   two encodings, so it reaches 30 as 5 folds x 6 seeds.
2. **Fold parity.** Folds come from ``baseline_ensemble.cv_folds``, imported
   rather than reimplemented, so the assignment is literally the same object
   the other arms use.
3. **Tuning parity is equal budget with scale-appropriate ranges, not
   transplanted values.** Structural features span four orders of magnitude
   (``global_iptm`` sd 0.004, ``pep_sasa_alone_A2`` sd 91) and are
   standardised to dense columns, so stage 2's sparse-one-hot ladders would be
   a handicap wearing the costume of fairness. That mistake cost the ESM arm
   0.093 SCC tonight and nearly produced a false negative. Same number of grid
   points, different values, and :func:`check_interior` refuses any arm whose
   selection lands on a ladder edge.

Confidence features are **features, never a row filter**: across the stage 4c
pilot they rank error but do not isolate its one genuine failure, so dropping
rows on a confidence threshold would discard usable predictions and bias the
evaluation set. Rows whose extraction did not return ``ok`` take the declared
**sequence fallback** instead of being dropped, so the arm is scored on the
full frozen cohort rather than a retrospectively convenient subset.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Pin BLAS before numpy is imported: this box runs several workstreams at once.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import eligible_alleles, score  # noqa: E402
from pepstab.mlp import MLPConfig, MLPRegressor  # noqa: E402
from scripts.baseline_ensemble import cv_folds  # noqa: E402
from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE  # noqa: E402

PRED_DIR = REPO / "preds"
REPORT_DIR = REPO / "reports"

#: The frozen name for this arm's final predictions. The submission's
#: cost-vs-accuracy figure auto-draws arms by this stem, so a different name
#: silently removes the structural arm from the figure.
PRED_PATH = PRED_DIR / "boltz_structural.csv"

#: The declared fallback for pairs with no usable structure.
FALLBACK_PRED = PRED_DIR / "seq_ensemble_pep_pseudo.csv"

N_FOLDS = 5
#: 5 folds x 6 seeds = 30, matching the sequence comparator's
#: 5 folds x 2 encodings x 3 seeds. Asserted, not assumed.
SEEDS = (0, 1, 2, 3, 4, 5)
PARITY_NETWORKS = 30

#: Scale-appropriate ladders for standardised dense features, at the same
#: *budget* as the comparator: stage 2 gave each arm 4 MLP grid points
#: (2 hidden x 2 L2) and 5 ridge alphas, and this arm gets 4 and 5.
#:
#: The 4 MLP points are spent as one hidden size x four L2 values rather than
#: 2 x 2, for two reasons. The hidden size is not in question -- (256, 64) is
#: stage 2's selection for both HLA arms and what the ESM sweep fixed for the
#: same reason. And a two-point ladder can never satisfy
#: ``min < selected < max``, so spending the budget on two L2 values would
#: make :func:`check_interior` impossible to pass. Stage 3b set the precedent
#: of widening a truncated ladder (it runs 3- and 6-point ladders against
#: stage 2's two).
#:
#: The values are shifted far above stage 2's ``(1e-5, 1e-3)``: these are
#: standardised dense columns, not sparse one-hot, and dense features at this
#: scale need much stronger shrinkage.
STRUCT_L2_GRID = (1e-3, 1e-2, 1e-1, 1.0)
STRUCT_ALPHA_GRID = (1.0, 10.0, 100.0, 1000.0, 10000.0)
HIDDEN = (256, 64)


# --------------------------------------------------------------------------
# feature groups
# --------------------------------------------------------------------------

#: Geometry: contacts, distances and burial against HLA residues 1-182.
#: Deliberately excludes every confidence array, so the two groups can be
#: tested separately before any combination, as the plan requires.
GEOMETRY_PREFIXES = (
    "groove_contacts_", "groove_residues_contacted",
    "pep_min_dist_", "pep_max_min_dist_", "pep_sasa_", "pep_buried_",
)

#: Confidence: pLDDT, PAE in both directions, pTM/ipTM. `global_iptm` is in
#: here under its global name -- in a three-chain complex it covers the
#: HLA/beta2m interface too and is never peptide-interface confidence.
CONFIDENCE_PREFIXES = (
    "pep_plddt_", "groove_plddt_", "hla_chain_plddt_", "alpha3_plddt_",
    "beta2m_plddt_", "complex_plddt_", "pae_", "global_", "chain_ptm_",
    "iptm_pair_",
)

#: Never features: identity, provenance, cost and run bookkeeping.
NON_FEATURE = {
    "run_id", "model", "allele", "peptide", "arm", "seed", "complex_id",
    "pair_id", "split", "cluster_id", "cohort_coverage", "status", "profile",
    "n_chains", "n_tokens", "peptide_length", "hla_chain_length",
    "beta2m_chain_length", "alpha3_source_allele", "alpha3_borrowed",
    "alpha3_provenance", "boltz_package", "weight_revision", "shard",
    "metadata_pair_id", "metadata_split", "seed_scope", "is_shard_first",
    "prediction_order_index", "prediction_order_size", "fold_s",
    "is_warmup", "prediction_order_recorded", "run_id",
    "peak_gpu_gb", "cgroup_peak_gb", "recycling_steps", "sampling_steps",
    "diffusion_samples", "msa_cap", "dist_to_train", "thalf_hours",
    "y_log1p", "hla_seq", "hla_pseudoseq",
}


def feature_columns(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Split the table's numeric columns into the two tested groups.

    Any numeric column that is neither geometry nor confidence nor explicitly
    non-feature is an error rather than a silent omission: a new feature added
    upstream should force a decision about which group it belongs to.
    """
    geometry, confidence, unclassified = [], [], []
    for col in frame.columns:
        if col in NON_FEATURE or col.endswith("_sha256") or col.startswith("msa_"):
            continue
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        if col.startswith(GEOMETRY_PREFIXES):
            geometry.append(col)
        elif col.startswith(CONFIDENCE_PREFIXES):
            confidence.append(col)
        else:
            unclassified.append(col)
    if unclassified:
        raise SystemExit(
            f"unclassified numeric columns: {unclassified}. Assign each to "
            "GEOMETRY_PREFIXES or CONFIDENCE_PREFIXES, or list it in "
            "NON_FEATURE -- silently dropping a feature would understate the arm."
        )
    return {"geometry": geometry, "confidence": confidence,
            "geometry+confidence": geometry + confidence}


def check_interior(arm: str, configs: dict[str, tuple]) -> list[dict]:
    """Refuse an arm whose selection sits at the edge of its own ladder.

    A boundary hit means the ladder was truncated, so the selected value is not
    the best available one and the arm is handicapped. The ESM arm measured
    that handicap at 0.093-0.109 median rho tonight; a silent boundary hit
    inside a comparison whose whole point is parity would reproduce it exactly.
    Same contract as ``scripts/stage3b_esm_multitask.check_interior``.
    """
    rows = []
    for key, (value, ladder, kind) in configs.items():
        interior = min(ladder) < value < max(ladder)
        rows.append({"arm": arm, "group": key, "kind": kind, "selected": value,
                     "ladder": "|".join(f"{v:g}" for v in ladder),
                     "n_points": len(ladder), "interior": interior})
        if not interior:
            raise SystemExit(
                f"arm {arm!r} group {key!r}: selected {kind}={value:g} is at the "
                f"edge of ladder {ladder}. Extend the ladder and re-select, or "
                "this arm is handicapped."
            )
    return rows


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def load_table(path: Path) -> pd.DataFrame:
    """The structural feature table, joined to the frozen splits.

    Joined on ``(allele, peptide)``, never positionally on ``pair_id``.
    """
    frame = pd.read_csv(path)
    for col in ("pair_id", "split", "cluster_id"):
        frame = frame.drop(columns=[col]) if col in frame else frame
    labels = load_with_splits()
    merged = frame.merge(
        labels[["allele", "peptide", "pair_id", "split", "cluster_id",
                "y_log1p", "dist_to_train"]],
        on=["allele", "peptide"], how="left", validate="one_to_one",
    )
    if merged["pair_id"].isna().any():
        raise SystemExit(
            f"{int(merged['pair_id'].isna().sum())} feature rows do not match any "
            "(allele, peptide) in the frozen splits"
        )
    return merged


def design(frame: pd.DataFrame, cols: list[str], mu=None, sd=None):
    """Standardised feature matrix. Columns that are all-NaN here are dropped.

    Standardisation is required, not cosmetic: these columns span four orders
    of magnitude, and an unstandardised L2 penalty would effectively regularise
    only the small-scale ones.
    """
    X = frame[cols].to_numpy(dtype=np.float64)
    if mu is None:
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    X = np.where(np.isfinite(X), X, mu)
    return (X - mu) / sd, mu, sd


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------


def ridge_path(X_fit, y_fit, X_dev, y_dev, alphas):
    """Ridge over ``alphas`` sharing one Gram matrix; alpha chosen on dev MSE."""
    d = X_fit.shape[1]
    mu = X_fit.mean(axis=0)
    ym = float(y_fit.mean())
    Xc = X_fit - mu
    G = Xc.T @ Xc
    b = Xc.T @ (y_fit - ym)
    Xd = X_dev - mu
    best = (None, np.inf, None)
    for alpha in alphas:
        w = np.linalg.solve(G + alpha * np.eye(d), b)
        mse = float(np.mean((Xd @ w + ym - y_dev) ** 2))
        if mse < best[1]:
            best = (alpha, mse, w)
    return best[0], best[1], best[2], mu, ym


def mode_grid(args) -> int:
    """Select one L2 and one ridge alpha per feature group, on inner CV dev."""
    table = load_table(args.table)
    groups = feature_columns(table)
    train = table[table.split == "train"].reset_index(drop=True)
    if len(train) < 50:
        print(f"!! only {len(train)} training rows in this table -- selection is "
              "a plumbing check, not a result")
    fold = cv_folds(train, N_FOLDS)
    dev = fold == 0

    selected, records = {}, []
    for name, cols in groups.items():
        if not cols:
            continue
        X, _, _ = design(train, cols)
        y = train.y_log1p.to_numpy()
        alpha, mse, *_ = ridge_path(X[~dev], y[~dev], X[dev], y[dev], STRUCT_ALPHA_GRID)
        best = (None, np.inf)
        for l2 in STRUCT_L2_GRID:
            model = MLPRegressor(MLPConfig(hidden=HIDDEN, l2=l2, seed=0,
                                           max_epochs=MAX_EPOCHS, patience=PATIENCE))
            model.fit(X[~dev], y[~dev], X[dev], y[dev])
            m = float(np.mean((model.predict(X[dev]) - y[dev]) ** 2))
            if m < best[1]:
                best = (l2, m)
        selected[name] = {"l2": best[0], "alpha": alpha, "n_features": len(cols)}
        records.append({"group": name, "n_features": len(cols),
                        "selected_l2": best[0], "mlp_dev_mse": best[1],
                        "selected_alpha": alpha, "ridge_dev_mse": mse})
        print(f"  {name:22s} {len(cols):4d} features  l2={best[0]:g} "
              f"(dev mse {best[1]:.4f})  alpha={alpha:g} (dev mse {mse:.4f})")

    interior = check_interior("boltz_structural", {
        **{f"{k} (mlp l2)": (v["l2"], STRUCT_L2_GRID, "l2") for k, v in selected.items()},
        **{f"{k} (ridge alpha)": (v["alpha"], STRUCT_ALPHA_GRID, "alpha")
           for k, v in selected.items()},
    })
    out = REPORT_DIR / "stage5_structural_grid.csv"
    pd.DataFrame(records).to_csv(out, index=False)
    pd.DataFrame(interior).to_csv(REPORT_DIR / "stage5_structural_interior.csv", index=False)
    (REPORT_DIR / "stage5_structural_selected.json").write_text(
        json.dumps(selected, indent=2) + "\n")
    print(f"\nall selections interior. wrote {out}")
    return 0


def _common_rows_diagnostic(val, covered, y_pred, fallback, name: str) -> dict:
    """Structural vs sequence on the rows that have a structure, same rows both.

    A diagnostic only. The honest headline is the full-cohort number with the
    declared fallback; this view answers the different question "where a
    structure exists, does it beat sequence?" and would be a cherry-picked
    cohort if it were reported as the result.
    """
    sub = val[covered.to_numpy()]
    if len(sub) < 20:
        return {"group": name, "n_rows": int(len(sub)), "note": "too few rows to score"}
    al = eligible_alleles(sub.allele, sub.y_log1p.to_numpy(), split="val")
    out = {"group": name, "n_rows": int(len(sub)), "n_alleles_eligible": len(al)}
    if not al:
        out["note"] = ("no allele reaches the validation row threshold on the "
                       "covered subset -- not scoreable")
        return out
    y_true = sub.y_log1p.to_numpy()
    st = score(f"structural[{name}] common", sub.allele, y_true,
               y_pred[covered.to_numpy()], alleles=al, split="val")
    sq = score("sequence common", sub.allele, y_true,
               np.array(sub.pair_id.map(fallback), dtype=float), alleles=al, split="val")
    out.update(structural_median_scc=st.median_spearman,
               sequence_median_scc=sq.median_spearman,
               delta_median_scc=st.median_spearman - sq.median_spearman)
    return out


def mode_ensemble(args) -> int:
    """Fit the parity ensemble and write validation predictions."""
    if args.split != "val":
        raise SystemExit(
            "this script writes validation predictions only; the test split is "
            "scored once at stage 6 under reports/TEST_SCORING_RUNBOOK.md"
        )
    table = load_table(args.table)
    groups = feature_columns(table)
    sel_path = REPORT_DIR / "stage5_structural_selected.json"
    if not sel_path.exists():
        raise SystemExit(f"{sel_path} is missing -- run the `grid` mode first")
    selected = json.loads(sel_path.read_text())

    labels = load_with_splits()
    val = labels[labels.split == "val"].sort_values("pair_id").reset_index(drop=True)
    train = table[table.split == "train"].reset_index(drop=True)
    have_val = table[table.split == "val"].copy().set_index("pair_id")

    fallback = pd.read_csv(FALLBACK_PRED).set_index("pair_id")["y_pred"]
    covered = val.pair_id.isin(have_val.index)
    print(f"validation rows {len(val)}; with a usable structure {int(covered.sum())}; "
          f"on the declared sequence fallback {int((~covered).sum())}")
    if len(train) < 50 or covered.sum() < 20:
        print("!! this is a plumbing check on a partial table, not a result")

    fold = cv_folds(train, N_FOLDS)
    y_train = train.y_log1p.to_numpy()
    alleles = eligible_alleles(val.allele, val.y_log1p.to_numpy(), split="val")
    rows, rows_common = [], []

    for name, cols in groups.items():
        if not cols or name not in selected:
            continue
        l2 = selected[name]["l2"]
        X_train, mu, sd = design(train, cols)
        # Rows with a structure get the model; the rest take the fallback.
        sub = have_val.reindex(val.pair_id[covered]).reset_index().copy()
        X_val, _, _ = design(sub, cols, mu, sd)

        preds, n_networks = np.zeros(len(sub)), 0
        started = time.perf_counter()
        for k in range(N_FOLDS):
            is_fit = fold != k
            for seed in SEEDS:
                model = MLPRegressor(MLPConfig(hidden=HIDDEN, l2=l2, seed=seed,
                                               max_epochs=MAX_EPOCHS,
                                               patience=PATIENCE))
                model.fit(X_train[is_fit], y_train[is_fit],
                          X_train[~is_fit], y_train[~is_fit])
                preds += model.predict(X_val)
                n_networks += 1
        if n_networks != PARITY_NETWORKS:
            raise SystemExit(
                f"group {name!r} fitted {n_networks} networks, not the "
                f"{PARITY_NETWORKS} the sequence comparator uses. Ensembling "
                "alone is worth ~0.090 median SCC here, so an under-ensembled "
                "arm loses on ensemble size and reads as a feature verdict."
            )
        preds /= n_networks

        # np.array, not .to_numpy(): a mapped Series can hand back a
        # read-only view, and the fallback slots have to be overwritable.
        y_pred = np.array(val.pair_id.map(fallback), dtype=float)
        y_pred[covered.to_numpy()] = preds
        if not np.isfinite(y_pred).all():
            raise SystemExit(f"group {name!r}: non-finite predictions")

        s = score(f"boltz_structural[{name}]", val.allele, val.y_log1p.to_numpy(),
                  y_pred, alleles=alleles, split="val")

        # Secondary DIAGNOSTIC on the structurally-covered rows only, against
        # the fallback on the same rows. The contract asks for this view and is
        # explicit that it is a diagnostic, not a retrospectively chosen
        # cohort: the primary number above is the full frozen cohort with the
        # declared fallback. It matters most while coverage is partial, when
        # the primary metric is dominated by untouched fallback alleles and
        # cannot distinguish the feature groups at all.
        diag = _common_rows_diagnostic(val, covered, y_pred, fallback, name)
        rows_common.append(diag)
        # as_row() is the canonical metric set every other arm reports, so the
        # comparison tables line up column for column: median per-allele
        # Spearman primary, MAE and precision@10 secondary, per EVALUATION.md.
        rows.append({"group": name, "n_features": len(cols), "l2": l2,
                     "n_networks": n_networks,
                     "n_structural_rows": int(covered.sum()),
                     "n_fallback_rows": int((~covered).sum()),
                     **s.as_row(),
                     "fit_s": round(time.perf_counter() - started, 1)})
        print(f"  {name:22s} median SCC {s.median_spearman:+.4f}  "
              f"MAE {s.mae_log1p:.4f}  P@10 {s.median_precision_at_k:.4f}  "
              f"({n_networks} networks, {rows[-1]['fit_s']}s)")
        if args.write_preds and name == args.select:
            PRED_DIR.mkdir(exist_ok=True)
            pd.DataFrame({"pair_id": val.pair_id, "y_pred": y_pred}).to_csv(
                PRED_PATH, index=False)
            print(f"  -> wrote {PRED_PATH}")

    out = REPORT_DIR / "stage5_structural_val.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    common = REPORT_DIR / "stage5_structural_common_rows_diagnostic.csv"
    pd.DataFrame(rows_common).to_csv(common, index=False)
    print(f"\nwrote {out}")
    print(f"wrote {common}  (DIAGNOSTIC on covered rows only, not the headline)")
    for d in rows_common:
        if "delta_median_scc" in d:
            print(f"  {d['group']:22s} covered rows {d['n_rows']:5d}  "
                  f"structural {d['structural_median_scc']:+.4f} vs sequence "
                  f"{d['sequence_median_scc']:+.4f}  delta {d['delta_median_scc']:+.4f}")
        else:
            print(f"  {d['group']:22s} covered rows {d['n_rows']:5d}  {d.get('note','')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["grid", "ensemble"])
    ap.add_argument("--table", type=Path, required=True,
                    help="structural feature table (stage 4c.5 output)")
    ap.add_argument("--split", default="val", choices=["val"],
                    help="validation only; the test split is stage 6's, scored once")
    ap.add_argument("--select", default="geometry+confidence",
                    help="which group's predictions to freeze as the arm")
    ap.add_argument("--write-preds", action="store_true",
                    help=f"write {PRED_PATH.name} for the selected group")
    args = ap.parse_args()
    return mode_grid(args) if args.mode == "grid" else mode_ensemble(args)


if __name__ == "__main__":
    raise SystemExit(main())
