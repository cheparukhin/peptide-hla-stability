#!/usr/bin/env python
"""Stage 8: score the FoldX arms under the frozen splits.

Implements section 5 of ``reports/stage8_foldx.md``, which was predeclared
before any FoldX number existed. Nothing here chooses a comparison, a ladder,
an ensemble size or a verdict rule -- all of those are read off that protocol.

    .venv/bin/python scripts/foldx_arm.py grid     --table reports/stage8_foldx_features_repair.csv
    .venv/bin/python scripts/foldx_arm.py ensemble --table reports/stage8_foldx_features_repair.csv
    .venv/bin/python scripts/foldx_arm.py diagnose --table reports/stage8_foldx_features_repair.csv

Validation only. The test split is scored once, at stage 6, by the orchestrator
under ``reports/TEST_SCORING_RUNBOOK.md``; ``--split`` accepts nothing else and
:func:`mode_diagnose` drops test rows before it correlates anything.

Two predeclared comparisons (protocol 5.1):

* **arm A**, ``foldx`` -- the ``foldx_`` block standing alone. Expected to land
  near or below the 0.580 measured-affinity ceiling from stage 2c.
* **arm B**, ``seq+foldx`` -- the sequence baseline's own features plus the
  ``foldx_`` block. **This is the headline**, and the only genuinely open
  question in the stage: a trained sequence model has never been given an
  explicit electrostatics or desolvation term, and whether one adds anything on
  top of it is bounded by neither the affinity ceiling nor stage 5's negative.

``seq_only`` is carried alongside them as a **control**, not a third arm, for
the reason stage 5 records: an additive arm otherwise differs from the
committed sequence baseline in three ways at once -- encoding count, L2
selection and the new block -- so a drop cannot be attributed to any one of
them. In stage 5 that control is what established the loss was the features
rather than the tuning, and it is the first thing to read here if arm B is
negative.

The three parity rules are the ones ``scripts/stage5_structural_arm.py``
enforces, each of which has already cost this project something:

1. **Ensemble parity** -- :data:`PARITY_NETWORKS` networks, asserted at run
   time. Ensembling alone is worth about +0.090 median SCC on this task, so an
   under-ensembled arm loses on ensemble size and the loss reads as a verdict
   on the feature.
2. **Fold parity** -- folds come from ``baseline_ensemble.cv_folds``, imported,
   so the assignment is literally the same object the other arms use.
3. **Tuning parity is equal budget over scale-appropriate ranges, not
   transplanted values.** See :data:`FOLDX_L2_GRID`.

Rows whose FoldX ``status`` is not ``ok`` take the declared sequence fallback
rather than being dropped, so the arm is scored on the full frozen cohort and
not on a retrospectively convenient subset of it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Pin BLAS before numpy is imported. Thread count changes float32 results at a
# fixed seed -- about 0.02 SCC on a single network here -- so a run whose
# thread count drifted would not be the run that was recorded.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pepstab.affinity import dual_labelled  # noqa: E402
from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import (  # noqa: E402
    describe_delta, eligible_alleles, paired_cluster_bootstrap, score,
)
from pepstab.features import build_features  # noqa: E402
from pepstab.foldx import FEATURE_PREFIX  # noqa: E402
from pepstab.mlp import MLPConfig, MLPRegressor  # noqa: E402
from pepstab.structural_features import alpha3_provenance  # noqa: E402
from scripts.baseline_ensemble import cv_folds  # noqa: E402
from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE  # noqa: E402

PRED_DIR = REPO / "preds"
REPORT_DIR = REPO / "reports"

#: The frozen name for this arm's predictions.
PRED_PATH = PRED_DIR / "foldx_arm.csv"

#: The declared fallback for pairs with no usable FoldX score (protocol 3.4).
FALLBACK_PRED = PRED_DIR / "seq_ensemble_pep_pseudo.csv"

N_FOLDS = 5
#: 5 folds x 6 seeds = 30, matching the sequence comparator's
#: 5 folds x 2 encodings x 3 seeds. Asserted, not assumed (protocol 5.2).
SEEDS = (0, 1, 2, 3, 4, 5)
PARITY_NETWORKS = 30

#: Protocol 5.3, fixed before any fit. Same *budget* as stage 5 -- 4 MLP points
#: and 5 ridge alphas -- shifted **one decade down**, because the FoldX block is
#: ~28 standardised dense columns against the structural block's 109 and ridge
#: regularisation scales roughly with the number of standardised predictors.
#: Transplanting stage 5's values instead would be a handicap wearing the
#: costume of fairness; that exact mistake cost the ESM arm 0.109 median SCC
#: and nearly produced a false negative.
#:
#: Neither ladder may be shortened to two points: a 2-point ladder can never
#: satisfy ``min < selected < max``, so it can never pass :func:`check_interior`.
FOLDX_L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
FOLDX_ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
HIDDEN = (256, 64)

#: Protocol 5.4: 2,000 resamples, seed 20261003, both arms on the same resample.
N_BOOT = 2000
BOOT_SEED = 20261003

#: The comparator's own input set and encoding, so arm B differs from the
#: sequence baseline by **exactly** the FoldX columns and nothing else.
SEQ_INPUT_SET = "pep_pseudo"
SEQ_ENCODING = "blosum"

#: Protocol 3.3: recorded on every row, excluded from the regression. These are
#: ``foldx_``-prefixed but are provenance, not features -- a binary SHA is not
#: an energy term. Kept as an explicit set so a new provenance column added
#: upstream has to be classified here rather than silently becoming a feature.
FOLDX_PROVENANCE = {
    f"{FEATURE_PREFIX}binary", f"{FEATURE_PREFIX}binary_sha256",
    f"{FEATURE_PREFIX}repair", f"{FEATURE_PREFIX}group1", f"{FEATURE_PREFIX}group2",
}

#: Never features: identity, provenance, cost and run bookkeeping. Timing
#: columns are in here deliberately -- how long FoldX took on a structure is
#: not a property of the peptide, and leaving ``total_s`` in the design matrix
#: would let the model read container scheduling noise as biology.
NON_FEATURE = {
    "allele", "peptide", "complex_id", "pair_id", "split", "cluster_id",
    "status", "profile", "cohort_coverage", "fold_dir",
    "peptide_chain", "hla_chain", "beta2m_chain",
    "n_atoms", "max_coord_dev_A",
    "convert_s", "repair_s", "analyse_s", "total_s",
    "chunk_wall_s", "chunk_procs",
    "dist_to_train", "thalf_hours", "y_log1p", "hla_seq", "hla_pseudoseq",
    "shard", "order_in_shard",
}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """The ``foldx_`` block: every numeric FoldX column that is not provenance.

    Protocol 3.2 takes the column *set* from the file's own header rather than
    from a remembered list, because FoldX's columns differ between releases.
    This mirrors that: it selects by prefix and drops the declared provenance
    columns, so a release that emits an extra term picks it up automatically.

    Any numeric column that is neither a ``foldx_`` column nor explicitly
    non-feature is an error rather than a silent omission -- a column added
    upstream should force a decision, not disappear.
    """
    features, unclassified = [], []
    for col in frame.columns:
        if col in NON_FEATURE or col in FOLDX_PROVENANCE or col.endswith("_sha256"):
            continue
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        if col.startswith(FEATURE_PREFIX):
            features.append(col)
        else:
            unclassified.append(col)
    if unclassified:
        raise SystemExit(
            f"unclassified numeric columns: {unclassified}. A FoldX energy term "
            f"should carry the {FEATURE_PREFIX!r} prefix; anything else belongs in "
            "NON_FEATURE or FOLDX_PROVENANCE. Silently dropping a feature would "
            "understate the arm, and silently keeping bookkeeping would inflate it."
        )
    if not features:
        raise SystemExit(
            f"no {FEATURE_PREFIX!r} feature columns in this table -- is it the "
            "output of scripts/foldx_concat.py?"
        )
    return features


def check_interior(arm: str, configs: dict[str, tuple]) -> list[dict]:
    """Classify each gated selection as interior or at a ladder edge.

    This **records**; it does not raise. :func:`enforce_interior` is the gate,
    and it runs only after the ladder measurements have been written to disk --
    because the protocol requires the objective to be measured before anyone
    concludes the ladder is too narrow, and an exception thrown before the CSV
    is written destroys exactly the evidence that decision needs.

    A boundary hit means the ladder was truncated, so the selected value is not
    the best available one and the arm is handicapped. Same contract as
    ``stage3b_esm_multitask`` and ``stage5_structural_arm`` -- including that
    version's known limitation, restated here because the protocol requires it
    to be: it assumes the objective has an interior optimum, so it reads
    "selection at an edge" as "ladder truncated". On a **flat** objective those
    differ, and the ladder may cover the plateau perfectly while the argmin
    lands on an edge by numerical noise. So measure the objective before
    concluding the ladder is too narrow: ``grid`` prints the dev MSE at every
    rung for exactly that purpose. The tolerance-based fix is deliberately not
    implemented, so that a threshold is not chosen after seeing a result it
    would decide.
    """
    rows = []
    for key, (value, ladder, kind) in configs.items():
        interior = min(ladder) < value < max(ladder)
        rows.append({"arm": arm, "group": key, "kind": kind, "selected": value,
                     "ladder": "|".join(f"{v:g}" for v in ladder),
                     "n_points": len(ladder), "interior": interior, "gated": True})
    return rows


def enforce_interior(rows: list[dict], curves: dict[str, dict]) -> None:
    """Refuse an arm whose gated selection sits at the edge of its own ladder.

    Called **after** the ladder and selection CSVs are on disk. A boundary hit
    means the ladder was truncated, so the selected value is not the best
    available one and the arm is handicapped.

    The message carries the measured dev-MSE spread across the ladder **and the
    worst within-rung seed sd**, because together those distinguish the two
    cases this check cannot tell apart on its own: a genuinely truncated ladder
    shows a gradient toward the edge that is large next to the noise, while a
    flat objective can put the argmin on an edge with the ladder covering the
    plateau perfectly well. When the sd is the larger of the two, widening the
    ladder is the wrong response and the message says so.
    """
    failed = [r for r in rows if r["gated"] and not r["interior"]]
    if not failed:
        return
    lines = []
    for r in failed:
        info = curves.get(r["group"], {})
        spread, sd = info.get("spread"), info.get("worst_sd")
        extra = ""
        if spread is not None:
            extra = f" Ladder spread {spread:.4f}"
            if sd is not None:
                extra += f", worst within-rung seed sd {sd:.4f}"
                extra += (" -- the objective is FLAT relative to its own noise, "
                          "so this edge is not evidence of truncation and "
                          "widening the ladder will not fix it (see R10)."
                          if sd >= spread else " -- a real gradient toward the "
                          "edge, so the ladder does look truncated.")
            else:
                extra += "."
        lines.append(f"  - {r['group']}: selected {r['kind']}={r['selected']:g} "
                     f"at the edge of {r['ladder']}.{extra}")
    raise SystemExit(
        "selection landed on a ladder edge:\n" + "\n".join(lines) +
        "\n\nExtend the ladder and re-select, recording the extension in the "
        "Results section of reports/stage8_foldx.md -- or, if the spread above "
        "shows the objective is flat across the ladder, say that explicitly "
        "with the numbers in reports/stage8_foldx_ladder.csv rather than "
        "widening on reflex. Those measurements have been written already, so "
        "this decision can be made from data."
    )


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def load_table(path: Path) -> pd.DataFrame:
    """The FoldX feature table, joined to the frozen splits.

    Joined on ``(allele, peptide)``, **never** positionally on ``pair_id``:
    ``pair_id`` indexes into the raw CSV, and joining on it would leak training
    rows into validation. Any ``pair_id``/``split``/``cluster_id`` already in
    the table is dropped and re-taken from ``data/splits.csv``, so a stale copy
    in the feature table cannot override the frozen assignment.
    """
    frame = pd.read_csv(path)
    frame = frame.drop(columns=[c for c in ("pair_id", "split", "cluster_id")
                                if c in frame.columns])
    labels = load_with_splits()
    merged = frame.merge(
        labels[["allele", "peptide", "pair_id", "split", "cluster_id",
                "y_log1p", "dist_to_train", "hla_seq", "hla_pseudoseq"]],
        on=["allele", "peptide"], how="left", validate="one_to_one",
    )
    if merged["pair_id"].isna().any():
        raise SystemExit(
            f"{int(merged['pair_id'].isna().sum())} feature rows do not match any "
            "(allele, peptide) in the frozen splits"
        )
    # Protocol 3.4: a FoldX failure is data about that complex, not a reason to
    # drop it. Failed rows leave the model's design matrix and take the
    # declared sequence fallback instead, which keeps the arm on the full
    # frozen cohort.
    if "status" in merged.columns:
        bad = merged["status"].astype(str) != "ok"
        if bad.any():
            print(f"  {int(bad.sum()):,} of {len(merged):,} rows have status != ok; "
                  "they take the declared sequence fallback (protocol 3.4)")
            for value, count in merged.loc[bad, "status"].value_counts().items():
                print(f"    {count:6,}  {value}")
        merged = merged[~bad].reset_index(drop=True)
    return merged


def design(frame: pd.DataFrame, cols: list[str], mu=None, sd=None):
    """Standardised feature matrix, with training moments reused on validation.

    Standardisation is required, not cosmetic: the FoldX terms span several
    orders of magnitude (an interaction energy of tens of kcal/mol against a
    cis-bond term that is usually 0), and an unstandardised L2 penalty would
    effectively regularise only the small-scale ones.
    """
    X = frame[cols].to_numpy(dtype=np.float64)
    if mu is None:
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    X = np.where(np.isfinite(X), X, mu)
    return (X - mu) / sd, mu, sd


def sequence_block(frame: pd.DataFrame) -> np.ndarray:
    """The comparator's own sequence encoding for these rows."""
    return build_features(frame, SEQ_INPUT_SET, SEQ_ENCODING).astype(np.float64)


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


def arms(features: list[str]) -> list[tuple[str, list[str], bool]]:
    """The predeclared arms: ``(name, foldx columns, uses the sequence block)``.

    Order is arm A, then the control, then arm B -- the headline last, so the
    comparison it belongs to is already on screen when it prints.
    """
    return [
        ("foldx", features, False),      # arm A: the block standing alone
        ("seq_only", [], True),          # control, not an arm
        ("seq+foldx", features, True),   # arm B: the headline
    ]


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------


def mode_grid(args) -> int:
    """Select one MLP L2 and one ridge alpha per arm, on inner CV dev."""
    table = load_table(args.table)
    features = feature_columns(table)
    train = table[table.split == "train"].reset_index(drop=True)
    print(f"{len(features)} foldx feature columns; {len(train):,} training rows")
    if len(train) < 50:
        print(f"!! only {len(train)} training rows -- selection is a plumbing "
              "check, not a result")

    fold = cv_folds(train, N_FOLDS)
    dev = fold == 0
    y = train.y_log1p.to_numpy()
    seq_train = sequence_block(train)

    selected, records, ladders, curves = {}, [], [], {}
    for name, cols, use_seq in arms(features):
        X, _, _ = (design(train, cols) if cols
                   else (np.empty((len(train), 0)), None, None))
        if use_seq:
            X = np.hstack([seq_train, X])
        alpha, mse, *_ = ridge_path(X[~dev], y[~dev], X[dev], y[dev], FOLDX_ALPHA_GRID)
        # Every rung is recorded, not just the argmin: check_interior cannot
        # tell a truncated ladder from a flat objective, so the numbers that
        # distinguish them have to be on paper before the gate fires.
        # Selection is on the **mean** dev MSE over SEEDS, not on seed 0 alone.
        # R10 measured why: on the sequence block the between-rung differences
        # (0.007-0.021) are the same size as the within-rung seed noise (sd up
        # to 0.0139), so a single-seed argmin lands wherever that noise puts
        # it. Two runs differing only in row order selected different rungs,
        # one interior and one on a boundary, which would have made the
        # interior gate's verdict a coin flip.
        #
        # This is not a new selection *rule* and not a widened ladder: it is a
        # lower-variance estimator of the same predeclared objective at the
        # same four points, and it is direction-neutral, so it cannot favour
        # more or less regularisation. The budget stays at 4 MLP points and 5
        # ridge alphas, and every arm gets the identical estimator -- which is
        # the parity invariant that has actually cost this project when broken.
        # A 1-SE or tolerance rule was rejected for the reason section 5.3
        # gives: it is directional, and choosing it with these numbers visible
        # would be choosing a threshold after seeing the result it decides.
        curve = {}
        best = (None, np.inf)
        for l2 in FOLDX_L2_GRID:
            per_seed = []
            for seed in SEEDS:
                model = MLPRegressor(MLPConfig(hidden=HIDDEN, l2=l2, seed=seed,
                                               max_epochs=MAX_EPOCHS,
                                               patience=PATIENCE))
                model.fit(X[~dev], y[~dev], X[dev], y[dev])
                per_seed.append(
                    float(np.mean((model.predict(X[dev]) - y[dev]) ** 2)))
                ladders.append({"group": name, "kind": "l2", "value": l2,
                                "seed": seed, "dev_mse": per_seed[-1]})
            mean_mse = float(np.mean(per_seed))
            sd_mse = float(np.std(per_seed))
            curve[l2] = (mean_mse, sd_mse)
            ladders.append({"group": name, "kind": "l2", "value": l2,
                            "seed": "mean", "dev_mse": mean_mse,
                            "dev_mse_sd": sd_mse, "n_seeds": len(SEEDS)})
            if mean_mse < best[1]:
                best = (l2, mean_mse)
        means = [m for m, _ in curve.values()]
        spread = max(means) - min(means)
        worst_sd = max(s for _, s in curve.values())
        curves[f"{name} (mlp l2)"] = {"spread": spread, "worst_sd": worst_sd}
        selected[name] = {"l2": best[0], "alpha": alpha,
                          "n_features": int(X.shape[1]),
                          "selection": f"mean dev MSE over {len(SEEDS)} seeds"}
        records.append({"group": name, "n_features": int(X.shape[1]),
                        "selected_l2": best[0], "mlp_dev_mse": best[1],
                        "mlp_dev_mse_spread": spread,
                        "mlp_dev_mse_worst_seed_sd": worst_sd,
                        "n_selection_seeds": len(SEEDS),
                        "selected_alpha": alpha, "ridge_dev_mse": mse})
        # Printing the spread next to the worst within-rung sd is the whole
        # point: when sd is the larger of the two, the ladder is flat relative
        # to its own noise and no argmin over it means much, however it is
        # estimated.
        flat = " FLAT vs noise" if worst_sd >= spread else ""
        print(f"  {name:12s} {X.shape[1]:5d} features  l2={best[0]:g} "
              f"(mean dev mse {best[1]:.4f} over {len(SEEDS)} seeds; ladder "
              f"spread {spread:.4f}, worst within-rung sd {worst_sd:.4f}"
              f"{flat})  alpha={alpha:g} (dev mse {mse:.4f})")

    # The hard gate covers the MLP L2 only: that is the configuration
    # `mode_ensemble` actually fits, so it is the only one that can handicap
    # the arm that writes predictions. Gating the ridge alpha would be a
    # category error -- the ridge is a linear reference, it generates nothing,
    # and stage 5 measured its dev MSE moving by 2e-4 across five decades, so
    # its argmin is numerical noise on a plateau.
    interior = check_interior("foldx", {
        f"{k} (mlp l2)": (v["l2"], FOLDX_L2_GRID, "l2") for k, v in selected.items()
    })
    for key, value in selected.items():
        at_edge = value["alpha"] in (min(FOLDX_ALPHA_GRID), max(FOLDX_ALPHA_GRID))
        interior.append({"arm": "foldx", "group": f"{key} (ridge alpha)",
                         "kind": "alpha", "selected": value["alpha"],
                         "ladder": "|".join(f"{v:g}" for v in FOLDX_ALPHA_GRID),
                         "n_points": len(FOLDX_ALPHA_GRID), "interior": not at_edge,
                         "gated": False})
        if at_edge:
            print(f"  note: {key} ridge alpha={value['alpha']:g} is at a ladder "
                  "edge. Reported, not gated: the ridge is a reference and does "
                  "not produce the arm's predictions.")

    # Written before the gate fires, so a boundary hit leaves the evidence
    # behind rather than destroying it (see `enforce_interior`).
    REPORT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(REPORT_DIR / "stage8_foldx_grid.csv", index=False)
    pd.DataFrame(ladders).to_csv(REPORT_DIR / "stage8_foldx_ladder.csv", index=False)
    # Protocol 5.3 names this file explicitly.
    pd.DataFrame(interior).to_csv(REPORT_DIR / "stage8_foldx_interior.csv", index=False)
    print("wrote reports/stage8_foldx_{grid,ladder,interior}.csv")

    enforce_interior(interior, curves)

    # Only reached when every gated selection is interior: a selections file
    # that exists is a selections file `ensemble` is entitled to fit from, so
    # it must not be written for a handicapped arm.
    (REPORT_DIR / "stage8_foldx_selected.json").write_text(
        json.dumps(selected, indent=2) + "\n")
    print("all gated MLP selections interior. wrote "
          "reports/stage8_foldx_selected.json")
    return 0


def _covered_diagnostic(val, covered, y_pred, fallback, name: str) -> dict:
    """FoldX vs sequence on the rows that have a FoldX score, same rows both.

    A diagnostic only. The honest headline is the full-cohort number with the
    declared fallback; this view answers the different question "where FoldX
    succeeded, does it beat sequence?" and would be a cherry-picked cohort if
    it were reported as the result.
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
    fx = score(f"foldx[{name}] covered", sub.allele, y_true,
               y_pred[covered.to_numpy()], alleles=al, split="val")
    sq = score("sequence covered", sub.allele, y_true,
               np.array(sub.pair_id.map(fallback), dtype=float), alleles=al, split="val")
    out.update(foldx_median_scc=fx.median_spearman,
               sequence_median_scc=sq.median_spearman,
               delta_median_scc=fx.median_spearman - sq.median_spearman)
    return out


def _awkward_cases(table: pd.DataFrame) -> dict[str, pd.Series]:
    """Protocol section 6's flagged subsets, as boolean masks over ``table``.

    The cohort spells the engineered alleles **with the suffix** --
    ``HLA-B*14:01(C67S)``. An ``isin`` against the bare name silently matches
    nothing and reports them as absent, which is exactly the way this check
    failed once already (R7). Matching on the suffix reproduces section 6's
    counts: 1,135 C67S rows and 1,103 borrowed-alpha3 rows on the full cohort.
    """
    c67s = table.allele.astype(str).str.contains("(C67S)", regex=False)
    prov = alpha3_provenance()
    borrowed_alleles = set(
        prov.loc[prov.alpha3_borrowed.fillna(False), "allele"].astype(str))
    borrowed = table.allele.astype(str).isin(borrowed_alleles)
    return {"c67s": c67s, "borrowed_alpha3": borrowed}


def _subset_scores(name: str, subset: str, mask, val, y_pred, seq) -> dict:
    """Per-subset sensitivity, never a headline (protocol 6)."""
    entry = {"group": name, "subset": subset, "n_rows": int(mask.sum())}
    if mask.sum() < 20:
        entry["note"] = "too few validation rows to score"
        return entry
    y_true = val.y_log1p.to_numpy()
    al = eligible_alleles(val.allele[mask], y_true[mask], split="val")
    entry["n_alleles_eligible"] = len(al)
    if not al:
        entry["note"] = "no allele reaches the validation row threshold"
        return entry
    fx = score(f"{subset} foldx", val.allele[mask], y_true[mask], y_pred[mask],
               alleles=al, split="val")
    sq = score(f"{subset} seq", val.allele[mask], y_true[mask], seq[mask],
               alleles=al, split="val")
    entry.update(foldx_median_scc=fx.median_spearman,
                 sequence_median_scc=sq.median_spearman,
                 delta_median_scc=fx.median_spearman - sq.median_spearman)
    return entry


def mode_ensemble(args) -> int:
    """Fit the parity ensemble, score both arms, write validation predictions."""
    if args.split != "val":
        raise SystemExit(
            "this script writes validation predictions only; the test split is "
            "scored once at stage 6 under reports/TEST_SCORING_RUNBOOK.md"
        )
    table = load_table(args.table)
    features = feature_columns(table)
    sel_path = REPORT_DIR / "stage8_foldx_selected.json"
    if not sel_path.exists():
        raise SystemExit(f"{sel_path} is missing -- run the `grid` mode first")
    selected = json.loads(sel_path.read_text())

    labels = load_with_splits()
    val = labels[labels.split == "val"].sort_values("pair_id").reset_index(drop=True)
    train = table[table.split == "train"].reset_index(drop=True)
    have_val = table[table.split == "val"].copy().set_index("pair_id")

    fallback = pd.read_csv(FALLBACK_PRED).set_index("pair_id")["y_pred"]
    covered = val.pair_id.isin(have_val.index)
    print(f"{len(features)} foldx features; validation rows {len(val):,}; "
          f"with a FoldX score {int(covered.sum()):,}; on the declared sequence "
          f"fallback {int((~covered).sum()):,}")
    if len(train) < 50 or covered.sum() < 20:
        print("!! this is a plumbing check on a partial table, not a result")

    fold = cv_folds(train, N_FOLDS)
    y_train = train.y_log1p.to_numpy()
    y_true = val.y_log1p.to_numpy()
    alleles = eligible_alleles(val.allele, y_true, split="val")
    seq = np.array(val.pair_id.map(fallback), dtype=float)
    masks = _awkward_cases(table)

    seq_train = sequence_block(train)
    seq_val_covered = sequence_block(val[covered.to_numpy()])
    rows, rows_covered, rows_boot, rows_subset = [], [], [], []

    for name, cols, use_seq in arms(features):
        if name not in selected:
            raise SystemExit(f"{sel_path} has no selection for arm {name!r} -- "
                             "re-run `grid` against this table")
        l2 = selected[name]["l2"]
        X_train, mu, sd = (design(train, cols) if cols
                           else (np.empty((len(train), 0)), None, None))
        sub = have_val.reindex(val.pair_id[covered]).reset_index().copy()
        X_val, _, _ = (design(sub, cols, mu, sd) if cols
                       else (np.empty((len(sub), 0)), None, None))
        if use_seq:
            X_train = np.hstack([seq_train, X_train])
            X_val = np.hstack([seq_val_covered, X_val])

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
                f"arm {name!r} fitted {n_networks} networks, not the "
                f"{PARITY_NETWORKS} the sequence comparator uses. Ensembling "
                "alone is worth ~0.090 median SCC here, so an under-ensembled "
                "arm loses on ensemble size and reads as a feature verdict."
            )
        preds /= n_networks

        # np.array, not .to_numpy(): a mapped Series can hand back a read-only
        # view, and the fallback slots have to be overwritable.
        y_pred = np.array(val.pair_id.map(fallback), dtype=float)
        y_pred[covered.to_numpy()] = preds
        if not np.isfinite(y_pred).all():
            raise SystemExit(f"arm {name!r}: non-finite predictions")

        s = score(f"foldx[{name}]", val.allele, y_true, y_pred,
                  alleles=alleles, split="val")
        rows_covered.append(_covered_diagnostic(val, covered, y_pred, fallback, name))

        # Paired peptide-cluster bootstrap against the sequence baseline on
        # identical rows, clusters resampled whole so a cluster spanning many
        # alleles stays intact. Report the difference and its interval, not two
        # separate intervals. A CI crossing zero is inconclusive, not negative.
        boot = paired_cluster_bootstrap(
            val.cluster_id, val.allele, y_true, seq, y_pred,
            alleles=alleles, n_boot=N_BOOT, seed=BOOT_SEED, split="val")
        verdict = describe_delta(boot)
        lo, hi = boot.get("ci95", (float("nan"), float("nan")))
        point = boot.get("delta_median_spearman", float("nan"))
        rows_boot.append({"group": name, "delta_median_scc": point,
                          "ci95_low": lo, "ci95_high": hi,
                          "n_boot": N_BOOT, "seed": BOOT_SEED, "verdict": verdict})

        for subset, mask in masks.items():
            val_mask = val.pair_id.isin(table.loc[mask, "pair_id"]).to_numpy()
            rows_subset.append(
                _subset_scores(name, subset, val_mask, val, y_pred, seq))

        rows.append({"group": name, "n_features": int(X_train.shape[1]), "l2": l2,
                     "n_networks": n_networks,
                     "n_foldx_rows": int(covered.sum()),
                     "n_fallback_rows": int((~covered).sum()),
                     **s.as_row(),
                     "fit_s": round(time.perf_counter() - started, 1)})
        print(f"  {name:12s} median SCC {s.median_spearman:+.4f}  "
              f"MAE {s.mae_log1p:.4f}  P@10 {s.median_precision_at_k:.4f}  "
              f"({n_networks} networks, {rows[-1]['fit_s']}s)")
        print(f"  {name:12s} vs sequence: delta {point:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]  {verdict}")
        if args.write_preds and name == args.select:
            PRED_DIR.mkdir(exist_ok=True)
            pd.DataFrame({"pair_id": val.pair_id, "y_pred": y_pred}).to_csv(
                PRED_PATH, index=False)
            print(f"  -> wrote {PRED_PATH.relative_to(REPO)}")

    REPORT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORT_DIR / "stage8_foldx_val.csv", index=False)
    pd.DataFrame(rows_boot).to_csv(REPORT_DIR / "stage8_foldx_bootstrap.csv", index=False)
    pd.DataFrame(rows_covered).to_csv(
        REPORT_DIR / "stage8_foldx_covered_rows_diagnostic.csv", index=False)
    pd.DataFrame(rows_subset).to_csv(
        REPORT_DIR / "stage8_foldx_subset_sensitivity.csv", index=False)
    print("\nwrote reports/stage8_foldx_{val,bootstrap,covered_rows_diagnostic,"
          "subset_sensitivity}.csv")
    print("headline is `seq+foldx` (arm B); `foldx` is arm A; `seq_only` is the "
          "control that says whether a loss is the features or the tuning.")
    return 0


def mode_diagnose(args) -> int:
    """Protocol 5.5.1: FoldX energy against *measured* affinity. A diagnostic.

    This separates two failure modes that look identical in the headline
    number and have completely different implications: "FoldX is a bad Delta-G
    estimator on these predicted structures" versus "Delta-G is the wrong
    quantity for a dissociation half-life". It may never be used to select a
    model.

    Train and validation rows only. The test split is not loaded here either --
    a correlation is not a score, but the rule is that this workstream does not
    touch those rows, and a rule with an exception for cheap cases is not a rule.
    """
    table = load_table(args.table)
    table = table[table.split.isin(["train", "val"])].reset_index(drop=True)
    energy_col = f"{FEATURE_PREFIX}interaction_energy"
    if energy_col not in table.columns:
        raise SystemExit(f"{energy_col} is not in {args.table}")

    affinity = dual_labelled(table)
    have = affinity.notna().to_numpy() & np.isfinite(
        table[energy_col].to_numpy(dtype=float))
    print(f"{int(have.sum()):,} dual-labelled pairs with a FoldX energy "
          f"(train+val only; test is stage 6's)")
    if have.sum() < 30:
        print("!! too few dual-labelled pairs to report a correlation")
        return 0

    sub = table[have].reset_index(drop=True)
    aff = affinity[have].to_numpy(dtype=float)
    energy = sub[energy_col].to_numpy(dtype=float)

    # Sign, stated so the number can be read: `affinity_to_target` is oriented
    # so that *larger* means *tighter* binding, and a FoldX interaction energy
    # is more negative for tighter binding. A working Delta-G estimator should
    # therefore give a **negative** Spearman here.
    overall = stats.spearmanr(energy, aff)
    print(f"pooled Spearman(energy, affinity target) = {overall.statistic:+.4f} "
          f"(p={overall.pvalue:.3g}); negative is the direction that means "
          "FoldX is tracking affinity")

    per_allele = []
    for allele, grp in sub.assign(_aff=aff).groupby("allele"):
        if len(grp) < args.min_pairs:
            continue
        e, a = grp[energy_col].to_numpy(dtype=float), grp["_aff"].to_numpy(dtype=float)
        if len(np.unique(e)) < 2 or len(np.unique(a)) < 2:
            continue
        r = stats.spearmanr(e, a)
        per_allele.append({"allele": allele, "n_pairs": len(grp),
                           "spearman": r.statistic, "p_value": r.pvalue})
    frame = pd.DataFrame(per_allele)
    out = REPORT_DIR / "stage8_foldx_affinity_diagnostic.csv"
    REPORT_DIR.mkdir(exist_ok=True)
    frame.to_csv(out, index=False)
    if not frame.empty:
        med = float(frame.spearman.median())
        q1, q3 = frame.spearman.quantile([0.25, 0.75])
        print(f"per-allele Spearman over {len(frame)} alleles with "
              f">= {args.min_pairs} dual pairs: median {med:+.4f} "
              f"[IQR {q1:+.4f}, {q3:+.4f}]")
    print(f"wrote {out.relative_to(REPO)}  (DIAGNOSTIC -- never a selection "
          "criterion, never a headline)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["grid", "ensemble", "diagnose"])
    ap.add_argument("--table", type=Path, required=True,
                    help="FoldX feature table (scripts/foldx_concat.py output)")
    ap.add_argument("--split", default="val", choices=["val"],
                    help="validation only; the test split is stage 6's, scored once")
    ap.add_argument("--select", default="seq+foldx",
                    help="which arm's predictions to freeze (default arm B, the headline)")
    ap.add_argument("--write-preds", action="store_true",
                    help=f"write {PRED_PATH.name} for the selected arm")
    ap.add_argument("--min-pairs", type=int, default=30,
                    help="minimum dual-labelled pairs for an allele to enter the "
                         "affinity diagnostic; 30 matches stage 2c's ceiling")
    args = ap.parse_args()
    if args.mode == "grid":
        return mode_grid(args)
    if args.mode == "ensemble":
        return mode_ensemble(args)
    return mode_diagnose(args)


if __name__ == "__main__":
    raise SystemExit(main())
