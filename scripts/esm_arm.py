"""Stage 3: the frozen ESM-2 arm, selected on validation and matched to stage 2.

    .venv/bin/python scripts/esm_arm.py sweep                 # representation sweep
    .venv/bin/python scripts/esm_arm.py grid --rep pos_contact_final
    .venv/bin/python scripts/esm_arm.py ensemble --checkpoint esm2_t12_35M_UR50D

Three modes, in the order they are meant to be run:

``sweep``
    Every (peptide rep x HLA rep x layer) combination, with ridge and a single
    MLP config at 3 seeds, under the stage 2 **single-network** protocol
    (``inner_folds()``). Cheap, and it is the only thing that chooses the
    representation. Validation only.

``grid``
    The stage 2 tuning budget -- 2 hidden sizes x 2 L2 values x 3 seeds -- on
    the representations the sweep selected, so the ESM head is tuned no harder
    and no softer than the sequence head was.

``ensemble``
    The matched comparison. Stage 2's strong baseline is **5 CV folds x 2
    encodings x 3 seeds = 30 networks** (``scripts/baseline_ensemble.py``);
    ensembling alone was worth +0.090 mean SCC there, so an un-ensembled ESM
    arm against it would manufacture a negative result exactly as an ensembled
    ESM arm against a single sequence network would manufacture a positive one.
    This mode builds **5 CV folds x 2 layers x 3 seeds = 30 networks** using
    ``cv_folds()`` imported from that same script -- identical fold assignment,
    identical seeds, identical head class -- and reports a paired cluster
    bootstrap against the stage 2 ensemble's saved validation predictions.

    The two-way representation axis (mid layer / final layer) stands in for
    stage 2's two-way encoding axis (one-hot / BLOSUM62). Member count, fold
    protocol, seeds, and head family all match; what differs is only what the
    networks are fed.

Preprocessing, applied identically to every arm and fitted **without labels**:
columns are standardised, and HLA blocks are reduced by PCA. The HLA reduction
is lossless by construction -- the dataset holds 75 distinct HLA domains, so
any HLA representation has rank <= 74 no matter how many columns it has
(verified: the 16,320-column contact block has exactly 74 non-negligible
singular values). Transforms are fitted on the *unique sequences appearing in
the training split*, never on validation.

Test is never loaded.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab import esm as pesm  # noqa: E402
from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import (  # noqa: E402
    MIN_WORTHWHILE_DELTA_SPEARMAN,
    describe_delta,
    eligible_alleles,
    paired_cluster_bootstrap,
    score,
)
from pepstab.features import ENCODINGS, build_features  # noqa: E402
from pepstab.mlp import MLPConfig, MLPRegressor  # noqa: E402
from scripts.baseline_ensemble import N_FOLDS, cv_folds  # noqa: E402
from scripts.baseline_sequence import (  # noqa: E402
    ALPHA_GRID,
    HIDDEN_GRID,
    L2_GRID,
    MAX_EPOCHS,
    PATIENCE,
    SEEDS,
    inner_folds,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = REPO_ROOT / "preds"
REPORT_DIR = REPO_ROOT / "reports"
RUNS_CSV = REPORT_DIR / "stage3_runs.csv"

#: HLA blocks are reduced to this many components. 75 distinct HLA domains
#: means rank <= 74, so this is lossless, not a compression choice.
HLA_PCA = 74

#: Peptide blocks are reduced to this many components by default. Unlike the
#: HLA reduction this one **is** lossy -- 5,633 distinct peptides can support
#: far more than 256 directions -- so it is a resource decision, taken because
#: this machine has 16 GB shared across the whole team and the full
#: per-position block is 4,320 columns (346 MB for the training split alone,
#: before any float64 copy a solver makes).
#:
#: It is not taken on trust: ``--pep-pca 0`` disables it, and the selected
#: representation is re-run uncompressed as a control so the report can state
#: what, if anything, the reduction cost. Explained variance is recorded per
#: run in ``reports/stage3_runs.csv``.
PEP_PCA = 256

#: The single config the representation sweep uses, so every combination gets
#: the same budget. It is stage 2's selected architecture for both HLA arms.
SWEEP_CONFIG = ((256, 64), 1e-2)

#: **Tuning parity is budget parity, not value parity.** Stage 2 gave each arm
#: 5 ridge alphas and 4 MLP grid points at 3 seeds. The ESM arm gets the same
#: counts -- but not the same *values*, because the feature scale is different
#: by orders of magnitude. Stage 2's inputs are sparse one-hot columns over 860
#: dimensions; the ESM inputs are dense standardised components, and the HLA
#: side carries only 75 distinct values no matter how many columns it occupies.
#: Dense features at that scale need far stronger shrinkage, so transplanting
#: stage 2's ``(0.1 ... 1000)`` alphas and ``(1e-5, 1e-3)`` L2 onto them would
#: be a handicap wearing the costume of fairness -- and would manufacture
#: exactly the false negative this stage exists to avoid.
#:
#: Both ladders are checked for **boundary hits**: if the selected value is the
#: largest or smallest tried, the grid was truncated and the run says so, so a
#: reader can see the selection was interior.
ESM_ALPHA_GRID = (10.0, 100.0, 1000.0, 10000.0, 100000.0)
ESM_L2_GRID = (1e-3, 1e-1)

#: Representation combinations compared on validation.
#: ``(peptide rep, HLA rep)``; each is run at both layers.
REP_COMBOS = (
    ("pos", "contact"),
    ("pos", "mean"),
    ("mean", "contact"),
    ("mean", "mean"),
    ("pos", "none"),
)
LAYERS = ("mid", "final")


def ridge_path(X_fit: np.ndarray, y_fit: np.ndarray, X_dev: np.ndarray,
               y_dev: np.ndarray, X_eval: np.ndarray,
               alphas=ESM_ALPHA_GRID) -> tuple[float, float, np.ndarray, float]:
    """Ridge over ``alphas`` sharing one Gram matrix; alpha chosen on dev.

    Mathematically identical to fitting ``sklearn.linear_model.Ridge`` once per
    alpha (asserted in ``tests/test_esm.py``), but sklearn rebuilds
    ``X.T @ X`` inside every call. On the widest ESM block -- 17,744 rows x
    4,394 columns -- that one product is ~20 s, so a 5-point alpha sweep spent
    two minutes recomputing the same matrix. Here it is formed once in float64
    (chunked, to avoid a 600 MB float64 copy of the features) and each alpha is
    a Cholesky solve on top. Same grid, same answers, ~6x less time.

    Returns ``(best_alpha, dev_mse, eval_predictions, fit_seconds)``.
    """
    t0 = time.perf_counter()
    d = X_fit.shape[1]
    mu = X_fit.mean(axis=0, dtype=np.float64)
    ym = float(np.mean(y_fit, dtype=np.float64))
    yc = np.asarray(y_fit, dtype=np.float64) - ym

    G = np.zeros((d, d))
    b = np.zeros(d)
    for s in range(0, len(X_fit), 2048):
        B = X_fit[s:s + 2048].astype(np.float64)
        B -= mu
        G += B.T @ B
        b += B.T @ yc[s:s + 2048]

    Xd = np.asarray(X_dev, dtype=np.float64) - mu
    best = (None, np.inf, None)
    eye = np.eye(d)
    for alpha in alphas:
        w = cho_solve(cho_factor(G + alpha * eye, overwrite_a=False), b)
        mse = float(np.mean((Xd @ w + ym - y_dev) ** 2))
        if mse < best[1]:
            best = (alpha, mse, w)
    del Xd, G, eye
    alpha, mse, w = best
    pred = (np.asarray(X_eval, dtype=np.float64) - mu) @ w + ym
    return float(alpha), mse, pred, time.perf_counter() - t0


def rep_label(pep: str, hla: str, layer: str, raw: str | None = None) -> str:
    base = f"pep{pep}_hla{hla}_{layer}"
    return f"{base}+raw" if raw else base


# --- feature assembly -------------------------------------------------------


def _unique_matrix(kind: str, rep: str, checkpoint: str, layer: str
                   ) -> tuple[dict[str, int], np.ndarray]:
    """Per-unique-sequence feature matrix, flattened to 2-D."""
    lookup, reps = pesm.load_cache(checkpoint, kind, layer)
    if rep == "mean":
        M = reps.mean(axis=1)
    elif rep == "pos":
        M = reps
    elif rep == "contact":
        M = reps[:, np.asarray(pesm.CONTACT_POSITIONS_1BASED) - 1, :]
    else:
        raise ValueError(f"unknown rep {rep!r}")
    return lookup, M.reshape(len(M), -1)


@dataclass
class _Block:
    """One standardised (and optionally PCA-reduced) per-unique-sequence block."""

    matrix: np.ndarray           # (n_unique, k) after the transform
    index: np.ndarray            # row -> unique index, for the training split
    explained: float             # PCA explained-variance share, 1.0 if no PCA


def _fit_block(kind: str, rep: str, checkpoint: str, layer: str,
               train_values: np.ndarray, n_pca: int
               ) -> tuple[np.ndarray, dict[str, int], float]:
    """Fit standardisation (+PCA) on the sequences present in ``train_values``.

    Fitting on *unique* sequences rather than on rows means a peptide measured
    on 36 alleles does not get 36 votes in the mean and variance. The transform
    is unsupervised either way; this is just the version that matches how the
    cache is keyed.
    """
    lookup, M = _unique_matrix(kind, rep, checkpoint, layer)
    train_rows = np.unique([lookup[str(v)] for v in np.unique(train_values)])
    fit = M[train_rows]

    mu = fit.mean(axis=0)
    sd = fit.std(axis=0)
    sd[sd < 1e-6] = 1.0
    M = (M - mu) / sd
    fit = M[train_rows]

    explained = 1.0
    if n_pca and n_pca < M.shape[1]:
        centre = fit.mean(axis=0)
        from sklearn.utils.extmath import randomized_svd

        k = min(n_pca, min(fit.shape) - 1)
        _, s, Vt = randomized_svd(fit - centre, n_components=k, random_state=0)
        total = float(((fit - centre) ** 2).sum())
        explained = float((s ** 2).sum() / total) if total else 1.0
        M = (M - centre) @ Vt.T
    return M.astype(np.float32), lookup, explained


def assemble(train: pd.DataFrame, evals: list[pd.DataFrame], checkpoint: str,
             layer: str, pep_rep: str, hla_rep: str, raw: str | None = None,
             pep_pca: int = PEP_PCA) -> tuple[np.ndarray, list[np.ndarray], dict]:
    """Build matched feature matrices for ``train`` and each frame in ``evals``.

    Returns ``(X_train, [X_eval...], info)``. Every transform is fitted on
    ``train`` only.
    """
    pieces_train: list[np.ndarray] = []
    pieces_eval: list[list[np.ndarray]] = [[] for _ in evals]
    info: dict = {"blocks": []}

    for kind, col, rep, n_pca in (
            ("peptide", "peptide", pep_rep, pep_pca if pep_rep == "pos" else 0),
            ("hla", "hla_seq", hla_rep, HLA_PCA)):
        if rep == "none":
            continue
        M, lookup, explained = _fit_block(kind, rep, checkpoint, layer,
                                          train[col].to_numpy(), n_pca)
        pieces_train.append(M[[lookup[str(v)] for v in train[col].to_numpy()]])
        for i, frame in enumerate(evals):
            pieces_eval[i].append(M[[lookup[str(v)] for v in frame[col].to_numpy()]])
        info["blocks"].append({"kind": kind, "rep": rep, "n_columns": M.shape[1],
                               "pca_explained": round(explained, 6)})
        info[f"{kind}_explained"] = round(explained, 6)
        del M
        gc.collect()

    if raw:
        pieces_train.append(build_features(train, "pep_pseudo", raw))
        for i, frame in enumerate(evals):
            pieces_eval[i].append(build_features(frame, "pep_pseudo", raw))
        info["blocks"].append({"kind": "raw", "rep": f"pep_pseudo_{raw}",
                               "n_columns": int(pieces_train[-1].shape[1]),
                               "pca_explained": 1.0})

    if not pieces_train:
        raise ValueError("no feature blocks selected")
    X_train = np.concatenate(pieces_train, axis=1) if len(pieces_train) > 1 else pieces_train[0]
    X_evals = [np.concatenate(p, axis=1) if len(p) > 1 else p[0] for p in pieces_eval]
    info["n_features"] = int(X_train.shape[1])
    info["train_mb"] = round(X_train.nbytes / 1e6, 1)
    return np.ascontiguousarray(X_train), [np.ascontiguousarray(x) for x in X_evals], info


# --- scoring helpers --------------------------------------------------------


def metrics(name: str, val: pd.DataFrame, y_pred: np.ndarray,
            alleles: list[str]) -> dict:
    row = score(name, val.allele, val.y_log1p.to_numpy(), y_pred, alleles).as_row()
    return {k: row[k] for k in ("median_per_allele_spearman", "iqr_low", "iqr_high",
                                "mae_log1p", "median_precision_at_10",
                                "pooled_spearman", "pooled_pearson_log1p")}


def append_runs(records: list[dict]) -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    frame = pd.DataFrame(records)
    if RUNS_CSV.exists():
        frame = pd.concat([pd.read_csv(RUNS_CSV), frame], ignore_index=True)
    frame.to_csv(RUNS_CSV, index=False)


def load_data():
    df = load_with_splits()
    train = df[df.split == "train"]
    val = df[df.split == "val"].sort_values("pair_id").reset_index(drop=True)
    alleles = eligible_alleles(val.allele, val.y_log1p, split="val")
    return df, train, val, alleles


# --- mode: sweep ------------------------------------------------------------


def mode_sweep(args) -> int:
    df, train, val, alleles = load_data()
    fold = inner_folds(train)
    is_fit = (fold == "fit").to_numpy()
    y_train = train.y_log1p.to_numpy()

    combos = [(p, h, L) for (p, h) in REP_COMBOS for L in LAYERS]
    if args.with_raw:
        combos += [(p, h, L, args.with_raw) for (p, h) in REP_COMBOS[:2] for L in LAYERS]

    hidden, l2 = SWEEP_CONFIG
    records = []
    print(f"checkpoint {args.checkpoint}  train={len(train):,} "
          f"(fit={is_fit.sum():,} / dev={(~is_fit).sum():,})  val={len(val):,}  "
          f"{len(alleles)} eligible alleles\n")
    for combo in combos:
        pep, hla, layer = combo[0], combo[1], combo[2]
        raw = combo[3] if len(combo) > 3 else None
        label = rep_label(pep, hla, layer, raw)

        t0 = time.perf_counter()
        X_train, (X_val,), info = assemble(train, [val], args.checkpoint, layer,
                                           pep, hla, raw, pep_pca=args.pep_pca)
        feat_s = time.perf_counter() - t0
        X_fit, X_dev = X_train[is_fit], X_train[~is_fit]
        y_fit, y_dev = y_train[is_fit], y_train[~is_fit]
        shared = {"stage": "sweep", "checkpoint": args.checkpoint, "rep": label,
                  "pep_rep": pep, "hla_rep": hla, "layer": layer,
                  "raw": raw or "", "n_features": info["n_features"],
                  "feature_seconds": round(feat_s, 2), "pep_pca": args.pep_pca,
                  "peptide_explained": info.get("peptide_explained", 1.0),
                  "hla_explained": info.get("hla_explained", 1.0),
                  "n_fit_rows": int(is_fit.sum()), "protocol": "inner_folds"}
        print(f"{label:28s} {info['n_features']:>6,} features "
              f"({info['train_mb']:.0f} MB, {feat_s:.1f}s build)")

        best_alpha, best_dev, p, ridge_s = ridge_path(X_fit, y_fit, X_dev, y_dev,
                                                      X_val)
        at_edge = best_alpha in (min(ESM_ALPHA_GRID), max(ESM_ALPHA_GRID))
        records.append({**shared, "family": "ridge", "config": f"alpha={best_alpha:g}",
                        "seed": 0, "fit_seconds": round(ridge_s, 2),
                        "dev_mse": round(best_dev, 5), "best_epoch": np.nan,
                        "alpha_at_grid_boundary": at_edge,
                        **metrics(f"ridge_{label}", val, p, alleles)})
        print(f"  ridge  alpha={best_alpha:<7g}{' [GRID BOUNDARY]' if at_edge else '':16s} "
              f"rho={records[-1]['median_per_allele_spearman']:+.4f} ({ridge_s:.1f}s)")

        rhos = []
        for seed in SEEDS:
            cfg = MLPConfig(hidden=hidden, l2=l2, seed=seed,
                            max_epochs=MAX_EPOCHS, patience=PATIENCE)
            model = MLPRegressor(cfg).fit(X_fit, y_fit, X_dev, y_dev)
            t0 = time.perf_counter()
            p = model.predict(X_val)
            infer = 1000 * (time.perf_counter() - t0) / len(X_val)
            m = metrics(f"mlp_{label}_s{seed}", val, p, alleles)
            rhos.append(m["median_per_allele_spearman"])
            records.append({**shared, "family": "mlp", "config": cfg.label(),
                            "seed": seed, "fit_seconds": round(model.fit_seconds_, 2),
                            "infer_seconds_per_1k": round(infer, 4),
                            "best_epoch": model.best_epoch_, **m})
        print(f"  mlp    {hidden} l2={l2:g}  rho={np.mean(rhos):+.4f} "
              f"[{min(rhos):+.4f}, {max(rhos):+.4f}] over {len(SEEDS)} seeds "
              f"({np.mean([r['fit_seconds'] for r in records[-3:]]):.0f}s/net)\n")
        del X_train, X_val, X_fit, X_dev
        gc.collect()

    append_runs(records)
    print(f"wrote {RUNS_CSV.relative_to(REPO_ROOT)} (+{len(records)} rows)")
    return 0


# --- mode: grid -------------------------------------------------------------


def mode_grid(args) -> int:
    df, train, val, alleles = load_data()
    fold = inner_folds(train)
    is_fit = (fold == "fit").to_numpy()
    y_train = train.y_log1p.to_numpy()

    records = []
    for layer in args.layers:
        X_train, (X_val,), info = assemble(train, [val], args.checkpoint, layer,
                                           args.pep_rep, args.hla_rep, args.with_raw,
                                           pep_pca=args.pep_pca)
        X_fit, X_dev = X_train[is_fit], X_train[~is_fit]
        y_fit, y_dev = y_train[is_fit], y_train[~is_fit]
        label = rep_label(args.pep_rep, args.hla_rep, layer, args.with_raw)
        print(f"\n{label}: {info['n_features']:,} features")

        # Linear reference on the *extended* alpha ladder. The stage 2 ladder
        # topped out at 1000 and the sweep selected 1000 on nearly every arm --
        # a truncated grid, so the selection was never interior and the ridge
        # numbers it produced understate what a linear head can do here.
        alpha, dev_mse, p, ridge_s = ridge_path(X_fit, y_fit, X_dev, y_dev, X_val)
        at_edge = alpha in (min(ESM_ALPHA_GRID), max(ESM_ALPHA_GRID))
        records.append({"stage": "grid", "checkpoint": args.checkpoint,
                        "rep": label, "pep_rep": args.pep_rep,
                        "hla_rep": args.hla_rep, "layer": layer,
                        "raw": args.with_raw or "", "pep_pca": args.pep_pca,
                        "n_features": info["n_features"], "family": "ridge",
                        "config": f"alpha={alpha:g}", "seed": 0,
                        "protocol": "inner_folds", "dev_mse": round(dev_mse, 5),
                        "alpha_at_grid_boundary": at_edge,
                        "alpha_grid": "|".join(f"{v:g}" for v in ESM_ALPHA_GRID),
                        "n_fit_rows": int(is_fit.sum()),
                        "fit_seconds": round(ridge_s, 2),
                        **metrics(f"ridge_{label}", val, p, alleles)})
        print(f"  ridge alpha={alpha:<8g}{' [GRID BOUNDARY]' if at_edge else '':16s} "
              f"rho={records[-1]['median_per_allele_spearman']:+.4f}")

        l2_grid = tuple(args.l2_grid) if args.l2_grid else ESM_L2_GRID
        for (hidden, l2), seed in itertools.product(
                [(h, l) for h in HIDDEN_GRID for l in l2_grid], SEEDS):
            cfg = MLPConfig(hidden=hidden, l2=l2, seed=seed,
                            max_epochs=MAX_EPOCHS, patience=PATIENCE)
            model = MLPRegressor(cfg).fit(X_fit, y_fit, X_dev, y_dev)
            m = metrics(f"mlp_{label}_{cfg.label()}_s{seed}", val,
                        model.predict(X_val), alleles)
            records.append({"stage": "grid", "checkpoint": args.checkpoint,
                            "rep": label, "pep_rep": args.pep_rep,
                            "hla_rep": args.hla_rep, "layer": layer,
                            "raw": args.with_raw or "", "pep_pca": args.pep_pca,
                            "peptide_explained": info.get("peptide_explained", 1.0),
                            "n_features": info["n_features"], "family": "mlp",
                            "config": cfg.label(), "seed": seed,
                            "l2_grid": "|".join(f"{v:g}" for v in l2_grid),
                            "protocol": "inner_folds",
                            "n_fit_rows": int(is_fit.sum()),
                            "fit_seconds": round(model.fit_seconds_, 2),
                            "best_epoch": model.best_epoch_, **m})
            print(f"  {cfg.label()}_s{seed:<2} rho="
                  f"{m['median_per_allele_spearman']:+.4f} "
                  f"({model.fit_seconds_:.0f}s)")
        del X_train, X_val, X_fit, X_dev
        gc.collect()

    append_runs(records)
    frame = pd.DataFrame(records)
    mlps = frame[frame.family == "mlp"]
    agg = mlps.groupby(["rep", "config"]).median_per_allele_spearman.agg(
        ["mean", "min", "max"]).sort_values("mean", ascending=False)
    print("\nmean validation rho over seeds:")
    print(agg.to_string())
    print("\nboundary check (a selected value at the edge means the ladder was "
          "truncated and must be extended):")
    for rep, grp in mlps.groupby("rep"):
        best = grp.groupby("config").median_per_allele_spearman.mean().idxmax()
        l2 = float(best.split("_l2")[1].split("_lr")[0])
        ladder = tuple(args.l2_grid) if args.l2_grid else ESM_L2_GRID
        edge = l2 in (min(ladder), max(ladder))
        print(f"  {rep:32s} selected l2={l2:g} of {ladder} "
              f"-> {'AT BOUNDARY, extend' if edge else 'interior'}")
    print(f"\nwrote {RUNS_CSV.relative_to(REPO_ROOT)} (+{len(records)} rows)")
    return 0


# --- mode: ensemble ---------------------------------------------------------


def mode_ensemble(args) -> int:
    df, train, val, alleles = load_data()
    fold = cv_folds(train, N_FOLDS)          # identical assignment to stage 2
    y_train = train.y_log1p.to_numpy()
    y_val = val.y_log1p.to_numpy()

    selected = json.loads(Path(args.selected).read_text()) if args.selected else {}

    # The two-way axis the 30 members are spread over. Stage 2 spreads its 30
    # networks over {one-hot, BLOSUM62}; this arm has to spread its 30 over
    # *something* comparable, and which something depends on the question.
    #
    #   --axis layer     the ESM-only arm. Members differ by which ESM-2 layer
    #                    they read (middle / final). Answers "can frozen ESM-2
    #                    features REPLACE sequence features?"
    #
    #   --axis encoding  the additive arm, and the decision-relevant one.
    #                    Members differ by the raw encoding exactly as stage 2's
    #                    do -- same folds, same encodings, same seeds -- with the
    #                    ESM block appended to every one of them. The arm then
    #                    differs from `scripts/baseline_ensemble.py` by exactly
    #                    one thing: the ESM features. Answers "do frozen ESM-2
    #                    features IMPROVE on sequence features?", which is the
    #                    question the challenge actually poses. A representation
    #                    can be worse standalone and still carry information the
    #                    baseline lacks, so these two arms can disagree and the
    #                    additive one is the one that decides.
    if args.axis == "encoding":
        groups = [(enc, args.layers[0], enc) for enc in ENCODINGS]
        axis_desc = f"{len(ENCODINGS)} raw encodings (ESM layer fixed at {args.layers[0]})"
    else:
        groups = [(L, L, args.with_raw) for L in args.layers]
        axis_desc = f"{len(args.layers)} ESM layers"

    n_members = N_FOLDS * len(groups) * len(SEEDS)
    print(f"checkpoint {args.checkpoint}")
    print(f"ensemble: {N_FOLDS} CV folds x {axis_desc} x "
          f"{len(SEEDS)} seeds = {n_members} networks "
          f"(stage 2 baseline: {N_FOLDS} x 2 encodings x {len(SEEDS)} = 30)")
    print(f"fold sizes: {np.bincount(fold).tolist()}\n")

    members, records = [], []
    infer_seconds = 0.0          # summed over all members, for cost per 1k rows
    feature_seconds = 0.0
    started = time.perf_counter()
    for key, layer, raw in groups:
        hidden, l2 = tuple(selected.get(key, SWEEP_CONFIG))
        hidden = tuple(hidden)
        t_feat = time.perf_counter()
        X_train, (X_val,), info = assemble(train, [val], args.checkpoint, layer,
                                           args.pep_rep, args.hla_rep, raw,
                                           pep_pca=args.pep_pca)
        feature_seconds += time.perf_counter() - t_feat
        label = rep_label(args.pep_rep, args.hla_rep, layer, raw)
        print(f"{label}: {info['n_features']:,} features, config h{hidden} l2={l2:g}")
        for k in range(N_FOLDS):
            is_fit = fold != k
            for seed in SEEDS:
                cfg = MLPConfig(hidden=hidden, l2=l2, seed=seed,
                                max_epochs=MAX_EPOCHS, patience=PATIENCE)
                model = MLPRegressor(cfg).fit(X_train[is_fit], y_train[is_fit],
                                              X_train[~is_fit], y_train[~is_fit])
                t_inf = time.perf_counter()
                p = model.predict(X_val)
                infer_seconds += time.perf_counter() - t_inf
                members.append(p)
                m = metrics(f"member_{label}_f{k}_s{seed}", val, p, alleles)
                records.append({"stage": "ensemble_member", "checkpoint": args.checkpoint,
                                "rep": label, "pep_rep": args.pep_rep,
                                "hla_rep": args.hla_rep, "layer": layer,
                                "raw": raw or "", "family": "mlp",
                                "config": cfg.label(), "seed": seed, "fold": k,
                                "protocol": "cv_folds",
                                "n_features": info["n_features"],
                                "n_fit_rows": int(is_fit.sum()),
                                "fit_seconds": round(model.fit_seconds_, 2),
                                "best_epoch": model.best_epoch_, **m})
        done = [r for r in records if r["rep"] == label]
        print(f"  {len(done)} networks, member rho "
              f"{np.mean([r['median_per_allele_spearman'] for r in done]):.3f} "
              f"(min {min(r['median_per_allele_spearman'] for r in done):.3f}, "
              f"max {max(r['median_per_allele_spearman'] for r in done):.3f}), "
              f"{np.mean([r['fit_seconds'] for r in done]):.0f}s/net")
        del X_train, X_val
        gc.collect()
    wall = time.perf_counter() - started

    ens = np.mean(members, axis=0)
    name = args.name or f"esm_{args.checkpoint.split('_')[1]}_{args.axis}"
    s = score(name, val.allele, y_val, ens, alleles)
    member_mean = float(np.mean([r["median_per_allele_spearman"] for r in records]))
    print(f"\naverage single member : median rho {member_mean:.4f}")
    print(f"{n_members}-network ensemble: median rho {s.median_spearman:.4f} "
          f"(ensembling gain {s.median_spearman - member_mean:+.4f})")
    print(pd.DataFrame([s.as_row()]).to_string(index=False))

    # Head-side inference cost, kept separate from embedding extraction (which
    # reports/stage3_embedding_cost.csv already carries). This is what it costs
    # to score 1,000 already-embedded pairs with the whole ensemble.
    infer_per_1k = 1000 * infer_seconds / len(val)
    feat_per_1k = 1000 * feature_seconds / (len(train) + len(val))
    print(f"\nhead cost: {n_members} networks, "
          f"{wall / 60:.1f} min total fit, "
          f"{1000 * feature_seconds / (len(train) + len(val)):.4f}s feature assembly "
          f"per 1,000 rows, {infer_per_1k:.4f}s ensemble inference per 1,000 rows")

    PRED_DIR.mkdir(exist_ok=True)
    dest = PRED_DIR / f"{name}.csv"
    pd.DataFrame({"pair_id": val.pair_id.to_numpy(), "y_pred": ens}).to_csv(dest, index=False)

    records.append({"stage": "ensemble", "checkpoint": args.checkpoint,
                    "rep": f"{args.pep_rep}/{args.hla_rep}/axis={args.axis}",
                    "pep_rep": args.pep_rep, "hla_rep": args.hla_rep,
                    "layer": "+".join(args.layers), "raw": args.with_raw or "",
                    "axis": args.axis,
                    "family": "mlp_ensemble", "config": f"{n_members} networks",
                    "seed": -1, "protocol": "cv_folds",
                    "n_features": info["n_features"],
                    "n_members": n_members,
                    "fit_seconds": round(wall, 1),
                    "infer_seconds_per_1k": round(infer_per_1k, 4),
                    "feature_seconds_per_1k": round(feat_per_1k, 4),
                    **{k: v for k, v in score(name, val.allele, y_val, ens,
                                              alleles).as_row().items()
                       if k in ("median_per_allele_spearman", "iqr_low", "iqr_high",
                                "mae_log1p", "median_precision_at_10",
                                "pooled_spearman", "pooled_pearson_log1p")}})
    append_runs(records)

    # --- matched comparison against the stage 2 ensembles -------------------
    comparisons = []
    for base_name, base_file in (("seq_ensemble_pep_pseudo", "seq_ensemble_pep_pseudo.csv"),
                                 ("seq_ensemble_pep_domain", "seq_ensemble_pep_domain.csv")):
        path = PRED_DIR / base_file
        if not path.exists():
            print(f"  (missing {base_file}; skipping)")
            continue
        base = pd.read_csv(path)
        merged = val[["pair_id"]].merge(base, on="pair_id", how="left", validate="one_to_one")
        if merged.y_pred.isna().any():
            raise ValueError(f"{base_file} does not cover every validation row")
        b = merged.y_pred.to_numpy()
        bs = score(base_name, val.allele, y_val, b, alleles)
        res = paired_cluster_bootstrap(val.cluster_id, val.allele, y_val, b, ens,
                                       alleles=alleles)
        lo, hi = res["ci95"]
        print(f"\n{name} vs {base_name} (paired cluster bootstrap, "
              f"{res['n_boot']} resamples, same {len(alleles)} alleles, "
              f"same {len(val):,} rows):")
        print(f"  baseline median rho {bs.median_spearman:.4f}  "
              f"esm {s.median_spearman:.4f}  "
              f"delta {res['delta_median_spearman']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        print(f"  verdict: {describe_delta(res)}")
        comparisons.append({
            "esm_arm": name, "baseline": base_name, "n_rows": len(val),
            "n_alleles": len(alleles),
            "baseline_median_rho": round(bs.median_spearman, 4),
            "baseline_iqr_low": round(bs.iqr_spearman[0], 4),
            "baseline_iqr_high": round(bs.iqr_spearman[1], 4),
            "esm_median_rho": round(s.median_spearman, 4),
            "esm_iqr_low": round(s.iqr_spearman[0], 4),
            "esm_iqr_high": round(s.iqr_spearman[1], 4),
            "delta": round(res["delta_median_spearman"], 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "verdict": describe_delta(res),
            "baseline_mae": round(bs.mae_log1p, 4), "esm_mae": round(s.mae_log1p, 4),
            "baseline_p10": round(bs.median_precision_at_k, 4),
            "esm_p10": round(s.median_precision_at_k, 4),
            "n_members_esm": n_members, "n_members_baseline": 30,
            "esm_infer_seconds_per_1k": round(infer_per_1k, 4),
        })

    if comparisons:
        cmp_path = REPORT_DIR / "stage3_comparisons.csv"
        frame = pd.DataFrame(comparisons)
        if cmp_path.exists():
            old = pd.read_csv(cmp_path)
            old = old[~old.esm_arm.isin(frame.esm_arm.unique())]
            frame = pd.concat([old, frame], ignore_index=True)
        frame.to_csv(cmp_path, index=False)
        print(f"\nwrote {cmp_path.relative_to(REPO_ROOT)}")
    print(f"wrote {dest.relative_to(REPO_ROOT)}")
    print(f"total fit wall time: {wall / 60:.1f} min "
          f"({wall / n_members:.0f}s per network)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p):
        p.add_argument("--checkpoint", default=pesm.DEFAULT_CHECKPOINT,
                       choices=sorted(pesm.CHECKPOINTS))
        p.add_argument("--with-raw", default=None, choices=["onehot", "blosum"],
                       help="concatenate the stage 2 peptide+pseudosequence features")
        p.add_argument("--pep-pca", type=int, default=PEP_PCA,
                       help="PCA components for the peptide per-position block; "
                            "0 keeps all 9*dim columns (the uncompressed control)")

    s = sub.add_parser("sweep"); common(s)
    g = sub.add_parser("grid"); common(g)
    g.add_argument("--pep-rep", default="pos", choices=pesm.PEPTIDE_REPS)
    g.add_argument("--hla-rep", default="contact", choices=pesm.HLA_REPS)
    g.add_argument("--layers", nargs="+", default=list(LAYERS))
    g.add_argument("--l2-grid", nargs="+", type=float, default=None,
                   help=f"MLP L2 ladder (default {ESM_L2_GRID}); same number of "
                        "grid points as stage 2, ranges chosen for the feature scale")
    e = sub.add_parser("ensemble"); common(e)
    e.add_argument("--pep-rep", default="pos", choices=pesm.PEPTIDE_REPS)
    e.add_argument("--hla-rep", default="contact", choices=pesm.HLA_REPS)
    e.add_argument("--layers", nargs="+", default=list(LAYERS))
    e.add_argument("--axis", default="layer", choices=["layer", "encoding"],
                   help="what the 30 members are spread over: 'layer' for the "
                        "ESM-only arm, 'encoding' for the additive "
                        "baseline+ESM arm (see mode_ensemble)")
    e.add_argument("--selected", default=None,
                   help="JSON mapping axis key -> [hidden, l2] from the grid")
    e.add_argument("--name", default=None)

    args = ap.parse_args()
    return {"sweep": mode_sweep, "grid": mode_grid, "ensemble": mode_ensemble}[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
