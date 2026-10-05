"""Phase C §6.2: frozen joint-chain vs separate-chain representations.

    .venv/bin/python scripts/phase_c_frozen_comparison.py time --arm joint
    .venv/bin/python scripts/phase_c_frozen_comparison.py fit  --arm joint
    .venv/bin/python scripts/phase_c_frozen_comparison.py fit  --arm separate
    .venv/bin/python scripts/phase_c_frozen_comparison.py fit  --arm esm2
    .venv/bin/python scripts/phase_c_frozen_comparison.py compare

Three frozen arms, one head, one feature contract (9 per-position peptide
vectors + the mean-pooled partner = 10 x 1,280 = 12,800 dims):

* ``joint``    -- base MINT, both chains in one forward pass, chains attend
* ``separate`` -- base MINT, each chain in its own pass, chains do not attend
* ``esm2``     -- ESM-2 650M, each chain in its own pass (already cached)

``joint`` vs ``separate`` isolates cross-chain context at inference with the
weights held fixed. ``separate`` vs ``esm2`` isolates the STRING PPI pretraining
with the chain treatment held fixed. ``joint`` vs ``esm2`` is the headline and is
confounded by both, which is exactly what §6.2 warned about; it is reported as
such and not as evidence about cross-attention.

**Protocol inherited, not declared** (`phase_c_gate.md` §8.1):

* Outer folds loaded from ``phase_a_outer_folds.csv``; never recomputed.
* Early stopping and the L2 ladder use inner cluster folds strictly inside each
  outer fitting partition. No outer-fold label chooses an epoch or an L2.
* Scoring and intervals from ``pepstab.evaluation`` verbatim -- 2,000 resamples
  at seed 20261003, read through ``describe_delta``.
* Primary metric on **uncalibrated** predictions. No calibration map is fitted
  anywhere in this script, so none can reach the ranking path.
* Equal tuning budget per arm, selected per arm, with a 1-SE parsimony rule
  because an L2 ladder on three same-geometry arms is expected to be flat.
* Two paired seeds, averaged identically across arms.

The out-of-fold predictions are the deliverable, not just the summary: they are
what a later blend experiment (§6.5) needs, and they are written per arm so the
comparison can be re-read without refitting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pepstab import evaluation as ev  # noqa: E402
from pepstab.mlp import MLPConfig, MLPRegressor  # noqa: E402

OUT = ROOT / "reports" / "further_research"
MINT_FEATURES = ROOT / "features" / "mint" / "mint_base"
ESM_CHECKPOINT = "esm2_t33_650M_UR50D"
ARMS = ("joint", "separate", "esm2")

#: Equal budget for every arm. Four L2 values, selected per arm on inner folds
#: with a 1-SE parsimony rule -- not a transplanted ladder, which cost ESM-2
#: 0.093 SCC elsewhere in this project.
L2_LADDER = (1e-5, 1e-4, 1e-3, 1e-2)
#: Declared **extension**, not a silent expansion -- §3.3 requires logging one.
#: The prespecified ladder above had its 1-SE rule select the boundary value 1e-2
#: on 12 of 15 arm-folds, and unevenly (separate 5/5, esm2 4/5, joint 3/5), which
#: says the optimum lies outside the grid and that the handicap is not symmetric.
#: Since `separate` is both the worst arm and the most boundary-pinned, the
#: truncation plausibly inflates BOTH significant contrasts, so it is tested
#: rather than caveated. Select with --ladder extended.
L2_LADDER_EXTENDED = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
SEEDS = (0, 1)
#: Inner folds inside each outer fitting partition, whole-cluster. Five, so the
#: early-stopping fold is 20% -- the dev fraction the project already uses --
#: rather than the 33% a three-way split would hold back from fitting.
INNER_FOLDS = 5
#: How many of those folds the L2 ladder is scored on. Three, not five: §3.3
#: asks for a bounded screen, and the budget is declared here before any
#: number is read rather than extended if the ladder disappoints.
LADDER_FOLDS = 3


def load_rows() -> pd.DataFrame:
    """Dev rows with labels, clusters, and the inherited outer folds.

    Joined on ``(allele, peptide)`` per EVALUATION.md. The outer-fold file is
    joined on ``pair_id``, which is safe *only* because both it and
    ``data/splits.csv`` are keyed positionally into the same raw CSV; the
    prohibition in CLAUDE.md is about joining ``pair_id`` across files with
    different row counts. Asserted below rather than trusted.
    """
    splits = pd.read_csv(ROOT / "data" / "splits.csv")
    raw = pd.read_csv(ROOT / "data" / "rasmussen_et_al_dataset.csv")
    dev = splits[splits.split.isin(["train", "val"])].merge(
        raw[["allele", "peptide", "thalf_hours", "hla_seq"]],
        on=["allele", "peptide"], how="left", validate="one_to_one",
    )
    assert dev.thalf_hours.notna().all() and dev.hla_seq.notna().all()
    dev["y"] = np.log1p(dev.thalf_hours)

    folds = pd.read_csv(OUT / "phase_a_outer_folds.csv")
    n_train = int((splits.split == "train").sum())
    assert len(folds) == n_train, f"fold file has {len(folds)} rows, train has {n_train}"
    dev = dev.merge(folds, on="pair_id", how="left", validate="one_to_one")
    assert dev.loc[dev.split == "train", "outer_fold"].notna().all(), "train rows missing a fold"
    assert dev.loc[dev.split == "val", "outer_fold"].isna().all(), "val rows got a fold"
    return dev


def inner_fold_ids(frame: pd.DataFrame, n_folds: int) -> np.ndarray:
    """Whole-cluster inner folds, deterministic, aligned to ``frame``'s rows.

    Clusters are the independence unit, so an inner split that cut through one
    would let an early-stopping decision see a near-duplicate of a fitting row.
    """
    clusters = np.sort(frame.cluster_id.unique())
    assign = {c: i % n_folds for i, c in enumerate(clusters)}
    return frame.cluster_id.map(assign).to_numpy()


def features(arm: str, dev: pd.DataFrame) -> np.ndarray:
    """``(n, 12800)`` float32 for one arm. One arm at a time, by design.

    All three arms are loaded as float32 for the head regardless of how they were
    stored: the MINT features were *computed* in fp32 and stored in fp16 to halve
    the egress bill (`mint_features.py`), which is a storage choice, not a
    precision arm. §5.5's rule is that dtype must not differ between arms, and it
    does not -- the head sees float32 in every case.
    """
    if arm == "esm2":
        from pepstab import esm

        pep = esm.peptide_block(dev.peptide.tolist(), ESM_CHECKPOINT, "final", "pos")
        hla = esm.hla_block(dev.hla_seq.tolist(), ESM_CHECKPOINT, "final", "mean")
        X = np.hstack([pep, hla])
    elif arm == "separate":
        pep_idx = json.loads((MINT_FEATURES / "sep_peptide.index.json").read_text())["sequences"]
        hla_idx = json.loads((MINT_FEATURES / "sep_hla.index.json").read_text())["sequences"]
        pep_reps = np.load(MINT_FEATURES / "sep_peptide_L33.npz")["reps"]
        hla_reps = np.load(MINT_FEATURES / "sep_hla_L33.npz")["reps"]
        pep_at = {s: i for i, s in enumerate(pep_idx)}
        hla_at = {s: i for i, s in enumerate(hla_idx)}
        pi = np.array([pep_at[p] for p in dev.peptide])
        hi = np.array([hla_at[h] for h in dev.hla_seq])
        X = np.hstack([
            pep_reps[pi].reshape(len(pi), -1).astype(np.float32),
            hla_reps[hi].astype(np.float32),
        ])
    elif arm == "joint":
        keys = json.loads((MINT_FEATURES / "joint.index.json").read_text())["keys"]
        reps = np.load(MINT_FEATURES / "joint_L33.npz")["reps"]
        at = {k: i for i, k in enumerate(keys)}
        want = [f"{a}|{p}" for a, p in zip(dev.allele, dev.peptide)]
        missing = [k for k in want if k not in at]
        assert not missing, f"{len(missing)} pairs absent from the joint cache, e.g. {missing[:2]}"
        idx = np.array([at[k] for k in want])
        X = reps[idx].reshape(len(idx), -1).astype(np.float32)
    else:
        raise ValueError(f"arm must be one of {ARMS}")

    assert X.shape[1] == 12800, f"{arm}: expected 12800 dims, got {X.shape[1]}"
    assert np.isfinite(X).all(), f"{arm}: non-finite features"
    return np.ascontiguousarray(X, dtype=np.float32)


def standardise(fit: np.ndarray, apply_to: list[np.ndarray]) -> list[np.ndarray]:
    """Centre and scale using **fitting-partition** statistics only.

    §3.2 requires every feature transform to be fitted inside training. A
    standardiser fitted on all rows would leak the outer fold's distribution.
    """
    mu = fit.mean(axis=0, keepdims=True)
    sd = fit.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return [((a - mu) / sd).astype(np.float32) for a in apply_to]


def one_fit(X_fit, y_fit, X_dev, y_dev, l2: float, seed: int,
            hidden: tuple[int, ...], max_epochs: int, patience: int) -> MLPRegressor:
    cfg = MLPConfig(hidden=hidden, l2=l2, lr=1e-3, batch_size=256,
                    max_epochs=max_epochs, patience=patience, seed=seed)
    return MLPRegressor(cfg).fit(X_fit, y_fit, X_dev, y_dev)


def pick_l2(scores: dict[float, list[float]]) -> tuple[float, dict]:
    """1-SE parsimony: the strongest regularisation within 1 SE of the best.

    An L2 ladder across three same-geometry arms is expected to be flat, and
    argmax on a flat ladder chases the grid boundary. The rule is declared here
    before any number is read.
    """
    means = {l2: float(np.mean(v)) for l2, v in scores.items()}
    ses = {l2: float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
           for l2, v in scores.items()}
    best = max(means, key=means.get)
    threshold = means[best] - ses[best]
    eligible = [l2 for l2 in sorted(means) if means[l2] >= threshold]
    chosen = max(eligible)  # strongest regularisation that is not meaningfully worse
    return chosen, {"means": means, "ses": ses, "argmax": best,
                    "threshold": round(threshold, 5), "chosen": chosen}


def inner_score(y_true, y_pred, allele) -> float:
    """Median per-allele Spearman on an inner fold, project scoring rules."""
    alleles = ev.eligible_alleles(allele, y_true, split="val")
    if not alleles:
        return float("-inf")
    rho = ev.per_allele_spearman(allele, y_true, y_pred, alleles)
    panel = ev.panel_spearman(rho, ev.rankable_alleles(allele, y_true, alleles))
    return float(panel.median()) if len(panel) else float("-inf")


def run_arm(arm: str, dev: pd.DataFrame, hidden: tuple[int, ...],
            max_epochs: int, patience: int,
            ladder: tuple[float, ...] = L2_LADDER) -> tuple[pd.DataFrame, dict]:
    """Nested out-of-fold predictions for one arm over the inherited outer folds."""
    X = features(arm, dev)
    train = dev[dev.split == "train"].reset_index(drop=True)
    train_pos = dev.index[dev.split == "train"].to_numpy()

    oof = np.full(len(train), np.nan)
    selection, timings = [], []

    for fold in sorted(train.outer_fold.unique()):
        held = (train.outer_fold == fold).to_numpy()
        fit_rows = train[~held].reset_index(drop=True)
        fit_pos = train_pos[~held]
        out_pos = train_pos[held]

        inner = inner_fold_ids(fit_rows, INNER_FOLDS)
        t0 = time.monotonic()

        # --- L2 ladder on inner folds, equal budget, no outer label involved --
        scores: dict[float, list[float]] = {l2: [] for l2 in ladder}
        for k in range(LADDER_FOLDS):
            dev_mask = inner == k
            Xi_fit, Xi_dev = standardise(
                X[fit_pos][~dev_mask], [X[fit_pos][~dev_mask], X[fit_pos][dev_mask]]
            )
            yi_fit = fit_rows.y.to_numpy()[~dev_mask]
            yi_dev = fit_rows.y.to_numpy()[dev_mask]
            for l2 in ladder:
                m = one_fit(Xi_fit, yi_fit, Xi_dev, yi_dev, l2, SEEDS[0],
                            hidden, max_epochs, patience)
                scores[l2].append(
                    inner_score(yi_dev, m.predict(Xi_dev), fit_rows.allele[dev_mask])
                )
        l2, detail = pick_l2(scores)

        # --- finalist: paired seeds, averaged, inner fold 0 stops the epoch ---
        dev_mask = inner == 0
        Xf, Xd, Xo = standardise(
            X[fit_pos][~dev_mask],
            [X[fit_pos][~dev_mask], X[fit_pos][dev_mask], X[out_pos]],
        )
        yf = fit_rows.y.to_numpy()[~dev_mask]
        yd = fit_rows.y.to_numpy()[dev_mask]
        preds, epochs = [], []
        for seed in SEEDS:
            m = one_fit(Xf, yf, Xd, yd, l2, seed, hidden, max_epochs, patience)
            preds.append(m.predict(Xo))
            epochs.append(m.best_epoch_)
        oof[held] = np.mean(preds, axis=0)

        elapsed = time.monotonic() - t0
        timings.append(elapsed)
        selection.append({"arm": arm, "outer_fold": int(fold), "l2_chosen": l2,
                          "best_epochs": epochs, "seconds": round(elapsed, 1),
                          "fit_rows": int((~dev_mask).sum()),
                          "inner_dev_rows": int(dev_mask.sum()),
                          "outer_rows": int(held.sum()), **detail})
        print(f"  {arm} fold {fold}: l2={l2:g} epochs={epochs} "
              f"{elapsed:.0f}s  (argmax was {detail['argmax']:g})")

    assert np.isfinite(oof).all(), "some training row never received a prediction"
    out = train[["pair_id", "allele", "peptide", "cluster_id", "outer_fold", "y"]].copy()
    out["y_pred"] = oof
    return out, {"arm": arm, "hidden": list(hidden), "max_epochs": max_epochs,
                 "patience": patience, "l2_ladder": list(ladder),
                 "seeds": list(SEEDS), "inner_folds": INNER_FOLDS,
                 "ladder_folds": LADDER_FOLDS,
                 "selection": selection, "total_seconds": round(sum(timings), 1)}


def cmd_time(args) -> None:
    """One fit, measured, so the grid is sized from a number and not a guess."""
    dev = load_rows()
    X = features(args.arm, dev)
    train = dev[dev.split == "train"].reset_index(drop=True)
    pos = dev.index[dev.split == "train"].to_numpy()
    inner = inner_fold_ids(train, INNER_FOLDS)
    dev_mask = inner == 0
    Xf, Xd = standardise(X[pos][~dev_mask], [X[pos][~dev_mask], X[pos][dev_mask]])
    print(f"{args.arm}: X {X.shape} {X.dtype}, fit {Xf.shape[0]:,} rows, "
          f"inner dev {Xd.shape[0]:,}")
    t0 = time.monotonic()
    m = one_fit(Xf, train.y.to_numpy()[~dev_mask], Xd, train.y.to_numpy()[dev_mask],
                1e-4, 0, tuple(args.hidden), args.max_epochs, args.patience)
    elapsed = time.monotonic() - t0
    n_fits = len(L2_LADDER) * LADDER_FOLDS + len(SEEDS)
    print(f"one fit: {elapsed:.1f} s, stopped at epoch {m.best_epoch_}, "
          f"{m.n_parameters():,} head params")
    print(f"per arm: {n_fits} fits x 5 outer folds = {n_fits * 5} fits "
          f"-> {n_fits * 5 * elapsed / 60:.0f} min")
    print(f"all three arms: {3 * n_fits * 5 * elapsed / 3600:.1f} h on this machine")


def cmd_fit(args) -> None:
    dev = load_rows()
    print(f"{args.arm}: fitting over 5 inherited outer folds")
    ladder = L2_LADDER_EXTENDED if args.ladder == "extended" else L2_LADDER
    suffix = "" if args.ladder == "prespecified" else f"_{args.ladder}"
    preds, manifest = run_arm(args.arm, dev, tuple(args.hidden),
                              args.max_epochs, args.patience, ladder=ladder)
    manifest["ladder_label"] = args.ladder
    OUT.mkdir(parents=True, exist_ok=True)
    preds.to_csv(OUT / f"phase_c_oof_{args.arm}{suffix}.csv", index=False)
    (OUT / f"phase_c_oof_{args.arm}{suffix}.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    s = ev.score(args.arm, preds.allele, preds.y.to_numpy(), preds.y_pred.to_numpy())
    print(f"\n{args.arm}: median per-allele Spearman {s.median_spearman:.4f} "
          f"over {s.n_alleles_scored} alleles, {s.n_rows:,} rows, "
          f"MAE(log1p) {s.mae_log1p:.4f}")
    print(f"wrote phase_c_oof_{args.arm}{suffix}.csv ({manifest['total_seconds']:.0f} s)")


def cmd_compare(args) -> None:
    """The §6.2 comparison. Reads the three OOF files; refits nothing."""
    frames = {}
    for arm in ARMS:
        suffix = "" if args.ladder == "prespecified" else f"_{args.ladder}"
        p = OUT / f"phase_c_oof_{arm}{suffix}.csv"
        if not p.exists():
            print(f"missing {p.name} -- run `fit --arm {arm}` first")
            return
        frames[arm] = pd.read_csv(p).sort_values("pair_id").reset_index(drop=True)

    base = frames[ARMS[0]]
    for arm, f in frames.items():
        assert (f.pair_id.to_numpy() == base.pair_id.to_numpy()).all(), f"{arm} row order"

    y = base.y.to_numpy()
    # One panel for every arm, fixed by the labels alone, so no arm can change
    # the denominator it is scored on.
    panel = ev.eligible_alleles(base.allele, y, split="val")
    rows = []
    for arm, f in frames.items():
        s = ev.score(arm, f.allele, y, f.y_pred.to_numpy(), alleles=panel)
        rows.append({"arm": arm, "median_spearman": round(s.median_spearman, 4),
                     "iqr_low": round(s.iqr_spearman[0], 4),
                     "iqr_high": round(s.iqr_spearman[1], 4),
                     "mae_log1p": round(s.mae_log1p, 4),
                     "pooled_spearman": round(s.pooled_spearman, 4),
                     "median_precision_at_k": round(s.median_precision_at_k, 4),
                     "n_alleles_scored": s.n_alleles_scored,
                     "n_alleles_unranked": s.n_alleles_unranked,
                     "n_rows": s.n_rows})
    summary = pd.DataFrame(rows)

    contrasts = [("separate", "joint", "cross-chain context at inference (weights fixed)"),
                 ("esm2", "separate", "STRING PPI pretraining (chain treatment fixed)"),
                 ("esm2", "joint", "headline -- CONFOUNDED by both")]
    deltas = []
    for a, b, isolates in contrasts:
        res = ev.paired_cluster_bootstrap(
            base.cluster_id, base.allele, y,
            frames[a].y_pred.to_numpy(), frames[b].y_pred.to_numpy(), alleles=panel,
        )
        lo, hi = res["ci95"]
        deltas.append({"baseline": a, "comparison": b, "isolates": isolates,
                       "delta": round(res["delta_median_spearman"], 4),
                       "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                       "verdict": ev.describe_delta(res),
                       "n_boot": res.get("n_boot"), "seed": ev.BOOTSTRAP_SEED,
                       # Reported rather than hidden inside the CI: a large share
                       # means the interval rests on shifting allele panels.
                       "n_resamples_with_dropped_alleles":
                           res.get("n_resamples_with_dropped_alleles"),
                       "n_degenerate_resamples": res.get("n_degenerate_resamples"),
                       "panel_size_full": res.get("panel_size_full")})
    delta_frame = pd.DataFrame(deltas)

    # The difference of medians is not the median of differences, and on this
    # metric they can disagree badly. The primary metric is a median over a fixed
    # allele panel, so a gap can come from *where each arm's median allele sits*
    # rather than from a consistent per-allele difference -- the two arms' rho
    # distributions can be nearly identical and still show a large median gap.
    #
    # The tell is a bootstrap CI that excludes its own point estimate. When that
    # happens the difference of medians is not describing a per-allele effect and
    # `describe_delta`'s verdict should not be read as one. The paired sign test
    # below carries no distributional assumption and is the honest check.
    rank = ev.rankable_alleles(base.allele, y, panel)
    rho = {arm: ev.panel_spearman(
        ev.per_allele_spearman(base.allele, y, f.y_pred.to_numpy(), panel), rank)
        for arm, f in frames.items()}
    paired = []
    for a, b_, isolates in contrasts:
        d = (rho[b_] - rho[a]).dropna()
        common = d.index
        try:
            from scipy.stats import wilcoxon
            p = float(wilcoxon(rho[b_].reindex(common), rho[a].reindex(common)).pvalue)
        except Exception:
            p = float("nan")
        row = delta_frame[(delta_frame.baseline == a) & (delta_frame.comparison == b_)].iloc[0]
        inside = row.ci95_low <= row.delta <= row.ci95_high
        paired.append({
            "baseline": a, "comparison": b_,
            "diff_of_medians": row.delta,
            "median_of_paired_diffs": round(float(d.median()), 4),
            "mean_of_paired_diffs": round(float(d.mean()), 4),
            "alleles_favouring_comparison": int((d > 0).sum()),
            "alleles_compared": len(d),
            "wilcoxon_p": round(p, 6),
            "point_estimate_inside_ci": bool(inside),
            # Set when the two functionals disagree about whether there is an
            # effect at all. Then the median gap is a placement artefact.
            "median_placement_artefact": bool(not inside or (p > 0.05 and abs(row.delta) > 0.03)),
        })
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(OUT / f"phase_c_frozen_paired_diagnostic{suffix}.csv", index=False)

    # Per fold as well as pooled. A0's fold medians span 0.048 with the model
    # held fixed -- essentially the whole 0.05 worthwhile bar -- so a single-fold
    # gap between two arms is fold variation until shown otherwise, and a pooled
    # number alone hides that.
    per_fold = []
    for fold in sorted(base.outer_fold.unique()):
        m = (base.outer_fold == fold).to_numpy()
        row = {"outer_fold": int(fold), "rows": int(m.sum()),
               "alleles": int(base.allele[m].nunique()),
               "clusters": int(base.cluster_id[m].nunique())}
        # The panel is refixed per fold from that fold's labels alone, so no arm
        # can change the denominator it is scored on within the fold either.
        fold_panel = ev.eligible_alleles(base.allele[m], y[m], split="val")
        row["panel_alleles"] = len(fold_panel)
        for arm, f in frames.items():
            if fold_panel:
                s = ev.score(arm, base.allele[m], y[m], f.y_pred.to_numpy()[m],
                             alleles=fold_panel)
                row[arm] = round(s.median_spearman, 4)
            else:
                row[arm] = float("nan")
        per_fold.append(row)
    fold_frame = pd.DataFrame(per_fold)
    spans = {a: round(fold_frame[a].max() - fold_frame[a].min(), 4) for a in ARMS}

    cov = fold_frame

    summary.to_csv(OUT / f"phase_c_frozen_summary{suffix}.csv", index=False)
    delta_frame.to_csv(OUT / f"phase_c_frozen_deltas{suffix}.csv", index=False)
    cov.to_csv(OUT / f"phase_c_frozen_coverage{suffix}.csv", index=False)

    print(summary.to_string(index=False))
    print()
    print(delta_frame.to_string(index=False))
    print()
    print(paired_frame.to_string(index=False))
    for r in paired_frame.itertuples():
        if r.median_placement_artefact:
            print(f"\n  !! {r.baseline} vs {r.comparison}: difference of medians "
                  f"{r.diff_of_medians:+.4f} but median paired per-allele difference "
                  f"only {r.median_of_paired_diffs:+.4f} (Wilcoxon p={r.wilcoxon_p:g}, "
                  f"{r.alleles_favouring_comparison}/{r.alleles_compared} alleles favour "
                  f"{r.comparison}).")
            print(f"     The median gap is a PLACEMENT ARTEFACT, not a per-allele effect. "
                  f"Point estimate inside its own CI: {r.point_estimate_inside_ci}.")
            print(f"     Do not read describe_delta's verdict as an effect here.")
    print(f"\npanel: {len(panel)} alleles, >=20 rows and >=2 distinct labels")
    print("\nper outer fold (panel refixed per fold from that fold's labels):")
    print(cov.to_string(index=False))
    print(f"\nfold-to-fold span per arm: {spans}")
    print("A0 spans 0.048 across these same folds with the model fixed, so a gap")
    print("smaller than an arm's own span is not a result.")
    print("\nwrote phase_c_frozen_{summary,deltas,coverage,paired_diagnostic}.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("time", cmd_time), ("fit", cmd_fit), ("compare", cmd_compare)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        if name != "compare":
            p.add_argument("--arm", required=True, choices=ARMS)
            p.add_argument("--hidden", type=int, nargs="+", default=[64])
            p.add_argument("--max-epochs", type=int, default=60)
            p.add_argument("--patience", type=int, default=8)
            p.add_argument("--ladder", default="prespecified",
                           choices=("prespecified", "extended"))
        else:
            p.add_argument("--ladder", default="prespecified",
                           choices=("prespecified", "extended"))
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
