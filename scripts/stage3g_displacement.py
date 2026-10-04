"""Stage 3g: why a width-matched uninformative block is not a capacity control.

    .venv/bin/python scripts/stage3g_displacement.py diagnose

**Result: this experiment does not work, and the reason is the finding.** The
file is kept so the dead end is recorded rather than rediscovered, and so the
diagnostic that killed it is reproducible.

The intent was to settle whether the shipped additive arm's −0.0170 is capacity
displacement by appending a block that is width-matched but uninformative. It
cannot be settled that way. A 330-column block that carries no information is
not an inert consumer of capacity in this pipeline -- it is a **memorisation
channel that breaks dev-fold early stopping**, and it takes the arm to near
zero rather than costing it a measurable fraction. Single network, fold 0, seed
0, l2=1e-2 (the shipped arm's selected value):

| arm on the baseline | width | rho | best_epoch |
|---|---:|---:|---:|
| baseline only | 860 | 0.5833 | 27 |
| + real ESM-2 | 1,190 | 0.5679 | 105 |
| + BLOSUM cross (stage 3f) | 1,166 | 0.5668 | 22 |
| + shuffled ESM-2 | 1,190 | 0.0453 | 5 |
| + shuffled, unit-scaled | 1,190 | 0.0752 | 1 |
| + random Gaussian | 1,190 | 0.1281 | 3 |

330 columns that uniquely key the peptide -- and a permutation over 5,633
distinct peptides still *identifies* the peptide, it only destroys the geometry
-- let the network drive training loss down without learning anything that
generalises. Dev loss bottoms at epoch 1-5 instead of 27 and stopping fires
before the baseline signal is learned.

It is not only truncation, which is the part worth recording. Truncating the
**baseline alone** to the same epoch still leaves it far ahead:

| baseline-only, max_epochs | 1 | 2 | 3 | 5 | 10 | 27 |
|---|---:|---:|---:|---:|---:|---:|
| rho | 0.2175 | 0.3567 | 0.4460 | 0.5210 | 0.5578 | 0.5833 |

Baseline alone stopped at epoch 5 reaches 0.5210; the shuffled arm stopped at
epoch 5 reaches 0.0453. So the uninformative block both terminates stopping
early *and* dominates the fitted function out of sample. Unit-scaling does not
rescue it, so scale is a contributor rather than the cause -- though the scales
are worth knowing, since mean per-column standard deviation is 0.084 for the
baseline's sparse one-hot, 0.411 for the BLOSUM cross block, 1.000 for random,
and **4.788 (max 54.5)** for the real and shuffled ESM blocks.

**What this leaves standing.** The ``shuffled`` arm is still a valid *sign*
test, and a strong one: real ESM-2 at 0.5679 against its own permutation at
0.0453, with identical width, scale, marginals, covariance and rank, and with
the only difference being which embedding belongs to which sequence. The ESM
block is not inert padding. But because the permuted arm collapses rather than
degrades, the size of that gap is not an effect size and must not be quoted as
one. And no healthy uninformative arm exists here, so **what pure width costs
remains unmeasured** -- the candidate mechanism ``reports/stage3_esm.md`` names
stays a candidate.

Separating capacity from stopping would need a protocol in which no arm's model
is dev-selected -- a fixed epoch count for every arm. That is not run here.

The original ensemble driver is kept as ``run`` and still works, but its output
is not interpretable as a displacement measurement; it prints that caveat.

``reports/stage3_esm.md`` reports the additive arm at −0.0170 [−0.0468, +0.0320]
and names capacity displacement as a candidate mechanism without settling it.
Stage 3f narrowed it: a width-matched block of BLOSUM cross-features costs
−0.0832, conclusively. But that block is **redundant and structured** -- a
deterministic bilinear re-encoding of residues the baseline already has -- so it
could be actively misleading rather than merely wide, and those two readings
support different conclusions. This file supplies the two controls that separate
them, at **exactly** the additive arm's width.

Every arm is 860 baseline one-hot columns plus a 330-column block, 1,190
features, built through the identical standardise-then-PCA path as the real ESM
block (``scripts.esm_arm._fit_block``, imported rather than reimplemented --
if the control's path differed from the arm's, the comparison would measure the
path):

* ``+esm`` -- the real thing. Stage 3's shipped additive arm, quoted from
  ``reports/stage3_comparisons.csv``, not refitted here.
* ``+random`` -- Gaussian columns at matched width. No information, no
  identifiability, no geometry. This is the **capacity control proper**: what
  330 dense columns cost the baseline purely by being there.
* ``+shuffled`` -- the real ESM block with the sequence-to-embedding
  correspondence permuted over *unique* peptides and *unique* alleles, so a
  peptide keeps one wrong embedding everywhere it appears. Same marginals, same
  covariance, same rank, same width.

**What ``shuffled`` does and does not destroy**, because this is easy to get
wrong. 5,633 distinct peptides permuted among themselves still *identify* the
peptide uniquely -- the block remains a faithful, if scrambled, categorical
encoding, so a model can still learn a per-peptide offset from it. What the
permutation destroys is the **similarity geometry**: which peptides the
pretrained model placed near which. So ``real − shuffled`` is not "the
information in ESM-2"; it is specifically the value of ESM-2's *geometry*, which
is the thing a pretrained representation is supposed to contribute and the thing
stage 3 §4 already showed matters more than which model produced it. The same
argument applies with more force on the HLA side, where there are only 75
distinct domains.

Reading the three together:

* random ≈ blosum_cross ≈ −0.08  ->  displacement is real and content-blind;
  the ESM block recovers most of a cost any block of that width would pay.
* random materially smaller than −0.08  ->  the BLOSUM block was actively
  harmful, stage 3f's control was unusually bad, and the +0.066 claim weakens.
* shuffled ≈ real  ->  ESM-2's geometry contributes nothing here, and whatever
  the block does is available from any scrambled block of the same shape.
* shuffled materially below real  ->  the geometry does contribute, and the
  additive null is a statement about sufficiency rather than about content.

Each arm gets its own boundary-checked L2 ladder, for the reason stage 3 §2
gives. Ensembling matches: 5 CV folds x the {onehot, blosum} encoding axis x 3
seeds = 30 networks, which is the axis the shipped additive arm uses.
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

from pepstab import esm as pesm  # noqa: E402
from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import (  # noqa: E402
    describe_delta,
    eligible_alleles,
    paired_cluster_bootstrap,
    score,
)
from pepstab.features import ENCODINGS, build_features  # noqa: E402
from pepstab.mlp import MLPConfig, MLPRegressor  # noqa: E402
from scripts.baseline_ensemble import N_FOLDS, cv_folds  # noqa: E402
from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE, SEEDS  # noqa: E402
# Private, and imported deliberately: the control blocks have to travel the
# *same* standardisation and PCA path as the real ESM block, and a second
# implementation of that path would be a second thing that could differ.
from scripts.esm_arm import HLA_PCA, PEP_PCA, _fit_block  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = REPO_ROOT / "preds"
REPORT_DIR = REPO_ROOT / "reports"

HIDDEN = (256, 64)
#: Five points spanning stage 2's range and stage 3's selection, same ladder for
#: every arm. Equal budget; the selected values are free to differ.
L2_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
CONTROL_SEED = 20261003
ARMS = ("random", "shuffled")
#: The shipped additive arm, for reference. Quoted, not refitted.
REAL_ARM_PRED = "esm_plus_seq_ensemble"
BASELINE_PRED = "seq_ensemble_pep_pseudo"


def esm_unique_blocks(train: pd.DataFrame, checkpoint: str, layer: str):
    """The per-unique-sequence ESM blocks, exactly as the shipped arm builds them."""
    pep_M, pep_lookup, pep_var = _fit_block("peptide", "pos", checkpoint, layer,
                                            train.peptide.to_numpy(), PEP_PCA)
    hla_M, hla_lookup, hla_var = _fit_block("hla", "contact", checkpoint, layer,
                                            train.hla_seq.to_numpy(), HLA_PCA)
    return (pep_M, pep_lookup, pep_var), (hla_M, hla_lookup, hla_var)


def gather(M: np.ndarray, lookup: dict, values, perm: np.ndarray | None = None
           ) -> np.ndarray:
    """Fan a per-unique-sequence block out to rows, optionally through ``perm``.

    ``perm`` maps a unique-sequence row to a *different* unique-sequence row. It
    is applied to the index, not to the rows of the output, so one peptide keeps
    one wrong embedding everywhere it appears -- permuting output rows instead
    would also destroy the repeated-peptide structure and confound two effects.
    """
    idx = np.fromiter((lookup[str(v)] for v in values), dtype=np.int64,
                      count=len(values))
    if perm is not None:
        idx = perm[idx]
    return M[idx]


def build_arms(train: pd.DataFrame, val: pd.DataFrame, checkpoint: str,
               layer: str) -> tuple[dict, dict]:
    """``{(arm, encoding): (X_train, X_val)}`` plus an info dict."""
    (pep_M, pep_lk, pep_var), (hla_M, hla_lk, hla_var) = esm_unique_blocks(
        train, checkpoint, layer)
    width = pep_M.shape[1] + hla_M.shape[1]
    rng = np.random.default_rng(CONTROL_SEED)

    real_t = np.hstack([gather(pep_M, pep_lk, train.peptide),
                        gather(hla_M, hla_lk, train.hla_seq)])
    real_v = np.hstack([gather(pep_M, pep_lk, val.peptide),
                        gather(hla_M, hla_lk, val.hla_seq)])

    # Shuffled: permute both unique-sequence axes, consistently across train and
    # val, so the correspondence is wrong in the same way everywhere.
    pep_perm = rng.permutation(len(pep_M))
    hla_perm = rng.permutation(len(hla_M))
    shuf_t = np.hstack([gather(pep_M, pep_lk, train.peptide, pep_perm),
                        gather(hla_M, hla_lk, train.hla_seq, hla_perm)])
    shuf_v = np.hstack([gather(pep_M, pep_lk, val.peptide, pep_perm),
                        gather(hla_M, hla_lk, val.hla_seq, hla_perm)])

    # Random: standardised on the training rows, the same treatment the real
    # block's components get, so the two differ in content and not in scale.
    raw_t = rng.normal(size=(len(train), width)).astype(np.float32)
    raw_v = rng.normal(size=(len(val), width)).astype(np.float32)
    mu, sd = raw_t.mean(axis=0, keepdims=True), raw_t.std(axis=0, keepdims=True) + 1e-6
    rand_t, rand_v = (raw_t - mu) / sd, (raw_v - mu) / sd

    blocks = {"random": (rand_t, rand_v), "shuffled": (shuf_t, shuf_v)}
    out = {}
    for enc in ENCODINGS:
        base_t = build_features(train, "pep_pseudo", enc)
        base_v = build_features(val, "pep_pseudo", enc)
        for arm, (bt, bv) in blocks.items():
            out[(arm, enc)] = (np.hstack([base_t, bt]).astype(np.float32),
                               np.hstack([base_v, bv]).astype(np.float32))

    # How much of the real block does the permutation leave intact? Marginals and
    # covariance are preserved exactly by construction; this records that the
    # row-level correspondence really is gone.
    corr = float(np.mean([np.corrcoef(real_t[:, j], shuf_t[:, j])[0, 1]
                          for j in range(0, width, max(1, width // 32))]))
    info = {
        "layer": layer, "block_width": int(width),
        "stacked_width": int(out[("random", ENCODINGS[0])][0].shape[1]),
        "peptide_components": int(pep_M.shape[1]),
        "hla_components": int(hla_M.shape[1]),
        "peptide_pca_explained": round(float(pep_var), 6),
        "hla_pca_explained": round(float(hla_var), 6),
        "n_unique_peptides_permuted": int(len(pep_M)),
        "n_unique_hla_permuted": int(len(hla_M)),
        "mean_column_corr_real_vs_shuffled": round(corr, 4),
        "real_vs_shuffled_marginals_identical": bool(
            np.allclose(np.sort(real_t[:, 0]), np.sort(shuf_t[:, 0]))
            if len(real_t) == len(shuf_t) else False),
    }
    return out, info


def _fit(Xt, y_train, Xv, is_fit, l2, seed):
    model = MLPRegressor(MLPConfig(hidden=HIDDEN, l2=l2, seed=seed,
                                   max_epochs=MAX_EPOCHS, patience=PATIENCE))
    model.fit(Xt[is_fit], y_train[is_fit], Xt[~is_fit], y_train[~is_fit])
    return model.predict(Xv), model


def mode_run(args) -> int:
    df = load_with_splits()
    train = df[df.split == "train"]
    val = df[df.split == "val"].sort_values("pair_id").reset_index(drop=True)
    fold_id = cv_folds(train, N_FOLDS)
    alleles = eligible_alleles(val.allele, val.y_log1p, split="val")
    y_train, y_val = train.y_log1p.to_numpy(), val.y_log1p.to_numpy()

    t0 = time.perf_counter()
    blocks, info = build_arms(train, val, args.checkpoint, args.layer)
    info["feature_seconds"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(info, indent=1))
    if info["stacked_width"] != 1190:
        print(f"NOTE: stacked width is {info['stacked_width']}, not the shipped "
              f"additive arm's 1,190 -- the width match is the point of this "
              f"file, so check PEP_PCA/HLA_PCA before reading the deltas.")

    grid = tuple(args.l2_grid) if args.l2_grid else L2_GRID
    print(f"\nladder {[f'{g:g}' for g in grid]}, fold {args.fold}, "
          f"{len(SEEDS)} seeds, same budget for every arm")
    records, selected, sens = [], {}, []
    for arm in ARMS:
        Xt, Xv = blocks[(arm, ENCODINGS[0])]
        for l2 in grid:
            for seed in SEEDS:
                pred, model = _fit(Xt, y_train, Xv, fold_id != args.fold, l2, seed)
                records.append({
                    "stage": "ladder", "arm": arm, "encoding": ENCODINGS[0],
                    "l2": l2, "seed": seed, "fold": args.fold,
                    "n_features": int(Xt.shape[1]),
                    "best_epoch": model.best_epoch_,
                    "fit_seconds": round(model.fit_seconds_, 1),
                    "member_median_rho": round(score("m", val.allele, y_val, pred,
                                                     alleles).median_spearman, 6)})
            got = [r for r in records if r["arm"] == arm and r["l2"] == l2]
            print(f"  {arm:9s} l2={l2:<7g} rho "
                  f"{np.mean([r['member_median_rho'] for r in got]):.4f} "
                  f"(spread {np.ptp([r['member_median_rho'] for r in got]):.4f})")
        sub = pd.DataFrame([r for r in records if r["arm"] == arm])
        means = sub.groupby("l2").member_median_rho.mean()
        tried = sorted(means.index)
        best = float(means.idxmax())
        interior = len(tried) >= 3 and min(tried) < best < max(tried)
        selected[arm] = best
        sens.append({"arm": arm, "ladder": " ".join(f"{t:g}" for t in tried),
                     "n_points": len(tried), "selected": best,
                     "interior": bool(interior),
                     "rho_at_selected": round(float(means[best]), 4),
                     "rho_at_min": round(float(means[min(tried)]), 4),
                     "rho_at_max": round(float(means[max(tried)]), 4)})
        print(f"\n{arm}: selected l2={best:g} "
              f"({'interior' if interior else 'AT BOUNDARY'})")

    REPORT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(sens).to_csv(REPORT_DIR / "stage3g_tuning.csv", index=False)
    if not all(s["interior"] for s in sens):
        raise SystemExit("\nA selection is on a ladder edge; extend with "
                         "--l2-grid and re-run before the ensemble.")

    n_members = N_FOLDS * len(ENCODINGS) * len(SEEDS)
    print(f"\nensemble: {N_FOLDS} folds x {len(ENCODINGS)} encodings x "
          f"{len(SEEDS)} seeds = {n_members} networks per arm\n")
    members: dict[str, list[np.ndarray]] = {}
    started = time.perf_counter()
    for arm in ARMS:
        for enc in ENCODINGS:
            Xt, Xv = blocks[(arm, enc)]
            for k in range(N_FOLDS):
                for seed in SEEDS:
                    pred, model = _fit(Xt, y_train, Xv, fold_id != k,
                                       selected[arm], seed)
                    members.setdefault(arm, []).append(pred)
                    records.append({
                        "stage": "ensemble_member", "arm": arm, "encoding": enc,
                        "l2": selected[arm], "seed": seed, "fold": k,
                        "n_features": int(Xt.shape[1]),
                        "best_epoch": model.best_epoch_,
                        "fit_seconds": round(model.fit_seconds_, 1),
                        "member_median_rho": round(score("m", val.allele, y_val,
                                                         pred, alleles
                                                         ).median_spearman, 6)})
            got = [r for r in records if r["arm"] == arm and r["encoding"] == enc
                   and r["stage"] == "ensemble_member"]
            print(f"  {arm:9s} {enc:7s}: {len(got)} networks, member rho "
                  f"{np.mean([r['member_median_rho'] for r in got]):.4f}, "
                  f"{np.mean([r['fit_seconds'] for r in got]):.0f}s/net")
    wall = time.perf_counter() - started

    runs = pd.DataFrame(records)
    runs.to_csv(REPORT_DIR / "stage3g_runs.csv", index=False)
    ens = {a: np.mean(members[a], axis=0) for a in ARMS}
    for arm, pred in ens.items():
        pd.DataFrame({"pair_id": val.pair_id.to_numpy(), "y_pred": pred}).to_csv(
            PRED_DIR / f"stage3g_{arm}_stacked.csv", index=False)

    def load(name):
        path = PRED_DIR / f"{name}.csv"
        if not path.exists():
            return None
        merged = val[["pair_id"]].merge(pd.read_csv(path), on="pair_id",
                                        how="left", validate="one_to_one")
        if merged.y_pred.isna().any():
            raise ValueError(f"{name} does not cover every validation row")
        return merged.y_pred.to_numpy()

    base = load(BASELINE_PRED)
    real = load(REAL_ARM_PRED)
    if base is None:
        raise SystemExit(f"missing preds/{BASELINE_PRED}.csv, the reference every "
                         f"delta is measured against")

    # Every arm that exists, including the real one and stage 3f's, so the
    # displacement table can be read in one place.
    panel = {"+esm (shipped additive)": real,
             "+shuffled": ens["shuffled"], "+random": ens["random"],
             "+blosum_cross (stage 3f)": load("stage3f_cross_stacked")}
    rows, comparisons = [], []
    for name, pred in panel.items():
        if pred is None:
            print(f"  (missing {name}; omitted)")
            continue
        s = score(name, val.allele, y_val, pred, alleles)
        res = paired_cluster_bootstrap(val.cluster_id, val.allele, y_val, base,
                                       pred, alleles=alleles)
        lo, hi = res["ci95"]
        rows.append({"arm": name, "median_rho": round(s.median_spearman, 4),
                     "delta_vs_baseline": round(res["delta_median_spearman"], 4),
                     "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                     "verdict": describe_delta(res)})
        print(f"\n{name} vs baseline: {s.median_spearman:.4f}  "
              f"delta {res['delta_median_spearman']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        print(f"  {describe_delta(res)}")

    # The head-to-heads that decide the two readings.
    if real is not None:
        for other in ("+shuffled", "+random", "+blosum_cross (stage 3f)"):
            if panel.get(other) is None:
                continue
            res = paired_cluster_bootstrap(val.cluster_id, val.allele, y_val,
                                           panel[other], real, alleles=alleles)
            lo, hi = res["ci95"]
            print(f"\n+esm vs {other}: delta "
                  f"{res['delta_median_spearman']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
            print(f"  {describe_delta(res)}")
            comparisons.append({
                "arm": "+esm (shipped additive)", "reference": other,
                "delta": round(res["delta_median_spearman"], 4),
                "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                "verdict": describe_delta(res),
                "n_rows": len(val), "n_alleles": len(alleles)})

    pd.DataFrame(rows).to_csv(REPORT_DIR / "stage3g_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(REPORT_DIR / "stage3g_comparisons.csv",
                                     index=False)
    (REPORT_DIR / "stage3g_cost.json").write_text(json.dumps({
        **info, "selected_l2": selected, "n_networks": int(len(runs)),
        "ensemble_wall_minutes": round(wall / 60, 1),
        "total_fit_seconds": float(runs.fit_seconds.sum()),
        "median_best_epoch": {a: float(g.best_epoch.median())
                              for a, g in runs.groupby("arm")},
    }, indent=2))
    print(f"\n{len(runs)} networks; ensemble {wall / 60:.1f} min")
    print("wrote reports/stage3g_{tuning,runs,summary,comparisons}.csv, "
          "stage3g_cost.json, preds/stage3g_*.csv")
    return 0


def mode_diagnose(args) -> int:
    """Reproduce the two tables in the module docstring.

    Cheap -- twelve single networks -- and it is the whole evidential basis for
    not reporting a displacement number, so it should be rerunnable.
    """
    df = load_with_splits()
    train = df[df.split == "train"]
    val = df[df.split == "val"].sort_values("pair_id").reset_index(drop=True)
    fold_id = cv_folds(train, N_FOLDS)
    alleles = eligible_alleles(val.allele, val.y_log1p, split="val")
    yt, yv = train.y_log1p.to_numpy(), val.y_log1p.to_numpy()
    is_fit = fold_id != args.fold

    base_t = build_features(train, "pep_pseudo", "onehot")
    base_v = build_features(val, "pep_pseudo", "onehot")
    blocks, info = build_arms(train, val, args.checkpoint, args.layer)
    from scripts.esm_arm import assemble
    from scripts.stage3f_blosum_cross import cross_features

    real_t, (real_v,), _ = assemble(train, [val], args.checkpoint, args.layer,
                                    "pos", "contact", None, pep_pca=PEP_PCA)
    rnd = blocks[("random", "onehot")]
    shf = blocks[("shuffled", "onehot")]
    rnd_t, rnd_v = rnd[0][:, base_t.shape[1]:], rnd[1][:, base_t.shape[1]:]
    shf_t, shf_v = shf[0][:, base_t.shape[1]:], shf[1][:, base_t.shape[1]:]
    cross_t, cross_v = cross_features(train), cross_features(val)

    sd = shf_t.std(axis=0, keepdims=True) + 1e-6
    arms = [
        ("baseline only", base_t, base_v),
        ("+ real ESM-2", np.hstack([base_t, real_t]), np.hstack([base_v, real_v])),
        ("+ BLOSUM cross", np.hstack([base_t, cross_t]), np.hstack([base_v, cross_v])),
        ("+ shuffled ESM-2", shf[0], shf[1]),
        ("+ shuffled, unit-scaled", np.hstack([base_t, shf_t / sd]),
         np.hstack([base_v, shf_v / sd])),
        ("+ random Gaussian", rnd[0], rnd[1]),
    ]
    scales = {"baseline onehot": base_t, "real ESM-2": real_t,
              "shuffled ESM-2": shf_t, "random": rnd_t, "BLOSUM cross": cross_t}
    print("mean per-column standard deviation of each block (training rows):")
    srows = []
    for name, blk in scales.items():
        s = blk.std(axis=0)
        print(f"  {name:18s} width {blk.shape[1]:5d}  mean {s.mean():7.3f}  "
              f"max {s.max():7.3f}")
        srows.append({"block": name, "width": int(blk.shape[1]),
                      "mean_col_std": round(float(s.mean()), 4),
                      "max_col_std": round(float(s.max()), 4)})

    print(f"\nsingle network, fold {args.fold}, seed 0, l2={args.l2:g}:")
    rows = []
    for name, Xt, Xv in arms:
        pred, model = _fit(Xt, yt, Xv, is_fit, args.l2, 0)
        rho = score("m", val.allele, yv, pred, alleles).median_spearman
        print(f"  {name:24s} {Xt.shape[1]:5d} feat  rho {rho:.4f}  "
              f"best_epoch {model.best_epoch_:3d}")
        rows.append({"arm": name, "n_features": int(Xt.shape[1]),
                     "median_rho": round(float(rho), 4),
                     "best_epoch": int(model.best_epoch_),
                     "l2": args.l2, "fold": args.fold, "seed": 0})

    print("\nbaseline alone, truncated -- is the collapse merely 'stopped early'?")
    trunc = []
    for n in (1, 2, 3, 5, 10, 27, MAX_EPOCHS):
        m = MLPRegressor(MLPConfig(hidden=HIDDEN, l2=args.l2, seed=0,
                                   max_epochs=n, patience=n))
        m.fit(base_t[is_fit], yt[is_fit], base_t[~is_fit], yt[~is_fit])
        rho = score("m", val.allele, yv, m.predict(base_v), alleles).median_spearman
        print(f"  max_epochs={n:4d}  best_epoch {m.best_epoch_:3d}  rho {rho:.4f}")
        trunc.append({"max_epochs": n, "best_epoch": int(m.best_epoch_),
                      "median_rho": round(float(rho), 4)})

    REPORT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(srows).to_csv(REPORT_DIR / "stage3g_block_scales.csv", index=False)
    pd.DataFrame(rows).to_csv(REPORT_DIR / "stage3g_diagnostic.csv", index=False)
    pd.DataFrame(trunc).to_csv(REPORT_DIR / "stage3g_truncation.csv", index=False)
    (REPORT_DIR / "stage3g_cost.json").write_text(json.dumps({
        **info, "verdict": "uninformative width-matched blocks are a "
                           "memorisation channel, not a capacity control; no "
                           "displacement number is reportable from them",
    }, indent=2))
    print("\nwrote reports/stage3g_{block_scales,diagnostic,truncation}.csv "
          "and stage3g_cost.json")
    return 0


def mode_scaleprobe(args) -> int:
    """Does per-component rescaling after PCA move the additive arm?

    The shipped additive arm appends PCA components straight to the baseline's
    sparse one-hot columns, and PCA components carry decreasing variance, so the
    mean per-column standard deviation is 0.084 on the baseline side against
    4.788 on the ESM side -- a ~57x disparity inside the arm being shipped. The
    diagnostic above shows this pipeline is fragile to a dominant appended
    block, so the question is whether the arm's small negative delta is partly
    that rather than an information result.

    This is a **probe for next steps, not a re-run of the shipped arm.** The
    shipped number stays as published; re-selecting it after seeing this would
    be the post-hoc selection the frozen contract exists to prevent. Both
    variants get the same small ladder, because rescaling changes the geometry
    and holding L2 fixed across a scale change is precisely the handicap
    ``reports/stage3_esm.md`` §2 warns about.
    """
    df = load_with_splits()
    train = df[df.split == "train"]
    val = df[df.split == "val"].sort_values("pair_id").reset_index(drop=True)
    fold_id = cv_folds(train, N_FOLDS)
    alleles = eligible_alleles(val.allele, val.y_log1p, split="val")
    yt, yv = train.y_log1p.to_numpy(), val.y_log1p.to_numpy()

    base_t = build_features(train, "pep_pseudo", "onehot")
    base_v = build_features(val, "pep_pseudo", "onehot")
    from scripts.esm_arm import assemble
    esm_t, (esm_v,), _ = assemble(train, [val], args.checkpoint, args.layer,
                                  "pos", "contact", None, pep_pca=PEP_PCA)
    sd = esm_t.std(axis=0, keepdims=True) + 1e-6
    variants = {
        "as_shipped": (np.hstack([base_t, esm_t]), np.hstack([base_v, esm_v])),
        "unit_rescaled": (np.hstack([base_t, esm_t / sd]),
                          np.hstack([base_v, esm_v / sd])),
        "baseline_only": (base_t, base_v),
    }
    grid = tuple(args.l2_grid) if args.l2_grid else (1e-3, 1e-2, 1e-1)
    folds = tuple(range(args.probe_folds))
    seeds = SEEDS[:args.probe_seeds]
    print(f"ladder {[f'{g:g}' for g in grid]}, folds {folds}, seeds {seeds}: "
          f"{len(variants) * len(grid) * len(folds) * len(seeds)} networks\n")

    rows = []
    for name, (Xt, Xv) in variants.items():
        for l2 in grid:
            for k in folds:
                for seed in seeds:
                    pred, model = _fit(Xt, yt, Xv, fold_id != k, l2, seed)
                    rows.append({
                        "variant": name, "l2": l2, "fold": k, "seed": seed,
                        "n_features": int(Xt.shape[1]),
                        "best_epoch": int(model.best_epoch_),
                        "median_rho": round(float(score("m", val.allele, yv, pred,
                                                        alleles).median_spearman), 6)})
            got = [r for r in rows if r["variant"] == name and r["l2"] == l2]
            print(f"  {name:14s} l2={l2:<7g} rho "
                  f"{np.mean([r['median_rho'] for r in got]):.4f} "
                  f"(spread {np.ptp([r['median_rho'] for r in got]):.4f}, "
                  f"best_epoch {np.mean([r['best_epoch'] for r in got]):.0f})")

    runs = pd.DataFrame(rows)
    REPORT_DIR.mkdir(exist_ok=True)
    runs.to_csv(REPORT_DIR / "stage3g_scale_probe.csv", index=False)
    best = runs.groupby(["variant", "l2"]).median_rho.mean().reset_index()
    picked = best.loc[best.groupby("variant").median_rho.idxmax()]
    print("\nbest point per variant (single-network protocol, not the ensemble):")
    print(picked.to_string(index=False))
    b = float(picked[picked.variant == "baseline_only"].median_rho.iloc[0])
    for v in ("as_shipped", "unit_rescaled"):
        r = float(picked[picked.variant == v].median_rho.iloc[0])
        print(f"  {v:14s} vs baseline_only: {r - b:+.4f}")
    print("\nSingle-network deltas, no bootstrap: read the sign and whether the "
          "two variants differ by more than the seed spread, nothing finer.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("diagnose", "scaleprobe", "run"))
    ap.add_argument("--probe-folds", type=int, default=2)
    ap.add_argument("--probe-seeds", type=int, default=2)
    ap.add_argument("--checkpoint", default=pesm.DEFAULT_CHECKPOINT)
    ap.add_argument("--layer", default="mid")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--l2-grid", type=float, nargs="*", default=None)
    args = ap.parse_args()
    if args.mode == "run":
        print("NOTE: `run` fits the full ensembles, but its deltas are NOT "
              "interpretable as displacement costs -- see the module docstring. "
              "`diagnose` is the mode that produced the reported evidence.\n")
        return mode_run(args)
    if args.mode == "scaleprobe":
        return mode_scaleprobe(args)
    return mode_diagnose(args)


if __name__ == "__main__":
    raise SystemExit(main())
