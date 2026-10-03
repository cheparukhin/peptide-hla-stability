"""Stage 2b: does weak-binder augmentation help, and does the source matter?

    .venv/bin/python scripts/augment_affinity.py     # build the manifests first
    .venv/bin/python scripts/baseline_augmented.py

Four arms against the stage 2 measured-only baseline, each at assumed-label
weights 0.1 and 0.25, three seeds apiece:

- **measured_only** -- the stage 2 selected peptide+pseudosequence MLP,
  unchanged. Because the weighted loss reduces exactly to the unweighted one at
  uniform weight 1, this arm is bit-identical to the stage 2 run, which is
  asserted rather than assumed.
- **measured_affinity** -- plus pairs whose measured affinity is at or above
  20,000 nM, labelled assumed-zero.
- **predicted_affinity** -- plus random natural 9-mers whose *predicted*
  affinity is weaker than 20,000 nM, at **identical per-allele counts** on the
  **same 51 alleles**, so the only difference from the measured arm is where the
  negatives came from.
- **predicted_affinity_full** -- secondary. The same predicted source with the
  measured-availability cap lifted, so it reaches all 75 alleles. Its manifest
  is a superset of the matched one, which makes it a clean read on added allele
  coverage rather than a separate draw.

**What is held fixed.** Every arm uses the same config
(``hidden=(256, 64)``, ``l2=1e-5``, one-hot peptide+pseudosequence), the same
17,744 measured fit rows, the same 1,972-row measured stopping fold, the same
three seeds, and the same validation panel. Augmented rows enter the *fit* set
only. The stopping fold stays measured-only and unweighted on purpose: it
decides the epoch, and an epoch chosen against a different objective in each arm
would make the arms incomparable for a reason that has nothing to do with
augmentation.

**Assumed zeros never enter validation or test**, so every number here is scored
on measured labels alone. Selection is on validation; stage 6 scores the test
split once.
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

from pepstab.augment import (  # noqa: E402
    ASSUMED_WEIGHTS,
    attach_hla,
    verify_manifest,
)
from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import (  # noqa: E402
    describe_delta,
    eligible_alleles,
    paired_cluster_bootstrap,
    per_allele_spearman,
    score,
)
from pepstab.features import build_features  # noqa: E402
from pepstab.mlp import MLPConfig, MLPRegressor  # noqa: E402
from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE, SEEDS, inner_folds  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
AUG_DIR = REPO_ROOT / "data" / "augmentation"
PRED_DIR = REPO_ROOT / "preds"
REPORT_DIR = REPO_ROOT / "reports"

#: The stage 2 selected peptide+pseudosequence arm, taken from
#: ``reports/stage2_summary.csv`` and not re-tuned. Augmentation is the only
#: thing that changes between arms.
INPUT_SET = "pep_pseudo"
ENCODING = "onehot"
HIDDEN = (256, 64)
L2 = 1e-5

#: Manifest stem -> whether it is part of the matched primary comparison.
ARMS = {"measured_affinity": True,
        "predicted_affinity": True,
        "predicted_affinity_full": False}


def load_manifest(stem: str, df: pd.DataFrame, fold: pd.Series) -> pd.DataFrame:
    """Read one manifest, re-verify it against the splits, attach HLA sequences.

    :func:`pepstab.augment.verify_manifest` raises on a distance violation, a
    non-zero assumed label, or an augmented pair that already carries a measured
    half-life. It runs here, on every arm, every run -- a manifest is a committed
    CSV and the cheapest way for a leak to reach a model is for someone to edit
    one.
    """
    path = AUG_DIR / f"{stem}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Build the manifests with "
            "`.venv/bin/python scripts/augment_affinity.py` first.")
    manifest = pd.read_csv(path)
    checks = verify_manifest(manifest, df, fold, protocol="inner_folds")
    print(f"  {stem:26s} {checks['n_rows']:>6,} rows  "
          f"{checks['n_alleles']:>3} alleles  "
          f"{checks['n_peptides']:>6,} peptides  "
          f"min Hamming to held-out {checks['min_dist_holdout']}")
    return attach_hla(manifest, df)


def fit_arm(X_fit: np.ndarray, y_fit: np.ndarray, weight: np.ndarray | None,
            X_dev: np.ndarray, y_dev: np.ndarray, X_val: np.ndarray,
            seeds: tuple[int, ...]) -> tuple[list[np.ndarray], list[dict]]:
    """Fit one arm at every seed. Returns per-seed validation predictions."""
    preds, records = [], []
    for seed in seeds:
        model = MLPRegressor(MLPConfig(hidden=HIDDEN, l2=L2, seed=seed,
                                       max_epochs=MAX_EPOCHS, patience=PATIENCE))
        model.fit(X_fit, y_fit, X_dev, y_dev, sample_weight=weight)
        preds.append(model.predict(X_val))
        records.append({"seed": seed, "best_epoch": model.best_epoch_,
                        "epochs": model.n_epochs_,
                        "dev_mse": round(model.dev_mse_, 5),
                        "fit_seconds": round(model.fit_seconds_, 2)})
    return preds, records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=float, nargs="*", default=list(ASSUMED_WEIGHTS),
                    help=f"assumed-label weights to try (default {list(ASSUMED_WEIGHTS)})")
    ap.add_argument("--quick", action="store_true",
                    help="one seed, matched arms only, no bootstrap -- a "
                         "pipeline check. Each paired interval is 2,000 "
                         "resamples and costs about a minute.")
    args = ap.parse_args()

    seeds = (SEEDS[0],) if args.quick else SEEDS
    arms = {k: v for k, v in ARMS.items() if v} if args.quick else ARMS

    df = load_with_splits()
    train = df[df.split == "train"]
    fold = inner_folds(train)
    is_fit = (fold == "fit").to_numpy()
    fit, dev = train[is_fit], train[~is_fit]
    val = df[df.split == "val"].sort_values("pair_id").reset_index(drop=True)
    alleles = eligible_alleles(val.allele, val.y_log1p, split="val")

    print(f"arm config: {ENCODING}_{INPUT_SET} hidden={HIDDEN} l2={L2:g} "
          f"seeds={list(seeds)}")
    print(f"measured fit={len(fit):,}  stopping fold={len(dev):,}  "
          f"val={len(val):,} rows, {len(alleles)} eligible alleles\n")

    print("manifests (re-verified against the frozen splits):")
    manifests = {stem: load_manifest(stem, df, fold) for stem in arms}

    X_fit_measured = build_features(fit, INPUT_SET, ENCODING)
    X_dev = build_features(dev, INPUT_SET, ENCODING)
    X_val = build_features(val, INPUT_SET, ENCODING)
    y_fit_measured = fit.y_log1p.to_numpy(dtype=np.float32)
    y_dev = dev.y_log1p.to_numpy(dtype=np.float32)
    y_val = val.y_log1p.to_numpy()
    val_alleles = val.allele.to_numpy()

    # Which alleles actually received augmentation, from the matched manifest.
    # The split below is a diagnostic, not the primary metric: it asks whether
    # the added rows moved the alleles they were added to.
    augmented_alleles = set(manifests["measured_affinity"]["allele"].unique()) \
        if "measured_affinity" in manifests else set()
    on_panel = [a for a in alleles if a in augmented_alleles]
    off_panel = [a for a in alleles if a not in augmented_alleles]
    print(f"\nsub-panels for the diagnostic columns: {len(on_panel)} eligible "
          f"alleles receive matched augmentation, {len(off_panel)} do not")

    jobs: list[tuple[str, pd.DataFrame | None, float | None]] = [
        ("measured_only", None, None)]
    for stem in arms:
        for w in args.weights:
            jobs.append((f"{stem}_w{w:g}", manifests[stem], float(w)))

    started = time.perf_counter()
    rows, seed_rows, preds = [], [], {}
    for name, manifest, weight in jobs:
        if manifest is None:
            X_arm, y_arm, w_arm, n_added = X_fit_measured, y_fit_measured, None, 0
        else:
            X_add = build_features(manifest, INPUT_SET, ENCODING)
            X_arm = np.concatenate([X_fit_measured, X_add])
            y_arm = np.concatenate([
                y_fit_measured,
                manifest.y_log1p.to_numpy(dtype=np.float32)])
            w_arm = np.concatenate([
                np.ones(len(X_fit_measured), dtype=np.float32),
                np.full(len(X_add), weight, dtype=np.float32)])
            n_added = len(X_add)

        members, fits = fit_arm(X_arm, y_arm, w_arm, X_dev, y_dev, X_val, seeds)
        preds[name] = np.mean(members, axis=0)

        per_seed = [score(name, val.allele, y_val, p, alleles) for p in members]
        for rec, s in zip(fits, per_seed):
            seed_rows.append({"arm": name, "n_added": n_added,
                              "assumed_weight": weight, **rec,
                              **{k: s.as_row()[k] for k in
                                 ("median_per_allele_spearman", "mae_log1p",
                                  "median_precision_at_10")}})
        rhos = [s.median_spearman for s in per_seed]
        # Per-allele series per seed, so the two sub-panels below can be read
        # off without rescoring. NaN means the model could not rank that allele,
        # which the contract scores as 0 (EVALUATION.md, UNRANKED_CONTRIBUTION).
        per_allele = [per_allele_spearman(val.allele, y_val, p, alleles)
                      for p in members]
        rows.append({
            "arm": name,
            "source": "-" if manifest is None else manifest["source"].iloc[0],
            "matched": "-" if manifest is None else str(ARMS[name.rsplit("_w", 1)[0]]),
            "assumed_weight": weight,
            "n_added_rows": n_added,
            "n_added_alleles": 0 if manifest is None else manifest["allele"].nunique(),
            "n_added_peptides": 0 if manifest is None else manifest["peptide"].nunique(),
            "rho_mean": float(np.mean(rhos)),
            "rho_min": float(np.min(rhos)),
            "rho_max": float(np.max(rhos)),
            "rho_augmented_alleles": float(np.mean(
                [r.reindex(on_panel).fillna(0.0).median() for r in per_allele])),
            "rho_other_alleles": float(np.mean(
                [r.reindex(off_panel).fillna(0.0).median() for r in per_allele])),
            "mae_mean": float(np.mean([s.mae_log1p for s in per_seed])),
            "p10_mean": float(np.mean([s.median_precision_at_k for s in per_seed])),
            "fit_seconds_mean": float(np.mean([r["fit_seconds"] for r in fits])),
        })
        print(f"\n{name:30s} +{n_added:>5,} rows  "
              f"rho={rows[-1]['rho_mean']:+.4f} "
              f"[{rows[-1]['rho_min']:+.4f}, {rows[-1]['rho_max']:+.4f}]  "
              f"mae={rows[-1]['mae_mean']:.4f}  p@10={rows[-1]['p10_mean']:.3f}")
        print(f"{'':30s} augmented alleles rho={rows[-1]['rho_augmented_alleles']:+.4f}  "
              f"other alleles rho={rows[-1]['rho_other_alleles']:+.4f}")
    wall = time.perf_counter() - started

    results = pd.DataFrame(rows)
    seed_frame = pd.DataFrame(seed_rows)

    # --- paired comparisons against measured-only -------------------------
    deltas: list[dict] = []
    source_rows: list[dict] = []
    comparisons = [] if args.quick else [n for n, _, _ in jobs
                                         if n != "measured_only"]
    if args.quick:
        print("\n--quick: skipping the paired cluster bootstraps "
              "(2,000 resamples each, ~1 min per comparison).")
    else:
        print(f"\npaired cluster bootstrap against measured_only "
              f"(seed-mean predictions, {len(seeds)} seeds):")
    for name in comparisons:
        boot = paired_cluster_bootstrap(val.cluster_id, val.allele, y_val,
                                        preds["measured_only"], preds[name],
                                        alleles)
        lo, hi = boot["ci95"]
        verdict = describe_delta(boot)
        deltas.append({"arm": name, "delta_median_rho": boot["delta_median_spearman"],
                       "ci_low": lo, "ci_high": hi, "verdict": verdict})
        print(f"  {name:30s} {boot['delta_median_spearman']:+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]   {verdict}")

    # The source question, asked directly: predicted minus measured at matched
    # counts. This is the comparison stage 2b exists for, and it is paired on
    # the same resamples rather than read off two intervals.
    if not args.quick:
        print("\nlabel source, at matched per-allele counts "
              "(predicted minus measured):")
    for w in ([] if args.quick else args.weights):
        a, b = f"measured_affinity_w{w:g}", f"predicted_affinity_w{w:g}"
        if a not in preds or b not in preds:
            continue
        boot = paired_cluster_bootstrap(val.cluster_id, val.allele, y_val,
                                        preds[a], preds[b], alleles)
        lo, hi = boot["ci95"]
        verdict = describe_delta(boot)
        source_rows.append({"arm": f"predicted - measured (w={w:g})",
                            "delta_median_rho": boot["delta_median_spearman"],
                            "ci_low": lo, "ci_high": hi, "verdict": verdict})
        print(f"  w={w:<5g} {boot['delta_median_spearman']:+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]   {verdict}")
    deltas.extend(source_rows)

    # --- write ------------------------------------------------------------
    suffix = "_quick" if args.quick else ""
    REPORT_DIR.mkdir(exist_ok=True)
    PRED_DIR.mkdir(exist_ok=True)
    results.to_csv(REPORT_DIR / f"stage2b_arms{suffix}.csv", index=False)
    seed_frame.to_csv(REPORT_DIR / f"stage2b_runs{suffix}.csv", index=False)
    pd.DataFrame(deltas).to_csv(REPORT_DIR / f"stage2b_deltas{suffix}.csv", index=False)
    for name, p in preds.items():
        pd.DataFrame({"pair_id": val.pair_id.to_numpy(), "y_pred": p}).to_csv(
            PRED_DIR / f"stage2b_{name}{suffix}.csv", index=False)

    if not args.quick:
        best = results[results.arm != "measured_only"].sort_values(
            "rho_mean", ascending=False).iloc[0]
        (REPORT_DIR / "stage2b_headline.json").write_text(json.dumps({
            "baseline": "measured_only",
            "baseline_rho_mean": float(results.loc[results.arm == "measured_only",
                                                   "rho_mean"].iloc[0]),
            "best_augmented_arm": best.arm,
            "best_augmented_rho_mean": float(best.rho_mean),
            "selected_on": "val median per-allele Spearman, mean over seeds",
            "config": {"input_set": INPUT_SET, "encoding": ENCODING,
                       "hidden": list(HIDDEN), "l2": L2, "seeds": list(seeds)},
            "weights": list(args.weights),
            "note": "assumed zeros never enter validation or test",
        }, indent=2) + "\n")

    print(f"\nwrote reports/stage2b_arms{suffix}.csv, "
          f"reports/stage2b_runs{suffix}.csv, reports/stage2b_deltas{suffix}.csv "
          f"and {len(preds)} prediction files")
    print(f"total fit wall time: {wall / 60:.1f} min "
          f"({wall / max(1, len(jobs) * len(seeds)):.1f} s per network)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
