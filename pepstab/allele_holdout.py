"""Leave-allele-out evaluation, stratified by HLA pseudosequence distance.

Stage 7b. **A second, separate contract.** ``EVALUATION.md`` is frozen and this
module does not touch, modify or reinterpret it; ``data/splits.csv`` is read
only to restrict the cohort and is never recomputed. The two evaluations answer
different questions and their numbers are not comparable:

===============================  ==========================================
frozen split (``EVALUATION.md``)  unseen *peptides*, on alleles we trained on
this module                       unseen *allotype*, on peptides we largely saw
===============================  ==========================================

Why run it at all, when ``HACKATHON_PLAN.md`` gives four reasons it is not the
headline: pan-allele transfer is where pretrained representations have the
strongest prior of winning, so a gain confined to distant alleles would be a
real result even if the pooled comparison came out flat. The four reasons do not
go away by being measured, and ``reports/stage7_allele_holdout.md`` restates all
four beside every number.

**Test rows are never read.** The cohort is ``split in {"train", "val"}``, so
this evaluation cannot consume the single-use test budget. The cost is that each
held-out allele is scored on roughly 80% of its rows.

**On the plan's reason 1.** The plan says holding out an allele also holds out
its peptide panel. :func:`cohort_table` measures that directly and it is not
what the data shows -- for the median eligible allele, *every* one of its rows
carries a peptide that is also assayed on some other allele. The confound that
does survive is panel *composition* (each allele's panel was partly selected by
predicted affinity for that allele), and :func:`cohort_table` reports the
per-allele zero share and median label so that it stays visible.

Arms plug in through :class:`Arm`, which names feature *sources* rather than
carrying matrices, so a worker process can materialise them itself and a later
ESM-2 arm is a one-line change at the call site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .data import PSEUDOSEQ_LENGTH, encode_sequences, load_with_splits
from .evaluation import (
    BOOTSTRAP_SEED,
    N_BOOTSTRAP,
    PRECISION_K,
    STABILITY_THRESHOLD_HOURS,
    precision_at_k,
    validate_finite,
)
from .features import build_features
from .mlp import MLPConfig, MLPRegressor
from .splits import assign_clusters

#: The splits this evaluation may read. The frozen test split is excluded by
#: construction, not by convention, so no code path here can score it.
COHORT_SPLITS = ("train", "val")

#: Rows an allele needs, in scope, to be held out and scored. Not a judgement
#: call: in-scope per-allele counts jump from 30 to 177 with nothing between, so
#: every bar in (30, 177] selects the same 68 alleles.
MIN_ROWS_PER_HELDOUT_ALLELE = 50

#: Pseudosequence-distance strata, fixed from the distance distribution over
#: eligible alleles before any fold was fitted (25 / 23 / 20 alleles -- near
#: tertiles). d >= 5 holds only 9 alleles, so a finer split is not supported.
DISTANCE_STRATA = {
    "near (d<=1)": (0, 1),
    "intermediate (d=2-3)": (2, 3),
    "distant (d>=4)": (4, PSEUDOSEQ_LENGTH),
}

#: Alleles sharing a pseudosequence with another allele (distance 0). A
#: pseudosequence-only model sees its exact input in training when one of these
#: is held out, so the fold is degenerate. Kept in the panel -- dropping them
#: after the fact would be a post-hoc panel change -- but flagged, and the near
#: stratum is reported with and without them.
DEGENERATE_DISTANCE = 0

#: Fraction of the fit rows reserved for early stopping, cut along whole
#: peptide clusters. Matches ``scripts/baseline_sequence.DEV_FRACTION``.
DEV_FRACTION = 0.10

#: Networks averaged per fold. The contract, so that any later arm matches it.
ENSEMBLE_SIZE = 6


# --------------------------------------------------------------------------
# cohort
# --------------------------------------------------------------------------

def load_cohort() -> pd.DataFrame:
    """The in-scope rows: the frozen splits' train and val, test excluded.

    Splits are loaded, never recomputed. The join is on ``(pair_id)`` inside
    :func:`pepstab.data.load_with_splits`, which is itself keyed off the
    read-only CSV, so nothing here can reorder or re-derive a split.
    """
    df = load_with_splits()
    out = df[df["split"].isin(COHORT_SPLITS)].copy()
    return out.sort_values("pair_id").reset_index(drop=True)


def pseudoseq_distances(cohort: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Hamming distance between allele contact pseudosequences.

    Square frame indexed and columned by allele. The diagonal is 0.
    """
    seqs = cohort.groupby("allele")["hla_pseudoseq"].first().sort_index()
    codes = encode_sequences(seqs.tolist(), PSEUDOSEQ_LENGTH)
    dist = (codes[:, None, :] != codes[None, :, :]).sum(axis=2)
    return pd.DataFrame(dist, index=seqs.index, columns=seqs.index)


def eligible_holdout_alleles(cohort: pd.DataFrame,
                             min_rows: int = MIN_ROWS_PER_HELDOUT_ALLELE
                             ) -> list[str]:
    """Alleles with enough in-scope rows and enough label spread to rank.

    Derived from the cohort and the labels alone, never from predictions, so
    every arm is scored on the same 68 folds.
    """
    grouped = cohort.groupby("allele")["y_log1p"]
    keep = (grouped.size() >= min_rows) & (grouped.nunique() >= 2)
    return sorted(keep.index[keep].tolist())


def stratum_of(distance: int) -> str:
    for name, (lo, hi) in DISTANCE_STRATA.items():
        if lo <= distance <= hi:
            return name
    raise ValueError(f"distance {distance} falls in no stratum")


def cohort_table(cohort: pd.DataFrame, alleles: list[str] | None = None
                 ) -> pd.DataFrame:
    """Per-allele design facts, computed before any model is fitted.

    Carries the two things a reader needs in order not to over-read a stratum
    difference: ``peptide_seen_share`` (how much of this allele's panel the fit
    set already contains -- the plan's reason 1, measured) and
    ``zero_share`` / ``median_label`` (how hard this allele's panel is, which is
    correlated with distance and cannot be separated from it here).
    """
    if alleles is None:
        alleles = eligible_holdout_alleles(cohort)
    dist = pseudoseq_distances(cohort)
    rows = []
    for allele in alleles:
        mine = cohort[cohort["allele"] == allele]
        others = cohort[cohort["allele"] != allele]
        others_peptides = set(others["peptide"])
        # Nearest allele still in the fit set -- every allele but this one.
        d = int(dist.loc[allele].drop(index=allele).min())
        rows.append({
            "allele": allele,
            "n_rows": len(mine),
            "n_peptides": int(mine["peptide"].nunique()),
            "dist_to_nearest_fit_allele": d,
            "stratum": stratum_of(d),
            "degenerate_pseudoseq": d == DEGENERATE_DISTANCE,
            "peptide_seen_share": float(mine["peptide"].isin(others_peptides).mean()),
            "zero_share": float((mine["thalf_hours"] == 0).mean()),
            "median_label_log1p": float(mine["y_log1p"].median()),
        })
    return pd.DataFrame(rows).set_index("allele")


# --------------------------------------------------------------------------
# arms
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSource:
    """Where one representation's feature matrix comes from.

    Named rather than materialised so that an :class:`Arm` stays small enough to
    ship to a worker process, and so that a feature table produced by another
    workstream drops in without this module importing it.

    - ``kind="sequence"``: built by :func:`pepstab.features.build_features` from
      ``input_set`` and ``encoding``.
    - ``kind="npy"``: a ``.npy`` matrix plus a ``pair_id`` index, aligned to the
      cohort by ``pair_id`` -- never positionally, which is how a feature table
      silently desynchronises from the labels.
    """

    name: str
    kind: str
    input_set: str | None = None
    encoding: str | None = None
    matrix_path: str | None = None
    pair_id_path: str | None = None

    def build(self, cohort: pd.DataFrame) -> np.ndarray:
        if self.kind == "sequence":
            return build_features(cohort, self.input_set, self.encoding)
        if self.kind == "npy":
            matrix = np.load(self.matrix_path)
            pair_ids = np.load(self.pair_id_path)
            if len(matrix) != len(pair_ids):
                raise ValueError(
                    f"{self.name}: matrix has {len(matrix)} rows but the pair_id "
                    f"index has {len(pair_ids)}")
            lookup = pd.Series(np.arange(len(pair_ids)), index=pair_ids)
            wanted = cohort["pair_id"].to_numpy()
            missing = np.setdiff1d(wanted, pair_ids)
            if len(missing):
                raise ValueError(
                    f"{self.name}: no features for {len(missing)} cohort rows, "
                    f"first pair_id {missing[0]}")
            return np.asarray(matrix[lookup.loc[wanted].to_numpy()], dtype=np.float32)
        raise ValueError(f"unknown feature source kind {self.kind!r}")


@dataclass(frozen=True)
class Arm:
    """One model family to run through the leave-allele-out contract.

    ``sources`` x ``seeds`` is the ensemble, and its size is checked against
    :data:`ENSEMBLE_SIZE` so two arms cannot silently differ in ensemble size --
    stage 3 measured ensembling alone at nearly twice the worthwhile-gain bar,
    so an unmatched ensemble would swamp whatever is being compared.
    """

    name: str
    sources: tuple[FeatureSource, ...]
    configs: dict[str, tuple[tuple[int, ...], float]]
    seeds: tuple[int, ...]
    max_epochs: int = 300
    patience: int = 15

    def n_members(self) -> int:
        return len(self.sources) * len(self.seeds)

    def validate(self, expected: int = ENSEMBLE_SIZE) -> "Arm":
        if self.n_members() != expected:
            raise ValueError(
                f"arm {self.name!r} has {len(self.sources)} representations x "
                f"{len(self.seeds)} seeds = {self.n_members()} networks, but the "
                f"stage 7b contract fixes the ensemble at {expected}. Matching "
                "ensemble size across arms is not optional.")
        missing = [s.name for s in self.sources if s.name not in self.configs]
        if missing:
            raise ValueError(f"arm {self.name!r}: no config for {missing}")
        return self


# --------------------------------------------------------------------------
# one fold
# --------------------------------------------------------------------------

def inner_dev_mask(fit: pd.DataFrame, dev_fraction: float = DEV_FRACTION
                   ) -> np.ndarray:
    """Boolean mask over ``fit`` marking the early-stopping fold.

    Cut along whole Hamming <= 3 peptide clusters with the same water-fill the
    frozen splits use, so no network stops on a peptide within 3 substitutions
    of one it trained on. Deterministic given the fit rows.
    """
    weights = fit.groupby("cluster_id").size()
    placement = assign_clusters(weights,
                               fractions=(1 - dev_fraction, dev_fraction),
                               names=("fit", "dev"))
    return (fit["cluster_id"].map(placement) == "dev").to_numpy()


def run_fold(cohort: pd.DataFrame, features: dict[str, np.ndarray],
             arm: Arm, held_out: str) -> tuple[np.ndarray, list[dict]]:
    """Fit ``arm``'s ensemble on every allele but ``held_out``; predict that one.

    ``features`` maps representation name to a matrix aligned row-for-row with
    ``cohort``. Returns the mean ensemble prediction for the held-out allele's
    rows, in cohort order, and one record per member.
    """
    is_out = (cohort["allele"] == held_out).to_numpy()
    if not is_out.any():
        raise ValueError(f"{held_out!r} has no rows in the cohort")
    fit_rows = cohort.loc[~is_out]
    is_dev = inner_dev_mask(fit_rows)
    y_fit = fit_rows["y_log1p"].to_numpy()

    members, records = [], []
    for source in arm.sources:
        hidden, l2 = arm.configs[source.name]
        X = features[source.name]
        X_fit = X[~is_out]
        for seed in arm.seeds:
            cfg = MLPConfig(hidden=hidden, l2=l2, seed=seed,
                            max_epochs=arm.max_epochs, patience=arm.patience)
            model = MLPRegressor(cfg).fit(X_fit[~is_dev], y_fit[~is_dev],
                                          X_fit[is_dev], y_fit[is_dev])
            members.append(model.predict(X[is_out]))
            records.append({
                "arm": arm.name, "held_out_allele": held_out,
                "representation": source.name, "seed": seed,
                "n_fit_rows": int((~is_dev).sum()), "n_dev_rows": int(is_dev.sum()),
                "n_eval_rows": int(is_out.sum()),
                "best_epoch": model.best_epoch_, "epochs": model.n_epochs_,
                "dev_mse": round(float(model.dev_mse_), 5),
                "fit_seconds": round(model.fit_seconds_, 2),
            })
    return np.mean(members, axis=0), records


def score_fold(cohort: pd.DataFrame, held_out: str, y_pred: np.ndarray) -> dict:
    """Metrics for one held-out allele.

    Spearman with average ranks (the floor makes ties common), MAE on ``log1p``,
    and precision@10 at 2 hours through the shared
    :func:`pepstab.evaluation.precision_at_k`, so the tie handling matches the
    frozen contract even though the panel does not.
    """
    rows = cohort[cohort["allele"] == held_out]
    y_true = rows["y_log1p"].to_numpy()
    y_pred = validate_finite(y_pred, f"{held_out}: predictions")
    if len(y_pred) != len(y_true):
        raise ValueError(f"{held_out}: {len(y_pred)} predictions for "
                         f"{len(y_true)} rows")

    rho = (float(stats.spearmanr(y_true, y_pred).statistic)
           if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1
           else float("nan"))
    prec = precision_at_k(rows["allele"], np.expm1(y_true), y_pred,
                          k=PRECISION_K, threshold_hours=STABILITY_THRESHOLD_HOURS)
    return {
        "held_out_allele": held_out,
        "n_rows": len(rows),
        "spearman": rho,
        "mae_log1p": float(np.abs(y_true - y_pred).mean()),
        "precision_at_10": float(prec["precision_at_k"].iloc[0]),
        "base_rate": float(prec["base_rate"].iloc[0]),
        "ceiling": float(prec["ceiling"].iloc[0]),
    }


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def _bootstrap_medians(values: np.ndarray, n_boot: int, rng) -> np.ndarray:
    picks = rng.integers(0, len(values), size=(n_boot, len(values)))
    return np.median(values[picks], axis=1)


def stratum_summary(per_allele: pd.DataFrame, cohort_info: pd.DataFrame,
                    n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED
                    ) -> pd.DataFrame:
    """Median per-allele Spearman per stratum, with a bootstrap interval.

    **The allele is the resampling unit**, because each allele is one fold and
    the folds are this design's independent replicates. Resampling rows instead
    would ignore that the whole fold stands or falls together.

    With 20-25 alleles per stratum the intervals are wide. That is the finding,
    not a presentational problem, so the width is reported rather than smoothed.
    """
    joined = per_allele.join(cohort_info[["stratum", "dist_to_nearest_fit_allele",
                                          "degenerate_pseudoseq"]],
                             on="held_out_allele")
    rng = np.random.default_rng(seed)
    rows = []
    groups = [(name, joined[joined["stratum"] == name])
              for name in DISTANCE_STRATA]
    groups.append(("near (d<=1), excl. shared-pseudoseq pair",
                   joined[(joined["stratum"] == "near (d<=1)")
                          & ~joined["degenerate_pseudoseq"]]))
    groups.append(("all strata pooled", joined))
    for name, grp in groups:
        vals = grp["spearman"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            continue
        boot = _bootstrap_medians(vals, n_boot, rng)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({
            "stratum": name,
            "n_alleles": len(vals),
            "n_rows": int(grp["n_rows"].sum()),
            "median_spearman": float(np.median(vals)),
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "iqr_low": float(np.percentile(vals, 25)),
            "iqr_high": float(np.percentile(vals, 75)),
            "min_spearman": float(vals.min()),
            "max_spearman": float(vals.max()),
            "median_mae_log1p": float(grp["mae_log1p"].median()),
            "median_precision_at_10": float(grp["precision_at_10"].median()),
        })
    return pd.DataFrame(rows)


def paired_allele_bootstrap(per_allele_a: pd.DataFrame, per_allele_b: pd.DataFrame,
                            cohort_info: pd.DataFrame, stratum: str | None = None,
                            n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED
                            ) -> dict:
    """Paired CI for ``b - a`` on the median per-allele Spearman.

    The same drawn alleles are scored for both arms, which is what makes the
    interval a statement about the *difference* rather than about two separately
    noisy medians. Ready for the ESM-2 arm: pass its per-allele frame as ``b``.
    """
    a = per_allele_a.set_index("held_out_allele")["spearman"]
    b = per_allele_b.set_index("held_out_allele")["spearman"]
    shared = a.index.intersection(b.index)
    if stratum is not None:
        in_stratum = cohort_info.index[cohort_info["stratum"] == stratum]
        shared = shared.intersection(in_stratum)
    a, b = a.loc[shared].to_numpy(float), b.loc[shared].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if not len(a):
        raise ValueError("no alleles scored by both arms")

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(a), size=(n_boot, len(a)))
    deltas = np.median(b[picks], axis=1) - np.median(a[picks], axis=1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "stratum": stratum or "all strata pooled",
        "n_alleles": int(len(a)),
        "delta_median_spearman": float(np.median(b) - np.median(a)),
        "ci95": (float(lo), float(hi)),
        "n_boot": n_boot,
    }


def sequence_arm(seeds: tuple[int, ...], input_set: str = "pep_pseudo") -> Arm:
    """The stage 2 sequence baseline, as a leave-allele-out arm.

    Configs are inherited from ``scripts/baseline_ensemble.SELECTED`` rather than
    re-tuned: nothing in stage 7b is selected on the allele axis.
    """
    from scripts.baseline_ensemble import SELECTED

    sources = tuple(FeatureSource(name=enc, kind="sequence",
                                  input_set=input_set, encoding=enc)
                    for enc in ("onehot", "blosum"))
    configs = {enc: SELECTED[(input_set, enc)] for enc in ("onehot", "blosum")}
    return Arm(name=f"seq_{input_set}", sources=sources, configs=configs,
               seeds=seeds)
