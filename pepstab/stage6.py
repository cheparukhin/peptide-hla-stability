"""Stage 6 analysis machinery: the arm-agnostic tables the submission rests on.

:mod:`pepstab.evaluation` is the frozen contract -- the primary metric, the
eligible-allele rule, precision@10, the distance strata and the paired cluster
bootstrap were all predeclared at stage 1 and nothing here changes them. This
module *reuses* those primitives and adds the five stage 6 analyses that the
contract names but does not implement:

1. **Distance stratification** -- census of the ``dist_to_train`` strata and the
   common allele panel, plus a scoring entry point that exposes the
   stratum row bar (:func:`score_strata`). At the frozen bar of 20 rows it
   reproduces :func:`pepstab.evaluation.score_by_distance` exactly; the bar is a
   parameter only because the *validation* split cannot support 20 (see
   :func:`stratum_panel` and ``reports/stage6_evaluation_machinery.md``).
2. **The differential target** -- delta log half-life between two alleles
   carrying the same peptide, which subtracts out everything intrinsic to the
   peptide (:func:`differential_pairs`, :func:`score_differential`).
3. **Paired uncertainty** -- a generic peptide-cluster bootstrap
   (:func:`paired_cluster_delta`) so every statistic in this module, not only
   the panel median, can carry a paired interval. The resampling unit is the
   peptide *cluster* from ``data/splits.csv``; :func:`resampling_unit_widths`
   measures what a row bootstrap would have understated.
4. **Precision@10** -- :func:`precision_report` wraps
   :func:`pepstab.evaluation.precision_at_k` unchanged (ties already credited by
   expectation) and adds lift over the base rate and the share of the reachable
   ceiling attained.
5. **Nested near-neighbour evaluation** -- mutant ranking inside the training
   split, on peptide groups at Hamming <= 2 (:func:`near_neighbour_groups`,
   :func:`mutant_pairs`, :func:`nested_folds`, :func:`score_mutant_ranking`).
   The frozen split clusters at Hamming <= 3, so every Hamming <= 2 group is
   contained in one split by construction -- verified, not assumed, by
   :func:`verify_groups_within_split`.

Everything is keyed on ``pair_id`` and takes predictions as a plain
``pair_id -> y_pred`` series, so an ESM-2 arm or a structural arm drops in with
no change. Nothing in this module knows which arm produced a column.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import evaluation
from .data import PEPTIDE_LENGTH, TARGET, encode_sequences, load_with_splits
from .evaluation import (
    BOOTSTRAP_SEED,
    DISTANCE_STRATA,
    MIN_ROWS_PER_ALLELE_IN_STRATUM,
    N_BOOTSTRAP,
    eligible_alleles,
    panel_spearman,
    per_allele_spearman,
    rankable_alleles,
    validate_finite,
)

#: Credit for a comparison the model cannot call either way. A tied prediction
#: on a decidable pair is worth the coin flip it is, exactly as
#: :func:`pepstab.evaluation.precision_at_k` credits ties by expectation. The
#: consequence is the property we want: a constant predictor scores 0.5 on every
#: concordance in this module, never better.
TIED_PREDICTION_CREDIT = 0.5

#: Hamming radius defining a "point mutant" group for the nested evaluation
#: (analysis 5). Strictly below the frozen split's clustering radius of 3, so
#: these groups never straddle a split.
MUTANT_RADIUS = 2

#: Folds for the nested near-neighbour cross-validation, and its seed.
NESTED_FOLDS = 5
NESTED_SEED = 20261004


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_split(split: str) -> pd.DataFrame:
    """The frozen split, sorted by ``pair_id``.

    Splits are read from ``data/splits.csv`` through
    :func:`pepstab.data.load_with_splits`; they are never recomputed, and the
    join is on ``pair_id`` as written by that loader, never positional.
    """
    df = load_with_splits()
    out = df[df["split"] == split]
    if out.empty:
        raise ValueError(f"no rows in split {split!r}; expected one of "
                         f"{sorted(df['split'].unique())}")
    return out.sort_values("pair_id").reset_index(drop=True)


def read_predictions(path: str | Path, frame: pd.DataFrame) -> pd.Series:
    """Read a ``pair_id,y_pred`` file and align it to ``frame``'s row order.

    Mirrors the validation in ``scripts/evaluate.py`` -- same two-column
    contract, same rejection of duplicates, non-numeric values and non-finite
    values -- but raises :class:`ValueError` instead of exiting, because this is
    a library. ``tests/test_stage6.py`` asserts the two agree on a real file.

    Predictions covering rows outside ``frame`` are ignored, so one file may
    cover several splits.
    """
    path = Path(path)
    preds = pd.read_csv(path)
    missing = {"pair_id", "y_pred"} - set(preds.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
    if preds["pair_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate pair_id rows")

    y = pd.to_numeric(preds["y_pred"], errors="coerce")
    bad = y.isna() & preds["y_pred"].notna()
    if bad.any():
        where = ", ".join(str(i) for i in preds.index[bad][:5])
        raise ValueError(f"{path}: {int(bad.sum()):,} non-numeric y_pred "
                         f"value(s), first at row(s) {where}")
    preds = preds.assign(y_pred=y)

    merged = frame[["pair_id"]].merge(preds, on="pair_id", how="left")
    if merged["y_pred"].isna().any():
        n = int(merged["y_pred"].isna().sum())
        raise ValueError(f"{path}: no prediction for {n:,} of {len(frame):,} "
                         "rows in this split")
    out = merged["y_pred"]
    validate_finite(out.to_numpy(), f"{path}: predictions")
    return out.rename(path.stem)


# --------------------------------------------------------------------------
# 1. nearest-neighbour distance stratification
# --------------------------------------------------------------------------

def distance_census(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows and distinct peptides at each ``dist_to_train``, with shares.

    Verifies the split composition the plan states rather than trusting it.
    ``dist_to_train`` is read from the frozen ``data/splits.csv``; it is not
    recomputed here.
    """
    g = frame.groupby("dist_to_train")
    out = pd.DataFrame({
        "n_rows": g.size(),
        "n_peptides": g["peptide"].nunique(),
        "n_alleles": g["allele"].nunique(),
    })
    out["row_share"] = out["n_rows"] / len(frame)
    return out.reset_index()


def stratum_panel(frame: pd.DataFrame,
                  min_rows: int = MIN_ROWS_PER_ALLELE_IN_STRATUM,
                  strata: dict[str, tuple[int, int]] | None = None) -> dict:
    """The common allele panel for the distance strata, and what it costs.

    Returns the per-stratum eligible sets, their intersection, and the share of
    each stratum's rows the intersection covers. The intersection is the panel
    both strata are scored on: without it the comparison would contrast allele
    panels rather than distances.

    Computed from the split and the labels alone, so it is identical for every
    model -- the same rule :func:`pepstab.evaluation.eligible_alleles` follows.
    """
    strata = strata or DISTANCE_STRATA
    per: dict[str, list[str]] = {}
    sizes: dict[str, int] = {}
    for label, (lo, hi) in strata.items():
        sub = frame[(frame["dist_to_train"] >= lo) & (frame["dist_to_train"] <= hi)]
        sizes[label] = len(sub)
        if sub.empty:
            per[label] = []
            continue
        per[label] = eligible_alleles(sub["allele"], sub[TARGET].to_numpy(),
                                      min_rows=min_rows)
    common = set.intersection(*(set(v) for v in per.values())) if per else set()
    coverage = {}
    for label, (lo, hi) in strata.items():
        sub = frame[(frame["dist_to_train"] >= lo) & (frame["dist_to_train"] <= hi)]
        coverage[label] = (float(sub["allele"].isin(common).mean())
                           if len(sub) else float("nan"))
    return {
        "min_rows": min_rows,
        "n_rows": sizes,
        "eligible": {k: sorted(v) for k, v in per.items()},
        "n_eligible": {k: len(v) for k, v in per.items()},
        "common": sorted(common),
        "n_common": len(common),
        "row_coverage": coverage,
    }


def score_strata(name: str, frame: pd.DataFrame, y_pred: np.ndarray,
                 min_rows: int = MIN_ROWS_PER_ALLELE_IN_STRATUM,
                 strata: dict[str, tuple[int, int]] | None = None) -> pd.DataFrame:
    """Score each distance stratum on the shared allele panel.

    At the frozen ``min_rows`` of 20 this is
    :func:`pepstab.evaluation.score_by_distance` -- same panel rule, same
    per-stratum :func:`pepstab.evaluation.score` call -- and
    ``tests/test_stage6.py`` asserts the two agree row for row. The bar is a
    parameter here only so the *validation* split, where a 20-row bar leaves a
    6-allele panel covering 15% of the d=4 rows, can be exercised at a workable
    bar. The test split keeps 20.
    """
    strata = strata or DISTANCE_STRATA
    y_pred = validate_finite(y_pred, f"{name}: predictions")
    y_true = validate_finite(frame[TARGET].to_numpy(), f"{name}: labels")
    panel = stratum_panel(frame, min_rows=min_rows, strata=strata)
    alleles = panel["common"]

    rows = []
    for label, (lo, hi) in strata.items():
        mask = ((frame["dist_to_train"] >= lo)
                & (frame["dist_to_train"] <= hi)).to_numpy()
        if not mask.any():
            continue
        if not alleles:
            rows.append({"model": name, "stratum": label,
                         "n_rows": int(mask.sum()), "n_alleles": 0,
                         "median_per_allele_spearman": float("nan"),
                         "mae_log1p": float(np.abs(y_true[mask]
                                                   - y_pred[mask]).mean())})
            continue
        s = evaluation.score(f"{name}[{label}]", frame["allele"][mask],
                             y_true[mask], y_pred[mask], alleles)
        row = s.as_row()
        row["model"], row["stratum"] = name, label
        rows.append(row)
    cols = ["model", "stratum", "n_rows", "n_alleles",
            "median_per_allele_spearman", "iqr_low", "iqr_high", "mae_log1p",
            "median_precision_at_10"]
    out = pd.DataFrame(rows)
    return out[[c for c in cols if c in out.columns]]


def stratum_median_delta(frame: pd.DataFrame, y_pred: np.ndarray,
                         min_rows: int = MIN_ROWS_PER_ALLELE_IN_STRATUM,
                         strata: dict[str, tuple[int, int]] | None = None) -> float:
    """``median rho(d=4) - median rho(d>=5)`` for one model, on the shared panel.

    Positive means the model does better on the peptides that sit closer to
    training, which is the signature of exploiting residual similarity at the
    split boundary. Exposed separately so :func:`paired_cluster_delta` can put a
    cluster-bootstrap interval on it.
    """
    table = score_strata("x", frame, y_pred, min_rows=min_rows, strata=strata)
    table = table.set_index("stratum")["median_per_allele_spearman"]
    labels = list((strata or DISTANCE_STRATA).keys())
    if len(labels) < 2 or not set(labels[:2]) <= set(table.index):
        return float("nan")
    return float(table[labels[0]] - table[labels[1]])


# --------------------------------------------------------------------------
# 2. the differential target
# --------------------------------------------------------------------------

#: Minimum shared peptides before an allele *pair* gets its own Spearman in the
#: per-allele-pair table. Below this the correlation is noise; the pair still
#: counts in the pooled and concordance numbers.
MIN_PEPTIDES_PER_ALLELE_PAIR = 10


def differential_pairs(frame: pd.DataFrame,
                       alleles: list[str] | None = None) -> pd.DataFrame:
    """Every (peptide, allele_a, allele_b) comparison available in ``frame``.

    **Definition, fixed before any number was computed.**

    - *Which pairs.* For each peptide measured on two or more alleles within the
      scored rows, every unordered pair of those alleles, canonicalised so that
      ``allele_a < allele_b`` lexicographically. Ordering is a convention, not a
      choice: ``delta`` changes sign with the order, and every metric below is
      sign-symmetric, so the canonical order removes the double counting without
      affecting anything else. A peptide on ``k`` alleles contributes
      ``k(k-1)/2`` pairs.
    - *The quantity.* ``delta_true = y_log1p(b) - y_log1p(a)`` and
      ``delta_pred`` likewise. Both rows carry the same peptide, so anything
      intrinsic to the peptide -- including any peptide-only term a model has
      learned -- cancels exactly. What is left is groove chemistry.
    - *Ties.* A pair with ``delta_true == 0`` is **undecidable**: the two alleles
      hold the peptide equally long (very often both at the assay floor), so no
      prediction can be right or wrong about the direction. Undecidable pairs are
      excluded from concordance and reported as a count, never silently dropped.
      A *predicted* tie on a decidable pair is credited
      :data:`TIED_PREDICTION_CREDIT`.
    - *Aggregation.* Reported three ways, because they answer different
      questions and can disagree: pair-level (every comparison counts once,
      so a peptide on 36 alleles carries 630 of them), peptide-weighted (each
      peptide's pairs average to one number first, so every peptide counts
      once), and per-allele-pair (a Spearman across the peptides shared by one
      allele pair, then the median over pairs with at least
      :data:`MIN_PEPTIDES_PER_ALLELE_PAIR` of them).

    Restricted to ``alleles`` when given -- pass the split's eligible panel so
    every arm is compared on the same allele set.

    Returns one row per comparison with ``pair_id_a``/``pair_id_b`` so a
    prediction column joins on without any positional assumption.
    """
    cols = ["pair_id", "peptide", "allele", "cluster_id", TARGET]
    sub = frame[cols]
    if alleles is not None:
        sub = sub[sub["allele"].isin(alleles)]
    sub = sub.sort_values(["peptide", "allele"])

    counts = sub.groupby("peptide")["allele"].nunique()
    sub = sub[sub["peptide"].isin(counts.index[counts >= 2])]

    left = sub.rename(columns={"pair_id": "pair_id_a", "allele": "allele_a",
                               TARGET: "y_a", "cluster_id": "cluster_id_a"})
    right = sub.rename(columns={"pair_id": "pair_id_b", "allele": "allele_b",
                                TARGET: "y_b", "cluster_id": "cluster_id_b"})
    merged = left.merge(right, on="peptide")
    merged = merged[merged["allele_a"] < merged["allele_b"]]

    out = merged[["peptide", "cluster_id_a", "allele_a", "allele_b",
                  "pair_id_a", "pair_id_b", "y_a", "y_b"]].copy()
    out = out.rename(columns={"cluster_id_a": "cluster_id"})
    out["delta_true"] = out["y_b"] - out["y_a"]
    return out.sort_values(["peptide", "allele_a", "allele_b"]).reset_index(drop=True)


def _concordance(delta_true: np.ndarray, delta_pred: np.ndarray) -> np.ndarray:
    """Per-pair credit in [0, 1] on the decidable pairs only (caller filters)."""
    agree = np.sign(delta_true) == np.sign(delta_pred)
    credit = agree.astype(float)
    credit[delta_pred == 0] = TIED_PREDICTION_CREDIT
    return credit


def score_differential(name: str, pairs: pd.DataFrame,
                       pred_by_pair_id: pd.Series) -> dict:
    """Score one arm on the differential target defined by :func:`differential_pairs`.

    ``pred_by_pair_id`` is a ``pair_id -> y_pred`` series on the log1p scale.
    """
    lookup = pd.Series(pred_by_pair_id.to_numpy(dtype=float),
                       index=pd.Index(pred_by_pair_id.index, name="pair_id"))
    validate_finite(lookup.to_numpy(), f"{name}: predictions")
    pa = lookup.reindex(pairs["pair_id_a"].to_numpy()).to_numpy()
    pb = lookup.reindex(pairs["pair_id_b"].to_numpy()).to_numpy()
    if np.isnan(pa).any() or np.isnan(pb).any():
        raise ValueError(f"{name}: predictions do not cover every differential pair")

    delta_pred = pb - pa
    delta_true = pairs["delta_true"].to_numpy(dtype=float)
    decidable = delta_true != 0

    credit = _concordance(delta_true[decidable], delta_pred[decidable])
    per_peptide = (pd.DataFrame({"peptide": pairs["peptide"].to_numpy()[decidable],
                                 "credit": credit})
                   .groupby("peptide")["credit"].mean())

    rho_pooled = (float(stats.spearmanr(delta_true[decidable],
                                        delta_pred[decidable]).statistic)
                  if decidable.sum() >= 2
                  and len(np.unique(delta_pred[decidable])) >= 2 else float("nan"))

    per_pair = []
    for (a, b), grp in pairs.groupby(["allele_a", "allele_b"]):
        idx = grp.index.to_numpy()
        dt = delta_true[idx]
        dp = delta_pred[idx]
        if len(grp) < MIN_PEPTIDES_PER_ALLELE_PAIR:
            continue
        rho = (float(stats.spearmanr(dt, dp).statistic)
               if len(np.unique(dt)) >= 2 and len(np.unique(dp)) >= 2
               else float("nan"))
        dec = dt != 0
        per_pair.append({
            "allele_a": a, "allele_b": b, "n_peptides": len(grp),
            "n_decidable": int(dec.sum()),
            "spearman_delta": rho,
            "concordance": (float(_concordance(dt[dec], dp[dec]).mean())
                            if dec.any() else float("nan")),
        })
    pair_table = pd.DataFrame(per_pair)
    # An allele pair the model ranks constantly carries no information; the
    # predeclared convention for that (EVALUATION.md) is 0, applied here too so
    # a model cannot raise its median by going flat on the hard allele pairs.
    panel = (pair_table["spearman_delta"].fillna(evaluation.UNRANKED_CONTRIBUTION)
             if len(pair_table) else pd.Series(dtype=float))

    return {
        "model": name,
        "n_peptides": int(pairs["peptide"].nunique()),
        "n_pairs": int(len(pairs)),
        "n_decidable": int(decidable.sum()),
        "n_undecidable": int((~decidable).sum()),
        "concordance": float(credit.mean()) if decidable.any() else float("nan"),
        "concordance_peptide_weighted": (float(per_peptide.mean())
                                         if len(per_peptide) else float("nan")),
        "spearman_delta_pooled": rho_pooled,
        "n_allele_pairs_scored": int(len(pair_table)),
        "median_allele_pair_spearman": (float(panel.median())
                                        if len(panel) else float("nan")),
        "mae_delta_log1p": float(np.abs(delta_true - delta_pred).mean()),
        "_per_allele_pair": pair_table,
    }


def delta_concordance_statistic(pairs: pd.DataFrame):
    """A ``(idx, delta_pred) -> float`` concordance statistic over pair rows.

    The shape :func:`paired_cluster_delta` wants, for any table of comparisons
    carrying a ``delta_true`` column -- the differential pairs of analysis 2 and
    the mutant pairs of analysis 5 both qualify. ``delta_pred`` is indexed like
    ``pairs``, so the bootstrap resamples *comparisons* and never has to re-join
    predictions.

    Resampling these rows by ``cluster_id`` keeps the peptide cluster as the
    independence unit: a mutant pair lies wholly inside one cluster, and a
    differential pair shares one peptide and therefore one cluster, so no
    comparison is ever torn in half by a resample.
    """
    delta_true = pairs["delta_true"].to_numpy(dtype=float)

    def stat(idx: np.ndarray, delta_pred: np.ndarray) -> float:
        dt = delta_true[idx]
        dp = delta_pred[idx]
        dec = dt != 0
        if not dec.any():
            return float("nan")
        return float(_concordance(dt[dec], dp[dec]).mean())

    return stat


def differential_concordance(pairs: pd.DataFrame, delta_pred: np.ndarray) -> float:
    """Pair-level concordance from precomputed deltas, over all of ``pairs``."""
    return delta_concordance_statistic(pairs)(np.arange(len(pairs)), delta_pred)


# --------------------------------------------------------------------------
# 3. paired uncertainty on the peptide cluster
# --------------------------------------------------------------------------

def _cluster_index(cluster_id: np.ndarray) -> tuple[np.ndarray, dict]:
    uniq = np.unique(cluster_id)
    return uniq, {c: np.nonzero(cluster_id == c)[0] for c in uniq}


def paired_cluster_delta(cluster_id, statistic, pred_a: np.ndarray,
                         pred_b: np.ndarray, n_boot: int = N_BOOTSTRAP,
                         seed: int = BOOTSTRAP_SEED,
                         label: str = "statistic") -> dict:
    """Paired cluster bootstrap for ``statistic(b) - statistic(a)``, generic.

    ``statistic(idx, pred)`` takes the resampled row indices and one prediction
    vector and returns a float. Both arms are evaluated on the *same* resample,
    which is what makes the interval paired: the shared sampling noise cancels
    instead of being counted twice.

    **The resampling unit is the peptide cluster** from ``data/splits.csv``
    (``cluster_id``), not the row and not the peptide. One peptide is measured on
    up to 36 alleles, and near-identical peptides inside a cluster are not
    independent of each other either, so a row bootstrap treats correlated rows
    as fresh evidence and returns an interval that is too narrow.
    :func:`resampling_unit_widths` measures the difference on real data rather
    than asserting it.

    Whole clusters are drawn with replacement, ``len(clusters)`` of them, and all
    of a drawn cluster's rows enter the resample together.

    :func:`pepstab.evaluation.paired_cluster_bootstrap` remains the contract
    implementation for the primary metric; this is the same procedure, same
    seed, for the statistics the contract does not cover.
    """
    cluster_id = np.asarray(cluster_id)
    pred_a = validate_finite(pred_a, f"{label}: baseline predictions")
    pred_b = validate_finite(pred_b, f"{label}: comparison predictions")
    uniq, rows_by_cluster = _cluster_index(cluster_id)
    rng = np.random.default_rng(seed)

    full = np.arange(len(cluster_id))
    observed = statistic(full, pred_b) - statistic(full, pred_a)

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_by_cluster[c] for c in picked])
        deltas[i] = statistic(idx, pred_b) - statistic(idx, pred_a)

    finite = deltas[np.isfinite(deltas)]
    lo, hi = (np.percentile(finite, [2.5, 97.5]) if len(finite)
              else (float("nan"), float("nan")))
    return {
        "statistic": label,
        "delta": float(observed),
        "ci95": (float(lo), float(hi)),
        "n_boot": n_boot,
        "n_clusters": int(len(uniq)),
        "n_degenerate_resamples": int(n_boot - len(finite)),
        "conclusive": bool(np.isfinite(lo) and np.isfinite(hi)
                           and (lo > 0 or hi < 0)),
    }


#: Seeds the resampling-unit comparison averages over. A single bootstrap's CI
#: *width* is itself noisy -- at 60-120 resamples the row/cluster ratio moved
#: between 0.73 and 0.90 -- so the comparison is run at several seeds and the
#: spread is reported next to the mean. One seed would invite quoting noise.
UNIT_COMPARISON_SEEDS = (BOOTSTRAP_SEED, BOOTSTRAP_SEED + 1, BOOTSTRAP_SEED + 2)


def resampling_unit_widths(frame: pd.DataFrame, pred_a: np.ndarray,
                           pred_b: np.ndarray, alleles: list[str],
                           n_boot: int = 1000,
                           seeds: tuple[int, ...] = UNIT_COMPARISON_SEEDS
                           ) -> pd.DataFrame:
    """How wide the interval is under three resampling units, over several seeds.

    Evidence for the contract's choice rather than an assertion of it. Only the
    unit resampled changes: ``cluster`` (the contract), ``peptide``, and ``row``.
    A narrower interval here is not a better one -- it is an interval that has
    counted correlated rows as independent evidence.

    Two quantities, because they separate cleanly:

    - ``paired delta`` -- the contract's comparison, ``b`` minus ``a``.
    - ``single arm`` -- ``b``'s own panel median, obtained by pairing it against
      a constant predictor (which scores 0 by the predeclared rule). Pairing
      already removes the sampling noise two arms share, so the dependence
      between rows shows more plainly here.

    Returns one row per (seed, unit, quantity); aggregate with
    ``groupby(["quantity", "unit"])["ci_width"]``.
    """
    y_true = frame[TARGET].to_numpy(dtype=float)
    allele = frame["allele"].to_numpy()
    units = {
        "cluster": frame["cluster_id"].to_numpy(),
        "peptide": frame["peptide"].to_numpy(),
        "row": np.arange(len(frame)),
    }

    def median_rho(idx: np.ndarray, pred: np.ndarray) -> float:
        rankable = rankable_alleles(allele[idx], y_true[idx], alleles)
        if not rankable:
            return float("nan")
        rho = per_allele_spearman(allele[idx], y_true[idx], pred[idx], alleles)
        return float(panel_spearman(rho, rankable).median())

    constant = np.zeros(len(frame))
    quantities = {"paired delta": (pred_a, pred_b),
                  "single arm": (constant, pred_b)}

    rows = []
    for seed in seeds:
        for unit, ids in units.items():
            for quantity, (qa, qb) in quantities.items():
                r = paired_cluster_delta(ids, median_rho, qa, qb,
                                         n_boot=n_boot, seed=seed, label=unit)
                lo, hi = r["ci95"]
                rows.append({"seed": seed, "unit": unit, "quantity": quantity,
                             "n_units": r["n_clusters"], "estimate": r["delta"],
                             "ci_low": lo, "ci_high": hi, "ci_width": hi - lo})
    out = pd.DataFrame(rows)
    ref = (out[out["unit"] == "cluster"]
           .groupby("quantity")["ci_width"].mean())
    out["width_vs_cluster"] = out["ci_width"] / out["quantity"].map(ref)
    return out


# --------------------------------------------------------------------------
# 4. precision@10
# --------------------------------------------------------------------------

def precision_report(name: str, frame: pd.DataFrame, y_pred: np.ndarray,
                     alleles: list[str] | None = None,
                     k: int = evaluation.PRECISION_K,
                     threshold_hours: float = evaluation.STABILITY_THRESHOLD_HOURS
                     ) -> pd.DataFrame:
    """Per-allele precision@k with base rate, ceiling, lift and ceiling share.

    The precision itself comes straight from
    :func:`pepstab.evaluation.precision_at_k` -- unchanged, including its
    tie-by-expectation rule, which is what makes a constant predictor score
    exactly its base rate (``tests/test_stage6.py`` asserts that property).
    This function only adds the two comparisons a bare precision cannot
    support:

    - ``lift`` -- precision@k minus the allele's base rate. Zero means the top
      10 are no better than a random 10.
    - ``ceiling_share`` -- precision@k divided by the best precision@k reachable
      on that allele. An allele with 4 positives in 60 rows caps at 0.4, so 0.4
      there is a perfect score and 0.4 on an allele with 30 positives is not.
      Left NaN where the ceiling is 0 -- an allele with no peptide above the
      threshold scores 0 for every model and there is nothing to be a share of.
    """
    if alleles is None:
        alleles = eligible_alleles(frame["allele"], frame[TARGET].to_numpy(),
                                   split=str(frame["split"].iloc[0]))
    y_pred = validate_finite(y_pred, f"{name}: predictions")
    hours = np.expm1(frame[TARGET].to_numpy(dtype=float))
    table = evaluation.precision_at_k(frame["allele"], hours, y_pred, alleles,
                                      k=k, threshold_hours=threshold_hours)
    table.insert(0, "model", name)
    table["lift"] = table["precision_at_k"] - table["base_rate"]
    with np.errstate(invalid="ignore", divide="ignore"):
        table["ceiling_share"] = np.where(table["ceiling"] > 0,
                                          table["precision_at_k"] / table["ceiling"],
                                          np.nan)
    return table


def precision_summary(table: pd.DataFrame) -> dict:
    """Panel aggregates for one arm's :func:`precision_report` table.

    ``n_alleles_no_positives`` is reported next to ``n_alleles_at_ceiling``
    because those alleles are counted in it: with no peptide above the
    threshold, 0 *is* the ceiling, and every model reaches it. Hiding them
    would let an impossible allele read as a success.
    """
    reachable = table["ceiling"] > 0
    return {
        "model": str(table["model"].iloc[0]),
        "n_alleles": int(len(table)),
        "median_precision_at_10": float(table["precision_at_k"].median()),
        "median_base_rate": float(table["base_rate"].median()),
        "median_ceiling": float(table["ceiling"].median()),
        "median_lift": float(table["lift"].median()),
        "median_ceiling_share": float(table.loc[reachable, "ceiling_share"].median()),
        "n_alleles_at_ceiling": int((table["precision_at_k"]
                                     >= table["ceiling"] - 1e-9).sum()),
        "n_alleles_no_positives": int((~reachable).sum()),
        "n_alleles_below_base_rate": int((table["lift"] < 0).sum()),
    }


def median_precision(frame: pd.DataFrame, alleles: list[str],
                     k: int = evaluation.PRECISION_K,
                     threshold_hours: float = evaluation.STABILITY_THRESHOLD_HOURS):
    """A ``(idx, pred) -> float`` statistic for :func:`paired_cluster_delta`."""
    hours = np.expm1(frame[TARGET].to_numpy(dtype=float))
    allele = frame["allele"].to_numpy()

    def stat(idx: np.ndarray, pred: np.ndarray) -> float:
        t = evaluation.precision_at_k(pd.Series(allele[idx]), hours[idx],
                                      pred[idx], alleles, k=k,
                                      threshold_hours=threshold_hours)
        return float(t["precision_at_k"].median()) if len(t) else float("nan")

    return stat


# --------------------------------------------------------------------------
# 5. nested near-neighbour evaluation inside the training split
# --------------------------------------------------------------------------

def near_neighbour_groups(frame: pd.DataFrame,
                          radius: int = MUTANT_RADIUS) -> pd.Series:
    """Single-linkage peptide groups at Hamming <= ``radius``, within ``frame``.

    This is **not** a split. The frozen split assignments in
    ``data/splits.csv`` are untouched and never recomputed; this is a nested
    grouping *inside* one split, used only to build cross-validation folds.
    Because the frozen split clusters at Hamming <= 3 and ``radius`` is strictly
    smaller, every group here is a subset of exactly one frozen cluster and
    therefore of exactly one split -- :func:`verify_groups_within_split`
    checks that on the real data instead of assuming it.

    Returns a ``peptide -> group_id`` series.
    """
    if radius >= 3:
        raise ValueError(
            f"radius={radius} is not strictly below the frozen clustering "
            "radius of 3; groups could then straddle a split")
    from .splits import single_linkage_clusters

    peptides = sorted(frame["peptide"].unique())
    codes = encode_sequences(peptides, PEPTIDE_LENGTH)
    labels = single_linkage_clusters(codes, threshold=radius)
    return pd.Series(labels, index=pd.Index(peptides, name="peptide"),
                     name="nn_group")


def verify_groups_within_split(radius: int = MUTANT_RADIUS) -> dict:
    """Check that no Hamming <= ``radius`` peptide group straddles a frozen split.

    Run over the *whole* dataset, not one split, because that is the claim: the
    nested evaluation can stay inside training only if these groups do.
    """
    df = load_with_splits()
    groups = near_neighbour_groups(df, radius=radius)
    spanning = (df.assign(g=df["peptide"].map(groups))
                .groupby("g")["split"].nunique())
    return {"radius": radius, "n_groups": int(len(spanning)),
            "n_groups_spanning_splits": int((spanning > 1).sum())}


def mutant_pairs(frame: pd.DataFrame, radius: int = MUTANT_RADIUS) -> pd.DataFrame:
    """Same-allele peptide pairs at 1 <= Hamming <= ``radius``.

    The unit of the mutant-ranking question: two peptides that differ by one or
    two substitutions, both measured on the *same* allele. Asking whether the
    model orders them correctly is the question the grouped split deliberately
    cannot ask, because it places every such pair on the same side of the split.

    Same tie rules as the differential target: a pair with equal labels is
    undecidable and excluded from concordance; a predicted tie on a decidable
    pair is credited :data:`TIED_PREDICTION_CREDIT`.
    """
    peptides = np.array(sorted(frame["peptide"].unique()))
    codes = encode_sequences(peptides.tolist(), PEPTIDE_LENGTH)
    blocks = []  # chunked neighbour search; the full matrix would be 5.6k^2
    for start in range(0, len(codes), 512):
        block = codes[start:start + 512]
        dist = (block[:, None, :] != codes[None, :, :]).sum(axis=2)
        r, c = np.nonzero((dist >= 1) & (dist <= radius))
        keep = (r + start) < c  # upper triangle only: each pair once
        blocks.append(np.stack([r[keep] + start, c[keep],
                                dist[r[keep], c[keep]]], axis=1))
    near = np.concatenate(blocks) if blocks else np.empty((0, 3), dtype=int)

    pep_pairs = pd.DataFrame({"peptide_a": peptides[near[:, 0]],
                              "peptide_b": peptides[near[:, 1]],
                              "hamming": near[:, 2].astype(int)})
    rows = frame[["pair_id", "peptide", "allele", "cluster_id", TARGET]]
    left = rows.rename(columns={"peptide": "peptide_a", "pair_id": "pair_id_a",
                                TARGET: "y_a"})
    right = (rows.drop(columns="cluster_id")
             .rename(columns={"peptide": "peptide_b", "pair_id": "pair_id_b",
                              TARGET: "y_b"}))
    out = (pep_pairs.merge(left, on="peptide_a")
           .merge(right, on=["peptide_b", "allele"]))
    out["delta_true"] = out["y_b"] - out["y_a"]
    cols_out = ["peptide_a", "peptide_b", "hamming", "allele", "cluster_id",
                "pair_id_a", "pair_id_b", "y_a", "y_b", "delta_true"]
    return out[cols_out].sort_values(["peptide_a", "peptide_b", "allele"]
                                     ).reset_index(drop=True)


def nested_folds(frame: pd.DataFrame, radius: int = MUTANT_RADIUS,
                 n_folds: int = NESTED_FOLDS, seed: int = NESTED_SEED) -> pd.Series:
    """Assign each row of ``frame`` to a cross-validation fold, by **peptide**.

    The fold unit is the peptide, not the row and not the near-neighbour group.
    That is the whole design: a mutant pair must be *split across* folds, so the
    model sees one member while predicting the other. Folding by group would
    hold out whole mutant families and reproduce the grouped split's blind spot
    at smaller scale.

    Leakage that is still excluded: all rows of a held-out peptide leave
    together, across every allele, so no allele can leak the peptide's label.

    Folds are balanced by row count, largest peptide first, deterministically
    (ties break by peptide). ``seed`` permutes equal-sized peptides so the
    assignment is not an artifact of alphabetical order.
    """
    counts = frame.groupby("peptide").size()
    rng = np.random.default_rng(seed)
    jitter = pd.Series(rng.random(len(counts)), index=counts.index)
    order = sorted(counts.index, key=lambda p: (-int(counts[p]), float(jitter[p]), p))

    load = np.zeros(n_folds)
    assignment: dict[str, int] = {}
    for pep in order:
        f = int(np.argmin(load))
        assignment[pep] = f
        load[f] += float(counts[pep])
    return frame["peptide"].map(assignment).astype(int)


def ridge_oof_predictions(frame: pd.DataFrame, folds: pd.Series,
                          input_set: str = "pep_pseudo",
                          encoding: str = "blosum", alpha: float = 10.0
                          ) -> pd.Series:
    """Out-of-fold predictions from a ridge reference arm. CPU, seconds.

    A *reference* implementation so the nested harness can be exercised and
    validated end to end without waiting on another workstream. Any arm plugs in
    by supplying its own out-of-fold prediction series on the same folds --
    nothing downstream knows how these numbers were produced.
    """
    from sklearn.linear_model import Ridge

    from .features import build_features

    X = build_features(frame, input_set, encoding)
    y = frame[TARGET].to_numpy(dtype=float)
    out = np.empty(len(frame))
    for f in sorted(folds.unique()):
        test = (folds == f).to_numpy()
        model = Ridge(alpha=alpha).fit(X[~test], y[~test])
        out[test] = model.predict(X[test])
    return pd.Series(out, index=frame["pair_id"].to_numpy(), name="oof")


def score_mutant_ranking(name: str, pairs: pd.DataFrame,
                         pred_by_pair_id: pd.Series,
                         folds_by_pair_id: pd.Series | None = None) -> dict:
    """Score mutant ranking on :func:`mutant_pairs` with out-of-fold predictions.

    ``folds_by_pair_id``, when given, restricts scoring to pairs whose two rows
    sit in **different** folds. That is the honest subset: if both members were
    held out together the model never saw either, and if both were in training
    neither prediction is out-of-sample. Either way the count is reported.
    """
    lookup = pd.Series(pred_by_pair_id.to_numpy(dtype=float),
                       index=pd.Index(pred_by_pair_id.index, name="pair_id"))
    validate_finite(lookup.to_numpy(), f"{name}: predictions")
    use = pairs
    n_all = len(pairs)
    if folds_by_pair_id is not None:
        fa = folds_by_pair_id.reindex(pairs["pair_id_a"].to_numpy()).to_numpy()
        fb = folds_by_pair_id.reindex(pairs["pair_id_b"].to_numpy()).to_numpy()
        use = pairs[fa != fb]

    pa = lookup.reindex(use["pair_id_a"].to_numpy()).to_numpy()
    pb = lookup.reindex(use["pair_id_b"].to_numpy()).to_numpy()
    if np.isnan(pa).any() or np.isnan(pb).any():
        raise ValueError(f"{name}: predictions do not cover every mutant pair")
    delta_pred = pb - pa
    delta_true = use["delta_true"].to_numpy(dtype=float)
    dec = delta_true != 0

    out = {
        "model": name,
        "n_pairs_available": n_all,
        "n_pairs_scored": int(len(use)),
        "n_decidable": int(dec.sum()),
        "n_undecidable": int((~dec).sum()),
        "n_alleles": int(use["allele"].nunique()) if len(use) else 0,
        "n_peptides": int(pd.unique(np.concatenate(
            [use["peptide_a"].to_numpy(), use["peptide_b"].to_numpy()])).size)
            if len(use) else 0,
        "concordance": (float(_concordance(delta_true[dec],
                                           delta_pred[dec]).mean())
                        if dec.any() else float("nan")),
        "mae_delta_log1p": (float(np.abs(delta_true - delta_pred).mean())
                            if len(use) else float("nan")),
    }
    for h in sorted(use["hamming"].unique()) if len(use) else []:
        m = (use["hamming"].to_numpy() == h) & dec
        out[f"concordance_d{int(h)}"] = (
            float(_concordance(delta_true[m], delta_pred[m]).mean())
            if m.any() else float("nan"))
        out[f"n_decidable_d{int(h)}"] = int(m.sum())
    return out


#: Mutant-pair concordance uses the same statistic factory as the differential
#: target: both are "did the model get the direction of this delta right", only
#: the comparison table differs.
mutant_concordance_statistic = delta_concordance_statistic
