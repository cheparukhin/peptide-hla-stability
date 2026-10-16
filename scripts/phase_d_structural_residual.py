#!/usr/bin/env python
"""Phase D: a strongly shrunk structural residual correction, on existing folds.

Protocol, frozen before any arm was fitted:
``reports/further_research/phase_d_protocol.md``. This script implements it and
refuses to deviate: the ladder, the arms, the fold contract, the selection rule
and the eligibility bar are module constants, asserted at run time.

    .venv/bin/python scripts/phase_d_structural_residual.py --reference-arm ridge
    .venv/bin/python scripts/phase_d_structural_residual.py \
        --seq-oof reports/further_research/phase_a_seq_oof_cluster.csv \
        --outer-folds reports/further_research/phase_a_outer_folds.csv

No folding, no GPU. Every structural number comes off disk.

The thing most likely to go wrong here is not the model, it is the plumbing:
a residual harness that accidentally fits on the rows it later scores will
report a large, clean, entirely fake improvement. Three defences:

1. **Training split only.** Validation and test are never read. Asserted.
2. **Outer folds are whole peptide clusters**, from
   ``baseline_ensemble.cv_folds`` imported rather than reimplemented, and
   fingerprinted so a silent change is fatal.
3. **A shuffled-structure control (DX).** Its columns are row-permuted inside
   the fitting partition. If DX improves on held-out clusters, the harness
   leaks and no other arm's number may be reported. Sign test only -- a
   width-matched shuffled block is not a capacity control, so its effect size
   is not interpretable and is never quoted as one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import (  # noqa: E402
    N_BOOTSTRAP, BOOTSTRAP_SEED, describe_delta, eligible_alleles,
    paired_cluster_bootstrap, per_allele_spearman, score,
)
from pepstab.features import build_features  # noqa: E402
from scripts.baseline_ensemble import cv_folds  # noqa: E402
from scripts.stage5_structural_arm import (  # noqa: E402
    CONFIDENCE_PREFIXES, GEOMETRY_PREFIXES, NON_FEATURE,
)

OUT = REPO / "reports" / "further_research"
PRODUCTION = REPO / "reports" / "stage4c5_features_production.csv"

N_OUTER = 5
N_INNER = 5
#: sha256 (first 16 hex) of the int64 outer-fold vector over the training split,
#: sorted by pair_id. Recorded in the protocol; a mismatch is fatal.
OUTER_FOLD_FINGERPRINT = "50fdca745c9e8e99"

#: The whole search space, §5 of the protocol. `inf` is the exact
#: no-correction point, so "do not correct" is reachable by selection.
ALPHA_LADDER = (1e1, 1e2, 1e3, 1e4, 1e5, 1e6, np.inf)

#: §7. Phase A's bar, adopted for cross-phase panel consistency (§9).
MIN_ROWS = 20
#: Declared sensitivity check, not a second primary.
MIN_ROWS_STRICT = 50

#: Fixed seed for the DX shuffled control, so it is reproducible and cannot be
#: re-rolled until it behaves.
SHUFFLE_SEED = 20261004

_POSITIONAL = re.compile(r"_p([1-9])$")


# --------------------------------------------------------------------------
# feature groups
# --------------------------------------------------------------------------


def structural_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Geometry and confidence columns, using the structural arm's own groups.

    Imported prefixes rather than a fresh list: the grouping is the one stage 5
    already committed to, so Phase D cannot quietly regroup a feature into
    whichever block it helps.
    """
    geometry, confidence = [], []
    for col in frame.columns:
        if col in NON_FEATURE or col.endswith("_sha256") or col.startswith("msa_"):
            continue
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        if col.startswith(GEOMETRY_PREFIXES):
            geometry.append(col)
        elif col.startswith(CONFIDENCE_PREFIXES):
            confidence.append(col)
    return geometry, confidence


def positions_of(cols: list[str]) -> dict[str, int | None]:
    out = {}
    for c in cols:
        m = _POSITIONAL.search(c)
        out[c] = int(m.group(1)) if m else None
    return out


def build_arms(geometry: list[str], confidence: list[str]) -> dict[str, list[str]]:
    """The prespecified arms of §3. Order is the report's row order."""
    pos = positions_of(geometry)
    aggregate = [c for c in geometry if pos[c] is None]

    def at(keep: set[int]) -> list[str]:
        # Aggregates are kept in every geometry subset: an anchor-focused arm
        # restricts *which positions* are resolved, not whether the peptide's
        # overall burial and contact count are available. Dropping aggregates
        # too would confound position selection with losing the summary.
        return aggregate + [c for c in geometry if pos[c] in keep]

    return {
        "D0_none": [],
        "D1_geometry": geometry,
        "D2_confidence": confidence,
        "D3_anchor_geometry": at({2, 9}),
        "D3b_anchor_geometry_p1p2p9": at({1, 2, 9}),
        "D4_geometry+confidence": geometry + confidence,
        "D3c_central_geometry": at({3, 4, 5, 6, 7, 8}),
        "DX_geometry_shuffled": geometry,
    }


#: Arms whose number may be read as an effect; DX is excluded by construction.
CANDIDATE_ARMS = ("D0_none", "D1_geometry", "D2_confidence", "D3_anchor_geometry",
                  "D3b_anchor_geometry_p1p2p9", "D4_geometry+confidence",
                  "D3c_central_geometry")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def load_training_frame() -> pd.DataFrame:
    """Training rows only, joined to the structural table on (allele, peptide)."""
    labels = load_with_splits()
    train = labels[labels.split == "train"].sort_values("pair_id").reset_index(drop=True)
    assert (train.split == "train").all(), "non-training row reached the frame"

    feat = pd.read_csv(PRODUCTION)
    for col in ("pair_id", "split", "cluster_id", "y_log1p", "dist_to_train",
                "hla_seq", "hla_pseudoseq"):
        if col in feat:
            feat = feat.drop(columns=[col])
    if "status" in feat:
        bad = int((feat.status != "ok").sum())
        if bad:
            raise SystemExit(f"{bad} structural rows are not status=ok")
    merged = train.merge(feat, on=["allele", "peptide"], how="left",
                         validate="one_to_one")
    missing = int(merged[feat.columns[0]].isna().sum()) if len(feat.columns) else 0
    if missing:
        raise SystemExit(
            f"{missing} training rows have no structure. Phase D was specified "
            "against 100% coverage; a fallback rule would have to be declared "
            "before, not after, seeing this."
        )
    return merged


def outer_assignment(train: pd.DataFrame, path: Path | None) -> np.ndarray:
    """Outer folds: read from Phase A's file if given, else recomputed. Both
    are fingerprinted against the protocol's recorded hash."""
    if path is not None:
        got = pd.read_csv(path)
        m = train[["pair_id"]].merge(got, on="pair_id", how="left", validate="one_to_one")
        if m.outer_fold.isna().any():
            raise SystemExit(f"{path} does not cover every training pair_id")
        folds = m.outer_fold.to_numpy(dtype=np.int64)
        source = str(Path(path).resolve())
    else:
        folds = np.asarray(cv_folds(train, N_OUTER), dtype=np.int64)
        source = "recomputed from scripts.baseline_ensemble.cv_folds"
    digest = hashlib.sha256(np.ascontiguousarray(folds).tobytes()).hexdigest()[:16]
    if digest != OUTER_FOLD_FINGERPRINT:
        raise SystemExit(
            f"outer fold fingerprint {digest} != protocol's "
            f"{OUTER_FOLD_FINGERPRINT} (source: {source}). The fold contract "
            "changed; the protocol must be amended before any arm is rerun."
        )
    # Clusters must not straddle outer folds, or the assessment is not
    # cluster-held-out at all.
    for k in range(N_OUTER):
        inside = set(train.cluster_id[folds == k])
        outside = set(train.cluster_id[folds != k])
        if inside & outside:
            raise SystemExit(f"outer fold {k} shares clusters with the rest")
    print(f"outer folds: {np.bincount(folds).tolist()} rows, fingerprint "
          f"{digest} ok ({source})")
    return folds


# --------------------------------------------------------------------------
# the sequence reference arm
# --------------------------------------------------------------------------


def ridge_fit_predict(X_fit, y_fit, X_apply, alphas):
    """Ridge over several alphas sharing one Gram matrix. Returns {alpha: pred}.

    `inf` returns the zero correction exactly, not a very large alpha's
    approximation to it.
    """
    mu = X_fit.mean(axis=0)
    ym = float(y_fit.mean())
    Xc = X_fit - mu
    G = Xc.T @ Xc
    b = Xc.T @ (y_fit - ym)
    eye = np.eye(X_fit.shape[1])
    Xa = X_apply - mu
    out = {}
    for a in alphas:
        if not np.isfinite(a):
            out[a] = np.zeros(len(X_apply))
            continue
        w = np.linalg.solve(G + a * eye, b)
        out[a] = Xa @ w + ym
    return out


def reference_sequence_oof(train: pd.DataFrame, outer: np.ndarray,
                           alpha: float = 10.0) -> pd.DataFrame:
    """A ridge sequence arm's nested OOF, in Phase A's long format.

    A *reference* so the harness can be validated end to end before the MLP
    ensemble's predictions exist -- the same role `stage6.ridge_oof_predictions`
    plays for the nested near-neighbour harness. It is not the comparator; the
    report labels any number produced from it as a plumbing check.
    """
    X_all = build_features(train, "pep_pseudo", "blosum").astype(np.float64)
    y_all = train.y_log1p.to_numpy()
    rows = []
    for f in range(N_OUTER):
        held = outer == f
        fit_idx = np.nonzero(~held)[0]
        fit_rows = train.iloc[fit_idx].reset_index(drop=True)
        inner = np.asarray(cv_folds(fit_rows, N_INNER))
        # outer_held: fitted on the whole fitting partition
        pred = ridge_fit_predict(X_all[fit_idx], y_all[fit_idx],
                                 X_all[held], [alpha])[alpha]
        rows.append(pd.DataFrame({"pair_id": train.pair_id.to_numpy()[held],
                                  "outer_fold": f, "role": "outer_held",
                                  "y_pred": pred}))
        # inner_oof: each inner fold predicted by a fit on the other four
        for k in range(N_INNER):
            ih = inner == k
            gi, gh = fit_idx[~ih], fit_idx[ih]
            p = ridge_fit_predict(X_all[gi], y_all[gi], X_all[gh], [alpha])[alpha]
            rows.append(pd.DataFrame({"pair_id": train.pair_id.to_numpy()[gh],
                                      "outer_fold": f, "role": "inner_oof",
                                      "y_pred": p}))
    return pd.concat(rows, ignore_index=True)


#: Which inner role the correction is fitted on, most preferred first.
#: ``inner_oof_30`` matches the outer-held ensemble's arity; ``inner_oof_6``
#: is the cheap 6-member variant, carried for the residual-scale diagnostic;
#: ``inner_oof`` is the single-role form the ridge reference emits.
INNER_ROLE_PREFERENCE = ("inner_oof_30", "inner_oof", "inner_oof_6")


def check_oof_contract(oof: pd.DataFrame, train: pd.DataFrame) -> dict:
    """Every training pair once as ``outer_held`` and four times as each
    inner role present. Arity is **asserted**, not taken on description.

    Phase A ships two inner roles: ``inner_oof_30``, whose members match the
    outer-held ensemble's arity, and ``inner_oof_6``, the cheap variant. The
    correction is fitted on the arity-matched one; the other is kept so the
    report can size both the mismatch that was fixed and the training-size
    mismatch that nested CV leaves behind irreducibly.
    """
    ids = set(train.pair_id)
    roles = sorted(oof.role.unique())
    if "outer_held" not in roles:
        raise SystemExit(f"no outer_held rows; roles present: {roles}")
    held = oof[oof.role == "outer_held"]
    if set(held.pair_id) != ids or held.pair_id.duplicated().any():
        raise SystemExit("outer_held rows do not cover every training pair exactly once")
    inner_roles = [r for r in roles if r != "outer_held"]
    if not inner_roles:
        raise SystemExit("no inner out-of-fold role in the OOF file")
    for r in inner_roles:
        counts = oof[oof.role == r].pair_id.value_counts()
        if set(counts.index) != ids or set(counts.unique()) != {N_OUTER - 1}:
            raise SystemExit(
                f"role {r!r} must appear {N_OUTER - 1}x per pair; got "
                f"{sorted(counts.unique())} over {len(counts)} pairs")
    if not np.isfinite(oof.y_pred).all():
        raise SystemExit("non-finite sequence prediction in the OOF file")
    primary = next((r for r in INNER_ROLE_PREFERENCE if r in inner_roles), None)
    if primary is None:
        raise SystemExit(
            f"none of {INNER_ROLE_PREFERENCE} present; inner roles are "
            f"{inner_roles}. Phase D will not guess which to fit on.")
    return {"n_outer_held": int(len(held)), "roles": roles,
            "inner_roles": inner_roles, "fitting_role": primary,
            "n_rows_per_inner_role": {r: int((oof.role == r).sum())
                                      for r in inner_roles}}


def mse_decomposition(allele: pd.Series, y: np.ndarray, pred: np.ndarray) -> dict:
    """Split squared error into within-allele and between-allele components.

    Median per-allele Spearman is invariant to any per-allele offset, so a
    correction cannot improve it by shifting alleles around. If a correction
    improves ranking *and* worsens squared error, the error has to be going
    somewhere, and the two places are different diagnoses:

    * **within-allele** -- the correction orders rows correctly but overshoots
      in magnitude. A monotone calibration map applied afterwards recovers the
      loss and keeps the ranking gain, so the objectives are not in conflict.
    * **between-allele** -- the correction improves ordering inside alleles
      while displacing alleles relative to each other, degrading a component
      the primary metric cannot see. Calibration cannot rescue that.

    The split is exact: with d = y - pred and allele means d_bar,
    sum(d^2) = sum_a sum_i (d_ai - d_bar_a)^2 + sum_a n_a * d_bar_a^2.
    """
    d = y - pred
    frame = pd.DataFrame({"allele": np.asarray(allele), "d": d})
    means = frame.groupby("allele")["d"].transform("mean").to_numpy()
    within_ss = float(np.sum((d - means) ** 2))
    between_ss = float(np.sum(means ** 2))
    n = len(d)
    return {"mse": float(np.mean(d ** 2)),
            "mse_within_allele": within_ss / n,
            "mse_between_allele": between_ss / n}


def linear_rescale(x_fit: np.ndarray, y_fit: np.ndarray, x_apply: np.ndarray
                   ) -> tuple[np.ndarray, float, float]:
    """Strictly monotone affine recalibration, fitted on the fitting partition.

    This is the probe matched to the "direction right, scale too large"
    mechanism: least squares gives `y ~ a + b*x`, and a slope `b < 1` says the
    correction overshoots in magnitude. Because `b > 0` makes the map strictly
    monotone, it cannot move the rank order of predictions within *any* subset
    of rows, so median per-allele Spearman is unchanged to floating-point
    tolerance. That is what makes it a clean one-sided probe: whatever MSE it
    recovers is pure calibration, borrowed from nothing.

    A degenerate fit (`b <= 0`) would not be monotone, so it is refused rather
    than silently applied.
    """
    b, a = np.polyfit(x_fit, y_fit, 1)
    if not (b > 0):
        raise SystemExit(f"recalibration slope {b:g} is not positive; an affine "
                         "map with a non-positive slope is not monotone")
    return a + b * x_apply, float(a), float(b)


def isotonic_map(x_fit: np.ndarray, y_fit: np.ndarray, x_apply: np.ndarray
                 ) -> np.ndarray:
    """Isotonic calibration, fitted on the fitting partition only.

    Reported alongside the affine probe, with one caveat that is easy to get
    wrong and which this harness caught by assertion: isotonic regression is
    **weakly** monotone, not strictly. It is a step function, so it collapses
    ranges of distinct predictions into equal values, and those new ties *do*
    move per-allele Spearman -- measured here at up to -0.016 median SCC on
    the reference run. So isotonic is not a rank-preserving probe and its MSE
    recovery is not free; the affine map is the one that isolates calibration.
    """
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(x_fit, y_fit)
    return iso.predict(x_apply)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def select_alpha(inner_metric: dict[float, list[float]], higher_is_better: bool
                 ) -> tuple[float, dict]:
    """1-SE rule toward stronger shrinkage (§6.2).

    The ladder is expected to be flat; an argmax on a flat ladder chases
    boundaries. Among alphas within one standard error of the best, take the
    largest -- the strongest shrinkage, nearest to no correction. Parsimony
    points toward not correcting, and that direction was declared before any
    arm was fitted.
    """
    alphas = sorted(inner_metric, key=lambda a: (np.isinf(a), a))
    means = np.array([np.mean(inner_metric[a]) for a in alphas])
    n = len(next(iter(inner_metric.values())))
    ses = np.array([np.std(inner_metric[a], ddof=1) / np.sqrt(n)
                    for a in alphas])
    best = int(np.argmax(means) if higher_is_better else np.argmin(means))
    thresh = (means[best] - ses[best]) if higher_is_better else (means[best] + ses[best])
    ok = (means >= thresh) if higher_is_better else (means <= thresh)
    chosen = max((a for a, k in zip(alphas, ok) if k),
                 key=lambda a: (np.isinf(a), a))
    return chosen, {
        "argbest_alpha": alphas[best], "argbest_mean": float(means[best]),
        "argbest_se": float(ses[best]), "selected_alpha": chosen,
        "selected_mean": float(means[alphas.index(chosen)]),
        "n_within_1se": int(ok.sum()), "n_ladder": len(alphas),
        "selected_is_no_correction": bool(np.isinf(chosen)),
        "selected_at_low_edge": bool(chosen == min(a for a in alphas if np.isfinite(a))),
        "ladder_profile": "|".join(f"{m:.5f}" for m in means),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    from pepstab.legacy_assessment import refuse_legacy_assessment
    refuse_legacy_assessment("phase_d_structural_residual.py")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq-oof", type=Path,
                    help="Phase A's long-format nested sequence OOF")
    ap.add_argument("--outer-folds", type=Path,
                    help="Phase A's pair_id,outer_fold file")
    ap.add_argument("--fitting-role", default=None,
                    help="override the declared fitting-role preference. Used "
                         "only for the declared sensitivity check; the primary "
                         "run takes INNER_ROLE_PREFERENCE.")
    ap.add_argument("--provenance", type=Path,
                    help="Phase A's A0 provenance JSON; member arity is "
                         "asserted against it rather than assumed")
    ap.add_argument("--runs", type=Path,
                    help="Phase A's per-member run table; row count asserted")
    ap.add_argument("--reference-arm", choices=["ridge"],
                    help="generate the sequence OOF from a ridge reference "
                         "instead (plumbing validation, not a comparator)")
    ap.add_argument("--tag", default=None, help="suffix for output filenames")
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    args = ap.parse_args()
    if (args.seq_oof is None) == (args.reference_arm is None):
        raise SystemExit("pass exactly one of --seq-oof or --reference-arm")
    tag = args.tag or (f"reference_{args.reference_arm}" if args.reference_arm
                       else "phase_a")
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    train = load_training_frame()
    geometry, confidence = structural_columns(train)
    arms = build_arms(geometry, confidence)
    print(f"training rows {len(train):,}; geometry {len(geometry)} cols, "
          f"confidence {len(confidence)} cols")
    for name, cols in arms.items():
        print(f"  {name:30s} {len(cols):4d} columns")

    outer = outer_assignment(train, args.outer_folds)

    if args.reference_arm:
        print("\ngenerating the ridge reference sequence OOF "
              "(plumbing validation, not the comparator)")
        oof = reference_sequence_oof(train, outer)
    else:
        oof = pd.read_csv(args.seq_oof)
    contract = check_oof_contract(oof, train)
    print(f"sequence OOF contract ok: {contract}")

    # Assert the ensemble arity we were handed rather than trusting the
    # description of it. The whole point of the 30-member inner set is that
    # it matches the outer-held arity; if it does not, the correction is
    # fitted on a different error distribution than it is applied to.
    prov = {}
    if args.provenance:
        prov = json.loads(args.provenance.read_text())
        mpf = prov.get("members_per_outer_fold", {})
        if prov.get("networks_total") != 900:
            raise SystemExit(f"provenance networks_total={prov.get('networks_total')}, "
                             "expected 900 (5 outer x 30, plus 5 outer x 5 inner x 30)")
        if mpf.get("level2_outer_held_and_inner_oof_6") != N_INNER * 2 * 3:
            raise SystemExit(f"level-2 arity {mpf} is not 5 inner x 2 encodings x 3 seeds")
        if mpf.get("level3_inner_oof_30") != N_INNER * N_INNER * 2 * 3:
            raise SystemExit(f"level-3 arity {mpf} is not 5 sub x 5 inner x 2 enc x 3 seeds")
        print(f"A0 arity asserted: {mpf}, fit rows {prov.get('fit_rows_by_level')}")
    if args.runs:
        rn = pd.read_csv(args.runs)
        if len(rn) != 900:
            raise SystemExit(f"{args.runs} has {len(rn)} member rows, expected 900")
        l2 = rn[rn.sub_fold.lt(0)]
        l3 = rn[rn.sub_fold.ge(0)]
        for label, sub, keys in (("level2", l2, ["outer_fold"]),
                                 ("level3", l3, ["outer_fold", "inner_fold"])):
            sizes = set(sub.groupby(keys).size().unique().tolist())
            if sizes != {30}:
                raise SystemExit(f"{label} members per {keys} = {sizes}, expected 30")
        print(f"A0 run table asserted: 900 members, 30 per ensemble at both levels; "
              f"dev MSE {l2.dev_mse.mean():.4f} (level 2) vs {l3.dev_mse.mean():.4f} "
              f"(level 3), fit rows {l2.n_fit_rows.mean():.0f} vs {l3.n_fit_rows.mean():.0f}")

    y_all = train.y_log1p.to_numpy()
    pair_index = pd.Series(np.arange(len(train)), index=train.pair_id.to_numpy())
    held_pred = np.full(len(train), np.nan)
    h = oof[oof.role == "outer_held"]
    held_pred[pair_index.loc[h.pair_id.to_numpy()].to_numpy()] = h.y_pred.to_numpy()
    assert np.isfinite(held_pred).all()

    # Residual-scale diagnostic: the correction is fitted on inner out-of-fold
    # errors and applied to outer-held errors. Where the two differ -- in
    # ensemble arity, or merely in training-set size because each nesting level
    # removes another fifth -- the fitting residuals are systematically larger
    # than the ones being corrected, which biases the correction's scale
    # upward. Nested CV cannot remove this entirely; measuring it is the
    # honest alternative to assuming it away.
    fitting_role = args.fitting_role or contract["fitting_role"]
    if fitting_role not in contract["inner_roles"]:
        raise SystemExit(f"fitting role {fitting_role!r} not in the OOF file "
                         f"(has {contract['inner_roles']})")
    if args.fitting_role:
        print(f"!! fitting role overridden to {fitting_role!r} -- this is the "
              "declared sensitivity check, not the primary run")
    outer_resid_sd = float(np.std(y_all - held_pred, ddof=1))
    resid_scale = {"outer_held": outer_resid_sd}
    for r in contract["inner_roles"]:
        sub = oof[oof.role == r]
        idx = pair_index.loc[sub.pair_id.to_numpy()].to_numpy()
        resid_scale[r] = float(np.std(y_all[idx] - sub.y_pred.to_numpy(), ddof=1))
    print("residual sd by role (ratio against outer_held):")
    for r, v in resid_scale.items():
        mark = "  <- fitted on" if r == fitting_role else ""
        print(f"  {r:16s} {v:.4f}  ratio {v / outer_resid_sd:.4f}{mark}")
    inner_all = oof[oof.role == fitting_role]
    inner_resid_sd = resid_scale[fitting_role]

    # Eligibility, from labels and counts alone, frozen before any fit.
    panel = eligible_alleles(train.allele, y_all, min_rows=MIN_ROWS)
    panel_strict = eligible_alleles(train.allele, y_all, min_rows=MIN_ROWS_STRICT)
    print(f"panel: {len(panel)}/{train.allele.nunique()} alleles at >={MIN_ROWS} "
          f"rows, coverage {train.allele.isin(panel).mean():.4f}; strict "
          f"{len(panel_strict)} at >={MIN_ROWS_STRICT}")

    rng = np.random.default_rng(SHUFFLE_SEED)
    preds: dict[str, np.ndarray] = {}
    calib: dict[str, np.ndarray] = {}
    recal: dict[str, np.ndarray] = {}
    sel_rows, fold_rows = [], []

    for arm, cols in arms.items():
        if not cols:
            preds[arm] = held_pred.copy()
            calib[arm] = held_pred.copy()
            recal[arm] = held_pred.copy()
            continue
        corrected = np.full(len(train), np.nan)
        calibrated = np.full(len(train), np.nan)
        rescaled = np.full(len(train), np.nan)
        for f in range(N_OUTER):
            held = outer == f
            fit_idx = np.nonzero(~held)[0]
            fit_rows = train.iloc[fit_idx].reset_index(drop=True)
            inner = np.asarray(cv_folds(fit_rows, N_INNER))

            # Sequence inner-OOF prediction for this outer fold's fitting rows.
            io = inner_all[inner_all.outer_fold == f]
            seq_inner = pd.Series(io.y_pred.to_numpy(),
                                  index=io.pair_id.to_numpy()).reindex(
                                      fit_rows.pair_id.to_numpy())
            if seq_inner.isna().any():
                raise SystemExit(f"outer fold {f}: inner_oof missing fitting rows")
            seq_inner = seq_inner.to_numpy()
            resid = fit_rows.y_log1p.to_numpy() - seq_inner

            Xraw = train[cols].to_numpy(dtype=np.float64)
            if arm.startswith("DX"):
                # Permute rows *inside the fitting partition only*, so the
                # control cannot accidentally carry information across folds.
                perm = rng.permutation(len(fit_idx))
                Xfit_raw = Xraw[fit_idx][perm]
            else:
                Xfit_raw = Xraw[fit_idx]
            Xheld_raw = Xraw[held]

            # Standardisation fitted on the fitting partition only.
            mu = np.nanmean(Xfit_raw, axis=0)
            sd = np.nanstd(Xfit_raw, axis=0)
            sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
            Xfit = (np.where(np.isfinite(Xfit_raw), Xfit_raw, mu) - mu) / sd
            Xheld = (np.where(np.isfinite(Xheld_raw), Xheld_raw, mu) - mu) / sd

            # Inner selection.
            scc: dict[float, list[float]] = {a: [] for a in ALPHA_LADDER}
            mse: dict[float, list[float]] = {a: [] for a in ALPHA_LADDER}
            for k in range(N_INNER):
                ih = inner == k
                p = ridge_fit_predict(Xfit[~ih], resid[~ih], Xfit[ih], ALPHA_LADDER)
                sub = fit_rows[ih]
                inner_panel = eligible_alleles(sub.allele,
                                               sub.y_log1p.to_numpy(),
                                               min_rows=MIN_ROWS)
                for a, r_hat in p.items():
                    yc = seq_inner[ih] + r_hat
                    s = score(f"{arm}:a{a}", sub.allele, sub.y_log1p.to_numpy(),
                              yc, alleles=inner_panel)
                    scc[a].append(s.median_spearman)
                    mse[a].append(float(np.mean((yc - sub.y_log1p.to_numpy()) ** 2)))
            a_scc, info_scc = select_alpha(scc, higher_is_better=True)
            a_mse, info_mse = select_alpha(mse, higher_is_better=False)

            # Refit on the whole fitting partition at the selected alpha and
            # apply to the outer-held rows.
            r_hat = ridge_fit_predict(Xfit, resid, Xheld, [a_scc])[a_scc]
            corrected[held] = held_pred[held] + r_hat

            # A monotone calibration map, fitted on out-of-sample corrected
            # predictions inside the fitting partition and applied to the
            # outer-held rows. Needed to tell "overshoots scale" from
            # "displaces alleles"; see mse_decomposition.
            fit_corrected = np.empty(len(fit_idx))
            for k in range(N_INNER):
                ih = inner == k
                rk = ridge_fit_predict(Xfit[~ih], resid[~ih], Xfit[ih],
                                       [a_scc])[a_scc]
                fit_corrected[ih] = seq_inner[ih] + rk
            y_fit_true = fit_rows.y_log1p.to_numpy()
            rescaled[held], a_cal, b_cal = linear_rescale(
                fit_corrected, y_fit_true, corrected[held])
            calibrated[held] = isotonic_map(
                fit_corrected, y_fit_true, corrected[held])

            sel_rows.append({
                "arm": arm, "outer_fold": f, "n_features": len(cols),
                "criterion": "inner_median_scc",
                **{f"scc_{k}": v for k, v in info_scc.items()},
                "mse_selected_alpha": a_mse,
                "mse_argbest_alpha": info_mse["argbest_alpha"],
                "selections_agree": bool(a_scc == a_mse),
                "mse_ladder_profile": info_mse["ladder_profile"],
                "correction_sd": float(np.std(r_hat, ddof=1)),
                "correction_abs_mean": float(np.mean(np.abs(r_hat))),
                "recal_intercept": a_cal, "recal_slope": b_cal,
            })
        assert np.isfinite(corrected).all()
        assert np.isfinite(calibrated).all()
        assert np.isfinite(rescaled).all()
        preds[arm] = corrected
        calib[arm] = calibrated
        recal[arm] = rescaled

    # ---- scoring ---------------------------------------------------------
    rows = []
    base = preds["D0_none"]
    for arm in arms:
        p = preds[arm]
        s = score(arm, train.allele, y_all, p, alleles=panel)
        s_strict = score(arm, train.allele, y_all, p, alleles=panel_strict)
        iqr_lo, iqr_hi = s.iqr_spearman
        row = {
            "arm": arm, "n_features": len(arms[arm]),
            "median_scc": s.median_spearman,
            "iqr_lo": iqr_lo, "iqr_hi": iqr_hi,
            "mean_scc": float(s.panel.mean()),
            "median_scc_strict_panel": s_strict.median_spearman,
            "mae_log1p": s.mae_log1p,
            "median_precision_at_10": s.median_precision_at_k,
            "pooled_spearman": s.pooled_spearman,
            "n_alleles": len(panel), "n_unranked": s.n_alleles_unranked,
            "correction_sd": float(np.std(p - base, ddof=1)),
            "readable_as_effect": arm in CANDIDATE_ARMS,
        }
        # MSE decomposition, and what a monotone calibration map recovers.
        dec = mse_decomposition(train.allele, y_all, p)
        dec_base = mse_decomposition(train.allele, y_all, base)
        row.update({f"{k}": v for k, v in dec.items()})
        row.update({f"d_{k}": dec[k] - dec_base[k] for k in dec})
        # Affine recalibration: strictly monotone, so ranks cannot move --
        # but the guarantee holds **within an outer fold**, which is the scope
        # the map is fitted at. Each fold gets its own (a, b), so the pooled
        # prediction vector is five different affine transforms stitched
        # together, and that is not a single monotone map on the pooled rows.
        # Within one allele, rows drawn from different outer folds can
        # therefore reorder. The assertion is made at the scope where the
        # guarantee actually holds; the pooled shift is measured and reported.
        rc = recal[arm]
        dec_r = mse_decomposition(train.allele, y_all, rc)
        row.update({f"recal_{k}": v for k, v in dec_r.items()})
        s_rec = score(arm + "+affine", train.allele, y_all, rc, alleles=panel)
        row["recal_median_scc"] = s_rec.median_spearman
        row["recal_scc_shift_pooled"] = s_rec.median_spearman - s.median_spearman
        worst = 0.0
        for f in range(N_OUTER):
            m = outer == f
            sub = train[m]
            fp = eligible_alleles(sub.allele, sub.y_log1p.to_numpy(), min_rows=MIN_ROWS)
            a0 = score("u", sub.allele, sub.y_log1p.to_numpy(), p[m], alleles=fp)
            a1 = score("c", sub.allele, sub.y_log1p.to_numpy(), rc[m], alleles=fp)
            worst = max(worst, abs(a1.median_spearman - a0.median_spearman))
        row["recal_scc_shift_within_fold_max"] = worst
        if arm != "D0_none" and worst > 1e-8:
            raise SystemExit(
                f"{arm}: a strictly monotone affine map moved within-fold median "
                f"SCC by {worst:.3e}. Ranks cannot move under a positive-slope "
                "affine map applied at fold scope; the harness is wrong.")
        # Isotonic: weakly monotone only. It creates ties, which DO move
        # Spearman, so the shift is reported rather than asserted away.
        c = calib[arm]
        dec_c = mse_decomposition(train.allele, y_all, c)
        row.update({f"iso_{k}": v for k, v in dec_c.items()})
        s_cal = score(arm + "+iso", train.allele, y_all, c, alleles=panel)
        row["iso_median_scc"] = s_cal.median_spearman
        row["iso_scc_shift"] = s_cal.median_spearman - s.median_spearman
        if arm != "D0_none":
            boot = paired_cluster_bootstrap(
                train.cluster_id, train.allele, y_all, base, p,
                alleles=panel, n_boot=args.n_boot, seed=BOOTSTRAP_SEED)
            row.update(delta_median_scc=boot["delta_median_spearman"],
                       ci_lo=boot["ci95"][0], ci_hi=boot["ci95"][1],
                       verdict=describe_delta(boot))
        rows.append(row)
        print(f"  {arm:30s} median {row['median_scc']:.4f}"
              + (f"  delta {row.get('delta_median_scc', 0):+.4f} "
                 f"[{row.get('ci_lo', 0):+.4f}, {row.get('ci_hi', 0):+.4f}]  "
                 f"{row.get('verdict', '')}" if arm != "D0_none" else "  (reference)"))

    # ---- per-fold stability ---------------------------------------------
    for arm in arms:
        for f in range(N_OUTER):
            m = outer == f
            sub = train[m]
            fp = eligible_alleles(sub.allele, sub.y_log1p.to_numpy(), min_rows=MIN_ROWS)
            s = score(arm, sub.allele, sub.y_log1p.to_numpy(), preds[arm][m], alleles=fp)
            fold_rows.append({"arm": arm, "outer_fold": f, "n_rows": int(m.sum()),
                              "n_alleles_eligible": len(fp),
                              "allele_coverage": float(sub.allele.isin(fp).mean()),
                              "median_scc": s.median_spearman,
                              "mae_log1p": s.mae_log1p})

    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / f"phase_d_residual_runs_{tag}.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(OUT / f"phase_d_residual_selection_{tag}.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / f"phase_d_residual_by_fold_{tag}.csv", index=False)

    per_allele = pd.DataFrame({
        arm: per_allele_spearman(train.allele, y_all, preds[arm], alleles=panel)
        for arm in arms})
    per_allele.index.name = "allele"
    per_allele.to_csv(OUT / f"phase_d_residual_per_allele_{tag}.csv")

    manifest = {
        "tag": tag, "sequence_arm": args.reference_arm or str(args.seq_oof),
        "n_train_rows": int(len(train)), "outer_fingerprint": OUTER_FOLD_FINGERPRINT,
        "alpha_ladder": [("inf" if not np.isfinite(a) else a) for a in ALPHA_LADDER],
        "min_rows": MIN_ROWS, "min_rows_strict": MIN_ROWS_STRICT,
        "n_alleles_panel": len(panel), "n_alleles_strict": len(panel_strict),
        "n_boot": args.n_boot, "bootstrap_seed": BOOTSTRAP_SEED,
        "shuffle_seed": SHUFFLE_SEED,
        "fitting_role": fitting_role,
        "residual_sd_by_role": resid_scale,
        "residual_sd_ratio_fitted_over_applied": inner_resid_sd / outer_resid_sd,
        "oof_contract": contract,
        "a0_provenance": {k: prov.get(k) for k in
                          ("networks_total", "members_per_outer_fold",
                           "fit_rows_by_level", "five_fold_median_of_medians")}
                         if prov else None,
        "arm_feature_counts": {k: len(v) for k, v in arms.items()},
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    (OUT / f"phase_d_residual_manifest_{tag}.json").write_text(
        json.dumps(manifest, indent=2) + "\n")

    dx = runs[runs.arm == "DX_geometry_shuffled"].iloc[0]
    print(f"\nDX shuffled control: delta {dx.delta_median_scc:+.4f} "
          f"[{dx.ci_lo:+.4f}, {dx.ci_hi:+.4f}] -- sign test only")
    if dx.ci_lo > 0:
        print("!! DX improves on held-out clusters. The harness leaks; no other "
              "arm's number may be reported.")
    print(f"\nwall {manifest['wall_seconds']}s; wrote 5 files to "
          f"{OUT.relative_to(REPO)}/ with tag {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
