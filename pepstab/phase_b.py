"""Phase B science layer: one censored-arm fitting unit, callable locally or on Modal.

The unit is deliberately the smallest thing that can be assessed on its own: one
**(arm, outer fold)** cell, which fits ``inner x encoding x seed`` members and
returns every member's prediction separately. Both
``scripts/phase_b_censored_nested.py`` and ``modal_app/phase_b_calibration.py``
call :func:`fit_arm_unit`, so the local pilot and the cloud batch cannot drift
apart -- the same failure mode the feature definitions live in a repo module to
prevent.

Three arms, declared in ``reports/further_research/phase_b_declaration.md`` §3
before any fit: the ``log1p``-MSE reference, the censored loss stopped on dev
NLL, and the censored loss stopped on dev MSE. A ranking-stopped fourth arm was
declined in advance, with reasons, in §3.1.

**Why per-member output rather than the ensemble mean.**
``pepstab.nested.fit_mlp_ensemble`` returns the mean, which is all a ranking
metric needs. A floor *probability* needs more: the declared construction is the
mixture ``mean_k Phi((c - mu_k)/sigma_k)``, which cannot be recovered from the
mean of ``mu`` at all. Collapsing to the mean first and then applying one
Gaussian is the plug-in rule, and it is measurably more confident in both tails
-- mixing the two across arms is what made stage 7a's reliability table disagree
with its own reported ECE.

**The MSE arm's scale.** It has no fitted ``sigma``, so one is taken from the
RMSE of that member's **inner out-of-fold** residuals on measured rows only.
Measured rows only, because squaring ``(0 - mu)`` at a floor code would measure
the floor coding rather than the model's spread. Inner out-of-fold rather than
``dev_mse``, because ``dev_mse`` is the best-epoch value early stopping selected
on and is therefore optimistic about out-of-sample spread -- and ``sigma`` is the
parameter a floor probability is most sensitive to. Both are returned so the gap
is visible rather than assumed negligible.
"""

from __future__ import annotations

import os

#: Mirrors the thread variables the Modal image sets. ``modal_app`` asserts the
#: two agree rather than trusting that they still do.
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def pin_threads(n: int = 1) -> None:
    """Pin BLAS to ``n`` threads. **Only effective before numpy is imported.**

    The thread pool is sized at import time, so calling this afterwards is a
    no-op. On Modal the image's ``.env`` does it instead, which is the only point
    early enough; locally a script must call this above its numpy import. This
    project has recorded that the same seed at a different thread count moves a
    single network by about 0.02 median Spearman, so a run record without the
    thread count is not reproducible.
    """
    for var in THREAD_ENV_VARS:
        os.environ.setdefault(var, str(n))


from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .censored import DEFAULT_FLOOR_HOURS, CensoredMLPRegressor, censored_mask  # noqa: E402
from .features import build_features  # noqa: E402
from .mlp import MLPRegressor  # noqa: E402
from .nested import REFERENCE_MLP  # noqa: E402

#: ``(arm id, model kind, stop_on, label)``. B0 is first because the
#: worthwhile-Brier bar is derived from it before any other arm is compared.
ARMS = (
    ("B0", "mse", None, "log1p-MSE reference"),
    ("B1", "censored", "nll", "censored, stopped on dev NLL"),
    ("B2", "censored", "mse", "censored, stopped on dev MSE"),
)

#: A0's exact per-encoding architecture and L2, inherited rather than re-tuned.
#: Tuning parity in this project means equal *budget*, not transplanted values --
#: but these two arms are the same geometry as the reference and differ only in
#: the objective, so holding them common is the correct choice, not a shortcut.
SELECTED = {"onehot": ((256, 64), 1e-5), "blosum": ((256, 64), 1e-3)}
ENCODINGS = ("onehot", "blosum")
SEEDS = (0, 1, 2)
INPUT_SET = "pep_pseudo"

#: Columns a unit's payload must carry. Records, not feature matrices: the
#: encoding happens where the fit happens, so a container is never handed a
#: matrix built under a different numpy.
UNIT_COLUMNS = ("pair_id", "allele", "peptide", "hla_pseudoseq",
                "thalf_hours", "y_log1p")


def as_frame(records) -> pd.DataFrame:
    """Rebuild a unit frame from plain records, checking the payload is complete."""
    frame = pd.DataFrame(list(records))
    missing = [c for c in UNIT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"unit payload is missing {missing}; expected {UNIT_COLUMNS}")
    return frame


def fit_arm_unit(fit_records, held_records, inner, *, kind: str,
                 stop_on: str | None, encodings=ENCODINGS, seeds=SEEDS,
                 selected=None, input_set: str = INPUT_SET,
                 floor_hours: float = DEFAULT_FLOOR_HOURS,
                 base_config=REFERENCE_MLP, verbose: bool = False) -> dict:
    """Fit one ``(arm, outer fold)`` unit and return every member separately.

    Mirrors ``pepstab.nested.fit_mlp_ensemble``'s loop exactly -- same
    encoding/inner-fold/seed order, same ``replace(base_config, ...)``, same
    row-local ``build_features`` -- so the budget is inherited rather than
    retyped. ``scripts/phase_b_censored_nested.py --parity-check`` asserts the
    MSE arm's ensemble mean matches that public function to 1e-9, which makes
    "inherited" a checked claim rather than a comment.

    Returns plain lists so the result crosses a Modal boundary unchanged.
    """
    selected = SELECTED if selected is None else selected
    fit_df = as_frame(fit_records)
    held_df = as_frame(held_records)
    inner = np.asarray(inner, dtype=int)
    if len(inner) != len(fit_df):
        raise ValueError(f"{len(inner)} inner labels for {len(fit_df)} fit rows")
    if kind not in ("mse", "censored"):
        raise ValueError(f"kind must be 'mse' or 'censored', got {kind!r}")

    n_inner = int(inner.max()) + 1
    per_inner = len(encodings) * len(seeds)
    y_fit = fit_df["y_log1p"].to_numpy(dtype=float)
    cens_fit = censored_mask(fit_df["thalf_hours"], floor_hours)

    mus, sigmas, sigmas_dev_mse, records = [], [], [], []
    inner_sum = np.zeros(len(fit_df))
    inner_count = np.zeros(len(fit_df), dtype=int)

    for encoding in encodings:
        hidden, l2 = selected[encoding]
        X_fit = build_features(fit_df, input_set, encoding)
        X_held = build_features(held_df, input_set, encoding)
        for k in range(n_inner):
            is_fit = inner != k
            for seed in seeds:
                config = replace(base_config, hidden=tuple(hidden), l2=l2, seed=seed)
                if kind == "mse":
                    model = MLPRegressor(config).fit(
                        X_fit[is_fit], y_fit[is_fit], X_fit[~is_fit], y_fit[~is_fit])
                else:
                    model = CensoredMLPRegressor(config, floor_hours=floor_hours,
                                                 stop_on=stop_on)
                    model.fit(X_fit[is_fit], y_fit[is_fit],
                              X_fit[~is_fit], y_fit[~is_fit],
                              cens_fit[is_fit], cens_fit[~is_fit])
                dev_pred = model.predict(X_fit[~is_fit])
                inner_sum[~is_fit] += dev_pred
                inner_count[~is_fit] += 1

                sigma_dev_mse = float(np.sqrt(model.dev_mse_))
                if kind == "mse":
                    measured = ~cens_fit[~is_fit]
                    if measured.sum() < 2:
                        raise ValueError("inner fold has fewer than two measured rows")
                    sigma = float(np.sqrt(np.mean(
                        (y_fit[~is_fit][measured] - dev_pred[measured]) ** 2)))
                else:
                    sigma = float(model.sigma_)

                mus.append(model.predict(X_held).astype(float).tolist())
                sigmas.append(sigma)
                sigmas_dev_mse.append(sigma_dev_mse)
                records.append({
                    "kind": kind, "stop_on": stop_on or "mse (only rule)",
                    "inner_fold": k, "encoding": encoding, "seed": seed,
                    "n_fit_rows": int(is_fit.sum()),
                    "n_censored_fit_rows": int(cens_fit[is_fit].sum()),
                    "best_epoch": int(model.best_epoch_),
                    "epochs": int(model.n_epochs_),
                    "sigma": sigma, "sigma_from_dev_mse": sigma_dev_mse,
                    "dev_mse": float(model.dev_mse_),
                    "fit_seconds": round(float(model.fit_seconds_), 2),
                    "threads": os.environ.get("OMP_NUM_THREADS", "unset"),
                })
                if verbose:
                    print(f"    {kind}/{stop_on} {encoding:7s} inner {k} seed {seed}"
                          f"  epoch {model.best_epoch_:3d}  sigma {sigma:.4f}",
                          flush=True)
        del X_fit, X_held

    if not np.all(inner_count == per_inner):
        raise AssertionError(
            f"each fit row should be held out by exactly {per_inner} members "
            f"({len(encodings)} encodings x {len(seeds)} seeds), got counts "
            f"{sorted(set(inner_count.tolist()))}")

    return {
        "mus": mus,
        "sigmas": sigmas,
        "sigmas_from_dev_mse": sigmas_dev_mse,
        "inner_oof": (inner_sum / per_inner).tolist(),
        "held_pair_id": held_df["pair_id"].astype(int).tolist(),
        "fit_pair_id": fit_df["pair_id"].astype(int).tolist(),
        "records": records,
        "n_members": len(mus),
    }


# --------------------------------------------------------------------------
# the two-part floor model


#: The two components. ``above`` predicts ``P(recorded label is above the floor)``
#: and ``magnitude`` predicts ``E[log1p(t) | above the floor]``, fitted on
#: above-floor rows only.
TWO_PART_COMPONENTS = ("above", "magnitude")


def fit_two_part_unit(fit_records, held_records, inner, *, encodings=ENCODINGS,
                      seeds=SEEDS, selected=None, input_set: str = INPUT_SET,
                      base_config=REFERENCE_MLP,
                      verbose: bool = False) -> dict:
    """Both halves of the two-part floor model on one outer fold.

    Part one regresses the **floor indicator** under squared error, which is the
    Brier-optimal least-squares probability estimate for a binary target; its
    output is clipped to ``[0, 1]`` only when read as a probability, never during
    fitting. Part two regresses ``y_log1p`` on **above-floor rows only**, so it
    estimates a conditional magnitude rather than a mixture.

    **What the product means, stated here because it is the thing most easily
    misread.** ``p_above * mu_conditional`` is an expectation of the *recorded*
    label, and it is correct as that: the recorded value at a floor row is
    literally 0, so the floor-coded component contributes zero to the expectation.
    It is **not** an estimate of true sub-floor lifetime -- a peptide below the
    assay's detection limit has a positive half-life that the assay could not
    measure, and nothing here estimates it. Nor is the product generally the
    optimal *ranking* score: multiplying a probability by a conditional magnitude
    can reorder pairs that the conditional magnitude alone orders correctly, so
    its ranking is an empirical question and is compared against the ordinary
    regressor rather than assumed.

    Each part keeps the same inner/encoding/seed budget as a single censored arm,
    so a unit here is twice the networks of a unit in :func:`fit_arm_unit`.
    """
    selected = SELECTED if selected is None else selected
    fit_df = as_frame(fit_records)
    held_df = as_frame(held_records)
    inner = np.asarray(inner, dtype=int)
    if len(inner) != len(fit_df):
        raise ValueError(f"{len(inner)} inner labels for {len(fit_df)} fit rows")

    n_inner = int(inner.max()) + 1
    per_inner = len(encodings) * len(seeds)
    y_fit = fit_df["y_log1p"].to_numpy(dtype=float)
    # The floor event is the recorded zero, matching
    # ``pepstab.calibration.floor_indicator(..., "recorded_zero")`` and the
    # declaration. ``above`` is its complement.
    above_fit = (fit_df["thalf_hours"].to_numpy(dtype=float) > 0.0).astype(float)

    out: dict = {"held_pair_id": held_df["pair_id"].astype(int).tolist(),
                 "fit_pair_id": fit_df["pair_id"].astype(int).tolist(),
                 "records": []}
    for component in TWO_PART_COMPONENTS:
        preds: list[list[float]] = []
        inner_sum = np.zeros(len(fit_df))
        inner_count = np.zeros(len(fit_df), dtype=int)
        for encoding in encodings:
            hidden, l2 = selected[encoding]
            X_fit = build_features(fit_df, input_set, encoding)
            X_held = build_features(held_df, input_set, encoding)
            for k in range(n_inner):
                is_fit = inner != k
                is_dev = ~is_fit
                if component == "above":
                    target, rows_fit, rows_dev = above_fit, is_fit, is_dev
                else:
                    # Above-floor rows only, on BOTH sides: a conditional
                    # magnitude early-stopped on a fold containing floor-coded
                    # zeros would be stopped against the wrong quantity.
                    keep = above_fit > 0.5
                    rows_fit, rows_dev = is_fit & keep, is_dev & keep
                    target = y_fit
                    if rows_dev.sum() < 2:
                        raise ValueError(
                            f"inner fold {k} has fewer than two above-floor dev rows")
                for seed in seeds:
                    config = replace(base_config, hidden=tuple(hidden), l2=l2,
                                     seed=seed)
                    model = MLPRegressor(config).fit(
                        X_fit[rows_fit], target[rows_fit],
                        X_fit[rows_dev], target[rows_dev])
                    preds.append(model.predict(X_held).astype(float).tolist())
                    dev_pred = model.predict(X_fit[rows_dev])
                    inner_sum[rows_dev] += dev_pred
                    inner_count[rows_dev] += 1
                    out["records"].append({
                        "component": component, "inner_fold": k,
                        "encoding": encoding, "seed": seed,
                        "n_fit_rows": int(rows_fit.sum()),
                        "n_dev_rows": int(rows_dev.sum()),
                        "best_epoch": int(model.best_epoch_),
                        "epochs": int(model.n_epochs_),
                        "dev_mse": float(model.dev_mse_),
                        "fit_seconds": round(float(model.fit_seconds_), 2),
                        "threads": os.environ.get("OMP_NUM_THREADS", "unset"),
                    })
                    if verbose:
                        print(f"    two-part/{component} {encoding:7s} inner {k} "
                              f"seed {seed}  epoch {model.best_epoch_:3d}", flush=True)
            del X_fit, X_held
        expected = per_inner if component == "above" else per_inner
        touched = inner_count > 0
        if not np.all(inner_count[touched] == expected):
            raise AssertionError(
                f"{component}: each held-out fit row should be predicted by "
                f"{expected} members, got {sorted(set(inner_count[touched].tolist()))}")
        out[f"{component}_preds"] = preds
        out[f"{component}_inner_oof"] = np.where(
            touched, inner_sum / np.maximum(inner_count, 1), np.nan).tolist()
    out["n_members"] = len(out["records"])
    return out
