"""Stage 3b: does auxiliary affinity training help the **ESM-2** arm?

    .venv/bin/python scripts/stage3b_esm_multitask.py --smoke        # harness check
    .venv/bin/python scripts/stage3b_esm_multitask.py --arms additive seq
    .venv/bin/python scripts/stage3b_esm_multitask.py --arms additive seq --mde

Stage 2c answered this for the **sequence** arm and the answer was a bounded
null: 20 paired comparisons, every 95% CI crossing zero and every upper bound
below the 0.05 bar. Its explanation was that affinity is *redundant* with what a
sequence model already extracts -- measured affinity used directly as a
stability predictor ranks at 0.580, below the 0.610 a single sequence network
reaches from stability labels alone.

**Redundancy with a sequence model does not establish redundancy with ESM-2
features.** That is the open question here, and the plan is explicit that if
multi-task training closes the gap between the sequence baseline and the ESM
arm, it is worth reporting: cheap extra labels substituting for expensive
pretrained features would be a real finding.

So the quantity of interest is not one delta but **two, compared**:

    Delta_esm = rho(ESM, lambda>0)      - rho(ESM, lambda=0)
    Delta_seq = rho(sequence, lambda>0) - rho(sequence, lambda=0)
    difference-in-differences = Delta_esm - Delta_seq

Both arms are bootstrapped on the *same* peptide clusters, so the
difference-in-differences gets its own paired interval rather than being eyeballed
from two separate ones.

## What is held fixed

Everything except the loss weight. ``pepstab.multitask.MultiTaskMLPRegressor``
is protocol-agnostic -- it takes a feature matrix -- so the same class, the same
fold assignment (``cv_folds`` from ``scripts/baseline_ensemble``), the same
seeds and the same stopping criterion (stability dev MSE) serve every arm. At
``lambda_aff = 0`` the network is bit-identical to ``pepstab.mlp.MLPRegressor``
at the same seed, asserted in ``tests/test_stage3b.py`` **on the real ESM
feature grid**, not only on a toy. So each arm's single-task comparator *is*
that arm's stage 3 model, not a near-replica of it.

## Tuning parity is budget parity, not value parity

This bit hard tonight. ESM features are dense and standardised; stage 2's inputs
are sparse one-hot columns. `esm-arm` measured that transplanting stage 2's L2
ladder onto the ESM arm cost **0.109** median rho and nearly manufactured a
false negative. Every arm here therefore inherits the L2 **its own** stage 3
tuning selected, and :func:`check_interior` refuses to run an arm whose selected
value sits at the edge of its ladder -- a boundary hit means the ladder was
truncated and the number would understate that arm.

## Leakage

The probe trains only on affinity labels attached to pairs that already carry a
measured half-life, so it adds no peptide and moves none across a split
boundary. ``--audit-expansion`` re-derives the exclusion table for the broader
IEDB set from the raw reference rather than trusting the stage 2c write-up.
Absence is tested on ``peptide`` alone, never on ``(allele, peptide)``: the
reference's own ``padding_eligible`` flag tests the pair and therefore leaks.

Validation only. The test split is never read.
"""

from __future__ import annotations

import os

# BLAS pinned before numpy is imported -- the pool is sized at import time and
# numpy here links Accelerate, so VECLIB_MAXIMUM_THREADS is the binding one.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pepstab import esm as pesm  # noqa: E402
from pepstab.affinity import (  # noqa: E402
    auxiliary_only,
    dual_labelled,
    filter_by_distance,
    held_out_peptides,
)
from pepstab.data import load_with_splits  # noqa: E402
from pepstab.evaluation import (  # noqa: E402
    MIN_WORTHWHILE_DELTA_SPEARMAN,
    N_BOOTSTRAP,
    BOOTSTRAP_SEED,
    UNRANKED_CONTRIBUTION,
    describe_delta,
    eligible_alleles,
    paired_cluster_bootstrap,
    per_allele_spearman,
    panel_spearman,
    rankable_alleles,
    score,
)
from pepstab.features import ENCODINGS, build_features  # noqa: E402
from pepstab.multitask import MultiTaskMLPConfig, MultiTaskMLPRegressor  # noqa: E402
from scripts.baseline_ensemble import N_FOLDS, cv_folds  # noqa: E402
from scripts.baseline_sequence import MAX_EPOCHS, PATIENCE, SEEDS, inner_folds  # noqa: E402
from scripts.esm_arm import assemble  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = REPO_ROOT / "reports"
PRED_DIR = REPO_ROOT / "preds"

#: Same sweep stage 2c used, so the two arms' curves are directly comparable.
LAMBDA_GRID = (0.0, 0.1, 0.3, 1.0, 3.0)

#: The stage 3 selection, read from `reports/stage3_tuning_sensitivity.csv`
#: rather than re-tuned here. Each entry is (hidden, l2, ladder) and the ladder
#: is carried so :func:`check_interior` can prove the value is not at an edge.
#:
#: Nothing in stage 3b re-tunes anything: the loss weight is the only thing
#: swept, exactly as in stage 2c.
ARM_CONFIGS: dict[str, dict[str, tuple]] = {
    # Additive: ESM block + the stage 2 raw encoding. The decision-relevant arm
    # -- it differs from the sequence baseline by exactly the ESM features.
    "additive": {
        "onehot": ((256, 64), 1e-2, (1e-3, 1e-2, 1e-1)),
        "blosum": ((256, 64), 1e-2, (1e-3, 1e-2, 1e-1)),
    },
    # ESM features alone, spread over the two layers (stage 3's "esm-only" arm).
    "esm": {
        "mid": ((256, 64), 1e-2, (1e-5, 1e-3, 1e-2, 1e-1, 1.0, 10.0)),
        "final": ((256, 64), 1e-2, (1e-5, 1e-3, 1e-2, 1e-1, 1.0, 10.0)),
    },
    # The sequence comparator, on its *extended* ladder. Stage 2c ran this on
    # stage 2's truncated ladder, so it is re-run here rather than reused.
    "seq": {
        "onehot": ((256, 64), 1e-2, (1e-7, 1e-6, 1e-5, 1e-2, 1e-1)),
        "blosum": ((256, 64), 1e-5, (1e-7, 1e-6, 1e-5, 1e-2, 1e-1)),
    },
}

#: The stage 3 representation the ESM arms use. Matches what `esm-arm` selected.
ESM_CHECKPOINT = "esm2_t12_35M_UR50D"
ESM_PEP_REP = "pos"
ESM_HLA_REP = "contact"
ESM_LAYER = "mid"


def check_interior(arm: str, configs: dict[str, tuple]) -> list[dict]:
    """Refuse an arm whose selected L2 sits at the edge of its own ladder.

    A boundary hit means the ladder was truncated, so the selected value is not
    the best available one and the arm is being handicapped. `esm-arm` measured
    that handicap at 0.109 median rho on the ESM arm; a silent boundary hit here
    would reproduce exactly that failure inside a comparison whose whole point
    is parity.
    """
    rows = []
    for key, (hidden, l2, ladder) in configs.items():
        interior = min(ladder) < l2 < max(ladder)
        rows.append({"arm": arm, "group": key, "hidden": "x".join(map(str, hidden)),
                     "l2": l2, "ladder": "|".join(f"{v:g}" for v in ladder),
                     "n_points": len(ladder), "interior": interior})
        if not interior:
            raise SystemExit(
                f"arm {arm!r} group {key!r}: selected l2={l2:g} is at the edge of "
                f"ladder {ladder}. The ladder was truncated -- extend it in stage 3 "
                "and re-select before running stage 3b, or this arm is handicapped.")
    return rows


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

@dataclass
class Group:
    """One ensemble group: a feature matrix pair plus the config to fit on it."""

    key: str
    X_train: np.ndarray
    X_val: np.ndarray
    hidden: tuple[int, ...]
    l2: float
    n_features: int = 0

    def __post_init__(self):
        self.n_features = int(self.X_train.shape[1])


def build_groups(arm: str, train: pd.DataFrame, val: pd.DataFrame,
                 checkpoint: str = ESM_CHECKPOINT) -> list[Group]:
    """The two ensemble groups for ``arm``, with features already assembled.

    Each arm spreads its 30 networks over a two-way axis, mirroring stage 2's
    {one-hot, BLOSUM62}: the additive and sequence arms over raw encodings, the
    ESM-only arm over the two ESM layers. Member count, folds and seeds match
    across arms; only the inputs differ.
    """
    cfgs = ARM_CONFIGS[arm]
    groups: list[Group] = []
    for key, (hidden, l2, _ladder) in cfgs.items():
        if arm == "seq":
            Xt = build_features(train, "pep_pseudo", key)
            Xv = build_features(val, "pep_pseudo", key)
        elif arm == "additive":
            Xt, (Xv,), _ = assemble(train, [val], checkpoint, ESM_LAYER,
                                    ESM_PEP_REP, ESM_HLA_REP, raw=key)
        elif arm == "esm":
            Xt, (Xv,), _ = assemble(train, [val], checkpoint, key,
                                    ESM_PEP_REP, ESM_HLA_REP, raw=None)
        else:
            raise ValueError(f"unknown arm {arm!r}")
        groups.append(Group(key, Xt, Xv, hidden, l2))
        gc.collect()
    return groups


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

def affinity_head_quality(model: MultiTaskMLPRegressor, X: np.ndarray,
                          y_aff: np.ndarray) -> float:
    """Spearman of the auxiliary head against held-out affinity labels.

    The diagnostic that separates "affinity does not help" from "the affinity
    head never trained". At lambda=0 the head receives only L2 decay, so this
    should sit near 0; if it does not, the controlled comparison is broken.

    A **constant** head scores 0, not NaN. At lambda=0 the head decays to a
    constant, which is the expected outcome -- and the project's frozen scoring
    rule (``pepstab.evaluation.UNRANKED_CONTRIBUTION``) already says a constant
    prediction carries no ranking information and is worth 0. Returning NaN
    instead would poison the mean over ensemble members and make the control
    unreadable, which is the opposite of what this diagnostic is for. NaN is
    reserved for "not enough labels to say", which is a different statement.
    """
    ok = np.isfinite(y_aff)
    if ok.sum() < 3:
        return float("nan")
    pred = model.predict_affinity(X)[ok]
    truth = y_aff[ok]
    if len(np.unique(truth)) < 2:
        return float("nan")
    if len(np.unique(pred)) < 2:
        return UNRANKED_CONTRIBUTION
    return float(stats.spearmanr(pred, truth).statistic)


def run_lambda(arm: str, groups: list[Group], y_train: np.ndarray,
               aff_train: np.ndarray, aff_val: np.ndarray, fold: np.ndarray,
               lam: float, seeds: tuple[int, ...], n_folds: int,
               ) -> tuple[np.ndarray, list[dict]]:
    """Fit this arm's whole ensemble at one ``lambda_aff``; return mean predictions."""
    members, records = [], []
    for g in groups:
        for k in range(n_folds):
            is_fit = fold != k
            for seed in seeds:
                cfg = MultiTaskMLPConfig(hidden=g.hidden, l2=g.l2, seed=seed,
                                         max_epochs=MAX_EPOCHS, patience=PATIENCE,
                                         lambda_aff=lam)
                m = MultiTaskMLPRegressor(cfg).fit(
                    g.X_train[is_fit], y_train[is_fit], aff_train[is_fit],
                    g.X_train[~is_fit], y_train[~is_fit])
                members.append(m.predict(g.X_val))
                records.append({
                    "arm": arm, "group": g.key, "lambda_aff": lam, "fold": k,
                    "seed": seed, "n_features": g.n_features, "l2": g.l2,
                    "hidden": "x".join(map(str, g.hidden)),
                    "n_fit_rows": int(is_fit.sum()),
                    "n_affinity_rows": m.n_affinity_rows_,
                    "best_epoch": m.best_epoch_, "epochs": m.n_epochs_,
                    "dev_mse": round(float(m.dev_mse_), 5),
                    "affinity_head_rho": round(
                        affinity_head_quality(m, g.X_val, aff_val), 4),
                    "fit_seconds": round(m.fit_seconds_, 2),
                })
    return np.mean(members, axis=0), records


# --------------------------------------------------------------------------
# minimum detectable effect
# --------------------------------------------------------------------------

def degrade(pred: np.ndarray, allele: np.ndarray, t: float,
            rng: np.random.Generator) -> np.ndarray:
    """Blend ``pred`` toward a within-allele shuffle of itself.

    ``t = 0`` returns the predictions unchanged, ``t = 1`` returns a pure
    within-allele permutation (median rho ~ 0). Shuffling *within* allele keeps
    the marginal distribution and the between-allele offsets intact, so the only
    thing the blend destroys is within-allele ranking -- which is the quantity
    the primary metric measures.
    """
    out = pred.copy()
    for a in np.unique(allele):
        idx = np.flatnonzero(allele == a)
        out[idx] = (1 - t) * pred[idx] + t * pred[rng.permutation(idx)]
    return out


def measure_mde(val: pd.DataFrame, pred: np.ndarray, alleles: list[str],
                n_boot: int, seed: int = BOOTSTRAP_SEED,
                levels=(0.05, 0.10, 0.15, 0.20, 0.30, 0.45)) -> pd.DataFrame:
    """Measured detectability curve on *these* rows, alleles and clusters.

    For each blend level, report the realised change in median per-allele
    Spearman and the paired cluster bootstrap CI against the undegraded arm. The
    MDE is the smallest realised |delta| whose CI excludes 0.

    This is measured, not assumed: it uses the same `paired_cluster_bootstrap`
    the real comparisons use, on the same 2,817 validation rows and the same
    eligible-allele panel, so the number transfers directly to the
    lambda-vs-lambda comparisons below.
    """
    y = val.y_log1p.to_numpy()
    allele = val.allele.to_numpy()
    rng = np.random.default_rng(seed)
    base = score("base", val.allele, y, pred, alleles).median_spearman
    rows = []
    for t in levels:
        worse = degrade(pred, allele, t, rng)
        s = score(f"t={t}", val.allele, y, worse, alleles).median_spearman
        res = paired_cluster_bootstrap(val.cluster_id, val.allele, y, worse, pred,
                                       alleles, n_boot=n_boot)
        lo, hi = res["ci95"]
        rows.append({
            "blend_t": t,
            "median_rho_degraded": round(s, 4),
            "realised_delta": round(base - s, 4),
            "bootstrap_delta": round(res["delta_median_spearman"], 4),
            "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "n_boot": n_boot,
        })
        print(f"    t={t:<5} realised delta {base - s:+.4f}  "
              f"CI [{lo:+.4f}, {hi:+.4f}]  "
              f"{'detected' if (lo > 0 or hi < 0) else 'NOT detected'}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# difference-in-differences
# --------------------------------------------------------------------------

def _panel_median(cluster_rows, idx, allele, y, pred, alleles, rankable):
    rho = per_allele_spearman(allele[idx], y[idx], pred[idx], alleles)
    panel = panel_spearman(rho, rankable)
    return float(panel.median()) if len(panel) else float("nan")


def difference_in_differences(val: pd.DataFrame, alleles: list[str],
                              esm_single: np.ndarray, esm_multi: np.ndarray,
                              seq_single: np.ndarray, seq_multi: np.ndarray,
                              n_boot: int = N_BOOTSTRAP,
                              seed: int = BOOTSTRAP_SEED) -> dict:
    """Paired CI for ``(esm_multi - esm_single) - (seq_multi - seq_single)``.

    **This is the stage 3b question.** "Does affinity help the ESM arm
    *differently* from the sequence arm" is a statement about the gap between
    two deltas, and reading it off two separately-bootstrapped intervals would
    both lose the pairing and overstate the uncertainty. All four arms are
    scored on the same resampled clusters, so the four-way correlation is
    carried through.
    """
    y = val.y_log1p.to_numpy()
    allele = val.allele.to_numpy()
    cluster = val.cluster_id.to_numpy()
    uniq = np.unique(cluster)
    rows_by_cluster = {c: np.flatnonzero(cluster == c) for c in uniq}
    rng = np.random.default_rng(seed)

    def did(idx, rankable):
        e = (_panel_median(None, idx, allele, y, esm_multi, alleles, rankable)
             - _panel_median(None, idx, allele, y, esm_single, alleles, rankable))
        s = (_panel_median(None, idx, allele, y, seq_multi, alleles, rankable)
             - _panel_median(None, idx, allele, y, seq_single, alleles, rankable))
        return e, s

    full = np.arange(len(y))
    full_rankable = rankable_alleles(allele, y, alleles)
    obs_e, obs_s = did(full, full_rankable)

    draws = np.empty((n_boot, 2))
    for i in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_by_cluster[c] for c in picked])
        rankable = rankable_alleles(allele[idx], y[idx], alleles)
        draws[i] = did(idx, rankable)

    diff = draws[:, 0] - draws[:, 1]
    lo, hi = np.nanpercentile(diff, [2.5, 97.5])
    return {
        "delta_esm": obs_e, "delta_seq": obs_s,
        "difference_in_differences": obs_e - obs_s,
        "ci95": (float(lo), float(hi)), "n_boot": n_boot,
        "conclusive": bool(lo > 0 or hi < 0),
    }


# --------------------------------------------------------------------------
# leakage audit
# --------------------------------------------------------------------------

def audit_expansion(df: pd.DataFrame, train: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Re-derive the IEDB exclusion table from the raw reference.

    Not read from the stage 2c report: recomputed, so the numbers quoted in
    `reports/stage3b_esm_multitask.md` are this run's own.
    """
    fold = inner_folds(train)
    dev_pep = train.loc[(fold == "dev").to_numpy(), "peptide"]
    held = held_out_peptides(df, dev_peptides=dev_pep)
    aux = auxiliary_only(df)
    kept, table = filter_by_distance(aux, held)
    summary = {
        "n_held_out_peptides": len(held),
        "n_candidate_rows": len(aux),
        "n_candidate_peptides": int(aux["peptide"].nunique()),
        "n_kept_rows_hamming_gt_3": len(kept),
        "n_kept_peptides_hamming_gt_3": int(kept["peptide"].nunique()),
        "absence_tested_on": "peptide (never the (allele, peptide) pair)",
    }
    return table, summary


# --------------------------------------------------------------------------

def probe_l2_interiority(arm: str, train: pd.DataFrame, val: pd.DataFrame,
                         alleles: list[str], lambdas=(0.0, 3.0),
                         checkpoint: str = ESM_CHECKPOINT) -> pd.DataFrame:
    """Re-select L2 **with the auxiliary head present**, and check it is interior.

    ``check_interior()`` runs before any fit, on the L2 that stage 3 selected for
    a *single-headed* network. Adding a second head changes the effective
    regularisation, so a value that was interior single-task is not guaranteed to
    stay interior multi-task -- and `esm-arm` measured the ESM arm as roughly
    50x more sensitive to the regularisation range than the baseline (0.109
    against +0.002). The additive ladder has only three points with the middle
    one selected, so there is no headroom to absorb a shift.

    Protocol: the stage 2/3 *selection* protocol (one permanent inner fit/dev
    cut, 3 seeds), not the 30-network ensemble -- selection is what is being
    re-checked, and this must stay cheap enough to run as a gate.

    Returns one row per (lambda, group, l2). A selected value at an edge means
    the ladder was truncated once the head was added, and the arm's deltas
    should not be trusted until it is extended.
    """
    fold = inner_folds(train)
    is_fit = (fold == "fit").to_numpy()
    y = train.y_log1p.to_numpy()
    aff = dual_labelled(train).to_numpy()
    y_val = val.y_log1p.to_numpy()

    rows = []
    for key, (hidden, _sel, ladder) in ARM_CONFIGS[arm].items():
        if arm == "seq":
            Xt, Xv = (build_features(train, "pep_pseudo", key),
                      build_features(val, "pep_pseudo", key))
        elif arm == "additive":
            Xt, (Xv,), _ = assemble(train, [val], checkpoint, ESM_LAYER,
                                    ESM_PEP_REP, ESM_HLA_REP, raw=key)
        else:
            Xt, (Xv,), _ = assemble(train, [val], checkpoint, key,
                                    ESM_PEP_REP, ESM_HLA_REP, raw=None)
        for lam in lambdas:
            for l2 in ladder:
                rhos = []
                for seed in SEEDS:
                    cfg = MultiTaskMLPConfig(hidden=hidden, l2=l2, seed=seed,
                                             max_epochs=MAX_EPOCHS,
                                             patience=PATIENCE, lambda_aff=lam)
                    m = MultiTaskMLPRegressor(cfg).fit(
                        Xt[is_fit], y[is_fit], aff[is_fit],
                        Xt[~is_fit], y[~is_fit])
                    rhos.append(score(f"p_{key}_{l2}_{seed}", val.allele, y_val,
                                      m.predict(Xv), alleles).median_spearman)
                rows.append({"arm": arm, "group": key, "lambda_aff": lam,
                             "l2": l2, "ladder_min": min(ladder),
                             "ladder_max": max(ladder),
                             "rho_mean": round(float(np.mean(rhos)), 4),
                             "rho_spread": round(float(max(rhos) - min(rhos)), 4)})
        del Xt, Xv
        gc.collect()

    frame = pd.DataFrame(rows)
    verdicts = []
    for (grp, lam), g in frame.groupby(["group", "lambda_aff"]):
        best = g.loc[g.rho_mean.idxmax()]
        interior = bool(best.ladder_min < best.l2 < best.ladder_max)
        verdicts.append({"arm": arm, "group": grp, "lambda_aff": lam,
                         "selected_l2": best.l2, "rho_mean": best.rho_mean,
                         "interior_with_head": interior})
        flag = "interior" if interior else "AT BOUNDARY -- extend the ladder"
        print(f"    {grp:8s} lambda={lam:<4g} -> l2={best.l2:g} "
              f"(rho {best.rho_mean:+.4f})  {flag}")
    return frame.merge(pd.DataFrame(verdicts),
                       on=["arm", "group", "lambda_aff"], how="left")


def load_arm_predictions(arm: str, lambdas, val: pd.DataFrame
                         ) -> dict[float, np.ndarray]:
    """Re-load an arm's per-lambda ensemble predictions from ``preds/``.

    The coordinator's sequencing runs the sequence arm before the ESM arms are
    cleared, and the difference-in-differences needs both in one process.
    Re-fitting the sequence arm later would cost 21 minutes *and* risk the DiD
    being computed against a different fit than the one already reported, so it
    is reloaded instead.

    Coverage is validated against ``val`` by ``pair_id`` -- never positionally.
    A missing or partial file raises rather than quietly shrinking the panel.
    """
    out: dict[float, np.ndarray] = {}
    want = val["pair_id"].to_numpy()
    for lam in lambdas:
        path = PRED_DIR / f"stage3b_{arm}_lam{lam:g}.csv"
        if not path.exists():
            raise SystemExit(
                f"--reuse-arms {arm}: {path} is missing. Run that arm first "
                f"(--arms {arm}), or drop it from --reuse-arms.")
        frame = pd.read_csv(path)
        merged = pd.DataFrame({"pair_id": want}).merge(
            frame, on="pair_id", how="left", validate="one_to_one")
        if merged["y_pred"].isna().any():
            raise SystemExit(f"{path} does not cover every validation row")
        out[lam] = merged["y_pred"].to_numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=["additive", "seq"],
                    choices=sorted(ARM_CONFIGS))
    ap.add_argument("--lambdas", nargs="+", type=float, default=None)
    ap.add_argument("--checkpoint", default=ESM_CHECKPOINT,
                    choices=sorted(pesm.CHECKPOINTS))
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--mde", action="store_true",
                    help="measure the detectability curve (adds bootstraps)")
    ap.add_argument("--mde-boot", type=int, default=500,
                    help="resamples per MDE level (reduced; the curve needs "
                         "many intervals, the headline comparisons use --n-boot)")
    ap.add_argument("--audit-expansion", action="store_true")
    ap.add_argument("--probe-l2", action="store_true",
                    help="re-select L2 with the auxiliary head present and "
                         "check it is still interior, before fitting anything")
    ap.add_argument("--reuse-arms", nargs="*", default=[],
                    help="arms whose per-lambda predictions are loaded from "
                         "preds/ instead of refitted. Used to carry the "
                         "sequence arm into the ESM run without refitting it, "
                         "so the difference-in-differences uses exactly the "
                         "predictions already reported.")
    ap.add_argument("--smoke", action="store_true",
                    help="1 fold, 1 seed, 2 lambdas -- harness check only")
    args = ap.parse_args()

    lambdas = tuple(args.lambdas) if args.lambdas else LAMBDA_GRID
    seeds = (SEEDS[0],) if args.smoke else SEEDS
    n_folds = 1 if args.smoke else N_FOLDS
    if args.smoke:
        lambdas = (0.0, 1.0)
    suffix = "_smoke" if args.smoke else ""

    # Parity gate, before anything is fitted.
    ladder_rows = []
    for arm in args.arms:
        ladder_rows += check_interior(arm, ARM_CONFIGS[arm])
    print("L2 selection, checked interior to each arm's own ladder:")
    print(pd.DataFrame(ladder_rows).to_string(index=False))

    df = load_with_splits()
    train = df[df.split == "train"]
    val = df[df.split == "val"].sort_values("pair_id").reset_index(drop=True)
    alleles = eligible_alleles(val.allele, val.y_log1p, split="val")
    fold = cv_folds(train, N_FOLDS)
    y_train = train.y_log1p.to_numpy()
    y_val = val.y_log1p.to_numpy()

    aff_train = dual_labelled(train).to_numpy()
    aff_val = dual_labelled(val).to_numpy()
    n_aff = int(np.isfinite(aff_train).sum())
    print(f"\ncohort: train={len(train):,} val={len(val):,}  "
          f"{len(alleles)} eligible alleles")
    print(f"auxiliary affinity labels on train rows: {n_aff:,} "
          f"({n_aff / len(train):.1%}), "
          f"{train.loc[np.isfinite(aff_train), 'allele'].nunique()} alleles")
    print(f"lambda grid: {lambdas}")
    print(f"ensemble: {n_folds} folds x 2 groups x {len(seeds)} seeds = "
          f"{n_folds * 2 * len(seeds)} networks per (arm, lambda)\n")

    if args.audit_expansion:
        table, summary = audit_expansion(df, train)
        print("expansion-surface leakage audit (recomputed, not quoted):")
        print(table.to_string(index=False))
        print(json.dumps(summary, indent=2))
        REPORT_DIR.mkdir(exist_ok=True)
        table.to_csv(REPORT_DIR / "stage3b_expansion_audit.csv", index=False)

    if args.probe_l2:
        print("\nre-selecting L2 with the auxiliary head present "
              "(adding a head changes the effective regularisation):")
        probes = []
        for arm in args.arms:
            probes.append(probe_l2_interiority(arm, train, val, alleles,
                                               checkpoint=args.checkpoint))
        probe = pd.concat(probes, ignore_index=True)
        probe.to_csv(REPORT_DIR / f"stage3b_l2_probe{suffix}.csv", index=False)
        bad = probe[~probe.interior_with_head.astype(bool)]
        if len(bad):
            raise SystemExit(
                "L2 is no longer interior once the auxiliary head is added for "
                f"{sorted(set(zip(bad.arm, bad.group)))}. Extend the ladder "
                "before trusting this arm's deltas.")
        print("  all groups interior with the head present\n")

    all_runs, all_rows, preds = [], [], {}

    # Reused arms first: they cost nothing and their metrics are recomputed
    # exactly from the saved predictions, so the sweep table is complete.
    prior = {}
    sweep_path = REPORT_DIR / "stage3b_lambda_sweep.csv"
    if args.reuse_arms and sweep_path.exists():
        old_sweep = pd.read_csv(sweep_path)
        prior = {(r["arm"], float(r["lambda_aff"])): r.get("affinity_head_rho")
                 for _, r in old_sweep.iterrows()}
    for arm in args.reuse_arms:
        loaded = load_arm_predictions(arm, lambdas, val)
        print(f"\n=== arm {arm} (reused from preds/, not refitted) ===")
        for lam, p in loaded.items():
            preds[(arm, lam)] = p
            s = score(f"{arm}_lam{lam:g}", val.allele, y_val, p, alleles)
            all_rows.append({
                "arm": arm, "lambda_aff": lam, "source": "reused",
                "median_per_allele_spearman": round(s.median_spearman, 4),
                "iqr_low": round(s.iqr_spearman[0], 4),
                "iqr_high": round(s.iqr_spearman[1], 4),
                "mae_log1p": round(s.mae_log1p, 4),
                "median_precision_at_10": round(s.median_precision_at_k, 4),
                "affinity_head_rho": prior.get((arm, lam), float("nan")),
                "n_members": np.nan, "fit_seconds": 0.0,
            })
            print(f"  lambda={lam:<4g} rho={s.median_spearman:+.4f}  "
                  f"MAE={s.mae_log1p:.4f}  (reused)")

    started = time.perf_counter()
    for arm in args.arms:
        if arm in args.reuse_arms:
            continue
        print(f"\n=== arm {arm} ===", flush=True)
        groups = build_groups(arm, train, val, args.checkpoint)
        for g in groups:
            print(f"  group {g.key}: {g.n_features:,} features, "
                  f"h{g.hidden} l2={g.l2:g}")
        for lam in lambdas:
            t0 = time.perf_counter()
            p, recs = run_lambda(arm, groups, y_train, aff_train, aff_val, fold,
                                 lam, seeds, n_folds)
            all_runs.extend(recs)
            preds[(arm, lam)] = p
            s = score(f"{arm}_lam{lam:g}", val.allele, y_val, p, alleles)
            head = float(np.mean([r["affinity_head_rho"] for r in recs]))
            all_rows.append({
                "arm": arm, "lambda_aff": lam, "source": "fitted",
                "median_per_allele_spearman": round(s.median_spearman, 4),
                "iqr_low": round(s.iqr_spearman[0], 4),
                "iqr_high": round(s.iqr_spearman[1], 4),
                "mae_log1p": round(s.mae_log1p, 4),
                "median_precision_at_10": round(s.median_precision_at_k, 4),
                "affinity_head_rho": round(head, 4),
                "n_members": len(recs),
                "fit_seconds": round(time.perf_counter() - t0, 1),
            })
            print(f"  lambda={lam:<4g} rho={s.median_spearman:+.4f}  "
                  f"MAE={s.mae_log1p:.4f}  affinity-head rho={head:+.3f}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
        del groups
        gc.collect()
    wall = time.perf_counter() - started

    # --- within-arm paired comparisons vs lambda = 0 ----------------------
    deltas = []
    for arm in list(dict.fromkeys(list(args.reuse_arms) + list(args.arms))):
        if (arm, 0.0) not in preds:
            continue
        base = preds[(arm, 0.0)]
        for lam in lambdas:
            if lam == 0.0:
                continue
            res = paired_cluster_bootstrap(val.cluster_id, val.allele, y_val,
                                           base, preds[(arm, lam)], alleles,
                                           n_boot=args.n_boot)
            lo, hi = res["ci95"]
            deltas.append({
                "arm": arm, "lambda_aff": lam,
                "delta_median_spearman": round(res["delta_median_spearman"], 4),
                "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                "verdict": describe_delta(res),
                "rules_out_min_worthwhile": res["rules_out_min_worthwhile"],
                "n_boot": res["n_boot"],
            })
            print(f"{arm} lambda={lam:<4g} vs lambda=0: "
                  f"{res['delta_median_spearman']:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
                  f"{describe_delta(res)}")

    # --- difference-in-differences ----------------------------------------
    did_rows = []
    have = set(args.arms) | set(args.reuse_arms)
    if "additive" in have and "seq" in have:
        for lam in lambdas:
            if lam == 0.0:
                continue
            r = difference_in_differences(
                val, alleles, preds[("additive", 0.0)], preds[("additive", lam)],
                preds[("seq", 0.0)], preds[("seq", lam)], n_boot=args.n_boot)
            lo, hi = r["ci95"]
            did_rows.append({
                "lambda_aff": lam,
                "delta_esm_additive": round(r["delta_esm"], 4),
                "delta_seq": round(r["delta_seq"], 4),
                "difference_in_differences": round(r["difference_in_differences"], 4),
                "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                "conclusive": r["conclusive"], "n_boot": r["n_boot"],
            })
            print(f"DiD lambda={lam:<4g}: esm {r['delta_esm']:+.4f} - "
                  f"seq {r['delta_seq']:+.4f} = "
                  f"{r['difference_in_differences']:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    # --- measured MDE ------------------------------------------------------
    mde = pd.DataFrame()
    if args.mde:
        ref_arm = "additive" if ("additive", 0.0) in preds else args.arms[0]
        print(f"\nmeasured detectability curve on the {ref_arm} arm "
              f"({args.mde_boot} resamples per level):")
        mde = measure_mde(val, preds[(ref_arm, 0.0)], alleles, args.mde_boot)
        hit = mde[mde.excludes_zero]
        floor = float(hit.realised_delta.min()) if len(hit) else float("nan")
        print(f"  -> smallest detected |delta| = {floor:.4f} "
              f"(predeclared worthwhile gain {MIN_WORTHWHILE_DELTA_SPEARMAN})")

    # --- write -------------------------------------------------------------
    REPORT_DIR.mkdir(exist_ok=True)
    PRED_DIR.mkdir(exist_ok=True)
    pd.DataFrame(all_runs).to_csv(
        REPORT_DIR / f"stage3b_runs{suffix}.csv", index=False)
    pd.DataFrame(all_rows).to_csv(
        REPORT_DIR / f"stage3b_lambda_sweep{suffix}.csv", index=False)
    pd.DataFrame(deltas).to_csv(
        REPORT_DIR / f"stage3b_deltas{suffix}.csv", index=False)
    pd.DataFrame(ladder_rows).to_csv(
        REPORT_DIR / f"stage3b_tuning_parity{suffix}.csv", index=False)
    if did_rows:
        pd.DataFrame(did_rows).to_csv(
            REPORT_DIR / f"stage3b_difference_in_differences{suffix}.csv", index=False)
    if len(mde):
        mde.to_csv(REPORT_DIR / f"stage3b_mde{suffix}.csv", index=False)
    for (arm, lam), p in preds.items():
        if arm in args.reuse_arms:
            continue        # already on disk; rewriting risks a silent change
        pd.DataFrame({"pair_id": val.pair_id.to_numpy(), "y_pred": p}).to_csv(
            PRED_DIR / f"stage3b_{arm}_lam{lam:g}{suffix}.csv", index=False)

    print(f"\nwrote reports/stage3b_*{suffix}.csv and {len(preds)} prediction files")
    print(f"total fit wall time: {wall / 60:.1f} min over {len(all_runs)} networks")
    if args.smoke:
        print("SMOKE RUN -- *_smoke filenames; canonical reports untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
