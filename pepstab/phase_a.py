"""Phase A learner arms: boosted trees, allele-centred targets, ranking objectives.

Implements arms A1-A4 of
``reports/further_research/phase_a_prespecification.md`` plus the shared
selection machinery. The prespecification is binding; every grid, every
held-fixed value and the parsimony rule come from it, not from this file.

**Training split only.** :func:`load_payload` filters to ``split == "train"``
and carries nothing else. Phase A scores no validation or test row.

**Three functions are the execution interface** the Modal app drives:
:func:`load_payload` (client side, reads disk), :func:`enumerate_units` (client
side, pure) and :func:`run_unit` (container side, pure -- no file I/O, no
network, everything arrives in ``payload``). Features are rebuilt from
sequences inside the container rather than shipped, because the 860-column
float32 block is 68 MB per encoding against ~1.4 MB of sequence records.

**A work unit is one fit.** ``(arm, config, encoding, seed, outer fold, inner
fold)``. The unit fits on the outer fitting partition minus one inner fold,
early-stops on that inner fold, and returns both roles of the A0 output
contract: ``oof`` (the inner-held rows, which is the selection and stacking
surface) and ``held`` (the outer-held rows, which the caller averages over
inner folds x encodings x seeds to assemble the 30-member ensemble the
prespecification requires for parity). Selection reads the mean *per inner
fold* median Spearman, so the unit is also the scoring granularity.

``pepstab/mlp.py`` is not touched: 540 tests depend on its gradients. The
ranking term arrives through :class:`RankingMLPRegressor`, a subclass whose
only departure is the gradient fed to the inherited ``_backward``, and whose
zero-weight path is bit-identical to the parent (checked in
``tests/test_phase_a_arms.py``).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import resource
import time

import numpy as np
import pandas as pd

from .data import TARGET
from .features import ENCODINGS, build_features
from .mlp import DTYPE, MLPConfig, MLPRegressor

#: Bumped whenever a unit dict or a returned row changes shape, so a Modal run
#: manifest cannot be read against the wrong schema.
UNITS_SCHEMA_VERSION: str = "phase_a.1"

#: The one input set Phase A uses: peptide + HLA pseudosequence, 43 positions x
#: 20 residues = 860 float32 columns. Same geometry as the deployed reference.
INPUT_SET = "pep_pseudo"

#: Columns :func:`load_payload` ships. Nothing else leaves the client.
PAYLOAD_COLUMNS = ("pair_id", "allele", "peptide", "hla_pseudoseq",
                   "y_log1p", "cluster_id", "outer_fold")

ARMS = ("A1", "A2", "A3", "A4")

#: Seeds per stage. Two paired seeds screen (plan §3.3); assessment uses the
#: reference's three, because ensemble parity has to hold where the comparison
#: is made.
SCREEN_SEEDS = (0, 1)
ASSESS_SEEDS = (0, 1, 2)

# --- reference MLP recipe ---------------------------------------------------

#: The deployed reference member, held common across A2 and A3 so a
#: same-geometry ablation isolates one difference. Mirrors
#: ``scripts.baseline_sequence.MAX_EPOCHS/PATIENCE`` and
#: ``scripts.baseline_ensemble.SELECTED[("pep_pseudo", enc)]``; a test asserts
#: the two stay equal rather than trusting this copy.
REFERENCE_MLP = {"hidden": (256, 64), "lr": 1e-3, "batch_size": 256,
                 "max_epochs": 300, "patience": 15}

#: Per-encoding L2, selected at stage 2 and never re-tuned. It is a property of
#: the encoding, not of the config, so it lives on the unit rather than in the
#: config dict that selection pools over.
REFERENCE_L2 = {"onehot": 1e-5, "blosum": 1e-3}

# --- A1 / A4: CatBoost -----------------------------------------------------

#: The §A1 grid, 12 configurations -- the plan §3.3 ceiling. A4 searches the
#: same three axes on its own budget rather than inheriting A1's winner: in this
#: project tuning parity is equal budget, not equal values, and a transplanted
#: L2 ladder once cost an arm 0.093 median Spearman.
TREE_DEPTHS = (4, 6, 8)
TREE_LEARNING_RATES = (0.03, 0.06)
TREE_L2_LEAF_REG = (3, 10)

#: Held fixed across both tree arms by §A1. Early stopping is on the inner fold,
#: never on an outer-held label.
CATBOOST_FIXED = {
    "iterations": 2000,
    "od_type": "Iter",
    "od_wait": 100,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "rsm": 1.0,
    "random_strength": 1.0,
    "border_count": 128,
}

#: §A4's two declared groupwise objectives. Both are *surrogates* for median
#: per-allele Spearman, not the metric itself.
A4_OBJECTIVES = ("YetiRank", "PairLogit")

# --- A2: allele-centred targets --------------------------------------------

#: The only axis A2 varies. James-Stein style shrinkage toward the fitting-fold
#: global mean; larger k shrinks harder, so larger k is the more regularised
#: choice for the parsimony rule.
A2_K_GRID = (1, 5, 20, 100)

# --- A3: objective x sampling ----------------------------------------------

#: The 2x2 of §A3. ``lam`` is the ranking-term weight; the MSE cells carry None.
A3_CELLS = {
    "A3-a": {"loss": "mse", "sampling": "natural"},
    "A3-b": {"loss": "mse", "sampling": "balanced"},
    "A3-c": {"loss": "mse+rank", "sampling": "natural"},
    "A3-d": {"loss": "mse+rank", "sampling": "balanced"},
}
A3_RANKING_CELLS = ("A3-c", "A3-d")
A3_LAMBDA_GRID = (0.1, 0.3, 1.0)

#: The stopping rule is its own axis, not folded into the loss. Stage 7's
#: censored-loss result traced an apparent ranking gain to the stopping rule
#: rather than the gradient; running both rules in every cell is what keeps that
#: confound out of A3.
A3_STOPPING = ("mse", "rank")

#: Keys that must be identical across the cells of a same-geometry ablation.
#: Enforced by :func:`assert_same_geometry_common`, not left to prose.
HELD_COMMON = {
    "A2": ("hidden", "lr", "batch_size", "max_epochs", "patience"),
    "A3": ("hidden", "lr", "batch_size", "max_epochs", "patience"),
}

# --- selection --------------------------------------------------------------

#: Per family, the axes of "most regularised" in priority order, with +1 meaning
#: smaller is more regularised and -1 meaning larger is. The tree ordering is
#: §A1's, verbatim. A2's is larger shrinkage. A3's ``lam`` is an objective
#: weight rather than a capacity control, so parsimony there means the smallest
#: departure from the reference MSE objective.
REGULARISATION_ORDER = {
    "tree": (("l2_leaf_reg", -1), ("depth", +1), ("learning_rate", +1)),
    "A2": (("k", -1),),
    "A3": (("lam", +1),),
}

SCORE_COLUMN = "median_per_allele_spearman"


# --- threads ---------------------------------------------------------------

#: BLAS/OMP thread count moves a single network by roughly 0.02 median Spearman
#: at a fixed seed here, so it is pinned per container and recorded on every
#: returned row. Conclusions rest on ensembles, which are stable to it.
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def pin_threads(n: int) -> None:
    """Set every BLAS/OMP thread variable to ``n``.

    Must run **before** numpy is imported to take effect -- the container entry
    point, not :func:`run_unit`, is the place for it.
    """
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(int(n))


def n_threads() -> int:
    """The pinned thread count, or the machine's core count if unpinned."""
    for var in THREAD_ENV_VARS:
        value = os.environ.get(var)
        if value and value.strip().isdigit():
            return int(value)
    return os.cpu_count() or 1


def _peak_rss_mb() -> float:
    """Peak resident set size of this process, in MiB.

    ``ru_maxrss`` is bytes on Darwin and kibibytes on Linux; the branch is on
    the magnitude because a container never legitimately peaks below 1 MiB.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 ** 2) if raw > 2 ** 20 else raw / 1024


# --- lazy imports -----------------------------------------------------------


def _nested():
    """The harness module. Imported lazily so this module loads without it."""
    from . import nested
    return nested


def _catboost():
    """CatBoost, imported lazily so A2 and A3 run without it installed."""
    import catboost
    return catboost


# --- payload ---------------------------------------------------------------


def load_payload() -> dict:
    """Client-side: the training split's sequence records and its outer folds.

    Column-oriented lists rather than per-row dicts, which is what keeps this
    at ~1.4 MB for 19,716 rows. Feature matrices are **not** shipped; they are
    rebuilt from the sequences in the container.
    """
    from .data import load_with_splits

    nested = _nested()
    df = load_with_splits()
    train = df[df.split == "train"].sort_values("pair_id").reset_index(drop=True)
    train = train.assign(outer_fold=nested.outer_folds(train))

    rows = {}
    for col in PAYLOAD_COLUMNS:
        series = train[col]
        if col in ("pair_id", "cluster_id", "outer_fold"):
            rows[col] = [int(v) for v in series]
        elif col == TARGET:
            rows[col] = [float(v) for v in series]
        else:
            rows[col] = [str(v) for v in series]

    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"schema": UNITS_SCHEMA_VERSION, "split": "train",
            "n_rows": len(train), "columns": list(PAYLOAD_COLUMNS),
            "rows": rows, "digest": digest}


def payload_frame(payload: dict) -> pd.DataFrame:
    """Rebuild the training frame inside the container. No disk access."""
    if payload.get("schema") != UNITS_SCHEMA_VERSION:
        raise ValueError(f"payload schema {payload.get('schema')!r} is not "
                         f"{UNITS_SCHEMA_VERSION!r}")
    if payload.get("split") != "train":
        raise ValueError("Phase A runs on the training split only; payload "
                         f"declares split {payload.get('split')!r}")
    frame = pd.DataFrame(payload["rows"])
    return frame[list(payload["columns"])]


# --- allele-centred targets (A2) -------------------------------------------


def shrunk_allele_means(allele, y, k: float) -> tuple[dict[str, float], float]:
    """Per-allele shrunk means and the global mean, from these rows alone.

    ``shrunk_mean(a) = (n_a * mean_a + k * global_mean) / (n_a + k)``, the §A2
    grid's single formula. The caller must pass the **fitting** rows only: a
    mean taken over the inner dev fold would tune the stopping epoch on its own
    target, and a mean taken over outer-held rows would let a held-out label
    choose a transform.
    """
    allele = np.asarray(allele)
    y = np.asarray(y, dtype=float)
    if not len(y):
        raise ValueError("no rows to take allele means over")
    global_mean = float(y.mean())
    frame = pd.DataFrame({"allele": allele, "y": y}).groupby("allele")["y"]
    n = frame.size().to_numpy(dtype=float)
    means = frame.mean().to_numpy(dtype=float)
    shrunk = (n * means + k * global_mean) / (n + k)
    return dict(zip(frame.size().index.tolist(), shrunk.tolist())), global_mean


def allele_offsets(allele, means: dict[str, float], global_mean: float) -> np.ndarray:
    """The offset to subtract (or add back) per row. Unseen alleles get the global mean."""
    return np.array([means.get(a, global_mean) for a in np.asarray(allele)],
                    dtype=float)


# --- allele-balanced sampling (A3) -----------------------------------------


def allele_balance_weights(allele) -> np.ndarray:
    """Per-row weights that give every allele equal total pull.

    Implemented as a weighted loss rather than as resampling. ``MLPRegressor``
    normalises the weighted loss by row count, so weights are scaled to mean 1
    and the natural-sampling cell is recovered exactly at all-ones -- resampling
    would instead add draw noise as a second difference between the cells.
    """
    allele = np.asarray(allele)
    counts = pd.Series(allele).value_counts()
    w = np.array([1.0 / counts[a] for a in allele], dtype=float)
    return (w / w.mean()).astype(DTYPE)


# --- within-allele pairwise ranking (A3) -----------------------------------


def ranking_pairs(allele, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Within-allele ordered pairs ``(higher, lower)`` with per-pair weights.

    §A3's three rules, in code:

    - pairs form only inside one allele;
    - **tied targets are skipped**, floor-floor pairs included -- no ordering is
      claimed for a pair whose assay intervals overlap, and 20.2% of labels sit
      exactly at the floor, so this is most of the pairs an unguarded rule would
      invent;
    - weights are ``1 / (A * P_a)`` for an allele contributing ``P_a`` pairs out
      of ``A`` contributing alleles, so a many-measurement allele cannot
      dominate quadratically.

    Returns three parallel arrays; all empty when no pair survives.
    """
    allele = np.asarray(allele)
    y = np.asarray(y, dtype=float)
    hi_parts, lo_parts, counts = [], [], []
    for a in np.unique(allele):
        idx = np.flatnonzero(allele == a)
        if len(idx) < 2:
            continue
        diff = y[idx][:, None] - y[idx][None, :]
        hi, lo = np.nonzero(diff > 0.0)
        if not len(hi):
            continue
        hi_parts.append(idx[hi])
        lo_parts.append(idx[lo])
        counts.append(len(hi))
    if not hi_parts:
        empty_i = np.zeros(0, dtype=np.int64)
        return empty_i, empty_i.copy(), np.zeros(0, dtype=float)
    n_alleles = len(counts)
    weight = np.concatenate([np.full(c, 1.0 / (n_alleles * c)) for c in counts])
    return (np.concatenate(hi_parts), np.concatenate(lo_parts), weight)


def pairwise_ranking_loss(pred, pairs) -> tuple[float, np.ndarray]:
    """RankNet logistic loss over ``pairs``, and its gradient w.r.t. ``pred``.

    ``loss = sum_pairs w * softplus(-(pred_hi - pred_lo))`` with the weights
    :func:`ranking_pairs` assigns. ``softplus`` via ``logaddexp`` so a large
    negative margin does not overflow.
    """
    pred = np.asarray(pred, dtype=float)
    hi, lo, weight = pairs
    grad = np.zeros(len(pred), dtype=float)
    if not len(hi):
        return 0.0, grad
    margin = pred[hi] - pred[lo]
    loss = float(np.dot(weight, np.logaddexp(0.0, -margin)))
    # d/dmargin softplus(-margin) = -sigmoid(-margin)
    g = -weight / (1.0 + np.exp(margin))
    np.add.at(grad, hi, g)
    np.add.at(grad, lo, -g)
    return loss, grad


class RankingMLPRegressor(MLPRegressor):
    """The reference MLP with an optional within-allele ranking term.

    A subclass, not an edit to :mod:`pepstab.mlp`: 540 existing tests depend on
    that module's gradients. The training loop is repeated here because the
    parent's ``fit`` has no hook for a custom ``dL/dpred``, and the repetition
    is guarded -- at ``rank_weight == 0`` with ``dev_metric == "mse"`` this path
    produces bit-identical weights to ``MLPRegressor.fit``.

    The gradient reaches the inherited ``_backward`` unmodified: that method
    forms ``delta = (2/n) * residual``, so a ranking contribution is passed as
    ``(n/2) * lambda * rank_grad`` added to the residual. No new backward pass
    exists.

    ``dev_metric`` selects the stopping criterion -- ``"mse"`` or the same
    pairwise ranking loss on the dev fold. §A3 reports it as its own axis.
    """

    def __init__(self, config: MLPConfig | None = None, *, rank_weight: float = 0.0,
                 dev_metric: str = "mse", **kwargs):
        super().__init__(config, **kwargs)
        if dev_metric not in A3_STOPPING:
            raise ValueError(f"dev_metric must be one of {A3_STOPPING}")
        if rank_weight < 0:
            raise ValueError("rank_weight must be non-negative")
        # ``rank_weight == 0`` with ``dev_metric == "rank"`` is legal and used:
        # §A3 runs the MSE cells under the ranking stopping rule too.
        self.rank_weight = float(rank_weight)
        self.dev_metric = dev_metric
        self.dev_rank_curve_: list[float] = []
        self.dev_rank_: float = float("nan")

    def fit(self, X, y, X_dev, y_dev, sample_weight=None,
            allele=None, allele_dev=None) -> "RankingMLPRegressor":
        """Fit with the §A3 objective. ``allele`` is required once ranking is on."""
        needs_allele = bool(self.rank_weight) or self.dev_metric == "rank"
        if needs_allele and (allele is None or allele_dev is None):
            raise ValueError("ranking loss and ranking-based stopping both need "
                             "allele labels for the fit and dev rows")

        cfg = self.config
        X = np.asarray(X, dtype=DTYPE)
        y = np.asarray(y, dtype=DTYPE)
        X_dev = np.asarray(X_dev, dtype=DTYPE)
        y_dev = np.asarray(y_dev, dtype=DTYPE)
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
        if not len(X_dev):
            raise ValueError("the dev fold is empty; early stopping needs rows")

        if sample_weight is None:
            weight = None
        else:
            weight = np.asarray(sample_weight, dtype=DTYPE)
            if weight.shape != (len(X),):
                raise ValueError(f"sample_weight has shape {weight.shape}, "
                                 f"expected ({len(X)},)")
            if not np.all(np.isfinite(weight)) or np.any(weight < 0):
                raise ValueError("sample_weight must be finite and non-negative")
            if not weight.sum():
                raise ValueError("sample_weight sums to 0; nothing to fit on")
        self.sample_weight_sum_ = float(len(X) if weight is None else weight.sum())

        rng = np.random.default_rng(cfg.seed)
        self.y_offset_ = float(y.mean() if weight is None
                               else np.dot(weight, y) / weight.sum())
        y_centred = y - self.y_offset_
        self._init_params(X.shape[1], rng)

        from .mlp import _Adam
        opt_w = [_Adam.for_param(W) for W in self.weights_]
        opt_b = [_Adam.for_param(b) for b in self.biases_]

        allele_arr = None if allele is None else np.asarray(allele)
        dev_pairs = (ranking_pairs(allele_dev, y_dev) if self.dev_metric == "rank"
                     else None)

        started = time.perf_counter()
        best_metric, best_epoch, best_state = np.inf, 0, self._snapshot()
        self.dev_curve_, self.dev_rank_curve_ = [], []
        step = 0
        for epoch in range(1, cfg.max_epochs + 1):
            order = rng.permutation(len(X))
            for start in range(0, len(order), cfg.batch_size):
                idx = order[start:start + cfg.batch_size]
                pred, acts = self._forward(X[idx])
                residual = pred - y_centred[idx]
                if self.rank_weight:
                    # Pairs are formed on the *target* passed in, so ties in it
                    # are skipped; prediction differences are unaffected by the
                    # constant y_offset_.
                    _, rank_grad = pairwise_ranking_loss(
                        pred, ranking_pairs(allele_arr[idx], y[idx]))
                    scaled = residual if weight is None else residual * weight[idx]
                    # _backward forms delta = (2/n) * residual, so the ranking
                    # term enters pre-scaled by n/2 and nothing is rescaled twice.
                    effective = (scaled + (len(idx) / 2.0) * self.rank_weight
                                 * rank_grad).astype(DTYPE)
                    gw, gb = self._backward(acts, effective, None)
                else:
                    gw, gb = self._backward(acts, residual,
                                            None if weight is None else weight[idx])
                step += 1
                for k in range(len(self.weights_)):
                    opt_w[k].step(self.weights_[k], gw[k] + cfg.l2 * self.weights_[k],
                                  cfg.lr, step)
                    opt_b[k].step(self.biases_[k], gb[k], cfg.lr, step)

            dev_pred = self.predict(X_dev)
            dev_mse = float(np.mean((dev_pred - y_dev) ** 2))
            self.dev_curve_.append(dev_mse)
            if self.dev_metric == "rank":
                dev_rank, _ = pairwise_ranking_loss(dev_pred, dev_pairs)
                self.dev_rank_curve_.append(dev_rank)
                metric = dev_rank
            else:
                metric = dev_mse
            if metric < best_metric:
                best_metric, best_epoch, best_state = metric, epoch, self._snapshot()
            elif epoch - best_epoch >= cfg.patience:
                break

        self.weights_, self.biases_ = best_state
        self.n_epochs_ = epoch
        self.best_epoch_ = best_epoch
        self.dev_mse_ = self.dev_curve_[best_epoch - 1]
        self.dev_rank_ = (self.dev_rank_curve_[best_epoch - 1]
                          if self.dev_rank_curve_ else float("nan"))
        self.fit_seconds_ = time.perf_counter() - started
        return self


# --- unit enumeration ------------------------------------------------------


def _label(parts: dict) -> str:
    return ",".join(f"{k}={'' if v is None else v}" for k, v in sorted(parts.items()))


def arm_configs(arm: str) -> list[dict]:
    """Every screened configuration of ``arm``, in a deterministic order."""
    out: list[dict] = []
    if arm == "A1":
        for depth, lr, l2 in itertools.product(TREE_DEPTHS, TREE_LEARNING_RATES,
                                               TREE_L2_LEAF_REG):
            out.append({"depth": depth, "learning_rate": lr, "l2_leaf_reg": l2})
    elif arm == "A4":
        # A4's own equal-budget search: the same three axes, 12 points per
        # objective, not A1's selected values transplanted.
        for objective in A4_OBJECTIVES:
            for depth, lr, l2 in itertools.product(TREE_DEPTHS, TREE_LEARNING_RATES,
                                                   TREE_L2_LEAF_REG):
                out.append({"objective": objective, "depth": depth,
                            "learning_rate": lr, "l2_leaf_reg": l2})
    elif arm == "A2":
        for k in A2_K_GRID:
            out.append({**REFERENCE_MLP, "hidden": list(REFERENCE_MLP["hidden"]),
                        "k": k})
    elif arm == "A3":
        for cell in A3_CELLS:
            lams = ((None,) if cell not in A3_RANKING_CELLS else A3_LAMBDA_GRID)
            for lam in lams:
                for stopping in A3_STOPPING:
                    out.append({**REFERENCE_MLP,
                                "hidden": list(REFERENCE_MLP["hidden"]),
                                "cell": cell, **A3_CELLS[cell],
                                "lam": lam, "stopping": stopping})
    else:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    for cfg in out:
        cfg["label"] = _label({k: v for k, v in cfg.items()
                               if k not in REFERENCE_MLP})
    return out


def _assess_configs(arm: str, selected: dict) -> list[dict]:
    """The finalist configurations for ``arm`` at assessment.

    ``selected[arm]`` holds what the inner ladder chose. A3 expands its 2x2 and
    both stopping rules at **one** common lambda -- the prespecification's
    same-geometry requirement, enforced here rather than left to prose.
    """
    if arm not in selected:
        raise ValueError(f"assessment needs a selected config for {arm}; "
                         f"got keys {sorted(selected)}")
    choice = selected[arm]
    if arm == "A3":
        lam = choice["lam"] if isinstance(choice, dict) else choice
        out = []
        for cell in A3_CELLS:
            for stopping in A3_STOPPING:
                cfg = {**REFERENCE_MLP, "hidden": list(REFERENCE_MLP["hidden"]),
                       "cell": cell, **A3_CELLS[cell],
                       "lam": None if cell not in A3_RANKING_CELLS else lam,
                       "stopping": stopping}
                cfg["label"] = _label({k: v for k, v in cfg.items()
                                       if k not in REFERENCE_MLP})
                out.append(cfg)
        return out
    chosen = choice if isinstance(choice, list) else [choice]
    out = []
    for cfg in chosen:
        cfg = dict(cfg)
        if arm == "A2":
            cfg = {**REFERENCE_MLP, "hidden": list(REFERENCE_MLP["hidden"]), **cfg}
        cfg["label"] = _label({k: v for k, v in cfg.items()
                               if k not in REFERENCE_MLP and k != "label"})
        out.append(cfg)
    return out


def enumerate_units(arms: list[str], stage: str = "screen",
                    selected: dict | None = None,
                    encodings: tuple[str, ...] = ENCODINGS) -> list[dict]:
    """One dict per ``(arm, config, encoding, seed, outer fold, inner fold)`` fit.

    ``stage="screen"`` sweeps each arm's prespecified grid at
    ``SCREEN_SEEDS`` x ``SCREEN_INNER_FOLDS``; ``stage="assess"`` runs the
    finalists in ``selected`` at ``ASSESS_SEEDS`` x ``ASSESS_INNER_FOLDS``, which
    is what makes an arm's per-outer-fold ensemble 5 x 2 x 3 = 30 members, the
    same arity as the deployed reference.

    ``encoding`` is a unit axis, not a grid axis: §A1 fixes the grid at 12
    points and the reference ensemble spans both encodings, so widening the grid
    to 24 would break the §3.3 ceiling while dropping an encoding would break
    ensemble parity.
    """
    nested = _nested()
    if stage == "screen":
        seeds, n_inner = SCREEN_SEEDS, nested.SCREEN_INNER_FOLDS
    elif stage == "assess":
        seeds, n_inner = ASSESS_SEEDS, nested.ASSESS_INNER_FOLDS
    else:
        raise ValueError(f"stage must be 'screen' or 'assess', got {stage!r}")

    units: list[dict] = []
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        configs = (arm_configs(arm) if stage == "screen"
                   else _assess_configs(arm, selected or {}))
        for config in configs:
            for encoding in encodings:
                for seed in seeds:
                    for fold in range(nested.OUTER_FOLDS):
                        for inner in range(n_inner):
                            units.append({
                                "schema": UNITS_SCHEMA_VERSION,
                                "key": (f"{arm}|{config['label']}|enc={encoding}"
                                        f"|seed={seed}|of={fold}|if={inner}"
                                        f"|{stage}"),
                                "arm": arm, "stage": stage, "config": config,
                                "encoding": encoding, "l2": REFERENCE_L2[encoding],
                                "seed": seed, "fold": fold,
                                "inner_fold": inner, "n_inner": n_inner,
                            })
    assert_same_geometry_common(units)
    return units


def assert_same_geometry_common(units: list[dict]) -> None:
    """Raise unless each same-geometry arm's cells share their held-common keys.

    §3 of the prespecification: where two arms differ only in objective,
    sampling or target parameterisation, they are compared at one common
    hyperparameter value. At assessment that extends to A3's lambda, which must
    be a single value across the two ranking cells -- screening is the step that
    sweeps it.
    """
    for arm, keys in HELD_COMMON.items():
        arm_units = [u for u in units if u["arm"] == arm]
        for key in keys:
            seen = {json.dumps(u["config"].get(key), sort_keys=True)
                    for u in arm_units}
            if len(seen) > 1:
                raise ValueError(f"{arm} is a same-geometry ablation but its "
                                 f"units carry {len(seen)} values of {key!r}: "
                                 f"{sorted(seen)}")
    assess_a3 = [u for u in units
                 if u["arm"] == "A3" and u["stage"] == "assess"
                 and u["config"]["cell"] in A3_RANKING_CELLS]
    lams = {u["config"]["lam"] for u in assess_a3}
    if len(lams) > 1:
        raise ValueError("A3's ranking cells must be assessed at one common "
                         f"lambda chosen on the pooled inner ladder; got {sorted(lams)}")


def smoke_units(arms: list[str] | None = None) -> list[dict]:
    """One unit per arm -- first config, first encoding, seed 0, fold 0, inner 0.

    CLAUDE.md: no batch job without a passing end-to-end run on 3-5 examples.
    A4 contributes **one unit per objective**, because its two objectives take
    different code paths -- ``PairLogit`` is handed ~800k explicit pairs while
    ``YetiRank`` generates its own -- and the explicit-pair path is both the
    slower one and the one a batch cannot afford to discover is broken.
    """
    picked = []
    for arm in (arms or list(ARMS)):
        units = [u for u in enumerate_units([arm])
                 if u["seed"] == 0 and u["fold"] == 0 and u["inner_fold"] == 0
                 and u["encoding"] == ENCODINGS[0]]
        if arm == "A4":
            for objective in A4_OBJECTIVES:
                picked.append(next(u for u in units
                                   if u["config"]["objective"] == objective))
        else:
            picked.append(units[0])
    return picked


# --- running a unit --------------------------------------------------------


def _split_masks(frame: pd.DataFrame, unit: dict):
    """Outer fitting / outer held masks, and the inner fold assignment on the fit rows."""
    nested = _nested()
    held_mask = (frame["outer_fold"].to_numpy() == unit["fold"])
    fit_rows = frame.loc[~held_mask].reset_index(drop=True)
    inner = np.asarray(nested.inner_folds(fit_rows, unit["n_inner"]))
    if len(inner) != len(fit_rows):
        raise ValueError(f"inner_folds returned {len(inner)} labels for "
                         f"{len(fit_rows)} fitting rows")
    return held_mask, fit_rows, inner


def _score_block(name: str, frame: pd.DataFrame, pred: np.ndarray) -> dict:
    """Median per-allele Spearman and coverage on one block of rows."""
    nested = _nested()
    allele = frame["allele"]
    y = frame[TARGET].to_numpy(dtype=float)
    min_rows = nested.MIN_ROWS_PER_ALLELE_FOLD
    scores = nested.score_fold(name, allele, y, np.asarray(pred, dtype=float),
                               min_rows=min_rows)
    coverage = nested.fold_coverage(allele, y, min_rows=min_rows)
    return {
        "median_per_allele_spearman": float(scores.median_spearman),
        "n_alleles_eligible": int(scores.n_alleles_eligible),
        "eligible_row_coverage": float(coverage["eligible_row_coverage"]),
        "mae_log1p": float(scores.mae_log1p),
        "coverage": coverage,
    }


def _mlp_member(unit: dict) -> tuple[MLPConfig, dict]:
    """The MLP config for an A2/A3 unit, with its arm-specific extras."""
    config = unit["config"]
    cfg = MLPConfig(hidden=tuple(config["hidden"]), l2=unit["l2"],
                    lr=config["lr"], batch_size=config["batch_size"],
                    max_epochs=config["max_epochs"], patience=config["patience"],
                    seed=unit["seed"])
    return cfg, config


def _run_a2(unit, frame, held_mask, fit_rows, inner) -> dict:
    """A2: allele-centred targets, reference hyperparameters, only ``k`` varying."""
    cfg, config = _mlp_member(unit)
    is_fit = inner != unit["inner_fold"]
    dev = fit_rows.loc[~is_fit]
    fit = fit_rows.loc[is_fit]
    held = frame.loc[held_mask]

    X = build_features(fit_rows, INPUT_SET, unit["encoding"])
    X_held = build_features(held, INPUT_SET, unit["encoding"])
    y = fit_rows[TARGET].to_numpy(dtype=float)

    # Means come from the rows the network actually fits on -- not the dev fold,
    # whose target would otherwise help choose the stopping epoch.
    means, global_mean = shrunk_allele_means(fit["allele"], y[is_fit], config["k"])
    off_fit = allele_offsets(fit["allele"], means, global_mean)
    off_dev = allele_offsets(dev["allele"], means, global_mean)
    off_held = allele_offsets(held["allele"], means, global_mean)

    model = MLPRegressor(cfg)
    model.fit(X[is_fit], y[is_fit] - off_fit, X[~is_fit], y[~is_fit] - off_dev)
    # Added back on the log1p scale, which is the scale the target was centred on.
    pred_dev = model.predict(X[~is_fit]) + off_dev
    pred_held = model.predict(X_held) + off_held
    return {"pred_oof": pred_dev, "pred_held": pred_held, "dev": dev, "held": held,
            "fit_seconds": model.fit_seconds_, "best_epoch": model.best_epoch_,
            "n_epochs": model.n_epochs_, "n_fit_rows": int(is_fit.sum()),
            "n_unseen_alleles_held": int(
                (~held["allele"].isin(list(means))).sum())}


def _run_a3(unit, frame, held_mask, fit_rows, inner) -> dict:
    """A3: one cell of the objective x sampling 2x2, at one stopping rule."""
    cfg, config = _mlp_member(unit)
    is_fit = inner != unit["inner_fold"]
    dev = fit_rows.loc[~is_fit]
    fit = fit_rows.loc[is_fit]
    held = frame.loc[held_mask]

    X = build_features(fit_rows, INPUT_SET, unit["encoding"])
    X_held = build_features(held, INPUT_SET, unit["encoding"])
    y = fit_rows[TARGET].to_numpy(dtype=float)

    # The ranking term is already normalised per allele, so the sampling axis
    # reaches the MSE term only -- which is exactly the separation §A3 wants.
    weights = (allele_balance_weights(fit["allele"])
               if config["sampling"] == "balanced" else None)
    rank_weight = config["lam"] if config["loss"] == "mse+rank" else 0.0
    model = RankingMLPRegressor(cfg, rank_weight=rank_weight or 0.0,
                                dev_metric=config["stopping"])
    model.fit(X[is_fit], y[is_fit], X[~is_fit], y[~is_fit],
              sample_weight=weights,
              allele=fit["allele"].to_numpy(), allele_dev=dev["allele"].to_numpy())
    return {"pred_oof": model.predict(X[~is_fit]),
            "pred_held": model.predict(X_held), "dev": dev, "held": held,
            "fit_seconds": model.fit_seconds_, "best_epoch": model.best_epoch_,
            "n_epochs": model.n_epochs_, "n_fit_rows": int(is_fit.sum()),
            "dev_mse": float(model.dev_mse_),
            # None rather than NaN under the MSE stopping rule: strict JSON has
            # no NaN, and the run manifest is read back as JSON.
            "dev_rank": (float(model.dev_rank_)
                         if np.isfinite(model.dev_rank_) else None)}


def _catboost_params(unit: dict, loss: str) -> dict:
    config = unit["config"]
    return {**CATBOOST_FIXED, "loss_function": loss,
            "depth": config["depth"], "learning_rate": config["learning_rate"],
            "l2_leaf_reg": config["l2_leaf_reg"], "random_seed": unit["seed"],
            "thread_count": n_threads(), "verbose": False,
            "allow_writing_files": False}


def _run_a1(unit, frame, held_mask, fit_rows, inner) -> dict:
    """A1: CatBoost RMSE on the 860-column positional encoding. No allele ID."""
    cb = _catboost()
    is_fit = inner != unit["inner_fold"]
    dev = fit_rows.loc[~is_fit]
    held = frame.loc[held_mask]

    # Plain float32 numpy, never a DataFrame: pandas 3.0 hands back
    # pyarrow-backed frames and the tree library wants contiguous float32.
    X = np.ascontiguousarray(build_features(fit_rows, INPUT_SET, unit["encoding"]),
                             dtype=np.float32)
    X_held = np.ascontiguousarray(build_features(held, INPUT_SET, unit["encoding"]),
                                  dtype=np.float32)
    y = fit_rows[TARGET].to_numpy(dtype=np.float32)

    model = cb.CatBoostRegressor(**_catboost_params(unit, "RMSE"))
    started = time.perf_counter()
    model.fit(X[is_fit], y[is_fit], eval_set=(X[~is_fit], y[~is_fit]),
              use_best_model=True)
    fit_seconds = time.perf_counter() - started
    return {"pred_oof": model.predict(X[~is_fit]),
            "pred_held": model.predict(X_held), "dev": dev, "held": held,
            "fit_seconds": fit_seconds, "best_epoch": int(model.get_best_iteration()),
            "n_epochs": int(model.tree_count_), "n_fit_rows": int(is_fit.sum()),
            "model_trees": int(model.tree_count_),
            # §A1 fixes od_type/od_wait but not the stopping metric, so the one
            # CatBoost resolved is recorded rather than left implicit.
            "eval_metric": str(model.get_all_params().get("eval_metric"))}


def _run_a4(unit, frame, held_mask, fit_rows, inner) -> dict:
    """A4: CatBoost groupwise ranker, allele as the ranking group.

    Ranking requires group-contiguous rows, so each block is sorted by allele
    and the predictions are scattered back to the caller's order.

    ``PairLogit`` is given **explicit** pairs built by :func:`ranking_pairs`, so
    §A3's within-allele, tie-skipping, per-allele-normalisation rules hold
    literally. ``YetiRank`` generates its own pairs from the labels inside each
    group and that generation cannot be overridden, so there the rules hold only
    through the group structure -- recorded on the row as ``pairs_explicit``.

    A ranker has no calibrated magnitude, so ``mae_log1p`` is reported only
    after an isotonic map fitted **inside the fitting rows**. Spearman stays on
    the raw score: isotonic flattening introduces ties, which would move the
    primary metric for a reason that has nothing to do with ranking.
    """
    cb = _catboost()
    from sklearn.isotonic import IsotonicRegression

    objective = unit["config"]["objective"]
    is_fit = inner != unit["inner_fold"]
    dev = fit_rows.loc[~is_fit]
    held = frame.loc[held_mask]

    X_all = np.ascontiguousarray(build_features(fit_rows, INPUT_SET, unit["encoding"]),
                                 dtype=np.float32)
    y_all = fit_rows[TARGET].to_numpy(dtype=np.float32)
    allele_all = fit_rows["allele"].to_numpy()

    def grouped(X, y, allele):
        """Rows sorted by group, plus the order needed to scatter back."""
        order = np.argsort(allele, kind="stable")
        return order, X[order], y[order], allele[order]

    fit_idx = np.flatnonzero(is_fit)
    dev_idx = np.flatnonzero(~is_fit)
    _, Xf, yf, af = grouped(X_all[fit_idx], y_all[fit_idx], allele_all[fit_idx])
    d_order, Xd, yd, ad = grouped(X_all[dev_idx], y_all[dev_idx],
                                  allele_all[dev_idx])
    X_held = np.ascontiguousarray(build_features(held, INPUT_SET, unit["encoding"]),
                                  dtype=np.float32)
    h_order, Xh, _, _ = grouped(X_held, held[TARGET].to_numpy(dtype=np.float32),
                                held["allele"].to_numpy())

    explicit = objective == "PairLogit"
    n_pairs = 0

    def pool(X, y, allele):
        nonlocal n_pairs
        kwargs = {"data": X, "label": y, "group_id": allele}
        if explicit:
            hi, lo, weight = ranking_pairs(allele, y)
            n_pairs = max(n_pairs, len(hi))
            if len(hi):
                kwargs["pairs"] = np.column_stack([hi, lo])
                # Rescaled to mean 1: CatBoost's pair weights are absolute, and
                # 1/(A*P_a) sums to 1 over ~800k pairs, which would shrink every
                # leaf update to nothing.
                kwargs["pairs_weight"] = weight / weight.mean()
        return cb.Pool(**kwargs)

    model = cb.CatBoost(_catboost_params(unit, objective))
    started = time.perf_counter()
    model.fit(pool(Xf, yf, af), eval_set=pool(Xd, yd, ad), use_best_model=True)
    fit_seconds = time.perf_counter() - started

    raw_fit = np.asarray(model.predict(Xf), dtype=float)
    raw_dev = np.asarray(model.predict(Xd), dtype=float)
    raw_held = np.asarray(model.predict(Xh), dtype=float)

    # Fitted on the fitting rows only; the inner dev fold chose the iteration
    # count and must not also choose the calibration map.
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_fit, yf.astype(float))
    cal_dev, cal_held = iso.predict(raw_dev), iso.predict(raw_held)

    def unsort(order, values):
        out = np.empty(len(values), dtype=float)
        out[order] = values
        return out

    return {"pred_oof": unsort(d_order, raw_dev),
            "pred_held": unsort(h_order, raw_held),
            "pred_oof_calibrated": unsort(d_order, cal_dev),
            "pred_held_calibrated": unsort(h_order, cal_held),
            "dev": dev, "held": held, "fit_seconds": fit_seconds,
            "best_epoch": int(model.get_best_iteration()),
            "n_epochs": int(model.tree_count_), "n_fit_rows": int(len(fit_idx)),
            "model_trees": int(model.tree_count_),
            # CatBoost picks the stopping metric from the objective -- PFound for
            # YetiRank, which is top-heavy and not the panel median. Recorded so
            # the stopping surface is visible next to the score.
            "eval_metric": str(model.get_all_params().get("eval_metric")),
            "pairs_explicit": explicit, "n_pairs": int(n_pairs)}


_RUNNERS = {"A1": _run_a1, "A2": _run_a2, "A3": _run_a3, "A4": _run_a4}


def run_unit(unit: dict, payload: dict) -> dict:
    """Run one fit in a container. Pure: no file I/O, no network, no repo reads.

    Returns a JSON-able row. On an internal failure it returns ``ok=False`` with
    an ``error`` string rather than raising, so one bad unit does not take a
    batch down.
    """
    row = {"schema": UNITS_SCHEMA_VERSION, "key": unit.get("key"),
           "arm": unit.get("arm"), "stage": unit.get("stage"),
           "config": unit.get("config"), "encoding": unit.get("encoding"),
           "seed": unit.get("seed"), "fold": unit.get("fold"),
           "inner_fold": unit.get("inner_fold"), "n_inner": unit.get("n_inner"),
           "n_threads": n_threads(),
           "thread_env": {v: os.environ.get(v) for v in THREAD_ENV_VARS},
           "payload_digest": payload.get("digest"), "ok": False}
    started = time.perf_counter()
    try:
        if unit.get("schema") != UNITS_SCHEMA_VERSION:
            raise ValueError(f"unit schema {unit.get('schema')!r} is not "
                             f"{UNITS_SCHEMA_VERSION!r}")
        frame = payload_frame(payload)
        held_mask, fit_rows, inner = _split_masks(frame, unit)
        out = _RUNNERS[unit["arm"]](unit, frame, held_mask, fit_rows, inner)

        inner_scores = _score_block(f"{unit['key']}:inner", out["dev"],
                                    out["pred_oof"])
        held_scores = _score_block(f"{unit['key']}:held", out["held"],
                                   out["pred_held"])
        row.update(inner_scores)
        row["held_scores"] = held_scores
        if "pred_oof_calibrated" in out:
            # A pure ranker's magnitude is meaningless; log1p MAE comes from the
            # isotonic map and the raw score keeps the ranking metric tie-free.
            row["mae_log1p_raw"] = row["mae_log1p"]
            row["mae_log1p"] = _score_block(f"{unit['key']}:cal", out["dev"],
                                            out["pred_oof_calibrated"])["mae_log1p"]
            row["oof_calibrated"] = [
                [int(p), float(v)] for p, v in
                zip(out["dev"]["pair_id"], out["pred_oof_calibrated"])]
            row["held_calibrated"] = [
                [int(p), float(v)] for p, v in
                zip(out["held"]["pair_id"], out["pred_held_calibrated"])]

        row["oof"] = [[int(p), float(v)]
                      for p, v in zip(out["dev"]["pair_id"], out["pred_oof"])]
        row["held"] = [[int(p), float(v)]
                       for p, v in zip(out["held"]["pair_id"], out["pred_held"])]
        for key in ("fit_seconds", "best_epoch", "n_epochs", "n_fit_rows",
                    "model_trees", "eval_metric", "dev_mse", "dev_rank",
                    "pairs_explicit", "n_pairs", "n_unseen_alleles_held"):
            if key in out:
                row[key] = out[key]
        row["fit_seconds"] = float(out["fit_seconds"])
        row["ok"] = True
    except Exception as exc:  # one bad unit must not take the batch down
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["unit_seconds"] = time.perf_counter() - started
    row["peak_rss_mb"] = _peak_rss_mb()
    return row


# --- selection machinery ---------------------------------------------------


def parsimonious_select(runs: pd.DataFrame, family: str) -> dict:
    """The one-standard-error parsimony rule, from ``stage3e_coupling::_parsimonious``.

    Among configurations whose mean inner-fold median Spearman is within one
    standard error of the best, take the **most regularised** one, with "most
    regularised" defined per family in :data:`REGULARISATION_ORDER`. The SE is
    the seed/fold spread at the best configuration divided by the root of its
    run count, exactly as the house rule computes it.

    Argmax and selected are returned side by side with the spread at each and an
    ``interior`` flag, because a selection at a grid boundary is flagged, not
    quietly accepted. Axes with only two tried values can never be interior, so
    ``boundary_axes`` names them rather than letting ``interior`` read as a
    verdict on the whole grid.
    """
    if family not in REGULARISATION_ORDER:
        raise ValueError(f"unknown family {family!r}; expected one of "
                         f"{sorted(REGULARISATION_ORDER)}")
    order = REGULARISATION_ORDER[family]
    axes = [name for name, _ in order]
    missing = [c for c in [*axes, SCORE_COLUMN] if c not in runs.columns]
    if missing:
        raise ValueError(f"runs is missing columns {missing}")
    runs = runs.dropna(subset=axes + [SCORE_COLUMN])
    if runs.empty:
        raise ValueError("no runs to select over")

    grouped = runs.groupby(axes, sort=True)[SCORE_COLUMN]
    means = grouped.mean()
    counts = grouped.count()
    stds = grouped.std()
    best = means.idxmax()
    n = max(int(counts[best]), 2)
    se = float(stds[best]) / np.sqrt(n)
    if not np.isfinite(se):
        se = 0.0
    ok = means[means >= means.max() - se]

    def as_tuple(key):
        return key if isinstance(key, tuple) else (key,)

    def regularisation_key(key):
        return tuple(sign * float(v) for (_, sign), v in zip(order, as_tuple(key)))

    selected = min(ok.index, key=regularisation_key)

    def spread(key):
        mask = np.ones(len(runs), dtype=bool)
        for name, value in zip(axes, as_tuple(key)):
            mask &= (runs[name].to_numpy() == value)
        sub = runs.loc[mask]
        if "seed" in sub.columns:
            per_seed = sub.groupby("seed")[SCORE_COLUMN].mean()
            return float(np.ptp(per_seed.to_numpy())) if len(per_seed) else 0.0
        return float(np.ptp(sub[SCORE_COLUMN].to_numpy()))

    tried = {name: sorted(runs[name].unique().tolist()) for name in axes}
    boundary = [name for name, value in zip(axes, as_tuple(selected))
                if value in (min(tried[name]), max(tried[name]))]
    interior = all(len(tried[name]) >= 3 and name not in boundary for name in axes)
    return {
        "family": family, "axes": axes, "tried": tried,
        "rule": "most regularised within 1 SE of the best, "
                + " then ".join(f"{'larger' if s < 0 else 'smaller'} {a}"
                                for a, s in order),
        "argmax": dict(zip(axes, as_tuple(best))),
        "selected": dict(zip(axes, as_tuple(selected))),
        "score_at_argmax": float(means[best]),
        "score_at_selected": float(means[selected]),
        "seed_spread_at_argmax": spread(best),
        "seed_spread_at_selected": spread(selected),
        "standard_error": float(se), "n_runs_at_argmax": int(counts[best]),
        "n_configs": int(len(means)), "n_within_1se": int(len(ok)),
        "interior": bool(interior), "boundary_axes": boundary,
    }


def common_hyperparameters(runs: pd.DataFrame, family: str, cells: tuple[str, ...],
                           cell_column: str = "cell") -> dict:
    """One hyperparameter value for a set of same-geometry cells.

    §3: where two arms differ only in an objective, a sampling scheme or a
    target parameterisation, they are run at **one** value chosen on the
    *pooled* inner ladder -- letting them sit at different values adds a second
    difference and the gap stops isolating the intended one. Pooling is the
    enforcement, so this refuses to run unless every named cell is present.
    """
    if cell_column not in runs.columns:
        raise ValueError(f"runs has no {cell_column!r} column to pool over")
    pooled = runs[runs[cell_column].isin(cells)]
    present = set(pooled[cell_column].unique())
    if set(cells) - present:
        raise ValueError(f"pooled ladder is missing cells "
                         f"{sorted(set(cells) - present)}; a common value chosen "
                         "on a partial pool is not common")
    out = parsimonious_select(pooled, family)
    out["pooled_cells"] = list(cells)
    out["n_runs_pooled"] = int(len(pooled))
    return out


def runs_frame(rows: list[dict]) -> pd.DataFrame:
    """Flatten :func:`run_unit` rows into the frame the selection rules read.

    Config axes are lifted to top-level columns so ``parsimonious_select`` can
    group on them; the prediction lists are dropped, because a selection frame
    that carries 20k predictions per row is unusable.
    """
    flat = []
    for row in rows:
        if not row.get("ok"):
            continue
        config = row.get("config") or {}
        flat.append({
            "key": row["key"], "arm": row["arm"], "stage": row["stage"],
            "encoding": row["encoding"], "seed": row["seed"], "fold": row["fold"],
            "inner_fold": row["inner_fold"], "n_threads": row["n_threads"],
            SCORE_COLUMN: row[SCORE_COLUMN],
            "n_alleles_eligible": row["n_alleles_eligible"],
            "eligible_row_coverage": row["eligible_row_coverage"],
            "mae_log1p": row["mae_log1p"], "fit_seconds": row["fit_seconds"],
            "peak_rss_mb": row.get("peak_rss_mb"),
            **{k: v for k, v in config.items() if k != "label"},
            "config_label": config.get("label"),
        })
    return pd.DataFrame(flat)
