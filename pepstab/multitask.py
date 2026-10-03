"""A two-headed MLP: stability plus an auxiliary affinity target. Stage 2c's model.

The question stage 2c asks is whether binding-affinity labels, which are cheap
and cover far more peptides than the stability assay, teach the shared encoder
anything the stability labels alone do not. The architecture is the stage 2
network with one extra output unit:

    X -> [shared ReLU trunk] -> stability head (1 unit)   <- y_log1p
                             -> affinity head  (1 unit)   <- y_affinity

``lambda_aff`` weights the auxiliary loss. **``lambda_aff = 0`` is the
single-task comparator**, and it is the same code path, the same initialisation
and the same minibatch order as ``lambda_aff > 0`` -- the affinity head is
always allocated and always drawn from its own generator, so changing
``lambda_aff`` changes nothing but the gradient. That is what makes the
comparison a controlled one instead of two models that happen to be similar.

At ``lambda_aff = 0`` this class is also **bit-identical to**
:class:`pepstab.mlp.MLPRegressor` at the same seed and config: the main
generator draws the trunk and the stability head in exactly the same order, then
the minibatch permutations, while the affinity head comes from a second
generator that never touches that stream. ``tests/test_multitask.py`` asserts
it. So stage 2c's single-task arm is not a near-replica of the stage 2 baseline;
it is the stage 2 baseline.

Two details that decide whether the auxiliary head means anything:

**The affinity loss is masked.** Only 5,135 of the 19,716 training rows (26.0%)
carry an affinity measurement. The auxiliary MSE is averaged over the *labelled*
rows in each minibatch, not over all of them. Averaging over all rows would make
the effective ``lambda_aff`` drift with each batch's label rate and shrink it by
~4x overall, so a swept value would not mean what it says.

**Stopping is on stability alone.** Both arms stop on the same quantity -- dev
MSE on ``y_log1p`` -- so the auxiliary task cannot win by buying itself extra
epochs, and the epoch count stays comparable across ``lambda_aff``.

A note on what ``lambda_aff`` actually controls under Adam. Adam normalises each
parameter's update by its own gradient RMS, so scaling the affinity head's
gradient by a constant barely changes that head's step size. What ``lambda_aff``
does change is the *ratio* of the two gradients where they sum, at the last
shared activation -- which is the mixing weight we want to sweep. The affinity
head trains at roughly full speed at any non-zero ``lambda_aff``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .mlp import DTYPE, _Adam

#: Seed offset for the affinity head's initialisation. It is drawn from its own
#: generator so the main stream -- trunk init, stability head init, then every
#: minibatch permutation -- is untouched by this class's existence. Without
#: this, adding a second head would shift every subsequent random draw and the
#: lambda=0 arm could not be checked against stage 2.
AFFINITY_HEAD_SEED_OFFSET = 97


@dataclass(frozen=True)
class MultiTaskMLPConfig:
    """One point in the stage 2c grid. Mirrors :class:`pepstab.mlp.MLPConfig`.

    ``lambda_aff`` is the only field stage 2c sweeps; everything else is pinned
    to whatever stage 2 selected for the arm under test, so the auxiliary loss
    is the only thing that changes.
    """

    hidden: tuple[int, ...] = (256, 64)
    l2: float = 1e-5
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 300
    patience: int = 15
    seed: int = 0
    lambda_aff: float = 0.0

    def label(self) -> str:
        return (f"h{'x'.join(map(str, self.hidden))}_l2{self.l2:g}"
                f"_lr{self.lr:g}_lam{self.lambda_aff:g}")

    def as_dict(self) -> dict:
        return {
            "hidden": "x".join(map(str, self.hidden)),
            "l2": self.l2,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "seed": self.seed,
            "lambda_aff": self.lambda_aff,
        }


class MultiTaskMLPRegressor:
    """ReLU trunk with a stability head and an auxiliary affinity head.

    ``fit`` takes the affinity target as a parallel array with NaN wherever the
    pair has no measurement; nothing else about the call signature differs from
    :meth:`pepstab.mlp.MLPRegressor.fit`.

    Reports the same cost fields stage 2 reports -- :attr:`fit_seconds_`,
    :attr:`n_epochs_`, :attr:`best_epoch_` -- plus :attr:`n_affinity_rows_`,
    the number of auxiliary labels the fit actually saw.
    """

    def __init__(self, config: MultiTaskMLPConfig | None = None, **kwargs):
        self.config = config or MultiTaskMLPConfig(**kwargs)
        #: Shared trunk only. The two heads live in ``head_w_`` / ``head_b_``.
        self.weights_: list[np.ndarray] = []
        self.biases_: list[np.ndarray] = []
        #: ``[stability, affinity]`` -- order fixed, because the stability head
        #: must occupy the same position in the main generator's draw sequence
        #: as :class:`pepstab.mlp.MLPRegressor`'s output layer.
        self.head_w_: list[np.ndarray] = []
        self.head_b_: list[np.ndarray] = []
        self.y_offset_: float = 0.0
        self.aff_offset_: float = 0.0
        self.dev_curve_: list[float] = []
        self.best_epoch_: int = 0
        self.n_epochs_: int = 0
        self.dev_mse_: float = float("nan")
        self.fit_seconds_: float = float("nan")
        self.n_affinity_rows_: int = 0

    # --- internals --------------------------------------------------------

    def _init_params(self, n_in: int, rng: np.random.Generator) -> None:
        """Draw the trunk and stability head from ``rng``, affinity head apart.

        The draw order is deliberately identical to
        :meth:`pepstab.mlp.MLPRegressor._init_params` for the sizes
        ``(n_in, *hidden, 1)``, so the main generator is left in exactly the
        state stage 2 leaves it in before the first epoch.
        """
        sizes = (n_in, *self.config.hidden, 1)
        self.weights_, self.biases_ = [], []
        self.head_w_, self.head_b_ = [], []
        for k, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = k == len(sizes) - 2
            # He for ReLU layers; the linear head starts small so the first
            # predictions sit near the centred target. Same as stage 2.
            scale = np.sqrt(2.0 / a) if not last else np.sqrt(1.0 / a)
            W = rng.normal(0.0, scale, size=(a, b)).astype(DTYPE)
            bias = np.zeros(b, dtype=DTYPE)
            if last:
                self.head_w_.append(W)
                self.head_b_.append(bias)
            else:
                self.weights_.append(W)
                self.biases_.append(bias)

        # Affinity head, from its own generator. Always allocated -- including
        # at lambda_aff = 0, where it simply never receives a gradient -- so the
        # single-task and multi-task arms differ in no other respect.
        head_rng = np.random.default_rng(self.config.seed + AFFINITY_HEAD_SEED_OFFSET)
        n_last = sizes[-2]
        self.head_w_.append(
            head_rng.normal(0.0, np.sqrt(1.0 / n_last), size=(n_last, 1)).astype(DTYPE))
        self.head_b_.append(np.zeros(1, dtype=DTYPE))

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        """Return ``(stability, affinity, activations)``, both heads raw."""
        acts = [X]
        h = X
        for W, b in zip(self.weights_, self.biases_):
            h = np.maximum(h @ W + b, 0.0)
            acts.append(h)
        stab = (h @ self.head_w_[0] + self.head_b_[0]).ravel()
        aff = (h @ self.head_w_[1] + self.head_b_[1]).ravel()
        return stab, aff, acts

    def _backward(self, acts: list[np.ndarray], stab: np.ndarray, aff: np.ndarray,
                  y_stab: np.ndarray, y_aff: np.ndarray, mask: np.ndarray
                  ) -> tuple[list[np.ndarray], list[np.ndarray],
                             list[np.ndarray], list[np.ndarray]]:
        """Gradients of ``MSE_stab + lambda_aff * MSE_aff`` over one minibatch.

        ``y_stab`` and ``y_aff`` are already centred; ``mask`` marks the rows
        carrying an affinity label. The stability term averages over every row,
        the affinity term over the masked rows only -- see the module docstring
        for why that normalisation is the one that makes ``lambda_aff``
        interpretable.

        Extracted from :meth:`fit` so ``tests/test_multitask.py`` can check it
        against central differences. The two-head sum at ``delta`` is the step
        worth checking: get it wrong and the auxiliary task would silently fail
        to reach the shared trunk, which is the entire mechanism under test.
        """
        lam = self.config.lambda_aff
        n = len(stab)
        h_last = acts[-1]

        d_stab = ((2.0 / n) * (stab - y_stab)[:, None]).astype(DTYPE)
        n_lab = int(mask.sum())
        d_aff = np.zeros((n, 1), dtype=DTYPE)
        if lam > 0 and n_lab:
            d_aff[mask, 0] = (2.0 * lam / n_lab) * (aff[mask] - y_aff[mask])

        ghw = [h_last.T @ d_stab, h_last.T @ d_aff]
        ghb = [d_stab.sum(axis=0), d_aff.sum(axis=0)]

        # Both heads' gradients meet here. This sum is the only place
        # lambda_aff has real leverage under Adam.
        delta = (d_stab @ self.head_w_[0].T + d_aff @ self.head_w_[1].T)
        delta = (delta * (h_last > 0.0)).astype(DTYPE)

        gw: list[np.ndarray] = [None] * len(self.weights_)  # type: ignore[list-item]
        gb: list[np.ndarray] = [None] * len(self.biases_)  # type: ignore[list-item]
        for k in range(len(self.weights_) - 1, -1, -1):
            gw[k] = acts[k].T @ delta
            gb[k] = delta.sum(axis=0)
            if k:
                delta = ((delta @ self.weights_[k].T) * (acts[k] > 0.0)).astype(DTYPE)
        return gw, gb, ghw, ghb

    def _snapshot(self):
        return ([W.copy() for W in self.weights_], [b.copy() for b in self.biases_],
                [W.copy() for W in self.head_w_], [b.copy() for b in self.head_b_])

    # --- public API -------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, y_aff: np.ndarray,
            X_dev: np.ndarray, y_dev: np.ndarray) -> "MultiTaskMLPRegressor":
        """Fit on ``(X, y, y_aff)``, stopping on stability dev MSE.

        ``y_aff`` is the auxiliary target aligned to ``X``, NaN where the pair
        carries no affinity measurement. Those rows contribute to the stability
        loss and not to the auxiliary one.

        The dev fold is used for one thing: choosing the epoch, on stability
        only. It is the caller's job to keep it separated from ``X`` by whole
        peptide clusters, exactly as in stage 2.
        """
        cfg = self.config
        X = np.asarray(X, dtype=DTYPE)
        y = np.asarray(y, dtype=DTYPE)
        y_aff = np.asarray(y_aff, dtype=np.float64)
        X_dev = np.asarray(X_dev, dtype=DTYPE)
        y_dev = np.asarray(y_dev, dtype=DTYPE)
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
        if len(y_aff) != len(y):
            raise ValueError(f"y_aff has {len(y_aff)} rows but y has {len(y)}")
        if not len(X_dev):
            raise ValueError("the dev fold is empty; early stopping needs rows")
        if cfg.lambda_aff < 0:
            raise ValueError(f"lambda_aff must be >= 0, got {cfg.lambda_aff}")

        has_aff = np.isfinite(y_aff)
        self.n_affinity_rows_ = int(has_aff.sum())
        if cfg.lambda_aff > 0 and not self.n_affinity_rows_:
            raise ValueError("lambda_aff > 0 but no row carries an affinity label")

        rng = np.random.default_rng(cfg.seed)
        # Centre both targets so a zero-initialised head bias predicts the mean.
        # Added back in predict(); neither is a fitted parameter.
        self.y_offset_ = float(y.mean())
        y_centred = y - self.y_offset_
        self.aff_offset_ = (float(y_aff[has_aff].mean()) if self.n_affinity_rows_
                            else 0.0)
        aff_centred = np.where(has_aff, y_aff - self.aff_offset_, 0.0).astype(DTYPE)

        self._init_params(X.shape[1], rng)
        opt_w = [_Adam.for_param(W) for W in self.weights_]
        opt_b = [_Adam.for_param(b) for b in self.biases_]
        opt_hw = [_Adam.for_param(W) for W in self.head_w_]
        opt_hb = [_Adam.for_param(b) for b in self.head_b_]

        started = time.perf_counter()
        best_mse, best_epoch, best_state = np.inf, 0, self._snapshot()
        self.dev_curve_ = []
        step = 0
        for epoch in range(1, cfg.max_epochs + 1):
            order = rng.permutation(len(X))
            for start in range(0, len(order), cfg.batch_size):
                idx = order[start:start + cfg.batch_size]
                stab, aff, acts = self._forward(X[idx])
                gw, gb, ghw, ghb = self._backward(
                    acts, stab, aff, y_centred[idx], aff_centred[idx], has_aff[idx])

                step += 1
                # L2 on weights only, heads included; penalising biases just
                # fights the target offset. At lambda_aff = 0 the affinity head
                # receives L2 decay and nothing else, so it carries no
                # information about affinity: a shrinking random projection of
                # the trunk early on, collapsing to a constant if the run is
                # long enough. Both read as "never trained", which is the
                # control affinity_head_quality() is interpreted against.
                for k in range(len(self.weights_)):
                    opt_w[k].step(self.weights_[k], gw[k] + cfg.l2 * self.weights_[k],
                                  cfg.lr, step)
                    opt_b[k].step(self.biases_[k], gb[k], cfg.lr, step)
                for j in range(2):
                    opt_hw[j].step(self.head_w_[j], ghw[j] + cfg.l2 * self.head_w_[j],
                                   cfg.lr, step)
                    opt_hb[j].step(self.head_b_[j], ghb[j], cfg.lr, step)

            # Stability only: both arms stop on the same quantity.
            dev_mse = float(np.mean((self.predict(X_dev) - y_dev) ** 2))
            self.dev_curve_.append(dev_mse)
            if dev_mse < best_mse:
                best_mse, best_epoch, best_state = dev_mse, epoch, self._snapshot()
            elif epoch - best_epoch >= cfg.patience:
                break

        self.weights_, self.biases_, self.head_w_, self.head_b_ = best_state
        self.n_epochs_ = epoch
        self.best_epoch_ = best_epoch
        self.dev_mse_ = best_mse
        self.fit_seconds_ = time.perf_counter() - started
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Stability prediction on the ``log1p`` scale. The reported output."""
        if not self.weights_ and not self.head_w_:
            raise RuntimeError("predict() before fit()")
        stab, _, _ = self._forward(np.asarray(X, dtype=DTYPE))
        return stab + self.y_offset_

    def predict_affinity(self, X: np.ndarray) -> np.ndarray:
        """Auxiliary head output, on the transformed affinity scale.

        Diagnostic only -- it says whether the auxiliary task was learned at
        all, which is what distinguishes "affinity does not help" from "the
        affinity head never trained".
        """
        if not self.head_w_:
            raise RuntimeError("predict_affinity() before fit()")
        _, aff, _ = self._forward(np.asarray(X, dtype=DTYPE))
        return aff + self.aff_offset_

    def n_parameters(self) -> int:
        return (sum(W.size for W in self.weights_) + sum(b.size for b in self.biases_)
                + sum(W.size for W in self.head_w_) + sum(b.size for b in self.head_b_))
