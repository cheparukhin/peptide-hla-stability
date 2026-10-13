"""A small MLP regressor in numpy. Stage 2's model, deliberately plain.

Why hand-rolled rather than ``sklearn.neural_network.MLPRegressor``: stopping
needs an **explicit** held-out fold. sklearn's ``early_stopping`` carves its own
fold out of the training rows at random, which splits peptide clusters across
the fit/stop boundary and would tune the epoch count on leaked near-duplicates
-- the one thing the frozen splits exist to prevent. Here the caller passes the
fold, so :mod:`scripts.baseline_sequence` can cut it along cluster boundaries.

Everything else is ordinary: ReLU hidden layers, a linear output, mean squared
error on ``y_log1p``, Adam, L2 on weights (not biases), and the best-dev weights
restored at the end. Fitting is deterministic given ``seed`` -- the only
randomness is initialisation and minibatch order, both from one seeded
generator.

``fit`` accepts an optional per-row ``sample_weight``, which stage 2b needs to
down-weight assumed-zero augmentation rows against measured ones. The weighted
loss is ``sum_i w_i (pred_i - y_i)^2 / n_batch`` -- normalised by the batch's
*row count*, not its weight sum, so uniform weights of 1 reproduce the
unweighted loss exactly and the measured-only arm is bit-identical to a run
that passes no weights at all. The dev fold is never weighted: it holds
measured rows only, and its MSE has to mean the same thing in every arm for the
stopping epoch to be comparable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

#: Parameters and activations are float32. The features are one-hot or scaled
#: BLOSUM rows, the widest arm is 3,820 columns, and float32 halves both the
#: matmul time and the 300 MB feature block; float64 changes the fitted
#: predictions by far less than the seed-to-seed spread.
DTYPE = np.float32


@dataclass(frozen=True)
class MLPConfig:
    """One point in the stage 2 tuning grid."""

    hidden: tuple[int, ...] = (256, 64)
    l2: float = 1e-5
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 150
    patience: int = 15
    seed: int = 0

    def label(self) -> str:
        return f"h{'x'.join(map(str, self.hidden))}_l2{self.l2:g}_lr{self.lr:g}"

    def as_dict(self) -> dict:
        return {
            "hidden": "x".join(map(str, self.hidden)),
            "l2": self.l2,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "seed": self.seed,
        }


@dataclass
class _Adam:
    """Adam state for one parameter array."""

    m: np.ndarray
    v: np.ndarray
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    @classmethod
    def for_param(cls, p: np.ndarray) -> "_Adam":
        return cls(m=np.zeros_like(p), v=np.zeros_like(p))

    def step(self, p: np.ndarray, grad: np.ndarray, lr: float, t: int) -> None:
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad * grad
        m_hat = self.m / (1 - self.beta1 ** t)
        v_hat = self.v / (1 - self.beta2 ** t)
        p -= lr * m_hat / (np.sqrt(v_hat) + self.eps)


class MLPRegressor:
    """ReLU MLP trained by Adam, stopped on a caller-supplied dev fold.

    ``fit`` records what stage 2 has to report next to accuracy:
    :attr:`fit_seconds_`, :attr:`n_epochs_` (epochs actually run) and
    :attr:`best_epoch_` (the one whose weights were kept).
    """

    def __init__(self, config: MLPConfig | None = None, **kwargs):
        self.config = config or MLPConfig(**kwargs)
        self.weights_: list[np.ndarray] = []
        self.biases_: list[np.ndarray] = []
        self.y_offset_: float = 0.0
        self.dev_curve_: list[float] = []
        self.best_epoch_: int = 0
        self.n_epochs_: int = 0
        self.dev_mse_: float = float("nan")
        self.fit_seconds_: float = float("nan")
        self.sample_weight_sum_: float = float("nan")

    # --- internals --------------------------------------------------------

    def _init_params(self, n_in: int, rng: np.random.Generator) -> None:
        sizes = (n_in, *self.config.hidden, 1)
        self.weights_, self.biases_ = [], []
        for k, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = k == len(sizes) - 2
            # He for ReLU layers; the linear output layer starts small so the
            # first predictions sit near the centred target rather than far
            # from it.
            scale = np.sqrt(2.0 / a) if not last else np.sqrt(1.0 / a)
            self.weights_.append(rng.normal(0.0, scale, size=(a, b)).astype(DTYPE))
            self.biases_.append(np.zeros(b, dtype=DTYPE))

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Return the output and the per-layer activations fed into each layer."""
        acts = [X]
        h = X
        for k, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            z = h @ W + b
            h = z if k == len(self.weights_) - 1 else np.maximum(z, 0.0)
            acts.append(h)
        return h.ravel(), acts

    def _backward(self, acts: list[np.ndarray], residual: np.ndarray,
                  weight: np.ndarray | None = None
                  ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        n = len(residual)
        # d/dpred of  sum_i w_i (pred_i - y_i)^2 / n, which is mean((pred-y)^2)
        # when every w_i is 1.
        scaled = residual if weight is None else residual * weight
        delta = ((2.0 / n) * scaled[:, None]).astype(DTYPE)
        gw: list[np.ndarray] = [None] * len(self.weights_)  # type: ignore[list-item]
        gb: list[np.ndarray] = [None] * len(self.biases_)  # type: ignore[list-item]
        for k in range(len(self.weights_) - 1, -1, -1):
            gw[k] = acts[k].T @ delta
            gb[k] = delta.sum(axis=0)
            if k:
                delta = (delta @ self.weights_[k].T) * (acts[k] > 0.0)
        return gw, gb

    def _snapshot(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return ([W.copy() for W in self.weights_], [b.copy() for b in self.biases_])

    # --- public API -------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_dev: np.ndarray, y_dev: np.ndarray,
            sample_weight: np.ndarray | None = None) -> "MLPRegressor":
        """Fit on ``(X, y)``, stopping on ``(X_dev, y_dev)``.

        The dev fold is only ever used to choose the epoch. It is the caller's
        job to make sure it is separated from ``X`` the same way the frozen
        splits separate train from val -- by whole peptide clusters.

        ``sample_weight`` is an optional non-negative per-row weight on the fit
        rows. Stage 2b uses it to give assumed-zero augmentation rows less pull
        than measured ones. It also weights the target offset, so adding a block
        of assumed zeros at weight 0.1 moves the centring by a tenth of what the
        same block at weight 1 would. Passing ``None`` or all-ones is the
        unweighted path, unchanged.
        """
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
        # Centre the target so a zero-initialised output bias already predicts
        # the mean. Added back in predict(); it is not a fitted parameter.
        self.y_offset_ = float(y.mean() if weight is None
                               else np.dot(weight, y) / weight.sum())
        y_centred = y - self.y_offset_
        self._init_params(X.shape[1], rng)

        opt_w = [_Adam.for_param(W) for W in self.weights_]
        opt_b = [_Adam.for_param(b) for b in self.biases_]

        started = time.perf_counter()
        best_mse, best_epoch, best_state = np.inf, 0, self._snapshot()
        self.dev_curve_ = []
        step = 0
        for epoch in range(1, cfg.max_epochs + 1):
            order = rng.permutation(len(X))
            for start in range(0, len(order), cfg.batch_size):
                idx = order[start:start + cfg.batch_size]
                pred, acts = self._forward(X[idx])
                gw, gb = self._backward(acts, pred - y_centred[idx],
                                        None if weight is None else weight[idx])
                step += 1
                for k in range(len(self.weights_)):
                    # L2 on weights only: penalising biases just fights the
                    # target offset.
                    opt_w[k].step(self.weights_[k], gw[k] + cfg.l2 * self.weights_[k],
                                  cfg.lr, step)
                    opt_b[k].step(self.biases_[k], gb[k], cfg.lr, step)

            dev_mse = float(np.mean((self.predict(X_dev) - y_dev) ** 2))
            self.dev_curve_.append(dev_mse)
            if dev_mse < best_mse:
                best_mse, best_epoch, best_state = dev_mse, epoch, self._snapshot()
            elif epoch - best_epoch >= cfg.patience:
                break

        self.weights_, self.biases_ = best_state
        self.n_epochs_ = epoch
        self.best_epoch_ = best_epoch
        self.dev_mse_ = best_mse
        self.fit_seconds_ = time.perf_counter() - started
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.weights_:
            raise RuntimeError("predict() before fit()")
        out, _ = self._forward(np.asarray(X, dtype=DTYPE))
        return out + self.y_offset_

    def n_parameters(self) -> int:
        return sum(W.size for W in self.weights_) + sum(b.size for b in self.biases_)
