"""Cross-attention head: let the groove condition the peptide, instead of
concatenating two frozen descriptions of it.

Stage 3's shipped arms hand the head ``[flatten(peptide embedding) ;
flatten(groove embedding)]`` and ask it to learn every peptide-position x
pocket-residue interaction in its first layer, from 19,716 rows. That is the
one explanation ``reports/stage3_esm.md`` §9 leaves open for the null: not that
frozen ESM-2 carries nothing useful, but that concatenation cannot use what it
carries. This module builds the interaction into the architecture instead --
the peptide's nine per-position vectors are the queries, the allele's 34
contact-residue vectors are the keys and values, and one attention block lets
each peptide position read the pocket residues that matter for *it*.

**The control is the same module with ``couple=False``**, which replaces the
attention weights by a uniform average over the 34 contact residues. Layer
sizes, folds, seeds, tuning budget and training budget are then identical, and
so is the parameter count: the query and key projections exist in both arms and
are simply unused in the ablation (their gradients are ``None``, so Adam skips
them and they keep their initial values). The only difference between the arms
is whether the groove is allowed to change how the peptide is read, so a gap
between them belongs to coupling rather than to capacity.

Torch rather than :mod:`pepstab.mlp` because attention needs autograd. The
training loop keeps mlp.py's contract -- caller-supplied dev fold, best-dev
weights restored, L2 on weight matrices but not on biases or LayerNorm gains,
and determinism given ``seed``.

Adapted from the ``stage3-esm`` branch, which wrote the module and the
parameter-identity control but never ran either. Three changes were needed
before it could be trusted:

1. **The model was not seeded.** ``torch.manual_seed`` was called inside
   ``fit``, after the caller had already constructed the module, so the
   initialisation came from whatever ambient RNG state the process happened to
   be in -- two "seed 0" members could differ, and a rerun could not be
   reproduced. Seeding now happens in ``__init__``.
2. **Rows were fanned out before training.** The driver materialised one
   ``(34, d)`` groove block per measurement row, which for the HLA side is
   19,716 copies of 75 distinct domains -- 171 MB of pure duplication on a
   machine sharing 16 GB with a live structural fold. Gathering inside the
   batch instead costs one index op and ~4x less memory.
3. **L2 was applied through Adam's ``weight_decay``**, which also decays biases
   and LayerNorm gains. mlp.py deliberately does not, and the two arms have to
   be regularised the same way to be compared with it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AttnConfig:
    """One point in the coupling grid.

    ``l2`` has **no default worth trusting**: the whole lesson of
    ``reports/stage3_esm.md`` §2 is that a regularisation strength selected for
    one feature geometry is a handicap on another, so the caller is expected to
    select it on a ladder of its own and record whether the selection was
    interior. The value here is a placeholder for smoke tests.
    """

    d_model: int = 64
    n_heads: int = 4
    hidden: int = 128
    dropout: float = 0.1
    lr: float = 1e-3
    l2: float = 1e-3
    batch_size: int = 1024
    max_epochs: int = 300
    patience: int = 15
    seed: int = 0
    couple: bool = True

    def label(self) -> str:
        kind = "xattn" if self.couple else "meanpool"
        return f"{kind}_d{self.d_model}h{self.n_heads}_f{self.hidden}_l2{self.l2:g}"


class CrossAttentionHead(nn.Module):
    """Peptide positions attend over HLA contact residues, then a small MLP.

    ``forward`` takes ``pep`` of shape ``(batch, 9, d_in)`` and ``hla`` of shape
    ``(batch, 34, d_in)``. Both are projected to ``d_model`` by the same linear
    map in both arms.
    """

    def __init__(self, d_in: int, cfg: AttnConfig, n_pep: int = 9):
        super().__init__()
        # Seed before any parameter exists, so initialisation is part of what
        # the seed determines. The branch seeded in fit(), by which point every
        # weight had already been drawn.
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        d = cfg.d_model
        if d % cfg.n_heads:
            raise ValueError(f"d_model {d} is not divisible by n_heads {cfg.n_heads}")
        self.pep_proj = nn.Linear(d_in, d)
        self.hla_proj = nn.Linear(d_in, d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_pep * d, cfg.hidden), nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, 1),
        )

    def forward(self, pep: torch.Tensor, hla: torch.Tensor) -> torch.Tensor:
        p = self.pep_proj(pep)                      # (B, 9, d)
        h = self.hla_proj(hla)                      # (B, 34, d)
        values = self.v(h)
        if self.cfg.couple:
            b, n_p, d = p.shape
            heads, dh = self.cfg.n_heads, d // self.cfg.n_heads
            q = self.q(p).view(b, n_p, heads, dh).transpose(1, 2)
            k = self.k(h).view(b, h.shape[1], heads, dh).transpose(1, 2)
            v = values.view(b, h.shape[1], heads, dh).transpose(1, 2)
            # Fused kernel: the explicit softmax form is ~5x slower on CPU at
            # this size, and the weights themselves are only needed by
            # attention_map(), which recomputes them.
            ctx = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n_p, d)
        else:
            # Same information, no conditioning: every peptide position sees an
            # identical groove summary, so the groove cannot change how the
            # peptide is read.
            ctx = values.mean(dim=1, keepdim=True).expand(-1, p.shape[1], -1)
        return self.head(self.norm(p + ctx)).squeeze(-1)

    def attention_map(self, pep: torch.Tensor, hla: torch.Tensor) -> torch.Tensor:
        """Mean-over-heads ``(batch, 9, 34)`` coupling matrix, for interpretation."""
        if not self.cfg.couple:
            raise ValueError("the mean-pool ablation has no attention map")
        with torch.no_grad():
            p, h = self.pep_proj(pep), self.hla_proj(hla)
            b, n_p, d = p.shape
            heads, dh = self.cfg.n_heads, d // self.cfg.n_heads
            q = self.q(p).view(b, n_p, heads, dh).transpose(1, 2)
            k = self.k(h).view(b, h.shape[1], heads, dh).transpose(1, 2)
            return torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(dh), dim=-1).mean(1)

    def param_groups(self) -> list[dict]:
        """Adam groups applying ``l2`` to weight matrices only.

        mlp.py penalises weights and not biases, on the grounds that shrinking a
        bias just fights the target offset. The same split is kept here, and
        LayerNorm gains are excluded for the same reason -- decaying a gain
        toward zero is a capacity change, not a smoothness prior. Passing
        ``weight_decay`` to the optimiser wholesale, as the branch did, would
        regularise all three.
        """
        decay, free = [], []
        for name, p in self.named_parameters():
            (free if name.endswith("bias") or "norm" in name else decay).append(p)
        return [{"params": decay, "weight_decay": self.cfg.l2},
                {"params": free, "weight_decay": 0.0}]


class Bank:
    """Unique per-sequence residue blocks plus the row -> block index.

    The point is that 19,716 training rows carry only 5,633 distinct peptides
    and 75 distinct HLA domains, so fanning the blocks out to rows before
    training stores the same 75 groove matrices thousands of times over. The
    blocks stay deduplicated here and a batch gathers what it needs.
    """

    def __init__(self, blocks: np.ndarray, index: np.ndarray):
        self.blocks = torch.from_numpy(np.ascontiguousarray(blocks, dtype=np.float32))
        self.index = torch.from_numpy(np.asarray(index, dtype=np.int64))
        if int(self.index.max()) >= len(self.blocks):
            raise ValueError("index points past the end of the block bank")

    def __len__(self) -> int:
        return len(self.index)

    def gather(self, rows: torch.Tensor) -> torch.Tensor:
        return self.blocks[self.index[rows]]

    def subset(self, mask: np.ndarray) -> "Bank":
        """A view over a row subset, sharing the same deduplicated blocks."""
        out = Bank.__new__(Bank)
        out.blocks = self.blocks
        out.index = self.index[torch.from_numpy(np.asarray(mask))]
        return out

    @property
    def d_in(self) -> int:
        return int(self.blocks.shape[-1])


def _dev_mse(model: CrossAttentionHead, pep: Bank, hla: Bank, y: torch.Tensor,
             batch_size: int) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            rows = torch.arange(start, min(start + batch_size, len(y)))
            total += float(((model(pep.gather(rows), hla.gather(rows))
                             - y[rows]) ** 2).sum())
    return total / len(y)


def fit(model: CrossAttentionHead, pep: Bank, hla: Bank, y: np.ndarray,
        pep_dev: Bank, hla_dev: Bank, y_dev: np.ndarray) -> dict:
    """Train on ``(pep, hla, y)``, stopping on the caller-supplied dev fold.

    The dev fold must be separated from the fit rows by whole peptide clusters,
    exactly as in :mod:`pepstab.mlp`; that is the caller's job, and
    ``scripts.baseline_ensemble.cv_folds`` is the function that does it.
    """
    cfg = model.cfg
    opt = torch.optim.Adam(model.param_groups(), lr=cfg.lr)
    offset = float(np.mean(y))
    ty = torch.from_numpy((np.asarray(y) - offset).astype(np.float32))
    tdy = torch.from_numpy((np.asarray(y_dev) - offset).astype(np.float32))

    rng = np.random.default_rng(cfg.seed)
    best, best_epoch, best_state = np.inf, 0, None
    started = time.perf_counter()
    epoch = 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = rng.permutation(len(ty))
        for start in range(0, len(order), cfg.batch_size):
            rows = torch.from_numpy(order[start:start + cfg.batch_size])
            opt.zero_grad()
            loss = ((model(pep.gather(rows), hla.gather(rows)) - ty[rows]) ** 2).mean()
            loss.backward()
            opt.step()
        dev = _dev_mse(model, pep_dev, hla_dev, tdy, cfg.batch_size)
        if dev < best:
            best, best_epoch = dev, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= cfg.patience:
            break
    model.load_state_dict(best_state)
    return {"offset": offset, "best_epoch": best_epoch, "dev_mse": best,
            "n_epochs": epoch, "fit_seconds": time.perf_counter() - started,
            "n_parameters": sum(p.numel() for p in model.parameters())}


def predict(model: CrossAttentionHead, pep: Bank, hla: Bank, offset: float,
            batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(pep), batch_size):
            rows = torch.arange(start, min(start + batch_size, len(pep)))
            out.append(model(pep.gather(rows), hla.gather(rows)).numpy())
    return np.concatenate(out) + offset
