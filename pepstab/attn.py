"""Cross-attention head: let the groove condition the peptide, instead of
concatenating two frozen descriptions of it.

Stage 3's first arms hand the head ``[flatten(peptide embedding) ;
flatten(groove embedding)]`` and ask it to learn every peptide-position x
pocket-residue interaction in its first layer, from 15,773 rows. This module
builds the interaction into the architecture instead: the peptide's nine
per-position vectors are the queries, the allele's 34 contact-residue vectors
are the keys and values, and one attention block lets each peptide position
read the pocket residues that matter for *it*.

The ablation is the same module with :class:`CrossAttentionHead` constructed
with ``couple=False``, which replaces the attention weights by a uniform
average over the 34 contact residues. Layer sizes, folds, seeds and training
budget are then identical and the only difference between the two arms is
whether the groove is allowed to change how the peptide is read -- so a gap
belongs to coupling rather than to capacity. The query and key projections
(two d x d matrices) exist in both arms and are simply unused when
``couple=False``, which keeps the parameter count identical as well.

Torch rather than :mod:`pepstab.mlp` because attention needs autograd; the
training loop keeps mlp.py's contract -- caller-supplied dev fold, best-dev
weights restored, deterministic given the seed.
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
    """One point in the stage 3 coupling grid."""

    d_model: int = 64
    n_heads: int = 4
    hidden: int = 128
    dropout: float = 0.1
    lr: float = 1e-3
    l2: float = 1e-5
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

    Input is two blocks per row: ``pep`` of shape (batch, 9, d_in) and ``hla`` of
    shape (batch, 34, d_in). Both are projected to ``d_model`` by the *same*
    linear map in both arms.
    """

    def __init__(self, d_in: int, cfg: AttnConfig, n_pep: int = 9):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
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
            # this size, and the attention weights are only needed by
            # attention_map(), which recomputes them.
            ctx = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n_p, d)
        else:
            # Same information, no conditioning: every peptide position sees the
            # identical groove summary, so the groove cannot change how the
            # peptide is read.
            ctx = values.mean(dim=1, keepdim=True).expand(-1, p.shape[1], -1)
        return self.head(self.norm(p + ctx)).squeeze(-1)

    def attention_map(self, pep: torch.Tensor, hla: torch.Tensor) -> torch.Tensor:
        """Mean-over-heads (9, 34) coupling matrix, for interpretation."""
        if not self.cfg.couple:
            raise ValueError("the mean-pool ablation has no attention map")
        with torch.no_grad():
            p, h = self.pep_proj(pep), self.hla_proj(hla)
            b, n_p, d = p.shape
            heads, dh = self.cfg.n_heads, d // self.cfg.n_heads
            q = self.q(p).view(b, n_p, heads, dh).transpose(1, 2)
            k = self.k(h).view(b, h.shape[1], heads, dh).transpose(1, 2)
            return torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(dh), dim=-1).mean(1)


def fit(model: CrossAttentionHead, pep: np.ndarray, hla: np.ndarray, y: np.ndarray,
        pep_dev: np.ndarray, hla_dev: np.ndarray, y_dev: np.ndarray) -> dict:
    """Train on ``(pep, hla, y)``, stop on the caller-supplied dev fold.

    The dev fold must be separated from the fit rows by whole peptide clusters,
    exactly as in :mod:`pepstab.mlp`; that is the caller's job.
    """
    cfg = model.cfg
    torch.manual_seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.l2)
    loss_fn = nn.MSELoss()
    offset = float(y.mean())

    tp, th = torch.from_numpy(pep), torch.from_numpy(hla)
    ty = torch.from_numpy((y - offset).astype(np.float32))
    dp, dh = torch.from_numpy(pep_dev), torch.from_numpy(hla_dev)
    dy = torch.from_numpy((y_dev - offset).astype(np.float32))

    rng = np.random.default_rng(cfg.seed)
    best, best_epoch, best_state = np.inf, 0, None
    started = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = rng.permutation(len(ty))
        for start in range(0, len(order), cfg.batch_size):
            idx = torch.from_numpy(order[start:start + cfg.batch_size])
            opt.zero_grad()
            loss_fn(model(tp[idx], th[idx]), ty[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            dev = float(loss_fn(model(dp, dh), dy))
        if dev < best:
            best, best_epoch = dev, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= cfg.patience:
            break
    model.load_state_dict(best_state)
    return {"offset": offset, "best_epoch": best_epoch, "dev_mse": best,
            "n_epochs": epoch, "fit_seconds": time.perf_counter() - started,
            "n_parameters": sum(p.numel() for p in model.parameters())}


def predict(model: CrossAttentionHead, pep: np.ndarray, hla: np.ndarray,
            offset: float, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(pep), batch_size):
            out.append(model(torch.from_numpy(pep[start:start + batch_size]),
                             torch.from_numpy(hla[start:start + batch_size])).numpy())
    return np.concatenate(out) + offset
