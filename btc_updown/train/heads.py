"""DirectHead: a deterministic P(settles Up) readout on top of Kronos's hidden state.

Given the backbone's hidden state at the decision bar t (the last lookback bar), a horizon h
(bars to expiry) and a strike K (typical price of the bar that closed when the market window
opened, j bars ago), the head predicts the logit of ``1[TP(t+h) >= K]``.  It is trained with
BCE on labels computed from the klines alone, so it needs no market data, and at inference it
costs one backbone forward per decision instead of N sampled paths.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectHead(nn.Module):
    N_FEATS = 5

    def __init__(self, d_model: int, hidden: int = 256, max_h: int = 15, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden
        self.max_h = max_h
        self.dropout_p = float(dropout)
        self.proj_ctx = nn.Linear(d_model, hidden)
        self.emb_h = nn.Embedding(max_h + 1, hidden)
        self.proj_feats = nn.Linear(self.N_FEATS, hidden)
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, ctx: torch.Tensor, h: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """ctx [B, d_model]; h [P] or [B, P] int; feats [B, P, 5]  ->  logits [B, P]."""
        if h.dim() == 1:
            h = h[None, :].expand(feats.shape[0], -1)
        z = self.proj_ctx(ctx.float())[:, None, :] + self.emb_h(h.long()) + self.proj_feats(feats.float())
        return self.mlp(z).squeeze(-1)

    @staticmethod
    def loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        l = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        if mask is not None:
            l = l * mask
            return l.sum() / mask.sum().clamp_min(1.0)
        return l.mean()


def brier(prob: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return ((prob - labels) ** 2).mean()
