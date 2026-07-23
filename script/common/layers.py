"""Shared neural building blocks used by PAVR branches."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Two-layer MLP projection followed by L2 normalization.

    Used to map pooled branch vectors into a unit-sphere embedding space before
    concatenation into the supervised probe.
    """

    def __init__(self, input_dim: int, projection_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = max(input_dim, projection_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project ``[B, input_dim]`` to L2-normalized ``[B, projection_dim]``."""
        return F.normalize(self.net(x), dim=-1)


def masked_mean_pool(x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean-pool a sequence tensor while ignoring padded positions.

    Parameters
    ----------
    x:
        Token features of shape ``[B, L, D]``.
    padding_mask:
        Optional boolean mask ``[B, L]`` where ``True`` marks padding.

    Returns
    -------
    torch.Tensor
        Pooled features of shape ``[B, D]``.
    """
    if padding_mask is None:
        return x.mean(dim=1)
    if padding_mask.shape != x.shape[:2]:
        raise ValueError(f"padding_mask must be [B, L], got {tuple(padding_mask.shape)} for {tuple(x.shape)}")
    valid = (~padding_mask.to(device=x.device, dtype=torch.bool)).unsqueeze(-1).to(dtype=x.dtype)
    denom = valid.sum(dim=1).clamp_min(1.0)
    return (x * valid).sum(dim=1) / denom


def _masked_logits(logits: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    """Apply a keep-mask to attention logits.

    Positions with ``keep_mask == False`` are set to ``-inf``. If an entire row is
    masked out, the mask falls back to all-ones so softmax remains defined.
    """
    if keep_mask.shape != logits.shape:
        raise ValueError(f"keep_mask must match logits shape, got {tuple(keep_mask.shape)} and {tuple(logits.shape)}")
    keep_mask = keep_mask.to(device=logits.device, dtype=torch.bool)
    has_any = keep_mask.any(dim=1, keepdim=True)
    keep_mask = torch.where(has_any, keep_mask, torch.ones_like(keep_mask))
    return logits.masked_fill(~keep_mask, float("-inf"))


class GlobalAttentionPool(nn.Module):
    """Additive (Bahdanau-style) attention pooling over a token sequence.

    Optionally accepts a per-token ``logit_bias`` (e.g. soft-window log-mass) that
    is added to the raw attention scores before softmax.
    """

    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        logit_bias: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Pool ``x`` of shape ``[B, L, D]``.

        Returns
        -------
        dict
            ``pooled`` of shape ``[B, D]`` and ``attention`` of shape ``[B, L]``.
        """
        logits = self.score(x).squeeze(-1)
        if logit_bias is not None:
            if logit_bias.shape != logits.shape:
                raise ValueError(
                    f"logit_bias must be [B, L], got {tuple(logit_bias.shape)} for logits {tuple(logits.shape)}"
                )
            logits = logits + logit_bias.to(device=logits.device, dtype=logits.dtype)
        if padding_mask is not None:
            if padding_mask.shape != logits.shape:
                raise ValueError(
                    f"padding_mask must be [B, L], got {tuple(padding_mask.shape)} for logits {tuple(logits.shape)}"
                )
            logits = logits.masked_fill(padding_mask.to(device=x.device, dtype=torch.bool), float("-inf"))
        attention = torch.softmax(logits, dim=-1)
        pooled = (x * attention.unsqueeze(-1)).sum(dim=1)
        return {"pooled": pooled, "attention": attention}
