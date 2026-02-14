"""CTR prediction head for recommendation models."""

from __future__ import annotations

import torch
from torch import nn


class CTRHead(nn.Module):
    """Project hidden states to click-through-rate logits."""

    def __init__(self, hidden_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return click logits with shape [..., 1]."""
        return self.proj(self.dropout(hidden_states))
