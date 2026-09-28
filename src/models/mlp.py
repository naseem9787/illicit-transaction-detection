"""Phase 8 (E1): feature-only MLP baseline.

Same 165 input features as Random Forest / Logistic Regression, no graph
structure at all. Exists to separate two possible explanations for why the
GNNs (Phase 3-7) trail Random Forest: "neural networks are worse than trees
on this tabular data" vs "the graph structure itself isn't helping". This
model answers the first question in isolation.

Architecture: 165 -> 128 -> 1, ReLU, Dropout(0.5), matching the fixed shape
requested for E1. Same forward(x) -> (N,) raw-logit convention as the GCN/
GraphSAGE models, so the same BCEWithLogitsLoss + pos_weight + threshold
code paths apply unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, in_channels: int = 165, hidden_channels: int = 128, dropout: float = 0.5):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x.squeeze(-1)  # (N,) raw logits
