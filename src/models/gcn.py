"""Basic GNN baseline (Phase 3): a plain 2-layer GCN.

Architecture, fixed and deliberately simple (not tuned — see
docs/DECISIONS.md D9 for the same "don't over-tune" reasoning applied to
Random Forest):

    GCNConv(in, hidden) -> ReLU -> Dropout(p) -> GCNConv(hidden, 1)

Outputs a single raw logit per node (probability of "illicit" is
sigmoid(logit)), analogous to `model.predict_proba(X)[:, 1]` in the
sklearn baselines, so the same threshold-selection and metrics code in
src/evaluation/ can be reused unchanged. Uses only node features and
edge_index — no edge features, no residual/skip connections, no
normalization beyond what GCNConv already does internally.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x.squeeze(-1)  # (N,) raw logits
