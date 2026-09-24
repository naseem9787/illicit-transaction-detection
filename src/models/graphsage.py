"""Phase 5: a GraphSAGE variant of the basic GCN, for the controlled static
GNN improvement study (docs/EXPERIMENTS.md Phase 5).

Architecture, mirroring src/models/gcn.py's shape exactly so the two are a
fair controlled comparison — only the convolution operator differs:

    SAGEConv(in, hidden) -> ReLU -> Dropout(p) -> SAGEConv(hidden, 1)

Same forward(x, edge_index) -> (N,) raw-logit interface as GCN, so it is a
drop-in replacement for src/training/gnn_training.py::train_gcn, which is
architecture-agnostic despite its name (it only calls `model(x, edge_index)`
and never references GCN-specific internals).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x.squeeze(-1)  # (N,) raw logits
