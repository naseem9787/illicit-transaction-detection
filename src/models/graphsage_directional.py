"""Phase 8 (E3): direction-aware graph aggregation for GraphSAGE.

The Phase 5 GraphSAGE (src/models/graphsage.py) uses `edge_index` exactly as
supplied by the dataset. PyG's SAGEConv aggregates from src -> dst (the
"flow" direction), so for a node acting as dst it only ever sees messages
from nodes that point TO it — its "incoming" neighbors. Its "outgoing"
neighbors (nodes it points to) never contribute a message under the raw
edge orientation. This module tests whether using both directions helps.

Two variants, both built from the same SAGEConv building block and both
matching the Phase 5 baseline's forward(x, edge_index) -> (N,) logits
interface exactly, so they plug into the existing, unmodified
src/training/gnn_training.py::train_gcn / predict_labeled without any
change to that code.

Both variants operate purely on `x` (f1..f165, never labels) and
`edge_index` as given for the current (already block-diagonal-batched)
snapshot set. Symmetrizing/reversing edges never creates a new edge between
two different original snapshots — flipping an edge (u, v) into (v, u)
does not change which snapshot u or v belongs to, since batching already
guarantees u and v share a time_step for every input edge (see
src/data/pyg_adapter.py). Verified directly in
tests/test_phase8_e3_direction_aware.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


def symmetrize(edge_index: torch.Tensor) -> torch.Tensor:
    """Undirected view of a directed edge_index: union of the edges and
    their reverse. The dataset has zero mutual pairs (verified in
    docs/DATA_AUDIT.md — no edge's reverse is already present), so this
    never double-counts an existing edge as a duplicate."""
    if edge_index.numel() == 0:
        return edge_index
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


class SymmetrizedGraphSAGE(torch.nn.Module):
    """Variant 1: identical architecture to the Phase 5 GraphSAGE
    (SAGEConv(165,128) -> ReLU -> Dropout(0.5) -> SAGEConv(128,1)), except
    both conv layers see the symmetrized (undirected) edge set instead of
    the raw directed one."""

    def __init__(self, in_channels: int, hidden_channels: int = 128, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        sym_edge_index = symmetrize(edge_index)
        x = self.conv1(x, sym_edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, sym_edge_index)
        return x.squeeze(-1)


class DirectionAwareGraphSAGE(torch.nn.Module):
    """Variant 2: separate first-layer SAGEConv branches for incoming
    (raw edge_index, src->dst) and outgoing (reversed edge_index) neighbors,
    each producing hidden_channels // 2 dimensions so the concatenated
    hidden representation matches the 128-dim baseline width for a
    controlled comparison (rather than doubling it to 256). The output
    layer aggregates over the symmetrized edge set, same as variant 1."""

    def __init__(self, in_channels: int, hidden_channels: int = 128, dropout: float = 0.5):
        super().__init__()
        assert hidden_channels % 2 == 0, "hidden_channels must be even to split in/out evenly"
        half = hidden_channels // 2
        self.conv_in = SAGEConv(in_channels, half)
        self.conv_out = SAGEConv(in_channels, half)
        self.conv2 = SAGEConv(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h_in = self.conv_in(x, edge_index)  # aggregates from incoming neighbors (as supplied)
        h_out = self.conv_out(x, edge_index.flip(0))  # aggregates from outgoing neighbors (reversed)
        h = torch.cat([h_in, h_out], dim=-1)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        sym_edge_index = symmetrize(edge_index)
        out = self.conv2(h, sym_edge_index)
        return out.squeeze(-1)
