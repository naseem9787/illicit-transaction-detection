"""Convert graph_builder.Snapshot objects into torch_geometric Data objects.

Kept separate from src/data/graph_builder.py (rather than adding a torch
import there) so Phase 1's pipeline and tests stay fully importable and
runnable without a PyTorch installation, per graph_builder.py's module
docstring and docs/DECISIONS.md D6. graph_builder.Snapshot's field names
(`x`, `edge_index`, `y`) already match torch_geometric.data.Data exactly,
so this conversion is a direct field-for-field wrap with no remapping.

`torch_geometric.data.Batch.from_data_list` is the mechanism used
downstream (src/training/gnn_training.py) to combine several snapshots
into one forward pass. It is a plain block-diagonal stack: edge_index
values are offset per graph and no edges are ever added between graphs,
so batching several snapshots together is mathematically identical to
running each snapshot through the model separately (verified directly:
see tests/test_gnn.py::test_batch_is_block_diagonal). This preserves the
"one independent graph per time_step, never a merged temporal graph"
property established in docs/DECISIONS.md D3.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from src.data.graph_builder import Snapshot


def snapshot_to_data(snapshot: Snapshot) -> Data:
    """Wrap one Snapshot as a torch_geometric Data object.

    Extra per-node attributes (`is_labeled`, `time_step`) ride along and
    are concatenated correctly by Batch.from_data_list, so predictions can
    be traced back to the original split/time_step after batching.
    """
    # .copy(): the feature/label arrays backing a Snapshot can be non-writable
    # views into the cached node_features.npz array; torch.from_numpy requires
    # a writable buffer.
    n = len(snapshot.node_ids)
    return Data(
        x=torch.from_numpy(snapshot.x.copy()),
        edge_index=torch.from_numpy(snapshot.edge_index.copy()),
        y=torch.from_numpy(snapshot.y.copy()),
        is_labeled=torch.from_numpy(snapshot.is_labeled.copy()),
        time_step=torch.full((n,), snapshot.time_step, dtype=torch.long),
    )
