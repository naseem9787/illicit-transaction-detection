"""Build per-time-step graph snapshots.

Because every edge in this dataset connects two nodes at the same time_step
(verified in DATA_AUDIT.md, and re-checked at load time in loader.validate_dataset),
the natural and leakage-safe graph representation is one independent snapshot
graph per time_step, rather than a single global graph.

Each snapshot is returned as plain numpy/python types with field names that
match torch_geometric.data.Data exactly (`x`, `edge_index`, `y`), so building
a Data object later is a direct `Data(**snapshot_to_torch(snapshot))` with no
remapping logic duplicated. This lets Phase 1 be fully verified without a
PyTorch installation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Snapshot:
    time_step: int
    node_ids: np.ndarray  # (N,) original txId, row order matches x/y/mask
    x: np.ndarray  # (N, F) float32 feature matrix
    y: np.ndarray  # (N,) int label, -1/0/1
    edge_index: np.ndarray  # (2, E) int64, LOCAL indices into node_ids
    is_labeled: np.ndarray  # (N,) bool


def build_snapshot(
    time_step: int,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    feature_cols: list[str],
) -> Snapshot:
    """Build one snapshot graph for a single time_step.

    `nodes` must already be filtered to this time_step by the caller (or be
    the full node table — this function filters internally either way).
    `edges` may be the full edge list; only edges with both endpoints in
    this time_step's node set are kept (should be all-or-nothing per the
    validated same-time-step property, but filtered defensively regardless).
    """
    node_rows = nodes[nodes["time_step"] == time_step].reset_index(drop=True)
    if node_rows.empty:
        raise ValueError(f"No nodes found for time_step={time_step}")

    node_ids = node_rows["txId"].to_numpy()
    id_to_local = {tx_id: i for i, tx_id in enumerate(node_ids)}

    x = node_rows[feature_cols].to_numpy(dtype=np.float32)
    y = node_rows["label"].to_numpy(dtype=np.int64)
    is_labeled = node_rows["is_labeled"].to_numpy(dtype=bool)

    snap_edges = edges[
        edges["txId1"].isin(id_to_local) & edges["txId2"].isin(id_to_local)
    ]
    src = snap_edges["txId1"].map(id_to_local).to_numpy(dtype=np.int64)
    dst = snap_edges["txId2"].map(id_to_local).to_numpy(dtype=np.int64)
    edge_index = np.vstack([src, dst]) if len(src) else np.zeros((2, 0), dtype=np.int64)

    return Snapshot(
        time_step=time_step,
        node_ids=node_ids,
        x=x,
        y=y,
        edge_index=edge_index,
        is_labeled=is_labeled,
    )


def build_all_snapshots(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    feature_cols: list[str],
    time_steps: list[int] | None = None,
) -> dict[int, Snapshot]:
    if time_steps is None:
        time_steps = sorted(nodes["time_step"].unique().tolist())
    snapshots = {}
    for t in time_steps:
        snap = build_snapshot(t, nodes, edges, feature_cols)
        snapshots[t] = snap
        logger.debug(
            "Built snapshot t=%d: %d nodes, %d edges, %d labeled",
            t,
            len(snap.node_ids),
            snap.edge_index.shape[1],
            int(snap.is_labeled.sum()),
        )
    logger.info("Built %d snapshots (time_steps %s)", len(snapshots), time_steps[:1] + ["..."] + time_steps[-1:])
    return snapshots


def snapshot_summary(snapshots: dict[int, Snapshot]) -> pd.DataFrame:
    rows = []
    for t, snap in sorted(snapshots.items()):
        n_illicit = int((snap.y == 1).sum())
        n_licit = int((snap.y == 0).sum())
        n_unknown = int((snap.y == -1).sum())
        rows.append(
            {
                "time_step": t,
                "num_nodes": len(snap.node_ids),
                "num_edges": snap.edge_index.shape[1],
                "num_illicit": n_illicit,
                "num_licit": n_licit,
                "num_unknown": n_unknown,
            }
        )
    return pd.DataFrame(rows)
