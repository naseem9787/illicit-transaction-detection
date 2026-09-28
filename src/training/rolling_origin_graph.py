"""Phase 8 (E3): rolling-origin validation for graph-based models
(plain GraphSAGE, symmetrized-edge GraphSAGE, direction-aware GraphSAGE).

Reuses ROLLING_ORIGIN_FOLDS / TEST_RANGE from rolling_origin.py (E1/E2) as
the single source of truth for fold boundaries, so E3 uses exactly the same
three folds. Reuses src/training/gnn_training.py::train_gcn and
predict_labeled UNCHANGED — that function only ever calls
`model(x, edge_index)` and never references GCN-specific internals, so it
works for GraphSAGE and both E3 variants without modification (already
established in Phase 5/6/7).

`build_graph_batches_for_range` is the rolling-origin analogue of
src/training/gnn_training.py::build_split_batches, which builds one Batch
per FIXED split name (train/val/test). This instead builds one Batch for an
explicit inclusive time_step range, so arbitrary fold boundaries can be
expressed — mirrors src/training/rolling_origin.py::get_labeled_by_time_range's
role for the tabular (E1/E2) models.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch_geometric.data import Batch

from src.data.graph_builder import build_all_snapshots
from src.data.pyg_adapter import snapshot_to_data
from src.evaluation.metrics import compute_binary_metrics, select_threshold_maximizing_f1
from src.models.graphsage import GraphSAGE
from src.models.graphsage_directional import DirectionAwareGraphSAGE, SymmetrizedGraphSAGE
from src.training.gnn_training import predict_labeled, train_gcn
from src.training.rolling_origin import ROLLING_ORIGIN_FOLDS, TEST_RANGE

logger = logging.getLogger(__name__)

E3_SEEDS = [42, 123, 456]
E3_HIDDEN = 128
E3_DROPOUT = 0.5
E3_LR = 0.01
E3_WEIGHT_DECAY = 5e-4
E3_EPOCHS = 200  # same budget as the Phase 5 GraphSAGE grid

E3_ARCHITECTURES = {
    "plain_graphsage": GraphSAGE,  # unmodified Phase 5 model — the control arm
    "symmetrized_graphsage": SymmetrizedGraphSAGE,
    "direction_aware_graphsage": DirectionAwareGraphSAGE,
}


def build_graph_batches_for_range(
    node_table: pd.DataFrame, edges: pd.DataFrame, feature_cols: list[str], start: int, end: int
) -> Batch:
    """One block-diagonal Batch for time_step in [start, end] (inclusive)."""
    if end >= TEST_RANGE[0]:
        raise ValueError(f"range [{start}, {end}] reaches into the test period {TEST_RANGE} — refusing")
    time_steps = list(range(start, end + 1))
    snapshots = build_all_snapshots(node_table, edges, feature_cols, time_steps=time_steps)
    data_list = [snapshot_to_data(snapshots[t]) for t in time_steps]
    return Batch.from_data_list(data_list)


@dataclass
class E3Result:
    architecture: str
    fold: str
    seed: int
    best_epoch: int
    val_pr_auc: float
    val_f1: float
    val_threshold: float
    train_time_seconds: float


def train_one(architecture: str, in_channels: int, train_batch: Batch, val_batch: Batch, seed: int) -> E3Result:
    model_cls = E3_ARCHITECTURES[architecture]
    torch.manual_seed(seed)
    model = model_cls(in_channels=in_channels, hidden_channels=E3_HIDDEN, dropout=E3_DROPOUT)
    result = train_gcn(model, train_batch, val_batch, seed=seed, epochs=E3_EPOCHS, lr=E3_LR, weight_decay=E3_WEIGHT_DECAY)
    model.load_state_dict(result.best_state_dict)

    y_val, p_val, _ = predict_labeled(model, val_batch)
    threshold = select_threshold_maximizing_f1(y_val, p_val)  # validation only
    m = compute_binary_metrics(y_val, p_val, threshold)

    return E3Result(
        architecture=architecture, fold="", seed=seed, best_epoch=result.best_epoch,
        val_pr_auc=result.best_val_pr_auc, val_f1=m["f1"], val_threshold=threshold,
        train_time_seconds=round(result.train_time_seconds, 2),
    )


def run_e3(node_table: pd.DataFrame, edges: pd.DataFrame, feature_cols: list[str]) -> list[dict]:
    in_channels = len(feature_cols)
    rows = []
    for fold in ROLLING_ORIGIN_FOLDS:
        train_batch = build_graph_batches_for_range(node_table, edges, feature_cols, fold["train_start"], fold["train_end"])
        val_batch = build_graph_batches_for_range(node_table, edges, feature_cols, fold["val_start"], fold["val_end"])
        for architecture in E3_ARCHITECTURES:
            for seed in E3_SEEDS:
                t0 = time.time()
                r = train_one(architecture, in_channels, train_batch, val_batch, seed)
                r.fold = fold["name"]
                wall = time.time() - t0
                logger.info("[E3][%s][%s][seed=%d] val_pr_auc=%.4f val_f1=%.4f best_epoch=%d (%.1fs)",
                            architecture, fold["name"], seed, r.val_pr_auc, r.val_f1, r.best_epoch, wall)
                rows.append({
                    "architecture": architecture, "fold": fold["name"], "seed": seed,
                    "n_train_nodes": int(train_batch.num_nodes), "n_train_labeled": int(train_batch.is_labeled.sum()),
                    "n_val_nodes": int(val_batch.num_nodes), "n_val_labeled": int(val_batch.is_labeled.sum()),
                    "best_epoch": r.best_epoch, "val_pr_auc": r.val_pr_auc, "val_f1": r.val_f1,
                    "val_threshold": r.val_threshold, "train_time_seconds": r.train_time_seconds,
                })
    return rows
