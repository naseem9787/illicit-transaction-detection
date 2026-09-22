"""Training/evaluation helpers for the Phase 3 basic GCN.

Mirrors the shape of src/baselines/*.py (plain functions the scripts call)
rather than a generic trainer class, to match the project's existing
style and keep this easy to read end to end.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch_geometric.data import Batch

from src.data.graph_builder import build_all_snapshots
from src.data.pyg_adapter import snapshot_to_data
from src.models.gcn import GCN

logger = logging.getLogger(__name__)


def build_split_batches(
    node_table: pd.DataFrame,
    edges: pd.DataFrame,
    feature_cols: list[str],
    split_cfg: dict,
) -> dict[str, Batch]:
    """Build one block-diagonal Batch per split (train/val/test).

    Each Batch is a disjoint union of that split's per-time-step
    snapshots (see src/data/pyg_adapter.py docstring for why this is
    equivalent to processing snapshots one at a time). Time steps outside
    all three ranges ("excluded") are not built.
    """
    ranges = {
        "train": (split_cfg["train_start"], split_cfg["train_end"]),
        "val": (split_cfg["val_start"], split_cfg["val_end"]),
        "test": (split_cfg["test_start"], split_cfg["test_end"]),
    }
    all_time_steps = sorted(
        {t for start, end in ranges.values() for t in range(start, end + 1)}
    )
    snapshots = build_all_snapshots(node_table, edges, feature_cols, time_steps=all_time_steps)

    batches = {}
    for split_name, (start, end) in ranges.items():
        data_list = [snapshot_to_data(snapshots[t]) for t in range(start, end + 1)]
        batches[split_name] = Batch.from_data_list(data_list)
        n_nodes = batches[split_name].num_nodes
        n_labeled = int(batches[split_name].is_labeled.sum())
        logger.info(
            "%s batch: %d snapshots, %d nodes, %d labeled",
            split_name, len(data_list), n_nodes, n_labeled,
        )
    return batches


def compute_pos_weight(y_train: np.ndarray) -> float:
    """BCEWithLogitsLoss pos_weight = n_negative / n_positive on TRAIN labels only.

    This is the standard PyTorch idiom for binary class imbalance: it
    scales the positive class's loss contribution so both classes
    contribute comparably in expectation, in the same spirit as sklearn's
    class_weight="balanced" used for the Logistic Regression / Random
    Forest baselines (src/baselines/*.py) — but not numerically identical
    to sklearn's exact formula, so the two are analogous, not equivalent.
    Computed from the training split only; never touches val/test.
    """
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    return n_neg / n_pos


@dataclass
class TrainResult:
    best_state_dict: dict
    best_epoch: int
    best_val_pr_auc: float
    epoch_log: list[dict]
    train_time_seconds: float


def train_gcn(
    model: GCN,
    train_batch: Batch,
    val_batch: Batch,
    seed: int,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
) -> TrainResult:
    """Full-batch training over all training snapshots per epoch.

    Loss is masked to `is_labeled` nodes only (unknown-labeled nodes still
    participate in the GCNConv message passing forward pass — they are
    real graph neighbors — but never contribute a gradient). Model
    checkpoint is selected by best validation PR-AUC (never test), the
    same imbalance-appropriate ranking metric used for Random Forest
    config selection (src/baselines/random_forest.py, docs/DECISIONS.md
    D9), so model selection is validation-only across the whole project.
    """
    import time

    torch.manual_seed(seed)

    y_train = train_batch.y[train_batch.is_labeled].numpy()
    pos_weight = torch.tensor(compute_pos_weight(y_train), dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_mask = train_batch.is_labeled
    train_y = train_batch.y[train_mask].float()
    val_mask = val_batch.is_labeled
    val_y = val_batch.y[val_mask].numpy()

    best_state_dict = None
    best_epoch = -1
    best_val_pr_auc = -1.0
    epoch_log = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(train_batch.x, train_batch.edge_index)
        loss = criterion(logits[train_mask], train_y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_batch.x, val_batch.edge_index)
            val_prob = torch.sigmoid(val_logits[val_mask]).numpy()
        val_pr_auc = float(average_precision_score(val_y, val_prob))

        epoch_log.append({"epoch": epoch, "train_loss": float(loss.item()), "val_pr_auc": val_pr_auc})
        if epoch % 20 == 0 or epoch == 1:
            logger.info("epoch %d: train_loss=%.4f val_pr_auc=%.4f", epoch, loss.item(), val_pr_auc)

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

    train_time = time.time() - t0
    logger.info(
        "Training complete in %.2fs. Best val PR-AUC=%.4f at epoch %d",
        train_time, best_val_pr_auc, best_epoch,
    )
    return TrainResult(
        best_state_dict=best_state_dict,
        best_epoch=best_epoch,
        best_val_pr_auc=best_val_pr_auc,
        epoch_log=epoch_log,
        train_time_seconds=train_time,
    )


def predict_labeled(model: GCN, batch: Batch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, y_prob, time_step) for the labeled nodes of a batch."""
    model.eval()
    with torch.no_grad():
        logits = model(batch.x, batch.edge_index)
        prob = torch.sigmoid(logits)
    mask = batch.is_labeled
    y_true = batch.y[mask].numpy()
    y_prob = prob[mask].numpy()
    time_step = batch.time_step[mask].numpy()
    return y_true, y_prob, time_step
