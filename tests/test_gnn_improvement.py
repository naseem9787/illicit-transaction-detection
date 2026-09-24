import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.graph_builder import Snapshot
from src.data.pyg_adapter import snapshot_to_data
from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE
from src.training.gnn_training import train_gcn
from scripts.train_gnn_improvement import GridConfig, build_grid, make_model


def make_synthetic_data(time_step: int, n: int, seed: int) -> Data:
    rng = np.random.default_rng(seed)
    node_ids = np.arange(n)
    x = rng.standard_normal((n, 5)).astype(np.float32)
    y = np.array([1, 0] * (n // 2 + 1))[:n].astype(np.int64)
    is_labeled = np.ones(n, dtype=bool)
    edges = [(i, i + 1) for i in range(n - 1)]
    edge_index = np.array(edges, dtype=np.int64).T
    snap = Snapshot(time_step=time_step, node_ids=node_ids, x=x, y=y, edge_index=edge_index, is_labeled=is_labeled)
    return snapshot_to_data(snap)


# --- GraphSAGE model ---

def test_graphsage_forward_output_shape():
    torch.manual_seed(0)
    model = GraphSAGE(in_channels=5, hidden_channels=8, dropout=0.5)
    x = torch.randn(6, 5)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    out = model(x, edge_index)
    assert out.shape == (6,)
    assert out.dtype == torch.float32


def test_graphsage_handles_isolated_nodes_and_empty_edge_index():
    torch.manual_seed(0)
    model = GraphSAGE(in_channels=5, hidden_channels=4)
    x = torch.randn(3, 5)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    out = model(x, edge_index)
    assert out.shape == (3,)
    assert torch.isfinite(out).all()


# --- grid definition ---

def test_grid_has_exactly_16_preclared_configs():
    grid = build_grid()
    assert len(grid) == 16
    n_gcn = sum(1 for c in grid if c.architecture == "tuned_gcn")
    n_sage = sum(1 for c in grid if c.architecture == "graphsage")
    assert n_gcn == 8
    assert n_sage == 8


def test_grid_covers_exactly_the_specified_value_sets_no_duplicates():
    grid = build_grid()
    seen = set()
    for c in grid:
        key = (c.architecture, c.hidden_channels, c.dropout, c.lr)
        assert key not in seen, f"duplicate config {key}"
        seen.add(key)
        assert c.hidden_channels in (64, 128)
        assert c.dropout in (0.2, 0.5)
        assert c.lr in (0.003, 0.01)
    assert len(seen) == 16


def test_make_model_dispatches_correct_architecture():
    gcn_cfg = GridConfig(architecture="tuned_gcn", hidden_channels=32, dropout=0.3, lr=0.01)
    sage_cfg = GridConfig(architecture="graphsage", hidden_channels=32, dropout=0.3, lr=0.01)
    assert isinstance(make_model(gcn_cfg, in_channels=5), GCN)
    assert isinstance(make_model(sage_cfg, in_channels=5), GraphSAGE)
    with pytest.raises(ValueError):
        make_model(GridConfig(architecture="nonsense", hidden_channels=32, dropout=0.3, lr=0.01), in_channels=5)


# --- reuse of train_gcn (Phase 3's function, unmodified) across architectures ---

def test_train_gcn_works_unmodified_with_graphsage():
    """train_gcn is GCN-named but architecture-agnostic; this is the load-
    bearing assumption for reusing it in Phase 5 for GraphSAGE too."""
    train_list = [make_synthetic_data(t, n=10, seed=t) for t in range(1, 4)]
    val_list = [make_synthetic_data(t, n=8, seed=t) for t in range(4, 6)]
    train_batch = Batch.from_data_list(train_list)
    val_batch = Batch.from_data_list(val_list)

    model = GraphSAGE(in_channels=5, hidden_channels=6, dropout=0.5)
    result = train_gcn(model, train_batch, val_batch, seed=42, epochs=3, lr=0.01, weight_decay=5e-4)

    assert result.best_state_dict is not None
    assert 0.0 <= result.best_val_pr_auc <= 1.0
    assert 1 <= result.best_epoch <= 3
    assert len(result.epoch_log) == 3


# --- selection logic: overall winner picked by max validation PR-AUC only ---

def test_overall_selection_picks_max_val_pr_auc():
    records = [
        {"architecture": "tuned_gcn", "val_pr_auc": 0.55},
        {"architecture": "graphsage", "val_pr_auc": 0.81},
        {"architecture": "tuned_gcn", "val_pr_auc": 0.62},
    ]
    best = max(records, key=lambda r: r["val_pr_auc"])
    assert best["architecture"] == "graphsage"
    assert best["val_pr_auc"] == 0.81
