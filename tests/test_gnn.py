import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import load_node_table_with_features
from src.data.graph_builder import Snapshot
from src.data.loader import load_edges
from src.data.pyg_adapter import snapshot_to_data
from src.models.gcn import GCN
from src.training.gnn_training import build_split_batches, compute_pos_weight, predict_labeled

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def make_snapshot(time_step: int, n: int, edges: list[tuple[int, int]], seed: int = 0) -> Snapshot:
    rng = np.random.default_rng(seed)
    node_ids = np.arange(n)
    x = rng.standard_normal((n, 3)).astype(np.float32)
    y = np.array([1, 0, -1] * (n // 3 + 1))[:n].astype(np.int64)
    is_labeled = y >= 0
    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    return Snapshot(time_step=time_step, node_ids=node_ids, x=x, y=y, edge_index=edge_index, is_labeled=is_labeled)


# --- pyg_adapter ---

def test_snapshot_to_data_preserves_shapes_and_values():
    snap = make_snapshot(1, n=4, edges=[(0, 1), (1, 2)])
    data = snapshot_to_data(snap)
    assert data.x.shape == (4, 3)
    assert data.edge_index.shape == (2, 2)
    assert torch.equal(data.y, torch.from_numpy(snap.y))
    assert torch.equal(data.is_labeled, torch.from_numpy(snap.is_labeled))
    assert (data.time_step == 1).all()


def test_batch_is_block_diagonal():
    """Batching two snapshots must never create an edge between them, and
    node/label/mask ordering must be preserved by simple concatenation —
    this is the property src/data/pyg_adapter.py's docstring relies on to
    claim batching == processing snapshots independently."""
    snap1 = make_snapshot(1, n=3, edges=[(0, 1), (1, 2)])
    snap2 = make_snapshot(2, n=2, edges=[(0, 1)])
    batch = Batch.from_data_list([snapshot_to_data(snap1), snapshot_to_data(snap2)])

    assert batch.num_nodes == 5
    # snap2's local edge (0,1) must be offset to (3,4), not connect to snap1
    assert batch.edge_index.shape[1] == 3
    for src, dst in batch.edge_index.t().tolist():
        same_graph = (src < 3 and dst < 3) or (src >= 3 and dst >= 3)
        assert same_graph, f"edge ({src},{dst}) crosses snapshot boundary"

    assert torch.equal(batch.time_step, torch.tensor([1, 1, 1, 2, 2]))
    assert torch.equal(batch.y, torch.cat([torch.from_numpy(snap1.y), torch.from_numpy(snap2.y)]))


# --- GCN model ---

def test_gcn_forward_output_shape():
    torch.manual_seed(0)
    model = GCN(in_channels=3, hidden_channels=8, dropout=0.5)
    x = torch.randn(6, 3)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    out = model(x, edge_index)
    assert out.shape == (6,)
    assert out.dtype == torch.float32


def test_gcn_handles_isolated_nodes_and_empty_edge_index():
    torch.manual_seed(0)
    model = GCN(in_channels=3, hidden_channels=4)
    x = torch.randn(3, 3)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    out = model(x, edge_index)
    assert out.shape == (3,)
    assert torch.isfinite(out).all()


# --- class imbalance handling ---

def test_compute_pos_weight_matches_manual_ratio():
    y = np.array([1, 1, 0, 0, 0, 0])  # 2 positive, 4 negative
    assert compute_pos_weight(y) == pytest.approx(4 / 2)


# --- split batching against the real processed cache (mirrors tests/test_data.py) ---

@pytest.fixture(scope="module")
def real_batches(cfg):
    processed_dir = PROJECT_ROOT / cfg["dataset"]["processed_dir"]
    raw_dir = PROJECT_ROOT / cfg["dataset"]["raw_dir"]
    node_table, feature_cols = load_node_table_with_features(processed_dir)
    edges = load_edges(raw_dir / cfg["dataset"]["edges_file"])
    return build_split_batches(node_table, edges, feature_cols, cfg["split"]), feature_cols


def test_split_batches_match_known_node_counts(real_batches):
    batches, _ = real_batches
    assert batches["train"].num_nodes == 120804
    assert batches["val"].num_nodes == 15461
    assert batches["test"].num_nodes == 67504
    assert int(batches["train"].is_labeled.sum()) == 26381
    assert int(batches["val"].is_labeled.sum()) == 3513
    assert int(batches["test"].is_labeled.sum()) == 16670


def test_unknown_labels_excluded_from_is_labeled_but_present_in_graph(real_batches):
    batches, _ = real_batches
    for split_name in ("train", "val", "test"):
        b = batches[split_name]
        unknown_mask = b.y == -1
        assert not b.is_labeled[unknown_mask].any()
        # unknown nodes still have feature rows / participate in x, not dropped
        assert b.x.shape[0] == b.num_nodes


def test_no_edges_cross_snapshot_boundaries_in_real_batches(real_batches):
    batches, _ = real_batches
    b = batches["train"]
    # every edge's endpoints must share the same time_step (same snapshot)
    src_t = b.time_step[b.edge_index[0]]
    dst_t = b.time_step[b.edge_index[1]]
    assert torch.equal(src_t, dst_t)


def test_predict_labeled_shapes_match_is_labeled_count(real_batches):
    batches, feature_cols = real_batches
    torch.manual_seed(42)
    model = GCN(in_channels=len(feature_cols), hidden_channels=8)
    y_true, y_prob, time_step = predict_labeled(model, batches["val"])
    n_labeled = int(batches["val"].is_labeled.sum())
    assert y_true.shape == (n_labeled,)
    assert y_prob.shape == (n_labeled,)
    assert time_step.shape == (n_labeled,)
    assert set(np.unique(y_true)).issubset({0, 1})
    assert ((y_prob >= 0) & (y_prob <= 1)).all()
