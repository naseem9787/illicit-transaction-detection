import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.graph_builder import Snapshot
from src.data.pyg_adapter import snapshot_to_data
from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE
from src.models.graphsage_directional import DirectionAwareGraphSAGE, SymmetrizedGraphSAGE, symmetrize
from src.training.rolling_origin_graph import E3_ARCHITECTURES, TEST_RANGE, build_graph_batches_for_range
from torch_geometric.data import Batch


def make_data(time_step: int, n: int, edges: list[tuple[int, int]], seed: int = 0):
    rng = np.random.default_rng(seed + time_step)
    x = rng.standard_normal((n, 4)).astype(np.float32)
    y = np.array([1, 0] * (n // 2 + 1))[:n].astype(np.int64)
    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    snap = Snapshot(time_step=time_step, node_ids=np.arange(n), x=x, y=y, edge_index=edge_index,
                    is_labeled=np.ones(n, dtype=bool))
    return snapshot_to_data(snap)


# --- symmetrize() ---

def test_symmetrize_adds_reverse_edges():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    sym = symmetrize(edge_index)
    assert sym.shape[1] == edge_index.shape[1] * 2
    edge_set = {tuple(e) for e in edge_index.t().tolist()}
    sym_set = {tuple(e) for e in sym.t().tolist()}
    for (u, v) in edge_set:
        assert (u, v) in sym_set
        assert (v, u) in sym_set


def test_symmetrize_handles_empty_edge_index():
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    sym = symmetrize(edge_index)
    assert sym.shape == (2, 0)


# --- no cross-time edges after symmetrization / direction reversal ---

def test_no_cross_time_edges_after_symmetrization():
    snap1 = make_data(1, n=3, edges=[(0, 1), (1, 2)])
    snap2 = make_data(2, n=2, edges=[(0, 1)])
    batch = Batch.from_data_list([snap1, snap2])

    sym_edge_index = symmetrize(batch.edge_index)
    src_t = batch.time_step[sym_edge_index[0]]
    dst_t = batch.time_step[sym_edge_index[1]]
    assert torch.equal(src_t, dst_t), "symmetrization must never connect two different snapshots"

    reversed_edge_index = batch.edge_index.flip(0)
    src_t2 = batch.time_step[reversed_edge_index[0]]
    dst_t2 = batch.time_step[reversed_edge_index[1]]
    assert torch.equal(src_t2, dst_t2), "reversing edge direction must never connect two different snapshots"


def test_build_graph_batches_for_range_never_touches_test_period():
    with pytest.raises(ValueError):
        build_graph_batches_for_range(None, None, None, 30, 35)  # end reaches into test period
    with pytest.raises(ValueError):
        build_graph_batches_for_range(None, None, None, TEST_RANGE[0], TEST_RANGE[1])


# --- direction handling correctness ---

def test_symmetrized_model_output_identical_regardless_of_input_edge_direction():
    """A purely symmetric-aggregation model should give the same output
    whether it's handed edge_index or its reverse, since symmetrize()
    produces the same edge set either way."""
    torch.manual_seed(0)
    model = SymmetrizedGraphSAGE(in_channels=4, hidden_channels=8, dropout=0.0)
    model.eval()
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    with torch.no_grad():
        out_fwd = model(x, edge_index)
        out_rev = model(x, edge_index.flip(0))
    torch.testing.assert_close(out_fwd, out_rev)


def test_direction_aware_model_is_sensitive_to_edge_direction():
    """Unlike the symmetrized variant, the direction-aware model treats
    incoming and outgoing neighbors through different weights, so swapping
    the input edge direction should generally change the output."""
    torch.manual_seed(0)
    model = DirectionAwareGraphSAGE(in_channels=4, hidden_channels=8, dropout=0.0)
    model.eval()
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    with torch.no_grad():
        out_fwd = model(x, edge_index)
        out_rev = model(x, edge_index.flip(0))
    assert not torch.allclose(out_fwd, out_rev)


def test_direction_aware_uses_both_conv_in_and_conv_out():
    model = DirectionAwareGraphSAGE(in_channels=4, hidden_channels=8, dropout=0.0)
    assert hasattr(model, "conv_in") and hasattr(model, "conv_out")
    assert model.conv_in.in_channels == 4
    assert model.conv_out.in_channels == 4
    # each branch is half of hidden_channels so the concatenation matches the 128-style baseline width
    assert model.conv_in.out_channels == 4  # 8 // 2


def test_direction_aware_requires_even_hidden_channels():
    with pytest.raises(AssertionError):
        DirectionAwareGraphSAGE(in_channels=4, hidden_channels=7)


# --- output shapes ---

def test_both_variants_forward_output_shape():
    x = torch.randn(6, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    for cls in (SymmetrizedGraphSAGE, DirectionAwareGraphSAGE):
        model = cls(in_channels=4, hidden_channels=8, dropout=0.5)
        out = model(x, edge_index)
        assert out.shape == (6,)
        assert out.dtype == torch.float32


def test_both_variants_handle_empty_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    for cls in (SymmetrizedGraphSAGE, DirectionAwareGraphSAGE):
        model = cls(in_channels=4, hidden_channels=8, dropout=0.5)
        out = model(x, edge_index)
        assert out.shape == (3,)
        assert torch.isfinite(out).all()


# --- no labels used as graph features ---

def test_models_only_take_x_and_edge_index_never_y():
    for cls in (SymmetrizedGraphSAGE, DirectionAwareGraphSAGE, GraphSAGE):
        params = list(inspect.signature(cls.forward).parameters)
        assert params == ["self", "x", "edge_index"], f"{cls.__name__} forward signature must not include y/label"


def test_e3_architectures_include_the_unmodified_phase5_model_as_control():
    assert E3_ARCHITECTURES["plain_graphsage"] is GraphSAGE
    assert E3_ARCHITECTURES["symmetrized_graphsage"] is SymmetrizedGraphSAGE
    assert E3_ARCHITECTURES["direction_aware_graphsage"] is DirectionAwareGraphSAGE


# --- Phase 1-7 behavior unchanged (regression smoke checks) ---

def test_original_gcn_and_graphsage_models_unchanged():
    """GCN/GraphSAGE (Phase 3/5) must still behave exactly as before —
    this file adds new models, it must not alter the existing ones."""
    torch.manual_seed(0)
    gcn = GCN(in_channels=4, hidden_channels=8, dropout=0.5)
    sage = GraphSAGE(in_channels=4, hidden_channels=8, dropout=0.5)
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    assert gcn(x, edge_index).shape == (5,)
    assert sage(x, edge_index).shape == (5,)
    assert list(inspect.signature(GCN.forward).parameters) == ["self", "x", "edge_index"]
    assert list(inspect.signature(GraphSAGE.forward).parameters) == ["self", "x", "edge_index"]
