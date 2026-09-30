import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.graph_features import (
    GRAPH_FEATURE_SCALAR_NAMES,
    HGB_BASE_CONFIG,
    compute_graph_features,
    graph_feature_names,
)


# --- no label usage ---

def test_compute_graph_features_signature_never_takes_labels():
    params = list(inspect.signature(compute_graph_features).parameters)
    assert params == ["x", "edge_index"]
    assert "y" not in params and "label" not in params


# --- correct incoming/outgoing handling ---

def test_degrees_and_neighbor_means_are_direction_correct():
    # chain 0 -> 1 -> 2: node 1's incoming neighbor is 0, outgoing neighbor is 2
    x = np.array([[10.0, 0.0], [0.0, 0.0], [0.0, 20.0]], dtype=np.float32)
    edge_index = np.array([[0, 1], [1, 2]])
    feats = compute_graph_features(x, edge_index)

    in_degree, out_degree, total_degree, two_hop = feats[1, :4]
    assert in_degree == 1
    assert out_degree == 1
    assert total_degree == 2

    d = x.shape[1]
    in_mean = feats[1, 4 : 4 + d]
    out_mean = feats[1, 4 + d : 4 + 2 * d]
    np.testing.assert_array_almost_equal(in_mean, x[0])  # node 1's incoming neighbor is node 0
    np.testing.assert_array_almost_equal(out_mean, x[2])  # node 1's outgoing neighbor is node 2


def test_raw_edges_alone_would_give_node0_no_incoming_and_node2_no_outgoing():
    """Sanity check the direction split is real: node 0 (chain start) has no
    incoming neighbor, node 2 (chain end) has no outgoing neighbor."""
    x = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    edge_index = np.array([[0, 1], [1, 2]])
    feats = compute_graph_features(x, edge_index)
    assert feats[0, 0] == 0  # node 0 in_degree
    assert feats[2, 1] == 0  # node 2 out_degree


# --- nodes with no neighbors ---

def test_isolated_node_gets_zero_degree_and_zero_neighbor_means():
    x = np.array([[5.0, -3.0], [1.0, 1.0]], dtype=np.float32)
    edge_index = np.zeros((2, 0), dtype=np.int64)  # no edges at all
    feats = compute_graph_features(x, edge_index)
    d = x.shape[1]
    for v in range(2):
        in_degree, out_degree, total_degree, two_hop = feats[v, :4]
        assert in_degree == 0 and out_degree == 0 and total_degree == 0 and two_hop == 0
        np.testing.assert_array_equal(feats[v, 4 : 4 + d], np.zeros(d))
        np.testing.assert_array_equal(feats[v, 4 + d : 4 + 2 * d], np.zeros(d))


def test_node_with_only_incoming_neighbor_has_zero_outgoing_mean():
    x = np.array([[1.0], [2.0]], dtype=np.float32)
    edge_index = np.array([[0], [1]])  # 0 -> 1: node 1 has an incoming neighbor, no outgoing
    feats = compute_graph_features(x, edge_index)
    assert feats[1, 1] == 0  # out_degree
    d = x.shape[1]
    np.testing.assert_array_equal(feats[1, 4 + d : 4 + 2 * d], np.zeros(d))  # out_neighbor_mean


# --- 2-hop reach correctness ---

def test_two_hop_reach_on_a_chain():
    # chain 0-1-2-3-4 (symmetrized): node 2's 1-hop neighbors = {1,3},
    # 2-hop reach adds neighbors-of-neighbors = {0,1,2,3,4} minus self = {0,1,3,4}
    x = np.zeros((5, 1), dtype=np.float32)
    edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    feats = compute_graph_features(x, edge_index)
    assert feats[2, 3] == 4  # two_hop_reach for node 2: {0,1,3,4}
    assert feats[0, 3] == 2  # node 0: 1-hop={1}, 2-hop adds {2} -> {1,2}


# --- determinism ---

def test_compute_graph_features_deterministic():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((15, 4)).astype(np.float32)
    edge_index = rng.integers(0, 15, size=(2, 25))
    out1 = compute_graph_features(x, edge_index)
    out2 = compute_graph_features(x, edge_index)
    np.testing.assert_array_equal(out1, out2)


# --- original 165 (here: D) features never modified ---

def test_original_features_unmodified_when_concatenated():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((10, 6)).astype(np.float32)
    edge_index = rng.integers(0, 10, size=(2, 15))
    graph_feats = compute_graph_features(x, edge_index)
    augmented = np.concatenate([x, graph_feats], axis=1)
    np.testing.assert_array_equal(augmented[:, : x.shape[1]], x)


# --- no cross-time features (per-snapshot isolation) ---

def test_features_computed_independently_per_snapshot():
    """compute_graph_features takes a single (x, edge_index) pair — there is
    no mechanism for one call to see another snapshot's data. Two different
    snapshots with the same local index range must not interfere."""
    x1 = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    edge_index1 = np.array([[0, 1], [1, 2]])
    x2 = np.array([[100.0], [200.0]], dtype=np.float32)
    edge_index2 = np.array([[0], [1]])

    feats1_first = compute_graph_features(x1, edge_index1)
    _ = compute_graph_features(x2, edge_index2)  # different snapshot, computed after
    feats1_second = compute_graph_features(x1, edge_index1)
    np.testing.assert_array_equal(feats1_first, feats1_second)


# --- predeclared feature set is fixed ---

def test_graph_feature_names_and_dimensions_are_fixed():
    assert GRAPH_FEATURE_SCALAR_NAMES == ["in_degree", "out_degree", "total_degree", "two_hop_reach"]
    feature_cols = [f"f{i}" for i in range(1, 166)]
    names = graph_feature_names(feature_cols)
    assert len(names) == 4 + 2 * 165
    assert names[:4] == GRAPH_FEATURE_SCALAR_NAMES
    assert names[4] == "in_neighbor_mean_f1"
    assert names[4 + 165] == "out_neighbor_mean_f1"


def test_hgb_base_config_matches_e2_selected_config():
    assert HGB_BASE_CONFIG == {"max_depth": None, "learning_rate": 0.1, "max_iter": 100, "min_samples_leaf": 50}


def test_compute_graph_features_output_shape():
    x = np.random.default_rng(0).standard_normal((7, 165)).astype(np.float32)
    edge_index = np.array([[0, 1, 2], [1, 2, 3]])
    out = compute_graph_features(x, edge_index)
    assert out.shape == (7, 4 + 2 * 165)
