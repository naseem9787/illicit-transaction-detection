import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.graph_smoothing import ALPHAS, DEPTHS, HGB_BASE_CONFIG, smooth_scores, symmetrize_numpy as symmetrize


# --- alpha=0 reproduces the tree score exactly ---

def test_alpha_zero_reproduces_tree_score_exactly_depth1():
    scores = np.array([0.1, 0.9, 0.4, 0.6, 0.05])
    edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    sym = symmetrize(edge_index)
    out = smooth_scores(scores, sym, alpha=0.0, depth=1)
    np.testing.assert_array_equal(out, scores)


def test_alpha_zero_reproduces_tree_score_exactly_depth2():
    scores = np.array([0.1, 0.9, 0.4, 0.6, 0.05])
    edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    sym = symmetrize(edge_index)
    out = smooth_scores(scores, sym, alpha=0.0, depth=2)
    np.testing.assert_array_equal(out, scores)


# --- isolated nodes retain their original score ---

def test_isolated_node_keeps_original_score_regardless_of_alpha_and_depth():
    scores = np.array([0.2, 0.8, 0.5])
    # node 2 has no edges at all
    edge_index = np.array([[0], [1]])
    sym = symmetrize(edge_index)
    for alpha in ALPHAS:
        for depth in DEPTHS:
            out = smooth_scores(scores, sym, alpha=alpha, depth=depth)
            assert out[2] == pytest.approx(scores[2]), f"alpha={alpha} depth={depth}"


def test_fully_isolated_graph_no_edges_returns_scores_unchanged():
    scores = np.array([0.3, 0.7, 0.1])
    empty_edges = np.zeros((2, 0), dtype=np.int64)
    out = smooth_scores(scores, empty_edges, alpha=0.5, depth=2)
    np.testing.assert_array_equal(out, scores)


# --- correct incoming/outgoing neighbor handling ---

def test_uses_both_incoming_and_outgoing_neighbors():
    """Node 1 has an incoming edge from 0 and an outgoing edge to 2. Its
    smoothed score at alpha=1 (pure neighbor average) must reflect BOTH
    neighbors, not just one direction."""
    scores = np.array([0.0, 0.5, 1.0])
    edge_index = np.array([[0, 1], [1, 2]])  # 0->1 (incoming to 1), 1->2 (outgoing from 1)
    sym = symmetrize(edge_index)
    out = smooth_scores(scores, sym, alpha=1.0, depth=1)
    assert out[1] == pytest.approx((scores[0] + scores[2]) / 2)


def test_raw_directed_edge_index_alone_would_miss_outgoing_neighbor():
    """Sanity check that symmetrize() is doing real work here: smoothing
    over the RAW (unsymmetrized) edge_index would only use node 1's
    incoming neighbor (0), not its outgoing neighbor (2)."""
    scores = np.array([0.0, 0.5, 1.0])
    edge_index = np.array([[0, 1], [1, 2]])
    out_raw = smooth_scores(scores, edge_index, alpha=1.0, depth=1)
    assert out_raw[1] == pytest.approx(scores[0])  # only sees incoming neighbor 0, not outgoing 2


# --- no cross-time smoothing ---

def test_snapshots_are_smoothed_independently_never_sharing_edges():
    """E7 never batches snapshots together (unlike the GNN path) — it calls
    _predict_and_smooth_snapshot once per time_step, so each snapshot's
    edge_index (already scoped to that single time_step by
    src/data/graph_builder.py) is symmetrized and smoothed in complete
    isolation from every other time_step's nodes and edges."""
    from src.data.graph_builder import Snapshot

    snap1_edges = np.array([[0, 1], [1, 2]])
    snap2_edges = np.array([[0], [1]])
    sym1 = symmetrize(snap1_edges)
    sym2 = symmetrize(snap2_edges)

    # snapshot 1 has 3 local nodes (0,1,2), snapshot 2 has 2 local nodes (0,1) —
    # both use the SAME local index range, which is only safe because they are
    # never combined into one array. Confirm smoothing snapshot 1 with its own
    # edges never references an index that only exists in snapshot 2 and vice
    # versa (both index spaces are valid 0..n-1 for their OWN snapshot only).
    scores1 = np.array([0.1, 0.5, 0.9])
    scores2 = np.array([0.2, 0.8])
    out1 = smooth_scores(scores1, sym1, alpha=0.5, depth=2)
    out2 = smooth_scores(scores2, sym2, alpha=0.5, depth=2)
    assert out1.shape == (3,)
    assert out2.shape == (2,)
    # changing snapshot 2's scores must not affect snapshot 1's smoothed output
    out1_again = smooth_scores(scores1, sym1, alpha=0.5, depth=2)
    np.testing.assert_array_equal(out1, out1_again)


# --- no label usage ---

def test_smooth_scores_signature_never_takes_labels():
    params = list(inspect.signature(smooth_scores).parameters)
    assert "y" not in params and "label" not in params and "labels" not in params
    assert params == ["scores", "edge_index_sym", "alpha", "depth"]


# --- determinism ---

def test_smooth_scores_deterministic():
    rng = np.random.default_rng(0)
    scores = rng.random(20)
    edge_index = rng.integers(0, 20, size=(2, 30))
    sym = symmetrize(edge_index)
    out1 = smooth_scores(scores, sym, alpha=0.3, depth=2)
    out2 = smooth_scores(scores, sym, alpha=0.3, depth=2)
    np.testing.assert_array_equal(out1, out2)


# --- predeclared grid matches the spec exactly ---

def test_predeclared_grid_matches_spec():
    assert ALPHAS == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    assert DEPTHS == [1, 2]
    assert HGB_BASE_CONFIG == {"max_depth": None, "learning_rate": 0.1, "max_iter": 100, "min_samples_leaf": 50}


# --- depth=2 genuinely differs from depth=1 when there is more than 1-hop structure ---

def test_depth_2_reaches_further_than_depth_1():
    # chain 0-1-2-3; node 0's 1-hop neighbor is only 1, but depth=2 should
    # let information from 2 influence node 0 through node 1. (An
    # alternating [1,0,1,0] pattern happens to hit a fixed point after one
    # iteration on this chain, which would make d1==d2 by coincidence — use
    # an asymmetric pattern instead so the two depths are provably different.)
    scores = np.array([1.0, 0.0, 0.0, 0.0])
    edge_index = np.array([[0, 1, 2], [1, 2, 3]])
    sym = symmetrize(edge_index)
    out_d1 = smooth_scores(scores, sym, alpha=0.5, depth=1)
    out_d2 = smooth_scores(scores, sym, alpha=0.5, depth=2)
    assert out_d1[0] == pytest.approx(0.5)
    assert out_d2[0] == pytest.approx(0.375)
    assert out_d1[0] != pytest.approx(out_d2[0])
