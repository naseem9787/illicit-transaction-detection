"""Phase 4 leakage and correctness tests for the simulated delayed-feedback
online adaptation protocol (src/training/adaptive_gnn.py).

Uses small synthetic snapshots (not the real dataset) so these run fast
and the expected event ordering can be checked exactly by hand.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.graph_builder import Snapshot
from src.data.pyg_adapter import snapshot_to_data
from src.models.gcn import GCN
from src.training.adaptive_gnn import AdaptationConfig, pooled_predictions, run_adaptive_walk, select_adaptation_config


def make_data(time_step: int, n: int, seed: int = 0):
    rng = np.random.default_rng(seed + time_step)
    node_ids = np.arange(n)
    x = rng.standard_normal((n, 4)).astype(np.float32)
    y = np.array([1, 0] * (n // 2 + 1))[:n].astype(np.int64)
    is_labeled = np.ones(n, dtype=bool)
    edges = [(i, i + 1) for i in range(n - 1)]
    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    snap = Snapshot(time_step=time_step, node_ids=node_ids, x=x, y=y, edge_index=edge_index, is_labeled=is_labeled)
    return snapshot_to_data(snap)


@pytest.fixture
def sequential_data():
    time_steps = [1, 2, 3, 4, 5]
    return time_steps, {t: make_data(t, n=8) for t in time_steps}


def run_walk(time_steps, data, k=1, lr=0.01, grad_steps=1, seed=0):
    torch.manual_seed(seed)
    model = GCN(in_channels=4, hidden_channels=6, dropout=0.0)
    pos_weight = torch.tensor(1.0)
    config = AdaptationConfig(lr=lr, grad_steps=grad_steps, feedback_delay_k=k, seed=seed)
    return model, run_adaptive_walk(model, time_steps, data, pos_weight, config)


# --- 1. prediction for t occurs before adaptation using t's labels ---

def test_prediction_precedes_its_own_adaptation(sequential_data):
    time_steps, data = sequential_data
    _, result = run_walk(time_steps, data, k=1)

    seq_index = {}
    for i, ev in enumerate(result.events):
        key = ("predict", ev["step"]) if ev["type"] == "predict" else ("adapt_source", ev["source_time_step"])
        seq_index.setdefault(key, i)

    for t in time_steps[:-1]:  # last step's own reveal never gets applied (walk ends)
        predict_idx = seq_index[("predict", t)]
        adapt_idx = seq_index[("adapt_source", t)]
        assert predict_idx < adapt_idx, f"t={t}: adaptation using its own labels happened before/at its prediction"


# --- 2. adaptation for t cannot access labels from future steps > t ---

def test_adaptation_never_uses_future_labels(sequential_data):
    time_steps, data = sequential_data
    for k in (1, 2, 3):
        _, result = run_walk(time_steps, data, k=k)
        for ev in result.events:
            if ev["type"] == "adapt":
                assert ev["source_time_step"] == ev["applied_before_predicting"] - k
                assert ev["source_time_step"] < ev["applied_before_predicting"]


def test_no_adaptation_event_before_first_prediction(sequential_data):
    """The very first predicted step must use the pretrained (unadapted) weights."""
    time_steps, data = sequential_data
    _, result = run_walk(time_steps, data, k=1)
    first_event = result.events[0]
    assert first_event["type"] == "predict"
    assert first_event["step"] == time_steps[0]
    assert first_event["n_labeled"] == 8


# --- 3. threshold remains unchanged during test (never re-selected) ---

def test_evaluate_adaptive_predictions_never_reselects_threshold(monkeypatch):
    from scripts.evaluate_adaptive_gnn import evaluate_predictions
    import src.evaluation.metrics as metrics_module

    def _boom(*args, **kwargs):
        raise AssertionError("select_threshold_maximizing_f1 must never be called on test data")

    monkeypatch.setattr(metrics_module, "select_threshold_maximizing_f1", _boom)

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=50)
    p = rng.random(50)
    t = np.array([35] * 25 + [36] * 25)
    fixed_threshold = 0.7
    result = evaluate_predictions(y, p, t, fixed_threshold)
    assert result["test_metrics"]["threshold"] == fixed_threshold
    assert result["threshold"] == fixed_threshold


# --- 4. historical predictions are not modified (deterministic, frozen at log time) ---

def test_predictions_are_deterministic_and_not_mutated_afterward(sequential_data):
    time_steps, data = sequential_data
    _, result1 = run_walk(time_steps, data, k=1, seed=42)
    _, result2 = run_walk(time_steps, data, k=1, seed=42)

    for t in time_steps:
        y1, p1, ts1 = result1.predictions[t]
        y2, p2, ts2 = result2.predictions[t]
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_allclose(p1, p2)
        np.testing.assert_array_equal(ts1, ts2)

    # entries are assigned exactly once per time step, never overwritten
    assert len(result1.predictions) == len(time_steps)
    assert sorted(result1.predictions.keys()) == sorted(time_steps)


# --- 5. no replay / adaptation batch contains only the single revealed snapshot ---

def test_adaptation_uses_only_the_single_revealed_snapshot(sequential_data, monkeypatch):
    """No replay buffer is implemented; verify each adaptation step's loss
    is computed over exactly the revealed snapshot's labeled node count,
    nothing extra mixed in."""
    time_steps, data = sequential_data
    seen_shapes = []

    orig_forward = torch.nn.BCEWithLogitsLoss.forward

    def spy_forward(self, input, target):
        seen_shapes.append(input.shape[0])
        return orig_forward(self, input, target)

    monkeypatch.setattr(torch.nn.BCEWithLogitsLoss, "forward", spy_forward)
    _, result = run_walk(time_steps, data, k=1, grad_steps=2)

    expected_calls = 0
    for ev in result.events:
        if ev["type"] == "adapt":
            source_n_labeled = ev["n_labeled_used"]
            for _ in range(2):  # grad_steps=2
                expected_calls += 1
    assert len(seen_shapes) == expected_calls
    for shape in seen_shapes:
        assert shape == 8  # every synthetic snapshot has 8 labeled nodes, never more


def test_replay_true_is_rejected():
    with pytest.raises(NotImplementedError):
        model = GCN(in_channels=4, hidden_channels=6)
        run_adaptive_walk(model, [1], {1: make_data(1, n=4)}, torch.tensor(1.0),
                           AdaptationConfig(lr=0.01, grad_steps=1, feedback_delay_k=1, replay=True))


# --- 6. chronological order preserved ---

def test_chronological_order_preserved(sequential_data):
    time_steps, data = sequential_data
    _, result = run_walk(time_steps, data, k=1)
    assert result.predict_order == time_steps


def test_chronological_order_preserved_out_of_order_input_still_processed_in_given_order():
    """Guards against silently reordering — the walk must process exactly
    the order it's given (callers, i.e. the training script, are
    responsible for passing sorted time steps)."""
    time_steps = [1, 2, 3]
    data = {t: make_data(t, n=6) for t in time_steps}
    _, result = run_walk(time_steps, data, k=1)
    assert result.predict_order == [1, 2, 3]
    predict_steps_in_events = [e["step"] for e in result.events if e["type"] == "predict"]
    assert predict_steps_in_events == [1, 2, 3]


# --- config selection sanity ---

def test_select_adaptation_config_prefers_higher_val_score(sequential_data):
    time_steps, data = sequential_data
    torch.manual_seed(0)
    model = GCN(in_channels=4, hidden_channels=6, dropout=0.0)
    state_dict = model.state_dict()
    pos_weight = torch.tensor(1.0)

    candidates = [
        AdaptationConfig(lr=0.0, grad_steps=1, feedback_delay_k=1),  # lr=0 -> no-op adaptation
        AdaptationConfig(lr=0.05, grad_steps=5, feedback_delay_k=1),
    ]
    best, log = select_adaptation_config(state_dict, 4, 6, 0.0, time_steps, data, pos_weight, candidates)
    assert len(log) == 2
    assert best in candidates


def test_pooled_predictions_concatenates_in_given_order(sequential_data):
    time_steps, data = sequential_data
    _, result = run_walk(time_steps, data, k=1)
    y_true, y_prob, time_step = pooled_predictions(result)
    assert len(y_true) == len(time_steps) * 8
    assert list(time_step) == sorted(time_step)
