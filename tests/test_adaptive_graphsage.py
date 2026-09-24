"""Phase 6 leakage/correctness tests: the Phase 4 delayed-feedback protocol
applied to GraphSAGE. Synthetic snapshots for exact ordering checks, plus a
few checks against the real saved Phase 5/6 artifacts."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_adaptive_graphsage import evaluate_predictions, load_frozen_threshold
from scripts.train_adaptive_graphsage import CANDIDATE_GRID
from src.data.graph_builder import Snapshot
from src.data.pyg_adapter import snapshot_to_data
from src.models.graphsage import GraphSAGE
from src.training.adaptive_gnn import AdaptationConfig, pooled_predictions, run_adaptive_walk
from src.training.adaptive_graphsage import select_adaptation_config_with_factory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
EXPECTED_THRESHOLD = 0.8980084657669067


def make_data(time_step: int, n: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed + time_step)
    x = rng.standard_normal((n, 4)).astype(np.float32)
    y = np.array([1, 0] * (n // 2 + 1))[:n].astype(np.int64)
    edges = np.array([(i, i + 1) for i in range(n - 1)], dtype=np.int64).T
    snap = Snapshot(time_step=time_step, node_ids=np.arange(n), x=x, y=y, edge_index=edges,
                    is_labeled=np.ones(n, dtype=bool))
    return snapshot_to_data(snap)


def factory():
    return GraphSAGE(in_channels=4, hidden_channels=6, dropout=0.5)


@pytest.fixture
def seq():
    steps = [1, 2, 3, 4, 5]
    return steps, {t: make_data(t) for t in steps}


def run_walk(steps, data, k=1, lr=0.01, grad_steps=1, seed=0):
    torch.manual_seed(seed)
    model = factory()
    cfg = AdaptationConfig(lr=lr, grad_steps=grad_steps, feedback_delay_k=k, seed=seed)
    return model, run_adaptive_walk(model, steps, data, torch.tensor(1.0), cfg)


# --- the six Phase 4 leakage checks, on GraphSAGE ---

def test_prediction_precedes_its_own_adaptation(seq):
    steps, data = seq
    _, result = run_walk(steps, data, k=1)
    idx = {}
    for i, ev in enumerate(result.events):
        key = ("predict", ev["step"]) if ev["type"] == "predict" else ("adapt", ev["source_time_step"])
        idx.setdefault(key, i)
    for t in steps[:-1]:
        assert idx[("predict", t)] < idx[("adapt", t)]


def test_adaptation_never_uses_future_labels(seq):
    steps, data = seq
    for k in (1, 2, 3):
        _, result = run_walk(steps, data, k=k)
        for ev in result.events:
            if ev["type"] == "adapt":
                assert ev["source_time_step"] == ev["applied_before_predicting"] - k
                assert ev["source_time_step"] < ev["applied_before_predicting"]


def test_evaluation_never_reselects_threshold(monkeypatch):
    import src.evaluation.metrics as metrics_module

    def boom(*a, **k):
        raise AssertionError("threshold must never be selected on test data")

    monkeypatch.setattr(metrics_module, "select_threshold_maximizing_f1", boom)
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 50), rng.random(50)
    t = np.array([35] * 25 + [36] * 25)
    out = evaluate_predictions(y, p, t, EXPECTED_THRESHOLD)
    assert out["threshold"] == EXPECTED_THRESHOLD
    assert out["test_metrics"]["threshold"] == EXPECTED_THRESHOLD


def test_predictions_deterministic_and_not_rewritten(seq):
    steps, data = seq
    _, r1 = run_walk(steps, data, seed=42)
    _, r2 = run_walk(steps, data, seed=42)
    assert sorted(r1.predictions) == steps
    for t in steps:
        for a, b in zip(r1.predictions[t], r2.predictions[t]):
            np.testing.assert_array_equal(a, b)


def test_adaptation_uses_only_revealed_snapshot(seq, monkeypatch):
    steps, data = seq
    shapes = []
    orig = torch.nn.BCEWithLogitsLoss.forward

    def spy(self, input, target):
        shapes.append(input.shape[0])
        return orig(self, input, target)

    monkeypatch.setattr(torch.nn.BCEWithLogitsLoss, "forward", spy)
    _, result = run_walk(steps, data, k=1, grad_steps=2)
    n_adapt = sum(1 for e in result.events if e["type"] == "adapt")
    assert len(shapes) == n_adapt * 2
    assert all(s == 8 for s in shapes)


def test_chronological_order_preserved(seq):
    steps, data = seq
    _, result = run_walk(steps, data)
    assert result.predict_order == steps
    assert [e["step"] for e in result.events if e["type"] == "predict"] == steps


# --- first prediction identical to static, before any update ---

def test_first_prediction_bit_identical_to_static_synthetic(seq):
    steps, data = seq
    torch.manual_seed(0)
    base = factory()
    state = {k: v.clone() for k, v in base.state_dict().items()}

    static = factory()
    static.load_state_dict(state)
    static.eval()
    with torch.no_grad():
        static_prob = torch.sigmoid(static(data[1].x, data[1].edge_index)).numpy()

    adaptive = factory()
    adaptive.load_state_dict(state)
    result = run_adaptive_walk(adaptive, steps, data, torch.tensor(1.0),
                               AdaptationConfig(lr=0.01, grad_steps=1, feedback_delay_k=1))
    assert result.events[0]["type"] == "predict"
    np.testing.assert_array_equal(result.predictions[1][1], static_prob)


def test_real_artifacts_t35_identical_and_same_population():
    static = np.load(METRICS_DIR / "predictions" / "gnn_improvement_test.npz")
    adaptive = np.load(METRICS_DIR / "predictions" / "adaptive_graphsage_test.npz")
    assert np.array_equal(static["time_step"], adaptive["time_step"])
    assert np.array_equal(static["y_true"], adaptive["y_true"])
    m = static["time_step"] == 35
    assert np.array_equal(static["y_prob"][m], adaptive["y_prob"][m])
    later = static["time_step"] == 36
    assert not np.array_equal(static["y_prob"][later], adaptive["y_prob"][later])


# --- frozen threshold ---

def test_frozen_threshold_is_phase5_validation_value():
    assert load_frozen_threshold(METRICS_DIR) == pytest.approx(EXPECTED_THRESHOLD)
    assert round(load_frozen_threshold(METRICS_DIR), 3) == 0.898


# --- k=3 is sensitivity only ---

def test_selection_rejects_non_k1_candidates(seq):
    steps, data = seq
    state = factory().state_dict()
    bad = [AdaptationConfig(lr=0.01, grad_steps=1, feedback_delay_k=1),
           AdaptationConfig(lr=0.01, grad_steps=1, feedback_delay_k=3)]
    with pytest.raises(ValueError):
        select_adaptation_config_with_factory(factory, state, steps, data, torch.tensor(1.0), bad)


def test_script_grid_is_exactly_the_predeclared_k1_grid():
    assert len(CANDIDATE_GRID) == 4
    assert {(c.lr, c.grad_steps) for c in CANDIDATE_GRID} == {(0.001, 1), (0.001, 3), (0.005, 1), (0.005, 3)}
    assert all(c.feedback_delay_k == 1 for c in CANDIDATE_GRID)
    assert not any(c.replay for c in CANDIDATE_GRID)


def test_k3_walk_cannot_change_k1_selection(seq):
    steps, data = seq
    torch.manual_seed(0)
    state = {k: v.clone() for k, v in factory().state_dict().items()}
    grid = [AdaptationConfig(lr=0.001, grad_steps=1, feedback_delay_k=1),
            AdaptationConfig(lr=0.05, grad_steps=3, feedback_delay_k=1)]
    best_before, log_before = select_adaptation_config_with_factory(factory, state, steps, data, torch.tensor(1.0), grid)
    k3_model = factory()
    k3_model.load_state_dict(state)
    run_adaptive_walk(k3_model, steps, data, torch.tensor(1.0), AdaptationConfig(lr=0.05, grad_steps=3, feedback_delay_k=3))
    best_after, log_after = select_adaptation_config_with_factory(factory, state, steps, data, torch.tensor(1.0), grid)
    assert (best_before.lr, best_before.grad_steps) == (best_after.lr, best_after.grad_steps)
    assert log_before == log_after
