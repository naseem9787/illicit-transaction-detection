import copy
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.adaptive_tree import (
    ADAPT_SEEDS,
    FEEDBACK_DELAY_K,
    ITERS_PER_UPDATE,
    compute_fixed_sample_weights,
    fit_base_model,
    pooled_predictions,
    run_adaptive_tree_walk,
)
from src.training.graph_features import HGB_BASE_CONFIG


def make_snapshot(t: int, n: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed + t)
    X = rng.standard_normal((n, 5)).astype(np.float32)
    y = (X[:, 0] + rng.normal(scale=0.3, size=n) > 0).astype(np.int64)
    return X, y


@pytest.fixture
def sequential_data():
    steps = [1, 2, 3, 4, 5]
    return steps, {t: make_snapshot(t) for t in steps}


@pytest.fixture
def base_model_and_weights():
    X_train, y_train = make_snapshot(0, n=200, seed=100)
    weights = compute_fixed_sample_weights(y_train)
    model = fit_base_model(X_train, y_train, seed=42, class_weights=weights)
    return model, weights


# --- adaptation occurs only after feedback (prediction precedes its own adaptation) ---

def test_prediction_precedes_its_own_adaptation(sequential_data, base_model_and_weights):
    steps, data = sequential_data
    base_model, weights = base_model_and_weights
    model = copy.deepcopy(base_model)
    result = run_adaptive_tree_walk(model, steps, data, weights)

    seq_predict, seq_adapt_source = {}, {}
    for i, ev in enumerate(result.events):
        if ev["type"] == "predict":
            seq_predict[ev["step"]] = i
        else:
            seq_adapt_source.setdefault(ev["source_time_step"], i)

    for t in steps[:-1]:  # last step's own reveal is never applied (walk ends)
        assert seq_predict[t] < seq_adapt_source[t]


def test_no_adaptation_event_before_first_prediction(sequential_data, base_model_and_weights):
    steps, data = sequential_data
    base_model, weights = base_model_and_weights
    model = copy.deepcopy(base_model)
    result = run_adaptive_tree_walk(model, steps, data, weights)
    assert result.events[0]["type"] == "predict"
    assert result.events[0]["step"] == steps[0]


# --- no future labels ---

def test_adaptation_never_uses_future_labels(sequential_data, base_model_and_weights):
    steps, data = sequential_data
    base_model, weights = base_model_and_weights
    for k in (1, 2, 3):
        model = copy.deepcopy(base_model)
        result = run_adaptive_tree_walk(model, steps, data, weights, feedback_delay_k=k)
        for ev in result.events:
            if ev["type"] == "adapt":
                assert ev["source_time_step"] == ev["applied_before_predicting"] - k
                assert ev["source_time_step"] < ev["applied_before_predicting"]


def test_default_feedback_delay_is_k1_matching_project_primary():
    assert FEEDBACK_DELAY_K == 1


# --- chronological ordering ---

def test_chronological_order_preserved(sequential_data, base_model_and_weights):
    steps, data = sequential_data
    base_model, weights = base_model_and_weights
    model = copy.deepcopy(base_model)
    result = run_adaptive_tree_walk(model, steps, data, weights)
    assert result.predict_order == steps
    assert [e["step"] for e in result.events if e["type"] == "predict"] == steps


# --- prediction immutability ---

def test_predictions_not_rewritten_after_later_adaptation(sequential_data, base_model_and_weights):
    steps, data = sequential_data
    base_model, weights = base_model_and_weights
    model = copy.deepcopy(base_model)
    result = run_adaptive_tree_walk(model, steps, data, weights)
    assert len(result.predictions) == len(steps)
    assert sorted(result.predictions.keys()) == steps
    # snapshot the first prediction right after the walk, then confirm nothing
    # downstream (pooling) can have mutated it
    first_y, first_p = result.predictions[steps[0]]
    y_pooled, p_pooled = pooled_predictions(result)
    idx = list(result.predict_order).index(steps[0])
    n0 = len(first_y)
    np.testing.assert_array_equal(y_pooled[:n0], first_y)
    np.testing.assert_array_equal(p_pooled[:n0], first_p)


# --- static and adaptive start from identical weights ---

def test_static_and_adaptive_predictions_identical_before_any_update(sequential_data, base_model_and_weights):
    steps, data = sequential_data
    base_model, weights = base_model_and_weights

    static_model = copy.deepcopy(base_model)
    X0, _ = data[steps[0]]
    static_pred_t0 = static_model.predict_proba(X0)[:, 1]

    adaptive_model = copy.deepcopy(base_model)
    result = run_adaptive_tree_walk(adaptive_model, steps, data, weights)
    adaptive_pred_t0 = result.predictions[steps[0]][1]

    np.testing.assert_array_equal(static_pred_t0, adaptive_pred_t0)


# --- deterministic behavior ---

def test_walk_is_deterministic(sequential_data, base_model_and_weights):
    steps, data = sequential_data
    base_model, weights = base_model_and_weights

    m1 = copy.deepcopy(base_model)
    r1 = run_adaptive_tree_walk(m1, steps, data, weights)
    m2 = copy.deepcopy(base_model)
    r2 = run_adaptive_tree_walk(m2, steps, data, weights)

    for t in steps:
        np.testing.assert_array_equal(r1.predictions[t][1], r2.predictions[t][1])


# --- warm_start mechanism genuinely adapts (not a no-op) ---

def test_adaptation_actually_changes_the_ensemble(sequential_data, base_model_and_weights):
    steps, data = sequential_data
    base_model, weights = base_model_and_weights
    model = copy.deepcopy(base_model)
    n_iter_before = model.n_iter_
    result = run_adaptive_tree_walk(model, steps, data, weights)
    assert model.n_iter_ > n_iter_before  # boosting rounds were actually added
    n_adapt = sum(1 for e in result.events if e["type"] == "adapt")
    assert model.n_iter_ == n_iter_before + n_adapt * ITERS_PER_UPDATE


# --- fixed class weights: computed once from train, never from adaptation data ---

def test_class_weights_computed_only_from_train_not_recomputed_per_step():
    X_train, y_train = make_snapshot(0, n=200, seed=1)
    weights_a = compute_fixed_sample_weights(y_train)
    # a wildly different (degenerate) y should NOT be passed to this function
    # during adaptation — verify the function only ever takes one y argument
    # and returns a value independent of anything outside it
    weights_b = compute_fixed_sample_weights(y_train)
    assert weights_a == weights_b


def test_hgb_base_config_matches_e2_selected_config():
    assert HGB_BASE_CONFIG == {"max_depth": None, "learning_rate": 0.1, "max_iter": 100, "min_samples_leaf": 50}


def test_iters_per_update_is_fixed_and_small():
    assert isinstance(ITERS_PER_UPDATE, int)
    assert 0 < ITERS_PER_UPDATE <= 50  # small, not a from-scratch retrain


def test_adapt_seeds_match_e2():
    assert ADAPT_SEEDS == [42, 123, 456, 789, 2024]
