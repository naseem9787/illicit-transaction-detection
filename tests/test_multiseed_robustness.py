"""Phase 7 tests: multi-seed robustness of Static -> Adaptive GraphSAGE."""

import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_multiseed_robustness import evaluate_run, run_paths, summarize
from scripts.train_multiseed_robustness import (
    ADAPT_GRAD_STEPS, ADAPT_LR, DROPOUT, FEEDBACK_DELAY_K, FIXED_THRESHOLD, HIDDEN,
    REFERENCE_SEED, SEEDS, adaptation_config_for_seed, init_model, run_adaptive_for_seed,
    train_static_for_seed,
)
from src.data.graph_builder import Snapshot
from src.data.pyg_adapter import snapshot_to_data
from src.evaluation.metrics import select_threshold_maximizing_f1

ROOT = Path(__file__).resolve().parents[1]
MS_DIR = ROOT / "results" / "metrics" / "multiseed"
ALL_LABELS = ["seed_42_reference", "seed_123", "seed_456", "seed_789", "seed_2024", "seed_42_replicate"]

needs_artifacts = pytest.mark.skipif(not (MS_DIR / "manifest_seed_2024.json").exists(),
                                     reason="Phase 7 artifacts not generated yet")


def make_data(t, n=8, seed=0):
    rng = np.random.default_rng(seed + t)
    x = rng.standard_normal((n, 4)).astype(np.float32)
    y = np.array([1, 0] * (n // 2 + 1))[:n].astype(np.int64)
    ei = np.array([(i, i + 1) for i in range(n - 1)], dtype=np.int64).T
    return snapshot_to_data(Snapshot(time_step=t, node_ids=np.arange(n), x=x, y=y, edge_index=ei,
                                     is_labeled=np.ones(n, dtype=bool)))


# --- declared design ---

def test_seeds_are_exactly_the_declared_set():
    assert SEEDS == [42, 123, 456, 789, 2024]
    assert len(set(SEEDS)) == 5
    assert REFERENCE_SEED == 42


def test_no_retuning_configuration_is_identical_for_every_seed():
    cfgs = [adaptation_config_for_seed(s) for s in SEEDS]
    assert {(c.lr, c.grad_steps, c.feedback_delay_k, c.replay) for c in cfgs} == {(ADAPT_LR, ADAPT_GRAD_STEPS, FEEDBACK_DELAY_K, False)}
    assert (ADAPT_LR, ADAPT_GRAD_STEPS, FEEDBACK_DELAY_K) == (0.001, 1, 1)
    assert (HIDDEN, DROPOUT) == (128, 0.5)


def test_static_training_has_no_test_argument_and_config_takes_only_seed():
    params = inspect.signature(train_static_for_seed).parameters
    assert not any("test" in p for p in params)
    assert list(inspect.signature(adaptation_config_for_seed).parameters) == ["seed"]


# --- identical initialization ---

def test_init_is_a_pure_function_of_the_seed():
    a, b = init_model(123, 4).state_dict(), init_model(123, 4).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)
    c = init_model(456, 4).state_dict()
    assert not all(torch.equal(a[k], c[k]) for k in a)


@pytest.mark.parametrize("seed", SEEDS)
def test_static_and_adaptive_start_from_identical_weights(seed):
    data = {t: make_data(t) for t in (1, 2, 3)}
    state = {k: v.clone() for k, v in init_model(seed, 4).state_dict().items()}
    static = init_model(seed, 4)
    static.load_state_dict(state)
    static.eval()
    with torch.no_grad():
        static_p = torch.sigmoid(static(data[1].x, data[1].edge_index)).numpy()
    walk = run_adaptive_for_seed(seed, state, 4, [1, 2, 3], data, torch.tensor(1.0))
    assert walk.events[0]["type"] == "predict"
    np.testing.assert_array_equal(walk.predictions[1][1], static_p)
    # passing the state dict in must not have been mutated by adaptation
    assert all(torch.equal(state[k], init_model(seed, 4).state_dict()[k]) for k in state)


# --- leakage / immutability (per seed) ---

@pytest.mark.parametrize("seed", SEEDS)
def test_no_future_label_access_and_predictions_not_rewritten(seed):
    steps = [1, 2, 3, 4, 5]
    data = {t: make_data(t) for t in steps}
    state = init_model(seed, 4).state_dict()
    w1 = run_adaptive_for_seed(seed, state, 4, steps, data, torch.tensor(1.0))
    w2 = run_adaptive_for_seed(seed, state, 4, steps, data, torch.tensor(1.0))
    assert w1.predict_order == steps
    for ev in w1.events:
        if ev["type"] == "adapt":
            assert ev["source_time_step"] == ev["applied_before_predicting"] - FEEDBACK_DELAY_K
    for t in steps:
        np.testing.assert_array_equal(w1.predictions[t][1], w2.predictions[t][1])


# --- threshold ---

def test_threshold_fixed_at_0_898_and_never_selected_on_test(monkeypatch):
    assert FIXED_THRESHOLD == 0.898
    with open(ROOT / "results" / "metrics" / "gnn_improvement_train_manifest.json") as f:
        assert json.load(f)["selected"]["selected_threshold_on_validation"] == pytest.approx(0.898, abs=5e-4)
    import src.evaluation.metrics as mm
    monkeypatch.setattr(mm, "select_threshold_maximizing_f1",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no threshold selection on test")))
    if (MS_DIR / "manifest_seed_123.json").exists():
        out = evaluate_run(run_paths(ROOT, "seed_123"), FIXED_THRESHOLD)
        assert out["static"]["threshold"] == FIXED_THRESHOLD == out["adaptive"]["threshold"]


@needs_artifacts
def test_summary_records_fixed_threshold():
    with open(MS_DIR / "multiseed_summary.json") as f:
        assert json.load(f)["primary_threshold"] == 0.898


@needs_artifacts
@pytest.mark.parametrize("label", ["seed_123", "seed_456", "seed_789", "seed_2024", "seed_42_replicate"])
def test_own_threshold_came_from_validation_predictions_only(label):
    with open(MS_DIR / f"manifest_{label}.json") as f:
        stored = json.load(f)["own_validation_threshold"]
    v = np.load(MS_DIR / "predictions" / f"static_{label}_val.npz")
    assert select_threshold_maximizing_f1(v["y_true"], v["y_prob"]) == pytest.approx(stored)


# --- populations / fairness on the real artifacts ---

@needs_artifacts
def test_test_populations_match_between_static_adaptive_and_across_seeds():
    ref_y = None
    for label in ALL_LABELS:
        p = run_paths(ROOT, label)
        r = evaluate_run(p, FIXED_THRESHOLD)
        assert r["same_population"], label
        assert r["t35_identical"], label
        assert r["n_test"] == 16670
        y = np.load(p["static"])["y_true"]
        ref_y = y if ref_y is None else ref_y
        assert np.array_equal(y, ref_y), label


@needs_artifacts
def test_manifests_record_reproducibility_fields():
    for label in ["seed_123", "seed_456", "seed_789", "seed_2024", "seed_42_replicate"]:
        with open(MS_DIR / f"manifest_{label}.json") as f:
            m = json.load(f)
        for key in ("seed", "architecture", "hyperparameters", "checkpoint_selection_criterion",
                    "fixed_threshold_primary", "adaptation_config", "test_population_size"):
            assert key in m, (label, key)
        assert m["fixed_threshold_primary"] == 0.898
        assert m["adaptation_config"]["replay"] is False


def test_summarize_uses_sample_std():
    s = summarize([1.0, 2.0, 3.0, 4.0])
    assert s["mean"] == 2.5 and s["median"] == 2.5 and s["min"] == 1.0 and s["max"] == 4.0
    assert s["std"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))


# --- Phase 1-6 artifacts unchanged ---

def test_phase1_to_6_tracked_files_unmodified():
    try:
        out = subprocess.run(["git", "diff", "--name-only", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("git unavailable")
    changed = [p for p in out.splitlines() if p and "multiseed" not in p and not p.startswith("docs/")]
    assert changed == [], f"tracked Phase 1-6 files modified: {changed}"
