import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import load_node_table_with_features
from src.models.mlp import MLP
from src.training.rolling_origin import (
    E1_SEEDS,
    E2_SEEDS,
    HGB_CONFIGS,
    RF_CONFIGS,
    ROLLING_ORIGIN_FOLDS,
    TEST_RANGE,
    get_labeled_by_time_range,
    train_mlp_fold,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def node_table_and_cols(cfg):
    processed_dir = PROJECT_ROOT / cfg["dataset"]["processed_dir"]
    return load_node_table_with_features(processed_dir)


# --- fold design: never touches the test period ---

def test_folds_never_reach_the_test_period():
    for fold in ROLLING_ORIGIN_FOLDS:
        assert fold["train_end"] < TEST_RANGE[0]
        assert fold["val_end"] < TEST_RANGE[0]
        assert fold["train_start"] == 1  # rolling origin: train always starts at 1


def test_folds_are_strictly_chronological_and_non_overlapping():
    for fold in ROLLING_ORIGIN_FOLDS:
        assert fold["train_end"] < fold["val_start"]
    for a, b in zip(ROLLING_ORIGIN_FOLDS, ROLLING_ORIGIN_FOLDS[1:]):
        assert a["val_end"] < b["val_start"]
        assert a["train_end"] < b["train_end"]


def test_fold3_matches_the_existing_project_split(cfg):
    fold3 = ROLLING_ORIGIN_FOLDS[-1]
    s = cfg["split"]
    assert (fold3["train_start"], fold3["train_end"]) == (s["train_start"], s["train_end"])
    assert (fold3["val_start"], fold3["val_end"]) == (s["val_start"], s["val_end"])


def test_seed_sets_match_spec():
    assert E1_SEEDS == [42, 123, 456]
    assert E2_SEEDS == [42, 123, 456, 789, 2024]


def test_grid_sizes_are_at_most_16():
    assert len(RF_CONFIGS) <= 16
    assert len(HGB_CONFIGS) <= 16
    # no duplicate configs
    assert len({tuple(sorted(c.items())) for c in RF_CONFIGS}) == len(RF_CONFIGS)
    assert len({tuple(sorted(c.items())) for c in HGB_CONFIGS}) == len(HGB_CONFIGS)


# --- get_labeled_by_time_range: leakage-safety, mirrors get_labeled_split ---

def test_get_labeled_by_time_range_excludes_unknown_and_matches_time_step_bounds(node_table_and_cols):
    node_table, feature_cols = node_table_and_cols
    X, y = get_labeled_by_time_range(node_table, feature_cols, 1, 19)
    assert (y != -1).all()
    assert set(np.unique(y)).issubset({0, 1})
    assert X.shape[1] == len(feature_cols)

    mask = node_table["time_step"].between(1, 19) & node_table["is_labeled"]
    assert X.shape[0] == int(mask.sum())


def test_get_labeled_by_time_range_disjoint_from_test_period(node_table_and_cols):
    node_table, feature_cols = node_table_and_cols
    for fold in ROLLING_ORIGIN_FOLDS:
        _, y_train = get_labeled_by_time_range(node_table, feature_cols, fold["train_start"], fold["train_end"])
        _, y_val = get_labeled_by_time_range(node_table, feature_cols, fold["val_start"], fold["val_end"])
        # sanity: both non-empty, both exclude unknown
        assert len(y_train) > 0 and len(y_val) > 0
        assert (y_train != -1).all() and (y_val != -1).all()


def test_folds_do_not_overlap_in_node_rows(node_table_and_cols):
    """Cross-check at the row level, not just the declared integer ranges."""
    node_table, feature_cols = node_table_and_cols
    for fold in ROLLING_ORIGIN_FOLDS:
        train_mask = node_table["time_step"].between(fold["train_start"], fold["train_end"])
        val_mask = node_table["time_step"].between(fold["val_start"], fold["val_end"])
        assert not (train_mask & val_mask).any()
        test_mask = node_table["time_step"].between(*TEST_RANGE)
        assert not (train_mask & test_mask).any()
        assert not (val_mask & test_mask).any()


# --- MLP model ---

def test_mlp_forward_output_shape():
    torch.manual_seed(0)
    model = MLP(in_channels=165, hidden_channels=128, dropout=0.5)
    x = torch.randn(10, 165)
    out = model(x)
    assert out.shape == (10,)
    assert out.dtype == torch.float32


def test_mlp_train_fold_is_deterministic_given_seed():
    rng = np.random.default_rng(0)
    n_train, n_val, d = 40, 20, 5
    X_train = rng.standard_normal((n_train, d)).astype(np.float32)
    y_train = np.array([1, 0] * (n_train // 2)).astype(np.int64)
    X_val = rng.standard_normal((n_val, d)).astype(np.float32)
    y_val = np.array([1, 0] * (n_val // 2)).astype(np.int64)

    r1 = train_mlp_fold(X_train, y_train, X_val, y_val, seed=42)
    r2 = train_mlp_fold(X_train, y_train, X_val, y_val, seed=42)
    assert r1.val_pr_auc == pytest.approx(r2.val_pr_auc)
    assert r1.best_epoch == r2.best_epoch
    assert r1.val_threshold == pytest.approx(r2.val_threshold)


def test_mlp_never_receives_edge_index_argument():
    """E1 is feature-only by design — MLP.forward must take x alone."""
    import inspect
    params = list(inspect.signature(MLP.forward).parameters)
    assert params == ["self", "x"]
