import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import get_labeled_split, load_node_table_with_features
from src.evaluation.metrics import compute_binary_metrics, select_threshold_maximizing_f1

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


def test_labeled_splits_exclude_unknown(node_table_and_cols):
    node_table, feature_cols = node_table_and_cols
    for split in ("train", "val", "test"):
        _, y, _ = get_labeled_split(node_table, feature_cols, split)
        assert (y != -1).all()
        assert set(np.unique(y)).issubset({0, 1})


def test_no_time_step_by_default(node_table_and_cols):
    node_table, feature_cols = node_table_and_cols
    X, _, _ = get_labeled_split(node_table, feature_cols, "train", include_time_step=False)
    assert X.shape[1] == len(feature_cols)


def test_with_time_step_adds_one_column(node_table_and_cols):
    node_table, feature_cols = node_table_and_cols
    X, _, _ = get_labeled_split(node_table, feature_cols, "train", include_time_step=True)
    assert X.shape[1] == len(feature_cols) + 1


def test_threshold_selection_uses_only_given_data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    p = rng.random(200)
    t1 = select_threshold_maximizing_f1(y, p)
    t2 = select_threshold_maximizing_f1(y, p)
    assert t1 == t2  # deterministic given the same inputs


def test_compute_binary_metrics_confusion_matrix_sums_to_n():
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.1, 0.6, 0.9, 0.4, 0.8])
    m = compute_binary_metrics(y, p, threshold=0.5)
    cm = m["confusion_matrix"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y)
