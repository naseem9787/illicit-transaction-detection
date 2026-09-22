import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.graph_builder import build_all_snapshots
from src.data.loader import load_raw_dataset, validate_dataset
from src.data.preprocessing import add_split_and_masks, build_node_table, verify_no_leakage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def raw_data(cfg):
    raw_dir = PROJECT_ROOT / cfg["dataset"]["raw_dir"]
    return load_raw_dataset(
        raw_dir,
        cfg["dataset"]["classes_file"],
        cfg["dataset"]["edges_file"],
        cfg["dataset"]["features_file"],
        cfg["dataset"]["num_raw_feature_columns"],
    )


@pytest.fixture(scope="module")
def node_table(raw_data, cfg):
    nodes = build_node_table(raw_data)
    nodes = add_split_and_masks(nodes, cfg["split"])
    return nodes


def test_raw_shapes(raw_data):
    assert raw_data.classes.shape == (203769, 2)
    assert raw_data.edges.shape == (234355, 2)
    assert raw_data.features.shape[0] == 203769
    assert len(raw_data.feature_cols) == 165


def test_label_distribution_matches_known_values(raw_data):
    counts = raw_data.classes["class"].value_counts().to_dict()
    assert counts["unknown"] == 157205
    assert counts["2"] == 42019
    assert counts["1"] == 4545


def test_validate_dataset_passes_and_finds_no_cross_time_edges(raw_data):
    report = validate_dataset(raw_data)
    assert report["cross_time_edges"] == 0
    assert report["duplicate_edges"] == 0
    assert report["self_loops"] == 0
    assert report["num_time_steps"] == 49


def test_label_encoding_no_row_lost_or_duplicated(node_table, raw_data):
    assert len(node_table) == len(raw_data.classes)
    assert node_table["label"].isin([-1, 0, 1]).all()
    assert int((node_table["label"] == 1).sum()) == 4545
    assert int((node_table["label"] == 0).sum()) == 42019
    assert int((node_table["label"] == -1).sum()) == 157205


def test_split_ranges_are_disjoint_and_exhaustive(cfg):
    s = cfg["split"]
    ranges = [
        set(range(s["train_start"], s["train_end"] + 1)),
        set(range(s["val_start"], s["val_end"] + 1)),
        set(range(s["test_start"], s["test_end"] + 1)),
    ]
    # pairwise disjoint
    assert ranges[0].isdisjoint(ranges[1])
    assert ranges[0].isdisjoint(ranges[2])
    assert ranges[1].isdisjoint(ranges[2])
    # exhaustive over the dataset's 49 time steps
    covered = ranges[0] | ranges[1] | ranges[2]
    assert covered == set(range(1, 50))


def test_verify_no_leakage_passes(node_table, cfg):
    verify_no_leakage(node_table, cfg["split"])  # should not raise


def test_verify_no_leakage_catches_overlap(node_table):
    bad_cfg = {
        "train_start": 1, "train_end": 30,
        "val_start": 25, "val_end": 34,  # overlaps train
        "test_start": 35, "test_end": 49,
    }
    with pytest.raises(ValueError, match="overlap"):
        verify_no_leakage(node_table, bad_cfg)


def test_no_time_step_leaks_across_split(node_table, cfg):
    s = cfg["split"]
    train_max = node_table.loc[node_table["time_split"] == "train", "time_step"].max()
    val_min = node_table.loc[node_table["time_split"] == "val", "time_step"].min()
    test_min = node_table.loc[node_table["time_split"] == "test", "time_step"].min()
    assert train_max <= s["train_end"]
    assert val_min >= s["val_start"]
    assert test_min >= s["test_start"]


def test_unknown_labels_never_marked_is_labeled(node_table):
    unknown_rows = node_table[node_table["label"] == -1]
    assert not unknown_rows["is_labeled"].any()


def test_snapshots_have_no_cross_time_edges_and_valid_local_indices(raw_data, node_table):
    snapshots = build_all_snapshots(node_table, raw_data.edges, raw_data.feature_cols, time_steps=[1, 13, 49])
    for t, snap in snapshots.items():
        n = len(snap.node_ids)
        if snap.edge_index.shape[1] > 0:
            assert snap.edge_index.min() >= 0
            assert snap.edge_index.max() < n
        assert snap.x.shape == (n, 165)
        assert snap.y.shape == (n,)


def test_snapshot_node_count_matches_time_step_filter(raw_data, node_table):
    snapshots = build_all_snapshots(node_table, raw_data.edges, raw_data.feature_cols, time_steps=[27])
    expected = int((node_table["time_step"] == 27).sum())
    assert len(snapshots[27].node_ids) == expected
