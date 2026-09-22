"""Load and validate the raw Elliptic Bitcoin dataset files.

The raw release ships three CSVs with no shared preprocessing applied by us:
  - elliptic_txs_classes.csv  : txId, class ('1'=illicit, '2'=licit, 'unknown')
  - elliptic_txs_edgelist.csv : txId1, txId2 (directed edge, no header issues)
  - elliptic_txs_features.csv : txId, time_step, f1..f165 (NO header row)

All validation checks here re-run the same checks performed manually during
the DATA_AUDIT.md investigation, so that any pipeline run fails loudly if a
different copy of the dataset (different size, different label encoding,
cross-time edges, etc.) is ever substituted in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

LABEL_ILLICIT = "1"
LABEL_LICIT = "2"
LABEL_UNKNOWN = "unknown"


@dataclass
class RawDataset:
    """Container for the three raw dataframes plus derived column names."""

    classes: pd.DataFrame
    edges: pd.DataFrame
    features: pd.DataFrame
    feature_cols: list[str]


def load_classes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected_cols = {"txId", "class"}
    if set(df.columns) != expected_cols:
        raise ValueError(
            f"elliptic_txs_classes.csv has unexpected columns {list(df.columns)}, "
            f"expected {expected_cols}"
        )
    df["class"] = df["class"].astype(str)
    logger.info("Loaded classes: %d rows", len(df))
    return df


def load_edges(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected_cols = {"txId1", "txId2"}
    if set(df.columns) != expected_cols:
        raise ValueError(
            f"elliptic_txs_edgelist.csv has unexpected columns {list(df.columns)}, "
            f"expected {expected_cols}"
        )
    logger.info("Loaded edges: %d rows", len(df))
    return df


def load_features(path: Path, num_raw_feature_columns: int = 165) -> pd.DataFrame:
    """Load the header-less feature file and assign explicit column names.

    Column 0 = txId, column 1 = time_step, columns 2..(1+num_raw_feature_columns)
    are anonymized, already-standardized numeric features (f1..fN).
    """
    df = pd.read_csv(path, header=None)
    expected_num_cols = 2 + num_raw_feature_columns
    if df.shape[1] != expected_num_cols:
        raise ValueError(
            f"elliptic_txs_features.csv has {df.shape[1]} columns, "
            f"expected {expected_num_cols} (txId + time_step + "
            f"{num_raw_feature_columns} features). Refusing to guess a layout."
        )
    feature_cols = [f"f{i}" for i in range(1, num_raw_feature_columns + 1)]
    df.columns = ["txId", "time_step"] + feature_cols
    logger.info(
        "Loaded features: %d rows, %d feature columns", len(df), len(feature_cols)
    )
    return df


def load_raw_dataset(
    raw_dir: Path,
    classes_file: str,
    edges_file: str,
    features_file: str,
    num_raw_feature_columns: int = 165,
) -> RawDataset:
    classes = load_classes(raw_dir / classes_file)
    edges = load_edges(raw_dir / edges_file)
    features = load_features(raw_dir / features_file, num_raw_feature_columns)
    feature_cols = [c for c in features.columns if c.startswith("f")]
    return RawDataset(classes=classes, edges=edges, features=features, feature_cols=feature_cols)


def validate_dataset(data: RawDataset) -> dict:
    """Re-run the structural checks from DATA_AUDIT.md against loaded data.

    Raises ValueError on any check that would silently corrupt downstream
    results (ID mismatch, unexpected label values, missing values, cross-time
    edges). Returns a report dict of everything checked, including checks
    that only warn rather than fail.
    """
    report: dict = {}

    # --- classes ---
    allowed_labels = {LABEL_ILLICIT, LABEL_LICIT, LABEL_UNKNOWN}
    bad_labels = set(data.classes["class"].unique()) - allowed_labels
    if bad_labels:
        raise ValueError(f"Unexpected label values found: {bad_labels}")
    dup_ids = data.classes["txId"].duplicated().sum()
    if dup_ids:
        raise ValueError(f"{dup_ids} duplicate txId rows in classes file")
    report["label_counts"] = data.classes["class"].value_counts().to_dict()

    # --- features ---
    if data.features.isnull().sum().sum() != 0:
        raise ValueError("Missing values found in features file")
    dup_feat_ids = data.features["txId"].duplicated().sum()
    if dup_feat_ids:
        raise ValueError(f"{dup_feat_ids} duplicate txId rows in features file")
    report["num_time_steps"] = int(data.features["time_step"].nunique())
    report["time_step_range"] = (
        int(data.features["time_step"].min()),
        int(data.features["time_step"].max()),
    )

    # --- ID consistency across files ---
    class_ids = set(data.classes["txId"])
    feat_ids = set(data.features["txId"])
    if class_ids != feat_ids:
        raise ValueError(
            f"txId sets differ between classes and features: "
            f"{len(class_ids - feat_ids)} only in classes, "
            f"{len(feat_ids - class_ids)} only in features"
        )
    edge_ids = set(data.edges["txId1"]) | set(data.edges["txId2"])
    unknown_edge_ids = edge_ids - feat_ids
    if unknown_edge_ids:
        raise ValueError(
            f"{len(unknown_edge_ids)} txIds referenced by edges but absent "
            f"from features/classes"
        )

    # --- edges: duplicates, self-loops ---
    dup_edges = data.edges.duplicated().sum()
    self_loops = (data.edges["txId1"] == data.edges["txId2"]).sum()
    report["duplicate_edges"] = int(dup_edges)
    report["self_loops"] = int(self_loops)
    if dup_edges or self_loops:
        logger.warning(
            "Found %d duplicate edges and %d self-loops (audit found 0 in "
            "the reference copy — this dataset differs)",
            dup_edges,
            self_loops,
        )

    # --- edges never cross time steps (this assumption is load-bearing for
    # graph_builder.py, so it must fail loudly rather than warn if violated) ---
    tstep_map = data.features.set_index("txId")["time_step"]
    t1 = data.edges["txId1"].map(tstep_map)
    t2 = data.edges["txId2"].map(tstep_map)
    cross_time_edges = int((t1 != t2).sum())
    report["cross_time_edges"] = cross_time_edges
    if cross_time_edges:
        raise ValueError(
            f"{cross_time_edges} edges cross time steps in this copy of the "
            f"dataset. graph_builder.py assumes per-time-step snapshot "
            f"construction and must be revisited before proceeding."
        )

    logger.info("Validation passed: %s", report)
    return report
