"""Load the Phase 1 processed cache (node_table_meta.csv + node_features.npz)
into a form baselines and the GNN can both consume.

This module reads Phase 1 outputs only — it does not re-derive anything
from the raw CSVs, so it stays consistent with whatever chronological split
and label encoding Phase 1 already validated.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_node_table_with_features(processed_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Return (meta_df with f1..f165 columns attached, feature_cols).

    Row order in node_table_meta.csv and node_features.npz was written from
    the same dataframe in scripts/audit_dataset.py, so they realign by
    position — asserted here rather than assumed.
    """
    meta = pd.read_csv(processed_dir / "node_table_meta.csv")
    npz = np.load(processed_dir / "node_features.npz", allow_pickle=True)
    x = npz["x"]
    feature_cols = [str(c) for c in npz["feature_cols"]]

    if len(meta) != x.shape[0]:
        raise ValueError(
            f"node_table_meta.csv has {len(meta)} rows but node_features.npz "
            f"has {x.shape[0]} — Phase 1 cache is inconsistent, rerun "
            f"scripts/audit_dataset.py"
        )

    feat_df = pd.DataFrame(x, columns=feature_cols, index=meta.index)
    full = pd.concat([meta, feat_df], axis=1)
    logger.info(
        "Loaded processed node table: %d rows, %d feature columns", len(full), len(feature_cols)
    )
    return full, feature_cols


def get_labeled_split(
    node_table: pd.DataFrame,
    feature_cols: list[str],
    split_name: str,
    include_time_step: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, time_step) for the labeled rows of one split.

    Unknown-label rows (label == -1) are excluded — they must never enter
    supervised training or evaluation. X excludes txId always; excludes
    time_step unless include_time_step=True (see docs/EXPERIMENTS.md for
    why time_step is kept out of the primary baseline).
    """
    subset = node_table[(node_table["time_split"] == split_name) & (node_table["is_labeled"])]
    if subset.empty:
        raise ValueError(f"No labeled rows found for split={split_name}")
    cols = feature_cols + (["time_step"] if include_time_step else [])
    X = subset[cols].to_numpy(dtype=np.float32)
    y = subset["label"].to_numpy(dtype=np.int64)
    time_step = subset["time_step"].to_numpy(dtype=np.int64)
    return X, y, time_step
