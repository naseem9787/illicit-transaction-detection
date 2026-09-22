"""Label encoding, merging, and chronological (leakage-safe) splitting.

Label encoding convention used throughout this project:
    1 = illicit
    0 = licit
   -1 = unknown / unlabeled  (NEVER treated as a legitimate training label)

This mapping is deliberately NOT {1, 2, NaN} as in the raw file, to make
"is this row usable for supervised loss" a single unambiguous check
(`label >= 0`) everywhere downstream instead of a string comparison repeated
in multiple places.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.loader import LABEL_ILLICIT, LABEL_LICIT, LABEL_UNKNOWN, RawDataset

logger = logging.getLogger(__name__)

RAW_TO_ENCODED = {LABEL_ILLICIT: 1, LABEL_LICIT: 0, LABEL_UNKNOWN: -1}


def encode_labels(classes: pd.DataFrame) -> pd.DataFrame:
    out = classes.copy()
    out["label"] = out["class"].map(RAW_TO_ENCODED)
    if out["label"].isnull().any():
        unmapped = out.loc[out["label"].isnull(), "class"].unique()
        raise ValueError(f"encode_labels found unmapped class values: {unmapped}")
    out["label"] = out["label"].astype(int)
    return out[["txId", "label"]]


def build_node_table(data: RawDataset) -> pd.DataFrame:
    """One row per transaction: txId, time_step, f1..fN, label."""
    labels = encode_labels(data.classes)
    nodes = data.features.merge(labels, on="txId", how="left", validate="one_to_one")
    if nodes["label"].isnull().any():
        raise ValueError("Some feature rows have no matching label after merge")
    return nodes


def assign_time_split(
    time_step: pd.Series,
    train_start: int,
    train_end: int,
    val_start: int,
    val_end: int,
    test_start: int,
    test_end: int,
) -> pd.Series:
    """Map each time_step value to 'train' / 'val' / 'test' / 'excluded'.

    Boundaries are inclusive. Any time_step outside all three ranges is
    marked 'excluded' rather than silently dropped, so gaps are visible.
    """
    conditions = [
        time_step.between(train_start, train_end),
        time_step.between(val_start, val_end),
        time_step.between(test_start, test_end),
    ]
    choices = ["train", "val", "test"]
    split = pd.Series(np.select(conditions, choices, default="excluded"), index=time_step.index)
    return split


def add_split_and_masks(nodes: pd.DataFrame, split_cfg: dict) -> pd.DataFrame:
    """Attach `time_split` (train/val/test/excluded) and `is_labeled` columns.

    Supervised masks for training/evaluation are `time_split == 'train' &
    is_labeled`, etc. — computed here explicitly rather than left implicit,
    so every downstream script uses the identical definition.
    """
    out = nodes.copy()
    out["time_split"] = assign_time_split(
        out["time_step"],
        split_cfg["train_start"],
        split_cfg["train_end"],
        split_cfg["val_start"],
        split_cfg["val_end"],
        split_cfg["test_start"],
        split_cfg["test_end"],
    )
    out["is_labeled"] = out["label"] >= 0

    n_excluded = (out["time_split"] == "excluded").sum()
    if n_excluded:
        logger.warning(
            "%d nodes fall outside the configured train/val/test time "
            "ranges and are excluded from all splits",
            n_excluded,
        )

    for split_name in ("train", "val", "test"):
        subset = out[out["time_split"] == split_name]
        n_labeled = int(subset["is_labeled"].sum())
        n_illicit = int((subset["label"] == 1).sum())
        logger.info(
            "%s split: %d nodes total, %d labeled (%d illicit, %.2f%% of labeled)",
            split_name,
            len(subset),
            n_labeled,
            n_illicit,
            100 * n_illicit / n_labeled if n_labeled else 0.0,
        )
    return out


def verify_no_leakage(nodes: pd.DataFrame, split_cfg: dict) -> None:
    """Assert the split boundaries are exactly respected and non-overlapping.

    This is a sanity check meant to run every pipeline invocation, not just
    in tests, since a silent off-by-one in the split config would otherwise
    only surface as an unexplained metric anomaly much later.
    """
    ranges = {
        "train": (split_cfg["train_start"], split_cfg["train_end"]),
        "val": (split_cfg["val_start"], split_cfg["val_end"]),
        "test": (split_cfg["test_start"], split_cfg["test_end"]),
    }
    seen_steps: dict[int, str] = {}
    for name, (start, end) in ranges.items():
        if start > end:
            raise ValueError(f"{name} split has start > end ({start} > {end})")
        for t in range(start, end + 1):
            if t in seen_steps:
                raise ValueError(
                    f"time_step {t} assigned to both '{seen_steps[t]}' and '{name}' — "
                    f"split ranges overlap"
                )
            seen_steps[t] = name

    for split_name, (start, end) in ranges.items():
        actual_range = nodes.loc[nodes["time_split"] == split_name, "time_step"]
        if len(actual_range) and (actual_range.min() < start or actual_range.max() > end):
            raise ValueError(
                f"{split_name} split contains time_step values outside "
                f"[{start}, {end}]: found range "
                f"[{actual_range.min()}, {actual_range.max()}]"
            )
