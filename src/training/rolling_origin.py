"""Phase 8: rolling-origin validation for E1 (MLP) and E2 (tree benchmark).

The existing config.yaml split (train 1-29 / val 30-34 / test 35-49) is a
single fixed origin. To pick an architecture/hyperparameters more robustly
before ever touching test, this module builds three chronological folds,
each strictly earlier than the next, and never touching t=35-49:

    fold 1: train  1-19, validate 20-24
    fold 2: train  1-24, validate 25-29
    fold 3: train  1-29, validate 30-34   (= the existing project split)

This file adds new functionality only — it does not modify
src/data/dataset.py, src/training/gnn_training.py, src/baselines/*.py, or
any Phase 1-7 file. `get_labeled_by_time_range` is the rolling-origin
analogue of dataset.py::get_labeled_split (which filters by the fixed
time_split column); this filters by an explicit inclusive time_step range
instead, so arbitrary fold boundaries can be expressed without touching the
existing time_split assignment.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score

from src.evaluation.metrics import compute_binary_metrics, select_threshold_maximizing_f1
from src.models.mlp import MLP
from src.training.gnn_training import compute_pos_weight

logger = logging.getLogger(__name__)

TEST_RANGE = (35, 49)  # must never appear in any fold below

ROLLING_ORIGIN_FOLDS = [
    {"name": "fold1", "train_start": 1, "train_end": 19, "val_start": 20, "val_end": 24},
    {"name": "fold2", "train_start": 1, "train_end": 24, "val_start": 25, "val_end": 29},
    {"name": "fold3", "train_start": 1, "train_end": 29, "val_start": 30, "val_end": 34},
]

for _f in ROLLING_ORIGIN_FOLDS:
    assert _f["train_end"] < _f["val_start"], f"{_f['name']}: train/val overlap"
    assert _f["val_end"] < TEST_RANGE[0], f"{_f['name']}: val range reaches into the test period"


def get_labeled_by_time_range(
    node_table: pd.DataFrame, feature_cols: list[str], start: int, end: int
) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) for labeled rows with time_step in [start, end] (inclusive).

    Mirrors src/data/dataset.py::get_labeled_split's leakage discipline
    (unknown-label rows, label == -1, are always excluded) but filters by an
    explicit time_step range instead of the fixed time_split column, so
    rolling-origin folds can be expressed without touching Phase 1's split
    assignment.
    """
    mask = node_table["time_step"].between(start, end) & node_table["is_labeled"]
    subset = node_table[mask]
    if subset.empty:
        raise ValueError(f"No labeled rows found for time_step in [{start}, {end}]")
    X = subset[feature_cols].to_numpy(dtype=np.float32)
    y = subset["label"].to_numpy(dtype=np.int64)
    return X, y


MLP_HIDDEN = 128
MLP_DROPOUT = 0.5
MLP_LR = 0.01
MLP_WEIGHT_DECAY = 5e-4
MLP_EPOCHS = 200  # same epoch budget as the Phase 5 GraphSAGE grid
E1_SEEDS = [42, 123, 456]


@dataclass
class MLPFoldResult:
    fold: str
    seed: int
    best_epoch: int
    val_pr_auc: float
    val_f1: float
    val_threshold: float
    train_time_seconds: float


def train_mlp_fold(X_train, y_train, X_val, y_val, seed: int) -> MLPFoldResult:
    """One MLP fit on one fold with one seed. Checkpoint selection is by
    validation PR-AUC every epoch, mirroring src/training/gnn_training.py::
    train_gcn's loop (same optimizer, same epoch budget, same pos_weight
    convention) but without a graph (this model takes x only)."""
    torch.manual_seed(seed)
    model = MLP(in_channels=X_train.shape[1], hidden_channels=MLP_HIDDEN, dropout=MLP_DROPOUT)

    pos_weight = torch.tensor(compute_pos_weight(y_train), dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY)

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train).float()
    X_val_t = torch.from_numpy(X_val)

    best_state, best_epoch, best_val_pr_auc = None, -1, -1.0
    t0 = time.time()
    for epoch in range(1, MLP_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(model(X_val_t)).numpy()
        val_pr_auc = float(average_precision_score(y_val, val_prob))
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(X_val_t)).numpy()
    val_threshold = select_threshold_maximizing_f1(y_val, val_prob)  # validation only
    val_f1 = compute_binary_metrics(y_val, val_prob, val_threshold)["f1"]

    return MLPFoldResult(
        fold="", seed=seed, best_epoch=best_epoch, val_pr_auc=best_val_pr_auc,
        val_f1=val_f1, val_threshold=val_threshold, train_time_seconds=round(train_time, 2),
    )


def run_e1_mlp(node_table: pd.DataFrame, feature_cols: list[str]) -> list[dict]:
    results = []
    for fold in ROLLING_ORIGIN_FOLDS:
        X_train, y_train = get_labeled_by_time_range(node_table, feature_cols, fold["train_start"], fold["train_end"])
        X_val, y_val = get_labeled_by_time_range(node_table, feature_cols, fold["val_start"], fold["val_end"])
        for seed in E1_SEEDS:
            r = train_mlp_fold(X_train, y_train, X_val, y_val, seed)
            r.fold = fold["name"]
            logger.info("[E1][%s][seed=%d] val_pr_auc=%.4f val_f1=%.4f best_epoch=%d",
                        fold["name"], seed, r.val_pr_auc, r.val_f1, r.best_epoch)
            results.append({
                "fold": fold["name"], "seed": seed, "n_train": int(len(y_train)), "n_val": int(len(y_val)),
                "best_epoch": r.best_epoch, "val_pr_auc": r.val_pr_auc, "val_f1": r.val_f1,
                "val_threshold": r.val_threshold, "train_time_seconds": r.train_time_seconds,
            })
    return results


E2_SEEDS = [42, 123, 456, 789, 2024]

# Predeclared, exactly 16 configurations each (not a full cartesian grid —
# hand-picked to cover all four axes without an unbounded search).
RF_CONFIGS = [
    {"max_depth": 8, "min_samples_leaf": 1, "max_features": "sqrt", "n_estimators": 200},
    {"max_depth": 8, "min_samples_leaf": 1, "max_features": "log2", "n_estimators": 300},
    {"max_depth": 8, "min_samples_leaf": 5, "max_features": "sqrt", "n_estimators": 300},
    {"max_depth": 8, "min_samples_leaf": 5, "max_features": "log2", "n_estimators": 200},
    {"max_depth": 16, "min_samples_leaf": 1, "max_features": "sqrt", "n_estimators": 300},
    {"max_depth": 16, "min_samples_leaf": 1, "max_features": "log2", "n_estimators": 200},
    {"max_depth": 16, "min_samples_leaf": 5, "max_features": "sqrt", "n_estimators": 200},
    {"max_depth": 16, "min_samples_leaf": 5, "max_features": "log2", "n_estimators": 300},
    {"max_depth": 24, "min_samples_leaf": 1, "max_features": "sqrt", "n_estimators": 300},
    {"max_depth": 24, "min_samples_leaf": 1, "max_features": "log2", "n_estimators": 200},
    {"max_depth": 24, "min_samples_leaf": 5, "max_features": "sqrt", "n_estimators": 200},
    {"max_depth": 24, "min_samples_leaf": 5, "max_features": "log2", "n_estimators": 300},
    {"max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt", "n_estimators": 300},
    {"max_depth": None, "min_samples_leaf": 1, "max_features": "log2", "n_estimators": 200},
    {"max_depth": None, "min_samples_leaf": 5, "max_features": "sqrt", "n_estimators": 200},
    {"max_depth": None, "min_samples_leaf": 5, "max_features": "log2", "n_estimators": 300},
]
assert len(RF_CONFIGS) == 16

HGB_CONFIGS = [
    {"max_depth": None, "learning_rate": 0.05, "max_iter": 100, "min_samples_leaf": 20},
    {"max_depth": None, "learning_rate": 0.05, "max_iter": 200, "min_samples_leaf": 50},
    {"max_depth": None, "learning_rate": 0.1, "max_iter": 100, "min_samples_leaf": 50},
    {"max_depth": None, "learning_rate": 0.1, "max_iter": 200, "min_samples_leaf": 20},
    {"max_depth": None, "learning_rate": 0.2, "max_iter": 100, "min_samples_leaf": 20},
    {"max_depth": None, "learning_rate": 0.2, "max_iter": 200, "min_samples_leaf": 50},
    {"max_depth": 6, "learning_rate": 0.05, "max_iter": 100, "min_samples_leaf": 20},
    {"max_depth": 6, "learning_rate": 0.05, "max_iter": 200, "min_samples_leaf": 50},
    {"max_depth": 6, "learning_rate": 0.1, "max_iter": 100, "min_samples_leaf": 50},
    {"max_depth": 6, "learning_rate": 0.1, "max_iter": 200, "min_samples_leaf": 20},
    {"max_depth": 6, "learning_rate": 0.2, "max_iter": 200, "min_samples_leaf": 20},
    {"max_depth": 10, "learning_rate": 0.05, "max_iter": 100, "min_samples_leaf": 50},
    {"max_depth": 10, "learning_rate": 0.05, "max_iter": 200, "min_samples_leaf": 20},
    {"max_depth": 10, "learning_rate": 0.1, "max_iter": 100, "min_samples_leaf": 20},
    {"max_depth": 10, "learning_rate": 0.1, "max_iter": 200, "min_samples_leaf": 50},
    {"max_depth": 10, "learning_rate": 0.2, "max_iter": 100, "min_samples_leaf": 50},
]
assert len(HGB_CONFIGS) == 16


def _fit_eval(model, X_train, y_train, X_val, y_val) -> dict:
    model.fit(X_train, y_train)
    val_prob = model.predict_proba(X_val)[:, 1]
    threshold = select_threshold_maximizing_f1(y_val, val_prob)  # validation only
    m = compute_binary_metrics(y_val, val_prob, threshold)
    return {"val_pr_auc": m["pr_auc"], "val_f1": m["f1"], "val_threshold": threshold}


def run_e2_trees(node_table: pd.DataFrame, feature_cols: list[str]) -> dict:
    fold_data = {}
    for fold in ROLLING_ORIGIN_FOLDS:
        X_train, y_train = get_labeled_by_time_range(node_table, feature_cols, fold["train_start"], fold["train_end"])
        X_val, y_val = get_labeled_by_time_range(node_table, feature_cols, fold["val_start"], fold["val_end"])
        fold_data[fold["name"]] = (X_train, y_train, X_val, y_val)

    def run_grid(name, configs, build_model):
        rows = []
        for cfg_idx, cfg in enumerate(configs):
            for fold in ROLLING_ORIGIN_FOLDS:
                X_train, y_train, X_val, y_val = fold_data[fold["name"]]
                for seed in E2_SEEDS:
                    model = build_model(cfg, seed)
                    r = _fit_eval(model, X_train, y_train, X_val, y_val)
                    rows.append({"model": name, "config_index": cfg_idx, **cfg, "fold": fold["name"],
                                "seed": seed, "n_train": int(len(y_train)), "n_val": int(len(y_val)), **r})
            logger.info("[E2][%s] config %d/%d done: %s", name, cfg_idx + 1, len(configs), cfg)
        return rows

    rf_rows = run_grid("random_forest", RF_CONFIGS, lambda cfg, seed: RandomForestClassifier(
        n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"], max_features=cfg["max_features"],
        class_weight="balanced", random_state=seed, n_jobs=-1))

    hgb_rows = run_grid("hist_gradient_boosting", HGB_CONFIGS, lambda cfg, seed: HistGradientBoostingClassifier(
        max_depth=cfg["max_depth"], learning_rate=cfg["learning_rate"], max_iter=cfg["max_iter"],
        min_samples_leaf=cfg["min_samples_leaf"], class_weight="balanced", random_state=seed))

    return {"random_forest": rf_rows, "hist_gradient_boosting": hgb_rows}
