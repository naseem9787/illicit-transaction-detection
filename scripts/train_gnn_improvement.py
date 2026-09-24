"""Phase 5 training entry point: controlled static GNN improvement study.

Usage:
    python -m scripts.train_gnn_improvement --config config.yaml

Trains a small, pre-declared grid of 16 configurations (8 "Tuned GCN" +
8 GraphSAGE, see CANDIDATE_GRID below) using the exact same training
function Phase 3 used for the basic GCN (src/training/gnn_training.py::
train_gcn is architecture-agnostic — it only calls `model(x, edge_index)`
and never references GCN-specific internals, so it is reused unmodified
here for GraphSAGE too). Model checkpoint selection within each config is
by validation PR-AUC (train_gcn's existing behavior, unchanged). The
single OVERALL winner across all 16 configs is also selected by
validation PR-AUC — never by test performance.

Epochs are fixed at 200 for every config, identical to the frozen Phase 3
basic-GCN reference, so the comparison isn't confounded by a shorter
training budget for the new configs. This was decided from a timing probe
(hidden=64 ~71s/200ep, hidden=128 ~137s/200ep on this machine, ~28 min for
all 16 configs) BEFORE running the grid, per the Phase 5 brief's
"predeclared subset" allowance — the subset actually used is the full
16-config grid; only the epoch budget was fixed in advance, not reduced.

CRITICAL: test data is never touched anywhere in this script except once,
right at the end, for the single already-selected winner. The 15
non-winning configs never have their test performance computed at all —
not even for reporting — so there is no way to (even inadvertently) pick
a "best test configuration."

Saves:
    results/metrics/gnn_improvement_train_manifest.json  (all 16 configs)
    results/models/gnn_improvement_selected.pt
    results/metrics/predictions/gnn_improvement_val.npz
    results/metrics/predictions/gnn_improvement_test.npz  (selected model only)

Does not modify any Phase 1/2/3/4 file.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import load_node_table_with_features
from src.data.loader import load_edges
from src.evaluation.metrics import compute_binary_metrics, select_threshold_maximizing_f1
from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE
from src.training.gnn_training import build_split_batches, predict_labeled, train_gcn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_gnn_improvement")

EPOCHS = 200  # fixed, matches the frozen Phase 3 basic GCN's budget
WEIGHT_DECAY = 5e-4  # fixed across the whole grid, per the brief
SEED = 42


@dataclass
class GridConfig:
    architecture: str  # "tuned_gcn" | "graphsage"
    hidden_channels: int
    dropout: float
    lr: float


def build_grid() -> list[GridConfig]:
    """Small, pre-declared grid: 2 (hidden) x 2 (dropout) x 2 (lr) = 8
    configs per architecture, 16 total. Declared once, in full, before any
    training happens."""
    grid = []
    for architecture in ("tuned_gcn", "graphsage"):
        for hidden in (64, 128):
            for dropout in (0.2, 0.5):
                for lr in (0.003, 0.01):
                    grid.append(GridConfig(architecture, hidden, dropout, lr))
    return grid


def make_model(cfg: GridConfig, in_channels: int):
    if cfg.architecture == "tuned_gcn":
        return GCN(in_channels=in_channels, hidden_channels=cfg.hidden_channels, dropout=cfg.dropout)
    elif cfg.architecture == "graphsage":
        return GraphSAGE(in_channels=in_channels, hidden_channels=cfg.hidden_channels, dropout=cfg.dropout)
    raise ValueError(f"unknown architecture {cfg.architecture!r}")


def main(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    project_root = Path(config_path).resolve().parent
    raw_dir = project_root / cfg["dataset"]["raw_dir"]
    processed_dir = project_root / cfg["dataset"]["processed_dir"]
    models_dir = project_root / cfg["paths"]["models_dir"]
    metrics_dir = project_root / cfg["paths"]["metrics_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading processed node table + raw edges")
    node_table, feature_cols = load_node_table_with_features(processed_dir)
    edges = load_edges(raw_dir / cfg["dataset"]["edges_file"])
    in_channels = len(feature_cols)

    logger.info("Building train/val/test batches (test batch built now, evaluated only once, at the end, only for the winner)")
    batches = build_split_batches(node_table, edges, feature_cols, cfg["split"])
    train_batch, val_batch, test_batch = batches["train"], batches["val"], batches["test"]

    grid = build_grid()
    logger.info("Running %d pre-declared configs (%d tuned_gcn + %d graphsage)",
                len(grid), sum(1 for c in grid if c.architecture == "tuned_gcn"),
                sum(1 for c in grid if c.architecture == "graphsage"))

    config_log = []
    best_overall = {"val_pr_auc": -1.0}

    for i, gc in enumerate(grid, 1):
        logger.info("[%d/%d] %s hidden=%d dropout=%.1f lr=%.4f", i, len(grid), gc.architecture, gc.hidden_channels, gc.dropout, gc.lr)
        model = make_model(gc, in_channels)
        result = train_gcn(model, train_batch, val_batch, seed=SEED, epochs=EPOCHS, lr=gc.lr, weight_decay=WEIGHT_DECAY)
        model.load_state_dict(result.best_state_dict)

        y_val, p_val, _ = predict_labeled(model, val_batch)
        val_threshold = select_threshold_maximizing_f1(y_val, p_val)
        val_metrics = compute_binary_metrics(y_val, p_val, val_threshold)

        record = {
            "architecture": gc.architecture,
            "hidden_channels": gc.hidden_channels,
            "dropout": gc.dropout,
            "learning_rate": gc.lr,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            "training_epochs": EPOCHS,
            "best_validation_epoch": result.best_epoch,
            "validation_pr_auc": result.best_val_pr_auc,
            "validation_f1": val_metrics["f1"],
            "selected_threshold_on_validation": val_threshold,
            "train_time_seconds": round(result.train_time_seconds, 2),
        }
        config_log.append(record)
        logger.info("  -> val PR-AUC=%.4f val F1=%.4f (threshold=%.4f)", result.best_val_pr_auc, val_metrics["f1"], val_threshold)

        if result.best_val_pr_auc > best_overall["val_pr_auc"]:
            best_overall = {
                "val_pr_auc": result.best_val_pr_auc,
                "config": gc,
                "state_dict": result.best_state_dict,
                "threshold": val_threshold,
                "record": record,
            }

    logger.info(
        "Selected overall (validation PR-AUC only): %s hidden=%d dropout=%.1f lr=%.4f (val PR-AUC=%.4f)",
        best_overall["config"].architecture, best_overall["config"].hidden_channels,
        best_overall["config"].dropout, best_overall["config"].lr, best_overall["val_pr_auc"],
    )

    # ---- Test evaluated exactly ONCE, only for the already-selected winner ----
    winner_model = make_model(best_overall["config"], in_channels)
    winner_model.load_state_dict(best_overall["state_dict"])
    torch.save(winner_model.state_dict(), models_dir / "gnn_improvement_selected.pt")

    pred_dir = metrics_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    y_val, p_val, t_val = predict_labeled(winner_model, val_batch)
    np.savez_compressed(pred_dir / "gnn_improvement_val.npz", y_true=y_val, y_prob=p_val, time_step=t_val)
    y_test, p_test, t_test = predict_labeled(winner_model, test_batch)
    np.savez_compressed(pred_dir / "gnn_improvement_test.npz", y_true=y_test, y_prob=p_test, time_step=t_test)

    manifest = {
        "study": "Phase 5 controlled static GNN improvement (tuned GCN + GraphSAGE grid)",
        "grid_size": len(grid),
        "epochs_per_config": EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
        "selection_criterion": "validation PR-AUC (30-34), single overall winner across all 16 configs",
        "all_configs": config_log,
        "selected": best_overall["record"],
        "torch_version": torch.__version__,
    }
    with open(metrics_dir / "gnn_improvement_train_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info("Phase 5 training complete. Manifest at %s", metrics_dir / "gnn_improvement_train_manifest.json")
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    result = main(args.config)
    summary = {k: v for k, v in result.items() if k != "all_configs"}
    print(json.dumps(summary, indent=2, default=str))
