"""Phase 3 training entry point: basic 2-layer GCN.

Usage:
    python -m scripts.train_gnn --config config.yaml

Loads the Phase 1 processed cache (node_table_meta.csv + node_features.npz)
for features/labels/split, and re-reads the raw edge list directly (Phase 1
never caches edges to disk — see docs/DATA_AUDIT.md / graph_builder.py),
then builds one block-diagonal graph batch per split and trains a GCN with
validation-only checkpoint selection and no test-set access at all.

Saves:
    results/models/gcn.pt
    results/metrics/predictions/gcn_{split}.npz  (y_true, y_prob, time_step)
    results/metrics/gcn_train_manifest.json

Does not touch any Phase 2 file (train_run_manifest.json, baseline_results.json,
*.joblib, logreg/random_forest predictions) — this is a separate model with
its own manifest, by design (see docs/EXPERIMENTS.md Phase 3 section).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import load_node_table_with_features
from src.data.loader import load_edges
from src.models.gcn import GCN
from src.training.gnn_training import build_split_batches, predict_labeled, train_gcn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_gnn")

HIDDEN_CHANNELS = 64
DROPOUT = 0.5
EPOCHS = 200
LR = 0.01
WEIGHT_DECAY = 5e-4


def save_predictions(model: GCN, splits: dict, pred_dir: Path) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    for split_name, batch in splits.items():
        y_true, y_prob, time_step = predict_labeled(model, batch)
        np.savez_compressed(
            pred_dir / f"gcn_{split_name}.npz", y_true=y_true, y_prob=y_prob, time_step=time_step
        )


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
    seed = cfg["seed"]

    logger.info("Loading processed node table + features")
    node_table, feature_cols = load_node_table_with_features(processed_dir)

    logger.info("Loading raw edges")
    edges = load_edges(raw_dir / cfg["dataset"]["edges_file"])

    logger.info("Building per-split graph batches")
    t0 = time.time()
    batches = build_split_batches(node_table, edges, feature_cols, cfg["split"])
    t_batches = time.time() - t0

    torch.manual_seed(seed)
    model = GCN(in_channels=len(feature_cols), hidden_channels=HIDDEN_CHANNELS, dropout=DROPOUT)

    logger.info(
        "Training GCN: hidden=%d dropout=%.2f epochs=%d lr=%.4f weight_decay=%.0e",
        HIDDEN_CHANNELS, DROPOUT, EPOCHS, LR, WEIGHT_DECAY,
    )
    result = train_gcn(
        model, batches["train"], batches["val"], seed,
        epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
    )
    model.load_state_dict(result.best_state_dict)

    torch.save(model.state_dict(), models_dir / "gcn.pt")
    save_predictions(model, batches, metrics_dir / "predictions")

    manifest = {
        "seed": seed,
        "feature_cols": feature_cols,
        "model": "GCN",
        "architecture": "GCNConv(in,hidden) -> ReLU -> Dropout -> GCNConv(hidden,1)",
        "hyperparameters": {
            "hidden_channels": HIDDEN_CHANNELS,
            "dropout": DROPOUT,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "Adam",
            "loss": "BCEWithLogitsLoss(pos_weight=n_neg/n_pos on train labels only)",
        },
        "torch_version": torch.__version__,
        "batch_build_time_seconds": round(t_batches, 3),
        "train_time_seconds": round(result.train_time_seconds, 3),
        "best_epoch": result.best_epoch,
        "best_val_pr_auc": result.best_val_pr_auc,
        "epoch_log": result.epoch_log,
    }
    with open(metrics_dir / "gcn_train_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info(
        "GCN training complete. Manifest written to %s", metrics_dir / "gcn_train_manifest.json"
    )
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    result = main(args.config)
    # epoch_log omitted from stdout dump (verbose); full log is in the manifest file
    summary = {k: v for k, v in result.items() if k != "epoch_log"}
    print(json.dumps(summary, indent=2, default=str))
