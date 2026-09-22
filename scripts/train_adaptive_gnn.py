"""Phase 4 training entry point: simulated delayed-feedback online
adaptation of the Phase 3 GCN.

Usage:
    python -m scripts.train_adaptive_gnn --config config.yaml

This does NOT retrain the base GCN — it loads the exact Phase 3 pretrained
checkpoint (results/models/gcn.pt) as the starting point for every walk in
this script, so Static GCN and Adaptive GCN start from identical weights
and the only experimental variable is whether the weights evolve during
the 35-49 walk.

Two stages:

1. Hyperparameter selection (validation 30-34 only, k=1). A small,
   pre-declared grid over (learning rate, gradient steps per update) is
   walked forward starting fresh from the pretrained checkpoint for each
   candidate; the winner is picked by pooled validation PR-AUC. The
   adapted weights produced during this search are discarded — only the
   (lr, grad_steps) pair survives into stage 2. See
   src/training/adaptive_gnn.py::select_adaptation_config.

2. Primary experiment (test 35-49, k=1): fresh pretrained weights, walked
   forward with the selected config. Saves predictions.

   Secondary sensitivity experiment (test 35-49, k=3, SAME selected
   lr/grad_steps): reported separately, never used to choose between k=1
   and k=3 (see docs/EXPERIMENTS.md Phase 4).

Saves:
    results/metrics/adaptive_hparam_selection.json
    results/metrics/predictions/adaptive_gcn_test.npz        (k=1, primary)
    results/metrics/predictions/adaptive_gcn_test_k3.npz     (k=3, sensitivity)
    results/metrics/adaptive_train_manifest.json

Does not modify any Phase 1/2/3 file.
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
from src.training.adaptive_gnn import (
    AdaptationConfig,
    build_per_step_data,
    pooled_predictions,
    run_adaptive_walk,
    select_adaptation_config,
)
from src.training.gnn_training import compute_pos_weight

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_adaptive_gnn")

HIDDEN_CHANNELS = 64
DROPOUT = 0.5

# Pre-declared, small (5 validation time steps -> keep the grid tiny to
# avoid overfitting the selection to noise). No replay dimension: replay
# is not implemented in this experiment (see docs/EXPERIMENTS.md).
CANDIDATE_GRID = [
    AdaptationConfig(lr=0.001, grad_steps=1, feedback_delay_k=1),
    AdaptationConfig(lr=0.001, grad_steps=3, feedback_delay_k=1),
    AdaptationConfig(lr=0.005, grad_steps=1, feedback_delay_k=1),
    AdaptationConfig(lr=0.005, grad_steps=3, feedback_delay_k=1),
]

K_SENSITIVITY = 3  # secondary, reported separately, not used for selection


def save_predictions(result, name: str, pred_dir: Path) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    y_true, y_prob, time_step = pooled_predictions(result)
    np.savez_compressed(pred_dir / f"{name}.npz", y_true=y_true, y_prob=y_prob, time_step=time_step)


def main(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    project_root = Path(config_path).resolve().parent
    raw_dir = project_root / cfg["dataset"]["raw_dir"]
    processed_dir = project_root / cfg["dataset"]["processed_dir"]
    models_dir = project_root / cfg["paths"]["models_dir"]
    metrics_dir = project_root / cfg["paths"]["metrics_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading processed node table + raw edges")
    node_table, feature_cols = load_node_table_with_features(processed_dir)
    edges = load_edges(raw_dir / cfg["dataset"]["edges_file"])

    split_cfg = cfg["split"]
    val_time_steps = list(range(split_cfg["val_start"], split_cfg["val_end"] + 1))
    test_time_steps = list(range(split_cfg["test_start"], split_cfg["test_end"] + 1))

    logger.info("Building per-step Data objects for val %s and test %s", val_time_steps, test_time_steps)
    val_data = build_per_step_data(node_table, edges, feature_cols, val_time_steps)
    test_data = build_per_step_data(node_table, edges, feature_cols, test_time_steps)

    # pos_weight: computed from TRAIN labels only (1-29), same convention as
    # Phase 3 pretraining. Fixed for the whole experiment — not recomputed
    # per adaptation step, for gradient stability given how small and
    # skewed a single test snapshot's labeled set can be.
    train_mask = (node_table["time_split"] == "train") & (node_table["is_labeled"])
    pos_weight = torch.tensor(compute_pos_weight(node_table.loc[train_mask, "label"].to_numpy()), dtype=torch.float32)
    logger.info("pos_weight (fixed, from train 1-29): %.4f", pos_weight.item())

    pretrained_state_dict = torch.load(models_dir / "gcn.pt", weights_only=True)
    in_channels = len(feature_cols)

    # ---- Stage 1: hyperparameter selection on validation (30-34), k=1 ----
    logger.info("Selecting adaptation hyperparameters on validation 30-34 (k=1)")
    best_config, selection_log = select_adaptation_config(
        pretrained_state_dict, in_channels, HIDDEN_CHANNELS, DROPOUT,
        val_time_steps, val_data, pos_weight, CANDIDATE_GRID,
    )
    with open(metrics_dir / "adaptive_hparam_selection.json", "w") as f:
        json.dump({
            "candidate_grid": [
                {"lr": c.lr, "grad_steps": c.grad_steps, "feedback_delay_k": c.feedback_delay_k}
                for c in CANDIDATE_GRID
            ],
            "selection_log": selection_log,
            "selected": {"lr": best_config.lr, "grad_steps": best_config.grad_steps},
            "selection_criterion": "pooled PR-AUC over validation 30-34 predictions, k=1",
            "replay": False,
            "replay_note": "Not implemented in this experiment (see docs/EXPERIMENTS.md Phase 4).",
        }, f, indent=2)

    # ---- Stage 2: primary experiment, test 35-49, k=1, fresh pretrained weights ----
    logger.info("Primary adaptive walk: test 35-49, k=1, lr=%.4f grad_steps=%d", best_config.lr, best_config.grad_steps)
    model_k1 = GCN(in_channels=in_channels, hidden_channels=HIDDEN_CHANNELS, dropout=DROPOUT)
    model_k1.load_state_dict(pretrained_state_dict)
    t0 = time.time()
    result_k1 = run_adaptive_walk(model_k1, test_time_steps, test_data, pos_weight, best_config)
    time_k1 = time.time() - t0
    save_predictions(result_k1, "adaptive_gcn_test", metrics_dir / "predictions")

    # ---- Secondary sensitivity experiment, test 35-49, k=3, SAME lr/grad_steps ----
    logger.info("Sensitivity adaptive walk: test 35-49, k=%d (same lr/grad_steps, not used for selection)", K_SENSITIVITY)
    config_k3 = AdaptationConfig(lr=best_config.lr, grad_steps=best_config.grad_steps, feedback_delay_k=K_SENSITIVITY)
    model_k3 = GCN(in_channels=in_channels, hidden_channels=HIDDEN_CHANNELS, dropout=DROPOUT)
    model_k3.load_state_dict(pretrained_state_dict)
    t0 = time.time()
    result_k3 = run_adaptive_walk(model_k3, test_time_steps, test_data, pos_weight, config_k3)
    time_k3 = time.time() - t0
    save_predictions(result_k3, "adaptive_gcn_test_k3", metrics_dir / "predictions")

    manifest = {
        "model": "Adaptive GCN (online weight adaptation, simulated delayed feedback)",
        "base_checkpoint": "results/models/gcn.pt (Phase 3 static GCN, unmodified)",
        "hidden_channels": HIDDEN_CHANNELS,
        "dropout": DROPOUT,
        "pos_weight_fixed_from_train": pos_weight.item(),
        "hyperparameter_selection": {
            "period": "validation 30-34",
            "candidate_grid_size": len(CANDIDATE_GRID),
            "selection_log": selection_log,
            "selected_lr": best_config.lr,
            "selected_grad_steps": best_config.grad_steps,
        },
        "replay": False,
        "primary_experiment": {
            "feedback_delay_k": 1,
            "test_time_steps": test_time_steps,
            "n_adapt_events": sum(1 for e in result_k1.events if e["type"] == "adapt"),
            "walk_time_seconds": round(time_k1, 3),
        },
        "sensitivity_experiment": {
            "feedback_delay_k": K_SENSITIVITY,
            "note": "Reported separately; NOT used to choose between k=1 and k=3.",
            "n_adapt_events": sum(1 for e in result_k3.events if e["type"] == "adapt"),
            "walk_time_seconds": round(time_k3, 3),
        },
        "torch_version": torch.__version__,
    }
    with open(metrics_dir / "adaptive_train_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info("Phase 4 training complete. Manifest at %s", metrics_dir / "adaptive_train_manifest.json")
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    result = main(args.config)
    print(json.dumps(result, indent=2, default=str))
