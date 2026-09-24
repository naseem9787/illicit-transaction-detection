"""Phase 6 training entry point: simulated delayed-feedback online adaptation
of the Phase 5 selected static GraphSAGE.

Usage:
    python -m scripts.train_adaptive_graphsage --config config.yaml

Identical protocol to Phase 4 (scripts/train_adaptive_gnn.py), with only the
base model changed. Static and adaptive models start from the exact same
checkpoint (results/models/gnn_improvement_selected.pt: GraphSAGE, hidden=128,
dropout=0.5). The feedback delay is a simulated experimental assumption, not
a dataset fact (docs/LIMITATIONS.md L2).

Stage 1: pre-declared 4-config grid, validation 30-34 only, k=1, pooled PR-AUC.
Stage 2: primary walk on test 35-49, k=1, selected config.
         Sensitivity walk, k=3, SAME selected lr/grad_steps (never used for selection).

Saves (all new Phase 6 files; nothing from Phases 1-5 is modified):
    results/metrics/adaptive_graphsage_hparam_selection.json
    results/metrics/predictions/adaptive_graphsage_test.npz      (k=1)
    results/metrics/predictions/adaptive_graphsage_test_k3.npz   (k=3)
    results/metrics/adaptive_graphsage_train_manifest.json
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
from src.models.graphsage import GraphSAGE
from src.training.adaptive_gnn import (
    AdaptationConfig,
    build_per_step_data,
    pooled_predictions,
    run_adaptive_walk,
)
from src.training.adaptive_graphsage import select_adaptation_config_with_factory
from src.training.gnn_training import compute_pos_weight

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_adaptive_graphsage")

HIDDEN_CHANNELS = 128
DROPOUT = 0.5

CANDIDATE_GRID = [
    AdaptationConfig(lr=0.001, grad_steps=1, feedback_delay_k=1),
    AdaptationConfig(lr=0.001, grad_steps=3, feedback_delay_k=1),
    AdaptationConfig(lr=0.005, grad_steps=1, feedback_delay_k=1),
    AdaptationConfig(lr=0.005, grad_steps=3, feedback_delay_k=1),
]
K_SENSITIVITY = 3


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

    node_table, feature_cols = load_node_table_with_features(processed_dir)
    edges = load_edges(raw_dir / cfg["dataset"]["edges_file"])
    in_channels = len(feature_cols)

    split_cfg = cfg["split"]
    val_time_steps = list(range(split_cfg["val_start"], split_cfg["val_end"] + 1))
    test_time_steps = list(range(split_cfg["test_start"], split_cfg["test_end"] + 1))
    val_data = build_per_step_data(node_table, edges, feature_cols, val_time_steps)
    test_data = build_per_step_data(node_table, edges, feature_cols, test_time_steps)

    train_mask = (node_table["time_split"] == "train") & (node_table["is_labeled"])
    pos_weight = torch.tensor(
        compute_pos_weight(node_table.loc[train_mask, "label"].to_numpy()), dtype=torch.float32
    )
    logger.info("pos_weight (fixed, from train 1-29): %.4f", pos_weight.item())

    pretrained_state_dict = torch.load(models_dir / "gnn_improvement_selected.pt", weights_only=True)

    def model_factory() -> GraphSAGE:
        return GraphSAGE(in_channels=in_channels, hidden_channels=HIDDEN_CHANNELS, dropout=DROPOUT)

    logger.info("Stage 1: selecting adaptation config on validation 30-34 (k=1)")
    best_config, selection_log = select_adaptation_config_with_factory(
        model_factory, pretrained_state_dict, val_time_steps, val_data, pos_weight, CANDIDATE_GRID
    )
    with open(metrics_dir / "adaptive_graphsage_hparam_selection.json", "w") as f:
        json.dump({
            "candidate_grid": [
                {"lr": c.lr, "grad_steps": c.grad_steps, "feedback_delay_k": c.feedback_delay_k}
                for c in CANDIDATE_GRID
            ],
            "selection_log": selection_log,
            "selected": {"lr": best_config.lr, "grad_steps": best_config.grad_steps},
            "selection_criterion": "pooled PR-AUC over validation 30-34 predictions, k=1",
            "replay": False,
        }, f, indent=2)

    logger.info("Stage 2: primary walk k=1 lr=%.4f grad_steps=%d", best_config.lr, best_config.grad_steps)
    model_k1 = model_factory()
    model_k1.load_state_dict(pretrained_state_dict)
    t0 = time.time()
    result_k1 = run_adaptive_walk(model_k1, test_time_steps, test_data, pos_weight, best_config)
    time_k1 = time.time() - t0
    save_predictions(result_k1, "adaptive_graphsage_test", metrics_dir / "predictions")

    logger.info("Sensitivity walk k=%d (same lr/grad_steps; not used for selection)", K_SENSITIVITY)
    config_k3 = AdaptationConfig(
        lr=best_config.lr, grad_steps=best_config.grad_steps, feedback_delay_k=K_SENSITIVITY
    )
    model_k3 = model_factory()
    model_k3.load_state_dict(pretrained_state_dict)
    t0 = time.time()
    result_k3 = run_adaptive_walk(model_k3, test_time_steps, test_data, pos_weight, config_k3)
    time_k3 = time.time() - t0
    save_predictions(result_k3, "adaptive_graphsage_test_k3", metrics_dir / "predictions")

    manifest = {
        "model": "Adaptive GraphSAGE (online weight adaptation, simulated delayed feedback)",
        "base_checkpoint": "results/models/gnn_improvement_selected.pt (Phase 5, unmodified)",
        "architecture": "SAGEConv(165,128) -> ReLU -> Dropout(0.5) -> SAGEConv(128,1)",
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
    with open(metrics_dir / "adaptive_graphsage_train_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    print(json.dumps(main(args.config), indent=2, default=str))
