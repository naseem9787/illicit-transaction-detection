"""Phase 7 training entry point: multi-seed robustness of Static -> Adaptive GraphSAGE.

Usage:
    python -m scripts.train_multiseed_robustness --config config.yaml

Robustness study only. No retuning: every seed uses the already-selected
Phase 5/6 configuration (GraphSAGE h=128, dropout=0.5, lr=0.01, wd=5e-4,
200 epochs, checkpoint by validation PR-AUC; adaptation lr=0.001,
grad_steps=1, k=1, no replay).

Seeds are exactly SEEDS = [42, 123, 456, 789, 2024].

Seed 42 is the already-completed Phase 5/6 reference and is NOT retrained:
its artifacts (gnn_improvement_selected.pt, gnn_improvement_test.npz,
adaptive_graphsage_test.npz) are reused read-only. NOTE: that Phase 5
checkpoint was initialised from the RNG state left by the previous grid
config (the model was constructed before train_gcn's own reseed), so it cannot
be regenerated from a bare seed=42 script. To check continuity, the Phase 7
protocol is also run from scratch for seed 42 ("seed_42_replicate"); any
discrepancy vs the reference is reported, never silently substituted.

Phase 7 protocol for a trained seed s:
    torch.manual_seed(s); model = GraphSAGE(...); train_gcn(seed=s, ...)
    static  = best-validation-PR-AUC checkpoint, evaluated on test
    adaptive= walk (Phase 4 run_adaptive_walk, unchanged) starting from a
              copy of the SAME checkpoint weights

Test labels are never used for any selection: the training function has no
test-batch argument, and the adaptation configuration is a fixed constant.

Saves under results/metrics/multiseed/ and results/models/multiseed/ only.
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
from src.evaluation.metrics import select_threshold_maximizing_f1
from src.models.graphsage import GraphSAGE
from src.training.adaptive_gnn import AdaptationConfig, build_per_step_data, pooled_predictions, run_adaptive_walk
from src.training.gnn_training import build_split_batches, compute_pos_weight, predict_labeled, train_gcn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_multiseed_robustness")

SEEDS = [42, 123, 456, 789, 2024]
REFERENCE_SEED = 42
FIXED_THRESHOLD = 0.898  # Phase 6 threshold; verified against the Phase 5 manifest at runtime

HIDDEN = 128
DROPOUT = 0.5
LR = 0.01
WEIGHT_DECAY = 5e-4
EPOCHS = 200

ADAPT_LR = 0.001
ADAPT_GRAD_STEPS = 1
FEEDBACK_DELAY_K = 1

CHECKPOINT_CRITERION = "best validation PR-AUC (30-34) over 200 epochs, via train_gcn (unchanged)"


def init_model(seed: int, in_channels: int) -> GraphSAGE:
    """Seed the global RNG, then construct: initial weights are a pure function of the seed."""
    torch.manual_seed(seed)
    return GraphSAGE(in_channels=in_channels, hidden_channels=HIDDEN, dropout=DROPOUT)


def train_static_for_seed(seed: int, in_channels: int, train_batch, val_batch):
    """Static GraphSAGE for one seed. Takes NO test batch by design: nothing
    here can be selected using test data."""
    model = init_model(seed, in_channels)
    result = train_gcn(model, train_batch, val_batch, seed=seed, epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY)
    model.load_state_dict(result.best_state_dict)
    return model, result


def adaptation_config_for_seed(seed: int) -> AdaptationConfig:
    """Fixed for every seed. The seed only drives dropout RNG during updates."""
    return AdaptationConfig(lr=ADAPT_LR, grad_steps=ADAPT_GRAD_STEPS, feedback_delay_k=FEEDBACK_DELAY_K, seed=seed)


def run_adaptive_for_seed(seed, state_dict, in_channels, test_time_steps, test_data, pos_weight):
    """Adaptive GraphSAGE starting from a copy of the SAME seed's static checkpoint."""
    model = GraphSAGE(in_channels=in_channels, hidden_channels=HIDDEN, dropout=DROPOUT)
    model.load_state_dict({k: v.clone() for k, v in state_dict.items()})
    return run_adaptive_walk(model, test_time_steps, test_data, pos_weight, adaptation_config_for_seed(seed))


def main(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    root = Path(config_path).resolve().parent
    raw_dir = root / cfg["dataset"]["raw_dir"]
    processed_dir = root / cfg["dataset"]["processed_dir"]
    metrics_dir = root / cfg["paths"]["metrics_dir"]
    models_dir = root / cfg["paths"]["models_dir"]
    out_metrics = metrics_dir / "multiseed"
    out_pred = out_metrics / "predictions"
    out_models = models_dir / "multiseed"
    for d in (out_pred, out_models):
        d.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "gnn_improvement_train_manifest.json") as f:
        phase5 = json.load(f)["selected"]
    assert abs(phase5["selected_threshold_on_validation"] - FIXED_THRESHOLD) < 5e-4, "threshold drifted from 0.898"
    assert (phase5["hidden_channels"], phase5["dropout"], phase5["learning_rate"]) == (HIDDEN, DROPOUT, LR)

    node_table, feature_cols = load_node_table_with_features(processed_dir)
    edges = load_edges(raw_dir / cfg["dataset"]["edges_file"])
    in_channels = len(feature_cols)
    split_cfg = cfg["split"]
    test_time_steps = list(range(split_cfg["test_start"], split_cfg["test_end"] + 1))

    batches = build_split_batches(node_table, edges, feature_cols, split_cfg)
    test_data = build_per_step_data(node_table, edges, feature_cols, test_time_steps)
    train_mask = (node_table["time_split"] == "train") & (node_table["is_labeled"])
    pos_weight = torch.tensor(compute_pos_weight(node_table.loc[train_mask, "label"].to_numpy()), dtype=torch.float32)

    common = {
        "architecture": f"SAGEConv(165,{HIDDEN}) -> ReLU -> Dropout({DROPOUT}) -> SAGEConv({HIDDEN},1)",
        "hyperparameters": {"hidden": HIDDEN, "dropout": DROPOUT, "lr": LR, "weight_decay": WEIGHT_DECAY, "epochs": EPOCHS},
        "checkpoint_selection_criterion": CHECKPOINT_CRITERION,
        "fixed_threshold_primary": FIXED_THRESHOLD,
        "adaptation_config": {"lr": ADAPT_LR, "grad_steps": ADAPT_GRAD_STEPS, "k": FEEDBACK_DELAY_K, "replay": False,
                              "selection": "none (fixed Phase 6 configuration; not retuned per seed)"},
        "test_population_size": int(batches["test"].is_labeled.sum()),
        "pos_weight_fixed_from_train": pos_weight.item(),
    }

    # ---- seed 42 reference: existing Phase 5/6 artifacts, verified, not rerun ----
    ref_manifest = {
        "label": "seed_42_reference", "seed": REFERENCE_SEED, **common,
        "source": "existing Phase 5/6 artifacts (read-only, not retrained)",
        "static_checkpoint": "results/models/gnn_improvement_selected.pt",
        "static_test_predictions": "results/metrics/predictions/gnn_improvement_test.npz",
        "adaptive_test_predictions": "results/metrics/predictions/adaptive_graphsage_test.npz",
        "own_validation_threshold": phase5["selected_threshold_on_validation"],
        "best_validation_epoch": phase5["best_validation_epoch"],
        "validation_pr_auc": phase5["validation_pr_auc"],
    }
    for key in ("static_checkpoint", "static_test_predictions", "adaptive_test_predictions"):
        assert (root / ref_manifest[key]).exists(), f"missing reference artifact {ref_manifest[key]}"
    with open(out_metrics / "manifest_seed_42_reference.json", "w") as f:
        json.dump(ref_manifest, f, indent=2)

    runs = [("seed_123", 123), ("seed_456", 456), ("seed_789", 789), ("seed_2024", 2024),
            ("seed_42_replicate", REFERENCE_SEED)]
    for label, seed in runs:
        manifest_path = out_metrics / f"manifest_{label}.json"
        if manifest_path.exists():
            logger.info("[%s] already complete, skipping", label)
            continue
        logger.info("[%s] training static GraphSAGE (seed=%d)", label, seed)
        model, result = train_static_for_seed(seed, in_channels, batches["train"], batches["val"])
        torch.save(model.state_dict(), out_models / f"static_{label}.pt")

        yv, pv, tv = predict_labeled(model, batches["val"])
        own_threshold = select_threshold_maximizing_f1(yv, pv)  # validation only
        np.savez_compressed(out_pred / f"static_{label}_val.npz", y_true=yv, y_prob=pv, time_step=tv)
        ys, ps, ts = predict_labeled(model, batches["test"])  # evaluated after checkpoint is fixed
        np.savez_compressed(out_pred / f"static_{label}_test.npz", y_true=ys, y_prob=ps, time_step=ts)

        logger.info("[%s] adaptive walk from the same checkpoint (k=1)", label)
        t0 = time.time()
        walk = run_adaptive_for_seed(seed, result.best_state_dict, in_channels, test_time_steps, test_data, pos_weight)
        ya, pa, ta = pooled_predictions(walk)
        np.savez_compressed(out_pred / f"adaptive_{label}_test.npz", y_true=ya, y_prob=pa, time_step=ta)

        manifest = {
            "label": label, "seed": seed, **common,
            "init_protocol": "torch.manual_seed(seed); construct GraphSAGE; train_gcn(seed=seed)",
            "best_validation_epoch": result.best_epoch,
            "validation_pr_auc": result.best_val_pr_auc,
            "own_validation_threshold": own_threshold,
            "train_time_seconds": round(result.train_time_seconds, 2),
            "adaptive_walk_seconds": round(time.time() - t0, 3),
            "n_adapt_events": sum(1 for e in walk.events if e["type"] == "adapt"),
            "torch_version": torch.__version__,
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("[%s] done: val PR-AUC=%.4f best_epoch=%d", label, result.best_val_pr_auc, result.best_epoch)

    return {"seeds": SEEDS, "runs": [r[0] for r in runs]}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    print(json.dumps(main(args.config), indent=2))
