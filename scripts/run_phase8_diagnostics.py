"""Phase 8 diagnostics: E1 (feature-only MLP) and E2 (equal-budget tree
benchmark), both under rolling-origin validation. Test steps 35-49 are never
touched — every fold's validation range is strictly earlier than t=35.

Usage:
    python -m scripts.run_phase8_diagnostics --config config.yaml

Reuses (unchanged): src/data/dataset.py::load_node_table_with_features,
src/evaluation/metrics.py, src/training/gnn_training.py::compute_pos_weight.
Adds only new files — does not modify any Phase 1-7 file.

Saves (Phase 8 files only):
    results/metrics/phase8/e1_mlp_results.csv
    results/metrics/phase8/e2_tree_results.csv
    results/metrics/phase8/phase8_summary.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import load_node_table_with_features
from src.training.rolling_origin import (
    E1_SEEDS,
    E2_SEEDS,
    HGB_CONFIGS,
    MLP_DROPOUT,
    MLP_EPOCHS,
    MLP_HIDDEN,
    MLP_LR,
    MLP_WEIGHT_DECAY,
    RF_CONFIGS,
    ROLLING_ORIGIN_FOLDS,
    TEST_RANGE,
    run_e1_mlp,
    run_e2_trees,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_phase8_diagnostics")


def summarize(df: pd.DataFrame, group_cols: list[str], metric: str) -> pd.DataFrame:
    return df.groupby(group_cols)[metric].agg(["mean", "std", "min", "max"]).reset_index()


def main(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    project_root = Path(config_path).resolve().parent
    processed_dir = project_root / cfg["dataset"]["processed_dir"]
    out_dir = project_root / cfg["paths"]["metrics_dir"] / "phase8"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Rolling-origin folds: %s (test period %s is never touched)", ROLLING_ORIGIN_FOLDS, TEST_RANGE)

    node_table, feature_cols = load_node_table_with_features(processed_dir)

    logger.info("=== E1: MLP (165 -> %d -> 1), %d seeds x %d folds ===", MLP_HIDDEN, len(E1_SEEDS), len(ROLLING_ORIGIN_FOLDS))
    t0 = time.time()
    e1_rows = run_e1_mlp(node_table, feature_cols)
    e1_time = time.time() - t0
    e1_df = pd.DataFrame(e1_rows)
    e1_df.to_csv(out_dir / "e1_mlp_results.csv", index=False)

    logger.info("=== E2: RF (%d configs) + HistGradientBoosting (%d configs), %d seeds x %d folds ===",
                len(RF_CONFIGS), len(HGB_CONFIGS), len(E2_SEEDS), len(ROLLING_ORIGIN_FOLDS))
    t0 = time.time()
    e2 = run_e2_trees(node_table, feature_cols)
    e2_time = time.time() - t0
    e2_df = pd.concat([pd.DataFrame(e2["random_forest"]), pd.DataFrame(e2["hist_gradient_boosting"])], ignore_index=True)
    e2_df.to_csv(out_dir / "e2_tree_results.csv", index=False)

    e1_per_fold = summarize(e1_df, ["fold"], "val_pr_auc")
    e1_overall = {"mean": float(e1_df["val_pr_auc"].mean()), "std": float(e1_df["val_pr_auc"].std()),
                  "f1_mean": float(e1_df["val_f1"].mean())}

    e2_by_config = summarize(e2_df, ["model", "config_index"], "val_pr_auc").sort_values(
        ["model", "mean"], ascending=[True, False])
    best_rf_idx = int(e2_by_config[e2_by_config["model"] == "random_forest"].iloc[0]["config_index"])
    best_hgb_idx = int(e2_by_config[e2_by_config["model"] == "hist_gradient_boosting"].iloc[0]["config_index"])
    best_rf_row = e2_by_config[(e2_by_config["model"] == "random_forest") & (e2_by_config["config_index"] == best_rf_idx)].iloc[0]
    best_hgb_row = e2_by_config[(e2_by_config["model"] == "hist_gradient_boosting") & (e2_by_config["config_index"] == best_hgb_idx)].iloc[0]

    summary = {
        "test_period_touched": False,
        "test_range": list(TEST_RANGE),
        "folds": ROLLING_ORIGIN_FOLDS,
        "e1_mlp": {
            "architecture": f"165 -> {MLP_HIDDEN} -> 1, ReLU, Dropout({MLP_DROPOUT})",
            "hyperparameters": {"lr": MLP_LR, "weight_decay": MLP_WEIGHT_DECAY, "epochs": MLP_EPOCHS},
            "seeds": E1_SEEDS,
            "n_runs": len(e1_rows),
            "per_fold_val_pr_auc": e1_per_fold.to_dict(orient="records"),
            "overall_val_pr_auc_mean": e1_overall["mean"],
            "overall_val_pr_auc_std": e1_overall["std"],
            "overall_val_f1_mean": e1_overall["f1_mean"],
            "wall_time_seconds": round(e1_time, 1),
        },
        "e2_trees": {
            "seeds": E2_SEEDS,
            "n_rf_configs": len(RF_CONFIGS),
            "n_hgb_configs": len(HGB_CONFIGS),
            "n_runs": len(e2_df),
            "best_random_forest_config": {"config_index": best_rf_idx, **RF_CONFIGS[best_rf_idx],
                                          "mean_val_pr_auc": float(best_rf_row["mean"]),
                                          "std_val_pr_auc": float(best_rf_row["std"])},
            "best_hist_gradient_boosting_config": {"config_index": best_hgb_idx, **HGB_CONFIGS[best_hgb_idx],
                                                    "mean_val_pr_auc": float(best_hgb_row["mean"]),
                                                    "std_val_pr_auc": float(best_hgb_row["std"])},
            "wall_time_seconds": round(e2_time, 1),
        },
        "note": "Selection metric is mean validation PR-AUC across folds/seeds. "
                "No configuration was selected using, or evaluated on, t=35-49.",
    }
    with open(out_dir / "phase8_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Phase 8 diagnostics complete. Summary at %s", out_dir / "phase8_summary.json")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    print(json.dumps(main(args.config), indent=2, default=str))
