"""Step 2: adaptive-tree control experiment.

Usage:
    python -m scripts.run_adaptive_tree_control --config config.yaml

Answers: "does the strongest tabular model (HistGradientBoosting, exact
E2-selected config) show a similar gain to Adaptive GraphSAGE simply
because it also receives newly revealed labels during a chronological
walk?" This is a CONTROL for the adaptive-GNN result, not a new model
family — see src/training/adaptive_tree.py for the full mechanism and
fairness rationale.

Rolling-origin folds (reused unchanged from rolling_origin.py) x 5 seeds
(same as E2). Validation only — the test period is never built or touched.

Saves (new files only):
    results/metrics/phase8/adaptive_tree_control_results.csv
    results/metrics/phase8/adaptive_tree_control_summary.json
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
from src.data.loader import load_edges
from src.training.adaptive_tree import ADAPT_SEEDS, FEEDBACK_DELAY_K, ITERS_PER_UPDATE, run_adaptive_tree_control
from src.training.graph_features import HGB_BASE_CONFIG
from src.training.rolling_origin import ROLLING_ORIGIN_FOLDS, TEST_RANGE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_adaptive_tree_control")


def main(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    project_root = Path(config_path).resolve().parent
    raw_dir = project_root / cfg["dataset"]["raw_dir"]
    processed_dir = project_root / cfg["dataset"]["processed_dir"]
    out_dir = project_root / cfg["paths"]["metrics_dir"] / "phase8"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Base HGB config (frozen from E2): %s", HGB_BASE_CONFIG)
    logger.info("Adaptation: feedback_delay_k=%d, iters_per_update=%d (fixed, not tuned)", FEEDBACK_DELAY_K, ITERS_PER_UPDATE)
    logger.info("Folds: %s | seeds: %s (test period %s never touched)", ROLLING_ORIGIN_FOLDS, ADAPT_SEEDS, TEST_RANGE)

    node_table, feature_cols = load_node_table_with_features(processed_dir)
    edges = load_edges(raw_dir / cfg["dataset"]["edges_file"])

    t0 = time.time()
    rows = run_adaptive_tree_control(node_table, edges, feature_cols)
    wall_time = time.time() - t0

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "adaptive_tree_control_results.csv", index=False)

    static = df[df["condition"] == "static"]
    adaptive = df[df["condition"] == "adaptive"]
    static_mean, adaptive_mean = float(static["val_pr_auc"].mean()), float(adaptive["val_pr_auc"].mean())

    per_fold = df.groupby(["condition", "fold"])[["val_pr_auc", "val_f1"]].agg(["mean", "std"])
    per_seed = df.groupby(["condition", "seed"])[["val_pr_auc", "val_f1"]].agg(["mean"])

    # per-(fold,seed) paired delta, mirrors the paired comparisons used throughout the project
    paired = static[["fold", "seed", "val_pr_auc", "val_f1"]].merge(
        adaptive[["fold", "seed", "val_pr_auc", "val_f1"]], on=["fold", "seed"], suffixes=("_static", "_adaptive"))
    paired["delta_pr_auc"] = paired["val_pr_auc_adaptive"] - paired["val_pr_auc_static"]
    paired["delta_f1"] = paired["val_f1_adaptive"] - paired["val_f1_static"]

    summary = {
        "test_period_touched": False,
        "test_range": list(TEST_RANGE),
        "base_hgb_config": HGB_BASE_CONFIG,
        "feedback_delay_k": FEEDBACK_DELAY_K,
        "iters_per_update": ITERS_PER_UPDATE,
        "seeds": ADAPT_SEEDS,
        "folds": ROLLING_ORIGIN_FOLDS,
        "n_runs": len(rows),
        "static_mean_val_pr_auc": static_mean,
        "static_std_val_pr_auc": float(static["val_pr_auc"].std()),
        "adaptive_mean_val_pr_auc": adaptive_mean,
        "adaptive_std_val_pr_auc": float(adaptive["val_pr_auc"].std()),
        "absolute_improvement_pr_auc": adaptive_mean - static_mean,
        "static_mean_val_f1": float(static["val_f1"].mean()),
        "adaptive_mean_val_f1": float(adaptive["val_f1"].mean()),
        "absolute_improvement_f1": float(adaptive["val_f1"].mean()) - float(static["val_f1"].mean()),
        "n_folds_seeds_improved_pr_auc": int((paired["delta_pr_auc"] > 0).sum()),
        "n_folds_seeds_total": int(len(paired)),
        "per_fold": {
            f"{cond}__{fold}": {
                "val_pr_auc_mean": float(per_fold.loc[(cond, fold), ("val_pr_auc", "mean")]),
                "val_pr_auc_std": float(per_fold.loc[(cond, fold), ("val_pr_auc", "std")]),
                "val_f1_mean": float(per_fold.loc[(cond, fold), ("val_f1", "mean")]),
            }
            for cond, fold in per_fold.index
        },
        "per_seed": {
            f"{cond}__seed{seed}": {
                "val_pr_auc_mean": float(per_seed.loc[(cond, seed), ("val_pr_auc", "mean")]),
                "val_f1_mean": float(per_seed.loc[(cond, seed), ("val_f1", "mean")]),
            }
            for cond, seed in per_seed.index
        },
        "paired_deltas": paired[["fold", "seed", "delta_pr_auc", "delta_f1"]].to_dict(orient="records"),
        "wall_time_seconds": round(wall_time, 1),
        "note": "Validation-only control. No configuration was selected using, or evaluated on, t=35-49. "
                "This experiment does not select a final model — it informs interpretation of the adaptive GNN result.",
    }
    with open(out_dir / "adaptive_tree_control_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Adaptive-tree control complete. static=%.4f adaptive=%.4f delta=%+.4f (%d/%d fold*seed improved)",
                static_mean, adaptive_mean, adaptive_mean - static_mean,
                summary["n_folds_seeds_improved_pr_auc"], summary["n_folds_seeds_total"])
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    print(json.dumps(main(args.config), indent=2, default=str))
