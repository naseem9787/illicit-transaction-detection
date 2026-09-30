"""Phase 8 (E5) diagnostics: tree + engineered graph features.

Usage:
    python -m scripts.run_phase8_e5_diagnostics --config config.yaml

Condition A: HGB (exact E2-selected config) on the original 165 features.
Condition B: same HGB config on 165 + predeclared, label-free graph
features (in/out/total degree, 2-hop reach, in/out neighbor feature means).
Same rolling-origin folds and 5 seeds as E2. Test steps 35-49 are never
built or touched.

Saves (Phase 8 files only):
    results/metrics/phase8/e5_graph_features_results.csv
    results/metrics/phase8/e5_summary.json
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
from src.training.graph_features import (
    E5_SEEDS,
    GRAPH_FEATURE_SCALAR_NAMES,
    HGB_BASE_CONFIG,
    graph_feature_names,
    run_e5,
)
from src.training.rolling_origin import ROLLING_ORIGIN_FOLDS, TEST_RANGE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_phase8_e5_diagnostics")


def main(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    project_root = Path(config_path).resolve().parent
    raw_dir = project_root / cfg["dataset"]["raw_dir"]
    processed_dir = project_root / cfg["dataset"]["processed_dir"]
    out_dir = project_root / cfg["paths"]["metrics_dir"] / "phase8"
    out_dir.mkdir(parents=True, exist_ok=True)

    node_table, feature_cols = load_node_table_with_features(processed_dir)
    edges = load_edges(raw_dir / cfg["dataset"]["edges_file"])

    n_graph_features = len(graph_feature_names(feature_cols))
    logger.info("E5 base HGB config (frozen from E2): %s", HGB_BASE_CONFIG)
    logger.info("Graph feature set: %d columns (%s + 2x%d neighbor-mean cols) — predeclared, fixed",
                n_graph_features, GRAPH_FEATURE_SCALAR_NAMES, len(feature_cols))
    logger.info("Folds: %s | seeds: %s (test period %s never touched)", ROLLING_ORIGIN_FOLDS, E5_SEEDS, TEST_RANGE)

    t0 = time.time()
    rows = run_e5(node_table, edges, feature_cols)
    wall_time = time.time() - t0

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "e5_graph_features_results.csv", index=False)

    baseline = df[df["condition"] == "baseline_165"]
    augmented = df[df["condition"] == "graph_augmented"]

    baseline_mean = float(baseline["val_pr_auc"].mean())
    augmented_mean = float(augmented["val_pr_auc"].mean())

    per_fold = df.groupby(["condition", "fold"])[["val_pr_auc", "val_f1"]].agg(["mean", "std"])
    per_seed = df.groupby(["condition", "seed"])[["val_pr_auc", "val_f1"]].agg(["mean"])

    summary = {
        "test_period_touched": False,
        "test_range": list(TEST_RANGE),
        "base_hgb_config": HGB_BASE_CONFIG,
        "seeds": E5_SEEDS,
        "folds": ROLLING_ORIGIN_FOLDS,
        "n_original_features": len(feature_cols),
        "n_graph_features": n_graph_features,
        "n_augmented_features": len(feature_cols) + n_graph_features,
        "graph_feature_names_prefix": GRAPH_FEATURE_SCALAR_NAMES,
        "n_runs": len(rows),
        "baseline_165_mean_val_pr_auc": baseline_mean,
        "baseline_165_std_val_pr_auc": float(baseline["val_pr_auc"].std()),
        "graph_augmented_mean_val_pr_auc": augmented_mean,
        "graph_augmented_std_val_pr_auc": float(augmented["val_pr_auc"].std()),
        "absolute_improvement": augmented_mean - baseline_mean,
        "baseline_165_mean_val_f1": float(baseline["val_f1"].mean()),
        "graph_augmented_mean_val_f1": float(augmented["val_f1"].mean()),
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
        "wall_time_seconds": round(wall_time, 1),
        "note": "Selection metric is mean validation PR-AUC across folds/seeds. "
                "No configuration was selected using, or evaluated on, t=35-49.",
    }
    with open(out_dir / "e5_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("E5 complete. baseline=%.4f graph_augmented=%.4f delta=%+.4f",
                baseline_mean, augmented_mean, augmented_mean - baseline_mean)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    print(json.dumps(main(args.config), indent=2, default=str))
