"""Phase 8 (E7) diagnostics: graph-smoothing of HistGradientBoosting scores.

Usage:
    python -m scripts.run_phase8_e7_diagnostics --config config.yaml

Base tree: the exact E2-selected HistGradientBoosting configuration
(max_depth=None, learning_rate=0.1, max_iter=100, min_samples_leaf=50),
never modified here. Only its output probabilities are post-processed.

Rolling-origin folds (reused from rolling_origin.py, unchanged) x 5 seeds
(same as E2) x {alpha in [0, .1, .2, .3, .4, .5]} x {depth in [1, 2]}.
Test steps 35-49 are never built or touched.

Saves (Phase 8 files only):
    results/metrics/phase8/e7_graph_smoothing_results.csv
    results/metrics/phase8/e7_summary.json
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
from src.training.graph_smoothing import ALPHAS, DEPTHS, E7_SEEDS, HGB_BASE_CONFIG, run_e7
from src.training.rolling_origin import ROLLING_ORIGIN_FOLDS, TEST_RANGE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_phase8_e7_diagnostics")


def main(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    project_root = Path(config_path).resolve().parent
    raw_dir = project_root / cfg["dataset"]["raw_dir"]
    processed_dir = project_root / cfg["dataset"]["processed_dir"]
    out_dir = project_root / cfg["paths"]["metrics_dir"] / "phase8"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("E7 base HGB config (frozen from E2): %s", HGB_BASE_CONFIG)
    logger.info("Folds: %s | seeds: %s | alphas: %s | depths: %s (test period %s never touched)",
                ROLLING_ORIGIN_FOLDS, E7_SEEDS, ALPHAS, DEPTHS, TEST_RANGE)

    node_table, feature_cols = load_node_table_with_features(processed_dir)
    edges = load_edges(raw_dir / cfg["dataset"]["edges_file"])

    t0 = time.time()
    rows = run_e7(node_table, edges, feature_cols)
    wall_time = time.time() - t0

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "e7_graph_smoothing_results.csv", index=False)

    smoothed = df[df["variant"] == "smoothed"]
    baseline = df[df["variant"] == "tree_only_baseline"]
    baseline_mean_pr_auc = float(baseline["val_pr_auc"].mean())

    by_config = smoothed.groupby(["alpha", "depth"])["val_pr_auc"].agg(["mean", "std"]).reset_index()
    by_config = by_config.sort_values("mean", ascending=False)
    best = by_config.iloc[0]
    best_alpha, best_depth = float(best["alpha"]), int(best["depth"])

    per_fold_baseline = baseline.groupby("fold")[["val_pr_auc", "val_f1"]].agg(["mean", "std"])
    best_rows = smoothed[(smoothed["alpha"] == best_alpha) & (smoothed["depth"] == best_depth)]
    per_fold_best = best_rows.groupby("fold")[["val_pr_auc", "val_f1"]].agg(["mean", "std"])

    # sanity check executed as part of the real run: alpha=0 must exactly equal the baseline
    # (for every depth, since a no-op smoothing step composed with itself is still a no-op)
    alpha0 = smoothed[smoothed["alpha"] == 0.0][["fold", "seed", "depth", "val_pr_auc"]]
    merged = alpha0.merge(baseline[["fold", "seed", "val_pr_auc"]], on=["fold", "seed"], suffixes=("_alpha0", "_baseline"))
    max_alpha0_diff = float((merged["val_pr_auc_alpha0"] - merged["val_pr_auc_baseline"]).abs().max()) if len(merged) else None

    summary = {
        "test_period_touched": False,
        "test_range": list(TEST_RANGE),
        "base_hgb_config": HGB_BASE_CONFIG,
        "seeds": E7_SEEDS,
        "alphas": ALPHAS,
        "depths": DEPTHS,
        "n_runs": len(rows),
        "baseline_mean_val_pr_auc": baseline_mean_pr_auc,
        "baseline_per_fold": {
            fold: {"val_pr_auc_mean": float(per_fold_baseline.loc[fold, ("val_pr_auc", "mean")]),
                   "val_pr_auc_std": float(per_fold_baseline.loc[fold, ("val_pr_auc", "std")]),
                   "val_f1_mean": float(per_fold_baseline.loc[fold, ("val_f1", "mean")])}
            for fold in per_fold_baseline.index
        },
        "grid_ranked_by_mean_val_pr_auc": by_config.to_dict(orient="records"),
        "best_alpha": best_alpha,
        "best_depth": best_depth,
        "best_mean_val_pr_auc": float(best["mean"]),
        "absolute_improvement_vs_baseline": float(best["mean"]) - baseline_mean_pr_auc,
        "best_config_per_fold": {
            fold: {"val_pr_auc_mean": float(per_fold_best.loc[fold, ("val_pr_auc", "mean")]),
                   "val_pr_auc_std": float(per_fold_best.loc[fold, ("val_pr_auc", "std")]),
                   "val_f1_mean": float(per_fold_best.loc[fold, ("val_f1", "mean")])}
            for fold in per_fold_best.index
        },
        "alpha0_exactly_matches_baseline_max_abs_diff": max_alpha0_diff,
        "wall_time_seconds": round(wall_time, 1),
        "note": "Selection metric is mean validation PR-AUC across folds/seeds. "
                "No configuration was selected using, or evaluated on, t=35-49.",
    }
    with open(out_dir / "e7_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("E7 complete. Best alpha=%.1f depth=%d mean_val_pr_auc=%.4f (baseline %.4f, delta %+.4f)",
                best_alpha, best_depth, float(best["mean"]), baseline_mean_pr_auc, float(best["mean"]) - baseline_mean_pr_auc)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    print(json.dumps(main(args.config), indent=2, default=str))
