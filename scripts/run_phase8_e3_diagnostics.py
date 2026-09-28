"""Phase 8 (E3) diagnostics: direction-aware graph aggregation.

Usage:
    python -m scripts.run_phase8_e3_diagnostics --config config.yaml

Trains three arms under identical rolling-origin validation (3 folds x 3
seeds = 9 runs each, 27 total): plain GraphSAGE (unmodified Phase 5 model,
the control), symmetrized-edge GraphSAGE, and direction-aware GraphSAGE.
Test steps 35-49 are never built or touched —
rolling_origin_graph.py::build_graph_batches_for_range refuses any range
that reaches t=35 or beyond.

Reuses (unchanged): src/data/dataset.py::load_node_table_with_features,
src/data/loader.py::load_edges, src/training/gnn_training.py::train_gcn /
predict_labeled, src/evaluation/metrics.py. Adds only new files.

Saves (Phase 8 files only):
    results/metrics/phase8/e3_direction_aware_results.csv
    results/metrics/phase8/e3_summary.json
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
from src.training.rolling_origin import ROLLING_ORIGIN_FOLDS, TEST_RANGE
from src.training.rolling_origin_graph import (
    E3_ARCHITECTURES,
    E3_DROPOUT,
    E3_EPOCHS,
    E3_HIDDEN,
    E3_LR,
    E3_SEEDS,
    run_e3,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_phase8_e3_diagnostics")


def main(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    project_root = Path(config_path).resolve().parent
    raw_dir = project_root / cfg["dataset"]["raw_dir"]
    processed_dir = project_root / cfg["dataset"]["processed_dir"]
    out_dir = project_root / cfg["paths"]["metrics_dir"] / "phase8"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("E3 folds: %s (test period %s is never touched)", ROLLING_ORIGIN_FOLDS, TEST_RANGE)
    logger.info("Architectures: %s | seeds: %s", list(E3_ARCHITECTURES), E3_SEEDS)

    node_table, feature_cols = load_node_table_with_features(processed_dir)
    edges = load_edges(raw_dir / cfg["dataset"]["edges_file"])

    t0 = time.time()
    rows = run_e3(node_table, edges, feature_cols)
    wall_time = time.time() - t0

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "e3_direction_aware_results.csv", index=False)

    per_arch = df.groupby("architecture")["val_pr_auc"].agg(["mean", "std", "min", "max"]).reset_index()
    per_arch_fold = df.groupby(["architecture", "fold"])[["val_pr_auc", "val_f1"]].agg(["mean", "std"]).reset_index()

    baseline_mean = float(df[df["architecture"] == "plain_graphsage"]["val_pr_auc"].mean())
    deltas = {}
    for arch in ("symmetrized_graphsage", "direction_aware_graphsage"):
        arch_mean = float(df[df["architecture"] == arch]["val_pr_auc"].mean())
        deltas[arch] = {"mean_val_pr_auc": arch_mean, "delta_vs_plain": arch_mean - baseline_mean}

    summary = {
        "test_period_touched": False,
        "test_range": list(TEST_RANGE),
        "folds": ROLLING_ORIGIN_FOLDS,
        "seeds": E3_SEEDS,
        "architecture_settings": {"hidden": E3_HIDDEN, "dropout": E3_DROPOUT, "lr": E3_LR, "epochs": E3_EPOCHS},
        "n_runs": len(rows),
        "per_architecture_overall": per_arch.to_dict(orient="records"),
        "per_architecture_per_fold": [
            {"architecture": row[("architecture", "")], "fold": row[("fold", "")],
             "val_pr_auc_mean": row[("val_pr_auc", "mean")], "val_pr_auc_std": row[("val_pr_auc", "std")],
             "val_f1_mean": row[("val_f1", "mean")], "val_f1_std": row[("val_f1", "std")]}
            for _, row in per_arch_fold.iterrows()
        ],
        "plain_graphsage_baseline_mean_val_pr_auc": baseline_mean,
        "deltas_vs_plain_baseline": deltas,
        "wall_time_seconds": round(wall_time, 1),
        "note": "Selection metric is mean validation PR-AUC across folds/seeds. "
                "No configuration was selected using, or evaluated on, t=35-49.",
    }
    with open(out_dir / "e3_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("E3 complete. Summary at %s", out_dir / "e3_summary.json")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    print(json.dumps(main(args.config), indent=2, default=str))
