"""Phase 1 entry point: load, validate, preprocess, split, build graphs.

Usage:
    python -m scripts.audit_dataset --config config.yaml

Produces:
    data/processed/node_table.csv        (merged, labeled, split-tagged nodes)
    data/processed/snapshot_summary.csv  (per-time-step node/edge/label counts)
    data/processed/audit_summary.json    (machine-readable summary of this run)
    results/figures/class_distribution.png
    results/figures/illicit_rate_over_time.png
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.graph_builder import build_all_snapshots, snapshot_summary
from src.data.loader import load_raw_dataset, validate_dataset
from src.data.preprocessing import add_split_and_masks, build_node_table, verify_no_leakage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("audit_dataset")


def make_class_distribution_figure(node_table, out_path: Path) -> None:
    counts = node_table["label"].map({1: "illicit", 0: "licit", -1: "unknown"}).value_counts()
    counts = counts.reindex(["illicit", "licit", "unknown"])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(counts.index, counts.values, color=["#c0392b", "#2980b9", "#95a5a6"])
    ax.set_ylabel("Number of transactions")
    ax.set_title("Elliptic dataset: label distribution\n(all 49 time steps)")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def make_illicit_rate_figure(node_table, split_cfg: dict, out_path: Path) -> None:
    labeled = node_table[node_table["is_labeled"]]
    rate = labeled.groupby("time_step")["label"].mean()  # mean of {0,1} = illicit rate
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rate.index, rate.values, marker="o", markersize=3, color="#c0392b", linewidth=1)
    ax.axvspan(split_cfg["train_start"], split_cfg["train_end"], color="#2980b9", alpha=0.08, label="train")
    ax.axvspan(split_cfg["val_start"], split_cfg["val_end"], color="#f39c12", alpha=0.12, label="val")
    ax.axvspan(split_cfg["test_start"], split_cfg["test_end"], color="#7f8c8d", alpha=0.08, label="test")
    ax.set_xlabel("time_step")
    ax.set_ylabel("illicit rate among labeled transactions")
    ax.set_title("Illicit rate over time (labeled transactions only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main(config_path: str) -> dict:
    t0 = time.time()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    project_root = Path(config_path).resolve().parent
    raw_dir = project_root / cfg["dataset"]["raw_dir"]
    processed_dir = project_root / cfg["dataset"]["processed_dir"]
    figures_dir = project_root / cfg["paths"]["figures_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw dataset from %s", raw_dir)
    data = load_raw_dataset(
        raw_dir,
        cfg["dataset"]["classes_file"],
        cfg["dataset"]["edges_file"],
        cfg["dataset"]["features_file"],
        cfg["dataset"]["num_raw_feature_columns"],
    )
    t_loaded = time.time()

    logger.info("Validating dataset structure")
    validation_report = validate_dataset(data)
    t_validated = time.time()

    logger.info("Building node table (merge features + labels)")
    node_table = build_node_table(data)
    node_table = add_split_and_masks(node_table, cfg["split"])
    verify_no_leakage(node_table, cfg["split"])
    t_preprocessed = time.time()

    logger.info("Building per-time-step graph snapshots")
    snapshots = build_all_snapshots(node_table, data.edges, data.feature_cols)
    snap_summary_df = snapshot_summary(snapshots)
    t_graphs = time.time()

    # Store metadata (small, human-readable) and the 165-d float feature
    # matrix (large) separately. A 664MB text CSV re-dump of already-float64
    # feature values was tried and discarded here: it's ~5x larger on disk
    # than a compressed float32 .npz for no reproducibility benefit, and
    # this sandbox is disk-constrained. Row order is identical between the
    # two files, so they realign by position.
    meta_cols = ["txId", "time_step", "label", "time_split", "is_labeled"]
    node_table[meta_cols].to_csv(processed_dir / "node_table_meta.csv", index=False)
    np.savez_compressed(
        processed_dir / "node_features.npz",
        x=node_table[data.feature_cols].to_numpy(dtype=np.float32),
        feature_cols=np.array(data.feature_cols),
    )
    snap_summary_df.to_csv(processed_dir / "snapshot_summary.csv", index=False)

    make_class_distribution_figure(node_table, figures_dir / "class_distribution.png")
    make_illicit_rate_figure(node_table, cfg["split"], figures_dir / "illicit_rate_over_time.png")

    split_counts = (
        node_table.groupby("time_split")
        .agg(
            num_nodes=("txId", "count"),
            num_labeled=("is_labeled", "sum"),
            num_illicit=("label", lambda s: int((s == 1).sum())),
        )
        .to_dict(orient="index")
    )

    summary = {
        "runtime_seconds": {
            "load": round(t_loaded - t0, 2),
            "validate": round(t_validated - t_loaded, 2),
            "preprocess": round(t_preprocessed - t_validated, 2),
            "build_graphs": round(t_graphs - t_preprocessed, 2),
            "total": round(t_graphs - t0, 2),
        },
        "validation_report": validation_report,
        "split_config": cfg["split"],
        "split_counts": split_counts,
        "num_snapshots_built": len(snapshots),
        "total_nodes": int(len(node_table)),
        "total_edges": int(sum(s.edge_index.shape[1] for s in snapshots.values())),
    }
    with open(processed_dir / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("Pipeline complete in %.2fs", summary["runtime_seconds"]["total"])
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    result = main(args.config)
    print(json.dumps(result, indent=2, default=str))
