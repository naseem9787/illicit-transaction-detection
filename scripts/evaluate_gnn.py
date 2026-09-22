"""Phase 3 evaluation entry point: basic GCN vs. the Phase 2 baselines.

Usage:
    python -m scripts.evaluate_gnn --config config.yaml

Reads GCN predictions saved by scripts/train_gnn.py (never re-touches the
test set for fitting anything) and, read-only, the already-computed Phase 2
Logistic Regression / Random Forest predictions and results
(results/metrics/predictions/*_no_tstep_test.npz, baseline_results.json) to
build a like-for-like three-way comparison. Nothing in results/metrics/
belonging to Phase 2 is modified.

Saves:
    results/metrics/gcn_results.json      (val + test metrics, threshold)
    results/metrics/temporal_gcn.csv      (per-time-step test metrics)
    results/figures/gcn_vs_baselines_comparison.png
    results/figures/gcn_f1_over_time.png
    results/figures/pr_roc_curves_all_models.png
    results/figures/gcn_confusion_matrix.png
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import compute_binary_metrics, select_threshold_maximizing_f1
from src.evaluation.temporal_metrics import compute_metrics_by_time_step

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate_gnn")

BASELINE_LABELS = {"logreg_no_tstep": "Logistic Regression", "random_forest_no_tstep": "Random Forest"}
ALL_LABELS = {**BASELINE_LABELS, "gcn": "GCN (basic)"}


def load_predictions(metrics_dir: Path, name: str, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    npz = np.load(metrics_dir / "predictions" / f"{name}_{split}.npz")
    return npz["y_true"], npz["y_prob"], npz["time_step"]


def evaluate_gcn(metrics_dir: Path) -> dict:
    y_val, p_val, _ = load_predictions(metrics_dir, "gcn", "val")
    y_test, p_test, t_test = load_predictions(metrics_dir, "gcn", "test")

    threshold = select_threshold_maximizing_f1(y_val, p_val)
    val_metrics = compute_binary_metrics(y_val, p_val, threshold)
    test_metrics = compute_binary_metrics(y_test, p_test, threshold)
    temporal_df = compute_metrics_by_time_step(y_test, p_test, t_test, threshold)

    return {
        "threshold_selected_on": "validation",
        "threshold": threshold,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "temporal_df": temporal_df,
        "raw": {"y_test": y_test, "p_test": p_test, "t_test": t_test},
    }


def make_comparison_figure(gcn_test_metrics: dict, baseline_results: dict, out_path: Path) -> None:
    names = list(BASELINE_LABELS.keys()) + ["gcn"]
    labels = [ALL_LABELS[n] for n in names]
    metric_names = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    x = np.arange(len(metric_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, name in enumerate(names):
        if name == "gcn":
            vals = [gcn_test_metrics[m] for m in metric_names]
        else:
            vals = [baseline_results["primary_models"][name]["test_metrics"][m] for m in metric_names]
        ax.bar(x + i * width, vals, width, label=labels[i])
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Logistic Regression vs Random Forest vs basic GCN\nTEST set (time steps 35-49)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def make_f1_over_time_figure(temporal_df, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(temporal_df["time_step"], temporal_df["f1"], marker="o", markersize=4, color="#8e44ad", label="F1")
    ax1.set_xlabel("time_step")
    ax1.set_ylabel("F1", color="#8e44ad")
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis="y", labelcolor="#8e44ad")

    ax2 = ax1.twinx()
    ax2.bar(temporal_df["time_step"], temporal_df["n_labeled"], alpha=0.15, color="gray", label="n labeled")
    ax2.set_ylabel("labeled transactions", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    ax1.set_title("GCN (basic): F1 over test time steps (35-49)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def make_pr_roc_figures(gcn_raw: dict, metrics_dir: Path, figures_dir: Path) -> None:
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(12, 5))

    for name, label in BASELINE_LABELS.items():
        y_test, p_test, _ = load_predictions(metrics_dir, name, "test")
        prec, rec, _ = precision_recall_curve(y_test, p_test)
        fpr, tpr, _ = roc_curve(y_test, p_test)
        ax_pr.plot(rec, prec, label=label)
        ax_roc.plot(fpr, tpr, label=label)

    prec, rec, _ = precision_recall_curve(gcn_raw["y_test"], gcn_raw["p_test"])
    fpr, tpr, _ = roc_curve(gcn_raw["y_test"], gcn_raw["p_test"])
    ax_pr.plot(rec, prec, label=ALL_LABELS["gcn"], linewidth=2, linestyle="--")
    ax_roc.plot(fpr, tpr, label=ALL_LABELS["gcn"], linewidth=2, linestyle="--")

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall curves (test set) — all models")
    ax_pr.legend()

    ax_roc.plot([0, 1], [0, 1], linestyle=":", color="gray", linewidth=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC curves (test set) — all models")
    ax_roc.legend()

    fig.tight_layout()
    fig.savefig(figures_dir / "pr_roc_curves_all_models.png", dpi=150)
    plt.close(fig)
    logger.info("Saved pr_roc_curves_all_models.png")


def make_confusion_matrix_figure(gcn_raw: dict, threshold: float, out_path: Path) -> None:
    y_test, p_test = gcn_raw["y_test"], gcn_raw["p_test"]
    y_pred = (p_test >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["licit", "illicit"], ax=ax, colorbar=False
    )
    ax.set_title("Confusion matrix: GCN (basic)\ntest set, time steps 35-49")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def main(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    project_root = Path(config_path).resolve().parent
    metrics_dir = project_root / cfg["paths"]["metrics_dir"]
    figures_dir = project_root / cfg["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "baseline_results.json") as f:
        baseline_results = json.load(f)

    logger.info("Evaluating GCN")
    gcn = evaluate_gcn(metrics_dir)
    gcn["temporal_df"].to_csv(metrics_dir / "temporal_gcn.csv", index=False)

    make_comparison_figure(gcn["test_metrics"], baseline_results, figures_dir / "gcn_vs_baselines_comparison.png")
    make_f1_over_time_figure(gcn["temporal_df"], figures_dir / "gcn_f1_over_time.png")
    make_pr_roc_figures(gcn["raw"], metrics_dir, figures_dir)
    make_confusion_matrix_figure(gcn["raw"], gcn["threshold"], figures_dir / "gcn_confusion_matrix.png")

    output = {
        "model": "GCN (basic, 2-layer GCNConv)",
        "threshold": gcn["threshold"],
        "val_metrics": gcn["val_metrics"],
        "test_metrics": gcn["test_metrics"],
        "comparison_test_f1": {
            "logreg_no_tstep": baseline_results["primary_models"]["logreg_no_tstep"]["test_metrics"]["f1"],
            "random_forest_no_tstep": baseline_results["primary_models"]["random_forest_no_tstep"]["test_metrics"]["f1"],
            "gcn": gcn["test_metrics"]["f1"],
        },
    }
    with open(metrics_dir / "gcn_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("GCN evaluation complete. Results at %s", metrics_dir / "gcn_results.json")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    result = main(args.config)
    print(json.dumps(result, indent=2, default=str))
