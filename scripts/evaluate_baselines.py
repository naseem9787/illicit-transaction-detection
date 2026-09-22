"""Phase 2 evaluation entry point.

Usage:
    python -m scripts.evaluate_baselines --config config.yaml

Reads the predictions saved by scripts/train_baselines.py (never re-touches
the test set for fitting anything). For each primary model:
    1. Selects a decision threshold on VALIDATION ONLY (maximize F1).
    2. Freezes that threshold and evaluates on TEST.
    3. Computes per-time-step metrics for the test period (35-49).

Saves:
    results/metrics/baseline_results.json      (val + test metrics, thresholds)
    results/metrics/temporal_{model}.csv        (per-time-step test metrics)
    results/figures/baseline_metric_comparison.png
    results/figures/f1_over_time_logreg.png
    results/figures/f1_over_time_random_forest.png
    results/figures/pr_curves.png
    results/figures/roc_curves.png
    results/figures/confusion_matrix_best.png
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
logger = logging.getLogger("evaluate_baselines")

# Only the primary (no time_step) models are compared head-to-head.
# logreg_with_tstep is evaluated separately and reported as a secondary note.
PRIMARY_MODELS = {
    "logreg_no_tstep": "Logistic Regression",
    "random_forest_no_tstep": "Random Forest",
}
SECONDARY_MODELS = {"logreg_with_tstep": "Logistic Regression (+ time_step, secondary)"}


def load_predictions(metrics_dir: Path, name: str, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    npz = np.load(metrics_dir / "predictions" / f"{name}_{split}.npz")
    return npz["y_true"], npz["y_prob"], npz["time_step"]


def evaluate_model(name: str, metrics_dir: Path) -> dict:
    y_val, p_val, _ = load_predictions(metrics_dir, name, "val")
    y_test, p_test, t_test = load_predictions(metrics_dir, name, "test")

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


def make_comparison_figure(results: dict, out_path: Path) -> None:
    names = list(results.keys())
    labels = [PRIMARY_MODELS[n] for n in names]
    metric_names = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(names):
        vals = [results[name]["test_metrics"][m] for m in metric_names]
        ax.bar(x + i * width, vals, width, label=labels[i])
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Baseline comparison on TEST set (time steps 35-49)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def make_f1_over_time_figure(name: str, label: str, temporal_df, split_cfg: dict, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(temporal_df["time_step"], temporal_df["f1"], marker="o", markersize=4, color="#c0392b", label="F1")
    ax1.set_xlabel("time_step")
    ax1.set_ylabel("F1", color="#c0392b")
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis="y", labelcolor="#c0392b")

    ax2 = ax1.twinx()
    ax2.bar(temporal_df["time_step"], temporal_df["n_labeled"], alpha=0.15, color="gray", label="n labeled")
    ax2.set_ylabel("labeled transactions", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    ax1.set_title(f"{label}: F1 over test time steps (35-49)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def make_pr_roc_figures(results: dict, figures_dir: Path) -> None:
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    for name in results:
        y_test, p_test = results[name]["raw"]["y_test"], results[name]["raw"]["p_test"]
        prec, rec, _ = precision_recall_curve(y_test, p_test)
        fpr, tpr, _ = roc_curve(y_test, p_test)
        ax_pr.plot(rec, prec, label=PRIMARY_MODELS[name])
        ax_roc.plot(fpr, tpr, label=PRIMARY_MODELS[name])

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall curves (test set)")
    ax_pr.legend()
    fig_pr.tight_layout()
    fig_pr.savefig(figures_dir / "pr_curves.png", dpi=150)
    plt.close(fig_pr)

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC curves (test set)")
    ax_roc.legend()
    fig_roc.tight_layout()
    fig_roc.savefig(figures_dir / "roc_curves.png", dpi=150)
    plt.close(fig_roc)
    logger.info("Saved pr_curves.png and roc_curves.png")


def make_confusion_matrix_figure(best_name: str, results: dict, out_path: Path) -> None:
    y_test = results[best_name]["raw"]["y_test"]
    p_test = results[best_name]["raw"]["p_test"]
    threshold = results[best_name]["threshold"]
    y_pred = (p_test >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["licit", "illicit"], ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion matrix (best baseline: {PRIMARY_MODELS[best_name]})\ntest set, time steps 35-49")
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

    results = {}
    for name in PRIMARY_MODELS:
        logger.info("Evaluating %s", name)
        results[name] = evaluate_model(name, metrics_dir)
        results[name]["temporal_df"].to_csv(metrics_dir / f"temporal_{name}.csv", index=False)

    secondary_results = {}
    for name in SECONDARY_MODELS:
        logger.info("Evaluating secondary model %s", name)
        secondary_results[name] = evaluate_model(name, metrics_dir)
        secondary_results[name]["temporal_df"].to_csv(metrics_dir / f"temporal_{name}.csv", index=False)

    best_name = max(results, key=lambda n: results[n]["test_metrics"]["f1"])
    logger.info("Best primary baseline by test F1: %s", best_name)

    make_comparison_figure(results, figures_dir / "baseline_metric_comparison.png")
    for name in PRIMARY_MODELS:
        fig_name = "logreg" if "logreg" in name else "random_forest"
        make_f1_over_time_figure(
            name, PRIMARY_MODELS[name], results[name]["temporal_df"], cfg["split"],
            figures_dir / f"f1_over_time_{fig_name}.png",
        )
    make_pr_roc_figures(results, figures_dir)
    make_confusion_matrix_figure(best_name, results, figures_dir / "confusion_matrix_best.png")

    output = {
        "primary_models": {
            name: {
                "label": PRIMARY_MODELS[name],
                "threshold": results[name]["threshold"],
                "val_metrics": results[name]["val_metrics"],
                "test_metrics": results[name]["test_metrics"],
            }
            for name in results
        },
        "secondary_models": {
            name: {
                "label": SECONDARY_MODELS[name],
                "threshold": secondary_results[name]["threshold"],
                "val_metrics": secondary_results[name]["val_metrics"],
                "test_metrics": secondary_results[name]["test_metrics"],
            }
            for name in secondary_results
        },
        "best_primary_model_by_test_f1": best_name,
    }
    with open(metrics_dir / "baseline_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("Evaluation complete. Results at %s", metrics_dir / "baseline_results.json")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    result = main(args.config)
    print(json.dumps(result, indent=2, default=str))
