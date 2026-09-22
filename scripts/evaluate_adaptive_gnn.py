"""Phase 4 evaluation entry point: Static GCN vs Adaptive GCN.

Usage:
    python -m scripts.evaluate_adaptive_gnn --config config.yaml

PRIMARY comparison: Static GCN (Phase 3, frozen) vs Adaptive GCN (Phase 4,
k=1, simulated delayed feedback). Random Forest and Logistic Regression are
included in the summary figure only as secondary reference points, per the
Phase 4 design brief — they are not the scientific comparison this script
exists to make.

The SAME threshold already selected once on validation for the Static GCN
(results/metrics/gcn_results.json, threshold=0.863) is reused for the
Adaptive GCN rather than re-selecting a second threshold. This isolates
online weight adaptation as the only variable between the two models —
not a second, independently-chosen threshold. No threshold is ever
selected using test data, for either model.

Reads (read-only — nothing under Phase 1/2/3 is modified):
    results/metrics/gcn_results.json                  (Static GCN threshold + test metrics)
    results/metrics/predictions/gcn_test.npz           (Static GCN test predictions)
    results/metrics/predictions/adaptive_gcn_test.npz  (Adaptive GCN, k=1, primary)
    results/metrics/predictions/adaptive_gcn_test_k3.npz (Adaptive GCN, k=3, sensitivity)
    results/metrics/baseline_results.json              (RF/LogReg, secondary reference only)

Saves:
    results/metrics/adaptive_results.json
    results/metrics/temporal_adaptive_gcn.csv
    results/metrics/temporal_static_vs_adaptive_f1_delta.csv
    results/figures/adaptive_vs_static_comparison.png
    results/figures/per_time_step_f1_static_vs_adaptive.png
    results/figures/f1_delta_over_time.png
    results/figures/pr_roc_curves_adaptive_vs_static.png
    results/figures/adaptive_vs_static_confusion_matrices.png
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
import pandas as pd
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.temporal_metrics import compute_metrics_by_time_step

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate_adaptive_gnn")


def load_predictions(metrics_dir: Path, filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    npz = np.load(metrics_dir / "predictions" / filename)
    return npz["y_true"], npz["y_prob"], npz["time_step"]


def evaluate_predictions(y_true, y_prob, time_step, threshold: float) -> dict:
    """Compute test metrics + per-time-step breakdown at a FIXED, externally
    supplied threshold. Never calls select_threshold_maximizing_f1 — that
    function is only ever used once, on validation, for the Static GCN in
    scripts/evaluate_gnn.py. See tests/test_adaptive_gnn.py for a test that
    asserts this function never re-selects a threshold."""
    test_metrics = compute_binary_metrics(y_true, y_prob, threshold)
    temporal_df = compute_metrics_by_time_step(y_true, y_prob, time_step, threshold)
    return {"threshold": threshold, "test_metrics": test_metrics, "temporal_df": temporal_df}


def make_comparison_figure(static_metrics, adaptive_metrics, baseline_results, out_path: Path) -> None:
    metric_names = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    series = {
        "Random Forest (secondary ref.)": [baseline_results["primary_models"]["random_forest_no_tstep"]["test_metrics"][m] for m in metric_names],
        "Logistic Regression (secondary ref.)": [baseline_results["primary_models"]["logreg_no_tstep"]["test_metrics"][m] for m in metric_names],
        "Static GCN": [static_metrics[m] for m in metric_names],
        "Adaptive GCN": [adaptive_metrics[m] for m in metric_names],
    }
    x = np.arange(len(metric_names))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#95a5a6", "#bdc3c7", "#c0392b", "#27ae60"]
    for i, (label, vals) in enumerate(series.items()):
        ax.bar(x + i * width, vals, width, label=label, color=colors[i])
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("PRIMARY: Static GCN vs Adaptive GCN (test set, 35-49)\nRF/LogReg shown as secondary reference only")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def make_f1_over_time_figure(static_temporal: pd.DataFrame, adaptive_temporal: pd.DataFrame, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(static_temporal["time_step"], static_temporal["f1"], marker="o", markersize=4,
              color="#c0392b", label="Static GCN", linewidth=1.5)
    ax1.plot(adaptive_temporal["time_step"], adaptive_temporal["f1"], marker="s", markersize=4,
              color="#27ae60", label="Adaptive GCN", linewidth=1.5)
    ax1.axvline(43, color="gray", linestyle=":", linewidth=1, label="t=43 (documented collapse point)")
    ax1.set_xlabel("time_step")
    ax1.set_ylabel("F1")
    ax1.set_ylim(0, 1)
    ax1.set_title("Static vs Adaptive GCN: F1 over test time steps (35-49)")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def make_f1_delta_figure(delta_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["#27ae60" if v >= 0 else "#c0392b" for v in delta_df["f1_delta"].fillna(0)]
    ax.bar(delta_df["time_step"], delta_df["f1_delta"].fillna(0), color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(42.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("time_step")
    ax.set_ylabel("F1 delta (Adaptive − Static)")
    ax.set_title("Per-time-step F1 delta: Adaptive GCN − Static GCN\n(positive = adaptive did better; dotted line marks the t=43 collapse boundary)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def make_pr_roc_figures(static_raw, adaptive_raw, out_path: Path) -> None:
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(12, 5))
    for (y_true, y_prob, label, style) in [
        (static_raw[0], static_raw[1], "Static GCN", "-"),
        (adaptive_raw[0], adaptive_raw[1], "Adaptive GCN", "--"),
    ]:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax_pr.plot(rec, prec, label=label, linestyle=style)
        ax_roc.plot(fpr, tpr, label=label, linestyle=style)

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall (test set)")
    ax_pr.legend()

    ax_roc.plot([0, 1], [0, 1], linestyle=":", color="gray", linewidth=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC (test set)")
    ax_roc.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def make_confusion_matrices_figure(static_raw, adaptive_raw, threshold: float, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (y_true, y_prob, title) in zip(axes, [
        (static_raw[0], static_raw[1], "Static GCN"),
        (adaptive_raw[0], adaptive_raw[1], "Adaptive GCN"),
    ]):
        y_pred = (y_prob >= threshold).astype(int)
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=["licit", "illicit"], ax=ax, colorbar=False
        )
        ax.set_title(title)
    fig.suptitle("Confusion matrices (test set, time steps 35-49, shared threshold)")
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

    with open(metrics_dir / "gcn_results.json") as f:
        static_results = json.load(f)
    with open(metrics_dir / "baseline_results.json") as f:
        baseline_results = json.load(f)

    threshold = static_results["threshold"]
    logger.info("Reusing Static GCN's validation-selected threshold: %.4f (never re-selected)", threshold)

    static_y, static_p, static_t = load_predictions(metrics_dir, "gcn_test.npz")
    adaptive_y, adaptive_p, adaptive_t = load_predictions(metrics_dir, "adaptive_gcn_test.npz")

    static_eval = evaluate_predictions(static_y, static_p, static_t, threshold)
    adaptive_eval = evaluate_predictions(adaptive_y, adaptive_p, adaptive_t, threshold)

    adaptive_eval["temporal_df"].to_csv(metrics_dir / "temporal_adaptive_gcn.csv", index=False)

    delta_df = static_eval["temporal_df"][["time_step", "f1", "n_labeled", "illicit_rate"]].rename(columns={"f1": "static_f1"})
    delta_df["adaptive_f1"] = adaptive_eval["temporal_df"]["f1"].values
    delta_df["f1_delta"] = delta_df["adaptive_f1"] - delta_df["static_f1"]
    delta_df.to_csv(metrics_dir / "temporal_static_vs_adaptive_f1_delta.csv", index=False)

    # k=3 sensitivity (secondary, reported but not used for any decision)
    k3_path = metrics_dir / "predictions" / "adaptive_gcn_test_k3.npz"
    k3_summary = None
    if k3_path.exists():
        k3_y, k3_p, k3_t = load_predictions(metrics_dir, "adaptive_gcn_test_k3.npz")
        k3_eval = evaluate_predictions(k3_y, k3_p, k3_t, threshold)
        k3_summary = {"feedback_delay_k": 3, "test_metrics": k3_eval["test_metrics"],
                       "note": "Sensitivity check only; NOT compared against k=1 to make any selection."}

    make_comparison_figure(static_eval["test_metrics"], adaptive_eval["test_metrics"], baseline_results,
                            figures_dir / "adaptive_vs_static_comparison.png")
    make_f1_over_time_figure(static_eval["temporal_df"], adaptive_eval["temporal_df"],
                              figures_dir / "per_time_step_f1_static_vs_adaptive.png")
    make_f1_delta_figure(delta_df, figures_dir / "f1_delta_over_time.png")
    make_pr_roc_figures((static_y, static_p), (adaptive_y, adaptive_p),
                         figures_dir / "pr_roc_curves_adaptive_vs_static.png")
    make_confusion_matrices_figure((static_y, static_p), (adaptive_y, adaptive_p), threshold,
                                    figures_dir / "adaptive_vs_static_confusion_matrices.png")

    f1_after_43 = delta_df[delta_df["time_step"] >= 43]["f1_delta"].mean()
    f1_before_43 = delta_df[delta_df["time_step"] < 43]["f1_delta"].mean()

    output = {
        "primary_comparison": "Static GCN vs Adaptive GCN (k=1, simulated delayed feedback)",
        "threshold_reused_from_static_gcn": threshold,
        "static_gcn_test_metrics": static_eval["test_metrics"],
        "adaptive_gcn_test_metrics": adaptive_eval["test_metrics"],
        "f1_delta_summary": {
            "mean_f1_delta_all_test_steps": float(delta_df["f1_delta"].mean()),
            "mean_f1_delta_before_t43": float(f1_before_43) if pd.notna(f1_before_43) else None,
            "mean_f1_delta_from_t43_onward": float(f1_after_43) if pd.notna(f1_after_43) else None,
        },
        "secondary_reference_baselines_test_f1": {
            "logreg_no_tstep": baseline_results["primary_models"]["logreg_no_tstep"]["test_metrics"]["f1"],
            "random_forest_no_tstep": baseline_results["primary_models"]["random_forest_no_tstep"]["test_metrics"]["f1"],
        },
        "k3_sensitivity": k3_summary,
    }
    with open(metrics_dir / "adaptive_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("Phase 4 evaluation complete. Results at %s", metrics_dir / "adaptive_results.json")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    result = main(args.config)
    print(json.dumps(result, indent=2, default=str))
