"""Phase 5 evaluation entry point: selected static GNN vs LogReg, RF, and
the frozen Phase 3 basic GCN.

Usage:
    python -m scripts.evaluate_gnn_improvement --config config.yaml

Reads (read-only — nothing under Phase 1/2/3/4 is modified):
    results/metrics/gnn_improvement_train_manifest.json (selected config + threshold)
    results/metrics/predictions/gnn_improvement_test.npz (selected model, test)
    results/metrics/predictions/gcn_test.npz             (Phase 3 basic GCN, test)
    results/metrics/baseline_results.json                (RF/LogReg, secondary reference)

The threshold used here is the one already selected on validation F1 for
THIS model in scripts/train_gnn_improvement.py (read from the manifest,
never recomputed here, never touched against test data).

Saves:
    results/metrics/gnn_improvement_results.json
    results/metrics/temporal_gnn_improvement.csv
    results/figures/gnn_improvement_vs_all_comparison.png
    results/figures/gnn_improvement_f1_over_time.png
    results/figures/gnn_improvement_pr_roc_curves.png
    results/figures/gnn_improvement_confusion_matrix.png
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

from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.temporal_metrics import compute_metrics_by_time_step

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate_gnn_improvement")


def load_predictions(metrics_dir: Path, filename: str):
    npz = np.load(metrics_dir / "predictions" / filename)
    return npz["y_true"], npz["y_prob"], npz["time_step"]


def make_comparison_figure(results: dict, out_path: Path) -> None:
    metric_names = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    labels = ["Logistic Regression", "Random Forest", "Basic GCN (Phase 3)", "Selected Phase 5 model"]
    keys = ["logreg", "random_forest", "basic_gcn", "phase5_selected"]
    x = np.arange(len(metric_names))
    width = 0.2
    colors = ["#95a5a6", "#bdc3c7", "#c0392b", "#2980b9"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, key in enumerate(keys):
        vals = [results[key][m] for m in metric_names]
        ax.bar(x + i * width, vals, width, label=labels[i], color=colors[i])
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Phase 5: selected static GNN vs LogReg, RF, Basic GCN (test set, 35-49)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def make_f1_over_time_figure(basic_temporal, selected_temporal, selected_label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(basic_temporal["time_step"], basic_temporal["f1"], marker="o", markersize=4,
            color="#c0392b", label="Basic GCN (Phase 3)", linewidth=1.5)
    ax.plot(selected_temporal["time_step"], selected_temporal["f1"], marker="s", markersize=4,
            color="#2980b9", label=selected_label, linewidth=1.5)
    ax.axvline(43, color="gray", linestyle=":", linewidth=1, label="t=43 (documented collapse point)")
    ax.set_xlabel("time_step")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_title(f"Basic GCN vs {selected_label}: F1 over test time steps (35-49)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def make_pr_roc_figures(basic_raw, selected_raw, selected_label: str, out_path: Path) -> None:
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(12, 5))
    for (y_true, y_prob, label, style) in [
        (basic_raw[0], basic_raw[1], "Basic GCN (Phase 3)", "-"),
        (selected_raw[0], selected_raw[1], selected_label, "--"),
    ]:
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax_pr.plot(rec, prec, label=label, linestyle=style)
        ax_roc.plot(fpr, tpr, label=label, linestyle=style)
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall (test set)"); ax_pr.legend()
    ax_roc.plot([0, 1], [0, 1], linestyle=":", color="gray", linewidth=1)
    ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC (test set)"); ax_roc.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def make_confusion_matrix_figure(y_true, y_prob, threshold: float, label: str, out_path: Path) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["licit", "illicit"], ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix: {label}\ntest set, time steps 35-49")
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

    with open(metrics_dir / "gnn_improvement_train_manifest.json") as f:
        train_manifest = json.load(f)
    with open(metrics_dir / "gcn_results.json") as f:
        basic_gcn_results = json.load(f)
    with open(metrics_dir / "baseline_results.json") as f:
        baseline_results = json.load(f)

    selected = train_manifest["selected"]
    threshold = selected["selected_threshold_on_validation"]
    selected_label = f"{selected['architecture']} (h={selected['hidden_channels']}, drop={selected['dropout']}, lr={selected['learning_rate']})"
    logger.info("Selected model: %s | threshold (from validation F1, frozen): %.4f", selected_label, threshold)

    sel_y, sel_p, sel_t = load_predictions(metrics_dir, "gnn_improvement_test.npz")
    basic_y, basic_p, basic_t = load_predictions(metrics_dir, "gcn_test.npz")

    sel_test_metrics = compute_binary_metrics(sel_y, sel_p, threshold)
    sel_temporal = compute_metrics_by_time_step(sel_y, sel_p, sel_t, threshold)
    sel_temporal.to_csv(metrics_dir / "temporal_gnn_improvement.csv", index=False)

    basic_temporal = compute_metrics_by_time_step(basic_y, basic_p, basic_t, basic_gcn_results["threshold"])

    comparison = {
        "logreg": baseline_results["primary_models"]["logreg_no_tstep"]["test_metrics"],
        "random_forest": baseline_results["primary_models"]["random_forest_no_tstep"]["test_metrics"],
        "basic_gcn": basic_gcn_results["test_metrics"],
        "phase5_selected": sel_test_metrics,
    }

    make_comparison_figure(comparison, figures_dir / "gnn_improvement_vs_all_comparison.png")
    make_f1_over_time_figure(basic_temporal, sel_temporal, selected_label, figures_dir / "gnn_improvement_f1_over_time.png")
    make_pr_roc_figures((basic_y, basic_p), (sel_y, sel_p), selected_label, figures_dir / "gnn_improvement_pr_roc_curves.png")
    make_confusion_matrix_figure(sel_y, sel_p, threshold, selected_label, figures_dir / "gnn_improvement_confusion_matrix.png")

    # per-time-step F1 delta vs Basic GCN, for "is the improvement consistent" reporting
    delta_df = basic_temporal[["time_step", "f1", "n_labeled", "illicit_rate"]].rename(columns={"f1": "basic_gcn_f1"})
    delta_df["phase5_selected_f1"] = sel_temporal["f1"].values
    delta_df["f1_delta"] = delta_df["phase5_selected_f1"] - delta_df["basic_gcn_f1"]
    delta_df.to_csv(metrics_dir / "temporal_gnn_improvement_vs_basic_gcn_delta.csv", index=False)

    n_steps_improved = int((delta_df["f1_delta"] > 0).sum())
    n_steps_worse = int((delta_df["f1_delta"] < 0).sum())
    n_steps_equal = int((delta_df["f1_delta"] == 0).sum())

    output = {
        "selected_model": selected,
        "threshold_selected_on_validation_f1": threshold,
        "test_metrics": {
            "logreg_no_tstep": comparison["logreg"],
            "random_forest_no_tstep": comparison["random_forest"],
            "basic_gcn_phase3": comparison["basic_gcn"],
            "phase5_selected_model": comparison["phase5_selected"],
        },
        "temporal_consistency_vs_basic_gcn": {
            "n_time_steps_improved": n_steps_improved,
            "n_time_steps_worse": n_steps_worse,
            "n_time_steps_unchanged": n_steps_equal,
            "mean_f1_delta_all_steps": float(delta_df["f1_delta"].mean()),
            "mean_f1_delta_before_t43": float(delta_df[delta_df["time_step"] < 43]["f1_delta"].mean()),
            "mean_f1_delta_from_t43_onward": float(delta_df[delta_df["time_step"] >= 43]["f1_delta"].mean()),
        },
    }
    with open(metrics_dir / "gnn_improvement_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("Phase 5 evaluation complete. Results at %s", metrics_dir / "gnn_improvement_results.json")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    result = main(args.config)
    print(json.dumps(result, indent=2, default=str))
