"""Phase 6 evaluation: Static GraphSAGE vs Adaptive GraphSAGE (k=1 primary).

Usage:
    python -m scripts.evaluate_adaptive_graphsage --config config.yaml

The frozen Phase 5 validation threshold (0.898, from
results/metrics/gnn_improvement_train_manifest.json) is used for BOTH models
and is never re-selected. Random Forest appears only as a secondary reference.

Reads (read-only): gnn_improvement_train_manifest.json, gnn_improvement_test.npz,
adaptive_graphsage_test{,_k3}.npz, baseline_results.json.

Saves:
    results/metrics/adaptive_graphsage_results.json
    results/metrics/temporal_adaptive_graphsage.csv
    results/metrics/temporal_static_vs_adaptive_graphsage_f1_delta.csv
    results/figures/adaptive_graphsage_comparison.png
    results/figures/adaptive_graphsage_f1_over_time.png
    results/figures/adaptive_graphsage_f1_delta.png
    results/figures/adaptive_graphsage_pr_roc_curves.png
    results/figures/adaptive_graphsage_confusion_matrices.png
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
from scipy.stats import wilcoxon
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.temporal_metrics import compute_metrics_by_time_step

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate_adaptive_graphsage")


def load_predictions(metrics_dir: Path, filename: str):
    npz = np.load(metrics_dir / "predictions" / filename)
    return npz["y_true"], npz["y_prob"], npz["time_step"]


def load_frozen_threshold(metrics_dir: Path) -> float:
    """Phase 5's validation-F1 threshold for the selected GraphSAGE. Read once;
    never recomputed from test data."""
    with open(metrics_dir / "gnn_improvement_train_manifest.json") as f:
        return json.load(f)["selected"]["selected_threshold_on_validation"]


def evaluate_predictions(y_true, y_prob, time_step, threshold: float) -> dict:
    """Metrics at a FIXED, externally supplied threshold; never selects one."""
    return {
        "threshold": threshold,
        "test_metrics": compute_binary_metrics(y_true, y_prob, threshold),
        "temporal_df": compute_metrics_by_time_step(y_true, y_prob, time_step, threshold),
    }


def exploratory_wilcoxon(delta: np.ndarray) -> dict:
    """Paired Wilcoxon signed-rank on per-step F1 deltas. Exploratory only:
    15 temporally ordered, non-independent observations."""
    n_nonzero = int((delta != 0).sum())
    if n_nonzero == 0:
        return {"n_steps": int(len(delta)), "n_nonzero_deltas": 0, "statistic": None, "p_value": None}
    res = wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
    return {"n_steps": int(len(delta)), "n_nonzero_deltas": n_nonzero,
            "statistic": float(res.statistic), "p_value": float(res.pvalue)}


def bar_comparison(static_m, adaptive_m, rf_m, out_path: Path) -> None:
    names = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    series = [("Random Forest (secondary ref.)", rf_m, "#95a5a6"),
              ("Static GraphSAGE", static_m, "#2980b9"),
              ("Adaptive GraphSAGE", adaptive_m, "#27ae60")]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (label, m, color) in enumerate(series):
        ax.bar(x + i * 0.27, [m[k] for k in names], 0.27, label=label, color=color)
    ax.set_xticks(x + 0.27)
    ax.set_xticklabels(["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"])
    ax.set_ylim(0, 1)
    ax.set_title("Static vs Adaptive GraphSAGE (test 35-49, shared threshold)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def f1_over_time(static_t, adaptive_t, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(static_t["time_step"], static_t["f1"], marker="o", markersize=4, color="#2980b9", label="Static GraphSAGE")
    ax.plot(adaptive_t["time_step"], adaptive_t["f1"], marker="s", markersize=4, color="#27ae60", label="Adaptive GraphSAGE")
    ax.axvline(42.5, color="gray", linestyle=":", linewidth=1, label="t=42/43 boundary")
    ax.set_xlabel("time_step"); ax.set_ylabel("F1"); ax.set_ylim(0, 1)
    ax.set_title("Static vs Adaptive GraphSAGE: F1 per test time step")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def delta_plot(delta_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    vals = delta_df["f1_delta"].fillna(0)
    ax.bar(delta_df["time_step"], vals, color=["#27ae60" if v >= 0 else "#c0392b" for v in vals])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(42.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("time_step"); ax.set_ylabel("F1 delta (Adaptive - Static)")
    ax.set_title("Per-step F1 delta: Adaptive GraphSAGE - Static GraphSAGE")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def pr_roc(static_raw, adaptive_raw, out_path: Path) -> None:
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(12, 5))
    for (y, p), label, style in [(static_raw, "Static GraphSAGE", "-"), (adaptive_raw, "Adaptive GraphSAGE", "--")]:
        prec, rec, _ = precision_recall_curve(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        ax_pr.plot(rec, prec, label=label, linestyle=style)
        ax_roc.plot(fpr, tpr, label=label, linestyle=style)
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision"); ax_pr.set_title("Precision-Recall (test)"); ax_pr.legend()
    ax_roc.plot([0, 1], [0, 1], linestyle=":", color="gray")
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR"); ax_roc.set_title("ROC (test)"); ax_roc.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def confusion_plot(static_raw, adaptive_raw, threshold: float, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (y, p), title in zip(axes, [static_raw, adaptive_raw], ["Static GraphSAGE", "Adaptive GraphSAGE"]):
        ConfusionMatrixDisplay.from_predictions(y, (p >= threshold).astype(int),
                                                display_labels=["licit", "illicit"], ax=ax, colorbar=False)
        ax.set_title(title)
    fig.suptitle("Confusion matrices (test 35-49, shared threshold)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    project_root = Path(config_path).resolve().parent
    metrics_dir = project_root / cfg["paths"]["metrics_dir"]
    figures_dir = project_root / cfg["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "baseline_results.json") as f:
        baseline = json.load(f)
    threshold = load_frozen_threshold(metrics_dir)
    logger.info("Frozen Phase 5 threshold (both models): %.4f", threshold)

    sy, sp, st = load_predictions(metrics_dir, "gnn_improvement_test.npz")
    ay, ap, at = load_predictions(metrics_dir, "adaptive_graphsage_test.npz")
    static_eval = evaluate_predictions(sy, sp, st, threshold)
    adaptive_eval = evaluate_predictions(ay, ap, at, threshold)
    adaptive_eval["temporal_df"].to_csv(metrics_dir / "temporal_adaptive_graphsage.csv", index=False)

    delta_df = static_eval["temporal_df"][["time_step", "f1", "n_labeled", "illicit_rate"]].rename(columns={"f1": "static_f1"})
    delta_df["adaptive_f1"] = adaptive_eval["temporal_df"]["f1"].values
    delta_df["f1_delta"] = delta_df["adaptive_f1"] - delta_df["static_f1"]
    delta_df.to_csv(metrics_dir / "temporal_static_vs_adaptive_graphsage_f1_delta.csv", index=False)

    k3_path = metrics_dir / "predictions" / "adaptive_graphsage_test_k3.npz"
    k3_summary = None
    if k3_path.exists():
        ky, kp, kt = load_predictions(metrics_dir, "adaptive_graphsage_test_k3.npz")
        k3_summary = {"feedback_delay_k": 3, "test_metrics": evaluate_predictions(ky, kp, kt, threshold)["test_metrics"],
                      "note": "Sensitivity check only; not used for any selection."}

    rf_m = baseline["primary_models"]["random_forest_no_tstep"]["test_metrics"]
    bar_comparison(static_eval["test_metrics"], adaptive_eval["test_metrics"], rf_m, figures_dir / "adaptive_graphsage_comparison.png")
    f1_over_time(static_eval["temporal_df"], adaptive_eval["temporal_df"], figures_dir / "adaptive_graphsage_f1_over_time.png")
    delta_plot(delta_df, figures_dir / "adaptive_graphsage_f1_delta.png")
    pr_roc((sy, sp), (ay, ap), figures_dir / "adaptive_graphsage_pr_roc_curves.png")
    confusion_plot((sy, sp), (ay, ap), threshold, figures_dir / "adaptive_graphsage_confusion_matrices.png")

    sm, am = static_eval["test_metrics"], adaptive_eval["test_metrics"]
    output = {
        "primary_comparison": "Static GraphSAGE vs Adaptive GraphSAGE (k=1, simulated delayed feedback)",
        "frozen_threshold_from_phase5_validation": threshold,
        "static_graphsage_test_metrics": sm,
        "adaptive_graphsage_test_metrics": am,
        "metric_changes_adaptive_minus_static": {k: am[k] - sm[k] for k in ["precision", "recall", "f1", "roc_auc", "pr_auc"]},
        "f1_delta_summary": {
            "mean_delta_t35_42": float(delta_df[delta_df["time_step"] <= 42]["f1_delta"].mean()),
            "mean_delta_t43_49": float(delta_df[delta_df["time_step"] >= 43]["f1_delta"].mean()),
            "mean_delta_all": float(delta_df["f1_delta"].mean()),
        },
        "exploratory_wilcoxon_f1_deltas": exploratory_wilcoxon(delta_df["f1_delta"].to_numpy()),
        "random_forest_reference_test_f1": rf_m["f1"],
        "k3_sensitivity": k3_summary,
    }
    with open(metrics_dir / "adaptive_graphsage_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    print(json.dumps(main(args.config), indent=2, default=str))
