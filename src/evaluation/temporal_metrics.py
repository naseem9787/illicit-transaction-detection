"""Per-time-step metrics, for inspecting whether performance is stable
across the test period (35-49) or drifts — the central question for the
eventual adaptive-vs-static GNN comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_binary_metrics

MIN_LABELED_FOR_METRICS = 5  # below this, precision/recall/F1 are too noisy to report


def compute_metrics_by_time_step(
    y_true: np.ndarray, y_prob: np.ndarray, time_steps: np.ndarray, threshold: float
) -> pd.DataFrame:
    rows = []
    for t in sorted(np.unique(time_steps)):
        mask = time_steps == t
        yt, yp = y_true[mask], y_prob[mask]
        n_labeled = int(len(yt))
        n_illicit = int((yt == 1).sum())
        illicit_rate = n_illicit / n_labeled if n_labeled else float("nan")

        row = {
            "time_step": int(t),
            "n_labeled": n_labeled,
            "n_illicit": n_illicit,
            "illicit_rate": illicit_rate,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "pr_auc": None,
        }
        has_both_classes = n_illicit > 0 and n_illicit < n_labeled
        if n_labeled >= MIN_LABELED_FOR_METRICS and has_both_classes:
            m = compute_binary_metrics(yt, yp, threshold)
            row.update(
                {
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "roc_auc": m["roc_auc"],
                    "pr_auc": m["pr_auc"],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)
