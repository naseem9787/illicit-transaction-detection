"""Binary classification metrics, and threshold selection that only ever
looks at validation data (never test) — the caller is responsible for
passing the right split in.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_threshold_maximizing_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Sweep thresholds from a precision-recall curve, return the one with
    the best F1. Must only ever be called with validation data.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns len(thresholds) == len(precisions) - 1
    f1s = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12),
        0.0,
    )
    if len(f1s) == 0:
        return 0.5
    best_idx = int(np.argmax(f1s))
    return float(thresholds[best_idx])


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """All metrics needed for the required reporting table, at a fixed,
    already-chosen threshold. Rank-based metrics (ROC-AUC, PR-AUC) don't
    depend on the threshold and are included for completeness.
    """
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    can_compute_auc = n_pos > 0 and n_neg > 0

    return {
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "n_illicit": n_pos,
        "n_licit": n_neg,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if can_compute_auc else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if can_compute_auc else None,
        "accuracy": float(accuracy),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
