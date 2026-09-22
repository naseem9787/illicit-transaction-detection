"""Baseline B: Random Forest on the same feature inputs as Logistic
Regression, so the comparison between the two is fair.

Model selection is deliberately shallow: 3 candidate configurations,
picked by validation PR-AUC (the imbalance-appropriate ranking metric),
not an exhaustive grid search. This is a college-project baseline, not a
Kaggle submission — see docs/DECISIONS.md.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)

CANDIDATE_CONFIGS = [
    {"n_estimators": 300, "max_depth": 8},
    {"n_estimators": 300, "max_depth": 16},
    {"n_estimators": 300, "max_depth": None},
]


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> tuple[RandomForestClassifier, dict]:
    best_model, best_config, best_pr_auc = None, None, -1.0
    selection_log = []

    for cfg in CANDIDATE_CONFIGS:
        model = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, val_prob)
        selection_log.append({**cfg, "val_pr_auc": float(pr_auc)})
        logger.info("RF config %s -> val PR-AUC=%.4f", cfg, pr_auc)
        if pr_auc > best_pr_auc:
            best_model, best_config, best_pr_auc = model, cfg, pr_auc

    logger.info("Selected RF config %s (val PR-AUC=%.4f)", best_config, best_pr_auc)
    return best_model, {"selected_config": best_config, "selection_log": selection_log}
