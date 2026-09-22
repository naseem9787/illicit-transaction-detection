"""Baseline A: Logistic Regression on transaction feature vectors alone.

No graph information, no txId, no raw label leakage. class_weight='balanced'
handles the imbalance by reweighting the loss using the TRAINING label
distribution only (scikit-learn computes this internally from y at fit
time) — this does not touch validation/test data, so it introduces no
leakage.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> LogisticRegression:
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    logger.info("Logistic regression fit on %d samples, %d features", *X_train.shape)
    return model
