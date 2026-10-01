"""Step 2: adaptive-tree CONTROL for the delayed-feedback protocol.

Purpose: this is a CONTROL, not a new model family. Every prior adaptive
result in this project (Phases 4, 6, 7) compares an adaptive GNN against
its own static counterpart — never against a tabular model given the same
newly-revealed-label access. This module answers: "does a strong tabular
model show a similar gain, simply because it also gets new labels?" If it
does, the GNN-specific framing of the adaptive result weakens. If it
doesn't, that strengthens the claim that the GNN's adaptive gain is doing
something the tabular model can't.

MECHANISM (deliberately the simplest sklearn-native option, not invented):
HistGradientBoostingClassifier is not naturally gradient-incremental like
a neural network, but sklearn's own `warm_start=True` + incrementing
`max_iter` + calling `.fit()` again on NEW data is a documented, built-in
mechanism for adding more boosting rounds on top of an existing ensemble's
current state (verified directly before writing this module: refitting a
warm-started model on a different, smaller dataset does add new trees
fit against that new data's residuals, and does change predictions on
held-out data — this is genuine incremental adaptation, not a no-op).
`iters_per_update` (how many extra boosting rounds per revealed snapshot)
is a single fixed, pre-declared value — analogous in spirit to
`grad_steps` in the GNN protocol — never tuned against any result.

WALK-FORWARD PROTOCOL — identical ordering to
src/training/adaptive_gnn.py::run_adaptive_walk, applied to a tree instead
of a neural network:

    for t in [t0, t0+1, ..., tN]:
        1. If an earlier step's labels are due to be revealed now (k=1:
           immediately before predicting t), add `iters_per_update` more
           boosting rounds fit on that step's revealed labeled data.
        2. Predict t using the (possibly just-updated) ensemble.
        3. Freeze/log prediction(t) immediately — never revisited.
        4. Schedule t's own labels to be revealed k steps later.

FAIRNESS: Static HGB and Adaptive HGB start from an IDENTICAL fitted model
(a deep copy of the same trained ensemble) — matching the GNN protocol's
explicit fairness rule that static and adaptive must start from identical
weights, with the only difference being what happens after t0.

CLASS IMBALANCE: `class_weight="balanced"` cannot be used consistently
across incremental fits, because sklearn recomputes it from whatever `y`
is passed to that specific `.fit()` call — and a single revealed
snapshot's labels can be tiny and highly skewed (mirroring exactly the
concern already documented for the GNN protocol's fixed `pos_weight`, see
docs/EXPERIMENTS.md Phase 4 "Class imbalance handling in the adaptation
loop"). Instead, per-class weights are computed ONCE from the training
split only (`sklearn.utils.class_weight.compute_class_weight`) and applied
as an explicit, FIXED `sample_weight` on every fit call — the initial
training fit and every subsequent adaptation fit — never recomputed from
test-period or adaptation-period data.

NO FUTURE LABELS: only ever fits on time step t' when applying the
update scheduled to be revealed at the current step t = t' + k, and t' is
always < t. Never touches labels from t or later at prediction time.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_class_weight

from src.evaluation.metrics import compute_binary_metrics, select_threshold_maximizing_f1
from src.training.graph_features import HGB_BASE_CONFIG  # exact E2-selected config, frozen
from src.training.rolling_origin import E2_SEEDS, ROLLING_ORIGIN_FOLDS, TEST_RANGE, get_labeled_by_time_range

logger = logging.getLogger(__name__)

FEEDBACK_DELAY_K = 1  # primary, matches every other adaptive experiment in this project
ITERS_PER_UPDATE = 10  # fixed, pre-declared boosting-round increment per revealed snapshot
ADAPT_SEEDS = E2_SEEDS  # same 5 seeds as E2, for direct comparability


def compute_fixed_sample_weights(y_train: np.ndarray) -> dict[int, float]:
    """Per-class weights computed ONCE from training labels only — the
    tree analogue of src/training/gnn_training.py::compute_pos_weight.
    Never recomputed from adaptation-period or test-period data."""
    classes = np.array([0, 1])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {0: float(weights[0]), 1: float(weights[1])}


def _sample_weight(y: np.ndarray, class_weights: dict[int, float]) -> np.ndarray:
    return np.where(y == 1, class_weights[1], class_weights[0]).astype(np.float64)


def fit_base_model(X_train: np.ndarray, y_train: np.ndarray, seed: int, class_weights: dict[int, float]) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_depth=HGB_BASE_CONFIG["max_depth"], learning_rate=HGB_BASE_CONFIG["learning_rate"],
        max_iter=HGB_BASE_CONFIG["max_iter"], min_samples_leaf=HGB_BASE_CONFIG["min_samples_leaf"],
        warm_start=True, random_state=seed,
    )
    model.fit(X_train, y_train, sample_weight=_sample_weight(y_train, class_weights))
    return model


@dataclass
class TreeWalkResult:
    predictions: dict  # t -> (y_true, y_prob)
    predict_order: list
    events: list  # {"type": "predict"|"adapt", ...} — mirrors adaptive_gnn.py's event log exactly


def run_adaptive_tree_walk(
    base_model: HistGradientBoostingClassifier,
    ordered_time_steps: list[int],
    snapshots_by_t: dict[int, tuple[np.ndarray, np.ndarray]],
    class_weights: dict[int, float],
    iters_per_update: int = ITERS_PER_UPDATE,
    feedback_delay_k: int = FEEDBACK_DELAY_K,
) -> TreeWalkResult:
    """`base_model` is mutated in place (deep-copy it before calling if the
    caller needs an unmodified starting point — exactly the same contract
    as src/training/adaptive_gnn.py::run_adaptive_walk)."""
    model = base_model
    pending_reveals: dict[int, int] = {}
    predictions: dict[int, tuple] = {}
    predict_order: list[int] = []
    events: list[dict] = []

    for t in ordered_time_steps:
        if t in pending_reveals:
            source_t = pending_reveals.pop(t)
            X_source, y_source = snapshots_by_t[source_t]
            model.set_params(max_iter=model.n_iter_ + iters_per_update)
            model.fit(X_source, y_source, sample_weight=_sample_weight(y_source, class_weights))
            events.append({
                "type": "adapt", "applied_before_predicting": t, "source_time_step": source_t,
                "n_labeled_used": int(len(y_source)), "n_iter_after": int(model.n_iter_),
            })

        X_t, y_t = snapshots_by_t[t]
        y_prob = model.predict_proba(X_t)[:, 1]
        predictions[t] = (y_t.copy(), y_prob.copy())
        predict_order.append(t)
        events.append({"type": "predict", "step": t, "n_iter": int(model.n_iter_), "n_labeled": len(y_t)})

        pending_reveals[t + feedback_delay_k] = t

    return TreeWalkResult(predictions=predictions, predict_order=predict_order, events=events)


def pooled_predictions(result: TreeWalkResult, time_steps: list[int] | None = None):
    ts = time_steps if time_steps is not None else result.predict_order
    y_true = np.concatenate([result.predictions[t][0] for t in ts])
    y_prob = np.concatenate([result.predictions[t][1] for t in ts])
    return y_true, y_prob


def run_adaptive_tree_control(node_table, edges, feature_cols) -> list[dict]:
    """Static HGB vs Adaptive HGB, rolling-origin, ADAPT_SEEDS, validation
    only. `edges` is accepted for interface symmetry with the graph
    experiments but unused — this is a purely tabular control."""
    del edges
    rows = []
    for fold in ROLLING_ORIGIN_FOLDS:
        assert fold["val_end"] < TEST_RANGE[0], "refusing to build a range reaching the test period"
        X_train, y_train = get_labeled_by_time_range(node_table, feature_cols, fold["train_start"], fold["train_end"])
        val_time_steps = list(range(fold["val_start"], fold["val_end"] + 1))
        snapshots_by_t = {
            t: get_labeled_by_time_range(node_table, feature_cols, t, t) for t in val_time_steps
        }

        for seed in ADAPT_SEEDS:
            class_weights = compute_fixed_sample_weights(y_train)  # from TRAIN only, fixed for this seed/fold
            t0 = time.time()
            base_model = fit_base_model(X_train, y_train, seed, class_weights)
            base_train_time = time.time() - t0

            # --- Static HGB: frozen model, predict the whole val range at once ---
            static_model = copy.deepcopy(base_model)
            y_static_all, p_static_all = [], []
            for t in val_time_steps:
                X_t, y_t = snapshots_by_t[t]
                p = static_model.predict_proba(X_t)[:, 1]
                y_static_all.append(y_t)
                p_static_all.append(p)
            y_static = np.concatenate(y_static_all)
            p_static = np.concatenate(p_static_all)
            static_pr_auc = float(average_precision_score(y_static, p_static))
            static_threshold = select_threshold_maximizing_f1(y_static, p_static)
            static_f1 = compute_binary_metrics(y_static, p_static, static_threshold)["f1"]

            # --- Adaptive HGB: identical starting weights (deep copy), walk-forward ---
            adaptive_model = copy.deepcopy(base_model)
            t0 = time.time()
            walk = run_adaptive_tree_walk(adaptive_model, val_time_steps, snapshots_by_t, class_weights)
            walk_time = time.time() - t0
            y_adaptive, p_adaptive = pooled_predictions(walk)
            adaptive_pr_auc = float(average_precision_score(y_adaptive, p_adaptive))
            adaptive_threshold = select_threshold_maximizing_f1(y_adaptive, p_adaptive)
            adaptive_f1 = compute_binary_metrics(y_adaptive, p_adaptive, adaptive_threshold)["f1"]

            n_adapt_events = sum(1 for e in walk.events if e["type"] == "adapt")
            logger.info(
                "[adaptive_tree][%s][seed=%d] static_pr_auc=%.4f adaptive_pr_auc=%.4f (delta=%+.4f)",
                fold["name"], seed, static_pr_auc, adaptive_pr_auc, adaptive_pr_auc - static_pr_auc,
            )

            rows.append({
                "fold": fold["name"], "seed": seed, "condition": "static",
                "val_pr_auc": static_pr_auc, "val_f1": static_f1, "val_threshold": static_threshold,
                "n_val_labeled": int(len(y_static)), "train_time_seconds": round(base_train_time, 3),
            })
            rows.append({
                "fold": fold["name"], "seed": seed, "condition": "adaptive",
                "val_pr_auc": adaptive_pr_auc, "val_f1": adaptive_f1, "val_threshold": adaptive_threshold,
                "n_val_labeled": int(len(y_adaptive)), "n_adapt_events": n_adapt_events,
                "train_time_seconds": round(base_train_time, 3), "walk_time_seconds": round(walk_time, 3),
            })
    return rows
