"""Phase 8 (E5): label-free, per-snapshot engineered graph features for the
tree benchmark.

Predeclared feature set (fixed before any validation result was seen — not
added to or pruned afterward):

    in_degree, out_degree, total_degree, two_hop_reach,
    in_neighbor_mean_f1..f165, out_neighbor_mean_f1..f165

`compute_graph_features(x, edge_index)` takes only a node feature matrix and
an edge_index for ONE snapshot — never `y`/labels, never any other time
step's data. Everything is computed from that single snapshot's own
structure, mirroring the same per-snapshot isolation already used in E7
(src/training/graph_smoothing.py) and enforced structurally by
src/data/graph_builder.py (a snapshot's edge_index only ever contains
node-local indices for that one time_step).

Neighbor "statistics" here are the mean of neighbors' own f1..f165 feature
vectors, separately for incoming and outgoing neighbors — the direct tabular
analogue of a single mean-aggregation GraphSAGE layer (E3's direction-aware
design), but as engineered columns for a tree rather than a learned
embedding. A node with zero neighbors in a given direction gets that
direction's mean vector filled with 0.0 (the features are already globally
z-scored — see docs/DATA_AUDIT.md — so 0 is the natural "no information"
value, not an arbitrary choice) and 0 for the corresponding degree.

Caveat worth stating plainly (see docs/DATA_AUDIT.md section 8): the
released f94-f165 are documented (by external, unverified-against-the-raw-
file convention) as already being one-hop "aggregated" neighbor features.
If that convention holds, part of what in/out_neighbor_mean recomputes here
may already be present in the original 165 columns — this is an honest
reason E5 might show a small or null effect, not evidence of a bug.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.data.graph_builder import build_all_snapshots
from src.evaluation.metrics import compute_binary_metrics, select_threshold_maximizing_f1
from src.training.rolling_origin import E2_SEEDS, ROLLING_ORIGIN_FOLDS, TEST_RANGE

logger = logging.getLogger(__name__)

# Exact E2-selected HistGradientBoosting configuration — frozen, identical
# for both the baseline (A) and graph-augmented (B) conditions, so the only
# thing that differs between them is the feature set.
HGB_BASE_CONFIG = {"max_depth": None, "learning_rate": 0.1, "max_iter": 100, "min_samples_leaf": 50}
E5_SEEDS = E2_SEEDS  # same 5 seeds as E2

# Fixed at 4 scalar columns + 2 * len(feature_cols) neighbor-mean columns.
GRAPH_FEATURE_SCALAR_NAMES = ["in_degree", "out_degree", "total_degree", "two_hop_reach"]


def graph_feature_names(feature_cols: list[str]) -> list[str]:
    return (
        GRAPH_FEATURE_SCALAR_NAMES
        + [f"in_neighbor_mean_{c}" for c in feature_cols]
        + [f"out_neighbor_mean_{c}" for c in feature_cols]
    )


def compute_graph_features(x: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """(N, 4 + 2*D) label-free graph feature matrix for one snapshot.

    Column order: [in_degree, out_degree, total_degree, two_hop_reach,
    in_neighbor_mean (D cols), out_neighbor_mean (D cols)].
    """
    n, d = x.shape
    in_neighbors: list[set[int]] = [set() for _ in range(n)]
    out_neighbors: list[set[int]] = [set() for _ in range(n)]
    if edge_index.shape[1] > 0:
        src, dst = edge_index[0].tolist(), edge_index[1].tolist()
        for s, t in zip(src, dst):
            out_neighbors[s].add(t)
            in_neighbors[t].add(s)
    sym_neighbors = [in_neighbors[i] | out_neighbors[i] for i in range(n)]

    in_degree = np.array([len(s) for s in in_neighbors], dtype=np.float32)
    out_degree = np.array([len(s) for s in out_neighbors], dtype=np.float32)
    total_degree = in_degree + out_degree

    two_hop_reach = np.zeros(n, dtype=np.float32)
    for v in range(n):
        reach = set(sym_neighbors[v])
        for u in sym_neighbors[v]:
            reach |= sym_neighbors[u]
        reach.discard(v)
        two_hop_reach[v] = len(reach)

    in_mean = np.zeros((n, d), dtype=np.float32)
    out_mean = np.zeros((n, d), dtype=np.float32)
    for v in range(n):
        if in_neighbors[v]:
            in_mean[v] = x[list(in_neighbors[v])].mean(axis=0)
        if out_neighbors[v]:
            out_mean[v] = x[list(out_neighbors[v])].mean(axis=0)

    scalars = np.stack([in_degree, out_degree, total_degree, two_hop_reach], axis=1)
    return np.concatenate([scalars, in_mean, out_mean], axis=1).astype(np.float32)


def build_features_for_range(
    node_table: pd.DataFrame, edges: pd.DataFrame, feature_cols: list[str], start: int, end: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X_original_165, X_augmented_165_plus_graph, y) for labeled nodes with
    time_step in [start, end]. Graph features are computed per snapshot
    (over ALL nodes, including unlabeled ones — only features flow through,
    never labels) then filtered to labeled rows, same masking convention as
    src/training/rolling_origin.py::get_labeled_by_time_range."""
    time_steps = list(range(start, end + 1))
    snapshots = build_all_snapshots(node_table, edges, feature_cols, time_steps=time_steps)

    X_orig_parts, X_graph_parts, y_parts = [], [], []
    for t in time_steps:
        snap = snapshots[t]
        graph_feats = compute_graph_features(snap.x, snap.edge_index)
        mask = snap.is_labeled
        X_orig_parts.append(snap.x[mask])
        X_graph_parts.append(graph_feats[mask])
        y_parts.append(snap.y[mask])

    X_orig = np.concatenate(X_orig_parts, axis=0)
    X_graph = np.concatenate(X_graph_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    X_augmented = np.concatenate([X_orig, X_graph], axis=1)
    return X_orig, X_augmented, y


def _fit_eval(X_train, y_train, X_val, y_val, seed: int) -> dict:
    model = HistGradientBoostingClassifier(
        max_depth=HGB_BASE_CONFIG["max_depth"], learning_rate=HGB_BASE_CONFIG["learning_rate"],
        max_iter=HGB_BASE_CONFIG["max_iter"], min_samples_leaf=HGB_BASE_CONFIG["min_samples_leaf"],
        class_weight="balanced", random_state=seed,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    val_prob = model.predict_proba(X_val)[:, 1]
    threshold = select_threshold_maximizing_f1(y_val, val_prob)  # validation only
    m = compute_binary_metrics(y_val, val_prob, threshold)
    return {"val_pr_auc": m["pr_auc"], "val_f1": m["f1"], "val_threshold": threshold,
            "train_time_seconds": round(train_time, 2), "n_features": X_train.shape[1]}


def run_e5(node_table: pd.DataFrame, edges: pd.DataFrame, feature_cols: list[str]) -> list[dict]:
    """Condition A (baseline_165) vs Condition B (graph_augmented), same
    HGB config, same folds, same seeds — the only variable is the feature
    set. Test period is never built or touched."""
    rows = []
    for fold in ROLLING_ORIGIN_FOLDS:
        assert fold["val_end"] < TEST_RANGE[0], "refusing to build a range reaching the test period"
        X_orig_train, X_aug_train, y_train = build_features_for_range(
            node_table, edges, feature_cols, fold["train_start"], fold["train_end"])
        X_orig_val, X_aug_val, y_val = build_features_for_range(
            node_table, edges, feature_cols, fold["val_start"], fold["val_end"])

        for seed in E5_SEEDS:
            r_baseline = _fit_eval(X_orig_train, y_train, X_orig_val, y_val, seed)
            rows.append({"fold": fold["name"], "seed": seed, "condition": "baseline_165", **r_baseline})

            r_augmented = _fit_eval(X_aug_train, y_train, X_aug_val, y_val, seed)
            rows.append({"fold": fold["name"], "seed": seed, "condition": "graph_augmented", **r_augmented})

            logger.info("[E5][%s][seed=%d] baseline_pr_auc=%.4f graph_augmented_pr_auc=%.4f (delta=%+.4f)",
                        fold["name"], seed, r_baseline["val_pr_auc"], r_augmented["val_pr_auc"],
                        r_augmented["val_pr_auc"] - r_baseline["val_pr_auc"])
    return rows
