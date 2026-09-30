"""Phase 8 (E7): graph-smoothing of tree (HistGradientBoosting) scores.

Tests whether graph structure improves the strongest tabular model (E2's
HGB) before building a heavier hybrid (E5). The tree itself is never
modified — this only post-processes its predicted probabilities using the
graph, so nothing about src/baselines/*.py or E1/E2 changes.

    smoothed_score = (1 - alpha) * tree_score + alpha * neighbor_score

`neighbor_score` is the mean tree_score of a node's neighbors under the
SYMMETRIZED (both incoming and outgoing) edge set for that node's own
snapshot — same direction-aware convention established in E3's
src/models/graphsage_directional.py::symmetrize, reimplemented here for
plain numpy arrays (E7 has no PyTorch dependency: `Snapshot.edge_index` is
a numpy array, not a torch.Tensor — see src/data/graph_builder.py). Only
tree *scores* (model output) ever flow between nodes — never labels. A
node's own tree_score is
computed purely from its own f1..f165 features; using another node's
already-computed score as a smoothing input is not the same as using that
node's label, and no `y` array is ever touched inside the smoothing
functions (see tests/test_phase8_e7_graph_smoothing.py).

`depth` applies the same 1-hop propagation step `depth` times (standard
iterative graph-smoothing / Correct-and-Smooth-style formulation), so
depth=2 lets information reach 2-hop neighbors with a second decaying
alpha-weighted step, rather than requiring a separately-defined 2-hop
neighbor SET. Isolated nodes (zero neighbors in their own snapshot) have no
well-defined neighbor_score, so the update degenerates to a no-op for
them at every iteration — they retain their original tree_score exactly,
regardless of alpha or depth.

Because edges never cross time steps (established since Phase 1;
re-verified per-snapshot here), and symmetrizing/reversing an edge cannot
change which snapshot its endpoints belong to (established in E3), smoothing
computed independently per snapshot cannot mix information across time
steps or use anything not present in that node's own validation snapshot.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from src.data.graph_builder import build_all_snapshots
from src.evaluation.metrics import compute_binary_metrics, select_threshold_maximizing_f1
from src.training.rolling_origin import E2_SEEDS, ROLLING_ORIGIN_FOLDS, TEST_RANGE, get_labeled_by_time_range

logger = logging.getLogger(__name__)

# Exact E2-selected HistGradientBoosting configuration — frozen, not re-tuned here.
HGB_BASE_CONFIG = {"max_depth": None, "learning_rate": 0.1, "max_iter": 100, "min_samples_leaf": 50}

ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DEPTHS = [1, 2]
E7_SEEDS = E2_SEEDS  # same 5 seeds as E2


def symmetrize_numpy(edge_index: np.ndarray) -> np.ndarray:
    """Numpy analogue of src/models/graphsage_directional.py::symmetrize:
    union of a directed edge_index and its reverse. The dataset has zero
    mutual pairs (docs/DATA_AUDIT.md), so this never double-counts an edge
    that was already bidirectional."""
    if edge_index.shape[1] == 0:
        return edge_index
    return np.concatenate([edge_index, edge_index[::-1]], axis=1)


def smooth_scores(scores: np.ndarray, edge_index_sym: np.ndarray, alpha: float, depth: int) -> np.ndarray:
    """Apply `depth` iterations of (1-alpha)*score + alpha*mean(neighbor
    scores) over a symmetrized edge_index. A node with zero neighbors keeps
    its current score unchanged at every iteration (falls back to its own
    score rather than 0), so isolated nodes are provably unaffected by
    alpha or depth. Never touches labels — operates purely on `scores`."""
    n = len(scores)
    current = scores.astype(np.float64).copy()
    if edge_index_sym.shape[1] == 0:
        return current  # no edges at all: every node is isolated
    src, dst = edge_index_sym[0], edge_index_sym[1]
    for _ in range(depth):
        sums = np.zeros(n)
        counts = np.zeros(n)
        np.add.at(sums, dst, current[src])
        np.add.at(counts, dst, 1)
        has_neighbors = counts > 0
        neighbor_score = np.where(has_neighbors, sums / np.maximum(counts, 1), current)
        current = (1 - alpha) * current + alpha * neighbor_score
    return current


def _predict_and_smooth_snapshot(model, snap, alpha: float, depth: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, smoothed_score) for the LABELED nodes of one snapshot.
    Tree scores are computed for every node (labeled and unlabeled — only
    features are used, never labels), so unlabeled nodes can still supply a
    legitimate neighbor signal, mirroring how unlabeled nodes participate in
    GNN message passing elsewhere in this project without ever contributing
    a label."""
    tree_score = model.predict_proba(snap.x)[:, 1]
    sym_edge_index = symmetrize_numpy(snap.edge_index)
    # cross-time-edge guard: every edge in a single snapshot's edge_index is
    # already scoped to that snapshot by construction (graph_builder.py); this
    # assert documents that invariant rather than silently trusting it.
    assert snap.edge_index.shape[1] == 0 or (sym_edge_index < len(snap.node_ids)).all()
    smoothed = smooth_scores(tree_score, sym_edge_index, alpha, depth)
    mask = snap.is_labeled
    return snap.y[mask], smoothed[mask]


def run_e7(node_table: pd.DataFrame, edges: pd.DataFrame, feature_cols: list[str]) -> list[dict]:
    rows = []
    for fold in ROLLING_ORIGIN_FOLDS:
        val_time_steps = list(range(fold["val_start"], fold["val_end"] + 1))
        assert fold["val_end"] < TEST_RANGE[0], "refusing to build a validation range reaching the test period"
        val_snapshots = build_all_snapshots(node_table, edges, feature_cols, time_steps=val_time_steps)

        X_train, y_train = get_labeled_by_time_range(node_table, feature_cols, fold["train_start"], fold["train_end"])

        for seed in E7_SEEDS:
            t0 = time.time()
            model = HistGradientBoostingClassifier(
                max_depth=HGB_BASE_CONFIG["max_depth"], learning_rate=HGB_BASE_CONFIG["learning_rate"],
                max_iter=HGB_BASE_CONFIG["max_iter"], min_samples_leaf=HGB_BASE_CONFIG["min_samples_leaf"],
                class_weight="balanced", random_state=seed,
            )
            model.fit(X_train, y_train)
            train_time = time.time() - t0

            # Baseline (alpha=0 is the algebraic identity, but compute the
            # unsmoothed tree metric directly too as an independent check).
            y_all, score_all = [], []
            for t in val_time_steps:
                snap = val_snapshots[t]
                tree_score = model.predict_proba(snap.x)[:, 1]
                mask = snap.is_labeled
                y_all.append(snap.y[mask])
                score_all.append(tree_score[mask])
            y_val = np.concatenate(y_all)
            baseline_score = np.concatenate(score_all)
            baseline_pr_auc = float(average_precision_score(y_val, baseline_score))
            baseline_threshold = select_threshold_maximizing_f1(y_val, baseline_score)
            baseline_f1 = compute_binary_metrics(y_val, baseline_score, baseline_threshold)["f1"]
            rows.append({
                "fold": fold["name"], "seed": seed, "alpha": 0.0, "depth": 0, "variant": "tree_only_baseline",
                "n_val_labeled": int(len(y_val)), "val_pr_auc": baseline_pr_auc, "val_f1": baseline_f1,
                "val_threshold": baseline_threshold, "train_time_seconds": round(train_time, 2),
            })

            for depth in DEPTHS:
                for alpha in ALPHAS:
                    y_all, score_all = [], []
                    for t in val_time_steps:
                        snap = val_snapshots[t]
                        y_lab, smoothed_lab = _predict_and_smooth_snapshot(model, snap, alpha, depth)
                        y_all.append(y_lab)
                        score_all.append(smoothed_lab)
                    y_val2 = np.concatenate(y_all)
                    smoothed_score = np.concatenate(score_all)
                    pr_auc = float(average_precision_score(y_val2, smoothed_score))
                    threshold = select_threshold_maximizing_f1(y_val2, smoothed_score)  # validation only
                    f1 = compute_binary_metrics(y_val2, smoothed_score, threshold)["f1"]
                    rows.append({
                        "fold": fold["name"], "seed": seed, "alpha": alpha, "depth": depth, "variant": "smoothed",
                        "n_val_labeled": int(len(y_val2)), "val_pr_auc": pr_auc, "val_f1": f1,
                        "val_threshold": threshold, "train_time_seconds": round(train_time, 2),
                    })
            logger.info("[E7][%s][seed=%d] baseline_pr_auc=%.4f (train %.2fs)", fold["name"], seed, baseline_pr_auc, train_time)
    return rows
