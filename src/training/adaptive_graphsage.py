"""Phase 6: model-factory wrapper for adaptive GraphSAGE.

Phase 4's src/training/adaptive_gnn.py::select_adaptation_config hard-codes
`GCN(...)`, so it cannot be reused for GraphSAGE without editing Phase 4
code. This module re-implements only that selection loop around a
caller-supplied model factory; the walk-forward protocol itself
(run_adaptive_walk) is reused UNCHANGED from Phase 4, so the delayed-
feedback semantics are exactly the same.

Selection is restricted to feedback_delay_k == 1 candidates: the k=3
sensitivity run must never be able to influence which configuration is
chosen, so a non-k=1 candidate is rejected outright.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
from sklearn.metrics import average_precision_score
from torch_geometric.data import Data

from src.training.adaptive_gnn import AdaptationConfig, pooled_predictions, run_adaptive_walk

logger = logging.getLogger(__name__)


def select_adaptation_config_with_factory(
    model_factory: Callable[[], torch.nn.Module],
    pretrained_state_dict: dict,
    val_time_steps: list[int],
    val_snapshots_by_t: dict[int, Data],
    pos_weight: torch.Tensor,
    candidate_configs: list[AdaptationConfig],
) -> tuple[AdaptationConfig, list[dict]]:
    """Pre-declared grid search on validation only, k=1, pooled PR-AUC.

    A FRESH model is built and loaded from the pretrained checkpoint for
    every candidate. Adapted weights from the search are discarded; only the
    (lr, grad_steps) pair is kept.
    """
    if any(c.feedback_delay_k != 1 for c in candidate_configs):
        raise ValueError("Adaptation selection must use k=1 candidates only; k=3 is sensitivity-only.")

    selection_log = []
    best_config, best_score = None, -1.0
    for cfg in candidate_configs:
        model = model_factory()
        model.load_state_dict(pretrained_state_dict)
        result = run_adaptive_walk(model, val_time_steps, val_snapshots_by_t, pos_weight, cfg)
        y_true, y_prob, _ = pooled_predictions(result)
        pr_auc = float(average_precision_score(y_true, y_prob))
        selection_log.append({
            "lr": cfg.lr,
            "grad_steps": cfg.grad_steps,
            "feedback_delay_k": cfg.feedback_delay_k,
            "val_pooled_pr_auc": pr_auc,
        })
        logger.info("candidate lr=%.4f grad_steps=%d -> val pooled PR-AUC=%.4f", cfg.lr, cfg.grad_steps, pr_auc)
        if pr_auc > best_score:
            best_score, best_config = pr_auc, cfg
    logger.info("Selected lr=%.4f grad_steps=%d (val pooled PR-AUC=%.4f)", best_config.lr, best_config.grad_steps, best_score)
    return best_config, selection_log
