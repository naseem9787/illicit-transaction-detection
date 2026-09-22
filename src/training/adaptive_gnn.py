"""Phase 4: simulated delayed-feedback online adaptation for the Phase 3 GCN.

IMPORTANT — this is a SIMULATED experimental protocol, not a property of
the Elliptic dataset. The dataset does not tell us when real-world labels
actually become available; "labels for time step t become available before
prediction at t + k" is an explicit, documented assumption (see
docs/LIMITATIONS.md L2), not a fact discovered in the data.

Walk-forward protocol for a fixed feedback delay k, over an ordered list
of time steps [t0, t0+1, ..., tN]:

    for t in [t0, t0+1, ..., tN]:
        1. If an earlier step's labels are due to be revealed at t
           (i.e. some t' with t' + k == t was predicted earlier), run the
           adaptation update using t' 's revealed labeled nodes now,
           BEFORE predicting t. This updates the model weights.
        2. Predict t using the (possibly just-updated) weights.
        3. Freeze/log prediction(t) immediately — this dict entry is never
           touched again.
        4. Schedule t's own labels to be revealed at t + k.

This ordering guarantees, by construction, that prediction(t) is always
computed using weights that reflect adaptation from steps strictly
earlier than t (specifically <= t - k), never from t itself or any later
step — see tests/test_adaptive_gnn.py for the automated leakage checks
that verify this directly against the executed event log rather than
just the code's intent.

No replay buffer is implemented in this experiment (see docs/EXPERIMENTS.md
Phase 4 "Replay" subsection for why) — each adaptation update trains on
exactly the one revealed snapshot's labeled nodes, nothing else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch_geometric.data import Data

from src.data.graph_builder import build_all_snapshots
from src.data.pyg_adapter import snapshot_to_data
from src.models.gcn import GCN

logger = logging.getLogger(__name__)


def build_per_step_data(
    node_table: pd.DataFrame,
    edges: pd.DataFrame,
    feature_cols: list[str],
    time_steps: list[int],
) -> dict[int, Data]:
    """Per-time-step (unbatched) Data objects, needed for the walk-forward
    loop since each step is predicted/adapted on individually rather than
    as one combined block-diagonal batch (contrast with
    src/training/gnn_training.py::build_split_batches, used by Phase 3)."""
    snapshots = build_all_snapshots(node_table, edges, feature_cols, time_steps=time_steps)
    return {t: snapshot_to_data(snap) for t, snap in snapshots.items()}


@dataclass
class AdaptationConfig:
    lr: float
    grad_steps: int
    feedback_delay_k: int
    replay: bool = False  # not implemented in this experiment; always False
    seed: int = 42


@dataclass
class WalkResult:
    config: AdaptationConfig
    predictions: dict  # t -> (y_true, y_prob, time_step) numpy arrays, frozen at log time
    predict_order: list  # time steps in the exact order they were predicted
    events: list  # chronological event log: {"type": "predict"|"adapt", ...}
    final_state_dict: dict = field(repr=False)


def _param_fingerprint(model: GCN) -> float:
    """Cheap scalar summary of current weights, for leakage-test assertions."""
    with torch.no_grad():
        return float(sum(p.sum().item() for p in model.parameters()))


def run_adaptive_walk(
    model: GCN,
    ordered_time_steps: list[int],
    snapshots_by_t: dict[int, Data],
    pos_weight: torch.Tensor,
    config: AdaptationConfig,
) -> WalkResult:
    """Execute the walk-forward protocol described in the module docstring.

    `model` is mutated in place (its weights evolve across the walk) —
    callers that need an unmodified starting point must pass a model
    freshly loaded from the pretrained checkpoint.
    """
    if config.replay:
        raise NotImplementedError(
            "Replay buffer is not implemented in this experiment (see "
            "docs/EXPERIMENTS.md Phase 4). AdaptationConfig.replay must be False."
        )

    torch.manual_seed(config.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    pending_reveals: dict[int, int] = {}  # due_at_t -> source_time_step
    predictions: dict[int, tuple] = {}
    predict_order: list[int] = []
    events: list[dict] = []

    for t in ordered_time_steps:
        # 1. Apply any adaptation update due at this step (source is strictly < t).
        if t in pending_reveals:
            source_t = pending_reveals.pop(t)
            source_data = snapshots_by_t[source_t]
            model.train()
            last_loss = None
            for _ in range(config.grad_steps):
                optimizer.zero_grad()
                logits = model(source_data.x, source_data.edge_index)
                mask = source_data.is_labeled
                loss = criterion(logits[mask], source_data.y[mask].float())
                loss.backward()
                optimizer.step()
                last_loss = float(loss.item())
            events.append({
                "type": "adapt",
                "applied_before_predicting": t,
                "source_time_step": source_t,
                "n_labeled_used": int(source_data.is_labeled.sum()),
                "final_loss": last_loss,
            })

        # 2. Predict t using current (possibly just-updated) weights.
        model.eval()
        param_hash = _param_fingerprint(model)
        with torch.no_grad():
            data_t = snapshots_by_t[t]
            logits = model(data_t.x, data_t.edge_index)
            prob = torch.sigmoid(logits)
            mask = data_t.is_labeled
            y_true = data_t.y[mask].numpy().copy()
            y_prob = prob[mask].numpy().copy()
            time_step_arr = data_t.time_step[mask].numpy().copy()

        # 3. Freeze/log prediction(t) immediately.
        predictions[t] = (y_true, y_prob, time_step_arr)
        predict_order.append(t)
        events.append({"type": "predict", "step": t, "param_hash": param_hash, "n_labeled": len(y_true)})

        # 4. Schedule t's own labels to be revealed k steps later.
        pending_reveals[t + config.feedback_delay_k] = t

    return WalkResult(
        config=config,
        predictions=predictions,
        predict_order=predict_order,
        events=events,
        final_state_dict={k: v.clone() for k, v in model.state_dict().items()},
    )


def pooled_predictions(result: WalkResult, time_steps: list[int] | None = None):
    """Concatenate per-step predictions into one (y_true, y_prob, time_step) tuple."""
    ts = time_steps if time_steps is not None else result.predict_order
    y_true = np.concatenate([result.predictions[t][0] for t in ts])
    y_prob = np.concatenate([result.predictions[t][1] for t in ts])
    time_step = np.concatenate([result.predictions[t][2] for t in ts])
    return y_true, y_prob, time_step


def select_adaptation_config(
    pretrained_state_dict: dict,
    in_channels: int,
    hidden_channels: int,
    dropout: float,
    val_time_steps: list[int],
    val_snapshots_by_t: dict[int, Data],
    pos_weight: torch.Tensor,
    candidate_configs: list[AdaptationConfig],
) -> tuple[AdaptationConfig, list[dict]]:
    """Small, pre-declared grid search over validation (30-34) only.

    For each candidate, a FRESH model is loaded from the pretrained
    checkpoint (so candidates never contaminate each other) and walked
    forward over the validation period with k=1. Selection criterion is
    pooled PR-AUC across all validation predictions — the same
    imbalance-appropriate, threshold-free metric already used for Random
    Forest config selection (docs/DECISIONS.md D9) and the Phase 3 GCN
    checkpoint selection. The winning config's adapted weights from this
    search are discarded; only the (lr, grad_steps) pair is kept. The
    actual test-period run restarts from the pretrained checkpoint fresh.
    """
    selection_log = []
    best_config, best_score = None, -1.0

    for cfg in candidate_configs:
        model = GCN(in_channels=in_channels, hidden_channels=hidden_channels, dropout=dropout)
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
