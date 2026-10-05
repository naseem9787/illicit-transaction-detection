# Illicit Transaction Detection Using Graph-Based Adaptive Learning

Elliptic Bitcoin dataset, illicit-transaction / AML-oriented detection.

**Status: Phases 1–8 complete.** This is an empirical investigation, not a
novel-algorithm paper: it studies whether graph-based learning and a
delayed-feedback online-adaptation protocol improve illicit transaction
detection on this dataset relative to a strong tabular baseline, under a
strictly chronological evaluation protocol. The adaptive mechanism is
**online fine-tuning on newly revealed labels** — not reinforcement
learning or meta-learning. See `docs/METHODOLOGY.md` for the full write-up
and `docs/LIMITATIONS.md` for what this project does and does not claim.

## Research question

> Can adaptive graph-based learning improve illicit transaction detection
> when transaction patterns change over time?

Tested via a chronological train/validation/test split (never shuffled),
a static-vs-adaptive comparison at matched architecture/weights, and a
5-seed robustness check — see `docs/EXPERIMENTS.md` for the full story.

## Key results (test set, time steps 35–49)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.202 | 0.811 | 0.324 | 0.857 | 0.210 |
| **Random Forest** | **0.926** | 0.689 | **0.790** | **0.936** | **0.787** |
| Basic GCN | 0.306 | 0.536 | 0.390 | 0.808 | 0.265 |
| Adaptive GCN | 0.389 | 0.529 | 0.448 | 0.829 | 0.310 |
| Static GraphSAGE | 0.440 | 0.608 | 0.510 | 0.875 | 0.430 |
| Adaptive GraphSAGE | 0.497 | 0.647 | 0.562 | 0.889 | 0.524 |
| Static GraphSAGE (5-seed mean) | — | — | 0.494 ± 0.042 | — | 0.462 ± 0.041 |
| Adaptive GraphSAGE (5-seed mean) | — | — | 0.566 ± 0.025 | — | 0.553 ± 0.022 |
| Static HGB (context, 5-seed mean) | 0.878 | 0.715 | 0.787 ± 0.021 | 0.939 | 0.795 ± 0.003 |
| Adaptive HGB (context, 5-seed mean) | 0.348 | 0.734 | 0.469 ± 0.060 | 0.803 | 0.205 ± 0.037 |

The final comparison and its protocol are in `docs/RESULTS.md` and
`docs/FINAL_MODEL_PROTOCOL.md`. Adaptive HGB — the same delayed label
feedback given to a boosted tree — made the tree **worse** (0/5 seeds
improved); this negative result is reported, not hidden.

**Random Forest remains the strongest model on test F1.** This project
does not claim otherwise, and does not treat that as a failure — see
"What this project does and does not claim" below and
`docs/EXPERIMENTS.md` for why. The finding of interest is that Adaptive
GraphSAGE improves consistently over its static counterpart (5/5 seeds, on
both F1 and PR-AUC), and that three independent attempts to add graph
information to the *stronger* tabular model (an MLP control, engineered
graph features, and graph-score smoothing — all under `docs/EXPERIMENTS.md`
Phase 8) did not help — a genuine, reported negative-result pattern, not a
gap in the write-up.

## What this project does and does not claim

- **Does claim:** a working, leakage-safe chronological pipeline; a
  reproducible online-adaptation protocol with a consistent, multi-seed
  gain over a matched static GNN; and an honest empirical picture of where
  graph information does and doesn't help on this dataset.
- **Does not claim:** a novel algorithm, state-of-the-art performance,
  beating Random Forest, real-time deployment readiness, a real (as
  opposed to simulated) feedback-delay assumption, or generalization
  beyond the Elliptic dataset. See `docs/LIMITATIONS.md`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place the raw dataset at:
```
data/raw/elliptic_bitcoin_dataset/elliptic_txs_classes.csv
data/raw/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv
data/raw/elliptic_bitcoin_dataset/elliptic_txs_features.csv
```

Run the full test suite (147 tests, all currently passing):

```bash
python3 -m pytest tests/ -v
```

## Dataset

203,769 transaction nodes, 234,355 directed edges, 49 time steps, 165
anonymized numeric features per node. Labels: 4,545 illicit, 42,019 licit,
157,205 unknown (excluded from loss/metrics throughout). Edges never cross
time steps — the graph is 49 structurally independent snapshots, not one
temporal graph. Full structural audit in `docs/DATA_AUDIT.md`.

## Chronological split (fixed throughout every phase)

```
Train:      time steps 1–29
Validation: time steps 30–34
Test:       time steps 35–49
```

Test is touched exactly once per model family, only after
validation-only model/hyperparameter/threshold selection is frozen.

## Project structure

```
config.yaml                      all paths/split/seed config

src/data/                        Phase 1: loading, validation, chronological split, graph construction
src/baselines/                   Phase 2: Logistic Regression, Random Forest
src/models/                      GCN, GraphSAGE, direction-aware GraphSAGE variants, MLP control
src/training/gnn_training.py     Phase 3 training/checkpointing (reused unchanged through Phase 7)
src/training/adaptive_gnn.py     Phase 4 delayed-feedback walk-forward protocol (reused through Phase 6/7)
src/training/adaptive_graphsage.py   Phase 6 model-factory wrapper around the Phase 4 protocol
src/training/rolling_origin*.py  Phase 8 rolling-origin validation (folds, tree grids, graph batches)
src/training/graph_smoothing.py  Phase 8 E7: graph-score smoothing of tree predictions
src/training/graph_features.py   Phase 8 E5: engineered graph features for trees
src/training/adaptive_tree.py    Adaptive-tree control (warm-start HGB under the delayed-feedback walk)
src/evaluation/                  metrics, validation-only threshold selection, per-time-step breakdown

scripts/audit_dataset.py             Phase 1 CLI entry point
scripts/train_baselines.py, evaluate_baselines.py            Phase 2
scripts/train_gnn.py, evaluate_gnn.py                        Phase 3 (basic GCN)
scripts/train_adaptive_gnn.py, evaluate_adaptive_gnn.py      Phase 4 (adaptive GCN)
scripts/train_gnn_improvement.py, evaluate_gnn_improvement.py Phase 5 (GraphSAGE selection)
scripts/train_adaptive_graphsage.py, evaluate_adaptive_graphsage.py  Phase 6
scripts/train_multiseed_robustness.py, evaluate_multiseed_robustness.py  Phase 7
scripts/run_phase8_diagnostics.py       Phase 8 E1 (MLP) + E2 (tree benchmark)
scripts/run_phase8_e3_diagnostics.py    Phase 8 E3 (direction-aware GraphSAGE)
scripts/run_phase8_e5_diagnostics.py    Phase 8 E5 (tree + graph features)
scripts/run_phase8_e7_diagnostics.py    Phase 8 E7 (graph-score smoothing)
scripts/run_adaptive_tree_control.py    Adaptive-tree control (validation only)
scripts/run_final_evaluation.py         Final consolidated test evaluation (run once)

tests/                           147 tests across all phases, run against real data/artifacts
```

## Documentation

- `docs/DATA_AUDIT.md` — full structural audit of the dataset, including
  why parts of the original adaptive-mechanism brief aren't computable
  from this data (no persistent per-entity history is possible).
- `docs/DECISIONS.md` — every non-obvious design choice and why (D1–D9).
- `docs/METHODOLOGY.md` — the complete methodology: models, adaptive
  protocol, threshold selection, validation design, and what "adaptive"
  does and doesn't mean here.
- `docs/EXPERIMENTS.md` — the full chronological experimental narrative,
  Phase 2 through Phase 8, with every reported number traceable to a
  saved result file.
- `docs/LIMITATIONS.md` — dataset-level and methodological caveats,
  including the simulated feedback delay and what is not yet controlled
  for (see L5, L6).
- `docs/FINAL_MODEL_PROTOCOL.md` — the frozen finalists, thresholds and
  evaluation rules, written before the final test evaluation.
- `docs/RESULTS.md` — the final results table and what it does and does
  not show.
- `docs/PAPER_OUTLINE.md`, `docs/PRESENTATION.md`, `docs/VIVA_QA.md` —
  paper outline, 10-slide structure with 5/10-minute scripts, and viva
  questions and answers.

## Reproducing results

Each phase's training/evaluation scripts write to `results/metrics/` and
`results/figures/` without overwriting earlier phases' outputs — every
phase's artifacts are still present and independently reproducible. See
`docs/EXPERIMENTS.md` for the exact command for each phase.
