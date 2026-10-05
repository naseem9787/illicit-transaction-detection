# METHODOLOGY.md

This is the complete methodology for the project. It describes what was
built and why, at the level a reader needs to evaluate whether the results
in `docs/EXPERIMENTS.md` are trustworthy. Exact numbers live in
`docs/EXPERIMENTS.md` and the underlying `results/metrics/*.json` files;
this document is about *how* those numbers were produced, not what they
say.

**Framing, stated once and held throughout:** this project is an empirical
investigation of graph-based and adaptive learning on one dataset, not a
new algorithm. The adaptive mechanism used everywhere below is **online
fine-tuning on newly revealed labels** — ordinary gradient-based continual
learning — not reinforcement learning, not meta-learning, and not the
mechanism of any specific prior system used as motivation for the project.
Where that distinction matters, it is called out explicitly.

---

## 1. Dataset

The original Elliptic Bitcoin transaction dataset: 203,769 transaction
nodes, 234,355 directed edges, 49 time steps, 165 anonymized numeric
features per node. Every feature column is already globally z-scored by
the dataset's authors (verified directly: every column's mean ≈ 0, std ≈ 1
— `docs/DATA_AUDIT.md` §4), so no additional feature scaling is applied
anywhere in this project.

Two structural facts drive most of the design decisions below, both
verified directly against the raw files rather than assumed:

- **Every transaction ID is unique.** No transaction recurs across time
  steps, so there is no persistent per-entity identity to track "history"
  for at the individual-transaction level (`docs/DATA_AUDIT.md` §5.2,
  `docs/DECISIONS.md` D4). This is why the adaptive mechanism operates on
  model weights, not on per-transaction memory.
- **No edge crosses a time step boundary.** 100% of the 234,355 edges
  connect two nodes with the same `time_step` value. The graph is
  therefore 49 structurally independent snapshot graphs, not one temporal
  graph (`docs/DATA_AUDIT.md` §5.2, `docs/DECISIONS.md` D3). Every model
  and every batching scheme in this project is built to respect this —
  verified directly by tests, not just assumed (e.g.
  `tests/test_gnn.py::test_no_edges_cross_snapshot_boundaries_in_real_batches`).

## 2. Label handling

Raw labels are the string values `'1'` (illicit), `'2'` (licit), and
`'unknown'`. These are re-encoded once, in `src/data/preprocessing.py`, to
`{illicit: 1, licit: 0, unknown: -1}` (`docs/DECISIONS.md` D1), so that
"is this row usable for supervised loss" is a single check (`label >= 0`)
everywhere downstream instead of a string comparison repeated in every
module. Unknown-labeled nodes (157,205 of 203,769) are **excluded from
every loss computation and every reported metric**, throughout every
phase — but they are **not** excluded from graph message passing: a GNN's
neighbors within a snapshot may legitimately be unlabeled, and dropping
them would throw away real structural information available to their
labeled neighbors. This distinction (participate in structure, never
contribute a label or a gradient) is enforced via an `is_labeled` mask
that is separate from graph construction, and is checked directly by
tests in every phase that touches it (e.g.
`tests/test_gnn.py::test_unknown_labels_excluded_from_is_labeled_but_present_in_graph`).

## 3. Chronological split

```
Train:      time steps 1–29
Validation: time steps 30–34
Test:       time steps 35–49
```

Fixed once in `config.yaml` (`docs/DECISIONS.md` D2) and unchanged across
every one of the eight phases. The test boundary (35–49) matches the
convention used in the paper that released this dataset and essentially
all follow-up work, so results stay comparable to literature-reported
numbers on the same test region. A model is evaluated on test **exactly
once** per model family, only after every selection decision (checkpoint,
hyperparameters, threshold) has already been frozen using validation data
only. This rule is enforced structurally in the Phase 8 diagnostic code
(`TEST_RANGE` constants with runtime assertions refusing to build any data
range that reaches t=35) and checked directly by tests throughout.

Illicit rate is highly non-stationary across time steps — from under 0.5%
to over 35% (`docs/DATA_AUDIT.md` §6) — which is the direct motivation for
evaluating per-time-step, not just on a single pooled number, and for the
adaptive-learning research question in the first place.

## 4. Graph construction

Per-time-step snapshots are built once (`src/data/graph_builder.py`,
Phase 1) as plain numpy structures with field names matching
`torch_geometric.data.Data` exactly (`x`, `edge_index`, `y`), so wrapping
them for GNN use is a direct field-for-field conversion
(`src/data/pyg_adapter.py`) with no PyTorch dependency in the Phase 1
pipeline itself. Multiple snapshots are combined into one training batch
via `torch_geometric.data.Batch.from_data_list`, which produces a
block-diagonal graph — edges are only ever offset within their own
snapshot, never created between snapshots. This equivalence (batching
several snapshots together is mathematically identical to processing them
one at a time) is verified directly, not just asserted
(`tests/test_gnn.py::test_batch_is_block_diagonal`).

## 5. Baseline models (Phase 2)

Logistic Regression and Random Forest, both on the 165 features alone (no
graph, no `txId`, `time_step` excluded from the primary comparison — a
secondary experiment confirmed `time_step` acts as a shortcut that doesn't
transfer to test: validation F1 0.757 vs. test F1 0.323, see
`docs/EXPERIMENTS.md` Phase 2). Both use `class_weight="balanced"`,
computed from the training split's label distribution only. Random
Forest's 3-candidate depth sweep, selected by validation PR-AUC, is a
deliberately shallow search (`docs/DECISIONS.md` D9) — this is a college
research project, not an exhaustive tuning exercise, and that scope
decision is held consistently through every later phase's hyperparameter
grids.

## 6. Basic GCN (Phase 3)

A plain 2-layer GCN (`GCNConv(165,64) → ReLU → Dropout(0.5) → GCNConv(64,1)`),
node features and `edge_index` only. `BCEWithLogitsLoss` with a fixed
`pos_weight = n_negative/n_positive` computed from training labels only —
the same imbalance-handling intent as `class_weight="balanced"`, not
numerically identical to it. Checkpoint selected by best validation
PR-AUC over a fixed 200-epoch budget; decision threshold selected
separately, by maximizing F1 on validation predictions only, then frozen
before test.

## 7. Delayed-feedback adaptive protocol (Phase 4, reused through Phase 7)

**This is the core mechanism of the project, so it is described precisely.**
Given a sequence of time steps, and starting from an already-trained
(frozen-weights) checkpoint:

```
for t in [t0, t0+1, ..., tN]:
    1. If an earlier step's labels are due to be revealed now (per a
       fixed delay k), run a small number of gradient steps on that
       step's revealed labeled nodes — this updates the model's weights.
    2. Predict t using the (possibly just-updated) weights.
    3. Freeze and log prediction(t) immediately — never revisited.
    4. Schedule t's own labels to be revealed k steps later.
```

This ordering guarantees, by construction, that prediction(t) never uses
labels from t itself or any later step — verified two ways: (1) directly
from the executed event log of a real run, not just from reading the code
(the actual production run's log was audited step-by-step in this
project's own history), and (2) as an internal correctness check: since no
adaptation has occurred before the very first prediction, the static and
adaptive models' first-step predictions are *bit-identical*, confirmed
directly against saved prediction files.

**What is adapting:** the model's own weights, via `torch.optim.Adam` and
a handful of gradient steps per revealed snapshot — no replay buffer
(explicitly not implemented, a documented scope decision — see
`docs/EXPERIMENTS.md` Phase 4 "Replay"), no reinforcement signal, no
meta-learned update rule. This is continual/online fine-tuning, a standard
and well-understood technique — the contribution is applying it under a
carefully leakage-audited chronological protocol and measuring its effect
honestly, not inventing a new adaptation algorithm.

**The feedback delay is a simulated experimental assumption, not a
dataset fact** (`docs/LIMITATIONS.md` L2). The Elliptic dataset contains
no timestamped label-arrival information at all — there is no way to know
from the data how or when a transaction's label was actually confirmed
relative to its own time step. "Labels for t become available before
prediction at t+1" (k=1, primary) is adopted to make a well-defined,
testable experiment possible, and every result derived from it is
conditional on that assumption. A k=3 sensitivity variant is run
alongside k=1 wherever the primary experiment is, always reported
separately and never used to select between k values based on outcome.

## 8. GraphSAGE and direction-aware aggregation (Phase 5, Phase 8 E3)

**Phase 5** replaced the basic GCN with a validated, stronger static
model. A small, pre-declared grid (2 architectures — a tuned GCN variant
and GraphSAGE — × 2 hidden sizes × 2 dropout rates × 2 learning rates = 16
configurations) was trained under the *same* protocol as Phase 3 (same
`train_gcn` function, unmodified), and the single overall winner was
selected by validation PR-AUC alone, before any test contact. The winner
was **GraphSAGE, hidden=128, dropout=0.5, lr=0.01** (validation PR-AUC
0.864), which then became the base model for every later phase.
`src/training/gnn_training.py::train_gcn` — despite its name — is
architecture-agnostic (it only ever calls `model(x, edge_index)`), so it
was reused unmodified for GraphSAGE without any change to Phase 3's code.

**Phase 8 (E3)** investigated whether the *direction* of message passing
matters. The raw `edge_index` used by GCN and GraphSAGE aggregates only
from a node's incoming neighbors (PyG's default `src → dst` flow); a
node's outgoing neighbors never contribute a message under that raw
orientation. Two direction-aware variants were tested under rolling-origin
validation (§10) against a plain-GraphSAGE control trained identically:
a **symmetrized-edge** variant (both convolution layers see the union of
edges and their reverse) and a **direction-aware** variant (separate
convolution branches for incoming and outgoing neighbors, concatenated).
Both beat the plain-GraphSAGE control in every one of 3 rolling-origin
folds; direction-aware was the strongest (+0.042 mean validation PR-AUC
over plain GraphSAGE). This result is validation-only — it has not been
evaluated on the held-out test period as of this document.

## 9. Threshold selection

Held identical across every phase: `select_threshold_maximizing_f1`
(`src/evaluation/metrics.py`) sweeps a precision-recall curve computed on
**validation predictions only** and returns the F1-maximizing threshold.
That threshold is then frozen and applied unchanged to test predictions —
never re-selected, never touched by test labels. Where a Static/Adaptive
comparison is being made (Phases 4, 6), the *same* threshold (the one
already selected for the static model) is deliberately reused for the
adaptive model rather than re-selecting a second one, so the comparison
isolates online weight adaptation as the only variable rather than being
confounded by two independently chosen thresholds.

## 10. Rolling-origin validation (Phase 8)

The fixed train/val/test split gives exactly one validation window (30–34,
5 time steps) for every prior phase's hyperparameter decisions — adequate,
but a single window limits how much confidence any one selection decision
can carry. Phase 8 introduced three chronological folds, each strictly
earlier than the next and none touching the test period, specifically for
model/hyperparameter comparisons that warranted more validation signal
than one window could give:

```
fold 1: train 1–19,  validate 20–24
fold 2: train 1–24,  validate 25–29
fold 3: train 1–29,  validate 30–34   (identical to the original fixed split)
```

fold3 deliberately reproduces the original split exactly, so Phase 8
results can be cross-checked against earlier phases' numbers on the same
train/val boundary. Every fold-based experiment (E1 MLP, E2 tree
benchmark, E3 direction-aware GraphSAGE, E5 graph features, E7 graph
smoothing) reuses this same fold definition from a single source of
truth (`src/training/rolling_origin.py::ROLLING_ORIGIN_FOLDS`), with a
runtime assertion refusing any range that reaches the test period.

## 11. Multi-seed robustness (Phase 7)

The Phase 6 static-vs-adaptive GraphSAGE comparison was repeated at 5
predeclared seeds (42, 123, 456, 789, 2024), with the *same* adaptation
configuration for every seed (no per-seed retuning — the configuration
selected once in Phase 6 was held fixed). Seed 42 uses the existing
Phase 5/6 checkpoint as the official reference rather than retraining it,
because that checkpoint's initial weights came from RNG state left by the
previous entry in the Phase 5 grid search and cannot be exactly
regenerated from a bare `seed=42` invocation — a real reproducibility gap,
reported rather than hidden (`docs/EXPERIMENTS.md` Phase 7, "Seed-42
reproducibility issue"). A from-scratch seed-42 replicate was run
separately as a continuity check and never substituted for the official
reference.

## 12. Phase 8 diagnostics: why the GNNs trail Random Forest

Phase 8 exists to explain, not just report, the gap between the GNN
family (best test F1 0.562) and Random Forest (test F1 0.790). Four
experiments, all validation-only:

- **E1 (MLP control):** a plain feature-only MLP scored far below the
  tuned trees on the same rolling-origin folds (e.g. fold3 val PR-AUC
  0.893 vs. the trees' ~0.99), separating "neural networks underperform
  trees on this tabular data" from "the graph specifically isn't helping"
  — both are true, and this experiment isolates the first from the second.
- **E2 (equal-budget tree benchmark):** Random Forest and
  HistGradientBoosting, each tuned over a small predeclared 16-config
  grid under the same rolling-origin protocol, to make sure the GNN
  comparison is against a fairly-tuned tabular model, not an
  under-tuned strawman.
- **E5 (tree + engineered graph features):** label-free, per-snapshot
  graph features (in/out/total degree, 2-hop reach, directional
  neighbor-feature means) appended to the strongest tree's input. Result:
  **no measurable improvement** (−0.0022 mean validation PR-AUC, within
  seed noise, negative in 4 of 5 seeds).
- **E7 (graph-score smoothing):** blending the strongest tree's
  predicted probabilities with its neighbors' predicted probabilities
  (never labels), at several blend weights and propagation depths.
  Result: **no improvement** — the best configuration found was no
  smoothing at all (α=0), with monotonically worse validation PR-AUC as
  the blend weight increased.

**These are reported as genuine negative results, not omitted.** The
consistent finding across E1, E5, and E7 is that naive, post-hoc or
feature-level incorporation of graph information does not improve an
already-strong tabular model on this dataset — a plausible explanation,
not independently confirmed, is that the released features (f94–f165, by
external documentation this project cannot verify against the raw file)
already encode one-hop aggregated neighbor information, leaving little
room for a simple graph augmentation to add.

## 13. What "adaptive" does and does not mean here

To state this as plainly as possible, since it is the most
misunderstanding-prone part of the project:

- **Is:** ordinary gradient-based fine-tuning of an already-trained
  model's weights, triggered by newly revealed ground-truth labels,
  under a simulated feedback-delay assumption, with no replay buffer.
- **Is not:** reinforcement learning, meta-learning, a learned adaptation
  policy, topology adaptation, or any mechanism specific to a particular
  prior system used as early motivation for this project. This project
  deliberately did not implement RL or meta-learning — not because they
  were tried and failed, but because the simplest mechanism that could
  still test the research question was preferred over a more complex one
  that would add uncontrolled variables to an already-careful leakage
  audit.
- **Adaptive-tree control (partial):** the same walk-forward protocol was
  applied to the strongest tree (HistGradientBoosting) using sklearn's
  `warm_start` to add 10 boosting rounds per revealed snapshot, with class
  weights fixed from the training split. On rolling-origin validation this
  did not help the tree (−0.0095 mean PR-AUC). Because that control ran on
  validation folds where the tree is near-saturated, it does not fully
  isolate the test-period drift regime in which the GNN's gain appeared;
  the like-for-like comparison is deferred to the single final test
  evaluation. See `docs/EXPERIMENTS.md` and `docs/LIMITATIONS.md` L5.

## 14. Reproducibility

Every result in `docs/EXPERIMENTS.md` is traceable to a specific file
under `results/metrics/` or `results/figures/`, produced by a specific
script under `scripts/`, using a specific seed or seed set stated in that
phase's section. 147 tests (`tests/`) cover leakage-safety (chronological
ordering, no future-label access, no cross-snapshot edges), correctness
(shapes, determinism, threshold behavior), and — where a specific
methodological claim was made in this project's own history (e.g. "batching
is equivalent to processing snapshots independently") — a direct test of
that claim rather than only a code comment asserting it.
