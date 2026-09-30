# EXPERIMENTS.md

## Phase 2 — Classical ML Baselines

### Purpose

Establish a trustworthy, leakage-safe performance floor before any GNN work,
using only per-transaction feature vectors (no graph structure). This tells
us how much of the detection problem a non-relational model can already
solve, so later phases can attribute any GNN improvement specifically to
graph structure (and, eventually, to the adaptive mechanism) rather than to
generic modeling capacity.

### Models

- **Baseline A — Logistic Regression** (`src/baselines/logistic_regression.py`).
  `class_weight="balanced"`, `max_iter=2000`, `lbfgs` solver, seed=42.
- **Baseline B — Random Forest** (`src/baselines/random_forest.py`).
  `class_weight="balanced"`, 3 candidate configs (`max_depth` in
  {8, 16, None}, `n_estimators=300` fixed), selected by validation PR-AUC.
  Not an exhaustive grid search — chosen deliberately shallow per the
  project brief ("do not over-tune").

### Input features

`f1..f165` — the 165 anonymized, already-standardized feature columns from
`elliptic_txs_features.csv` (see `docs/DATA_AUDIT.md`). No `txId`, no graph
structure, no raw labels, no future information.

`time_step` is **excluded from the primary baselines** by design: it's a
temporal index, and a classifier could use it as a shortcut (memorize
"time steps with value near X tend to be illicit") rather than learning
genuine transaction risk patterns. A secondary Logistic Regression
including `time_step` was trained to check this concern directly — see
Results below; it confirms the concern was warranted.

### Temporal split (from Phase 1, unchanged)

train = time steps 1–29, val = 30–34, test = 35–49. See `docs/DATA_AUDIT.md`
section 9 and `docs/DECISIONS.md` D2. No shuffling; test set never touched
for any fitting or threshold decision.

### Preprocessing

No additional feature scaling — the released features are already globally
z-scored by the dataset authors (verified in Phase 1: every column has
mean≈0, std≈1). Scaling again would be redundant and adds a step that
could subtly differ between train/val/test fits for no benefit. `class_weight
="balanced"` for both models is fit using the training split's label
distribution only, so it introduces no leakage.

Unknown-labeled rows (`label == -1`) are excluded from all training and
evaluation splits by construction (`src/data/dataset.py::get_labeled_split`)
— they are never treated as licit.

### Threshold selection

For each primary model, the decision threshold is the one that maximizes
F1 on the **validation set only** (`select_threshold_maximizing_f1` in
`src/evaluation/metrics.py`), then frozen and applied unchanged to the test
set. Test data is never used to pick a threshold.

| model | threshold (picked on val) |
|---|---|
| Logistic Regression | 0.956 |
| Random Forest | 0.5333 |

### Metrics

Precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix (TN/FP/FN/TP),
false-positive rate, false-negative rate — computed on val (for threshold
selection context) and test (final, frozen-threshold numbers). Accuracy is
reported but not treated as a primary metric, given the ~6.5% illicit rate
in the test period.

### Results — TEST set (time steps 35–49), frozen threshold

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.202 | 0.813 | 0.323 | 0.856 | 0.209 |
| Random Forest | 0.926 | 0.689 | 0.790 | 0.936 | 0.787 |

Random Forest is the stronger baseline by a wide margin on every metric
except recall, where Logistic Regression is slightly higher (0.813 vs
0.689) but only by predicting far more false positives (3,464 vs 60 out
of 15,587 licit test transactions — see confusion matrices in
`results/metrics/baseline_results.json`).

Random Forest test confusion matrix (threshold 0.5333): TN=15527, FP=60,
FN=337, TP=746.

### Secondary experiment: Logistic Regression WITH time_step

| split | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| validation (30–34) | 0.695 | 0.831 | 0.757 | 0.689 |
| test (35–49) | 0.201 | 0.814 | 0.323 | 0.211 |

Validation F1 (0.757) looks competitive with the no-time_step model, but
test F1 collapses to 0.323 — almost identical to the no-time_step
Logistic Regression's test score, i.e. **adding time_step gave no real
test-time benefit**. The concern that `time_step` invites shortcut
learning that doesn't transfer to a new time period is directly supported
by this result, which is why it's kept out of the primary comparison.

### Temporal breakdown (test period, 35–49)

Full per-time-step numbers in `results/metrics/temporal_logreg_no_tstep.csv`
and `results/metrics/temporal_random_forest_no_tstep.csv`, and
`results/figures/f1_over_time_{logreg,random_forest}.png`.

Both models perform reasonably through time step 42 (Random Forest F1
mostly 0.75–0.96 in that range), then **both collapse sharply from time
step 43 onward** — Random Forest's F1 drops to 0.0 at steps 43, 45, 47, 48
and stays low elsewhere in that range; Logistic Regression's F1 also drops
(to ~0.02–0.09 at those same steps) though it never fully collapses to
zero because its very low, val-tuned threshold keeps recall high at the
cost of precision throughout.

This lines up exactly with the illicit-rate figure from Phase 1
(`illicit_rate_over_time.png`): time steps 43–49 have a much lower and
much less variable illicit rate (mostly <3%, versus 10–15% for 35–42) —
a genuine regime shift, not noise or a bug. Both baselines were fit on
time steps 1–29, which look statistically more like 35–42 than 43–49.

---

## Phase 3 — Basic GNN (2-layer GCN)

### Purpose

Answer the Phase 3 research question directly: **does graph structure add
predictive value over the feature-only Random Forest baseline?** This is a
basic, deliberately untuned GNN — not the adaptive mechanism (Phase 4),
which is out of scope here.

### How snapshots are processed (methodology requirement)

Per `docs/DECISIONS.md` D3, the dataset is 49 structurally independent
graphs, one per time_step, with zero cross-time edges. The GCN is trained
and evaluated on that same structure, not on one merged temporal graph:

- `src/data/graph_builder.py::build_all_snapshots` (Phase 1, unchanged)
  builds the 49 per-time-step snapshots.
- `src/data/pyg_adapter.py` (new) wraps each `Snapshot` as a
  `torch_geometric.data.Data` object with matching field names (`x`,
  `edge_index`, `y`), plus `is_labeled` and `time_step` carried along per
  node.
- `src/training/gnn_training.py::build_split_batches` combines a split's
  snapshots (e.g. all of time steps 1–29 for train) into one
  `torch_geometric.data.Batch` via `Batch.from_data_list`. This is a plain
  **block-diagonal** stack — `edge_index` values are offset per graph and
  no edge is ever created between two snapshots. Batching is therefore
  mathematically identical to running each snapshot through the model one
  at a time; it's used only for training/inference efficiency. Verified
  directly in `tests/test_gnn.py::test_batch_is_block_diagonal` and
  `test_no_edges_cross_snapshot_boundaries_in_real_batches`.
- Raw edges are re-read directly from `elliptic_txs_edgelist.csv` (Phase 1
  never caches edges to `data/processed/` — see `docs/DATA_AUDIT.md`), so
  no change was needed to any Phase 1 output file.

Unknown-labeled nodes (`label == -1`) **do participate in message
passing** — they are real graph neighbors and dropping them would throw
away real structural information for their labeled neighbors — but they
never contribute to the loss or to any reported metric. This is enforced
by masking on `is_labeled` at the loss (`src/training/gnn_training.py::
train_gcn`) and at prediction time (`predict_labeled`), and is checked
directly in `tests/test_gnn.py::
test_unknown_labels_excluded_from_is_labeled_but_present_in_graph`.

No future information reaches training: train/val/test are the same
chronological ranges as Phase 1/2 (1–29 / 30–34 / 35–49), the model is
never shown val or test snapshots during backpropagation, and the decision
threshold is selected on validation predictions only (see below) —
identical discipline to Phase 2.

### Model

`src/models/gcn.py::GCN` — a plain 2-layer GCN, node features + edges
only, no edge features, no residual connections:

```
GCNConv(165, 64) -> ReLU -> Dropout(0.5) -> GCNConv(64, 1)
```

Output is a single raw logit per node; `sigmoid(logit)` is the illicit
probability, matching the same `y_prob` convention `src/evaluation/
metrics.py` already expects from the sklearn baselines, so no evaluation
code had to change. Deliberately fixed, not tuned — same "don't
over-tune" scope decision as Random Forest's 3-config search
(`docs/DECISIONS.md` D9).

### Class imbalance handling

`BCEWithLogitsLoss(pos_weight=n_negative/n_positive)`, computed from the
**training split's labeled nodes only** (`src/training/gnn_training.py::
compute_pos_weight`). This is the standard PyTorch idiom for binary
imbalance and is analogous in intent to `class_weight="balanced"` used
for Logistic Regression / Random Forest, but not numerically identical to
sklearn's formula — documented as an analogy, not an equivalence.

### Training configuration

| setting | value |
|---|---|
| optimizer | Adam |
| learning rate | 0.01 |
| weight decay | 5e-4 |
| epochs | 200 (fixed) |
| checkpoint selection | best validation PR-AUC across all 200 epochs (never test) |
| seed | 42 (project-wide seed, `config.yaml`) |
| batching | full-batch per split per epoch (block-diagonal, see above) |
| PyTorch | 2.14.0+cpu |
| PyTorch Geometric | 2.8.0.post1 |
| device | CPU (dataset is small enough — max snapshot ≈7,880 nodes — that GPU wasn't needed; kept the install CPU-only to avoid an unnecessary CUDA download) |
| train time | 63.5s (`results/metrics/gcn_train_manifest.json`, hardware-dependent) |
| best epoch | 171 / 200 |
| best validation PR-AUC | 0.7707 |

Preprocessing: none beyond Phase 1's existing global standardization — no
additional feature scaling, same as Phase 2 (`f1..f165` are already
z-scored by the dataset authors; see `docs/DATA_AUDIT.md` §4 and
`docs/LIMITATIONS.md` L1 for the caveat on that standardization's
provenance).

### Threshold selection

Same procedure as Phase 2: `select_threshold_maximizing_f1` run on
validation predictions only, then frozen for test
(`scripts/evaluate_gnn.py::evaluate_gcn`). Selected threshold: **0.863**.

### Results — TEST set (time steps 35–49), frozen threshold

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.202 | 0.813 | 0.323 | 0.856 | 0.209 |
| Random Forest | 0.926 | 0.689 | 0.790 | 0.936 | 0.787 |
| **GCN (basic)** | 0.306 | 0.536 | 0.390 | 0.808 | 0.265 |

GCN test confusion matrix (threshold 0.863): TN=14272, FP=1315, FN=502,
TP=581.

Validation metrics (for context — not the comparison number, since
val/test are known to diverge sharply on this dataset for every model
tried so far): precision 0.724, recall 0.809, F1 0.764, ROC-AUC 0.943,
PR-AUC 0.771.

### Answering the Phase 3 research question

**Graph structure alone, in this basic un-tuned form, does not beat the
feature-only Random Forest baseline.** The GCN beats Logistic Regression
on F1 (0.390 vs 0.323) and is roughly comparable on ROC-AUC (0.808 vs
0.856), but is well below Random Forest on every metric (F1 0.390 vs
0.790, PR-AUC 0.265 vs 0.787). This is a real, negative-leaning result for
"does graph structure help" as posed — not a bug: the same val→test
generalization gap and the same time-step regime shift documented for the
Phase 2 baselines (see below) also affects the GCN, and a single fixed,
untuned architecture/hyperparameter set is not expected to be
competitive with an already-selected Random Forest.

This result should not be read as "graph structure is useless for this
problem" — the published literature on Elliptic (including the dataset's
own release paper) also finds plain GCN weaker than tree-based/ensemble
methods on this exact dataset, with better GNN results typically requiring
either more feature engineering, deeper/tuned architectures, or exactly
the kind of temporal modeling Phase 4 will explore. It does establish a
concrete, honest floor for what "graph, minimally applied" achieves here,
which is what Phase 3 was scoped to determine.

### Temporal breakdown (test period, 35–49)

Full per-time-step numbers in `results/metrics/temporal_gcn.csv` and
`results/figures/gcn_f1_over_time.png`.

The GCN shows the same qualitative pattern as both Phase 2 baselines:
moderate F1 (0.14–0.70) through time step 42, then a collapse from time
step 43 onward (F1 exactly 0.0 at steps 44, 45, 48, 49; near-zero at 43,
46, 47). This matches the same regime shift already documented for
Logistic Regression and Random Forest — time steps 43–49 have a much
lower, much less variable illicit rate than the 1–29 training period —
and is further evidence that this collapse is a property of the dataset's
temporal drift, not specific to any one model family. This is exactly the
kind of temporal instability that motivates Phase 4, without this
document making any claim about what Phase 4's design should be.

### Figures

`results/figures/gcn_vs_baselines_comparison.png` (3-way bar chart),
`results/figures/gcn_f1_over_time.png`,
`results/figures/pr_roc_curves_all_models.png` (PR and ROC curves for all
three models overlaid), `results/figures/gcn_confusion_matrix.png`.

### Files

`src/models/gcn.py`, `src/data/pyg_adapter.py`,
`src/training/gnn_training.py`, `scripts/train_gnn.py`,
`scripts/evaluate_gnn.py`, `tests/test_gnn.py`. Outputs:
`results/models/gcn.pt`, `results/metrics/predictions/gcn_{train,val,test}.npz`,
`results/metrics/gcn_train_manifest.json`, `results/metrics/gcn_results.json`,
`results/metrics/temporal_gcn.csv`. No Phase 1 or Phase 2 file was
modified — Phase 2's `baseline_results.json` and prediction files are
only read, never written, by the Phase 3 scripts.

---

## Phase 4 — Adaptive GCN (simulated delayed-feedback online adaptation)

### Purpose and scope

Tests whether **online weight adaptation** helps the existing Phase 3 GCN
respond to the temporal regime shift the dataset already shows (the t=43
collapse documented above). This experiment implements exactly one
adaptive mechanism — Candidate C from the Phase 4 design review (online
fine-tuning on revealed labels with a fixed feedback delay) — and nothing
else. Per the design brief, this run deliberately excludes reinforcement
learning, meta-learning, drift-statistics features, graph rewiring,
EvolveGCN-style weight evolution, and any other adaptive mechanism; those
remain candidates for future work, not part of this experiment.

**The primary scientific comparison is Static GCN vs Adaptive GCN**, both
on time steps 35–49. Logistic Regression and Random Forest appear in the
summary figure only as secondary reference points.

### The feedback-delay assumption is simulated, not a dataset fact

This is the central methodological clarification for Phase 4, and it is
restated here deliberately: **the Elliptic dataset does not establish
that real-world labels become available one time step later.** The
one-step (or three-step) delay used below is an explicit, documented
experimental assumption:

> After predicting time step t, labels for t are assumed to become
> available before prediction at t+1 (k=1), or before prediction at t+3
> for the k=3 sensitivity run.

This is a simulation of a plausible investigative-lag scenario, not a
property discovered in or guaranteed by the data. See `docs/LIMITATIONS.md`
L2 for the full caveat — it applies to every result in this section.

### Protocol

Both models start from **exactly the same pretrained weights**
(`results/models/gcn.pt`, unmodified Phase 3 checkpoint) — the only
experimental variable is whether weights evolve during the 35–49 walk.

**Static GCN**: pretrained on 1–29, frozen, predicts 35–49 (this is
literally Phase 3's `gcn_test.npz`, reused read-only, not recomputed).

**Adaptive GCN**, for `t = 35 … 49`, strictly in order:

```
prediction(t) → freeze/log prediction(t) → reveal labels(t)
              → adaptation update (t's revealed labels only)
              → prediction(t+1)
```

Implemented in `src/training/adaptive_gnn.py::run_adaptive_walk`. At each
iteration, any earlier step's labels that are due to be revealed now (per
the fixed delay k) are used for a small number of gradient steps
*before* that iteration's own prediction is made; the prediction is then
logged immediately and never revisited. This ordering guarantees, by
construction, that no prediction ever uses labels from its own or any
later time step — verified directly by the leakage tests in
`tests/test_adaptive_gnn.py`, not just asserted by the code's intent.

As an internal correctness check: since no adaptation has occurred before
the very first prediction (t=35), Static and Adaptive GCN produce
*bit-identical* predictions for t=35 — confirmed directly
(`np.allclose(static_probs[t=35], adaptive_probs[t=35])` → `True`). This
is strong evidence the protocol is wired correctly, not a coincidence.

### Class imbalance handling in the adaptation loop

The `BCEWithLogitsLoss` pos_weight used during adaptation is **fixed** at
the value computed from the original training split (1–29 labeled nodes,
pos_weight ≈ 8.19) — the same value used to pretrain the Static GCN in
Phase 3. It is *not* recomputed per test time step. A single test
snapshot's revealed labels can be extremely small and skewed (e.g. t=45
has only a handful of illicit nodes among ~1,221 labeled), so a
per-step-recomputed pos_weight would make individual adaptation updates
numerically unstable; using the stable, train-derived value avoids that
without touching any test-period information beyond what's already
revealed.

### Threshold

**Reused from Phase 3, not re-selected.** The Static GCN's validation-only
threshold (0.863, selected once on 30–34 in `scripts/evaluate_gnn.py`) is
applied to both Static and Adaptive GCN predictions throughout the walk.
This isolates online weight adaptation as the only variable between the
two models — a second, independently-chosen threshold would have
confounded the comparison. No threshold is ever selected using test data,
for either model (`tests/test_adaptive_gnn.py::
test_evaluate_adaptive_predictions_never_reselects_threshold` asserts
this directly by monkeypatching `select_threshold_maximizing_f1` to raise
if called).

### Hyperparameter selection (validation 30–34 only)

A small, pre-declared grid — chosen before running anything, given only 5
validation time steps to select on:

| lr | grad_steps | k | val pooled PR-AUC |
|---:|---:|---:|---:|
| 0.001 | 1 | 1 | 0.7601 |
| 0.001 | 3 | 1 | 0.7449 |
| 0.005 | 1 | 1 | 0.7190 |
| 0.005 | 3 | 1 | 0.7055 |

Selected: **lr=0.001, grad_steps=1** (highest pooled PR-AUC over
validation 30–34). For each candidate, a *fresh* copy of the pretrained
model is walked forward over validation with k=1
(`src/training/adaptive_gnn.py::select_adaptation_config`) — candidates
never contaminate each other, and the adapted weights produced during
this search are discarded; only the winning (lr, grad_steps) pair
survives into the actual test-period run, which restarts from the
unmodified pretrained checkpoint.

Worth stating plainly: **every candidate's validation pooled PR-AUC
(0.706–0.760) is below the Static GCN's own validation PR-AUC (0.7707,
from Phase 3)**. Adaptation did not help on the validation walk itself —
this is reported honestly because it directly informed how the test-set
result below should be read (see "Interpretation").

**Statistical power caveat**: this selection is based on only **5
validation time steps** (30–34). With that few points and four candidates
this close together (0.7601 vs 0.7449 vs 0.7190 vs 0.7055), the selection
carries real variance — a different validation slice could plausibly have
picked a different winner. This is a genuine limitation of the available
validation period, not a flaw in the selection procedure itself (which is
otherwise leakage-safe — see the Phase 4 audit), and should be kept in
mind when weighing how much to read into the specific (lr, grad_steps)
pair chosen.

**Replay**: not implemented in this experiment. Each adaptation update
trains on exactly the one revealed snapshot's labeled nodes — no replay
buffer from 1–29 is mixed in. This was a deliberate scope decision (the
design brief explicitly permits deferring replay if it adds significant
complexity): correctly batching a fixed set of replay snapshots alongside
the revealed test snapshot, without merging their graphs via message
passing, is nontrivial to get right, and skipping it keeps the first
adaptive experiment's one variable (does weight adaptation itself help)
uncontaminated by a second one (does replay change the answer). This is a
documented limitation, not an oversight — a natural next experiment.

**Optimizer state note**: the Adam optimizer is instantiated once per walk
(`src/training/adaptive_gnn.py::run_adaptive_walk`) and its momentum/
variance state persists and compounds across all sequential adaptation
events within that walk — it is never reset between events. This means
`grad_steps=1` is **not** equivalent to an independent, fixed-size update
applied identically at every event; the effective step taken at, say,
t=49's adaptation reflects Adam's accumulated history from all 13 prior
adaptation events in the same walk, not just that one event's gradient in
isolation. This is not a leakage concern (the accumulated state is purely
a function of past, causally-legitimate gradients — see the Phase 4 audit),
but it is a real interpretive caveat on what "1 gradient step" means here.

Full configuration:

| setting | value |
|---|---|
| learning rate | 0.001 |
| gradient steps per update | 1 |
| feedback delay k (primary) | 1 |
| feedback delay k (sensitivity) | 3 |
| replay | not used |
| pos_weight | 8.1888 (fixed, from train 1–29) |
| seed | 42 |
| base checkpoint | `results/models/gcn.pt` (Phase 3, unmodified) |

### Results — TEST set (time steps 35–49), shared frozen threshold

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Static GCN | 0.306 | 0.536 | 0.390 | 0.808 | 0.265 |
| **Adaptive GCN (k=1)** | 0.389 | 0.529 | **0.448** | 0.829 | 0.310 |

Static GCN confusion matrix: TN=14272, FP=1315, FN=502, TP=581.
Adaptive GCN confusion matrix: TN=14687, FP=900, FN=510, TP=573.

Secondary reference (unchanged from Phase 2/3, not the comparison this
experiment is about): Logistic Regression F1=0.323, Random Forest
F1=0.790.

### Interpretation — where the improvement comes from, and where it doesn't

The Adaptive GCN does beat the Static GCN in the pooled test-set numbers
above (F1 +0.058, PR-AUC +0.045, ROC-AUC +0.021), and this improvement is
real, not an artifact — but the per-time-step breakdown
(`results/metrics/temporal_static_vs_adaptive_f1_delta.csv`) shows it is
**not evenly distributed**, and specifically does **not** answer "does
adaptation help the model recover from the t=43 regime shift" the way one
might hope:

| period | mean F1 delta (Adaptive − Static) |
|---|---:|
| t=35–42 (before the collapse) | **+0.058** |
| t=43–49 (collapse region) | **−0.001** (essentially zero) |

- **t=35**: F1 delta is exactly 0 — Static and Adaptive are bit-identical
  here by construction (no adaptation has happened yet). This is the
  internal correctness check mentioned above, not a result.
- **t=36–42**: consistent, meaningful positive deltas (+0.01 to +0.13 F1),
  peaking at t=38 (+0.126). The adaptive model genuinely tracks the
  still-"normal" regime better than the frozen static model here.
- **t=43–45**: both models collapse to near-zero F1 together
  (delta ≈ 0) — adaptation does not rescue the model from the regime
  shift; it fails the same way the static model does.
- **t=46–49**: mixed, small in magnitude (+0.038, **−0.049**, 0, 0) — no
  consistent direction, most plausibly noise given how few labeled
  illicit examples exist in this region (as low as ~0.3–2.6% illicit
  rate per step here).

**Honest conclusion**: online weight adaptation, in this minimal form,
improves overall test performance, but the improvement is concentrated in
the pre-collapse period, not in the exact regime-shift region (t≥43) that
motivated the Phase 4 research question in the first place. This is a
genuine, useful finding — it does **not** demonstrate that this adaptive
mechanism solves the problem it was aimed at, and this report does not
claim that it does. It's the honest floor for what one-step, no-replay
online fine-tuning achieves here.

### Sensitivity check: k=3 (secondary, not used for any selection)

Same selected hyperparameters (lr=0.001, grad_steps=1), only the feedback
delay changed to k=3, run once and reported as-is — **not** compared
against k=1 to pick a "better" delay, per the design brief:

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Adaptive GCN (k=3) | 0.365 | 0.534 | 0.434 | 0.823 | 0.294 |

k=3 sits between Static and k=1 on every metric. Consistent with k=1's
finding (adaptation helps, moderately, somewhere between "not at all" and
"a lot"), not a contradiction, but reported strictly as a secondary data
point.

### Leakage verification

Six automated tests in `tests/test_adaptive_gnn.py`, run against
synthetic sequential snapshots for exact, hand-checkable expected
behavior:

1. `test_prediction_precedes_its_own_adaptation` — prediction(t) is
   logged strictly before any adaptation event that uses t's own labels.
2. `test_adaptation_never_uses_future_labels` — every adaptation event's
   source time step equals `applied_at − k` (< applied_at) for k ∈ {1,2,3}.
3. `test_evaluate_adaptive_predictions_never_reselects_threshold` —
   `select_threshold_maximizing_f1` is asserted (via monkeypatch) to
   never be called during test-period evaluation.
4. `test_predictions_are_deterministic_and_not_mutated_afterward` — two
   independent runs with the same seed produce byte-identical logged
   predictions; each time step's dict entry is assigned exactly once.
5. `test_adaptation_uses_only_the_single_revealed_snapshot` — spies on
   the loss function's input shape during every adaptation step and
   confirms it equals exactly the one revealed snapshot's labeled node
   count (nothing else, e.g. a would-be replay buffer, is mixed in).
6. `test_chronological_order_preserved` — predictions are produced in
   exactly the given chronological order, no skips or reordering.

Plus `test_replay_true_is_rejected` (replay=True raises `NotImplementedError`
rather than silently doing nothing) and two config-selection sanity tests.

### Figures

`results/figures/adaptive_vs_static_comparison.png` (Static vs Adaptive
primary comparison, RF/LogReg as secondary reference bars),
`per_time_step_f1_static_vs_adaptive.png`, `f1_delta_over_time.png`
(highlights the t=43 boundary), `pr_roc_curves_adaptive_vs_static.png`,
`adaptive_vs_static_confusion_matrices.png`.

### Files

`src/training/adaptive_gnn.py`, `scripts/train_adaptive_gnn.py`,
`scripts/evaluate_adaptive_gnn.py`, `tests/test_adaptive_gnn.py`.
Outputs: `results/metrics/adaptive_hparam_selection.json`,
`results/metrics/adaptive_train_manifest.json`,
`results/metrics/adaptive_results.json`,
`results/metrics/predictions/adaptive_gcn_test{,_k3}.npz`,
`results/metrics/temporal_adaptive_gcn.csv`,
`results/metrics/temporal_static_vs_adaptive_f1_delta.csv`. No Phase
1/2/3 file was modified — `gcn_test.npz`, `gcn_results.json`, and
`baseline_results.json` are only read, never written, by the Phase 4
scripts.

---

## Phase 5 — Controlled static GNN improvement study

### Purpose

Determine whether the basic GCN's weak performance (test F1 0.390) was
primarily an architecture/training-setup limitation, by controlled
comparison against a small, pre-declared grid of alternatives — not an
open-ended architecture search.

### Protocol

Existing data protocol held fixed: same chronological split, same
unknown-label exclusion, same seed (42), same checkpoint-selection
criterion (validation PR-AUC) and threshold-selection criterion
(validation F1, frozen before test). Test was not inspected for model or
hyperparameter selection.

Two model families, each swept over a small grid — 2 hidden sizes (64,
128) × 2 dropout rates (0.2, 0.5) × 2 learning rates (0.003, 0.01) = 8
configurations per family, 16 total:

- **Tuned GCN**: the same `GCNConv → ReLU → Dropout → GCNConv` architecture
  as Phase 3, with hidden size/dropout/lr swept.
- **GraphSAGE**: `SAGEConv → ReLU → Dropout → SAGEConv`, same grid.

All 16 configurations were trained with `src/training/gnn_training.py::
train_gcn` (Phase 3's function, unmodified — it is architecture-agnostic
despite its name) at a fixed 200-epoch budget, matching the Phase 3
reference exactly so the comparison isn't confounded by a shorter
training budget for the new configurations. The single overall winner
across all 16 was selected by validation PR-AUC alone; test data was
never touched for any of the 15 non-winning configurations — not even for
reporting — so there was no way to select a "best test configuration."

### Validation results (all 16 configurations)

| Architecture | Hidden | Dropout | LR | Best Epoch | Val PR-AUC | Val F1 |
|---|---:|---:|---:|---:|---:|---:|
| tuned_gcn | 64 | 0.2 | 0.003 | 194 | 0.7476 | 0.7403 |
| tuned_gcn | 64 | 0.2 | 0.01 | 150 | 0.7826 | 0.7870 |
| tuned_gcn | 64 | 0.5 | 0.003 | 200 | 0.6971 | 0.7456 |
| tuned_gcn | 64 | 0.5 | 0.01 | 194 | 0.7760 | 0.7944 |
| tuned_gcn | 128 | 0.2 | 0.003 | 200 | 0.7402 | 0.7478 |
| tuned_gcn | 128 | 0.2 | 0.01 | 191 | 0.8000 | 0.7557 |
| tuned_gcn | 128 | 0.5 | 0.003 | 199 | 0.7319 | 0.7594 |
| tuned_gcn | 128 | 0.5 | 0.01 | 188 | 0.7937 | 0.7884 |
| graphsage | 64 | 0.2 | 0.003 | 83 | 0.8614 | 0.8449 |
| graphsage | 64 | 0.2 | 0.01 | 64 | 0.8611 | 0.8464 |
| graphsage | 64 | 0.5 | 0.003 | 87 | 0.8122 | 0.8177 |
| graphsage | 64 | 0.5 | 0.01 | 47 | 0.7979 | 0.8161 |
| graphsage | 128 | 0.2 | 0.003 | 77 | 0.8438 | 0.8272 |
| graphsage | 128 | 0.2 | 0.01 | 40 | 0.8401 | 0.8303 |
| graphsage | 128 | 0.5 | 0.003 | 68 | 0.8554 | 0.8496 |
| **graphsage** | **128** | **0.5** | **0.01** | **126** | **0.8640** | **0.8718** |

Every GraphSAGE config outperformed every tuned-GCN config on validation
PR-AUC — evidence that the convolution operator, not just hyperparameters,
was the more consequential factor. **Selected: GraphSAGE, hidden=128,
dropout=0.5, lr=0.01** (validation PR-AUC 0.8640, threshold 0.8980,
selected on validation F1). This checkpoint became the base model for
Phases 6, 7, and Phase 8's E3/direction-aware work.

### Test results (35–49), frozen threshold 0.898

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.202 | 0.813 | 0.323 | 0.856 | 0.209 |
| Random Forest | 0.926 | 0.689 | 0.790 | 0.936 | 0.787 |
| Basic GCN (Phase 3) | 0.306 | 0.536 | 0.390 | 0.808 | 0.265 |
| **Selected GraphSAGE** | 0.440 | 0.608 | **0.510** | 0.875 | 0.430 |

Confusion matrix: TN=14748, FP=839, FN=425, TP=658.

**Answer to the Phase 5 question:** the architecture/training-setup change
closes part of the gap to Random Forest (F1 0.390→0.510, and all
threshold-independent metrics improved too, so this isn't a threshold
artifact) but does not close it — Random Forest remains well ahead on
every metric. The basic GCN's weakness was **partly, not primarily**, an
architecture/training-setup problem.

### Temporal consistency vs. Basic GCN

11 of 15 test time steps improved, 2 got worse, 2 unchanged. Mean F1
delta: +0.103 for t=35–42, +0.006 for t=43–49 — same pattern seen in every
later phase: gains concentrate in the earlier, calmer test period.

### Files

`src/models/graphsage.py`, `scripts/train_gnn_improvement.py`,
`scripts/evaluate_gnn_improvement.py`, `tests/test_gnn_improvement.py`.
Outputs: `results/models/gnn_improvement_selected.pt`,
`results/metrics/gnn_improvement_{train_manifest,results}.json`,
`results/metrics/predictions/gnn_improvement_{val,test}.npz`,
`results/metrics/temporal_gnn_improvement*.csv`. No Phase 1–4 file was
modified.

---

## Phase 6 — Adaptive GraphSAGE (delayed-feedback online adaptation on the Phase 5 static model)

### Research question

Does delayed-feedback online adaptation still improve performance when it is
applied to the stronger static GraphSAGE model selected in Phase 5, rather
than to the basic GCN used in Phase 4? The comparison is Static GraphSAGE
vs Adaptive GraphSAGE. Random Forest is a secondary reference only.

### Static GraphSAGE configuration (frozen Phase 5 model, unmodified)

`SAGEConv(165, 128) -> ReLU -> Dropout(0.5) -> SAGEConv(128, 1)`, Adam
lr=0.01, weight_decay=5e-4, seed 42, checkpoint
`results/models/gnn_improvement_selected.pt` (selected in Phase 5 by
validation PR-AUC only). The classification threshold is the Phase 5
validation-F1 threshold, **0.898**, and it is frozen for both the static and
adaptive models. It is never re-selected. Static test predictions are the
existing `gnn_improvement_test.npz`, read-only.

### Adaptive GraphSAGE protocol

Identical to Phase 4; only the base model differs. Both models start from
exactly the same checkpoint. For `t = 35 ... 49`, strictly in order:
`prediction(t) -> freeze/log prediction(t) -> reveal labels(t) -> adaptation
update -> prediction(t+1)`. The walk is `run_adaptive_walk` from
`src/training/adaptive_gnn.py`, reused unchanged. The loss is
`BCEWithLogitsLoss` with a fixed train-derived `pos_weight` (8.1888,
train 1-29). There is no replay, no drift features, no RL/meta-learning and
no EvolveGCN. Each update trains only on the one revealed snapshot's
labeled nodes.

**The feedback delay is a simulated experimental assumption**, not a
property of the Elliptic dataset: after predicting time step t, labels for t
are assumed to become available before prediction at t+1 (k=1, primary) or
t+3 (k=3, sensitivity). See `docs/LIMITATIONS.md` L2.

### Validation-only adaptation hyperparameter selection

Phase 4's `select_adaptation_config` hard-codes `GCN`, so Phase 6 adds a
model-factory wrapper (`src/training/adaptive_graphsage.py`) rather than
editing Phase 4 code. The wrapper rejects any non-k=1 candidate, so the k=3
run cannot influence selection. The pre-declared grid was walked forward
over validation steps 30-34 (k=1), starting from a fresh copy of the
checkpoint for each candidate. Selection is by pooled validation PR-AUC.

| lr | grad_steps | k | val pooled PR-AUC |
|---:|---:|---:|---:|
| **0.001** | **1** | 1 | **0.8624** |
| 0.001 | 3 | 1 | 0.8532 |
| 0.005 | 1 | 1 | 0.8373 |
| 0.005 | 3 | 1 | 0.8223 |

**Selected adaptation configuration: lr=0.001, grad_steps=1.** Every
candidate is below the static model's own validation PR-AUC (0.8640), so
adaptation did not improve the validation walk itself, the same pattern seen
in Phase 4. Only 5 validation time steps were available for this selection.

### Test metrics (time steps 35-49, shared frozen threshold 0.898)

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Accuracy | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Static GraphSAGE | 0.440 | 0.608 | 0.510 | 0.875 | 0.430 | 0.924 | 0.054 | 0.392 |
| Adaptive GraphSAGE (k=1) | 0.497 | 0.647 | 0.562 | 0.889 | 0.524 | 0.935 | 0.046 | 0.353 |
| Adaptive − Static | +0.057 | +0.040 | **+0.052** | +0.014 | **+0.095** | | | |

Static confusion matrix: TN=14748, FP=839, FN=425, TP=658. Adaptive
confusion matrix: TN=14877, FP=710, FN=382, TP=701. Both models are
evaluated on the identical 16,670 labeled test nodes. Verified directly:
`time_step` and `y_true` arrays are element-for-element equal, and the t=35
predictions are bit-identical before the first update.

### k=3 sensitivity result

Same selected lr/grad_steps, only the delay changed. F1 = 0.534,
PR-AUC = 0.498, ROC-AUC = 0.884 (Precision 0.456, Recall 0.644). It sits
between the static model and k=1 on these metrics. It is reported
separately and was **not** used for any model or configuration selection.

### Per-time-step results

| t | n labeled | illicit rate | Static F1 | Adaptive F1 | Delta (Adaptive − Static) |
|---:|---:|---:|---:|---:|---:|
| 35 | 1341 | 0.136 | 0.756 | 0.756 | 0.000 |
| 36 | 1708 | 0.019 | 0.313 | 0.339 | +0.026 |
| 37 | 498 | 0.080 | 0.475 | 0.605 | +0.129 |
| 38 | 756 | 0.147 | 0.584 | 0.768 | +0.184 |
| 39 | 1183 | 0.068 | 0.718 | 0.756 | +0.038 |
| 40 | 1211 | 0.092 | 0.528 | 0.558 | +0.030 |
| 41 | 1132 | 0.102 | 0.544 | 0.635 | +0.091 |
| 42 | 2154 | 0.111 | 0.633 | 0.662 | +0.029 |
| 43 | 1370 | 0.018 | 0.000 | 0.000 | 0.000 |
| 44 | 1591 | 0.015 | 0.043 | 0.036 | −0.006 |
| 45 | 1221 | 0.004 | 0.000 | 0.000 | 0.000 |
| 46 | 712 | 0.003 | 0.074 | 0.091 | +0.017 |
| 47 | 846 | 0.026 | 0.000 | 0.000 | 0.000 |
| 48 | 471 | 0.076 | 0.000 | 0.000 | 0.000 |
| 49 | 476 | 0.118 | 0.023 | 0.000 | −0.023 |

Files: `results/metrics/temporal_static_vs_adaptive_graphsage_f1_delta.csv`,
`temporal_adaptive_graphsage.csv`, and the `adaptive_graphsage_*` figures in
`results/figures/`.

### Temporal-period comparison

| period | mean per-step F1 delta (Adaptive − Static) |
|---|---:|
| t=35-42 | **+0.066** |
| t=43-49 | **−0.002** |

The observed aggregate improvement is concentrated in the earlier test
period, and the later period shows essentially no mean improvement. This
document does not claim that adaptation causally fails after t=43, and it
does not claim that t=43 is a proven regime boundary. The split at 43 is the
same descriptive cut used in Phases 3 and 4.

### Exploratory Wilcoxon analysis

An exploratory paired Wilcoxon signed-rank test across the 15 temporal F1
differences yielded W=4.0 and two-sided p=0.0166. Five differences were
exactly zero and were dropped, leaving 10 non-zero pairs. Because the
observations are temporally ordered rather than independent, and the
experiment uses a single seed and selected configuration, this is treated as
descriptive supporting evidence rather than evidence of generalization. It
is not presented as proof of statistical significance.

### Random Forest comparison

Random Forest test F1 = 0.790; Adaptive GraphSAGE test F1 = 0.562; absolute
gap ≈ 0.228. Adaptive GraphSAGE improves substantially over its static
GraphSAGE baseline but remains below the Random Forest benchmark on the
reported test F1. The adaptive GNN does not outperform Random Forest.

### Connection to Phase 4

The qualitative temporal pattern is similar in the two experiments:

| | overall F1 | mean delta t=35-42 | mean delta t=43-49 |
|---|---|---:|---:|
| Phase 4: Adaptive GCN | 0.390 → 0.448 | +0.058 | −0.001 |
| Phase 6: Adaptive GraphSAGE | 0.510 → 0.562 | +0.066 | −0.002 |

This is a descriptive similarity only. It does not show that the same
mechanism caused both effects.

### Limitations / caveats

- Single seed (42); no multi-seed robustness study has been run.
- Adaptation hyperparameters were selected on only 5 validation time steps
  (30-34), so the selection has limited statistical power and some variance.
- The feedback delay is simulated, not a dataset property.
- The Adam optimizer state persists across sequential adaptation events, so
  `grad_steps=1` is not an independent fixed-size update at each event.
- The 15 per-step observations behind the Wilcoxon test are temporally
  ordered, not independent.
- Illicit prevalence in t=43-49 is low (about 0.3-11.8% per step, with
  several steps under 3%), so per-step F1 there is noisy; several steps are
  exactly 0 for both models.
- No replay was used.
- See also `docs/LIMITATIONS.md` L2 and L3.

### Files

`src/training/adaptive_graphsage.py`, `scripts/train_adaptive_graphsage.py`,
`scripts/evaluate_adaptive_graphsage.py`, `tests/test_adaptive_graphsage.py`.
Outputs: `results/metrics/adaptive_graphsage_{hparam_selection,train_manifest,results}.json`,
`results/metrics/predictions/adaptive_graphsage_test{,_k3}.npz`, and the
temporal CSVs and figures listed above. No Phase 1-5 file was modified.

---

## Phase 7 — Multi-seed robustness of Static vs Adaptive GraphSAGE

### Research question

Is the Phase 6 improvement from Static GraphSAGE to Adaptive GraphSAGE
(F1 0.510 to 0.562 at seed 42) reproducible across random seeds, or specific
to the single seed used there? This is a robustness study only. No new
architecture or adaptation mechanism was introduced, and nothing was retuned.

### Design

- **Seeds:** exactly 42, 123, 456, 789, 2024.
- **Static GraphSAGE (same for every seed):**
  `SAGEConv(165,128) -> ReLU -> Dropout(0.5) -> SAGEConv(128,1)`, Adam
  lr=0.01, weight_decay=5e-4, 200 epochs, checkpoint chosen by validation
  PR-AUC (30-34), same loss and train-derived `pos_weight` as Phase 5.
- **Adaptive GraphSAGE (same for every seed):** initialized from the same
  seed's static checkpoint; delayed-feedback walk from Phase 4 reused
  unchanged; lr=0.001, grad_steps=1, k=1, no replay. The configuration was not
  reselected per seed.
- **Split:** train 1-29, validation 30-34, test 35-49. Unknown labels are
  excluded from loss and metrics. All comparisons use the same 16,670 labeled
  test nodes.
- **Primary threshold:** fixed at **0.898** (the Phase 6 value) for every
  seed. It is never selected with test labels.
- **Seed 42 is the official reference.** Its result is the existing Phase 5/6
  artifacts, reused read-only and not retrained.
- **Code:** `scripts/train_multiseed_robustness.py`,
  `scripts/evaluate_multiseed_robustness.py`,
  `tests/test_multiseed_robustness.py`. Outputs are under
  `results/metrics/multiseed/`, `results/models/multiseed/` and
  `results/figures/multiseed_robustness.png`.

The feedback delay (labels for t available before prediction at t+1) is a
simulated experimental assumption, not a dataset property (`docs/LIMITATIONS.md`
L2).

### Per-seed results (fixed threshold 0.898)

| Seed | Static F1 | Adaptive F1 | ΔF1 | Static PR-AUC | Adaptive PR-AUC | ΔPR-AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 42 (reference) | 0.510 | 0.562 | +0.052 | 0.430 | 0.524 | +0.095 |
| 123 | 0.438 | 0.537 | +0.099 | 0.511 | 0.560 | +0.049 |
| 456 | 0.462 | 0.549 | +0.086 | 0.411 | 0.536 | +0.125 |
| 789 | 0.534 | 0.598 | +0.064 | 0.479 | 0.567 | +0.087 |
| 2024 | 0.526 | 0.584 | +0.058 | 0.481 | 0.578 | +0.097 |

Remaining per-seed metrics (Static → Adaptive):

| Seed | Precision | Recall | ROC-AUC | ΔROC-AUC |
|---:|---|---|---|---:|
| 42 (reference) | 0.440 → 0.497 | 0.608 → 0.647 | 0.875 → 0.889 | +0.014 |
| 123 | 0.326 → 0.498 | 0.668 → 0.583 | 0.883 → 0.889 | +0.006 |
| 456 | 0.361 → 0.483 | 0.644 → 0.635 | 0.870 → 0.887 | +0.017 |
| 789 | 0.459 → 0.539 | 0.638 → 0.672 | 0.879 → 0.893 | +0.014 |
| 2024 | 0.482 → 0.530 | 0.579 → 0.650 | 0.880 → 0.895 | +0.015 |

Validation checkpoint selection per seed (validation PR-AUC, best epoch):
seed 42 reference 0.864 (epoch 126); 123: 0.861 (38); 456: 0.872 (63);
789: 0.865 (125); 2024: 0.872 (197).

### Aggregate results across the five seeds (seed 42 = official reference)

| Metric | Mean | Std | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Static F1 | 0.494 | 0.042 | 0.510 | 0.438 | 0.534 |
| Adaptive F1 | 0.566 | 0.025 | 0.562 | 0.537 | 0.598 |
| **ΔF1** | **+0.072** | 0.020 | +0.064 | +0.052 | +0.099 |
| Static PR-AUC | 0.462 | 0.041 | 0.479 | 0.411 | 0.511 |
| Adaptive PR-AUC | 0.553 | 0.022 | 0.560 | 0.524 | 0.578 |
| **ΔPR-AUC** | **+0.091** | 0.028 | +0.095 | +0.049 | +0.125 |

Std is the sample standard deviation (n=5).

**Across the five predefined seeds, Adaptive GraphSAGE improved over its
matched Static GraphSAGE baseline in 5/5 seeds on F1 and 5/5 seeds on
PR-AUC.** ROC-AUC also increased in all five seeds (+0.006 to +0.017). The
smallest ΔF1 (+0.052) is the original seed-42 reference. With only five seeds
this is descriptive evidence of a consistent direction and effect size, not a
significance claim. Static F1 itself varies by about 0.10 across seeds
(0.438 to 0.534), so a single-seed static number carries real uncertainty.

### Temporal pattern (mean per-step ΔF1, Adaptive − Static)

| Seed | t=35-42 | t=43-49 |
|---:|---:|---:|
| 42 (reference) | +0.066 | −0.002 |
| 123 | +0.057 | −0.009 |
| 456 | +0.068 | +0.008 |
| 789 | +0.092 | −0.003 |
| 2024 | +0.091 | −0.026 |
| **Across seeds: mean ± std** | **+0.075 ± 0.016** (5/5 positive) | **−0.007 ± 0.012** (1/5 positive) |

The observed improvement is concentrated in t=35-42 in every seed, while
t=43-49 shows essentially no mean improvement, with mixed sign. **This is a
descriptive pattern, not a causal claim.** The document does not claim that
t=43 is a proven regime boundary or explain why the later period differs; the
split at 43 is the same descriptive cut used in Phases 3, 4 and 6. Illicit
prevalence in t=43-49 is low, so per-step F1 there is noisy.

### Seed-42 reproducibility issue and the clean replicate

The original Phase 5 grid built each model before `train_gcn` reseeded the
random-number generator, so the selected seed-42 checkpoint (grid config 16 of
16) started from the RNG state left by the previous configuration. That
checkpoint therefore **cannot be recreated from a bare `seed=42`**. The
original Phase 6 result (Static 0.510, Adaptive 0.562) remains the **official
seed-42 reference** and is what the five-seed table above uses.

To test continuity, the Phase 7 protocol (seed, then construct the model, then
train) was also run from scratch for seed 42 as a separate replicate. It is
reported here and is **not** substituted for the reference:

| Seed-42 run | Static F1 | Adaptive F1 | ΔF1 | Static PR-AUC | Adaptive PR-AUC | Val PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Phase 6 reference (official) | 0.510 | 0.562 | +0.052 | 0.430 | 0.524 | 0.864 |
| Phase 7 clean replicate | 0.440 | 0.537 | +0.098 | 0.462 | 0.552 | 0.859 |

The replicate does not reproduce the reference, so exact reproducibility of
the original seed-42 numbers is not claimed. The two are different but equally
valid seed-42 models. Adaptation improved the replicate as well (its mean ΔF1
by period is +0.055 for t=35-42 and +0.009 for t=43-49). As a sensitivity
check only, replacing the reference by the replicate would give a five-seed
mean ΔF1 of +0.081 with 5/5 seeds improved; this substitution is not used in
any headline number.

### Secondary analysis: per-seed validation-F1 threshold (SECONDARY)

The primary analysis stays at the fixed 0.898. As a clearly secondary
sensitivity analysis, each seed's own validation-F1-maximizing threshold
(selected on validation predictions only) was also applied:

| Seed | Own threshold | Static F1 | Adaptive F1 | ΔF1 |
|---:|---:|---:|---:|---:|
| 42 (reference) | 0.898 | 0.510 | 0.562 | +0.052 |
| 123 | 0.950 | 0.483 | 0.553 | +0.070 |
| 456 | 0.941 | 0.482 | 0.566 | +0.085 |
| 789 | 0.913 | 0.539 | 0.604 | +0.065 |
| 2024 | 0.877 | 0.530 | 0.582 | +0.052 |

Mean ΔF1 is +0.065 (std 0.014), positive in 5/5 seeds. This matches the
direction of the primary analysis and is not used to change any primary
conclusion. It does show that 0.898 is calibrated to the seed-42 model; other
seeds' own thresholds range from 0.877 to 0.950.

### Comparison with Random Forest

Random Forest test F1 remains 0.790. Adaptive GraphSAGE averages 0.566
(range 0.537 to 0.598), so Random Forest stays stronger on test F1 in every
seed, by roughly 0.19 to 0.25. Adaptive GraphSAGE improves over its own static
baseline but does not reach the Random Forest benchmark.

### Integrity checks (all passed)

- Static and Adaptive test populations are identical in every seed (same
  `time_step` and `y_true` arrays, n=16,670) and identical across seeds.
- t=35 predictions are bit-identical between Static and Adaptive in every seed,
  confirming the adaptive run starts from the matched checkpoint.
- No future labels are accessed (event-log check per seed); predictions are
  deterministic and are not rewritten after later adaptation.
- The primary analysis uses the fixed threshold 0.898; each seed's own
  threshold, recomputed from its saved validation predictions, matches the
  stored value, so no test data was involved.
- The adaptation configuration is identical for every seed
  (lr=0.001, grad_steps=1, k=1, no replay) and static training has no test
  argument.
- The seed set is exactly {42, 123, 456, 789, 2024}.
- No Phase 1-6 tracked file was modified. Full suite: 80 tests passed
  (55 existing + 25 new Phase 7 tests).

### Limitations

See `docs/LIMITATIONS.md` L4. In brief: only five seeds, a single dataset,
a single main adaptive configuration, temporally ordered non-independent
observations, the seed-42 initialization issue, and threshold sensitivity.
No claim is made beyond the Elliptic dataset.

---

## Phase 8 — Diagnostics: why do the GNNs trail Random Forest?

### Purpose

Phases 3–7 established that every GNN variant, static or adaptive, trails
Random Forest (test F1 0.790) by a wide margin. Phase 8 does not chase a
higher number — it investigates *why*, through four small, hypothesis-
driven, pre-declared diagnostic experiments, all evaluated under a
stricter rolling-origin validation protocol (§ below) and none touching
the test period. Every experiment answers one specific question; none is
an open-ended architecture or hyperparameter search.

### Rolling-origin validation protocol

Introduced specifically for Phase 8 because the fixed split gives only one
validation window (5 time steps). Three chronological folds, each
strictly earlier than the next, none reaching the test period:

```
fold 1: train 1–19,  validate 20–24
fold 2: train 1–24,  validate 25–29
fold 3: train 1–29,  validate 30–34   (identical to the original fixed split)
```

`src/training/rolling_origin.py::ROLLING_ORIGIN_FOLDS` and `TEST_RANGE`
are the single source of truth, reused unchanged by every Phase 8
experiment; `TEST_RANGE` is enforced with a runtime assertion that refuses
to build any data range reaching t=35, not just documented as a rule.

### E1 — Feature-only MLP control

**Question:** is the GNN family's weakness "neural networks underperform
trees on this tabular data" or "the graph specifically isn't helping" —
these are different claims and need to be separated.

165→128→1 MLP, `ReLU`, `Dropout(0.5)`, same training conventions as the
GraphSAGE grid (Adam lr=0.01, weight_decay=5e-4, 200 epochs, checkpoint by
validation PR-AUC), 3 seeds (42, 123, 456) × 3 folds.

| Fold | Val PR-AUC (mean ± std) | Val F1 (mean ± std) |
|---|---:|---:|
| fold1 | 0.762 ± 0.007 | 0.739 ± 0.003 |
| fold2 | 0.939 ± 0.006 | 0.918 ± 0.001 |
| fold3 | 0.893 ± 0.005 | 0.881 ± 0.005 |

**Answer:** even a plain feature-only MLP (no graph at all) trails the
tuned trees substantially on the same folds (E2 below: trees reach ~0.95
mean val PR-AUC). This is evidence that neural-network-vs-tree is at
least part of the gap — separate from, and prior to, any question about
graph structure.

### E2 — Equal-budget tree benchmark

**Question:** is the Random Forest baseline (Phase 2's 3-config shallow
search) a fair comparison target, or an under-tuned strawman?

Random Forest (16 predeclared configs: `max_depth` ∈ {8,16,24,None},
`min_samples_leaf` ∈ {1,5}, `max_features` ∈ {sqrt,log2}, `n_estimators` ∈
{200,300}) and HistGradientBoostingClassifier (16 predeclared configs:
`max_depth` ∈ {None,6,10}, `learning_rate` ∈ {0.05,0.1,0.2}, `max_iter` ∈
{100,200}, `min_samples_leaf` ∈ {20,50}), 5 seeds, same 3 folds.

| Model | Best config | Mean val PR-AUC |
|---|---|---:|
| Random Forest | max_depth=24, min_samples_leaf=1, max_features=sqrt, n_estimators=300 | 0.948 |
| **HistGradientBoosting** | **max_depth=None, learning_rate=0.1, max_iter=100, min_samples_leaf=50** | **0.950** |

**Answer:** yes, the tree benchmark is fair — a proper tuning pass over
both tree families lands within 0.003 PR-AUC of Phase 2's original RF
selection, confirming Phase 2's shallow search wasn't accidentally
under-tuned relative to what a larger budget finds. HistGradientBoosting's
selected config became the frozen base model for E5 and E7 below.

### E3 — Direction-aware graph aggregation

**Question:** does the raw, one-directional edge orientation throw away
usable structural signal?

PyG's `SAGEConv` aggregates only from a node's incoming neighbors under
the raw `edge_index`; outgoing neighbors never contribute a message. Two
variants tested against a plain-GraphSAGE control (Phase 5's architecture,
unmodified), same 128-hidden/0.5-dropout/0.01-lr settings, 3 seeds × 3
folds:

- **Symmetrized**: both `SAGEConv` layers see the union of edges and
  their reverse.
- **Direction-aware**: separate `SAGEConv` branches for incoming and
  outgoing neighbors (each 64-dim, concatenated to 128 to match the
  baseline's hidden width), output layer over symmetrized edges.

| Architecture | Mean val PR-AUC | Δ vs. plain |
|---|---:|---:|
| Plain GraphSAGE (control) | 0.860 | — |
| Symmetrized | 0.894 | +0.034 |
| **Direction-aware** | **0.902** | **+0.042** |

Both variants beat the control in **all 3 folds**, not just on average.
fold3 (identical train/val range to every earlier phase) shows the
largest gain: plain 0.864 → direction-aware 0.951.

**Answer:** yes — how a GNN uses the graph matters. This is the strongest
positive signal in Phase 8, and it has **not yet been evaluated on the
test period** — see `docs/LIMITATIONS.md` L5 for why that evaluation was
deliberately deferred rather than skipped.

### E5 — Tree + engineered graph features

**Question:** does adding explicit, label-free graph-derived features to
the *strongest* model (E2's HistGradientBoosting) help?

334 predeclared features appended to the original 165 (499 total): in/out/
total degree, 2-hop reach, and 165-dim mean feature vectors over incoming
and outgoing neighbors separately — computed per snapshot, never using
labels, never crossing time steps. Same frozen HGB config as E2, 5 seeds
× 3 folds, condition A (165 features) vs. condition B (499 features).

| Condition | Mean val PR-AUC | Mean val F1 |
|---|---:|---:|
| Baseline (165 features) | 0.9505 | 0.9204 |
| Graph-augmented (499 features) | 0.9483 | 0.9203 |
| **Δ** | **−0.0022** | **−0.0001** |

Consistent small decline across all 3 folds and 4 of 5 seeds — small
relative to the ~0.058 seed-to-seed standard deviation, so the honest
reading is "no measurable benefit," not "actively harmful."

**Answer:** no. A plausible, unverified explanation: the released features
f94–f165 are documented, by external convention this project cannot
verify against the raw file, as already being one-hop aggregated neighbor
features — if so, the 334 new columns may mostly duplicate information
the tree already had, adding dimensionality without new signal.

### E7 — Graph-score smoothing

**Question:** can post-hoc blending of a tree's predictions with its
neighbors' predictions improve the strongest tabular model?

`smoothed = (1-α)·tree_score + α·mean(neighbor tree_scores)`, over both
incoming and outgoing neighbors (never labels), at α ∈ {0, 0.1, ..., 0.5}
and propagation depth ∈ {1, 2}. Same frozen HGB config and folds/seeds as
E5.

| α | Depth=1 mean val PR-AUC |
|---:|---:|
| **0.0 (no smoothing)** | **0.9505** |
| 0.1 | 0.9469 |
| 0.2 | 0.9435 |
| 0.3 | 0.9395 |
| 0.4 | 0.9344 |
| 0.5 | 0.9220 |

Monotonically worse as α increases; depth=2 is uniformly worse than
depth=1 at every matching α. **Best configuration: no smoothing at all.**

**Answer:** no. Two of the two folds with near-saturated baseline PR-AUC
(fold2, fold3) show a marginal gain at α=0.1 before declining — a genuine
but small nuance, well within noise, that does not change the selection.

### Synthesis: the strongest defensible conclusion

E1, E5, and E7 together support: **naive, post-hoc or feature-level
incorporation of graph information does not improve an already-strong
tabular classifier on this dataset.** E3 shows the opposite is *not* true
for how a GNN itself processes the graph internally — direction-aware
message passing measurably helps a GNN, on validation. These are two
different, both-true findings; neither should be generalized into the
other. In particular, this project does **not** conclude that "graph-based
message passing combined with adaptation provides a superior modeling
pathway" — the adaptive GNN family still trails Random Forest by a wide
margin (§ Phase 6/7 above), and no experiment here demonstrates that graph
information beats what the strongest tabular model already achieves.

### Leakage verification

Every Phase 8 module enforces the test-period boundary structurally
(`TEST_RANGE` assertions in `rolling_origin.py`, `rolling_origin_graph.py`,
`graph_smoothing.py`, `graph_features.py`), not just by convention. No
test-period prediction file exists anywhere under
`results/metrics/predictions/` for E1, E2, E3, E5, or E7 — confirmed
directly, not assumed. 64 dedicated tests across
`tests/test_phase8_*.py` cover: no cross-time features/smoothing, no
label usage in any graph-derived feature or smoothing computation, correct
incoming/outgoing directional handling, deterministic behavior, and
correct handling of isolated (no-neighbor) nodes.

### Files

`src/models/mlp.py`, `src/models/graphsage_directional.py`,
`src/training/rolling_origin.py`, `src/training/rolling_origin_graph.py`,
`src/training/graph_smoothing.py`, `src/training/graph_features.py`,
`scripts/run_phase8_diagnostics.py`, `scripts/run_phase8_e3_diagnostics.py`,
`scripts/run_phase8_e5_diagnostics.py`, `scripts/run_phase8_e7_diagnostics.py`,
`tests/test_phase8_*.py`. Outputs: `results/metrics/phase8/*.json`,
`results/metrics/phase8/*.csv`. No Phase 1–7 file was modified.
