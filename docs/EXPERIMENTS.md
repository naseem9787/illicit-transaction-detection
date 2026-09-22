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
