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
