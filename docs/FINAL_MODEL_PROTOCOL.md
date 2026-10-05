# FINAL_MODEL_PROTOCOL.md

Written **before** the final Adaptive-HGB test evaluation was run. It freezes
which models are reported, how each was configured, how thresholds were set,
and how the final comparison is produced. After this file is committed, no
model, configuration, threshold rule or seed list may be changed in response
to any test number.

## 1. Purpose and honest scope

The goal of the project is **not** to beat Random Forest. It is an
empirical study of whether online adaptation on newly revealed labels helps a
graph model under temporal drift on the Elliptic dataset, with an honest
comparison against strong tabular models.

**Disclosure about test-set exposure.** The test period (t = 35–49) was
evaluated in earlier phases for the Logistic Regression, Random Forest,
GCN, Static GraphSAGE and Adaptive GraphSAGE models (Phases 3–7). In every
one of those phases hyperparameters and thresholds were selected on
validation only (t = 30–34 or rolling-origin folds below t = 35); no model was
tuned on test. The final table therefore **reuses those frozen test results
without re-fitting or re-tuning**. The only model that sees the test period
for the first time in this step is Adaptive HGB (and its static twin). The
phrase "evaluated once" applies to that model and to the consolidation
step; it is not a claim that the earlier models never touched test.

## 2. Frozen finalists

| # | Model | Role | Configuration (frozen) | Threshold (frozen, validation) | Source of test result |
|---|---|---|---|---|---|
| 1 | Logistic Regression | simple baseline | `class_weight=balanced`, 165 features, no time_step | 0.956 (max F1 on val) | `baseline_results.json` |
| 2 | Random Forest | strongest tabular benchmark | 300 trees, `max_depth=None`, `class_weight=balanced` (selected by val PR-AUC) | 0.533 (max F1 on val) | `baseline_results.json` |
| 3 | Static GraphSAGE | graph model, no adaptation | 2-layer GraphSAGE, hidden 128, dropout 0.5, lr 0.01, wd 5e-4, 200 epochs, best epoch by val PR-AUC; seed 42 reference | 0.898 (max F1 on val) | `adaptive_graphsage_results.json` |
| 4 | Adaptive GraphSAGE | the proposed method | Model 3 + online fine-tuning: lr 0.001, 1 gradient step, delay k = 1, fixed train-derived `pos_weight`, no replay | same 0.898, shared with Model 3 | `adaptive_graphsage_results.json` |
| 5 | Adaptive HGB | context model (is the adaptive gain graph-specific?) | HistGradientBoosting, `max_depth=None`, lr 0.1, `max_iter=100`, `min_samples_leaf=50`; adaptation = `warm_start` +10 boosting rounds per revealed step, fixed train-derived class sample weights, k = 1 | per seed: max F1 on val (t = 30–34) of the **static** model; shared with Static HGB | **new** — `final_evaluation_results.json` |

Static HGB is run alongside Adaptive HGB as its necessary pair (identical
starting model, identical threshold). It is not an additional finalist.

Excluded from the final comparison (reported elsewhere as diagnostics):
GCN, Adaptive GCN, MLP, direction-aware GraphSAGE, engineered graph features,
graph-score smoothing.

## 3. Data, split and metrics

- Train t = 1–29, validation t = 30–34, test t = 35–49; unknown labels are
  excluded from loss, thresholds and every metric.
- Test population: 16,670 labelled nodes, 1,083 illicit.
- Metrics at the frozen threshold: precision, recall, F1 (primary,
  illicit class), ROC-AUC, PR-AUC; plus per-time-step F1.
- Seed policy: GraphSAGE reference = seed-42 checkpoint from Phases 5–6, with
  the five-seed Phase 7 study (42, 123, 456, 789, 2024) as robustness
  evidence. HGB uses the same five seeds. LR and RF are single-run
  (seed 42).
- Seed-42 GraphSAGE caveat (L4): the reference checkpoint cannot be
  regenerated from a bare seed; its clean replicate gave F1 0.440 / 0.537.

## 4. Adaptive HGB test protocol (new evaluation)

For each seed in {42, 123, 456, 789, 2024}:

1. Class weights from training labels (t = 1–29) only.
2. Fit the base HGB on train (t = 1–29).
3. Predict validation (t = 30–34) with the **static** model; set the
   threshold to the pooled max-F1 point. Freeze it.
4. Static HGB: predict t = 35–49 with the frozen base model.
5. Adaptive HGB: deep-copy the base model; walk t = 35 → 49; before
   predicting step t, apply the update from the labels of step t − 1
   (+10 boosting rounds); predict t; freeze that prediction.
6. Both use the same frozen threshold. Validation labels (t = 30–34) are not
   used for adaptation, matching the GraphSAGE protocol, in which the walk
   starts at t = 35.

The 10-round increment and k = 1 were declared in Step 2 and are not tuned.
The feedback delay is a simulated assumption (L2), not a dataset fact.

The script refuses to overwrite an existing result file, so the evaluation
cannot be silently re-run and re-selected.

## 5. Reporting rules

- Report all five models in one table, including every model that loses.
- Report the Adaptive HGB result whatever it is. If Adaptive HGB also gains
  from adaptation, the claim that the adaptive gain is specific to graph
  models is withdrawn. If it does not, the claim is only that it did not
  reproduce under this one mechanism.
- No change of finalists, thresholds, seeds or mechanism after seeing test
  numbers. Any further idea is recorded as future work.
- Do not claim: novelty, state of the art, beating Random Forest, real-time
  operation, a real-world feedback delay, generalization beyond Elliptic, or
  RL / meta-learning.
- Qualify every adaptive result with L2 (simulated delay) and every
  "leakage-free" statement with L1 (feature-standardization provenance).

## 6. Decisions fixed here

- Step 4 (direction-aware adaptive GraphSAGE) was **not** run; direction-aware
  GraphSAGE is validation-only evidence (+0.042 PR-AUC, 3/3 folds) and is not
  a finalist.
- After the final evaluation and the submission package, experimentation
  stops.
