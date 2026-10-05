# RESULTS.md

Final results of the study. Everything here is on the Elliptic test period
(t = 35–49, 16,670 labelled nodes, 1,083 illicit), at thresholds frozen on
validation. Produced by `scripts/run_final_evaluation.py` under
`docs/FINAL_MODEL_PROTOCOL.md`. Raw files: `results/metrics/final/`.

Read these numbers with the limitations in `docs/LIMITATIONS.md`, in
particular L1 (feature-standardization provenance) and L2 (the feedback delay
is a simulated assumption).

## 1. Headline

1. **A static tree ensemble is the best model in this study.** Random Forest
   (F1 0.790) and Static HGB (F1 0.787 ± 0.021, five seeds) are far ahead of
   every graph model.
2. **Adaptive GraphSAGE improves on Static GraphSAGE but does not come close
   to the tree models.** Five-seed mean F1 0.566 ± 0.025 against 0.494 ± 0.042
   for static; improvement in 5/5 seeds. It stays about 0.22 F1 below Random
   Forest.
3. **The adaptive-HGB context model failed badly.** Giving HistGradientBoosting
   the same delayed label feedback, through warm-start boosting, *reduced* test
   F1 from 0.787 to 0.469 and PR-AUC from 0.795 to 0.205, in 5/5 seeds. This
   is a negative result, reported as is.

## 2. Final comparison (test t = 35–49)

| Model | Runs | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 1 | 0.202 | 0.811 | 0.324 | 0.857 | 0.210 |
| **Random Forest** | 1 | 0.926 | 0.689 | **0.790** | 0.936 | 0.788 |
| Static GraphSAGE (seed-42 reference) | 1 | 0.440 | 0.608 | 0.510 | 0.875 | 0.430 |
| Adaptive GraphSAGE (seed-42 reference) | 1 | 0.497 | 0.647 | 0.562 | 0.889 | 0.524 |
| Static GraphSAGE (5 seeds) | 5 | 0.413 | 0.627 | 0.494 ± 0.042 | 0.878 | 0.462 ± 0.041 |
| Adaptive GraphSAGE (5 seeds) | 5 | 0.509 | 0.637 | 0.566 ± 0.025 | 0.891 | 0.553 ± 0.022 |
| Static HGB (context) | 5 | 0.878 | 0.715 | 0.787 ± 0.021 | 0.939 | 0.795 ± 0.003 |
| Adaptive HGB (context) | 5 | 0.348 | 0.734 | 0.469 ± 0.060 | 0.803 | 0.205 ± 0.037 |

Thresholds (frozen on validation): Logistic Regression 0.956, Random Forest
0.533, GraphSAGE (static and adaptive, shared) 0.898, HGB per seed 0.72–0.84
(shared between its static and adaptive version). Logistic Regression, Random
Forest and the seed-42 GraphSAGE rows are single runs; no variance is
available for them.

Figures: `results/figures/final_comparison.png`,
`results/figures/final_f1_over_time.png`.

## 3. What each comparison shows

### 3.1 Adaptive vs Static GraphSAGE (the project's main adaptive result)

| | Static | Adaptive | Δ |
|---|---|---|---|
| F1, seed 42 reference | 0.510 | 0.562 | +0.052 |
| F1, 5-seed mean | 0.494 | 0.566 | +0.072 |
| PR-AUC, 5-seed mean | 0.462 | 0.553 | +0.091 |

Adaptive beat static in 5/5 seeds. The gain is concentrated in t = 35–42
(mean per-step F1 change +0.066 in the seed-42 reference) and is absent in
t = 43–49 (−0.002), where illicit prevalence is very low and per-step F1 is
noisy. Under the Phase 6 sensitivity check with a 3-step delay the seed-42
F1 is 0.534. All of this is conditional on the simulated feedback delay.

### 3.2 Why Random Forest wins

- Tree ensembles on the released features already reach F1 ≈ 0.79. GraphSAGE
  starts from a lower level and adaptation recovers only part of the gap.
- Graph-side diagnostics (Phase 8) did not close the gap: an MLP control, a
  tree benchmark, tree + engineered graph features (E5) and graph-score
  smoothing (E7) were all negative or neutral. A plausible, **unverified**
  explanation is that features f94–f165 already contain one-hop neighbour
  aggregates (the 94/72 local/aggregated split is an external convention not
  verified against the raw file, `docs/DATA_AUDIT.md` §8), leaving little
  extra signal for message passing.
- The GNNs lose ground under temporal drift: static GraphSAGE validation
  PR-AUC 0.864 falls to 0.430 on test, while Static HGB holds 0.795.

### 3.3 Adaptive HGB: what happened, and what it does and does not mean

Adaptive HGB is identical to Static HGB at t = 35 (the first prediction comes
before any adaptation), then falls behind step by step: per-step F1 at
t = 41 is 0.476 against 0.947 for the static model. Precision collapses
(0.878 → 0.348) while recall rises slightly (0.715 → 0.734), and ROC-AUC also
falls (0.939 → 0.803), so ranking quality is damaged, not just calibration at
the fixed threshold.

What it supports:

- The claim "any model improves simply because it receives newly revealed
  labels" is **not** supported. For this boosting mechanism the opposite
  happened.
- Therefore the Adaptive GraphSAGE gain is not an automatic effect of label
  access.

What it does **not** support:

- It does not show that graph models are inherently better at adaptation. The
  two mechanisms are not like-for-like: the GNN update is a tiny perturbation
  (learning rate 0.001, one gradient step), whereas 10 extra boosting rounds
  at learning rate 0.1, fitted to one snapshot of about 1,000 labelled nodes,
  are a large one. It is one crude tree mechanism, not every way a tree could
  use new labels (retraining on train plus revealed steps was not tried).
- Unverified hypotheses for the collapse (not tested, no tuning allowed after
  seeing test): over-fitting each small snapshot; accumulation over a 15-step
  test walk versus the 5-step validation walks (14 versus 4 updates).
- The Step 2 validation control predicted only a mild effect (−0.0095
  PR-AUC); the test effect (−0.59 PR-AUC) is much larger. Validation walks
  are a poor predictor of adaptive behaviour under test-period drift, which
  is also a limitation for the GNN adaptation settings that were selected on
  validation.

## 4. What this study can and cannot claim

Can claim:

- A leakage-safe chronological pipeline on Elliptic, with every threshold
  and hyperparameter chosen on validation.
- Online fine-tuning on revealed labels improved GraphSAGE on this test
  period, in 5/5 seeds, under the simulated k = 1 protocol.
- That gain is neither large enough to compete with tree ensembles nor
  generic: the same idea applied to HGB hurt.
- Several graph-side remedies (MLP, graph features, smoothing) did not beat
  tree ensembles.

Cannot claim: novelty, state of the art, beating Random Forest, real-time
detection, a real-world feedback delay, generalization beyond Elliptic, or
that the adaptation is reinforcement learning or meta-learning. Direction-
aware GraphSAGE (validation gain of +0.042 PR-AUC, 3/3 folds) was not
evaluated on test.

## 5. Test-set exposure (honest note)

Logistic Regression, Random Forest, GCN and GraphSAGE results on t = 35–49
were first produced in Phases 3–7, with all selection done on validation. The
final table re-scores those frozen predictions; only HGB is a new test
evaluation. See `docs/FINAL_MODEL_PROTOCOL.md` §1.

## 6. Reproduce

```bash
.venv/Scripts/python -m scripts.run_final_evaluation --config config.yaml
```

The script refuses to overwrite an existing result, so it is run once by
design.
