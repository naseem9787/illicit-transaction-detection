# PAPER_OUTLINE.md

Working title: **Illicit Transaction Detection Using Graph-Based Adaptive
Learning: An Empirical Study on the Elliptic Bitcoin Dataset**

Type of paper: an honest empirical study, not a new-method paper. The
contribution is a careful, leakage-safe comparison and a set of documented
negative results.

## Abstract (about 150 words, draft)

We study whether online adaptation on newly revealed labels helps graph neural
networks detect illicit Bitcoin transactions under temporal drift, using the
Elliptic dataset (203,769 transactions, 49 time steps) with a strict
chronological split. Under a simulated delayed-feedback protocol, fine-tuning
GraphSAGE on revealed labels raised test F1 from 0.494 to 0.566 (five seeds,
improvement in all five). However, Random Forest (F1 0.790) and
HistGradientBoosting (0.787) remain far stronger, and giving
HistGradientBoosting the same feedback through warm-start boosting reduced its
F1 to 0.469. Diagnostics with an MLP, engineered graph features and graph-score
smoothing did not close the gap. We conclude that on this dataset a static
tree ensemble is the strongest model, that the adaptive gain is real but modest
and mechanism-specific, and we document the limitations of the simulated
protocol.

## 1. Introduction

- Anti-money-laundering on transaction graphs; drift as new fraud patterns
  appear.
- Question: does adapting a graph model on newly revealed labels help, and is
  the gain specific to graph models?
- Contributions (modest): (i) a leakage-audited chronological pipeline;
  (ii) a delayed-feedback adaptation protocol; (iii) a multi-seed adaptive
  GraphSAGE study; (iv) honest controls: MLP, trees, graph features,
  smoothing, adaptive HGB.
- Explicitly not claimed: novelty, SOTA, beating Random Forest.

## 2. Related work (from the Lab 7 literature review)

- Elliptic dataset and its original benchmark (Weber et al. 2019): Random
  Forest strong, GCN weaker.
- GCN, GraphSAGE; temporal GNNs for Elliptic (cite from
  `Lab7_Literature_Review.pdf`; verify every citation before submission).
- Concept drift and online learning in fraud detection.

## 3. Data

Elliptic: 203,769 nodes, 234,355 edges, 165 features, 49 time steps, edges
never cross time steps, labels licit/illicit/unknown. Chronological split:
train 1–29, validation 30–34, test 35–49. Unknown labels excluded from loss
and metrics. Caveats: L1, and the unverified f94–f165 convention.

## 4. Methods

4.1 Leakage-safe pipeline; thresholds from validation only.
4.2 Baselines: Logistic Regression, Random Forest, HGB.
4.3 GCN and GraphSAGE (2 layers, hidden 128).
4.4 Delayed-feedback adaptive protocol: predict, freeze, reveal, adapt;
k = 1 primary, k = 3 sensitivity; fixed train-derived `pos_weight`; no replay.
4.5 Adaptive HGB: warm-start boosting, +10 rounds per revealed step.
4.6 Evaluation: precision, recall, F1 on illicit class, ROC-AUC, PR-AUC,
per-step F1; five seeds.

## 5. Results

- Table: `docs/RESULTS.md` §2. Figure: `final_comparison.png`,
  `final_f1_over_time.png`.
- Adaptive vs static GraphSAGE; temporal pattern (gain in t = 35–42).
- Random Forest and Static HGB dominate.
- Adaptive HGB failure.

## 6. Diagnostics (why the graph models trail)

E1 MLP control; E2 tree benchmark with rolling-origin validation; E3
direction-aware GraphSAGE (validation gain only); E5 graph features; E7
smoothing; Step 2 adaptive-tree control on validation. All negative or
neutral for the "graphs beat trees" hypothesis.

## 7. Discussion

- Feature redundancy hypothesis (unverified).
- Why adaptation helps a weak model but hurts a strong one under this
  mechanism; mechanism size mismatch.
- Validation walks under-predict test behaviour.

## 8. Limitations

L1–L5 plus the final-evaluation notes: single dataset; simulated delay;
five seeds; seed-42 reproducibility gap; test-set exposure of earlier
phases; direction-aware GraphSAGE untested on test; one tree mechanism.

## 9. Conclusion and future work

Conclusion: static tree ensembles win on Elliptic; adaptation gives GraphSAGE
a real but modest gain. Future work (not done): direction-aware adaptive
GraphSAGE; retraining-based tree adaptation; replay buffers; other datasets
with real label timestamps; verifying the f94–f165 provenance.

## Reproducibility appendix

Seeds, split, thresholds, commands: `docs/METHODOLOGY.md` §14 and
`docs/FINAL_MODEL_PROTOCOL.md`.
