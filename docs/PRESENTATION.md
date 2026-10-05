# PRESENTATION.md

Slide structure and speaking scripts for the final presentation. Numbers come
from `docs/RESULTS.md`. Use the figures in `results/figures/`.

## The one-sentence message

> Online adaptation gives GraphSAGE a real but modest gain under drift, but a
> well-featured tree ensemble is still the strongest model on Elliptic, and
> the same adaptation idea hurt the tree model.

## What we present as "the model"

There is no single winner to crown. We present a **comparison with a finalist
list**:

| Model | Role |
|---|---|
| Logistic Regression | simple baseline |
| Random Forest | strongest benchmark and best detector (F1 0.790) |
| Static GraphSAGE | graph model, no adaptation (F1 0.494 ± 0.042) |
| **Adaptive GraphSAGE** | **the studied method (F1 0.566 ± 0.025)** |
| Static / Adaptive HGB | context: does adaptation help a tree too? (0.787 → 0.469) |

## Ten slides

1. **Title and question.** Does adapting a graph model on newly revealed
   labels help detect illicit Bitcoin transactions under drift?
2. **Why it matters.** AML, drift, labels arrive late. Elliptic dataset facts.
3. **Data and split.** 49 time steps, 165 features, train 1–29, val 30–34,
   test 35–49; unknown labels excluded.
4. **No-leakage pipeline.** Chronological split, thresholds from validation,
   unknown labels never in loss or metrics, audits and tests (140+ tests).
5. **Models.** Logistic Regression, Random Forest, GCN, GraphSAGE; why
   GraphSAGE was chosen (beat GCN).
6. **Adaptive protocol.** Diagram: predict t, freeze, reveal labels of t,
   fine-tune, predict t+1. State clearly that the delay is simulated and the
   mechanism is plain fine-tuning (not RL, not meta-learning).
7. **Main result.** Table from `RESULTS.md` §2 with `final_comparison.png`.
   Adaptive beats static in 5/5 seeds; Random Forest still far ahead.
8. **Per-step view.** `final_f1_over_time.png`: gain is in t = 35–42, none
   in t = 43–49.
9. **Controls and negative results.** MLP, graph features, smoothing,
   direction-aware (validation only), and Adaptive HGB collapsing
   (0.787 → 0.469). Say it plainly.
10. **Conclusions, limitations, future work.** Static trees win; adaptation is
    modest and mechanism-specific; simulated delay; one dataset; future work.

## 5-minute story (about 600 words of speech)

Minute 1 (slides 1–2): Fraud changes over time, and labels arrive late. We
asked whether a graph model that keeps learning from newly confirmed labels
holds up better than a frozen one.
Minute 2 (slides 3–4): We used the Elliptic Bitcoin graph, split by time so
the test period is entirely in the future. Every threshold was chosen on
validation, never on test.
Minute 3 (slides 5–6): We compared Logistic Regression, Random Forest, and
GraphSAGE. Adaptive GraphSAGE predicts a time step, then receives its labels
and fine-tunes a little before the next one. The one-step delay is an
assumption we made, not a property of the dataset.
Minute 4 (slides 7–8): Adaptation raised GraphSAGE from F1 0.494 to 0.566,
in all five seeds. But Random Forest scores 0.790. We do not beat it, and we
say so.
Minute 5 (slides 9–10): When we gave a gradient-boosted tree the same
feedback, it got worse, 0.787 to 0.469. So adaptation is not automatic, and
our evidence is limited to one dataset and a simulated delay. The honest
conclusion: on Elliptic, a static tree ensemble is the best detector.

## 10-minute story (split across speakers)

Suggested split for a team of three. Adjust names and slide ranges.

- **Speaker A (slides 1–4, about 3 min):** problem, drift, dataset, split,
  leakage controls, why we exclude unknown labels, what the 147 tests cover.
- **Speaker B (slides 5–8, about 4 min):** models, adaptive protocol (draw
  the loop), main table, multi-seed result, per-step temporal pattern,
  k = 3 sensitivity (F1 0.534).
- **Speaker C (slides 9–10, about 3 min):** why Random Forest wins (features
  already carry neighbour information, hypothesis), E1/E5/E7 negatives,
  Adaptive HGB failure and what it does and does not prove, limitations,
  future work.

## Demo / backup material

- `results/figures/adaptive_graphsage_f1_over_time.png`,
  `adaptive_graphsage_f1_delta.png`, `adaptive_graphsage_pr_roc_curves.png`.
- Per-seed table: `results/metrics/multiseed/per_seed_results.csv`.
- Phase 8 summary: `results/metrics/phase8/phase8_summary.json`.

## Delivery rules

- Never say "we beat" Random Forest. Never say "novel" or "state of the
  art". Never say "real-time" or "real-world feedback delay".
- Always attach "simulated delay" the first time an adaptive number appears.
- Show the failure (Adaptive HGB) on a slide, not only in the appendix.
