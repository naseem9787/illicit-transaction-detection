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

## 10-minute story (four speakers, about 2.5 minutes each)

Replace A-D with your names. Each speaker also owns the viva questions in the
last column, so every examiner question has a named first responder.

| Speaker | Slides | Time | Covers | Viva owner (`docs/VIVA_QA.md`) |
|---|---|---|---|---|
| **A** | 1-3 | 2.5 min | Question, why drift and late labels matter, Elliptic facts, chronological split, why unknown labels are excluded | Data, unknown labels, "is there leakage?", L1 |
| **B** | 4-6 | 2.5 min | Leakage-safe pipeline and the 147 tests, baselines and GraphSAGE, the adaptive loop (draw it), the simulated delay, "this is plain fine-tuning, not RL or meta-learning" | "What is adaptive?", delay realism, thresholds, why GraphSAGE |
| **C** | 7-8 | 2.5 min | Main results table, five-seed gain (0.494 to 0.566), per-step pattern (gain in t = 35-42), k = 3 sensitivity (F1 0.534) | Significance, seeds, "why does the gain vanish after t = 43?", seed-42 gap |
| **D** | 9-10 | 2.5 min | Why Random Forest wins, MLP/graph-feature/smoothing negatives, Adaptive HGB failure and what it does and does not prove, limitations, future work | "Why RF beats you", "what is novel", adaptive HGB, test-set exposure |

Hand-offs: A ends on "so how do we avoid cheating on time?" (B). B ends on
the adaptive loop and "so does it help?" (C). C ends on "but Random Forest is
still far ahead, why?" (D).

Rehearse once with a timer; if over time, cut slide 8 detail (C) and the E1
to E7 listing (D) first, never the limitations.

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
