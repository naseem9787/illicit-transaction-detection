# LIMITATIONS.md

Known limitations and caveats that apply to the project as a whole,
independent of any single phase. This is not a place for phase-specific
open items (those live in `DATA_AUDIT.md` / `DECISIONS.md` / `EXPERIMENTS.md`)
— only for caveats that can't be resolved by better code, because they are
properties of the released dataset itself.

---

## L1 — Unknown provenance of the released feature standardization

Every one of the 165 feature columns in `elliptic_txs_features.csv` is
already globally standardized (mean ≈ 0, std ≈ 1 per column) by the dataset
authors before public release — verified directly in `docs/DATA_AUDIT.md`
section 4.

We do not know, and cannot verify from the released files alone, whether the
mean/std used for that standardization were computed only from an early
subset of time steps or from all 49 time steps (including what are, from our
chronological split's perspective, "future" val/test time steps).

**What this does and does not mean:**

- If the standardization statistics were computed over all 49 time steps,
  then a small amount of dataset-wide aggregate information (the global
  mean/std per feature) is baked into every feature value we receive, before
  our pipeline ever touches the data. This would technically mean the
  released features are not *strictly* independent of the val/test period.
- This is **not** the same as label leakage, and it does **not** mean our
  split, training, or evaluation code leaks information — every check in
  `src/data/preprocessing.py` (`verify_no_leakage`) and every test in
  `tests/test_data.py` confirms the split itself is correctly chronological
  and that no label or index information crosses from val/test into train.
- We have **not** confirmed this actually occurred, and we are not claiming
  it did. It is a documented, unresolved possibility about the public
  release's preprocessing, not a finding.
- It is also not something this project can correct without access to the
  original, unnormalized feature values, which were never released.

**How to treat this:** when reporting results, any claim of a "fully
leakage-free pipeline" should be qualified with this caveat — our pipeline
introduces no leakage, but we cannot rule out a small amount already present
in the upstream released data. This is a known, general caveat about the
public Elliptic release, not something specific to this codebase.

---

## L2 — The Phase 4 feedback delay is a simulated experimental assumption, not a dataset fact

Phase 4's Adaptive GCN (`docs/EXPERIMENTS.md` Phase 4 section) uses a
protocol where, after predicting time step t, t's true labels are treated
as "revealed" and used to adapt the model before predicting t+k, for a
fixed delay k (k=1 primary, k=3 sensitivity).

**This delay is not established by the Elliptic dataset in any way.** The
dataset contains no timestamped label-arrival information — nothing in
`elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`, or
`elliptic_txs_features.csv` says when a transaction's illicit/licit
determination became known relative to its own time step. The dataset
only gives a static `time_step` per transaction and a final label; how
and when that label was actually confirmed in the real investigative
process that produced this dataset is not represented in the data at all.

**What this means:**

- "k=1: labels for t become available before prediction at t+1" is an
  **explicit, documented experimental assumption** adopted to make a
  walk-forward online-adaptation experiment well-defined and testable —
  not a claim about how AML investigations actually worked when this
  dataset was produced.
- Every Phase 4 result (Adaptive GCN test metrics, the k=1 vs k=3
  sensitivity comparison, the per-time-step F1 deltas) should be read as
  conditional on this assumption: *if* labels became available with this
  delay, *then* this is what online weight adaptation achieves. No
  stronger claim is made or should be inferred.
- Choosing a shorter delay (more optimistic about how fast feedback
  arrives) tends to let the model adapt faster; a longer delay is more
  conservative. Neither k=1 nor k=3 was chosen by testing which produces
  better results on the test period — both were fixed in advance, and
  k=3 is reported only as a sensitivity check, never used to select
  between the two (`docs/EXPERIMENTS.md` Phase 4, "Sensitivity check: k=3").

**How to treat this:** any statement about the Adaptive GCN's performance
in a report, presentation, or comparison to Static GCN must carry this
qualifier — it is a result about a *simulated* delayed-feedback protocol
on this dataset, not a validated real-world feedback-latency finding.

---

## L3 — Phase 6 (adaptive GraphSAGE) limitations

These apply to every Phase 6 result in `docs/EXPERIMENTS.md`, in addition to
L2 (the feedback delay is a simulated experimental assumption, not a dataset
fact).

- **Single seed.** All results use seed 42. There is no multi-seed
  robustness study yet, so run-to-run variance is unquantified.
- **Small validation slice.** Adaptation hyperparameters (lr, grad_steps)
  were selected on only 5 validation time steps (30-34) among four close
  candidates, so the selection has limited statistical power and some
  selection variance.
- **Simulated feedback delay.** k=1 (primary) and k=3 (sensitivity) are
  assumptions. Neither was chosen by comparing test performance.
- **Persistent Adam state.** The optimizer is created once per walk and its
  state persists across adaptation events, so `grad_steps=1` is not an
  independent fixed-size update at every event.
- **Wilcoxon caveat.** The exploratory paired Wilcoxon test uses 15 per-step
  F1 differences (10 non-zero) that are temporally ordered rather than
  independent. It is descriptive supporting evidence, not evidence of
  generalization or proof of significance.
- **Noisy later-period metrics.** Illicit prevalence in t=43-49 is low, so
  per-step F1 there is noisy and often exactly 0 for both models.
- **No replay** and no other adaptive mechanism was used; results describe
  this minimal online fine-tuning protocol only.

---

## L4 — Phase 7 (multi-seed robustness) limitations

These apply to every Phase 7 result in `docs/EXPERIMENTS.md`, in addition to
L2 (simulated feedback delay) and L3 (Phase 6 limitations).

- **Only five seeds.** The seeds (42, 123, 456, 789, 2024) were predefined.
  Five observations give a descriptive picture of direction, mean, spread and
  range, not a precise estimate of run-to-run variance. Improvement in 5/5
  seeds is not a proof of significance or of general reliability.
- **Single dataset.** All results are on the original Elliptic Bitcoin
  dataset with one fixed chronological split. No claim is made that they
  generalize to other datasets, other time periods or real-world deployments.
- **Single main adaptive configuration.** Every seed uses the configuration
  selected in Phase 6 (lr=0.001, grad_steps=1, k=1, no replay), which was
  chosen on five validation steps with a seed-42 model. Other adaptation
  settings, and replay, were not examined across seeds.
- **Temporally ordered, non-independent observations.** Per-step results
  within a seed are temporally ordered and not independent, and seeds share
  the same data and split. No pooled seed x time-step test was performed, and
  none should be read into the numbers.
- **Descriptive temporal pattern.** The concentration of improvement in
  t=35-42 and its absence in t=43-49 is an observed pattern only. It is not a
  causal explanation, and t=43 is not a proven regime boundary. Illicit
  prevalence in t=43-49 is low, so per-step F1 there is noisy.
- **Seed-42 initialization and reproducibility.** The original Phase 5 grid
  constructed each model before `train_gcn` reseeded, so the official seed-42
  checkpoint cannot be regenerated from a bare `seed=42`. A clean replicate
  gave different numbers (Static F1 0.440, Adaptive F1 0.537 vs the
  reference 0.510 / 0.562). The original Phase 6 result remains the official
  reference, and exact reproducibility of it is not claimed.
- **Threshold sensitivity.** The fixed threshold 0.898 was validation-tuned
  for the seed-42 model, and other seeds' own validation thresholds range from
  0.877 to 0.950. The secondary per-seed-threshold analysis agrees in
  direction (5/5 seeds improved) but is secondary and does not replace the
  primary fixed-threshold result.
- **Simulated feedback delay and persistent Adam state** (L2, L3) still apply
  to every seed.
- **Random Forest remains stronger.** Adaptive GraphSAGE averages F1 0.566
  against Random Forest's 0.790.
