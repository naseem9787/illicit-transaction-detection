# DECISIONS.md

A running log of decisions that shape the project, each with the reasoning
and what would change if the reasoning turned out to be wrong. Phase-1
decisions only; will grow as later phases add their own.

---

### D1 — Label encoding: {illicit: 1, licit: 0, unknown: -1}

Raw file uses strings `'1'`/`'2'`/`'unknown'`. Re-encoded so "is this row
usable for supervised loss" is a single check (`label >= 0`) everywhere,
instead of re-deriving it from strings in every module. `unknown` is never
mapped to a class — it is excluded from `is_labeled` and therefore from any
loss/metric computation by construction (see `src/data/preprocessing.py`).

### D2 — Chronological split: train 1–29, val 30–34, test 35–49

The literature convention for this exact dataset (Weber et al. 2019 and
essentially all follow-up work) is train 1–34 / test 35–49. We kept the
35–49 test boundary unchanged for comparability with reported numbers, and
carved validation out of the tail of the training region (30–34) rather
than inventing a new boundary. See `DATA_AUDIT.md` section 9 for the
measured split sizes and the figure that supports this choice.

**What would change this**: if Phase 5's adaptive mechanism needs a longer
"burn-in" period of confirmed feedback before it can do anything useful, we
may need to push val/test later and accept a smaller effective test region
— would be revisited then, not assumed now.

### D3 — Graph representation: one independent snapshot per time_step, not one global graph

100% of edges connect same-time-step nodes (verified, see `DATA_AUDIT.md`
section 5.2), and this is re-checked at every pipeline run in
`loader.validate_dataset` (raises if violated). Given that, there is no
information lost by treating the 49 time steps as 49 independent graphs —
doing otherwise (e.g. one big block-diagonal graph) would just waste memory
without adding any cross-time edges to exploit.

### D4 — "Adaptive risk" must operate at the time-step-aggregate level, not per-transaction

Every `txId` is unique (a transaction never recurs), so there is no
persistent entity to accumulate "history" about at the individual-node
level, ruling out candidates A/B/D from the original brief in their literal
per-transaction form. Full reasoning and the surviving reframing in
`DATA_AUDIT.md` section 7. This is a structural fact about the dataset, not
a stylistic preference — confirmed directly from the data, not assumed.

### D5 — Processed feature cache stored as compressed `.npz`, not CSV

First attempt stored the full merged node table (features + metadata) as
one 664MB CSV. Switched to a small metadata CSV
(`data/processed/node_table_meta.csv`: txId, time_step, label, time_split,
is_labeled) plus a compressed float32 `.npz` for the 165-d feature matrix
(`data/processed/node_features.npz`), ~40MB total — same information,
~16x smaller on disk, and this sandbox has a tight disk quota. Row order is
identical between the two files; they realign by position.

### D6 — No PyTorch / PyTorch Geometric in Phase 1

This sandbox could not install PyTorch (disk quota exceeded on the CPU
wheel download). Phase 1's `graph_builder.py` therefore returns plain numpy
arrays with field names (`x`, `edge_index`, `y`) that match
`torch_geometric.data.Data` exactly, so wrapping them for Phase 3/4 is a
one-line `Data(**...)` call with zero logic duplicated — not a rewrite.
Phase 3 (GNN baseline) will need to run in an environment with PyTorch
installed (team laptops / Colab / similar), not necessarily this sandbox.

---

## Phase 2 decisions

### D7 — time_step excluded from primary baselines, tested separately

Ran a secondary Logistic Regression with `time_step` included specifically
to check whether it acts as a shortcut. It does: validation F1 looked fine
(0.757) but test F1 collapsed to 0.323, statistically indistinguishable
from the no-time_step model's test F1 (0.323). Confirms the primary
baselines are right to exclude it — see `docs/EXPERIMENTS.md`.

### D8 — Threshold selection: maximize F1 on validation only, freeze for test

Chosen over a fixed 0.5 threshold because with `class_weight="balanced"`
training, 0.5 is not a meaningful decision boundary for either model, and a
frozen literature-arbitrary threshold would make the val/test comparison
less interpretable than one actually tuned (on val only) for this task.

### D9 — Random Forest model selection: 3 fixed configs, not a grid search

Per the project brief's explicit instruction not to over-tune. 3 configs
(`max_depth` in {8, 16, None}, `n_estimators=300` fixed) selected by
validation PR-AUC. Total RF training time for all 3 configs is on the
order of single-digit seconds to low tens of seconds depending on the
machine (see `train_time_seconds` in `results/metrics/train_run_manifest.json`
for the figure from the most recent run) — model selection cost was not a
constraint here, the shallow search was a deliberate scope decision, not a
compute-forced one.
