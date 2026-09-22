# DATA_AUDIT.md

Elliptic Bitcoin Dataset — Structural Audit

Source file: `archive.zip` → `elliptic_bitcoin_dataset/` containing
`elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`, `elliptic_txs_features.csv`.
All numbers below were produced by directly loading and inspecting these three
files (see `scripts used` at the bottom). Nothing here is taken from memory of
the original paper.

---

## 1. File-level facts

| file | rows (excl. header where present) | columns | header row? |
|---|---|---|---|
| `elliptic_txs_classes.csv` | 203,769 | 2 (`txId`, `class`) | yes |
| `elliptic_txs_edgelist.csv` | 234,355 | 2 (`txId1`, `txId2`) | yes |
| `elliptic_txs_features.csv` | 203,769 | 167 | **no** |

`elliptic_txs_features.csv` has no header row and no documented column names.
Column 0 = `txId`, column 1 = `time_step`, columns 2–166 = **165 anonymized
numeric features**. So "166 features" is only correct if the time step is
counted as one of them (1 time_step + 165 anonymized = 166 non-ID columns).
We use `time_step` + `f1..f165` as the naming convention going forward and
will **not** claim 166 "features" without qualifying that one of them is the
time index, not a transaction attribute.

## 2. Labels (`elliptic_txs_classes.csv`)

- `class` column is stored as **string**, values `'1'`, `'2'`, `'unknown'` — not
  a numeric/NaN encoding. Must be explicitly mapped during preprocessing.
- Distribution (exact, matches the commonly cited numbers):

  | label | meaning | count |
  |---|---|---|
  | unknown | unlabeled | 157,205 |
  | 2 | licit | 42,019 |
  | 1 | illicit | 4,545 |

- No nulls. No duplicate `txId` rows (203,769 unique IDs, one row per ID).

## 3. Edges (`elliptic_txs_edgelist.csv`)

- 234,355 directed edges, `txId1 → txId2`. No nulls, **zero duplicate edge
  rows, zero self-loops**, zero mutual pairs (no edge has its reverse also
  present).
- Out-degree: min 1, median 1, mean 1.41, max 472 (166,345 / 203,769 nodes
  have ≥1 outgoing edge). In-degree: min 1, median 1, mean 1.58, max 284
  (148,447 / 203,769 nodes have ≥1 incoming edge). Every node in `classes` is
  reachable by at least one edge end (0 fully isolated nodes), but a large
  fraction have only in- or only out-edges, not both.

## 4. Feature matrix (`elliptic_txs_features.csv`)

- Shape confirmed: `(203769, 167)`.
- `txId` set is **identical** across `features` and `classes` (no missing or
  extra IDs either direction).
- `time_step` ranges 1–49 inclusive, all 49 present, no gaps. Row counts per
  time step range from 1,089 (t=27) to 7,880 (t=1) — snapshot sizes are
  uneven, not fixed.
- **Zero missing values** anywhere in the 165 feature columns.
- **All 165 feature columns have mean ≈ 0 and std ≈ 1** (verified per-column:
  every single column's std falls in [0.999998, 1.000003]). None are
  integer-valued, none are non-negative. This is definitive: the released
  feature file has already been **globally standardized** by the dataset
  authors. There is no raw BTC amount, fee, count, or any other
  human-interpretable raw value in the file — everything is a pre-transformed,
  anonymized continuous variable.

## 5. Two structural findings that affect the whole project design

### 5.1 Features are pre-standardized — "raw amount" does not exist in this data

Every one of the 165 feature columns is already z-scored (mean 0, std 1)
before we received it. There is no column that is plausibly a raw transaction
amount, fee, or count — all such information, if originally present, has been
folded into an anonymized, already-normalized representation.

**Consequence for Rule 2 (candidate A — "amount anomaly"):** we cannot compute
a currency-based anomaly ("this transaction moved an unusually large amount
of BTC") because we don't have BTC amounts. What we *can* do is compute a
**feature-space deviation** (e.g. Mahalanobis-style distance, or per-time-step
z-score-of-the-z-scores) of a transaction's feature vector relative to the
distribution of other transactions at or before its time step. This is a
legitimate, computable signal — but it is a materially weaker and differently
-scoped claim than "amount abnormality," and should be named accordingly
(e.g. "feature-space deviation score," not "amount anomaly") so the report
doesn't overstate what's being measured.

One caveat worth documenting honestly in `docs/LIMITATIONS.md` (see L1
there): we don't know
whether the *global* standardization statistics (mean/std) used by the
dataset authors were computed only from early time steps or from the full
49-step dataset. If the latter, the released feature values themselves carry
a small amount of dataset-wide (technically "future") information baked in at
the source, before we ever touch it. This is a known property of the public
release, not something we introduced — but it should be named as a caveat on
any claim of a perfectly leakage-free pipeline, since it isn't fully
correctable without the original unnormalized data.

### 5.2 Edges never cross time steps — and node identity never recurs

Checked directly: **100% of the 234,355 edges connect two nodes with the
identical `time_step` value.** Zero edges cross time steps, zero go backward.
The graph is not one connected temporal graph — it is a **disjoint union of
49 independent snapshot graphs**, one per time step.

Additionally, every `txId` is unique across the whole dataset — a transaction
node exists at exactly one time step and never recurs. There is no persistent
"entity" (address, wallet, account) identifier in this dataset that would let
us track the same real-world actor's behavior over multiple time steps.

**Consequence for Rule 2 (candidates B and D — "frequency abnormality
relative to entity's historical behavior" and "historical
feedback/previous confirmed illicit evidence" for the same node):** these,
as literally stated in the prompt, assume a recurring entity we accumulate
history about. That entity does not exist at the transaction-node level in
this dataset — each node is a one-time event, and its graph neighbors are
by construction all at the *same* time step as itself (never earlier). So:

- "Has this transaction's own node been unusually active lately?" — **not
  answerable**, a transaction node has no past occurrences of itself.
- "What is this node's neighbor's historical confirmed label?" **as a
  same-node-neighbor lookup — not directly answerable either**, since a
  node's graph-neighbors never precede it in time (they're all in its own
  snapshot). Any usable neighbor signal must come from confirmed labels
  *within the current time step's already-revealed portion* (if we're
  willing to assume partial in-snapshot label availability) or must be
  redefined at a coarser level (see below).

This is the single biggest design implication of the whole audit: the
"adaptive risk that updates as new temporal evidence arrives" from the
project brief cannot be implemented as literal per-transaction history
tracking, because the dataset doesn't support it structurally. It has to be
reconceived at one of these levels, which is a decision for Phase 5, not
this document — flagging now so it can be discussed before any code is
written:

1. **Time-step-aggregate level** — maintain a running, time-indexed summary
   (e.g. observed illicit rate, feature-distribution drift) computed only
   from time steps `< t` (or `≤ t` with a documented feedback delay), and
   feed that as a shared prior/feature to all transactions in snapshot `t`.
   This is fully computable from what we have and keeps causality clean.
2. **Structural-similarity level** — compare a new snapshot's local
   subgraph/feature patterns to patterns from previous snapshots (embedding
   similarity), without needing persistent node identity.
3. **Model-state level** — let the GNN's own parameters/hidden state evolve
   across snapshots (closer to how EvolveGCN-style temporal-GNN literature
   handles this dataset), rather than tracking a hand-crafted per-node risk
   score across time.

Recency (candidate E) survives in modified form: exponential decay can weight
*time-step-level* aggregate statistics by how long ago they were observed,
which is well-defined even without persistent node identity.

## 6. Label distribution over time (illicit rate is highly non-stationary)

Using only labeled (non-`unknown`) transactions, illicit rate per time step
swings from below 0.5% (t=45, t=46) to over 35% (t=13). This is a large,
genuine regime shift over the 49 steps, not noise — it strongly supports
using a chronological split (Rule 3) and evaluating performance per-time-step
rather than with a single aggregate number, since a model's behavior clearly
cannot be assumed stationary across this range. (Full per-time-step table
generated during this audit; will go into the EDA notebook as a figure
in Phase 2 rather than reproduced here.)

Fraction of each time step that is labeled at all also varies (11%–43%), so
"how much supervision is available" is itself non-stationary over time —
relevant when we design the feedback-delay assumption in Phase 5.

## 7. What Rule 2's candidate features actually reduce to, given the above

| candidate (from prompt) | literally computable? | notes |
|---|---|---|
| A. Amount abnormality | **No**, as stated (no raw amount exists) | Redefine as feature-space deviation score, computed from `f1..f165` only, comparing a transaction to its time-local (or ≤t) cohort. |
| B. Frequency abnormality | **No**, as stated (no recurring entity) | Only computable at aggregate level (e.g. snapshot size, in/out-degree within the snapshot) — not "this node's own history." |
| C. Neighbor risk | **Partially** | Neighbors exist only within the same snapshot; usable only if we accept using in-snapshot partially-revealed labels, or redefine as neighbor feature-similarity rather than neighbor-label lookup. Needs explicit design decision + leakage argument in Phase 5. |
| D. Historical feedback | **No**, as literally stated (no recurring entity) | Only computable as an aggregate/global prior updated over time steps, not per-transaction. |
| E. Recency | **Yes**, in modified form | Exponential decay over *time-step-aggregate* statistics, not per-node history. |

None of candidates A/B/D survive in their literal per-transaction form. This
isn't a reason to abandon the "adaptive" idea — it's a reason to redefine it
at the snapshot/aggregate level, which is actually a *more* defensible and
more literature-consistent framing for this dataset (this is closer to how
published temporal-GNN work on Elliptic — e.g. EvolveGCN — treats it) than
per-node history would have been. This will be the starting point for the
literature check in Phase 5, before any algorithm is finalized.

## 8. Column-group structure (documented in the literature, not independently
verifiable from this file since there are no column headers)

The original Elliptic dataset paper describes the 94 "local" features
(including the time step) and 72 "aggregated" (one-hop neighbor) features. If
we adopt that convention, in our indexing that would map to `time_step` +
`f1..f93` = local (94 total) and `f94..f165` = aggregated (72 total). We
flag explicitly that **this mapping is asserted by external documentation,
not verified against this file directly** — the raw CSV carries no column
names, so we cannot confirm the boundary index independently. Any analysis
that depends on this split (e.g. "local vs. aggregated feature importance")
should note this as an assumption, not a verified fact.

## 9. Split boundary — resolved

Confirmed against literature (Weber et al. 2019, the paper that released
this dataset, and essentially every follow-up paper found in a search):
the standard chronological split is **time steps 1–34 for training, 35–49
for testing** (a 70:30 split). We keep the 35–49 test boundary unchanged
for comparability with reported numbers on this dataset, and carve
validation out of the tail of the training region rather than inventing a
new boundary:

| split | time steps | rationale |
|---|---|---|
| train | 1–29 | tail of literature's training region, held out for val |
| val | 30–34 | last 5 steps of literature's training region |
| test | 35–49 | unchanged from literature convention |

`results/figures/illicit_rate_over_time.png` (generated in Phase 1)
visually confirms this boundary makes sense: the test region (35–49) is
visibly calmer and less volatile than the train region, consistent with
the documented post-shutdown quiet period in this dataset — not an
arbitrary cut.

Measured split sizes (from `scripts/audit_dataset.py`, this run):

| split | total nodes | labeled nodes | illicit (of labeled) | illicit rate |
|---|---|---|---|---|
| train (1–29) | 120,804 | 26,381 | 2,871 | 10.88% |
| val (30–34) | 15,461 | 3,513 | 591 | 16.82% |
| test (35–49) | 67,504 | 16,670 | 1,083 | 6.50% |

Note the illicit rate is not stable across splits (10.9% → 16.8% → 6.5%) —
expected given section 6, but worth remembering when interpreting any
single aggregate metric: per-time-step evaluation (Phase 7 onward) will
matter more than a single train/val/test number.

## 10. Open items still to resolve in Phase 5 (not blocking Phase 2/3)

- Feedback-delay assumption (how many time steps after a transaction until
  its label is "revealed" to the adaptive mechanism) — currently
  undecided; must be documented explicitly once chosen.
- Final wording/scope of the "adaptive risk" mechanism given section 7 above.

## 11. Scripts used for this audit

Ad hoc pandas scripts run against the three raw CSVs (row/column counts,
dtype checks, null checks, duplicate/self-loop checks, ID-set comparisons,
per-column mean/std, time_step crosstabs, edge time-step-crossing check).
Will be formalized into `src/data/loader.py` validation functions in Phase 1
implementation so these checks are re-run automatically on every pipeline run
rather than living only in this one-off audit.
