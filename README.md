# Illicit Transaction Detection Using Graph-Based Adaptive Learning

Elliptic Bitcoin dataset, illicit-transaction / AML-oriented detection.
Status: **Phase 1 (data pipeline) complete and verified.** GNN and adaptive
mechanism not yet implemented — see `docs/DECISIONS.md` D4 for why the
adaptive mechanism's design changed from the original brief once the data
was actually inspected.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Place the raw dataset at:
```
data/raw/elliptic_bitcoin_dataset/elliptic_txs_classes.csv
data/raw/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv
data/raw/elliptic_bitcoin_dataset/elliptic_txs_features.csv
```

## Phase 1 — data pipeline

Run the full pipeline (load → validate → preprocess → chronological split →
per-time-step graph construction → EDA figures):

```bash
python3 -m scripts.audit_dataset --config config.yaml
```

Run the Phase 1 data-pipeline tests (11 tests, all currently passing
against the real files):

```bash
python3 -m pytest tests/test_data.py -v
```

### Outputs

- `data/processed/node_table_meta.csv` — txId, time_step, label, time_split,
  is_labeled (one row per transaction)
- `data/processed/node_features.npz` — the 165-d feature matrix, row order
  matching `node_table_meta.csv`
- `data/processed/snapshot_summary.csv` — per-time-step node/edge/label counts
- `data/processed/audit_summary.json` — machine-readable run summary
  (validation report, split config, split counts, timing)
- `results/figures/class_distribution.png`
- `results/figures/illicit_rate_over_time.png`

## Phase 2 — classical ML baselines

Train, then evaluate (separate scripts so predictions are cached and
re-evaluation doesn't require retraining):

```bash
python3 -m scripts.train_baselines --config config.yaml
python3 -m scripts.evaluate_baselines --config config.yaml
```

### Outputs

- `results/models/*.joblib` — trained Logistic Regression / Random Forest models
- `results/metrics/predictions/*.npz` — raw probabilities per model/split
- `results/metrics/train_run_manifest.json` — model configs, seeds, timing
- `results/metrics/baseline_results.json` — val/test metrics, thresholds
- `results/metrics/temporal_*.csv` — per-time-step test metrics
- `results/figures/baseline_metric_comparison.png`, `f1_over_time_logreg.png`,
  `f1_over_time_random_forest.png`, `pr_curves.png`, `roc_curves.png`,
  `confusion_matrix_best.png`

Full results and interpretation in `docs/EXPERIMENTS.md`.

Run the Phase 2 baseline tests (5 tests; these read the processed cache
and Phase 2 outputs above, so run `train_baselines.py` first if
`data/processed/` or `results/` are missing):

```bash
python3 -m pytest tests/test_baselines.py -v
```

Run the full test suite (16 tests total, all currently passing against
the real dataset and the generated Phase 1/Phase 2 outputs):

```bash
python3 -m pytest tests/ -v
```

## Project structure

```
config.yaml                 all paths/split/seed config — no hard-coded paths in code
src/data/loader.py          load + validate raw CSVs
src/data/preprocessing.py   label encoding, chronological split, leakage checks
src/data/graph_builder.py   per-time-step snapshot construction (numpy, PyG-compatible field names)
src/data/dataset.py         load Phase 1 processed cache for ML consumption
src/baselines/              Logistic Regression, Random Forest
src/evaluation/metrics.py           binary classification metrics, threshold selection
src/evaluation/temporal_metrics.py  per-time-step metrics
scripts/audit_dataset.py    Phase 1 CLI entry point
scripts/train_baselines.py  Phase 2 training entry point
scripts/evaluate_baselines.py  Phase 2 evaluation entry point
tests/test_data.py          Phase 1 pytest suite, runs against the real dataset
tests/test_baselines.py     Phase 2 pytest suite, runs against the processed cache + Phase 2 outputs
docs/DATA_AUDIT.md          full structural audit of the dataset
docs/DECISIONS.md           design-decision log with rationale
docs/LIMITATIONS.md         dataset-level caveats that no amount of code can resolve
```

## Documentation

- `docs/DATA_AUDIT.md` — what's actually in the dataset, verified by code,
  including why some of the originally proposed adaptive features (raw
  amount, per-node history) are not computable from this data.
- `docs/DECISIONS.md` — every non-obvious design choice and why.
- `docs/EXPERIMENTS.md` — Phase 2 baseline results and interpretation
  (filled in; kept in sync with `results/metrics/baseline_results.json`).
- `docs/LIMITATIONS.md` — dataset-level caveats (e.g. unknown provenance of
  the released feature standardization) that aren't fixable in code.
- `docs/METHODOLOGY.md` — stub, to be filled in starting Phase 5.

## Next steps (not yet started)

Phase 3 (static GNN baseline — needs PyTorch, see `docs/DECISIONS.md` D6),
Phase 4 (adaptive mechanism design, informed by the literature check +
structural constraints already documented in `docs/DATA_AUDIT.md` and the
temporal collapse observed in `docs/EXPERIMENTS.md`).
