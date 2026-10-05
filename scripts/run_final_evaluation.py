"""Step 5: final consolidated evaluation on test t=35-49.

Usage:
    python -m scripts.run_final_evaluation --config config.yaml

Follows docs/FINAL_MODEL_PROTOCOL.md. Nothing is tuned here.

  * Logistic Regression, Random Forest, Static/Adaptive GraphSAGE: frozen
    Phase 3-7 test predictions are re-read and re-scored at their frozen
    validation thresholds (no refit, no re-selection).
  * Static HGB / Adaptive HGB: the only NEW test evaluation. Five seeds,
    threshold = max-F1 on validation (t=30-34) of the static model, shared by
    both conditions. Refuses to overwrite an existing result file.

Saves (new files only):
    results/metrics/final/final_evaluation_results.json
    results/metrics/final/final_comparison_table.csv
    results/metrics/final/adaptive_hgb_per_seed.csv
    results/metrics/final/temporal_final_f1.csv
    results/figures/final_comparison.png
    results/figures/final_f1_over_time.png
"""

from __future__ import annotations

import copy
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import load_node_table_with_features
from src.evaluation.metrics import compute_binary_metrics, select_threshold_maximizing_f1
from src.evaluation.temporal_metrics import compute_metrics_by_time_step
from src.training.adaptive_tree import (
    ADAPT_SEEDS, FEEDBACK_DELAY_K, ITERS_PER_UPDATE, compute_fixed_sample_weights,
    fit_base_model, pooled_predictions, run_adaptive_tree_walk,
)
from src.training.graph_features import HGB_BASE_CONFIG
from src.training.rolling_origin import get_labeled_by_time_range

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_final_evaluation")

TRAIN = (1, 29)
VAL = (30, 34)
TEST_STEPS = list(range(35, 50))
METRICS = ["precision", "recall", "f1", "roc_auc", "pr_auc"]


def load_npz(metrics_dir: Path, name: str):
    d = np.load(metrics_dir / "predictions" / name)
    return d["y_true"], d["y_prob"], d["time_step"]


def score(y, p, t, threshold):
    return compute_binary_metrics(y, p, threshold), compute_metrics_by_time_step(y, p, t, threshold)


def hgb_final(node_table, feature_cols):
    X_train, y_train = get_labeled_by_time_range(node_table, feature_cols, *TRAIN)
    X_val, y_val = get_labeled_by_time_range(node_table, feature_cols, *VAL)
    test_snaps = {t: get_labeled_by_time_range(node_table, feature_cols, t, t) for t in TEST_STEPS}
    rows, step_rows = [], []
    for seed in ADAPT_SEEDS:
        cw = compute_fixed_sample_weights(y_train)
        base = fit_base_model(X_train, y_train, seed, cw)
        threshold = select_threshold_maximizing_f1(y_val, base.predict_proba(X_val)[:, 1])

        static = copy.deepcopy(base)
        ys, ps, ts = [], [], []
        for t in TEST_STEPS:
            X_t, y_t = test_snaps[t]
            ys.append(y_t)
            ps.append(static.predict_proba(X_t)[:, 1])
            ts.append(np.full(len(y_t), t))
        ys, ps, ts = map(np.concatenate, (ys, ps, ts))

        walk = run_adaptive_tree_walk(copy.deepcopy(base), TEST_STEPS, test_snaps, cw)
        ya, pa = pooled_predictions(walk)
        assert np.array_equal(ys, ya), "static/adaptive populations differ"

        for cond, p in (("static", ps), ("adaptive", pa)):
            m, tdf = score(ys, p, ts, threshold)
            rows.append({"seed": seed, "condition": cond, "threshold": threshold, **{k: m[k] for k in METRICS}})
            for _, r in tdf.iterrows():
                step_rows.append({"seed": seed, "condition": cond, "time_step": int(r["time_step"]), "f1": float(r["f1"])})
        r_s, r_a = rows[-2], rows[-1]
        logger.info("[HGB seed=%d] thr=%.3f static F1=%.4f adaptive F1=%.4f | PR-AUC %.4f -> %.4f",
                    seed, threshold, r_s["f1"], r_a["f1"], r_s["pr_auc"], r_a["pr_auc"])
    return pd.DataFrame(rows), pd.DataFrame(step_rows)


def main(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    root = Path(config_path).resolve().parent
    metrics_dir = root / cfg["paths"]["metrics_dir"]
    figures_dir = root / cfg["paths"]["figures_dir"]
    out_dir = metrics_dir / "final"
    out_json = out_dir / "final_evaluation_results.json"
    if out_json.exists():
        raise SystemExit(f"{out_json} already exists; the final evaluation is run once (see FINAL_MODEL_PROTOCOL.md).")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- frozen models: re-score stored predictions at frozen thresholds ----
    with open(metrics_dir / "baseline_results.json") as f:
        base = json.load(f)["primary_models"]
    with open(metrics_dir / "gnn_improvement_train_manifest.json") as f:
        sage_thr = json.load(f)["selected"]["selected_threshold_on_validation"]
    frozen = {
        "Logistic Regression": ("logreg_no_tstep_test.npz", base["logreg_no_tstep"]["threshold"]),
        "Random Forest": ("random_forest_no_tstep_test.npz", base["random_forest_no_tstep"]["threshold"]),
        "Static GraphSAGE": ("gnn_improvement_test.npz", sage_thr),
        "Adaptive GraphSAGE": ("adaptive_graphsage_test.npz", sage_thr),
    }
    table, temporal = [], {}
    for name, (fn, thr) in frozen.items():
        y, p, t = load_npz(metrics_dir, fn)
        m, tdf = score(y, p, t, thr)
        table.append({"model": name, "threshold": thr, "n_runs": 1, **{k: m[k] for k in METRICS},
                      **{k + "_std": np.nan for k in METRICS}})
        temporal[name] = tdf.set_index("time_step")["f1"]
        logger.info("[frozen] %s F1=%.4f PR-AUC=%.4f", name, m["f1"], m["pr_auc"])

    # ---- five-seed GraphSAGE robustness (Phase 7, already computed) ----
    seeds = pd.read_csv(metrics_dir / "multiseed" / "per_seed_results.csv")
    seed_rows = {
        cond: {k: (float(seeds[f"{cond}_{k}"].mean()), float(seeds[f"{cond}_{k}"].std())) for k in METRICS}
        for cond in ("static", "adaptive")
    }

    # ---- new: HGB ----
    node_table, feature_cols = load_node_table_with_features(root / cfg["dataset"]["processed_dir"])
    hgb, hgb_steps = hgb_final(node_table, feature_cols)
    hgb.to_csv(out_dir / "adaptive_hgb_per_seed.csv", index=False)
    for cond, label in (("static", "Static HGB (context)"), ("adaptive", "Adaptive HGB (context)")):
        sub = hgb[hgb["condition"] == cond]
        table.append({"model": label, "threshold": float(sub["threshold"].mean()), "n_runs": len(sub),
                      **{k: float(sub[k].mean()) for k in METRICS},
                      **{k + "_std": float(sub[k].std()) for k in METRICS}})
        temporal[label] = hgb_steps[hgb_steps["condition"] == cond].groupby("time_step")["f1"].mean()
    for cond, label in (("static", "Static GraphSAGE (5-seed)"), ("adaptive", "Adaptive GraphSAGE (5-seed)")):
        table.append({"model": label, "threshold": sage_thr, "n_runs": 5,
                      **{k: seed_rows[cond][k][0] for k in METRICS},
                      **{k + "_std": seed_rows[cond][k][1] for k in METRICS}})

    tdf = pd.DataFrame(table)
    tdf.to_csv(out_dir / "final_comparison_table.csv", index=False)
    pd.DataFrame(temporal).rename_axis("time_step").to_csv(out_dir / "temporal_final_f1.csv")

    # ---- figures ----
    main_models = ["Logistic Regression", "Random Forest", "Static GraphSAGE", "Adaptive GraphSAGE",
                   "Static HGB (context)", "Adaptive HGB (context)"]
    sub = tdf.set_index("model").loc[main_models]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(METRICS))
    w = 0.13
    for i, name in enumerate(main_models):
        ax.bar(x + (i - 2.5) * w, [sub.loc[name, k] for k in METRICS], w, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"])
    ax.set_ylim(0, 1)
    ax.set_title("Final test comparison (t=35-49, validation-frozen thresholds)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "final_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for name in main_models:
        s = temporal[name]
        ax.plot(s.index, s.values, marker="o", markersize=3, label=name)
    ax.set_xlabel("time_step")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_title("Per-step test F1, all finalists")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "final_f1_over_time.png", dpi=150)
    plt.close(fig)

    ps = hgb[hgb["condition"] == "static"].set_index("seed")
    pa = hgb[hgb["condition"] == "adaptive"].set_index("seed")
    out = {
        "protocol": "docs/FINAL_MODEL_PROTOCOL.md",
        "test_range": [35, 49], "n_test_labeled": 16670,
        "hgb_config": HGB_BASE_CONFIG, "iters_per_update": ITERS_PER_UPDATE, "feedback_delay_k": FEEDBACK_DELAY_K,
        "table": table,
        "adaptive_hgb_vs_static_hgb": {
            "delta_f1_mean": float((pa["f1"] - ps["f1"]).mean()),
            "delta_pr_auc_mean": float((pa["pr_auc"] - ps["pr_auc"]).mean()),
            "n_seeds_f1_improved": int(((pa["f1"] - ps["f1"]) > 0).sum()),
            "n_seeds_pr_auc_improved": int(((pa["pr_auc"] - ps["pr_auc"]) > 0).sum()),
            "n_seeds": len(ps),
        },
        "note": "Frozen finalists re-scored from stored predictions at frozen validation thresholds; "
                "HGB is the only new test evaluation.",
    }
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    res = main(ap.parse_args().config)
    print(json.dumps(res["adaptive_hgb_vs_static_hgb"], indent=2))
