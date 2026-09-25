"""Phase 7 evaluation: multi-seed robustness of Static -> Adaptive GraphSAGE.

Usage:
    python -m scripts.evaluate_multiseed_robustness --config config.yaml

Primary analysis: the five predeclared seeds [42, 123, 456, 789, 2024], all
evaluated at the SAME fixed Phase 6 threshold (0.898). Seed 42 is the
existing Phase 5/6 reference. The from-scratch seed-42 replicate is reported
separately as a continuity check and never silently replaces the reference.

Secondary sensitivity (clearly labeled, not the primary result): each seed's
own VALIDATION-F1 threshold, since 0.898 was tuned on the seed-42 model's
calibration. No threshold is ever selected with test labels.

No pooled seed x time-step statistical test is performed (those observations
are not independent). The primary robustness question is descriptive: how
often adaptive beats its matched static baseline across the five seeds.

Saves (Phase 7 files only):
    results/metrics/multiseed/per_seed_results.csv
    results/metrics/multiseed/per_seed_step_deltas.csv
    results/metrics/multiseed/multiseed_summary.json
    results/figures/multiseed_robustness.png
"""

from __future__ import annotations

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

from scripts.train_multiseed_robustness import FIXED_THRESHOLD, REFERENCE_SEED, SEEDS
from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.temporal_metrics import compute_metrics_by_time_step

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate_multiseed_robustness")

METRICS = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
PRE_STEPS = range(35, 43)
POST_STEPS = range(43, 50)


def load_npz(path: Path):
    d = np.load(path)
    return d["y_true"], d["y_prob"], d["time_step"]


def summarize(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}


def run_paths(root: Path, label: str) -> dict:
    """Locations of a run's static/adaptive test predictions and its own validation threshold."""
    m = root / "results" / "metrics"
    if label == "seed_42_reference":
        with open(m / "gnn_improvement_train_manifest.json") as f:
            own = json.load(f)["selected"]["selected_threshold_on_validation"]
        return {"static": m / "predictions" / "gnn_improvement_test.npz",
                "adaptive": m / "predictions" / "adaptive_graphsage_test.npz", "own_threshold": own}
    with open(m / "multiseed" / f"manifest_{label}.json") as f:
        own = json.load(f)["own_validation_threshold"]
    return {"static": m / "multiseed" / "predictions" / f"static_{label}_test.npz",
            "adaptive": m / "multiseed" / "predictions" / f"adaptive_{label}_test.npz", "own_threshold": own}


def evaluate_run(paths: dict, threshold: float) -> dict:
    """Static vs adaptive at a FIXED externally supplied threshold. Also checks
    the fairness invariants (identical population, identical t=35 predictions)."""
    sy, sp, st = load_npz(paths["static"])
    ay, ap, at = load_npz(paths["adaptive"])
    same_population = bool(np.array_equal(st, at) and np.array_equal(sy, ay))
    t35_identical = bool(np.array_equal(sp[st == 35], ap[at == 35]))
    sm = compute_binary_metrics(sy, sp, threshold)
    am = compute_binary_metrics(ay, ap, threshold)
    s_t = compute_metrics_by_time_step(sy, sp, st, threshold)
    a_t = compute_metrics_by_time_step(ay, ap, at, threshold)
    steps = s_t[["time_step"]].copy()
    steps["static_f1"] = pd.to_numeric(s_t["f1"], errors="coerce")
    steps["adaptive_f1"] = pd.to_numeric(a_t["f1"], errors="coerce")
    steps["delta_f1"] = steps["adaptive_f1"] - steps["static_f1"]
    pre = steps[steps["time_step"].isin(PRE_STEPS)]["delta_f1"].mean()
    post = steps[steps["time_step"].isin(POST_STEPS)]["delta_f1"].mean()
    return {"static": sm, "adaptive": am, "steps": steps, "n_test": int(len(sy)),
            "same_population": same_population, "t35_identical": t35_identical,
            "mean_delta_t35_42": float(pre), "mean_delta_t43_49": float(post)}


def aggregate(results: dict, labels: list[str]) -> dict:
    out = {"labels": labels}
    for name in METRICS:
        s = [results[l]["static"][name] for l in labels]
        a = [results[l]["adaptive"][name] for l in labels]
        d = [x - y for x, y in zip(a, s)]
        out[name] = {"static": summarize(s), "adaptive": summarize(a), "delta": summarize(d),
                     "n_adaptive_gt_static": int(sum(x > 0 for x in d)), "n_seeds": len(labels)}
    out["temporal"] = {
        "mean_delta_t35_42_across_seeds": summarize([results[l]["mean_delta_t35_42"] for l in labels]),
        "mean_delta_t43_49_across_seeds": summarize([results[l]["mean_delta_t43_49"] for l in labels]),
        "n_seeds_positive_t35_42": int(sum(results[l]["mean_delta_t35_42"] > 0 for l in labels)),
        "n_seeds_positive_t43_49": int(sum(results[l]["mean_delta_t43_49"] > 0 for l in labels)),
    }
    return out


def make_figure(table: pd.DataFrame, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(table))
    ax1.bar(x - 0.2, table["static_f1"], 0.4, label="Static", color="#2980b9")
    ax1.bar(x + 0.2, table["adaptive_f1"], 0.4, label="Adaptive (k=1)", color="#27ae60")
    ax1.axhline(0.790, color="gray", linestyle=":", label="Random Forest F1 (0.790)")
    ax1.set_xticks(x); ax1.set_xticklabels(table["seed"].astype(str)); ax1.set_ylim(0, 1)
    ax1.set_xlabel("seed"); ax1.set_ylabel("test F1"); ax1.set_title("Test F1 per seed (threshold 0.898)"); ax1.legend()
    w = 0.4
    ax2.bar(x - w / 2, table["mean_delta_t35_42"], w, label="t=35-42", color="#27ae60")
    ax2.bar(x + w / 2, table["mean_delta_t43_49"], w, label="t=43-49", color="#c0392b")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(table["seed"].astype(str))
    ax2.set_xlabel("seed"); ax2.set_ylabel("mean per-step delta F1"); ax2.set_title("Mean delta F1 by test period"); ax2.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def main(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    root = Path(config_path).resolve().parent
    out_dir = root / cfg["paths"]["metrics_dir"] / "multiseed"
    fig_dir = root / cfg["paths"]["figures_dir"]
    with open(root / cfg["paths"]["metrics_dir"] / "baseline_results.json") as f:
        rf_f1 = json.load(f)["primary_models"]["random_forest_no_tstep"]["test_metrics"]["f1"]

    label_of = {s: (f"seed_{s}" if s != REFERENCE_SEED else "seed_42_reference") for s in SEEDS}
    labels = [label_of[s] for s in SEEDS]
    all_labels = labels + ["seed_42_replicate"]
    paths = {l: run_paths(root, l) for l in all_labels}

    primary = {l: evaluate_run(paths[l], FIXED_THRESHOLD) for l in all_labels}
    own = {l: evaluate_run(paths[l], paths[l]["own_threshold"]) for l in all_labels}

    rows, step_rows = [], []
    for s in SEEDS:
        l = label_of[s]
        r = primary[l]
        rows.append({"seed": s, "label": l, "n_test": r["n_test"],
                     **{f"static_{m}": r["static"][m] for m in METRICS},
                     **{f"adaptive_{m}": r["adaptive"][m] for m in METRICS},
                     **{f"delta_{m}": r["adaptive"][m] - r["static"][m] for m in METRICS},
                     "mean_delta_t35_42": r["mean_delta_t35_42"], "mean_delta_t43_49": r["mean_delta_t43_49"],
                     "same_population": r["same_population"], "t35_identical": r["t35_identical"]})
        st = r["steps"].copy(); st.insert(0, "seed", s); step_rows.append(st)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "per_seed_results.csv", index=False)
    pd.concat(step_rows).to_csv(out_dir / "per_seed_step_deltas.csv", index=False)

    ref, rep = primary["seed_42_reference"], primary["seed_42_replicate"]
    continuity = {
        "note": "Phase 7 protocol from scratch for seed 42 vs the existing Phase 5/6 reference. "
                "Differences are expected because the Phase 5 checkpoint's initial weights came from the RNG state of the previous grid config.",
        "reference": {"static_f1": ref["static"]["f1"], "adaptive_f1": ref["adaptive"]["f1"],
                      "delta_f1": ref["adaptive"]["f1"] - ref["static"]["f1"]},
        "replicate": {"static_f1": rep["static"]["f1"], "adaptive_f1": rep["adaptive"]["f1"],
                      "delta_f1": rep["adaptive"]["f1"] - rep["static"]["f1"]},
        "static_f1_difference": rep["static"]["f1"] - ref["static"]["f1"],
        "adaptive_f1_difference": rep["adaptive"]["f1"] - ref["adaptive"]["f1"],
    }
    swap = [l if l != "seed_42_reference" else "seed_42_replicate" for l in labels]

    def row_of(r: dict) -> dict:
        return {"static": {m: r["static"][m] for m in METRICS}, "adaptive": {m: r["adaptive"][m] for m in METRICS},
                "delta": {m: r["adaptive"][m] - r["static"][m] for m in METRICS},
                "mean_delta_t35_42": r["mean_delta_t35_42"], "mean_delta_t43_49": r["mean_delta_t43_49"],
                "threshold": r["static"]["threshold"], "n_test": r["n_test"]}

    summary = {
        "seeds": SEEDS,
        "primary_threshold": FIXED_THRESHOLD,
        "per_seed_primary_fixed_threshold": {l: row_of(primary[l]) for l in labels},
        "seed_42_replicate_primary_fixed_threshold": row_of(primary["seed_42_replicate"]),
        "per_seed_secondary_own_validation_threshold": {l: row_of(own[l]) for l in all_labels},
        "primary_analysis_fixed_threshold": aggregate(primary, labels),
        "sensitivity_replicate_substituted_for_reference": aggregate(primary, swap),
        "sensitivity_own_validation_threshold_per_seed": {
            "thresholds": {l: paths[l]["own_threshold"] for l in labels},
            **aggregate(own, labels)},
        "continuity_seed_42": continuity,
        "random_forest_test_f1": rf_f1,
        "population_and_fairness_checks": {
            l: {"n_test": primary[l]["n_test"], "same_population": primary[l]["same_population"],
                "t35_identical": primary[l]["t35_identical"]} for l in all_labels},
    }
    with open(out_dir / "multiseed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    make_figure(table, fig_dir / "multiseed_robustness.png")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
    print("done")
