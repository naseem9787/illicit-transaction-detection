"""Phase 2 training entry point.

Usage:
    python -m scripts.train_baselines --config config.yaml

Trains:
    - Logistic Regression, WITHOUT time_step (primary baseline A)
    - Logistic Regression, WITH time_step (secondary, clearly labeled)
    - Random Forest, WITHOUT time_step (primary baseline B, same feature
      set as the primary logistic regression for a fair comparison)

Saves:
    results/models/{name}.joblib
    results/metrics/predictions/{name}_{split}.npz  (y_true, y_prob, time_step)
    results/metrics/train_run_manifest.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines.logistic_regression import train_logistic_regression
from src.baselines.random_forest import train_random_forest
from src.data.dataset import get_labeled_split, load_node_table_with_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_baselines")


def save_predictions(model, name: str, splits: dict, metrics_dir: Path) -> None:
    pred_dir = metrics_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for split_name, (X, y, t) in splits.items():
        y_prob = model.predict_proba(X)[:, 1]
        np.savez_compressed(
            pred_dir / f"{name}_{split_name}.npz", y_true=y, y_prob=y_prob, time_step=t
        )


def main(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    project_root = Path(config_path).resolve().parent
    processed_dir = project_root / cfg["dataset"]["processed_dir"]
    models_dir = project_root / cfg["paths"]["models_dir"]
    metrics_dir = project_root / cfg["paths"]["metrics_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]

    node_table, feature_cols = load_node_table_with_features(processed_dir)

    manifest: dict = {"seed": seed, "feature_cols": feature_cols, "models": {}}

    # ---- primary feature set: f1..f165, no time_step ----
    X_train, y_train, t_train = get_labeled_split(node_table, feature_cols, "train", include_time_step=False)
    X_val, y_val, t_val = get_labeled_split(node_table, feature_cols, "val", include_time_step=False)
    X_test, y_test, t_test = get_labeled_split(node_table, feature_cols, "test", include_time_step=False)
    splits_no_tstep = {"train": (X_train, y_train, t_train), "val": (X_val, y_val, t_val), "test": (X_test, y_test, t_test)}
    logger.info(
        "Primary feature set (no time_step): train=%d val=%d test=%d, %d features",
        len(y_train), len(y_val), len(y_test), X_train.shape[1],
    )

    # ---- Logistic Regression, primary ----
    t0 = time.time()
    logreg = train_logistic_regression(X_train, y_train, seed)
    train_time = time.time() - t0
    joblib.dump(logreg, models_dir / "logreg_no_tstep.joblib")
    save_predictions(logreg, "logreg_no_tstep", splits_no_tstep, metrics_dir)
    manifest["models"]["logreg_no_tstep"] = {
        "type": "LogisticRegression",
        "includes_time_step": False,
        "class_weight": "balanced",
        "train_time_seconds": round(train_time, 3),
    }

    # ---- Logistic Regression, secondary (WITH time_step) ----
    X_train_t, y_train_t, t_train_t = get_labeled_split(node_table, feature_cols, "train", include_time_step=True)
    X_val_t, y_val_t, t_val_t = get_labeled_split(node_table, feature_cols, "val", include_time_step=True)
    X_test_t, y_test_t, t_test_t = get_labeled_split(node_table, feature_cols, "test", include_time_step=True)
    splits_with_tstep = {"train": (X_train_t, y_train_t, t_train_t), "val": (X_val_t, y_val_t, t_val_t), "test": (X_test_t, y_test_t, t_test_t)}

    t0 = time.time()
    logreg_t = train_logistic_regression(X_train_t, y_train_t, seed)
    train_time = time.time() - t0
    joblib.dump(logreg_t, models_dir / "logreg_with_tstep.joblib")
    save_predictions(logreg_t, "logreg_with_tstep", splits_with_tstep, metrics_dir)
    manifest["models"]["logreg_with_tstep"] = {
        "type": "LogisticRegression",
        "includes_time_step": True,
        "note": "secondary experiment only, not the primary baseline",
        "class_weight": "balanced",
        "train_time_seconds": round(train_time, 3),
    }

    # ---- Random Forest, primary (same feature set as primary logreg) ----
    t0 = time.time()
    rf, rf_selection = train_random_forest(X_train, y_train, X_val, y_val, seed)
    train_time = time.time() - t0
    joblib.dump(rf, models_dir / "random_forest_no_tstep.joblib")
    save_predictions(rf, "random_forest_no_tstep", splits_no_tstep, metrics_dir)
    manifest["models"]["random_forest_no_tstep"] = {
        "type": "RandomForestClassifier",
        "includes_time_step": False,
        "class_weight": "balanced",
        "train_time_seconds": round(train_time, 3),
        **rf_selection,
    }

    with open(metrics_dir / "train_run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info("Training complete. Manifest written to %s", metrics_dir / "train_run_manifest.json")
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    result = main(args.config)
    print(json.dumps(result, indent=2, default=str))
