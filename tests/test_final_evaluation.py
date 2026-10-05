import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_final_evaluation as fe  # noqa: E402
from src.evaluation.metrics import compute_binary_metrics  # noqa: E402

METRICS_DIR = ROOT / "results" / "metrics"
FINAL_JSON = METRICS_DIR / "final" / "final_evaluation_results.json"

pytestmark = pytest.mark.skipif(not FINAL_JSON.exists(), reason="final evaluation not run")


def test_protocol_document_exists_and_names_all_finalists():
    text = (ROOT / "docs" / "FINAL_MODEL_PROTOCOL.md").read_text(encoding="utf-8")
    for name in ["Logistic Regression", "Random Forest", "Static GraphSAGE", "Adaptive GraphSAGE", "Adaptive HGB"]:
        assert name in text


def test_final_script_refuses_to_overwrite_existing_result():
    with pytest.raises(SystemExit):
        fe.main(str(ROOT / "config.yaml"))


def test_frozen_models_match_original_phase_results():
    table = {r["model"]: r for r in json.loads(FINAL_JSON.read_text())["table"]}
    base = json.loads((METRICS_DIR / "baseline_results.json").read_text())["primary_models"]
    sage = json.loads((METRICS_DIR / "adaptive_graphsage_results.json").read_text())
    assert table["Random Forest"]["f1"] == pytest.approx(base["random_forest_no_tstep"]["test_metrics"]["f1"])
    assert table["Logistic Regression"]["f1"] == pytest.approx(base["logreg_no_tstep"]["test_metrics"]["f1"])
    assert table["Static GraphSAGE"]["f1"] == pytest.approx(sage["static_graphsage_test_metrics"]["f1"])
    assert table["Adaptive GraphSAGE"]["f1"] == pytest.approx(sage["adaptive_graphsage_test_metrics"]["f1"])


def test_graphsage_models_share_one_frozen_threshold():
    table = {r["model"]: r for r in json.loads(FINAL_JSON.read_text())["table"]}
    assert table["Static GraphSAGE"]["threshold"] == table["Adaptive GraphSAGE"]["threshold"]


def test_hgb_static_and_adaptive_share_threshold_per_seed():
    df = pd.read_csv(METRICS_DIR / "final" / "adaptive_hgb_per_seed.csv")
    for seed, g in df.groupby("seed"):
        assert g["threshold"].nunique() == 1


def test_hgb_uses_five_predeclared_seeds_and_test_population():
    df = pd.read_csv(METRICS_DIR / "final" / "adaptive_hgb_per_seed.csv")
    assert sorted(df["seed"].unique()) == [42, 123, 456, 789, 2024]
    assert fe.TEST_STEPS == list(range(35, 50))
    assert fe.TRAIN == (1, 29) and fe.VAL == (30, 34)
    assert json.loads(FINAL_JSON.read_text())["n_test_labeled"] == 16670


def test_rescoring_stored_predictions_reproduces_metrics():
    d = np.load(METRICS_DIR / "predictions" / "random_forest_no_tstep_test.npz")
    thr = json.loads((METRICS_DIR / "baseline_results.json").read_text())["primary_models"]["random_forest_no_tstep"]["threshold"]
    m = compute_binary_metrics(d["y_true"], d["y_prob"], thr)
    assert m["n_samples"] == 16670 and m["n_illicit"] == 1083
