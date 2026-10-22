"""Tests for reproducible bootstrap confidence intervals and expected-cost models."""

import json
from pathlib import Path
import numpy as np
import pytest
from sklearn.metrics import recall_score, roc_auc_score

from src.evaluation_stats import (
    compute_bootstrap_ci,
    compute_expected_cost,
    compute_break_even_ratio,
    evaluate_cost_model_across_thresholds,
    compute_all_reproducible_evaluation_stats,
    PROJECT_ROOT,
    RESULTS_DIR,
)


def test_bootstrap_ci_determinism():
    """Confirms bootstrap CI is 100% deterministic given a fixed random state."""
    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0])
    y_score = np.array([0.1, 0.8, 0.2, 0.7, 0.9, 0.3, 0.1, 0.6, 0.4, 0.2, 0.85, 0.15, 0.75, 0.25, 0.3])
    y_pred = (y_score >= 0.5).astype(int)

    ci1 = compute_bootstrap_ci(y_true, y_pred, metric="recall", n_bootstraps=500, random_state=42)
    ci2 = compute_bootstrap_ci(y_true, y_pred, metric="recall", n_bootstraps=500, random_state=42)

    assert ci1["ci_lower"] == ci2["ci_lower"]
    assert ci1["ci_upper"] == ci2["ci_upper"]
    assert ci1["point_estimate"] == ci2["point_estimate"]
    assert ci1["valid_replicates"] == ci2["valid_replicates"]
    assert ci1["ci_lower"] <= ci1["point_estimate"] <= ci1["ci_upper"]


def test_bootstrap_ci_single_class_resample_resilience():
    """Confirms procedure safely skips resamples with fewer than 2 classes for ROC-AUC."""
    # Highly imbalanced toy sample
    y_true = np.array([0] * 48 + [1, 1])
    y_score = np.linspace(0.01, 0.99, 50)

    # Should compute without raising ValueError from roc_auc_score
    result = compute_bootstrap_ci(y_true, y_score, metric="roc_auc", n_bootstraps=100, random_state=42)
    assert result["valid_replicates"] > 0
    assert result["skipped_resamples"] >= 0
    assert 0.0 <= result["ci_lower"] <= result["ci_upper"] <= 1.0


def test_bootstrap_ci_held_out_test_predictions():
    """Validates exact deterministic bootstrap bounds on the actual held-out test predictions."""
    test_csv = RESULTS_DIR / "test_predictions.csv"
    assert test_csv.exists(), f"Missing artifact: {test_csv}"

    stats = compute_all_reproducible_evaluation_stats(seed=42, n_bootstraps=2000)

    # 1. Recall at operational threshold 0.40
    rec_info = stats["bootstrap_confidence_intervals"]["recall_threshold_0_40"]
    assert pytest.approx(rec_info["point_estimate"], abs=1e-4) == 0.5849
    # Deterministic bounds for seed=42, 2000 replicates
    assert pytest.approx(rec_info["ci_lower"], abs=1e-3) == 0.4510
    assert pytest.approx(rec_info["ci_upper"], abs=1e-3) == 0.7174
    assert rec_info["ci_percentage_formatted"] == "[45.1%, 71.7%]"

    # 2. ROC-AUC
    auc_info = stats["bootstrap_confidence_intervals"]["roc_auc"]
    assert pytest.approx(auc_info["point_estimate"], abs=1e-4) == 0.7356
    assert pytest.approx(auc_info["ci_lower"], abs=1e-3) == 0.6524
    assert pytest.approx(auc_info["ci_upper"], abs=1e-3) == 0.8101


def test_expected_cost_exact_values():
    """Recalculates the expected-cost table directly from saved test confusion matrices.

    Verifies the reported values under stated 8:1 cost ratio:
    - Threshold 0.40: 22 FN, 88 FP -> 264 cost units
    - Threshold 0.50: 30 FN, 41 FP -> 281 cost units
    - Threshold 0.55: 39 FN, 23 FP -> 335 cost units
    """
    # 0.40 CM: [[359, 88], [22, 31]]
    cost_040 = compute_expected_cost([[359, 88], [22, 31]], cost_ratio=8.0, cost_alarm=1.0)
    assert cost_040["fn"] == 22
    assert cost_040["fp"] == 88
    assert cost_040["expected_loss"] == (22 * 8.0) + (88 * 1.0)  # 176 + 88 = 264.0

    # 0.50 CM: [[406, 41], [30, 23]]
    cost_050 = compute_expected_cost([[406, 41], [30, 23]], cost_ratio=8.0, cost_alarm=1.0)
    assert cost_050["fn"] == 30
    assert cost_050["fp"] == 41
    assert cost_050["expected_loss"] == (30 * 8.0) + (41 * 1.0)  # 240 + 41 = 281.0

    # 0.55 CM: [[424, 23], [39, 14]]
    cost_055 = compute_expected_cost([[424, 23], [39, 14]], cost_ratio=8.0, cost_alarm=1.0)
    assert cost_055["fn"] == 39
    assert cost_055["fp"] == 23
    assert cost_055["expected_loss"] == (39 * 8.0) + (23 * 1.0)  # 312 + 23 = 335.0


def test_break_even_ratio_logic():
    """Verifies break-even cost ratio where threshold 0.40 outperforms 0.50.

    Formula: 22*R + 88 = 30*R + 41 => 8*R = 47 => R = 47/8 = 5.875
    """
    be_ratio = compute_break_even_ratio(fn_a=22, fp_a=88, fn_b=30, fp_b=41)
    assert be_ratio == pytest.approx(5.875, abs=1e-4)

    # When R > 5.875 (e.g. R=6 or R=8), Loss(0.40) < Loss(0.50)
    loss_040_high = (22 * 6.0) + 88  # 132 + 88 = 220
    loss_050_high = (30 * 6.0) + 41  # 180 + 41 = 221
    assert loss_040_high < loss_050_high

    # When R < 5.875 (e.g. R=5), Loss(0.40) > Loss(0.50)
    loss_040_low = (22 * 5.0) + 88  # 110 + 88 = 198
    loss_050_low = (30 * 5.0) + 41  # 150 + 41 = 191
    assert loss_040_low > loss_050_low


def test_evaluation_uncertainty_artifact_matches_module():
    """Confirms results/evaluation_uncertainty.json is consistent with live module calculations."""
    artifact_path = RESULTS_DIR / "evaluation_uncertainty.json"
    assert artifact_path.exists(), f"Missing artifact: {artifact_path}"

    with open(artifact_path, "r", encoding="utf-8") as f:
        saved_data = json.load(f)

    live_data = compute_all_reproducible_evaluation_stats(seed=42, n_bootstraps=2000)

    assert saved_data["metadata"]["seed"] == live_data["metadata"]["seed"]
    assert saved_data["metadata"]["n_bootstraps"] == live_data["metadata"]["n_bootstraps"]

    # Check recall CI
    saved_rec = saved_data["bootstrap_confidence_intervals"]["recall_threshold_0_40"]
    live_rec = live_data["bootstrap_confidence_intervals"]["recall_threshold_0_40"]
    assert saved_rec["point_estimate"] == live_rec["point_estimate"]
    assert saved_rec["ci_lower"] == live_rec["ci_lower"]
    assert saved_rec["ci_upper"] == live_rec["ci_upper"]

    # Check expected cost
    saved_cost = saved_data["expected_cost_model"]["threshold_0_40"]["expected_cost_units"]
    live_cost = live_data["expected_cost_model"]["threshold_0_40"]["expected_cost_units"]
    assert saved_cost == live_cost == 264.0
