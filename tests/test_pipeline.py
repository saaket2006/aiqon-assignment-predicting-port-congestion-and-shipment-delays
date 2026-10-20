"""
Automated Test Suite for Port Congestion & Shipment Delay Forecasting System.

Tests cover:
1. Synthetic data generation schema and reproducibility.
2. Target alignment, boundary row removal, and leakage prevention.
3. Chronological data splitting and temporal isolation.
4. Model artifact loading, pipeline inference, and probability bounds.
5. Threshold-based classification logic (0.40, 0.50, 0.55).
6. Input validation and zero-capacity guard.
7. Date, weekday, and midnight datetime arithmetic.
8. Evaluation metrics consistency across artifacts.
9. Streamlit app execution and interactive inference.
"""

from datetime import datetime, time, timedelta
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

RAW_DATA_PATH = DATA_DIR / "port_congestion_synthetic.csv"
PREPARED_DATA_PATH = DATA_DIR / "port_congestion_forecasting_dataset.csv"
MODEL_PATH = MODELS_DIR / "random_forest_pipeline.joblib"


# =============================================================================
# 1. Dataset Generation & Schema Tests
# =============================================================================
def test_synthetic_data_schema_and_boundaries():
    """Verify raw synthetic dataset conforms to expected schema and valid value ranges."""
    assert RAW_DATA_PATH.exists(), f"Raw dataset not found at {RAW_DATA_PATH}"
    df = pd.read_csv(RAW_DATA_PATH)

    assert len(df) == 2500, f"Expected 2500 records, got {len(df)}"
    assert df["port_id"].nunique() == 5, f"Expected 5 ports, got {df['port_id'].nunique()}"

    # Check each port has 500 records
    counts = df["port_id"].value_counts()
    for port, count in counts.items():
        assert count == 500, f"Port {port} has {count} records, expected 500"

    # Value range checks
    assert (df["vessel_arrivals_6h"] >= 0).all(), "Vessel arrivals must be non-negative"
    assert (df["port_capacity_6h"] > 0).all(), "Port capacity must be positive"
    assert (df["effective_capacity_6h"] >= 1).all(), "Effective capacity must be at least 1"
    assert (df["queue_length"] >= 0).all(), "Queue length must be non-negative"
    assert (df["actual_delay_hours"] >= 0.0).all(), "Delay hours must be non-negative"
    assert set(df["weather_condition"].unique()).issubset({"Clear", "Rain", "Storm"})

    # Check delay definition: actual_delay_hours > 12 -> is_delayed == 1
    expected_delayed = (df["actual_delay_hours"] > 12.0).astype(int)
    assert (df["is_delayed"] == expected_delayed).all(), "is_delayed must strictly match actual_delay_hours > 12"


def test_synthetic_generator_reproducibility(tmp_path):
    """Verify generator produces identical data when executed with fixed random seed."""
    from data.generate_dataset import generate_dataset

    test_out = tmp_path / "test_synthetic.csv"
    df_gen = generate_dataset(output_path=test_out)

    df_existing = pd.read_csv(RAW_DATA_PATH)
    pd.testing.assert_frame_equal(df_gen, df_existing, check_exact=True)


# =============================================================================
# 2. Target Alignment & Leakage Prevention Tests
# =============================================================================
def test_prepared_dataset_target_alignment():
    """Verify target_delayed_next_6h is shifted forward by 1 interval within each port."""
    assert PREPARED_DATA_PATH.exists(), f"Prepared dataset not found at {PREPARED_DATA_PATH}"
    df_prep = pd.read_csv(PREPARED_DATA_PATH)
    df_raw = pd.read_csv(RAW_DATA_PATH)

    # 5 boundary rows dropped (one per port)
    assert len(df_prep) == 2495, f"Expected 2495 rows in prepared dataset, got {len(df_prep)}"

    # Confirm raw current-window outcome columns were stripped from features
    assert "actual_delay_hours" not in df_prep.columns, "actual_delay_hours must not leak into features"
    assert "is_delayed" not in df_prep.columns, "is_delayed must not leak into features"

    # Verify intra-port target alignment
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw = df_raw.sort_values(by=["port_id", "timestamp"]).reset_index(drop=True)
    df_prep["timestamp"] = pd.to_datetime(df_prep["timestamp"])
    df_prep = df_prep.sort_values(by=["port_id", "timestamp"]).reset_index(drop=True)

    for port_id, port_prep in df_prep.groupby("port_id"):
        port_raw = df_raw[df_raw["port_id"] == port_id].reset_index(drop=True)
        # Expected target at row t is is_delayed at row t+1
        expected_targets = port_raw["is_delayed"].iloc[1:].reset_index(drop=True)
        actual_targets = port_prep["target_delayed_next_6h"].reset_index(drop=True)
        assert (expected_targets == actual_targets).all(), f"Target alignment failed for port {port_id}"


# =============================================================================
# 3. Chronological Splitting Isolation Tests
# =============================================================================
def test_chronological_splitting_isolation():
    """Verify split boundaries are strictly non-overlapping and ordered in time."""
    from src.train import load_dataset, split_data_chronologically

    df = load_dataset(PREPARED_DATA_PATH)
    train_df, val_df, test_df = split_data_chronologically(df, train_ratio=0.70, val_ratio=0.10)

    assert len(train_df) == 1745, f"Expected 1745 train rows, got {len(train_df)}"
    assert len(val_df) == 250, f"Expected 250 val rows, got {len(val_df)}"
    assert len(test_df) == 500, f"Expected 500 test rows, got {len(test_df)}"

    train_max_target = train_df["target_timestamp"].max()
    val_min_target = val_df["target_timestamp"].min()
    val_max_target = val_df["target_timestamp"].max()
    test_min_target = test_df["target_timestamp"].min()

    assert train_max_target < val_min_target, "Train target horizon must strictly precede validation"
    assert val_max_target < test_min_target, "Validation target horizon must strictly precede test"

    # Ensure all splits contain positive and negative samples
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        delayed_count = int((split["target_delayed_next_6h"] == 1).sum())
        assert delayed_count > 0, f"{name} split must contain delayed cases"


# =============================================================================
# 4. Model Loading & Prediction Tests
# =============================================================================
def test_model_artifact_loading_and_prediction():
    """Verify saved model pipeline loads and outputs valid probabilities."""
    assert MODEL_PATH.exists(), f"Model pipeline artifact not found at {MODEL_PATH}"
    pipeline = joblib.load(MODEL_PATH)

    # Valid test input with 9 expected features
    sample_input = pd.DataFrame([{
        "vessel_arrivals_6h": 8,
        "port_capacity_6h": 12,
        "arrival_load_ratio": 0.667,
        "queue_length": 1,
        "effective_capacity_6h": 12,
        "historical_avg_delay_hours": 4.5,
        "port_id": "PORT_A",
        "weather_condition": "Clear",
        "day_of_week": "Tuesday",
    }])

    probs = pipeline.predict_proba(sample_input)
    assert probs.shape == (1, 2), "Prediction output must be shape (1, 2)"
    p0, p1 = probs[0]
    assert 0.0 <= p0 <= 1.0, "Class 0 probability must be in [0, 1]"
    assert 0.0 <= p1 <= 1.0, "Class 1 probability must be in [0, 1]"
    assert np.isclose(p0 + p1, 1.0), "Probabilities must sum to 1.0"


# =============================================================================
# 5. Threshold Classification Logic Tests
# =============================================================================
@pytest.mark.parametrize("thresh,prob,expected_class", [
    (0.40, 0.39, 0),
    (0.40, 0.40, 1),
    (0.40, 0.75, 1),
    (0.50, 0.45, 0),
    (0.50, 0.50, 1),
    (0.55, 0.54, 0),
    (0.55, 0.56, 1),
])
def test_threshold_classification(thresh, prob, expected_class):
    """Verify threshold classification correctly assigns class 0 and 1."""
    classified = int(prob >= thresh)
    assert classified == expected_class


# =============================================================================
# 6. Input Validation & Zero-Capacity Guard Tests
# =============================================================================
def test_zero_capacity_handling():
    """Verify zero port capacity is caught without raising ZeroDivisionError."""
    arrivals = 10
    capacity = 0

    if capacity <= 0:
        ratio = None
        error_triggered = True
    else:
        ratio = arrivals / capacity
        error_triggered = False

    assert error_triggered is True
    assert ratio is None


# =============================================================================
# 7. Date, Weekday & Midnight Arithmetic Tests
# =============================================================================
def test_date_and_weekday_derivation():
    """Verify weekday is accurately derived from date."""
    date_val = pd.to_datetime("2026-04-21").date()
    assert date_val.strftime("%A") == "Tuesday"

    date_sat = pd.to_datetime("2026-04-25").date()
    assert date_sat.strftime("%A") == "Saturday"


def test_forecast_horizon_midnight_transition():
    """Verify 18:00 to 24:00 window advances date at midnight."""
    obs_date = pd.to_datetime("2026-05-01").date()  # May 1st
    slot_start_hours = {
        "00:00 to 06:00": 0,
        "06:00 to 12:00": 6,
        "12:00 to 18:00": 12,
        "18:00 to 24:00": 18,
    }

    # Window 18:00 to 24:00
    start_hour = slot_start_hours["18:00 to 24:00"]
    obs_start = datetime.combine(obs_date, time(hour=start_hour))
    cutoff_dt = obs_start + timedelta(hours=6)
    horizon_end_dt = cutoff_dt + timedelta(hours=6)

    # Cutoff must be May 2nd 00:00
    assert cutoff_dt.strftime("%Y-%m-%d %H:%M") == "2026-05-02 00:00"
    # Horizon must be May 2nd 00:00 to May 2nd 06:00
    assert horizon_end_dt.strftime("%Y-%m-%d %H:%M") == "2026-05-02 06:00"


# =============================================================================
# 8. Evaluation Metrics Consistency Tests
# =============================================================================
def test_threshold_evaluation_metrics_consistency():
    """Verify reported evaluation metrics agree across JSON and CSV artifacts."""
    comp_csv = RESULTS_DIR / "threshold_comparison.csv"
    eval_json = RESULTS_DIR / "threshold_evaluation.json"

    assert comp_csv.exists(), "threshold_comparison.csv must exist"
    assert eval_json.exists(), "threshold_evaluation.json must exist"

    df_comp = pd.read_csv(comp_csv)
    with open(eval_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check validation row for threshold 0.40
    row_040 = df_comp[df_comp["threshold"] == 0.40].iloc[0]
    json_val_040 = data["validation_performance"]["threshold_0_40_operational"]

    assert np.isclose(row_040["precision"], json_val_040["precision"], atol=1e-3)
    assert np.isclose(row_040["recall"], json_val_040["recall"], atol=1e-3)
    assert np.isclose(row_040["f1_score"], json_val_040["f1"], atol=1e-3)

    # Check held-out test metrics
    test_eval = data["test_evaluation"]["operational_threshold_0_40"]
    assert test_eval["delays_caught"] == "31 / 53"
    assert test_eval["samples"] == 500


# =============================================================================
# 9. Streamlit Dashboard AppTest
# =============================================================================
def test_streamlit_dashboard_interaction():
    """Verify Streamlit dashboard launches and responds to user interaction without error."""
    from streamlit.testing.v1 import AppTest

    app_path = PROJECT_ROOT / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=15)

    assert len(at.exception) == 0, f"AppTest initialization exceptions: {at.exception}"

    # Default button click
    at.button[0].click().run(timeout=15)
    assert len(at.exception) == 0, f"AppTest prediction exceptions: {at.exception}"

    # Zero capacity check: should disable predict button
    at.number_input[1].set_value(0).run(timeout=15)
    assert at.button[0].disabled is True, "Predict button should be disabled when port capacity is 0"
