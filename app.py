"""
Port Congestion & Delay Risk Forecaster - Streamlit Web Application

This app provides a beginner-friendly prediction interface for the port congestion
forecasting model. It allows an operator or analyst to input conditions observed
during the current 6-hour interval and forecast whether the port will experience
a delay (>12h) during the next 6-hour interval.

Run with:
    streamlit run app.py
"""

from datetime import datetime, time, timedelta
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st


# -----------------------------------------------------------------------------
# Configuration and Asset Loading
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Port Congestion Forecaster",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_pipeline.joblib"
FEAT_IMP_PATH = PROJECT_ROOT / "results" / "feature_importance.csv"
EVAL_JSON_PATH = PROJECT_ROOT / "results" / "threshold_evaluation.json"

# Training set benchmark averages for context comparison
BENCHMARKS = {
    "queue_length": 1.04,
    "historical_avg_delay_hours": 6.43,
    "arrival_load_ratio": 0.74,
    "vessel_arrivals_6h": 8.96,
    "effective_capacity_6h": 11.52,
    "port_capacity_6h": 12.20,
}


@st.cache_resource
def load_model_pipeline():
    """Loads the trained Random Forest model and preprocessing pipeline once."""
    if not MODEL_PATH.exists():
        st.error(f"Trained model not found at: {MODEL_PATH}. Run 'python src/train.py' first.")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_importances():
    """Loads the global Random Forest feature importances."""
    if FEAT_IMP_PATH.exists():
        return pd.read_csv(FEAT_IMP_PATH)
    return None


@st.cache_data
def load_evaluation_summary():
    """Loads evaluation comparison metrics if available."""
    if EVAL_JSON_PATH.exists():
        with open(EVAL_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


pipeline = load_model_pipeline()
feat_imp_df = load_feature_importances()
eval_summary = load_evaluation_summary()


def show_table(df, hide_index=True):
    """Renders a dataframe cleanly across Streamlit versions with stretch width."""
    try:
        st.dataframe(df, hide_index=hide_index, width="stretch")
    except TypeError:
        st.dataframe(df, hide_index=hide_index, use_container_width=True)


# -----------------------------------------------------------------------------
# Header and System Overview
# -----------------------------------------------------------------------------
st.title("🚢 Port Congestion & Delay Risk Forecaster")
st.markdown(
    """
    **Purpose:** Forecast whether a simulated commercial port will experience a **delay (>12 hours)**
    during the **next six-hour interval**, using operating conditions observed at the end of the current six-hour interval.
    """
)

with st.expander("ℹ️ How This Forecasting System Works", expanded=False):
    st.markdown(
        """
        - **Observation Window:** The current six-hour operational window (e.g., 06:00 to 12:00) during which port activities are observed.
        - **Forecast Cutoff:** The closing instant of the observation window (e.g., 12:00) at which the prediction is issued.
        - **Forecast Horizon (Next 6h):** The immediate following six-hour operational interval (e.g., 12:00 to 18:00) for which delay risk is predicted.
        - **Target Definition:** Class 1 indicates port delay > 12 hours during the forecast horizon; Class 0 indicates on-time (<= 12 hours).
        - **No Retrospective Leakage:** Current interval delays, future outcomes, and target fields are strictly excluded from predictors.
        - **Operating Threshold:** Defaults to **0.40** (the operational higher-recall threshold selected on validation data to prioritize catching bottlenecks early).
        - **Uncalibrated Probability:** The percentage represents the raw fraction of decision trees voting for a delay; it is uncalibrated and should not be treated as a validated real-world probability.
        """
    )

st.markdown("---")


# -----------------------------------------------------------------------------
# Preset Scenario Helper
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Preset Scenarios")
st.sidebar.markdown("Quickly load typical operational conditions:")

scenario = st.sidebar.radio(
    "Choose a preset:",
    ["Custom Input", "Normal Operations", "Elevated Delay Risk", "Borderline Case"],
    index=0,
)

# Default values based on scenario
if scenario == "Normal Operations":
    default_port = "PORT_A"
    default_date = pd.to_datetime("2026-04-21").date()  # Tuesday
    default_weather = "Clear"
    default_arrivals = 7
    default_port_cap = 12
    default_eff_cap = 12
    default_queue = 0
    default_hist_delay = 4.2
elif scenario == "Elevated Delay Risk":
    default_port = "PORT_E"
    default_date = pd.to_datetime("2026-04-25").date()  # Saturday
    default_weather = "Storm"
    default_arrivals = 12
    default_port_cap = 9
    default_eff_cap = 6
    default_queue = 8
    default_hist_delay = 8.5
elif scenario == "Borderline Case":
    default_port = "PORT_B"
    default_date = pd.to_datetime("2026-04-24").date()  # Friday
    default_weather = "Rain"
    default_arrivals = 9
    default_port_cap = 10
    default_eff_cap = 9
    default_queue = 2
    default_hist_delay = 6.2
else:
    default_port = "PORT_A"
    default_date = pd.to_datetime("2026-05-01").date()  # Friday
    default_weather = "Clear"
    default_arrivals = 9
    default_port_cap = 12
    default_eff_cap = 11
    default_queue = 1
    default_hist_delay = 5.5

st.sidebar.markdown("---")
st.sidebar.header("🎯 Decision Threshold")
threshold = st.sidebar.slider(
    "Classification Threshold:",
    min_value=0.20,
    max_value=0.60,
    value=0.40,
    step=0.05,
    help="Default 0.40 is the selected operational threshold prioritizing higher recall (>70% on validation). 0.50 is the unadjusted model default.",
)

if threshold == 0.40:
    st.sidebar.info("📌 **Threshold 0.40**: Selected operational threshold. Prioritizes delay detection (71.8% validation recall).")
elif threshold == 0.50:
    st.sidebar.info("📌 **Threshold 0.50**: Default model threshold. Balanced precision and recall.")
elif threshold == 0.55:
    st.sidebar.info("📌 **Threshold 0.55**: Maximizes validation F1 (0.5676) with lower recall (53.8%).")


# -----------------------------------------------------------------------------
# Input Form
# -----------------------------------------------------------------------------
st.subheader("1. Enter Current Six-Hour Observation")

col_time1, col_time2, col_time3 = st.columns(3)
with col_time1:
    obs_date = st.date_input("Observation Date:", value=default_date)
    day_of_week = obs_date.strftime("%A")

with col_time2:
    obs_interval = st.selectbox(
        "Observation Time Window (Current):",
        ["00:00 to 06:00", "06:00 to 12:00", "12:00 to 18:00", "18:00 to 24:00"],
        index=1,
    )

with col_time3:
    # Compute forecast cutoff and target horizon using exact date-time arithmetic
    slot_start_hours = {
        "00:00 to 06:00": 0,
        "06:00 to 12:00": 6,
        "12:00 to 18:00": 12,
        "18:00 to 24:00": 18,
    }
    start_hour = slot_start_hours[obs_interval]
    obs_start_dt = datetime.combine(obs_date, time(hour=start_hour))
    cutoff_dt = obs_start_dt + timedelta(hours=6)
    horizon_end_dt = cutoff_dt + timedelta(hours=6)

    cutoff_str = cutoff_dt.strftime("%Y-%m-%d %H:%M")
    target_horizon_str = f"{cutoff_dt.strftime('%Y-%m-%d %H:%M')} to {horizon_end_dt.strftime('%Y-%m-%d %H:%M')}"
    st.info(f"**Forecast Cutoff:** `{cutoff_str}`\n\n**Forecast Horizon (Next 6h):** `{target_horizon_str}`")

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### Port Identity & Environmental Factors")
    port_id = st.selectbox(
        "Port Identifier (`port_id`):",
        ["PORT_A", "PORT_B", "PORT_C", "PORT_D", "PORT_E"],
        index=["PORT_A", "PORT_B", "PORT_C", "PORT_D", "PORT_E"].index(default_port),
    )
    st.text_input(
        "Day of Week (`day_of_week`):",
        value=day_of_week,
        disabled=True,
        help="Automatically derived from the selected Observation Date above to maintain internal consistency.",
    )
    weather_condition = st.selectbox(
        "Weather Condition (`weather_condition`):",
        ["Clear", "Rain", "Storm"],
        index=["Clear", "Rain", "Storm"].index(default_weather),
        help="Adverse weather reduces operational efficiency and effective service capacity.",
    )
    vessel_arrivals_6h = st.number_input(
        "Vessel Arrivals in Window (`vessel_arrivals_6h`):",
        min_value=0,
        max_value=50,
        value=default_arrivals,
        step=1,
        help="Number of incoming vessels observed during this 6-hour interval.",
    )

with col2:
    st.markdown("##### Capacity & Backlog Conditions")
    port_capacity_6h = st.number_input(
        "Nominal Port Capacity (`port_capacity_6h`):",
        min_value=0,
        max_value=50,
        value=default_port_cap,
        step=1,
        help="Standard nominal handling capacity of the port per 6-hour window.",
    )
    effective_capacity_6h = st.number_input(
        "Effective Service Capacity (`effective_capacity_6h`):",
        min_value=1,
        max_value=50,
        value=default_eff_cap,
        step=1,
        help="Weather-adjusted operating capacity available during the interval.",
    )
    if effective_capacity_6h > port_capacity_6h and port_capacity_6h > 0:
        st.caption(f"⚠️ Note: Effective Capacity ({effective_capacity_6h}) exceeds Nominal Capacity ({port_capacity_6h}). In standard operations, weather reductions usually keep effective capacity $\\le$ nominal capacity.")

    queue_length = st.number_input(
        "Queue Backlog at Start of Window (`queue_length`):",
        min_value=0,
        max_value=50,
        value=default_queue,
        step=1,
        help="Number of waiting vessels waiting for berth at the start of the interval.",
    )
    historical_avg_delay_hours = st.number_input(
        "Historical Average Delay (`historical_avg_delay_hours`):",
        min_value=0.0,
        max_value=72.0,
        value=float(default_hist_delay),
        step=0.1,
        help="Recent rolling average delay (hours) observed at this port prior to this interval.",
    )

# Safely compute arrival_load_ratio
is_valid_capacity = port_capacity_6h > 0
if is_valid_capacity:
    arrival_load_ratio = round(vessel_arrivals_6h / port_capacity_6h, 3)
    ratio_color = "normal" if arrival_load_ratio <= 1.0 else "inverse"
    st.metric(
        label="Calculated Arrival Load Ratio (`vessel_arrivals_6h / port_capacity_6h`):",
        value=f"{arrival_load_ratio:.3f}",
        delta=f"{(arrival_load_ratio - 1.0):.1%} vs Capacity" if arrival_load_ratio > 1.0 else "Within nominal capacity",
        delta_color=ratio_color,
    )
else:
    arrival_load_ratio = None
    st.error("⚠️ **Validation Error:** Nominal Port Capacity (`port_capacity_6h`) must be greater than 0 to calculate the arrival load ratio.")


# -----------------------------------------------------------------------------
# Prediction Execution
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("2. Model Forecast")

predict_clicked = st.button("🔮 Generate Forecast for Next 6h Horizon", type="primary", disabled=not is_valid_capacity)

if predict_clicked and is_valid_capacity:
    # Prepare input dataframe with exact feature names matching trained pipeline
    input_features = pd.DataFrame(
        [
            {
                "vessel_arrivals_6h": vessel_arrivals_6h,
                "port_capacity_6h": port_capacity_6h,
                "arrival_load_ratio": arrival_load_ratio,
                "queue_length": queue_length,
                "effective_capacity_6h": effective_capacity_6h,
                "historical_avg_delay_hours": historical_avg_delay_hours,
                "port_id": port_id,
                "weather_condition": weather_condition,
                "day_of_week": day_of_week,
            }
        ]
    )

    # Inference without retraining
    predicted_prob = float(pipeline.predict_proba(input_features)[0][1])
    is_delayed = predicted_prob >= threshold

    # Results Layout
    res_col1, res_col2, res_col3 = st.columns([1.5, 2, 2])

    with res_col1:
        st.metric(
            label="Model Risk Score (Delay Probability)",
            value=f"{predicted_prob:.1%}",
            delta=f"{predicted_prob - threshold:+.1%} vs threshold",
            delta_color="inverse" if is_delayed else "normal",
        )

    with res_col2:
        if is_delayed:
            st.error(f"🚨 **DELAY PREDICTED (Class 1)**\n\nProbability ({predicted_prob:.1%}) $\\ge$ Threshold ({threshold:.2f})")
        else:
            st.success(f"✅ **ON-TIME PREDICTED (Class 0)**\n\nProbability ({predicted_prob:.1%}) < Threshold ({threshold:.2f})")

    with res_col3:
        st.info(
            f"**Forecast Horizon (Next 6h):**\n`{target_horizon_str}`\n\n"
            f"**Decision Threshold:** `{threshold:.2f}`\n\n"
            f"*(Delay defined as actual delay > 12h)*"
        )

    st.caption(
        "⚠️ **Statistical & Calibration Note:** This model risk score reflects the tree voting proportion across 100 decision trees. "
        "Because it has not undergone formal Platt/isotonic calibration, it represents relative operational risk rather than a certified real-world confidence interval."
    )

    # -------------------------------------------------------------------------
    # Situational Context and Explanations
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("3. Operational Context & Feature Influences")

    exp_col1, exp_col2 = st.columns([1.2, 1.8])

    with exp_col1:
        st.markdown("##### Global Feature Importance (Trained Model)")
        if feat_imp_df is not None:
            agg_imp = (
                feat_imp_df.groupby("original_feature")["importance"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            agg_imp.columns = ["Feature", "Global Weight"]
            show_table(agg_imp.head(6), hide_index=True)
            st.caption("Global weights represent Mean Decrease in Impurity (MDI) importance shares across tree splits, indicating overall model sensitivity.")

    with exp_col2:
        st.markdown("##### Entered Conditions vs. Historical Benchmarks")
        context_rows = [
            {
                "Feature": "Queue Backlog (`queue_length`)",
                "Your Input": f"{queue_length} vessels",
                "Training Baseline Average": f"{BENCHMARKS['queue_length']:.2f} vessels",
                "Situational Status": "Elevated Backlog" if queue_length > BENCHMARKS["queue_length"] else "Normal / Low",
            },
            {
                "Feature": "Arrival Load (`arrival_load_ratio`)",
                "Your Input": f"{arrival_load_ratio:.2f}",
                "Training Baseline Average": f"{BENCHMARKS['arrival_load_ratio']:.2f}",
                "Situational Status": "Exceeds Capacity" if arrival_load_ratio > 1.0 else ("High Demand" if arrival_load_ratio > BENCHMARKS["arrival_load_ratio"] else "Within Capacity"),
            },
            {
                "Feature": "Recent Historical Delay",
                "Your Input": f"{historical_avg_delay_hours:.1f} hours",
                "Training Baseline Average": f"{BENCHMARKS['historical_avg_delay_hours']:.1f} hours",
                "Situational Status": "Elevated Historical Delays" if historical_avg_delay_hours > BENCHMARKS["historical_avg_delay_hours"] else "Normal History",
            },
            {
                "Feature": "Weather Condition",
                "Your Input": weather_condition,
                "Training Baseline Average": "Clear (66% of intervals)",
                "Situational Status": "Capacity Disrupted" if weather_condition in ["Storm", "Rain"] else "Normal Weather",
            },
        ]
        show_table(pd.DataFrame(context_rows), hide_index=True)

    st.info(
        "🔍 **Non-Causal Governance Notice:** The observed conditions above provide operational background associated with past synthetic delays. "
        "The model does **not** establish that an individual feature caused the forecast, and exact local attribution values (such as SHAP scores) "
        "are not computed in this lightweight milestone."
    )


# -----------------------------------------------------------------------------
# Separate Model Evaluation Section
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("📊 Historical Model Performance & Threshold Evaluation (Validation & Test Sets)", expanded=False):
    st.markdown(
        """
        *Note: Model evaluation is strictly separated from prediction mode. The prediction interface above does not access or display
        any future ground-truth labels.*
        """
    )
    if eval_summary is not None:
        st.markdown("##### Held-Out Test Set Performance (500 samples, 53 delayed)")
        test_eval = eval_summary.get("test_evaluation", {})

        eval_table = [
            {
                "Configuration": "Dummy Baseline (Always On-time)",
                "Threshold": "N/A",
                "Accuracy": f"{test_eval['dummy_baseline']['accuracy']:.1%}",
                "Precision": f"{test_eval['dummy_baseline']['precision']:.1%}",
                "Recall": f"{test_eval['dummy_baseline']['recall']:.1%}",
                "F1-Score": f"{test_eval['dummy_baseline']['f1']:.4f}",
                "Delays Caught": test_eval["dummy_baseline"].get("delays_caught", "0 / 53"),
            },
            {
                "Configuration": "RF (Default Baseline)",
                "Threshold": "0.50",
                "Accuracy": f"{test_eval['default_threshold_0_50']['accuracy']:.1%}",
                "Precision": f"{test_eval['default_threshold_0_50']['precision']:.1%}",
                "Recall": f"{test_eval['default_threshold_0_50']['recall']:.1%}",
                "F1-Score": f"{test_eval['default_threshold_0_50']['f1']:.4f}",
                "Delays Caught": test_eval["default_threshold_0_50"].get("delays_caught", "23 / 53"),
            },
            {
                "Configuration": "RF (Operational Higher-Recall)",
                "Threshold": "0.40",
                "Accuracy": f"{test_eval['operational_threshold_0_40']['accuracy']:.1%}",
                "Precision": f"{test_eval['operational_threshold_0_40']['precision']:.1%}",
                "Recall": f"{test_eval['operational_threshold_0_40']['recall']:.1%}",
                "F1-Score": f"{test_eval['operational_threshold_0_40']['f1']:.4f}",
                "Delays Caught": test_eval["operational_threshold_0_40"].get("delays_caught", "31 / 53"),
            },
            {
                "Configuration": "RF (Validation F1-Max)",
                "Threshold": "0.55",
                "Accuracy": f"{test_eval['f1_maximizing_threshold_0_55']['accuracy']:.1%}",
                "Precision": f"{test_eval['f1_maximizing_threshold_0_55']['precision']:.1%}",
                "Recall": f"{test_eval['f1_maximizing_threshold_0_55']['recall']:.1%}",
                "F1-Score": f"{test_eval['f1_maximizing_threshold_0_55']['f1']:.4f}",
                "Delays Caught": test_eval["f1_maximizing_threshold_0_55"].get("delays_caught", "14 / 53"),
            },
        ]
        show_table(pd.DataFrame(eval_table), hide_index=True)
        st.caption("Selected threshold 0.40 catches 31 of 53 delays on the held-out test set compared to 23 at threshold 0.50 and 14 at threshold 0.55.")
    else:
        st.info("Evaluation artifacts not found. Run 'python src/evaluate_thresholds.py' to generate performance tables.")
