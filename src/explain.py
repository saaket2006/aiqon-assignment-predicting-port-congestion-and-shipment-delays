"""
Prediction Explanation Tool for Port Congestion Forecasting.

This script implements Milestone 2 prediction explanations:
1. Reports global Random Forest feature importances.
2. For any given observation, extracts predicted probability, applies the
   chosen operational threshold, and reports observed values for the strongest
   predictive features (e.g., queue length, historical average delay, arrival load).
3. Clearly distinguishes global feature importance from case-specific observations.
4. Includes explicit non-causal disclaimers and methodology limitations
   (no SHAP/LIME or manufactured local attributions).
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from train import get_project_root, load_dataset, split_data_chronologically


def load_explainer_assets():
    """Loads trained pipeline, dataset, and feature importance artifacts."""
    project_root = get_project_root()
    model_path = project_root / "models" / "random_forest_pipeline.joblib"
    feat_imp_path = project_root / "results" / "feature_importance.csv"
    data_path = project_root / "data" / "port_congestion_forecasting_dataset.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}.")
    if not feat_imp_path.exists():
        raise FileNotFoundError(f"Feature importance file not found at {feat_imp_path}.")

    pipeline = joblib.load(model_path)
    feat_imp_df = pd.read_csv(feat_imp_path)
    df = load_dataset(data_path)
    train_df, val_df, test_df = split_data_chronologically(df)

    # Compute baseline reference stats from training data only
    num_cols = [
        "queue_length",
        "historical_avg_delay_hours",
        "arrival_load_ratio",
        "vessel_arrivals_6h",
        "effective_capacity_6h",
        "port_capacity_6h",
    ]
    train_stats = {
        col: {
            "mean": round(float(train_df[col].mean()), 2),
            "std": round(float(train_df[col].std()), 2),
            "median": round(float(train_df[col].median()), 2),
        }
        for col in num_cols
    }

    return pipeline, feat_imp_df, train_stats, test_df


def format_explanation(
    observation: pd.Series | dict,
    model,
    feat_imp_df: pd.DataFrame,
    train_stats: dict,
    threshold: float = 0.40,
) -> dict:
    """
    Generates a beginner-friendly explanation for a single prediction.
    """
    feature_cols = [
        "vessel_arrivals_6h",
        "port_capacity_6h",
        "arrival_load_ratio",
        "queue_length",
        "effective_capacity_6h",
        "historical_avg_delay_hours",
        "port_id",
        "weather_condition",
        "day_of_week",
    ]

    obs_series = pd.Series(observation) if isinstance(observation, dict) else observation
    input_df = pd.DataFrame([obs_series[feature_cols]])

    # Generate probability
    prob = float(model.predict_proba(input_df)[0][1])
    is_delayed_pred = int(prob >= threshold)
    label_str = "DELAYED (Predicted Delay)" if is_delayed_pred == 1 else "ON-TIME (Normal)"

    # Top global features
    # Aggregate importance by original feature
    agg_imp = (
        feat_imp_df.groupby("original_feature")["importance"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    top_features = []
    for _, row in agg_imp.head(5).iterrows():
        feat = row["original_feature"]
        raw_val = obs_series.get(feat, "N/A")
        if isinstance(raw_val, (np.integer, int)):
            val = int(raw_val)
        elif isinstance(raw_val, (np.floating, float)):
            val = round(float(raw_val), 4)
        else:
            val = str(raw_val)

        ref_mean = train_stats.get(feat, {}).get("mean", None)
        if ref_mean is not None:
            ref_mean = float(ref_mean)

        feat_info = {
            "feature": feat,
            "global_importance_weight": round(float(row["importance"]), 4),
            "observed_value_context": val,
            "training_mean_baseline": ref_mean,
        }
        top_features.append(feat_info)

    # Narrative explanation based on observed conditions
    q_val = int(round(float(obs_series.get("queue_length", 0))))
    hist_val = obs_series.get("historical_avg_delay_hours", 0)
    load_val = obs_series.get("arrival_load_ratio", 0)
    weather_val = obs_series.get("weather_condition", "Clear")

    context_points = []
    if q_val > train_stats["queue_length"]["mean"]:
        context_points.append(
            f"queue backlog ({q_val} vessels) is above typical training average ({train_stats['queue_length']['mean']} vessels)"
        )
    else:
        context_points.append(
            f"queue backlog ({q_val} vessels) is at or below baseline ({train_stats['queue_length']['mean']} vessels)"
        )

    if hist_val > train_stats["historical_avg_delay_hours"]["mean"]:
        context_points.append(
            f"recent historical delay average ({hist_val}h) is elevated relative to average ({train_stats['historical_avg_delay_hours']['mean']}h)"
        )

    if load_val > 1.0:
        context_points.append(
            f"arrival load ratio ({load_val:.2f}) indicates incoming vessels exceed nominal capacity"
        )
    elif load_val > train_stats["arrival_load_ratio"]["mean"]:
        context_points.append(
            f"arrival load ratio ({load_val:.2f}) is moderately high"
        )

    if weather_val in ["Storm", "Rain"]:
        context_points.append(
            f"weather condition ({weather_val}) reduces effective operating capacity"
        )

    narrative = (
        f"For port {obs_series.get('port_id', 'Unknown')}, the model estimates a "
        f"{prob:.1%} probability of delay in the upcoming 6-hour horizon ({obs_series.get('target_timestamp', 'next interval')}). "
        f"Classification is {label_str} at threshold {threshold:.2f}. "
        f"Observed operational context: {'; '.join(context_points)}."
    )

    explanation = {
        "prediction_metadata": {
            "observation_timestamp": str(obs_series.get("timestamp", "N/A")),
            "target_horizon_timestamp": str(obs_series.get("target_timestamp", "N/A")),
            "port_id": str(obs_series.get("port_id", "N/A")),
            "predicted_delay_probability": round(prob, 4),
            "threshold_applied": threshold,
            "predicted_classification": label_str,
            "ground_truth_target": int(obs_series.get("target_delayed_next_6h", -1))
            if "target_delayed_next_6h" in obs_series
            else None,
        },
        "global_feature_importance_and_observed_context": top_features,
        "case_observation_narrative": narrative,
        "methodology_and_limitations": {
            "observed_values_as_context": (
                "Observed feature values provide operational background describing current conditions. "
                "They are situational context, not mathematically certified local feature contributions or SHAP values."
            ),
            "global_vs_case_distinction": (
                "Feature importances represent the model's global sensitivity across all 100 decision trees "
                "in the training dataset. They do not calculate exact mathematical contribution points for "
                "this single prediction."
            ),
            "uncalibrated_probability_disclaimer": (
                "The predicted delay probability represents the raw proportion of ensemble trees voting for a delay. "
                "It is uncalibrated and should not be interpreted as a validated real-world operational probability."
            ),
            "non_causal_disclaimer": (
                "Observed factors reflect statistical associations in synthetic simulation history. "
                "The model does NOT establish that these factors caused the predicted outcome."
            ),
        },
    }

    return explanation


def print_explanation_card(expl: dict) -> None:
    """Prints a clean, human-readable terminal explanation card."""
    meta = expl["prediction_metadata"]
    print("+" + "-" * 72 + "+")
    print(f"| FORECAST EXPLANATION: {meta['port_id']} -> Horizon: {meta['target_horizon_timestamp']}")
    print("+" + "-" * 72 + "+")
    print(f"| Observation Cutoff:          {meta['observation_timestamp']}")
    print(f"| Predicted Delay Probability: {meta['predicted_delay_probability']:.1%} (Uncalibrated)")
    print(f"| Operational Threshold:       {meta['threshold_applied']:.2f}")
    print(f"| Predicted Classification:    {meta['predicted_classification']}")
    if meta["ground_truth_target"] is not None:
        gt_str = "DELAYED (1)" if meta["ground_truth_target"] == 1 else "ON-TIME (0)"
        print(f"| Actual Ground-Truth Label:   {gt_str}")
    print("|" + "-" * 72)
    print("| Top Global Features & Observed Situational Context:")
    print(f"|   {'Feature Name':<28}{'Global Weight':<15}{'Observed Context':<18}{'Train Mean'}")
    for f in expl["global_feature_importance_and_observed_context"]:
        mean_str = f"{f['training_mean_baseline']:.2f}" if f["training_mean_baseline"] is not None else "N/A"
        print(
            f"|   {f['feature']:<28}{f['global_importance_weight']:<15.4f}{str(f['observed_value_context']):<18}{mean_str}"
        )
    print("|" + "-" * 72)
    print("| Contextual Summary:")
    print(f"|   {expl['case_observation_narrative']}")
    print("|" + "-" * 72)
    print("| Limitations & Governance Disclaimers:")
    lims = expl["methodology_and_limitations"]
    print(f"|   * Context Only: {lims['observed_values_as_context']}")
    print(f"|   * Probability:  {lims['uncalibrated_probability_disclaimer']}")
    print(f"|   * Non-Causal:   {lims['non_causal_disclaimer']}")
    print(f"|   * Global Scope: {lims['global_vs_case_distinction']}")
    print("+" + "-" * 72 + "+\n")


def run_explanation_demo() -> None:
    """Runs prediction explanations on representative test set cases."""
    print("=" * 74)
    print("PORT CONGESTION FORECASTING: PREDICTION EXPLANATIONS DEMO")
    print("=" * 74)

    pipeline, feat_imp_df, train_stats, test_df = load_explainer_assets()

    feature_cols = [
        "vessel_arrivals_6h",
        "port_capacity_6h",
        "arrival_load_ratio",
        "queue_length",
        "effective_capacity_6h",
        "historical_avg_delay_hours",
        "port_id",
        "weather_condition",
        "day_of_week",
    ]
    test_probs = pipeline.predict_proba(test_df[feature_cols])[:, 1]
    test_df_copy = test_df.reset_index(drop=True)
    test_df_copy["predicted_prob"] = test_probs

    # Select 3 representative test cases:
    # 1. High-probability true positive (delayed)
    # 2. Low-probability true negative (on-time)
    # 3. Borderline case near recommended 0.40 threshold
    high_risk_idx = test_df_copy[
        (test_df_copy["predicted_prob"] > 0.70) & (test_df_copy["target_delayed_next_6h"] == 1)
    ].index[0]

    low_risk_idx = test_df_copy[
        (test_df_copy["predicted_prob"] < 0.20) & (test_df_copy["target_delayed_next_6h"] == 0)
    ].index[0]

    borderline_idx = test_df_copy[
        (test_df_copy["predicted_prob"] >= 0.38) & (test_df_copy["predicted_prob"] <= 0.45)
    ].index[0]

    cases = [
        ("Case 1: Elevated Delay Risk Forecast", test_df_copy.loc[high_risk_idx]),
        ("Case 2: Baseline On-Time Operations Forecast", test_df_copy.loc[low_risk_idx]),
        ("Case 3: Borderline Operational Forecast", test_df_copy.loc[borderline_idx]),
    ]

    all_explanations = []
    for title, row in cases:
        print(f"\n>>> {title} (Test row index {row.name})")
        expl = format_explanation(row, pipeline, feat_imp_df, train_stats, threshold=0.40)
        print_explanation_card(expl)
        all_explanations.append(expl)

    # Save to results/prediction_explanations.json
    project_root = get_project_root()
    output_path = project_root / "results" / "prediction_explanations.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_explanations, f, indent=2)
    print(f"[Artifact Saved] Prediction explanations JSON: {output_path}")


if __name__ == "__main__":
    run_explanation_demo()
