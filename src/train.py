"""
Model Training and Evaluation Pipeline for Port Congestion Forecasting.

This script executes Milestone 1:
1. Loads the prepared forecasting dataset.
2. Performs chronological data splitting (~70% train, ~10% val, ~20% test)
   using shared timestamp cutoffs across ports to prevent temporal leakage.
3. Builds scikit-learn preprocessing (OneHotEncoder for categorical features,
   passthrough for numeric features) fitted only on the training set.
4. Trains a DummyClassifier baseline (most_frequent).
5. Trains a RandomForestClassifier with balanced class weighting.
6. Evaluates both models on validation and held-out test sets.
7. Saves the trained pipeline, evaluation metrics JSON, test predictions CSV,
   and feature importances CSV.
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Loads and validates that the forecasting dataset exists."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {dataset_path}. "
            "Please run 'python src/prepare_data.py' first."
        )
    df = pd.read_csv(dataset_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "target_timestamp" in df.columns:
        df["target_timestamp"] = pd.to_datetime(df["target_timestamp"])
    return df


def split_data_chronologically(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset chronologically based on shared target timestamp cutoffs.

    By using target_timestamp (the start of the horizon being predicted),
    we ensure that:
    1. All ports for any given timestamp remain in the same partition.
    2. The partition boundaries are defined strictly by the forecast horizon,
       meaning no training sample predicts an outcome occurring in the
       validation horizon.
    Because target_timestamp is shifted by exactly 6 hours relative to timestamp,
    this maintains a clean, synchronized, and non-overlapping sequence across partitions.
    """
    unique_target_timestamps = (
        pd.Series(df["target_timestamp"].unique()).sort_values().reset_index(drop=True)
    )
    total_timestamps = len(unique_target_timestamps)

    if total_timestamps < 10:
        raise ValueError(
            f"Insufficient unique timestamps ({total_timestamps}) for chronological splitting."
        )

    train_cutoff_idx = int(total_timestamps * train_ratio)
    val_cutoff_idx = int(total_timestamps * (train_ratio + val_ratio))

    train_cutoff = unique_target_timestamps.iloc[train_cutoff_idx]
    val_cutoff = unique_target_timestamps.iloc[val_cutoff_idx]

    train_df = df[df["target_timestamp"] < train_cutoff].copy()
    val_df = df[
        (df["target_timestamp"] >= train_cutoff) & (df["target_timestamp"] < val_cutoff)
    ].copy()
    test_df = df[df["target_timestamp"] >= val_cutoff].copy()

    # Integrity verification
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        if len(split) == 0:
            raise ValueError(f"Chronological split produced an empty {name} partition.")
        pos_count = int((split["target_delayed_next_6h"] == 1).sum())
        if pos_count == 0:
            raise ValueError(
                f"Chronological split produced {name} partition with 0 delayed instances."
            )

    return train_df, val_df, test_df


def print_split_summary(name: str, split_df: pd.DataFrame, total_rows: int) -> None:
    """Prints date range, row counts, and target class balance for a split."""
    count_0 = int((split_df["target_delayed_next_6h"] == 0).sum())
    count_1 = int((split_df["target_delayed_next_6h"] == 1).sum())
    obs_start = split_df["timestamp"].min()
    obs_end = split_df["timestamp"].max()
    target_start = split_df["target_timestamp"].min()
    target_end = split_df["target_timestamp"].max()

    print(f"\n--- {name} Split ---")
    print(f"  Rows: {len(split_df)} ({len(split_df) / total_rows:.1%})")
    print(f"  Observation range:    {obs_start} to {obs_end}")
    print(f"  Target horizon range: {target_start} to {target_end}")
    print(f"  On-time (0): {count_0} ({count_0 / len(split_df):.1%})")
    print(f"  Delayed (1): {count_1} ({count_1 / len(split_df):.1%})")


def evaluate_predictions(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None) -> dict:
    """Computes standard classification evaluation metrics safely."""
    metrics = {
        "samples": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # ROC-AUC is meaningful only if both classes are present in y_true and probabilities vary
    if len(np.unique(y_true)) > 1 and y_prob is not None:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def print_metrics(model_name: str, split_name: str, metrics: dict) -> None:
    """Prints evaluation metrics clearly to the console."""
    print(f"\n[{model_name}] - {split_name} Evaluation:")
    print(f"  Samples:    {metrics['samples']}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1-Score:   {metrics['f1']:.4f}")
    auc_str = f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "N/A"
    print(f"  ROC-AUC:    {auc_str}")
    cm = metrics["confusion_matrix"]
    print(f"  Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")


def map_feature_to_original(feature_name: str, original_features: list[str]) -> str:
    """Maps a transformed (one-hot) feature name back to the original input column."""
    # Match longest prefix first to avoid partial string ambiguity
    for orig in sorted(original_features, key=len, reverse=True):
        if feature_name == orig or feature_name.startswith(orig + "_"):
            return orig
    return feature_name


def train_and_evaluate() -> None:
    """Main workflow: data loading, splitting, modeling, evaluation, and saving."""
    project_root = get_project_root()
    dataset_path = project_root / "data" / "port_congestion_forecasting_dataset.csv"
    models_dir = project_root / "models"
    results_dir = project_root / "results"

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PORT CONGESTION FORECASTING: MODEL TRAINING & EVALUATION")
    print("=" * 60)

    # 1. Load forecasting dataset
    df = load_dataset(dataset_path)
    total_rows = len(df)
    print(f"Loaded prepared forecasting dataset with {total_rows} rows.")

    # 2. Chronological data split
    train_df, val_df, test_df = split_data_chronologically(df)
    print("\nDataset Partitioning (Chronological):")
    print_split_summary("Training", train_df, total_rows)
    print_split_summary("Validation", val_df, total_rows)
    print_split_summary("Held-out Test", test_df, total_rows)

    # 3. Define feature sets and targets strictly avoiding leakage
    # Excluded: timestamp, target_timestamp, target_actual_delay_hours, and target column
    numeric_features = [
        "vessel_arrivals_6h",
        "port_capacity_6h",
        "arrival_load_ratio",
        "queue_length",
        "effective_capacity_6h",
        "historical_avg_delay_hours",
    ]
    categorical_features = [
        "port_id",
        "weather_condition",
        "day_of_week",
    ]
    feature_columns = numeric_features + categorical_features
    target_column = "target_delayed_next_6h"

    X_train, y_train = train_df[feature_columns], train_df[target_column]
    X_val, y_val = val_df[feature_columns], val_df[target_column]
    X_test, y_test = test_df[feature_columns], test_df[target_column]

    print(f"\nModel Predictor Count: {len(feature_columns)}")
    print(f"  Numeric features:     {numeric_features}")
    print(f"  Categorical features: {categorical_features}")

    # 4. Scikit-learn Preprocessing Pipeline
    # Fitted only on training data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        verbose_feature_names_out=False,
    )

    # 5. Baseline Model: DummyClassifier (most_frequent)
    print("\n" + "=" * 60)
    print("1. TRAINING BASELINE MODEL (DummyClassifier: most_frequent)")
    print("=" * 60)
    dummy_model = DummyClassifier(strategy="most_frequent")
    dummy_model.fit(X_train, y_train)

    # Evaluate Dummy on Val and Test
    dummy_val_pred = dummy_model.predict(X_val)
    dummy_val_prob = dummy_model.predict_proba(X_val)[:, 1]
    dummy_val_metrics = evaluate_predictions(y_val, dummy_val_pred, dummy_val_prob)
    print_metrics("Baseline (Dummy)", "Validation", dummy_val_metrics)

    dummy_test_pred = dummy_model.predict(X_test)
    dummy_test_prob = dummy_model.predict_proba(X_test)[:, 1]
    dummy_test_metrics = evaluate_predictions(y_test, dummy_test_pred, dummy_test_prob)
    print_metrics("Baseline (Dummy)", "Held-out Test", dummy_test_metrics)

    # 6. Main Model: RandomForestClassifier
    print("\n" + "=" * 60)
    print("2. TRAINING PRIMARY MODEL (RandomForestClassifier)")
    print("=" * 60)
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", rf_classifier),
        ]
    )

    rf_pipeline.fit(X_train, y_train)

    # Evaluate RF on Validation Set
    rf_val_pred = rf_pipeline.predict(X_val)
    rf_val_prob = rf_pipeline.predict_proba(X_val)[:, 1]
    rf_val_metrics = evaluate_predictions(y_val, rf_val_pred, rf_val_prob)
    print_metrics("Random Forest", "Validation", rf_val_metrics)

    # Evaluate RF on Held-out Test Set
    rf_test_pred = rf_pipeline.predict(X_test)
    rf_test_prob = rf_pipeline.predict_proba(X_test)[:, 1]
    rf_test_metrics = evaluate_predictions(y_test, rf_test_pred, rf_test_prob)
    print_metrics("Random Forest", "Held-out Test", rf_test_metrics)

    # 7. Extract Feature Importances
    fitted_preprocessor = rf_pipeline.named_steps["preprocessor"]
    transformed_feature_names = fitted_preprocessor.get_feature_names_out()
    fitted_rf = rf_pipeline.named_steps["classifier"]
    importances = fitted_rf.feature_importances_

    feature_imp_df = pd.DataFrame(
        {
            "feature": transformed_feature_names,
            "original_feature": [
                map_feature_to_original(f, feature_columns)
                for f in transformed_feature_names
            ],
            "importance": np.round(importances, 6),
        }
    ).sort_values(by="importance", ascending=False).reset_index(drop=True)

    print("\nTop 10 Feature Importances (Random Forest):")
    for _, row in feature_imp_df.head(10).iterrows():
        print(f"  {row['feature']:<28} (original: {row['original_feature']:<20}) : {row['importance']:.4f}")

    # 8. Save Outputs
    # A. Save trained model pipeline
    model_artifact_path = models_dir / "random_forest_pipeline.joblib"
    joblib.dump(rf_pipeline, model_artifact_path)
    print(f"\n[Artifact Saved] Model pipeline: {model_artifact_path}")

    # B. Save evaluation metrics JSON
    metrics_output = {
        "dataset": {
            "source": str(dataset_path.name),
            "total_rows": total_rows,
            "splits": {
                "train": {
                    "samples": len(train_df),
                    "observation_date_range": [
                        str(train_df["timestamp"].min()),
                        str(train_df["timestamp"].max()),
                    ],
                    "target_date_range": [
                        str(train_df["target_timestamp"].min()),
                        str(train_df["target_timestamp"].max()),
                    ],
                    "target_counts": {
                        "on_time_0": int((train_df[target_column] == 0).sum()),
                        "delayed_1": int((train_df[target_column] == 1).sum()),
                    },
                },
                "validation": {
                    "samples": len(val_df),
                    "observation_date_range": [
                        str(val_df["timestamp"].min()),
                        str(val_df["timestamp"].max()),
                    ],
                    "target_date_range": [
                        str(val_df["target_timestamp"].min()),
                        str(val_df["target_timestamp"].max()),
                    ],
                    "target_counts": {
                        "on_time_0": int((val_df[target_column] == 0).sum()),
                        "delayed_1": int((val_df[target_column] == 1).sum()),
                    },
                },
                "test": {
                    "samples": len(test_df),
                    "observation_date_range": [
                        str(test_df["timestamp"].min()),
                        str(test_df["timestamp"].max()),
                    ],
                    "target_date_range": [
                        str(test_df["target_timestamp"].min()),
                        str(test_df["target_timestamp"].max()),
                    ],
                    "target_counts": {
                        "on_time_0": int((test_df[target_column] == 0).sum()),
                        "delayed_1": int((test_df[target_column] == 1).sum()),
                    },
                },
            },
        },
        "features": {
            "numeric": numeric_features,
            "categorical": categorical_features,
            "target": target_column,
        },
        "models": {
            "dummy_baseline": {
                "strategy": "most_frequent",
                "validation_metrics": dummy_val_metrics,
                "test_metrics": dummy_test_metrics,
            },
            "random_forest": {
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "min_samples_leaf": 2,
                    "random_state": 42,
                    "class_weight": "balanced",
                },
                "validation_metrics": rf_val_metrics,
                "test_metrics": rf_test_metrics,
            },
        },
    }

    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=2)
    print(f"[Artifact Saved] Evaluation metrics: {metrics_path}")

    # C. Save test predictions CSV
    test_predictions_df = pd.DataFrame(
        {
            "timestamp": test_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "target_timestamp": test_df["target_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            if pd.api.types.is_datetime64_any_dtype(test_df["target_timestamp"])
            else test_df["target_timestamp"].values,
            "port_id": test_df["port_id"].values,
            "true_target": test_df[target_column].values,
            "predicted_label": rf_test_pred,
            "predicted_delay_probability": np.round(rf_test_prob, 4),
        }
    )
    predictions_path = results_dir / "test_predictions.csv"
    test_predictions_df.to_csv(predictions_path, index=False)
    print(f"[Artifact Saved] Test predictions: {predictions_path} ({len(test_predictions_df)} rows)")

    # D. Save feature importances CSV
    feature_imp_path = results_dir / "feature_importance.csv"
    feature_imp_df.to_csv(feature_imp_path, index=False)
    print(f"[Artifact Saved] Feature importances: {feature_imp_path}")

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    train_and_evaluate()
