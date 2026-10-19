"""
Data Preparation Script for Port Congestion Forecasting.

Prepares the forecasting dataset by:
1. Loading the original synthetic dataset.
2. Sorting observations by port and timestamp.
3. Creating the next-interval target (target_delayed_next_6h) by shifting the
   delayed indicator forward by 1 step within each port.
4. Creating target_actual_delay_hours and target_timestamp for evaluation reference.
5. Removing terminal rows per port that lack a next-interval target.
6. Dropping current-interval outcome columns to eliminate data leakage.
7. Saving the resulting dataset to data/port_congestion_forecasting_dataset.csv.
"""

from pathlib import Path
import pandas as pd


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def prepare_forecasting_dataset(
    input_filename: str | None = None,
    output_filename: str = "port_congestion_forecasting_dataset.csv",
) -> pd.DataFrame:
    """
    Loads raw port records and transforms them into a next-interval forecasting dataset.
    """
    project_root = get_project_root()
    data_dir = project_root / "data"

    # Locate input file: check provided name, fallback to known variants
    if input_filename:
        input_path = data_dir / input_filename
    else:
        # Check standard synthetic file first, or v2 variant if present
        candidate_paths = [
            data_dir / "port_congestion_synthetic.csv",
            data_dir / "port_congestion_synthetic_v2.csv",
        ]
        input_path = next((p for p in candidate_paths if p.exists()), None)
        if input_path is None:
            raise FileNotFoundError(
                f"No synthetic dataset found in {data_dir}. Expected port_congestion_synthetic.csv"
            )

    print(f"Loading raw dataset from: {input_path}")
    df = pd.read_csv(input_path)
    initial_rows = len(df)
    print(f"Loaded {initial_rows} rows and {len(df.columns)} columns across {df['port_id'].nunique()} ports.")

    # 1. Ensure timestamp is parsed and sort chronologically within each port
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by=["port_id", "timestamp"]).reset_index(drop=True)

    # 2. Shift outcomes within each port to form next-interval targets
    # Shift -1 means row t gets the outcome from row t+1 of the same port
    df["target_timestamp"] = df.groupby("port_id")["timestamp"].shift(-1)
    df["target_delayed_next_6h"] = df.groupby("port_id")["is_delayed"].shift(-1)
    if "actual_delay_hours" in df.columns:
        df["target_actual_delay_hours"] = df.groupby("port_id")["actual_delay_hours"].shift(-1)

    # 3. Drop boundary records (the last interval for each port has no next target)
    df = df.dropna(subset=["target_delayed_next_6h"]).reset_index(drop=True)
    df["target_delayed_next_6h"] = df["target_delayed_next_6h"].astype(int)

    # 4. Remove current-interval outcome columns to prevent any target leakage
    columns_to_drop = ["actual_delay_hours", "is_delayed"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Format timestamps back to standard ISO string representation
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["target_timestamp"] = pd.to_datetime(df["target_timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # 5. Save the prepared forecasting dataset
    output_path = data_dir / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nPrepared forecasting dataset saved to: {output_path}")
    print(f"Final row count: {len(df)} (dropped {initial_rows - len(df)} boundary rows)")
    print("Target class counts (target_delayed_next_6h):")
    counts = df["target_delayed_next_6h"].value_counts().rename(index={0: "On-time (0)", 1: "Delayed (1)"})
    for label, count in counts.items():
        print(f"  {label}: {count} ({count / len(df):.1%})")

    return df


if __name__ == "__main__":
    prepare_forecasting_dataset()
