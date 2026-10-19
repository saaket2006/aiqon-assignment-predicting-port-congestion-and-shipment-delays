"""
Reproducible synthetic dataset generator for the Aiqon port congestion assignment.
Run: python generate_dataset.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_PORTS = 5
ROWS_PER_PORT = 500
START = pd.Timestamp("2026-01-01 00:00:00")
WEATHER = ["Clear", "Rain", "Storm"]
WEATHER_PROBS = [0.66, 0.27, 0.07]
WEATHER_CAPACITY_FACTOR = {"Clear": 1.00, "Rain": 0.88, "Storm": 0.68}
PORTS = {
    "PORT_A": {"capacity": 12, "arrival_rate": 8.0},
    "PORT_B": {"capacity": 10, "arrival_rate": 7.2},
    "PORT_C": {"capacity": 14, "arrival_rate": 9.2},
    "PORT_D": {"capacity": 16, "arrival_rate": 11.0},
    "PORT_E": {"capacity": 9, "arrival_rate": 6.5},
}

def generate_dataset(output_path="port_congestion_synthetic.csv"):
    rng = np.random.default_rng(SEED)
    rows = []

    # Simulate each port in chronological 6-hour intervals so queues persist.
    for port_id, port in PORTS.items():
        backlog = 0
        prior_delays = []
        for t in range(ROWS_PER_PORT):
            timestamp = START + pd.Timedelta(hours=6 * t)
            hour = timestamp.hour
            weekend_factor = 0.90 if timestamp.dayofweek >= 5 else 1.0
            peak_factor = 1.12 if hour in (6, 12, 18) else 1.0
            weather = rng.choice(WEATHER, p=WEATHER_PROBS)

            # Arrival counts vary around a port-specific rate and temporal pattern.
            arrival_rate = port["arrival_rate"] * weekend_factor * peak_factor
            if weather == "Storm":
                arrival_rate *= 1.08  # vessels may bunch up around disruptions
            arrivals = int(rng.poisson(arrival_rate))

            base_capacity = port["capacity"]
            effective_capacity = max(
                1, int(round(base_capacity * WEATHER_CAPACITY_FACTOR[weather]))
            )
            queue_length = int(backlog)
            arrival_load_ratio = arrivals / base_capacity

            # Historical average uses only earlier observations at this port.
            historical_avg = float(np.mean(prior_delays[-12:])) if prior_delays else 2.5

            # Delay components
            arrival_pressure_effect = max(0.0, arrival_load_ratio - 0.70) * 11.0
            queue_effect = queue_length * 1.15
            weather_effect = {"Clear": 0.0, "Rain": 1.8, "Storm": 5.5}[weather]
            history_effect = 0.22 * historical_avg
            random_noise = rng.normal(0.0, 2.7)
            disruption = rng.binomial(1, 0.035) * rng.uniform(4.0, 10.0)

            actual_delay = (
                1.2
                + arrival_pressure_effect
                + queue_effect
                + weather_effect
                + history_effect
                + random_noise
                + disruption
            )

            actual_delay = float(np.clip(actual_delay, 0.0, 72.0))
            is_delayed = int(actual_delay > 12.0)

            rows.append({
                "timestamp": timestamp.isoformat(sep=" "),
                "port_id": port_id,
                "vessel_arrivals_6h": arrivals,
                "port_capacity_6h": base_capacity,
                "arrival_load_ratio": round(float(arrival_load_ratio), 3),
                "queue_length": queue_length,
                "effective_capacity_6h": effective_capacity,
                "weather_condition": weather,
                "historical_avg_delay_hours": round(historical_avg, 2),
                "day_of_week": timestamp.day_name(),
                "actual_delay_hours": round(actual_delay, 2),
                "is_delayed": is_delayed,
            })

            # Update backlog for the next 6-hour window after service.
            backlog = max(0, queue_length + arrivals - effective_capacity)
            prior_delays.append(actual_delay)

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["timestamp", "port_id"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    result = generate_dataset()

    print(f"Saved {len(result)} records.")
    print(f"Number of ports: {result['port_id'].nunique()}")
    print(f"Records per port:\n{result['port_id'].value_counts().sort_index()}")
    print("\nDelay distribution:")
    print(
        result["is_delayed"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
        .rename(index={0: "On time", 1: "Delayed"})
    )
    print(f"\nDelay rate: {result['is_delayed'].mean():.1%}")
    print(f"Average delay: {result['actual_delay_hours'].mean():.2f} hours")
    print(f"Maximum queue length: {result['queue_length'].max()}")
    