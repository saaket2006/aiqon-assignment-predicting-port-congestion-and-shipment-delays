# Port Congestion and Shipping Delay Project
## Dataset Validation Report

**Dataset file:** `port_congestion_synthetic.csv`  
**Purpose:** Document the dataset structure, target-label integrity, simulated operational patterns, and checks completed before model development.

## 1. Dataset overview

- **Records:** 2,500
- **Columns:** 12
- **Ports:** 5
- **Records per port:** PORT_A: 500, PORT_B: 500, PORT_C: 500, PORT_D: 500, PORT_E: 500
- **Delayed records:** 296 (11.8%)
- **On-time records:** 2,204 (88.2%)
- **Mean simulated delay:** 6.39 hours
- **Maximum queue length:** 19
- **Overall validation status:** Passed

The dataset is synthetic and generated from explicit simulation assumptions. It is not observed operational port data, and its statistics must not be interpreted as actual port performance.

## 2. Dataset schema

| Column | Meaning |
|---|---|
| `timestamp` | Start time of the six-hour observation window |
| `port_id` | Synthetic port identifier |
| `vessel_arrivals_6h` | Vessel arrivals during the six-hour window |
| `port_capacity_6h` | Nominal service capacity during the window |
| `arrival_load_ratio` | Vessel arrivals divided by nominal capacity |
| `queue_length` | Backlog at the start of the window |
| `effective_capacity_6h` | Weather-adjusted service capacity used by the queue simulation |
| `weather_condition` | Simulated weather category: Clear, Rain, or Storm |
| `historical_avg_delay_hours` | Average delay over previous observations for that port, based on the generator's history window |
| `day_of_week` | Day of the week for the observation |
| `actual_delay_hours` | Simulated delay in hours; outcome only |
| `is_delayed` | Target label: 1 when actual delay exceeds 12 hours, otherwise 0 |

## 3. Structural and integrity checks

| Check | Result | Status |
|---|---:|---|
| Expected schema | 12/12 expected columns; 0 unexpected columns | PASS |
| Missing cells | 0 | PASS |
| Invalid timestamps | 0 | PASS |
| Duplicate port-timestamp pairs | 0 | PASS |
| Chronological order within each port | True | PASS |
| Target-label mismatches | 0 | PASS |
| Valid weather categories | True | PASS |
| Effective-capacity formula mismatches | 0 | PASS |
| Queue transition mismatches | 0/2495 | PASS |

The target rule is `is_delayed = 1` when `actual_delay_hours > 12`; otherwise `is_delayed = 0`.

## 4. Port-level summary

| Port | Records | Mean arrivals / 6h | Nominal capacity | Mean effective capacity | Mean arrival load | Mean queue | Maximum queue | Delay rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PORT_A | 500 | 8.47 | 12 | 11.48 | 70.6% | 0.53 | 13 | 7.2% |
| PORT_B | 500 | 7.74 | 10 | 9.54 | 77.4% | 1.63 | 19 | 17.4% |
| PORT_C | 500 | 9.92 | 14 | 13.22 | 70.9% | 0.54 | 9 | 5.2% |
| PORT_D | 500 | 11.72 | 16 | 15.00 | 73.2% | 0.76 | 15 | 9.6% |
| PORT_E | 500 | 6.93 | 9 | 8.52 | 77.0% | 1.54 | 13 | 19.8% |

Differences between ports are produced by their simulated capacity and arrival-rate assumptions. They are not rankings of real ports.

## 5. Delay behavior by weather

| Weather | Records | Mean effective capacity | Mean delay (hours) | Delay rate |
|---|---:|---:|---:|---:|
| Clear | 1651 | 12.20 | 5.51 | 8.8% |
| Rain | 675 | 10.72 | 7.34 | 13.3% |
| Storm | 174 | 8.64 | 11.10 | 35.1% |

These are descriptive patterns in the synthetic data, not causal estimates for real operations.

## 6. Feature-to-outcome correlations

Pearson correlations with `actual_delay_hours`:

- Arrival load ratio vs. actual delay: **0.404**
- Queue length vs. actual delay: **0.611**
- Effective capacity vs. actual delay: **-0.272**
- Historical average delay vs. actual delay: **0.391**

The correlations reflect the assumptions and equations used by the simulator. They do not establish real-world causal relationships.

## 7. Modeling precautions

- Use `is_delayed` as the classification target.
- Exclude `actual_delay_hours` from model inputs because it directly defines the target.
- Do not use features that depend on future observations.
- Preserve time order when constructing training, validation, and test sets.
- Encode categorical predictors such as `port_id`, `weather_condition`, and `day_of_week`.
- Evaluate with precision, recall, F1-score, ROC-AUC, and a confusion matrix; compare against a simple baseline.
- Describe model performance as performance on simulated data, not as proven real-world shipping-delay prediction.

## 8. Conclusion

The dataset passed for the structural and simulation-consistency checks listed above. These checks establish internal consistency with the documented simulation rules; they do not establish that the simulation fully represents real port operations.

**Recommended next step:** finalize the feature set and chronological evaluation design before training the first model.
