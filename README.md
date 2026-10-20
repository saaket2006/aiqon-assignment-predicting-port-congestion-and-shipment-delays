# Port Congestion and Shipping Delay Forecaster

An interpretable machine learning system and interactive Streamlit interface for predicting operational port congestion and vessel shipment delays (>12 hours) over the upcoming six-hour operational window.

---

## 1. Problem Statement & Project Objective

Maritime shipping terminals experience dynamic operational friction driven by fluctuating vessel arrival rates, queuing bottlenecks, berth capacity constraints, and adverse weather events. When a port's effective throughput capacity is exceeded, vessel queues form, leading to idle vessel time, berth congestion, and steep demurrage fees.

### Objective
Provide port authorities, logistics dispatchers, and shipping operators with a **forward-looking early warning system** that forecasts whether a commercial port will experience a **delay (>12 hours)** during the **next 6-hour interval**, using conditions observed during the **current 6-hour interval**.

---

## 2. Forecast Target & Temporal Framing

- **Binary Classification Target (`target_delayed_next_6h`)**:
  $$\text{target\_delayed\_next\_6h} = \begin{cases} 1 & \text{if } \text{actual\_delay\_hours} > 12.0 \text{ in interval } t+1 \\ 0 & \text{otherwise} \end{cases}$$
- **Forecast Horizon**: The forecast is generated at the exact close of the current 6-hour observation window ($t$) and predicts congestion for the subsequent 6-hour horizon ($t \to t+6\text{h}$).
- **Target Alignment**: Forward shifting (`shift(-1)`) is computed strictly within port groups (`port_id`). The final observation record for each port lacks a subsequent interval and is dropped (5 terminal records removed; dataset transitions from 2,500 to 2,495 rows).
- **Target Leakage Prevention**: All current-interval delay and outcome fields (`delay_hours`, `delayed`, `actual_delay_hours`) are strictly purged from predictor features to ensure zero retrospective information leakage.

---

## 3. Dataset Description & Generation

The system utilizes a synthetic operational dataset generated to simulate realistic marine terminal queue dynamics across 5 major ports over consecutive 6-hour intervals from **January 1, 2026 to May 5, 2026**.

- **Raw Data Path**: `data/port_congestion_synthetic.csv` (2,500 records across 5 ports: `PORT_A` through `PORT_E`, 500 intervals each).
- **Prepared Forecasting Data Path**: `data/port_congestion_forecasting_dataset.csv` (2,495 records: 2,199 on-time [88.1%], 296 delayed [11.9%]).
- **Regeneration Command**:
  ```bash
  python data/generate_dataset.py
  ```

### Predictor Features (9 Inputs)
| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `vessel_arrivals_6h` | Numeric | Count of incoming vessels arriving during the 6-hour observation window. |
| `port_capacity_6h` | Numeric | Standard nominal handling capacity of the port per 6-hour window. |
| `arrival_load_ratio` | Numeric | Derived ratio: `vessel_arrivals_6h / port_capacity_6h` (captures operational load). |
| `queue_length` | Numeric | Number of queued vessels waiting for berth at the start of the window. |
| `effective_capacity_6h` | Numeric | Weather-adjusted operational berth capacity available during the window. |
| `historical_avg_delay_hours` | Numeric | Recent rolling average delay observed at the port prior to this interval. |
| `port_id` | Categorical | Unique terminal identifier (`PORT_A`, `PORT_B`, `PORT_C`, `PORT_D`, `PORT_E`). |
| `weather_condition` | Categorical | Prevailing environmental conditions (`Clear`, `Rain`, `Storm`). |
| `day_of_week` | Categorical | Day of observation (`Monday` through `Sunday`). |

---

## 4. Chronological Splitting & Modeling Approach

To prevent temporal lookahead leakage, data is partitioned using **strict chronological cutoffs** shared across all five ports:

```
[----------------- Train: 70% (1,745 rows) -----------------] [--- Val: 10% (250) ---] [--- Test: 20% (500) ---]
2026-01-01 00:00                             2026-03-29 00:00 2026-03-29 06:00    2026-04-10 12:00     2026-05-05 12:00
```

- **Training Set**: 1,745 samples (204 delays, 11.7% delay rate).
- **Validation Set**: 250 samples (39 delays, 15.6% delay rate) — used exclusively for threshold selection.
- **Held-out Test Set**: 500 samples (53 delays, 10.6% delay rate) — evaluated only once under fixed thresholds.

### Models
1. **Zero-Intelligence Baseline (`DummyClassifier`)**: Uses `strategy="most_frequent"`, constantly predicting the majority negative class (Class 0: On-time).
2. **Primary Model (`RandomForestClassifier`)**: 100 decision trees, `max_depth=6`, `min_samples_leaf=2`, `class_weight="balanced"`, wrapped in a scikit-learn `Pipeline` with a `ColumnTransformer` (`OneHotEncoder` for categoricals, passthrough for numeric features).

---

## 5. Measured Model Performance & Threshold Evaluation

### Validation Set Performance (250 samples, 39 delays)

| Model Configuration | Threshold | Accuracy | Precision | Recall | F1-Score | Delays Caught |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline (Always On-Time)** | N/A | 84.40% | 0.00% | 0.00% | 0.0000 | 0 / 39 |
| **RF (Operational Higher-Recall)** | **0.40** | **77.20%** | **37.84%** | **71.79%** | **0.4956** | **28 / 39** |
| **RF (Default Baseline)** | 0.50 | 83.60% | 48.00% | 61.54% | 0.5393 | 24 / 39 |
| **RF (Validation F1-Max)** | 0.55 | 87.20% | 60.00% | 53.85% | 0.5676 | 21 / 39 |

### Held-Out Test Set Performance (500 samples, 53 delays)

| Model Configuration | Threshold | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Delays Caught |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline (Always On-Time)** | N/A | 89.40% | 0.00% | 0.00% | 0.0000 | 0.5000 | 0 / 53 |
| **RF (Operational Higher-Recall)** | **0.40** | **78.00%** | **26.05%** | **58.49%** | **0.3605** | **0.7356** | **31 / 53** |
| **RF (Default Baseline)** | 0.50 | 85.80% | 35.94% | 43.40% | 0.3932 | 0.7356 | 23 / 53 |
| **RF (Validation F1-Max)** | 0.55 | 87.60% | 37.84% | 26.42% | 0.3111 | 0.7356 | 14 / 53 |

> **Operational Decision Rationale**: In marine container logistics, missing an unexpected congestion event leads to vessel queuing cascades, berth schedule collapse, and demurrage costs that vastly exceed the operational cost of an alert. **Threshold 0.40** is selected as the primary operating threshold because it catches **31 of 53 delays** on the held-out test set (58.5% recall), compared to only 23 delays caught at 0.50 and 14 delays caught at 0.55.

---

## 6. Installation & Execution Guide

### 1. Environment Setup
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. End-to-End Execution Pipeline
Execute each step sequentially from the project root:

```bash
# Step 1: Prepare forecasting dataset (align targets, eliminate boundary rows, strip leakage)
python src/prepare_data.py

# Step 2: Fit preprocessing and train Random Forest model with chronological splits
python src/train.py

# Step 3: Run threshold evaluation across candidate cutoffs (0.20 to 0.60)
python src/evaluate_thresholds.py

# Step 4: Generate situational prediction explanations for representative cases
python src/explain.py
```

### 3. Launching the Interactive Web Interface
Launch the Streamlit web application:
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`.

### 4. Running the Automated Test Suite
Execute all 17 automated regression and unit tests:
```bash
pytest -v
```

### 5. Compiling the Technical PDF Report
Compile the official 3-page executive technical report:
```bash
python reports/generate_pdf_report.py
```
The resulting PDF is saved to `reports/technical_report.pdf`.

### 6. Five-Minute Demonstration Walkthrough
Refer to [`docs/demo_plan.md`](docs/demo_plan.md) for a structured 5-minute presentation guide covering live dashboard interactions and scenario testing.

---

## 7. Project Structure

```
├── Architecture.md           # High-level architecture documentation
├── README.md                 # Project guide, performance benchmarks, and instructions
├── requirements.txt          # Python runtime dependencies
├── app.py                    # Interactive Streamlit prediction interface
├── data/
│   ├── generate_dataset.py                       # Synthetic operational log generator
│   ├── port_congestion_synthetic.csv             # Raw simulated terminal logs (2,500 rows)
│   ├── port_congestion_forecasting_dataset.csv   # Prepared forecasting dataset (2,495 rows)
│   └── port_congestion_validation_report.md      # Data profile and validation summary
├── docs/
│   ├── demo_plan.md                              # Five-minute interview / demonstration guide
│   ├── problem_definition_and_scope.md           # Problem framing and boundary conditions
│   ├── product_requirements_document.md          # User stories, operating threshold needs
│   ├── software_requirements_specification.md     # Pipeline functional & non-functional requirements
│   ├── ai_system_design_report.md                # System components and ML architecture
│   └── implementation_and_test_plan.md           # Verification and validation protocols
├── models/
│   └── random_forest_pipeline.joblib             # Serialized preprocessor + Random Forest model
├── reports/
│   ├── generate_pdf_report.py                    # ReportLab script generating executive PDF report
│   └── technical_report.pdf                      # Official 3-page technical PDF report
├── results/
│   ├── metrics.json                              # Machine-readable validation and test metrics
│   ├── test_predictions.csv                      # Test predictions, probabilities, and targets
│   ├── feature_importance.csv                    # Global MDI feature importance weights
│   ├── threshold_comparison.csv                  # Validation performance across candidate thresholds
│   ├── threshold_evaluation.json                 # Comparative evaluation of threshold policies
│   └── prediction_explanations.json              # Situational contextual summaries for test cases
├── src/
│   ├── prepare_data.py                           # Target alignment and data sanitation
│   ├── train.py                                  # Chronological partitioning and model training
│   ├── evaluate_thresholds.py                    # Decision threshold tuning and policy evaluation
│   └── explain.py                                # Situational context and explanation generator
└── tests/
    ├── conftest.py                               # Pytest project configuration
    └── test_pipeline.py                          # 17 automated regression and unit tests
```

---

## 8. Limitations & Operational Governance

1. **Synthetic Data Boundaries**: The underlying dataset was generated via synthetic simulation. While it captures queuing bottlenecks, weather impacts, and capacity shocks, it does not reflect real-world complexities such as live AIS tracking, crane maintenance outages, labor disputes, or dynamic tidal drafts.
2. **Uncalibrated Model Probabilities**: Random Forest probability outputs represent raw ensemble tree vote proportions. They are monotonic with risk but have not undergone Platt scaling or isotonic calibration, and should not be interpreted as certified real-world likelihoods.
3. **Situational Context vs. Causal Attribution**: Feature influences presented in the interface and explanation artifacts reflect global model sensitivities and historical correlations. They do not constitute mathematical local feature contributions (e.g., SHAP values) or establish real-world causality.
4. **Single-Step Forecast Horizon**: Predictions apply strictly to the immediate subsequent 6-hour interval. Multi-step autoregressive forecasts or multi-day forecasts require future modeling extensions.
