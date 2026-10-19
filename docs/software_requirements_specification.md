# Software Requirements Specification (SRS)
## Port Congestion and Shipping Delay Forecasting

**Status:** Proposed requirements baseline  
**System type:** Local Python ML prototype

## 1. Purpose
Specify the software behavior and quality constraints for a prototype that forecasts next-six-hour delay risk for simulated ports.

## 2. System overview
The software loads and validates synthetic port observations, constructs a next-interval target, builds a leakage-safe feature set, trains and evaluates classifiers, and generates predictions with probabilities and explanations.

### 2.1 Operating environment
- Python 3.x (exact supported version to be pinned during implementation).
- Pandas and NumPy for data handling.
- scikit-learn for preprocessing, classifiers, and evaluation.
- Optional explanation/calibration libraries only if justified and documented.
- Local execution; no external service required for the MVP.

## 3. Inputs
- Synthetic CSV containing timestamps, port identifiers, operational conditions, historical features, and current-interval outcome fields.
- A user-supplied or selected observation containing only permitted features available at the forecast cutoff.
- Configuration for file paths, random seed, time cutoffs, and model settings.

The actual CSV schema must be checked against the chosen generator before this SRS is treated as implementation-ready.

## 4. Outputs
- Forecast target for the next six-hour interval.
- Estimated probability of delay.
- Binary predicted class.
- Evaluation metrics and confusion matrix.
- Global feature importance and optional local explanation.
- Saved model and preprocessing pipeline.
- Validation errors and execution logs that help diagnose invalid inputs.

## 5. Functional requirements
- **SRS-FR-01:** The system shall load the configured CSV and verify required columns.
- **SRS-FR-02:** The system shall validate unique `(port_id, timestamp)` keys and chronological ordering within each port.
- **SRS-FR-03:** The system shall derive the delayed label using the defined threshold of `actual_delay_hours > 12`.
- **SRS-FR-04:** The system shall construct the forecast target from the next chronological observation of the same port.
- **SRS-FR-05:** The system shall exclude rows without a next-interval target from supervised training and testing.
- **SRS-FR-06:** The system shall exclude `actual_delay_hours`, current-row `is_delayed`, and any target-derived field from predictors.
- **SRS-FR-07:** The system shall exclude any feature that is not known at the forecast cutoff.
- **SRS-FR-08:** The system shall split data chronologically, with common time cutoffs across ports.
- **SRS-FR-09:** The system shall fit preprocessing only on training data.
- **SRS-FR-10:** The system shall train and evaluate a Dummy Classifier baseline.
- **SRS-FR-11:** The system shall train and evaluate a Random Forest Classifier.
- **SRS-FR-12:** The system shall report precision, recall, F1-score, confusion matrix, and ROC-AUC where meaningful; accuracy shall be secondary.
- **SRS-FR-13:** The system shall produce a probability estimate and predicted class for next-interval delay.
- **SRS-FR-14:** The system shall provide a readable explanation of influential features and shall not describe associations as proven causes.
- **SRS-FR-15:** The system shall save the trained pipeline and provide a documented inference entry point.
- **SRS-FR-16:** The system shall report missing columns, invalid categories, and incompatible input values clearly.

## 6. Data and modeling constraints
- The dataset is synthetic and must be labelled as such.
- The target is next-interval delay, not the current row's label.
- No random train/test split.
- No use of future observations or outcome fields as predictors.
- The final row for each port is not eligible for next-interval supervised target construction.
- Probability estimates must not be called calibrated confidence unless calibration is evaluated.
- Test metrics must be calculated from held-out test data and never invented or manually adjusted.

## 7. Quality requirements
- **Reliability:** Invalid input must fail with a clear error instead of silently producing a prediction.
- **Reproducibility:** Use fixed random seeds where applicable and record configuration.
- **Maintainability:** Separate data validation, target construction, preprocessing, training, evaluation, and inference.
- **Usability:** Provide simple commands and a README for installation, training, evaluation, and prediction.
- **Traceability:** Map test cases to functional requirements.
- **Integrity:** Preserve the official dataset and clearly identify generated artifacts.

## 8. Verification and testing
Tests shall cover:
1. Schema and data-type validation.
2. Duplicate and missing-value detection.
3. Chronological ordering per port.
4. Label threshold correctness.
5. Target shift alignment and no cross-port leakage.
6. Removal of the last row per port from target-bearing data.
7. Predictor exclusion rules.
8. Chronological split boundaries.
9. Baseline and model training.
10. Metric generation and class-count reporting.
11. Probability and explanation output structure.
12. Invalid inference input handling.
13. Repeatable execution from a clean environment.

## 9. Assumptions and risks
- The generator's timestamp semantics and each feature's availability need explicit verification.
- A next-row label shift alone does not guarantee valid forecasting if predictors describe the interval being predicted.
- The simulation may produce patterns that are easier to learn than real port operations.
- The dataset covers only five simulated ports and omits many real-world variables.
- The project is a prototype and is not approved for operational decisions.

## 10. Traceability
Requirements are linked to the PRD's functional requirements and the implementation checklist. The final test report should record pass/fail status and any deferred requirements.
