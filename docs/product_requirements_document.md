# Product Requirements Document (PRD)
## Port Congestion and Shipping Delay Forecasting

**Status:** Proposed MVP  
**Audience:** Project implementers and reviewers

## 1. Product vision
Build a small, reproducible AI prototype that helps an analyst anticipate simulated port delay risk for the next six-hour interval and understand the factors associated with the forecast.

## 2. Users and needs
### Primary user
A port operations analyst or logistics planner reviewing conditions at a simulated port.

### User needs
- See whether the next six-hour interval is predicted to be delayed.
- Understand the estimated probability rather than seeing only a hard label.
- See the main factors associated with the prediction.
- Understand model limitations and whether the probability has been calibrated.
- Reproduce the results from the repository.

## 3. MVP user journey
1. The user selects or supplies a port observation available at the forecast cutoff.
2. The system validates the input and prepares model features.
3. The model estimates the probability of delay in the following six-hour interval.
4. The system returns the probability, risk label, and an explanation.
5. The user can inspect model evaluation metrics and limitations.

## 4. Functional requirements
- **FR-01 Data loading:** Load the official synthetic CSV from a documented path.
- **FR-02 Schema validation:** Fail clearly if required columns are missing or unexpected data types/categories are found.
- **FR-03 Forecast target:** Create the next-six-hour target by shifting the delayed label within each port's chronological sequence.
- **FR-04 Leakage prevention:** Exclude actual outcome fields and any feature unavailable at forecast time.
- **FR-05 Chronological evaluation:** Create train, validation, and test sets using time cutoffs, not a random split.
- **FR-06 Baseline:** Train a Dummy Classifier.
- **FR-07 Primary model:** Train a Random Forest Classifier using a reproducible configuration.
- **FR-08 Probability output:** Return the estimated probability of next-interval delay.
- **FR-09 Risk label:** Return a binary classification using a threshold selected on validation data or a documented default.
- **FR-10 Explanation:** Report global feature importance and, if feasible, a local explanation for individual predictions.
- **FR-11 Evaluation report:** Record metrics, confusion matrix, class counts, and baseline comparison.
- **FR-12 Reproducibility:** Document dependencies, seed, training command, and prediction command.
- **FR-13 Input errors:** Provide actionable errors for missing, invalid, or unsupported input fields.

## 5. Non-functional requirements
- **NFR-01 Interpretability:** Explain predictions in plain language and avoid unsupported causal claims.
- **NFR-02 Reproducibility:** Re-running the workflow with the same inputs and configuration should produce repeatable results within expected library behavior.
- **NFR-03 Maintainability:** Keep data validation, feature preparation, training, evaluation, and inference logically separated.
- **NFR-04 Portability:** Run locally using documented Python dependencies.
- **NFR-05 Honest communication:** Clearly identify synthetic data and avoid implying real-world validation.
- **NFR-06 Data integrity:** Never overwrite the official dataset with a different schema without an explicit validation step.

## 6. Product outputs
MVP outputs:
- Next-six-hour estimated delay probability.
- Predicted class (delayed / not delayed).
- Explanation of influential factors.
- Model evaluation metrics and confusion matrix.
- Saved model/preprocessing pipeline or reproducible training artifact.
- README and technical report.

A web dashboard is optional and should be considered only after the core pipeline works.

## 7. Acceptance criteria
- A clean run loads the intended CSV and passes schema checks.
- The next-period target is correctly aligned within each port.
- No current-row outcome or future-derived feature enters model predictors.
- Baseline and Random Forest results are produced on a chronological test set.
- The report includes metrics from the actual run and the number of positive examples in each split.
- The inference path returns the probability, class, and explanation.
- Setup and execution instructions work in a clean environment.
- Limitations of synthetic data are stated in the README and report.

## 8. Prioritization
### Must-have
Target alignment, leakage-safe predictors, chronological evaluation, baseline, Random Forest, metrics, probability output, basic explanation, README, technical report.

### Should-have
Calibration assessment, per-port evaluation, error analysis, robust input validation.

### Could-have
Interactive dashboard, richer local explanations, additional model comparisons.

### Won't-have for this assignment
Live AIS integration, production hosting, real-time streaming, automatic operational decisions.
