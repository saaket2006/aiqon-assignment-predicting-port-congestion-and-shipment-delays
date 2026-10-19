# AI System Design Report
## Port Congestion and Shipping Delay Forecasting

**Status:** Design baseline pending data-alignment verification

## 1. Summary
The proposed system estimates the probability that a simulated port will experience a delay greater than 12 hours in the next six-hour interval. It is designed to provide a risk label, probability estimate, and interpretable factors, with evaluation based on chronologically held-out data.

## 2. End-to-end design
1. Load the selected synthetic CSV.
2. Validate schema, missing values, duplicate port-timestamp keys, chronological ordering, allowed categories, and label consistency.
3. Confirm what each timestamp means and which features are available at the forecast cutoff.
4. Construct `target_delayed_next_6h` by shifting the current row's `is_delayed` one row forward within each port after sorting by timestamp.
5. Remove the last observation for each port because its next-interval label is unknown.
6. Select only features available at the cutoff; exclude `actual_delay_hours`, current-row `is_delayed`, and all target-derived fields.
7. Split chronologically using common timestamp cutoffs across ports.
8. Fit preprocessing on training data only.
9. Train a Dummy Classifier baseline and Random Forest primary model.
10. Evaluate on validation/test data and generate prediction explanations.

## 3. Candidate predictors
Potential candidates include `port_id`, `port_capacity_6h`, `queue_length`, `day_of_week`, and weather/arrival/capacity variables only if they are known by the forecast cutoff. In particular, `vessel_arrivals_6h`, `arrival_load_ratio`, `effective_capacity_6h`, and `weather_condition` must be checked against timestamp semantics. If they describe the interval being predicted, they are not valid predictors for a forecast issued before that interval begins.

`historical_avg_delay_hours` is permitted only if computed exclusively from prior observations available at the cutoff. `actual_delay_hours` and current-row `is_delayed` are outcomes and must not be predictors.

## 4. Modeling approach
### Baseline
Use scikit-learn `DummyClassifier` to establish a simple reference score.

### Primary model
Use scikit-learn `RandomForestClassifier` for nonlinear relationships in tabular features. Fix the random seed and document key parameters. Avoid extensive tuning due to the short project schedule.

### Preprocessing
Use a scikit-learn pipeline with a `ColumnTransformer` or equivalent to encode categorical features and pass numeric features appropriately. Fit transformations on training data only.

## 5. Evaluation design
Use time-ordered train/validation/test sets with common timestamp cutoffs across ports. The initial proposed proportions are 70%/10%/20%, subject to checking positive-class counts. Validation data may be used to choose a classification threshold or perform limited tuning. The test set is held back for final evaluation.

Report:
- Precision
- Recall
- F1-score
- ROC-AUC when both classes and sufficient examples are present
- Confusion matrix
- Accuracy as a secondary measure
- Positive and negative case counts by split
- Comparison against the Dummy Classifier
- Per-port performance where sample counts permit

Do not claim a model is successful solely because accuracy is high. Do not invent results before the final run.

## 6. Probability and explanations
The model's positive-class probability is presented as an estimated next-six-hour delay probability. It must not automatically be described as calibrated confidence. Assess calibration if feasible and disclose limitations.

Use global feature importance to describe which features the model relies on overall. If time permits, add local explanations for individual forecasts. Explanations should use language such as “associated with” or “contributed to this model prediction,” not causal claims.

## 7. Operational output
A prediction record should contain:
- Port identifier and forecast cutoff timestamp.
- Forecast horizon (following six-hour interval).
- Estimated delay probability.
- Predicted class.
- Main explanatory factors, if available.
- Model version/configuration identifier where practical.

## 8. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Future-data leakage | Define forecast cutoff and audit every feature |
| Incorrect target shift | Shift within port and test alignment explicitly |
| Synthetic-to-real gap | Clearly state results apply only to the simulation |
| Class imbalance | Report precision/recall/F1 and class counts |
| Misleading probability | Assess calibration and label estimates carefully |
| Schema drift | Validate CSV columns against generator output |
| Overly complex scope | Prioritize the local model pipeline over a dashboard |

## 9. Implementation gate
Do not start model training until:
1. The exact generator and CSV schema are reconciled.
2. Timestamp start/end semantics are documented.
3. Candidate feature availability is reviewed.
4. The next-interval target is unit-tested.
5. A chronological split is verified to contain enough delayed examples.

## 10. Limitations
This is a synthetic-data prototype. Its results cannot establish performance on live port data. The simulation simplifies operations and omits many variables present in real logistics systems. The report must be updated after implementation with actual results and any deviations from this design.
