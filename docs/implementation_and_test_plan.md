# Implementation and Test Plan
## Port Congestion and Shipping Delay Forecasting

## Phase 1 — Data contract
- [ ] Select the official generator and official CSV.
- [ ] Run the generator to a temporary output path and compare headers and rows with the intended CSV.
- [ ] Confirm 2,500 rows, five ports, and 500 observations per port.
- [ ] Check missing values, duplicate `(port_id, timestamp)` keys, allowed weather values, and chronological order.
- [ ] Confirm the `actual_delay_hours > 12` label rule.
- [ ] Document whether timestamps mark interval start or end.
- [ ] Decide which features are genuinely known at the forecast cutoff.

## Phase 2 — Forecast table
- [ ] Sort chronologically within each port.
- [ ] Create `target_delayed_next_6h` by shifting `is_delayed` by -1 within each port.
- [ ] Verify target alignment manually on sample rows.
- [ ] Confirm the final row per port has a missing next-interval target and is excluded.
- [ ] Confirm no target shift crosses port boundaries.
- [ ] Drop current outcome columns from predictors.
- [ ] Drop any feature unavailable at the cutoff.

## Phase 3 — Training and evaluation
- [ ] Create chronological train/validation/test sets with common time cutoffs.
- [ ] Check class counts in each split.
- [ ] Fit preprocessing only on training data.
- [ ] Train Dummy Classifier.
- [ ] Train Random Forest Classifier.
- [ ] Select threshold using validation data only.
- [ ] Calculate precision, recall, F1, confusion matrix, ROC-AUC where meaningful, and secondary accuracy.
- [ ] Compare primary model with baseline.
- [ ] Inspect false positives, false negatives, and per-port results.
- [ ] Assess probability calibration if feasible.

## Phase 4 — Prediction and explanation
- [ ] Implement a prediction function or CLI entry point.
- [ ] Return forecast cutoff, horizon, probability, and class.
- [ ] Generate global feature importance.
- [ ] Add local explanation if feasible.
- [ ] Check invalid input handling and explain limitations.

## Phase 5 — Documentation and delivery
- [ ] Pin dependencies and record Python version.
- [ ] Write README installation and execution instructions.
- [ ] Include a reproducible training/evaluation command.
- [ ] Generate the final metrics report from the actual run.
- [ ] Write the 2–3-page technical PDF report.
- [ ] Optional: prepare a demo of no more than five minutes.
- [ ] Verify repository cleanliness and final submission checklist.

## Test matrix
| ID | Test | Expected result |
|---|---|---|
| T-01 | Required column absent | Clear schema error |
| T-02 | Duplicate port/timestamp | Validation flags duplicates |
| T-03 | Incorrect delay label | Validation flags mismatch |
| T-04 | Unsorted observations | Data is sorted or validation fails clearly |
| T-05 | Target shift at port boundary | No cross-port target assignment |
| T-06 | Last row of each port | No target; excluded from modeling |
| T-07 | Outcome column in predictors | Pipeline rejects or removes it |
| T-08 | Feature unavailable at cutoff | Feature excluded before training |
| T-09 | Test timestamp earlier than training | Split validation fails |
| T-10 | Test set has only one class | ROC-AUC marked not meaningful; counts reported |
| T-11 | Missing inference field | Clear actionable error |
| T-12 | Repeat training with same config | Reproducible data split and results within expected library behavior |
