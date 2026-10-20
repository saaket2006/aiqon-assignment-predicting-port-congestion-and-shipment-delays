# Implementation and Test Plan
## Port Congestion and Shipping Delay Forecasting

**Status:** Completed and Verified Baseline

## Phase 1 — Data contract
- [x] Select the official generator and official CSV (`data/generate_dataset.py`, `data/port_congestion_synthetic.csv`).
- [x] Run the generator to a temporary output path and compare headers and rows with the intended CSV.
- [x] Confirm 2,500 rows, five ports, and 500 observations per port.
- [x] Check missing values, duplicate `(port_id, timestamp)` keys, allowed weather values, and chronological order.
- [x] Confirm the `actual_delay_hours > 12` label rule.
- [x] Document whether timestamps mark interval start or end.
- [x] Decide which features are genuinely known at the forecast cutoff.

## Phase 2 — Forecast table
- [x] Sort chronologically within each port.
- [x] Create `target_delayed_next_6h` by shifting `is_delayed` by -1 within each port.
- [x] Verify target alignment manually on sample rows.
- [x] Confirm the final row per port has a missing next-interval target and is excluded (2,500 -> 2,495 rows).
- [x] Confirm no target shift crosses port boundaries.
- [x] Drop current outcome columns from predictors (`actual_delay_hours`, `is_delayed`).
- [x] Drop any feature unavailable at the cutoff.

## Phase 3 — Training and evaluation
- [x] Create chronological train/validation/test sets with common time cutoffs.
- [x] Check class counts in each split.
- [x] Fit preprocessing only on training data.
- [x] Train Dummy Classifier.
- [x] Train Random Forest Classifier.
- [x] Select threshold using validation data only (0.40 operational vs 0.50 default vs 0.55 F1-max).
- [x] Calculate precision, recall, F1, confusion matrix, ROC-AUC where meaningful, and secondary accuracy.
- [x] Compare primary model with baseline.
- [x] Inspect false positives, false negatives, and operational trade-offs.
- [x] Assess probability calibration and disclose uncalibrated tree-voting limitations.

## Phase 4 — Prediction and explanation
- [x] Implement a prediction function and interactive dashboard (`app.py`).
- [x] Return forecast cutoff, horizon, probability, and class.
- [x] Generate global feature importance.
- [x] Add situational context and benchmark comparisons for individual predictions.
- [x] Check invalid input handling and explain limitations.

## Phase 5 — Documentation and delivery
- [x] Pin dependencies and record Python version (`requirements.txt`).
- [x] Write README installation and execution instructions.
- [x] Include a reproducible training/evaluation command.
- [x] Generate the final metrics report from the actual run.
- [x] Write the 2–3-page technical PDF report.
- [x] Prepare a 5-minute demo walkthrough (`docs/demo_plan.md`).
- [x] Verify repository cleanliness and final submission checklist.

## Test matrix
| ID | Test | Status | Verification Mechanism |
|---|---|:---:|---|
| T-01 | Required column absent | **PASSED** | Validated in `tests/test_pipeline.py` schema checks. |
| T-02 | Duplicate port/timestamp | **PASSED** | Unique key validation in data preparation and generator. |
| T-03 | Incorrect delay label | **PASSED** | Tested in `test_synthetic_data_schema_and_boundaries` (`actual_delay_hours > 12`). |
| T-04 | Unsorted observations | **PASSED** | Explicit chronological sort by `[port_id, timestamp]` in `prepare_data.py`. |
| T-05 | Target shift at port boundary | **PASSED** | Tested in `test_prepared_dataset_target_alignment` (`groupby('port_id').shift(-1)`). |
| T-06 | Last row of each port | **PASSED** | Verified in `test_prepared_dataset_target_alignment` (5 boundary rows excluded). |
| T-07 | Outcome column in predictors | **PASSED** | Verified in `test_prepared_dataset_target_alignment` (`actual_delay_hours`, `is_delayed` dropped). |
| T-08 | Feature unavailable at cutoff | **PASSED** | Only 9 observable predictor features admitted. |
| T-09 | Test timestamp earlier than training | **PASSED** | Tested in `test_chronological_splitting_isolation` (strict temporal cutoff). |
| T-10 | Single-class split check | **PASSED** | Verified in `test_chronological_splitting_isolation` (all splits have delayed samples). |
| T-11 | Missing inference field | **PASSED** | Tested in model loading and `app.py` validation. |
| T-12 | Repeat training with same config | **PASSED** | Tested in `test_synthetic_generator_reproducibility` and model evaluation. |
