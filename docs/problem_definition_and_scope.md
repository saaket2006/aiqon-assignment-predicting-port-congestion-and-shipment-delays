# Port Congestion and Shipping Delay Forecasting
## Problem Definition and Project Scope

**Document status:** Planning baseline  
**Version:** 1.0  
**Purpose:** Establish a shared problem definition before implementation.

## 1. Executive summary
This project prototypes an interpretable AI system that forecasts whether a simulated port is likely to experience a shipping delay during the next six-hour interval. It estimates a delay probability, produces a risk classification, and explains the input factors associated with its prediction.

The system is designed as a time-boxed take-home assignment prototype, not a production port operations platform.

## 2. Problem statement
Port congestion can result from vessel-arrival pressure, limited capacity, accumulated queues, weather disruptions, and recent delay patterns. A system that only classifies a period after its outcome is known is descriptive rather than predictive. The project therefore adopts a forward-looking objective: use information available at a defined forecast cutoff to predict the next interval's delay status.

### 2.1 Primary question
Given the port conditions known at the forecast cutoff, what is the estimated probability that the same port's following six-hour interval will have a simulated delay greater than 12 hours?

### 2.2 Intended user and decision support
The intended conceptual user is a port operations analyst or logistics planner who wants an early risk signal for prioritizing review or operational planning. The prototype does not issue automatic operational commands and does not replace expert judgment.

## 3. Objectives
1. Generate or load a reproducible synthetic dataset representing multiple ports over consecutive six-hour intervals.
2. Forecast next-interval delay risk.
3. Produce a delay probability and a clearly defined binary risk classification.
4. Explain which factors influence the model's prediction, without implying causation.
5. Evaluate the model using time-ordered held-out data and compare it with a simple baseline.
6. Document assumptions, limitations, reproducibility, and setup instructions.

## 4. Out of scope
- Live AIS ingestion or real-time port integrations.
- Production deployment, authentication, multi-user access, or automated operational control.
- Claims of real-world accuracy based only on synthetic data.
- Detailed vessel-level ETA prediction.
- Optimization of berth allocation, workforce, or port schedules.
- A complex dashboard before the model pipeline is validated.

## 5. Definitions
- **Observation interval:** A six-hour interval represented by a dataset row.
- **Forecast cutoff:** The instant at which the system is assumed to have received all permitted input information.
- **Forecast horizon:** The following six-hour interval after the forecast cutoff.
- **Delay outcome:** The synthetic interval's `actual_delay_hours`.
- **Delayed label:** `1` if the interval's delay is greater than 12 hours; otherwise `0`.
- **Next-interval target:** The delayed label for the next chronological row of the same port.
- **Estimated delay probability:** The model's estimated probability that the next-interval target is delayed.
- **Confidence/calibration:** Probability estimates are not automatically calibrated confidence; calibration must be assessed separately.

## 6. Success criteria
The prototype is considered functionally successful when it:
- Builds a next-interval target without crossing port boundaries.
- Excludes unavailable future information and outcome-derived fields from predictors.
- Trains a baseline and a primary classifier.
- Reports precision, recall, F1-score, confusion matrix, and ROC-AUC when meaningful; accuracy is secondary.
- Produces probability estimates and an explanation of influential factors.
- Runs reproducibly using documented steps.
- Includes a concise technical report and repository instructions.

No minimum model-performance score is promised in advance. Results must be reported honestly, even if performance is weak.

## 7. Data and assumption status
The current plan uses a synthetic dataset with five simulated ports and 500 observations per port, for 2,500 rows. The generator uses probabilistic arrivals, nominal capacity, weather-adjusted effective capacity, queue persistence, recent delay history, noise, and occasional disruption events. The exact CSV schema must be verified against the selected generator before training.

The simulation's ground-truth delay is generated from designed rules. Consequently, performance reflects the model's ability to learn those simulated rules and does not establish real-world operational validity.

## 8. Key design decision requiring verification
The generator describes six-hour observations, but forecasting requires a precise statement of what is known at the cutoff. Features such as `vessel_arrivals_6h`, `arrival_load_ratio`, `effective_capacity_6h`, and `weather_condition` may describe the interval being predicted rather than information already available before it. Before training, the implementation must:
1. Define whether each timestamp marks interval start or interval end.
2. Define the forecast cutoff explicitly.
3. Include only predictors available by that cutoff.
4. Shift the target forward within each port.
5. Exclude the final row per port because no next-interval target exists.

If the existing features cannot support a genuine forecast, either exclude them or revise the data construction so the inputs represent current/known conditions and the target represents the following interval. Do not silently claim a forecast if the data alignment is only retrospective.
