# Five-Minute Demonstration Plan
## Port Congestion & Shipment Delay Forecaster

This demonstration walkthrough is designed for an interview or stakeholder presentation. It guides the presenter through the live Streamlit dashboard in under 5 minutes.

---

### Step 1: Launch the Application (30 seconds)
1. Open terminal in the project root.
2. Run the command:
   ```bash
   streamlit run app.py
   ```
3. Highlight the application header:
   - System objective: Predict whether a simulated commercial port will experience a **delay (>12 hours)** in the **upcoming six-hour operational window**.
   - Note the uncalibrated nature of tree voting probabilities and the 6-hour forecast cutoff.

---

### Step 2: Load a Preset Scenario (60 seconds)
1. In the sidebar under **⚙️ Preset Scenarios**, select **"Elevated Delay Risk"**.
2. Point out how the input fields automatically update to reflect an operational storm surge:
   - **Port Identifier**: `PORT_E`
   - **Observation Date**: `2026-04-25` (automatically deriving `Saturday`)
   - **Weather Condition**: `Storm`
   - **Vessel Arrivals in 6h**: `12`
   - **Nominal Capacity**: `9`
   - **Effective Capacity**: `6` (weather-reduced)
   - **Queue Backlog**: `8` vessels waiting at berth
   - **Historical Average Delay**: `8.5` hours
3. Highlight the calculated **Arrival Load Ratio** metric:
   - `1.333` (+33.3% above nominal capacity), colored in inverse/alert styling.

---

### Step 3: Change Operational Conditions (30 seconds)
1. Show responsiveness: Adjust the Observation Time Window dropdown from `06:00 to 12:00` to `18:00 to 24:00`.
2. Direct attention to the calculated forecast metadata card:
   - **Forecast Cutoff**: `2026-04-26 00:00` (demonstrating exact date-time arithmetic rolling into the next calendar day).
   - **Predicted Horizon**: `2026-04-26 00:00 to 2026-04-26 06:00`.

---

### Step 4: Generate Forecast & Interpret Results (75 seconds)
1. Click the primary button: **🔮 Generate Forecast for Next 6h Horizon**.
2. Review the forecast display card:
   - **Estimated Delay Probability**: `89.3%`
   - **Classification Banner**: `🚨 DELAY PREDICTED (Class 1)`
   - **Decision Threshold**: `0.40`
3. Explain the **Operational Threshold Rationale**:
   - The system defaults to **0.40** rather than 0.50 because maritime demurrage and berth congestion penalties make false negatives far costlier than false positives.
   - At threshold 0.40, the model achieves **71.8% recall** on validation data and catches **31 of 53 delays** on the held-out test set (vs. 23 at 0.50 and 14 at 0.55).

---

### Step 5: Review Operational Context & Limitations (60 seconds)
1. Scroll down to **3. Operational Context & Feature Influences**:
   - **Global Feature Importance**: Note that `queue_length` (0.2423), `historical_avg_delay_hours` (0.2064), and `arrival_load_ratio` (0.1603) explain over 60% of variance across the Random Forest trees.
   - **Entered Conditions vs. Baseline Benchmarks**: Explain how the table contrasts the user's input against the 1,745-row training baseline (Queue 8 vs. 1.04 baseline; Load 1.33 vs. 0.74 baseline).
2. Point out the **Non-Causal Governance Notice**:
   - Emphasize that the system provides situational context, not proven individual causation or fabricated local attribution scores.

---

### Step 6: Historical Performance Review (45 seconds)
1. Expand the bottom section: **📊 Historical Model Performance & Threshold Evaluation**.
2. Show that live prediction mode is strictly isolated from evaluation:
   - Compare the **Dummy Baseline** (0% recall, 0 delays caught) with the **Random Forest** at 0.40 (58.5% test recall, 31 delays caught).
3. Conclude with key operational takeaways:
   - The system serves as an early-warning filter for port dispatchers to pre-emptively divert vessels or mobilize contingency stevedores.
