# Graph Report - aiqon-assignment-predicting-port-congestion-and-shipment-delays  (2026-10-20)

## Corpus Check
- 24 files · ~18,013 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 221 nodes · 242 edges · 20 communities (18 shown, 2 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `08735af7`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- train.py
- Product Requirements Document (PRD)
- AI System Design Report
- Software Requirements Specification (SRS)
- Port Congestion and Shipping Delay Forecasting
- Port Congestion and Shipping Delay Project
- Implementation and Test Plan
- get_project_root
- High-Level System Structure
- test_pipeline.py
- rules/graphify.md
- workflows/graphify.md
- evaluate_threshold_range
- format_explanation
- app.py
- Port Congestion and Shipping Delay Forecaster
- Port Congestion & Shipment Delay Forecaster
- NumberedCanvas

## God Nodes (most connected - your core abstractions)
1. `AI System Design Report` - 12 edges
2. `Software Requirements Specification (SRS)` - 12 edges
3. `load_dataset()` - 11 edges
4. `split_data_chronologically()` - 11 edges
5. `train_and_evaluate()` - 10 edges
6. `Port Congestion and Shipping Delay Project` - 10 edges
7. `Port Congestion and Shipping Delay Forecasting` - 10 edges
8. `Product Requirements Document (PRD)` - 10 edges
9. `get_project_root()` - 9 edges
10. `Port Congestion and Shipping Delay Forecaster` - 9 edges

## Surprising Connections (you probably didn't know these)
- `test_synthetic_generator_reproducibility()` --calls--> `generate_dataset()`  [EXTRACTED]
  tests/test_pipeline.py → data/generate_dataset.py
- `test_chronological_splitting_isolation()` --calls--> `load_dataset()`  [EXTRACTED]
  tests/test_pipeline.py → src/train.py
- `test_chronological_splitting_isolation()` --calls--> `split_data_chronologically()`  [EXTRACTED]
  tests/test_pipeline.py → src/train.py
- `evaluate_thresholds()` --calls--> `get_project_root()`  [EXTRACTED]
  src/evaluate_thresholds.py → src/train.py
- `evaluate_thresholds()` --calls--> `load_dataset()`  [EXTRACTED]
  src/evaluate_thresholds.py → src/train.py

## Import Cycles
- None detected.

## Communities (20 total, 2 thin omitted)

### Community 0 - "train.py"
Cohesion: 0.10
Nodes (33): evaluate_thresholds(), Decision Threshold Evaluation for Port Congestion Forecasting. This script…, Runs threshold analysis on validation data and evaluates on held-out test data., load_explainer_assets(), print_explanation_card(), Prediction Explanation Tool for Port Congestion Forecasting. This script…, Prints a clean, human-readable terminal explanation card., Runs prediction explanations on representative test set cases. (+25 more)

### Community 1 - "Product Requirements Document (PRD)"
Cohesion: 0.12
Nodes (16): 1. Product vision, 2. Users and needs, 3. MVP user journey, 4. Functional requirements, 5. Non-functional requirements, 6. Product outputs, 7. Acceptance criteria, 8. Prioritization (+8 more)

### Community 2 - "AI System Design Report"
Cohesion: 0.12
Nodes (15): 10. Limitations, 1. Summary, 2. End-to-end design, 3. Candidate predictors, 4. Modeling approach, 5. Evaluation design, 6. Probability and explanations, 7. Operational output (+7 more)

### Community 3 - "Software Requirements Specification (SRS)"
Cohesion: 0.14
Nodes (13): 10. Traceability, 1. Purpose, 2.1 Operating environment, 2. System overview, 3. Inputs, 4. Outputs, 5. Functional requirements, 6. Data and modeling constraints (+5 more)

### Community 4 - "Port Congestion and Shipping Delay Forecasting"
Cohesion: 0.15
Nodes (12): 1. Executive summary, 2.1 Primary question, 2.2 Intended user and decision support, 2. Problem statement, 3. Objectives, 4. Out of scope, 5. Definitions, 6. Success criteria (+4 more)

### Community 5 - "Port Congestion and Shipping Delay Project"
Cohesion: 0.18
Nodes (10): 1. Dataset overview, 2. Dataset schema, 3. Structural and integrity checks, 4. Port-level summary, 5. Delay behavior by weather, 6. Feature-to-outcome correlations, 7. Modeling precautions, 8. Conclusion (+2 more)

### Community 6 - "Implementation and Test Plan"
Cohesion: 0.22
Nodes (8): Implementation and Test Plan, Phase 1 — Data contract, Phase 2 — Forecast table, Phase 3 — Training and evaluation, Phase 4 — Prediction and explanation, Phase 5 — Documentation and delivery, Port Congestion and Shipping Delay Forecasting, Test matrix

### Community 7 - "get_project_root"
Cohesion: 0.29
Nodes (7): get_project_root(), prepare_forecasting_dataset(), DataFrame, Path, Data Preparation Script for Port Congestion Forecasting. Prepares the…, Returns the absolute path to the project root directory., Loads raw port records and transforms them into a next-interval forecasting…

### Community 8 - "High-Level System Structure"
Cohesion: 0.29
Nodes (6): 1. Major Components, 2. Primary Data Stores & Artifacts, 3. Data Flow, 4. Major Dependencies & Execution Boundaries, High-Level System Structure, System Architecture

### Community 9 - "test_pipeline.py"
Cohesion: 0.08
Nodes (24): generate_dataset(), Reproducible synthetic dataset generator for the Aiqon port congestion…, parametrize, Automated Test Suite for Port Congestion & Shipment Delay Forecasting System.…, Verify saved model pipeline loads and outputs valid probabilities., Verify threshold classification correctly assigns class 0 and 1., Verify zero port capacity is caught without raising ZeroDivisionError., Verify weekday is accurately derived from date. (+16 more)

### Community 13 - "evaluate_threshold_range"
Cohesion: 0.40
Nodes (5): evaluate_threshold_range(), DataFrame, ndarray, Series, Evaluates classification performance across a range of probability thresholds.

### Community 14 - "format_explanation"
Cohesion: 0.50
Nodes (4): format_explanation(), DataFrame, Series, Generates a beginner-friendly explanation for a single prediction.

### Community 15 - "app.py"
Cohesion: 0.22
Nodes (9): load_evaluation_summary(), load_feature_importances(), load_model_pipeline(), Port Congestion & Delay Risk Forecaster - Streamlit Web Application This app…, Loads the trained Random Forest model and preprocessing pipeline once., Loads the global Random Forest feature importances., Loads evaluation comparison metrics if available., cache_data (+1 more)

### Community 16 - "Port Congestion and Shipping Delay Forecaster"
Cohesion: 0.10
Nodes (20): 1. Environment Setup, 1. Problem Statement & Project Objective, 2. End-to-End Execution Pipeline, 2. Forecast Target & Temporal Framing, 3. Dataset Description & Generation, 3. Launching the Interactive Web Interface, 4. Chronological Splitting & Modeling Approach, 4. Running the Automated Test Suite (+12 more)

### Community 17 - "Port Congestion & Shipment Delay Forecaster"
Cohesion: 0.20
Nodes (8): Five-Minute Demonstration Plan, Port Congestion & Shipment Delay Forecaster, Step 1: Launch the Application (30 seconds), Step 2: Load a Preset Scenario (60 seconds), Step 3: Change Operational Conditions (30 seconds), Step 4: Generate Forecast & Interpret Results (75 seconds), Step 5: Review Operational Context & Limitations (60 seconds), Step 6: Historical Performance Review (45 seconds)

### Community 18 - "NumberedCanvas"
Cohesion: 0.28
Nodes (3): NumberedCanvas, Script to generate the official 3-page Technical PDF Report using ReportLab.…, Canvas that computes total pages dynamically for 'Page X of Y' footers.

## Knowledge Gaps
- **89 isolated node(s):** `graphify`, `Workflow: graphify`, `1. Major Components`, `2. Primary Data Stores & Artifacts`, `3. Data Flow` (+84 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `split_data_chronologically()` connect `train.py` to `test_pipeline.py`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Why does `load_dataset()` connect `train.py` to `test_pipeline.py`?**
  _High betweenness centrality (0.028) - this node is a cross-community bridge._
- **Why does `Port Congestion and Shipping Delay Forecaster` connect `Port Congestion and Shipping Delay Forecaster` to `Port Congestion & Shipment Delay Forecaster`?**
  _High betweenness centrality (0.014) - this node is a cross-community bridge._
- **What connects `graphify`, `Workflow: graphify`, `1. Major Components` to the rest of the system?**
  _89 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `train.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1 - nodes in this community are weakly interconnected._
- **Should `Product Requirements Document (PRD)` be split into smaller, more focused modules?**
  _Cohesion score 0.11764705882352941 - nodes in this community are weakly interconnected._
- **Should `AI System Design Report` be split into smaller, more focused modules?**
  _Cohesion score 0.125 - nodes in this community are weakly interconnected._