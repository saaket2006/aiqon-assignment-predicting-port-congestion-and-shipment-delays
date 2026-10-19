# Graph Report - aiqon-assignment-predicting-port-congestion-and-shipment-delays  (2026-10-19)

## Corpus Check
- 19 files · ~12,572 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 156 nodes · 173 edges · 16 communities (13 shown, 3 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `9891a194`
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
- generate_dataset.py
- rules/graphify.md
- workflows/graphify.md
- evaluate_threshold_range
- format_explanation
- app.py

## God Nodes (most connected - your core abstractions)
1. `AI System Design Report` - 12 edges
2. `Software Requirements Specification (SRS)` - 12 edges
3. `train_and_evaluate()` - 10 edges
4. `Port Congestion and Shipping Delay Project` - 10 edges
5. `Port Congestion and Shipping Delay Forecasting` - 10 edges
6. `Product Requirements Document (PRD)` - 10 edges
7. `get_project_root()` - 9 edges
8. `load_dataset()` - 9 edges
9. `split_data_chronologically()` - 9 edges
10. `Implementation and Test Plan` - 8 edges

## Surprising Connections (you probably didn't know these)
- `evaluate_thresholds()` --calls--> `get_project_root()`  [EXTRACTED]
  src/evaluate_thresholds.py → src/train.py
- `evaluate_thresholds()` --calls--> `load_dataset()`  [EXTRACTED]
  src/evaluate_thresholds.py → src/train.py
- `evaluate_thresholds()` --calls--> `split_data_chronologically()`  [EXTRACTED]
  src/evaluate_thresholds.py → src/train.py
- `load_explainer_assets()` --calls--> `get_project_root()`  [EXTRACTED]
  src/explain.py → src/train.py
- `load_explainer_assets()` --calls--> `load_dataset()`  [EXTRACTED]
  src/explain.py → src/train.py

## Import Cycles
- None detected.

## Communities (16 total, 3 thin omitted)

### Community 0 - "train.py"
Cohesion: 0.11
Nodes (31): evaluate_thresholds(), Decision Threshold Evaluation for Port Congestion Forecasting. This script…, Runs threshold analysis on validation data and evaluates on held-out test data., load_explainer_assets(), print_explanation_card(), Prediction Explanation Tool for Port Congestion Forecasting. This script…, Prints a clean, human-readable terminal explanation card., Runs prediction explanations on representative test set cases. (+23 more)

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

### Community 13 - "evaluate_threshold_range"
Cohesion: 0.40
Nodes (5): evaluate_threshold_range(), DataFrame, ndarray, Series, Evaluates classification performance across a range of probability thresholds.

### Community 14 - "format_explanation"
Cohesion: 0.50
Nodes (4): format_explanation(), DataFrame, Series, Generates a beginner-friendly explanation for a single prediction.

### Community 15 - "app.py"
Cohesion: 0.22
Nodes (9): load_evaluation_summary(), load_feature_importances(), load_model_pipeline(), Port Congestion & Delay Risk Forecaster - Streamlit Web Application This app…, Loads the trained Random Forest model and preprocessing pipeline once., Loads the global Random Forest feature importances., Loads evaluation comparison metrics if available., cache_data (+1 more)

## Knowledge Gaps
- **69 isolated node(s):** `graphify`, `Workflow: graphify`, `1. Major Components`, `2. Primary Data Stores & Artifacts`, `3. Data Flow` (+64 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `evaluate_threshold_range()` connect `evaluate_threshold_range` to `train.py`?**
  _High betweenness centrality (0.013) - this node is a cross-community bridge._
- **What connects `graphify`, `Workflow: graphify`, `1. Major Components` to the rest of the system?**
  _69 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `train.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10695187165775401 - nodes in this community are weakly interconnected._
- **Should `Product Requirements Document (PRD)` be split into smaller, more focused modules?**
  _Cohesion score 0.11764705882352941 - nodes in this community are weakly interconnected._
- **Should `AI System Design Report` be split into smaller, more focused modules?**
  _Cohesion score 0.125 - nodes in this community are weakly interconnected._
- **Should `Software Requirements Specification (SRS)` be split into smaller, more focused modules?**
  _Cohesion score 0.14285714285714285 - nodes in this community are weakly interconnected._