# Audit of Project Codebase

This document audits all pipeline components, noting modifications, enhancements, and design rationale across Core, Advanced, and Expert tiers.

## File Audit Summary

| Component | Status | Actions Taken | Rationale |
|---|---|---|---|
| `project/pipeline/config.py` | Kept & Enhanced | Added canonical output paths for hierarchy reconciliation, EV shift, macro lag, what-if scenario, and stockouts. Added `Unassigned` to canonical regions. | Ensures consistent output management and eliminates untracked string literals across the pipeline. |
| `project/pipeline/load.py` | Kept | Maintained clean data ingest functions with strictly read-only access to `student_data/`. | Strictly honors Ground Rule 1 (raw files immutable). |
| `project/pipeline/audit.py` | Kept | Standardized audit logging format across all cleaning steps. | Ensures full auditability for every data modification per Ground Rule 2. |
| `project/pipeline/clean_master_macro_events.py` | Enhanced | Resolved PMP-004 BU conflict to Advanced Manufacturing (Rule S1). Checked M2 2019-2020 continuity step (<25.0). Added Macro_Imputed and Is_COVID_Period flags. | Replaces uncleaned master ambiguities with deterministic business rules. |
| `project/pipeline/clean_train.py` | Enhanced | Replaced ambiguous 5-20x price band with annual SKU reference price. Assigned blank regions to Unassigned. Dropped matched negative unit reversals and aligned stockouts. Kept genuine multi-customer duplicate loads. | Eliminates ambiguous cases and enforces strict transactional integrity. |
| `project/pipeline/unconstrained.py` | Enhanced | Reconstructed true demand using requested units from inventory stockouts. Identified suspected unlisted stockouts via run-length drops. Exported cleaned stockouts to outputs. | Separates unconstrained market demand from operational supply constraints. |
| `project/pipeline/ordering_cycles.py` | Created | Classified ordering periodicity (monthly, bi-monthly, quarterly, irregular) per customer-SKU series. Flagged churned accounts. | Supports Expert tier customer cycle tracking and dynamic features. |
| `project/pipeline/ev_shift.py` | Created | Measured ICE vs EV sales progression across powertrain types. Evaluated EV adoption index lead-lag cross-correlations and regressions. | Fulfills Expert tier powertrain technological transition modeling. |
| `project/pipeline/structural_breaks.py` | Created | Quantified pre-announcement early buying (EV-030), regional holiday seasonality, and master data lifecycle integrity. | Isolates genuine structural market shocks from baseline trends. |
| `project/pipeline/features.py` | Rewritten | Fixed YoY growth feature calculation. Added customer cycle and intermittency features. Embedded dynamic holiday and event flags. | Eliminates duplicate features and provides rich context for tree-based models. |
| `project/pipeline/models/lightgbm_model.py` | Rewritten | Implemented early stopping validation with subsequent full-dataset refit. Enforced non-crossing quantile ordering. Removed silent fallback. | Maximizes historical data utilization and guarantees valid prediction intervals. |
| `project/pipeline/forecast.py` | Rewritten | Implemented recursive multi-step forecasting for horizons 1-19. Dynamically adjusted for forecast-window events. Reconciled hierarchy. Generated what-if scenario. | Prevents lag feature freeze and ensures top-down hierarchy consistency. |
| `project/pipeline/backtest.py` | Rewritten | Built 3-origin rolling backtest (2023-10, 2024-10, 2025-03) evaluating WAPE and Bias across all 19 horizon months, families, and regions. | Provides rigorous out-of-sample model verification. |
| `project/pipeline/anomalies.py` | Rewritten | Removed all mentions of pending items. Documented market contractions, early buying, and stockouts with explicit actions. | Complies with the complete decision coverage mandate. |
| `project/pipeline/reports.py` | Rewritten | Refactored pending review section into decision rules applied with full rationale. | Clear audit presentation for governance review. |
| `project/run_pipeline.py` | Rewritten | Unified end-to-end execution in one reproducible script with no silent errors or bare exceptions. | Satisfies Ground Rules 4, 5, and 6. |
| `project/app/dashboard.py` | Rewritten | Ensured all data is loaded exclusively from `outputs/`. Fixed filter population. Implemented 5 required views and 4 advanced views. | Satisfies Ground Rules 5 and 7. |
| `project/tests/test_cleaning_rules.py` | Enhanced | Added unit tests for Unassigned region, S1 BU resolution, YoY feature uniqueness, quantile ordering, and hierarchy tolerance. | Guarantees test coverage for every cleaning rule and model invariant. |
