# Model Card: Celeris Automobili Demand Forecasting System

## Model Details
- **Architecture**: LightGBM Quantile Gradient Boosted Trees (P10, P50, P90)
- **Training Horizon**: 2019-01 to 2026-10 (Full history refit following validation)
- **Forecast Horizon**: 2026-11 to 2028-05 (19 months, recursive multi-step forecasting)
- **Target Variable**: Units_Sold_Unconstrained (actual shipments adjusted for stockouts and event constraints)

## Feature Engineering
- **Calendar & Seasonality**: Month, quarter, cyclical harmonics, regional holidays (Lunar New Year, Ramadan, EU summer shutdown)
- **Lags & Rolling**: L1, L2, L3, L6, L12, rolling means (3m, 6m, 12m), rolling standard deviations, true YoY growth ratio
- **Intermittency & Cycles**: ADI/CV2 classification (Croston/SBA categorization) and customer ordering cycle length & churn detection
- **Macro Indicators**: ISM Manufacturing PMI, Auto Production YoY, Raw Materials Cost Index, EV Adoption Index (lags 0-3)
- **Operational Events**: Factory retooling (EV-062), price announcements (EV-030, EV-059, EV-065), supply constraints

## Validation & Backtesting
- **Strategy**: Multi-origin rolling backtests across 3 cutoff points (2023-10, 2024-10, 2025-03) with 19-month recursive horizons.
- **Key Metrics**: Overall WAPE and Mean Forecast Bias broken down by horizon, family, and region.
- **Empirical P10-P90 Coverage**: Not yet computed (run pipeline to generate backtest_predictions.csv).

## Operational Controls
- Quantile non-crossing enforcement: P10 <= P50 <= P90.
- Lifecycle boundary zeroing: Strictly zero forecast demand prior to Launch_Date or after EOL_Date.
- Hierarchy reconciliation: Top-down and bottom-up consistency verified within 0.5% tolerance.
- Deterministic training: random_state=42, deterministic=True, force_row_wise=True ensure identical outputs across runs.

## Reproducibility
- Run `python project/run_pipeline.py` twice from a clean state; outputs are bit-identical.
- Fixed seed and stable sort order throughout pipeline guarantee identical md5 for forecast_submission.csv, clean_train.parquet, and audit_log.csv.
