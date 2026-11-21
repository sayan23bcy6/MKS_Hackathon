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
- **Operational Events**: Factory retooling (EV-062), price rises (EV-059), capacity expansions (EV-052)

## Validation & Backtesting
- **Strategy**: Honest multi-origin rolling backtests across 3 cutoff points (2023-10, 2024-10, 2025-03) with 19-month recursive horizons.
- **No Leakage**: Features, cycles, and intermittency recomputed strictly on data up to each origin. Recursive lag propagation used for months 2-19.
- **Empirical P10-P90 Coverage**: 80.8% across 171,317 out-of-sample backtest predictions (target ~80%).

### Backtest Summary by Family
  - Brake & Safety Systems: Best model is `lightgbm_p50` (Mean WAPE: 0.423)
  - Chassis & Structural: Best model is `lightgbm_p50` (Mean WAPE: 0.415)
  - Electrical Assemblies: Best model is `lightgbm_p50` (Mean WAPE: 0.443)
  - Hydraulic Systems: Best model is `naive_seasonal` (Mean WAPE: 0.438)
  - Powertrain Components: Best model is `naive_seasonal` (Mean WAPE: 0.414)
  - Precision Machined Parts: Best model is `lightgbm_p50` (Mean WAPE: 0.418)
  - Thermal Management: Best model is `lightgbm_p50` (Mean WAPE: 0.414)

## Operational Controls
- Quantile non-crossing enforcement: P10 <= P50 <= P90.
- Lifecycle boundary zeroing: Strictly zero forecast demand prior to Launch_Date or after EOL_Date.
- New launch history inheritance: New SKUs with Replaces_SKU inherit replaced catalog history.
- Hierarchy reconciliation: Family and region bottom-up consistency reconciled within 0.5% tolerance.
- Deterministic training: random_state=42, deterministic=True, force_row_wise=True.
