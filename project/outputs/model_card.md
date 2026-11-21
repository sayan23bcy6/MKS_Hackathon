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
- **Operational Events**: Retooling shutdown (EV-062), capacity expansion (EV-052 estimated from India sales), price change (EV-059 native feature)

## Validation & Backtesting
- **Strategy**: Honest multi-origin rolling backtests across 3 cutoff points (2023-10, 2024-10, 2025-03) with 19-month recursive horizons.
- **No Leakage**: Features, cycles, and intermittency recomputed strictly on data up to each origin. Recursive lag propagation used for months 2-19.
- **P10-P90 Coverage**: Raw coverage 66.2%; Calibrated coverage 72.6% across 171,317 out-of-sample predictions.

### Benchmark Comparisons by Origin
  - Origin 2023-10: LightGBM WAPE 0.396 vs Seasonal Naive 0.406 (Beats naive)
  - Origin 2024-10: LightGBM WAPE 0.360 vs Seasonal Naive 0.388 (Beats naive)
  - Origin 2025-03: LightGBM WAPE 0.361 vs Seasonal Naive 0.374 (Beats naive)

### Backtest Summary by Family
  - Brake & Safety Systems: Best model is `lightgbm_p50` (Mean WAPE: 0.357)
  - Chassis & Structural: Best model is `lightgbm_p50` (Mean WAPE: 0.378)
  - Electrical Assemblies: Best model is `lightgbm_p50` (Mean WAPE: 0.401)
  - Hydraulic Systems: Best model is `naive_seasonal` (Mean WAPE: 0.391)
  - Powertrain Components: Best model is `lightgbm_p50` (Mean WAPE: 0.375)
  - Precision Machined Parts: Best model is `lightgbm_p50` (Mean WAPE: 0.370)
  - Thermal Management: Best model is `lightgbm_p50` (Mean WAPE: 0.351)

## Operational Controls
- Quantile non-crossing enforcement: P10 <= P50 <= P90.
- Lifecycle boundary zeroing: Strictly zero forecast demand prior to Launch_Date or after EOL_Date.
- New launch history inheritance: Replaces_SKU first 6 months scaled by relative ratio, or family-region ratio before data exists.
- Hierarchy reconciliation: Top-level independent projections reconciled with maximum raw deviation of 16.4%.
- Deterministic training: random_state=42, deterministic=True, force_row_wise=True.
