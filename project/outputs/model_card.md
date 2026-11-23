# Model Card: Celeris Automobili Demand Forecasting System

## Model Details
- **Architecture**: LightGBM Quantile Gradient Boosted Trees (P10, P50, P90) as the global ML candidate
- **Forecast assembly**: each product family uses the model selected by the backtest rule (see the per-family section below)
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
- **Strategy**: Rolling-origin backtests across 3 cutoff points (2023-10, 2024-10, 2025-03) with 19-month recursive horizons.
- **No Leakage**: Features, cycles, and intermittency recomputed strictly on data up to each origin. Recursive lag propagation used for months 2-19.
- **P10-P90 Coverage**: Raw coverage 66.55%; Calibrated coverage 72.97% across 169229 (169,229) out-of-sample LightGBM predictions. Bands exist only for LightGBM rows.

### Benchmark Comparisons by Origin
  - Origin 2023-10: LightGBM WAPE 0.372 vs Seasonal Naive 0.358 (Loses to naive)
  - Origin 2024-10: LightGBM WAPE 0.330 vs Seasonal Naive 0.334 (Beats naive)
  - Origin 2025-03: LightGBM WAPE 0.333 vs Seasonal Naive 0.328 (Loses to naive)

### Backtest Summary by Family
  - Brake & Safety Systems: Best model is `naive_seasonal` (Mean WAPE: 0.296)
  - Chassis & Structural: Best model is `naive_seasonal` (Mean WAPE: 0.338)
  - Electrical Assemblies: Best model is `lightgbm_p50` (Mean WAPE: 0.365)
  - Hydraulic Systems: Best model is `naive_seasonal` (Mean WAPE: 0.348)
  - Powertrain Components: Best model is `naive_seasonal` (Mean WAPE: 0.337)
  - Precision Machined Parts: Best model is `lightgbm_p50` (Mean WAPE: 0.350)
  - Thermal Management: Best model is `lightgbm_p50` (Mean WAPE: 0.328)

## Operational Controls
- Quantile non-crossing enforcement: P10 <= P50 <= P90.
- Lifecycle boundary zeroing: Strictly zero forecast demand prior to Launch_Date or after EOL_Date.
- New launch history inheritance: Replaces_SKU first 6 months scaled by relative ratio, or family-region ratio before data exists.
- Hierarchy addition: Total forecast units: 3,004,640; adds_up: true; level rows checked: 18. Trend gap values are in hierarchy_diagnostic.csv and are diagnostic only.
- Deterministic training: random_state=42, deterministic=True, force_row_wise=True.

## Per-family model selection

Each family uses the model with the lowest mean WAPE over the selection origins 2023-10 and 2024-10. The 2025-03 origin is the holdout and was not used for the choice.

| Family | Selected model | Selection WAPE (selected) | Holdout WAPE 2025-03 (selected) |
|---|---|---:|---:|
| Brake & Safety Systems | naive_seasonal | 0.2977 | 0.2936 |
| Chassis & Structural | naive_seasonal | 0.3419 | 0.3301 |
| Electrical Assemblies | lightgbm_p50 | 0.3729 | 0.3488 |
| Hydraulic Systems | naive_seasonal | 0.3505 | 0.3432 |
| Powertrain Components | naive_seasonal | 0.3453 | 0.3205 |
| Precision Machined Parts | lightgbm_p50 | 0.3564 | 0.3361 |
| Thermal Management | lightgbm_p50 | 0.3355 | 0.3134 |
