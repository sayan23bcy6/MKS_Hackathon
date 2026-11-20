# Model Card – Celeris Automobili Demand Forecasting

## Model: LightGBM Quantile Regression

## Features Used
### Calendar
- month, quarter, year, month_sin, month_cos (cyclical encoding)
- Holiday flags: Lunar New Year (Asia Pacific/India), Ramadan (MEA/India), Summer shutdown (Europe)
- Year-end, Q1-start flags

### Events
- Price change flag (all global price events from events_calendar)
- Supply constraint flag (EV-019 semiconductor shortage)
- Disruption flag (COVID events EV-009 to EV-013)
- System event flag (ERP migration EV-033)
- Early buying window flag (EV-030 announcement Nov 2022)
- Orion retooling flag (EV-062, 2027-07 to 2027-08)

### Lifecycle
- months_since_launch, months_to_eol
- is_replacement_sku, has_replacement

### Lag / Rolling
- Lag 1, 2, 3, 6, 12 months
- Rolling mean and std: 3m, 6m, 12m
- YoY lag (12m)

### Macro (lags 0–3 months)
- ISM Manufacturing PMI, Auto Production YoY%, Vehicle Inventory Days
- Raw Material Cost Index, New Vehicle Orders Index
- Industrial Utilisation Rate, Fuel Price Index, EV Adoption Index

### Encodings
- SKU_ID, Region, Customer_Name, Product_Family, Sales_Channel (label-encoded)
- Intermittency class (ADI/CV² classification)

## Training Strategy
- Time-based train/validation split at 2025-03
- Rolling-origin backtest with 19-month horizon
- Early stopping on validation WAPE

## Assumptions
1. Forecast is UNCONSTRAINED demand (R7) – supply limits do not apply
2. Stockout months use Requested_Units from inventory_stockouts as true demand
3. Missing rows are not zeros (R3); excluded from model training
4. R5 lifecycle rule applied: zero forecast before launch, after EOL
5. EV-062 Orion retooling: zero forecast for Orion Driveline NA in 2027-07 to 2027-08
6. EV-059 price rise 2027-01: forward-looking demand incorporated via event flag
7. New launches (PMP-020, PTC-029, THM-012) inherit history from Replaces_SKU

## Limitations
- ERP migration in Latin America (2023-03 to 2023-05) may have incomplete data; model may underforecast that region
- The 5–20× price ratio band (390 rows) remains unresolved; kept for demand, excluded from revenue
- 2022-06 macro duplicate resolved by neighbour interpolation – minor uncertainty in that month's macro features
- Extreme COVID period (2020-04: -45.4% Auto Production) is modelled via event flags, but deep uncertainty remains
- Intermittent SKU-region series (ADI>1.32) use Croston/SBA/TSB; LightGBM may not generalise well to these
- Forecast intervals (P10/P90) are quantile regression outputs – not prediction intervals in the strict frequentist sense

## Backtest Performance (19-month horizon, all SKU-Region-Customer)
| Model | WAPE | Bias |
|---|---|---|
| naive_seasonal | 0.4195 | -3.27 |
| moving_average | 0.5964 | 2.12 |
| lightgbm_p50 | 0.3290 | -2.90 |
