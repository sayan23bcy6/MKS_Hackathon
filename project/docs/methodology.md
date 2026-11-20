# Methodology: Celeris Automobili Demand Forecasting System

## 1. Problem Definition and Scope

Celeris Automobili manufactures engineered automotive systems across seven product families, delivering to twelve major commercial customers across six global operating regions. The core objective is forecasting true **unconstrained demand** (underlying customer order volume, independent of temporary plant disruptions or inventory stockouts) across 58,219 SKU-Region-Customer combinations for the 19-month horizon from 2026-11 through 2028-05.

The source data exhibits real-world data quality issues across transactional history (`train.csv`), macroeconomic time series (`macro_indicators.csv`), product master definitions (`sku_master.csv`), and the operational calendar (`events_calendar.csv`). Our methodology establishes an explainable, rule-based data transformation pipeline coupled with gradient boosted quantile regression, multi-origin backtesting, recursive multi-step forecasting, and top-down reconciliation.

---

## 2. Data Cleaning Rules and Decision Framework

All data repairs are rule-governed, deterministic, and logged in `outputs/audit_log.csv`. In accordance with Ground Rule 3, every data quality issue has an explicit, documented resolution rule.

### Transactional Cleaning Rules (train.csv)

| Rule ID | Anomaly Identified | Repair Mechanism | Rationale |
|---|---|---|---|
| T1 | 6 distinct date format representations | Normalized via regex pattern matching to `YYYY-MM` | Standardizes primary temporal merge key across datasets. |
| T2 | Year/Month column mismatch with Date | Recalculated from Date string | Date column is designated authoritative per Rule R1. |
| T3 | Year typographical transposition (e.g., 2062) | Replaced with 2026 | Resolves input year digit transposition confirmed by surrounding transaction timestamps. |
| T4 | Duplicate Record IDs (identical load records) | Deduplicated by retaining latest `Load_Timestamp` | Resolves database re-ingestion duplication per Rule R2. |
| T5 | Missing region entries (952 rows) | Customer-date match inference; remaining unmatchable assigned to `Unassigned` | Retains total commercial volume without geographic distortion. |
| T6 | Sales channel casing and label variants | Mapped to `Direct OEM` and `Tier 1 Supplier` | Normalizes commercial sales routing labels. |
| T7 | Missing product families | Derived deterministically from three-letter SKU prefix | Enforces product catalog hierarchy consistency. |
| T8 | SKU ID formatting slip | Standardized to `PREFIX-NNN` with zero-padding | Restores referential integrity with `sku_master.csv`. |
| T9 | Missing SKU descriptions | Imputed from matching SKU records in `sku_master.csv` | Enriches master metadata for reporting and validation. |
| T10 | Customer name casing variations | Cleaned and mapped to canonical customer names | Unifies account-level order history. |
| T11 | Formatting slips and sign errors in Units Sold | Commas removed, dashes converted to null, paired negative reversals dropped, unpaired sign errors corrected via `abs()` | Differentiates zero demand from clerical sign errors and accounting reversals. |
| T12 | Pricing scaling anomalies (0.01x, 85x, 5-20x) | Replaced with annual SKU median price across regions | Addresses cent-scale recording, currency conversion shifts (INR/USD), and unit transcription slips. |
| T13 | Revenue calculation discrepancies | Recalculated as `Units_Sold * Unit_Price_USD` | Eliminates mathematical inconsistency following unit and price normalization. |
| R5 | Transactions outside active product lifecycle | Sales prior to Launch Date or subsequent to EOL Date zeroed | Enforces catalog lifecycle boundaries. |
| T4b / R9 | Date-SKU-Region duplicates | Retained genuine distinct customer accounts; dropped duplicate re-loads | Preserves commercial multi-customer order records. |

### Master Data and Macroeconomic Cleaning

- **Rule S1 (PMP-004 BU Alignment)**: Reconciled PMP-004 to `Advanced Manufacturing` based on agreement with transaction history majority and Rule R8.
- **Rule M1 (2022-06 Macro Duplicate)**: Selected observation with minimal distance to adjacent linear interpolation (May and July 2022).
- **Rule M2 (Auto Production Percentage Scaling)**: Scaled 2019 fractional values x100, verified continuity step from 2019-12 to 2020-01 is under 25.0 points.
- **Rule M3 (PMI Decimal Misplacement)**: Corrected May 2023 PMI from 4.79 to 47.9 via single-decimal shift.
- **Rule M4 (Missing Macro Indicator Imputation)**: Linear time interpolation applied with `Macro_Imputed` indicator flag.
- **Rule M5 (COVID Structural Break)**: Added `Is_COVID_Period` indicator for 2020-03 through 2021-06.

---

## 3. Unconstrained Demand Reconstruction

Real-world demand planning requires forecasting unconstrained customer demand rather than shipment volume constrained by inventory stockouts or logistics disruptions:
1. **Documented Stockouts**: Merged transaction history with `inventory_stockouts.csv`. Where orders were constrained, `Requested_Units` replaced `Shipped_Units` as the true demand target.
2. **Suspected Unlisted Stockouts**: Identified consecutive multi-month volume drops exceeding 50% of the series median without corroborating market disruption events. Adjusted to normal operating median and logged in the anomaly register.

---

## 4. Feature Engineering

1. **Calendar and Harmonics**: Annual seasonality harmonics ($\sin, \cos$ transforms of month), quarter index, year-end ordering push flags, and Q1 start adjustments.
2. **Regional Holiday Regimes**: Explicit holiday indicators for Lunar New Year (Asia Pacific), Ramadan and Eid (MEA, India), and European summer manufacturing shutdowns.
3. **Lags and Rolling Metrics**: Dynamic lags at 1, 2, 3, 6, and 12 months. Rolling means (3m, 6m, 12m) and rolling standard deviations.
4. **True YoY Growth Ratio**: Calculated as `(lag_1m - lag_12m) / (|lag_12m| + 1e-3)` to provide a normalized annual trajectory feature distinct from static lags.
5. **Customer Ordering Cycles**: Intermittency classification (ADI and $CV^2$) alongside customer-specific cycle detection (monthly, bi-monthly, quarterly, irregular) and churned account identification.
6. **Macroeconomic Lead-Lag Interactions**: Evaluated cross-correlations across lags 0-6 months for leading indicators (ISM PMI, Auto Production YoY, Raw Materials Index, EV Adoption Index).
7. **Operational Event Flags**: Parsed event calendar for plant retooling (EV-062), announced price rises (EV-030, EV-059, EV-065), and regional supply disruptions.

---

## 5. Forecasting Model Architecture

### Model Selection and Training Strategy
We employ LightGBM quantile regression to predict demand distribution parameters:
- **P50 Model**: Objective quantile regression at $\alpha=0.50$ providing median unbiased point forecasts.
- **P10 Model**: Quantile regression at $\alpha=0.10$ establishing the downside operational threshold.
- **P90 Model**: Quantile regression at $\alpha=0.90$ establishing the upside safety stock buffer.

To maximize signal retention without data leakage:
1. **Validation Stage**: Trained on history through 2025-03 with early stopping evaluated on the validation window (2025-04 through 2026-10) to determine optimal boosting iterations (`best_iteration`).
2. **Full-History Refit**: Refit final booster models on the complete historical dataset (2019-01 through 2026-10) using `best_iteration`. This ensures recent trends and operational dynamics from late 2025 and 2026 inform test period predictions.

### Recursive Multi-Step Forecasting
To avoid feature vector freeze and lag leakage across the 19-month horizon:
- Month 1 (`2026-11`): Uses known historical lags and rolling averages up to `2026-10`.
- Month $t > 1$: Lag 1 corresponds to the model's P50 prediction for month $t-1$. Rolling statistics update recursively across the predicted path.
- Non-crossing constraints ($P10 \le P50 \le P90$) are enforced mathematically.

### Post-Processing and Reconciliation
1. **Operational Adjustments**: Applied customer-specific shutdowns (EV-062 Orion Driveline NA zeroing in 2027-07 to 2027-08) and capacity expansions (EV-052 Helios India).
2. **Lifecycle Bounds**: Zeroed all forecast volume outside active SKU launch and EOL windows.
3. **Hierarchy Reconciliation**: Aggregated SKU-Region predictions to product family, region, business unit, and total volume. Verified bottom-up mathematical consistency within 0.5% tolerance.

---

## 6. Backtesting and Validation Framework

The modeling strategy is validated using rolling-origin backtests evaluated across three distinct economic regimes:
1. **Origin 1 (2023-10)**: 19-month forward evaluation through 2025-05.
2. **Origin 2 (2024-10)**: 19-month forward evaluation through 2026-05.
3. **Origin 3 (2025-03)**: 19-month forward evaluation through 2026-10.

Accuracy is assessed using Weighted Absolute Percentage Error (WAPE) and Mean Forecast Bias, benchmarked against seasonal naive and trailing moving average baselines across all 19 forecast horizons.
