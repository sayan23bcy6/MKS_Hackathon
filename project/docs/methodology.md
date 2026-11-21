# Methodology: Celeris Automobili Demand Forecasting System

## 1. Problem Definition and Scope

Celeris Automobili manufactures engineered automotive systems across seven product families, delivering to twelve major commercial customers across six global operating regions. The core objective is forecasting true unconstrained demand (underlying customer order volume, independent of temporary plant disruptions or inventory stockouts) across 58,219 SKU-Region-Customer combinations for the 19-month horizon from 2026-11 through 2028-05.

The source data exhibits real-world data quality issues across transactional history (`train.csv`), macroeconomic time series (`macro_indicators.csv`), product master definitions (`sku_master.csv`), and the operational calendar (`events_calendar.csv`). Our methodology establishes an explainable, rule-based data transformation pipeline coupled with gradient boosted quantile regression, honest rolling-origin backtesting, recursive multi-step forecasting, and top-down reconciliation.

---

## 2. Data Cleaning Rules and Decision Framework

All data repairs are rule-governed, deterministic, and logged in `outputs/audit_log.csv`. Every data quality issue has an explicit, documented resolution rule.

### Transactional Cleaning Rules (train.csv)

| Rule ID | Anomaly Identified | Repair Mechanism | Rationale |
|---|---|---|---|
| T1 | 6 distinct date format representations | Normalized via regex pattern matching to `YYYY-MM` | Standardizes primary temporal merge key across datasets. |
| T2 | Year/Month column mismatch with Date | Recalculated from Date string | Date column is designated authoritative per Rule R1. |
| T3 | Year typographical transposition (e.g., 2062) | Replaced with 2026 | Resolves input year digit transposition confirmed by surrounding transaction timestamps. |
| T4 | Duplicate Record IDs (identical load records) | Deduplicated by retaining latest `Load_Timestamp` | Resolves database re-ingestion duplication per Rule R2. |
| T4b | 167 series-month rows with differing channels | Deduplicated strictly on `Date, SKU_ID, Region, Customer_Name` by retaining latest `Load_Timestamp` (Record_ID tie-break) | Ensures exactly one authoritative observation per customer series-month; clean train row count is exactly 244,321. |
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
| A2 | Multi-account stockout allocation | Proportional integer allocation by customer volume with remainder to largest customer | Guarantees that allocated customer units sum exactly to total reported stockout units. |

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
2. **Multi-Customer Key Allocation (Rule A2)**: In `inventory_stockouts.csv`, records are identified at the `Date-SKU-Region` level. For series where multiple commercial customer accounts operate in the same region, stockout volume is allocated across accounts proportionally based on their non-negative actual demand. Floor integer allocation is applied, with any rounding remainder assigned to the largest customer account, guaranteeing exact conservation of stockout units.
3. **Suspected Unlisted Stockouts**: Identified consecutive multi-month volume drops exceeding 50% of the series median without corroborating market disruption events. Adjusted to normal operating median and logged in the anomaly register.

---

## 4. Feature Engineering

1. **Calendar and Harmonics**: Annual seasonality harmonics (sin, cos transforms of month), quarter index, year-end ordering push flags, and Q1 start adjustments.
2. **Regional Holiday Regimes**: Explicit holiday indicators for Lunar New Year (Asia Pacific), Ramadan and Eid (MEA, India), and European summer manufacturing shutdowns.
3. **Lags and Rolling Metrics**: Dynamic lags at 1, 2, 3, 6, and 12 months. Rolling means (3m, 6m, 12m) and rolling standard deviations.
4. **True YoY Growth Ratio**: Calculated as `(lag_1m - lag_12m) / (|lag_12m| + 1e-3)` to provide a normalized annual trajectory feature distinct from static lags.
5. **Customer Ordering Cycles**: Intermittency classification (Syntetos-Boylan ADI and CV^2) alongside customer-specific cycle detection (monthly, bi-monthly, quarterly, irregular) and churned account identification.
6. **Macroeconomic Lead-Lag Interactions**: Evaluated cross-correlations across lags 0-6 months for leading indicators (ISM PMI, Auto Production YoY, Raw Materials Index, EV Adoption Index).
7. **Operational Event Flags**: Parsed event calendar for plant retooling (EV-062), announced price rises (EV-030, EV-059, EV-065), and regional supply disruptions.

---

## 5. Forecasting Model Architecture

### Model Selection and Training Strategy
We employ LightGBM quantile regression to predict demand distribution parameters:
- **P50 Model**: Quantile regression at alpha=0.50 providing median point forecasts.
- **P10 Model**: Quantile regression at alpha=0.10 establishing the downside operational threshold.
- **P90 Model**: Quantile regression at alpha=0.90 establishing the upside safety stock buffer.

To maximize signal retention without data leakage:
1. **Validation Stage**: Trained on history through 2025-03 with early stopping evaluated on the validation window (2025-04 through 2026-10) to determine optimal boosting iterations (`best_iteration`).
2. **Full-History Refit**: Refit final booster models on the complete historical dataset (2019-01 through 2026-10) using `best_iteration`. This ensures recent trends and operational dynamics from late 2025 and 2026 inform test period predictions.

### Recursive Multi-Step Forecasting
To avoid feature vector freeze and lag leakage across the 19-month horizon:
- Month 1 (`2026-11`): Uses known historical lags and rolling averages up to `2026-10`.
- Month t > 1: Lag 1 corresponds to the model's P50 prediction for month t-1. Rolling statistics update recursively across the predicted path.
- Non-crossing constraints (P10 <= P50 <= P90) are enforced mathematically.

### Post-Processing, Hierarchy Reconciliation, and Lifecycle Rules
1. **Operational Adjustments**: Applied customer-specific shutdowns (EV-062 Orion Driveline NA zeroing in 2027-07 to 2027-08) and capacity expansions (EV-052 Helios Auto Parts India, factor 1.0828).
2. **New Launch History Inheritance**: New SKUs with designated replacement relationships (`Replaces_SKU`) inherit historical demand patterns from the predecessor SKU prior to launch. Unmatched new launches adopt product family average demand profiles. Demand prior to the official `Launch_Date` is strictly zeroed.
3. **Lifecycle Bounds**: Zeroed all forecast volume outside active SKU launch and EOL windows.
4. **Hierarchy Reconciliation**: Forecasts are generated bottom-up across SKU-Region-Customer series and reconciled against independent top-level product family and regional projection models. Discrepancies are balanced proportionally and exported to `outputs/hierarchy_reconciliation.csv`.

---

## 6. Backtesting and Validation Framework

The modeling strategy is validated using an honest rolling-origin backtesting framework across three historical economic regimes:
1. **Origin 1 (2023-10)**: 19-month forward evaluation through 2025-05.
2. **Origin 2 (2024-10)**: 19-month forward evaluation through 2026-05.
3. **Origin 3 (2025-03)**: 19-month forward evaluation through 2026-10.

### Leakage Prevention Protocol
- **Strict Temporal Boundary**: For each origin cutoff, training data is strictly filtered to observations occurring on or before that origin date (`Date <= origin`).
- **Dynamic Feature Isolation**: Customer ordering cycle classification, Syntetos-Boylan intermittency parameters, and churn flags are recomputed strictly from historical observations up to that cutoff.
- **Recursive Forward Path**: Evaluated across 19 forward months where future lags are fed from model predictions rather than ground truth actuals.
- **Interval Predictions**: Detailed backtest predictions including `p10`, `p50`, and `p90` are logged in `outputs/backtest_predictions.csv` to allow rigorous coverage validation.
- **Benchmarks**: LightGBM is compared against seasonal naive, moving average, and Croston/SBA/TSB baselines on identical test rows.

---

## 7. Macroeconomic Sensitivity (What-If Analysis)

To stress-test supply chain resiliency under fluctuating economic conditions, the system evaluates demand sensitivity across a calibrated grid of ISM Manufacturing PMI shifts (-6.0 to +6.0 points in increments of 1.0):
- Model Sensitivity Reruns: The LightGBM quantile regression models evaluate modified macroeconomic feature vectors.
- Scenario Aggregations: Demand shifts are aggregated by product family and region, exported to `outputs/whatif_scenarios.csv`, and rendered through an interactive slider in the control tower dashboard.

---

## 8. Reproducibility and Deliverable Hashes

The pipeline is completely deterministic. With fixed seeds (`random_state=42`), `deterministic=True`, and stable sorting before export, identical outputs are produced on every run:

| Deliverable File | Description | SHA-256 / MD5 Digest |
|---|---|---|
| `forecast_submission.csv` | 19-month unconstrained forward forecast (58,219 rows) | `984692f768968bd3eea173d378a45c75` |
| `clean_train.parquet` | Cleaned transactional sales history (244,321 rows) | `5425c4efbcfd22510a442b6d3f9e4194` |
| `audit_log.csv` | Granular audit trail for all data transformations (106,226 rows) | `de057d5da661236b0eeb6b089b9ecf78` |
| `backtest_predictions.csv` | Honest rolling-origin out-of-sample predictions (685,268 rows) | `94cce1a0bb27266d30a62e3d18eca90e` |
| `backtest_metrics.csv` | Slice-level backtest evaluations across horizons and families (456 rows) | `664ccfa6aaaf30d196b8ce87bfdb6043` |

