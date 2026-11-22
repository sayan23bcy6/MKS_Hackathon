# Methodology: Celeris Automobili Demand Forecasting System

## 1. Problem Definition and Scope

Celeris Automobili manufactures engineered automotive systems across seven product families, delivering to twelve major commercial customers across six global operating regions. The core objective is forecasting true unconstrained demand (underlying customer order volume, independent of temporary plant disruptions or inventory stockouts) across 58,219 forecast rows for the 19-month horizon from 2026-11 through 2028-05.

The source data exhibits real-world data quality issues across transactional history (`train.csv`), macroeconomic time series (`macro_indicators.csv`), product master definitions (`sku_master.csv`), and the operational calendar (`events_calendar.csv`). Our methodology establishes an explainable, rule-based data transformation pipeline coupled with gradient boosted quantile regression, rolling-origin backtesting, recursive multi-step forecasting, and exact hierarchy addition reporting.

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
| T5 | Missing region entries (942 rows) | Customer-date match inference; remaining unmatchable assigned to `Unassigned` | Retains total commercial volume without geographic distortion. |
| T6 | Sales channel casing and label variants | Mapped to `Direct OEM` and `Tier-1 Distributor` | Normalizes commercial sales routing labels. |
| T7 | Missing product families | Derived deterministically from three-letter SKU prefix | Enforces product catalog hierarchy consistency. |
| T8 | SKU ID formatting slip | Standardized to `PREFIX-NNN` with zero-padding | Restores referential integrity with `sku_master.csv`. |
| T9 | Missing SKU descriptions | Imputed from matching SKU records in `sku_master.csv` | Enriches master metadata for reporting and validation. |
| T10 | Customer name casing variations | Cleaned and mapped to canonical customer names | Unifies account-level order history. |
| T11 | Formatting slips and sign errors in Units Sold | Commas removed, dashes converted to null, paired negative reversals dropped, unpaired sign errors corrected via `abs()` | Differentiates zero demand from clerical sign errors and accounting reversals. |
| T12 | Pricing scaling anomalies (0.01x, 85x, 5-20x) | Replaced with annual SKU median price across regions | Addresses cent-scale recording, currency conversion shifts (INR/USD), and unit transcription slips. |
| T13 | Revenue calculation discrepancies | Recalculated as `Units_Sold * Unit_Price_USD` | Eliminates mathematical inconsistency following unit and price normalization. |
| T14 | Real market shock preservation | Logged to anomaly register without artificial outlier clipping | Preserves real supply chain shocks and early buying spikes for model training. |
| T15 | Zero-demand vs missing cells | Differentiated zero orders from non-recorded transactions | Enables classification of unrecorded vs zero-demand series-months (B1). |
| R5 | Transactions outside active product lifecycle | Sales prior to Launch Date or subsequent to EOL Date zeroed | Enforces catalog lifecycle boundaries. |
| R9 | Co-occurring customer transactions | Preserved multi-customer orders on identical Date-SKU-Region keys | Ensures distinct customer accounts in the same territory are never conflated or dropped. |
| A2 | Multi-account stockout allocation | Proportional integer allocation by customer volume with remainder to largest customer | Guarantees that allocated customer units sum exactly to total reported stockout units. |

### Master Data and Macroeconomic Cleaning

- **Rule S1 (PMP-004 BU Alignment)**: Reconciled PMP-004 to `Advanced Manufacturing` based on agreement with transaction history majority and Rule R8.
- **Rule S2 (Lifecycle Date Integrity)**: Validated Launch_Date and EOL_Date formats in SKU master, ensuring chronological ordering (Launch <= EOL).
- **Rule M1 (2022-06 Macro Duplicate)**: Selected observation with minimal distance to adjacent linear interpolation (May and July 2022).
- **Rule M2 (Auto Production Percentage Scaling)**: Scaled 2019 fractional values x100, verified continuity step from 2019-12 to 2020-01 is under 25.0 points.
- **Rule M3 (PMI Decimal Misplacement)**: Corrected May 2023 PMI from 4.79 to 47.9 via single-decimal shift.
- **Rule M4 (Missing Macro Indicator Imputation)**: Linear time interpolation applied with `Macro_Imputed` indicator flag.
- **Rule M5 (COVID Structural Break)**: Added `Is_COVID_Period` indicator for 2020-03 through 2021-06.
- **Rule E1 (Event Date Normalisation)**: Formatted Start_Date and End_Date strings to canonical ISO `YYYY-MM-DD`.
- **Rule E2 (Open-Ended Event Bounding)**: Preserved ongoing multi-year structural transitions while bounding temporary operational shocks.
- **Rule E3 (Event Scope Separation)**: Parsed compound `Scope_Value` strings (e.g. `Customer | Region`) into discrete `Scope_Customer` and `Scope_Region` attributes.
- **Rule E4 (Event Referential Validation)**: Aligned operational events with transaction anomalies and documented impact mechanisms.

### Anomaly Classification and Treatment: Errors vs. Real Events (Part 1.6)

A central requirement of the specification is distinguishing clerical data errors from real physical market events:
- **Data Errors (Fix Them)**: Systematic recording errors, typographical slips, currency/unit transcription mistakes, and duplicate database records where the true commercial activity was unchanged. These are corrected deterministically using verified reference baselines.
- **Real Events (Model or Exclude Them)**: Genuine supply chain disruptions, customer pre-buying behaviors, and macro shocks where transaction volume genuinely moved. These points are preserved in history and incorporated as model features, structural break indicators, or unconstrained demand adjustments.

The table below summarizes every flagged anomaly category in `outputs/anomaly_register.csv` (1,529 total records):

| Anomaly Type | Classification | Treatment / Action | Record Count | Rationale and Evidence |
|---|---|---|---|---|
| `inventory_stockout` | Real Event (Supply Limit) | Unconstrained demand reconstructed | 500 | Physical inventory shortages where shipped units were constrained by stockouts. Replaced with requested units to model true demand. |
| `sign_reversal` | Data Error | Absolute value applied (`abs()`) | 491 | Negative units sold resulting from clerical transcription or accounting entry errors without paired positive reversal records. |
| `early_buying` | Real Event (Demand Surge) | Modeled as event feature | 277 | Surge in order volume in Nov-Dec 2022 preceding the announced 1 Jan 2023 catalog price increase (EV-030). |
| `price_scaling_correction` | Data Error | Reference price applied | 200 | Extreme unit price scaling slips (e.g. 100x cent-scale recording, 85x currency transposition) reconciled to SKU-year median price. |
| `high_volume_order` | Real Event (Tender Win) | Retained in history | 31 | Legitimate large commercial fleet tender wins and bulk ordering batches; preserved in sales history to prevent under-forecasting peak demand. |
| `pandemic_supply_shock` | Real Event (Systemic Shock) | Modeled as structural break | 30 | Widespread global automotive manufacturing stoppage during initial COVID-19 pandemic wave (2020-03 to 2020-05); modeled with structural break flag. |

### Events Calendar Treatment Matrix (Part 2.5)

The events calendar (`student_data/events_calendar.csv`) records 68 historical and forward events across 11 categories. Each category is assigned a documented treatment because not every event affects demand:

| Event Category | Count | Scope | Treatment | Business and Modeling Rationale |
|---|---|---|---|---|
| `Holiday` | 39 | Regional (NA, EU, APAC, LATAM, MEA) | Modeled as Feature | Captures recurring calendar shutdowns (Lunar New Year, Ramadan/Eid, European August plant closures); regional effects quantified in `holiday_effects.csv`. |
| `Price Change` | 11 | Global catalog & surcharges | Modeled as Feature & Scenarios | Annual catalog adjustments on 1 Jan; pre-buying surges (EV-030) captured as pre-event demand features; forward price changes (EV-062, EV-066) drive what-if price elasticity scenarios. |
| `Disruption` | 5 | Global logistics / transit | Excluded from Demand Baseline / Supply Bottleneck | Transit shocks (Suez Canal blockage EV-005, semiconductor shortage EV-012, European transport strike EV-018, Port congestion EV-035, Red Sea rerouting EV-045) constrained shipments, not customer demand. Unconstrained series restores true demand without lowering it for supply constraints. |
| `Customer Event` | 3 | Account-level | Modeled as Feature | Key account onboarding (EV-015), large fleet tender win (EV-025), customer tier restructuring (EV-049) drive persistent customer volume shifts tracked in `customer_switches.csv`. |
| `Marketing` | 2 | Segment / Regional | Modeled as Feature | Commercial Fleet Push (EV-052) and Global Aftermarket Campaign (EV-023) introduce positive demand lift features without post-forecast double counting. |
| `System` | 2 | Enterprise operational | Excluded from Organic Demand Model | ERP migration billing freeze (EV-029) and WMS warehouse upgrade (EV-058) caused billing delays and deferred shipments rather than actual shifts in underlying customer consumption. |
| `Corporate` | 2 | Corporate administration | Excluded (Admin) / Modeled (Partnership) | HQ relocation (EV-038) is non-demand administrative event (excluded); OEM strategic partnership (EV-059) represents structural commercial expansion. |
| `Announcement` | 1 | Regulatory standard | Modeled as Transition Catalyst | Euro 7 emissions standard announcement (EV-050) acts as catalyst in `ev_shift_analysis.csv` driving powertrain component mix transition. |
| `Market Entry` | 1 | Geographic expansion | Modeled as Feature | Expansion into ASEAN manufacturing hubs (EV-042) represents geographic addressable market expansion in Asia Pacific. |
| `Supply Constraint` | 1 | Upstream materials | Reconstructed in Unconstrained Demand | Raw material allocation quota (EV-032) constrained production capacity; true customer demand reconstructed via unconstrained pipeline. |
| `Logistics` | 1 | Freight rate spike | Excluded from Demand Model | Air freight rate spike (EV-047) was a margin/cost shock absorbed under fixed-price OEM contracts; did not alter contracted vehicle manufacturing build volume. |

---

## 3. Unconstrained Demand Reconstruction

Real-world demand planning requires forecasting unconstrained customer demand rather than shipment volume constrained by inventory stockouts or logistics disruptions:
1. **Documented Stockouts**: Merged transaction history with `inventory_stockouts.csv`. Where orders were constrained, `Requested_Units` replaced `Shipped_Units` as the true demand target.
2. **Multi-Customer Key Allocation (Rule A2)**: In `inventory_stockouts.csv`, records are identified at the `Date-SKU-Region` level. For series where multiple commercial customer accounts operate in the same region, stockout volume is allocated across accounts proportionally based on their non-negative actual demand. Floor integer allocation is applied, with any rounding remainder assigned to the largest customer account, guaranteeing exact conservation of stockout units.

### Unlisted Stockout Identification and Reconstruction Methodology (Part 2.6)

The project specification highlights that `inventory_stockouts.csv` is *"known to be incomplete"*. Relying strictly on logged stockout records would leave unrecorded stockout periods in the training set as apparent demand collapses, causing machine learning models to under-forecast forward baseline demand.

To address unlisted stockouts:
1. **Series Eligibility Filter**: Analysis is restricted to stable SKU-Region-Customer series with at least 12 months of active historical transaction history and a normal median monthly sales volume $\ge 10$ units.
2. **Exclusion of Exogenous Shocks**: Known market disruption event periods (e.g., initial COVID-19 automotive industry shutdowns, documented regional transport strikes, enterprise ERP cutovers) are excluded from the stockout search space so broad demand contractions are not erroneously flagged as inventory stockouts.
3. **Run-Length Anomaly Detection**: The algorithm scans each eligible series for runs of 2 or more consecutive months where observed shipment volume falls below 50% of the series normal historical median, in periods with no recorded inventory stockout in `inventory_stockouts.csv`.
4. **Unconstrained Demand Imputation**: For each identified unlisted stockout month, unconstrained demand is reconstructed by setting `Units_Sold_Unconstrained` to the series normal operating median. The record is flagged with `stockout_status = 'suspected_stockout'` in `outputs/unconstrained_demand.parquet` and logged in `outputs/anomaly_register.csv`. This prevents transient supplier stockouts from distorting the recursive forward forecast.

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
- P10, P50 and P90 forecasts are kept in non-crossing order.

### Post-Processing, Hierarchy Reconciliation, and Lifecycle Rules
1. **Operational Adjustments**: Applied customer-specific shutdowns (EV-062 Orion Driveline NA zeroing in 2027-07 to 2027-08) and capacity expansions (EV-052 Helios Auto Parts India, factor 1.0828).
2. **New Launch History Inheritance**: New SKUs with designated replacement relationships (`Replaces_SKU`) inherit historical demand patterns from the predecessor SKU prior to launch. Unmatched new launches adopt product family average demand profiles. Demand prior to the official `Launch_Date` is strictly zeroed.
3. **Lifecycle Bounds**: Zeroed all forecast volume outside active SKU launch and EOL windows.
4. **Hierarchy Addition**: Exact bottom-up sums are reported across family, region, business unit and total. Trend projections are in `outputs/hierarchy_diagnostic.csv` as diagnostic only and do not change any status or the submission.

---

## 6. Backtesting and Validation Framework

The modeling strategy is validated using a rolling-origin backtesting framework across three historical economic regimes:
1. **Origin 1 (2023-10)**: 19-month forward evaluation through 2025-05.
2. **Origin 2 (2024-10)**: 19-month forward evaluation through 2026-05.
3. **Origin 3 (2025-03)**: 19-month forward evaluation through 2026-10.

### Leakage Prevention Protocol
- **Strict Temporal Boundary**: For each origin cutoff, training data is strictly filtered to observations occurring on or before that origin date (`Date <= origin`).
- **Dynamic Feature Isolation**: Customer ordering cycle classification, Syntetos-Boylan intermittency parameters, and churn flags are recomputed strictly from historical observations up to that cutoff.
- **Recursive Forward Path**: Evaluated across 19 forward months where future lags are fed from model predictions rather than ground truth actuals.
- **Interval Predictions**: Detailed backtest predictions including `p10`, `p50`, and `p90` are logged in `outputs/backtest_predictions.csv` to allow rigorous coverage validation.
- **Benchmarks**: LightGBM is compared against seasonal naive, moving average, and Croston/SBA/TSB baselines on identical test rows.

### Evaluation Metrics and Sign Conventions
Backtest evaluations in `outputs/backtest_metrics.csv` compute two primary accuracy and error metrics across slices:

1. **Weighted Absolute Percentage Error (WAPE)**:
   $$\text{WAPE} = \frac{\sum_{i=1}^N |y_{\text{pred}, i} - y_{\text{true}, i}|}{\sum_{i=1}^N y_{\text{true}, i}}$$
   WAPE provides a volume-weighted percentage error metric that avoids division by zero on intermittent, zero-demand slices.

2. **Mean Forecast Bias**:
   $$\text{Bias} = \frac{1}{N} \sum_{i=1}^N (y_{\text{pred}, i} - y_{\text{true}, i})$$
   - **Units**: Physical units per series-month (count of automotive components).
   - **Sign Convention**: Following standard industrial inventory and supply chain convention, $\text{mean}(y_{\text{pred}} - y_{\text{true}})$ is signed such that a **negative value indicates under-forecasting** (the model projected fewer units than actual demand), while a **positive value indicates over-forecasting**.

### Bias finding

The bias formula is $\text{bias} = \text{mean(predicted - actual)}$. A negative value means under-forecasting.

Measured LightGBM P50 and seasonal naive bias by rolling origin, with `breakdown = overall` in `outputs/backtest_metrics.csv`:

| Origin | LightGBM P50 bias | Seasonal naive bias |
|---|---:|---:|
| 2023-10 | -14.76 | -4.53 |
| 2024-10 | -8.40 | -2.45 |
| 2025-03 | -9.39 | -0.87 |

Using matched rows in `outputs/backtest_predictions.csv` and `outputs/clean_train.parquet`, LightGBM P50 mean bias against `Units_Sold_Unconstrained` and against shipped `Units_Sold` was:

| Origin | Bias vs Units_Sold_Unconstrained | Bias vs Units_Sold |
|---|---:|---:|
| 2023-10 | -14.755922 | -7.283128 |
| 2024-10 | -8.399352 | -0.787794 |
| 2025-03 | -9.388353 | -1.850973 |

The comparison shows that changing the measured target substantially reduces the negative bias, especially at 2024-10 and 2025-03, but the shipped-target comparison does not explain the full negative bias in the unconstrained backtest.

### EV shift and lag stability findings

From `outputs/ev_shift_analysis.csv`, the EV_Adoption_Index coefficient against ICE volume is -109.6675 at lag 1 month. The output p-value is 0.0000032684215054787183, so the coefficient is statistically significant at 5%. This result can be described as a statistically significant relationship in the latest output; the methodology does not convert it into an unsupported causal shift claim.

The EV-coded share of volume is 0.00% in 2019 and 10.52% in 2026. From `outputs/lag_stability.csv`, at least one best lag moved between the 2019-2022 and 2023-2026 windows for Brake & Safety Systems, Chassis & Structural, Electrical Assemblies, Hydraulic Systems, Powertrain Components, Precision Machined Parts and Thermal Management.

### Investigation of Systematic Under-Forecasting (Priority A5)
Across all three evaluation origins, LightGBM P50 exhibits systematic negative bias (-8.40 to -14.76 units/month overall), while outperforming all baseline models on WAPE (0.360 to 0.396 vs 0.374 to 0.406 for seasonal naive). A controlled diagnostic investigation into three candidate drivers reveals:

1. **Target Change (Unconstrained Demand vs Shipped History)**:
   - *Hypothesis*: The model is trained and scored against `Units_Sold_Unconstrained` (which reconstructs lost volume from documented stockout records), rather than raw shipped transactions (`Units_Sold`).
   - *Evidence*: Scoring the exact same LightGBM predictions against raw shipped actuals (`lightgbm_p50_vs_shipped`) cuts the measured negative bias roughly in half (from -14.76 units to -7.28 units on Origin 2023-10). Evaluating against the unconstrained ground truth naturally widens the negative gap by ~7.5 units because stockout replenishment adds genuine latent demand that constrained historical shipments never captured.

2. **Recursive Multi-Step Lag Propagation**:
   - *Hypothesis*: In recursive multi-step forecasting, predictions for horizon $h > 1$ rely on model-generated lag features rather than true actuals, potentially compounding negative bias over the 19-month window.
   - *Evidence*: Empirical horizon tracking confirms monotonic divergence across the forward horizon:
     - Month 1: Bias = -3.83 units (WAPE: 0.274)
     - Month 6: Bias = -10.43 units (WAPE: 0.336)
     - Month 12: Bias = -19.66 units (WAPE: 0.401)
     - Month 19: Bias = -26.16 units (WAPE: 0.473)
     Because quantile regression at $\alpha = 0.50$ estimates the conditional median rather than the conditional mean, slight median under-prediction in early horizons compounds into lower lag features, mildly dampening subsequent predictions.

3. **Quantile Loss Objective and Demand Asymmetry**:
   - *Hypothesis*: The P50 booster optimizes pinball loss at $\alpha = 0.50$, which estimates the median rather than the mean.
   - *Evidence*: Automotive aftermarket and OEM spare parts demand displays heavy right-skew: series frequently record zero or small orders punctuated by large bulk procurement runs. In right-skewed distributions, the conditional median is strictly less than the conditional mean ($\text{Median} < \text{Mean}$). An unbiased median forecast will therefore produce a negative mean bias when evaluated using $\text{mean}(y_{\text{pred}} - y_{\text{true}})$. Baseline models like Croston and moving averages optimize for mean demand and display near-zero or positive bias, but suffer significantly worse WAPE (0.430 to 0.443 vs 0.360 to 0.396 for LightGBM).

*Model Policy*: No bias offsets or heuristic scaling factors were injected into model predictions. The LightGBM P50 model is retained as the forecasting engine because it achieves lower WAPE across the backtest origins.

---

## 7. Macroeconomic Sensitivity (What-If Analysis)

To stress-test supply chain resiliency under fluctuating economic conditions, the system evaluates demand sensitivity across a calibrated grid of ISM Manufacturing PMI shifts from -6.0 to +6.0 in steps of 1.5:
- Model Sensitivity Reruns: The LightGBM quantile regression models evaluate modified macroeconomic feature vectors.
- Scenario Aggregations: Demand shifts are aggregated by product family and region, exported to `outputs/whatif_scenarios.csv`, and rendered through an interactive slider in the control tower dashboard.

---

## 7.1 Originality and Innovation

The solution's choices are documented with their reasons: unconstrained demand reconstruction preserves stockout-era demand, rolling-origin backtesting prevents future leakage, recursive lags update at forecast horizon, and hierarchy reporting separates exact submission sums from trend diagnostics.

---

## 8. Reproducibility and Deliverable Hashes

The pipeline is completely deterministic. With fixed seeds (`random_state=42`), `deterministic=True`, and stable sorting before export, identical outputs are produced on every run.

The table below documents MD5 values from the latest local pipeline run:

| Deliverable File | MD5 Digest |
|---|---|
| `project/outputs/clean_train.parquet` | `5425c4efbcfd22510a442b6d3f9e4194` |
| `project/outputs/audit_log.csv` | `de057d5da661236b0eeb6b089b9ecf78` |
| `project/outputs/forecast_submission.csv` | `9a929f7a58348c19f6ecbe43cd181867` |
| `project/outputs/backtest_metrics.csv` | `481afbb89c16e213db6f3fc6b913b80e` |

---

## 9. Seasonal Shape Dynamics and Correlation Analysis

The latest checker output measures a calendar-month seasonal correlation of 0.70 between historical monthly actuals (2023-2026) and the 19-month forward forecast (2026-11 to 2028-05). The reported variance is associated with catalog and operating conditions:

1. **Catalog Lifecycle Transitions (EOL & Launch Dynamics)**:
   Multiple high-volume legacy component lines reach their scheduled End-of-Life (EOL) during 2027 and early 2028, while new replacement SKUs ramp up under `Replaces_SKU` lifecycle boundaries. This catalog phase-out naturally shifts monthly aggregated volume compared to historical years where legacy SKUs were in full production.
2. **Operational Plant Shutdowns (EV-062)**:
   In July and August 2027, Orion Driveline's North American assembly facilities undergo a complete tooling changeover (EV-062), resulting in zero demand for that customer account across those months. Historically, July and August represent high-volume ordering periods; the operational zeroing intentionally alters the forward seasonal shape.
3. **Recursive Lag Evolution**:
   Under recursive multi-step forecasting, dynamic lags and rolling means update from predicted unconstrained demand rather than repeating historical supply-constrained shipment dips. The model preserves underlying structural demand rather than mimicking transient historical volatility.

Forcing calendar-month correlation to match historical profiles would ignore confirmed plant shutdowns and catalog lifecycle boundaries.

---

## 10. Pipeline Outputs: Regenerated vs Committed Artifacts

Every file in `project/outputs/` is regenerated end-to-end when executing `python project/run_pipeline.py`.

### Fully Regenerated Output Artifacts
- **Cleaned Data**: `clean_train.parquet`, `unconstrained_demand.parquet`, `inventory_stockouts.csv`, `clean_macro.csv`, `clean_events.csv`, `audit_log.csv`.
- **Feature Matrix**: `features.parquet`.
- **Forecasts & Intervals**: `forecast_submission.csv` (58,219 rows), `forecast_intervals.csv`.
- **Governance & Verification**: `backtest_metrics.csv`, `backtest_predictions.csv`, `hierarchy_addition.csv`, `hierarchy_diagnostic.csv`, `event_impact_estimates.csv`.
- **Scenarios & Dynamics**: `whatif_scenarios.csv`, `whatif_scenario_forecast.csv`, `ev_shift_analysis.csv`, `macro_lag_analysis.csv`, `anomaly_register.csv`.
- **Documentation**: `data_quality_report.json`, `data_quality_report.md`, `model_card.md`, `check_report.txt`.

### Committed Repository Deliverables
- Core codebase in `project/pipeline/` and control tower dashboard in `project/app/dashboard.py`.
- Automated test suite in `project/tests/`.
- Technical documentation in `project/docs/`.
- Benchmark outputs in `project/outputs/` retained to support verification.

---

## AI assistance

AI coding tools were used for a small part of the work: drafting some unit test code and fixture setup, and tidying some documentation. The data rules, cleaning logic, backtest design, forecasting model, hierarchy checks and dashboard design were written and checked by the team. Every AI-drafted test was run and reviewed before it was kept.


