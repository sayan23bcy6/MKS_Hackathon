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

Before cleaning, 18,531 revenue rows differed from Units_Sold multiplied by Unit_Price_USD (counted after the price repair step). The 1,744 figure in an earlier profile is superseded.

### Master Data and Macroeconomic Cleaning

- **Rule S1 (PMP-004 BU Alignment)**: Reconciled PMP-004 to `Advanced Manufacturing` based on agreement with transaction history majority and Rule R8.
- **Rule S2 (Lifecycle Date Integrity)**: Validated Launch_Date and EOL_Date formats in SKU master, ensuring chronological ordering (Launch <= EOL).
- **Rule M1 (2022-06 Macro Duplicate)**: Selected observation with minimal distance to adjacent linear interpolation (May and July 2022).
- **Rule M2 (Auto Production Percentage Scaling)**: Scaled 2019 fractional values x100; the continuity step from 2019-12 to 2020-01 is under 25.0 points.
- **Rule M3 (PMI Decimal Misplacement)**: The 2023-04 PMI is 47.2, 2023-05 was corrected from 4.79 to 47.9, and 2023-06 is 46.1.
- **Rule M4 (Missing Macro Indicator Imputation)**: Linear time interpolation applied with `Macro_Imputed` indicator flag.
- **Rule M5 (COVID Structural Break)**: Added `Is_COVID_Period` indicator for 2020-03 through 2021-06.
- **Rule E1 (Event Date Normalisation)**: Formatted Start_Date and End_Date strings to canonical ISO `YYYY-MM-DD`.
- **Rule E2 (Open-Ended Event Bounding)**: Preserved ongoing multi-year structural transitions while bounding temporary operational shocks.
- **Rule E3 (Event Scope Separation)**: Parsed compound `Scope_Value` strings (e.g. `Customer | Region`) into discrete `Scope_Customer` and `Scope_Region` attributes.
- **Rule E4 (Event Referential Validation)**: Aligned operational events with transaction anomalies and documented impact mechanisms.

### Anomaly Classification and Treatment: Errors vs. Real Events (Part 1.6)

A central requirement of the specification is distinguishing clerical data errors from real physical market events:
- **Data Errors**: Systematic recording errors, typographical slips, currency/unit transcription mistakes, and duplicate database records where the true commercial activity was unchanged. These are corrected using reference baselines.
- **Real Events (Model or Exclude Them)**: Genuine supply chain disruptions, customer pre-buying behaviors, and macro shocks where transaction volume genuinely moved. These points are preserved in history and incorporated as model features, structural break indicators, or unconstrained demand adjustments.

The table below summarizes every flagged anomaly category in `outputs/anomaly_register.csv` (1,605 total records):

| Anomaly Type | Classification | Treatment / Action | Record Count | Rationale and Evidence |
|---|---|---|---|---|
| `inventory_stockout` | Real Event (Supply Limit) | Unconstrained demand reconstructed | 500 | Physical inventory shortages where shipped units were constrained by stockouts. Replaced with requested units to model true demand. |
| `sign_reversal` | Data Error | Absolute value applied (`abs()`) | 491 | Negative units sold resulting from clerical transcription or accounting entry errors without paired positive reversal records. |
| `early_buying` | Real Event (Demand Surge) | Modeled as event feature | 280 | Surge in order volume in Nov-Dec 2022 preceding the announced 1 Jan 2023 catalog price increase (EV-030). |
| `price_scaling_correction` | Data Error | Reference price applied | 200 | Extreme unit price scaling slips (e.g. 100x cent-scale recording, 85x currency transposition) reconciled to SKU-year median price. |
| `high_volume_order` | Data Entry Check | Replaced with SKU-year median | 104 | Rows above 20 times their SKU-year median were replaced and logged as suspected entry errors. |
| `pandemic_supply_shock` | Real Event (Systemic Shock) | Modeled as structural break | 30 | Widespread global automotive manufacturing stoppage during initial COVID-19 pandemic wave (2020-03 to 2020-05); modeled with structural break flag. |

The sign-reversal count is 491 in both the T11 negative-units profile and the sign_reversal register, so no difference requires reconciliation.

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
The forecast is chosen per product family. Each family uses the candidate with the lowest mean WAPE over the two selection origins, 2023-10 and 2024-10. The holdout origin, 2025-03, is not used for the choice. Candidates are seasonal naive, the 12-month moving average, Croston/SBA/TSB intermittent methods, and LightGBM P50. Ties go to the simpler model. The selection table is written to `outputs/family_model_selection.csv` and reproduced here:

| Family | Selected model | Selection WAPE (2023-10, 2024-10 mean) | Holdout WAPE 2025-03, selected | Holdout WAPE 2025-03, LightGBM | Holdout WAPE 2025-03, seasonal naive |
|---|---|---:|---:|---:|---:|
| Brake & Safety Systems | naive_seasonal | 0.2977 | 0.2936 | 0.3062 | 0.2936 |
| Chassis & Structural | naive_seasonal | 0.3419 | 0.3301 | 0.3479 | 0.3301 |
| Electrical Assemblies | lightgbm_p50 | 0.3729 | 0.3488 | 0.3488 | 0.3624 |
| Hydraulic Systems | naive_seasonal | 0.3505 | 0.3432 | 0.3481 | 0.3432 |
| Powertrain Components | naive_seasonal | 0.3453 | 0.3205 | 0.3505 | 0.3205 |
| Precision Machined Parts | lightgbm_p50 | 0.3564 | 0.3361 | 0.3361 | 0.3391 |
| Thermal Management | lightgbm_p50 | 0.3355 | 0.3134 | 0.3134 | 0.3276 |

On the holdout origin, the selected models give a mean WAPE of 0.3265 across families. LightGBM alone gives 0.3359, and seasonal naive alone gives 0.3309. Only the LightGBM candidate has P10 and P90 bands.

LightGBM quantile regression provides the LightGBM candidate and the P10 and P90 bands:
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
- This recursion drives the LightGBM candidate for every family. Families whose selected model is a baseline use the baseline values instead (see the assembly step below).

### Post-Processing, Hierarchy Reconciliation, and Lifecycle Rules
1. **Operational Adjustments**: Applied the EV-062 Orion Driveline NA zeroing in 2027-07 to 2027-08. EV-052 (Helios Auto Parts, India) is handled as a model feature; its estimated uplift factor (1.2056 in `outputs/event_impact_estimates.csv`) is recorded but not applied as a post-forecast multiplier.
2. **New Launch History Inheritance**: New SKUs with designated replacement relationships (`Replaces_SKU`) inherit historical demand patterns from the predecessor SKU prior to launch. Unmatched new launches adopt product family average demand profiles. Demand prior to the official `Launch_Date` is strictly zeroed.
3. **Lifecycle Bounds**: Zeroed all forecast volume outside active SKU launch and EOL windows.
4. **Per-family assembly**: Each row takes the value of its family's selected model. Rows selected as LightGBM carry P10, P50 and P90. Rows selected as a baseline carry a point forecast only, and their P10 and P90 are blank.
5. **Hierarchy Addition**: Exact bottom-up sums are reported across family, region, business unit and total. Trend projections are in `outputs/hierarchy_diagnostic.csv` as diagnostic only and do not change any status or the submission.

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
- **Interval Predictions**: Backtest predictions including `p10`, `p50`, and `p90` are logged in `outputs/backtest_predictions.csv` for coverage validation.
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
| 2023-10 | -12.44 | -3.13 |
| 2024-10 | -5.62 | -0.97 |
| 2025-03 | -7.15 | -0.21 |

Using matched rows in `outputs/backtest_predictions.csv` and `outputs/clean_train.parquet`, LightGBM P50 mean bias against `Units_Sold_Unconstrained` and against shipped `Units_Sold` was:

| Origin | Bias vs Units_Sold_Unconstrained | Bias vs Units_Sold |
|---|---:|---:|
| 2023-10 | -12.44 | -3.13 |
| 2024-10 | -5.62 | -0.97 |
| 2025-03 | -7.15 | -0.21 |

Against shipped `Units_Sold`, the negative bias at 2023-10 falls from -12.44 to -4.96 units; a negative bias remains. At 2024-10 and 2025-03 the shipped-target bias is +1.99 and +0.38. The unconstrained target therefore accounts for most of the negative bias at those two origins.

## Effect of entry-error replacement

Entry-error replacement (rule A3) is applied in this build: 104 rows above 20 times their SKU-year median were replaced. Totals and metrics from earlier builds are not reproduced here, so this section reports only the values from this build.

| Origin | LightGBM WAPE | Seasonal naive WAPE |
|---|---:|---:|
| 2023-10 | 0.3721 | 0.3585 |
| 2024-10 | 0.3302 | 0.3337 |
| 2025-03 | 0.3333 | 0.3278 |

Coverage in this build: raw 0.6037 at 2023-10, 0.7023 at 2024-10 and 0.6893 at 2025-03; calibrated 0.6037, 0.7763 and 0.8068 at the same origins. The 2023-10 origin has no earlier residuals, so its calibrated value equals its raw value.

Top five families in this build: Brake & Safety Systems 580,555, Chassis & Structural 469,648, Thermal Management 437,813, Precision Machined Parts 434,644, Powertrain Components 418,604.

Coverage is calculated as the fraction of non-null out-of-sample evaluation rows where actual demand falls within the prediction interval:
$$\text{Coverage} = \frac{1}{N} \sum_{i=1}^N \mathbf{1}(P10_i \le y_i \le P90_i)$$
The calculation uses an unweighted observation average (giving equal weight to every series-month evaluation observation) across all $N = 169,229$ out-of-sample predictions across the three rolling backtest origins (2023-10, 2024-10, and 2025-03). The overall result covers 169,229 predictions: raw coverage 66.55% (0.6655) and calibrated coverage 72.97% (0.7297).

### Zero-order shares

The panel reports these zero-order rows and active SKU-region-customer series per year:

| Year | Zero-order rows | Active series | Zero-order rows per active series |
|---|---:|---:|---:|
| 2019 | 3,457 | 2,068 | 1.671663 |
| 2020 | 3,675 | 2,211 | 1.662144 |
| 2021 | 3,848 | 2,347 | 1.639540 |
| 2022 | 4,826 | 2,916 | 1.655007 |
| 2023 | 4,880 | 3,049 | 1.600525 |
| 2024 | 5,073 | 3,113 | 1.629618 |
| 2025 | 5,211 | 3,142 | 1.658498 |
| 2026 | 4,408 | 3,144 | 1.402036 |

The rise from 2021 to 2022 was accompanied by an increase in active series from 2,347 to 2,916 and a small increase in zero-order rows per active series from 1.639540 to 1.655007. Both more active series and slightly more zero orders per series contributed.

### Revenue and unassigned regions

Forecast revenue uses the most recent positive Unit_Price_USD for each SKU in the cleaned training data. Unassigned rows are sales with no region in the source data. Their LightGBM P50 WAPE is 0.7392 for 2023-10, 0.8776 for 2024-10, and 0.7978 for 2025-03, against 0.3713, 0.3286 and 0.3320 for the named regions at the same origins.

### EV shift and lag stability findings

The level regression in `outputs/ev_shift_analysis.csv` has coefficient -122.9144 against ICE volume at lag 1 month, R^2 0.2834 and p-value 3.97e-08. Shared trends can inflate this level result.

The first-difference regressions report:

| Lag | Coefficient | t-statistic | p-value | Observations | Significant at 5% |
|---:|---:|---:|---:|---:|---|
| 0 | -448.124081 | -0.7024 | 0.484195 | 93 | No |
| 1 | 383.495079 | 0.5991 | 0.550604 | 92 | No |
| 2 | -311.055674 | -0.4933 | 0.623038 | 91 | No |
| 3 | 327.805299 | 0.5180 | 0.605725 | 90 | No |
| 4 | -209.259518 | -0.3280 | 0.743702 | 89 | No |
| 5 | 722.582679 | 1.1467 | 0.254667 | 88 | No |
| 6 | -451.048432 | -0.7072 | 0.481387 | 87 | No |

The first-difference result is not significant at 5%, so it is not described as a shift or a causal effect.

The EV-coded share of volume is 0.00% in 2019 and 9.89% in 2026. In `outputs/lag_stability.csv`, 12 of 56 family-indicator pairs keep the same best lag between the 2019-2022 and 2023-2026 windows (21.4%); the other 44 move.

### Investigation of systematic under-forecasting
Across the three evaluation origins, LightGBM P50 has negative bias against the unconstrained target (-5.62 to -12.44 units per month overall). LightGBM beats seasonal naive on WAPE only at 2024-10 (0.3302 vs 0.3337); it loses at 2023-10 (0.3721 vs 0.3585) and 2025-03 (0.3333 vs 0.3278). A controlled diagnostic investigation into three candidate drivers reveals:

1. **Target Change (Unconstrained Demand vs Shipped History)**:
   - *Hypothesis*: The model is trained and scored against `Units_Sold_Unconstrained` (which reconstructs lost volume from documented stockout records), rather than raw shipped transactions (`Units_Sold`).
   - *Evidence*: Scoring the same predictions against shipped `Units_Sold` reduces the negative bias at 2023-10 from -12.44 to -4.96 units. At 2024-10 and 2025-03 the shipped-target bias is +1.99 and +0.38, so the unconstrained target accounts for most of the negative bias at those origins.

2. **Recursive Multi-Step Lag Propagation**:
   - *Hypothesis*: In recursive multi-step forecasting, predictions for horizon $h > 1$ rely on model-generated lag features rather than true actuals, potentially compounding negative bias over the 19-month window.
   - *Evidence*: Empirical horizon tracking confirms monotonic divergence across the forward horizon:
     - Month 1: Bias = -2.73 units (WAPE: 0.258), pooled over the three origins
     - Month 6: Bias = -6.58 units (WAPE: 0.308), pooled over the three origins
     - Month 12: Bias = -10.30 units (WAPE: 0.340), pooled over the three origins
     - Month 19: Bias = -15.08 units (WAPE: 0.406), pooled over the three origins
     Because quantile regression at $\alpha = 0.50$ estimates the conditional median rather than the conditional mean, slight median under-prediction in early horizons compounds into lower lag features, mildly dampening subsequent predictions.

3. **Quantile Loss Objective and Demand Asymmetry**:
   - *Hypothesis*: The P50 booster optimizes pinball loss at $\alpha = 0.50$, which estimates the median rather than the mean.
   - *Evidence*: Automotive aftermarket and OEM spare parts demand displays heavy right-skew: series frequently record zero or small orders punctuated by large bulk procurement runs. In right-skewed distributions, the conditional median is strictly less than the conditional mean ($\text{Median} < \text{Mean}$). An unbiased median forecast will therefore produce a negative mean bias when evaluated using $\text{mean}(y_{\text{pred}} - y_{\text{true}})$. Baseline models like Croston and moving averages optimize for mean demand and display near-zero or positive bias, but suffer significantly worse WAPE (0.374 to 0.402, against 0.330 to 0.372 for LightGBM).

*Model Policy*: No bias offsets or heuristic scaling factors were injected into model predictions. Each family uses the model chosen by the selection rule above, so the forecast is not a single model. LightGBM is retained as the global ML candidate that the brief asks for, and it is used for the families where it wins on the selection origins. On this backtest it does not beat seasonal naive overall: mean WAPE across the three origins is 0.345 for LightGBM and 0.340 for seasonal naive, and LightGBM is better only at 2024-10. This is reported as a limitation.

---

## 7. Macroeconomic Sensitivity (What-If Analysis)

To stress-test supply chain resiliency under fluctuating economic conditions, the system evaluates demand sensitivity across a calibrated grid of ISM Manufacturing PMI shifts from -6.0 to +6.0 in steps of 1.5:
- Model Sensitivity Reruns: The LightGBM quantile regression models evaluate modified macroeconomic feature vectors.
- Scope of the shock: Only families whose selected model is LightGBM respond to the PMI shift. Families selected for a baseline have no macro input and stay fixed in every scenario. At -3.0 points the total moves from 3,004,640 to 2,974,988 units (-29,652, or -0.99%).
- Scenario Aggregations: Demand shifts are aggregated by product family and region, exported to `outputs/whatif_scenarios.csv`, and rendered through an interactive slider in the control tower dashboard.

---

## 7.1 Originality and Innovation

The solution's choices are documented with their reasons: unconstrained demand reconstruction preserves stockout-era demand, rolling-origin backtesting prevents future leakage, recursive lags update at forecast horizon, and hierarchy reporting separates exact submission sums from trend diagnostics.

---

## 8. Reproducibility and Deliverable Hashes

The pipeline sets fixed seeds (`random_state=42`), `deterministic=True`, `num_threads=1` for LightGBM, and sorts before export. The results are reproducible when the library versions in `requirements.txt` are used (Python 3.10, pandas 2.2.3, numpy 2.1.2, LightGBM 4.7.0). A run in the sandbox with Python 3.12, pandas 3.0 and numpy 2.5 gave a different forecast (3,016,755 units); the same pipeline with the pinned versions reproduces the team's forecast of 3,004,640 units exactly. The cleaned training data is identical in every run; the audit log content is identical apart from line endings.

The hash check is opt-in. Run `CELERIS_CHECK_HASHES=1 pytest project/tests` to compare this machine's outputs with the table below.

The table below documents MD5 values from the reference run with the pinned versions:

| Deliverable File | MD5 Digest |
|---|---|
| `project/outputs/clean_train.parquet` | `625c0a51a6552299b57a8624cc1f5d22` |
| `project/outputs/audit_log.csv` | `7d668b0e7ba3fd139856b2ef9ab9b675` |
| `project/outputs/forecast_submission.csv` | `cad40b2354854eaa87b6c047903b0d60` |
| `project/outputs/backtest_metrics.csv` | `5a7477b9894a1f997403064094fe6a49` |

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


