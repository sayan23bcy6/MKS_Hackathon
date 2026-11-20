# Presentation Outline: Demand Forecasting and Planning Control Tower

A structured guide for presenting and defending the Celeris Automobili demand forecasting solution.

---

## Slide 1: Executive Title and Problem Statement
- **Project Title**: Celeris Automobili - Industrial Demand Forecasting and Planning Control Tower
- **Business Scope**: Commercial demand forecasting across 7 component product families, 12 OEM and Tier 1 customers, and 6 global regions.
- **Mission**: Deliver a reliable, explainable, and unconstrained 19-month forward forecast (2026-11 to 2028-05) across 58,219 SKU-Region-Customer series.
- **Key Challenge**: Decoupling real market demand signals from historical supply chain disruptions, stockouts, and widespread clerical data corruption.

---

## Slide 2: End-to-End Architecture
- **Pipeline Workflow**:
  `Raw Data (student_data/) -> Deterministic Rule Cleaning -> Unconstrained Demand Reconstruction -> Feature Engineering -> Full-History Quantile Boosters -> Recursive Multi-Step Forecasting -> Top-Down Reconciliation -> Control Tower Dashboard`
- **Core Principles**:
  - Raw data files are strictly read-only and immutable.
  - Every data modification is logged in `outputs/audit_log.csv` with rule ID, raw value, clean value, and reason.
  - Complete decision coverage: every ambiguity has an explicit, tested decision rule.
  - Single reproducible execution command rebuilds all artifacts.

---

## Slide 3: Data Cleaning and Defensible Decision Rules
- **Date and Master Integrity**:
  - Normalized 6 mixed date formats to `YYYY-MM` (Rule T1). Date column held authoritative per Rule R1 (Rule T2).
  - Resolved Record ID duplication by retaining latest timestamp (Rule T4).
  - Preserved multi-customer co-occurring orders on identical Date-SKU-Region combinations (Rule R9).
- **Price and Volume Normalization**:
  - Negative units handled: paired reversals dropped; clerical sign flips corrected via absolute value (Rule T11).
  - Out-of-band pricing (cent-scale slips, currency disparities) repaired using annual SKU median reference prices across regions (Rule T12).
  - Revenue mathematically recomputed as `Units_Sold * Unit_Price_USD` (Rule T13).
- **Master Data Harmonization**:
  - PMP-004 Business Unit conflict resolved to `Advanced Manufacturing` based on transaction history majority (Rule S1).
  - Blank geographic regions classified under canonical `Unassigned` category (Rule T5).

---

## Slide 4: Real Market Dynamics vs Clerical Errors
- **Distinguishing Signals from Errors**:
  - Real shocks must never be smoothed away as outliers.
  - COVID-19 pandemic shock (2020-02 to 2020-06) preserved and encoded with `Is_COVID_Period` and event flags.
  - Pre-announcement early buying (Nov-Dec 2022) ahead of the EV-030 price rise captured via explicit event feature.
  - Inventory stockouts decoupled: `Requested_Units` utilized as true market demand target, adjusting constrained shipments.
  - Suspected unlisted stockouts detected via multi-month run-length collapse (>50% drop) and adjusted to normal operating median.

---

## Slide 5: Feature Engineering and Market Drivers
- **Temporal and Cyclical Dynamics**:
  - Cyclic harmonics ($\sin, \cos$ transforms of month), quarter flags, and regional holiday regimes (Lunar New Year, Ramadan, European summer shutdown).
- **Demand Persistence and Growth**:
  - Historical lags at 1, 2, 3, 6, and 12 months.
  - True YoY growth ratio: `(lag_1m - lag_12m) / (|lag_12m| + 1e-3)` distinct from static lag features.
  - Rolling demand averages and rolling volatility (3m, 6m, 12m).
- **Customer Buying Behavior**:
  - Intermittency classification via Syntetos-Boylan ADI and $CV^2$ thresholds.
  - Customer ordering periodicity (monthly, bi-monthly, quarterly, irregular) and churned account detection.
- **Macroeconomic Lead-Lag Cross-Correlation**:
  - Cross-correlations across lags 0-6 months for ISM PMI, Auto Production YoY, and Raw Materials Indices.
  - Evaluation of powertrain transition: ICE vs EV component demand shifts correlated with the EV Adoption Index.

---

## Slide 6: Machine Learning Architecture
- **LightGBM Quantile Gradient Boosted Trees**:
  - Direct quantile modeling for P10 (downside threshold), P50 (median point forecast), and P90 (upside buffer).
- **Two-Stage Training Strategy**:
  1. Validation stage (train through 2025-03, validate 2025-04 through 2026-10) using early stopping to identify optimal `best_iteration`.
  2. Full-history refit on all data through 2026-10 using `best_iteration`, ensuring the latest market dynamics inform future projections.
- **Recursive Multi-Step Forecasting**:
  - Month-by-month prediction rollout across the 19-month test horizon.
  - Dynamic recalculation of lag and rolling features using prior horizon predictions, preventing feature vector freeze.
- **Post-Processing Invariants**:
  - Non-crossing quantile ordering ($P10 \le P50 \le P90$) strictly enforced.
  - Lifecycle boundary zeroing: strictly zero forecast prior to Launch Date or subsequent to EOL Date.
  - Dynamic plant retooling shutdown adjustments (EV-062 Orion Driveline NA zeroing).

---

## Slide 7: Out-of-Sample Backtesting and Governance
- **Rolling-Origin Evaluation**:
  - Three distinct historical cutoffs: 2023-10, 2024-10, and 2025-03.
  - 19-month out-of-sample forward horizon evaluated per origin.
- **Metrics Tracked**:
  - Weighted Absolute Percentage Error (WAPE) and Mean Forecast Bias.
  - Performance degradation tracking across horizon months 1 through 19.
  - Slices across product families, operating regions, and commercial channels.
- **Benchmark Superiority**:
  - Quantified accuracy improvement of LightGBM P50 over seasonal naive and trailing moving average benchmarks.

---

## Slide 8: Control Tower Dashboard & Decision Support
- **Architecture**: Streamlit application reading strictly from pipeline outputs (`project/outputs/`).
- **Core Views**:
  1. Executive Overview: Consolidated historical actuals, P50 forecast, and P10-P90 interval band.
  2. Data Health and Audit Log: Live audit tracking, cleaning rule counts, and decision rule register.
  3. Demand Explorer: Multi-level drill-down by product family, region, channel, and customer.
  4. Anomaly Monitor: Historical demand overlaid with structural breaks, stockouts, and market events.
  5. Forecast and Accuracy: Rolling-origin backtest tables, WAPE curves across 19 horizons.
- **Advanced Views**:
  6. Macro Lead/Lag Heatmap: Interactive cross-correlation matrix across lags 0-6 months.
  7. Stockouts and Lost Demand: Requested vs shipped vs lost volume by region and cause.
  8. What-If Macro Scenario: Interactive ISM PMI slider recalculating shifted demand by family.
  9. Lifecycle & Powertrain View: ICE vs EV adoption curves and transition velocity.

---

## Slide 9: Conclusion and Business Impact
- **Operational Readiness**: Eliminates supply shortages on critical components while reducing safety stock holding costs.
- **Transparency**: Every single prediction is backed by traceable data lineage, verifiable cleaning rules, and automated unit tests.
- **Reproducibility**: Entire system rebuilds from a single command: `python project/run_pipeline.py`.
