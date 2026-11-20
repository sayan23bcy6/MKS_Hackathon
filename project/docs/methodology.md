# Methodology – Celeris Automobili Demand Forecasting

## Problem Understanding

Celeris Automobili supplies automotive components to 12 customers across 6 regions and 7 product families. The goal is to forecast **unconstrained demand** (what customers will request, not what can be supplied) for 58,219 SKU-Region-Customer rows from 2026-11 to 2028-05 (19 months).

The dataset is intentionally messy: 15 known data quality issues in `train.csv` alone, plus quality issues in macro indicators, the events calendar, and the SKU master. The key challenge is to fix errors without destroying real business signal (COVID disruptions, price-rise anticipation, stockouts).

---

## WS1 – Data Cleaning

### Why rule-based cleaning?

Every fix must be **explainable in a presentation**. We use explicit rule dicts (config.py), not fuzzy matching or ML-based imputation, so any judge can read the rule and understand why a value was changed. Every change is logged to `audit_log.csv` with the raw value, new value, rule ID, and reason.

### Key decisions

| Issue | What we detected | How we fixed it | Why |
|---|---|---|---|
| T1 | 6 date formats in one column | Regex + month-name map → YYYY-MM | Date is used as the join key; inconsistent formats break every merge |
| T2 | Year/Month ≠ Date (276 rows) | Recomputed from Date | Business rule R1: Date is authoritative |
| T3 | 2062-xx (30 rows) | Corrected to 2026-xx when Year col=2026 | Obvious digit transposition; Year col confirms intent |
| T4 | 3,682 duplicate Record_IDs | Kept latest Load_Timestamp | Business rule R2 |
| T5 | 952 blank regions, 35+ label variants | Explicit dictionary map; blanks recovered from SKU+Customer+Date match | Unknown values fail loudly (no silent pass-through) |
| T8 | 2,992 rows with non-canonical SKU_ID | Strip→upper→replace_→insert dash→zero-pad→validate | SKU_ID is a foreign key; must join to sku_master |
| T11 | Dash "-" = not-reported, not 0 | Set to NULL (R3: missing ≠ zero) | Forecasting zeros as demand would bias the model downward |
| T11 | Negative Units_Sold (497 rows) | abs() applied; sign-entry error | Cannot have negative shipments |
| T12 | Zero prices on real orders (299 rows) | Filled with SKU-year median price | R4: price is defined per SKU per year |
| T12 | ×100 price band (386 rows) | Replaced with reference price | Likely cents-to-dollars unit-of-measure error |
| T12 | ×85 price band (231 rows) | Replaced with reference price | Plausible INR→USD denomination error; India region concentration confirmed |

### What we did NOT auto-fix (Section 3 items)

- **5–20× price band (390 rows)**: No single explanation fits. Flagged in anomaly register, excluded from price analysis, kept for demand analysis.
- **Large units >5,000 (32 rows)**: Cross-checked with stockout Requested_Units. Retained as-is pending manual review.
- **S1 BU conflict**: The one PMP SKU mapped to Power & Drive Systems is flagged; the transaction data would resolve this but it requires a business decision.

### Errors vs Real Events

- **COVID (2020-02 to 2020-06)**: Demand drops are real, model-able events. NOT corrected. Flagged with event flags (EV-009 to EV-013).
- **Early buying (Nov-Dec 2022)**: Legitimate demand pull-forward before the announced 2023 price rise (EV-030). NOT removed. Modelled as an event feature.
- **Stockout months**: Low/zero `Units_Sold` in train is constrained demand. `Requested_Units` from `inventory_stockouts.csv` is used as the true signal (unconstrained demand).
- **ERP migration (Latin America, 2023-03 to 2023-05)**: Low sales in those months may be missing data, not real demand drops. Flagged via `flag_system_event`; documented as a data gap.

---

## WS2 – Anomaly Detection

We distinguish three categories:
1. **Data errors** (T-rules): fixed with a documented rule
2. **Real demand events** (COVID, strikes, stockouts): preserved and modelled
3. **Ambiguous cases**: flagged in `anomaly_register.csv` with `action` field explaining the decision

For stockouts, we cross-validated `train.Units_Sold` against `stockouts.Shipped_Units`. The match rate is reported in the DQ report.

For zero-demand runs (3+ consecutive months), we cross-check against the stockout file. Unexplained long zero runs are flagged as potential hidden stockouts.

---

## WS3 – Feature Engineering

### Calendar features
- Month/quarter encoded cyclically (sin/cos) to capture seasonality without boundary effects
- Region-specific holiday flags: Lunar New Year (Asia Pacific/India), Ramadan (MEA/India), European summer shutdown

### Lag and rolling features
- Lags: 1, 2, 3, 6, 12 months (capturing monthly, quarterly, and annual cycles per R6)
- Rolling mean and std: 3m, 6m, 12m windows (smoothed signal + volatility)
- YoY lag (12m) for year-over-year comparison

### Intermittency classification (ADI/CV²)
Per SKU-Region-Customer series:
- **Smooth** (ADI≤1.32, CV²≤0.49): standard time series methods
- **Erratic** (ADI≤1.32, CV²>0.49): high variability but frequent
- **Intermittent** (ADI>1.32, CV²≤0.49): infrequent but stable when it occurs → Croston/SBA
- **Lumpy** (ADI>1.32, CV²>0.49): infrequent and variable → TSB

### Macro indicators
8 macro indicators merged with lags 0–3 months. The best lag per product family was identified by cross-correlation analysis (see `macro_lag_analysis.csv`). The EV Adoption Index was specifically compared against EV/Universal vs ICE SKU sales trends.

### Event features
Binary flags for all event types that affect demand: price changes, disruptions, supply constraints, system events, holiday shutdowns, early buying windows, and customer-specific events (EV-062 Orion retooling).

---

## WS4 – Forecasting

### Model hierarchy
1. **Baseline 1**: Naive seasonal (last-year same-month average per SKU-Region-Customer)
2. **Baseline 2**: 12-month trailing moving average
3. **Intermittent methods**: Croston, SBA, TSB (applied by intermittency class)
4. **Main model**: LightGBM quantile regression for P10/P50/P90

### Why LightGBM?
- Handles mixed categorical/numerical features natively
- Supports quantile regression directly (no post-hoc intervals)
- Robust to outliers and irregular time series
- Fast to train and tune even on 240k+ rows

### Backtest strategy
Rolling-origin backtest: train on data up to 2025-03, evaluate on 2025-04 to 2026-10 (19 months). This exactly mirrors the forecast window length. Metrics: WAPE and bias by family, region, and horizon month.

### Lifecycle rules (R5, Expert)
- Zero forecast after `EOL_Date`
- Zero forecast before `Launch_Date`
- New launches (PMP-020, PTC-029, THM-012) inherit historical pattern from `Replaces_SKU`

### Event adjustments in forecast window
- **EV-062**: Orion Driveline NA retooling (2027-07 to 2027-08) → zero forecast
- **EV-059**: 2027 price rise anticipated demand pull-forward in 2026-11/12 → modelled via event feature
- Planned Lunar New Year, Ramadan, and European summer shutdown months flagged for all 19 forecast months

### Hierarchy reconciliation
Forecasts verified to sum consistently at SKU → Family → Region → Total level within 5% tolerance.

---

## WS5 – Dashboard

### Design principles
- All views share the same filters: Family, SKU, Region, Channel, Customer
- Numbers always read from pipeline output files (never hard-coded)
- History and forecast shown on a **single continuous timeline** with forecast clearly demarcated

### Required views
1. **Executive overview**: Units and revenue trend 2019-01 to 2028-05; mix charts by family/region/channel
2. **Data health**: DQ issues by type with before/after counts from `data_quality_report.json`
3. **Demand explorer**: Drill-down from family → SKU → region → customer; seasonality; zero months; lifecycle
4. **Anomaly monitor**: Flagged points from `anomaly_register.csv` on the time series
5. **Forecast & accuracy**: Forecast with P10/P50/P90 bands vs history; WAPE and bias from `backtest_metrics.csv`

### Optional views built (Advanced tier: 2)
1. **Macro lead/lag heatmap** (View 1): Cross-correlation matrix from `macro_lag_analysis.csv`
   - *Why*: Directly answers "which macro indicator drives which family and at what lag" – high business value for procurement planning
   - *Dependency*: `macro_lag_analysis.csv` from WS3
2. **Stockouts and lost sales** (View 5): Requested vs Shipped vs Lost units from `inventory_stockouts.csv` and `unconstrained_demand.parquet`
   - *Why*: Shows planners exactly where supply failure destroyed revenue; directly supports the control-tower use case
   - *Dependency*: `inventory_stockouts.csv` and `unconstrained_demand.parquet`

---

## Limitations and Honest Assessment

- Latin America 2023-03 to 2023-05 (ERP migration): model may underforecast; flagged but not corrected
- 390 rows with 5–20× price ratio: unresolved; excluded from revenue analysis
- Intermittent SKU-region series: LightGBM may underperform vs Croston/SBA; both outputs available
- Forecast uncertainty grows with horizon (months 13-19 have higher WAPE than months 1-6)
- COVID-era training data creates a structural break; model weights post-2021 data more heavily via recent lags

