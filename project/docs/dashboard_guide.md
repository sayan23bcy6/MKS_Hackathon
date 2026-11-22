# Demand Forecasting Control Tower - Dashboard Guide

## Running the Dashboard Locally

Ensure the pipeline has generated all required output files in `project/outputs/`:

```bash
python project/run_pipeline.py
```

Then start the Streamlit dashboard application:

```bash
streamlit run project/app/dashboard.py
```

The application will be accessible at `http://localhost:8501`.

The public dashboard URL is not considered deployment-verified until it responds with HTTP 200.

## Deployment to Streamlit Community Cloud (Steps to Follow)

To deploy the dashboard to Streamlit Community Cloud:
1. Connect your GitHub repository (`sayan23bcy6/MKS_Hackathon`).
2. Set the main file path to `project/app/dashboard.py`.
3. Set the requirements file path to `project/requirements.txt`.
4. Deploy application and update the public link in `project/README.md`.

## Required Committed Outputs

The dashboard and deployment depend on the following outputs tracked in version control (`git ls-files`):

| Output File | Size | Description |
|---|---|---|
| `clean_train.parquet` | ~6.0 MB | Cleaned commercial sales transaction history |
| `clean_macro.csv` | < 0.1 MB | Cleaned macroeconomic indicators and lead indices |
| `clean_events.csv` | < 0.1 MB | Scoped corporate and market events calendar |
| `audit_log.csv` | ~12.5 MB | Complete deterministic audit log of every corrected record |
| `data_quality_report.json` | < 0.1 MB | Machine-readable data health and rule counts |
| `data_quality_report.md` | < 0.1 MB | Human-readable data governance report |
| `anomaly_register.csv` | ~0.3 MB | Documented market disruptions and operational shocks |
| `unconstrained_demand.parquet` | ~6.0 MB | True unconstrained demand with stockouts added back |
| `inventory_stockouts.csv` | ~0.5 MB | Historical stockout transactions and fill rate metrics |
| `features.parquet` | ~13.0 MB | Feature store including lags, rolling statistics, and macro factors |
| `forecast_submission.csv` | ~1.8 MB | 19-month forward commercial forecast submission |
| `forecast_intervals.csv` | ~2.1 MB | Quantile forecast intervals P10, P50, and P90 |
| `hierarchy_addition.csv` | < 0.1 MB | Exact bottom-up hierarchy addition across families, regions and business units |
| `model_card.md` | < 0.1 MB | Model architecture, calibration, and governance card |
| `backtest_metrics.csv` | < 0.1 MB | Rolling-origin backtest slices across 3 evaluation cutoffs |
| `coverage_breakdown.csv` | < 0.1 MB | P10-P90 interval empirical coverage by origin and family |
| `panel_status.csv` | ~18.0 MB | Complete panel series status, intermittent flags, and zero-demand counts |
| `master_data_contradictions.csv` | < 0.1 MB | Reconciliation log of contradictory catalog metadata |
| `early_buying.csv` | < 0.1 MB | Measured volume advance purchases ahead of price rises |
| `holiday_effects.csv` | < 0.1 MB | Regional holiday demand shifts (LNY, Ramadan, Summer shutdowns) |
| `structural_breaks.csv` | < 0.1 MB | Statistical structural breaks in component demand profiles |
| `customer_switches.csv` | < 0.1 MB | Account supplier switching and vendor consolidation tracking |
| `lost_accounts.csv` | < 0.1 MB | Account attrition and churn detection register |
| `macro_lag_analysis.csv` | < 0.1 MB | Macroeconomic lead-lag cross-correlations (lags 0 to 6 months) |
| `lag_stability.csv` | < 0.1 MB | Macro lead stability evaluation comparing historical regimes |
| `ev_shift_analysis.csv` | < 0.1 MB | Internal combustion vs electric powertrain transition analysis |
| `whatif_scenarios.csv` | < 0.1 MB | Macroeconomic PMI shock grid simulations (-6.0 to +6.0 points) |
| `whatif_scenario_forecast.csv` | ~2.0 MB | Full SKU-region forecast for the target -3.0 PMI recession scenario |
| `event_impact_estimates.csv` | < 0.1 MB | Operational event adjustment and capacity uplift calibration |

> Note: `backtest_predictions.csv` (~66 MB) is excluded from version control to prevent repository bloat; all dashboard visualisations rely strictly on `coverage_breakdown.csv` and `backtest_metrics.csv`.

---

## Error Fallback and Troubleshooting

If the Streamlit dashboard fails to display or encounters missing data:
1. **Verify output generation:** Ensure all required pipeline outputs have been created by running:
   ```bash
   python project/run_pipeline.py
   ```
2. **Missing or stale cache:** Clear Streamlit cache via the top-right menu (`Clear cache`) or restart Streamlit:
   ```bash
   streamlit run project/app/dashboard.py --server.runOnSave true
   ```
3. **Port conflicts:** If port 8501 is occupied, specify an alternative port:
   ```bash
   streamlit run project/app/dashboard.py --server.port 8502
   ```
4. **Offline fallback:** All underlying analytical artifacts exist as standalone CSV, Parquet, JSON, and Markdown files in `project/outputs/` and can be inspected directly in Excel, pandas, or text viewers without running Streamlit.

---

## Global Filter Behavior

The sidebar provides shared filters that update all active dashboard views simultaneously:
- **Product Family**: Filter by one or multiple component categories.
- **SKU ID**: Multiselect list populated directly from sales history and master records. Leave empty to include all SKUs.
- **Region**: Filter by operational geographic market (including `Unassigned` records).
- **Sales Channel**: Direct OEM vs Tier 1 Supplier.
- **Customer Name**: Specific customer accounts populated from historical transactions. Leave empty to include all customers.

---

## Navigation & Views

The dashboard organizes operational demand intelligence into nine distinct tabs:

### 1. Executive Overview
- Visualizes historical actual shipments alongside the 19-month point forecast (P50) and prediction intervals (P10-P90).
- Summarizes total historical units, 19-month forward forecast volume, trailing 12-month baseline, and projected run-rate shift.

### 2. Data Health & Audit Log
- Displays the complete audit log count and anomalous event register.
- Presents tabular summaries of each data cleaning rule (T1 through T15, R5, R9, M1-M5, S1, E1-E4).
- Details all documented decision rules, count of affected records, and business rationale.

### 3. Demand Explorer
- Slice-and-dice demand profiles by Product Family, Region, Sales Channel, or Customer Name.
- Analyzes annual seasonality curves and zero-demand proportions across SKU cohorts.

### 4. Anomaly Monitor
- Overlays real-world market events on sales trajectories (COVID-19 market shocks, pre-announcement early buying, plant retooling, and documented stockouts).
- Provides searchable records from the anomaly register.

### 5. Forecast & Model Accuracy
- Displays rolling-origin backtest results evaluated across 3 cutoff points (2023-10, 2024-10, 2025-03).
- Compares WAPE and forecast bias between baseline benchmarks (Naive Seasonal, Moving Average) and LightGBM P50.
- Tracks forecast error degradation over horizons 1 through 19.

### 6. Macro Lead/Lag Analysis
- Interactive heatmap showing Pearson cross-correlations between leading macroeconomic indicators (ISM PMI, Auto Production YoY, Raw Materials Index) and product family demand.
- Slider allows exploring lag horizons from 0 to 6 months.

### 7. Stockouts & Lost Demand
- Compares requested customer demand, actual shipments, and lost units from inventory stockout events.
- Evaluates operational order fill rate and unconstrained market volume.

### 8. What-If Macro Scenario
- Interactive sensitivity slider allowing users to simulate shifts in ISM Manufacturing PMI (-6.0 to +6.0 points).
- Shows real-time recalculation of baseline vs shifted demand trajectories across product families.

### 9. Lifecycle & Powertrain Transition View
- Monitors the volume progression of internal combustion engine (ICE) vs electric vehicle (EV) components.
- Displays regression and correlation metrics between EV market penetration and component demand.

---

## Data Source Integrity

In compliance with Ground Rule 7, the dashboard reads exclusively from output files generated in `project/outputs/`:
- `clean_train.parquet`
- `forecast_submission.csv`
- `forecast_intervals.csv`
- `backtest_metrics.csv`
- `data_quality_report.json`
- `anomaly_register.csv`
- `macro_lag_analysis.csv`
- `inventory_stockouts.csv`
- `unconstrained_demand.parquet`
- `ev_shift_analysis.csv`
- `whatif_scenario_forecast.csv`

No numbers are hard-coded in the presentation layer.
