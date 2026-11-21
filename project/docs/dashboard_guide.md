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

## Deployment to Streamlit Community Cloud (Steps to Follow)

To deploy the dashboard to Streamlit Community Cloud:
1. Connect your GitHub repository (`sayan23bcy6/MKS_Hackathon`).
2. Set the main file path to `project/app/dashboard.py`.
3. Set the requirements file path to `project/requirements.txt`.
4. Deploy application and update the public link in `project/README.md`.

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
