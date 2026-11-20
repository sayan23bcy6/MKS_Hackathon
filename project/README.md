# Celeris Automobili - Demand Forecasting Control Tower

An end-to-end demand forecasting pipeline and operational planning control tower built for the Celeris Automobili hackathon project.

The system cleans commercial sales transactions, reconstructs unconstrained demand from stockout records, computes calendar, lifecycle, and customer cycle features, trains LightGBM quantile regression models, and generates 19-month forecasts with prediction intervals and top-down reconciliation.

---

## Getting Started

### 1. Prerequisites and Installation

The project runs on Python 3.10+ with lightgbm, pandas, pyarrow, scikit-learn, plotly, streamlit, and pytest:

```bash
pip install -r project/requirements.txt
```

### 2. Run the Full Pipeline

Execute the entire data cleaning, feature engineering, model training, backtesting, and forecasting pipeline with one command:

```bash
python project/run_pipeline.py
```

This reads input data from `student_data/` and saves all cleaned datasets, models, metrics, and forecast files into `project/outputs/`.

### 3. Run the Test Suite

Run unit tests covering every data cleaning rule, feature transformation, and model constraint:

```bash
pytest project/tests -v
```

### 4. Launch the Dashboard

Open the interactive Streamlit Control Tower in your web browser:

```bash
streamlit run project/app/dashboard.py
```

The application runs locally at `http://localhost:8501`.

---

## Repository Structure

```
MKS_Hackathon/
  student_data/                 Raw input files (strictly read-only)
    train.csv                   Sales history (2019-01 to 2026-10)
    test.csv                    Forecast rows to predict (2026-11 to 2028-05)
    macro_indicators.csv        Monthly economic indicators
    sku_master.csv              Catalog of 601 SKUs with lifecycle dates
    events_calendar.csv         Operational events, disruptions, price changes
    inventory_stockouts.csv     Order fill rates and lost unit records
    sample_submission.csv       Target format verification
  project/
    run_pipeline.py             Master pipeline script
    requirements.txt            Python dependencies
    pipeline/
      config.py                 File paths, canonical label dictionaries, thresholds
      load.py                   Read-only loaders for student_data files
      audit.py                  Row-level audit logging mechanism
      clean_train.py            Rules T1-T15: dates, regions, prices, units, lifecycle
      clean_master_macro_events.py  Rules S1-S2, M1-M5, E1-E4
      unconstrained.py          Stockout adjustment and unconstrained demand recovery
      ordering_cycles.py        Customer ordering cadence and churn detection
      ev_shift.py               ICE vs EV adoption and powertrain transition analysis
      structural_breaks.py      Change-point detection and holiday impact analysis
      anomalies.py              Anomaly register assembly
      features.py               Feature engineering (lags, rollings, harmonics, events)
      backtest.py               3-origin rolling backtests (2023-10, 2024-10, 2025-03)
      forecast.py               Recursive 19-month forecasting and reconciliation
      reports.py                Data quality report generation (JSON & Markdown)
      models/
        baselines.py            Seasonal naive and rolling average benchmarks
        croston.py              Croston, SBA, and TSB intermittent methods
        lightgbm_model.py       P10, P50, P90 quantile gradient boosting models
    tests/
      test_cleaning_rules.py    Automated tests for every data cleaning rule
    app/
      dashboard.py              Streamlit dashboard with 9 analytical views
    outputs/                    Generated pipeline artifacts
    docs/
      methodology.md            Methodology and model explanation
      presentation_outline.md   Presentation structure and defense talking points
      dashboard_guide.md        Operator instructions for dashboard views and filters
      audit_of_existing_work.md Summary of modified, created, and audited files
```

---

## Key Design Principles

1. **Immutable Inputs**: Files in `student_data/` are never modified or overwritten. All transformations happen in memory and save to `project/outputs/`.
2. **Auditable Cleaning**: Every single change made to the data is recorded in `outputs/audit_log.csv` with the source file, row key, field, original value, cleaned value, rule ID, and reason.
3. **Explicit Decision Rules**: Every data quality issue has an explicit, documented resolution rule. Ambiguous cases are handled through transparent, tested rules rather than silent deletion.
4. **No Silent Exceptions**: Failures raise descriptive exceptions naming the step and cause.
5. **Human Code Style**: Written in plain, maintainable Python without decorative Unicode, emojis, or ASCII banners.
6. **Strict Output-Driven UI**: The Streamlit dashboard loads its data solely from `project/outputs/`. No hard-coded metrics exist in the presentation layer.

---

## Primary Pipeline Outputs

After running `run_pipeline.py`, the following files are produced in `project/outputs/`:

| Output File | Description |
|---|---|
| `forecast_submission.csv` | Final 19-month point forecast matching the 58,219 row submission format |
| `forecast_intervals.csv` | P10, P50, and P90 prediction intervals for uncertainty planning |
| `backtest_metrics.csv` | WAPE and bias scores evaluated across 3 origins and 19 horizons |
| `audit_log.csv` | Full row-level log of every value adjusted during cleaning |
| `data_quality_report.json` | Machine-readable breakdown of issues found and rules applied |
| `data_quality_report.md` | Human-readable documentation of data health |
| `hierarchy_reconciliation.csv` | Verification of bottom-up consistency within 0.5% tolerance |
| `whatif_scenario_forecast.csv` | Macroeconomic sensitivity simulation (ISM PMI shift) |
| `ev_shift_analysis.csv` | Powertrain technological transition progression |
| `macro_lag_analysis.csv` | Cross-correlations between economic indicators and family volume |
| `model_card.md` | Model governance documentation |
