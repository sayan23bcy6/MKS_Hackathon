# Celeris Automobili – Demand Forecasting & Planning Control Tower

## Quick Start (One Command)

```bash
# From the repo root (MKS_Hackathon/)
pip install -r project/requirements.txt
python project/run_pipeline.py
# Then launch the dashboard:
streamlit run project/app/dashboard.py
```

## Project Structure

```
MKS_Hackathon/
  student_data/           ← RAW, IMMUTABLE
  project/
    run_pipeline.py       ← ONE-COMMAND runner
    requirements.txt      ← pinned dependencies
    pipeline/
      config.py           ← all paths, label maps, thresholds
      load.py             ← read-only file loader
      audit.py            ← audit log singleton
      clean_train.py      ← T1–T15, R5 (train.csv)
      clean_master_macro_events.py  ← S1–S2, M1–M5, E1–E4
      unconstrained.py    ← unconstrained demand from stockouts
      anomalies.py        ← anomaly register (WS2)
      features.py         ← feature engineering (WS3)
      backtest.py         ← rolling-origin backtest (WS4)
      forecast.py         ← final forecast + lifecycle rules (WS4)
      reports.py          ← DQ report JSON + Markdown (WS1)
      models/
        baselines.py      ← Naive seasonal, Moving average
        croston.py        ← Croston, SBA, TSB
        lightgbm_model.py ← LightGBM quantile regression (P10/P50/P90)
    tests/
      test_cleaning_rules.py  ← pytest, one test per rule
    app/
      dashboard.py        ← Streamlit dashboard (WS5)
    outputs/              ← all pipeline outputs (auto-created)
    docs/
      methodology.md
      presentation_outline.md
```

## Assumptions

1. **Raw files are immutable** – never edited, only read.
2. **Date is authoritative** (R1) – Year and Month recomputed from Date.
3. **Missing row ≠ zero** (R3) – NULL demand rows excluded from training.
4. **Forecast is unconstrained** (R7) – Requested_Units used for stockout months.
5. **Prices are per SKU in USD** (R4) – same in every region, change 1 Jan only.
6. **Lifecycle rules enforced** (R5) – zero forecast before launch, after EOL.

## Reproducibility

All cleaning rules are deterministic and rule-based (no random seed needed for cleaning).
LightGBM training uses a fixed seed. Run the full pipeline with:

```bash
python project/run_pipeline.py
```

Outputs are written to `project/outputs/`. The dashboard reads **only** from this directory.

## Verification

After the pipeline runs, check:
- `project/outputs/forecast_submission.csv` – 58,219 rows, columns: Date, SKU_ID, Region, Units_Sold_Forecast
- `project/outputs/audit_log.csv` – one row per change
- `project/outputs/data_quality_report.md` – human-readable summary
- `project/outputs/data_quality_report.json` – machine-readable

Run tests:
```bash
cd project
pytest tests/ -v
```

