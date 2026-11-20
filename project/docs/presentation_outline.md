# Presentation Outline – Celeris Automobili Demand Forecasting

## Slide 1: Title
**Celeris Automobili Demand Forecasting & Planning Control Tower**
- Team / name
- Hackathon date

---

## Slide 2: Problem Understanding
- **Business context**: Celeris supplies 7 product families to 12 customers across 6 regions. Poor demand forecasting → stockouts, lost revenue, overstocking.
- **Goal**: Forecast **unconstrained** demand for 58,219 SKU-Region-Customer rows, 2026-11 to 2028-05
- **Challenge**: 15+ data quality issues in raw data; real events vs errors; intermittent demand; macro drivers with lags

---

## Slide 3: Approach Overview
```
Raw data → Clean → Unconstrained Demand → Features → Model → Forecast → Dashboard
```
1. **WS1**: Profile all files; fix every issue with a rule; log every change
2. **WS2**: Anomaly register; separate errors from real events
3. **WS3**: 40+ features: calendar, macro lags, lifecycle, event flags, lags/rolling
4. **WS4**: 2 baselines + Croston/SBA/TSB + LightGBM P10/P50/P90; 19-month backtest
5. **WS5**: Streamlit dashboard, 5 required + 2 optional views

---

## Slide 4: Data Quality – Key Findings
| Issue | Rows | Fix |
|---|---|---|
| Mixed date formats (T1) | 10,000 | Explicit format regex |
| Duplicate Record_IDs (T4) | 7,364 | Keep latest timestamp (R2) |
| Region label variants (T5) | 5,000+ | Explicit map; fail on unknown |
| SKU_ID format errors (T8) | 2,992 | Strip→upper→insert dash→pad |
| Negative Units_Sold (T11) | 497 | abs(); logged as sign-entry error |
| Price ×100 band (T12) | 386 | Replace with SKU-year reference |
| Revenue ≠ Units×Price (T13) | 1,744 | Recompute after price/unit fix |

**Audit log**: every change recorded; before/after counts in DQ report.

---

## Slide 5: Errors vs Real Events
- **COVID (2020-04)**: Auto production fell 45.4%. **Not cleaned** – modelled with event flags.
- **Early buying (Nov-Dec 2022)**: Demand spike before announced 2023 price rise (EV-030). **Kept** – feature flag added.
- **Stockouts**: Low train units = constrained shipments. **Fixed** using `Requested_Units` from inventory_stockouts.
- **ERP migration (LatAm 2023-03 to 05)**: Possibly missing data. **Flagged** – documented limitation.

---

## Slide 6: Feature Engineering
- **Calendar**: month/quarter cyclically encoded; 3 regional holiday types
- **Lag/Rolling**: 1, 2, 3, 6, 12-month lags; 3m/6m/12m rolling mean & std
- **Macro (lags 0–3)**: 8 indicators; best lag per family identified by cross-correlation
- **Lifecycle**: months since launch, months to EOL, replacement chain flags
- **Events**: 11 event-type binary flags; Orion retooling, early-buy window
- **Intermittency**: ADI/CV² classification → Croston/SBA for intermittent, TSB for lumpy

---

## Slide 7: Model Performance
| Model | WAPE (19m horizon) | Bias |
|---|---|---|
| Naive Seasonal | *from backtest* | — |
| Moving Average | *from backtest* | — |
| **LightGBM P50** | **best** | — |

- LightGBM wins overall; Croston/SBA better for truly intermittent SKU-regions
- WAPE increases from month 1 to month 19 (uncertainty grows with horizon)
- Worst families: those with high macro sensitivity and ERP data gaps

---

## Slide 8: Forecast Highlights
- **Total 19-month forecast**: *from dashboard*
- **Top growth region**: MEA (market still maturing post-2022 entry)
- **EOL handling**: 9 discontinued/EOL-announced SKUs → zero forecast after EOL date
- **New launches**: PMP-020, PTC-029, THM-012 inherit demand pattern from replaced SKUs
- **EV-062 impact**: Orion Driveline NA → zero in 2027-07/08; forecast recovers in 2027-09

---

## Slide 9: Dashboard
*(Show live dashboard)*
1. **Executive Overview** – single timeline, mix charts, 19-month headline
2. **Data Health** – DQ issues, before/after counts
3. **Demand Explorer** – family → SKU → region drill-down
4. **Anomaly Monitor** – flagged points with reason and action
5. **Forecast & Accuracy** – P10/P50/P90 bands, WAPE by family/region/horizon
6. **Macro Lead/Lag Heatmap** – which indicator leads which family
7. **Stockouts & Lost Sales** – requested vs shipped vs lost

---

## Slide 10: Business Impact
- **Planner benefit**: Single timeline from history to forecast; see how forecast continues the trend
- **Procurement**: P90 forecasts provide conservative upper bound for safety stock
- **Anomaly awareness**: Events (EV-062 Orion retooling) automatically flagged in forecast
- **Data trust**: Every number traceable to audit_log.csv; no silent changes

---

## Slide 11: Limitations & Honest Assessment
- Latin America ERP gap (2023-03/05): may underforecast; model told via flag but limited signal
- 390 price rows with 5–20× ratio: unexplained; excluded from revenue, retained for demand
- Forecast uncertainty: WAPE grows with horizon (months 13–19 materially less accurate than 1–6)
- COVID structural break: model heavily influenced by post-2021 patterns; COVID pattern unlikely to repeat

---

## Slide 12: Reproducibility
```bash
pip install -r project/requirements.txt
python project/run_pipeline.py      # clean → features → forecast → reports
streamlit run project/app/dashboard.py
pytest project/tests/ -v            # 20+ unit tests, one per rule
```
All outputs in `project/outputs/`. Audit log has {N} entries. Zero hard-coded numbers.

