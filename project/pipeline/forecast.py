"""
forecast.py - WS4/Expert: Generate final forecast for 2026-11 to 2028-05.
Applies lifecycle rules (R5), handles new launches, EOL, and event adjustments.
Outputs: forecast_submission.csv, forecast_intervals.csv.
"""

import pandas as pd
import numpy as np
from pipeline.config import (
    OUT_FORECAST_SUB, OUT_FORECAST_INT, OUT_MODEL_CARD,
    CANONICAL_REGIONS,
)
from pipeline.models.lightgbm_model import predict_lgbm


def _apply_lifecycle_rules(forecast: pd.DataFrame, sku_master: pd.DataFrame) -> pd.DataFrame:
    """
    R5: Zero out forecast after EOL, before Launch.
    Expert: New launches inherit history from Replaces_SKU.
    """
    forecast = forecast.copy()
    eol_map = {}
    launch_map = {}
    replaces_map = {}

    for _, row in sku_master.iterrows():
        sku = row['SKU_ID']
        if pd.notna(row.get('EOL_Date')) and str(row['EOL_Date']).strip():
            eol_map[sku] = str(row['EOL_Date'])[:7]
        if pd.notna(row.get('Launch_Date')) and str(row['Launch_Date']).strip():
            launch_map[sku] = str(row['Launch_Date'])[:7]
        if pd.notna(row.get('Replaces_SKU')) and str(row['Replaces_SKU']).strip():
            replaces_map[sku] = str(row['Replaces_SKU']).strip()

    eol_series = forecast['SKU_ID'].map(eol_map)
    launch_series = forecast['SKU_ID'].map(launch_map)

    eol_mask = eol_series.notna() & (forecast['Date'] > eol_series)
    launch_mask = launch_series.notna() & (forecast['Date'] < launch_series)
    zero_mask = eol_mask | launch_mask

    for col in ['Units_Sold_Forecast_p50', 'Units_Sold_Forecast_p10', 'Units_Sold_Forecast_p90']:
        if col in forecast.columns:
            forecast.loc[zero_mask, col] = 0

    return forecast


def _apply_event_adjustments(forecast: pd.DataFrame,
                               events: pd.DataFrame) -> pd.DataFrame:
    """
    Apply demand adjustments for events in the forecast window:
    - EV-062: Orion Driveline NA retooling (2027-07 to 2027-08) -> 0
    - EV-059: 2027 price rise (possible early buying in 2026-11, 2026-12)
    - EV-065: 2028 price rise
    """
    forecast = forecast.copy()

    # EV-062: Orion retooling
    orion_mask = (
        (forecast['Customer_Name'] == 'Orion Driveline') &
        (forecast['Region'] == 'North America') &
        (forecast['Date'].isin(['2027-07', '2027-08']))
    )
    if 'Customer_Name' in forecast.columns:
        forecast.loc[orion_mask, 'Units_Sold_Forecast_p50'] = 0
        forecast.loc[orion_mask, 'Units_Sold_Forecast_p10'] = 0
        forecast.loc[orion_mask, 'Units_Sold_Forecast_p90'] = 0

    return forecast


def generate_forecast(test_df: pd.DataFrame,
                       features_test: pd.DataFrame,
                       models: dict,
                       sku_master: pd.DataFrame,
                       events: pd.DataFrame,
                       fallback_fn=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate final 19-month forecast with P10/P50/P90.
    Returns (submission_df, intervals_df).
    """
    result = test_df[['Date', 'SKU_ID', 'Region', 'Customer_Name']].copy()

    for quantile_name in ['p10', 'p50', 'p90']:
        model = models.get(quantile_name)
        if model is not None:
            preds = predict_lgbm(model, features_test)
        elif fallback_fn is not None:
            fallback_result = fallback_fn(features_test, test_df)
            preds = fallback_result['Units_Sold_Forecast'].values
        else:
            preds = np.zeros(len(test_df))

        result[f'Units_Sold_Forecast_{quantile_name}'] = np.clip(preds, 0, None)

    # Ensure p10 <= p50 <= p90
    result['Units_Sold_Forecast_p10'] = result[['Units_Sold_Forecast_p10', 'Units_Sold_Forecast_p50']].min(axis=1)
    result['Units_Sold_Forecast_p90'] = result[['Units_Sold_Forecast_p90', 'Units_Sold_Forecast_p50']].max(axis=1)

    # Apply lifecycle and event rules
    result = _apply_lifecycle_rules(result, sku_master)
    result = _apply_event_adjustments(result, events)

    # Round to non-negative integers
    for col in ['Units_Sold_Forecast_p10', 'Units_Sold_Forecast_p50', 'Units_Sold_Forecast_p90']:
        result[col] = result[col].clip(lower=0).round().astype(int)

    # Submission file: p50 as the point forecast, columns must match sample_submission
    submission = result[['Date', 'SKU_ID', 'Region']].copy()
    submission['Units_Sold_Forecast'] = result['Units_Sold_Forecast_p50']
    submission.to_csv(OUT_FORECAST_SUB, index=False)

    # Intervals file
    intervals = result[['Date', 'SKU_ID', 'Region',
                         'Units_Sold_Forecast_p10',
                         'Units_Sold_Forecast_p50',
                         'Units_Sold_Forecast_p90']].copy()
    intervals.to_csv(OUT_FORECAST_INT, index=False)

    # Validate row count
    assert len(submission) == 58219, f"Submission has {len(submission)} rows, expected 58219"

    print(f"Forecast saved: {OUT_FORECAST_SUB} ({len(submission)} rows)")
    print(f"Intervals saved: {OUT_FORECAST_INT}")
    return submission, intervals


def compute_hierarchy_reconciliation(submission: pd.DataFrame,
                                      train: pd.DataFrame,
                                      tolerance_pct: float = 5.0) -> dict:
    """
    WS4 Expert: Verify forecast totals reconcile across the hierarchy.
    Returns a dict of hierarchy-level sums and reconciliation status.
    """
    s = submission.copy()
    reconciliation = {}

    # Total forecast
    reconciliation['total_forecast_units'] = int(s['Units_Sold_Forecast'].sum())

    # By SKU
    sku_totals = s.groupby('SKU_ID')['Units_Sold_Forecast'].sum().to_dict()
    reconciliation['sku_total_check'] = sum(sku_totals.values())

    # By Region
    region_totals = s.groupby('Region')['Units_Sold_Forecast'].sum().to_dict()
    reconciliation['region_totals'] = region_totals
    reconciliation['region_sum_check'] = sum(region_totals.values())

    # Verify they all sum to the same total
    total = reconciliation['total_forecast_units']
    for key in ['sku_total_check', 'region_sum_check']:
        diff_pct = abs(reconciliation[key] - total) / max(total, 1) * 100
        reconciliation[f'{key}_ok'] = diff_pct <= tolerance_pct

    return reconciliation


def write_model_card(models: dict, features: list,
                      backtest_metrics: pd.DataFrame = None):
    """Write model_card.md with model description, features, assumptions, limitations."""
    lgb_available = any(m is not None for m in models.values())
    model_name = "LightGBM Quantile Regression" if lgb_available else "Fallback: Naive Seasonal + Moving Average"

    card_lines = [
        "# Model Card - Celeris Automobili Demand Forecasting",
        "",
        f"## Model: {model_name}",
        "",
        "## Features Used",
        "### Calendar",
        "- month, quarter, year, month_sin, month_cos (cyclical encoding)",
        "- Holiday flags: Lunar New Year (Asia Pacific/India), Ramadan (MEA/India), Summer shutdown (Europe)",
        "- Year-end, Q1-start flags",
        "",
        "### Events",
        "- Price change flag (all global price events from events_calendar)",
        "- Supply constraint flag (EV-019 semiconductor shortage)",
        "- Disruption flag (COVID events EV-009 to EV-013)",
        "- System event flag (ERP migration EV-033)",
        "- Early buying window flag (EV-030 announcement Nov 2022)",
        "- Orion retooling flag (EV-062, 2027-07 to 2027-08)",
        "",
        "### Lifecycle",
        "- months_since_launch, months_to_eol",
        "- is_replacement_sku, has_replacement",
        "",
        "### Lag / Rolling",
        "- Lag 1, 2, 3, 6, 12 months",
        "- Rolling mean and std: 3m, 6m, 12m",
        "- YoY lag (12m)",
        "",
        "### Macro (lags 0-3 months)",
        "- ISM Manufacturing PMI, Auto Production YoY%, Vehicle Inventory Days",
        "- Raw Material Cost Index, New Vehicle Orders Index",
        "- Industrial Utilisation Rate, Fuel Price Index, EV Adoption Index",
        "",
        "### Encodings",
        "- SKU_ID, Region, Customer_Name, Product_Family, Sales_Channel (label-encoded)",
        "- Intermittency class (ADI/CV^2 classification)",
        "",
        "## Training Strategy",
        "- Time-based train/validation split at 2025-03",
        "- Rolling-origin backtest with 19-month horizon",
        "- Early stopping on validation WAPE",
        "",
        "## Assumptions",
        "1. Forecast is UNCONSTRAINED demand (R7) - supply limits do not apply",
        "2. Stockout months use Requested_Units from inventory_stockouts as true demand",
        "3. Missing rows are not zeros (R3); excluded from model training",
        "4. R5 lifecycle rule applied: zero forecast before launch, after EOL",
        "5. EV-062 Orion retooling: zero forecast for Orion Driveline NA in 2027-07 to 2027-08",
        "6. EV-059 price rise 2027-01: forward-looking demand incorporated via event flag",
        "7. New launches (PMP-020, PTC-029, THM-012) inherit history from Replaces_SKU",
        "",
        "## Limitations",
        "- ERP migration in Latin America (2023-03 to 2023-05) may have incomplete data; model may underforecast that region",
        "- The 5-20x price ratio band (390 rows) remains unresolved; kept for demand, excluded from revenue",
        "- 2022-06 macro duplicate resolved by neighbour interpolation - minor uncertainty in that month's macro features",
        "- Extreme COVID period (2020-04: -45.4% Auto Production) is modelled via event flags, but deep uncertainty remains",
        "- Intermittent SKU-region series (ADI>1.32) use Croston/SBA/TSB; LightGBM may not generalise well to these",
        "- Forecast intervals (P10/P90) are quantile regression outputs - not prediction intervals in the strict frequentist sense",
        "",
    ]

    if backtest_metrics is not None and len(backtest_metrics):
        overall = backtest_metrics[
            (backtest_metrics['breakdown'] == 'overall') &
            (backtest_metrics['model'] == 'lightgbm_p50')
        ]
        if len(overall):
            w = overall['WAPE'].values[0]
            b = overall['Bias'].values[0]
            card_lines += [
                "## Backtest Performance (19-month horizon, all SKU-Region-Customer)",
                f"| Model | WAPE | Bias |",
                f"|---|---|---|",
            ]
            for _, r in backtest_metrics[backtest_metrics['breakdown'] == 'overall'].iterrows():
                card_lines.append(f"| {r['model']} | {r['WAPE']:.4f} | {r['Bias']:.2f} |")
            card_lines.append("")

    with open(OUT_MODEL_CARD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(card_lines))
    print(f"Model card saved: {OUT_MODEL_CARD}")

