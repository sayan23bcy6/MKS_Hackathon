"""
backtest.py - Rolling-origin evaluation across multi-year origins.
Tests 19-month forecast horizons across multiple cutoff points (2023-10, 2024-10, 2025-03).
Calculates WAPE and Bias by origin, horizon month, product family, region, and channel.
"""

import numpy as np
import pandas as pd
from pipeline.config import OUT_BACKTEST
from pipeline.models.lightgbm_model import compute_wape, compute_bias


def evaluate_backtest_slice(actual_series: pd.Series, pred_series: pd.Series,
                            origin: str, model_name: str, breakdown: str,
                            group_name: str, horizon_month: int = -1) -> dict:
    """Compute WAPE and Bias metrics for an evaluation slice."""
    wape = compute_wape(actual_series, pred_series)
    bias = compute_bias(actual_series, pred_series)
    return {
        'origin': origin,
        'model': model_name,
        'breakdown': breakdown,
        'group': group_name,
        'horizon_month': horizon_month,
        'WAPE': round(wape, 4) if pd.notna(wape) else np.nan,
        'Bias': round(bias, 2) if pd.notna(bias) else np.nan,
        'n_obs': len(actual_series),
    }


def run_origin_evaluation(df: pd.DataFrame,
                          model_fn,
                          origin: str,
                          horizon: int = 19,
                          model_name: str = 'model') -> list:
    """Run 19-month backtest from a specific origin date."""
    target_col = 'Units_Sold_Unconstrained' if 'Units_Sold_Unconstrained' in df.columns else 'Units_Sold'
    all_dates = sorted(df['Date'].unique())

    if origin not in all_dates:
        return []

    origin_idx = all_dates.index(origin)
    train_dates = all_dates[:origin_idx + 1]
    test_dates = all_dates[origin_idx + 1: origin_idx + 1 + horizon]

    if not test_dates:
        return []

    train_df = df[df['Date'].isin(train_dates)].copy()
    test_df = df[df['Date'].isin(test_dates)].copy()

    preds = model_fn(train_df, test_df)
    test_df = test_df.copy()

    if isinstance(preds, (np.ndarray, list)):
        test_df['pred'] = np.array(preds)
    elif isinstance(preds, pd.Series):
        test_df['pred'] = preds.values
    elif isinstance(preds, pd.DataFrame) and 'Units_Sold_Forecast' in preds.columns:
        if len(preds) == len(test_df):
            test_df['pred'] = preds['Units_Sold_Forecast'].values
        else:
            merge_cols = [c for c in ['Date', 'SKU_ID', 'Region', 'Customer_Name'] if c in preds.columns and c in test_df.columns]
            merged = test_df.merge(
                preds[merge_cols + ['Units_Sold_Forecast']].drop_duplicates(merge_cols),
                on=merge_cols, how='left'
            )
            test_df['pred'] = merged['Units_Sold_Forecast'].fillna(0).values
    else:
        test_df['pred'] = 0

    test_df['pred'] = test_df['pred'].clip(lower=0)
    test_df['horizon_month'] = test_df['Date'].apply(
        lambda d: test_dates.index(d) + 1 if d in test_dates else -1
    )

    records = []

    # Overall slice
    records.append(evaluate_backtest_slice(
        test_df[target_col], test_df['pred'], origin, model_name, 'overall', 'ALL'
    ))

    # Horizon breakdown (1 to 19)
    for h, grp in test_df.groupby('horizon_month'):
        records.append(evaluate_backtest_slice(
            grp[target_col], grp['pred'], origin, model_name, 'horizon', f"month_{h}", horizon_month=int(h)
        ))

    # Family breakdown
    if 'Product_Family' in test_df.columns:
        for fam, grp in test_df.groupby('Product_Family'):
            records.append(evaluate_backtest_slice(
                grp[target_col], grp['pred'], origin, model_name, 'family', str(fam)
            ))

    # Region breakdown
    if 'Region' in test_df.columns:
        for reg, grp in test_df.groupby('Region'):
            records.append(evaluate_backtest_slice(
                grp[target_col], grp['pred'], origin, model_name, 'region', str(reg)
            ))

    # Sales Channel breakdown
    if 'Sales_Channel' in test_df.columns:
        for ch, grp in test_df.groupby('Sales_Channel'):
            records.append(evaluate_backtest_slice(
                grp[target_col], grp['pred'], origin, model_name, 'channel', str(ch)
            ))

    return records


def run_all_backtests(features_df: pd.DataFrame,
                      baseline1_fn,
                      baseline2_fn,
                      lgbm_fn,
                      origins: list = ['2023-10', '2024-10', '2025-03']) -> pd.DataFrame:
    """
    Run rolling-origin backtests across 3 origins and all candidate models.
    Saves outputs/backtest_metrics.csv.
    """
    models = [
        ('naive_seasonal', baseline1_fn),
        ('moving_average', baseline2_fn),
        ('lightgbm_p50', lgbm_fn),
    ]

    all_records = []
    for origin in origins:
        print(f"  Evaluating backtest origin: {origin}...")
        for name, fn in models:
            records = run_origin_evaluation(
                features_df, fn, origin=origin, horizon=19, model_name=name
            )
            all_records.extend(records)

    results_df = pd.DataFrame(all_records)
    results_df.to_csv(OUT_BACKTEST, index=False)
    print(f"Backtest metrics saved to {OUT_BACKTEST} ({len(results_df)} evaluation slices)")
    return results_df
