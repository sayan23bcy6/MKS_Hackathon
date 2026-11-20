"""
backtest.py - WS4: Rolling-origin backtest with 19-month horizon.
Computes WAPE and bias by model, family, region, and horizon month.
"""

import pandas as pd
import numpy as np
from pipeline.config import OUT_BACKTEST
from pipeline.models.lightgbm_model import compute_wape, compute_bias


def rolling_origin_backtest(features_df: pd.DataFrame,
                             model_fn,
                             train_cutoff: str = '2025-03',
                             horizon: int = 19,
                             model_name: str = 'model') -> pd.DataFrame:
    """
    Rolling-origin backtest: train on data up to `train_cutoff`,
    forecast the next `horizon` months, evaluate WAPE and bias.

    model_fn: callable(train_df, test_df) -> pd.Series of predictions indexed as test_df.index
    """
    df = features_df.copy()
    df['Units_Sold_Unconstrained'] = pd.to_numeric(df.get('Units_Sold_Unconstrained', df.get('Units_Sold', 0)), errors='coerce').fillna(0)

    all_dates = sorted(df['Date'].unique())
    cutoff_idx = all_dates.index(train_cutoff) if train_cutoff in all_dates else -horizon - 1

    train_dates = all_dates[:cutoff_idx + 1]
    test_dates = all_dates[cutoff_idx + 1: cutoff_idx + 1 + horizon]

    if not test_dates:
        print(f"Backtest: no test dates found after {train_cutoff}")
        return pd.DataFrame()

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

    # Overall metrics
    rows = []
    overall_wape = compute_wape(test_df['Units_Sold_Unconstrained'], test_df['pred'])
    overall_bias = compute_bias(test_df['Units_Sold_Unconstrained'], test_df['pred'])
    rows.append({
        'model': model_name, 'breakdown': 'overall',
        'group': 'ALL', 'horizon_month': -1,
        'WAPE': overall_wape, 'Bias': overall_bias,
        'n_obs': len(test_df),
    })

    # By Product_Family
    for fam, grp in test_df.groupby('Product_Family'):
        rows.append({
            'model': model_name, 'breakdown': 'family',
            'group': fam, 'horizon_month': -1,
            'WAPE': compute_wape(grp['Units_Sold_Unconstrained'], grp['pred']),
            'Bias': compute_bias(grp['Units_Sold_Unconstrained'], grp['pred']),
            'n_obs': len(grp),
        })

    # By Region
    for reg, grp in test_df.groupby('Region'):
        rows.append({
            'model': model_name, 'breakdown': 'region',
            'group': reg, 'horizon_month': -1,
            'WAPE': compute_wape(grp['Units_Sold_Unconstrained'], grp['pred']),
            'Bias': compute_bias(grp['Units_Sold_Unconstrained'], grp['pred']),
            'n_obs': len(grp),
        })

    # By horizon month
    for h, grp in test_df.groupby('horizon_month'):
        rows.append({
            'model': model_name, 'breakdown': 'horizon',
            'group': f'month_{h}', 'horizon_month': h,
            'WAPE': compute_wape(grp['Units_Sold_Unconstrained'], grp['pred']),
            'Bias': compute_bias(grp['Units_Sold_Unconstrained'], grp['pred']),
            'n_obs': len(grp),
        })

    return pd.DataFrame(rows)


def run_all_backtests(features_df: pd.DataFrame,
                       baseline1_fn,
                       baseline2_fn,
                       lgbm_fn) -> pd.DataFrame:
    """Run backtest for all three models and concatenate results."""
    results = []
    for fn, name in [
        (baseline1_fn, 'naive_seasonal'),
        (baseline2_fn, 'moving_average'),
        (lgbm_fn, 'lightgbm_p50'),
    ]:
        print(f"Backtesting {name}...")
        r = rolling_origin_backtest(features_df, fn, model_name=name)
        results.append(r)

    all_results = pd.concat(results, ignore_index=True)
    all_results.to_csv(OUT_BACKTEST, index=False)
    print(f"Backtest metrics saved: {OUT_BACKTEST}")
    return all_results

