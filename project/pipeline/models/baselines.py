"""
models/baselines.py - WS4 baselines: Naive seasonal (last-year same month) and
Moving Average (12-month trailing mean).
"""

import pandas as pd
import numpy as np


def naive_seasonal_forecast(train: pd.DataFrame,
                             test_keys: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline 1: Naive seasonal - use same SKU-Region-Customer-Month from prior year.
    """
    df = train.copy()
    df['Units_Sold_num'] = pd.to_numeric(df['Units_Sold'], errors='coerce').fillna(0)
    df['month'] = df['Date'].str[5:7].astype(int)

    # Build lookup: (SKU, Region, Customer, Month) -> mean(Units_Sold)
    lookup = df.groupby(
        ['SKU_ID', 'Region', 'Customer_Name', 'month']
    )['Units_Sold_num'].mean().reset_index()
    lookup.columns = ['SKU_ID', 'Region', 'Customer_Name', 'month', 'forecast']

    test = test_keys.copy()
    test['month'] = test['Date'].str[5:7].astype(int)
    result = test.merge(lookup, on=['SKU_ID', 'Region', 'Customer_Name', 'month'], how='left')
    result['forecast'] = result['forecast'].fillna(0).clip(lower=0).round().astype(int)
    result.columns = [c if c != 'forecast' else 'Units_Sold_Forecast' for c in result.columns]
    return result[['Date', 'SKU_ID', 'Region', 'Customer_Name', 'Units_Sold_Forecast']]


def moving_average_forecast(train: pd.DataFrame,
                             test_keys: pd.DataFrame,
                             window: int = 12) -> pd.DataFrame:
    """
    Baseline 2: 12-month trailing moving average per SKU-Region-Customer.
    """
    df = train.copy()
    df['Units_Sold_num'] = pd.to_numeric(df['Units_Sold'], errors='coerce').fillna(0)
    df = df.sort_values(['SKU_ID', 'Region', 'Customer_Name', 'Date'])

    # Last `window` months average
    lookup = df.groupby(
        ['SKU_ID', 'Region', 'Customer_Name']
    )['Units_Sold_num'].apply(lambda s: s.tail(window).mean()).reset_index()
    lookup.columns = ['SKU_ID', 'Region', 'Customer_Name', 'forecast']

    test = test_keys.copy()
    result = test.merge(lookup, on=['SKU_ID', 'Region', 'Customer_Name'], how='left')
    result['forecast'] = result['forecast'].fillna(0).clip(lower=0).round().astype(int)
    result.columns = [c if c != 'forecast' else 'Units_Sold_Forecast' for c in result.columns]
    return result[['Date', 'SKU_ID', 'Region', 'Customer_Name', 'Units_Sold_Forecast']]

