"""
models/croston.py - Intermittent demand methods: Croston, SBA (Syntetos-Boylan),
and TSB (Teunter-Syntetos-Boylan).
"""

import pandas as pd
import numpy as np


def croston_method(series: pd.Series, alpha: float = 0.1) -> float:
    """
    Croston's method for intermittent demand.
    Returns one-step-ahead forecast of demand per period.
    """
    series = series.dropna().values
    if len(series) == 0:
        return 0.0

    # Initialise
    nz = series[series > 0]
    if len(nz) == 0:
        return 0.0

    z = float(nz[0])  # demand estimate
    # Inter-demand interval estimate
    intervals = []
    last_demand = 0
    for i, v in enumerate(series):
        if v > 0:
            if last_demand > 0:
                intervals.append(i - last_demand)
            last_demand = i
    q = float(np.mean(intervals)) if intervals else 1.0

    for v in series[1:]:
        if v > 0:
            z = alpha * v + (1 - alpha) * z
            q = alpha * 1 + (1 - alpha) * q  # simplified: 1 period since last

    return z / q if q > 0 else z


def sba_method(series: pd.Series, alpha: float = 0.1) -> float:
    """
    Syntetos-Boylan Approximation (SBA): Croston x (1 - alpha/2).
    Reduces bias in Croston's method.
    """
    return croston_method(series, alpha) * (1 - alpha / 2)


def tsb_method(series: pd.Series,
               alpha_demand: float = 0.1,
               alpha_prob: float = 0.1) -> float:
    """
    Teunter-Syntetos-Boylan (TSB) method.
    Updates probability of non-zero demand separately.
    """
    series = series.dropna().values
    if len(series) == 0:
        return 0.0

    nz = series[series > 0]
    if len(nz) == 0:
        return 0.0

    # Initialise
    p = len(nz) / len(series)  # probability of demand
    mu = float(np.mean(nz))    # mean demand size

    for v in series:
        if v > 0:
            p = alpha_prob * 1 + (1 - alpha_prob) * p
            mu = alpha_demand * v + (1 - alpha_demand) * mu
        else:
            p = alpha_prob * 0 + (1 - alpha_prob) * p
            # mu unchanged

    return p * mu


def apply_intermittent_forecasts(train: pd.DataFrame,
                                  test_keys: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Croston/SBA/TSB to SKU-Region-Customer series classified as
    intermittent or lumpy, return a DataFrame with forecasts.
    Uses SBA as default for intermittent, TSB for lumpy.
    """
    df = train.copy()
    df['Units_Sold_num'] = pd.to_numeric(df['Units_Sold'], errors='coerce').fillna(0)

    # Get intermittency classification if available
    if 'intermittency_class' not in df.columns:
        df['intermittency_class'] = 'smooth'

    forecasts = {}
    for key, grp in df.groupby(['SKU_ID', 'Region', 'Customer_Name']):
        series = grp.sort_values('Date')['Units_Sold_num']
        cls = grp['intermittency_class'].mode().iloc[0] if 'intermittency_class' in grp.columns else 'smooth'
        if cls == 'lumpy':
            f = tsb_method(series)
        elif cls == 'intermittent':
            f = sba_method(series)
        else:
            f = series.mean()  # smooth/erratic: simple mean
        forecasts[key] = max(0, f)

    test = test_keys.copy()
    test['_key'] = list(zip(test['SKU_ID'], test['Region'], test['Customer_Name']))
    test['Units_Sold_Forecast'] = test['_key'].map(forecasts).fillna(0)
    test['Units_Sold_Forecast'] = test['Units_Sold_Forecast'].clip(lower=0).round().astype(int)
    return test[['Date', 'SKU_ID', 'Region', 'Units_Sold_Forecast']]

