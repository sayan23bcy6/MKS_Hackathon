"""
unconstrained.py - Build unconstrained demand series from inventory_stockouts and identify suspected stockouts (Rule R7, A2).
"""

import pandas as pd
import numpy as np
from pipeline.config import OUT_UNCONSTRAINED, OUT_STOCKOUTS


def build_unconstrained_demand(clean_train: pd.DataFrame,
                                stockouts: pd.DataFrame,
                                events: pd.DataFrame = None) -> tuple[pd.DataFrame, float]:
    """
    WS2 & A2: Merge clean_train with stockouts to build unconstrained demand.
    - Listed stockouts: use Requested_Units as true demand (R7).
    - Cross-validate: train Units_Sold vs stockout Shipped_Units.
    - Suspected unlisted stockouts: consecutive months (< 50% median) without supporting events.
      Estimate unconstrained demand from normal median and label as 'suspected stockout'.
    """
    df = clean_train.copy()
    df['Units_Sold_num'] = pd.to_numeric(df['Units_Sold'], errors='coerce')

    so = stockouts.copy()
    so['Requested_Units_num'] = pd.to_numeric(so['Requested_Units'], errors='coerce')
    so['Shipped_Units_num'] = pd.to_numeric(so['Shipped_Units'], errors='coerce')

    so_key = ['Date', 'SKU_ID', 'Region']
    so_lookup = so.groupby(so_key).agg({
        'Shipped_Units_num': 'max',
        'Requested_Units_num': 'max'
    }).to_dict(orient='index')

    # Cross-validation: train Units_Sold vs stockout Shipped_Units
    merge = df.merge(
        so[so_key + ['Shipped_Units_num', 'Requested_Units_num']],
        on=so_key, how='inner'
    )
    match_rate = 100.0
    if len(merge):
        tolerance = 1  # unit
        matches = (merge['Units_Sold_num'] - merge['Shipped_Units_num']).abs() <= tolerance
        match_rate = float(matches.mean() * 100)
        print(f"Stockout cross-validation: {match_rate:.1f}% match rate ({matches.sum()}/{len(merge)} rows)")

    result = df.copy()
    result['Units_Sold_Unconstrained'] = result['Units_Sold_num']
    result['stockout_status'] = 'none'

    # 1. Listed stockouts correction
    listed_corrected = 0
    for idx, row in result.iterrows():
        key = (row['Date'], row['SKU_ID'], row['Region'])
        if key in so_lookup:
            req = so_lookup[key]['Requested_Units_num']
            if pd.notna(req):
                result.at[idx, 'Units_Sold_Unconstrained'] = req
                result.at[idx, 'stockout_status'] = 'listed_stockout'
                listed_corrected += 1

    print(f"Listed stockouts corrected: {listed_corrected} rows using Requested_Units")

    # 2. Suspected unlisted stockouts (A2)
    # Identify known event periods that explain drops (COVID, strikes, ERP migration)
    event_months = set()
    if events is not None and not events.empty:
        for _, ev in events.iterrows():
            st_d = str(ev.get('Start_Date', ''))[:7]
            en_d = str(ev.get('End_Date', ''))[:7] if pd.notna(ev.get('End_Date')) else st_d
            if st_d:
                ev_dates = pd.date_range(f"{st_d}-01", f"{en_d}-01", freq='MS').strftime('%Y-%m')
                event_months.update(ev_dates)

    # Detect 2+ consecutive months where demand is < 50% of normal median without an event
    suspected_count = 0
    result = result.sort_values(['SKU_ID', 'Region', 'Date']).reset_index(drop=True)

    for (sku, reg), grp in result.groupby(['SKU_ID', 'Region']):
        active_vals = grp.loc[grp['Units_Sold_num'] > 0, 'Units_Sold_num']
        if len(active_vals) < 12:
            continue
        normal_median = float(active_vals.median())
        if normal_median < 10:
            continue

        drop_threshold = 0.5 * normal_median
        is_low = (grp['Units_Sold_num'] < drop_threshold) & (grp['stockout_status'] == 'none') & (~grp['Date'].isin(event_months))

        # Check for consecutive low months (run length >= 2)
        low_runs = is_low & (is_low.shift(1, fill_value=False) | is_low.shift(-1, fill_value=False))
        if low_runs.any():
            low_indices = grp.index[low_runs]
            for l_idx in low_indices:
                result.at[l_idx, 'stockout_status'] = 'suspected_stockout'
                result.at[l_idx, 'Units_Sold_Unconstrained'] = normal_median
                suspected_count += 1

    print(f"Suspected unlisted stockouts detected: {suspected_count} rows adjusted to normal median")

    result['is_stockout_corrected'] = result['stockout_status'] != 'none'

    # Save to outputs
    result.to_parquet(OUT_UNCONSTRAINED, index=False)
    so.to_csv(OUT_STOCKOUTS, index=False)
    return result, match_rate
