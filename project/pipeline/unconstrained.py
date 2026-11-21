"""
unconstrained.py - Build unconstrained demand series from inventory_stockouts and identify suspected stockouts (Rule R7, A2).
"""

import numpy as np
import pandas as pd
from pipeline.config import OUT_UNCONSTRAINED, OUT_STOCKOUTS
from pipeline.audit import audit


def allocate_stockout_metric(df: pd.DataFrame, stockouts: pd.DataFrame,
                             metric_col: str,
                             target_demand_col: str = 'Units_Sold') -> pd.DataFrame:
    """
    Allocate Date-SKU-Region stockout metric (Shipped_Units or Requested_Units)
    across customers matching that key in df (Rule A2).
    - If 1 customer: assigns value directly.
    - If > 1 customers: splits in proportion to each customer's demand.
      If all are zero, splits equally.
    - Allocated integer values sum to the stockout total exactly, with rounding
      remainder placed on the customer with the largest demand.
    Returns DataFrame matching df.index with columns:
      allocated_<metric>, share_<metric>, total_<metric>, n_cust_matched.
    """
    so = stockouts.drop_duplicates(['Date', 'SKU_ID', 'Region'])[['Date', 'SKU_ID', 'Region', metric_col]].copy()
    so[metric_col] = pd.to_numeric(so[metric_col], errors='coerce').fillna(0).round().astype(int)

    sub = df[['Date', 'SKU_ID', 'Region', target_demand_col]].copy()
    sub['orig_idx'] = sub.index
    sub['_demand'] = pd.to_numeric(sub[target_demand_col], errors='coerce').fillna(0).clip(lower=0)

    merged = sub.merge(so, on=['Date', 'SKU_ID', 'Region'], how='inner')
    if merged.empty:
        res = pd.DataFrame(index=df.index)
        res[f'allocated_{metric_col}'] = np.nan
        res[f'share_{metric_col}'] = np.nan
        res[f'total_{metric_col}'] = np.nan
        res['n_cust_matched'] = 0
        return res

    allocated_vals = {}
    share_vals = {}
    total_vals = {}
    n_cust_map = {}

    for (d, s, r), grp in merged.groupby(['Date', 'SKU_ID', 'Region'], sort=False):
        tot = int(grp[metric_col].iloc[0])
        n = len(grp)
        demands = grp['_demand'].values
        sum_d = demands.sum()

        if n == 1:
            idx = grp['orig_idx'].iloc[0]
            allocated_vals[idx] = tot
            share_vals[idx] = 1.0
            total_vals[idx] = tot
            n_cust_map[idx] = 1
            continue

        if sum_d > 0:
            weights = demands / sum_d
        else:
            weights = np.full(n, 1.0 / n)

        raw_alloc = tot * weights
        floored = np.floor(raw_alloc).astype(int)
        remainder = tot - int(floored.sum())

        fractions = raw_alloc - floored
        order = np.lexsort((-fractions, -demands))
        for k in range(remainder):
            floored[order[k % n]] += 1

        for orig_idx, a_val, w_val in zip(grp['orig_idx'], floored, weights):
            allocated_vals[orig_idx] = int(a_val)
            share_vals[orig_idx] = float(w_val)
            total_vals[orig_idx] = tot
            n_cust_map[orig_idx] = n

    res = pd.DataFrame(index=df.index)
    res[f'allocated_{metric_col}'] = pd.Series(allocated_vals, dtype='Float64')
    res[f'share_{metric_col}'] = pd.Series(share_vals, dtype='Float64')
    res[f'total_{metric_col}'] = pd.Series(total_vals, dtype='Float64')
    res['n_cust_matched'] = pd.Series(n_cust_map, dtype='Int64').fillna(0)
    return res


def build_unconstrained_demand(clean_train: pd.DataFrame,
                                stockouts: pd.DataFrame,
                                events: pd.DataFrame = None) -> tuple[pd.DataFrame, float]:
    """
    WS2 & A2: Merge clean_train with stockouts to build unconstrained demand.
    - Listed stockouts: use proportionally allocated Requested_Units as true demand (R7, A2).
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
    # Cross-validation: train Units_Sold vs stockout Shipped_Units
    merge_cv = df.merge(
        so[so_key + ['Shipped_Units_num', 'Requested_Units_num']],
        on=so_key, how='inner'
    )
    match_rate = 100.0
    if len(merge_cv):
        tolerance = 1  # unit
        matches = (merge_cv['Units_Sold_num'] - merge_cv['Shipped_Units_num']).abs() <= tolerance
        match_rate = float(matches.mean() * 100)
        print(f"Stockout cross-validation: {match_rate:.1f}% match rate ({matches.sum()}/{len(merge_cv)} rows)")

    result = df.copy()
    result['Units_Sold_Unconstrained'] = result['Units_Sold_num']
    result['stockout_status'] = 'none'

    # 1. Listed stockouts correction using proportional allocation (Rule A2)
    alloc = allocate_stockout_metric(result, stockouts, 'Requested_Units', target_demand_col='Units_Sold_num')
    has_req = alloc['allocated_Requested_Units'].notna()

    for idx in result.index[has_req]:
        req_val = alloc.at[idx, 'allocated_Requested_Units']
        share = alloc.at[idx, 'share_Requested_Units']
        tot = alloc.at[idx, 'total_Requested_Units']
        n_cust = alloc.at[idx, 'n_cust_matched']
        rid = result.at[idx, 'Record_ID']
        raw_us = result.at[idx, 'Units_Sold_num']

        result.at[idx, 'Units_Sold_Unconstrained'] = req_val
        result.at[idx, 'stockout_status'] = 'listed_stockout'

        if n_cust > 1:
            audit.log("train.csv", rid, "Units_Sold_Unconstrained",
                      raw_us, req_val, "A2",
                      f"Requested_Units allocated to customer: share={share:.4f} of total {int(tot)}")

    listed_corrected = int(has_req.sum())
    print(f"Listed stockouts corrected: {listed_corrected} rows using allocated Requested_Units")

    # 2. Suspected unlisted stockouts (A2)
    # Identify known event periods that explain drops (COVID, strikes, ERP migration)
    event_months = set()
    if events is not None and not events.empty:
        for _, ev in events.iterrows():
            st_d = str(ev.get('Start_Date', ''))[:7]
            en_d = str(ev.get('End_Date', ''))[:7] if pd.notna(ev.get('End_Date')) and str(ev.get('End_Date')).strip() else st_d
            if st_d:
                ev_dates = pd.date_range(f"{st_d}-01", f"{en_d}-01", freq='MS').strftime('%Y-%m')
                event_months.update(ev_dates)

    # Detect 2+ consecutive months where demand is < 50% of normal median without an event
    suspected_count = 0
    result = result.sort_values(['SKU_ID', 'Region', 'Customer_Name', 'Date']).reset_index(drop=True)

    for (sku, reg, cust), grp in result.groupby(['SKU_ID', 'Region', 'Customer_Name']):
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
