"""
unconstrained.py – Build the unconstrained demand series from inventory_stockouts.
For stockout months, Requested_Units is the true demand (R7).
"""

import pandas as pd
import numpy as np
from pipeline.config import OUT_UNCONSTRAINED


def build_unconstrained_demand(clean_train: pd.DataFrame,
                                stockouts: pd.DataFrame) -> pd.DataFrame:
    """
    WS2 Expert: Merge clean_train with stockouts to build unconstrained demand.

    Rules:
    - For rows in inventory_stockouts, use Requested_Units as Units_Sold_Unconstrained
    - Verify: train Units_Sold should equal stockout Shipped_Units (report match rate)
    - For non-stockout rows, Units_Sold_Unconstrained = Units_Sold
    """
    df = clean_train.copy()
    df['Units_Sold_num'] = pd.to_numeric(df['Units_Sold'], errors='coerce')

    so = stockouts.copy()
    so['Requested_Units_num'] = pd.to_numeric(so['Requested_Units'], errors='coerce')
    so['Shipped_Units_num'] = pd.to_numeric(so['Shipped_Units'], errors='coerce')

    # Build stockout lookup
    so_key = ['Date', 'SKU_ID', 'Region']
    so_lookup = so.set_index(so_key)

    # Cross-validate: train Units_Sold vs stockout Shipped_Units
    merge = df.merge(
        so[so_key + ['Shipped_Units_num', 'Requested_Units_num']],
        on=so_key, how='inner'
    )
    match_rate = None
    if len(merge):
        tolerance = 1  # unit
        matches = (merge['Units_Sold_num'] - merge['Shipped_Units_num']).abs() <= tolerance
        match_rate = float(matches.mean() * 100)
        mismatches = merge[~matches]
        print(f"Stockout cross-validation: {match_rate:.1f}% match rate ({matches.sum()}/{len(merge)} rows)")
        if len(mismatches):
            print(f"  Mismatches: {len(mismatches)} rows")

    # Build unconstrained series
    result = df.copy()
    result['Units_Sold_Unconstrained'] = result['Units_Sold_num']
    result['is_stockout_corrected'] = False

    so_index = set(zip(so['Date'], so['SKU_ID'], so['Region']))
    for idx, row in result.iterrows():
        key = (row['Date'], row['SKU_ID'], row['Region'])
        if key in so_index:
            so_row = so_lookup.loc[key] if key in so_lookup.index else None
            if so_row is not None and not isinstance(so_row, pd.DataFrame):
                req = so_row['Requested_Units_num']
                result.at[idx, 'Units_Sold_Unconstrained'] = req
                result.at[idx, 'is_stockout_corrected'] = True
            elif isinstance(so_row, pd.DataFrame):
                # Multiple rows – take max Requested_Units
                req = so_row['Requested_Units_num'].max()
                result.at[idx, 'Units_Sold_Unconstrained'] = req
                result.at[idx, 'is_stockout_corrected'] = True

    stockout_corrected_count = result['is_stockout_corrected'].sum()
    print(f"Unconstrained demand: {stockout_corrected_count} rows corrected from stockout Requested_Units")

    # Save
    result.to_parquet(OUT_UNCONSTRAINED, index=False)
    return result, match_rate

