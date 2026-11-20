"""
ordering_cycles.py - Expert analysis of customer ordering cycles, switches, and lost accounts (Rule R6).
"""

import pandas as pd
import numpy as np


def analyze_customer_cycles(clean_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyze ordering cycles, customer switches, and lost accounts per Rule R6.
    Returns:
    - cycles_df: summary of order cycle and change points per SKU-region-customer
    - switches_df: customer replacement events on the same SKU-region
    - lost_accounts_df: accounts that ceased ordering for >= 3 consecutive cycles
    """
    df = clean_train.copy()
    df['Units_num'] = pd.to_numeric(df['Units_Sold'], errors='coerce').fillna(0)
    orders = df[df['Units_num'] > 0].copy()

    # Convert YYYY-MM to integer month index
    def to_month_idx(dt_str):
        parts = dt_str.split('-')
        return int(parts[0]) * 12 + int(parts[1])

    orders['month_idx'] = orders['Date'].apply(to_month_idx)
    orders = orders.sort_values(['SKU_ID', 'Region', 'Customer_Name', 'month_idx'])

    max_train_month_idx = to_month_idx('2026-10')

    cycle_records = []
    lost_records = []

    # 1. Ordering cycles and lost accounts
    for (sku, reg, cust), grp in orders.groupby(['SKU_ID', 'Region', 'Customer_Name']):
        months = grp['month_idx'].values
        dates = grp['Date'].values
        n_orders = len(months)

        if n_orders >= 2:
            intervals = np.diff(months)
            med_interval = float(np.median(intervals))
            if med_interval <= 1.3:
                cycle_type = 'Monthly'
                cycle_code = 1
                expected_gap = 1
            elif med_interval <= 2.3:
                cycle_type = 'Bi-monthly'
                cycle_code = 2
                expected_gap = 2
            elif med_interval <= 3.5:
                cycle_type = 'Quarterly'
                cycle_code = 3
                expected_gap = 3
            else:
                cycle_type = 'Irregular'
                cycle_code = 4
                expected_gap = int(round(med_interval))

            # Detect change points in ordering interval
            change_dates = []
            if len(intervals) >= 4:
                half = len(intervals) // 2
                first_half_med = np.median(intervals[:half])
                second_half_med = np.median(intervals[half:])
                if abs(second_half_med - first_half_med) >= 1.0:
                    change_idx = half
                    change_dates.append(dates[change_idx])
        else:
            cycle_type = 'Sporadic'
            cycle_code = 5
            expected_gap = 6
            change_dates = []

        last_order_month_idx = months[-1]
        last_order_date = dates[-1]
        months_inactive = max_train_month_idx - last_order_month_idx

        # Lost account: ceased ordering for >= 3 consecutive expected cycles and not in last months
        is_lost = False
        if months_inactive >= (3 * expected_gap) and months_inactive >= 6:
            is_lost = True
            lost_records.append({
                'SKU_ID': sku,
                'Region': reg,
                'Customer_Name': cust,
                'Order_Cycle': cycle_type,
                'Expected_Gap_Months': expected_gap,
                'Last_Order_Date': last_order_date,
                'Months_Inactive': months_inactive,
                'Status': 'Lost Account',
            })

        cycle_records.append({
            'SKU_ID': sku,
            'Region': reg,
            'Customer_Name': cust,
            'Order_Count': n_orders,
            'Cycle_Type': cycle_type,
            'Cycle_Code': cycle_code,
            'Expected_Gap_Months': expected_gap,
            'Change_Points': '; '.join(change_dates) if change_dates else 'None',
            'Last_Order_Date': last_order_date,
            'Is_Lost': is_lost,
        })

    cycles_df = pd.DataFrame(cycle_records)
    lost_accounts_df = pd.DataFrame(lost_records)

    # 2. Customer switches on same SKU-Region
    switch_records = []
    for (sku, reg), grp in orders.groupby(['SKU_ID', 'Region']):
        cust_spans = grp.groupby('Customer_Name').agg(
            first_m=('month_idx', 'min'),
            last_m=('month_idx', 'max'),
            first_date=('Date', 'min'),
            last_date=('Date', 'max')
        ).reset_index()

        if len(cust_spans) >= 2:
            cust_spans = cust_spans.sort_values('first_m').reset_index(drop=True)
            for i in range(len(cust_spans) - 1):
                c_prev = cust_spans.iloc[i]
                c_next = cust_spans.iloc[i + 1]
                # If prev customer stopped and next started within 1 to 4 months
                gap = c_next['first_m'] - c_prev['last_m']
                if 0 <= gap <= 4 and c_prev['Customer_Name'] != c_next['Customer_Name']:
                    switch_records.append({
                        'SKU_ID': sku,
                        'Region': reg,
                        'Prior_Customer': c_prev['Customer_Name'],
                        'Prior_Last_Date': c_prev['last_date'],
                        'New_Customer': c_next['Customer_Name'],
                        'New_Start_Date': c_next['first_date'],
                        'Transition_Gap_Months': gap,
                    })

    switches_df = pd.DataFrame(switch_records)
    return cycles_df, switches_df, lost_accounts_df


# Alias for backward compatibility
analyze_ordering_cycles = analyze_customer_cycles


def add_cycle_features(features_df: pd.DataFrame, cycles_df: pd.DataFrame = None) -> pd.DataFrame:
    """Add customer ordering cycle features to the feature set."""
    df = features_df.copy()
    if cycles_df is None:
        cycles_df, _, _ = analyze_customer_cycles(df)

    if cycles_df.empty:
        df['order_cycle_code'] = 1
        df['is_lost_account'] = 0
        return df

    lookup = cycles_df.set_index(['SKU_ID', 'Region', 'Customer_Name'])
    cycle_codes = []
    lost_flags = []

    for _, row in df.iterrows():
        key = (row['SKU_ID'], row['Region'], row['Customer_Name'])
        if key in lookup.index:
            info = lookup.loc[key]
            if isinstance(info, pd.DataFrame):
                info = info.iloc[0]
            cycle_codes.append(int(info['Cycle_Code']))
            lost_flags.append(1 if info['Is_Lost'] else 0)
        else:
            cycle_codes.append(1)
            lost_flags.append(0)

    df['order_cycle_code'] = cycle_codes
    df['is_lost_account'] = lost_flags
    return df
