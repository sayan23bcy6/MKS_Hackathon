"""
features.py - Feature engineering for demand forecasting.
Generates calendar, event, lifecycle, macro, lag, rolling, intermittency,
and customer ordering cycle features.
"""

import pandas as pd
import numpy as np
from pipeline.config import (
    OUT_FEATURES,
    PREFIX_TO_FAMILY,
)
from pipeline.ordering_cycles import add_cycle_features

MONTH_TO_QUARTER = {
    1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2,
    7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4
}

# Lunar New Year months (Asia Pacific and China-dependent markets)
LNY_MONTHS = {
    '2019': 2, '2020': 1, '2021': 2, '2022': 2,
    '2023': 1, '2024': 2, '2025': 1, '2026': 2,
    '2027': 2, '2028': 1,
}

# Ramadan months (Middle East & Africa and India)
RAMADAN_MONTHS = {
    '2019': 5, '2020': 4, '2021': 4, '2022': 4,
    '2023': 3, '2024': 3, '2025': 3, '2026': 3,
    '2027': 2, '2028': 2,
}

MACRO_COLS = [
    'ISM_Manufacturing_PMI', 'Auto_Production_YoY_Pct',
    'Vehicle_Inventory_Days', 'Raw_Material_Cost_Index',
    'New_Vehicle_Orders_Index', 'Industrial_Util_Rate_Pct',
    'Fuel_Price_Index', 'EV_Adoption_Index'
]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar cycle, seasonal, and regional holiday flags."""
    df = df.copy()
    date_dt = pd.to_datetime(df['Date'], format='%Y-%m')
    df['year'] = date_dt.dt.year
    df['month'] = date_dt.dt.month
    df['quarter'] = df['month'].map(MONTH_TO_QUARTER)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    yr_mo = list(zip(df['year'].astype(str), df['month']))
    is_lny_month = pd.Series([LNY_MONTHS.get(y) == m for y, m in yr_mo], index=df.index)
    df['flag_lny'] = (df['Region'].isin(['Asia Pacific', 'India']) & is_lny_month).astype(int)

    is_ram_month = pd.Series([RAMADAN_MONTHS.get(y) == m for y, m in yr_mo], index=df.index)
    df['flag_ramadan'] = (df['Region'].isin(['Middle East & Africa', 'India']) & is_ram_month).astype(int)

    df['flag_summer_shutdown'] = (
        (df['Region'] == 'Europe') & (df['month'].isin([7, 8]))
    ).astype(int)

    df['flag_year_end'] = (df['month'] == 12).astype(int)
    df['flag_q1_start'] = (df['month'] == 1).astype(int)
    return df


def add_event_features(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Add binary event flags and interactions from events calendar."""
    df = df.copy()
    df['flag_price_change'] = 0
    df['flag_supply_constraint'] = 0
    df['flag_disruption'] = 0
    df['flag_system_event'] = 0
    df['flag_early_buy_window'] = 0
    df['flag_orion_retool'] = 0

    if events is None or len(events) == 0:
        return df

    for _, ev in events.iterrows():
        ev_start = str(ev.get('Start_Date', ''))[:7]
        ev_end = str(ev.get('End_Date', ''))[:7] if pd.notna(ev.get('End_Date')) and str(ev.get('End_Date')).strip() else ev_start
        ev_type = str(ev.get('Event_Type', '')).strip()
        scope_type = str(ev.get('Scope_Type', '')).strip()
        scope_val = str(ev.get('Scope_Value', '')).strip()
        ev_id = str(ev.get('Event_ID', '')).strip()

        month_mask = (df['Date'] >= ev_start) & (df['Date'] <= ev_end)

        if scope_type == 'Region':
            scope_region = ev.get('Scope_Region') or scope_val
            if scope_region:
                month_mask = month_mask & (df['Region'] == scope_region)
        elif scope_type == 'Product_Family':
            month_mask = month_mask & (df['Product_Family'] == scope_val)
        elif scope_type == 'Channel':
            month_mask = month_mask & (df['Sales_Channel'] == scope_val)
        elif scope_type == 'Customer':
            cust = ev.get('Scope_Customer') or scope_val
            if '|' in str(cust):
                parts = str(cust).split('|')
                cust_name = parts[0].strip()
                reg_name = parts[1].strip() if len(parts) > 1 else None
                month_mask = month_mask & (df['Customer_Name'] == cust_name)
                if reg_name:
                    month_mask = month_mask & (df['Region'] == reg_name)
            else:
                month_mask = month_mask & (df['Customer_Name'] == cust)

        if ev_type == 'Price Change':
            df.loc[month_mask, 'flag_price_change'] = 1
        elif ev_type == 'Supply Constraint':
            df.loc[month_mask, 'flag_supply_constraint'] = 1
        elif ev_type == 'Disruption':
            df.loc[month_mask, 'flag_disruption'] = 1
        elif ev_type == 'System':
            df.loc[month_mask, 'flag_system_event'] = 1
        elif ev_type == 'Announcement' and ev_id == 'EV-030':
            df.loc[month_mask, 'flag_early_buy_window'] = 1

        if ev_id == 'EV-062':
            df.loc[month_mask, 'flag_orion_retool'] = 1

    return df


def add_lifecycle_features(df: pd.DataFrame, sku_master: pd.DataFrame) -> pd.DataFrame:
    """Add lifecycle age, distance to EOL, and replacement relationships."""
    df = df.copy()
    launch_map = dict(zip(sku_master['SKU_ID'], sku_master['Launch_Date']))
    eol_map = dict(zip(sku_master['SKU_ID'], sku_master['EOL_Date']))
    replaces_map = dict(zip(sku_master['SKU_ID'], sku_master['Replaces_SKU']))

    date_dt = pd.to_datetime(df['Date'].str[:7], format='%Y-%m')
    launch_dt = pd.to_datetime(df['SKU_ID'].map(launch_map).str[:7], format='%Y-%m', errors='coerce')
    eol_dt = pd.to_datetime(df['SKU_ID'].map(eol_map).str[:7], format='%Y-%m', errors='coerce')

    df['months_since_launch'] = (date_dt.dt.year - launch_dt.dt.year) * 12 + (date_dt.dt.month - launch_dt.dt.month)
    df['months_to_eol'] = (eol_dt.dt.year - date_dt.dt.year) * 12 + (eol_dt.dt.month - date_dt.dt.month)

    df['is_replacement_sku'] = df['SKU_ID'].map(lambda s: 1 if replaces_map.get(s) else 0)
    df['has_replacement'] = df['SKU_ID'].map(
        lambda s: 1 if s in set(replaces_map.values()) else 0
    )
    return df


def add_macro_features(df: pd.DataFrame, macro: pd.DataFrame,
                       lags: list = [0, 1, 2, 3]) -> pd.DataFrame:
    """Add macro indicators at specified monthly lags."""
    df = df.copy()
    macro = macro.copy().set_index('Date')

    for col in MACRO_COLS:
        if col not in macro.columns:
            continue
        for lag in lags:
            new_col = f"macro_{col}_lag{lag}"
            shifted = macro[col].copy()
            shifted.index = pd.to_datetime(macro.index, format='%Y-%m') + pd.DateOffset(months=lag)
            shifted.index = shifted.index.strftime('%Y-%m')
            df[new_col] = df['Date'].map(shifted.to_dict())

    return df


def add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute historical lag (1, 2, 3, 6, 12m), rolling statistics (3, 6, 12m),
    and true YoY growth ratio per SKU-Region-Customer series.
    Fixes F4: lag_12m_yoy is the growth ratio, distinct from lag_12m.
    """
    df = df.copy()
    df = df.sort_values(['SKU_ID', 'Region', 'Customer_Name', 'Date']).reset_index(drop=True)

    units_col = 'Units_Sold_Unconstrained' if 'Units_Sold_Unconstrained' in df.columns else 'Units_Sold'

    grp = df.groupby(['SKU_ID', 'Region', 'Customer_Name'])[units_col]
    df['lag_1m'] = grp.shift(1)
    df['lag_2m'] = grp.shift(2)
    df['lag_3m'] = grp.shift(3)
    df['lag_6m'] = grp.shift(6)
    df['lag_12m'] = grp.shift(12)

    # True YoY growth ratio: (lag_1m - lag_12m) / (lag_12m + 1e-3)
    df['lag_12m_yoy'] = (df['lag_1m'] - df['lag_12m']) / (df['lag_12m'].abs() + 1e-3)

    grp_s1 = df.groupby(['SKU_ID', 'Region', 'Customer_Name'])['lag_1m']
    df['roll_mean_3m'] = grp_s1.rolling(3, min_periods=1).mean().values
    df['roll_mean_6m'] = grp_s1.rolling(6, min_periods=1).mean().values
    df['roll_mean_12m'] = grp_s1.rolling(12, min_periods=1).mean().values

    df['roll_std_3m'] = grp_s1.rolling(3, min_periods=2).std().values
    df['roll_std_6m'] = grp_s1.rolling(6, min_periods=2).std().values

    return df


def classify_intermittency(df: pd.DataFrame) -> pd.DataFrame:
    """
    ADI/CV^2 intermittency classification per SKU-Region-Customer.
    ADI > 1.32 -> intermittent; CV^2 > 0.49 -> lumpy.
    Classes: smooth (0), erratic (1), intermittent (2), lumpy (3).
    """
    df = df.copy()
    units_col = 'Units_Sold_Unconstrained' if 'Units_Sold_Unconstrained' in df.columns else 'Units_Sold'

    grp = df.groupby(['SKU_ID', 'Region', 'Customer_Name'])[units_col]
    total_periods = grp.count()

    nz_df = df[df[units_col] > 0]
    nz_cnt = nz_df.groupby(['SKU_ID', 'Region', 'Customer_Name'])[units_col].count().reindex(total_periods.index, fill_value=0)
    nz_mean = nz_df.groupby(['SKU_ID', 'Region', 'Customer_Name'])[units_col].mean().reindex(total_periods.index, fill_value=0)
    nz_std = nz_df.groupby(['SKU_ID', 'Region', 'Customer_Name'])[units_col].std().reindex(total_periods.index, fill_value=0)

    adi = (total_periods / nz_cnt.replace(0, np.nan)).fillna(999.0)
    cv2 = ((nz_std / nz_mean.replace(0, np.nan)) ** 2).fillna(0.0)

    classes = pd.Series('lumpy', index=total_periods.index)
    classes[(adi <= 1.32) & (cv2 <= 0.49)] = 'smooth'
    classes[(adi <= 1.32) & (cv2 > 0.49)] = 'erratic'
    classes[(adi > 1.32) & (cv2 <= 0.49)] = 'intermittent'

    class_dict = classes.to_dict()
    df['intermittency_class'] = [
        class_dict.get((s, r, c), 'lumpy')
        for s, r, c in zip(df['SKU_ID'], df['Region'], df['Customer_Name'])
    ]

    class_encoding = {'smooth': 0, 'erratic': 1, 'intermittent': 2, 'lumpy': 3}
    df['intermittency_code'] = df['intermittency_class'].map(class_encoding)
    return df


def add_sku_region_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical identifiers into stable integer codes."""
    for col in ['SKU_ID', 'Region', 'Customer_Name', 'Product_Family',
                'Sales_Channel', 'Business_Unit']:
        if col in df.columns:
            enc_col = f"{col}_enc"
            df[enc_col] = df[col].astype('category').cat.codes
    return df


def build_features(clean_train: pd.DataFrame,
                   clean_macro: pd.DataFrame,
                   clean_events: pd.DataFrame,
                   sku_master: pd.DataFrame,
                   unconstrained: pd.DataFrame = None) -> pd.DataFrame:
    """
    Main feature engineering pipeline.
    Combines calendar, events, lifecycle, macro, lags, intermittency, and ordering cycles.
    """
    df = clean_train.copy()

    if 'Units_Sold_Unconstrained' not in df.columns:
        if unconstrained is not None and len(unconstrained) and 'Units_Sold_Unconstrained' in unconstrained.columns:
            merge_cols = [c for c in ['Date', 'SKU_ID', 'Region', 'Customer_Name'] if c in df.columns and c in unconstrained.columns]
            uc_dedup = unconstrained[merge_cols + ['Units_Sold_Unconstrained']].drop_duplicates(merge_cols)
            df = df.merge(uc_dedup, on=merge_cols, how='left', validate="one_to_one")
            df['Units_Sold_Unconstrained'] = df['Units_Sold_Unconstrained'].fillna(df.get('Units_Sold', np.nan))
        else:
            df['Units_Sold_Unconstrained'] = df.get('Units_Sold', np.nan)

    df['Units_Sold'] = pd.to_numeric(df['Units_Sold'], errors='coerce')
    df['Units_Sold_Unconstrained'] = pd.to_numeric(df['Units_Sold_Unconstrained'], errors='coerce')

    df = add_calendar_features(df)
    df = add_event_features(df, clean_events)
    df = add_lifecycle_features(df, sku_master)
    df = add_macro_features(df, clean_macro, lags=[0, 1, 2, 3])
    df = add_lag_rolling_features(df)
    df = classify_intermittency(df)
    df = add_cycle_features(df)
    df = add_sku_region_encodings(df)

    # Assert zero duplicate Date-SKU-Region-Customer rows
    dup_count = int(df.duplicated(['Date', 'SKU_ID', 'Region', 'Customer_Name']).sum())
    if dup_count > 0:
        raise ValueError(f"build_features: {dup_count} duplicate series-month rows found on Date-SKU-Region-Customer")

    df.to_parquet(OUT_FEATURES, index=False)
    return df


def compute_macro_lag_analysis(clean_train: pd.DataFrame,
                               clean_macro: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-correlation of each macro indicator against each product family's units at lags 0-6.
    """
    df = clean_train.copy()
    df['Units_Sold_num'] = pd.to_numeric(df['Units_Sold'], errors='coerce')
    monthly = df.groupby(['Date', 'Product_Family'])['Units_Sold_num'].sum().reset_index()

    macro = clean_macro.copy().set_index('Date')

    results = []
    for family in monthly['Product_Family'].unique():
        fam_series = monthly[monthly['Product_Family'] == family].set_index('Date')['Units_Sold_num']
        for col in MACRO_COLS:
            if col not in macro.columns:
                continue
            for lag in range(0, 7):
                mac_shifted = macro[col].copy()
                common = fam_series.index.intersection(mac_shifted.index)
                if len(common) < 6:
                    continue
                fam_v = fam_series.loc[common]
                mac_v = mac_shifted.loc[common]
                mac_lagged = mac_v.shift(lag)
                aligned = pd.concat([fam_v, mac_lagged], axis=1).dropna()
                if len(aligned) < 6:
                    continue
                corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                results.append({
                    'Product_Family': family,
                    'MacroIndicator': col,
                    'Lag_Months': lag,
                    'Correlation': round(corr, 4),
                })

    return pd.DataFrame(results)
