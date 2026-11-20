"""
anomalies.py – Anomaly register: flags outliers, structural breaks, real events vs errors.
"""

import pandas as pd
import numpy as np
from pipeline.config import (
    UNITS_LARGE_THRESHOLD,
    OUT_ANOMALY_REG,
    PRICE_SCALE_MED_LOW, PRICE_SCALE_MED_HIGH,
)

ANOMALY_COLS = [
    "Date", "SKU_ID", "Region", "Customer_Name",
    "anomaly_type", "field", "value", "baseline",
    "action", "reason", "event_reference"
]


class AnomalyRegister:
    def __init__(self):
        self._records = []

    def add(self, date, sku_id, region, customer,
            anomaly_type, field, value, baseline,
            action, reason, event_ref=""):
        self._records.append({
            "Date": date, "SKU_ID": sku_id, "Region": region,
            "Customer_Name": customer,
            "anomaly_type": anomaly_type, "field": field,
            "value": value, "baseline": baseline,
            "action": action, "reason": reason,
            "event_reference": event_ref,
        })

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records, columns=ANOMALY_COLS)

    def save(self):
        df = self.as_dataframe()
        df.to_csv(OUT_ANOMALY_REG, index=False)
        return len(df)


anomaly_reg = AnomalyRegister()


def build_anomaly_register(clean_train: pd.DataFrame,
                           events: pd.DataFrame,
                           stockouts: pd.DataFrame,
                           audit_log: pd.DataFrame) -> pd.DataFrame:
    """
    WS2: Build comprehensive anomaly register.
    - Large unit values (T11)
    - Price ratio 5–20× (T12 unresolved band)
    - Early buying before EV-030 price rise
    - COVID and disruption events
    - Stockout months
    - Zero runs (potential hidden stockouts)
    """
    reg = AnomalyRegister()
    df = clean_train.copy()
    df['Units_Sold_num'] = pd.to_numeric(df['Units_Sold'], errors='coerce')
    df['Price_num'] = pd.to_numeric(df['Unit_Price_USD'], errors='coerce')

    # 1. Large unit values (>5000)
    large = df[df['Units_Sold_num'] > UNITS_LARGE_THRESHOLD]
    for _, row in large.iterrows():
        # Compare to SKU-region median
        sku_med = df[df['SKU_ID'] == row['SKU_ID']]['Units_Sold_num'].median()
        reg.add(row['Date'], row['SKU_ID'], row['Region'], row.get('Customer_Name', ''),
                "large_units", "Units_Sold", row['Units_Sold_num'], sku_med,
                "flagged_for_review",
                f"Units_Sold={row['Units_Sold_num']} exceeds {UNITS_LARGE_THRESHOLD} threshold (SKU median={sku_med:.0f})",
                "T11")

    # 2. Negative units (already abs'd – flag the original event)
    neg_audit = audit_log[
        (audit_log['rule_id'] == 'T11') &
        (audit_log['reason'].str.contains('Negative', na=False))
    ] if audit_log is not None and len(audit_log) else pd.DataFrame()
    for _, row in neg_audit.iterrows():
        reg.add('', '', '', '',
                "sign_error", "Units_Sold", row['raw_value'], 0,
                "abs_applied",
                "Negative Units_Sold: sign-entry error; absolute value used",
                "T11")

    # 3. Price 5–20× band (unresolved)
    price_audit = audit_log[
        (audit_log['rule_id'] == 'T12') &
        (audit_log['reason'].str.contains('5-20', na=False))
    ] if audit_log is not None and len(audit_log) else pd.DataFrame()
    for _, row in price_audit.iterrows():
        reg.add('', '', '', '',
                "price_ratio_5_20x", "Unit_Price_USD", row['raw_value'], '',
                "excluded_from_price_analysis",
                "Price 5–20× SKU median – unresolved, kept for demand analysis",
                "T12_unresolved")

    # 4. Early buying before EV-030 (2023 price rise announced Nov 2022)
    early_buy_months = ['2022-11', '2022-12']
    early_buy = df[df['Date'].isin(early_buy_months) & (df['Units_Sold_num'] > 0)]
    for sku, grp in early_buy.groupby('SKU_ID'):
        # Compare to same months in prior years
        baseline_months = []
        for yr in [2019, 2020, 2021]:
            for mo in ['11', '12']:
                baseline_months.append(f"{yr}-{mo}")
        baseline = df[
            df['Date'].isin(baseline_months) & (df['SKU_ID'] == sku)
        ]['Units_Sold_num'].mean()
        actual = grp['Units_Sold_num'].mean()
        if pd.notna(baseline) and pd.notna(actual) and baseline > 0 and actual > 1.2 * baseline:
            reg.add(', '.join(early_buy_months), sku, 'ALL', 'ALL',
                    "early_buying", "Units_Sold",
                    round(float(actual), 1), round(float(baseline), 1),
                    "modelled_as_event_feature",
                    f"Nov-Dec 2022 demand {actual:.0f} vs historical {baseline:.0f} – possible early buying before 2023 price rise (EV-030)",
                    "EV-030")

    # 5. COVID disruption (EV-009 to EV-013: 2020-02 to 2020-06)
    covid_months = ['2020-02', '2020-03', '2020-04', '2020-05', '2020-06']
    covid = df[df['Date'].isin(covid_months) & (df['Units_Sold_num'] > 0)]
    for (d, reg_name), grp in covid.groupby(['Date', 'Region']):
        reg.add(d, 'ALL_SKUS', reg_name, 'ALL_CUSTOMERS',
                "covid_disruption", "Units_Sold", int(grp['Units_Sold_num'].sum()), '',
                "real_event_modelled",
                f"COVID-19 disruption period ({d}): total {int(grp['Units_Sold_num'].sum())} units across {len(grp)} transactions; real constrained event, not error",
                "EV-009,EV-010,EV-011,EV-012,EV-013")

    # 6. Stockout months
    so = stockouts.copy()
    so['fill_rate'] = pd.to_numeric(so['Fill_Rate_Pct'], errors='coerce')
    so['lost'] = pd.to_numeric(so['Lost_Units'], errors='coerce')
    for _, row in so[so['fill_rate'] < 100].head(500).iterrows():
        reg.add(row['Date'], row['SKU_ID'], row['Region'], '',
                "stockout", "Fill_Rate_Pct", row['fill_rate'], 100,
                "unconstrained_demand_applied",
                f"Stockout in inventory_stockouts: Fill_Rate={row['fill_rate']}%, Lost_Units={row['lost']}, Reason={row['Reason']}",
                "inventory_stockouts")

    # 7. Zero runs (potential hidden stockouts) – check for SKUs with sudden zero runs
    so_sku_reg = set(zip(stockouts['SKU_ID'], stockouts['Region']))
    # Fast grouping: find series that had active sales then long zero runs
    active_series = df.groupby(['SKU_ID', 'Region']).agg(
        total_months=('Date', 'count'),
        zero_months=('Units_Sold_num', lambda s: (s == 0).sum() + s.isna().sum()),
        max_demand=('Units_Sold_num', 'max')
    ).reset_index()
    # High demand series with high zero proportion
    hidden_candidates = active_series[
        (active_series['max_demand'] > 50) &
        (active_series['zero_months'] >= 6)
    ].head(100)

    for _, row in hidden_candidates.iterrows():
        sku, reg_name = row['SKU_ID'], row['Region']
        has_so = (sku, reg_name) in so_sku_reg
        action = "stockout_corroborated" if has_so else "potential_hidden_stockout_investigate"
        reg.add('2019-2026', sku, reg_name, 'ALL',
                "zero_run", "Units_Sold", int(row['zero_months']), int(row['max_demand']),
                action,
                f"SKU {sku} in {reg_name} has {int(row['zero_months'])} zero/missing months despite peak demand of {int(row['max_demand'])} units (stockout recorded: {has_so})",
                "")

    # 8. MEA before 2022-01 (EV-024 market entry check)
    mea_before = df[
        (df['Region'] == 'Middle East & Africa') &
        (df['Date'] < '2022-01') &
        (df['Units_Sold_num'] > 0)
    ]
    if len(mea_before):
        for _, row in mea_before.iterrows():
            reg.add(row['Date'], row['SKU_ID'], 'Middle East & Africa', '',
                    "mea_before_market_entry", "Units_Sold", row['Units_Sold_num'], 0,
                    "flagged_for_review",
                    "MEA sales before 2022-01 market entry (EV-024) – should not exist",
                    "EV-024")

    # Save global register
    reg_df = reg.as_dataframe()
    reg_df.to_csv(OUT_ANOMALY_REG, index=False)
    return reg_df

