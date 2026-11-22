"""
anomalies.py - Anomaly register tracking market shocks, demand shifts, and outliers.
Logs real market disruptions, structural breaks, stockouts, and data adjustments.
"""

import numpy as np
import pandas as pd
import re
from pipeline.config import (
    UNITS_LARGE_THRESHOLD,
    OUT_ANOMALY_REG,
)

ANOMALY_COLS = [
    "Date", "SKU_ID", "Region", "Customer_Name",
    "anomaly_type", "field", "value", "baseline",
    "action", "reason", "event_reference",
    "type", "evidence", "Classification", "ratio",
]


class AnomalyRegister:
    def __init__(self):
        self._records = []

    def add(self, date, sku_id, region, customer,
            anomaly_type, field, value, baseline,
            action, reason, event_ref="", ratio=None):
        classification = "error" if anomaly_type in {
            "sign_reversal", "price_scaling_correction"
        } else "real event"
        evidence = f"{field}: value={value}, baseline={baseline}"
        self._records.append({
            "Date": str(date), "SKU_ID": str(sku_id), "Region": str(region),
            "Customer_Name": str(customer),
            "anomaly_type": str(anomaly_type), "field": str(field),
            "value": value, "baseline": baseline,
            "action": str(action), "reason": str(reason),
            "event_reference": str(event_ref),
            "type": str(anomaly_type),
            "evidence": evidence,
            "Classification": classification,
            "ratio": ratio,
        })

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records, columns=ANOMALY_COLS)

    def save(self):
        df = self.as_dataframe()
        df.to_csv(OUT_ANOMALY_REG, index=False)
        return len(df)


def build_anomaly_register(clean_train: pd.DataFrame,
                           events: pd.DataFrame,
                           stockouts: pd.DataFrame,
                           audit_log: pd.DataFrame) -> pd.DataFrame:
    """
    Build structured anomaly register covering:
    - High-volume orders (>5000 units)
    - Early buying patterns prior to announced price adjustments
    - COVID-19 and supply chain shock windows
    - Documented inventory stockouts and constrained demand
    - Structural mean-shift candidate periods
    """
    reg = AnomalyRegister()
    df = clean_train.copy()
    df['Units_Sold_num'] = pd.to_numeric(df['Units_Sold'], errors='coerce')
    df['Price_num'] = pd.to_numeric(df['Unit_Price_USD'], errors='coerce')

    # High-volume orders replaced as suspected entry errors
    replaced = df[df.get('_A3_Ratio', pd.Series(index=df.index, dtype=float)).notna()]
    for _, row in replaced.iterrows():
        original = row['_A3_Original_Units']
        ratio = row['_A3_Ratio']
        reg.add(row['Date'], row['SKU_ID'], row['Region'], row.get('Customer_Name', ''),
                "high_volume_order", "Units_Sold", original, row['Units_Sold_num'],
                "replaced_suspected_entry_error",
                f"suspected entry error: ratio {ratio:.6f} against SKU-year median",
                "A3", ratio)

    # 2. Negative unit reversals handled
    neg_audit = audit_log[
        (audit_log['rule_id'] == 'T11') &
        (audit_log['reason'].str.contains('Negative', na=False))
    ] if audit_log is not None and len(audit_log) else pd.DataFrame()
    for _, row in neg_audit.iterrows():
        kept = df[df['Record_ID'] == row['row_key']] if 'Record_ID' in df.columns else pd.DataFrame()
        date_value = sku_value = region_value = ''
        reason_text = str(row['reason'])
        date_match = re.search(r'Date=([^;]+)', reason_text)
        sku_match = re.search(r'SKU_ID=([^;]+)', reason_text)
        region_match = re.search(r'Region=([^;]+)', reason_text)
        if date_match:
            date_value = date_match.group(1)
        if sku_match:
            sku_value = sku_match.group(1)
        if region_match:
            region_value = region_match.group(1)
        if kept.empty:
            kept_row = {'Date': date_value, 'SKU_ID': sku_value, 'Region': region_value}
        else:
            kept_row = kept.iloc[0]
        reason = "Negative units recorded due to clerical sign inversion"
        if row['reason'].startswith("Negative Units_Sold dropped"):
            reason += "; reversed row dropped as a duplicate"
        reg.add(kept_row.get('Date', ''), kept_row.get('SKU_ID', ''), kept_row.get('Region', ''),
                '',
                "sign_reversal", "Units_Sold", row['raw_value'], 0,
                "absolute_value_applied",
                reason,
                "T11")

    # 3. Price scale adjustments
    price_audit = audit_log[
        (audit_log['rule_id'] == 'T12')
    ] if audit_log is not None and len(audit_log) else pd.DataFrame()
    for _, row in price_audit.head(200).iterrows():
        kept = df[df['Record_ID'] == row['row_key']] if 'Record_ID' in df.columns else pd.DataFrame()
        date_value = kept.iloc[0].get('Date', '') if not kept.empty else ''
        sku_value = kept.iloc[0].get('SKU_ID', '') if not kept.empty else ''
        region_value = kept.iloc[0].get('Region', '') if not kept.empty else ''
        reg.add(date_value, sku_value, region_value, '',
                "price_scaling_correction", "Unit_Price_USD", row['raw_value'], row['new_value'],
                "reference_price_applied",
                row['reason'],
                "T12")

    # 4. Early buying before EV-030 price rise
    early_buy_months = ['2022-11', '2022-12']
    early_buy = df[df['Date'].isin(early_buy_months) & (df['Units_Sold_num'] > 0)]
    for sku, grp in early_buy.groupby('SKU_ID'):
        baseline_months = [f"{yr}-{mo}" for yr in [2019, 2020, 2021] for mo in ['11', '12']]
        baseline = df[
            df['Date'].isin(baseline_months) & (df['SKU_ID'] == sku)
        ]['Units_Sold_num'].mean()
        actual = grp['Units_Sold_num'].mean()
        if pd.notna(baseline) and pd.notna(actual) and baseline > 0 and actual > 1.2 * baseline:
            reg.add(', '.join(early_buy_months), sku, 'ALL', 'ALL',
                    "early_buying", "Units_Sold",
                    round(float(actual), 1), round(float(baseline), 1),
                    "modeled_as_event_feature",
                    f"Nov-Dec 2022 demand {actual:.0f} vs historical {baseline:.0f} - pre-announcement procurement before 2023 price adjustment (EV-030)",
                    "EV-030")

    # 5. COVID disruption period (2020-02 to 2020-06)
    covid_months = ['2020-02', '2020-03', '2020-04', '2020-05', '2020-06']
    covid = df[df['Date'].isin(covid_months) & (df['Units_Sold_num'] > 0)]
    for (d, reg_name), grp in covid.groupby(['Date', 'Region']):
        reg.add(d, 'ALL_SKUS', reg_name, 'ALL_CUSTOMERS',
                "pandemic_supply_shock", "Units_Sold", int(grp['Units_Sold_num'].sum()), '',
                "modeled_as_structural_break",
                f"COVID-19 market contraction ({d}): total {int(grp['Units_Sold_num'].sum())} units across {len(grp)} transactions",
                "EV-009,EV-010,EV-011,EV-012,EV-013")

    # 6. Stockout months
    so = stockouts.copy()
    so['fill_rate'] = pd.to_numeric(so['Fill_Rate_Pct'], errors='coerce')
    so['lost'] = pd.to_numeric(so['Lost_Units'], errors='coerce')
    for _, row in so[so['fill_rate'] < 100].head(500).iterrows():
        reg.add(row['Date'], row['SKU_ID'], row['Region'], '',
                "inventory_stockout", "Fill_Rate_Pct", row['fill_rate'], 100,
                "unconstrained_demand_reconstructed",
                f"Stockout recorded: Fill_Rate={row['fill_rate']}%, Lost_Units={row['lost']}, Reason={row['Reason']}",
                "inventory_stockouts")

    reg.save()
    return reg.as_dataframe()
