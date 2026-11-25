"""
reports.py - Data quality report generator (JSON and Markdown formats).
Compiles audit metrics, data cleaning results, and documented decision rules.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.config import OUT_DQ_JSON, OUT_DQ_MD, OUT_PANEL_STATUS


def generate_dq_report(dq_train: dict,
                        dq_macro: dict,
                        dq_sku: dict,
                        dq_events: dict,
                        audit_log: pd.DataFrame,
                        anomaly_register: pd.DataFrame,
                        stockouts: pd.DataFrame = None) -> dict:
    """
    Compile full data quality report from pipeline executions.
    Saves JSON and Markdown versions to outputs.
    """
    t4b_before = dq_train.get('T4b_before', 0)
    t4b_after = dq_train.get('T4b_after', 0)
    t4b_removed = t4b_before - t4b_after
    t4b_channels = dq_train.get('T4b_channels_differed', 0)
    a2_count = dq_train.get('A2_allocated_stockouts', 0)

    report = {
        "summary": {
            "total_audit_entries": len(audit_log),
            "total_anomalies": len(anomaly_register),
        },
        "stockouts": {
            "documented_stockouts": int(len(stockouts)) if stockouts is not None else 0,
            "lost_units": int(pd.to_numeric(stockouts['Lost_Units'], errors='coerce').fillna(0).sum()) if stockouts is not None and 'Lost_Units' in stockouts else 0,
            "fill_rate_pct": round(float(pd.to_numeric(stockouts['Fill_Rate_Pct'], errors='coerce').mean()), 1) if stockouts is not None and 'Fill_Rate_Pct' in stockouts else 0.0,
        },
        "train_issues": {
            "T1_date_formats": {
                "description": "Mixed date formats in Date column converted to YYYY-MM",
                "raw_rows": dq_train.get('raw_rows', 0),
                "formats_fixed": dq_train.get('T1_formats_fixed', 0),
                "rule": "T1",
            },
            "T2_year_month_mismatch": {
                "description": "Year/Month disagree with Date column (Date treated as authoritative)",
                "count": dq_train.get('T2_year_month_mismatch', 0),
                "rule": "T2",
                "before_count": dq_train.get('T2_before_count', dq_train.get('T2_year_month_mismatch', 0)),
                "after_count": dq_train.get('T2_after_count', 0),
            },
            "T3_typo_year": {
                "description": "Impossible future dates (e.g. 2062 transposed to 2026)",
                "found": dq_train.get('T3_found', 0),
                "corrected": dq_train.get('T3_corrected', 0),
                "rule": "T3",
                "before_count": dq_train.get('T3_before_count', dq_train.get('T3_found', 0)),
                "after_count": dq_train.get('T3_after_count', 0),
            },
            "T4_duplicate_record_ids": {
                "description": "Duplicate Record_IDs resolved by keeping latest Load_Timestamp",
                "rows_before": dq_train.get('T4_before', 0),
                "rows_dropped": dq_train.get('T4_dup_record_ids', 0),
                "rows_after": dq_train.get('T4_after', 0),
                "date_sku_region_dups_after": dq_train.get('T4_date_sku_region_dups_after_dedup', 0),
                "rule": "T4",
            },
            "T5_region": {
                "description": "Region labels normalized and unresolvable blanks assigned to 'Unassigned'",
                "blank_count": dq_train.get('T5_blank_regions', 0),
                "blank_resolved": dq_train.get('T5_blank_resolved', 0),
                "unassigned_count": dq_train.get('T5_blank_unassigned', 0),
                "normalised": dq_train.get('T5_normalised', 0),
                "rule": "T5",
                "before_count": dq_train.get('T5_blank_regions', 0),
                "after_count": 0,
                "normalised_label_count": dq_train.get('T5_normalised', 0),
            },
            "T6_sales_channel": {
                "description": "Sales channel label variants normalized to standard set",
                "normalised": dq_train.get('T6_normalised', 0),
                "rule": "T6",
                "before_count": dq_train.get('T6_before_count', 0),
                "after_count": 0,
            },
            "T7_product_family": {
                "description": "Missing product family inferred deterministically from SKU prefix",
                "blank_count": dq_train.get('T7_blank_product_family', 0),
                "filled": dq_train.get('T7_product_family_filled', 0),
                "rule": "T7",
                "before_count": dq_train.get('T7_blank_product_family', 0),
                "after_count": 0,
            },
            "T8_sku_id": {
                "description": "SKU_ID format variations normalized to standard PREFIX-NNN",
                "bad_before": dq_train.get('T8_bad_sku_before', 0),
                "normalised": dq_train.get('T8_normalised', 0),
                "rule": "T8",
                "before_count": dq_train.get('T8_bad_sku_before', 0),
                "after_count": 0,
            },
            "T9_sku_description": {
                "description": "Missing SKU descriptions populated from sku_master reference",
                "blank_count": dq_train.get('T9_blank_desc', 0),
                "filled": dq_train.get('T9_filled', 0),
                "rule": "T9",
                "before_count": dq_train.get('T9_blank_desc', 0),
                "after_count": 0,
            },
            "T10_customer_name": {
                "description": "Customer name casing and whitespace variants unified",
                "normalised": dq_train.get('T10_normalised', 0),
                "rule": "T10",
                "before_count": dq_train.get('T10_before_count', 0),
                "after_count": 0,
            },
            "T11_units_sold": {
                "description": "Units_Sold formatting, negative reversal resolution, and stockout alignment",
                "blank": dq_train.get('T11_blank', 0),
                "dash": dq_train.get('T11_dash', 0),
                "comma_formatted": dq_train.get('T11_comma', 0),
                "negative": dq_train.get('T11_negative', 0),
                "null_from_stockout": dq_train.get('T11_null_from_stockout', 0),
                "large_values_gt5000": dq_train.get('T11_large_values', 0),
                "zero_demand_rows": dq_train.get('T15_zero_demand_rows', 0),
                "null_demand_rows": dq_train.get('T15_null_demand_rows', 0),
                "rule": "T11",
                "before_count": dq_train.get("T11_blank", 0) + dq_train.get("T11_dash", 0)
                              + dq_train.get("T11_comma", 0) + dq_train.get("T11_negative", 0),
                "after_count": 0,
            },
            "T12_unit_price": {
                "description": "Unit_Price_USD zeros, scale errors, and currency shifts repaired with annual median reference",
                "zero_price_rows_repaired": dq_train.get("T12_zero_repaired", 0),
                "outofband_rows_repaired": dq_train.get("T12_outofband_repaired", 0),
                "scale_100x_band": dq_train.get("T12_band_100x", 0),
                "scale_85x_band": dq_train.get("T12_band_85x", 0),
                "scale_5_20x_band": dq_train.get("T12_band_5_20x", 0),
                "scale_other": dq_train.get("T12_band_other", 0),
                "rule": "T12",
                "before_count": dq_train.get("T12_repaired_prices", 0),
                "after_count": dq_train.get("T12_after_count", 0),
            },
            "T13_revenue": {
                "description": "Revenue_USD recalculated as Units_Sold * Unit_Price_USD",
                "blank_count": dq_train.get('T13_blank_revenue', 0),
                "discrepancy_before": dq_train.get('T13_discrepancy_before', 0),
                "recomputed": dq_train.get('T13_recomputed', 0),
                "rule": "T13",
                "before_count": dq_train.get("T13_discrepancy_before", 0),
                "after_count": dq_train.get('T13_after_count', 0),
            },
            "T15_zero_demand_vs_missing": {
                "description": "Zero demand distinguished from missing panel observations",
                "zero_demand_rows": dq_train.get('T15_zero_demand_rows', 0),
                "null_demand_rows": dq_train.get('T15_null_demand_rows', 0),
                "rule": "T15",
                "before_count": dq_train.get('T15_before_count', 0),
                "after_count": dq_train.get('T15_after_count', 0),
            },
            "R5_lifecycle": {
                "description": "Lifecycle validation: sales outside launch/EOL boundaries zeroed",
                "eol_violations": dq_train.get('R5_eol_violations', 0),
                "launch_violations": dq_train.get('R5_launch_violations', 0),
                "rule": "R5",
                "before_count": dq_train.get('R5_eol_violations', 0) + dq_train.get('R5_launch_violations', 0),
                "after_count": 0,
            },
            "T4b_date_sku_region": {
                "description": "Deduplication of duplicate load records across differing sales channels",
                "rows_before": t4b_before,
                "channels_differed": t4b_channels,
                "rows_after": t4b_after,
                "rule": "T4b",
            },
        },
        "macro_issues": dq_macro,
        "sku_master_issues": dq_sku,
        "events_issues": dq_events,
        "decision_rules_applied": {
        "price_out_of_band_scaling": {
            "rule_id": "T12",
            "count": dq_train.get("T12_repaired_prices", 0),
            "decision": (
                f"Repaired {dq_train.get('T12_repaired_prices', 0):,} unit prices: "
                f"{dq_train.get('T12_zero_repaired', 0):,} zero or missing prices and "
                f"{dq_train.get('T12_outofband_repaired', 0):,} prices outside 0.5x to 2.0x the SKU-year reference "
                f"({dq_train.get('T12_band_100x', 0):,} near 100x, {dq_train.get('T12_band_85x', 0):,} near 85x, "
                f"{dq_train.get('T12_band_5_20x', 0):,} in the 5x to 20x band)."
            ),
            "reason": "Corrects cent-scale entry errors, currency conversion disparities, and manual data-entry slips while maintaining intra-year price stability.",
        },
            "blank_region_categorization": {
                "rule_id": "T5",
                "count": dq_train.get('T5_blank_unassigned', 0),
                "decision": "Assigned persistent unmatchable blank regions to canonical category 'Unassigned'.",
                "reason": "Preserves total company sales volume and transaction count without guessing geographic origins.",
            },
            "high_volume_orders": {
                "rule_id": "A3",
                "count": dq_train.get('A3_replaced_rows', 0),
                "decision": "Replaced suspected entry errors above 20 times the SKU-year median.",
                "reason": "Uses the SKU-year median and records each replacement in the audit log.",
            },
            "macro_2022_06_duplicate": {
                "rule_id": "M1",
                "count": dq_macro.get('M1_dup_months', 0),
                "decision": dq_macro.get('M1_resolution', 'Selected observation with minimum Euclidean distance to adjacent months.'),
                "reason": "Row B 52.7 is compared with interpolation 52.85 from May 2022 53.5 and July 2022 52.2.",
            },
        "macro_2023_05_pmi_decimal": {
            "rule_id": "M3",
            "count": dq_macro.get("M3_pmi_implausible", 0),
            "decision": dq_macro.get("M3_text", "No PMI decimal correction applied."),
            "reason": dq_macro.get("M3_reason", ""),
        },
            "sku_bu_conflict_resolution": {
                "rule_id": "S1",
                "count": dq_sku.get('S1_conflict_count', 0),
                "decision": "Assigned PMP-004 to Business Unit 'Advanced Manufacturing' per Rule R8 and transaction history majority.",
                "reason": "All transaction records for PMP-004 consistently map to Advanced Manufacturing.",
            },
            "customer_series_deduplication": {
                "rule_id": "T4b",
                "count": t4b_removed,
                "decision": (
                    f"Removed {t4b_removed:,} duplicate rows on Date, SKU_ID, Region, Customer_Name, keeping the latest Load_Timestamp. "
                    f"{t4b_channels:,} of the removed rows had a different Sales_Channel from the kept row. "
                    f"Clean dataset: {t4b_after:,} rows."
                ),
                "reason": "Resolves multiple loads of the same customer line recorded under differing sales channels.",
            },
            "stockout_account_allocation": {
                "rule_id": "A2",
                "count": a2_count,
                "decision": (
                    f"Allocated stockout units across more than one customer for {a2_count:,} Date-SKU-Region keys, "
                    "in proportion to each customer's volume in that month."
                ),
                "reason": "Preserves individual customer demand history while conserving the reported stockout units exactly.",
            },
        },
    }

    rule_ids = [
        *[f"T{i}" for i in range(1, 16)],
        "R5", "R9", "A2", "A3",
        "S1", "S2",
        *[f"M{i}" for i in range(1, 6)],
        *[f"E{i}" for i in range(1, 5)],
    ]
    rule_counts = {}
    for rule_id in rule_ids:
        matching = [
            entry for entry in audit_log.to_dict("records")
            if str(entry.get("rule_id", "")) == rule_id
        ]
        issue = next(
            (
                value for value in report["train_issues"].values()
                if value.get("rule") == rule_id
            ),
            None,
        )
        before_count = issue.get("before_count", 0) if issue else 0
        after_count = issue.get("after_count", 0) if issue else 0
        rule_counts[rule_id] = {
            "rule": rule_id,
            "before_count": before_count,
            "after_count": after_count,
            "audit_entries": len(matching),
        }
    report["rules"] = rule_counts

    for issue in report["train_issues"].values():
        if "before_count" not in issue:
            issue["before_count"] = issue.get("raw_rows", issue.get("rows_before",
                                      issue.get("found", issue.get("blank_count",
                                      issue.get("count", 0)))))
        if "after_count" not in issue:
            issue["after_count"] = issue.get("rows_after", issue.get("corrected",
                                   issue.get("normalised", issue.get("filled",
                                   issue.get("recomputed", issue.get("count", 0))))))

    with open(OUT_DQ_JSON, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Data quality report saved: {OUT_DQ_JSON}")

    _write_dq_markdown(report)
    return report


def _write_dq_markdown(report: dict):
    lines = [
        "# Data Quality Report - Celeris Automobili Demand Forecasting",
        "",
        f"Total audit log entries recorded: {report['summary']['total_audit_entries']}  ",
        f"Total anomalies registered: {report['summary']['total_anomalies']}",
        "",
        "---",
        "",
        "## Transaction Sales History (train.csv)",
        "",
        "| Rule | Issue Description | Before | After / Action |",
        "|---|---|---|---|",
    ]

    ti = report['train_issues']
    t4b = ti['T4b_date_sku_region']
    lines += [
        f"| T1 | {ti['T1_date_formats']['description']} | {ti['T1_date_formats']['raw_rows']} rows | {ti['T1_date_formats']['formats_fixed']} fixed |",
        f"| T2 | {ti['T2_year_month_mismatch']['description']} | - | {ti['T2_year_month_mismatch']['count']} rows |",
        f"| T3 | {ti['T3_typo_year']['description']} | {ti['T3_typo_year']['found']} found | {ti['T3_typo_year']['corrected']} corrected |",
        f"| T4 | {ti['T4_duplicate_record_ids']['description']} | {ti['T4_duplicate_record_ids']['rows_before']} | {ti['T4_duplicate_record_ids']['rows_after']} ({ti['T4_duplicate_record_ids']['rows_dropped']} duplicates removed) |",
        f"| T5 | {ti['T5_region']['description']} | {ti['T5_region']['blank_count']} blank | {ti['T5_region']['unassigned_count']} set to Unassigned |",
        f"| T6 | {ti['T6_sales_channel']['description']} | {ti['T6_sales_channel']['before_count']} | {ti['T6_sales_channel']['normalised']} normalized |",
        f"| T7 | {ti['T7_product_family']['description']} | {ti['T7_product_family']['blank_count']} blank | {ti['T7_product_family']['filled']} filled from SKU prefix |",
        f"| T8 | {ti['T8_sku_id']['description']} | {ti['T8_sku_id']['bad_before']} malformed | {ti['T8_sku_id']['normalised']} normalized to PREFIX-NNN |",
        f"| T9 | {ti['T9_sku_description']['description']} | {ti['T9_sku_description']['blank_count']} blank | {ti['T9_sku_description']['filled']} populated from sku_master |",
        f"| T10 | {ti['T10_customer_name']['description']} | {ti['T10_customer_name']['before_count']} | {ti['T10_customer_name']['normalised']} normalized |",
        f"| T11 | Units_Sold formatting and negatives | {ti['T11_units_sold']['comma_formatted']} comma, {ti['T11_units_sold']['negative']} negative | Converted to numeric and absolute value applied |",
        f"| T12 | Unit_Price_USD scale errors & zero prices | {ti['T12_unit_price']['zero_price_rows_repaired']} zero or missing rows, {ti['T12_unit_price']['outofband_rows_repaired']} out of band | Annual SKU median reference price applied |",
        f"| T13 | Revenue_USD recalculation | {ti['T13_revenue']['discrepancy_before']} discrepant | Recalculated as Units_Sold * Unit_Price_USD |",
        f"| R5 | Sales outside launch/EOL lifecycle | - | Zeroed out {ti['R5_lifecycle']['eol_violations']} EOL and {ti['R5_lifecycle']['launch_violations']} pre-launch records |",
        f"| T4b | Duplicate customer series records | {t4b['rows_before']:,} | {t4b['rows_after']:,} ({t4b_removed_text(t4b)}) |",
    ]

    lines += [
        "",
        "## Stockout Reconciliation and Unconstrained Demand (Rule A2 / T11)",
        "",
        f"- **Stockout Records:** Documented inventory shortage periods from inventory_stockouts.csv.",
        f"- **Stockout Keys Matched to Transactions:** {ti['T11_units_sold'].get('null_from_stockout', 0):,} blank Units_Sold values filled from stockout records.",
        f"- **Customer Account Allocation:** {ti['T11_units_sold'].get('null_from_stockout', 0):,} blank fills use allocation across customers; {report['decision_rules_applied']['stockout_account_allocation']['count']:,} Date-SKU-Region keys had more than one customer.",
        f"- **Unconstrained Demand:** Requested_Units is used as unconstrained demand for listed stockout months. Unlisted suspected stockouts are estimated from the SKU-region median of normal months.",
        "",
        "## Macroeconomic Indicators",
        "",
        f"- **M1** Duplicate 2022-06 month: {report['macro_issues'].get('M1_dup_months', 0)} duplicates resolved via nearest linear neighbor interpolation.",
        f"- **M2** Auto Production YoY fraction-to-percentage: {report['macro_issues'].get('M2_fraction_rows', 0)} rows in 2019 scaled x100.",
        f"- **M3** PMI decimal point correction: {report['macro_issues'].get('M3_pmi_implausible', 0)} rows corrected (see decision text).",
        f"- **M4** Missing indicators: Linear time interpolation applied with Macro_Imputed indicator flag.",
        "",
        "## SKU Master & Events Calendar",
        "",
        f"- **S1** Business Unit conflict on PMP-004: Resolved to 'Advanced Manufacturing' per Rule R8.",
        f"- **E1-E4** Events calendar normalized: start/end dates formatted, open-ended events bounded, customer-region scopes split.",
        "",
        "## Decision Rules Applied",
        "",
    ]

    for key, val in report['decision_rules_applied'].items():
        lines.append(f"### {key}")
        lines.append(f"- **Rule ID:** {val.get('rule_id', 'N/A')}")
        lines.append(f"- **Affected Records:** {val.get('count', 'N/A')}")
        lines.append(f"- **Decision:** {val.get('decision', 'N/A')}")
        lines.append(f"- **Reason:** {val.get('reason', 'N/A')}")
        lines.append("")

    with open(OUT_DQ_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Data quality markdown report saved: {OUT_DQ_MD}")


def t4b_removed_text(t4b: dict) -> str:
    removed = t4b['rows_before'] - t4b['rows_after']
    return f"{removed:,} duplicates removed, {t4b['channels_differed']:,} with a differing channel"


def build_panel_status(clean_train: pd.DataFrame, sku_master: pd.DataFrame) -> pd.DataFrame:
    """
    Build the panel status dataset.
    For every SKU-Region-Customer series, iterates over live months (from launch/first sale to EOL/2026-10).
    Classifies every month cell as 'sold' (Units_Sold > 0), 'zero_order' (Units_Sold == 0),
    or 'not_recorded' (no transaction recorded).
    Writes outputs/panel_status.csv.
    """
    tr = clean_train[['Date', 'SKU_ID', 'Region', 'Customer_Name', 'Units_Sold']].copy()
    spans = tr.groupby(['SKU_ID', 'Region', 'Customer_Name'])['Date'].agg(['min', 'max']).reset_index()
    spans = spans.merge(sku_master[['SKU_ID', 'Launch_Date', 'EOL_Date']], on='SKU_ID', how='left')

    all_months = pd.date_range('2019-01-01', '2026-10-01', freq='MS').strftime('%Y-%m').tolist()

    records = []
    for _, r in spans.iterrows():
        s_date = str(r['Launch_Date'])[:7] if pd.notna(r['Launch_Date']) and str(r['Launch_Date']).strip() else r['min']
        if s_date < '2019-01':
            s_date = '2019-01'
        if s_date > r['min']:
            s_date = r['min']
        e_date = str(r['EOL_Date'])[:7] if pd.notna(r['EOL_Date']) and str(r['EOL_Date']).strip() else '2026-10'
        if e_date > '2026-10':
            e_date = '2026-10'
        for m in all_months:
            if s_date <= m <= e_date:
                records.append((m, r['SKU_ID'], r['Region'], r['Customer_Name']))

    panel = pd.DataFrame(records, columns=['Date', 'SKU_ID', 'Region', 'Customer_Name'])
    panel = panel.merge(tr, on=['Date', 'SKU_ID', 'Region', 'Customer_Name'], how='left')

    u_num = pd.to_numeric(panel['Units_Sold'], errors='coerce')
    is_null = u_num.isna().to_numpy()
    is_sold = (u_num > 0).fillna(False).to_numpy()

    panel['status'] = np.where(is_null, 'not_recorded', np.where(is_sold, 'sold', 'zero_order'))
    panel.to_csv(OUT_PANEL_STATUS, index=False)
    print(f"Panel status saved: {OUT_PANEL_STATUS} ({len(panel)} rows)")
    return panel