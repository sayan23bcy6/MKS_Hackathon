"""
reports.py - Data quality report generator (JSON and Markdown formats).
Compiles audit metrics, data cleaning results, and documented decision rules.
"""

import json
from pathlib import Path
import pandas as pd
from pipeline.config import OUT_DQ_JSON, OUT_DQ_MD


def generate_dq_report(dq_train: dict,
                        dq_macro: dict,
                        dq_sku: dict,
                        dq_events: dict,
                        audit_log: pd.DataFrame,
                        anomaly_register: pd.DataFrame) -> dict:
    """
    Compile full data quality report from pipeline executions.
    Saves JSON and Markdown versions to outputs.
    """
    report = {
        "summary": {
            "total_audit_entries": len(audit_log),
            "total_anomalies": len(anomaly_register),
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
            },
            "T3_typo_year": {
                "description": "Impossible future dates (e.g. 2062 transposed to 2026)",
                "found": dq_train.get('T3_found', 0),
                "corrected": dq_train.get('T3_corrected', 0),
                "rule": "T3",
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
            },
            "T6_sales_channel": {
                "description": "Sales channel label variants normalized to standard set",
                "normalised": dq_train.get('T6_normalised', 0),
                "rule": "T6",
            },
            "T7_product_family": {
                "description": "Missing product family inferred deterministically from SKU prefix",
                "blank_count": dq_train.get('T7_blank_product_family', 0),
                "filled": dq_train.get('T7_product_family_filled', 0),
                "rule": "T7",
            },
            "T8_sku_id": {
                "description": "SKU_ID format variations normalized to standard PREFIX-NNN",
                "bad_before": dq_train.get('T8_bad_sku_before', 0),
                "normalised": dq_train.get('T8_normalised', 0),
                "rule": "T8",
            },
            "T9_sku_description": {
                "description": "Missing SKU descriptions populated from sku_master reference",
                "blank_count": dq_train.get('T9_blank_desc', 0),
                "filled": dq_train.get('T9_filled', 0),
                "rule": "T9",
            },
            "T10_customer_name": {
                "description": "Customer name casing and whitespace variants unified",
                "normalised": dq_train.get('T10_normalised', 0),
                "rule": "T10",
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
            },
            "T12_unit_price": {
                "description": "Unit_Price_USD zeros, scale errors, and currency shifts repaired with annual median reference",
                "filled_zero_price_rows": dq_train.get('T12_filled_zero_price', 0),
                "scale_100x_band": dq_train.get('T12_scale_100x', 0),
                "scale_85x_band": dq_train.get('T12_scale_85x', 0),
                "scale_5_20x_band": dq_train.get('T12_scale_5_20x', 0),
                "rule": "T12",
            },
            "T13_revenue": {
                "description": "Revenue_USD recalculated as Units_Sold * Unit_Price_USD",
                "blank_count": dq_train.get('T13_blank_revenue', 0),
                "discrepancy_before": dq_train.get('T13_discrepancy_before', 0),
                "recomputed": dq_train.get('T13_recomputed', 0),
                "rule": "T13",
            },
            "R5_lifecycle": {
                "description": "Lifecycle validation: sales outside launch/EOL boundaries zeroed",
                "eol_violations": dq_train.get('R5_eol_violations', 0),
                "launch_violations": dq_train.get('R5_launch_violations', 0),
                "rule": "R5",
            },
            "T4b_date_sku_region": {
                "description": "Deduplication of duplicate load records vs genuine distinct customer orders",
                "dsr_dups": dq_train.get('T4b_dsr_dups', 0),
                "rows_after": dq_train.get('T4b_after', 0),
                "rule": "T4b/R9",
            },
        },
        "macro_issues": dq_macro,
        "sku_master_issues": dq_sku,
        "events_issues": dq_events,
        "decision_rules_applied": {
            "price_out_of_band_scaling": {
                "rule_id": "T12",
                "count": dq_train.get('T12_scale_5_20x', 0) + dq_train.get('T12_scale_85x', 0) + dq_train.get('T12_scale_100x', 0),
                "decision": "Replaced zero, negative, and out-of-band (<0.5x or >2.0x median) unit prices with the annual SKU median price across regions.",
                "reason": "Corrects cent-scale entry errors, currency conversion disparities, and manual data-entry slips while maintaining intra-year price stability.",
            },
            "blank_region_categorization": {
                "rule_id": "T5",
                "count": dq_train.get('T5_blank_unassigned', 0),
                "decision": "Assigned persistent unmatchable blank regions to canonical category 'Unassigned'.",
                "reason": "Preserves total company sales volume and transaction count without guessing geographic origins.",
            },
            "high_volume_orders": {
                "rule_id": "T11",
                "count": dq_train.get('T11_large_values', 0),
                "decision": "Retained genuine large orders (>5000 units) in historical training data after validating non-corrupt status.",
                "reason": "Reflects genuine commercial customer bulk procurement rather than data transmission errors.",
            },
            "macro_2022_06_duplicate": {
                "rule_id": "M1",
                "count": dq_macro.get('M1_dup_months', 0),
                "decision": dq_macro.get('M1_resolution', 'Selected observation with minimum Euclidean distance to adjacent months.'),
                "reason": "Preserves continuity with May 2022 and July 2022 macroeconomic indicators.",
            },
            "macro_2023_05_pmi_decimal": {
                "rule_id": "M3",
                "count": dq_macro.get('M3_pmi_implausible', 0),
                "decision": "Corrected 4.79 to 47.9 via decimal point shift.",
                "reason": "April 2023 was 47.2 and June 2023 was 46.1; value of 4.79 was a typographical single-decimal slip.",
            },
            "sku_bu_conflict_resolution": {
                "rule_id": "S1",
                "count": dq_sku.get('S1_conflict_count', 0),
                "decision": "Assigned PMP-004 to Business Unit 'Advanced Manufacturing' per Rule R8 and transaction history majority.",
                "reason": "All transaction records for PMP-004 consistently map to Advanced Manufacturing.",
            },
        },
    }

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
    lines += [
        f"| T1 | {ti['T1_date_formats']['description']} | {ti['T1_date_formats']['raw_rows']} rows | {ti['T1_date_formats']['formats_fixed']} fixed |",
        f"| T2 | {ti['T2_year_month_mismatch']['description']} | - | {ti['T2_year_month_mismatch']['count']} rows |",
        f"| T3 | {ti['T3_typo_year']['description']} | {ti['T3_typo_year']['found']} found | {ti['T3_typo_year']['corrected']} corrected |",
        f"| T4 | {ti['T4_duplicate_record_ids']['description']} | {ti['T4_duplicate_record_ids']['rows_before']} | {ti['T4_duplicate_record_ids']['rows_after']} ({ti['T4_duplicate_record_ids']['rows_dropped']} duplicates removed) |",
        f"| T5 | {ti['T5_region']['description']} | {ti['T5_region']['blank_count']} blank | {ti['T5_region']['unassigned_count']} set to Unassigned |",
        f"| T6 | {ti['T6_sales_channel']['description']} | - | {ti['T6_sales_channel']['normalised']} normalized |",
        f"| T7 | {ti['T7_product_family']['description']} | {ti['T7_product_family']['blank_count']} blank | {ti['T7_product_family']['filled']} filled from SKU prefix |",
        f"| T8 | {ti['T8_sku_id']['description']} | {ti['T8_sku_id']['bad_before']} malformed | {ti['T8_sku_id']['normalised']} normalized to PREFIX-NNN |",
        f"| T9 | {ti['T9_sku_description']['description']} | {ti['T9_sku_description']['blank_count']} blank | {ti['T9_sku_description']['filled']} populated from sku_master |",
        f"| T10 | {ti['T10_customer_name']['description']} | - | {ti['T10_customer_name']['normalised']} normalized |",
        f"| T11 | Units_Sold formatting and negatives | {ti['T11_units_sold']['comma_formatted']} comma, {ti['T11_units_sold']['negative']} negative | Converted to numeric and absolute value applied |",
        f"| T12 | Unit_Price_USD scale errors & zero prices | {ti['T12_unit_price']['filled_zero_price_rows']} zero rows | Annual SKU median reference price applied |",
        f"| T13 | Revenue_USD recalculation | {ti['T13_revenue']['discrepancy_before']} discrepant | Recalculated as Units_Sold * Unit_Price_USD |",
        f"| R5 | Sales outside launch/EOL lifecycle | - | Zeroed out {ti['R5_lifecycle']['eol_violations']} EOL and {ti['R5_lifecycle']['launch_violations']} pre-launch records |",
        f"| T4b | Duplicate Date-SKU-Region records | - | Kept genuine distinct customers; resolved duplicate loads |",
    ]

    lines += [
        "",
        "## Macroeconomic Indicators",
        "",
        f"- **M1** Duplicate 2022-06 month: {report['macro_issues'].get('M1_dup_months', 0)} duplicates resolved via nearest linear neighbor interpolation.",
        f"- **M2** Auto Production YoY fraction-to-percentage: {report['macro_issues'].get('M2_fraction_rows', 0)} rows in 2019 scaled x100.",
        f"- **M3** PMI decimal point correction: {report['macro_issues'].get('M3_pmi_implausible', 0)} rows corrected (4.79 to 47.9).",
        f"- **M4** Missing indicators: Linear time interpolation applied with Macro_Imputed indicator flag.",
        f"- **M5** COVID period structural break: Is_COVID_Period flag active for 2020-03 through 2021-06.",
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
