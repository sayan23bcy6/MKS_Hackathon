"""
reports.py – WS1 Advanced: Generate data-quality report (JSON + Markdown).
"""

import json
import pandas as pd
from pathlib import Path
from pipeline.config import OUT_DQ_JSON, OUT_DQ_MD


def generate_dq_report(dq_train: dict,
                        dq_macro: dict,
                        dq_sku: dict,
                        dq_events: dict,
                        audit_log: pd.DataFrame,
                        anomaly_register: pd.DataFrame) -> dict:
    """
    Compile the full data-quality report from all pipeline outputs.
    Returns the full report dict (also saved as JSON and Markdown).
    """
    report = {
        "summary": {
            "total_audit_entries": len(audit_log),
            "total_anomalies": len(anomaly_register),
        },
        "train_issues": {
            "T1_date_formats": {
                "description": "Mixed date formats in Date column",
                "raw_rows": dq_train.get('raw_rows', 0),
                "formats_fixed": dq_train.get('T1_formats_fixed', 0),
                "rule": "T1",
            },
            "T2_year_month_mismatch": {
                "description": "Year/Month disagree with Date (R1: Date authoritative)",
                "count": dq_train.get('T2_year_month_mismatch', 0),
                "rule": "T2",
            },
            "T3_typo_year": {
                "description": "Impossible future dates (2062-xx typo for 2026-xx)",
                "found": dq_train.get('T3_found', 0),
                "corrected": dq_train.get('T3_corrected', 0),
                "rule": "T3",
            },
            "T4_duplicate_record_ids": {
                "description": "Duplicate Record_IDs (loaded twice) – kept latest timestamp",
                "rows_before": dq_train.get('T4_before', 0),
                "rows_dropped": dq_train.get('T4_dup_record_ids', 0),
                "rows_after": dq_train.get('T4_after', 0),
                "date_sku_region_dups_after": dq_train.get('T4_date_sku_region_dups_after_dedup', 0),
                "rule": "T4",
            },
            "T5_region": {
                "description": "Region labels inconsistent or blank",
                "blank_count": dq_train.get('T5_blank_regions', 0),
                "blank_resolved": dq_train.get('T5_blank_resolved', 0),
                "blank_unresolved": dq_train.get('T5_blank_unresolved', 0),
                "normalised": dq_train.get('T5_normalised', 0),
                "rule": "T5",
            },
            "T6_sales_channel": {
                "description": "Sales_Channel label variants normalised",
                "normalised": dq_train.get('T6_normalised', 0),
                "rule": "T6",
            },
            "T7_product_family": {
                "description": "Product_Family blank – derived from SKU prefix",
                "blank_count": dq_train.get('T7_blank_product_family', 0),
                "filled": dq_train.get('T7_product_family_filled', 0),
                "rule": "T7",
            },
            "T8_sku_id": {
                "description": "SKU_ID format errors – normalised to PREFIX-NNN",
                "bad_before": dq_train.get('T8_bad_sku_before', 0),
                "normalised": dq_train.get('T8_normalised', 0),
                "rule": "T8",
            },
            "T9_sku_description": {
                "description": "SKU_Description blank – filled from sku_master",
                "blank_count": dq_train.get('T9_blank_desc', 0),
                "filled": dq_train.get('T9_filled', 0),
                "rule": "T9",
            },
            "T10_customer_name": {
                "description": "Customer_Name variants normalised",
                "normalised": dq_train.get('T10_normalised', 0),
                "rule": "T10",
            },
            "T11_units_sold": {
                "description": "Units_Sold: blank, dash, comma, negative, large values",
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
                "description": "Unit_Price_USD zeros and scale errors",
                "filled_zero_price_rows": dq_train.get('T12_filled_zero_price', 0),
                "scale_100x_band": dq_train.get('T12_scale_100x', 0),
                "scale_85x_band": dq_train.get('T12_scale_85x', 0),
                "scale_5_20x_band_unresolved": dq_train.get('T12_scale_5_20x', 0),
                "rule": "T12",
            },
            "T13_revenue": {
                "description": "Revenue_USD recomputed after price/unit cleaning",
                "blank_count": dq_train.get('T13_blank_revenue', 0),
                "discrepancy_before": dq_train.get('T13_discrepancy_before', 0),
                "recomputed": dq_train.get('T13_recomputed', 0),
                "rule": "T13",
            },
            "R5_lifecycle": {
                "description": "Sales after EOL_Date or before Launch_Date (should be 0)",
                "eol_violations": dq_train.get('R5_eol_violations', 0),
                "launch_violations": dq_train.get('R5_launch_violations', 0),
                "rule": "R5",
            },
            "T4b_date_sku_region": {
                "description": "Duplicate Date-SKU-Region after full cleaning (R9)",
                "dsr_dups": dq_train.get('T4b_dsr_dups', 0),
                "rows_after": dq_train.get('T4b_after', 0),
                "rule": "T4b/R9",
            },
        },
        "macro_issues": dq_macro,
        "sku_master_issues": dq_sku,
        "events_issues": dq_events,
        "unresolved_items": {
            "price_5_20x_band": {
                "count": dq_train.get('T12_scale_5_20x', 0),
                "decision": "Flagged in anomaly register; kept for demand analysis; excluded from price-based analysis",
                "reason": "No clear single-factor explanation; not auto-corrected per Rule 7",
            },
            "price_85x_band_rupee_hypothesis": {
                "count": dq_train.get('T12_scale_85x', 0),
                "decision": "Treated as currency denomination error (INR→USD) and replaced with SKU-year reference price. India region subset showed higher concentration of this band.",
                "reason": "Approximately 85× ratio consistent with INR/USD exchange rate in relevant period (~80-85 INR/USD). Hypothesis is plausible and documented.",
            },
            "large_units_gt5000": {
                "count": dq_train.get('T11_large_values', 0),
                "decision": "Flagged in anomaly register; retained as-is pending cross-check with inventory_stockouts Requested_Units",
                "reason": "Cannot determine if data error or real bulk order without stockout corroboration",
            },
            "macro_2022_06_duplicate": {
                "count": dq_macro.get('M1_dup_months', 0),
                "decision": dq_macro.get('M1_resolution', 'Resolved by neighbour interpolation'),
                "reason": "Chosen row is closer to the linear interpolation of 2022-05 and 2022-07 neighbours",
            },
            "macro_2023_05_pmi": {
                "count": dq_macro.get('M3_pmi_implausible', 0),
                "decision": "Corrected 4.79 → 47.9 (×10 decimal shift) based on neighbour confirmation",
                "reason": "2023-04 PMI=47.2 and 2023-06 PMI=46.1; 47.9 fits the trend; 4.79 does not",
            },
            "sku_bu_conflict": {
                "count": dq_sku.get('S1_conflict_count', 0),
                "skus": dq_sku.get('S1_conflict_skus', []),
                "decision": "Flagged for manual review; not auto-corrected; transaction data should be used to determine correct BU",
                "reason": "Cannot determine correct BU from master data alone without transaction evidence",
            },
        },
    }

    # Save JSON
    with open(OUT_DQ_JSON, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"DQ report (JSON) saved: {OUT_DQ_JSON}")

    # Save Markdown
    _write_dq_markdown(report)
    return report


def _write_dq_markdown(report: dict):
    lines = [
        "# Data Quality Report – Celeris Automobili Demand Forecasting",
        "",
        f"**Total audit log entries:** {report['summary']['total_audit_entries']}  ",
        f"**Total anomalies flagged:** {report['summary']['total_anomalies']}",
        "",
        "---",
        "",
        "## Train.csv Issues",
        "",
        "| Issue | Description | Before | After / Count | Rule |",
        "|---|---|---|---|---|",
    ]

    ti = report['train_issues']
    lines += [
        f"| T1 | {ti['T1_date_formats']['description']} | {ti['T1_date_formats']['raw_rows']} rows | {ti['T1_date_formats']['formats_fixed']} fixed | T1 |",
        f"| T2 | {ti['T2_year_month_mismatch']['description']} | — | {ti['T2_year_month_mismatch']['count']} rows | T2 |",
        f"| T3 | {ti['T3_typo_year']['description']} | {ti['T3_typo_year']['found']} found | {ti['T3_typo_year']['corrected']} corrected | T3 |",
        f"| T4 | {ti['T4_duplicate_record_ids']['description']} | {ti['T4_duplicate_record_ids']['rows_before']} | {ti['T4_duplicate_record_ids']['rows_after']} (+{ti['T4_duplicate_record_ids']['rows_dropped']} dropped) | T4 |",
        f"| T5 | {ti['T5_region']['description']} | {ti['T5_region']['blank_count']} blank | {ti['T5_region']['blank_unresolved']} unresolved | T5 |",
        f"| T6 | {ti['T6_sales_channel']['description']} | — | {ti['T6_sales_channel']['normalised']} normalised | T6 |",
        f"| T7 | {ti['T7_product_family']['description']} | {ti['T7_product_family']['blank_count']} blank | {ti['T7_product_family']['filled']} filled | T7 |",
        f"| T8 | {ti['T8_sku_id']['description']} | {ti['T8_sku_id']['bad_before']} bad | {ti['T8_sku_id']['normalised']} fixed | T8 |",
        f"| T9 | {ti['T9_sku_description']['description']} | {ti['T9_sku_description']['blank_count']} blank | {ti['T9_sku_description']['filled']} filled | T9 |",
        f"| T10 | {ti['T10_customer_name']['description']} | — | {ti['T10_customer_name']['normalised']} normalised | T10 |",
        f"| T11 | Units_Sold: blanks | {ti['T11_units_sold']['blank']} | — | T11 |",
        f"| T11 | Units_Sold: dashes (NULL) | {ti['T11_units_sold']['dash']} | — | T11 |",
        f"| T11 | Units_Sold: comma-formatted | {ti['T11_units_sold']['comma_formatted']} | — | T11 |",
        f"| T11 | Units_Sold: negatives | {ti['T11_units_sold']['negative']} | abs() applied | T11 |",
        f"| T11 | Units_Sold: large (>5000) | {ti['T11_units_sold']['large_values_gt5000']} | flagged | T11 |",
        f"| T12 | Unit_Price_USD zero (with units>0) | — | {ti['T12_unit_price']['filled_zero_price_rows']} filled | T12 |",
        f"| T12 | Price ×100 band | {ti['T12_unit_price']['scale_100x_band']} | reference price applied | T12 |",
        f"| T12 | Price ×85 band (INR hypothesis) | {ti['T12_unit_price']['scale_85x_band']} | reference price applied | T12 |",
        f"| T12 | Price 5–20× band (unresolved) | {ti['T12_unit_price']['scale_5_20x_band_unresolved']} | flagged, kept for demand | T12 |",
        f"| T13 | Revenue_USD recomputed | {ti['T13_revenue']['discrepancy_before']} discrepant | {ti['T13_revenue']['recomputed']} recomputed | T13 |",
        f"| R5 | Sales after EOL or before Launch | — | {ti['R5_lifecycle']['eol_violations']} EOL, {ti['R5_lifecycle']['launch_violations']} launch violations | R5 |",
        f"| R9/T4b | Duplicate Date-SKU-Region after cleaning | — | {ti['T4b_date_sku_region']['dsr_dups']} | T4b |",
    ]

    lines += [
        "",
        "## Macro Indicators Issues",
        "",
        f"- **M1** Duplicate 2022-06: {report['macro_issues'].get('M1_dup_months', 0)} months affected. Resolution: {report['macro_issues'].get('M1_resolution', 'N/A')}",
        f"- **M2** Auto_Production_YoY_Pct fraction→percent for 2019: {report['macro_issues'].get('M2_fraction_rows', 0)} rows fixed",
        f"- **M3** PMI decimal shift (2023-05): {report['macro_issues'].get('M3_pmi_implausible', 0)} rows corrected",
        f"- **M4** Blank cells interpolated: {report['macro_issues'].get('M4_blank_cells', {})}",
        f"- **M5** Real extreme values noted: Min Auto_Prod_YoY={report['macro_issues'].get('M5_min_auto_prod', 'N/A'):.1f}%, Max={report['macro_issues'].get('M5_max_auto_prod', 'N/A'):.1f}% (COVID period – not errors)",
        "",
        "## SKU Master Issues",
        "",
        f"- **S1** Precision Machined Parts BU conflict: {report['sku_master_issues'].get('S1_conflict_count', 0)} SKU(s): {report['sku_master_issues'].get('S1_conflict_skus', [])}",
        f"- **S2** Planned launches: {report['sku_master_issues'].get('S2_planned_launch_count', 0)}, EOL SKUs: {report['sku_master_issues'].get('S2_eol_count', 0)}",
        "",
        "## Events Calendar Issues",
        "",
        f"- **E1** Date formats normalised: {report['events_issues'].get('E1_dates_normalised', 0)} rows",
        f"- **E2** Open-ended events (no End_Date): {report['events_issues'].get('E2_open_ended_events', 0)}",
        f"- **E3** Customer|Region scope split: {report['events_issues'].get('E3_split_scope', 0)}",
        f"- **E4** Event type data-impact annotations applied",
        "",
        "## Unresolved Items",
        "",
    ]

    for key, val in report['unresolved_items'].items():
        lines.append(f"### {key}")
        lines.append(f"- **Count/affected:** {val.get('count', 'N/A')}")
        lines.append(f"- **Decision:** {val.get('decision', 'N/A')}")
        lines.append(f"- **Reason:** {val.get('reason', 'N/A')}")
        lines.append("")

    with open(OUT_DQ_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"DQ report (Markdown) saved: {OUT_DQ_MD}")

