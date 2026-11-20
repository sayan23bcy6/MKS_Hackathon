"""
clean_master_macro_events.py – Cleans sku_master, macro_indicators, events_calendar.
Rules: S1-S2, M1-M5, E1-E4.
"""

import re
import pandas as pd
import numpy as np
from pipeline.config import (
    PREFIX_TO_FAMILY,
    MACRO_PCT_FRACTION_YEAR_MAX,
    PMI_MIN, PMI_MAX, PMI_DECIMAL_SHIFT_THRESHOLD,
)
from pipeline.audit import audit

DQ_MME = {}


# ══════════════════════════════════════════════════════════════════════════════
# SKU MASTER CLEANING (S1, S2)
# ══════════════════════════════════════════════════════════════════════════════

def clean_sku_master(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """S1–S2: Validate sku_master. No silent changes."""
    dq = {}
    df = raw.copy()

    # S1: Family/BU rule breach (R8)
    # Precision Machined Parts should map to exactly one BU
    pmp_bus = df[df['Product_Family'] == 'Precision Machined Parts']['Business_Unit'].value_counts()
    dq['S1_pmp_bu_counts'] = pmp_bus.to_dict()
    conflict_skus = df[
        (df['Product_Family'] == 'Precision Machined Parts') &
        (df['Business_Unit'] != 'Advanced Manufacturing')
    ]
    dq['S1_conflict_count'] = len(conflict_skus)
    if len(conflict_skus):
        dq['S1_conflict_skus'] = conflict_skus['SKU_ID'].tolist()
        # Flag but do NOT auto-correct (documented as unresolved)
        for _, row in conflict_skus.iterrows():
            audit.log("sku_master.csv", row['SKU_ID'], "Business_Unit",
                      row['Business_Unit'], "CONFLICT_FLAGGED", "S1",
                      f"SKU {row['SKU_ID']} has Product_Family=Precision Machined Parts but Business_Unit={row['Business_Unit']} (expected Advanced Manufacturing) – flagged as unresolved per R8")

    # S2: Lifecycle fields check
    has_eol = df[df['EOL_Date'].notna() & (df['EOL_Date'].str.strip() != '')]
    has_launch = df[df['Lifecycle_Status'] == 'Planned Launch']
    dq['S2_eol_count'] = len(has_eol)
    dq['S2_planned_launch_count'] = len(has_launch)

    # Verify Replaces_SKU targets exist
    all_skus = set(df['SKU_ID'])
    for _, row in df[df['Replaces_SKU'].notna()].iterrows():
        ref = str(row['Replaces_SKU']).strip()
        if ref and ref not in all_skus:
            audit.log("sku_master.csv", row['SKU_ID'], "Replaces_SKU",
                      ref, "MISSING_REFERENCE", "S2",
                      f"Replaces_SKU target {ref} not found in sku_master")
            dq['S2_missing_replaces_sku'] = dq.get('S2_missing_replaces_sku', 0) + 1

    # Rule 7 checks on sku_master
    # Check Launch_Date / EOL_Date formats
    for col in ['Launch_Date', 'EOL_Date']:
        for _, row in df[df[col].notna()].iterrows():
            d = str(row[col]).strip()
            if d and not (re.match(r'^\d{4}-\d{2}$', d) or re.match(r'^\d{4}-\d{2}-\d{2}$', d)):
                audit.log("sku_master.csv", row['SKU_ID'], col,
                          d, "FORMAT_ERROR", "S_R7",
                          f"{col} has unexpected format: {d}")
                dq['S_R7_date_format_errors'] = dq.get('S_R7_date_format_errors', 0) + 1

    return df, dq


# ══════════════════════════════════════════════════════════════════════════════
# MACRO INDICATORS CLEANING (M1–M5)
# ══════════════════════════════════════════════════════════════════════════════

def clean_macro(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """M1–M5: Clean macro_indicators."""
    dq = {}
    df = raw.copy()

    # Parse numeric columns
    num_cols = [
        'ISM_Manufacturing_PMI', 'Auto_Production_YoY_Pct',
        'Vehicle_Inventory_Days', 'Raw_Material_Cost_Index',
        'New_Vehicle_Orders_Index', 'Industrial_Util_Rate_Pct',
        'Fuel_Price_Index', 'EV_Adoption_Index'
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # M1: Duplicate month 2022-06
    dup_months = df[df.duplicated('Date', keep=False)]
    dq['M1_dup_months'] = int(dup_months['Date'].nunique())
    if dq['M1_dup_months'] > 0:
        # For 2022-06: compare each conflicting value to neighbours
        for dup_date in dup_months['Date'].unique():
            idxs = df[df['Date'] == dup_date].index.tolist()
            # Get neighbours
            prev_date = df[df['Date'] < dup_date]['Date'].max()
            next_date = df[df['Date'] > dup_date]['Date'].min()
            prev_row = df[df['Date'] == prev_date].iloc[0] if prev_date else None
            next_row = df[df['Date'] == next_date].iloc[0] if next_date else None

            chosen_idx = None
            reason = ""
            for col in ['ISM_Manufacturing_PMI', 'New_Vehicle_Orders_Index']:
                v1 = df.at[idxs[0], col]
                v2 = df.at[idxs[1], col]
                if prev_row is not None and next_row is not None:
                    # Pick the value closer to the linear interpolation of neighbours
                    interp = (prev_row[col] + next_row[col]) / 2
                    if abs(v1 - interp) < abs(v2 - interp):
                        chosen_idx = idxs[0]
                        reason = f"Row A ({v1}) is closer to neighbour interpolation ({interp:.2f})"
                    else:
                        chosen_idx = idxs[1]
                        reason = f"Row B ({v2}) is closer to neighbour interpolation ({interp:.2f})"
                    break

            if chosen_idx is None:
                chosen_idx = idxs[0]
                reason = "Could not resolve by neighbours; Row A selected; both values set to NA for interpolation"
                for col in ['ISM_Manufacturing_PMI', 'New_Vehicle_Orders_Index']:
                    df.at[chosen_idx, col] = np.nan

            drop_idx = [i for i in idxs if i != chosen_idx]
            audit.log("macro_indicators.csv", dup_date, "row",
                      f"2 rows for {dup_date}", f"kept idx={chosen_idx}", "M1",
                      f"Duplicate month {dup_date}: {reason}")
            df = df.drop(index=drop_idx)
            dq['M1_resolution'] = reason

    # M2: Auto_Production_YoY_Pct fraction → percent for 2019
    year_col = df['Year'].astype(int) if 'Year' in df.columns else df['Date'].str[:4].astype(int)
    frac_mask = (year_col <= MACRO_PCT_FRACTION_YEAR_MAX) & (df['Auto_Production_YoY_Pct'].abs() < 1.0)
    dq['M2_fraction_rows'] = int(frac_mask.sum())
    for idx in df.index[frac_mask]:
        old_val = df.at[idx, 'Auto_Production_YoY_Pct']
        new_val = old_val * 100
        audit.log("macro_indicators.csv", df.at[idx, 'Date'],
                  'Auto_Production_YoY_Pct', old_val, new_val, "M2",
                  "2019 values stored as fractions; multiplied by 100 to match Pct header")
        df.at[idx, 'Auto_Production_YoY_Pct'] = new_val

    # M3: PMI decimal shift (4.79 → 47.9)
    pmi_bad = (df['ISM_Manufacturing_PMI'] < PMI_DECIMAL_SHIFT_THRESHOLD) & df['ISM_Manufacturing_PMI'].notna()
    dq['M3_pmi_implausible'] = int(pmi_bad.sum())
    for idx in df.index[pmi_bad]:
        old_val = df.at[idx, 'ISM_Manufacturing_PMI']
        candidate = old_val * 10
        date_v = df.at[idx, 'Date']
        # Check neighbours
        prev_dates = df[df['Date'] < date_v]['Date']
        next_dates = df[df['Date'] > date_v]['Date']
        if len(prev_dates) and len(next_dates):
            prev_pmi = df[df['Date'] == prev_dates.max()]['ISM_Manufacturing_PMI'].values[0]
            next_pmi = df[df['Date'] == next_dates.min()]['ISM_Manufacturing_PMI'].values[0]
            interp = (prev_pmi + next_pmi) / 2
            if PMI_MIN <= candidate <= PMI_MAX and abs(candidate - interp) < abs(old_val - interp):
                new_val = candidate
                reason = f"Decimal shift: {old_val}→{new_val} fits neighbour average {interp:.1f}"
            else:
                new_val = np.nan
                reason = f"PMI {old_val} implausible but ×10 not confirmed by neighbours; set to NA for interpolation"
        else:
            new_val = candidate if PMI_MIN <= candidate <= PMI_MAX else np.nan
            reason = f"PMI decimal shift correction applied"
        audit.log("macro_indicators.csv", date_v, "ISM_Manufacturing_PMI",
                  old_val, new_val, "M3", reason)
        df.at[idx, 'ISM_Manufacturing_PMI'] = new_val

    # M4: Blank cells – linear interpolation
    blank_cells = {}
    for col in num_cols:
        blanks = df[df[col].isna()]['Date'].tolist()
        if blanks:
            blank_cells[col] = blanks

    # Sort by Date so linear interpolation works chronologically
    df = df.sort_values('Date').reset_index(drop=True)
    for col, dates in blank_cells.items():
        df[col] = df[col].astype(float).interpolate(method='linear')
        for d in dates:
            new_val = df.loc[df['Date'] == d, col].values[0]
            audit.log("macro_indicators.csv", d, col,
                      "NaN", round(float(new_val), 4), "M4",
                      "Blank cell filled by linear time-based interpolation")
    dq['M4_blank_cells'] = {k: v for k, v in blank_cells.items()}

    # M5: Note real extreme values
    extreme_check = df['Auto_Production_YoY_Pct']
    dq['M5_min_auto_prod'] = float(extreme_check.min())
    dq['M5_max_auto_prod'] = float(extreme_check.max())

    # Rule 7 on macro: check for any negative PMI
    neg_pmi = df[df['ISM_Manufacturing_PMI'] < 0]
    if len(neg_pmi):
        dq['M_R7_negative_pmi'] = len(neg_pmi)
        for idx in neg_pmi.index:
            audit.log("macro_indicators.csv", df.at[idx, 'Date'],
                      'ISM_Manufacturing_PMI', df.at[idx, 'ISM_Manufacturing_PMI'],
                      'FLAGGED', 'M_R7', 'Negative PMI is impossible; flagged for review')

    return df, dq


# ══════════════════════════════════════════════════════════════════════════════
# EVENTS CALENDAR CLEANING (E1–E4)
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_event_date(d: str) -> tuple[str, str]:
    """
    Return (start_date YYYY-MM-DD, end_date YYYY-MM-DD).
    If only YYYY-MM, end is last day of month.
    """
    if pd.isna(d) or str(d).strip() == '':
        return None, None
    s = str(d).strip()
    if re.match(r'^\d{4}-\d{2}$', s):
        # YYYY-MM: start = first of month, end = last of month
        yr, mo = int(s[:4]), int(s[5:7])
        import calendar
        last_day = calendar.monthrange(yr, mo)[1]
        return f"{s}-01", f"{s}-{last_day:02d}"
    if re.match(r'^\d{4}-\d{2}-\d{2}$', s):
        return s, s
    raise ValueError(f"Unrecognised event date format: {d!r}")


def clean_events(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """E1–E4: Clean events_calendar."""
    dq = {}
    df = raw.copy()

    # E1: Normalise mixed date formats
    start_dates = []
    end_dates = []
    formats_changed = 0
    for idx, row in df.iterrows():
        s_start, _ = _normalise_event_date(row['Start_Date'])
        raw_end = row['End_Date']
        if pd.isna(raw_end) or str(raw_end).strip() == '':
            # E2: Missing End_Date
            s_end = None
        else:
            _, s_end = _normalise_event_date(raw_end)
        start_dates.append(s_start)
        end_dates.append(s_end)
        if s_start != row['Start_Date']:
            audit.log("events_calendar.csv", row['Event_ID'], "Start_Date",
                      row['Start_Date'], s_start, "E1", "Date normalised to YYYY-MM-DD")
            formats_changed += 1

    df['Start_Date'] = start_dates
    df['End_Date_Clean'] = end_dates
    dq['E1_dates_normalised'] = formats_changed

    # E2: EV-052 open-ended event
    open_ended = df[df['End_Date_Clean'].isna()]
    dq['E2_open_ended_events'] = len(open_ended)
    for _, row in open_ended.iterrows():
        audit.log("events_calendar.csv", row['Event_ID'], "End_Date",
                  "(blank)", "OPEN_ENDED", "E2",
                  f"Event {row['Event_ID']} has no End_Date – treated as open-ended per description")

    # E3: Split "Customer | Region" in Scope_Value
    df['Scope_Customer'] = None
    df['Scope_Region'] = None
    for idx, row in df.iterrows():
        scope_val = str(row.get('Scope_Value', '')).strip()
        if '|' in scope_val:
            parts = [p.strip() for p in scope_val.split('|')]
            df.at[idx, 'Scope_Customer'] = parts[0]
            df.at[idx, 'Scope_Region'] = parts[1] if len(parts) > 1 else None
            audit.log("events_calendar.csv", row['Event_ID'], "Scope_Value",
                      scope_val, f"customer={parts[0]}, region={parts[1] if len(parts)>1 else ''}", "E3",
                      "Combined Customer|Region scope split into Scope_Customer and Scope_Region")
        else:
            df.at[idx, 'Scope_Customer'] = None
            df.at[idx, 'Scope_Region'] = scope_val if row.get('Scope_Type') == 'Region' else None
    dq['E3_split_scope'] = int(df['Scope_Customer'].notna().sum())

    # E4: Annotate data-impact events
    impact_types = {
        'Price Change': 'price_feature',
        'Announcement': 'demand_shift_feature',
        'Supply Constraint': 'stockout_check',
        'Disruption': 'stockout_check',
        'System': 'data_quality_investigation',
        'Market Entry': 'region_start_check',
        'Holiday': 'seasonal_feature',
        'Customer Event': 'customer_demand_feature',
        'Logistics': 'channel_demand_feature',
        'Corporate': 'informational',
        'Marketing': 'demand_boost_feature',
    }
    df['Data_Impact'] = df['Event_Type'].map(impact_types).fillna('informational')
    dq['E4_impact_annotations'] = df['Event_Type'].value_counts().to_dict()

    # Rule 7: Check for duplicate Event_IDs
    dup_ev = df[df.duplicated('Event_ID', keep=False)]
    dq['E_R7_dup_event_ids'] = len(dup_ev)

    # Replace original End_Date column
    df['End_Date'] = df['End_Date_Clean']
    df = df.drop(columns=['End_Date_Clean'])

    return df, dq

