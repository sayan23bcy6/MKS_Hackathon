"""
clean_train.py - Applies cleaning rules T1-T15 to train.csv.
Every change is logged to audit.py:audit.
Outputs: a cleaned DataFrame, a dict of before/after counts per issue.
"""

import re
import pandas as pd
import numpy as np
from pipeline.config import (
    REGION_MAP, CANONICAL_REGIONS,
    CHANNEL_MAP, CANONICAL_CHANNELS,
    PREFIX_TO_FAMILY, CUSTOMER_MAP, CANONICAL_CUSTOMERS,
    MONTH_NAME_MAP,
    TRAIN_DATE_MIN, TRAIN_DATE_MAX,
    PRICE_SCALE_100X_LOW, PRICE_SCALE_100X_HIGH,
    PRICE_SCALE_85X_LOW, PRICE_SCALE_85X_HIGH,
    PRICE_SCALE_MED_LOW, PRICE_SCALE_MED_HIGH,
    UNITS_LARGE_THRESHOLD,
)
from pipeline.audit import audit
from pipeline.unconstrained import allocate_stockout_metric

DQ = {}  # data-quality counts, populated as we clean


def _parse_date_to_yyyymm(raw: str) -> str | None:
    """Parse any known date format to YYYY-MM string. Returns None if unparseable."""
    s = str(raw).strip()
    # Already correct
    if re.match(r'^\d{4}-\d{2}$', s):
        return s
    # YYYY-MM-DD
    if re.match(r'^\d{4}-\d{2}-\d{2}$', s):
        return s[:7]
    # YYYY/MM
    if re.match(r'^\d{4}/\d{2}$', s):
        return s[:4] + '-' + s[5:7]
    # MM/YYYY - first field must be <=12
    if re.match(r'^\d{2}/\d{4}$', s):
        mm, yyyy = s.split('/')
        if int(mm) > 12:
            raise ValueError(f"MM/YYYY first field > 12: {s}")
        return f"{yyyy}-{mm}"
    # MM-YYYY
    if re.match(r'^\d{2}-\d{4}$', s):
        mm, yyyy = s.split('-')
        if int(mm) > 12:
            raise ValueError(f"MM-YYYY first field > 12: {s}")
        return f"{yyyy}-{mm}"
    # Month name: "Jan 2019" or "Jan-2019"
    months = '|'.join(MONTH_NAME_MAP.keys())
    m = re.match(rf'^({months})[\s\-](\d{{4}})$', s, re.IGNORECASE)
    if m:
        mon_str = m.group(1).lower()
        yyyy = m.group(2)
        return f"{yyyy}-{MONTH_NAME_MAP[mon_str]}"
    return None


def apply_t1_date_parsing(df: pd.DataFrame) -> pd.DataFrame:
    """T1: Normalise all Date formats to YYYY-MM."""
    before = len(df)
    parsed = df['Date'].apply(_parse_date_to_yyyymm)
    bad_mask = parsed.isna()
    if bad_mask.any():
        raise RuntimeError(f"T1: {bad_mask.sum()} dates could not be parsed. Examples: {df.loc[bad_mask, 'Date'].unique()[:5]}")

    changed = parsed != df['Date']
    DQ['T1_changed'] = int(changed.sum())
    for idx in df.index[changed]:
        audit.log("train.csv", df.at[idx, 'Record_ID'], "Date",
                  df.at[idx, 'Date'], parsed[idx], "T1",
                  "Date format normalised to YYYY-MM")

    df = df.copy()
    df['Date'] = parsed
    DQ['T1_before'] = before
    DQ['T1_formats_fixed'] = int(changed.sum())
    return df


def apply_t3_typo_year(df: pd.DataFrame) -> pd.DataFrame:
    """T3: Correct 2062-xx typos to 2026-xx when Year col agrees."""
    df = df.copy()
    mask = df['Date'].str[:4] > '2026'
    # Only correct when Year col says 2026 (digit transposition 2062<->2026)
    typo_mask = mask & (df['Year'].str.strip() == '2026')
    DQ['T3_found'] = int(mask.sum())
    DQ['T3_corrected'] = int(typo_mask.sum())
    DQ['T3_before_count'] = int(typo_mask.sum())
    for idx in df.index[typo_mask]:
        raw_date = df.at[idx, 'Date']
        corrected = '2026' + raw_date[4:]
        audit.log("train.csv", df.at[idx, 'Record_ID'], "Date",
                  raw_date, corrected, "T3",
                  "Year digit typo 2062->2026 corrected (Year col=2026, date out of 2019-2026 range)")
        df.at[idx, 'Date'] = corrected
    # Remaining dates still > 2026-10 that couldn't be corrected: fail loudly
    still_bad = df['Date'].str[:7] > TRAIN_DATE_MAX
    DQ['T3_after_count'] = int(still_bad.sum())
    if still_bad.any():
        raise RuntimeError(f"T3: {still_bad.sum()} rows with Date > {TRAIN_DATE_MAX} remain after correction. Manual review needed.")
    return df


def apply_t2_year_month(df: pd.DataFrame) -> pd.DataFrame:
    """T2: Recompute Year and Month from cleaned Date (R1: Date is authoritative)."""
    df = df.copy()
    expected_year = df['Date'].str[:4]
    expected_month = df['Date'].str[5:7].str.lstrip('0')  # strip leading zero for comparison
    actual_year = df['Year'].str.strip()
    actual_month = df['Month'].str.strip()

    year_wrong = actual_year != expected_year
    month_wrong = actual_month != expected_month
    any_wrong = year_wrong | month_wrong

    DQ['T2_year_month_mismatch'] = int(any_wrong.sum())
    DQ['T2_before_count'] = int(any_wrong.sum())
    for idx in df.index[any_wrong]:
        rid = df.at[idx, 'Record_ID']
        if year_wrong[idx]:
            audit.log("train.csv", rid, "Year", df.at[idx, 'Year'],
                      expected_year[idx], "T2", "Year recomputed from Date (R1)")
        if month_wrong[idx]:
            audit.log("train.csv", rid, "Month", df.at[idx, 'Month'],
                      expected_month[idx], "T2", "Month recomputed from Date (R1)")

    df['Year'] = expected_year
    df['Month'] = df['Date'].str[5:7].str.lstrip('0')
    recomputed_wrong = (
        (df['Year'].astype(str) != df['Date'].str[:4])
        | (df['Month'].astype(str) != df['Date'].str[5:7].str.lstrip('0'))
    )
    DQ['T2_after_count'] = int(recomputed_wrong.sum())
    return df


def apply_t4_dedup_record_id(df: pd.DataFrame) -> pd.DataFrame:
    """T4: Keep the row with the latest Load_Timestamp per Record_ID (R2)."""
    df = df.copy()
    df['_load_ts'] = pd.to_datetime(df['Load_Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    before = len(df)
    # For each Record_ID group with >1 row, log the dropped ones
    dup_mask = df.duplicated('Record_ID', keep=False)
    if dup_mask.any():
        dup_df = df[dup_mask].copy()
        dup_df = dup_df.sort_values(['Record_ID', '_load_ts'], ascending=[True, False])
        # The row to keep is the one with max timestamp
        keep_idx = dup_df.groupby('Record_ID')['_load_ts'].idxmax()
        drop_idx = dup_df.index.difference(keep_idx)
        for idx in drop_idx:
            rid = df.at[idx, 'Record_ID']
            audit.log("train.csv", rid, "Record_ID",
                      f"row with ts={df.at[idx,'Load_Timestamp']}",
                      "DROPPED", "T4", "Duplicate Record_ID - kept row with latest Load_Timestamp (R2)")

    df = df.sort_values('_load_ts', ascending=False).drop_duplicates('Record_ID', keep='first')
    df = df.drop(columns=['_load_ts'])
    DQ['T4_before'] = before
    DQ['T4_dup_record_ids'] = (before - len(df))
    DQ['T4_after'] = len(df)

    # Also check for Date-SKU-Region duplicates with DIFFERENT Record_IDs (R9, report only)
    df_valid = df.dropna(subset=['Date', 'SKU_ID', 'Region'])
    dsr_dups = df_valid.duplicated(['Date', 'SKU_ID', 'Region'], keep=False)
    DQ['T4_date_sku_region_dups_after_dedup'] = int(dsr_dups.sum())
    return df


def _normalise_region(raw) -> str | None:
    """Return canonical region or None if unresolvable."""
    if pd.isna(raw) or str(raw).strip() == '':
        return None  # blank - handle separately
    s = str(raw).strip()
    # Already canonical
    if s in CANONICAL_REGIONS:
        return s
    # Normalise: casefold, collapse multiple spaces, replace hyphens
    # Preserve & for MEA lookup
    normalised = re.sub(r'\s+', ' ', s.casefold().replace('-', ' ')).strip()
    # Build several search variants
    variants = [
        normalised,
        normalised.replace(' and ', ' & '),  # "middle east and africa" -> "middle east & africa"
        normalised.replace(' and ', '&'),
    ]
    for key, val in REGION_MAP.items():
        # Normalise the map key the same way
        key_n = re.sub(r'\s+', ' ', key.casefold().replace('-', ' ')).strip()
        if any(v == key_n for v in variants):
            return val
    return None  # unknown - will fail pipeline


def apply_t5_region(df: pd.DataFrame) -> pd.DataFrame:
    """T5: Normalise Region to canonical set. Blank -> attempt recovery, else 'Unassigned'."""
    df = df.copy()
    raw_region = df['Region'].copy()

    # First, attempt to fix blank regions from duplicate Record_ID partners
    # (The de-dup has already run, so we look for SKU+Customer+Date matches)
    blank_mask = raw_region.isna() | (raw_region.str.strip() == '')
    DQ['T5_blank_regions'] = int(blank_mask.sum())

    # Pre-build fast lookup for SKU+Customer+Date among non-blank rows
    valid_df = df[~blank_mask]
    lookup_region = valid_df.groupby(['Date', 'SKU_ID', 'Customer_Name'])['Region'].unique().to_dict()

    resolved_from_match = 0
    for idx in df.index[blank_mask]:
        date_v = df.at[idx, 'Date']
        sku_v = df.at[idx, 'SKU_ID']
        cust_v = df.at[idx, 'Customer_Name']
        cands = lookup_region.get((date_v, sku_v, cust_v), np.array([]))
        if len(cands) == 1:
            resolved = _normalise_region(cands[0])
            if resolved:
                audit.log("train.csv", df.at[idx, 'Record_ID'], "Region",
                          "(blank)", resolved, "T5",
                          "Blank region recovered from SKU+Customer+Date match")
                df.at[idx, 'Region'] = resolved
                raw_region[idx] = resolved
                resolved_from_match += 1
                continue
        # Cannot recover: assign to Unassigned (real category kept in totals and dashboard views)
        audit.log("train.csv", df.at[idx, 'Record_ID'], "Region",
                  "(blank)", "Unassigned", "T5",
                  "Blank region unassigned - kept in totals and dashboard views")
        df.at[idx, 'Region'] = "Unassigned"

    DQ['T5_blank_resolved'] = resolved_from_match
    DQ['T5_blank_unassigned'] = int(blank_mask.sum()) - resolved_from_match

    # Now normalise all non-blank, non-canonical values
    unique_vals = [v for v in df['Region'].dropna().unique() if v not in CANONICAL_REGIONS]
    val_map = {}
    unknown = []
    for uv in unique_vals:
        mapped = _normalise_region(uv)
        if mapped is None:
            unknown.append(uv)
        else:
            val_map[uv] = mapped

    if unknown:
        raise RuntimeError(f"T5: {len(unknown)} unknown Region values not in map: {set(unknown)}")

    changed = 0
    needs_map = df['Region'].isin(val_map.keys())
    for idx in df.index[needs_map]:
        r = df.at[idx, 'Region']
        mapped = val_map[r]
        if mapped != r:
            audit.log("train.csv", df.at[idx, 'Record_ID'], "Region",
                      r, mapped, "T5", "Region label normalised")
            changed += 1
        df.at[idx, 'Region'] = mapped

    DQ['T5_normalised'] = changed
    return df


def _normalise_channel(raw) -> str | None:
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    if s in CANONICAL_CHANNELS:
        return s
    normalised = re.sub(r'\s+', ' ', s.casefold()).strip()
    return CHANNEL_MAP.get(normalised)


def apply_t6_channel(df: pd.DataFrame) -> pd.DataFrame:
    """T6: Normalise Sales_Channel."""
    df = df.copy()
    unique_vals = [v for v in df['Sales_Channel'].dropna().unique() if v not in CANONICAL_CHANNELS]
    val_map = {}
    unknown = []
    for uv in unique_vals:
        mapped = _normalise_channel(uv)
        if mapped is None:
            if uv.strip():
                unknown.append(uv)
        else:
            val_map[uv] = mapped

    if unknown:
        raise RuntimeError(f"T6: {len(unknown)} unknown Sales_Channel values: {set(unknown)}")

    changed = 0
    needs_map = df['Sales_Channel'].isin(val_map.keys())
    for idx in df.index[needs_map]:
        raw = df.at[idx, 'Sales_Channel']
        mapped = val_map[raw]
        if mapped != raw.strip():
            audit.log("train.csv", df.at[idx, 'Record_ID'], "Sales_Channel",
                      raw, mapped, "T6", "Sales_Channel label normalised")
            changed += 1
        df.at[idx, 'Sales_Channel'] = mapped

    DQ['T6_normalised'] = changed
    DQ['T6_before_count'] = changed
    return df


def apply_t7_product_family(df: pd.DataFrame, sku_master: pd.DataFrame) -> pd.DataFrame:
    """T7: Fill blank Product_Family from SKU prefix; validate against sku_master."""
    df = df.copy()
    sku_to_family = dict(zip(sku_master['SKU_ID'], sku_master['Product_Family']))
    sku_to_bu = dict(zip(sku_master['SKU_ID'], sku_master['Business_Unit']))

    blank_pf = df['Product_Family'].isna() | (df['Product_Family'].str.strip() == '')
    DQ['T7_blank_product_family'] = int(blank_pf.sum())
    filled = 0

    for idx in df.index[blank_pf]:
        sku = df.at[idx, 'SKU_ID']
        fam = sku_to_family.get(str(sku).strip())
        if fam is None:
            prefix = str(sku).strip()[:3].upper()
            fam = PREFIX_TO_FAMILY.get(prefix)
        if fam:
            audit.log("train.csv", df.at[idx, 'Record_ID'], "Product_Family",
                      "(blank)", fam, "T7", "Product_Family derived from SKU_ID")
            df.at[idx, 'Product_Family'] = fam
            filled += 1

    DQ['T7_product_family_filled'] = filled

    # Fill/validate Business_Unit from sku_master (fast vectorized check)
    expected_bu_series = df['SKU_ID'].astype(str).str.strip().map(sku_to_bu)
    blank_bu = df['Business_Unit'].isna() | (df['Business_Unit'].str.strip() == '')
    mismatch_mask = expected_bu_series.notna() & (blank_bu | (df['Business_Unit'] != expected_bu_series))

    for idx in df.index[mismatch_mask]:
        expected_bu = expected_bu_series[idx]
        if not blank_bu[idx]:
            audit.log("train.csv", df.at[idx, 'Record_ID'], "Business_Unit",
                      df.at[idx, 'Business_Unit'], expected_bu, "T7",
                      "Business_Unit corrected to match sku_master (R8)")
        else:
            audit.log("train.csv", df.at[idx, 'Record_ID'], "Business_Unit",
                      "(blank)", expected_bu, "T7", "Business_Unit filled from sku_master (R8)")
        df.at[idx, 'Business_Unit'] = expected_bu

    return df


def _normalise_sku(raw: str, canonical_set: set) -> str | None:
    """Normalise a raw SKU_ID to canonical form [PREFIX]-[NNN]."""
    s = str(raw).strip().upper().replace('_', '-')
    m = re.match(r'^([A-Z]{3})(\d+)$', s)
    if m:
        s = f"{m.group(1)}-{m.group(2)}"
    m2 = re.match(r'^([A-Z]{3})-(\d+)$', s)
    if m2:
        s = f"{m2.group(1)}-{m2.group(2).zfill(3)}"
    if s in canonical_set:
        return s
    return None


def apply_t8_sku_id(df: pd.DataFrame, sku_master: pd.DataFrame) -> pd.DataFrame:
    """T8: Normalise SKU_ID to canonical form; fail if unresolvable."""
    df = df.copy()
    canonical_set = set(sku_master['SKU_ID'].str.strip())
    bad_mask = ~df['SKU_ID'].isin(canonical_set)
    DQ['T8_bad_sku_before'] = int(bad_mask.sum())

    # Fast mapping for unique non-canonical values
    unique_bad = df.loc[bad_mask, 'SKU_ID'].unique()
    val_map = {}
    failures = []
    for uv in unique_bad:
        norm = _normalise_sku(uv, canonical_set)
        if norm is None:
            failures.append(uv)
        else:
            val_map[uv] = norm

    if failures:
        raise RuntimeError(f"T8: {len(failures)} SKU_IDs could not be mapped to sku_master: {set(failures)}")

    changed = 0
    for idx in df.index[bad_mask]:
        raw = df.at[idx, 'SKU_ID']
        normalised = val_map[raw]
        audit.log("train.csv", df.at[idx, 'Record_ID'], "SKU_ID",
                  raw, normalised, "T8", "SKU_ID normalised to canonical form")
        df.at[idx, 'SKU_ID'] = normalised
        changed += 1

    DQ['T8_normalised'] = changed
    return df


def apply_t9_sku_description(df: pd.DataFrame, sku_master: pd.DataFrame) -> pd.DataFrame:
    """T9: Fill blank SKU_Description from sku_master."""
    df = df.copy()
    sku_to_desc = dict(zip(sku_master['SKU_ID'], sku_master['SKU_Description']))
    blank = df['SKU_Description'].isna() | (df['SKU_Description'].str.strip() == '')
    DQ['T9_blank_desc'] = int(blank.sum())
    filled = 0
    for idx in df.index[blank]:
        desc = sku_to_desc.get(str(df.at[idx, 'SKU_ID']).strip())
        if desc:
            audit.log("train.csv", df.at[idx, 'Record_ID'], "SKU_Description",
                      "(blank)", desc, "T9", "SKU_Description filled from sku_master")
            df.at[idx, 'SKU_Description'] = desc
            filled += 1
    DQ['T9_filled'] = filled
    return df


import difflib

CANON_COMPACT_MAP = {re.sub(r'[^a-z0-9]', '', c.lower()): c for c in CANONICAL_CUSTOMERS}


def _normalise_customer(raw) -> str | None:
    if pd.isna(raw):
        return None
    s = str(raw).strip()
    s = re.sub(r'\s+Inc\.?$', '', s, flags=re.IGNORECASE).strip()
    s = re.sub(r'\s+', ' ', s)

    # 1. Direct casefold match
    key = s.casefold()
    if key in CUSTOMER_MAP:
        return CUSTOMER_MAP[key]

    # 2. Compact match (handles missing spaces e.g. "VantaraAuto")
    compact = re.sub(r'[^a-z0-9]', '', s.lower())
    if compact in CANON_COMPACT_MAP:
        return CANON_COMPACT_MAP[compact]

    # 3. Fuzzy matching for typos (Rule 7, edit distance / similarity >= 0.70)
    close = difflib.get_close_matches(s, CANONICAL_CUSTOMERS, n=1, cutoff=0.70)
    if close:
        return close[0]

    close_compact = difflib.get_close_matches(compact, list(CANON_COMPACT_MAP.keys()), n=1, cutoff=0.70)
    if close_compact:
        return CANON_COMPACT_MAP[close_compact[0]]

    return None


def apply_t10_customer(df: pd.DataFrame) -> pd.DataFrame:
    """T10: Normalise Customer_Name."""
    df = df.copy()
    unique_custs = df['Customer_Name'].dropna().unique()
    cust_map = {}
    unknown = []
    for uc in unique_custs:
        mapped = _normalise_customer(uc)
        if mapped is None:
            if uc.strip():
                unknown.append(uc)
        else:
            cust_map[uc] = mapped

    if unknown:
        raise RuntimeError(f"T10: {len(unknown)} unknown Customer_Name values: {set(unknown)}")

    # Only loop over rows where Customer_Name actually changes
    needs_change = df['Customer_Name'].map(lambda c: cust_map.get(c) != c if pd.notna(c) else False)
    changed = 0
    for idx in df.index[needs_change]:
        raw = df.at[idx, 'Customer_Name']
        mapped = cust_map[raw]
        audit.log("train.csv", df.at[idx, 'Record_ID'], "Customer_Name",
                  raw, mapped, "T10", "Customer_Name normalised")
        df.at[idx, 'Customer_Name'] = mapped
        changed += 1

    DQ['T10_normalised'] = changed
    DQ['T10_before_count'] = changed
    return df


def apply_t11_units_sold(df: pd.DataFrame, stockouts: pd.DataFrame) -> pd.DataFrame:
    """T11 & A2: Clean Units_Sold: blanks, dashes, commas, negatives, and proportional stockout allocation."""
    df = df.copy()
    us = df['Units_Sold'].copy()

    # Pre-allocate stockouts for blank filling
    so_alloc = allocate_stockout_metric(df, stockouts, 'Shipped_Units', target_demand_col='Units_Sold')

    blank_count = 0
    dash_count = 0
    comma_count = 0
    neg_count = 0
    null_from_so = 0
    reversals_dropped = 0
    stockout_mismatches_fixed = 0
    a2_record_ids = set()  # rows whose stockout allocation used more than one customer

    new_us = []
    for idx, raw in zip(df.index, us):
        rid = df.at[idx, 'Record_ID']

        # Blank
        if pd.isna(raw) or str(raw).strip() == '':
            blank_count += 1
            so_val = so_alloc.at[idx, 'allocated_Shipped_Units']
            if pd.notna(so_val):
                n_cust = so_alloc.at[idx, 'n_cust_matched']
                share = so_alloc.at[idx, 'share_Shipped_Units']
                tot = so_alloc.at[idx, 'total_Shipped_Units']
                rule_id = "A2" if n_cust > 1 else "T11"
                reason = (f"Blank Units_Sold filled from allocated Shipped_Units (share={share:.4f} of {int(tot)})"
                          if n_cust > 1 else "Blank Units_Sold filled from inventory_stockouts Shipped_Units")
                audit.log("train.csv", rid, "Units_Sold",
                          "(blank)", int(so_val), rule_id, reason)
                if n_cust > 1:
                    a2_record_ids.add(rid)
                new_us.append(float(so_val))
                null_from_so += 1
            else:
                new_us.append(np.nan)
            continue

        s = str(raw).strip()

        # Dash (placeholder for not-reported)
        if s == '-':
            dash_count += 1
            audit.log("train.csv", rid, "Units_Sold",
                      s, "NULL", "T11", "Dash '-' treated as NULL (not reported, R3)")
            new_us.append(np.nan)
            continue

        # Remove thousands separator commas
        if ',' in s:
            comma_count += 1
            s_cleaned = s.replace(',', '')
            try:
                val = float(s_cleaned)
                if val != int(val):
                    raise ValueError("not integer")
                audit.log("train.csv", rid, "Units_Sold",
                          s, int(val), "T11", "Comma-formatted number parsed")
                s = str(int(val))
            except (ValueError, TypeError):
                raise RuntimeError(f"T11: Comma-formatted Units_Sold not integer: {raw}")

        # Try numeric
        try:
            val = float(s)
        except ValueError:
            raise RuntimeError(f"T11: Cannot parse Units_Sold: {raw!r} at Record_ID {rid}")

        new_us.append(val)

    df['Units_Sold'] = new_us

    # Identify positive rows by Date-SKU-Region for reversal matching
    pos_mask = df['Units_Sold'].notna() & (df['Units_Sold'] > 0)
    pos_keys = set(zip(df.loc[pos_mask, 'Date'], df.loc[pos_mask, 'SKU_ID'], df.loc[pos_mask, 'Region']))

    # Handle negative values
    drop_indices = []
    neg_mask = df['Units_Sold'].notna() & (df['Units_Sold'] < 0)
    neg_count = int(neg_mask.sum())
    for idx in df.index[neg_mask]:
        rid = df.at[idx, 'Record_ID']
        val = df.at[idx, 'Units_Sold']
        key = (df.at[idx, 'Date'], df.at[idx, 'SKU_ID'], df.at[idx, 'Region'])
        if key in pos_keys:
            audit.log("train.csv", rid, "Units_Sold",
                      val, "DROPPED", "T11",
                      f"Negative Units_Sold dropped as transaction reversal; Date={df.at[idx, 'Date']}; SKU_ID={df.at[idx, 'SKU_ID']}; Region={df.at[idx, 'Region']}")
            drop_indices.append(idx)
            reversals_dropped += 1
        else:
            abs_val = abs(val)
            audit.log("train.csv", rid, "Units_Sold",
                      val, abs_val, "T11",
                      "Negative Units_Sold sign-entry error; absolute value applied")
            df.at[idx, 'Units_Sold'] = abs_val

    if drop_indices:
        df = df.drop(index=drop_indices)

    # Where train and stockout share a key, Units_Sold must equal Shipped_Units (proportional allocation if multiple customers)
    matched_so = 0
    df['_us_temp'] = df['Units_Sold']
    so_alloc_aligned = allocate_stockout_metric(df, stockouts, 'Shipped_Units', target_demand_col='_us_temp')
    has_so_match = so_alloc_aligned['allocated_Shipped_Units'].notna()

    for idx in df.index[has_so_match]:
        matched_so += 1
        so_shipped = so_alloc_aligned.at[idx, 'allocated_Shipped_Units']
        current_us = df.at[idx, 'Units_Sold']
        n_cust = so_alloc_aligned.at[idx, 'n_cust_matched']
        share = so_alloc_aligned.at[idx, 'share_Shipped_Units']
        tot = so_alloc_aligned.at[idx, 'total_Shipped_Units']

        if pd.notna(current_us) and current_us != so_shipped:
            rule_id = "A2" if n_cust > 1 else "T11"
            reason = (f"Units_Sold aligned with allocated inventory_stockouts Shipped_Units (share={share:.4f} of total {int(tot)})"
                      if n_cust > 1 else "Units_Sold aligned with inventory_stockouts Shipped_Units")
            audit.log("train.csv", df.at[idx, 'Record_ID'], "Units_Sold",
                      current_us, int(so_shipped), rule_id, reason)
            df.at[idx, 'Units_Sold'] = so_shipped
            stockout_mismatches_fixed += 1
            if n_cust > 1:
                a2_record_ids.add(df.at[idx, 'Record_ID'])

    df = df.drop(columns=['_us_temp'], errors='ignore')

    df['Units_Sold'] = pd.array(df['Units_Sold'], dtype='Float64')

    DQ['T11_blank'] = blank_count
    DQ['T11_dash'] = dash_count
    DQ['T11_comma'] = comma_count
    DQ['T11_negative'] = neg_count
    DQ['T11_reversals_dropped'] = reversals_dropped
    DQ['T11_null_from_stockout'] = null_from_so
    DQ['T11_stockout_mismatches_fixed'] = stockout_mismatches_fixed
    DQ['T11_stockout_matched_rows'] = matched_so
    DQ['T11_large_values'] = int((df['Units_Sold'] > UNITS_LARGE_THRESHOLD).sum())
    DQ['A2_allocated_stockouts'] = len(a2_record_ids)

    return df


def apply_t12_prices(df: pd.DataFrame) -> pd.DataFrame:
    """T12: One rule for all price anomalies. Replace out-of-band and zero prices with SKU-year median."""
    df = df.copy()
    original_price = df['Unit_Price_USD'].apply(
        lambda x: float(str(x).strip().replace(',', '')) if pd.notna(x) else np.nan
    )
    DQ['T12_before_count'] = int(((original_price == 0) | original_price.isna()).sum())

    def _to_float(x):
        try:
            return float(str(x).strip().replace(',', ''))
        except (ValueError, TypeError):
            return np.nan

    df['Unit_Price_USD'] = df['Unit_Price_USD'].apply(_to_float)
    df['_year'] = df['Date'].str[:4]
    units_num = pd.to_numeric(df['Units_Sold'], errors='coerce')

    # Build reference prices: median of non-zero prices per SKU and calendar year across all regions
    ref_price_map = {}
    valid_prices = df[df['Unit_Price_USD'] > 0]
    for (sku, yr), grp in valid_prices.groupby(['SKU_ID', '_year']):
        ref_price_map[(sku, yr)] = grp['Unit_Price_USD'].median()

    sku_yr_tuples = list(zip(df['SKU_ID'], df['_year']))
    ref_prices = [ref_price_map.get(k) for k in sku_yr_tuples]
    for i, ref in enumerate(ref_prices):
        if ref is None:
            sku, yr = sku_yr_tuples[i]
            for y_off in [1, -1, 2, -2]:
                try:
                    r = ref_price_map.get((sku, str(int(yr) + y_off)))
                    if r:
                        ref_prices[i] = r
                        break
                except ValueError:
                    pass

    df['_ref_price'] = ref_prices

    p = df['Unit_Price_USD']
    ref = df['_ref_price']
    has_ref = ref.notna() & (ref > 0)
    has_p = p.notna() & (p > 0)

    # EV-027 surcharge window: Jul-Dec 2022 (6% surcharge is valid)
    is_surcharge_month = (df['Date'] >= '2022-07') & (df['Date'] <= '2022-12') & (df['SKU_ID'].str.startswith('HYD'))

    ratio = p / ref
    zero_price_mask = (p.isna() | (p == 0)) & has_ref
    out_of_band_mask = has_p & has_ref & ((ratio < 0.5) | (ratio > 2.0))

    repaired_count = 0
    zero_repaired = 0
    for idx in df.index[zero_price_mask]:
        rid = df.at[idx, 'Record_ID']
        price = df.at[idx, 'Unit_Price_USD']
        ref_val = df.at[idx, '_ref_price']
        audit.log("train.csv", rid, "Unit_Price_USD",
                  price, ref_val, "T12",
                  "Zero or missing price replaced with SKU-year reference price")
        df.at[idx, 'Unit_Price_USD'] = ref_val
        repaired_count += 1
        zero_repaired += 1

    band_counts = {'T12_band_85x': 0, 'T12_band_100x': 0, 'T12_band_5_20x': 0, 'T12_band_other': 0}
    for idx in df.index[out_of_band_mask]:
        rid = df.at[idx, 'Record_ID']
        price = df.at[idx, 'Unit_Price_USD']
        ref_val = df.at[idx, '_ref_price']
        r_val = price / ref_val
        audit.log("train.csv", rid, "Unit_Price_USD",
                  price, ref_val, "T12",
                  f"Price out of band (x{r_val:.2f} vs reference); replaced with SKU-year reference price")
        df.at[idx, 'Unit_Price_USD'] = ref_val
        repaired_count += 1
        if 80 <= r_val < 95:
            band_counts['T12_band_85x'] += 1
        elif 95 <= r_val <= 120:
            band_counts['T12_band_100x'] += 1
        elif 5 <= r_val <= 20:
            band_counts['T12_band_5_20x'] += 1
        else:
            band_counts['T12_band_other'] += 1

    # Check R4: within each SKU and year, price changes only on 1 January or in surcharge months
    r4_violations = 0
    non_surcharge = ~is_surcharge_month
    p_after = df['Unit_Price_USD']
    ref_after = df['_ref_price']
    diff_mask = non_surcharge & has_ref & (p_after.notna()) & (p_after != ref_after)
    for idx in df.index[diff_mask]:
        rid = df.at[idx, 'Record_ID']
        price = df.at[idx, 'Unit_Price_USD']
        ref_val = df.at[idx, '_ref_price']
        audit.log("train.csv", rid, "Unit_Price_USD",
                  price, ref_val, "R4",
                  "Intra-year price deviation outside surcharge period corrected to annual reference price (R4)")
        df.at[idx, 'Unit_Price_USD'] = ref_val
        r4_violations += 1

    df = df.drop(columns=['_year', '_ref_price'])
    DQ['T12_repaired_prices'] = repaired_count
    DQ['T12_zero_repaired'] = zero_repaired
    DQ['T12_outofband_repaired'] = repaired_count - zero_repaired
    DQ.update(band_counts)
    DQ['T12_r4_violations_corrected'] = r4_violations
    DQ['T12_after_count'] = int(((df['Unit_Price_USD'].isna()) | (df['Unit_Price_USD'] == 0)).sum())

    return df


def apply_t13_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """T13: Recompute Revenue_USD = Units_Sold x Unit_Price_USD."""
    df = df.copy()

    def _to_float(x):
        try:
            return float(str(x).strip().replace(',', ''))
        except (ValueError, TypeError):
            return np.nan

    df['Revenue_USD'] = df['Revenue_USD'].apply(_to_float)
    orig_rev = df['Revenue_USD'].copy()

    has_both = df['Units_Sold'].notna() & (df['Unit_Price_USD'].notna()) & (df['Unit_Price_USD'] > 0)
    expected_rev = df.loc[has_both, 'Units_Sold'] * df.loc[has_both, 'Unit_Price_USD']
    discrepancy = (orig_rev[has_both] - expected_rev).abs() > 1.0

    DQ['T13_blank_revenue'] = int(df['Revenue_USD'].isna().sum())
    DQ['T13_discrepancy_before'] = int(discrepancy.sum())

    for idx in df.index[has_both][discrepancy[has_both].values]:
        rid = df.at[idx, 'Record_ID']
        raw_rev = orig_rev[idx]
        new_rev = expected_rev[idx]
        audit.log("train.csv", rid, "Revenue_USD",
                  raw_rev, round(new_rev, 2), "T13",
                  "Revenue_USD recomputed as Units_Sold x Unit_Price_USD (within $1 tolerance failed)")

    df.loc[has_both, 'Revenue_USD'] = expected_rev
    DQ['T13_recomputed'] = int(discrepancy.sum())
    remaining = (df.loc[has_both, 'Revenue_USD'] - expected_rev).abs() > 1.0
    DQ['T13_after_count'] = int(remaining.sum())
    return df


def apply_t15_zero_demand_flag(df: pd.DataFrame) -> pd.DataFrame:
    """T15: Flag zero-demand rows vs missing rows (R3). No edit needed."""
    zero_rows = (df['Units_Sold'] == 0).sum()
    null_rows = df['Units_Sold'].isna().sum()
    DQ['T15_zero_demand_rows'] = int(zero_rows)
    DQ['T15_null_demand_rows'] = int(null_rows)
    DQ['T15_before_count'] = int(zero_rows + null_rows)
    DQ['T15_after_count'] = int(zero_rows + null_rows)
    return df


def apply_a3_high_volume_replacements(df: pd.DataFrame) -> pd.DataFrame:
    """Replace suspected high-volume entries above 20 times the SKU-year median."""
    df = df.copy()
    units = pd.to_numeric(df['Units_Sold'], errors='coerce')
    year = df['Date'].str[:4]
    medians = (
        pd.DataFrame({'SKU_ID': df['SKU_ID'], 'Year': year, 'Units': units})
        .query('Units > 0')
        .groupby(['SKU_ID', 'Year'])['Units']
        .median()
    )

    replaced = 0
    df['_A3_Original_Units'] = np.nan
    df['_A3_Ratio'] = np.nan
    for idx in df.index:
        value = units.at[idx] if idx in units.index else np.nan
        if pd.isna(value) or value <= 0:
            continue
        median = medians.get((df.at[idx, 'SKU_ID'], year.at[idx]))
        if pd.isna(median) or median <= 0:
            continue
        ratio = float(value) / float(median)
        if ratio > 20:
            new_value = int(round(float(median)))
            df.at[idx, 'Units_Sold'] = new_value
            df.at[idx, '_A3_Original_Units'] = float(value)
            df.at[idx, '_A3_Ratio'] = round(ratio, 6)
            audit.log(
                "train.csv",
                df.at[idx, 'Record_ID'],
                "Units_Sold",
                value,
                new_value,
                "A3",
                f"suspected entry error: ratio {ratio:.6f} against SKU-year median",
            )
            replaced += 1

    DQ['A3_replaced_rows'] = replaced
    DQ['A3_before_count'] = replaced
    DQ['A3_after_count'] = 0
    return df


def apply_r5_lifecycle(df: pd.DataFrame, sku_master: pd.DataFrame) -> pd.DataFrame:
    """R5: Flag (not remove) sales after EOL_Date or before Launch_Date."""
    df = df.copy()
    eol_map = {}
    launch_map = {}
    for _, row in sku_master.iterrows():
        sku = row['SKU_ID']
        if pd.notna(row.get('EOL_Date')) and str(row['EOL_Date']).strip():
            eol_map[sku] = str(row['EOL_Date'])[:7]
        if pd.notna(row.get('Launch_Date')) and str(row['Launch_Date']).strip():
            launch_map[sku] = str(row['Launch_Date'])[:7]

    r5_eol_violations = 0
    r5_launch_violations = 0
    relevant_skus = set(eol_map.keys()) | set(launch_map.keys())
    check_mask = df['SKU_ID'].isin(relevant_skus) & (df['Units_Sold'] > 0)
    for idx in df.index[check_mask]:
        sku = df.at[idx, 'SKU_ID']
        date = df.at[idx, 'Date']
        eol = eol_map.get(sku)
        launch = launch_map.get(sku)
        if eol and date > eol and df.at[idx, 'Units_Sold'] > 0:
            r5_eol_violations += 1
            audit.log("train.csv", df.at[idx, 'Record_ID'], "Units_Sold",
                      df.at[idx, 'Units_Sold'], "R5_VIOLATION",
                      "R5", f"Sale after EOL_Date={eol} for SKU {sku}")
        if launch and date < launch and df.at[idx, 'Units_Sold'] > 0:
            r5_launch_violations += 1
            audit.log("train.csv", df.at[idx, 'Record_ID'], "Units_Sold",
                      df.at[idx, 'Units_Sold'], "R5_VIOLATION",
                      "R5", f"Sale before Launch_Date={launch} for SKU {sku}")

    DQ['R5_eol_violations'] = r5_eol_violations
    DQ['R5_launch_violations'] = r5_launch_violations
    return df


def apply_t4b_dsr_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """
    T4b/R9: One clean row per Date-SKU-Region-Customer.
    Sales_Channel is dropped from the deduplication key.
    Keep the row with the latest Load_Timestamp (R2).
    Log every dropped row with rule ID T4b, the dropped Record_ID, the kept Record_ID,
    and whether the channels differed.
    """
    df = df.copy()
    before = len(df)
    DQ['T4b_before'] = before
    df['_load_ts'] = pd.to_datetime(df['Load_Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    key_cols = ['Date', 'SKU_ID', 'Region', 'Customer_Name']

    # Sort descending by load timestamp, then Record_ID descending for determinism
    df = df.sort_values(['_load_ts', 'Record_ID'], ascending=[False, False])
    dup_mask = df.duplicated(key_cols, keep=False)

    DQ['T4b_channels_differed'] = 0

    if dup_mask.any():
        dup_df = df[dup_mask]
        DQ['T4b_dsr_dups'] = int(df.duplicated(key_cols, keep='first').sum())

        for _, grp in dup_df.groupby(key_cols, sort=False):
            kept_row = grp.iloc[0]
            kept_rid = kept_row['Record_ID']
            kept_chan = kept_row.get('Sales_Channel', '')
            for i in range(1, len(grp)):
                dropped_row = grp.iloc[i]
                dropped_rid = dropped_row['Record_ID']
                dropped_chan = dropped_row.get('Sales_Channel', '')
                channels_differed = (dropped_chan != kept_chan)
                if channels_differed:
                    DQ['T4b_channels_differed'] += 1
                reason = (f"Duplicate load for Date-SKU-Region-Customer (T4b/R9) - "
                          f"dropped {dropped_rid}, kept {kept_rid} (channels differed: {channels_differed})")
                audit.log("train.csv", dropped_rid, "Date-SKU-Region-Customer",
                          f"channel={dropped_chan}", f"kept_{kept_rid}", "T4b", reason)

        df = df.drop_duplicates(key_cols, keep='first')
    else:
        DQ['T4b_dsr_dups'] = 0

    df = df.drop(columns=['_load_ts']).sort_index()
    DQ['T4b_after'] = len(df)
    return df


def clean_train(raw: pd.DataFrame, sku_master: pd.DataFrame,
                stockouts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Run all cleaning rules T1-T15 + R5 on raw train data.
    Returns (clean_df, dq_counts_dict).
    """
    global DQ
    DQ = {}
    DQ['raw_rows'] = len(raw)
    DQ['T5_blank_regions'] = int(raw['Region'].isna().sum() if 'Region' in raw else 0)
    DQ['T7_blank_product_family'] = int(raw['Product_Family'].isna().sum() if 'Product_Family' in raw else 0)
    sku_text = raw['SKU_ID'].astype(str) if 'SKU_ID' in raw else pd.Series(dtype=str)
    DQ['T8_bad_sku_before'] = int((~sku_text.str.fullmatch(r'[A-Z]{3}-[0-9]{3}')).sum())
    DQ['T9_blank_desc'] = int(raw['SKU_Description'].isna().sum() if 'SKU_Description' in raw else 0)
    units_raw = raw['Units_Sold'].copy() if 'Units_Sold' in raw else pd.Series(dtype=object)
    unit_text = units_raw.astype(str).str.strip()
    DQ['T11_blank'] = int(units_raw.isna().sum())
    DQ['T11_dash'] = int(unit_text.eq('-').sum())
    DQ['T11_comma'] = int(unit_text.str.contains(',', regex=False).sum())
    numeric_units = pd.to_numeric(units_raw.astype(str).str.replace(',', '', regex=False), errors='coerce')
    DQ['T11_negative'] = int((numeric_units < 0).sum())
    if 'Revenue_USD' in raw and 'Units_Sold' in raw and 'Unit_Price_USD' in raw:
        revenue = pd.to_numeric(raw['Revenue_USD'], errors='coerce')
        units = pd.to_numeric(units_raw.astype(str).str.replace(',', '', regex=False), errors='coerce')
        prices = pd.to_numeric(raw['Unit_Price_USD'], errors='coerce')
        mismatch = (revenue - units * prices).abs() > 1
        DQ['T13_blank_revenue'] = int(raw['Revenue_USD'].isna().sum())
        DQ['T13_discrepancy_before'] = int(mismatch.sum())

    df = raw.copy()
    df = apply_t1_date_parsing(df)
    df = apply_t3_typo_year(df)
    df = apply_t2_year_month(df)
    df = apply_t4_dedup_record_id(df)
    df = apply_t5_region(df)
    df = apply_t6_channel(df)
    df = apply_t8_sku_id(df, sku_master)   # must run before T7 so SKU keys are clean
    df = apply_t7_product_family(df, sku_master)
    df = apply_t9_sku_description(df, sku_master)
    df = apply_t10_customer(df)
    df = apply_t11_units_sold(df, stockouts)
    df = apply_a3_high_volume_replacements(df)
    df = apply_t12_prices(df)
    df = apply_t13_revenue(df)
    df = apply_t15_zero_demand_flag(df)
    df = apply_r5_lifecycle(df, sku_master)
    df = apply_t4b_dsr_dedup(df)

    # Ensure no duplicate Date, SKU_ID, Region, Customer_Name rows remain
    dups = df[df.duplicated(['Date', 'SKU_ID', 'Region', 'Customer_Name'], keep=False)]
    if not dups.empty:
        first_five = dups[['Date', 'SKU_ID', 'Region', 'Customer_Name']].drop_duplicates().head(5).to_dict(orient='records')
        raise RuntimeError(f"clean_train: duplicate Date, SKU_ID, Region, Customer_Name keys found: {first_five}")

    DQ['clean_rows'] = len(df)
    return df, DQ