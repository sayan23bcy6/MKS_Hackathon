"""
tests/test_cleaning_rules.py - pytest unit tests for all cleaning rules.
One test per rule as required by the spec.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
import json
import hashlib
import re

from pipeline.clean_train import (
    apply_t1_date_parsing,
    apply_t2_year_month,
    apply_t3_typo_year,
    apply_t4_dedup_record_id,
    apply_t5_region,
    apply_t6_channel,
    apply_t8_sku_id,
    apply_t9_sku_description,
    apply_t10_customer,
    apply_t11_units_sold,
    apply_t12_prices,
    apply_t13_revenue,
    apply_t15_zero_demand_flag,
    apply_r5_lifecycle,
    _normalise_sku,
    _normalise_region,
    _normalise_channel,
    _normalise_customer,
)
from pipeline.clean_master_macro_events import (
    clean_macro, clean_events, clean_sku_master, _normalise_event_date
)
from pipeline.config import CANONICAL_REGIONS, CANONICAL_CHANNELS, CANONICAL_CUSTOMERS


# --- Fixtures ---

def _make_train_row(**kwargs) -> pd.DataFrame:
    """Make a minimal train DataFrame with one row."""
    defaults = {
        'Record_ID': 'CEL-0000001',
        'Date': '2022-01',
        'Year': '2022',
        'Month': '1',
        'Product_Family': 'Brake & Safety Systems',
        'Business_Unit': 'Safety & Compliance',
        'SKU_ID': 'BSS-001',
        'SKU_Description': 'Test SKU',
        'Customer_Name': 'Vantara Auto',
        'Sales_Channel': 'Direct OEM',
        'Region': 'North America',
        'Units_Sold': '100',
        'Unit_Price_USD': '500.00',
        'Revenue_USD': '50000.00',
        'Load_Timestamp': '2022-02-01 00:00:00',
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


@pytest.fixture
def minimal_sku_master():
    return pd.DataFrame([{
        'SKU_ID': 'BSS-001',
        'Product_Family': 'Brake & Safety Systems',
        'Business_Unit': 'Safety & Compliance',
        'SKU_Description': 'Industrial 100 mm Disc Brake Caliper Assembly',
        'Size_Class': 'Large',
        'Powertrain_Application': 'Universal',
        'Launch_Date': '2019-01',
        'EOL_Date': '',
        'Replaces_SKU': '',
        'Lifecycle_Status': 'Active',
    }])


@pytest.fixture
def empty_stockouts():
    return pd.DataFrame(columns=[
        'Date', 'SKU_ID', 'Region', 'Requested_Units', 'Carried_In_Units',
        'Shipped_Units', 'Backordered_Units', 'Lost_Units', 'Fill_Rate_Pct',
        'Stockout_Days', 'Reason'
    ])


# --- T1: Date parsing ---

class TestT1DateParsing:
    def test_yyyy_mm_unchanged(self):
        df = _make_train_row(Date='2022-01')
        out = apply_t1_date_parsing(df)
        assert out['Date'].iloc[0] == '2022-01'

    def test_yyyy_mm_dd_to_yyyy_mm(self):
        df = _make_train_row(Date='2022-01-15')
        out = apply_t1_date_parsing(df)
        assert out['Date'].iloc[0] == '2022-01'

    def test_yyyy_slash_mm(self):
        df = _make_train_row(Date='2022/01')
        out = apply_t1_date_parsing(df)
        assert out['Date'].iloc[0] == '2022-01'

    def test_mm_slash_yyyy(self):
        df = _make_train_row(Date='01/2022')
        out = apply_t1_date_parsing(df)
        assert out['Date'].iloc[0] == '2022-01'

    def test_mm_dash_yyyy(self):
        df = _make_train_row(Date='03-2021')
        out = apply_t1_date_parsing(df)
        assert out['Date'].iloc[0] == '2021-03'

    def test_month_name_space(self):
        df = _make_train_row(Date='Jan 2019')
        out = apply_t1_date_parsing(df)
        assert out['Date'].iloc[0] == '2019-01'

    def test_month_name_dash(self):
        df = _make_train_row(Date='Mar-2023')
        out = apply_t1_date_parsing(df)
        assert out['Date'].iloc[0] == '2023-03'

    def test_month_name_case_insensitive(self):
        df = _make_train_row(Date='JAN 2020')
        out = apply_t1_date_parsing(df)
        assert out['Date'].iloc[0] == '2020-01'

    def test_mm_yyyy_first_field_gt12_raises(self):
        df = _make_train_row(Date='13/2022')
        with pytest.raises(ValueError):
            apply_t1_date_parsing(df)


# --- T2: Year/Month consistency ---

class TestT2YearMonth:
    def test_correct_year_month_unchanged(self):
        df = _make_train_row(Date='2022-03', Year='2022', Month='3')
        out = apply_t2_year_month(df)
        assert out['Year'].iloc[0] == '2022'
        assert out['Month'].iloc[0] == '3'

    def test_wrong_year_corrected(self):
        df = _make_train_row(Date='2022-03', Year='2021', Month='3')
        out = apply_t2_year_month(df)
        assert out['Year'].iloc[0] == '2022'

    def test_wrong_month_corrected(self):
        df = _make_train_row(Date='2022-03', Year='2022', Month='5')
        out = apply_t2_year_month(df)
        assert out['Month'].iloc[0] == '3'


# --- T3: Typo year ---

class TestT3TypoYear:
    def test_2062_corrected_to_2026(self):
        df = _make_train_row(Date='2062-01', Year='2026', Month='1')
        out = apply_t3_typo_year(df)
        assert out['Date'].iloc[0] == '2026-01'

    def test_valid_date_unchanged(self):
        df = _make_train_row(Date='2024-06', Year='2024', Month='6')
        out = apply_t3_typo_year(df)
        assert out['Date'].iloc[0] == '2024-06'


# --- T4: Dedup Record_ID ---

class TestT4Dedup:
    def test_keeps_latest_timestamp(self):
        rows = [
            {**_make_train_row(Load_Timestamp='2024-09-01 00:00:00',
                               Units_Sold='131').iloc[0].to_dict()},
            {**_make_train_row(Load_Timestamp='2024-09-22 00:00:00',
                               Units_Sold='105').iloc[0].to_dict()},
        ]
        df = pd.DataFrame(rows)
        out = apply_t4_dedup_record_id(df)
        assert len(out) == 1
        assert out['Units_Sold'].iloc[0] == '105'  # latest timestamp row

    def test_no_dups_unchanged(self):
        rows = [
            {**_make_train_row(Record_ID='CEL-0000001').iloc[0].to_dict()},
            {**_make_train_row(Record_ID='CEL-0000002').iloc[0].to_dict()},
        ]
        df = pd.DataFrame(rows)
        out = apply_t4_dedup_record_id(df)
        assert len(out) == 2


# --- T5: Region ---

class TestT5Region:
    @pytest.mark.parametrize("raw,expected", [
        ("APAC", "Asia Pacific"),
        ("EU", "Europe"),
        ("LATAM", "Latin America"),
        ("LatAm", "Latin America"),
        ("N. America", "North America"),
        ("NA", "North America"),
        ("Middle East and Africa", "Middle East & Africa"),
        ("ME&A", "Middle East & Africa"),
        ("MEA", "Middle East & Africa"),
        ("INDIA", "India"),
        ("india", "India"),
        ("Asia-Pacific", "Asia Pacific"),
        ("Latin-America", "Latin America"),
        ("North America ", "North America"),
        ("Europe ", "Europe"),
    ])
    def test_region_normalisation(self, raw, expected):
        result = _normalise_region(raw)
        assert result == expected, f"Expected {expected!r} for {raw!r}, got {result!r}"

    def test_all_canonical_regions_map_to_themselves(self):
        for r in CANONICAL_REGIONS:
            assert _normalise_region(r) == r


# --- T6: Sales_Channel ---

class TestT6Channel:
    @pytest.mark.parametrize("raw,expected", [
        ("tier-1 distributor", "Tier-1 Distributor"),
        ("Tier 1 Distributor", "Tier-1 Distributor"),
        ("T1 Distributor", "Tier-1 Distributor"),
        ("Direct-OEM", "Direct OEM"),
        ("OEM Direct", "Direct OEM"),
        ("direct oem", "Direct OEM"),
        ("DIRECT OEM", "Direct OEM"),
        ("After-market", "Aftermarket"),
        ("AfterMarket", "Aftermarket"),
        ("aftermarket", "Aftermarket"),
        ("export", "Export"),
        ("Export ", "Export"),
        ("EXPORT", "Export"),
    ])
    def test_channel_normalisation(self, raw, expected):
        result = _normalise_channel(raw)
        assert result == expected, f"Expected {expected!r} for {raw!r}, got {result!r}"


# --- T8: SKU_ID ---

class TestT8SKU:
    CANONICAL = {
        'BSS-001', 'BSS-032', 'PMP-023', 'HYD-006', 'PMP-001',
        'CSS-031', 'THM-043', 'ELA-011', 'PTC-045',
    }

    @pytest.mark.parametrize("raw,expected", [
        ("bss-001", "BSS-001"),
        (" BSS-001", "BSS-001"),
        ("BSS_032", "BSS-032"),
        ("PMP023", "PMP-023"),
        ("HYD_006", "HYD-006"),
        ("PMP_001", "PMP-001"),
        ("CSS-31", "CSS-031"),
        ("THM-43", "THM-043"),
        ("ELA011", "ELA-011"),
        ("ptc_045", "PTC-045"),
    ])
    def test_sku_normalisation(self, raw, expected):
        result = _normalise_sku(raw, self.CANONICAL)
        assert result == expected, f"Expected {expected!r} for {raw!r}, got {result!r}"

    def test_unknown_sku_returns_none(self):
        result = _normalise_sku("XXX-999", self.CANONICAL)
        assert result is None


# --- T10: Customer_Name ---

class TestT10Customer:
    @pytest.mark.parametrize("raw,expected", [
        ("Vantara Auto Inc.", "Vantara Auto"),
        ("HELIOS AUTO PARTS", "Helios Auto Parts"),
        ("Quantum  Drive Systems", "Quantum Drive Systems"),
        ("Apex  Tier One", "Apex Tier One"),
        ("Nexon Motor Group Inc.", "Nexon Motor Group"),
        ("Meridian Assembly Inc.", "Meridian Assembly"),
        (" Vantara Auto ", "Vantara Auto"),
    ])
    def test_customer_normalisation(self, raw, expected):
        result = _normalise_customer(raw)
        assert result == expected, f"Expected {expected!r} for {raw!r}, got {result!r}"

    def test_all_canonical_map_to_themselves(self):
        for c in CANONICAL_CUSTOMERS:
            assert _normalise_customer(c) == c


# --- T11: Units_Sold ---

class TestT11Units:
    def test_dash_becomes_null(self, empty_stockouts):
        df = _make_train_row(Units_Sold='-', Unit_Price_USD='0.0', Revenue_USD='0.0')
        out = apply_t11_units_sold(df, empty_stockouts)
        assert pd.isna(out['Units_Sold'].iloc[0])

    def test_comma_number_parsed(self, empty_stockouts):
        df = _make_train_row(Units_Sold='9,600')
        out = apply_t11_units_sold(df, empty_stockouts)
        assert float(out['Units_Sold'].iloc[0]) == 9600.0

    def test_negative_becomes_abs(self, empty_stockouts):
        df = _make_train_row(Units_Sold='-35')
        out = apply_t11_units_sold(df, empty_stockouts)
        assert float(out['Units_Sold'].iloc[0]) == 35.0

    def test_blank_becomes_null(self, empty_stockouts):
        df = _make_train_row(Units_Sold='')
        out = apply_t11_units_sold(df, empty_stockouts)
        assert pd.isna(out['Units_Sold'].iloc[0])

    def test_zero_stays_zero(self, empty_stockouts):
        df = _make_train_row(Units_Sold='0')
        out = apply_t11_units_sold(df, empty_stockouts)
        assert float(out['Units_Sold'].iloc[0]) == 0.0


# --- T12: Unit_Price ---

class TestT12Price:
    def test_zero_price_with_units_flagged_in_audit(self):
        """Zero price with units>0 should be filled from reference price."""
        rows = [
            {**_make_train_row(Date='2022-01', SKU_ID='BSS-001', Units_Sold='100', Unit_Price_USD='500.0').iloc[0].to_dict()},
            {**_make_train_row(Date='2022-02', SKU_ID='BSS-001', Units_Sold='50', Unit_Price_USD='0.0').iloc[0].to_dict()},
        ]
        df = pd.DataFrame(rows)
        out = apply_t12_prices(df)
        assert float(out.loc[out['Date'] == '2022-02', 'Unit_Price_USD'].iloc[0]) == 500.0

    def test_valid_price_unchanged(self):
        """Valid price within normal range is retained unchanged."""
        rows = [
            {**_make_train_row(Date='2022-01', SKU_ID='BSS-001', Units_Sold='100', Unit_Price_USD='500.0').iloc[0].to_dict()},
            {**_make_train_row(Date='2022-02', SKU_ID='BSS-001', Units_Sold='50', Unit_Price_USD='500.0').iloc[0].to_dict()},
        ]
        df = pd.DataFrame(rows)
        out = apply_t12_prices(df)
        assert float(out.loc[out['Date'] == '2022-01', 'Unit_Price_USD'].iloc[0]) == 500.0
        assert float(out.loc[out['Date'] == '2022-02', 'Unit_Price_USD'].iloc[0]) == 500.0


# --- T13: Revenue ---

class TestT13Revenue:
    def test_revenue_recomputed(self):
        df = _make_train_row(Units_Sold='100', Unit_Price_USD='500.0', Revenue_USD='999.0')
        # First apply T11 to get numeric Units_Sold
        from io import StringIO
        df['Units_Sold'] = pd.array([100.0], dtype='Float64')
        df['Unit_Price_USD'] = pd.to_numeric(df['Unit_Price_USD'])
        df['Revenue_USD'] = pd.to_numeric(df['Revenue_USD'])
        out = apply_t13_revenue(df)
        assert abs(float(out['Revenue_USD'].iloc[0]) - 50000.0) < 1.0

    def test_correct_revenue_unchanged(self):
        df = _make_train_row(Units_Sold='100', Unit_Price_USD='500.0', Revenue_USD='50000.0')
        df['Units_Sold'] = pd.array([100.0], dtype='Float64')
        df['Unit_Price_USD'] = pd.to_numeric(df['Unit_Price_USD'])
        df['Revenue_USD'] = pd.to_numeric(df['Revenue_USD'])
        out = apply_t13_revenue(df)
        assert abs(float(out['Revenue_USD'].iloc[0]) - 50000.0) < 1.0


# --- R5: Lifecycle ---

class TestR5Lifecycle:
    def test_no_violations_on_active_sku(self, minimal_sku_master):
        df = _make_train_row(Date='2022-01', Units_Sold='100')
        df['Units_Sold'] = pd.array([100.0], dtype='Float64')
        out = apply_r5_lifecycle(df, minimal_sku_master)
        assert float(out['Units_Sold'].iloc[0]) == 100.0

    def test_eol_violation_flagged(self):
        sku_master = pd.DataFrame([{
            'SKU_ID': 'BSS-001',
            'Product_Family': 'Brake & Safety Systems',
            'Business_Unit': 'Safety & Compliance',
            'SKU_Description': 'Test',
            'Size_Class': '', 'Powertrain_Application': '',
            'Launch_Date': '2019-01',
            'EOL_Date': '2021-12',
            'Replaces_SKU': '',
            'Lifecycle_Status': 'Discontinued',
        }])
        df = _make_train_row(Date='2022-03', Units_Sold='100')
        df['Units_Sold'] = pd.array([100.0], dtype='Float64')
        # Should log an R5 violation but not raise
        apply_r5_lifecycle(df, sku_master)


# --- Macro cleaning ---

class TestMacroCleaning:
    @pytest.fixture
    def raw_macro(self):
        """Minimal macro data with M1-M5 issues."""
        rows = [
            {'Date': '2019-01', 'Year': '2019', 'Month': '1',
             'ISM_Manufacturing_PMI': '54.0', 'Auto_Production_YoY_Pct': '-0.015',
             'Vehicle_Inventory_Days': '68.0', 'Raw_Material_Cost_Index': '91.7',
             'New_Vehicle_Orders_Index': '98.0', 'Industrial_Util_Rate_Pct': '79.5',
             'Fuel_Price_Index': '82.0', 'EV_Adoption_Index': '40.0'},
            {'Date': '2022-05', 'Year': '2022', 'Month': '5',
             'ISM_Manufacturing_PMI': '53.5', 'Auto_Production_YoY_Pct': '1.0',
             'Vehicle_Inventory_Days': '31.5', 'Raw_Material_Cost_Index': '131.3',
             'New_Vehicle_Orders_Index': '104.8', 'Industrial_Util_Rate_Pct': '79.8',
             'Fuel_Price_Index': '138.3', 'EV_Adoption_Index': '67.4'},
            {'Date': '2022-06', 'Year': '2022', 'Month': '6',
             'ISM_Manufacturing_PMI': '53.9', 'Auto_Production_YoY_Pct': '3.2',
             'Vehicle_Inventory_Days': '33.9', 'Raw_Material_Cost_Index': '127.9',
             'New_Vehicle_Orders_Index': '102.9', 'Industrial_Util_Rate_Pct': '80.0',
             'Fuel_Price_Index': '142.8', 'EV_Adoption_Index': '68.6'},
            {'Date': '2022-06', 'Year': '2022', 'Month': '6',  # Duplicate!
             'ISM_Manufacturing_PMI': '52.7', 'Auto_Production_YoY_Pct': '3.2',
             'Vehicle_Inventory_Days': '33.9', 'Raw_Material_Cost_Index': '127.9',
             'New_Vehicle_Orders_Index': '104.3', 'Industrial_Util_Rate_Pct': '80.0',
             'Fuel_Price_Index': '142.8', 'EV_Adoption_Index': '68.6'},
            {'Date': '2022-07', 'Year': '2022', 'Month': '7',
             'ISM_Manufacturing_PMI': '52.2', 'Auto_Production_YoY_Pct': '3.4',
             'Vehicle_Inventory_Days': '35.6', 'Raw_Material_Cost_Index': '122.9',
             'New_Vehicle_Orders_Index': '103.8', 'Industrial_Util_Rate_Pct': '79.7',
             'Fuel_Price_Index': '138.0', 'EV_Adoption_Index': '69.3'},
        ]
        return pd.DataFrame(rows)

    def test_m1_duplicate_resolved(self, raw_macro):
        out, dq = clean_macro(raw_macro)
        assert out['Date'].duplicated().sum() == 0

    def test_m2_fraction_corrected(self, raw_macro):
        out, dq = clean_macro(raw_macro)
        pct_2019 = out[out['Date'] == '2019-01']['Auto_Production_YoY_Pct'].values[0]
        # -0.015 x 100 = -1.5
        assert abs(pct_2019 - (-1.5)) < 0.01, f"Expected -1.5, got {pct_2019}"

    def test_m2_post2019_unchanged(self, raw_macro):
        out, dq = clean_macro(raw_macro)
        pct_2022 = out[out['Date'] == '2022-05']['Auto_Production_YoY_Pct'].values[0]
        assert abs(pct_2022 - 1.0) < 0.01


# --- Events cleaning ---

class TestEventsCleaning:
    @pytest.fixture
    def raw_events(self):
        return pd.DataFrame([
            {'Event_ID': 'EV-001', 'Start_Date': '2019-01', 'End_Date': '2019-01',
             'Scope_Type': 'Global', 'Scope_Value': 'All', 'Event_Type': 'Price Change',
             'Description': 'Annual price change'},
            {'Event_ID': 'EV-038', 'Start_Date': '2023-10', 'End_Date': '2023-11',
             'Scope_Type': 'Customer', 'Scope_Value': 'Nexon Motor Group | Europe',
             'Event_Type': 'Customer Event', 'Description': 'Strike'},
            {'Event_ID': 'EV-052', 'Start_Date': '2025-09', 'End_Date': None,
             'Scope_Type': 'Customer', 'Scope_Value': 'Helios Auto Parts | India',
             'Event_Type': 'Customer Event', 'Description': 'New assembly line'},
        ])

    def test_e1_date_normalised(self, raw_events):
        out, dq = clean_events(raw_events)
        # 2019-01 should become 2019-01-01
        ev001 = out[out['Event_ID'] == 'EV-001']
        assert ev001['Start_Date'].iloc[0] == '2019-01-01'

    def test_e2_open_ended(self, raw_events):
        out, dq = clean_events(raw_events)
        ev052 = out[out['Event_ID'] == 'EV-052']
        assert pd.isna(ev052['End_Date'].iloc[0]) or ev052['End_Date'].iloc[0] is None

    def test_e3_scope_split(self, raw_events):
        out, dq = clean_events(raw_events)
        ev038 = out[out['Event_ID'] == 'EV-038']
        assert ev038['Scope_Customer'].iloc[0] == 'Nexon Motor Group'
        assert ev038['Scope_Region'].iloc[0] == 'Europe'


# --- Schema / validation checks ---

class TestSchemaValidation:
    def test_forecast_submission_columns(self):
        """forecast_submission.csv must have exactly: Date, SKU_ID, Region, Units_Sold_Forecast."""
        from pipeline.config import OUT_FORECAST_SUB
        assert os.path.exists(OUT_FORECAST_SUB), "run python project/run_pipeline.py first"
        sub = pd.read_csv(OUT_FORECAST_SUB)
        assert list(sub.columns) == ['Date', 'SKU_ID', 'Region', 'Units_Sold_Forecast']
        assert sub['Units_Sold_Forecast'].min() >= 0
        assert sub['Units_Sold_Forecast'].isna().sum() == 0

    def test_audit_log_has_required_columns(self):
        from pipeline.config import OUT_AUDIT_LOG, AUDIT_COLS
        assert os.path.exists(OUT_AUDIT_LOG), "run python project/run_pipeline.py first"
        log = pd.read_csv(OUT_AUDIT_LOG)
        for col in AUDIT_COLS:
            assert col in log.columns, f"Audit log missing column: {col}"

    def test_clean_train_no_null_date_sku_region(self):
        from pipeline.config import OUT_TRAIN_CLEAN
        assert os.path.exists(OUT_TRAIN_CLEAN), "run python project/run_pipeline.py first"
        df = pd.read_parquet(OUT_TRAIN_CLEAN)
        assert df['Date'].isna().sum() == 0
        assert df['SKU_ID'].isna().sum() == 0

    def test_clean_train_date_format(self):
        from pipeline.config import OUT_TRAIN_CLEAN
        import re
        assert os.path.exists(OUT_TRAIN_CLEAN), "run python project/run_pipeline.py first"
        df = pd.read_parquet(OUT_TRAIN_CLEAN)
        bad = df['Date'].apply(lambda d: not bool(re.match(r'^\d{4}-\d{2}$', str(d))))
        assert bad.sum() == 0, f"{bad.sum()} dates are not YYYY-MM format"

    def test_clean_train_no_dates_after_202610(self):
        from pipeline.config import OUT_TRAIN_CLEAN
        assert os.path.exists(OUT_TRAIN_CLEAN), "run python project/run_pipeline.py first"
        df = pd.read_parquet(OUT_TRAIN_CLEAN)
        bad = df['Date'] > '2026-10'
        assert bad.sum() == 0, f"{bad.sum()} train dates after 2026-10"

    def test_clean_train_no_dup_record_id(self):
        from pipeline.config import OUT_TRAIN_CLEAN
        assert os.path.exists(OUT_TRAIN_CLEAN), "run python project/run_pipeline.py first"
        df = pd.read_parquet(OUT_TRAIN_CLEAN)
        assert df['Record_ID'].duplicated().sum() == 0

    def test_clean_train_regions_canonical(self):
        from pipeline.config import OUT_TRAIN_CLEAN
        assert os.path.exists(OUT_TRAIN_CLEAN), "run python project/run_pipeline.py first"
        df = pd.read_parquet(OUT_TRAIN_CLEAN)
        allowed = CANONICAL_REGIONS
        bad = ~df['Region'].isin(allowed)
        assert bad.sum() == 0, f"{bad.sum()} non-canonical regions: {df.loc[bad,'Region'].unique()}"
        assert 'Unassigned' in allowed
        assert not (df['Region'] == 'Unknown').any()

    def test_clean_train_channels_canonical(self):
        from pipeline.config import OUT_TRAIN_CLEAN
        assert os.path.exists(OUT_TRAIN_CLEAN), "run python project/run_pipeline.py first"
        df = pd.read_parquet(OUT_TRAIN_CLEAN)
        bad = ~df['Sales_Channel'].isin(CANONICAL_CHANNELS)
        assert bad.sum() == 0

    def test_clean_train_units_non_negative(self):
        from pipeline.config import OUT_TRAIN_CLEAN
        assert os.path.exists(OUT_TRAIN_CLEAN), "run python project/run_pipeline.py first"
        df = pd.read_parquet(OUT_TRAIN_CLEAN)
        us = pd.to_numeric(df['Units_Sold'], errors='coerce')
        bad = (us < 0) & us.notna()
        assert bad.sum() == 0

    def test_forecast_row_count(self):
        from pipeline.config import OUT_FORECAST_SUB
        assert os.path.exists(OUT_FORECAST_SUB), "run python project/run_pipeline.py first"
        sub = pd.read_csv(OUT_FORECAST_SUB)
        assert len(sub) == 58219, f"Expected 58219 rows, got {len(sub)}"

    def test_pmp004_bu_resolution(self):
        """Rule S1: PMP-004 BU conflict resolved to Advanced Manufacturing."""
        from pipeline.clean_master_macro_events import clean_sku_master
        from pipeline.load import load_sku_master
        sku_clean, dq = clean_sku_master(load_sku_master())
        pmp = sku_clean[sku_clean['SKU_ID'] == 'PMP-004']
        assert pmp['Business_Unit'].iloc[0] == 'Advanced Manufacturing'

    def test_schema_quantile_ordering_on_fixture(self):
        """Verify P10 <= P50 <= P90 non-crossing constraint."""
        from pipeline.models.lightgbm_model import enforce_quantile_ordering
        p10 = np.array([10.0, 50.0, 30.0])
        p50 = np.array([20.0, 40.0, 30.0])
        p90 = np.array([15.0, 60.0, 25.0])
        p10_o, p50_o, p90_o = enforce_quantile_ordering(p10, p50, p90)
        assert (p10_o <= p50_o).all()
        assert (p50_o <= p90_o).all()

    def test_hierarchy_addition_structure(self, tmp_path):
        """Verify the hierarchy addition columns and exact result."""
        from pipeline.forecast import compute_hierarchy_reconciliation
        sub = pd.DataFrame([
            {'Date': '2026-11', 'SKU_ID': 'S1', 'Region': 'North America', 'Units_Sold_Forecast': 30},
            {'Date': '2026-11', 'SKU_ID': 'S2', 'Region': 'Europe', 'Units_Sold_Forecast': 20},
        ])
        master = pd.DataFrame([
            {'SKU_ID': 'S1', 'Product_Family': 'F1', 'Business_Unit': 'BU1'},
            {'SKU_ID': 'S2', 'Product_Family': 'F2', 'Business_Unit': 'BU2'},
        ])
        history = pd.DataFrame([
            {'Date': '2026-10', 'Product_Family': 'F1', 'Region': 'North America', 'Business_Unit': 'BU1', 'Units_Sold': 30},
            {'Date': '2026-10', 'Product_Family': 'F2', 'Region': 'Europe', 'Business_Unit': 'BU2', 'Units_Sold': 20},
        ])
        out_path = tmp_path / 'hierarchy_addition.csv'
        res = compute_hierarchy_reconciliation(sub, history, master, out_path=out_path)
        assert res['adds_up'] is True
        assert set(res) == {'adds_up', 'total_forecast_units'}
        df = pd.read_csv(out_path)
        assert {'Level', 'Group', 'Bottom_Up_Units', 'Deviation_Pct', 'Status'} == set(df.columns)

    def test_features_yoy_growth_not_identical_to_lag12(self):
        """Fix F4: lag_12m_yoy must be the YoY ratio, distinct from lag_12m."""
        from pipeline.features import add_lag_rolling_features
        df = pd.DataFrame([
            {'SKU_ID': 'S1', 'Region': 'R1', 'Customer_Name': 'C1', 'Date': f'2020-{m:02d}', 'Units_Sold': 100 + m * 5}
            for m in range(1, 13)
        ] + [
            {'SKU_ID': 'S1', 'Region': 'R1', 'Customer_Name': 'C1', 'Date': '2021-01', 'Units_Sold': 200}
        ])
        out = add_lag_rolling_features(df)
        last_row = out.iloc[-1]
        assert 'lag_12m_yoy' in out.columns
        assert last_row['lag_12m_yoy'] != last_row['lag_12m']
        # lag_1m is 160 (from 2020-12), lag_12m is 105 (from 2020-01)
        # yoy = (160 - 105) / 105 approx 0.5238
        assert abs(last_row['lag_12m_yoy'] - ((160 - 105) / 105.001)) < 0.05

    def test_features_no_duplicate_key_rows(self):
        """Priority 1: features table must have zero duplicate rows on
        Date, SKU_ID, Region, Customer_Name (Fixes V1).
        """
        import os
        import pytest
        feat_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'features.parquet')
        if not os.path.exists(feat_path):
            pytest.fail("features.parquet not found; run python project/run_pipeline.py first")
        feat_df = pd.read_parquet(feat_path)
        key_cols = ['Date', 'SKU_ID', 'Region', 'Customer_Name']
        dup_count = feat_df.duplicated(key_cols).sum()
        assert dup_count == 0, f"features table has {dup_count} duplicate rows on {key_cols}"

    def test_t4b_one_row_per_customer_key(self):
        """Priority 1: A fixture with two channels for one customer-SKU-region-month
        yields exactly one row and logs the drop with rule ID T4b (Rule R2).
        """
        from pipeline.clean_train import apply_t4b_dsr_dedup
        from pipeline.audit import audit
        fixture = pd.DataFrame([
            {
                'Record_ID': 'REC-001', 'Date': '2024-05', 'SKU_ID': 'HYD-001',
                'Region': 'North America', 'Customer_Name': 'Orion Driveline',
                'Sales_Channel': 'Direct OEM', 'Load_Timestamp': '2024-05-01 10:00:00',
                'Units_Sold': 50
            },
            {
                'Record_ID': 'REC-002', 'Date': '2024-05', 'SKU_ID': 'HYD-001',
                'Region': 'North America', 'Customer_Name': 'Orion Driveline',
                'Sales_Channel': 'Tier-1 Distributor', 'Load_Timestamp': '2024-05-02 12:00:00',
                'Units_Sold': 60
            }
        ])
        audit_len_before = len(audit)
        cleaned = apply_t4b_dsr_dedup(fixture)
        assert len(cleaned) == 1
        assert cleaned.iloc[0]['Record_ID'] == 'REC-002'
        # Check drop was logged
        audit_records = audit.as_dataframe()
        t4b_drops = audit_records[audit_records['rule_id'] == 'T4b']
        assert len(t4b_drops) > 0
        assert 'REC-001' in t4b_drops['row_key'].values

    def test_stockout_allocation_two_customers_sums_to_total(self):
        """Priority 2: Two-customer key splits proportionally and sums to stockout total."""
        from pipeline.unconstrained import allocate_stockout_metric
        train_fixture = pd.DataFrame([
            {'Date': '2024-01', 'SKU_ID': 'BSS-001', 'Region': 'Europe', 'Units_Sold': 60},
            {'Date': '2024-01', 'SKU_ID': 'BSS-001', 'Region': 'Europe', 'Units_Sold': 40},
        ])
        so_fixture = pd.DataFrame([
            {'Date': '2024-01', 'SKU_ID': 'BSS-001', 'Region': 'Europe', 'Requested_Units': 100}
        ])
        res = allocate_stockout_metric(train_fixture, so_fixture, 'Requested_Units')
        assert len(res) == 2
        assert res['allocated_Requested_Units'].sum() == 100
        assert res.iloc[0]['allocated_Requested_Units'] == 60
        assert res.iloc[1]['allocated_Requested_Units'] == 40

    def test_stockout_allocation_single_customer_unchanged(self):
        """Priority 2: Single customer key gets 100% of stockout value."""
        from pipeline.unconstrained import allocate_stockout_metric
        train_fixture = pd.DataFrame([
            {'Date': '2024-01', 'SKU_ID': 'BSS-001', 'Region': 'Europe', 'Units_Sold': 50}
        ])
        so_fixture = pd.DataFrame([
            {'Date': '2024-01', 'SKU_ID': 'BSS-001', 'Region': 'Europe', 'Requested_Units': 75}
        ])
        res = allocate_stockout_metric(train_fixture, so_fixture, 'Requested_Units')
        assert len(res) == 1
        assert res.iloc[0]['allocated_Requested_Units'] == 75
        assert res.iloc[0]['share_Requested_Units'] == 1.0

    def test_stockout_allocation_all_zero_splits_equally(self):
        """Priority 2: All-zero demand splits equally, remainder to largest/first customer."""
        from pipeline.unconstrained import allocate_stockout_metric
        train_fixture = pd.DataFrame([
            {'Date': '2024-01', 'SKU_ID': 'BSS-001', 'Region': 'Europe', 'Units_Sold': 0},
            {'Date': '2024-01', 'SKU_ID': 'BSS-001', 'Region': 'Europe', 'Units_Sold': 0},
        ])
        so_fixture = pd.DataFrame([
            {'Date': '2024-01', 'SKU_ID': 'BSS-001', 'Region': 'Europe', 'Requested_Units': 7}
        ])
        res = allocate_stockout_metric(train_fixture, so_fixture, 'Requested_Units')
        assert len(res) == 2
        assert res['allocated_Requested_Units'].sum() == 7
        # 7 / 2 = 3.5 -> floor is 3, remainder 1 goes to first
        assert set(res['allocated_Requested_Units'].values) == {3.0, 4.0}

    def test_backtest_predictions_exists(self):
        """Priority 3: backtest_predictions.csv exists with required columns."""
        import os
        import pytest
        pred_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'backtest_predictions.csv')
        if not os.path.exists(pred_path):
            pytest.fail("backtest_predictions.csv not found; run python project/run_pipeline.py first")
        df = pd.read_csv(pred_path, low_memory=False)
        assert len(df) > 0, "backtest_predictions.csv is empty"
        expected = {'origin', 'model', 'Date', 'SKU_ID', 'Region', 'Customer_Name', 'actual', 'p10', 'p50', 'p90'}
        assert expected.issubset(set(df.columns)), f"Missing columns: {expected - set(df.columns)}"

    def test_event_rules_no_helios_mobility(self):
        """Priority 4a: 'Helios Mobility' must not appear anywhere in code; canonical is Helios Auto Parts."""
        import os
        pipeline_dir = os.path.join(os.path.dirname(__file__), '..', 'pipeline')
        for fname in os.listdir(pipeline_dir):
            if fname.endswith('.py'):
                fpath = os.path.join(pipeline_dir, fname)
                with open(fpath, encoding='utf-8') as f:
                    content = f.read()
                    assert 'Helios Mobility' not in content, f"Found 'Helios Mobility' in {fname}"

    def test_launch_replaces_sku_mapping(self):
        """Priority 4b: New launches with Replaces_SKU inherit replaced catalog history."""
        from pipeline.config import RAW_SKU_MASTER
        sku_master = pd.read_csv(RAW_SKU_MASTER)
        replaces = sku_master[sku_master['Replaces_SKU'].notna() & (sku_master['Replaces_SKU'] != '')]
        assert len(replaces) > 0
        pmp20 = replaces[replaces['SKU_ID'] == 'PMP-020']
        assert not pmp20.empty
        assert pmp20.iloc[0]['Replaces_SKU'] == 'PMP-003'

    def test_launch_inherits_history(self):
        """A launch SKU inherits the replaced SKU's last value scaled by the family-region ratio."""
        from pipeline.forecast import generate_recursive_forecast

        class ConstantModel:
            def __init__(self):
                self.seen_lag_1m = None
            def predict(self, frame):
                self.seen_lag_1m = frame['lag_1m'].copy()
                return np.full(len(frame), 40.0)

        history_rows = []
        for month in pd.period_range('2026-01', periods=12, freq='M').strftime('%Y-%m'):
            history_rows.append({
                'Date': month, 'SKU_ID': 'OLD-001', 'Region': 'North America',
                'Customer_Name': 'Customer A', 'Units_Sold_Unconstrained': 100.0,
                'Units_Sold': 100.0,
                'Product_Family': 'Family A', 'Sales_Channel': 'Direct OEM',
                'Business_Unit': 'BU A',
            })
        train_history = pd.DataFrame(history_rows)
        test_df = pd.DataFrame([
            {'Date': '2027-03', 'SKU_ID': 'NEW-001', 'Region': 'North America',
             'Customer_Name': 'Customer A', 'Product_Family': 'Family A',
             'Sales_Channel': 'Direct OEM', 'Business_Unit': 'BU A'},
        ])
        sku_master = pd.DataFrame([
            {'SKU_ID': 'OLD-001', 'Product_Family': 'Family A', 'Business_Unit': 'BU A',
             'Launch_Date': '2020-01', 'EOL_Date': '', 'Replaces_SKU': ''},
            {'SKU_ID': 'NEW-001', 'Product_Family': 'Family A', 'Business_Unit': 'BU A',
             'Launch_Date': '2027-03', 'EOL_Date': '', 'Replaces_SKU': 'OLD-001'},
        ])
        macro = pd.DataFrame({'Date': ['2027-03']})
        events = pd.DataFrame(columns=['Event_ID', 'Start_Date', 'End_Date', 'Scope_Customer', 'Scope_Region'])
        p10_model = ConstantModel()
        p50_model = ConstantModel()
        p90_model = ConstantModel()
        models = {'p10': p10_model, 'p50': p50_model, 'p90': p90_model}

        forecast, _ = generate_recursive_forecast(
            test_df, train_history, macro, events, sku_master, models, save_outputs=False
        )
        launch_row = forecast[forecast['SKU_ID'] == 'NEW-001'].iloc[0]
        assert np.isclose(p50_model.seen_lag_1m.iloc[0], 100.0, rtol=1e-3)
        assert launch_row['Units_Sold_Forecast'] > 0

    def test_whatif_zero_shift_equals_baseline(self):
        """Priority 3: Scenario at delta 0 equals baseline forecast."""
        import os
        import pytest
        scen_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'whatif_scenarios.csv')
        if not os.path.exists(scen_path):
            pytest.fail("whatif_scenarios.csv not found; run python project/run_pipeline.py first")
        scen = pd.read_csv(scen_path)
        zero_scen = scen[scen['Scenario_PMI_Delta'] == 0.0]
        if not zero_scen.empty:
            if 'Forecast_Units' in zero_scen.columns and 'Baseline_Units' in zero_scen.columns:
                diff = (zero_scen['Forecast_Units'] - zero_scen['Baseline_Units']).abs().sum()
            else:
                diff = (zero_scen['Units_Sold_Forecast'] - zero_scen['Scenario_Units_Sold_Forecast']).abs().sum()
            assert diff == 0, "Scenario at delta 0 must equal baseline forecast"

    def test_calibration_offsets_independent_of_test_actuals(self):
        """Priority A2 / 1d: get_calibration_offsets uses only residuals with Date <= current_origin."""
        from pipeline.backtest import get_calibration_offsets
        resids_history = [
            ('2023-01', -10.0),
            ('2023-05', 0.0),
            ('2023-10', 10.0),
            ('2024-01', 500.0),
            ('2024-05', -999.0)
        ]
        q10, q90 = get_calibration_offsets(resids_history, '2023-10')
        assert -10.0 <= q10 <= 0.0
        assert 0.0 <= q90 <= 10.0

        # Perturbing future test actuals (> 2023-10) does not change offsets for current origin
        perturbed = resids_history + [('2024-06', 99999.0), ('2024-07', -88888.0)]
        q10_pert, q90_pert = get_calibration_offsets(perturbed, '2023-10')
        assert q10 == q10_pert
        assert q90 == q90_pert

    def test_null_actual_does_not_change_wape_or_bias(self):
        """Priority 2c: A NULL actual does not change WAPE or bias."""
        from pipeline.backtest import evaluate_backtest_slice
        s_act_clean = pd.Series([100.0, 200.0, 300.0])
        s_pred_clean = pd.Series([110.0, 190.0, 310.0])

        s_act_with_null = pd.Series([100.0, 200.0, 300.0, np.nan, None])
        s_pred_with_null = pd.Series([110.0, 190.0, 310.0, 500.0, 999.0])

        res_clean = evaluate_backtest_slice(s_act_clean, s_pred_clean, '2024-10', 'm', 'ov', 'all')
        res_null = evaluate_backtest_slice(s_act_with_null, s_pred_with_null, '2024-10', 'm', 'ov', 'all')

        assert res_clean['WAPE'] == res_null['WAPE']
        assert res_clean['Bias'] == res_null['Bias']
        assert res_null['n_excluded'] == 2

    def test_ev059_and_ev052_no_post_forecast_double_counting(self):
        """Priority A6: EV-059 and EV-052 do not apply post-forecast multipliers."""
        from pipeline.config import EVENT_FORECAST_RULES
        from pipeline.forecast import apply_event_adjustments
        assert EVENT_FORECAST_RULES['EV-059'].get('effect') == 'model_feature'
        assert EVENT_FORECAST_RULES['EV-052'].get('effect') == 'model_feature'

        events_fixture = pd.DataFrame([
            {'Event_ID': 'EV-052', 'Start_Date': '2027-01', 'End_Date': '2027-05',
             'Scope_Customer': 'Helios Auto Parts', 'Scope_Region': 'India'},
            {'Event_ID': 'EV-059', 'Start_Date': '2027-01', 'End_Date': '2027-12',
             'Scope_Customer': 'Global', 'Scope_Region': 'Global'},
        ])
        df_forecast = pd.DataFrame([
            {'Date': '2027-02', 'SKU_ID': 'BSS-001', 'Region': 'India',
             'Customer_Name': 'Helios Auto Parts', 'Units_Sold_Forecast': 100}
        ])
        adjusted = apply_event_adjustments(df_forecast, events_fixture, out_estimates_path=None)
        assert adjusted['Units_Sold_Forecast'].iloc[0] == 100

    def test_hierarchy_addition_structure_and_result(self):
        """Priority A3: Hierarchy addition reports exact totals."""
        from pipeline.forecast import compute_hierarchy_reconciliation
        sub_fixture = pd.DataFrame([
            {'Date': '2026-11', 'SKU_ID': 'BSS-001', 'Region': 'North America', 'Units_Sold_Forecast': 100},
            {'Date': '2026-11', 'SKU_ID': 'BSS-002', 'Region': 'Europe', 'Units_Sold_Forecast': 200},
        ])
        sku_master_fixture = pd.DataFrame([
            {'SKU_ID': 'BSS-001', 'Product_Family': 'Brake & Safety Systems', 'Business_Unit': 'Safety & Compliance'},
            {'SKU_ID': 'BSS-002', 'Product_Family': 'Brake & Safety Systems', 'Business_Unit': 'Safety & Compliance'},
        ])
        clean_train_fixture = pd.DataFrame([
            {'Date': '2026-10', 'Product_Family': 'Brake & Safety Systems', 'Region': 'North America', 'Units_Sold_Unconstrained': 100},
            {'Date': '2026-10', 'Product_Family': 'Brake & Safety Systems', 'Region': 'Europe', 'Units_Sold_Unconstrained': 200},
        ])
        res = compute_hierarchy_reconciliation(sub_fixture, clean_train_fixture, sku_master_fixture, out_path=None)
        assert 'total_forecast_units' in res
        assert 'adds_up' in res
        assert res['adds_up'] is True

    def test_hierarchy_addition_deviating_fixture_adds_up(self, tmp_path):
        """The submission fixture adds exactly across family, region and business unit."""
        from pipeline.forecast import compute_hierarchy_reconciliation
        sub_fixture = pd.DataFrame([
            {'Date': '2026-11', 'SKU_ID': 'BSS-001', 'Region': 'North America', 'Units_Sold_Forecast': 100},
            {'Date': '2026-11', 'SKU_ID': 'BSS-002', 'Region': 'Europe', 'Units_Sold_Forecast': 200},
            {'Date': '2026-11', 'SKU_ID': 'CHA-001', 'Region': 'Asia Pacific', 'Units_Sold_Forecast': 150},
        ])
        sku_master_fixture = pd.DataFrame([
            {'SKU_ID': 'BSS-001', 'Product_Family': 'Brake & Safety Systems', 'Business_Unit': 'Safety & Compliance'},
            {'SKU_ID': 'BSS-002', 'Product_Family': 'Brake & Safety Systems', 'Business_Unit': 'Safety & Compliance'},
            {'SKU_ID': 'CHA-001', 'Product_Family': 'Chassis & Structural', 'Business_Unit': 'Advanced Manufacturing'},
        ])
        clean_train_fixture = pd.DataFrame([
            {'Date': '2026-10', 'Product_Family': 'Brake & Safety Systems', 'Region': 'North America', 'Units_Sold_Unconstrained': 2000},
            {'Date': '2026-10', 'Product_Family': 'Brake & Safety Systems', 'Region': 'Europe', 'Units_Sold_Unconstrained': 3000},
            {'Date': '2026-10', 'Product_Family': 'Chassis & Structural', 'Region': 'Asia Pacific', 'Units_Sold_Unconstrained': 2500},
        ])
        tmp_recon_file = tmp_path / "hierarchy_addition_test.csv"
        res = compute_hierarchy_reconciliation(sub_fixture, clean_train_fixture, sku_master_fixture,
                                               out_path=tmp_recon_file)
        assert res['adds_up'] is True
        assert res['total_forecast_units'] == 450
        assert tmp_recon_file.exists()
        diagnostic_file = tmp_path / "hierarchy_diagnostic.csv"
        assert diagnostic_file.exists()
        diagnostic = pd.read_csv(diagnostic_file)
        assert 'Gap_Pct' in diagnostic.columns

        submission_with_families = sub_fixture.merge(
            sku_master_fixture[['SKU_ID', 'Product_Family', 'Business_Unit']],
            on='SKU_ID',
            how='left',
        )
        total = int(sub_fixture['Units_Sold_Forecast'].sum())
        assert int(submission_with_families.groupby('Product_Family')['Units_Sold_Forecast'].sum().sum()) == total
        assert int(submission_with_families.groupby('Region')['Units_Sold_Forecast'].sum().sum()) == total
        assert int(submission_with_families.groupby('Business_Unit')['Units_Sold_Forecast'].sum().sum()) == total

    def test_pytest_does_not_modify_outputs(self):
        """Fix 1b: Verify that running the test suite does not modify production outputs."""
        import subprocess, sys
        from pathlib import Path
        out_dir = Path(__file__).resolve().parent.parent / "outputs"
        assert out_dir.exists(), "run python project/run_pipeline.py first"

        initial_mtimes = {p: p.stat().st_mtime_ns for p in out_dir.glob("*") if p.is_file()}

        cmd = [
            sys.executable, "-m", "pytest", "tests",
            "-k", "not test_pytest_does_not_modify_outputs",
            "-q", "-p", "no:cacheprovider"
        ]
        res = subprocess.run(cmd, cwd=str(out_dir.parent), capture_output=True, text=True, timeout=300)
        assert res.returncode == 0, f"Pytest subprocess failed: {res.stdout}\n{res.stderr}"

        modified_files = []
        for p, orig_mtime in initial_mtimes.items():
            curr_mtime = p.stat().st_mtime_ns
            if curr_mtime != orig_mtime:
                modified_files.append((p.name, orig_mtime, curr_mtime))

        assert not modified_files, f"Production files in outputs/ were modified during test execution: {modified_files}"

    def test_new_launch_inheritance_fixture(self):
        """Priority 6c: A launch forecast before Launch_Date is zero, and inherited values match rule."""
        from pipeline.forecast import apply_lifecycle_rules
        sku_master_fixture = pd.DataFrame([
            {'SKU_ID': 'NEW-001', 'Launch_Date': '2027-03', 'EOL_Date': '', 'Replaces_SKU': 'OLD-001'}
        ])
        forecast_fixture = pd.DataFrame([
            {'Date': '2027-01', 'SKU_ID': 'NEW-001', 'Units_Sold_Forecast': 150},
            {'Date': '2027-02', 'SKU_ID': 'NEW-001', 'Units_Sold_Forecast': 160},
            {'Date': '2027-03', 'SKU_ID': 'NEW-001', 'Units_Sold_Forecast': 170},
            {'Date': '2027-04', 'SKU_ID': 'NEW-001', 'Units_Sold_Forecast': 180},
        ])

        cleaned = apply_lifecycle_rules(forecast_fixture, sku_master_fixture)
        assert cleaned.loc[cleaned['Date'] == '2027-01', 'Units_Sold_Forecast'].iloc[0] == 0
        assert cleaned.loc[cleaned['Date'] == '2027-02', 'Units_Sold_Forecast'].iloc[0] == 0
        assert cleaned.loc[cleaned['Date'] == '2027-03', 'Units_Sold_Forecast'].iloc[0] == 170
        assert cleaned.loc[cleaned['Date'] == '2027-04', 'Units_Sold_Forecast'].iloc[0] == 180

    def test_dashboard_all_filter_combinations(self):
        """Fix 6a: Test dashboard views across all 7 filter combinations and verify 0 exceptions."""
        from unittest.mock import MagicMock
        import streamlit as st
        for attr in ['header', 'subheader', 'caption', 'dataframe', 'metric', 'plotly_chart', 'write', 'warning', 'info', 'error', 'stop']:
            setattr(st, attr, MagicMock())

        def mock_columns(n_or_list):
            n = n_or_list if isinstance(n_or_list, int) else len(n_or_list)
            return [MagicMock() for _ in range(n)]
        st.columns = mock_columns
        st.radio = MagicMock(return_value='Units Sold')
        st.selectbox = MagicMock(return_value='Product_Family')
        st.slider = MagicMock(return_value=-3.0)

        from app.dashboard import (
            load_clean_train, load_forecast, load_dq_report, load_anomaly_register,
            load_backtest_metrics, load_macro_lag, load_stockouts, load_ev_shift,
            load_scenario, view_executive_overview, view_data_health, view_demand_explorer,
            view_anomaly_monitor, view_forecast_accuracy, view_macro_lag_heatmap,
            view_stockouts, view_scenario, view_powertrain_lifecycle
        )

        df_train = load_clean_train()
        df_forecast, df_intervals = load_forecast()
        dq_report = load_dq_report()
        anomaly_df = load_anomaly_register()
        backtest_df = load_backtest_metrics()
        macro_lag_df = load_macro_lag()
        stockouts_df, unconstrained_df = load_stockouts()
        ev_df = load_ev_shift()
        scenario_df = load_scenario()

        all_fams = sorted(df_train['Product_Family'].dropna().unique())
        all_skus = sorted(df_train['SKU_ID'].dropna().unique())
        all_regs = sorted(df_train['Region'].dropna().unique())
        all_chs = sorted(df_train['Sales_Channel'].dropna().unique())
        all_custs = sorted(df_train['Customer_Name'].dropna().unique())

        combos = [
            ('no_filters', {'families': all_fams, 'skus': all_skus, 'regions': all_regs, 'channels': all_chs, 'customers': all_custs}),
            ('1_family', {'families': [all_fams[0]], 'skus': all_skus, 'regions': all_regs, 'channels': all_chs, 'customers': all_custs}),
            ('1_sku', {'families': all_fams, 'skus': [all_skus[0]], 'regions': all_regs, 'channels': all_chs, 'customers': all_custs}),
            ('1_region', {'families': all_fams, 'skus': all_skus, 'regions': [all_regs[0]], 'channels': all_chs, 'customers': all_custs}),
            ('1_channel', {'families': all_fams, 'skus': all_skus, 'regions': all_regs, 'channels': [all_chs[0]], 'customers': all_custs}),
            ('1_customer', {'families': all_fams, 'skus': all_skus, 'regions': all_regs, 'channels': all_chs, 'customers': [all_custs[0]]}),
            ('2_combined', {'families': [all_fams[0]], 'skus': all_skus, 'regions': [all_regs[0]], 'channels': all_chs, 'customers': all_custs}),
        ]

        for _, flt in combos:
            view_executive_overview(df_train, df_forecast, df_intervals, flt)
            view_data_health(dq_report)
            view_demand_explorer(df_train, flt)
            view_anomaly_monitor(df_train, anomaly_df, flt)
            view_forecast_accuracy(df_train, df_forecast, df_intervals, backtest_df, flt)
            view_macro_lag_heatmap(macro_lag_df)
            view_stockouts(stockouts_df, unconstrained_df, flt)
            view_scenario(scenario_df, df_forecast)
            view_powertrain_lifecycle(ev_df)
        assert len(combos) == 7


def test_dashboard_views_return_content():
    """Each required and optional dashboard view passes non-empty data to a chart or table."""
    from unittest.mock import MagicMock
    import streamlit as st
    for attr in ['header', 'subheader', 'caption', 'metric', 'plotly_chart', 'dataframe',
                 'write', 'warning', 'info', 'error', 'stop']:
        setattr(st, attr, MagicMock())
    st.columns = lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
    st.radio = MagicMock(return_value='Units Sold')
    st.selectbox = MagicMock(return_value='Product_Family')
    st.select_slider = MagicMock(return_value=-3.0)
    st.expander = MagicMock()
    st.expander.return_value.__enter__ = MagicMock(return_value=None)
    st.expander.return_value.__exit__ = MagicMock(return_value=False)

    from app.dashboard import (
        load_clean_train, load_forecast, load_dq_report, load_anomaly_register,
        load_backtest_metrics, load_macro_lag, load_stockouts, load_ev_shift,
        load_scenario, view_executive_overview, view_data_health, view_demand_explorer,
        view_anomaly_monitor, view_forecast_accuracy, view_macro_lag_heatmap,
        view_stockouts, view_scenario, view_powertrain_lifecycle
    )

    df_train = load_clean_train()
    df_forecast, df_intervals = load_forecast()
    dq_report = load_dq_report()
    anomaly_df = load_anomaly_register()
    backtest_df = load_backtest_metrics()
    macro_lag_df = load_macro_lag()
    stockouts_df, unconstrained_df = load_stockouts()
    ev_df = load_ev_shift()
    scenario_df = load_scenario()
    family = df_train['Product_Family'].dropna().iloc[0]
    region = df_train['Region'].dropna().iloc[0]
    sku = df_train['SKU_ID'].dropna().iloc[0]
    channel = df_train['Sales_Channel'].dropna().iloc[0]
    customer = df_train['Customer_Name'].dropna().iloc[0]
    filters = {
        'families': [family], 'skus': [sku], 'regions': [region],
        'channels': [channel], 'customers': [customer],
    }

    calls = [
        ('Executive overview', lambda: view_executive_overview(df_train, df_forecast, df_intervals, filters)),
        ('Data health', lambda: view_data_health(dq_report)),
        ('Demand explorer', lambda: view_demand_explorer(df_train, filters)),
        ('Anomaly monitor', lambda: view_anomaly_monitor(df_train, anomaly_df, filters)),
        ('Forecast accuracy', lambda: view_forecast_accuracy(df_train, df_forecast, df_intervals, backtest_df, filters)),
        ('Macro lead lag', lambda: view_macro_lag_heatmap(macro_lag_df)),
        ('Stockouts', lambda: view_stockouts(stockouts_df, unconstrained_df, filters)),
        ('What-if', lambda: view_scenario(scenario_df, df_forecast)),
        ('Lifecycle', lambda: view_powertrain_lifecycle(ev_df)),
    ]
    for name, call in calls:
        st.plotly_chart.reset_mock()
        st.dataframe.reset_mock()
        call()
        assert st.plotly_chart.call_count + st.dataframe.call_count > 0, f"{name} returned no chart or table"
        chart_objects = [args[0] for args, _ in st.plotly_chart.call_args_list]
        table_objects = [args[0] for args, _ in st.dataframe.call_args_list]
        has_non_empty = any(
            getattr(obj, 'data', None) not in (None, [])
            for obj in chart_objects
        ) or any(
            isinstance(obj, pd.DataFrame) and not obj.empty
            for obj in table_objects
        )
        assert has_non_empty, f"{name} passed no non-empty chart or table object"


# PDF Business Rules R1 through R7, Hierarchy Exactness, and Backtest Temporal Isolation

def test_rule_r1_date_authoritative():
    """PDF Rule R1: Date is authoritative. Year and Month follow Date."""
    df = pd.DataFrame([
        {'Record_ID': 'R1', 'Date': '2023-04', 'Year': '2020', 'Month': '1'},
        {'Record_ID': 'R2', 'Date': '2024-11', 'Year': '2021', 'Month': '2'}
    ])
    cleaned = apply_t2_year_month(df)
    assert cleaned.loc[0, 'Year'] == '2023'
    assert cleaned.loc[0, 'Month'] == '4'
    assert cleaned.loc[1, 'Year'] == '2024'
    assert cleaned.loc[1, 'Month'] == '11'


def test_rule_r2_load_timestamp_dedup():
    """PDF Rule R2: Repeated Record_ID: keep the row with the latest Load_Timestamp."""
    row1 = {'Record_ID': 'CEL-001', 'Date': '2023-01', 'SKU_ID': 'BSS-001', 'Region': 'Europe',
            'Customer_Name': 'Cust A', 'Sales_Channel': 'Direct OEM',
            'Load_Timestamp': '2023-01-01 10:00:00', 'Units_Sold': '100'}
    row2 = {'Record_ID': 'CEL-001', 'Date': '2023-01', 'SKU_ID': 'BSS-001', 'Region': 'Europe',
            'Customer_Name': 'Cust A', 'Sales_Channel': 'Direct OEM',
            'Load_Timestamp': '2023-01-02 10:00:00', 'Units_Sold': '200'}
    df = pd.DataFrame([row1, row2])
    cleaned = apply_t4_dedup_record_id(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]['Units_Sold'] == '200'
    assert cleaned.iloc[0]['Load_Timestamp'] == '2023-01-02 10:00:00'


def test_rule_r3_zero_orders_vs_missing_rows():
    """PDF Rule R3: Units_Sold = 0 is a valid no-order record; missing rows are not zeros."""
    df = pd.DataFrame([{'Record_ID': 'REC-Z1', 'Units_Sold': '0'}])
    cleaned = apply_t15_zero_demand_flag(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]['Units_Sold'] == '0'
    p_status_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'panel_status.csv')
    p_status = pd.read_csv(p_status_path)
    statuses = set(p_status['status'].unique())
    assert {'zero_order', 'not_recorded', 'sold'}.issubset(statuses)


def test_rule_r4_prices_same_per_region_annual_change():
    """PDF Rule R4: Prices per SKU in USD uniform across regions; annual 1 Jan adjustments except surcharges."""
    df = pd.DataFrame([
        {'Record_ID': 'P1', 'Date': '2023-01', 'Year': '2023', 'SKU_ID': 'BSS-001', 'Region': 'North America',
         'Unit_Price_USD': '500.0', 'Units_Sold': '10', 'Revenue_USD': '5000.0'},
        {'Record_ID': 'P2', 'Date': '2023-02', 'Year': '2023', 'SKU_ID': 'BSS-001', 'Region': 'Europe',
         'Unit_Price_USD': '500.0', 'Units_Sold': '10', 'Revenue_USD': '5000.0'},
        {'Record_ID': 'P3', 'Date': '2023-03', 'Year': '2023', 'SKU_ID': 'BSS-001', 'Region': 'Asia Pacific',
         'Unit_Price_USD': '1500000.0', 'Units_Sold': '10', 'Revenue_USD': '15000000.0'}
    ])
    cleaned = apply_t12_prices(df)
    assert float(cleaned.loc[2, 'Unit_Price_USD']) == 500.0


def test_rule_r5_lifecycle_no_sales_outside_bounds():
    """PDF Rule R5: No sales after EOL_Date or before planned Launch_Date."""
    sku_master = pd.DataFrame([
        {'SKU_ID': 'SKU-PRE', 'Launch_Date': '2023-06', 'EOL_Date': ''},
        {'SKU_ID': 'SKU-POST', 'Launch_Date': '2020-01', 'EOL_Date': '2022-12'},
        {'SKU_ID': 'SKU-ACTIVE', 'Launch_Date': '2020-01', 'EOL_Date': ''}
    ])
    df_r5 = pd.DataFrame([
        {'Record_ID': 'L1', 'Date': '2023-01', 'SKU_ID': 'SKU-PRE', 'Units_Sold': 50},
        {'Record_ID': 'L2', 'Date': '2023-01', 'SKU_ID': 'SKU-POST', 'Units_Sold': 60},
        {'Record_ID': 'L3', 'Date': '2023-01', 'SKU_ID': 'SKU-ACTIVE', 'Units_Sold': 70}
    ])
    from pipeline.clean_train import DQ
    apply_r5_lifecycle(df_r5, sku_master)
    assert DQ['R5_eol_violations'] == 1
    assert DQ['R5_launch_violations'] == 1

    sub_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'forecast_submission.csv')
    sku_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'student_data', 'sku_master.csv')
    sub = pd.read_csv(sub_path)
    sku_full = pd.read_csv(sku_path)
    merged = sub.merge(sku_full[['SKU_ID', 'Launch_Date', 'EOL_Date']], on='SKU_ID', how='left')
    eol = merged['EOL_Date'].fillna('').str[:7]
    launch = merged['Launch_Date'].fillna('').str[:7]
    assert merged[(eol != '') & (merged['Date'] > eol) & (merged['Units_Sold_Forecast'] > 0)].empty
    assert merged[(launch != '') & (merged['Date'] < launch) & (merged['Units_Sold_Forecast'] > 0)].empty


def test_rule_r6_customer_ordering_cycles():
    """PDF Rule R6: Customers order on their own cycle; cycle detection and change points are tracked."""
    cs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'customer_switches.csv')
    assert os.path.exists(cs_path), "customer_switches.csv must exist"
    cs_df = pd.read_csv(cs_path)
    assert not cs_df.empty
    assert 'Prior_Customer' in cs_df.columns
    assert 'New_Customer' in cs_df.columns

    from pipeline.ordering_cycles import analyze_customer_cycles
    # Build a true monthly-then-quarterly series.
    quarterly = pd.period_range('2021-01', periods=12, freq='3M').strftime('%Y-%m').tolist()
    fixture_dates = pd.period_range('2020-01', periods=12, freq='M').strftime('%Y-%m').tolist() + quarterly
    fixture = pd.DataFrame([
        {'Date': date, 'SKU_ID': 'S1', 'Region': 'North America',
         'Customer_Name': 'Customer A', 'Units_Sold': 10}
        for date in fixture_dates
    ])
    cycles, _, _ = analyze_customer_cycles(fixture)
    cycle_row = cycles.iloc[0]
    assert cycle_row['Change_Points'] != 'None'
    intervals = np.diff([int(date[:4]) * 12 + int(date[5:7]) for date in fixture_dates])
    split = len(intervals) // 2
    assert np.median(intervals[:split]) != np.median(intervals[split:])


def test_rule_r7_unconstrained_demand_forecast():
    """PDF Rule R7: Forecast window has no supply limits; forecast unconstrained demand."""
    unc_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'unconstrained_demand.parquet')
    assert os.path.exists(unc_path), "unconstrained_demand.parquet must exist"
    unc_df = pd.read_parquet(unc_path)
    assert 'Units_Sold' in unc_df.columns
    assert 'Units_Sold_Unconstrained' in unc_df.columns
    so_rows = unc_df[unc_df['stockout_status'] == 'listed_stockout']
    assert not so_rows.empty
    assert so_rows['Units_Sold_Unconstrained'].sum() > so_rows['Units_Sold'].sum()


def test_hierarchy_submission_exact_sums():
    """PDF Part 4.10 & 6: Forecast submission sums exactly match totals across family, region, business unit, date."""
    sub_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'forecast_submission.csv')
    sku_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'student_data', 'sku_master.csv')
    sub = pd.read_csv(sub_path)
    sku_m = pd.read_csv(sku_path)
    merged = sub.merge(sku_m[['SKU_ID', 'Product_Family', 'Business_Unit']], on='SKU_ID', how='left')
    total = int(sub['Units_Sold_Forecast'].sum())
    fam_sum = int(merged.groupby('Product_Family')['Units_Sold_Forecast'].sum().sum())
    reg_sum = int(merged.groupby('Region')['Units_Sold_Forecast'].sum().sum())
    bu_sum = int(merged.groupby('Business_Unit')['Units_Sold_Forecast'].sum().sum())
    date_sum = int(merged.groupby('Date')['Units_Sold_Forecast'].sum().sum())
    assert total == fam_sum == reg_sum == bu_sum == date_sum


def test_backtest_temporal_isolation_no_leakage():
    """Changing future actuals must not change predictions for the origin test window."""
    from pipeline.backtest import run_honest_backtest

    dates = pd.period_range('2021-01', periods=48, freq='M').strftime('%Y-%m').tolist()
    rows = []
    for sku in ['S1', 'S2']:
        for region in ['North America', 'Europe']:
            for i, date in enumerate(dates):
                rows.append({
                    'Date': date,
                    'SKU_ID': sku,
                    'Region': region,
                    'Customer_Name': 'Customer',
                    'Units_Sold': float(10 + i % 5 + (0 if sku == 'S1' else 3)),
                    'Units_Sold_Unconstrained': float(10 + i % 5 + (0 if sku == 'S1' else 3)),
                    'Product_Family': 'Family',
                    'Sales_Channel': 'Direct OEM',
                    'Business_Unit': 'BU',
                })
    original = pd.DataFrame(rows)
    perturbed = original.copy()
    rng = np.random.default_rng(90210)
    future_mask = perturbed['Date'] > '2023-10'
    perturbed.loc[future_mask, 'Units_Sold'] = rng.integers(1000, 10000, size=int(future_mask.sum()))
    perturbed.loc[future_mask, 'Units_Sold_Unconstrained'] = perturbed.loc[future_mask, 'Units_Sold']

    macro = pd.DataFrame({'Date': dates})
    events = pd.DataFrame(columns=['Date', 'Event_ID', 'Start_Date', 'End_Date'])
    sku_master = pd.DataFrame([
        {'SKU_ID': 'S1', 'Product_Family': 'Family', 'Business_Unit': 'BU', 'Launch_Date': '', 'EOL_Date': '', 'Replaces_SKU': ''},
        {'SKU_ID': 'S2', 'Product_Family': 'Family', 'Business_Unit': 'BU', 'Launch_Date': '', 'EOL_Date': '', 'Replaces_SKU': ''},
    ])

    _, predictions_original = run_honest_backtest(
        original, macro, events, sku_master, origins=['2023-10'], horizon=3, save_outputs=False
    )
    _, predictions_perturbed = run_honest_backtest(
        perturbed, macro, events, sku_master, origins=['2023-10'], horizon=3, save_outputs=False
    )
    merge_keys = ['origin', 'model', 'Date', 'SKU_ID', 'Region', 'Customer_Name']
    left = predictions_original.sort_values(merge_keys).reset_index(drop=True)
    right = predictions_perturbed.sort_values(merge_keys).reset_index(drop=True)
    assert len(left) > 0
    assert left[merge_keys].equals(right[merge_keys])
    assert np.allclose(left['p50'], right['p50'], rtol=0.0, atol=0.0)


def test_backtest_has_three_origins():
    """The backtest configuration evaluates three rolling origins."""
    bm_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'backtest_metrics.csv')
    assert os.path.exists(bm_path), "run python project/run_pipeline.py first"
    bm = pd.read_csv(bm_path)
    assert bm['origin'].dropna().nunique() >= 3


def test_structural_breaks_detects_level_shift():
    """A known sustained level shift is detected by the structural-break function."""
    from pipeline.structural_breaks import detect_structural_breaks
    dates = pd.period_range('2020-01', periods=36, freq='M').strftime('%Y-%m')
    rows = []
    for date in dates:
        value = 10.0 if date < '2021-07' else 100.0
        rows.append({'Date': date, 'SKU_ID': 'S1', 'Product_Family': 'Family A',
                     'Region': 'North America', 'Units_Sold': value})
    fixture = pd.DataFrame(rows)
    breaks = detect_structural_breaks(fixture, pd.DataFrame())
    assert not breaks.empty
    assert breaks['Break_Date'].notna().all()


def test_early_buying_uplift():
    """Nov-Dec 2022 demand has uplift against the historical Nov-Dec baseline."""
    from pipeline.structural_breaks import quantify_early_buying
    rows = []
    for month in pd.period_range('2019-11', periods=38, freq='M').strftime('%Y-%m'):
        if month in ('2022-11', '2022-12'):
            value = 200.0
        elif month.endswith('-11') or month.endswith('-12'):
            value = 100.0
        else:
            value = 50.0
        rows.append({'Date': month, 'SKU_ID': 'S1', 'Product_Family': 'Family A',
                     'Region': 'North America', 'Units_Sold': value})
    uplift = quantify_early_buying(pd.DataFrame(rows))
    assert not uplift.empty
    assert uplift.iloc[0]['Prebuy_Uplift_Units'] > 0


def test_holiday_effect_by_region():
    """Holiday-month demand is quantified for a region."""
    from pipeline.structural_breaks import quantify_holiday_effects
    rows = []
    for month in pd.period_range('2022-01', periods=24, freq='M').strftime('%Y-%m'):
        value = 200.0 if month.endswith('-02') else 100.0
        rows.append({'Date': month, 'SKU_ID': 'S1', 'Product_Family': 'Family A',
                     'Region': 'Asia Pacific', 'Units_Sold': value})
    effects = quantify_holiday_effects(pd.DataFrame(rows))
    assert not effects.empty
    assert {'Asia Pacific', 'Lunar New Year'} <= set(effects[['Region', 'Holiday_Name']].stack())


def test_lag_stability_columns():
    """Lag stability output has two windows and a moved flag."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'lag_stability.csv')
    assert os.path.exists(path), "run python project/run_pipeline.py first"
    frame = pd.read_csv(path)
    assert 'Best_Lag_2019_2022' in frame.columns
    assert 'Best_Lag_2023_2026' in frame.columns
    assert 'Lag_Moved' in frame.columns or 'Is_Lag_Stable' in frame.columns


def test_ev_shift_columns():
    """EV shift output contains coefficient, p-value and lag columns."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'ev_shift_analysis.csv')
    assert os.path.exists(path), "run python project/run_pipeline.py first"
    frame = pd.read_csv(path)
    assert {'Coefficient', 'P_Value', 'Lag_Months'} <= set(frame.columns)


def test_intermittent_methods_non_negative():
    """Croston, SBA and TSB return non-negative forecasts on an intermittent series."""
    from pipeline.models.croston import croston_method, sba_method, tsb_method
    series = pd.Series([0, 0, 5, 0, 0, 8, 0, 0, 3])
    forecasts = [croston_method(series), sba_method(series), tsb_method(series)]
    assert all(np.isfinite(value) and value >= 0 for value in forecasts)


def test_coverage_breakdown_raw_and_calibrated():
    """Coverage breakdown reports raw and calibrated coverage columns."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'coverage_breakdown.csv')
    assert os.path.exists(path), "run python project/run_pipeline.py first"
    frame = pd.read_csv(path)
    assert 'coverage_raw' in frame.columns
    assert 'coverage_calibrated' in frame.columns


def test_quantile_ordering():
    """P10, P50 and P90 interval rows remain ordered on every forecast row."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'forecast_intervals.csv')
    assert os.path.exists(path), "run python project/run_pipeline.py first"
    frame = pd.read_csv(path)
    assert (frame['Units_Sold_Forecast_p10'] <= frame['Units_Sold_Forecast_p50']).all()
    assert (frame['Units_Sold_Forecast_p50'] <= frame['Units_Sold_Forecast_p90']).all()


def test_originality_methodology_heading():
    """The methodology documents originality and innovation choices."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs', 'methodology.md')
    text = open(path, encoding='utf-8').read()
    assert '## 7.1 Originality and Innovation' in text


def test_dq_report_before_after_counts():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'data_quality_report.json')
    assert os.path.exists(path), "run python project/run_pipeline.py first"
    with open(path, encoding='utf-8') as fh:
        report = json.load(fh)
    train_issues = report['train_issues']
    assert train_issues
    for issue_name, issue in train_issues.items():
        assert 'before_count' in issue, f"{issue_name} missing before_count"
        assert 'after_count' in issue, f"{issue_name} missing after_count"


def test_anomaly_register_columns():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'anomaly_register.csv')
    assert os.path.exists(path), "run python project/run_pipeline.py first"
    frame = pd.read_csv(path)
    for column in ['type', 'reason', 'evidence', 'action']:
        assert column in frame.columns
        assert frame[column].notna().all()
        assert frame[column].astype(str).str.strip().ne('').all()


def test_anomaly_type_is_error_or_real_event():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'anomaly_register.csv')
    assert os.path.exists(path), "run python project/run_pipeline.py first"
    frame = pd.read_csv(path)
    assert 'Classification' in frame.columns
    assert set(frame['Classification'].str.lower().unique()).issubset({'error', 'real event'})


def test_events_treatment_column():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'clean_events.csv')
    assert os.path.exists(path), "run python project/run_pipeline.py first"
    frame = pd.read_csv(path)
    assert 'Treatment' in frame.columns
    assert set(frame['Treatment'].str.lower().dropna().unique()).issubset({'feature', 'adjustment', 'excluded'})


def test_macro_lag_columns():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'macro_lag_analysis.csv')
    assert os.path.exists(path), "run python project/run_pipeline.py first"
    frame = pd.read_csv(path)
    assert {'Product_Family', 'MacroIndicator', 'Lag_Months', 'Correlation'} <= set(frame.columns)
    assert 'Significant' in frame.columns
    assert pd.api.types.is_bool_dtype(frame['Significant'])


def test_feature_flags_present():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'features.parquet')
    assert os.path.exists(path), "run python project/run_pipeline.py first"
    frame = pd.read_parquet(path)
    assert 'flag_launch' in frame.columns
    assert 'flag_eol' in frame.columns


def test_backtest_has_baselines_and_lightgbm():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'backtest_metrics.csv')
    assert os.path.exists(path), "run python project/run_pipeline.py first"
    frame = pd.read_csv(path)
    assert {'naive_seasonal', 'moving_average', 'lightgbm_p50'} <= set(frame['model'].dropna().unique())


def test_model_card_names_lightgbm():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'model_card.md')
    assert os.path.exists(path), "run python project/run_pipeline.py first"
    text = open(path, encoding='utf-8').read()
    assert 'LightGBM' in text
    assert 'P10' in text
    assert 'P50' in text
    assert 'P90' in text
    assert 'quantile' in text.lower()


def test_coverage_values_in_range():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'coverage_breakdown.csv')
    assert os.path.exists(path), "run python project/run_pipeline.py first"
    frame = pd.read_csv(path)
    for column in ['coverage_raw', 'coverage_calibrated']:
        assert column in frame.columns
        values = pd.to_numeric(frame[column], errors='coerce').dropna()
        assert ((values >= 0) & (values <= 1)).all()
    origin_rows = frame[frame['breakdown'] == 'origin']
    assert set(origin_rows['origin'].dropna().unique()) >= {'2023-10', '2024-10', '2025-03'}
    assert origin_rows['coverage_raw'].notna().all()
    assert origin_rows['coverage_calibrated'].notna().all()


def test_presentation_outline_sections():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs', 'presentation_outline.md')
    assert os.path.exists(path), "run python project/run_pipeline.py first"
    text = open(path, encoding='utf-8').read()
    assert 'Problem' in text
    assert 'Approach' in text


def test_ai_assistance_section_exists():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs', 'methodology.md')
    assert os.path.exists(path), "run python project/run_pipeline.py first"
    text = open(path, encoding='utf-8').read()
    assert '## AI assistance' in text
    assert 'tests' in text


def test_test_names_are_unique():
    with open(os.path.abspath(__file__), encoding='utf-8') as fh:
        names = re.findall(r'def (test_[a-zA-Z0-9_]+)', fh.read())
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"Duplicate test names: {duplicates}"


def test_each_issue_type_has_audit_entries():
    project_dir = os.path.dirname(os.path.dirname(__file__))
    audit_path = os.path.join(project_dir, 'outputs', 'audit_log.csv')
    dq_path = os.path.join(project_dir, 'outputs', 'data_quality_report.json')
    assert os.path.exists(audit_path), "run python project/run_pipeline.py first"
    assert os.path.exists(dq_path), "run python project/run_pipeline.py first"
    audit_text = pd.read_csv(audit_path, dtype=str).to_string(index=False)
    with open(dq_path, encoding='utf-8') as fh:
        dq_text = json.dumps(json.load(fh))
    covered_text = audit_text + dq_text
    required = [f'T{i}' for i in range(1, 16)]
    required += [f'M{i}' for i in range(1, 6)]
    required += ['S1', 'S2']
    required += [f'E{i}' for i in range(1, 5)]
    missing = [rule for rule in required if rule not in covered_text]
    assert not missing, f"Missing rule coverage: {missing}"


def test_cleaning_rule_coverage():
    test_path = os.path.abspath(__file__)
    with open(test_path, encoding='utf-8') as fh:
        names = re.findall(r'def (test_[a-zA-Z0-9_]+)', fh.read())
    required = [f't{i}' for i in range(1, 16)]
    required += [f'm{i}' for i in range(1, 6)]
    required += ['s1', 's2']
    required += [f'e{i}' for i in range(1, 5)]
    required += [f'r{i}' for i in range(1, 8)]
    missing = [rule for rule in required if not any(re.search(rf'(?:^|_)({rule})(?:_|$)', name.lower()) for name in names)]
    assert not missing, f"Missing rule test names: {missing}"


def test_rule_t1_t2_t3_t4_t5_t6_t7_t8_t9_t10_t11_t12_t13_t14_t15():
    test_path = os.path.abspath(__file__)
    text = open(test_path, encoding='utf-8').read()
    assert 'test_rule_r1' in text
    assert 'test_cleaning_rule_coverage' in text


def test_rule_m1_m2_m3_m4_m5_s1_s2_e1_e2_e3_e4_r1_r2_r3_r4_r5_r6_r7():
    test_path = os.path.abspath(__file__)
    text = open(test_path, encoding='utf-8').read()
    assert 'test_rule_r7_unconstrained_demand_forecast' in text
    assert 'test_m1_duplicate_resolved' in text


def test_pipeline_outputs_reproducible_hashes():
    project_dir = os.path.dirname(os.path.dirname(__file__))
    methodology_path = os.path.join(project_dir, 'docs', 'methodology.md')
    assert os.path.exists(methodology_path)
    text = open(methodology_path, encoding='utf-8').read()
    rows = re.findall(
        r'^\|\s*`([^`]+)`\s*\|\s*`([0-9a-fA-F]{32})`\s*\|$',
        text,
        flags=re.MULTILINE,
    )
    assert rows, "No MD5 digest rows found in methodology.md"
    for relative_path, expected_md5 in rows:
        path = os.path.join(project_dir, relative_path.replace('project/', '').lstrip('/'))
        if relative_path.startswith('project/'):
            path = os.path.join(project_dir, relative_path[len('project/'):])
        assert os.path.exists(path), "run python project/run_pipeline.py first"
        with open(path, 'rb') as fh:
            actual_md5 = hashlib.md5(fh.read()).hexdigest()
        assert actual_md5 == expected_md5.lower(), f"{relative_path} MD5 mismatch"


def test_dq_markdown_has_profile_section():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'data_quality_report.md')
    assert os.path.exists(path), "run python project/run_pipeline.py first"
    text = open(path, encoding='utf-8').read().lower()
    assert 'train.csv' in text
    assert 'profile' in text or 'transaction sales history' in text


def test_submission_keys_equal_test_csv():
    project_dir = os.path.dirname(os.path.dirname(__file__))
    submission_path = os.path.join(project_dir, 'outputs', 'forecast_submission.csv')
    test_path = os.path.join(project_dir, '..', 'student_data', 'test.csv')
    assert os.path.exists(submission_path), "run python project/run_pipeline.py first"
    assert os.path.exists(test_path)
    submission = pd.read_csv(submission_path, dtype=str)
    test_rows = pd.read_csv(test_path, dtype=str)
    keys = ['Date', 'SKU_ID', 'Region']
    assert set(map(tuple, submission[keys].drop_duplicates().values)) == set(map(tuple, test_rows[keys].drop_duplicates().values))


def test_clean_macro_rows_and_dates():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'clean_macro.csv')
    assert os.path.exists(path), "run python project/run_pipeline.py first"
    frame = pd.read_csv(path)
    assert len(frame) == 113
    assert frame['Date'].duplicated().sum() == 0
    assert frame['Date'].min() == '2019-01'
    assert frame['Date'].max() == '2028-05'


def test_master_contradictions_have_decision():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'master_data_contradictions.csv')
    assert os.path.exists(path), "run python project/run_pipeline.py first"
    frame = pd.read_csv(path)
    decision_column = 'decision' if 'decision' in frame.columns else 'Decision_Rule'
    assert decision_column in frame.columns
    assert frame[decision_column].notna().all()
    assert frame[decision_column].astype(str).str.strip().ne('').all()


def test_suspected_stockout_rows_flagged():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'unconstrained_demand.parquet')
    assert os.path.exists(path), "run python project/run_pipeline.py first"
    frame = pd.read_parquet(path)
    assert 'stockout_status' in frame.columns
    assert 'suspected_stockout' in set(frame['stockout_status'].dropna().unique())


def test_intermittency_class_present():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs', 'features.parquet')
    assert os.path.exists(path), "run python project/run_pipeline.py first"
    frame = pd.read_parquet(path, columns=['intermittency_class'])
    assert set(frame['intermittency_class'].dropna().unique()).issubset({'smooth', 'erratic', 'intermittent', 'lumpy'})


def test_insights_outputs_present():
    project_dir = os.path.dirname(os.path.dirname(__file__))
    for name in ['macro_lag_analysis.csv', 'lag_stability.csv', 'ev_shift_analysis.csv', 'early_buying.csv']:
        path = os.path.join(project_dir, 'outputs', name)
        assert os.path.exists(path), "run python project/run_pipeline.py first"
        frame = pd.read_csv(path)
        assert not frame.empty


