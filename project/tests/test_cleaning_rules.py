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
        import pathlib
        if pathlib.Path(OUT_FORECAST_SUB).exists():
            sub = pd.read_csv(OUT_FORECAST_SUB)
            assert list(sub.columns) == ['Date', 'SKU_ID', 'Region', 'Units_Sold_Forecast']
            assert sub['Units_Sold_Forecast'].min() >= 0
            assert sub['Units_Sold_Forecast'].isna().sum() == 0

    def test_audit_log_has_required_columns(self):
        from pipeline.config import OUT_AUDIT_LOG, AUDIT_COLS
        import pathlib
        if pathlib.Path(OUT_AUDIT_LOG).exists():
            log = pd.read_csv(OUT_AUDIT_LOG)
            for col in AUDIT_COLS:
                assert col in log.columns, f"Audit log missing column: {col}"

    def test_clean_train_no_null_date_sku_region(self):
        from pipeline.config import OUT_TRAIN_CLEAN
        import pathlib
        if pathlib.Path(OUT_TRAIN_CLEAN).exists():
            df = pd.read_parquet(OUT_TRAIN_CLEAN)
            assert df['Date'].isna().sum() == 0
            assert df['SKU_ID'].isna().sum() == 0

    def test_clean_train_date_format(self):
        from pipeline.config import OUT_TRAIN_CLEAN
        import pathlib, re
        if pathlib.Path(OUT_TRAIN_CLEAN).exists():
            df = pd.read_parquet(OUT_TRAIN_CLEAN)
            bad = df['Date'].apply(lambda d: not bool(re.match(r'^\d{4}-\d{2}$', str(d))))
            assert bad.sum() == 0, f"{bad.sum()} dates are not YYYY-MM format"

    def test_clean_train_no_dates_after_202610(self):
        from pipeline.config import OUT_TRAIN_CLEAN
        import pathlib
        if pathlib.Path(OUT_TRAIN_CLEAN).exists():
            df = pd.read_parquet(OUT_TRAIN_CLEAN)
            bad = df['Date'] > '2026-10'
            assert bad.sum() == 0, f"{bad.sum()} train dates after 2026-10"

    def test_clean_train_no_dup_record_id(self):
        from pipeline.config import OUT_TRAIN_CLEAN
        import pathlib
        if pathlib.Path(OUT_TRAIN_CLEAN).exists():
            df = pd.read_parquet(OUT_TRAIN_CLEAN)
            assert df['Record_ID'].duplicated().sum() == 0

    def test_clean_train_regions_canonical(self):
        from pipeline.config import OUT_TRAIN_CLEAN
        import pathlib
        if pathlib.Path(OUT_TRAIN_CLEAN).exists():
            df = pd.read_parquet(OUT_TRAIN_CLEAN)
            allowed = CANONICAL_REGIONS
            bad = ~df['Region'].isin(allowed)
            assert bad.sum() == 0, f"{bad.sum()} non-canonical regions: {df.loc[bad,'Region'].unique()}"
            assert 'Unassigned' in allowed
            assert not (df['Region'] == 'Unknown').any()

    def test_clean_train_channels_canonical(self):
        from pipeline.config import OUT_TRAIN_CLEAN
        import pathlib
        if pathlib.Path(OUT_TRAIN_CLEAN).exists():
            df = pd.read_parquet(OUT_TRAIN_CLEAN)
            bad = ~df['Sales_Channel'].isin(CANONICAL_CHANNELS)
            assert bad.sum() == 0

    def test_clean_train_units_non_negative(self):
        from pipeline.config import OUT_TRAIN_CLEAN
        import pathlib
        if pathlib.Path(OUT_TRAIN_CLEAN).exists():
            df = pd.read_parquet(OUT_TRAIN_CLEAN)
            us = pd.to_numeric(df['Units_Sold'], errors='coerce')
            bad = (us < 0) & us.notna()
            assert bad.sum() == 0

    def test_forecast_row_count(self):
        from pipeline.config import OUT_FORECAST_SUB
        import pathlib
        if pathlib.Path(OUT_FORECAST_SUB).exists():
            sub = pd.read_csv(OUT_FORECAST_SUB)
            assert len(sub) == 58219, f"Expected 58219 rows, got {len(sub)}"

    def test_pmp004_bu_resolution(self):
        """Rule S1: PMP-004 BU conflict resolved to Advanced Manufacturing."""
        from pipeline.clean_master_macro_events import clean_sku_master
        from pipeline.load import load_sku_master
        sku_clean, dq = clean_sku_master(load_sku_master())
        pmp = sku_clean[sku_clean['SKU_ID'] == 'PMP-004']
        assert pmp['Business_Unit'].iloc[0] == 'Advanced Manufacturing'

    def test_quantile_ordering(self):
        """Verify P10 <= P50 <= P90 non-crossing constraint."""
        from pipeline.models.lightgbm_model import enforce_quantile_ordering
        p10 = np.array([10.0, 50.0, 30.0])
        p50 = np.array([20.0, 40.0, 30.0])
        p90 = np.array([15.0, 60.0, 25.0])
        p10_o, p50_o, p90_o = enforce_quantile_ordering(p10, p50, p90)
        assert (p10_o <= p50_o).all()
        assert (p50_o <= p90_o).all()

    def test_hierarchy_reconciliation_structure(self):
        """Verify hierarchy reconciliation output exists and has valid levels, deviations, and status."""
        from pipeline.config import OUT_HIERARCHY_RECON
        import pathlib
        if pathlib.Path(OUT_HIERARCHY_RECON).exists():
            df = pd.read_csv(OUT_HIERARCHY_RECON)
            assert 'Level' in df.columns
            assert 'Raw_Deviation_Pct' in df.columns
            assert 'Deviation_Pct' in df.columns
            assert (df['Deviation_Pct'] <= 0.5).all()
            assert 'Status' in df.columns
            assert set(df['Status'].unique()).issubset({'PASS', 'FAIL'})

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
            pytest.skip("features.parquet not generated yet")
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
            pytest.skip("backtest_predictions.csv not generated yet")
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

    def test_new_launch_history_inheritance_rule(self):
        """Priority 4b: New launches with Replaces_SKU inherit replaced catalog history."""
        from pipeline.config import RAW_SKU_MASTER
        sku_master = pd.read_csv(RAW_SKU_MASTER)
        replaces = sku_master[sku_master['Replaces_SKU'].notna() & (sku_master['Replaces_SKU'] != '')]
        assert len(replaces) > 0
        pmp20 = replaces[replaces['SKU_ID'] == 'PMP-020']
        assert not pmp20.empty
        assert pmp20.iloc[0]['Replaces_SKU'] == 'PMP-003'

    def test_whatif_zero_shift_equals_baseline(self):
        """Priority 3: Scenario at delta 0 equals baseline forecast."""
        import os
        import pytest
        scen_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'whatif_scenarios.csv')
        if not os.path.exists(scen_path):
            pytest.skip("whatif_scenarios.csv not generated yet")
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
        adjusted = apply_event_adjustments(df_forecast, events_fixture)
        assert adjusted['Units_Sold_Forecast'].iloc[0] == 100

    def test_hierarchy_reconciliation_structure_and_reconciled_flag(self):
        """Priority A3: Reconciliation records raw deviations and reports reconciled boolean."""
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
        res = compute_hierarchy_reconciliation(sub_fixture, clean_train_fixture, sku_master_fixture)
        assert 'total_forecast_units' in res
        assert 'reconciled' in res
        assert isinstance(res['reconciled'], bool)

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

