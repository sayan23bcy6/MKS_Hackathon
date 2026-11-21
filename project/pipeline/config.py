"""
config.py - Central configuration: paths, canonical label maps, thresholds, rule IDs.
All cleaning rules reference constants defined here.
"""

import os
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT.parent / "student_data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RAW_TRAIN      = RAW_DIR / "train.csv"
RAW_TEST       = RAW_DIR / "test.csv"
RAW_MACRO      = RAW_DIR / "macro_indicators.csv"
RAW_SKU_MASTER = RAW_DIR / "sku_master.csv"
RAW_EVENTS     = RAW_DIR / "events_calendar.csv"
RAW_STOCKOUTS  = RAW_DIR / "inventory_stockouts.csv"
SAMPLE_SUB     = RAW_DIR / "sample_submission.csv"

OUT_TRAIN_CLEAN    = OUTPUT_DIR / "clean_train.parquet"
OUT_MACRO_CLEAN    = OUTPUT_DIR / "clean_macro.csv"
OUT_EVENTS_CLEAN   = OUTPUT_DIR / "clean_events.csv"
OUT_AUDIT_LOG      = OUTPUT_DIR / "audit_log.csv"
OUT_DQ_JSON        = OUTPUT_DIR / "data_quality_report.json"
OUT_DQ_MD          = OUTPUT_DIR / "data_quality_report.md"
OUT_ANOMALY_REG    = OUTPUT_DIR / "anomaly_register.csv"
OUT_UNCONSTRAINED  = OUTPUT_DIR / "unconstrained_demand.parquet"
OUT_FEATURES       = OUTPUT_DIR / "features.parquet"
OUT_BACKTEST       = OUTPUT_DIR / "backtest_metrics.csv"
OUT_FORECAST_SUB   = OUTPUT_DIR / "forecast_submission.csv"
OUT_FORECAST_INT   = OUTPUT_DIR / "forecast_intervals.csv"
OUT_MODEL_CARD     = OUTPUT_DIR / "model_card.md"
OUT_HIERARCHY_RECON = OUTPUT_DIR / "hierarchy_reconciliation.csv"
OUT_EV_SHIFT        = OUTPUT_DIR / "ev_shift_analysis.csv"
OUT_MACRO_LAG       = OUTPUT_DIR / "macro_lag_analysis.csv"
OUT_SCENARIO        = OUTPUT_DIR / "whatif_scenario_forecast.csv"
OUT_WHATIF_SCENARIOS = OUTPUT_DIR / "whatif_scenarios.csv"
OUT_EVENT_ESTIMATES = OUTPUT_DIR / "event_impact_estimates.csv"
OUT_STOCKOUTS       = OUTPUT_DIR / "inventory_stockouts.csv"
OUT_PANEL_STATUS         = OUTPUT_DIR / "panel_status.csv"
OUT_MASTER_CONTRADICTIONS = OUTPUT_DIR / "master_data_contradictions.csv"
OUT_EARLY_BUYING         = OUTPUT_DIR / "early_buying.csv"
OUT_HOLIDAY_EFFECTS      = OUTPUT_DIR / "holiday_effects.csv"
OUT_STRUCTURAL_BREAKS    = OUTPUT_DIR / "structural_breaks.csv"
OUT_CUSTOMER_SWITCHES    = OUTPUT_DIR / "customer_switches.csv"
OUT_LOST_ACCOUNTS        = OUTPUT_DIR / "lost_accounts.csv"
OUT_LAG_STABILITY        = OUTPUT_DIR / "lag_stability.csv"
OUT_COVERAGE_BREAKDOWN   = OUTPUT_DIR / "coverage_breakdown.csv"

# --- T5: Region canonical map ---
# Strip + casefold applied first, then this dict (keys are casefolded+stripped)
REGION_MAP = {
    # India variants (all casefolded+stripped; hyphens already replaced with spaces in normaliser)
    "india": "India",
    # Asia Pacific variants
    "apac": "Asia Pacific",
    "asia pacific": "Asia Pacific",
    "asia pacific ": "Asia Pacific",  # trailing space
    # Europe variants
    "eu": "Europe",
    "europe": "Europe",
    "europe ": "Europe",
    # Latin America variants
    "latin america": "Latin America",
    "latin america ": "Latin America",
    "latam": "Latin America",
    # North America variants
    "north america": "North America",
    "north america ": "North America",
    "n. america": "North America",
    "na": "North America",
    # Middle East & Africa variants (& encoded as-is after the normaliser preserves &)
    "middle east & africa": "Middle East & Africa",
    "middle east &amp; africa": "Middle East & Africa",
    "middle east and africa": "Middle East & Africa",
    "me&a": "Middle East & Africa",
    "mea": "Middle East & Africa",
    "middle east & africa ": "Middle East & Africa",
    "unassigned": "Unassigned",
}

CANONICAL_REGIONS = {
    "North America",
    "Europe",
    "Asia Pacific",
    "India",
    "Latin America",
    "Middle East & Africa",
    "Unassigned",
}

# --- T6: Sales_Channel canonical map ---
CHANNEL_MAP = {
    "direct oem": "Direct OEM",
    "direct-oem": "Direct OEM",
    "oem direct": "Direct OEM",
    "tier-1 distributor": "Tier-1 Distributor",
    "tier 1 distributor": "Tier-1 Distributor",
    "t1 distributor": "Tier-1 Distributor",
    "aftermarket": "Aftermarket",
    "after-market": "Aftermarket",
    "aftermarket": "Aftermarket",   # already canonical, included for completeness
    "export": "Export",
}

CANONICAL_CHANNELS = {
    "Direct OEM",
    "Tier-1 Distributor",
    "Aftermarket",
    "Export",
}

# --- T7: SKU prefix -> Product_Family ---
PREFIX_TO_FAMILY = {
    "HYD": "Hydraulic Systems",
    "PTC": "Powertrain Components",
    "PMP": "Precision Machined Parts",
    "CSS": "Chassis & Structural",
    "BSS": "Brake & Safety Systems",
    "ELA": "Electrical Assemblies",
    "THM": "Thermal Management",
}

# --- T10: Customer_Name canonical map ---
CANONICAL_CUSTOMERS = [
    "Vantara Auto",
    "Helios Auto Parts",
    "Orion Driveline",
    "Quantum Drive Systems",
    "Meridian Assembly",
    "Apex Tier One",
    "Solaris Automotive",
    "Nexon Motor Group",
    "Crestline Manufacturing",
    "Atlas Mideast Components",
    "Falcon Gulf Motors",
    "Kestrel Mobility",
]

# Case-folded map built dynamically
CUSTOMER_MAP = {c.casefold(): c for c in CANONICAL_CUSTOMERS}

# --- Lifecycle status values ---
ACTIVE_STATUSES = {"Active", "EOL Announced"}
PLANNED_LAUNCH_STATUS = "Planned Launch"
DISCONTINUED_STATUSES = {"Discontinued", "EOL Announced"}

# --- Thresholds ---
TRAIN_DATE_MIN = "2019-01"
TRAIN_DATE_MAX = "2026-10"
FORECAST_DATE_MIN = "2026-11"
FORECAST_DATE_MAX = "2028-05"

PRICE_SCALE_100X_LOW  = 95.0   # x100 band
PRICE_SCALE_100X_HIGH = 120.0
PRICE_SCALE_85X_LOW   = 80.0   # x85 band (possible INR)
PRICE_SCALE_85X_HIGH  = 95.0
PRICE_SCALE_MED_LOW   = 5.0    # 5-20x band - repaired using annual median reference price
PRICE_SCALE_MED_HIGH  = 20.0

UNITS_LARGE_THRESHOLD = 5000   # T11: flag for review

PMI_MIN = 30.0
PMI_MAX = 70.0
PMI_DECIMAL_SHIFT_THRESHOLD = 10.0  # below this, check x10

# --- Macro correction windows ---
# M2: 2019 rows are fractions, multiply by 100
MACRO_PCT_FRACTION_YEAR_MAX = 2019

# --- Audit log columns ---
AUDIT_COLS = [
    "file", "row_key", "field", "raw_value", "new_value", "rule_id", "reason"
]

# --- Month name map (for T1) ---
MONTH_NAME_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}

# --- Forecast Event Adjustment Rules ---
EVENT_FORECAST_RULES = {
    'EV-062': {
        'effect': 'zero',
        'scope_customer': 'Orion Driveline',
        'scope_region': 'North America',
        'source': 'events_calendar',
        'description': 'Planned retooling shutdown at Orion Driveline North American plants',
    },
    'EV-052': {
        'effect': 'model_feature',
        'factor': 1.0,
        'scope_customer': 'Helios Auto Parts',
        'scope_region': 'India',
        'source': 'historical_learning_model_feature',
        'description': 'Helios Auto Parts capacity increase at India plant captured natively by tree model through customer-region features and lag dynamics without post-forecast multiplier to prevent double counting',
    },
    'EV-059': {
        'effect': 'model_feature',
        'factor': 1.0,
        'scope_customer': None,
        'scope_region': None,
        'source': 'native_model_feature',
        'description': 'Global price rise driven natively by model feature flag_price_change without post-hoc multiplier to prevent double counting',
    },
}
