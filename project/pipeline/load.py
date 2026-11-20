"""
load.py – Read raw files only. No transformations.
Returns DataFrames with all columns as strings.
"""

import pandas as pd
from pipeline.config import (
    RAW_TRAIN, RAW_TEST, RAW_MACRO,
    RAW_SKU_MASTER, RAW_EVENTS, RAW_STOCKOUTS, SAMPLE_SUB
)


def load_train() -> pd.DataFrame:
    """Load raw train.csv – all columns as str, no coercions."""
    return pd.read_csv(RAW_TRAIN, dtype=str, low_memory=False)


def load_test() -> pd.DataFrame:
    """Load raw test.csv – all columns as str."""
    return pd.read_csv(RAW_TEST, dtype=str, low_memory=False)


def load_macro() -> pd.DataFrame:
    """Load raw macro_indicators.csv – all columns as str."""
    return pd.read_csv(RAW_MACRO, dtype=str)


def load_sku_master() -> pd.DataFrame:
    """Load raw sku_master.csv – all columns as str."""
    return pd.read_csv(RAW_SKU_MASTER, dtype=str)


def load_events() -> pd.DataFrame:
    """Load raw events_calendar.csv – all columns as str."""
    return pd.read_csv(RAW_EVENTS, dtype=str)


def load_stockouts() -> pd.DataFrame:
    """Load raw inventory_stockouts.csv – all columns as str."""
    return pd.read_csv(RAW_STOCKOUTS, dtype=str, low_memory=False)


def load_sample_submission() -> pd.DataFrame:
    """Load sample_submission.csv – all columns as str."""
    return pd.read_csv(SAMPLE_SUB, dtype=str)

