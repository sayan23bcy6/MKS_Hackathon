"""
models/lightgbm_model.py – WS4 main model: LightGBM with full feature set.
Includes P10/P50/P90 quantile forecasts.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

FEATURE_COLS = [
    'month', 'quarter', 'year', 'month_sin', 'month_cos',
    'flag_lny', 'flag_ramadan', 'flag_summer_shutdown',
    'flag_year_end', 'flag_q1_start',
    'flag_price_change', 'flag_supply_constraint',
    'flag_disruption', 'flag_system_event', 'flag_early_buy_window',
    'flag_orion_retool',
    'months_since_launch', 'months_to_eol',
    'is_replacement_sku', 'has_replacement',
    'intermittency_code',
    'SKU_ID_enc', 'Region_enc', 'Customer_Name_enc',
    'Product_Family_enc', 'Sales_Channel_enc',
    'lag_1m', 'lag_2m', 'lag_3m', 'lag_6m', 'lag_12m',
    'roll_mean_3m', 'roll_mean_6m', 'roll_mean_12m',
    'roll_std_3m', 'roll_std_6m',
    'lag_12m_yoy',
]

MACRO_FEATURE_COLS = [
    f'macro_{col}_lag{lag}'
    for col in [
        'ISM_Manufacturing_PMI', 'Auto_Production_YoY_Pct',
        'Vehicle_Inventory_Days', 'Raw_Material_Cost_Index',
        'New_Vehicle_Orders_Index', 'Industrial_Util_Rate_Pct',
        'Fuel_Price_Index', 'EV_Adoption_Index'
    ]
    for lag in [0, 1, 2, 3]
]

TARGET_COL = 'Units_Sold_Unconstrained'


def _get_features(df: pd.DataFrame) -> list:
    available = FEATURE_COLS + MACRO_FEATURE_COLS
    return [c for c in available if c in df.columns]


def train_lgbm(train_df: pd.DataFrame,
               val_df: pd.DataFrame,
               quantile: float = 0.5) -> 'lgb.Booster | None':
    """Train a LightGBM quantile regression model."""
    if not HAS_LGB:
        return None

    features = _get_features(train_df)

    # Filter: only rows with non-null, non-zero target (real demand rows)
    train_valid = train_df[train_df[TARGET_COL].notna()].copy()
    val_valid = val_df[val_df[TARGET_COL].notna()].copy()

    X_train = train_valid[features].fillna(-1)
    y_train = train_valid[TARGET_COL].clip(lower=0)
    X_val = val_valid[features].fillna(-1)
    y_val = val_valid[TARGET_COL].clip(lower=0)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    params = {
        'objective': 'quantile',
        'alpha': quantile,
        'metric': 'quantile',
        'learning_rate': 0.08,
        'num_leaves': 63,
        'min_child_samples': 30,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42,
    }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=300,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
    )
    return model


def predict_lgbm(model, test_df: pd.DataFrame) -> np.ndarray:
    """Generate predictions from a trained LightGBM model."""
    if model is None:
        return np.zeros(len(test_df))
    features = _get_features(test_df)
    X = test_df[features].fillna(-1)
    return model.predict(X)


def train_all_quantiles(train_df: pd.DataFrame,
                         val_df: pd.DataFrame) -> dict:
    """Train P10, P50, P90 models."""
    models = {}
    for q, name in [(0.1, 'p10'), (0.5, 'p50'), (0.9, 'p90')]:
        print(f"  Training LightGBM {name} (quantile={q})...")
        m = train_lgbm(train_df, val_df, quantile=q)
        models[name] = m
    return models


def compute_wape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Weighted Absolute Percentage Error."""
    denom = y_true.sum()
    if denom == 0:
        return np.nan
    return float((y_true - y_pred).abs().sum() / denom)


def compute_bias(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean bias: positive = over-forecast."""
    if len(y_true) == 0:
        return np.nan
    return float((y_pred - y_true).mean())

