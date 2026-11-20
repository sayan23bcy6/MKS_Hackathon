"""
lightgbm_model.py - Quantile gradient boosting demand forecast model.
Trains P10, P50, and P90 models with early stopping validation and full-history refit.
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
except ImportError as exc:
    raise RuntimeError("lightgbm is required for the pipeline model step. Please install lightgbm.") from exc

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
    'order_cycle_code', 'is_lost_account',
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


def get_feature_names(df: pd.DataFrame) -> list:
    """Return available feature columns in dataframe order."""
    available = FEATURE_COLS + MACRO_FEATURE_COLS
    return [c for c in available if c in df.columns]


def train_single_quantile(train_df: pd.DataFrame,
                          val_df: pd.DataFrame,
                          full_df: pd.DataFrame = None,
                          quantile: float = 0.5) -> lgb.Booster:
    """
    Train a LightGBM quantile regression model.
    Uses validation set to determine early stopping best_iteration,
    then optionally refits on the full dataset (F1).
    """
    features = get_feature_names(train_df)

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
        'deterministic': True,
        'force_row_wise': True,
    }

    val_model = lgb.train(
        params,
        dtrain,
        num_boost_round=300,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
    )

    best_iter = val_model.best_iteration
    if best_iter is None or best_iter <= 0:
        best_iter = 150

    # Refit on full historical dataset if provided (Fix F1)
    if full_df is not None and len(full_df) > 0:
        full_valid = full_df[full_df[TARGET_COL].notna()].copy()
        X_full = full_valid[features].fillna(-1)
        y_full = full_valid[TARGET_COL].clip(lower=0)
        dfull = lgb.Dataset(X_full, label=y_full)

        final_model = lgb.train(
            params,
            dfull,
            num_boost_round=best_iter,
            callbacks=[lgb.log_evaluation(period=-1)],
        )
        return final_model

    return val_model


def train_all_quantiles(train_df: pd.DataFrame,
                         val_df: pd.DataFrame,
                         full_df: pd.DataFrame = None) -> dict:
    """Train P10, P50, and P90 quantile models."""
    models = {}
    for q, name in [(0.1, 'p10'), (0.5, 'p50'), (0.9, 'p90')]:
        print(f"  Training LightGBM {name} (quantile={q})...")
        m = train_single_quantile(train_df, val_df, full_df=full_df, quantile=q)
        models[name] = m
    return models


def predict_lgbm(model: lgb.Booster, test_df: pd.DataFrame) -> np.ndarray:
    """Generate predictions from a trained LightGBM booster."""
    if model is None:
        raise ValueError("Model is None; cannot generate predictions.")
    features = get_feature_names(test_df)
    X = test_df[features].fillna(-1)
    return model.predict(X)


def enforce_quantile_ordering(p10: np.ndarray, p50: np.ndarray, p90: np.ndarray):
    """Enforce non-crossing constraint: P10 <= P50 <= P90."""
    p10_ordered = np.minimum(p10, p50)
    p90_ordered = np.maximum(p90, p50)
    p50_ordered = np.clip(p50, p10_ordered, p90_ordered)
    return p10_ordered, p50_ordered, p90_ordered


def compute_empirical_coverage(y_true: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> float:
    """Compute proportion of ground truth values that fall within the P10-P90 interval."""
    mask = (y_true >= p10) & (y_true <= p90)
    return float(np.mean(mask)) if len(mask) > 0 else np.nan


def compute_wape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Weighted Absolute Percentage Error."""
    denom = y_true.sum()
    if denom == 0:
        return np.nan
    return float((y_true - y_pred).abs().sum() / denom)


def compute_bias(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean forecast bias (positive indicates over-forecasting)."""
    if len(y_true) == 0:
        return np.nan
    return float((y_pred - y_true).mean())
