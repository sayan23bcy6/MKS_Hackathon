"""
run_pipeline.py – Main pipeline runner.
ONE COMMAND: python project/run_pipeline.py
Runs: clean → features → unconstrained → backtest → forecast → reports → dashboard data prep
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pipeline.config import (
    OUT_TRAIN_CLEAN, OUT_MACRO_CLEAN, OUT_EVENTS_CLEAN,
    OUT_AUDIT_LOG, OUT_FEATURES, OUT_BACKTEST,
    OUT_FORECAST_SUB, OUT_FORECAST_INT,
)
from pipeline.load import (
    load_train, load_test, load_macro,
    load_sku_master, load_events, load_stockouts
)
from pipeline.audit import audit
from pipeline.clean_train import clean_train
from pipeline.clean_master_macro_events import (
    clean_sku_master, clean_macro, clean_events
)
from pipeline.unconstrained import build_unconstrained_demand
from pipeline.features import build_features, compute_macro_lag_analysis
from pipeline.anomalies import build_anomaly_register
from pipeline.reports import generate_dq_report
from pipeline.models.baselines import naive_seasonal_forecast, moving_average_forecast
from pipeline.models.croston import apply_intermittent_forecasts
from pipeline.backtest import run_all_backtests
from pipeline.forecast import generate_forecast, compute_hierarchy_reconciliation, write_model_card

try:
    from pipeline.models.lightgbm_model import train_all_quantiles, predict_lgbm
    HAS_LGB = True
except Exception:
    HAS_LGB = False


def main():
    t0 = time.time()
    print("=" * 60)
    print("CELERIS AUTOMOBILI – DEMAND FORECASTING PIPELINE")
    print("=" * 60)

    # ── PHASE 1: LOAD RAW ─────────────────────────────────────────────────────
    print("\n[1/8] Loading raw files...")
    raw_train = load_train()
    raw_test = load_test()
    raw_macro = load_macro()
    raw_sku = load_sku_master()
    raw_events = load_events()
    raw_stockouts = load_stockouts()
    print(f"  train: {raw_train.shape}, test: {raw_test.shape}, macro: {raw_macro.shape}")
    print(f"  sku_master: {raw_sku.shape}, events: {raw_events.shape}, stockouts: {raw_stockouts.shape}")

    # ── PHASE 2: CLEAN MASTER FILES ───────────────────────────────────────────
    print("\n[2/8] Cleaning SKU master, macro indicators, events calendar...")
    clean_sku, dq_sku = clean_sku_master(raw_sku)
    clean_mac, dq_mac = clean_macro(raw_macro)
    clean_ev, dq_ev = clean_events(raw_events)

    clean_mac.to_csv(OUT_MACRO_CLEAN, index=False)
    clean_ev.to_csv(OUT_EVENTS_CLEAN, index=False)
    print(f"  Macro cleaned: {len(clean_mac)} rows, Events cleaned: {len(clean_ev)} rows")

    # ── PHASE 3: CLEAN TRAIN ──────────────────────────────────────────────────
    print("\n[3/8] Cleaning train.csv (T1–T15 + R5)...")
    df_clean, dq_train = clean_train(raw_train, clean_sku, raw_stockouts)

    # Save cleaned train
    df_clean.to_parquet(OUT_TRAIN_CLEAN, index=False)
    print(f"  Cleaned train: {len(df_clean)} rows (from {dq_train['raw_rows']})")

    # Save audit log
    n_audit = audit.save()
    print(f"  Audit log: {n_audit} entries saved to {OUT_AUDIT_LOG}")
    audit_df = pd.read_csv(OUT_AUDIT_LOG)

    # ── PHASE 4: UNCONSTRAINED DEMAND ─────────────────────────────────────────
    print("\n[4/8] Building unconstrained demand from stockouts...")
    df_unconstrained, match_rate = build_unconstrained_demand(df_clean, raw_stockouts)
    dq_train['stockout_match_rate_pct'] = match_rate

    # ── PHASE 5: ANOMALIES ────────────────────────────────────────────────────
    print("\n[5/8] Building anomaly register...")
    anomaly_df = build_anomaly_register(df_clean, clean_ev, raw_stockouts, audit_df)
    print(f"  Anomaly register: {len(anomaly_df)} entries")

    # ── PHASE 6: FEATURES ─────────────────────────────────────────────────────
    print("\n[6/8] Building features (WS3)...")
    features_df = build_features(df_unconstrained, clean_mac, clean_ev, clean_sku, df_unconstrained)
    print(f"  Features: {features_df.shape}")

    # Macro lag analysis
    lag_analysis = compute_macro_lag_analysis(df_clean, clean_mac)
    lag_analysis.to_csv(str(OUT_FEATURES).replace('features.parquet', 'macro_lag_analysis.csv'), index=False)
    print(f"  Macro lag analysis: {len(lag_analysis)} rows")

    # ── PHASE 7: FORECAST ─────────────────────────────────────────────────────
    print("\n[7/8] Training models and generating forecast...")

    # Prepare test features
    test_df = raw_test.copy()
    test_features_proper = _build_test_features(test_df, features_df, clean_mac, clean_ev, clean_sku)

    models = {}
    if HAS_LGB:
        # Time-based split: train up to 2025-03, validate 2025-04 to 2026-10
        TRAIN_CUT = '2025-03'
        VAL_START = '2025-04'
        train_set = features_df[features_df['Date'] <= TRAIN_CUT].copy()
        val_set = features_df[features_df['Date'] >= VAL_START].copy()

        # Only train on rows with real demand (non-null, non-zero is preferred but nulls excluded)
        TARGET_COL = 'Units_Sold_Unconstrained'
        train_set_clean = train_set[train_set[TARGET_COL].notna()].copy()
        val_set_clean = val_set[val_set[TARGET_COL].notna()].copy()

        print(f"  LightGBM train set: {len(train_set_clean)}, val set: {len(val_set_clean)}")
        models = train_all_quantiles(train_set_clean, val_set_clean)
    else:
        print("  LightGBM not available; using baseline models for forecast")

    # Backtest
    def _lgbm_forecast_fn(train_df, test_df):
        if HAS_LGB and models.get('p50'):
            from pipeline.models.lightgbm_model import _get_features
            feats = _get_features(test_df)
            X = test_df[feats].fillna(-1)
            preds = models['p50'].predict(X)
            result = test_df[['Date', 'SKU_ID', 'Region', 'Customer_Name']].copy()
            result['Units_Sold_Forecast'] = np.clip(preds, 0, None)
            return result
        else:
            return naive_seasonal_forecast(train_df, test_df)

    backtest_results = run_all_backtests(
        features_df,
        lambda tr, te: naive_seasonal_forecast(tr, te),
        lambda tr, te: moving_average_forecast(tr, te),
        _lgbm_forecast_fn,
    )
    print(f"  Backtest complete: {len(backtest_results)} metric rows")

    # Final forecast
    submission, intervals = generate_forecast(
        test_df=test_df,
        features_test=test_features_proper if test_features_proper is not None else test_df,
        models=models,
        sku_master=clean_sku,
        events=clean_ev,
        fallback_fn=lambda feats, test: naive_seasonal_forecast(features_df, test),
    )

    # Hierarchy reconciliation
    recon = compute_hierarchy_reconciliation(submission, df_clean)
    print(f"  Hierarchy reconciliation: total={recon['total_forecast_units']}")

    write_model_card(models, features=[], backtest_metrics=backtest_results)

    # ── PHASE 8: REPORTS ──────────────────────────────────────────────────────
    print("\n[8/8] Generating data-quality report...")
    dq_report = generate_dq_report(
        dq_train, dq_mac, dq_sku, dq_ev,
        audit_df, anomaly_df
    )

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"  Audit entries: {n_audit}")
    print(f"  Clean train rows: {len(df_clean)}")
    print(f"  Forecast rows: {len(submission)}")
    print(f"{'='*60}")
    print("\nOutputs:")
    for path in [OUT_TRAIN_CLEAN, OUT_MACRO_CLEAN, OUT_EVENTS_CLEAN,
                 OUT_AUDIT_LOG, str(OUT_FEATURES).replace('features.parquet', 'macro_lag_analysis.csv'),
                 OUT_BACKTEST, OUT_FORECAST_SUB, OUT_FORECAST_INT]:
        import pathlib
        p = pathlib.Path(str(path))
        if p.exists():
            print(f"  ✓ {p.name} ({p.stat().st_size/1024:.1f} KB)")
    return 0


def _build_test_features(test_df: pd.DataFrame,
                           train_features: pd.DataFrame,
                           clean_mac: pd.DataFrame,
                           clean_ev: pd.DataFrame,
                           clean_sku: pd.DataFrame) -> pd.DataFrame:
    """Build feature set for test rows using lag values from train history."""
    try:
        from pipeline.features import (
            _add_calendar_features, _add_event_features,
            _add_lifecycle_features, _add_macro_features,
            _add_sku_region_encodings
        )
        # Create test df with same structure as train features
        test = test_df.copy()
        test['Units_Sold'] = np.nan
        test['Units_Sold_Unconstrained'] = np.nan

        # Merge encodings from training set
        enc_cols = ['SKU_ID', 'Region', 'Customer_Name', 'Product_Family', 'Sales_Channel']
        # Use same encoding as training (merge category codes)
        for col in enc_cols:
            enc_col = f"{col}_enc"
            lookup = train_features.drop_duplicates(col)[[col, enc_col]] if enc_col in train_features.columns else None
            if lookup is not None:
                test = test.merge(lookup, on=col, how='left')
                test[enc_col] = test[enc_col].fillna(-1).astype(int)

        test = _add_calendar_features(test)
        test = _add_event_features(test, clean_ev)
        test = _add_lifecycle_features(test, clean_sku)
        test = _add_macro_features(test, clean_mac, lags=[0, 1, 2, 3])

        # Add lag features from last known history
        HIST_COLS = ['lag_1m', 'lag_2m', 'lag_3m', 'lag_6m', 'lag_12m',
                     'roll_mean_3m', 'roll_mean_6m', 'roll_mean_12m',
                     'roll_std_3m', 'roll_std_6m', 'lag_12m_yoy']
        # Get last known values per SKU-Region-Customer from training
        last_known = train_features.sort_values('Date').groupby(
            ['SKU_ID', 'Region', 'Customer_Name']
        ).last().reset_index()
        avail_hist = [col for col in HIST_COLS if col in last_known.columns]
        if avail_hist:
            test = test.merge(
                last_known[['SKU_ID', 'Region', 'Customer_Name'] + avail_hist],
                on=['SKU_ID', 'Region', 'Customer_Name'], how='left'
            )

        # Expert: New launches inherit history from Replaces_SKU
        replaces_map = dict(zip(clean_sku['SKU_ID'], clean_sku['Replaces_SKU']))
        for col in avail_hist:
            if col in test.columns and test[col].isna().any():
                rep_sku = test['SKU_ID'].map(replaces_map)
                rep_lookup = last_known.groupby(['SKU_ID', 'Region'])[col].mean().to_dict()
                rep_vals = [rep_lookup.get((s, r)) for s, r in zip(rep_sku, test['Region'])]
                test[col] = test[col].fillna(pd.Series(rep_vals, index=test.index))

        # Intermittency from training
        if 'intermittency_class' in train_features.columns and 'intermittency_code' in train_features.columns:
            int_lookup = train_features.drop_duplicates(['SKU_ID', 'Region', 'Customer_Name'])[
                ['SKU_ID', 'Region', 'Customer_Name', 'intermittency_class', 'intermittency_code']
            ]
            test = test.merge(int_lookup, on=['SKU_ID', 'Region', 'Customer_Name'], how='left')
            test['intermittency_code'] = test.get('intermittency_code', pd.Series(-1 * len(test))).fillna(3)

        return test
    except Exception as e:
        print(f"  Warning: test feature building failed ({e}); using fallback")
        return None


if __name__ == '__main__':
    sys.exit(main())

