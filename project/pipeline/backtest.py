"""
backtest.py - Rolling-origin backtesting across multiple years.
Evaluates 19-month horizons at origins 2023-10, 2024-10, and 2025-03.
- Target: Units_Sold_Unconstrained.
- NULL Units_Sold / target rows are strictly excluded (never filled with 0).
- Secondary metric: shipped Units_Sold score reported alongside unconstrained target.
- No leakage: features, customer ordering cycles, and intermittency classifications are
  computed strictly using data up to and including each evaluation origin.
- Models (P10, P50, P90) are trained and refit strictly on data up to each origin.
- Recursive 19-month multi-step forecasting: lags in months 2-19 use model predictions.
- Out-of-sample coverage calibration: residual quantiles estimated only
  from origins prior to the one being scored. Raw and calibrated coverage reported side by side.
- Baselines: Seasonal Naive, Moving Average (12m), and Croston / SBA / TSB intermittent methods.
- Saves outputs/backtest_metrics.csv and outputs/backtest_predictions.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from pipeline.config import OUT_BACKTEST
from pipeline.models.lightgbm_model import (
    train_all_quantiles, predict_lgbm, enforce_quantile_ordering,
    compute_wape, compute_bias, get_feature_names
)
from pipeline.features import (
    add_calendar_features, add_event_features,
    add_lifecycle_features, add_macro_features,
    add_lag_rolling_features, classify_intermittency,
    add_sku_region_encodings
)
from pipeline.ordering_cycles import add_cycle_features
from pipeline.models.croston import sba_method, tsb_method

OUT_BACKTEST_PREDS = Path(str(OUT_BACKTEST).replace('backtest_metrics.csv', 'backtest_predictions.csv'))
OUT_COVERAGE_BREAKDOWN = Path(str(OUT_BACKTEST).replace('backtest_metrics.csv', 'coverage_breakdown.csv'))


def get_calibration_offsets(residuals_with_date: list[tuple[str, float]], current_origin: str) -> tuple[float, float]:
    """
    Compute P10 and P90 residual offsets using residuals from dates <= current_origin.
    residuals_with_date is a list of (date_str, residual_float).
    Returns (q10_offset, q90_offset). If no prior residuals, returns (0.0, 0.0).
    """
    valid_resids = [r for d, r in residuals_with_date if str(d)[:7] <= str(current_origin)[:7]]
    if not valid_resids:
        return 0.0, 0.0
    q10_off = float(np.quantile(valid_resids, 0.10))
    q90_off = float(np.quantile(valid_resids, 0.90))
    return q10_off, q90_off


def evaluate_backtest_slice(actual_series: pd.Series, pred_series: pd.Series,
                            origin: str, model_name: str, breakdown: str,
                            group_name: str, horizon_month: int = -1) -> dict:
    """Compute WAPE and Bias metrics for one evaluation slice, strictly excluding NULL actuals."""
    valid = actual_series.notna()
    act = actual_series[valid].astype(float)
    pr = pred_series[valid].astype(float)
    n_excluded = int((~valid).sum())

    wape = compute_wape(act, pr) if len(act) > 0 else np.nan
    bias = compute_bias(act, pr) if len(act) > 0 else np.nan

    return {
        'origin': origin,
        'model': model_name,
        'breakdown': breakdown,
        'group': group_name,
        'horizon_month': horizon_month,
        'WAPE': round(wape, 4) if pd.notna(wape) else np.nan,
        'Bias': round(bias, 2) if pd.notna(bias) else np.nan,
        'n_obs': len(act),
        'n_excluded': n_excluded,
    }


def run_honest_backtest(clean_train: pd.DataFrame,
                        clean_macro: pd.DataFrame,
                        clean_events: pd.DataFrame,
                        sku_master: pd.DataFrame,
                        origins: list = ['2023-10', '2024-10', '2025-03'],
                        horizon: int = 19,
                        save_outputs: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run honest out-of-sample rolling-origin backtests.
    Ensures zero forward data leakage: models, cycle features, intermittency, and lags
    are computed strictly per origin.
    Primary target: Units_Sold_Unconstrained.
    Secondary metric: shipped Units_Sold.
    """
    all_dates = sorted(clean_train['Date'].unique())
    series_keys = ['SKU_ID', 'Region', 'Customer_Name']

    all_metrics = []
    all_preds = []
    residuals_with_date = []

    for origin in origins:
        print(f"\n  Evaluating honest backtest origin: {origin}...")
        origin_idx = all_dates.index(origin) if origin in all_dates else -1
        if origin_idx == -1:
            continue

        train_dates = all_dates[:origin_idx + 1]
        test_dates = all_dates[origin_idx + 1: origin_idx + 1 + horizon]
        if len(test_dates) < horizon:
            print(f"    Skipping origin {origin}: insufficient test dates ({len(test_dates)} < {horizon})")
            continue

        # Isolate historical training data at the origin
        train_hist = clean_train[clean_train['Date'].isin(train_dates)].copy()
        test_actuals = clean_train[clean_train['Date'].isin(test_dates)].copy()

        # Primary target: Units_Sold_Unconstrained (do not fillna with 0, drop null targets)
        if 'Units_Sold_Unconstrained' in train_hist.columns:
            train_hist['target'] = pd.to_numeric(train_hist['Units_Sold_Unconstrained'], errors='coerce')
        else:
            train_hist['target'] = pd.to_numeric(train_hist['Units_Sold'], errors='coerce')

        if 'Units_Sold_Unconstrained' in test_actuals.columns:
            test_actuals['actual'] = pd.to_numeric(test_actuals['Units_Sold_Unconstrained'], errors='coerce')
        else:
            test_actuals['actual'] = pd.to_numeric(test_actuals['Units_Sold'], errors='coerce')

        test_actuals['shipped_actual'] = pd.to_numeric(test_actuals['Units_Sold'], errors='coerce')

        n_null_train = int(train_hist['target'].isna().sum())
        n_null_test = int(test_actuals['actual'].isna().sum())
        print(f"    Excluded {n_null_train} NULL rows from training target, {n_null_test} NULL rows from test evaluation")

        # Recompute intermittency and customer cycles at the origin
        train_hist_valid = train_hist.dropna(subset=['target']).copy()
        train_hist_valid['Units_Sold_Unconstrained'] = train_hist_valid['target']
        train_hist_valid = classify_intermittency(train_hist_valid)
        train_hist_valid = add_cycle_features(train_hist_valid)

        # Build features from historical data
        train_feat = add_calendar_features(train_hist_valid)
        train_feat = add_event_features(train_feat, clean_events)
        train_feat = add_lifecycle_features(train_feat, sku_master)
        train_feat = add_macro_features(train_feat, clean_macro, lags=[0, 1, 2, 3])
        train_feat = add_lag_rolling_features(train_feat)
        train_feat = add_sku_region_encodings(train_feat)

        # Drop any remaining null targets
        train_feat = train_feat.dropna(subset=['target']).copy()

        # Train LightGBM models with an internal validation window
        val_start_dt = pd.to_datetime(origin + '-01') - pd.DateOffset(months=5)
        val_start = val_start_dt.strftime('%Y-%m')
        tr_split = train_feat[train_feat['Date'] < val_start].copy()
        val_split = train_feat[train_feat['Date'] >= val_start].copy()

        models = train_all_quantiles(tr_split, val_split, full_df=train_feat)

        # Forecast recursively across the test window
        cycle_cols = [c for c in ['order_cycle_code', 'is_lost_account', 'intermittency_code'] if c in train_feat.columns]
        cycle_meta = train_feat.drop_duplicates(series_keys)[series_keys + cycle_cols].set_index(series_keys) if cycle_cols else None

        enc_map = {}
        for col in ['SKU_ID', 'Region', 'Customer_Name', 'Product_Family', 'Sales_Channel', 'Business_Unit']:
            enc_col = f"{col}_enc"
            if enc_col in train_feat.columns:
                enc_map[col] = dict(zip(train_feat[col], train_feat[enc_col]))

        # Rolling history buffer strictly up to origin
        hist_buffer = train_hist_valid[['Date', 'SKU_ID', 'Region', 'Customer_Name',
                                        'Units_Sold_Unconstrained', 'Product_Family',
                                        'Sales_Channel', 'Business_Unit']].copy()

        lgb_monthly_preds = []
        for m_idx, current_month in enumerate(test_dates):
            current_test = test_actuals[test_actuals['Date'] == current_month].copy()
            if current_test.empty:
                continue

            current_test = add_calendar_features(current_test)
            current_test = add_event_features(current_test, clean_events)
            current_test = add_lifecycle_features(current_test, sku_master)
            current_test = add_macro_features(current_test, clean_macro, lags=[0, 1, 2, 3])

            for col in ['SKU_ID', 'Region', 'Customer_Name', 'Product_Family', 'Sales_Channel', 'Business_Unit']:
                if col in current_test.columns and col in enc_map:
                    current_test[f"{col}_enc"] = current_test[col].map(enc_map[col]).fillna(-1).astype(int)

            if cycle_meta is not None:
                current_test = current_test.join(cycle_meta, on=series_keys, how='left')
                for c in cycle_cols:
                    current_test[c] = current_test[c].fillna(0)

            # Compute lags from hist_buffer (contains predictions for m_idx > 0, not actuals)
            recent_dict = hist_buffer.sort_values('Date').groupby(series_keys)['Units_Sold_Unconstrained'].apply(list).to_dict()

            keys_tuples = list(zip(current_test['SKU_ID'], current_test['Region'], current_test['Customer_Name']))
            l1_l, l2_l, l3_l, l6_l, l12_l = [], [], [], [], []
            rm3_l, rm6_l, rm12_l = [], [], []
            rs3_l, rs6_l = [], []
            yoy_l = []

            for k in keys_tuples:
                h = recent_dict.get(k, [])
                n = len(h)
                l1 = h[-1] if n >= 1 else 0.0
                l2 = h[-2] if n >= 2 else l1
                l3 = h[-3] if n >= 3 else l2
                l6 = h[-6] if n >= 6 else l3
                l12 = h[-12] if n >= 12 else l6

                l1_l.append(l1)
                l2_l.append(l2)
                l3_l.append(l3)
                l6_l.append(l6)
                l12_l.append(l12)

                w3 = h[-3:] if n >= 3 else (h if n > 0 else [0.0])
                w6 = h[-6:] if n >= 6 else (h if n > 0 else [0.0])
                w12 = h[-12:] if n >= 12 else (h if n > 0 else [0.0])

                rm3_l.append(float(np.mean(w3)))
                rm6_l.append(float(np.mean(w6)))
                rm12_l.append(float(np.mean(w12)))
                rs3_l.append(float(np.std(w3)) if len(w3) > 1 else 0.0)
                rs6_l.append(float(np.std(w6)) if len(w6) > 1 else 0.0)
                yoy_l.append(float((l1 - l12) / (abs(l12) + 1e-3)))

            current_test['lag_1m'] = l1_l
            current_test['lag_2m'] = l2_l
            current_test['lag_3m'] = l3_l
            current_test['lag_6m'] = l6_l
            current_test['lag_12m'] = l12_l
            current_test['lag_12m_yoy'] = yoy_l
            current_test['roll_mean_3m'] = rm3_l
            current_test['roll_mean_6m'] = rm6_l
            current_test['roll_mean_12m'] = rm12_l
            current_test['roll_std_3m'] = rs3_l
            current_test['roll_std_6m'] = rs6_l

            p10 = predict_lgbm(models['p10'], current_test)
            p50 = predict_lgbm(models['p50'], current_test)
            p90 = predict_lgbm(models['p90'], current_test)

            p10_ord, p50_ord, p90_ord = enforce_quantile_ordering(p10, p50, p90)

            current_test['p10'] = np.clip(p10_ord, 0, None).round()
            current_test['p50'] = np.clip(p50_ord, 0, None).round()
            current_test['p90'] = np.clip(p90_ord, 0, None).round()
            current_test['pred'] = current_test['p50']

            lgb_monthly_preds.append(current_test)

            # Append predictions into hist_buffer for next recursive month
            app_df = current_test[series_keys].copy()
            app_df['Date'] = current_month
            app_df['Units_Sold_Unconstrained'] = current_test['p50']
            app_df['Product_Family'] = current_test['Product_Family']
            app_df['Sales_Channel'] = current_test['Sales_Channel']
            app_df['Business_Unit'] = current_test.get('Business_Unit', 'Default')
            hist_buffer = pd.concat([hist_buffer, app_df], ignore_index=True)

        lgb_results = pd.concat(lgb_monthly_preds, ignore_index=True)
        lgb_results['horizon_month'] = lgb_results['Date'].apply(
            lambda d: test_dates.index(d) + 1 if d in test_dates else -1
        )

        # Store raw quantile forecasts
        lgb_results['p10_raw'] = lgb_results['p10'].copy()
        lgb_results['p90_raw'] = lgb_results['p90'].copy()

        val_rows = lgb_results['actual'].notna()
        raw_cov = float(((lgb_results.loc[val_rows, 'actual'] >= lgb_results.loc[val_rows, 'p10_raw']) &
                         (lgb_results.loc[val_rows, 'actual'] <= lgb_results.loc[val_rows, 'p90_raw'])).mean())

        # Calibrate using prior origins' residuals
        q10_off, q90_off = get_calibration_offsets(residuals_with_date, origin)
        if q10_off != 0.0 or q90_off != 0.0:
            cal_p10 = np.clip(lgb_results['p50'] + q10_off, 0, None).round()
            cal_p90 = np.clip(lgb_results['p50'] + q90_off, 0, None).round()
            cal_p10, cal_p50, cal_p90 = enforce_quantile_ordering(cal_p10.values, lgb_results['p50'].values, cal_p90.values)

            lgb_results['p10_cal'] = cal_p10
            lgb_results['p90_cal'] = cal_p90
            cal_cov = float(((lgb_results.loc[val_rows, 'actual'] >= lgb_results.loc[val_rows, 'p10_cal']) &
                             (lgb_results.loc[val_rows, 'actual'] <= lgb_results.loc[val_rows, 'p90_cal'])).mean())
            # Use calibrated intervals for reporting
            lgb_results['p10'] = lgb_results['p10_cal']
            lgb_results['p90'] = lgb_results['p90_cal']
        else:
            lgb_results['p10_cal'] = lgb_results['p10_raw']
            lgb_results['p90_cal'] = lgb_results['p90_raw']
            cal_cov = raw_cov

        # Save current origin residuals for later calibration
        for d, act, pred in zip(lgb_results.loc[val_rows, 'Date'],
                                lgb_results.loc[val_rows, 'actual'],
                                lgb_results.loc[val_rows, 'p50']):
            residuals_with_date.append((str(d)[:7], float(act - pred)))

        # Record LightGBM predictions
        id_cols = ['Date', 'SKU_ID', 'Region', 'Customer_Name']
        extra_save_cols = [c for c in ['Product_Family', 'Region', 'horizon_month'] if c in lgb_results.columns and c not in id_cols]
        lgb_save = lgb_results[id_cols + extra_save_cols + ['actual', 'p10', 'p50', 'p90', 'p10_raw', 'p90_raw', 'p10_cal', 'p90_cal']].copy()
        lgb_save.insert(0, 'model', 'lightgbm_p50')
        lgb_save.insert(0, 'origin', origin)
        all_preds.append(lgb_save)

        # 6. Baselines (evaluated on identical test rows)
        extra_cols = [c for c in ['actual', 'shipped_actual', 'horizon_month', 'Product_Family', 'Sales_Channel'] if c in lgb_results.columns]
        test_copy = lgb_results[id_cols + extra_cols].copy()
        test_copy['month'] = test_copy['Date'].str[5:7].astype(int)

        # Baseline 1: Naive Seasonal
        train_hist_valid['month'] = train_hist_valid['Date'].str[5:7].astype(int)
        ns_lookup = train_hist_valid.groupby(series_keys + ['month'])['Units_Sold_Unconstrained'].mean().to_dict()

        keys_ns = list(zip(test_copy['SKU_ID'], test_copy['Region'], test_copy['Customer_Name'], test_copy['month']))
        test_copy['pred'] = [round(ns_lookup.get(k, 0.0)) for k in keys_ns]

        ns_save = test_copy[id_cols + ['actual', 'pred']].copy()
        ns_save.rename(columns={'pred': 'p50'}, inplace=True)
        ns_save['p10'] = np.nan
        ns_save['p90'] = np.nan
        ns_save['p10_raw'] = np.nan
        ns_save['p90_raw'] = np.nan
        ns_save['p10_cal'] = np.nan
        ns_save['p90_cal'] = np.nan
        ns_save.insert(0, 'model', 'naive_seasonal')
        ns_save.insert(0, 'origin', origin)
        all_preds.append(ns_save)

        # Baseline 2: 12-month Moving Average
        ma_lookup = train_hist_valid.groupby(series_keys)['Units_Sold_Unconstrained'].apply(lambda s: s.tail(12).mean()).to_dict()
        test_copy_ma = test_copy.copy()
        keys_ma = list(zip(test_copy_ma['SKU_ID'], test_copy_ma['Region'], test_copy_ma['Customer_Name']))
        test_copy_ma['pred'] = [round(ma_lookup.get(k, 0.0)) for k in keys_ma]

        ma_save = test_copy_ma[id_cols + ['actual', 'pred']].copy()
        ma_save.rename(columns={'pred': 'p50'}, inplace=True)
        ma_save['p10'] = np.nan
        ma_save['p90'] = np.nan
        ma_save['p10_raw'] = np.nan
        ma_save['p90_raw'] = np.nan
        ma_save['p10_cal'] = np.nan
        ma_save['p90_cal'] = np.nan
        ma_save.insert(0, 'model', 'moving_average')
        ma_save.insert(0, 'origin', origin)
        all_preds.append(ma_save)

        # Baseline 3: Intermittent methods (Croston, SBA, TSB)
        interm_lookup = {}
        for k, grp in train_hist_valid.groupby(series_keys):
            cls = grp['intermittency_class'].iloc[-1] if 'intermittency_class' in grp.columns else 'smooth'
            series = grp.sort_values('Date')['Units_Sold_Unconstrained']
            if cls == 'lumpy':
                f = tsb_method(series)
            elif cls == 'intermittent':
                f = sba_method(series)
            else:
                f = series.mean()
            interm_lookup[k] = round(max(0, f))

        test_copy_int = test_copy.copy()
        keys_int = list(zip(test_copy_int['SKU_ID'], test_copy_int['Region'], test_copy_int['Customer_Name']))
        test_copy_int['pred'] = [interm_lookup.get(k, 0) for k in keys_int]

        int_save = test_copy_int[id_cols + ['actual', 'pred']].copy()
        int_save.rename(columns={'pred': 'p50'}, inplace=True)
        int_save['p10'] = np.nan
        int_save['p90'] = np.nan
        int_save['p10_raw'] = np.nan
        int_save['p90_raw'] = np.nan
        int_save['p10_cal'] = np.nan
        int_save['p90_cal'] = np.nan
        int_save.insert(0, 'model', 'croston_sba_tsb')
        int_save.insert(0, 'origin', origin)
        all_preds.append(int_save)

        # 7. Slices evaluation for metrics
        models_eval = [
            ('lightgbm_p50', lgb_results),
            ('naive_seasonal', test_copy),
            ('moving_average', test_copy_ma),
            ('croston_sba_tsb', test_copy_int)
        ]

        # Overall and slice metrics (Primary target: Units_Sold_Unconstrained)
        for m_name, df_eval in models_eval:
            slice_row = evaluate_backtest_slice(df_eval['actual'], df_eval['pred'], origin, m_name, 'overall', 'ALL')
            if m_name == 'lightgbm_p50':
                slice_row['coverage_raw'] = round(raw_cov, 4)
                slice_row['coverage_calibrated'] = round(cal_cov, 4)
            all_metrics.append(slice_row)

            # Evaluate against shipped actuals
            if 'shipped_actual' in df_eval.columns:
                all_metrics.append(evaluate_backtest_slice(
                    df_eval['shipped_actual'], df_eval['pred'], origin, f"{m_name}_vs_shipped", 'secondary_shipped', 'ALL'
                ))

            for h, grp in df_eval.groupby('horizon_month'):
                all_metrics.append(evaluate_backtest_slice(
                    grp['actual'], grp['pred'], origin, m_name, 'horizon', f"month_{h}", horizon_month=int(h)
                ))

            if 'Product_Family' in df_eval.columns:
                for fam, grp in df_eval.groupby('Product_Family'):
                    all_metrics.append(evaluate_backtest_slice(grp['actual'], grp['pred'], origin, m_name, 'family', str(fam)))

            if 'Region' in df_eval.columns:
                for reg, grp in df_eval.groupby('Region'):
                    all_metrics.append(evaluate_backtest_slice(grp['actual'], grp['pred'], origin, m_name, 'region', str(reg)))

            if 'Sales_Channel' in df_eval.columns:
                for ch, grp in df_eval.groupby('Sales_Channel'):
                    all_metrics.append(evaluate_backtest_slice(grp['actual'], grp['pred'], origin, m_name, 'channel', str(ch)))

    metrics_df = pd.DataFrame(all_metrics)
    preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    if save_outputs:
        metrics_df.to_csv(OUT_BACKTEST, index=False)
        print(f"\n  Backtest metrics saved to {OUT_BACKTEST} ({len(metrics_df)} slices)")
        preds_df.to_csv(OUT_BACKTEST_PREDS, index=False)
        print(f"  Backtest predictions saved to {OUT_BACKTEST_PREDS} ({len(preds_df)} rows)")

    if preds_df.empty or not save_outputs:
        return metrics_df, preds_df

    # Generate and export coverage breakdown
    lgb_preds = preds_df[preds_df['model'] == 'lightgbm_p50'].copy()
    cov_rows = []
    val_overall = lgb_preds['actual'].notna()
    c_raw = float(((lgb_preds.loc[val_overall, 'actual'] >= lgb_preds.loc[val_overall, 'p10_raw']) &
                   (lgb_preds.loc[val_overall, 'actual'] <= lgb_preds.loc[val_overall, 'p90_raw'])).mean())
    c_cal = float(((lgb_preds.loc[val_overall, 'actual'] >= lgb_preds.loc[val_overall, 'p10_cal']) &
                   (lgb_preds.loc[val_overall, 'actual'] <= lgb_preds.loc[val_overall, 'p90_cal'])).mean())
    cov_rows.append({
        'origin': 'ALL', 'breakdown': 'overall', 'group': 'ALL', 'horizon_month': 'all',
        'n_obs': int(val_overall.sum()), 'coverage_raw': round(c_raw, 4), 'coverage_calibrated': round(c_cal, 4)
    })
    for orig, grp in lgb_preds.groupby('origin'):
        v = grp['actual'].notna()
        c_raw = float(((grp.loc[v, 'actual'] >= grp.loc[v, 'p10_raw']) & (grp.loc[v, 'actual'] <= grp.loc[v, 'p90_raw'])).mean())
        c_cal = float(((grp.loc[v, 'actual'] >= grp.loc[v, 'p10_cal']) & (grp.loc[v, 'actual'] <= grp.loc[v, 'p90_cal'])).mean())
        cov_rows.append({
            'origin': str(orig), 'breakdown': 'origin', 'group': str(orig), 'horizon_month': -1,
            'n_obs': int(v.sum()), 'coverage_raw': round(c_raw, 4), 'coverage_calibrated': round(c_cal, 4)
        })
    if 'horizon_month' in lgb_preds.columns:
        for h, grp in lgb_preds.groupby('horizon_month'):
            v = grp['actual'].notna()
            c_raw = float(((grp.loc[v, 'actual'] >= grp.loc[v, 'p10_raw']) & (grp.loc[v, 'actual'] <= grp.loc[v, 'p90_raw'])).mean())
            c_cal = float(((grp.loc[v, 'actual'] >= grp.loc[v, 'p10_cal']) & (grp.loc[v, 'actual'] <= grp.loc[v, 'p90_cal'])).mean())
            cov_rows.append({
                'origin': 'ALL', 'breakdown': 'horizon', 'group': f"month_{h}", 'horizon_month': int(h),
                'n_obs': int(v.sum()), 'coverage_raw': round(c_raw, 4), 'coverage_calibrated': round(c_cal, 4)
            })
    if 'Product_Family' in lgb_preds.columns:
        for fam, grp in lgb_preds.groupby('Product_Family'):
            v = grp['actual'].notna()
            c_raw = float(((grp.loc[v, 'actual'] >= grp.loc[v, 'p10_raw']) & (grp.loc[v, 'actual'] <= grp.loc[v, 'p90_raw'])).mean())
            c_cal = float(((grp.loc[v, 'actual'] >= grp.loc[v, 'p10_cal']) & (grp.loc[v, 'actual'] <= grp.loc[v, 'p90_cal'])).mean())
            cov_rows.append({
                'origin': 'ALL', 'breakdown': 'family', 'group': str(fam), 'horizon_month': -1,
                'n_obs': int(v.sum()), 'coverage_raw': round(c_raw, 4), 'coverage_calibrated': round(c_cal, 4)
            })
    cov_df = pd.DataFrame(cov_rows)
    cov_df.to_csv(OUT_COVERAGE_BREAKDOWN, index=False)
    print(f"  Coverage breakdown saved to {OUT_COVERAGE_BREAKDOWN} ({len(cov_df)} breakdown rows)")

    return metrics_df, preds_df
