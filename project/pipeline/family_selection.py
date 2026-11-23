"""
Per-family model selection.

Each product family is forecast by the model with the lowest backtest WAPE over the
two selection origins (2023-10 and 2024-10). The 2025-03 origin is not used for the
choice and is reported as the holdout result. Ties go to the simpler model.
"""

import numpy as np
import pandas as pd

from pipeline.config import OUT_FAMILY_SELECTION
from pipeline.features import classify_intermittency
from pipeline.models.baselines import naive_seasonal_forecast, moving_average_forecast
from pipeline.models.croston import sba_method, tsb_method
from pipeline.forecast import apply_lifecycle_rules

SELECTION_ORIGINS = ['2023-10', '2024-10']
HOLDOUT_ORIGIN = '2025-03'
# Simpler models first: this order breaks ties.
CANDIDATES = ['naive_seasonal', 'moving_average', 'croston_sba_tsb', 'lightgbm_p50']
MODEL_COLUMN = {
    'naive_seasonal': 'Units_Sold_Forecast_naive',
    'moving_average': 'Units_Sold_Forecast_moving',
    'croston_sba_tsb': 'Units_Sold_Forecast_croston',
    'lightgbm_p50': 'Units_Sold_Forecast_p50',
}
SERIES_KEYS = ['SKU_ID', 'Region', 'Customer_Name']


def _wape(actual: pd.Series, pred: pd.Series) -> float:
    total = actual.sum()
    return float(np.nan) if total == 0 else float((pred - actual).abs().sum() / total)


def family_wape_table(backtest_preds: pd.DataFrame) -> pd.DataFrame:
    """WAPE by origin, family and model, over rows with a non-null actual."""
    d = backtest_preds.dropna(subset=['actual']).copy()
    rows = []
    for (origin, fam, model), g in d.groupby(['origin', 'Product_Family', 'model']):
        rows.append({'origin': origin, 'Product_Family': fam, 'model': model,
                     'WAPE': _wape(g['actual'], g['p50']), 'n_obs': len(g)})
    return pd.DataFrame(rows)


def select_family_models(backtest_preds: pd.DataFrame, out_path=OUT_FAMILY_SELECTION) -> pd.DataFrame:
    """Choose one model per family and record the selection evidence."""
    table = family_wape_table(backtest_preds)
    families = sorted(table['Product_Family'].unique())
    rows = []
    for fam in families:
        sel = table[(table['Product_Family'] == fam) & table['origin'].isin(SELECTION_ORIGINS)]
        means = {}
        for m in CANDIDATES:
            vals = sel[sel['model'] == m]['WAPE']
            means[m] = float(vals.mean()) if len(vals) == len(SELECTION_ORIGINS) else np.inf
        chosen = min(CANDIDATES, key=lambda m: (means[m], CANDIDATES.index(m)))
        hold = table[(table['Product_Family'] == fam) & (table['origin'] == HOLDOUT_ORIGIN)]
        holdout = {m: float(hold[hold['model'] == m]['WAPE'].iloc[0]) for m in CANDIDATES
                   if (hold['model'] == m).any()}
        rows.append({
            'Product_Family': fam,
            'Selected_Model': chosen,
            'Selection_WAPE_naive_seasonal': means['naive_seasonal'],
            'Selection_WAPE_moving_average': means['moving_average'],
            'Selection_WAPE_croston_sba_tsb': means['croston_sba_tsb'],
            'Selection_WAPE_lightgbm_p50': means['lightgbm_p50'],
            'Holdout_WAPE_selected': holdout.get(chosen, np.nan),
            'Holdout_WAPE_naive_seasonal': holdout.get('naive_seasonal', np.nan),
            'Holdout_WAPE_moving_average': holdout.get('moving_average', np.nan),
            'Holdout_WAPE_croston_sba_tsb': holdout.get('croston_sba_tsb', np.nan),
            'Holdout_WAPE_lightgbm_p50': holdout.get('lightgbm_p50', np.nan),
            'Holdout_origin': HOLDOUT_ORIGIN,
        })
    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False)
    return result


def compute_baseline_forecasts(history: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline forecasts for the test rows, built the same way the backtest builds them:
    the target is Units_Sold_Unconstrained, and the intermittent methods use the
    intermittency class of each series.
    """
    hist = history.copy()
    hist['Units_Sold_Unconstrained'] = pd.to_numeric(hist['Units_Sold_Unconstrained'], errors='coerce')
    hist = hist.dropna(subset=['Units_Sold_Unconstrained'])
    hist = classify_intermittency(hist)

    naive = naive_seasonal_forecast(
        hist.assign(Units_Sold=hist['Units_Sold_Unconstrained']), test_df[['Date'] + SERIES_KEYS]
    ).rename(columns={'Units_Sold_Forecast': MODEL_COLUMN['naive_seasonal']})
    moving = moving_average_forecast(
        hist.assign(Units_Sold=hist['Units_Sold_Unconstrained']), test_df[['Date'] + SERIES_KEYS]
    ).rename(columns={'Units_Sold_Forecast': MODEL_COLUMN['moving_average']})

    interm = {}
    for key, grp in hist.sort_values('Date').groupby(SERIES_KEYS):
        cls = grp['intermittency_class'].iloc[-1]
        series = grp['Units_Sold_Unconstrained']
        if cls == 'lumpy':
            f = tsb_method(series)
        elif cls == 'intermittent':
            f = sba_method(series)
        else:
            f = series.mean()
        interm[key] = round(max(0.0, float(f)))
    out = test_df[['Date'] + SERIES_KEYS].copy()
    out[MODEL_COLUMN['croston_sba_tsb']] = [
        interm.get(k, 0) for k in zip(out['SKU_ID'], out['Region'], out['Customer_Name'])
    ]
    out = out.merge(naive, on=['Date'] + SERIES_KEYS, how='left')
    out = out.merge(moving, on=['Date'] + SERIES_KEYS, how='left')
    for col in MODEL_COLUMN.values():
        if col in out.columns:
            out[col] = out[col].fillna(0).clip(lower=0).round().astype(int)
    return out


def assemble_family_forecast(lgb_submission: pd.DataFrame, lgb_intervals: pd.DataFrame,
                             baselines: pd.DataFrame, selection: pd.DataFrame,
                             test_df: pd.DataFrame, sku_master: pd.DataFrame):
    """
    Build the final submission and intervals.
    Each row takes its family's selected model. Only LightGBM rows carry P10 and P90.
    Lifecycle zeroing is applied to the final values.
    """
    keys = ['Date', 'SKU_ID', 'Region']
    fam_map = dict(zip(selection['Product_Family'], selection['Selected_Model']))
    sku_fam = dict(zip(sku_master['SKU_ID'], sku_master['Product_Family']))

    base = test_df[keys + ['Customer_Name']].copy()
    base['Product_Family'] = base['SKU_ID'].map(sku_fam).fillna('Unknown')
    base = base.merge(baselines, on=['Date', 'SKU_ID', 'Region', 'Customer_Name'], how='left')
    base = base.merge(lgb_submission[keys + ['Units_Sold_Forecast']], on=keys, how='left')
    base = base.merge(lgb_intervals[keys + ['Units_Sold_Forecast_p10', 'Units_Sold_Forecast_p50',
                                             'Units_Sold_Forecast_p90']], on=keys, how='left')
    base = base.rename(columns={'Units_Sold_Forecast': MODEL_COLUMN['lightgbm_p50'] + '_sub'})

    chosen = base['Product_Family'].map(fam_map).fillna('lightgbm_p50')
    point = np.select(
        [chosen == 'naive_seasonal', chosen == 'moving_average', chosen == 'croston_sba_tsb'],
        [base[MODEL_COLUMN['naive_seasonal']], base[MODEL_COLUMN['moving_average']],
         base[MODEL_COLUMN['croston_sba_tsb']]],
        default=base['Units_Sold_Forecast_p50'],
    ).astype(float)
    is_lgb = (chosen == 'lightgbm_p50')

    final = base[keys + ['Product_Family']].copy()
    final['Selected_Model'] = chosen.values
    final['Units_Sold_Forecast_p50'] = point
    final['Units_Sold_Forecast_p10'] = np.where(is_lgb, base['Units_Sold_Forecast_p10'], np.nan)
    final['Units_Sold_Forecast_p90'] = np.where(is_lgb, base['Units_Sold_Forecast_p90'], np.nan)
    final['Units_Sold_Forecast'] = point

    final = apply_lifecycle_rules(final, sku_master)
    for col in ['Units_Sold_Forecast', 'Units_Sold_Forecast_p50']:
        final[col] = final[col].fillna(0).clip(lower=0).round().astype(int)
    submission = final[keys + ['Units_Sold_Forecast']].copy()
    intervals = final[keys + ['Units_Sold_Forecast_p10', 'Units_Sold_Forecast_p50',
                              'Units_Sold_Forecast_p90']].copy()
    return submission, intervals, final[keys + ['Product_Family', 'Selected_Model']].copy()
